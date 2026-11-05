"""Tests for the Cosmos 2D tokenizer (causal-in-time wavelet + 2D conv encoder/decoder).

Covers:
- Patcher2D <-> UnPatcher2D bit-exact roundtrip across wavelet families.
- Encoder2D causality along the time axis (no future leak past the latent step).
- End-to-end forward() for both continuous (AE) and discrete (FSQ) bottlenecks.
- LitModule training_step / backward.
"""

from __future__ import annotations

from typing import cast

import pytest
import torch

from ntt.contrib.loss import PoissonNLLLossSpec
from ntt.contrib.tokenizer.cosmos import (
    CosmosTokenizer2D,
    CosmosTokenizer2DSpec,
)
from ntt.contrib.tokenizer.cosmos.layers2d import Encoder2D
from ntt.contrib.tokenizer.cosmos.patcher import Patcher2D, UnPatcher2D
from ntt.core.components.types import BinnedSpikesInput, LatentType
from ntt.core.model.litmodule import NTTLitModule


def _make_batch(spec: CosmosTokenizer2DSpec, batch_size: int = 2) -> BinnedSpikesInput:
    n_bins = spec.window_length_bins
    n_units = spec.n_neurons
    spikes = torch.poisson(torch.full((batch_size, n_bins, n_units), 0.1)).float()
    return BinnedSpikesInput(
        spike_matrix=spikes,
        pad_and_gap_mask=torch.ones_like(spikes, dtype=torch.int64),
        sampling_rate=1000.0,
        unit_ids=torch.arange(n_units).unsqueeze(0).expand(batch_size, -1),
        unit_ids_pad_mask=torch.ones(batch_size, n_units, dtype=torch.int64),
        duration_sec=torch.full((batch_size,), float(spec.window_length_seconds)),
        session_id=("s",) * batch_size,
        subject_id=("sub",) * batch_size,
        species=("test",) * batch_size,
        absolute_start=torch.zeros(batch_size, dtype=torch.float64),
        absolute_end=torch.full(
            (batch_size,), float(spec.window_length_seconds), dtype=torch.float64
        ),
    )


def _make_spec_2d(bottleneck_type: str = "ae", **overrides: object) -> CosmosTokenizer2DSpec:
    kwargs: dict[str, object] = {
        "n_neurons": 32,
        "window_length_seconds": 1.0,
        "bin_size_ms": 1.0,
        "bottleneck_type": bottleneck_type,
        "channels": 16,
        "channels_mult": [1, 2, 2],
        "num_res_blocks": 1,
        "z_channels": 16,
        "neuron_compression": 16,
        "temporal_compression": 16,
        "patch_h": 4,
        "patch_w": 4,
        "latent_channels": 8,
        "fsq_levels": [8, 5, 5, 5],
    }
    kwargs.update(overrides)
    return CosmosTokenizer2DSpec(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "method,H,W,patch",
    [
        # Every wavelet round-trips bit-exactly under circular padding. Haar
        # accepts any axis length; longer wavelets need each axis / patch_size
        # >= (L - 2) / 2, so we avoid H=32 + patch=16 + L=8 (would need >= 48).
        ("haar", 16, 32, 4),
        ("haar", 32, 1000, 16),
        ("haar", 13, 1000, 4),
        ("db2", 16, 32, 4),
        ("db2", 32, 1000, 16),
        ("db4", 32, 128, 4),
        ("db4", 48, 1000, 16),
        ("sym4", 32, 128, 4),
        ("sym4", 48, 1000, 16),
    ],
)
def test_patcher2d_roundtrip(method: str, H: int, W: int, patch: int) -> None:
    torch.manual_seed(0)
    p = Patcher2D(patch_h=patch, patch_w=patch, patch_method=method)
    u = UnPatcher2D(patch_h=patch, patch_w=patch, patch_method=method)
    x = torch.randn(2, 1, H, W)
    y = p(x)
    assert y.shape[1] == patch * patch
    xr = u(y, trim_h=H, trim_w=W)
    assert xr.shape == x.shape
    assert torch.allclose(xr, x, atol=1e-5)


def test_encoder2d_is_causal_along_time() -> None:
    torch.manual_seed(0)
    enc = Encoder2D(
        in_channels=1,
        channels=16,
        channels_mult=[1, 2, 2],
        num_res_blocks=1,
        dropout=0.0,
        z_channels=8,
        neuron_compression=16,
        temporal_compression=16,
        patch_h=4,
        patch_w=4,
        patch_method="haar",
    ).eval()
    x = torch.randn(1, 1, 32, 1024)
    t_pert = 800
    with torch.no_grad():
        z = enc(x)
        x2 = x.clone()
        x2[..., t_pert:] += torch.randn_like(x2[..., t_pert:]) * 5.0
        z2 = enc(x2)
    # Sum diff over B, C, H — we only want causality along W.
    diff = (z2 - z).abs().sum(dim=(0, 1, 2))
    first_aff = next((j for j in range(z.shape[-1]) if diff[j].item() > 1e-5), None)
    assert first_aff is not None
    assert first_aff * 16 >= t_pert - 16, (
        f"Causality leak along W: latent {first_aff} affected by t={t_pert}"
    )


@pytest.mark.parametrize("bottleneck_type", ["ae", "fsq"])
def test_cosmos2d_forward_shapes(bottleneck_type: str) -> None:
    spec = _make_spec_2d(bottleneck_type=bottleneck_type)
    tok = CosmosTokenizer2D(spec)
    x = _make_batch(spec, batch_size=2)
    out = tok(x)

    B, T, N = x.spike_matrix.shape
    T_lat = spec.num_latent_time_steps
    U_lat = spec.num_latent_units

    assert out.reconstruction.shape == (B, T, N)
    assert out.reconstruction.ge(0).all()
    assert out.latents is not None
    assert out.latents.shape[0] == B
    assert out.latents.shape[1] == T_lat
    assert out.latents.shape[2] == U_lat

    if bottleneck_type == "fsq":
        assert spec.latent_type == LatentType.DISCRETE
        assert out.tokens is not None
        # Tokens preserve spatial layout: [B, T_lat, U_lat] (matches output.latents
        # without the channel dim). Forward() avoids flattening so reconstruct()
        # can reshape correctly under longer wavelets where actual T'/U' diverge
        # from the spec's Haar-approximate num_latent_* helpers.
        assert out.tokens.shape == (B, T_lat, U_lat)
        assert int(out.tokens.max().item()) < spec.get_vocab_size()
    else:
        assert spec.latent_type == LatentType.CONTINUOUS
        assert out.tokens is None


@pytest.mark.parametrize("bottleneck_type", ["ae", "fsq"])
def test_cosmos2d_encode_decode_roundtrip(bottleneck_type: str) -> None:
    spec = _make_spec_2d(bottleneck_type=bottleneck_type)
    tok = CosmosTokenizer2D(spec)
    x = _make_batch(spec)
    z = tok.encode(x)
    assert isinstance(z, torch.Tensor)
    z = cast(torch.Tensor, z)
    assert z.ndim == 4
    recon = tok.decode(z)
    assert recon.shape == x.spike_matrix.shape
    assert recon.ge(0).all()


@pytest.mark.parametrize("bottleneck_type", ["ae", "fsq"])
def test_cosmos2d_litmodule_training_step(bottleneck_type: str) -> None:
    spec = _make_spec_2d(bottleneck_type=bottleneck_type)
    loss_spec = PoissonNLLLossSpec(reduction="mean", log_input=False, full=False)
    lit = NTTLitModule(
        tokenizer_spec=spec,
        loss_spec=loss_spec,
        optimizer_gen=lambda p: torch.optim.Adam(p, lr=1e-3),
    )
    lit.configure_model()
    opt_result = lit.configure_optimizers()
    raw = opt_result["optimizer"] if isinstance(opt_result, dict) else opt_result
    assert isinstance(raw, torch.optim.Optimizer)
    optimizer: torch.optim.Optimizer = raw

    lit.train()
    for step in range(2):
        batch = _make_batch(spec)
        optimizer.zero_grad()
        loss = lit.training_step(batch, step)
        assert torch.isfinite(loss), f"non-finite loss at step {step}"
        loss.backward()
        optimizer.step()
