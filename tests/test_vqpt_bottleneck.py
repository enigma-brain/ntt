"""Tests for VQPTBottleneck — the wrapper around lucidrains/vector-quantize-pytorch.

Covers:
1. Shape contract: encode() preserves (B, z_dim, T, N) and get_indices() flattens (T, N).
2. Index round-trip: indices_to_codes(get_indices(h), T) yields the same shape,
   and matches encode() output for FSQ-family variants where indices are lossless.
3. Gradient flow through encode().
4. End-to-end integration via WanNeuralTokenizer.forward() with bottleneck_type="vqpt".
"""

from __future__ import annotations

import pytest
import torch

from ntt.contrib.tokenizer.wanneural.bottleneck import VQPTBottleneck
from ntt.contrib.tokenizer.wanneural.wanneural import (
    WanNeuralTokenizer,
    WanNeuralTokenizerSpec,
)
from tests.fixtures import make_dummy_binned_input

# Cheap configs that exercise distinct codepaths in the wrapper:
# - vq: classic VectorQuantize, single 2D index stream
# - fsq: deterministic, no aux loss
# - lfq: spherical=False, entropy aux loss
# - residual_fsq: 3D index stream, no aux loss
# - residual_vq: 3D index stream, commit-loss aggregation across layers
# - sim_vq: simplified VQ with rotation trick
# - latent_quantize: channel-first input layout
_CASES = [
    ("vq", {"codebook_size": 64}),
    ("fsq", {"levels": [8, 5, 5, 5]}),
    ("lfq", {"codebook_size": 64}),
    ("residual_fsq", {"levels": [8, 5, 5, 5], "num_quantizers": 2}),
    ("residual_vq", {"codebook_size": 64, "num_quantizers": 3}),
    ("sim_vq", {"codebook_size": 64}),
    ("latent_quantize", {"levels": [5, 5, 5, 5]}),
]


@pytest.mark.parametrize("vq_type,kwargs", _CASES)
def test_encode_preserves_shape(vq_type, kwargs):
    torch.manual_seed(0)
    B, C, T, N = 2, 16, 4, 5
    bn = VQPTBottleneck(z_dim=C, vq_type=vq_type, vq_kwargs=kwargs).train()
    h = torch.randn(B, C, T, N)

    z, aux = bn.encode(h)
    assert z.shape == (B, C, T, N), f"{vq_type}: encode shape {z.shape}"
    if aux is not None:
        assert aux.dim() == 0, f"{vq_type}: aux loss must be scalar, got {aux.shape}"
        assert torch.isfinite(aux), f"{vq_type}: non-finite aux loss"


@pytest.mark.parametrize("vq_type,kwargs", _CASES)
def test_indices_roundtrip(vq_type, kwargs):
    """get_indices → indices_to_codes recovers the (B, z_dim, T, N) layout."""
    torch.manual_seed(0)
    B, C, T, N = 2, 16, 4, 5
    bn = VQPTBottleneck(z_dim=C, vq_type=vq_type, vq_kwargs=kwargs).eval()
    h = torch.randn(B, C, T, N)

    idx = bn.get_indices(h)
    # Sequence-flat layout: (B, T*N, ...) for all variants.
    assert idx.shape[0] == B and idx.shape[1] == T * N

    codes = bn.indices_to_codes(idx, T=T)
    assert codes.shape == (B, C, T, N)


@pytest.mark.parametrize(
    "vq_type,kwargs",
    [
        ("fsq", {"levels": [8, 5, 5, 5]}),
        ("residual_fsq", {"levels": [8, 5, 5, 5], "num_quantizers": 2}),
    ],
)
def test_fsq_indices_match_encode(vq_type, kwargs):
    """For FSQ-family quantizers indices are a lossless code → indices_to_codes
    must reproduce encode() exactly (up to the wrapped class's project_out)."""
    torch.manual_seed(0)
    B, C, T, N = 2, 16, 4, 5
    bn = VQPTBottleneck(z_dim=C, vq_type=vq_type, vq_kwargs=kwargs).eval()
    h = torch.randn(B, C, T, N)

    z_enc, _ = bn.encode(h)
    idx = bn.get_indices(h)
    z_rec = bn.indices_to_codes(idx, T=T)
    torch.testing.assert_close(z_enc, z_rec, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "vq_type,kwargs",
    [
        ("vq", {"codebook_size": 64}),
        ("fsq", {"levels": [8, 5, 5, 5]}),
        ("lfq", {"codebook_size": 64}),
        ("residual_vq", {"codebook_size": 64, "num_quantizers": 2}),
    ],
)
def test_grad_flows_through_encode(vq_type, kwargs):
    torch.manual_seed(0)
    B, C, T, N = 2, 16, 4, 3
    bn = VQPTBottleneck(z_dim=C, vq_type=vq_type, vq_kwargs=kwargs).train()
    h = torch.randn(B, C, T, N, requires_grad=True)

    z, aux = bn.encode(h)
    loss = z.pow(2).mean()
    if aux is not None:
        loss = loss + aux
    loss.backward()

    assert h.grad is not None
    assert torch.isfinite(h.grad).all()
    assert h.grad.abs().mean() > 0, f"{vq_type}: zero gradient through bottleneck"


def _make_spec(**overrides) -> WanNeuralTokenizerSpec:
    defaults = {
        "n_neurons": 8,
        "window_length_seconds": 0.2,
        "bin_size_ms": 1.0,
        "dim": 16,
        "z_dim": 8,
        "num_levels": 3,
        "dim_mult": (1, 2, 4),
        "num_res_blocks": 1,
        "attn_levels": (),
        "num_attn_heads": 1,
        "temporal_patch_size": 1,
        "temporal_downsample": (True, True),
        "token_bin_size_ms": 4.0,
    }
    defaults.update(overrides)
    return WanNeuralTokenizerSpec(**defaults)


@pytest.mark.parametrize(
    "vq_type,kwargs,has_aux",
    [
        ("vq", {"codebook_size": 32}, True),  # commitment loss
        ("fsq", {"levels": [8, 5, 5, 5]}, False),  # no aux loss
        ("residual_vq", {"codebook_size": 32, "num_quantizers": 2}, True),
    ],
)
def test_wanneural_forward_with_vqpt(vq_type, kwargs, has_aux):
    """Full WanNeuralTokenizer.forward() works with bottleneck_type='vqpt'."""
    spec = _make_spec(
        bottleneck_type="vqpt",
        vqpt_type=vq_type,
        vqpt_kwargs=kwargs,
    )
    tok = WanNeuralTokenizer(spec).train()
    x = make_dummy_binned_input(n_neurons=8, time_bins=200, batch_size=2)

    out = tok(x)

    assert out.reconstruction.shape == (2, 200, 8)
    assert out.latents is not None
    assert out.latents.shape == (2, spec.num_latent_time_steps, 8, spec.z_dim)
    assert out.tokens is not None
    if has_aux:
        assert out.aux_loss is not None and "vqpt" in out.aux_loss
        assert torch.isfinite(out.aux_loss["vqpt"])
    else:
        # FSQ-family quantizers have no auxiliary loss; the wanneural forward
        # leaves aux_loss as None when bottleneck_loss is None.
        assert out.aux_loss is None


def test_wanneural_reconstruct_from_2d_tokens():
    """For non-residual vqpt variants tokens are 2D and reconstruct() routes
    through indices_to_codes."""
    spec = _make_spec(
        bottleneck_type="vqpt",
        vqpt_type="fsq",
        vqpt_kwargs={"levels": [8, 5, 5, 5]},
    )
    tok = WanNeuralTokenizer(spec).eval()
    x = make_dummy_binned_input(n_neurons=8, time_bins=200, batch_size=2)

    out = tok(x)
    assert out.tokens is not None and out.tokens.dim() == 2

    recon = tok.reconstruct(out.tokens)
    assert recon.reconstruction.shape == (2, 200, 8)


def test_dim_kwarg_rejected():
    """User-supplied `dim` collides with the auto-supplied one."""
    with pytest.raises(ValueError, match="dim"):
        VQPTBottleneck(z_dim=16, vq_type="vq", vq_kwargs={"dim": 32, "codebook_size": 64})


def test_unknown_vq_type_raises():
    with pytest.raises(ValueError, match="Unknown vq_type"):
        VQPTBottleneck(z_dim=16, vq_type="not_a_real_quantizer")
