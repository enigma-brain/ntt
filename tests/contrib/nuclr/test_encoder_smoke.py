"""Permanent smoke tests for the NuCLR encoders.

Cheap guard for forward-shape contracts and gradient flow.
"""

from __future__ import annotations

import pytest
import torch

from ntt.contrib.nuclr.encoder import NuclrEncoder
from ntt.contrib.nuclr.encoder_binned import NuclrBinnedEncoder

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="flex_attention requires CUDA",
)


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda")


def test_spike_encoder_shapes(device: torch.device) -> None:
    """NuclrEncoder.forward produces (total_units, dim) for heterogeneous batches."""
    torch.manual_seed(0)
    encoder = (
        NuclrEncoder(
            ctx_duration=1.0,
            latent_step=0.25,
            dim=64,
            cross_heads=4,
            self_heads=4,
            dim_head=16,
            t_layers=1,
            st_layers=1,
            rot_ratio=0.5,
        )
        .to(device)
        .eval()
    )

    # batch=2 samples, units=[3, 4], spikes per unit in {5..12}.
    unit_seqlen = torch.tensor([3, 4], device=device)
    spike_seqlen = torch.tensor([5, 9, 7, 6, 12, 8, 11], device=device)
    spikes = torch.rand(int(spike_seqlen.sum()), device=device)

    out = encoder(spikes=spikes, spike_seqlen=spike_seqlen, unit_seqlen=unit_seqlen)
    assert out.shape == (int(unit_seqlen.sum()), 64), out.shape


def test_spike_encoder_handles_silent_unit(device: torch.device) -> None:
    """A unit silent in the window (zero spikes) must not crash or NaN.

    Zero-length kv segments are valid: ``ntt_to_nuclr_format`` emits ``spike_seqlen == 0``
    for units with no spikes in a window. The spike encoder must produce finite
    embeddings (forward and backward) for such units rather than raising.
    """
    torch.manual_seed(0)
    encoder = (
        NuclrEncoder(
            ctx_duration=1.0,
            latent_step=0.25,
            dim=64,
            cross_heads=4,
            self_heads=4,
            dim_head=16,
            t_layers=1,
            st_layers=1,
            rot_ratio=0.5,
        )
        .to(device)
        .train()
    )

    # batch=2, units=[3, 4]; the 2nd unit (index 1) is SILENT — 0 spikes.
    unit_seqlen = torch.tensor([3, 4], device=device)
    spike_seqlen = torch.tensor([5, 0, 7, 6, 12, 8, 11], device=device)
    spikes = torch.rand(int(spike_seqlen.sum()), device=device)

    out = encoder(spikes=spikes, spike_seqlen=spike_seqlen, unit_seqlen=unit_seqlen)
    assert out.shape == (int(unit_seqlen.sum()), 64), out.shape
    assert torch.isfinite(out).all(), "encoder produced non-finite embeddings"

    out.sum().backward()
    grads = [p.grad for p in encoder.parameters() if p.grad is not None]
    assert grads, "no gradients produced"
    assert all(torch.isfinite(g).all() for g in grads), "non-finite gradient"


def test_binned_encoder_shapes_heterogeneous(device: torch.device) -> None:
    """NuclrBinnedEncoder.forward produces (total_units, dim)."""
    torch.manual_seed(1)
    encoder = (
        NuclrBinnedEncoder(
            ctx_duration=1.0,
            latent_step=0.25,
            bin_size=0.05,
            dim=64,
            self_heads=4,
            dim_head=16,
            t_layers=1,
            st_layers=1,
            rot_ratio=0.5,
        )
        .to(device)
        .eval()
    )

    unit_seqlen = torch.tensor([3, 4], device=device)
    total_units = int(unit_seqlen.sum())
    num_latents = encoder.num_latents
    bins_per_latent = encoder.bins_per_latent

    bins = torch.randn(
        total_units * num_latents,
        bins_per_latent,
        device=device,
    )
    out = encoder(bins, unit_seqlen)
    assert out.shape == (total_units, 64), out.shape


def test_binned_encoder_grad_flows(device: torch.device) -> None:
    """Every attention-module parameter receives a non-zero gradient."""
    torch.manual_seed(2)
    encoder = (
        NuclrBinnedEncoder(
            ctx_duration=1.0,
            latent_step=0.25,
            bin_size=0.05,
            dim=64,
            self_heads=4,
            dim_head=16,
            t_layers=1,
            st_layers=1,
            rot_ratio=0.5,
        )
        .to(device)
        .train()
    )

    unit_seqlen = torch.tensor([3, 4], device=device)
    total_units = int(unit_seqlen.sum())
    num_latents = encoder.num_latents
    bins_per_latent = encoder.bins_per_latent
    bins = torch.randn(
        total_units * num_latents,
        bins_per_latent,
        device=device,
        requires_grad=True,
    )

    out = encoder(bins, unit_seqlen)
    out.sum().backward()

    grad_missing = [
        name
        for name, p in encoder.named_parameters()
        if p.requires_grad and (p.grad is None or torch.all(p.grad == 0))
    ]
    assert not grad_missing, f"Params with no/zero grad: {grad_missing}"
