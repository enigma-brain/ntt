"""Tests for BSQBottleneck round-trip identity and STE gradient.

Covers two properties of `BSQBottleneck`:
1. The STE in `_quantize` must scale by `codebook_scale` *inside* the detach so
   that ∂(quantized)/∂z = 1 (matching Infinity's `quantize_new`), not 1/√d.
2. `indices_to_codes` must produce the same pre-`project_out` magnitude as
   `encode` so round-tripping through integer indices is lossless.
"""

from __future__ import annotations

import pytest
import torch

from ntt.contrib.tokenizer.wanneural.bottleneck import BSQBottleneck


@pytest.mark.parametrize("z_dim,codebook_dim", [(16, None), (16, 24), (8, 32)])
def test_bsq_roundtrip_matches_encode(z_dim, codebook_dim):
    """encode(h) == indices_to_codes(get_indices(h), T) for any h."""
    torch.manual_seed(0)
    bsq = BSQBottleneck(z_dim=z_dim, codebook_dim=codebook_dim).eval()
    h = torch.randn(2, z_dim, 4, 5)

    z_enc, _ = bsq.encode(h)
    idx = bsq.get_indices(h)
    z_rec = bsq.indices_to_codes(idx, T=4)

    assert z_enc.shape == z_rec.shape
    torch.testing.assert_close(z_enc, z_rec, rtol=1e-5, atol=1e-5)


def test_bsq_ste_gradient_passthrough():
    """STE must yield ∂(quantized)/∂z ≈ 1, not codebook_scale = 1/√d."""
    torch.manual_seed(0)
    z_dim = 16
    bsq = BSQBottleneck(z_dim=z_dim, codebook_dim=z_dim)
    # Bypass project_in/out so we isolate the STE behavior.
    bsq.project_in = torch.nn.Identity()
    bsq.project_out = torch.nn.Identity()

    h = torch.randn(1, z_dim, 1, 1, requires_grad=True)
    z_q, _ = bsq.encode(h)
    z_q.sum().backward()

    # With identity projections, l2norm is the only nonlinearity; for unit-norm
    # inputs its Jacobian has unit operator norm. A correct STE should give a
    # gradient of order 1; a buggy STE scales it down by 1/√d ≈ 0.25.
    grad_norm = h.grad.norm().item()
    assert grad_norm > 0.5, f"encoder gradient suspiciously small: {grad_norm}"
