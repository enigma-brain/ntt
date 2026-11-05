"""Wavelet registry backed by ``pywt`` + ``pytorch_wavelets``.

We delegate all coefficient maintenance to PyWavelets and all filter-bank
plumbing to ``pytorch_wavelets``. This module is just a thin helper that
exposes ``(h0a, h1a, g0a, g1a)`` tensors pre-formatted for
``pytorch_wavelets.dwt.lowlevel.afb1d`` / ``sfb1d``.

Symmetric (reflect) extension is used throughout the patchers, which gives
bit-exact perfect reconstruction for any supported orthogonal wavelet — see
``tests/test_cosmos_tokenizer.py::test_patcher_roundtrip_is_bit_exact``.
"""

from __future__ import annotations

import pywt
import torch
from pytorch_wavelets.dwt.lowlevel import prep_filt_afb1d, prep_filt_sfb1d

SUPPORTED_WAVELETS: tuple[str, ...] = ("haar", "db2", "db4", "sym4")


def get_filters(
    name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(h0a, h1a, g0a, g1a)`` analysis / synthesis filter tensors.

    Each tensor has shape ``(1, 1, L)`` — the format expected by
    ``pytorch_wavelets`` 1D filter-bank ops.
    """
    if name not in SUPPORTED_WAVELETS:
        raise ValueError(f"Unknown wavelet '{name}'. Supported: {SUPPORTED_WAVELETS}")
    w = pywt.Wavelet(name)  # pyright: ignore[reportAttributeAccessIssue]  # missing in pywt stubs
    h0a, h1a = prep_filt_afb1d(w.dec_lo, w.dec_hi)
    g0a, g1a = prep_filt_sfb1d(w.rec_lo, w.rec_hi)
    return h0a, h1a, g0a, g1a
