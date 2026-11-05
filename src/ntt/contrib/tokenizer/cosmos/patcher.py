"""Causal-in-time 2D wavelet-packet patcher.

Applies ``log2(patch_h)`` and ``log2(patch_w)`` levels of packet-style DWT
along the neuron (H) and time (W) axes respectively. Output has
``patch_h * patch_w`` times the input channel count, downsampled by
``patch_h`` along H and ``patch_w`` along W.

Implementation delegates to ``pytorch_wavelets.dwt.lowlevel.{afb1d, sfb1d}``
with ``mode="symmetric"`` (reflect extension), giving bit-exact perfect
reconstruction across the orthogonal wavelets we support
(haar, db2, db4, sym4).

Non-causal at the wavelet boundary, but local: symmetric extension mixes
only ~L samples at each edge (L = filter length). Downstream encoder layers
remain strictly causal in time (see ``layers2d.py::CausalConv2d``).

Channel ordering note: ``afb1d`` interleaves subbands per input channel
(``[ch0_lo, ch0_hi, ch1_lo, ch1_hi, ...]``); the inverse splits even/odd
channel indices to feed ``sfb1d``.

Adapted from NVIDIA Cosmos-Tokenizer, used under the Apache-2.0 license:
https://github.com/NVIDIA/Cosmos-Tokenizer
(see ``LICENSES/cosmos-tokenizer-Apache-2.0.txt``).
"""

from __future__ import annotations

import math
from typing import cast

import torch
from pytorch_wavelets.dwt.lowlevel import afb1d, sfb1d
from torch import nn

from ntt.contrib.tokenizer.cosmos.wavelets import get_filters


def _assert_power_of_two(n: int, name: str) -> None:
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"{name} must be a positive power of two, got {n}")


class Patcher2D(nn.Module):
    """Multi-level 2D wavelet packet patcher with independent H and W ratios.

    Input  ``[B, C, H, W]``. Each axis is decomposed by ``log2(patch_*)``
    levels; ``patch_h`` or ``patch_w`` of ``1`` skips that axis (identity).
    Output ``[B, C * patch_h * patch_w, H_out, W_out]``.
    """

    def __init__(
        self,
        patch_h: int = 4,
        patch_w: int = 4,
        patch_method: str = "haar",
    ) -> None:
        super().__init__()
        _assert_power_of_two(patch_h, "patch_h")
        _assert_power_of_two(patch_w, "patch_w")
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_method = patch_method
        self.num_levels_h = int(math.log2(patch_h))
        self.num_levels_w = int(math.log2(patch_w))

        h0a, h1a, _, _ = get_filters(patch_method)
        self.register_buffer("h0a", h0a, persistent=False)
        self.register_buffer("h1a", h1a, persistent=False)
        self.h0a: torch.Tensor
        self.h1a: torch.Tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Patcher2D expects [B, C, H, W], got shape {tuple(x.shape)}")
        # W first, then H; matches the original ordering for the symmetric case.
        for _ in range(self.num_levels_w):
            x = afb1d(x, self.h0a, self.h1a, mode="symmetric", dim=3)
        for _ in range(self.num_levels_h):
            x = afb1d(x, self.h0a, self.h1a, mode="symmetric", dim=2)
        return x


class UnPatcher2D(nn.Module):
    """Inverse of ``Patcher2D`` with independent H and W ratios."""

    def __init__(
        self,
        patch_h: int = 4,
        patch_w: int = 4,
        patch_method: str = "haar",
    ) -> None:
        super().__init__()
        _assert_power_of_two(patch_h, "patch_h")
        _assert_power_of_two(patch_w, "patch_w")
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_method = patch_method
        self.num_levels_h = int(math.log2(patch_h))
        self.num_levels_w = int(math.log2(patch_w))

        _, _, g0a, g1a = get_filters(patch_method)
        self.register_buffer("g0a", g0a, persistent=False)
        self.register_buffer("g1a", g1a, persistent=False)
        self.g0a: torch.Tensor
        self.g1a: torch.Tensor

    def forward(
        self,
        x: torch.Tensor,
        trim_h: int | None = None,
        trim_w: int | None = None,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"UnPatcher2D expects [B, C, H, W], got shape {tuple(x.shape)}")
        # Inverse of Patcher2D: H first (last in by Patcher), then W.
        for _ in range(self.num_levels_h):
            x = cast(
                torch.Tensor,
                sfb1d(x[:, 0::2], x[:, 1::2], self.g0a, self.g1a, mode="symmetric", dim=2),
            )
        for _ in range(self.num_levels_w):
            x = cast(
                torch.Tensor,
                sfb1d(x[:, 0::2], x[:, 1::2], self.g0a, self.g1a, mode="symmetric", dim=3),
            )
        if trim_h is not None and x.shape[-2] > trim_h:
            x = x[..., :trim_h, :]
        if trim_w is not None and x.shape[-1] > trim_w:
            x = x[..., :trim_w]
        return x
