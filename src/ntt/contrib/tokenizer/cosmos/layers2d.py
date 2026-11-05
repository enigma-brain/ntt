"""Causal-in-time 2D layers for the Cosmos 2D tokenizer.

Tensor layout throughout: ``[B, C, H, W]`` where ``H`` is the neuron axis
(non-causal) and ``W`` is the time axis (causal). Convolutions pad the left
side of W only; H is padded symmetrically. GroupNorm is applied per-time-
column so normalization statistics don't leak the future.

Adapted from NVIDIA Cosmos-Tokenizer, used under the Apache-2.0 license:
https://github.com/NVIDIA/Cosmos-Tokenizer
(see ``LICENSES/cosmos-tokenizer-Apache-2.0.txt``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from ntt.contrib.tokenizer.cosmos.patcher import Patcher2D, UnPatcher2D


def _nonlinearity(x: torch.Tensor) -> torch.Tensor:
    """Swish / SiLU activation, matching Cosmos's ``nonlinearity``."""
    return x * torch.sigmoid(x)


def _pick_num_groups(num_channels: int, preferred: int) -> int:
    """Largest divisor of ``num_channels`` that is ``<= preferred``."""
    groups = min(preferred, num_channels)
    while groups > 1 and num_channels % groups != 0:
        groups -= 1
    return groups


class CausalGroupNorm2d(nn.Module):
    """GroupNorm applied per time column — preserves causality along W.

    Standard ``nn.GroupNorm`` on ``[B, C, H, W]`` reduces over ``(C/G, H, W)``,
    leaking future time into each output position. We fold ``W`` into the batch
    axis so each column is normalized independently over ``(C/G, H)``.
    """

    def __init__(self, num_channels: int, num_groups: int = 32) -> None:
        super().__init__()
        groups = _pick_num_groups(num_channels, num_groups)
        self.norm = nn.GroupNorm(
            num_groups=groups, num_channels=num_channels, eps=1e-6, affine=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # [B, C, H, W] -> [B*W, C, H] so GroupNorm normalizes per (B, W).
        x = x.permute(0, 3, 1, 2).reshape(b * w, c, h)
        x = self.norm(x)
        return x.reshape(b, w, c, h).permute(0, 2, 3, 1)


class CausalConv2d(nn.Module):
    """Conv2d that is causal along W (time) and symmetric along H (neuron).

    ``kernel_size`` / ``stride`` / ``dilation`` each accept ``int`` or
    ``tuple[int, int]`` in ``(kh, kw)`` order. W padding is ``(kw - 1) * dil_w``
    on the left only; H padding is ``(kh - 1) // 2`` on each side so spatial
    size is preserved when stride_h = 1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        dilation: int | tuple[int, int] = 1,
    ) -> None:
        super().__init__()
        kh, kw = _pair(kernel_size)
        sh, sw = _pair(stride)
        dh, dw = _pair(dilation)

        if kh % 2 == 0:
            raise ValueError(f"H kernel must be odd for symmetric padding, got {kh}")

        self.left_pad_w = (kw - 1) * dw
        pad_h = ((kh - 1) // 2) * dh

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kh, kw),
            stride=(sh, sw),
            dilation=(dh, dw),
            padding=(pad_h, 0),  # H symmetric, W handled manually below
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.left_pad_w, 0, 0, 0))
        return self.conv(x)


class CausalDownsample2d(nn.Module):
    """Stride-2 (per axis) causal Conv2d. ``stride_h`` / ``stride_w`` ∈ {1, 2}."""

    def __init__(self, channels: int, stride_h: int = 2, stride_w: int = 2) -> None:
        super().__init__()
        self.conv = CausalConv2d(channels, channels, kernel_size=3, stride=(stride_h, stride_w))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class CausalUpsample2d(nn.Module):
    """Per-axis 2x upsample via repeat-interleave + causal Conv2d. Strides ∈ {1, 2}."""

    def __init__(self, channels: int, stride_h: int = 2, stride_w: int = 2) -> None:
        super().__init__()
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.conv = CausalConv2d(channels, channels, kernel_size=3, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride_h > 1:
            x = x.repeat_interleave(self.stride_h, dim=-2)
        if self.stride_w > 1:
            x = x.repeat_interleave(self.stride_w, dim=-1)
        return self.conv(x)


class ResBlock2d(nn.Module):
    """GroupNorm + Swish + CausalConv2d (x2) with 1x1 shortcut."""

    def __init__(self, in_channels: int, out_channels: int | None, dropout: float) -> None:
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels

        self.norm1 = CausalGroupNorm2d(in_channels)
        self.conv1 = CausalConv2d(in_channels, out_channels, kernel_size=3)
        self.norm2 = CausalGroupNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = CausalConv2d(out_channels, out_channels, kernel_size=3)
        self.shortcut: nn.Module = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(_nonlinearity(self.norm1(x)))
        h = self.conv2(self.dropout(_nonlinearity(self.norm2(h))))
        return self.shortcut(x) + h


class _DownStage2d(nn.Module):
    """One encoder resolution: ResBlocks then optional per-axis downsample."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int,
        dropout: float,
        do_downsample_h: bool,
        do_downsample_w: bool,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ResBlock2d(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    dropout=dropout,
                )
                for i in range(num_res_blocks)
            ]
        )
        self.downsample: nn.Module = (
            CausalDownsample2d(
                out_channels,
                stride_h=2 if do_downsample_h else 1,
                stride_w=2 if do_downsample_w else 1,
            )
            if (do_downsample_h or do_downsample_w)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.downsample(x)


class _UpStage2d(nn.Module):
    """One decoder resolution: ResBlocks then optional per-axis upsample."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int,
        dropout: float,
        do_upsample_h: bool,
        do_upsample_w: bool,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ResBlock2d(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    dropout=dropout,
                )
                for i in range(num_res_blocks + 1)
            ]
        )
        self.upsample: nn.Module = (
            CausalUpsample2d(
                out_channels,
                stride_h=2 if do_upsample_h else 1,
                stride_w=2 if do_upsample_w else 1,
            )
            if (do_upsample_h or do_upsample_w)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.upsample(x)


class Encoder2D(nn.Module):
    """2D encoder with independent H and W compression.

    Input  ``[B, in_channels, H, W]``
    Output ``[B, z_channels, H / neuron_compression, W / temporal_compression]``

    Each axis's total downsample = ``patch_*`` (wavelet) × ``2^k`` (strided
    Conv2d stages, where ``k = log2(compression / patch_*)``). The first
    ``k_h`` stages stride H by 2; the first ``k_w`` stages stride W by 2;
    stages beyond ``max(k_h, k_w)`` are pure ResBlocks.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        channels: int,
        channels_mult: list[int],
        num_res_blocks: int,
        dropout: float,
        z_channels: int,
        neuron_compression: int,
        temporal_compression: int,
        patch_h: int = 1,
        patch_w: int = 4,
        patch_method: str = "haar",
    ) -> None:
        super().__init__()
        _validate_compression(neuron_compression, patch_h, "neuron")
        _validate_compression(temporal_compression, patch_w, "temporal")
        num_h_downsamples = (neuron_compression // patch_h).bit_length() - 1
        num_w_downsamples = (temporal_compression // patch_w).bit_length() - 1
        num_stages_needed = max(num_h_downsamples, num_w_downsamples)
        if num_stages_needed > len(channels_mult):
            raise ValueError(
                f"need {num_stages_needed} downsample stages "
                f"(max(H={num_h_downsamples}, W={num_w_downsamples})) but channels_mult "
                f"has only {len(channels_mult)} entries"
            )

        self.patcher = Patcher2D(patch_h=patch_h, patch_w=patch_w, patch_method=patch_method)
        self.conv_in = CausalConv2d(in_channels * patch_h * patch_w, channels, kernel_size=3)

        in_ch_mult = (1, *channels_mult)
        self.down = nn.ModuleList(
            [
                _DownStage2d(
                    in_channels=channels * in_ch_mult[i],
                    out_channels=channels * channels_mult[i],
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    do_downsample_h=i < num_h_downsamples,
                    do_downsample_w=i < num_w_downsamples,
                )
                for i in range(len(channels_mult))
            ]
        )

        final_channels = channels * channels_mult[-1]
        self.mid_block_1 = ResBlock2d(final_channels, final_channels, dropout)
        self.mid_block_2 = ResBlock2d(final_channels, final_channels, dropout)
        self.norm_out = CausalGroupNorm2d(final_channels)
        self.conv_out = CausalConv2d(final_channels, z_channels, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patcher(x)
        h = self.conv_in(x)
        for stage in self.down:
            h = stage(h)
        h = self.mid_block_1(h)
        h = self.mid_block_2(h)
        return self.conv_out(_nonlinearity(self.norm_out(h)))


class Decoder2D(nn.Module):
    """Mirror of ``Encoder2D``: per-axis upsample + UnPatcher2D back to input shape.

    Input  ``[B, z_channels, H / neuron_compression, W / temporal_compression]``
    Output ``[B, out_channels, H, W]`` (optionally trimmed via ``trim_h`` / ``trim_w``).
    """

    def __init__(
        self,
        *,
        out_channels: int,
        channels: int,
        channels_mult: list[int],
        num_res_blocks: int,
        dropout: float,
        z_channels: int,
        neuron_compression: int,
        temporal_compression: int,
        patch_h: int = 1,
        patch_w: int = 4,
        patch_method: str = "haar",
    ) -> None:
        super().__init__()
        _validate_compression(neuron_compression, patch_h, "neuron")
        _validate_compression(temporal_compression, patch_w, "temporal")
        num_h_upsamples = (neuron_compression // patch_h).bit_length() - 1
        num_w_upsamples = (temporal_compression // patch_w).bit_length() - 1
        num_stages_needed = max(num_h_upsamples, num_w_upsamples)
        num_resolutions = len(channels_mult)
        if num_stages_needed > num_resolutions:
            raise ValueError(
                f"need {num_stages_needed} upsample stages but channels_mult "
                f"has only {num_resolutions} entries"
            )

        final_channels = channels * channels_mult[-1]
        self.conv_in = CausalConv2d(z_channels, final_channels, kernel_size=3)
        self.mid_block_1 = ResBlock2d(final_channels, final_channels, dropout)
        self.mid_block_2 = ResBlock2d(final_channels, final_channels, dropout)

        # Match encoder: stage i upsamples H if i < num_h_upsamples, W if i < num_w_upsamples.
        stages: list[nn.Module] = []
        block_in = final_channels
        for i_level in reversed(range(num_resolutions)):
            block_out = channels * channels_mult[i_level]
            stages.insert(
                0,
                _UpStage2d(
                    in_channels=block_in,
                    out_channels=block_out,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    do_upsample_h=i_level < num_h_upsamples,
                    do_upsample_w=i_level < num_w_upsamples,
                ),
            )
            block_in = block_out
        self.up = nn.ModuleList(stages)

        self.norm_out = CausalGroupNorm2d(block_in)
        # UnPatcher2D expects ``out_channels * patch_h * patch_w`` fan-in to invert the DWT.
        self.conv_out = CausalConv2d(block_in, out_channels * patch_h * patch_w, kernel_size=3)
        self.unpatcher = UnPatcher2D(patch_h=patch_h, patch_w=patch_w, patch_method=patch_method)

    def forward(
        self,
        z: torch.Tensor,
        trim_h: int | None = None,
        trim_w: int | None = None,
    ) -> torch.Tensor:
        h = self.conv_in(z)
        h = self.mid_block_1(h)
        h = self.mid_block_2(h)
        for stage in reversed(self.up):
            h = stage(h)
        h = self.conv_out(_nonlinearity(self.norm_out(h)))
        return self.unpatcher(h, trim_h=trim_h, trim_w=trim_w)


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    return value


def _validate_compression(compression: int, patch_size: int, name: str) -> None:
    if compression < 1 or (compression & (compression - 1)) != 0:
        raise ValueError(f"{name}_compression must be a positive power of two, got {compression}")
    if compression % patch_size != 0:
        raise ValueError(
            f"{name}_compression ({compression}) must be a multiple of patch_{name[0]} "
            f"({patch_size})"
        )
    remaining = compression // patch_size
    if remaining <= 0 or (remaining & (remaining - 1)) != 0:
        raise ValueError(f"{name}_compression / patch must be a power of two, got {remaining}")
