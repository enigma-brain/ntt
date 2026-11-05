"""Building blocks for the WanNeural tokenizer.

Adapts the 3D convolutional blocks from Wan2pt1/Wan2pt2 video VAE architectures
to 2D (time x neurons) for neural data tokenization.

Key adaptations from Wan2:
- CausalConv3d(kt, kh, kw) → CausalConv2d(kt, kn): causal along time, symmetric along neurons
- 5D tensors (B, C, T, H, W) → 4D tensors (B, C, T, N)
- Spatial attention across H*W → attention across N (neurons)
- Neuron kernel size kn is configurable (1 = per-neuron, 3 = local mixing)

Adapted from the Wan2.1/Wan2.2 video VAE, used under the Apache-2.0 license:
https://github.com/Wan-Video/Wan2.2 (see ``LICENSES/wan-Apache-2.0.txt``).
"""

from __future__ import annotations

from typing import Literal, cast

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


class RMSNorm2d(nn.Module):
    """RMS normalization for 4D tensors (B, C, T, N).

    Adapted from Wan2.1's RMS_norm. Normalizes across C, scales by
      learnable gamma with (optional) learnable bias.
    """

    def __init__(self, dim: int, bias: bool = False):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1))
        self.bias = nn.Parameter(torch.zeros(dim, 1, 1)) if bias else 0.0
        self.scale = dim**0.5

    def forward(self, x: Tensor) -> Tensor:
        """(B, C, T, N) → (B, C, T, N)."""
        return F.normalize(x, dim=1) * self.scale * self.gamma + self.bias


def _make_norm(
    dim: int, norm_type: Literal["rms", "group"] = "rms", num_groups: int = 32
) -> nn.Module:
    """Build a normalization layer for 4D (B, C, T, N) tensors.

    norm_type="rms" returns RMSNorm2d (Wan2 default).
    norm_type="group" returns nn.GroupNorm — matches the LDM autoencoder.
    GroupNorm requires ``dim % num_groups == 0``; if ``dim < num_groups`` the
    group count falls back to ``dim`` (one channel per group = InstanceNorm).
    """
    if norm_type == "rms":
        return RMSNorm2d(dim)
    if norm_type == "group":
        ng = min(num_groups, dim)
        if dim % ng != 0:
            raise ValueError(
                f"GroupNorm requires num_channels ({dim}) to be divisible by num_groups ({ng})."
            )
        return nn.GroupNorm(ng, dim, eps=1e-6, affine=True)
    raise ValueError(f"Unknown norm_type: {norm_type!r} (expected 'rms' or 'group')")


# ---------------------------------------------------------------------------
# Causal Conv2d
# ---------------------------------------------------------------------------


class CausalConv2d(nn.Conv2d):
    """2D convolution with causal (left-only) padding along time, symmetric along neurons.

    Adapted from Wan2pt2's CausalConv3d. Inherits nn.Conv2d and automatically
    computes padding from kernel_size and dilation — callers should NOT pass
    ``padding`` explicitly.

    Padding (when ``causal=True``, the default):
        T (dim 0): ``dilation[0] * (kernel_size[0] - 1)`` on the left, 0 on the right
        N (dim 1): ``dilation[1] * (kernel_size[1] - 1) // 2`` on each side (symmetric)

    Padding (when ``causal=False``):
        T (dim 0): ``dilation[0] * (kernel_size[0] - 1) // 2`` on each side (symmetric)
        N (dim 1): same as causal (symmetric)
        Output is "same"-shaped for stride 1; for stride > 1 the output length is
        ``ceil(T / stride)``, matching the causal-padded variant when T is a
        multiple of stride.
    """

    def __init__(self, *args, causal: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.padding == (0, 0), "Padding passed explicitly to CausalConv2d"

        # Compute padding from kernel_size and dilation
        kt, kn = self.kernel_size
        dt, dn = self.dilation
        pad_n = dn * (kn - 1) // 2  # symmetric: equal on both sides
        if causal:
            pad_t_left = dt * (kt - 1)  # causal: all on left
            pad_t_right = 0
        else:
            # Symmetric same-padding along time. For odd kernels this is exact;
            # for even kernels we put the extra bin on the left (matches Conv2d
            # semantics with padding=floor((k-1)/2)).
            pad_t_left = dt * (kt - 1) // 2
            pad_t_right = dt * (kt - 1) - pad_t_left
        # F.pad order: (N_left, N_right, T_left, T_right)
        self._padding = (pad_n, pad_n, pad_t_left, pad_t_right)
        self.causal = causal

    def forward(self, x):
        x = F.pad(x, self._padding)
        return super().forward(x)


class CausalConvTranspose2d(nn.Module):
    """Transposed 2D convolution that undoes CausalConv2d's temporal stride.

    Used for upsampling in the decoder. Only upsamples along T (stride along N is 1).

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: (kt, kn) kernel dimensions.
        stride: (st, sn) stride dimensions — st > 1 upsamples temporal.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int] = (3, 1),
        stride: tuple[int, int] = (2, 1),
        bias: bool = True,
        causal: bool = True,
    ):
        super().__init__()
        kt, kn = kernel_size
        st, sn = stride

        # Use built-in padding for neuron dim (symmetric), but not for
        # temporal dim (needs causal/symmetric trim, handled manually in forward).
        pad_n = (kn - 1) // 2
        self.conv_t = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
            padding=(0, pad_n),
            output_padding=0,
        )
        # Trim along T so the output matches the desired length T*st.
        # nn.ConvTranspose2d with stride st produces (T-1)*st + kt timesteps.
        # Causal: trim (kt - st) from the start to keep the right edge aligned.
        # Non-causal: split the trim across both ends symmetrically.
        total_trim = kt - st
        self.causal = causal
        if causal:
            self.trim_t_left = max(0, total_trim)
            self.trim_t_right = 0
        else:
            self.trim_t_left = max(0, total_trim) // 2
            self.trim_t_right = max(0, total_trim) - self.trim_t_left

    def forward(self, x: Tensor) -> Tensor:
        """(B, C, T, N) → (B, C_out, T*st, N)."""
        out = self.conv_t(x)
        if self.trim_t_left > 0:
            out = out[:, :, self.trim_t_left :, :]
        if self.trim_t_right > 0:
            out = out[:, :, : -self.trim_t_right, :]
        return out


# ---------------------------------------------------------------------------
# Residual Block
# ---------------------------------------------------------------------------


class ResidualBlock(nn.Module):
    """Residual block with two CausalConv2d layers and configurable normalization.

    Adapted from Wan2pt1's ResidualBlock; dropout placement matches the LDM
    autoencoder (between the second SiLU and the second conv).

    Architecture:
        norm → SiLU → CausalConv2d(in→out) → norm → SiLU → [Dropout if p>0] → CausalConv2d(out→out)
        + shortcut (identity or 1x1 projection)

    Args:
        in_channels: Input channel dimension.
        out_channels: Output channel dimension.
        neuron_kernel_size: Kernel size along the neuron dimension (1 or 3).
        temporal_kernel_size: Kernel size along the temporal dimension (1 or 3).
        dropout: Dropout probability before the second conv. 0.0 disables it.
        norm_type: "rms" (default, Wan2 RMSNorm) or "group" (LDM GroupNorm).
        num_groups: Number of groups when ``norm_type="group"``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        neuron_kernel_size: int = 1,
        temporal_kernel_size: int = 3,
        causal: bool = True,
        dropout: float = 0.0,
        norm_type: Literal["rms", "group"] = "rms",
        num_groups: int = 32,
    ):
        super().__init__()
        kt = temporal_kernel_size
        kn = neuron_kernel_size

        layers: list[nn.Module] = [
            _make_norm(in_channels, norm_type, num_groups),
            nn.SiLU(),
            CausalConv2d(in_channels, out_channels, (kt, kn), causal=causal),
            _make_norm(out_channels, norm_type, num_groups),
            nn.SiLU(),
        ]
        # Only insert Dropout when active: with dropout=0 the layer is omitted,
        # so the residual.* indices (and state_dict keys) are unchanged.
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(CausalConv2d(out_channels, out_channels, (kt, kn), causal=causal))
        self.residual = nn.Sequential(*layers)
        if in_channels != out_channels:
            self.shortcut = CausalConv2d(in_channels, out_channels, 1, causal=causal)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """(B, C_in, T, N) → (B, C_out, T, N)."""
        return self.residual(x) + self.shortcut(x)


# ---------------------------------------------------------------------------
# Attention Block (cross-neuron)
# ---------------------------------------------------------------------------


class AttentionBlock(nn.Module):
    """Self-attention along the neuron or temporal axis, optionally with FFN and RoPE.

    Adapted from Wan2's AttentionBlock. The neuron-axis branch is unchanged
    from the original: for each time frame independently, attend over N.
    The temporal-axis branch attends over T for each neuron independently,
    optionally restricted to a block-diagonal mask of width ``block_size``
    (with optional within-block causality).

    Optionally extends to a full pre-norm Transformer block by adding a
    feed-forward residual after the attention residual (``use_ffn=True``).
    For temporal-axis attention, optionally applies Rotary Positional
    Embeddings to Q and K (``use_rope=True``).

    Args:
        channels: Number of channels.
        num_heads: Number of attention heads.
        axis: Which axis to attend along. ``"neuron"`` is the original behavior.
        block_size: For ``axis="temporal"`` only. ``> 0`` builds a block-diagonal
            (T, T) mask of width ``block_size`` so frames only attend within
            the same block. ``0`` disables block masking (full T).
        causal: For ``axis="temporal"`` only. If True, AND the mask with a
            lower-triangular pattern so each frame only attends to current
            and earlier frames within its block.
        use_ffn: If True, append a pre-norm FFN residual after the attention
            residual (turns the block into a full Transformer block). The FFN
            is ``Conv2d(C → ffn_dim_mult·C, 1) → SiLU → Conv2d(ffn_dim_mult·C → C, 1)``;
            the second conv is zero-init so the block is identity at start.
        ffn_dim_mult: Hidden-dim multiplier for the FFN (ignored if
            ``use_ffn=False``). Default 4 matches CALM.
        use_rope: If True, apply Rotary Positional Embedding to Q and K.
            Only valid for ``axis="temporal"`` — neurons have no canonical
            ordering so RoPE on the neuron axis is meaningless and raises.
            Requires ``head_dim = channels // num_heads`` to be even.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 1,
        norm_type: Literal["rms", "group"] = "rms",
        num_groups: int = 32,
        axis: Literal["neuron", "temporal"] = "neuron",
        block_size: int = 0,
        causal: bool = False,
        use_ffn: bool = False,
        ffn_dim_mult: int = 4,
        use_rope: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.axis = axis
        self.block_size = block_size
        self.causal = causal
        self.use_ffn = use_ffn
        self.use_rope = use_rope

        # Cache (T, device) → mask to avoid O(T²) allocation on every forward.
        self._mask_cache: dict[tuple[int, str], Tensor | None] = {}

        self.norm = _make_norm(channels, norm_type, num_groups)

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)

        # Zero-initialize output projection for stable residual learning
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)  # type: ignore

        if use_rope:
            if axis != "temporal":
                raise ValueError(
                    f"AttentionBlock: use_rope=True requires axis='temporal' "
                    f"(neurons have no canonical ordering); got axis={axis!r}"
                )
            head_dim = channels // num_heads
            if head_dim % 2 != 0:
                raise ValueError(
                    f"AttentionBlock: use_rope=True requires even head_dim "
                    f"(channels // num_heads); got channels={channels}, "
                    f"num_heads={num_heads} → head_dim={head_dim}"
                )
            # Import locally to avoid pulling calm at module import time when
            # RoPE isn't used.
            from ntt.contrib.tokenizer.calm.rope import RotaryPositionalEmbedding

            self.rope = RotaryPositionalEmbedding(head_dim)

        if use_ffn:
            self.norm2 = _make_norm(channels, norm_type, num_groups)
            ffn_hidden = ffn_dim_mult * channels
            self.ffn = nn.Sequential(
                nn.Conv2d(channels, ffn_hidden, kernel_size=1),
                nn.SiLU(),
                nn.Conv2d(ffn_hidden, channels, kernel_size=1),
            )
            # Zero-init the last FFN conv so the block is identity at start
            # (matching the out_proj convention).
            last_conv = cast(nn.Conv2d, self.ffn[-1])
            assert last_conv.bias is not None
            nn.init.zeros_(last_conv.weight)
            nn.init.zeros_(last_conv.bias)

    def _temporal_mask(self, t: int, device: torch.device) -> Tensor | None:
        """Build the (T, T) bool attention mask for the temporal axis.

        Returns None when no constraint is active (no block masking, no causal),
        which lets SDPA skip mask handling entirely. Result is cached on
        (t, device) to avoid O(T²) allocation on every forward pass.
        """
        if self.block_size <= 0 and not self.causal:
            return None
        key = (t, str(device))
        if key in self._mask_cache:
            return self._mask_cache[key]
        if self.block_size > 0:
            block_idx = torch.arange(t, device=device) // self.block_size
            mask = block_idx[:, None] == block_idx[None, :]
        else:
            mask = torch.ones(t, t, dtype=torch.bool, device=device)
        if self.causal:
            mask = mask & torch.tril(torch.ones(t, t, dtype=torch.bool, device=device))
        self._mask_cache[key] = mask
        return mask

    def forward(
        self,
        x: Float[Tensor, "b c t n"],
        neuron_mask: Bool[Tensor, "b n"] | None = None,
    ) -> Float[Tensor, "b c t n"]:
        """(B, C, T, N) → (B, C, T, N).

        Args:
            x: Input tensor.
            neuron_mask: Optional (B, N) bool, True = valid. Used by ``axis="neuron"``
                to exclude padded neurons from attention. Accepted but unused for
                ``axis="temporal"`` (each neuron's time series is its own sequence).
        """
        b, c, t, n = x.shape

        qkv = self.qkv(self.norm(x))

        if self.axis == "neuron":
            d = qkv.shape[1] // (3 * self.num_heads)
            # "b (three heads d) t n -> three (b t) heads n d"
            q, k, v = (
                qkv.reshape(b, 3, self.num_heads, d, t, n)
                .permute(1, 0, 4, 2, 5, 3)  # (3, b, t, heads, n, d)
                .reshape(3, b * t, self.num_heads, n, d)
                .unbind(0)
            )

            attn_mask = None
            if neuron_mask is not None:
                attn_mask = neuron_mask[:, None, None, :].repeat_interleave(t, dim=0)

            h = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

            # "(b t) heads n d -> b (heads d) t n"
            d_h = h.shape[3]
            h = (
                h.reshape(b, t, self.num_heads, n, d_h)
                .permute(0, 2, 4, 1, 3)
                .reshape(b, self.num_heads * d_h, t, n)
            )
        else:  # axis == "temporal"
            d = qkv.shape[1] // (3 * self.num_heads)
            # "b (three heads d) t n -> three (b n) heads t d"
            q, k, v = (
                qkv.reshape(b, 3, self.num_heads, d, t, n)
                .permute(1, 0, 5, 2, 4, 3)  # (3, b, n, heads, t, d)
                .reshape(3, b * n, self.num_heads, t, d)
                .unbind(0)
            )

            if self.use_rope:
                from ntt.contrib.tokenizer.calm.rope import apply_rotary_emb

                cos, sin = self.rope(t, device=x.device)
                q, k = apply_rotary_emb(q, k, cos, sin)

            attn_mask = self._temporal_mask(t, x.device)

            h = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

            # "(b n) heads t d -> b (heads d) t n"
            d_h = h.shape[3]
            h = (
                h.reshape(b, n, self.num_heads, t, d_h)
                .permute(0, 2, 4, 3, 1)
                .reshape(b, self.num_heads * d_h, t, n)
            )

        x = x + self.out_proj(h)
        if self.use_ffn:
            x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Downsampling / Upsampling
# ---------------------------------------------------------------------------


class TemporalDownsample(nn.Module):
    """Downsample temporal dimension using strided causal conv.

    Args:
        channels: Number of channels.
        neuron_kernel_size: Kernel size along N.
        temporal_kernel_size: Kernel size along T.
        temporal_stride: Stride along T (default 2).
    """

    def __init__(
        self,
        channels: int,
        neuron_kernel_size: int = 1,
        temporal_kernel_size: int = 3,
        temporal_stride: int = 2,
        causal: bool = True,
    ):
        super().__init__()
        kt = temporal_kernel_size
        kn = neuron_kernel_size
        self.conv = CausalConv2d(
            channels, channels, (kt, kn), stride=(temporal_stride, 1), causal=causal
        )

    def forward(self, x: Tensor) -> Tensor:
        """(B, C, T, N) → (B, C, T//stride, N)."""
        return self.conv(x)


class TemporalUpsample(nn.Module):
    """Upsample temporal dimension using nearest interpolation + conv.

    Args:
        channels: Number of channels.
        neuron_kernel_size: Kernel size along N.
        temporal_kernel_size: Kernel size along T.
        temporal_stride: Upsample factor along T (default 2).
    """

    def __init__(
        self,
        channels: int,
        neuron_kernel_size: int = 1,
        temporal_kernel_size: int = 3,
        temporal_stride: int = 2,
        causal: bool = True,
    ):
        super().__init__()
        self.temporal_stride = temporal_stride
        kt = temporal_kernel_size
        kn = neuron_kernel_size
        self.conv = CausalConv2d(channels, channels, (kt, kn), causal=causal)

    def forward(self, x: Tensor) -> Tensor:
        """(B, C, T, N) → (B, C, T*stride, N)."""
        x = F.interpolate(
            x.float(), scale_factor=(float(self.temporal_stride), 1.0), mode="nearest"
        )
        return self.conv(x)


class AvgDown2D(nn.Module):
    """Average-pooling shortcut for temporal downsampling.

    Adapted from Wan2pt2's AvgDown3D. Optionally projects channels.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        t_stride: Temporal stride (default 2).
    """

    def __init__(
        self, in_channels: int, out_channels: int, t_stride: int = 2, causal: bool = True
    ):
        super().__init__()
        self.t_stride = t_stride
        self.causal = causal
        if in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.proj = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """(B, C_in, T, N) → (B, C_out, ceil(T/stride), N)."""
        x = self.proj(x)
        # Pad T to next multiple of stride. Causal: pad on left; non-causal:
        # split between left and right.
        pad_t = (self.t_stride - x.shape[2] % self.t_stride) % self.t_stride
        if pad_t > 0:
            if self.causal:
                pad_l, pad_r = pad_t, 0
            else:
                pad_l = pad_t // 2
                pad_r = pad_t - pad_l
            x = F.pad(x, (0, 0, pad_l, pad_r))
        b, c, t_full, n = x.shape
        t = t_full // self.t_stride
        return x.reshape(b, c, t, self.t_stride, n).mean(dim=3)


class DupUp2D(nn.Module):
    """Repeat-interleave shortcut for temporal upsampling.

    Adapted from Wan2pt2's DupUp3D. Optionally projects channels.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        t_stride: Temporal repeat factor (default 2).
    """

    def __init__(self, in_channels: int, out_channels: int, t_stride: int = 2):
        super().__init__()
        self.t_stride = t_stride
        if in_channels != out_channels:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.proj = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """(B, C_in, T, N) → (B, C_out, T*stride, N)."""
        x = self.proj(x)
        return x.repeat_interleave(self.t_stride, dim=2)


# ---------------------------------------------------------------------------
# Integrated Down/Up Residual Blocks (Wan2pt2-style)
# ---------------------------------------------------------------------------


class DownStage(nn.Module):
    """Encoder stage: residual blocks + optional attention + optional temporal downsample.

    Adapted from Wan2pt2's Down_ResidualBlock with AvgDown shortcut.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        num_res_blocks: Number of residual blocks in this stage.
        use_attention: Whether to include an attention block.
        temporal_downsample: Whether to downsample T.
        neuron_kernel_size: Conv kernel size along N.
        temporal_kernel_size: Temporal kernel size for residual blocks.
        down_temporal_kernel_size: Temporal kernel size for the downsample conv.
        temporal_stride: Temporal stride for downsampling.
        num_attn_heads: Number of attention heads.
        use_temporal_attention: Whether to insert a temporal-axis AttentionBlock before the
            neuron-axis one.
        temp_block_size: Block-diagonal width for the temporal mask (0 = full sequence).
        temp_num_attn_heads: Number of heads for the temporal AttentionBlock.
        temp_attn_causal: Whether to apply within-block causal masking on the temporal
            AttentionBlock.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int = 2,
        use_attention: bool = False,
        use_temporal_attention: bool = False,
        temporal_downsample: bool = True,
        neuron_kernel_size: int = 1,
        temporal_kernel_size: int = 3,
        down_temporal_kernel_size: int = 3,
        temporal_stride: int = 2,
        num_attn_heads: int = 1,
        attn_use_ffn: bool = False,
        attn_ffn_dim_mult: int = 4,
        temp_block_size: int = 0,
        temp_num_attn_heads: int = 1,
        temp_attn_causal: bool = False,
        temp_attn_use_ffn: bool = False,
        temp_attn_ffn_dim_mult: int = 4,
        temp_attn_use_rope: bool = False,
        use_stage_shortcut: bool = True,
        causal: bool = True,
        dropout: float = 0.0,
        norm_type: Literal["rms", "group"] = "rms",
        num_groups: int = 32,
    ):
        super().__init__()
        kt = temporal_kernel_size
        kn = neuron_kernel_size
        self.use_stage_shortcut = use_stage_shortcut

        # Residual blocks
        blocks: list[nn.Module] = []
        ch_in = in_channels
        for _ in range(num_res_blocks):
            blocks.append(
                ResidualBlock(
                    ch_in,
                    out_channels,
                    neuron_kernel_size=kn,
                    temporal_kernel_size=kt,
                    causal=causal,
                    dropout=dropout,
                    norm_type=norm_type,
                    num_groups=num_groups,
                )
            )
            ch_in = out_channels
        # Temporal attention before neuron attention: coarser (time) axis first,
        # then finer (neuron) axis. This ordering is intentional — the temporal
        # block contextualises each position across time before the neuron block
        # contextualises it across the neural population within each time frame.
        if use_temporal_attention:
            blocks.append(
                AttentionBlock(
                    out_channels,
                    num_heads=temp_num_attn_heads,
                    axis="temporal",
                    block_size=temp_block_size,
                    causal=temp_attn_causal,
                    use_ffn=temp_attn_use_ffn,
                    ffn_dim_mult=temp_attn_ffn_dim_mult,
                    use_rope=temp_attn_use_rope,
                )
            )
        if use_attention:
            blocks.append(
                AttentionBlock(
                    out_channels,
                    num_heads=num_attn_heads,
                    norm_type=norm_type,
                    num_groups=num_groups,
                    use_ffn=attn_use_ffn,
                    ffn_dim_mult=attn_ffn_dim_mult,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        # Downsample
        self.temporal_downsample = temporal_downsample
        if temporal_downsample:
            self.down_main = TemporalDownsample(
                out_channels,
                neuron_kernel_size=kn,
                temporal_kernel_size=down_temporal_kernel_size,
                temporal_stride=temporal_stride,
                causal=causal,
            )
        else:
            self.down_main = nn.Identity()

        # Stage-level shortcut (Wan2pt2-style: always present in encoder)
        if use_stage_shortcut:
            self.down_shortcut: nn.Module = AvgDown2D(
                in_channels,
                out_channels,
                t_stride=temporal_stride if temporal_downsample else 1,
                causal=causal,
            )

    def forward(self, x: Tensor, neuron_mask: Bool[Tensor, "b n"] | None = None) -> Tensor:
        """(B, C_in, T, N) → (B, C_out, T[//2], N)."""
        h = x
        for block in self.blocks:
            if isinstance(block, AttentionBlock):
                h = block(h, neuron_mask=neuron_mask)
            else:
                h = block(h)
        h = self.down_main(h)
        if self.use_stage_shortcut:
            h = h + self.down_shortcut(x)
        return h


class UpStage(nn.Module):
    """Decoder stage: residual blocks + optional attention + optional temporal upsample.

    Adapted from Wan2pt2's Up_ResidualBlock with DupUp shortcut.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        num_res_blocks: Number of residual blocks in this stage.
        use_attention: Whether to include an attention block.
        temporal_upsample: Whether to upsample T.
        neuron_kernel_size: Conv kernel size along N.
        temporal_kernel_size: Temporal kernel size for residual blocks.
        up_temporal_kernel_size: Temporal kernel size for the upsample conv.
        temporal_stride: Temporal stride for upsampling.
        num_attn_heads: Number of attention heads.
        use_temporal_attention: Whether to insert a temporal-axis AttentionBlock before the
            neuron-axis one.
        temp_block_size: Block-diagonal width for the temporal mask (0 = full sequence).
        temp_num_attn_heads: Number of heads for the temporal AttentionBlock.
        temp_attn_causal: Whether to apply within-block causal masking on the temporal
            AttentionBlock.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int = 2,
        use_attention: bool = False,
        use_temporal_attention: bool = False,
        temporal_upsample: bool = True,
        neuron_kernel_size: int = 1,
        temporal_kernel_size: int = 3,
        up_temporal_kernel_size: int = 3,
        temporal_stride: int = 2,
        num_attn_heads: int = 1,
        attn_use_ffn: bool = False,
        attn_ffn_dim_mult: int = 4,
        temp_block_size: int = 0,
        temp_num_attn_heads: int = 1,
        temp_attn_causal: bool = False,
        temp_attn_use_ffn: bool = False,
        temp_attn_ffn_dim_mult: int = 4,
        temp_attn_use_rope: bool = False,
        use_stage_shortcut: bool = True,
        causal: bool = True,
        dropout: float = 0.0,
        norm_type: Literal["rms", "group"] = "rms",
        num_groups: int = 32,
    ):
        super().__init__()
        kt = temporal_kernel_size
        kn = neuron_kernel_size
        self.use_stage_shortcut = use_stage_shortcut

        # Residual blocks (decoder gets one extra res block, like Wan2)
        blocks: list[nn.Module] = []
        ch_in = in_channels
        for _ in range(num_res_blocks + 1):
            blocks.append(
                ResidualBlock(
                    ch_in,
                    out_channels,
                    neuron_kernel_size=kn,
                    temporal_kernel_size=kt,
                    causal=causal,
                    dropout=dropout,
                    norm_type=norm_type,
                    num_groups=num_groups,
                )
            )
            ch_in = out_channels
        # Temporal attention before neuron attention: coarser (time) axis first,
        # then finer (neuron) axis. This ordering is intentional — the temporal
        # block contextualises each position across time before the neuron block
        # contextualises it across the neural population within each time frame.
        if use_temporal_attention:
            blocks.append(
                AttentionBlock(
                    out_channels,
                    num_heads=temp_num_attn_heads,
                    axis="temporal",
                    block_size=temp_block_size,
                    causal=temp_attn_causal,
                    use_ffn=temp_attn_use_ffn,
                    ffn_dim_mult=temp_attn_ffn_dim_mult,
                    use_rope=temp_attn_use_rope,
                )
            )
        if use_attention:
            blocks.append(
                AttentionBlock(
                    out_channels,
                    num_heads=num_attn_heads,
                    norm_type=norm_type,
                    num_groups=num_groups,
                    use_ffn=attn_use_ffn,
                    ffn_dim_mult=attn_ffn_dim_mult,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        # Upsample
        self.temporal_upsample = temporal_upsample
        if temporal_upsample:
            self.up_main = TemporalUpsample(
                out_channels,
                neuron_kernel_size=kn,
                temporal_kernel_size=up_temporal_kernel_size,
                temporal_stride=temporal_stride,
                causal=causal,
            )
        else:
            self.up_main = nn.Identity()

        # Stage-level shortcut (Wan2pt2-style: only when upsampling in decoder)
        self._has_shortcut = use_stage_shortcut and temporal_upsample
        if self._has_shortcut:
            self.up_shortcut = DupUp2D(in_channels, out_channels, t_stride=temporal_stride)

    def forward(self, x: Tensor, neuron_mask: Bool[Tensor, "b n"] | None = None) -> Tensor:
        """(B, C_in, T, N) → (B, C_out, T[*2], N)."""
        h = x
        for block in self.blocks:
            if isinstance(block, AttentionBlock):
                h = block(h, neuron_mask=neuron_mask)
            else:
                h = block(h)
        h = self.up_main(h)
        if self._has_shortcut:
            h = h + self.up_shortcut(x)
        return h


# ---------------------------------------------------------------------------
# Patchify / Unpatchify (temporal)
# ---------------------------------------------------------------------------


def temporal_patchify(x: Tensor, patch_size: int) -> Tensor:
    """Fold temporal bins into channels for free temporal compression.

    Adapted from Wan2pt2's spatial patchify.

    Args:
        x: (B, C, T, N) — T must be divisible by patch_size.
        patch_size: Number of time bins to fold into channels.

    Returns:
        (B, C * patch_size, T // patch_size, N)
    """
    b, c, t_full, n = x.shape
    t = t_full // patch_size
    # einops: "b c (t p) n -> b (c p) t n"
    # reshape splits T, permute moves p next to c, then merge c*p
    return (
        x.reshape(b, c, t, patch_size, n).permute(0, 1, 3, 2, 4).reshape(b, c * patch_size, t, n)
    )


def temporal_unpatchify(x: Tensor, patch_size: int) -> Tensor:
    """Unfold channels back into temporal bins.

    Args:
        x: (B, C * patch_size, T_comp, N)
        patch_size: Patch size used during patchification.

    Returns:
        (B, C, T_comp * patch_size, N)
    """
    b, cp, t, n = x.shape
    c = cp // patch_size
    # einops: "b (c p) t n -> b c (t p) n"
    return (
        x.reshape(b, c, patch_size, t, n).permute(0, 1, 3, 2, 4).reshape(b, c, t * patch_size, n)
    )
