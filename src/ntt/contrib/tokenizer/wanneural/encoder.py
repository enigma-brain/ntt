"""Encoder for the WanNeural tokenizer.

Adapts Wan2pt2's Encoder3d to 2D (time x neurons) for neural data.

Adapted from the Wan2.1/Wan2.2 video VAE, used under the Apache-2.0 license:
https://github.com/Wan-Video/Wan2.2 (see ``LICENSES/wan-Apache-2.0.txt``).
"""

from __future__ import annotations

from typing import Literal

from jaxtyping import Bool
from torch import Tensor, nn

from .blocks import (
    AttentionBlock,
    CausalConv2d,
    DownStage,
    ResidualBlock,
    _make_norm,
)


class Encoder2d(nn.Module):
    """2D encoder for neural data, adapted from Wan2pt2's Encoder3d.

    Architecture:
        1. Initial CausalConv2d projection
        2. L downsampling stages (ResidualBlocks + optional Attention + optional
           TemporalDownsample)
        3. Middle blocks (ResidualBlock + AttentionBlock + ResidualBlock)
        4. Head: RMS_norm → SiLU → CausalConv2d → output

    Args:
        in_channels: Input channels (1 for raw data, patch_size for patchified).
        dim: Base channel dimension.
        head_dim: Output channels (z_dim*2 for VAE, z_dim for FSQ).
        dim_mult: Channel multipliers per level (e.g., (1, 2, 4)).
        num_res_blocks: Residual blocks per stage.
        attn_levels: Indices of levels that use attention.
        temporal_downsample: Per-level flags for temporal downsampling.
            Length must equal len(dim_mult) - 1.
        neuron_kernel_size: Conv kernel along N (1 = per-neuron).
        temporal_kernel_size: Conv kernel along T for residual blocks, mid blocks, and head conv.
        down_temporal_kernel_size: Conv kernel along T for downsample convs.
        temporal_stride: Temporal stride for downsampling.
        k_t_init: Initial conv temporal kernel size.
        s_t_init: Initial conv temporal stride.
        num_attn_heads: Number of attention heads.
        use_stage_shortcut: Whether to use Wan2pt2-style stage-level shortcuts.
    """

    def __init__(
        self,
        in_channels: int,
        dim: int,
        head_dim: int,
        dim_mult: tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        attn_levels: tuple[int, ...] = (2,),
        temporal_downsample: tuple[bool, ...] = (True, True),
        neuron_kernel_size: int = 1,
        temporal_kernel_size: int = 3,
        down_temporal_kernel_size: int = 3,
        temporal_stride: int = 2,
        k_t_init: int = 3,
        s_t_init: int = 1,
        num_attn_heads: int = 1,
        attn_use_ffn: bool = False,
        attn_ffn_dim_mult: int = 4,
        use_stage_shortcut: bool = True,
        use_mid_attn: bool = True,
        causal: bool = True,
        dropout: float = 0.0,
        norm_type: Literal["rms", "group"] = "rms",
        num_groups: int = 32,
        # --- Temporal attention ---
        temp_attn_levels: tuple[int, ...] = (),
        temp_num_attn_heads: int = 1,
        use_mid_temp_attn: bool = False,
        temp_attn_causal: bool = False,
        temp_attn_use_ffn: bool = False,
        temp_attn_ffn_dim_mult: int = 4,
        temp_attn_use_rope: bool = False,
        bin_size_ms: float | None = None,
        token_bin_size_ms: float | None = None,
        temporal_patch_size: int = 1,
    ):
        super().__init__()
        kt = temporal_kernel_size
        kn = neuron_kernel_size
        num_levels = len(dim_mult)

        assert len(temporal_downsample) == num_levels - 1, (
            f"temporal_downsample length ({len(temporal_downsample)}) must be "
            f"num_levels - 1 ({num_levels - 1})"
        )

        # --- Dimensions (matching Wan2pt2: [1] + dim_mult) ---
        dims = [dim * u for u in [1] + list(dim_mult)]

        # --- Initial projection ---
        self.conv_in = CausalConv2d(
            in_channels,
            dims[0],
            (k_t_init, kn),
            stride=(s_t_init, 1),
            causal=causal,
        )

        # --- Temporal-attention block-size resolver (returns 0 if disabled) ---
        def _temp_block_size_for(cum_stride: int) -> int:
            if bin_size_ms is None or token_bin_size_ms is None:
                return 0
            return int(round(token_bin_size_ms / (bin_size_ms * cum_stride)))

        # --- Downsampling stages (track cum_stride) ---
        self.stages = nn.ModuleList()
        cum_stride = temporal_patch_size * s_t_init
        for i, (in_ch, out_ch) in enumerate(zip(dims[:-1], dims[1:], strict=False)):
            do_downsample = temporal_downsample[i] if i < len(temporal_downsample) else False
            use_attn = i in attn_levels
            use_temp = i in temp_attn_levels

            self.stages.append(
                DownStage(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    num_res_blocks=num_res_blocks,
                    use_attention=use_attn,
                    use_temporal_attention=use_temp,
                    temporal_downsample=do_downsample,
                    neuron_kernel_size=kn,
                    temporal_kernel_size=kt,
                    down_temporal_kernel_size=down_temporal_kernel_size,
                    temporal_stride=temporal_stride,
                    num_attn_heads=num_attn_heads,
                    attn_use_ffn=attn_use_ffn,
                    attn_ffn_dim_mult=attn_ffn_dim_mult,
                    temp_block_size=_temp_block_size_for(cum_stride) if use_temp else 0,
                    temp_num_attn_heads=temp_num_attn_heads,
                    temp_attn_causal=temp_attn_causal,
                    temp_attn_use_ffn=temp_attn_use_ffn,
                    temp_attn_ffn_dim_mult=temp_attn_ffn_dim_mult,
                    temp_attn_use_rope=temp_attn_use_rope,
                    use_stage_shortcut=use_stage_shortcut,
                    causal=causal,
                    dropout=dropout,
                    norm_type=norm_type,
                    num_groups=num_groups,
                )
            )
            if do_downsample:
                cum_stride *= temporal_stride

        # --- Middle blocks ---
        final_dim = dims[-1]
        self.mid_block1 = ResidualBlock(
            final_dim,
            final_dim,
            neuron_kernel_size=kn,
            temporal_kernel_size=kt,
            causal=causal,
            dropout=dropout,
            norm_type=norm_type,
            num_groups=num_groups,
        )
        self.mid_temp_attn: nn.Module = (
            AttentionBlock(
                final_dim,
                num_heads=temp_num_attn_heads,
                axis="temporal",
                block_size=_temp_block_size_for(cum_stride),
                causal=temp_attn_causal,
                use_ffn=temp_attn_use_ffn,
                ffn_dim_mult=temp_attn_ffn_dim_mult,
                use_rope=temp_attn_use_rope,
            )
            if use_mid_temp_attn
            else nn.Identity()
        )
        self.mid_attn = (
            AttentionBlock(
                final_dim,
                num_heads=num_attn_heads,
                norm_type=norm_type,
                num_groups=num_groups,
                use_ffn=attn_use_ffn,
                ffn_dim_mult=attn_ffn_dim_mult,
            )
            if use_mid_attn
            else nn.Identity()
        )
        self.mid_block2 = ResidualBlock(
            final_dim,
            final_dim,
            neuron_kernel_size=kn,
            temporal_kernel_size=kt,
            causal=causal,
            dropout=dropout,
            norm_type=norm_type,
            num_groups=num_groups,
        )

        # --- Head ---
        self.norm_out = _make_norm(final_dim, norm_type, num_groups)
        self.act_out = nn.SiLU()
        self.conv_out = CausalConv2d(final_dim, head_dim, (kt, kn), causal=causal)

    def forward(
        self,
        x: Tensor,
        neuron_mask: Bool[Tensor, "b n"] | None = None,
    ) -> Tensor:
        """Encode input to latent representation.

        Args:
            x: (B, in_channels, T, N)
            neuron_mask: Optional (B, N) bool mask, True = valid neuron.

        Returns:
            (B, head_dim, T_comp, N)
        """
        h = self.conv_in(x)

        for stage in self.stages:
            h = stage(h, neuron_mask=neuron_mask)

        h = self.mid_block1(h)
        h = (
            self.mid_temp_attn(h, neuron_mask=neuron_mask)
            if isinstance(self.mid_temp_attn, AttentionBlock)
            else self.mid_temp_attn(h)
        )
        h = (
            self.mid_attn(h, neuron_mask=neuron_mask)
            if isinstance(self.mid_attn, AttentionBlock)
            else self.mid_attn(h)
        )
        h = self.mid_block2(h)

        h = self.conv_out(self.act_out(self.norm_out(h)))
        return h
