"""Decoder for the WanNeural tokenizer.

Adapts Wan2pt2's Decoder3d to 2D (time x neurons) for neural data.

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
    CausalConvTranspose2d,
    ResidualBlock,
    UpStage,
    _make_norm,
)


class Decoder2d(nn.Module):
    """2D decoder for neural data, adapted from Wan2pt2's Decoder3d.

    Mirror of Encoder2d. Architecture:
        1. Initial CausalConv2d projection from latent
        2. Middle blocks (ResidualBlock + AttentionBlock + ResidualBlock)
        3. L upsampling stages (ResidualBlocks + optional Attention + optional TemporalUpsample)
        4. Head: RMS_norm → SiLU → output conv (undoing initial stride if needed)

    Args:
        out_channels: Output channels (1 for raw data, patch_size for patchified).
        z_dim: Latent channel dimension (input to decoder).
        dim: Base channel dimension (can differ from encoder's dim).
        dim_mult: Channel multipliers per level, reversed relative to encoder.
        num_res_blocks: Residual blocks per stage.
        attn_levels: Indices of levels that use attention (in encoder ordering).
        temporal_upsample: Per-level flags for temporal upsampling.
            Length must equal len(dim_mult) - 1.
        neuron_kernel_size: Conv kernel along N.
        temporal_kernel_size: Conv kernel along T for residual blocks, mid blocks, and head conv.
        up_temporal_kernel_size: Conv kernel along T for upsample convs.
        temporal_stride: Temporal stride for upsampling.
        k_t_init: Output conv temporal kernel size (mirrors encoder's initial conv).
        s_t_init: Output conv temporal stride (mirrors encoder's initial stride).
        num_attn_heads: Number of attention heads.
        use_stage_shortcut: Whether to use Wan2pt2-style stage-level shortcuts.
    """

    def __init__(
        self,
        out_channels: int,
        z_dim: int,
        dim: int,
        dim_mult: tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        attn_levels: tuple[int, ...] = (2,),
        temporal_upsample: tuple[bool, ...] = (True, True),
        neuron_kernel_size: int = 1,
        temporal_kernel_size: int = 3,
        up_temporal_kernel_size: int = 3,
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
        # temporal_stride is the upsample stride in decoder context
        s_t_up = temporal_stride

        assert len(temporal_upsample) == num_levels - 1, (
            f"temporal_upsample length ({len(temporal_upsample)}) must be "
            f"num_levels - 1 ({num_levels - 1})"
        )

        # --- Dimensions (matching Wan2pt2: [dim_mult[-1]] + dim_mult[::-1]) ---
        dims = [dim * u for u in [dim_mult[-1]] + list(reversed(dim_mult))]

        # --- Initial projection from latent ---
        self.conv_in = CausalConv2d(z_dim, dims[0], (kt, kn), causal=causal)

        # --- Temporal-attention block-size resolver ---
        def _temp_block_size_for(cum_stride: int) -> int:
            if bin_size_ms is None or token_bin_size_ms is None:
                return 0
            return int(round(token_bin_size_ms / (bin_size_ms * cum_stride)))

        # Cumulative stride at the deepest layer (decoder mid input)
        cum_stride = temporal_patch_size * s_t_init * (s_t_up ** sum(temporal_upsample))

        # --- Middle blocks ---
        self.mid_block1 = ResidualBlock(
            dims[0],
            dims[0],
            neuron_kernel_size=kn,
            temporal_kernel_size=kt,
            causal=causal,
            dropout=dropout,
            norm_type=norm_type,
            num_groups=num_groups,
        )
        self.mid_temp_attn: nn.Module = (
            AttentionBlock(
                dims[0],
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
                dims[0],
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
            dims[0],
            dims[0],
            neuron_kernel_size=kn,
            temporal_kernel_size=kt,
            causal=causal,
            dropout=dropout,
            norm_type=norm_type,
            num_groups=num_groups,
        )

        # --- Upsampling stages (reversed order relative to encoder) ---
        self.stages = nn.ModuleList()
        for i, (in_ch, out_ch) in enumerate(zip(dims[:-1], dims[1:], strict=False)):
            do_upsample = temporal_upsample[i] if i < len(temporal_upsample) else False
            enc_level = num_levels - 1 - i
            use_attn = enc_level in attn_levels
            use_temp = enc_level in temp_attn_levels

            self.stages.append(
                UpStage(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    num_res_blocks=num_res_blocks,
                    use_attention=use_attn,
                    use_temporal_attention=use_temp,
                    temporal_upsample=do_upsample,
                    neuron_kernel_size=kn,
                    temporal_kernel_size=kt,
                    up_temporal_kernel_size=up_temporal_kernel_size,
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
            if do_upsample:
                cum_stride //= s_t_up

        # --- Head ---
        head_in = dims[-1]
        self.norm_out = _make_norm(head_in, norm_type, num_groups)
        self.act_out = nn.SiLU()

        if s_t_init > 1:
            self.conv_out = CausalConvTranspose2d(
                head_in,
                out_channels,
                kernel_size=(k_t_init, kn if kn > 1 else 1),
                stride=(s_t_init, 1),
                causal=causal,
            )
        else:
            self.conv_out = CausalConv2d(
                head_in,
                out_channels,
                (k_t_init, kn),
                causal=causal,
            )

    def forward(
        self,
        z: Tensor,
        neuron_mask: Bool[Tensor, "b n"] | None = None,
    ) -> Tensor:
        """Decode latent to reconstruction.

        Args:
            z: (B, z_dim, T_comp, N)
            neuron_mask: Optional (B, N) bool mask, True = valid neuron.

        Returns:
            (B, out_channels, T, N) — should match original input dimensions.
        """
        h = self.conv_in(z)

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

        for stage in self.stages:
            h = stage(h, neuron_mask=neuron_mask)

        h = self.conv_out(self.act_out(self.norm_out(h)))
        return h
