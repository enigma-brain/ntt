"""Single-stream perceiver encoder for neuron identity learning.

Direct port of ``NuclrV2gaSpikes`` from ``nuclr/src/models/nuclr_v2ga_spikes.py``.
Produces per-neuron identity embeddings by mean-pooling latents over time.

Architecture:
    1. Input cross-attention (spikes → latents)
    2. ``t_layers`` temporal blocks: [self_attn, cross_attn, FFN]
    3. ``st_layers`` spatio-temporal blocks: [spatial_attn, spatial_cross_attn,
       spatial_FFN, temporal_attn, temporal_cross_attn, temporal_FFN]
    4. Mean pool over latent dim → (total_units, dim)

Imports shared NN modules from ``ntt.contrib.nuclr.nn``.

Adapted from the NuCLR reference implementation, used under the Apache-2.0
license: https://github.com/nerdslab/nuclr (Arora et al., 2025, NeurIPS;
https://arxiv.org/abs/2512.01199).
"""

from collections.abc import Iterable
from typing import cast

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import Tensor

from ntt.contrib.nuclr.nn._block_mask import build_block_diagonal_mask

from .nn import (
    FFN,
    RotaryCrossAttention,
    RotaryEmbedding,
    RotarySelfAttention,
)


class NuclrEncoder(nn.Module):
    """Single-stream perceiver encoder for neural spike data.

    Takes flat spike timestamps (sorted by unit) and produces per-neuron
    identity embeddings via mean-pooling over temporal latent positions.

    Args:
        ctx_duration: Context window duration in seconds.
        latent_step: Spacing between latent positions in seconds.
        dim: Model dimension.
        cross_heads: Number of heads for cross-attention.
        self_heads: Number of heads for self-attention.
        dim_head: Dimension per attention head.
        atn_dropout: Attention dropout probability.
        lin_dropout: Linear/FFN dropout probability.
        t_layers: Number of temporal-only attention layers.
        st_layers: Number of spatio-temporal attention layers.
        rot_ratio: Fraction of dim_head used for rotary encoding.
    """

    def __init__(
        self,
        ctx_duration: float = 10.0,
        latent_step: float = 1.0,
        dim: int = 256,
        cross_heads: int = 4,
        self_heads: int = 4,
        dim_head: int = 64,
        atn_dropout: float = 0.0,
        lin_dropout: float = 0.2,
        t_layers: int = 2,
        st_layers: int = 2,
        rot_ratio: float = 0.5,
    ):
        super().__init__()

        self.ctx_duration = ctx_duration
        self.latent_step = latent_step
        self.emb_dim = self.dim = dim

        t_min, t_max = 1e-4, 8 * ctx_duration

        self.rotary_emb = RotaryEmbedding(
            head_dim=dim_head,
            rotate_dim=int(dim_head * rot_ratio),
            t_min=t_min,
            t_max=t_max,
        )

        self.num_latents = num_latents = int(np.ceil(ctx_duration / latent_step))
        latent_times = (torch.arange(num_latents) + 0.5) * latent_step
        self.register_buffer("latent_times", latent_times)

        # Input cross-attention (spikes → latents)
        self.cross_attn = RotaryCrossAttention(
            dim=dim,
            heads=cross_heads,
            dim_head=dim_head,
            rotate_value=True,
            pre_norm_q=False,
            pre_norm_kv=False,
            atn_dropout=atn_dropout,
        )
        self.cross_ffn = FFN(dim=dim, mult=4, dropout=lin_dropout, pre_norm=True)

        # Temporal attention layers
        self.t_blocks: Iterable[Iterable[nn.Module]] = cast(
            "Iterable[Iterable[nn.Module]]",
            nn.ModuleList(
                [
                    nn.ModuleList(
                        [
                            RotarySelfAttention(
                                dim,
                                self_heads,
                                dim_head,
                                atn_dropout,
                                rotate_value=True,
                            ),
                            RotaryCrossAttention(
                                dim=dim,
                                heads=cross_heads,
                                dim_head=dim_head,
                                atn_dropout=atn_dropout,
                                rotate_value=True,
                                pre_norm_kv=False,
                            ),
                            FFN(dim=dim, mult=4, dropout=lin_dropout, pre_norm=True),
                        ]
                    )
                    for _ in range(t_layers)
                ]
            ),
        )

        # Spatio-temporal attention layers
        # NOTE: Modules are registered temporal first, spatial second, even though
        # execution order is spatial-then-temporal. The registration order fixes
        # the state_dict keys of NuCLR contrastive checkpoints.
        self.st_blocks: Iterable[Iterable[nn.Module]] = cast(
            "Iterable[Iterable[nn.Module]]",
            nn.ModuleList(
                [
                    nn.ModuleList(
                        [
                            RotarySelfAttention(
                                dim,
                                self_heads,
                                dim_head,
                                atn_dropout,
                                rotate_value=True,
                            ),
                            RotaryCrossAttention(
                                dim=dim,
                                heads=cross_heads,
                                dim_head=dim_head,
                                atn_dropout=atn_dropout,
                                rotate_value=True,
                                pre_norm_kv=False,
                            ),
                            FFN(dim=dim, mult=4, dropout=lin_dropout, pre_norm=True),
                            RotarySelfAttention(
                                dim,
                                self_heads,
                                dim_head,
                                atn_dropout,
                                rotate_value=True,
                            ),
                            RotaryCrossAttention(
                                dim=dim,
                                heads=cross_heads,
                                dim_head=dim_head,
                                atn_dropout=atn_dropout,
                                rotate_value=True,
                                pre_norm_kv=False,
                            ),
                            FFN(dim=dim, mult=4, dropout=lin_dropout, pre_norm=True),
                        ]
                    )
                    for _ in range(st_layers)
                ]
            ),
        )

        self.dp = nn.Dropout(lin_dropout)

    def forward(
        self,
        spikes: Tensor,
        spike_seqlen: Tensor,
        unit_seqlen: Tensor,
    ) -> Tensor:
        """Encode spike data into per-neuron identity embeddings.

        Args:
            spikes: Flat spike timestamps sorted by unit. Shape ``(total_spikes,)``.
            spike_seqlen: Number of spikes per unit. Shape ``(total_units,)``.
            unit_seqlen: Number of units per batch element. Shape ``(batch,)``.

        Returns:
            Per-neuron identity embeddings ``(total_units, dim)``.
        """
        device = spikes.device
        # AMP autocast ignores float64; cast to float32 so bf16-mixed works.
        spikes = spikes.float()
        num_units = len(spike_seqlen)
        temp_latent_seqlen = [self.num_latents for _ in range(num_units)]
        spax_latent_seqlen = unit_seqlen.tolist() * self.num_latents

        cross_attn_bias = build_block_diagonal_mask(
            q_seqlen=temp_latent_seqlen,
            kv_seqlen=spike_seqlen.tolist(),
            device=device,
        )
        spax_attn_bias = build_block_diagonal_mask(
            q_seqlen=spax_latent_seqlen,
            device=device,
        )
        temp_attn_bias = build_block_diagonal_mask(
            q_seqlen=temp_latent_seqlen,
            device=device,
        )

        with torch.autocast(device_type="cuda", enabled=False):
            ctx_rot = self.rotary_emb(spikes)
            x_rot = self.rotary_emb(self.latent_times)

        ctx = spikes.new_zeros((len(spikes), self.dim))
        x = spikes.new_zeros((num_units * self.num_latents, self.dim))  # (U L) D
        x_rot = repeat(x_rot, "L H -> (U L) H", U=num_units)

        # Input cross attention
        x = x + self.cross_attn(x, ctx, x_rot, ctx_rot, cross_attn_bias)
        x = x + self.dp(self.cross_ffn(x))

        # Temporal attention layers
        for t_attn, tc_attn, t_ffn in self.t_blocks:
            x = x + self.dp(t_attn(x=x, rotary=x_rot, block_mask=temp_attn_bias))
            x = x + self.dp(tc_attn(x, ctx, x_rot, ctx_rot, cross_attn_bias))
            x = x + self.dp(t_ffn(x))

        # Spatio-temporal attention layers
        # Unpacking matches original: positions 0-2 = temporal, 3-5 = spatial
        for t_attn, tc_attn, t_ffn, s_attn, sc_attn, s_ffn in self.st_blocks:
            # Spatial
            x = rearrange(x, "(U L) D -> (L U) D", L=self.num_latents, U=num_units)
            x = x + self.dp(s_attn(x, block_mask=spax_attn_bias))
            x = rearrange(x, "(L U) D -> (U L) D", U=num_units, L=self.num_latents)
            x = x + self.dp(sc_attn(x, ctx, x_rot, ctx_rot, cross_attn_bias))
            x = x + self.dp(s_ffn(x))

            # Temporal
            x = x + self.dp(t_attn(x=x, rotary=x_rot, block_mask=temp_attn_bias))
            x = x + self.dp(tc_attn(x, ctx, x_rot, ctx_rot, cross_attn_bias))
            x = x + self.dp(t_ffn(x))

        x = rearrange(x, "(U L) D -> U L D", U=num_units, L=self.num_latents)
        y = x.mean(1)
        return y
