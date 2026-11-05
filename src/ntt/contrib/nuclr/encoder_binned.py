"""Binned encoder for neuron identity learning.

Direct port of ``NuclrV2gaBin`` from ``nuclr/src/models/nuclr_v2ga_bin.py``.
Produces per-neuron identity embeddings by mean-pooling latents over time.

Unlike the spike-based encoder, this operates on binned spike counts and uses
only self-attention (no cross-attention to raw spike times).

Architecture:
    1. Linear read-in: bins_per_latent → dim
    2. ``t_layers`` temporal blocks: [self_attn, FFN]
    3. ``st_layers`` spatio-temporal blocks: [spatial_attn, spatial_FFN,
       temporal_attn, temporal_FFN]
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
from einops import rearrange
from torch import Tensor

from ntt.contrib.nuclr.nn._block_mask import build_block_diagonal_mask

from .nn import FFN, RotaryEmbedding, RotarySelfAttention


class NuclrBinnedEncoder(nn.Module):
    """Binned encoder for neural spike data.

    Takes binned spike counts (reshaped as (U*L, bins_per_latent)) and produces
    per-neuron identity embeddings via mean-pooling over temporal latent positions.

    Args:
        ctx_duration: Context window duration in seconds.
        latent_step: Spacing between latent positions in seconds.
        bin_size: Bin size in seconds (must evenly divide latent_step).
        dim: Model dimension.
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
        bin_size: float = 0.02,
        dim: int = 256,
        self_heads: int = 4,
        dim_head: int = 64,
        atn_dropout: float = 0.0,
        lin_dropout: float = 0.2,
        t_layers: int = 2,
        st_layers: int = 2,
        rot_ratio: float = 0.5,
    ):
        super().__init__()

        assert abs(latent_step / bin_size - round(latent_step / bin_size)) < 1e-6, (
            f"latent_step ({latent_step}) must be divisible by bin_size ({bin_size})"
        )
        assert abs(ctx_duration / latent_step - round(ctx_duration / latent_step)) < 1e-6, (
            f"ctx_duration ({ctx_duration}) must be divisible by latent_step ({latent_step})"
        )

        self.ctx_duration = ctx_duration
        self.latent_step = latent_step
        self.bin_size = bin_size
        self.emb_dim = self.dim = dim

        # Derived params
        self.bins_per_latent = int(round(latent_step / bin_size))
        self.num_latents = int(np.ceil(ctx_duration / latent_step))

        t_min, t_max = 1.0, 8.0 * self.num_latents
        self.rotary_emb = RotaryEmbedding(
            head_dim=dim_head,
            rotate_dim=int(dim_head * rot_ratio),
            t_min=t_min,
            t_max=t_max,
        )

        self.read_in = nn.Linear(self.bins_per_latent, dim)

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
                            FFN(dim=dim, mult=4, dropout=lin_dropout, pre_norm=True),
                        ]
                    )
                    for _ in range(t_layers)
                ]
            ),
        )

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
                            FFN(dim=dim, mult=4, dropout=lin_dropout, pre_norm=True),
                            RotarySelfAttention(
                                dim,
                                self_heads,
                                dim_head,
                                atn_dropout,
                                rotate_value=True,
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
        bins: Tensor,
        unit_seqlen: Tensor,
    ) -> Tensor:
        """Encode binned spike data into per-neuron identity embeddings.

        Args:
            bins: Binned spike counts, shape ``(total_units * num_latents, bins_per_latent)``.
            unit_seqlen: Number of units per batch element, shape ``(batch,)``.

        Returns:
            Per-neuron identity embeddings ``(total_units, dim)``.
        """
        device = bins.device
        num_units = int(unit_seqlen.sum())

        spax_attn_bias = build_block_diagonal_mask(
            q_seqlen=unit_seqlen.tolist() * self.num_latents,
            device=device,
        )

        x = self.read_in(bins.float())
        x = rearrange(x, "(U L) D -> U L D", U=num_units, L=self.num_latents)
        t = torch.arange(self.num_latents, device=device).repeat(num_units, 1).float()
        x_rot = self.rotary_emb(t)

        # Temporal attention layers
        for t_attn, t_ffn in self.t_blocks:
            x = x + self.dp(t_attn(x=x, rotary=x_rot))
            x = x + self.dp(t_ffn(x))

        # Spatio-temporal attention layers
        for t_attn, t_ffn, s_attn, s_ffn in self.st_blocks:
            # Spatial
            x = rearrange(x, "U L D -> (L U) D", L=self.num_latents, U=num_units)
            x = x + self.dp(s_attn(x, block_mask=spax_attn_bias))
            x = x + self.dp(s_ffn(x))
            x = rearrange(x, "(L U) D -> U L D", U=num_units, L=self.num_latents)

            # Temporal
            x = x + self.dp(t_attn(x=x, rotary=x_rot))
            x = x + self.dp(t_ffn(x))

        y = x.mean(1)
        return y
