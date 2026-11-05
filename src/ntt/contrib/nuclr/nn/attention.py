"""Cross-attention and self-attention with rotary positional embeddings.

Uses ``torch.nn.attention.flex_attention`` with a ``BlockMask`` to enforce
per-segment attention isolation. The ``BlockMask`` is constructed by
``ntt.contrib.nuclr.nn._block_mask.build_block_diagonal_mask`` at the
encoder callsite and passed in via the ``block_mask`` kwarg.

``flex_attention`` is wrapped with ``torch.compile(dynamic=True)`` so the
Triton-fused kernel is used instead of the unfused fallback. ``dynamic=True``
keeps a single compilation across the variable segment lengths we see at
runtime; without it, every new shape triggers a recompile.

The attention modules are adapted from the NuCLR reference implementation
(rewritten here on ``flex_attention``), used under the Apache-2.0 license:
https://github.com/nerdslab/nuclr (Arora et al., 2025, NeurIPS;
https://arxiv.org/abs/2512.01199).
"""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.attention.flex_attention import BlockMask
from torch.nn.attention.flex_attention import flex_attention as _flex_attention

from .rotary_embedding import apply_rotary_pos_emb, invert_rotary_pos_emb

flex_attention = torch.compile(_flex_attention, dynamic=True)


class RotaryCrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        ctx_dim: int | None = None,
        heads: int = 8,
        dim_head: int | None = None,
        atn_dropout: float = 0.0,
        rotate_value: bool = False,
        to_kv_bias: bool = True,
        to_q_bias: bool = True,
        to_out_bias: bool = True,
        pre_norm_q: bool = True,
        pre_norm_kv: bool = True,
    ):
        super().__init__()
        self.heads = heads
        self.atn_dropout = atn_dropout
        self.rotate_value = rotate_value
        dim_head = dim_head or (dim // heads)

        ctx_dim = ctx_dim or dim
        self.norm = nn.LayerNorm(dim) if pre_norm_q else nn.Identity()
        self.norm_context = nn.LayerNorm(ctx_dim) if pre_norm_kv else nn.Identity()

        inner_dim = dim_head * heads
        self.to_q = nn.Linear(dim, inner_dim, bias=to_q_bias)
        self.to_kv = nn.Linear(ctx_dim, inner_dim * 2, bias=to_kv_bias)
        self.to_out = nn.Linear(inner_dim, dim, bias=to_out_bias)

    def forward(
        self,
        x_q: torch.Tensor,
        x_ctx: torch.Tensor,
        rotary_q: torch.Tensor | None = None,
        rotary_ctx: torch.Tensor | None = None,
        block_mask: BlockMask | None = None,
    ) -> torch.Tensor:
        q = self.to_q(self.norm(x_q))
        k, v = self.to_kv(self.norm_context(x_ctx)).chunk(2, dim=-1)

        if q.ndim == 2:
            q = rearrange(q, "n (h d) -> 1 n h d", h=self.heads)
            k = rearrange(k, "n (h d) -> 1 n h d", h=self.heads)
            v = rearrange(v, "n (h d) -> 1 n h d", h=self.heads)
            batched_input = False
        elif q.ndim == 3:
            q = rearrange(q, "b n (h d) -> b n h d", h=self.heads)
            k = rearrange(k, "b n (h d) -> b n h d", h=self.heads)
            v = rearrange(v, "b n (h d) -> b n h d", h=self.heads)
            batched_input = True
        else:
            raise ValueError("Unknown input format")

        if rotary_q is not None:
            assert rotary_ctx is not None
            if not batched_input:
                assert rotary_q.ndim == 2
                assert rotary_ctx.ndim == 2
                rotary_q = rotary_q[None, :, :]
                rotary_ctx = rotary_ctx[None, :, :]
            q = apply_rotary_pos_emb(rotary_q, q)
            k = apply_rotary_pos_emb(rotary_ctx, k)
            if self.rotate_value:
                v = apply_rotary_pos_emb(rotary_ctx, v)

        # flex_attention expects (B, H, S, D).
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        out = cast(torch.Tensor, flex_attention(q, k, v, block_mask=block_mask))
        out = out.transpose(1, 2)

        if self.training and self.atn_dropout > 0.0:
            out = F.dropout(out, p=self.atn_dropout)

        if rotary_ctx is not None and self.rotate_value:
            assert rotary_q is not None
            out = apply_rotary_pos_emb(invert_rotary_pos_emb(rotary_q), out)

        if batched_input:
            out = rearrange(out, "b n h d -> b n (h d)")
        else:
            out = rearrange(out, "b n h d -> (b n) (h d)", b=1)

        out = self.to_out(out)
        return out


class RotarySelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int | None = None,
        atn_dropout: float = 0.0,
        rotate_value: bool = False,
        to_qkv_bias: bool = True,
        to_out_bias: bool = True,
        pre_norm: bool = True,
    ):
        super().__init__()
        self.heads = heads
        self.atn_dropout = atn_dropout
        self.rotate_value = rotate_value
        dim_head = dim_head or (dim // heads)

        inner_dim = dim_head * heads
        self.norm = nn.LayerNorm(dim) if pre_norm else nn.Identity()
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=to_qkv_bias)
        self.to_out = nn.Linear(inner_dim, dim, bias=to_out_bias)

    def forward(
        self,
        x: torch.Tensor,
        rotary: torch.Tensor | None = None,
        block_mask: BlockMask | None = None,
    ) -> torch.Tensor:
        q, k, v = self.to_qkv(self.norm(x)).chunk(3, dim=-1)

        if q.ndim == 2:
            q = rearrange(q, "n (h d) -> 1 n h d", h=self.heads)
            k = rearrange(k, "n (h d) -> 1 n h d", h=self.heads)
            v = rearrange(v, "n (h d) -> 1 n h d", h=self.heads)
            batched_input = False
        elif q.ndim == 3:
            q = rearrange(q, "b n (h d) -> b n h d", h=self.heads)
            k = rearrange(k, "b n (h d) -> b n h d", h=self.heads)
            v = rearrange(v, "b n (h d) -> b n h d", h=self.heads)
            batched_input = True
        else:
            raise ValueError("Unknown input format")

        if rotary is not None:
            if not batched_input:
                assert rotary.ndim == 2
                rotary = rotary[None, :, :]
            q = apply_rotary_pos_emb(rotary, q)
            k = apply_rotary_pos_emb(rotary, k)
            if self.rotate_value:
                v = apply_rotary_pos_emb(rotary, v)

        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()

        out = cast(torch.Tensor, flex_attention(q, k, v, block_mask=block_mask))
        out = out.transpose(1, 2)

        if self.training and self.atn_dropout > 0.0:
            out = F.dropout(out, p=self.atn_dropout)

        if rotary is not None and self.rotate_value:
            out = apply_rotary_pos_emb(invert_rotary_pos_emb(rotary), out)

        if batched_input:
            out = rearrange(out, "b n h d -> b n (h d)")
        else:
            out = rearrange(out, "b n h d -> (b n) (h d)", b=1)

        out = self.to_out(out)
        return out
