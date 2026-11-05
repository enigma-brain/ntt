"""CALM Encoder: Transformer encoder with RoPE and causal masking.

Takes per-neuron traces [B*N, T, 1] and produces encoded features [B*N, T_d, C]
via patchification followed by a stack of Transformer blocks.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .rope import RotaryPositionalEmbedding, apply_rotary_emb


class RoPEMultiheadAttention(nn.Module):
    """Multi-head self-attention with Rotary Positional Embeddings.

    Args:
        embed_dim: Total embedding dimension.
        num_heads: Number of attention heads.
        dropout: Attention-weights dropout probability (active in training).
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.rope = RotaryPositionalEmbedding(self.head_dim)

    def forward(
        self,
        x: Tensor,
        is_causal: bool = False,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, T, C).
            is_causal: If True, apply causal masking (enables FlashAttention
                when no key-padding mask is supplied).
            key_padding_mask: Optional (B, T) bool mask where True marks valid
                keys. When supplied, the causal-only FlashAttention path is
                disabled and an explicit combined mask is built.

        Returns:
            Output tensor of shape (B, T, C).
        """
        B, T, C = x.shape
        H, D = self.num_heads, self.head_dim

        qkv = self.qkv(x)  # (B, T, 3*C)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B, T, C)

        q = q.view(B, T, H, D).transpose(1, 2)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)

        cos, sin = self.rope(T, device=x.device)
        q, k = apply_rotary_emb(q, k, cos, sin)

        dropout_p = self.dropout if self.training else 0.0

        if key_padding_mask is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal, dropout_p=dropout_p)
        else:
            # Combine causal (if any) with key-padding into a single bool mask
            # of shape (B, 1, T, T) — True = attend. sdpa forbids passing
            # is_causal=True together with attn_mask, so we merge them here.
            key_mask = key_padding_mask[:, None, :]  # (B, 1, T)
            if is_causal:
                causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
                combined = causal[None, :, :] & key_mask  # (B, T, T)
            else:
                combined = key_mask.expand(B, T, T)
            attn_mask = combined.unsqueeze(1)  # (B, 1, T, T)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block.

    Args:
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        dropout: Dropout probability used at three sites: attention weights
            (inside ``RoPEMultiheadAttention``), post-attention output, and
            post-FFN output. All three share the same ``p``.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = RoPEMultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.attn_drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.ffn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: Tensor,
        is_causal: bool = False,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward pass with pre-norm residual connections.

        Args:
            x: Input tensor of shape (B, T, C).
            is_causal: If True, apply causal masking.
            key_padding_mask: Optional (B, T) bool mask, True = valid key.

        Returns:
            Output tensor of shape (B, T, C).
        """
        x = x + self.attn_drop(
            self.attn(self.norm1(x), is_causal=is_causal, key_padding_mask=key_padding_mask)
        )
        x = x + self.ffn_drop(self.ffn(self.norm2(x)))
        return x


class PopulationAttention(nn.Module):
    """Multi-head self-attention over the neuron (N) axis.

    No positional encoding (neurons are an unordered set) and no causal mask.
    Accepts a boolean attention mask ``attn_mask: [B, N, N]`` where True means
    "query may attend to this key" (PyTorch sdpa convention).
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: (B_flat, N, C) — neurons along dim 1.
            attn_mask: (B_flat, N, N) bool — True = allow attend.

        Returns:
            (B_flat, N, C) — same shape.
        """
        B_flat, N, C = x.shape
        H, D = self.num_heads, self.head_dim

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B_flat, N, H, D).transpose(1, 2)
        k = k.view(B_flat, N, H, D).transpose(1, 2)
        v = v.view(B_flat, N, H, D).transpose(1, 2)

        # sdpa bool mask: True = attend (PyTorch >= 2.1). Unsqueeze head axis to broadcast.
        mask_4d = attn_mask.unsqueeze(1)  # (B_flat, 1, N, N)
        dropout_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask_4d, dropout_p=dropout_p)

        out = out.transpose(1, 2).contiguous().view(B_flat, N, C)
        return self.out_proj(out)


class PopulationBlock(nn.Module):
    """Pre-norm Transformer block over the neuron axis.

    Input shape: ``[B, T_d, N, C]``. Internally flattens the leading ``B * T_d``
    into a single batch and applies self-attention across N. No positional
    encoding.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = PopulationAttention(embed_dim, num_heads, dropout=dropout)
        self.attn_drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        self.ffn_drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: (B, T_d, N, C).
            attn_mask: (B, N, N) bool — broadcast across T_d.

        Returns:
            (B, T_d, N, C).
        """
        B, T_d, N, C = x.shape
        # Flatten B and T_d so each time step is an independent "population".
        h = x.reshape(B * T_d, N, C)
        # Broadcast attention mask across T_d.
        mask = attn_mask.unsqueeze(1).expand(B, T_d, N, N).reshape(B * T_d, N, N)

        h = h + self.attn_drop(self.attn(self.norm1(h), attn_mask=mask))
        h = h + self.ffn_drop(self.ffn(self.norm2(h)))
        return h.reshape(B, T_d, N, C)


class CalmEncoder(nn.Module):
    """CALM Encoder: patchify + temporal Transformer stack + (optional)
    population stack.

    The temporal stack operates on per-neuron traces ``[B*N, T_d, C]`` with
    causal or bidirectional attention over time. If ``num_population_layers>0``,
    encoded features are reshaped to ``[B, T_d, N, C]`` and passed through a
    stack of ``PopulationBlock``s that attend over the neuron axis (with an
    optional mask), then reshaped back.

    Args:
        embed_dim: Embedding dimension (number of output channels).
        num_layers: Number of temporal Transformer blocks.
        num_heads: Number of attention heads per temporal block.
        patch_length: Size of each temporal patch (stride = patch_length).
        causal: If True, temporal attention is causal.
        dropout: Global dropout probability (attn weights + post-attn + post-FFN).
        num_population_layers: Number of PopulationBlock layers after the
            temporal stack. 0 → no cross-neuron attention.
        population_num_heads: Head count for population attention. None →
            inherit ``num_heads``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        patch_length: int,
        causal: bool = True,
        dropout: float = 0.0,
        num_population_layers: int = 0,
        population_num_heads: int | None = None,
    ):
        super().__init__()
        self.causal = causal
        self.patch_length = patch_length

        self.patchify = nn.Conv1d(1, embed_dim, kernel_size=patch_length, stride=patch_length)
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, dropout=dropout) for _ in range(num_layers)]
        )

        pop_heads = population_num_heads if population_num_heads is not None else num_heads
        self.population_layers = nn.ModuleList(
            [
                PopulationBlock(embed_dim, pop_heads, dropout=dropout)
                for _ in range(num_population_layers)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: Tensor,
        population_attn_mask: Tensor | None = None,
        time_key_padding_mask: Tensor | None = None,
        B: int | None = None,
        N: int | None = None,
    ) -> Tensor:
        """Encode per-neuron traces.

        Args:
            x: Input tensor of shape (B*N, T, 1).
            population_attn_mask: (B, N, N) bool mask used by the population
                stack. Ignored if there are no population layers. Must be
                provided when population layers are active.
            time_key_padding_mask: Optional (B*N, T_d) bool mask, True = valid
                token. Applied to every TransformerBlock in the temporal stack
                so that valid queries don't attend to trailing-padded tokens.
            B: Batch size. Required when population layers are active (cannot
                be inferred from ``x`` alone which has shape (B*N, T, 1)).
            N: Number of neurons. Required when population layers are active.

        Returns:
            Encoded features of shape (B*N, T_d, C), T_d = T // patch_length.
        """
        # x: (B*N, T, 1) -> transpose to (B*N, 1, T) for Conv1d
        h = x.transpose(1, 2)
        h = self.patchify(h)  # (B*N, C, T_d)
        h = h.transpose(1, 2)  # (B*N, T_d, C)

        for layer in self.layers:
            h = layer(
                h,
                is_causal=self.causal,
                key_padding_mask=time_key_padding_mask,
            )

        if len(self.population_layers) > 0:
            if B is None or N is None:
                raise ValueError(
                    "CalmEncoder: B and N must be provided when num_population_layers > 0."
                )
            if population_attn_mask is None:
                raise ValueError(
                    "CalmEncoder: population_attn_mask must be provided when "
                    "num_population_layers > 0."
                )
            BN, T_d, C = h.shape
            assert BN == B * N
            h = h.reshape(B, N, T_d, C).transpose(1, 2).contiguous()  # (B, T_d, N, C)
            for layer in self.population_layers:
                h = layer(h, attn_mask=population_attn_mask)
            h = h.transpose(1, 2).contiguous().reshape(B * N, T_d, C)

        return self.norm(h)
