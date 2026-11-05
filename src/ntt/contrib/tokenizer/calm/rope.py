"""Rotary Positional Embedding (RoPE) for CALM tokenizer.

Implements RoPE following Su et al., 2024. Applied to Q and K in
self-attention layers to encode relative temporal position.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RotaryPositionalEmbedding(nn.Module):
    """Precompute cos/sin tables for rotary positional embedding.

    Args:
        dim: Embedding dimension (must be even — each pair of dims is rotated).
        base: Base for the geometric frequency series (default 10000).
    """

    def __init__(self, dim: int, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, f"RoPE dim must be even, got {dim}"
        self.dim = dim
        self.base = base
        self._cached_seq_len: int = -1
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    @torch.no_grad()
    def forward(self, seq_len: int, device: torch.device) -> tuple[Tensor, Tensor]:
        """Compute cos/sin positional tables (cached across calls).

        Args:
            seq_len: Sequence length.
            device: Target device.

        Returns:
            (cos, sin) each of shape (seq_len, dim).
        """
        if (
            self._cached_seq_len != seq_len
            or self._cos_cached is None
            or self._cos_cached.device != device
        ):
            inv_freq = 1.0 / (
                self.base ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim)
            )
            t = torch.arange(seq_len, device=device).float()
            freqs = torch.outer(t, inv_freq)  # (seq_len, dim/2)
            emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()
            self._cached_seq_len = seq_len
        assert self._cos_cached is not None and self._sin_cached is not None
        return self._cos_cached, self._sin_cached


def apply_rotary_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """Apply rotary positional embedding to query and key tensors.

    Args:
        q: Query tensor (B, H, T, D).
        k: Key tensor (B, H, T, D).
        cos: Cosine table (T, D) from RotaryPositionalEmbedding.
        sin: Sine table (T, D) from RotaryPositionalEmbedding.

    Returns:
        Rotated (q, k) with same shapes.
    """
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    def _rotate_half(x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot
