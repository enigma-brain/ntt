"""Autoregressive regularizer for discrete tokenizers.

A reusable module that adds next-token prediction as an auxiliary training loss.
Works with any discrete bottleneck (VQ, FSQ, BSQ).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ARRegularizer(nn.Module):
    """Next-token prediction head for tokenizer AR regularization.

    Args:
        embed_dim: Dimension of input latent vectors (C).
        vocab_size: Size of the discrete token vocabulary.
        num_heads: Number of attention heads in the causal Transformer layer.
        horizon: Number of future steps to predict (1 = next-step only).
        dropout: Dropout probability passed to ``nn.TransformerEncoderLayer``.
    """

    def __init__(
        self,
        embed_dim: int,
        vocab_size: int,
        num_heads: int = 4,
        horizon: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.horizon = horizon

        self.attn_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.pred_heads = nn.ModuleList([nn.Linear(embed_dim, vocab_size) for _ in range(horizon)])

    def _get_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Get causal attention mask, cached for repeated calls with same T."""
        if (
            not hasattr(self, "_cached_mask")
            or self._cached_mask.shape[0] != T
            or self._cached_mask.device != device
        ):
            self._cached_mask = nn.Transformer.generate_square_subsequent_mask(T, device=device)
        return self._cached_mask

    def get_logits(self, z_seq: torch.Tensor) -> torch.Tensor:
        """Get next-token logits (for testing). Returns (B, T_d, vocab_size)."""
        mask = self._get_causal_mask(z_seq.shape[1], z_seq.device)
        h = self.attn_layer(z_seq, src_mask=mask, is_causal=True)
        return self.pred_heads[0](h)

    def forward(
        self,
        z_seq: torch.Tensor,
        token_indices: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute AR cross-entropy loss.

        Args:
            z_seq: (B, T_d, C) sequence of latent vectors.
            token_indices: (B, T_d) ground-truth discrete token indices.
            valid_mask: Optional (B, T_d) bool mask, True = real position.
                When supplied, loss is zero-averaged over pairs (t, t+h) where
                **both** source and target positions are valid. Padded-tail
                tokens are therefore excluded from the AR signal.

        Returns:
            Scalar CE loss averaged over valid positions and horizon steps.
        """
        T_d = z_seq.shape[1]
        mask = self._get_causal_mask(T_d, z_seq.device)
        attended = self.attn_layer(z_seq, src_mask=mask, is_causal=True)

        total_loss = torch.zeros((), device=z_seq.device, dtype=z_seq.dtype)
        num_steps = 0

        for h in range(1, self.horizon + 1):
            if T_d - h <= 0:
                continue
            logits = self.pred_heads[h - 1](attended[:, : T_d - h])  # (B, T_d-h, vocab_size)
            targets = token_indices[:, h:]  # (B, T_d-h)
            if valid_mask is None:
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                )
            else:
                # Pair is valid iff source (t) AND target (t+h) are both valid.
                pair_mask = valid_mask[:, : T_d - h] & valid_mask[:, h:]  # (B, T_d-h)
                pair_flat = pair_mask.reshape(-1)
                per_elem = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                    reduction="none",
                )  # (B*(T_d-h),)
                num_valid = pair_flat.sum()
                if num_valid > 0:
                    loss = (per_elem * pair_flat.to(per_elem.dtype)).sum() / num_valid
                else:
                    # No valid pairs at this horizon — skip this step.
                    continue
            total_loss = total_loss + loss
            num_steps += 1

        if num_steps == 0:
            return total_loss

        return total_loss / num_steps
