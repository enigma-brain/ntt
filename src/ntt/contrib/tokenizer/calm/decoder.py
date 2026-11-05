"""CALM Transformer decoder with TransposedConv1D unpatchification.

Mirrors the encoder: Transformer layers -> TransposedConv1D to reconstruct
the original temporal resolution.
  [B*N, T_d, C] -> Transformer layers -> TransposedConv1D -> [B*N, T, 1]
"""

from __future__ import annotations

from torch import Tensor, nn

from .encoder import TransformerBlock


class CalmDecoder(nn.Module):
    """CALM-style per-neuron Transformer decoder.

    Bidirectional by default — reconstruction is not autoregressive, so each
    token may attend to past and future tokens. Pass ``causal=True`` to force
    causal attention (useful for variable-length training where trailing
    padded tokens would otherwise leak into valid positions).

    Args:
        embed_dim: Transformer embedding dimension (C).
        num_layers: Number of Transformer layers.
        num_heads: Number of attention heads per layer.
        patch_length: Temporal expansion factor (inverse of patchification).
        dropout: Dropout probability passed to each TransformerBlock.
        causal: If True, apply causal self-attention inside each layer.
    """

    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        patch_length: int,
        dropout: float = 0.0,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.patch_length = patch_length
        self.causal = causal
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.unpatchify = nn.ConvTranspose1d(
            in_channels=embed_dim,
            out_channels=1,
            kernel_size=patch_length,
            stride=patch_length,
        )

    def forward(
        self,
        z: Tensor,
        time_key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Decode quantized features back to per-neuron traces.

        Args:
            z: (B*N, T_d, C) quantized features.
            time_key_padding_mask: Optional (B*N, T_d) bool mask, True = valid
                token. Prevents valid queries from attending to trailing-padded
                tokens (essential when ``causal=False`` and samples in the
                batch were variable-length cropped).

        Returns:
            (B*N, T, 1) reconstructed traces, T = T_d * patch_length.
        """
        h = z
        for layer in self.layers:
            h = layer(
                h,
                is_causal=self.causal,
                key_padding_mask=time_key_padding_mask,
            )
        h = self.norm(h)
        h = h.transpose(1, 2)
        h = self.unpatchify(h)
        h = h.transpose(1, 2)
        return h
