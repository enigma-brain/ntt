"""FSQFFN (Finite Scalar Quantization Feed-Forward Network) tokenizer.

This module implements a tokenizer for compressing temporal windows of continuous
neural signals (e.g., calcium traces) into discrete tokens using Finite Scalar
Quantization (FSQ).

The tokenizer:
1. Takes binned continuous signals as input
2. Encodes via feed-forward networks to a latent representation
3. Quantizes the latent using FSQ to produce discrete tokens
4. Decodes back to reconstructed signals

Key components:
- FSQuantizer: Implements FSQ quantization from "Finite Scalar Quantization: VQ-VAE Made Simple"
- FeedForward/GatedFeedForwardLayer: Multi-layer FFN blocks with gated activation
- FSQFFNTokenizer: Main tokenizer combining encoder, quantizer, and decoder

Example usage:
    from ntt.contrib.tokenizer.fsqffn import FSQFFNTokenizerSpec, FSQFFNTokenizer

    spec = FSQFFNTokenizerSpec(
        n_neurons=1,
        window_length_seconds=1.0,
        bin_size_ms=10.0,
        n_samples_per_token=100,
        q_levels=(8, 5, 5, 5),  # 1024 codebook size
        ffn_model_dim=128,
    )
    tokenizer = FSQFFNTokenizer(spec)

Based on:
- FSQ: "Finite Scalar Quantization: VQ-VAE Made Simple" - https://arxiv.org/abs/2309.15505
- Adapted from: https://github.com/lucidrains/vector-quantize-pytorch
"""

from .fsqffn import FSQFFNTokenizerSpec

__all__ = [
    "FSQFFNTokenizerSpec",
]
