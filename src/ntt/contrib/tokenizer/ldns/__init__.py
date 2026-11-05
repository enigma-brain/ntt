"""LDNS (Latent Diffusion for Neural Spiking) tokenizer.

This module implements a tokenizer based on the LDNS paper, using S4 (Structured
State Space) blocks for efficient temporal modeling of neural spike data.

The tokenizer:
1. Takes binned spike counts as input
2. Encodes to a continuous latent representation
3. Decodes to predicted firing rates
4. Supports optional history-dependent spike generation

Example usage:
    from ntt.contrib.tokenizer.ldns import LDNSTokenizerSpec, LDNSTokenizer

    spec = LDNSTokenizerSpec(
        n_neurons=128,
        hidden_dim=256,
        latent_dim=32,
    )
    tokenizer = LDNSTokenizer(spec)

Based on:
Kapoor, J., et al., 2024. "Latent diffusion for neural spiking data."

Adapted from the LDNS reference implementation, used under the MIT license:
https://github.com/mackelab/LDNS (see ``LICENSES/ldns-MIT.txt``).
"""

from .ldns_tokenizer import LDNSTokenizer, LDNSTokenizerSpec

__all__ = [
    "LDNSTokenizer",
    "LDNSTokenizerSpec",
]
