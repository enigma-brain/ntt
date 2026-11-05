"""Causal wavelet + convolutional tokenizer family adapted from NVIDIA Cosmos.

``CosmosTokenizer2D`` treats the spike matrix as a single-channel (neuron ×
time) image. Wavelet + conv operations are causal along the time axis and
non-causal along the neuron axis. The latent preserves a unit-like axis.

Adapted from NVIDIA Cosmos-Tokenizer, used under the Apache-2.0 license:
https://github.com/NVIDIA/Cosmos-Tokenizer
(see ``LICENSES/cosmos-tokenizer-Apache-2.0.txt``).
"""

from ntt.contrib.tokenizer.cosmos.cosmos_2d import (
    CosmosTokenizer2D,
    CosmosTokenizer2DSpec,
)

__all__ = [
    "CosmosTokenizer2D",
    "CosmosTokenizer2DSpec",
]
