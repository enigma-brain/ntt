"""User-contributed implementations for tokenizers and loss functions.

This module provides a plugin system for community contributions.
See the submodules for details:

- `contrib.tokenizer`: Tokenizer implementations
- `contrib.loss`: Loss function implementations
"""

from .loss import LossSpecKind
from .tokenizer import TokenizerSpecKind

__all__ = [
    # Tokenizers
    "TokenizerSpecKind",
    # Losses
    "LossSpecKind",
]
