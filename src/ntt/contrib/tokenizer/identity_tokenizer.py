"""Identity tokenizer - a simple pass-through example.

This is a minimal example tokenizer that demonstrates the pattern.
It doesn't actually tokenize - just passes data through.
"""

from typing import Literal

import torch

from ntt.core.components.types import (
    BinnedSpikesInput,
    DataType,
    LatentType,
    TokenizerOutput,
)
from ntt.core.model.tokenizer.base import (
    BaseTokenizer,
    TokenizerSpec,
    register_tokenizer,
)


class IdentityTokenizerSpec(TokenizerSpec, frozen=True):
    """Spec for identity tokenizer.

    This example has no additional parameters beyond the base spec.
    In practice, you'd add algorithm-specific params here.

    Parameters:
        kind: Discriminator field (always "identity")
        vocab_size (int): Size of vocabulary (for this example only)
    """

    # Discriminator field
    KIND: Literal["identity"] = "identity"

    vocab_size: int = 1024

    # Capability metadata
    latent_type: LatentType = LatentType.DISCRETE
    compress_n: bool = False
    compress_t: bool = False
    variable_n: bool = True
    variable_t: bool = True
    variable_dt: bool = True
    input_type: DataType = DataType.SPIKES
    output_type: DataType = DataType.SPIKES

    def get_vocab_size(self) -> int:
        """Return the vocabulary size."""
        return self.vocab_size

    def get_codes_per_token(self) -> int:
        """One token index per position."""
        return 1


@register_tokenizer
class IdentityTokenizer(BaseTokenizer):
    """Identity tokenizer - passes data through unchanged.

    This is a minimal example demonstrating the tokenizer pattern.
    Real tokenizers would implement actual encoding/decoding logic.
    """

    # IMPORTANT: Must set this class attribute for registry
    spec_class = IdentityTokenizerSpec

    def __init__(self, spec: IdentityTokenizerSpec):
        super().__init__(spec)
        # In a real tokenizer, you'd initialize your architecture here
        # e.g., encoder networks, codebook, etc.

        # Store input for identity reconstruction
        self._stored_input: torch.Tensor | None = None

    def encode(self, x: BinnedSpikesInput) -> torch.Tensor:
        """Encode neural data to discrete tokens.

        For identity tokenizer, store the input and return dummy tokens.

        Args:
            x: BinnedSpikesInput with spike_matrix

        Returns:
            Token indices of shape (batch, sequence_length)
        """
        # Store input for identity reconstruction
        self._stored_input = x.spike_matrix.clone()

        batch_size = x.spike_matrix.shape[0]
        seq_len = x.spike_matrix.shape[1] if x.spike_matrix.ndim > 1 else 1
        # Return dummy token indices in range [0, vocab_size)
        return torch.zeros(batch_size, seq_len, dtype=torch.long, device=x.spike_matrix.device)

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode discrete tokens back to neural data.

        For identity tokenizer, return the stored input unchanged.

        Args:
            tokens: Token indices (ignored for identity)

        Returns:
            Original input data
        """
        if self._stored_input is None:
            raise RuntimeError(
                "decode() called before encode(). Cannot reconstruct without stored input."
            )
        return self._stored_input.clone()

    def forward(self, x: BinnedSpikesInput) -> TokenizerOutput:
        """Forward pass - encode and decode.

        Args:
            x: BinnedSpikesInput with spike_matrix

        Returns:
            TokenizerOutput with tokens and reconstruction (no auxiliary loss)
        """
        tokens = self.encode(x)
        reconstruction = self.decode(tokens)

        return TokenizerOutput(
            tokens=tokens,
            reconstruction=reconstruction,
            aux_loss=None,  # Identity tokenizer has no auxiliary loss
        )
