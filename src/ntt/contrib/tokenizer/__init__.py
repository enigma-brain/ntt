"""User-contributed tokenizer implementations.

Contributors should place their tokenizer implementations here.

Each tokenizer file should:
1. Define a TokenizerSpec subclass with a `KIND` discriminator field
2. Define a BaseTokenizer subclass with the implementation
3. Set spec_class as a class attribute on the tokenizer
4. Register using @register_tokenizer decorator

Example structure:
    from typing import Literal

    from ntt.core.model.tokenizer.base import (
        BaseTokenizer,
        TokenizerSpec,
        register_tokenizer,
    )

    class MyTokenizerSpec(TokenizerSpec):
        KIND: Literal["my_tokenizer"] = "my_tokenizer"

        # Add your custom params
        hidden_dim: int = 256
        num_layers: int = 3

        def get_vocab_size(self) -> int:
            return 1024

        def get_codes_per_token(self) -> int:
            return 1  # codes of size get_vocab_size() per [t, u] latent position

    @register_tokenizer
    class MyTokenizer(BaseTokenizer):
        spec_class = MyTokenizerSpec

        def __init__(self, spec: MyTokenizerSpec):
            super().__init__(spec)
            # Initialize your model architecture

        def encode(self, x):
            ...

        def decode(self, tokens):
            ...

        def forward(self, x):
            ...
"""

import typing as T

from pydantic import Discriminator

from .calm import CalmTokenizerSpec
from .compressed_sensing import CompressedSensingTokenizerSpec
from .conv_vae import ConvVAETokenizerSpec
from .cosmos import CosmosTokenizer2DSpec
from .frequency_based_lut import FrequencyBasedLUTTokenizer, FrequencyBasedLUTTokenizerSpec
from .fsqffn import FSQFFNTokenizerSpec
from .identity_tokenizer import IdentityTokenizerSpec
from .ldns import LDNSTokenizerSpec
from .pca_baseline import PCABaselineTokenizerSpec
from .perceiver import PerceiverTokenizerSpec
from .poisson_baseline import PoissonBaselineTokenizerSpec
from .simple_vae import SimpleVAETokenizerSpec
from .wanneural import WanNeuralTokenizerSpec

# Discriminated union of all tokenizer specs
# This allows automatic deserialization based on the "KIND" field
# Usage: TokenizerSpecKind.model_validate({"KIND": "identity", ...})
TokenizerSpecKind = T.Annotated[
    T.Union[  # noqa: UP007
        CalmTokenizerSpec,
        CompressedSensingTokenizerSpec,
        ConvVAETokenizerSpec,
        CosmosTokenizer2DSpec,
        IdentityTokenizerSpec,
        PCABaselineTokenizerSpec,
        PoissonBaselineTokenizerSpec,
        LDNSTokenizerSpec,
        FrequencyBasedLUTTokenizerSpec,
        PerceiverTokenizerSpec,
        SimpleVAETokenizerSpec,
        FSQFFNTokenizerSpec,
        WanNeuralTokenizerSpec,
        None,
    ],
    Discriminator("KIND"),
]

__all__ = [
    "CalmTokenizerSpec",
    "CompressedSensingTokenizerSpec",
    "ConvVAETokenizerSpec",
    "CosmosTokenizer2DSpec",
    "IdentityTokenizerSpec",
    "PCABaselineTokenizerSpec",
    "PoissonBaselineTokenizerSpec",
    "LDNSTokenizerSpec",
    "FrequencyBasedLUTTokenizer",
    "FrequencyBasedLUTTokenizerSpec",
    "PerceiverTokenizerSpec",
    "SimpleVAETokenizerSpec",
    "FSQFFNTokenizerSpec",
    "WanNeuralTokenizerSpec",
    "TokenizerSpecKind",
]
