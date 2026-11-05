# Contributing Tokenizers

This directory contains user-contributed tokenizer implementations for neural spike data.

## Quick Start

To contribute a new tokenizer:

1. **Create your tokenizer** as a module or within a subpackage (if you break it down into multiple modules) in  `src/ntt/contrib/tokenizer/`
2. **Define a tokenizer spec class** inheriting from `TokenizerSpec`
3. **Implement the tokenizer base class** inheriting from `BaseTokenizer`
4. **Register the tokenizer base class** using the `@register_tokenizer` decorator
5. **Add an import for the tokenizer spec class** to the `TokenizerSpecKind` var in `src/ntt/contrib/tokenizer/__init__.py`
6. **Create a config yaml for the tokenizer** in `config/tokenizer/contrib/`

## Example Implementation

```python
"""My custom tokenizer implementation."""

from typing import Literal

import torch
import torch.nn as nn

from ntt.core.model.tokenizer.base import (
    BaseTokenizer,
    DataType,
    LatentType,
    TokenizerSpec,
    register_tokenizer,
)


class MyTokenizerSpec(TokenizerSpec):
    """Configuration for my tokenizer.

    Parameters:
        hidden_dim: Size of hidden layers
        num_layers: Number of encoder/decoder layers
        codebook_size: Number of discrete codes (overrides vocab_size)
        commitment_weight: Weight for commitment loss
    """


    KIND: Literal["my_tokenizer"] = "my_tokenizer"

    latent_type: LatentType = LatentType.DISCRETE
    input_type: DataType = DataType.SPIKES
    output_type: DataType = DataType.SPIKES

    compress_n: bool = True
    compress_t: bool = True
    variable_n: bool = True
    variable_t: bool = False
    variable_dt: bool = False

    hidden_dim: int = 256
    num_layers: int = 3
    codebook_size: int = 1024
    commitment_weight: float = 0.25

    def get_vocab_size(self) -> int:
        """Return the vocabulary size."""
        return self.codebook_size


# `@register_tokenizer` decorator reads `KIND` from `spec_class.model_fields`,
@register_tokenizer
class MyTokenizer(BaseTokenizer):
    """My custom tokenizer implementation.

    This tokenizer uses [describe your approach here].
    """

    # REQUIRED: Set this to your spec class
    spec_class = MyTokenizerSpec

    def __init__(self, spec: MyTokenizerSpec):
        super().__init__(spec)

        # Access spec parameters
        self.hidden_dim = spec.hidden_dim
        self.num_layers = spec.num_layers

        # Build your architecture
        self.encoder = self._build_encoder()
        self.codebook = nn.Embedding(spec.codebook_size, self.hidden_dim)
        self.decoder = self._build_decoder()

    def _build_encoder(self) -> nn.Module:
        """Build the encoder network."""
        # Implement your encoder
        return nn.Sequential(
            # Your layers here
        )

    def _build_decoder(self) -> nn.Module:
        """Build the decoder network."""
        # Implement your decoder
        return nn.Sequential(
            # Your layers here
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode neural data to discrete tokens.

        Args:
            x: Input neural data tensor

        Returns:
            Discrete token indices
        """
        # Implement encoding logic
        z = self.encoder(x)
        # Quantize to discrete codes
        tokens = self._quantize(z)
        return tokens

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode tokens back to neural data.

        Args:
            tokens: Discrete token indices

        Returns:
            Reconstructed neural data
        """
        # Look up embeddings
        z_q = self.codebook(tokens)
        # Decode
        reconstruction = self.decoder(z_q)
        return reconstruction

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Full forward pass with all outputs.

        Args:
            x: Input neural data

        Returns:
            Dictionary containing:
                - 'tokens': Discrete token indices
                - 'reconstruction': Reconstructed data
                - 'loss': Optional tokenizer-specific loss
                - Any other outputs needed for training/analysis
        """
        # Encode
        tokens = self.encode(x)

        # Decode
        reconstruction = self.decode(tokens)

        # Compute any tokenizer-specific losses
        # e.g., commitment loss, codebook loss, etc.
        commitment_loss = self._compute_commitment_loss(x, tokens)

        return {
            "tokens": tokens,
            "reconstruction": reconstruction,
            "commitment_loss": commitment_loss,
        }

    def _quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Quantize continuous embeddings to discrete codes."""
        # Implement quantization logic
        pass

    def _compute_commitment_loss(
        self, x: torch.Tensor, tokens: torch.Tensor
    ) -> torch.Tensor:
        """Compute commitment loss or other auxiliary losses."""
        # Implement loss computation
        pass
```

## Configuration File

Create a corresponding config file at `config/tokenizer/contrib/my_tokenizer.yaml`:

```yaml
_target_: ntt.contrib.tokenizer.my_tokenizer.MyTokenizerSpec
KIND: my_tokenizer  # must match the spec's KIND literal — Hydra/Pydantic dispatch on it

# Base parameters (required)
window_length_seconds: 1.0
bin_size_ms: 10.0       # use `false` for unbinned spike-times input
n_neurons: false         # int / Dist / false (= use full population)
shuffling_strategy: "none"
sampling_strategy: "uniform"

# Your tokenizer-specific parameters
hidden_dim: 256
num_layers: 3
codebook_size: 1024
commitment_weight: 0.25

# Path-style fields default to None on the spec — write `null` explicitly
# in yaml when not yet set, to avoid Hydra interpolation surprises.
# fitted_artifact_path: null
```

## Register Your Tokenizer

Add **only the spec class** to `src/ntt/contrib/tokenizer/__init__.py`, and add it to the `TokenizerSpecKind` discriminated union. Importing the spec executes `my_tokenizer.py` top-to-bottom, which fires `@register_tokenizer` as a side effect — so the tokenizer class is registered without ever appearing in `__init__.py`:

```python
from .my_tokenizer import MyTokenizerSpec  # also fires @register_tokenizer for MyTokenizer

TokenizerSpecKind = T.Annotated[
    T.Union[  # noqa: UP007
        # ... existing specs
        MyTokenizerSpec,
        None,
    ],
    Discriminator("KIND"),
]

__all__ = [
    # ... existing specs
    "MyTokenizerSpec",
    "TokenizerSpecKind",
]
```

### Why we re-export only the spec

`@register_tokenizer` registers the tokenizer class into a global registry keyed on `spec.KIND`, so callers reach the tokenizer via `create_tokenizer(spec)` rather than by importing the class directly. The decorator runs as a side effect of executing the module top-to-bottom — i.e. the **first time the module is imported**. By convention every contrib tokenizer co-locates its spec and tokenizer in one file (`my_tokenizer.py` defines both `MyTokenizerSpec` and `MyTokenizer`), so importing the spec into `__init__.py` is enough to execute the file and fire the decorator. Adding the spec to `TokenizerSpecKind` is what enables Pydantic's discriminated-union-based deserialization of yaml configs (Pydantic dispatches on the `KIND` field to pick the right spec subclass).

If you split your spec and tokenizer into separate modules, ensure the tokenizer module is imported somewhere that runs at package load — e.g. add `from . import my_tokenizer  # noqa: F401` to your spec file — otherwise the decorator never fires and `create_tokenizer` raises an unknown-`KIND` error.

## Using Your Tokenizer

Once registered, your tokenizer can be used via config:

```bash
uv run ntt paths.run_name=my_run \
  tokenizer@job.tokenizer_config=contrib/my_tokenizer \
  loss@job.loss_config=contrib/mse
```

Or programmatically:

```python
from ntt.core.model.tokenizer.base import create_tokenizer
from ntt.contrib.tokenizer.my_tokenizer import MyTokenizerSpec

spec = MyTokenizerSpec(
    window_length_seconds=1.0,
    bin_size_ms=10.0,
    hidden_dim=512,
    num_layers=4,
)

tokenizer = create_tokenizer(spec)
```

## Requirements

Your tokenizer implementation must:

1. **Inherit from `BaseTokenizer`**
2. **Set `spec_class` as a class attribute**
3. **Implement all abstract methods**:
  - `encode(x) -> tokens/latents`
     - Continuous tokenizers: return `latents`, continuous unified 4D tensor, see below.
     - Discrete tokenizers: either return `latents` (unified 4D format),
                             or `tokens`, integer token IDs (arbitrary shape).
  - For discrete tokenizers where `encode()` returns integer token IDs:
    - `tokens_to_latents(tokens) -> latents`
  - `decode(latents) -> reconstruction`
  - `forward(x) -> TokenizerOutput`: End-to-end input-to-output
4. **Define a spec class inheriting from `TokenizerSpec`**
5. **Implement `get_vocab_size()` in your spec** (discrete tokenizers only)
6. **Set capability metadata** (see Capability Metadata section below)
7. **Set latent dimension semantics** (see Unified Latent Shape section below)
8. **Register using `@register_tokenizer`**

If a discrete tokenizer does not convert its token IDs to continuous latents,
`test_discrete_tokens_to_latents_returns_continuous_values` fails and latent
robustness (M2) scores for that tokenizer must not be interpreted.

(Note: Some tokenizers populate `.tokens` or `.latents` in `forward()` but
this is not used in the evaluation.)


## Tokenizers that are computed once from data (no gradient training)

Some tokenizers are not trained step by step but computed once from the training data. To write one:

- Accumulate what you need batch by batch (e.g. running sums and a count). Store these as buffers (`register_buffer`) so they are saved in the checkpoint.
- Implement `fit_from_running_stats()`: compute final parameters from those accumulated values, set `fitted` to True, and return True. If already fitted or not enough data, return False.

At each epoch's end, `src/ntt/core/model/litmodule.py` calls `fit_from_running_stats()` on the tokenizer if that method exists, so one epoch is enough.

Optionally, set `every_n_train_steps: null` for the checkpoint callback in your experiment config. Checkpoints saved partway through an epoch come from the model before fitting has happened, so they are not useful.


## Unified Latent Shape Convention (4D: `[B, T, U, D]`)

All continuous tokenizers must return latents/tokens from `encode()` and
`forward()` in the **unified 4D shape** `[B, T, U, D]`:


| Dimension | Meaning         | When to use singleton (1)                                    |
| --------- | --------------- | ------------------------------------------------------------ |
| **B**     | Batch           | Never                                                        |
| **T**     | Time tokens     | When `compress_t=True` and whole window maps to one token    |
| **U**     | Unit/neuron     | When `compress_n=True` (neurons compressed into latent)      |
| **D**     | Latent channels | When there is no multi-dimensional latent (e.g. scalar rate) |


Insert singleton dimensions (`unsqueeze` / `rearrange`) for axes that your
tokenizer does not produce. This lets downstream code (evaluation pipeline,
metrics, collation) handle all tokenizers uniformly without special-casing.

### Spec fields describing which dimensions are meaningful

```python
# In your TokenizerSpec subclass:
has_time_dim: bool    # True if T > 1 (multiple tokens over time)
has_unit_dim: bool    # True if U > 1 (per-unit representations)
has_latent_dim: bool  # True if D > 1 (multi-dimensional latent per token)
```

### Examples from existing tokenizers


| Tokenizer        | Encode output         | has_time | has_unit | has_latent |
| ---------------- | --------------------- | -------- | -------- | ---------- |
| LDNS             | `[B, T, 1, C_latent]` | True     | False    | True       |
| CS per_neuron    | `[B, 1, U, m]`        | False    | True     | True       |
| CS per_time_bin  | `[B, T, 1, m]`        | True     | False    | True       |
| CS full_window   | `[B, 1, 1, m]`        | False    | False    | True       |
| SimpleVAE        | `[B, 1, 1, D]`        | False    | False    | True       |
| Poisson baseline | `[B, T', U, 1]`       | True     | True     | False      |


### `decode()` must accept the same 4D format

For continuous tokenizers, your `decode()` should accept the 4D tensor returned
by `encode()`. Handle the singleton dimensions internally (e.g. squeeze before your decoder, then
reshape the output back to `[B, T, U]`).
You can support internal use of `decode(encode(x))`, by adding an if/else check to
             `decode()` to check whether its input is continuous or integer-valued.
             And converting to latents in `decode()` only when receiving integer IDs.
             But the code is built for `decode(tokens_to_latents(encode(x)))`.

The `reconstruct()` method (called by the evaluation pipeline via
`reconstruct_from_input()`) receives, for continuous tokenizers (where
`tokens_to_latents()` is the identity), the output of `encode()` directly,
so it must also handle 4D input. For discrete tokenizers,
the base evaluation path first applies `tokens_to_latents()`.

## Base TokenizerSpec Parameters

All tokenizer specs must include these parameters:

- `window_length_seconds`: Duration of input window (float or Dist)
- `bin_size_ms`: Temporal bin size in milliseconds (float or False for unbinned)
- `n_neurons`: Number of neurons (int, Dist, or False for all)
- `shuffling_strategy`: Data shuffling approach (str)
- `sampling_strategy`: Window sampling approach (str)

You can access these in your datamodule to configure data loading appropriately.

## Capability Metadata

All tokenizer specs must declare capability metadata that describes what the tokenizer can do. This acts as an "identity card" for your tokenizer and enables automatic validation and testing.

### Required Capability Fields

```python
from ntt.core.model.tokenizer.base import LatentType, DataType

class MyTokenizerSpec(TokenizerSpec):
    # ... your parameters ...

    # Capability metadata - REQUIRED
    latent_type: LatentType = LatentType.DISCRETE  # or LatentType.CONTINUOUS
    compress_n: bool = True   # Compresses along neuron dimension?
    compress_t: bool = True   # Compresses along time dimension?
    variable_n: bool = True   # Supports variable number of neurons?
    variable_t: bool = False  # Supports variable time length?
    variable_dt: bool = False # Supports variable bin size?
    input_type: DataType = DataType.SPIKES   # What input format?
    output_type: DataType = DataType.SPIKES   # What output format?
```

### Capability Types

- `**latent_type**`: `LatentType.CONTINUOUS` or `LatentType.DISCRETE`
- `**input_type` / `output_type**`: `DataType.SPIKES`, `DataType.RATES`, or `DataType.SPIKE_TIMES`
- **Compression flags**: `True` if the tokenizer compresses along that dimension
- **Variable flags**: `True` if the tokenizer can handle variable sizes

### Validation

The base class provides automatic validation:

- Checks consistency between capabilities and configuration
- Validates that discrete tokenizers have vocab_size > 0
- Ensures fixed-size requirements match variable flags

## Tips

- **Use descriptive names**: Choose a clear, unique name for your tokenizer
- **Document parameters**: Add docstrings explaining all config parameters
- **Type hints**: Use proper type hints for all methods
- **Validation**: Use Pydantic validators in your spec for parameter validation
- **Testing**: Consider adding tests for your tokenizer
- **Examples**: Provide example configs demonstrating different use cases

## Support

If you have questions or need help:

- Check the `identity_tokenizer.py` example in this directory
- Review the base classes in `../base.py`
- Open an issue on the repo
