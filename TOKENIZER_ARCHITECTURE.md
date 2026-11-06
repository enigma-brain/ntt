# NTT Tokenizer Architecture

This document describes the tokenizer infrastructure for Neurons To Tokens (NTT).

## Overview

The NTT tokenizer system provides a flexible, extensible framework for researchers to experiment with different tokenization algorithms for neural spike data. It uses:

- **Pydantic** for type-safe configuration
- **Registry pattern** for contributor tokenizers
- **Factory pattern** for instantiation
- **Hydra** for config composition

## Architecture

### Core Components

```
src/ntt/
├── core/model/tokenizer/
│   ├── __init__.py                # Public API + checkpoint loader
│   ├── base.py                    # Base classes, registry, factory
│   └── validation.py              # Dummy data generation for tokenizer validation
└── contrib/                       # User contributions
    ├── __init__.py                # Re-exports TokenizerSpecKind, LossSpecKind
    ├── README.md                  # Contributor guide
    └── tokenizer/
        ├── __init__.py            # TokenizerSpecKind discriminated union
        ├── identity_tokenizer.py  # Minimal pass-through example
        └── simple_vae.py          # Single-neuron VAE example
```

### Class Hierarchy

```
Dist (ABC, BaseModel)
  ↓
  [Concrete distribution classes]

TokenizerSpec (ABC, BaseModel)
  ↓
  ├── IdentityTokenizerSpec
  ├── SimpleVAETokenizerSpec
  └── [User-contributed specs]

BaseTokenizer (ABC, nn.Module)
  ↓
  ├── IdentityTokenizer
  ├── SimpleVAETokenizer
  └── [User-contributed tokenizers]
```

## Key Design Patterns

### 1. Specification + Implementation Separation

Each tokenizer consists of two classes:

**TokenizerSpec (Pydantic BaseModel)**

- Frozen configuration object
- Contains both:
  - Shared attributes (window_length, n_neurons, etc.) used by datamodule
  - Algorithm-specific parameters (hidden_dim, codebook_size, etc.)
- Validated at instantiation
- Serializable for reproducibility

**BaseTokenizer (nn.Module)**

- Implementation of the algorithm
- Takes spec in `__init`__
- Defines `encode()`, `decode()`, `forward()`

### 2. Registry Pattern

Tokenizers are registered using a decorator:

```python
@register_tokenizer
class MyTokenizer(BaseTokenizer):
    spec_class = MyTokenizerSpec
    ...
```

Benefits:

- Contributors add a module under `src/ntt/contrib/tokenizer/`
- No central registry file to modify (the spec is added to `TokenizerSpecKind`)
- Clear coupling between spec and implementation

### 3. Factory Pattern

Tokenizers are instantiated from specs:

```python
spec = MyTokenizerSpec(window_length_seconds=1.0, ...)
tokenizer = create_tokenizer(spec)  # Factory looks up registered class
```

### 4. Hydra Integration

Each tokenizer has a YAML config under `config/tokenizer/contrib/` whose `_target_` is the spec class:

```yaml
# config/tokenizer/contrib/simple_vae.yaml
_target_: ntt.contrib.tokenizer.SimpleVAETokenizerSpec
KIND: simple_vae
latent_dim: 8
beta: 1.0
```

The train job (`config/job/train.yaml`) composes the `tokenizer` config group into `job.tokenizer_config`, so a config is selected with the group override `tokenizer@job.tokenizer_config=<name>`. The LitModule receives the instantiated spec and uses the factory:

```python
class NTTLitModule(L.LightningModule):
    def __init__(self, tokenizer_spec: TokenizerSpec, ...):
        ...
        self.tokenizer = create_tokenizer(tokenizer_spec)
```

## Workflow for Contributors

The contributor guide, [`src/ntt/contrib/README.md`](src/ntt/contrib/README.md), has the full walkthrough and a complete example. In short:

### 1. Create Tokenizer Implementation

Create `src/ntt/contrib/tokenizer/my_tokenizer.py` (or a subpackage if it has several modules):

```python
from typing import Literal

from ntt.core.model.tokenizer.base import (
    BaseTokenizer, TokenizerSpec, register_tokenizer
)

class MyTokenizerSpec(TokenizerSpec):
    KIND: Literal["my_tokenizer"] = "my_tokenizer"

    # Add algorithm-specific params
    my_param: int = 42

    def get_vocab_size(self) -> int:
        return 1024

@register_tokenizer
class MyTokenizer(BaseTokenizer):
    spec_class = MyTokenizerSpec

    def encode(self, x): ...
    def decode(self, tokens): ...
    def forward(self, x): ...
```

### 2. Register the Spec

Import the spec in `src/ntt/contrib/tokenizer/__init__.py` and add it to the `TokenizerSpecKind` discriminated union (and to `__all__`). Importing the spec runs the module, which fires `@register_tokenizer`, so the tokenizer class itself does not need to appear there:

```python
from .my_tokenizer import MyTokenizerSpec

TokenizerSpecKind = T.Annotated[
    T.Union[  # noqa: UP007
        # ... existing specs
        MyTokenizerSpec,
        None,
    ],
    Discriminator("KIND"),
]
```

### 3. Create Config File

Create `config/tokenizer/contrib/my_tokenizer.yaml`:

```yaml
_target_: ntt.contrib.tokenizer.my_tokenizer.MyTokenizerSpec
KIND: my_tokenizer

# Base params
window_length_seconds: 1.0
bin_size_ms: 10.0

# Your params
my_param: 42
```

### 4. Use via CLI or Code

**Via CLI:** select the tokenizer (and a loss) with Hydra group overrides:

```bash
uv run ntt paths.run_name=my_run \
  tokenizer@job.tokenizer_config=contrib/my_tokenizer \
  loss@job.loss_config=contrib/mse
```

**Via Code:**

```python
spec = MyTokenizerSpec(window_length_seconds=1.0)
tokenizer = create_tokenizer(spec)
```

## Integration with DataModule

The datamodule can access tokenizer spec attributes to configure data loading:

```python
class NTTDataModule(L.LightningDataModule):
    def __init__(self, tokenizer_spec: TokenizerSpec):
        # Use spec to configure data loading
        self.window_length = tokenizer_spec.window_length_seconds
        self.bin_size = tokenizer_spec.bin_size_seconds
        self.n_neurons = tokenizer_spec.n_neurons

        # Sample from distributions if needed
        if isinstance(tokenizer_spec.window_length_seconds, Dist):
            self.window_length = tokenizer_spec.window_length_seconds.sample()
```

## Distributional Parameters

The `Dist` base class allows specs to specify distributional parameters:

```python
class UniformDist(Dist):
    low: float
    high: float

    def sample(self) -> float:
        return random.uniform(self.low, self.high)

spec = MyTokenizerSpec(
    window_length_seconds=UniformDist(low=0.5, high=2.0),
    ...
)
```

This enables sampling different values during training for data augmentation.

## Required Methods

### TokenizerSpec

- `get_vocab_size() -> int`: Return vocabulary size

### BaseTokenizer

- `encode(x: BinnedSpikesInput | SpikeTimesInput) -> Tensor`: Neural data → discrete tokens
- `decode(tokens: Tensor) -> Tensor`: Discrete tokens → reconstructed data
- `forward(x: BinnedSpikesInput | SpikeTimesInput) -> TokenizerOutput`: Full forward pass returning a `TokenizerOutput` with:
  - `tokens`: Discrete token indices
  - `reconstruction`: Reconstructed data
  - Optional: `aux_loss` (extra losses for this algorithm), `latents`, etc.

## Examples

See:

- `src/ntt/contrib/tokenizer/identity_tokenizer.py` - Minimal example
- `src/ntt/contrib/tokenizer/simple_vae.py` - Realistic example
- `examples/tokenizer_demo.py` - Usage demonstration
- `src/ntt/contrib/README.md` - Contributor guide

## Benefits

1. **Type Safety**: Pydantic validates configs at instantiation
2. **Reproducibility**: Specs are serializable and version-controllable
3. **Modularity**: Contributors work in isolated files
4. **Composability**: Hydra configs can be mixed and matched
5. **Introspection**: Datamodule can query tokenizer requirements
6. **Extensibility**: Easy to add new tokenizers without modifying core code

## Future Enhancements

Potential extensions to consider:

1. **Metrics**: Standard tokenizer evaluation metrics
2. **Multi-tokenizer**: Support for ensemble or hierarchical tokenizers
