"""Utility functions for FSQFFN tokenizer.

Provides normalization builders, weight initialization, and type definitions
that are used by the FSQFFN tokenizer components.
"""

from collections.abc import Iterable
from typing import Literal

from torch import nn

# Type definitions
SupportedNorm = Literal["layer", "layer_unbiased", "rms", "softplus"]
SupportedActivation = Literal["silu", "gelu"]
SupportedDepthInit = Literal["current", "global"] | None


def build_norm(
    norm_type: SupportedNorm | None,
    dim: int,
    **kwargs,
) -> nn.Module:
    """Build a normalization layer based on the specified type.

    Adapted from TorchTitan normalization utilities (BSD-3-Clause; see
    ``LICENSES/torchtitan-BSD-3-Clause.txt``).

    Args:
        norm_type: The type of normalization layer to build.
            Supported types: layer, layer_unbiased, rms, softplus
        dim: The dimension of the normalization layer.
        **kwargs: Additional arguments to pass to the normalization layer.

    Returns:
        The built normalization layer.

    Raises:
        NotImplementedError: If an unknown norm_type is provided.
    """
    if norm_type == "layer":
        kwargs.pop("bias", None)
        return nn.LayerNorm(dim, bias=True, **kwargs)
    elif norm_type == "layer_unbiased":
        kwargs.pop("bias", None)
        return nn.LayerNorm(dim, bias=False, **kwargs)
    elif norm_type == "rms":
        return nn.RMSNorm(dim, **kwargs)
    elif norm_type == "softplus":
        return nn.Softplus(**kwargs)
    elif norm_type is None:
        return nn.Identity()
    else:
        raise NotImplementedError(f"Unknown norm_type: '{norm_type}'")


def init_weights(
    modules_and_params: Iterable[nn.Module | nn.Parameter | None],
    init_std: float | None = None,
    depth: int | None = None,
):
    """Initialize weights for given modules using truncated normal distribution.

    This helper can be used within model initialization to standardize weight
    initialization across different modules.

    Initialization strategies adapted from weight initialization used for LLaMA
    in the Lingua repo. Parameters are initialized with a truncated normal distribution
    with a specified base standard deviation (e.g., 0.02), or with the inverse of the
    squared "input" dimension (i.e. such that for `x @ W`, the resulting activations
    maintain approximately unit variance).

    Args:
        modules_and_params: Iterable of modules or parameters to initialize
        init_std: Standard deviation for truncated normal initialization.
            If None, uses input-dimension-based scaling.
        depth: Optional depth index for depth-aware initialization scaling.
            If provided, scales the std by (2 * (depth + 1))^(-0.5).

    Raises:
        ValueError: If init_std is required but not provided (e.g., for Conv layers)
        NotImplementedError: If an unsupported module/parameter type is encountered
    """
    for mop in modules_and_params:
        if mop is None or isinstance(mop, nn.Identity):
            continue
        elif isinstance(mop, (nn.RMSNorm, nn.LayerNorm)):
            mop.reset_parameters()
        elif isinstance(mop, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            _init_std = init_std
            if _init_std is None:
                raise ValueError("`init_std` must be provided for `nn.Convnd` modules!")
            nn.init.trunc_normal_(mop.weight, std=_init_std)
            if mop.bias is not None:
                mop.bias.data.zero_()
        elif isinstance(mop, nn.Linear):
            _init_std = init_std or mop.in_features ** (-0.5)
            if depth is not None:
                _init_std *= (2 * (depth + 1)) ** (-0.5)
            nn.init.trunc_normal_(mop.weight, std=_init_std)
            if mop.bias is not None:
                mop.bias.data.zero_()
        elif isinstance(mop, nn.Embedding):
            _init_std = init_std or mop.embedding_dim ** (-0.5)
            nn.init.trunc_normal_(mop.weight, std=_init_std)
            if mop.padding_idx is not None:
                mop.weight.data[mop.padding_idx].zero_()
        elif isinstance(mop, nn.Parameter):
            # NOTE: Uses final dimension as the embedding/feature dimension
            _init_std = init_std or mop.shape[-1] ** (-0.5)
            nn.init.trunc_normal_(mop, std=_init_std)
        else:
            raise NotImplementedError(f"Unsupported module or parameter type: {type(mop)}")
