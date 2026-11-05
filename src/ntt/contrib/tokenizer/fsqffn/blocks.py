"""Feed-forward network blocks for FSQFFN tokenizer.

Implements gated feed-forward layers with residual connections for encoding
and decoding calcium trace data.
"""

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import nn

from .utils import (
    SupportedActivation,
    SupportedDepthInit,
    SupportedNorm,
    build_norm,
)
from .utils import (
    init_weights as init_weights_util,
)


class GatedFeedForwardLayer(nn.Module):
    """Gated feed-forward layer with residual connection.

    A gated layer with pre/post normalization, SiLU (or other) activation,
    and optional gated activation mechanism.

    Args:
        d_model: Model dimension
        mult: Multiplier for hidden dimension
        multiple_of: Round hidden dim to nearest multiple of this value
        use_projection_bias: Whether to use bias in linear projections
        norm: Type of normalization to use
        use_pre_norm: Whether to apply normalization before projection
        use_post_norm: Whether to apply normalization after projection
        activation: Type of activation function
        use_gated_activation: Whether to use gated activation (SwiGLU-style)
        use_post_projection: Whether to project back to d_model
    """

    def __init__(
        self,
        d_model: int,
        mult: float = 1.0,
        multiple_of: int = 128,
        use_projection_bias: bool = True,
        norm: SupportedNorm | None = "layer",
        use_pre_norm: bool = True,
        use_post_norm: bool = True,
        activation: SupportedActivation | None = "silu",
        use_gated_activation: bool = True,
        use_post_projection: bool = True,
    ):
        super().__init__()
        hidden_dim = int(multiple_of * ((d_model * mult + multiple_of - 1) // multiple_of))
        assert hidden_dim == d_model or use_post_projection, (
            "Post projection is needed if hidden_dim != d_model"
        )
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=use_projection_bias)
        self.gate_proj = (
            nn.Linear(d_model, hidden_dim, bias=use_projection_bias)
            if use_gated_activation
            else nn.Identity()
        )
        self.down_proj = (
            nn.Linear(hidden_dim, d_model, bias=use_projection_bias)
            if use_post_projection
            else nn.Identity()
        )
        self.activation = activation
        self.pre_norm = build_norm(norm, d_model) if use_pre_norm else nn.Identity()
        self.post_norm = build_norm(norm, d_model) if use_post_norm else nn.Identity()

        self._use_gated_activation = use_gated_activation

    def init_weights(self, init_std: float | None = None, depth: int | None = None):
        """Initialize weights with optional depth scaling."""
        # Initialize normalization and projection layers
        init_weights_util(
            (
                self.pre_norm,
                self.up_proj,
                self.gate_proj,
                self.post_norm,
            ),
            init_std,
        )
        # Initialize gate/output layers (optionally, scaled by depth)
        init_weights_util((self.down_proj,), init_std=init_std, depth=depth)

    def activation_fn(self, x: Float[torch.Tensor, "batch seq dim"]):
        """Apply activation function."""
        if self.activation == "silu":
            return F.silu(x)
        elif self.activation == "gelu":
            return F.gelu(x)
        elif self.activation is None:
            return x
        else:
            raise NotImplementedError(f"Unknown activation function: '{self.activation}'")

    def forward(self, x: Float[torch.Tensor, "batch seq dim"]):
        """Forward pass with residual connection."""
        residual = x
        x_norm = self.pre_norm(x)
        h = self.activation_fn(self.up_proj(x_norm))
        if self._use_gated_activation:
            h = h * self.gate_proj(x_norm)
        h = self.down_proj(h)
        h = self.post_norm(h)
        return h + residual


class FeedForward(nn.Module):
    """Multi-layer feed-forward network with gated activation.

    Stacks multiple GatedFeedForwardLayer modules with input/output projections.
    Adapted from the TorchTitan implementation (BSD-3-Clause; see
    ``LICENSES/torchtitan-BSD-3-Clause.txt``).

    Args:
        input_dim: Input feature dimension
        output_dim: Output feature dimension
        model_dim: Internal model dimension
        hidden_mult: Multiplier for hidden dimension
        hidden_multiple_of: Round hidden dim to nearest multiple of this value
        activation: Type of activation function
        use_gated_activation: Whether to use gated activation
        use_post_projection: Whether to project back to model_dim in each layer
        use_projection_bias: Whether to use bias in projections
        n_layers: Number of gated layers to stack
        norm: Type of normalization to use
        use_pre_norm: Whether to apply pre-normalization
        use_post_norm: Whether to apply post-normalization
        init_depth_scaling: Depth-based initialization scaling strategy
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        model_dim: int,
        hidden_mult: float = 1.0,
        hidden_multiple_of: int = 128,
        activation: SupportedActivation | None = "silu",
        use_gated_activation: bool = True,
        use_post_projection: bool = True,
        use_projection_bias: bool = True,
        n_layers: int = 1,
        norm: SupportedNorm | None = "layer",
        use_pre_norm: bool = True,
        use_post_norm: bool = True,
        init_depth_scaling: SupportedDepthInit | None = None,
    ):
        super().__init__()
        # Initialize processing layers
        self.in_proj = nn.Linear(input_dim, model_dim)
        self.proc = nn.Sequential()
        for _ in range(n_layers):
            self.proc.append(
                GatedFeedForwardLayer(
                    model_dim,
                    mult=hidden_mult,
                    multiple_of=hidden_multiple_of,
                    use_projection_bias=use_projection_bias,
                    norm=norm,
                    use_pre_norm=use_pre_norm,
                    use_post_norm=use_post_norm,
                    activation=activation,
                    use_gated_activation=use_gated_activation,
                    use_post_projection=use_post_projection,
                )
            )
        self.out_proj = nn.Linear(model_dim, output_dim)
        self.init_depth_scaling = init_depth_scaling

    def init_weights(self, init_std: float | None = None):
        """Initialize weights with optional depth scaling."""
        init_weights_util((self.in_proj,), init_std)
        depth = len(self.proc) if self.init_depth_scaling == "global" else None
        for i, layer in enumerate(self.proc):
            depth = i if self.init_depth_scaling == "current" else depth
            assert isinstance(layer, GatedFeedForwardLayer)
            layer.init_weights(init_std, depth=depth)
        init_weights_util((self.out_proj,), init_std, depth=depth)

    def forward(
        self, x: Float[torch.Tensor, "batch seq dim"]
    ) -> Float[torch.Tensor, "batch seq dim"]:
        """Forward pass through the feed-forward network."""
        h = self.in_proj(x)
        h = self.proc(h)
        h = self.out_proj(h)
        return h
