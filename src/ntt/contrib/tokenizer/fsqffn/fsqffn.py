"""FSQFFN Tokenizer for calcium imaging or continuous neural data.

Implements a quantizing autoencoder using Finite Scalar Quantization (FSQ)
to compress temporal windows of calcium traces or continuous neural signals
into discrete tokens.

The tokenizer:
1. Takes binned continuous signals (e.g., calcium traces) as input
2. Encodes temporal windows via feed-forward networks
3. Quantizes to discrete tokens using FSQ
4. Decodes back to reconstructed signals
"""

import math
from typing import Literal

from jaxtyping import Float, Int
from torch import Tensor, nn

from ntt.core.components.types import (
    DataType,
    LatentType,
    RatesInput,
    TokenizerOutput,
)
from ntt.core.model.tokenizer.base import (
    BaseTokenizer,
    TokenizerSpec,
    register_tokenizer,
)

from .blocks import FeedForward
from .quantizer import FSQuantizer
from .utils import SupportedActivation, SupportedDepthInit, SupportedNorm


class FSQFFNTokenizerSpec(TokenizerSpec, frozen=True):
    """Specification for the FSQFFN tokenizer.

    This tokenizer compresses temporal windows of continuous neural signals
    (e.g., calcium traces) into discrete tokens using Finite Scalar Quantization.
    It's designed for single-neuron or few-neuron recordings where temporal
    dynamics are compressed into a bottleneck.

    Architecture:
        - FFN encoder: Compresses n_samples_per_token time bins to latent
        - FSQ quantizer: Discretizes latent into codebook indices
        - FFN decoder: Reconstructs original signal from quantized latent

    Attributes:
        KIND: Discriminator field for registry
        n_neurons: Number of neurons (typically 1 for calcium traces)
        window_length_seconds: Duration of input window in seconds
        bin_size_ms: Temporal bin size in milliseconds
        n_samples_per_token: Number of time bins compressed into one token

        q_levels: FSQ quantization levels (e.g., [8,5,5,5] for 2^10 codebook)
        q_dim: Latent dimension after encoder (defaults to len(q_levels))
        q_scale: Optional scale parameter for FSQ
        use_timestamps: Whether to concatenate relative timestamps to input

        ffn_model_dim: Internal model dimension for FFN blocks
        ffn_n_layers: Number of gated FFN layers
        ffn_mult: Hidden dimension multiplier
        ffn_multiple_of: Round hidden dim to multiple of this value
        ffn_activation: Activation function type
        ffn_use_gated_activation: Whether to use gated activation (SwiGLU-style)
        ffn_use_post_projection: Whether to project back to model_dim
        ffn_use_projection_bias: Whether to use bias in projections
        ffn_use_pre_norm: Whether to apply pre-normalization
        ffn_use_post_norm: Whether to apply post-normalization
        ffn_norm: Type of normalization
        ffn_init_depth_scaling: Depth-based initialization scaling

        init_std: Standard deviation for weight initialization
    """

    KIND: Literal["fsqffn"] = "fsqffn"

    # Data parameters
    n_neurons: int = 1  # Single neuron by default (calcium traces)
    window_length_seconds: float = 1.0  # 1 second window
    bin_size_ms: float = 1.0  # 1ms bins (1000 Hz sampling)

    # Architecture parameters
    q_levels: tuple[int, ...] = (8, 5, 5, 5)  # 2^10 = 1024 codebook size
    q_dim: int | None = None  # Defaults to len(q_levels)
    q_scale: float | None = None

    # FFN parameters
    ffn_model_dim: int = 128
    ffn_n_layers: int = 1
    ffn_mult: float = 1.0
    ffn_multiple_of: int = 128
    ffn_activation: SupportedActivation = "silu"
    ffn_use_gated_activation: bool = True
    ffn_use_post_projection: bool = True
    ffn_use_projection_bias: bool = True
    ffn_use_pre_norm: bool = True
    ffn_use_post_norm: bool = True
    ffn_norm: SupportedNorm = "layer"
    ffn_init_depth_scaling: SupportedDepthInit | None = None

    # Initialization
    init_std: float | None = None

    # Output activation
    output_act_beta: float = 1e-1

    # Capability metadata
    latent_type: LatentType = LatentType.DISCRETE
    compress_n: bool = False  # Does not compress neuron dimension
    compress_t: bool = True  # Compresses n_samples_per_token -> 1 token
    variable_n: bool = False  # Fixed neuron count
    variable_t: bool = False  # No variable time length (fixed window)
    variable_dt: bool = False  # Fixed bin size
    input_type: DataType = DataType.RATES  # Continuous calcium traces
    output_type: DataType = DataType.RATES  # Reconstructed traces

    def get_vocab_size(self) -> int:
        """Return the FSQ codebook size."""
        return math.prod(self.q_levels)

    def get_codes_per_token(self) -> int:
        """One FSQ index per token (the ``q_levels`` values form a single code)."""
        return 1


@register_tokenizer
class FSQFFNTokenizer(BaseTokenizer):
    """FSQFFN tokenizer using FSQ quantization and feed-forward networks.

    This tokenizer compresses temporal windows of continuous signals (e.g., calcium
    traces) into discrete tokens. It uses:
    1. FFN encoder to compress temporal data to latent representation
    2. FSQ quantizer to discretize the latent into codebook indices
    3. FFN decoder to reconstruct the original signal

    The architecture is designed for calcium imaging data where n_samples_per_token
    samples (e.g., 100 time bins) are compressed into a single discrete token.
    """

    spec_class = FSQFFNTokenizerSpec
    spec: FSQFFNTokenizerSpec  # Type hint for self.spec

    def __init__(self, spec: FSQFFNTokenizerSpec):
        super().__init__(spec)
        # Compute dimensions
        input_dim = spec.window_length_bins
        latent_dim = spec.q_dim or len(spec.q_levels)
        output_dim = spec.window_length_bins
        # FFN configuration
        ffn_kwargs = {
            "model_dim": spec.ffn_model_dim,
            "hidden_mult": spec.ffn_mult,
            "hidden_multiple_of": spec.ffn_multiple_of,
            "activation": spec.ffn_activation,
            "use_gated_activation": spec.ffn_use_gated_activation,
            "use_post_projection": spec.ffn_use_post_projection,
            "use_projection_bias": spec.ffn_use_projection_bias,
            "n_layers": spec.ffn_n_layers,
            "norm": spec.ffn_norm,
            "use_pre_norm": spec.ffn_use_pre_norm,
            "use_post_norm": spec.ffn_use_post_norm,
            "init_depth_scaling": spec.ffn_init_depth_scaling,
        }
        # Build encoder, quantizer, decoder
        self.encoder = FeedForward(input_dim=input_dim, output_dim=latent_dim, **ffn_kwargs)
        self.quantizer = FSQuantizer(
            levels=list(spec.q_levels), latent_dim=latent_dim, scale=spec.q_scale
        )
        self.decoder = FeedForward(input_dim=latent_dim, output_dim=output_dim, **ffn_kwargs)
        self.output_act = nn.Softplus(beta=spec.output_act_beta)

        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        self.encoder.init_weights(self.spec.init_std)
        self.quantizer.init_weights(self.spec.init_std)
        self.decoder.init_weights(self.spec.init_std)

    def encode(self, x: RatesInput) -> Int[Tensor, "batch 1"]:  # noqa: F821
        """Encode input to discrete token indices.

        Args:
            x: RatesInput with rate_matrix of shape (batch, window_length_bins, 1)
        Returns:
            Token indices of shape (batch, 1)
        """
        rates = x.rate_matrix.float()
        _, n_bins, n_neurons = rates.shape
        assert n_bins == self.spec.window_length_bins, "Input time bin count mismatch."
        assert n_neurons == self.spec.n_neurons == 1, "Input neuron count mismatch."

        # remove singleton neuron dimension
        rates = rates.squeeze(-1)  # (batch, window_length_bins)
        # Encode to latent
        z: Float[Tensor, "batch latent_dim"] = self.encoder(rates)
        # Quantize to get discrete indices
        tokens: Int[Tensor, "batch"] = self.quantizer(z, return_codes=False)  # noqa: F821, UP037
        return tokens.unsqueeze(1)  # (batch, 1) — single token per window

    def tokens_to_latents(self, tokens: Tensor) -> Tensor:
        """Convert integer IDs [B, 1] to continuous decoder inputs [B, 1, 1, D]."""
        if tokens.is_floating_point() or tokens.ndim != 2 or tokens.shape[1] != 1:
            raise ValueError("Expected integer token IDs with shape [B, 1]")
        codes = self.quantizer.indices_to_codes(tokens[:, 0], project_out=True)
        return codes[:, None, None, :]

    def decode(  # noqa: F821
        self, tokens: Tensor
    ) -> Float[Tensor, "batch window_length_bins 1"]:
        """Decode token IDs or continuous latents back to reconstructed signal.

        Args:
            tokens: Integer token IDs [B, 1] or continuous latents [B, 1, 1, D].

        Returns:
            Reconstructed signals of shape (batch, n_samples_per_token, n_neurons)
        """
        if tokens.is_floating_point():
            expected = (1, 1, self.quantizer.latent_dim)
            if tokens.ndim != 4 or tuple(tokens.shape[1:]) != expected:
                raise ValueError(f"Expected continuous latents [B, 1, 1, {expected[-1]}]")
            z_quantized = tokens[:, 0, 0, :]
        else:
            z_quantized = self.tokens_to_latents(tokens)[:, 0, 0, :]
        # Decode
        logits: Float[Tensor, "batch window_length_bins"] = self.decoder(z_quantized)
        reconstructed_rates: Float[Tensor, "batch window_length_bins"] = self.output_act(logits)
        # add neuron dimension back (batch, window_length_bins, 1)
        reconstructed_rates = reconstructed_rates.unsqueeze(-1)
        return reconstructed_rates

    def forward(self, x: RatesInput) -> TokenizerOutput:
        """Full forward pass with encoding, quantization, and decoding.

        Args:
            x: RatesInput with rate_matrix of shape (batch, time, 1)

        Returns:
            TokenizerOutput with:
                - tokens: Discrete indices (batch, 1)
                - reconstruction: Reconstructed signals (batch, time, 1)
                - aux_loss: None (no auxiliary loss for FSQ)
        """
        tokens = self.encode(x)
        reconstructed_rates = self.decode(tokens)

        return TokenizerOutput(
            tokens=tokens,  # Discrete token indices (batch, 1)
            reconstruction=reconstructed_rates,  # (batch, n_samples_per_token, 1)
            aux_loss=None,  # FSQ has no auxiliary loss (no commitment loss like VQ-VAE)
        )
