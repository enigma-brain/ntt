"""Simple VAE tokenizer for single-neuron binned spike counts."""

import math
from typing import Literal

import torch
from torch import nn

from ntt.core.components.types import (
    BinnedSpikesInput,
    DataType,
    LatentType,
    TokenizerOutput,
)
from ntt.core.model.tokenizer.base import BaseTokenizer, TokenizerSpec, register_tokenizer


class SimpleVAETokenizerSpec(TokenizerSpec, frozen=True):
    """Spec for a simple single-neuron VAE tokenizer.

    Takes 1000ms of binned spike counts (1ms bins) from one neuron
    and encodes to a continuous latent via a small FC encoder-decoder.

    Algorithm-specific parameters:
        latent_dim: Dimension of the VAE latent space
        hidden_dims: Hidden layer sizes for encoder/decoder
        beta: Weight for KL divergence term (beta-VAE)
    """

    # Discriminator field
    KIND: Literal["simple_vae"] = "simple_vae"

    # Fixed architecture constraints
    n_neurons: Literal[1] = 1
    window_length_seconds: float = 1  # Must be fixed, defaults to 1000 ms
    bin_size_ms: float = 1.0  # Must be fixed, defaults to 1 ms bins

    # Capability metadata - fixed for this architecture
    latent_type: LatentType = LatentType.CONTINUOUS
    compress_n: bool = False  # Does not compress the neuron dimension
    compress_t: bool = True  # Compresses 1000 time bins -> latent
    variable_n: bool = False  # Fixed: single neuron
    variable_t: bool = False  # Fixed: 1000ms window
    variable_dt: bool = False  # Fixed: 1ms bins
    input_type: DataType = DataType.SPIKES
    output_type: DataType = DataType.SPIKES

    # Latent dimension semantics
    # VAE compresses whole window into a single latent vector per neuron
    has_time_dim: bool = False  # Whole window -> single token
    has_unit_dim: bool = False  # Single neuron (no unit dimension)
    has_latent_dim: bool = True  # Multi-dimensional latent vector

    # Algorithm-specific parameters
    latent_dim: int = 8
    hidden_dims: tuple[int, ...] = (64, 32)
    beta: float = 1.0  # KL weight for beta-VAE

    def get_vocab_size(self) -> int:
        """Continuous latent -> no discrete vocab."""
        raise NotImplementedError("Continuous VAE has no discrete vocabulary")


@register_tokenizer
class SimpleVAETokenizer(BaseTokenizer):
    """Simple VAE tokenizer for single-neuron binned spike counts.

    Encodes binned spike counts to a continuous latent space using a
    variational autoencoder with fully-connected encoder and decoder.
    """

    spec_class = SimpleVAETokenizerSpec  # Links spec to implementation
    spec: SimpleVAETokenizerSpec  # Type hint for self.spec

    def __init__(self, spec: SimpleVAETokenizerSpec):
        super().__init__(spec)  # Stores self.spec, validates capabilities

        # Build architecture using spec parameters
        input_dim = math.ceil(spec.window_length_seconds * 1000 / spec.bin_size_ms)

        # Encoder: input -> hidden layers -> (mu, logvar)
        encoder_layers = []
        in_features = input_dim
        for hidden_dim in spec.hidden_dims:
            encoder_layers.extend(
                [
                    nn.Linear(in_features, hidden_dim),
                    nn.ReLU(),
                ]
            )
            in_features = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(spec.hidden_dims[-1], spec.latent_dim)
        self.fc_logvar = nn.Linear(spec.hidden_dims[-1], spec.latent_dim)

        # Decoder: latent -> hidden layers (reversed) -> output
        decoder_layers = []
        in_features = spec.latent_dim
        for hidden_dim in reversed(spec.hidden_dims):
            decoder_layers.extend(
                [
                    nn.Linear(in_features, hidden_dim),
                    nn.ReLU(),
                ]
            )
            in_features = hidden_dim
        decoder_layers.append(nn.Linear(in_features, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: BinnedSpikesInput) -> torch.Tensor:
        """Encode binned spike counts to latent representation.

        Args:
            x: BinnedSpikesInput with spike_matrix of shape (batch, time, 1)

        Returns:
            Latent representation z of shape (batch, 1, 1, latent_dim)
        """
        # x.spike_matrix: (batch, time, 1) -> latent: (batch, latent_dim)
        h = self.encoder(x.spike_matrix.float().squeeze(-1))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        # Reparameterization trick
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        # Reshape to unified 4D: [B, 1, 1, D] (singleton time and unit dims)
        return z.unsqueeze(1).unsqueeze(2)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation back to spike counts.

        Args:
            z: Latent representation of shape (batch, 1, 1, latent_dim).

        Returns:
            Reconstructed spike counts of shape (batch, time, 1)
        """
        z = z.squeeze(1).squeeze(1)  # [B, 1, 1, D] -> [B, D]
        return self.decoder(z).unsqueeze(-1)

    def forward(self, x: BinnedSpikesInput) -> TokenizerOutput:
        """Full forward pass with auxiliary loss computation.

        Args:
            x: BinnedSpikesInput with spike_matrix of shape (batch, time, 1)

        Returns:
            TokenizerOutput with tokens (latent z), reconstruction, and KL loss
        """
        # Encode
        spike_data = x.spike_matrix.float().squeeze(-1)
        h = self.encoder(spike_data)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)

        # Reparameterization trick
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

        # Decode
        reconstruction = self.decoder(z).unsqueeze(-1)

        # Compute KL divergence (auxiliary loss intrinsic to VAE)
        # Sum over latent dim (dim 1), then mean over batch (dim 0)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        # Reshape z to unified 4D: [B, 1, 1, D] (singleton time and unit dims)
        tokens_4d = z.unsqueeze(1).unsqueeze(2)

        return TokenizerOutput(
            tokens=tokens_4d,  # For continuous VAE, "tokens" is the latent
            reconstruction=reconstruction,
            aux_loss={"kl": self.spec.beta * kl_loss},
        )
