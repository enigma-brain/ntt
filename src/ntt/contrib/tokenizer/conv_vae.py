"""Convolutional VAE tokenizer for multi-neuron binned spike data.

A 1D convolutional VAE inspired by Stable Diffusion's AutoencoderKL,
adapted for neural spike data. Uses Conv1d along the temporal dimension.

Supports two modes:
- compress_n=False (default): each neuron is encoded independently with
  shared weights (folded into batch dim). Latent shape (B, T', U, L).
- compress_n=True: all neurons are mixed as channels in the conv network.
  Latent shape (B, T', 1, L).

Temporal compression is controlled by ``downsample_factors``, a tuple of
per-stage stride values (e.g. (5, 4) → 20x compression). This is
independent of ``channel_multipliers`` which only controls width.

Reconstruction outputs firing rates for use with Poisson NLL loss.
"""

import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from ntt.core.components.types import (
    BinnedSpikesInput,
    DataType,
    LatentType,
    ReconstructionOutput,
    TokenizerOutput,
)
from ntt.core.model.tokenizer.base import (
    BaseTokenizer,
    TokenizerSpec,
    register_tokenizer,
)

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ResBlock1d(nn.Module):
    """Residual block with 1D convolutions and GroupNorm."""

    def __init__(self, channels_in: int, channels_out: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, channels_in), channels_in)
        self.conv1 = nn.Conv1d(channels_in, channels_out, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(min(32, channels_out), channels_out)
        self.conv2 = nn.Conv1d(channels_out, channels_out, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if channels_in != channels_out:
            self.shortcut = nn.Conv1d(channels_in, channels_out, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return h + self.shortcut(x)


class Downsample1d(nn.Module):
    """Temporal downsampling by a given factor via strided conv."""

    def __init__(self, channels: int, factor: int = 2):
        super().__init__()
        # kernel_size = factor + 2 gives a receptive field slightly larger than
        # the stride, with padding = 1 on each side for clean alignment.
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=factor + 2,
            stride=factor,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    """Temporal upsampling by a given factor via interpolation + conv."""

    def __init__(self, channels: int, factor: int = 2):
        super().__init__()
        self.factor = factor
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.interpolate(x, scale_factor=float(self.factor), mode="nearest")
        return self.conv(h)


# ---------------------------------------------------------------------------
# Encoder / Decoder
# ---------------------------------------------------------------------------


class Encoder1d(nn.Module):
    """1D convolutional encoder.

    Takes (B, C_in, T) and produces (B, 2*latent_channels, T').

    ``channel_multipliers`` controls the hidden width at each stage.
    ``downsample_factors`` controls temporal compression; its length must
    be <= len(channel_multipliers). Downsamples are placed after the first
    len(downsample_factors) stages.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        latent_channels: int,
        channel_multipliers: tuple[int, ...],
        downsample_factors: tuple[int, ...],
        num_res_blocks: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv_in = nn.Conv1d(
            in_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
        )

        self.blocks = nn.ModuleList()
        ch_in = hidden_channels
        for i, mult in enumerate(channel_multipliers):
            ch_out = hidden_channels * mult
            for _ in range(num_res_blocks):
                self.blocks.append(ResBlock1d(ch_in, ch_out, dropout=dropout))
                ch_in = ch_out
            # Downsample after first len(downsample_factors) stages
            if i < len(downsample_factors):
                self.blocks.append(Downsample1d(ch_out, downsample_factors[i]))

        # Middle blocks
        self.blocks.append(ResBlock1d(ch_out, ch_out, dropout=dropout))
        self.blocks.append(ResBlock1d(ch_out, ch_out, dropout=dropout))

        # Output projection
        self.norm_out = nn.GroupNorm(min(32, ch_out), ch_out)
        self.conv_out = nn.Conv1d(
            ch_out,
            2 * latent_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        for block in self.blocks:
            h = block(h)
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h


class Decoder1d(nn.Module):
    """1D convolutional decoder.

    Takes (B, latent_channels, T') and produces (B, out_channels, T).

    Mirrors the encoder: upsample factors are applied in reverse order.
    """

    def __init__(
        self,
        out_channels: int,
        hidden_channels: int,
        latent_channels: int,
        channel_multipliers: tuple[int, ...],
        upsample_factors: tuple[int, ...],
        num_res_blocks: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        ch = hidden_channels * channel_multipliers[-1]

        self.conv_in = nn.Conv1d(latent_channels, ch, kernel_size=3, padding=1)

        # Middle blocks
        self.blocks = nn.ModuleList()
        self.blocks.append(ResBlock1d(ch, ch, dropout=dropout))
        self.blocks.append(ResBlock1d(ch, ch, dropout=dropout))

        # Upsampling blocks (reverse order)
        n_stages = len(channel_multipliers)
        reversed_multipliers = list(reversed(channel_multipliers))
        # Upsample factors are reversed to mirror the encoder
        reversed_upsample = list(reversed(upsample_factors))

        for i, mult in enumerate(reversed_multipliers):
            ch_out = hidden_channels * mult
            for _ in range(num_res_blocks + 1):
                self.blocks.append(ResBlock1d(ch, ch_out, dropout=dropout))
                ch = ch_out
            # Upsample at the last len(upsample_factors) stages (in reverse)
            upsample_start = n_stages - len(upsample_factors)
            if i >= upsample_start and i < n_stages:
                factor = reversed_upsample[i - upsample_start]
                self.blocks.append(Upsample1d(ch, factor))

        self.norm_out = nn.GroupNorm(min(32, ch), ch)
        self.conv_out = nn.Conv1d(ch, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        for block in self.blocks:
            h = block(h)
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h


# ---------------------------------------------------------------------------
# Spec + Tokenizer
# ---------------------------------------------------------------------------


class ConvVAETokenizerSpec(TokenizerSpec, frozen=True):
    """Spec for the convolutional VAE tokenizer.

    Architecture parameters:
        hidden_channels: Base channel count for encoder/decoder.
        latent_channels: Number of latent channels per temporal position.
        channel_multipliers: Hidden-width multipliers per stage. Controls
            network depth and capacity but NOT temporal compression.
        downsample_factors: Per-stage temporal stride values. Length must be
            <= len(channel_multipliers). Total temporal compression =
            product of all factors. E.g. (5, 4) on T=100 gives T'=5.
        num_res_blocks: Residual blocks per stage.
        dropout: Dropout probability in residual blocks.

    Neuron handling:
        compress_n: If False (default), each neuron is encoded independently
            with shared weights (encoder in_channels=1). The latent has a
            real neuron dimension: (B, T', U, L).
            If True, all neurons are mixed as conv channels. A linear layer
            projects n_neurons -> hidden_channels at input and back at output.
            Latent shape: (B, T', 1, L).

    KL parameters (cyclical cosine annealing, Fu et al. 2019):
        kl_weight: Maximum weight for KL divergence loss.
        kl_anneal_steps: Total training steps over which cycles run.
        kl_n_cycles: Number of cosine ramp-and-hold cycles.
        kl_cycle_ratio: Fraction of each cycle spent ramping (rest holds at max).
    """

    KIND: Literal["conv_vae"] = "conv_vae"

    # Data parameters
    n_neurons: int = 64
    window_length_seconds: float = 0.5  # 500ms
    bin_size_ms: float = 5.0  # 5ms bins → 100 time bins

    # Architecture
    hidden_channels: int = 64
    latent_channels: int = 8
    channel_multipliers: tuple[int, ...] = (1, 2, 4)
    downsample_factors: tuple[int, ...] = (2, 2)  # 4x compression
    num_res_blocks: int = 2
    dropout: float = 0.0

    # KL — cyclical cosine annealing (Fu et al., 2019)
    kl_weight: float = 1e-4
    kl_anneal_steps: int = 10000  # Total steps over which cycles run
    kl_n_cycles: int = 4  # Number of cosine cycles
    kl_cycle_ratio: float = 0.5  # Fraction of each cycle spent ramping up (rest = held at max)

    # Capability metadata
    latent_type: LatentType = LatentType.CONTINUOUS
    compress_n: bool = False
    compress_t: bool = True
    variable_n: bool = False
    variable_t: bool = False
    variable_dt: bool = False
    input_type: DataType = DataType.SPIKES
    output_type: DataType = DataType.RATES

    # Latent dimension semantics
    has_time_dim: bool = True
    has_unit_dim: bool = True  # Real neuron dim when compress_n=False
    has_latent_dim: bool = True

    @property
    def _temporal_compression_factor(self) -> int:
        result = 1
        for f in self.downsample_factors:
            result *= f
        return result

    def model_post_init(self, __context: object) -> None:
        """Validate and set token_bin_size_ms."""
        super().model_post_init(__context)

        if len(self.downsample_factors) > len(self.channel_multipliers):
            raise ValueError(
                f"len(downsample_factors)={len(self.downsample_factors)} > "
                f"len(channel_multipliers)={len(self.channel_multipliers)}. "
                f"Cannot have more downsample stages than channel stages."
            )

        if self.token_bin_size_ms is None:
            computed = float(self.bin_size_ms) * self._temporal_compression_factor
            object.__setattr__(self, "token_bin_size_ms", computed)

    def get_vocab_size(self) -> int:
        raise NotImplementedError("Continuous VAE has no discrete vocabulary")


@register_tokenizer
class ConvVAETokenizer(BaseTokenizer):
    """Convolutional VAE tokenizer for multi-neuron binned spike data.

    compress_n=False: folds neurons into batch dim, encoder sees (B*U, 1, T).
        Latent: (B, T', U, L).
    compress_n=True: neurons mixed as channels, encoder sees (B, C, T).
        Latent: (B, T', 1, L).
    """

    spec_class = ConvVAETokenizerSpec
    spec: ConvVAETokenizerSpec
    neuron_proj_in: nn.Linear | None
    neuron_proj_out: nn.Linear | None
    _step: torch.Tensor

    def __init__(self, spec: ConvVAETokenizerSpec):
        super().__init__(spec)

        compression = spec._temporal_compression_factor
        n_time_bins = int(spec.window_length_seconds * 1000 / spec.bin_size_ms)
        if n_time_bins % compression != 0:
            raise ValueError(
                f"Number of time bins ({n_time_bins}) must be divisible by "
                f"temporal compression factor ({compression}). "
                f"Adjust window_length_seconds, bin_size_ms, or "
                f"downsample_factors."
            )

        # Determine encoder input / decoder output channels
        if spec.compress_n:
            self.neuron_proj_in = nn.Linear(spec.n_neurons, spec.hidden_channels)
            self.neuron_proj_out = nn.Linear(spec.hidden_channels, spec.n_neurons)
            enc_in_channels = spec.hidden_channels
            dec_out_channels = spec.hidden_channels
        else:
            # Per-neuron mode: encoder sees single-channel temporal signal
            self.neuron_proj_in = None
            self.neuron_proj_out = None
            enc_in_channels = 1
            dec_out_channels = 1

        self.encoder = Encoder1d(
            in_channels=enc_in_channels,
            hidden_channels=spec.hidden_channels,
            latent_channels=spec.latent_channels,
            channel_multipliers=spec.channel_multipliers,
            downsample_factors=spec.downsample_factors,
            num_res_blocks=spec.num_res_blocks,
            dropout=spec.dropout,
        )

        self.decoder = Decoder1d(
            out_channels=dec_out_channels,
            hidden_channels=spec.hidden_channels,
            latent_channels=spec.latent_channels,
            channel_multipliers=spec.channel_multipliers,
            upsample_factors=spec.downsample_factors,
            num_res_blocks=spec.num_res_blocks,
            dropout=spec.dropout,
        )

        # KL annealing state
        self.register_buffer("_step", torch.tensor(0, dtype=torch.long))

    @property
    def kl_weight_current(self) -> float:
        """Current KL weight with cyclical cosine annealing.

        Divides training into ``kl_n_cycles`` cycles over
        ``kl_anneal_steps``. Within each cycle, the first
        ``kl_cycle_ratio`` fraction ramps the weight from 0 → kl_weight
        following a cosine curve, and the remaining fraction holds at
        kl_weight. After all anneal steps, weight stays at kl_weight.

        Reference: Fu et al., "Cyclical Annealing Schedule", 2019.
        """
        step = self._step.item()
        spec = self.spec
        if spec.kl_anneal_steps <= 0 or spec.kl_n_cycles <= 0:
            return spec.kl_weight

        # After annealing period, hold at full weight
        if step >= spec.kl_anneal_steps:
            return spec.kl_weight

        # Which cycle are we in, and how far through it?
        cycle_length = spec.kl_anneal_steps / spec.kl_n_cycles
        tau = (step % cycle_length) / cycle_length  # 0..1 within cycle

        if tau < spec.kl_cycle_ratio:
            # Ramping phase: cosine from 0 → 1
            progress = 0.5 * (1.0 - math.cos(math.pi * tau / spec.kl_cycle_ratio))
        else:
            # Holding phase: stay at max
            progress = 1.0

        return spec.kl_weight * progress

    def _reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + std * torch.randn_like(std)
        return mu

    # ---- Input/output preparation ----

    def _prepare_input(self, x: BinnedSpikesInput) -> torch.Tensor:
        """Prepare input for the encoder.

        compress_n=False: (B, T, U) → (B*U, 1, T)  per-neuron
        compress_n=True:  (B, T, U) → (B, C, T)     neurons mixed
        """
        h = x.spike_matrix.float()  # (B, T, U)
        if self.spec.compress_n:
            # (B, T, U) → linear → (B, T, C) → (B, C, T)
            assert self.neuron_proj_in is not None
            h = self.neuron_proj_in(h)
            h = h.permute(0, 2, 1)
        else:
            # (B, T, U) → (B, U, T) → (B*U, 1, T)
            B, T, U = h.shape
            h = h.permute(0, 2, 1).reshape(B * U, 1, T)
        return h

    def _prepare_output(self, h: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Convert decoder output back to (B, T, U) rates.

        compress_n=False: (B*U, 1, T) → (B, T, U)
        compress_n=True:  (B, C, T) → (B, T, U)
        """
        if self.spec.compress_n:
            # (B, C, T) → (B, T, C) → linear → (B, T, U)
            h = h.permute(0, 2, 1)
            assert self.neuron_proj_out is not None
            h = self.neuron_proj_out(h)
        else:
            # (B*U, 1, T) → (B, U, T) → (B, T, U)
            T = h.shape[-1]
            U = h.shape[0] // batch_size
            h = h.reshape(batch_size, U, T).permute(0, 2, 1)
        return F.softplus(h)

    # ---- 4D format helpers ----

    def _latent_to_4d(self, z: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Convert encoder latent to unified (B, T', U, L).

        compress_n=False: z is (B*U, L, T') → (B, T', U, L)
        compress_n=True:  z is (B, L, T')   → (B, T', 1, L)
        """
        if self.spec.compress_n:
            # (B, L, T') → (B, T', L) → (B, T', 1, L)
            return z.permute(0, 2, 1).unsqueeze(2)
        else:
            # (B*U, L, T') → (B, U, L, T') → (B, T', U, L)
            U = z.shape[0] // batch_size
            z = z.reshape(batch_size, U, z.shape[1], z.shape[2])
            return z.permute(0, 3, 1, 2)

    def _latent_from_4d(self, z: torch.Tensor) -> torch.Tensor:
        """Convert unified (B, T', U, L) back for decoder.

        compress_n=False: (B, T', U, L) → (B*U, L, T')
        compress_n=True:  (B, T', 1, L) → (B, L, T')
        """
        if self.spec.compress_n:
            return z.squeeze(2).permute(0, 2, 1)
        else:
            # (B, T', U, L) → (B, U, L, T') → (B*U, L, T')
            B, Tp, U, L = z.shape
            return z.permute(0, 2, 3, 1).reshape(B * U, L, Tp)

    # ---- Core methods ----

    def encode(self, x: BinnedSpikesInput) -> torch.Tensor:
        """Encode to latent (B, T', U, L) or (B, T', 1, L)."""
        B = x.spike_matrix.shape[0]
        h = self._prepare_input(x)
        moments = self.encoder(h)
        mu = moments[:, : self.spec.latent_channels]
        return self._latent_to_4d(mu, B)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to firing rates (B, T, U)."""
        B = z.shape[0]
        z_conv = self._latent_from_4d(z)
        h = self.decoder(z_conv)
        return self._prepare_output(h, B)

    def reconstruct(self, tokens: torch.Tensor) -> ReconstructionOutput:
        """Decode and optionally Poisson-sample at eval time."""
        rates = self.decode(tokens)
        if not self.training:
            reconstruction = torch.poisson(rates)
            # After sampling, output is spike counts, not rates
            data_type = DataType.SPIKES
        else:
            reconstruction = rates
            data_type = self.spec.output_type
        return ReconstructionOutput(
            reconstruction=reconstruction,
            data_type=data_type,
            bin_size_seconds=self.spec.bin_size_ms / 1000.0,
        )

    def forward(self, x: BinnedSpikesInput) -> TokenizerOutput:
        """Full forward pass with KL loss computation."""
        if self.training:
            self._step += 1

        B = x.spike_matrix.shape[0]
        h = self._prepare_input(x)
        moments = self.encoder(h)

        mu = moments[:, : self.spec.latent_channels]
        logvar = moments[:, self.spec.latent_channels :]
        logvar = logvar.clamp(-30.0, 20.0)

        z = self._reparameterize(mu, logvar)

        h_dec = self.decoder(z)
        rates = self._prepare_output(h_dec, B)

        # KL: sum over latent channels, mean over batch (and neurons/time)
        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss.sum(dim=1).mean()

        tokens = self._latent_to_4d(z, B)

        return TokenizerOutput(
            tokens=tokens,
            reconstruction=rates,
            aux_loss={"kl": self.kl_weight_current * kl_loss},
        )
