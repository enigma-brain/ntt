"""LDNS Tokenizer for neural spike data.

Implements the autoencoder-based tokenizer from the LDNS (Latent Diffusion for Neural
Spiking) paper. Uses S4 (Structured State Space) blocks for efficient temporal modeling.

Based on:
Kapoor, J., et al., 2024. "Latent diffusion for neural spiking data."

Adapted from the LDNS reference implementation, used under the MIT license:
https://github.com/mackelab/LDNS (see ``LICENSES/ldns-MIT.txt``).
"""

import math
from typing import Literal

import pydantic as pdt
import torch
from einops import rearrange
from jaxtyping import Float

from ntt.contrib.loss.sigreg import SIGReg
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

from .blocks import AutoEncoder
from .count_wrapper import CountWrapper
from .spike_adapter import SpikeHistoryAdapter, extract_timewise_histories


class LDNSTokenizerSpec(TokenizerSpec, frozen=True):  # pyright: ignore[reportGeneralTypeIssues, reportCallIssue]
    """Specification for the LDNS tokenizer.

    The LDNS tokenizer uses an S4-based autoencoder to learn a continuous latent
    representation of neural spike data. It predicts firing rates from spike counts
    and supports optional history-dependent spike generation.

    Attributes:
        KIND: Discriminator field for registry
        n_neurons: Number of neurons (input channels)
        window_length_seconds: Duration of input window in seconds
        bin_size_ms: Temporal bin size in milliseconds

        hidden_dim: Hidden dimension in the autoencoder (C)
        latent_dim: Latent space dimension (C_latent)
        num_encoder_blocks: Number of encoder blocks
        num_decoder_blocks: Number of decoder blocks (0 = just activation)
        num_lin_per_mlp: Number of linear layers per MLP in blocks
        bidirectional: Whether to use bidirectional S4 convolution

        latent_beta: Weight for latent L2 regularization
        latent_td_beta: Weight for temporal dynamics regularization
        mask_prob: Probability of coordinated dropout masking during training

        enable_spike_adapter: Whether to enable the spike history adapter
        spike_history_length: Length of spike history for adapter
        freeze_autoencoder: Whether to freeze AE weights (for Stage 2 training)
    """

    KIND: Literal["ldns"] = "ldns"

    # Data parameters
    n_neurons: int = 128
    window_length_seconds: float = 2.56  # 512 bins at 5ms
    bin_size_ms: float = 5.0

    # Architecture parameters
    hidden_dim: int = 256
    latent_dim: int = 32
    num_encoder_blocks: int = 6
    num_decoder_blocks: int = 0  # 0 = just GELU + projection
    num_lin_per_mlp: int = 2
    bidirectional: bool = False  # Causal by default
    encoder_bidirectional: bool | None = None  # Per-component override (None → use bidirectional)
    decoder_bidirectional: bool | None = None  # Per-component override (None → use bidirectional)

    # Regularization parameters
    latent_beta: float = 0.001
    latent_td_beta: float = 0.02
    latent_td_k: int = 5
    latent_variance_beta: float = 0.0  # Variance regularization (prevents mode collapse)
    latent_decorrelation_beta: float = 0.0  # Decorrelation penalty (disentangled dims)
    mask_prob: float = 0.20
    neuron_dropout_prob: float = 0.0  # Per-neuron dropout probability (0 = disabled)
    use_sigreg: bool = False
    # LeJEPA-style tradeoff: SIGReg gets beta, view alignment gets 1 - beta.
    sigreg_beta: float = 0.01
    sigreg_knots: int = 17
    sigreg_t_max: float = 3.0
    sigreg_sketch_dim: int = 256
    sigreg_seed: int = 42
    # Extra latent views; if active, they play the JEPA prediction role in the beta split.
    views: int = 1
    view_alignment_mask_prob: float = 0.20
    view_alignment_neuron_dropout_prob: float = 0.0

    # Spike adapter parameters (Stage 2)
    enable_spike_adapter: bool = False
    spike_history_length: int = 20
    freeze_autoencoder: bool = False
    max_spikes: int = 5  # Max spike count per bin (for adapter training/sampling)

    # Capability metadata
    latent_type: LatentType = LatentType.CONTINUOUS
    compress_n: bool = True  # Compresses neurons to latent
    compress_t: bool = False  # Maintains temporal resolution
    variable_n: bool = False  # Fixed neuron count
    variable_t: bool = True  # Can handle variable time lengths
    variable_dt: bool = False  # Fixed bin size
    input_type: DataType = DataType.SPIKES
    output_type: DataType = DataType.RATES  # Outputs firing rates

    # Latent dimension semantics
    # LDNS compresses neurons into C_latent channels, maintains time
    has_time_dim: bool = True  # Maintains temporal resolution
    has_unit_dim: bool = False  # Neurons compressed into latent
    has_latent_dim: bool = True  # Multi-dimensional latent per time bin

    @pdt.model_validator(mode="after")
    def _validate_view_alignment(self) -> "LDNSTokenizerSpec":
        """Require valid multi-view settings when alignment is enabled."""
        if self.views < 0:
            raise ValueError("LDNS view alignment requires `views` to be non-negative.")
        if self.views == 0 or not self.use_sigreg:
            return self
        if not 0.0 <= self.sigreg_beta <= 1.0:
            raise ValueError(
                "LDNS multi-view regularization requires `sigreg_beta` to be in [0, 1]."
            )
        if self.view_alignment_mask_prob <= 0 and self.view_alignment_neuron_dropout_prob <= 0:
            raise ValueError(
                "LDNS view alignment requires at least one non-zero corruption probability "
                "when `views` is greater than zero."
            )
        return self

    def get_vocab_size(self) -> int:
        """Continuous latent has no discrete vocabulary."""
        raise NotImplementedError("LDNS has continuous latent space, no discrete vocabulary")


@register_tokenizer
class LDNSTokenizer(BaseTokenizer):
    """LDNS tokenizer using S4 blocks for temporal modeling.

    This tokenizer:
    1. Takes binned spike counts as input
    2. Encodes to a continuous latent representation
    3. Decodes to predicted firing rates
    4. Applies softplus to ensure non-negative rates

    Training includes:
    - Coordinated dropout masking (20% of timesteps)
    - Latent regularization (L2 + temporal smoothness)
    - Optional multi-view latent alignment during Stage 1 training
    - Optional spike history adapter for Stage 2 training
    """

    spec_class = LDNSTokenizerSpec
    spec: LDNSTokenizerSpec  # Type hint for self.spec

    def __init__(self, spec: LDNSTokenizerSpec):
        super().__init__(spec)

        # Compute sequence length from spec
        n_bins = int(spec.window_length_seconds * 1000 / spec.bin_size_ms)
        n_neurons = spec.n_neurons if spec.n_neurons else 128

        # Build autoencoder
        ae = AutoEncoder(
            C_in=n_neurons,
            C=spec.hidden_dim,
            C_latent=spec.latent_dim,
            L=n_bins,
            num_blocks=spec.num_encoder_blocks,
            num_blocks_decoder=spec.num_decoder_blocks,
            num_lin_per_mlp=spec.num_lin_per_mlp,
            bidirectional=spec.bidirectional,
            encoder_bidirectional=spec.encoder_bidirectional,
            decoder_bidirectional=spec.decoder_bidirectional,
        )
        self.count_wrapper = CountWrapper(ae)

        self.sigreg: SIGReg | None = None
        self._latest_sigreg_loss: torch.Tensor | None = None
        self._latest_sigreg_full_loss: torch.Tensor | None = None
        if spec.use_sigreg:
            self.sigreg = SIGReg(
                knots=spec.sigreg_knots,
                t_max=spec.sigreg_t_max,
                sketch_dim=spec.sigreg_sketch_dim,
                seed=spec.sigreg_seed,
            )

        # Optional spike adapter (Stage 2)
        self.spike_adapter: SpikeHistoryAdapter | None = None
        if spec.enable_spike_adapter:
            self.spike_adapter = SpikeHistoryAdapter(
                n_neurons=n_neurons,
                history_length=spec.spike_history_length,
            )

        # Freeze autoencoder if specified (for Stage 2 training)
        if spec.freeze_autoencoder:
            for param in self.count_wrapper.parameters():
                param.requires_grad = False

    def encode(self, x: BinnedSpikesInput) -> torch.Tensor:
        """Encode neural data to continuous latent representation.

        Args:
            x: BinnedSpikesInput with spike_matrix of shape (B, T, N)

        Returns:
            Latent tensor of shape (B, T, 1, C_latent)
        """
        # Transpose: NTT (B, T, N) -> LDNS (B, N, T)
        spike_matrix = x.spike_matrix.float().transpose(-1, -2)
        z = self.count_wrapper.encode(spike_matrix)  # (B, C_latent, T)
        # Reshape to unified 4D: (B, C_latent, T) -> (B, T, 1, C_latent)
        return rearrange(z, "b c t -> b t 1 c")

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to firing rates.

        Args:
            z: Latent tensor of shape (B, T, 1, C_latent)

        Returns:
            Firing rates of shape (B, T, N)
        """
        # Handle 4D format [B, T, 1, C_latent] -> [B, C_latent, T]
        if z.ndim == 4:
            z = rearrange(z, "b t 1 c -> b c t")
        rates = self.count_wrapper.decode(z)  # (B, N, T)
        return rearrange(rates, "b n t -> b t n")  # (B, T, N)

    def reconstruct(self, z: torch.Tensor) -> ReconstructionOutput:
        """Reconstruct firing rates from latent representation.

        Args:
            z: Latent tensor of shape (B, T, 1, C_latent) — unified 4D format
               from ``encode()``.

        Returns:
            ReconstructionOutput with firing rates of shape (B, T, N)
        """
        # Convert from unified 4D [B, T, 1, C_latent] -> internal [B, C_latent, T]
        if z.ndim == 4:
            z = rearrange(z, "b t 1 c -> b c t")
        rates = self.count_wrapper.decode(z)  # (B, N, T)

        rates = rearrange(rates, "b n t -> b t n")  # (B, T, N)

        # Reconstruct what output_type declares (the BaseTokenizer contract):
        # RATES -> expected counts per bin; SPIKES -> sampled counts, through the
        # spike adapter when it is enabled.
        if self.spec.output_type != DataType.SPIKES:
            output = rates
        elif self.spec.enable_spike_adapter:
            output = self.sample_spike_counts_with_history(rates)
        else:
            output = self.sample_spike_counts(rates)
        data_type = self.spec.output_type

        return ReconstructionOutput(
            reconstruction=output,
            data_type=data_type,
            bin_size_seconds=self.spec.bin_size_ms / 1000.0,
        )

    def _apply_input_corruptions(
        self,
        spike_matrix: torch.Tensor,
        *,
        mask_prob: float,
        neuron_dropout_prob: float,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply LDNS input corruptions without mutating the source spikes.

        Returns:
            A tuple of (corrupted_signal, neuron_mask).  ``neuron_mask`` is
            ``None`` when ``neuron_dropout_prob`` is zero; otherwise it is a
            float tensor of shape ``(B, N, 1)`` with 0/1 values indicating
            which neuron channels were kept.
        """
        input_signal = spike_matrix

        if mask_prob > 0:
            # Mask across timesteps (same mask for all neurons in a sample).
            mask = (
                torch.rand(
                    spike_matrix.shape[0], 1, spike_matrix.shape[2], device=spike_matrix.device
                )
                > mask_prob
            ).float()
            # Scale visible values to maintain expectation.
            input_signal = input_signal * mask / (1 - mask_prob)

        neuron_mask: torch.Tensor | None = None
        if neuron_dropout_prob > 0:
            # Zeroes entire neuron channels to simulate missing recordings.
            batch_size, n_neurons, _num_bins = input_signal.shape
            neuron_mask = (
                torch.rand(batch_size, n_neurons, 1, device=input_signal.device)
                > neuron_dropout_prob
            ).float()
            input_signal = input_signal * neuron_mask

        return input_signal, neuron_mask

    def _make_view_alignment_input(self, spike_matrix: torch.Tensor) -> torch.Tensor:
        """Create one aligned LDNS view with independently sampled corruptions."""
        corrupted, _neuron_mask = self._apply_input_corruptions(
            spike_matrix,
            mask_prob=self.spec.view_alignment_mask_prob,
            neuron_dropout_prob=self.spec.view_alignment_neuron_dropout_prob,
        )
        return corrupted

    def _compute_view_alignment_loss(
        self,
        reference_z: torch.Tensor,
        aligned_z: torch.Tensor,
    ) -> torch.Tensor:
        """Match latent trajectories from two corrupted views of the same window."""
        return (reference_z - aligned_z).pow(2).mean()

    def _compute_view_alignment_aux_loss(
        self,
        spike_matrix: torch.Tensor,
        reference_z: torch.Tensor,
        views: int = 1,
    ) -> torch.Tensor:
        """Encode independently corrupted views and align them to the primary latent."""
        if views <= 0:
            return reference_z.new_zeros(())

        aligned_z = []
        for _ in range(views):
            aligned_input = self._make_view_alignment_input(spike_matrix)
            aligned_z.append(self.count_wrapper.encode(aligned_input))
        aligned_z = torch.stack(aligned_z, dim=0)  # (views, B, C_latent, T)
        reference_z = reference_z.unsqueeze(0).expand_as(aligned_z)

        return self._compute_view_alignment_loss(reference_z, aligned_z)

    def _view_alignment_weight(self) -> float:
        """Get the view-alignment weight implied by the shared regularization budget.

        LeJEPA uses a single tradeoff hyperparameter between the predictive JEPA
        objective and SIGReg. In LDNS, the closest analogue of that predictive
        objective is matching the reference latent to independently corrupted
        latent views, so we assign that term the complementary weight
        `1 - sigreg_beta` whenever `views > 0`.
        """
        return 1.0 - self.spec.sigreg_beta

    def get_effective_pad_mask(self) -> torch.Tensor | None:
        """Return dropout-aware pad mask for the reconstruction loss.

        After ``forward()`` runs with ``neuron_dropout_prob > 0``, this
        returns a mask that excludes both padded positions *and* neurons
        that were randomly dropped.  The litmodule passes this to the loss
        so the model is not penalized for failing to reconstruct unseen
        neurons.  Returns ``None`` when no neuron dropout was applied
        (eval mode, stage 2, or ``neuron_dropout_prob == 0``).
        """
        return getattr(self, "_effective_pad_mask", None)

    def forward(self, x: BinnedSpikesInput) -> TokenizerOutput:
        """Full forward pass with auxiliary loss computation.

        Args:
            x: BinnedSpikesInput with spike_matrix of shape (B, T, N)

        Returns:
            TokenizerOutput with:
                - tokens: Continuous latent (B, T, 1, C_latent)
                - reconstruction: Predicted rates (B, T, N)
                - aux_loss: Latent regularization (Stage 1) or adapter loss (Stage 2)
        """
        # Transpose: NTT (B, T, N) -> LDNS (B, N, T)
        spike_matrix = x.spike_matrix.float().transpose(-1, -2)
        self._latest_sigreg_loss = None
        self._latest_sigreg_full_loss = None

        # Apply Stage 1 input corruptions to the reconstruction view.
        # Stage 2 uses clean frozen-AE rates, matching the original LDNS pipeline.
        self._effective_pad_mask: torch.Tensor | None = None
        if self.training and not self.spec.freeze_autoencoder:
            input_signal, neuron_mask = self._apply_input_corruptions(
                spike_matrix,
                mask_prob=self.spec.mask_prob,
                neuron_dropout_prob=self.spec.neuron_dropout_prob,
            )
            if neuron_mask is not None and x.pad_and_gap_mask is not None:
                # Build a loss-only mask on a *copy* — never mutate the batch.
                # pad_and_gap_mask is (B, T, N); neuron_mask is (B, N, 1).
                self._effective_pad_mask = x.pad_and_gap_mask.clone().bitwise_and_(
                    neuron_mask.transpose(-1, -2).bool()
                )
        else:
            input_signal = spike_matrix

        # Forward through autoencoder
        rates, z = self.count_wrapper(input_signal)

        # Compute auxiliary losses depending on training stage
        _stage2 = self.spike_adapter is not None and self.spec.freeze_autoencoder
        if _stage2 and self.training:
            # Stage 2: Train adapter using spike histories and AE rates
            aux_loss = {"adapter": self._compute_adapter_loss(spike_matrix, rates)}
        else:
            # Stage 1 (or eval): Latent regularization (returns named dict)
            aux_loss = self._compute_latent_loss(z)
            if (
                self.training
                and self.spec.use_sigreg
                and self.sigreg is not None
                and self.spec.views > 0
                and self._view_alignment_weight() > 0
            ):
                aux_loss["view_alignment"] = self._view_alignment_weight() * (
                    self._compute_view_alignment_aux_loss(spike_matrix, z, views=self.spec.views)
                )

        # Transpose back: LDNS (B, N, T) -> NTT (B, T, N)
        # Detach in Stage 2 to avoid wasteful gradient computation through frozen AE
        reconstruction = rates.transpose(-1, -2)
        if _stage2:
            reconstruction = reconstruction.detach()

        # Reshape z to unified 4D: (B, C_latent, T) -> (B, T, 1, C_latent)
        tokens_4d = rearrange(z, "b c t -> b t 1 c")

        return TokenizerOutput(
            tokens=tokens_4d,  # Continuous latent (B, T, 1, C_latent)
            reconstruction=reconstruction,
            aux_loss=aux_loss,
        )

    def _compute_latent_loss(
        self, z: Float[torch.Tensor, "batch latent_dim time"]
    ) -> dict[str, torch.Tensor]:
        """Compute LDNS latent regularization losses.

        Implements the regularizer from the original LDNS paper that penalizes:
        1. L2 norm of latents (sum of squared values)
        2. Temporal differences with decay weighting (GP prior-like loss)

        Optionally adds:
        3. Variance regularization (prevents mode collapse)
        4. Decorrelation penalty (encourages disentangled latent dimensions)

        Args:
            z: Latent tensor of shape (B, C_latent, T) — internal autoencoder
               format (NOT the unified 4D output format).

        Returns:
            Dict of named regularization losses (weighted scalars).
        """
        losses: dict[str, torch.Tensor] = {}

        # L2 regularization on latent (mean-reduced to stay scale-compatible
        # with the mean-reduced Poisson NLL reconstruction loss)
        numel = z.numel()
        losses["l2"] = self.spec.latent_beta * (z**2).sum() / numel

        # Temporal dynamics regularization with multiple time differences
        # Uses decay weighting 1/(1+i) for difference at lag i
        k = self.spec.latent_td_k  # number of time differences to consider
        temporal_difference_loss = torch.tensor(0.0, device=z.device)

        for i in range(1, k + 1):
            if z.shape[-1] > i:
                z_diff = ((z[:, :, :-i] - z[:, :, i:]) ** 2 * (1 / (1 + i))).sum() / numel
                temporal_difference_loss = temporal_difference_loss + z_diff

        losses["temporal"] = self.spec.latent_td_beta * temporal_difference_loss

        # Variance regularization: penalize variance of per-sample latent norms
        # across the batch to prevent mode collapse
        if self.spec.latent_variance_beta > 0:
            norms = z.pow(2).sum(dim=1).mean(dim=-1)  # (B,) mean squared norm per sample
            losses["variance"] = self.spec.latent_variance_beta * norms.var()

        # Decorrelation: penalize off-diagonal covariance between latent dimensions
        if self.spec.latent_decorrelation_beta > 0:
            z_flat = z.permute(0, 2, 1).reshape(-1, z.shape[1])  # (B*T, C_lat)
            z_centered = z_flat - z_flat.mean(dim=0)
            cov = (z_centered.T @ z_centered) / (z_flat.shape[0] - 1)
            off_diag = cov - torch.diag(cov.diag())
            decorr_loss = off_diag.pow(2).sum() / z.shape[1]
            losses["decorrelation"] = self.spec.latent_decorrelation_beta * decorr_loss

        # SIGReg is the anti-collapse / distribution-matching term: it pushes the
        # pooled latent samples toward the isotropic-Gaussian target used by LeJEPA.
        if self.training and self.spec.use_sigreg and self.sigreg is not None:
            sigreg_loss = self._compute_sigreg_loss(z)
            losses["sigreg"] = self.spec.sigreg_beta * sigreg_loss
            self._latest_sigreg_loss = sigreg_loss.detach()
            self._latest_sigreg_full_loss = (sigreg_loss * (z.shape[0] * z.shape[-1])).detach()

        return losses

    def _compute_sigreg_loss(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the SIGReg distribution-matching loss on pooled latent samples.

        Following LeJEPA, we treat all batch and sequence positions as samples
        from the embedding distribution and regularize them so their random
        projections match an isotropic Gaussian target.
        """
        if self.sigreg is None:
            return z.new_zeros(())
        samples = z.permute(0, 2, 1).reshape(-1, z.shape[1]).contiguous()
        return self.sigreg(samples)

    def _compute_adapter_loss(
        self, spikes_BNT: torch.Tensor, rates_BNT: torch.Tensor
    ) -> torch.Tensor:
        """Compute spike history adapter loss for Stage 2 training.

        Extracts spike history windows and trains the adapter to predict
        spike counts from history + log(rates), matching the original LDNS
        training procedure.

        Args:
            spikes_BNT: Original spike counts [B, N, T]
            rates_BNT: Predicted firing rates from AE [B, N, T]

        Returns:
            Poisson NLL loss for the adapter (scalar)
        """
        assert self.spike_adapter is not None

        # Clamp spikes at max_spikes (matches original train_history_model.py line 153)
        spikes_BNT = spikes_BNT.clamp(max=self.spec.max_spikes)

        # Compute log rates (matching original: np.log(ae_rates))
        log_rates_BNT = torch.log(rates_BNT + 1e-8)

        # Extract training windows (vectorized over timesteps)
        spike_history, current_log_rates, target_spikes = extract_timewise_histories(
            spikes_BNT, log_rates_BNT, self.spec.spike_history_length
        )

        # Compute adapter loss
        loss = self.spike_adapter.compute_loss(spike_history, current_log_rates, target_spikes)

        # Mark adapter as trained (persists in checkpoint)
        if not self.spike_adapter.is_trained:
            self.spike_adapter.is_trained = True

        return loss

    def sample_spike_counts(self, rates: torch.Tensor, max_count: int = 10) -> torch.Tensor:
        """Sample spike counts from predicted rates using Poisson distribution.

        Args:
            rates: Predicted firing rates (B, T, N), scaled to expected counts per bin
            max_count: Maximum spike count per bin (for numerical stability)

        Returns:
            Sampled spike counts (B, T, N)
        """
        return torch.poisson(rates).clamp(max=max_count).long()

    def sample_binary_spikes(
        self,
        spike_counts: torch.Tensor,
        sampling_rate_hz: float,
        uniform_spacing: bool = False,
    ) -> torch.Tensor:
        """Sample binary spikes at a higher temporal resolution.

        This places already-sampled spike counts within each original bin at the requested
        ``sampling_rate_hz`` to produce binary spikes (0/1).

        Args:
            spike_counts: Sampled spike counts (B, T, N)
            sampling_rate_hz: Target temporal resolution for binary spikes (Hz)
            uniform_spacing: If True, place spikes evenly within the bin

        Returns:
            Binary spike tensor of shape (B, T * sub_bins, N)
        """
        if sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be positive.")

        if torch.is_floating_point(spike_counts):
            rounded = spike_counts.round()
            if not torch.allclose(spike_counts, rounded, rtol=0.0, atol=1e-6):
                raise ValueError(
                    "spike_counts must contain integer values. "
                    "Run sample_spike_counts or sample_spike_counts_with_history first."
                )

        # Convert original bin size to seconds to compute sub-bin count.
        bin_size_seconds = self.spec.bin_size_ms / 1000.0
        sub_bins_float = bin_size_seconds * sampling_rate_hz
        sub_bins = int(round(sub_bins_float))
        if not math.isclose(sub_bins_float, sub_bins, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                "sampling_rate_hz must evenly divide the original bin size "
                f"({self.spec.bin_size_ms} ms)."
            )
        if sub_bins <= 0:
            raise ValueError("sampling_rate_hz must yield at least one sub-bin per bin.")

        # Ensure no more than one spike per sub-bin.
        counts = spike_counts.clamp(max=sub_bins)
        B, T, N = counts.shape
        device = spike_counts.device
        binary_spikes = torch.zeros(B, T * sub_bins, N, device=device, dtype=torch.long)

        for t in range(T):
            # Map original bin index to its sub-bin slice in the output.
            base = t * sub_bins
            counts_t = counts[:, t, :]
            for b in range(B):
                for n in range(N):
                    count = int(counts_t[b, n])
                    if count <= 0:
                        continue
                    if uniform_spacing:
                        # Evenly spaced indices across the sub-bins (deterministic).
                        indices = (
                            torch.linspace(0, sub_bins - 1, steps=count, device=device)
                            .round()
                            .long()
                        )
                    else:
                        # Uniformly random placement within the sub-bins.
                        indices = torch.randperm(sub_bins, device=device)[:count]
                    binary_spikes[b, base + indices, n] = 1

        return binary_spikes

    def sample_spike_counts_with_history(
        self,
        rates: torch.Tensor,
        initial_spikes: torch.Tensor | None = None,
        max_count: int | None = None,
    ) -> torch.Tensor:
        """Sample spike counts using history-dependent adapter.

        Requires the spike adapter to be enabled and trained.

        Args:
            rates: Predicted firing rates (B, T, N)
            initial_spikes: Initial spike history (B, N, history_length), or None for zeros
            max_count: Maximum spike count per bin. Defaults to spec.max_spikes.

        Returns:
            Sampled spike counts (B, T, N)
        """
        if self.spike_adapter is None or not self.spike_adapter.is_trained:
            raise RuntimeError(
                "Spike adapter not enabled or not trained. "
                "Set enable_spike_adapter=True and train it after training the tokenizer."
            )

        if max_count is None:
            max_count = self.spec.max_spikes

        B, T, N = rates.shape
        history_len = self.spec.spike_history_length
        device = rates.device

        # Compute log rates internally (matching training convention)
        log_rates = torch.log(rates + 1e-8)

        # Initialize history
        if initial_spikes is None:
            history = torch.zeros(B, N, history_len, device=device)
        else:
            history = initial_spikes

        spikes = []
        for t in range(T):
            # Get adapted rate for current timestep using log_rates
            adapted_rate = self.spike_adapter(history, log_rates[:, t, :])
            # Sample from Poisson with max_count clamping (matches original)
            spike_t = torch.poisson(adapted_rate).clamp(max=max_count).long()
            spikes.append(spike_t)
            # Update history (shift left, add new spike)
            history = torch.cat([history[:, :, 1:], spike_t.unsqueeze(-1).float()], dim=-1)

        return torch.stack(spikes, dim=1)  # (B, T, N)
