from enum import StrEnum
from typing import NamedTuple

import numpy as np
from jaxtyping import Float, Int
from torch import Tensor


class DataType(StrEnum):
    """Type of neural data representation."""

    SPIKES = "binary_spikes"  # Binary spikes at 1 ms (or lower) resolution
    RATES = "rates"  # Continuous firing rates with bin size > 1 ms
    SPIKE_TIMES = "spikes_times"  # List of continuous spike times, POYO-style


class LatentType(StrEnum):
    """Type of latent representation used by the tokenizer."""

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"


class ShufflingStrategy(StrEnum):
    """Strategy for shuffling tokens."""

    NONE = "none"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    BOTH = "both"


class SamplingStrategy(StrEnum):
    """Strategy for sampling tokens."""

    UNIFORM = "uniform"
    RANDOM = "random"
    STRATIFIED = "stratified"
    LINEAR_WEIGHTED = "linear_weighted"
    LOGARITHMIC_WEIGHTED = "logarithmic_weighted"
    SQUARE_ROOT_WEIGHTED = "square_root_weighted"
    SQUARED_WEIGHTED = "squared_weighted"
    DETERMINISTIC = "deterministic"


# ---------------------------------------------------------------------------
# Model Input Types
# ---------------------------------------------------------------------------


class SpikeTimesInput(NamedTuple):
    """Model input for DataType.SPIKE_TIMES."""

    # Spike data
    spike_unit_index: Int[Tensor, "*batch num_spikes"]
    spike_timestamps: Float[Tensor, "*batch num_spikes"]
    spike_pad_mask: Int[Tensor, "*batch num_spikes"]
    # Unit metadata
    unit_ids: Int[Tensor, "*batch num_units"]
    unit_ids_pad_mask: Int[Tensor, "*batch num_units"]
    # Session metadata
    duration_sec: Float[Tensor, "*batch"]
    session_id: str | tuple[str, ...]
    subject_id: str | tuple[str, ...]
    species: str | tuple[str, ...]
    # Absolute time extent of the real (non-padded) data in this window, in seconds.
    absolute_start: Float[Tensor, "*batch"] | None = None
    absolute_end: Float[Tensor, "*batch"] | None = None
    # Behavioral data (dict of arrays with keys like task, cursor_pos, trial_target_id, etc.)
    behavioral_data: dict[str, np.ndarray] | None = None
    # Sparse interval metadata for evaluation-time segmentation.
    behavioral_intervals: dict[str, dict[str, np.ndarray]] | None = None


class BinnedSpikesInput(NamedTuple):
    """Model input for DataType.SPIKES (binned spike counts)."""

    # Spike data
    spike_matrix: Int[Tensor, "*batch num_time_bins num_units"]
    pad_and_gap_mask: Int[Tensor, "*batch num_time_bins num_units"]
    sampling_rate: float
    # Unit metadata
    unit_ids: Int[Tensor, "*batch num_units"]
    unit_ids_pad_mask: Int[Tensor, "*batch num_units"]
    # Session metadata
    duration_sec: Float[Tensor, "*batch"]
    # Absolute time extent of the real (non-padded) data in this window, in seconds.
    session_id: str | tuple[str, ...]
    subject_id: str | tuple[str, ...]
    species: str | tuple[str, ...]
    absolute_start: Float[Tensor, "*batch"]
    absolute_end: Float[Tensor, "*batch"]
    behavioral_data: dict[str, np.ndarray] | None = None
    behavioral_intervals: dict[str, dict[str, np.ndarray]] | None = None


class RatesInput(NamedTuple):
    """Model input for DataType.RATES.

    Warning:
        In the current data pipeline, ``rate_matrix`` carries per-bin binned counts
        / expected counts, not Hz-valued firing rates. This differs from
        ``EvalData.rates_hz`` in evaluation code, which is explicitly stored in Hz.
    """

    # Spike data
    rate_matrix: Float[Tensor, "*batch num_time_bins num_units"]
    pad_and_gap_mask: Int[Tensor, "*batch num_time_bins num_units"]
    sampling_rate: float
    # Unit metadata
    unit_ids: Int[Tensor, "*batch num_units"]
    unit_ids_pad_mask: Int[Tensor, "*batch num_units"]
    # Session metadata
    duration_sec: Float[Tensor, "*batch"]
    # Absolute time extent of the real (non-padded) data in this window, in seconds.
    session_id: str | tuple[str, ...]
    subject_id: str | tuple[str, ...]
    species: str | tuple[str, ...]
    absolute_start: Float[Tensor, "*batch"] | None = None
    absolute_end: Float[Tensor, "*batch"] | None = None
    behavioral_data: dict[str, np.ndarray] | None = None
    behavioral_intervals: dict[str, dict[str, np.ndarray]] | None = None


# Union type for any tokenizer input
TokenizerInputType = SpikeTimesInput | BinnedSpikesInput | RatesInput

# Union type for binned inputs (spike counts or rates — same structure, different field names)
BinnedInputType = BinnedSpikesInput | RatesInput


def get_data_matrix(batch: BinnedInputType) -> Tensor:
    """Extract the data matrix from a BinnedSpikesInput or RatesInput.


    Returns ``spike_matrix`` or ``rate_matrix`` depending on the input type.
    Raises TypeError for unsupported types.
    """
    if isinstance(batch, BinnedSpikesInput):
        return batch.spike_matrix
    elif isinstance(batch, RatesInput):
        return batch.rate_matrix
    else:
        raise TypeError(f"Expected BinnedSpikesInput or RatesInput, got {type(batch).__name__}")


# Mapping from DataType to input type
MODEL_INPUT_TYPES = {
    DataType.SPIKE_TIMES: SpikeTimesInput,
    DataType.SPIKES: BinnedSpikesInput,
    DataType.RATES: RatesInput,
}


# ---------------------------------------------------------------------------
# Model Output Types
# ---------------------------------------------------------------------------


class TokenizerOutput(NamedTuple):
    """Output from a tokenizer's forward pass.

    The tokenizer returns tokens and reconstruction. Reconstruction loss is
    computed externally by the litmodule using a configurable LossSpec.

    Tokenizers may also return an auxiliary loss (e.g., KL divergence for VAE,
    commitment loss for VQ-VAE) that is intrinsic to the architecture.
    """

    tokens: Int[Tensor, "*batch num_tokens"] | None = None
    reconstruction: Tensor | None = (
        None  # Shape varies by tokenizer (e.g., *batch num_time_bins num_units)
    )
    aux_loss: dict[str, Float[Tensor, ""]] | Float[Tensor, ""] | None = (
        None  # Named or scalar auxiliary losses
    )
    latents: Float[Tensor, "*batch num_tokens latent_dim"] | None = (
        None  # Optional continuous latents (e.g., VAE)
    )
    codes: Int[Tensor, "*batch num_tokens latent_dim"] | None = (
        None  # Optional discrete codes (e.g., VQ-VAE)
    )
    precomputed_recon_loss: Float[Tensor, ""] | None = (
        None  # Optional tokenizer-owned reconstruction loss
    )
    aux_losses: dict[str, Float[Tensor, ""]] | None = (
        None  # Optional named auxiliary losses for logging
    )


class ReconstructionOutput(NamedTuple):
    """Output from a tokenizer's reconstruct() method.

    Bundles the reconstruction tensor with declared output metadata
    (data_type, bin_size_seconds) so downstream consumers (eval pipeline,
    callbacks) receive self-describing data.

    Importantly, the reconstruction tensor is *not* necessarily identical
    to ``forward().reconstruction``.  The ``reconstruct()`` method is an
    **inference function**: it may apply post-processing that turns the
    raw model output into data that is directly comparable with the
    original input.  For example, a Poisson-loss tokenizer's ``forward()``
    returns rate (lambda) estimates, whereas its ``reconstruct()`` should
    apply Poisson sampling so the output looks like spike counts.
    """

    reconstruction: Tensor  # Shape varies by tokenizer (e.g., *batch num_time_bins num_units)
    data_type: DataType
    bin_size_seconds: float | None  # None for spike-time tokenizers (bin_size_ms=False)
