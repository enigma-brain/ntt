"""Data structures and enums for evaluation framework.

This module defines the core data types used throughout the evaluation pipeline.

Dimension Conventions
---------------------
The evaluation module uses (units, time) dimension ordering for all data:
- binned_spikes: shape (n_units, n_time_bins)
- binned_counts: shape (n_units, n_time_bins)
- rates_hz: shape (n_units, n_time_bins)
- spike_times: list of n_units arrays, each containing spike times

This differs from the training module's BinnedSpikesInput which uses
(batch, time, units) ordering. Use EvalData.from_binned_input() to convert
between them - it handles the transpose automatically.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import torch

from ntt.core.components.types import BinnedSpikesInput, DataType, get_data_matrix


class MetricDirection(Enum):
    """Whether higher or lower values indicate better reconstruction."""

    HIGHER_IS_BETTER = "higher"
    LOWER_IS_BETTER = "lower"


class MetricLevel(Enum):
    """Granularity at which the metric operates."""

    PER_UNIT = "per_unit"
    PER_PAIR = "per_pair"
    POPULATION = "population"


class MetricCategory(Enum):
    """Evaluation categories.

    Aligned with the paper-level grouping: four categories computed from
    reconstructed spikes (TEMPORAL_FIDELITY, SINGLE_UNIT_STATISTICS,
    POPULATION_STRUCTURE, LATENT_QUALITY) plus two computed from
    downstream models that consume / produce neural activity
    (DOWNSTREAM_ENCODING for stimulus → neural, DOWNSTREAM_DECODING
    for neural → behavior).

    The TEMPORAL_FIDELITY vs SINGLE_UNIT_STATISTICS split is the key
    one: TEMPORAL_FIDELITY metrics require time-aligned comparison of
    original ↔ reconstructed (per-bin or per-spike), whereas
    SINGLE_UNIT_STATISTICS metrics compare distributional / summary
    properties of the spike train and do not require bin alignment.
    """

    TEMPORAL_FIDELITY = "temporal_fidelity"
    SINGLE_UNIT_STATISTICS = "single_unit_statistics"
    POPULATION_STRUCTURE = "population_structure"
    LATENT_QUALITY = "latent_quality"
    DOWNSTREAM_ENCODING = "downstream_encoding"
    DOWNSTREAM_DECODING = "downstream_decoding"


@dataclass
class EvalData:
    """
    Container for neural data used in evaluation metrics.

    This class serves as the unified data container for the evaluation pipeline,
    supporting multiple data representations (spike times, binned spikes, binned counts,
    true rates)
    and providing conversion utilities from training data structures.

    Collation Recommendations
    -------------------------
    EvalData is designed to work best with collated validation data. Many metrics
    (especially ISI-based, autocorrelation, and pairwise metrics) require longer
    time segments to produce reliable estimates. Single training windows (typically
    1-5 seconds) are often too short.

    For reliable evaluation, use the validation collation utilities to aggregate
    multiple windows into longer per-session segments before creating EvalData:

        >>> from ntt.evaluation.utils import iter_collated_validation_segments
        >>> for segment in iter_collated_validation_segments(val_dataloader, tokenizer_fn):
        ...     original = EvalData.from_tensor(segment.original, ...)
        ...     reconstructed = EvalData.from_tensor(segment.reconstructed, ...)
        ...     report = pipeline.run(original, reconstructed)

    The Lightning callback handles this automatically when using full session
    evaluation (full_eval_at_training_end=True). Quick per-batch evaluation
    bypasses collation for speed but may produce noisier metric estimates.

    Dimension Convention
    --------------------
    All binned data uses (n_units, n_time_bins) shape ordering:
    - First dimension: units/neurons
    - Second dimension: time bins

    This is the natural ordering for per-unit metrics where we iterate over units.
    Note this differs from the training convention (batch, time, units).

    Data Representations
    --------------------
    Exactly one of the following should be provided:
    - spike_times: List of arrays, one per unit, containing spike times in seconds
    - binned_spikes: Binary spike indicators/counts, shape (n_units, n_bins)
    - binned_counts: Non-binary spike counts / expected counts per bin, shape (n_units, n_bins)
    - rates_hz: Firing rates in Hz, shape (n_units, n_bins)

    The data_type field indicates which representation is active. Some metrics
    require specific representations; the DataConverter class handles conversions.

    Examples
    --------
    From collated validation segment (recommended):
        >>> for segment in iter_collated_validation_segments(dataloader, tokenizer_fn):
        ...     data = EvalData.from_tensor(segment.original, data_type=DataType.SPIKES)

    From spike times:
        >>> data = EvalData.from_spike_times_list(
        ...     [[0.1, 0.5, 1.2], [0.3, 0.8]],  # Two units
        ...     duration=2.0
        ... )

    From binned spikes (single batch, for quick evaluation only):
        >>> data = EvalData.from_binned_input(binned_spikes_input, batch_index=0)

    From a PyTorch tensor:
        >>> data = EvalData.from_tensor(
        ...     tensor,  # shape (n_units, n_bins)
        ...     data_type=DataType.SPIKES,
        ...     bin_size_seconds=0.001
        ... )
    """

    # Spike times format: list of arrays, one per unit
    spike_times: list[np.ndarray] | None = None

    # Binned binary spikes: shape (n_units, n_bins)
    binned_spikes: np.ndarray | None = None

    # Non-binary spike counts / expected counts per bin: shape (n_units, n_bins)
    binned_counts: np.ndarray | None = None

    # Firing rates in Hz: shape (n_units, n_bins)
    # Warning: this is not on the same scale as ``RatesInput.rate_matrix`` in the
    # data pipeline, which currently stores per-bin binned counts / expected counts.
    rates_hz: np.ndarray | None = None

    # Metadata
    duration: float | None = None  # Total duration in seconds
    start_time: float | None = (
        None  # Start time of window in seconds (not necessarily first spike)
    )
    bin_size_seconds: float | None = None  # Bin size in seconds (None for spike times)
    n_units: int | None = None  # Number of units/neurons

    # Data type (inferred from provided data if not set)
    data_type: DataType | None = None

    # Latent representations (optional, for structure preservation metrics)
    # Shape: [N_windows, ...] — one latent vector per validation window.
    latent: np.ndarray | None = None

    # Threshold for "fine resolution" bins that can be converted to spike times
    FINE_BIN_THRESHOLD_SECONDS: float = 0.001  # 1ms

    # Memo for has_integer_counts() (a full scan of binned_counts).
    _integer_counts: bool | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        # Infer data type if not provided
        if self.data_type is None:
            if self.spike_times is not None:
                self.data_type = DataType.SPIKE_TIMES
            elif self.binned_spikes is not None:
                self.data_type = DataType.SPIKES
            elif self.binned_counts is not None or self.rates_hz is not None:
                self.data_type = DataType.RATES
            else:
                raise ValueError("At least one data format must be provided")

        # Infer n_units
        if self.n_units is None:
            if self.spike_times is not None:
                self.n_units = len(self.spike_times)
            elif self.binned_spikes is not None:
                self.n_units = self.binned_spikes.shape[0]
            elif self.binned_counts is not None:
                self.n_units = self.binned_counts.shape[0]
            elif self.rates_hz is not None:
                self.n_units = self.rates_hz.shape[0]

        # Infer duration for binned data
        if self.duration is None and self.bin_size_seconds is not None:
            if self.binned_spikes is not None:
                self.duration = self.binned_spikes.shape[1] * self.bin_size_seconds
            elif self.binned_counts is not None:
                self.duration = self.binned_counts.shape[1] * self.bin_size_seconds
            elif self.rates_hz is not None:
                self.duration = self.rates_hz.shape[1] * self.bin_size_seconds

        # Ensure spike times are sorted
        if self.spike_times is not None:
            self.spike_times = [np.sort(st) for st in self.spike_times]

    @property
    def is_fine_resolution(self) -> bool:
        """Check if binned data has fine enough resolution for spike time conversion."""
        if self.data_type == DataType.SPIKE_TIMES:
            return True
        if self.bin_size_seconds is None:
            return True
        return self.bin_size_seconds <= self.FINE_BIN_THRESHOLD_SECONDS

    @property
    def bin_size_ms(self) -> float | None:
        """Bin size in milliseconds (convenience property)."""
        if self.bin_size_seconds is None:
            return None
        return self.bin_size_seconds * 1000

    def counts_per_bin(self) -> np.ndarray | None:
        """Binned data as (expected) spike counts per bin, shape (n_units, n_bins).

        ``rates_hz`` is scaled by the bin size, so every binned representation
        lands on the same scale. Returns None for spike times or missing data.
        """
        if self.binned_spikes is not None:
            return self.binned_spikes
        if self.binned_counts is not None:
            return self.binned_counts
        if self.rates_hz is not None:
            if self.bin_size_seconds is None:
                raise ValueError("bin_size_seconds is required to convert rates_hz to counts.")
            return self.rates_hz * self.bin_size_seconds
        return None

    def has_integer_counts(self) -> bool:
        """True if ``binned_counts`` holds observed spike counts (non-negative integers),
        as opposed to expected counts. Memoized; ``binned_counts`` is not expected to change.
        """
        if self.binned_counts is None:
            return False
        if self._integer_counts is None:
            counts = self.binned_counts
            self._integer_counts = bool(np.all(counts >= 0) and np.all(counts == np.round(counts)))
        return self._integer_counts

    @classmethod
    def from_tensor(
        cls,
        tensor: torch.Tensor,
        data_type: DataType,
        bin_size_seconds: float | None = None,
        duration: float | None = None,
    ) -> "EvalData":
        """
        Create EvalData from a PyTorch tensor.

        Args:
            tensor: Neural data tensor of shape (n_units, n_bins) or (batch, n_units, n_bins).
                    Note: expects (units, time) ordering, NOT (time, units).
            data_type: Type of data in tensor (SPIKES or RATES). RATES tensors are
                    stored as ``binned_counts``: data-pipeline and tokenizer RATES carry
                    spike counts / expected counts per bin, not Hz (see ``RatesInput``).
            bin_size_seconds: Bin size in seconds (required for SPIKES and RATES)
            duration: Total duration (optional, inferred from tensor shape
                if bin_size_seconds provided)

        Returns:
            EvalData with the tensor data in (n_units, n_bins) format
        """
        arr = tensor.detach().cpu().numpy()

        # A batch dimension is handled by taking the first item only.
        if arr.ndim == 3:
            arr = arr[0]

        if data_type == DataType.SPIKE_TIMES:
            raise NotImplementedError(
                "Spike times from tensor requires custom format handling. "
                "Use from_spike_times_list() instead."
            )
        elif data_type == DataType.SPIKES:
            return cls(
                binned_spikes=arr,
                bin_size_seconds=bin_size_seconds,
                duration=duration,
                data_type=data_type,
            )
        elif data_type == DataType.RATES:
            return cls(
                binned_counts=arr,
                bin_size_seconds=bin_size_seconds,
                duration=duration,
                data_type=data_type,
            )
        else:
            raise ValueError(f"Unknown data type: {data_type}")

    @classmethod
    def from_binned_input(
        cls,
        binned_input,  # BinnedSpikesInput type, avoiding circular import
        batch_index: int = 0,
    ) -> "EvalData":
        """
        Create EvalData from a BinnedSpikesInput (training data structure).

        This method handles the dimension transpose between training and evaluation:
        - BinnedSpikesInput shape: (batch, time, units)
        - EvalData shape: (units, time)

        Args:
            binned_input: BinnedSpikesInput NamedTuple from ntt.core.components.types.
                          Must have attributes: spike_matrix, sampling_rate, duration_sec
            batch_index: Which batch element to extract (default: 0)

        Returns:
            EvalData with shape (n_units, n_time_bins)

        Raises:
            IndexError: If batch_index is out of bounds
            AttributeError: If binned_input is missing required attributes
            ValueError: If sampling_rate is invalid (zero or negative)

        Example:
            >>> from ntt.core.components.types import BinnedSpikesInput
            >>> # After getting a batch from the dataloader:
            >>> eval_data = EvalData.from_binned_input(batch, batch_index=0)
        """
        # Validate batch_index
        data_matrix = get_data_matrix(binned_input)
        batch_size = data_matrix.shape[0]
        if batch_index < 0 or batch_index >= batch_size:
            raise IndexError(
                f"batch_index {batch_index} out of bounds for batch size {batch_size}"
            )

        # Extract and transpose: (time, units) -> (units, time)
        data_array = data_matrix[batch_index].cpu().numpy().T

        # Extract sampling rate (handle both scalar and tensor)
        sampling_rate = binned_input.sampling_rate
        if isinstance(sampling_rate, torch.Tensor):
            # Handle both scalar and batch tensors
            if sampling_rate.ndim > 0:
                sampling_rate = sampling_rate[batch_index]
            sampling_rate = sampling_rate.item()
        sampling_rate = float(sampling_rate)

        if sampling_rate <= 0:
            raise ValueError(f"sampling_rate must be positive, got {sampling_rate}")

        bin_size_seconds = 1.0 / sampling_rate

        # Extract duration (handle both batch tensor and scalar)
        duration = binned_input.duration_sec
        if isinstance(duration, torch.Tensor):
            if duration.ndim > 0:
                duration = duration[batch_index]
            duration = duration.item()
        elif isinstance(duration, (list, tuple)):
            duration = duration[batch_index]
        duration = float(duration)

        if isinstance(binned_input, BinnedSpikesInput):
            return cls(
                binned_spikes=data_array,
                bin_size_seconds=bin_size_seconds,
                duration=duration,
                data_type=DataType.SPIKES,
                n_units=data_array.shape[0],
            )
        return cls(
            binned_counts=data_array,
            bin_size_seconds=bin_size_seconds,
            duration=duration,
            data_type=DataType.RATES,
            n_units=data_array.shape[0],
        )

    @classmethod
    def from_binned_batch_collated(
        cls,
        binned_input,  # BinnedSpikesInput type
        reconstructed: torch.Tensor | None = None,
    ) -> dict[str, "EvalData"] | dict[str, tuple["EvalData", "EvalData"]]:
        """
        Create EvalData by collating batch items from the same session.

        This method groups batch items by session_id and concatenates them
        along the time axis to produce longer segments suitable for evaluation.
        This is the recommended way to create EvalData for quick evaluation,
        as single batch items are often too short for reliable metric estimates.

        Args:
            binned_input: BinnedSpikesInput with shape (batch, time, units).
                          Must have session_id attribute (str, list, or tensor).
            reconstructed: Optional reconstructed tensor with same shape as
                          binned_input.spike_matrix. If provided, returns
                          tuples of (original, reconstructed) EvalData.

        Returns:
            If reconstructed is None:
                Dict mapping session_id to collated EvalData.
            If reconstructed is provided:
                Dict mapping session_id to (original_EvalData, reconstructed_EvalData).

        Example:
            >>> # Collate a full batch for evaluation
            >>> session_data = EvalData.from_binned_batch_collated(batch)
            >>> for session_id, eval_data in session_data.items():
            ...     report = pipeline.run(eval_data, reconstructed_data)

            >>> # With reconstructions
            >>> session_pairs = EvalData.from_binned_batch_collated(batch, reconstruction)
            >>> for session_id, (orig, recon) in session_pairs.items():
            ...     report = pipeline.run(orig, recon)
        """
        data_matrix = get_data_matrix(binned_input)  # (batch, time, units) or (time, units)
        data_type = (
            DataType.SPIKES if isinstance(binned_input, BinnedSpikesInput) else DataType.RATES
        )

        # Handle case where batch dimension is missing (single item)
        if data_matrix.ndim == 2:
            data_matrix = data_matrix.unsqueeze(0)

        batch_size = data_matrix.shape[0]

        # Normalize session_ids to list of strings
        session_ids = binned_input.session_id
        if isinstance(session_ids, str):
            session_ids = [session_ids] * batch_size
        elif isinstance(session_ids, torch.Tensor):
            if session_ids.ndim == 0:
                session_ids = [str(session_ids.item())] * batch_size
            else:
                session_ids = [str(s) for s in session_ids.cpu().tolist()]
        elif isinstance(session_ids, (list, tuple)):
            session_ids = [str(s) for s in session_ids]
        else:
            session_ids = [str(session_ids)] * batch_size

        # Extract sampling rate
        sampling_rate = binned_input.sampling_rate
        if isinstance(sampling_rate, torch.Tensor):
            if sampling_rate.ndim > 0:
                sampling_rate = sampling_rate[0]
            sampling_rate = sampling_rate.item()
        sampling_rate = float(sampling_rate)
        bin_size_seconds = 1.0 / sampling_rate

        # Group indices by session_id
        session_indices: dict[str, list[int]] = {}
        for idx, sid in enumerate(session_ids):
            session_indices.setdefault(sid, []).append(idx)

        results: dict[str, Any] = {}

        for session_id, indices in session_indices.items():
            # Gather and concatenate binned matrices for this session
            # Each item has shape (time, units), we concatenate along time
            session_arrays = [data_matrix[i].cpu().numpy() for i in indices]
            # Stack along time: each is (time, units), result is (total_time, units)
            collated_array = np.concatenate(session_arrays, axis=0)
            # Transpose to (units, time) for EvalData
            collated_array = collated_array.T

            total_duration = collated_array.shape[1] * bin_size_seconds

            original_kwargs: dict[str, Any] = (
                {"binned_spikes": collated_array}
                if data_type == DataType.SPIKES
                else {"binned_counts": collated_array}
            )
            original_eval = cls(
                **original_kwargs,
                bin_size_seconds=bin_size_seconds,
                duration=total_duration,
                data_type=data_type,
                n_units=collated_array.shape[0],
            )

            if reconstructed is not None:
                # Handle case where reconstructed is missing batch dimension
                recon = reconstructed
                if recon.ndim == 2:
                    recon = recon.unsqueeze(0)

                # Same collation for reconstructed
                session_recon = [recon[i].cpu().numpy() for i in indices]
                collated_recon = np.concatenate(session_recon, axis=0).T
                recon_kwargs: dict[str, Any] = (
                    {"binned_spikes": collated_recon}
                    if data_type == DataType.SPIKES
                    else {"binned_counts": collated_recon}
                )

                recon_eval = cls(
                    **recon_kwargs,
                    bin_size_seconds=bin_size_seconds,
                    duration=total_duration,
                    data_type=data_type,
                    n_units=collated_recon.shape[0],
                )
                results[session_id] = (original_eval, recon_eval)
            else:
                results[session_id] = original_eval

        return results

    @classmethod
    def from_spike_times_list(
        cls,
        spike_times: list[np.ndarray] | list[Sequence[float]],
        duration: float,
    ) -> "EvalData":
        """
        Create EvalData from a list of spike time arrays.

        Args:
            spike_times: List of arrays, one per unit. Each array contains
                        spike times in seconds for that unit.
            duration: Total recording duration in seconds

        Returns:
            EvalData with spike_times representation
        """
        # Ensure numpy arrays with float64 dtype for precision
        spike_times_arrays: list[np.ndarray] = [
            np.asarray(st, dtype=np.float64) for st in spike_times
        ]

        return cls(
            spike_times=spike_times_arrays,
            duration=duration,
            data_type=DataType.SPIKE_TIMES,
        )

    @classmethod
    def from_tokenizer_output(
        cls,
        tensor: torch.Tensor,
        spec: Any,  # TokenizerSpec type
        duration: float | None = None,
    ) -> "EvalData":
        """
        Create EvalData from tokenizer output using spec for metadata.

        Args:
            tensor: Output tensor from tokenizer, shape (n_units, n_bins)
            spec: TokenizerSpec with output_type and bin_size_seconds attributes
            duration: Override duration (otherwise inferred from tensor shape)

        Returns:
            EvalData configured according to the tokenizer spec
        """
        return cls.from_tensor(
            tensor=tensor,
            data_type=spec.output_type,
            bin_size_seconds=spec.bin_size_seconds,
            duration=duration,
        )

    def has_spike_times(self) -> bool:
        return self.spike_times is not None

    def has_binned(self) -> bool:
        return (
            self.binned_spikes is not None
            or self.binned_counts is not None
            or self.rates_hz is not None
        )


@dataclass
class MetricResult:
    """Container for a single metric's output."""

    name: str
    category: MetricCategory
    direction: MetricDirection
    level: MetricLevel
    raw_value: float | np.ndarray  # scalar for population-level, array for per-unit/per-pair
    normalized_value: float | np.ndarray | None = None  # filled in after normalization
    metadata: dict[str, Any] = field(default_factory=dict)  # optional extra info

    # Theoretical value range (lower, upper); None means unbounded
    value_range: tuple[float | None, float | None] = (None, None)

    # Track what data type was used for this computation
    source_data_type: DataType | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "name": self.name,
            "category": self.category.value,
            "direction": self.direction.value,
            "level": self.level.value,
            "raw_value": _serialize_value(self.raw_value),
            "normalized_value": _serialize_value(self.normalized_value),
            "metadata": {str(k): _serialize_value(v) for k, v in self.metadata.items()},
            "value_range": list(self.value_range),
            "source_data_type": self.source_data_type.value if self.source_data_type else None,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MetricResult":
        """Reconstruct from a serialized ``to_dict()`` dict."""
        raw = d["raw_value"]
        if isinstance(raw, list):
            raw = np.array(raw)

        normalized = d.get("normalized_value")
        if isinstance(normalized, list):
            normalized = np.array(normalized)

        metadata = {k: _deserialize_metadata_value(v) for k, v in d.get("metadata", {}).items()}

        vr = d.get("value_range", [None, None])
        source_dt = d.get("source_data_type")

        return cls(
            name=d["name"],
            category=MetricCategory(d["category"]),
            direction=MetricDirection(d["direction"]),
            level=MetricLevel(d["level"]),
            raw_value=raw,
            normalized_value=normalized,
            metadata=metadata,
            value_range=(vr[0], vr[1]),
            source_data_type=DataType(source_dt) if source_dt else None,
        )


def _deserialize_metadata_value(v: Any) -> Any:
    """Convert a JSON-deserialized metadata value back to numpy when appropriate.

    Only converts lists whose elements are all numeric (int/float) to numpy
    arrays.  Lists containing strings, dicts, None, or mixed types are left
    as plain Python lists so that truthiness checks (``if value:``) work
    without the ambiguous-truth-value error that numpy arrays raise.
    """
    if not isinstance(v, list):
        return v
    if not v:
        return v  # empty list — keep as list
    if all(isinstance(el, (int, float)) for el in v):
        return np.array(v)
    # Nested list of numbers (e.g. confusion matrix)
    if all(isinstance(el, list) for el in v) and all(
        isinstance(x, (int, float)) for el in v for x in el
    ):
        return np.array(v)
    return v


def _serialize_key(k: Any) -> str | int | float | bool | None:
    """Convert dict key to JSON-compatible type (must be hashable)."""
    if isinstance(k, (str, int, float, bool, type(None))):
        return k
    if isinstance(k, np.integer):
        return int(k)
    if isinstance(k, np.floating):
        return float(k)
    # For tuples, arrays, or other types, convert to string
    return str(k)


def _serialize_value(v: Any) -> Any:
    """Convert numpy types and nested structures to JSON-compatible Python types."""
    from collections.abc import Mapping

    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, Mapping):
        return {_serialize_key(k): _serialize_value(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_serialize_value(item) for item in v]
    return v


@dataclass
class EvaluationReport:
    """Container for full evaluation results."""

    results: dict[str, MetricResult]
    category_scores: dict[MetricCategory, float] | None = None
    config: dict[str, Any] = field(default_factory=dict)

    # Track which metrics were skipped due to data type incompatibility
    skipped_metrics: dict[str, str] = field(default_factory=dict)

    # Source data types
    original_data_type: DataType | None = None
    reconstructed_data_type: DataType | None = None

    def get_category_results(self, category: MetricCategory) -> dict[str, MetricResult]:
        return {k: v for k, v in self.results.items() if v.category == category}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "results": {name: r.to_dict() for name, r in self.results.items()},
            "category_scores": (
                {k.value: _serialize_value(v) for k, v in self.category_scores.items()}
                if self.category_scores
                else None
            ),
            "skipped_metrics": self.skipped_metrics,
            "config": _serialize_value(self.config),
            "original_data_type": (
                self.original_data_type.value if self.original_data_type else None
            ),
            "reconstructed_data_type": (
                self.reconstructed_data_type.value if self.reconstructed_data_type else None
            ),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EvaluationReport":
        """Reconstruct from a serialized dict.

        When a per-result dict has no ``name``, its dict key is used.

        ``EvaluationReport`` carries no per-session breakdown — it's a
        post-aggregation artefact. Reports loaded here cannot be
        retroactively filtered for sessions belonging to brainsets later
        excluded from production splits (see
        ``ntt.evaluation.excluded_brainsets``); the metric scores already
        bake those contributions in. To get a brainset-filtered view,
        load the per-session ``EvalArtifact`` instead, call
        ``without_brainsets(...)``, and re-aggregate via
        ``aggregate_per_session_dicts``.
        """
        results: dict[str, MetricResult] = {}
        for name, r_dict in d.get("results", {}).items():
            if "name" not in r_dict:
                r_dict = {**r_dict, "name": name}
            results[name] = MetricResult.from_dict(r_dict)

        cat_scores_raw = d.get("category_scores")
        category_scores = (
            {MetricCategory(k): v for k, v in cat_scores_raw.items()} if cat_scores_raw else None
        )
        config = d.get("config", {})

        orig_dt = d.get("original_data_type")
        recon_dt = d.get("reconstructed_data_type")

        return cls(
            results=results,
            category_scores=category_scores,
            config=config,
            skipped_metrics=d.get("skipped_metrics", {}),
            original_data_type=DataType(orig_dt) if orig_dt else None,
            reconstructed_data_type=DataType(recon_dt) if recon_dt else None,
        )
