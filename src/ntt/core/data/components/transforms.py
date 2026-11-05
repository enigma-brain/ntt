import hashlib
import math
import sys
from typing import Any

import numpy as np
import torch
from jaxtyping import Float, Float64, Int, Num
from pydantic import BaseModel, Field
from temporaldata import ArrayDict, Data, Interval, RegularTimeSeries
from torch import Tensor, as_tensor

from ntt.core.components.dist import Dist
from ntt.core.components.types import (
    BinnedSpikesInput,
    DataType,
    RatesInput,
    SamplingStrategy,
    SpikeTimesInput,
    TokenizerInputType,
)
from ntt.core.data.components.behavior_transforms import (
    _extract_behavioral_fields,
    _extract_behavioral_intervals,
)
from ntt.core.data.components.dynamic import (
    dyn,
    pad2d_field,
    pad_field,
    track_mask2d_field,
    track_mask_field,
)

"""
TODO: Adapt variable window size transform from torch_brain
TODO: Define base configuration and Factory for serialization of transforms
"""

# Float slack, in units of the system epsilon, allowed when asking whether one
# duration divides another exactly. `sys.float_info.epsilon` is the representable
# resolution *at 1.0*, so it has to be scaled by the magnitude of the timestamps
# whose rounding it stands in for; unscaled it is already a no-op at the
# magnitudes involved here (`419.9 / 0.1 == 4198.999999999999`, and the ULP at
# 4199 is 9.1e-13, four thousand times the unscaled epsilon). Four epsilons of
# headroom covers the roundings in `end - start` and in
# `patch_length * bin_size_ms / 1000`.
_FLOAT_SLACK_EPSILONS = 4


def as_tensor_64(array: Num[np.ndarray, "*"]) -> Float64[Tensor, "*"]:
    return as_tensor(array, dtype=torch.float64)


def _get_domain_bounds(domain: Interval) -> tuple[float, float]:
    intervals = _get_domain_intervals(domain)
    if len(intervals) == 1:
        return intervals[0]

    # Multi-interval domains occur with gaps; use full bounds for binning.
    starts, ends = zip(*intervals, strict=True)
    return float(min(starts)), float(max(ends))


def _get_domain_intervals(domain: Interval) -> list[tuple[float, float]]:
    start = np.asarray(dyn(domain).start, dtype=float)
    end = np.asarray(dyn(domain).end, dtype=float)
    if start.size != end.size:
        raise ValueError(f"Interval start/end size mismatch: start={start.size}, end={end.size}.")
    return list(zip(start.tolist(), end.tolist(), strict=True))


def _pad_unit_field(value: np.ndarray, n_pad: int) -> np.ndarray:
    if n_pad <= 0:
        return value
    # Preserve all unit metadata fields when padding neurons.
    pad_shape = (n_pad, *value.shape[1:])
    if value.dtype.kind in ("U", "S"):
        pad_values = np.full(pad_shape, "", dtype=value.dtype)
    elif value.dtype.kind == "O":
        pad_values = np.full(pad_shape, None, dtype=object)
    else:
        pad_values = np.zeros(pad_shape, dtype=value.dtype)
    return np.concatenate([value, pad_values], axis=0)


class VariableWindowCrop:
    """Randomly crop temporal windows to variable lengths from a distribution.

    Operates on raw temporal data (before binning). Sampled crop lengths are
    snapped to the nearest valid length that is a multiple of
    ``patch_length * bin_size_ms`` milliseconds, ensuring the downstream
    patchification layer always receives an evenly divisible time axis.

    Inspired by ``torch_brain.transforms.RandomCrop``.

    Args:
        dist: Distribution to sample crop lengths from (in seconds).
        bin_size_ms: Bin size in milliseconds.
        patch_length: Number of bins per patch (for alignment).
    """

    def __init__(self, dist: Dist, bin_size_ms: float, patch_length: int):
        self.dist = dist
        self.bin_size_sec = bin_size_ms / 1000.0
        self.patch_length = patch_length
        self.patch_duration_sec = patch_length * bin_size_ms / 1000.0

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"dist={self.dist.__class__.__name__}, "
            f"patch_duration_sec={self.patch_duration_sec})"
        )

    def __call__(self, data: Data) -> Data:
        data_start, data_end = data.start, data.end
        assert data_start is not None and data_end is not None
        full_length = data_end - data_start

        # Sample target length from distribution
        target_sec = self.dist.sample()

        # Snap to nearest multiple of patch_duration
        n_patches = max(1, round(target_sec / self.patch_duration_sec))
        crop_sec = n_patches * self.patch_duration_sec

        # Clamp to [one patch, full window]
        crop_sec = max(crop_sec, self.patch_duration_sec)
        if crop_sec > full_length:
            # Re-snap DOWN to a whole number of patches. Clamping straight to
            # `full_length` emitted an UNALIGNED crop whenever the available
            # interval was shorter than the sampled length, breaking this class's
            # documented divisibility contract for arbitrary downstream consumers.
            #
            # Consumers that assert on that contract fail hard when it is broken.
            # `pool_nuclr_latents`, for one, deliberately requires the latent time
            # to divide the input time (tests/contrib/loss/test_nuclr_distill.py::
            # test_distill_rejects_non_divisible_temporal_resolution), so with
            # patch_length=100 a truncated 4297-bin crop against 43 latent steps
            # (43 * 100 = 4300) violates it: "NuCLR latent time 43 must divide
            # evenly into input time 4297". Rare per step, since only truncated
            # crops are affected, but near-certain over a long run.
            #
            # Fixing it here rather than relaxing that guard keeps the guard able
            # to catch genuinely malformed latent shapes, which it cannot
            # distinguish from a legitimately ceil-padded window on shapes alone.
            timestamp_scale = max(abs(data_start), abs(data_end), 1.0)
            slack_sec = _FLOAT_SLACK_EPSILONS * sys.float_info.epsilon * timestamp_scale
            n_whole_patches = math.floor((full_length + slack_sec) / self.patch_duration_sec)
            if n_whole_patches >= 1:
                crop_sec = n_whole_patches * self.patch_duration_sec
            else:
                # Interval is shorter than a single patch, so no aligned crop
                # exists; emit what there is rather than over-slicing. Such short
                # intervals are normally filtered out upstream.
                crop_sec = full_length

        # Random offset within full window
        slack = full_length - crop_sec
        if slack > 1e-9:
            offset = torch.rand(1).item() * slack
        else:
            offset = 0.0

        new_start = data_start + offset
        new_end = new_start + crop_sec
        return data.slice(new_start, new_end)


class PadNeuronsToFixed:
    """Transform that pads neuron populations to a fixed size.

    This is useful for models that require a fixed number of neurons (e.g., LDNS tokenizer).
    Padding is applied by adding dummy neurons with zero spikes/counts. When used with
    binned spike counts, the spike count array is zero-padded along the neuron dimension.
    When used with spike times, no extra spikes are added but the units metadata is extended.

    The transform tracks which neurons are real vs padded through the `real_neuron_mask`
    field in `data.units`. Padded neurons are assigned sequential unit_numbers starting
    from the max existing unit_number + 1.

    Args:
        target_n_neurons: Target number of neurons to pad to. Must be >= the number of
            neurons in the input data.

    Note:
        - If data already has >= target_n_neurons, no padding is applied
        - This transform should run AFTER UnitDropout (if used) to ensure consistent neuron counts
        - When used with CreateBinnedSpikeCountsArray, run this AFTER binning to pad the
          counts array
    """

    def __init__(self, target_n_neurons: int):
        if target_n_neurons <= 0:
            raise ValueError(f"target_n_neurons must be positive, got {target_n_neurons}")
        self._target_n_neurons = target_n_neurons

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(target_n_neurons={self._target_n_neurons})"

    def __call__(self, data: Data) -> Data:
        units = dyn(data).units
        num_units = len(units)

        if num_units >= self._target_n_neurons:
            # Already at or above target, nothing to do
            return data

        n_pad = self._target_n_neurons - num_units

        # Extend units metadata with dummy neurons
        existing_unit_numbers = units.unit_number
        if len(existing_unit_numbers) > 0:
            max_unit_number = int(np.max(existing_unit_numbers))
        else:
            max_unit_number = -1

        padded_unit_numbers = np.arange(
            max_unit_number + 1,
            max_unit_number + 1 + n_pad,
            dtype=existing_unit_numbers.dtype,
        )
        new_unit_number = np.concatenate([existing_unit_numbers, padded_unit_numbers])

        unit_fields: dict[str, Any] = {}
        for key in units.keys():
            value = getattr(units, key)
            if key == "unit_number":
                unit_fields[key] = new_unit_number
                continue
            if key == "id":
                pad_ids = np.array([f"pad_{i}" for i in range(n_pad)], dtype=value.dtype)
                unit_fields[key] = np.concatenate([value, pad_ids])
                continue
            unit_fields[key] = _pad_unit_field(value, n_pad)

        if "id" not in unit_fields:
            existing_ids = np.array(
                [f"unit_{int(n)}" for n in existing_unit_numbers],
                dtype=str,
            )
            pad_ids = np.array([f"pad_{i}" for i in range(n_pad)], dtype=str)
            unit_fields["id"] = np.concatenate([existing_ids, pad_ids])

        if "real_neuron_mask" not in unit_fields:
            unit_fields["real_neuron_mask"] = np.concatenate(
                [
                    np.ones(num_units, dtype=bool),
                    np.zeros(n_pad, dtype=bool),
                ]
            )

        # Update units with new arrays
        data.units = ArrayDict(**unit_fields)

        # Pad binned spike counts if present
        if hasattr(data, "binned_spike_counts"):
            binned_spike_counts = dyn(data).binned_spike_counts
            counts: Int[np.ndarray, "n_time_bins num_units"] = binned_spike_counts.counts
            n_time_bins = counts.shape[0]

            # Pad with zeros along neuron dimension
            padded_counts = np.zeros((n_time_bins, self._target_n_neurons), dtype=counts.dtype)
            padded_counts[:, :num_units] = counts
            binned_spike_counts.counts = padded_counts

        # Spikes don't need modification - padded neurons simply have no spikes

        return data


class UnitDropout:
    r"""Augmentation that randomly drops units from the sample.

    Adapted from :py:class:`torch_brain.transforms.UnitDropout`, used under the
    Apache-2.0 license (see ``LICENSES/torch_brain-Apache-2.0.txt``).

    This transform assumes that the data has a `units` object. It works for both
    :class:`IrregularTimeSeries` and :class:`RegularTimeSeries`. For the former, it will
    drop spikes from the units that are not kept. For the latter, it will drop the
    corresponding columns from the data.

    Args:
        neuron_sel: Selection method for generating the number of neurons to keep. Can be an
          integer for a fixed number, a float in (0, 1] for a minimum fraction of units to
          keep (randomly samples between fraction and all units), or a :class:`Dist` object
          to sample from. Defaults to 100.
        max_neurons: Optional hard upper bound on the number of neurons to keep. Applied after
          sampling from neuron_sel. Useful with fractional neuron_sel to bound memory usage
          on sessions with many neurons. Defaults to None (no cap).
        reset_index: Whether to reset unit indices after dropout. Defaults to True.

    TODO: Use pydantic type validation to check that Data has the expected structure
    TODO: Support more diverse / configurable sampling strategies
    TODO: Support activity sampling based on spikes (i.e. binary bins)
    """

    def __init__(
        self,
        neuron_sel: int | float | Dist = 100,
        max_neurons: int | None = None,
        reset_index: bool = True,
        exclude_zero: bool = True,
        sampling_strategy: SamplingStrategy = SamplingStrategy.UNIFORM,
        sampling_temperature: float = 1.0,
        sampling_weight_minimum: float = 1e-3,
        sampling_use_global_stats: bool = False,
        seed: int | None = None,
        deterministic: bool = False,
    ):
        self._reset_index = reset_index
        self._neuron_sel = neuron_sel
        self._max_neurons = max_neurons
        self._exclude_zero = exclude_zero
        self._sampling_strategy = sampling_strategy
        self._sampling_temperature = sampling_temperature
        self._sampling_weight_minimum = sampling_weight_minimum
        self._sampling_use_global_stats = sampling_use_global_stats
        self._deterministic = deterministic
        self._rng = np.random.default_rng(seed)
        if deterministic:
            self._sampling_strategy = SamplingStrategy.DETERMINISTIC

    def sample_n_neurons(self, num_units: int) -> int:
        """Sample the number of neurons to keep.

        Args:
            num_units: Total number of units available in the sample.
        """
        if isinstance(self._neuron_sel, float):
            min_units = max(1, int(num_units * self._neuron_sel))
            n = int(self._rng.integers(min_units, num_units + 1))
        elif isinstance(self._neuron_sel, int):
            n = self._neuron_sel
        elif isinstance(self._neuron_sel, Dist):
            n = int(self._neuron_sel.sample())
        else:
            raise TypeError(f"`neuron_sel` has invalid type: {type(self._neuron_sel)}")

        if self._max_neurons is not None:
            n = min(n, self._max_neurons)
        return n

    def __repr__(self) -> str:
        if isinstance(self._neuron_sel, float):
            neuron_sel_str = f"{self._neuron_sel} (min fraction)"
        elif isinstance(self._neuron_sel, int):
            neuron_sel_str = f"{self._neuron_sel} (fixed)"
        elif isinstance(self._neuron_sel, Dist):
            neuron_sel_str = f"{self._neuron_sel.__class__.__name__} (distribution)"
        else:
            neuron_sel_str = str(self._neuron_sel)
        max_str = f", max_neurons={self._max_neurons}" if self._max_neurons is not None else ""
        return (
            f"{self.__class__.__name__}(neuron_sel={neuron_sel_str}{max_str}, "
            f"reset_index={self._reset_index})"
        )

    def __call__(self, data: Data) -> Data:
        # extract unit metadata and spikes object
        units = dyn(data).units  # get units from data
        # TODO: Create Spikes type
        spikes = dyn(data).spikes  # get spikes from data

        # determine number of units and number to sample
        num_units = len(units)
        assert num_units > 0, "No units available in population!"
        # Clamp to valid range
        num_units_sampled = max(1, min(self.sample_n_neurons(num_units), num_units))

        # no-op if all units are kept
        if num_units_sampled == num_units:
            return data

        # generate unit mask based on sampling strategy
        # TODO: Optionally exclude zero here
        if self._sampling_strategy == SamplingStrategy.DETERMINISTIC:
            # Take the first N neurons deterministically
            unit_mask = np.zeros(num_units, dtype=bool)
            unit_mask[:num_units_sampled] = True
        elif self._sampling_strategy == SamplingStrategy.UNIFORM:
            unit_mask = self._rng.permutation(num_units) < num_units_sampled
        elif self._sampling_strategy in (
            SamplingStrategy.LINEAR_WEIGHTED,
            SamplingStrategy.LOGARITHMIC_WEIGHTED,
            SamplingStrategy.SQUARE_ROOT_WEIGHTED,
            SamplingStrategy.SQUARED_WEIGHTED,
        ):
            # Try to extract weights from units metadata. TODO: Standardize field in exported data
            counts_attr = (
                "count"
                if hasattr(units, "count")
                else "spike_count"
                if hasattr(units, "spike_count")
                else None
            )
            if self._sampling_use_global_stats and (counts_attr is not None):
                weights = getattr(units, counts_attr).astype(np.float32)
            # Fallback to batch-wise spike counts if not present in metadata.
            else:
                # compute weights based on spike counts.
                weights = np.zeros(num_units, dtype=np.float32)
                unit_index, counts = np.unique(spikes.unit_index, return_counts=True)
                weights[unit_index] = counts
            # add weight to zero units. TODO: If non enough non-zero units, we may need to include
            #  zero units in sampling to meet num_units_sampled. Alternatively we could remove
            #  gaurantee that dropout returns fixed number of units.
            nonzero_weights_mask = weights > 0
            sufficient_nonzero = nonzero_weights_mask.sum() >= num_units_sampled
            if not self._exclude_zero or not sufficient_nonzero:
                _sampling_weight_minimum = self._sampling_weight_minimum
                if not sufficient_nonzero and _sampling_weight_minimum == 0:
                    _sampling_weight_minimum = 1e-6
                weights = np.where(nonzero_weights_mask, weights, _sampling_weight_minimum)
            # apply weighting strategy
            if self._sampling_strategy == SamplingStrategy.SQUARE_ROOT_WEIGHTED:
                weights = np.sqrt(weights + 1e-6)
            elif self._sampling_strategy == SamplingStrategy.LOGARITHMIC_WEIGHTED:
                weights = np.log(weights + 1.0)
            elif self._sampling_strategy == SamplingStrategy.SQUARED_WEIGHTED:
                weights = weights**2
            # apply temperature
            if self._sampling_temperature != 1.0:
                weights = weights ** (1.0 / self._sampling_temperature)
            # sample units based on weights
            unit_mask = np.zeros(num_units, dtype=bool)
            unit_sel = self._rng.choice(
                num_units,
                size=num_units_sampled,
                replace=False,
                p=(weights / weights.sum()),
            )
            unit_mask[unit_sel] = True
        else:
            raise NotImplementedError(
                f"Sampling strategy {self._sampling_strategy} not implemented"
            )
        return _apply_unit_mask(data, unit_mask, reset_index=self._reset_index)


def _apply_unit_mask(data: Data, unit_mask: np.ndarray, reset_index: bool = True) -> Data:
    """Keep only the units in ``unit_mask``: spikes, binned counts and, with
    ``reset_index``, the units table (spike unit indices relabelled to match)."""
    units = dyn(data).units
    spikes = dyn(data).spikes
    # apply unit mask to spikes and units
    spikes_mask = unit_mask[spikes.unit_index]
    data.spikes = spikes.select_by_mask(spikes_mask)

    # apply unit mask to binned spike counts if present
    # TODO: Create a check funciton using type validation of whether this data
    #  object has binned spike counts computed (see todo in class docstring)
    if hasattr(data, "binned_spike_counts"):
        binned_spike_counts = dyn(data).binned_spike_counts
        counts: Int[np.ndarray, "n_time_bins num_units"] = binned_spike_counts.counts
        binned_spike_counts.counts = counts[:, unit_mask]
    # if reset_index is True, relabel unit indices to selected subset
    if reset_index:
        data.units = units.select_by_mask(unit_mask)
        relabel_mask = np.zeros_like(unit_mask, dtype=int)
        relabel_mask[unit_mask] = np.arange(int(unit_mask.sum()))
        spikes = dyn(data).spikes
        spikes.unit_index = relabel_mask[spikes.unit_index]

    return data


def session_seed(base_seed: int, session_id: str) -> int:
    """Combine a base seed with a session_id for per-session reproducibility."""
    h = int(hashlib.sha256(session_id.encode()).hexdigest(), 16)
    return base_seed + h % (2**31)


def select_eval_units(
    n_units: int, max_neurons: int | None, seed: int, session_id: str
) -> np.ndarray:
    """The units an evaluation uses from a session, as sorted indices into its units table.

    All units when ``max_neurons`` is None or the session has no more than that;
    otherwise ``max_neurons`` units drawn at random with a per-session seed. Every
    evaluation path (model eval, Poisson baseline, robustness passes, the metric-time
    cap) selects through this function, so they all use the same units.
    """
    if max_neurons is None or n_units <= max_neurons:
        return np.arange(n_units)
    rng = np.random.default_rng(session_seed(seed, session_id))
    return np.sort(rng.choice(n_units, size=max_neurons, replace=False))


class EvalUnitSelectionConfig(BaseModel, frozen=True):
    """Settings for :class:`EvalUnitSelection` (see :func:`select_eval_units`)."""

    max_neurons: int | None = Field(default=None, ge=1)
    seed: int = 42


class EvalUnitSelection:
    """Eval transform that keeps the units chosen by :func:`select_eval_units`.

    Selects from the whole session's units table, so every window of a session keeps
    the same units. Never pads.
    """

    def __init__(self, config: EvalUnitSelectionConfig):
        self._config = config

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(max_neurons={self._config.max_neurons}, "
            f"seed={self._config.seed})"
        )

    def __call__(self, data: Data) -> Data:
        num_units = len(dyn(data).units)
        selected = select_eval_units(
            num_units, self._config.max_neurons, self._config.seed, str(dyn(data).session.id)
        )
        if len(selected) == num_units:
            return data
        unit_mask = np.zeros(num_units, dtype=bool)
        unit_mask[selected] = True
        return _apply_unit_mask(data, unit_mask)


class ShufflePopulation:
    """Augmentation that randomly shuffles the order of units in the population.

    This permutes the unit ordering in ``data.units``, remaps
    ``spikes.unit_index`` so each spike still refers to the correct unit, and
    permutes the columns of ``binned_spike_counts.counts`` if present.

    Args:
        seed: Optional random seed for reproducibility.
    """

    def __init__(self, seed: int | None = None, deterministic: bool = False):
        self._rng = np.random.default_rng(seed)
        self._deterministic = deterministic

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(deterministic={self._deterministic})"

    def __call__(self, data: Data) -> Data:
        if self._deterministic:
            return data

        units = dyn(data).units
        num_units = len(units)

        if num_units <= 1:
            return data

        # Random permutation: perm[new_pos] = old_pos
        perm = self._rng.permutation(num_units)

        # Inverse permutation: inv_perm[old_pos] = new_pos
        inv_perm = np.empty_like(perm)
        inv_perm[perm] = np.arange(num_units)

        # Reorder units metadata
        unit_fields = {}
        for key in units.keys():
            value = getattr(units, key)
            unit_fields[key] = value[perm]
        data.units = ArrayDict(**unit_fields)

        # Remap spike unit indices
        if hasattr(data, "spikes"):
            spikes = dyn(data).spikes
            spikes.unit_index = inv_perm[spikes.unit_index]

        # Permute binned spike counts columns
        if hasattr(data, "binned_spike_counts"):
            binned_spike_counts = dyn(data).binned_spike_counts
            binned_spike_counts.counts = binned_spike_counts.counts[:, perm]

        return data


class CreateBinnedSpikeCountsArray:
    def __init__(self, bin_size_ms: float = 1, max_spikes: int | None = None):
        self._bin_size = 1e-3 * bin_size_ms
        self._max_spikes = max_spikes

    def __repr__(self) -> str:
        bin_size_ms = self._bin_size * 1000
        max_spikes_str = f", max_spikes={self._max_spikes}" if self._max_spikes is not None else ""
        return f"{self.__class__.__name__}(bin_size_ms={bin_size_ms}{max_spikes_str})"

    def __call__(self, data: Data) -> Data:
        domain = data.domain
        assert domain is not None
        intervals = _get_domain_intervals(domain)
        start, end = _get_domain_bounds(domain)
        # TODO: Standardize window size/bin size -> num bins calculation across repo
        num_time_bins = round((end - start) / self._bin_size)
        # Update end to be exactly num_time_bins * bin_size from start
        truncated_end = start + num_time_bins * self._bin_size

        units = dyn(data).units
        num_units = len(units)
        assert num_units > 0, "No units available in population!"

        spikes = dyn(data).spikes
        unit_index: Int[np.ndarray, "num_spikes"] = spikes.unit_index  # noqa: F821, UP037
        timestamps: Float[np.ndarray, "num_spikes"] = spikes.timestamps  # noqa: F821, UP037
        # Build a valid-time mask that zero-fills gaps between intervals.
        bin_starts = start + np.arange(num_time_bins) * self._bin_size
        bin_ends = bin_starts + self._bin_size
        valid_time_mask = np.zeros(num_time_bins, dtype=bool)
        # Use small tolerance for boundary comparisons — floating-point
        # accumulation in bin_ends can exceed interval_end by ~1e-14.
        eps = self._bin_size * 1e-9
        for interval_start, interval_end in intervals:
            valid_time_mask |= (bin_starts >= interval_start - eps) & (
                bin_ends <= interval_end + eps
            )

        # Filter out spikes beyond truncated end time and outside valid bins.
        valid_mask = (timestamps >= start) & (timestamps < truncated_end)
        timestamps = timestamps[valid_mask]
        unit_index = unit_index[valid_mask]
        temporal_index = ((timestamps - start) / self._bin_size).astype(int)
        if temporal_index.size:
            in_valid_bins = valid_time_mask[temporal_index]
            temporal_index = temporal_index[in_valid_bins]
            unit_index = unit_index[in_valid_bins]

        binned_spike_counts = np.zeros((num_time_bins, num_units), dtype=int)
        np.add.at(binned_spike_counts, (temporal_index, unit_index), 1)
        if self._max_spikes is not None:
            binned_spike_counts.clip(max=self._max_spikes, out=binned_spike_counts)
        if not valid_time_mask.all():
            # Gap bins are zero-filled and masked out for loss/metrics.
            binned_spike_counts[~valid_time_mask, :] = 0

        # Update domain to reflect truncated end time
        truncated_domain = Interval(start, truncated_end)
        dyn(data).binned_spike_counts = RegularTimeSeries(
            sampling_rate=(1 / self._bin_size),
            domain=truncated_domain,
            counts=binned_spike_counts,  # type: ignore
        )
        # Persist gap mask so downstream loss can ignore invalid bins.
        dyn(data).binned_spike_counts.gap_mask = valid_time_mask

        return data


class DataToModelInput:
    """Transform that converts Data object to model input format.

    The input Data object is a single time window, one window from one session. (Not
    batched.)

    For SPIKES (binary_spikes) mode, outputs binarized spike_matrix with shape (time, units).
    For RATES mode, outputs spike counts (can be > 1) with shape (time, units).
    For SPIKE_TIMES mode, outputs spike timestamps and unit indices.
    """

    def __init__(
        self,
        mode: DataType = DataType.SPIKE_TIMES,
        extract_behavior: bool = False,
        binarize_spikes: bool = True,
    ):
        self.mode = mode
        self.extract_behavior = extract_behavior
        self.binarize_spikes = binarize_spikes

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mode={self.mode!r})"

    def __call__(self, data: Data) -> TokenizerInputType:
        units = dyn(data).units
        num_units = len(units)
        assert num_units > 0, "No units available in population!"
        _unit_ids: Int[np.ndarray, "num_units"] = units.unit_number  # noqa: UP037
        assert data.domain is not None
        start, end = _get_domain_bounds(data.domain)
        duration = end - start

        # Extract behavioral data (includes task if available)
        if self.extract_behavior:
            if not hasattr(data, "binned_spike_counts"):
                raise ValueError(
                    "Extracting behavioral data currently only supported for"
                    " pipelines converting to binned spike counts, alignment "
                    "is targeted to those. "
                )
            behavioral_data = _extract_behavioral_fields(data)
            behavioral_intervals = _extract_behavioral_intervals(data)
        else:
            behavioral_data = None
            behavioral_intervals = None

        base = {
            "unit_ids": pad_field(as_tensor(_unit_ids)),
            "unit_ids_pad_mask": track_mask_field(_unit_ids),
            "duration_sec": as_tensor(duration),
            "absolute_start": as_tensor_64(data.absolute_start),  # type: ignore
            "absolute_end": as_tensor_64(data.absolute_start + duration),  # type: ignore
            "session_id": dyn(data).session.id,
            "subject_id": dyn(data).subject.id,
            "species": dyn(data).subject.species,
            "behavioral_data": behavioral_data,
            "behavioral_intervals": behavioral_intervals,
        }

        if self.mode == DataType.SPIKE_TIMES:
            spikes = dyn(data).spikes
            _unit_index: Int[np.ndarray, "num_spikes"] = spikes.unit_index  # noqa: F821, UP037
            unit_index = pad_field(as_tensor(_unit_index))
            _timestamps: Float[np.ndarray, "num_spikes"] = spikes.timestamps  # noqa: F821, UP037
            timestamps = pad_field(as_tensor(_timestamps))
            spikes_pad_mask = track_mask_field(_unit_index)
            return SpikeTimesInput(
                **base,
                spike_unit_index=unit_index,
                spike_timestamps=timestamps,
                spike_pad_mask=spikes_pad_mask,
            )
        # TODO: Avoid pad2d for single neuron case
        elif self.mode in (DataType.SPIKES, DataType.RATES):
            binned_spike_counts_rts = dyn(data).binned_spike_counts
            counts = torch.as_tensor(binned_spike_counts_rts.counts)
            gap_mask = getattr(binned_spike_counts_rts, "gap_mask", None)
            if gap_mask is not None:
                # Use gap mask as the pad_and_gap_mask so loss ignores invalid bins.
                gap_mask = torch.as_tensor(gap_mask, dtype=torch.bool)
                gap_mask_2d = gap_mask[:, None].expand(-1, counts.shape[1])
                pad_and_gap_mask = pad2d_field(gap_mask_2d)
            else:
                pad_and_gap_mask = track_mask2d_field(counts)
            if self.mode == DataType.SPIKES:
                if self.binarize_spikes:
                    counts = (counts > 0).to(counts.dtype)
                return BinnedSpikesInput(
                    **base,
                    spike_matrix=pad2d_field(counts),
                    pad_and_gap_mask=pad_and_gap_mask,
                    sampling_rate=binned_spike_counts_rts.sampling_rate,
                )
            else:
                return RatesInput(
                    **base,
                    rate_matrix=pad2d_field(counts),
                    pad_and_gap_mask=pad_and_gap_mask,
                    sampling_rate=binned_spike_counts_rts.sampling_rate,
                )
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented!")
