"""
Behavioural data extraction for 'downstream' evals.
Includes transforms, e.g., to align behavioural fields with the temporal axis.

Note: the extraction configuration is specific to Perich-Miller; reusing it
for other datasets requires adding their fields to
``BEHAVIORAL_EXTRACTION_CONFIG``.
"""

import logging
from typing import Any

import numpy as np
from temporaldata import Data

from ntt.core.data.components.dynamic import dyn

LOGGER = logging.getLogger(__name__)

# Mappings from string values to integer codes for categorical behavioral fields
# Maps (brainset, field_key) -> {string_value: int_code}
BEHAVIORAL_STRING_MAPPINGS = {
    ("perich_miller_population_2018", "trial_result"): {
        "R": 0,  # Reward/success
        "A": 1,  # Abort
        "F": 2,  # Fail
        "I": 3,  # Incomplete
        # Verified from NWB trials column metadata:
        # <raw_root>/perich_miller_population_2018/
        #   sub-C/sub-C_ses-CO-20131003_behavior+ecephys.nwb
        # nwbfile.trials["result"].description ==
        # "trial outcome, R: reward, A: abort, F: fail, I: incomplete"
    },
    # Add more string mappings as needed for other datasets
}

# Fixed codebook for movement phase labels. Kept here for locality because
# extraction code and tests both depend on these exact integer IDs.
MOVEMENT_PHASE_LABELS = {
    "inter_trial": 0,
    "hold_period": 1,
    "reach_period": 2,
    "return_period": 3,
    "random_period": 4,
}
MOVEMENT_PHASE_LABELS_REVERSE = {value: key for key, value in MOVEMENT_PHASE_LABELS.items()}


# Centralized configuration for behavioral data extraction per dataset
# Maps brainset names to extraction specifications
# Each spec is a dict: {output_key: (data_attr, field_attr)}
# where data_attr is the attribute on Data object, field_attr is the attribute to extract
# Special values:
#   - "SPECIAL_MOVEMENT_PHASE": Extract movement phases as categorical labels
#   - "SPECIAL_EXPAND_TO_TIMEBINS": Extract per-trial field and expand to time bins
#   - "SPECIAL_TRIAL_ID": Generate trial IDs from absolute start times and expand to time bins
#   - ("SPECIAL_RESAMPLE_TIMESERIES", attr, field): Extract IrregularTimeSeries and
#                                                   resample to time bins
# Each entry has one of two formats: a two-tuple (data_attr, field_attr), or a
# three-tuple ("SPECIAL_...", data_attr, field_attr) whose first item selects a
# special extraction (listed above). The special values are reusable for other
# datasets' behavioural fields.
BEHAVIORAL_EXTRACTION_CONFIG: dict[str, dict[str, str | tuple[Any, ...]]] = {
    "perich_miller_population_2018": {
        # Cursor data - resampled to token time bins
        "cursor_pos": ("SPECIAL_RESAMPLE_TIMESERIES", "cursor", "pos"),
        "cursor_vel": ("SPECIAL_RESAMPLE_TIMESERIES", "cursor", "vel"),
        # Per-trial fields expanded to time bins (using trial_start/trial_end boundaries)
        "trial_id": "SPECIAL_TRIAL_ID",
        "trial_target_id": ("SPECIAL_EXPAND_TO_TIMEBINS", "target_id"),
        "trial_target_dir": ("SPECIAL_EXPAND_TO_TIMEBINS", "target_dir"),
        "trial_result": ("SPECIAL_EXPAND_TO_TIMEBINS", "result"),
        "trial_is_valid": ("SPECIAL_EXPAND_TO_TIMEBINS", "is_valid"),
        # trial_start/trial_end are used internally for expansion, not returned
        "movement_phase": "SPECIAL_MOVEMENT_PHASE",  # Special handling for Interval objects
    },
    "churchland_shenoy_neural_2012": {
        "cursor_timestamps": ("cursor", "timestamps"),
        "cursor_pos": ("cursor", "pos"),
        "cursor_vel": ("cursor", "vel"),
        "hand_timestamps": ("hand", "timestamps"),
        "hand_pos": ("hand", "pos"),
        "hand_vel": ("hand", "vel"),
        "eye_timestamps": ("eye", "timestamps"),
        "eye_pos": ("eye", "pos"),
        "eye_vel": ("eye", "vel"),
    },
    "flint_slutzky_accurate_2012": {
        "hand_timestamps": ("hand", "timestamps"),
        "hand_pos": ("hand", "pos"),
        "hand_vel": ("hand", "vel"),
    },
    "odoherty_sabes_nonhuman_2017": {
        "cursor_timestamps": ("SPECIAL_RESAMPLE_TIMESERIES", "cursor", "timestamps"),
        "cursor_pos": ("SPECIAL_RESAMPLE_TIMESERIES", "cursor", "pos"),
        "cursor_vel": ("SPECIAL_RESAMPLE_TIMESERIES", "cursor", "vel"),
        "cursor_acc": ("SPECIAL_RESAMPLE_TIMESERIES", "cursor", "acc"),
        "cursor_target_pos": ("SPECIAL_RESAMPLE_TIMESERIES", "cursor", "target_pos"),
    },
    "cowley_v4_compact_2024": {
        # One id per stimulus-presentation interval, with NaN outside the
        # interval and an optional visual-latency extension past stimulus end.
        # 100ms extension past stim_end matches the within-block inter-stim gap
        # so every stim gets a 200ms labeled range (with last-writer-wins
        # clipping producing the same 200ms for within-block stims and for
        # last-of-block stims alike — uniform across trials).
        "image_id": ("SPECIAL_EXPAND_TO_TIMEBINS", "stimuli", "image_id", np.nan, 0.1),
    },
    # Add more datasets as needed
}

# Sparse interval metadata that should be forwarded through the model-input
# pipeline so evaluation can make segmentation decisions from exact interval
# boundaries rather than from derived dense arrays.
BEHAVIORAL_INTERVAL_CONFIG = {
    "cowley_v4_compact_2024": {
        "stimuli": {
            "fields": ("image_id",),
            "end_offset": 0.2,
        },
    },
}


def _intervals_to_categorical_labels(
    time_bins: np.ndarray,
    intervals_dict: dict[int, tuple[np.ndarray, np.ndarray]],
    fill_label: int,
) -> np.ndarray:
    """Convert multiple intervals to categorical labels per time bin.

    Args:
        time_bins: Array of time bin centers (n_bins,)
        intervals_dict: Dict mapping integer label ID to (start_times, end_times)
            tuples.
        fill_label: Label used for bins that fall into no interval.

    Returns:
        Categorical labels array (n_bins,). Interval semantics are half-open:
        start <= t < end. When intervals overlap, later assignments overwrite
        earlier ones.
    """
    labels = np.full(len(time_bins), fill_label, dtype=np.int32)

    for label_idx, (starts, ends) in intervals_dict.items():
        # For each interval, mark all time bins that fall within it
        for start, end in zip(starts, ends, strict=True):
            # Half-open intervals avoid double-assignment at exact boundaries.
            mask = (time_bins >= start) & (time_bins < end)
            labels[mask] = label_idx

    return labels


def _expand_trial_field_to_timebins(
    field_values: np.ndarray,
    trial_starts: np.ndarray,
    trial_ends: np.ndarray,
    time_bins: np.ndarray,
    fill_value: int = -1,
) -> np.ndarray:
    """Expand per-trial values to per-timebin array.

    Args:
        field_values: Per-trial values, shape (n_trials,) or (n_trials, ...).
        trial_starts: Start times of each trial, shape (n_trials,)
        trial_ends: End times of each trial, shape (n_trials,)
        time_bins: Time bin centers, shape (n_timebins,)
        fill_value: Value to use for inter-trial periods (default: -1)

    Returns:
        Array of shape (n_timebins,) or (n_timebins, ...) with trial values
        tiled across their duration. Uses half-open interval semantics:
        start <= t < end, which avoids boundary double-assignment.
    """
    # Convert booleans to int so inter-trial fill_value (-1) is distinguishable
    # from False (0) and True (1).
    if np.issubdtype(field_values.dtype, np.bool_):
        field_values = field_values.astype(np.int32)

    out_shape = (len(time_bins),) + field_values.shape[1:]
    if np.issubdtype(field_values.dtype, np.integer):
        result = np.full(out_shape, fill_value, dtype=np.int32)
    else:
        result = np.full(out_shape, float(fill_value), dtype=np.float64)

    for start, end, value in zip(trial_starts, trial_ends, field_values, strict=True):
        mask = (time_bins >= start) & (time_bins < end)
        result[mask] = value

    return result


def _expand_interval_field_to_timebins(
    field_values: np.ndarray,
    interval_starts: np.ndarray,
    interval_ends: np.ndarray,
    time_bins: np.ndarray,
    fill_value: int | float = np.nan,
) -> np.ndarray:
    """Expand per-interval values to per-timebin array.

    Unlike ``_expand_trial_field_to_timebins``, this helper supports NaN fill
    values for integer inputs by promoting the output dtype to float.
    """
    if np.issubdtype(field_values.dtype, np.bool_):
        field_values = field_values.astype(np.int32)

    out_shape = (len(time_bins),) + field_values.shape[1:]
    fill_is_nan = isinstance(fill_value, float) and np.isnan(fill_value)

    if fill_is_nan:
        result = np.full(out_shape, np.nan, dtype=np.float64)
        values_to_assign = field_values.astype(np.float64, copy=False)
    elif np.issubdtype(field_values.dtype, np.integer):
        result = np.full(out_shape, fill_value, dtype=np.int32)
        values_to_assign = field_values
    else:
        result = np.full(out_shape, float(fill_value), dtype=np.float64)
        values_to_assign = field_values.astype(np.float64, copy=False)

    for start, end, value in zip(interval_starts, interval_ends, values_to_assign, strict=True):
        mask = (time_bins >= start) & (time_bins < end)
        result[mask] = value

    return result


def extend_interval_ends_with_post_event_window(
    interval_starts: np.ndarray,
    interval_ends: np.ndarray,
    *,
    end_offset: float = 0.0,
) -> np.ndarray:
    """Extend interval ends and clamp each interval at the next start.

    A post-event response window can extend beyond the raw interval end, but
    it must not spill into the next interval.
    """
    starts = np.asarray(interval_starts, dtype=np.float64)
    ends = np.asarray(interval_ends, dtype=np.float64)
    if starts.shape != ends.shape:
        raise ValueError(
            "interval_starts and interval_ends must have identical shapes; "
            f"got {starts.shape} and {ends.shape}."
        )
    if len(ends) == 0 or end_offset == 0.0:
        return ends.copy()

    extended_ends = ends + float(end_offset)
    extended_ends[:-1] = np.minimum(extended_ends[:-1], starts[1:])
    return extended_ends


def _resample_timeseries_to_bins(
    timestamps: np.ndarray,
    values: np.ndarray,
    time_bins: np.ndarray,
    *,
    max_extrapolation_factor: float = 2.0,
) -> np.ndarray:
    """Resample a time series to match time bin centers.

    Uses linear interpolation between samples, and clamps to the nearest
    sample value for ``time_bins`` slightly past either end of the input
    sample range. Bins **farther than ``max_extrapolation_factor *
    (median sample spacing)``** past the first or last sample are NaN-masked
    — that distance threshold protects against fabricating values at
    per-window edges where a continuously-sampled series naturally has its
    first/last sample a fraction of a sample period inside the window.

    Caveat: the NaN-mask only catches missing coverage at the *edges* of
    the supplied ``timestamps``. Recording gaps that fall *inside*
    ``[timestamps[0], timestamps[-1]]`` are not detected by this helper —
    they are linearly interpolated across. Detection of internal recording
    dropouts is out of scope here.

    The caller in ``_extract_behavioral_fields`` invokes this once per data
    window. Continuously-recorded cursor data (e.g. perich_miller, 10 ms
    sample spacing) has its first/last sample near the centre of the window's
    first/last 10 ms, leaving the 1 ms bins covering the first/last few ms of
    the window outside the sample range. Clamping to the nearest sample
    (instead of NaN-masking every such bin) avoids regular NaN gaps at every
    window boundary, which downstream regression metrics would otherwise drop
    wholesale.

    Args:
        timestamps: Original timestamps of the time series, shape
            (n_samples,). Must be non-decreasing (``np.interp`` requires a
            monotonic ``xp``). Duplicate timestamps are tolerated —
            perich_miller carries a few from 10 ms timestamp rounding
            and ``np.interp`` interpolates across them fine — as long
            as they stay a minority, so the median sample spacing derived
            from ``median(np.diff(timestamps))`` remains positive. An
            all-/mostly-duplicate input (median spacing 0) has no usable time
            span, so it returns all-NaN (treated as unobserved) rather than
            raising — a short window can legitimately land on nothing but a
            duplicate pair.
        values: Values at each timestamp, shape (n_samples,) or
            (n_samples, n_dims).
        time_bins: Target time bin centers, shape (n_bins,).
        max_extrapolation_factor: Distance, expressed as a multiple of
            the median sample spacing, beyond which extrapolation is
            replaced with NaN. Defaults to 2.0.

    Returns:
        Resampled values, shape (n_bins,) or (n_bins, n_dims). Bins
        beyond ``max_extrapolation_factor`` sample periods past either
        end of the input timestamps are NaN; other bins are linearly
        interpolated or clamped to the nearest sample value. If the
        input has fewer than 2 samples we have no sample-spacing context
        to bound the extrapolation, so the returned array is entirely
        NaN — the caller's per-bin finite filter will then treat the
        window as unobserved rather than letting one sample fabricate a
        whole bin grid.

    Raises:
        AssertionError: if ``timestamps`` is not non-decreasing.
    """
    # Fewer than 2 samples → no spacing context. Return all NaN so the
    # caller's finite filter (existing per-bin handling downstream)
    # treats the window as unobserved instead of silently clamping every
    # bin to a single value.
    if len(timestamps) < 2 or len(values) == 0:
        if values.ndim == 1:
            return np.full(len(time_bins), np.nan, dtype=np.float64)
        else:
            return np.full((len(time_bins), values.shape[1]), np.nan, dtype=np.float64)

    diffs = np.diff(timestamps)
    assert np.all(diffs >= 0), (
        "_resample_timeseries_to_bins: timestamps must be non-decreasing; "
        f"saw min(diff)={float(np.min(diffs))!r}. Sort upstream before "
        "resampling."
    )

    dt_median = float(np.median(diffs))
    # Duplicate timestamps (diff == 0) are tolerated: ``np.interp`` handles
    # them fine (perich_miller carries a few from 10 ms timestamp rounding),
    # and the median stays positive as long as they are a
    # minority. A non-positive median means (nearly) every timestamp in this
    # window collides — e.g. a short window that happens to land on one of
    # perich_miller's duplicate 10 ms-rounded pairs as its only samples.
    # There is then no usable time span to bound extrapolation, exactly like
    # the <2-sample case above, so we return all-NaN and let the caller's
    # finite filter treat the window as unobserved rather than fabricate a
    # bin grid from a degenerate input.
    if dt_median <= 0:
        if values.ndim == 1:
            return np.full(len(time_bins), np.nan, dtype=np.float64)
        return np.full((len(time_bins), values.shape[1]), np.nan, dtype=np.float64)

    extrapolation_limit = max_extrapolation_factor * dt_median
    t0 = float(timestamps[0])
    t1 = float(timestamps[-1])
    nan_mask = (time_bins < t0 - extrapolation_limit) | (time_bins > t1 + extrapolation_limit)

    # Handle multi-dimensional values (e.g., 2D cursor position).
    # ``np.interp`` defaults clamp to the endpoint values for out-of-range
    # ``time_bins`` (no NaN); we then NaN-mask the truly far-out bins above.
    if values.ndim == 1:
        result = np.interp(time_bins, timestamps, values).astype(np.float64, copy=False)
        if nan_mask.any():
            result = result.copy()
            result[nan_mask] = np.nan
        return result
    else:
        result = np.zeros((len(time_bins), values.shape[1]), dtype=np.float64)
        for dim in range(values.shape[1]):
            result[:, dim] = np.interp(time_bins, timestamps, values[:, dim])
        if nan_mask.any():
            result[nan_mask] = np.nan
        return result


def _normalize_categorical_value(value: object) -> str:
    """Normalize categorical values so mapping is robust across bytes/str dtypes."""
    if isinstance(value, bytes | np.bytes_):
        return value.decode("utf-8")
    if isinstance(value, str | np.str_):
        return str(value)
    return str(value)


def _map_string_values_strict(
    values: np.ndarray,
    mapping: dict[str, int],
    *,
    field_name: str,
) -> np.ndarray:
    """Map categorical values to ints and raise on unknown categories."""
    normalized = [_normalize_categorical_value(x) for x in values]
    unknown = sorted({x for x in normalized if x not in mapping})
    if unknown:
        raise AssertionError(
            f"Unknown categorical values for {field_name}: {unknown}. "
            f"Known categories: {sorted(mapping.keys())}"
        )
    return np.asarray([mapping[x] for x in normalized], dtype=np.int32)


def warn_if_untracked_phases(data: Data) -> None:
    mp = dyn(data).movement_phases

    available_phase_names = []
    for name in dir(mp):
        if name.startswith("_") or name == "domain":
            continue
        if "period" not in name:
            continue
        interval = getattr(mp, name)
        if hasattr(interval, "start") and hasattr(interval, "end"):
            available_phase_names.append(name)

    known_phase_names = set(MOVEMENT_PHASE_LABELS) - {"inter_trial"}
    unknown_phase_names = sorted(set(available_phase_names) - known_phase_names)
    if unknown_phase_names:
        LOGGER.warning(
            f"Unknown movement phase(s) in data.movement_phases: {unknown_phase_names}. "
            f"Known phases: {sorted(known_phase_names)}. "
            "These phases will be ignored unless added to MOVEMENT_PHASE_LABELS."
        )


def _extract_behavioral_intervals(
    data: Data,
) -> dict[str, dict[str, np.ndarray]] | None:
    """Extract sparse interval metadata for evaluation-time control flow.

    Returned starts/ends are in absolute seconds so downstream collation can
    compare them directly with ``absolute_start`` / ``absolute_end``.
    """
    if not (hasattr(data, "brainset") and hasattr(dyn(data).brainset, "id")):
        return None

    brainset = dyn(data).brainset.id
    config = BEHAVIORAL_INTERVAL_CONFIG.get(brainset)
    if config is None:
        return None

    if data.absolute_start is None:
        raise AssertionError(
            "absolute_start not set for this time window. This is required "
            "for forwarding absolute interval boundaries."
        )

    intervals_out: dict[str, dict[str, np.ndarray]] = {}
    absolute_start = float(data.absolute_start)

    for source_name, source_cfg in config.items():
        if not hasattr(data, source_name):
            continue
        obj = getattr(data, source_name)
        if not (hasattr(obj, "start") and hasattr(obj, "end")):
            continue

        end_offset = float(source_cfg.get("end_offset", 0.0))
        interval_dict: dict[str, np.ndarray] = {
            "start": absolute_start + np.asarray(obj.start, dtype=np.float64),
            "end": absolute_start + np.asarray(obj.end, dtype=np.float64) + end_offset,
        }

        for field in source_cfg.get("fields", ()):
            if hasattr(obj, field):
                interval_dict[field] = np.asarray(getattr(obj, field))

        intervals_out[source_name] = interval_dict

    return intervals_out or None


def _extract_default_behavioral_fields(data: Data, behavioral_data: dict[str, np.ndarray]) -> None:
    """Default extraction for brainsets without an entry in BEHAVIORAL_EXTRACTION_CONFIG.

    Takes the session task plus the common cursor/hand/eye/trials fields that exist on
    ``data`` and writes them into ``behavioral_data`` in place.
    """
    if hasattr(data, "session") and hasattr(dyn(data).session, "task"):
        behavioral_data["task"] = np.asarray(dyn(data).session.task)

    for data_attr in ["cursor", "hand", "eye", "trials"]:
        if not hasattr(data, data_attr):
            continue
        obj = getattr(data, data_attr)
        # Common fields for time series
        for field in ["timestamps", "pos", "vel", "acc", "target_pos"]:
            if hasattr(obj, field):
                key = f"{data_attr}_{field}"
                behavioral_data[key] = np.asarray(getattr(obj, field))
        # Trial-specific fields
        if data_attr == "trials":
            for field in ["target_id", "result", "start", "end"]:
                if hasattr(obj, field):
                    key = f"trial_{field}"
                    behavioral_data[key] = np.asarray(getattr(obj, field))


def _extract_behavioral_fields(data: Data) -> dict[str, np.ndarray] | None:
    """Extract behavioral fields from Data object into dict of numpy arrays.

    Uses BEHAVIORAL_EXTRACTION_CONFIG to determine which fields to extract
    based on the brainset name. This ensures we only extract fields that
    exist for each dataset and handles temporal slicing automatically through
    temporaldata's slice mechanism.

    Returns:
        Dictionary mapping field names to numpy arrays. Temporal data
        (IrregularTimeSeries) is automatically sliced when the Data object
        is sliced, so timestamps and values align with the spike data window.

    Movement phase:
        - this is hold/reach/return
        - each phase is an own attribute of data, and is an interval
            - those intervals apparently have arrays of start and end times.
        - uses _intervals_to_categorical_labels(time_bins, intervals_dict) to
          go from a dict of labels: (start_array, end_array) to an integer
          array of labels, one per timestamp. (Todo check its test.)
    Trial ID:
        - data.trials has .start and .end arrays, we assume those define trials
        - we assign each of them as ID its _absolute start time_. This means
          that we have repeated IDs across sessions, but not within a session.
        - then it uses _expand_trial_field_to_timebins(
                    local_trial_ids, trial_starts, trial_ends, time_bins, fill_value=-1
                ),
          to probably take the (n_trials,)-shaped trial-ids, and the corresponding
          times, and assigns to each timestamp in time_bins(? - why that name?)
          the local_trial_id within which it falls (Not asserting no overlap, most likely.)
    SPECIAL_EXPAND_TO_TIMEBINS:
        - Parsed exactly the same as Trial ID: with
             _expand_trial_field_to_timebins(
                    values, trial_starts, trial_ends, time_bins, fill_value=-1
                ),
            and values must have shape (n_trials,...) and each time stamp (bin?)
            gets the value of the trial it falls into.

    """
    behavioral_data = {}

    # Get brainset name from data.brainset.id
    if hasattr(data, "brainset") and hasattr(dyn(data).brainset, "id"):
        brainset = dyn(data).brainset.id
    else:
        # No brainset id: the default extraction below applies.
        brainset = None

    if hasattr(data, "binned_spike_counts"):
        binned = dyn(data).binned_spike_counts
        sampling_rate = binned.sampling_rate
        bin_size = 1.0 / sampling_rate
        start = binned.domain.start.item()
        n_bins = binned.counts.shape[0]
        time_bins = start + (np.arange(n_bins) + 0.5) * bin_size
    elif hasattr(data, "spikes"):
        spikes = dyn(data).spikes
        if hasattr(spikes, "bin_centers"):
            time_bins = np.asarray(spikes.bin_centers)
        elif hasattr(spikes, "timestamps"):
            raise AssertionError(
                "Behavior extraction requires bin-aligned time bins. "
                "Found spikes.timestamps but no spikes.bin_centers. "
                "Provide data.binned_spike_counts or spikes.bin_centers."
            )
        else:
            time_bins = None
    else:
        time_bins = None

    # Get extraction config for this brainset
    if brainset in BEHAVIORAL_EXTRACTION_CONFIG:
        config = BEHAVIORAL_EXTRACTION_CONFIG[brainset]
        for output_key, spec in config.items():
            # Handle special movement phase extraction
            if spec == "SPECIAL_MOVEMENT_PHASE":
                # Extract movement phases as categorical labels
                if hasattr(data, "movement_phases"):
                    warn_if_untracked_phases(data)
                    if time_bins is None:
                        continue

                    # Extract intervals for each movement phase
                    mp = dyn(data).movement_phases
                    intervals_dict: dict[int, tuple[np.ndarray, np.ndarray]] = {}
                    for phase_name in MOVEMENT_PHASE_LABELS.keys():
                        # ["hold_period", "reach_period",
                        #               "return_period", "random_period"]:
                        if hasattr(mp, phase_name):
                            interval = getattr(mp, phase_name)
                            assert hasattr(interval, "start") and hasattr(interval, "end")
                            starts = np.asarray(interval.start)
                            ends = np.asarray(interval.end)
                            phase_label = MOVEMENT_PHASE_LABELS[phase_name]
                            intervals_dict[phase_label] = (starts, ends)

                    if intervals_dict:
                        labels = _intervals_to_categorical_labels(
                            time_bins,
                            intervals_dict,
                            fill_label=MOVEMENT_PHASE_LABELS["inter_trial"],
                        )
                        behavioral_data[output_key] = labels
                continue

            # Handle special trial_id generation (absolute start times as IDs)
            # Note: this assumes a specific format for the trial info stored in the
            # data object; other brainsets may need a different implementation.
            # What this does: We assign each trial its start time, then assign
            # as "ID" to each time point the start time of the trial it falls into.
            if spec == "SPECIAL_TRIAL_ID":
                if not hasattr(data, "trials"):
                    continue
                trials = dyn(data).trials
                if not (hasattr(trials, "start") and hasattr(trials, "end")):
                    continue

                if time_bins is None:
                    continue

                # Generate sequential trial IDs
                trial_starts = np.asarray(trials.start)
                trial_ends = np.asarray(trials.end)
                if data.absolute_start is None:
                    raise AssertionError(
                        "absolute_start not set for this time window."
                        " This is required for generating unique trial IDs."
                        " If that's intended, you must find a different way to create "
                        "unique trial IDs for/during validation collation."
                    )
                absolute_trial_starts = data.absolute_start + trial_starts
                # local_trial_ids = np.arange(len(trial_starts), dtype=np.int32)
                trial_ids = absolute_trial_starts

                # Expand to time bins
                expanded = _expand_trial_field_to_timebins(
                    trial_ids, trial_starts, trial_ends, time_bins, fill_value=-1
                )
                behavioral_data[output_key] = expanded
                continue

            # Handle special per-interval field expansion to time bins.
            # Short form: ("SPECIAL_EXPAND_TO_TIMEBINS", "field") -> trials.field
            # Full form:  ("SPECIAL_EXPAND_TO_TIMEBINS", "source", "field",
            #              fill_value?, end_offset?)
            if isinstance(spec, tuple) and spec[0] == "SPECIAL_EXPAND_TO_TIMEBINS":
                if len(spec) == 2:
                    data_attr = "trials"
                    field_attr = spec[1]
                    fill_value = -1
                    end_offset = 0.0
                else:
                    _, data_attr, field_attr, *rest = spec
                    fill_value = rest[0] if len(rest) >= 1 else -1
                    end_offset = float(rest[1]) if len(rest) >= 2 else 0.0

                if not hasattr(data, data_attr):
                    continue
                obj = getattr(data, data_attr)
                if not (
                    hasattr(obj, "start") and hasattr(obj, "end") and hasattr(obj, field_attr)
                ):
                    continue

                if time_bins is None:
                    continue

                interval_starts = np.asarray(obj.start, dtype=np.float64)
                interval_ends = extend_interval_ends_with_post_event_window(
                    interval_starts,
                    np.asarray(obj.end, dtype=np.float64),
                    end_offset=end_offset,
                )
                field_values = np.asarray(getattr(obj, field_attr))

                mapping_key = (brainset, output_key)
                if mapping_key in BEHAVIORAL_STRING_MAPPINGS and len(field_values) > 0:
                    mapping = BEHAVIORAL_STRING_MAPPINGS[mapping_key]
                    field_values = _map_string_values_strict(
                        field_values, mapping, field_name=f"{brainset}:{output_key}"
                    )

                expanded = _expand_interval_field_to_timebins(
                    field_values,
                    interval_starts,
                    interval_ends,
                    time_bins,
                    fill_value=fill_value,
                )
                behavioral_data[output_key] = expanded
                continue

            # Handle special time series resampling to time bins
            # Extract the IrregularTimeSeries field and resample it to the time bins
            if (isinstance(spec, tuple) and len(spec) == 3) and (
                spec[0] == "SPECIAL_RESAMPLE_TIMESERIES"
            ):
                _, data_attr, field_attr = spec
                if not hasattr(data, data_attr):
                    continue  # a missing attribute is skipped, not an error
                obj = getattr(data, data_attr)
                if not (hasattr(obj, "timestamps") and hasattr(obj, field_attr)):
                    continue

                if time_bins is None:
                    continue

                # Extract timestamps and values from the IrregularTimeSeries
                src_timestamps = np.asarray(obj.timestamps)
                src_values = np.asarray(getattr(obj, field_attr))

                # Resample to time bins
                resampled = _resample_timeseries_to_bins(src_timestamps, src_values, time_bins)
                behavioral_data[output_key] = resampled
                continue

            # Normal extraction
            assert isinstance(spec, tuple) and len(spec) == 2, f"Bad extraction spec: {spec!r}"
            data_attr, field_attr = spec
            if not hasattr(data, data_attr):
                continue
            obj = getattr(data, data_attr)
            if hasattr(obj, field_attr):
                value = getattr(obj, field_attr)
                array = np.asarray(value)

                # Check if this field has a string-to-int mapping
                mapping_key = (brainset, output_key)
                if mapping_key in BEHAVIORAL_STRING_MAPPINGS:
                    mapping = BEHAVIORAL_STRING_MAPPINGS[mapping_key]
                    behavioral_data[output_key] = _map_string_values_strict(
                        array, mapping, field_name=f"{brainset}:{output_key}"
                    )
                else:
                    behavioral_data[output_key] = array
    else:
        _extract_default_behavioral_fields(data, behavioral_data)

    return behavioral_data or None
