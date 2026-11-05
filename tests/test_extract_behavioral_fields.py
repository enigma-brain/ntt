"""Tests for behavioral data extraction functions in behavior_transforms.py.

Context: How are these functions used:

In DataToModelInput:
    if not hasattr(data, 'binned_spike_counts'):
        raise ValueError("...")
    behavioral_data = _extract_behavioral_fields(data)

In behavior_transforms.py:
    THIS is likely the main function to check: _extract_behavioral_fields()

_intervals_to_categorical_labels():
   - take a set/list/whatever of intervals (a dict, lables: interval list?)
   - and an array of time stamps
   - return an array of integer labels, one per time stamp; 0 if it falls
     into none of the intervals, the integer id of the label otherwise

There:
    # Generate sequential trial IDs
    trial_starts = np.asarray(trials.start)
    trial_ends = np.asarray(trials.end)
    trial_ids = np.arange(len(trial_starts), dtype=np.int32)

    # Expand to time bins
    expanded = _expand_trial_field_to_timebins(
                    trial_ids, trial_starts, trial_ends, time_bins, fill_value=-1
    )


"""

import numpy as np
import pytest

from ntt.core.data.components.behavior_transforms import (
    _expand_trial_field_to_timebins,
    _extract_behavioral_fields,
    _extract_behavioral_intervals,
    _intervals_to_categorical_labels,
    _resample_timeseries_to_bins,
)

# ---------------------------------------------------------------------------
# Helpers for building mock Data objects
# ---------------------------------------------------------------------------


def _make_obj(**kwargs):
    """Create a simple namespace object with given attributes."""
    return type("Obj", (), kwargs)()


def _make_binned_spike_counts(start, n_bins, sampling_rate):
    """Build a mock binned_spike_counts with domain, sampling_rate, counts."""
    domain = _make_obj(start=np.float64(start))
    counts = np.zeros((n_bins, 1))
    return _make_obj(domain=domain, sampling_rate=sampling_rate, counts=counts)


def _make_data_with_cursor(
    brainset_id="perich_miller_population_2018",
    n_bins=10,
    sampling_rate=50.0,
    start=0.0,
):
    """Build a mock Data object with cursor IrregularTimeSeries and binned spikes."""
    bin_size = 1.0 / sampling_rate
    # Cursor: 2D position sampled at higher rate across the same time range
    n_cursor = 50
    cursor_ts = np.linspace(start, start + n_bins * bin_size, n_cursor)
    cursor_pos = np.column_stack(
        [
            np.sin(cursor_ts),  # x
            np.cos(cursor_ts),  # y
        ]
    )
    cursor_vel = np.column_stack(
        [
            np.cos(cursor_ts),
            -np.sin(cursor_ts),
        ]
    )
    cursor = _make_obj(timestamps=cursor_ts, pos=cursor_pos, vel=cursor_vel)

    binned = _make_binned_spike_counts(start, n_bins, sampling_rate)
    brainset = _make_obj(id=brainset_id)

    return _make_obj(
        brainset=brainset,
        binned_spike_counts=binned,
        cursor=cursor,
    )


# ---------------------------------------------------------------------------
# Tests for _resample_timeseries_to_bins
# ---------------------------------------------------------------------------


class TestResampleTimeseriesToBins:
    def test_1d_interpolation(self):
        """Linear values should be interpolated exactly."""
        timestamps = np.array([0.0, 1.0, 2.0])
        values = np.array([0.0, 10.0, 20.0])
        time_bins = np.array([0.5, 1.0, 1.5])
        result = _resample_timeseries_to_bins(timestamps, values, time_bins)
        np.testing.assert_allclose(result, [5.0, 10.0, 15.0])

    def test_2d_interpolation(self):
        """Multi-dim values should be interpolated per dimension."""
        timestamps = np.array([0.0, 1.0])
        values = np.array([[0.0, 10.0], [2.0, 20.0]])
        time_bins = np.array([0.5])
        result = _resample_timeseries_to_bins(timestamps, values, time_bins)
        assert result.shape == (1, 2)
        np.testing.assert_allclose(result[0], [1.0, 15.0])

    def test_empty_input_1d(self):
        """Empty timestamps/values should return NaN-filled array."""
        result = _resample_timeseries_to_bins(np.array([]), np.array([]), np.array([0.5, 1.5]))
        assert result.shape == (2,)
        assert np.all(np.isnan(result))

    def test_empty_input_2d(self):
        """Empty 2D values should return NaN-filled (n_bins, n_dims) array."""
        result = _resample_timeseries_to_bins(np.array([]), np.zeros((0, 3)), np.array([0.5, 1.5]))
        assert result.shape == (2, 3)
        assert np.all(np.isnan(result))

    def test_per_window_edge_no_nan_for_continuous_signal(self):
        """Per-window resampling of a continuously
        recorded signal must not produce NaN at the window's first/last few
        bins just because they fall outside the sample range.

        Setup mimics the perich-miller cursor case: 10 ms-spaced samples at
        +0.005, +0.015, ..., +0.095 s within a 100 ms window; bin centres at
        every 1 ms. Bins 0-4 and 95-99 fall outside the sample range and must
        not be NaN'd, otherwise every concatenated window boundary would carry
        a ~10 ms NaN run that the regression metric's train-side filter would
        treat as data corruption.
        """
        timestamps = np.linspace(0.005, 0.095, 10)
        values = np.cos(2 * np.pi * timestamps).astype(np.float64)
        time_bins = (np.arange(100, dtype=np.float64) + 0.5) / 1000.0
        result = _resample_timeseries_to_bins(timestamps, values, time_bins)
        assert result.shape == (100,)
        assert np.all(np.isfinite(result)), (
            f"per-window edge bins should clamp, not NaN; got "
            f"{int(np.isnan(result).sum())} NaN out of {result.size}"
        )
        # Clamped edges should equal the endpoint sample values.
        np.testing.assert_allclose(result[:5], values[0])
        np.testing.assert_allclose(result[-5:], values[-1])

    def test_distant_extrapolation_still_nan(self):
        """Extrapolation beyond ~2 × sample-period must remain NaN — that is
        how genuine recording gaps continue to be marked.
        """
        timestamps = np.linspace(0.005, 0.095, 10)  # 10 ms spacing
        values = np.zeros_like(timestamps)
        # Bins 100 ms before / after the sample range — well beyond the
        # 2 × 10 ms tolerance.
        time_bins = np.array([-0.1, 0.05, 0.2])
        result = _resample_timeseries_to_bins(timestamps, values, time_bins)
        assert np.isnan(result[0]), "bin 100 ms before samples should be NaN"
        assert np.isfinite(result[1]), "bin within samples should be finite"
        assert np.isnan(result[2]), "bin 100 ms after samples should be NaN"

    def test_2d_per_window_edge_no_nan(self):
        """Same as ``test_per_window_edge_no_nan_for_continuous_signal``
        but for the 2D ``cursor_pos``/``cursor_vel`` path that the regression
        metric actually consumes.
        """
        timestamps = np.linspace(0.005, 0.095, 10)
        values = np.column_stack(
            [np.cos(2 * np.pi * timestamps), np.sin(2 * np.pi * timestamps)]
        ).astype(np.float64)
        time_bins = (np.arange(100, dtype=np.float64) + 0.5) / 1000.0
        result = _resample_timeseries_to_bins(timestamps, values, time_bins)
        assert result.shape == (100, 2)
        assert np.all(np.isfinite(result)), (
            f"per-window edge bins should clamp, not NaN; got "
            f"{int(np.isnan(result).sum())} NaN out of {result.size}"
        )

    def test_single_sample_window_returns_all_nan(self):
        """A single-sample window has no spacing context. The helper returns
        all-NaN rather than silently clamping every bin to that one value —
        the caller's per-bin finite filter then treats the window as
        unobserved. The opposite policy would let a one-shot behaviour sample
        fabricate a whole trace.
        """
        timestamps = np.array([0.05])
        values = np.array([1.0])
        time_bins = np.array([0.01, 0.05, 0.09])
        result = _resample_timeseries_to_bins(timestamps, values, time_bins)
        assert result.shape == (3,)
        assert np.all(np.isnan(result))

        # 2D path: same.
        values_2d = np.array([[1.0, 2.0]])
        result_2d = _resample_timeseries_to_bins(timestamps, values_2d, time_bins)
        assert result_2d.shape == (3, 2)
        assert np.all(np.isnan(result_2d))

    def test_non_monotonic_timestamps_rejected(self):
        """Out-of-order timestamps make ``np.interp`` results meaningless
        and the extrapolation threshold inferred from
        ``median(diff(timestamps))`` becomes a function of the disorder.
        The helper rejects them (a negative diff) rather than silently
        producing whatever ``np.interp`` happens to return.
        """
        timestamps = np.array([0.0, 1.0, 0.5])
        values = np.array([0.0, 1.0, 2.0])
        time_bins = np.array([0.5])
        with pytest.raises(AssertionError, match="non-decreasing"):
            _resample_timeseries_to_bins(timestamps, values, time_bins)

    def test_duplicate_timestamps_tolerated(self):
        """perich_miller carries a few duplicate 10 ms-rounded timestamps
        with genuinely different values. ``np.interp`` handles them,
        so the helper tolerates duplicates (guard relaxed to non-decreasing)
        rather than crashing. Interior output must be finite and identical
        to a direct ``np.interp`` reference.
        """
        timestamps = np.array([0.0, 0.01, 0.01, 0.02, 0.03])
        values = np.array([0.0, 1.0, 1.2, 2.0, 3.0])
        time_bins = np.array([0.005, 0.015, 0.025])  # all within [t0, t1]
        result = _resample_timeseries_to_bins(timestamps, values, time_bins)
        assert np.all(np.isfinite(result))
        np.testing.assert_allclose(result, np.interp(time_bins, timestamps, values))

    def test_all_duplicate_timestamps_return_nan(self):
        """A window where (nearly) every timestamp collides has no usable
        time span — the median sample spacing collapses to zero — so the
        helper returns all-NaN (window treated as unobserved) rather than
        fabricating a bin grid from a degenerate input. This happens
        for real: a short eval window can land on nothing but one of
        perich_miller's duplicate 10 ms-rounded cursor pairs, and it must
        NOT crash the whole eval. Matches the <2-sample guard's semantics.
        """
        timestamps = np.array([0.02, 0.02, 0.02])
        values = np.array([1.0, 2.0, 3.0])
        time_bins = np.array([0.01, 0.02, 0.03])
        result = _resample_timeseries_to_bins(timestamps, values, time_bins)
        assert np.all(np.isnan(result)), "degenerate window must be all-NaN"

    def test_duplicate_pair_only_window_return_nan(self):
        """A window
        whose only two cursor samples are a duplicate 10 ms pair. Median
        spacing is 0 → all-NaN, no crash.
        """
        timestamps = np.array([0.02, 0.02])
        values = np.array([[1.0, 5.0], [2.0, 6.0]])  # 2D cursor pos
        time_bins = np.array([0.01, 0.02, 0.03])
        result = _resample_timeseries_to_bins(timestamps, values, time_bins)
        assert result.shape == (3, 2)
        assert np.all(np.isnan(result))

    def test_threshold_exact_boundary_is_inclusive_to_finite(self):
        """A bin exactly at ``timestamps[-1] + max_extrapolation_factor *
        dt_median`` is NOT NaN-masked (mask uses strict ``>``). Documents
        the half-open semantic.
        """
        # 10 ms-spaced samples, default max_extrapolation_factor=2.0 → 20 ms
        timestamps = np.array([0.005, 0.015, 0.025])
        values = np.array([1.0, 2.0, 3.0])
        # bins at exactly t1 + 20 ms = 0.045, just past it = 0.046,
        # within range = 0.020
        time_bins = np.array([0.020, 0.045, 0.046])
        result = _resample_timeseries_to_bins(timestamps, values, time_bins)
        assert np.isfinite(result[0]), "bin within sample range"
        assert np.isfinite(result[1]), "bin exactly at threshold should be finite"
        assert np.isnan(result[2]), "bin past threshold should be NaN"

    def test_no_internal_gap_detection(self):
        """Document the limitation: an internal gap between adjacent
        samples (i.e. inside ``[timestamps[0], timestamps[-1]]``) is NOT
        masked. Linear interpolation fabricates a straight line across
        any internal gap. Kept as a test so the limitation is auditable.
        """
        timestamps = np.array([0.005, 0.095])  # 90 ms gap, far above 10 ms
        values = np.array([0.0, 10.0])
        time_bins = np.array([0.050])  # midpoint, deep inside the gap
        result = _resample_timeseries_to_bins(timestamps, values, time_bins)
        # The interpolated value is a finite straight line midpoint, not NaN.
        np.testing.assert_allclose(result, [5.0])


# ---------------------------------------------------------------------------
# Tests for _intervals_to_categorical_labels
# ---------------------------------------------------------------------------


class TestIntervalsToCategoricalLabels:
    def test_non_overlapping_intervals(self):
        """Each interval should get its own label, gaps get 0."""
        time_bins = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        intervals = {
            1: (np.array([0.0]), np.array([1.0])),
            2: (np.array([2.0]), np.array([3.0])),
        }
        labels = _intervals_to_categorical_labels(time_bins, intervals, fill_label=0)
        # 0.5 in [0,1] → hold=1; 1.5 gap → 0; 2.5 in [2,3] → reach=2; 3.5,4.5 gap → 0
        np.testing.assert_array_equal(labels, [1, 0, 2, 0, 0])

    def test_overlapping_intervals_last_wins(self):
        """When intervals overlap, later dict entry overwrites earlier."""
        time_bins = np.array([0.5])
        intervals = {
            1: (np.array([0.0]), np.array([1.0])),
            2: (np.array([0.0]), np.array([1.0])),
        }
        labels = _intervals_to_categorical_labels(time_bins, intervals, fill_label=0)
        # 'b' is label_idx=2, should overwrite 'a'=1
        np.testing.assert_array_equal(labels, [2])

    def test_empty_intervals(self):
        """No intervals → all zeros."""
        time_bins = np.array([0.5, 1.5])
        labels = _intervals_to_categorical_labels(time_bins, {}, fill_label=0)
        np.testing.assert_array_equal(labels, [0, 0])


# ---------------------------------------------------------------------------
# Tests for _expand_trial_field_to_timebins
# ---------------------------------------------------------------------------


class TestExpandTrialFieldToTimebins:
    def test_basic_expansion(self):
        """Per-trial values should be tiled across their time range."""
        field_values = np.array([10, 20])
        trial_starts = np.array([0.0, 2.0])
        trial_ends = np.array([1.0, 3.0])
        time_bins = np.array([0.5, 1.5, 2.5])
        result = _expand_trial_field_to_timebins(field_values, trial_starts, trial_ends, time_bins)
        # 0.5 in [0,1]→10; 1.5 inter-trial→-1; 2.5 in [2,3]→20
        np.testing.assert_array_equal(result, [10, -1, 20])

    def test_boolean_field_converted_to_int(self):
        """Boolean fields should become int so -1 fill is distinguishable from False."""
        field_values = np.array([True, False])
        trial_starts = np.array([0.0, 2.0])
        trial_ends = np.array([1.0, 3.0])
        time_bins = np.array([0.5, 1.5, 2.5])
        result = _expand_trial_field_to_timebins(field_values, trial_starts, trial_ends, time_bins)
        assert result.dtype == np.int32
        np.testing.assert_array_equal(result, [1, -1, 0])

    def test_float_field_uses_float_fill(self):
        """Float field should use float fill value."""
        field_values = np.array([1.5, 2.5])
        trial_starts = np.array([0.0, 2.0])
        trial_ends = np.array([1.0, 3.0])
        time_bins = np.array([0.5, 1.5, 2.5])
        result = _expand_trial_field_to_timebins(field_values, trial_starts, trial_ends, time_bins)
        assert result.dtype == np.float64
        np.testing.assert_allclose(result, [1.5, -1.0, 2.5])


# ---------------------------------------------------------------------------
# Tests for _extract_behavioral_fields (integration)
# ---------------------------------------------------------------------------


class TestExtractBehavioralFields:
    def test_cursor_resampling_perich_miller(self):
        """cursor_pos and cursor_vel should be resampled to binned spike time bins."""
        n_bins = 10
        sampling_rate = 50.0
        data = _make_data_with_cursor(n_bins=n_bins, sampling_rate=sampling_rate)
        result = _extract_behavioral_fields(data)

        assert "cursor_pos" in result
        assert "cursor_vel" in result
        assert result["cursor_pos"].shape == (n_bins, 2)
        assert result["cursor_vel"].shape == (n_bins, 2)

    def test_movement_phase_extraction(self):
        """movement_phase should produce categorical labels from interval data."""
        n_bins = 20
        sampling_rate = 100.0
        start = 0.0

        binned = _make_binned_spike_counts(start, n_bins, sampling_rate)
        brainset = _make_obj(id="perich_miller_population_2018")

        # Movement phases: hold [0, 0.05], reach [0.08, 0.15]
        hold = _make_obj(start=np.array([0.0]), end=np.array([0.05]))
        reach = _make_obj(start=np.array([0.08]), end=np.array([0.15]))
        movement_phases = _make_obj(
            hold_period=hold,
            reach_period=reach,
        )

        data = _make_obj(
            brainset=brainset,
            binned_spike_counts=binned,
            movement_phases=movement_phases,
        )

        result = _extract_behavioral_fields(data)
        assert "movement_phase" in result
        labels = result["movement_phase"]
        assert labels.shape == (n_bins,)
        # Should have 0 (none), 1 (hold), 2 (reach) but no 3 (no return_period)
        assert set(np.unique(labels)).issubset({0, 1, 2})
        # At least some bins should be labeled hold (1) and reach (2)
        assert 1 in labels
        assert 2 in labels

    def test_trial_id_and_expand_fields(self):
        """trial_id and per-trial fields should be expanded to time bins."""
        n_bins = 20
        sampling_rate = 100.0
        start = 0.0

        binned = _make_binned_spike_counts(start, n_bins, sampling_rate)
        brainset = _make_obj(id="perich_miller_population_2018")

        # Two trials covering part of the time range
        trials = _make_obj(
            start=np.array([0.0, 0.1]),
            end=np.array([0.05, 0.15]),
            target_id=np.array([3, 7]),
            target_dir=np.array([0.5, 1.2]),
            result=np.array([b"R", b"A"]),
            is_valid=np.array([True, False]),
        )

        data = _make_obj(
            brainset=brainset,
            binned_spike_counts=binned,
            trials=trials,
            absolute_start=start,
        )

        result = _extract_behavioral_fields(data)

        # trial_id should be present (generated from absolute start times)
        assert "trial_id" in result
        assert result["trial_id"].shape == (n_bins,)

        # trial_target_id
        assert "trial_target_id" in result
        assert result["trial_target_id"].shape == (n_bins,)

        # trial_result: should be mapped via BEHAVIORAL_STRING_MAPPINGS
        assert "trial_result" in result
        # b'R' → 0, b'A' → 1 per the mapping
        unique_vals = set(np.unique(result["trial_result"]))
        assert 0 in unique_vals  # R
        assert 1 in unique_vals  # A

        # trial_is_valid: boolean → int
        assert "trial_is_valid" in result
        assert result["trial_is_valid"].dtype == np.int32

    def test_unknown_brainset_generic_extraction(self):
        """Unknown brainset should fall through to generic extraction."""
        cursor = _make_obj(
            timestamps=np.array([0.0, 1.0]),
            pos=np.array([[1.0, 2.0], [3.0, 4.0]]),
        )
        session = _make_obj(id="sess1", task="reaching")
        brainset = _make_obj(id="unknown_dataset_xyz")

        data = _make_obj(
            brainset=brainset,
            cursor=cursor,
            session=session,
        )

        result = _extract_behavioral_fields(data)
        # Generic extraction should find cursor_timestamps, cursor_pos
        assert "cursor_timestamps" in result
        assert "cursor_pos" in result
        # And session task
        assert "task" in result

    def test_no_brainset_attribute(self):
        """Data with no brainset should use generic extraction (not crash)."""
        data = _make_obj()
        result = _extract_behavioral_fields(data)
        # No behavioral data available → None
        assert result is None

    def test_missing_behavioral_attributes_silently_skipped(self):
        """When configured fields are missing from Data, they are skipped (not errored)."""
        # Perich miller config expects cursor, trials, movement_phases, etc.
        # Provide none of them.
        brainset = _make_obj(id="perich_miller_population_2018")
        binned = _make_binned_spike_counts(0.0, 10, 50.0)
        data = _make_obj(brainset=brainset, binned_spike_counts=binned)

        result = _extract_behavioral_fields(data)
        # Nothing extracted, but no error
        assert result is None

    def test_churchland_shenoy_plain_extraction(self):
        """Churchland-Shenoy config uses plain (data_attr, field_attr) extraction."""
        cursor = _make_obj(
            timestamps=np.array([0.0, 0.5, 1.0]),
            pos=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            vel=np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
        )
        brainset = _make_obj(id="churchland_shenoy_neural_2012")

        data = _make_obj(brainset=brainset, cursor=cursor)

        result = _extract_behavioral_fields(data)
        assert "cursor_timestamps" in result
        assert "cursor_pos" in result
        assert "cursor_vel" in result
        np.testing.assert_array_equal(result["cursor_timestamps"], [0.0, 0.5, 1.0])

    def test_cowley_image_id_expands_from_stimuli(self):
        """Cowley image_id should expand from stimuli intervals to time bins."""
        n_bins = 20
        sampling_rate = 100.0
        start = 0.0

        binned = _make_binned_spike_counts(start, n_bins, sampling_rate)
        brainset = _make_obj(id="cowley_v4_compact_2024")
        stimuli = _make_obj(
            start=np.array([0.0, 0.1]),
            end=np.array([0.05, 0.15]),
            image_id=np.array([17, 42]),
        )

        data = _make_obj(
            brainset=brainset,
            binned_spike_counts=binned,
            stimuli=stimuli,
            absolute_start=start,
        )

        result = _extract_behavioral_fields(data)
        assert "image_id" in result
        assert result["image_id"].shape == (n_bins,)
        np.testing.assert_array_equal(result["image_id"][:10], np.full(10, 17.0))
        np.testing.assert_array_equal(result["image_id"][10:], np.full(10, 42.0))

    def test_cowley_intervals_forwarded_as_absolute_times(self):
        """Sparse interval metadata should be forwarded in absolute time."""
        brainset = _make_obj(id="cowley_v4_compact_2024")
        stimuli = _make_obj(
            start=np.array([0.0, 0.1]),
            end=np.array([0.05, 0.15]),
            image_id=np.array([17, 42]),
        )
        data = _make_obj(
            brainset=brainset,
            stimuli=stimuli,
            absolute_start=3.0,
        )

        result = _extract_behavioral_intervals(data)
        assert result is not None
        assert "stimuli" in result
        np.testing.assert_allclose(result["stimuli"]["start"], [3.0, 3.1])
        np.testing.assert_allclose(result["stimuli"]["end"], [3.25, 3.35])
        np.testing.assert_array_equal(result["stimuli"]["image_id"], [17, 42])
