"""Tests for the per-stimulus sampler."""

import pytest

from ntt.evaluation.utils.per_stimulus_sampler import PerStimulusSampler
from tests.fixtures import SimpleDataset


def test_one_index_per_stim():
    """Each stimulus interval should contribute exactly one sampler index."""
    ds = SimpleDataset.with_stimuli(
        session_id="S1",
        stim_starts=[0.10, 0.40, 0.80],
        stim_ends=[0.20, 0.50, 0.90],
        image_ids=[7, 11, 13],
        n_units=4,
        sampling_rate=1000,
        duration=1.0,
    )
    sampler = PerStimulusSampler(
        ds,
        session_ids=["S1"],
        interval_source="stimuli",
        grouping_field="image_id",
        interval_start_lag_ms=0.0,
        interval_end_lag_ms=0.0,
    )
    assert len(list(sampler)) == 3


def test_lag_clamps_at_neighbor_boundary():
    """Trailing lag should clamp at the following stimulus start."""
    ds = SimpleDataset.with_stimuli(
        session_id="S1",
        stim_starts=[0.10, 0.30, 0.50],
        stim_ends=[0.15, 0.35, 0.55],
        image_ids=[1, 2, 3],
        n_units=4,
        sampling_rate=1000,
        duration=1.0,
    )
    sampler = PerStimulusSampler(
        ds,
        session_ids=["S1"],
        interval_source="stimuli",
        grouping_field="image_id",
        interval_start_lag_ms=0.0,
        interval_end_lag_ms=200.0,
        cross_stim_padding_ms=0.0,
    )
    geoms = sampler.window_geometries()
    assert geoms[0][1] == pytest.approx(0.30)
    assert geoms[1][1] == pytest.approx(0.50)


def test_stim_bounds_for_returns_raw_stimulus_bounds():
    """Raw stimulus bounds should stay separate from lag-extended geometry."""
    ds = SimpleDataset.with_stimuli(
        session_id="S1",
        stim_starts=[0.10, 0.30, 0.50],
        stim_ends=[0.15, 0.35, 0.55],
        image_ids=[1, 2, 3],
        n_units=4,
        sampling_rate=1000,
        duration=1.0,
    )
    sampler = PerStimulusSampler(
        ds,
        session_ids=["S1"],
        interval_source="stimuli",
        grouping_field="image_id",
        interval_start_lag_ms=0.0,
        interval_end_lag_ms=200.0,
        cross_stim_padding_ms=0.0,
    )

    assert sampler.geometry_for(1) == ("S1", pytest.approx(0.30), pytest.approx(0.50), 2)
    assert sampler.stim_bounds_for(1) == (pytest.approx(0.30), pytest.approx(0.35))


def test_cross_stim_padding_extends_effective_end_into_next_stim():
    """With 20 ms cross-stim padding, effective_end can extend 20 ms past next_stim_start."""
    ds = SimpleDataset.with_stimuli(
        session_id="S1",
        stim_starts=[0.10, 0.30],
        stim_ends=[0.15, 0.35],
        image_ids=[1, 2],
        n_units=4,
        sampling_rate=1000,
        duration=1.0,
    )
    sampler = PerStimulusSampler(
        ds,
        session_ids=["S1"],
        interval_source="stimuli",
        grouping_field="image_id",
        interval_end_lag_ms=300.0,
        cross_stim_padding_ms=20.0,
    )

    geoms = sampler.window_geometries()
    assert geoms[0][1] == pytest.approx(0.32)


def test_cross_stim_padding_does_not_extend_past_session_end():
    """Last stim has no next_stim_start, so effective_end stays at stim_end + lag."""
    ds = SimpleDataset.with_stimuli(
        session_id="S1",
        stim_starts=[0.10, 0.30],
        stim_ends=[0.15, 0.35],
        image_ids=[1, 2],
        n_units=4,
        sampling_rate=1000,
        duration=1.0,
    )
    sampler = PerStimulusSampler(
        ds,
        session_ids=["S1"],
        interval_source="stimuli",
        grouping_field="image_id",
        interval_end_lag_ms=300.0,
        cross_stim_padding_ms=20.0,
    )

    geoms = sampler.window_geometries()
    assert geoms[1][1] == pytest.approx(0.65)


def test_raises_on_overlapping_intervals():
    """Overlapping stimulus intervals should fail loudly at init."""
    ds = SimpleDataset.with_stimuli(
        session_id="S1",
        stim_starts=[0.10, 0.15],
        stim_ends=[0.20, 0.25],
        image_ids=[1, 2],
        n_units=4,
        sampling_rate=1000,
        duration=1.0,
    )
    with pytest.raises(ValueError, match="overlap"):
        PerStimulusSampler(
            ds,
            session_ids=["S1"],
            interval_source="stimuli",
            grouping_field="image_id",
            interval_end_lag_ms=0.0,
        )


def test_raises_clearly_when_ignore_splits_false_yields_zero_intervals():
    """A split-filtered session should mention ignore_splits in the failure."""
    ds = SimpleDataset.with_stimuli(
        session_id="S1",
        stim_starts=[0.10],
        stim_ends=[0.20],
        image_ids=[1],
        n_units=4,
        sampling_rate=1000,
        duration=1.0,
        ignore_splits=False,
        split="train",
        stim_in_split="test",
    )

    with pytest.raises(ValueError, match="ignore_splits"):
        PerStimulusSampler(
            ds,
            session_ids=["S1"],
            interval_source="stimuli",
            grouping_field="image_id",
            interval_end_lag_ms=0.0,
        )


# ────────────────────────────────────────────────────────────────────
# Fixed-window-duration mode (cowley-style: ~90 ms stims, ~120 ms gap;
# 100 ms-bin tokenizers need a 200 ms window to yield 2 tokens / stim).
# ────────────────────────────────────────────────────────────────────


def test_fixed_window_duration_yields_constant_length_windows():
    """`fixed_window_duration_ms` overrides stim-end-based windowing.

    With stim_starts=[0.10, 0.30] and fixed_window_duration_ms=200 the
    window for the first stim is [0.10, 0.30], independent of stim_end.
    """
    ds = SimpleDataset.with_stimuli(
        session_id="S1",
        stim_starts=[0.10, 0.32],
        stim_ends=[0.19, 0.41],
        image_ids=[1, 2],
        n_units=4,
        sampling_rate=1000,
        duration=1.0,
    )
    sampler = PerStimulusSampler(
        ds,
        session_ids=["S1"],
        interval_source="stimuli",
        grouping_field="image_id",
        interval_start_lag_ms=0.0,
        fixed_window_duration_ms=200.0,
        cross_stim_padding_ms=20.0,
        label_extension_ms=100.0,
    )

    geoms = sampler.window_geometries()
    assert geoms[0] == (pytest.approx(0.10), pytest.approx(0.30))
    assert geoms[1] == (pytest.approx(0.32), pytest.approx(0.52))


def test_fixed_window_duration_overlap_within_padding_is_allowed():
    """An overlap into the next stim equal to cross_stim_padding_ms is OK."""
    ds = SimpleDataset.with_stimuli(
        session_id="S1",
        # next_stim_start = 0.30, fixed window for first stim ends at 0.32 → 20 ms overlap
        stim_starts=[0.12, 0.30],
        stim_ends=[0.20, 0.38],
        image_ids=[1, 2],
        n_units=4,
        sampling_rate=1000,
        duration=1.0,
    )
    sampler = PerStimulusSampler(
        ds,
        session_ids=["S1"],
        interval_source="stimuli",
        grouping_field="image_id",
        interval_start_lag_ms=0.0,
        fixed_window_duration_ms=200.0,
        cross_stim_padding_ms=20.0,
        label_extension_ms=100.0,
    )

    geoms = sampler.window_geometries()
    assert geoms[0][1] == pytest.approx(0.32)


def test_fixed_window_duration_overlap_beyond_padding_raises():
    """An overlap > cross_stim_padding_ms must raise loudly."""
    ds = SimpleDataset.with_stimuli(
        session_id="S1",
        # next_stim_start = 0.25, fixed window ends at 0.30 → 50 ms overlap
        stim_starts=[0.10, 0.25],
        stim_ends=[0.18, 0.33],
        image_ids=[1, 2],
        n_units=4,
        sampling_rate=1000,
        duration=1.0,
    )
    with pytest.raises(ValueError, match="cross_stim_padding_ms"):
        PerStimulusSampler(
            ds,
            session_ids=["S1"],
            interval_source="stimuli",
            grouping_field="image_id",
            interval_start_lag_ms=0.0,
            fixed_window_duration_ms=200.0,
            cross_stim_padding_ms=20.0,
            label_extension_ms=100.0,
        )


def test_mutually_exclusive_with_interval_end_lag():
    """Setting both `interval_end_lag_ms` and `fixed_window_duration_ms` raises."""
    ds = SimpleDataset.with_stimuli(
        session_id="S1",
        stim_starts=[0.10],
        stim_ends=[0.19],
        image_ids=[1],
        n_units=4,
        sampling_rate=1000,
        duration=1.0,
    )
    with pytest.raises(ValueError, match="exactly one"):
        PerStimulusSampler(
            ds,
            session_ids=["S1"],
            interval_source="stimuli",
            grouping_field="image_id",
            interval_end_lag_ms=0.0,
            fixed_window_duration_ms=200.0,
            label_extension_ms=100.0,
        )


def test_neither_set_raises():
    """Omitting both windowing params raises (no default)."""
    ds = SimpleDataset.with_stimuli(
        session_id="S1",
        stim_starts=[0.10],
        stim_ends=[0.19],
        image_ids=[1],
        n_units=4,
        sampling_rate=1000,
        duration=1.0,
    )
    with pytest.raises(ValueError, match="exactly one"):
        PerStimulusSampler(
            ds,
            session_ids=["S1"],
            interval_source="stimuli",
            grouping_field="image_id",
        )
