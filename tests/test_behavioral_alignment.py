"""Tests for resolution-independent behavioral alignment.

Core tests for resample_fields_to_token_timestamps and align_behavioral_to_token_chunks.
"""

import numpy as np
import pytest

from ntt.evaluation.metrics.downstream_prediction_utils import (
    align_behavioral_to_chunks,
)
from ntt.evaluation.utils.alignment import resample_fields_to_token_timestamps


def test_resample_filter_fields_basic():
    """Only selected token-level filtering fields are resampled."""
    behavioral_data = {
        "movement_phase": np.array([0, 0, 1, 1, 2, 2, 2, 2, 3, 3]),
        "trial_id": np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]),
        "cursor_pos": np.random.randn(10, 2),  # continuous — should NOT be resampled
    }
    source_ts = np.arange(10, dtype=np.float64) * 0.001 + 0.0005  # 1ms bins, centered
    target_ts = np.array([0.002, 0.006])  # 2 token centers

    result = resample_fields_to_token_timestamps(
        behavioral_data,
        source_ts,
        target_ts,
        fields=["movement_phase", "trial_id"],
    )

    # Only requested filtering fields in result
    assert set(result.keys()) == {"movement_phase", "trial_id"}
    assert len(result["movement_phase"]) == 2
    assert len(result["trial_id"]) == 2
    # Nearest-neighbor: token at t=0.002 -> source idx 1 or 2 (phase 0 or 1)
    # token at t=0.006 -> source idx 5 or 6 (phase 2)
    assert result["movement_phase"][1] == 2


def test_resample_filter_fields_default_fields():
    """Default token-level filtering fields are used when not specified."""
    behavioral_data = {
        "movement_phase": np.array([0, 0, 1, 1]),
        "cursor_vel": np.random.randn(4, 2),
    }
    source_ts = np.arange(4, dtype=np.float64) * 0.001 + 0.0005
    target_ts = np.array([0.001])

    result = resample_fields_to_token_timestamps(behavioral_data, source_ts, target_ts)
    assert "movement_phase" in result
    assert "cursor_vel" not in result


def test_resample_filter_fields_empty_target():
    """Empty target timestamps returns empty arrays."""
    behavioral_data = {
        "movement_phase": np.array([0, 1, 2]),
    }
    source_ts = np.arange(3, dtype=np.float64) * 0.001 + 0.0005
    target_ts = np.array([])

    result = resample_fields_to_token_timestamps(behavioral_data, source_ts, target_ts)
    assert "movement_phase" in result
    assert len(result["movement_phase"]) == 0


def test_resample_filter_fields_rejects_non_categorical():
    """ValueError is raised for non-categorical, non-integer fields."""
    behavioral_data = {
        "cursor_pos": np.random.randn(10, 2),  # float, not in KNOWN_CATEGORICAL_FIELDS
    }
    source_ts = np.arange(10, dtype=np.float64) * 0.001 + 0.0005
    target_ts = np.array([0.002, 0.006])

    with pytest.raises(ValueError, match="KNOWN_CATEGORICAL_FIELDS"):
        resample_fields_to_token_timestamps(
            behavioral_data,
            source_ts,
            target_ts,
            fields=["cursor_pos"],
        )


# ── align_behavioral_to_chunks tests ─────────────────────────────────


def test_token_chunk_mapping_uses_documented_interval_convention():
    """Verify that chunk at timestamp T maps to behavioral indices [T*sr, T*sr + K).


    With 100 behavioral bins at 1kHz, tokens at 10ms, 2 tokens per chunk:
    chunk at timestamp 0.005s -> start_idx = round(0.005 * 1000) = 5,
    K_input_res = 2 * 10 = 20, so indices [5, 25).
    """
    n_behavioral = 100
    sampling_rate = 1000.0

    # Behavioral data: each bin has a unique value so we can verify exact indices
    behavioral_values = np.arange(n_behavioral, dtype=np.float64)
    behavioral_data = {
        "cursor_pos": behavioral_values.reshape(-1, 1),
        "trial_id": np.ones(n_behavioral),
    }

    # Single chunk at index 0: chunk_window_ms=20ms with sampling_rate=1000Hz
    # → K_input_res = 20ms / (1000/1000) = 20 bins
    # Chunk 0 starts at index 0, spans [0, 20).
    chunk_indices = np.array([0])
    chunk_window_ms = 20.0  # 2 tokens × 10ms

    y, _, keep_mask, metadata = align_behavioral_to_chunks(
        chunk_indices=chunk_indices,
        chunk_window_ms=chunk_window_ms,
        behavioral_data_input_res=behavioral_data,
        sampling_rate=sampling_rate,
        target_field="cursor_pos",
        mode="regression",
        grouping_field="trial_id",
    )

    assert keep_mask[0], "Chunk should be kept"
    assert metadata["K_per_chunk"] == 20
    # y should contain behavioral indices 0 through 19 (flattened 1D target)
    expected = behavioral_values[0:20]
    np.testing.assert_array_equal(y[0], expected)


def test_dropping_one_token_chunk_does_not_change_neighbors():
    """Dropping the middle chunk (low majority) does not alter neighbor y values.


    Three chunks (indices 0, 1, 2), each spanning 20ms = 20 bins at 1kHz.
    The middle chunk has a near-even label split (<75% majority) so it
    gets dropped by the classification majority threshold.
    First and last chunks' y values must be identical to running alignment
    on just those chunks individually.
    """
    sampling_rate = 1000.0
    n_behavioral = 100

    # 3 chunks, chunk_window_ms=20 -> each chunk spans 20 behavioral bins
    # chunk 0: bins [0, 20)   — all label 0
    # chunk 1: bins [20, 40)  — 10x label 0 + 10x label 1 = 50% majority (< 75% threshold)
    # chunk 2: bins [40, 60)  — all label 1
    chunk_indices = np.array([0, 1, 2])
    chunk_window_ms = 20.0

    labels = np.zeros(n_behavioral)
    labels[30:] = 1.0  # splits chunk 1 evenly at bin 30

    behavioral_data = {
        "cursor_pos": labels,
        "trial_id": np.ones(n_behavioral),
    }

    y_all, _, keep_mask_all, meta_all = align_behavioral_to_chunks(
        chunk_indices=chunk_indices,
        chunk_window_ms=chunk_window_ms,
        behavioral_data_input_res=behavioral_data,
        sampling_rate=sampling_rate,
        target_field="cursor_pos",
        mode="classification",
        grouping_field="trial_id",
    )

    # Middle chunk should be dropped (50% majority < 75% threshold)
    assert not keep_mask_all[1], "Middle chunk should be dropped (low majority)"
    assert keep_mask_all[0] and keep_mask_all[2], "First and last chunks should survive"
    assert meta_all["n_chunks_dropped_low_majority"] >= 1

    # Run alignment on just chunk 0 individually
    y_first, _, keep_first, _ = align_behavioral_to_chunks(
        chunk_indices=np.array([0]),
        chunk_window_ms=chunk_window_ms,
        behavioral_data_input_res=behavioral_data,
        sampling_rate=sampling_rate,
        target_field="cursor_pos",
        mode="classification",
        grouping_field="trial_id",
    )

    # Run alignment on just chunk 2 individually
    y_last, _, keep_last, _ = align_behavioral_to_chunks(
        chunk_indices=np.array([2]),
        chunk_window_ms=chunk_window_ms,
        behavioral_data_input_res=behavioral_data,
        sampling_rate=sampling_rate,
        target_field="cursor_pos",
        mode="classification",
        grouping_field="trial_id",
    )

    assert keep_first[0] and keep_last[0]
    np.testing.assert_array_equal(y_all[0], y_first[0])
    np.testing.assert_array_equal(y_all[1], y_last[0])


def test_grouping_values_align_with_kept_token_chunks():
    """grouping_values length matches keep_mask.sum() and values are correct.


    4 chunks of 20ms each at 1kHz:
    chunk 0: bins [0, 20) trial 1
    chunk 1: bins [20, 40) trial 1
    chunk 2: bins [40, 60) trial 2
    chunk 3: bins [60, 80) trial 2
    """
    sampling_rate = 1000.0
    n_behavioral = 100

    chunk_indices = np.array([0, 1, 2, 3])
    chunk_window_ms = 20.0  # 2 tokens * 10ms

    trial_ids = np.ones(n_behavioral)
    trial_ids[40:] = 2.0  # boundary at chunk 2 start -> clean split

    behavioral_data = {
        "cursor_pos": np.random.randn(n_behavioral, 2),
        "trial_id": trial_ids,
    }

    y, grouping_values, keep_mask, _ = align_behavioral_to_chunks(
        chunk_indices=chunk_indices,
        chunk_window_ms=chunk_window_ms,
        behavioral_data_input_res=behavioral_data,
        sampling_rate=sampling_rate,
        target_field="cursor_pos",
        mode="regression",
        grouping_field="trial_id",
    )

    assert grouping_values is not None
    assert len(grouping_values) == keep_mask.sum(), (
        f"grouping_values length {len(grouping_values)} != kept chunks {keep_mask.sum()}"
    )
    # All 4 chunks have uniform trial_id within their range, so all are kept
    assert keep_mask.sum() == 4
    np.testing.assert_array_equal(grouping_values, [1.0, 1.0, 2.0, 2.0])


def test_grouping_values_stay_aligned_after_early_dropped_chunk():
    """Dropping an early chunk must not shift later grouping labels.


    4 chunks of 20ms each at 1kHz:
    chunk 0: bins [0, 20)   - target label split 50/50 → below 75% majority → dropped
    chunk 1: bins [20, 40)  - all label 7, trial 2
    chunk 2: bins [40, 60)  - all label 7, trial 2
    chunk 3: bins [60, 80)  - all label 7, trial 3
    """
    sampling_rate = 1000.0
    n_behavioral = 120

    chunk_indices = np.array([0, 1, 2, 3])
    chunk_window_ms = 20.0

    trial_ids = np.ones(n_behavioral)
    trial_ids[20:60] = 2.0
    trial_ids[60:] = 3.0

    # Target labels: chunk 0 has 50% label 7, 50% label 8 → low majority → dropped
    target_labels = np.full(n_behavioral, 7.0)
    target_labels[10:20] = 8.0  # bins 10-19 in chunk 0

    behavioral_data = {
        "trial_target_id": target_labels,
        "trial_id": trial_ids,
    }

    y, grouping_values, keep_mask, metadata = align_behavioral_to_chunks(
        chunk_indices=chunk_indices,
        chunk_window_ms=chunk_window_ms,
        behavioral_data_input_res=behavioral_data,
        sampling_rate=sampling_rate,
        target_field="trial_target_id",
        mode="classification",
        grouping_field="trial_id",
        classification_majority_threshold=0.75,
    )

    assert metadata["n_chunks_dropped_low_majority"] == 1
    np.testing.assert_array_equal(keep_mask, [False, True, True, True])
    assert len(y) == 3
    assert grouping_values is not None
    np.testing.assert_array_equal(grouping_values, [2.0, 2.0, 3.0])


def test_missing_required_fields_raises():
    """align_behavioral_to_chunks raises when target_field is missing."""
    behavioral_data = {
        "trial_id": np.ones(50),
    }

    with pytest.raises(AssertionError, match="target_field.*not found"):
        align_behavioral_to_chunks(
            chunk_indices=np.array([0]),
            chunk_window_ms=20.0,
            behavioral_data_input_res=behavioral_data,
            sampling_rate=1000.0,
            target_field="cursor_pos",
            mode="regression",
        )


def test_behavioral_bin_size_downsampling():
    """behavioral_bin_size_ms=5.0 at 1kHz produces K = chunk_ms / 5.


    A 20ms chunk at 1ms input resolution has K=20.
    With behavioral_bin_size_ms=5.0, K should be 20/5 = 4.
    """
    n_behavioral = 100
    behavioral_data = {
        "cursor_pos": np.arange(n_behavioral, dtype=np.float64).reshape(-1, 1),
        "trial_id": np.ones(n_behavioral),
    }

    chunk_indices = np.array([0])
    chunk_window_ms = 20.0  # 2 tokens * 10ms

    # Full resolution: K = 20
    _, _, keep_full, meta_full = align_behavioral_to_chunks(
        chunk_indices=chunk_indices,
        chunk_window_ms=chunk_window_ms,
        behavioral_data_input_res=behavioral_data,
        sampling_rate=1000.0,
        target_field="cursor_pos",
        mode="regression",
        grouping_field="trial_id",
    )

    # Downsampled: K = 4
    y_ds, _, keep_ds, meta_ds = align_behavioral_to_chunks(
        chunk_indices=chunk_indices,
        chunk_window_ms=chunk_window_ms,
        behavioral_data_input_res=behavioral_data,
        sampling_rate=1000.0,
        target_field="cursor_pos",
        mode="regression",
        grouping_field="trial_id",
        behavioral_bin_size_ms=5.0,
    )

    assert meta_full["K_per_chunk"] == 20
    assert meta_ds["K_per_chunk"] == 4
    assert meta_ds["K_per_chunk"] == meta_full["K_per_chunk"] // 5
    # Downsampled y should have 4 values (1D target)
    assert y_ds.shape[1] == 4


def test_behavioral_bin_size_downsampling_uses_ceil_when_not_divisible():
    """Non-divisible chunk/bin ratios should match Python slicing semantics.

    chunk_window_ms=180, behavioral_bin_size_ms=50 at 1kHz:
    K_input_res = 180, downsample factor = 50, K_per_chunk = ceil(180/50) = 4.
    Chunk 0 starts at bin 0, downsampled bins at indices 0, 50, 100, 150.
    """
    n_behavioral = 300
    behavioral_data = {
        "cursor_pos": np.arange(n_behavioral, dtype=np.float64).reshape(-1, 1),
        "trial_id": np.ones(n_behavioral),
    }

    y, _, keep_mask, meta = align_behavioral_to_chunks(
        chunk_indices=np.array([0]),
        chunk_window_ms=180.0,  # 3 tokens * 60ms
        behavioral_data_input_res=behavioral_data,
        sampling_rate=1000.0,
        target_field="cursor_pos",
        mode="regression",
        grouping_field="trial_id",
        behavioral_bin_size_ms=50.0,
    )

    assert keep_mask[0]
    assert meta["K_input_res_per_chunk"] == 180
    assert meta["K_per_chunk"] == 4
    np.testing.assert_array_equal(y[0], [0.0, 50.0, 100.0, 150.0])
