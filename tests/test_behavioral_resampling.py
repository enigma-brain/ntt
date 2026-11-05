"""Tests for resample_behavioral_data() in alignment.py.

The lower tests in this file are intentionally repetitive and largely redundant
with the earlier value-level cases.
"""

import numpy as np
import pytest

from ntt.evaluation.utils.alignment import resample_behavioral_data

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_timestamps(start, stop, step):
    """Inclusive-ish arange returning float64 timestamps."""
    return np.arange(start, stop, step, dtype=np.float64)


# ---------------------------------------------------------------------------
# Continuous field resampling
# ---------------------------------------------------------------------------


class TestContinuousResampling:
    """Linear interpolation for continuous (float) fields."""

    def test_1d_downsample_known_values(self):
        """1-ms source → 20-ms target for a linear signal: interpolation should
        reproduce the exact linear values."""
        src_ts = _make_timestamps(0, 100, 1)  # 0..99 ms, 100 points
        tgt_ts = _make_timestamps(0, 100, 20)  # 0, 20, 40, 60, 80
        # y = 2*t + 1  (perfectly linear)
        data = {"velocity": 2.0 * src_ts + 1.0}

        result = resample_behavioral_data(data, src_ts, tgt_ts)

        expected = 2.0 * tgt_ts + 1.0
        np.testing.assert_allclose(result["velocity"], expected)

    def test_2d_cursor_pos(self):
        """2-D continuous field (e.g. cursor_pos) downsampled from 1 ms to 20 ms."""
        src_ts = _make_timestamps(0, 100, 1)
        tgt_ts = _make_timestamps(0, 100, 20)
        # x = t,  y = -t
        cursor = np.stack([src_ts, -src_ts], axis=1)
        data = {"cursor_pos": cursor}

        result = resample_behavioral_data(data, src_ts, tgt_ts)

        assert result["cursor_pos"].shape == (len(tgt_ts), 2)
        np.testing.assert_allclose(result["cursor_pos"][:, 0], tgt_ts)
        np.testing.assert_allclose(result["cursor_pos"][:, 1], -tgt_ts)

    def test_3d_velocity(self):
        """3-D continuous field (e.g. velocity_xyz)."""
        src_ts = _make_timestamps(0, 50, 1)
        tgt_ts = _make_timestamps(0, 50, 10)
        vel = np.stack([src_ts, src_ts**2 / 50, np.sin(src_ts)], axis=1)
        data = {"velocity_xyz": vel}

        result = resample_behavioral_data(data, src_ts, tgt_ts)

        assert result["velocity_xyz"].shape == (len(tgt_ts), 3)
        # At source grid points the interpolation should be exact
        for d in range(3):
            np.testing.assert_allclose(
                result["velocity_xyz"][:, d],
                np.interp(tgt_ts, src_ts, vel[:, d]),
            )

    def test_interpolation_at_midpoints(self):
        """Target timestamps fall exactly between source timestamps —
        linear interpolation should return the average of neighbours."""
        src_ts = np.array([0.0, 10.0, 20.0])
        tgt_ts = np.array([5.0, 15.0])
        data = {"signal": np.array([0.0, 10.0, 20.0])}

        result = resample_behavioral_data(data, src_ts, tgt_ts)

        np.testing.assert_allclose(result["signal"], [5.0, 15.0])


# ---------------------------------------------------------------------------
# Categorical field resampling
# ---------------------------------------------------------------------------


class TestCategoricalResampling:
    """Nearest-neighbor for categorical / integer fields."""

    def test_integer_dtype_uses_nearest_neighbor(self):
        """Integer dtype triggers nearest-neighbor even without explicit
        categorical_fields list."""
        src_ts = np.array([0.0, 10.0, 20.0, 30.0])
        tgt_ts = np.array([4.0, 16.0, 25.0])
        labels = np.array([0, 1, 2, 3], dtype=np.int64)
        data = {"labels": labels}

        result = resample_behavioral_data(data, src_ts, tgt_ts, categorical_fields=[])

        # 4 is closer to 0 than to 10 → label 0
        # 16 is closer to 20 than to 10 → label 2
        # 25 is closer to 20 than to 30 → label 2
        np.testing.assert_array_equal(result["labels"], [0, 2, 2])

    def test_explicit_categorical_field(self):
        """Field named in categorical_fields uses nearest-neighbor
        regardless of dtype."""
        src_ts = np.array([0.0, 10.0, 20.0])
        tgt_ts = np.array([3.0, 14.0])
        # Float dtype but declared categorical
        phase = np.array([1.0, 2.0, 3.0])
        data = {"movement_phase": phase}

        result = resample_behavioral_data(
            data, src_ts, tgt_ts, categorical_fields=["movement_phase"]
        )

        # 3 closer to 0 → 1.0;  14 closer to 10 → 2.0
        np.testing.assert_array_equal(result["movement_phase"], [1.0, 2.0])

    def test_default_categorical_fields(self):
        """Default categorical list includes 'movement_phase' etc."""
        src_ts = np.array([0.0, 10.0, 20.0])
        tgt_ts = np.array([6.0])
        data = {"movement_phase": np.array([10.0, 20.0, 30.0])}

        result = resample_behavioral_data(data, src_ts, tgt_ts)

        # 6 is closer to 10 → value 20.0
        np.testing.assert_array_equal(result["movement_phase"], [20.0])

    def test_categorical_boundary_exact_midpoint(self):
        """When target is exactly at the midpoint between two source timestamps,
        document which neighbour wins (argmin picks the first minimum)."""
        src_ts = np.array([0.0, 10.0])
        tgt_ts = np.array([5.0])
        labels = np.array([0, 1], dtype=np.int64)
        data = {"label": labels}

        result = resample_behavioral_data(data, src_ts, tgt_ts, categorical_fields=[])

        # np.argmin returns first occurrence → index 0 → label 0
        assert result["label"][0] == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Same-length, out-of-range, single-element, etc."""

    def test_same_length_timestamps(self):
        """Source and target timestamps have the same length."""
        ts = np.array([0.0, 5.0, 10.0])
        data = {"x": np.array([1.0, 2.0, 3.0])}

        result = resample_behavioral_data(data, ts, ts)

        np.testing.assert_allclose(result["x"], data["x"])

    def test_identical_timestamps(self):
        """When source == target, output equals input (identity).

        What's the difference to the test above?
        """
        ts = _make_timestamps(0, 50, 5)
        vals = np.random.default_rng(42).standard_normal(len(ts))
        data = {"v": vals}

        result = resample_behavioral_data(data, ts, ts)

        np.testing.assert_allclose(result["v"], vals)

    def test_target_outside_source_range_continuous(self):
        """np.interp clamps to boundary values for out-of-range targets."""
        src_ts = np.array([10.0, 20.0, 30.0])
        tgt_ts = np.array([0.0, 40.0])  # both outside
        data = {"sig": np.array([100.0, 200.0, 300.0])}

        result = resample_behavioral_data(data, src_ts, tgt_ts)

        # np.interp clamps: left → first value, right → last value
        np.testing.assert_allclose(result["sig"], [100.0, 300.0])

    def test_target_outside_source_range_categorical(self):
        """Nearest-neighbor for out-of-range targets picks boundary element."""
        src_ts = np.array([10.0, 20.0, 30.0])
        tgt_ts = np.array([0.0, 40.0])
        labels = np.array([5, 6, 7], dtype=np.int64)
        data = {"cat": labels}

        result = resample_behavioral_data(data, src_ts, tgt_ts, categorical_fields=[])

        # 0 closest to 10 → 5;  40 closest to 30 → 7
        np.testing.assert_array_equal(result["cat"], [5, 7])

    def test_single_source_timestamp(self):
        """Only one source point — everything maps to that value."""
        src_ts = np.array([5.0])
        tgt_ts = np.array([0.0, 5.0, 10.0])
        data = {"val": np.array([42.0])}

        result = resample_behavioral_data(data, src_ts, tgt_ts)

        np.testing.assert_allclose(result["val"], [42.0, 42.0, 42.0])

    def test_length_mismatch_raises(self):
        """Data length != source_timestamps length should raise."""
        src_ts = np.array([0.0, 1.0, 2.0])
        tgt_ts = np.array([0.5])
        data = {"bad": np.array([1.0, 2.0])}  # length 2 != 3

        with pytest.raises(AssertionError, match="length"):
            resample_behavioral_data(data, src_ts, tgt_ts)

    def test_multiple_fields_mixed(self):
        """Mix of continuous and categorical fields in one call."""
        src_ts = np.array([0.0, 10.0, 20.0])
        tgt_ts = np.array([5.0, 15.0])
        data = {
            "cursor_x": np.array([0.0, 10.0, 20.0]),  # continuous
            "trial_id": np.array([1, 1, 2], dtype=np.int64),  # categorical (int)
        }

        result = resample_behavioral_data(data, src_ts, tgt_ts, categorical_fields=[])

        # Continuous: linear interp → 5, 15
        np.testing.assert_allclose(result["cursor_x"], [5.0, 15.0])
        # Categorical (int dtype): nearest → 5 closer to 0→1, 15 closer to 10→1? or 20→2?
        # 5 equidistant 0,10 → argmin picks first → index 0 → 1
        # 15 equidistant 10,20 → argmin picks first → index 1 → 1
        np.testing.assert_array_equal(result["trial_id"], [1, 1])
