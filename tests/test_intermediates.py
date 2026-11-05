"""Tests for intermediate computations in IntermediatesCache."""

import numpy as np
import pandas as pd
import pytest

from ntt.evaluation.core.intermediates import IntermediatesCache
from ntt.evaluation.core.types import EvalData
from ntt.evaluation.metrics.base import IntermediateRequest


class TestISIIntermediate:
    """Test ISI (inter-spike interval) computation."""

    def test_isi_evenly_spaced(self):
        """Test ISI with evenly spaced spikes."""
        # Create spike times: unit 0 at [0, 1, 2, ..., 9], unit 1 at [5, 6, 7, ..., 14]
        spike_times = [
            np.arange(10, dtype=np.float64),
            np.arange(10, dtype=np.float64) + 5.0,
        ]

        # Create EvalData
        original = EvalData.from_spike_times_list(spike_times, duration=20.0)
        reconstructed = EvalData.from_spike_times_list(
            spike_times, duration=20.0
        )  # same for simplicity

        cache = IntermediatesCache(original, reconstructed)

        # Expected ISIs: np.diff([0,1,2,...,9]) = [1,1,1,1,1,1,1,1,1]
        expected_isi = np.ones(9, dtype=np.float64)

        # Test both original and reconstructed
        for key in ["isi_original", "isi_reconstructed"]:
            request = IntermediateRequest.from_dict(key, {"smoothing_kernel": 1})
            isi = cache.get(request)
            assert len(isi) == 2, f"Should have ISI for 2 units ({key})"
            np.testing.assert_array_almost_equal(isi[0], expected_isi)
            np.testing.assert_array_almost_equal(isi[1], expected_isi)

    def test_isi_exponentially_spaced(self):
        """Test ISI with exponentially spaced spikes."""
        # Create spike times: unit 0 at [0, 1, 4, 9, 16, ...], unit 1 shifted by 5
        spike_times_unit0 = np.arange(10, dtype=np.float64) ** 2
        spike_times_unit1 = np.arange(10, dtype=np.float64) ** 2 + 5.0

        spike_times = [spike_times_unit0, spike_times_unit1]

        # Create EvalData
        original = EvalData.from_spike_times_list(spike_times, duration=100.0)
        # same for simplicity
        reconstructed = EvalData.from_spike_times_list(spike_times, duration=100.0)

        cache = IntermediatesCache(original, reconstructed)

        # Expected ISIs: np.diff([0, 1, 4, 9, 16, 25, 36, 49, 64, 81])
        expected_isi = [1, 3, 5, 7, 9, 11, 13, 15, 17]
        # expected_isi = np.diff(spike_times_unit0)

        # Test both original and reconstructed
        for key in ["isi_original", "isi_reconstructed"]:
            request = IntermediateRequest.from_dict(key, {"smoothing_kernel": 1})
            isi = cache.get(request)
            assert len(isi) == 2, f"Should have ISI for 2 units ({key})"
            np.testing.assert_array_almost_equal(isi[0], expected_isi)
            np.testing.assert_array_almost_equal(
                isi[1], expected_isi
            )  # Same pattern, shifted spikes

    def test_isi_count(self):
        """Test ISI with empty spike train."""
        spike_times = [
            np.array([], dtype=np.float64),
            np.arange(5, dtype=np.float64),
            np.array([5.0], dtype=np.float64),
            np.arange(3, dtype=np.float64),
        ]

        original = EvalData.from_spike_times_list(spike_times, duration=10.0)
        reconstructed = EvalData.from_spike_times_list(spike_times, duration=10.0)

        cache = IntermediatesCache(original, reconstructed)

        # Test both original and reconstructed
        for key in ["isi_original", "isi_reconstructed"]:
            request = IntermediateRequest.from_dict(key, {"smoothing_kernel": 1})
            isi = cache.get(request)
            # Empty spike train should give empty ISI array
            assert len(isi[0]) == 0, f"Empty spike train should have empty ISI ({key})"
            assert len(isi[1]) == 4, f"5 spikes should give 4 ISIs ({key})"
            assert len(isi[2]) == 0, f"1 spike should give 0 ISIs ({key})"
            assert len(isi[3]) == 2, f"3 spikes should give 2 ISIs ({key})"


class TestSpikeMatchesIntermediate:
    """Test spike matching computation."""

    def test_spike_matches_comprehensive(self):
        """Test spike matching with different scenarios across units."""
        # Set tolerance to 10ms = 0.01s (max allowed by config)
        tolerance_ms = 10.0

        # Unit 0: Perfect matches - 10 evenly spaced spikes in 1s
        unit0_spikes = np.linspace(0.0, 0.9, 10, dtype=np.float64)

        # Unit 1: Shifted by half tolerance (spacing > tolerance)
        # Spikes every 20ms, shifted by 5ms (half of 10ms)
        unit1_orig = np.array([0.0, 0.02, 0.04, 0.06, 0.08], dtype=np.float64)
        unit1_recon = unit1_orig + 0.005  # All within tolerance

        # Unit 2: Ambiguous - 2 original spikes, 1 reconstructed in between
        # (within tolerance of both)
        # Original at [0.4, 0.42] (20ms apart), reconstructed at [0.41]
        # (10ms from each)
        unit2_orig = np.array([0.4, 0.42], dtype=np.float64)
        unit2_recon = np.array([0.41], dtype=np.float64)

        # Unit 3: No matches - original at start, reconstructed at end (outside tolerance)
        unit3_orig = np.array([0.0, 0.01], dtype=np.float64)
        unit3_recon = np.array([0.9, 0.91], dtype=np.float64)

        # Unit 4: No spikes - edge case
        unit4_spikes = np.array([], dtype=np.float64)

        # Create EvalData
        original = EvalData.from_spike_times_list(
            [unit0_spikes, unit1_orig, unit2_orig, unit3_orig, unit4_spikes], duration=1.5
        )
        reconstructed = EvalData.from_spike_times_list(
            [unit0_spikes, unit1_recon, unit2_recon, unit3_recon, unit4_spikes], duration=1.5
        )

        cache = IntermediatesCache(original, reconstructed)

        # Get spike matches
        request = IntermediateRequest.from_dict("spike_matches", {"tolerance_ms": tolerance_ms})
        matches = cache.get(request)

        # Define expected values for each unit
        expected = [
            {
                "unit": 0,
                "description": "Perfect matches",
                "true_positives": 10,
                "false_positives": 0,
                "false_negatives": 0,
                "n_matched": 10,
                "n_timing_errors": 10,
                "expected_timing_errors": np.zeros(10),
                "matched_original_idx_check": None,  # No specific check needed
            },
            {
                "unit": 1,
                "description": "Shifted matches (all within tolerance)",
                "true_positives": 5,
                "false_positives": 0,
                "false_negatives": 0,
                "n_matched": 5,
                "n_timing_errors": 5,
                "expected_timing_errors": np.full(5, 0.005),  # +5ms shift
                "matched_original_idx_check": None,
            },
            {
                "unit": 2,
                "description": "Ambiguous case (greedy nearest-neighbor)",
                "true_positives": 1,
                "false_positives": 0,
                "false_negatives": 1,
                "n_matched": 1,
                "n_timing_errors": 1,
                "expected_timing_errors": np.array([0.01]),  # 0.41 - 0.4 = 10ms
                "matched_original_idx_check": 0,  # Should match to first original spike
            },
            {
                "unit": 3,
                "description": "No matches",
                "true_positives": 0,
                "false_positives": 2,
                "false_negatives": 2,
                "n_matched": 0,
                "n_timing_errors": 0,
                "expected_timing_errors": None,  # No timing errors
                "matched_original_idx_check": None,
            },
            {
                "unit": 4,
                "description": "No spikes (edge case)",
                "true_positives": 0,
                "false_positives": 0,
                "false_negatives": 0,
                "n_matched": 0,
                "n_timing_errors": 0,
                "expected_timing_errors": None,
                "matched_original_idx_check": None,
            },
        ]

        # Test each unit
        for exp in expected:
            unit_idx = exp["unit"]
            desc = exp["description"]

            assert matches["true_positives"][unit_idx] == exp["true_positives"], (
                f"Unit {unit_idx} ({desc}): expected {exp['true_positives']} TPs"
            )
            assert matches["false_positives"][unit_idx] == exp["false_positives"], (
                f"Unit {unit_idx} ({desc}): expected {exp['false_positives']} FPs"
            )
            assert matches["false_negatives"][unit_idx] == exp["false_negatives"], (
                f"Unit {unit_idx} ({desc}): expected {exp['false_negatives']} FNs"
            )
            assert len(matches["matched_original_idx"][unit_idx]) == exp["n_matched"], (
                f"Unit {unit_idx} ({desc}): expected {exp['n_matched']} matches"
            )
            assert len(matches["timing_errors"][unit_idx]) == exp["n_timing_errors"], (
                f"Unit {unit_idx} ({desc}): expected {exp['n_timing_errors']} timing errors"
            )

            if exp["expected_timing_errors"] is not None:
                np.testing.assert_array_almost_equal(
                    matches["timing_errors"][unit_idx],
                    exp["expected_timing_errors"],
                    err_msg=f"Unit {unit_idx} ({desc}): timing errors mismatch",
                )

            if exp["matched_original_idx_check"] is not None:
                expected_idx = exp["matched_original_idx_check"]
                assert matches["matched_original_idx"][unit_idx][0] == expected_idx, (
                    f"Unit {unit_idx} ({desc}): expected first match to original spike "
                    f"{expected_idx}"
                )

        # Verify structure of all outputs
        for key in ["matched_original_idx", "matched_reconstructed_idx", "timing_errors"]:
            assert len(matches[key]) == 5, f"{key} should have entries for 5 units"
        for key in ["true_positives", "false_positives", "false_negatives"]:
            assert len(matches[key]) == 5, f"{key} should have entries for 5 units"


class TestSpikeMatchingSearchsortedEquivalence:
    """Verify the searchsorted-based matching is equivalent to the O(N²) reference."""

    @staticmethod
    def _reference_match_single_unit(
        original: np.ndarray, reconstructed: np.ndarray, tolerance: float
    ) -> dict:
        """Original O(N²) brute-force implementation kept as a reference."""
        matched_orig_idx = []
        matched_recon_idx = []
        timing_errors = []
        recon_used: set[int] = set()

        for i, t_orig in enumerate(original):
            if len(reconstructed) == 0:
                break
            distances = np.abs(reconstructed - t_orig)
            min_idx = int(np.argmin(distances))
            if distances[min_idx] <= tolerance and min_idx not in recon_used:
                matched_orig_idx.append(i)
                matched_recon_idx.append(min_idx)
                timing_errors.append(reconstructed[min_idx] - t_orig)
                recon_used.add(min_idx)

        tp = len(matched_orig_idx)
        return {
            "matched_original_idx": np.array(matched_orig_idx),
            "matched_reconstructed_idx": np.array(matched_recon_idx),
            "timing_errors": np.array(timing_errors),
            "true_positives": tp,
            "false_positives": len(reconstructed) - tp,
            "false_negatives": len(original) - tp,
        }

    @staticmethod
    def _assert_match_equal(result: dict, reference: dict, label: str = "") -> None:
        prefix = f"[{label}] " if label else ""
        assert result["true_positives"] == reference["true_positives"], (
            f"{prefix}TP mismatch: {result['true_positives']} != {reference['true_positives']}"
        )
        assert result["false_positives"] == reference["false_positives"], f"{prefix}FP mismatch"
        assert result["false_negatives"] == reference["false_negatives"], f"{prefix}FN mismatch"
        np.testing.assert_array_equal(
            result["matched_original_idx"],
            reference["matched_original_idx"],
            err_msg=f"{prefix}matched_original_idx mismatch",
        )
        np.testing.assert_array_equal(
            result["matched_reconstructed_idx"],
            reference["matched_reconstructed_idx"],
            err_msg=f"{prefix}matched_reconstructed_idx mismatch",
        )
        np.testing.assert_array_almost_equal(
            result["timing_errors"],
            reference["timing_errors"],
            err_msg=f"{prefix}timing_errors mismatch",
        )

    def _run_optimized(self, original, reconstructed, tolerance):
        """Call the optimized method via a minimal IntermediatesCache."""
        dummy_data = EvalData.from_spike_times_list([np.array([0.0])], duration=1.0)
        cache = IntermediatesCache(dummy_data, dummy_data)
        return cache._match_spikes_single_unit(original, reconstructed, tolerance)

    @pytest.mark.parametrize(
        "n_orig,n_recon,tolerance_ms,seed",
        [
            (100, 100, 1.0, 0),
            (500, 500, 1.0, 1),
            (1000, 1000, 0.5, 2),
            (200, 50, 1.0, 3),  # more orig than recon
            (50, 200, 1.0, 4),  # more recon than orig
            (500, 500, 10.0, 5),  # wide tolerance → many conflicts
            (1000, 1200, 1.0, 6),  # mismatched counts
        ],
    )
    def test_matches_reference_on_random_data(self, n_orig, n_recon, tolerance_ms, seed):
        """Optimized and reference produce identical results on random spike trains."""
        rng = np.random.default_rng(seed)
        duration = max(n_orig, n_recon) / 20.0
        orig = np.sort(rng.uniform(0, duration, n_orig))
        recon = np.sort(rng.uniform(0, duration, n_recon))

        tolerance = tolerance_ms / 1000.0
        ref = self._reference_match_single_unit(orig, recon, tolerance)
        opt = self._run_optimized(orig, recon, tolerance)

        self._assert_match_equal(
            opt,
            ref,
            label=f"n_orig={n_orig},n_recon={n_recon},tol={tolerance_ms}ms",
        )

    def test_matches_reference_jittered_reconstruction(self):
        """Typical case: reconstruction is a jittered copy with extra/missing spikes."""
        rng = np.random.default_rng(42)
        n = 500
        duration = n / 20.0
        orig = np.sort(rng.uniform(0, duration, n))
        # Take half the original spikes, jitter them, add some spurious ones
        recon = np.sort(
            np.concatenate(
                [
                    orig[::2] + rng.normal(0, 0.0003, len(orig[::2])),
                    rng.uniform(0, duration, n // 10),
                ]
            )
        )

        for tol_ms in [0.5, 1.0, 5.0]:
            tolerance = tol_ms / 1000.0
            ref = self._reference_match_single_unit(orig, recon, tolerance)
            opt = self._run_optimized(orig, recon, tolerance)
            self._assert_match_equal(opt, ref, label=f"jittered,tol={tol_ms}ms")

    def test_empty_arrays(self):
        """Edge cases with empty spike trains."""
        empty = np.array([], dtype=np.float64)
        some = np.array([0.1, 0.2, 0.3])

        for orig, recon, label in [
            (empty, empty, "both empty"),
            (empty, some, "orig empty"),
            (some, empty, "recon empty"),
        ]:
            ref = self._reference_match_single_unit(orig, recon, 0.001)
            opt = self._run_optimized(orig, recon, 0.001)
            self._assert_match_equal(opt, ref, label=label)

    def test_dense_spikes_with_conflicts(self):
        """Dense spike trains where multiple orig spikes compete for the same recon."""
        # Clusters of spikes within tolerance of each other
        orig = np.array([1.000, 1.0005, 1.001, 2.000, 2.0003])
        recon = np.array([1.0002, 1.0008, 2.0001])
        tolerance = 0.001  # 1ms

        ref = self._reference_match_single_unit(orig, recon, tolerance)
        opt = self._run_optimized(orig, recon, tolerance)
        self._assert_match_equal(opt, ref, label="dense clusters")


class TestACGIntermediate:
    """Test ACG (autocorrelogram) computation."""

    def test_acg_evenly_spaced(self):
        """Test ACG with evenly spaced spikes."""
        # Create spikes every 10ms from 0 to 990ms (100 spikes total)
        spike_times = [np.arange(100, dtype=np.float64) * 0.01]  # 0, 0.01, 0.02, ..., 0.99 seconds

        # Set duration to 0.99s to make boundary effects symmetric
        original = EvalData.from_spike_times_list(spike_times, duration=0.99)
        reconstructed = EvalData.from_spike_times_list(spike_times, duration=0.99)

        cache = IntermediatesCache(original, reconstructed)

        # Test both original and reconstructed (max_lag=100ms, bin_size=1ms, no smoothing)
        for key in ["acg_original", "acg_reconstructed"]:
            request = IntermediateRequest.from_dict(
                key,
                {"max_lag_ms": 100.0, "bin_size_ms": 1.0, "smoothing_kernel": 1},
            )
            acg = cache.get(request)
            assert len(acg) == 1, f"Should have ACG for 1 unit ({key})"

            acg_unit = acg[0]
            # n_bins = int(2 * window_size / bin_size + 1) = 2 * 100 + 1 = 201
            # Symmetric bins covering lags from -100ms to +100ms
            assert len(acg_unit) == 201, f"ACG should have 201 bins ({key})"

            # Center bin is at index 100 (lag=0), should be 0 (self-correlation excluded)
            center_idx = 100
            assert acg_unit[center_idx] == 0, f"Center bin (lag=0) should be 0 ({key})"

            # Expected counts at each lag (theoretical):
            # - At lag ±10ms: 99 pairs (each spike except first/last has neighbor 10ms away)
            # - At lag ±20ms: 98 pairs (100 - 2 edge spikes)
            # - At lag ±100ms: 90 pairs (100 - 10 edge spikes)
            # General formula: 100 - lag_ms/10

            lags_ms = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            expected_counts = 100 - lags_ms / 10  # [99, 98, 97, ..., 90]

            # Check positive lags
            pos_indices = center_idx + lags_ms
            pos_counts = acg_unit[pos_indices]
            np.testing.assert_array_equal(
                pos_counts,
                expected_counts,
                err_msg=f"Positive lag counts don't match expected ({key})",
            )  # noqa: E501

            # Check negative lags (should be symmetric)
            neg_indices = center_idx - lags_ms
            neg_counts = acg_unit[neg_indices]
            np.testing.assert_array_equal(
                neg_counts,
                expected_counts,
                err_msg=f"Negative lag counts don't match expected ({key})",
            )

            # Check symmetry directly
            err_msg = f"ACG should be symmetric ({key})"
            np.testing.assert_array_equal(pos_counts, neg_counts, err_msg=err_msg)

            # All non-multiple-of-10ms bins should be 0
            all_indices = np.arange(len(acg_unit))
            lags = all_indices - center_idx
            non_peak_mask = lags % 10 != 0
            non_peak_counts = acg_unit[non_peak_mask]
            assert np.all(non_peak_counts == 0), (
                f"Non-peak bins should be 0, found {np.sum(non_peak_counts > 0)} non-zero ({key})"
            )

    def test_acg_edge_cases(self):
        """Test ACG with edge cases: 0 spikes, 1 spike, 2 spikes."""
        spike_times = [
            np.array([], dtype=np.float64),  # 0 spikes
            np.array([0.5], dtype=np.float64),  # 1 spike
            np.array([0.2, 0.3], dtype=np.float64),  # 2 spikes (100ms apart)
        ]

        original = EvalData.from_spike_times_list(spike_times, duration=1.0)
        reconstructed = EvalData.from_spike_times_list(spike_times, duration=1.0)

        cache = IntermediatesCache(original, reconstructed)

        for key in ["acg_original", "acg_reconstructed"]:
            request = IntermediateRequest.from_dict(
                key,
                {"max_lag_ms": 100.0, "bin_size_ms": 1.0, "smoothing_kernel": 1},
            )
            acg = cache.get(request)
            assert len(acg) == 3, f"Should have ACG for 3 units ({key})"

            # Unit 0: 0 spikes → all zeros
            assert np.all(acg[0] == 0), f"Unit with 0 spikes should have all-zero ACG ({key})"

            # Unit 1: 1 spike → all zeros
            assert np.all(acg[1] == 0), f"Unit with 1 spike should have all-zero ACG ({key})"

            # Unit 2: 2 spikes at 0.2s and 0.3s (100ms apart)
            # Should have non-zero counts at lag ±100ms
            center_idx = 100

            # Check that there are exactly 2 non-zero bins
            # (one positive, one negative lag)
            nonzero_bins = np.where(acg[2] > 0)[0]
            msg = f"2 spikes should produce 2 non-zero bins, got {len(nonzero_bins)} ({key})"
            assert len(nonzero_bins) == 2, msg

            # With 100ms lag: negative lag at index 0, positive lag at index 200
            assert np.isclose(acg[2][0], 1), f"ACG should have 1 at lag -100ms (index 0) ({key})"
            assert np.isclose(acg[2][200], 1), (
                f"ACG should have 1 at lag +100ms (index 200) ({key})"
            )

            # Check symmetry: counts at positive and negative lags should match
            msg = f"ACG should be symmetric ({key})"
            assert acg[2][nonzero_bins[0]] == acg[2][nonzero_bins[1]], msg

            # Center should be 0
            assert acg[2][center_idx] == 0, f"Center bin should be 0 ({key})"

    def test_acg_smoothing(self):
        """Test ACG smoothing with boxcar kernel."""
        # Create simple pattern: spikes at 0, 0.05s, 0.1s (50ms apart)
        spike_times = [np.array([0.0, 0.05, 0.1], dtype=np.float64)]

        original = EvalData.from_spike_times_list(spike_times, duration=0.2)
        reconstructed = EvalData.from_spike_times_list(spike_times, duration=0.2)

        cache = IntermediatesCache(original, reconstructed)

        # First without smoothing
        req_no_smooth = IntermediateRequest.from_dict(
            "acg_original",
            {"max_lag_ms": 100.0, "bin_size_ms": 1.0, "smoothing_kernel": 1},
        )
        acg_no_smooth = cache.get(req_no_smooth)[0]

        # Then with smoothing (kernel size 3) — different params = separate cache entry
        req_smooth = IntermediateRequest.from_dict(
            "acg_original",
            {"max_lag_ms": 100.0, "bin_size_ms": 1.0, "smoothing_kernel": 3},
        )
        acg_smooth = cache.get(req_smooth)[0]

        # Check that smoothing changed the ACG
        assert not np.allclose(acg_no_smooth, acg_smooth), "Smoothing should change the ACG"

        # The smoothed version should have lower peaks (spread out)
        center_idx = 100
        peak_idx = center_idx + 50  # +50ms lag

        # Original should have sharp peak at exactly 50ms
        expected_peak = len(spike_times[0]) - 1
        assert acg_no_smooth[peak_idx] == expected_peak, (
            "Unsmoothed ACG should have peak at lag +50ms"
        )

        # Smoothed version should spread counts to neighboring bins
        # Check that neighboring bins have non-zero values after smoothing
        neighbors_increased = (
            acg_smooth[peak_idx - 1] > acg_no_smooth[peak_idx - 1]
            or acg_smooth[peak_idx + 1] > acg_no_smooth[peak_idx + 1]
        )
        assert neighbors_increased, "Smoothing should spread counts to neighboring bins"

    def test_acg_evenly_spaced_vs_pandas(self):
        """Compare ACG with evenly spaced spikes against pandas autocorr
        (same data as failing test)."""
        # Use same data as test_acg_evenly_spaced
        eps = 1e-6
        spike_times = [np.arange(100, dtype=np.float64) * 0.01 + eps]  # 100 spikes, 10ms apart
        print("number of spikes: ", len(spike_times[0]))

        original = EvalData.from_spike_times_list(spike_times, duration=0.99)
        reconstructed = EvalData.from_spike_times_list(spike_times, duration=0.99)

        # Compute ACG with 1ms bins
        cache = IntermediatesCache(original, reconstructed)
        request = IntermediateRequest.from_dict(
            "acg_original",
            {"max_lag_ms": 100.0, "bin_size_ms": 1.0, "smoothing_kernel": 1},
        )
        acg = cache.get(request)[0]

        # Create binary spike train for pandas (1ms bins)
        bin_size = 0.001  # 1ms
        n_bins = int(0.99 / bin_size)
        spike_train = np.zeros(n_bins)
        for st in spike_times[0]:
            bin_idx = int(st / bin_size)
            if bin_idx < n_bins:
                spike_train[bin_idx] = 1

        series = pd.Series(spike_train)

        # ACG has 201 bins: center at index 100 (lag=0)
        # covering lags from -100ms to +100ms
        center_idx = 100

        print("\n=== ACG vs Pandas Autocorr (Evenly Spaced, 10ms period) ===")
        header1 = (
            f"{'Lag':<8} {'ACG+':<8} {'ACG-':<8} {'Pandas':<10} "
            f"{'N_pairs':<10} {'Scaled':<10} {'Expected':<10}"
        )
        print(header1)
        header2 = (
            f"{'(ms)':<8} {'Count':<8} {'Count':<8} {'Autocorr':<10} "
            f"{'(theory)':<10} {'Pandas':<10} {'Count':<10}"
        )
        print(header2)
        print("-" * 80)

        # Test at multiples of 10ms (where peaks should be)
        for lag_ms in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            acg_count_pos = acg[center_idx + lag_ms]
            acg_count_neg = acg[center_idx - lag_ms]
            pandas_corr = series.autocorr(lag=lag_ms)

            # Number of valid spike pairs at this lag (theoretical)
            num_valid_pairs = 100 - lag_ms / 10  # 99, 98, ..., 90

            # Scale pandas autocorr by number of valid pairs
            scaled_pandas = pandas_corr * num_valid_pairs

            print(
                f"{lag_ms:<8} {acg_count_pos:<8.1f} {acg_count_neg:<8.1f} {pandas_corr:<10.6f} "
                f"{num_valid_pairs:<10.1f} {scaled_pandas:<10.1f} {num_valid_pairs:<10.1f}"
            )

            # Pandas should show high autocorr for periodic spikes (close to 1.0)
            assert pandas_corr > 0.8, (
                f"Pandas autocorr at lag {lag_ms}ms should be high (>0.8) for 10ms periodic spikes"
            )

            # Scaled pandas should match expected count
            # (since autocorr ≈ 1.0 for perfect periodicity)
            err_msg = (
                f"Scaled pandas (autocorr × N_pairs) should equal expected count at lag {lag_ms}ms"
            )
            np.testing.assert_almost_equal(
                scaled_pandas, num_valid_pairs, decimal=1, err_msg=err_msg
            )

            # Test that ACG counts match expected values
            # NOTE: The ACG implementation is buggy and FAILS this test
            np.testing.assert_array_equal(
                [acg_count_pos],
                [num_valid_pairs],
                err_msg=f"ACG positive lag count should equal expected at lag {lag_ms}ms",
            )
            np.testing.assert_array_equal(
                [acg_count_neg],
                [num_valid_pairs],
                err_msg=f"ACG negative lag count should equal expected at lag {lag_ms}ms",
            )


class TestBinnedCorrelogram:
    """Test ACG/CCG computation from binned counts and rates."""

    def test_binned_acg_matches_spike_times(self):
        """Binned ACG at 1ms resolution should match spike-times ACG."""
        # Create spikes every 10ms, centered in 1ms bins to avoid float bin-edge issues
        # (e.g., int(0.07/0.001) = 69 due to float precision, but 0.0705/0.001 = 70)
        spike_times_list = [np.arange(100, dtype=np.float64) * 0.01 + 0.0005]
        duration = 1.0

        # Spike-times path
        st_data = EvalData.from_spike_times_list(spike_times_list, duration=duration)
        cache_st = IntermediatesCache(st_data, st_data)

        acg_params = {"max_lag_ms": 50.0, "bin_size_ms": 1.0, "smoothing_kernel": 1}
        acg_st = cache_st.get(IntermediateRequest.from_dict("acg_original", acg_params))[0]

        # Binned path: create 1ms bins from the same spike times
        bin_size_s = 0.001
        n_bins = int(np.ceil(duration / bin_size_s))
        binned = np.zeros((1, n_bins), dtype=np.float32)
        for st in spike_times_list[0]:
            idx = min(int(st / bin_size_s), n_bins - 1)
            binned[0, idx] = 1

        binned_data = EvalData(
            binned_spikes=binned,
            bin_size_seconds=bin_size_s,
            duration=duration,
            n_units=1,
        )
        cache_binned = IntermediatesCache(binned_data, binned_data)
        acg_binned = cache_binned.get(IntermediateRequest.from_dict("acg_original", acg_params))[0]

        # Both should have the same shape
        assert len(acg_st) == len(acg_binned), (
            f"Shape mismatch: spike-times ACG has {len(acg_st)} bins, "
            f"binned ACG has {len(acg_binned)} bins"
        )

        # Values should match (binary spikes at 1ms → identical to spike-times)
        np.testing.assert_array_almost_equal(
            acg_binned,
            acg_st,
            decimal=5,
            err_msg="Binned ACG at 1ms should match spike-times ACG",
        )

    def test_binned_acg_coarse_bins(self):
        """ACG from 5ms-binned SPIKES (LDNS-like scenario)."""
        # Create 2 units with spikes at 5ms resolution over 1 second
        bin_size_s = 0.005  # 5ms
        n_time_bins = 200  # 1 second
        n_units = 2

        binned = np.zeros((n_units, n_time_bins), dtype=np.float32)
        # Unit 0: spike every 20ms (every 4th bin)
        binned[0, ::4] = 1
        # Unit 1: spike every 50ms (every 10th bin)
        binned[1, ::10] = 1

        data = EvalData(
            binned_spikes=binned,
            bin_size_seconds=bin_size_s,
            duration=n_time_bins * bin_size_s,
            n_units=n_units,
        )
        cache = IntermediatesCache(data, data)

        acg_params = {"max_lag_ms": 100.0, "bin_size_ms": 5.0, "smoothing_kernel": 1}
        acgs = cache.get(IntermediateRequest.from_dict("acg_original", acg_params))

        assert len(acgs) == 2
        # n_corr_bins = 2 * int(100/5) + 1 = 41
        assert len(acgs[0]) == 41, f"Expected 41 bins, got {len(acgs[0])}"
        assert len(acgs[1]) == 41

        # Center bin (lag 0) should be 0 for ACGs
        center = 20
        assert acgs[0][center] == 0, "ACG center bin should be 0"
        assert acgs[1][center] == 0, "ACG center bin should be 0"

        # Unit 0 has spikes every 4 bins (20ms). At lag ±20ms (±4 bins in 5ms res),
        # we expect peaks. Lag 20ms → index center ± 4
        assert acgs[0][center + 4] > 0, "Should have peak at +20ms lag"
        assert acgs[0][center - 4] > 0, "Should have peak at -20ms lag"

        # Symmetry
        np.testing.assert_array_almost_equal(
            acgs[0][:center],
            acgs[0][center + 1 :][::-1],
            err_msg="ACG should be symmetric",
        )

    def test_binned_acg_from_rates(self):
        """ACG from RATES data (float Hz values)."""
        bin_size_s = 0.005  # 5ms
        n_time_bins = 200
        n_units = 1

        # Constant rate of 50 Hz = 0.25 expected spikes per 5ms bin
        rates = np.full((n_units, n_time_bins), 50.0, dtype=np.float32)

        data = EvalData(
            rates_hz=rates,
            bin_size_seconds=bin_size_s,
            duration=n_time_bins * bin_size_s,
            n_units=n_units,
        )
        cache = IntermediatesCache(data, data)

        acg_params = {"max_lag_ms": 50.0, "bin_size_ms": 5.0, "smoothing_kernel": 1}
        acgs = cache.get(IntermediateRequest.from_dict("acg_original", acg_params))

        assert len(acgs) == 1
        # n_corr_bins = 2 * int(50/5) + 1 = 21
        assert len(acgs[0]) == 21

        # Center should be 0
        center = 10
        assert acgs[0][center] == 0

        # Constant rate → all non-zero lag bins should have similar values
        nonzero = acgs[0][acgs[0] > 0]
        assert len(nonzero) > 0, "Should have non-zero ACG values for constant rate"

    def test_binned_acg_auto_adjust_bin_size(self):
        """Requesting bin_size_ms < data bin_size should warn and auto-adjust."""
        import ntt.evaluation.core.intermediates as _mod

        # Reset the warn-once flag so this test reliably sees the warning
        _mod._warned_correlogram_resolution = False

        bin_size_s = 0.005  # 5ms data
        n_time_bins = 100
        binned = np.zeros((1, n_time_bins), dtype=np.float32)
        binned[0, ::5] = 1

        data = EvalData(
            binned_spikes=binned,
            bin_size_seconds=bin_size_s,
            duration=n_time_bins * bin_size_s,
            n_units=1,
        )
        cache = IntermediatesCache(data, data)

        # Request 1ms bins on 5ms data → should warn and use 5ms
        acg_params = {"max_lag_ms": 50.0, "bin_size_ms": 1.0, "smoothing_kernel": 1}
        with pytest.warns(RuntimeWarning, match="finer than data bin size"):
            acgs = cache.get(IntermediateRequest.from_dict("acg_original", acg_params))

        # Should have bins for 5ms resolution: 2 * int(50/5) + 1 = 21
        assert len(acgs[0]) == 21

    def test_binned_acg_all_zeros(self):
        """ACG of all-zero data should be all zeros."""
        binned = np.zeros((2, 100), dtype=np.float32)
        data = EvalData(
            binned_spikes=binned,
            bin_size_seconds=0.001,
            duration=0.1,
            n_units=2,
        )
        cache = IntermediatesCache(data, data)

        acg_params = {"max_lag_ms": 10.0, "bin_size_ms": 1.0, "smoothing_kernel": 1}
        acgs = cache.get(IntermediateRequest.from_dict("acg_original", acg_params))

        assert len(acgs) == 2
        assert np.all(acgs[0] == 0), "ACG of zero data should be all zeros"
        assert np.all(acgs[1] == 0), "ACG of zero data should be all zeros"

    def test_binned_zero_spike_full_correlograms_keep_center_bin(self):
        """All-zero fine-bin SPIKES should still return the symmetric +center-bin geometry."""
        binned = np.zeros((2, 1000), dtype=np.float32)
        data = EvalData(
            binned_spikes=binned,
            bin_size_seconds=0.001,
            duration=1.0,
            n_units=2,
        )
        cache = IntermediatesCache(data, data)

        corr_params = {"max_lag_ms": 100.0, "bin_size_ms": 1.0}
        correlograms = cache.get(
            IntermediateRequest.from_dict("all_correlograms_original", corr_params)
        )
        acgs = cache.get(
            IntermediateRequest.from_dict(
                "acg_original",
                {"max_lag_ms": 100.0, "bin_size_ms": 1.0, "smoothing_kernel": 1},
            )
        )
        ccgs = cache.get(
            IntermediateRequest.from_dict(
                "ccg_original",
                {"max_lag_ms": 100.0, "bin_size_ms": 1.0, "max_pairs": 1},
            )
        )

        assert correlograms.shape == (2, 2, 201)
        assert len(acgs) == 2
        assert all(len(acg) == 201 for acg in acgs)
        assert ccgs
        assert all(len(ccg) == 201 for ccg in ccgs.values())
        assert np.all(correlograms == 0)

    def test_binned_ccg(self):
        """CCG between two units with known cross-correlation pattern."""
        bin_size_s = 0.001  # 1ms
        n_time_bins = 1000
        n_units = 2

        binned = np.zeros((n_units, n_time_bins), dtype=np.float32)
        # Unit 0: spikes every 50ms
        binned[0, ::50] = 1
        # Unit 1: spikes every 50ms, shifted by 10ms
        binned[1, 10::50] = 1

        data = EvalData(
            binned_spikes=binned,
            bin_size_seconds=bin_size_s,
            duration=n_time_bins * bin_size_s,
            n_units=n_units,
        )
        cache = IntermediatesCache(data, data)

        # Request full correlograms
        corr_params = {"max_lag_ms": 50.0, "bin_size_ms": 1.0}
        correlograms = cache.get(
            IntermediateRequest.from_dict("all_correlograms_original", corr_params)
        )

        assert correlograms.shape[0] == 2
        assert correlograms.shape[1] == 2
        # n_bins = 2 * 50 + 1 = 101
        assert correlograms.shape[2] == 101

        center = 50  # zero-lag bin

        # Convention: correlograms[i, j, center + k] = unit i fires k bins after unit j
        # Unit 1 fires 10ms after unit 0 → correlograms[1, 0, center + 10] > 0
        assert correlograms[1, 0, center + 10] > 0, (
            "Unit 1 fires 10ms after unit 0, so CCG[1,0] should peak at +10"
        )
        # Equivalently, correlograms[0, 1, center - 10] > 0
        assert correlograms[0, 1, center - 10] > 0, (
            "CCG[0,1] at lag -10 = unit 0 fires 10ms before unit 1"
        )

        # CCG(0,1) and CCG(1,0) should be time-reversed
        np.testing.assert_array_almost_equal(
            correlograms[0, 1, :],
            correlograms[1, 0, ::-1],
            err_msg="CCG(0,1) should be time-reverse of CCG(1,0)",
        )


class TestDirectACGMatchesFullMatrix:
    """Verify that _compute_acgs_direct produces the same results as the
    full-matrix path (extract diagonal from all_correlograms)."""

    def _make_binned_data(self, n_units=5, n_time_bins=500, bin_size_s=0.005):
        rng = np.random.default_rng(123)
        binned = (rng.random((n_units, n_time_bins)) < 0.05).astype(np.float32)
        return EvalData(
            binned_spikes=binned,
            bin_size_seconds=bin_size_s,
            duration=n_time_bins * bin_size_s,
            n_units=n_units,
        )

    def _make_spike_times_data(self, n_units=5):
        rng = np.random.default_rng(456)
        spike_times = [
            np.sort(rng.uniform(0, 2.0, size=rng.integers(20, 100))) for _ in range(n_units)
        ]
        return EvalData.from_spike_times_list(spike_times, duration=2.0)

    def test_binned_acg_direct_matches_full(self):
        """Direct ACG from binned data matches extracting diagonal of full matrix."""
        data = self._make_binned_data()
        cache = IntermediatesCache(data, data)
        params = {"max_lag_ms": 50.0, "bin_size_ms": 5.0, "smoothing_kernel": 1}

        # Direct path (new)
        acgs_direct = cache.get(IntermediateRequest.from_dict("acg_original", params))

        # Full-matrix path
        corr_params = {"max_lag_ms": 50.0, "bin_size_ms": 5.0}
        all_corr = cache.get(
            IntermediateRequest.from_dict("all_correlograms_original", corr_params)
        )
        acgs_full = cache._extract_acgs(all_corr, smoothing_kernel=1)

        assert len(acgs_direct) == len(acgs_full)
        for i in range(len(acgs_direct)):
            np.testing.assert_array_almost_equal(
                acgs_direct[i],
                acgs_full[i],
                err_msg=f"ACG mismatch for unit {i} (binned path)",
            )

    def test_spike_times_acg_direct_matches_full(self):
        """Direct ACG from spike times matches extracting diagonal of full matrix."""
        data = self._make_spike_times_data()
        cache = IntermediatesCache(data, data)
        params = {"max_lag_ms": 50.0, "bin_size_ms": 1.0, "smoothing_kernel": 1}

        acgs_direct = cache.get(IntermediateRequest.from_dict("acg_original", params))

        corr_params = {"max_lag_ms": 50.0, "bin_size_ms": 1.0}
        all_corr = cache.get(
            IntermediateRequest.from_dict("all_correlograms_original", corr_params)
        )
        acgs_full = cache._extract_acgs(all_corr, smoothing_kernel=1)

        assert len(acgs_direct) == len(acgs_full)
        for i in range(len(acgs_direct)):
            np.testing.assert_array_almost_equal(
                acgs_direct[i],
                acgs_full[i],
                err_msg=f"ACG mismatch for unit {i} (spike-times path)",
            )

    def test_rates_acg_direct_matches_full(self):
        """Direct ACG from rates data matches extracting diagonal of full matrix."""
        rng = np.random.default_rng(789)
        n_units, n_time_bins, bin_size_s = 4, 400, 0.005
        rates = rng.uniform(10, 100, (n_units, n_time_bins)).astype(np.float32)
        data = EvalData(
            rates_hz=rates,
            bin_size_seconds=bin_size_s,
            duration=n_time_bins * bin_size_s,
            n_units=n_units,
        )
        cache = IntermediatesCache(data, data)
        params = {"max_lag_ms": 50.0, "bin_size_ms": 5.0, "smoothing_kernel": 3}

        acgs_direct = cache.get(IntermediateRequest.from_dict("acg_original", params))

        corr_params = {"max_lag_ms": 50.0, "bin_size_ms": 5.0}
        all_corr = cache.get(
            IntermediateRequest.from_dict("all_correlograms_original", corr_params)
        )
        acgs_full = cache._extract_acgs(all_corr, smoothing_kernel=3)

        for i in range(len(acgs_direct)):
            np.testing.assert_array_almost_equal(
                acgs_direct[i],
                acgs_full[i],
                err_msg=f"ACG mismatch for unit {i} (rates path, smoothed)",
            )


class TestDirectCCGMatchesFullMatrix:
    """Verify that _compute_ccgs_direct produces the same results as the
    full-matrix path (extract pairs from all_correlograms)."""

    def _make_binned_data(self, n_units=6, n_time_bins=500, bin_size_s=0.005):
        rng = np.random.default_rng(321)
        binned = (rng.random((n_units, n_time_bins)) < 0.05).astype(np.float32)
        return EvalData(
            binned_spikes=binned,
            bin_size_seconds=bin_size_s,
            duration=n_time_bins * bin_size_s,
            n_units=n_units,
        )

    def _make_spike_times_data(self, n_units=6):
        rng = np.random.default_rng(654)
        spike_times = [
            np.sort(rng.uniform(0, 2.0, size=rng.integers(20, 100))) for _ in range(n_units)
        ]
        return EvalData.from_spike_times_list(spike_times, duration=2.0)

    def test_binned_ccg_direct_matches_full(self):
        """Direct CCG from binned data matches extracting pairs from full matrix."""
        data = self._make_binned_data()
        cache_direct = IntermediatesCache(data, data)
        cache_full = IntermediatesCache(data, data)

        max_pairs = 5
        ccg_params = {"max_lag_ms": 50.0, "bin_size_ms": 5.0, "max_pairs": max_pairs}

        # Direct path
        ccgs_direct = cache_direct.get(IntermediateRequest.from_dict("ccg_original", ccg_params))

        # Full-matrix path
        corr_params = {"max_lag_ms": 50.0, "bin_size_ms": 5.0}
        all_corr = cache_full.get(
            IntermediateRequest.from_dict("all_correlograms_original", corr_params)
        )
        ccgs_full = cache_full._extract_ccgs(all_corr, max_pairs=max_pairs)

        assert set(ccgs_direct.keys()) == set(ccgs_full.keys()), (
            f"Pair mismatch: direct={sorted(ccgs_direct.keys())}, full={sorted(ccgs_full.keys())}"
        )
        for pair in ccgs_direct:
            np.testing.assert_array_almost_equal(
                ccgs_direct[pair],
                ccgs_full[pair],
                err_msg=f"CCG mismatch for pair {pair} (binned path)",
            )

    def test_spike_times_ccg_direct_matches_full(self):
        """Direct CCG from spike times matches extracting pairs from full matrix."""
        data = self._make_spike_times_data()
        cache_direct = IntermediatesCache(data, data)
        cache_full = IntermediatesCache(data, data)

        max_pairs = 5
        ccg_params = {"max_lag_ms": 50.0, "bin_size_ms": 1.0, "max_pairs": max_pairs}

        ccgs_direct = cache_direct.get(IntermediateRequest.from_dict("ccg_original", ccg_params))

        corr_params = {"max_lag_ms": 50.0, "bin_size_ms": 1.0}
        all_corr = cache_full.get(
            IntermediateRequest.from_dict("all_correlograms_original", corr_params)
        )
        ccgs_full = cache_full._extract_ccgs(all_corr, max_pairs=max_pairs)

        assert set(ccgs_direct.keys()) == set(ccgs_full.keys())
        for pair in ccgs_direct:
            np.testing.assert_array_almost_equal(
                ccgs_direct[pair],
                ccgs_full[pair],
                err_msg=f"CCG mismatch for pair {pair} (spike-times path)",
            )

    def test_ccg_direct_fallback_all_pairs(self):
        """With max_pairs=None, direct CCG falls back to full matrix and returns all pairs."""
        data = self._make_binned_data(n_units=3)
        cache = IntermediatesCache(data, data)

        ccg_params = {"max_lag_ms": 50.0, "bin_size_ms": 5.0, "max_pairs": None}
        ccgs = cache.get(IntermediateRequest.from_dict("ccg_original", ccg_params))

        # 3 units → 3 pairs
        assert len(ccgs) == 3
        expected_pairs = {(0, 1), (0, 2), (1, 2)}
        assert set(ccgs.keys()) == expected_pairs


class TestBinnedCountsCoarseBins:
    """Test that binned counts skip infeasible bin sizes instead of crashing."""

    def test_skips_finer_than_data(self):
        """With 5ms data bins, requesting [1, 10, 50] should skip 1ms and return 10ms, 50ms."""
        import ntt.evaluation.core.intermediates as _mod

        # Reset the warn-once flag so this test reliably sees the warning
        _mod._warned_bin_size_skip = False

        n_units, n_bins = 3, 200  # 200 bins at 5ms = 1s of data
        bin_size_seconds = 0.005
        spikes = np.random.default_rng(0).poisson(0.1, (n_units, n_bins)).astype(np.float32)

        original = EvalData(
            binned_spikes=spikes,
            bin_size_seconds=bin_size_seconds,
            data_type="spikes",
        )
        reconstructed = EvalData(
            binned_spikes=spikes,
            bin_size_seconds=bin_size_seconds,
            data_type="spikes",
        )

        cache = IntermediatesCache(original, reconstructed)

        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            request = IntermediateRequest.from_dict(
                "binned_counts_original", {"bin_sizes_ms": (1.0, 10.0, 50.0)}
            )
            result = cache.get(request)

        # 1ms should have been skipped
        assert 1.0 not in result, "1ms bin should be skipped for 5ms data"
        assert 10.0 in result, "10ms bin should be present"
        assert 50.0 in result, "50ms bin should be present"
        assert result[10.0].shape == (n_units, 100)  # 200 * 5ms / 10ms = 100 bins
        assert result[50.0].shape == (n_units, 20)  # 200 * 5ms / 50ms = 20 bins

        # Should have emitted a warning
        runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
        assert len(runtime_warnings) == 1
        assert "1.0" in str(runtime_warnings[0].message)

    def test_all_infeasible_raises(self):
        """If ALL requested bin sizes are finer than data, raise ValueError."""
        n_units, n_bins = 2, 100
        bin_size_seconds = 0.010  # 10ms bins
        spikes = np.zeros((n_units, n_bins), dtype=np.float32)

        original = EvalData(
            binned_spikes=spikes,
            bin_size_seconds=bin_size_seconds,
            data_type="spikes",
        )
        reconstructed = EvalData(
            binned_spikes=spikes,
            bin_size_seconds=bin_size_seconds,
            data_type="spikes",
        )

        cache = IntermediatesCache(original, reconstructed)

        with pytest.raises(ValueError, match="No requested bin sizes"):
            request = IntermediateRequest.from_dict(
                "binned_counts_original", {"bin_sizes_ms": (1.0, 5.0)}
            )
            cache.get(request)

    def test_spike_times_path_unaffected(self):
        """Spike times path can always bin at any resolution — no filtering needed."""
        spike_times = [np.array([0.1, 0.2, 0.5], dtype=np.float64)]

        original = EvalData.from_spike_times_list(spike_times, duration=1.0)
        reconstructed = EvalData.from_spike_times_list(spike_times, duration=1.0)

        cache = IntermediatesCache(original, reconstructed)

        request = IntermediateRequest.from_dict(
            "binned_counts_original", {"bin_sizes_ms": (1.0, 10.0, 50.0)}
        )
        result = cache.get(request)

        # All bin sizes should be present (spike times can be binned at any resolution)
        assert 1.0 in result
        assert 10.0 in result
        assert 50.0 in result


class TestExpandResults:
    """Test that SpikeCountCorrelation.expand_results promotes per-bin values."""

    def test_expand_produces_per_bin_entries(self):
        """expand_results should create one entry per non-primary bin size."""
        from ntt.evaluation.metrics.spike_detection import SpikeCountCorrelation

        metric = SpikeCountCorrelation(bin_sizes_ms=[1.0, 10.0, 50.0])

        # Create spike times with known correlation structure
        rng = np.random.default_rng(42)
        n_units = 3
        spike_times_orig = [np.sort(rng.uniform(0, 1, 50)) for _ in range(n_units)]
        # Reconstructed = original + small jitter → high correlation
        spike_times_recon = [st + rng.normal(0, 0.001, len(st)) for st in spike_times_orig]

        original = EvalData.from_spike_times_list(spike_times_orig, duration=1.0)
        reconstructed = EvalData.from_spike_times_list(spike_times_recon, duration=1.0)

        cache = IntermediatesCache(original, reconstructed)

        from ntt.evaluation.core.types import DataType

        requests = metric.get_intermediate_requests(DataType.SPIKE_TIMES)
        intermediates = {req.key: cache.get(req) for req in requests}
        result = metric.compute(intermediates)

        expanded = metric.expand_results(result, "spike_count_correlation")

        # Primary bin is 1ms → expanded should have 10ms and 50ms
        assert "spike_count_correlation_10ms" in expanded
        assert "spike_count_correlation_50ms" in expanded
        assert len(expanded) == 2

        # Each expanded result should have per-unit values
        for _name, extra in expanded.items():
            assert isinstance(extra.raw_value, np.ndarray)
            assert len(extra.raw_value) == n_units
            assert extra.metadata["bin_size_ms"] in (10.0, 50.0)

    def test_expand_empty_for_single_bin(self):
        """With only one bin size, expand_results should return nothing."""
        from ntt.evaluation.metrics.spike_detection import SpikeCountCorrelation

        metric = SpikeCountCorrelation(bin_sizes_ms=[50.0])
        result = metric._make_result(
            raw_value=np.array([0.9]),
            metadata={
                "correlations_by_bin_size": {50.0: np.array([0.9])},
                "primary_bin_size_ms": 50.0,
            },
        )
        expanded = metric.expand_results(result, "spike_count_correlation")
        assert expanded == {}
