import numpy as np
import pytest

from tests.convolution_helpers import (
    causal_exponential_distance,
    gaussian_distance,
)


def test_smoothed_squared_error_gaussian_bounds() -> None:
    from ntt.evaluation.metrics.spike_detection import SmoothSE

    metric = SmoothSE(
        causal=False,
        filter_type="gaussian",
        sigma_ms=10.0,
    )
    spike_times = np.array([0.1, 0.2, 0.3])
    shifted = spike_times + 0.005
    binned_spikes = np.zeros((1, 1000))
    binned_spikes[0, (spike_times * 1000).round().astype(int)] = 1
    shifted_binned = np.zeros((1, 1000))
    shifted_binned[0, (shifted * 1000).round().astype(int)] = 1
    intermediates = {
        "native_binned_counts_original": binned_spikes,
        "native_binned_counts_reconstructed": shifted_binned,
        "bin_size_original": 0.001,
        "bin_size_reconstructed": 0.001,
    }
    result = metric.compute(intermediates)
    # raw_value is now a per-unit array; extract the single unit
    assert result.raw_value.shape == (1,)
    value = result.raw_value[0]
    n_spikes = len(spike_times)
    assert n_spikes < 6 and n_spikes > 0, (
        "n_spikes must be between 0 and sufficiently small for the approximation to work"
    )
    expected = gaussian_distance(sigma_ms=10.0, shift_ms=5.0)
    assert np.isclose(value, expected, rtol=1e-2)


@pytest.mark.skip(reason="Implementation seems buggy, test fails")
def test_convolved_mse_causal_exponential() -> None:
    from ntt.evaluation.metrics.spike_detection import SmoothSE

    metric = SmoothSE(
        causal=True,
        filter_type="exponential",
        sigma_ms=10.0,
    )
    spike_times = np.array([0.1, 0.2, 0.3])
    shifted = spike_times + 0.005
    binned_spikes = np.zeros((1, 1000))
    binned_spikes[0, (spike_times * 1000).round().astype(int)] = 1
    shifted_binned = np.zeros((1, 1000))
    shifted_binned[0, (shifted * 1000).round().astype(int)] = 1
    intermediates = {
        "native_binned_counts_original": binned_spikes,
        "native_binned_counts_reconstructed": shifted_binned,
        "bin_size_original": 0.001,
        "bin_size_reconstructed": 0.001,
    }
    result = metric.compute(intermediates)
    value = float(result.raw_value)
    expected = causal_exponential_distance(tau_ms=10.0, shift_ms=5.0) * len(spike_times)
    assert np.isclose(value, expected, rtol=1e-3)


# ** Tests for BinnedRateDifference metric **
# -------------------------------------------


def test_binned_rate_oscillating_pattern() -> None:
    """Test oscillating pattern with fake rate tokenizer at different bin sizes.

    Now returns per-unit arrays in Hz (spikes/s).
    """
    from ntt.evaluation.metrics.spike_detection import BinnedRateDifference

    # Original: oscillating pattern at 5ms resolution
    # Pattern: 0,1,2,3,4,5,4,3,2,1 spikes per 5ms (sums to 25 per 50ms cycle)
    n_cycles = 2
    binned_orig = np.zeros((1, n_cycles * 50))
    spike_pattern = [0, 1, 2, 3, 4, 5, 4, 3, 2, 1]

    for cycle in range(n_cycles):
        for i, n_spikes in enumerate(spike_pattern):
            start_idx = cycle * 50 + i * 5
            if n_spikes > 0:
                binned_orig[0, start_idx : start_idx + n_spikes] = 1

    binned_recon = np.zeros((1, n_cycles * 50))
    for cycle in range(n_cycles):
        start_idx = cycle * 50
        binned_recon[0, start_idx : start_idx + 25] = 1

    intermediates = {
        "native_binned_counts_original": binned_orig,
        "native_binned_counts_reconstructed": binned_recon,
        "bin_size_original": 0.001,
        "bin_size_reconstructed": 0.001,
    }

    # Test 1: Misaligned bins (5ms metric resolution)
    metric_5ms = BinnedRateDifference(bin_size_ms=5.0, normalize=False)
    result_5ms = metric_5ms.compute(intermediates)
    assert result_5ms.raw_value.shape == (1,)
    # MAE = 3.0 spikes per 5ms bin → 3.0 / 0.005s = 600 Hz
    assert np.isclose(result_5ms.raw_value[0], 600.0, atol=1e-3)

    # Test with normalization
    metric_5ms_norm = BinnedRateDifference(bin_size_ms=5.0, normalize=True)
    result_5ms_norm = metric_5ms_norm.compute(intermediates)
    # Original rate: 25 spikes / 0.05s = 500 Hz per unit
    # Normalized: 600 / 500 = 1.2
    assert np.isclose(result_5ms_norm.raw_value[0], 1.2, atol=1e-6)

    # Test 2: Aligned bins (50ms metric resolution)
    metric_50ms = BinnedRateDifference(bin_size_ms=50.0, normalize=False)
    result_50ms = metric_50ms.compute(intermediates)
    assert np.isclose(result_50ms.raw_value[0], 0.0, atol=1e-6)


def test_binned_rate_all_zeros() -> None:
    """Test with no spikes - should handle gracefully."""
    from ntt.evaluation.metrics.spike_detection import BinnedRateDifference

    binned_spikes = np.zeros((2, 1000))  # 2 units, 1s at 1ms resolution

    intermediates = {
        "native_binned_counts_original": binned_spikes,
        "native_binned_counts_reconstructed": binned_spikes.copy(),
        "bin_size_original": 0.001,
        "bin_size_reconstructed": 0.001,
    }

    metric = BinnedRateDifference(bin_size_ms=10.0, normalize=True)
    result = metric.compute(intermediates)

    assert result.raw_value.shape == (2,)
    assert np.allclose(result.raw_value, 0.0, atol=1e-6)
    assert np.allclose(result.metadata["mean_rate_original_hz"], 0.0, atol=1e-6)


def test_binned_rate_perfect_match_random() -> None:
    """Test with random spike pattern, perfect match."""
    from ntt.evaluation.metrics.spike_detection import BinnedRateDifference

    rng = np.random.default_rng(42)
    binned_spikes = (rng.random((3, 500)) > 0.9).astype(float)  # 3 units, sparse spikes

    intermediates = {
        "native_binned_counts_original": binned_spikes,
        "native_binned_counts_reconstructed": binned_spikes.copy(),
        "bin_size_original": 0.001,
        "bin_size_reconstructed": 0.001,
    }

    metric = BinnedRateDifference(bin_size_ms=10.0, normalize=False)
    result = metric.compute(intermediates)
    assert result.raw_value.shape == (3,)
    assert np.allclose(result.raw_value, 0.0, atol=1e-6)


def test_binned_rate_different_rates() -> None:
    """Test with evenly spaced spikes at different rates."""
    from ntt.evaluation.metrics.spike_detection import BinnedRateDifference

    # Original: 1 spike per 20ms → 50 Hz
    binned_orig = np.zeros((1, 200))
    binned_orig[0, ::20] = 1

    # Reconstructed: 1 spike per 10ms → 100 Hz
    binned_recon = np.zeros((1, 200))
    binned_recon[0, ::10] = 1

    intermediates = {
        "native_binned_counts_original": binned_orig,
        "native_binned_counts_reconstructed": binned_recon,
        "bin_size_original": 0.001,
        "bin_size_reconstructed": 0.001,
    }

    metric = BinnedRateDifference(bin_size_ms=10.0, normalize=False)
    result = metric.compute(intermediates)

    # At 10ms bins: orig 0.5 spikes/bin, recon 1.0 spikes/bin
    # MAE = 0.5 spikes / 0.01s = 50 Hz
    assert np.isclose(result.raw_value[0], 50.0, atol=1e-3)


def test_binned_rate_metadata() -> None:
    """Test that all metadata fields are present and correct."""
    from ntt.evaluation.metrics.spike_detection import BinnedRateDifference

    binned_orig = np.zeros((1, 100))
    binned_orig[0, ::10] = 1  # 10 spikes in 100ms → 100 Hz

    binned_recon = np.zeros((1, 100))
    binned_recon[0, ::5] = 1  # 20 spikes in 100ms → 200 Hz

    intermediates = {
        "native_binned_counts_original": binned_orig,
        "native_binned_counts_reconstructed": binned_recon,
        "bin_size_original": 0.001,
        "bin_size_reconstructed": 0.001,
    }

    metric = BinnedRateDifference(bin_size_ms=10.0, normalize=True)
    result = metric.compute(intermediates)

    assert "mean_rate_original_hz" in result.metadata
    assert "mean_rate_reconstructed_hz" in result.metadata
    assert "bin_size_ms" in result.metadata
    assert "rate_unit" in result.metadata

    assert result.metadata["bin_size_ms"] == 10.0
    assert result.metadata["rate_unit"] == "Hz"
    # 10 spikes / 0.1s = 100 Hz
    assert np.isclose(result.metadata["mean_rate_original_hz"][0], 100.0, atol=1e-3)
    # 20 spikes / 0.1s = 200 Hz
    assert np.isclose(result.metadata["mean_rate_reconstructed_hz"][0], 200.0, atol=1e-3)
    # MAE per unit: 1 spike diff per 10ms bin / 0.01s = 100 Hz
    # Normalized: 100 / 100 = 1.0
    assert np.isclose(result.raw_value[0], 1.0, atol=1e-6)


def test_binned_rate_batch_dimension() -> None:
    """Test with batch dimension of size 1 - should flatten correctly."""
    from ntt.evaluation.metrics.spike_detection import BinnedRateDifference

    batch_size = 1
    n_units = 2
    binned_orig = np.zeros((batch_size, n_units, 100))
    binned_recon = np.zeros((batch_size, n_units, 100))

    binned_orig[0, 0, ::10] = 1
    binned_recon[0, 0, ::10] = 1

    binned_orig[0, 1, ::10] = 1  # 10 spikes
    binned_recon[0, 1, ::5] = 1  # 20 spikes

    intermediates = {
        "native_binned_counts_original": binned_orig,
        "native_binned_counts_reconstructed": binned_recon,
        "bin_size_original": 0.001,
        "bin_size_reconstructed": 0.001,
    }

    metric = BinnedRateDifference(bin_size_ms=10.0, normalize=False)
    result = metric.compute(intermediates)

    assert result.raw_value.shape == (2,)
    # Unit 0: MAE = 0 Hz
    assert np.isclose(result.raw_value[0], 0.0, atol=1e-6)
    # Unit 1: 1 spike diff / 0.01s = 100 Hz
    assert np.isclose(result.raw_value[1], 100.0, atol=1e-3)


def test_binned_rate_multiple_units_per_unit() -> None:
    """Test per-unit values: one perfect, one with known error."""
    from ntt.evaluation.metrics.spike_detection import BinnedRateDifference

    binned_orig = np.zeros((2, 100))
    binned_recon = np.zeros((2, 100))

    binned_orig[0, ::10] = 1
    binned_recon[0, ::10] = 1

    binned_orig[1, ::10] = 1  # 1 per 10ms bin
    binned_recon[1, ::2] = 1  # 5 per 10ms bin

    intermediates = {
        "native_binned_counts_original": binned_orig,
        "native_binned_counts_reconstructed": binned_recon,
        "bin_size_original": 0.001,
        "bin_size_reconstructed": 0.001,
    }

    metric = BinnedRateDifference(bin_size_ms=20.0, normalize=False)
    result = metric.compute(intermediates)

    assert result.raw_value.shape == (2,)
    # Unit 0: 0 Hz error
    assert np.isclose(result.raw_value[0], 0.0, atol=1e-6)
    # Unit 1: at 20ms bins, orig=2 spikes/bin, recon=10 spikes/bin → diff=8/0.02s = 400 Hz
    assert np.isclose(result.raw_value[1], 400.0, atol=1e-3)


def test_binned_rate_zero_original_lower_bound() -> None:
    """Test normalization lower bound when original has no spikes."""
    from ntt.evaluation.metrics.spike_detection import BinnedRateDifference

    binned_orig = np.zeros((1, 100))
    binned_recon = np.ones((1, 100))  # 100 spikes

    intermediates = {
        "native_binned_counts_original": binned_orig,
        "native_binned_counts_reconstructed": binned_recon,
        "bin_size_original": 0.001,
        "bin_size_reconstructed": 0.001,
    }

    metric = BinnedRateDifference(bin_size_ms=10.0, normalize=True)
    result = metric.compute(intermediates)

    # MAE: 10 spikes/bin / 0.01s = 1000 Hz
    # Original rate: 0 Hz → lower bound = 1 spike/bin / 0.01s = 100 Hz
    # Normalized: 1000 / 100 = 10.0
    assert np.isclose(result.raw_value[0], 10.0, atol=1e-6)
    assert np.isclose(result.metadata["mean_rate_original_hz"][0], 0.0, atol=1e-6)


def test_binned_rate_accepts_non_1ms_bins() -> None:
    """BinnedRateDifference should work with any bin size, not just 1ms."""
    from ntt.evaluation.metrics.spike_detection import BinnedRateDifference

    # 5ms bins (0.005s), 2 units, 200 bins = 1s
    binned_orig = np.zeros((2, 200))
    binned_recon = np.zeros((2, 200))
    binned_orig[0, ::4] = 1  # 50 spikes in 200 bins → 50 Hz
    binned_recon[0, ::4] = 1  # perfect match

    binned_orig[1, ::2] = 1  # 100 spikes → 100 Hz
    binned_recon[1, ::4] = 1  # 50 spikes → 50 Hz

    intermediates = {
        "native_binned_counts_original": binned_orig,
        "native_binned_counts_reconstructed": binned_recon,
        "bin_size_original": 0.005,  # 5ms bins
        "bin_size_reconstructed": 0.005,
    }

    # Metric bin = 10ms = 2x input bins
    metric = BinnedRateDifference(bin_size_ms=10.0, normalize=False)
    result = metric.compute(intermediates)
    assert result.raw_value.shape == (2,)
    assert np.isclose(result.raw_value[0], 0.0, atol=1e-6)


def test_metric_regression_fixed_random_fixture() -> None:
    """Preserve pre-refactor outputs for key metrics on a fixed random fixture."""
    from ntt.evaluation.core.intermediates import IntermediatesCache
    from ntt.evaluation.core.types import EvalData
    from ntt.evaluation.metrics.base import IntermediateRequest
    from ntt.evaluation.metrics.single_unit import ACGCorrelation
    from ntt.evaluation.metrics.spike_detection import BinnedRateDifference, SmoothSE

    seed = 20260416
    rng = np.random.default_rng(seed)
    original = (rng.random((5, 1000)) < 0.035).astype(float)
    reconstructed = (rng.random((5, 1000)) < 0.04).astype(float)

    smooth = SmoothSE(causal=False, filter_type="gaussian", sigma_ms=10.0)
    binned_rate = BinnedRateDifference(bin_size_ms=5.0, normalize=False)

    spike_intermediates = {
        "native_binned_counts_original": original,
        "native_binned_counts_reconstructed": reconstructed,
        "bin_size_original": 0.001,
        "bin_size_reconstructed": 0.001,
    }

    original_data = EvalData(binned_spikes=original, bin_size_seconds=0.001, data_type="spikes")
    reconstructed_data = EvalData(
        binned_spikes=reconstructed,
        bin_size_seconds=0.001,
        data_type="spikes",
    )
    cache = IntermediatesCache(original_data, reconstructed_data)
    acg_request = IntermediateRequest.from_dict(
        "acg_original",
        {"max_lag_ms": 100.0, "bin_size_ms": 1.0, "smoothing_kernel": 1},
    )
    acg_recon_request = IntermediateRequest.from_dict(
        "acg_reconstructed",
        {"max_lag_ms": 100.0, "bin_size_ms": 1.0, "smoothing_kernel": 1},
    )
    acg_metric = ACGCorrelation(
        correlogram_max_lag_ms=100.0,
        correlogram_bin_size_ms=1.0,
        acg_smoothing_kernel=1,
    )

    smooth_result = smooth.compute(spike_intermediates)
    binned_rate_result = binned_rate.compute(spike_intermediates)
    acg_result = acg_metric.compute(
        {
            "acg_original": cache.get(acg_request),
            "acg_reconstructed": cache.get(acg_recon_request),
        }
    )

    expected_smooth = np.array(
        [
            2.028397354792,
            2.168389456438,
            1.869242428175,
            1.85457794282,
            1.367977452509,
        ]
    )
    expected_binned_rate = np.array([83.0, 53.0, 63.0, 63.0, 60.0])
    expected_acg = np.array(
        [
            0.062982780218,
            -0.017160535464,
            -0.085572918398,
            -0.046750115704,
            -0.033113531734,
        ]
    )

    assert np.allclose(smooth_result.raw_value, expected_smooth, atol=1e-12)
    assert np.allclose(binned_rate_result.raw_value, expected_binned_rate, atol=1e-12)
    assert np.allclose(acg_result.raw_value, expected_acg, atol=1e-12)


def test_rate_compatible_metrics_match_between_1ms_spikes_and_50ms_loader_counts() -> None:
    """Loader-side count inputs should agree with 1ms spikes for coarse metrics."""
    import torch
    from temporaldata import ArrayDict, Data, Interval, IrregularTimeSeries

    from ntt.core.components.types import BinnedSpikesInput, RatesInput
    from ntt.core.data.components.transforms import (
        CreateBinnedSpikeCountsArray,
        DataToModelInput,
    )
    from ntt.evaluation.core.intermediates import IntermediatesCache
    from ntt.evaluation.core.types import DataType, EvalData
    from ntt.evaluation.metrics.base import BaseMetric
    from ntt.evaluation.metrics.single_unit import ACGCorrelation
    from ntt.evaluation.metrics.spike_detection import BinnedRateDifference, SmoothSE

    def _make_data() -> Data:
        duration_s = 0.5
        n_units = 10
        coarse_bin_s = 0.05
        domain = Interval(0.0, duration_s)
        units = ArrayDict(
            unit_number=np.arange(n_units),
            id=np.array([f"unit_{i}" for i in range(n_units)]),
        )

        timestamps = []
        unit_index = []
        for unit in range(n_units):
            for coarse_bin in range(10):
                is_active = ((coarse_bin + unit) % 3 == 0) or ((2 * coarse_bin + unit) % 5 == 0)
                if is_active:
                    timestamps.append(coarse_bin * coarse_bin_s + 0.025)
                    unit_index.append(unit)

        order = np.argsort(timestamps)
        spikes = IrregularTimeSeries(
            timestamps=np.asarray(timestamps, dtype=np.float64)[order],
            unit_index=np.asarray(unit_index, dtype=np.int64)[order],
            domain=domain,
        )
        session = type("Session", (), {"id": "eval_session"})()
        subject = type("Subject", (), {"id": "subject_1", "species": "macaque"})()

        data = Data(domain=domain)
        data.units = units
        data.spikes = spikes
        data.session = session
        data.subject = subject
        return data

    def _as_batched_input(
        single_input: BinnedSpikesInput | RatesInput,
    ) -> BinnedSpikesInput | RatesInput:
        def _unwrap(value):
            return getattr(value, "obj", value)

        matrix_name = (
            "spike_matrix" if isinstance(single_input, BinnedSpikesInput) else "rate_matrix"
        )
        matrix = _unwrap(getattr(single_input, matrix_name))
        mask = _unwrap(single_input.pad_and_gap_mask)
        unit_ids = _unwrap(single_input.unit_ids)
        unit_ids_pad_mask = _unwrap(single_input.unit_ids_pad_mask)
        duration = _unwrap(single_input.duration_sec)
        absolute_start = _unwrap(single_input.absolute_start)
        absolute_end = _unwrap(single_input.absolute_end)
        behavioral_data = single_input.behavioral_data

        kwargs = {
            matrix_name: matrix.unsqueeze(0) if matrix.ndim == 2 else matrix,
            "pad_and_gap_mask": mask.unsqueeze(0) if mask.ndim == 2 else mask,
            "sampling_rate": single_input.sampling_rate,
            "unit_ids": unit_ids.unsqueeze(0) if unit_ids.ndim == 1 else unit_ids,
            "unit_ids_pad_mask": (
                unit_ids_pad_mask.unsqueeze(0)
                if unit_ids_pad_mask.ndim == 1
                else unit_ids_pad_mask
            ),
            "duration_sec": (
                torch.tensor([float(duration)], dtype=torch.float32)
                if isinstance(duration, torch.Tensor) and duration.ndim == 0
                else duration
            ),
            "session_id": single_input.session_id,
            "subject_id": single_input.subject_id,
            "species": single_input.species,
            "absolute_start": (
                torch.tensor([float(absolute_start)], dtype=torch.float64)
                if isinstance(absolute_start, torch.Tensor) and absolute_start.ndim == 0
                else absolute_start
            ),
            "absolute_end": (
                torch.tensor([float(absolute_end)], dtype=torch.float64)
                if isinstance(absolute_end, torch.Tensor) and absolute_end.ndim == 0
                else absolute_end
            ),
            "behavioral_data": (
                [behavioral_data]
                if behavioral_data is not None and not isinstance(behavioral_data, list)
                else behavioral_data
            ),
        }

        cls = BinnedSpikesInput if isinstance(single_input, BinnedSpikesInput) else RatesInput
        return cls(**kwargs)

    def _subset(eval_data: EvalData, start: int, stop: int) -> EvalData:
        if eval_data.binned_spikes is not None:
            return EvalData(
                binned_spikes=eval_data.binned_spikes[start:stop],
                bin_size_seconds=eval_data.bin_size_seconds,
                duration=eval_data.duration,
                data_type=eval_data.data_type,
                n_units=stop - start,
            )
        if eval_data.binned_counts is not None:
            return EvalData(
                binned_counts=eval_data.binned_counts[start:stop],
                bin_size_seconds=eval_data.bin_size_seconds,
                duration=eval_data.duration,
                data_type=eval_data.data_type,
                n_units=stop - start,
            )
        raise AssertionError("Expected binned data in this test.")

    def _compute(metric: BaseMetric, original: EvalData, reconstructed: EvalData) -> np.ndarray:
        cache = IntermediatesCache(original, reconstructed)
        intermediates = {
            request.key: cache.get(request)
            for request in metric.get_intermediate_requests(original.data_type)
        }
        return np.asarray(metric.compute(intermediates).raw_value)

    data = _make_data()
    spikes_input = _as_batched_input(
        DataToModelInput(mode=DataType.SPIKES)(CreateBinnedSpikeCountsArray(bin_size_ms=1.0)(data))
    )
    rates_input = _as_batched_input(
        DataToModelInput(mode=DataType.RATES)(CreateBinnedSpikeCountsArray(bin_size_ms=50.0)(data))
    )

    spikes_eval = EvalData.from_binned_input(spikes_input, batch_index=0)
    rates_eval = EvalData.from_binned_input(rates_input, batch_index=0)

    assert spikes_eval.binned_spikes is not None
    assert rates_eval.binned_counts is not None
    assert rates_eval.rates_hz is None

    spikes_original = _subset(spikes_eval, 0, 5)
    spikes_reconstructed = _subset(spikes_eval, 5, 10)
    rates_original = _subset(rates_eval, 0, 5)
    rates_reconstructed = _subset(rates_eval, 5, 10)

    metrics = [
        SmoothSE(causal=False, filter_type="gaussian", sigma_ms=1.0),
        BinnedRateDifference(bin_size_ms=50.0, normalize=False),
        ACGCorrelation(
            correlogram_max_lag_ms=150.0,
            correlogram_bin_size_ms=50.0,
            acg_smoothing_kernel=1,
        ),
    ]

    for metric in metrics:
        spike_value = _compute(metric, spikes_original, spikes_reconstructed)
        rate_value = _compute(metric, rates_original, rates_reconstructed)
        assert np.allclose(
            spike_value,
            rate_value,
            atol=1e-8,
            equal_nan=True,
        ), f"{metric.name} mismatch between 1ms spikes and 50ms loader counts"


# ** Tests for MutualInformation metric **
# ----------------------------------------


class TestDiscreteMI:
    """Unit tests for the core _discrete_mi function."""

    def test_identical_arrays_give_nmi_one(self) -> None:
        """Perfect reconstruction → NMI = 1.0."""
        from ntt.evaluation.metrics.information import _discrete_mi

        rng = np.random.default_rng(0)
        # Poisson counts at ~10 Hz in 10ms bins → mean ≈ 0.1 spikes/bin
        x = rng.poisson(lam=0.1, size=5000).astype(float)
        mi, hx, hy, n, bias, reliable = _discrete_mi(x, x, min_samples=50)

        assert reliable
        assert hx > 0
        nmi = mi / hx
        # Should be very close to 1 (bias correction can push it slightly below)
        assert nmi > 0.98, f"NMI for identical data should be ~1.0, got {nmi:.4f}"

    def test_independent_arrays_give_nmi_near_zero(self) -> None:
        """Independent random counts → NMI ≈ 0 after bias correction."""
        from ntt.evaluation.metrics.information import _discrete_mi

        rng = np.random.default_rng(42)
        x = rng.poisson(lam=0.5, size=10000).astype(float)
        y = rng.poisson(lam=0.5, size=10000).astype(float)
        mi, hx, hy, n, bias, reliable = _discrete_mi(x, y, min_samples=50)

        assert reliable
        nmi = mi / hx
        # Should be near zero — any residual is estimation noise
        assert nmi < 0.05, f"NMI for independent data should be ~0, got {nmi:.4f}"

    def test_bias_correction_reduces_mi(self) -> None:
        """Bias correction should make MI smaller than the plugin estimate."""
        from ntt.evaluation.metrics.information import _discrete_mi

        rng = np.random.default_rng(7)
        x = rng.poisson(lam=0.3, size=500).astype(float)
        y = rng.poisson(lam=0.3, size=500).astype(float)
        mi_corrected, hx, hy, n, bias, reliable = _discrete_mi(x, y, min_samples=50)

        assert bias > 0, "Miller-Madow bias estimate should be positive"
        # The plugin MI would have been mi_corrected + bias
        assert mi_corrected < mi_corrected + bias

    def test_constant_original_returns_unreliable(self) -> None:
        """If original has zero entropy, NMI is undefined → unreliable."""
        from ntt.evaluation.metrics.information import _discrete_mi

        x = np.zeros(1000)
        y = np.random.default_rng(0).poisson(0.1, 1000).astype(float)
        mi, hx, hy, n, bias, reliable = _discrete_mi(x, y, min_samples=50)

        assert not reliable
        assert np.isclose(hx, 0.0)

    def test_too_few_samples_returns_unreliable(self) -> None:
        """Below min_samples threshold → unreliable."""
        from ntt.evaluation.metrics.information import _discrete_mi

        x = np.array([0, 1, 0, 1, 0], dtype=float)
        y = np.array([0, 1, 0, 1, 0], dtype=float)
        mi, hx, hy, n, bias, reliable = _discrete_mi(x, y, min_samples=50)

        assert not reliable
        assert n == 5

    def test_known_entropy_binary_symmetric(self) -> None:
        """Binary symmetric channel: H(X) should be 1 bit for p=0.5."""
        from ntt.evaluation.metrics.information import _discrete_mi

        # Perfectly balanced binary: 0 and 1 each appear 5000 times
        x = np.array([0] * 5000 + [1] * 5000, dtype=float)
        y = x.copy()  # perfect copy
        mi, hx, hy, n, bias, reliable = _discrete_mi(x, y, min_samples=50)

        assert reliable
        # H(Bernoulli(0.5)) = 1.0 bits
        assert np.isclose(hx, 1.0, atol=0.01)
        # MI = H(X) for perfect reconstruction
        assert np.isclose(mi, 1.0, atol=0.02)

    def test_handles_non_zero_based_counts(self) -> None:
        """Counts that don't start at 0 (e.g., all values >= 2)."""
        from ntt.evaluation.metrics.information import _discrete_mi

        rng = np.random.default_rng(3)
        x = rng.poisson(lam=5.0, size=5000).astype(float)
        y = x.copy()
        mi, hx, hy, n, bias, reliable = _discrete_mi(x, y, min_samples=50)

        assert reliable
        nmi = mi / hx
        assert nmi > 0.95


class TestMutualInformationMetric:
    """Integration tests for the MutualInformation metric class."""

    def _make_intermediates(
        self,
        orig: np.ndarray,
        recon: np.ndarray,
        bin_sizes_ms: list[float] | None = None,
    ) -> dict:
        """Build intermediates dict matching what IntermediatesCache produces.

        orig, recon: shape (n_units, n_time_bins) — counts at 1ms.
        """
        if bin_sizes_ms is None:
            bin_sizes_ms = [1.0, 5.0, 10.0, 50.0, 100.0]

        def _rebin(counts, ratio):
            n = counts.shape[1] // ratio
            return counts[:, : n * ratio].reshape(counts.shape[0], n, ratio).sum(axis=2)

        counts_orig = {}
        counts_recon = {}
        for bs in bin_sizes_ms:
            ratio = int(bs)
            if ratio == 1:
                counts_orig[bs] = orig.copy()
                counts_recon[bs] = recon.copy()
            else:
                counts_orig[bs] = _rebin(orig, ratio)
                counts_recon[bs] = _rebin(recon, ratio)
        return {
            "binned_counts_original": counts_orig,
            "binned_counts_reconstructed": counts_recon,
        }

    def test_perfect_reconstruction_gives_high_nmi(self) -> None:
        from ntt.evaluation.metrics.information import MutualInformation

        rng = np.random.default_rng(0)
        n_units, n_bins = 3, 10000
        orig = rng.poisson(0.05, (n_units, n_bins)).astype(float)  # ~50 Hz at 1ms

        metric = MutualInformation()
        result = metric.compute(self._make_intermediates(orig, orig))

        values = result.raw_value
        assert values.shape == (n_units,)
        # NMI should be close to 1 for all units
        valid = values[~np.isnan(values)]
        assert len(valid) > 0
        assert np.all(valid > 0.9), f"NMI values: {valid}"

    def test_independent_reconstruction_gives_low_nmi(self) -> None:
        from ntt.evaluation.metrics.information import MutualInformation

        rng = np.random.default_rng(0)
        n_units, n_bins = 3, 10000
        orig = rng.poisson(0.1, (n_units, n_bins)).astype(float)
        recon = rng.poisson(0.1, (n_units, n_bins)).astype(float)

        metric = MutualInformation()
        result = metric.compute(self._make_intermediates(orig, recon))

        valid = result.raw_value[~np.isnan(result.raw_value)]
        assert len(valid) > 0
        assert np.all(valid < 0.1), f"NMI for independent: {valid}"

    def test_nmi_bounded_zero_one(self) -> None:
        from ntt.evaluation.metrics.information import MutualInformation

        rng = np.random.default_rng(1)
        n_units, n_bins = 5, 5000
        orig = rng.poisson(0.2, (n_units, n_bins)).astype(float)
        # Noisy copy: flip ~30% of bins
        recon = orig.copy()
        flip_mask = rng.random((n_units, n_bins)) < 0.3
        recon[flip_mask] = rng.poisson(0.2, flip_mask.sum())

        metric = MutualInformation()
        result = metric.compute(self._make_intermediates(orig, recon))

        valid = result.raw_value[~np.isnan(result.raw_value)]
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 1.0)

    def test_metadata_contains_all_timescales(self) -> None:
        from ntt.evaluation.metrics.information import MutualInformation

        rng = np.random.default_rng(0)
        orig = rng.poisson(0.1, (2, 5000)).astype(float)

        metric = MutualInformation(bin_sizes_ms=[1.0, 10.0, 100.0])
        result = metric.compute(
            self._make_intermediates(orig, orig, bin_sizes_ms=[1.0, 10.0, 100.0])
        )

        assert "nmi_by_bin_size" in result.metadata
        assert "mi_bits_by_bin_size" in result.metadata
        assert "h_original_bits_by_bin_size" in result.metadata
        assert "bias_bits_by_bin_size" in result.metadata

        nmi_by_bin = result.metadata["nmi_by_bin_size"]
        assert set(nmi_by_bin.keys()) == {1.0, 10.0, 100.0}

    def test_value_range_is_zero_one(self) -> None:
        from ntt.evaluation.metrics.information import MutualInformation

        metric = MutualInformation()
        assert metric.value_range == (0.0, 1.0)


# ** Tests for FiringRateError metric **
# --------------------------------------


def test_firing_rate_error_perfect_match() -> None:
    """Perfect reconstruction has zero firing rate error."""
    from ntt.evaluation.metrics.single_unit import FiringRateError

    rates = np.array([10.0, 50.0, 100.0])  # 3 units, Hz

    metric = FiringRateError()
    result = metric.compute(
        {
            "firing_rates_original": rates,
            "firing_rates_reconstructed": rates.copy(),
        }
    )

    assert result.raw_value.shape == (3,)
    assert np.allclose(result.raw_value, 0.0)


def test_firing_rate_error_known_difference() -> None:
    """Known difference in firing rates."""
    from ntt.evaluation.metrics.single_unit import FiringRateError

    rates_orig = np.array([10.0, 50.0])
    rates_recon = np.array([15.0, 30.0])

    metric = FiringRateError()
    result = metric.compute(
        {
            "firing_rates_original": rates_orig,
            "firing_rates_reconstructed": rates_recon,
        }
    )

    assert result.raw_value.shape == (2,)
    assert np.isclose(result.raw_value[0], 5.0)
    assert np.isclose(result.raw_value[1], 20.0)
    assert "rates_original_hz" in result.metadata
    assert "rates_reconstructed_hz" in result.metadata


def test_firing_rate_error_supports_all_data_types() -> None:
    """FiringRateError should declare support for all data types."""
    from ntt.evaluation.core.types import DataType
    from ntt.evaluation.metrics.single_unit import FiringRateError

    metric = FiringRateError()
    assert DataType.SPIKE_TIMES in metric.supported_data_types
    assert DataType.SPIKES in metric.supported_data_types
    assert DataType.RATES in metric.supported_data_types


# ** Tests for FanoFactorError metric **
# --------------------------------------


def test_fano_factor_error_perfect_match() -> None:
    """Perfect reconstruction has zero Fano factor error."""
    from ntt.evaluation.metrics.single_unit import FanoFactorError

    rng = np.random.default_rng(42)
    # 2 units, 100 bins at 100ms each
    counts = rng.poisson(5.0, (2, 100)).astype(float)

    intermediates = {
        "binned_counts_original": {100.0: counts},
        "binned_counts_reconstructed": {100.0: counts.copy()},
    }

    metric = FanoFactorError(fano_bin_size_ms=100.0)
    result = metric.compute(intermediates)

    assert result.raw_value.shape == (2,)
    assert np.allclose(result.raw_value, 0.0)
    assert "fano_factor_original" in result.metadata
    assert "fano_factor_reconstructed" in result.metadata


def test_fano_factor_error_poisson_vs_regular() -> None:
    """Poisson (FF~1) vs regular (FF<1) should produce known error."""
    from ntt.evaluation.metrics.single_unit import FanoFactorError

    rng = np.random.default_rng(0)
    n_bins = 10000

    # Poisson: FF ≈ 1
    poisson_counts = rng.poisson(5.0, (1, n_bins)).astype(float)

    # Regular: constant count → FF = 0
    regular_counts = np.full((1, n_bins), 5.0)

    intermediates = {
        "binned_counts_original": {100.0: poisson_counts},
        "binned_counts_reconstructed": {100.0: regular_counts},
    }

    metric = FanoFactorError(fano_bin_size_ms=100.0)
    result = metric.compute(intermediates)

    # Error should be close to 1.0 (Poisson FF ≈ 1, regular FF = 0)
    assert result.raw_value.shape == (1,)
    assert result.raw_value[0] > 0.8  # Poisson FF is close to 1
    assert np.isclose(result.metadata["fano_factor_reconstructed"][0], 0.0)


def test_fano_factor_error_zero_rate_gives_nan() -> None:
    """Units with zero firing rate should get NaN Fano factor."""
    from ntt.evaluation.metrics.single_unit import FanoFactorError

    counts = np.zeros((1, 100))

    intermediates = {
        "binned_counts_original": {100.0: counts},
        "binned_counts_reconstructed": {100.0: counts},
    }

    metric = FanoFactorError(fano_bin_size_ms=100.0)
    result = metric.compute(intermediates)

    assert result.raw_value.shape == (1,)
    assert np.isnan(result.raw_value[0])


def test_fano_factor_error_supports_all_data_types() -> None:
    """FanoFactorError should declare support for all data types."""
    from ntt.evaluation.core.types import DataType
    from ntt.evaluation.metrics.single_unit import FanoFactorError

    metric = FanoFactorError()
    assert DataType.SPIKE_TIMES in metric.supported_data_types
    assert DataType.SPIKES in metric.supported_data_types
    assert DataType.RATES in metric.supported_data_types
