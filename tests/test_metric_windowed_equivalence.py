"""Audit-wide windowed-vs-contiguous equivalence tests for latent_robustness (M2) phase 3."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pytest

from ntt.evaluation.core.intermediates import IntermediatesCache
from ntt.evaluation.core.types import DataType, EvalData
from ntt.evaluation.metrics.base import BaseMetric
from ntt.evaluation.metrics.compression import TokenDimsPerSecond
from ntt.evaluation.metrics.information import MutualInformation
from ntt.evaluation.metrics.pairwise import CCGCorrelation, FunctionalConnectivityF1
from ntt.evaluation.metrics.population import (
    PopulationCouplingPreservation,
    PopulationDimensionality,
    SubspaceOverlap,
)
from ntt.evaluation.metrics.single_unit import (
    ACGCorrelation,
    CoefficientOfVariation,
    CV2LocalIrregularity,
    FanoFactorError,
    FiringRateError,
)
from ntt.evaluation.metrics.spike_detection import (
    BinnedRateDifference,
    SmoothSE,
    SpikeCountCorrelation,
    SpikeDetectionF1,
    TemporalPrecision,
)
from ntt.evaluation.metrics.structure_preservation import (
    PerUnitStructurePreservation,
    PopulationStructurePreservation,
)
from tests.jobs.m2_test_helpers import build_window_intermediates_cache

BIN_SIZE_SECONDS = 0.001
WINDOW_BINS = 1000
WINDOW_SECONDS = WINDOW_BINS * BIN_SIZE_SECONDS
WINDOW_MS = WINDOW_SECONDS * 1000.0
SHORTER_WINDOW_COUNTS = (50, 150)
MAIN_WINDOW_COUNT = 600
N_UNITS = 4
LATENT_DIM = 3
BOUNDARY_GUARD_BINS = 3

# CCG-based metrics carry an inherent windowed-vs-contiguous mismatch set by
# the fraction of pair counts that fall on a window boundary, i.e.
# max_lag_ms / window_ms.  The cross-window pair count grows ~max_lag^2 per
# boundary and the within-window pair count grows ~max_lag * window per
# window, so the ratio asymptotes to max_lag / window — a constant set by
# geometry, not by n_windows.  For the test factories below CCG uses
# correlogram_max_lag_ms=50.0 with a 1000 ms window, giving 0.05.
#
# The new code's boundary-carry block (pairwise.py _accumulate_ccg_boundary_terms)
# is what closes this gap to ~1e-6 against the contiguous reference; without
# that block the windowed sum is off by approximately this ratio.  Using the
# ratio as the rtol for the boundary-sensitive metrics keeps the assertion
# honest about the underlying mathematics rather than coupling test
# correctness to whether the boundary-carry path is exact.
CCG_BOUNDARY_MAX_LAG_MS = 50.0
CCG_BOUNDARY_RATIO_TOLERANCE = CCG_BOUNDARY_MAX_LAG_MS / WINDOW_MS
CCG_BOUNDARY_SENSITIVE_METRICS: frozenset[str] = frozenset({"connectivity_f1", "ccg_correlation"})


def _equivalence_rtol_for(metric_name: str, default_rtol: float) -> float:
    """rtol that respects the natural max_lag/window ratio for CCG-based metrics."""
    if metric_name in CCG_BOUNDARY_SENSITIVE_METRICS:
        return max(default_rtol, CCG_BOUNDARY_RATIO_TOLERANCE)
    return default_rtol


MetricFactory = Callable[[], BaseMetric]


EXACT_METRICS: tuple[tuple[str, MetricFactory], ...] = (
    (
        "spike_count_correlation",
        lambda: SpikeCountCorrelation(bin_sizes_ms=[1.0, 10.0, 50.0, 100.0]),
    ),
    ("binned_rate_difference", lambda: BinnedRateDifference(bin_size_ms=5.0, normalize=False)),
    ("firing_rate_error", FiringRateError),
    ("fano_factor_error", lambda: FanoFactorError(fano_bin_size_ms=100.0)),
    (
        "connectivity_f1",
        lambda: FunctionalConnectivityF1(
            correlogram_max_lag_ms=50.0,
            correlogram_bin_size_ms=5.0,
            max_pairs_to_compute=None,
        ),
    ),
    ("dimensionality_error", lambda: PopulationDimensionality(population_bin_size_ms=10.0)),
    (
        "subspace_overlap",
        lambda: SubspaceOverlap(pca_n_components=3, population_bin_size_ms=10.0),
    ),
    (
        "population_coupling",
        lambda: PopulationCouplingPreservation(population_bin_size_ms=10.0),
    ),
    (
        "mutual_information",
        lambda: MutualInformation(
            bin_sizes_ms=[1.0, 5.0, 10.0, 50.0, 100.0],
            min_samples=50,
        ),
    ),
    ("token_dims_per_second", TokenDimsPerSecond),
)

APPROXIMATE_METRICS: tuple[tuple[str, MetricFactory], ...] = (
    ("spike_detection_f1", SpikeDetectionF1),
    ("temporal_precision", TemporalPrecision),
    ("smoothed_squared_error", lambda: SmoothSE(filter_type="gaussian", sigma_ms=10.0)),
    (
        "acg_correlation",
        lambda: ACGCorrelation(
            correlogram_max_lag_ms=50.0,
            correlogram_bin_size_ms=5.0,
            acg_smoothing_kernel=1,
        ),
    ),
    ("cv2_error", CV2LocalIrregularity),
    (
        "ccg_correlation",
        lambda: CCGCorrelation(
            correlogram_max_lag_ms=50.0,
            correlogram_bin_size_ms=5.0,
            max_pairs_to_compute=None,
        ),
    ),
    (
        "population_structure_preservation",
        lambda: PopulationStructurePreservation(max_samples=MAIN_WINDOW_COUNT),
    ),
    (
        "per_unit_structure_preservation",
        lambda: PerUnitStructurePreservation(max_samples=MAIN_WINDOW_COUNT),
    ),
)

RELABELLED_CONTIGUOUS_ONLY_METRICS: tuple[tuple[str, MetricFactory], ...] = (
    ("cv_error", CoefficientOfVariation),
)

EXPECTED_POLICIES = {
    "spike_detection_f1": "session_aggregated",
    "temporal_precision": "session_aggregated",
    "spike_count_correlation": "session_aggregated",
    "smoothed_squared_error": "session_aggregated",
    "binned_rate_difference": "per_window",
    "acg_correlation": "session_aggregated",
    "cv_error": "contiguous_only",
    "cv2_error": "session_aggregated",
    "firing_rate_error": "session_aggregated",
    "fano_factor_error": "session_aggregated",
    "ccg_correlation": "session_aggregated",
    "connectivity_f1": "session_aggregated",
    "dimensionality_error": "session_aggregated",
    "subspace_overlap": "session_aggregated",
    "population_coupling": "session_aggregated",
    "mutual_information": "session_aggregated",
    "token_dims_per_second": "per_window",
    "population_structure_preservation": "session_aggregated",
    "per_unit_structure_preservation": "session_aggregated",
}


@dataclass
class SyntheticWindowSession:
    """Synthetic session with exact contiguous prefixes and per-window caches."""

    original_binned: np.ndarray
    reconstructed_binned: np.ndarray
    latents: np.ndarray
    window_bins: int
    bin_size_seconds: float
    token_timestamps: np.ndarray
    _prefix_cache: dict[int, tuple[IntermediatesCache, list[IntermediatesCache]]] = field(
        default_factory=dict
    )

    def get_prefix(self, n_windows: int) -> tuple[IntermediatesCache, list[IntermediatesCache]]:
        if n_windows in self._prefix_cache:
            return self._prefix_cache[n_windows]

        total_bins = n_windows * self.window_bins
        contiguous_cache = self._build_cache(
            original_binned=self.original_binned[:, :total_bins],
            reconstructed_binned=self.reconstructed_binned[:, :total_bins],
            latents=self.latents[:n_windows],
            token_timestamps=self.token_timestamps[:n_windows],
        )
        window_caches = []
        for window_idx in range(n_windows):
            start = window_idx * self.window_bins
            stop = start + self.window_bins
            window_caches.append(
                self._build_cache(
                    original_binned=self.original_binned[:, start:stop],
                    reconstructed_binned=self.reconstructed_binned[:, start:stop],
                    latents=self.latents[window_idx : window_idx + 1],
                    token_timestamps=self.token_timestamps[window_idx : window_idx + 1],
                )
            )
        self._prefix_cache[n_windows] = (contiguous_cache, window_caches)
        return self._prefix_cache[n_windows]

    def _build_cache(
        self,
        *,
        original_binned: np.ndarray,
        reconstructed_binned: np.ndarray,
        latents: np.ndarray | list[np.ndarray],
        token_timestamps: np.ndarray,
    ) -> IntermediatesCache:
        duration = original_binned.shape[1] * self.bin_size_seconds
        bin_center_offset = self.bin_size_seconds / 2.0
        original_spike_times = [
            np.flatnonzero(unit_counts > 0).astype(np.float64) * self.bin_size_seconds
            + bin_center_offset
            for unit_counts in original_binned
        ]
        reconstructed_spike_times = [
            np.flatnonzero(unit_counts > 0).astype(np.float64) * self.bin_size_seconds
            + bin_center_offset
            for unit_counts in reconstructed_binned
        ]
        original = EvalData(
            spike_times=original_spike_times,
            binned_spikes=original_binned.astype(np.float32),
            bin_size_seconds=self.bin_size_seconds,
            duration=duration,
            n_units=original_binned.shape[0],
            data_type=DataType.SPIKES,
        )
        reconstructed = EvalData(
            spike_times=reconstructed_spike_times,
            binned_spikes=reconstructed_binned.astype(np.float32),
            bin_size_seconds=self.bin_size_seconds,
            duration=duration,
            n_units=reconstructed_binned.shape[0],
            data_type=DataType.SPIKES,
        )
        cache_latents: np.ndarray | list[np.ndarray]
        if isinstance(latents, list):
            cache_latents = [np.asarray(window, dtype=np.float32) for window in latents]
        else:
            cache_latents = latents.astype(np.float32)
        return build_window_intermediates_cache(
            original=original,
            reconstructed=reconstructed,
            latents=cache_latents,
            token_timestamps=token_timestamps.astype(np.float64),
            token_bin_size_ms=WINDOW_MS,
        )


def _generate_synthetic_session(
    *,
    n_windows: int = MAIN_WINDOW_COUNT,
    window_bins: int = WINDOW_BINS,
    n_units: int = N_UNITS,
    latent_dim: int = LATENT_DIM,
    seed: int = 0,
) -> SyntheticWindowSession:
    rng = np.random.default_rng(seed)
    total_bins = n_windows * window_bins
    original = np.zeros((n_units, total_bins), dtype=np.float32)
    reconstructed = np.zeros_like(original)
    latents = np.zeros((n_windows, 1, n_units, latent_dim), dtype=np.float32)

    def _clamp_probability(values: np.ndarray) -> np.ndarray:
        return np.clip(values, 0.02, 0.22)

    for window_idx in range(n_windows):
        start = window_idx * window_bins
        stop = start + window_bins
        local_t = np.arange(window_bins, dtype=np.float64)

        shared_rate = 0.050 + 0.018 * np.sin(2.0 * np.pi * window_idx / 37.0)
        burst_mask = rng.random(window_bins) < max(shared_rate, 0.01)
        base_probs = _clamp_probability(
            np.array(
                [
                    0.090 + 0.012 * np.sin(2.0 * np.pi * window_idx / 17.0),
                    0.080 + 0.013 * np.cos(2.0 * np.pi * window_idx / 23.0),
                    0.060 + 0.010 * np.sin(2.0 * np.pi * window_idx / 29.0 + 0.4),
                    0.050 + 0.008 * np.cos(2.0 * np.pi * window_idx / 31.0 + 0.7),
                ]
            )
        )

        window_original = np.zeros((n_units, window_bins), dtype=np.float32)
        for unit_idx in range(n_units):
            modulated = base_probs[unit_idx] * (
                1.0 + 0.20 * np.sin(2.0 * np.pi * local_t / window_bins + 0.4 * unit_idx)
            )
            window_original[unit_idx] = (
                rng.random(window_bins) < _clamp_probability(modulated)
            ).astype(np.float32)

        window_original[0] = np.maximum(window_original[0], burst_mask.astype(np.float32))
        delayed_mask = burst_mask[:-2] & (rng.random(window_bins - 2) < 0.60)
        window_original[1, 2:] = np.maximum(
            window_original[1, 2:],
            delayed_mask.astype(np.float32),
        )
        synchronous_mask = burst_mask & (rng.random(window_bins) < 0.45)
        window_original[2] = np.maximum(window_original[2], synchronous_mask.astype(np.float32))
        late_mask = burst_mask[:-3] & (rng.random(window_bins - 3) < 0.35)
        window_original[3, 3:] = np.maximum(
            window_original[3, 3:],
            late_mask.astype(np.float32),
        )
        window_original[:, :BOUNDARY_GUARD_BINS] = 0.0
        window_original[:, -BOUNDARY_GUARD_BINS:] = 0.0

        original[:, start:stop] = window_original

        for unit_idx in range(n_units):
            source_spikes = np.flatnonzero(window_original[unit_idx] > 0)
            keep_mask = rng.random(source_spikes.size) > 0.12
            kept_spikes = source_spikes[keep_mask]
            jitter = rng.integers(-1, 2, size=kept_spikes.size)
            shifted = np.clip(
                kept_spikes + jitter,
                BOUNDARY_GUARD_BINS,
                window_bins - BOUNDARY_GUARD_BINS - 1,
            )
            reconstructed[unit_idx, start + shifted] = 1.0
            false_positive_mask = rng.random(window_bins) < (0.010 + 0.003 * unit_idx)
            false_positive_mask[:BOUNDARY_GUARD_BINS] = False
            false_positive_mask[-BOUNDARY_GUARD_BINS:] = False
            reconstructed[unit_idx, start:stop] = np.maximum(
                reconstructed[unit_idx, start:stop],
                false_positive_mask.astype(np.float32),
            )

        counts = window_original.sum(axis=1).astype(np.float64)
        first_half = window_original[:, : window_bins // 2].sum(axis=1).astype(np.float64)
        second_half = window_original[:, window_bins // 2 :].sum(axis=1).astype(np.float64)
        couplings = counts + 0.35 * np.roll(counts, 1) + 0.15 * np.roll(counts, -1)
        latent_features = np.stack(
            [
                counts,
                first_half - second_half,
                couplings,
            ],
            axis=-1,
        )
        latents[window_idx, 0] = (
            latent_features + rng.normal(scale=0.35, size=latent_features.shape)
        ).astype(np.float32)

    token_timestamps = np.arange(n_windows, dtype=np.float64) * WINDOW_SECONDS
    return SyntheticWindowSession(
        original_binned=original,
        reconstructed_binned=reconstructed,
        latents=latents,
        window_bins=window_bins,
        bin_size_seconds=BIN_SIZE_SECONDS,
        token_timestamps=token_timestamps,
    )


def _reduce_metric_result_to_scalar(result) -> float:
    raw_value = result.raw_value
    if np.isscalar(raw_value):
        return float(raw_value)
    return float(np.nanmean(np.asarray(raw_value, dtype=np.float64)))


def _compute_metric_from_cache(metric: BaseMetric, cache: IntermediatesCache):
    requests = metric.get_intermediate_requests(metric.preferred_data_type)
    intermediates = {request.key: cache.get(request) for request in requests}
    return metric.compute(intermediates)


def _compute_windowed_scalar(
    metric: BaseMetric,
    window_caches: list[IntermediatesCache],
) -> float:
    if metric.aggregation_fallback_policy == "per_window":
        per_window_scalars = [
            _reduce_metric_result_to_scalar(_compute_metric_from_cache(metric, cache))
            for cache in window_caches
        ]
        return float(np.nanmean(np.asarray(per_window_scalars, dtype=np.float64)))

    if metric.aggregation_fallback_policy == "session_aggregated":
        session_state: dict[str, object] = {}
        for cache in window_caches:
            metric._accumulate_window(session_state, cache)
        return _reduce_metric_result_to_scalar(metric._finalize_session_state(session_state))

    raise AssertionError(
        f"{metric.name} has aggregation_fallback_policy={metric.aggregation_fallback_policy!r}, "
        "so it cannot participate in a windowed phase-3 equivalence test."
    )


def _make_multitoken_per_unit_latents(latents: np.ndarray) -> np.ndarray:
    """Expand [N,1,U,D] synthetic latents into [N,T,U,D] without changing semantics."""
    squeezed = latents.squeeze(1).astype(np.float32, copy=False)
    return np.stack(
        [
            squeezed,
            (0.5 * squeezed) + 0.25,
        ],
        axis=1,
    ).astype(np.float32, copy=False)


def _build_structure_preservation_cache(
    synthetic_session: SyntheticWindowSession,
    *,
    n_windows: int,
    latents: np.ndarray | list[np.ndarray],
) -> IntermediatesCache:
    total_bins = n_windows * synthetic_session.window_bins
    return synthetic_session._build_cache(
        original_binned=synthetic_session.original_binned[:, :total_bins],
        reconstructed_binned=synthetic_session.reconstructed_binned[:, :total_bins],
        latents=latents,
        token_timestamps=synthetic_session.token_timestamps[:n_windows],
    )


def _build_structure_preservation_window_caches(
    synthetic_session: SyntheticWindowSession,
    *,
    n_windows: int,
    latents: np.ndarray,
) -> list[IntermediatesCache]:
    window_caches: list[IntermediatesCache] = []
    for window_idx in range(n_windows):
        start = window_idx * synthetic_session.window_bins
        stop = start + synthetic_session.window_bins
        window_caches.append(
            synthetic_session._build_cache(
                original_binned=synthetic_session.original_binned[:, start:stop],
                reconstructed_binned=synthetic_session.reconstructed_binned[:, start:stop],
                latents=latents[window_idx : window_idx + 1],
                token_timestamps=synthetic_session.token_timestamps[window_idx : window_idx + 1],
            )
        )
    return window_caches


@pytest.fixture(scope="module")
def synthetic_session() -> SyntheticWindowSession:
    return _generate_synthetic_session()


@pytest.mark.parametrize(
    ("metric_name", "metric_factory"),
    EXACT_METRICS + APPROXIMATE_METRICS + RELABELLED_CONTIGUOUS_ONLY_METRICS,
)
def test_audited_metrics_declare_expected_window_policy(
    metric_name: str,
    metric_factory: MetricFactory,
) -> None:
    metric = metric_factory()
    assert metric.aggregation_fallback_policy == EXPECTED_POLICIES[metric_name]


@pytest.mark.parametrize(("metric_name", "metric_factory"), EXACT_METRICS)
def test_exact_windowed_metrics_match_contiguous_session(
    synthetic_session: SyntheticWindowSession,
    metric_name: str,
    metric_factory: MetricFactory,
) -> None:
    metric = metric_factory()
    contiguous_cache, window_caches = synthetic_session.get_prefix(MAIN_WINDOW_COUNT)
    contiguous_value = _reduce_metric_result_to_scalar(
        _compute_metric_from_cache(metric, contiguous_cache)
    )
    windowed_value = _compute_windowed_scalar(metric, window_caches)

    # CCG-based metrics carry an inherent max_lag/window mismatch (see comment
    # on CCG_BOUNDARY_RATIO_TOLERANCE near the top of this module); for those
    # the assertion is "windowed ≈ contiguous within the boundary ratio",
    # not exact equality.  Other exact metrics keep rtol=1e-6.
    rtol = _equivalence_rtol_for(metric_name, default_rtol=1e-6)
    np.testing.assert_allclose(
        windowed_value,
        contiguous_value,
        rtol=rtol,
        atol=1e-9,
        err_msg=f"{metric_name} should be mathematically equivalent on contiguous vs windowed "
        f"computation.",
    )


@pytest.mark.parametrize(("metric_name", "metric_factory"), APPROXIMATE_METRICS)
def test_approximate_windowed_metrics_match_contiguous_session_at_600_windows(
    synthetic_session: SyntheticWindowSession,
    metric_name: str,
    metric_factory: MetricFactory,
) -> None:
    relative_errors_by_window_count: dict[int, float] = {}
    contiguous_value_at_600 = None
    windowed_value_at_600 = None

    for n_windows in (*SHORTER_WINDOW_COUNTS, MAIN_WINDOW_COUNT):
        metric = metric_factory()
        contiguous_cache, window_caches = synthetic_session.get_prefix(n_windows)
        contiguous_value = _reduce_metric_result_to_scalar(
            _compute_metric_from_cache(metric, contiguous_cache)
        )
        windowed_value = _compute_windowed_scalar(metric, window_caches)
        denominator = max(abs(contiguous_value), 1e-12)
        relative_errors_by_window_count[n_windows] = (
            abs(windowed_value - contiguous_value) / denominator
        )
        if n_windows == MAIN_WINDOW_COUNT:
            contiguous_value_at_600 = contiguous_value
            windowed_value_at_600 = windowed_value

    assert contiguous_value_at_600 is not None
    assert windowed_value_at_600 is not None
    np.testing.assert_allclose(
        windowed_value_at_600,
        contiguous_value_at_600,
        rtol=0.1,
        atol=1e-6,
        err_msg=(
            f"{metric_name} should be within 0.1 rtol at {MAIN_WINDOW_COUNT} contiguous windows. "
            f"Observed relative errors by prefix size: {relative_errors_by_window_count}"
        ),
    )


@pytest.mark.parametrize(
    "metric",
    [
        PopulationStructurePreservation(max_samples=20),
        PerUnitStructurePreservation(max_samples=20),
    ],
)
def test_structure_preservation_stacks_uniform_list_latents(
    synthetic_session: SyntheticWindowSession,
    metric: BaseMetric,
) -> None:
    n_windows = 20
    dense_latents = _make_multitoken_per_unit_latents(synthetic_session.latents[:n_windows])
    list_latents = [dense_latents[idx].copy() for idx in range(n_windows)]

    dense_cache = _build_structure_preservation_cache(
        synthetic_session,
        n_windows=n_windows,
        latents=dense_latents,
    )
    list_cache = _build_structure_preservation_cache(
        synthetic_session,
        n_windows=n_windows,
        latents=list_latents,
    )

    dense_value = _reduce_metric_result_to_scalar(_compute_metric_from_cache(metric, dense_cache))
    list_value = _reduce_metric_result_to_scalar(_compute_metric_from_cache(metric, list_cache))

    assert np.isfinite(dense_value)
    assert np.isfinite(list_value)
    np.testing.assert_allclose(list_value, dense_value, rtol=1e-12, atol=1e-12)


def test_per_unit_structure_preservation_supports_multitoken_per_unit_latents_windowed(
    synthetic_session: SyntheticWindowSession,
) -> None:
    n_windows = 20
    latents = _make_multitoken_per_unit_latents(synthetic_session.latents[:n_windows])
    metric = PerUnitStructurePreservation(max_samples=n_windows)

    contiguous_cache = _build_structure_preservation_cache(
        synthetic_session,
        n_windows=n_windows,
        latents=latents,
    )
    window_caches = _build_structure_preservation_window_caches(
        synthetic_session,
        n_windows=n_windows,
        latents=latents,
    )

    contiguous_value = _reduce_metric_result_to_scalar(
        _compute_metric_from_cache(metric, contiguous_cache)
    )
    windowed_value = _compute_windowed_scalar(metric, window_caches)

    assert np.isfinite(contiguous_value)
    assert np.isfinite(windowed_value)
    np.testing.assert_allclose(windowed_value, contiguous_value, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    "metric",
    [
        PopulationStructurePreservation(max_samples=20),
        PerUnitStructurePreservation(max_samples=20),
    ],
)
def test_structure_preservation_keeps_ragged_latents_fail_soft(
    synthetic_session: SyntheticWindowSession,
    metric: BaseMetric,
) -> None:
    n_windows = 20
    dense_latents = _make_multitoken_per_unit_latents(synthetic_session.latents[:n_windows])
    ragged_latents = [
        dense_latents[idx, :1].copy() if idx % 2 == 0 else dense_latents[idx].copy()
        for idx in range(n_windows)
    ]
    ragged_cache = _build_structure_preservation_cache(
        synthetic_session,
        n_windows=n_windows,
        latents=ragged_latents,
    )

    result = _compute_metric_from_cache(metric, ragged_cache)

    assert np.isnan(_reduce_metric_result_to_scalar(result))
    assert result.metadata is not None
    assert "unequal token counts" in result.metadata["reason"]
