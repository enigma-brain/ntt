from __future__ import annotations

import numpy as np
import pytest

from ntt.evaluation.core.types import EvalData
from ntt.evaluation.metrics.single_unit import ACGCorrelation
from ntt.jobs.eval_robustness_latent import _aggregate_acg_mean_then_correlate
from tests.jobs.m2_test_helpers import build_window_intermediates_cache


class _StubWindowIntermediates:
    def __init__(self, values: dict[str, object]):
        self._values = values

    def get(self, request):
        return self._values[request.key]


def _make_spike_eval_data(binned_spikes: np.ndarray) -> EvalData:
    return EvalData(
        binned_spikes=binned_spikes.astype(np.int8, copy=False),
        bin_size_seconds=0.001,
        duration=binned_spikes.shape[1] * 0.001,
    )


@pytest.fixture
def acg_window_caches() -> list:
    rng = np.random.default_rng(0)
    caches = []
    for _ in range(3):
        clean = (rng.random((4, 256)) > 0.84).astype(np.int8)
        noisy = np.logical_xor(clean, rng.random(clean.shape) > 0.97).astype(np.int8)
        caches.append(
            build_window_intermediates_cache(
                original=_make_spike_eval_data(clean),
                reconstructed=_make_spike_eval_data(noisy),
            )
        )
    return caches


def test_acg_correlation_declares_session_aggregated_policy() -> None:
    assert ACGCorrelation.aggregation_fallback_policy == "session_aggregated"


def test_acg_correlation_session_aggregated_matches_mean_then_correlate(
    acg_window_caches: list,
) -> None:
    metric = ACGCorrelation()
    requests = {
        request.key: request
        for request in metric.get_intermediate_requests(metric.preferred_data_type)
    }

    expected = _aggregate_acg_mean_then_correlate(
        [cache.get(requests["acg_original"]) for cache in acg_window_caches],
        [cache.get(requests["acg_reconstructed"]) for cache in acg_window_caches],
    )

    state: dict[str, object] = {}
    for cache in acg_window_caches:
        metric._accumulate_window(state, cache)

    result = metric._finalize_session_state(state)

    assert float(result.raw_value) == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert result.metadata["n_windows_seen"] == len(acg_window_caches)
    assert len(result.metadata["per_unit_corrs"]) == 4


def test_acg_correlation_session_aggregated_raises_on_all_nan_correlations() -> None:
    zeros = np.zeros((4, 256), dtype=np.int8)
    cache = build_window_intermediates_cache(
        original=_make_spike_eval_data(zeros),
        reconstructed=_make_spike_eval_data(zeros),
    )
    metric = ACGCorrelation()
    state: dict[str, object] = {}

    metric._accumulate_window(state, cache)

    with pytest.raises(ValueError, match="every per-unit Pearson correlation is NaN"):
        metric._finalize_session_state(state)


def test_acg_correlation_session_aggregated_raises_on_partial_nan_window() -> None:
    metric = ACGCorrelation()
    state: dict[str, object] = {}
    cache = _StubWindowIntermediates(
        {
            "acg_original": [np.array([1.0, np.nan, 3.0]), np.array([1.0, 2.0, 3.0])],
            "acg_reconstructed": [np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])],
        }
    )

    metric._accumulate_window(state, cache)

    with pytest.raises(ValueError, match="contains NaN"):
        metric._finalize_session_state(state)
