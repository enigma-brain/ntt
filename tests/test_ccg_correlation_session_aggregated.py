from __future__ import annotations

import numpy as np
import pytest

from ntt.evaluation.core.types import EvalData
from ntt.evaluation.metrics.pairwise import CCGCorrelation
from ntt.jobs.eval_robustness_latent import _aggregate_ccg_mean_then_correlate
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
def ccg_window_caches() -> list:
    rng = np.random.default_rng(1)
    caches = []
    for _ in range(3):
        clean = (rng.random((5, 256)) > 0.85).astype(np.int8)
        noisy = np.logical_xor(clean, rng.random(clean.shape) > 0.975).astype(np.int8)
        caches.append(
            build_window_intermediates_cache(
                original=_make_spike_eval_data(clean),
                reconstructed=_make_spike_eval_data(noisy),
            )
        )
    return caches


def test_ccg_correlation_declares_session_aggregated_policy() -> None:
    assert CCGCorrelation.aggregation_fallback_policy == "session_aggregated"


def test_ccg_correlation_session_aggregated_matches_mean_then_correlate(
    ccg_window_caches: list,
) -> None:
    """Verifies that the session-aggregated CCG metric reproduces
    ``_aggregate_ccg_mean_then_correlate`` on the same per-window CCG dicts.

    ``CCGCorrelation._accumulate_window`` also runs a cross-window boundary-
    carry step that uses ``window_intermediates.original/.reconstructed``
    to add the contribution from spike pairs straddling adjacent window
    edges. ``_aggregate_ccg_mean_then_correlate`` does NOT — it sums
    per-window CCGs as-is, so the two only match when boundary-carry is
    skipped. We reproduce the no-carry condition by wrapping the per-window
    CCG dicts in ``_StubWindowIntermediates`` (no ``.original``/
    ``.reconstructed`` attributes); the metric's getattr-with-None fallback
    sees None on both sides and ``_accumulate_ccg_boundary_terms`` returns
    early without touching the running sum.
    """
    metric = CCGCorrelation()
    requests = {
        request.key: request
        for request in metric.get_intermediate_requests(metric.preferred_data_type)
    }

    expected = _aggregate_ccg_mean_then_correlate(
        [cache.get(requests["ccg_original"]) for cache in ccg_window_caches],
        [cache.get(requests["ccg_reconstructed"]) for cache in ccg_window_caches],
    )

    stub_windows = [
        _StubWindowIntermediates(
            {
                "ccg_original": cache.get(requests["ccg_original"]),
                "ccg_reconstructed": cache.get(requests["ccg_reconstructed"]),
            }
        )
        for cache in ccg_window_caches
    ]
    state: dict[str, object] = {}
    for window in stub_windows:
        metric._accumulate_window(state, window)

    result = metric._finalize_session_state(state)

    assert float(result.raw_value) == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert result.metadata["n_windows_seen"] == len(stub_windows)
    assert len(result.metadata["pair_correlations"]) > 0


def test_ccg_correlation_session_aggregated_raises_on_all_nan_correlations() -> None:
    zeros = np.zeros((5, 256), dtype=np.int8)
    cache = build_window_intermediates_cache(
        original=_make_spike_eval_data(zeros),
        reconstructed=_make_spike_eval_data(zeros),
    )
    metric = CCGCorrelation()
    state: dict[str, object] = {}

    metric._accumulate_window(state, cache)

    with pytest.raises(ValueError, match="every per-pair Pearson correlation is NaN"):
        metric._finalize_session_state(state)


def test_ccg_correlation_session_aggregated_raises_on_pair_set_drift() -> None:
    metric = CCGCorrelation()
    state: dict[str, object] = {}
    first = _StubWindowIntermediates(
        {
            "ccg_original": {
                (0, 1): np.array([1.0, 2.0, 3.0]),
                (0, 2): np.array([1.0, 2.0, 3.0]),
            },
            "ccg_reconstructed": {
                (0, 1): np.array([1.0, 2.0, 3.0]),
                (0, 2): np.array([1.0, 2.0, 3.0]),
            },
        }
    )
    second = _StubWindowIntermediates(
        {
            "ccg_original": {(0, 1): np.array([1.0, 2.0, 3.0])},
            "ccg_reconstructed": {(0, 1): np.array([1.0, 2.0, 3.0])},
        }
    )

    metric._accumulate_window(state, first)

    with pytest.raises(ValueError, match="pair order changed across windows|pair sets differ"):
        metric._accumulate_window(state, second)
