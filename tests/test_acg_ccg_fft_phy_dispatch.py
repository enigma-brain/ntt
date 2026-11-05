"""ACG/CCG FFT-vs-Phy dispatch (``IntermediatesCache._use_fft_correlograms``).

The Phy-compat FFT path and the all-pairs Phy matrix are bit-identical; the
dispatch only decides which is faster. These tests pin (a) the gate decisions
and (b) that whichever route runs, the ACG/CCG output is the same — so the
performance gate can never silently change results.
"""

from __future__ import annotations

import numpy as np
import pytest

from ntt.evaluation.core.intermediates import (
    _FFT_LONG_TRACE_MIN_LAG_MS,
    _FFT_SHORT_TRACE_MAX_1MS_BINS,
    IntermediatesCache,
)
from ntt.evaluation.core.types import EvalData
from ntt.evaluation.metrics.base import IntermediateRequest


def _spike_data(n_units: int, duration: float, rate_hz: float, seed: int) -> EvalData:
    rng = np.random.default_rng(seed)
    spikes = [
        np.sort(rng.uniform(0.0, duration, size=rng.poisson(rate_hz * duration)))
        for _ in range(n_units)
    ]
    return EvalData.from_spike_times_list(spikes, duration=duration)


# --------------------------------------------------------------------------- #
# Gate decisions
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("duration", "max_lag_ms", "bin_size_ms", "expected"),
    [
        # Short (windowed-latent_robustness (M2)) traces always take the FFT path.
        (1.0, 100.0, 1.0, True),
        (1.0, 500.0, 5.0, True),
        (8.0, 100.0, 1.0, True),
        # Long (full-session) traces: FFT only at wide lags.
        (120.0, 100.0, 1.0, False),  # the default 1 ms / 100 ms regression case
        (120.0, 500.0, 5.0, True),
        (120.0, 1000.0, 10.0, True),
        (120.0, 200.0, 1.0, False),  # lag below threshold -> Phy
        (120.0, 250.0, 1.0, True),  # lag at threshold -> FFT
    ],
)
def test_gate_decision(duration, max_lag_ms, bin_size_ms, expected) -> None:
    data = _spike_data(8, duration, rate_hz=15, seed=0)
    cache = IntermediatesCache(data, data)
    assert (
        cache._use_fft_correlograms(data, max_lag_ms=max_lag_ms, bin_size_ms=bin_size_ms)
        is expected
    )


def test_gate_rejects_non_integer_or_misaligned_bins() -> None:
    data = _spike_data(8, 1.0, rate_hz=15, seed=0)
    cache = IntermediatesCache(data, data)
    # Non-integer-ms bin: FFT path is ineligible -> Phy.
    assert cache._use_fft_correlograms(data, max_lag_ms=100.0, bin_size_ms=2.5) is False
    # max_lag not an integer multiple of bin_size -> Phy.
    assert cache._use_fft_correlograms(data, max_lag_ms=105.0, bin_size_ms=10.0) is False


def test_threshold_constants_are_sane() -> None:
    assert _FFT_SHORT_TRACE_MAX_1MS_BINS >= 1000  # at least a 1 s window
    assert 0.0 < _FFT_LONG_TRACE_MIN_LAG_MS <= 1000.0


# --------------------------------------------------------------------------- #
# Both routes produce identical correlograms
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("max_lag_ms", "bin_size_ms"), [(100.0, 1.0), (500.0, 5.0), (1000.0, 10.0)]
)
def test_fft_and_phy_acg_are_bit_identical(max_lag_ms, bin_size_ms) -> None:
    data = _spike_data(12, 3.0, rate_hz=20, seed=1)
    cache = IntermediatesCache(data, data)

    fft = cache._compute_acgs_phy_compat_fft(
        data, max_lag_ms=max_lag_ms, bin_size_ms=bin_size_ms, smoothing_kernel=1
    )
    all_corr = cache._compute_all_correlograms_from_spike_times(data, max_lag_ms, bin_size_ms)
    phy = cache._extract_acgs(all_corr, smoothing_kernel=1)

    assert len(fft) == len(phy)
    for a, b in zip(fft, phy, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize(
    ("max_lag_ms", "bin_size_ms"), [(100.0, 1.0), (500.0, 5.0), (1000.0, 10.0)]
)
def test_fft_and_phy_ccg_are_bit_identical(max_lag_ms, bin_size_ms) -> None:
    data = _spike_data(12, 3.0, rate_hz=20, seed=2)
    cache = IntermediatesCache(data, data)
    max_pairs = 30

    fft = cache._compute_ccgs_phy_compat_fft(
        data, max_lag_ms=max_lag_ms, bin_size_ms=bin_size_ms, max_pairs=max_pairs
    )
    all_corr = cache._compute_all_correlograms_from_spike_times(data, max_lag_ms, bin_size_ms)
    phy = cache._extract_ccgs(all_corr, max_pairs=max_pairs)

    assert set(fft) == set(phy)
    for pair in fft:
        np.testing.assert_array_equal(fft[pair], phy[pair])


def test_cache_get_acg_matches_across_gate(monkeypatch) -> None:
    """``cache.get('acg_*')`` returns the same values whichever route the gate picks."""
    data = _spike_data(10, 2.0, rate_hz=18, seed=3)
    params = {"max_lag_ms": 100.0, "bin_size_ms": 1.0}

    def _request() -> IntermediateRequest:
        return IntermediateRequest.from_dict("acg_original", {**params, "smoothing_kernel": 1})

    cache_fft = IntermediatesCache(data, data)
    monkeypatch.setattr(cache_fft, "_use_fft_correlograms", lambda *a, **k: True)
    via_fft = cache_fft.get(_request())

    cache_phy = IntermediatesCache(data, data)
    monkeypatch.setattr(cache_phy, "_use_fft_correlograms", lambda *a, **k: False)
    via_phy = cache_phy.get(_request())

    assert len(via_fft) == len(via_phy)
    for a, b in zip(via_fft, via_phy, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
