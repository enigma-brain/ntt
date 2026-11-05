"""Helpers shared across windowed metric implementations.

Includes ``metric_requests_by_key`` and ``_extract_spike_times``, used by both
``single_unit`` and ``spike_detection``, and the sufficient-statistics kernels
(``_variance_from_sufficient_statistics`` / ``_pearson_from_sufficient_statistics``),
the shared running-moment math the session-aggregated metrics rely on. Keeping
them in one place keeps the windowed accumulate/finalize logic consistent across
metrics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ..base import BaseMetric, IntermediateRequest

if TYPE_CHECKING:
    # Type-checker-only base so mixins can use ``self.params``, ``self._make_result`` etc.
    # At runtime mixins stay plain classes (the concrete metric supplies BaseMetric).
    MetricMixinBase = BaseMetric
else:
    MetricMixinBase = object


def metric_requests_by_key(metric: BaseMetric) -> dict[str, IntermediateRequest]:
    """Map intermediate-request key -> request for a metric's preferred data type."""
    return {
        request.key: request
        for request in metric.get_intermediate_requests(metric.preferred_data_type)
    }


def require_finite_2d_counts(counts: Any, *, context: str) -> np.ndarray:
    """Validate and return a finite, non-negative 2D ``[units, time]`` count array."""
    array = np.asarray(counts, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{context}: expected a 2D [units, time] array, got {array.shape}.")
    if np.any(~np.isfinite(array)):
        raise ValueError(f"{context}: counts contain NaN or inf.")
    if np.any(array < 0):
        raise ValueError(f"{context}: counts contain negative values.")
    return array


def variance_from_sufficient_statistics(
    *,
    sum_values: np.ndarray,
    sum_squares: np.ndarray,
    counts: int | np.ndarray,
    context: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-element (mean, variance) from running sum / sum-of-squares / counts.

    Returns NaN where ``counts == 0``; clamps tiny negative variances (round-off)
    to zero and raises if any variance is meaningfully negative.
    """
    if sum_values.shape != sum_squares.shape:
        raise ValueError(
            f"{context}: sum_values and sum_squares shape mismatch: "
            f"{sum_values.shape} vs {sum_squares.shape}."
        )

    counts_array = np.broadcast_to(np.asarray(counts, dtype=np.float64), sum_values.shape)
    if np.any(counts_array < 0):
        raise ValueError(f"{context}: counts must be non-negative.")

    mean = np.full(sum_values.shape, np.nan, dtype=np.float64)
    variance = np.full(sum_values.shape, np.nan, dtype=np.float64)
    valid = counts_array > 0
    mean[valid] = sum_values[valid] / counts_array[valid]
    variance[valid] = (sum_squares[valid] / counts_array[valid]) - (mean[valid] ** 2)
    if np.any(variance[valid] < -1e-12):
        raise ValueError(f"{context}: accumulated sufficient statistics imply negative variance.")
    variance[valid] = np.maximum(variance[valid], 0.0)
    return mean, variance


def pearson_from_sufficient_statistics(
    *,
    sum_x: np.ndarray,
    sum_y: np.ndarray,
    sum_x2: np.ndarray,
    sum_y2: np.ndarray,
    sum_xy: np.ndarray,
    n_observations: int,
    context: str,
) -> np.ndarray:
    """Per-element Pearson correlation from running sufficient statistics.

    Returns NaN where either variance is zero (or ``n_observations == 0``);
    raises if any accumulated variance is meaningfully negative.
    """
    correlations = np.full(sum_x.shape, np.nan, dtype=np.float64)
    if n_observations < 0:
        raise ValueError(f"{context}: n_observations must be non-negative, got {n_observations}.")
    if n_observations == 0:
        return correlations

    numerator = (n_observations * sum_xy) - (sum_x * sum_y)
    denom_x = (n_observations * sum_x2) - (sum_x**2)
    denom_y = (n_observations * sum_y2) - (sum_y**2)
    negative_variance = (denom_x < -1e-12) | (denom_y < -1e-12)
    if np.any(negative_variance):
        raise ValueError(f"{context}: accumulated sufficient statistics imply negative variance.")

    denom_x = np.maximum(denom_x, 0.0)
    denom_y = np.maximum(denom_y, 0.0)
    valid = (denom_x > 0) & (denom_y > 0)
    correlations[valid] = numerator[valid] / np.sqrt(denom_x[valid] * denom_y[valid])
    return correlations


def extract_spike_times(data: Any, *, context: str) -> list[np.ndarray]:
    """Recover per-unit spike-time arrays from spike-times or binary SPIKES data."""
    if data is None:
        raise ValueError(f"{context}: EvalData is required.")
    if data.spike_times is not None:
        spike_times = []
        for unit_spike_times in data.spike_times:
            array = np.asarray(unit_spike_times, dtype=np.float64)
            if array.ndim != 1:
                raise ValueError(f"{context}: spike_times must be 1D per unit, got {array.shape}.")
            if np.any(~np.isfinite(array)):
                raise ValueError(f"{context}: spike_times contain NaN or inf.")
            spike_times.append(array)
        return spike_times

    if data.binned_spikes is None or data.bin_size_seconds is None:
        raise ValueError(
            f"{context}: spike_times or binary binned_spikes with bin_size_seconds are required."
        )

    spikes = np.asarray(data.binned_spikes, dtype=np.float64)
    if spikes.ndim != 2:
        raise ValueError(f"{context}: expected 2D binned_spikes, got {spikes.shape}.")
    if np.any(~np.isfinite(spikes)):
        raise ValueError(f"{context}: binned_spikes contain NaN or inf.")
    if np.any((spikes < 0) | (spikes > 1)):
        raise ValueError(f"{context}: SPIKES data must be binary to recover spike times exactly.")

    bin_size_seconds = float(data.bin_size_seconds)
    return [
        np.flatnonzero(unit_spikes > 0).astype(np.float64) * bin_size_seconds
        for unit_spikes in spikes
    ]


def get_duration_seconds(data: Any, *, context: str) -> float:
    """Return a finite, positive duration in seconds, inferring from bins if needed."""
    if data is None:
        raise ValueError(f"{context}: EvalData is required.")
    duration = data.duration
    if duration is None and data.bin_size_seconds is not None and data.binned_spikes is not None:
        duration = data.binned_spikes.shape[1] * data.bin_size_seconds
    if duration is None:
        raise ValueError(f"{context}: duration is required.")
    duration = float(duration)
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError(f"{context}: duration must be finite and positive, got {duration}.")
    return duration
