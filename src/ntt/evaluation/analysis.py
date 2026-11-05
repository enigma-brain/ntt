"""User-facing convenience functions for computing neural data intermediates.

Thin wrappers around IntermediatesCache that hide the request/cache plumbing.
Intended for interactive use in notebooks and scripts.

Example::

    from ntt.evaluation import compute_acgs, EvalData

    data = EvalData.from_tensor(tensor, DataType.SPIKES, bin_size_seconds=0.005)
    acgs = compute_acgs(data, max_lag_ms=100, bin_size_ms=5)
"""

from __future__ import annotations

import numpy as np

from .core.intermediates import IntermediatesCache
from .core.types import EvalData
from .metrics.base import IntermediateRequest


def compute_acgs(
    data: EvalData,
    *,
    max_lag_ms: float = 100.0,
    bin_size_ms: float = 1.0,
    smoothing_kernel: int = 1,
) -> list[np.ndarray]:
    """Compute per-unit autocorrelograms.

    Args:
        data: Neural data (SPIKE_TIMES, SPIKES, or RATES).
        max_lag_ms: Maximum lag in milliseconds.
        bin_size_ms: Correlogram bin size in milliseconds.
            For RATES/coarse SPIKES, auto-adjusted upward if finer than data.
        smoothing_kernel: Boxcar smoothing kernel size (1 = no smoothing).

    Returns:
        List of length n_units, each a 1D array of shape (n_corr_bins,).
    """
    cache = IntermediatesCache(data, data)
    request = IntermediateRequest.from_dict(
        "acg_original",
        {
            "max_lag_ms": max_lag_ms,
            "bin_size_ms": bin_size_ms,
            "smoothing_kernel": smoothing_kernel,
        },
    )
    return cache.get(request)


def compute_ccgs(
    data: EvalData,
    *,
    max_lag_ms: float = 100.0,
    bin_size_ms: float = 1.0,
    max_pairs: int | None = 20,
) -> dict[tuple[int, int], np.ndarray]:
    """Compute cross-correlograms for unit pairs.

    Args:
        data: Neural data (SPIKE_TIMES, SPIKES, or RATES).
        max_lag_ms: Maximum lag in milliseconds.
        bin_size_ms: Correlogram bin size in milliseconds.
        max_pairs: Maximum number of unit pairs to compute.
            None = all pairs (can be slow for many units).

    Returns:
        Dict mapping (i, j) tuples to 1D CCG arrays.
    """
    cache = IntermediatesCache(data, data)
    request = IntermediateRequest.from_dict(
        "ccg_original",
        {"max_lag_ms": max_lag_ms, "bin_size_ms": bin_size_ms, "max_pairs": max_pairs},
    )
    return cache.get(request)


def compute_isis(
    data: EvalData,
    *,
    smoothing_kernel: int = 1,
) -> list[np.ndarray]:
    """Compute inter-spike interval sequences per unit.

    Args:
        data: Neural data with spike_times (DataType.SPIKE_TIMES).
        smoothing_kernel: Boxcar smoothing kernel size (1 = no smoothing).

    Returns:
        List of length n_units, each a 1D array of ISI values (seconds).

    Raises:
        ValueError: If data does not contain spike_times.
    """
    cache = IntermediatesCache(data, data)
    request = IntermediateRequest.from_dict(
        "isi_original",
        {"smoothing_kernel": smoothing_kernel},
    )
    return cache.get(request)
