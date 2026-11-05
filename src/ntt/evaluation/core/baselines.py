"""Shared utilities for binned-count operations.

Note: the Poisson baseline *tokenizer* lives in
``ntt.contrib.tokenizer.poisson_baseline``.  This module retains only the
helper functions used by evaluation metrics (e.g. ``BinnedRateDifference``).
"""

import numpy as np


def _rebin_counts(counts: np.ndarray, base_bin_size: float, target_bin_size: float) -> np.ndarray:
    """Aggregate binned counts into larger bins."""
    if target_bin_size < base_bin_size - 1e-12:
        raise ValueError("Target bin size must be >= source bin size for baseline estimation.")
    if np.isclose(target_bin_size, base_bin_size):
        return counts

    ratio = target_bin_size / base_bin_size
    if not np.isclose(ratio, round(ratio)):
        raise ValueError("Target bin size must be an integer multiple of source bin size.")

    ratio_int = int(round(ratio))
    n_bins = counts.shape[1] // ratio_int
    truncated = counts[:, : n_bins * ratio_int]
    return truncated.reshape(counts.shape[0], n_bins, ratio_int).sum(axis=2)
