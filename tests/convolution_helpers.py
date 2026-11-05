from __future__ import annotations

from typing import cast

import numpy as np
from scipy.stats import expon, norm


def _time_grid(duration_ms: float = 1000.0) -> np.ndarray:
    """Returns [-duration_ms/2, ..., 0, ..., duration_ms/2]"""
    half_len = int(duration_ms / 2)
    return np.linspace(-half_len, half_len, 2 * half_len + 1, endpoint=True)


def gaussian_kernel(sigma_ms: float, shift_ms: float = 0.0) -> np.ndarray:
    grid = _time_grid()
    pdf = norm.pdf(grid - shift_ms, scale=sigma_ms)
    normalized = pdf / np.sqrt(np.sum(pdf**2))
    assert np.isclose(np.sum(normalized**2), 1.0)
    return normalized


def gaussian_distance(sigma_ms: float, shift_ms: float) -> float:
    k1 = gaussian_kernel(sigma_ms)
    k2 = gaussian_kernel(sigma_ms, shift_ms)
    return float(np.sum((k1 - k2) ** 2))


def _causal_pdf(tau_ms: float, shift_ms: float = 0.0) -> np.ndarray:
    grid = _time_grid()
    center_idx = len(grid) // 2
    pdf = expon.pdf(grid[center_idx:] - shift_ms, scale=tau_ms)
    normalized = pdf / np.sqrt(np.sum(pdf**2))
    full_kernel = np.concatenate([np.zeros(center_idx), normalized])
    assert np.isclose(np.sum(full_kernel**2), 1.0)
    return full_kernel


def causal_exponential_distance(tau_ms: float, shift_ms: float) -> float:
    k1 = _causal_pdf(tau_ms)
    k2 = _causal_pdf(tau_ms, shift_ms)
    return float(np.sum((k1 - k2) ** 2))
