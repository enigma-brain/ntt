"""Reusable target transform components for loss functions.

These are building blocks that any loss can use to preprocess targets
or compute sample weights before loss computation.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class GaussianSmoothTargets(nn.Module):
    """Convolve binary spike targets with a Gaussian kernel along time.

    Uses a 2D convolution with kernel size (K, 1) — temporal smoothing
    only, no mixing across neurons. This avoids reshaping the (B, T, N)
    layout.

    Produces soft targets in [0, 1] where bins near spikes get partial
    credit proportional to a Gaussian centered on each spike.

    Args:
        sigma: Standard deviation of the Gaussian kernel in bins.
    """

    kernel: Tensor

    def __init__(self, sigma: float):
        super().__init__()
        self.sigma = sigma
        # Build kernel: extend to ±3σ, at least 1 bin each side
        radius = max(1, math.ceil(3.0 * sigma))
        t = torch.arange(-radius, radius + 1, dtype=torch.float32)
        kernel = torch.exp(-0.5 * (t / sigma) ** 2)
        # Peak normalization: a single spike maps to 1.0 at its center
        kernel = kernel / kernel.max()
        # Shape for conv2d: (1, 1, K_time, K_neuron=1)
        self.register_buffer("kernel", kernel.view(1, 1, -1, 1))
        self.padding_t = radius

    def forward(self, target: Tensor) -> Tensor:
        """Apply Gaussian smoothing along the time dimension.

        Args:
            target: (B, T, N) binary spike tensor.

        Returns:
            (B, T, N) soft target tensor clamped to [0, 1].
        """
        # (B, T, N) -> (B, 1, T, N) for conv2d
        x = target.unsqueeze(1)
        x = torch.nn.functional.conv2d(x, self.kernel, padding=(self.padding_t, 0))
        # (B, 1, T, N) -> (B, T, N)
        return x.squeeze(1).clamp(0.0, 1.0)


class DistanceWeightedMask(nn.Module):
    """Compute per-element weights that reduce penalty near true spikes.

    For each position, computes the distance to the nearest spike.
    Within ``window_size`` bins of a spike, the weight is
    ``distance / window_size`` (linear decay). Outside the window,
    weight is 1.0 (full penalty).

    This means false predictions *near* a true spike are penalized
    less than those far away.

    Args:
        window_size: Number of bins on each side defining the
            reduced-penalty zone.
    """

    tri_kernel: Tensor

    def __init__(self, window_size: int):
        super().__init__()
        self.window_size = window_size
        # Pre-compute triangular kernel: [0/K, 1/K, ..., K/K, ..., 1/K, 0/K]
        # Shape (1, 1, 2K+1, 1) for conv2d along time only.
        K = window_size
        offsets = torch.arange(-K, K + 1, dtype=torch.float32)
        tri_kernel = 1.0 - offsets.abs() / K  # peaks at 1.0 center, 0 at edges
        self.register_buffer("tri_kernel", tri_kernel.view(1, 1, -1, 1))

    def forward(self, target: Tensor) -> Tensor:
        """Compute distance-weighted mask from spike targets.

        Args:
            target: (B, T, N) binary spike tensor.

        Returns:
            (B, T, N) weight tensor in [0, 1]. Values near spikes
            are < 1 (reduced penalty), values far from spikes are 1.
        """
        K = self.window_size

        # (B, T, N) -> (B, 1, T, N) for conv2d
        x = target.unsqueeze(1)

        # Convolve: high values = close to spike
        proximity = torch.nn.functional.conv2d(x, self.tri_kernel, padding=(K, 0))
        proximity = proximity.squeeze(1)  # (B, T, N)

        # Clamp proximity to [0, 1].
        # Note: for dense spike patterns where windows overlap, the convolution
        # sums contributions from multiple spikes, so proximity may exceed 1.0
        # before clamping. After clamping, positions between adjacent spikes get
        # maximum reduction (weight ≈ 0). This is acceptable for typical neural
        # data where spikes are sparse at 1ms resolution.
        proximity = proximity.clamp(0.0, 1.0)

        # Weight = 1 - proximity for non-spike locations,
        # but spike locations themselves get weight 1.0 (they are true positives)
        weights = 1.0 - proximity
        # Restore spike positions to weight 1.0
        weights = torch.where(target > 0, torch.ones_like(weights), weights)

        return weights
