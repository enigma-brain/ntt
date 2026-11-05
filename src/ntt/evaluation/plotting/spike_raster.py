"""Canonical ``plot_binned_spike_raster`` shared by notebooks and the
eval-time plotting helpers.

This module is the single source of truth for binned spike rasters.
``ntt.evaluation.plotting.robustness.plot_spike_raster`` delegates here so all
spike-raster output renders identically.

Rasters are drawn with ``eventplot`` (one collection per unit) so dense rasters
render quickly.
"""

from __future__ import annotations

import numpy as np


def plot_binned_spike_raster(
    ax,
    arr_tu,
    *,
    bin_size_ms: float = 1.0,
    spike_height: float = 0.9,
    linewidth: float = 0.8,
    color: str = "black",
):
    """Draw a binned-spike raster on ``ax``.

    One vertical tick per non-zero ``[t, u]`` bin, x in seconds, y in
    unit indices. Y-axis is inverted so unit 0 sits at the top.

    Args:
        ax: target ``matplotlib`` axes.
        arr_tu: ``[T, U]`` array of binned spike counts (>0 ⇒ tick).
        bin_size_ms: width of one time bin in milliseconds (x-axis is in
            seconds).
        spike_height: vertical extent of each tick (in unit-rows).
        linewidth: tick line width.
        color: tick color.
    """
    # arr_tu: [T, U]
    ax.set_facecolor("white")

    arr = np.asarray(arr_tu)
    if arr.ndim == 1:
        arr = arr[:, None]
    T, U = arr.shape

    # Build per-unit spike-time lists in seconds; render with eventplot
    # so each unit is one collection (much faster than per-spike vlines
    # on dense rasters).
    positions: list[np.ndarray] = []
    for u in range(U):
        t_idx = np.flatnonzero(arr[:, u] > 0)
        positions.append((t_idx + 0.5) * bin_size_ms / 1000.0)

    if any(p.size for p in positions):
        ax.eventplot(
            positions,
            lineoffsets=np.arange(U),
            linelengths=spike_height,
            linewidths=linewidth,
            colors=color,
        )

    ax.set_ylim(-0.5, U - 0.5)
    ax.set_xlim(0.0, T * bin_size_ms / 1000.0)
    ax.invert_yaxis()
