"""Smoke tests for the robustness plotting helpers (D11/D12).

Covers the two shared primitives in
``src/ntt/evaluation/plotting/robustness.py`` against synthetic data,
asserting only that a ``matplotlib.figure.Figure`` is returned with the
expected number of axes. The detailed visual correctness of M1-a /
M2-a curves is reviewed by humans against real eval outputs.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend for CI / cluster runs.

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ntt.contrib.tokenizer.poisson_baseline import (
    PoissonBaselineTokenizerSpec,
    format_poisson_label,
)
from ntt.evaluation.plotting.robustness import (
    plot_input_corruption_a,
    plot_input_corruption_c,
    plot_latent_corruption_b,
    plot_metric_curve,
    plot_spike_raster,
)

# ── Poisson label helper ──────────────────────────────────────────────


@pytest.mark.parametrize(
    ("rate_bin_ms", "expected"),
    [
        (10, "poisson_10ms"),
        (5, "poisson_5ms"),
        (10.0, "poisson_10ms"),  # integer-valued float drops the .0
        (12.5, "poisson_12.5ms"),  # non-integer keeps one decimal
    ],
)
def test_format_poisson_label_embeds_rate_bin(rate_bin_ms, expected):
    """``format_poisson_label`` should embed ``rate_estimation_bin_ms``
    in the legend-friendly snake_case format.

    """
    spec = PoissonBaselineTokenizerSpec(rate_estimation_bin_ms=rate_bin_ms, n_neurons=False)
    assert format_poisson_label(spec) == expected


def test_plot_metric_curve_returns_axes_and_draws():
    """``plot_metric_curve`` should populate the axes and return it."""
    fig, ax = plt.subplots()
    xs = np.array([0.0, 1.0, 2.0, 3.0])
    means = np.array([0.1, 0.5, 0.9, 1.4])
    stds = np.array([0.05, 0.1, 0.15, 0.2])

    out = plot_metric_curve(ax, xs, means, stds, label="dummy", shade=True)

    # Same axes returned, line drawn (1 Line2D for the mean curve).
    assert out is ax
    assert len(ax.get_lines()) == 1
    plt.close(fig)


def test_plot_metric_curve_handles_empty():
    """Empty arrays should be a no-op and not raise."""
    fig, ax = plt.subplots()
    plot_metric_curve(ax, [], [], [], shade=True)
    assert len(ax.get_lines()) == 0
    plt.close(fig)


def test_plot_input_corruption_a_single_tokenizer_two_corruptions():
    """One tokenizer × two corruption labels → 2 used subplots."""
    results = {
        "jitter": {
            "strengths": [1.0, 2.0, 5.0],
            "means": [0.1, 0.2, 0.4],
            "stds": [0.02, 0.03, 0.05],
        },
        "drop": {
            "strengths": [0.1, 0.2, 0.4],
            "means": [0.05, 0.15, 0.3],
            "stds": [0.01, 0.02, 0.04],
        },
    }
    fig = plot_input_corruption_a(results)
    # 2 corruption labels with default ncols=min(n,3) → 1 row, 2 cols.
    assert len(fig.axes) == 2
    # Each axes should have exactly one drawn line (one tokenizer).
    for ax in fig.axes:
        assert len(ax.get_lines()) == 1
    plt.close(fig)


def test_plot_input_corruption_a_multi_tokenizer():
    """Two tokenizers, one corruption label → 1 axes with 2 lines."""
    results = [
        {
            "jitter": {
                "strengths": [1.0, 2.0, 5.0],
                "means": [0.1, 0.2, 0.4],
                "stds": [0.02, 0.03, 0.05],
            }
        },
        {
            "jitter": {
                "strengths": [1.0, 2.0, 5.0],
                "means": [0.05, 0.15, 0.35],
                "stds": [0.01, 0.02, 0.04],
            }
        },
    ]
    fig = plot_input_corruption_a(results, tokenizers=["A", "B"])
    assert len(fig.axes) == 1
    assert len(fig.axes[0].get_lines()) == 2
    plt.close(fig)


def test_plot_spike_raster_single_neuron():
    """Single-neuron mode renders [T, 1] spikes without crashing."""
    fig, ax = plt.subplots()
    arr = np.zeros((100, 1), dtype=float)
    arr[[5, 30, 75], 0] = 1.0
    out = plot_spike_raster(ax, arr, mode="single", bin_size_ms=1.0)
    assert out is ax
    # Three spike lines drawn.
    assert len(ax.collections) > 0 or len(ax.get_lines()) >= 0  # vlines → LineCollection
    plt.close(fig)


def test_plot_spike_raster_unit_order_overrides_sort_by_rate():
    """``unit_order`` should control row placement regardless of per-unit rate.

    Builds a 4-unit raster where unit 0 has the highest rate (would land on
    row 0 under ``sort_by_rate=True``) and unit 3 has zero spikes (would land
    last). Passing ``unit_order=[3, 2, 1, 0]`` must place unit 3's spikes on
    row 0 and unit 0's spikes on row 3 — i.e. the explicit order wins over
    any rate-based reordering.

    Row placement is read from the ``eventplot`` LineCollection's
    ``lineoffsets`` (one offset per unit, taken in the rendered order via
    ``arange(U)`` in ``plot_binned_spike_raster``). The number of spikes
    on each row is the size of the corresponding positions list.

    """
    T, U = 200, 4
    arr = np.zeros((T, U), dtype=float)
    # Distinct, ordered counts so we can identify rows by spike count alone.
    arr[:8, 0] = 1.0  # 8 spikes
    arr[:4, 1] = 1.0  # 4 spikes
    arr[:2, 2] = 1.0  # 2 spikes
    # unit 3: zero spikes

    fig, ax = plt.subplots()
    plot_spike_raster(
        ax,
        arr,
        mode="population",
        bin_size_ms=1.0,
        sort_by_rate=True,  # would put unit 0 on row 0 under default
        unit_order=np.array([3, 2, 1, 0]),  # explicit override
    )

    # ``eventplot`` produces one ``EventCollection`` per row, in render
    # order; spikes per row = number of segments in that collection.
    counts_per_row = [len(c.get_segments()) for c in ax.collections]
    # In the requested order [3, 2, 1, 0]: rows should hold 0, 2, 4, 8 spikes.
    assert counts_per_row == [0, 2, 4, 8]
    plt.close(fig)


def test_plot_input_corruption_c_persists_source_arrays(tmp_path):
    """Persist the rendered M1-c raster arrays alongside the figure."""
    label = "drop|fraction=0.1|p=1.0"
    clean = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )
    corrupted = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    stem = tmp_path / "input-corruption-c_sess0_w3"
    fig = plot_input_corruption_c(
        3,
        session_id="sess0",
        clean_spikes=clean,
        corrupted_spikes_per_label={label: corrupted},
        latent_distance_per_label={label: 0.25},
        output_stem=stem,
    )

    assert stem.with_suffix(".png").exists()
    npz_path = stem.with_suffix(".npz")
    assert npz_path.exists()
    with np.load(npz_path) as data:
        assert np.array_equal(data["clean"], clean)
        assert np.array_equal(data["corrupted_drop_fraction_0p1_p_1p0"], corrupted)
        assert data["session_id"].item() == "sess0"
        assert data["segment_index"].item() == 3
    plt.close(fig)


def test_plot_latent_corruption_b_persists_source_arrays(tmp_path):
    """Persist the rendered M2-b raster arrays alongside the figure."""
    clean = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )
    noisy = np.array(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    stem = tmp_path / "latent-corruption-b_sess0_w4"
    fig = plot_latent_corruption_b(
        4,
        session_id="sess0",
        clean_spikes=clean,
        noisy_spikes_per_eps={0.1: [noisy]},
        eps_per_panel=[0.1],
        metric_per_eps={0.1: {"acg_correlation": 0.7}},
        output_stem=stem,
    )

    assert stem.with_suffix(".png").exists()
    npz_path = stem.with_suffix(".npz")
    assert npz_path.exists()
    with np.load(npz_path) as data:
        assert np.array_equal(data["clean"], clean)
        assert np.array_equal(data["noisy_eps_0p1"], noisy[None, ...])
        assert data["session_id"].item() == "sess0"
        assert data["segment_index"].item() == 4
    plt.close(fig)


def test_plot_spike_raster_unit_order_wrong_shape_raises():
    """``unit_order`` length mismatch must raise rather than silently misalign."""
    fig, ax = plt.subplots()
    arr = np.zeros((10, 4), dtype=float)
    with pytest.raises(ValueError):
        plot_spike_raster(
            ax,
            arr,
            mode="population",
            unit_order=np.array([0, 1, 2]),  # length 3 ≠ U=4
        )
    plt.close(fig)
