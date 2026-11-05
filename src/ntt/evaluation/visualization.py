"""Visualization utilities for evaluation results."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .aggregation.comparison import MetricComparison, _to_scalar, extract_normalized_scores
from .core.types import EvaluationReport, MetricCategory, MetricDirection, MetricResult


def plot_radar(
    reports: dict[str, EvaluationReport],
    figsize: tuple[float, float] = (8, 8),
) -> Figure:
    """
    Create radar plot comparing models across categories.

    Args:
        reports: Dict mapping model names to evaluation reports
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    categories = list(MetricCategory)
    n_categories = len(categories)

    # Compute angles for radar plot
    angles = np.linspace(0, 2 * np.pi, n_categories, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"polar": True})

    for model_name, report in reports.items():
        if report.category_scores is None:
            continue

        values = [report.category_scores.get(cat, 0) for cat in categories]
        values += values[:1]  # Complete the loop

        ax.plot(angles, values, "o-", label=model_name, linewidth=2)
        ax.fill(angles, values, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([cat.value for cat in categories])
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

    plt.title("Model Comparison by Category")
    plt.tight_layout()

    return fig


def plot_metric_comparison(
    reports: dict[str, EvaluationReport],
    metric_name: str,
    figsize: tuple[float, float] = (10, 6),
) -> Figure:
    """
    Create box plot comparing a specific metric across models.

    Args:
        reports: Dict mapping model names to evaluation reports
        metric_name: Name of metric to compare
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    data = []
    labels = []

    for model_name, report in reports.items():
        if metric_name not in report.results:
            continue

        result = report.results[metric_name]
        values = result.raw_value

        if not np.isscalar(values):
            values = np.asarray(values)
            values = values[~np.isnan(values)]
            if len(values) > 0:
                data.append(values)
                labels.append(model_name)

    if data:
        # Stagger x tick labels to avoid overlap when many models.
        stair_labels = [("\n" * (i % 2)) + lab for i, lab in enumerate(labels)]
        ax.boxplot(data, tick_labels=stair_labels)
        ax.set_ylabel(metric_name)
        ax.set_title(f"Comparison: {metric_name}")

    plt.tight_layout()
    return fig


def plot_normalized_comparison_radar(
    comparisons: dict[str, MetricComparison],
    model_name: str = "Model",
    baseline_name: str = "Poisson",
    ceiling_name: str = "Perfect reconstruction",
    figsize: tuple[float, float] = (8, 8),
    clip_range: tuple[float, float] = (-0.5, 1.2),
) -> Figure:
    """Create radar plot of normalized scores relative to a floor and ceiling.

    The normalized score uses ``0`` for the baseline and ``1`` for the ceiling.
    Values below zero are worse than the baseline; values above one exceed the
    empirical ceiling. Values are clipped for display only.

    Args:
        comparisons: Metric name -> normalized comparison metadata.
        model_name: Label for the model.
        baseline_name: Label for the baseline/floor.
        ceiling_name: Label for the ceiling.
        figsize: Figure size.
        clip_range: Display-only clipping range for normalized scores.

    Returns:
        matplotlib Figure.
    """
    scores = extract_normalized_scores(comparisons)
    names = sorted(
        scores,
        key=lambda name: (comparisons[name].category.value, name),
    )
    values = [scores[name] for name in names]
    n = len(names)

    if n == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No metrics to plot", ha="center", va="center")
        return fig

    lower_clip, upper_clip = clip_range
    clipped_values = [float(np.clip(value, lower_clip, upper_clip)) for value in values]
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    radius_values = [value - lower_clip for value in clipped_values]
    values_closed = radius_values + radius_values[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"polar": True})

    baseline_radius = 0.0 - lower_clip
    ceiling_radius = 1.0 - lower_clip
    baseline_ring = [baseline_radius] * (n + 1)
    ceiling_ring = [ceiling_radius] * (n + 1)
    ax.plot(angles, baseline_ring, "k--", linewidth=1, alpha=0.5, label=baseline_name)
    ax.plot(angles, ceiling_ring, "k:", linewidth=1, alpha=0.6, label=ceiling_name)

    ax.plot(angles, values_closed, "o-", linewidth=2, label=model_name)
    ax.fill(angles, values_closed, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(names, size=8)

    ax.set_ylim(0, upper_clip - lower_clip)
    tick_values = [
        tick
        for tick in [lower_clip, 0.0, 0.5, 1.0, upper_clip]
        if lower_clip <= tick <= upper_clip
    ]
    tick_positions = [tick - lower_clip for tick in tick_values]
    tick_labels = [f"{tick:.1f}" for tick in tick_values]
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, size=8)
    ax.set_title(
        f"{model_name} vs {baseline_name}\n(0 = Poisson floor, 1 = perfect reconstruction)",
        pad=20,
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.05))
    plt.tight_layout()
    return fig


def plot_metric_dumbbell(
    model_report: EvaluationReport,
    baseline_report: EvaluationReport,
    model_name: str = "Model",
    baseline_name: str = "Poisson",
    figsize: tuple[float, float] = (12, 8),
) -> Figure:
    """Create a raw-value dumbbell plot for model vs baseline metrics."""
    metric_names = sorted(
        set(model_report.results) & set(baseline_report.results),
        key=lambda name: (model_report.results[name].category.value, name),
    )

    if not metric_names:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No metrics to plot", ha="center", va="center")
        return fig

    model_values = []
    baseline_values = []
    y_labels = []

    for metric_name in metric_names:
        model_result = model_report.results[metric_name]
        baseline_result = baseline_report.results[metric_name]
        model_values.append(_to_scalar(model_result.raw_value))
        baseline_values.append(_to_scalar(baseline_result.raw_value))
        direction_label = _direction_label(model_result.direction)
        y_labels.append(f"{metric_name} [{direction_label}]")

    y_positions = np.arange(len(metric_names))
    fig, ax = plt.subplots(figsize=figsize)

    for idx, (baseline_value, model_value) in enumerate(
        zip(baseline_values, model_values, strict=False)
    ):
        ax.plot(
            [baseline_value, model_value],
            [idx, idx],
            color="0.7",
            linewidth=2,
            zorder=1,
        )

    ax.scatter(baseline_values, y_positions, label=baseline_name, s=45, zorder=2)
    ax.scatter(model_values, y_positions, label=model_name, s=45, zorder=3)
    ax.axvline(0.0, color="0.85", linewidth=1, linestyle="--")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel("Raw metric value")
    ax.set_title(f"Raw Metric Comparison: {model_name} vs {baseline_name}")
    ax.legend()
    ax.invert_yaxis()

    # Clip x-axis when outliers crush the chart.
    all_vals = [abs(v) for v in model_values + baseline_values if not np.isnan(v) and v != 0]
    if all_vals:
        p90 = float(np.percentile(all_vals, 90))
        if max(all_vals) > p90 * 3:
            x_limit = p90 * 1.5
            ax.set_xlim(-x_limit * 0.1, x_limit)
            for idx, (bv, mv) in enumerate(zip(baseline_values, model_values, strict=False)):
                for v in (bv, mv):
                    if abs(v) > x_limit:
                        ax.annotate(
                            f"{v:.1f}",
                            xy=(x_limit * 0.97, idx),
                            fontsize=6,
                            ha="right",
                            va="center",
                            fontweight="bold",
                        )

    plt.tight_layout()
    return fig


def plot_metric_bars(
    reports: dict[str, EvaluationReport],
    figsize: tuple[float, float] = (12, 6),
) -> Figure:
    """Grouped horizontal bar chart comparing raw metric values across models.

    Each metric gets a group of bars (one per model). Metrics are sorted by
    category for visual grouping.

    Args:
        reports: Model name → EvaluationReport.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    model_names = list(reports.keys())
    if not model_names:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No reports to plot", ha="center", va="center")
        return fig

    # Collect all metric names present in any report, sorted by category
    all_metrics: list[str] = []
    seen: set[str] = set()
    for report in reports.values():
        for name, _result in sorted(report.results.items(), key=lambda x: x[1].category.value):
            if name not in seen:
                all_metrics.append(name)
                seen.add(name)

    n_metrics = len(all_metrics)
    n_models = len(model_names)
    bar_height = 0.8 / n_models
    y_positions = np.arange(n_metrics)

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]

    for i, model_name in enumerate(model_names):
        report = reports[model_name]
        values = []
        for metric_name in all_metrics:
            if metric_name in report.results:
                v = report.results[metric_name].raw_value
                if isinstance(v, np.ndarray):
                    v = float(np.nanmean(v))
                values.append(float(v))
            else:
                values.append(float("nan"))

        offset = (i - n_models / 2 + 0.5) * bar_height
        ax.barh(
            y_positions + offset,
            values,
            height=bar_height,
            label=model_name,
            color=colors[i % len(colors)],
            alpha=0.8,
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(all_metrics, fontsize=8)
    ax.set_xlabel("Metric Value")
    ax.set_title("Raw Metric Comparison")
    ax.legend()
    ax.invert_yaxis()

    # Clip x-axis when outliers crush the chart.  Use the 90th percentile
    # of all non-zero absolute values × 1.5 as the upper bound, and
    # annotate any bars that exceed it with their actual value.
    all_values = []
    for report in reports.values():
        for name in all_metrics:
            if name in report.results:
                v = report.results[name].raw_value
                if isinstance(v, np.ndarray):
                    v = float(np.nanmean(v))
                all_values.append(abs(float(v)))
    nonzero_vals = [v for v in all_values if v > 0]
    if nonzero_vals:
        p90 = float(np.percentile(nonzero_vals, 90))
        max_val = max(nonzero_vals)
        if max_val > p90 * 3:
            x_limit = p90 * 1.5
            ax.set_xlim(right=x_limit)
            # Annotate clipped bars
            for i, model_name in enumerate(model_names):
                report = reports[model_name]
                for j, metric_name in enumerate(all_metrics):
                    if metric_name not in report.results:
                        continue
                    v = report.results[metric_name].raw_value
                    if isinstance(v, np.ndarray):
                        v = float(np.nanmean(v))
                    v = float(v)
                    if abs(v) > x_limit:
                        offset = (i - n_models / 2 + 0.5) * bar_height
                        ax.text(
                            x_limit * 0.98,
                            y_positions[j] + offset,
                            f" {v:.1f} →",
                            va="center",
                            ha="right",
                            fontsize=7,
                            fontweight="bold",
                            color="white",
                        )

    plt.tight_layout()
    return fig


def _direction_label(direction: MetricDirection) -> str:
    if direction == MetricDirection.HIGHER_IS_BETTER:
        return "higher"
    return "lower"


def plot_structure_preservation(
    result: MetricResult,
    figsize: tuple[float, float] = (14, 5),
    title: str | None = None,
) -> Figure:
    """3-panel plot for structure preservation metric results.

    Panels: Input RDM, Latent RDM, k-NN preservation curve.

    Args:
        result: MetricResult from the StructurePreservation metric.
            Must have ``metadata`` containing ``rdm_input``, ``rdm_latent``,
            and ``knn_overlap``.
        figsize: Figure size.
        title: Optional suptitle.  Defaults to summary of RSA and CKA scores.

    Returns:
        matplotlib Figure.
    """
    meta = result.metadata
    rdm_input = meta.get("rdm_input")
    rdm_latent = meta.get("rdm_latent")
    knn_overlap = meta.get("knn_overlap", {})

    if rdm_input is None or rdm_latent is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No structure preservation data available", ha="center", va="center")
        return fig

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    if title is None:
        rsa_r = result.raw_value
        n_samples = meta.get("n_samples", "?")
        knn_mean = meta.get("knn_mean", float("nan"))
        title = (
            f"Population Structure Preservation (N={n_samples})\n"
            f"RSA r={rsa_r:.3f}  |  Mean k-NN overlap={knn_mean:.3f}"
        )
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.05)

    # Panel 1: Input RDM — each entry = distance between two population snapshots
    ax = axes[0]
    im1 = ax.imshow(rdm_input, cmap="viridis", aspect="auto")
    input_metric = meta.get("input_metric", "")
    ax.set_title(f"Population Input RDM ({input_metric})", fontsize=10)
    ax.set_xlabel("Time window")
    ax.set_ylabel("Time window")
    plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

    # Panel 2: Latent RDM — each entry = distance between two latent vectors
    ax = axes[1]
    im2 = ax.imshow(rdm_latent, cmap="viridis", aspect="auto")
    latent_metric = meta.get("latent_metric", "")
    ax.set_title(f"Population Latent RDM ({latent_metric})", fontsize=10)
    ax.set_xlabel("Time window")
    ax.set_ylabel("Time window")
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

    # Panel 3: k-NN preservation curve
    ax = axes[2]
    if knn_overlap:
        # Keys may be strings after JSON round-trip; convert to int
        knn_overlap = {int(k): v for k, v in knn_overlap.items()}
        ks = sorted(knn_overlap.keys())
        overlaps = [knn_overlap[k] for k in ks]
        ax.plot(ks, overlaps, "o-", color="#4CAF50", linewidth=2, markersize=5, label="Observed")

        # Random baseline: expected overlap of two random k-subsets from N-1 items
        N = rdm_input.shape[0]
        random_baseline = [k / (N - 1) for k in ks]
        ax.plot(
            ks,
            random_baseline,
            "--",
            color="gray",
            linewidth=1,
            alpha=0.7,
            label="Random baseline",
        )
        ax.fill_between(ks, random_baseline, overlaps, alpha=0.15, color="#4CAF50")

    ax.set_xlabel("Neighbourhood size k")
    ax.set_ylabel("Mean k-NN overlap")
    ax.set_title("Neighbourhood preservation", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_per_unit_structure_preservation(
    result: MetricResult,
    figsize: tuple[float, float] = (14, 5),
    title: str | None = None,
) -> Figure:
    """3-panel plot for inter-unit structure preservation results.

    Shows U×U RDMs comparing units (not time windows).  Each RDM entry (i,j)
    represents the distance between unit i and unit j based on their temporal
    activity patterns across all windows.

    Args:
        result: MetricResult from the PerUnitStructurePreservation metric.
            ``raw_value`` is the RSA score; ``metadata`` should contain
            ``rdm_input`` (U×U), ``rdm_latent`` (U×U), and ``knn_overlap``.
        figsize: Figure size.
        title: Optional suptitle.

    Returns:
        matplotlib Figure.
    """
    meta = result.metadata
    rdm_input = meta.get("rdm_input")
    rdm_latent = meta.get("rdm_latent")
    knn_overlap = meta.get("knn_overlap", {})

    if rdm_input is None or rdm_latent is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No per-unit structure preservation data", ha="center", va="center")
        return fig

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    if title is None:
        rsa_r = result.raw_value
        n_samples = meta.get("n_samples", "?")
        n_units = meta.get("n_units", "?")
        knn_mean = meta.get("knn_mean", float("nan"))
        title = (
            f"Inter-Unit Structure Preservation (N={n_samples} windows, U={n_units} units)\n"
            f"RSA r={rsa_r:.3f}  |  Mean k-NN overlap={knn_mean:.3f}"
        )
    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.05)

    # Panel 1: Input RDM — U×U matrix comparing units based on original activity
    ax = axes[0]
    im1 = ax.imshow(rdm_input, cmap="viridis", aspect="auto")
    input_metric = meta.get("input_metric", "")
    ax.set_title(f"Input RDM ({input_metric})", fontsize=10)
    ax.set_xlabel("Unit index")
    ax.set_ylabel("Unit index")
    plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)

    # Panel 2: Latent RDM — U×U matrix comparing units based on latent codes
    ax = axes[1]
    im2 = ax.imshow(rdm_latent, cmap="viridis", aspect="auto")
    latent_metric = meta.get("latent_metric", "")
    ax.set_title(f"Latent RDM ({latent_metric})", fontsize=10)
    ax.set_xlabel("Unit index")
    ax.set_ylabel("Unit index")
    plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)

    # Panel 3: k-NN preservation curve
    ax = axes[2]
    if knn_overlap:
        # Keys may be strings after JSON round-trip; convert to int
        knn_overlap = {int(k): v for k, v in knn_overlap.items()}
        ks = sorted(knn_overlap.keys())
        overlaps = [knn_overlap[k] for k in ks]
        ax.plot(ks, overlaps, "o-", color="#4CAF50", linewidth=2, markersize=5, label="Observed")

        # Random baseline: expected overlap of two random k-subsets from U-1 items
        U = rdm_input.shape[0]
        random_baseline = [k / (U - 1) for k in ks]
        ax.plot(
            ks,
            random_baseline,
            "--",
            color="gray",
            linewidth=1,
            alpha=0.7,
            label="Random baseline",
        )
        ax.fill_between(ks, random_baseline, overlaps, alpha=0.15, color="#4CAF50")

    ax.set_xlabel("Neighbourhood size k")
    ax.set_ylabel("Mean k-NN overlap")
    ax.set_title("Neighbourhood preservation", fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_repeats_snr(
    result: MetricResult,
    figsize: tuple[float, float] = (18, 7),
    title: str | None = None,
) -> Figure:
    """6-panel default plot for the RepeatsSNR metric.

    Layout (two rows):

    Row 1 — four time x latent-dim 2D heatmaps:
      1. Per-dim debiased SNR = ``sigma2_sig_debiased / (sigma2_noise + eps)``.
      2. Per-dim raw (biased) SNR = ``sigma2_sig_raw / (sigma2_noise + eps)``.
      3. ``σ²_noise`` per dim over time.
      4. ``σ²_sig_debiased`` per dim over time.

    Row 2 — two 1D aggregate plots (each spanning 2 of the 4 top-row columns):
      5. Overall debiased SNR vs time — trace ratio across dims per time bin
         (this reduces to ``result.raw_value`` when summed across time, not
         averaged across dims).
      6. Sorted per-dim overall SNR (dim ranking).

    When ``result.metadata['token_bin_size_ms']`` is present, time axes are
    labelled in milliseconds; otherwise in token index.
    """
    meta = result.metadata
    T, L = meta["token_shape"]
    sig_deb = np.asarray(meta["sigma2_sig_debiased"]).reshape(T, L)
    noise = np.asarray(meta["sigma2_noise"]).reshape(T, L)
    token_bin_ms = meta.get("token_bin_size_ms")

    # Use a tiny epsilon to avoid division-by-zero when a time-bin or dim has
    # exactly zero noise (degenerate synthetic data).
    eps = 1e-12
    # Per-dim SNR comes from the metric metadata directly; just reshape.
    per_dim_snr_deb_tl = np.asarray(meta["snr_per_dim_debiased"]).reshape(T, L)
    per_dim_snr_raw_tl = np.asarray(meta["snr_per_dim_raw"]).reshape(T, L)

    overall_snr_by_time = sig_deb.sum(axis=1) / (noise.sum(axis=1) + eps)
    per_dim_snr = sig_deb.sum(axis=0) / (noise.sum(axis=0) + eps)
    per_dim_snr_sorted = np.sort(per_dim_snr)[::-1]

    if token_bin_ms is not None:
        time_extent_upper = T * float(token_bin_ms)
        time_centers = (np.arange(T) + 0.5) * float(token_bin_ms)
        time_ylabel = "time since stim onset (ms)"
    else:
        time_extent_upper = float(T)
        time_centers = np.arange(T) + 0.5
        time_ylabel = "token index"

    # 4 columns on top row; bottom row: 2 panels each spanning 2 columns.
    fig, axd = plt.subplot_mosaic(
        [
            ["h1", "h2", "h3", "h4"],
            ["b1", "b1", "b2", "b2"],
        ],
        figsize=figsize,
    )

    heatmap_specs = [
        ("h1", per_dim_snr_deb_tl, "per-dim SNR (debiased)"),
        ("h2", per_dim_snr_raw_tl, "per-dim SNR (raw / biased)"),
        ("h3", noise, "σ²_noise"),
        ("h4", sig_deb, "σ²_sig (debiased)"),
    ]
    for key, data, panel_title in heatmap_specs:
        ax = axd[key]
        im = ax.imshow(
            data,
            aspect="auto",
            origin="lower",
            extent=(-0.5, L - 0.5, 0.0, float(time_extent_upper)),
        )
        ax.set_xlabel("latent dim")
        ax.set_ylabel(time_ylabel)
        ax.set_title(panel_title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Panel 5: Overall SNR vs time (trace-ratio per time bin)
    ax = axd["b1"]
    ax.plot(time_centers, overall_snr_by_time, marker="o", ms=3, lw=1.2)
    ax.set_xlabel(
        time_ylabel.replace("time since stim onset", "time since stim onset (token center)")
    )
    ax.set_ylabel("overall SNR (trace ratio)")
    ax.set_title("SNR vs time", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 6: sorted per-dim SNR (dim ranking)
    ax = axd["b2"]
    ax.plot(np.arange(L), per_dim_snr_sorted, lw=1.2)
    ax.set_xlabel("latent dim (sorted by SNR, descending)")
    ax.set_ylabel("overall SNR")
    ax.set_title("per-dim SNR ranking", fontsize=10)
    ax.grid(True, alpha=0.3)

    if title is None:
        snr_overall = result.raw_value
        n_stim = meta.get("n_stimuli", "?")
        n_rep = meta.get("n_repeats_mean_arithmetic", float("nan"))
        title = (
            f"Repeats SNR  (overall debiased = {snr_overall:.3f}, "
            f"n_stim = {n_stim}, n_rep_mean = {n_rep:.1f})"
        )
    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.02)

    plt.tight_layout()
    return fig


def print_summary_table(reports: dict[str, EvaluationReport]) -> None:
    """Print a summary table of category scores."""
    categories = list(MetricCategory)

    # Header
    header = "Model".ljust(20) + "".join(cat.value[:12].ljust(14) for cat in categories)
    print(header)
    print("-" * len(header))

    # Rows
    for model_name, report in reports.items():
        if report.category_scores is None:
            continue

        row = model_name[:20].ljust(20)
        for cat in categories:
            score = report.category_scores.get(cat, np.nan)
            if np.isnan(score):
                row += "N/A".ljust(14)
            else:
                row += f"{score:.3f}".ljust(14)
        print(row)
