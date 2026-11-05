#!/usr/bin/env python3
"""Scatter/correlation plot for two NTT metrics across tokenizers.

One point per tokenizer (for which both metrics are defined), colored by
tokenizer family, with optional Pearson-r annotation.

Usage::

    uv run python -m scripts.analysis.plot_correlation \\
        /path/to/tracker_with_agg.csv \\
        single_neuron_recon decoding \\
        --output plots/sn_recon_vs_decoding.pdf

    # per-metric paths also work:
    uv run python -m scripts.analysis.plot_correlation \\
        tracker.csv \\
        "sn_recon/spike_f1" "population_structure/participation_ratio"

    # load from local JSON cache (avoids repeated MLflow downloads):
    uv run python -m scripts.analysis.plot_correlation \\
        tracker.csv sn_recon decoding \\
        --cache-dir /tmp/ntt_scores_cache
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scripts.analysis.data import (
    extract_metric,
    family_color,
    load_scores,
    metric_label,
)

# ── Main ──────────────────────────────────────────────────────────────────────


def plot_correlation(
    df: pd.DataFrame,
    x_path: str,
    y_path: str,
    output: Path | None = None,
    annotate: bool = True,
    figsize: tuple[float, float] = (7, 6),
    label_points: bool = False,
) -> plt.Figure:
    """Create correlation scatter plot.

    Parameters
    ----------
    df:
        DataFrame from :func:`scripts.analysis.data.load_scores`.
    x_path, y_path:
        Metric paths (composite name, alias, or ``prefix/metric_name``).
    output:
        Save path.  If None, the figure is returned without saving.
    annotate:
        If True, annotate with Pearson r and p-value.
    label_points:
        If True, label each point with a short tokenizer slug.
    """
    xs, ys, families, names = [], [], [], []
    for _, row in df.iterrows():
        x = extract_metric(row["_scores_data"], x_path)
        y = extract_metric(row["_scores_data"], y_path)
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)
        families.append(row["family"])
        names.append(row["tokenizer"])

    if not xs:
        raise ValueError(f"No tokenizers have both '{x_path}' and '{y_path}' defined.")

    fig, ax = plt.subplots(figsize=figsize)

    seen_colors: dict[str, str] = {}
    used_families: dict[str, str] = {}
    for fam in families:
        used_families[fam] = family_color(fam, seen_colors)

    for x, y, fam, name in zip(xs, ys, families, names, strict=True):
        col = used_families[fam]
        ax.scatter(x, y, c=col, s=60, zorder=3, edgecolors="white", linewidths=0.5, alpha=0.85)
        if label_points:
            short = name[:30]
            ax.annotate(
                short,
                (x, y),
                fontsize=5.5,
                xytext=(3, 3),
                textcoords="offset points",
                color="dimgray",
                clip_on=True,
            )

    # Pearson r
    if annotate and len(xs) >= 3:
        r, p = stats.pearsonr(xs, ys)
        p_str = f"p={p:.2e}" if p < 0.001 else f"p={p:.3f}"
        ax.text(
            0.03,
            0.97,
            f"r = {r:.3f}  ({p_str})  n={len(xs)}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            color="0.3",
            bbox={"fc": "white", "ec": "none", "alpha": 0.7},
        )

    # regression line
    if len(xs) >= 2:
        slope, intercept, *_ = stats.linregress(xs, ys)
        xlim = (min(xs), max(xs))
        x_line = np.linspace(xlim[0], xlim[1], 200)
        ax.plot(x_line, slope * x_line + intercept, lw=1.2, color="0.4", ls="--", zorder=2)

    # Legend
    for fam, col in sorted(used_families.items()):
        ax.scatter([], [], c=col, s=40, label=fam)
    ax.legend(
        title="Family",
        fontsize=7,
        title_fontsize=7.5,
        loc="lower right",
        framealpha=0.8,
        ncol=max(1, len(used_families) // 10),
    )

    ax.set_xlabel(metric_label(x_path), fontsize=10)
    ax.set_ylabel(metric_label(y_path), fontsize=10)
    ax.set_title(f"Correlation: {x_path}  vs  {y_path}", fontsize=11)
    ax.grid(True, alpha=0.3, lw=0.5)
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved: {output}")

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Correlation scatter plot for two NTT metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "tracker_csv",
        help="Tracker CSV; needs an mlflow_aggregation column unless --scores-dir is given",
    )
    parser.add_argument(
        "x_metric", help="X-axis metric path (e.g. single_neuron_recon or sn_recon/spike_f1)"
    )
    parser.add_argument("y_metric", help="Y-axis metric path")
    parser.add_argument(
        "--output", "-o", help="Output path (.pdf/.png/.svg). Shows interactive plot if omitted."
    )
    parser.add_argument("--no-annot", action="store_true", help="Skip Pearson-r annotation")
    parser.add_argument(
        "--labels", action="store_true", help="Label each point with tokenizer name"
    )
    parser.add_argument("--cache-dir", help="Cache dir for downloaded category_scores.json files")
    parser.add_argument(
        "--scores-dir",
        help="Read category_scores.json from <dir>/<slug>.json instead of MLflow "
        "(build_aggregation_tables.py writes this as <outdir>/scores)",
    )
    parser.add_argument(
        "--common-parts-only",
        action="store_true",
        help="warn instead of failing when it cannot be checked whether all tokenizers "
        "were averaged the same way (see build_aggregation_tables.py --common-parts-only)",
    )
    parser.add_argument(
        "--mlflow-uri",
        default=None,
        help="MLflow server to read every run from (default: MLFLOW_TRACKING_URI)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    load_kwargs: dict = {
        "tracker_csv": args.tracker_csv,
        "cache_dir": args.cache_dir,
        "verbose": args.verbose,
    }
    if args.mlflow_uri:
        load_kwargs["mlflow_uri"] = args.mlflow_uri
    if args.scores_dir:
        load_kwargs["scores_dir"] = args.scores_dir
    load_kwargs["common_parts_only"] = args.common_parts_only

    print("Loading scores …")
    df = load_scores(**load_kwargs)
    print(f"  {len(df)} tokenizers loaded")

    out = Path(args.output) if args.output else None
    plot_correlation(
        df,
        x_path=args.x_metric,
        y_path=args.y_metric,
        output=out,
        annotate=not args.no_annot,
        label_points=args.labels,
    )
    if out is None:
        plt.show()


if __name__ == "__main__":
    main()
