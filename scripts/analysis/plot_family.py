#!/usr/bin/env python3
"""Strip/dot plot: tokenizer family on x-axis, metric value on y-axis.

One point per tokenizer (for which the metric is defined), jittered
horizontally within each family group.

Usage::

    uv run python -m scripts.analysis.plot_family \\
        /path/to/tracker_with_agg.csv \\
        single_neuron_recon \\
        --output plots/family_sn_recon.pdf

    # per-metric paths also work:
    uv run python -m scripts.analysis.plot_family \\
        tracker.csv "sn_recon/spike_f1" --labels

    # overlay median bar and hide legend:
    uv run python -m scripts.analysis.plot_family \\
        tracker.csv decoding --no-legend
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scripts.analysis.data import (
    extract_metric,
    family_color,
    load_scores,
    metric_label,
)

_RNG = np.random.default_rng(42)


def plot_family(
    df,
    metric_path: str,
    output: Path | None = None,
    show_median: bool = True,
    show_legend: bool = True,
    jitter: float = 0.15,
    figsize: tuple[float, float] | None = None,
    label_points: bool = False,
) -> plt.Figure:
    """Create a strip plot grouped by tokenizer family.

    Parameters
    ----------
    df:
        DataFrame from :func:`scripts.analysis.data.load_scores`.
    metric_path:
        Metric path (composite name, alias, or ``prefix/metric_name``).
    output:
        Save path.  If None, figure is returned without saving.
    show_median:
        Overlay a horizontal bar at the per-family median.
    show_legend:
        Show family colour legend.
    jitter:
        Horizontal jitter width within each family group.
    """
    # Group data by family.
    family_data: dict[str, list[tuple[float, str]]] = {}
    for _, row in df.iterrows():
        v = extract_metric(row["_scores_data"], metric_path)
        if v is None:
            continue
        fam = row["family"]
        family_data.setdefault(fam, []).append((v, row["tokenizer"]))

    if not family_data:
        raise ValueError(f"No tokenizers have metric '{metric_path}' defined.")

    # Sort families: primarily by median (descending), then alphabetically.
    families_sorted = sorted(
        family_data.keys(),
        key=lambda f: (-np.median([x for x, _ in family_data[f]]), f),
    )

    n = len(families_sorted)
    width = max(8, n * 0.9)
    fig_size = figsize or (width, 5.5)
    fig, ax = plt.subplots(figsize=fig_size)

    seen_colors: dict[str, str] = {}
    tick_positions = list(range(n))

    for i, fam in enumerate(families_sorted):
        vals_names = family_data[fam]
        vals = [v for v, _ in vals_names]
        names = [nm for _, nm in vals_names]
        col = family_color(fam, seen_colors)

        jx = _RNG.uniform(-jitter, jitter, len(vals))
        ax.scatter(
            np.full(len(vals), i) + jx,
            vals,
            c=col,
            s=55,
            zorder=3,
            edgecolors="white",
            linewidths=0.4,
            alpha=0.85,
            label=fam,
        )
        if label_points:
            for xi, yi, nm in zip(np.full(len(vals), i) + jx, vals, names, strict=True):
                ax.annotate(
                    nm[:25],
                    (xi, yi),
                    fontsize=4.5,
                    xytext=(3, 2),
                    textcoords="offset points",
                    color="dimgray",
                    clip_on=True,
                )

        if show_median and len(vals) >= 1:
            med = float(np.median(vals))
            ax.hlines(med, i - 0.35, i + 0.35, colors=col, linewidths=2.0, zorder=4)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        families_sorted,
        rotation=35,
        ha="right",
        fontsize=8,
    )
    ax.set_ylabel(metric_label(metric_path), fontsize=10)
    ax.set_title(
        f"{metric_label(metric_path)}  by tokenizer family",
        fontsize=11,
    )
    ax.grid(axis="y", alpha=0.3, lw=0.5)
    ax.set_xlim(-0.6, n - 0.4)

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        # Deduplicate legend entries (scatter may have added duplicates).
        seen_l: set[str] = set()
        h_dedup, l_dedup = [], []
        for h, lbl in zip(handles, labels, strict=True):
            if lbl not in seen_l:
                h_dedup.append(h)
                l_dedup.append(lbl)
                seen_l.add(lbl)
        ax.legend(
            h_dedup,
            l_dedup,
            title="Family",
            fontsize=7,
            title_fontsize=7.5,
            loc="lower right",
            framealpha=0.8,
            ncol=max(1, len(h_dedup) // 10),
        )

    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved: {output}")

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strip plot of a single NTT metric grouped by tokenizer family.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "tracker_csv",
        help="Tracker CSV; needs an mlflow_aggregation column unless --scores-dir is given",
    )
    parser.add_argument("metric", help="Metric path (e.g. decoding or sn_recon/spike_f1)")
    parser.add_argument(
        "--output", "-o", help="Output path (.pdf/.png/.svg). Shows interactive plot if omitted."
    )
    parser.add_argument("--no-median", action="store_true", help="Skip per-family median bar")
    parser.add_argument("--no-legend", action="store_true", help="Hide family colour legend")
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
    parser.add_argument("--mlflow-uri", default=None)
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
    plot_family(
        df,
        metric_path=args.metric,
        output=out,
        show_median=not args.no_median,
        show_legend=not args.no_legend,
        label_points=args.labels,
    )
    if out is None:
        plt.show()


if __name__ == "__main__":
    main()
