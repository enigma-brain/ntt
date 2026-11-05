#!/usr/bin/env python3
"""Radar (spider) plot of the 7 aggregate metric scores for one or more tokenizers.

Thin CLI wrapper: it loads the aggregated category scores, shapes them into the
tokenizer x category DataFrame that :func:`ntt.evaluation.radar.plot_radar`
already consumes, and delegates the drawing to that shared plotter. Axes that
are ``null`` for a tokenizer are shown at the floor (0); the legend notes how
many of the seven axes had data.

Usage::

    # single tokenizer:
    uv run python -m scripts.analysis.plot_radar \\
        tracker.csv wanx_A_vae_bce_dw_k1_noncausal

    # several tokenizers on one chart:
    uv run python -m scripts.analysis.plot_radar \\
        tracker.csv wanx_A_vae_bce_dw_k1_noncausal cosmos_tokenizer_DV4x8x8 \\
        --output plots/radar_comparison.pdf

    # fuzzy name matching (substring):
    uv run python -m scripts.analysis.plot_radar \\
        tracker.csv wanx_A_bsq cosmos --fuzzy --output plots/radar.pdf
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scripts.analysis.data import COMPOSITE_SCORES, SCORE_SHORT, load_scores

from ntt.evaluation.radar import plot_radar as plot_radar_frame


def scores_to_radar_frame(
    df: pd.DataFrame,
    tokenizer_names: list[str],
    score_clip: float = 1.0,
    compression_ceiling: float = 2.0,
) -> pd.DataFrame:
    """Shape :func:`scripts.analysis.data.load_scores` output into a radar frame.

    Returns a tokenizer x category DataFrame ready for
    :func:`ntt.evaluation.radar.plot_radar`: one row per tokenizer (indexed by
    ``"<name>  (n/7 axes)"``, with ``n`` = axes that had data), columns are the
    seven short axis labels. Missing axes are floored to ``0`` so each polygon
    stays closed; present values are clipped into ``[0, score_clip]``.
    Tokenizers absent from ``df`` are skipped with a warning.

    ``compression`` is a raw bit-ratio (input_bits / output_bits), not a [0,1]
    score. It is divided by ``compression_ceiling`` to put it on the same
    [0,1]-of-full-scale footing as its six peers before the shared
    ``score_clip`` applies; passing it straight through ``score_clip`` would
    pin every tokenizer at 1.0. The ceiling is named in the axis label, since
    the plotted fraction is meaningless without it. The value stored in
    ``category_scores.json`` is untouched and stays raw.

    The ``2.0`` default is sized to the observed range: across the published
    cohort ``compression_rate`` runs from below 1.0 (the latent is *larger* than
    the input) up to about 1.5, which the default spreads over
    roughly 0.20-0.75 of the axis. Set the ceiling from your own cohort's range
    — one sized for a different range flattens the axis, which is the failure
    this argument exists to prevent.
    """
    if not compression_ceiling > 0:
        raise ValueError(
            f"compression_ceiling must be positive, got {compression_ceiling!r} "
            "(it divides the raw bit-ratio to put it on the axis scale)"
        )
    axis_labels = [
        f"{SCORE_SHORT.get(s, s)} (max {compression_ceiling:g}x)"
        if s == "compression"
        else SCORE_SHORT.get(s, s)
        for s in COMPOSITE_SCORES
    ]
    rows: dict[str, dict[str, float]] = {}
    for name in tokenizer_names:
        match = df[df["tokenizer"] == name]
        if match.empty:
            print(f"  WARN: tokenizer '{name}' not found in loaded data — skipping")
            continue
        row = match.iloc[0]

        vals: list[float] = []
        n_present = 0
        for col in COMPOSITE_SCORES:
            v = row[col]
            if v is None or (isinstance(v, float) and math.isnan(v)):
                vals.append(0.0)
                continue
            if col == "compression":
                # Raw bit-ratio, not a [0,1] score — rescale by its own ceiling
                # first, then clip like everything else so the axis stays inside
                # the (0, score_clip) radial range handed to plot_radar.
                v = float(v) / compression_ceiling
            vals.append(max(0.0, min(float(v), score_clip)))
            n_present += 1

        short_name = name if len(name) <= 40 else name[:37] + "…"
        index = f"{short_name}  ({n_present}/{len(COMPOSITE_SCORES)} axes)"
        rows[index] = dict(zip(axis_labels, vals, strict=False))

    if not rows:
        raise ValueError("none of the requested tokenizers were found in the data")

    # from_dict(orient="index") preserves insertion order for rows; reindex
    # columns to lock the canonical axis order.
    return pd.DataFrame.from_dict(rows, orient="index")[axis_labels]


def _resolve_names(df, raw_names: list[str], fuzzy: bool) -> list[str]:
    all_names = set(df["tokenizer"].tolist())
    resolved: list[str] = []
    for raw in raw_names:
        if raw in all_names:
            resolved.append(raw)
            continue
        if fuzzy:
            matches = [n for n in all_names if raw.lower() in n.lower()]
            if not matches:
                print(f"  WARN: no tokenizer matching '{raw}' (fuzzy) — skipping")
            elif len(matches) == 1:
                resolved.append(matches[0])
                print(f"  fuzzy match '{raw}' → '{matches[0]}'")
            else:
                # Use the shortest (most specific) match.
                best = min(matches, key=len)
                resolved.append(best)
                print(f"  fuzzy match '{raw}' → '{best}' (from {len(matches)} candidates)")
        else:
            print(
                f"  WARN: tokenizer '{raw}' not found — "
                f"use --fuzzy for substring matching, or choose from:\n"
                + "\n".join(f"    {n}" for n in sorted(all_names))
            )
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Radar/spider plot of the 7 aggregate NTT metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "tracker_csv",
        help="Tracker CSV; needs an mlflow_aggregation column unless --scores-dir is given",
    )
    parser.add_argument("tokenizers", nargs="+", help="Tokenizer name(s) to plot")
    parser.add_argument(
        "--output", "-o", help="Output path (.pdf/.png/.svg). Shows interactive if omitted."
    )
    parser.add_argument("--fuzzy", action="store_true", help="Match tokenizer names by substring")
    parser.add_argument("--clip", type=float, default=1.0, help="Axis upper bound (default 1.0)")
    parser.add_argument(
        "--compression-ceiling",
        type=float,
        default=2.0,
        help=(
            "Bit-ratio mapped to a full compression axis (default 2.0, sized "
            "for the published cohort's compression range)"
        ),
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

    names = _resolve_names(df, args.tokenizers, args.fuzzy)
    if not names:
        print("No tokenizers to plot.")
        sys.exit(1)

    frame = scores_to_radar_frame(
        df, names, score_clip=args.clip, compression_ceiling=args.compression_ceiling
    )
    chart_for = names[0] if len(names) == 1 else f"{len(names)} tokenizers"
    fig = plot_radar_frame(
        frame,
        title=f"Aggregate metrics — {chart_for}",
        mode="normed",
        clip_range=(0.0, args.clip),
        floor_negative=False,
    )

    out = Path(args.output) if args.output else None
    if out:
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
