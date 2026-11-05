#!/usr/bin/env python3
"""Per-tier and per-metric generalization tables from recon ``results.json`` files.

Retention per metric and tier comes from
``ntt.evaluation.generalization.compute_from_per_session_reports``, called exactly as
``NTTAggregationJob._backfill_generalization`` calls it (``src/ntt/jobs/aggregation.py``),
so it covers the reconstruction metrics only. A tokenizer's tiers are compared over the
same metrics (those with a retention in all three tiers).

Each row also carries ``generalization_score``, the composite ``generalization`` score
of that same call. ``--check-scores`` fails (exit 1) unless it reproduces a
``category_scores.csv`` for every tokenizer of either table.

Writes ``generalization_by_tier.csv`` (one row per tokenizer) and
``generalization_by_metric.csv`` (one row per tokenizer x metric), the inputs of the
dashboard's Generalization tab. Reads files only (no MLflow).

    uv run python scripts/analysis/build_generalization_tables.py \\
        --inputs inputs.csv --outdir out/ --gen-split test \\
        --check-scores aggregation/category_scores.csv

``--inputs`` is a CSV with columns ``tokenizer,results_json``.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from ntt.evaluation.generalization import (
    DEFAULT_OOD_TIERS,
    DEFAULT_RETENTION_FLOOR,
    build_tier_index,
    compute_from_per_session_reports,
)

TIER_COLUMNS: list[str] = [
    "tokenizer",
    *[f"gen_tier/{t}" for t in DEFAULT_OOD_TIERS],
    *[f"gen_tier/{t}::mean" for t in DEFAULT_OOD_TIERS],
    *[f"gen_tier/{t}::n" for t in DEFAULT_OOD_TIERS],
    "generalization_score",
]
METRIC_COLUMNS: list[str] = [
    "tokenizer",
    "metric",
    "category",
    *[f"ret_{t}" for t in DEFAULT_OOD_TIERS],
]


def _geomean(values: list[float]) -> float:
    """Geometric mean with each value floored, as the generalization score does."""
    floored = [max(v, DEFAULT_RETENTION_FLOOR) for v in values]
    return math.exp(statistics.fmean(math.log(v) for v in floored))


def tokenizer_rows(
    tokenizer: str,
    recon: dict[str, Any],
    *,
    explicit_tiers: dict[tuple[str, str], str],
    xtime_excludes: dict[str, set[str]],
    baseline_tier: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The tokenizer's tier row (values absent when not computable) and per-metric rows."""
    per_session = recon.get("per_session") or {}
    row: dict[str, Any] = {"tokenizer": tokenizer}
    result, _ = compute_from_per_session_reports(
        per_session,
        explicit_tiers=explicit_tiers,
        xtime_excludes=xtime_excludes,
        baseline_tier=baseline_tier,
    )
    if result is None:
        return row, []
    row["generalization_score"] = result.summary["generalization_score"]
    by_metric: dict[tuple[str, str], dict[str, Any]] = {}
    for pm in result.per_metric:
        category = pm.get("category") or ""
        entry = by_metric.setdefault(
            (pm["metric"], category),
            {"tokenizer": tokenizer, "metric": pm["metric"], "category": category},
        )
        entry[f"ret_{pm['tier']}"] = pm["retention"]
    shared = [m for m in by_metric.values() if all(f"ret_{t}" in m for t in DEFAULT_OOD_TIERS)]
    if shared:
        for tier in DEFAULT_OOD_TIERS:
            values = [m[f"ret_{tier}"] for m in shared]
            row[f"gen_tier/{tier}"] = _geomean(values)
            row[f"gen_tier/{tier}::mean"] = statistics.fmean(values)
            row[f"gen_tier/{tier}::n"] = len(values)
    return row, [by_metric[key] for key in sorted(by_metric)]


def build_tables(
    inputs: pd.DataFrame,
    *,
    config_root: Path,
    gen_split: str,
    selection_family: str,
    baseline_tier: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    explicit, xtime_excludes = build_tier_index(
        Path(config_root), split=gen_split, selection_family=selection_family
    )
    tiers: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    for tokenizer, results_json in zip(inputs["tokenizer"], inputs["results_json"], strict=True):
        recon = json.loads(Path(results_json).read_text())
        t, m = tokenizer_rows(
            str(tokenizer),
            recon,
            explicit_tiers=explicit,
            xtime_excludes=xtime_excludes,
            baseline_tier=baseline_tier,
        )
        tiers.append(t)
        metrics += m
    return pd.DataFrame(tiers, columns=TIER_COLUMNS), pd.DataFrame(metrics, columns=METRIC_COLUMNS)


def check_scores(
    by_tier: pd.DataFrame, category_scores: pd.DataFrame, *, tol: float = 1e-9
) -> list[str]:
    """Tokenizers (sorted) whose ``generalization_score`` differs from the
    ``generalization`` category score, or is missing on either side."""
    ours_all = by_tier.set_index("tokenizer")["generalization_score"]
    shipped = category_scores.set_index("tokenizer")["generalization"]
    bad = []
    for tok in sorted(set(ours_all.index) | set(shipped.index)):
        ours, theirs = ours_all.get(tok), shipped.get(tok)
        if (
            ours is None
            or theirs is None
            or pd.isna(ours)
            or pd.isna(theirs)
            or not math.isclose(float(ours), float(theirs), rel_tol=0.0, abs_tol=tol)
        ):
            bad.append(str(tok))
    return bad


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--inputs", type=Path, required=True, help="CSV: tokenizer,results_json")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--gen-split", choices=("val", "test"), default="val")
    ap.add_argument("--selection-family", default="ntt_full")
    ap.add_argument("--baseline-tier", default="xtime")
    ap.add_argument("--config-root", type=Path, default=Path("config"))
    ap.add_argument("--check-scores", type=Path, help="category_scores.csv to reproduce")
    args = ap.parse_args(argv)

    inputs = pd.read_csv(args.inputs)
    dup = sorted(set(inputs.loc[inputs["tokenizer"].duplicated(), "tokenizer"]))
    if dup:
        print(f"duplicate tokenizers in {args.inputs}: {dup}", file=sys.stderr)
        return 2
    by_tier, by_metric = build_tables(
        inputs,
        config_root=args.config_root,
        gen_split=args.gen_split,
        selection_family=args.selection_family,
        baseline_tier=args.baseline_tier,
    )
    args.outdir.mkdir(parents=True, exist_ok=True)
    by_tier.to_csv(args.outdir / "generalization_by_tier.csv", index=False)
    by_metric.to_csv(args.outdir / "generalization_by_metric.csv", index=False)
    print(f"{len(inputs)} tokenizers, {len(by_metric)} metric rows -> {args.outdir}")
    if args.check_scores is not None:
        bad = check_scores(by_tier, pd.read_csv(args.check_scores))
        if bad:
            print(f"generalization check FAILED for {len(bad)}: {bad}", file=sys.stderr)
            return 1
        print("generalization check: generalization_score reproduces every category score")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
