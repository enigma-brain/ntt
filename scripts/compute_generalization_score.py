#!/usr/bin/env python
"""Compute split-wise generalization retention from per-session eval reports.

This script is a thin CLI around ``ntt.evaluation.generalization``. It exists
because users sometimes need to re-score existing per-session JSON outputs
(MLflow artifacts) for which no live eval job is being run.

The eval job emits the same scores directly into ``results.json`` under
the ``generalization`` key — prefer that path when you control the run.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

from ntt.evaluation.generalization import (
    DEFAULT_BASELINE_MODE,
    DEFAULT_EXCLUDED_BRAINSETS,
    DEFAULT_EXCLUDED_CATEGORIES,
    DEFAULT_EXCLUDED_METRICS,
    DEFAULT_RETENTION_FLOOR,
    GeneralizationResult,
    aggregate_metric_values,
    build_tier_index,
    compute_from_observations,
    load_observations,
)
from ntt.evaluation.generalization import (
    DEFAULT_OOD_TIERS as OOD_TIERS,
)

# -- CLI helpers -------------------------------------------------------


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def resolve_artifact_folder(path_or_uri: str) -> Path:
    if path_or_uri.startswith(("mlflow-artifacts:", "mlflow:")):
        import mlflow.artifacts

        return Path(mlflow.artifacts.download_artifacts(path_or_uri))
    return Path(path_or_uri).expanduser().resolve()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: dict[str, Any], sessions: list[dict[str, str]]) -> None:
    counts: dict[str, int] = {}
    for session in sessions:
        counts[session["tier"]] = counts.get(session["tier"], 0) + 1

    lines = [
        "# Generalization Retention Summary",
        "",
        f"- Generalization score: `{summary['generalization_score']}`",
        f"- Mean retention: `{summary['mean_retention']}`",
        f"- Retention terms: `{summary['n_retention_terms']}`",
        "",
        "## Sessions By Tier",
        "",
        "| Tier | Sessions |",
        "|---|---:|",
    ]
    for tier, count in sorted(counts.items()):
        lines.append(f"| {tier} | {count} |")
    lines.extend(
        [
            "",
            "## Tier Scores",
            "",
            "| Tier | Metrics | Mean | Geometric Mean |",
            "|---|---:|---:|---:|",
        ]
    )
    for tier, stats in summary["tier_scores"].items():
        lines.append(
            f"| {tier} | {stats['n_metrics']} | {stats['mean_retention']} | "
            f"{stats['geometric_mean_retention']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("per_session_uri")
    parser.add_argument("--output-dir", default="outputs/generalization_score")
    parser.add_argument("--config-root", default="config")
    parser.add_argument("--selection-family", default="ntt_full")
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--baseline-tier", default="xtime")
    parser.add_argument(
        "--baseline-mode",
        default=DEFAULT_BASELINE_MODE,
        choices=("brainset_matched", "tier_mean"),
        help="Compare each held-out session with its own brainset's xtime sessions "
        "(brainset_matched), or the tier mean with the pooled xtime mean (tier_mean).",
    )
    parser.add_argument("--env-file", default=".env")
    parser.add_argument(
        "--exclude-metric",
        action="append",
        default=list(DEFAULT_EXCLUDED_METRICS),
    )
    parser.add_argument(
        "--exclude-brainset",
        action="append",
        default=list(DEFAULT_EXCLUDED_BRAINSETS),
    )
    parser.add_argument("--include-token-dims", action="store_true")
    parser.add_argument("--no-clip-improvements", action="store_true")
    parser.add_argument(
        "--retention-floor",
        type=float,
        default=DEFAULT_RETENTION_FLOOR,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(Path(args.env_file))

    excluded_metrics = set(args.exclude_metric or [])
    if args.include_token_dims:
        excluded_metrics.discard("token_dims_per_second")
    excluded_brainsets = set(args.exclude_brainset or [])

    per_session_dir = resolve_artifact_folder(args.per_session_uri)

    # Run the pipeline step-by-step so we can emit `tier_metric_values.csv`
    # with full coverage. compute_from_per_session_dir is a fine one-shot
    # wrapper, but its `per_metric` payload only contains retention rows for
    # the (baseline ∩ ood) metric intersection — so deriving tier_metric_values
    # from it would drop metrics present in only one tier. Calling the
    # underlying helpers directly preserves the CSV's "all
    # observed (tier, metric) pairs, sorted" semantics.
    explicit_tiers, xtime_excludes = build_tier_index(
        Path(args.config_root),
        split=args.split,
        selection_family=args.selection_family,
        ood_tiers=OOD_TIERS,
    )
    observations, sessions_table = load_observations(
        per_session_dir,
        explicit_tiers,
        xtime_excludes,
        excluded_brainsets=excluded_brainsets,
        excluded_metrics=excluded_metrics,
        excluded_categories=DEFAULT_EXCLUDED_CATEGORIES,
    )
    metric_rows, metric_lookup = aggregate_metric_values(observations)
    if args.baseline_tier not in {t for (t, _) in metric_lookup}:
        raise ValueError(
            f"baseline tier {args.baseline_tier!r} has no metrics in "
            f"{per_session_dir} (consult session_tiers.csv for classification)."
        )
    retention_rows, score_summary = compute_from_observations(
        observations,
        baseline_tier=args.baseline_tier,
        ood_tiers=OOD_TIERS,
        baseline_mode=args.baseline_mode,
        clip_improvements=not args.no_clip_improvements,
        retention_floor=args.retention_floor,
    )
    result = GeneralizationResult(
        per_metric=retention_rows,
        summary=score_summary,
        baseline_tier=args.baseline_tier,
        ood_tiers=OOD_TIERS,
        excluded_metrics=tuple(sorted(excluded_metrics)),
        excluded_categories=tuple(sorted(DEFAULT_EXCLUDED_CATEGORIES)),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "session_tiers.csv", sessions_table)
    write_csv(output_dir / "tier_metric_values.csv", metric_rows)
    write_csv(output_dir / "metric_retention.csv", result.per_metric)

    payload = {
        "input": {
            "per_session_uri": args.per_session_uri,
            "resolved_per_session_dir": str(per_session_dir),
            "config_root": str(Path(args.config_root).resolve()),
            "selection_family": args.selection_family,
            "split": args.split,
            "baseline_tier": args.baseline_tier,
            "baseline_mode": args.baseline_mode,
            "excluded_brainsets": sorted(excluded_brainsets),
            "excluded_metrics": sorted(excluded_metrics),
            "excluded_categories": sorted(DEFAULT_EXCLUDED_CATEGORIES),
            "clip_improvements": not args.no_clip_improvements,
            "retention_floor": args.retention_floor,
        },
        "summary": result.summary,
    }
    (output_dir / "generalization_summary.json").write_text(json.dumps(payload, indent=2))
    write_markdown(output_dir / "generalization_summary.md", result.summary, sessions_table)

    print(f"Wrote generalization outputs to {output_dir.resolve()}")
    print(f"generalization_score={result.summary['generalization_score']}")


if __name__ == "__main__":
    main()
