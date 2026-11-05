#!/usr/bin/env python
"""Aggregate NTT eval job outputs into a tree consumable by RadarBundle.

Reads a CSV manifest with columns:

    tokenizer              (required) display name; must be filesystem-safe
    results_uri            (required) local path or mlflow-artifacts:/... URI

For each row, the script copies the resolved results.json into:

    <output_dir>/<tokenizer>/results.json

If ``--default-poisson-reference`` is given, the shared baseline lands at:

    <output_dir>/_reference/poisson_baseline.json

This shared file is the only location ``RadarBundle.from_local_dir`` reads
as an override; use it when you want one canonical floor for paper figures.
Omit it and the radar falls back to each tokenizer's in-results.json
baseline.
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
import sys
from pathlib import Path

LOGGER = logging.getLogger("aggregate_runs")


def _resolve_uri(uri: str) -> Path:
    """Resolve a results_uri to a local path; download via MLflow if needed."""
    if uri.startswith(("mlflow-artifacts:", "mlflow:")):
        import mlflow.artifacts

        return Path(mlflow.artifacts.download_artifacts(uri))
    return Path(uri).expanduser().resolve()


def _copy_to(src: Path, dst: Path) -> None:
    """Copy a JSON file (or the right JSON inside a downloaded directory) to dst."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        # The MLflow downloader sometimes returns a directory holding the file.
        candidates = sorted(src.glob("*.json"))
        if not candidates:
            raise FileNotFoundError(f"No JSON file inside resolved dir {src}")
        # Prefer a file whose name matches the destination's name (e.g.
        # results.json or poisson_baseline.json) to avoid silently picking
        # whichever the filesystem returned first. Otherwise raise.
        named_match = [c for c in candidates if c.name == dst.name]
        if named_match:
            src = named_match[0]
        elif len(candidates) == 1:
            src = candidates[0]
        else:
            raise FileNotFoundError(
                f"Multiple JSON files inside {src}; could not determine "
                f"which to use as {dst.name}: {[c.name for c in candidates]}"
            )
    shutil.copy2(src, dst)


def aggregate(
    manifest_path: Path,
    output_dir: Path,
    *,
    default_poisson_reference: Path | None = None,
) -> Path | None:
    """Aggregate runs. Returns the path to the produced manifest.csv, or None."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_out: list[dict[str, str]] = []
    with manifest_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            tokenizer = (row.get("tokenizer") or "").strip()
            results_uri = (row.get("results_uri") or "").strip()
            if not tokenizer or not results_uri:
                LOGGER.warning("Skipping row with missing tokenizer or results_uri: %r", row)
                continue

            results_dst = output_dir / tokenizer / "results.json"
            try:
                _copy_to(_resolve_uri(results_uri), results_dst)
            except Exception as e:
                LOGGER.error("Failed to copy %s for %s: %s", results_uri, tokenizer, e)
                continue

            rows_out.append(
                {
                    "tokenizer": tokenizer,
                    "results_path": str(results_dst.relative_to(output_dir)),
                    "results_uri": results_uri,
                }
            )

    if default_poisson_reference is not None:
        try:
            _copy_to(
                _resolve_uri(str(default_poisson_reference)),
                output_dir / "_reference" / "poisson_baseline.json",
            )
        except Exception as e:
            LOGGER.error("Failed to copy --default-poisson-reference: %s", e)

    out_manifest = output_dir / "manifest.csv"
    if rows_out:
        with out_manifest.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows_out[0]))
            writer.writeheader()
            writer.writerows(rows_out)
        return out_manifest
    LOGGER.warning("No rows processed; manifest.csv not written.")
    return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--default-poisson-reference", type=Path, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    out_manifest = aggregate(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        default_poisson_reference=args.default_poisson_reference,
    )
    if out_manifest is None:
        print("No rows aggregated.")
        return 1
    print(f"Wrote {out_manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
