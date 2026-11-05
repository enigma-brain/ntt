"""Tests for scripts/analysis/aggregate_runs.py — local-paths-only round-trip."""

from __future__ import annotations

import csv
import importlib
import json
import sys
from pathlib import Path


def _run_aggregator(manifest, output_dir, default_ref=None):
    """Invoke aggregate_runs.main with synthetic argv."""
    script_dir = Path(__file__).resolve().parents[2] / "scripts" / "analysis"
    sys.path.insert(0, str(script_dir))
    try:
        import aggregate_runs  # noqa: F401

        importlib.reload(aggregate_runs)
    finally:
        sys.path.pop(0)
    argv = ["--manifest", str(manifest), "--output-dir", str(output_dir)]
    if default_ref is not None:
        argv += ["--default-poisson-reference", str(default_ref)]
    aggregate_runs.main(argv)


def _write_results(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_aggregator_copies_local_results(tmp_path):
    # Two fake source results.json files.
    src_a = tmp_path / "src_a"
    src_b = tmp_path / "src_b"
    payload = {
        "aggregated": {"results": {}},
        "poisson_baseline": {"results": {}},
        "perfect_ceiling": {"results": {}},
    }
    _write_results(src_a / "results.json", payload)
    _write_results(src_b / "results.json", payload)

    manifest = tmp_path / "runs.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tokenizer", "results_uri"])
        writer.writeheader()
        writer.writerow(
            {
                "tokenizer": "tok_a",
                "results_uri": str(src_a / "results.json"),
            }
        )
        writer.writerow(
            {
                "tokenizer": "tok_b",
                "results_uri": str(src_b / "results.json"),
            }
        )

    out_dir = tmp_path / "out"
    _run_aggregator(manifest, out_dir)

    assert (out_dir / "tok_a" / "results.json").exists()
    assert (out_dir / "tok_b" / "results.json").exists()
    assert (out_dir / "manifest.csv").exists()

    # Verify manifest.csv has the required columns.
    with (out_dir / "manifest.csv").open() as f:
        manifest_rows = list(csv.DictReader(f))
    assert len(manifest_rows) == 2
    assert set(manifest_rows[0].keys()) == {
        "tokenizer",
        "results_path",
        "results_uri",
    }
    tokenizers_in_manifest = {row["tokenizer"] for row in manifest_rows}
    assert tokenizers_in_manifest == {"tok_a", "tok_b"}


def test_aggregator_writes_shared_reference(tmp_path):
    src = tmp_path / "src"
    _write_results(
        src / "results.json",
        {
            "aggregated": {"results": {}},
            "poisson_baseline": {"results": {}},
            "perfect_ceiling": {"results": {}},
        },
    )
    ref = tmp_path / "ref" / "poisson_baseline.json"
    _write_results(ref, {"aggregated": {"results": {}}, "perfect_aggregated": {"results": {}}})

    manifest = tmp_path / "runs.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tokenizer", "results_uri"])
        writer.writeheader()
        writer.writerow({"tokenizer": "tok_a", "results_uri": str(src / "results.json")})

    out_dir = tmp_path / "out"
    _run_aggregator(manifest, out_dir, default_ref=ref)
    assert (out_dir / "_reference" / "poisson_baseline.json").exists()
