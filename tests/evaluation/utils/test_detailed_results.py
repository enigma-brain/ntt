"""Core tests for detailed metric JSONL output."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from ntt.evaluation.core.types import (
    EvaluationReport,
    MetricCategory,
    MetricDirection,
    MetricLevel,
    MetricResult,
)
from ntt.evaluation.utils.detailed_results import write_metrics_detailed_results_jsonl


def test_detailed_results_records_include_opaque_run_metadata(tmp_path: Path):
    """Config provenance in run_metadata should flow into each JSONL record."""
    report = EvaluationReport(
        results={
            "mse": MetricResult(
                name="mse",
                raw_value=0.25,
                direction=MetricDirection.LOWER_IS_BETTER,
                category=MetricCategory.TEMPORAL_FIDELITY,
                level=MetricLevel.POPULATION,
            )
        }
    )
    segment = SimpleNamespace(segment_index=0, is_partial=False)
    metadata = {
        "eval_config": {"eval_window_seconds": 1.0},
        "data_config": {"selection_name": "dev-val"},
        "hydra_overrides": ["job=eval"],
        "composed_config_snapshot_path": "config_snapshot.yaml",
    }
    output_path = tmp_path / "metrics_detailed_results.jsonl"

    write_metrics_detailed_results_jsonl(
        session_reports={"S1": [(segment, {"report": report})]},
        output_path=output_path,
        run_metadata=metadata,
    )

    first_record = json.loads(output_path.read_text().splitlines()[0])
    assert first_record["run"] == metadata
