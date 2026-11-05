"""Convert per-session evaluation results to pandas DataFrames."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..aggregation.comparison import MetricComparison
from ..core.types import EvaluationReport
from ..utils.validation_collation import CollatedValidationSegment


def session_reports_to_dataframe(
    session_reports: dict[str, list[tuple[CollatedValidationSegment, dict[str, Any]]]],
) -> pd.DataFrame:
    """Convert per-session evaluation results to a tidy DataFrame.

    Each row corresponds to one (session, segment) pair. Columns include
    session metadata and per-metric summary statistics.

    Args:
        session_reports: Output of ``run_validation_evaluation_per_session()``.

    Returns:
        DataFrame with one row per segment.
    """
    rows: list[dict[str, Any]] = []

    for session_id, segment_list in session_reports.items():
        for segment, data in segment_list:
            report: EvaluationReport = data["report"]
            metadata: dict[str, Any] = data.get("metadata", {})

            # Split session_id into brainset / session_name
            if "/" in session_id:
                brainset, session_name = session_id.split("/", 1)
            else:
                brainset, session_name = session_id, session_id

            sr = metadata.get("sampling_rate")
            if sr is not None:
                import torch

                if isinstance(sr, torch.Tensor):
                    sr = sr[0].item() if sr.ndim > 0 else sr.item()
                sr = float(sr)
            shape = (
                segment.original_shape
                if segment.original_shape is not None
                else (tuple(segment.original.shape) if segment.original is not None else None)
            )
            if shape is None:
                raise RuntimeError("segment has neither original tensor nor original_shape")
            n_time_bins = int(shape[1])

            row: dict[str, Any] = {
                "session_id": session_id,
                "brainset": brainset,
                "session_name": session_name,
                "segment_index": segment.segment_index,
                "is_partial": segment.is_partial,
                "n_units": int(shape[0]),
                "n_time_bins": n_time_bins,
                "duration_sec": n_time_bins / sr if sr is not None else None,
                "sampling_rate": sr,
                "subject_id": metadata.get("subject_id"),
                "species": metadata.get("species"),
            }

            _add_metric_columns(row, report)
            rows.append(row)

    return pd.DataFrame(rows)


def report_to_dataframe(
    report: EvaluationReport,
    session_id: str = "unknown",
) -> pd.DataFrame:
    """Convert a single EvaluationReport to a one-row DataFrame.

    Useful for standalone evaluation without validation collation.

    Args:
        report: The evaluation report.
        session_id: Label for the session.

    Returns:
        Single-row DataFrame with metric columns.
    """
    row: dict[str, Any] = {"session_id": session_id}
    _add_metric_columns(row, report)
    return pd.DataFrame([row])


def save_results_csv(df: pd.DataFrame, path: str | Path) -> None:
    """Save results DataFrame to CSV."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)


def load_results_csv(path: str | Path) -> pd.DataFrame:
    """Load results DataFrame from CSV."""
    return pd.read_csv(path)


def normalized_comparisons_to_dataframe(
    comparisons: dict[str, MetricComparison],
) -> pd.DataFrame:
    """Convert normalized comparisons to a tidy DataFrame.

    One row per metric with columns for model/baseline/ceiling values,
    the normalized score, category, direction, and status.
    """
    rows: list[dict[str, Any]] = []
    for name, comp in comparisons.items():
        rows.append(
            {
                "metric": name,
                "category": comp.category.value,
                "direction": comp.direction.value,
                "level": comp.level.value,
                "model_value": comp.model_value,
                "baseline_value": comp.baseline_value,
                # What baseline_value is: "poisson", or another reference (ALTERNATIVE_FLOORS).
                "baseline_kind": comp.baseline_kind,
                "ceiling_value": comp.ceiling_value,
                "normalized_score": comp.normalized_score,
                "status": comp.status,
                "reason": comp.reason,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _add_metric_columns(row: dict[str, Any], report: EvaluationReport) -> None:
    """Populate *row* dict with per-metric mean/std and category scores."""
    for name, result in report.results.items():
        val = result.raw_value
        if isinstance(val, np.ndarray):
            row[name] = float(np.nanmean(val))
            row[f"{name}_std"] = float(np.nanstd(val))
        else:
            row[name] = float(val) if not np.isnan(val) else np.nan
            row[f"{name}_std"] = np.nan

    if report.category_scores:
        for cat, score in report.category_scores.items():
            row[f"category_{cat.value}"] = score
