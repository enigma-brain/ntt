"""Write a per-metric detailed results file for plotting/analysis.

This complements the existing `results.json` (hierarchical) and `results.csv` (wide)
outputs by writing a long-form JSONL file with one record per metric evaluation
instance, including one record per parameter sweep run when present.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ntt.evaluation.core.types import EvaluationReport, MetricResult
from ntt.evaluation.utils.serialization import json_default


def _iter_sweep_points(metric_result: MetricResult) -> Iterable[dict[str, Any]]:
    """Yield sweep-point records from a MetricResult's metadata.

    Convention today:
    - downstream prediction metrics store `metadata["chunk_window_sweep"]` as a dict
      mapping chunk_window_ms -> {raw_value, metadata, status, error?}

    Other existing "sweep-like" conventions:
    - timescale/bin-size dictionaries such as `*_by_bin_size` (e.g. correlations_by_bin_size,
      nmi_by_bin_size), mapping bin_size_ms -> per-unit array (or scalar).
    - k-NN overlap dictionaries such as `knn_overlap`, mapping k -> overlap (scalar).

    This function supports:
    - explicit `*_sweep` dicts whose values are payload dicts
    - sweep-like dicts mapping param_value -> numeric/array values
    """

    meta = metric_result.metadata or {}
    if not isinstance(meta, dict) or not meta:
        return

    # Special-case known sweep names whose parameter should be more explicit.
    sweep_param_name_overrides = {
        "chunk_window_sweep": "chunk_window_ms",
    }

    def _is_numeric_scalar(x: Any) -> bool:
        return isinstance(x, (int, float)) and x == x  # noqa: PLR0124 (NaN check)

    def _param_name_for_dict_key(key_name: str) -> str:
        if key_name == "knn_overlap":
            return "k"
        if "bin_size" in key_name:
            return "bin_size_ms"
        return key_name

    for key_name, maybe_sweep in meta.items():
        if not isinstance(key_name, str):
            continue
        if not isinstance(maybe_sweep, dict) or not maybe_sweep:
            continue

        # 1) Explicit sweeps: `*_sweep` mapping param_value -> payload dict
        if key_name.endswith("_sweep"):
            param_name = sweep_param_name_overrides.get(key_name, key_name.removesuffix("_sweep"))
            for param_value, payload in maybe_sweep.items():
                if not isinstance(payload, dict):
                    payload = {"raw_value": payload}
                yield {
                    "sweep_key": key_name,
                    "param_name": param_name,
                    "param_value": param_value,
                    "raw_value": payload.get("raw_value"),
                    "status": payload.get("status"),
                    "error": payload.get("error"),
                    "run_metadata": payload.get("metadata"),
                    "payload_extra": {
                        k: v
                        for k, v in payload.items()
                        if k not in {"raw_value", "status", "error", "metadata"}
                    }
                    or None,
                }
            continue

        # 2) Sweep-like dicts: bin-size / k dictionaries mapping param_value -> numeric/array
        #    Heuristic: keys are scalar-ish (numbers/strings) and values are numeric scalars
        #    or arrays/lists (per-unit results). We ignore dict-of-dict here.
        if (
            key_name.endswith("_by_bin_size")
            or key_name.endswith("_by_bin_size_ms")
            or key_name
            in {
                "knn_overlap",
            }
        ):
            param_name = _param_name_for_dict_key(key_name)
            for param_value, value in maybe_sweep.items():
                if isinstance(value, dict):
                    # This is likely not a simple by-bin dict; skip to avoid double-parsing.
                    continue
                # Accept numeric scalar, list, or ndarray (serialized later).
                if _is_numeric_scalar(value) or isinstance(value, (list, tuple)):
                    raw_val = value
                else:
                    raw_val = value
                yield {
                    "sweep_key": key_name,
                    "param_name": param_name,
                    "param_value": param_value,
                    "raw_value": raw_val,
                    "status": "ok",
                    "error": None,
                    "run_metadata": None,
                    "payload_extra": None,
                }


def write_metrics_detailed_results_jsonl(
    *,
    session_reports: dict[str, list[tuple[Any, dict[str, Any]]]],
    output_path: str | Path,
    run_metadata: dict[str, Any] | None = None,
) -> Path:
    """Write detailed metric results as JSON Lines.

    One line is written per (session, segment, metric) result, plus one line per
    sweep-point when a metric stores a `*_sweep` dict in its metadata.

    Notes:
    - This does NOT replace existing outputs; it is an additional file.
    - No aggregation is performed here; aggregation remains in `results.json`
      and `results.csv`.
    """

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base_run_meta = dict(run_metadata or {})

    with out_path.open("w") as f:
        for session_id, segment_list in (session_reports or {}).items():
            for segment, data in segment_list or []:
                report = data.get("report")
                if not isinstance(report, EvaluationReport):
                    continue

                segment_index = getattr(segment, "segment_index", None)
                is_partial = getattr(segment, "is_partial", None)

                for metric_key, res in (report.results or {}).items():
                    # 1) Base metric result record (best / scalar-as-defined by metric)
                    base_rec: dict[str, Any] = {
                        "record_type": "metric_result",
                        "run": base_run_meta,
                        "session_id": session_id,
                        "segment_index": segment_index,
                        "is_partial": is_partial,
                        "metric_key": metric_key,
                        "metric": res.to_dict(),
                    }
                    f.write(json.dumps(base_rec, default=json_default) + "\n")

                    # 2) One record per sweep run (if present)
                    for sp in _iter_sweep_points(res):
                        sweep_rec: dict[str, Any] = {
                            "record_type": "sweep_point",
                            "run": base_run_meta,
                            "session_id": session_id,
                            "segment_index": segment_index,
                            "is_partial": is_partial,
                            "metric_key": metric_key,
                            # Re-state metric identity without repeating huge metadata.
                            "metric_info": {
                                "name": res.name,
                                "category": res.category.value,
                                "direction": res.direction.value,
                                "level": res.level.value,
                                "value_range": list(res.value_range),
                                "source_data_type": (
                                    res.source_data_type.value if res.source_data_type else None
                                ),
                            },
                            "sweep": sp,
                        }
                        f.write(json.dumps(sweep_rec, default=json_default) + "\n")

    return out_path
