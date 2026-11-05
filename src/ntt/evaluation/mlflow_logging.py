"""MLflow logging utilities for NTT evaluation results."""

from __future__ import annotations

import json
import logging
import math
import re
import secrets
import tempfile
import time
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pydantic as pdt

from ntt.evaluation.aggregation.comparison import MetricComparison
from ntt.evaluation.core.types import EvaluationReport

try:
    import mlflow
    from mlflow.entities import Metric
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient
except ImportError:  # pragma: no cover
    mlflow = None  # type: ignore[assignment]
    Metric = None  # type: ignore[assignment,misc]
    MlflowException = None  # type: ignore[assignment,misc]
    MlflowClient = None  # type: ignore[assignment,misc]

# MLflow's server-side cap on the number of metrics accepted by a single
# ``log_batch`` call. We chunk to stay strictly under it.
_MLFLOW_LOG_BATCH_MAX = 200
_MLFLOW_ARTIFACT_UPLOAD_PAUSE_SECONDS = 0.1

log = logging.getLogger(__name__)

_MLFLOW_RUN_ID_PATTERNS = (
    re.compile(r"^mlflow(?:-artifacts)?:/runs/([^/]+)/"),
    re.compile(r"^mlflow-artifacts:/[^/]+/([^/]+)/artifacts/"),
)

# Matches metric display names like "acg_correlation_10ms" or
# "smoothed_squared_error_25ms" — captures the base name and the integer
# bin size.  Used to render temporal sweeps as line plots in MLflow.
_TEMPORAL_SUFFIX_PATTERN = re.compile(r"^(.+)_(\d+)ms$")

# A single MLflow log point: the canonical 3-tuple for ``log_metric``.
StepPoint = tuple[str, float, int]

# MLflow metric names allow alphanumerics, underscore, dash, period, space,
# colon, slash. input_robustness (M1) sub_labels embed corruption-spec params with ``|`` and
# ``=`` (e.g. ``jitter|k_ms=10.0|p=0.5``), so collapse those to ``-`` rather
# than silently losing the information.
_MLFLOW_INVALID_CHAR_RE = re.compile(r"[^A-Za-z0-9_\-./:]+")


def parse_training_run_id(checkpoint_path: str) -> str | None:
    """Extract the MLflow run ID from an mlflow-artifacts:/ URI.

    Returns None if the path is local or doesn't contain a run ID.
    """
    for pattern in _MLFLOW_RUN_ID_PATTERNS:
        match = pattern.match(checkpoint_path)
        if match is not None:
            return match.group(1)
    return None


def resolve_eval_run_name(
    explicit: str | None,
    ckpt_path: str | Path | None,
    spec_kind: str,
    tracking_uri: str | None = None,
) -> str:
    """Build a human-readable MLflow run name for an eval run.

    Resolution order — each branch appends a 6-hex disambiguator so that
    re-evaluating the same model produces distinct run names in the UI:

    1. ``explicit`` (when set in ``mlflow_logging.run_name``).
    2. The training run's ``run_name`` looked up via a scoped
       ``MlflowClient(tracking_uri=...)``, when ``ckpt_path`` is an
       ``mlflow-artifacts:/runs/<id>/...`` URI. The client is local to
       this call — the process-global tracking URI is left untouched.
       Any lookup failure (unreachable tracking URI, missing run, MLflow
       not installed) is logged at debug level and falls through to step 3.
    3. ``spec_kind`` (the tokenizer's ``KIND``), as the final fallback.
    """
    suffix = secrets.token_hex(3)

    if explicit:
        return f"{explicit}-{suffix}"

    if ckpt_path is not None and MlflowClient is not None:
        training_run_id = parse_training_run_id(str(ckpt_path))
        if training_run_id is not None:
            try:
                client = MlflowClient(tracking_uri=tracking_uri)
                training_name = client.get_run(training_run_id).info.run_name
                if training_name:
                    return f"{training_name}-{suffix}"
            except Exception as e:
                log.debug(
                    "resolve_eval_run_name: training run name lookup failed for "
                    "run_id=%s tracking_uri=%s — falling back to spec_kind: %s",
                    training_run_id,
                    tracking_uri,
                    e,
                )

    return f"{spec_kind}-{suffix}"


def flatten_report_to_metrics(
    report: EvaluationReport,
    prefix: str = "eval",
) -> dict[str, float]:
    """Convert an EvaluationReport into a flat dict of scalar metrics.

    Per-unit/per-pair array metrics are reduced to mean and std.
    NaN values are excluded (MLflow rejects them).
    """
    metrics: dict[str, float] = {}

    for name, result in report.results.items():
        val = result.raw_value
        name = sanitize_metric_name_for_mlflow(name)
        if isinstance(val, np.ndarray):
            mean = float(np.nanmean(val))
            std = float(np.nanstd(val))
            # Exclude non-finite reductions: MLflow accepts ±inf but it corrupts
            # dashboards/aggregates, and the series paths already drop it.
            if np.isfinite(mean):
                metrics[f"{prefix}/{name}"] = mean
            if np.isfinite(std):
                metrics[f"{prefix}/{name}/std"] = std
        else:
            if np.isfinite(float(val)):
                metrics[f"{prefix}/{name}"] = float(val)

    if report.category_scores:
        for cat, score in report.category_scores.items():
            if np.isfinite(float(score)):
                metrics[f"{prefix}/category/{cat.value}"] = float(score)

    metrics[f"{prefix}/n_skipped"] = float(len(report.skipped_metrics))

    return metrics


def extract_temporal_curves(
    report: EvaluationReport,
    prefix: str = "eval/temporal",
) -> dict[str, list[tuple[int, float]]]:
    """Group temporal-sweep metrics into curves keyed by base metric name.

    Looks for metric display names matching ``{base_name}_{N}ms`` and groups
    them into ``{prefix}/{base_name}`` -> ``[(bin_ms, value), ...]`` sorted
    by bin size.  Per-unit array metrics are reduced to their mean.  Logged
    via ``mlflow.log_metric(name, value, step=bin_ms)`` so MLflow renders
    them as line plots (x = bin size, y = value).

    When a bare ``{base_name}`` entry (no suffix) also exists in the report
    *and* that base name has at least one ``_{N}ms`` variant, the bare entry
    is treated as the 1 ms point on the curve.  This supports the
    "default-as-1 ms" override pattern (e.g. hill_climb): the base metric
    entry is configured to evaluate at 1 ms via deep-merge override on the
    metric YAML instead of being nulled and replaced by a separate ``_1ms``
    display name.
    """

    def _to_scalar(result) -> float | None:  # type: ignore[no-untyped-def]
        val = result.raw_value
        scalar = float(np.nanmean(val)) if isinstance(val, np.ndarray) else float(val)
        return None if math.isnan(scalar) else scalar

    curves: dict[str, list[tuple[int, float]]] = {}
    swept_bases: set[str] = set()

    # First pass: collect _{N}ms variants.
    for name, result in report.results.items():
        match = _TEMPORAL_SUFFIX_PATTERN.match(name)
        if match is None:
            continue

        base_name = match.group(1)
        bin_ms = int(match.group(2))

        scalar = _to_scalar(result)
        if scalar is None:
            continue

        curves.setdefault(f"{prefix}/{base_name}", []).append((bin_ms, scalar))
        swept_bases.add(base_name)

    # Second pass: any bare ``{base_name}`` that has sweep variants is
    # logged as the 1 ms point on its curve.
    for name, result in report.results.items():
        if name not in swept_bases:
            continue
        if _TEMPORAL_SUFFIX_PATTERN.match(name) is not None:
            continue  # already consumed in first pass
        scalar = _to_scalar(result)
        if scalar is None:
            continue
        curves[f"{prefix}/{name}"].append((1, scalar))

    for key in curves:
        curves[key].sort(key=lambda t: t[0])

    return curves


def sanitize_metric_name_for_mlflow(name: str) -> str:
    """Replace MLflow-illegal chars in a metric name with ``-``.

    Collapses runs of illegal chars to a single ``-`` to avoid double dashes.
    """
    return _MLFLOW_INVALID_CHAR_RE.sub("-", name)


def _safe_float(v: Any) -> float | None:
    """Coerce ``v`` to a finite float or return ``None``."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return f


def _flatten_generalization_metrics(block: dict) -> dict[str, float]:
    """Flatten the generalization block's ``summary`` into MLflow metrics.

    Emits ``eval/generalization/<path>`` for every finite numeric leaf under
    the ``summary`` sub-block (the headline retention scores). Non-numeric and
    non-finite values, and the verbose ``per_metric`` list, are skipped.
    """
    out: dict[str, float] = {}

    def _walk(obj: Any, prefix: str) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                _walk(value, f"{prefix}/{sanitize_metric_name_for_mlflow(str(key))}")
            return
        if isinstance(obj, bool):
            return
        f = _safe_float(obj)
        if f is not None:
            out[prefix] = f

    summary = block.get("summary")
    if isinstance(summary, dict):
        _walk(summary, "eval/generalization")
    return out


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file, returning the parsed records as a list."""
    out: list[dict] = []
    with path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def flatten_input_robustness_jsonl_to_metric_series(
    jsonl_path: Path,
    label_to_type_strength: dict[str, tuple[str, float]],
    prefix: str = "eval_m1",
) -> tuple[list[StepPoint], dict[str, float]]:
    """Flatten an input_robustness (M1) ``results.jsonl`` into MLflow series + scalars."""
    records = _read_jsonl(jsonl_path)

    groups: dict[tuple[str, str, str, str | None], list[tuple[float, float]]] = defaultdict(list)
    n_records = 0
    n_skipped = 0
    for rec in records:
        if rec.get("status") != "ok":
            n_skipped += 1
            continue
        n_records += 1
        label = rec.get("corruption")
        if label is None or label not in label_to_type_strength:
            continue
        family, strength = label_to_type_strength[label]
        sub_label = rec.get("sub_label")
        v = _safe_float(rec.get("raw_value"))
        if v is None:
            continue
        key = (rec["session_id"], family, rec["metric"], sub_label)
        groups[key].append((float(strength), v))

    # Steps are assigned from a per-family strength axis shared by every metric
    # in that family (the union of all strengths). This keeps the single
    # ``_x_strength`` series a correct step->strength map even when metrics cover
    # different strength sets (e.g. one is NaN-dropped at some strength) — a
    # per-metric ``enumerate`` would otherwise mislabel those steps.
    family_strengths: dict[tuple[str, str], set[float]] = defaultdict(set)
    for (session_id, family, _metric, _sub_label), pairs in groups.items():
        for strength, _value in pairs:
            family_strengths[(session_id, family)].add(float(strength))
    family_step: dict[tuple[str, str], dict[float, int]] = {
        fam: {strength: step for step, strength in enumerate(sorted(strengths))}
        for fam, strengths in family_strengths.items()
    }

    series: list[StepPoint] = []
    seen_x_axes: set[tuple[str, str]] = set()
    for (session_id, family, metric, sub_label), pairs in groups.items():
        per_strength: dict[float, list[float]] = defaultdict(list)
        for strength, value in pairs:
            per_strength[float(strength)].append(value)

        sub_suffix = f"/{sub_label}" if sub_label else ""
        name = sanitize_metric_name_for_mlflow(
            f"{prefix}/{session_id}/{family}/{metric}{sub_suffix}"
        )
        x_name = sanitize_metric_name_for_mlflow(f"{prefix}/{session_id}/{family}/_x_strength")
        step_of = family_step[(session_id, family)]
        for strength, vals in sorted(per_strength.items()):
            series.append((name, float(np.mean(vals)), step_of[strength]))
        if (session_id, family) not in seen_x_axes:
            for strength, step in step_of.items():
                series.append((x_name, strength, step))
            seen_x_axes.add((session_id, family))

    scalars = {
        f"{prefix}/n_records": float(n_records),
        f"{prefix}/n_skipped": float(n_skipped),
    }
    return series, scalars


def extract_input_robustness_jsonl_summary_metrics(
    jsonl_path: Path,
    *,
    prefix: str = "eval_m1",
) -> dict[str, float]:
    """Extract compact input_robustness (M1) summary metrics from ``results.jsonl``.

    The input_robustness (M1) JSONL contains the derived ``robustness_input_core_avg`` rows that
    are currently missing from the report-based MLflow logging path because the
    derived metric has no captured schema.  These rows are already the source
    of truth for the local sheet backfill, so we reuse them here and log only
    readable scalar keys:

    - ``{prefix}/joint/core_avg``
    - ``{prefix}/session/<sid>/joint/core_avg``
    """
    records = _read_jsonl(jsonl_path)
    per_session_core: dict[str, list[float]] = defaultdict(list)

    for rec in records:
        if rec.get("status") != "ok":
            continue
        if rec.get("metric") != "robustness_input_core_avg":
            continue
        session_id = rec.get("session_id")
        if session_id is None:
            continue
        value = _safe_float(rec.get("raw_value"))
        if value is None:
            continue
        per_session_core[str(session_id)].append(value)

    out: dict[str, float] = {}
    all_values: list[float] = []
    for session_id, values in sorted(per_session_core.items()):
        safe_sid = session_id.replace("/", "__")
        out[sanitize_metric_name_for_mlflow(f"{prefix}/session/{safe_sid}/joint/core_avg")] = (
            float(np.mean(values))
        )
        all_values.extend(values)

    if all_values:
        out[sanitize_metric_name_for_mlflow(f"{prefix}/joint/core_avg")] = float(
            np.mean(all_values)
        )
    return out


def merge_input_robustness_metric_sources(
    sources: Iterable[tuple[str, dict[str, float]]],
) -> tuple[dict[str, float], dict[str, list[tuple[str, float]]]]:
    """First-wins merge over ``(source_name, metrics)`` pairs in trust order.

    Returns ``(merged, conflicts)`` where ``conflicts`` maps each contested key
    to its ``[(source_name, value), ...]`` candidates (most-trusted first). A
    key landing in two sources with materially different values is a bug (e.g.
    the report path and the jsonl-summary path disagreeing on
    ``joint/core_avg``) — but this runs post-compute, so we must not abort an
    otherwise-complete eval. The caller surfaces conflicts (error log + run
    tag); the merged dict keeps the value from the earliest — most trusted —
    source. NaN-vs-NaN is not a conflict (``equal_nan=True``).
    """
    merged: dict[str, float] = {}
    provenance: dict[str, str] = {}
    conflicts: dict[str, list[tuple[str, float]]] = {}
    for source_name, metrics in sources:
        for key, value in metrics.items():
            if key not in merged:
                merged[key] = value
                provenance[key] = source_name
            elif not np.isclose(merged[key], value, equal_nan=True):
                conflicts.setdefault(key, [(provenance[key], merged[key])]).append(
                    (source_name, value)
                )
    return merged, conflicts


def remap_input_robustness_metric_keys_for_mlflow(
    metrics: dict[str, float],
    *,
    prefix: str = "eval_m1",
) -> dict[str, float]:
    """Rewrite verbose input_robustness (M1) report keys into a shorter stable layout.

    Input keys are ``{prefix}/robustness_input/...`` (aggregate joint scores),
    ``{prefix}/robustness_input_per_dim/...`` (aggregate per-dim scores) and
    ``{prefix}/per_session/<sid>/...`` (session scores).

    The output keeps only scalar metrics and makes the hierarchy explicit:
    - ``{prefix}/joint/...``
    - ``{prefix}/per_dim/...``
    - ``{prefix}/session/<sid>/joint/...``
    - ``{prefix}/session/<sid>/per_dim/...``
    """

    def _insert(out: dict[str, float], key: str, value: float) -> None:
        if key in out and not np.isclose(out[key], value):
            # This runs post-compute, so we must not abort an otherwise-complete
            # eval (mirrors merge_input_robustness_metric_sources). Keep the first value and
            # warn rather than raise on a remap collision.
            log.warning(
                "Conflicting input_robustness (M1) MLflow metric remap for key %r: keeping "
                "existing=%r, dropping new=%r",
                key,
                out[key],
                value,
            )
            return
        out[key] = value

    def _rewrite_tail(tail: str, *, session_prefix: str | None) -> str:
        if tail == "n_skipped":
            if session_prefix is None:
                return f"{prefix}/n_skipped"
            return f"{session_prefix}/n_skipped"
        if tail == "robustness_input_core_avg":
            if session_prefix is None:
                return f"{prefix}/joint/core_avg"
            return f"{session_prefix}/joint/core_avg"
        if tail == "robustness_input_per_dim_core_avg":
            if session_prefix is None:
                return f"{prefix}/per_dim/core_avg"
            return f"{session_prefix}/per_dim/core_avg"
        if tail.startswith("robustness_input_per_dim/"):
            suffix = tail[len("robustness_input_per_dim/") :]
            if session_prefix is None:
                return f"{prefix}/per_dim/{suffix}"
            return f"{session_prefix}/per_dim/{suffix}"
        if tail.startswith("robustness_input/"):
            suffix = tail[len("robustness_input/") :]
            if session_prefix is None:
                return f"{prefix}/joint/{suffix}"
            return f"{session_prefix}/joint/{suffix}"
        if session_prefix is None:
            return f"{prefix}/{tail}"
        return f"{session_prefix}/{tail}"

    out: dict[str, float] = {}
    session_root = f"{prefix}/per_session/"
    for key, value in metrics.items():
        if key.startswith(session_root):
            rest = key[len(session_root) :]
            safe_sid, sep, tail = rest.partition("/")
            if not sep:
                raise ValueError(
                    f"Malformed input_robustness (M1) per-session metric key: {key!r}"
                )
            new_key = _rewrite_tail(tail, session_prefix=f"{prefix}/session/{safe_sid}")
        elif key.startswith(f"{prefix}/"):
            tail = key[len(f"{prefix}/") :]
            new_key = _rewrite_tail(tail, session_prefix=None)
        else:
            new_key = key
        _insert(out, sanitize_metric_name_for_mlflow(new_key), float(value))
    return out


def flatten_latent_robustness_jsonl_to_metric_series(
    jsonl_path: Path,
    prefix: str = "eval_m2",
) -> tuple[list[StepPoint], dict[str, float]]:
    """Flatten a latent_robustness (M2) ``results.jsonl`` into MLflow series + scalars.

    Two things are emitted, both keyed under ``prefix`` (typically
    ``eval_m2``):

    1. **Summary scalars** — currently only ``n_records`` and
       ``n_skipped``. Per-(session, metric, epsilon) scalars come
       from ``flatten_report_to_metrics`` applied to per-session reports
       (which encode per-epsilon metrics via keys like
       ``<metric>/eps_<value>``); this raw-row flattener emits only the
       summary ``n_records``/``n_skipped`` so the same score is not
       logged twice under different key layouts.

    2. **Per-(session, metric) step-indexed series** — the curve across
       epsilons, with ``step = int(round(epsilon * 10000))``. MLflow
       only accepts integer ``step`` values, so we encode epsilon as a
       fixed-point integer (multiplier 10000 keeps every default
       epsilon distinct down to 0.0001). Consumers that read
       ``metric_history`` see the value at each epsilon directly via
       ``step``; no parallel ``_x_epsilon`` curve is needed.

    """
    records = _read_jsonl(jsonl_path)

    groups: dict[tuple[str, str], dict[float, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    n_records = 0
    n_skipped = 0
    for rec in records:
        if rec.get("status") != "ok":
            n_skipped += 1
            continue
        n_records += 1
        v = _safe_float(rec.get("raw_value"))
        if v is None:
            continue
        eps = _safe_float(rec.get("epsilon"))
        if eps is None:
            continue
        key = (rec["session_id"], rec["metric"])
        groups[key][eps].append(v)

    scalars: dict[str, float] = {
        f"{prefix}/n_records": float(n_records),
        f"{prefix}/n_skipped": float(n_skipped),
    }
    series: list[StepPoint] = []
    for (session_id, metric), per_eps in groups.items():
        curve_name = sanitize_metric_name_for_mlflow(f"{prefix}/{session_id}/{metric}")
        for eps, vals in sorted(per_eps.items(), key=lambda kv: kv[0]):
            mean = float(np.mean(vals))
            # Curve point (step encodes epsilon as fixed-point int).
            series.append((curve_name, mean, int(round(eps * 10000))))
    return series, scalars


def log_step_series_to_mlflow(series: Iterable[StepPoint], *, batched: bool = True) -> None:
    """Forward ``(name, value, step)`` tuples to MLflow.

    When ``batched`` is true, uses ``MlflowClient.log_batch`` in chunks of up
    to ``_MLFLOW_LOG_BATCH_MAX`` points. When false, uses one
    ``MlflowClient.log_metric`` call per point. Semantics (metric name, step,
    value) match the previous per-step loop exactly. Empty input is skipped.
    Requires an active MLflow run (matches prior behavior — ``mlflow.log_metric``
    also relied on the active run context, e.g. via ``mlflow.start_run()``).
    """
    if mlflow is None or Metric is None or MlflowClient is None:
        raise ImportError("mlflow is required to log step series.")
    # Materialize once so we can both check emptiness and chunk through it
    # without consuming a one-shot iterable twice.
    points = list(series)
    if not points:
        return
    active = mlflow.active_run()
    if active is None:
        raise RuntimeError(
            "log_step_series_to_mlflow requires an active MLflow run; call "
            "inside ``with mlflow.start_run(): ...``."
        )
    run_id = active.info.run_id
    client = MlflowClient()
    timestamp_ms = int(time.time() * 1000)

    if not batched:
        # Fallback path: one HTTPS call per (key, value, step). Far slower
        # than log_batch; only enable if the batched path starts erroring
        # with many HTTP 500s. Batch size, not batching itself, is what
        # triggers those errors, so keep the batched path as the default.
        for name, value, step in points:
            client.log_metric(
                run_id=run_id,
                key=name,
                value=float(value),
                timestamp=timestamp_ms,
                step=int(step),
            )
        return

    metrics: list[Any] = [
        Metric(key=name, value=float(value), timestamp=timestamp_ms, step=int(step))
        for name, value, step in points
    ]

    for i in range(0, len(metrics), _MLFLOW_LOG_BATCH_MAX):
        chunk = metrics[i : i + _MLFLOW_LOG_BATCH_MAX]
        if chunk:
            _log_metric_batch_with_backoff(client=client, run_id=run_id, metrics=chunk)


def _log_curves_batched(
    curves: dict[str, list[tuple[int, float]]],
) -> None:
    """Log temporal curves to the active MLflow run via batched ``log_batch``.

    Replaces ``for step, value in curve: mlflow.log_metric(...)`` (one HTTPS
    request per step — the source of the 500-storm) with a single
    ``MlflowClient.log_batch`` per chunk of up to ``_MLFLOW_LOG_BATCH_MAX``
    points. Empty curves are skipped. Semantics (metric name, step, value)
    match the previous per-step loop exactly.
    """
    if mlflow is None or Metric is None or MlflowClient is None:
        raise ImportError("mlflow is required for batched curve logging.")
    if not curves:
        return
    active = mlflow.active_run()
    if active is None:
        raise RuntimeError(
            "_log_curves_batched requires an active MLflow run; call inside "
            "``with mlflow.start_run(): ...``."
        )
    run_id = active.info.run_id
    client = MlflowClient()
    timestamp_ms = int(time.time() * 1000)

    metrics: list[Any] = []
    for curve_name, points in curves.items():
        if not points:
            continue
        for step, value in points:
            metrics.append(
                Metric(
                    key=curve_name,
                    value=float(value),
                    timestamp=timestamp_ms,
                    step=int(step),
                )
            )

    for i in range(0, len(metrics), _MLFLOW_LOG_BATCH_MAX):
        chunk = metrics[i : i + _MLFLOW_LOG_BATCH_MAX]
        if chunk:
            _log_metric_batch_with_backoff(client=client, run_id=run_id, metrics=chunk)


def _log_metric_batch_with_backoff(
    client: Any,
    run_id: str,
    metrics: list[Any],
) -> None:
    """Log one metric batch, recursively splitting on transient server failures.

    Some MLflow deployments intermittently return HTTP 500 for larger
    ``log_batch`` payloads even when they are below the documented server-side
    limit. Keep semantics identical but reduce request size adaptively before
    giving up.
    """
    try:
        client.log_batch(run_id=run_id, metrics=metrics)
        return
    except Exception as exc:
        is_mlflow_exc = MlflowException is not None and isinstance(exc, MlflowException)
        if not is_mlflow_exc:
            raise
        if len(metrics) <= 1:
            metric = metrics[0]
            log.warning(
                "MLflow log_batch failed for a single metric %s; retrying via log_metric.",
                metric.key,
            )
            client.log_metric(
                run_id=run_id,
                key=metric.key,
                value=metric.value,
                timestamp=metric.timestamp,
                step=metric.step,
            )
            return
        mid = len(metrics) // 2
        if mid <= 0 or mid >= len(metrics):
            raise
        log.warning(
            "MLflow log_batch failed for %d metrics; retrying as two smaller batches.",
            len(metrics),
        )
        _log_metric_batch_with_backoff(client=client, run_id=run_id, metrics=metrics[:mid])
        _log_metric_batch_with_backoff(client=client, run_id=run_id, metrics=metrics[mid:])


def _log_artifacts_best_effort(
    *,
    artifact_paths: Iterable[Path],
    artifact_root: Path | None,
    checkpoint_local_path: Path | None,
    upload_checkpoint: bool,
) -> None:
    """Upload artifacts for the active run, spacing requests slightly.

    This is intentionally best-effort at the caller level: callers catch and
    record failures so metrics remain durable even when artifact storage is
    flaky.
    """
    artifact_list = [Path(path) for path in artifact_paths if path is not None]
    if upload_checkpoint and checkpoint_local_path is not None:
        artifact_list.append(Path(checkpoint_local_path))

    for idx, p in enumerate(artifact_list):
        if not p.exists():
            continue
        if mlflow is None:
            raise ImportError("mlflow is required to log artifacts.")
        if (
            upload_checkpoint
            and checkpoint_local_path is not None
            and p == Path(checkpoint_local_path)
        ):
            mlflow.log_artifact(str(p), artifact_path="checkpoints")
        else:
            artifact_path = None
            if artifact_root is not None:
                try:
                    rel_parent = p.parent.relative_to(artifact_root)
                except ValueError:
                    rel_parent = None
                if rel_parent is not None and str(rel_parent) != ".":
                    artifact_path = rel_parent.as_posix()
            if artifact_path is None:
                mlflow.log_artifact(str(p))
            else:
                mlflow.log_artifact(str(p), artifact_path=artifact_path)
        if idx + 1 < len(artifact_list):
            time.sleep(_MLFLOW_ARTIFACT_UPLOAD_PAUSE_SECONDS)


class MLflowLoggingConfig(pdt.BaseModel):
    """Configuration for MLflow evaluation logging."""

    enabled: bool = False
    experiment_name: str = "ntt-eval"
    tracking_uri: str | None = None
    run_name: str | None = None
    # Appended to ``run_name`` to disambiguate parallel passes (main / snr /
    # m1 / m2) that share the same checkpoint. The four passes get separate
    # MLflow runs all tagged with the same ``ntt.eval.checkpoint_path`` so
    # they're discoverable from one another.
    run_name_suffix: str = ""
    # Top-level prefix for metric names and the comparison block, e.g.
    # "eval", "eval_snr", "eval_m1", "eval_m2". Allows secondary passes to
    # avoid colliding with the primary pass's metric tab.
    metric_prefix: str = "eval"
    tags: dict[str, str] = pdt.Field(default_factory=dict)
    # When the source ckpt is already addressable via an ``mlflow-artifacts:/``
    # URI, the default is to leave the bytes where they are (the run is still
    # linked via ``ntt.eval.training_run_id`` tag). Flip to True to duplicate
    # the bytes into the eval run's artifacts — useful for cross-machine
    # setups where the training run's artifact store may not be reachable.
    always_copy_checkpoint: bool = False
    # Whether this pass should upload the checkpoint bytes as an artifact.
    # The primary pass uploads (default); secondary passes (snr, m1, m2)
    # set False to avoid duplicating the bytes — they still tag the run
    # with ``ntt.eval.checkpoint_path`` so the linkage is preserved.
    upload_checkpoint: bool = True
    # Robustness runs can optionally upload extra local files useful for
    # replotting / audit (e.g. raw JSONL / per-session CSV). Disabled by
    # default to keep the remote artifact set compact and robust.
    upload_artifacts_for_replotting: bool = False


def remap_latent_robustness_metric_keys_for_mlflow(
    metrics: dict[str, float],
    *,
    prefix: str = "eval_m2",
) -> dict[str, float]:
    """Rewrite verbose latent_robustness (M2) report keys into compact ``session/<sid>/`` keys.

    Mirrors the input_robustness (M1) cleanup: per-session metrics move from
    ``{prefix}/per_session/<sid>/<metric>/eps_<v>`` to
    ``{prefix}/session/<sid>/<metric>/eps_<v>`` so latent_robustness (M2) uses the same
    ``session/<sid>/`` hierarchy as input_robustness (M1).

    Aggregate keys (``{prefix}/<metric>/eps_<v>``) and summary scalars are left
    untouched on purpose, so the sheet's exact-key lookup
    ``eval_m2/smoothed_squared_error/eps_0.03`` keeps resolving. latent_robustness (M2) has no
    joint/per-dim split, so there is nothing analogous to input_robustness (M1)'s
    ``robustness_input`` / ``robustness_input_per_dim`` regrouping.
    """

    def _insert(out: dict[str, float], key: str, value: float) -> None:
        if key in out and not np.isclose(out[key], value):
            raise ValueError(
                "Conflicting latent_robustness (M2) MLflow metric remap for key "
                f"{key!r}: existing={out[key]!r}, new={value!r}"
            )
        out[key] = value

    out: dict[str, float] = {}
    session_root = f"{prefix}/per_session/"
    for key, value in metrics.items():
        if key.startswith(session_root):
            new_key = f"{prefix}/session/" + key[len(session_root) :]
        else:
            new_key = key
        _insert(out, sanitize_metric_name_for_mlflow(new_key), float(value))
    return out


def _find_nested_key(obj: Any, key: str) -> Any | None:
    """Find the first value for ``key`` in a nested dict/list structure."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = _find_nested_key(value, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _find_nested_key(value, key)
            if found is not None:
                return found
    return None


def _mlflow_param_value(value: Any) -> str | int | float | bool | None:
    """Return a scalar MLflow param value, stringifying path-like objects."""
    if value is None:
        return None
    if isinstance(value, str | int | float | bool):
        return value
    return str(value)


def _val_dataset_identity(data_config: Any) -> tuple[str | None, int | None]:
    """Derive (val_brainsets, n_val_sessions) from a data-config payload.

    Reads the ``val_selections`` structure of an ``NTTDataModuleConfig`` dump.
    Returns ``(None, None)`` when the payload is missing or malformed.
    """
    if not isinstance(data_config, dict):
        return None, None
    val_selections = data_config.get("val_selections")
    if not isinstance(val_selections, list):
        return None, None
    brainsets: set[str] = set()
    n_sessions = 0
    for group in val_selections:
        if not isinstance(group, dict):
            continue
        for entry in group.get("selection") or []:
            if not isinstance(entry, dict):
                continue
            brainset = entry.get("brainset")
            if isinstance(brainset, str):
                brainsets.add(brainset)
            sessions = entry.get("sessions")
            if isinstance(sessions, list):
                n_sessions += len(sessions)
            elif entry.get("session") is not None:
                n_sessions += 1
    brainsets_str = ",".join(sorted(brainsets)) if brainsets else None
    return brainsets_str, (n_sessions if n_sessions else None)


def _robustness_metadata_params(
    *,
    run_metadata: dict[str, Any] | None,
    per_session_reports: dict[str, EvaluationReport] | None,
) -> tuple[dict[str, str | int | float | bool], list[str]]:
    """Extract the hand-picked searchable MLflow params from run metadata."""
    metadata = run_metadata if isinstance(run_metadata, dict) else {}
    eval_config = metadata.get("eval_config")
    data_config = metadata.get("data_config")
    tokenizer_config = metadata.get("tokenizer_config")
    params: dict[str, str | int | float | bool] = {}
    missing: list[str] = []

    val_brainsets, n_val_sessions = _val_dataset_identity(data_config)
    candidates = {
        "eval_window_seconds": _find_nested_key(eval_config, "eval_window_seconds"),
        "n_eval_windows": _find_nested_key(eval_config, "n_eval_windows"),
        # ``NTTDataModuleConfig`` has no selection_name/dataset_yaml fields; the
        # dataset identity lives in ``val_selections`` (brainsets + sessions).
        "data.val_brainsets": val_brainsets,
        "data.n_val_sessions": n_val_sessions,
        "tokenizer.KIND": _find_nested_key(tokenizer_config, "KIND"),
        "tokenizer.token_bin_size_ms": _find_nested_key(
            tokenizer_config,
            "token_bin_size_ms",
        ),
        "n_sessions": metadata.get(
            "n_sessions",
            len(per_session_reports) if per_session_reports is not None else None,
        ),
    }
    for key, value in candidates.items():
        param_value = _mlflow_param_value(value)
        if param_value is None:
            missing.append(key)
            continue
        params[key] = param_value
    return params, missing


def log_robustness_run_to_mlflow(
    *,
    mode: str,
    config: MLflowLoggingConfig,
    jsonl_path: Path,
    report: EvaluationReport | None = None,
    per_session_reports: dict[str, EvaluationReport] | None = None,
    label_to_type_strength: dict[str, tuple[str, float]] | None = None,
    checkpoint_path: str | None = None,
    checkpoint_local_path: Path | None = None,
    artifact_paths: Iterable[Path] = (),
    artifact_root: Path | None = None,
    include_jsonl_artifact: bool = True,
    extra_tags: dict[str, str] | None = None,
    extra_metrics: dict[str, float] | None = None,
    run_metadata: dict[str, Any] | None = None,
) -> str | None:
    """Open one MLflow run and log input_/latent_robustness (M1/M2) sweep results + artifacts."""
    if not config.enabled:
        return None
    if mode not in ("m1", "m2"):
        raise ValueError(f"mode must be 'm1' or 'm2', got {mode!r}")
    if mlflow is None:
        raise ImportError("mlflow is required for robustness MLflow logging.")

    if config.tracking_uri is not None:
        mlflow.set_tracking_uri(config.tracking_uri)
    mlflow.set_experiment(config.experiment_name)

    composed_run_name: str | None
    if config.run_name is None and not config.run_name_suffix:
        composed_run_name = None
    else:
        composed_run_name = f"{config.run_name or ''}{config.run_name_suffix}"

    if mode == "m1":
        if label_to_type_strength is None:
            raise ValueError("label_to_type_strength is required for mode='m1'")
        series, scalars = flatten_input_robustness_jsonl_to_metric_series(
            jsonl_path, label_to_type_strength, prefix=config.metric_prefix
        )
        jsonl_summary_metrics = extract_input_robustness_jsonl_summary_metrics(
            jsonl_path,
            prefix=config.metric_prefix,
        )
    else:
        series, scalars = flatten_latent_robustness_jsonl_to_metric_series(
            jsonl_path, prefix=config.metric_prefix
        )
        jsonl_summary_metrics = {}

    with mlflow.start_run(run_name=composed_run_name) as run:
        run_id = run.info.run_id

        # Set associability tags FIRST so the run is discoverable by the
        # downstream sheet-export tooling (which filters runs by these tags)
        # even if metric / artifact logging below raises. If
        # ``flatten_report_to_metrics`` raises on illegal mlflow
        # metric names, an untagged run would be invisible to the tag-filter
        # search; doing tags first prevents
        # that orphaning.
        all_tags: dict[str, str] = {"ntt.eval.pass": mode}
        if checkpoint_path is not None:
            all_tags["ntt.eval.checkpoint_path"] = checkpoint_path
            training_run_id = parse_training_run_id(checkpoint_path)
            if training_run_id is not None:
                all_tags["ntt.eval.training_run_id"] = training_run_id
        if config.tags:
            all_tags.update(config.tags)
        if extra_tags:
            all_tags.update(extra_tags)
        mlflow.set_tags(all_tags)
        metadata_params, missing_param_keys = _robustness_metadata_params(
            run_metadata=run_metadata,
            per_session_reports=per_session_reports,
        )
        if metadata_params:
            mlflow.log_params(metadata_params)
        if missing_param_keys:
            log.warning(
                "Robustness MLflow metadata params absent and skipped: %s",
                sorted(missing_param_keys),
            )

        if mode == "m1":
            # Trust order for the first-wins merge: the on-disk jsonl is the
            # source of truth (it backs the local sheet backfill), so its
            # summary + series scalars outrank the report-derived paths.
            metric_sources: list[tuple[str, dict[str, float]]] = [
                ("jsonl_summary", jsonl_summary_metrics),
                ("jsonl_series_scalars", scalars),
            ]
            if report is not None:
                aggregate_metrics = remap_input_robustness_metric_keys_for_mlflow(
                    flatten_report_to_metrics(report, prefix=config.metric_prefix),
                    prefix=config.metric_prefix,
                )
                aggregate_metrics.pop(
                    sanitize_metric_name_for_mlflow(f"{config.metric_prefix}/n_skipped"),
                    None,
                )
                metric_sources.append(("report_aggregate", aggregate_metrics))
            if per_session_reports:
                for sid, session_report in per_session_reports.items():
                    safe_sid = sid.replace("/", "__")
                    metric_sources.append(
                        (
                            f"per_session_report/{safe_sid}",
                            remap_input_robustness_metric_keys_for_mlflow(
                                flatten_report_to_metrics(
                                    session_report,
                                    prefix=f"{config.metric_prefix}/per_session/{safe_sid}",
                                ),
                                prefix=config.metric_prefix,
                            ),
                        )
                    )
            combined_metrics, merge_conflicts = merge_input_robustness_metric_sources(
                metric_sources
            )
            if merge_conflicts:
                # A conflict means two logging paths disagree on the same key —
                # a bug to investigate, but not one worth aborting a completed
                # eval over. The trusted (jsonl) value was kept; make the run
                # discoverable via tag and leave the full evidence in the log.
                log.error(
                    "input_robustness (M1) MLflow metric merge conflicts "
                    "(most-trusted source wins): %s",
                    merge_conflicts,
                )
                mlflow.set_tag(
                    "ntt.eval.m1_metric_merge_conflicts",
                    # MLflow caps tag values at 5000 chars on some backends.
                    ",".join(sorted(merge_conflicts))[:4900],
                )
            if combined_metrics:
                mlflow.log_metrics(combined_metrics)
            # input_robustness (M1) does not log the per-step curve ``series``: it duplicates the
            # compact scalars logged above. (The same dedup applies to latent_robustness (M2)
            # below.)
        else:
            # latent_robustness (M2): collect the aggregate + per-session report metrics, then
            # rewrite them into the compact ``session/<sid>/`` layout (mirrors
            # the input_robustness (M1) cleanup). Aggregate keys are unchanged so the sheet's exact
            # ``eval_m2/smoothed_squared_error/eps_0.03`` lookup still resolves.
            if scalars:
                mlflow.log_metrics(scalars)
            combined_metrics = {}
            if report is not None:
                combined_metrics.update(
                    flatten_report_to_metrics(report, prefix=config.metric_prefix)
                )
            if per_session_reports:
                for sid, session_report in per_session_reports.items():
                    safe_sid = sid.replace("/", "__")
                    combined_metrics.update(
                        flatten_report_to_metrics(
                            session_report,
                            prefix=f"{config.metric_prefix}/per_session/{safe_sid}",
                        )
                    )
            if combined_metrics:
                mlflow.log_metrics(
                    remap_latent_robustness_metric_keys_for_mlflow(
                        combined_metrics, prefix=config.metric_prefix
                    )
                )
            # The per-(session, metric) epsilon curve in ``series`` duplicates the
            # per-epsilon scalars just logged under
            # ``eval_m2/session/<sid>/<metric>/eps_<v>``, so it is not logged.
        if extra_metrics:
            mlflow.log_metrics(extra_metrics)
        mlflow.set_tag("ntt.eval.metric_upload_status", "ok")

        report_artifact_cm = tempfile.TemporaryDirectory(prefix="ntt_m2_eval_report_")
        with report_artifact_cm as report_tmpdir:
            artifact_inputs: list[Path] = []
            if include_jsonl_artifact:
                artifact_inputs.append(jsonl_path)
            artifact_inputs.extend(Path(path) for path in artifact_paths if path is not None)
            if report is not None:
                # Prefer the canonical in-memory report over any stale on-disk
                # eval_report.json that may have been written by older code.
                artifact_inputs = [p for p in artifact_inputs if p.name != "eval_report.json"]
                synthesized_report_path = Path(report_tmpdir) / "eval_report.json"
                synthesized_report_path.write_text(json.dumps(report.to_dict(), indent=2))
                artifact_inputs.append(synthesized_report_path)
            if artifact_inputs or (config.upload_checkpoint and checkpoint_local_path is not None):
                try:
                    _log_artifacts_best_effort(
                        artifact_paths=artifact_inputs,
                        artifact_root=artifact_root,
                        checkpoint_local_path=checkpoint_local_path,
                        upload_checkpoint=config.upload_checkpoint,
                    )
                    mlflow.set_tag("ntt.eval.artifact_upload_status", "ok")
                except Exception:
                    log.warning(
                        "MLflow artifact upload failed after metrics were already logged.",
                        exc_info=True,
                    )
                    try:
                        mlflow.set_tag("ntt.eval.artifact_upload_status", "failed")
                    except Exception:
                        log.warning(
                            "Failed to set MLflow artifact failure tag after upload error.",
                            exc_info=True,
                        )
            else:
                mlflow.set_tag("ntt.eval.artifact_upload_status", "skipped")

    return run_id


class MLflowEvalLogger:
    """Logs EvaluationReport results to an MLflow experiment."""

    def __init__(
        self,
        experiment_name: str = "ntt-eval",
        tracking_uri: str | None = None,
        run_name: str | None = None,
        run_name_suffix: str = "",
        metric_prefix: str = "eval",
    ):
        """
        ``metric_prefix`` is the top-level namespace for logged metrics
        (``eval`` for the primary pass; ``eval_snr`` / ``eval_m1`` /
        ``eval_m2`` for secondary passes so they don't collide).

        ``run_name_suffix`` is appended to ``run_name`` to give each pass its
        own MLflow run while keeping a shared base name for discovery.
        """
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.run_name = run_name
        self.run_name_suffix = run_name_suffix
        self.metric_prefix = metric_prefix

    def log(
        self,
        report: EvaluationReport,
        per_session_reports: dict[str, EvaluationReport] | None = None,
        tier_reports: dict[str, EvaluationReport] | None = None,
        comparisons: dict[str, MetricComparison] | None = None,
        generalization: dict | None = None,
        extra_artifacts: list[Path] | None = None,
        checkpoint_path: str | None = None,
        tags: dict[str, str] | None = None,
        checkpoint_local_path: Path | None = None,
    ) -> str:
        """Log evaluation results to MLflow. Returns the MLflow run ID.

        ``checkpoint_local_path`` (when provided) is uploaded as an artifact
        under ``checkpoints/`` in the run. The caller is responsible for
        resolving or synthesizing that path (see
        ``ntt.core.model.checkpoint_io.resolve_checkpoint_for_eval``).

        ``generalization`` (the retention-score block) is logged both as
        ``eval/generalization/*`` metrics and as a ``generalization.json``
        artifact. ``extra_artifacts`` are local files (e.g. the full
        ``results.json`` / ``run_meta.json``) mirrored to the run so nothing
        saved locally is lost when callers rely on MLflow for storage.
        """
        if mlflow is None:
            raise ImportError(
                "mlflow is required for MLflowEvalLogger. Install it with: pip install mlflow"
            )

        if self.tracking_uri is not None:
            mlflow.set_tracking_uri(self.tracking_uri)

        mlflow.set_experiment(self.experiment_name)

        composed_run_name: str | None
        if self.run_name is None and not self.run_name_suffix:
            composed_run_name = None
        else:
            composed_run_name = f"{self.run_name or ''}{self.run_name_suffix}"

        with mlflow.start_run(run_name=composed_run_name) as run:
            run_id = run.info.run_id

            # -- Tags (set FIRST so the run is associable even if metric
            #    or artifact logging below raises). The downstream
            #    sheet-export tooling filters runs by these tags; an
            #    untagged run is invisible. The tag
            #    set here intentionally matches what eval.py has always
            #    produced — no new tags are introduced for the main /
            #    SNR passes; only the order changed (set BEFORE metric
            #    logging instead of after).
            all_tags: dict[str, str] = {}
            if checkpoint_path is not None:
                all_tags["ntt.eval.checkpoint_path"] = checkpoint_path
                training_run_id = parse_training_run_id(checkpoint_path)
                if training_run_id is not None:
                    all_tags["ntt.eval.training_run_id"] = training_run_id
            if per_session_reports is not None:
                all_tags["ntt.eval.n_sessions"] = str(len(per_session_reports))
            if tags:
                all_tags.update(tags)
            if all_tags:
                mlflow.set_tags(all_tags)

            # -- Metrics --
            flat_metrics = flatten_report_to_metrics(report, prefix=self.metric_prefix)

            if comparisons:
                for name, comp in comparisons.items():
                    if comp.normalized_score is not None and comp.status == "ok":
                        flat_metrics[f"{self.metric_prefix}/normalized/{name}"] = float(
                            comp.normalized_score
                        )

            mlflow.log_metrics(flat_metrics)

            # -- Temporal curves --
            # Metrics with _{N}ms suffixes are logged as stepped series so
            # MLflow renders them as line plots (x=bin_size, y=value).
            # Batched via ``MlflowClient.log_batch`` to avoid one HTTPS
            # request per step (one request per step floods the tracking server
            # and triggers ``too many 500 error responses``).
            _log_curves_batched(extract_temporal_curves(report))

            # Per-session temporal curves
            if per_session_reports:
                for sid, session_report in per_session_reports.items():
                    safe_sid = sid.replace("/", "__")
                    session_curves = extract_temporal_curves(
                        session_report,
                        prefix=f"eval/temporal/{safe_sid}",
                    )
                    _log_curves_batched(session_curves)

            # -- Per-tier metrics and temporal curves --
            if tier_reports:
                for tier_name, tier_report in tier_reports.items():
                    tier_metrics = flatten_report_to_metrics(
                        tier_report, prefix=f"eval/tier/{tier_name}"
                    )
                    mlflow.log_metrics(tier_metrics)

                    tier_curves = extract_temporal_curves(
                        tier_report,
                        prefix=f"eval/temporal/tier/{tier_name}",
                    )
                    _log_curves_batched(tier_curves)

            # -- Generalization retention score --
            if generalization:
                gen_metrics = _flatten_generalization_metrics(generalization)
                if gen_metrics:
                    mlflow.log_metrics(gen_metrics)

            # -- Artifacts --
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)

                # Aggregated report
                report_path = tmpdir_path / "eval_report.json"
                report_path.write_text(json.dumps(report.to_dict(), indent=2, default=str))
                mlflow.log_artifact(str(report_path))

                # Generalization retention score (full block)
                if generalization:
                    gen_path = tmpdir_path / "generalization.json"
                    gen_path.write_text(json.dumps(generalization, indent=2, default=str))
                    mlflow.log_artifact(str(gen_path))

                # Extra local artifacts mirrored to MLflow (e.g. results.json,
                # run_meta.json) so the local-save gate never loses results.
                if extra_artifacts:
                    for artifact_path in extra_artifacts:
                        if artifact_path is not None and Path(artifact_path).exists():
                            mlflow.log_artifact(str(artifact_path))

                # Per-session reports
                if per_session_reports:
                    session_dir = tmpdir_path / "per_session"
                    session_dir.mkdir()
                    for sid, session_report in per_session_reports.items():
                        # Session IDs may contain ``/`` (e.g.
                        # brainset/session); flatten to a filesystem-safe
                        # filename for the artifact upload.
                        safe_sid = sid.replace("/", "__")
                        session_path = session_dir / f"{safe_sid}.json"
                        session_path.write_text(
                            json.dumps(session_report.to_dict(), indent=2, default=str)
                        )
                        mlflow.log_artifact(str(session_path), artifact_path="per_session")

                # Comparisons
                if comparisons:
                    comp_path = tmpdir_path / "poisson_comparison.json"
                    comp_path.write_text(
                        json.dumps(
                            {n: c.to_dict() for n, c in comparisons.items()},
                            indent=2,
                            default=str,
                        )
                    )
                    mlflow.log_artifact(str(comp_path), artifact_path="comparisons")

                # Checkpoint (resolved or synthesized by the caller)
                if checkpoint_local_path is not None:
                    mlflow.log_artifact(str(checkpoint_local_path), artifact_path="checkpoints")

        return run_id
