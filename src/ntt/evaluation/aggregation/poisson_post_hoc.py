"""Post-hoc Poisson normalization for existing eval runs.

Use case: a model run and a Poisson run were evaluated independently (neither
was launched with ``job.poisson_reference=...``). After the fact you want
Poisson-normalized scores for plotting or analysis. This module gives that —
**read-only**; nothing is written, no result file is overwritten.

Three public entry points, one per eval mode:

- ``normalize_m1_results_against_poisson`` — input-corruption robustness.
- ``normalize_m2_results_against_poisson`` — latent-noise robustness.
- ``compare_snr_against_poisson`` — RepeatsSNR. **No ceiling normalization**:
  SNR is a ratio and can be arbitrarily high at low noise, so a
  ``(model - poisson) / (ceiling - poisson)`` formula has no natural ceiling.
  Returns the raw, direction-corrected ``model − poisson`` delta per metric.

Each function reads the per-session reports written by the corresponding
eval job: ``per_session/*.json`` for M1/M2, and ``results.csv`` for SNR
(the main eval pipeline writes per-segment rows there). **No reliance on
``poisson_comparison.json`` or ``results.json``** — the first only exists
for M1/M2 runs already launched with ``job.poisson_reference`` (which the
back-fix case is meant to handle the absence of), and the second is empty
for SNR runs in practice. The SNR helper takes ``higher_is_better`` as a
parameter (default True, matching RepeatsSNR) so we don't need to read
metric metadata from anywhere.

For M1/M2, both normalization flavors are produced — ``aggregate_first`` (the
main reconstruction-eval pipeline's default) and ``per_session_then_mean``
(normalize each session against its own poisson + ceiling, then mean across
sessions). The caller picks one via ``aggregation_order``; the other flavor is
also surfaced in the returned dict for inspection.

Note on duplication: the per-session-JSON loading + ceiling synthesis here
mirror the logic in ``NTTEvalRobustnessInputJob._write_poisson_comparison_artifact``
and the M2 equivalent. Both sides aggregate model / poisson / ceiling over
the **shared** session subset before comparing, so when sessions overlap
perfectly the post-hoc and live results agree bit-equal; when they don't,
both honestly report the shared-population comparison (and warn that some
sessions were dropped). When changing the normalization semantics, both sides
must be updated.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Literal

import numpy as np

from ..core.types import EvaluationReport, MetricResult
from .comparison import (
    MetricComparison,
    compute_normalized_comparisons,
    extract_normalized_scores,
)

log = logging.getLogger(__name__)


AggregationOrder = Literal["aggregate_first", "per_session_then_mean"]


# ────────────────────────── public surface ──────────────────────────


def normalize_m1_results_against_poisson(
    *,
    model_results_dir: Path | str,
    poisson_results_dir: Path | str,
    aggregation_order: AggregationOrder = "aggregate_first",
) -> dict:
    """Post-hoc Poisson normalization for an existing pair of M1 runs.

    ``model_results_dir`` and ``poisson_results_dir`` each must contain the
    ``per_session/`` directory and an ``eval_report.json`` that an M1 job
    writes. The ceiling is synthesized from the **model**'s own
    zero-corruption metric rows (same as the live job path:
    ``_zero_strength_m1_label`` collapses ``max_shift_ms=…``, ``k_ms=…`` and
    ``fraction=…`` sub-labels to their zero values).

    Returns a dict with the same top-level shape as ``poisson_comparison.json``,
    plus an ``aggregation_order`` field naming which flavor the caller asked
    for. Nothing is written.
    """
    model_per_session = _load_per_session_reports(Path(model_results_dir))
    poisson_per_session = _load_per_session_reports(Path(poisson_results_dir))
    ceiling_per_session = {
        sid: _m1_ceiling_report(report) for sid, report in model_per_session.items()
    }
    return _compose_comparison_payload(
        model_per_session=model_per_session,
        poisson_per_session=poisson_per_session,
        ceiling_per_session=ceiling_per_session,
        model_results_dir=Path(model_results_dir),
        poisson_results_dir=Path(poisson_results_dir),
        aggregation_order=aggregation_order,
        mode="M1",
    )


def normalize_m2_results_against_poisson(
    *,
    model_results_dir: Path | str,
    poisson_results_dir: Path | str,
    aggregation_order: AggregationOrder = "aggregate_first",
) -> dict:
    """Post-hoc Poisson normalization for an existing pair of M2 runs.

    Identical to ``normalize_m1_results_against_poisson`` except the ceiling
    lookup uses ``<metric>/eps_0`` (epsilon=0 row) instead of the M1
    sub-label collapse.
    """
    model_dir = Path(model_results_dir)
    poisson_dir = Path(poisson_results_dir)
    model_per_session = _load_per_session_reports(model_dir)
    poisson_per_session = _load_per_session_reports(poisson_dir)
    ceiling_per_session = {
        sid: _m2_ceiling_report(report) for sid, report in model_per_session.items()
    }
    return _compose_comparison_payload(
        model_per_session=model_per_session,
        poisson_per_session=poisson_per_session,
        ceiling_per_session=ceiling_per_session,
        model_results_dir=model_dir,
        poisson_results_dir=poisson_dir,
        aggregation_order=aggregation_order,
        mode="M2",
    )


def compare_snr_against_poisson(
    *,
    model_results_dir: Path | str,
    poisson_results_dir: Path | str,
    metric_columns: list[str] | None = None,
    higher_is_better: bool = True,
    aggregation_order: AggregationOrder = "aggregate_first",
) -> dict:
    """Post-hoc model-vs-Poisson SNR comparison (no ceiling).

    Reads ``results.csv`` (per-segment, written by the main eval pipeline)
    from each run. SNR is unbounded above when noise is low, so there is no
    natural ceiling to normalize against — this function returns the raw
    delta ``model − poisson`` (sign-flipped to ``poisson − model`` when
    ``higher_is_better=False``) per metric column. Positive means model is
    better.

    ``metric_columns``: list of CSV column names to compare. Defaults to
    ``["repeats_snr"]`` — the RepeatsSNR metric. Pass other column names
    explicitly if comparing other higher-is-better main-eval metrics.

    ``higher_is_better``: applies uniformly to the metrics in
    ``metric_columns``. RepeatsSNR is higher-is-better, hence the default.

    Returns a dict with ``aggregate_delta``, ``per_session_delta``,
    ``per_session_mean_delta``, and ``delta_summary`` (the chosen flavor
    selected via ``aggregation_order``). Nothing is written.

    Note: this helper deliberately does NOT read ``results.json`` for
    metric metadata — SNR runs typically have an empty
    ``results.json/results`` block, and we want the back-fix to work on
    those untouched. Direction is supplied as a parameter instead.
    """
    model_dir = Path(model_results_dir)
    poisson_dir = Path(poisson_results_dir)
    if metric_columns is None:
        metric_columns = ["repeats_snr"]

    model_per_session = _load_main_eval_metric_means_per_session(model_dir, metric_columns)
    poisson_per_session = _load_main_eval_metric_means_per_session(poisson_dir, metric_columns)

    shared_sessions = sorted(set(model_per_session) & set(poisson_per_session))
    if not shared_sessions:
        log.warning(
            "compare_snr_against_poisson: no overlap between model sessions "
            "(%d) and poisson sessions (%d). per_session_delta will be empty.",
            len(model_per_session),
            len(poisson_per_session),
        )

    sign = 1.0 if higher_is_better else -1.0

    per_session_delta: dict[str, dict[str, float]] = {}
    per_session_collected: dict[str, list[float]] = {}
    for sid in shared_sessions:
        m = model_per_session[sid]
        p = poisson_per_session[sid]
        deltas: dict[str, float] = {}
        for metric in metric_columns:
            mv = m.get(metric)
            pv = p.get(metric)
            if mv is None or pv is None:
                continue
            if not (np.isfinite(mv) and np.isfinite(pv)):
                continue
            deltas[metric] = float(sign * (mv - pv))
        per_session_delta[sid] = deltas
        for metric_name, value in deltas.items():
            per_session_collected.setdefault(metric_name, []).append(value)

    # Aggregate-first: average each side over the shared subset, then delta.
    # Restrict to the *paired* subset per metric — sessions where both sides
    # have a finite value — so the two means describe the same session
    # population (filtering each side independently could otherwise average
    # model and poisson over different sessions for a metric missing on one
    # side, re-introducing the population mismatch this module avoids).
    aggregate_delta: dict[str, float] = {}
    if shared_sessions:
        for metric in metric_columns:
            paired = [
                (model_per_session[sid][metric], poisson_per_session[sid][metric])
                for sid in shared_sessions
                if metric in model_per_session[sid]
                and metric in poisson_per_session[sid]
                and np.isfinite(model_per_session[sid][metric])
                and np.isfinite(poisson_per_session[sid][metric])
            ]
            if not paired:
                continue
            m_vals = [mv for mv, _ in paired]
            p_vals = [pv for _, pv in paired]
            aggregate_delta[metric] = float(sign * (np.mean(m_vals) - np.mean(p_vals)))

    per_session_mean_delta = {
        metric_name: float(np.mean(values))
        for metric_name, values in per_session_collected.items()
        if values
    }

    if aggregation_order == "aggregate_first":
        delta_summary = aggregate_delta
    else:
        delta_summary = per_session_mean_delta

    return {
        "model_results_dir": str(model_dir),
        "poisson_results_dir": str(poisson_dir),
        "metric_columns": list(metric_columns),
        "higher_is_better": higher_is_better,
        "aggregation_order": aggregation_order,
        "aggregate_delta": aggregate_delta,
        "per_session_delta": per_session_delta,
        "per_session_mean_delta": per_session_mean_delta,
        "delta_summary": delta_summary,
    }


# ────────────────────────── M1/M2 internals ──────────────────────────


def _compose_comparison_payload(
    *,
    model_per_session: dict[str, EvaluationReport],
    poisson_per_session: dict[str, EvaluationReport],
    ceiling_per_session: dict[str, EvaluationReport],
    model_results_dir: Path,
    poisson_results_dir: Path,
    aggregation_order: AggregationOrder,
    mode: Literal["M1", "M2"],
) -> dict:
    """Run ``compute_normalized_comparisons`` per session and on aggregates,
    then assemble the dict the caller gets back. Mirrors the in-job artifact
    shape so consumers can swap between live and post-hoc results."""
    from ..artifact import aggregate_per_session_dicts
    from ..config.schema import AggregationStrategy

    shared_sessions = sorted(
        set(model_per_session) & set(poisson_per_session) & set(ceiling_per_session)
    )
    if not shared_sessions:
        log.warning(
            "%s post-hoc: no overlap between model sessions (%d), poisson "
            "sessions (%d), and ceiling sessions (%d). per_session is empty "
            "and the aggregate comparison is over disjoint session sets.",
            mode,
            len(model_per_session),
            len(poisson_per_session),
            len(ceiling_per_session),
        )

    per_session_comparisons: dict[str, dict[str, dict]] = {}
    per_session_collected: dict[str, list[float]] = {}
    for sid in shared_sessions:
        comparisons = compute_normalized_comparisons(
            model_per_session[sid],
            poisson_per_session[sid],
            ceiling_per_session[sid],
        )
        if not comparisons:
            continue
        per_session_comparisons[sid] = {name: comp.to_dict() for name, comp in comparisons.items()}
        for metric_name, comp in comparisons.items():
            if comp.normalized_score is None or comp.status != "ok":
                continue
            per_session_collected.setdefault(metric_name, []).append(float(comp.normalized_score))

    # Aggregate-first flavor: aggregate each side then normalize once.
    # Restrict every side to the **shared** session set so the three
    # aggregates describe the same population. The live job path
    # (``_write_poisson_comparison_artifact``) currently aggregates each
    # side over its full session set instead — which biases the
    # aggregate-first number when model and poisson session sets don't
    # fully match (e.g. the poisson run was evaluated on extra sessions
    # the model didn't see). For a clean analysis helper that bias is
    # just noise, so we restrict here. Consequence: when sessions overlap
    # perfectly this matches the live artifact bit-equal; when they don't
    # the two numbers differ in the direction "post-hoc is honest about
    # which sessions it compared, live mixes in non-shared sessions".
    shared_set = set(shared_sessions)
    model_shared = {sid: r for sid, r in model_per_session.items() if sid in shared_set}
    poisson_shared = {sid: r for sid, r in poisson_per_session.items() if sid in shared_set}
    ceiling_shared = {sid: r for sid, r in ceiling_per_session.items() if sid in shared_set}
    model_aggregated = _aggregate_report_dict_via_helper(
        model_shared, aggregate_per_session_dicts, AggregationStrategy.MEAN
    )
    poisson_aggregated = _aggregate_report_dict_via_helper(
        poisson_shared, aggregate_per_session_dicts, AggregationStrategy.MEAN
    )
    ceiling_aggregated = _aggregate_report_dict_via_helper(
        ceiling_shared, aggregate_per_session_dicts, AggregationStrategy.MEAN
    )
    aggregate_comparisons = compute_normalized_comparisons(
        model_aggregated, poisson_aggregated, ceiling_aggregated
    )

    normalized_scores_vs_poisson = extract_normalized_scores(aggregate_comparisons)
    normalized_scores_from_per_session_mean = {
        metric_name: float(np.mean(values))
        for metric_name, values in per_session_collected.items()
        if values
    }

    if aggregation_order == "aggregate_first":
        normalized_summary = normalized_scores_vs_poisson
    else:
        normalized_summary = normalized_scores_from_per_session_mean

    return {
        "model_results_dir": str(model_results_dir),
        "poisson_results_dir": str(poisson_results_dir),
        "mode": mode,
        "aggregation_order": aggregation_order,
        "aggregate_vs_poisson": {
            name: comp.to_dict() for name, comp in aggregate_comparisons.items()
        },
        "normalized_scores_vs_poisson": normalized_scores_vs_poisson,
        "per_session": per_session_comparisons,
        "normalized_scores_from_per_session_mean": normalized_scores_from_per_session_mean,
        "normalized_summary": normalized_summary,
    }


def _aggregate_report_dict_via_helper(
    per_session: dict[str, EvaluationReport],
    aggregate_per_session_dicts,
    strategy,
) -> EvaluationReport:
    """Thin wrapper: ``aggregate_per_session_dicts`` takes dicts of dicts."""
    return aggregate_per_session_dicts(
        {sid: report.to_dict() for sid, report in per_session.items()},
        strategy,
    )


def _m1_ceiling_report(report: EvaluationReport) -> EvaluationReport:
    """Build a per-session ceiling report. Ceiling is pinned to the
    theoretical optimum (M1 NMSE = 0) for every M1 corruption metric.

    M1 is a lower-is-better NMSE in latent space; the only theoretically
    achievable optimum is 0 (the model produces no latent movement under
    the corruption). Using the model's own zero-strength M1 row as the
    ceiling would give deterministic-in-eval tokenizers 0, but stochastic-in-eval
    tokenizers (notably ``DiffusionPriorBottleneck`` with its always-on σ₀
    noise) a ceiling of ~1.4, making the ``(baseline - ceiling)`` denominator
    small relative to its variance and producing normalized scores far outside
    [0, 1].

    Pinning ceiling = 0 strictly across tokenizers:
      - bounds the normalized score at ≤ 1 from above (a normalized score
        > 1 would require model M1 < 0, which is impossible);
      - removes the small-``(baseline - ceiling)`` regime that drove the
        instability on sub-bin corruptions and on DPP tokenizers;
      - folds any encoder noise the model has into its raw M1, rather
        than crediting it against the corruption response — which is the
        honest accounting since the σ₀ noise is in the latent every
        forward pass.

    Metrics without a zero-strength counterpart (``permute_neurons``) are
    still dropped from the ceiling so the comparison's metric set is
    unchanged.
    """
    ceiling_results: dict[str, MetricResult] = {}
    for metric_name, metric_result in report.results.items():
        if "/" not in metric_name:
            continue
        _, label = metric_name.split("/", 1)
        zero_label = _zero_strength_m1_label(label)
        if zero_label is None:
            continue
        ceiling_results[metric_name] = MetricResult(
            name=metric_name,
            category=metric_result.category,
            direction=metric_result.direction,
            level=metric_result.level,
            raw_value=0.0,
            normalized_value=None,
            metadata={"ceiling_source_metric": "theoretical_zero"},
            value_range=metric_result.value_range,
            source_data_type=metric_result.source_data_type,
        )
    return EvaluationReport(results=ceiling_results)


def _m2_ceiling_report(report: EvaluationReport) -> EvaluationReport:
    """Per-session ceiling = each metric's ``eps_0`` sibling."""
    ceiling_results: dict[str, MetricResult] = {}
    for metric_name, metric_result in report.results.items():
        base_metric = metric_name.split("/", 1)[0]
        zero_metric_name = f"{base_metric}/eps_0"
        zero_metric = report.results.get(zero_metric_name)
        if zero_metric is None:
            continue
        ceiling_results[metric_name] = MetricResult(
            name=metric_name,
            category=metric_result.category,
            direction=metric_result.direction,
            level=metric_result.level,
            raw_value=float(zero_metric.raw_value),
            normalized_value=None,
            metadata={"ceiling_source_metric": zero_metric_name},
            value_range=metric_result.value_range,
            source_data_type=metric_result.source_data_type,
        )
    return EvaluationReport(results=ceiling_results)


def _zero_strength_m1_label(label: str) -> str | None:
    """Same rule as the job's static method — kept here so the post-hoc
    module stays self-contained."""
    if label.startswith("permute_neurons"):
        return None
    if label.startswith("window_shift|"):
        return re.sub(r"max_shift_ms=[^|]+", "max_shift_ms=0.0", label)
    if label.startswith("jitter|"):
        return re.sub(r"k_ms=[^|]+", "k_ms=0.0", label)
    if "|fraction=" in label:
        return re.sub(r"fraction=[^|]+", "fraction=0.0", label)
    return None


def _load_per_session_reports(results_dir: Path) -> dict[str, EvaluationReport]:
    """Load every ``per_session/{safe_sid}.json`` from a robustness run dir.

    The robustness jobs write one JSON per session under ``per_session/``,
    each a fully-serialized ``EvaluationReport``. Session IDs containing ``/``
    are stored with ``/`` replaced by ``__`` (matching the writer side); we
    keep that on-disk form here because the comparison only needs consistent
    keys across the three sources.
    """
    per_session_dir = results_dir / "per_session"
    if not per_session_dir.is_dir():
        raise FileNotFoundError(
            f"Expected {per_session_dir} to hold per-session JSON reports "
            "(written by robustness eval jobs). Found no such directory — "
            "is this a robustness run output dir?"
        )
    out: dict[str, EvaluationReport] = {}
    for json_path in sorted(per_session_dir.glob("*.json")):
        out[json_path.stem] = EvaluationReport.from_dict(json.loads(json_path.read_text()))
    if not out:
        log.warning(
            "post-hoc: %s contains no per-session JSON reports.",
            per_session_dir,
        )
    return out


# ────────────────────────── SNR internals ──────────────────────────


def _load_main_eval_metric_means_per_session(
    results_dir: Path,
    metric_columns: list[str],
) -> dict[str, dict[str, float]]:
    """Parse the main eval's ``results.csv`` into per-session metric means.

    The main eval writes one wide row per ``(session_id, segment_index)``.
    For each session we average each requested metric column over segments.
    Non-numeric / empty / non-finite cells are dropped. Sessions that end up
    with no valid value for any requested column are dropped entirely.

    We deliberately **don't** read ``results.json`` for metric metadata —
    SNR runs typically have an empty ``results.json/results`` block.
    """
    import csv

    csv_path = results_dir / "results.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Expected {csv_path} (the main eval pipeline's per-segment "
            "results CSV). Is this a main-eval / SNR run output dir?"
        )

    grouped: dict[str, dict[str, list[float]]] = {}
    seen_columns: set[str] = set()
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        header = set(reader.fieldnames or [])
        missing = [m for m in metric_columns if m not in header]
        if missing:
            log.warning(
                "SNR post-hoc: %s is missing CSV columns for %d requested "
                "metric(s): %s. Header had: %s",
                csv_path,
                len(missing),
                missing,
                sorted(header),
            )
        for row in reader:
            sid = row.get("session_id")
            if not sid:
                continue
            for metric in metric_columns:
                raw = row.get(metric)
                if raw is None or raw == "":
                    continue
                try:
                    value = float(raw)
                except ValueError:
                    continue
                if not np.isfinite(value):
                    continue
                grouped.setdefault(sid, {}).setdefault(metric, []).append(value)
                seen_columns.add(metric)

    per_session: dict[str, dict[str, float]] = {}
    for sid, by_metric in grouped.items():
        per_session[sid] = {metric: float(np.mean(values)) for metric, values in by_metric.items()}
    return per_session


__all__ = [
    "AggregationOrder",
    "MetricComparison",
    "compare_snr_against_poisson",
    "normalize_m1_results_against_poisson",
    "normalize_m2_results_against_poisson",
]
