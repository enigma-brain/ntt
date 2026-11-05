"""Compare evaluation reports across tokenizers."""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from ..core.types import (
    EvaluationReport,
    MetricCategory,
    MetricDirection,
    MetricLevel,
    MetricResult,
)

if TYPE_CHECKING:
    from ..config.schema import AggregationStrategy

logger = logging.getLogger(__name__)

EPS = 1e-12

# Minimum floor↔ceiling separation required to Poisson-normalize a metric into a
# category score. The skill score (model - floor) / (ceiling - floor) amplifies the
# numerator by 1 / (ceiling - floor); once that separation drops below ~0.1 the
# amplification exceeds ~10x and becomes comparable to the floor's own Monte-Carlo
# estimation noise, so the normalized score is dominated by amplification rather than
# by model quality and corrupts aggregate category means.
#
# Such a metric is "low power": the rate-matched Poisson baseline is statistically
# indistinguishable from perfect reconstruction, so the axis cannot discriminate
# tokenizers regardless of normalization. We mark it invalid (reason
# ``insufficient_baseline_separation``) so it drops out of the category mean instead.
# This is a property of (metric, eval-data) only — the model never enters the
# denominator — so the decision is identical for every tokenizer; it removes/keeps an
# axis uniformly and can never penalize one tokenizer relative to another.
#
# Real-data evidence (population_coupling at >=5 ms bins):
# the Poisson coupling floor sits at 0.97-0.999, giving separations of 0.001-0.05.
#
# Scope: this single absolute threshold is calibrated for bounded, correlation-like
# axes (separation measured in [0, 1] / [-1, 1] units). Applying one raw-unit constant
# to open-ended metrics (e.g. dimensionality_error, range [0, inf)) is not meaningful —
# their floor↔ceiling gaps live on an unrelated scale — so the gate is restricted to
# bounded metrics via ``_is_bounded``. Unbounded metrics keep only the EPS
# zero-denominator guard.
MIN_SEPARATION = 0.1

POISSON = "poisson"
POPULATION_MEAN_RATE = "population_mean_rate"

# Metrics scored against a reference other than the rate-matched Poisson floor (metric ->
# reference kind). The Poisson floor keeps every unit's own mean rate, so its
# firing_rate_error is only sampling noise (~0.1 Hz), and the skill score of any real
# tokenizer falls far below the clip. firing_rate_error is scored against a
# population-mean-rate reference instead: every unit is given the mean rate of its segment,
# which keeps the population's average rate and drops each unit's own rate. See
# ``with_alternative_floors``.
ALTERNATIVE_FLOORS: dict[str, str] = {"firing_rate_error": POPULATION_MEAN_RATE}
# Key of each session's reference error in that session's firing_rate_error metadata.
POPULATION_MEAN_RATE_ERROR_KEY = "population_mean_rate_error_hz"


def _is_bounded(value_range: tuple[float | None, float | None] | None) -> bool:
    """True if both ends of the metric's value range are finite.

    The :data:`MIN_SEPARATION` gate is only well-defined for bounded axes; unbounded
    metrics (an open ``None`` / non-finite bound) skip it.
    """
    if value_range is None:
        return False
    low, high = value_range
    return low is not None and high is not None and np.isfinite(low) and np.isfinite(high)


@dataclass(frozen=True)
class MetricComparison:
    """Direction-aware comparison of one metric across model, floor, and ceiling."""

    name: str
    category: MetricCategory
    direction: MetricDirection
    level: MetricLevel
    model_value: float
    baseline_value: float
    ceiling_value: float
    normalized_score: float | None
    value_range: tuple[float | None, float | None] = (None, None)
    status: str = "ok"
    reason: str | None = None
    # What ``baseline_value`` is: the rate-matched Poisson tokenizer, or the reference a
    # metric in ``ALTERNATIVE_FLOORS`` is scored against instead (then the Poisson value
    # is kept in ``poisson_baseline_value``).
    baseline_kind: str = POISSON
    poisson_baseline_value: float | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize to a JSON-compatible dictionary."""
        reference = (
            {}
            if self.baseline_kind == POISSON
            else {
                "baseline_kind": self.baseline_kind,
                "poisson_baseline_value": self.poisson_baseline_value,
            }
        )
        return {
            "name": self.name,
            "category": self.category.value,
            "direction": self.direction.value,
            "level": self.level.value,
            "model_value": self.model_value,
            "baseline_value": self.baseline_value,
            "ceiling_value": self.ceiling_value,
            "normalized_score": self.normalized_score,
            "value_range": list(self.value_range),
            "status": self.status,
            "reason": self.reason,
            **reference,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MetricComparison":
        """Deserialize from a JSON-compatible dictionary."""
        return cls(
            name=d["name"],
            category=MetricCategory(d["category"]),
            direction=MetricDirection(d["direction"]),
            level=MetricLevel(d["level"]),
            model_value=d["model_value"],
            baseline_value=d["baseline_value"],
            ceiling_value=d["ceiling_value"],
            normalized_score=d["normalized_score"],
            value_range=tuple(d["value_range"]) if "value_range" in d else (None, None),
            status=d.get("status", "ok"),
            reason=d.get("reason"),
            baseline_kind=d.get("baseline_kind", POISSON),
            poisson_baseline_value=d.get("poisson_baseline_value"),
        )

    @property
    def is_valid(self) -> bool:
        return self.normalized_score is not None and self.status == "ok"


def compute_normalized_comparisons(
    model_report: EvaluationReport,
    baseline_report: EvaluationReport,
    ceiling_report: EvaluationReport,
) -> dict[str, MetricComparison]:
    """Compute direction-aware normalized metric comparisons.

    The baseline report defines a score of ``0`` and the ceiling report defines a
    score of ``1``:

    - higher-is-better: ``(model - baseline) / (ceiling - baseline)``
    - lower-is-better: ``(baseline - model) / (baseline - ceiling)``

    Metrics present in only a subset of reports are skipped. Metrics with NaN/inf
    values or degenerate floor/ceiling separation are returned with
    ``normalized_score=None`` and a reason describing why normalization was skipped.
    """
    comparisons: dict[str, MetricComparison] = {}

    shared_metric_names = (
        set(model_report.results) & set(baseline_report.results) & set(ceiling_report.results)
    )

    for name in sorted(shared_metric_names):
        model_result = model_report.results[name]
        baseline_result = baseline_report.results[name]
        ceiling_result = ceiling_report.results[name]

        model_val = _to_scalar(model_result.raw_value)
        baseline_val = _to_scalar(baseline_result.raw_value)
        ceiling_val = _to_scalar(ceiling_result.raw_value)

        score, reason = normalized_score_vs_baseline(
            name,
            model_val,
            baseline_val,
            ceiling_val,
            model_result.direction,
            model_result.value_range,
        )
        comparisons[name] = MetricComparison(
            name=name,
            category=model_result.category,
            direction=model_result.direction,
            level=model_result.level,
            model_value=model_val,
            baseline_value=baseline_val,
            ceiling_value=ceiling_val,
            normalized_score=score,
            value_range=model_result.value_range,
            status="ok" if reason is None else "invalid",
            reason=reason,
        )

    return comparisons


def normalized_score_vs_baseline(
    name: str,
    model_val: float,
    baseline_val: float,
    ceiling_val: float,
    direction: MetricDirection,
    value_range: tuple[float | None, float | None] | None,
) -> tuple[float | None, str | None]:
    """Score one metric with the Poisson baseline at 0 and the ceiling at 1, whether higher or
    lower is better. Return (score, None), or (None, reason) if meaningless. Used for the eval
    job and ridge columns in the wide table."""
    if not np.isfinite(model_val) or not np.isfinite(baseline_val) or not np.isfinite(ceiling_val):
        return None, "non_finite_value"

    model_oriented = _orient_value(model_val, direction)
    baseline_oriented = _orient_value(baseline_val, direction)
    ceiling_oriented = _orient_value(ceiling_val, direction)
    denominator = ceiling_oriented - baseline_oriented

    if abs(denominator) < EPS:
        logger.warning(
            "Gating metric %r out of aggregation: degenerate floor==ceiling (separation %.3g).",
            name,
            abs(denominator),
        )
        return None, "degenerate_floor_ceiling"

    if abs(denominator) < MIN_SEPARATION and _is_bounded(value_range):
        # Floor and ceiling are too close for a stable skill score (see
        # MIN_SEPARATION). Mark invalid so the amplified, noise-dominated score
        # does not contaminate the category mean. The raw model/baseline/ceiling
        # values are still reported for transparency. Restricted to bounded axes
        # (the threshold is in correlation-like units); unbounded metrics skip it.
        logger.warning(
            "Gating metric %r out of aggregation: floor↔ceiling separation "
            "%.4g < MIN_SEPARATION (%.2g) on a bounded axis; the skill score would "
            "be dominated by amplification noise.",
            name,
            abs(denominator),
            MIN_SEPARATION,
        )
        return None, "insufficient_baseline_separation"

    return float((model_oriented - baseline_oriented) / denominator), None


def extract_normalized_scores(
    comparisons: dict[str, MetricComparison],
) -> dict[str, float]:
    """Extract only valid normalized scores from a comparison mapping."""
    return {
        name: comparison.normalized_score
        for name, comparison in comparisons.items()
        if comparison.normalized_score is not None and comparison.status == "ok"
    }


# Per session, one ``{metric name: result}`` mapping per segment: ``EvaluationReport.results``
# in the eval job, the saved ``per_session_per_segment[sid][i]["results"]`` dicts otherwise.
SegmentResults = Mapping[str, Sequence[Mapping[str, MetricResult | Mapping[str, Any]]]]


def with_alternative_floors(
    comparisons: dict[str, MetricComparison],
    per_session_segment_results: SegmentResults | None,
    aggregation_strategy: "AggregationStrategy | str | None",
    segment_weights: Mapping[str, Sequence[float]] | None = None,
) -> tuple[dict[str, MetricComparison], dict[str, dict[str, float]]]:
    """Score each metric in ``ALTERNATIVE_FLOORS`` against its reference instead of the
    Poisson floor.

    The one place this happens: the eval job calls it before saving ``results.json``, and
    the aggregation calls it on reports whose comparison is still against the Poisson floor
    (no ``baseline_kind``). A comparison already against its reference is
    kept as is, without reading ``per_session_segment_results``.

    For ``firing_rate_error`` (per-unit ``|r_orig - r_recon|``, lower is better, ceiling 0):
    the reference error of one segment is ``R_seg = mean_u |r_u - mean_u' r_u'|`` over the
    units the model's error is computed on. ``R_seg`` is aggregated over segments and then
    sessions exactly as the eval aggregates the model's error ``E``, and the score is taken
    once at the aggregate level: ``1 - E / R``. The Poisson floor value is kept in
    ``poisson_baseline_value``.

    ``segment_weights`` (time bins per segment, per session) is needed only for the
    ``weighted_mean`` strategy; ``results.json`` does not save it.

    Returns the updated comparisons and ``{metric: {session_id: reference error}}``.
    Raises ``ValueError`` when the per-segment data needed is missing, or when aggregating
    the saved per-segment errors does not reproduce the reported ``E``.
    """
    updated = dict(comparisons)
    per_session_references: dict[str, dict[str, float]] = {}
    for name, kind in ALTERNATIVE_FLOORS.items():
        comparison = updated.get(name)
        if comparison is None or comparison.baseline_kind == kind:
            continue
        updated[name], per_session_references[name] = _population_mean_rate_comparison(
            comparison, per_session_segment_results, aggregation_strategy, segment_weights
        )
    return updated, per_session_references


def population_mean_rate_unit_errors(
    rates_original_hz: Sequence[float] | np.ndarray,
    rates_reconstructed_hz: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Per-unit firing-rate error of the population-mean-rate reference, ``|r_u - mean r|``.

    The population mean is taken over the units the model's error is defined on (both
    rates finite); the other units are NaN, as in the model's per-unit error."""
    original = np.asarray(rates_original_hz, dtype=float)
    reconstructed = np.asarray(rates_reconstructed_hz, dtype=float)
    if original.ndim != 1 or original.shape != reconstructed.shape:
        raise ValueError(
            "rates_original_hz and rates_reconstructed_hz must be 1-D and the same length, "
            f"got shapes {original.shape} and {reconstructed.shape}"
        )
    scored = np.isfinite(original) & np.isfinite(reconstructed)
    errors = np.full(original.shape, np.nan)
    if scored.any():
        errors[scored] = np.abs(original[scored] - original[scored].mean())
    return errors


def _population_mean_rate_comparison(
    poisson_comparison: MetricComparison,
    per_session_segment_results: SegmentResults | None,
    aggregation_strategy: "AggregationStrategy | str | None",
    segment_weights: Mapping[str, Sequence[float]] | None,
) -> tuple[MetricComparison, dict[str, float]]:
    from ..config.schema import AggregationStrategy

    name = poisson_comparison.name
    if not np.isfinite(poisson_comparison.model_value):  # nothing to score, as against Poisson
        return replace(
            poisson_comparison,
            baseline_value=float("nan"),
            normalized_score=None,
            status="invalid",
            reason="non_finite_value",
            baseline_kind=POPULATION_MEAN_RATE,
            poisson_baseline_value=poisson_comparison.baseline_value,
        ), {}
    if not per_session_segment_results:
        raise ValueError(
            f"Cannot score {name} against its {POPULATION_MEAN_RATE} reference: the report has "
            "no per-segment results (per_session_per_segment)."
        )
    if aggregation_strategy is None:
        raise ValueError(f"Cannot recompute the {name} reference: no aggregation_strategy saved.")
    strategy = AggregationStrategy(aggregation_strategy)
    if strategy == AggregationStrategy.WEIGHTED_MEAN and segment_weights is None:
        raise ValueError(
            f"Cannot recompute the {name} reference under the weighted_mean strategy without "
            "segment weights (time bins per segment), which results.json does not save."
        )

    # Per session, per segment: the model's per-unit errors and the reference's.
    model_errors: dict[str, list[MetricResult]] = {}
    reference_errors: dict[str, list[MetricResult]] = {}
    weights: dict[str, list[float]] = {}
    for sid, segments in per_session_segment_results.items():
        for index, results in enumerate(segments):
            if name not in results:  # not computed on this segment; E skips it too
                continue
            entry = results[name]
            result = (
                entry if isinstance(entry, MetricResult) else MetricResult.from_dict(dict(entry))
            )
            original = result.metadata.get("rates_original_hz")
            reconstructed = result.metadata.get("rates_reconstructed_hz")
            if original is None or reconstructed is None:
                raise ValueError(
                    f"Cannot compute the {name} reference: segment {index} of session {sid!r} "
                    "has no rates_original_hz / rates_reconstructed_hz in its metadata."
                )
            # A NaN unit is saved as JSON null; read it back as NaN, as the eval's nanmean saw it.
            model_errors.setdefault(sid, []).append(
                replace(result, raw_value=np.asarray(result.raw_value, dtype=float), metadata={})
            )
            reference_errors.setdefault(sid, []).append(
                replace(
                    result,
                    raw_value=population_mean_rate_unit_errors(original, reconstructed),
                    metadata={},
                )
            )
            weights.setdefault(sid, []).append(
                float(segment_weights[sid][index]) if segment_weights is not None else 1.0
            )
    if not model_errors:
        raise ValueError(f"Cannot compute the {name} reference: no segment has a {name} result.")

    # As in ``aggregate_session_reports``: a session weighs its segments' total time bins.
    session_weights = {
        sid: float(sum(segment_weights[sid])) if segment_weights is not None else 1.0
        for sid in model_errors
    }
    model_value, _ = _aggregate_segments_then_sessions(
        name, model_errors, weights, session_weights, strategy
    )
    if not np.isclose(model_value, poisson_comparison.model_value, rtol=1e-6, atol=1e-9):
        raise ValueError(
            f"Aggregating the per-segment {name} values ({model_value!r}) does not reproduce "
            f"the reported value ({poisson_comparison.model_value!r}); the reference would "
            "not be aggregated like the model's error."
        )
    reference, per_session = _aggregate_segments_then_sessions(
        name, reference_errors, weights, session_weights, strategy
    )
    score, reason = normalized_score_vs_baseline(
        name,
        poisson_comparison.model_value,
        reference,
        poisson_comparison.ceiling_value,
        poisson_comparison.direction,
        poisson_comparison.value_range,
    )
    comparison = replace(
        poisson_comparison,
        baseline_value=reference,
        normalized_score=score,
        status="ok" if reason is None else "invalid",
        reason=reason,
        baseline_kind=POPULATION_MEAN_RATE,
        poisson_baseline_value=poisson_comparison.baseline_value,
    )
    return comparison, per_session


def _aggregate_segments_then_sessions(
    name: str,
    per_session_results: dict[str, list[MetricResult]],
    segment_weights: dict[str, list[float]],
    session_weights: dict[str, float],
    strategy: "AggregationStrategy",
) -> tuple[float, dict[str, float]]:
    """Two-level aggregation of one metric, through the helper the eval uses at both
    levels (``aggregate_segment_reports`` / ``aggregate_session_reports``): segments within
    a session, then sessions. Returns the aggregate and the per-session values."""
    from ..utils.validation_collation import _aggregate_evaluation_reports

    sessions: dict[str, EvaluationReport] = {}
    for sid, results in per_session_results.items():
        session = _aggregate_evaluation_reports(
            [EvaluationReport(results={name: r}) for r in results], segment_weights[sid], strategy
        )
        if name in session.results:  # every segment NaN: E leaves the session out too
            sessions[sid] = session
    aggregate = _aggregate_evaluation_reports(
        list(sessions.values()), [session_weights[sid] for sid in sessions], strategy
    )
    if name not in aggregate.results:
        raise ValueError(f"No finite {name} value in any session.")
    per_session = {sid: _to_scalar(r.results[name].raw_value) for sid, r in sessions.items()}
    return _to_scalar(aggregate.results[name].raw_value), per_session


def _orient_value(value: float, direction: MetricDirection) -> float:
    """Map all metrics to a higher-is-better axis."""
    if direction == MetricDirection.HIGHER_IS_BETTER:
        return value
    return -value


def _to_scalar(value: float | np.ndarray) -> float:
    """Reduce array-valued metric to a scalar via nanmean.

    Robustness (M1/M2) metrics intentionally store one report entry per
    fully-qualified bucket — ``<metric>/<corruption>|<strength>`` for M1,
    ``<metric>/eps_<epsilon>`` for M2 — with a scalar ``raw_value`` each, so
    that ``compute_normalized_comparisons`` produces one normalized score per
    (metric, corruption, strength). An array landing here means a caller
    collapsed buckets upstream (or registered a genuinely array-valued
    metric) and the resulting normalized score will be a single number across
    all buckets — a quiet loss of granularity. We log a warning rather than
    raising, to stay compatible with array-valued metrics in the main eval
    pipeline that legitimately collapse via nanmean.
    """
    if isinstance(value, np.ndarray):
        if value.size != 1:
            logger.warning(
                "comparison._to_scalar collapsing array of shape %s via nanmean — "
                "per-bucket granularity is lost. Robustness metrics should hand "
                "in one scalar raw_value per (metric, corruption, strength) "
                "bucket; check that buckets weren't aggregated upstream.",
                value.shape,
            )
        return float(np.nanmean(value))
    return float(value)
