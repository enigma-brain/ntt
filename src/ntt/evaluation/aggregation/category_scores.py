"""Category score aggregation from eval result JSONs.

Pure library — no I/O, no MLflow, no Hydra. All inputs are already loaded: eval reports
as dicts, downstream records as ``DownstreamScoreView``s (``downstream_records``).
"""

from __future__ import annotations

import logging
import math
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ntt.evaluation.aggregation.comparability import (
    RIDGE_EXCLUDED_SESSIONS,
    _finite_float,
    _is_position_regression_key,
    check_reference_match,
    poisson_norm_session_ids,
    selected_chunk_windows,
)
from ntt.evaluation.aggregation.comparison import (
    ALTERNATIVE_FLOORS,
    MetricComparison,
    compute_normalized_comparisons,
    with_alternative_floors,
)
from ntt.evaluation.behavioral_names import REGRESSION_PREFIXES, REGRESSION_RESULT_NAMES
from ntt.evaluation.core.types import EvaluationReport, MetricCategory

if TYPE_CHECKING:
    from ntt.evaluation.aggregation.downstream_records import DownstreamScoreView

log = logging.getLogger(__name__)


_BIN_SUFFIX_RE = re.compile(r"_(?:\d+(?:\.\d+)?)ms$")


def family(name: str) -> str:
    """Base metric-family key: drop the `|param=val` strength tail and the
    trailing `_<N>ms` bin suffix so all variants of one conceptual metric share a key."""
    base = name.split("|", 1)[0]
    return _BIN_SUFFIX_RE.sub("", base)


def family_weighted_mean(pairs: list[tuple[str, float]]) -> float | None:
    """Mean where each `family(name)` contributes total weight 1 (its members
    split that weight equally). Equivalent to the mean of per-family means."""
    groups: dict[str, list[float]] = defaultdict(list)
    for name, val in pairs:
        groups[family(name)].append(val)
    family_means = [statistics.fmean(vals) for vals in groups.values() if vals]
    return statistics.fmean(family_means) if family_means else None


# ── Category sets ──────────────────────────────────────────────────────────────

_SN_RECON_CATEGORIES: frozenset[MetricCategory] = frozenset(
    {
        MetricCategory.TEMPORAL_FIDELITY,
        MetricCategory.SINGLE_UNIT_STATISTICS,
    }
)
_POP_STRUCTURE_CATEGORIES: frozenset[MetricCategory] = frozenset(
    {
        MetricCategory.POPULATION_STRUCTURE,
    }
)
# Metric names excluded from every aggregated composite column. This is a
# *scoring* decision — these are computed and reported, but are not headline
# quality scores: ``token_dims_per_second`` is a throughput/cost metric, and
# ``repeats_snr``/``repeats_baseline`` are unbounded diagnostics (value_range
# upper bound is None) that don't belong on the Poisson-normalized composite
# scale. ``repeats_snr``/``repeats_baseline`` still appear in the wide
# raw-metrics table.
#
# Do NOT fold this into ``ntt.evaluation.utils.metric_selection`` — that module
# is a distinct, *pipeline-execution* concern (which metrics to COMPUTE in the
# recon / M2 / interval-windowed passes), and its membership deliberately
# differs from this set: e.g. ``population_structure_preservation`` is
# "non-recon" there yet IS included in a composite here, and
# ``token_dims_per_second`` is allowed on interval windows there yet excluded
# here. The two sets share only metric-name strings, not a concept.
_EXCLUDED_FROM_AGGREGATES: frozenset[str] = frozenset(
    {
        "token_dims_per_second",
        "repeats_snr",
        "repeats_baseline",
    }
)


# ── Output types ───────────────────────────────────────────────────────────────


@dataclass
class MetricScore:
    """Per-metric contribution within a column."""

    normalized: float  # Score after the column's transform (Poisson-normalized
    # in the recon buckets; a bounded [0,1] transform in
    # latent_structure_and_robustness — see score_latent_structure_and_robustness)
    clipped: bool  # True if score_clip was applied (recon buckets only)
    raw_value: float  # Original metric value from the report
    direction: str  # "higher" | "lower"


@dataclass
class ColumnDiagnostics:
    """Detailed breakdown for one aggregated column."""

    score: float | None  # Final column value; None if no valid metrics
    n_metrics: int  # Number of metrics that contributed
    normalization: str  # "precomputed" | "recomputed" | ...
    per_metric: dict[str, MetricScore] = field(default_factory=dict)
    provenance: dict[str, str] = field(default_factory=dict)  # e.g. {"m1": "...", "m2": "..."}
    # Mean of each family the column pools, e.g. {"poyo": ..., "ridge": ...} for decoding.
    family_means: dict[str, float] = field(default_factory=dict)


@dataclass
class CategoryScores:
    """Seven aggregated category scores for one tokenizer checkpoint.

    ``scores`` section — the 7 primary columns (None = not computed), followed by the
    two family scores that ``decoding`` averages:
    """

    single_neuron_recon: float | None
    population_structure: float | None
    latent_structure_and_robustness: float | None
    decoding: float | None
    downstream_encoding: float | None
    generalization: float | None
    compression: float | None

    # The decoding breakdown: ``decoding`` is the family-weighted mean of these two. Each is
    # None when its family contributed nothing (absent, switched off, or ``decoding`` empty).
    decoding_poyo: float | None = None  # mean POYO test R² over sessions
    decoding_linear: float | None = None  # mean ridge regression R² over targets

    # Keyed by column name; None for scalar pass-throughs (generalization, compression)
    # and for scalar columns with no per-metric breakdown (downstream_encoding).
    diagnostics: dict[str, ColumnDiagnostics | None] = field(default_factory=dict)

    # Sessions the Poisson normalization (single_neuron_recon, population_structure)
    # was computed over: {"source": "precomputed"|"recomputed", "session_ids": [...] or
    # None when the reports do not record it}. None when there is no recon report.
    poisson_norm_sessions: dict | None = None

    # Metrics of the Poisson-normalized columns scored against another reference
    # (``ALTERNATIVE_FLOORS``): {metric: reference kind}. None when there is no recon report.
    metric_references: dict[str, str] | None = None


# ── Internal: normalization resolution ────────────────────────────────────────


def _get_comparisons(
    recon_report: dict,
    poisson_baseline: dict | None,
    force_recompute: bool,
    *,
    common_parts_only: bool = False,
) -> tuple[dict[str, MetricComparison], str]:
    """Resolve which comparison_vs_poisson dict to use and return it plus a label.

    Three cases:
      - No baseline: use precomputed ``comparison_vs_poisson`` (raises if missing)
      - Baseline + force_recompute=False: use precomputed if present, else recompute
      - Baseline + force_recompute=True: always recompute from baseline
    """
    use_precomputed = poisson_baseline is None or (
        not force_recompute and "comparison_vs_poisson" in recon_report
    )

    if use_precomputed:
        raw = recon_report.get("comparison_vs_poisson")
        if raw is None:
            raise ValueError(
                "comparison_vs_poisson not found in recon_report. "
                "Re-run eval with compare_poisson=true, or provide a poisson_reference."
            )
        return (
            {name: MetricComparison.from_dict(d) for name, d in raw.items()},
            "precomputed",
        )

    # Recompute path (use_precomputed is False only when a baseline was given).
    assert poisson_baseline is not None
    # results.json wraps the report under "aggregated"; eval_report.json is the
    # EvaluationReport dict directly (no wrapper key).
    check_reference_match(recon_report, poisson_baseline, common_parts_only=common_parts_only)
    model = EvaluationReport.from_dict(recon_report.get("aggregated", recon_report))
    floor = EvaluationReport.from_dict(poisson_baseline["aggregated"])
    ceiling = EvaluationReport.from_dict(poisson_baseline["perfect_aggregated"])
    return compute_normalized_comparisons(model, floor, ceiling), "recomputed"


def apply_alternative_floors(
    comparisons: dict[str, MetricComparison], recon_report: dict
) -> tuple[dict[str, MetricComparison], list[str]]:
    """Score the ``ALTERNATIVE_FLOORS`` metrics against their reference. A comparison
    that is still against the Poisson floor (the eval ran without ``compare_poisson``, or
    the report was scored against another reference) is recomputed from the report's
    per-unit rates (raises if the report has none).

    Returns the comparisons and one ``"<metric>=<reference>(<source>)"`` note per metric
    for the diagnostics' ``normalization`` string."""
    saved = {
        name
        for name, kind in ALTERNATIVE_FLOORS.items()
        if name in comparisons and comparisons[name].baseline_kind == kind
    }
    per_segment = recon_report.get("per_session_per_segment")
    updated, _ = with_alternative_floors(
        comparisons,
        (
            {
                sid: [segment["results"] for segment in segments]
                for sid, segments in per_segment.items()
            }
            if isinstance(per_segment, dict)
            else None
        ),
        recon_report.get("aggregation_strategy"),
    )
    notes = [
        f"{name}={kind}({'saved' if name in saved else 'recomputed_from_rates'})"
        for name, kind in ALTERNATIVE_FLOORS.items()
        if name in updated
    ]
    return updated, notes


# ── Internal: column scorers ───────────────────────────────────────────────────


def _apply_clip(val: float, score_clip: float | None) -> tuple[float, bool]:
    if score_clip is not None and val < score_clip:
        return score_clip, True
    return val, False


def _score_comparison_bucket(
    comparisons: dict[str, MetricComparison],
    categories: frozenset[MetricCategory],
    score_clip: float | None,
    normalization: str,
) -> ColumnDiagnostics:
    """Family-weighted Poisson-normalized score over recon comparisons in ``categories``.

    Skips invalid comparisons, comparisons outside ``categories``, and any metric in
    ``_EXCLUDED_FROM_AGGREGATES``. Each metric family contributes equal weight to the mean.
    """
    per_metric: dict[str, MetricScore] = {}
    pairs: list[tuple[str, float]] = []

    for name, comp in comparisons.items():
        if not comp.is_valid or comp.category not in categories:
            continue
        if name in _EXCLUDED_FROM_AGGREGATES:
            continue

        if comp.normalized_score is None:  # unreachable for valid comparisons
            continue
        clipped_val, clipped = _apply_clip(comp.normalized_score, score_clip)
        per_metric[name] = MetricScore(
            normalized=comp.normalized_score,
            clipped=clipped,
            raw_value=comp.model_value,
            direction=comp.direction.value,
        )
        pairs.append((name, clipped_val))

    return ColumnDiagnostics(
        score=family_weighted_mean(pairs),
        n_metrics=len(per_metric),
        normalization=normalization + ";family_weighted",
        per_metric=per_metric,
    )


def _score_generalization(recon_report: dict) -> float | None:
    try:
        return recon_report["generalization"]["summary"]["generalization_score"]
    except (KeyError, TypeError):
        return None


def _score_compression(recon_report: dict, compression_metric: str) -> float | None:
    try:
        return recon_report["compression"][compression_metric]
    except (KeyError, TypeError):
        return None


# The downstream encoding metric ntt reports: noise-ceiling-corrected R².
ENCODING_METRIC = "nc_r2"


def _score_encoding(
    encoding: DownstreamScoreView,
) -> tuple[float | None, ColumnDiagnostics | None]:
    """Score downstream encoding: the view's aggregated ``nc_r2`` value (for a v2 record,
    ``metrics.test.aggregated.nc_r2.value``).

    Returns ``(score, None)`` — no per-metric diagnostics for this scalar pass-through.
    ``(None, None)`` with a warning when the view has no ``nc_r2``, e.g. a record whose
    final eval failed (its ``final_eval`` diagnostic is in the warning).
    """
    score = _finite_float((encoding.aggregated.get(ENCODING_METRIC) or {}).get("value"))
    if score is None:
        log.warning(
            "Downstream encoding has no %s score in its %s block (final_eval: %s): "
            "downstream_encoding is None.",
            ENCODING_METRIC,
            encoding.split,
            encoding.provenance.get("final_eval"),
        )
    return score, None


# ── Public API ─────────────────────────────────────────────────────────────────


def compute_category_scores(
    recon_report: dict | None = None,
    m1_report: dict | None = None,
    m2_report: dict | None = None,
    encoding: DownstreamScoreView | None = None,
    decoding: DownstreamScoreView | None = None,
    *,
    poisson_baseline: dict | None = None,
    force_recompute_poisson: bool = False,
    score_clip: float | None = -1.0,
    compression_metric: str = "compression_rate",
    include_m2: bool = False,
    m2_metric: str = "acg_correlation",
    m2_epsilon: float = 0.1,
    include_m1: bool = True,
    include_poyo: bool = True,
    include_ridge: bool = True,
    common_parts_only: bool = False,
    allow_incomplete_decoding_cohort: bool = False,
) -> CategoryScores:
    """Compute the seven aggregated category scores from eval result JSONs.

    All inputs are optional; unspecified columns are returned as None. ``encoding`` and
    ``decoding`` are downstream records read by ``load_downstream_record``; see
    ``_score_encoding`` and ``score_decoding_merged`` for what each column reads.

    The ``latent_structure_and_robustness`` bucket is **not** Poisson-normalized:
    it reads raw values from ``m1_report``/``m2_report`` (eval_report.json shape)
    and the recon comparisons, and maps each onto [0,1] via a monotonic bounded
    transform (M1 retention ``1/(1+d)``, structure ``max(corr,0)``, M2 ``max(acg,0)``).
    ``score_clip`` therefore does not apply to that bucket — only to the
    Poisson-normalized recon buckets. ``include_m2``/``m2_metric``/``m2_epsilon`` are
    GLOBAL knobs, fixed across all tokenizers in a comparison (never per-tokenizer).
    """
    sn_diag: ColumnDiagnostics | None = None
    pop_diag: ColumnDiagnostics | None = None
    comparisons: dict[str, MetricComparison] = {}

    if recon_report is not None:
        comparisons, normalization = _get_comparisons(
            recon_report,
            poisson_baseline,
            force_recompute_poisson,
            common_parts_only=common_parts_only,
        )
        comparisons, reference_notes = apply_alternative_floors(comparisons, recon_report)
        metric_references = {
            name: comparisons[name].baseline_kind
            for name in ALTERNATIVE_FLOORS
            if name in comparisons
        }
        bucket_normalization = ";".join([normalization, *reference_notes])
        sn_diag = _score_comparison_bucket(
            comparisons, _SN_RECON_CATEGORIES, score_clip, bucket_normalization
        )
        pop_diag = _score_comparison_bucket(
            comparisons, _POP_STRUCTURE_CATEGORIES, score_clip, bucket_normalization
        )
        poisson_norm_sessions = {
            "source": normalization,
            "session_ids": poisson_norm_session_ids(recon_report, poisson_baseline, normalization),
        }
    else:
        poisson_norm_sessions = None
        metric_references = None

    lsr_diag = score_latent_structure_and_robustness(
        comparisons,
        m1_report,
        m2_report,
        recon_report=recon_report,
        include_m2=include_m2,
        m2_metric=m2_metric,
        m2_epsilon=m2_epsilon,
        include_m1=include_m1,
    )

    generalization = _score_generalization(recon_report) if recon_report is not None else None
    compression = (
        _score_compression(recon_report, compression_metric) if recon_report is not None else None
    )

    enc_score, enc_diag = _score_encoding(encoding) if encoding is not None else (None, None)
    dec_score, dec_diag = score_decoding_merged(
        decoding,
        recon_report,
        include_poyo=include_poyo,
        include_ridge=include_ridge,
        allow_incomplete_cohort=allow_incomplete_decoding_cohort,
    )
    dec_families = dec_diag.family_means if dec_diag is not None else {}

    return CategoryScores(
        single_neuron_recon=sn_diag.score if sn_diag is not None else None,
        population_structure=pop_diag.score if pop_diag is not None else None,
        latent_structure_and_robustness=lsr_diag.score if lsr_diag is not None else None,
        decoding=dec_score,
        downstream_encoding=enc_score,
        generalization=generalization,
        compression=compression,
        decoding_poyo=dec_families.get("poyo"),
        decoding_linear=dec_families.get("ridge"),
        diagnostics={
            "single_neuron_recon": sn_diag,
            "population_structure": pop_diag,
            "latent_structure_and_robustness": lsr_diag,
            "decoding": dec_diag,
            "downstream_encoding": enc_diag,
            "generalization": None,
            "compression": None,
        },
        poisson_norm_sessions=poisson_norm_sessions,
        metric_references=metric_references,
    )


# The three canonical non-population corruptions whose mean raw latent-NMSE is
# M1's aggregate score (mirrors ``_ROBUSTNESS_INPUT_CORE_SCORE_COMPONENTS`` in
# ``jobs/eval_robustness_input.py``, here prefixed with the ``robustness_input/``
# metric namespace as they appear in the eval report's ``results`` block).
# Kept local so this pure library doesn't import the heavyweight job module.
_M1_CANONICAL_COMPONENTS: tuple[str, ...] = (
    "robustness_input/window_shift|max_shift_ms=3.0",
    "robustness_input/jitter|k_ms=1.0|p=0.2",
    "robustness_input/drop_and_add|fraction=0.02",
)


def _report_results(report: dict | None) -> dict:
    """Pull the ``results`` block from an eval report dict, accepting both the compact
    ``eval_report.json`` shape (``{"results": {...}}``) and the ``results.json`` wrapper
    (``{"aggregated": {"results": {...}}}``).

    Serves the M1/M2 robustness reports and the recon report (whose ridge decoders
    ``score_decoding_merged`` reads), so it is not robustness-specific."""
    if not report:
        return {}
    results = report.get("results")
    if isinstance(results, dict):
        return results
    agg = report.get("aggregated")
    if isinstance(agg, dict) and isinstance(agg.get("results"), dict):
        return agg["results"]
    return {}


def _retention(d: float) -> float:
    """Bounded latent-NMSE → retention transform: monotone-decreasing, maps
    ``d ∈ [0, ∞)`` onto ``(0, 1]`` (d=0 → 1; larger corruption response → lower)."""
    return 1.0 / (1.0 + d)


def resolve_m1_retention(m1_report: dict | None) -> tuple[float | None, float | None, str]:
    """Resolve the M1 input-corruption retention score from a raw M1 report.

    ``retention = 1/(1+d)`` where ``d`` is the mean raw latent-NMSE over the three
    canonical non-population corruptions (``window_shift|max_shift_ms=3.0``,
    ``jitter|k_ms=1.0|p=0.2``, ``drop_and_add|fraction=0.02``). If any is missing
    or non-finite, warn loudly and exclude — that means a non-default M1
    corruption config, and a partial mean would not be comparable across
    tokenizers.

    Returns ``(retention, raw_d, provenance)``; ``(None, None, <reason>)`` if excluded.
    Permute_neurons, ``*_per_dim/*`` and zero-strength variants are never read here.
    """
    if not m1_report:
        log.warning(
            "Input-corruption robustness report absent (no such eval, or empty "
            "tracker cell): EXCLUDING it from "
            "latent_structure_and_robustness, so it is averaged over fewer members "
            "than for tokenizers that have one."
        )
        return None, None, "excluded_no_report"
    results = _report_results(m1_report)

    values: list[float] = []
    missing: list[str] = []
    for name in _M1_CANONICAL_COMPONENTS:
        entry = results.get(name)
        raw = _finite_float(entry.get("raw_value")) if isinstance(entry, dict) else None
        if raw is None:
            missing.append(name)
        else:
            values.append(raw)
    if not missing:
        d = statistics.fmean(values)
        if d > -1.0:
            return _retention(d), d, "components_mean"

    log.warning(
        "M1 retention not computed: not all three canonical components (%s) were "
        "present with finite raw_value in the M1 report (missing: %s). This usually "
        "means a non-default M1 corruption config. EXCLUDING M1 from "
        "latent_structure_and_robustness.",
        list(_M1_CANONICAL_COMPONENTS),
        missing,
    )
    return None, None, "excluded_no_components"


def resolve_m2_acg(
    m2_report: dict | None, m2_metric: str, m2_epsilon: float
) -> tuple[float | None, float | None, str, str | None]:
    """Resolve the M2 latent-noise robustness score from a raw M2 report.

    Reads the ``m2_metric`` (default ``acg_correlation``) row whose
    ``metadata.epsilon`` equals ``m2_epsilon`` and bounds it via ``max(acg, 0)``
    (ACG correlation is in [-1,1]; a non-positive value scores 0). The on-disk key looks like
    ``acg_correlation/eps_0.1`` but the suffix formatting is unreliable
    (e.g. ``eps_1`` for ε=1.0), so matching is done on ``metadata.epsilon``.

    Returns ``(bounded, raw_acg, provenance, matched_name)``;
    ``(None, None, <reason>, None)`` if the metric/epsilon is absent (warns loudly).
    """
    if not m2_report:
        log.warning(
            "Latent-corruption robustness report absent (no such eval, or empty "
            "tracker cell) although include_m2=True: EXCLUDING it from "
            "latent_structure_and_robustness, so it "
            "is averaged over fewer members than for tokenizers that have one."
        )
        return None, None, "excluded_no_report", None
    results = _report_results(m2_report)

    matches: list[tuple[str, dict]] = []
    for name, entry in results.items():
        if not isinstance(entry, dict):
            continue
        if name.split("/", 1)[0] != m2_metric:
            continue
        eps = (entry.get("metadata") or {}).get("epsilon")
        if eps is None:
            continue
        try:
            eps_f = float(eps)
        except (TypeError, ValueError):
            continue
        if math.isclose(eps_f, float(m2_epsilon), rel_tol=0.0, abs_tol=1e-9):
            matches.append((name, entry))

    if not matches:
        log.warning(
            "M2 metric %r at epsilon=%s not found in the M2 report (include_m2=True). "
            "EXCLUDING M2 from latent_structure_and_robustness.",
            m2_metric,
            m2_epsilon,
        )
        return None, None, "excluded_missing_key", None
    if len(matches) > 1:
        log.warning(
            "M2 metric %r matched %d rows at epsilon=%s (%s); using the first.",
            m2_metric,
            len(matches),
            m2_epsilon,
            [n for n, _ in matches],
        )

    name, entry = matches[0]
    acg = _finite_float(entry.get("raw_value"))
    if acg is None:
        log.warning(
            "M2 metric %r at epsilon=%s has no finite raw_value; EXCLUDING M2.",
            m2_metric,
            m2_epsilon,
        )
        return None, None, "excluded_missing_key", None
    return max(acg, 0.0), acg, "included", name


def _is_regression_decoder(key: str, entry: dict) -> bool:
    """True for the in-repo ridge/linear regression decoders in a recon report.

    ``aggregated.results`` is keyed by *display* name, which the pipeline suffixes per
    behavioral target (``expand_results``, ``validation_collation.py``): the real keys are
    ``behavioral_regression_cursor_vel`` etc. while ``entry["name"]`` stays the bare
    ``behavioral_regression``. Match on ``name`` — it survives a change to the display-name
    scheme — and fall back to the key prefix when an entry carries no ``name``, so a missing
    field cannot silently drop the whole family the way a bare-name key lookup did.

    ``behavioral_classification`` is excluded by both arms: it reports accuracy, not R².
    """
    name = entry.get("name")
    if isinstance(name, str):
        return name in REGRESSION_RESULT_NAMES
    return key.startswith(REGRESSION_PREFIXES)


def ridge_session_values(
    key: str,
    report: dict,
    *,
    poisson: bool = False,
    excluded: frozenset[str] = RIDGE_EXCLUDED_SESSIONS,
) -> dict[str, float]:
    """``{session: score}`` for ridge entry ``key``, leaving out ``excluded`` sessions, from the
    tokenizer or the Poisson baseline (``poisson=True``). A NaN score raises an error; keep an
    infinite score."""
    if report["aggregation_strategy"] != "mean":
        raise ValueError(f"Ridge entry {key!r}: can only redo a 'mean' session aggregation")
    if poisson:
        results_by_session = {}
        for session, segments in report["poisson_per_session_per_segment"].items():
            if len(segments) != 1:
                raise ValueError(
                    f"Ridge entry {key!r}: expected one Poisson segment for session "
                    f"{session!r}, got {len(segments)}"
                )
            results_by_session[session] = segments[0]["results"]
    else:
        results_by_session = {s: entry["results"] for s, entry in report["per_session"].items()}
    values = {}
    for session, results in results_by_session.items():
        if session in excluded:
            continue
        if key not in results:  # no score for this session
            continue
        value = float(results[key]["raw_value"])
        if math.isnan(value):
            raise ValueError(f"Ridge entry {key!r}: NaN score in session {session!r}")
        values[session] = value
    return values


def _ridge_value_without_excluded_sessions(key: str, report: dict) -> float | None:
    """Average ridge entry ``key`` over the sessions in ``report["per_session"]``, leaving out
    ``RIDGE_EXCLUDED_SESSIONS``. A NaN session value raises an error. An infinite value is kept,
    so the average becomes infinite too. Return ``None`` if no remaining session has this entry."""
    values = ridge_session_values(key, report)
    return statistics.fmean(values.values()) if values else None


def score_decoding_merged(
    decoding: DownstreamScoreView | None,
    recon_report: dict | None,
    *,
    include_poyo: bool = True,
    include_ridge: bool = True,
    allow_incomplete_cohort: bool = False,
) -> tuple[float | None, ColumnDiagnostics | None]:
    """Raw R² of the POYO decoders and the in-repo ridge regression decoders, pooled as
    two equal families.

    The POYO members are the decoding view's per-session test R² (``poyo/<session>``;
    for a v2 cohort record, ``metrics.test.per_session[<session>].r2``). An incomplete
    cohort (cells missing) gives a ``None`` column with a warning naming the missing
    sessions, unless ``allow_incomplete_cohort`` — then it scores over the found cells.

    The POYO sessions pool to one family and the ridge targets to a second, each carrying
    total weight 1 — the same family weighting as every other composite (the recon buckets
    and the latent bucket both go through ``family_weighted_mean``). A flat mean over the
    members would hand ~12/14 of the column to the POYO sessions and let the linear
    decoders barely register.

    The two families measure different things — a trained decoder's ceiling vs linear
    decodability of the latent — so each family mean is also recorded, in ``family_means``
    (the ``decoding_poyo`` / ``decoding_linear`` scores) and as text in ``provenance``.

    Excludes behavioral_classification (accuracy, not R²).

    ``include_poyo`` / ``include_ridge`` = False leave that family out (provenance
    ``off``) — used to drop a family for every tokenizer of a table when only some
    tokenizers have it."""
    if decoding is not None and include_poyo and not decoding.complete:
        missing = decoding.provenance.get("missing")
        if not allow_incomplete_cohort:
            log.warning(
                "Decoding cohort is incomplete (missing sessions: %s): decoding is None. "
                "Set allow_incomplete_decoding_cohort to score it over the found sessions.",
                missing,
            )
            return None, None
        log.warning(
            "Decoding cohort is incomplete (missing sessions: %s); scoring it over the "
            "found sessions (allow_incomplete_decoding_cohort).",
            missing,
        )

    per_metric: dict[str, MetricScore] = {}
    poyo_vals: list[float] = []
    ridge_vals: list[float] = []

    # Both sources drop non-finite R² (None / NaN / inf) via _finite_float, so a
    # single bad value can't poison statistics.fmean and NaN the whole column.
    sessions = decoding.per_session if decoding is not None and include_poyo else {}
    for sid, sdata in sessions.items():
        r2 = _finite_float(sdata.get("r2")) if isinstance(sdata, dict) else None
        if r2 is None:
            continue
        per_metric[f"poyo/{sid}"] = MetricScore(
            normalized=r2, clipped=False, raw_value=r2, direction="higher"
        )
        poyo_vals.append(r2)

    # Accept both the wrapped ``results.json`` and the compact ``eval_report.json``
    # shapes: a compact report would otherwise silently drop the whole ridge family
    # from the column.
    results = _report_results(recon_report) if include_ridge else {}

    # Position regression counts as excluded when the report has it but none of it reaches
    # the score: not 200 ms, not finite, or (only under skipped metrics) every chunk-size
    # sweep failed. Being listed under skipped metrics is not enough on its own: the merged
    # skip reasons also cover sessions without behavior ("skipped on 17 of 20 sessions"),
    # so nearly every report lists it there, including those whose 200 ms score is used.
    position_reported = any(_is_position_regression_key(name) for name in results)
    if include_ridge:
        report = recon_report or {}
        metric_report = report.get("aggregated", report)
        skipped_metrics = metric_report.get("skipped_metrics") or {}
        position_reported = position_reported or any(
            _is_position_regression_key(name) for name in skipped_metrics
        )
    position_included = False

    chunk_sizes = selected_chunk_windows(recon_report) if recon_report else {}
    for key, entry in sorted(results.items()):
        if not isinstance(entry, dict) or not _is_regression_decoder(key, entry):
            continue
        # When calculating each tokenizer’s combined decoding score for the results table,
        # exclude position linear regression if any contributing session has no saved selected
        # chunk size or has a saved selected chunk size other than 200 ms.
        if _is_position_regression_key(key):
            sizes = [value for name, value in chunk_sizes.items() if name.endswith(f"/{key}")]
            if not sizes or any(
                value is None or any(size != 200 for size in value) for value in sizes
            ):
                log.warning("Excluding %s from decoding: no verified 200 ms position score", key)
                continue
        raw = _ridge_value_without_excluded_sessions(key, recon_report or {})
        if raw is None:
            continue
        position_included = position_included or _is_position_regression_key(key)
        # Keyed by the suffixed display name, so two targets of one decoder can't
        # clobber each other in the diagnostics.
        per_metric[f"ridge/{key}"] = MetricScore(
            normalized=raw, clipped=False, raw_value=raw, direction="higher"
        )
        ridge_vals.append(raw)

    groups = (
        ("poyo", poyo_vals, "session", include_poyo),
        ("ridge", ridge_vals, "target", include_ridge),
    )
    pairs: list[tuple[str, float]] = []
    provenance: dict[str, str] = {}
    family_means: dict[str, float] = {}
    for label, vals, unit, included in groups:
        if not included:
            provenance[label] = "off"
            continue
        if not vals:
            provenance[label] = "absent"
            continue
        pairs.extend((label, v) for v in vals)
        family_means[label] = statistics.fmean(vals)
        provenance[label] = (
            f"{family_means[label]:.6g} ({len(vals)} {unit}{'' if len(vals) == 1 else 's'})"
        )

    if position_reported and not position_included:
        provenance["position_regression"] = "excluded_no_200ms"

    # Every pair in a group shares one family key, so each family contributes weight 1
    # regardless of how many sessions or targets it holds. A single surviving family
    # carries the column alone.
    mean = family_weighted_mean(pairs)
    if mean is None:
        return None, None
    return mean, ColumnDiagnostics(
        score=mean,
        n_metrics=len(per_metric),
        normalization="raw_r2;family_weighted(poyo,ridge)",
        per_metric=per_metric,
        provenance=provenance,
        family_means=family_means,
    )


_STRUCTURE_PRESERVATION_METRICS: frozenset[str] = frozenset(
    {
        "population_structure_preservation",
        "per_unit_structure_preservation",
    }
)


def score_latent_structure_and_robustness(
    comparisons: dict[str, MetricComparison],
    m1_report: dict | None,
    m2_report: dict | None,
    *,
    recon_report: dict | None = None,
    include_m2: bool = False,
    m2_metric: str = "acg_correlation",
    m2_epsilon: float = 0.1,
    include_m1: bool = True,
) -> ColumnDiagnostics | None:
    """Family-weighted mean over the latent bucket's members, each mapped onto
    [0,1] by a monotonic bounded transform (NO Poisson normalization, NO clip):

      - structure-preservation: ``max(corr, 0)`` on the raw RSA ``model_value``;
      - M1 input-corruption: retention ``1/(1+d)`` of the canonical components' mean;
      - M2 latent-noise (opt-in via ``include_m2``): ``max(acg, 0)`` of the chosen
        metric at ``m2_epsilon``.

    The correlation-valued members (structure, M2) pin zero at correlation 0 — a
    non-positive correlation is "no signal", not partial credit.

    Each member is its own family (weight 1); the score is the mean over families.
    Returns ``None`` if no member contributes.
    """
    per_metric: dict[str, MetricScore] = {}
    pairs: list[tuple[str, float]] = []

    # Raw RSA validity is independent of floor/ceiling validity. Prefer the recon
    # report, including metrics absent from the comparison's three-way intersection.
    # Without a recon report entry, the comparison's model value is used.
    results = _report_results(recon_report)
    for name in sorted(_STRUCTURE_PRESERVATION_METRICS):
        comp = comparisons.get(name)
        if name in results:
            entry = results[name]
            corr = _finite_float(entry.get("raw_value")) if isinstance(entry, dict) else None
        else:
            corr = _finite_float(comp.model_value) if comp is not None else None
        if corr is None:
            continue
        bounded = max(float(corr), 0.0)
        per_metric[name] = MetricScore(
            normalized=bounded,
            clipped=False,
            raw_value=float(corr),
            direction="higher",
        )
        pairs.append((name, bounded))

    # M1 input-corruption retention 1/(1+d) of the canonical components' mean.
    # include_m1=False leaves input-corruption robustness out (provenance "off") — used
    # to drop it for every tokenizer of a table when only some tokenizers have it.
    if include_m1:
        m1_retention, m1_d, m1_prov = resolve_m1_retention(m1_report)
    else:
        m1_retention, m1_d, m1_prov = None, None, "off"
    if m1_retention is not None:
        key = "m1:components_mean"
        per_metric[key] = MetricScore(
            normalized=m1_retention,
            clipped=False,
            raw_value=float(m1_d) if m1_d is not None else float("nan"),
            direction="higher",
        )
        pairs.append((key, m1_retention))

    # M2 latent-noise robustness — gated on include_m2 (this pure-library default
    # is False so direct callers opt in explicitly; the production job defaults it
    # to True). include_m2/m2_epsilon are GLOBAL knobs (uniform across all
    # tokenizers in a comparison).
    if include_m2:
        m2_bounded, m2_acg, m2_prov, m2_name = resolve_m2_acg(m2_report, m2_metric, m2_epsilon)
        if m2_bounded is not None:
            key = f"m2:{m2_name}"
            per_metric[key] = MetricScore(
                normalized=m2_bounded,
                clipped=False,
                raw_value=float(m2_acg) if m2_acg is not None else float("nan"),
                direction="higher",
            )
            pairs.append((key, m2_bounded))
    else:
        m2_prov = "off"

    if not pairs:
        return None

    provenance = {"m1": m1_prov, "m2": m2_prov}
    normalization = (
        f"denorm_bounded;family_weighted;m1=1/(1+d)[{m1_prov}];structure=max(corr,0);m2={m2_prov}"
    )
    return ColumnDiagnostics(
        score=family_weighted_mean(pairs),
        n_metrics=len(per_metric),
        normalization=normalization,
        per_metric=per_metric,
        provenance=provenance,
    )
