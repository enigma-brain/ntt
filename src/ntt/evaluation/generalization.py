"""Generalization retention score for a single tokenizer's per-session eval reports.

Entry points, all scoring per-session values:

- ``compute_from_session_tiers``: in-memory, with each session's tier given
  (the eval job).
- ``compute_from_per_session_reports``: in-memory ``results.json["per_session"]``,
  tiers resolved from the composite selection YAMLs (the aggregation job).
- ``compute_from_per_session_dir``: per-session JSON files, tiers resolved the
  same way (the standalone script).

Retention of a value against a baseline value is direction-aware:

- ``direction = HIGHER_IS_BETTER`` -> ``retention = value / baseline``
- ``direction = LOWER_IS_BETTER`` -> ``retention = baseline / value``

and clipped to ``[0, 1]`` by default; the unclipped value is also reported as
``retention_raw``. Two baselines (``baseline_mode``):

- ``brainset_matched`` (default): each held-out session is compared with the
  ``xtime`` session(s) of its own brainset, and a tier's retention for a metric
  is the mean of these per-session retentions. Sessions of one brainset share
  the recording setup, unit selection and firing-rate range, which set the
  scale of most metrics; a pooled baseline would instead make the score depend
  on which brainsets a tier holds. A session whose brainset has no ``xtime``
  session (``xsource``) is compared with the mean over all ``xtime`` sessions.
  A session is left out when its baseline is not positive (a ratio to 0 is
  undefined, and a ratio to a negative correlation inverts: a worse session
  would score higher), or when its brainset has ``xtime`` sessions but none
  with a value for the metric.
- ``tier_mean``: the mean over the tier's sessions against the mean over all
  ``xtime`` sessions.

The overall ``generalization_score`` is the geometric mean over the
(tier x metric) retention terms of the ``score_tiers`` (``xsource`` by default,
``DEFAULT_SCORE_TIERS``), floored by ``retention_floor`` for numerical
stability. It is scored over the reconstruction metrics only
(``DEFAULT_EXCLUDED_CATEGORIES``). Per-metric rows and per-tier summaries cover
every tier in ``ood_tiers``.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

log = logging.getLogger(__name__)

DEFAULT_EXCLUDED_METRICS: tuple[str, ...] = ("n_skipped", "token_dims_per_second")
# Generalization asks whether *reconstruction* holds up out of distribution, so it is
# scored over the metrics every dataset has. The latent-quality metrics are
# signed RSA correlations near zero, so their ratio to the baseline is close to a
# 0-or-1 coin flip. The decoding metrics exist only for datasets with behaviour,
# so they would weight the score towards those datasets.
DEFAULT_EXCLUDED_CATEGORIES: tuple[str, ...] = (
    "latent_quality",
    "downstream_encoding",
    "downstream_decoding",
)
DEFAULT_EXCLUDED_BRAINSETS: tuple[str, ...] = ("tostado_marcos_birdsong_2024",)
DEFAULT_RETENTION_FLOOR: float = 1e-3
DEFAULT_OOD_TIERS: tuple[str, ...] = ("xsession", "xsubject", "xsource")
# Only unseen datasets enter the score. On held-out sessions and subjects of the
# training datasets, reconstruction does not measurably degrade (clipping aside,
# retention is centred above 1), and a model's retention on one such session does
# not predict its retention on another, so those tiers would add noise.
DEFAULT_SCORE_TIERS: tuple[str, ...] = ("xsource",)
BaselineMode = Literal["brainset_matched", "tier_mean"]
DEFAULT_BASELINE_MODE: BaselineMode = "brainset_matched"


# -- Pure math --------------------------------------------------------


def compute_retention(
    baseline_value: float,
    tier_value: float,
    direction: str,
    *,
    clip_improvements: bool = True,
) -> tuple[float, float]:
    if direction == "higher":
        if baseline_value == 0:
            raw = 1.0 if tier_value >= baseline_value else 0.0
        else:
            raw = tier_value / baseline_value
    elif direction == "lower":
        if baseline_value == 0:
            raw = 1.0 if tier_value <= baseline_value else 0.0
        elif tier_value == 0:
            raw = 1.0
        else:
            raw = baseline_value / tier_value
    else:
        raise ValueError(f"Unknown metric direction: {direction!r}")

    clipped = min(max(raw, 0.0), 1.0) if clip_improvements else raw
    return raw, clipped


def _geometric_mean(values: list[float], *, floor: float) -> float | None:
    if not values:
        return None
    if floor < 0:
        raise ValueError(f"retention floor must be non-negative, got {floor}")
    floored = [max(value, floor) for value in values]
    if any(value <= 0 for value in floored):
        return 0.0
    return math.exp(statistics.fmean(math.log(value) for value in floored))


# -- Result type ------------------------------------------------------


@dataclass(frozen=True)
class GeneralizationResult:
    per_metric: list[dict[str, Any]]
    summary: dict[str, Any]
    baseline_tier: str
    ood_tiers: tuple[str, ...]
    excluded_metrics: tuple[str, ...]
    excluded_categories: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_tier": self.baseline_tier,
            "ood_tiers": list(self.ood_tiers),
            "excluded_metrics": list(self.excluded_metrics),
            "excluded_categories": list(self.excluded_categories),
            "summary": self.summary,
            "per_metric": self.per_metric,
        }


# -- Retention terms ------------------------------------------------


def _compute_from_lookup(
    metric_lookup: dict[tuple[str, str], dict[str, Any]],
    *,
    baseline_tier: str,
    ood_tiers: tuple[str, ...],
    score_tiers: tuple[str, ...],
    clip_improvements: bool,
    retention_floor: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baseline_metrics = {m for (t, m) in metric_lookup if t == baseline_tier}
    rows: list[dict[str, Any]] = []
    for tier in ood_tiers:
        tier_metrics = {m for (t, m) in metric_lookup if t == tier}
        for metric in sorted(baseline_metrics & tier_metrics):
            baseline = metric_lookup[(baseline_tier, metric)]
            tier_row = metric_lookup[(tier, metric)]
            if baseline["direction"] != tier_row["direction"]:
                raise ValueError(f"Metric {metric!r} direction differs between tiers")
            raw, clipped = compute_retention(
                baseline["value"],
                tier_row["value"],
                baseline["direction"],
                clip_improvements=clip_improvements,
            )
            rows.append(
                {
                    "metric": metric,
                    "category": baseline["category"],
                    "direction": baseline["direction"],
                    "baseline_tier": baseline_tier,
                    "baseline_value": baseline["value"],
                    "baseline_n_sessions": baseline["n_sessions"],
                    "tier": tier,
                    "tier_value": tier_row["value"],
                    "tier_n_sessions": tier_row["n_sessions"],
                    "retention_raw": raw,
                    "retention": clipped,
                }
            )

    return rows, _summarize(
        rows, ood_tiers=ood_tiers, score_tiers=score_tiers, retention_floor=retention_floor
    )


def _matched_rows(
    observations: list[MetricObservation],
    *,
    baseline_tier: str,
    ood_tiers: tuple[str, ...],
    clip_improvements: bool,
) -> list[dict[str, Any]]:
    """Retention rows with each OOD session compared to its brainset's baseline sessions.

    ``baseline_value`` / ``tier_value`` are still the tier means, for reference;
    ``retention`` is the mean of the per-session retentions, not their ratio.
    """
    _, lookup = aggregate_metric_values(observations)
    baseline_brainsets = {obs.brainset for obs in observations if obs.tier == baseline_tier}
    grouped: dict[tuple[str, str], list[MetricObservation]] = {}
    for obs in observations:
        grouped.setdefault((obs.tier, obs.metric), []).append(obs)

    rows: list[dict[str, Any]] = []
    for tier in ood_tiers:
        for metric in sorted(
            {m for (t, m) in lookup if t == baseline_tier} & {m for (t, m) in lookup if t == tier}
        ):
            baseline = lookup[(baseline_tier, metric)]
            tier_row = lookup[(tier, metric)]
            if baseline["direction"] != tier_row["direction"]:
                raise ValueError(f"Metric {metric!r} direction differs between tiers")
            by_brainset: dict[str, list[float]] = {}
            for obs in grouped[(baseline_tier, metric)]:
                by_brainset.setdefault(obs.brainset, []).append(obs.raw_value)
            raws: list[float] = []
            clipped: list[float] = []
            n_matched = n_skipped = 0
            for obs in grouped[(tier, metric)]:
                matched = obs.brainset in by_brainset
                if matched:
                    reference = statistics.fmean(by_brainset[obs.brainset])
                elif obs.brainset in baseline_brainsets:
                    n_skipped += 1  # the brainset's baseline sessions lack this metric
                    continue
                else:
                    reference = baseline["value"]
                if reference <= 0:
                    n_skipped += 1
                    continue
                if matched:
                    n_matched += 1
                raw, kept = compute_retention(
                    reference,
                    obs.raw_value,
                    baseline["direction"],
                    clip_improvements=clip_improvements,
                )
                raws.append(raw)
                clipped.append(kept)
            if not clipped:
                continue
            rows.append(
                {
                    "metric": metric,
                    "category": baseline["category"],
                    "direction": baseline["direction"],
                    "baseline_tier": baseline_tier,
                    "baseline_value": baseline["value"],
                    "baseline_n_sessions": baseline["n_sessions"],
                    "tier": tier,
                    "tier_value": tier_row["value"],
                    "tier_n_sessions": tier_row["n_sessions"],
                    "tier_n_matched": n_matched,
                    "tier_n_skipped": n_skipped,
                    "retention_raw": statistics.fmean(raws),
                    "retention": statistics.fmean(clipped),
                }
            )
    return rows


def _summarize(
    rows: list[dict[str, Any]],
    *,
    ood_tiers: tuple[str, ...],
    score_tiers: tuple[str, ...],
    retention_floor: float,
) -> dict[str, Any]:
    if not set(score_tiers) <= set(ood_tiers):
        raise ValueError(f"score_tiers {score_tiers} must be among ood_tiers {ood_tiers}")
    tier_scores: dict[str, dict[str, float | int | None]] = {}
    for tier in ood_tiers:
        values = [r["retention"] for r in rows if r["tier"] == tier]
        tier_scores[tier] = {
            "n_metrics": len(values),
            "mean_retention": statistics.fmean(values) if values else None,
            "geometric_mean_retention": _geometric_mean(values, floor=retention_floor),
        }
    score_values = [r["retention"] for r in rows if r["tier"] in score_tiers]
    return {
        "generalization_score": _geometric_mean(score_values, floor=retention_floor),
        "mean_retention": statistics.fmean(score_values) if score_values else None,
        "n_retention_terms": len(score_values),
        "score_tiers": list(score_tiers),
        "retention_floor": retention_floor,
        "tier_scores": tier_scores,
    }


# -- File-system entry point + reusable loaders -----------------------


@dataclass(frozen=True)
class SessionId:
    brainset: str
    session: str

    @property
    def key(self) -> str:
        return f"{self.brainset}__{self.session}"


@dataclass(frozen=True)
class MetricObservation:
    session_key: str
    brainset: str
    tier: str
    metric: str
    raw_value: float
    direction: str
    category: str | None


def parse_session_filename(path: Path) -> SessionId:
    stem = path.stem
    if "__" not in stem:
        raise ValueError(
            f"Per-session filename {path.name!r} does not contain the expected "
            "'<brainset>__<session>.json' separator."
        )
    brainset, session = stem.split("__", 1)
    return SessionId(brainset=brainset, session=session)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text())
    return loaded if isinstance(loaded, dict) else {}


def _explicit_session_pairs(path: Path) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    config = _load_yaml(path)
    for block in config.get("selection") or []:
        brainset = block.get("brainset")
        if not brainset:
            continue
        # str(): YAML reads an unquoted numeric session id (e.g. Tolias
        # 3649771255234) as an int, while results.json carries it as a string.
        for session in block.get("sessions") or []:
            pairs.add((brainset, str(session)))
    return pairs


def _build_tier_index_from_selection_root(
    selection_root: Path,
    *,
    selection_family: str,
    split: str,
    ood_tiers: tuple[str, ...],
) -> tuple[dict[tuple[str, str], str], dict[str, set[str]]]:
    """Build the (explicit OOD pairs, xtime excludes) maps from a selection root.

    Internal helper. Callers in this module pass ``selection_root`` directly;
    the public ``build_tier_index`` derives it from a ``config_root``.
    """
    explicit: dict[tuple[str, str], str] = {}
    for tier in ood_tiers:
        path = selection_root / f"{selection_family}_{split}_{tier}.yaml"
        for pair in _explicit_session_pairs(path):
            explicit[pair] = tier

    xtime_path = selection_root / f"{selection_family}_val_xtime.yaml"
    xtime_config = _load_yaml(xtime_path)
    xtime_excludes: dict[str, set[str]] = {}
    for block in xtime_config.get("selection") or []:
        brainset = block.get("brainset")
        if brainset:
            # str(): same unquoted-numeric-id issue as in _explicit_session_pairs.
            xtime_excludes[brainset] = {str(s) for s in block.get("exclude_sessions") or []}
    return explicit, xtime_excludes


def build_tier_index(
    config_root: Path,
    *,
    split: str,
    selection_family: str,
    ood_tiers: tuple[str, ...] = DEFAULT_OOD_TIERS,
) -> tuple[dict[tuple[str, str], str], dict[str, set[str]]]:
    """Build explicit OOD tier mapping plus xtime exclusion blocks.

    ``xsession``, ``xsubject``, and ``xsource`` are explicit session lists.
    ``xtime`` is represented as train sessions minus the held-out-session
    exclusions, so we keep its per-brainset exclusion sets for fallback
    classification.
    """
    selection_root = Path(config_root) / "data" / "selections" / "composite"
    return _build_tier_index_from_selection_root(
        selection_root,
        selection_family=selection_family,
        split=split,
        ood_tiers=tuple(ood_tiers),
    )


def classify_session(
    session: SessionId,
    explicit_tiers: dict[tuple[str, str], str],
    xtime_excludes: dict[str, set[str]],
) -> str | None:
    pair = (session.brainset, session.session)
    in_xtime = (
        session.brainset in xtime_excludes
        and session.session not in xtime_excludes[session.brainset]
    )
    if pair in explicit_tiers:
        # The explicit OOD tier wins (documented precedence, also in
        # tiering.assign_sessions_to_tiers). But a held-out session should be in its
        # brainset's xtime exclude list; if not, the selection YAMLs disagree, so say so.
        if in_xtime:
            log.warning(
                "Session %s is listed in the %r tier but not excluded from xtime; "
                "assigning it to %r. Check the selection YAMLs.",
                session.key,
                explicit_tiers[pair],
                explicit_tiers[pair],
            )
        return explicit_tiers[pair]
    if in_xtime:
        return "xtime"
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        f = float(value)
        return f if math.isfinite(f) else None
    return None


def _observations_from_results(
    session: SessionId,
    tier: str,
    results: dict,
    *,
    excluded_metrics: set[str],
    excluded_categories: set[str],
) -> list[MetricObservation]:
    """Build MetricObservations from one session's ``results`` map.

    Shared by the file-based (``load_observations``) and in-memory
    (``compute_from_per_session_reports``) entry points so they classify and
    filter metrics identically.
    """
    out: list[MetricObservation] = []
    if not isinstance(results, dict):
        return out
    for metric, result in results.items():
        if metric in excluded_metrics or not isinstance(result, dict):
            continue
        if result.get("category") in excluded_categories:
            continue
        value = _as_float(result.get("raw_value"))
        direction = result.get("direction")
        if value is None or direction not in {"higher", "lower"}:
            continue
        out.append(
            MetricObservation(
                session_key=session.key,
                brainset=session.brainset,
                tier=tier,
                metric=metric,
                raw_value=value,
                direction=direction,
                category=result.get("category"),
            )
        )
    return out


def load_observations(
    per_session_dir: Path,
    explicit_tiers: dict[tuple[str, str], str],
    xtime_excludes: dict[str, set[str]],
    *,
    excluded_brainsets: set[str],
    excluded_metrics: set[str],
    excluded_categories: Iterable[str] = DEFAULT_EXCLUDED_CATEGORIES,
) -> tuple[list[MetricObservation], list[dict[str, str]]]:
    """Walk per-session JSON reports, classify, return observations + sessions table.

    The sessions table includes one row per JSON file, with ``tier`` set
    to the resolved OOD/xtime label, ``"excluded"`` if the brainset is in
    ``excluded_brainsets``, or ``"unmatched"`` if the session is neither
    in an explicit OOD set nor in the xtime fallback.
    """
    observations: list[MetricObservation] = []
    sessions: list[dict[str, str]] = []

    files = sorted(Path(per_session_dir).glob("*.json"))
    if not files:
        raise ValueError(f"No JSON reports found in {per_session_dir}")

    for path in files:
        session = parse_session_filename(path)
        is_excluded = session.brainset in excluded_brainsets
        tier = None if is_excluded else classify_session(session, explicit_tiers, xtime_excludes)
        sessions.append(
            {
                "session_key": session.key,
                "brainset": session.brainset,
                "session": session.session,
                "tier": "excluded" if is_excluded else tier or "unmatched",
                "report_file": path.name,
            }
        )
        if tier is None:
            continue

        with path.open() as f:
            payload = json.load(f)
        results = payload.get("results", payload)
        observations.extend(
            _observations_from_results(
                session,
                tier,
                results,
                excluded_metrics=excluded_metrics,
                excluded_categories=set(excluded_categories),
            )
        )
    return observations, sessions


def aggregate_metric_values(
    observations: list[MetricObservation],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    """Group ``observations`` by (tier, metric); return (rows, lookup).

    Rows are returned in ``(tier, metric)`` sort order so downstream CSVs
    are stable across runs.
    """
    grouped: dict[tuple[str, str], list[MetricObservation]] = {}
    for obs in observations:
        grouped.setdefault((obs.tier, obs.metric), []).append(obs)

    rows: list[dict[str, Any]] = []
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for (tier, metric), items in sorted(grouped.items()):
        directions = {item.direction for item in items}
        if len(directions) != 1:
            raise ValueError(f"Metric {metric!r} has inconsistent directions: {directions}")
        categories = {item.category for item in items if item.category is not None}
        row = {
            "tier": tier,
            "metric": metric,
            "direction": next(iter(directions)),
            "category": sorted(categories)[0] if categories else None,
            "value": statistics.fmean(item.raw_value for item in items),
            "n_sessions": len({item.session_key for item in items}),
        }
        rows.append(row)
        lookup[(tier, metric)] = row
    return rows, lookup


def compute_generalization(
    metric_lookup: dict[tuple[str, str], dict[str, Any]],
    *,
    baseline_tier: str,
    ood_tiers: tuple[str, ...] = DEFAULT_OOD_TIERS,
    score_tiers: tuple[str, ...] = DEFAULT_SCORE_TIERS,
    clip_improvements: bool = True,
    retention_floor: float = DEFAULT_RETENTION_FLOOR,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Per-metric retention rows and a summary from tier means (``baseline_mode="tier_mean"``).

    ``metric_lookup`` is the second value of :func:`aggregate_metric_values`. It
    holds no per-session values, so it cannot give the brainset-matched baseline;
    use :func:`compute_from_observations` for that.
    """
    return _compute_from_lookup(
        metric_lookup,
        baseline_tier=baseline_tier,
        ood_tiers=tuple(ood_tiers),
        score_tiers=tuple(score_tiers),
        clip_improvements=clip_improvements,
        retention_floor=retention_floor,
    )


def compute_from_observations(
    observations: list[MetricObservation],
    *,
    baseline_tier: str,
    ood_tiers: tuple[str, ...] = DEFAULT_OOD_TIERS,
    score_tiers: tuple[str, ...] = DEFAULT_SCORE_TIERS,
    baseline_mode: BaselineMode = DEFAULT_BASELINE_MODE,
    clip_improvements: bool = True,
    retention_floor: float = DEFAULT_RETENTION_FLOOR,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Per-metric retention rows and a summary from per-session observations."""
    if baseline_mode == "brainset_matched":
        rows = _matched_rows(
            observations,
            baseline_tier=baseline_tier,
            ood_tiers=tuple(ood_tiers),
            clip_improvements=clip_improvements,
        )
        summary = _summarize(
            rows,
            ood_tiers=tuple(ood_tiers),
            score_tiers=tuple(score_tiers),
            retention_floor=retention_floor,
        )
    elif baseline_mode == "tier_mean":
        _, lookup = aggregate_metric_values(observations)
        rows, summary = compute_generalization(
            lookup,
            baseline_tier=baseline_tier,
            ood_tiers=tuple(ood_tiers),
            score_tiers=tuple(score_tiers),
            clip_improvements=clip_improvements,
            retention_floor=retention_floor,
        )
    else:
        raise ValueError(f"Unknown baseline_mode: {baseline_mode!r}")
    summary["baseline_mode"] = baseline_mode
    return rows, summary


def compute_from_per_session_dir(
    per_session_dir: Path,
    *,
    selection_root: Path,
    selection_family: str,
    split: Literal["val", "test"] = "val",
    baseline_tier: str = "xtime",
    ood_tiers: tuple[str, ...] = DEFAULT_OOD_TIERS,
    excluded_metrics: Iterable[str] = DEFAULT_EXCLUDED_METRICS,
    excluded_categories: Iterable[str] = DEFAULT_EXCLUDED_CATEGORIES,
    excluded_brainsets: Iterable[str] = DEFAULT_EXCLUDED_BRAINSETS,
    score_tiers: tuple[str, ...] = DEFAULT_SCORE_TIERS,
    baseline_mode: BaselineMode = DEFAULT_BASELINE_MODE,
    clip_improvements: bool = True,
    retention_floor: float = DEFAULT_RETENTION_FLOOR,
) -> tuple[GeneralizationResult, list[dict[str, str]]]:
    """Compute generalization from a directory of per-session JSON reports.

    Returns the GeneralizationResult plus a session-classification table
    (one row per JSON file, including sessions classified as "unmatched"
    or "excluded").
    """
    excluded_metric_set = set(excluded_metrics)
    excluded_category_set = set(excluded_categories)
    excluded_brainset_set = set(excluded_brainsets)

    explicit, xtime_excludes = _build_tier_index_from_selection_root(
        selection_root,
        selection_family=selection_family,
        split=split,
        ood_tiers=tuple(ood_tiers),
    )

    observations, sessions_table = load_observations(
        per_session_dir,
        explicit,
        xtime_excludes,
        excluded_brainsets=excluded_brainset_set,
        excluded_metrics=excluded_metric_set,
        excluded_categories=excluded_category_set,
    )
    if not any(obs.tier == baseline_tier for obs in observations):
        raise ValueError(
            f"baseline tier {baseline_tier!r} has no metrics in "
            f"{per_session_dir} (consult the session table for classification)."
        )

    rows, summary = compute_from_observations(
        observations,
        baseline_tier=baseline_tier,
        ood_tiers=tuple(ood_tiers),
        score_tiers=score_tiers,
        baseline_mode=baseline_mode,
        clip_improvements=clip_improvements,
        retention_floor=retention_floor,
    )
    return (
        GeneralizationResult(
            per_metric=rows,
            summary=summary,
            baseline_tier=baseline_tier,
            ood_tiers=tuple(ood_tiers),
            excluded_metrics=tuple(sorted(excluded_metric_set)),
            excluded_categories=tuple(sorted(excluded_category_set)),
        ),
        sessions_table,
    )


def compute_from_session_tiers(
    per_session_reports: dict[str, dict],
    session_tiers: dict[str, str],
    *,
    baseline_tier: str = "xtime",
    ood_tiers: tuple[str, ...] = DEFAULT_OOD_TIERS,
    excluded_metrics: Iterable[str] = DEFAULT_EXCLUDED_METRICS,
    excluded_categories: Iterable[str] = DEFAULT_EXCLUDED_CATEGORIES,
    score_tiers: tuple[str, ...] = DEFAULT_SCORE_TIERS,
    baseline_mode: BaselineMode = DEFAULT_BASELINE_MODE,
    clip_improvements: bool = True,
    retention_floor: float = DEFAULT_RETENTION_FLOOR,
) -> GeneralizationResult | None:
    """Compute generalization from per-session reports whose tiers are already known.

    ``per_session_reports`` is keyed by ``"<brainset>/<session>"`` (the shape of
    ``results.json["per_session"]``); each value is a report dict carrying a
    ``results`` map. ``session_tiers`` maps a session id to its tier; sessions
    without one are left out. Returns ``None`` when the result is not trustworthy:
    the baseline tier has no metrics, or no OOD retention terms were produced.
    """
    excluded_metric_set = set(excluded_metrics)
    excluded_category_set = set(excluded_categories)
    observations: list[MetricObservation] = []
    for sid, tier in session_tiers.items():
        entry = per_session_reports.get(sid)
        if not isinstance(entry, dict):
            continue
        brainset, _, session_name = sid.partition("/")
        observations.extend(
            _observations_from_results(
                SessionId(brainset=brainset, session=session_name),
                tier,
                entry.get("results", entry),
                excluded_metrics=excluded_metric_set,
                excluded_categories=excluded_category_set,
            )
        )
    if not any(obs.tier == baseline_tier for obs in observations):
        return None

    rows, summary = compute_from_observations(
        observations,
        baseline_tier=baseline_tier,
        ood_tiers=tuple(ood_tiers),
        score_tiers=score_tiers,
        baseline_mode=baseline_mode,
        clip_improvements=clip_improvements,
        retention_floor=retention_floor,
    )
    if summary["n_retention_terms"] == 0:
        return None
    return GeneralizationResult(
        per_metric=rows,
        summary=summary,
        baseline_tier=baseline_tier,
        ood_tiers=tuple(ood_tiers),
        excluded_metrics=tuple(sorted(excluded_metric_set)),
        excluded_categories=tuple(sorted(excluded_category_set)),
    )


def compute_from_per_session_reports(
    per_session_reports: dict[str, dict],
    *,
    explicit_tiers: dict[tuple[str, str], str],
    xtime_excludes: dict[str, set[str]],
    baseline_tier: str = "xtime",
    ood_tiers: tuple[str, ...] = DEFAULT_OOD_TIERS,
    excluded_metrics: Iterable[str] = DEFAULT_EXCLUDED_METRICS,
    excluded_categories: Iterable[str] = DEFAULT_EXCLUDED_CATEGORIES,
    excluded_brainsets: Iterable[str] = DEFAULT_EXCLUDED_BRAINSETS,
    score_tiers: tuple[str, ...] = DEFAULT_SCORE_TIERS,
    baseline_mode: BaselineMode = DEFAULT_BASELINE_MODE,
    clip_improvements: bool = True,
    retention_floor: float = DEFAULT_RETENTION_FLOOR,
) -> tuple[GeneralizationResult | None, list[dict[str, str]]]:
    """Compute generalization from an in-memory ``per_session`` mapping.

    ``per_session_reports`` is keyed by ``"<brainset>/<session>"`` (the shape of
    ``results.json["per_session"]``); each value is a report dict carrying a
    ``results`` map. Tiers are re-derived from ``explicit_tiers`` / ``xtime_excludes``
    (built by :func:`build_tier_index`), then scored by
    :func:`compute_from_session_tiers`. Returns ``(None, sessions_table)`` when the
    result is not trustworthy.
    """
    excluded_brainset_set = set(excluded_brainsets)

    session_tiers: dict[str, str] = {}
    sessions: list[dict[str, str]] = []
    for sid in per_session_reports:
        brainset, _, session_name = sid.partition("/")
        session = SessionId(brainset=brainset, session=session_name)
        is_excluded = session.brainset in excluded_brainset_set
        tier = None if is_excluded else classify_session(session, explicit_tiers, xtime_excludes)
        sessions.append(
            {
                "session_key": session.key,
                "brainset": session.brainset,
                "session": session.session,
                "tier": "excluded" if is_excluded else tier or "unmatched",
                "report_file": sid,
            }
        )
        if tier is not None:
            session_tiers[sid] = tier

    unmatched = sorted(row["session_key"] for row in sessions if row["tier"] == "unmatched")
    if unmatched:
        log.warning(
            "generalization: %d session(s) match no tier and are left out: %s",
            len(unmatched),
            unmatched,
        )

    result = compute_from_session_tiers(
        per_session_reports,
        session_tiers,
        baseline_tier=baseline_tier,
        ood_tiers=ood_tiers,
        excluded_metrics=excluded_metrics,
        excluded_categories=excluded_categories,
        score_tiers=score_tiers,
        baseline_mode=baseline_mode,
        clip_improvements=clip_improvements,
        retention_floor=retention_floor,
    )
    return result, sessions
