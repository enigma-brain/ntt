"""Eval artifact: serialisable session-aggregated eval results.

Used both for early model-eval persistence (durability — written before
the high-OOM-risk Poisson phase) and as the publish/consume format for
the standalone Poisson-baseline job.
"""

from __future__ import annotations

import json
import subprocess
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pydantic as pdt

from ntt.evaluation.neuron_selection import NeuronSelection, check_neuron_selection
from ntt.evaluation.utils.paths import resolve_artifact_path
from ntt.evaluation.utils.serialization import _json_default as _eval_json_default

SCHEMA_VERSION = 1

# Hard-required structural fields — raise on mismatch
_HARD_FIELDS: tuple[str, ...] = (
    "bin_size_ms",
    "rate_estimation_bin_ms",
    "split",
    "max_eval_seconds",
    "decoding_bypass_fields",
)
# Hard fields that are list-typed and should be compared as unordered sets
# (the producer sorts these, but the contract is set-equality so consumers
# don't get spurious mismatches from a different ordering).
_HARD_FIELDS_SET_COMPARE: frozenset[str] = frozenset({"decoding_bypass_fields"})
# Soft fields — warn on mismatch.
#
# ``window_length_seconds`` is soft on purpose: a shared Poisson reference keeps
# its own window (0.1 s for the published references) whatever window a model is
# scored on. Metrics on the **segment-level concatenated** time axis (spike
# statistics, correlations, population metrics; anything that consumes
# ``CollatedValidationSegment.original`` / ``.reconstructed``) only see boundary
# effects. The two structure-preservation metrics (population / per-unit) do
# depend on it: they compare per-window vectors, so their Poisson values are the
# reference window's. The warning keeps that visible.
_SOFT_FIELDS: tuple[str, ...] = ("n_realisations", "window_length_seconds")
_SOFT_FIELD_NOTES: dict[str, str] = {
    "n_realisations": "metrics still comparable; baseline noisier if smaller",
    "window_length_seconds": (
        "metrics on segment-level concatenated tensors are unaffected; the "
        "structure-preservation metrics compare per-window vectors and use the "
        "reference's window"
    ),
}


class EvalArtifactMetadata(pdt.BaseModel, frozen=True):
    """Metadata describing the run that produced an EvalArtifact."""

    schema_version: int = SCHEMA_VERSION
    kind: Literal["model_eval", "poisson_baseline"]
    bin_size_ms: float
    window_length_seconds: float
    rate_estimation_bin_ms: float | None = None
    split: Literal["val", "test"]
    max_eval_seconds: float | None
    decoding_bypass_fields: list[str]
    sessions: list[str]
    n_realisations: int | None = None
    seed: int | None = None
    git_sha: str | None = None
    timestamp: str  # ISO 8601
    # Cross-session aggregation strategy of ``aggregated``/``perfect_aggregated``, so a
    # consumer can check it against its own. None in artifacts written before it existed.
    aggregation_strategy: str | None = None
    # Units each session was scored on; ``validate_against`` requires the consumer's to
    # match. None in artifacts written before it existed (a consumer then fails, or warns
    # in padded mode).
    neuron_selection: NeuronSelection | None = None


class EvalArtifact(pdt.BaseModel):
    """Serialisable session-aggregated eval results.

    ``per_session`` / ``aggregated`` hold model-dump'd metric reports.
    ``perfect_*`` fields are populated only for ``kind="poisson_baseline"``.

    ``per_session_weights``: per-session weights = sum of segment time-bin
    counts within the session. Required for a consumer using
    ``WEIGHTED_MEAN`` to reproduce the producer's aggregation exactly;
    when missing, ``aggregate_per_session_dicts`` falls back to equal
    weights and warns. The same weights apply to ``per_session`` and
    ``perfect_per_session`` (both are computed from identical segments).
    """

    metadata: EvalArtifactMetadata
    per_session: dict[str, dict[str, Any]]
    aggregated: dict[str, Any]
    perfect_per_session: dict[str, dict[str, Any]] | None = None
    perfect_aggregated: dict[str, Any] | None = None
    per_session_weights: dict[str, float] | None = None

    def dump_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.model_dump(by_alias=True), default=_eval_json_default, indent=2)
        path.write_text(text)

    @classmethod
    def load_json(cls, path: str | Path) -> EvalArtifact:
        resolved = resolve_artifact_path(str(path))
        try:
            with open(resolved) as f:
                payload = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Corrupt JSON in EvalArtifact at {resolved}: {e}") from e
        try:
            artifact = cls.model_validate(payload)
        except pdt.ValidationError as e:
            raise ValueError(f"Failed to parse EvalArtifact at {resolved}: {e}") from e
        artifact._warn_if_contains_excluded_sessions(source=str(resolved))
        return artifact

    def _warn_if_contains_excluded_sessions(self, source: str = "<artifact>") -> None:
        """Emit a UserWarning if any per-session entry or metadata.session
        belongs to a brainset listed in ``excluded_brainsets``.

        ``aggregated`` cannot be cleaned retroactively — the consumer must
        call ``without_brainsets`` and re-aggregate via
        ``aggregate_per_session_dicts``. The warning makes that requirement
        loud at load time so consumers don't silently keep reporting metrics
        contaminated by an excluded dataset.
        """
        from ntt.evaluation.excluded_brainsets import find_excluded_sessions

        candidates = set(self.metadata.sessions) | set(self.per_session.keys())
        found = find_excluded_sessions(candidates)
        if not found:
            return
        lines = [f"EvalArtifact at {source} contains sessions from excluded brainsets:"]
        for brainset, sids in found.items():
            lines.append(f"  - {brainset}: {len(sids)} session(s)")
        lines.append(
            "These contributed to .aggregated, which cannot be retroactively "
            "filtered. Use .without_brainsets([...]) to drop them from "
            "per_session, then re-aggregate via aggregate_per_session_dicts()."
        )
        warnings.warn("\n".join(lines), UserWarning, stacklevel=3)

    def without_brainsets(self, brainsets: list[str]) -> EvalArtifact:
        """Return a copy with sessions belonging to ``brainsets`` removed.

        Strips matching session IDs from ``per_session``,
        ``per_session_weights``, ``perfect_per_session``, and
        ``metadata.sessions``. Unknown brainset names are silently ignored
        (consistent with ``excluded_brainsets.sessions_for_brainsets``).

        ``aggregated`` and ``perfect_aggregated`` are passed through
        unchanged: re-aggregation depends on the original metric pipeline's
        ``AggregationStrategy`` (and per-session weights), so this method
        cannot do it itself. Re-aggregate by calling
        ``aggregate_per_session_dicts(returned.per_session, strategy,
        weights=returned.per_session_weights)`` after filtering.
        """
        from ntt.evaluation.excluded_brainsets import sessions_for_brainsets

        drop = sessions_for_brainsets(brainsets)
        if not drop:
            return self
        new_per_session = {k: v for k, v in self.per_session.items() if k not in drop}
        new_weights = (
            {k: v for k, v in self.per_session_weights.items() if k not in drop}
            if self.per_session_weights is not None
            else None
        )
        new_perfect = (
            {k: v for k, v in self.perfect_per_session.items() if k not in drop}
            if self.perfect_per_session is not None
            else None
        )
        new_sessions = [s for s in self.metadata.sessions if s not in drop]
        new_metadata = self.metadata.model_copy(update={"sessions": new_sessions})
        return self.model_copy(
            update={
                "metadata": new_metadata,
                "per_session": new_per_session,
                "per_session_weights": new_weights,
                "perfect_per_session": new_perfect,
            }
        )

    def validate_against(
        self, other: EvalArtifactMetadata, *, strict_neuron_selection: bool = True
    ) -> None:
        """Validate that ``other`` (consumer metadata) is compatible with self.metadata.

        The neuron selections must match on the consumer's sessions; a mismatch or a
        missing record raises, or only logs a warning when ``strict_neuron_selection``
        is False (padded-mode consumers).

        Raises:
            ValueError: on hard-field mismatch or session-set superset.
            NeuronSelectionMismatchError: on neuron-selection mismatch (a ValueError).
        Warns:
            UserWarning: on soft-field mismatch.
        """
        baseline = self.metadata
        mismatches = []
        for field in _HARD_FIELDS:
            b_val = getattr(baseline, field)
            c_val = getattr(other, field)
            if field in _HARD_FIELDS_SET_COMPARE:
                differs = set(b_val) != set(c_val)
            else:
                differs = b_val != c_val
            if differs:
                mismatches.append(f"{field}: baseline={b_val!r}, consumer={c_val!r}")
        if mismatches:
            raise ValueError(
                "EvalArtifact metadata mismatch (hard):\n  - " + "\n  - ".join(mismatches)
            )
        baseline_sessions = set(baseline.sessions)
        consumer_sessions = set(other.sessions)
        missing = consumer_sessions - baseline_sessions
        if missing:
            sample = sorted(missing)[:5]
            tail = "..." if len(missing) > 5 else ""
            raise ValueError(
                f"Consumer has {len(missing)} session(s) not in baseline: {sample}{tail}"
            )
        check_neuron_selection(
            baseline.neuron_selection,
            other.neuron_selection,
            what="Poisson reference",
            strict=strict_neuron_selection,
        )
        for field in _SOFT_FIELDS:
            b_val = getattr(baseline, field)
            c_val = getattr(other, field)
            if b_val != c_val:
                note = _SOFT_FIELD_NOTES.get(field, "metrics may differ")
                warnings.warn(
                    f"EvalArtifact {field} mismatch: baseline={b_val}, consumer={c_val} ({note})",
                    UserWarning,
                    stacklevel=2,
                )


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    return None


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def aggregate_per_session_dicts(
    per_session: dict[str, dict[str, Any]],
    strategy,
    weights: dict[str, float] | None = None,
):
    """Re-aggregate per-session artifact dicts into a single cross-session report.

    Used by consumers loading a baseline artifact: they filter the baseline's
    per-session dicts to their own session subset, then call this to recover
    a cross-session aggregate equivalent to ``aggregate_session_reports``.

    ``weights``: optional per-session weights (sum of segment time-bin counts
    per session, as the producer recorded them). Required for
    ``WEIGHTED_MEAN`` to match the inline (in-eval) computation exactly;
    when omitted under ``WEIGHTED_MEAN`` we fall back to equal weights and
    warn. ``MEAN`` and ``MEDIAN`` ignore weights, so omission is harmless
    under those strategies.
    """
    from ntt.evaluation.config.schema import AggregationStrategy
    from ntt.evaluation.core.types import EvaluationReport
    from ntt.evaluation.utils.validation_collation import _aggregate_evaluation_reports

    if not per_session:
        return EvaluationReport(results={})
    sids = list(per_session.keys())
    reports = [EvaluationReport.from_dict(per_session[sid]) for sid in sids]

    if weights is not None:
        report_weights = [float(weights.get(sid, 1.0)) for sid in sids]
    else:
        if strategy == AggregationStrategy.WEIGHTED_MEAN:
            warnings.warn(
                "aggregate_per_session_dicts: WEIGHTED_MEAN requested but no "
                "per-session weights provided; falling back to equal weights. "
                "Result may differ from the inline computation when sessions "
                "had unequal segment counts.",
                UserWarning,
                stacklevel=2,
            )
        report_weights = [1.0] * len(reports)

    return _aggregate_evaluation_reports(reports, report_weights, strategy)


def compute_decoding_bypass_fields(metrics) -> list[str]:
    """Describe which behavioral fields trigger the ``max_eval_seconds`` bypass.

    Mirrors the logic in ``run_validation_evaluation_per_session``: each
    ``DOWNSTREAM_DECODING`` metric with a ``target_field`` contributes the
    fields it cannot compute without (``required_behavioral_fields``), joined
    with ``+``; a session is exempt from the cap only if it carries all fields
    of at least one entry. Returns a sorted, deduplicated list.

    ``metrics``: an iterable of ``(name, metric)`` pairs (the shape of
    ``EvaluationPipeline.metrics``). Producer and consumer both pass their
    own pipeline so ``EvalArtifact.validate_against`` actually checks
    structural agreement (and is not tautologically true because the
    consumer copied the baseline's value into its own metadata).
    """
    from ntt.evaluation.core.types import MetricCategory
    from ntt.evaluation.metrics.downstream_prediction import required_behavioral_fields

    return sorted(
        {
            "+".join(sorted(required_behavioral_fields(m)))
            for _name, m in metrics
            if m.category == MetricCategory.DOWNSTREAM_DECODING
            and getattr(m, "target_field", None)
        }
    )


def compute_per_session_weights(
    session_reports: dict[str, list[tuple[Any, dict[str, Any]]]],
) -> dict[str, float]:
    """Compute per-session weights = sum of segment time-bin counts.

    Mirrors the weight computation inside ``aggregate_session_reports``
    (validation_collation.py) so a producer can record weights into the
    artifact for later use by ``aggregate_per_session_dicts``.
    """
    weights: dict[str, float] = {}
    for sid, segment_list in session_reports.items():
        total_time = 0.0
        for segment, _data in segment_list:
            shape = (
                segment.original_shape
                if getattr(segment, "original_shape", None) is not None
                else (
                    tuple(segment.original.shape)
                    if getattr(segment, "original", None) is not None
                    else None
                )
            )
            if shape is None:
                raise RuntimeError(
                    f"Segment for session {sid!r} has neither original_shape "
                    f"nor original; cannot compute weight."
                )
            total_time += float(shape[1])
        weights[sid] = total_time
    return weights


@dataclass(frozen=True)
class _SyntheticSegment:
    """Placeholder segment for reference-path reconstruction.

    ``poisson_session_reports`` from the inline path holds per-segment data;
    a baseline artifact only stores per-session aggregates, so when a
    consumer loads a reference we surface a synthetic single-segment-per-
    session structure. Downstream consumers that just iterate ``(_, data)``
    pairs and read ``data["report"]`` (the typical pattern: results.json
    save, behavioral plots) work unchanged. Code that re-aggregates from
    the segment-tuple structure with WEIGHTED_MEAN would silently lose
    weighting fidelity here — but no current consumer does that.
    """

    session_id: str
    segment_index: int = 0
    original_shape: tuple | None = None
    original: Any = None


def reconstruct_session_reports_from_per_session(
    per_session: dict[str, dict[str, Any]],
) -> dict[str, list[tuple[_SyntheticSegment, dict[str, Any]]]]:
    """Build a synthetic ``{session_id: [(segment, data)]}`` from a baseline's
    per-session dicts.

    Used by the ``poisson_reference`` branch of the eval job to populate the
    4th return value of ``_run_poisson_comparison`` with a non-empty,
    iteration-compatible structure.
    """
    from ntt.evaluation.core.types import EvaluationReport

    out: dict[str, list[tuple[_SyntheticSegment, dict[str, Any]]]] = {}
    for sid, report_dict in per_session.items():
        out[sid] = [
            (
                _SyntheticSegment(session_id=sid),
                {"report": EvaluationReport.from_dict(report_dict), "metadata": {}},
            )
        ]
    return out


def make_metadata(
    *,
    kind: Literal["model_eval", "poisson_baseline"],
    bin_size_ms: float,
    window_length_seconds: float,
    rate_estimation_bin_ms: float | None,
    split: Literal["val", "test"],
    max_eval_seconds: float | None,
    decoding_bypass_fields: list[str],
    sessions: list[str],
    n_realisations: int | None,
    seed: int | None,
    aggregation_strategy: str | None = None,
    neuron_selection: NeuronSelection | None = None,
) -> EvalArtifactMetadata:
    """Convenience factory that fills in git_sha and timestamp."""
    return EvalArtifactMetadata(
        kind=kind,
        bin_size_ms=bin_size_ms,
        window_length_seconds=window_length_seconds,
        rate_estimation_bin_ms=rate_estimation_bin_ms,
        split=split,
        max_eval_seconds=max_eval_seconds,
        decoding_bypass_fields=list(decoding_bypass_fields),
        sessions=list(sessions),
        n_realisations=n_realisations,
        seed=seed,
        aggregation_strategy=aggregation_strategy,
        neuron_selection=neuron_selection,
        git_sha=_git_sha(),
        timestamp=_now_iso(),
    )
