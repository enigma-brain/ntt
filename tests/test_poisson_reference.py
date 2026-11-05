"""End-to-end test for the produce-then-consume Poisson reference flow.

These tests cover the artifact-level logic (load + validate + filter +
re-aggregate) in isolation. The full NTTEvalJob.poisson_reference branch
that calls into this logic is exercised by integration tests via the
real eval pipeline (out of scope for unit tests; covered by manual
verification on a small synthetic config).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from ntt.evaluation.artifact import (
    EvalArtifact,
    aggregate_per_session_dicts,
    compute_decoding_bypass_fields,
    compute_per_session_weights,
    make_metadata,
    reconstruct_session_reports_from_per_session,
)
from ntt.evaluation.config.schema import AggregationStrategy
from ntt.evaluation.core.types import (
    EvaluationReport,
    MetricCategory,
    MetricDirection,
    MetricLevel,
    MetricResult,
)
from ntt.evaluation.neuron_selection import NeuronSelection, SessionUnits


def _make_metric_result(name: str, value: float) -> MetricResult:
    """Construct a minimal MetricResult for tests."""
    return MetricResult(
        name=name,
        category=MetricCategory.POPULATION_STRUCTURE,
        direction=MetricDirection.HIGHER_IS_BETTER,
        level=MetricLevel.POPULATION,
        raw_value=value,
    )


def _per_session_dump(session_to_value: dict[str, float], metric_name: str = "metric_x"):
    """Build per_session dict in the artifact's serialised form."""
    out = {}
    for sid, value in session_to_value.items():
        report = EvaluationReport(results={metric_name: _make_metric_result(metric_name, value)})
        out[sid] = report.to_dict()
    return out


def _neuron_selection(sessions=("A", "B", "C")) -> NeuronSelection:
    """The same units per session on both sides (a matching record)."""
    return NeuronSelection(
        policy="eval_unit_selection",
        max_neurons=200,
        seed=42,
        padded_to=None,
        sessions={
            sid: SessionUnits(n_units=10, n_selected=10, n_padded=0, unit_ids_sha256=sid * 64)
            for sid in sessions
        },
    )


def _make_baseline_artifact(
    tmp_path,
    sessions=("A", "B", "C"),
    bin_size_ms=1.0,
    per_session_values=None,
    perfect_values=None,
    per_session_weights=None,
):
    if per_session_values is None:
        per_session_values = dict.fromkeys(sessions, 0.5)
    if perfect_values is None:
        perfect_values = dict.fromkeys(sessions, 1.0)
    art = EvalArtifact(
        metadata=make_metadata(
            kind="poisson_baseline",
            bin_size_ms=bin_size_ms,
            window_length_seconds=0.1,
            rate_estimation_bin_ms=10.0,
            split="val",
            max_eval_seconds=600.0,
            decoding_bypass_fields=["cursor_pos", "cursor_vel"],
            sessions=list(sessions),
            n_realisations=5,
            seed=42,
            neuron_selection=_neuron_selection(sessions),
        ),
        per_session=_per_session_dump(per_session_values),
        aggregated=EvaluationReport(
            results={"metric_x": _make_metric_result("metric_x", 0.5)}
        ).to_dict(),
        perfect_per_session=_per_session_dump(perfect_values),
        perfect_aggregated=EvaluationReport(
            results={"metric_x": _make_metric_result("metric_x", 1.0)}
        ).to_dict(),
        per_session_weights=per_session_weights,
    )
    path = tmp_path / "poisson_baseline.json"
    art.dump_json(path)
    return path


def _consumer_metadata(**overrides):
    base = {
        "kind": "poisson_baseline",
        "bin_size_ms": 1.0,
        "window_length_seconds": 0.1,
        "rate_estimation_bin_ms": 10.0,
        "split": "val",
        "max_eval_seconds": 600.0,
        "decoding_bypass_fields": ["cursor_pos", "cursor_vel"],
        "sessions": ["A", "B", "C"],
        "n_realisations": 5,
        "seed": 42,
    }
    base.update(overrides)
    base.setdefault("neuron_selection", _neuron_selection(base["sessions"]))
    return make_metadata(**base)


def test_produce_then_consume_round_trip(tmp_path):
    path = _make_baseline_artifact(tmp_path)
    artifact = EvalArtifact.load_json(path)
    consumer = _consumer_metadata()
    artifact.validate_against(consumer)
    consumer_set = set(consumer.sessions)
    filtered = {sid: r for sid, r in artifact.per_session.items() if sid in consumer_set}
    assert set(filtered.keys()) == consumer_set
    assert artifact.perfect_per_session is not None


def test_consume_subset_re_aggregates_correctly(tmp_path):
    """Filtering to a subset and re-aggregating must give the subset's mean."""
    # A=0.2, B=0.4, C=0.9 — full mean is 0.5; subset {A,B} mean is 0.3
    path = _make_baseline_artifact(
        tmp_path,
        sessions=("A", "B", "C"),
        per_session_values={"A": 0.2, "B": 0.4, "C": 0.9},
    )
    artifact = EvalArtifact.load_json(path)
    consumer = _consumer_metadata(sessions=["A", "B"])
    artifact.validate_against(consumer)
    consumer_set = set(consumer.sessions)
    filtered = {sid: r for sid, r in artifact.per_session.items() if sid in consumer_set}
    aggregated = aggregate_per_session_dicts(filtered, AggregationStrategy.MEAN)
    assert "metric_x" in aggregated.results
    # Mean of 0.2 and 0.4 → 0.3 (within float tolerance)
    assert abs(aggregated.results["metric_x"].raw_value - 0.3) < 1e-9


def test_consume_superset_raises(tmp_path):
    path = _make_baseline_artifact(tmp_path, sessions=("A", "B"))
    artifact = EvalArtifact.load_json(path)
    consumer = _consumer_metadata(sessions=["A", "B", "C"])
    with pytest.raises(ValueError, match="not in baseline"):
        artifact.validate_against(consumer)


def test_consume_bin_size_mismatch_raises(tmp_path):
    path = _make_baseline_artifact(tmp_path, bin_size_ms=1.0)
    artifact = EvalArtifact.load_json(path)
    consumer = _consumer_metadata(bin_size_ms=2.0)
    with pytest.raises(ValueError, match="bin_size_ms"):
        artifact.validate_against(consumer)


def test_consume_n_realisations_mismatch_warns(tmp_path):
    path = _make_baseline_artifact(tmp_path)
    artifact = EvalArtifact.load_json(path)
    consumer = _consumer_metadata(n_realisations=10)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        artifact.validate_against(consumer)
    msgs = [str(x.message) for x in w]
    assert any("n_realisations" in m for m in msgs), msgs


def test_load_via_str_path(tmp_path):
    """The load_json signature accepts str or Path."""
    path = _make_baseline_artifact(tmp_path)
    art1 = EvalArtifact.load_json(str(path))
    art2 = EvalArtifact.load_json(path)
    assert art1.metadata == art2.metadata


def test_aggregate_per_session_dicts_empty():
    """Empty session set → empty report (degenerate but well-defined)."""
    result = aggregate_per_session_dicts({}, AggregationStrategy.MEAN)
    assert result.results == {}


def test_eval_job_has_poisson_reference_field():
    """Smoke check: NTTEvalJob exposes poisson_reference as a config field."""
    from ntt.jobs.eval import NTTEvalJob

    assert "poisson_reference" in NTTEvalJob.model_fields
    assert NTTEvalJob.model_fields["poisson_reference"].default is None
    assert "poisson_n_workers" in NTTEvalJob.model_fields


def test_standalone_poisson_baseline_job_imports_with_required_fields():
    """Smoke check: standalone job is importable and exposes the matching fields."""
    from ntt.jobs.poisson_baseline import NTTPoissonBaselineJob

    fields = NTTPoissonBaselineJob.model_fields
    # Fields a consumer's poisson_reference must validate against
    for name in (
        "bin_size_ms",
        "window_length_seconds",
        "rate_estimation_bin_ms",
        "split",
        "n_realisations",
        "seed",
        "n_workers",
    ):
        assert name in fields, f"missing required field {name}"


# ── per-session weights for WEIGHTED_MEAN ────────────────────────────────


def test_aggregate_per_session_dicts_weighted_mean_uses_provided_weights():
    """With weights, WEIGHTED_MEAN deviates from the equal-weight mean.

    A=0.2 (weight=1), B=0.4 (weight=1), C=0.9 (weight=8).
    Equal-weight MEAN = (0.2 + 0.4 + 0.9) / 3 = 0.5.
    WEIGHTED_MEAN with weights [1, 1, 8] = (0.2 + 0.4 + 8*0.9) / 10 = 0.78.
    """
    per_session = _per_session_dump({"A": 0.2, "B": 0.4, "C": 0.9})
    weights = {"A": 1.0, "B": 1.0, "C": 8.0}
    aggregated = aggregate_per_session_dicts(
        per_session, AggregationStrategy.WEIGHTED_MEAN, weights=weights
    )
    assert "metric_x" in aggregated.results
    assert abs(aggregated.results["metric_x"].raw_value - 0.78) < 1e-9


def test_aggregate_per_session_dicts_warns_when_weights_missing_for_weighted_mean():
    """WEIGHTED_MEAN without weights → fallback + UserWarning."""
    per_session = _per_session_dump({"A": 0.2, "B": 0.4, "C": 0.9})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        aggregated = aggregate_per_session_dicts(
            per_session, AggregationStrategy.WEIGHTED_MEAN, weights=None
        )
    msgs = [str(x.message) for x in w]
    assert any("WEIGHTED_MEAN" in m for m in msgs), msgs
    # Falls back to equal weights → equivalent to MEAN
    assert abs(aggregated.results["metric_x"].raw_value - 0.5) < 1e-9


def test_aggregate_per_session_dicts_silent_when_mean_strategy():
    """No warning when MEAN/MEDIAN are used without weights (weights ignored)."""
    per_session = _per_session_dump({"A": 0.2, "B": 0.4})
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        aggregate_per_session_dicts(per_session, AggregationStrategy.MEAN, weights=None)
    weighted_warnings = [x for x in w if "WEIGHTED_MEAN" in str(x.message)]
    assert weighted_warnings == [], weighted_warnings


def test_artifact_round_trip_preserves_weights(tmp_path):
    """per_session_weights survives dump_json / load_json."""
    weights = {"A": 100.0, "B": 250.0, "C": 75.0}
    path = _make_baseline_artifact(tmp_path, per_session_weights=weights)
    loaded = EvalArtifact.load_json(path)
    assert loaded.per_session_weights == weights


def test_artifact_default_weights_is_none(tmp_path):
    """The weights field is optional and defaults to unweighted."""
    path = _make_baseline_artifact(tmp_path)
    loaded = EvalArtifact.load_json(path)
    assert loaded.per_session_weights is None


def test_compute_per_session_weights_sums_segment_time_bin_counts():
    """Mirror of aggregate_session_reports' weight computation."""

    class _StubSegment:
        def __init__(self, original_shape):
            self.original_shape = original_shape
            self.original = None

    session_reports = {
        "A": [
            (_StubSegment(original_shape=(10, 100)), {}),
            (_StubSegment(original_shape=(10, 50)), {}),
        ],
        "B": [(_StubSegment(original_shape=(20, 1000)), {})],
    }
    weights = compute_per_session_weights(session_reports)
    assert weights == {"A": 150.0, "B": 1000.0}


# ── synthetic per-session-per-segment reconstruction ─────────────────────


def test_reconstruct_session_reports_returns_synthetic_segments():
    """Each session gets a single virtual segment with the deserialised report."""
    per_session = _per_session_dump({"A": 0.2, "B": 0.4})
    out = reconstruct_session_reports_from_per_session(per_session)
    assert set(out.keys()) == {"A", "B"}
    for sid in ("A", "B"):
        assert len(out[sid]) == 1
        segment, data = out[sid][0]
        assert segment.session_id == sid
        assert segment.segment_index == 0
        assert segment.original_shape is None
        assert "report" in data
        assert "metadata" in data
        # The deserialised report carries the metric value back
        assert "metric_x" in data["report"].results


def test_reconstruct_session_reports_compatible_with_save_results_pattern():
    """Downstream consumers that iterate ``(_, data)`` and read
    ``data["report"].to_dict()`` (the results.json save pattern) work
    unchanged on the synthetic structure."""
    per_session = _per_session_dump({"A": 0.2, "B": 0.4})
    reconstructed = reconstruct_session_reports_from_per_session(per_session)
    serialised = {
        sid: [data["report"].to_dict() for _, data in segments]
        for sid, segments in reconstructed.items()
        if segments
    }
    assert set(serialised.keys()) == {"A", "B"}
    assert all(len(reports) == 1 for reports in serialised.values())


# ── set-semantics for decoding_bypass_fields ─────────────────────────────


def test_validate_against_decoding_bypass_fields_order_invariant(tmp_path):
    """Reordering decoding_bypass_fields must not trigger a mismatch."""
    path = _make_baseline_artifact(tmp_path)  # baseline has ["cursor_pos", "cursor_vel"]
    artifact = EvalArtifact.load_json(path)
    # Consumer presents the same set in opposite order
    consumer = _consumer_metadata(decoding_bypass_fields=["cursor_vel", "cursor_pos"])
    artifact.validate_against(consumer)  # must not raise


def test_validate_against_decoding_bypass_fields_actual_diff_raises(tmp_path):
    """Real set difference (not just reorder) must raise."""
    path = _make_baseline_artifact(tmp_path)  # baseline has ["cursor_pos", "cursor_vel"]
    artifact = EvalArtifact.load_json(path)
    consumer = _consumer_metadata(decoding_bypass_fields=["cursor_pos", "choice"])
    with pytest.raises(ValueError, match="decoding_bypass_fields"):
        artifact.validate_against(consumer)


# ── compute_decoding_bypass_fields helper ────────────────────────────────


class _StubMetric:
    """Stand-in for an EvaluationPipeline metric entry."""

    def __init__(self, category, target_field=None):
        self.category = category
        self.target_field = target_field


def test_compute_decoding_bypass_fields_extracts_only_downstream_with_target():
    """Only DOWNSTREAM_PREDICTION metrics with non-empty target_field count."""
    metrics = [
        ("a", _StubMetric(MetricCategory.DOWNSTREAM_DECODING, "cursor_pos")),
        ("b", _StubMetric(MetricCategory.POPULATION_STRUCTURE, "ignored_field")),  # wrong category
        ("c", _StubMetric(MetricCategory.DOWNSTREAM_DECODING, None)),  # no target
        ("d", _StubMetric(MetricCategory.DOWNSTREAM_DECODING, "")),  # empty target (falsy)
        ("e", _StubMetric(MetricCategory.DOWNSTREAM_DECODING, "cursor_vel")),
    ]
    result = compute_decoding_bypass_fields(metrics)
    assert result == ["cursor_pos", "cursor_vel"]


def test_compute_decoding_bypass_fields_empty_when_no_downstream():
    metrics = [
        ("a", _StubMetric(MetricCategory.POPULATION_STRUCTURE, "x")),
        ("b", _StubMetric(MetricCategory.SINGLE_UNIT_STATISTICS, "y")),
    ]
    assert compute_decoding_bypass_fields(metrics) == []


def test_compute_decoding_bypass_fields_dedup_and_sort():
    """Duplicate target fields collapse; output is sorted."""
    metrics = [
        ("a", _StubMetric(MetricCategory.DOWNSTREAM_DECODING, "z")),
        ("b", _StubMetric(MetricCategory.DOWNSTREAM_DECODING, "a")),
        ("c", _StubMetric(MetricCategory.DOWNSTREAM_DECODING, "z")),  # dup
        ("d", _StubMetric(MetricCategory.DOWNSTREAM_DECODING, "m")),
    ]
    assert compute_decoding_bypass_fields(metrics) == ["a", "m", "z"]


# ── rate-override logging helper ─────────────────────────────────────────


def test_maybe_log_rate_override_silent_when_consumer_none(caplog):
    from ntt.jobs.eval import _maybe_log_rate_override

    with caplog.at_level("WARNING"):
        _maybe_log_rate_override(consumer_rate=None, baseline_rate=10.0)
    assert all("rate_bin_ms" not in r.message for r in caplog.records)


def test_maybe_log_rate_override_silent_when_rates_match(caplog):
    from ntt.jobs.eval import _maybe_log_rate_override

    with caplog.at_level("WARNING"):
        _maybe_log_rate_override(consumer_rate=10.0, baseline_rate=10.0)
    assert all("rate_bin_ms" not in r.message for r in caplog.records)


def test_maybe_log_rate_override_warns_when_rates_differ(caplog):
    from ntt.jobs.eval import _maybe_log_rate_override

    with caplog.at_level("WARNING"):
        _maybe_log_rate_override(consumer_rate=5.0, baseline_rate=10.0)
    rate_warnings = [r for r in caplog.records if "Overriding configured" in r.message]
    assert len(rate_warnings) == 1
    assert "5.0" in rate_warnings[0].message
    assert "10.0" in rate_warnings[0].message


# ── per-job partial output dir resolution ────────────────────────────────


def test_resolve_partial_output_dir_uses_explicit_path(tmp_path):
    from ntt.evaluation.utils.paths import resolve_partial_output_dir

    explicit = tmp_path / "my_output_dir"
    assert resolve_partial_output_dir(explicit) == explicit
    assert resolve_partial_output_dir(str(explicit)) == explicit


def test_resolve_partial_output_dir_falls_back_to_cwd_with_warning(monkeypatch, caplog):
    """When output_path is None and Hydra hasn't initialised, fall back to
    CWD with a clear warning."""
    from ntt.evaluation.utils.paths import resolve_partial_output_dir

    # Force the Hydra path to fail so we hit the CWD fallback. HydraConfig.get()
    # raises ValueError outside an active Hydra context, which is the natural
    # state for this unit test (pytest is not running under @hydra.main).
    with caplog.at_level("WARNING", logger="ntt.evaluation.utils.paths"):
        result = resolve_partial_output_dir(None)
    assert result == Path.cwd()
    fallback_warnings = [r for r in caplog.records if "falling back to CWD" in r.message]
    assert len(fallback_warnings) >= 1
