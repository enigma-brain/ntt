"""Tests for EvalArtifact format + matching policy."""

from __future__ import annotations

import json
import warnings

import pytest

from ntt.evaluation.artifact import EvalArtifact, EvalArtifactMetadata, make_metadata
from ntt.evaluation.neuron_selection import (
    NeuronSelection,
    NeuronSelectionMismatchError,
    SessionUnits,
)


def _selection(**session_overrides) -> NeuronSelection:
    """Units scored per session A/B/C (``session_overrides`` replaces sessions)."""
    sessions = {
        sid: SessionUnits(n_units=10, n_selected=10, n_padded=0, unit_ids_sha256=sid * 64)
        for sid in "ABC"
    }
    sessions.update(session_overrides)
    return NeuronSelection(
        policy="eval_unit_selection",
        max_neurons=200,
        seed=42,
        padded_to=None,
        sessions=sessions,
    )


def _baseline_metadata(**overrides):
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
        "neuron_selection": _selection(),
    }
    base.update(overrides)
    return make_metadata(**base)


def _baseline_artifact(**meta_overrides):
    return EvalArtifact(
        metadata=_baseline_metadata(**meta_overrides),
        per_session={
            "A": {"metric_x": 0.5},
            "B": {"metric_x": 0.4},
            "C": {"metric_x": 0.6},
        },
        aggregated={"metric_x": 0.5},
        perfect_per_session={
            "A": {"metric_x": 1.0},
            "B": {"metric_x": 1.0},
            "C": {"metric_x": 1.0},
        },
        perfect_aggregated={"metric_x": 1.0},
    )


def test_round_trip_json(tmp_path):
    art = _baseline_artifact()
    path = tmp_path / "x.json"
    art.dump_json(path)
    loaded = EvalArtifact.load_json(path)
    assert loaded.metadata == art.metadata
    assert loaded.per_session == art.per_session
    assert loaded.aggregated == art.aggregated
    assert loaded.perfect_per_session == art.perfect_per_session
    assert loaded.perfect_aggregated == art.perfect_aggregated


def test_load_corrupt_json_raises_value_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not valid json")
    with pytest.raises(ValueError, match="Corrupt JSON"):
        EvalArtifact.load_json(path)


def test_load_schema_incompatible_raises_value_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"metadata": {}, "per_session": {}, "aggregated": {}}')
    with pytest.raises(ValueError, match="Failed to parse"):
        EvalArtifact.load_json(path)


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("bin_size_ms", 2.0),
        ("rate_estimation_bin_ms", 20.0),
        ("split", "test"),
        ("max_eval_seconds", 300.0),
        ("decoding_bypass_fields", ["cursor_pos"]),
    ],
)
def test_validate_against_hard_field_mismatch_raises(field, bad_value):
    art = _baseline_artifact()
    consumer = _baseline_metadata(**{field: bad_value})
    with pytest.raises(ValueError, match=field):
        art.validate_against(consumer)


def test_validate_against_window_length_seconds_mismatch_warns():
    """window_length_seconds is intentionally soft: default metric suites
    consume segment-level concatenated tensors, so a window mismatch only
    matters for metrics that read per-window latents directly. Should
    warn, not raise."""
    art = _baseline_artifact()  # window_length_seconds=0.1
    consumer = _baseline_metadata(window_length_seconds=2.0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        art.validate_against(consumer)  # no exception
    msgs = [str(x.message) for x in w]
    assert any("window_length_seconds" in m for m in msgs), msgs


def test_validate_against_session_subset_ok():
    art = _baseline_artifact()  # A, B, C
    consumer = _baseline_metadata(sessions=["A", "B"])
    art.validate_against(consumer)  # no exception


def test_validate_against_session_exact_ok():
    art = _baseline_artifact()  # A, B, C
    consumer = _baseline_metadata(sessions=["A", "B", "C"])
    art.validate_against(consumer)


def test_validate_against_session_superset_raises():
    art = _baseline_artifact()  # A, B, C
    consumer = _baseline_metadata(sessions=["A", "B", "D"])
    with pytest.raises(ValueError, match="not in baseline"):
        art.validate_against(consumer)


def test_validate_against_n_realisations_mismatch_warns():
    art = _baseline_artifact()  # n_realisations=5
    consumer = _baseline_metadata(n_realisations=10)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        art.validate_against(consumer)
    msgs = [str(x.message) for x in w]
    assert any("n_realisations" in m for m in msgs), msgs


def test_validate_against_seed_mismatch_silent():
    art = _baseline_artifact()  # seed=42
    consumer = _baseline_metadata(seed=99)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        art.validate_against(consumer)
    seed_warnings = [
        x for x in w if "seed" in str(x.message) and issubclass(x.category, UserWarning)
    ]
    assert seed_warnings == [], seed_warnings


def test_metadata_kind_literal_validation():
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        EvalArtifactMetadata.model_validate(
            {
                "kind": "wrong_kind",
                "bin_size_ms": 1.0,
                "window_length_seconds": 0.1,
                "split": "val",
                "max_eval_seconds": None,
                "decoding_bypass_fields": [],
                "sessions": [],
                "timestamp": "2026-04-29T00:00:00+00:00",
            }
        )


# ---------------------------------------------------------------------------
# Excluded-brainsets handling: load-time warning + .without_brainsets()
# ---------------------------------------------------------------------------


# A real session ID from the excluded tostado_marcos_birdsong_2024 registry.
_TOSTADO_SID = "finch_z_r12r13_21_20210626_held_in_calib"


def _artifact_with_tostado_session():
    """Build an EvalArtifact whose per_session contains one tostado session."""
    return EvalArtifact(
        metadata=_baseline_metadata(sessions=["A", "B", _TOSTADO_SID]),
        per_session={
            "A": {"metric_x": 0.5},
            "B": {"metric_x": 0.4},
            _TOSTADO_SID: {"metric_x": 0.9},
        },
        aggregated={"metric_x": 0.6},
        per_session_weights={"A": 100.0, "B": 100.0, _TOSTADO_SID: 50.0},
        perfect_per_session={
            "A": {"metric_x": 1.0},
            "B": {"metric_x": 1.0},
            _TOSTADO_SID: {"metric_x": 1.0},
        },
        perfect_aggregated={"metric_x": 1.0},
    )


def test_load_warns_when_excluded_brainset_session_present(tmp_path):
    art = _artifact_with_tostado_session()
    path = tmp_path / "with_tostado.json"
    art.dump_json(path)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        EvalArtifact.load_json(path)
    msgs = [str(x.message) for x in w if issubclass(x.category, UserWarning)]
    assert any("tostado_marcos_birdsong_2024" in m for m in msgs), msgs
    assert any("without_brainsets" in m for m in msgs), msgs


def test_load_clean_artifact_emits_no_excluded_warning(tmp_path):
    art = _baseline_artifact()  # sessions A, B, C — none excluded
    path = tmp_path / "clean.json"
    art.dump_json(path)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        EvalArtifact.load_json(path)
    excluded_warnings = [
        x
        for x in w
        if issubclass(x.category, UserWarning) and "excluded brainset" in str(x.message).lower()
    ]
    assert excluded_warnings == [], excluded_warnings


def test_without_brainsets_strips_excluded_sessions():
    art = _artifact_with_tostado_session()
    cleaned = art.without_brainsets(["tostado_marcos_birdsong_2024"])

    # per_session and per_session_weights drop the tostado entry
    assert set(cleaned.per_session.keys()) == {"A", "B"}
    assert cleaned.per_session_weights is not None
    assert set(cleaned.per_session_weights.keys()) == {"A", "B"}
    assert cleaned.perfect_per_session is not None
    assert set(cleaned.perfect_per_session.keys()) == {"A", "B"}

    # metadata.sessions drops the tostado entry, preserving order of survivors
    assert cleaned.metadata.sessions == ["A", "B"]

    # .aggregated is intentionally unchanged — see docstring
    assert cleaned.aggregated == art.aggregated
    assert cleaned.perfect_aggregated == art.perfect_aggregated

    # Original artifact is not mutated
    assert _TOSTADO_SID in art.per_session


def test_without_brainsets_unknown_name_is_noop():
    art = _baseline_artifact()
    same = art.without_brainsets(["not_a_real_brainset"])
    # No-op should return the original instance (cheap fast path)
    assert same is art


def test_without_brainsets_handles_missing_optional_fields():
    """When per_session_weights / perfect_per_session are None, filtering
    must not crash and must keep them None."""
    minimal = EvalArtifact(
        metadata=_baseline_metadata(sessions=["A", _TOSTADO_SID]),
        per_session={
            "A": {"metric_x": 0.5},
            _TOSTADO_SID: {"metric_x": 0.9},
        },
        aggregated={"metric_x": 0.7},
        # per_session_weights, perfect_per_session, perfect_aggregated all None
    )
    cleaned = minimal.without_brainsets(["tostado_marcos_birdsong_2024"])
    assert set(cleaned.per_session.keys()) == {"A"}
    assert cleaned.metadata.sessions == ["A"]
    assert cleaned.per_session_weights is None
    assert cleaned.perfect_per_session is None
    assert cleaned.perfect_aggregated is None


def test_aggregation_strategy_round_trips_and_defaults_to_none(tmp_path):
    # Saved so a consumer can check it against its own; absent when not recorded.
    path = tmp_path / "a.json"
    _baseline_artifact(aggregation_strategy="median").dump_json(path)
    assert EvalArtifact.load_json(path).metadata.aggregation_strategy == "median"
    old = json.loads(path.read_text())
    del old["metadata"]["aggregation_strategy"]
    path.write_text(json.dumps(old))
    assert EvalArtifact.load_json(path).metadata.aggregation_strategy is None


def test_validate_against_neuron_selection_subset_ok():
    art = _baseline_artifact()
    consumer = _baseline_metadata(
        sessions=["A"],
        neuron_selection=_selection().model_copy(
            update={"sessions": {"A": _selection().sessions["A"]}}
        ),
    )
    art.validate_against(consumer)


def test_validate_against_different_units_raises():
    """Same session, different units (here: zero-padded on the consumer side)."""
    art = _baseline_artifact()
    padded = SessionUnits(n_units=10, n_selected=10, n_padded=22, unit_ids_sha256="A" * 64)
    consumer = _baseline_metadata(neuron_selection=_selection(A=padded))
    with pytest.raises(NeuronSelectionMismatchError, match="A: 10/10 units \\+ 22 zero-padded"):
        art.validate_against(consumer)


def test_validate_against_missing_neuron_selection_raises_or_warns(caplog):
    """An artifact written before the record existed cannot be shown to match."""
    art = _baseline_artifact(neuron_selection=None)
    with pytest.raises(NeuronSelectionMismatchError, match="no neuron_selection record"):
        art.validate_against(_baseline_metadata())
    with caplog.at_level("WARNING"):
        art.validate_against(_baseline_metadata(), strict_neuron_selection=False)
    assert "no neuron_selection record" in caplog.text


def test_neuron_selection_round_trips(tmp_path):
    path = tmp_path / "a.json"
    _baseline_artifact().dump_json(path)
    assert EvalArtifact.load_json(path).metadata.neuron_selection == _selection()
