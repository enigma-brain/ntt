"""Tests for the upfront poisson_reference hard-field validation.

The eval job validates a configured ``poisson_reference`` *before* running
the (slow) tokenizer eval. The session-set check is deferred to the late
path because consumer sessions are only known after the eval runs. These
tests cover the early-path function in isolation.
"""

from __future__ import annotations

import warnings

import pytest

from ntt.evaluation.artifact import EvalArtifact, make_metadata
from ntt.evaluation.neuron_selection import (
    NeuronSelection,
    NeuronSelectionMismatchError,
    SessionUnits,
)
from ntt.jobs.eval import _validate_poisson_reference_hard_fields

_SELECTION = NeuronSelection(
    policy="eval_unit_selection",
    max_neurons=200,
    seed=42,
    padded_to=None,
    sessions={
        sid: SessionUnits(n_units=10, n_selected=10, n_padded=0, unit_ids_sha256=sid * 64)
        for sid in "ABC"
    },
)


def _baseline_artifact(**meta_overrides) -> EvalArtifact:
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
        "neuron_selection": _SELECTION,
    }
    base.update(meta_overrides)
    return EvalArtifact(
        metadata=make_metadata(**base),
        per_session={"A": {"x": 0.1}, "B": {"x": 0.2}, "C": {"x": 0.3}},
        aggregated={"x": 0.2},
        perfect_per_session={"A": {"x": 1.0}, "B": {"x": 1.0}, "C": {"x": 1.0}},
        perfect_aggregated={"x": 1.0},
    )


def _consumer_kwargs(**overrides) -> dict:
    base = {
        "bin_size_ms": 1.0,
        "window_length_seconds": 0.1,
        "split": "val",
        "max_eval_seconds": 600.0,
        "decoding_bypass_fields": ["cursor_pos", "cursor_vel"],
        "n_realisations": 5,
        "seed": 42,
        "poisson_rate_bin_ms": None,
        "neuron_selection": _SELECTION,
    }
    base.update(overrides)
    return base


def test_accepts_matching_baseline(tmp_path):
    art = _baseline_artifact()
    path = tmp_path / "ref.json"
    art.dump_json(path)

    out = _validate_poisson_reference_hard_fields(path, **_consumer_kwargs())
    assert out.metadata.bin_size_ms == 1.0
    assert sorted(out.metadata.sessions) == ["A", "B", "C"]


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("bin_size_ms", 2.0),
        ("split", "test"),
        ("max_eval_seconds", 300.0),
        ("decoding_bypass_fields", ["cursor_pos"]),
    ],
)
def test_rejects_hard_field_mismatch(tmp_path, field, bad_value):
    art = _baseline_artifact()
    path = tmp_path / "ref.json"
    art.dump_json(path)

    with pytest.raises(ValueError, match=field):
        _validate_poisson_reference_hard_fields(path, **_consumer_kwargs(**{field: bad_value}))


def test_window_length_mismatch_warns_does_not_raise(tmp_path):
    """window_length_seconds is soft: a shared reference keeps its own window
    (0.1 s) whatever window the model is scored on. A window mismatch should
    warn but allow the eval to proceed."""
    art = _baseline_artifact()  # window_length_seconds=0.1
    path = tmp_path / "ref.json"
    art.dump_json(path)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = _validate_poisson_reference_hard_fields(
            path, **_consumer_kwargs(window_length_seconds=2.0)
        )
    msgs = [str(x.message) for x in w]
    assert any("window_length_seconds" in m for m in msgs), msgs
    assert out.metadata.window_length_seconds == 0.1


def test_skips_session_set_check(tmp_path):
    """The early path uses the baseline's own session list as the consumer's
    sessions, so the session-set ⊆ check is always satisfied here. The real
    consumer-vs-baseline session check stays in _run_poisson_comparison.
    """
    art = _baseline_artifact(sessions=["A", "B", "C"])
    path = tmp_path / "ref.json"
    art.dump_json(path)

    out = _validate_poisson_reference_hard_fields(path, **_consumer_kwargs())
    assert sorted(out.metadata.sessions) == ["A", "B", "C"]


def test_adopts_baseline_rate(tmp_path):
    """Consumer-side ``poisson_rate_bin_ms`` is informational only — the
    consumer metadata always uses the baseline's rate (matching the late-
    path adoption). A user-set rate that differs from the baseline produces
    a warning log line but does not fail validation."""
    art = _baseline_artifact(rate_estimation_bin_ms=10.0)
    path = tmp_path / "ref.json"
    art.dump_json(path)

    out = _validate_poisson_reference_hard_fields(
        path, **_consumer_kwargs(poisson_rate_bin_ms=5.0)
    )
    assert out.metadata.rate_estimation_bin_ms == 10.0


def test_corrupt_artifact_raises_value_error(tmp_path):
    path = tmp_path / "ref.json"
    path.write_text("{not valid json")

    with pytest.raises(ValueError, match="Corrupt JSON"):
        _validate_poisson_reference_hard_fields(path, **_consumer_kwargs())


def test_rejects_a_reference_scored_on_other_units_before_the_eval_runs(tmp_path):
    """Fail-fast: the units check runs before the slow eval, and only warns in padded mode."""
    art = _baseline_artifact(neuron_selection=None)
    path = tmp_path / "ref.json"
    art.dump_json(path)

    with pytest.raises(NeuronSelectionMismatchError, match="no neuron_selection record"):
        _validate_poisson_reference_hard_fields(path, **_consumer_kwargs())
    _validate_poisson_reference_hard_fields(
        path, **_consumer_kwargs(strict_neuron_selection=False)
    )
