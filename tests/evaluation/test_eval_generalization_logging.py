"""Tests for the generalization + no-waste-save wiring.

The generalization retention score must not depend on ``output_path`` being set
(most eval experiments leave it unset), otherwise the score and ``results.json``
would silently never be produced. These tests pin the decoupled behavior:

- ``_compute_generalization`` runs independent of ``output_path``;
- ``_get_output_dir`` falls back so results land in the Hydra run dir;
- ``_flatten_generalization_metrics`` turns the score block into MLflow metrics.

The two ``NTTEvalJob`` methods read only ``self.output_path`` / module globals,
so they're exercised here with a lightweight stand-in ``self`` rather than a
fully-constructed (pydantic) job.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ntt.evaluation.mlflow_logging import _flatten_generalization_metrics
from ntt.evaluation.tiering import TierMembership
from ntt.jobs.eval import NTTEvalJob

# ── _flatten_generalization_metrics ──────────────────────────────────


def test_flatten_generalization_emits_summary_numeric_leaves():
    block = {
        "baseline_tier": "xtime",
        "ood_tiers": ["xsession"],
        "summary": {
            "mean_retention": 0.8,
            "per_tier": {"xsession": 0.7, "xsubject": float("nan")},
            "note": "text",  # non-numeric → skipped
            "converged": True,  # bool → skipped
        },
        # The verbose per-metric list is intentionally not flattened.
        "per_metric": [{"metric": "x", "retention": 0.5}],
    }

    metrics = _flatten_generalization_metrics(block)

    assert metrics == {
        "eval/generalization/mean_retention": 0.8,
        "eval/generalization/per_tier/xsession": 0.7,
    }


def test_flatten_generalization_handles_missing_summary():
    assert _flatten_generalization_metrics({"per_metric": []}) == {}
    assert _flatten_generalization_metrics({"summary": None}) == {}


# ── NTTEvalJob._compute_generalization ───────────────────────────────


def _entry(value: float) -> dict:
    return {"results": {"acc": {"raw_value": value, "direction": "higher", "category": "c"}}}


# Two brainsets with different scales: matched retention is mean(0.4/0.8, 0.1/0.1) = 0.75,
# the pooled tier mean would give 0.25/0.45.
PER_SESSION = {
    "bsA/t": _entry(0.8),
    "bsB/t": _entry(0.1),
    "bsA/s": _entry(0.4),
    "bsB/s": _entry(0.1),
}
TIER_MAP = {
    "xtime": TierMembership(
        "xtime", exclude_brainsets={"bsA": frozenset({"s"}), "bsB": frozenset({"s"})}
    ),
    "xsource": TierMembership("xsource", include=frozenset({"bsA/s", "bsB/s"})),
}


def test_compute_generalization_none_without_tiers():
    assert NTTEvalJob._compute_generalization(SimpleNamespace(), PER_SESSION, None) is None
    assert NTTEvalJob._compute_generalization(SimpleNamespace(), PER_SESSION, {}) is None


def test_compute_generalization_none_when_xtime_missing():
    tier_map = {"xsource": TIER_MAP["xsource"]}
    assert NTTEvalJob._compute_generalization(SimpleNamespace(), PER_SESSION, tier_map) is None


def test_compute_generalization_scores_the_tiered_sessions():
    block = NTTEvalJob._compute_generalization(SimpleNamespace(), PER_SESSION, TIER_MAP)

    assert block is not None
    assert block["summary"]["baseline_mode"] == "brainset_matched"
    assert block["summary"]["generalization_score"] == pytest.approx(0.75)


def test_compute_generalization_none_without_a_scored_tier():
    # Only xsource enters the score; an eval without xsource sessions has none.
    tier_map = {
        "xtime": TIER_MAP["xtime"],
        "xsession": TierMembership("xsession", include=frozenset({"bsA/s", "bsB/s"})),
    }
    assert NTTEvalJob._compute_generalization(SimpleNamespace(), PER_SESSION, tier_map) is None


def test_compute_generalization_swallows_errors(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("nope")

    monkeypatch.setattr("ntt.jobs.eval.compute_from_session_tiers", _boom)

    assert NTTEvalJob._compute_generalization(SimpleNamespace(), PER_SESSION, TIER_MAP) is None


# ── NTTEvalJob._get_output_dir ───────────────────────────────────────


def test_get_output_dir_uses_explicit_output_path(tmp_path):
    target = tmp_path / "eval_out"
    out = NTTEvalJob._get_output_dir(SimpleNamespace(output_path=str(target)))
    assert out == target
    assert out.is_dir()  # created


def test_get_output_dir_none_without_output_path_or_hydra(monkeypatch):
    # No output_path and (in the test process) no active Hydra run → None,
    # so callers skip local saving gracefully instead of raising.
    import hydra.core.hydra_config as hc

    def _raise():
        raise ValueError("no active Hydra run")

    monkeypatch.setattr(hc.HydraConfig, "get", staticmethod(_raise))
    assert NTTEvalJob._get_output_dir(SimpleNamespace(output_path=None)) is None
