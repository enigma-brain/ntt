"""Core tests for the post-hoc Poisson-normalization analysis helpers.

Each test builds a tiny synthetic results directory on disk that matches
the on-disk layout the live eval jobs write, then runs the public API and
checks the returned dict's shape + a key numeric value. Read-only — none
of these tests cause any writes outside ``tmp_path``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from ntt.evaluation.aggregation.poisson_post_hoc import (
    compare_snr_against_poisson,
    normalize_m1_results_against_poisson,
    normalize_m2_results_against_poisson,
)
from ntt.evaluation.core.types import (
    EvaluationReport,
    MetricCategory,
    MetricDirection,
    MetricLevel,
    MetricResult,
)

# ────────────────── fixture helpers ──────────────────


def _mk_metric(
    name: str,
    raw_value: float,
    direction: MetricDirection = MetricDirection.LOWER_IS_BETTER,
) -> MetricResult:
    return MetricResult(
        name=name,
        category=MetricCategory.LATENT_QUALITY,
        direction=direction,
        level=MetricLevel.POPULATION,
        raw_value=float(raw_value),
        normalized_value=None,
        metadata={},
        value_range=(None, None),
    )


def _write_per_session(
    results_dir: Path, sid_to_metrics: dict[str, dict[str, MetricResult]]
) -> None:
    """Mirror what the M1/M2 jobs write: ``per_session/{safe_sid}.json``."""
    per_session_dir = results_dir / "per_session"
    per_session_dir.mkdir(parents=True, exist_ok=True)
    for sid, metrics in sid_to_metrics.items():
        report = EvaluationReport(results=metrics)
        safe_sid = sid.replace("/", "__")
        (per_session_dir / f"{safe_sid}.json").write_text(json.dumps(report.to_dict()))


# ────────────────── M1 ──────────────────


def test_m1_normalization_three_sessions(tmp_path: Path) -> None:
    """Model NMSE 1 < Poisson NMSE 4; ceiling (max_shift_ms=0) is 0.
    Lower-is-better → normalized = (poisson - model)/(poisson - ceiling) = 3/4."""
    model_dir = tmp_path / "model"
    poisson_dir = tmp_path / "poisson"
    metric = "cs_corr/window_shift|max_shift_ms=10.0"
    zero_metric = "cs_corr/window_shift|max_shift_ms=0.0"
    _write_per_session(
        model_dir,
        {
            f"sess_{i}": {
                metric: _mk_metric(metric, 1.0),
                zero_metric: _mk_metric(zero_metric, 0.0),
            }
            for i in range(3)
        },
    )
    _write_per_session(
        poisson_dir,
        {f"sess_{i}": {metric: _mk_metric(metric, 4.0)} for i in range(3)},
    )

    out = normalize_m1_results_against_poisson(
        model_results_dir=model_dir, poisson_results_dir=poisson_dir
    )
    assert out["mode"] == "M1"
    assert out["aggregation_order"] == "aggregate_first"
    assert out["normalized_scores_vs_poisson"][metric] == pytest.approx(0.75)
    # Per-session-then-mean must give the same number when every session has
    # the same values.
    assert out["normalized_scores_from_per_session_mean"][metric] == pytest.approx(0.75)
    # When aggregation_order="aggregate_first", the summary points at the
    # aggregate-first flavor.
    assert out["normalized_summary"][metric] == pytest.approx(0.75)


def test_m1_per_session_then_mean_flavor_selected(tmp_path: Path) -> None:
    """The two flavors only diverge when sessions are asymmetric, but for the
    summary-pointer wiring we just need to confirm the right field is
    surfaced under each ``aggregation_order``."""
    model_dir = tmp_path / "model"
    poisson_dir = tmp_path / "poisson"
    metric = "cs_corr/window_shift|max_shift_ms=10.0"
    zero_metric = "cs_corr/window_shift|max_shift_ms=0.0"
    _write_per_session(
        model_dir,
        {
            "sess_0": {
                metric: _mk_metric(metric, 1.0),
                zero_metric: _mk_metric(zero_metric, 0.0),
            }
        },
    )
    _write_per_session(poisson_dir, {"sess_0": {metric: _mk_metric(metric, 4.0)}})

    out = normalize_m1_results_against_poisson(
        model_results_dir=model_dir,
        poisson_results_dir=poisson_dir,
        aggregation_order="per_session_then_mean",
    )
    # Both flavors are 0.75 here (single session, no asymmetry), so the
    # difference is only which dict normalized_summary points at.
    assert out["normalized_summary"][metric] == pytest.approx(
        out["normalized_scores_from_per_session_mean"][metric]
    )


def test_m1_partial_overlap_aggregate_uses_shared_subset(tmp_path: Path) -> None:
    """Smoke-test finding: when model and Poisson have only some sessions in
    common, the aggregate-first flavor must aggregate over the shared subset
    on both sides — not over each side's full set — otherwise the two
    flavors disagree even though they SHOULD describe the same comparison.

    Model has session A only; Poisson has A and B with very different values
    on B. Aggregate-first must use Poisson's A value (not the mean of A+B).
    """
    model_dir = tmp_path / "model"
    poisson_dir = tmp_path / "poisson"
    metric = "cs_corr/window_shift|max_shift_ms=10.0"
    zero_metric = "cs_corr/window_shift|max_shift_ms=0.0"
    _write_per_session(
        model_dir,
        {
            "sess_A": {
                metric: _mk_metric(metric, 1.0),
                zero_metric: _mk_metric(zero_metric, 0.0),
            }
        },
    )
    _write_per_session(
        poisson_dir,
        {
            "sess_A": {metric: _mk_metric(metric, 4.0)},
            # B has a very different baseline; if aggregate-first averages it
            # in we'd see a wildly different normalized score.
            "sess_B": {metric: _mk_metric(metric, 100.0)},
        },
    )

    out = normalize_m1_results_against_poisson(
        model_results_dir=model_dir, poisson_results_dir=poisson_dir
    )
    # Lower-is-better: (poisson_A - model_A)/(poisson_A - ceiling_A) = 3/4
    assert out["normalized_scores_vs_poisson"][metric] == pytest.approx(0.75)
    assert out["normalized_scores_from_per_session_mean"][metric] == pytest.approx(0.75)
    # The two flavors must agree to floating-point precision when the shared
    # subset has one session — different aggregate-first behavior would
    # break this invariant.
    assert (
        out["normalized_scores_vs_poisson"][metric]
        == out["normalized_scores_from_per_session_mean"][metric]
    )


def test_m1_warns_on_disjoint_sessions(tmp_path: Path, caplog) -> None:
    caplog.set_level(logging.WARNING)
    model_dir = tmp_path / "model"
    poisson_dir = tmp_path / "poisson"
    metric = "cs_corr/window_shift|max_shift_ms=10.0"
    zero_metric = "cs_corr/window_shift|max_shift_ms=0.0"
    _write_per_session(
        model_dir,
        {
            "sess_a": {
                metric: _mk_metric(metric, 1.0),
                zero_metric: _mk_metric(zero_metric, 0.0),
            }
        },
    )
    _write_per_session(poisson_dir, {"sess_b": {metric: _mk_metric(metric, 4.0)}})

    out = normalize_m1_results_against_poisson(
        model_results_dir=model_dir, poisson_results_dir=poisson_dir
    )
    assert out["per_session"] == {}
    assert any("no overlap" in r.getMessage() for r in caplog.records)


# ────────────────── M2 ──────────────────


def test_m2_normalization_epsilon_key(tmp_path: Path) -> None:
    """M2 ceiling lookup uses ``<metric>/eps_0``."""
    model_dir = tmp_path / "model"
    poisson_dir = tmp_path / "poisson"
    metric = "acg_correlation/eps_0.1"
    zero_metric = "acg_correlation/eps_0"
    # Higher-is-better correlation: model 0.9, poisson 0.3, ceiling 1.0
    # → normalized = (model - poisson)/(ceiling - poisson) = 0.6/0.7
    _write_per_session(
        model_dir,
        {
            "sess_0": {
                metric: _mk_metric(metric, 0.9, MetricDirection.HIGHER_IS_BETTER),
                zero_metric: _mk_metric(zero_metric, 1.0, MetricDirection.HIGHER_IS_BETTER),
            }
        },
    )
    _write_per_session(
        poisson_dir,
        {"sess_0": {metric: _mk_metric(metric, 0.3, MetricDirection.HIGHER_IS_BETTER)}},
    )

    out = normalize_m2_results_against_poisson(
        model_results_dir=model_dir, poisson_results_dir=poisson_dir
    )
    assert out["mode"] == "M2"
    assert out["normalized_scores_vs_poisson"][metric] == pytest.approx(0.6 / 0.7)


# ────────────────── SNR ──────────────────


def _write_snr_run(results_dir: Path, snr_by_session: dict[str, float]) -> None:
    """Synthesize a main-eval-style SNR results dir with a minimal results.csv.

    Real SNR runs in eval_outputs/ all have an empty ``results.json/results``
    block, so the SNR helper is built not to need it. We write only the CSV
    here to mirror that on-disk reality.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    lines = ["session_id,segment_index,repeats_snr"]
    for sid, val in snr_by_session.items():
        lines.append(f"{sid},0,{val}")
    (results_dir / "results.csv").write_text("\n".join(lines) + "\n")


def test_snr_delta_no_ceiling(tmp_path: Path) -> None:
    """SNR delta is direction-corrected ``model − poisson``; positive means
    model is better."""
    model_dir = tmp_path / "model"
    poisson_dir = tmp_path / "poisson"
    _write_snr_run(model_dir, {"sess_0": 5.0, "sess_1": 7.0})
    _write_snr_run(poisson_dir, {"sess_0": 2.0, "sess_1": 3.0})

    out = compare_snr_against_poisson(model_results_dir=model_dir, poisson_results_dir=poisson_dir)
    # Aggregate: mean(5,7) - mean(2,3) = 6 - 2.5 = 3.5
    assert out["aggregate_delta"]["repeats_snr"] == pytest.approx(3.5)
    assert out["per_session_delta"]["sess_0"]["repeats_snr"] == pytest.approx(3.0)
    assert out["per_session_delta"]["sess_1"]["repeats_snr"] == pytest.approx(4.0)
    # per_session_then_mean = mean(3, 4) = 3.5 in this symmetric case
    assert out["per_session_mean_delta"]["repeats_snr"] == pytest.approx(3.5)
    # Default points at aggregate_first.
    assert out["delta_summary"]["repeats_snr"] == pytest.approx(3.5)


def test_snr_per_session_then_mean(tmp_path: Path) -> None:
    """When asymmetric, ``aggregate_first`` and ``per_session_then_mean`` give
    the same answer for SNR because delta is linear; the test confirms the
    summary pointer flips correctly."""
    model_dir = tmp_path / "model"
    poisson_dir = tmp_path / "poisson"
    _write_snr_run(model_dir, {"sess_0": 5.0})
    _write_snr_run(poisson_dir, {"sess_0": 2.0})

    out = compare_snr_against_poisson(
        model_results_dir=model_dir,
        poisson_results_dir=poisson_dir,
        aggregation_order="per_session_then_mean",
    )
    assert out["delta_summary"]["repeats_snr"] == pytest.approx(
        out["per_session_mean_delta"]["repeats_snr"]
    )


def test_snr_higher_is_better_false_flips_sign(tmp_path: Path) -> None:
    """For a hypothetical lower-is-better main-eval metric, the helper
    sign-flips so positive still means 'model is better than poisson'."""
    model_dir = tmp_path / "model"
    poisson_dir = tmp_path / "poisson"
    _write_snr_run(model_dir, {"sess_0": 1.0})  # lower model value
    _write_snr_run(poisson_dir, {"sess_0": 3.0})  # higher poisson value

    out = compare_snr_against_poisson(
        model_results_dir=model_dir,
        poisson_results_dir=poisson_dir,
        higher_is_better=False,
    )
    # poisson - model = 3 - 1 = +2 (model "wins" under lower-is-better)
    assert out["aggregate_delta"]["repeats_snr"] == pytest.approx(2.0)


def test_snr_aggregate_uses_paired_subset_per_metric(tmp_path: Path) -> None:
    """`aggregate_delta` averages model and poisson over the *same* session
    population per metric, also when a metric is finite on
    both sides for some shared sessions but missing on one side for others:
    the two means must not be taken over different session subsets.

    Setup: both sessions share ``repeats_snr`` (so both are "shared"), but a
    second column ``aux`` is present on both sides only for ``sess_0`` — the
    poisson side has an empty ``aux`` cell for ``sess_1``, so that session
    drops ``aux`` on the poisson side only.
    """
    model_dir = tmp_path / "model"
    poisson_dir = tmp_path / "poisson"
    model_dir.mkdir()
    poisson_dir.mkdir()
    (model_dir / "results.csv").write_text(
        "session_id,segment_index,repeats_snr,aux\nsess_0,0,5.0,10.0\nsess_1,0,7.0,20.0\n"
    )
    (poisson_dir / "results.csv").write_text(
        "session_id,segment_index,repeats_snr,aux\n"
        "sess_0,0,2.0,4.0\n"
        "sess_1,0,3.0,\n"  # poisson is missing aux for sess_1
    )

    out = compare_snr_against_poisson(
        model_results_dir=model_dir,
        poisson_results_dir=poisson_dir,
        metric_columns=["repeats_snr", "aux"],
    )
    # repeats_snr present on both sides for both sessions → unaffected:
    # mean(5,7) - mean(2,3) = 6 - 2.5 = 3.5
    assert out["aggregate_delta"]["repeats_snr"] == pytest.approx(3.5)
    # aux paired only on sess_0 → 10 - 4 = 6.0.
    # Independent per-side filtering would average model over {sess_0,
    # sess_1} (mean 15) but poisson over {sess_0} (4), giving 11.0.
    assert out["aggregate_delta"]["aux"] == pytest.approx(6.0)


def test_snr_warns_on_disjoint_sessions(tmp_path: Path, caplog) -> None:
    caplog.set_level(logging.WARNING)
    model_dir = tmp_path / "model"
    poisson_dir = tmp_path / "poisson"
    _write_snr_run(model_dir, {"sess_a": 5.0})
    _write_snr_run(poisson_dir, {"sess_b": 2.0})

    out = compare_snr_against_poisson(model_results_dir=model_dir, poisson_results_dir=poisson_dir)
    assert out["per_session_delta"] == {}
    assert any("no overlap" in r.getMessage() for r in caplog.records)
