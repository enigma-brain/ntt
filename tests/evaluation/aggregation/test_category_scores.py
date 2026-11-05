"""Tests for evaluation/aggregation/category_scores.py."""

from __future__ import annotations

import logging
import statistics

import pytest

from ntt.evaluation.aggregation.category_scores import (
    CategoryScores,
    ColumnDiagnostics,
    MetricScore,
)
from ntt.evaluation.aggregation.downstream_records import DownstreamScoreView


def test_metric_score_fields():
    ms = MetricScore(normalized=0.8, clipped=False, raw_value=0.6, direction="higher")
    assert ms.normalized == 0.8
    assert ms.clipped is False
    assert ms.raw_value == 0.6
    assert ms.direction == "higher"


def test_column_diagnostics_fields():
    ms = MetricScore(normalized=0.5, clipped=True, raw_value=-2.0, direction="lower")
    diag = ColumnDiagnostics(
        score=0.5,
        n_metrics=1,
        normalization="precomputed",
        per_metric={"some_metric": ms},
    )
    assert diag.score == 0.5
    assert diag.n_metrics == 1
    assert diag.normalization == "precomputed"
    assert "some_metric" in diag.per_metric


def test_category_scores_all_none():
    scores = CategoryScores(
        single_neuron_recon=None,
        population_structure=None,
        latent_structure_and_robustness=None,
        decoding=None,
        downstream_encoding=None,
        generalization=None,
        compression=None,
        diagnostics={},
    )
    assert scores.single_neuron_recon is None
    assert scores.compression is None


from ntt.evaluation.aggregation.category_scores import (
    _SN_RECON_CATEGORIES,
    _score_comparison_bucket,
)
from ntt.evaluation.aggregation.comparison import MetricComparison
from ntt.evaluation.core.types import MetricCategory, MetricDirection, MetricLevel


def _score_sn_recon(comparisons, score_clip, normalization):
    """Test shim: score the single-neuron-recon bucket via the unified helper."""
    return _score_comparison_bucket(comparisons, _SN_RECON_CATEGORIES, score_clip, normalization)


def _make_comparison(
    normalized_score: float | None,
    category: str = "temporal_fidelity",
    direction: str = "higher",
    model_value: float = 0.5,
    status: str = "ok",
) -> MetricComparison:
    """Build a MetricComparison for tests without needing a real report."""
    return MetricComparison(
        name="test_metric",
        category=MetricCategory(category),
        direction=MetricDirection(direction),
        level=MetricLevel("population"),
        model_value=model_value,
        baseline_value=0.0,
        ceiling_value=1.0,
        normalized_score=normalized_score,
        status=status,
    )


def test_sn_recon_averages_valid_temporal_fidelity():
    comparisons = {
        "spike_f1": _make_comparison(0.8, category="temporal_fidelity"),
        "rate_error": _make_comparison(-0.4, category="single_unit_statistics", direction="lower"),
    }
    result = _score_sn_recon(comparisons, score_clip=None, normalization="precomputed")
    assert result.score == pytest.approx((0.8 + -0.4) / 2)
    assert result.n_metrics == 2
    assert result.normalization == "precomputed;family_weighted"


def test_sn_recon_excludes_invalid_status():
    comparisons = {
        "spike_f1": _make_comparison(0.8, category="temporal_fidelity"),
        "acg_corr": _make_comparison(None, category="temporal_fidelity", status="invalid"),
    }
    result = _score_sn_recon(comparisons, score_clip=None, normalization="precomputed")
    assert result.n_metrics == 1
    assert "acg_corr" not in result.per_metric


def test_sn_recon_clips_below_threshold():
    comparisons = {
        "bad": _make_comparison(-5.0, category="temporal_fidelity", model_value=99.0),
    }
    result = _score_sn_recon(comparisons, score_clip=-1.0, normalization="precomputed")
    assert result.score == pytest.approx(-1.0)
    assert result.per_metric["bad"].clipped is True
    assert result.per_metric["bad"].normalized == -5.0  # pre-clip value stored


def test_sn_recon_excludes_population_category():
    comparisons = {
        "participation_ratio": _make_comparison(0.9, category="population_structure"),
        "spike_f1": _make_comparison(0.6, category="temporal_fidelity"),
    }
    result = _score_sn_recon(comparisons, score_clip=None, normalization="precomputed")
    assert result.n_metrics == 1
    assert "participation_ratio" not in result.per_metric
    assert result.score == pytest.approx(0.6)


def test_sn_recon_returns_none_score_when_no_valid_metrics():
    comparisons = {
        "population_metric": _make_comparison(0.5, category="population_structure"),
    }
    result = _score_sn_recon(comparisons, score_clip=-1.0, normalization="precomputed")
    assert result.score is None
    assert result.n_metrics == 0


from ntt.evaluation.aggregation.category_scores import _get_comparisons


def _make_eval_report_dict(metric_name: str, raw_value: float) -> dict:
    """Minimal EvaluationReport.to_dict() with a single metric."""
    return {
        "results": {
            metric_name: {
                "name": metric_name,
                "category": "temporal_fidelity",
                "direction": "higher",
                "level": "population",
                "raw_value": raw_value,
                "normalized_value": None,
                "metadata": {},
                "value_range": [None, None],
                "source_data_type": None,
            }
        },
        "category_scores": None,
        "skipped_metrics": {},
        "config": {},
        "original_data_type": None,
        "reconstructed_data_type": None,
    }


def test_get_comparisons_uses_precomputed_when_present():
    recon = {
        "comparison_vs_poisson": {
            "spike_f1": {
                "name": "spike_f1",
                "category": "temporal_fidelity",
                "direction": "higher",
                "level": "population",
                "model_value": 0.74,
                "baseline_value": 0.0,
                "ceiling_value": 1.0,
                "normalized_score": 0.85,
                "value_range": [None, None],
                "status": "ok",
                "reason": None,
            }
        }
    }
    baseline = {
        "aggregated": _make_eval_report_dict("spike_f1", 0.0),
        "perfect_aggregated": _make_eval_report_dict("spike_f1", 1.0),
    }
    # baseline is provided but force_recompute=False → precomputed wins
    comps, norm = _get_comparisons(recon, baseline, force_recompute=False)
    assert norm == "precomputed"
    assert comps["spike_f1"].normalized_score == pytest.approx(0.85)


def test_get_comparisons_recomputes_when_forced():
    recon = {
        "comparison_vs_poisson": {
            "spike_f1": {
                "name": "spike_f1",
                "category": "temporal_fidelity",
                "direction": "higher",
                "level": "population",
                "model_value": 0.74,
                "baseline_value": 0.0,
                "ceiling_value": 1.0,
                "normalized_score": 0.85,
                "value_range": [None, None],
                "status": "ok",
                "reason": None,
            }
        },
        "aggregated": _make_eval_report_dict("spike_f1", 0.6),
    }
    baseline = {
        "aggregated": _make_eval_report_dict("spike_f1", 0.0),
        "perfect_aggregated": _make_eval_report_dict("spike_f1", 1.0),
    }
    # These minimal fixtures record no sessions/settings; the math is under test here,
    # so accept the unrecorded aspects with a warning (see the comparability tests).
    comps, norm = _get_comparisons(recon, baseline, force_recompute=True, common_parts_only=True)
    assert norm == "recomputed"
    assert comps["spike_f1"].normalized_score == pytest.approx(0.6)  # (0.6-0.0)/(1.0-0.0)


def test_get_comparisons_recomputes_when_precomputed_missing():
    recon = {"aggregated": _make_eval_report_dict("spike_f1", 0.4)}
    baseline = {
        "aggregated": _make_eval_report_dict("spike_f1", 0.0),
        "perfect_aggregated": _make_eval_report_dict("spike_f1", 1.0),
    }
    comps, norm = _get_comparisons(recon, baseline, force_recompute=False, common_parts_only=True)
    assert norm == "recomputed"
    assert comps["spike_f1"].normalized_score == pytest.approx(0.4)


def test_get_comparisons_raises_when_no_baseline_and_no_precomputed():
    recon = {"aggregated": _make_eval_report_dict("spike_f1", 0.4)}
    with pytest.raises(ValueError, match="comparison_vs_poisson not found"):
        _get_comparisons(recon, poisson_baseline=None, force_recompute=False)


def test_get_comparisons_recomputes_with_compact_eval_report_format():
    # eval_report.json is EvaluationReport.to_dict() directly — no "aggregated" wrapper.
    # Verify the recompute path handles both results.json and eval_report.json formats.
    compact_recon = _make_eval_report_dict("spike_f1", 0.6)  # no "aggregated" key
    assert "aggregated" not in compact_recon
    baseline = {
        "aggregated": _make_eval_report_dict("spike_f1", 0.0),
        "perfect_aggregated": _make_eval_report_dict("spike_f1", 1.0),
    }
    comps, norm = _get_comparisons(
        compact_recon, baseline, force_recompute=False, common_parts_only=True
    )
    assert norm == "recomputed"
    assert comps["spike_f1"].normalized_score == pytest.approx(0.6)


from ntt.evaluation.aggregation.category_scores import _POP_STRUCTURE_CATEGORIES


def _score_population_structure(comparisons, score_clip, normalization):
    """Test shim: score the population-structure bucket via the unified helper."""
    return _score_comparison_bucket(
        comparisons, _POP_STRUCTURE_CATEGORIES, score_clip, normalization
    )


def _pop_comparison(
    name: str, normalized: float, category: str = "population_structure"
) -> tuple[str, MetricComparison]:
    return name, MetricComparison(
        name=name,
        category=MetricCategory(category),
        direction=MetricDirection("higher"),
        level=MetricLevel("population"),
        model_value=0.5,
        baseline_value=0.0,
        ceiling_value=1.0,
        normalized_score=normalized,
        status="ok",
    )


def test_pop_structure_averages_population_structure_metrics():
    comps = dict(
        [
            _pop_comparison("participation_ratio", 0.7),
            _pop_comparison("variance_explained", 0.5),
        ]
    )
    result = _score_population_structure(comps, score_clip=None, normalization="precomputed")
    assert result.score == pytest.approx(0.6)  # two distinct families -> plain mean
    assert result.n_metrics == 2
    assert result.normalization == "precomputed;family_weighted"


def test_pop_structure_includes_subspace_overlap():
    # subspace_overlap is intentionally not excluded; its bin variants share one family.
    comps = dict(
        [
            _pop_comparison("participation_ratio", 0.7),
            _pop_comparison("subspace_overlap", -2.0),
            _pop_comparison("subspace_overlap_5ms", -3.0),
        ]
    )
    result = _score_population_structure(comps, score_clip=None, normalization="precomputed")
    assert result.n_metrics == 3
    assert "subspace_overlap" in result.per_metric
    assert "subspace_overlap_5ms" in result.per_metric
    # families: participation_ratio(0.7), subspace_overlap(mean(-2.0,-3.0)=-2.5) -> (0.7-2.5)/2
    assert result.score == pytest.approx((0.7 + -2.5) / 2)


def test_pop_structure_excludes_sn_recon_categories():
    comps = dict(
        [
            _pop_comparison("participation_ratio", 0.7, category="population_structure"),
            _pop_comparison("spike_f1", 0.9, category="temporal_fidelity"),
        ]
    )
    result = _score_population_structure(comps, score_clip=None, normalization="precomputed")
    assert result.n_metrics == 1
    assert "spike_f1" not in result.per_metric


def test_pop_structure_excludes_latent_quality():
    # latent_quality metrics are not part of the population_structure bucket.
    comps = dict(
        [
            _pop_comparison("participation_ratio", 0.7, category="population_structure"),
            _pop_comparison("recon_mse", 0.9, category="latent_quality"),
        ]
    )
    result = _score_population_structure(comps, score_clip=None, normalization="precomputed")
    assert result.n_metrics == 1
    assert "recon_mse" not in result.per_metric
    assert result.score == pytest.approx(0.7)


def test_pop_structure_clips_below_threshold():
    comps = dict([_pop_comparison("bad_metric", -5.0)])
    result = _score_population_structure(comps, score_clip=-1.0, normalization="precomputed")
    assert result.score == pytest.approx(-1.0)
    assert result.per_metric["bad_metric"].clipped is True


def test_pop_structure_excludes_aggregate_blocklist():
    # token_dims_per_second / repeats_* never count in any aggregated column.
    comps = dict(
        [
            _pop_comparison("participation_ratio", 0.7),
            _pop_comparison("token_dims_per_second", 0.9),
            _pop_comparison("repeats_snr", 0.9),
        ]
    )
    result = _score_population_structure(comps, score_clip=None, normalization="precomputed")
    assert result.n_metrics == 1
    assert "token_dims_per_second" not in result.per_metric
    assert "repeats_snr" not in result.per_metric


def _m1_report(raw_by_name: dict[str, float]) -> dict:
    """Build a raw M1 robustness eval report (eval_report.json shape).

    The de-normalized latent bucket reads raw latent-NMSE values keyed by
    corruption from ``results`` and computes retention ``1/(1+d)``.
    """
    return {
        "results": {
            name: {"raw_value": v, "category": "latent_quality", "direction": "lower"}
            for name, v in raw_by_name.items()
        }
    }


def test_m1_retention_from_canonical_components_mean():
    # d = mean of the three canonical components; retention = 1/(1+d).
    # per-dim, permute_neurons, and other variants are ignored.
    m1 = _m1_report(
        {
            "robustness_input/window_shift|max_shift_ms=3.0": 0.8,
            "robustness_input/jitter|k_ms=1.0|p=0.2": 0.1,
            "robustness_input/drop_and_add|fraction=0.02": 0.3,
            "robustness_input_per_dim/jitter|k_ms=1.0|p=0.2": 99.0,
            "robustness_input/permute_neurons": 99.0,
        }
    )
    retention, d, prov = resolve_m1_retention(m1)
    assert prov == "components_mean"
    assert d == pytest.approx((0.8 + 0.1 + 0.3) / 3)
    assert retention == pytest.approx(1.0 / (1.0 + (0.8 + 0.1 + 0.3) / 3))


def test_m1_resolves_via_components_mean_on_a_real_report_shape():
    """M1 resolves from the three canonical components, and says so.

    The aggregate row ``robustness_input_core_avg/aggregate_non_population`` is
    never emitted by the M1 job (sub_label is None, so _row_metric_key returns
    the bare metric), and the derived row is dropped before reaching the report
    for want of a captured schema. Real M1 eval_report.json files carry
    26 result keys, none matching.
    """
    report = _m1_report(
        {
            "robustness_input/window_shift|max_shift_ms=3.0": 0.10,
            "robustness_input/jitter|k_ms=1.0|p=0.2": 0.20,
            "robustness_input/drop_and_add|fraction=0.02": 0.30,
            "robustness_input/permute_neurons": 9.9,
        }
    )
    retention, raw_d, provenance = resolve_m1_retention(report)

    # mean(0.10, 0.20, 0.30) = 0.20; retention = 1/(1+0.20) = 0.8333...
    assert raw_d == pytest.approx(0.20)
    assert retention == pytest.approx(1.0 / 1.2)
    assert provenance == "components_mean"


def test_m1_retention_excluded_when_components_incomplete():
    # A non-default M1 config missing a canonical component -> exclude (warn).
    m1 = _m1_report(
        {
            "robustness_input/window_shift|max_shift_ms=3.0": 0.8,
            "robustness_input/jitter|k_ms=1.0|p=0.2": 0.1,
        }
    )
    retention, d, prov = resolve_m1_retention(m1)
    assert retention is None and d is None
    assert prov == "excluded_no_components"


def test_m1_retention_excluded_when_no_report():
    assert resolve_m1_retention(None) == (None, None, "excluded_no_report")
    assert resolve_m1_retention({}) == (None, None, "excluded_no_report")


def test_m1_retention_excludes_on_malformed_raw_value_without_crashing():
    # A non-numeric / non-finite raw_value must exclude (warn), never raise.
    m1 = _m1_report(
        {
            "robustness_input/window_shift|max_shift_ms=3.0": 0.3,
            "robustness_input/jitter|k_ms=1.0|p=0.2": 0.1,
        }
    )
    m1["results"]["robustness_input/drop_and_add|fraction=0.02"] = {"raw_value": "oops"}
    retention, d, prov = resolve_m1_retention(m1)
    assert retention is None and prov == "excluded_no_components"


def test_latent_bucket_pools_m1_retention():
    # M1 contributes a single retention member (its own family, weight 1).
    m1 = _m1_report(
        {
            "robustness_input/window_shift|max_shift_ms=3.0": 0.0,
            "robustness_input/jitter|k_ms=1.0|p=0.2": 0.0,
            "robustness_input/drop_and_add|fraction=0.02": 0.0,
        }
    )
    diag = score_latent_structure_and_robustness({}, m1, None)
    assert diag is not None
    assert diag.score == pytest.approx(1.0)  # d=0 -> retention 1.0
    assert diag.n_metrics == 1
    assert set(diag.per_metric) == {"m1:components_mean"}
    assert diag.provenance == {"m1": "components_mean", "m2": "off"}


def test_latent_bucket_labels_m1_member_as_components_mean():
    """The bucket's per_metric key must name the path that actually resolved."""
    m1 = _m1_report(
        {
            "robustness_input/window_shift|max_shift_ms=3.0": 0.0,
            "robustness_input/jitter|k_ms=1.0|p=0.2": 0.0,
            "robustness_input/drop_and_add|fraction=0.02": 0.0,
        }
    )
    diag = score_latent_structure_and_robustness({}, m1, None, include_m2=False)

    assert diag is not None
    assert "m1:components_mean" in diag.per_metric
    assert not any("aggregate_non_population" in k for k in diag.per_metric)


from ntt.evaluation.aggregation.category_scores import (
    _score_compression,
    _score_encoding,
    _score_generalization,
)


def test_generalization_reads_from_summary():
    recon = {"generalization": {"summary": {"generalization_score": 0.87}}}
    assert _score_generalization(recon) == pytest.approx(0.87)


def test_generalization_returns_none_when_missing():
    assert _score_generalization({}) is None
    assert _score_generalization({"generalization": {}}) is None
    assert _score_generalization({"generalization": {"summary": {}}}) is None


def test_compression_reads_named_metric():
    recon = {"compression": {"compression_rate": 8.0, "latent_dim_reduction": 80.0}}
    assert _score_compression(recon, "compression_rate") == pytest.approx(8.0)
    assert _score_compression(recon, "latent_dim_reduction") == pytest.approx(80.0)


def test_compression_returns_none_when_missing():
    assert _score_compression({}, "compression_rate") is None
    assert _score_compression({"compression": {}}, "compression_rate") is None


def _enc(aggregated: dict, final_eval: dict | None = None) -> DownstreamScoreView:
    return DownstreamScoreView(
        split="test",
        aggregated=aggregated,
        per_session={},
        complete=True,
        provenance={"final_eval": final_eval},
    )


def _dec(
    r2_by_session: dict[str, float | None], *, complete: bool = True, missing: list | None = None
) -> DownstreamScoreView:
    return DownstreamScoreView(
        split="test",
        aggregated={},
        per_session={sid: {"r2": r2} for sid, r2 in r2_by_session.items()},
        complete=complete,
        provenance={"missing": missing or []},
    )


def test_encoding_returns_nc_r2_value():
    view = _enc({"rate_pearson": {"value": 0.260}, "nc_r2": {"value": 0.289, "n": 23}})
    score, diag = _score_encoding(view)
    assert score == pytest.approx(0.289)
    assert diag is None


def test_encoding_handles_negative_nc_r2():
    score, diag = _score_encoding(_enc({"nc_r2": {"value": -0.374}}))
    assert score == pytest.approx(-0.374)
    assert diag is None


def test_encoding_returns_none_and_warns_when_nc_r2_missing(caplog):
    with caplog.at_level(logging.WARNING):
        score, _ = _score_encoding(_enc({}, final_eval={"status": "failed", "error": "oom"}))
    assert score is None
    assert any("'error': 'oom'" in r.message for r in caplog.records)


def test_decoding_averages_r2_across_sessions():
    score, diag = score_decoding_merged(_dec({"session_a": 0.82, "session_b": 0.76}), None)
    assert score == pytest.approx((0.82 + 0.76) / 2)
    assert diag.n_metrics == 2


def test_decoding_returns_none_when_no_sessions():
    assert score_decoding_merged(_dec({}), None)[0] is None


def test_decoding_skips_sessions_with_missing_r2():
    score, _ = score_decoding_merged(_dec({"good": 0.80, "bad": None}), None)
    assert score == pytest.approx(0.80)


def test_incomplete_decoding_cohort_scores_none_by_default(caplog):
    view = _dec({"a": 0.8}, complete=False, missing=["b"])
    with caplog.at_level(logging.WARNING):
        assert score_decoding_merged(view, None) == (None, None)
    assert any("['b']" in r.message for r in caplog.records)


def test_incomplete_decoding_cohort_scores_over_found_sessions_when_allowed():
    view = _dec({"a": 0.8, "c": 0.6}, complete=False, missing=["b"])
    score, diag = score_decoding_merged(view, None, allow_incomplete_cohort=True)
    assert score == pytest.approx(0.7)
    assert diag.n_metrics == 2


def test_compute_category_scores_passes_the_incomplete_cohort_opt_in():
    view = _dec({"a": 0.8}, complete=False, missing=["b"])
    assert compute_category_scores(decoding=view).decoding is None
    allowed = compute_category_scores(decoding=view, allow_incomplete_decoding_cohort=True)
    assert allowed.decoding == pytest.approx(0.8)


from ntt.evaluation.aggregation.category_scores import compute_category_scores


def _full_recon_report() -> dict:
    """Minimal recon report with comparison_vs_poisson, generalization, compression."""
    return {
        "comparison_vs_poisson": {
            "spike_f1": {
                "name": "spike_f1",
                "category": "temporal_fidelity",
                "direction": "higher",
                "level": "population",
                "model_value": 0.74,
                "baseline_value": 0.0,
                "ceiling_value": 1.0,
                "normalized_score": 0.8,
                "value_range": [None, None],
                "status": "ok",
                "reason": None,
            },
            "participation_ratio": {
                "name": "participation_ratio",
                "category": "population_structure",
                "direction": "higher",
                "level": "population",
                "model_value": 0.5,
                "baseline_value": 0.0,
                "ceiling_value": 1.0,
                "normalized_score": 0.5,
                "value_range": [None, None],
                "status": "ok",
                "reason": None,
            },
        },
        "generalization": {"summary": {"generalization_score": 0.87}},
        "compression": {"compression_rate": 8.0},
    }


def test_compute_category_scores_recon_only():
    scores = compute_category_scores(recon_report=_full_recon_report())
    assert scores.single_neuron_recon == pytest.approx(0.8)
    assert scores.population_structure == pytest.approx(0.5)
    assert scores.generalization == pytest.approx(0.87)
    assert scores.compression == pytest.approx(8.0)
    # no structure-preservation metrics and no M1/M2 -> latent bucket has no contributors
    assert scores.latent_structure_and_robustness is None
    assert scores.downstream_encoding is None
    assert scores.decoding is None


def test_compute_category_scores_diagnostics_populated():
    scores = compute_category_scores(recon_report=_full_recon_report())
    assert scores.diagnostics["single_neuron_recon"] is not None
    assert scores.diagnostics["single_neuron_recon"].n_metrics == 1
    assert scores.diagnostics["population_structure"] is not None
    # Scalar pass-throughs have None diagnostics
    assert scores.diagnostics["generalization"] is None
    assert scores.diagnostics["compression"] is None


def test_compute_category_scores_no_inputs_returns_all_none():
    scores = compute_category_scores()
    assert scores.single_neuron_recon is None
    assert scores.population_structure is None
    assert scores.generalization is None
    assert scores.compression is None
    assert scores.latent_structure_and_robustness is None


def test_compute_category_scores_raises_when_recon_lacks_precomputed_and_no_baseline():
    recon = {"aggregated": _make_eval_report_dict("spike_f1", 0.5)}  # no comparison_vs_poisson
    with pytest.raises(ValueError, match="comparison_vs_poisson not found"):
        compute_category_scores(recon_report=recon)


def test_compute_category_scores_with_m1():
    # Raw M1 robustness report -> retention pools into latent_structure_and_robustness.
    m1 = _m1_report(
        {
            "robustness_input/window_shift|max_shift_ms=3.0": 0.0,
            "robustness_input/jitter|k_ms=1.0|p=0.2": 0.0,
            "robustness_input/drop_and_add|fraction=0.02": 0.0,
        }
    )
    scores = compute_category_scores(m1_report=m1)
    assert scores.latent_structure_and_robustness == pytest.approx(1.0)  # d=0 -> retention 1.0
    assert scores.single_neuron_recon is None  # no recon provided


def test_public_exports_importable_from_package():
    from ntt.evaluation.aggregation import CategoryScores, compute_category_scores  # noqa: F401


from ntt.evaluation.aggregation.category_scores import family, family_weighted_mean


def test_family_strips_bin_suffix():
    assert family("acg_correlation_5ms") == "acg_correlation"
    assert family("acg_correlation_100ms") == "acg_correlation"
    assert family("spike_count_correlation") == "spike_count_correlation"


def test_family_strips_perturbation_tail():
    assert family("robustness_input/jitter|k_ms=1.0|p=0.2") == "robustness_input/jitter"
    assert family("robustness_input/drop_neurons|fraction=0.15") == "robustness_input/drop_neurons"


def test_family_plain_name_unchanged():
    assert family("population_structure_preservation") == "population_structure_preservation"


def test_family_weighted_mean_balances_families():
    # one 3-bin family (mean 0.0) + one single metric (1.0) -> (0.0 + 1.0)/2 = 0.5
    pairs = [("acg_5ms", -0.1), ("acg_10ms", 0.0), ("acg_100ms", 0.1), ("subspace_overlap", 1.0)]
    assert family_weighted_mean(pairs) == 0.5


def test_family_weighted_mean_empty_is_none():
    assert family_weighted_mean([]) is None


from ntt.evaluation.aggregation.category_scores import resolve_m1_retention, resolve_m2_acg


def _m2_report(eps_to_acg: dict[float, float], metric: str = "acg_correlation") -> dict:
    """Build a raw M2 latent-noise eval report (eval_report.json shape) with an
    epsilon sweep keyed under ``<metric>/eps_<eps>`` and ``metadata.epsilon`` set."""
    return {
        "results": {
            f"{metric}/eps_{eps}": {
                "raw_value": acg,
                "direction": "higher",
                "category": "single_unit_statistics",
                "metadata": {"epsilon": eps},
            }
            for eps, acg in eps_to_acg.items()
        },
    }


def test_resolve_m2_acg_matches_on_epsilon_metadata():
    # The on-disk key suffix is unreliable (eps_1 for 1.0); match metadata.epsilon.
    # Transform is max(acg, 0): a negative ACG clamps to 0, but raw is preserved.
    m2 = _m2_report({0.0: 0.9, 0.1: -0.2, 1.0: 0.3})
    bounded, acg, prov, name = resolve_m2_acg(m2, "acg_correlation", 0.1)
    assert prov == "included"
    assert acg == pytest.approx(-0.2)  # raw preserved
    assert bounded == pytest.approx(0.0)  # max(-0.2, 0)
    assert name == "acg_correlation/eps_0.1"
    # A positive ACG passes through unchanged.
    bounded_pos, acg_pos, _, _ = resolve_m2_acg(m2, "acg_correlation", 1.0)
    assert acg_pos == pytest.approx(0.3)
    assert bounded_pos == pytest.approx(0.3)


def test_resolve_m2_acg_excluded_when_epsilon_absent(caplog):
    import logging

    m2 = _m2_report({0.0: 0.9, 1.0: 0.3})  # no eps=0.1
    with caplog.at_level(logging.WARNING):
        bounded, acg, prov, name = resolve_m2_acg(m2, "acg_correlation", 0.1)
    assert bounded is None and acg is None and name is None
    assert prov == "excluded_missing_key"
    assert any("at epsilon=0.1 not found" in r.message for r in caplog.records)


def test_resolve_m2_acg_excluded_when_no_report():
    assert resolve_m2_acg(None, "acg_correlation", 0.1) == (None, None, "excluded_no_report", None)


def test_latent_bucket_m2_off_by_default():
    # include_m2 defaults to False: M2 is not read even when a report is supplied.
    m2 = _m2_report({0.1: 0.5})
    diag = score_latent_structure_and_robustness(
        {"population_structure_preservation": _cmp("population_structure_preservation", 1.0)},
        None,
        m2,
    )
    assert diag is not None
    assert diag.provenance["m2"] == "off"
    assert not any(k.startswith("m2:") for k in diag.per_metric)


def test_latent_bucket_m2_included_when_flag_on():
    m2 = _m2_report({0.0: 0.9, 0.1: 0.4, 1.0: 0.3})
    diag = score_latent_structure_and_robustness(
        {}, None, m2, include_m2=True, m2_metric="acg_correlation", m2_epsilon=0.1
    )
    assert diag is not None
    # single member: max(acg, 0) at eps=0.1 = max(0.4, 0) = 0.4
    assert diag.score == pytest.approx(0.4)
    assert diag.provenance["m2"] == "included"
    assert set(diag.per_metric) == {"m2:acg_correlation/eps_0.1"}


def test_latent_bucket_m2_negative_acg_clamps_to_zero():
    # A negative ACG (e.g. the soft1 wanx ≈ -0.16) contributes 0, not partial credit.
    m2 = _m2_report({0.1: -0.16})
    diag = score_latent_structure_and_robustness({}, None, m2, include_m2=True, m2_epsilon=0.1)
    assert diag is not None
    assert diag.score == pytest.approx(0.0)
    assert diag.per_metric["m2:acg_correlation/eps_0.1"].raw_value == pytest.approx(-0.16)
    assert diag.per_metric["m2:acg_correlation/eps_0.1"].normalized == pytest.approx(0.0)


from ntt.evaluation.aggregation.category_scores import score_decoding_merged


def _with_sessions(report, per_session_r2=None):
    """Add the per-session block that the ridge scorer reads.
    Without ``per_session_r2``: one session, ``"s1"``, holding the report's results.
    With ``per_session_r2`` (session name -> R²): one session per entry, each holding the
    report's single ridge entry with that session's R²."""
    results = report.get("aggregated", report)["results"]
    if per_session_r2 is None:
        per_session = {"s1": {"results": results}}
    else:
        [(key, entry)] = results.items()
        per_session = {}
        for session, r2 in per_session_r2.items():
            per_session[session] = {"results": {key: {**entry, "raw_value": r2}}}
    return {**report, "aggregation_strategy": "mean", "per_session": per_session}


def test_decoding_poyo_only():
    dec = _dec({"s1": 0.4, "s2": 0.6})
    score, diag = score_decoding_merged(dec, None)
    assert score == 0.5
    assert diag.n_metrics == 2


def test_decoding_ridge_only_excludes_classification():
    recon = {
        "aggregated": {
            "results": {
                "behavioral_regression_cursor_pos": {
                    "metadata": {"chunk_window_ms": 200},
                    "name": "behavioral_regression",
                    "category": "downstream_decoding",
                    "raw_value": 0.2,
                },
                "behavioral_temporal_decoder_cursor_pos": {
                    "name": "behavioral_temporal_decoder",
                    "category": "downstream_decoding",
                    "raw_value": 0.4,
                },
                "behavioral_classification_movement_phase": {
                    "name": "behavioral_classification",
                    "category": "downstream_decoding",
                    "raw_value": 0.99,
                },
            }
        }
    }
    score, diag = score_decoding_merged(None, _with_sessions(recon))
    assert score == pytest.approx(
        0.3
    )  # (0.2 + 0.4)/2 ; classification excluded (fmean float error)


def test_decoding_merges_both():
    dec = _dec({"s1": 1.0})
    recon = {
        "aggregated": {
            "results": {
                "behavioral_regression_cursor_pos": {
                    "metadata": {"chunk_window_ms": 200},
                    "name": "behavioral_regression",
                    "category": "downstream_decoding",
                    "raw_value": 0.0,
                },
            }
        }
    }
    score, _ = score_decoding_merged(dec, _with_sessions(recon))
    assert score == 0.5


def test_decoding_none_when_empty():
    assert score_decoding_merged(None, None) == (None, None)


def test_decoding_skips_nonfinite_ridge_and_poyo():
    # A single NaN/inf POYO R² must be dropped, not poison statistics.fmean and
    # turn the whole decoding column into NaN. Ridge is different: NaN stops the scoring.
    dec = _dec({"good": 0.8, "bad": float("inf")})
    recon = {
        "aggregated": {
            "results": {
                "behavioral_regression_cursor_pos": {
                    "metadata": {"chunk_window_ms": 200},
                    "name": "behavioral_regression",
                    "category": "downstream_decoding",
                    "raw_value": 0.4,
                },
            }
        }
    }
    score, diag = score_decoding_merged(dec, _with_sessions(recon))
    # only the two finite values contribute: poyo good (0.8) + ridge regression (0.4)
    assert score == pytest.approx((0.8 + 0.4) / 2)
    assert set(diag.per_metric) == {"poyo/good", "ridge/behavioral_regression_cursor_pos"}
    # A NaN ridge score in a session stops the scoring instead of being dropped.
    recon["aggregated"]["results"]["behavioral_regression_cursor_pos"]["raw_value"] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        score_decoding_merged(dec, _with_sessions(recon))


def test_decoding_picks_up_the_real_per_target_ridge_key_shape():
    """Real recon reports key each entry by its *display* name — the pipeline suffixes
    ``_<target>`` per target (``expand_results``) while ``name`` stays bare. Values and
    keys are copied from ``07a_p20_w0/recon/results.json``. Looking the bare name up as
    a key finds nothing and drops every ridge decoder at no log level at all."""
    recon = {
        "aggregated": {
            "results": {
                "behavioral_regression_cursor_pos": {
                    "metadata": {"chunk_window_ms": 200},
                    "name": "behavioral_regression",
                    "category": "downstream_decoding",
                    "raw_value": 0.22885168122677638,
                },
                "behavioral_regression_cursor_vel": {
                    "name": "behavioral_regression",
                    "category": "downstream_decoding",
                    "raw_value": 0.29549947949701805,
                },
                "behavioral_classification_movement_phase": {
                    "name": "behavioral_classification",
                    "category": "downstream_decoding",
                    "raw_value": 0.7335873637943453,
                },
                "behavioral_classification_trial_target_id": {
                    "name": "behavioral_classification",
                    "category": "downstream_decoding",
                    "raw_value": 0.4132037804408144,
                },
            }
        }
    }
    score, diag = score_decoding_merged(None, _with_sessions(recon))
    # Both regression targets contribute, keyed by the suffixed key so two targets
    # cannot clobber each other in the diagnostics. Classification is accuracy, not R².
    assert set(diag.per_metric) == {
        "ridge/behavioral_regression_cursor_pos",
        "ridge/behavioral_regression_cursor_vel",
    }
    assert score == pytest.approx((0.22885168122677638 + 0.29549947949701805) / 2)


def test_decoding_falls_back_to_the_key_prefix_when_name_is_absent():
    """An entry with no ``name`` field is still recognised by its key prefix, so a
    missing field cannot silently drop the whole ridge family the way the bare-name
    key lookup did."""
    recon = {
        "aggregated": {
            "results": {
                "behavioral_regression_cursor_vel": {
                    "category": "downstream_decoding",
                    "raw_value": 0.4,
                },
            }
        }
    }
    score, diag = score_decoding_merged(None, _with_sessions(recon))
    assert score == pytest.approx(0.4)
    assert set(diag.per_metric) == {"ridge/behavioral_regression_cursor_vel"}


def test_decoding_weights_ridge_as_a_peer_family_of_all_poyo_sessions():
    """POYO and ridge are two families of weight 1 each, like every other composite.
    A flat mean would give the 2 ridge targets 2/14 of the column and let 12 POYO
    sessions set it almost alone (12/14 = 0.857 here)."""
    dec = _dec({f"s{i}": 1.0 for i in range(12)})
    recon = {
        "aggregated": {
            "results": {
                "behavioral_regression_cursor_pos": {
                    "metadata": {"chunk_window_ms": 200},
                    "name": "behavioral_regression",
                    "raw_value": 0.0,
                },
                "behavioral_regression_cursor_vel": {
                    "name": "behavioral_regression",
                    "raw_value": 0.0,
                },
            }
        }
    }
    score, diag = score_decoding_merged(dec, _with_sessions(recon))
    assert score == pytest.approx(0.5)
    assert diag.n_metrics == 14


def test_decoding_reports_each_family_mean_in_provenance():
    """The two sub-scores measure different things — a trained decoder's ceiling vs
    linear decodability of the latent — so both are recorded next to the composite."""
    dec = _dec({"s1": 0.8, "s2": 0.6})
    recon = {
        "aggregated": {
            "results": {
                "behavioral_regression_cursor_vel": {
                    "name": "behavioral_regression",
                    "raw_value": 0.2,
                },
            }
        }
    }
    score, diag = score_decoding_merged(dec, _with_sessions(recon))
    assert score == pytest.approx((0.7 + 0.2) / 2)
    assert float(diag.provenance["poyo"].split()[0]) == pytest.approx(0.7)
    assert diag.provenance["poyo"].endswith("(2 sessions)")
    assert float(diag.provenance["ridge"].split()[0]) == pytest.approx(0.2)
    assert diag.provenance["ridge"].endswith("(1 target)")


def test_decoding_provenance_marks_an_absent_family():
    dec = _dec({"s1": 0.8})
    score, diag = score_decoding_merged(dec, None)
    assert score == pytest.approx(0.8)  # a single family carries the column alone
    assert diag.provenance["ridge"] == "absent"


def test_decoding_accepts_compact_eval_report_shape():
    """A compact eval_report.json must not silently drop the ridge family.

    ``_report_results`` already accepts both shapes; ``score_decoding_merged``
    hardcoded the ``aggregated`` wrapper, so the ridge half of decoding vanished
    for compact reports while the recon buckets scored fine.
    """
    ridge_results = {
        "behavioral_regression_cursor_vel": {
            "name": "behavioral_regression",
            "raw_value": 0.1,
        },
    }
    dec = _dec({f"s{i}": 0.9 for i in range(12)})

    wrapped_score, wrapped_diag = score_decoding_merged(
        dec, _with_sessions({"aggregated": {"results": ridge_results}})
    )
    compact_score, compact_diag = score_decoding_merged(
        dec, _with_sessions({"results": ridge_results})
    )

    # Two equal families: (mean(poyo) + mean(ridge)) / 2 = (0.9 + 0.1) / 2 = 0.5
    assert compact_score == pytest.approx(0.5)
    assert compact_score == pytest.approx(wrapped_score)
    assert compact_diag.n_metrics == 13  # 12 POYO sessions + 1 ridge target
    assert compact_diag.provenance == wrapped_diag.provenance


from ntt.evaluation.aggregation.category_scores import (
    _STRUCTURE_PRESERVATION_METRICS,
    score_latent_structure_and_robustness,
)


def _cmp(name, model_value, *, status="ok"):
    """A structure-preservation comparison carrying a raw RSA ``model_value`` in
    [-1,1]. The de-normalized bucket transforms ``model_value`` via ``max(corr, 0)``
    and does NOT read ``normalized_score`` (left None to prove that)."""
    return MetricComparison(
        name=name,
        category=MetricCategory.LATENT_QUALITY,
        direction=MetricDirection.HIGHER_IS_BETTER,
        level=MetricLevel.POPULATION,
        model_value=model_value,
        baseline_value=0.0,
        ceiling_value=1.0,
        normalized_score=None,
        status=status,
    )


def test_latent_bucket_pools_structure_and_robustness():
    # structure: max(corr, 0) on raw RSA model_value; M1: retention 1/(1+d).
    comps = {
        "population_structure_preservation": _cmp("population_structure_preservation", 0.6),
        "per_unit_structure_preservation": _cmp("per_unit_structure_preservation", -0.5),
    }
    m1 = _m1_report(
        {
            "robustness_input/window_shift|max_shift_ms=3.0": 0.0,
            "robustness_input/jitter|k_ms=1.0|p=0.2": 0.0,
            "robustness_input/drop_and_add|fraction=0.02": 0.0,
        }
    )
    diag = score_latent_structure_and_robustness(comps, m1, None)
    assert diag is not None
    # families: pop_struct max(0.6,0)=0.6, per_unit max(-0.5,0)=0.0, m1 retention(d=0)=1.0
    # mean of family means = (0.6 + 0.0 + 1.0)/3
    assert diag.score == pytest.approx((0.6 + 0.0 + 1.0) / 3)
    assert diag.n_metrics == 3  # 2 structure + 1 m1 retention
    # raw RSA preserved in diagnostics; bounded transform stored in `normalized`
    pop = diag.per_metric["population_structure_preservation"]
    assert pop.raw_value == pytest.approx(0.6)
    assert pop.normalized == pytest.approx(0.6)
    assert pop.clipped is False
    # negative RSA clamps to 0 but keeps its raw value
    per_unit = diag.per_metric["per_unit_structure_preservation"]
    assert per_unit.raw_value == pytest.approx(-0.5)
    assert per_unit.normalized == pytest.approx(0.0)


@pytest.mark.parametrize("floor", [0.95, 1.0, float("nan")])
def test_raw_structure_survives_poisson_normalization_failure(floor):
    name = "population_structure_preservation"

    def report(value):
        payload = _make_eval_report_dict(name, value)
        payload["results"][name]["category"] = "latent_quality"
        payload["results"][name]["value_range"] = [-1, 1]
        return payload

    scores = compute_category_scores(
        recon_report={"aggregated": report(0.6)},
        poisson_baseline={"aggregated": report(floor), "perfect_aggregated": report(1.0)},
        common_parts_only=True,  # minimal fixtures record no sessions/settings
    )
    assert scores.latent_structure_and_robustness == pytest.approx(0.6)
    assert scores.diagnostics["latent_structure_and_robustness"].n_metrics == 1


def test_raw_structure_does_not_require_a_poisson_comparison_entry():
    name = "population_structure_preservation"
    recon = {"aggregated": _make_eval_report_dict(name, 0.6), "comparison_vs_poisson": {}}
    scores = compute_category_scores(recon_report=recon)
    assert scores.latent_structure_and_robustness == pytest.approx(0.6)


def test_latent_bucket_skips_structure_with_nonfinite_model_value():
    # A structure metric with no usable raw value is skipped, not crashed on.
    comps = {
        "population_structure_preservation": _cmp("population_structure_preservation", 0.5),
        "per_unit_structure_preservation": _cmp("per_unit_structure_preservation", float("nan")),
    }
    diag = score_latent_structure_and_robustness(comps, None, None)
    assert diag is not None
    assert diag.n_metrics == 1
    assert "per_unit_structure_preservation" not in diag.per_metric
    assert diag.score == pytest.approx(0.5)  # max(0.5, 0)


def test_latent_bucket_none_when_no_contributors():
    # No structure, no M1, M2 off -> the bucket has nothing to score.
    assert score_latent_structure_and_robustness({}, None, None) is None


def test_structure_preservation_metric_set():
    assert _STRUCTURE_PRESERVATION_METRICS == frozenset(
        {"population_structure_preservation", "per_unit_structure_preservation"}
    )


def test_population_structure_includes_subspace_excludes_latent_quality():
    # subspace_overlap counted; token_dims_per_second + repeats_* excluded;
    # latent_quality structure metrics not in the population bucket.
    from ntt.evaluation.aggregation.category_scores import (
        _EXCLUDED_FROM_AGGREGATES,
        _POP_STRUCTURE_CATEGORIES,
    )
    from ntt.evaluation.core.types import MetricCategory

    assert _POP_STRUCTURE_CATEGORIES == frozenset({MetricCategory.POPULATION_STRUCTURE})
    assert {
        "token_dims_per_second",
        "repeats_snr",
        "repeats_baseline",
    } <= _EXCLUDED_FROM_AGGREGATES


def test_category_scores_has_new_field_names():
    from ntt.evaluation.aggregation.category_scores import CategoryScores

    fields = CategoryScores.__dataclass_fields__
    assert "population_structure" in fields
    assert "latent_structure_and_robustness" in fields
    assert "decoding" in fields
    assert "population_stats" not in fields
    assert "latent_robustness" not in fields
    assert "downstream_decoding" not in fields


# ── Poisson-recompute comparability check (via category_scores entry points) ──


def _neuron_selection(**session_overrides) -> dict:
    """A recorded ``neuron_selection`` for sessions s1/s2."""
    sessions = {
        sid: {"n_units": 10, "n_selected": 10, "n_padded": 0, "unit_ids_sha256": sid * 32}
        for sid in ("s1", "s2")
    }
    sessions.update(session_overrides)
    return {
        "policy": "eval_unit_selection",
        "max_neurons": 200,
        "seed": 42,
        "padded_to": None,
        "sessions": sessions,
    }


def _matching_recon_and_reference() -> tuple[dict, dict]:
    """A recon report and a Poisson reference that record the same sessions and
    settings."""
    payload = _make_eval_report_dict("population_structure_preservation", 0.6)
    recon = {
        "aggregated": {**payload, "config": {"max_eval_seconds": None}},
        "per_session": {"s1": {}, "s2": {}},
        "aggregation_strategy": "mean",
        "split": "val",
        "tokenizer_config": {"bin_size_ms": 1.0, "window_length_seconds": 1.0},
        "neuron_selection": _neuron_selection(),
    }
    reference = {
        "aggregated": payload,
        "perfect_aggregated": payload,
        "per_session": {"s1": {}, "s2": {}},
        "perfect_per_session": {"s1": {}, "s2": {}},
        "metadata": {
            "aggregation_strategy": "mean",
            "bin_size_ms": 1.0,
            "window_length_seconds": 1.0,
            "max_eval_seconds": None,
            "split": "val",
            "neuron_selection": _neuron_selection(),
        },
    }
    return recon, reference


def test_poisson_recompute_raises_on_different_session_sets():
    from ntt.evaluation.aggregation.comparability import ReferenceMismatchError

    recon, reference = _matching_recon_and_reference()
    reference["per_session"] = reference["perfect_per_session"] = {"s1": {}, "s3": {}}
    with pytest.raises(ReferenceMismatchError) as excinfo:
        _get_comparisons(recon, reference, force_recompute=True)
    assert "model: n=2, not shared by all: ['s2']" in str(excinfo.value)
    assert "poisson_reference: n=2, not shared by all: ['s3']" in str(excinfo.value)


def test_poisson_recompute_raises_on_a_different_setting():
    from ntt.evaluation.aggregation.comparability import ReferenceMismatchError

    recon, reference = _matching_recon_and_reference()
    reference["metadata"]["bin_size_ms"] = 10.0
    with pytest.raises(ReferenceMismatchError, match="bin_size_ms: model=1.0, poisson_ref"):
        _get_comparisons(recon, reference, force_recompute=True)


def test_poisson_recompute_raises_when_scored_on_different_units():
    """The bug this check exists for: the model scored its first 32 units, zero-padded,
    while the reference scored the session's real units."""
    from ntt.evaluation.aggregation.comparability import ReferenceMismatchError

    recon, reference = _matching_recon_and_reference()
    recon["neuron_selection"] = _neuron_selection(
        s1={"n_units": 10, "n_selected": 10, "n_padded": 22, "unit_ids_sha256": "s1" * 32}
    )
    with pytest.raises(ReferenceMismatchError, match="neuron_selection: s1: 10/10 units"):
        _get_comparisons(recon, reference, force_recompute=True)


def test_poisson_recompute_cannot_check_unrecorded_neuron_selection():
    from ntt.evaluation.aggregation.comparability import UncheckedComparabilityError

    recon, reference = _matching_recon_and_reference()
    del recon["neuron_selection"]
    with pytest.raises(
        UncheckedComparabilityError, match="neuron_selection, because the recon results.json"
    ):
        _get_comparisons(recon, reference, force_recompute=True)


def test_poisson_recompute_matching_is_silent(caplog):
    recon, reference = _matching_recon_and_reference()
    with caplog.at_level("WARNING"):
        _get_comparisons(recon, reference, force_recompute=True)
    assert not any("Poisson recompute" in r.getMessage() for r in caplog.records)


def test_poisson_recompute_unrecorded_aspect_raises_unless_common_parts_only(caplog):
    from ntt.evaluation.aggregation.comparability import UncheckedComparabilityError

    # A reference that does not save its aggregation strategy.
    recon, reference = _matching_recon_and_reference()
    del reference["metadata"]["aggregation_strategy"]
    with pytest.raises(UncheckedComparabilityError) as excinfo:
        _get_comparisons(recon, reference, force_recompute=True)
    assert "aggregation_strategy, because the Poisson reference does not save it" in str(
        excinfo.value
    )
    assert "Pass --common-parts-only to continue with a warning instead" in str(excinfo.value)

    with caplog.at_level("WARNING"):
        _get_comparisons(recon, reference, force_recompute=True, common_parts_only=True)
    warning = next(r.getMessage() for r in caplog.records if "Poisson recompute" in r.getMessage())
    assert warning.startswith("WARNING: Poisson recompute: could NOT check")


def test_poisson_norm_sessions_recorded_on_recompute():
    name = "population_structure_preservation"
    payload = _make_eval_report_dict(name, 0.6)
    scores = compute_category_scores(
        recon_report={"aggregated": payload, "per_session": {"s1": {}, "s2": {}}},
        poisson_baseline={
            "aggregated": payload,
            "perfect_aggregated": payload,
            "per_session": {"s1": {}, "s2": {}},
        },
        force_recompute_poisson=True,
        common_parts_only=True,  # minimal fixtures record no settings
    )
    assert scores.poisson_norm_sessions == {"source": "recomputed", "session_ids": ["s1", "s2"]}


@pytest.mark.parametrize("size, expected", [(200, 0.6), (1000, 0.4), (None, 0.4)])
def test_position_regression_requires_verified_200ms(size, expected):
    report = {
        "results": {
            "behavioral_regression_cursor_pos": {
                "name": "behavioral_regression",
                "raw_value": 0.8,
                "metadata": {"primary_chunk_window_ms": size},
            },
            "behavioral_regression_cursor_vel": {
                "name": "behavioral_regression",
                "raw_value": 0.4,
                "metadata": {"chunk_window_ms": 1000},
            },
        }
    }
    score, diagnostics = score_decoding_merged(None, _with_sessions(report))
    assert score == pytest.approx(expected)
    if size == 200:
        assert "ridge/behavioral_regression_cursor_pos" in diagnostics.per_metric
    else:
        assert set(diagnostics.per_metric) == {"ridge/behavioral_regression_cursor_vel"}
        assert diagnostics.provenance["position_regression"] == "excluded_no_200ms"


def test_position_skipped_on_sessions_without_behavior_is_not_excluded():
    """Sessions without behavior list position regression under the merged skipped metrics
    of every report. A 200 ms position score that is used is still not excluded; a position
    regression that only appears as skipped (every sweep failed) is."""
    no_behavior = {
        "behavioral_regression_cursor_pos": "Computation error: Behavioral field 'cursor_pos' "
        "not found in data. Available: [] (skipped on 17 of 20 sessions)"
    }
    vel = {
        "name": "behavioral_regression",
        "raw_value": 0.4,
        "metadata": {"primary_chunk_window_ms": 200},
    }
    used = {
        "results": {
            "behavioral_regression_cursor_pos": {
                "name": "behavioral_regression",
                "raw_value": 0.8,
                "metadata": {"primary_chunk_window_ms": 200},
            },
            "behavioral_regression_cursor_vel": vel,
        },
        "skipped_metrics": no_behavior,
    }
    _, diagnostics = score_decoding_merged(None, _with_sessions(used))
    assert "ridge/behavioral_regression_cursor_pos" in diagnostics.per_metric
    assert "position_regression" not in diagnostics.provenance

    only_skipped = {
        "results": {"behavioral_regression_cursor_vel": vel},
        "skipped_metrics": no_behavior,
    }
    _, diagnostics = score_decoding_merged(None, _with_sessions(only_skipped))
    assert diagnostics.provenance["position_regression"] == "excluded_no_200ms"


VEL_RESULT = {"name": "behavioral_regression"}


def test_ridge_leaves_out_the_excluded_session():
    dropped = "perich_miller_population_2018/c_20131029_random_target_reaching"
    r2 = {dropped: -0.5, "a": 0.1, "b": 0.3}
    mean = statistics.fmean(r2.values())
    report = {"results": {"behavioral_regression_cursor_vel": {**VEL_RESULT, "raw_value": mean}}}
    score, _ = score_decoding_merged(None, _with_sessions(report, r2))
    assert score == pytest.approx(0.2)


# ── Decoding breakdown: the two family scores the composite averages ──────────


def _recon_with_ridge(pos: float, vel: float) -> dict:
    """A recon report that scores the Poisson-normalized buckets and carries both ridge
    regression targets (position at the 200 ms chunk size, so it enters the composite)."""
    results = {
        "behavioral_regression_cursor_pos": {
            "metadata": {"chunk_window_ms": 200},
            "name": "behavioral_regression",
            "category": "downstream_decoding",
            "raw_value": pos,
        },
        "behavioral_regression_cursor_vel": {
            "name": "behavioral_regression",
            "category": "downstream_decoding",
            "raw_value": vel,
        },
        "behavioral_classification_trial_target_id": {
            "name": "behavioral_classification",
            "category": "downstream_decoding",
            "raw_value": 0.9,
        },
    }
    return _with_sessions({**_full_recon_report(), "results": results})


def test_decoding_breakdown_reproduces_the_composite():
    scores = compute_category_scores(
        recon_report=_recon_with_ridge(pos=0.1, vel=0.3),
        decoding=_dec({"s1": 0.8, "s2": 0.9, "s3": 0.85}),
    )
    assert scores.decoding_poyo == pytest.approx(0.85)
    # Classification accuracy is not part of the linear family.
    assert scores.decoding_linear == pytest.approx(0.2)
    assert scores.decoding == pytest.approx(0.525)
    assert family_weighted_mean(
        [("poyo", scores.decoding_poyo), ("ridge", scores.decoding_linear)]
    ) == pytest.approx(scores.decoding, abs=1e-15)
