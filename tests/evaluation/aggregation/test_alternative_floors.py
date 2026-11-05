"""firing_rate_error is scored against a population-mean-rate reference, not the Poisson floor."""

import numpy as np
import pytest

from ntt.evaluation.aggregation.category_scores import compute_category_scores
from ntt.evaluation.aggregation.comparison import (
    ALTERNATIVE_FLOORS,
    POPULATION_MEAN_RATE,
    MetricComparison,
    population_mean_rate_unit_errors,
    with_alternative_floors,
)

# Two sessions. Session "a": one segment, E = 0.5, R = mean |[2, 4, 6, 8] - 5| = 2.
# Session "b": two segments, (E, R) = (0, 1) and (0.5, 5), so E = 0.25, R = 3.
# Aggregate: E = 0.375, R = 2.5, score = 1 - 0.375 / 2.5 = 0.85. (The mean of the
# per-session ratios would be 0.833 instead: the score is taken at the aggregate level.)
_RATES = {
    "a": [([2.0, 4.0, 6.0, 8.0], [2.0, 4.0, 6.0, 10.0])],
    "b": [([1.0, 3.0], [1.0, 3.0]), ([0.0, 10.0], [1.0, 10.0])],
}
_E, _R = 0.375, 2.5


def _firing_rate_result(original: list[float], reconstructed: list[float]) -> dict:
    errors = np.abs(np.asarray(original) - np.asarray(reconstructed))
    return {
        "name": "firing_rate_error",
        "category": "single_unit_statistics",
        "direction": "lower",
        "level": "per_unit",
        "raw_value": errors.tolist(),
        "normalized_value": None,
        "metadata": {"rates_original_hz": original, "rates_reconstructed_hz": reconstructed},
        "value_range": [0.0, None],
        "source_data_type": "binary_spikes",
    }


def _poisson_comparison(model_value: float = _E, baseline_value: float = 0.1) -> dict:
    """``comparison_vs_poisson.firing_rate_error`` as saved before the population-mean-rate
    reference: the Poisson floor, no ``baseline_kind``."""
    return {
        "name": "firing_rate_error",
        "category": "single_unit_statistics",
        "direction": "lower",
        "level": "per_unit",
        "model_value": model_value,
        "baseline_value": baseline_value,
        "ceiling_value": 0.0,
        "normalized_score": 1.0 - model_value / baseline_value,
        "value_range": [0.0, None],
        "status": "ok",
        "reason": None,
    }


def _spike_f1_comparison(score: float = 0.8) -> dict:
    return {
        "name": "spike_f1",
        "category": "temporal_fidelity",
        "direction": "higher",
        "level": "population",
        "model_value": score,
        "baseline_value": 0.0,
        "ceiling_value": 1.0,
        "normalized_score": score,
        "value_range": [0.0, 1.0],
        "status": "ok",
        "reason": None,
    }


def _recon_report(aggregation_strategy: str = "mean") -> dict:
    """The parts of a recon ``results.json`` the aggregation reads."""
    return {
        "aggregation_strategy": aggregation_strategy,
        "per_session_per_segment": {
            sid: [{"results": {"firing_rate_error": _firing_rate_result(o, r)}} for o, r in segs]
            for sid, segs in _RATES.items()
        },
        "comparison_vs_poisson": {
            "firing_rate_error": _poisson_comparison(),
            "spike_f1": _spike_f1_comparison(),
        },
    }


def _segment_results(report: dict) -> dict:
    return {
        sid: [s["results"] for s in segs]
        for sid, segs in report["per_session_per_segment"].items()
    }


def _comparisons(report: dict) -> dict[str, MetricComparison]:
    return {n: MetricComparison.from_dict(d) for n, d in report["comparison_vs_poisson"].items()}


def test_registry_covers_only_firing_rate_error():
    assert ALTERNATIVE_FLOORS == {"firing_rate_error": POPULATION_MEAN_RATE}


def test_unit_errors_use_the_population_mean_over_the_units_the_model_is_scored_on():
    np.testing.assert_allclose(
        population_mean_rate_unit_errors([2.0, 4.0, 6.0, 8.0], [2.0, 4.0, 6.0, 8.0]),
        [3.0, 1.0, 1.0, 3.0],
    )
    # Unit 0 has no reconstructed rate, so the model's error leaves it out; so does the
    # reference (and the population mean is taken over the other three units).
    np.testing.assert_allclose(
        population_mean_rate_unit_errors([2.0, 4.0, 6.0, 8.0], [np.nan, 4.0, 6.0, 8.0]),
        [np.nan, 2.0, 0.0, 2.0],
    )


def test_score_is_one_minus_aggregate_error_over_aggregate_reference():
    report = _recon_report()
    comparisons, per_session = with_alternative_floors(
        _comparisons(report), _segment_results(report), "mean"
    )
    fr = comparisons["firing_rate_error"]
    assert fr.normalized_score == pytest.approx(1.0 - _E / _R)
    assert fr.baseline_value == pytest.approx(_R)
    assert fr.baseline_kind == POPULATION_MEAN_RATE
    assert fr.poisson_baseline_value == pytest.approx(0.1)
    assert fr.model_value == pytest.approx(_E)
    assert fr.ceiling_value == 0.0
    assert per_session == {"firing_rate_error": {"a": pytest.approx(2.0), "b": pytest.approx(3.0)}}
    # Every other metric keeps its Poisson comparison.
    assert comparisons["spike_f1"] == _comparisons(report)["spike_f1"]


def test_saved_comparison_records_the_reference_and_round_trips():
    report = _recon_report()
    comparisons, _ = with_alternative_floors(
        _comparisons(report), _segment_results(report), "mean"
    )
    saved = comparisons["firing_rate_error"].to_dict()
    assert saved["baseline_kind"] == POPULATION_MEAN_RATE
    assert saved["poisson_baseline_value"] == pytest.approx(0.1)
    assert MetricComparison.from_dict(saved) == comparisons["firing_rate_error"]
    # A Poisson-floor comparison carries no baseline_kind.
    assert "baseline_kind" not in comparisons["spike_f1"].to_dict()


def test_a_comparison_already_against_the_reference_is_kept():
    report = _recon_report()
    comparisons, _ = with_alternative_floors(
        _comparisons(report), _segment_results(report), "mean"
    )
    again, per_session = with_alternative_floors(comparisons, None, "mean")
    assert again == comparisons
    assert per_session == {}


def test_median_strategy_aggregates_like_the_eval():
    report = _recon_report("median")
    # Median over two sessions is their mean; a 2-segment median is the segments' mean too.
    comparisons, _ = with_alternative_floors(
        _comparisons(report), _segment_results(report), "median"
    )
    assert comparisons["firing_rate_error"].baseline_value == pytest.approx(_R)


def test_weighted_mean_needs_segment_weights():
    report = _recon_report("weighted_mean")
    with pytest.raises(ValueError, match="segment weights"):
        with_alternative_floors(_comparisons(report), _segment_results(report), "weighted_mean")
    # Weights: "a" 1 bin; "b" segments of 1 and 3 bins.
    # E_b = (0 + 3 * 0.5) / 4 = 0.375, R_b = (1 + 3 * 5) / 4 = 4; sessions weighted 1 : 4.
    e = (0.5 + 4 * 0.375) / 5
    r = (2.0 + 4 * 4.0) / 5
    report["comparison_vs_poisson"]["firing_rate_error"] = _poisson_comparison(model_value=e)
    comparisons, _ = with_alternative_floors(
        _comparisons(report),
        _segment_results(report),
        "weighted_mean",
        segment_weights={"a": [1.0], "b": [1.0, 3.0]},
    )
    assert comparisons["firing_rate_error"].normalized_score == pytest.approx(1.0 - e / r)


def test_raises_when_the_segments_do_not_reproduce_the_reported_error():
    report = _recon_report()
    report["comparison_vs_poisson"]["firing_rate_error"] = _poisson_comparison(model_value=0.9)
    with pytest.raises(ValueError, match="does not reproduce"):
        with_alternative_floors(_comparisons(report), _segment_results(report), "mean")


def test_raises_when_per_unit_rates_are_missing():
    report = _recon_report()
    del report["per_session_per_segment"]["b"][1]["results"]["firing_rate_error"]["metadata"][
        "rates_original_hz"
    ]
    with pytest.raises(ValueError, match="rates_original_hz"):
        with_alternative_floors(_comparisons(report), _segment_results(report), "mean")


def test_aggregation_recomputes_old_reports_for_sn_recon():
    scores = compute_category_scores(recon_report=_recon_report())
    diag = scores.diagnostics["single_neuron_recon"]
    assert diag is not None
    assert diag.per_metric["firing_rate_error"].normalized == pytest.approx(0.85)
    assert not diag.per_metric["firing_rate_error"].clipped
    # One family each: (0.85 + 0.8) / 2. With the Poisson floor it was (-1 + 0.8) / 2.
    assert scores.single_neuron_recon == pytest.approx(0.825)
    assert "firing_rate_error=population_mean_rate" in diag.normalization
    assert scores.metric_references == {"firing_rate_error": POPULATION_MEAN_RATE}


def test_aggregation_uses_a_saved_reference_comparison_as_is():
    report = _recon_report()
    comparisons, _ = with_alternative_floors(
        _comparisons(report), _segment_results(report), "mean"
    )
    report["comparison_vs_poisson"] = {n: c.to_dict() for n, c in comparisons.items()}
    del report["per_session_per_segment"]  # not needed: nothing to recompute
    scores = compute_category_scores(recon_report=report)
    assert scores.single_neuron_recon == pytest.approx(0.825)


def test_aggregation_raises_when_an_old_report_has_no_per_segment_rates():
    report = _recon_report()
    del report["per_session_per_segment"]
    with pytest.raises(ValueError, match="per_session_per_segment"):
        compute_category_scores(recon_report=report)


def test_wide_table_norm_column_uses_the_reference():
    from scripts.analysis.build_metrics_table import flatten_tokenizer_metrics

    report = _recon_report()
    report["aggregated"] = {
        "results": {
            "firing_rate_error": {"category": "single_unit_statistics", "raw_value": _E},
            "spike_f1": {"category": "temporal_fidelity", "raw_value": 0.8},
        }
    }
    out = flatten_tokenizer_metrics(report, None, None, None)
    assert out["single_unit_statistics/firing_rate_error::norm"] == pytest.approx(0.85)
    assert out["temporal_fidelity/spike_f1::norm"] == pytest.approx(0.8)


def test_replot_comparisons_use_the_reference():
    from scripts.replot_from_eval_results import load_comparisons_from_results

    comparisons = load_comparisons_from_results(_recon_report())
    assert comparisons is not None
    assert comparisons["firing_rate_error"].normalized_score == pytest.approx(0.85)


def test_a_non_finite_model_error_stays_invalid_without_reading_rates():
    report = _recon_report()
    report["comparison_vs_poisson"]["firing_rate_error"] = _poisson_comparison(
        model_value=float("nan")
    )
    comparisons, per_session = with_alternative_floors(_comparisons(report), None, "mean")
    fr = comparisons["firing_rate_error"]
    assert fr.normalized_score is None and fr.reason == "non_finite_value"
    assert fr.baseline_kind == POPULATION_MEAN_RATE
    assert per_session == {"firing_rate_error": {}}


def test_comparability_records_the_metric_references():
    from ntt.evaluation.aggregation.comparability import (
        COMPARABILITY_ASPECTS,
        NO_INPUT,
        provenance_record,
    )

    assert "metric_references" in COMPARABILITY_ASPECTS

    def record(meta: dict) -> object:
        return provenance_record({"metadata": meta})["metric_references"]

    recon = {"inputs": {"recon": "results.json"}}
    references = {"firing_rate_error": POPULATION_MEAN_RATE}
    assert record({**recon, "metric_references": references}) == (
        "firing_rate_error=population_mean_rate"
    )
    assert record(recon) is None  # aggregated before the field: cannot be checked
    assert record({"inputs": {}}) == NO_INPUT


def test_a_nan_unit_saved_as_json_null_is_left_out_like_in_the_eval():
    # results.json saves a NaN as null. Unit 0 has no reconstructed rate: E = mean(0, 0, 2),
    # R = mean |[4, 6, 8] - 6| = 4/3, so the score is 1 - (2/3) / (4/3) = 0.5.
    segment = _firing_rate_result([2.0, 4.0, 6.0, 8.0], [2.0, 4.0, 6.0, 10.0])
    segment["raw_value"] = [None, 0.0, 0.0, 2.0]
    segment["metadata"]["rates_reconstructed_hz"] = [None, 4.0, 6.0, 10.0]
    comparisons, _ = with_alternative_floors(
        {"firing_rate_error": MetricComparison.from_dict(_poisson_comparison(model_value=2 / 3))},
        {"s": [{"firing_rate_error": segment}]},
        "mean",
    )
    assert comparisons["firing_rate_error"].normalized_score == pytest.approx(0.5)


def test_segments_and_sessions_without_a_finite_error_are_left_out_like_in_the_eval():
    report = _recon_report()
    # A segment without the metric, and a session where every unit's error is NaN.
    report["per_session_per_segment"]["b"].append({"results": {}})
    report["per_session_per_segment"]["c"] = [
        {"results": {"firing_rate_error": _firing_rate_result([1.0, 2.0], [np.nan, np.nan])}}
    ]
    comparisons, per_session = with_alternative_floors(
        _comparisons(report), _segment_results(report), "mean"
    )
    assert comparisons["firing_rate_error"].normalized_score == pytest.approx(1.0 - _E / _R)
    assert set(per_session["firing_rate_error"]) == {"a", "b"}


def test_normalized_scores_csv_names_the_reference():
    from ntt.evaluation.aggregation.results_dataframe import normalized_comparisons_to_dataframe

    report = _recon_report()
    comparisons, _ = with_alternative_floors(
        _comparisons(report), _segment_results(report), "mean"
    )
    kinds = normalized_comparisons_to_dataframe(comparisons).set_index("metric")["baseline_kind"]
    assert kinds.to_dict() == {"firing_rate_error": POPULATION_MEAN_RATE, "spike_f1": "poisson"}


def test_radar_leaves_firing_rate_error_out_of_normalized_scores():
    """The radar bundle holds aggregated reports only, without per-unit rates: it must not
    show firing_rate_error's Poisson-floor score."""
    from ntt.evaluation.core.types import EvaluationReport, MetricResult
    from ntt.evaluation.radar import CategorySpec, category_scores_for

    def report(firing_rate: float, spike_f1: float) -> EvaluationReport:
        results = {
            "firing_rate_error": _firing_rate_result([2.0], [2.0]) | {"raw_value": firing_rate},
            "spike_f1": {**_firing_rate_result([2.0], [2.0]), "name": "spike_f1"}
            | {"direction": "higher", "category": "temporal_fidelity", "raw_value": spike_f1},
        }
        return EvaluationReport(results={n: MetricResult.from_dict(d) for n, d in results.items()})

    spec = CategorySpec({"rates": ["firing_rate_error"], "timing": ["spike_f1"]})
    normalized = category_scores_for(report(0.29, 0.6), report(0.1, 0.0), report(0.0, 1.0), spec)
    assert np.isnan(normalized["rates"])
    assert normalized["timing"] == pytest.approx(0.6)
    raw = category_scores_for(
        report(0.29, 0.6), report(0.1, 0.0), report(0.0, 1.0), spec, normalize=False
    )
    assert raw["rates"] == pytest.approx(0.29)
