"""Tests for normalized floor/ceiling metric comparisons."""

import numpy as np

from ntt.evaluation.aggregation.comparison import (
    compute_normalized_comparisons,
    extract_normalized_scores,
)
from ntt.evaluation.core.types import (
    EvaluationReport,
    MetricCategory,
    MetricDirection,
    MetricLevel,
    MetricResult,
)
from ntt.jobs.eval import NTTEvalJob


def _make_result(
    value: float | np.ndarray,
    direction: MetricDirection,
    *,
    category: MetricCategory = MetricCategory.SINGLE_UNIT_STATISTICS,
    value_range: tuple[float | None, float | None] = (None, None),
) -> MetricResult:
    return MetricResult(
        name="test_metric",
        raw_value=value,
        direction=direction,
        category=category,
        level=MetricLevel.POPULATION,
        metadata={},
        value_range=value_range,
    )


def _make_report(results: dict[str, MetricResult]) -> EvaluationReport:
    return EvaluationReport(results=results, category_scores={}, config={})


class TestComputeNormalizedComparisons:
    def test_higher_is_better_uses_baseline_zero_and_ceiling_one(self):
        model = _make_report({"corr": _make_result(0.4, MetricDirection.HIGHER_IS_BETTER)})
        baseline = _make_report({"corr": _make_result(0.0, MetricDirection.HIGHER_IS_BETTER)})
        ceiling = _make_report({"corr": _make_result(1.0, MetricDirection.HIGHER_IS_BETTER)})

        comparisons = compute_normalized_comparisons(model, baseline, ceiling)

        assert np.isclose(comparisons["corr"].normalized_score, 0.4)

    def test_lower_is_better_uses_baseline_zero_and_ceiling_one(self):
        model = _make_report({"mse": _make_result(0.6, MetricDirection.LOWER_IS_BETTER)})
        baseline = _make_report({"mse": _make_result(1.0, MetricDirection.LOWER_IS_BETTER)})
        ceiling = _make_report({"mse": _make_result(0.0, MetricDirection.LOWER_IS_BETTER)})

        comparisons = compute_normalized_comparisons(model, baseline, ceiling)

        assert np.isclose(comparisons["mse"].normalized_score, 0.4)

    def test_signed_metric_with_negative_poisson_floor(self):
        model = _make_report(
            {"acg": _make_result(0.2, MetricDirection.HIGHER_IS_BETTER, value_range=(-1.0, 1.0))}
        )
        baseline = _make_report(
            {"acg": _make_result(-0.4, MetricDirection.HIGHER_IS_BETTER, value_range=(-1.0, 1.0))}
        )
        ceiling = _make_report(
            {"acg": _make_result(1.0, MetricDirection.HIGHER_IS_BETTER, value_range=(-1.0, 1.0))}
        )

        comparisons = compute_normalized_comparisons(model, baseline, ceiling)

        assert np.isclose(comparisons["acg"].normalized_score, 0.6 / 1.4)

    def test_values_below_floor_and_above_ceiling_are_preserved(self):
        model_worse = _make_report({"corr": _make_result(-0.2, MetricDirection.HIGHER_IS_BETTER)})
        model_better = _make_report({"corr": _make_result(1.2, MetricDirection.HIGHER_IS_BETTER)})
        baseline = _make_report({"corr": _make_result(0.0, MetricDirection.HIGHER_IS_BETTER)})
        ceiling = _make_report({"corr": _make_result(1.0, MetricDirection.HIGHER_IS_BETTER)})

        worse = compute_normalized_comparisons(model_worse, baseline, ceiling)
        better = compute_normalized_comparisons(model_better, baseline, ceiling)

        assert worse["corr"].normalized_score < 0.0
        assert better["corr"].normalized_score > 1.0

    def test_unbounded_lower_is_better_metric_uses_empirical_ceiling(self):
        model = _make_report({"isi": _make_result(2.0, MetricDirection.LOWER_IS_BETTER)})
        baseline = _make_report({"isi": _make_result(5.0, MetricDirection.LOWER_IS_BETTER)})
        ceiling = _make_report({"isi": _make_result(0.5, MetricDirection.LOWER_IS_BETTER)})

        comparisons = compute_normalized_comparisons(model, baseline, ceiling)

        assert np.isclose(comparisons["isi"].normalized_score, (5.0 - 2.0) / (5.0 - 0.5))

    def test_unbounded_higher_is_better_metric_uses_empirical_ceiling(self):
        model = _make_report({"mi": _make_result(3.0, MetricDirection.HIGHER_IS_BETTER)})
        baseline = _make_report({"mi": _make_result(1.0, MetricDirection.HIGHER_IS_BETTER)})
        ceiling = _make_report({"mi": _make_result(5.0, MetricDirection.HIGHER_IS_BETTER)})

        comparisons = compute_normalized_comparisons(model, baseline, ceiling)

        assert np.isclose(comparisons["mi"].normalized_score, (3.0 - 1.0) / (5.0 - 1.0))

    def test_degenerate_floor_and_ceiling_marked_invalid(self):
        model = _make_report({"corr": _make_result(0.5, MetricDirection.HIGHER_IS_BETTER)})
        baseline = _make_report({"corr": _make_result(0.2, MetricDirection.HIGHER_IS_BETTER)})
        ceiling = _make_report({"corr": _make_result(0.2, MetricDirection.HIGHER_IS_BETTER)})

        comparisons = compute_normalized_comparisons(model, baseline, ceiling)

        assert comparisons["corr"].normalized_score is None
        assert comparisons["corr"].reason == "degenerate_floor_ceiling"

    def test_insufficient_baseline_separation_marked_invalid(self):
        # Floor↔ceiling gap of 0.02 < MIN_SEPARATION (0.1) on a *bounded* axis: the
        # skill score would amplify (model - floor) by 50x and be dominated by floor
        # noise, so the metric is gated out (e.g. population_coupling at >=5 ms).
        vr = (-1.0, 1.0)
        model = _make_report(
            {"corr": _make_result(0.95, MetricDirection.HIGHER_IS_BETTER, value_range=vr)}
        )
        baseline = _make_report(
            {"corr": _make_result(0.98, MetricDirection.HIGHER_IS_BETTER, value_range=vr)}
        )
        ceiling = _make_report(
            {"corr": _make_result(1.0, MetricDirection.HIGHER_IS_BETTER, value_range=vr)}
        )

        comparisons = compute_normalized_comparisons(model, baseline, ceiling)

        assert comparisons["corr"].normalized_score is None
        assert comparisons["corr"].reason == "insufficient_baseline_separation"

    def test_separation_gate_skipped_for_unbounded_metric(self):
        # MIN_SEPARATION is only meaningful for bounded, correlation-like axes. An
        # open-ended metric (value_range upper bound None) with a small absolute
        # floor↔ceiling gap must NOT be gated; it normalizes through the usual path.
        vr = (0.0, None)
        model = _make_report(
            {"pr": _make_result(0.96, MetricDirection.HIGHER_IS_BETTER, value_range=vr)}
        )
        baseline = _make_report(
            {"pr": _make_result(0.98, MetricDirection.HIGHER_IS_BETTER, value_range=vr)}
        )
        ceiling = _make_report(
            {"pr": _make_result(1.0, MetricDirection.HIGHER_IS_BETTER, value_range=vr)}
        )

        comparisons = compute_normalized_comparisons(model, baseline, ceiling)

        assert comparisons["pr"].normalized_score is not None
        assert np.isclose(comparisons["pr"].normalized_score, (0.96 - 0.98) / (1.0 - 0.98))

    def test_gated_bounded_axis_is_logged(self, caplog):
        import logging

        vr = (-1.0, 1.0)
        model = _make_report(
            {"corr": _make_result(0.95, MetricDirection.HIGHER_IS_BETTER, value_range=vr)}
        )
        baseline = _make_report(
            {"corr": _make_result(0.98, MetricDirection.HIGHER_IS_BETTER, value_range=vr)}
        )
        ceiling = _make_report(
            {"corr": _make_result(1.0, MetricDirection.HIGHER_IS_BETTER, value_range=vr)}
        )

        with caplog.at_level(logging.WARNING, logger="ntt.evaluation.aggregation.comparison"):
            compute_normalized_comparisons(model, baseline, ceiling)

        assert "Gating metric" in caplog.text
        assert "'corr'" in caplog.text

    def test_separation_above_threshold_is_normalized(self):
        # Gap of 0.15 (> MIN_SEPARATION) normalizes as usual.
        model = _make_report({"corr": _make_result(0.3, MetricDirection.HIGHER_IS_BETTER)})
        baseline = _make_report({"corr": _make_result(0.2, MetricDirection.HIGHER_IS_BETTER)})
        ceiling = _make_report({"corr": _make_result(0.35, MetricDirection.HIGHER_IS_BETTER)})

        comparisons = compute_normalized_comparisons(model, baseline, ceiling)

        assert comparisons["corr"].normalized_score is not None
        assert np.isclose(comparisons["corr"].normalized_score, (0.3 - 0.2) / (0.35 - 0.2))

    def test_non_finite_values_marked_invalid(self):
        model = _make_report(
            {"corr": _make_result(float("nan"), MetricDirection.HIGHER_IS_BETTER)}
        )
        baseline = _make_report({"corr": _make_result(0.2, MetricDirection.HIGHER_IS_BETTER)})
        ceiling = _make_report({"corr": _make_result(1.0, MetricDirection.HIGHER_IS_BETTER)})

        comparisons = compute_normalized_comparisons(model, baseline, ceiling)

        assert comparisons["corr"].normalized_score is None
        assert comparisons["corr"].reason == "non_finite_value"

    def test_array_valued_metrics_are_reduced_by_nanmean(self):
        model = _make_report(
            {"f1": _make_result(np.array([0.8, 0.9, 0.7]), MetricDirection.HIGHER_IS_BETTER)}
        )
        baseline = _make_report(
            {"f1": _make_result(np.array([0.2, 0.3, 0.1]), MetricDirection.HIGHER_IS_BETTER)}
        )
        ceiling = _make_report(
            {"f1": _make_result(np.array([1.0, 1.0, 1.0]), MetricDirection.HIGHER_IS_BETTER)}
        )

        comparisons = compute_normalized_comparisons(model, baseline, ceiling)

        assert np.isclose(comparisons["f1"].normalized_score, (0.8 - 0.2) / (1.0 - 0.2))

    def test_only_shared_metrics_are_compared(self):
        model = _make_report(
            {
                "corr": _make_result(0.8, MetricDirection.HIGHER_IS_BETTER),
                "only_in_model": _make_result(0.1, MetricDirection.HIGHER_IS_BETTER),
            }
        )
        baseline = _make_report({"corr": _make_result(0.2, MetricDirection.HIGHER_IS_BETTER)})
        ceiling = _make_report({"corr": _make_result(1.0, MetricDirection.HIGHER_IS_BETTER)})

        comparisons = compute_normalized_comparisons(model, baseline, ceiling)

        assert set(comparisons) == {"corr"}

    def test_extract_normalized_scores_filters_invalid_metrics(self):
        model = _make_report(
            {
                "valid": _make_result(0.5, MetricDirection.HIGHER_IS_BETTER),
                "invalid": _make_result(0.5, MetricDirection.HIGHER_IS_BETTER),
            }
        )
        baseline = _make_report(
            {
                "valid": _make_result(0.0, MetricDirection.HIGHER_IS_BETTER),
                "invalid": _make_result(0.3, MetricDirection.HIGHER_IS_BETTER),
            }
        )
        ceiling = _make_report(
            {
                "valid": _make_result(1.0, MetricDirection.HIGHER_IS_BETTER),
                "invalid": _make_result(0.3, MetricDirection.HIGHER_IS_BETTER),
            }
        )

        comparisons = compute_normalized_comparisons(model, baseline, ceiling)
        scores = extract_normalized_scores(comparisons)

        assert scores == {"valid": 0.5}


class TestKnownPerfectCeilings:
    def test_injects_structure_and_downstream_ceiling_metrics(self):
        model = _make_report(
            {
                "population_structure_preservation": _make_result(
                    0.4,
                    MetricDirection.HIGHER_IS_BETTER,
                    category=MetricCategory.POPULATION_STRUCTURE,
                    value_range=(-1.0, 1.0),
                ),
                "behavioral_regression_cursor_pos": _make_result(
                    0.6,
                    MetricDirection.HIGHER_IS_BETTER,
                    category=MetricCategory.DOWNSTREAM_DECODING,
                    value_range=(None, None),
                ),
            }
        )
        perfect = _make_report({})

        updated = NTTEvalJob._inject_known_perfect_ceiling_metrics(perfect, model)

        assert updated.results["population_structure_preservation"].raw_value == 1.0
        assert updated.results["behavioral_regression_cursor_pos"].raw_value == 1.0
        assert updated.results["population_structure_preservation"].metadata["synthetic_ceiling"]
        assert updated.results["behavioral_regression_cursor_pos"].metadata["synthetic_ceiling"]

    def test_keeps_existing_ceiling_metric_unchanged(self):
        existing = _make_result(
            0.95,
            MetricDirection.HIGHER_IS_BETTER,
            category=MetricCategory.DOWNSTREAM_DECODING,
        )
        perfect = _make_report({"behavioral_regression_cursor_pos": existing})
        model = _make_report(
            {
                "behavioral_regression_cursor_pos": _make_result(
                    0.6,
                    MetricDirection.HIGHER_IS_BETTER,
                    category=MetricCategory.DOWNSTREAM_DECODING,
                )
            }
        )

        updated = NTTEvalJob._inject_known_perfect_ceiling_metrics(perfect, model)

        assert updated.results["behavioral_regression_cursor_pos"].raw_value == 0.95

    def test_injects_correlogram_ceiling_metrics(self):
        """Correlogram metrics excluded from lightweight pipeline should be injected."""
        model = _make_report(
            {
                "acg_correlation": _make_result(
                    0.6,
                    MetricDirection.HIGHER_IS_BETTER,
                    category=MetricCategory.SINGLE_UNIT_STATISTICS,
                    value_range=(-1.0, 1.0),
                ),
                "ccg_correlation": _make_result(
                    0.4,
                    MetricDirection.HIGHER_IS_BETTER,
                    category=MetricCategory.POPULATION_STRUCTURE,
                    value_range=(-1.0, 1.0),
                ),
                "connectivity_f1": _make_result(
                    0.7,
                    MetricDirection.HIGHER_IS_BETTER,
                    category=MetricCategory.POPULATION_STRUCTURE,
                    value_range=(0.0, 1.0),
                ),
            }
        )
        perfect = _make_report({})

        updated = NTTEvalJob._inject_known_perfect_ceiling_metrics(perfect, model)

        assert updated.results["acg_correlation"].raw_value == 1.0
        assert updated.results["ccg_correlation"].raw_value == 1.0
        assert updated.results["connectivity_f1"].raw_value == 1.0
        for name in ("acg_correlation", "ccg_correlation", "connectivity_f1"):
            assert updated.results[name].metadata["synthetic_ceiling"]

    def test_injects_lower_is_better_ceiling_with_zero(self):
        """Lower-is-better known-perfect metrics should get ceiling=0.0."""
        model = _make_report(
            {
                "behavioral_regression_mse": _make_result(
                    0.3,
                    MetricDirection.LOWER_IS_BETTER,
                    category=MetricCategory.DOWNSTREAM_DECODING,
                ),
            }
        )
        perfect = _make_report({})

        updated = NTTEvalJob._inject_known_perfect_ceiling_metrics(perfect, model)

        assert "behavioral_regression_mse" in updated.results
        assert updated.results["behavioral_regression_mse"].raw_value == 0.0
        assert updated.results["behavioral_regression_mse"].metadata["synthetic_ceiling"]
