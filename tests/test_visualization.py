"""Smoke tests for evaluation comparison visualizations."""

import matplotlib.pyplot as plt

from ntt.evaluation.aggregation.comparison import MetricComparison
from ntt.evaluation.core.types import (
    EvaluationReport,
    MetricCategory,
    MetricDirection,
    MetricLevel,
    MetricResult,
)
from ntt.evaluation.visualization import (
    plot_metric_bars,
    plot_metric_dumbbell,
    plot_normalized_comparison_radar,
)


def _make_report(metric_name: str, value: float, direction: MetricDirection) -> EvaluationReport:
    result = MetricResult(
        name=metric_name,
        category=MetricCategory.SINGLE_UNIT_STATISTICS,
        direction=direction,
        level=MetricLevel.POPULATION,
        raw_value=value,
        metadata={},
        value_range=(-1.0, 1.0) if direction == MetricDirection.HIGHER_IS_BETTER else (0.0, None),
    )
    return EvaluationReport(results={metric_name: result}, category_scores={}, config={})


def test_plot_normalized_comparison_radar_smoke() -> None:
    comparisons = {
        "acg_correlation": MetricComparison(
            name="acg_correlation",
            category=MetricCategory.SINGLE_UNIT_STATISTICS,
            direction=MetricDirection.HIGHER_IS_BETTER,
            level=MetricLevel.POPULATION,
            model_value=0.2,
            baseline_value=-0.4,
            ceiling_value=1.0,
            normalized_score=0.43,
            value_range=(-1.0, 1.0),
        )
    }

    fig = plot_normalized_comparison_radar(comparisons)

    assert fig.axes
    plt.close(fig)


def test_plot_metric_dumbbell_smoke() -> None:
    model_report = _make_report("isi_divergence", 1.2, MetricDirection.LOWER_IS_BETTER)
    baseline_report = _make_report("isi_divergence", 2.5, MetricDirection.LOWER_IS_BETTER)

    fig = plot_metric_dumbbell(
        model_report,
        baseline_report,
        model_name="Model",
        baseline_name="Poisson",
    )

    assert fig.axes
    plt.close(fig)


def test_plot_metric_bars_with_missing_metrics_smoke() -> None:
    """Missing metrics should use NaN (not 0.0) and render without error."""
    report_a = _make_report("metric_a", 0.8, MetricDirection.HIGHER_IS_BETTER)
    report_b = _make_report("metric_b", 0.5, MetricDirection.LOWER_IS_BETTER)

    fig = plot_metric_bars({"Model A": report_a, "Model B": report_b})

    assert fig.axes
    plt.close(fig)
