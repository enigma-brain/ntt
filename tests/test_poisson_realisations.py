"""Tests for multi-realisation Poisson baseline averaging in NTTEvalJob."""

import numpy as np
import pytest

from ntt.evaluation.core.types import (
    DataType,
    EvaluationReport,
    MetricCategory,
    MetricDirection,
    MetricLevel,
    MetricResult,
)
from ntt.jobs.eval import NTTEvalJob


def _make_report(metric_values: dict[str, float | np.ndarray]) -> EvaluationReport:
    """Helper to build a minimal EvaluationReport from metric name → value."""
    results = {}
    for name, value in metric_values.items():
        results[name] = MetricResult(
            name=name,
            category=MetricCategory.TEMPORAL_FIDELITY,
            direction=MetricDirection.HIGHER_IS_BETTER,
            level=MetricLevel.POPULATION,
            raw_value=value,
        )
    return EvaluationReport(results=results)


class TestAverageRealisationReports:
    """Tests for NTTEvalJob._average_realisation_reports."""

    def test_single_report_returned_unchanged(self):
        """With one report, should return it as-is."""
        report = _make_report({"accuracy": 0.8})
        result = NTTEvalJob._average_realisation_reports([report], MetricResult)
        assert result is report

    def test_scalar_mean_and_std(self):
        """Mean and std computed correctly for scalar metrics."""
        reports = [
            _make_report({"accuracy": 0.7}),
            _make_report({"accuracy": 0.8}),
            _make_report({"accuracy": 0.9}),
        ]
        result = NTTEvalJob._average_realisation_reports(reports, MetricResult)

        acc = result.results["accuracy"]
        assert acc.raw_value == pytest.approx(0.8, abs=1e-10)
        assert acc.metadata["realisation_std"] == pytest.approx(np.std([0.7, 0.8, 0.9]), abs=1e-10)
        assert acc.metadata["n_realisations"] == 3

    def test_array_mean_and_std(self):
        """Mean and std computed correctly for per-unit array metrics."""
        reports = [
            _make_report({"per_unit": np.array([1.0, 2.0, 3.0])}),
            _make_report({"per_unit": np.array([2.0, 3.0, 4.0])}),
        ]
        result = NTTEvalJob._average_realisation_reports(reports, MetricResult)

        pu = result.results["per_unit"]
        np.testing.assert_allclose(pu.raw_value, [1.5, 2.5, 3.5])
        np.testing.assert_allclose(pu.metadata["realisation_std"], [0.5, 0.5, 0.5])
        assert pu.metadata["n_realisations"] == 2

    def test_multiple_metrics_preserved(self):
        """All metrics from the base report are averaged."""
        reports = [
            _make_report({"a": 1.0, "b": 2.0}),
            _make_report({"a": 3.0, "b": 4.0}),
        ]
        result = NTTEvalJob._average_realisation_reports(reports, MetricResult)
        assert set(result.results.keys()) == {"a", "b"}
        assert result.results["a"].raw_value == pytest.approx(2.0)
        assert result.results["b"].raw_value == pytest.approx(3.0)

    def test_metadata_fields_preserved(self):
        """Original metadata fields from base report are preserved."""
        r = MetricResult(
            name="x",
            category=MetricCategory.SINGLE_UNIT_STATISTICS,
            direction=MetricDirection.LOWER_IS_BETTER,
            level=MetricLevel.PER_UNIT,
            raw_value=1.0,
            metadata={"existing_key": "hello"},
        )
        report1 = EvaluationReport(results={"x": r})
        r2 = MetricResult(
            name="x",
            category=MetricCategory.SINGLE_UNIT_STATISTICS,
            direction=MetricDirection.LOWER_IS_BETTER,
            level=MetricLevel.PER_UNIT,
            raw_value=3.0,
        )
        report2 = EvaluationReport(results={"x": r2})

        result = NTTEvalJob._average_realisation_reports([report1, report2], MetricResult)
        assert result.results["x"].metadata["existing_key"] == "hello"
        assert "realisation_std" in result.results["x"].metadata

    def test_report_level_fields_preserved(self):
        """Report-level fields (skipped_metrics, config, etc.) come from base."""
        report1 = EvaluationReport(
            results={},
            skipped_metrics={"bad_metric": "not enough data"},
            config={"some": "config"},
            original_data_type=DataType.SPIKES,
        )
        report2 = EvaluationReport(results={})

        result = NTTEvalJob._average_realisation_reports([report1, report2], MetricResult)
        assert result.skipped_metrics == {"bad_metric": "not enough data"}
        assert result.config == {"some": "config"}
        assert result.original_data_type == DataType.SPIKES
