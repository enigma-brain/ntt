"""Tests for plot_all utility functions."""

import numpy as np

from ntt.evaluation.core.types import (
    EvaluationReport,
    MetricCategory,
    MetricDirection,
    MetricLevel,
    MetricResult,
)
from ntt.evaluation.plot_all import _find_first_valid_result


def _make_result(value, name="test_metric"):
    return MetricResult(
        name=name,
        category=MetricCategory.SINGLE_UNIT_STATISTICS,
        direction=MetricDirection.HIGHER_IS_BETTER,
        level=MetricLevel.PER_UNIT,
        raw_value=value,
        metadata={},
    )


def _make_report(results_dict):
    return EvaluationReport(results=results_dict, config={})


class TestFindFirstValidResult:
    def test_returns_scalar_result(self):
        report = _make_report({"m": _make_result(0.5)})
        result = _find_first_valid_result([report], "m")
        assert result is not None
        assert result.raw_value == 0.5

    def test_skips_nan_scalar(self):
        nan_report = _make_report({"m": _make_result(float("nan"))})
        ok_report = _make_report({"m": _make_result(0.5)})
        result = _find_first_valid_result([nan_report, ok_report], "m")
        assert result is not None
        assert result.raw_value == 0.5

    def test_returns_none_when_missing(self):
        report = _make_report({"other": _make_result(0.5)})
        result = _find_first_valid_result([report], "m")
        assert result is None

    def test_handles_array_valued_metric_without_crash(self):
        """Array-valued raw_value should not raise ValueError."""
        arr = np.array([0.8, 0.9, float("nan"), 0.7])
        report = _make_report({"m": _make_result(arr)})
        result = _find_first_valid_result([report], "m")
        assert result is not None

    def test_skips_all_nan_array(self):
        """Array of all NaNs should be skipped like a scalar NaN."""
        all_nan = np.array([float("nan"), float("nan")])
        ok_report = _make_report({"m": _make_result(0.5)})
        nan_report = _make_report({"m": _make_result(all_nan)})
        result = _find_first_valid_result([nan_report, ok_report], "m")
        assert result is not None
        assert result.raw_value == 0.5
