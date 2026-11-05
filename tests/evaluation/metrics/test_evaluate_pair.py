"""Contract tests for :func:`ntt.evaluation.metrics.robustness_latent.evaluate_pair`.

``evaluate_pair`` takes ``(display_name, metric)`` pairs — the same shape as
:attr:`EvaluationPipeline.metrics` — and keys its results by display name.
Keying by ``metric.name`` instead silently collapses configs that instantiate
one metric class several times (``acg_correlation`` / ``acg_correlation_5ms``
/ ``acg_correlation_10ms`` in ``config/evaluation/metrics/robustness_latent.yaml``).
"""

import numpy as np
import pytest

from ntt.evaluation.config import DEFAULT_CONFIG
from ntt.evaluation.core.types import (
    DataType,
    EvalData,
    MetricCategory,
    MetricDirection,
    MetricLevel,
)
from ntt.evaluation.metrics.base import BaseMetric, IntermediateRequest
from ntt.evaluation.metrics.robustness_latent import evaluate_pair


class _ScaledMeanRate(BaseMetric):
    """One class, parameterised — the shape whose variants must not collapse."""

    name = "scaled_mean_rate"
    category = MetricCategory.SINGLE_UNIT_STATISTICS
    direction = MetricDirection.HIGHER_IS_BETTER
    level = MetricLevel.POPULATION
    _required_intermediates = []
    _default_params = {"scale": 1.0}
    supported_data_types = {DataType.RATES}
    preferred_data_type = DataType.RATES

    def get_intermediate_requests(self, data_type: DataType) -> list[IntermediateRequest]:
        return [IntermediateRequest("firing_rates_reconstructed")]

    def compute(self, intermediates):
        value = float(np.mean(intermediates["firing_rates_reconstructed"])) * self.params["scale"]
        return self._make_result(value)


class _AlwaysFails(BaseMetric):
    name = "always_fails"
    category = MetricCategory.SINGLE_UNIT_STATISTICS
    direction = MetricDirection.HIGHER_IS_BETTER
    level = MetricLevel.POPULATION
    _required_intermediates = []
    supported_data_types = {DataType.RATES}
    preferred_data_type = DataType.RATES

    def compute(self, intermediates):
        raise RuntimeError("boom")


def _rates(value: float) -> EvalData:
    return EvalData(
        rates_hz=np.full((3, 8), value, dtype=np.float64),
        bin_size_seconds=0.001,
        data_type=DataType.RATES,
    )


def test_evaluate_pair_keys_results_by_display_name_not_metric_class():
    results = evaluate_pair(
        clean=_rates(1.0),
        noisy=_rates(2.0),
        metrics=[
            ("scaled_mean_rate", _ScaledMeanRate(scale=1.0)),
            ("scaled_mean_rate_x10", _ScaledMeanRate(scale=10.0)),
        ],
        config=DEFAULT_CONFIG,
    )

    assert set(results) == {"scaled_mean_rate", "scaled_mean_rate_x10"}
    assert results["scaled_mean_rate"]["status"] == "ok"
    assert results["scaled_mean_rate_x10"]["status"] == "ok"
    assert results["scaled_mean_rate"]["result"].raw_value == 2.0
    assert results["scaled_mean_rate_x10"]["result"].raw_value == 20.0


def test_evaluate_pair_reports_failures_under_the_display_name():
    results = evaluate_pair(
        clean=_rates(1.0),
        noisy=_rates(2.0),
        metrics=[("renamed_failure", _AlwaysFails())],
        config=DEFAULT_CONFIG,
    )

    assert set(results) == {"renamed_failure"}
    assert results["renamed_failure"]["status"] == "skipped"
    assert "boom" in results["renamed_failure"]["reason"]


def test_evaluate_pair_refuses_duplicate_display_names():
    with pytest.raises(ValueError, match="duplicate display names"):
        evaluate_pair(
            clean=_rates(1.0),
            noisy=_rates(2.0),
            metrics=[
                ("scaled_mean_rate", _ScaledMeanRate(scale=1.0)),
                ("scaled_mean_rate", _ScaledMeanRate(scale=10.0)),
            ],
            config=DEFAULT_CONFIG,
        )
