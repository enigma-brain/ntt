"""Tests for the latent_robustness (M2) metric capability contract on ``BaseMetric``."""

from __future__ import annotations

import pytest

from ntt.evaluation.core.types import MetricCategory, MetricDirection, MetricLevel
from ntt.evaluation.metrics.base import BaseMetric


class _ConcreteMetric(BaseMetric):
    name = "test_metric_capability_flag_default"
    category = MetricCategory.TEMPORAL_FIDELITY
    direction = MetricDirection.HIGHER_IS_BETTER
    level = MetricLevel.POPULATION
    _required_intermediates = []

    def compute(self, intermediates):
        return self._make_result(0.0)


def test_aggregation_fallback_policy_defaults_to_contiguous_only() -> None:
    metric = _ConcreteMetric()

    assert _ConcreteMetric.aggregation_fallback_policy == "contiguous_only"
    assert metric.aggregation_fallback_policy == "contiguous_only"


def test_aggregation_fallback_policy_allows_per_window_override() -> None:
    class _PerWindowMetric(_ConcreteMetric):
        name = "test_metric_capability_flag_per_window"
        aggregation_fallback_policy = "per_window"

    metric = _PerWindowMetric()

    assert _PerWindowMetric.aggregation_fallback_policy == "per_window"
    assert metric.aggregation_fallback_policy == "per_window"


def test_invalid_aggregation_fallback_policy_is_rejected() -> None:
    with pytest.raises(TypeError, match="aggregation_fallback_policy"):

        class _InvalidPolicyMetric(_ConcreteMetric):
            name = "test_metric_capability_flag_invalid_policy"
            aggregation_fallback_policy = "session_aggregate"


def test_session_aggregated_requires_accumulate_window_hook() -> None:
    with pytest.raises(TypeError, match="_accumulate_window"):

        class _MissingAccumulateMetric(_ConcreteMetric):
            name = "test_metric_capability_flag_missing_accumulate"
            aggregation_fallback_policy = "session_aggregated"

            def _finalize_session_state(self, session_state):
                return self._make_result(0.0)


def test_session_aggregated_requires_finalize_session_state_hook() -> None:
    with pytest.raises(TypeError, match="_finalize_session_state"):

        class _MissingFinalizeMetric(_ConcreteMetric):
            name = "test_metric_capability_flag_missing_finalize"
            aggregation_fallback_policy = "session_aggregated"

            def _accumulate_window(self, session_state, window_intermediates):
                return None


def test_overriding_session_hooks_requires_session_aggregated_policy() -> None:
    with pytest.raises(TypeError, match="session_aggregated"):

        class _HookOverrideWithoutPolicyMetric(_ConcreteMetric):
            name = "test_metric_capability_flag_hook_override_without_policy"

            def _accumulate_window(self, session_state, window_intermediates):
                return None

            def _finalize_session_state(self, session_state):
                return self._make_result(0.0)


def test_consistent_session_aggregated_subclass_is_allowed() -> None:
    class _SessionAggregatedMetric(_ConcreteMetric):
        name = "test_metric_capability_flag_session_aggregated"
        aggregation_fallback_policy = "session_aggregated"

        def _accumulate_window(self, session_state, window_intermediates):
            return None

        def _finalize_session_state(self, session_state):
            return self._make_result(0.0)

    metric = _SessionAggregatedMetric()

    assert _SessionAggregatedMetric.aggregation_fallback_policy == "session_aggregated"
    assert metric.aggregation_fallback_policy == "session_aggregated"
