"""Tests for limit_to_datasets per-metric dataset filtering."""

import numpy as np
import torch

from ntt.evaluation.config.metric_config import (
    MetricInstanceConfig,
    instantiate_metrics_from_config,
)
from ntt.evaluation.core.types import DataType, EvalData
from ntt.evaluation.metrics.base import BaseMetric
from ntt.evaluation.pipeline import EvaluationPipeline


class TestLimitToDatasets:
    """Core tests for the limit_to_datasets filtering feature."""

    def test_metric_skipped_for_non_matching_session(self):
        """A metric with limit_to_datasets is skipped when session doesn't match."""
        # Use spike_detection_f1 as a concrete metric that's always available
        metrics_dict = {
            "test_metric": {
                "metric_class": "spike_detection_f1",
                "limit_to_datasets": ["odoherty"],
            }
        }
        metrics = instantiate_metrics_from_config(metrics_dict)
        assert len(metrics) == 1
        display_name, metric = metrics[0]

        # Verify limit_to_datasets was stored on the instance
        assert metric.limit_to_datasets == ["odoherty"]

        # Should NOT run for a perich session
        assert not metric.should_run_for_session(
            "perich_miller_population_2018/c_20131022_center_out_reaching"
        )

        # Should run for an odoherty session
        assert metric.should_run_for_session("odoherty_sabes_nonhuman_2017/indy_20160407_02")

    def test_metric_runs_for_all_sessions_when_no_filter(self):
        """A metric without limit_to_datasets runs for any session."""
        metrics_dict = {
            "test_metric": {
                "metric_class": "spike_detection_f1",
            }
        }
        metrics = instantiate_metrics_from_config(metrics_dict)
        _, metric = metrics[0]

        assert metric.limit_to_datasets is None
        assert metric.should_run_for_session("anything/at_all")

    def test_pipeline_skips_metric_for_non_matching_session(self):
        """Pipeline.run() skips metrics when session_id doesn't match limit_to_datasets."""
        metrics_dict = {
            "odoherty_only": {
                "metric_class": "spike_detection_f1",
                "limit_to_datasets": ["odoherty"],
            },
            "all_datasets": {
                "metric_class": "spike_detection_f1",
                "spike_match_tolerance_ms": 0.5,
            },
        }
        metric_pairs = instantiate_metrics_from_config(metrics_dict)

        pipeline = EvaluationPipeline(metrics=metric_pairs)

        # Create sparse spike data (density must be <50% to avoid
        # SPIKE_TIMES conversion rejection)
        n_units, n_time = 5, 100
        spikes = (torch.rand(n_units, n_time) < 0.1).float()
        original = EvalData.from_tensor(
            tensor=spikes, data_type=DataType.SPIKES, bin_size_seconds=0.001
        )
        reconstructed = EvalData.from_tensor(
            tensor=spikes, data_type=DataType.SPIKES, bin_size_seconds=0.001
        )

        # Run with a perich session — odoherty_only metric should be skipped
        report = pipeline.run(
            original,
            reconstructed,
            session_id="perich_miller_population_2018/c_20131022_center_out_reaching",
        )

        assert "odoherty_only" in report.skipped_metrics
        assert "limit_to_datasets" in report.skipped_metrics["odoherty_only"]
        assert "all_datasets" in report.results

    def test_should_run_for_session_substring_match(self):
        """limit_to_datasets uses substring matching, not exact match."""
        metric = BaseMetric.__subclasses__()[0]()  # any concrete metric
        metric.limit_to_datasets = ["odoherty_sabes"]

        # Full dataset name is a substring of session_id
        assert metric.should_run_for_session("odoherty_sabes_nonhuman_2017/indy_20160407_02")
        # Partial match also works
        metric.limit_to_datasets = ["odoherty"]
        assert metric.should_run_for_session("odoherty_sabes_nonhuman_2017/indy_20160407_02")
        # No match
        assert not metric.should_run_for_session("perich_miller_population_2018/c_20131022")
