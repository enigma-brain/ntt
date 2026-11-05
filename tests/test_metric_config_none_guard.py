"""Test that instantiate_metrics_from_config skips None entries."""

from ntt.evaluation.config.metric_config import instantiate_metrics_from_config


def test_none_metric_entry_is_skipped():
    """Setting a metric to null in Hydra should skip it, not crash."""
    metrics_dict = {
        "spike_detection_f1": {
            "metric_class": "spike_detection_f1",
            "spike_match_tolerance_ms": 1.0,
        },
        "disabled_metric": None,
    }
    result = instantiate_metrics_from_config(metrics_dict)
    assert len(result) == 1
    assert result[0][0] == "spike_detection_f1"
