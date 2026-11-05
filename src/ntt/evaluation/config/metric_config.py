"""Per-metric configuration and instantiation."""

from typing import Any

from pydantic import BaseModel

from ..metrics.base import METRIC_REGISTRY, BaseMetric

# Reserved metric names whose real ``BaseMetric`` subclass does not exist
# yet. Kept out of ``METRIC_REGISTRY`` so ``get_all_metrics()`` and friends
# don't try to instantiate them; but ``instantiate_metrics_from_config``
# below checks this set so such names are accepted in configs and produce
# a clear ``NotImplementedError`` on actual use (fail loud).
_RESERVED_METRIC_NAMES: set[str] = set()


class MetricInstanceConfig(BaseModel):
    """Configuration for a single metric instance.

    In YAML, each entry under the ``metrics`` dict maps a display name
    to a ``MetricInstanceConfig``. The ``metric_class`` field references a
    registered metric name from ``METRIC_REGISTRY``. All other fields
    are forwarded as keyword arguments to the metric constructor.

    Example YAML::

        metrics:
          spike_detection_f1:
            metric_class: spike_detection_f1
            spike_match_tolerance_ms: 1.0

          f1_strict:
            metric_class: spike_detection_f1
            spike_match_tolerance_ms: 0.5
    """

    metric_class: str  # registered name in METRIC_REGISTRY

    # Optional dataset filter.  When set, the metric is only computed for
    # sessions whose session_id contains at least one of these strings
    # (substring match).  Example: ["odoherty_sabes_nonhuman_2017"] will
    # match session_id "odoherty_sabes_nonhuman_2017/indy_20160407_02".
    limit_to_datasets: list[str] | None = None

    class Config:
        extra = "allow"  # remaining fields are metric params


def instantiate_metrics_from_config(
    metrics_dict: dict[str, Any],
) -> list[tuple[str, BaseMetric]]:
    """Create (display_name, metric_instance) pairs from a config dict.

    Args:
        metrics_dict: Maps display_name -> dict with at least ``metric_class``
            key plus optional parameter overrides.

    Returns:
        List of (display_name, metric_instance) tuples.

    Raises:
        ValueError: If a ``metric_class`` is not found in the registry or
            if unknown parameters are provided.
    """
    result: list[tuple[str, BaseMetric]] = []

    for display_name, raw_cfg in metrics_dict.items():
        if raw_cfg is None:
            continue
        # Accept both raw dicts (from Hydra/OmegaConf) and MetricInstanceConfig
        if isinstance(raw_cfg, MetricInstanceConfig):
            cfg = raw_cfg
        else:
            cfg = MetricInstanceConfig(**raw_cfg)

        metric_cls = METRIC_REGISTRY.get(cfg.metric_class)
        if metric_cls is None:
            # Names reserved for metrics not yet implemented: accepted at
            # config-load time, raise loudly when actually used.
            if cfg.metric_class in _RESERVED_METRIC_NAMES:
                raise NotImplementedError(
                    f"Metric '{cfg.metric_class}' (display name "
                    f"'{display_name}') is reserved but its real "
                    f"implementation has not landed yet."
                )
            available = ", ".join(sorted(METRIC_REGISTRY.keys()))
            raise ValueError(
                f"Unknown metric_class '{cfg.metric_class}' for display name "
                f"'{display_name}'. Available: {available}"
            )

        # Collect metric params (all fields except metric_class and
        # limit_to_datasets, which is handled separately).
        _non_metric_keys = {"metric_class", "limit_to_datasets"}
        params = {k: v for k, v in cfg.model_dump().items() if k not in _non_metric_keys}

        instance = metric_cls(**params)
        instance.limit_to_datasets = cfg.limit_to_datasets
        result.append((display_name, instance))

    return result
