"""Evaluation metrics."""

# Import all metric modules to trigger registration
from . import (
    compression,  # noqa: F401
    downstream_prediction,  # noqa: F401
    information,  # noqa: F401
    pairwise,  # noqa: F401
    population,  # noqa: F401
    repeats,  # noqa: F401
    repeats_baseline,  # noqa: F401
    robustness_input,  # noqa: F401
    single_unit,  # noqa: F401
    spike_detection,  # noqa: F401
    structure_preservation,  # noqa: F401
)
from .base import (
    BaseMetric,
    IntermediateRequest,
    get_all_metrics,
    get_metric_names,
    get_metrics_by_category,
    get_metrics_by_names,
    register_metric,
)

__all__ = [
    "BaseMetric",
    "IntermediateRequest",
    "register_metric",
    "get_all_metrics",
    "get_metric_names",
    "get_metrics_by_category",
    "get_metrics_by_names",
]
