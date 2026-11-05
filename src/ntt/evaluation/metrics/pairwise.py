"""Pairwise Interactions metrics.

CCG-based pairwise metrics support DataType.RATES and coarse-binned SPIKES via
discrete cross-correlation of binned counts. When spike times are available,
the metrics use the spike-time correlogram path instead.
"""

import warnings
from typing import Any

import numpy as np

from ..core.types import (
    DataType,
    MetricCategory,
    MetricDirection,
    MetricLevel,
    MetricResult,
)
from .base import BaseMetric, IntermediateRequest, register_metric
from .windowed.pairwise import (
    CCGCorrelationWindowedMixin,
    FunctionalConnectivityF1WindowedMixin,
)


def _warn_if_max_pairs_none(metric_name: str, max_pairs: int | None) -> None:
    if max_pairs is None:
        warnings.warn(
            (
                f"{metric_name}: max_pairs_to_compute is None; this will compute all "
                "pairs and can be quite slow. To cap pairwise pairs at runtime, "
                "override max_pairs_to_compute when constructing metric instances "
                "via the metrics dict."
            ),
            RuntimeWarning,
            stacklevel=3,
        )


def _is_significant_ccg(ccg: np.ndarray, threshold: float) -> bool:
    if np.std(ccg) == 0:
        return False
    z_scores = (ccg - np.mean(ccg)) / np.std(ccg)
    return bool(np.max(np.abs(z_scores)) > threshold)


@register_metric
class CCGCorrelation(CCGCorrelationWindowedMixin, BaseMetric):
    """Pearson correlation between original and reconstructed CCGs.

    Not yet testable: it needs CCG intermediates, which are not implemented."""

    name = "ccg_correlation"
    category = MetricCategory.POPULATION_STRUCTURE
    direction = MetricDirection.HIGHER_IS_BETTER
    level = MetricLevel.PER_PAIR
    value_range = (-1.0, 1.0)
    _required_intermediates = ["ccg_original", "ccg_reconstructed"]
    # To cap pairwise pairs at runtime (e.g. based on unit count),
    # override max_pairs_to_compute when constructing metric instances via the
    # metrics dict: {"ccg_correlation": {"metric_class": "ccg_correlation",
    # "max_pairs_to_compute": 1234}}
    _default_params = {
        "correlogram_max_lag_ms": 100.0,
        "correlogram_bin_size_ms": 1.0,
        "max_pairs_to_compute": 20,
    }
    aggregation_fallback_policy = "session_aggregated"

    def get_intermediate_requests(self, data_type: DataType) -> list[IntermediateRequest]:
        _warn_if_max_pairs_none(self.name, self.params["max_pairs_to_compute"])
        ccg_params = {
            "max_lag_ms": self.params["correlogram_max_lag_ms"],
            "bin_size_ms": self.params["correlogram_bin_size_ms"],
            "max_pairs": self.params["max_pairs_to_compute"],
        }
        return [
            IntermediateRequest.from_dict("ccg_original", ccg_params),
            IntermediateRequest.from_dict("ccg_reconstructed", ccg_params),
        ]

    supported_data_types = {DataType.SPIKE_TIMES, DataType.SPIKES, DataType.RATES}
    preferred_data_type = DataType.SPIKE_TIMES

    def compute(self, intermediates: dict[str, Any]) -> MetricResult:
        ccg_orig = intermediates["ccg_original"]
        ccg_recon = intermediates["ccg_reconstructed"]

        correlations = {}
        for pair in ccg_orig.keys():
            if pair in ccg_recon:
                orig = ccg_orig[pair]
                recon = ccg_recon[pair]

                if np.std(orig) > 0 and np.std(recon) > 0:
                    r = np.corrcoef(orig, recon)[0, 1]
                    correlations[pair] = r
                else:
                    correlations[pair] = np.nan

        correlation_values = np.array(list(correlations.values()))

        return self._make_result(
            raw_value=correlation_values,
            metadata={"pair_correlations": correlations},
        )


@register_metric
class FunctionalConnectivityF1(FunctionalConnectivityF1WindowedMixin, BaseMetric):
    """Precision/recall for detecting significantly connected pairs.

    Supports RATES and coarse-binned SPIKES via discrete cross-correlation.
    """

    name = "connectivity_f1"
    category = MetricCategory.POPULATION_STRUCTURE
    direction = MetricDirection.HIGHER_IS_BETTER
    level = MetricLevel.POPULATION
    value_range = (0.0, 1.0)
    _required_intermediates = ["ccg_original", "ccg_reconstructed"]
    # To cap pairwise pairs at runtime, override max_pairs_to_compute
    # via the metrics dict (see CCGCorrelation above).
    _default_params = {
        "ccg_significance_threshold": 3.0,
        "correlogram_max_lag_ms": 100.0,
        "correlogram_bin_size_ms": 1.0,
        "max_pairs_to_compute": 20,
    }
    aggregation_fallback_policy = "session_aggregated"

    def get_intermediate_requests(self, data_type: DataType) -> list[IntermediateRequest]:
        _warn_if_max_pairs_none(self.name, self.params["max_pairs_to_compute"])
        ccg_params = {
            "max_lag_ms": self.params["correlogram_max_lag_ms"],
            "bin_size_ms": self.params["correlogram_bin_size_ms"],
            "max_pairs": self.params["max_pairs_to_compute"],
        }
        return [
            IntermediateRequest.from_dict("ccg_original", ccg_params),
            IntermediateRequest.from_dict("ccg_reconstructed", ccg_params),
        ]

    supported_data_types = {DataType.SPIKE_TIMES, DataType.SPIKES, DataType.RATES}
    preferred_data_type = DataType.SPIKE_TIMES

    def compute(self, intermediates: dict[str, Any]) -> MetricResult:
        ccg_orig = intermediates["ccg_original"]
        ccg_recon = intermediates["ccg_reconstructed"]

        threshold = self.params["ccg_significance_threshold"]

        orig_connected = {
            pair for pair, ccg in ccg_orig.items() if _is_significant_ccg(ccg, threshold)
        }
        recon_connected = {
            pair for pair, ccg in ccg_recon.items() if _is_significant_ccg(ccg, threshold)
        }

        tp = len(orig_connected & recon_connected)
        fp = len(recon_connected - orig_connected)
        fn = len(orig_connected - recon_connected)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return self._make_result(
            raw_value=f1,
            metadata={
                "precision": precision,
                "recall": recall,
                "n_original_connections": len(orig_connected),
                "n_reconstructed_connections": len(recon_connected),
            },
        )
