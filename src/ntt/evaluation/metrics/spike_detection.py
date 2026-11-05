"""Spike Detection & Timing metrics.

RATES compatibility: Some metrics in this module (SpikeDetectionF1, TemporalPrecision)
require spike_matches intermediates which depend on spike times. Others (SmoothSE,
BinnedRateDifference) only support binary SPIKES. For RATES-based tokenizers:
- SpikeCountCorrelation already supports RATES
- The other metrics have no rate-based alternative
"""

from typing import Any

import numpy as np
from scipy.signal import convolve

from ..core.types import (
    DataType,
    MetricCategory,
    MetricDirection,
    MetricLevel,
    MetricResult,
)
from .base import BaseMetric, IntermediateRequest, register_metric
from .windowed.spike_detection import (
    SmoothSEWindowedMixin,
    SpikeCountCorrelationWindowedMixin,
    SpikeDetectionF1WindowedMixin,
    TemporalPrecisionWindowedMixin,
)


@register_metric
class SpikeDetectionF1(SpikeDetectionF1WindowedMixin, BaseMetric):
    """Precision, recall, and F1 for spike detection within tolerance window.

    NOTE: Requires spike_matches intermediates which need spike times.
    Not directly compatible with DataType.RATES.
    """

    name = "spike_detection_f1"
    category = MetricCategory.TEMPORAL_FIDELITY
    direction = MetricDirection.HIGHER_IS_BETTER
    level = MetricLevel.PER_UNIT
    value_range = (0.0, 1.0)
    _required_intermediates = ["spike_matches"]
    _default_params = {"spike_match_tolerance_ms": 1.0}
    aggregation_fallback_policy = "session_aggregated"

    def get_intermediate_requests(self, data_type: DataType) -> list[IntermediateRequest]:
        return [
            IntermediateRequest.from_dict(
                "spike_matches",
                {"tolerance_ms": self.params["spike_match_tolerance_ms"]},
            )
        ]

    # Requires spike times for spike matching - NOT compatible with RATES
    supported_data_types = {DataType.SPIKE_TIMES, DataType.SPIKES}
    preferred_data_type = DataType.SPIKE_TIMES

    def compute(self, intermediates: dict[str, Any]) -> MetricResult:
        matches = intermediates["spike_matches"]

        f1_scores = []
        precisions = []
        recalls = []

        for i in range(len(matches["true_positives"])):
            tp = matches["true_positives"][i]
            fp = matches["false_positives"][i]
            fn = matches["false_negatives"][i]

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            f1_scores.append(f1)
            precisions.append(precision)
            recalls.append(recall)

        return self._make_result(
            raw_value=np.array(f1_scores),
            metadata={
                "precision": np.array(precisions),
                "recall": np.array(recalls),
                "mean_true_positives": np.array(matches["true_positives"]).mean(),
                "mean_false_positives": np.array(matches["false_positives"]).mean(),
                "mean_false_negatives": np.array(matches["false_negatives"]).mean(),
                "tolerance": matches["tolerance"],
            },
        )


@register_metric
class TemporalPrecision(TemporalPrecisionWindowedMixin, BaseMetric):
    """Median timing error (ms) for matched spikes.

    Matches each original spike to the closest reconstructed spike within
    ``spike_match_tolerance_ms`` and reports the median absolute timing
    difference.  A wider tolerance than :class:`SpikeDetectionF1` is used
    by default so the metric produces a useful gradient even on binned data
    (where timing is quantised to the bin size).

    A perfect reconstruction scores 0.0 ms.
    """

    name = "temporal_precision"
    category = MetricCategory.TEMPORAL_FIDELITY
    direction = MetricDirection.LOWER_IS_BETTER
    level = MetricLevel.PER_UNIT
    value_range = (0.0, None)
    _required_intermediates = ["spike_matches"]
    _default_params = {"spike_match_tolerance_ms": 5.0}
    aggregation_fallback_policy = "session_aggregated"

    def get_intermediate_requests(self, data_type: DataType) -> list[IntermediateRequest]:
        return [
            IntermediateRequest.from_dict(
                "spike_matches",
                {"tolerance_ms": self.params["spike_match_tolerance_ms"]},
            )
        ]

    # Requires spike times for spike matching - NOT compatible with RATES
    supported_data_types = {DataType.SPIKE_TIMES, DataType.SPIKES}
    preferred_data_type = DataType.SPIKE_TIMES

    def compute(self, intermediates: dict[str, Any]) -> MetricResult:
        matches = intermediates["spike_matches"]

        median_errors = []
        iqr_errors = []

        for timing_errors in matches["timing_errors"]:
            if len(timing_errors) > 0:
                abs_errors = np.abs(timing_errors) * 1000  # convert to ms
                median_errors.append(np.median(abs_errors))
                iqr_errors.append(np.percentile(abs_errors, 75) - np.percentile(abs_errors, 25))
            else:
                median_errors.append(np.nan)
                iqr_errors.append(np.nan)

        return self._make_result(
            raw_value=np.array(median_errors), metadata={"iqr": np.array(iqr_errors)}
        )


@register_metric
class SpikeCountCorrelation(SpikeCountCorrelationWindowedMixin, BaseMetric):
    """Correlation of binned spike counts across multiple timescales."""

    name = "spike_count_correlation"
    category = MetricCategory.TEMPORAL_FIDELITY
    direction = MetricDirection.HIGHER_IS_BETTER
    level = MetricLevel.PER_UNIT
    value_range = (-1.0, 1.0)
    _required_intermediates = ["binned_counts_original", "binned_counts_reconstructed"]
    _default_params = {"bin_sizes_ms": [1.0, 10.0, 50.0, 100.0]}
    aggregation_fallback_policy = "session_aggregated"

    # Works with any data type
    supported_data_types = {DataType.SPIKE_TIMES, DataType.SPIKES, DataType.RATES}
    preferred_data_type = DataType.SPIKES

    def get_intermediate_requests(self, data_type: DataType) -> list[IntermediateRequest]:
        bin_params = {"bin_sizes_ms": self.params["bin_sizes_ms"]}
        return [
            IntermediateRequest.from_dict("binned_counts_original", bin_params),
            IntermediateRequest.from_dict("binned_counts_reconstructed", bin_params),
        ]

    def compute(self, intermediates: dict[str, Any]) -> MetricResult:
        counts_orig = intermediates["binned_counts_original"]
        counts_recon = intermediates["binned_counts_reconstructed"]

        # Use the finest available bin size for primary metric
        bin_sizes = sorted(counts_orig.keys())
        if not bin_sizes:
            return self._make_result(raw_value=np.array([np.nan]))
        primary_bin = bin_sizes[0]

        correlations = []
        for i in range(counts_orig[primary_bin].shape[0]):
            orig = counts_orig[primary_bin][i]
            recon = counts_recon[primary_bin][i]

            if np.std(orig) > 0 and np.std(recon) > 0:
                r = np.corrcoef(orig, recon)[0, 1]
                correlations.append(r)
            else:
                correlations.append(np.nan)

        # Also compute for other bin sizes as metadata
        correlations_by_bin = {}
        for bin_size in bin_sizes:
            corrs = []
            for i in range(counts_orig[bin_size].shape[0]):
                orig = counts_orig[bin_size][i]
                recon = counts_recon[bin_size][i]
                if np.std(orig) > 0 and np.std(recon) > 0:
                    corrs.append(np.corrcoef(orig, recon)[0, 1])
                else:
                    corrs.append(np.nan)
            correlations_by_bin[bin_size] = np.array(corrs)

        return self._make_result(
            raw_value=np.array(correlations),
            metadata={
                "correlations_by_bin_size": correlations_by_bin,
                "primary_bin_size_ms": primary_bin,
            },
        )

    def expand_results(self, result: MetricResult, display_name: str) -> dict[str, MetricResult]:
        by_bin = result.metadata.get("correlations_by_bin_size", {})
        primary = result.metadata.get("primary_bin_size_ms")
        expanded = {}
        for bin_size_ms, corrs in by_bin.items():
            if bin_size_ms == primary:
                continue  # already the primary result
            suffix = (
                f"_{int(bin_size_ms)}ms"
                if bin_size_ms == int(bin_size_ms)
                else f"_{bin_size_ms}ms"
            )
            expanded[f"{display_name}{suffix}"] = self._make_result(
                raw_value=corrs,
                metadata={"bin_size_ms": bin_size_ms},
            )
        return expanded


@register_metric
class SmoothSE(SmoothSEWindowedMixin, BaseMetric):
    """
    Summed squared error between smoothed spike trains.
    Smoothing kernel configurable between exponential or Gaussian filters.

    When `filter_type` is "exponential" the loss equals the summed squared
    difference between causal (or optionally non-causal) exponential filters,
    recovering the classical Van Rossum interpretation when `causal` is True,
    up to scaling (by 1/tau x a factor related to window length in actual units).
    Param 'causal' only is applied to exponential filters.
    In that case (VanRossum:)
    Formula:
        Uses numerical integration at 1ms resolution:
        D² = Σ_t [f_1(t) - f_2(t)]²

        where f(t) = 1/Z * Σ_i exp(-(t - t_i)/sigma) for t ≥ t_i (causal exponential kernel).

    Reference for Van Rossum: https://ieeexplore.ieee.org/abstract/document/6790198

    When `filter_type` is "gaussian" the exponential kernel is replaced with a
    symmetric Gaussian of width `sigma_ms` (std in milliseconds).

    The filters are L2-normalized (i.e. the sum of squared kernel values is 1).
    Combined with the per-spike normalization, each completely mismatched spike
    contributes ~1.0 to the per-unit error.

    Supports native-resolution count-valued inputs for both SPIKES and RATES.
    RATES inputs are interpreted via native expected counts per bin, so the
    Gaussian width in milliseconds naturally spans fewer bins when the input
    bin size is coarser.
    """

    name = "smoothed_squared_error"
    category = MetricCategory.TEMPORAL_FIDELITY
    direction = MetricDirection.LOWER_IS_BETTER
    level = MetricLevel.PER_UNIT
    value_range = (0.0, None)
    _required_intermediates = [
        "native_binned_counts_original",
        "native_binned_counts_reconstructed",
        "bin_size_original",
        "bin_size_reconstructed",
    ]
    _default_params = {"causal": False, "filter_type": "gaussian", "sigma_ms": 10.0}
    aggregation_fallback_policy = "session_aggregated"

    supported_data_types = {DataType.SPIKES, DataType.RATES}
    preferred_data_type = DataType.SPIKES

    def __init__(self, **params):
        super().__init__(**params)
        if self.params["filter_type"] not in ["exponential", "gaussian"]:
            raise ValueError("Invalid filter type. Must be 'exponential' or 'gaussian'.")
        if self.params["filter_type"] == "exponential":
            raise NotImplementedError(
                "The causal exponential filter is not supported; use filter_type='gaussian'."
            )

    def compute(self, intermediates: dict[str, Any]) -> MetricResult:
        sigma = self.params["sigma_ms"] / 1000.0  # convert to seconds

        bin_size = self._get_bin_size(intermediates)
        distances = self._compute_from_binned(
            intermediates.get("native_binned_counts_original"),
            intermediates.get("native_binned_counts_reconstructed"),
            bin_size,
            sigma,
        )

        return self._make_result(raw_value=np.array(distances))

    def _get_bin_size(self, intermediates: dict[str, Any]) -> float | None:
        bin_size_original = intermediates.get("bin_size_original")
        bin_size_reconstructed = intermediates.get("bin_size_reconstructed")
        if (
            bin_size_original is not None
            and bin_size_reconstructed is not None
            and not np.isclose(bin_size_original, bin_size_reconstructed)
        ):
            raise ValueError(
                "SmoothSE requires matching original/reconstructed bin sizes, got "
                f"{bin_size_original} vs {bin_size_reconstructed}."
            )
        if bin_size_original is not None:
            return bin_size_original
        return bin_size_reconstructed

    def _compute_from_binned(
        self,
        binned_orig: np.ndarray | None,
        binned_recon: np.ndarray | None,
        bin_size: float | None,
        sigma: float,
    ) -> list[float]:
        numerator, spike_counts = self._compute_window_terms_from_binned(
            binned_orig=binned_orig,
            binned_recon=binned_recon,
            bin_size=bin_size,
            sigma=sigma,
        )
        # Divide by number of spikes, so that we get the fraction
        #  of spikes the reconstructed got wrong (preserving dimensions)
        # Treat units with no original spikes as having 1 spike for normalization,
        # so that the metric is 0 when reconstruction also has no spikes.
        se = numerator / np.maximum(spike_counts, 1.0)
        return se.tolist()

    def _prepare_binned_counts(
        self,
        binned_orig: np.ndarray | None,
        binned_recon: np.ndarray | None,
        *,
        context: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        if binned_orig is None or binned_recon is None:
            raise ValueError(f"{context}: native binned counts are required.")

        orig = np.asarray(binned_orig, dtype=np.float64)
        recon = np.asarray(binned_recon, dtype=np.float64)

        if orig.ndim == 3:
            if orig.shape[0] != 1 or recon.shape[0] != 1:
                raise ValueError(
                    f"{context}: expected no batch dim or singleton batch dim, got "
                    f"{orig.shape} and {recon.shape}."
                )
            orig = orig[0]
            recon = recon[0]
        if orig.ndim != 2 or recon.ndim != 2:
            raise ValueError(
                f"{context}: expected 2D [units, time] arrays, got {orig.shape} and {recon.shape}."
            )
        if orig.shape != recon.shape:
            raise ValueError(
                f"{context}: original/reconstructed count shapes differ: "
                f"{orig.shape} vs {recon.shape}."
            )
        if np.any(~np.isfinite(orig)) or np.any(~np.isfinite(recon)):
            raise ValueError(f"{context}: counts contain NaN or inf.")
        if np.any(orig < 0) or np.any(recon < 0):
            raise ValueError(f"{context}: counts contain negative values.")
        return orig, recon

    def _compute_window_terms_from_binned(
        self,
        *,
        binned_orig: np.ndarray | None,
        binned_recon: np.ndarray | None,
        bin_size: float | None,
        sigma: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if bin_size is None or not np.isfinite(bin_size) or bin_size <= 0:
            raise ValueError(f"SmoothSE requires a finite positive bin size, got {bin_size}.")

        orig, recon = self._prepare_binned_counts(
            binned_orig,
            binned_recon,
            context="SmoothSE._compute_window_terms_from_binned",
        )
        spike_counts = orig.sum(axis=-1)
        filtered_orig = self._filter_array(orig, bin_size, sigma)
        filtered_recon = self._filter_array(recon, bin_size, sigma)
        diff_squared = (filtered_orig - filtered_recon) ** 2
        numerator = np.sum(diff_squared, axis=-1)
        if np.any(~np.isfinite(numerator)) or np.any(~np.isfinite(spike_counts)):
            raise ValueError(
                "SmoothSE._compute_window_terms_from_binned: numerator or spike counts "
                "contain NaN or inf."
            )
        return numerator, spike_counts

    def _filter_array(self, spikes: np.ndarray, bin_size: float, sigma: float) -> np.ndarray:
        window_size_seconds = spikes.shape[1] * bin_size
        kernel = self._make_binned_filter(bin_size, sigma, window_size_seconds)
        return convolve(spikes, kernel[None, :], mode="same")

    def _make_binned_filter(
        self,
        bin_size: float,
        sigma: float,
        window_size_seconds: float,
    ) -> np.ndarray:
        window_bins = min(
            int(round((6.0 * sigma) / bin_size)),
            int(round(window_size_seconds / bin_size)),
        )
        size = window_bins if window_bins % 2 == 1 else window_bins + 1
        if self.params["filter_type"] == "gaussian":
            positions = np.arange(size) - size // 2
            kernel = np.exp(-0.5 * (positions * bin_size / sigma) ** 2)
        else:
            if self.params["causal"]:
                positions = np.arange(size // 2)
            else:
                positions = np.arange(-size // 2, size // 2 + 1)
            kernel = np.exp(-np.abs(positions) * (bin_size / sigma))
        # kernel /= kernel.sum()
        kernel /= np.sqrt((kernel**2).sum())
        # \ So that its "norm", sum of squared kernel values, is 1, i.e.
        #   we get a sum of 1 per completely mismatched spike
        return kernel

    def _apply_binned_filter(
        self,
        spikes: np.ndarray,
        kernel: np.ndarray,
    ) -> np.ndarray:
        kernel = np.asarray(kernel)
        return convolve(spikes, kernel[None, :], mode="same")


@register_metric
class BinnedRateDifference(BaseMetric):
    """Per-unit mean absolute difference between binned spike rates.

    Bins spikes at specified resolution and computes the mean absolute error
    (over time) per unit. Rates are reported in Hz (spikes/s).
    Optionally normalizes by the original mean rate.

    Supports native-resolution count-valued inputs for both SPIKES and RATES.
    """

    name = "binned_rate_difference"
    category = MetricCategory.TEMPORAL_FIDELITY
    direction = MetricDirection.LOWER_IS_BETTER
    level = MetricLevel.PER_UNIT
    value_range = (0.0, None)
    _required_intermediates = [
        "native_binned_counts_original",
        "native_binned_counts_reconstructed",
        "bin_size_original",
        "bin_size_reconstructed",
    ]
    _default_params = {"bin_size_ms": 5.0, "normalize": False}
    aggregation_fallback_policy = "per_window"

    supported_data_types = {DataType.SPIKES, DataType.RATES}
    preferred_data_type = DataType.SPIKES

    def compute(self, intermediates: dict[str, Any]) -> MetricResult:
        from ..core.baselines import _rebin_counts  # lazy to avoid circular import

        binned_orig = intermediates["native_binned_counts_original"]
        binned_recon = intermediates["native_binned_counts_reconstructed"]
        bin_size_seconds = intermediates.get("bin_size_original")
        assert bin_size_seconds == intermediates.get("bin_size_reconstructed")

        if binned_orig is None or binned_recon is None:
            raise ValueError("Native binned counts are required for this metric.")
        if bin_size_seconds is None:
            raise ValueError("Bin size is required for this metric.")

        # Handle batch dimension: flatten if batch_size=1, otherwise raise error
        if binned_orig.ndim == 3:
            if binned_orig.shape[0] != 1:
                raise ValueError(
                    f"Expected no batch dim or a batch dim of size 1, "
                    f"got shape {binned_orig.shape}. "
                )
            binned_orig = binned_orig[0]  # (batch=1, units, time) -> (units, time)
            binned_recon = binned_recon[0]
        if binned_orig.shape != binned_recon.shape:
            raise ValueError("Original and reconstructed counts must have same shape.")

        # Convert bin size to seconds for metric computation
        metric_bin_size_seconds = self.params["bin_size_ms"] / 1000.0

        # Validate metric bin size is >= input bin size
        if metric_bin_size_seconds < bin_size_seconds - 1e-12:
            raise ValueError(
                f"Metric bin size ({self.params['bin_size_ms']}ms) must be >= "
                f"input bin size ({bin_size_seconds * 1000:.1f}ms)."
            )

        # Rebin if necessary
        binned_orig_at_metric_res = _rebin_counts(
            binned_orig, bin_size_seconds, metric_bin_size_seconds
        )
        binned_recon_at_metric_res = _rebin_counts(
            binned_recon, bin_size_seconds, metric_bin_size_seconds
        )

        # Per-unit MAE over time (in spike counts per metric bin)
        abs_diff = np.abs(binned_orig_at_metric_res - binned_recon_at_metric_res)
        mae_per_unit = np.mean(abs_diff, axis=-1)  # shape: (n_units,)

        # Convert to Hz (spikes/s)
        metric_bin_size_s = self.params["bin_size_ms"] / 1000.0
        mae_hz = mae_per_unit / metric_bin_size_s

        # Per-unit mean rates in Hz
        duration_s = binned_orig_at_metric_res.shape[-1] * metric_bin_size_s
        rate_orig_hz = binned_orig_at_metric_res.sum(axis=-1) / duration_s
        rate_recon_hz = binned_recon_at_metric_res.sum(axis=-1) / duration_s

        # Normalize per unit if requested
        if self.params["normalize"]:
            lower_bound_hz = 1.0 / metric_bin_size_s  # 1 spike per bin in Hz
            norm_factors = np.maximum(rate_orig_hz, lower_bound_hz)
            raw_value = mae_hz / norm_factors
        else:
            raw_value = mae_hz

        return self._make_result(
            raw_value=raw_value,
            metadata={
                "mean_rate_original_hz": rate_orig_hz,
                "mean_rate_reconstructed_hz": rate_recon_hz,
                "bin_size_ms": self.params["bin_size_ms"],
                "rate_unit": "Hz",
            },
        )
