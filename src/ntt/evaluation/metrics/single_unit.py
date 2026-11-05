"""Single-Unit Temporal Statistics metrics.

ISI-based metrics (ISIDistributionDivergence, CoefficientOfVariation, CV2LocalIrregularity)
require spike times and are NOT compatible with DataType.RATES.

ACGCorrelation supports RATES and coarse-binned SPIKES via discrete cross-correlation
of binned counts (correlogram resolution limited to the data bin size).
"""

from typing import Any

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

from ..core.types import (
    DataType,
    MetricCategory,
    MetricDirection,
    MetricLevel,
    MetricResult,
)
from .base import BaseMetric, IntermediateRequest, register_metric
from .windowed.single_unit import (
    ACGCorrelationWindowedMixin,
    CV2LocalIrregularityWindowedMixin,
    FanoFactorErrorWindowedMixin,
    FiringRateErrorWindowedMixin,
)


def _compute_cv2_terms(isis: Any, *, context: str) -> np.ndarray:
    isis_array = np.asarray(isis, dtype=np.float64)
    if isis_array.ndim != 1:
        raise ValueError(f"{context}: expected a 1D ISI array, got {isis_array.shape}.")
    if np.any(~np.isfinite(isis_array)):
        raise ValueError(f"{context}: ISIs contain NaN or inf.")
    if np.any(isis_array < 0):
        raise ValueError(f"{context}: ISIs contain negative values.")
    if isis_array.size < 2:
        return np.array([], dtype=np.float64)

    denominator = isis_array[1:] + isis_array[:-1]
    if np.any(denominator <= 0):
        raise ValueError(f"{context}: consecutive ISIs produced a non-positive CV2 denominator.")

    cv2_terms = 2.0 * np.abs(isis_array[1:] - isis_array[:-1]) / denominator
    if np.any(~np.isfinite(cv2_terms)):
        raise ValueError(f"{context}: CV2 terms contain NaN or inf.")
    return cv2_terms


def _extract_counts_and_duration(data: Any, *, context: str) -> tuple[np.ndarray, float]:
    if data is None:
        raise ValueError(f"{context}: EvalData is required.")

    duration = data.duration
    if duration is None:
        bin_size_seconds = data.bin_size_seconds
        if bin_size_seconds is None:
            raise ValueError(f"{context}: duration is missing and cannot be inferred.")
        if data.binned_spikes is not None:
            duration = data.binned_spikes.shape[1] * bin_size_seconds
        elif data.binned_counts is not None:
            duration = data.binned_counts.shape[1] * bin_size_seconds
        elif data.rates_hz is not None:
            duration = data.rates_hz.shape[1] * bin_size_seconds
        else:
            raise ValueError(f"{context}: duration is missing and cannot be inferred.")
    duration = float(duration)
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError(f"{context}: duration must be finite and positive, got {duration}.")

    if data.spike_times is not None:
        counts = np.array(
            [len(unit_spike_times) for unit_spike_times in data.spike_times], dtype=np.float64
        )
    elif data.binned_spikes is not None:
        counts = np.asarray(data.binned_spikes, dtype=np.float64).sum(axis=1)
    elif data.binned_counts is not None:
        counts = np.asarray(data.binned_counts, dtype=np.float64).sum(axis=1)
    elif data.rates_hz is not None:
        if data.bin_size_seconds is None:
            raise ValueError(f"{context}: bin_size_seconds required to convert rates to counts.")
        counts = np.asarray(data.rates_hz, dtype=np.float64).sum(axis=1) * float(
            data.bin_size_seconds
        )
    else:
        raise ValueError(f"{context}: no supported data representation found.")

    if counts.ndim != 1:
        raise ValueError(f"{context}: expected per-unit counts, got shape {counts.shape}.")
    if np.any(~np.isfinite(counts)):
        raise ValueError(f"{context}: counts contain NaN or inf.")
    if np.any(counts < 0):
        raise ValueError(f"{context}: counts contain negative values.")
    return counts, duration


@register_metric
class ISIDistributionDivergence(BaseMetric):
    """Wasserstein distance between original and reconstructed ISI distributions.

    NOTE: Not compatible with DataType.RATES. Requires spike times or fine-resolution spikes.
    """

    name = "isi_divergence"
    category = MetricCategory.SINGLE_UNIT_STATISTICS
    direction = MetricDirection.LOWER_IS_BETTER
    level = MetricLevel.PER_UNIT
    value_range = (0.0, None)
    _required_intermediates = ["isi_original", "isi_reconstructed"]
    _default_params = {"isi_smoothing_kernel": 1}

    # ISI metrics require spike times - NOT compatible with RATES
    supported_data_types = {DataType.SPIKE_TIMES, DataType.SPIKES}
    preferred_data_type = DataType.SPIKE_TIMES

    def get_intermediate_requests(self, data_type: DataType) -> list[IntermediateRequest]:
        isi_params = {"smoothing_kernel": self.params["isi_smoothing_kernel"]}
        return [
            IntermediateRequest.from_dict("isi_original", isi_params),
            IntermediateRequest.from_dict("isi_reconstructed", isi_params),
        ]

    def compute(self, intermediates: dict[str, Any]) -> MetricResult:
        isi_orig = intermediates["isi_original"]
        isi_recon = intermediates["isi_reconstructed"]

        wasserstein_distances = []
        ks_statistics = []

        for orig, recon in zip(isi_orig, isi_recon, strict=False):
            if len(orig) > 1 and len(recon) > 1:
                wasserstein_distances.append(wasserstein_distance(orig, recon))
                ks_stat, _ = ks_2samp(orig, recon)
                ks_statistics.append(ks_stat)
            else:
                wasserstein_distances.append(np.nan)
                ks_statistics.append(np.nan)

        return self._make_result(
            raw_value=np.array(wasserstein_distances),
            metadata={"ks_statistic": np.array(ks_statistics)},
        )


@register_metric
class ACGCorrelation(ACGCorrelationWindowedMixin, BaseMetric):
    """Pearson correlation between original and reconstructed autocorrelograms.

    Supports RATES and coarse-binned SPIKES via discrete cross-correlation.
    When data bin size is coarser than correlogram_bin_size_ms, the correlogram
    resolution is automatically adjusted to the data bin size.
    """

    name = "acg_correlation"
    category = MetricCategory.SINGLE_UNIT_STATISTICS
    direction = MetricDirection.HIGHER_IS_BETTER
    level = MetricLevel.PER_UNIT
    value_range = (-1.0, 1.0)
    _required_intermediates = ["acg_original", "acg_reconstructed"]
    _default_params = {
        "correlogram_max_lag_ms": 100.0,
        "correlogram_bin_size_ms": 1.0,
        "acg_smoothing_kernel": 1,
    }
    aggregation_fallback_policy = "session_aggregated"

    supported_data_types = {DataType.SPIKE_TIMES, DataType.SPIKES, DataType.RATES}
    preferred_data_type = DataType.SPIKE_TIMES

    def get_intermediate_requests(self, data_type: DataType) -> list[IntermediateRequest]:
        acg_params = {
            "max_lag_ms": self.params["correlogram_max_lag_ms"],
            "bin_size_ms": self.params["correlogram_bin_size_ms"],
            "smoothing_kernel": self.params["acg_smoothing_kernel"],
        }
        return [
            IntermediateRequest.from_dict("acg_original", acg_params),
            IntermediateRequest.from_dict("acg_reconstructed", acg_params),
        ]

    def compute(self, intermediates: dict[str, Any]) -> MetricResult:
        acg_orig = intermediates["acg_original"]
        acg_recon = intermediates["acg_reconstructed"]

        correlations = []
        for orig, recon in zip(acg_orig, acg_recon, strict=False):
            if np.std(orig) > 0 and np.std(recon) > 0:
                r = np.corrcoef(orig, recon)[0, 1]
                correlations.append(r)
            else:
                correlations.append(np.nan)

        return self._make_result(raw_value=np.array(correlations))


@register_metric
class CoefficientOfVariation(BaseMetric):
    """CV (std/mean of ISIs) comparison.

    NOTE: Not compatible with DataType.RATES. Requires spike times or fine-resolution spikes.
    """

    name = "cv_error"
    category = MetricCategory.SINGLE_UNIT_STATISTICS
    direction = MetricDirection.LOWER_IS_BETTER
    level = MetricLevel.PER_UNIT
    value_range = (0.0, None)
    _required_intermediates = ["isi_original", "isi_reconstructed"]
    _default_params = {"isi_smoothing_kernel": 1}
    aggregation_fallback_policy = "contiguous_only"

    def get_intermediate_requests(self, data_type: DataType) -> list[IntermediateRequest]:
        isi_params = {"smoothing_kernel": self.params["isi_smoothing_kernel"]}
        return [
            IntermediateRequest.from_dict("isi_original", isi_params),
            IntermediateRequest.from_dict("isi_reconstructed", isi_params),
        ]

    # CV requires ISI data - NOT compatible with RATES
    supported_data_types = {DataType.SPIKE_TIMES, DataType.SPIKES}
    preferred_data_type = DataType.SPIKE_TIMES

    def compute(self, intermediates: dict[str, Any]) -> MetricResult:
        isi_orig = intermediates["isi_original"]
        isi_recon = intermediates["isi_reconstructed"]

        cv_errors = []
        cv_orig_values = []
        cv_recon_values = []

        for orig, recon in zip(isi_orig, isi_recon, strict=False):
            if len(orig) > 1 and len(recon) > 1:
                cv_orig = np.std(orig) / np.mean(orig)
                cv_recon = np.std(recon) / np.mean(recon)
                cv_orig_values.append(cv_orig)
                cv_recon_values.append(cv_recon)
                cv_errors.append(np.abs(cv_orig - cv_recon))
            else:
                cv_orig_values.append(np.nan)
                cv_recon_values.append(np.nan)
                cv_errors.append(np.nan)

        return self._make_result(
            raw_value=np.array(cv_errors),
            metadata={
                "cv_original": np.array(cv_orig_values),
                "cv_reconstructed": np.array(cv_recon_values),
            },
        )


@register_metric
class CV2LocalIrregularity(CV2LocalIrregularityWindowedMixin, BaseMetric):
    """CV2 (local ISI irregularity) comparison.

    NOTE: Not compatible with DataType.RATES. Requires spike times or fine-resolution spikes.
    """

    name = "cv2_error"
    category = MetricCategory.SINGLE_UNIT_STATISTICS
    direction = MetricDirection.LOWER_IS_BETTER
    level = MetricLevel.PER_UNIT
    value_range = (0.0, None)
    _required_intermediates = ["isi_original", "isi_reconstructed"]
    _default_params = {"isi_smoothing_kernel": 1}
    aggregation_fallback_policy = "session_aggregated"

    def get_intermediate_requests(self, data_type: DataType) -> list[IntermediateRequest]:
        isi_params = {"smoothing_kernel": self.params["isi_smoothing_kernel"]}
        return [
            IntermediateRequest.from_dict("isi_original", isi_params),
            IntermediateRequest.from_dict("isi_reconstructed", isi_params),
        ]

    # CV2 requires ISI data - NOT compatible with RATES
    supported_data_types = {DataType.SPIKE_TIMES, DataType.SPIKES}
    preferred_data_type = DataType.SPIKE_TIMES

    def compute(self, intermediates: dict[str, Any]) -> MetricResult:
        isi_orig = intermediates["isi_original"]
        isi_recon = intermediates["isi_reconstructed"]

        def compute_cv2(isis):
            if len(isis) < 2:
                return np.nan
            cv2_values = _compute_cv2_terms(isis, context="CV2LocalIrregularity.compute")
            return np.mean(cv2_values)

        cv2_errors = []
        for orig, recon in zip(isi_orig, isi_recon, strict=False):
            cv2_orig = compute_cv2(orig)
            cv2_recon = compute_cv2(recon)
            if not np.isnan(cv2_orig) and not np.isnan(cv2_recon):
                cv2_errors.append(np.abs(cv2_orig - cv2_recon))
            else:
                cv2_errors.append(np.nan)

        return self._make_result(raw_value=np.array(cv2_errors))


@register_metric
class FiringRateError(FiringRateErrorWindowedMixin, BaseMetric):
    """Per-unit absolute error in mean firing rate (Hz).

    Computes |rate_orig - rate_recon| for each unit.  A simple but important
    sanity check: if the tokenizer systematically over- or under-predicts
    spikes, this metric catches it.

    Supports all data types (spike times, binary spikes, rates).
    """

    name = "firing_rate_error"
    category = MetricCategory.SINGLE_UNIT_STATISTICS
    direction = MetricDirection.LOWER_IS_BETTER
    level = MetricLevel.PER_UNIT
    value_range = (0.0, None)
    _required_intermediates = ["firing_rates_original", "firing_rates_reconstructed"]
    aggregation_fallback_policy = "session_aggregated"

    supported_data_types = {DataType.SPIKE_TIMES, DataType.SPIKES, DataType.RATES}
    preferred_data_type = DataType.SPIKES

    def compute(self, intermediates: dict[str, Any]) -> MetricResult:
        rates_orig = intermediates["firing_rates_original"]
        rates_recon = intermediates["firing_rates_reconstructed"]

        abs_errors = np.abs(rates_orig - rates_recon)

        return self._make_result(
            raw_value=abs_errors,
            metadata={
                "rates_original_hz": rates_orig,
                "rates_reconstructed_hz": rates_recon,
            },
        )


@register_metric
class FanoFactorError(FanoFactorErrorWindowedMixin, BaseMetric):
    """Per-unit absolute error in Fano factor (variance / mean of spike counts).

    The Fano factor measures the dispersion of spike counts relative to a
    Poisson process (FF = 1).  FF < 1 indicates sub-Poisson regularity,
    FF > 1 indicates over-dispersion (bursting or slow rate fluctuations).

    This metric computes |FF_orig - FF_recon| per unit at a configurable
    timescale, capturing whether the reconstruction preserves the
    variability structure of each neuron's firing.

    Uses the ``binned_counts`` intermediates (shared with
    ``spike_count_correlation`` and ``mutual_information``).
    """

    name = "fano_factor_error"
    category = MetricCategory.SINGLE_UNIT_STATISTICS
    direction = MetricDirection.LOWER_IS_BETTER
    level = MetricLevel.PER_UNIT
    value_range = (0.0, None)
    _required_intermediates = ["binned_counts_original", "binned_counts_reconstructed"]
    _default_params = {"fano_bin_size_ms": 100.0}
    aggregation_fallback_policy = "session_aggregated"

    supported_data_types = {DataType.SPIKE_TIMES, DataType.SPIKES, DataType.RATES}
    preferred_data_type = DataType.SPIKES

    def get_intermediate_requests(self, data_type: DataType) -> list[IntermediateRequest]:
        bin_params = {"bin_sizes_ms": [self.params["fano_bin_size_ms"]]}
        return [
            IntermediateRequest.from_dict("binned_counts_original", bin_params),
            IntermediateRequest.from_dict("binned_counts_reconstructed", bin_params),
        ]

    def compute(self, intermediates: dict[str, Any]) -> MetricResult:
        counts_orig = intermediates["binned_counts_original"]
        counts_recon = intermediates["binned_counts_reconstructed"]

        bin_size = self.params["fano_bin_size_ms"]

        # Use the requested bin size (or closest available)
        available = sorted(counts_orig.keys())
        if bin_size in counts_orig:
            target_bin = bin_size
        else:
            target_bin = available[-1]

        orig = counts_orig[target_bin]  # (n_units, n_bins)
        recon = counts_recon[target_bin]

        n_units = orig.shape[0]
        ff_orig = np.full(n_units, np.nan)
        ff_recon = np.full(n_units, np.nan)
        ff_errors = np.full(n_units, np.nan)

        for i in range(n_units):
            mean_o = np.mean(orig[i])
            mean_r = np.mean(recon[i])

            if mean_o > 0:
                ff_orig[i] = np.var(orig[i]) / mean_o
            if mean_r > 0:
                ff_recon[i] = np.var(recon[i]) / mean_r

            if not np.isnan(ff_orig[i]) and not np.isnan(ff_recon[i]):
                ff_errors[i] = np.abs(ff_orig[i] - ff_recon[i])

        return self._make_result(
            raw_value=ff_errors,
            metadata={
                "fano_factor_original": ff_orig,
                "fano_factor_reconstructed": ff_recon,
                "bin_size_ms": target_bin,
            },
        )
