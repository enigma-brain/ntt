"""Information-theoretic metrics.

Provides mutual information between original and reconstructed spike counts,
computed from the exact discrete distribution of integer counts with
Miller-Madow bias correction.
"""

from typing import Any

import numpy as np

from ..core.types import DataType, MetricCategory, MetricDirection, MetricLevel, MetricResult
from .base import BaseMetric, IntermediateRequest, register_metric
from .windowed.information import MutualInformationWindowedMixin


@register_metric
class MutualInformation(MutualInformationWindowedMixin, BaseMetric):
    """Normalized mutual information between original and reconstructed spike counts.

    Measures what fraction of information in the original spike counts is
    preserved in the reconstruction, at each of several temporal bin sizes.

    **Method.**  Spike counts at a given bin size are integer-valued, so the
    joint distribution p(x_orig, x_recon) is built as an exact contingency
    table — no histogram binning or kernel density estimation is needed.
    MI = H(orig) + H(recon) − H(joint), where H are Shannon entropies
    of the marginal and joint spike count distributions from this table.
    The plugin MI estimate is then bias-corrected with the Miller-Madow
    correction [1]_, which subtracts the leading-order finite-sample bias:

        bias ≈ (S_xy − S_x − S_y + 1) / (2 N ln 2)   [bits]

    where S_xy, S_x, S_y are the numbers of occupied cells in the joint and
    marginal distributions, and N is the number of time bins.

    **Primary value** is the *normalized* MI at the finest available timescale:

        NMI = MI_corrected / H(original)

    This is bounded in [0, 1]: 0 means reconstruction carries no information,
    1 means perfect reconstruction.  Raw MI (bits), H(original), H(reconstructed),
    and the bias magnitude are available per timescale in metadata.

    **When results are unreliable.**  Returns NaN for a given unit/timescale
    when (a) the sample size is below ``min_samples``, (b) the original
    entropy is effectively zero (constant firing), or (c) the
    sample-to-occupied-cell ratio N / S_xy < 5, indicating that the
    contingency table is too sparse for the correction to be trustworthy.

    References
    ----------
    .. [1] Miller, G. (1955). "Note on the bias of information estimates."
       Information Theory in Psychology, 2, 95-100.

    See Also
    --------
    spike_count_correlation : Pearson correlation of binned counts (linear
        dependence only; scale-invariant).  MI captures all statistical
        dependencies, including nonlinear ones, but is harder to estimate
        reliably.
    """

    name = "mutual_information"
    category = MetricCategory.TEMPORAL_FIDELITY
    direction = MetricDirection.HIGHER_IS_BETTER
    level = MetricLevel.PER_UNIT
    value_range = (0.0, 1.0)
    _required_intermediates = ["binned_counts_original", "binned_counts_reconstructed"]
    _default_params = {
        "bin_sizes_ms": [1.0, 5.0, 10.0, 50.0, 100.0],
        "min_samples": 50,
    }
    # aggregation_fallback_policy ("session_aggregated") + the windowed
    # accumulate/finalize come from MutualInformationWindowedMixin.

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

        bin_sizes = sorted(counts_orig.keys())
        if not bin_sizes:
            return self._make_result(raw_value=np.array([np.nan]))

        min_samples = self.params["min_samples"]

        # Per-timescale, per-unit results
        nmi_by_bin: dict[float, np.ndarray] = {}
        mi_by_bin: dict[float, np.ndarray] = {}
        hx_by_bin: dict[float, np.ndarray] = {}
        hy_by_bin: dict[float, np.ndarray] = {}
        bias_by_bin: dict[float, np.ndarray] = {}

        for bin_size in bin_sizes:
            orig_at_bin = counts_orig[bin_size]
            recon_at_bin = counts_recon[bin_size]
            n_units = orig_at_bin.shape[0]

            nmi_arr = np.full(n_units, np.nan)
            mi_arr = np.full(n_units, np.nan)
            hx_arr = np.full(n_units, np.nan)
            hy_arr = np.full(n_units, np.nan)
            bias_arr = np.full(n_units, np.nan)

            for i in range(n_units):
                mi_c, hx, hy, n, bias, reliable = _discrete_mi(
                    orig_at_bin[i], recon_at_bin[i], min_samples
                )
                hx_arr[i] = hx
                hy_arr[i] = hy
                bias_arr[i] = bias

                if not reliable:
                    continue

                mi_arr[i] = mi_c
                # Normalize by original entropy
                nmi_arr[i] = min(mi_c / hx, 1.0) if hx > 0 else np.nan

            nmi_by_bin[bin_size] = nmi_arr
            mi_by_bin[bin_size] = mi_arr
            hx_by_bin[bin_size] = hx_arr
            hy_by_bin[bin_size] = hy_arr
            bias_by_bin[bin_size] = bias_arr

        # Primary: NMI at finest available timescale
        primary_bin = bin_sizes[0]

        return self._make_result(
            raw_value=nmi_by_bin[primary_bin],
            metadata={
                "nmi_by_bin_size": nmi_by_bin,
                "mi_bits_by_bin_size": mi_by_bin,
                "h_original_bits_by_bin_size": hx_by_bin,
                "h_reconstructed_bits_by_bin_size": hy_by_bin,
                "bias_bits_by_bin_size": bias_by_bin,
                "primary_bin_size_ms": primary_bin,
            },
        )


# ---------------------------------------------------------------------------
# Core computation (module-level for testability)
# ---------------------------------------------------------------------------

#: Minimum ratio of samples to occupied joint cells for the Miller-Madow
#: correction to be trustworthy.  Below this the contingency table is too
#: sparse and the bias term itself has high variance.
_MIN_SAMPLES_PER_CELL = 5


def _discrete_mi(
    x: np.ndarray,
    y: np.ndarray,
    min_samples: int = 50,
) -> tuple[float, float, float, int, float, bool]:
    """Mutual information between integer-valued count arrays.

    Uses the exact discrete joint distribution and applies the Miller-Madow
    bias correction.

    Parameters
    ----------
    x, y : 1-D arrays of spike counts (same length).
    min_samples : Minimum number of time bins required.

    Returns
    -------
    mi_corrected : Bias-corrected MI in bits (clamped >= 0).
    h_x : Entropy of *x* in bits.
    h_y : Entropy of *y* in bits.
    n : Sample count.
    bias : Bias estimate that was subtracted (bits).
    reliable : Whether the estimate passes quality checks.
    """
    n = len(x)
    nan_out = (np.nan, np.nan, np.nan, n, np.nan, False)

    if n < min_samples:
        return nan_out

    x_int = np.round(x).astype(np.intp)
    y_int = np.round(y).astype(np.intp)

    # Shift to 0-based indices for bincount
    x_min, y_min = x_int.min(), y_int.min()
    xs = x_int - x_min
    ys = y_int - y_min
    nx = int(xs.max()) + 1
    ny = int(ys.max()) + 1

    # Joint contingency table via flat indexing
    flat = xs * ny + ys
    joint = np.bincount(flat, minlength=nx * ny).reshape(nx, ny)

    # Marginals
    mx = joint.sum(axis=1)
    my = joint.sum(axis=0)

    # Entropies (bits)
    h_x = _entropy_bits(mx, n)
    h_y = _entropy_bits(my, n)
    h_xy = _entropy_bits(joint.ravel(), n)
    mi = h_x + h_y - h_xy

    # Miller-Madow bias correction
    s_xy = int(np.count_nonzero(joint))
    s_x = int(np.count_nonzero(mx))
    s_y = int(np.count_nonzero(my))
    bias = (s_xy - s_x - s_y + 1) / (2.0 * n * np.log(2.0))
    mi_corrected = max(0.0, mi - bias)

    # Reliability checks
    if h_x < 1e-10:
        # Original has no entropy (constant); NMI is undefined
        return mi_corrected, h_x, h_y, n, bias, False

    if s_xy > 0 and n / s_xy < _MIN_SAMPLES_PER_CELL:
        # Contingency table too sparse for correction to be trustworthy
        return mi_corrected, h_x, h_y, n, bias, False

    return mi_corrected, h_x, h_y, n, bias, True


def _discrete_mi_from_sparse_counts(
    *,
    joint_counts: dict[tuple[int, int], int],
    orig_counts: dict[int, int],
    recon_counts: dict[int, int],
    n: int,
    min_samples: int = 50,
) -> tuple[float, float, float, int, float, bool]:
    """Mutual information from already-aggregated sparse contingency counts."""
    nan_out = (np.nan, np.nan, np.nan, n, np.nan, False)
    if n < min_samples:
        return nan_out

    joint = np.fromiter(joint_counts.values(), dtype=np.int64)
    mx = np.fromiter(orig_counts.values(), dtype=np.int64)
    my = np.fromiter(recon_counts.values(), dtype=np.int64)

    if joint.size == 0 or mx.size == 0 or my.size == 0:
        return nan_out

    h_x = _entropy_bits(mx, n)
    h_y = _entropy_bits(my, n)
    h_xy = _entropy_bits(joint, n)
    mi = h_x + h_y - h_xy

    s_xy = int(joint.size)
    s_x = int(mx.size)
    s_y = int(my.size)
    bias = (s_xy - s_x - s_y + 1) / (2.0 * n * np.log(2.0))
    mi_corrected = max(0.0, mi - bias)

    if h_x < 1e-10:
        return mi_corrected, h_x, h_y, n, bias, False
    if s_xy > 0 and n / s_xy < _MIN_SAMPLES_PER_CELL:
        return mi_corrected, h_x, h_y, n, bias, False

    return mi_corrected, h_x, h_y, n, bias, True


def _entropy_bits(counts: np.ndarray, n: int) -> float:
    """Shannon entropy in bits from an array of integer counts."""
    p = counts[counts > 0] / n
    return float(-np.sum(p * np.log2(p)))
