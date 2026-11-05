"""Repeats-based metrics for stimulus-aligned evaluation (Cowley-style).

Consumes the ``stimulus_grouped_latents`` intermediate produced by the
interval-windowing pipeline: one logical evaluation window per stimulus
presentation, grouped by the stimulus label.

Reference for debiasing:
Sahani, M., & Linden, J. F. (2003). *How linear are auditory cortical
responses?* NeurIPS 15.
"""

from __future__ import annotations

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


@register_metric
class RepeatsSNR(BaseMetric):
    """Repeats-based signal-to-noise ratio.

    Primary scalar is the debiased trace-ratio
    ``sum_j sigma2_sig_debiased(j) / sum_j sigma2_noise(j)``.
    """

    name = "repeats_snr"
    category = MetricCategory.LATENT_QUALITY
    direction = MetricDirection.HIGHER_IS_BETTER
    level = MetricLevel.POPULATION
    value_range = (0.0, None)
    _default_params = {"grouping_field": "image_id"}
    _required_intermediates = ["stimulus_grouped_latents"]

    # Follow PopulationDimensionality / SubspaceOverlap for supported types
    # — the metric consumes only latents, so any binned representation works.
    supported_data_types = {DataType.SPIKE_TIMES, DataType.SPIKES, DataType.RATES}
    preferred_data_type = DataType.RATES

    def get_intermediate_requests(self, data_type: DataType) -> list[IntermediateRequest]:
        """Forward the grouping_field parameter to the intermediate cache."""
        return [
            IntermediateRequest.from_dict(
                "stimulus_grouped_latents",
                {"grouping_field": self.params["grouping_field"]},
            )
        ]

    def compute(self, intermediates: dict[str, Any]) -> MetricResult:
        """Compute the debiased repeats-SNR from the grouped-latents intermediate.

        Uses a homoscedastic pooled noise-variance estimator for the debias:
        ``sigma2_sig_debiased = max(0, sigma2_sig_raw - sigma2_noise * mean_s(1/N_s))``,
        where ``sigma2_noise`` is the mean across stimuli of the per-stim
        within-repeat variance. Cowley noise is effectively homoscedastic
        across stimuli, and the homoscedastic form has lower variance at
        low N_s than the per-stim heteroscedastic form. A heteroscedasticity
        diagnostic is reported alongside a finite-sample floor for
        interpretation (see metadata).

        """
        inter = intermediates["stimulus_grouped_latents"]
        Z: np.ndarray = inter["latents"]  # [S, R_max, D]
        M: np.ndarray = inter["repeat_mask"]  # [S, R_max]
        repeat_counts: np.ndarray = inter["repeat_counts"]  # [S]
        token_shape = inter["token_shape"]
        token_bin_size_ms = inter.get("token_bin_size_ms")
        # For forwarding to results metadata:
        n_tokens_cut_info = inter["n_tokens_cut_info"]
        tokens_dropped_info = inter.get("tokens_dropped_info")
        short_window_info = inter.get("short_window_info")

        S, R_max, D = Z.shape

        if S < 2:
            raise ValueError(
                f"RepeatsSNR requires at least 2 stimuli to estimate between-stimulus "
                f"variance; got S={S}."
            )

        min_reps = int(repeat_counts.min())
        max_reps = int(repeat_counts.max())
        if min_reps < 2:
            raise ValueError(
                f"RepeatsSNR requires at least 2 repeats per stimulus to estimate "
                f"within-stimulus (noise) variance; got min repeats = {min_reps}."
            )

        # Masked array: mask is True where entry is INVALID (numpy convention).
        # repeat_mask in the intermediate is True where VALID, so we invert.
        invalid = ~M  # [S, R_max]
        invalid_3d = np.broadcast_to(invalid[:, :, None], Z.shape)
        Zm = np.ma.array(Z, mask=invalid_3d)

        # Per-stimulus mean across repeats: shape [S, D]
        mean_r = Zm.mean(axis=1)
        # Unbiased per-stim within-repeat variance (ddof=1). We already require
        # N_s >= 2 above, so this is well-defined for every stimulus.
        var_r_per_stim = np.asarray(Zm.var(axis=1, ddof=1))  # [S, D]

        # Sample variance across stimuli of per-stimulus means (ddof=1): [D]
        # mean_r is fully populated (every stimulus has >= 2 valid repeats).
        mean_r_arr = np.asarray(mean_r)  # unmask; no masked entries remain
        sigma2_sig_raw = mean_r_arr.var(axis=0, ddof=1)

        # A heteroscedastic bias correction (per-stimulus noise variance divided by
        # its repeat count) is not used: the repeat noise is effectively
        # homoscedastic and that variant has high variance at low N_s.

        # Homoscedastic pooled noise-variance estimator: mean of per-stim variances
        # across stimuli. This is sigma2_noise (the same one stored in metadata).
        sigma2_noise = var_r_per_stim.mean(axis=0)  # [D]

        # Under homoscedasticity the expected per-stim bias of V_sample(mean_r z_s)
        # is sigma2_noise / N_s, so the total bias averaged across stims is
        # sigma2_noise * mean_s(1 / N_s).
        mean_inv_repeat_count = float((1.0 / repeat_counts.astype(np.float64)).mean())
        sigma2_sig_debiased = np.maximum(
            0.0, sigma2_sig_raw - sigma2_noise * mean_inv_repeat_count
        )

        # Descriptive diagnostic: how heteroscedastic is the per-stim noise?
        # CV_d = std_s(sigma2_noise[s, d]) / mean_s(sigma2_noise[s, d]).
        # Near 0 => homoscedastic; >> 1 => strongly heteroscedastic. Purely
        # informational — does not gate any branch in compute().
        eps_cv = 1e-12
        sigma2_noise_std_per_dim = var_r_per_stim.std(axis=0, ddof=1)  # [D]
        heteroscedasticity_coeff_of_var_per_dim = sigma2_noise_std_per_dim / (
            sigma2_noise + eps_cv
        )
        heteroscedasticity_coeff_of_var_mean = float(
            heteroscedasticity_coeff_of_var_per_dim.mean()
        )

        # Under homoscedastic Gaussian noise, the sample variance itself has
        # Var(s^2) = 2 sigma^4 / (N_s - 1), so its CV is sqrt(2/(N_s - 1)). Across
        # stimuli with ragged N_s we report the mean of the per-stim floors as a
        # single scalar baseline. Values of heteroscedasticity_coeff_of_var_per_dim
        # near this floor just reflect sampling noise in the per-stim variance
        # estimate, NOT real heteroscedasticity; only values meaningfully above
        # this floor indicate that stimuli have different true noise variances.
        floor_per_stim = np.sqrt(2.0 / (repeat_counts.astype(np.float64) - 1.0))
        heteroscedasticity_coeff_of_var_finite_sample_floor = float(floor_per_stim.mean())

        sum_noise = float(sigma2_noise.sum())
        sum_sig_raw = float(sigma2_sig_raw.sum())
        sum_sig_debiased = float(sigma2_sig_debiased.sum())

        if sum_noise <= 0.0:
            raise ValueError(
                f"RepeatsSNR: total noise variance is zero ({sum_noise}); cannot form "
                f"an SNR ratio. This suggests all repeats are identical within every "
                f"stimulus."
            )

        snr_debiased = sum_sig_debiased / sum_noise
        snr_raw = sum_sig_raw / sum_noise

        # Per-dim SNR arrays (convenience; reshape happens at viz time).
        eps = 1e-12
        sigma2_noise_clip = np.maximum(sigma2_noise, eps)
        snr_per_dim_debiased = sigma2_sig_debiased / sigma2_noise_clip
        snr_per_dim_raw = sigma2_sig_raw / sigma2_noise_clip

        metadata: dict[str, Any] = {
            "snr_debiased": snr_debiased,
            "snr_raw": snr_raw,
            "sigma2_sig_raw": sigma2_sig_raw,
            "sigma2_sig_debiased": sigma2_sig_debiased,
            "sigma2_noise": sigma2_noise,
            "heteroscedasticity_coeff_of_var_per_dim": heteroscedasticity_coeff_of_var_per_dim,
            "heteroscedasticity_coeff_of_var_mean": heteroscedasticity_coeff_of_var_mean,
            "heteroscedasticity_coeff_of_var_finite_sample_floor": (
                heteroscedasticity_coeff_of_var_finite_sample_floor
            ),
            "mean_inv_repeat_count": mean_inv_repeat_count,
            "snr_per_dim_debiased": snr_per_dim_debiased,
            "snr_per_dim_raw": snr_per_dim_raw,
            "n_stimuli": int(S),
            "n_repeats_mean_arithmetic": float(repeat_counts.mean()),
            "n_repeats_min": min_reps,
            "n_repeats_max": max_reps,
            "grouping_field": self.params["grouping_field"],
            "token_shape": token_shape,
            # This is the post-cut token count per stimulus repeat that
            # actually enters the SNR computation.
            "n_tokens_per_stimulus_repeat_used": int(token_shape[0]),
            "token_bin_size_ms": token_bin_size_ms,
            "n_tokens_cut_info": n_tokens_cut_info,
            "tokens_dropped_info": tokens_dropped_info,
            "short_window_info": short_window_info,
        }

        return self._make_result(raw_value=snr_debiased, metadata=metadata)
