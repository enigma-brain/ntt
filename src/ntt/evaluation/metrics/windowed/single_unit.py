"""Windowed (session-aggregated) implementations for single-unit metrics.

Holds the per-window accumulate/finalize for ACGCorrelation, CV2LocalIrregularity,
FiringRateError and FanoFactorError. The metric classes (and their non-windowed
``compute()`` plus the ISI/count helpers ``_compute_cv2_terms`` /
``_extract_counts_and_duration``) stay in
:mod:`ntt.evaluation.metrics.single_unit`; those helpers are reached via a local
import to avoid an import cycle (single_unit imports these mixins).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...core.types import MetricResult
from ._shared import MetricMixinBase, metric_requests_by_key, variance_from_sufficient_statistics


class ACGCorrelationWindowedMixin(MetricMixinBase):
    """Session-aggregated ACG correlation: mean ACG per unit, then Pearson."""

    aggregation_fallback_policy = "session_aggregated"

    def _accumulate_window(
        self,
        session_state: dict[str, Any],
        window_intermediates,
    ) -> None:
        requests = metric_requests_by_key(self)
        acg_original = np.stack(
            [
                np.asarray(unit_acg, dtype=np.float64)
                for unit_acg in window_intermediates.get(requests["acg_original"])
            ],
            axis=0,
        )
        acg_reconstructed = np.stack(
            [
                np.asarray(unit_acg, dtype=np.float64)
                for unit_acg in window_intermediates.get(requests["acg_reconstructed"])
            ],
            axis=0,
        )
        if acg_original.shape != acg_reconstructed.shape:
            raise ValueError(
                "ACGCorrelation._accumulate_window: per-window original/reconstructed "
                f"ACG shape mismatch: {acg_original.shape} vs {acg_reconstructed.shape}."
            )

        if "acg_original_running_sum" not in session_state:
            session_state["acg_original_running_sum"] = np.zeros_like(
                acg_original,
                dtype=np.float64,
            )
            session_state["acg_reconstructed_running_sum"] = np.zeros_like(
                acg_reconstructed,
                dtype=np.float64,
            )
            session_state["n_windows_seen"] = 0
        elif session_state["acg_original_running_sum"].shape != acg_original.shape:
            raise ValueError(
                "ACGCorrelation._accumulate_window: per-window ACG shape changed across "
                "windows: expected "
                f"{session_state['acg_original_running_sum'].shape}, got {acg_original.shape}."
            )

        session_state["acg_original_running_sum"] += acg_original
        session_state["acg_reconstructed_running_sum"] += acg_reconstructed
        session_state["n_windows_seen"] += 1

    def _finalize_session_state(self, session_state: dict[str, Any]) -> MetricResult:
        n_windows_seen = int(session_state.get("n_windows_seen", 0))
        if n_windows_seen == 0:
            raise ValueError(
                "ACGCorrelation._finalize_session_state called with no accumulated windows."
            )

        mean_acg_original = session_state["acg_original_running_sum"] / n_windows_seen
        mean_acg_reconstructed = session_state["acg_reconstructed_running_sum"] / n_windows_seen

        nan_sources: list[str] = []
        if np.any(np.isnan(mean_acg_original)):
            nan_sources.append("acg_original_running_sum")
        if np.any(np.isnan(mean_acg_reconstructed)):
            nan_sources.append("acg_reconstructed_running_sum")
        if nan_sources:
            raise ValueError(
                "ACGCorrelation._finalize_session_state: accumulated ACG running sum "
                f"contains NaN in {nan_sources}. Metric skipped."
            )

        per_unit_corrs = np.full(mean_acg_original.shape[0], np.nan, dtype=np.float64)
        for unit_idx, (orig, recon) in enumerate(
            zip(mean_acg_original, mean_acg_reconstructed, strict=False)
        ):
            if np.std(orig) > 0 and np.std(recon) > 0:
                per_unit_corrs[unit_idx] = float(np.corrcoef(orig, recon)[0, 1])

        if np.all(np.isnan(per_unit_corrs)):
            raise ValueError(
                "ACGCorrelation._finalize_session_state: every per-unit Pearson "
                "correlation is NaN (all units had zero variance or NaN accumulators)."
            )

        return self._make_result(
            raw_value=float(np.nanmean(per_unit_corrs)),
            metadata={
                "per_unit_corrs": per_unit_corrs.tolist(),
                "n_windows_seen": n_windows_seen,
            },
        )


class CV2LocalIrregularityWindowedMixin(MetricMixinBase):
    """Session-aggregated CV2: accumulate per-unit CV2-term sums and counts."""

    aggregation_fallback_policy = "session_aggregated"

    def _accumulate_window(
        self,
        session_state: dict[str, Any],
        window_intermediates,
    ) -> None:
        from ..single_unit import _compute_cv2_terms

        requests = metric_requests_by_key(self)
        isi_orig = window_intermediates.get(requests["isi_original"])
        isi_recon = window_intermediates.get(requests["isi_reconstructed"])
        if len(isi_orig) != len(isi_recon):
            raise ValueError(
                "CV2LocalIrregularity._accumulate_window: unit count mismatch between "
                f"original and reconstructed ISIs: {len(isi_orig)} vs {len(isi_recon)}."
            )

        n_units = len(isi_orig)
        if "orig_cv2_sum" not in session_state:
            session_state["orig_cv2_sum"] = np.zeros(n_units, dtype=np.float64)
            session_state["orig_cv2_count"] = np.zeros(n_units, dtype=np.int64)
            session_state["recon_cv2_sum"] = np.zeros(n_units, dtype=np.float64)
            session_state["recon_cv2_count"] = np.zeros(n_units, dtype=np.int64)
            session_state["n_windows_seen"] = 0
        elif session_state["orig_cv2_sum"].shape != (n_units,):
            raise ValueError(
                "CV2LocalIrregularity._accumulate_window: unit count changed across windows: "
                f"expected {session_state['orig_cv2_sum'].shape}, got {(n_units,)}."
            )

        for unit_idx, (orig_unit_isis, recon_unit_isis) in enumerate(
            zip(isi_orig, isi_recon, strict=False)
        ):
            orig_array = np.asarray(orig_unit_isis, dtype=np.float64)
            recon_array = np.asarray(recon_unit_isis, dtype=np.float64)
            if orig_array.ndim != 1 or recon_array.ndim != 1:
                raise ValueError(
                    "CV2LocalIrregularity._accumulate_window: ISIs must be 1D per unit."
                )
            if np.any(~np.isfinite(orig_array)) or np.any(~np.isfinite(recon_array)):
                raise ValueError(
                    "CV2LocalIrregularity._accumulate_window: ISIs contain NaN or inf."
                )
            if np.any(orig_array < 0) or np.any(recon_array < 0):
                raise ValueError(
                    "CV2LocalIrregularity._accumulate_window: ISIs contain negative values."
                )

            orig_terms = (
                _compute_cv2_terms(
                    orig_array,
                    context="CV2LocalIrregularity._accumulate_window original ISIs",
                )
                if orig_array.size >= 2
                else np.array([], dtype=np.float64)
            )
            recon_terms = (
                _compute_cv2_terms(
                    recon_array,
                    context="CV2LocalIrregularity._accumulate_window reconstructed ISIs",
                )
                if recon_array.size >= 2
                else np.array([], dtype=np.float64)
            )
            session_state["orig_cv2_sum"][unit_idx] += float(orig_terms.sum())
            session_state["orig_cv2_count"][unit_idx] += int(orig_terms.size)
            session_state["recon_cv2_sum"][unit_idx] += float(recon_terms.sum())
            session_state["recon_cv2_count"][unit_idx] += int(recon_terms.size)

        session_state["n_windows_seen"] += 1

    def _finalize_session_state(self, session_state: dict[str, Any]) -> MetricResult:
        if int(session_state.get("n_windows_seen", 0)) == 0:
            raise ValueError(
                "CV2LocalIrregularity._finalize_session_state called with no accumulated windows."
            )

        orig_counts = np.asarray(session_state["orig_cv2_count"], dtype=np.int64)
        recon_counts = np.asarray(session_state["recon_cv2_count"], dtype=np.int64)
        cv2_orig = np.full(orig_counts.shape, np.nan, dtype=np.float64)
        cv2_recon = np.full(recon_counts.shape, np.nan, dtype=np.float64)
        cv2_errors = np.full(orig_counts.shape, np.nan, dtype=np.float64)

        valid_orig = orig_counts > 0
        valid_recon = recon_counts > 0
        cv2_orig[valid_orig] = (
            np.asarray(session_state["orig_cv2_sum"], dtype=np.float64)[valid_orig]
            / orig_counts[valid_orig]
        )
        cv2_recon[valid_recon] = (
            np.asarray(session_state["recon_cv2_sum"], dtype=np.float64)[valid_recon]
            / recon_counts[valid_recon]
        )
        valid_both = valid_orig & valid_recon
        cv2_errors[valid_both] = np.abs(cv2_orig[valid_both] - cv2_recon[valid_both])

        return self._make_result(
            raw_value=cv2_errors,
            metadata={
                "cv2_original": cv2_orig,
                "cv2_reconstructed": cv2_recon,
                "n_windows_seen": int(session_state["n_windows_seen"]),
            },
        )


class FiringRateErrorWindowedMixin(MetricMixinBase):
    """Session-aggregated firing-rate error: accumulate counts + durations."""

    aggregation_fallback_policy = "session_aggregated"

    def _accumulate_window(
        self,
        session_state: dict[str, Any],
        window_intermediates,
    ) -> None:
        from ..single_unit import _extract_counts_and_duration

        counts_orig, duration_orig = _extract_counts_and_duration(
            window_intermediates.original,
            context="FiringRateError._accumulate_window original",
        )
        counts_recon, duration_recon = _extract_counts_and_duration(
            window_intermediates.reconstructed,
            context="FiringRateError._accumulate_window reconstructed",
        )
        if counts_orig.shape != counts_recon.shape:
            raise ValueError(
                "FiringRateError._accumulate_window: unit count mismatch between original "
                f"and reconstructed counts: {counts_orig.shape} vs {counts_recon.shape}."
            )

        if "original_count_total" not in session_state:
            session_state["original_count_total"] = np.zeros_like(counts_orig, dtype=np.float64)
            session_state["reconstructed_count_total"] = np.zeros_like(
                counts_recon, dtype=np.float64
            )
            session_state["original_duration_total"] = 0.0
            session_state["reconstructed_duration_total"] = 0.0
            session_state["n_windows_seen"] = 0
        elif session_state["original_count_total"].shape != counts_orig.shape:
            raise ValueError(
                "FiringRateError._accumulate_window: unit count changed across windows: "
                f"expected {session_state['original_count_total'].shape}, got {counts_orig.shape}."
            )

        session_state["original_count_total"] += counts_orig
        session_state["reconstructed_count_total"] += counts_recon
        session_state["original_duration_total"] += duration_orig
        session_state["reconstructed_duration_total"] += duration_recon
        session_state["n_windows_seen"] += 1

    def _finalize_session_state(self, session_state: dict[str, Any]) -> MetricResult:
        if int(session_state.get("n_windows_seen", 0)) == 0:
            raise ValueError(
                "FiringRateError._finalize_session_state called with no accumulated windows."
            )

        duration_orig = float(session_state["original_duration_total"])
        duration_recon = float(session_state["reconstructed_duration_total"])
        if not np.isfinite(duration_orig) or duration_orig <= 0:
            raise ValueError(
                "FiringRateError._finalize_session_state: accumulated original duration "
                f"must be finite and positive, got {duration_orig}."
            )
        if not np.isfinite(duration_recon) or duration_recon <= 0:
            raise ValueError(
                "FiringRateError._finalize_session_state: accumulated reconstructed duration "
                f"must be finite and positive, got {duration_recon}."
            )

        counts_orig = np.asarray(session_state["original_count_total"], dtype=np.float64)
        counts_recon = np.asarray(session_state["reconstructed_count_total"], dtype=np.float64)
        rates_orig = counts_orig / duration_orig
        rates_recon = counts_recon / duration_recon
        abs_errors = np.abs(rates_orig - rates_recon)

        return self._make_result(
            raw_value=abs_errors,
            metadata={
                "rates_original_hz": rates_orig,
                "rates_reconstructed_hz": rates_recon,
                "n_windows_seen": int(session_state["n_windows_seen"]),
            },
        )


class FanoFactorErrorWindowedMixin(MetricMixinBase):
    """Session-aggregated Fano factor: accumulate per-unit count sufficient stats."""

    aggregation_fallback_policy = "session_aggregated"

    def _accumulate_window(
        self,
        session_state: dict[str, Any],
        window_intermediates,
    ) -> None:
        requests = metric_requests_by_key(self)
        counts_orig = window_intermediates.get(requests["binned_counts_original"])
        counts_recon = window_intermediates.get(requests["binned_counts_reconstructed"])

        available_bins = tuple(sorted(counts_orig.keys()))
        if available_bins != tuple(sorted(counts_recon.keys())):
            raise ValueError(
                "FanoFactorError._accumulate_window: original/reconstructed available "
                "bin sizes differ."
            )
        target_bin = (
            self.params["fano_bin_size_ms"]
            if self.params["fano_bin_size_ms"] in counts_orig
            else available_bins[-1]
        )
        orig = np.asarray(counts_orig[target_bin], dtype=np.float64)
        recon = np.asarray(counts_recon[target_bin], dtype=np.float64)
        if orig.ndim != 2 or recon.ndim != 2:
            raise ValueError(
                "FanoFactorError._accumulate_window: expected 2D [units, bins] counts, got "
                f"{orig.shape} and {recon.shape}."
            )
        if orig.shape != recon.shape:
            raise ValueError(
                "FanoFactorError._accumulate_window: original/reconstructed count shape "
                f"mismatch: {orig.shape} vs {recon.shape}."
            )
        if np.any(~np.isfinite(orig)) or np.any(~np.isfinite(recon)):
            raise ValueError("FanoFactorError._accumulate_window: counts contain NaN or inf.")
        if np.any(orig < 0) or np.any(recon < 0):
            raise ValueError("FanoFactorError._accumulate_window: counts contain negative values.")

        if "original_sum" not in session_state:
            session_state["original_sum"] = np.zeros(orig.shape[0], dtype=np.float64)
            session_state["original_sum_sq"] = np.zeros(orig.shape[0], dtype=np.float64)
            session_state["reconstructed_sum"] = np.zeros(recon.shape[0], dtype=np.float64)
            session_state["reconstructed_sum_sq"] = np.zeros(recon.shape[0], dtype=np.float64)
            session_state["bin_count_total"] = 0
            session_state["target_bin_ms"] = float(target_bin)
            session_state["n_windows_seen"] = 0
        elif session_state["original_sum"].shape != (orig.shape[0],):
            raise ValueError(
                "FanoFactorError._accumulate_window: unit count changed across windows: "
                f"expected {session_state['original_sum'].shape}, got {(orig.shape[0],)}."
            )
        elif not np.isclose(session_state["target_bin_ms"], target_bin):
            raise ValueError(
                "FanoFactorError._accumulate_window: target bin size changed across windows: "
                f"expected {session_state['target_bin_ms']}, got {target_bin}."
            )

        session_state["original_sum"] += orig.sum(axis=1)
        session_state["original_sum_sq"] += np.square(orig).sum(axis=1)
        session_state["reconstructed_sum"] += recon.sum(axis=1)
        session_state["reconstructed_sum_sq"] += np.square(recon).sum(axis=1)
        session_state["bin_count_total"] += int(orig.shape[1])
        session_state["n_windows_seen"] += 1

    def _finalize_session_state(self, session_state: dict[str, Any]) -> MetricResult:
        if int(session_state.get("n_windows_seen", 0)) == 0:
            raise ValueError(
                "FanoFactorError._finalize_session_state called with no accumulated windows."
            )

        n_bins_total = int(session_state["bin_count_total"])
        original_sum = np.asarray(session_state["original_sum"], dtype=np.float64)
        reconstructed_sum = np.asarray(session_state["reconstructed_sum"], dtype=np.float64)
        ff_orig = np.full(original_sum.shape, np.nan, dtype=np.float64)
        ff_recon = np.full(reconstructed_sum.shape, np.nan, dtype=np.float64)
        ff_errors = np.full(original_sum.shape, np.nan, dtype=np.float64)
        if n_bins_total > 0:
            original_mean, original_variance = variance_from_sufficient_statistics(
                sum_values=original_sum,
                sum_squares=np.asarray(session_state["original_sum_sq"], dtype=np.float64),
                counts=n_bins_total,
                context="FanoFactorError._finalize_session_state original counts",
            )
            reconstructed_mean, reconstructed_variance = variance_from_sufficient_statistics(
                sum_values=reconstructed_sum,
                sum_squares=np.asarray(
                    session_state["reconstructed_sum_sq"],
                    dtype=np.float64,
                ),
                counts=n_bins_total,
                context="FanoFactorError._finalize_session_state reconstructed counts",
            )

            valid_orig = original_mean > 0
            valid_recon = reconstructed_mean > 0
            ff_orig[valid_orig] = original_variance[valid_orig] / original_mean[valid_orig]
            ff_recon[valid_recon] = (
                reconstructed_variance[valid_recon] / reconstructed_mean[valid_recon]
            )
            valid_both = valid_orig & valid_recon
            ff_errors[valid_both] = np.abs(ff_orig[valid_both] - ff_recon[valid_both])

        return self._make_result(
            raw_value=ff_errors,
            metadata={
                "fano_factor_original": ff_orig,
                "fano_factor_reconstructed": ff_recon,
                "bin_size_ms": float(session_state["target_bin_ms"]),
                "n_windows_seen": int(session_state["n_windows_seen"]),
            },
        )
