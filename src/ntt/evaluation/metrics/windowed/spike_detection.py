"""Windowed (session-aggregated) implementations for spike-detection metrics.

Streaming per-window accumulate/finalize for SpikeDetectionF1, TemporalPrecision,
SpikeCountCorrelation and SmoothSE, plus the streaming spike-match helpers they
use. The metric classes and their non-windowed ``compute()`` (and SmoothSE's
``_get_bin_size`` / ``_compute_window_terms_from_binned`` instance methods, which
the mixin reaches via ``self``) stay in
:mod:`ntt.evaluation.metrics.spike_detection`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...core.types import MetricResult
from ._shared import (
    MetricMixinBase,
    extract_spike_times,
    get_duration_seconds,
    metric_requests_by_key,
    pearson_from_sufficient_statistics,
    require_finite_2d_counts,
)


def _match_spikes_single_unit(
    original: np.ndarray,
    reconstructed: np.ndarray,
    tolerance: float,
) -> dict[str, Any]:
    n_orig = len(original)
    n_recon = len(reconstructed)

    if n_orig == 0 or n_recon == 0:
        return {
            "matched_original_idx": np.array([], dtype=int),
            "matched_reconstructed_idx": np.array([], dtype=int),
            "timing_errors": np.array([], dtype=float),
            "true_positives": 0,
            "false_positives": n_recon,
            "false_negatives": n_orig,
        }

    matched_orig_idx = []
    matched_recon_idx = []
    timing_errors = []
    recon_used = np.zeros(n_recon, dtype=bool)

    for i in range(n_orig):
        t_orig = float(original[i])
        pos = np.searchsorted(reconstructed, t_orig)
        best_j = -1
        best_dist = tolerance + 1.0
        for candidate in (pos - 1, pos):
            if 0 <= candidate < n_recon:
                distance = abs(float(reconstructed[candidate]) - t_orig)
                if distance < best_dist:
                    best_dist = distance
                    best_j = candidate

        if best_j >= 0 and best_dist <= tolerance and not recon_used[best_j]:
            matched_orig_idx.append(i)
            matched_recon_idx.append(best_j)
            timing_errors.append(reconstructed[best_j] - t_orig)
            recon_used[best_j] = True

    tp = len(matched_orig_idx)
    return {
        "matched_original_idx": np.array(matched_orig_idx, dtype=int),
        "matched_reconstructed_idx": np.array(matched_recon_idx, dtype=int),
        "timing_errors": np.asarray(timing_errors, dtype=np.float64),
        "true_positives": tp,
        "false_positives": n_recon - tp,
        "false_negatives": n_orig - tp,
    }


def _resolve_streaming_window_start(
    session_state: dict[str, Any],
    *,
    data: Any,
    duration_seconds: float,
    context: str,
) -> float:
    expected_window_start = float(session_state.get("next_window_start_seconds", 0.0))
    window_start = data.start_time
    if window_start is None:
        window_start = expected_window_start
    else:
        window_start = float(window_start)
        if "next_window_start_seconds" in session_state and not np.isclose(
            expected_window_start,
            window_start,
        ):
            raise ValueError(
                f"{context}: non-contiguous window start. "
                f"Expected {expected_window_start}, got {window_start}."
            )
    session_state["next_window_start_seconds"] = window_start + duration_seconds
    return window_start


def _finalize_spike_match_stream_step(
    *,
    carry_original: np.ndarray,
    carry_reconstructed: np.ndarray,
    current_original: np.ndarray,
    current_reconstructed: np.ndarray,
    window_start_seconds: float,
    window_duration_seconds: float,
    tolerance_seconds: float,
    is_last_window: bool,
    context: str,
) -> dict[str, Any]:
    current_original_abs = np.asarray(current_original, dtype=np.float64) + window_start_seconds
    current_reconstructed_abs = (
        np.asarray(current_reconstructed, dtype=np.float64) + window_start_seconds
    )
    original_abs = (
        np.concatenate((carry_original, current_original_abs))
        if carry_original.size or current_original_abs.size
        else np.array([], dtype=np.float64)
    )
    reconstructed_abs = (
        np.concatenate((carry_reconstructed, current_reconstructed_abs))
        if carry_reconstructed.size or current_reconstructed_abs.size
        else np.array([], dtype=np.float64)
    )
    match_result = _match_spikes_single_unit(
        original=original_abs,
        reconstructed=reconstructed_abs,
        tolerance=tolerance_seconds,
    )

    matched_original_idx = np.asarray(match_result["matched_original_idx"], dtype=np.int64)
    matched_reconstructed_idx = np.asarray(
        match_result["matched_reconstructed_idx"], dtype=np.int64
    )
    timing_errors = np.asarray(match_result["timing_errors"], dtype=np.float64)
    if matched_original_idx.shape != matched_reconstructed_idx.shape:
        raise ValueError(
            f"{context}: matched original/reconstructed index arrays have different shapes."
        )
    if timing_errors.shape != matched_original_idx.shape:
        raise ValueError(f"{context}: timing_errors shape does not match matched index shape.")

    original_is_matched = np.zeros(original_abs.shape[0], dtype=bool)
    reconstructed_is_matched = np.zeros(reconstructed_abs.shape[0], dtype=bool)
    original_is_matched[matched_original_idx] = True
    reconstructed_is_matched[matched_reconstructed_idx] = True

    finalize_before = np.inf
    if not is_last_window:
        finalize_before = window_start_seconds + window_duration_seconds - tolerance_seconds
    original_finalizable = original_abs < finalize_before
    reconstructed_finalizable = reconstructed_abs < finalize_before

    finalized_pair_mask = np.asarray(
        [
            original_finalizable[orig_idx] or reconstructed_finalizable[recon_idx]
            for orig_idx, recon_idx in zip(
                matched_original_idx,
                matched_reconstructed_idx,
                strict=False,
            )
        ],
        dtype=bool,
    )
    finalized_original_idx = matched_original_idx[finalized_pair_mask]
    finalized_reconstructed_idx = matched_reconstructed_idx[finalized_pair_mask]

    carry_original_mask = original_abs >= finalize_before
    carry_reconstructed_mask = reconstructed_abs >= finalize_before
    carry_original_mask[finalized_original_idx] = False
    carry_reconstructed_mask[finalized_reconstructed_idx] = False

    return {
        "tp": int(finalized_pair_mask.sum()),
        "fp": int(np.count_nonzero(~reconstructed_is_matched & reconstructed_finalizable)),
        "fn": int(np.count_nonzero(~original_is_matched & original_finalizable)),
        "timing_errors": timing_errors[finalized_pair_mask],
        "carry_original": original_abs[carry_original_mask],
        "carry_reconstructed": reconstructed_abs[carry_reconstructed_mask],
    }


class SpikeDetectionF1WindowedMixin(MetricMixinBase):
    """Session-aggregated streaming spike-match F1 (carry spikes across windows)."""

    aggregation_fallback_policy = "session_aggregated"

    def _accumulate_window(
        self,
        session_state: dict[str, Any],
        window_intermediates,
    ) -> None:
        tolerance_seconds = float(self.params["spike_match_tolerance_ms"]) / 1000.0
        if not np.isfinite(tolerance_seconds) or tolerance_seconds <= 0:
            raise ValueError(
                "SpikeDetectionF1._accumulate_window: spike match tolerance must be finite and "
                f"positive, got {tolerance_seconds}."
            )
        original_spike_times = extract_spike_times(
            window_intermediates.original,
            context="SpikeDetectionF1._accumulate_window original",
        )
        reconstructed_spike_times = extract_spike_times(
            window_intermediates.reconstructed,
            context="SpikeDetectionF1._accumulate_window reconstructed",
        )
        if len(original_spike_times) != len(reconstructed_spike_times):
            raise ValueError(
                "SpikeDetectionF1._accumulate_window: unit count mismatch between original "
                f"and reconstructed spike trains: {len(original_spike_times)} vs "
                f"{len(reconstructed_spike_times)}."
            )

        duration_original = get_duration_seconds(
            window_intermediates.original,
            context="SpikeDetectionF1._accumulate_window original",
        )
        duration_reconstructed = get_duration_seconds(
            window_intermediates.reconstructed,
            context="SpikeDetectionF1._accumulate_window reconstructed",
        )
        if not np.isclose(duration_original, duration_reconstructed):
            raise ValueError(
                "SpikeDetectionF1._accumulate_window: original/reconstructed window durations "
                f"differ: {duration_original} vs {duration_reconstructed}."
            )
        window_start = _resolve_streaming_window_start(
            session_state,
            data=window_intermediates.original,
            duration_seconds=duration_original,
            context="SpikeDetectionF1._accumulate_window",
        )

        if "tp" not in session_state:
            n_units = len(original_spike_times)
            session_state["tp"] = np.zeros(n_units, dtype=np.int64)
            session_state["fp"] = np.zeros(n_units, dtype=np.int64)
            session_state["fn"] = np.zeros(n_units, dtype=np.int64)
            session_state["carry_original_spikes_by_unit"] = [
                np.array([], dtype=np.float64) for _ in range(n_units)
            ]
            session_state["carry_reconstructed_spikes_by_unit"] = [
                np.array([], dtype=np.float64) for _ in range(n_units)
            ]
            session_state["tolerance_seconds"] = tolerance_seconds
            session_state["n_windows_seen"] = 0
        elif session_state["tp"].shape != (len(original_spike_times),):
            raise ValueError(
                "SpikeDetectionF1._accumulate_window: unit count changed across windows: "
                f"expected {session_state['tp'].shape}, got {(len(original_spike_times),)}."
            )
        elif not np.isclose(session_state["tolerance_seconds"], tolerance_seconds):
            raise ValueError(
                "SpikeDetectionF1._accumulate_window: spike match tolerance changed across "
                f"windows: expected {session_state['tolerance_seconds']}, got {tolerance_seconds}."
            )

        for unit_idx, (orig_unit_spikes, recon_unit_spikes) in enumerate(
            zip(original_spike_times, reconstructed_spike_times, strict=False)
        ):
            step_result = _finalize_spike_match_stream_step(
                carry_original=np.asarray(
                    session_state["carry_original_spikes_by_unit"][unit_idx],
                    dtype=np.float64,
                ),
                carry_reconstructed=np.asarray(
                    session_state["carry_reconstructed_spikes_by_unit"][unit_idx],
                    dtype=np.float64,
                ),
                current_original=orig_unit_spikes,
                current_reconstructed=recon_unit_spikes,
                window_start_seconds=window_start,
                window_duration_seconds=duration_original,
                tolerance_seconds=tolerance_seconds,
                is_last_window=False,
                context=f"SpikeDetectionF1._accumulate_window unit {unit_idx}",
            )
            session_state["tp"][unit_idx] += step_result["tp"]
            session_state["fp"][unit_idx] += step_result["fp"]
            session_state["fn"][unit_idx] += step_result["fn"]
            session_state["carry_original_spikes_by_unit"][unit_idx] = step_result[
                "carry_original"
            ]
            session_state["carry_reconstructed_spikes_by_unit"][unit_idx] = step_result[
                "carry_reconstructed"
            ]
        session_state["n_windows_seen"] += 1

    def _finalize_session_state(self, session_state: dict[str, Any]) -> MetricResult:
        if int(session_state.get("n_windows_seen", 0)) == 0:
            raise ValueError(
                "SpikeDetectionF1._finalize_session_state called with no accumulated windows."
            )

        tp = np.asarray(session_state["tp"], dtype=np.float64).copy()
        fp = np.asarray(session_state["fp"], dtype=np.float64).copy()
        fn = np.asarray(session_state["fn"], dtype=np.float64).copy()
        tolerance_seconds = float(session_state["tolerance_seconds"])
        for unit_idx, (carry_original, carry_reconstructed) in enumerate(
            zip(
                session_state["carry_original_spikes_by_unit"],
                session_state["carry_reconstructed_spikes_by_unit"],
                strict=False,
            )
        ):
            step_result = _finalize_spike_match_stream_step(
                carry_original=np.asarray(carry_original, dtype=np.float64),
                carry_reconstructed=np.asarray(carry_reconstructed, dtype=np.float64),
                current_original=np.array([], dtype=np.float64),
                current_reconstructed=np.array([], dtype=np.float64),
                window_start_seconds=0.0,
                window_duration_seconds=0.0,
                tolerance_seconds=tolerance_seconds,
                is_last_window=True,
                context=f"SpikeDetectionF1._finalize_session_state unit {unit_idx}",
            )
            tp[unit_idx] += step_result["tp"]
            fp[unit_idx] += step_result["fp"]
            fn[unit_idx] += step_result["fn"]

        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
        f1_scores = np.divide(
            2.0 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision),
            where=(precision + recall) > 0,
        )

        return self._make_result(
            raw_value=f1_scores,
            metadata={
                "precision": precision,
                "recall": recall,
                "mean_true_positives": float(np.mean(tp)),
                "mean_false_positives": float(np.mean(fp)),
                "mean_false_negatives": float(np.mean(fn)),
                "tolerance": [float(session_state["tolerance_seconds"])],
                "n_windows_seen": int(session_state["n_windows_seen"]),
            },
        )


class TemporalPrecisionWindowedMixin(MetricMixinBase):
    """Session-aggregated timing-error precision via streaming spike matches."""

    aggregation_fallback_policy = "session_aggregated"

    def _accumulate_window(
        self,
        session_state: dict[str, Any],
        window_intermediates,
    ) -> None:
        tolerance_seconds = float(self.params["spike_match_tolerance_ms"]) / 1000.0
        if not np.isfinite(tolerance_seconds) or tolerance_seconds <= 0:
            raise ValueError(
                "TemporalPrecision._accumulate_window: spike match tolerance must be finite and "
                f"positive, got {tolerance_seconds}."
            )

        original_spike_times = extract_spike_times(
            window_intermediates.original,
            context="TemporalPrecision._accumulate_window original",
        )
        reconstructed_spike_times = extract_spike_times(
            window_intermediates.reconstructed,
            context="TemporalPrecision._accumulate_window reconstructed",
        )
        if len(original_spike_times) != len(reconstructed_spike_times):
            raise ValueError(
                "TemporalPrecision._accumulate_window: unit count mismatch between original "
                f"and reconstructed spike trains: {len(original_spike_times)} vs "
                f"{len(reconstructed_spike_times)}."
            )

        duration_original = get_duration_seconds(
            window_intermediates.original,
            context="TemporalPrecision._accumulate_window original",
        )
        duration_reconstructed = get_duration_seconds(
            window_intermediates.reconstructed,
            context="TemporalPrecision._accumulate_window reconstructed",
        )
        if not np.isclose(duration_original, duration_reconstructed):
            raise ValueError(
                "TemporalPrecision._accumulate_window: original/reconstructed window durations "
                f"differ: {duration_original} vs {duration_reconstructed}."
            )
        window_start = _resolve_streaming_window_start(
            session_state,
            data=window_intermediates.original,
            duration_seconds=duration_original,
            context="TemporalPrecision._accumulate_window",
        )

        if "timing_error_chunks_by_unit" not in session_state:
            session_state["timing_error_chunks_by_unit"] = [
                [] for _ in range(len(original_spike_times))
            ]
            session_state["carry_original_spikes_by_unit"] = [
                np.array([], dtype=np.float64) for _ in range(len(original_spike_times))
            ]
            session_state["carry_reconstructed_spikes_by_unit"] = [
                np.array([], dtype=np.float64) for _ in range(len(original_spike_times))
            ]
            session_state["tolerance_seconds"] = tolerance_seconds
            session_state["n_windows_seen"] = 0
        elif len(session_state["timing_error_chunks_by_unit"]) != len(original_spike_times):
            raise ValueError(
                "TemporalPrecision._accumulate_window: unit count changed across windows: "
                f"expected {len(session_state['timing_error_chunks_by_unit'])}, got "
                f"{len(original_spike_times)}."
            )
        elif not np.isclose(session_state["tolerance_seconds"], tolerance_seconds):
            raise ValueError(
                "TemporalPrecision._accumulate_window: spike match tolerance changed across "
                f"windows: expected {session_state['tolerance_seconds']}, got {tolerance_seconds}."
            )

        for unit_idx, (orig_unit_spikes, recon_unit_spikes) in enumerate(
            zip(original_spike_times, reconstructed_spike_times, strict=False)
        ):
            step_result = _finalize_spike_match_stream_step(
                carry_original=np.asarray(
                    session_state["carry_original_spikes_by_unit"][unit_idx],
                    dtype=np.float64,
                ),
                carry_reconstructed=np.asarray(
                    session_state["carry_reconstructed_spikes_by_unit"][unit_idx],
                    dtype=np.float64,
                ),
                current_original=orig_unit_spikes,
                current_reconstructed=recon_unit_spikes,
                window_start_seconds=window_start,
                window_duration_seconds=duration_original,
                tolerance_seconds=tolerance_seconds,
                is_last_window=False,
                context=f"TemporalPrecision._accumulate_window unit {unit_idx}",
            )
            if step_result["timing_errors"].size > 0:
                session_state["timing_error_chunks_by_unit"][unit_idx].append(
                    step_result["timing_errors"]
                )
            session_state["carry_original_spikes_by_unit"][unit_idx] = step_result[
                "carry_original"
            ]
            session_state["carry_reconstructed_spikes_by_unit"][unit_idx] = step_result[
                "carry_reconstructed"
            ]
        session_state["n_windows_seen"] += 1

    def _finalize_session_state(self, session_state: dict[str, Any]) -> MetricResult:
        if int(session_state.get("n_windows_seen", 0)) == 0:
            raise ValueError(
                "TemporalPrecision._finalize_session_state called with no accumulated windows."
            )

        n_units = len(session_state["timing_error_chunks_by_unit"])
        median_errors = np.full(n_units, np.nan, dtype=np.float64)
        iqr_errors = np.full(n_units, np.nan, dtype=np.float64)
        tolerance_seconds = float(session_state["tolerance_seconds"])
        for unit_idx in range(n_units):
            step_result = _finalize_spike_match_stream_step(
                carry_original=np.asarray(
                    session_state["carry_original_spikes_by_unit"][unit_idx],
                    dtype=np.float64,
                ),
                carry_reconstructed=np.asarray(
                    session_state["carry_reconstructed_spikes_by_unit"][unit_idx],
                    dtype=np.float64,
                ),
                current_original=np.array([], dtype=np.float64),
                current_reconstructed=np.array([], dtype=np.float64),
                window_start_seconds=0.0,
                window_duration_seconds=0.0,
                tolerance_seconds=tolerance_seconds,
                is_last_window=True,
                context=f"TemporalPrecision._finalize_session_state unit {unit_idx}",
            )
            timing_error_chunks = list(session_state["timing_error_chunks_by_unit"][unit_idx])
            if step_result["timing_errors"].size > 0:
                timing_error_chunks.append(step_result["timing_errors"])
            timing_errors = (
                np.abs(np.concatenate(timing_error_chunks)) * 1000.0
                if timing_error_chunks
                else np.array([], dtype=np.float64)
            )
            if timing_errors.size == 0:
                continue
            median_errors[unit_idx] = float(np.median(timing_errors))
            iqr_errors[unit_idx] = float(
                np.percentile(timing_errors, 75) - np.percentile(timing_errors, 25)
            )

        return self._make_result(
            raw_value=median_errors,
            metadata={
                "iqr": iqr_errors,
                "tolerance": [float(session_state["tolerance_seconds"])],
                "n_windows_seen": int(session_state["n_windows_seen"]),
            },
        )


class SpikeCountCorrelationWindowedMixin(MetricMixinBase):
    """Session-aggregated spike-count Pearson via running sufficient statistics."""

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
                "SpikeCountCorrelation._accumulate_window: original/reconstructed available "
                "bin sizes differ."
            )
        if not available_bins:
            raise ValueError(
                "SpikeCountCorrelation._accumulate_window: no feasible bin sizes available."
            )

        if "stats_by_bin_size" not in session_state:
            session_state["stats_by_bin_size"] = {}
            session_state["available_bin_sizes"] = available_bins
            session_state["n_windows_seen"] = 0
        elif session_state["available_bin_sizes"] != available_bins:
            raise ValueError(
                "SpikeCountCorrelation._accumulate_window: available bin sizes changed across "
                f"windows: expected {session_state['available_bin_sizes']}, got {available_bins}."
            )

        for bin_size in available_bins:
            orig = require_finite_2d_counts(
                counts_orig[bin_size],
                context=(f"SpikeCountCorrelation._accumulate_window original {bin_size}ms counts"),
            )
            recon = require_finite_2d_counts(
                counts_recon[bin_size],
                context=(
                    f"SpikeCountCorrelation._accumulate_window reconstructed {bin_size}ms counts"
                ),
            )
            if orig.shape != recon.shape:
                raise ValueError(
                    "SpikeCountCorrelation._accumulate_window: original/reconstructed "
                    f"{bin_size}ms counts shape mismatch: {orig.shape} vs {recon.shape}."
                )

            stats_by_bin = session_state["stats_by_bin_size"]
            if bin_size not in stats_by_bin:
                stats_by_bin[bin_size] = {
                    "sum_x": np.zeros(orig.shape[0], dtype=np.float64),
                    "sum_y": np.zeros(orig.shape[0], dtype=np.float64),
                    "sum_x2": np.zeros(orig.shape[0], dtype=np.float64),
                    "sum_y2": np.zeros(orig.shape[0], dtype=np.float64),
                    "sum_xy": np.zeros(orig.shape[0], dtype=np.float64),
                    "n_observations": 0,
                }
            elif stats_by_bin[bin_size]["sum_x"].shape != (orig.shape[0],):
                raise ValueError(
                    "SpikeCountCorrelation._accumulate_window: unit count changed across "
                    f"windows for {bin_size}ms bins: expected "
                    f"{stats_by_bin[bin_size]['sum_x'].shape}, got {(orig.shape[0],)}."
                )

            stats = stats_by_bin[bin_size]
            stats["sum_x"] += orig.sum(axis=1)
            stats["sum_y"] += recon.sum(axis=1)
            stats["sum_x2"] += np.square(orig).sum(axis=1)
            stats["sum_y2"] += np.square(recon).sum(axis=1)
            stats["sum_xy"] += (orig * recon).sum(axis=1)
            stats["n_observations"] += int(orig.shape[1])

        session_state["n_windows_seen"] += 1

    def _finalize_session_state(self, session_state: dict[str, Any]) -> MetricResult:
        if int(session_state.get("n_windows_seen", 0)) == 0:
            raise ValueError(
                "SpikeCountCorrelation._finalize_session_state called with no accumulated windows."
            )

        available_bins = session_state["available_bin_sizes"]
        correlations_by_bin: dict[float, np.ndarray] = {}
        for bin_size in available_bins:
            stats = session_state["stats_by_bin_size"][bin_size]
            correlations_by_bin[bin_size] = pearson_from_sufficient_statistics(
                sum_x=stats["sum_x"],
                sum_y=stats["sum_y"],
                sum_x2=stats["sum_x2"],
                sum_y2=stats["sum_y2"],
                sum_xy=stats["sum_xy"],
                n_observations=int(stats["n_observations"]),
                context=(
                    f"SpikeCountCorrelation._finalize_session_state {bin_size}ms correlation"
                ),
            )

        primary_bin = available_bins[0]
        return self._make_result(
            raw_value=correlations_by_bin[primary_bin],
            metadata={
                "correlations_by_bin_size": correlations_by_bin,
                "primary_bin_size_ms": primary_bin,
                "n_windows_seen": int(session_state["n_windows_seen"]),
            },
        )


class SmoothSEWindowedMixin(MetricMixinBase):
    """Session-aggregated smoothed squared-error accumulation."""

    aggregation_fallback_policy = "session_aggregated"

    if TYPE_CHECKING:
        # Provided by the concrete SmoothSE metric (declared here for the type checker only).
        def _get_bin_size(self, intermediates: dict[str, Any]) -> float | None: ...

        def _compute_window_terms_from_binned(
            self,
            *,
            binned_orig: np.ndarray | None,
            binned_recon: np.ndarray | None,
            bin_size: float | None,
            sigma: float,
        ) -> tuple[np.ndarray, np.ndarray]: ...

    def _accumulate_window(
        self,
        session_state: dict[str, Any],
        window_intermediates,
    ) -> None:
        sigma = self.params["sigma_ms"] / 1000.0
        requests = metric_requests_by_key(self)
        window_intermediates_dict = {
            "native_binned_counts_original": window_intermediates.get(
                requests["native_binned_counts_original"]
            ),
            "native_binned_counts_reconstructed": window_intermediates.get(
                requests["native_binned_counts_reconstructed"]
            ),
            "bin_size_original": window_intermediates.get(requests["bin_size_original"]),
            "bin_size_reconstructed": window_intermediates.get(requests["bin_size_reconstructed"]),
        }
        numerator, spike_counts = self._compute_window_terms_from_binned(
            binned_orig=window_intermediates_dict["native_binned_counts_original"],
            binned_recon=window_intermediates_dict["native_binned_counts_reconstructed"],
            bin_size=self._get_bin_size(window_intermediates_dict),
            sigma=sigma,
        )

        if "numerator_running_sum" not in session_state:
            session_state["numerator_running_sum"] = np.zeros_like(numerator, dtype=np.float64)
            session_state["spike_count_total"] = np.zeros_like(spike_counts, dtype=np.float64)
            session_state["n_windows_seen"] = 0
        elif session_state["numerator_running_sum"].shape != numerator.shape:
            raise ValueError(
                "SmoothSE._accumulate_window: unit count changed across windows: expected "
                f"{session_state['numerator_running_sum'].shape}, got {numerator.shape}."
            )

        session_state["numerator_running_sum"] += numerator
        session_state["spike_count_total"] += spike_counts
        session_state["n_windows_seen"] += 1

    def _finalize_session_state(self, session_state: dict[str, Any]) -> MetricResult:
        if int(session_state.get("n_windows_seen", 0)) == 0:
            raise ValueError(
                "SmoothSE._finalize_session_state called with no accumulated windows."
            )

        numerator = np.asarray(session_state["numerator_running_sum"], dtype=np.float64)
        spike_counts = np.asarray(session_state["spike_count_total"], dtype=np.float64)
        if np.any(~np.isfinite(numerator)) or np.any(~np.isfinite(spike_counts)):
            raise ValueError(
                "SmoothSE._finalize_session_state: accumulated numerator or spike counts "
                "contain NaN or inf."
            )

        distances = numerator / np.maximum(spike_counts, 1.0)
        return self._make_result(
            raw_value=distances,
            metadata={
                "original_spike_count": spike_counts,
                "n_windows_seen": int(session_state["n_windows_seen"]),
            },
        )
