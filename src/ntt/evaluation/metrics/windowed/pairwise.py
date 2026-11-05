"""Windowed (session-aggregated) implementations for pairwise CCG metrics.

CCGCorrelation and FunctionalConnectivityF1 accumulate a running mean CCG per
unit-pair across windows (with exact cross-window boundary correction) and
finalize from the pooled mean. The streaming CCG machinery lives here; the
metric classes and their non-windowed ``compute()`` (plus ``_is_significant_ccg``
and ``_warn_if_max_pairs_none``) stay in :mod:`ntt.evaluation.metrics.pairwise`.
``FunctionalConnectivityF1._finalize_session_state`` reuses ``self.compute`` on
the pooled mean CCG, so the F1/significance logic is not duplicated here.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from ...core.types import DataType, EvalData, MetricResult
from ..base import BaseMetric, IntermediateRequest
from ._shared import MetricMixinBase

CCG_SAMPLING_RATE = 1000


def _get_ccg_requests(
    metric: BaseMetric,
    data_type: DataType,
) -> dict[str, IntermediateRequest]:
    return {request.key: request for request in metric.get_intermediate_requests(data_type)}


def _extract_ccg_pair_matrices(
    ccg_original: dict[tuple[int, int], np.ndarray],
    ccg_reconstructed: dict[tuple[int, int], np.ndarray],
    *,
    metric_name: str,
) -> tuple[tuple[tuple[int, int], ...], np.ndarray, np.ndarray]:
    pair_order = tuple(sorted(ccg_original))
    if pair_order != tuple(sorted(ccg_reconstructed)):
        raise ValueError(
            f"{metric_name}._accumulate_window: per-window original/reconstructed "
            "CCG pair sets differ."
        )

    ccg_original_matrix = np.stack(
        [np.asarray(ccg_original[pair], dtype=np.float64) for pair in pair_order],
        axis=0,
    )
    ccg_reconstructed_matrix = np.stack(
        [np.asarray(ccg_reconstructed[pair], dtype=np.float64) for pair in pair_order],
        axis=0,
    )
    if ccg_original_matrix.shape != ccg_reconstructed_matrix.shape:
        raise ValueError(
            f"{metric_name}._accumulate_window: per-window original/reconstructed "
            f"CCG shape mismatch: {ccg_original_matrix.shape} vs {ccg_reconstructed_matrix.shape}."
        )
    return pair_order, ccg_original_matrix, ccg_reconstructed_matrix


def _compute_ccg_window_geometry(
    *,
    max_lag_ms: float,
    bin_size_ms: float,
    metric_name: str,
) -> tuple[int, int, int]:
    max_lag_samples = int((float(max_lag_ms) / 1000.0) * CCG_SAMPLING_RATE)
    bin_samples = int((float(bin_size_ms) / 1000.0) * CCG_SAMPLING_RATE)
    if max_lag_samples <= 0 or bin_samples <= 0:
        raise ValueError(
            f"{metric_name}: correlogram max_lag_ms and bin_size_ms must yield positive "
            f"sample counts, got max_lag_samples={max_lag_samples}, "
            f"bin_samples={bin_samples}."
        )
    if max_lag_samples % bin_samples != 0:
        raise ValueError(
            f"{metric_name}: max_lag_samples={max_lag_samples} is not divisible by "
            f"bin_samples={bin_samples}."
        )
    num_half_bins = max_lag_samples // bin_samples
    num_bins = (2 * num_half_bins) + 1
    return max_lag_samples, bin_samples, num_bins


def _extract_spike_sample_indices_by_unit(
    data: EvalData | None,
    *,
    context: str,
) -> tuple[list[np.ndarray], int]:
    if data is None:
        raise ValueError(f"{context}: EvalData is required.")
    if data.duration is None:
        raise ValueError(f"{context}: EvalData.duration is required.")
    duration_samples_float = float(data.duration) * CCG_SAMPLING_RATE
    duration_samples = int(round(duration_samples_float))
    if not np.isclose(duration_samples, duration_samples_float):
        raise ValueError(
            f"{context}: duration {data.duration} does not align to the "
            f"{CCG_SAMPLING_RATE} Hz correlogram sampling grid."
        )
    if duration_samples <= 0:
        raise ValueError(f"{context}: duration must be positive, got {data.duration}.")

    if data.spike_times is not None:
        sample_indices_by_unit: list[np.ndarray] = []
        for unit_idx, unit_spike_times in enumerate(data.spike_times):
            unit_array = np.asarray(unit_spike_times, dtype=np.float64)
            if unit_array.ndim != 1:
                raise ValueError(
                    f"{context}: spike_times for unit {unit_idx} must be 1D, "
                    f"got {unit_array.shape}."
                )
            if np.any(~np.isfinite(unit_array)) or np.any(unit_array < 0):
                raise ValueError(
                    f"{context}: spike_times for unit {unit_idx} contain invalid values."
                )
            unit_samples = np.floor(unit_array * CCG_SAMPLING_RATE).astype(np.int64, copy=False)
            if np.any(unit_samples >= duration_samples):
                raise ValueError(
                    f"{context}: spike_times for unit {unit_idx} exceed duration {data.duration}s."
                )
            sample_indices_by_unit.append(unit_samples)
        return sample_indices_by_unit, duration_samples

    if data.binned_spikes is None or data.bin_size_seconds is None:
        raise ValueError(
            f"{context}: exact windowed CCG boundary handling requires spike_times or "
            "count-valued binned_spikes with bin_size_seconds."
        )

    counts = np.asarray(data.binned_spikes, dtype=np.float64)
    if counts.ndim != 2:
        raise ValueError(f"{context}: expected 2D binned_spikes, got {counts.shape}.")
    if np.any(~np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError(f"{context}: binned_spikes contain invalid values.")

    bin_size_samples_float = float(data.bin_size_seconds) * CCG_SAMPLING_RATE
    bin_size_samples = int(round(bin_size_samples_float))
    if not np.isclose(bin_size_samples, bin_size_samples_float) or bin_size_samples <= 0:
        raise ValueError(
            f"{context}: bin_size_seconds={data.bin_size_seconds} is incompatible with the "
            f"{CCG_SAMPLING_RATE} Hz correlogram sampling grid."
        )

    # Match the fine-bin SPIKES -> SPIKE_TIMES conversion used by the direct
    # correlogram path: each non-zero bin becomes one spike at the bin center.
    # Flooring the center time to the 1 kHz sample grid lands it back in the
    # original bin index, so one sample index per non-zero bin is the exact
    # boundary-carry representation for this path.
    if (
        data.data_type == DataType.SPIKES
        and float(data.bin_size_seconds) <= (1.0 / CCG_SAMPLING_RATE) + 1e-12
    ):
        nonzero_frac = float(np.mean(counts > 0))
        if nonzero_frac > 0.5:
            raise ValueError(
                f"{context}: fine-bin SPIKES are too dense ({nonzero_frac:.1%} non-zero bins) "
                "for exact spike-time boundary carry."
            )
        sample_indices_by_unit = [
            np.flatnonzero(unit_counts > 0).astype(np.int64) * bin_size_samples
            for unit_counts in counts
        ]
        return sample_indices_by_unit, duration_samples

    rounded_counts = np.rint(counts)
    if not np.allclose(counts, rounded_counts, atol=1e-9):
        raise ValueError(
            f"{context}: exact windowed CCG boundary handling requires integer-valued counts."
        )
    sample_indices_by_unit = [
        np.repeat(
            np.arange(unit_counts.shape[0], dtype=np.int64) * bin_size_samples,
            unit_counts.astype(np.int64, copy=False),
        )
        for unit_counts in rounded_counts
    ]
    return sample_indices_by_unit, duration_samples


def _extract_integer_binned_counts(
    data: EvalData | None,
    *,
    context: str,
) -> tuple[np.ndarray, float]:
    if data is None:
        raise ValueError(f"{context}: EvalData is required.")
    if data.binned_spikes is None or data.bin_size_seconds is None:
        raise ValueError(
            f"{context}: integer-valued binned_spikes with bin_size_seconds are required."
        )

    counts = np.asarray(data.binned_spikes, dtype=np.float64)
    if counts.ndim != 2:
        raise ValueError(f"{context}: expected 2D binned_spikes, got {counts.shape}.")
    if np.any(~np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError(f"{context}: binned_spikes contain invalid values.")

    rounded_counts = np.rint(counts)
    if not np.allclose(counts, rounded_counts, atol=1e-9):
        raise ValueError(f"{context}: exact boundary carry requires integer-valued counts.")
    return rounded_counts.astype(np.int64, copy=False), float(data.bin_size_seconds)


def _rebin_integer_counts(
    counts: np.ndarray,
    *,
    base_bin_size_seconds: float,
    target_bin_size_seconds: float,
    context: str,
) -> np.ndarray:
    if target_bin_size_seconds < base_bin_size_seconds - 1e-12:
        raise ValueError(
            f"{context}: target bin size must be >= source bin size for exact boundary carry."
        )
    if np.isclose(target_bin_size_seconds, base_bin_size_seconds):
        return counts

    ratio = target_bin_size_seconds / base_bin_size_seconds
    if not np.isclose(ratio, round(ratio)):
        raise ValueError(
            f"{context}: target bin size must be an integer multiple of the source bin size."
        )

    ratio_int = int(round(ratio))
    n_bins = counts.shape[1] // ratio_int
    truncated = counts[:, : n_bins * ratio_int]
    return truncated.reshape(counts.shape[0], n_bins, ratio_int).sum(axis=2)


def _extract_binned_boundary_counts(
    data: EvalData | None,
    *,
    max_lag_ms: float,
    bin_size_ms: float,
    context: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    counts, base_bin_size_seconds = _extract_integer_binned_counts(data, context=context)
    requested_bin_size_seconds = float(bin_size_ms) / 1000.0
    effective_bin_size_seconds = max(requested_bin_size_seconds, base_bin_size_seconds)
    rebinned_counts = _rebin_integer_counts(
        counts,
        base_bin_size_seconds=base_bin_size_seconds,
        target_bin_size_seconds=effective_bin_size_seconds,
        context=context,
    )
    effective_bin_size_ms = effective_bin_size_seconds * 1000.0
    max_lag_bins = int(float(max_lag_ms) / effective_bin_size_ms)
    if max_lag_bins <= 0:
        raise ValueError(
            f"{context}: max_lag_ms={max_lag_ms} and effective_bin_size_ms="
            f"{effective_bin_size_ms} yield no lag bins."
        )
    return rebinned_counts[:, :max_lag_bins], rebinned_counts[:, -max_lag_bins:], max_lag_bins


def _compute_cross_boundary_ccg_matrix_from_binned_counts(
    *,
    pair_order: tuple[tuple[int, int], ...],
    previous_tail_counts: np.ndarray,
    current_head_counts: np.ndarray,
    max_lag_bins: int,
) -> np.ndarray:
    boundary_matrix = np.zeros((len(pair_order), (2 * max_lag_bins) + 1), dtype=np.float64)
    for pair_idx, (unit_i, unit_j) in enumerate(pair_order):
        negative_lags = np.convolve(
            previous_tail_counts[unit_i, ::-1],
            current_head_counts[unit_j],
            mode="full",
        )[:max_lag_bins]
        positive_lags = np.convolve(
            previous_tail_counts[unit_j, ::-1],
            current_head_counts[unit_i],
            mode="full",
        )[:max_lag_bins]
        boundary_matrix[pair_idx, :max_lag_bins] = negative_lags[::-1]
        boundary_matrix[pair_idx, max_lag_bins + 1 :] = positive_lags
    return boundary_matrix


def _compute_cross_boundary_ccg_matrix(
    *,
    pair_order: tuple[tuple[int, int], ...],
    carry_by_unit: list[np.ndarray],
    current_by_unit: list[np.ndarray],
    previous_window_duration_samples: int,
    max_lag_samples: int,
    bin_samples: int,
) -> np.ndarray:
    num_half_bins = max_lag_samples // bin_samples
    boundary_matrix = np.zeros((len(pair_order), (2 * num_half_bins) + 1), dtype=np.float64)
    for pair_idx, (unit_i, unit_j) in enumerate(pair_order):
        prev_i = np.asarray(carry_by_unit[unit_i], dtype=np.int64)
        prev_j = np.asarray(carry_by_unit[unit_j], dtype=np.int64)
        curr_i = np.asarray(current_by_unit[unit_i], dtype=np.int64)
        curr_j = np.asarray(current_by_unit[unit_j], dtype=np.int64)
        if curr_i.size:
            curr_i = curr_i[curr_i < max_lag_samples]
        if curr_j.size:
            curr_j = curr_j[curr_j < max_lag_samples]

        if prev_i.size and curr_j.size:
            diffs = (previous_window_duration_samples - prev_i)[:, None] + curr_j[None, :]
            valid_diffs = diffs[diffs <= max_lag_samples]
            if valid_diffs.size:
                negative_bins = np.floor_divide(-valid_diffs, bin_samples) + num_half_bins
                np.add.at(boundary_matrix[pair_idx], negative_bins, 1.0)

        if prev_j.size and curr_i.size:
            diffs = (previous_window_duration_samples - prev_j)[:, None] + curr_i[None, :]
            valid_diffs = diffs[diffs <= max_lag_samples]
            if valid_diffs.size:
                positive_bins = np.floor_divide(valid_diffs, bin_samples) + num_half_bins
                np.add.at(boundary_matrix[pair_idx], positive_bins, 1.0)

    return boundary_matrix


def _tail_spike_samples_by_unit(
    spike_samples_by_unit: list[np.ndarray],
    *,
    window_duration_samples: int,
    max_lag_samples: int,
) -> list[np.ndarray]:
    tail_start = max(window_duration_samples - max_lag_samples, 0)
    return [
        np.asarray(unit_samples, dtype=np.int64)[
            np.asarray(unit_samples, dtype=np.int64) >= tail_start
        ]
        for unit_samples in spike_samples_by_unit
    ]


def _accumulate_ccg_boundary_terms(
    session_state: dict[str, Any],
    *,
    pair_order: tuple[tuple[int, int], ...],
    original_data: EvalData | None,
    reconstructed_data: EvalData | None,
    max_lag_ms: float,
    bin_size_ms: float,
    metric_name: str,
    strict_boundary_carry: bool,
) -> None:
    max_lag_samples, bin_samples, num_bins = _compute_ccg_window_geometry(
        max_lag_ms=max_lag_ms,
        bin_size_ms=bin_size_ms,
        metric_name=metric_name,
    )
    if (
        original_data is not None
        and reconstructed_data is not None
        and original_data.spike_times is None
        and reconstructed_data.spike_times is None
        and not (
            original_data.data_type == DataType.SPIKES
            and reconstructed_data.data_type == DataType.SPIKES
            and original_data.bin_size_seconds is not None
            and reconstructed_data.bin_size_seconds is not None
            and float(original_data.bin_size_seconds) <= (1.0 / CCG_SAMPLING_RATE) + 1e-12
            and float(reconstructed_data.bin_size_seconds) <= (1.0 / CCG_SAMPLING_RATE) + 1e-12
        )
    ):
        try:
            original_head_counts, original_tail_counts, original_max_lag_bins = (
                _extract_binned_boundary_counts(
                    original_data,
                    max_lag_ms=max_lag_ms,
                    bin_size_ms=bin_size_ms,
                    context=f"{metric_name} original",
                )
            )
            reconstructed_head_counts, reconstructed_tail_counts, reconstructed_max_lag_bins = (
                _extract_binned_boundary_counts(
                    reconstructed_data,
                    max_lag_ms=max_lag_ms,
                    bin_size_ms=bin_size_ms,
                    context=f"{metric_name} reconstructed",
                )
            )
        except ValueError:
            pass
        else:
            if (
                original_max_lag_bins != reconstructed_max_lag_bins
                or (2 * original_max_lag_bins) + 1 != num_bins
            ):
                raise ValueError(
                    f"{metric_name}: binned-count boundary geometry mismatch between original "
                    f"and reconstructed data ({original_max_lag_bins}, "
                    f"{reconstructed_max_lag_bins}, num_bins={num_bins})."
                )

            carry_mode = session_state.get("ccg_boundary_carry_mode")
            if carry_mode is None:
                session_state["ccg_boundary_carry_mode"] = "binned_counts"
            elif carry_mode != "binned_counts":
                raise ValueError(
                    f"{metric_name}: boundary carry mode changed from {carry_mode} to "
                    "'binned_counts' within one session."
                )

            previous_original_tail_counts = session_state.get("original_tail_counts")
            previous_reconstructed_tail_counts = session_state.get("reconstructed_tail_counts")
            if (
                previous_original_tail_counts is not None
                and previous_reconstructed_tail_counts is not None
            ):
                session_state["ccg_original_running_sum"] += (
                    _compute_cross_boundary_ccg_matrix_from_binned_counts(
                        pair_order=pair_order,
                        previous_tail_counts=np.asarray(
                            previous_original_tail_counts, dtype=np.int64
                        ),
                        current_head_counts=original_head_counts,
                        max_lag_bins=original_max_lag_bins,
                    )
                )
                session_state["ccg_reconstructed_running_sum"] += (
                    _compute_cross_boundary_ccg_matrix_from_binned_counts(
                        pair_order=pair_order,
                        previous_tail_counts=np.asarray(
                            previous_reconstructed_tail_counts,
                            dtype=np.int64,
                        ),
                        current_head_counts=reconstructed_head_counts,
                        max_lag_bins=reconstructed_max_lag_bins,
                    )
                )

            session_state["original_tail_counts"] = np.asarray(
                original_tail_counts, dtype=np.int64
            )
            session_state["reconstructed_tail_counts"] = np.asarray(
                reconstructed_tail_counts,
                dtype=np.int64,
            )
            session_state["ccg_boundary_carry_enabled"] = True
            return

    try:
        original_samples_by_unit, original_duration_samples = (
            _extract_spike_sample_indices_by_unit(
                original_data,
                context=f"{metric_name} original",
            )
        )
        reconstructed_samples_by_unit, reconstructed_duration_samples = (
            _extract_spike_sample_indices_by_unit(
                reconstructed_data,
                context=f"{metric_name} reconstructed",
            )
        )
    except ValueError as exc:
        if strict_boundary_carry:
            raise
        if "ccg_boundary_carry_enabled" not in session_state:
            warnings.warn(
                f"{metric_name}: exact cross-window CCG boundary handling is disabled: {exc}",
                RuntimeWarning,
                stacklevel=3,
            )
            session_state["ccg_boundary_carry_enabled"] = False
            session_state["ccg_boundary_carry_disable_reason"] = str(exc)
        return

    carry_mode = session_state.get("ccg_boundary_carry_mode")
    if carry_mode is None:
        session_state["ccg_boundary_carry_mode"] = "spike_samples"
    elif carry_mode != "spike_samples":
        raise ValueError(
            f"{metric_name}: boundary carry mode changed from {carry_mode} to "
            "'spike_samples' within one session."
        )

    if len(original_samples_by_unit) != len(reconstructed_samples_by_unit):
        raise ValueError(
            f"{metric_name}: unit count mismatch between original and reconstructed data: "
            f"{len(original_samples_by_unit)} vs {len(reconstructed_samples_by_unit)}."
        )

    running_sum_shape = session_state["ccg_original_running_sum"].shape
    expected_shape = (len(pair_order), num_bins)
    if running_sum_shape != expected_shape:
        raise ValueError(
            f"{metric_name}: running-sum CCG shape {running_sum_shape} does not match expected "
            f"shape {expected_shape} from lag geometry."
        )

    if "ccg_boundary_carry_enabled" not in session_state:
        session_state["ccg_boundary_carry_enabled"] = True
        session_state["ccg_max_lag_samples"] = max_lag_samples
        session_state["ccg_bin_samples"] = bin_samples
        session_state["carry_original_spike_samples_by_unit"] = _tail_spike_samples_by_unit(
            original_samples_by_unit,
            window_duration_samples=original_duration_samples,
            max_lag_samples=max_lag_samples,
        )
        session_state["carry_reconstructed_spike_samples_by_unit"] = _tail_spike_samples_by_unit(
            reconstructed_samples_by_unit,
            window_duration_samples=reconstructed_duration_samples,
            max_lag_samples=max_lag_samples,
        )
        session_state["previous_original_window_duration_samples"] = original_duration_samples
        session_state["previous_reconstructed_window_duration_samples"] = (
            reconstructed_duration_samples
        )
        return

    if not session_state["ccg_boundary_carry_enabled"]:
        return

    if session_state["ccg_max_lag_samples"] != max_lag_samples:
        raise ValueError(
            f"{metric_name}: max_lag_samples changed across windows: expected "
            f"{session_state['ccg_max_lag_samples']}, got {max_lag_samples}."
        )
    if session_state["ccg_bin_samples"] != bin_samples:
        raise ValueError(
            f"{metric_name}: bin_samples changed across windows: expected "
            f"{session_state['ccg_bin_samples']}, got {bin_samples}."
        )

    session_state["ccg_original_running_sum"] += _compute_cross_boundary_ccg_matrix(
        pair_order=pair_order,
        carry_by_unit=session_state["carry_original_spike_samples_by_unit"],
        current_by_unit=original_samples_by_unit,
        previous_window_duration_samples=session_state[
            "previous_original_window_duration_samples"
        ],
        max_lag_samples=max_lag_samples,
        bin_samples=bin_samples,
    )
    session_state["ccg_reconstructed_running_sum"] += _compute_cross_boundary_ccg_matrix(
        pair_order=pair_order,
        carry_by_unit=session_state["carry_reconstructed_spike_samples_by_unit"],
        current_by_unit=reconstructed_samples_by_unit,
        previous_window_duration_samples=session_state[
            "previous_reconstructed_window_duration_samples"
        ],
        max_lag_samples=max_lag_samples,
        bin_samples=bin_samples,
    )
    session_state["carry_original_spike_samples_by_unit"] = _tail_spike_samples_by_unit(
        original_samples_by_unit,
        window_duration_samples=original_duration_samples,
        max_lag_samples=max_lag_samples,
    )
    session_state["carry_reconstructed_spike_samples_by_unit"] = _tail_spike_samples_by_unit(
        reconstructed_samples_by_unit,
        window_duration_samples=reconstructed_duration_samples,
        max_lag_samples=max_lag_samples,
    )
    session_state["previous_original_window_duration_samples"] = original_duration_samples
    session_state["previous_reconstructed_window_duration_samples"] = (
        reconstructed_duration_samples
    )


def _accumulate_ccg_window_state(
    session_state: dict[str, Any],
    *,
    pair_order: tuple[tuple[int, int], ...],
    ccg_original_matrix: np.ndarray,
    ccg_reconstructed_matrix: np.ndarray,
    original_data: EvalData | None,
    reconstructed_data: EvalData | None,
    max_lag_ms: float,
    bin_size_ms: float,
    metric_name: str,
    strict_boundary_carry: bool,
) -> None:
    if "ccg_original_running_sum" not in session_state:
        session_state["ccg_original_running_sum"] = np.zeros_like(
            ccg_original_matrix,
            dtype=np.float64,
        )
        session_state["ccg_reconstructed_running_sum"] = np.zeros_like(
            ccg_reconstructed_matrix,
            dtype=np.float64,
        )
        session_state["pair_order"] = pair_order
        session_state["n_windows_seen"] = 0
    elif session_state["pair_order"] != pair_order:
        raise ValueError(f"{metric_name}._accumulate_window: pair order changed across windows.")

    session_state["ccg_original_running_sum"] += ccg_original_matrix
    session_state["ccg_reconstructed_running_sum"] += ccg_reconstructed_matrix
    _accumulate_ccg_boundary_terms(
        session_state,
        pair_order=pair_order,
        original_data=original_data,
        reconstructed_data=reconstructed_data,
        max_lag_ms=max_lag_ms,
        bin_size_ms=bin_size_ms,
        metric_name=metric_name,
        strict_boundary_carry=strict_boundary_carry,
    )
    session_state["n_windows_seen"] += 1


def _finalize_mean_ccg_state(
    session_state: dict[str, Any],
    *,
    metric_name: str,
) -> tuple[tuple[tuple[int, int], ...], np.ndarray, np.ndarray, int]:
    n_windows_seen = int(session_state.get("n_windows_seen", 0))
    if n_windows_seen == 0:
        raise ValueError(
            f"{metric_name}._finalize_session_state called with no accumulated windows."
        )

    mean_ccg_original = session_state["ccg_original_running_sum"] / n_windows_seen
    mean_ccg_reconstructed = session_state["ccg_reconstructed_running_sum"] / n_windows_seen

    nan_sources: list[str] = []
    if np.any(np.isnan(mean_ccg_original)):
        nan_sources.append("ccg_original_running_sum")
    if np.any(np.isnan(mean_ccg_reconstructed)):
        nan_sources.append("ccg_reconstructed_running_sum")
    if nan_sources:
        raise ValueError(
            f"{metric_name}._finalize_session_state: accumulated CCG running sum "
            f"contains NaN in {nan_sources}. Metric skipped."
        )

    return (
        session_state["pair_order"],
        mean_ccg_original,
        mean_ccg_reconstructed,
        n_windows_seen,
    )


def _ccg_dict_from_pair_matrix(
    pair_order: tuple[tuple[int, int], ...],
    ccg_matrix: np.ndarray,
    *,
    context: str,
) -> dict[tuple[int, int], np.ndarray]:
    if ccg_matrix.ndim != 2:
        raise ValueError(
            f"{context}: expected a 2D pair-by-lag CCG matrix, got {ccg_matrix.shape}."
        )
    if ccg_matrix.shape[0] != len(pair_order):
        raise ValueError(
            f"{context}: pair count mismatch between pair_order ({len(pair_order)}) "
            f"and CCG matrix ({ccg_matrix.shape[0]})."
        )
    return {
        pair: np.asarray(ccg_matrix[pair_idx], dtype=np.float64)
        for pair_idx, pair in enumerate(pair_order)
    }


class CCGCorrelationWindowedMixin(MetricMixinBase):
    """Session-aggregated CCG correlation: running mean CCG per pair, then Pearson."""

    aggregation_fallback_policy = "session_aggregated"

    def _accumulate_window(
        self,
        session_state: dict[str, Any],
        window_intermediates,
    ) -> None:
        requests = _get_ccg_requests(self, self.preferred_data_type)
        ccg_original = window_intermediates.get(requests["ccg_original"])
        ccg_reconstructed = window_intermediates.get(requests["ccg_reconstructed"])
        pair_order, ccg_original_matrix, ccg_reconstructed_matrix = _extract_ccg_pair_matrices(
            ccg_original,
            ccg_reconstructed,
            metric_name=type(self).__name__,
        )
        _accumulate_ccg_window_state(
            session_state,
            pair_order=pair_order,
            ccg_original_matrix=ccg_original_matrix,
            ccg_reconstructed_matrix=ccg_reconstructed_matrix,
            # ``getattr`` with None default so stub-style ``window_intermediates``
            # objects (which expose only ``.get()``) don't AttributeError before
            # the pair-order check; ``_accumulate_ccg_boundary_terms`` already
            # treats ``None`` as "skip cross-window carry for this window".
            original_data=getattr(window_intermediates, "original", None),
            reconstructed_data=getattr(window_intermediates, "reconstructed", None),
            max_lag_ms=float(self.params["correlogram_max_lag_ms"]),
            bin_size_ms=float(self.params["correlogram_bin_size_ms"]),
            metric_name=type(self).__name__,
            strict_boundary_carry=False,
        )

    def _finalize_session_state(self, session_state: dict[str, Any]) -> MetricResult:
        pair_order, mean_ccg_original, mean_ccg_reconstructed, n_windows_seen = (
            _finalize_mean_ccg_state(session_state, metric_name=type(self).__name__)
        )

        per_pair_corrs = np.full(mean_ccg_original.shape[0], np.nan, dtype=np.float64)
        for pair_idx, (orig, recon) in enumerate(
            zip(mean_ccg_original, mean_ccg_reconstructed, strict=False)
        ):
            if np.std(orig) > 0 and np.std(recon) > 0:
                per_pair_corrs[pair_idx] = float(np.corrcoef(orig, recon)[0, 1])

        if np.all(np.isnan(per_pair_corrs)):
            raise ValueError(
                "CCGCorrelation._finalize_session_state: every per-pair Pearson "
                "correlation is NaN (all pairs had zero variance or NaN accumulators)."
            )

        pair_correlations = {
            pair: float(per_pair_corrs[idx]) for idx, pair in enumerate(pair_order)
        }
        return self._make_result(
            raw_value=float(np.nanmean(per_pair_corrs)),
            metadata={
                "pair_correlations": pair_correlations,
                "n_windows_seen": n_windows_seen,
            },
        )


class FunctionalConnectivityF1WindowedMixin(MetricMixinBase):
    """Session-aggregated connectivity F1 via pooled-mean CCG fed back through compute()."""

    aggregation_fallback_policy = "session_aggregated"

    def _accumulate_window(
        self,
        session_state: dict[str, Any],
        window_intermediates,
    ) -> None:
        requests = _get_ccg_requests(self, self.preferred_data_type)
        ccg_original = window_intermediates.get(requests["ccg_original"])
        ccg_reconstructed = window_intermediates.get(requests["ccg_reconstructed"])
        pair_order, ccg_original_matrix, ccg_reconstructed_matrix = _extract_ccg_pair_matrices(
            ccg_original,
            ccg_reconstructed,
            metric_name="FunctionalConnectivityF1",
        )
        _accumulate_ccg_window_state(
            session_state,
            pair_order=pair_order,
            ccg_original_matrix=ccg_original_matrix,
            ccg_reconstructed_matrix=ccg_reconstructed_matrix,
            # ``getattr`` with None default so stub-style ``window_intermediates``
            # objects (which expose only ``.get()``) don't AttributeError before
            # the pair-order check; ``_accumulate_ccg_boundary_terms`` already
            # treats ``None`` as "skip cross-window carry for this window".
            original_data=getattr(window_intermediates, "original", None),
            reconstructed_data=getattr(window_intermediates, "reconstructed", None),
            max_lag_ms=float(self.params["correlogram_max_lag_ms"]),
            bin_size_ms=float(self.params["correlogram_bin_size_ms"]),
            metric_name="FunctionalConnectivityF1",
            strict_boundary_carry=True,
        )

    def _finalize_session_state(self, session_state: dict[str, Any]) -> MetricResult:
        pair_order, mean_ccg_original, mean_ccg_reconstructed, n_windows_seen = (
            _finalize_mean_ccg_state(
                session_state,
                metric_name="FunctionalConnectivityF1",
            )
        )
        result = self.compute(
            {
                "ccg_original": _ccg_dict_from_pair_matrix(
                    pair_order,
                    mean_ccg_original,
                    context="FunctionalConnectivityF1._finalize_session_state original",
                ),
                "ccg_reconstructed": _ccg_dict_from_pair_matrix(
                    pair_order,
                    mean_ccg_reconstructed,
                    context="FunctionalConnectivityF1._finalize_session_state reconstructed",
                ),
            }
        )
        result.metadata["n_windows_seen"] = n_windows_seen
        return result
