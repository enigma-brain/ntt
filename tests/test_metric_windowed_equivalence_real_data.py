"""Real-data windowed-vs-contiguous equivalence tests for latent_robustness (M2).

These tests need processed brainsets (perich_miller_population_2018). Point
``NTT_TEST_DATA_ROOT`` at the processed brainsets directory; they skip when no data
root is found.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import partial
from itertools import islice
from pathlib import Path

import numpy as np
import pytest
import torch

from ntt.contrib.tokenizer.poisson_baseline import (
    PoissonBaselineTokenizer,
    PoissonBaselineTokenizerSpec,
)
from ntt.core.components.types import DataType
from ntt.core.data import NTTDataModule, NTTDataModuleConfig
from ntt.core.data.components.config import DataLoaderConfig, SelectionConfig
from ntt.evaluation.core.intermediates import IntermediatesCache
from ntt.evaluation.core.types import EvalData
from ntt.evaluation.utils.validation_collation import (
    CollatedValidationSegment,
    _apply_pad_and_gap_mask,
    _concatenate_segments,
    _extract_scalar_sampling_rate,
    iter_collated_validation_segments,
)
from tests.jobs.m2_test_helpers import build_window_intermediates_cache
from tests.test_metric_windowed_equivalence import (
    APPROXIMATE_METRICS,
    CCG_BOUNDARY_RATIO_TOLERANCE,
    CCG_BOUNDARY_SENSITIVE_METRICS,
    EXACT_METRICS,
    MAIN_WINDOW_COUNT,
    SHORTER_WINDOW_COUNTS,
    _compute_metric_from_cache,
    _compute_windowed_scalar,
    _equivalence_rtol_for,
    _reduce_metric_result_to_scalar,
)

BIN_SIZE_SECONDS = 0.001
SESSION_BRAINSET = "perich_miller_population_2018"
SESSION_NAME = "c_20131022_center_out_reaching"
SESSION_ID = f"{SESSION_BRAINSET}/{SESSION_NAME}"
_ADJACENCY_TOLERANCE_SECONDS = 1e-12
_DATA_ROOT_CANDIDATES = [
    os.environ.get("NTT_TEST_DATA_ROOT", ""),
    os.path.join(os.environ.get("PROJECT_ROOT", "."), "data", "brainsets", "processed"),
]
TOKENIZER_SPEC = PoissonBaselineTokenizerSpec(
    window_length_seconds=1.0,
    bin_size_ms=1.0,
    rate_estimation_bin_ms=100.0,
)


def _find_data_root() -> str | None:
    for candidate in filter(None, _DATA_ROOT_CANDIDATES):
        path = Path(os.path.expanduser(candidate))
        if path.exists():
            return str(path)
    return None


_DATA_ROOT = _find_data_root()


def _to_numpy_tensor(tensor: torch.Tensor | None) -> np.ndarray | None:
    if tensor is None:
        return None
    return tensor.detach().cpu().numpy()


def _normalize_scalar(value) -> float:
    if isinstance(value, torch.Tensor):
        flat = value.detach().cpu().reshape(-1)
        assert flat.numel() == 1, f"Expected scalar tensor, got shape {tuple(value.shape)}"
        return float(flat.item())
    if isinstance(value, (tuple, list)):
        assert len(value) == 1, f"Expected single-item sequence, got {value!r}"
        return _normalize_scalar(value[0])
    if isinstance(value, np.ndarray):
        flat = value.reshape(-1)
        assert flat.size == 1, f"Expected scalar array, got shape {value.shape}"
        return float(flat[0])
    return float(value)


def _segment_to_cache(segment: CollatedValidationSegment) -> IntermediatesCache:
    original = _apply_pad_and_gap_mask(segment.original, segment.pad_and_gap_mask)
    reconstructed = _apply_pad_and_gap_mask(segment.reconstructed, segment.pad_and_gap_mask)
    assert original is not None
    assert reconstructed is not None

    latents_np: np.ndarray | list[np.ndarray] | None = None
    if isinstance(segment.latents, torch.Tensor):
        latents_np = _to_numpy_tensor(segment.latents)
    elif isinstance(segment.latents, list):
        latents_np = [
            item.detach().cpu().numpy() if isinstance(item, torch.Tensor) else np.asarray(item)
            for item in segment.latents
        ]

    original_data = _build_eval_data_with_spike_times(original)
    reconstructed_data = _build_eval_data_with_spike_times(reconstructed)
    return build_window_intermediates_cache(
        original=original_data,
        reconstructed=reconstructed_data,
        latents=latents_np,
        behavioral_data_token_res=segment.behavioral_data_token_res,
        behavioral_data_input_res=segment.behavioral_data_input_res,
        token_timestamps=_to_numpy_tensor(segment.token_timestamps),
        sampling_rate=_extract_scalar_sampling_rate(segment.metadata.get("sampling_rate")),
        segment_absolute_start=segment.segment_absolute_start,
        token_bin_size_ms=TOKENIZER_SPEC.token_bin_size_ms,
    )


def _build_eval_data_with_spike_times(spike_tensor: torch.Tensor) -> EvalData:
    spike_array = spike_tensor.detach().cpu().numpy().astype(np.float64, copy=False)
    if spike_array.ndim != 2:
        raise ValueError(f"Expected 2D spike tensor, got shape {spike_array.shape}.")
    if np.any(~np.isfinite(spike_array)) or np.any(spike_array < 0):
        raise ValueError("Spike tensor must be finite and non-negative.")

    rounded_counts = np.rint(spike_array).astype(np.int64)
    if not np.allclose(spike_array, rounded_counts, atol=1e-6):
        raise ValueError("Real-data equivalence test expects integer-valued fine-bin counts.")

    spike_times: list[np.ndarray] = []
    for unit_counts in rounded_counts:
        unit_times: list[np.ndarray] = []
        nonzero_bins = np.flatnonzero(unit_counts > 0)
        for bin_idx in nonzero_bins:
            count = int(unit_counts[bin_idx])
            within_bin = ((np.arange(count, dtype=np.float64) + 0.5) / count) * BIN_SIZE_SECONDS
            unit_times.append(bin_idx * BIN_SIZE_SECONDS + within_bin)
        if unit_times:
            spike_times.append(np.concatenate(unit_times))
        else:
            spike_times.append(np.empty(0, dtype=np.float64))

    return EvalData(
        spike_times=spike_times,
        binned_spikes=rounded_counts.astype(np.int64, copy=False),
        bin_size_seconds=BIN_SIZE_SECONDS,
        duration=rounded_counts.shape[1] * BIN_SIZE_SECONDS,
        n_units=rounded_counts.shape[0],
        data_type=DataType.SPIKES,
    )


def _canonicalize_group_ids(values: np.ndarray) -> np.ndarray:
    canonical = np.empty_like(values, dtype=np.int64)
    next_group_id = 0
    seen: dict[int, int] = {}
    for idx, value in enumerate(np.asarray(values).tolist()):
        group_id = int(value)
        canonical[idx] = seen.setdefault(group_id, next_group_id)
        if seen[group_id] == next_group_id:
            next_group_id += 1
    return canonical


@dataclass
class RealWindowSession:
    raw_batches: list
    tokenizer_fun: partial
    full_segment: CollatedValidationSegment
    combined_window_segment: CollatedValidationSegment
    window_segments: list[CollatedValidationSegment]
    max_abs_interbatch_gap_seconds: float
    _window_caches: list[IntermediatesCache] = field(repr=False)
    _prefix_cache: dict[int, tuple[IntermediatesCache, list[IntermediatesCache]]] = field(
        default_factory=dict,
        repr=False,
    )

    def get_prefix(self, n_windows: int) -> tuple[IntermediatesCache, list[IntermediatesCache]]:
        if n_windows in self._prefix_cache:
            return self._prefix_cache[n_windows]

        if n_windows == len(self.window_segments):
            contiguous_segment = self.full_segment
        else:
            contiguous_segment = _concatenate_segments(self.window_segments[:n_windows])
        value = (_segment_to_cache(contiguous_segment), self._window_caches[:n_windows])
        self._prefix_cache[n_windows] = value
        return value


@pytest.fixture(scope="module")
def real_perich_miller_window_session() -> RealWindowSession:
    assert _DATA_ROOT is not None

    selection = SelectionConfig(
        selection=[
            {
                "brainset": SESSION_BRAINSET,
                "sessions": [SESSION_NAME],
            }
        ],
        config={"ignore_splits": True},
    )
    datamodule_config = NTTDataModuleConfig(
        root=_DATA_ROOT,
        tokenizer_spec=TOKENIZER_SPEC,
        train_selections=[selection],
        val_selections=[selection],
        dataloader=DataLoaderConfig(batch_size=1, num_workers=0, pin_memory=False),
        seed=42,
    )
    datamodule = NTTDataModule(datamodule_config)
    datamodule.setup("fit")

    raw_batches = list(islice(datamodule.val_dataloader(), MAIN_WINDOW_COUNT))
    assert len(raw_batches) == MAIN_WINDOW_COUNT, (
        f"Expected {MAIN_WINDOW_COUNT} validation windows from {SESSION_ID}, "
        f"got {len(raw_batches)}."
    )

    session_ids = [raw_batch.session_id[0] for raw_batch in raw_batches]
    assert set(session_ids) == {SESSION_ID}, (
        f"Expected a single real session {SESSION_ID}, got {sorted(set(session_ids))}."
    )

    interbatch_gaps = [
        _normalize_scalar(next_batch.absolute_start)
        - _normalize_scalar(current_batch.absolute_end)
        for current_batch, next_batch in zip(raw_batches[:-1], raw_batches[1:], strict=False)
    ]
    max_abs_interbatch_gap_seconds = max((abs(gap) for gap in interbatch_gaps), default=0.0)
    assert max_abs_interbatch_gap_seconds <= _ADJACENCY_TOLERANCE_SECONDS, (
        f"Real-data test prefix for {SESSION_ID} is not perfectly adjacent: "
        f"max absolute gap {max_abs_interbatch_gap_seconds:.3e}s exceeds tolerance "
        f"{_ADJACENCY_TOLERANCE_SECONDS:.3e}s."
    )

    tokenizer = PoissonBaselineTokenizer(TOKENIZER_SPEC)
    tokenizer_fun = partial(tokenizer.reconstruct_from_input, return_latents=True)

    full_segments = list(
        iter_collated_validation_segments(
            raw_batches,
            tokenizer_fun,
            token_bin_size_ms=TOKENIZER_SPEC.token_bin_size_ms,
            max_windows=MAIN_WINDOW_COUNT,
            drop_overlapping_windows=True,
        )
    )
    assert len(full_segments) == 1
    full_segment = full_segments[0]
    assert full_segment.session_id == SESSION_ID
    assert isinstance(full_segment.latents, torch.Tensor)
    assert full_segment.latents.shape[0] == MAIN_WINDOW_COUNT

    window_segments = list(
        iter_collated_validation_segments(
            raw_batches,
            tokenizer_fun,
            token_bin_size_ms=TOKENIZER_SPEC.token_bin_size_ms,
            max_windows=1,
            drop_overlapping_windows=True,
        )
    )
    assert len(window_segments) == MAIN_WINDOW_COUNT
    assert {segment.session_id for segment in window_segments} == {SESSION_ID}

    combined_window_segment = _concatenate_segments(window_segments)
    assert combined_window_segment.segment_absolute_start == full_segment.segment_absolute_start
    assert torch.equal(combined_window_segment.original, full_segment.original)
    assert torch.equal(combined_window_segment.reconstructed, full_segment.reconstructed)
    assert torch.equal(combined_window_segment.latents, full_segment.latents)
    assert torch.equal(combined_window_segment.token_timestamps, full_segment.token_timestamps)
    assert torch.equal(combined_window_segment.token_start_times, full_segment.token_start_times)
    assert torch.equal(combined_window_segment.token_end_times, full_segment.token_end_times)

    if full_segment.behavioral_data_token_res is not None:
        assert combined_window_segment.behavioral_data_token_res is not None
        for key, expected_value in full_segment.behavioral_data_token_res.items():
            actual_value = combined_window_segment.behavioral_data_token_res[key]
            if key == "window_group_id_token_res":
                np.testing.assert_array_equal(
                    _canonicalize_group_ids(actual_value),
                    _canonicalize_group_ids(expected_value),
                    err_msg=(
                        "window_group_id_token_res changed only by label renumbering across "
                        "collation modes."
                    ),
                )
                continue
            np.testing.assert_array_equal(
                actual_value,
                expected_value,
                err_msg=f"behavioral_data_token_res[{key!r}] diverged across collation modes.",
            )
    if full_segment.behavioral_data_input_res is not None:
        assert combined_window_segment.behavioral_data_input_res is not None
        for key, expected_value in full_segment.behavioral_data_input_res.items():
            np.testing.assert_array_equal(
                combined_window_segment.behavioral_data_input_res[key],
                expected_value,
                err_msg=f"behavioral_data_input_res[{key!r}] diverged across collation modes.",
            )

    window_caches = [_segment_to_cache(segment) for segment in window_segments]
    return RealWindowSession(
        raw_batches=raw_batches,
        tokenizer_fun=tokenizer_fun,
        full_segment=full_segment,
        combined_window_segment=combined_window_segment,
        window_segments=window_segments,
        max_abs_interbatch_gap_seconds=max_abs_interbatch_gap_seconds,
        _window_caches=window_caches,
    )


@pytest.mark.skipif(
    _DATA_ROOT is None,
    reason=f"No data root found. Tried: {_DATA_ROOT_CANDIDATES}",
)
@pytest.mark.slow
def test_real_data_prefix_is_gap_free_and_identical_across_collation_modes(
    real_perich_miller_window_session: RealWindowSession,
) -> None:
    assert (
        real_perich_miller_window_session.max_abs_interbatch_gap_seconds
        <= _ADJACENCY_TOLERANCE_SECONDS
    )


@pytest.mark.skipif(
    _DATA_ROOT is None,
    reason=f"No data root found. Tried: {_DATA_ROOT_CANDIDATES}",
)
@pytest.mark.slow
@pytest.mark.parametrize(("metric_name", "metric_factory"), EXACT_METRICS)
def test_exact_windowed_metrics_match_contiguous_session_on_real_data(
    real_perich_miller_window_session: RealWindowSession,
    metric_name,
    metric_factory,
) -> None:
    metric = metric_factory()
    contiguous_cache, window_caches = real_perich_miller_window_session.get_prefix(
        MAIN_WINDOW_COUNT
    )
    contiguous_value = _reduce_metric_result_to_scalar(
        _compute_metric_from_cache(metric, contiguous_cache)
    )
    windowed_value = _compute_windowed_scalar(metric, window_caches)
    assert np.isfinite(contiguous_value), (
        f"{metric_name} contiguous real-data value should be finite."
    )
    assert np.isfinite(windowed_value), f"{metric_name} windowed real-data value should be finite."

    # CCG-based metrics carry an inherent max_lag/window mismatch — see
    # CCG_BOUNDARY_RATIO_TOLERANCE in test_metric_windowed_equivalence.py.
    # The new boundary-carry code closes that gap to ~1e-6 on real Perich
    # data, but the assertion form should reflect the underlying ratio.
    rtol = _equivalence_rtol_for(metric_name, default_rtol=1e-6)
    np.testing.assert_allclose(
        windowed_value,
        contiguous_value,
        rtol=rtol,
        atol=1e-9,
        err_msg=(
            f"{metric_name} should be mathematically equivalent on real-data "
            "contiguous vs windowed computation."
        ),
    )


@pytest.mark.skipif(
    _DATA_ROOT is None,
    reason=f"No data root found. Tried: {_DATA_ROOT_CANDIDATES}",
)
@pytest.mark.slow
@pytest.mark.parametrize(("metric_name", "metric_factory"), APPROXIMATE_METRICS)
def test_approximate_windowed_metrics_match_contiguous_session_at_600_windows_on_real_data(
    real_perich_miller_window_session: RealWindowSession,
    metric_name,
    metric_factory,
) -> None:
    relative_errors_by_window_count: dict[int, float] = {}
    contiguous_value_at_600 = None
    windowed_value_at_600 = None

    for n_windows in (*SHORTER_WINDOW_COUNTS, MAIN_WINDOW_COUNT):
        metric = metric_factory()
        contiguous_cache, window_caches = real_perich_miller_window_session.get_prefix(n_windows)
        contiguous_value = _reduce_metric_result_to_scalar(
            _compute_metric_from_cache(metric, contiguous_cache)
        )
        windowed_value = _compute_windowed_scalar(metric, window_caches)
        assert np.isfinite(contiguous_value), (
            f"{metric_name} contiguous real-data value should be finite at {n_windows} windows."
        )
        assert np.isfinite(windowed_value), (
            f"{metric_name} windowed real-data value should be finite at {n_windows} windows."
        )
        denominator = max(abs(contiguous_value), 1e-12)
        relative_errors_by_window_count[n_windows] = (
            abs(windowed_value - contiguous_value) / denominator
        )
        if n_windows == MAIN_WINDOW_COUNT:
            contiguous_value_at_600 = contiguous_value
            windowed_value_at_600 = windowed_value

    assert contiguous_value_at_600 is not None
    assert windowed_value_at_600 is not None
    np.testing.assert_allclose(
        windowed_value_at_600,
        contiguous_value_at_600,
        rtol=0.1,
        atol=1e-6,
        err_msg=(
            f"{metric_name} should be within 0.1 rtol at {MAIN_WINDOW_COUNT} contiguous "
            f"real-data windows. Observed relative errors by prefix size: "
            f"{relative_errors_by_window_count}"
        ),
    )
