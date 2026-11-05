"""Integration tests for the repeats-based SNR metric.

These tests cover realistic pipeline→intermediate→metric code paths that the
pure-numpy unit tests in ``test_repeats_snr.py`` miss:

1. Multi-element-batch ``sampling_rate`` extraction end-to-end (batch_size > 1).
2. Ragged ``segment.latents`` coming in as a Python list with unequal per-window
   token counts — what the real pipeline produces on boundary-straddle cases.
3. NaN-labeled tokens in ``behavioral_data_token_res`` (inter-stimulus blank
   tokens from the label-expansion padding).
4. Strict label uniformity: any mixed non-NaN labels within one window must
   raise immediately.

"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset
from torch_brain.data.dataset import DatasetIndex

from ntt.contrib.tokenizer.identity_tokenizer import IdentityTokenizerSpec
from ntt.core.components.types import BinnedSpikesInput
from ntt.core.data.datamodule import collate
from ntt.evaluation.core.intermediates import IntermediatesCache
from ntt.evaluation.core.types import DataType, EvalData
from ntt.evaluation.metrics.base import IntermediateRequest
from ntt.evaluation.metrics.repeats import RepeatsSNR
from ntt.evaluation.utils.validation_collation import (
    _iter_collated_validation_segments_with_per_stimulus_sampling,
)

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _wids_from_latents_list(latents_list) -> np.ndarray:
    """Build the token-res window_group_id array the collation pipeline would attach.

    One entry per token; entries in a given window all carry that window's index.

    """
    return np.concatenate(
        [
            np.full(np.asarray(arr).shape[0], i, dtype=np.int32)
            for i, arr in enumerate(latents_list)
        ]
    )


def _make_eval_data(n_units: int = 2, n_bins: int = 10) -> EvalData:
    """Tiny placeholder EvalData for intermediates that only consume latents."""
    arr = np.zeros((n_units, n_bins), dtype=np.float32)
    return EvalData(
        binned_spikes=arr,
        bin_size_seconds=0.001,
        duration=n_bins * 0.001,
        n_units=n_units,
        data_type=DataType.SPIKES,
    )


# ======================================================================
# Test 1: Multi-element-batch sampling_rate extraction (end-to-end per-stim path)
# ======================================================================


def test_multi_element_batch_sampling_rate_extraction_e2e():
    """Guard against ``float(batch.sampling_rate)`` failing when batch_size > 1.

    The per-stimulus SNR path extracts one scalar sampling rate per sample
    from collated batches whose ``sampling_rate`` may be a tensor of shape
    ``[B]``. Calling ``float(batch.sampling_rate)`` directly on
    such a tensor would raise:
        ``ValueError: only one element tensors can be converted to Python scalars``.

    We verify the per-stimulus iterator runs end-to-end with ``batch_size=2``,
    per-batch ``sampling_rate`` shape ``[2]``, and two sessions with multiple
    stimulus windows each.
    """
    rng = np.random.default_rng(0)
    n_units = 2
    sampling_rate = 10.0

    class _StimulusDataset(Dataset):
        def __init__(self) -> None:
            self._recording_objects = {
                "sessA": SimpleNamespace(
                    stimuli=SimpleNamespace(
                        start=np.asarray([0.0, 1.0], dtype=np.float64),
                        end=np.asarray([0.9, 1.9], dtype=np.float64),
                        image_id=np.asarray([1, 2], dtype=np.int64),
                    )
                ),
                "sessB": SimpleNamespace(
                    stimuli=SimpleNamespace(
                        start=np.asarray([0.0, 1.0], dtype=np.float64),
                        end=np.asarray([0.9, 1.9], dtype=np.float64),
                        image_id=np.asarray([3, 4], dtype=np.int64),
                    )
                ),
            }
            self._spikes = {
                session_id: torch.from_numpy(
                    rng.integers(0, 3, size=(20, n_units)).astype(np.int64)
                )
                for session_id in self._recording_objects
            }

        def __len__(self) -> int:
            return len(self._recording_objects)

        def _get_data_object(self, recording_id: str):
            return self._recording_objects[recording_id]

        def __getitem__(self, index: DatasetIndex | int) -> BinnedSpikesInput:
            if isinstance(index, DatasetIndex):
                session_id = index.recording_id
                start = float(index.start)
                end = float(index.end)
            else:
                session_id = list(self._recording_objects.keys())[index]
                start = 0.0
                end = 2.0

            start_bin = int(round(start * sampling_rate))
            end_bin = int(round(end * sampling_rate))
            spike_matrix = self._spikes[session_id][start_bin:end_bin].contiguous()
            return BinnedSpikesInput(
                spike_matrix=spike_matrix,
                pad_and_gap_mask=torch.ones_like(spike_matrix, dtype=torch.bool),
                sampling_rate=sampling_rate,
                unit_ids=torch.arange(n_units, dtype=torch.int64),
                unit_ids_pad_mask=torch.ones(n_units, dtype=torch.bool),
                duration_sec=torch.tensor(end - start, dtype=torch.float32),
                session_id=session_id,
                subject_id="subj",
                species="mouse",
                absolute_start=torch.tensor(start, dtype=torch.float64),
                absolute_end=torch.tensor(end, dtype=torch.float64),
                behavioral_data=None,
                behavioral_intervals=None,
            )

    dataset = _StimulusDataset()
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate)
    spec = IdentityTokenizerSpec(window_length_seconds=1.0, bin_size_ms=100.0)

    def tokenizer_fun(batch: BinnedSpikesInput) -> dict:
        """Trivial tokenizer: one token per input bin."""
        B = batch.spike_matrix.shape[0]
        T = batch.spike_matrix.shape[1]
        # Token bin size 20ms == 2 input bins at sr=10Hz? That's 200ms.
        # Use 100ms tokens at 10Hz sampling → 1 input bin per token to keep
        # math simple and token_bin_size_ms divides evenly.
        # But spec asks for 20ms; at sr=10Hz that does not divide evenly.
        # We set token_bin_size_ms=100 to satisfy the internal assertion
        # while still exercising the multi-element sampling_rate path.
        abs_start = batch.absolute_start.to(dtype=torch.float64)
        idx = torch.arange(T, dtype=torch.float64)
        token_start = abs_start[:, None] + idx[None, :] * 0.1
        token_end = token_start + 0.1
        token_center = token_start + 0.05
        D = 3
        return {
            "reconstruction": batch.spike_matrix,
            "latents": torch.from_numpy(rng.standard_normal(size=(B, T, D)).astype(np.float32)),
            "token_timestamps": token_center,
            "token_start_time": token_start,
            "token_end_time": token_end,
        }

    segments = list(
        _iter_collated_validation_segments_with_per_stimulus_sampling(
            loader,
            tokenizer_fun,
            tokenizer_spec=spec,
            interval_source="stimuli",
            grouping_field="image_id",
            fixed_window_duration_ms=1000.0,
            label_extension_ms=0.0,
            device=torch.device("cpu"),
        )
    )

    # One segment per session (two sessions).
    by_session = {s.session_id: s for s in segments}
    assert set(by_session) == {"sessA", "sessB"}, (
        f"Expected segments for both sessions, got {set(by_session)}"
    )

    # Per-session geometry: each session has two 1 s stimulus windows; at
    # 100 ms tokens that is 10 tokens/window, D=3. The scalar sampling_rate
    # must be extracted per sample for this to come out right — a wrong-element
    # extraction or mis-grouping would corrupt the shapes/labels below.
    n_windows, tokens_per_window, latent_dim = 2, 10, 3
    expected_token_starts = np.round(np.arange(0.0, 2.0, 0.1), 6)
    expected_wgid = np.array([0] * tokens_per_window + [1] * tokens_per_window)
    # Each session's two stimulus windows carry these image_ids, in order.
    session_to_image_ids = {"sessA": (1.0, 2.0), "sessB": (3.0, 4.0)}

    for session_id, segment in by_session.items():
        assert segment.latents is not None
        assert tuple(np.asarray(segment.latents).shape) == (
            n_windows,
            tokens_per_window,
            latent_dim,
        ), f"{session_id}: unexpected latent geometry {segment.latents.shape}"

        token_starts = np.asarray(segment.token_start_times, dtype=np.float64)
        assert token_starts.shape == (n_windows * tokens_per_window,)
        np.testing.assert_allclose(np.round(token_starts, 6), expected_token_starts, atol=1e-6)
        # Strictly increasing token starts (windows concatenated in time order).
        assert np.all(np.diff(token_starts) > 0), f"{session_id}: token starts not monotonic"

        wgid = np.asarray(segment.behavioral_data_token_res["window_group_id_token_res"])
        np.testing.assert_array_equal(wgid, expected_wgid)

        # Per-window image_id grouping: window 0 and window 1 carry the two
        # distinct stimulus ids for THIS session (proves correct grouping, not
        # just "labels present"). Inter-stimulus blank tokens are NaN.
        image_ids = np.asarray(segment.behavioral_data_token_res["image_id"], dtype=np.float64)
        first_id, second_id = session_to_image_ids[session_id]
        w0 = image_ids[wgid == 0]
        w1 = image_ids[wgid == 1]
        assert set(np.unique(w0[~np.isnan(w0)]).tolist()) == {first_id}, (
            f"{session_id}: window 0 labels {np.unique(w0)} != {{{first_id}}}"
        )
        assert set(np.unique(w1[~np.isnan(w1)]).tolist()) == {second_id}, (
            f"{session_id}: window 1 labels {np.unique(w1)} != {{{second_id}}}"
        )


# ======================================================================
# Test 2: Ragged segment.latents (list input to IntermediatesCache)
# ======================================================================


def test_ragged_latents_list_input_to_cache():
    """Guard against the list-input code path for ``latents``.

    Ragged windows (varying token counts per logical window) are what the real
    pipeline produces on interval splits. This test feeds a Python list of
    [T_i, D] arrays with unequal T_i directly into IntermediatesCache and
    verifies that ``stimulus_grouped_latents`` cuts to the smallest T_i,
    emits a UserWarning, and that ``RepeatsSNR`` can consume the result.

    """
    rng = np.random.default_rng(1)
    D_latent = 4
    T_per_window = [200, 210, 205, 195, 200, 200, 205, 210, 195, 205]
    N_windows = len(T_per_window)
    assert N_windows >= 10

    # 3 stimuli, with 3/3/4 repeats respectively — matches N_windows = 10.
    stim_ids_per_window = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2], dtype=np.int64)
    assert len(stim_ids_per_window) == N_windows

    # Build the ragged latents list and the per-token grouping field.
    latents_list: list[np.ndarray] = []
    image_id_chunks: list[np.ndarray] = []
    for T_i, sid in zip(T_per_window, stim_ids_per_window, strict=False):
        latents_list.append(rng.standard_normal(size=(T_i, D_latent)).astype(np.float64))
        image_id_chunks.append(np.full(T_i, float(sid), dtype=np.float64))
    image_id_token_res = np.concatenate(image_id_chunks, axis=0)

    # Anti-cheat: ensure we really have unequal shapes and a list, not an
    # array that happens to be object-dtype.
    assert isinstance(latents_list, list)
    shapes = {arr.shape for arr in latents_list}
    assert len(shapes) > 1, "Test requires unequal per-window token counts."

    eval_data = _make_eval_data(n_units=2, n_bins=max(N_windows, 2))
    cache = IntermediatesCache(
        original=eval_data,
        reconstructed=eval_data,
        latents=latents_list,  # <-- passed as a list, NOT an ndarray.
        behavioral_data_token_res={
            "image_id": image_id_token_res,
            "window_group_id_token_res": _wids_from_latents_list(latents_list),
        },
    )

    req = IntermediateRequest.from_dict("stimulus_grouped_latents", {"grouping_field": "image_id"})
    with pytest.warns(UserWarning, match=r"different token counts"):
        result = cache.get(req)

    T_min = min(T_per_window)
    D_flat_after_cut = T_min * D_latent

    assert result["latents"].shape == (3, 4, D_flat_after_cut), (
        f"Expected dense shape (3, R_max=4, {D_flat_after_cut}), got {result['latents'].shape}"
    )

    assert result["n_tokens_cut_info"] is not None, (
        "n_tokens_cut_info should be populated for ragged inputs."
    )
    assert result["n_tokens_cut_info"]["n_tokens_used"] == T_min
    assert result["n_tokens_cut_info"]["n_windows_dropped"] == 0
    np.testing.assert_array_equal(np.sort(result["stimulus_ids"]), np.array([0.0, 1.0, 2.0]))
    assert int(result["repeat_counts"].sum()) == N_windows

    metric = RepeatsSNR()
    metric_result = metric.compute({"stimulus_grouped_latents": result})
    assert np.isfinite(metric_result.raw_value)
    assert metric_result.raw_value >= 0.0


def test_short_window_filter_does_not_inflate_nan_drop_stats():
    """NaN-drop stats must count only windows that survive the short-window
    filter. A window with many NaN tokens becomes short after the NaN drop and
    is discarded by the short-outlier filter; its NaN-drop count must not leak
    into the reported totals (no stale n_nan_dropped_per_window)."""
    rng = np.random.default_rng(7)
    D_latent = 3
    n_real = 200

    # Four kept windows of 200 real tokens (labels 1..4); window 0 also carries
    # 2 NaN tokens (so it has genuine NaN drops but stays long). One short
    # window (label 5) has only 5 real tokens behind 195 NaN tokens → after the
    # NaN drop it is 5 tokens long and the short-outlier filter discards it.
    specs = [
        (1.0, n_real, 2),  # kept, 2 NaN drops
        (2.0, n_real, 0),  # kept
        (3.0, n_real, 0),  # kept
        (4.0, n_real, 0),  # kept
        (5.0, 5, 195),  # short after drop → discarded; 195 NaN drops must NOT count
    ]

    latents_list: list[np.ndarray] = []
    image_id_chunks: list[np.ndarray] = []
    for label, n_real_i, n_nan_i in specs:
        T_i = n_real_i + n_nan_i
        latents_list.append(rng.standard_normal(size=(T_i, D_latent)).astype(np.float64))
        image_id_chunks.append(
            np.concatenate(
                [
                    np.full(n_real_i, label, dtype=np.float64),
                    np.full(n_nan_i, np.nan, dtype=np.float64),
                ]
            )
        )
    image_id_token_res = np.concatenate(image_id_chunks, axis=0)

    eval_data = _make_eval_data(n_units=2, n_bins=8)
    cache = IntermediatesCache(
        original=eval_data,
        reconstructed=eval_data,
        latents=latents_list,
        behavioral_data_token_res={
            "image_id": image_id_token_res,
            "window_group_id_token_res": _wids_from_latents_list(latents_list),
        },
    )
    req = IntermediateRequest.from_dict("stimulus_grouped_latents", {"grouping_field": "image_id"})

    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", UserWarning)
        result = cache.get(req)

    # The short window was discarded by the short-outlier filter.
    assert result["short_window_info"] is not None
    assert result["short_window_info"]["n_windows_dropped"] == 1

    # Only the surviving window 0 (2 NaN drops) should be reflected — NOT the
    # 195 NaN drops of the discarded short window.
    tdi = result["tokens_dropped_info"]
    assert tdi is not None
    assert tdi["n_nan_tokens_dropped_total"] == 2, (
        f"stale short-window NaN drops leaked: {tdi['n_nan_tokens_dropped_total']}"
    )
    assert tdi["n_windows_with_nan_drops"] == 1
    assert tdi["max_nan_dropped_per_window"] == 2


# ======================================================================
# Test 3: NaN image_id tokens within windows
# ======================================================================


def test_nan_image_id_tokens_dropped_silently():
    """Guard against NaN-labeled tokens leaking into the metric.

    The label-expansion step in the pipeline pads stimuli with an
    ``end_offset = 0.2s`` margin. Tokens falling outside the expanded
    stimulus intervals get NaN labels. The intermediate must drop those
    silently (not count them as mismatched) and keep the windows.

    """
    rng = np.random.default_rng(2)
    D_latent = 3
    N_windows = 10
    # 3 stimuli with 3/3/4 repeats (same setup as test 2 for comparability).
    stim_ids_per_window = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2], dtype=np.int64)

    # Each window: 180 real-label tokens, then 20..30 NaN tokens at tail
    # (varying so the windows are also ragged — tests both paths at once).
    n_real_per_window = 180
    n_nan_per_window = [20, 25, 30, 22, 28, 20, 25, 30, 22, 20]
    assert len(n_nan_per_window) == N_windows

    latents_list: list[np.ndarray] = []
    image_id_chunks: list[np.ndarray] = []
    total_nan_injected = 0
    for n_nan, sid in zip(n_nan_per_window, stim_ids_per_window, strict=False):
        T_i = n_real_per_window + n_nan
        latents_list.append(rng.standard_normal(size=(T_i, D_latent)).astype(np.float64))
        chunk = np.concatenate(
            [
                np.full(n_real_per_window, float(sid), dtype=np.float64),
                np.full(n_nan, np.nan, dtype=np.float64),
            ]
        )
        image_id_chunks.append(chunk)
        total_nan_injected += n_nan
    image_id_token_res = np.concatenate(image_id_chunks, axis=0)
    assert total_nan_injected > 0  # anti-cheat

    eval_data = _make_eval_data(n_units=2, n_bins=max(N_windows, 2))
    cache = IntermediatesCache(
        original=eval_data,
        reconstructed=eval_data,
        latents=latents_list,
        behavioral_data_token_res={
            "image_id": image_id_token_res,
            "window_group_id_token_res": _wids_from_latents_list(latents_list),
        },
    )

    req = IntermediateRequest.from_dict("stimulus_grouped_latents", {"grouping_field": "image_id"})
    # Warning from ragged-after-drop — inner token counts still differ a bit.
    # We don't require UserWarning here (drops might equalize counts in some
    # configs), so just run the computation.
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", UserWarning)
        result = cache.get(req)

    tdi = result["tokens_dropped_info"]
    assert tdi is not None
    assert tdi["n_nan_tokens_dropped_total"] == total_nan_injected, (
        f"Expected {total_nan_injected} NaN tokens dropped, got "
        f"{tdi['n_nan_tokens_dropped_total']}"
    )
    # Mixed non-NaN labels raise (strict uniformity) instead of dropping the
    # minority, so there is no "mismatched-drop" bookkeeping.
    assert "n_mismatched_tokens_dropped_total" not in tdi
    assert "n_windows_with_mismatched_drops" not in tdi
    assert "max_mismatched_dropped_per_window" not in tdi
    # All windows should be kept (no windows dropped from tokens-too-few).
    assert int(result["repeat_counts"].sum()) == N_windows

    metric = RepeatsSNR()
    metric_result = metric.compute({"stimulus_grouped_latents": result})
    assert np.isfinite(metric_result.raw_value)


def test_window_with_zero_non_nan_tokens_dropped_silently():
    """All-NaN windows are dropped silently.

    The intermediate does not raise on such windows. They are
    dropped, and the count is reported via
    ``tokens_dropped_info["n_all_nan_windows_dropped"]``.

    """
    rng = np.random.default_rng(3)
    D_latent = 2
    # 2 labelled windows (needed — else grouping has only NaN window), plus
    # one all-NaN window.
    T_good = 5
    T_bad = 4
    latents_list = [
        rng.standard_normal(size=(T_good, D_latent)).astype(np.float64),
        rng.standard_normal(size=(T_good, D_latent)).astype(np.float64),
        rng.standard_normal(size=(T_bad, D_latent)).astype(np.float64),
    ]
    image_id_token_res = np.concatenate(
        [
            np.full(T_good, 0.0, dtype=np.float64),
            np.full(T_good, 1.0, dtype=np.float64),
            np.full(T_bad, np.nan, dtype=np.float64),
        ]
    )

    eval_data = _make_eval_data()
    cache = IntermediatesCache(
        original=eval_data,
        reconstructed=eval_data,
        latents=latents_list,
        behavioral_data_token_res={
            "image_id": image_id_token_res,
            "window_group_id_token_res": _wids_from_latents_list(latents_list),
        },
    )
    req = IntermediateRequest.from_dict("stimulus_grouped_latents", {"grouping_field": "image_id"})
    # Suppress the ragged-token-count warning from cut-to-smallest; it's
    # incidental to this test.
    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", UserWarning)
        result = cache.get(req)
    tdi = result["tokens_dropped_info"]
    assert tdi is not None
    assert tdi["n_all_nan_windows_dropped"] == 1
    # Only the 2 labelled windows remain; each stimulus has exactly 1 repeat.
    assert int(result["repeat_counts"].sum()) == 2


def test_all_windows_all_nan_still_raises():
    """If EVERY window is all-NaN, we still raise (nothing left to label).

    Differs from the per-window drop-and-count behaviour: if we end up with
    zero kept windows, there's no metric to compute.

    """
    T = 4
    D_latent = 2
    latents_list = [
        np.zeros((T, D_latent), dtype=np.float64),
        np.zeros((T, D_latent), dtype=np.float64),
    ]
    image_id_token_res = np.full(2 * T, np.nan, dtype=np.float64)
    eval_data = _make_eval_data()
    cache = IntermediatesCache(
        original=eval_data,
        reconstructed=eval_data,
        latents=latents_list,
        behavioral_data_token_res={
            "image_id": image_id_token_res,
            "window_group_id_token_res": _wids_from_latents_list(latents_list),
        },
    )
    req = IntermediateRequest.from_dict("stimulus_grouped_latents", {"grouping_field": "image_id"})
    with pytest.raises(ValueError, match=r"no stimulus"):
        cache.get(req)


# ======================================================================
# Test 4: Strict label uniformity
# ======================================================================


def test_4a_single_mismatched_token_raises_under_strict_uniformity():
    """Any mixed non-NaN labels within one window must raise immediately."""
    rng = np.random.default_rng(4)
    D_latent = 2
    T_per_window = 6  # 5 majority + 1 straddler
    # 2 stimuli, 3 repeats each → 6 windows.
    majority_per_window = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    neighbours = np.array([1, 1, 1, 0, 0, 0], dtype=np.int64)

    latents_list: list[np.ndarray] = []
    image_id_chunks: list[np.ndarray] = []
    for maj, other in zip(majority_per_window, neighbours, strict=False):
        latents_list.append(rng.standard_normal(size=(T_per_window, D_latent)).astype(np.float64))
        chunk = np.array([float(maj)] * (T_per_window - 1) + [float(other)], dtype=np.float64)
        image_id_chunks.append(chunk)
    image_id_token_res = np.concatenate(image_id_chunks, axis=0)

    eval_data = _make_eval_data()
    cache = IntermediatesCache(
        original=eval_data,
        reconstructed=eval_data,
        latents=latents_list,
        behavioral_data_token_res={
            "image_id": image_id_token_res,
            "window_group_id_token_res": _wids_from_latents_list(latents_list),
        },
    )
    req = IntermediateRequest.from_dict("stimulus_grouped_latents", {"grouping_field": "image_id"})
    with pytest.raises(ValueError, match=r"multiple non-NaN 'image_id' labels"):
        cache.get(req)


def test_4b_three_or_more_mismatched_tokens_raise():
    """Large mixed-label windows should raise the same strict-uniformity error."""
    rng = np.random.default_rng(5)
    D_latent = 2
    # Build a window with [A, A, A, A, B, B, B] — A wins majority (4) but 3
    # tokens of B are too many.
    good = np.array([0.0] * 5, dtype=np.float64)
    bad_window_chunk = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float64)

    latents_list = [
        rng.standard_normal(size=(5, D_latent)).astype(np.float64),  # window 0
        rng.standard_normal(size=(5, D_latent)).astype(np.float64),  # window 1
        rng.standard_normal(size=(7, D_latent)).astype(np.float64),  # window 2 (bad)
    ]
    image_id_token_res = np.concatenate(
        [
            good,
            np.full(5, 1.0, dtype=np.float64),  # window 1 — pure stimulus 1
            bad_window_chunk,
        ]
    )
    eval_data = _make_eval_data()
    cache = IntermediatesCache(
        original=eval_data,
        reconstructed=eval_data,
        latents=latents_list,
        behavioral_data_token_res={
            "image_id": image_id_token_res,
            "window_group_id_token_res": _wids_from_latents_list(latents_list),
        },
    )
    req = IntermediateRequest.from_dict("stimulus_grouped_latents", {"grouping_field": "image_id"})
    with pytest.raises(ValueError, match=r"multiple non-NaN 'image_id' labels"):
        cache.get(req)
