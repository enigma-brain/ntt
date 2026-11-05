"""Iterator-level tests: multi-interval sessions are collated, not dropped.

A session whose sampling domain has several intervals (e.g. one per trial)
produces time gaps between buffered windows. ``iter_collated_validation_segments``
must not drop the whole session at the first gap; it yields one segment
whose binned data, token times and behavior stay aligned window by window.
"""

import logging
from typing import Any

import numpy as np
import pytest
import torch
from temporaldata import Interval
from torch.utils.data import DataLoader, Dataset
from torch_brain.data.dataset import DatasetIndex
from torch_brain.data.sampler import SequentialFixedWindowSampler

from ntt.core.components.types import BinnedSpikesInput
from ntt.core.data.datamodule import collate
from ntt.evaluation.metrics.downstream_prediction_utils import (
    align_behavioral_to_chunks,
    chunk_by_temporal_window,
)
from ntt.evaluation.utils.validation_collation import iter_collated_validation_segments

SESSION = "brainset/multi_interval_session"
SAMPLING_RATE = 1000.0
WINDOW_S = 0.1
TOKEN_MS = 20.0
N_UNITS = 3
# Interval lengths that are not window multiples, so the sampler adds an
# overlapping tail window (end - W, end) per interval.
INTERVALS = [(0.0, 0.35), (1.0, 1.25), (2.0, 2.4)]


class _MultiIntervalDataset(Dataset):
    """Spike value = absolute bin index, behavior = bin-center time, trial = interval.

    ``domain_gaps`` marks a session's bins invalid (a recording-domain gap inside
    its sampling interval, as in flint monkey_c_e4); a window whose trailing bins
    fall in one is truncated, with ``absolute_end`` at the last valid bin.
    Behavior is binned over the whole window (as upstream binning does, padded
    bins included); ``raw_behavior`` sessions instead return unbinned rows
    (fewer rows than bins) on every other window, like churchland's irregular
    cursor samples. ``clipped`` maps a session to ``(window_start, end)``: that
    window ends at ``end`` (its partial last bin invalid) and is padded to the
    full window length with invalid bins, while its behavior is binned over its
    own span only (no padding rows).
    """

    def __init__(
        self,
        sessions: dict[str, list[tuple[float, float]]] | None = None,
        domain_gaps: dict[str, list[tuple[float, float]]] | None = None,
        raw_behavior: frozenset[str] = frozenset(),
        clipped: dict[str, tuple[float, float]] | None = None,
        with_trial_id: bool = True,
    ):
        self.sessions = sessions or {SESSION: INTERVALS}
        self.domain_gaps = domain_gaps or {}
        self.raw_behavior = raw_behavior
        self.clipped = clipped or {}
        self.with_trial_id = with_trial_id

    def __getitem__(self, index: DatasetIndex) -> BinnedSpikesInput:
        sid = index.recording_id
        window_end = index.end
        clip = self.clipped.get(sid)
        if clip is not None and abs(index.start - clip[0]) < 1e-9:
            window_end = clip[1]
        start_bin = int(round(index.start * SAMPLING_RATE))
        n_bins = int(round((window_end - index.start) * SAMPLING_RATE))
        bins = np.arange(start_bin, start_bin + n_bins)
        centers = (bins + 0.5) / SAMPLING_RATE
        valid = np.ones(n_bins, dtype=bool)
        for gap_start, gap_end in self.domain_gaps.get(sid, []):
            valid &= ~((centers >= gap_start) & (centers < gap_end))
        if index.start + n_bins / SAMPLING_RATE > window_end + 1e-12:
            valid[-1] = False  # partial last bin
        absolute_end = window_end
        if valid[0] and not valid[-1] and window_end == index.end:
            absolute_end = index.start + (np.flatnonzero(valid)[-1] + 1) / SAMPLING_RATE
        n_rows = n_bins
        n_padded = int(round((index.end - index.start) * SAMPLING_RATE))
        padded_bins = np.concatenate([bins, np.zeros(n_padded - n_bins, dtype=bins.dtype)])
        valid = np.concatenate([valid, np.zeros(n_padded - n_bins, dtype=bool)])
        spike_matrix = torch.from_numpy(np.repeat(padded_bins[:, None], N_UNITS, axis=1)).to(
            torch.int32
        )
        mask = torch.from_numpy(np.repeat(valid[:, None], N_UNITS, axis=1)).to(torch.int32)
        trial = next(
            i for i, (s, e) in enumerate(self.sessions[sid]) if s - 1e-9 <= index.start < e
        )
        if sid in self.raw_behavior and (start_bin // n_bins) % 2 == 1:
            n_rows -= 3
        return BinnedSpikesInput(
            spike_matrix=spike_matrix * mask,
            pad_and_gap_mask=mask,
            sampling_rate=SAMPLING_RATE,
            unit_ids=torch.arange(N_UNITS, dtype=torch.int32),
            unit_ids_pad_mask=torch.ones(N_UNITS, dtype=torch.int32),
            duration_sec=torch.tensor(n_bins / SAMPLING_RATE),
            session_id=sid,
            subject_id="subject",
            species="species",
            absolute_start=torch.tensor(index.start, dtype=torch.float64),
            absolute_end=torch.tensor(absolute_end, dtype=torch.float64),
            behavioral_data={
                "cursor_pos": centers[:n_rows, None],
                **(
                    {"trial_id": np.full(n_rows, trial, dtype=np.int64)}
                    if self.with_trial_id
                    else {}
                ),
            },
        )


def _tokenizer(batch: Any) -> dict:
    """20 ms tokens; each latent carries its token's absolute start time."""
    batch_size, n_bins, n_units = batch.spike_matrix.shape
    n_tokens = int(round(n_bins / (TOKEN_MS / 1000.0 * SAMPLING_RATE)))
    token_s = TOKEN_MS / 1000.0
    starts = (
        batch.absolute_start[:, None]
        + torch.arange(n_tokens, dtype=torch.float64)[None, :] * token_s
    )
    latents = starts[:, :, None, None].expand(batch_size, n_tokens, n_units, 1).float()
    return {
        "reconstruction": batch.spike_matrix,
        "latents": latents.clone(),
        "token_start_time": starts,
        "token_end_time": starts + token_s,
        "token_timestamps": starts + token_s / 2,
    }


def _loader(batch_size: int, **dataset_kwargs: Any) -> DataLoader:
    dataset = _MultiIntervalDataset(**dataset_kwargs)
    # torch_brain annotates sampling_intervals as lists of tuples, but takes an
    # Interval per session at runtime (as NTTDataModule passes it).
    intervals: dict[str, Any] = {
        sid: Interval(
            start=np.array([s for s, _ in ivs]),
            end=np.array([e for _, e in ivs]),
        )
        for sid, ivs in dataset.sessions.items()
    }
    sampler = SequentialFixedWindowSampler(
        sampling_intervals=intervals, window_length=WINDOW_S, drop_short=True
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate)


def _segments(batch_size: int, tokenizer=None, collate_kwargs=None, **dataset_kwargs) -> list:
    return list(
        iter_collated_validation_segments(
            _loader(batch_size, **dataset_kwargs),
            tokenizer or _tokenizer,
            token_bin_size_ms=TOKEN_MS,
            window_length_seconds=WINDOW_S,
            device=torch.device("cpu"),
            **(collate_kwargs or {}),
        )
    )


# Kept windows: the sampler's overlapping tail windows are dropped
# (drop_overlapping_windows=True), leaving whole 100 ms windows per interval.
KEPT_WINDOWS = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (1.0, 1.1), (1.1, 1.2)] + [
    (2.0, 2.1),
    (2.1, 2.2),
    (2.2, 2.3),
    (2.3, 2.4),
]


@pytest.mark.parametrize("batch_size", [1, 4])
def test_multi_interval_session_yields_one_aligned_segment(batch_size):
    segments = _segments(batch_size)

    assert len(segments) == 1, "multi-interval session must not be dropped"
    segment = segments[0]
    assert segment.session_id == SESSION
    assert segment.metadata["n_gaps"] == 2
    assert segment.metadata["gap_seconds"] == pytest.approx((1.0 - 0.3) + (2.0 - 1.2))

    # Binned data: exactly the kept windows' bins, in order.
    expected_bins = np.concatenate(
        [np.arange(round(s * SAMPLING_RATE), round(e * SAMPLING_RATE)) for s, e in KEPT_WINDOWS]
    )
    np.testing.assert_array_equal(segment.original[0].numpy(), expected_bins)
    np.testing.assert_array_equal(segment.reconstructed[0].numpy(), expected_bins)
    assert bool(segment.pad_and_gap_mask.all())

    # Behavior at input resolution: row i belongs to binned column i.
    cursor = segment.behavioral_data_input_res["cursor_pos"][:, 0]
    np.testing.assert_allclose(cursor, (expected_bins + 0.5) / SAMPLING_RATE)

    # Token times keep real values; latents match their own token times.
    token_s = TOKEN_MS / 1000.0
    expected_starts = np.concatenate(
        [s + np.arange(round((e - s) / token_s)) * token_s for s, e in KEPT_WINDOWS]
    )
    np.testing.assert_allclose(segment.token_start_times.numpy(), expected_starts)
    latents = segment.latents.reshape(-1, N_UNITS)[:, 0].numpy()
    np.testing.assert_allclose(latents, expected_starts, atol=1e-6)
    np.testing.assert_array_equal(
        segment.behavioral_data_token_res["trial_id"],
        np.repeat([0, 1, 2], [15, 10, 20]),
    )


def test_downstream_chunks_align_across_gaps():
    """The decoder's index-space chunking pairs each latent chunk with the
    behavior from the same bins, also for chunks after a gap."""
    segment = _segments(batch_size=4)[0]
    X = segment.latents.reshape(-1, N_UNITS).numpy()
    timestamps = segment.token_timestamps.numpy()
    chunk_ms = 40.0
    X_chunks, _, chunk_indices, tokens_per_chunk = chunk_by_temporal_window(
        X, timestamps, token_bin_size_ms=TOKEN_MS, chunk_window_ms=chunk_ms
    )
    y, _, keep, _ = align_behavioral_to_chunks(
        chunk_indices,
        chunk_ms,
        segment.behavioral_data_input_res,
        sampling_rate=SAMPLING_RATE,
        target_field="cursor_pos",
        mode="regression",
    )
    assert keep.all()
    first_token_start = X_chunks[:, 0]
    first_bin_center = y[:, 0]
    np.testing.assert_allclose(
        first_bin_center, first_token_start + 0.5 / SAMPLING_RATE, atol=1e-6
    )
    assert tokens_per_chunk == 2
    assert len(X_chunks) == 45 // 2


def test_training_callback_full_eval_includes_multi_interval_session():
    """TokenizerEvalCallback's full-session eval (same iterator) now evaluates
    a multi-interval val session instead of silently skipping it."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import pytorch_lightning as pl

    from ntt.evaluation.config.schema import LightningEvalConfig
    from ntt.evaluation.core.types import DataType
    from ntt.evaluation.lightning import TokenizerEvalCallback

    loader = _loader(batch_size=4)

    callback = TokenizerEvalCallback(
        lightning_config=LightningEvalConfig(full_eval_at_training_end=True)
    )
    report = MagicMock()
    report.results = {"spike_count_correlation": MagicMock(raw_value=0.5)}
    report.skipped_metrics = {}
    callback.pipeline.run = MagicMock(return_value=report)

    trainer = MagicMock(spec=pl.Trainer)
    trainer.val_dataloaders = [loader]
    trainer.logger = MagicMock()
    trainer.global_step = 0
    module = MagicMock()
    module.tokenizer_spec = SimpleNamespace(
        bin_size_ms=1.0, window_length_seconds=WINDOW_S, output_type=DataType.SPIKES
    )
    module.device = torch.device("cpu")
    module.transfer_batch_to_device.side_effect = lambda batch, device, dataloader_idx=0: batch
    module.tokenizer.reconstruct_from_input.side_effect = lambda batch: {
        "reconstruction": batch.spike_matrix.float()
    }

    callback._run_full_session_eval(trainer, module)

    assert callback.pipeline.run.call_count == 1
    original, _ = callback.pipeline.run.call_args.args
    assert callback.pipeline.run.call_args.kwargs == {"session_id": SESSION}
    assert original.binned_spikes.shape == (N_UNITS, 900)  # 9 kept windows x 100 bins
    logged = trainer.logger.log_metrics.call_args.args[0]
    assert logged["full_eval/n_sessions"] == 1.0


def _assert_index_aligned(segment) -> None:
    """Tokens, input bins and behavior rows agree, and the decoders'
    index-space chunks pair each latent chunk with its own bins."""
    bins_per_token = round(TOKEN_MS / 1000.0 * SAMPLING_RATE)
    latents = segment.latents
    X = (
        (
            torch.cat(list(latents))
            if isinstance(latents, list)
            else latents.reshape(-1, *latents.shape[2:])
        )
        .reshape(-1, N_UNITS)
        .numpy()
    )
    cursor = segment.behavioral_data_input_res["cursor_pos"]
    assert segment.original.shape[1] == len(cursor) == len(X) * bins_per_token
    X_chunks, _, chunk_indices, _ = chunk_by_temporal_window(
        X, segment.token_timestamps.numpy(), token_bin_size_ms=TOKEN_MS, chunk_window_ms=40.0
    )
    y, _, keep, _ = align_behavioral_to_chunks(
        chunk_indices,
        40.0,
        segment.behavioral_data_input_res,
        sampling_rate=SAMPLING_RATE,
        target_field="cursor_pos",
        mode="regression",
    )
    assert keep.all()
    np.testing.assert_allclose(y[:, 0], X_chunks[:, 0] + 0.5 / SAMPLING_RATE, atol=1e-6)


def test_sessions_in_one_loader_do_not_leak_state():
    """A gapped session followed by a contiguous one (sharing batches, and
    overlapping in absolute time) collates exactly as each does alone."""
    gapped, contiguous = "brainset/gapped", "brainset/contiguous"
    both = {gapped: INTERVALS, contiguous: [(0.0, 0.5)]}
    mixed = {s.session_id: s for s in _segments(batch_size=4, sessions=both)}
    for sid in both:
        (alone,) = _segments(batch_size=4, sessions={sid: both[sid]})
        seg = mixed[sid]
        torch.testing.assert_close(seg.original, alone.original)
        torch.testing.assert_close(seg.latents, alone.latents)
        torch.testing.assert_close(seg.token_start_times, alone.token_start_times)
        np.testing.assert_array_equal(
            seg.behavioral_data_input_res["cursor_pos"],
            alone.behavioral_data_input_res["cursor_pos"],
        )
        assert seg.metadata["n_gaps"] == alone.metadata["n_gaps"]
    assert mixed[gapped].metadata["n_gaps"] == 2
    assert mixed[contiguous].metadata["n_gaps"] == 0
    assert mixed[contiguous].original.shape[1] == 500


# Recording-domain gap from 298 ms: window (0.2, 0.3) keeps 98 valid bins, so
# only 4 whole 20 ms tokens (80 bins); the tail window (0.25, 0.35) has an
# interior gap and is skipped.
TRUNCATING_GAP = {SESSION: [(0.298, 0.32)]}


@pytest.mark.parametrize("batch_size", [1, 4])
def test_truncated_window_is_skipped_to_keep_alignment(batch_size):
    (segment,) = _segments(batch_size, domain_gaps=TRUNCATING_GAP)
    assert segment.original.shape[1] == 800  # 9 whole windows minus the truncated one
    assert segment.token_start_times[10].item() == pytest.approx(1.0)  # after 2 windows
    _assert_index_aligned(segment)


def _center_only_tokenizer(batch: Any) -> dict:
    """Like ``_tokenizer`` but reports only token center times (as the
    perceiver's reconstruct_from_input does)."""
    out = _tokenizer(batch)
    del out["token_start_time"], out["token_end_time"]
    return out


def test_center_only_tokenizer_with_truncated_window():
    """Center-time stripping keeps the truncated window's partial 5th token
    (center 290 ms <= 298 ms); the token-count check accepts it and the window
    is then skipped because 5 tokens x 20 bins != 98 bins."""
    (segment,) = _segments(4, tokenizer=_center_only_tokenizer, domain_gaps=TRUNCATING_GAP)
    assert segment.token_start_times is None
    assert segment.original.shape[1] == 800
    _assert_index_aligned(segment)


def test_unbinned_behavior_is_dropped_for_the_session(caplog):
    """Raw behavior (rows != bins on some windows) is dropped for the whole
    session, including windows buffered before the mismatch, with a WARNING;
    the session then gets no decoding bypass. A binned session is untouched."""
    raw, binned = "brainset/raw", "brainset/binned"
    with caplog.at_level(logging.WARNING, logger="ntt.evaluation.utils.validation_collation"):
        segments = _segments(
            batch_size=4,
            sessions={raw: [(0.0, 0.6)], binned: [(0.0, 0.5)]},
            raw_behavior=frozenset({raw}),
            collate_kwargs={"max_eval_seconds": 0.25, "decoding_target_fields": {"cursor_pos"}},
        )
    by_sid = {s.session_id: s for s in segments}
    assert by_sid[raw].behavioral_data_input_res is None
    assert by_sid[raw].behavioral_data_token_res is None
    assert by_sid[raw].is_partial  # capped at max_eval_seconds: no bypass
    assert by_sid[raw].original.shape[1] == 300
    assert any("not binned to the input" in r.getMessage() for r in caplog.records)
    assert not by_sid[binned].is_partial  # bypass kept: full 500 ms
    _assert_index_aligned(by_sid[binned])


@pytest.mark.parametrize("batch_size", [1, 4])
def test_truncated_window_with_binned_behavior_keeps_session_behavior(batch_size):
    """A domain-clipped window (98.8 ms -> 99 unpadded bins and 99 behavior rows),
    padded by collate next to full windows, must not trip the unbinned-behavior
    guard: behavior is kept for the session, and step [5c] skips the window."""
    (segment,) = _segments(
        batch_size,
        clipped={SESSION: (1.0, 1.0988)},
        with_trial_id=False,  # step [4] resamples categoricals at padded length
    )
    assert segment.behavioral_data_input_res is not None
    assert segment.original.shape[1] == 800  # 9 whole windows minus the clipped one
    _assert_index_aligned(segment)
