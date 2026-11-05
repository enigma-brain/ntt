"""Core test for the replica-aware validation collator.

Verifies the core invariant of input_robustness (M1): per source window, the
emitted segment carries a ``replica_latents`` stack of shape
``[K+1, T_tok, D]`` and replica-id 0 (the clean row) is byte-identical
to the unmodified clean tokenizer output.

Synthetic dataset + identity-style tokenizer keep the test fast and
model-free, matching the fixture pattern used in
``tests/test_validation_collation.py``.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from ntt.core.components.types import BinnedSpikesInput
from ntt.core.data.datamodule import collate
from ntt.core.data.replica.sampler import (
    ReplicaWindowIndex,
    WindowReplicaBatchSampler,
)
from ntt.evaluation.utils.replica_collation import (
    iter_collated_validation_segments_replicas,
)


class _IdentityReplicaDataset:
    """Tiny dataset that produces a replica-id-tagged spike pattern.

    Each source window has a distinctive base pattern; replicas 1..K add
    a per-replica offset to the spike matrix so we can verify replica 0
    stays clean. ``__getitem__`` accepts either a
    :class:`ReplicaWindowIndex` (used by the replica sampler/dataloader)
    or, for direct calls, a plain index.
    """

    def __init__(self, n_units: int = 3, time_bins_per_window: int = 100):
        self.n_units = n_units
        self.time_bins_per_window = time_bins_per_window
        self.sampling_rate = 1000.0  # 1 kHz

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: ReplicaWindowIndex) -> BinnedSpikesInput:
        # The standard ``DataLoader`` calls ``dataset[idx]`` with one
        # ``ReplicaWindowIndex`` per item from our batch_sampler.
        spike_matrix = torch.zeros((self.time_bins_per_window, self.n_units), dtype=torch.int32)
        # Base pattern: 5 spikes at the start of every unit.
        spike_matrix[:5, :] = 1
        # Replica-dependent perturbation (clean replica = 0, no offset).
        if index.replica_id > 0:
            spike_matrix[5 + index.replica_id : 5 + index.replica_id + 5, :] = 1

        pad_and_gap_mask = torch.ones_like(spike_matrix, dtype=torch.int32)
        unit_ids = torch.arange(self.n_units, dtype=torch.int32)
        unit_ids_pad_mask = torch.ones(self.n_units, dtype=torch.int32)
        duration_sec = torch.tensor(self.time_bins_per_window / self.sampling_rate)

        return BinnedSpikesInput(
            spike_matrix=spike_matrix,
            pad_and_gap_mask=pad_and_gap_mask,
            sampling_rate=self.sampling_rate,
            unit_ids=unit_ids,
            unit_ids_pad_mask=unit_ids_pad_mask,
            duration_sec=duration_sec,
            session_id=str(index.recording_id),
            subject_id="test_subject",
            species="test_species",
            absolute_start=torch.tensor(float(index.start), dtype=torch.float64),
            absolute_end=torch.tensor(float(index.end), dtype=torch.float64),
            behavioral_data=None,
        )


class _SeqSourceSampler:
    """Plain sampler yielding a fixed list of (recording_id, start, end)
    tuples. The replica batch sampler wraps these."""

    def __init__(self, items):
        self._items = list(items)

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)


def test_replica_batch_sampler_caps_per_session_seconds():
    """Per-session max_eval_seconds should stop yielding later windows.

    The threshold-crossing window is still kept, matching the current input_robustness (M1)
    collation semantics.
    """
    base_items = [
        ReplicaWindowIndex(recording_id="sess_a", start=0.0, end=1.0, replica_id=0),
        ReplicaWindowIndex(recording_id="sess_a", start=1.0, end=2.0, replica_id=0),
        ReplicaWindowIndex(recording_id="sess_a", start=2.0, end=3.0, replica_id=0),
        ReplicaWindowIndex(recording_id="sess_b", start=0.0, end=1.0, replica_id=0),
        ReplicaWindowIndex(recording_id="sess_b", start=1.0, end=2.0, replica_id=0),
    ]
    sampler = WindowReplicaBatchSampler(
        base_sampler=_SeqSourceSampler(base_items),
        n_replicas=2,
        max_eval_seconds=1.5,
    )

    batches = list(sampler)

    assert len(batches) == 4
    assert len(sampler) == 4
    assert [batch[0].recording_id for batch in batches] == [
        "sess_a",
        "sess_a",
        "sess_b",
        "sess_b",
    ]
    assert [batch[0].start for batch in batches] == [0.0, 1.0, 0.0, 1.0]
    assert all(len(batch) == 3 for batch in batches)


def test_replica_batch_sampler_uses_effective_window_seconds_override():
    """Sampler-side cap should count analysis-window seconds, not source-window seconds.

    WindowShift uses longer source windows in the sampler than the 1.0 s
    analysis windows input_robustness (M1) actually evaluates.
    """
    base_items = [
        ReplicaWindowIndex(recording_id="sess0", start=0.0, end=1.2, replica_id=0),
        ReplicaWindowIndex(recording_id="sess0", start=1.0, end=2.2, replica_id=0),
        ReplicaWindowIndex(recording_id="sess0", start=2.0, end=3.2, replica_id=0),
    ]
    sampler = WindowReplicaBatchSampler(
        base_sampler=_SeqSourceSampler(base_items),
        n_replicas=1,
        max_eval_seconds=1.5,
        effective_window_seconds=1.0,
    )

    batches = list(sampler)

    assert len(batches) == 2
    assert len(sampler) == 2
    assert [batch[0].start for batch in batches] == [0.0, 1.0]


def _identity_tokenizer_with_latents(batch: BinnedSpikesInput) -> dict:
    """Identity tokenizer that exposes ``latents`` and token timestamps.

    Latents are taken to be the spike matrix folded into ``[B, T_tok, U, D]``
    with ``T_tok = T`` and ``D = 1`` so we can directly check
    replica-0 == clean spikes downstream."""
    B, T_in, U = batch.spike_matrix.shape
    # 1 token per input bin → token_bin_size_ms == 1 / sampling_rate * 1000.
    abs_start = batch.absolute_start.to(dtype=torch.float64)  # [B]
    rate = batch.sampling_rate
    if isinstance(rate, torch.Tensor):
        rate = rate.flatten()[0]
    sr = float(rate)
    bin_sec = 1.0 / sr
    indices_t = torch.arange(T_in, dtype=torch.float64)
    token_start_time = abs_start[:, None] + indices_t[None, :] * bin_sec
    token_end_time = token_start_time + bin_sec
    token_timestamps = token_start_time + 0.5 * bin_sec

    # Latents [B, T_tok, D] — flatten units into the last dim so the
    # downstream metric (which expects [N_windows, K+1, T_tok, D]) sees a
    # 4D stack after the iterator stacks across windows.
    #
    latents = batch.spike_matrix.to(dtype=torch.float32)  # [B, T_in, U] → D=U

    return {
        "reconstruction": batch.spike_matrix,
        "latents": latents,
        "token_timestamps": token_timestamps,
        "token_start_time": token_start_time,
        "token_end_time": token_end_time,
    }


class _GappedReplicaDataset(_IdentityReplicaDataset):
    """Replica dataset where the window starting at ``gap_start`` carries an
    interior temporal gap in its pad_and_gap_mask (valid bins on both sides)."""

    def __init__(self, gap_start: float, **kwargs):
        super().__init__(**kwargs)
        self.gap_start = gap_start

    def __getitem__(self, index: ReplicaWindowIndex) -> BinnedSpikesInput:
        sample = super().__getitem__(index)
        if abs(float(index.start) - self.gap_start) < 1e-9:
            lo = self.time_bins_per_window // 3
            hi = 2 * self.time_bins_per_window // 3
            sample.pad_and_gap_mask[lo:hi, :] = 0  # interior gap, trailing bins valid
        return sample


def test_replica_iterator_skips_window_with_interior_gap():
    """A source window with an interior gap must be skipped on the replica path,
    not processed with zero-filled gap bins. Two windows in, one gapped → only
    the clean window survives along the N_windows axis of the replica stack."""
    n_units = 3
    time_bins_per_window = 100
    K = 2
    dataset = _GappedReplicaDataset(
        gap_start=0.1, n_units=n_units, time_bins_per_window=time_bins_per_window
    )

    base_items = [
        ReplicaWindowIndex(recording_id="sess0", start=0.0, end=0.1, replica_id=0),
        ReplicaWindowIndex(recording_id="sess0", start=0.1, end=0.2, replica_id=0),
    ]
    batch_sampler = WindowReplicaBatchSampler(
        base_sampler=_SeqSourceSampler(base_items),
        n_replicas=K,
        max_windows=None,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate,
    )

    segments = list(
        iter_collated_validation_segments_replicas(
            loader,
            _identity_tokenizer_with_latents,
            n_replicas=K,
            device=torch.device("cpu"),
        )
    )

    # Windows are concatenated per session along axis 0 of the replica stack.
    # Both windows clean → shape[0] == 2; with the gapped window (start=0.1)
    # skipped only the clean window (start=0.0) survives → shape[0] == 1.
    assert len(segments) == 1
    seg = segments[0]
    assert seg.replica_latents is not None
    assert seg.replica_latents.shape[0] == 1, (
        f"gapped window must be skipped; expected 1 surviving window along the "
        f"N_windows axis, got {seg.replica_latents.shape[0]}"
    )


class _ClippedReplicaDataset(_IdentityReplicaDataset):
    """Replica dataset where the window starting at ``clip_start`` is
    boundary-clipped: its last time bin is a partial bin, masked invalid."""

    def __init__(self, clip_start: float, **kwargs):
        super().__init__(**kwargs)
        self.clip_start = clip_start

    def __getitem__(self, index: ReplicaWindowIndex) -> BinnedSpikesInput:
        sample = super().__getitem__(index)
        if abs(float(index.start) - self.clip_start) < 1e-9:
            sample.pad_and_gap_mask[-1, :] = 0  # trailing partial bin
        return sample


def test_replica_iterator_drops_boundary_clipped_window_before_tokenization():
    """The pre-tokenization drop must test the last *time bin*, not the last *unit*,
    otherwise a boundary-clipped window reaches the tokenizer. With 10 ms bins its
    partial trailing bin would fail the bins_per_token sanity check and M1 would
    skip every window_shift spec (ldns_perceiver)."""
    K = 2
    dataset = _ClippedReplicaDataset(clip_start=0.1, n_units=3, time_bins_per_window=100)
    base_items = [
        ReplicaWindowIndex(recording_id="sess0", start=0.0, end=0.1, replica_id=0),
        ReplicaWindowIndex(recording_id="sess0", start=0.1, end=0.2, replica_id=0),
    ]
    loader = DataLoader(
        dataset=dataset,
        batch_sampler=WindowReplicaBatchSampler(
            base_sampler=_SeqSourceSampler(base_items), n_replicas=K, max_windows=None
        ),
        collate_fn=collate,
    )
    tokenized_starts: list[float] = []

    def tokenizer(batch: BinnedSpikesInput) -> dict:
        tokenized_starts.append(float(batch.absolute_start[0]))
        return _identity_tokenizer_with_latents(batch)

    segments = list(
        iter_collated_validation_segments_replicas(
            loader, tokenizer, n_replicas=K, device=torch.device("cpu")
        )
    )

    assert tokenized_starts == [0.0], "the clipped window must not reach the tokenizer"
    assert len(segments) == 1
    assert segments[0].replica_latents is not None
    assert segments[0].replica_latents.shape[0] == 1


def test_replica_iterator_emits_K_plus_one_replica_axis_with_clean_at_index_zero():
    """One source window → segment with replica axis ``K+1``; replica 0 == clean."""
    n_units = 3
    time_bins_per_window = 100
    K = 2  # number of corrupted replicas → expected_group = 3
    dataset = _IdentityReplicaDataset(n_units=n_units, time_bins_per_window=time_bins_per_window)

    base_items = [
        ReplicaWindowIndex(recording_id="sess0", start=0.0, end=0.1, replica_id=0),
    ]
    batch_sampler = WindowReplicaBatchSampler(
        base_sampler=_SeqSourceSampler(base_items),
        n_replicas=K,
        max_windows=None,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate,
    )

    segments = list(
        iter_collated_validation_segments_replicas(
            loader,
            _identity_tokenizer_with_latents,
            n_replicas=K,
            device=torch.device("cpu"),
        )
    )

    assert len(segments) == 1, f"expected 1 segment, got {len(segments)}"
    seg = segments[0]
    assert seg.replica_latents is not None, "replica_latents must be populated"
    # Stacked across N_windows=1 → [N_windows, K+1, T_tok, U, D]
    assert seg.replica_latents.shape[0] == 1
    assert seg.replica_latents.shape[1] == K + 1, (
        f"expected K+1={K + 1} replicas along axis 1, got {seg.replica_latents.shape[1]}"
    )
    # Replica 0 carries the clean spikes (identity tokenizer).
    clean_replica = seg.replica_latents[0, 0]  # [T_tok, D]
    # Clean spike pattern: 5 spikes at the start, zeros elsewhere.
    expected_clean = torch.zeros((time_bins_per_window, n_units), dtype=torch.float32)
    expected_clean[:5, :] = 1.0
    torch.testing.assert_close(clean_replica, expected_clean, atol=0.0, rtol=0.0)

    # And at least one corrupted replica differs from clean.
    diffs = (seg.replica_latents[0, 1:] - seg.replica_latents[0, 0]).abs().sum()
    assert diffs.item() > 0, "corrupted replicas must differ from clean"

    # replica_metadata should record the replica count.
    assert seg.replica_metadata is not None
    assert seg.replica_metadata.get("n_replicas") == K
