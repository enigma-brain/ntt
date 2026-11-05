"""CPU tests for the spike-times adapter (``ntt_to_nuclr_format``).

Covers unit-grouping and per-unit spike counting, including units that are
silent in the window (zero spikes) and unsorted input spikes. These are the
behaviours the perceiver spike encoder relies on; the adapter had no direct
test before.
"""

import torch

from ntt.contrib.nuclr.adapter import ntt_to_nuclr_format
from ntt.core.components.types import SpikeTimesInput


def _make_input(unit_index_per_b, times_per_b, n_units_per_b) -> SpikeTimesInput:
    """Build a padded, batched ``SpikeTimesInput`` from per-batch python lists."""
    batch = len(unit_index_per_b)
    max_spikes = max((len(t) for t in times_per_b), default=1)
    max_units = max(n_units_per_b)

    spike_unit_index = torch.zeros(batch, max_spikes, dtype=torch.long)
    spike_timestamps = torch.zeros(batch, max_spikes)
    spike_pad_mask = torch.zeros(batch, max_spikes, dtype=torch.long)
    unit_ids = torch.zeros(batch, max_units, dtype=torch.long)
    unit_ids_pad_mask = torch.zeros(batch, max_units, dtype=torch.long)

    for b in range(batch):
        ns = len(times_per_b[b])
        spike_unit_index[b, :ns] = torch.tensor(unit_index_per_b[b], dtype=torch.long)
        spike_timestamps[b, :ns] = torch.tensor(times_per_b[b])
        spike_pad_mask[b, :ns] = 1
        nu = n_units_per_b[b]
        unit_ids[b, :nu] = torch.arange(nu)
        unit_ids_pad_mask[b, :nu] = 1

    return SpikeTimesInput(
        spike_unit_index=spike_unit_index,
        spike_timestamps=spike_timestamps,
        spike_pad_mask=spike_pad_mask,
        unit_ids=unit_ids,
        unit_ids_pad_mask=unit_ids_pad_mask,
        duration_sec=torch.tensor([1.0] * batch),
        session_id=tuple(f"s{b}" for b in range(batch)),
        subject_id=tuple(f"subj{b}" for b in range(batch)),
        species=tuple("mouse" for _ in range(batch)),
    )


def test_spike_adapter_groups_units_and_counts_with_silent_unit() -> None:
    # 1 batch element, 3 units; unit 1 is SILENT; spikes supplied unsorted.
    # Times chosen to be exact in float32 (0.5, 0.25, 0.75).
    x = _make_input(
        unit_index_per_b=[[0, 2, 0]],
        times_per_b=[[0.5, 0.75, 0.25]],
        n_units_per_b=[3],
    )
    out = ntt_to_nuclr_format(x)

    assert out.unit_seqlen.tolist() == [3]
    assert out.spike_seqlen.tolist() == [2, 0, 1]  # silent unit -> 0
    assert out.spike_timestamps.shape[0] == 3  # total preserved
    # spikes are grouped contiguously by unit (order within a unit is irrelevant)
    assert set(out.spike_timestamps[:2].tolist()) == {0.5, 0.25}  # unit 0
    assert out.spike_timestamps[2:].tolist() == [0.75]  # unit 2


def test_spike_adapter_multi_batch_seqlens() -> None:
    # batch 0: units [0,1] both fire; batch 1: unit 0 fires, unit 1 silent.
    x = _make_input(
        unit_index_per_b=[[0, 1], [0]],
        times_per_b=[[0.5, 0.25], [0.5]],
        n_units_per_b=[2, 2],
    )
    out = ntt_to_nuclr_format(x)

    assert out.unit_seqlen.tolist() == [2, 2]
    assert out.spike_seqlen.tolist() == [1, 1, 1, 0]
    assert int(out.unit_seqlen.sum()) == out.spike_seqlen.shape[0]
    assert out.spike_timestamps.shape[0] == 3
