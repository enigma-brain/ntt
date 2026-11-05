"""Collate helpers for variable-N tokenizer training.

Neuron-axis splitting
---------------------
``neuron_split_collate`` chunks any ``BinnedSpikesInput`` whose neuron
dimension exceeds ``max_neurons`` into multiple batch elements before
``torch_brain``'s standard collation runs. The chunks **share session,
subject, species, absolute_start/end, and behavioral metadata** — the
only fields that differ between chunks are ``spike_matrix``,
``pad_and_gap_mask``, ``unit_ids`` and ``unit_ids_pad_mask``.

Implications for downstream code:

- DDP gradient averaging treats each chunk as an independent batch
  element, so a recording that gets split into K chunks contributes K×
  weight per step relative to a recording that doesn't split. Account
  for this in any sampling-balance accounting.
- Evaluation / aggregation code that groups by ``session_id`` will see
  multiple batch elements with the same id. Group with care.
- The split is purely along the unit axis; time axis and behavioral
  data are unchanged across chunks.
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Any

from ntt.core.components.types import BinnedSpikesInput
from ntt.core.data.components.dynamic import dyn, pad2d_field, pad_field

log = logging.getLogger(__name__)
_warned_once: bool = False


def _split_binned_sample(sample: BinnedSpikesInput, max_neurons: int) -> list[BinnedSpikesInput]:
    """Split one binned sample along the neuron axis before batch collation.

    All non-neuron metadata is shared across chunks — see module docstring
    for downstream implications.
    """
    # Pre-collate, these fields hold torch_brain ``PaddedObject`` wrappers (``.obj`` = tensor).
    spike_mat = dyn(sample.spike_matrix).obj
    pad_mask = dyn(sample.pad_and_gap_mask).obj
    unit_ids = dyn(sample.unit_ids).obj
    unit_mask = dyn(sample.unit_ids_pad_mask).obj

    chunks: list[BinnedSpikesInput] = []
    for start in range(0, spike_mat.shape[1], max_neurons):
        end = min(start + max_neurons, spike_mat.shape[1])
        chunks.append(
            BinnedSpikesInput(
                spike_matrix=pad2d_field(spike_mat[:, start:end].contiguous()),
                pad_and_gap_mask=pad2d_field(pad_mask[:, start:end].contiguous()),
                sampling_rate=sample.sampling_rate,
                unit_ids=pad_field(unit_ids[start:end].contiguous()),
                unit_ids_pad_mask=pad_field(unit_mask[start:end].contiguous()),
                duration_sec=sample.duration_sec,
                session_id=sample.session_id,
                subject_id=sample.subject_id,
                species=sample.species,
                absolute_start=sample.absolute_start,
                absolute_end=sample.absolute_end,
                behavioral_data=sample.behavioral_data,
                behavioral_intervals=sample.behavioral_intervals,
            )
        )
    return chunks


def neuron_split_collate(batch: list[Any], max_neurons: int | None = None) -> Any:
    """Split oversized binned samples, then use the repository collate wrapper.

    See module docstring for the metadata-sharing semantics.
    """
    from ntt.core.data.datamodule import collate

    if max_neurons is None:
        return collate(batch)

    expanded: list[Any] = []
    split_any = False
    for sample in batch:
        if not isinstance(sample, BinnedSpikesInput):
            expanded.append(sample)
            continue
        n_neurons = dyn(sample.spike_matrix).obj.shape[1]
        if n_neurons <= max_neurons:
            expanded.append(sample)
        else:
            expanded.extend(_split_binned_sample(sample, max_neurons))
            split_any = True

    global _warned_once
    if split_any and not _warned_once:
        log.warning(
            "neuron_split_collate: at least one BinnedSpikesInput exceeded "
            "max_neurons=%d and was split along the neuron axis. Chunks "
            "share session_id/subject_id/etc; downstream code that groups "
            "by session sees multiple entries per recording. This warning "
            "fires once per process.",
            max_neurons,
        )
        _warned_once = True

    return collate(expanded)


def make_neuron_split_collate_fn(max_neurons: int | None = None):
    """Return a DataLoader collate_fn with optional neuron-axis splitting."""
    if max_neurons is None:
        from ntt.core.data.datamodule import collate

        return collate
    return partial(neuron_split_collate, max_neurons=max_neurons)
