"""Dummy-data generation for tokenizer validation."""

from __future__ import annotations

# TYPE_CHECKING avoids circular imports — TokenizerSpec is only needed for
# the type signature, not at runtime.
from typing import TYPE_CHECKING

import torch

from ntt.core.components.dist import Dist
from ntt.core.components.types import (
    BinnedSpikesInput,
    DataType,
    SpikeTimesInput,
)

if TYPE_CHECKING:
    from ntt.core.model.tokenizer.base import TokenizerSpec

# session/subject/species id of every dummy input. No dataset holds it, so tokenizers
# that look sessions up (e.g. embeddings tables) must special-case it.
VALIDATION_SESSION_ID = "__validation__"


def make_dummy_input(
    spec: TokenizerSpec,
    batch_size: int = 1,
    *,
    window_length_seconds: float | None = None,
    n_neurons: int | None = None,
) -> SpikeTimesInput | BinnedSpikesInput:
    """Generate minimal synthetic input matching the spec's input_type.

    Used for dummy-data validation at init time.  The generated data is
    minimal (batch=``batch_size``, default 1) and zero-filled — we only care
    about shapes and metadata propagation, not actual spike content.

    Args:
        spec: Tokenizer spec to generate a matching dummy input for.
        batch_size: Batch dimension for every batch-shaped field (data
            tensors, spike-times tuples, and per-sample metadata such as
            duration/session_id/subject_id/species/absolute_start/
            absolute_end). Defaults to 1 to preserve prior behavior.
        window_length_seconds: Optional explicit window length, in seconds. When
            ``None`` (the default) a ``Dist`` window is **sampled**, which is what
            init-time shape validation wants — a representative draw. Callers that need
            a deterministic window on both sides of a ratio pass this explicitly:
            ``compute_compression`` pins it to ``spec.effective_window_length_seconds``,
            because sampling here while the numerator resolved the spec to ``Dist.max``
            made ``compression_rate`` a readout of one random draw.
        n_neurons: Optional explicit neuron count. When ``None`` (the default) it comes
            from the spec: its fixed count, a draw from its ``Dist``, or 8 when the spec
            has none (``n_neurons=False``).
    """
    # Resolve n_neurons: an explicit override wins, else the spec decides.
    if n_neurons is None:
        if spec.n_neurons is False:
            n_neurons = 8  # Arbitrary small default for validation
        elif isinstance(spec.n_neurons, Dist):
            n_neurons = int(spec.n_neurons.sample())
        else:
            n_neurons = spec.n_neurons

    # Resolve window length: an explicit override wins, else sample a Dist.
    if window_length_seconds is not None:
        window_length = float(window_length_seconds)
    elif isinstance(spec.window_length_seconds, Dist):
        window_length = float(spec.window_length_seconds.sample())
    else:
        window_length = spec.window_length_seconds

    if spec.input_type == DataType.SPIKE_TIMES:
        return SpikeTimesInput(
            spike_unit_index=torch.zeros((batch_size, 0), dtype=torch.long),
            spike_timestamps=torch.zeros((batch_size, 0), dtype=torch.float),
            spike_pad_mask=torch.zeros((batch_size, 0), dtype=torch.long),
            unit_ids=torch.arange(n_neurons, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1),
            unit_ids_pad_mask=torch.ones((batch_size, n_neurons), dtype=torch.long),
            duration_sec=torch.full((batch_size,), window_length),
            session_id=tuple(VALIDATION_SESSION_ID for _ in range(batch_size)),
            subject_id=tuple(VALIDATION_SESSION_ID for _ in range(batch_size)),
            species=tuple(VALIDATION_SESSION_ID for _ in range(batch_size)),
        )

    # SPIKES or RATES: create BinnedSpikesInput
    if spec.bin_size_ms is False:
        bin_size_ms = 1.0  # Fallback for variable-dt tokenizers
    elif isinstance(spec.bin_size_ms, Dist):
        bin_size_ms = float(spec.bin_size_ms.sample())
    else:
        bin_size_ms = spec.bin_size_ms

    n_time_bins = max(1, int(window_length * 1000 / bin_size_ms))
    sampling_rate = 1000.0 / bin_size_ms

    spike_matrix = torch.zeros((batch_size, n_time_bins, n_neurons), dtype=torch.int32)
    pad_and_gap_mask = torch.ones_like(spike_matrix, dtype=torch.int32)

    return BinnedSpikesInput(
        spike_matrix=spike_matrix,
        pad_and_gap_mask=pad_and_gap_mask,
        sampling_rate=sampling_rate,
        unit_ids=torch.arange(n_neurons, dtype=torch.int32).unsqueeze(0).repeat(batch_size, 1),
        unit_ids_pad_mask=torch.ones((batch_size, n_neurons), dtype=torch.int32),
        duration_sec=torch.full((batch_size,), window_length),
        session_id=tuple(VALIDATION_SESSION_ID for _ in range(batch_size)),
        subject_id=tuple(VALIDATION_SESSION_ID for _ in range(batch_size)),
        species=tuple(VALIDATION_SESSION_ID for _ in range(batch_size)),
        absolute_start=torch.zeros((batch_size,), dtype=torch.float64),
        absolute_end=torch.full((batch_size,), window_length, dtype=torch.float64),
    )
