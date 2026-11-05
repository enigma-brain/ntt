"""Input/output adapters between NTT types and nuclr encoder format.

Converts NTT model input types to the flat formats expected by the spike-based
perceiver encoder and the binned encoder.

The binning/format logic follows the NuCLR reference implementation, used
under the Apache-2.0 license: https://github.com/nerdslab/nuclr (Arora et al.,
2025, NeurIPS; https://arxiv.org/abs/2512.01199).
"""

import torch
from einops import rearrange
from torch import Tensor

from ntt.core.components.types import RatesInput, SpikeTimesInput


class NuclrInputData:
    """Preprocessed input for the spike-based nuclr encoder.

    The encoder operates on *flat* spike sequences (no batch or unit dimension).
    Spikes are sorted by unit, and sequence lengths track how many spikes belong
    to each unit.

    Attributes:
        spike_timestamps: Flat spike times, sorted by unit. Shape ``(total_spikes,)``.
        spike_seqlen: Number of spikes per unit. Shape ``(total_units,)``.
        unit_seqlen: Number of units per sample in the batch. Shape ``(batch,)``.
        unit_ids: Flat unit IDs (one per unit across the batch). Shape ``(total_units,)``.
        batch_size: Number of samples in the batch.
    """

    __slots__ = ("spike_timestamps", "spike_seqlen", "unit_seqlen", "unit_ids", "batch_size")

    def __init__(
        self,
        spike_timestamps: Tensor,
        spike_seqlen: Tensor,
        unit_seqlen: Tensor,
        unit_ids: Tensor,
        batch_size: int,
    ):
        self.spike_timestamps = spike_timestamps
        self.spike_seqlen = spike_seqlen
        self.unit_seqlen = unit_seqlen
        self.unit_ids = unit_ids
        self.batch_size = batch_size


class NuclrBinnedInputData:
    """Preprocessed input for the binned nuclr encoder.

    Spike times are binned and reshaped into (U*L, bins_per_latent) format
    where U = total units and L = num latents.

    Attributes:
        bins: Binned spike counts. Shape ``(total_units * num_latents, bins_per_latent)``.
        unit_seqlen: Number of units per sample in the batch. Shape ``(batch,)``.
        unit_ids: Flat unit IDs (one per unit across the batch). Shape ``(total_units,)``.
        batch_size: Number of samples in the batch.
    """

    __slots__ = ("bins", "unit_seqlen", "unit_ids", "batch_size")

    def __init__(
        self,
        bins: Tensor,
        unit_seqlen: Tensor,
        unit_ids: Tensor,
        batch_size: int,
    ):
        self.bins = bins
        self.unit_seqlen = unit_seqlen
        self.unit_ids = unit_ids
        self.batch_size = batch_size


def ntt_to_nuclr_format(x: SpikeTimesInput) -> NuclrInputData:
    """Convert NTT SpikeTimesInput to flat nuclr encoder format.

    The encoder expects spikes sorted by unit, with explicit sequence-length
    tensors that describe how spikes and units are grouped.

    Args:
        x: NTT SpikeTimesInput (batched or unbatched).

    Returns:
        NuclrInputData ready for the encoder.
    """
    spike_unit_index = x.spike_unit_index
    spike_timestamps = x.spike_timestamps
    spike_pad_mask = x.spike_pad_mask
    unit_ids = x.unit_ids
    unit_ids_pad_mask = x.unit_ids_pad_mask

    # Handle unbatched input
    if spike_unit_index.ndim == 1:
        spike_unit_index = spike_unit_index.unsqueeze(0)
        spike_timestamps = spike_timestamps.unsqueeze(0)
        spike_pad_mask = spike_pad_mask.unsqueeze(0)
        unit_ids = unit_ids.unsqueeze(0)
        unit_ids_pad_mask = unit_ids_pad_mask.unsqueeze(0)

    batch_size = spike_unit_index.shape[0]

    all_spike_times: list[Tensor] = []
    all_spike_seqlens: list[Tensor] = []
    all_unit_seqlens: list[int] = []
    all_unit_ids: list[Tensor] = []

    for b in range(batch_size):
        # Get valid spikes for this sample
        valid_mask = spike_pad_mask[b].bool()
        valid_units_mask = unit_ids_pad_mask[b].bool()

        s_times = spike_timestamps[b][valid_mask]
        s_units = spike_unit_index[b][valid_mask]
        b_unit_ids = unit_ids[b][valid_units_mask]

        if len(s_times) == 0:
            # No spikes — still contribute one empty entry per valid unit
            n_units = int(valid_units_mask.sum().item())
            all_spike_seqlens.append(torch.zeros(n_units, dtype=torch.long, device=s_times.device))
            all_unit_seqlens.append(n_units)
            all_unit_ids.append(b_unit_ids)
            continue

        # Group spikes contiguously by unit. A stable sort preserves input
        # order within each unit; within-unit order does not affect the encoder
        # (attention is permutation-invariant over keys, and the rotary
        # embedding is applied per spike timestamp regardless of order).
        sort_idx = torch.argsort(s_units, stable=True)
        s_times = s_times[sort_idx]
        s_units = s_units[sort_idx]

        # Count spikes per unit (some units may have 0 spikes). minlength keeps
        # trailing silent units at 0. s_units holds local indices 0..n_units-1.
        n_units = b_unit_ids.shape[0]
        spike_counts = torch.bincount(s_units, minlength=n_units).to(torch.long)

        all_spike_times.append(s_times)
        all_spike_seqlens.append(spike_counts)
        all_unit_seqlens.append(n_units)
        all_unit_ids.append(b_unit_ids)

    # Concatenate across batch
    if all_spike_times:
        flat_spike_times = torch.cat(all_spike_times, dim=0)
    else:
        flat_spike_times = torch.zeros(0, device=spike_timestamps.device)
    flat_spike_seqlen = torch.cat(all_spike_seqlens, dim=0)
    flat_unit_seqlen = torch.tensor(
        all_unit_seqlens, dtype=torch.long, device=spike_timestamps.device
    )
    flat_unit_ids = torch.cat(all_unit_ids, dim=0)

    return NuclrInputData(
        spike_timestamps=flat_spike_times,
        spike_seqlen=flat_spike_seqlen,
        unit_seqlen=flat_unit_seqlen,
        unit_ids=flat_unit_ids,
        batch_size=batch_size,
    )


def ntt_to_nuclr_binned_format(
    x: SpikeTimesInput,
    ctx_duration: float,
    bin_size: float,
    num_latents: int,
    bins_per_latent: int,
) -> NuclrBinnedInputData:
    """Convert NTT SpikeTimesInput to binned nuclr encoder format.

    Bins spike times into counts and reshapes to (U*L, bins_per_latent)
    matching the original NuCLR ``tokenize()`` method.

    Args:
        x: NTT SpikeTimesInput (batched or unbatched).
        ctx_duration: Context window duration in seconds.
        bin_size: Bin size in seconds.
        num_latents: Number of latent time positions.
        bins_per_latent: Number of bins per latent position.

    Returns:
        NuclrBinnedInputData ready for the binned encoder.
    """
    spike_unit_index = x.spike_unit_index
    spike_timestamps = x.spike_timestamps
    spike_pad_mask = x.spike_pad_mask
    unit_ids = x.unit_ids
    unit_ids_pad_mask = x.unit_ids_pad_mask

    # Handle unbatched input
    if spike_unit_index.ndim == 1:
        spike_unit_index = spike_unit_index.unsqueeze(0)
        spike_timestamps = spike_timestamps.unsqueeze(0)
        spike_pad_mask = spike_pad_mask.unsqueeze(0)
        unit_ids = unit_ids.unsqueeze(0)
        unit_ids_pad_mask = unit_ids_pad_mask.unsqueeze(0)

    batch_size = spike_unit_index.shape[0]
    num_bins = num_latents * bins_per_latent
    rate = 1.0 / bin_size

    all_bins: list[Tensor] = []
    all_unit_seqlens: list[int] = []
    all_unit_ids: list[Tensor] = []

    for b in range(batch_size):
        valid_mask = spike_pad_mask[b].bool()
        valid_units_mask = unit_ids_pad_mask[b].bool()

        s_times = spike_timestamps[b][valid_mask]
        s_units = spike_unit_index[b][valid_mask]
        b_unit_ids = unit_ids[b][valid_units_mask]
        n_units = b_unit_ids.shape[0]

        # Bin spikes: (n_units, num_bins+1) to handle edge case
        bins = torch.zeros(n_units, num_bins + 1, dtype=torch.int16, device=s_times.device)
        if len(s_times) > 0:
            bin_indices = torch.floor(s_times * rate).long().clamp(max=num_bins)
            bins.index_put_(
                indices=(s_units, bin_indices),
                values=torch.ones(len(s_times), dtype=torch.int16, device=s_times.device),
                accumulate=True,
            )
        # Trim to exact num_bins and reshape to (U*L, bins_per_latent)
        bins = bins[:, :num_bins]
        bins = rearrange(bins, "U (L B) -> (U L) B", L=num_latents, B=bins_per_latent)

        all_bins.append(bins)
        all_unit_seqlens.append(n_units)
        all_unit_ids.append(b_unit_ids)

    flat_bins = torch.cat(all_bins, dim=0)
    flat_unit_seqlen = torch.tensor(
        all_unit_seqlens, dtype=torch.long, device=spike_timestamps.device
    )
    flat_unit_ids = torch.cat(all_unit_ids, dim=0)

    return NuclrBinnedInputData(
        bins=flat_bins,
        unit_seqlen=flat_unit_seqlen,
        unit_ids=flat_unit_ids,
        batch_size=batch_size,
    )


def ntt_rates_to_nuclr_format(
    x: RatesInput,
    num_latents: int,
    bins_per_latent: int,
) -> NuclrBinnedInputData:
    """Convert NTT RatesInput (pre-binned by the data pipeline) to nuclr binned format.

    Takes the already-binned rate_matrix from the NTT pipeline and reshapes it
    to the (U*L, bins_per_latent) format expected by ``NuclrBinnedEncoder``.

    Args:
        x: NTT RatesInput (batched, shape ``(batch, num_time_bins, num_units)``).
        num_latents: Number of latent time positions (L).
        bins_per_latent: Number of bins per latent position (B).

    Returns:
        NuclrBinnedInputData ready for the binned encoder.
    """
    rate_matrix = x.rate_matrix  # (batch, T, U)
    pad_mask = x.pad_and_gap_mask
    unit_ids = x.unit_ids
    unit_ids_pad_mask = x.unit_ids_pad_mask

    # Handle unbatched input
    if rate_matrix.ndim == 2:
        rate_matrix = rate_matrix.unsqueeze(0)
        pad_mask = pad_mask.unsqueeze(0)
        unit_ids = unit_ids.unsqueeze(0)
        unit_ids_pad_mask = unit_ids_pad_mask.unsqueeze(0)

    batch_size = rate_matrix.shape[0]
    total_bins_needed = num_latents * bins_per_latent

    all_bins: list[Tensor] = []
    all_unit_seqlens: list[int] = []
    all_unit_ids: list[Tensor] = []

    for b in range(batch_size):
        valid_units_mask = unit_ids_pad_mask[b].bool()
        n_units = int(valid_units_mask.sum().item())
        b_unit_ids = unit_ids[b][valid_units_mask]

        # rate_matrix is (T, U_padded) — select valid units, transpose to (U, T)
        counts = rate_matrix[b][:, valid_units_mask]  # (T, U)
        counts = counts.T  # (U, T)

        # Truncate or pad time dimension to total_bins_needed
        T = counts.shape[1]
        if T >= total_bins_needed:
            counts = counts[:, :total_bins_needed]
        else:
            counts = torch.nn.functional.pad(counts, (0, total_bins_needed - T))

        # Reshape to (U*L, B)
        bins = rearrange(counts, "U (L B) -> (U L) B", L=num_latents, B=bins_per_latent)

        all_bins.append(bins)
        all_unit_seqlens.append(n_units)
        all_unit_ids.append(b_unit_ids)

    flat_bins = torch.cat(all_bins, dim=0)
    flat_unit_seqlen = torch.tensor(all_unit_seqlens, dtype=torch.long, device=rate_matrix.device)
    flat_unit_ids = torch.cat(all_unit_ids, dim=0)

    return NuclrBinnedInputData(
        bins=flat_bins,
        unit_seqlen=flat_unit_seqlen,
        unit_ids=flat_unit_ids,
        batch_size=batch_size,
    )
