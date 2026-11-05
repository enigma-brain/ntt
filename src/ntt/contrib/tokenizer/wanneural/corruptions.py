"""Input augmentations for the WanNeural tokenizer (training time only).

These run inside ``WanNeuralTokenizer.forward`` between data extraction and
the encoder. They corrupt the input ``BinnedSpikes`` tensor that flows into
the encoder while leaving the reconstruction *target* untouched, so the
latent has to be robust to the corruption.

All ops are vectorized over (B, T, N) and assume binarized 0/1 inputs (as
produced by ``BinnedSpikesInput``); they no-op on ``RatesInput`` in the caller.

Distributions are zero-centered Gaussians clipped to a hard bin cap, so most
samples fall in {-1, 0, +1} at small ``std_bins`` — keeping perturbations
gentle by default.
"""

from __future__ import annotations

import torch
from torch import Tensor


def jitter(
    data: Tensor,
    pad_mask: Tensor,
    *,
    std_bins: float,
    cap_bins: int,
) -> tuple[Tensor, Tensor]:
    """Per-spike bin shift along time.

    For every non-zero bin, sample an integer offset ``round(N(0, std_bins²))``
    clipped to ``[-cap_bins, +cap_bins]`` and move the spike to
    ``data[:, t + offset, n]``. Spikes that collide on the same destination
    bin or land in pad/gap bins collapse via a final binarization to {0, 1} —
    so an over-jittered or colliding spike effectively drops.

    Returns ``(data, pad_mask)``; ``pad_mask`` is unchanged because jitter
    only moves spikes within the existing valid region.
    """
    if std_bins <= 0:
        return data, pad_mask
    spikes = data > 0
    if not spikes.any():
        return data, pad_mask

    _, T, _ = data.shape
    offsets = (torch.randn_like(data) * std_bins).round().clamp(-cap_bins, cap_bins).long()
    t_arange = torch.arange(T, device=data.device).view(1, T, 1)
    new_t = t_arange + offsets
    in_range = (new_t >= 0) & (new_t < T)
    new_t = new_t.clamp(0, T - 1)

    contrib = (spikes & in_range).to(data.dtype)
    out = torch.zeros_like(data)
    out.scatter_add_(1, new_t, contrib)
    out = out * pad_mask.to(out.dtype)
    out = (out > 0).to(data.dtype)
    return out, pad_mask


def spot_noise(
    data: Tensor,
    pad_mask: Tensor,
    *,
    p: float,
) -> tuple[Tensor, Tensor]:
    """Independent bit-flip of each valid bin with probability ``p``.

    A 0 becomes 1; a non-zero becomes 0. Pad/gap bins are skipped.
    """
    if p <= 0:
        return data, pad_mask
    flip = (torch.rand_like(data) < p) & pad_mask.bool()
    flipped = 1.0 - (data > 0).to(data.dtype)
    out = torch.where(flip, flipped, data)
    return out, pad_mask


def time_shift(
    data: Tensor,
    pad_mask: Tensor,
    *,
    std_bins: float,
    cap_bins: int,
) -> tuple[Tensor, Tensor]:
    """Per-sample integer shift of (data, pad_mask) along time.

    Each batch element draws one shift ``round(N(0, std_bins²))`` clipped to
    ``[-cap_bins, +cap_bins]``. Both ``data`` and ``pad_mask`` are gathered
    along T by the same shift. Bins that would come from outside the original
    window are zero-filled (data) and marked invalid (pad_mask) — i.e. shift,
    not roll, so signal from one window edge never wraps to the other.
    """
    if std_bins <= 0:
        return data, pad_mask
    B, T, _ = data.shape
    shifts = (
        (torch.randn(B, device=data.device) * std_bins).round().clamp(-cap_bins, cap_bins).long()
    )
    if (shifts == 0).all():
        return data, pad_mask

    t_arange = torch.arange(T, device=data.device)
    src = t_arange.unsqueeze(0) - shifts.unsqueeze(1)  # (B, T)
    in_range = (src >= 0) & (src < T)
    src = src.clamp(0, T - 1)
    src_idx = src.unsqueeze(-1).expand_as(data)
    keep = in_range.unsqueeze(-1).to(data.dtype)

    data_out = torch.gather(data, 1, src_idx) * keep
    pad_out = (torch.gather(pad_mask.to(data.dtype), 1, src_idx) * keep).to(pad_mask.dtype)
    return data_out, pad_out


def apply_input_augmentations(
    data: Tensor,
    pad_mask: Tensor,
    *,
    jitter_std_bins: float,
    jitter_cap_bins: int,
    spot_noise_p: float,
    time_shift_std_bins: float,
    time_shift_cap_bins: int,
) -> tuple[Tensor, Tensor]:
    """Apply jitter → spot_noise → time_shift in sequence.

    Each stage is a no-op when its scale parameter is 0, so the cost when all
    augmentations are disabled is a few cheap branches.
    """
    data, pad_mask = jitter(data, pad_mask, std_bins=jitter_std_bins, cap_bins=jitter_cap_bins)
    data, pad_mask = spot_noise(data, pad_mask, p=spot_noise_p)
    data, pad_mask = time_shift(
        data, pad_mask, std_bins=time_shift_std_bins, cap_bins=time_shift_cap_bins
    )
    return data, pad_mask
