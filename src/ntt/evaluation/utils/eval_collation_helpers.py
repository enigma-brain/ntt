"""Shared leaf helpers for the validation-collation iterators.

This module hosts the small, dependency-free helpers that are used by
both the standard ``iter_collated_validation_segments`` (in
:mod:`ntt.evaluation.utils.validation_collation`) and the replica
iterator ``iter_collated_validation_segments_replicas`` (in
:mod:`ntt.evaluation.utils.replica_collation`). Lifting them out keeps
each iterator's module focused on its own pipeline shape while still
guaranteeing both call into byte-identical leaf behaviour.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _strip_input_padding(
    original: torch.Tensor,
    reconstructed: torch.Tensor | None,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Remove trailing batch-collation padding and surface interior gaps.

    Expects shape (units, time). Strips only *trailing* all-False rows (units)
    and columns (time). After that strip, any remaining ``False`` values are
    interior temporal gaps (zero-filled bins between disjoint valid intervals,
    as produced by ``CreateBinnedSpikeCountsArray``). This function does not
    decide what to do with them — it logs a warning and returns the still-gapped
    mask so the caller can detect ``not mask.all()`` and skip the window. Both
    iterators (``validation_collation`` and ``replica_collation``) skip gapped
    windows rather than feed garbage gap bins into metrics.

    ``reconstructed`` may be ``None`` when the upstream tokenizer is
    tokenize-only (no decoder); in that case ``original`` and ``mask`` are
    stripped normally and ``None`` is passed through unchanged.
    """
    if mask.dtype is not torch.bool:
        mask = mask.bool()

    # Strip trailing all-False unit rows
    units_any = mask.any(dim=1)  # (units,)
    if not units_any.all():
        # Find the last True row
        last_valid_unit = int(units_any.nonzero(as_tuple=False)[-1].item()) + 1
        original = original[:last_valid_unit]
        if reconstructed is not None:
            reconstructed = reconstructed[:last_valid_unit]
        mask = mask[:last_valid_unit]

    # Strip trailing all-False time columns
    time_any = mask.any(dim=0)  # (time,)
    if not time_any.all():
        # Find the last True column
        last_valid_time = int(time_any.nonzero(as_tuple=False)[-1].item()) + 1
        original = original[:, :last_valid_time]
        if reconstructed is not None:
            reconstructed = reconstructed[:, :last_valid_time]
        mask = mask[:, :last_valid_time]

    # After stripping trailing padding, the remaining mask should be all-True.
    # Any interior False values indicate temporal gaps.
    if not mask.all():
        n_false = int((~mask).sum().item())
        logger.warning(
            "Gap detected in validation data: %d interior False values "
            "in pad_and_gap_mask after stripping trailing padding. "
            "Mask shape after strip: %s.",
            n_false,
            tuple(mask.shape),
        )

    return original, reconstructed, mask


def _drop_tokens_outside_bounds(
    tokens: torch.Tensor,
    token_start_times: torch.Tensor,
    token_end_times: torch.Tensor,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop tokens outside temporal bounds. Returns (trimmed_tokens, keep_mask).


    Expects per-window (unbatched) inputs with scalar bounds.
    If tokens have a batch dim with different token counts per element,
    raises ValueError — caller should unbatch into a list first.

    Args:
        tokens: Latent tensor, shape [T, ...] (unbatched per-window)
        token_start_times: Shape [T], start time of each token
        token_end_times: Shape [T], end time of each token
        lower_bound: Drop tokens where token_start_time < lower_bound
        upper_bound: Drop tokens where token_end_time > upper_bound (with small tolerance)

    Returns:
        (trimmed_tokens, keep_mask) where keep_mask is boolean tensor of shape [T]
        Caller applies keep_mask to other arrays (behavioral data, timestamps, etc.)
    """
    # Guard against accidentally passing batched input where the batch dim
    # differs from the token count. A per-window token tensor can be 3D
    # (e.g. [T, U, D]) which is fine — we just need the first dim to match.
    if tokens.ndim >= 3 and token_start_times.ndim == 1:
        if tokens.shape[0] != token_start_times.shape[0]:
            raise ValueError(
                f"tokens has {tokens.ndim} dimensions and first dim {tokens.shape[0]} "
                f"but token_start_times has {token_start_times.shape[0]} entries. "
                f"This looks like batched input — unbatch into a list first."
            )

    T = tokens.shape[0]
    assert token_start_times.shape == (T,), (
        f"token_start_times shape {token_start_times.shape} != expected ({T},)"
    )
    assert token_end_times.shape == (T,), (
        f"token_end_times shape {token_end_times.shape} != expected ({T},)"
    )

    if lower_bound is None and upper_bound is None:
        return tokens, torch.ones(T, dtype=torch.bool)

    # Use float64 for timestamp comparisons
    starts_f64 = token_start_times.to(dtype=torch.float64)
    ends_f64 = token_end_times.to(dtype=torch.float64)

    keep_mask = torch.ones(T, dtype=torch.bool)

    if lower_bound is not None:
        keep_mask &= starts_f64 >= lower_bound - 1e-6

    if upper_bound is not None:
        keep_mask &= ends_f64 <= upper_bound + 1e-6

    return tokens[keep_mask], keep_mask


def _strip_token_padding(
    latents: torch.Tensor,
    token_start_times: torch.Tensor,
    token_end_times: torch.Tensor,
    absolute_end: float,
    behavioral_data: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Strip tokens that extend past absolute_end (padding from tokenizer).


    Wrapper around _drop_tokens_outside_bounds that applies the keep_mask
    to all token-resolution arrays.

    Args:
        latents: Token latents, shape [T, ...]
        token_start_times: Shape [T], start time of each token
        token_end_times: Shape [T], end time of each token
        absolute_end: Drop tokens whose end_time exceeds this
        behavioral_data: Optional dict of arrays at token resolution, each with
            first dim == T

    Returns:
        Dict with keys: "latents", "token_start_times", "token_end_times",
        "behavioral_data" (None if not provided), "keep_mask".
    """
    trimmed_latents, keep_mask = _drop_tokens_outside_bounds(
        latents,
        token_start_times,
        token_end_times,
        upper_bound=absolute_end,
    )

    trimmed_starts = token_start_times[keep_mask]
    trimmed_ends = token_end_times[keep_mask]

    trimmed_behavioral: dict[str, np.ndarray] | None = None
    if behavioral_data is not None:
        mask_np = keep_mask.numpy()
        trimmed_behavioral = {}
        for key, value in behavioral_data.items():
            assert value.shape[0] == keep_mask.shape[0], (
                f"Behavioral field '{key}' has first dim {value.shape[0]} but "
                f"expected {keep_mask.shape[0]} (number of tokens)"
            )
            trimmed_behavioral[key] = value[mask_np]

    return {
        "latents": trimmed_latents,
        "token_start_times": trimmed_starts,
        "token_end_times": trimmed_ends,
        "behavioral_data": trimmed_behavioral,
        "keep_mask": keep_mask,
    }


def _dbg_check_token_count_via_bins_per_token(
    n_valid_tokens: int,
    absolute_start: float,
    absolute_end: float,
    token_bin_size_ms: float,
    sampling_rate: float,
    stripped_on_center_times: bool = False,
) -> None:
    """Assert the timestamp-based valid-token count matches the bins_per_token calculation.


    Cross-checks that the token count produced by timestamp-based stripping equals
    the count implied by the window duration for tokenizers with fixed token bin
    size, using the same keep rule the stripping applied:
    whole tokens (``real_input_bins // bins_per_token``) when it compared token
    end times, or tokens whose center lies inside the window when only center
    times were available (``stripped_on_center_times``).
    Silently skips when ``token_bin_size_ms`` does not divide evenly into the input
    bin size (variable-token tokenizers or misaligned bin sizes).
    """
    bin_size_ms = 1000.0 / sampling_rate
    bins_per_token = token_bin_size_ms / bin_size_ms
    if abs(bins_per_token - round(bins_per_token)) > 1e-6:
        # Token bin doesn't divide evenly into input bins — skip check
        return
    bins_per_token_int = int(round(bins_per_token))
    real_duration_sec = absolute_end - absolute_start
    # Floor, not round: ``_strip_token_padding`` keeps only tokens whose end
    # time is <= absolute_end + 1e-6, i.e. whole tokens. Rounding a truncated
    # window's leftover (>= half a bin) up over-counts by one. The 1e-6 s
    # tolerance mirrors the stripping rule's so exact multiples still count.
    real_input_bins = int(math.floor((real_duration_sec + 1e-6) * sampling_rate))
    expected_tokens = real_input_bins // bins_per_token_int
    if stripped_on_center_times:
        # Center-time stripping keeps token i iff (i + 0.5) * token_dur <=
        # duration + 1e-6, i.e. floor((duration + 1e-6) / token_dur + 0.5)
        # tokens: a trailing partial token is kept once its center fits.
        expected_tokens = int(
            math.floor((real_duration_sec + 1e-6) * sampling_rate / bins_per_token_int + 0.5)
        )
    assert n_valid_tokens == expected_tokens, (
        f"bins_per_token sanity check failed: timestamp-based approach kept "
        f"{n_valid_tokens} tokens, but bins_per_token calculation expects "
        f"{expected_tokens} (real_duration={real_duration_sec:.6f}s, "
        f"real_input_bins={real_input_bins}, bins_per_token={bins_per_token_int})"
    )
