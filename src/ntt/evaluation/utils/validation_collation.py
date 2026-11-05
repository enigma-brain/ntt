"""
# Validation pipeline:


             session recording
                    │
        ┌─ DataLoader [1 & 2] ─────────┐
        │                              │
        │  Sampler → window indices    │
        │     (introduces overlap)     │
        │           │                  │
        │  Transforms, binning         |
        |  with gap detection,         │
        │  behavioral extraction       │
        │           │                  │
        │  collate() batches windows   │
        │  (adds padding to make       |
        |   batch elements same size)  │
        │                              │
        └───────────┬──────────────────┘
                    │
                    │
    ┌───────────┬───┴───────┬──────────────┐
    │           │           │              │
pad_and_     spikes    behav_data     abs_time
gap_mask        │           │              │
    │        [3 Tok] ◄─────────────────────┘
    │           │           │
    │     ┌─────┴────┐      │
    │     │          │      │
    │   recon    latents + token_ts
    │     │          │      │
    │     │          └─[4a]─┤
    │     │            │    └──[4b]
    │     │            │       │
    │     │     behav_token_res  behav_input_res
    │     │      (categoricals)  (all fields, unchanged)
    │     │            │         │
    └────[5 strip padding]       │  (input-res not
          │            │         │   stripped/trimmed)
       [6 strip/trim overlap]    │
          │            |         │
         [7]───────────┘─────────┘
          │
          ▼
    CollatedValidationSegment
          │
         [8]
          │
          ▼
    EvaluationReport

  Legend:
  1. Sampler — windows the recording, introduces overlap for coverage
  2. DataLoader batch collation — pads windows to equal size within batch
  3. Tokenizer — encode wrapper (→ latents + token_timestamps) + decode (→ reconstruction);
     wrapper uses abs_time + tokenizer spec to compute timestamps if tokenizer didn't.
     (ignorant of padding; wrapper needs to be given absolute time start and end
     **including the padding**.)
  4a. Resample categorical behavioral fields to token_timestamps (nearest-neighbor)
  4b. Keep ALL behavioral fields at input resolution (unchanged)
  5a & b. Strip batch padding (undo [2]) + check for gaps (raise if found)
      - separate operation for stripping input/reconstruction and tokens/timestamps
        (different logic needed)
      - only behav_token_res goes through strip/trim (it's at token resolution)
  6a. Trim reconstruction + spike_matrix at input-bin resolution (always done)
  6b. Trim latents + behav_token_res + timestamps at token-bin resolution (optional)
  7. Collation — buffer per session, concatenate/stack on flush
  8. EvaluationPipeline.run() — metrics computation. Downstream metrics:
     - preprocess_downstream_data(): chunk X, filter whole chunks via token-res fields
     - align_behavioral_to_chunks(): map each kept chunk to input-res behavioral
       samples by timestamp, aggregate per metric (flatten for regression,
       majority vote for classification)

"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from functools import partial
from itertools import chain
from typing import Any, cast

logger = logging.getLogger(__name__)

import numpy as np
import torch
from lightning.pytorch.utilities import move_data_to_device
from torch.utils.data import DataLoader, Dataset
from torch_brain.data.dataset import DatasetIndex
from tqdm.auto import tqdm

from ntt.core.components.types import (
    BinnedInputType,
    BinnedSpikesInput,
    RatesInput,
    ReconstructionOutput,
    get_data_matrix,
)
from ntt.core.data.components.collate import make_neuron_split_collate_fn
from ntt.core.model.tokenizer.base import DataType, TokenizerSpec
from ntt.evaluation.aggregation.summary import create_report
from ntt.evaluation.config import EvaluationConfig
from ntt.evaluation.config.schema import AggregationStrategy
from ntt.evaluation.core.intermediates import IntermediatesCache
from ntt.evaluation.core.types import (
    EvalData,
    EvaluationReport,
    MetricCategory,
    MetricResult,
)
from ntt.evaluation.pipeline import EvaluationPipeline
from ntt.evaluation.utils.metric_selection import get_interval_windowing_allowed_metric_names
from ntt.evaluation.utils.per_stimulus_sampler import PerStimulusSampler
from ntt.evaluation.utils.reconstruction_type import ReconstructionTypeRecorder
from ntt.evaluation.utils.window_chunking import (
    chunked_tokenizer_fn,
    requires_fixed_input_window,
)


def _unit_ids_equal(a: Any, b: Any) -> bool:
    """Compare unit_ids metadata that may be tensors, lists, or None."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return a.shape == b.shape and torch.equal(a, b)
    return a == b


def _extract_scalar_sampling_rate(sr: Any) -> float | None:
    """Extract a scalar sampling rate from metadata, which may be a Tensor, float, or None."""
    if sr is None:
        return None
    if isinstance(sr, torch.Tensor):
        sr_flat = sr.flatten()
        assert torch.all(sr_flat == sr_flat[0]), (
            f"sampling_rate tensor has non-uniform values: {sr_flat.tolist()}"
        )
        return float(sr_flat[0].item())
    return float(sr)


@dataclass
class CollatedValidationSegment:
    """Container for a single collated validation segment.

    A segment is formed by concatenating multiple validation windows from the
    same session. The fields use two different combination strategies:

    **Concatenated along time** (dim=1 for 2D, dim=0 for 1D):
        - original:           (units, time) — cat along time
        - reconstructed:      (units, time) — cat along time
        - pad_and_gap_mask:   (units, time) — cat along time, or None
        - token_start_times:  (T_tokens,)   — cat along token dim
        - token_end_times:    (T_tokens,)   — cat along token dim
        - token_timestamps:   (T_tokens,)   — cat along token dim
        - behavioral_data_token_res: dict of arrays, each (T_tokens, ...) — categorical
                              fields resampled to token resolution (for phase/trial filtering)
        - behavioral_data_input_res: dict of arrays, each (n_input_bins, ...) — all
                              fields at original input resolution (e.g. 1ms), for
                              resolution-independent regression/classification targets

    **Stacked along windows** (new dim=0):
        - latents:            (N_windows, T_tokens_per_window, ...) — stacked if all
                              windows have the same token count; list[Tensor] otherwise

    **Scalar / from first window**:
        - session_id:         str — from first window
        - metadata:           dict — from first window, contains:
                              "sampling_rate", "subject_id", "species"
        - segment_index:      int — assigned by caller
        - is_partial:         bool — True if segment was flushed early due to limits
        - segment_absolute_start: float | None — absolute start time of the first
                              window in this segment (seconds)
    """

    session_id: str
    original: torch.Tensor | None
    reconstructed: torch.Tensor | None
    pad_and_gap_mask: torch.Tensor | None
    metadata: dict[str, Any]
    segment_index: int
    is_partial: bool
    latents: torch.Tensor | list[torch.Tensor] | None = None
    behavioral_data_token_res: dict[str, np.ndarray] | None = None
    behavioral_data_input_res: dict[str, np.ndarray] | None = None
    token_timestamps: torch.Tensor | None = None
    token_start_times: torch.Tensor | None = None
    token_end_times: torch.Tensor | None = None
    segment_absolute_start: float | None = None
    # Shape of ``original`` preserved after strip_original() releases the tensor.
    original_shape: tuple[int, ...] | None = None
    # ── Branch (C) replica fan-out fields ─────────────────────────────
    # Populated ONLY when ``iter_collated_validation_segments`` is called
    # with ``n_replicas is not None``. For all other callers these remain
    # None, and the ``original``/``reconstructed``/``latents`` fields
    # above carry replica 0 (clean) data so every existing metric works
    # unchanged.
    # - replica_latents: per-window [K+1, T_tok, D] or stacked
    #   [N_windows, K+1, T_tok, D].
    # - replica_reconstructions: per-window [K+1, U, T] or stacked
    #   [N_windows, K+1, U, T].
    # - replica_originals: per-window [K+1, U, T] or stacked
    #   [N_windows, K+1, U, T] — the *input* spike data BEFORE any
    #   tokenizer pass. Row 0 is the clean replica-0 input; rows 1..K
    #   are the corruption-applied input replicas. This is the
    #   input-side originals intermediate, consumed by downstream metrics.
    # - replica_metadata: e.g. {"n_replicas": K, "labels": [...]} —
    #   carries the corruption labels for each non-clean replica row,
    #   plus any other per-replica scalars the input_robustness (M1) metric needs.
    replica_latents: torch.Tensor | list[torch.Tensor] | None = None
    replica_reconstructions: torch.Tensor | list[torch.Tensor] | None = None
    replica_originals: torch.Tensor | list[torch.Tensor] | None = None
    replica_metadata: dict | None = None

    def strip_original(self) -> None:
        """Release the ``original`` tensor, keeping only its shape.

        After this call, ``original`` is ``None`` and ``original_shape`` holds
        the shape that ``original`` had.  Downstream code that only needs the
        shape (aggregation weights, dataframe metadata) should use
        ``original_shape``.
        """
        if self.original is not None:
            self.original_shape = tuple(self.original.shape)
            self.original = None

    def tensor_free_copy(self) -> CollatedValidationSegment:
        """Return a copy with all torch.Tensor fields nulled.

        Used before shipping segments across a fork-mode ProcessPool: any
        CUDA-resident tensor in the IPC payload triggers
        ``torch.cuda._lazy_init()`` in the worker (which can't re-init CUDA
        in a forked subprocess) and crashes the pool. Numpy fields and
        non-tensor identity (session_id, segment_index, original_shape,
        metadata, behavioral numpy dicts) are preserved by reference.

        ``original_shape`` is filled from ``original.shape`` if the latter
        is still set, mirroring ``strip_original``'s contract — downstream
        aggregation depends on having a shape.
        """
        shape: tuple[int, ...] | None = self.original_shape
        if shape is None and self.original is not None:
            shape = tuple(self.original.shape)
        return replace(
            self,
            original=None,
            reconstructed=None,
            pad_and_gap_mask=None,
            latents=None,
            token_timestamps=None,
            token_start_times=None,
            token_end_times=None,
            original_shape=shape,
        )


def _normalize_session_ids(session_ids: Any, batch_size: int) -> list[str]:
    if isinstance(session_ids, (list, tuple)):
        return [str(item) for item in session_ids]
    if isinstance(session_ids, str):
        return [session_ids] * batch_size
    if isinstance(session_ids, torch.Tensor):
        return [str(item) for item in session_ids.cpu().tolist()]
    raise ValueError("Batch is missing a valid session_id field.")


def _ensure_batch_dim(
    spike_matrix: torch.Tensor, pad_and_gap_mask: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Add batch dimension if needed.

    Expects spike_matrix shape (time, units) -> (batch, time, units).
    """
    if spike_matrix.ndim == 2:
        spike_matrix = spike_matrix.unsqueeze(0)
    if pad_and_gap_mask is not None and pad_and_gap_mask.ndim == 2:
        pad_and_gap_mask = pad_and_gap_mask.unsqueeze(0)
    return spike_matrix, pad_and_gap_mask


def _extract_sampling_rate(batch: BinnedInputType) -> float:
    """Extract sampling rate from a binned spike or rates batch."""
    rate = batch.sampling_rate
    # sampling_rate can be a scalar or a per-batch tensor
    if isinstance(rate, torch.Tensor):
        if rate.ndim > 0:
            rate = rate[0]
        return float(rate.item())
    return float(rate)


def _assert_token_bin_divides_window_length(
    window_length_seconds: float | None,
    token_bin_size_ms: float | None,
    *,
    tolerance_ms: float = 1e-6,
) -> None:
    """Assert that ``token_bin_size_ms`` evenly divides the configured window length.

    Run ONCE at startup of an evaluation iterator on the configured parameter
    values (``NTTDataModuleConfig.window_length_seconds`` × ``token_bin_size_ms``),
    not per batch on runtime durations — session-end partial windows can be
    shorter than the configured length and would spuriously trip a runtime
    check.

    The eval pipeline currently requires the token bin to evenly divide the
    input window so that reconstruction and original have matched bin counts.
    Dynamic-bin tokenizers pass ``token_bin_size_ms=None`` and skip this check.

    Args:
        window_length_seconds: Configured window length (seconds), or None to
            skip the check (e.g. caller couldn't determine it).
        token_bin_size_ms: Token bin in ms, or None for dynamic-bin tokenizers.
        tolerance_ms: Numerical tolerance (in ms) for the modulo check.

    Raises:
        ValueError: If the configured window length is not a multiple of the
            token bin.

    """
    if token_bin_size_ms is None:
        return  # dynamic-bin tokenizer — totally legitimate
    if window_length_seconds is None:
        return  # caller couldn't supply a configured length — skip

    window_length_ms = float(window_length_seconds) * 1000.0
    # Symmetric remainder so that a duration just below an integer multiple
    # is also caught.
    remainder = window_length_ms - round(window_length_ms / token_bin_size_ms) * token_bin_size_ms
    if abs(remainder) > tolerance_ms:
        raise ValueError(
            f"Token bin size {token_bin_size_ms} ms does not divide configured "
            f"input window length {window_length_ms} ms (remainder: {remainder} ms). "
            f"The eval pipeline currently requires token_bin_size_ms to evenly "
            f"divide the input window so that reconstruction and original have "
            f"matched bin counts. Either (a) configure the dataloader to use a "
            f"window length that is an integer multiple of the token bin, or "
            f"(b) for dynamic-bin tokenizers, set token_bin_size_ms=None on "
            f"the spec."
        )


def _trim_overlap(
    spike_matrix: torch.Tensor,
    reconstructed: torch.Tensor | None,
    pad_and_gap_mask: torch.Tensor | None,
    overlap_bins: int,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Trim overlapping time bins. Expects shape (units, time).

    ``reconstructed`` is optional (``None`` for tokenize-only tokenizers).
    """
    if overlap_bins <= 0:
        return spike_matrix, reconstructed, pad_and_gap_mask
    if overlap_bins >= spike_matrix.shape[1]:
        return (
            spike_matrix[:, :0],
            reconstructed[:, :0] if reconstructed is not None else None,
            pad_and_gap_mask[:, :0] if pad_and_gap_mask is not None else None,
        )
    spike_matrix = spike_matrix[:, overlap_bins:]
    if reconstructed is not None:
        reconstructed = reconstructed[:, overlap_bins:]
    if pad_and_gap_mask is not None:
        pad_and_gap_mask = pad_and_gap_mask[:, overlap_bins:]
    return spike_matrix, reconstructed, pad_and_gap_mask


def _trim_input_overlap(
    original: torch.Tensor,
    reconstructed: torch.Tensor | None,
    mask: torch.Tensor,
    current_absolute_start: float,
    prev_absolute_end: float,
    sampling_rate: float,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, float]:
    """Trim overlapping input bins from the head of the current window.

    Expects shape (units, time). Computes overlap from absolute timestamps
    and removes that many bins from the start of the time axis.

    ``reconstructed`` is optional (``None`` for tokenize-only tokenizers).
    Returns (trimmed_original, trimmed_reconstructed_or_None,
    trimmed_mask, trimmed_start) where trimmed_start is the new absolute
    start time after trimming.

    """
    # Use float64 for time arithmetic
    overlap_seconds = float(np.float64(prev_absolute_end) - np.float64(current_absolute_start))

    # No overlap (with small tolerance for floating point)
    if overlap_seconds <= 1e-9:
        return original, reconstructed, mask, current_absolute_start

    overlap_bins = int(round(overlap_seconds * sampling_rate))

    if overlap_bins <= 0:
        return original, reconstructed, mask, current_absolute_start

    if overlap_bins >= original.shape[1]:
        # Entire window is overlap — return empty tensors
        return (
            original[:, :0],
            reconstructed[:, :0] if reconstructed is not None else None,
            mask[:, :0],
            current_absolute_start + original.shape[1] / sampling_rate,
        )

    trimmed_original = original[:, overlap_bins:]
    trimmed_reconstructed = reconstructed[:, overlap_bins:] if reconstructed is not None else None
    trimmed_mask = mask[:, overlap_bins:]

    trimmed_start = current_absolute_start + overlap_bins / sampling_rate

    # Sanity check: trimmed_start should be approximately prev_absolute_end
    bin_duration = 1.0 / sampling_rate
    assert abs(trimmed_start - prev_absolute_end) <= bin_duration + 1e-9, (
        f"trimmed_start ({trimmed_start}) does not match prev_absolute_end "
        f"({prev_absolute_end}) within 1 bin tolerance ({bin_duration}). "
        f"overlap_seconds={overlap_seconds}, overlap_bins={overlap_bins}"
    )

    return trimmed_original, trimmed_reconstructed, trimmed_mask, trimmed_start


def _trim_token_overlap(
    latents: torch.Tensor,
    token_start_times: torch.Tensor,
    token_end_times: torch.Tensor,
    prev_last_token_end_time: float | None,
    behavioral_data: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Trim overlapping tokens from the head of the current window.

    Uses _drop_tokens_outside_bounds with lower_bound to remove tokens
    whose start_time falls before the previous window's last token end time.


    Args:
        latents: Token latents, shape [T, ...]
        token_start_times: Shape [T], start time of each token
        token_end_times: Shape [T], end time of each token
        prev_last_token_end_time: End time of the last token in the previous
            window. If None, no trimming is performed.
        behavioral_data: Optional dict of arrays at token resolution, each with
            first dim == T

    Returns:
        Dict with keys: "latents", "token_start_times", "token_end_times",
        "behavioral_data" (None if not provided), "keep_mask".
    """
    if prev_last_token_end_time is None:
        T = latents.shape[0]
        return {
            "latents": latents,
            "token_start_times": token_start_times,
            "token_end_times": token_end_times,
            "behavioral_data": behavioral_data,
            "keep_mask": torch.ones(T, dtype=torch.bool),
        }

    trimmed_latents, keep_mask = _drop_tokens_outside_bounds(
        latents,
        token_start_times,
        token_end_times,
        lower_bound=prev_last_token_end_time,
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


def _build_window_part(
    *,
    original: torch.Tensor,
    reconstructed: torch.Tensor | None,
    mask: torch.Tensor | None,
    absolute_start: float,
    sampling_rate: float,
    session_id: str,
    metadata: dict[str, Any],
    has_overlap: bool,
    behavioral_data_input_res: dict[str, np.ndarray] | None,
    behavioral_data_token_res: dict[str, np.ndarray] | None,
    latents: torch.Tensor | None,
    token_start_times: torch.Tensor | None,
    token_end_times: torch.Tensor | None,
    token_timestamps: torch.Tensor | None,
    start_bin: int,
    end_bin: int,
) -> dict[str, Any]:
    """Build one buffered window dict for a slice of a tokenized logical window.

    The returned dict is the standard per-window object later consumed by
    ``_flush_segment()``. It contains one contiguous time slice of the already
    tokenized data, with all arrays trimmed to the requested input-bin range.
    Token-resolution arrays are filtered to the tokens whose timestamps fall
    inside the same slice.

    ``reconstructed`` may be ``None`` for tokenize-only tokenizers; in that
    case the part dict carries ``"reconstructed": None`` so downstream
    consumers (``_pad_window_for_small_mismatch``, ``_flush_segment``) can
    detect the absence and skip reconstruction-side work uniformly.
    """
    n_bins = original.shape[1]
    if start_bin < 0 or end_bin > n_bins or start_bin >= end_bin:
        raise ValueError(
            f"Invalid part bounds [{start_bin}, {end_bin}) for window with {n_bins} bins."
        )

    part_start = absolute_start + start_bin / sampling_rate
    part_end = absolute_start + end_bin / sampling_rate

    part_dict: dict[str, Any] = {
        "original": original[:, start_bin:end_bin],
        "reconstructed": (
            reconstructed[:, start_bin:end_bin] if reconstructed is not None else None
        ),
        "mask": mask[:, start_bin:end_bin] if mask is not None else None,
        "session_id": session_id,
        "metadata": metadata,
        "absolute_start": part_start,
        "absolute_end": part_end,
        "has_overlap": has_overlap,
    }

    if behavioral_data_input_res is not None:
        part_dict["behavioral_data_input_res"] = {
            key: value[start_bin:end_bin] for key, value in behavioral_data_input_res.items()
        }

    token_keep_mask: torch.Tensor | None = None
    tol = 1e-9
    if token_start_times is not None and token_end_times is not None:
        token_keep_mask = (token_start_times >= part_start - tol) & (
            token_end_times <= part_end + tol
        )
    elif token_timestamps is not None:
        token_keep_mask = (token_timestamps >= part_start - tol) & (
            token_timestamps < part_end - tol
        )

    if token_keep_mask is not None:
        if latents is not None:
            part_latents = latents[token_keep_mask]
            if part_latents.numel() > 0:
                part_dict["latents"] = part_latents
        if token_start_times is not None and token_end_times is not None:
            part_dict["token_start_times"] = token_start_times[token_keep_mask]
            part_dict["token_end_times"] = token_end_times[token_keep_mask]
        if token_timestamps is not None:
            part_dict["token_timestamps"] = token_timestamps[token_keep_mask]
        if behavioral_data_token_res is not None:
            mask_np = token_keep_mask.numpy()
            part_dict["behavioral_data_token_res"] = {
                key: value[mask_np] for key, value in behavioral_data_token_res.items()
            }
    else:
        if latents is not None:
            part_dict["latents"] = latents
        if token_start_times is not None and token_end_times is not None:
            part_dict["token_start_times"] = token_start_times
            part_dict["token_end_times"] = token_end_times
        if token_timestamps is not None:
            part_dict["token_timestamps"] = token_timestamps
        if behavioral_data_token_res is not None:
            part_dict["behavioral_data_token_res"] = behavioral_data_token_res

    return part_dict


def _merge_window_parts(
    parts: list[dict[str, Any]],
    *,
    window_group_id_token_res: int | None = None,
) -> dict[str, Any]:
    """Merge buffered slices that belong to the same logical evaluation window.

    Input:
    - a list of buffered-window dicts representing consecutive pieces of the
      same logical window

    Output:
    - one buffered-window dict with time-concatenated input arrays and
      token-concatenated token metadata, ready for ``_flush_segment()``
    """
    if not parts:
        raise ValueError("Cannot merge an empty logical window.")
    if len(parts) == 1:
        merged = dict(parts[0])
    else:
        # ``reconstructed`` is optional (tokenize-only tokenizers leave the
        # part dict's ``"reconstructed"`` slot as ``None``). Concatenate
        # when every part carries it; refuse a mixed batch — splicing a
        # half-reconstruction would silently misalign the merged window.
        recon_flags = [p.get("reconstructed") is not None for p in parts]
        if any(recon_flags) and not all(recon_flags):
            raise ValueError(
                "Mixed reconstruction presence across parts of one logical "
                "window; cannot merge — would silently misalign time bins."
            )
        merged_reconstructed = (
            torch.cat([p["reconstructed"] for p in parts], dim=1) if all(recon_flags) else None
        )
        merged = {
            "original": torch.cat([p["original"] for p in parts], dim=1),
            "reconstructed": merged_reconstructed,
            "mask": (
                torch.cat([p["mask"] for p in parts if p.get("mask") is not None], dim=1)
                if any(p.get("mask") is not None for p in parts)
                else None
            ),
            "session_id": parts[0]["session_id"],
            "metadata": parts[0]["metadata"],
            "absolute_start": parts[0]["absolute_start"],
            "absolute_end": parts[-1]["absolute_end"],
            "has_overlap": any(p.get("has_overlap", False) for p in parts),
        }

        if any(p.get("latents") is not None for p in parts):
            merged["latents"] = torch.cat(
                [p["latents"] for p in parts if p.get("latents") is not None],
                dim=0,
            )
        if any("token_start_times" in p for p in parts):
            merged["token_start_times"] = torch.cat(
                [p["token_start_times"] for p in parts if "token_start_times" in p],
                dim=0,
            )
            merged["token_end_times"] = torch.cat(
                [p["token_end_times"] for p in parts if "token_end_times" in p],
                dim=0,
            )
        if any("token_timestamps" in p for p in parts):
            merged["token_timestamps"] = torch.cat(
                [p["token_timestamps"] for p in parts if "token_timestamps" in p],
                dim=0,
            )
        if any(p.get("behavioral_data_token_res") for p in parts):
            all_keys = {
                key for p in parts for key in (p.get("behavioral_data_token_res") or {}).keys()
            }
            merged["behavioral_data_token_res"] = {
                key: np.concatenate(
                    [
                        p["behavioral_data_token_res"][key]
                        for p in parts
                        if p.get("behavioral_data_token_res") is not None
                        and key in p["behavioral_data_token_res"]
                    ],
                    axis=0,
                )
                for key in sorted(all_keys)
            }
        if any(p.get("behavioral_data_input_res") for p in parts):
            all_keys_ir = {
                key for p in parts for key in (p.get("behavioral_data_input_res") or {}).keys()
            }
            merged["behavioral_data_input_res"] = {
                key: np.concatenate(
                    [
                        p["behavioral_data_input_res"][key]
                        for p in parts
                        if p.get("behavioral_data_input_res") is not None
                        and key in p["behavioral_data_input_res"]
                    ],
                    axis=0,
                )
                for key in sorted(all_keys_ir)
            }

    if window_group_id_token_res is not None and merged.get("latents") is not None:
        # See `buffer_processed_window`: window_group_id_token_res only depends
        # on having latents, not on input-resolution behavioral data.
        n_tokens = merged["latents"].shape[0]
        if merged.get("behavioral_data_token_res") is None:
            merged["behavioral_data_token_res"] = {}
        merged["behavioral_data_token_res"]["window_group_id_token_res"] = np.full(
            n_tokens, window_group_id_token_res, dtype=np.int32
        )

    return merged


def _extract_metadata(batch: BinnedSpikesInput | RatesInput, index: int) -> dict[str, Any]:
    """Extract session and per-unit metadata at given batch index.

    ``sampling_rate`` is normalised to a Python float here even though the
    NamedTuple's type annotation says ``float``: collated batches actually
    carry a tensor (uniform across batch elements), and storing the tensor
    in segment.metadata leaks it into anything downstream that ships
    metadata across IPC — including the fork-pool path in
    ``run_poisson_realisations`` (logs/33166.err repro).
    """
    metadata: dict[str, Any] = {
        "sampling_rate": _extract_scalar_sampling_rate(batch.sampling_rate)
    }

    # Handle subject_id
    if isinstance(batch.subject_id, (list, tuple)):
        metadata["subject_id"] = batch.subject_id[index]
    else:
        metadata["subject_id"] = batch.subject_id

    # Handle species
    if isinstance(batch.species, (list, tuple)):
        metadata["species"] = batch.species[index]
    else:
        metadata["species"] = batch.species

    # Propagate unit_ids (integer unit_number values from the dataset)
    if batch.unit_ids is not None:
        unit_ids_tensor = batch.unit_ids
        if unit_ids_tensor.ndim > 1:
            unit_ids_tensor = unit_ids_tensor[index]
        pad_mask_bool = None
        if batch.unit_ids_pad_mask is not None:
            pad_mask = batch.unit_ids_pad_mask
            if pad_mask.ndim > 1:
                pad_mask = pad_mask[index]
            pad_mask_bool = pad_mask.bool()
            unit_ids_tensor = unit_ids_tensor[pad_mask_bool]
        metadata["unit_ids"] = unit_ids_tensor.cpu().tolist()

    return metadata


# ── Shared leaf helpers ──────────────────────────────────────────────
# These small helpers are used by both this module's standard iterator
# and the replica iterator in
# :mod:`ntt.evaluation.utils.replica_collation`. They live in
# ``eval_collation_helpers`` so the two iterators share byte-identical
# leaf behaviour. Re-exported here to keep existing imports
# (e.g. ``from ntt.evaluation.utils.validation_collation import _strip_input_padding``
# in tests) working unchanged.
from ntt.evaluation.utils.eval_collation_helpers import (  # noqa: E402
    _dbg_check_token_count_via_bins_per_token,
    _drop_tokens_outside_bounds,
    _strip_input_padding,
    _strip_token_padding,
)


def _pad_window_for_small_mismatch(
    window: dict[str, Any],
    *,
    sampling_rate: float,
    is_interval_windowed: bool = False,
) -> tuple[dict[str, Any], int] | tuple[dict[str, Any], int, bool]:
    """Zero-pad ``original`` and ``reconstructed`` to the larger of the two.

    Background: even with the divisibility check applied at the top of the
    iterator, ``_strip_token_padding`` and other end-of-window adjustments can
    leave the reconstruction one or a few bins shorter (or longer) than the
    original. Today this surfaces only at metric-time as a hard ValueError in
    metrics that compare them shape-for-shape (e.g. ``smoothed_squared_error``).

    If the absolute mismatch is at most ``MISMATCH_TOLERANCE_BINS`` (matching
    the existing 2 ms input-contiguity tolerance), pad both tensors to the
    larger length with zeros at the trailing end. The mask (if present) is
    padded with ``False`` so metrics that consume the mask know the padded
    bins are not valid.

    Behavioral data at input resolution is intentionally not extended — for
    example, ``image_id`` would silently get a 0 ID if we padded with zeros,
    which is wrong (0 is a valid id). The caller should treat the trailing
    pad as not-real-data via the mask.

    If the mismatch exceeds the tolerance, raise ``ValueError``. The intent is:
    tolerate only tiny end effects (up to the existing ~2 ms tolerance), but
    fail loud on anything larger because that indicates a real orig/recon
    alignment bug rather than harmless rounding.

    Returns a tuple ``(padded_window, n_padded_bins)`` where
    ``n_padded_bins`` is 0 when shapes already match. When
    ``is_interval_windowed`` is true, also returns ``is_large_mismatch`` so
    callers can audit structural padding separately from small end effects.

    """
    original = window["original"]
    reconstructed = window.get("reconstructed")
    if reconstructed is None:
        # Tokenize-only window — no reconstruction tensor to length-match.
        # ``original`` (and the mask) are left untouched.
        if is_interval_windowed:
            return window, 0, False
        return window, 0
    orig_t = original.shape[1]
    recon_t = reconstructed.shape[1]
    if orig_t == recon_t:
        if is_interval_windowed:
            return window, 0, False
        return window, 0

    diff = abs(orig_t - recon_t)
    # Match the existing 2 ms input-contiguity tolerance in _flush_segment.
    mismatch_tolerance_bins = max(1, round(2e-3 * sampling_rate))
    is_large_mismatch = diff > mismatch_tolerance_bins

    if is_large_mismatch and not is_interval_windowed:
        raise ValueError(
            f"Per-window time-bin mismatch between original and reconstructed "
            f"of {diff} bins (original={orig_t}, reconstructed={recon_t}, "
            f"tolerance={mismatch_tolerance_bins} bins ≈ 2 ms at "
            f"sampling_rate={sampling_rate} Hz). "
            f"Refusing to silently skip: small per-window mismatches "
            f"accumulate to a true temporal shift across windows, so a "
            f"mismatch this large is almost certainly an upstream "
            f"alignment bug."
        )

    target_t = max(orig_t, recon_t)
    n_units_orig = original.shape[0]
    n_units_recon = reconstructed.shape[0]
    if orig_t < target_t:
        pad = torch.zeros(n_units_orig, target_t - orig_t, dtype=original.dtype)
        original = torch.cat([original, pad], dim=1)
    if recon_t < target_t:
        pad = torch.zeros(n_units_recon, target_t - recon_t, dtype=reconstructed.dtype)
        reconstructed = torch.cat([reconstructed, pad], dim=1)

    new_window = dict(window)
    new_window["original"] = original
    new_window["reconstructed"] = reconstructed

    # Pad mask with False (or 0) so consumers know these bins are invalid.
    mask = window.get("mask")
    if mask is not None and mask.shape[1] < target_t:
        n_units_mask = mask.shape[0]
        if mask.dtype == torch.bool:
            mask_pad = torch.zeros(n_units_mask, target_t - mask.shape[1], dtype=torch.bool)
        else:
            mask_pad = torch.zeros(n_units_mask, target_t - mask.shape[1], dtype=mask.dtype)
        new_window["mask"] = torch.cat([mask, mask_pad], dim=1)

    # Note: behavioral_data_input_res is intentionally NOT extended — see
    # docstring. The mask flags the padded bins as invalid for any consumer.
    if is_interval_windowed:
        return new_window, diff, is_large_mismatch
    return new_window, diff


def _flush_segment(
    window_buffer: list[dict],
    drop_overlapping_windows: bool = True,
    skip_gap_check: bool = False,
    is_interval_windowed: bool = False,
    assemble_input_and_reconstruction: bool = True,
) -> CollatedValidationSegment:
    """Concatenate buffered windows into a CollatedValidationSegment.

    Counts between-window time gaps (> 2 ms) and concatenates across them:
    a session whose sampling domain has several intervals (e.g. trial-based
    test domains) yields one segment of its contiguous runs joined
    end-to-end. The count and total duration are recorded in
    ``metadata["n_gaps"]`` / ``metadata["gap_seconds"]``.
    Latents are stacked along a new window dimension (dim=0).
    Everything else (original, reconstructed, mask, token times,
    behavioral data) is concatenated along time.


    Args:
        window_buffer: List of per-window dicts. Expected keys:
            - "original", "reconstructed", "mask": shape (units, time)
            - "latents": shape (T_tokens, ...) per window
            - "absolute_start", "absolute_end": floats
            - "session_id": str
            - "metadata": dict
            - "has_overlap": bool
            Optional keys:
            - "token_start_times", "token_end_times", "token_timestamps": shape (T_tokens,)
            - "behavioral_data_token_res": dict of arrays (categoricals at token res)
            - "behavioral_data_input_res": dict of arrays (all fields at input res)
        drop_overlapping_windows: If True, windows marked with has_overlap=True
            are excluded from latent/token/behavioral data (but not from the
            binned original/reconstructed data which is already trimmed at the
            bin level).
        skip_gap_check: If True, skip the inter-window gap count (no
            ``n_gaps`` / ``gap_seconds`` metadata is recorded).
            Used by the input_robustness (M1) replica iterator, which fans windows out into
            per-window (clean, corrupted_k) pairs in ``expand_results`` and
            therefore does not require contiguous analysis sub-windows
            (e.g. WindowShift corruptions deliberately break contiguity).
            The standard validation iterator keeps the default (False) so
            its segments report how many gaps they span.
        assemble_input_and_reconstruction: If True (default), per-window
            original/reconstructed/mask tensors are length-matched and
            concatenated along time into ``segment.original``,
            ``segment.reconstructed``, ``segment.pad_and_gap_mask``. Set to
            False on the interval-windowing path: there, uneven window sizes
            cause incomplete tokenization and therefore incomplete
            reconstruction inside each window, so a concatenated trace
            cannot be aligned with the original input. In that case all
            three fields are left ``None`` and ``segment.original_shape`` is
            populated directly from the per-window shapes (used for
            aggregation weighting). Latents, token times, and behavioral
            data are still combined as usual — only the input/reconstruction
            tensors are skipped.

    Returns:
        CollatedValidationSegment with concatenated/stacked data.

    Raises:
        ValueError: If window_buffer is empty.
    """
    if not window_buffer:
        raise ValueError("Cannot flush empty window buffer.")

    if is_interval_windowed:
        skip_gap_check = True

    # --- Drop overlapping windows (the later of each overlapping pair) ---
    if drop_overlapping_windows:
        window_buffer = [w for w in window_buffer if not w.get("has_overlap", False)]
        if not window_buffer:
            raise ValueError(
                "All windows were marked as overlapping — nothing left after "
                "dropping. Set drop_overlapping_windows=False to keep them."
            )

    # --- Gap detection between adjacent windows ---
    # Gaps come from multi-interval sampling domains (one sampler interval per
    # trial) and from windows skipped for straddling a recording-domain gap.
    # Concatenate across them rather than drop the segment (which silently
    # dropped whole sessions). Alignment is unaffected: each window adds its
    # own bins, tokens (with real timestamps) and behavioral rows, and the
    # sampler's overlapping interval-tail windows were dropped above, so the
    # decoders' index-space chunking stays matched across joins. Accepted
    # side effect: time-lag metrics (ACG/CCG, smoothing) see the runs joined
    # end-to-end, equally in the original and the reconstruction.
    n_gaps = 0
    gap_seconds = 0.0
    if not skip_gap_check:
        GAP_TOLERANCE = 2e-3  # 2 ms
        for i in range(len(window_buffer) - 1):
            prev_end = np.float64(window_buffer[i]["absolute_end"])
            next_start = np.float64(window_buffer[i + 1]["absolute_start"])
            if next_start - prev_end > GAP_TOLERANCE:
                n_gaps += 1
                gap_seconds += float(next_start - prev_end)
        if n_gaps:
            logger.info(
                "Session %s: concatenating across %d temporal gap(s) between "
                "windows (%.3fs total).",
                window_buffer[0].get("session_id", "?"),
                n_gaps,
                gap_seconds,
            )

    # --- Pad small per-window orig/recon mismatches before concatenation ---
    # _strip_token_padding (and similar end-of-window adjustments) can leave
    # the reconstruction one or a few bins shorter than the original. Pad the
    # shorter side with zeros so metrics that compare them shape-for-shape
    # don't fail at compute time. Mismatches above the 2 ms tolerance raise.
    n_zero_padded_bins_total = 0
    n_padded_windows = 0
    n_large_mismatch_bins_total = 0
    n_large_mismatch_windows = 0
    original: torch.Tensor | None = None
    reconstructed: torch.Tensor | None = None
    pad_and_gap_mask: torch.Tensor | None = None
    original_shape: tuple[int, ...] | None = None
    has_reconstruction: bool = False
    if assemble_input_and_reconstruction:
        # Detect tokenize-only mode from the buffered windows: when no window
        # carries a non-None ``reconstructed`` entry we skip recon-side
        # padding and concatenation entirely. Mixed buffers (some windows
        # with, some without) would silently misalign segments, so refuse.
        has_recon_flags = [w.get("reconstructed") is not None for w in window_buffer]
        if any(has_recon_flags) and not all(has_recon_flags):
            raise ValueError(
                "Window buffer mixes reconstruction-bearing and reconstruction-"
                "less windows; refusing to concatenate a partial segment "
                "(would silently misalign time bins)."
            )
        has_reconstruction = all(has_recon_flags)

        padded_buffer: list[dict] = []
        for i, w in enumerate(window_buffer):
            sr = _extract_scalar_sampling_rate(w.get("metadata", {}).get("sampling_rate"))
            if sr is None:
                # Fail loud — sampling_rate must be present in metadata for the
                # tolerance computation. If we get here without it,
                # the upstream pipeline did not provide it.
                raise ValueError(
                    f"Window {i} is missing 'sampling_rate' in metadata; cannot "
                    f"compute mismatch-pad tolerance. Window keys: {list(w.keys())}, "
                    f"metadata keys: {list(w.get('metadata', {}).keys())}."
                )
            allow_large_mismatch = is_interval_windowed or (
                w.get("token_start_times") is not None and w.get("token_end_times") is not None
            )
            pad_result = _pad_window_for_small_mismatch(
                w,
                sampling_rate=sr,
                is_interval_windowed=allow_large_mismatch,
            )
            if allow_large_mismatch:
                padded_w, n_padded, is_large_mismatch = cast(
                    tuple[dict[str, Any], int, bool], pad_result
                )
            else:
                padded_w, n_padded = cast(tuple[dict[str, Any], int], pad_result)
                is_large_mismatch = False
            if n_padded > 0:
                if is_large_mismatch:
                    n_large_mismatch_bins_total += n_padded
                    n_large_mismatch_windows += 1
                else:
                    n_zero_padded_bins_total += n_padded
                    n_padded_windows += 1
                logger.debug(
                    "Padded window %d in segment by %d bin(s) "
                    "(orig=%d, recon=%d → max=%d, large=%s).",
                    i,
                    n_padded,
                    w["original"].shape[1],
                    w["reconstructed"].shape[1] if w.get("reconstructed") is not None else -1,
                    padded_w["original"].shape[1],
                    is_large_mismatch,
                )
            padded_buffer.append(padded_w)
        window_buffer = padded_buffer
        if is_interval_windowed and n_large_mismatch_windows:
            logger.warning(
                "Interval-windowing path padded %d window(s) by %d bin(s) "
                "because original/reconstructed lengths differed structurally.",
                n_large_mismatch_windows,
                n_large_mismatch_bins_total,
            )

        # --- Concatenate along time (dim=1) ---
        original = torch.cat([w["original"] for w in window_buffer], dim=1)
        if has_reconstruction:
            reconstructed = torch.cat([w["reconstructed"] for w in window_buffer], dim=1)

        masks = [w["mask"] for w in window_buffer if w.get("mask") is not None]
        pad_and_gap_mask = torch.cat(masks, dim=1) if masks else None
    else:
        # Interval-windowing path: uneven windows lead to incomplete
        # reconstruction within each window, so original and reconstructed
        # cannot be aligned at the segment level. Skip per-window padding
        # and time-axis concatenation; populate ``original_shape`` directly
        # from the unconcatenated buffer so downstream weighting still
        # works.
        n_units = int(window_buffer[0]["original"].shape[0])
        total_time_bins = int(sum(int(w["original"].shape[1]) for w in window_buffer))
        original_shape = (n_units, total_time_bins)

    # --- Stack latents along new window dimension (dim=0) ---
    # After overlap trimming, windows may have different token counts,
    # so stacking isn't always possible. Fall back to list in that case.
    latent_list = [w["latents"] for w in window_buffer if w.get("latents") is not None]
    latents: torch.Tensor | list[torch.Tensor] | None = None
    if latent_list:
        shapes = {latent.shape for latent in latent_list}
        if len(shapes) == 1:
            latents = torch.stack(latent_list, dim=0)
        else:
            latents = latent_list

    # --- Concatenate token start/end/center times ---
    has_token_starts = any("token_start_times" in w for w in window_buffer)
    token_start_times: torch.Tensor | None = None
    token_end_times: torch.Tensor | None = None
    token_timestamps: torch.Tensor | None = None
    if has_token_starts:
        start_list = [w["token_start_times"] for w in window_buffer if "token_start_times" in w]
        end_list = [w["token_end_times"] for w in window_buffer if "token_end_times" in w]
        if start_list:
            token_start_times = torch.cat(start_list, dim=0)
            token_end_times = torch.cat(end_list, dim=0)
    has_token_timestamps = any("token_timestamps" in w for w in window_buffer)
    if has_token_timestamps:
        ts_list = [w["token_timestamps"] for w in window_buffer if "token_timestamps" in w]
        if ts_list:
            token_timestamps = torch.cat(ts_list, dim=0)

    # --- Concatenate behavioral data (token resolution — categoricals only) ---
    collated_behavioral_token_res: dict[str, np.ndarray] | None = None
    has_behavioral_token_res = any(w.get("behavioral_data_token_res") for w in window_buffer)
    if has_behavioral_token_res:
        collated_behavioral_token_res = {}
        all_keys: set[str] = set()
        for w in window_buffer:
            if w.get("behavioral_data_token_res"):
                all_keys.update(w["behavioral_data_token_res"].keys())
        for key in sorted(all_keys):
            arrays = []
            for w in window_buffer:
                bd = w.get("behavioral_data_token_res")
                if bd is not None and key in bd:
                    arrays.append(bd[key])
            assert len(arrays) == len(window_buffer), (
                f"Token-res behavioral field '{key}' present in {len(arrays)} windows "
                f"but expected {len(window_buffer)}. All windows must have "
                f"the same behavioral fields."
            )
            collated_behavioral_token_res[key] = np.concatenate(arrays, axis=0)

    # --- Concatenate behavioral data (input resolution — all fields) ---
    collated_behavioral_input_res: dict[str, np.ndarray] | None = None
    has_behavioral_input_res = any(w.get("behavioral_data_input_res") for w in window_buffer)
    if has_behavioral_input_res:
        collated_behavioral_input_res = {}
        all_keys_ir: set[str] = set()
        for w in window_buffer:
            if w.get("behavioral_data_input_res"):
                all_keys_ir.update(w["behavioral_data_input_res"].keys())
        for key in sorted(all_keys_ir):
            arrays = []
            for w in window_buffer:
                bd = w.get("behavioral_data_input_res")
                if bd is not None and key in bd:
                    arrays.append(bd[key])
            assert len(arrays) == len(window_buffer), (
                f"Input-res behavioral field '{key}' present in {len(arrays)} "
                f"windows but expected {len(window_buffer)}."
            )
            collated_behavioral_input_res[key] = np.concatenate(arrays, axis=0)

    session_id = window_buffer[0]["session_id"]
    metadata = dict(window_buffer[0].get("metadata", {}))
    segment_absolute_start = float(window_buffer[0]["absolute_start"])
    # Track per-window mismatch padding so consumers can audit how much
    # reconstruction signal was zero-padded to make shapes match.
    metadata["n_zero_padded_bins_total"] = n_zero_padded_bins_total
    metadata["n_padded_windows"] = n_padded_windows
    metadata["n_large_mismatch_bins_total"] = n_large_mismatch_bins_total
    metadata["n_large_mismatch_windows"] = n_large_mismatch_windows
    if not skip_gap_check:
        metadata["n_gaps"] = n_gaps
        metadata["gap_seconds"] = gap_seconds
    if assemble_input_and_reconstruction and has_reconstruction:
        # --- Post-pad shape invariant ---
        # The padding logic above is supposed to enforce equal time lengths
        # between original and reconstructed. This assert catches bugs where
        # it didn't (e.g. a future code path that bypasses the helper).
        assert original is not None and reconstructed is not None
        assert original.shape[1] == reconstructed.shape[1], (
            f"original/reconstructed time-bin mismatch after _flush_segment "
            f"padding: {original.shape[1]} vs {reconstructed.shape[1]} "
            f"(cumulative {n_zero_padded_bins_total} bins padded across "
            f"{n_padded_windows} windows). This is an internal invariant — "
            f"_pad_window_for_small_mismatch should already have made these "
            f"equal."
        )

    return CollatedValidationSegment(
        session_id=session_id,
        original=original,
        reconstructed=reconstructed,
        pad_and_gap_mask=pad_and_gap_mask,
        metadata=metadata,
        segment_index=0,  # Caller can override if needed
        is_partial=False,  # Caller can override if needed
        latents=latents,
        behavioral_data_token_res=collated_behavioral_token_res,
        behavioral_data_input_res=collated_behavioral_input_res,
        token_timestamps=token_timestamps,
        token_start_times=token_start_times,
        token_end_times=token_end_times,
        segment_absolute_start=segment_absolute_start,
        original_shape=original_shape,
    )


def _extract_reconstruction(output: Any, *, require: bool = True) -> torch.Tensor | None:
    """Extract the reconstruction tensor from tokenizer output.

    Handles:
    - dict (from reconstruct_from_input with return_latents=True). The
      ``reconstruction``/``reconstructed`` key is optional when
      ``require=False`` so tokenize-only tokenizers (no decoder) can flow
      through the eval pipeline; in that case this returns ``None``.
    - ReconstructionOutput (from reconstruct_from_input without return_latents)
    - plain Tensor

    """
    if isinstance(output, dict):
        reconstructed = output.get("reconstruction", output.get("reconstructed"))
        if reconstructed is None:
            if require:
                raise ValueError(
                    "tokenizer_fun returned dict without 'reconstruction' or 'reconstructed' key."
                )
            return None
        return reconstructed
    elif isinstance(output, ReconstructionOutput):
        return output.reconstruction
    elif isinstance(output, torch.Tensor):
        return output
    else:
        raise TypeError(
            f"tokenizer_fun returned unexpected type {type(output).__name__}. "
            f"Expected dict, torch.Tensor, or ReconstructionOutput."
        )


def _extract_latents(output: Any) -> torch.Tensor | None:
    """Extract latents from tokenizer output, if present."""
    if isinstance(output, dict):
        return output.get("latents")
    else:
        return None


def _apply_pad_and_gap_mask(
    spike_matrix: torch.Tensor | None, pad_and_gap_mask: torch.Tensor | None
) -> torch.Tensor | None:
    """Apply padding mask to filter out invalid units. Expects shape (units, time).

    ``spike_matrix`` may be ``None`` for tokenize-only paths (no
    reconstruction available); in that case ``None`` is returned unchanged.
    """
    if spike_matrix is None:
        return None
    if pad_and_gap_mask is None:
        return spike_matrix
    if pad_and_gap_mask.dtype is not torch.bool:
        # Accept integer masks with only 0/1 values (e.g. from collation)
        unique_vals = pad_and_gap_mask.unique()
        assert all(v in (0, 1) for v in unique_vals.tolist()), (
            f"pad_and_gap_mask has non-boolean dtype {pad_and_gap_mask.dtype} "
            f"with values other than 0/1: {unique_vals.tolist()}"
        )
        pad_and_gap_mask = pad_and_gap_mask.bool()
    if not pad_and_gap_mask.all():
        raise AssertionError(
            "No masking allowed for validation. (If that's needed we need to "
            "forward the masks to the metrics.) For removing padding from "
            "start/end of a full segment, we need to ensure that the mask is "
            "consistent with that, only dropping full units and/or "
            "time bins at the start/end of the segment."
        )
    return spike_matrix


def _matched_decoding_target_fields(
    decoding_target_fields: set[str | frozenset[str]] | None,
    *,
    behavioral_data_input_res: dict[str, np.ndarray] | None,
    behavioral_data_token_res: dict[str, np.ndarray] | None = None,
) -> set[str]:
    """Return the decoding fields on this window that a decoder can consume.

    Used to detect, per session, whether a registered downstream decoding
    metric can run on the session's behavioral data — in which case
    ``max_eval_seconds`` is bypassed for that session (truncating a decoder's
    input is invalid). A ``str`` entry matches when that field is present; a
    ``frozenset`` entry (one metric's required fields, see
    ``required_behavioral_fields``) matches only when all of them are present.
    """
    if not decoding_target_fields:
        return set()
    present_keys: set[str] = set()
    if behavioral_data_input_res is not None:
        present_keys.update(behavioral_data_input_res.keys())
    if behavioral_data_token_res is not None:
        present_keys.update(behavioral_data_token_res.keys())
    matched: set[str] = set()
    for entry in decoding_target_fields:
        required = {entry} if isinstance(entry, str) else entry
        if required <= present_keys:
            matched |= required
    return matched


def iter_collated_validation_segments(
    dataloader: DataLoader,
    tokenizer_fun: Callable[[BinnedSpikesInput | RatesInput], Any],
    *,
    tokenizer_spec: TokenizerSpec | None = None,
    token_bin_size_ms: float | None = None,
    window_length_seconds: float | None = None,
    trim_token_overlap: bool = False,
    drop_overlapping_windows: bool = True,
    device: torch.device | None = None,
    max_windows: int | None = None,
    max_bytes: int | None = None,
    max_session_windows: int | None = None,
    max_eval_seconds: float | None = None,
    decoding_target_fields: set[str | frozenset[str]] | None = None,
    preserve_overlap_state_on_partial_flush: bool = False,
    _on_batch_done: Callable[[], None] | None = None,
) -> Iterator[CollatedValidationSegment]:
    """Iterate over validation windows and yield per-session collated segments.

    A "segment" is a concatenation of multiple validation windows (for a given
    session) into one longer chunk; windows separated by time gaps (multi-interval
    sampling domains) are joined end-to-end and the gaps recorded in the segment
    metadata. This is useful because many metrics are noisy or undefined on very
    short windows.

    Pipeline steps:
        [1+2] DataLoader batch (sampler + collation + padding) + move to device
        [3a] Tokenizer encode + reconstruct (via tokenizer_fun)
        [3b] Per-window unbatch
        [4] Resample behavioral data to token_timestamps (per window)
        [5a] Strip input padding  (_strip_input_padding)
        [5b] Strip token padding  (_strip_token_padding)
        [6a] Trim input overlap   (_trim_input_overlap)
        [6b] Trim token overlap   (_trim_token_overlap) — includes behavioral
        [7a] Buffer the window
        [7b] Session change or limit → _flush_segment → yield

    Parameters
    ----------
    dataloader:
        DataLoader yielding BinnedSpikesInput batches. Each batch must have
        absolute_start and absolute_end set.
    tokenizer_fun:
        Function that takes a BinnedSpikesInput and returns a tensor (reconstruction)
        or a dict with keys "reconstruction" (: Tensor), "latents" (: Tensor),
        and token timestamp entries ("token_timestamps", "token_start_time",
        "token_end_time" — all three required when latents are present). When
        ``tokenizer_spec`` is provided the iterator routes this function through
        :func:`chunked_tokenizer_fn` to wrap ``tokenizer_fun`` once at the
        start of iteration — fixed-input tokenizers get per-batch
        pre-chunking, variable-input tokenizers are a pass-through.
        Typically ``tokenizer_fun`` is ``tokenizer.reconstruct_from_input``
        with ``return_latents=True``.
    token_bin_size_ms:
        Temporal extent of each token in milliseconds. Used by the
        sanity check (_dbg_check_token_count_via_bins_per_token).
    trim_token_overlap:
        Whether to perform token-level overlap trimming (step 6b). Default False.
        When True, overlapping tokens are removed from the head of each window,
        which can produce windows with different token counts (making latent
        stacking impossible — structure preservation metrics will return NaN).
        When False, overlapping windows keep all tokens — timestamps make
        the overlap explicit. Padding stripping (step 5b) is always performed
        regardless of this setting. When True and token_start_time/token_end_time
        are not available, falls back to token_timestamps (center times).
    drop_overlapping_windows:
        If True (default), windows whose input time range overlaps with the
        previous window are excluded from the latent/token/behavioral segment
        (only the later window of each overlapping pair is dropped; the
        earlier is kept). This avoids duplicate time points in latent-space
        metrics like RSA, which would create artificial structure. Input-level
        overlap trimming (step 6a) still applies to the binned data.
    device:
        Device to run inference on.
    max_windows:
        Flush a segment after this many *windows* have been buffered for the
        current session. If None, do not flush based on window count.
        Note: when this triggers, the yielded segment is marked as `is_partial=True`
        (meaning it was cut early due to a limit, not necessarily that it is shorter
        than a "full" session).
    max_bytes:
        Flush a segment after accumulating approximately this many bytes in the
        buffers for the current session. If None, do not flush based on memory.
        Note: when this triggers, the yielded segment is marked as `is_partial=True`.
    max_session_windows:
        Stop evaluating a session after this many windows have been buffered and
        yielded for it. Unlike ``max_windows``, this is a true per-session cap:
        once reached, remaining windows of that session are skipped. If None,
        do not cap by processed-window count.
    max_eval_seconds:
        Maximum duration (in seconds) of data to evaluate per session. Once a
        session's accumulated window durations reach this limit, remaining
        windows for that session are skipped. If None, evaluate the full
        session.  Useful for bounding runtime on very long recordings.
    decoding_target_fields:
        Optional set of what registered downstream decoding metrics need:
        field names (e.g. ``{"cursor_pos"}``) or, per metric, the frozenset
        of fields it cannot compute without (``required_behavioral_fields``).
        When provided, ``max_eval_seconds`` is suppressed for any session
        whose per-window behavioral data satisfies at least one entry —
        those sessions run full-length so the decoder has enough data, while
        other sessions (including ones carrying a target field that no
        configured metric can consume) stay capped. Detection is lazy (flips
        on the first matching window). When None, ``max_eval_seconds``
        applies uniformly to every session.
    Yields
    ------
    CollatedValidationSegment
    """
    # Default path: each session collates into one segment (or splits via
    # max_windows / max_bytes). max_eval_seconds caps cumulative per-session
    # window duration; once hit, session_capped skips remaining windows for
    # that session.

    # ── State variables ─────────────────────────────────────────────────
    current_session_id: str | None = None
    window_buffer: list[dict] = []
    segment_index = 0
    next_window_group_id_token_res = 0
    buffered_bytes = 0
    session_window_count = 0
    session_accumulated_seconds: float = 0.0  # for max_eval_seconds cap
    session_capped: bool = False  # True → skip remaining windows for this session
    # True once we've seen behavioral data for a target field used by a
    # registered downstream decoding metric → suppress max_eval_seconds for
    # this session so the decoder gets full-length data.
    session_is_decoding_capable: bool = False
    # True once a window's behavior is found not binned to its input bins →
    # this session's behavioral data is dropped (see step [5d]).
    session_behavior_unaligned: bool = False
    # True once neuron_split_collate chunks were seen in this session (see the
    # neuron-split guard below).
    session_has_neuron_split: bool = False

    # Overlap tracking — uses absolute timestamps from batch, not sampler
    prev_absolute_end: float | None = None
    prev_last_token_end_time: float | None = None

    def flush(
        is_partial: bool,
        *,
        reset_overlap_state: bool = True,
    ) -> Iterator[CollatedValidationSegment]:
        nonlocal window_buffer, segment_index, buffered_bytes
        nonlocal prev_absolute_end, prev_last_token_end_time

        if current_session_id is None:
            return iter(())

        if not window_buffer:
            if reset_overlap_state:
                prev_absolute_end = None
                prev_last_token_end_time = None
            return iter(())

        segment = _flush_segment(window_buffer, drop_overlapping_windows=drop_overlapping_windows)
        window_buffer = []
        buffered_bytes = 0
        if reset_overlap_state:
            prev_absolute_end = None
            prev_last_token_end_time = None

        # Override segment_index and is_partial (_flush_segment sets defaults)
        segment = CollatedValidationSegment(
            session_id=segment.session_id,
            original=segment.original,
            reconstructed=segment.reconstructed,
            pad_and_gap_mask=segment.pad_and_gap_mask,
            metadata=segment.metadata,
            segment_index=segment_index,
            is_partial=is_partial,
            latents=segment.latents,
            behavioral_data_token_res=segment.behavioral_data_token_res,
            behavioral_data_input_res=segment.behavioral_data_input_res,
            token_timestamps=segment.token_timestamps,
            token_start_times=segment.token_start_times,
            token_end_times=segment.token_end_times,
            segment_absolute_start=segment.segment_absolute_start,
        )
        segment_index += 1
        return iter((segment,))

    def buffer_window(window_dict: dict[str, Any]) -> None:
        nonlocal buffered_bytes, next_window_group_id_token_res

        if (
            window_dict.get("latents") is not None
            and window_dict.get("behavioral_data_input_res") is not None
        ):
            window_dict = _merge_window_parts(
                [window_dict],
                window_group_id_token_res=next_window_group_id_token_res,
            )
            next_window_group_id_token_res += 1

        window_buffer.append(window_dict)
        original_window = window_dict["original"]
        reconstructed_window = window_dict.get("reconstructed")
        mask_window = window_dict.get("mask")
        buffered_bytes += original_window.numel() * original_window.element_size()
        if reconstructed_window is not None:
            buffered_bytes += reconstructed_window.numel() * reconstructed_window.element_size()
        if mask_window is not None:
            buffered_bytes += mask_window.numel() * mask_window.element_size()

    # Startup-time fail-loud: the rest of the pipeline assumes matched bin
    # counts between original and reconstruction, which requires
    # token_bin_size_ms to evenly divide the configured input window length.
    _assert_token_bin_divides_window_length(window_length_seconds, token_bin_size_ms)

    # Wrap the tokenizer once: for fixed-input tokenizers (spec is provided
    # and the spec demands a fixed window) the wrapper pre-chunks each batch
    # along time. For variable-input tokenizers it's a (near) no-op. See
    # ``chunked_tokenizer_fn`` for the contract.
    if tokenizer_spec is not None:
        tokenizer_fun = chunked_tokenizer_fn(tokenizer_fun, spec=tokenizer_spec)

    for batch in dataloader:
        # BinnedSpikesInput has shape (batch, time, units)
        if isinstance(batch, BinnedSpikesInput):
            spike_matrix = batch.spike_matrix
        elif isinstance(batch, RatesInput):
            spike_matrix = batch.rate_matrix
        else:
            raise ValueError(f"Unsupported batch type: {type(batch)}.")

        # Ensure batch dim
        pad_and_gap_mask = batch.pad_and_gap_mask
        spike_matrix, pad_and_gap_mask = _ensure_batch_dim(spike_matrix, pad_and_gap_mask)

        # Fast-path: if the current session is already capped and every idx in
        # this batch belongs to that same session, skip the whole batch — no
        # device transfer, no tokenizer call. The per-idx loop would `continue`
        # past every window anyway; this avoids the dominant per-batch cost.
        if session_capped and current_session_id is not None:
            batch_session_ids = _normalize_session_ids(batch.session_id, spike_matrix.shape[0])
            if all(sid == current_session_id for sid in batch_session_ids):
                if _on_batch_done is not None:
                    _on_batch_done()
                continue

        # Validate: absolute_start and absolute_end must be present
        assert batch.absolute_start is not None, (
            "BinnedSpikesInput must have absolute_start set for validation collation. "
            "This is set by the dataset/transform pipeline from temporaldata."
        )
        assert batch.absolute_end is not None, (
            "BinnedSpikesInput must have absolute_end set for validation collation."
        )

        session_ids = _normalize_session_ids(batch.session_id, spike_matrix.shape[0])
        unique_session_ids = list(dict.fromkeys(session_ids))
        if (
            session_capped
            and current_session_id is not None
            and len(unique_session_ids) == 1
            and unique_session_ids[0] == current_session_id
        ):
            if _on_batch_done is not None:
                _on_batch_done()
            continue

        # Move batch to device if needed
        if device is not None:
            batch = move_data_to_device(batch, device)

        # ── [3a] Tokenizer encode + reconstruct ─────────────────────────
        # ``tokenizer_fun`` has already been wrapped by
        # ``chunked_tokenizer_fn`` (above) when a tokenizer_spec was passed
        # in, so we can just call it directly. A session-boundary partial
        # window can be shorter than one tokenizer chunk for a chunked
        # (variable_t=False) tokenizer; the chunker raises a specific
        # ValueError for this. Skip just this batch instead of aborting the
        # whole per-session eval (mirrors the replica iterator guard).
        try:
            output = tokenizer_fun(batch)
        except ValueError as exc:
            if "shorter than one tokenizer chunk" in str(exc):
                logger.info(
                    "validation collation: dropping runtime partial window shorter "
                    "than one tokenizer chunk at tokenize time: %s",
                    exc,
                )
                if _on_batch_done is not None:
                    _on_batch_done()
                continue
            raise
        batch_token_start_times = None
        batch_token_end_times = None
        batch_token_timestamps = None

        reconstructed = _extract_reconstruction(output, require=False)
        batch_latents: Any = _extract_latents(output)

        if isinstance(output, dict):
            if "token_start_time" in output:
                batch_token_start_times = output["token_start_time"].cpu()
            if "token_end_time" in output:
                batch_token_end_times = output["token_end_time"].cpu()
            if "token_timestamps" in output:
                batch_token_timestamps = output["token_timestamps"].cpu()

        if isinstance(batch_latents, torch.Tensor):
            batch_latents = batch_latents.detach().cpu()
        elif isinstance(batch_latents, list):
            # Ragged per-sample latents (e.g. when sessions in the batch
            # have different unit counts and the tokenizer returns a
            # length-B list of [T, U_i, D] tensors instead of one stacked
            # tensor). Separate feature from window chunking.
            cpu_latents: list[torch.Tensor | None] = []
            for latent in batch_latents:
                if latent is None:
                    cpu_latents.append(None)
                elif isinstance(latent, torch.Tensor):
                    cpu_latents.append(latent.detach().cpu())
                else:
                    raise TypeError(
                        "Expected list latents to contain torch.Tensor or None, "
                        f"got {type(latent).__name__}."
                    )
            batch_latents = cpu_latents
        elif batch_latents is not None:
            raise TypeError(
                f"Expected latents to be a torch.Tensor, list, or None, got "
                f"{type(batch_latents).__name__}."
            )

        if reconstructed is not None:
            reconstructed = reconstructed.detach().cpu()
        spike_matrix = spike_matrix.detach().cpu()
        pad_and_gap_mask = (
            pad_and_gap_mask.detach().cpu() if pad_and_gap_mask is not None else None
        )

        sampling_rate = _extract_sampling_rate(batch)

        # ── [3b] Per-window unbatch and processing ──────────────────────
        for idx in range(spike_matrix.shape[0]):
            session_id = session_ids[idx]

            # Session tracking: detect session change → flush
            if current_session_id is None:
                current_session_id = session_id
                next_window_group_id_token_res = 0
                session_window_count = 0
                session_accumulated_seconds = 0.0
                session_capped = False
                session_is_decoding_capable = False
                session_behavior_unaligned = False
                session_has_neuron_split = False
            if session_id != current_session_id:
                yield from flush(is_partial=False)
                current_session_id = session_id
                next_window_group_id_token_res = 0
                session_window_count = 0
                session_accumulated_seconds = 0.0
                session_capped = False
                session_is_decoding_capable = False
                session_behavior_unaligned = False
                session_has_neuron_split = False

            # Skip windows if this session has hit the duration cap
            if session_capped:
                continue

            # Extract per-window absolute timestamps
            abs_s = batch.absolute_start
            abs_e = batch.absolute_end
            assert abs_s is not None and abs_e is not None
            window_absolute_start = float(abs_s[idx].item() if abs_s.ndim > 0 else abs_s.item())
            window_absolute_end = float(abs_e[idx].item() if abs_e.ndim > 0 else abs_e.item())

            # Transpose from (time, units) to (units, time) for internal processing
            # BinnedSpikesInput[idx] has shape (time, units), we need (units, time)
            original_window = spike_matrix[idx].T
            reconstructed_window = reconstructed[idx].T if reconstructed is not None else None
            mask_window = pad_and_gap_mask[idx].T if pad_and_gap_mask is not None else None

            # Unbatch behavioral data for this window
            window_behavioral_data: dict[str, np.ndarray] | None = None
            if batch.behavioral_data is not None:
                # The custom collate stores behavioral_data as list[dict|None].
                # If torch_brain's collate was used instead, it may be a plain
                # dict (or empty dict).  Handle both gracefully.
                if isinstance(batch.behavioral_data, dict):
                    raw = batch.behavioral_data  # same dict shared across batch
                elif isinstance(batch.behavioral_data, (list, tuple)):
                    raw = batch.behavioral_data[idx]
                else:
                    raw = None

                if isinstance(raw, dict) and raw:
                    window_behavioral_data = {}
                    for key, value in raw.items():
                        if isinstance(value, torch.Tensor):
                            value = value.cpu().numpy()
                        if not isinstance(value, np.ndarray):
                            continue
                        if value.ndim < 1:
                            continue
                        window_behavioral_data[key] = value
                    if not window_behavioral_data:
                        window_behavioral_data = None

            # ── [3c] Behavior must be binned like the input ────────────────
            # Behavioral rows are paired with input bins by index (step [4] and
            # the decoders). Raw, unbinned behavior (e.g. irregularly sampled
            # cursor data) would silently misalign, so drop the whole session's
            # behavioral data instead — including windows already buffered, so
            # fields stay consistent across the segment.
            if window_behavioral_data is not None and not session_behavior_unaligned:
                # Collate pads every window to the batch's longest; behavior is
                # not padded. A window shorter than that (truncated by a domain
                # gap) may carry rows for its own span: accept its own bin count
                # too (either rounding of a partial last bin). Step [5c] skips
                # such windows when tokens are present.
                n_window_bins = spike_matrix[idx].shape[0]
                own_bins = (window_absolute_end - window_absolute_start) * sampling_rate
                allowed_rows = {n_window_bins}
                if own_bins < n_window_bins - 1e-6:
                    allowed_rows |= {int(np.floor(own_bins + 1e-6)), int(np.ceil(own_bins - 1e-6))}
                unaligned = sorted(
                    k for k, v in window_behavioral_data.items() if len(v) not in allowed_rows
                )
                if unaligned:
                    logger.warning(
                        "[session=%s] behavioral fields %s are not binned to the input "
                        "(window at %.6fs has %d bins); dropping this session's "
                        "behavioral data so no decoder pairs misaligned rows.",
                        session_id,
                        unaligned,
                        window_absolute_start,
                        n_window_bins,
                    )
                    session_behavior_unaligned = True
                    session_is_decoding_capable = False
                    for buffered in window_buffer:
                        buffered["behavioral_data_input_res"] = None
                        buffered["behavioral_data_token_res"] = None
            if session_behavior_unaligned:
                window_behavioral_data = None

            # ── [4] Behavioral data alignment ─────────────────────────────
            # Behavioral data arrives at input-bin resolution. We produce two
            # versions:
            #   (a) categorical fields resampled to token-center timestamps
            #       (for phase/trial filtering in downstream metrics)
            #   (b) all fields kept at input resolution (for regression/
            #       classification targets in resolution-independent metrics)
            #
            # Only (a) goes through strip/trim steps [5b] and [6b] — it's at
            # token resolution and aligns with token masks. (b) stays untouched
            # and gets aligned to token chunks later by timestamps.
            window_behavioral_data_token_res: dict[str, np.ndarray] | None = None
            window_behavioral_data_input_res: dict[str, np.ndarray] | None = None

            if window_behavioral_data is not None and batch_token_timestamps is not None:
                from ntt.evaluation.utils.alignment import resample_fields_to_token_timestamps

                window_token_ts = batch_token_timestamps[idx]
                n_input_bins = spike_matrix[idx].shape[0]  # (time, units)
                source_timestamps = (
                    window_absolute_start
                    + (np.arange(n_input_bins, dtype=np.float64) + 0.5) / sampling_rate
                )
                target_timestamps = window_token_ts.numpy().astype(np.float64)

                # (a) Categorical fields at token resolution (for filtering)
                window_behavioral_data_token_res = resample_fields_to_token_timestamps(
                    window_behavioral_data,
                    source_timestamps,
                    target_timestamps,
                )

                # (b) All fields at input resolution (unchanged)
                window_behavioral_data_input_res = window_behavioral_data

            elif window_behavioral_data is not None:
                # No token timestamps — keep both as input-res (no resampling possible)
                window_behavioral_data_input_res = window_behavioral_data

            # ── [5a] Strip input padding ────────────────────────────────
            # Strip batch-level padding before buffering.  The DataLoader's
            # collate pads each batch to its max (time, units).  When sessions
            # with different unit counts share a batch, windows get padded to
            # the batch-max.  We use the mask to recover the true dimensions so
            # that all windows for a session have consistent shapes.
            pre_strip_valid_units: torch.Tensor | None = None
            if mask_window is not None and not mask_window.all():
                # Compute valid_units BEFORE stripping (post-strip mask is all-True
                # for a clean window — only trailing padding was removed)
                pre_strip_valid_units = mask_window.any(dim=1)  # (units,) bool
                original_window, reconstructed_window, mask_window = _strip_input_padding(
                    original_window, reconstructed_window, mask_window
                )
                # Trailing padding is gone; any remaining False is an interior
                # temporal gap (zero-filled bins between disjoint valid intervals).
                # Skip the window rather than feed garbage gap bins into metrics —
                # mirrors the numel()==0 skip just below.
                if mask_window is not None and not mask_window.all():
                    continue

            if original_window.numel() == 0:
                continue

            # ── [5b] Handle latents, strip token padding ────────────────
            window_latents: torch.Tensor | None = None
            window_token_start_times: torch.Tensor | None = None
            window_token_end_times: torch.Tensor | None = None
            window_token_timestamps: torch.Tensor | None = None

            if batch_latents is not None:
                window_latents = batch_latents[idx]
                # If latent has a unit axis matching unpadded units, strip it too.
                # Batch latents are [B, T, U, D], so a window is [T, U, D] after indexing.
                if pre_strip_valid_units is not None and window_latents is not None:
                    n_valid = len(pre_strip_valid_units)
                    if window_latents.ndim == 3 and window_latents.shape[1] == n_valid:
                        window_latents = window_latents[:, pre_strip_valid_units, :]

                # token_bin_size_ms is only used for the debug sanity check
                # below. Dynamic tokenizers with variable token durations may
                # leave it as None — the core alignment is timestamp-based.
                # if token_bin_size_ms is None:
                #     raise ValueError(
                #         "token_bin_size_ms is required when latents are present. "
                #         "Pass the tokenizer's spec.token_bin_size_ms."
                #     )

                # Extract per-window token timestamps
                if window_latents is not None:
                    if batch_token_start_times is not None:
                        assert batch_token_end_times is not None
                        window_token_start_times = batch_token_start_times[idx]
                        window_token_end_times = batch_token_end_times[idx]

                    assert batch_token_timestamps is not None, (
                        "token_timestamps (center times) are required when latents "
                        "are present. The tokenizer must provide them."
                    )
                    window_token_timestamps = batch_token_timestamps[idx]
                    assert window_token_timestamps is not None, (
                        "token_timestamps (center times) are required when latents "
                        "are present. The tokenizer must provide them."
                    )

                    # Effective start/end times for stripping and overlap trimming.
                    # Uses start/end times if available, falls back to center times.
                    if window_token_start_times is not None:
                        effective_token_starts = window_token_start_times
                        effective_token_ends = window_token_end_times
                    else:
                        effective_token_starts = window_token_timestamps
                        effective_token_ends = window_token_timestamps

                    # Strip tokens that extend past absolute_end (padding).
                    # Always performed.
                    assert effective_token_ends is not None
                    strip_result = _strip_token_padding(
                        window_latents,
                        effective_token_starts,
                        effective_token_ends,
                        absolute_end=window_absolute_end,
                        behavioral_data=window_behavioral_data_token_res,
                    )
                    keep_mask = strip_result["keep_mask"]
                    window_latents = cast(torch.Tensor, strip_result["latents"])
                    # Stripped effective times: real start/end times if those
                    # existed, otherwise the stripped center times.
                    effective_token_starts = strip_result["token_start_times"]
                    effective_token_ends = strip_result["token_end_times"]
                    if window_token_start_times is not None:
                        window_token_start_times = effective_token_starts
                        window_token_end_times = effective_token_ends
                    window_token_timestamps = window_token_timestamps[keep_mask]
                    window_behavioral_data_token_res = strip_result["behavioral_data"]

                    if window_latents.numel() > 0:
                        assert not torch.isnan(window_latents[-1]).any(), (
                            "Last token contains NaN after padding stripping."
                        )

                    # Debug sanity check: bins_per_token cross-validation
                    if token_bin_size_ms is not None:
                        _dbg_check_token_count_via_bins_per_token(
                            n_valid_tokens=window_latents.shape[0],
                            absolute_start=window_absolute_start,
                            absolute_end=window_absolute_end,
                            token_bin_size_ms=token_bin_size_ms,
                            sampling_rate=sampling_rate,
                            stripped_on_center_times=window_token_start_times is None,
                        )

            # ── [5c] Keep tokens and input bins index-aligned ──────────────
            # Decoders pair tokens with input bins / behavior rows by index over
            # the concatenated segment, so each buffered window must span exactly
            # n_tokens * bins_per_token bins. A window truncated by a recording-
            # domain gap keeps a partial tail (e.g. 98 bins but 9 tokens of 10
            # bins) that would offset every later window: skip it, as windows
            # with interior gaps are skipped above.
            if window_latents is not None and token_bin_size_ms is not None:
                bins_per_token = token_bin_size_ms * sampling_rate / 1000.0
                if abs(bins_per_token - round(bins_per_token)) < 1e-6:
                    n_token_bins = window_latents.shape[0] * round(bins_per_token)
                    if n_token_bins != original_window.shape[1]:
                        logger.info(
                            "[session=%s] skipping truncated window at %.6fs: %d input "
                            "bins but %d tokens x %d bins.",
                            session_id,
                            window_absolute_start,
                            original_window.shape[1],
                            window_latents.shape[0],
                            round(bins_per_token),
                        )
                        continue

            window_metadata = _extract_metadata(batch, idx)

            # Neuron-split collation can emit several batch elements with the
            # same time extent but different unit subsets. Treat those as
            # separate validation segments instead of trimming them as temporal
            # overlaps. Chunks arrive interleaved ([w0·A, w0·B, w1·A, ...]), so
            # once a split is seen, every unit-subset change in the session
            # flushes too; otherwise w1·A would be appended to w0·B. The flush
            # resets the overlap state, so a chunk is never overlap-trimmed
            # against another chunk's window. Sessions without a split are
            # unaffected (per-window UnitDropout subsets still collate into one
            # segment).
            if (
                prev_absolute_end is not None
                and window_buffer
                and not _unit_ids_equal(
                    window_metadata.get("unit_ids"),
                    window_buffer[-1]["metadata"].get("unit_ids"),
                )
            ):
                if (
                    abs(window_absolute_start - window_buffer[-1]["absolute_start"]) < 1e-9
                    and abs(window_absolute_end - window_buffer[-1]["absolute_end"]) < 1e-9
                ):
                    session_has_neuron_split = True
                if session_has_neuron_split:
                    yield from flush(is_partial=True)

            # Track whether this window has input overlap (for drop_overlapping_windows).
            # Only the later window of each overlapping pair is marked.
            window_has_input_overlap = (
                prev_absolute_end is not None and window_absolute_start < prev_absolute_end - 1e-9
            )

            # ── [6a] Trim input overlap ─────────────────────────────────
            # Uses absolute timestamps from batch for overlap detection
            if prev_absolute_end is not None:
                if mask_window is None:
                    mask_window = torch.ones_like(original_window, dtype=torch.bool)
                original_window, reconstructed_window, mask_window, trimmed_start = (
                    _trim_input_overlap(
                        original_window,
                        reconstructed_window,
                        mask_window,
                        current_absolute_start=window_absolute_start,
                        prev_absolute_end=prev_absolute_end,
                        sampling_rate=sampling_rate,
                    )
                )
                window_absolute_start = trimmed_start

            if original_window.numel() == 0:
                continue

            # ── [6b] Trim token overlap ─────────────────────────────────
            if trim_token_overlap and window_latents is not None:
                token_overlap_result = _trim_token_overlap(
                    latents=window_latents,
                    token_start_times=effective_token_starts,
                    token_end_times=effective_token_ends,
                    prev_last_token_end_time=prev_last_token_end_time,
                    behavioral_data=window_behavioral_data_token_res,
                )
                keep_mask = token_overlap_result["keep_mask"]
                window_latents = cast(torch.Tensor, token_overlap_result["latents"])
                effective_token_starts = token_overlap_result["token_start_times"]
                effective_token_ends = token_overlap_result["token_end_times"]
                if window_token_start_times is not None:
                    window_token_start_times = effective_token_starts
                    window_token_end_times = effective_token_ends
                assert window_token_timestamps is not None
                window_token_timestamps = window_token_timestamps[keep_mask]
                window_behavioral_data_token_res = token_overlap_result["behavioral_data"]

                # When all tokens are trimmed (e.g. full-window tokenizers
                # with a single token per window that falls entirely within
                # the previous window's coverage), discard the latents for
                # this window — the reconstruction is still kept.
                if window_latents.numel() == 0:
                    window_latents = None
                    window_token_start_times = None
                    window_token_end_times = None
                    window_token_timestamps = None

            # ── [7a] Buffer window ─────────────────────────────────────
            single_window = _build_window_part(
                original=original_window,
                reconstructed=reconstructed_window,
                mask=mask_window,
                absolute_start=window_absolute_start,
                sampling_rate=sampling_rate,
                session_id=session_id,
                metadata=window_metadata,
                has_overlap=window_has_input_overlap,
                behavioral_data_input_res=window_behavioral_data_input_res,
                behavioral_data_token_res=window_behavioral_data_token_res,
                latents=window_latents,
                token_start_times=window_token_start_times,
                token_end_times=window_token_end_times,
                token_timestamps=window_token_timestamps,
                start_bin=0,
                end_bin=original_window.shape[1],
            )
            buffer_window(single_window)

            # Decoding-capability detection: once we see any window that carries
            # a behavioral field a registered downstream metric needs, mark this
            # session as decoding-capable so max_eval_seconds is bypassed below.
            if not session_is_decoding_capable:
                matched = _matched_decoding_target_fields(
                    decoding_target_fields,
                    behavioral_data_input_res=window_behavioral_data_input_res,
                    behavioral_data_token_res=window_behavioral_data_token_res,
                )
                if matched:
                    session_is_decoding_capable = True
                    logger.info(
                        "[session=%s] max_eval_seconds bypassed: decoding-capable "
                        "(matched behavioral fields: %s)",
                        current_session_id,
                        sorted(matched),
                    )

            # Update overlap tracking state using the full processed window
            prev_absolute_end = window_absolute_end
            if window_latents is not None:
                assert effective_token_ends.numel() > 0, (
                    "effective_token_ends is empty after stripping/trimming — "
                    "this should not happen since empty windows are skipped above."
                )
                prev_last_token_end_time = float(effective_token_ends[-1].item())

            # ── [11] Track session duration and flush on limits ─────────
            # buffered_bytes is accumulated inside buffer_window.
            window_duration = window_absolute_end - window_absolute_start
            session_window_count += 1
            session_accumulated_seconds += window_duration

            if max_windows is not None and len(window_buffer) >= max_windows:
                yield from flush(
                    is_partial=True,
                    reset_overlap_state=not preserve_overlap_state_on_partial_flush,
                )
            if max_bytes is not None and buffered_bytes >= max_bytes:
                yield from flush(
                    is_partial=True,
                    reset_overlap_state=not preserve_overlap_state_on_partial_flush,
                )
            if max_session_windows is not None and session_window_count >= max_session_windows:
                logger.info(
                    "[session=%s] max_session_windows cap reached at %d window(s); "
                    "skipping remaining windows.",
                    current_session_id,
                    session_window_count,
                )
                yield from flush(
                    is_partial=True,
                    reset_overlap_state=not preserve_overlap_state_on_partial_flush,
                )
                session_capped = True
            if (
                max_eval_seconds is not None
                and not session_is_decoding_capable
                and session_accumulated_seconds >= max_eval_seconds
            ):
                logger.info(
                    "[session=%s] max_eval_seconds cap reached at %.1fs (limit=%.1fs); "
                    "skipping remaining windows.",
                    current_session_id,
                    session_accumulated_seconds,
                    max_eval_seconds,
                )
                carried = _matched_decoding_target_fields(
                    {field for entry in decoding_target_fields or () for field in entry}
                    if decoding_target_fields
                    else None,
                    behavioral_data_input_res=window_behavioral_data_input_res,
                    behavioral_data_token_res=window_behavioral_data_token_res,
                )
                if carried:
                    logger.info(
                        "[session=%s] not exempt from max_eval_seconds: its behavior "
                        "carries %s, but no configured decoding metric has all the "
                        "fields it needs (%s).",
                        current_session_id,
                        sorted(carried),
                        sorted(
                            sorted({e} if isinstance(e, str) else e)
                            for e in decoding_target_fields or ()
                        ),
                    )
                yield from flush(
                    is_partial=True,
                    reset_overlap_state=not preserve_overlap_state_on_partial_flush,
                )
                session_capped = True

        if _on_batch_done is not None:
            _on_batch_done()

    yield from flush(is_partial=False)


def _concatenate_segments(
    segments: list[CollatedValidationSegment],
) -> CollatedValidationSegment:
    """Concatenate multiple segments into one combined segment."""
    if not segments:
        raise ValueError("No segments to concatenate")
    if len(segments) == 1:
        return segments[0]

    # Concatenate latents along first (window) dimension
    latents_list = [s.latents for s in segments if isinstance(s.latents, torch.Tensor)]
    combined_latents = torch.cat(latents_list, dim=0) if latents_list else None

    # Concatenate token start/end/center times — these are absolute times, just cat them
    start_times_list = [s.token_start_times for s in segments if s.token_start_times is not None]
    end_times_list = [s.token_end_times for s in segments if s.token_end_times is not None]
    ts_list = [s.token_timestamps for s in segments if s.token_timestamps is not None]
    combined_start_times = torch.cat(start_times_list, dim=0) if start_times_list else None
    combined_end_times = torch.cat(end_times_list, dim=0) if end_times_list else None
    combined_timestamps = torch.cat(ts_list, dim=0) if ts_list else None

    # Concatenate behavioral data arrays (token resolution)
    combined_behavioral_token_res: dict[str, np.ndarray] | None = None
    if any(s.behavioral_data_token_res for s in segments):
        combined_behavioral_token_res = {}
        all_keys = set()
        for s in segments:
            if s.behavioral_data_token_res:
                all_keys.update(s.behavioral_data_token_res.keys())
        for key in all_keys:
            arrays = []
            offset = 0
            for s in segments:
                if not s.behavioral_data_token_res or key not in s.behavioral_data_token_res:
                    continue
                array = np.asarray(s.behavioral_data_token_res[key])
                if key == "window_group_id_token_res":
                    array = array.astype(np.int32, copy=True)
                    if array.size:
                        array += offset
                        offset = int(array.max()) + 1
                arrays.append(array)
            if arrays:
                combined_behavioral_token_res[key] = np.concatenate(arrays, axis=0)

    # Concatenate behavioral data arrays (input resolution)
    combined_behavioral_input_res: dict[str, np.ndarray] | None = None
    if any(s.behavioral_data_input_res for s in segments):
        combined_behavioral_input_res = {}
        all_keys_ir = set()
        for s in segments:
            if s.behavioral_data_input_res:
                all_keys_ir.update(s.behavioral_data_input_res.keys())
        for key in all_keys_ir:
            arrays = [
                s.behavioral_data_input_res[key]
                for s in segments
                if s.behavioral_data_input_res and key in s.behavioral_data_input_res
            ]
            if arrays:
                combined_behavioral_input_res[key] = np.concatenate(arrays, axis=0)

    # segment_absolute_start: only meaningful for same-session concatenation
    session_ids = {s.session_id for s in segments}
    if len(session_ids) == 1:
        starts = [
            s.segment_absolute_start for s in segments if s.segment_absolute_start is not None
        ]
        combined_absolute_start = min(starts) if starts else None
    else:
        combined_absolute_start = None  # cross-session: not meaningful

    # Original / reconstructed may be ``None`` on the tokenize-only and
    # interval-windowed paths; combine only the present sides (mirroring the
    # latents / behavioral guards above) rather than catting a list of ``None``.
    originals = [s.original for s in segments if s.original is not None]
    recons = [s.reconstructed for s in segments if s.reconstructed is not None]
    combined_original = torch.cat(originals, dim=1) if originals else None
    combined_reconstructed = torch.cat(recons, dim=1) if recons else None

    return CollatedValidationSegment(
        session_id="combined",
        original=combined_original,
        reconstructed=combined_reconstructed,
        pad_and_gap_mask=None,
        metadata={"n_sessions": len(segments), "session_ids": [s.session_id for s in segments]},
        segment_index=0,
        is_partial=False,
        latents=combined_latents,
        behavioral_data_token_res=combined_behavioral_token_res,
        behavioral_data_input_res=combined_behavioral_input_res,
        token_timestamps=combined_timestamps,
        token_start_times=combined_start_times,
        token_end_times=combined_end_times,
        segment_absolute_start=combined_absolute_start,
    )


def _run_interval_windowed_metrics(
    segment: CollatedValidationSegment,
    *,
    pipeline: EvaluationPipeline,
    eval_config: EvaluationConfig,
    data_type: DataType,
    token_bin_size_ms: float | None,
) -> EvaluationReport:
    """Compute metrics on an interval-windowed segment without an assembled trace.


    The interval-windowing path leaves ``segment.original`` and
    ``segment.reconstructed`` as ``None`` because uneven window sizes lead
    to incomplete reconstruction inside each window, so no per-segment
    alignment between input and reconstruction is possible. The metrics
    surviving the allow-list filter (currently only repeats-SNR) consume
    only latent-side intermediates; if any of them ever asks for an
    original/reconstructed intermediate, ``IntermediatesCache`` raises
    rather than substituting placeholder data.

    Mirrors the per-segment slice of ``EvaluationPipeline.run`` so that
    ``EvaluationReport`` has the same shape as on the regular trace path.
    """
    latents_np: np.ndarray | list[np.ndarray] | None = None
    if isinstance(segment.latents, torch.Tensor):
        latents_np = segment.latents.numpy()
    elif isinstance(segment.latents, list):
        latents_np = [
            item.numpy() if isinstance(item, torch.Tensor) else item for item in segment.latents
        ]
    token_timestamps_np: np.ndarray | None = None
    if isinstance(segment.token_timestamps, torch.Tensor):
        token_timestamps_np = segment.token_timestamps.numpy()

    cache = IntermediatesCache(
        original=None,
        reconstructed=None,
        latents=latents_np,
        behavioral_data_token_res=segment.behavioral_data_token_res,
        behavioral_data_input_res=segment.behavioral_data_input_res,
        token_timestamps=token_timestamps_np,
        sampling_rate=_extract_scalar_sampling_rate(segment.metadata.get("sampling_rate")),
        segment_absolute_start=segment.segment_absolute_start,
        token_bin_size_ms=token_bin_size_ms,
    )

    results: dict[str, MetricResult] = {}
    skipped: dict[str, str] = {}
    for display_name, metric in pipeline.metrics:
        if segment.session_id is not None and not metric.should_run_for_session(
            segment.session_id
        ):
            skipped[display_name] = (
                f"Skipped: limit_to_datasets={metric.limit_to_datasets} "
                f"does not match session '{segment.session_id}'"
            )
            continue
        try:
            requests = metric.get_intermediate_requests(data_type)
            intermediates = {req.key: cache.get(req) for req in requests}
            result = metric.compute(intermediates)
        except Exception as exc:
            # Match EvaluationPipeline.run's report-style reporting so the
            # caller's aggregation behaves identically. Cache-side raises
            # for original/reconstructed-needing intermediates land here
            # too, surfacing the metric/intermediate mismatch in the
            # report's skipped_metrics rather than aborting the run.
            skipped[display_name] = f"Computation error: {exc}"
            continue
        result.source_data_type = data_type
        results[display_name] = result
        for extra_name, extra_result in metric.expand_results(result, display_name).items():
            extra_result.source_data_type = data_type
            results[extra_name] = extra_result

    report = create_report(results, config=eval_config.model_dump())
    report.skipped_metrics = skipped
    report.original_data_type = data_type
    report.reconstructed_data_type = data_type
    return report


def _resolve_grouping_field_for_interval_metrics(
    metrics: list[tuple[str, Any]],
) -> str:
    """Return the shared grouping_field used by the active repeats metrics."""
    grouping_fields = {
        metric.params["grouping_field"]
        for _display_name, metric in metrics
        if hasattr(metric, "params") and "grouping_field" in metric.params
    }
    if not grouping_fields:
        return "image_id"
    if len(grouping_fields) != 1:
        raise ValueError(
            "Per-stimulus SNR path requires a single grouping_field across all "
            f"active interval metrics, got {sorted(grouping_fields)!r}."
        )
    return grouping_fields.pop()


def _resolve_session_ids_for_per_stimulus_loader(dataloader: DataLoader) -> list[str]:
    sampler_intervals = getattr(getattr(dataloader, "sampler", None), "sampling_intervals", None)
    if isinstance(sampler_intervals, dict) and sampler_intervals:
        return list(sampler_intervals.keys())

    dataset: Any = dataloader.dataset
    if hasattr(dataset, "recording_dict"):
        return list(dataset.recording_dict.keys())
    if hasattr(dataset, "_recording_objects"):
        return list(dataset._recording_objects.keys())
    if hasattr(dataset, "session_id"):
        return [str(dataset.session_id)]

    raise ValueError(
        "Could not resolve session_ids for PerStimulusSampler from the incoming "
        f"dataloader dataset type {type(dataset).__name__}."
    )


def _extract_max_neurons_from_collate_fn(collate_fn: Callable[..., Any] | None) -> int | None:
    """Best-effort recovery of max_neurons from a dataloader collate_fn."""
    if isinstance(collate_fn, partial):
        return cast(int | None, collate_fn.keywords.get("max_neurons"))
    return None


class _PerStimulusDatasetView(Dataset):
    """Dataset view that resolves PerStimulusSampler indices into DatasetIndex windows."""

    def __init__(self, dataset: Dataset, sampler: PerStimulusSampler) -> None:
        self.dataset = dataset
        self.sampler = sampler

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, index: int) -> tuple[BinnedInputType, int]:
        session_id, effective_start, effective_end, _group_value = self.sampler.geometry_for(index)
        sample = self.dataset[DatasetIndex(session_id, effective_start, effective_end)]
        if not isinstance(sample, (BinnedSpikesInput, RatesInput)):
            raise TypeError(
                "Per-stimulus SNR path expects BinnedSpikesInput or RatesInput samples, got "
                f"{type(sample).__name__}."
            )
        return sample, index


def _make_per_stimulus_collate_fn(max_neurons: int | None):
    """Collate `(sample, sampler_idx)` pairs while preserving sampler order metadata."""
    base_collate = make_neuron_split_collate_fn(max_neurons=max_neurons)

    def _collate(
        batch: list[tuple[BinnedInputType, int]],
    ) -> tuple[BinnedInputType, list[int]]:
        samples = [sample for sample, _index in batch]
        sampler_indices = [index for _sample, index in batch]
        collated = base_collate(samples)
        actual_batch_size = int(get_data_matrix(collated).shape[0])
        if actual_batch_size != len(sampler_indices):
            raise ValueError(
                "Per-stimulus SNR path does not support collate-time neuron-axis sample "
                "expansion: batch size changed from "
                f"{len(sampler_indices)} to {actual_batch_size}. "
                "Cap neurons before batching (e.g. via UnitDropout / eval_config.max_neurons) "
                "instead of relying on neuron_split_collate to fan one stimulus window out "
                "into multiple batch elements."
            )
        return collated, sampler_indices

    return _collate


def _build_per_stimulus_loader(
    dataloader: DataLoader,
    *,
    sampler: PerStimulusSampler,
) -> DataLoader:
    """Clone the incoming loader's batching settings for the per-stimulus SNR path."""
    max_neurons = _extract_max_neurons_from_collate_fn(dataloader.collate_fn)
    loader_kwargs: dict[str, Any] = {
        "batch_size": dataloader.batch_size or 1,
        "sampler": sampler,
        "collate_fn": _make_per_stimulus_collate_fn(max_neurons),
        "num_workers": dataloader.num_workers,
        "pin_memory": dataloader.pin_memory,
        "drop_last": False,
    }
    if dataloader.num_workers > 0:
        loader_kwargs["persistent_workers"] = dataloader.persistent_workers
        if dataloader.prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = dataloader.prefetch_factor
    return DataLoader(_PerStimulusDatasetView(dataloader.dataset, sampler), **loader_kwargs)


def _batch_metadata_value(value: Any, index: int) -> Any:
    """Return one sample's metadata value from a collated batch payload."""
    if isinstance(value, tuple):
        return value[index]
    return value


def _batch_sampling_rate_value(batch: BinnedInputType, index: int) -> float:
    """Return one sample's sampling rate as a scalar float."""
    sampling_rate = batch.sampling_rate
    if isinstance(sampling_rate, torch.Tensor):
        if sampling_rate.ndim == 0:
            return float(sampling_rate.item())
        return float(sampling_rate[index].item())
    return float(sampling_rate)


def _token_time_vector(
    sample_output: dict[str, Any],
    *,
    key: str,
) -> torch.Tensor | None:
    value = sample_output.get(key)
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected '{key}' to be a torch.Tensor, got {type(value).__name__}.")
    if value.ndim == 2:
        return value[0].detach().cpu()
    return value.detach().cpu()


_BATCHED_TENSOR_KEYS_FOR_SAMPLE_SLICE = (
    "reconstruction",
    "reconstructed",
    "latents",
    "token_timestamps",
    "token_start_time",
    "token_end_time",
)


def _select_batch_sample(batched_output: dict[str, Any], sample_index: int) -> dict[str, Any]:
    """Slice batch axis of a tokenizer output dict to a single sample.

    Returns a dict with the same keys; the recognised batched tensors are
    sliced to ``[sample_index:sample_index + 1]`` so each downstream
    consumer can keep its singleton batch-dim assumptions. Non-tensor fields
    are passed through unchanged.
    """
    sliced: dict[str, Any] = {}
    for key, value in batched_output.items():
        if (
            key in _BATCHED_TENSOR_KEYS_FOR_SAMPLE_SLICE
            and isinstance(value, torch.Tensor)
            and value.ndim >= 1
        ):
            sliced[key] = value[sample_index : sample_index + 1]
        else:
            sliced[key] = value
    return sliced


def _expected_tokens_in_fixed_stimulus_window(
    *,
    fixed_window_duration_ms: float | None,
    token_bin_size_ms: float | None,
) -> int | None:
    """Return the exact token count implied by a fixed stimulus window, if integral."""
    if fixed_window_duration_ms is None or token_bin_size_ms is None:
        return None
    if fixed_window_duration_ms <= 0.0 or token_bin_size_ms <= 0.0:
        return None
    ratio = float(fixed_window_duration_ms) / float(token_bin_size_ms)
    nearest = round(ratio)
    if nearest <= 0 or abs(ratio - nearest) > 1e-6:
        return None
    return int(nearest)


def _direct_short_fixed_input_output_is_compatible(
    *,
    batch: BinnedInputType,
    batched_output: Any,
    token_bin_size_ms: float | None,
    fixed_window_duration_ms: float,
) -> tuple[bool, str]:
    """Check whether a raw fixed-input tokenizer output is usable on short stimulus windows."""
    if not isinstance(batched_output, dict):
        return False, (
            "tokenizer output is not a dict with latent/token timing fields "
            f"(got {type(batched_output).__name__})"
        )

    expected_tokens = _expected_tokens_in_fixed_stimulus_window(
        fixed_window_duration_ms=fixed_window_duration_ms,
        token_bin_size_ms=token_bin_size_ms,
    )
    batch_size = get_data_matrix(batch).shape[0]

    for sample_index in range(batch_size):
        sample_output = _select_batch_sample(batched_output, sample_index)
        batch_latents = _extract_latents(sample_output)
        if not isinstance(batch_latents, torch.Tensor):
            return False, "tokenizer did not return latents"
        if batch_latents.ndim < 2:
            return False, (
                "latents are not shaped like [1, T, ...] after sample selection "
                f"(got {tuple(batch_latents.shape)})"
            )
        window_latents = batch_latents[0].detach().cpu()
        window_token_timestamps = _token_time_vector(sample_output, key="token_timestamps")
        if window_token_timestamps is None:
            return False, "tokenizer did not return token_timestamps"
        window_token_starts = _token_time_vector(sample_output, key="token_start_time")
        window_token_ends = _token_time_vector(sample_output, key="token_end_time")
        if window_token_starts is None or window_token_ends is None:
            window_token_starts = window_token_timestamps
            window_token_ends = window_token_timestamps

        assert batch.absolute_start is not None and batch.absolute_end is not None
        sample_absolute_start = float(batch.absolute_start[sample_index].item())
        sample_absolute_end = float(batch.absolute_end[sample_index].item())
        kept_latents, _keep_mask = _drop_tokens_outside_bounds(
            window_latents,
            window_token_starts,
            window_token_ends,
            lower_bound=sample_absolute_start,
            upper_bound=sample_absolute_end,
        )
        if kept_latents.shape[0] == 0:
            return False, "no tokens fall fully inside the stimulus window"
        if expected_tokens is not None and kept_latents.shape[0] != expected_tokens:
            return False, (
                f"expected {expected_tokens} token(s) in a {fixed_window_duration_ms:.1f} ms "
                f"stimulus window, got {kept_latents.shape[0]}"
            )

    return True, "token timing fits inside the fixed stimulus window"


def _iter_collated_validation_segments_with_per_stimulus_sampling(
    dataloader: DataLoader,
    tokenizer_fun: Callable[[BinnedInputType], Any],
    *,
    tokenizer_spec: TokenizerSpec,
    interval_source: str,
    grouping_field: str,
    interval_start_lag_ms: float = 0.0,
    interval_end_lag_ms: float | None = None,
    fixed_window_duration_ms: float | None = None,
    cross_stim_padding_ms: float = 20.0,
    label_extension_ms: float | None = None,
    device: torch.device | None = None,
    max_windows: int | None = None,
    max_eval_seconds: float | None = None,
    _on_batch_done: Callable[[], None] | None = None,
) -> Iterator[CollatedValidationSegment]:
    """Stimulus-aligned SNR iterator backed by PerStimulusSampler.

    Usually wraps ``tokenizer_fun`` with :func:`chunked_tokenizer_fn` so
    fixed-input tokenizers see correctly-sized chunks per batch. The exception
    is a narrow short-stimulus compatibility path: if the tokenizer's nominal
    input window is longer than the fixed stimulus window, we probe one raw
    batch and only keep the raw path when the emitted token timing already
    fits fully inside the stimulus window.
    """
    session_ids = _resolve_session_ids_for_per_stimulus_loader(dataloader)
    per_stim_sampler = PerStimulusSampler(
        dataloader.dataset,
        session_ids=session_ids,
        interval_source=interval_source,
        grouping_field=grouping_field,
        interval_start_lag_ms=interval_start_lag_ms,
        interval_end_lag_ms=interval_end_lag_ms,
        fixed_window_duration_ms=fixed_window_duration_ms,
        cross_stim_padding_ms=cross_stim_padding_ms,
        label_extension_ms=label_extension_ms,
    )
    per_stim_loader = _build_per_stimulus_loader(dataloader, sampler=per_stim_sampler)

    run_raw_short_fixed_input = False
    nominal_window_is_longer_than_stimulus = (
        fixed_window_duration_ms is not None
        and requires_fixed_input_window(tokenizer_spec)
        and float(tokenizer_spec.effective_window_length_seconds)
        > (float(fixed_window_duration_ms) / 1000.0)
    )
    if nominal_window_is_longer_than_stimulus:
        assert fixed_window_duration_ms is not None
        per_stim_loader_iter = iter(per_stim_loader)
        try:
            first_batch, first_sampler_indices = next(per_stim_loader_iter)
        except StopIteration:
            return
        if device is not None:
            first_batch = move_data_to_device(first_batch, device)
        try:
            with torch.inference_mode():
                first_output = tokenizer_fun(first_batch)
        except Exception as exc:
            logger.info(
                "Stimulus-windowed evaluation skipped: raw short-window probe for "
                "fixed-input tokenizer KIND=%r failed for nominal window %.1f ms "
                "against fixed_window_duration_ms=%.1f: %s. Returning no segments.",
                tokenizer_spec.KIND,
                float(tokenizer_spec.effective_window_length_seconds) * 1000.0,
                float(fixed_window_duration_ms),
                exc,
            )
            return
        compatible, reason = _direct_short_fixed_input_output_is_compatible(
            batch=first_batch,
            batched_output=first_output,
            token_bin_size_ms=getattr(tokenizer_spec, "token_bin_size_ms", None),
            fixed_window_duration_ms=float(fixed_window_duration_ms),
        )
        if not compatible:
            logger.info(
                "Stimulus-windowed evaluation skipped: fixed-input tokenizer KIND=%r "
                "with nominal window %.1f ms is not compatible with fixed "
                "stimulus windows of %.1f ms: %s. Returning no segments.",
                tokenizer_spec.KIND,
                float(tokenizer_spec.effective_window_length_seconds) * 1000.0,
                float(fixed_window_duration_ms),
                reason,
            )
            return
        logger.info(
            "Stimulus-windowed evaluation: fixed-input tokenizer KIND=%r with "
            "nominal window %.1f ms is directly compatible with fixed stimulus "
            "windows of %.1f ms (%s). Running the raw tokenizer on this path.",
            tokenizer_spec.KIND,
            float(tokenizer_spec.effective_window_length_seconds) * 1000.0,
            float(fixed_window_duration_ms),
            reason,
        )
        run_raw_short_fixed_input = True
        batch_iter = chain(
            ((first_batch, first_sampler_indices, first_output),),
            ((batch, sampler_indices, None) for batch, sampler_indices in per_stim_loader_iter),
        )
    else:
        # Default path: fixed-input tokenizers get pre-chunking; variable
        # tokenizers get a (near) no-op passthrough.
        tokenizer_fun = chunked_tokenizer_fn(tokenizer_fun, spec=tokenizer_spec)
        batch_iter = ((batch, sampler_indices, None) for batch, sampler_indices in per_stim_loader)

    current_session_id: str | None = None
    window_buffer: list[dict[str, Any]] = []
    segment_index = 0
    next_window_group_id_token_res = 0
    session_accumulated_seconds = 0.0
    session_capped = False

    def flush(is_partial: bool) -> Iterator[CollatedValidationSegment]:
        nonlocal window_buffer, segment_index
        if current_session_id is None or not window_buffer:
            return iter(())
        segment = _flush_segment(
            window_buffer,
            drop_overlapping_windows=False,
            skip_gap_check=True,
            is_interval_windowed=True,
            assemble_input_and_reconstruction=False,
        )
        window_buffer = []
        segment = replace(segment, segment_index=segment_index, is_partial=is_partial)
        segment_index += 1
        return iter((segment,))

    def buffer_processed_window(window_dict: dict[str, Any]) -> None:
        nonlocal next_window_group_id_token_res
        n_tokens = int(window_dict["latents"].shape[0])
        window_dict["behavioral_data_token_res"]["window_group_id_token_res"] = np.full(
            n_tokens,
            next_window_group_id_token_res,
            dtype=np.int32,
        )
        next_window_group_id_token_res += 1
        window_buffer.append(window_dict)

    for batch, sampler_indices, prefetched_output in batch_iter:
        if device is not None and prefetched_output is None:
            batch = move_data_to_device(batch, device)
        batched_output = (
            prefetched_output if prefetched_output is not None else tokenizer_fun(batch)
        )
        if not isinstance(batched_output, dict):
            raise TypeError(
                "Per-stimulus SNR path expects tokenizer output with latents and token "
                f"timestamps, got {type(batched_output).__name__}."
            )
        batch_size = get_data_matrix(batch).shape[0]
        batch_time_bins = int(get_data_matrix(batch).shape[1])
        for sample_index in range(batch_size):
            sample_output = _select_batch_sample(batched_output, sample_index)

            global_idx = sampler_indices[sample_index]
            session_id, effective_start, effective_end, group_value = (
                per_stim_sampler.geometry_for(global_idx)
            )
            stim_start, label_end = per_stim_sampler.label_bounds_for(global_idx)

            if current_session_id is None:
                current_session_id = session_id
            elif session_id != current_session_id:
                yield from flush(is_partial=False)
                current_session_id = session_id
                next_window_group_id_token_res = 0
                session_accumulated_seconds = 0.0
                session_capped = False

            if session_capped:
                continue

            batch_latents = _extract_latents(sample_output)
            if not isinstance(batch_latents, torch.Tensor):
                raise ValueError("Per-stimulus SNR path requires tokenizer latents per window.")
            if batch_latents.ndim < 2:
                raise ValueError(
                    "Expected batched latents with shape [1, T, ...], "
                    f"got {tuple(batch_latents.shape)}."
                )
            window_latents = batch_latents[0].detach().cpu()

            window_token_timestamps = _token_time_vector(sample_output, key="token_timestamps")
            if window_token_timestamps is None:
                raise ValueError("token_timestamps are required when latents are present.")
            window_token_starts = _token_time_vector(sample_output, key="token_start_time")
            window_token_ends = _token_time_vector(sample_output, key="token_end_time")
            if window_token_starts is None or window_token_ends is None:
                window_token_starts = window_token_timestamps
                window_token_ends = window_token_timestamps

            sampling_rate = _batch_sampling_rate_value(batch, sample_index)
            sample_absolute_start = float(batch.absolute_start[sample_index].item())
            sample_absolute_end = float(batch.absolute_end[sample_index].item())
            true_n_bins = int(round((sample_absolute_end - sample_absolute_start) * sampling_rate))

            # Fixed-duration stimulus windows are expected to be batch-collated
            # without time-axis padding. Variable-duration windows may be padded
            # to the batch max; in that case trim token-res outputs per sample
            # using the same token-padding helper as the main eval path.
            if fixed_window_duration_ms is not None:
                assert true_n_bins == batch_time_bins, (
                    "Per-stimulus SNR path with fixed_window_duration_ms expects no "
                    f"time padding after collation, but sample_index={sample_index} has "
                    f"true_n_bins={true_n_bins} and batch_time_bins={batch_time_bins}. "
                    "Either the loader is still padding fixed-length windows, or the "
                    "sample's absolute_start/absolute_end metadata no longer matches the "
                    "batched time axis."
                )
            lower_bound = sample_absolute_start if run_raw_short_fixed_input else None
            window_latents, keep_mask = _drop_tokens_outside_bounds(
                window_latents,
                window_token_starts,
                window_token_ends,
                lower_bound=lower_bound,
                upper_bound=sample_absolute_end,
            )
            window_token_starts = window_token_starts[keep_mask]
            window_token_ends = window_token_ends[keep_mask]
            window_token_timestamps = window_token_timestamps[keep_mask]

            labels = np.full(window_latents.shape[0], np.nan, dtype=np.float64)
            token_centers_np = window_token_timestamps.numpy().astype(np.float64)
            inside = (token_centers_np >= stim_start - 1e-9) & (token_centers_np < label_end)
            labels[inside] = float(group_value)

            original_window = (
                get_data_matrix(batch)[sample_index, :true_n_bins, :].T.detach().cpu()
            )

            buffer_processed_window(
                {
                    "session_id": session_id,
                    "original": original_window,
                    "absolute_start": float(effective_start),
                    "absolute_end": float(effective_end),
                    "latents": window_latents,
                    "token_timestamps": window_token_timestamps,
                    "token_start_times": window_token_starts,
                    "token_end_times": window_token_ends,
                    "behavioral_data_token_res": {grouping_field: labels},
                    "metadata": {
                        "sampling_rate": sampling_rate,
                        "subject_id": _batch_metadata_value(batch.subject_id, sample_index),
                        "species": _batch_metadata_value(batch.species, sample_index),
                    },
                    "has_overlap": False,
                }
            )

            if max_windows is not None and len(window_buffer) >= max_windows:
                yield from flush(is_partial=True)
                next_window_group_id_token_res = 0

            session_accumulated_seconds += effective_end - effective_start
            if max_eval_seconds is not None and session_accumulated_seconds >= max_eval_seconds:
                logger.info(
                    "[session=%s] max_eval_seconds cap reached at %.1fs (limit=%.1fs); "
                    "skipping remaining stimulus windows.",
                    current_session_id,
                    session_accumulated_seconds,
                    max_eval_seconds,
                )
                yield from flush(is_partial=True)
                session_capped = True
                next_window_group_id_token_res = 0

        if _on_batch_done is not None:
            _on_batch_done()

    yield from flush(is_partial=False)


def run_validation_evaluation_per_session(
    dataloader: DataLoader,
    tokenizer_fun: Callable[[BinnedInputType], Any],
    *,
    data_type: DataType,
    bin_size_seconds: float | None,
    token_bin_size_ms: float | None = None,
    window_length_seconds: float | None = None,
    trim_token_overlap: bool = False,
    drop_overlapping_windows: bool = True,
    eval_config: EvaluationConfig | None = None,
    device: torch.device | None = None,
    max_windows: int | None = None,
    max_bytes: int | None = None,
    concatenate_sessions: bool = False,
    progress: bool = True,
    stash_originals_dir: str | None = None,
    tokenizer_spec: TokenizerSpec | None = None,
    reconstructed_data_type: DataType | None = None,
) -> dict[str, list[tuple[CollatedValidationSegment, dict[str, Any]]]]:
    """Run evaluation per session over collated validation segments.

    ``data_type`` labels the original data (the tokenizer's input type). The
    reconstruction is labelled with the type the tokenizer outputs declare
    (``reconstruction_data_type``, see ``ntt.evaluation.utils.reconstruction_type``);
    ``reconstructed_data_type`` is the fallback for outputs that declare none, and
    ``data_type`` the fallback after that.

    Segment size is controlled by `max_windows` / `max_bytes`:
    - If provided as function arguments, those values are used.
    - Otherwise, values are taken from `eval_config.max_windows` / `eval_config.max_bytes`.
    - If both are None, each contiguous `session_id` block in the dataloader is
      concatenated into a single segment.

    After metrics are computed for each segment, heavy tensor fields
    (including ``original``) are released.  When ``stash_originals_dir``
    is set, the original numpy array is saved to disk *before* releasing
    the tensor; the file path is stored in the report metadata under
    ``"_original_stash_path"`` so callers can load it back.
    """
    eval_config = eval_config or EvaluationConfig()
    if max_windows is None:
        max_windows = eval_config.max_windows
    if max_bytes is None:
        max_bytes = eval_config.max_bytes
    interval_source = eval_config.interval_source
    interval_start_lag_ms = eval_config.interval_start_lag_ms
    interval_end_lag_ms = eval_config.interval_end_lag_ms
    fixed_window_duration_ms = eval_config.fixed_window_duration_ms
    cross_stim_padding_ms = eval_config.cross_stim_padding_ms
    label_extension_ms = eval_config.label_extension_ms
    max_eval_seconds = eval_config.max_eval_seconds
    use_per_stimulus_sampler = interval_source is not None

    # Built-in skip for the stimulus-windowed (SNR / repeats-SNR) path when a
    # token itself is longer than the entire fixed stimulus window. Without
    # this guard the downstream metric cannot get a meaningful per-stimulus
    # token sequence at all. ``token_bin_size_ms is None`` (dynamic-bin
    # tokenizers) bypasses the guard.
    if (
        use_per_stimulus_sampler
        and fixed_window_duration_ms is not None
        and token_bin_size_ms is not None
        and token_bin_size_ms > fixed_window_duration_ms
    ):
        logger.info(
            "Stimulus-windowed evaluation skipped: "
            "interval_source=%r, token_bin_size_ms=%s exceeds fixed_window_duration_ms=%s. "
            "Returning empty session-reports dict.",
            interval_source,
            token_bin_size_ms,
            fixed_window_duration_ms,
        )
        return {}

    pipeline = EvaluationPipeline(config=eval_config)
    grouping_field: str | None = None
    if use_per_stimulus_sampler:
        # Stimulus-windowed evaluation produces uneven windows whose
        # reconstructions don't fully cover their inputs, so neither a
        # per-window nor a per-segment alignment between original and
        # reconstruction can be assumed. We filter the metric list down to
        # those that work without an assembled input/reconstruction trace
        # (currently {repeats_snr}). The remaining metrics are run on the
        # separate non-stimulus-windowed eval pass.
        allowed_metric_names = get_interval_windowing_allowed_metric_names()
        kept: list[tuple[str, Any]] = []
        dropped: list[str] = []
        for display_name, metric in pipeline.metrics:
            if metric.name in allowed_metric_names:
                kept.append((display_name, metric))
            else:
                dropped.append(display_name)
        if dropped:
            logger.info(
                "Stimulus-windowed evaluation (interval_source=%r): skipping %d "
                "metric(s) that depend on a continuous original/recon "
                "trace: %s. Allowed on this path: %s.",
                interval_source,
                len(dropped),
                sorted(dropped),
                sorted(allowed_metric_names),
            )
        pipeline.metrics = kept
        grouping_field = _resolve_grouping_field_for_interval_metrics(pipeline.metrics)
    downstream_metrics = [
        metric
        for _name, metric in pipeline.metrics
        if metric.category == MetricCategory.DOWNSTREAM_DECODING
    ]
    has_downstream = bool(downstream_metrics)
    decoding_target_fields: set[str | frozenset[str]] | None = None
    if has_downstream:
        # max_windows / max_bytes split a session into multiple segments, which
        # corrupts decoding (each segment gets its own train/test split, then
        # aggregated — invalid). max_eval_seconds only truncates a session into
        # one shorter segment; we instead suppress it per-session for sessions
        # that actually carry the behavioral target fields a decoder needs.
        offending = {
            "max_windows": max_windows,
            "max_bytes": max_bytes,
        }
        offending = {k: v for k, v in offending.items() if v is not None}
        if offending:
            raise ValueError(
                f"{sorted(offending)} are not supported when downstream behavioral "
                "decoding metrics are enabled. Set them to None so each session is "
                "evaluated as a single collated segment. "
                f"Got: {offending}."
            )
        from ntt.evaluation.metrics.downstream_prediction import required_behavioral_fields

        # Bypass only for sessions a configured decoder can actually consume:
        # carrying a target field alone (e.g. cursor_pos without the
        # movement_phase its metric filters on) leaves the session capped.
        decoding_target_fields = {
            required_behavioral_fields(metric)
            for metric in downstream_metrics
            if getattr(metric, "target_field", None)
        }
        if decoding_target_fields and max_eval_seconds is not None:
            logger.info(
                "Per-session max_eval_seconds=%.1fs will be bypassed for sessions "
                "carrying all fields of any of these decoding metrics: %s",
                max_eval_seconds,
                sorted(sorted(fields) for fields in decoding_target_fields),
            )
    reports: dict[str, list[tuple[CollatedValidationSegment, dict[str, Any]]]] = {}

    # ── Progress bar ─────────────────────────────────────────────────
    try:
        n_batches = None if use_per_stimulus_sampler else len(dataloader)
    except TypeError:
        n_batches = None

    bar = tqdm(
        total=n_batches,
        desc="Tokenizing",
        disable=not progress,
        dynamic_ncols=True,
        leave=True,
    )

    # ── Phase 1: Tokenize & collate ──────────────────────────────────
    def _on_batch_done() -> None:
        bar.update(1)

    # Records the reconstruction type the tokenizer outputs declare; segments are
    # yielded only after their batches were tokenized, so it is set before use.
    recorder = ReconstructionTypeRecorder(tokenizer_fun)

    def _reconstruction_type() -> DataType:
        return recorder.declared or reconstructed_data_type or data_type

    # Task 5 SNR routing call-site: when interval_source is set, use the
    # PerStimulusSampler-backed loader. The per-stim iterator wraps
    # ``tokenizer_fun`` with ``chunked_tokenizer_fn`` internally; it needs
    # the spec to do so.
    if use_per_stimulus_sampler:
        if tokenizer_spec is None:
            raise ValueError(
                "interval_source requires tokenizer_spec so the SNR path can wrap "
                "the tokenizer with chunked_tokenizer_fn(spec=...)."
            )
        assert grouping_field is not None
        segment_iter = _iter_collated_validation_segments_with_per_stimulus_sampling(
            dataloader,
            recorder,
            tokenizer_spec=tokenizer_spec,
            interval_source=interval_source,
            grouping_field=grouping_field,
            interval_start_lag_ms=interval_start_lag_ms,
            interval_end_lag_ms=interval_end_lag_ms,
            fixed_window_duration_ms=fixed_window_duration_ms,
            cross_stim_padding_ms=cross_stim_padding_ms,
            label_extension_ms=label_extension_ms,
            device=device,
            max_windows=max_windows,
            max_eval_seconds=max_eval_seconds,
            _on_batch_done=_on_batch_done,
        )
    else:
        segment_iter = iter_collated_validation_segments(
            dataloader,
            recorder,
            tokenizer_spec=tokenizer_spec,
            token_bin_size_ms=token_bin_size_ms,
            window_length_seconds=window_length_seconds,
            trim_token_overlap=trim_token_overlap,
            drop_overlapping_windows=drop_overlapping_windows,
            device=device,
            max_windows=max_windows,
            max_bytes=max_bytes,
            max_eval_seconds=max_eval_seconds,
            decoding_target_fields=decoding_target_fields,
            _on_batch_done=_on_batch_done,
        )

    if concatenate_sessions:
        # Must materialise for cross-session concatenation.
        all_segments = list(segment_iter)
        if len(all_segments) > 1:
            all_segments = [_concatenate_segments(all_segments)]
        segment_iter = iter(all_segments)

    # ── Phase 2: Compute metrics per segment ─────────────────────────
    # (progress total unknown when streaming; update count only)
    bar.reset(total=None)
    bar.set_description("Computing metrics")

    _log = logging.getLogger(__name__)
    _SPIKE_COUNT_RATIO_WARN = 5.0

    for segment in segment_iter:
        t_segment_start = time.perf_counter()
        session_short = segment.session_id.rsplit("/", 1)[-1]
        bar.set_description(f"Metrics · {session_short}")

        if use_per_stimulus_sampler:
            # Stimulus-windowed segments have ``original``/``reconstructed``
            # left as None by ``_flush_segment``. Run only the allow-listed
            # latent-side metrics directly; the EvaluationPipeline.run path
            # below assumes both sides exist.
            report = _run_interval_windowed_metrics(
                segment,
                pipeline=pipeline,
                eval_config=eval_config,
                data_type=data_type,
                token_bin_size_ms=token_bin_size_ms,
            )
            t_segment_elapsed = time.perf_counter() - t_segment_start
            _log.debug(
                "Segment %s done in %.1fs (stimulus-windowed, latents-only)",
                session_short,
                t_segment_elapsed,
            )
            metadata = dict(segment.metadata)
            reports.setdefault(segment.session_id, []).append(
                (segment, {"report": report, "metadata": metadata})
            )
            # Match the trace path's tensor cleanup so memory does not pile
            # up across segments. ``original`` is already None on this path;
            # strip_original is a no-op but keeps the contract consistent.
            segment.strip_original()
            segment.reconstructed = None  # type: ignore[assignment]
            segment.latents = None
            segment.pad_and_gap_mask = None
            segment.token_start_times = None
            segment.token_end_times = None
            segment.token_timestamps = None
            segment.behavioral_data_token_res = None
            segment.behavioral_data_input_res = None
            bar.update(1)
            continue

        original = _apply_pad_and_gap_mask(segment.original, segment.pad_and_gap_mask)
        reconstructed = _apply_pad_and_gap_mask(segment.reconstructed, segment.pad_and_gap_mask)
        assert original is not None, "Collated segments carry the original data on this path."

        # ── Neuron subsampling ──────────────────────────────────────────
        # Cap the number of neurons before metric computation.  On the
        # full-neuron eval paths the DataModule's EvalUnitSelection already
        # selected these units at load time with the same function, so this
        # block is a no-op there. It remains for the padded path (e.g. a
        # tokenizer whose n_neurons exceeds max_neurons).
        _max_neurons = eval_config.max_neurons
        _n_units = original.shape[0]
        if _max_neurons is not None and _n_units > _max_neurons:
            from ntt.core.data.components.transforms import select_eval_units

            _sel = select_eval_units(
                _n_units, _max_neurons, eval_config.neuron_subsample_seed, segment.session_id
            )
            _sel_t = torch.tensor(_sel, dtype=torch.long)
            original = original[_sel_t]
            if reconstructed is not None:
                reconstructed = reconstructed[_sel_t]
            _log.debug(
                "Neuron cap: %s subsampled %d -> %d neurons for metrics",
                session_short,
                _n_units,
                _max_neurons,
            )

        # Spike count sanity check: if the reconstruction has vastly more
        # non-zero bins than the original, the tokenizer is likely producing
        # dense rates rather than sparse spikes.  This would cause downstream
        # algorithms (correlograms, spike matching) to become catastrophically
        # slow or produce meaningless results.
        if _reconstruction_type() == DataType.SPIKES and reconstructed is not None:
            orig_np = original.numpy() if hasattr(original, "numpy") else np.asarray(original)
            recon_np = (
                reconstructed.numpy()
                if hasattr(reconstructed, "numpy")
                else np.asarray(reconstructed)
            )
            orig_nnz = int(np.count_nonzero(orig_np))
            recon_nnz = int(np.count_nonzero(recon_np))
            if orig_nnz > 0:
                ratio = recon_nnz / orig_nnz
                if ratio > _SPIKE_COUNT_RATIO_WARN:
                    _log.warning(
                        "Reconstruction spike density for %s is %.1fx the "
                        "original (%s vs %s non-zero bins). This usually "
                        "means the tokenizer is outputting dense rates "
                        "instead of sparse spikes. Metrics requiring "
                        "SPIKE_TIMES conversion may be skipped.",
                        session_short,
                        ratio,
                        f"{recon_nnz:,}",
                        f"{orig_nnz:,}",
                    )

        duration = None
        if bin_size_seconds is not None:
            # Shape is (units, time), time is second dimension
            duration = original.shape[1] * bin_size_seconds

        # Data is already in (units, time) format for EvalData
        original_data = EvalData.from_tensor(
            tensor=original,
            data_type=data_type,
            bin_size_seconds=bin_size_seconds,
            duration=duration,
        )
        # Tokenize-only path: when ``reconstructed`` is None the driver
        # passes ``None`` to ``pipeline.run``; reconstruction-needing
        # metrics are filtered by the cache's ``_require_reconstructed``.
        reconstructed_data: EvalData | None = None
        if reconstructed is not None:
            reconstructed_data = EvalData.from_tensor(
                tensor=reconstructed,
                data_type=_reconstruction_type(),
                bin_size_seconds=bin_size_seconds,
                duration=duration,
            )

        # Convert latents to numpy if present (tensor or list[Tensor]).
        latents_np = None
        if isinstance(segment.latents, torch.Tensor):
            latents_np = segment.latents.numpy()
        elif isinstance(segment.latents, list):
            latents_np = [
                item.numpy() if isinstance(item, torch.Tensor) else item
                for item in segment.latents
            ]

        # Extract token timestamps as numpy
        token_start_times_np = None
        token_end_times_np = None
        token_timestamps_np = None
        if segment.token_start_times is not None:
            assert isinstance(segment.token_start_times, torch.Tensor)
            assert isinstance(segment.token_end_times, torch.Tensor)
            token_start_times_np = segment.token_start_times.numpy()
            token_end_times_np = segment.token_end_times.numpy()
        if segment.token_timestamps is not None:
            assert isinstance(segment.token_timestamps, torch.Tensor)
            token_timestamps_np = segment.token_timestamps.numpy()

        _seg_n_units = original.shape[0]
        _seg_n_time = original.shape[1]
        _seg_duration_s = duration if duration is not None else 0.0
        _log.debug(
            "Segment %s: %d units x %d bins (%.1fs)%s",
            session_short,
            _seg_n_units,
            _seg_n_time,
            _seg_duration_s,
            " [partial]" if segment.is_partial else "",
        )

        report = pipeline.run(
            original_data,
            reconstructed_data,
            compute_baseline=eval_config.compute_baseline,
            latents=latents_np,
            behavioral_data_token_res=segment.behavioral_data_token_res,
            behavioral_data_input_res=segment.behavioral_data_input_res,
            token_timestamps=token_timestamps_np,
            token_start_times=token_start_times_np,
            token_end_times=token_end_times_np,
            sampling_rate=_extract_scalar_sampling_rate(segment.metadata["sampling_rate"]),
            segment_absolute_start=segment.segment_absolute_start,
            token_bin_size_ms=token_bin_size_ms,
            session_id=segment.session_id,
        )

        t_segment_elapsed = time.perf_counter() - t_segment_start
        _log.debug(
            "Segment %s done in %.1fs (%d units x %.1fs data)",
            session_short,
            t_segment_elapsed,
            _seg_n_units,
            _seg_duration_s,
        )

        metadata = dict(segment.metadata)
        reports.setdefault(segment.session_id, []).append(
            (segment, {"report": report, "metadata": metadata})
        )

        # Release heavy tensor fields now that metrics are computed.
        # When stash_originals_dir is set, save original to disk first
        # so the Poisson comparison can reload it later.
        # NOTE: stash the post-subsampled local ``original`` (the same view
        # metrics just consumed), NOT ``segment.original`` — when
        # ``eval_config.max_neurons`` caps neurons above, downstream Poisson
        # realisations must see the same neuron view to produce per-unit
        # arrays of matching shape for cross-realisation averaging.
        if stash_originals_dir is not None and segment.original is not None:
            _stash_name = f"{segment.session_id.replace('/', '_')}_{segment.segment_index}.npy"
            _stash_path = f"{stash_originals_dir}/{_stash_name}"
            _orig_np = original.cpu().numpy() if hasattr(original, "cpu") else np.asarray(original)
            np.save(_stash_path, _orig_np)
            metadata["_original_stash_path"] = _stash_path
            del _orig_np
        segment.strip_original()
        segment.reconstructed = None  # type: ignore[assignment]
        segment.latents = None
        segment.pad_and_gap_mask = None
        segment.token_start_times = None
        segment.token_end_times = None
        segment.token_timestamps = None

        bar.update(1)

    bar.set_description("Eval complete")
    bar.close()

    return reports


def _scalar_metric_value(result: MetricResult) -> float | None:
    """Reduce array-valued metrics to a scalar for aggregation; skip empty/NaN."""
    value: Any = result.raw_value
    if hasattr(value, "__len__"):
        if len(value) == 0:
            return None
        value = np.nanmean(value)
    if np.isscalar(value) and not np.isnan(value):
        return float(cast(float, value))
    return None


def _aggregate_metric_stack(
    template: MetricResult,
    values: list[float],
    weights: list[float],
    metadata_items: list[dict[str, Any]],
    strategy: AggregationStrategy,
    *,
    count_key: str,
    count: int,
    metadata_fields_that_must_match: set[str] | None = None,
) -> MetricResult:
    """Combine parallel lists of metric values (and matching weights/metadata)."""
    from ntt.evaluation.core.types import MetricResult

    values_arr = np.array(values, dtype=float)
    weights_arr = np.array(weights, dtype=float)
    if strategy == AggregationStrategy.MEAN:
        agg_value = float(np.nanmean(values_arr))
    elif strategy == AggregationStrategy.MEDIAN:
        agg_value = float(np.nanmedian(values_arr))
    elif strategy == AggregationStrategy.WEIGHTED_MEAN:
        if len(weights_arr) == len(values_arr):
            valid_mask = ~np.isnan(values_arr)
            if valid_mask.any():
                agg_value = float(
                    np.average(values_arr[valid_mask], weights=weights_arr[valid_mask])
                )
            else:
                agg_value = float("nan")
        else:
            agg_value = float(np.nanmean(values_arr))
    else:
        agg_value = float(np.nanmean(values_arr))

    aggregated_metadata: dict[str, Any] = {
        count_key: count,
        "std": float(np.nanstd(values_arr)),
        "aggregation_strategy": strategy.value,
    }

    fields_that_must_match = {
        "chunk_window_ms",
        "primary_chunk_window_ms",
        "token_bin_size_ms",
        "behavioral_bin_size_ms",
        "eval_prediction_resolution_ms",
        "tokens_per_chunk",
        *(metadata_fields_that_must_match or set()),
    }
    if metadata_items:
        all_meta_keys = set().union(*(m.keys() for m in metadata_items))
        for key in sorted(all_meta_keys):
            if key == "best_chunk_window_ms":
                continue
            key_values = [m[key] for m in metadata_items if key in m]
            if not key_values:
                continue

            # Some sessions use 1000 ms chunks because 200 ms fails in those sessions.
            # Other sessions can use 200 ms chunks, so the selected chunk size differs
            # between sessions.
            if key in fields_that_must_match:
                if len(key_values) != len(metadata_items) or any(
                    not np.array_equal(value, key_values[0]) for value in key_values[1:]
                ):
                    raise ValueError(
                        f"Cannot aggregate {template.name}: parameter {key} differs "
                        f"or is missing across inputs: {key_values!r}"
                    )
                aggregated_metadata[key] = key_values[0]
                continue

            # Scalar numeric path.
            scalar_ok = all(
                not isinstance(v, bool) and isinstance(v, (int, float, np.number))
                for v in key_values
            )
            if scalar_ok:
                numeric_arr = np.array([float(v) for v in key_values], dtype=float)
                if strategy == AggregationStrategy.MEAN:
                    aggregated_metadata[key] = float(np.nanmean(numeric_arr))
                elif strategy == AggregationStrategy.MEDIAN:
                    aggregated_metadata[key] = float(np.nanmedian(numeric_arr))
                elif strategy == AggregationStrategy.WEIGHTED_MEAN:
                    if len(weights_arr) == len(numeric_arr):
                        valid_mask = ~np.isnan(numeric_arr)
                        if valid_mask.any():
                            aggregated_metadata[key] = float(
                                np.average(
                                    numeric_arr[valid_mask],
                                    weights=weights_arr[valid_mask],
                                )
                            )
                        else:
                            aggregated_metadata[key] = float("nan")
                    else:
                        aggregated_metadata[key] = float(np.nanmean(numeric_arr))
                else:
                    aggregated_metadata[key] = float(np.nanmean(numeric_arr))
                continue

            # Ndarray path: per-unit summary statistics like ``realisation_std``
            # arrive as np.ndarray (one value per unit) for PER_UNIT metrics.
            # Without this branch they failed the scalar check above and got
            # silently dropped, hiding cross-realisation variance for per-unit
            # metrics (for example realisation_std on spike_count_correlation,
            # binned_rate_difference and temporal_precision). Reduce
            # element-wise across segments/sessions, then collapse to a scalar
            # so the value survives JSON serialization and downstream consumers
            # that expect ``metadata[k]`` to be numeric.
            ndarray_ok = (
                all(isinstance(v, np.ndarray) for v in key_values)
                and len({v.shape for v in key_values}) == 1
                and key_values[0].ndim >= 1
            )
            if ndarray_ok:
                stacked = np.stack(key_values, axis=0)  # (n_items, *unit_shape)
                if strategy == AggregationStrategy.MEDIAN:
                    per_element = np.nanmedian(stacked, axis=0)
                else:
                    # MEAN, WEIGHTED_MEAN, and unknown strategies: element-wise
                    # nanmean. Weighted-mean over per-unit arrays would need
                    # masked-weight bookkeeping that's not earned by current
                    # callers; keep the simpler reduction.
                    per_element = np.nanmean(stacked, axis=0)
                aggregated_metadata[key] = float(np.nanmean(per_element))

    aggregated_metadata["parameter_metadata_checked"] = True
    return MetricResult(
        name=template.name,
        category=template.category,
        direction=template.direction,
        level=template.level,
        raw_value=agg_value,
        normalized_value=None,
        metadata=aggregated_metadata,
        value_range=template.value_range,
        source_data_type=template.source_data_type,
    )


def aggregate_segment_reports(
    segment_list: list[tuple[CollatedValidationSegment, dict[str, Any]]],
    strategy: AggregationStrategy = AggregationStrategy.MEAN,
) -> EvaluationReport:
    """Aggregate metrics across validation segments for a single session.

    Used when a session is evaluated in multiple windows; combines them with the
    same strategy (and segment duration weights for ``WEIGHTED_MEAN``) before
    cross-session aggregation.
    """
    all_results: dict[str, list[float]] = {}
    all_metadata: dict[str, list[dict[str, Any]]] = {}
    all_weights: dict[str, list[float]] = {}
    first_report: EvaluationReport | None = None
    # Merge skipped_metrics across segments. If a metric is skipped on multiple
    # segments with different reasons we keep the first reason and append
    # ``(+N more)`` rather than dropping any context, so real upstream errors
    # (e.g. SNR's per-window mismatch check) stay visible.
    merged_skipped: dict[str, str] = {}
    skipped_seen_count: dict[str, int] = {}

    for segment, data in segment_list:
        report: EvaluationReport = data["report"]
        if first_report is None:
            first_report = report
        shape = (
            segment.original_shape
            if segment.original_shape is not None
            else (tuple(segment.original.shape) if segment.original is not None else None)
        )
        if shape is None:
            raise RuntimeError(
                "segment has neither original tensor nor original_shape; "
                "call strip_original() before discarding the tensor"
            )
        weight = float(shape[1])
        for name, result in report.results.items():
            sv = _scalar_metric_value(result)
            if sv is not None:
                all_results.setdefault(name, []).append(sv)
                all_metadata.setdefault(name, []).append(dict(result.metadata))
                all_weights.setdefault(name, []).append(weight)
        for name, reason in report.skipped_metrics.items():
            if name in merged_skipped:
                skipped_seen_count[name] += 1
            else:
                merged_skipped[name] = reason
                skipped_seen_count[name] = 1

    if first_report is None:
        return EvaluationReport(results={})

    from ntt.evaluation.aggregation.summary import aggregate_by_category

    for _segment, data in segment_list:
        report = data["report"]
        if report.config != first_report.config:
            raise ValueError("Segment report configs differ; cannot aggregate safely.")
        if report.original_data_type != first_report.original_data_type:
            raise ValueError(
                "Segment report original_data_type values differ; cannot aggregate safely."
            )
        if report.reconstructed_data_type != first_report.reconstructed_data_type:
            raise ValueError(
                "Segment report reconstructed_data_type values differ; cannot aggregate safely."
            )

    # Annotate skip reasons that appeared in >1 segment.
    final_skipped = {
        name: (
            f"{reason} (skipped on {skipped_seen_count[name]} segments)"
            if skipped_seen_count[name] > 1
            else reason
        )
        for name, reason in merged_skipped.items()
    }

    aggregated_results: dict[str, MetricResult] = {}
    for name, values in all_results.items():
        if not values:
            continue
        template = first_report.results.get(name)
        if template is None:
            continue
        meta_items = all_metadata.get(name, [])
        w_slice = all_weights.get(name, [])
        evaluation_config = first_report.config or {}
        configured_metrics = evaluation_config.get("metrics") or {}
        metric_parameters = configured_metrics.get(name, {})
        metadata_fields_that_must_match = set()
        # The saved `n_splits` value counts the cross-validation folds that are valid. It
        # does not count the number of folds set in the configuration.
        for parameter_name, configured_value in metric_parameters.items():
            if configured_value is not None and parameter_name != "n_splits":
                metadata_fields_that_must_match.add(parameter_name)

        aggregated_results[name] = _aggregate_metric_stack(
            template,
            values,
            w_slice,
            meta_items,
            strategy,
            count_key="n_segments",
            count=len(values),
            metadata_fields_that_must_match=metadata_fields_that_must_match,
        )

    out = EvaluationReport(
        results=aggregated_results,
        category_scores=(
            aggregate_by_category(aggregated_results) if aggregated_results else None
        ),
        config=first_report.config,
        original_data_type=first_report.original_data_type,
        reconstructed_data_type=first_report.reconstructed_data_type,
    )
    out.skipped_metrics = final_skipped
    return out


def _aggregate_evaluation_reports(
    reports: list[EvaluationReport],
    session_weights: list[float],
    strategy: AggregationStrategy,
) -> EvaluationReport:
    """Aggregate one EvaluationReport per session across sessions."""
    from ntt.evaluation.core.types import EvaluationReport

    if not reports:
        return EvaluationReport(results={})
    if len(reports) != len(session_weights):
        raise ValueError("reports and session_weights must have the same length")

    first = reports[0]
    from ntt.evaluation.aggregation.summary import aggregate_by_category

    for report in reports[1:]:
        if report.config != first.config:
            raise ValueError("Report configs differ; cannot aggregate safely.")
        if report.original_data_type != first.original_data_type:
            raise ValueError("Report original_data_type values differ; cannot aggregate safely.")
        if report.reconstructed_data_type != first.reconstructed_data_type:
            raise ValueError(
                "Report reconstructed_data_type values differ; cannot aggregate safely."
            )
    metric_names: set[str] = set()
    for r in reports:
        metric_names |= set(r.results.keys())

    # Merge skipped_metrics across sessions, with per-session counts.
    merged_skipped: dict[str, str] = {}
    skipped_seen_count: dict[str, int] = {}
    for r in reports:
        for name, reason in r.skipped_metrics.items():
            if name in merged_skipped:
                skipped_seen_count[name] += 1
            else:
                merged_skipped[name] = reason
                skipped_seen_count[name] = 1
    final_skipped = {
        name: (
            f"{reason} (skipped on {skipped_seen_count[name]} of {len(reports)} sessions)"
            if skipped_seen_count[name] > 1
            else f"{reason} (skipped on 1 of {len(reports)} sessions)"
        )
        for name, reason in merged_skipped.items()
    }

    aggregated_results: dict[str, MetricResult] = {}
    for name in sorted(metric_names):
        values: list[float] = []
        weights: list[float] = []
        meta_items: list[dict[str, Any]] = []
        template: MetricResult | None = None
        for r, w in zip(reports, session_weights, strict=True):
            if name not in r.results:
                continue
            mr = r.results[name]
            if template is None:
                template = mr
            sv = _scalar_metric_value(mr)
            if sv is not None:
                values.append(sv)
                weights.append(float(w))
                meta_items.append(dict(mr.metadata))
        if not values or template is None:
            continue
        evaluation_config = first.config or {}
        configured_metrics = evaluation_config.get("metrics") or {}
        metric_parameters = configured_metrics.get(name, {})
        metadata_fields_that_must_match = set()
        # The saved `n_splits` value counts the cross-validation folds that are valid. It
        # does not count the number of folds set in the configuration.
        for parameter_name, configured_value in metric_parameters.items():
            if configured_value is not None and parameter_name != "n_splits":
                metadata_fields_that_must_match.add(parameter_name)

        aggregated_results[name] = _aggregate_metric_stack(
            template,
            values,
            weights,
            meta_items,
            strategy,
            count_key="n_sessions",
            count=len(values),
            metadata_fields_that_must_match=metadata_fields_that_must_match,
        )

    out = EvaluationReport(
        results=aggregated_results,
        category_scores=(
            aggregate_by_category(aggregated_results) if aggregated_results else None
        ),
        config=first.config,
        original_data_type=first.original_data_type,
        reconstructed_data_type=first.reconstructed_data_type,
    )
    out.skipped_metrics = final_skipped
    return out


def aggregate_session_reports(
    session_reports: dict[str, list[tuple[CollatedValidationSegment, dict[str, Any]]]],
    strategy: AggregationStrategy = AggregationStrategy.MEAN,
) -> EvaluationReport:
    """Aggregate per-session evaluation reports into a single run-level report.

    **Two-level aggregation:** metrics are first aggregated across all validation
    segments *within* each session, then across sessions. This matches the
    docstring intent (per-session means) and avoids letting sessions with more
    segments dominate a flat segment-level average.

    Args:
        session_reports: Output from run_validation_evaluation_per_session.
            Dict mapping session_id to list of (segment, {"report": report, ...}).
        strategy: Aggregation strategy at both levels (within session, across sessions).
            - MEAN: Simple mean
            - MEDIAN: Median
            - WEIGHTED_MEAN: Weight segments by duration; weight each session by its
              total segment duration when combining sessions

    Returns:
        Aggregated EvaluationReport (one scalar per metric at run level).
    """
    session_reports_list: list[EvaluationReport] = []
    session_weights: list[float] = []

    for _session_id, segment_list in session_reports.items():
        if not segment_list:
            continue
        session_reports_list.append(aggregate_segment_reports(segment_list, strategy))
        total_time = 0.0
        for segment, _ in segment_list:
            shape = (
                segment.original_shape
                if segment.original_shape is not None
                else (tuple(segment.original.shape) if segment.original is not None else None)
            )
            if shape is None:
                raise RuntimeError("segment has neither original tensor nor original_shape")
            total_time += float(shape[1])
        session_weights.append(total_time)

    return _aggregate_evaluation_reports(session_reports_list, session_weights, strategy)
