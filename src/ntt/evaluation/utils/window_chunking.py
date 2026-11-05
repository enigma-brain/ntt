"""Pre-chunking wrapper for fixed-input-window tokenizers in eval pipelines.

Some tokenizers (compressed sensing, simple_vae, cosmos_2d, ...) build their
architecture for a specific input window length at construction time and
cannot consume windows of arbitrary length. Eval pipelines, however, often
run with a window length (typically 1 s) longer than the tokenizer's training
window. When the two differ this wrapper splits the input window along the
time axis into integer-multiple chunks of width
``spec.effective_window_length_bins``, calls the tokenizer per chunk, and
concatenates outputs back along time.

The public entry point is :func:`chunked_tokenizer_fn` — call it once at
setup to obtain a drop-in replacement for the tokenizer's ``tokenize_fn``.
For variable-input tokenizers it is a no-op pass-through (except for the
shape-error hint described below). It does ONE thing: chunk for fixed-input
tokenizers. Padding, mask handling, and per-sample validity are the caller's
responsibility — this wrapper just gives the tokenizer correctly-sized
inputs and stitches per-chunk outputs back together.

Detection (:func:`requires_fixed_input_window`) is decided solely by the
spec's ``variable_t`` capability flag. Variable-T tokenizers (incl. calm) are
validated against the eval window by ``assert_tokenizer_accepts_eval_window``
and fed directly.

Trailing remainder (``T % T_chunk``) is dropped silently. If the input is
shorter than one tokenizer chunk we raise ``ValueError`` so the caller can
catch it and skip the batch.

Concatenated output fields (time axis 1):
  ``reconstruction``, ``latents``, ``token_timestamps``,
  ``token_start_time``, ``token_end_time``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import torch

from ntt.core.components.types import (
    BinnedInputType,
    BinnedSpikesInput,
    RatesInput,
    ReconstructionOutput,
    get_data_matrix,
)
from ntt.core.model.tokenizer.base import TokenizerSpec

_MODULE_PATH = "ntt.evaluation.utils.window_chunking"
_SHAPE_ERROR_KEYWORDS = ("size mismatch", "shape", "expected")

__all__ = [
    "requires_fixed_input_window",
    "chunked_tokenizer_fn",
]


def requires_fixed_input_window(spec: TokenizerSpec) -> bool:
    """Return whether ``spec`` must be fed fixed-size chunks.

    Derived solely from ``variable_t``: a tokenizer that cannot handle variable
    time lengths is chunked into fixed-size pieces by ``chunked_tokenizer_fn``.
    Variable-T tokenizers (incl. calm) are validated against the eval window by
    ``assert_tokenizer_accepts_eval_window`` and fed directly.
    """
    return not spec.variable_t


def _is_shape_error(exc: RuntimeError | ValueError) -> bool:
    """Return True when ``exc`` looks like a tensor shape mismatch."""
    message = str(exc).lower()
    return any(keyword in message for keyword in _SHAPE_ERROR_KEYWORDS)


def _shape_error_hint(spec: TokenizerSpec, exc: RuntimeError | ValueError) -> str:
    """Build the fixed-window hint for a misdetected variable tokenizer."""
    return (
        f"Tokenizer KIND={spec.KIND!r} was treated as variable-input by "
        f"`requires_fixed_input_window(spec)` in `{_MODULE_PATH}`, but tokenization "
        f"raised a shape-style error: {exc}. If this tokenizer actually needs fixed-size "
        f"chunking, re-examine its `variable_t` capability flag."
    )


def _effective_window_length_bins(spec: TokenizerSpec) -> int:
    """Resolve the tokenizer's required input length in bins.

    Prefers ``spec.effective_window_length_bins`` when the spec defines it
    (some specs do — e.g. variable-length training but fixed eval window).
    Otherwise computes ``round(effective_window_length_seconds /
    bin_size_seconds)``.
    """
    candidate = getattr(spec, "effective_window_length_bins", None)
    if candidate is not None:
        return int(candidate)
    bin_size_seconds = spec.bin_size_seconds
    if bin_size_seconds is False:
        raise ValueError(
            f"Cannot compute effective window length in bins for unbinned tokenizer "
            f"KIND={spec.KIND!r} (bin_size_seconds is False)."
        )
    return round(spec.effective_window_length_seconds / float(bin_size_seconds))


def _sampling_rate_scalar(batch: BinnedInputType) -> float:
    """Assert a uniform batch sampling rate and return it as a Python float."""
    sampling_rate = batch.sampling_rate
    if isinstance(sampling_rate, torch.Tensor):
        flat = sampling_rate.reshape(-1)
        if flat.numel() == 0:
            raise ValueError("sampling_rate tensor is empty.")
        assert bool(torch.all(flat == flat[0])), (
            "Expected all batch sampling_rate values to match before chunking, got "
            f"{flat.detach().cpu().tolist()}."
        )
        return float(flat[0].item())
    return float(sampling_rate)


def _slice_batch_time(
    batch: BinnedInputType,
    *,
    t_start: int,
    t_end: int,
    sampling_rate: float,
) -> BinnedInputType:
    """Return a copy of ``batch`` with the time axis sliced to ``[t_start, t_end)``.

    Time-dependent fields (``spike_matrix`` / ``rate_matrix``,
    ``pad_and_gap_mask``, ``absolute_start``, ``absolute_end``,
    ``duration_sec``) are recomputed for the sub-window. All other fields are
    passed through unchanged. Behavioural fields are intentionally NOT
    propagated into per-chunk batches — behavioural alignment in the eval
    pipeline operates on token timestamps assembled back at full-window
    resolution, not on per-chunk batches.
    """
    is_rates = isinstance(batch, RatesInput)
    data_matrix = batch.rate_matrix if is_rates else batch.spike_matrix
    sub_data = data_matrix[:, t_start:t_end, :]
    sub_mask = batch.pad_and_gap_mask[:, t_start:t_end, :]

    if batch.absolute_start is None:
        raise ValueError("_slice_batch_time requires absolute_start to be set.")
    chunk_seconds = float(t_end - t_start) / float(sampling_rate)
    new_absolute_start = batch.absolute_start + float(t_start) / float(sampling_rate)
    new_absolute_end = new_absolute_start + chunk_seconds
    new_duration = torch.full_like(batch.duration_sec, chunk_seconds)

    common = {
        "pad_and_gap_mask": sub_mask,
        "sampling_rate": batch.sampling_rate,
        "unit_ids": batch.unit_ids,
        "unit_ids_pad_mask": batch.unit_ids_pad_mask,
        "duration_sec": new_duration,
        "session_id": batch.session_id,
        "subject_id": batch.subject_id,
        "species": batch.species,
        "absolute_start": new_absolute_start,
        "absolute_end": new_absolute_end,
        "behavioral_data": None,
        "behavioral_intervals": None,
    }
    if is_rates:
        return RatesInput(rate_matrix=sub_data, **common)
    return BinnedSpikesInput(spike_matrix=sub_data, **common)


def _chunk_values_differ(a: Any, b: Any) -> bool:
    """Return True if two per-chunk extra values are not equal (tensor-aware)."""
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            return True
        return a.shape != b.shape or not torch.equal(a, b)
    return a != b


def _concat_chunk_outputs(
    chunk_outputs: list[dict[str, Any] | ReconstructionOutput],
) -> dict[str, Any] | ReconstructionOutput:
    """Concatenate per-chunk tokenizer outputs along time axis 1."""
    if not chunk_outputs:
        raise ValueError("_concat_chunk_outputs requires at least one chunk.")

    first = chunk_outputs[0]

    if isinstance(first, dict):
        # Chunks of one tokenizer share an output type, so all of them are dicts.
        dict_outputs = cast(list[dict[str, Any]], chunk_outputs)
        merged: dict[str, Any] = {}
        # ``reconstruction`` joins the all-or-none time-axis-concat group so that
        # a tokenize-only ``tokenize_fn`` (no decoder, no reconstruction key) is
        # supported uniformly with continuous outputs like ``latents``. A
        # tokenizer that returns reconstruction for some chunks but not others
        # still raises — that is a real inconsistency.
        for key in (
            "reconstruction",
            "latents",
            "token_timestamps",
            "token_start_time",
            "token_end_time",
        ):
            present = [key in out and out[key] is not None for out in dict_outputs]
            if all(present):
                merged[key] = torch.cat([out[key] for out in dict_outputs], dim=1)
            elif any(present):
                raise ValueError(
                    f"Chunk output key {key!r} is present in some chunks but not "
                    "all; refusing to silently drop it (would misalign the merged "
                    "output). This indicates a tokenizer returning inconsistent "
                    "outputs across equal-shaped chunks."
                )
        # Pass through extra keys (no time axis to concat along), but only if
        # they agree across chunks — a per-chunk-varying extra key cannot be
        # merged safely, so refuse rather than silently take chunk 0's value.
        for key in first:
            if key in merged:
                continue
            values = [out.get(key) for out in dict_outputs]
            if any(_chunk_values_differ(values[0], v) for v in values[1:]):
                raise ValueError(
                    f"Chunk output key {key!r} differs across chunks and has no "
                    "time axis to concatenate along; cannot merge safely."
                )
            merged[key] = values[0]
        return merged

    if isinstance(first, ReconstructionOutput):
        recon_outputs = cast(list[ReconstructionOutput], chunk_outputs)
        merged_recon = torch.cat([out.reconstruction for out in recon_outputs], dim=1)
        return first._replace(reconstruction=merged_recon)

    raise TypeError(
        f"Unsupported tokenizer output type for pre-chunking: {type(first).__name__}. "
        "Expected dict (from reconstruct_from_input(return_latents=True)) or "
        "ReconstructionOutput."
    )


def chunked_tokenizer_fn(
    tokenize_fn: Callable[[BinnedInputType], dict[str, Any] | ReconstructionOutput],
    *,
    spec: TokenizerSpec,
) -> Callable[[BinnedInputType], dict[str, Any] | ReconstructionOutput]:
    """Wrap ``tokenize_fn`` to pre-chunk inputs for fixed-input tokenizers.

    Use as a drop-in replacement for the tokenizer function: call this once at
    setup, then use the returned callable everywhere ``tokenize_fn`` would
    have been used. The returned callable accepts the same input and returns
    the same kind of output as ``tokenize_fn``; only the time-axis length of
    the relevant output fields differs (any trailing ``T % T_chunk`` bins are
    silently dropped).

    For variable-input tokenizers it is a near no-op: it returns a thin
    wrapper that catches shape-style errors and re-raises with a hint pointing
    at the spec's ``variable_t`` capability flag — so a tokenizer that should
    have been declared fixed-input fails loudly.

    For fixed-input tokenizers the returned callable slices each incoming
    batch along time axis 1 into ``n_chunks = T // T_chunk`` adjacent chunks,
    calls ``tokenize_fn`` per chunk, and concatenates the per-chunk outputs
    along time axis 1. Raises ``ValueError`` if the batch is shorter than one
    chunk.
    """
    if not requires_fixed_input_window(spec):

        def _passthrough(batch: BinnedInputType) -> dict[str, Any] | ReconstructionOutput:
            try:
                return tokenize_fn(batch)
            except (RuntimeError, ValueError) as exc:
                if _is_shape_error(exc):
                    raise type(exc)(_shape_error_hint(spec, exc)) from exc
                raise

        return _passthrough

    chunk_width = _effective_window_length_bins(spec)
    if chunk_width <= 0:
        raise ValueError(f"Chunk width must be positive, got {chunk_width}.")

    def _chunked(batch: BinnedInputType) -> dict[str, Any] | ReconstructionOutput:
        if not isinstance(batch, (BinnedSpikesInput, RatesInput)):
            raise TypeError(
                "chunked_tokenizer_fn expects BinnedSpikesInput or RatesInput, got "
                f"{type(batch).__name__}."
            )

        data_matrix = get_data_matrix(batch)
        n_bins_total = data_matrix.shape[1]
        n_chunks = n_bins_total // chunk_width
        if n_chunks == 0:
            raise ValueError(
                f"Input window ({n_bins_total} bins) is shorter than one tokenizer "
                f"chunk ({chunk_width} bins) for KIND={spec.KIND!r}; cannot tokenize."
            )

        if n_chunks == 1 and n_bins_total == chunk_width:
            # Exact match — no chunking work to do.
            return tokenize_fn(batch)

        sampling_rate = _sampling_rate_scalar(batch)
        chunk_outputs: list[dict[str, Any] | ReconstructionOutput] = []
        for k in range(n_chunks):
            t_start = k * chunk_width
            t_end = t_start + chunk_width
            sub_batch = _slice_batch_time(
                batch, t_start=t_start, t_end=t_end, sampling_rate=sampling_rate
            )
            chunk_outputs.append(tokenize_fn(sub_batch))

        return _concat_chunk_outputs(chunk_outputs)

    return _chunked
