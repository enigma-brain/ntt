"""Replica-aware validation collation (M1 input-corruption robustness).

This module holds the replica fan-out collation, separate from
:mod:`ntt.evaluation.utils.validation_collation`. The standard
``iter_collated_validation_segments`` knows nothing about replicas; the replica fan-out lives here
as :func:`iter_collated_validation_segments_replicas` and is consumed
by :mod:`ntt.jobs.eval_robustness_input` (the input_robustness (M1) job entry).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from typing import Any, cast

import numpy as np
import torch
from lightning.pytorch.utilities import move_data_to_device
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ntt.core.components.types import BinnedSpikesInput, RatesInput
from ntt.core.model.tokenizer.base import DataType
from ntt.evaluation.config import EvaluationConfig
from ntt.evaluation.core.types import EvalData
from ntt.evaluation.pipeline import EvaluationPipeline
from ntt.evaluation.utils.eval_collation_helpers import (
    _dbg_check_token_count_via_bins_per_token,
    _strip_input_padding,
    _strip_token_padding,
)
from ntt.evaluation.utils.reconstruction_type import ReconstructionTypeRecorder
from ntt.evaluation.utils.validation_collation import (
    CollatedValidationSegment,
    _apply_pad_and_gap_mask,
    _assert_token_bin_divides_window_length,
    _build_window_part,
    _ensure_batch_dim,
    _extract_latents,
    _extract_metadata,
    _extract_reconstruction,
    _extract_sampling_rate,
    _extract_scalar_sampling_rate,
    _flush_segment,
    _merge_window_parts,
    _normalize_session_ids,
)

logger = logging.getLogger(__name__)


def iter_collated_validation_segments_replicas(
    dataloader: DataLoader,
    tokenizer_fun: Callable[[BinnedSpikesInput | RatesInput], Any],
    *,
    n_replicas: int,
    token_bin_size_ms: float | None = None,
    window_length_seconds: float | None = None,
    device: torch.device | None = None,
    max_windows: int | None = None,
    max_bytes: int | None = None,
    max_eval_seconds: float | None = None,
    _on_batch_done: Callable[[], None] | None = None,
) -> Iterator[CollatedValidationSegment]:
    """Replica-aware validation collator — replica fan-out for input_robustness (M1).

    Note: input_robustness (M1) does not need window concatenation, so the
    buffering and contiguity checks in this iterator exceed what M1 requires.

    Each ``dataloader`` batch is interpreted as one logical source window
    replicated ``K+1 = n_replicas + 1`` times along dim 0: row 0 is the
    clean sample, rows 1..K are corrupted replicas produced by the
    upstream replica dataset. The tokenizer is called once on the full
    ``[K+1, T, U]`` batch; replica 0 drives the per-window padding strip
    and buffering so every existing metric sees clean data. The full
    ``K+1`` stacks of latents and reconstructions are threaded through
    the buffer as side-fields and exposed on the emitted
    ``CollatedValidationSegment`` via ``replica_latents`` /
    ``replica_reconstructions`` / ``replica_metadata``. The new input_robustness (M1)
    robustness metric consumes those fields; no other metric sees them.

    Unlike the standard iterator, no inter-window overlap trimming is
    performed (neither ``_trim_input_overlap`` nor ``_trim_token_overlap``).
    input_robustness (M1) is per-window and does not need overlap handling; if the val
    sampler produces overlapping windows the redundancy only inflates
    the per-session sample count, it does not corrupt input_robustness (M1)'s results.

    Build the replica dataloader with
    :func:`ntt.core.data.replica.build_replica_dataloader`.

    """
    # Startup-time fail-loud parity with ``iter_collated_validation_segments``:
    # token_bin_size_ms must evenly divide the configured input window length
    # so that reconstruction and original have matched bin counts. Skipped
    # when either value is None (dynamic-bin tokenizer or unknown window).
    _assert_token_bin_divides_window_length(window_length_seconds, token_bin_size_ms)

    expected_group = n_replicas + 1

    current_session_id: str | None = None
    window_buffer: list[dict] = []
    segment_index = 0
    next_window_group_id_token_res = 0
    buffered_bytes = 0
    session_accumulated_seconds: float = 0.0
    session_capped: bool = False

    def flush(is_partial: bool) -> Iterator[CollatedValidationSegment]:
        """Flush the buffered windows for the current session into one
        ``CollatedValidationSegment`` (clean side) plus stacked replica
        passthroughs."""
        nonlocal window_buffer, segment_index, buffered_bytes

        if current_session_id is None or not window_buffer:
            return iter(())

        # input_robustness (M1) does not drop overlapping windows — see module docstring.
        # Skip the inter-window gap check: input_robustness (M1) fans windows out into
        # per-window (clean, corrupted_k) pairs in ``expand_results``,
        # so analysis sub-windows are not required to be contiguous
        # (e.g. WindowShift corruptions deliberately break contiguity).
        segment = _flush_segment(
            window_buffer,
            drop_overlapping_windows=False,
            skip_gap_check=True,
        )
        kept_windows = list(window_buffer)
        window_buffer = []
        buffered_bytes = 0

        rep_recon_list = [
            w["replica_reconstructions"]
            for w in kept_windows
            if w.get("replica_reconstructions") is not None
        ]
        rep_latent_list = [
            w["replica_latents"] for w in kept_windows if w.get("replica_latents") is not None
        ]
        rep_orig_list = [
            w["replica_originals"] for w in kept_windows if w.get("replica_originals") is not None
        ]
        replica_reconstructions: torch.Tensor | list[torch.Tensor] | None = None
        if rep_recon_list:
            shapes = {t.shape for t in rep_recon_list}
            if len(shapes) == 1:
                replica_reconstructions = torch.stack(rep_recon_list, dim=0)
            else:
                replica_reconstructions = rep_recon_list
        replica_latents: torch.Tensor | list[torch.Tensor] | None = None
        if rep_latent_list:
            shapes = {t.shape for t in rep_latent_list}
            if len(shapes) == 1:
                replica_latents = torch.stack(rep_latent_list, dim=0)
            else:
                replica_latents = rep_latent_list
        replica_originals: torch.Tensor | list[torch.Tensor] | None = None
        if rep_orig_list:
            shapes = {t.shape for t in rep_orig_list}
            if len(shapes) == 1:
                replica_originals = torch.stack(rep_orig_list, dim=0)
            else:
                replica_originals = rep_orig_list

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
            replica_latents=replica_latents,
            replica_reconstructions=replica_reconstructions,
            replica_originals=replica_originals,
            replica_metadata={
                "n_replicas": n_replicas,
                "session_id": segment.session_id,
            },
        )
        segment_index += 1
        return iter((segment,))

    def buffer_window(window_dict: dict[str, Any]) -> None:
        """Append a per-window dict to the buffer, preserving replica passthroughs."""
        nonlocal buffered_bytes, next_window_group_id_token_res

        if (
            window_dict.get("latents") is not None
            and window_dict.get("behavioral_data_input_res") is not None
        ):
            replica_recon = window_dict.get("replica_reconstructions")
            replica_lat = window_dict.get("replica_latents")
            replica_orig = window_dict.get("replica_originals")
            window_dict = _merge_window_parts(
                [window_dict],
                window_group_id_token_res=next_window_group_id_token_res,
            )
            if replica_recon is not None:
                window_dict["replica_reconstructions"] = replica_recon
            if replica_lat is not None:
                window_dict["replica_latents"] = replica_lat
            if replica_orig is not None:
                window_dict["replica_originals"] = replica_orig
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

    for batch in dataloader:
        if isinstance(batch, BinnedSpikesInput):
            spike_matrix = batch.spike_matrix
        elif isinstance(batch, RatesInput):
            spike_matrix = batch.rate_matrix
        else:
            raise ValueError(f"Unsupported batch type: {type(batch)}.")

        pad_and_gap_mask = batch.pad_and_gap_mask
        spike_matrix, pad_and_gap_mask = _ensure_batch_dim(spike_matrix, pad_and_gap_mask)

        if spike_matrix.shape[0] != expected_group:
            raise ValueError(
                f"Replica collation expected batch leading dim to be "
                f"n_replicas + 1 = {expected_group}, got {spike_matrix.shape[0]}. "
                "Use `build_replica_dataloader()` from `ntt.core.data.replica`."
            )

        assert batch.absolute_start is not None
        assert batch.absolute_end is not None

        session_ids = _normalize_session_ids(batch.session_id, spike_matrix.shape[0])
        session_id = session_ids[0]
        if any(sid != session_id for sid in session_ids):
            raise ValueError(
                f"Replica group spans multiple sessions: {session_ids}. "
                "All K+1 rows of a replica batch must belong to the same source window."
            )
        if session_capped and current_session_id is not None and session_id == current_session_id:
            if _on_batch_done is not None:
                _on_batch_done()
            continue

        # Drop boundary-clipped windows before tokenization: fixed-length
        # tokenizers (e.g. compressed sensing) can fail on a short clean
        # replica before the later input-side strip logic gets a chance to
        # reject the window.
        if pad_and_gap_mask is not None:
            pre_token_mask_window = pad_and_gap_mask[0].T  # (units, time)
            time_valid = pre_token_mask_window.any(dim=0)
            if time_valid.numel() > 0 and not bool(time_valid[-1].item()):
                n_trailing_invalid = int((~time_valid).sum().item())
                window_absolute_start = float(
                    batch.absolute_start[0].item()
                    if batch.absolute_start.ndim > 0
                    else batch.absolute_start.item()
                )
                window_absolute_end = float(
                    batch.absolute_end[0].item()
                    if batch.absolute_end.ndim > 0
                    else batch.absolute_end.item()
                )
                logger.info(
                    "replica iterator: dropping boundary-clipped window "
                    "before tokenization (session_id=%r, abs=[%.6f, %.6f]s, "
                    "trailing invalid bins=%d)",
                    session_id,
                    window_absolute_start,
                    window_absolute_end,
                    n_trailing_invalid,
                )
                if _on_batch_done is not None:
                    _on_batch_done()
                continue

        if device is not None:
            batch = move_data_to_device(batch, device)

        # Fallback drop for partial last windows that weren't caught by the
        # pre-token pad_and_gap_mask path: the binner can produce a few-bin-
        # short window when the spike domain ends slightly before the
        # requested slice (typical at session boundaries; bites
        # window_shift|max_shift_ms=0/1 specs on fixed-input tokenizers
        # without a pad_and_gap_mask). The chunker raises a specific
        # ValueError for this; skip just this batch instead of aborting the
        # whole corruption spec.
        try:
            output = tokenizer_fun(batch)
        except ValueError as exc:
            msg = str(exc)
            if "shorter than one tokenizer chunk" in msg:
                logger.info(
                    "replica iterator: dropping partial window at tokenize "
                    "time (session_id=%r): %s",
                    session_id,
                    msg,
                )
                if _on_batch_done is not None:
                    _on_batch_done()
                continue
            raise

        reconstructed = _extract_reconstruction(output, require=False)
        batch_latents = _extract_latents(output)

        batch_token_start_times = None
        batch_token_end_times = None
        batch_token_timestamps = None
        if isinstance(output, dict):
            # ``.clone()`` avoids holding a view onto the worker's output buffer,
            # which could keep the underlying allocation alive across replicas.
            if "token_start_time" in output:
                batch_token_start_times = output["token_start_time"].cpu().clone()
            if "token_end_time" in output:
                batch_token_end_times = output["token_end_time"].cpu().clone()
            if "token_timestamps" in output:
                batch_token_timestamps = output["token_timestamps"].cpu().clone()

        if isinstance(batch_latents, torch.Tensor):
            batch_latents = batch_latents.detach().cpu().clone()
        elif batch_latents is not None:
            raise TypeError(
                f"Expected latents to be a torch.Tensor, got {type(batch_latents).__name__}."
            )

        if reconstructed is not None:
            reconstructed = reconstructed.detach().cpu().clone()
        spike_matrix = spike_matrix.detach().cpu().clone()
        pad_and_gap_mask = (
            pad_and_gap_mask.detach().cpu().clone() if pad_and_gap_mask is not None else None
        )

        sampling_rate = _extract_sampling_rate(batch)

        if current_session_id is None:
            current_session_id = session_id
            next_window_group_id_token_res = 0
            session_accumulated_seconds = 0.0
            session_capped = False
        if session_id != current_session_id:
            yield from flush(is_partial=False)
            current_session_id = session_id
            next_window_group_id_token_res = 0
            session_accumulated_seconds = 0.0
            session_capped = False

        if session_capped:
            if _on_batch_done is not None:
                _on_batch_done()
            continue

        abs_s = batch.absolute_start
        abs_e = batch.absolute_end
        assert abs_s is not None and abs_e is not None
        window_absolute_start = float(abs_s[0].item() if abs_s.ndim > 0 else abs_s.item())
        window_absolute_end = float(abs_e[0].item() if abs_e.ndim > 0 else abs_e.item())

        original_window = spike_matrix[0].T
        reconstructed_window = reconstructed[0].T if reconstructed is not None else None
        mask_window = pad_and_gap_mask[0].T if pad_and_gap_mask is not None else None

        full_reconstructed: torch.Tensor | None = (
            reconstructed.permute(0, 2, 1).contiguous() if reconstructed is not None else None
        )
        # Per-window K+1 stack of the *input* spikes — replica 0 is clean,
        # rows 1..K are the corruption-applied input replicas. Permuted to
        # match ``full_reconstructed``'s ``[K+1, U, T]`` layout so the same
        # trailing-only strip below applies uniformly. This is the
        # input-side originals intermediate, consumed by downstream
        # metrics.

        full_originals = spike_matrix.permute(0, 2, 1).contiguous()

        # input_robustness (M1) does not consume behavioral
        # data — the M1-a latent-distance metric does not read it — so the
        # behavioral alignment block is skipped in this iterator.
        #
        # Note: aligning behavioral fields on the replica path can fail. A
        # wan-A vae (conv-* family) input_robustness (M1) run hit
        # ``Field 'trial_id' length 996 != source_timestamps length 997`` in
        # resample_fields_to_token_timestamps. Standard
        # eval on the same dataset does not trip the same assertion —
        # the replica dataloader's transform pipeline (or the way
        # behavioral fields are carried through it) produces an
        # off-by-one between the spike_matrix bin count and
        # behavioral-field length that the standard val dataloader
        # does not. The root cause is not
        # investigated: input_robustness (M1) does not need this data, so the
        # block is skipped. A metric that needs behavioral data on the replica
        # path would require diagnosing the off-by-one first.
        window_behavioral_data_token_res: dict[str, np.ndarray] | None = None
        window_behavioral_data_input_res: dict[str, np.ndarray] | None = None

        pre_strip_valid_units: torch.Tensor | None = None
        pre_strip_valid_time: torch.Tensor | None = None
        if mask_window is not None and not mask_window.all():
            pre_strip_valid_units = mask_window.any(dim=1)
            pre_strip_valid_time = mask_window.any(dim=0)
            if not pre_strip_valid_time.any() or not pre_strip_valid_units.any():
                # Entire window is masked out — no valid (unit, time) cell.
                # Skip rather than fall through to ``nonzero()[-1]`` on an
                # empty tensor (which would IndexError).
                if _on_batch_done is not None:
                    _on_batch_done()
                continue
            original_window, reconstructed_window, mask_window = _strip_input_padding(
                original_window, reconstructed_window, mask_window
            )
            # Trailing padding is gone; any remaining False is an interior
            # temporal gap (zero-filled bins between disjoint valid intervals).
            # Skip the window rather than feed garbage gap bins into metrics —
            # mirrors the numel()==0 skip below. Done before the full_* strips
            # so we don't waste work on a window we are discarding.
            if mask_window is not None and not mask_window.all():
                if _on_batch_done is not None:
                    _on_batch_done()
                continue
            # Apply the same trailing-only strip strategy from
            # ``_strip_input_padding`` to the K+1-stacked reconstruction.
            # Units always strip safely (matches U). For the time axis,
            # the tokenizer's reconstruction dim (``T_recon``) may differ
            # from the input mask's dim (``T_input``) — e.g. when an
            # autoencoder pads internally to a multiple of its token
            # stride. We only trim time when the two axes agree, and we
            # use a positional slice (trailing-only), not boolean
            # indexing on the input mask, so that a length mismatch
            # raises in a known place rather than silently mis-aligning.
            if full_reconstructed is not None:
                full_reconstructed = full_reconstructed[:, pre_strip_valid_units, :]
                if pre_strip_valid_time.numel() == full_reconstructed.shape[-1]:
                    last_valid_time = (
                        int(pre_strip_valid_time.nonzero(as_tuple=False)[-1].item()) + 1
                    )
                    full_reconstructed = full_reconstructed[:, :, :last_valid_time]
                else:
                    logger.debug(
                        "replica iterator: T_input=%d != T_recon=%d; leaving the "
                        "K+1 reconstruction time axis untrimmed.",
                        pre_strip_valid_time.numel(),
                        full_reconstructed.shape[-1],
                    )
            # Strip the K+1 *input* stack with the same units selector and
            # trailing-time slice. ``full_originals`` always shares its
            # time axis with the input mask (no internal padding), so the
            # trailing-only slice is always applicable here.

            full_originals = full_originals[:, pre_strip_valid_units, :]
            assert pre_strip_valid_time.numel() == full_originals.shape[-1], (
                f"replica iterator: input mask T={pre_strip_valid_time.numel()} "
                f"!= full_originals T={full_originals.shape[-1]}; the input "
                f"spike stack and the pad/gap mask should always share the "
                f"same time axis."
            )
            last_valid_time_in = int(pre_strip_valid_time.nonzero(as_tuple=False)[-1].item()) + 1
            full_originals = full_originals[:, :, :last_valid_time_in]

        if original_window.numel() == 0:
            if _on_batch_done is not None:
                _on_batch_done()
            continue

        window_latents: torch.Tensor | None = None
        window_token_start_times: torch.Tensor | None = None
        window_token_end_times: torch.Tensor | None = None
        window_token_timestamps: torch.Tensor | None = None
        full_latents: torch.Tensor | None = None
        effective_token_starts: torch.Tensor | None = None
        effective_token_ends: torch.Tensor | None = None

        if batch_latents is not None:
            window_latents = batch_latents[0]
            full_latents = batch_latents

            if pre_strip_valid_units is not None:
                n_valid = len(pre_strip_valid_units)
                if window_latents.ndim == 3 and window_latents.shape[1] == n_valid:
                    window_latents = window_latents[:, pre_strip_valid_units, :]
                    full_latents = full_latents[:, :, pre_strip_valid_units, :]
                elif window_latents.ndim == 2 and window_latents.shape[0] == n_valid:
                    window_latents = window_latents[pre_strip_valid_units]
                    full_latents = full_latents[:, pre_strip_valid_units, ...]

            if batch_token_start_times is not None:
                assert batch_token_end_times is not None
                window_token_start_times = batch_token_start_times[0]
                window_token_end_times = batch_token_end_times[0]

            assert batch_token_timestamps is not None, (
                "token_timestamps are required when latents are present."
            )
            window_token_timestamps = batch_token_timestamps[0]

            if window_token_start_times is not None:
                effective_token_starts = window_token_start_times
                effective_token_ends = window_token_end_times
            else:
                effective_token_starts = window_token_timestamps
                effective_token_ends = window_token_timestamps

            assert effective_token_starts is not None and effective_token_ends is not None
            strip_result = _strip_token_padding(
                window_latents,
                effective_token_starts,
                effective_token_ends,
                absolute_end=window_absolute_end,
                behavioral_data=window_behavioral_data_token_res,
            )
            keep_mask = strip_result["keep_mask"]
            window_latents = cast(torch.Tensor, strip_result["latents"])
            effective_token_starts = strip_result["token_start_times"]
            effective_token_ends = strip_result["token_end_times"]
            if window_token_start_times is not None:
                window_token_start_times = effective_token_starts
                window_token_end_times = effective_token_ends
            assert window_token_timestamps is not None
            window_token_timestamps = window_token_timestamps[keep_mask]
            window_behavioral_data_token_res = strip_result["behavioral_data"]

            if full_latents is not None:
                full_latents = full_latents[:, keep_mask, ...]

            # The assertion's invariant — token count from timestamps
            # equals ``real_input_bins // bins_per_token`` — holds here
            # because (a) ``_params_to_spec`` snaps WindowShift's
            # ``analysis_window_ms`` to a multiple of the tokenizer's
            # ``input_bin_ms`` so the analysis window is always
            # bin-aligned, and (b) boundary-clipped windows whose
            # actual duration is shorter than the analysis window are
            # dropped earlier in this iterator (see the time-mask check
            # above). With both in place there is no round-vs-floor ambiguity,
            # so the cross-check is reliable.
            if token_bin_size_ms is not None and window_latents.numel() > 0:
                _dbg_check_token_count_via_bins_per_token(
                    n_valid_tokens=window_latents.shape[0],
                    absolute_start=window_absolute_start,
                    absolute_end=window_absolute_end,
                    token_bin_size_ms=token_bin_size_ms,
                    sampling_rate=sampling_rate,
                    stripped_on_center_times=window_token_start_times is None,
                )

        # input_robustness (M1) is per-window — no inter-window overlap trimming. The
        # standard iterator does both ``_trim_input_overlap`` (input
        # bins) and an optional ``_trim_token_overlap`` (token-level)
        # to keep stitched session-level reconstructions / behavioral
        # alignment from double-counting overlapped regions. input_robustness (M1) doesn't
        # stitch and doesn't read behavioral data, so neither is
        # needed here.
        if original_window.numel() == 0:
            if _on_batch_done is not None:
                _on_batch_done()
            continue

        window_metadata = _extract_metadata(batch, 0)
        single_window = _build_window_part(
            original=original_window,
            reconstructed=reconstructed_window,
            mask=mask_window,
            absolute_start=window_absolute_start,
            sampling_rate=sampling_rate,
            session_id=session_id,
            metadata=window_metadata,
            has_overlap=False,
            behavioral_data_input_res=window_behavioral_data_input_res,
            behavioral_data_token_res=window_behavioral_data_token_res,
            latents=window_latents,
            token_start_times=window_token_start_times,
            token_end_times=window_token_end_times,
            token_timestamps=window_token_timestamps,
            start_bin=0,
            end_bin=original_window.shape[1],
        )
        if full_reconstructed is not None:
            single_window["replica_reconstructions"] = full_reconstructed
        single_window["replica_originals"] = full_originals
        if full_latents is not None:
            single_window["replica_latents"] = full_latents
        buffer_window(single_window)

        window_duration = window_absolute_end - window_absolute_start
        session_accumulated_seconds += window_duration

        if max_windows is not None and len(window_buffer) >= max_windows:
            yield from flush(is_partial=True)
        if max_bytes is not None and buffered_bytes >= max_bytes:
            yield from flush(is_partial=True)
        if max_eval_seconds is not None and session_accumulated_seconds >= max_eval_seconds:
            yield from flush(is_partial=True)
            session_capped = True

        if _on_batch_done is not None:
            _on_batch_done()

    yield from flush(is_partial=False)


def run_validation_evaluation_per_session_replicas(
    dataloader: DataLoader,
    tokenizer_fun: Callable[[BinnedSpikesInput | RatesInput], Any],
    *,
    n_replicas: int,
    data_type: DataType,
    bin_size_seconds: float | None,
    token_bin_size_ms: float | None = None,
    window_length_seconds: float | None = None,
    eval_config: EvaluationConfig | None = None,
    device: torch.device | None = None,
    max_windows: int | None = None,
    max_bytes: int | None = None,
    progress: bool = True,
    corruption_label: str | None = None,
    reconstructed_data_type: DataType | None = None,
) -> dict[str, list[tuple[CollatedValidationSegment, dict[str, Any]]]]:
    """Per-session driver for the input_robustness via replica fan-out (M1) evaluation path.

    Data types follow :func:`run_validation_evaluation_per_session`: ``data_type``
    labels the original, the reconstruction gets the type the tokenizer outputs
    declare, else ``reconstructed_data_type``, else ``data_type``.

    Mirrors :func:`run_validation_evaluation_per_session` but uses
    :func:`iter_collated_validation_segments_replicas` and threads the
    ``replica_latents`` / ``replica_reconstructions`` / ``replica_metadata``
    passthroughs into ``EvaluationPipeline.run`` so the input_robustness (M1) metric receives
    the corrupted-replica stack alongside the clean-side intermediates.
    Heavy tensor fields are released after each segment is reported on.

    """
    eval_config = eval_config or EvaluationConfig()
    if max_windows is None:
        max_windows = eval_config.max_windows
    if max_bytes is None:
        max_bytes = eval_config.max_bytes
    max_eval_seconds = eval_config.max_eval_seconds

    pipeline = EvaluationPipeline(config=eval_config)
    reports: dict[str, list[tuple[CollatedValidationSegment, dict[str, Any]]]] = {}

    # ── Progress bar ─────────────────────────────────────────────────
    try:
        n_batches: int | None = len(dataloader)
    except TypeError:
        n_batches = None

    bar = tqdm(
        total=n_batches,
        desc="Tokenizing (replicas)",
        disable=not progress,
        dynamic_ncols=True,
        leave=True,
    )

    def _on_batch_done() -> None:
        bar.update(1)

    recorder = ReconstructionTypeRecorder(tokenizer_fun)
    segment_iter = iter_collated_validation_segments_replicas(
        dataloader,
        recorder,
        n_replicas=n_replicas,
        token_bin_size_ms=token_bin_size_ms,
        window_length_seconds=window_length_seconds,
        device=device,
        max_windows=max_windows,
        max_bytes=max_bytes,
        max_eval_seconds=max_eval_seconds,
        _on_batch_done=_on_batch_done,
    )

    bar.reset(total=None)
    bar.set_description("Computing metrics (replicas)")

    _log = logging.getLogger(__name__)

    for segment in segment_iter:
        t_segment_start = time.perf_counter()
        session_short = segment.session_id.rsplit("/", 1)[-1]
        bar.set_description(f"Metrics (replicas) · {session_short}")

        original = _apply_pad_and_gap_mask(segment.original, segment.pad_and_gap_mask)
        reconstructed = _apply_pad_and_gap_mask(segment.reconstructed, segment.pad_and_gap_mask)
        assert original is not None, "Replica segments always carry the original data."

        duration = None
        if bin_size_seconds is not None:
            duration = original.shape[1] * bin_size_seconds

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
                data_type=recorder.declared or reconstructed_data_type or data_type,
                bin_size_seconds=bin_size_seconds,
                duration=duration,
            )

        latents_np = None
        if isinstance(segment.latents, torch.Tensor):
            latents_np = segment.latents.numpy()
        elif isinstance(segment.latents, list):
            latents_np = [
                item.numpy() if isinstance(item, torch.Tensor) else item
                for item in segment.latents
            ]

        token_start_times_np = None
        token_end_times_np = None
        token_timestamps_np = None
        if segment.token_start_times is not None:
            assert segment.token_end_times is not None
            token_start_times_np = segment.token_start_times.numpy()
            token_end_times_np = segment.token_end_times.numpy()
        if segment.token_timestamps is not None:
            token_timestamps_np = segment.token_timestamps.numpy()

        replica_latents_np = None
        if isinstance(segment.replica_latents, torch.Tensor):
            replica_latents_np = segment.replica_latents.numpy()
        elif isinstance(segment.replica_latents, list):
            replica_latents_np = [
                item.numpy() if isinstance(item, torch.Tensor) else item
                for item in segment.replica_latents
            ]
        replica_reconstructions_np = None
        if isinstance(segment.replica_reconstructions, torch.Tensor):
            replica_reconstructions_np = segment.replica_reconstructions.numpy()
        elif isinstance(segment.replica_reconstructions, list):
            replica_reconstructions_np = [
                item.numpy() if isinstance(item, torch.Tensor) else item
                for item in segment.replica_reconstructions
            ]
        replica_originals_np = None
        if isinstance(segment.replica_originals, torch.Tensor):
            replica_originals_np = segment.replica_originals.numpy()
        elif isinstance(segment.replica_originals, list):
            replica_originals_np = [
                item.numpy() if isinstance(item, torch.Tensor) else item
                for item in segment.replica_originals
            ]

        # Inject the per-spec ``corruption_labels`` so the input_robustness (M1) metric
        # groups all K replicas under one label and computes std across
        # replicas (not across windows).
        if corruption_label is not None:
            if not isinstance(segment.replica_metadata, dict):
                segment.replica_metadata = {}
            segment.replica_metadata["corruption_labels"] = [str(corruption_label)] * int(
                n_replicas
            )

        report = pipeline.run(
            original_data,
            reconstructed_data,
            compute_baseline=eval_config.compute_baseline,
            latents=latents_np,
            behavioral_data_token_res=None,  # segment too small to compute behavioral metrics
            behavioral_data_input_res=None,  # same
            token_timestamps=token_timestamps_np,
            token_start_times=token_start_times_np,
            token_end_times=token_end_times_np,
            sampling_rate=_extract_scalar_sampling_rate(segment.metadata.get("sampling_rate")),
            segment_absolute_start=segment.segment_absolute_start,
            token_bin_size_ms=token_bin_size_ms,
            session_id=segment.session_id,
            replica_latents=replica_latents_np,
            replica_reconstructions=replica_reconstructions_np,
            replica_originals=replica_originals_np,
            replica_metadata=segment.replica_metadata,
        )

        t_segment_elapsed = time.perf_counter() - t_segment_start
        _log.debug(
            "Replica segment %s done in %.1fs",
            session_short,
            t_segment_elapsed,
        )
        # Surface metric-level skips from the pipeline so the input_robustness (M1) job log
        # actually reports when ``robustness_input`` (or any other metric)
        # was filtered/errored out by the pipeline. Otherwise the skip is
        # silent — empty rows in csv with no diagnostic.
        if getattr(report, "skipped_metrics", None):
            _log.warning(
                "Replica segment %s — pipeline skipped metrics: %s",
                session_short,
                dict(report.skipped_metrics),
            )

        metadata = dict(segment.metadata)
        reports.setdefault(segment.session_id, []).append(
            (segment, {"report": report, "metadata": metadata})
        )

        # Release heavy tensor fields now that metrics are computed.
        segment.strip_original()
        segment.reconstructed = None  # type: ignore[assignment]
        segment.latents = None
        segment.pad_and_gap_mask = None
        segment.token_start_times = None
        segment.token_end_times = None
        segment.token_timestamps = None
        segment.replica_latents = None
        segment.replica_reconstructions = None
        segment.replica_originals = None

        bar.update(1)

    bar.set_description("Replica eval complete")
    bar.close()

    return reports


__all__ = [
    "iter_collated_validation_segments_replicas",
    "run_validation_evaluation_per_session_replicas",
]
