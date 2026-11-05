"""Utilities for evaluating all neurons in a session.

When a tokenizer is trained with n_neurons < total neurons in a session,
this module provides tools to evaluate the full population:
- compress_t tokenizers: pass all neurons, adjust batch size
- compress_n tokenizers: partition neurons into groups, run each group
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from enum import StrEnum
from typing import Any

import numpy as np
import torch

from ntt.core.components.types import BinnedSpikesInput, RatesInput
from ntt.core.data.components.transforms import session_seed
from ntt.evaluation.utils.reconstruction_type import (
    RECONSTRUCTION_DATA_TYPE_KEY,
    ReconstructionTypeRecorder,
    declared_reconstruction_data_type,
)

logger = logging.getLogger(__name__)


class CompressNLatentStrategy(StrEnum):
    """How full-neuron compress_n eval should assemble latent groups."""

    ALL_GROUPS = "all_groups"
    EXCLUDE_PADDED_GROUP = "exclude_padded_group"


def compute_eval_batch_size(
    config_batch_size: int,
    config_n_neurons: int | None,
    max_neurons: int,
) -> int:
    """Compute adjusted batch size for full-neuron evaluation.

    Scales the batch size inversely with the neuron count to keep
    GPU memory usage roughly constant:
        neuron_budget = config_batch_size * config_n_neurons
        result = max(1, neuron_budget // max_neurons)

    Args:
        config_batch_size: Original batch size from config.
        config_n_neurons: Number of neurons the tokenizer was configured with.
            None means the tokenizer already uses all neurons (no scaling).
        max_neurons: Maximum neuron count across sessions in the dataset.

    Returns:
        Adjusted batch size (minimum 1).
    """
    if config_n_neurons is None:
        return config_batch_size
    neuron_budget = config_batch_size * config_n_neurons
    return max(1, neuron_budget // max_neurons)


def build_neuron_groups(
    n_total: int,
    n_per_group: int,
    seed: int = 42,
) -> tuple[list[list[int]], list[list[bool]]]:
    """Partition neurons into groups for covering evaluation.

    Creates a primary-assignment covering: every neuron appears as a
    primary member of exactly one group. The last group is padded with
    filler neurons (randomly drawn from other neurons) to reach
    n_per_group. Filler reconstructions are discarded by the caller.

    Args:
        n_total: Total number of neurons in the session.
        n_per_group: Number of neurons per group (tokenizer's n_neurons).
        seed: Random seed for reproducibility.

    Returns:
        (groups, primary_masks) where:
        - groups: list of lists, each of length n_per_group, containing
          neuron indices.
        - primary_masks: list of lists of bools, same shape as groups.
          True = primary (keep reconstruction), False = filler (discard).
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total).tolist()

    groups: list[list[int]] = []
    primary_masks: list[list[bool]] = []

    for start in range(0, n_total, n_per_group):
        chunk = perm[start : start + n_per_group]
        n_primary = len(chunk)

        if n_primary < n_per_group:
            # Fill with random neurons from the rest of the population.
            # Prefer neurons not already in chunk; fall back to all neurons
            # when the chunk already covers the whole population.
            n_fillers = n_per_group - n_primary
            filler_pool = [i for i in range(n_total) if i not in chunk]
            if not filler_pool:
                filler_pool = list(range(n_total))
            fillers = rng.choice(
                filler_pool,
                size=n_fillers,
                replace=len(filler_pool) < n_fillers,
            ).tolist()
            chunk = chunk + fillers

        mask = [True] * n_primary + [False] * (n_per_group - n_primary)
        groups.append(chunk)
        primary_masks.append(mask)

    return groups, primary_masks


def select_neurons(
    batch: BinnedSpikesInput | RatesInput,
    sample_idx: int,
    neuron_indices: list[int] | np.ndarray,
) -> BinnedSpikesInput | RatesInput:
    """Create a ``batch_size=1`` sub-batch with selected neuron columns.

    Extracts a single sample from *batch* and keeps only the neurons at
    *neuron_indices*.  Used by :class:`NeuronCoveringWrapper` for tiling and
    by the compress_t eval path for neuron capping.
    """
    is_rates = isinstance(batch, RatesInput)
    data_matrix = batch.rate_matrix if is_rates else batch.spike_matrix
    idx = torch.tensor(neuron_indices, dtype=torch.long)

    sub_data = data_matrix[sample_idx : sample_idx + 1, :, idx]
    sub_mask = batch.pad_and_gap_mask[sample_idx : sample_idx + 1, :, idx]
    sub_unit_ids = batch.unit_ids[sample_idx : sample_idx + 1, idx]
    sub_unit_ids_pad = torch.ones(
        1,
        len(neuron_indices),
        dtype=batch.unit_ids_pad_mask.dtype,
        device=batch.unit_ids_pad_mask.device,
    )

    sid = (
        batch.session_id[sample_idx]
        if isinstance(batch.session_id, (list, tuple))
        else batch.session_id
    )
    subject_id = (
        batch.subject_id[sample_idx]
        if isinstance(batch.subject_id, (list, tuple))
        else batch.subject_id
    )
    species = (
        batch.species[sample_idx] if isinstance(batch.species, (list, tuple)) else batch.species
    )

    if batch.absolute_start is None or batch.absolute_end is None:
        raise ValueError("select_neurons requires absolute_start and absolute_end to be set.")
    abs_start = batch.absolute_start[sample_idx : sample_idx + 1]
    abs_end = batch.absolute_end[sample_idx : sample_idx + 1]
    duration = batch.duration_sec[sample_idx : sample_idx + 1]

    data_field = "rate_matrix" if is_rates else "spike_matrix"
    cls = RatesInput if is_rates else BinnedSpikesInput
    data_kwargs: dict[str, Any] = {data_field: sub_data}

    return cls(
        **data_kwargs,
        pad_and_gap_mask=sub_mask,
        sampling_rate=batch.sampling_rate,
        unit_ids=sub_unit_ids,
        unit_ids_pad_mask=sub_unit_ids_pad,
        duration_sec=duration,
        session_id=(sid,),
        subject_id=(subject_id,),
        species=(species,),
        absolute_start=abs_start,
        absolute_end=abs_end,
        behavioral_data=None,
    )


class NeuronCoveringWrapper:
    """Wraps a compress_n tokenizer to evaluate all neurons via covering.

    For each sample in a batch, partitions the full neuron population into
    groups of ``n_neurons``, runs the inner tokenizer on each group, and
    assembles a full-population reconstruction where each neuron's output
    comes from its primary group assignment.

    The wrapper is transparent to the rest of the eval pipeline -- it
    accepts and returns the same formats as the inner tokenizer_fn.

    Args:
        tokenizer_fn: The inner tokenizer callable. Takes BinnedSpikesInput,
            returns dict with "reconstruction" key.
        n_neurons: Group size (the tokenizer's configured n_neurons).
        seed: Base seed for neuron permutation. Combined with a hash of
            session_id for per-session reproducibility.
        max_neurons: If set, cap the number of neurons per session to this
            value by randomly selecting a subset *before* tokenization. This
            ensures latents and reconstructions reflect only the selected
            neurons. None = no cap (use all neurons).
    """

    def __init__(
        self,
        tokenizer_fn: Callable[[BinnedSpikesInput | RatesInput], Any],
        n_neurons: int,
        seed: int = 42,
        max_neurons: int | None = None,
        latent_strategy: CompressNLatentStrategy = CompressNLatentStrategy.ALL_GROUPS,
    ):
        self._tokenizer_fn = ReconstructionTypeRecorder(tokenizer_fn)
        self._n_neurons = n_neurons
        self._seed = seed
        self._max_neurons = max_neurons
        self._latent_strategy = latent_strategy

    def _extract_latents(self, output: Any) -> torch.Tensor | None:
        """Extract latents tensor from tokenizer output, if present."""
        if isinstance(output, dict):
            latents = output.get("latents")
            if isinstance(latents, torch.Tensor):
                return latents
        return None

    def _extract_token_metadata(self, output: Any) -> dict[str, torch.Tensor]:
        """Extract token timestamp metadata from tokenizer output."""
        if not isinstance(output, dict):
            return {}
        metadata: dict[str, torch.Tensor] = {}
        for key in ("token_timestamps", "token_start_time", "token_end_time"):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                metadata[key] = value
        return metadata

    def _merge_group_latents(self, outputs: list[Any]) -> torch.Tensor | None:
        """Concatenate per-group latents along the unit axis."""
        if not outputs:
            return None

        latents = [self._extract_latents(output) for output in outputs]
        if any(latent is None for latent in latents):
            return None

        latent_tensors = [latent for latent in latents if latent is not None]
        if len(latent_tensors) == 1:
            return latent_tensors[0]

        ndim = latent_tensors[0].ndim
        if any(latent.ndim != ndim for latent in latent_tensors):
            raise ValueError("Cannot concatenate compress_n latents with differing ranks.")
        if ndim == 4:
            return torch.cat(latent_tensors, dim=2)
        if ndim == 3:
            return torch.cat(latent_tensors, dim=1)
        raise ValueError(
            "compress_n latent concatenation expects 3D [B, U, D] or 4D [B, T, U, D] latents."
        )

    def _merge_sample_latents(
        self,
        latents: list[torch.Tensor | None],
    ) -> torch.Tensor | list[torch.Tensor | None] | None:
        """Merge per-sample latents back into a batch output."""
        if all(latent is None for latent in latents):
            return None

        present = [latent for latent in latents if latent is not None]
        if len(present) == len(latents):
            shapes = {tuple(latent.shape) for latent in present}
            if len(shapes) == 1:
                return torch.cat(present, dim=0)

        per_sample: list[torch.Tensor | None] = []
        for latent in latents:
            if latent is None:
                per_sample.append(None)
            else:
                if latent.shape[0] != 1:
                    raise ValueError(
                        "Expected per-sample compress_n latents to have batch dimension 1."
                    )
                per_sample.append(latent[0])
        return per_sample

    def _include_group_latents(self, primary_mask: list[bool]) -> bool:
        """Return whether a group's latents should contribute to eval output."""
        if self._latent_strategy == CompressNLatentStrategy.ALL_GROUPS:
            return True
        return all(primary_mask)

    def __call__(self, batch: BinnedSpikesInput | RatesInput) -> dict[str, Any]:
        is_rates = isinstance(batch, RatesInput)
        data_matrix = batch.rate_matrix if is_rates else batch.spike_matrix
        batch_size = data_matrix.shape[0]
        n_units_padded = data_matrix.shape[2]

        assembled_recons: list[torch.Tensor] = []
        assembled_latents: list[torch.Tensor | None] = []
        token_metadata_per_sample: list[dict[str, torch.Tensor]] = []

        for i in range(batch_size):
            pad_mask = batch.unit_ids_pad_mask[i]
            actual_n = int(pad_mask.sum().item())

            # Apply neuron cap: subsample before tokenization so latents
            # and reconstruction reflect only the selected neurons.
            if self._max_neurons is not None and actual_n > self._max_neurons:
                sid = (
                    batch.session_id[i]
                    if isinstance(batch.session_id, (list, tuple))
                    else batch.session_id
                )
                rng = np.random.default_rng(session_seed(self._seed, str(sid)))
                selected = np.sort(rng.choice(actual_n, size=self._max_neurons, replace=False))
                batch_for_sample = select_neurons(batch, i, selected)
                actual_n = self._max_neurons
                if actual_n <= self._n_neurons:
                    sample_result = self._run_single_group(batch_for_sample, 0, actual_n)
                else:
                    sample_result = self._run_covering(batch_for_sample, 0, actual_n)
            else:
                if actual_n <= self._n_neurons:
                    sample_result = self._run_single_group(batch, i, actual_n)
                else:
                    sample_result = self._run_covering(batch, i, actual_n)

            recon, latents, token_metadata = sample_result
            assembled_recons.append(recon)
            assembled_latents.append(latents)
            token_metadata_per_sample.append(token_metadata)

        # Stack back into batch and pad to n_units_padded
        max_units = n_units_padded
        padded_recons = []
        for recon in assembled_recons:
            n = recon.shape[-1]
            if n < max_units:
                pad = torch.zeros(
                    *recon.shape[:-1],
                    max_units - n,
                    dtype=recon.dtype,
                    device=recon.device,
                )
                recon = torch.cat([recon, pad], dim=-1)
            padded_recons.append(recon)

        stacked = torch.cat(padded_recons, dim=0)
        result: dict[str, Any] = {
            "reconstruction": stacked,
            "latents": self._merge_sample_latents(assembled_latents),
        }
        if self._tokenizer_fn.declared is not None:
            result[RECONSTRUCTION_DATA_TYPE_KEY] = self._tokenizer_fn.declared
        for key in ("token_timestamps", "token_start_time", "token_end_time"):
            values = [metadata.get(key) for metadata in token_metadata_per_sample]
            if all(value is not None for value in values):
                result[key] = torch.cat([value for value in values if value is not None], dim=0)
        return result

    def _extract_reconstruction(self, output: Any) -> torch.Tensor:
        """Extract reconstruction tensor from tokenizer output."""
        if isinstance(output, dict):
            return output["reconstruction"]
        if hasattr(output, "reconstruction"):
            return output.reconstruction
        return output

    def _run_single_group(
        self,
        batch: BinnedSpikesInput | RatesInput,
        sample_idx: int,
        actual_n: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
        """Run tokenizer on a single group (session fits in n_neurons)."""
        indices = list(range(actual_n))
        if actual_n < self._n_neurons:
            rng = np.random.default_rng(self._seed)
            fillers = rng.choice(actual_n, size=self._n_neurons - actual_n, replace=True).tolist()
            indices = indices + fillers

        sub_batch = select_neurons(batch, sample_idx, indices)
        output = self._tokenizer_fn(sub_batch)
        recon = self._extract_reconstruction(output)
        if actual_n < self._n_neurons and (
            self._latent_strategy == CompressNLatentStrategy.EXCLUDE_PADDED_GROUP
        ):
            latents = None
        else:
            latents = self._extract_latents(output)
        return recon[0:1, :, :actual_n], latents, self._extract_token_metadata(output)

    def _run_covering(
        self,
        batch: BinnedSpikesInput | RatesInput,
        sample_idx: int,
        actual_n: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
        """Run tokenizer on multiple neuron groups and assemble."""
        session_id = (
            batch.session_id[sample_idx]
            if isinstance(batch.session_id, (list, tuple))
            else batch.session_id
        )
        seed = session_seed(self._seed, str(session_id))
        groups, primary_masks = build_neuron_groups(actual_n, self._n_neurons, seed=seed)

        first_sub = select_neurons(batch, sample_idx, groups[0])
        first_output = self._tokenizer_fn(first_sub)
        first_recon = self._extract_reconstruction(first_output)
        included_latent_outputs: list[Any] = []
        if self._include_group_latents(primary_masks[0]):
            included_latent_outputs.append(first_output)

        # Use reconstruction time dimension (may differ from input for
        # tokenizers with compress_t=True that change temporal resolution)
        recon_time_bins = first_recon.shape[1]

        full_recon = torch.zeros(
            recon_time_bins,
            actual_n,
            dtype=first_recon.dtype,
            device=first_recon.device,
        )

        for j, (neuron_idx, is_primary) in enumerate(
            zip(groups[0], primary_masks[0], strict=True)
        ):
            if is_primary:
                full_recon[:, neuron_idx] = first_recon[0, :, j]

        for group, mask in zip(groups[1:], primary_masks[1:], strict=True):
            sub_batch = select_neurons(batch, sample_idx, group)
            output = self._tokenizer_fn(sub_batch)
            recon = self._extract_reconstruction(output)
            if self._include_group_latents(mask):
                included_latent_outputs.append(output)

            for j, (neuron_idx, is_primary) in enumerate(zip(group, mask, strict=True)):
                if is_primary:
                    full_recon[:, neuron_idx] = recon[0, :, j]

        return (
            full_recon.unsqueeze(0),
            self._merge_group_latents(included_latent_outputs),
            self._extract_token_metadata(first_output),
        )


def _slice_batch(
    batch: BinnedSpikesInput | RatesInput,
    start: int,
    end: int,
) -> BinnedSpikesInput | RatesInput:
    """Slice a batch along the batch dimension."""
    is_rates = isinstance(batch, RatesInput)
    data_matrix = batch.rate_matrix if is_rates else batch.spike_matrix
    data_field = "rate_matrix" if is_rates else "spike_matrix"
    cls = type(batch)

    session_id = batch.session_id
    if isinstance(session_id, (list, tuple)):
        session_id = session_id[start:end]
    subject_id = batch.subject_id
    if isinstance(subject_id, (list, tuple)):
        subject_id = subject_id[start:end]
    species = batch.species
    if isinstance(species, (list, tuple)):
        species = species[start:end]

    # Collated batches may carry a per-sample list here, despite the dict annotation.
    batch_behavioral_data: Any = batch.behavioral_data
    behavioral_data: Any = None
    if batch_behavioral_data is not None and isinstance(batch_behavioral_data, list):
        behavioral_data = batch_behavioral_data[start:end]

    if batch.absolute_start is None or batch.absolute_end is None:
        raise ValueError("_slice_batch requires absolute_start and absolute_end to be set.")
    data_kwargs: dict[str, Any] = {data_field: data_matrix[start:end]}

    return cls(
        **data_kwargs,
        pad_and_gap_mask=batch.pad_and_gap_mask[start:end],
        sampling_rate=batch.sampling_rate,
        unit_ids=batch.unit_ids[start:end],
        unit_ids_pad_mask=batch.unit_ids_pad_mask[start:end],
        duration_sec=batch.duration_sec[start:end],
        session_id=session_id,
        subject_id=subject_id,
        species=species,
        absolute_start=batch.absolute_start[start:end],
        absolute_end=batch.absolute_end[start:end],
        behavioral_data=behavioral_data,
    )


def _as_output_dict(output: Any) -> dict[str, Any]:
    """A tokenizer output as a dict, keeping a ReconstructionOutput's declared type."""
    if isinstance(output, dict):
        return output
    if not hasattr(output, "reconstruction"):
        return {"reconstruction": output}
    result: dict[str, Any] = {"reconstruction": output.reconstruction}
    declared = declared_reconstruction_data_type(output)
    if declared is not None:
        result[RECONSTRUCTION_DATA_TYPE_KEY] = declared
    return result


def _concat_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Concatenate a list of tokenizer output dicts along batch dim."""
    recons = [r["reconstruction"] for r in results]
    merged: dict[str, Any] = {"reconstruction": torch.cat(recons, dim=0)}

    if all(r.get("latents") is not None for r in results):
        merged["latents"] = torch.cat([r["latents"] for r in results], dim=0)
    else:
        merged["latents"] = None

    for key in ("token_timestamps", "token_start_time", "token_end_time"):
        if all(key in r for r in results):
            merged[key] = torch.cat([r[key] for r in results], dim=0)

    declared = {declared_reconstruction_data_type(r) for r in results} - {None}
    if len(declared) > 1:
        raise ValueError(
            f"Merged tokenizer outputs declare different {RECONSTRUCTION_DATA_TYPE_KEY}: "
            f"{declared}"
        )
    if declared:
        merged[RECONSTRUCTION_DATA_TYPE_KEY] = declared.pop()
    return merged


def oom_safe_forward(
    fn: Callable,
    batch: BinnedSpikesInput | RatesInput,
    batch_size: int,
) -> tuple[dict[str, Any], int]:
    """Run tokenizer forward pass with OOM fallback.

    Tries the full batch first. On OOM, halves the effective batch size
    and processes in chunks. Repeats until success or batch_size=1 fails.

    Args:
        fn: Tokenizer callable.
        batch: Full input batch.
        batch_size: Current effective batch size.

    Returns:
        (result_dict, final_batch_size) -- the merged result and the batch
        size that succeeded (for logging / future calls).

    Raises:
        torch.cuda.OutOfMemoryError: If batch_size=1 still OOMs.
    """
    actual_B = (batch.rate_matrix if isinstance(batch, RatesInput) else batch.spike_matrix).shape[
        0
    ]
    effective_bs = min(batch_size, actual_B)

    while True:
        try:
            if effective_bs >= actual_B:
                return _as_output_dict(fn(batch)), effective_bs

            # Process in chunks
            chunks: list[dict[str, Any]] = []
            for start in range(0, actual_B, effective_bs):
                end = min(start + effective_bs, actual_B)
                sub = _slice_batch(batch, start, end)
                chunks.append(_as_output_dict(fn(sub)))
            return _concat_results(chunks), effective_bs

        except torch.cuda.OutOfMemoryError:
            if effective_bs <= 1:
                raise
            new_bs = max(1, effective_bs // 2)
            logger.warning(
                "OOM with batch_size=%d, retrying with batch_size=%d",
                effective_bs,
                new_bs,
            )
            torch.cuda.empty_cache()
            effective_bs = new_bs


def make_oom_safe_tokenize_fn(
    fn: Callable[[BinnedSpikesInput | RatesInput], Any],
) -> Callable[[BinnedSpikesInput | RatesInput], Any]:
    """Wrap a tokenizer callable so a batch that OOMs is split, not fatal.

    A batch that fits runs as a plain ``fn(batch)`` and its output is returned
    unchanged, so results are identical wherever the unwrapped call succeeds.
    On the first OOM the batch falls back to :func:`oom_safe_forward`, and the
    batch size that fit is reused for later batches, so each batch does not
    OOM again at full size first.
    """
    working_batch_size: int | None = None  # set after the first OOM

    def _forward(batch: BinnedSpikesInput | RatesInput) -> Any:
        nonlocal working_batch_size
        if working_batch_size is None:
            try:
                return fn(batch)
            except torch.cuda.OutOfMemoryError:
                data = batch.rate_matrix if isinstance(batch, RatesInput) else batch.spike_matrix
                if data.shape[0] <= 1:
                    raise
                working_batch_size = max(1, data.shape[0] // 2)
                logger.warning(
                    "OOM with batch_size=%d, retrying with batch_size=%d",
                    data.shape[0],
                    working_batch_size,
                )
            # Retry outside the except block so the failed attempt's tensors are freed.
            torch.cuda.empty_cache()
        result, working_batch_size = oom_safe_forward(fn, batch, batch_size=working_batch_size)
        return result

    return _forward
