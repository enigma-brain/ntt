"""Reconstruction visualization plotting functions.

Reusable functions for visualizing tokenizer reconstructions: rasters,
firing rate comparisons, and autocorrelograms. All functions accept a
``CollatedValidationSegment`` (from the validation collation utilities)
and a ``bin_size_seconds`` parameter, and return a ``matplotlib.Figure``.

Example::

    from ntt.evaluation.plotting import (
        get_validation_segment,
        plot_per_unit_raster,
        plot_firing_rate_comparison,
        plot_acg_comparison,
        plot_population_raster,
    )

    # Default: uses spec.n_neurons neurons
    segment, bin_s = get_validation_segment(tokenizer, datamodule, seconds=600)

    # Or request a specific number of neurons (e.g. all 80 from a session)
    segment, bin_s = get_validation_segment(
        tokenizer, datamodule, seconds=600, n_neurons=80, session_id="brainset/session"
    )

    fig = plot_per_unit_raster(segment, bin_s)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure
from scipy.stats import linregress

from ntt.core.components.types import (
    BinnedSpikesInput,
    DataType,
    RatesInput,
    ReconstructionOutput,
)
from ntt.evaluation.analysis import compute_acgs
from ntt.evaluation.core.types import EvalData
from ntt.evaluation.utils import (
    CollatedValidationSegment,
    iter_collated_validation_segments,
)

# ---------------------------------------------------------------------------
# Default colours
# ---------------------------------------------------------------------------

_COLOR_ORIGINAL = "k"
_COLOR_RECONSTRUCTED = "tab:red"

# ---------------------------------------------------------------------------
# Display mode
# ---------------------------------------------------------------------------


class DisplayMode(Enum):
    """How to render neural data in plots."""

    BINARY_SPIKES = "binary_spikes"  # eventplot rasters
    RATES = "rates"  # heatmap / line plots


def _detect_display_mode(
    data: np.ndarray,
    *,
    explicit: DisplayMode | None = None,
    bin_size_seconds: float | None = None,
) -> DisplayMode:
    """Infer whether data looks like binary spikes or continuous rates.

    Uses the bin size as a strong prior: at fine resolution (<=1 ms) we expect
    binary {0, 1} spikes; at coarser bins (rates regime) we only classify as
    binary if values are *strictly* in {0, 1}, since integer counts > 1 are
    normal at larger bin sizes.
    """
    if explicit is not None:
        return explicit

    is_fine_bin = bin_size_seconds is not None and bin_size_seconds <= 0.001

    if is_fine_bin:
        # At 1 ms bins: integer-valued data with occasional multi-spikes is
        # still best shown as an eventplot raster.
        if np.issubdtype(data.dtype, np.integer) or (
            np.allclose(data, data.astype(int)) and data.max() <= 2
        ):
            return DisplayMode.BINARY_SPIKES
        return DisplayMode.RATES

    # Coarser bins (rates regime): only binary if values are literally 0/1.
    unique = np.unique(data)
    if len(unique) <= 2 and set(unique.tolist()).issubset({0, 1}):
        return DisplayMode.BINARY_SPIKES
    return DisplayMode.RATES


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


@dataclass
class PlotMetadata:
    """Structured metadata for plot annotations."""

    session_id: str
    brainset_name: str
    session_name: str
    subject_id: str | None
    species: str | None
    bin_size_seconds: float
    n_units: int
    n_time_bins: int
    duration_seconds: float
    unit_ids: list[int] | None


def _extract_plot_metadata(
    segment: CollatedValidationSegment,
    bin_size_seconds: float,
) -> PlotMetadata:
    """Build structured metadata from a collated segment."""
    sid = segment.session_id
    if "/" in sid:
        brainset_name, session_name = sid.split("/", 1)
    else:
        brainset_name, session_name = "unknown", sid

    n_units, n_time_bins = _require_tensor(segment.original).shape
    duration_seconds = n_time_bins * bin_size_seconds

    meta = segment.metadata
    unit_ids = meta.get("unit_ids")
    # Validate length matches
    if unit_ids is not None and len(unit_ids) != n_units:
        unit_ids = None

    return PlotMetadata(
        session_id=sid,
        brainset_name=brainset_name,
        session_name=session_name,
        subject_id=meta.get("subject_id"),
        species=meta.get("species"),
        bin_size_seconds=bin_size_seconds,
        n_units=n_units,
        n_time_bins=n_time_bins,
        duration_seconds=duration_seconds,
        unit_ids=unit_ids,
    )


def _unit_label(meta: PlotMetadata, idx: int) -> str:
    """Return a human-readable label for a unit index."""
    if meta.unit_ids is not None:
        return f"Unit {meta.unit_ids[idx]}"
    return f"Unit {idx}"


def _suptitle_text(meta: PlotMetadata) -> str:
    """Build a standard suptitle string with session/brainset info."""
    parts = [meta.brainset_name, meta.session_name]
    return " / ".join(parts)


def _subtitle_text(meta: PlotMetadata) -> str:
    """Build a subtitle with technical details."""
    parts = [
        f"{meta.n_units} units",
        f"{meta.duration_seconds:.1f}s",
        f"{meta.bin_size_seconds * 1000:.1f}ms bins",
    ]
    if meta.species:
        parts.append(str(meta.species))
    return " | ".join(parts)


def _require_tensor(t: torch.Tensor | None) -> torch.Tensor:
    """Narrow an optional segment tensor (original/reconstructed) to a tensor."""
    if t is None:
        raise ValueError("Segment is missing original/reconstructed data.")
    return t


def _to_numpy(t: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert a tensor or array to numpy."""
    if isinstance(t, torch.Tensor):
        return t.cpu().numpy()
    return t


# ---------------------------------------------------------------------------
# Unit / time selection helpers
# ---------------------------------------------------------------------------


def _select_units_by_rate_quantile(
    data: np.ndarray,
    n_units: int,
    *,
    quantiles: tuple[float, ...] = (0.25, 0.5, 0.75),
) -> tuple[np.ndarray, list[str]]:
    """Select units from different firing rate quantiles.

    Returns (indices, labels) where labels describe the quantile band.
    """
    n_total = data.shape[0]
    if n_units >= n_total:
        labels = ["" for _ in range(n_total)]
        return np.arange(n_total), labels

    rates = data.sum(axis=1)
    sorted_idx = np.argsort(rates)

    n_quantiles = len(quantiles)
    per_q = n_units // n_quantiles
    remainder = n_units - per_q * n_quantiles

    selected = []
    labels = []
    quantile_names = {0.25: "low", 0.5: "mid", 0.75: "high"}

    for i, q in enumerate(quantiles):
        count = per_q + (1 if i < remainder else 0)
        center = int(q * (n_total - 1))
        start = max(0, center - count // 2)
        start = min(start, n_total - count)
        indices = sorted_idx[start : start + count]
        q_name = quantile_names.get(q, f"q{q:.0%}")
        for idx in indices:
            selected.append(idx)
            labels.append(q_name)

    return np.array(selected), labels


def _select_time_window_by_activity(
    data: np.ndarray,
    window_bins: int,
    *,
    quantile: float = 0.5,
    smoothing_bins: int = 100,
) -> tuple[int, int]:
    """Find a time window where population activity matches the requested quantile."""
    n_bins = data.shape[1]
    if window_bins >= n_bins:
        return 0, n_bins

    pop_rate = data.sum(axis=0).astype(float)
    if smoothing_bins > 1:
        kernel = np.ones(smoothing_bins) / smoothing_bins
        pop_rate = np.convolve(pop_rate, kernel, mode="same")

    # Compute windowed sums efficiently via cumsum
    cs = np.concatenate([[0], np.cumsum(pop_rate)])
    windowed = cs[window_bins:] - cs[: len(cs) - window_bins]

    target = np.percentile(windowed, quantile * 100)
    best_start = int(np.argmin(np.abs(windowed - target)))
    return best_start, best_start + window_bins


def _compute_grid_layout(n_items: int, max_cols: int = 5) -> tuple[int, int]:
    """Compute a grid layout for subplots."""
    ncols = min(n_items, max_cols)
    nrows = math.ceil(n_items / ncols)
    return nrows, ncols


# ---------------------------------------------------------------------------
# High-level data helper
# ---------------------------------------------------------------------------


class _FilteredSampler:
    """Minimal sampler wrapping a filtered list of DatasetIndex objects.

    Exposes ``_indices`` so that ``iter_collated_validation_segments``
    can use it for overlap detection.
    """

    def __init__(self, indices: list) -> None:
        self._indices = indices

    def __len__(self) -> int:
        return len(self._indices)

    def __iter__(self):
        yield from self._indices


def _build_filtered_dataloader(
    datamodule: Any,
    *,
    session_id: str | None = None,
    brainset: str | None = None,
) -> Any:
    """Build a DataLoader that only yields batches for matching sessions.

    Filters the sampler indices *before* iteration so no unnecessary
    computation or collation is performed for non-matching sessions.
    """
    from torch.utils.data import DataLoader

    from ntt.core.data.datamodule import collate

    dataset = datamodule.val_dataset
    sampler = datamodule.val_sampler

    # Get all sampler indices (DatasetIndex with recording_id, start, end)
    all_indices = sampler._indices

    # Filter by session_id / brainset prefix
    if session_id is not None or brainset is not None:
        filtered = []
        for idx in all_indices:
            rid = idx.recording_id
            if session_id is not None and rid != session_id:
                continue
            if brainset is not None and not rid.startswith(brainset):
                continue
            filtered.append(idx)

        if not filtered:
            available = sorted({idx.recording_id for idx in all_indices})
            filters = []
            if session_id:
                filters.append(f"session_id={session_id!r}")
            if brainset:
                filters.append(f"brainset={brainset!r}")
            raise ValueError(
                f"No matching sessions for {', '.join(filters)}. Available: {available}"
            )
        new_sampler = _FilteredSampler(filtered)
    else:
        new_sampler = sampler

    dl_config = datamodule.config.dataloader.model_dump()
    dl_config = {k: v for k, v in dl_config.items() if "hydra" not in k}
    return DataLoader(
        dataset=dataset,
        sampler=new_sampler,
        collate_fn=collate,
        **dl_config,
    )


# ---------------------------------------------------------------------------
# All-neurons data loading (for custom n_neurons)
# ---------------------------------------------------------------------------


class _DatasetWithTransform:
    """Wrapper that applies a different transform pipeline to an existing dataset.

    Avoids mutating the original dataset's transform attribute, which would
    be unsafe when DataLoader workers fork the dataset.
    """

    def __init__(self, dataset: Any, transform: Any) -> None:
        self._dataset = dataset
        self._transform = transform

    def __getitem__(self, index: Any) -> Any:
        sample = self._dataset.get(index.recording_id, index.start, index.end)
        if self._transform is not None:
            sample = self._transform(sample)
        return sample

    def __len__(self) -> int:
        return len(self._dataset)


class _SelectFirstNUnits:
    """Deterministically select the first *n* units from a Data sample.

    Unlike :class:`UnitDropout` which randomly selects units per window,
    this always takes the first *n* units.  This guarantees the same
    neurons appear in every window, which is required for temporally
    consistent collated segments.

    If the sample has fewer than *n* units, it is returned unchanged.
    """

    def __init__(self, n: int) -> None:
        self._n = n

    def __call__(self, data: Any) -> Any:
        num_units = len(data.units)
        if num_units <= self._n:
            return data

        unit_mask = np.zeros(num_units, dtype=bool)
        unit_mask[: self._n] = True

        # Filter spikes to only those from selected units
        spikes_mask = unit_mask[data.spikes.unit_index]
        data.spikes = data.spikes.select_by_mask(spikes_mask)

        # Filter units and re-index spike unit_index
        data.units = data.units.select_by_mask(unit_mask)
        relabel = np.zeros(num_units, dtype=int)
        relabel[unit_mask] = np.arange(self._n)
        data.spikes.unit_index = relabel[data.spikes.unit_index]

        return data


def _build_all_neurons_transform(spec: Any, *, n_neurons: int | None = None) -> Any:
    """Build a transform pipeline for custom neuron counts.

    Replaces UnitDropout (random per-window) with an optional deterministic
    selection of the first *n_neurons* units, then bins and converts to
    model input format.  PadNeuronsToFixed is omitted — the chunked
    reconstruction handles padding internally.
    """
    from torch_brain.transforms import Compose

    from ntt.core.data.components.transforms import (
        CreateBinnedSpikeCountsArray,
        DataToModelInput,
    )

    transforms: list = []
    if n_neurons is not None:
        transforms.append(_SelectFirstNUnits(n_neurons))
    if spec.bin_size_ms is not False:
        transforms.append(CreateBinnedSpikeCountsArray(bin_size_ms=spec.bin_size_ms))
    transforms.append(DataToModelInput(mode=spec.input_type))
    return Compose(transforms)


def _build_all_neurons_dataloader(
    datamodule: Any,
    transform: Any,
    *,
    session_id: str | None = None,
    brainset: str | None = None,
    batch_size: int | None = None,
) -> Any:
    """Build a DataLoader serving neurons for matching sessions.

    When *batch_size* is ``None``, defaults to 1 (safe for variable neuron
    counts).  Pass a larger value when a deterministic neuron-selection
    transform guarantees a fixed count across windows.
    """
    from torch.utils.data import DataLoader

    from ntt.core.data.datamodule import collate

    dataset = datamodule.val_dataset
    sampler = datamodule.val_sampler

    wrapped = _DatasetWithTransform(dataset, transform)

    # Filter sampler (same logic as _build_filtered_dataloader)
    all_indices = sampler._indices
    if session_id is not None or brainset is not None:
        filtered = []
        for idx in all_indices:
            rid = idx.recording_id
            if session_id is not None and rid != session_id:
                continue
            if brainset is not None and not rid.startswith(brainset):
                continue
            filtered.append(idx)

        if not filtered:
            available = sorted({idx.recording_id for idx in all_indices})
            filters = []
            if session_id:
                filters.append(f"session_id={session_id!r}")
            if brainset:
                filters.append(f"brainset={brainset!r}")
            raise ValueError(
                f"No matching sessions for {', '.join(filters)}. Available: {available}"
            )
        new_sampler = _FilteredSampler(filtered)
    else:
        new_sampler = sampler

    dl_config = datamodule.config.dataloader.model_dump()
    dl_config = {k: v for k, v in dl_config.items() if "hydra" not in k}
    dl_config["batch_size"] = batch_size if batch_size is not None else 1
    return DataLoader(
        dataset=cast(Any, wrapped),
        sampler=new_sampler,
        collate_fn=collate,
        **dl_config,
    )


# ---------------------------------------------------------------------------
# Chunked reconstruction (neuron dimension)
# ---------------------------------------------------------------------------


def _get_data_matrix(batch: BinnedSpikesInput | RatesInput) -> torch.Tensor:
    """Extract the (B, T, N) data matrix from either input type."""
    if isinstance(batch, RatesInput):
        return batch.rate_matrix
    return batch.spike_matrix


def _rebuild_input_batch(
    batch: BinnedSpikesInput | RatesInput,
    stacked_sm: torch.Tensor,
    stacked_pm: torch.Tensor,
    stacked_uid: torch.Tensor,
    stacked_uid_pm: torch.Tensor,
    dur: torch.Tensor | float,
    n_sub: int,
) -> BinnedSpikesInput | RatesInput:
    """Construct a new input batch of the same type as *batch*."""

    def _rep(field: Any, n: int) -> Any:
        if isinstance(field, str):
            return tuple([field] * n)
        if isinstance(field, (list, tuple)):
            return tuple(field) * n
        return field

    common = {
        "pad_and_gap_mask": stacked_pm,
        "sampling_rate": batch.sampling_rate,
        "unit_ids": stacked_uid,
        "unit_ids_pad_mask": stacked_uid_pm,
        "duration_sec": dur,
        "session_id": _rep(batch.session_id, n_sub),
        "subject_id": _rep(batch.subject_id, n_sub),
        "species": _rep(batch.species, n_sub),
        "absolute_start": batch.absolute_start,
        "absolute_end": batch.absolute_end,
    }
    if isinstance(batch, RatesInput):
        return RatesInput(rate_matrix=stacked_sm, **common)
    return BinnedSpikesInput(spike_matrix=stacked_sm, **common)


def _chunked_reconstruct(
    tokenizer: torch.nn.Module,
    batch: BinnedSpikesInput | RatesInput,
    chunk_size: int,
    *,
    max_batch_chunks: int = 32,
) -> ReconstructionOutput:
    """Reconstruct a batch by chunking along the neuron dimension.

    Neurons are split into groups of *chunk_size*, stacked along the batch
    dimension, and processed through the tokenizer.  The last chunk is
    zero-padded if the total neuron count is not an exact multiple of
    *chunk_size*.  After reconstruction the padding is trimmed and chunks
    are concatenated back to the original neuron ordering.

    When the number of neuron-chunks exceeds *max_batch_chunks*, chunks
    are processed in sub-batches to avoid GPU OOM.
    """
    sm = _get_data_matrix(batch)  # (B, T, N)
    B, T, N = sm.shape
    n_chunks = math.ceil(N / chunk_size)

    # ── Build per-chunk tensors ──────────────────────────────────────
    chunk_mats: list[torch.Tensor] = []
    chunk_pms: list[torch.Tensor] = []
    chunk_uids: list[torch.Tensor] = []
    chunk_uid_pms: list[torch.Tensor] = []
    actual_sizes: list[int] = []

    for i in range(n_chunks):
        s, e = i * chunk_size, min((i + 1) * chunk_size, N)
        actual = e - s
        actual_sizes.append(actual)

        # Spike matrix
        mat = sm[:, :, s:e]
        if actual < chunk_size:
            padded = torch.zeros(B, T, chunk_size, dtype=mat.dtype, device=mat.device)
            padded[:, :, :actual] = mat
            mat = padded
        chunk_mats.append(mat)

        # Pad mask
        if batch.pad_and_gap_mask is not None:
            pm = batch.pad_and_gap_mask[:, :, s:e]
        else:
            pm = torch.ones(B, T, actual, dtype=torch.int32, device=sm.device)
        if actual < chunk_size:
            padded_pm = torch.zeros(B, T, chunk_size, dtype=pm.dtype, device=pm.device)
            padded_pm[:, :, :actual] = pm
            pm = padded_pm
        chunk_pms.append(pm)

        # Unit IDs
        uid = batch.unit_ids
        uid_pm = batch.unit_ids_pad_mask
        if uid.dim() > 1:
            u, m = uid[:, s:e], uid_pm[:, s:e]
        else:
            u, m = uid[s:e].unsqueeze(0), uid_pm[s:e].unsqueeze(0)
        if actual < chunk_size:
            padded_u = torch.zeros(u.shape[0], chunk_size, dtype=u.dtype, device=u.device)
            padded_u[:, :actual] = u
            u = padded_u
            padded_m = torch.zeros(m.shape[0], chunk_size, dtype=m.dtype, device=m.device)
            padded_m[:, :actual] = m
            m = padded_m
        chunk_uids.append(u)
        chunk_uid_pms.append(m)

    # ── Process chunks (sub-batched if needed) ───────────────────────

    recon_parts: list[torch.Tensor] = []
    output_meta: ReconstructionOutput | None = None

    for start in range(0, n_chunks, max_batch_chunks):
        end = min(start + max_batch_chunks, n_chunks)
        n_sub = end - start

        stacked_sm = torch.cat(chunk_mats[start:end], dim=0)
        stacked_pm = torch.cat(chunk_pms[start:end], dim=0)
        stacked_uid = torch.cat(chunk_uids[start:end], dim=0)
        stacked_uid_pm = torch.cat(chunk_uid_pms[start:end], dim=0)

        dur = batch.duration_sec
        if isinstance(dur, torch.Tensor):
            dur = dur.repeat(n_sub)

        sub_batch = _rebuild_input_batch(
            batch, stacked_sm, stacked_pm, stacked_uid, stacked_uid_pm, dur, n_sub
        )

        output = cast(Any, tokenizer).reconstruct_from_input(sub_batch)
        recon = output.reconstruction  # (n_sub*B, T, chunk_size)

        # Split back into per-chunk tensors and trim padding
        sub_chunks = recon.split(B, dim=0)
        for j, rc in enumerate(sub_chunks):
            recon_parts.append(rc[:, :, : actual_sizes[start + j]])

        if output_meta is None:
            output_meta = output

    result = torch.cat(recon_parts, dim=2)  # (B, T, N)

    assert output_meta is not None
    return ReconstructionOutput(
        reconstruction=result,
        data_type=output_meta.data_type,
        bin_size_seconds=output_meta.bin_size_seconds,
    )


def _make_chunked_reconstruct_fn(
    tokenizer: torch.nn.Module,
) -> Any:
    """Wrap a tokenizer to handle arbitrary neuron counts via chunking.

    Returns a callable with the same signature as
    ``tokenizer.reconstruct_from_input``, suitable for passing to
    :func:`iter_collated_validation_segments`.

    - If the tokenizer has ``n_neurons=False`` (variable-n), the callable
      passes the batch through directly — no chunking.
    - Otherwise, batches whose neuron dimension differs from
      ``spec.n_neurons`` are chunked automatically.
    """
    tokenizer_any: Any = tokenizer
    spec = tokenizer_any.spec

    if spec.n_neurons is False:
        return tokenizer_any.reconstruct_from_input

    if not isinstance(spec.n_neurons, int):
        raise TypeError(
            f"n_neurons override requires spec.n_neurons to be int or False, "
            f"got {type(spec.n_neurons).__name__}"
        )

    chunk_size = spec.n_neurons

    def _fn(batch: BinnedSpikesInput | RatesInput) -> ReconstructionOutput:
        n_in = _get_data_matrix(batch).shape[-1]
        if n_in == chunk_size:
            return tokenizer_any.reconstruct_from_input(batch)
        return _chunked_reconstruct(tokenizer, batch, chunk_size)

    return _fn


def _trim_segment_neurons(
    seg: CollatedValidationSegment,
    n_neurons: int,
) -> CollatedValidationSegment:
    """Trim a collated segment to keep only the first *n_neurons* neurons."""
    unit_ids = seg.metadata.get("unit_ids")
    if unit_ids is not None:
        unit_ids = unit_ids[:n_neurons]
    return CollatedValidationSegment(
        session_id=seg.session_id,
        original=_require_tensor(seg.original)[:n_neurons],
        reconstructed=_require_tensor(seg.reconstructed)[:n_neurons],
        pad_and_gap_mask=seg.pad_and_gap_mask[:n_neurons]
        if seg.pad_and_gap_mask is not None
        else None,
        metadata={**seg.metadata, "unit_ids": unit_ids},
        segment_index=seg.segment_index,
        is_partial=seg.is_partial,
        latents=seg.latents,
    )


def get_validation_segment(
    tokenizer: torch.nn.Module,
    datamodule: Any,
    *,
    seconds: float = 600.0,
    n_neurons: int | None = None,
    session_id: str | None = None,
    brainset: str | None = None,
    device: torch.device | str | None = None,
) -> tuple[CollatedValidationSegment, float]:
    """Get a collated validation segment ready for plotting.

    This is a convenience wrapper around ``iter_collated_validation_segments``
    that handles setup, window count calculation, and optional session/brainset
    filtering.  When a ``session_id`` or ``brainset`` filter is provided, the
    dataloader sampler is filtered *before* iteration — no unnecessary
    reconstruction or collation is performed for non-matching sessions.

    Parameters
    ----------
    tokenizer:
        A trained tokenizer module. Must have a ``spec`` attribute with
        ``window_length_seconds`` and ``bin_size_ms``.
    datamodule:
        An NTTDataModule instance. Will be set up for validation if needed.
    seconds:
        How many seconds of data to collate.
    n_neurons:
        Number of neurons to include in the segment. When set, the full
        neuron population is loaded (bypassing the datamodule's UnitDropout)
        and the tokenizer is run in chunks of ``spec.n_neurons`` neurons.

        - For **single-neuron tokenizers** (``spec.n_neurons=1``), all
          requested neurons are batched together in a single tokenizer call.
        - For **population tokenizers** (``spec.n_neurons=K``), neurons are
          processed in groups of K.  The last group is zero-padded if the
          count is not an exact multiple of K; padding is trimmed from the
          reconstruction output.
        - If the session has fewer neurons than requested, all available
          neurons are returned (no synthetic padding in the output).

        ``None`` (default) uses the existing dataloader which serves
        ``spec.n_neurons`` neurons per batch.
    session_id:
        If given, only collate data from this exact session ID.
    brainset:
        If given, only collate data from sessions whose ID starts with
        this brainset name prefix.
    device:
        Device for inference.

    Returns
    -------
    (segment, bin_size_seconds)
        The collated segment and the bin size in seconds.

    Raises
    ------
    ValueError
        If no matching session is found in the datamodule.
    """
    tokenizer_any: Any = tokenizer
    spec = tokenizer_any.spec
    # Resolve to a concrete float — variable_t specs (e.g. CALM) expose a Dist
    # here, which would crash the divide. effective_window_length_seconds
    # returns Dist.max for distributions, the value itself for floats.
    # The datamodule's window is the eval window (it need not be the tokenizer's own).
    window_seconds = datamodule.config.window_length_seconds
    if not isinstance(window_seconds, int | float):
        window_seconds = spec.effective_window_length_seconds
    max_windows = int(seconds / window_seconds)
    bin_size_seconds = spec.bin_size_ms / 1000.0

    if device is not None and isinstance(device, str):
        device = torch.device(device)

    # Ensure datamodule is set up
    if datamodule.val_dataset is None:
        datamodule.setup("validate")

    if n_neurons is not None:
        # All-neurons path: bypass UnitDropout, chunk through tokenizer
        transform = _build_all_neurons_transform(spec, n_neurons=n_neurons)

        # Scale batch_size inversely with neuron-chunk count so total
        # items per tokenizer call stays bounded (avoids OOM for large
        # n_neurons with memory-heavy tokenizers like FISTA).
        default_bs = datamodule.config.dataloader.batch_size
        chunk_size = spec.n_neurons if isinstance(spec.n_neurons, int) else n_neurons
        n_chunks = math.ceil(n_neurons / chunk_size)
        batch_size = max(1, default_bs // n_chunks)

        val_loader = _build_all_neurons_dataloader(
            datamodule,
            transform,
            session_id=session_id,
            brainset=brainset,
            batch_size=batch_size,
        )
        reconstruct_fn = _make_chunked_reconstruct_fn(tokenizer)

        for seg in iter_collated_validation_segments(
            val_loader,
            reconstruct_fn,
            device=device,
            max_windows=max_windows,
        ):
            if _require_tensor(seg.original).shape[0] > n_neurons:
                seg = _trim_segment_neurons(seg, n_neurons)
            return seg, bin_size_seconds

        raise ValueError("Collation produced no segments.")

    # Default path: use existing filtered dataloader (spec.n_neurons)
    val_loader = _build_filtered_dataloader(
        datamodule,
        session_id=session_id,
        brainset=brainset,
    )

    from ntt.evaluation.utils.window_chunking import chunked_tokenizer_fn

    # A rigid tokenizer shorter than the eval window is fed its window in chunks.
    for seg in iter_collated_validation_segments(
        val_loader,
        chunked_tokenizer_fn(tokenizer_any.reconstruct_from_input, spec=spec),
        device=device,
        max_windows=max_windows,
    ):
        return seg, bin_size_seconds

    # Should not normally get here (filtering raises ValueError above),
    # but just in case the collation yields nothing.
    raise ValueError("Collation produced no segments.")


# ---------------------------------------------------------------------------
# Plot 1: Per-unit rasters
# ---------------------------------------------------------------------------


def plot_per_unit_raster(
    segment: CollatedValidationSegment,
    bin_size_seconds: float,
    *,
    n_units: int = 6,
    time_window_seconds: float | None = 20.0,
    time_offset_seconds: float = 0.0,
    unit_indices: np.ndarray | list[int] | None = None,
    quantiles: tuple[float, ...] = (0.25, 0.5, 0.75),
    display_mode: DisplayMode | None = None,
    color_original: str = _COLOR_ORIGINAL,
    color_reconstructed: str = _COLOR_RECONSTRUCTED,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Per-unit raster plots comparing original vs reconstructed activity.

    Each row shows one neuron with original spikes/rates on top and
    reconstructed below. Neurons are selected from different firing rate
    quantiles to show performance across regimes.

    Parameters
    ----------
    segment:
        Collated validation segment.
    bin_size_seconds:
        Temporal resolution of the data.
    n_units:
        Number of units to plot (ignored if unit_indices given).
    time_window_seconds:
        Duration of the time window to display. None = full segment.
    time_offset_seconds:
        Start of the time window (in seconds from segment start).
    unit_indices:
        Explicit unit indices to plot. Overrides n_units/quantiles.
    quantiles:
        Quantile positions for automatic unit selection.
    display_mode:
        Force BINARY_SPIKES or RATES rendering. None = auto-detect.
    figsize:
        Figure size. None = auto.
    """
    meta = _extract_plot_metadata(segment, bin_size_seconds)
    orig = _to_numpy(_require_tensor(segment.original))
    recon = _to_numpy(_require_tensor(segment.reconstructed))
    mode = _detect_display_mode(orig, explicit=display_mode, bin_size_seconds=bin_size_seconds)

    # Select units
    if unit_indices is not None:
        sel_indices = np.asarray(unit_indices)
        q_labels = ["" for _ in sel_indices]
    else:
        sel_indices, q_labels = _select_units_by_rate_quantile(
            orig,
            n_units,
            quantiles=quantiles,
        )

    n_sel = len(sel_indices)

    # Compute time window in bins
    start_bin = int(time_offset_seconds / bin_size_seconds)
    if time_window_seconds is not None:
        end_bin = min(
            start_bin + int(time_window_seconds / bin_size_seconds),
            meta.n_time_bins,
        )
    else:
        end_bin = meta.n_time_bins
    time_axis = np.arange(start_bin, end_bin) * bin_size_seconds

    if figsize is None:
        figsize = (14, 2.0 * n_sel)

    fig, axes = plt.subplots(
        n_sel,
        1,
        figsize=figsize,
        sharex=True,
        squeeze=False,
    )
    axes = axes[:, 0]

    for row, (unit_idx, q_label) in enumerate(zip(sel_indices, q_labels, strict=True)):
        ax = axes[row]
        o = orig[unit_idx, start_bin:end_bin]
        r = recon[unit_idx, start_bin:end_bin]
        rate_hz = orig[unit_idx].sum() / meta.duration_seconds

        label = _unit_label(meta, unit_idx)
        rate_str = f"{rate_hz:.1f} Hz"
        q_str = f" [{q_label}]" if q_label else ""

        if mode == DisplayMode.BINARY_SPIKES:
            orig_times = time_axis[o > 0]
            recon_times = time_axis[r > 0]
            ax.eventplot(
                orig_times,
                lineoffsets=0.2,
                linelengths=0.35,
                color=color_original,
                linewidths=0.8,
            )
            ax.eventplot(
                recon_times,
                lineoffsets=-0.2,
                linelengths=0.35,
                color=color_reconstructed,
                linewidths=0.8,
            )
            ax.set_ylim(-0.5, 0.5)
            ax.set_yticks([])
        else:
            ax.plot(
                time_axis,
                o,
                color=color_original,
                linewidth=0.8,
                label="Original",
            )
            ax.plot(
                time_axis,
                r,
                color=color_reconstructed,
                linewidth=0.8,
                alpha=0.8,
                label="Reconstructed",
            )
            if row == 0:
                ax.legend(fontsize=7, loc="upper right")

        ax.set_ylabel(f"{label}\n{rate_str}{q_str}", fontsize=8)

    axes[0].set_title("Original vs Reconstructed", fontsize=10)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(
        _suptitle_text(meta),
        fontsize=10,
    )
    fig.text(
        0.5,
        0.96,
        _subtitle_text(meta),
        fontsize=10,
        ha="center",
        color="gray",
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.93)
    return fig


# ---------------------------------------------------------------------------
# Plot 2: Population raster
# ---------------------------------------------------------------------------


def plot_population_raster(
    segment: CollatedValidationSegment,
    bin_size_seconds: float,
    *,
    n_units: int = 50,
    time_window_seconds: float = 20.0,
    activity_quantile: float = 0.5,
    unit_indices: np.ndarray | list[int] | None = None,
    display_mode: DisplayMode | None = None,
    color_original: str = _COLOR_ORIGINAL,
    color_reconstructed: str = _COLOR_RECONSTRUCTED,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Population raster plot: Original vs Reconstructed side by side.

    Displays many neurons at once. The time window is selected based on
    population activity quantile (e.g. 0.9 for a highly active period).

    Parameters
    ----------
    segment:
        Collated validation segment.
    bin_size_seconds:
        Temporal resolution.
    n_units:
        Max number of units to display (sorted by firing rate).
    time_window_seconds:
        Duration of the displayed time window.
    activity_quantile:
        Population activity quantile for time window selection.
        0.1 = quiet period, 0.5 = median, 0.9 = active period.
    unit_indices:
        Explicit unit indices. Overrides n_units.
    display_mode:
        Force rendering mode. None = auto-detect.
    figsize:
        Figure size. None = auto.
    """
    meta = _extract_plot_metadata(segment, bin_size_seconds)
    orig = _to_numpy(_require_tensor(segment.original))
    recon = _to_numpy(_require_tensor(segment.reconstructed))
    mode = _detect_display_mode(orig, explicit=display_mode, bin_size_seconds=bin_size_seconds)

    # Select units (sorted by firing rate for visual clarity)
    if unit_indices is not None:
        sel_indices = np.asarray(unit_indices)
    else:
        rates = orig.sum(axis=1)
        sorted_idx = np.argsort(rates)[::-1]  # highest first
        sel_indices = sorted_idx[: min(n_units, meta.n_units)]

    n_sel = len(sel_indices)

    # Select time window by activity quantile
    window_bins = int(time_window_seconds / bin_size_seconds)
    start_bin, end_bin = _select_time_window_by_activity(
        orig,
        window_bins,
        quantile=activity_quantile,
    )
    time_start_s = start_bin * bin_size_seconds
    time_end_s = end_bin * bin_size_seconds
    time_axis = np.arange(start_bin, end_bin) * bin_size_seconds

    if figsize is None:
        figsize = (16, max(6, n_sel * 0.15))

    fig, (ax_orig, ax_recon) = plt.subplots(
        1,
        2,
        figsize=figsize,
        sharey=True,
    )

    # Build y-axis labels
    y_labels = [_unit_label(meta, idx) for idx in sel_indices]

    if mode == DisplayMode.BINARY_SPIKES:
        for row, unit_idx in enumerate(sel_indices):
            o_times = time_axis[orig[unit_idx, start_bin:end_bin] > 0]
            r_times = time_axis[recon[unit_idx, start_bin:end_bin] > 0]
            ax_orig.eventplot(
                o_times,
                lineoffsets=row,
                linelengths=0.7,
                color=color_original,
                linewidths=0.6,
            )
            ax_recon.eventplot(
                r_times,
                lineoffsets=row,
                linelengths=0.7,
                color=color_reconstructed,
                linewidths=0.6,
            )

        ax_orig.set_ylim(-0.5, n_sel - 0.5)
        ax_recon.set_ylim(-0.5, n_sel - 0.5)
    else:
        # Heatmap for rates
        orig_slice = orig[sel_indices, start_bin:end_bin]
        recon_slice = recon[sel_indices, start_bin:end_bin]
        vmax = max(
            np.percentile(orig_slice, 99),
            np.percentile(recon_slice, 99),
        )
        extent = [time_start_s, time_end_s, n_sel - 0.5, -0.5]
        im_kw = {
            "aspect": "auto",
            "cmap": "hot",
            "vmin": 0,
            "vmax": vmax,
            "extent": extent,
        }
        ax_orig.imshow(orig_slice, **im_kw)
        ax_recon.imshow(recon_slice, **im_kw)

    # Y-axis labels (show every Nth to avoid clutter)
    step = max(1, n_sel // 20)
    ax_orig.set_yticks(range(0, n_sel, step))
    ax_orig.set_yticklabels([y_labels[i] for i in range(0, n_sel, step)], fontsize=6)

    ax_orig.set_title("Original", fontsize=10)
    ax_recon.set_title("Reconstructed", fontsize=10)
    ax_orig.set_xlabel("Time (s)")
    ax_recon.set_xlabel("Time (s)")
    ax_orig.set_ylabel("Neurons (sorted by rate)")

    q_label = f"activity quantile={activity_quantile}"
    fig.suptitle(
        f"{_suptitle_text(meta)}\n{_subtitle_text(meta)} | {q_label} | "
        f"t=[{time_start_s:.1f}s, {time_end_s:.1f}s]",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


def plot_population_raster_correspondence(
    segment: CollatedValidationSegment,
    bin_size_seconds: float,
    *,
    n_units: int = 50,
    time_window_seconds: float = 20.0,
    activity_quantile: float = 0.5,
    tolerance_bins: int = 1,
    unit_indices: np.ndarray | list[int] | None = None,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Population raster coloured by spike correspondence.

    For binary spike data, each spike is coloured according to whether
    it was correctly reconstructed:

    - **Green**: true positive (original spike matched by a
      reconstructed spike within *tolerance_bins*).
    - **Black**: false negative (original spike with no matching
      reconstruction).
    - **Red**: false positive (reconstructed spike with no matching
      original).

    Parameters
    ----------
    segment:
        Collated validation segment.
    bin_size_seconds:
        Temporal resolution.
    n_units:
        Max number of units to display (sorted by firing rate).
    time_window_seconds:
        Duration of the displayed time window.
    activity_quantile:
        Population activity quantile for time window selection.
    tolerance_bins:
        A reconstructed spike within this many bins of an original spike
        counts as a match. 0 = exact match only.
    unit_indices:
        Explicit unit indices. Overrides n_units.
    figsize:
        Figure size. None = auto.
    """
    meta = _extract_plot_metadata(segment, bin_size_seconds)
    orig = _to_numpy(_require_tensor(segment.original))
    recon = _to_numpy(_require_tensor(segment.reconstructed))

    # Select units
    if unit_indices is not None:
        sel_indices = np.asarray(unit_indices)
    else:
        rates = orig.sum(axis=1)
        sorted_idx = np.argsort(rates)[::-1]
        sel_indices = sorted_idx[: min(n_units, meta.n_units)]

    n_sel = len(sel_indices)

    # Select time window
    window_bins = int(time_window_seconds / bin_size_seconds)
    start_bin, end_bin = _select_time_window_by_activity(
        orig,
        window_bins,
        quantile=activity_quantile,
    )
    time_start_s = start_bin * bin_size_seconds
    time_end_s = end_bin * bin_size_seconds
    time_axis = np.arange(start_bin, end_bin) * bin_size_seconds

    if figsize is None:
        figsize = (14, max(6, n_sel * 0.15))

    fig, ax = plt.subplots(figsize=figsize)

    # Build y-axis labels
    y_labels = [_unit_label(meta, idx) for idx in sel_indices]

    for row, unit_idx in enumerate(sel_indices):
        o_slice = orig[unit_idx, start_bin:end_bin]
        r_slice = recon[unit_idx, start_bin:end_bin]

        o_bins = np.where(o_slice > 0)[0]
        r_bins = np.where(r_slice > 0)[0]

        # Match: for each original spike, is there a recon spike
        # within tolerance?
        o_matched = np.zeros(len(o_bins), dtype=bool)
        r_matched = np.zeros(len(r_bins), dtype=bool)

        if len(o_bins) > 0 and len(r_bins) > 0:
            for i, ob in enumerate(o_bins):
                dists = np.abs(r_bins - ob)
                closest = np.argmin(dists)
                if dists[closest] <= tolerance_bins:
                    o_matched[i] = True
                    r_matched[closest] = True

        # True positives (green): matched original spikes
        tp_times = time_axis[o_bins[o_matched]]
        # False negatives (black): unmatched original spikes
        fn_times = time_axis[o_bins[~o_matched]]
        # False positives (red): unmatched reconstructed spikes
        fp_times = time_axis[r_bins[~r_matched]]

        for times, color in [
            (tp_times, "tab:green"),
            (fn_times, "k"),
            (fp_times, "tab:red"),
        ]:
            if len(times) > 0:
                ax.eventplot(
                    times,
                    lineoffsets=row,
                    linelengths=0.7,
                    color=color,
                    linewidths=0.6,
                )

    ax.set_ylim(-0.5, n_sel - 0.5)
    step = max(1, n_sel // 20)
    ax.set_yticks(range(0, n_sel, step))
    ax.set_yticklabels(
        [y_labels[i] for i in range(0, n_sel, step)],
        fontsize=6,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Neurons (sorted by rate)")

    # Legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="tab:green", lw=2, label="True positive"),
        Line2D([0], [0], color="k", lw=2, label="False negative"),
        Line2D([0], [0], color="tab:red", lw=2, label="False positive"),
    ]
    ax.legend(
        handles=legend_elements,
        fontsize=8,
        loc="upper right",
    )

    tol_ms = tolerance_bins * bin_size_seconds * 1000
    q_label = f"activity quantile={activity_quantile}"
    fig.suptitle(
        f"Spike Correspondence — {_suptitle_text(meta)}\n"
        f"{_subtitle_text(meta)} | {q_label} | "
        f"tolerance={tol_ms:.1f}ms | "
        f"t=[{time_start_s:.1f}s, {time_end_s:.1f}s]",
        fontsize=10,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 3: Firing rate comparison
# ---------------------------------------------------------------------------


def plot_firing_rate_comparison(
    segment: CollatedValidationSegment,
    bin_size_seconds: float,
    *,
    color_reconstructed: str = _COLOR_RECONSTRUCTED,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Scatter plot of per-unit firing rates: original vs reconstructed.

    Includes identity line, linear regression fit, r-squared, and p-value.

    Parameters
    ----------
    segment:
        Collated validation segment.
    bin_size_seconds:
        Temporal resolution.
    figsize:
        Figure size. None = (6, 6).
    """
    meta = _extract_plot_metadata(segment, bin_size_seconds)
    orig = _to_numpy(_require_tensor(segment.original))
    recon = _to_numpy(_require_tensor(segment.reconstructed))

    orig_rates = orig.sum(axis=1) / meta.duration_seconds
    recon_rates = recon.sum(axis=1) / meta.duration_seconds

    if figsize is None:
        figsize = (6, 6)
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(
        orig_rates,
        recon_rates,
        alpha=0.6,
        s=20,
        c=color_reconstructed,
        edgecolors="none",
    )

    # Identity line
    max_rate = max(orig_rates.max(), recon_rates.max()) * 1.1
    ax.plot([0, max_rate], [0, max_rate], "k--", lw=0.8, label="y = x")

    # Linear regression
    result: Any = linregress(orig_rates, recon_rates)  # scipy stub lacks result attrs
    x_fit = np.linspace(0, max_rate, 100)
    ax.plot(
        x_fit,
        result.slope * x_fit + result.intercept,
        "tab:red",
        lw=1,
        alpha=0.7,
        label="Linear fit",
    )

    # Annotate stats
    r_sq = result.rvalue**2
    p_str = f"{result.pvalue:.2e}" if result.pvalue < 0.001 else f"{result.pvalue:.4f}"
    stat_text = f"r = {result.rvalue:.3f}\n$r^2$ = {r_sq:.3f}\np = {p_str}\nn = {meta.n_units}"
    ax.text(
        0.05,
        0.92,
        stat_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
    )

    ax.set_xlabel("Original firing rate (Hz)")
    ax.set_ylabel("Reconstructed firing rate (Hz)")
    ax.set_aspect("equal")
    ax.set_xlim(0, max_rate)
    ax.set_ylim(0, max_rate)
    ax.legend(fontsize=8, loc="lower right")

    ax.set_title(
        f"Per-Unit Firing Rates\n{_suptitle_text(meta)}",
        fontsize=10,
    )
    fig.text(0.5, -0.01, _subtitle_text(meta), ha="center", fontsize=8, color="gray")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Plot 4: ACG comparison
# ---------------------------------------------------------------------------


def plot_acg_comparison(
    segment: CollatedValidationSegment,
    bin_size_seconds: float,
    *,
    n_units: int = 9,
    unit_indices: np.ndarray | list[int] | None = None,
    quantiles: tuple[float, ...] = (0.25, 0.5, 0.75),
    max_lag_ms: float = 50.0,
    acg_bin_size_ms: float | None = None,
    smoothing_kernel: int = 1,
    max_cols: int = 5,
    color_original: str = _COLOR_ORIGINAL,
    color_reconstructed: str = _COLOR_RECONSTRUCTED,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Grid of autocorrelogram comparisons (original vs reconstructed).

    Parameters
    ----------
    segment:
        Collated validation segment.
    bin_size_seconds:
        Temporal resolution.
    n_units:
        Number of units to plot (ignored if unit_indices given).
    unit_indices:
        Explicit unit indices. Overrides n_units/quantiles.
    quantiles:
        Quantile positions for automatic unit selection.
    max_lag_ms:
        Maximum ACG lag in milliseconds.
    acg_bin_size_ms:
        ACG bin size in milliseconds. None = match data resolution.
    max_cols:
        Maximum columns in the grid layout.
    figsize:
        Figure size. None = auto.
    """
    meta = _extract_plot_metadata(segment, bin_size_seconds)
    orig = _to_numpy(_require_tensor(segment.original))

    if acg_bin_size_ms is None:
        acg_bin_size_ms = bin_size_seconds * 1000

    # Select units
    if unit_indices is not None:
        sel_indices = np.asarray(unit_indices)
        q_labels = ["" for _ in sel_indices]
    else:
        sel_indices, q_labels = _select_units_by_rate_quantile(
            orig,
            n_units,
            quantiles=quantiles,
        )

    n_sel = len(sel_indices)

    # Compute ACGs for ALL units
    original_data = EvalData.from_tensor(
        _require_tensor(segment.original),
        DataType.SPIKES,
        bin_size_seconds=bin_size_seconds,
        duration=meta.duration_seconds,
    )
    reconstructed_data = EvalData.from_tensor(
        _require_tensor(segment.reconstructed),
        DataType.SPIKES,
        bin_size_seconds=bin_size_seconds,
        duration=meta.duration_seconds,
    )
    acgs_orig = compute_acgs(
        original_data,
        max_lag_ms=max_lag_ms,
        bin_size_ms=acg_bin_size_ms,
        smoothing_kernel=smoothing_kernel,
    )
    acgs_recon = compute_acgs(
        reconstructed_data,
        max_lag_ms=max_lag_ms,
        bin_size_ms=acg_bin_size_ms,
        smoothing_kernel=smoothing_kernel,
    )

    # Grid layout
    nrows, ncols = _compute_grid_layout(n_sel, max_cols)
    if figsize is None:
        figsize = (3.5 * ncols, 2.8 * nrows)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        squeeze=False,
    )

    n_acg_bins = len(acgs_orig[0])
    acg_time = np.linspace(-max_lag_ms, max_lag_ms, n_acg_bins)

    for i, (unit_idx, q_label) in enumerate(
        zip(sel_indices, q_labels, strict=True),
    ):
        row, col = divmod(i, ncols)
        ax = axes[row, col]

        y_orig = acgs_orig[unit_idx]
        y_recon = acgs_recon[unit_idx]

        # Fill + step style
        ax.fill_between(
            acg_time,
            0,
            y_orig,
            step="mid",
            color=color_original,
            alpha=0.15,
        )
        ax.step(
            acg_time,
            y_orig,
            where="mid",
            color=color_original,
            lw=1,
            label="Original",
        )
        ax.fill_between(
            acg_time,
            0,
            y_recon,
            step="mid",
            color=color_reconstructed,
            alpha=0.15,
        )
        ax.step(
            acg_time,
            y_recon,
            where="mid",
            color=color_reconstructed,
            lw=1,
            label="Reconstructed",
        )

        # Refractory period lines at ±1 ms
        ylim_top = max(y_orig.max(), y_recon.max()) * 1.05
        if ylim_top > 0:
            ax.set_ylim(0, ylim_top)
            for ref_ms in (-1, 1):
                ax.axvline(
                    ref_ms,
                    color="k",
                    linestyle="--",
                    linewidth=0.7,
                    alpha=0.5,
                )

        label = _unit_label(meta, unit_idx)
        rate_hz = orig[unit_idx].sum() / meta.duration_seconds
        q_str = f" [{q_label}]" if q_label else ""
        ax.set_title(f"{label} ({rate_hz:.1f} Hz){q_str}", fontsize=8)

        if row == nrows - 1:
            ax.set_xlabel("Lag (ms)", fontsize=8)
        if col == 0:
            ax.set_ylabel("Count", fontsize=8)
        if i == 0:
            ax.legend(fontsize=6, loc="upper right")

    # Hide unused subplots
    for i in range(n_sel, nrows * ncols):
        row, col = divmod(i, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle(
        f"Autocorrelograms: Original vs Reconstructed\n{_suptitle_text(meta)}",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Convenience: all plots at once
# ---------------------------------------------------------------------------


def plot_reconstruction_summary(
    segment: CollatedValidationSegment,
    bin_size_seconds: float,
    *,
    n_units_raster: int = 6,
    n_units_population: int = 50,
    n_units_acg: int = 9,
    time_window_seconds: float = 20.0,
    display_mode: DisplayMode | None = None,
    color_original: str = _COLOR_ORIGINAL,
    color_reconstructed: str = _COLOR_RECONSTRUCTED,
) -> dict[str, Figure]:
    """Generate all four reconstruction plots at once.

    Returns a dict mapping plot names to figures:
    ``per_unit_raster``, ``population_raster``, ``firing_rate``, ``acg``.
    """
    return {
        "per_unit_raster": plot_per_unit_raster(
            segment,
            bin_size_seconds,
            n_units=n_units_raster,
            time_window_seconds=time_window_seconds,
            display_mode=display_mode,
            color_original=color_original,
            color_reconstructed=color_reconstructed,
        ),
        "population_raster": plot_population_raster(
            segment,
            bin_size_seconds,
            n_units=n_units_population,
            time_window_seconds=time_window_seconds,
            display_mode=display_mode,
            color_original=color_original,
            color_reconstructed=color_reconstructed,
        ),
        "firing_rate": plot_firing_rate_comparison(
            segment,
            bin_size_seconds,
            color_reconstructed=color_reconstructed,
        ),
        "acg": plot_acg_comparison(
            segment,
            bin_size_seconds,
            n_units=n_units_acg,
            color_original=color_original,
            color_reconstructed=color_reconstructed,
        ),
    }
