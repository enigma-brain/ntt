"""Extract per-neuron embeddings from a trained NuCLR checkpoint.

Standalone script: loads a checkpoint, runs forward pass over sessions,
and saves ``{embs, unit_ids, session_ids}`` to a ``.pt`` file.

Supports two modes:

1. **Single brainset**: specify ``--brainset`` and optionally
   ``--sessions`` to extract from one brainset.

2. **NTT data config**: specify ``--data-config ntt_medium`` and
   ``--split test`` to extract from all brainsets in an NTT data selection.

Usage::

    # Single brainset
    uv run python -m ntt.contrib.nuclr.eval.extract_embeddings \
        --ckpt-path <checkpoint.ckpt> \
        --data-root <brainsets_root> \
        --brainset IBL_brainwide_2023 \
        --output-path embeddings.pt

    # Full NTT data config
    uv run python -m ntt.contrib.nuclr.eval.extract_embeddings \
        --ckpt-path <checkpoint.ckpt> \
        --data-root <brainsets_root> \
        --data-config ntt_medium \
        --split test \
        --output-path embeddings.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import omegaconf
import torch
from torch.utils.data import DataLoader

from ntt.contrib.nuclr.adapter import ntt_to_nuclr_format
from ntt.contrib.nuclr.embedding_collector import EmbeddingCollector
from ntt.contrib.nuclr.litmodule import NuclrContrastiveLitModule
from ntt.core.components.types import DataType, SpikeTimesInput
from ntt.core.data.components.transforms import DataToModelInput

logger = logging.getLogger(__name__)


def _strip_brainset_prefix(session_id: str) -> str:
    """Strip brainset prefix from torch_brain session_id.

    torch_brain returns ``"brainset_name/session_stem"`` but labels.py
    indexes by ``session_stem`` (the H5 filename without extension).
    """
    return session_id.rsplit("/", 1)[-1] if "/" in session_id else session_id


def _get_session_ids_per_unit(
    batch: SpikeTimesInput,
    unit_seqlen: torch.Tensor,
) -> np.ndarray:
    """Expand per-batch session_ids to per-unit session_ids.

    Args:
        batch: SpikeTimesInput with session_id (str or tuple[str, ...]).
        unit_seqlen: Number of units per batch element. Shape ``(batch,)``.

    Returns:
        Numpy array of session IDs, one per unit (total_units,).
    """
    session_ids = batch.session_id
    if isinstance(session_ids, str):
        session_ids = (session_ids,)

    per_unit = []
    for sid, n_units in zip(session_ids, unit_seqlen.tolist(), strict=False):
        per_unit.extend([_strip_brainset_prefix(str(sid))] * n_units)
    return np.array(per_unit)


def _get_brainset_unit_ids(
    batch: SpikeTimesInput,
) -> np.ndarray:
    """Get string unit IDs that can be matched to brainsets metadata.

    The NTT pipeline uses ``unit_number`` (sequential int) as ``unit_ids``.
    To create globally unique IDs, we combine session_id + unit_number.

    Returns:
        Numpy array of string IDs like ``"session_stem/unit_number"``.
    """
    unit_ids = batch.unit_ids  # (batch, max_units)
    unit_mask = batch.unit_ids_pad_mask  # (batch, max_units)
    session_ids = batch.session_id
    if isinstance(session_ids, str):
        session_ids = (session_ids,)

    all_ids = []
    for b, sid in enumerate(session_ids):
        sid_clean = _strip_brainset_prefix(str(sid))
        valid = unit_mask[b].bool()
        uids = unit_ids[b][valid].cpu().numpy()
        for uid in uids:
            all_ids.append(f"{sid_clean}/{uid}")
    return np.array(all_ids)


def _load_model(
    ckpt_path: str | Path,
    device: str,
) -> NuclrContrastiveLitModule:
    """Load a NuCLR checkpoint and prepare for inference."""
    ckpt_path = Path(ckpt_path)
    logger.info("Loading checkpoint from %s", ckpt_path)
    model = NuclrContrastiveLitModule.load_from_checkpoint(
        str(ckpt_path),
        map_location="cpu",
        weights_only=False,
    )
    model.configure_model()
    model = model.to(device)
    model.eval()
    return model


def _checkpoint_path(output_path: str | Path) -> Path:
    """Return the checkpoint file path for a given output path."""
    return Path(str(output_path) + ".checkpoint")


def _save_checkpoint(
    checkpoint_path: Path,
    collector: EmbeddingCollector,
    all_session_ids: dict[str, str],
    batch_idx: int,
) -> None:
    """Save extraction progress to a checkpoint file."""
    state = {
        "collector": collector.state_dict(),
        "session_ids": all_session_ids,
        "batch_idx": batch_idx,
    }
    # Write to temp file first, then rename for atomicity
    tmp_path = checkpoint_path.with_suffix(".tmp")
    torch.save(state, tmp_path)
    tmp_path.rename(checkpoint_path)


def _load_checkpoint(
    checkpoint_path: Path,
    collector: EmbeddingCollector,
) -> tuple[dict[str, str], int]:
    """Load extraction progress from a checkpoint file.

    Returns:
        Tuple of (all_session_ids, last_completed_batch_idx).
    """
    # Progress checkpoint holds only the collector state_dict + plain dict/int.
    state = torch.load(checkpoint_path, weights_only=True)
    collector.load_state_dict(state["collector"])
    return state["session_ids"], state["batch_idx"]


@torch.no_grad()
def _extract_loop(
    model: NuclrContrastiveLitModule,
    loader: DataLoader,
    device: str,
    output_path: str | Path | None = None,
    checkpoint_interval: int = 200,
) -> dict:
    """Run forward pass over a DataLoader and collect per-unit mean embeddings.

    Supports checkpointing: if ``output_path`` is provided, progress is saved
    every ``checkpoint_interval`` batches to ``{output_path}.checkpoint``.
    On restart, if a checkpoint exists, extraction resumes from where it left
    off (batches before the checkpoint are skipped).

    Args:
        model: Loaded NuCLR model in eval mode.
        loader: DataLoader yielding SpikeTimesInput batches.
        device: Device string.
        output_path: Output file path (used to derive checkpoint path).
            If None, checkpointing is disabled.
        checkpoint_interval: Save checkpoint every N batches.

    Returns:
        Dict with ``embs`` (float32), ``unit_ids``, ``session_ids``.
    """
    collector = EmbeddingCollector(device=torch.device(device))
    all_session_ids: dict[str, str] = {}
    start_batch = 0

    # Try to resume from checkpoint
    ckpt_file = _checkpoint_path(output_path) if output_path else None
    if ckpt_file and ckpt_file.exists():
        all_session_ids, last_batch = _load_checkpoint(ckpt_file, collector)
        start_batch = last_batch + 1
        logger.info(
            "Resuming from checkpoint: skipping first %d batches (%d units already collected)",
            start_batch,
            len(collector._emb_sums),
        )

    total = len(loader)
    logger.info("Extracting embeddings (%d batches, starting at %d)...", total, start_batch)

    for batch_idx, batch in enumerate(loader):
        if batch_idx < start_batch:
            continue

        batch = batch._replace(
            **{
                f: v.to(device) if isinstance(v, torch.Tensor) else v
                for f, v in batch._asdict().items()
            }
        )

        nuclr_input = ntt_to_nuclr_format(batch)
        assert model.encoder is not None, "model.configure_model() must build the encoder"
        embeddings = model.encoder(
            nuclr_input.spike_timestamps,
            nuclr_input.spike_seqlen,
            nuclr_input.unit_seqlen,
        )

        composite_ids = _get_brainset_unit_ids(batch)
        session_ids_per_unit = _get_session_ids_per_unit(
            batch,
            nuclr_input.unit_seqlen,
        )

        collector.update(embeddings, composite_ids)
        for cid, sid in zip(composite_ids, session_ids_per_unit, strict=False):
            all_session_ids[cid] = sid

        if (batch_idx + 1) % 50 == 0:
            logger.info(
                "  processed %d/%d batches",
                batch_idx + 1,
                total,
            )

        if ckpt_file and (batch_idx + 1) % checkpoint_interval == 0:
            _save_checkpoint(ckpt_file, collector, all_session_ids, batch_idx)
            logger.info("  checkpoint saved at batch %d", batch_idx + 1)

    # Compute mean embeddings (collector returns float64, cast to float32)
    embs, unit_ids = collector.compute()
    embs = embs.cpu().float()

    session_ids_arr = np.array([all_session_ids[uid] for uid in unit_ids])

    # Clean up checkpoint on successful completion
    if ckpt_file and ckpt_file.exists():
        ckpt_file.unlink()
        logger.info("Checkpoint removed (extraction complete)")

    return {
        "embs": embs,
        "unit_ids": unit_ids,
        "session_ids": session_ids_arr,
    }


def _save_result(result: dict, output_path: str | Path) -> Path:
    """Save extraction result to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output_path)
    embs = result["embs"]
    logger.info(
        "Saved %d unit embeddings (dim=%d) to %s",
        embs.shape[0],
        embs.shape[1],
        output_path,
    )
    return output_path


@torch.no_grad()
def extract_embeddings(
    ckpt_path: str | Path,
    data_root: str | Path,
    brainset: str,
    output_path: str | Path,
    sessions: list[str] | None = None,
    batch_size: int = 32,
    window_length: float = 10.0,
    num_workers: int = 4,
    device: str = "cuda",
) -> Path:
    """Extract embeddings from a single brainset.

    Args:
        ckpt_path: Path to Lightning checkpoint.
        data_root: Root directory containing brainsets.
        brainset: Brainset name.
        output_path: Where to save the output ``.pt`` file.
        sessions: Optional list of session IDs to process.
        batch_size: Batch size for inference.
        window_length: Context window duration in seconds.
        num_workers: DataLoader workers.
        device: Device for inference.

    Returns:
        Path to saved embeddings file.
    """
    from torch_brain.data import Dataset as TBDataset
    from torch_brain.data.sampler import SequentialFixedWindowSampler

    from ntt.core.data.datamodule import collate

    model = _load_model(ckpt_path, device)

    ctx_duration = model.hparams["ctx_duration"]
    if window_length != ctx_duration:
        logger.warning(
            "window_length=%.1f differs from model ctx_duration=%.1f, using model's ctx_duration",
            window_length,
            ctx_duration,
        )
        window_length = ctx_duration

    # Build torch_brain config for session selection
    selection: dict = {"brainset": brainset}
    if sessions is not None:
        selection["sessions"] = [str(s) for s in sessions]
    config = omegaconf.OmegaConf.create([{"selection": [selection]}])

    logger.info("Setting up dataset: %s", brainset)
    transform = DataToModelInput(mode=DataType.SPIKE_TIMES)
    dataset = TBDataset(
        str(data_root),
        config=cast(Any, config),  # ListConfig is accepted by torch_brain at runtime
        split=None,
        transform=transform,
    )
    logger.info("Found %d recordings", len(dataset.recording_dict))

    sampling_intervals = dataset.get_sampling_intervals()
    sampler = SequentialFixedWindowSampler(
        sampling_intervals=sampling_intervals,
        window_length=window_length,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=(device != "cpu"),
    )

    result = _extract_loop(model, loader, device, output_path=output_path)
    return _save_result(result, output_path)


@torch.no_grad()
def extract_embeddings_from_selections(
    ckpt_path: str | Path,
    data_root: str | Path,
    selections: list,
    output_path: str | Path,
    split: str = "test",
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = "cuda",
) -> Path:
    """Extract embeddings from a list of SelectionConfig objects.

    This is the multi-brainset entry point. It builds a single dataset
    spanning all brainsets/sessions in the provided selections and runs
    extraction in one pass.

    Args:
        ckpt_path: Path to Lightning checkpoint.
        data_root: Root directory containing brainsets.
        selections: List of ``SelectionConfig`` objects (or dicts that
            can be passed to ``OmegaConf.create``).
        output_path: Where to save the output ``.pt`` file.
        split: Which split to use (``"train"``, ``"val"``, ``"test"``).
            Controls which time domains are sampled. Use ``None`` for full
            domain (no split restriction).
        batch_size: Batch size for inference.
        num_workers: DataLoader workers.
        device: Device for inference.

    Returns:
        Path to saved embeddings file.
    """
    from torch_brain.data.sampler import SequentialFixedWindowSampler

    from ntt.core.data.components.config import SelectionConfig
    from ntt.core.data.components.dataset import NTTDataset
    from ntt.core.data.datamodule import collate

    model = _load_model(ckpt_path, device)
    window_length = model.hparams["ctx_duration"]

    # Convert SelectionConfig objects to OmegaConf ListConfig
    serialised = []
    for s in selections:
        if isinstance(s, SelectionConfig):
            serialised.append(s.model_dump())
        elif isinstance(s, dict):
            serialised.append(s)
        else:
            raise TypeError(f"Expected SelectionConfig or dict, got {type(s)}")
    oc_config = omegaconf.ListConfig(serialised)

    # Map split name to torch_brain convention
    dataset_split: str | None
    if split is None:
        dataset_split = None
    elif split == "val":
        dataset_split = "valid"
    else:
        dataset_split = split

    transform = DataToModelInput(mode=DataType.SPIKE_TIMES)
    dataset = NTTDataset(
        root=str(data_root),
        split=dataset_split,
        config=oc_config,
        transform=transform,
    )
    logger.info(
        "Dataset: %d recordings across %d selections (split=%s)",
        len(dataset.recording_dict),
        len(selections),
        split,
    )

    sampling_intervals = dataset.get_sampling_intervals()
    sampler = SequentialFixedWindowSampler(
        sampling_intervals=sampling_intervals,
        window_length=window_length,
        drop_short=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=(device != "cpu"),
    )

    result = _extract_loop(model, loader, device, output_path=output_path)
    return _save_result(result, output_path)


def _load_selections_from_data_config(
    data_config: str,
    split: str,
) -> list:
    """Load SelectionConfig objects by parsing a data config YAML directly.

    Reads the data config file (e.g. ``config/data/ntt_full.yaml``), parses
    its Hydra defaults list to find selection file references for the requested
    split, then loads each selection YAML and constructs ``SelectionConfig``
    objects.

    This avoids full Hydra composition, which would require satisfying
    unrelated required fields (tokenizer, loss, paths, etc.).
    """
    import re

    import yaml

    import ntt
    from ntt.core.data.components.config import SelectionConfig

    config_dir = Path(ntt.__file__).resolve().parent.parent.parent / "config"
    data_config_path = config_dir / "data" / f"{data_config}.yaml"

    if not data_config_path.exists():
        available = [
            p.stem for p in (config_dir / "data").glob("*.yaml") if not p.stem.startswith("_")
        ]
        raise FileNotFoundError(
            f"Data config not found: {data_config_path}. Available configs: {available}"
        )

    with data_config_path.open() as f:
        data_cfg = yaml.safe_load(f)

    # Parse Hydra defaults list for selection references matching this split.
    # Format: "selections@{split}_selections.{idx}: {selection_name}"
    defaults = data_cfg.get("defaults", [])
    pattern = re.compile(rf"selections@{split}_selections\.(\d+)")

    selection_names: list[tuple[int, str]] = []
    for entry in defaults:
        if isinstance(entry, dict):
            for key, value in entry.items():
                m = pattern.match(key)
                if m:
                    selection_names.append((int(m.group(1)), value))

    selection_names.sort(key=lambda x: x[0])

    if not selection_names:
        raise ValueError(
            f"No {split}_selections found in data config '{data_config}'. "
            f"Check that {data_config_path} has defaults like "
            f"'selections@{split}_selections.N: <name>'"
        )

    # Load each selection YAML
    selections_dir = config_dir / "data" / "selections"
    selections = []
    for _, name in selection_names:
        sel_path = selections_dir / f"{name}.yaml"
        if not sel_path.exists():
            raise FileNotFoundError(f"Selection config not found: {sel_path}")
        with sel_path.open() as f:
            raw = yaml.safe_load(f)
        raw.pop("_target_", None)
        selections.append(SelectionConfig(**raw))

    return selections


def extract_embeddings_from_data_config(
    ckpt_path: str | Path,
    data_root: str | Path,
    data_config: str,
    output_path: str | Path,
    split: str = "test",
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = "cuda",
) -> Path:
    """Extract embeddings using a Hydra data config name (e.g. ``ntt_medium``).

    Parses the data config YAML to find selection files for the requested
    split, then delegates to :func:`extract_embeddings_from_selections`.

    Args:
        ckpt_path: Path to Lightning checkpoint.
        data_root: Root directory containing brainsets.
        data_config: Hydra data config name (e.g. ``"ntt_medium"``).
        output_path: Where to save the output ``.pt`` file.
        split: Which split's selections to use (``"train"``, ``"val"``,
            ``"test"``).
        batch_size: Batch size for inference.
        num_workers: DataLoader workers.
        device: Device for inference.

    Returns:
        Path to saved embeddings file.
    """
    selections = _load_selections_from_data_config(data_config, split)

    logger.info(
        "Loaded %d selections from data config '%s' (split=%s)",
        len(selections),
        data_config,
        split,
    )

    return extract_embeddings_from_selections(
        ckpt_path=ckpt_path,
        data_root=data_root,
        selections=selections,
        output_path=output_path,
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Extract NuCLR per-neuron embeddings from checkpoint",
    )
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument(
        "--data-root", type=str, required=True, help="Root directory containing brainsets"
    )

    # Mode 1: single brainset
    single = parser.add_argument_group("single brainset mode")
    single.add_argument(
        "--brainset", type=str, default=None, help="Brainset name (e.g. IBL_brainwide_2023)"
    )
    single.add_argument("--sessions", type=str, nargs="*", default=None)
    single.add_argument("--window-length", type=float, default=10.0)

    # Mode 2: NTT data config
    multi = parser.add_argument_group("data config mode")
    multi.add_argument(
        "--data-config", type=str, default=None, help="Hydra data config name (e.g. ntt_medium)"
    )
    multi.add_argument(
        "--split", type=str, default="test", help="Split to extract from (train/val/test)"
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.data_config is not None and args.brainset is not None:
        parser.error("--data-config and --brainset are mutually exclusive")

    if args.data_config is not None:
        extract_embeddings_from_data_config(
            ckpt_path=args.ckpt_path,
            data_root=args.data_root,
            data_config=args.data_config,
            output_path=args.output_path,
            split=args.split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device,
        )
    elif args.brainset is not None:
        extract_embeddings(
            ckpt_path=args.ckpt_path,
            data_root=args.data_root,
            brainset=args.brainset,
            output_path=args.output_path,
            sessions=args.sessions,
            batch_size=args.batch_size,
            window_length=args.window_length,
            num_workers=args.num_workers,
            device=args.device,
        )
    else:
        parser.error("Either --data-config or --brainset must be specified")


if __name__ == "__main__":
    main()
