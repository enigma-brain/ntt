from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

from torch.utils.data import DataLoader, Dataset

from .dataset import ReplicatedTransformDataset
from .sampler import ReplicaWindowIndex, WindowReplicaBatchSampler, _replica_seed


def build_replica_dataloader(
    *,
    base_dataset: Any,
    base_sampler: Iterable[Any],
    transform: Any,
    n_replicas: int,
    max_windows: int | None,
    max_eval_seconds: float | None,
    effective_window_seconds: float | None,
    base_seed: int,
    num_workers: int = 4,
    pin_memory: bool = False,
) -> DataLoader:
    """Build a dataloader that yields one grouped replica batch per source window."""
    from ntt.core.data.datamodule import collate

    dataset = ReplicatedTransformDataset(
        base_dataset,
        transform=transform,
        base_seed=base_seed,
    )
    batch_sampler = WindowReplicaBatchSampler(
        base_sampler,
        n_replicas=n_replicas,
        max_windows=max_windows,
        max_eval_seconds=max_eval_seconds,
        effective_window_seconds=effective_window_seconds,
    )
    return DataLoader(
        # Map-style dataset keyed by ReplicaWindowIndex (the batch sampler yields those).
        dataset=cast("Dataset[Any]", dataset),
        batch_sampler=batch_sampler,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


__all__ = [
    "ReplicaWindowIndex",
    "ReplicatedTransformDataset",
    "WindowReplicaBatchSampler",
    "_replica_seed",
    "build_replica_dataloader",
]
