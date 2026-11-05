from __future__ import annotations

from typing import Any

from .sampler import ReplicaWindowIndex, _replica_seed


class ReplicatedTransformDataset:
    """Wrap a dataset so each item is one replica of one source window."""

    def __init__(self, base_dataset: Any, *, transform: Any, base_seed: int):
        """Store the base dataset, per-sample transform, and master seed."""
        self._base_dataset = base_dataset
        self._transform = transform
        self._base_seed = int(base_seed)

    def __getitem__(self, index: ReplicaWindowIndex) -> Any:
        """Load a window, stamp replica id/seed, and run the transform chain."""
        sample = self._base_dataset.get(index.recording_id, index.start, index.end)
        sample._ntt_replica_id = int(index.replica_id)
        sample._ntt_replica_seed = _replica_seed(
            base_seed=self._base_seed,
            recording_id=index.recording_id,
            start=index.start,
            end=index.end,
            replica_id=index.replica_id,
        )
        return self._transform(sample)
