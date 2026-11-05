"""Per-unit embedding accumulator for evaluation.

Ported from nuclr/src/evaluators/collector.py with a dictionary-based
approach — no pre-known unit list needed. Accumulates embeddings by
unit_id dynamically using float64 for numerical stability.

Adapted from the NuCLR reference implementation, used under the Apache-2.0
license: https://github.com/nerdslab/nuclr (Arora et al., 2025, NeurIPS;
https://arxiv.org/abs/2512.01199).
"""

from typing import Any

import numpy as np
import torch


class EmbeddingCollector:
    """Dictionary-based per-unit embedding accumulator.

    Accumulates embeddings by unit_id dynamically. Uses float64 for
    numerical stability (matching the original NuCLR implementation).

    Usage::

        collector = EmbeddingCollector(device=torch.device("cuda"))
        for batch in val_loader:
            embeddings = model(batch)  # (N, dim)
            unit_ids = batch.unit_ids  # numpy array of unit IDs
            collector.update(embeddings, unit_ids)
        mean_emb, unit_ids = collector.compute()
    """

    def __init__(self, device: torch.device):
        self.device = device
        self._emb_sums: dict[Any, torch.Tensor] = {}
        self._counts: dict[Any, int] = {}
        self._dim: int | None = None

    def update(self, embeddings: torch.Tensor, unit_ids: np.ndarray) -> None:
        """Accumulate embeddings for the given unit IDs.

        Args:
            embeddings: Embedding tensor of shape ``(N, dim)``.
            unit_ids: Array of unit IDs of length ``N``.
        """
        with torch.inference_mode():
            if self._dim is None:
                self._dim = embeddings.shape[1]

            emb = embeddings.detach().to(dtype=torch.float64, device=self.device)
            for i, uid in enumerate(unit_ids):
                uid_key = uid.item() if hasattr(uid, "item") else uid
                if uid_key in self._emb_sums:
                    self._emb_sums[uid_key] += emb[i]
                    self._counts[uid_key] += 1
                else:
                    self._emb_sums[uid_key] = emb[i].clone()
                    self._counts[uid_key] = 1

    def compute(self) -> tuple[torch.Tensor, np.ndarray]:
        """Compute mean embeddings per unit.

        Returns:
            Tuple of ``(embeddings, unit_ids)`` where embeddings is
            ``(num_units, dim)`` float64 tensor and unit_ids is a sorted
            numpy array.
        """
        if not self._emb_sums:
            raise RuntimeError("No embeddings collected. Call update() first.")

        sorted_keys = sorted(self._emb_sums.keys())
        emb_list = []
        for k in sorted_keys:
            emb_list.append(self._emb_sums[k] / self._counts[k])

        embeddings = torch.stack(emb_list, dim=0)
        unit_ids = np.array(sorted_keys)

        self.reset()
        return embeddings, unit_ids

    def reset(self) -> None:
        """Clear all accumulated embeddings."""
        self._emb_sums.clear()
        self._counts.clear()
        self._dim = None

    def state_dict(self) -> dict:
        """Serialize collector state for checkpointing."""
        return {
            "emb_sums": {k: v.cpu() for k, v in self._emb_sums.items()},
            "counts": dict(self._counts),
            "dim": self._dim,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore collector state from a checkpoint."""
        self._dim = state["dim"]
        self._counts = dict(state["counts"])
        self._emb_sums = {
            k: v.to(dtype=torch.float64, device=self.device) for k, v in state["emb_sums"].items()
        }
