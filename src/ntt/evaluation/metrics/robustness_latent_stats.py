"""Latent-stat helpers for robustness metrics.

Computes and caches per-tokenizer/per-config latent standard deviations
(joint scalar and per-dim vector) used by the input_robustness (M1) metric
and the latent_robustness (M2) job. This is a small
implementation detail of the robustness metrics, not a general eval cache.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class LatentStats:
    """Latent-space scale statistics used as normalizers in robustness metrics."""

    sigma_joint: float
    sigma_per_dim: np.ndarray  # shape [D]

    def to_dict(self) -> dict:
        """Return a JSON-safe dict representation."""
        return {
            "sigma_joint": float(self.sigma_joint),
            "sigma_per_dim": np.asarray(self.sigma_per_dim).tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> LatentStats:
        """Reconstruct from a dict produced by :meth:`to_dict`."""
        return cls(
            sigma_joint=float(d["sigma_joint"]),
            sigma_per_dim=np.asarray(d["sigma_per_dim"], dtype=float),
        )


class LatentStatsCache:
    """In-memory (+ optional on-disk) cache of :class:`LatentStats`."""

    def __init__(self, on_disk_root: Path | None = None):
        """Initialize the cache.

        Args:
            on_disk_root: Optional directory for persistent JSON-backed
                cache entries. If ``None``, only the in-memory cache is
                used.
        """
        self._on_disk_root: Path | None = Path(on_disk_root) if on_disk_root is not None else None
        self._mem: dict[str, LatentStats] = {}
        if self._on_disk_root is not None:
            self._on_disk_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _hash_key(key_dict: dict) -> str:
        """Canonical-JSON -> blake2b -> first 16 hex chars."""
        canonical = json.dumps(key_dict, sort_keys=True, default=str).encode("utf-8")
        return hashlib.blake2b(canonical).hexdigest()[:16]

    def _disk_path(self, key_hash: str) -> Path | None:
        """Return the JSON path for ``key_hash`` when disk-backing is enabled."""
        if self._on_disk_root is None:
            return None
        return self._on_disk_root / f"{key_hash}.json"

    def get_or_compute(
        self,
        key_dict: dict,
        latents_fn: Callable[[], np.ndarray],
    ) -> LatentStats:
        """Return cached stats for ``key_dict`` or compute them."""
        key_hash = self._hash_key(key_dict)

        if key_hash in self._mem:
            return self._mem[key_hash]

        disk_path = self._disk_path(key_hash)
        if disk_path is not None and disk_path.exists():
            with disk_path.open("r") as f:
                stats = LatentStats.from_dict(json.load(f))
            self._mem[key_hash] = stats
            return stats

        latents = latents_fn()
        if not isinstance(latents, np.ndarray):
            raise TypeError(f"latents_fn must return np.ndarray, got {type(latents)!r}")
        if latents.ndim != 2:
            raise ValueError(
                f"latents_fn must return a 2D array (N, D); got shape {latents.shape}"
            )
        if latents.size == 0:
            raise ValueError("latents_fn returned an empty array; cannot compute sigma.")

        stats = LatentStats(
            sigma_joint=float(latents.std()),
            sigma_per_dim=latents.std(axis=0),
        )

        self._mem[key_hash] = stats
        if disk_path is not None:
            with disk_path.open("w") as f:
                json.dump(stats.to_dict(), f)

        return stats
