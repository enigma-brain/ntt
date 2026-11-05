from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator, Sized
from dataclasses import dataclass
from typing import Any, cast


def _replica_seed(
    *,
    base_seed: int,
    recording_id: str,
    start: float,
    end: float,
    replica_id: int,
) -> int:
    """Deterministic seed per (window, replica) for reproducible corruptions."""
    payload = (
        f"{int(base_seed)}|{recording_id}|{float(start):.9f}|{float(end):.9f}|{int(replica_id)}"
    ).encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0x7FFF_FFFF_FFFF_FFFF


@dataclass(frozen=True)
class ReplicaWindowIndex:
    """Index for one replica of one logical source window."""

    recording_id: str
    start: float
    end: float
    replica_id: int


class WindowReplicaBatchSampler:
    """Expand each sampled window into one clean item plus N corrupted replicas."""

    def __init__(
        self,
        base_sampler: Iterable[Any],
        *,
        n_replicas: int,
        max_windows: int | None = None,
        max_eval_seconds: float | None = None,
        effective_window_seconds: float | None = None,
    ) -> None:
        """Store base sampler and replica-count parameters."""
        if n_replicas < 0:
            raise ValueError(f"n_replicas must be >= 0, got {n_replicas}.")
        if max_eval_seconds is not None and max_eval_seconds < 0:
            raise ValueError(
                f"max_eval_seconds must be >= 0 when provided, got {max_eval_seconds}."
            )
        if effective_window_seconds is not None and effective_window_seconds < 0:
            raise ValueError(
                "effective_window_seconds must be >= 0 when provided, got "
                f"{effective_window_seconds}."
            )
        self._base_sampler = base_sampler
        self._n_replicas = int(n_replicas)
        self._max_windows = max_windows
        self._max_eval_seconds = max_eval_seconds
        self._effective_window_seconds = effective_window_seconds

    def _window_duration_seconds(self, index: Any) -> float:
        if self._effective_window_seconds is not None:
            return self._effective_window_seconds
        return max(0.0, float(index.end) - float(index.start))

    def _iter_selected_windows(self) -> Iterator[Any]:
        yielded_windows = 0
        session_seconds: dict[str, float] = {}
        session_window_counts: dict[str, int] = {}

        for index in self._base_sampler:
            if self._max_windows is not None and yielded_windows >= self._max_windows:
                break

            recording_id = str(index.recording_id)
            previous_count = session_window_counts.get(recording_id, 0)
            accumulated_seconds = session_seconds.get(recording_id, 0.0)
            if (
                self._max_eval_seconds is not None
                and previous_count > 0
                and accumulated_seconds >= self._max_eval_seconds
            ):
                continue

            yield index
            yielded_windows += 1
            session_window_counts[recording_id] = previous_count + 1
            session_seconds[recording_id] = accumulated_seconds + self._window_duration_seconds(
                index
            )

    def __iter__(self) -> Iterator[list[ReplicaWindowIndex]]:
        """Yield one list of ReplicaWindowIndex (clean + N replicas) per source window."""
        for index in self._iter_selected_windows():
            recording_id = str(index.recording_id)
            start = float(index.start)
            end = float(index.end)
            yield [
                ReplicaWindowIndex(
                    recording_id=recording_id,
                    start=start,
                    end=end,
                    replica_id=replica_id,
                )
                for replica_id in range(self._n_replicas + 1)
            ]

    def __len__(self) -> int:
        """Number of source windows (respecting max_windows cap)."""
        if self._max_eval_seconds is not None:
            return sum(1 for _ in self._iter_selected_windows())
        if self._max_windows is not None:
            try:
                return min(len(cast(Sized, self._base_sampler)), self._max_windows)
            except TypeError:
                return self._max_windows
        return len(cast(Sized, self._base_sampler))
