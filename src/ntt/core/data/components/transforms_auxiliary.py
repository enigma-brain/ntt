from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from torch_brain.transforms import Compose

from ntt.core.data.components.corruption import (
    AddSpec,
    DropAndAddSpec,
    DropNeuronsSpec,
    DropSpec,
    JitterSpec,
    PermuteNeuronsSpec,
    SpikeTimeCorruption,
    SpikeTimeSpec,
    UnitAxisCorruption,
    UnitAxisSpec,
    WindowShiftSpec,
)
from ntt.core.data.components.transforms import (
    CreateBinnedSpikeCountsArray,
    DataToModelInput,
)


class ReplicaAwareSpikeTimeCorruption:
    """Insert spike-time corruption into a normal single-sample transform chain."""

    def __init__(self, spec: Any):
        self._spec = spec

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(spec={self._spec!r})"

    def __call__(self, data: Any) -> Any:
        replica_id = int(getattr(data, "_ntt_replica_id", 0))
        seed = int(getattr(data, "_ntt_replica_seed", 0))

        if replica_id == 0:
            if isinstance(self._spec, WindowShiftSpec):
                centered = WindowShiftSpec(
                    max_shift_ms=self._spec.max_shift_ms,
                    analysis_window_ms=self._spec.analysis_window_ms,
                    source_window_ms=self._spec.source_window_ms,
                    fixed_shift_ms=0.0,
                )
                return SpikeTimeCorruption(centered, seed=seed)(data)
            return data

        return SpikeTimeCorruption(self._spec, seed=seed)(data)


class ReplicaAwareUnitAxisCorruption:
    """Apply unit-axis corruption only to non-clean replicas."""

    def __init__(self, spec: Any):
        self._spec = spec

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(spec={self._spec!r})"

    def __call__(self, data: Any) -> Any:
        replica_id = int(getattr(data, "_ntt_replica_id", 0))
        if replica_id == 0:
            return data

        if getattr(self._spec, "deterministic_per_replica", False):
            # Per-replica_id-only seed: same mask/permutation for every window
            # of a given replica, which is what ACG/CCG averaging needs.
            seed = int.from_bytes(
                hashlib.blake2b(f"unit_axis:{replica_id}".encode(), digest_size=8).digest(),
                "big",
            )
        else:
            seed = int(getattr(data, "_ntt_replica_seed", 0))
        return UnitAxisCorruption(self._spec, seed=seed)(data)


def build_transform_with_replica_corruption(base_transform: Any, spec: Any) -> Compose:
    """Insert one corruption stage into an existing Compose pipeline."""

    chain = list(getattr(base_transform, "transforms", []))
    if not chain:
        raise ValueError("Expected a Compose transform with a non-empty .transforms list.")

    is_spike_time = isinstance(spec, SpikeTimeSpec)
    is_unit_axis = isinstance(spec, UnitAxisSpec)
    if not (is_spike_time or is_unit_axis):
        raise TypeError(f"Unsupported corruption spec type: {type(spec)}")

    rebuilt: list[Any] = []
    inserted = False

    for transform in chain:
        if (
            is_spike_time
            and not inserted
            and isinstance(transform, (CreateBinnedSpikeCountsArray, DataToModelInput))
        ):
            rebuilt.append(ReplicaAwareSpikeTimeCorruption(spec))
            inserted = True

        rebuilt.append(transform)

        if is_unit_axis and isinstance(transform, CreateBinnedSpikeCountsArray):
            rebuilt.append(ReplicaAwareUnitAxisCorruption(spec))
            inserted = True

    if not inserted:
        corruption_kind = "spike-time" if is_spike_time else "unit-axis"
        raise ValueError(
            f"Could not place {corruption_kind} corruption into the provided transform chain."
        )

    return Compose(rebuilt)


@dataclass(frozen=True)
class CorruptionSpecLibrary:
    jitter: Any
    drop: Any
    add: Any
    drop_and_add: Any
    window_shift: Any
    permute_neurons: Any
    drop_neurons: Any


def default_corruption_specs(
    *,
    jitter_k_ms: float = 5.0,
    jitter_p: float = 0.5,
    drop_fraction: float = 0.1,
    add_fraction: float = 0.1,
    drop_and_add_fraction: float = 0.1,
    window_shift_ms: float = 10.0,
    analysis_window_ms: float = 1000.0,
    source_window_ms: float | None = None,
    drop_neurons_fraction: float = 0.1,
) -> CorruptionSpecLibrary:
    source_ms = (
        analysis_window_ms + 2.0 * abs(window_shift_ms)
        if source_window_ms is None
        else source_window_ms
    )

    return CorruptionSpecLibrary(
        jitter=JitterSpec(k_ms=jitter_k_ms, p=jitter_p),
        drop=DropSpec(fraction=drop_fraction),
        add=AddSpec(fraction=add_fraction),
        drop_and_add=DropAndAddSpec(fraction=drop_and_add_fraction),
        window_shift=WindowShiftSpec(
            max_shift_ms=window_shift_ms,
            analysis_window_ms=analysis_window_ms,
            source_window_ms=source_ms,
        ),
        permute_neurons=PermuteNeuronsSpec(),
        drop_neurons=DropNeuronsSpec(fraction=drop_neurons_fraction),
    )
