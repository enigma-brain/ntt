"""Corruption transforms for robustness analysis.

Localized copy of the qualitative-robustness corruption helpers so notebook
and transform composition code inside ``ntt`` does not depend on top-level
``scripts`` imports.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from temporaldata import Data, Interval, IrregularTimeSeries

from ntt.core.data.components.dynamic import dyn


@dataclass(frozen=True)
class JitterSpec:
    k_ms: float
    p: float = 0.5
    name: str = "jitter"


@dataclass(frozen=True)
class DropSpec:
    fraction: float
    name: str = "drop"


@dataclass(frozen=True)
class AddSpec:
    fraction: float
    min_weight: float = 0.05
    name: str = "add"


@dataclass(frozen=True)
class WindowShiftSpec:
    """Crop a longer source window to an analysis-length sub-window, shifted.

    Default (random) mode: each call to `apply_window_shift` draws a shift
    uniformly in `[-max_shift_ms, +max_shift_ms]` using the supplied `rng`.
    Each replica thus gets its own (seeded, reproducible) shift.

    Explicit (fixed) mode: set `fixed_shift_ms=X` to always apply exactly `X`
    ms of shift, ignoring the rng. Used internally to build the centered
    clean-replica (fixed_shift_ms=0.0) and available for sweeps that want a
    fixed schedule of shift values.

    The caller must provide a source window longer than the analysis window
    by at least `2 * max_shift_ms` ms. The sub-window starts at
    `source_start + D/2 + shift_s`, where D is the total extra length and
    `shift_s` is either the draw or `fixed_shift_ms`. For `fixed_shift_ms==0`
    the sub-window is centered.
    """

    max_shift_ms: float
    analysis_window_ms: float
    source_window_ms: float  # must be >= analysis + 2*max_shift_ms
    fixed_shift_ms: float | None = None
    name: str = "window_shift"

    def __post_init__(self) -> None:
        """Validate shift/source consistency."""
        if self.max_shift_ms < 0:
            raise ValueError(f"max_shift_ms must be >= 0, got {self.max_shift_ms}")
        if self.fixed_shift_ms is not None:
            if abs(self.fixed_shift_ms) > self.max_shift_ms + 1e-9:
                raise ValueError(
                    f"|fixed_shift_ms|={abs(self.fixed_shift_ms)} exceeds "
                    f"max_shift_ms={self.max_shift_ms}"
                )
        extra = self.source_window_ms - self.analysis_window_ms
        if extra < 2 * self.max_shift_ms - 1e-9:
            raise ValueError(
                f"source_window_ms={self.source_window_ms} not long enough "
                f"for max_shift_ms={self.max_shift_ms} + "
                f"analysis_window_ms={self.analysis_window_ms}. "
                f"Need source >= analysis + 2*max_shift."
            )


@dataclass(frozen=True)
class DropAndAddSpec:
    fraction: float
    min_weight: float = 0.05
    name: str = "drop_and_add"


@dataclass(frozen=True)
class PermuteNeuronsSpec:
    deterministic_per_replica: bool = False
    name: str = "permute_neurons"


@dataclass(frozen=True)
class DropNeuronsSpec:
    fraction: float
    deterministic_per_replica: bool = False
    name: str = "drop_neurons"


SpikeTimeSpec = JitterSpec | DropSpec | AddSpec | DropAndAddSpec | WindowShiftSpec
UnitAxisSpec = PermuteNeuronsSpec | DropNeuronsSpec
AnySpec = SpikeTimeSpec | UnitAxisSpec


def _clone_data(data: Data) -> Data:
    return copy.copy(data)


def _replace_spikes(
    data: Data,
    timestamps: np.ndarray,
    unit_index: np.ndarray,
) -> Data:
    new_data = _clone_data(data)
    # temporaldata annotates ``**kwargs`` of the time-series constructors as dicts of arrays
    new_data.spikes = IrregularTimeSeries(
        timestamps=np.asarray(timestamps, dtype=np.float64),
        unit_index=np.asarray(unit_index, dtype=np.int64),  # pyright: ignore[reportArgumentType]
        domain=dyn(data).spikes.domain,
    )
    return new_data


def apply_jitter(data: Data, spec: JitterSpec, rng: np.random.Generator) -> Data:
    spikes = dyn(data).spikes
    ts = np.array(spikes.timestamps, dtype=np.float64, copy=True)
    ui = np.array(spikes.unit_index, dtype=np.int64, copy=True)
    n = ts.size
    if n == 0:
        return _replace_spikes(data, ts, ui)

    k_s = spec.k_ms * 1e-3
    mask = rng.random(n) < spec.p
    offsets = rng.uniform(-k_s, k_s, size=n)
    ts[mask] += offsets[mask]

    start, end = spikes.domain.start, spikes.domain.end
    if hasattr(start, "__len__"):
        start_v = float(np.min(np.asarray(start)))
        end_v = float(np.max(np.asarray(end)))
    else:
        start_v = float(start)
        end_v = float(end)
    eps = 1e-9
    np.clip(ts, start_v, end_v - eps, out=ts)

    order = np.argsort(ts, kind="stable")
    return _replace_spikes(data, ts[order], ui[order])


def apply_drop(data: Data, spec: DropSpec, rng: np.random.Generator) -> Data:
    spikes = dyn(data).spikes
    ts = np.asarray(spikes.timestamps)
    ui = np.asarray(spikes.unit_index)
    n = ts.size
    if n == 0 or spec.fraction <= 0:
        return _replace_spikes(data, ts, ui)
    keep = rng.random(n) >= spec.fraction
    return _replace_spikes(data, ts[keep], ui[keep])


def _unit_weights(unit_index: np.ndarray, n_units: int, min_weight: float) -> np.ndarray:
    counts = np.bincount(unit_index, minlength=n_units).astype(np.float64)
    total = counts.sum()
    if total > 0:
        p_prop = counts / total
    else:
        p_prop = np.full(n_units, 1.0 / max(n_units, 1))
    p_floor = min_weight / max(n_units, 1)
    p = p_floor + (1.0 - min_weight) * p_prop
    p /= p.sum()
    return p


def apply_add(data: Data, spec: AddSpec, rng: np.random.Generator) -> Data:
    spikes = dyn(data).spikes
    ts = np.asarray(spikes.timestamps, dtype=np.float64)
    ui = np.asarray(spikes.unit_index, dtype=np.int64)
    n_units = int(len(dyn(data).units))
    n = ts.size
    n_add = int(round(spec.fraction * n))
    if n_add <= 0:
        return _replace_spikes(data, ts, ui)

    weights = _unit_weights(ui, n_units, spec.min_weight)
    new_ui = rng.choice(n_units, size=n_add, p=weights).astype(np.int64)
    start_v, end_v = _domain_bounds(spikes.domain)
    new_ts = rng.uniform(start_v, end_v, size=n_add).astype(np.float64)

    combined_ts = np.concatenate([ts, new_ts])
    combined_ui = np.concatenate([ui, new_ui])
    order = np.argsort(combined_ts, kind="stable")
    return _replace_spikes(data, combined_ts[order], combined_ui[order])


def apply_drop_and_add(data: Data, spec: DropAndAddSpec, rng: np.random.Generator) -> Data:
    dropped = apply_drop(data, DropSpec(fraction=spec.fraction), rng)
    return apply_add(dropped, AddSpec(fraction=spec.fraction, min_weight=spec.min_weight), rng)


def _domain_bounds(domain: Interval) -> tuple[float, float]:
    start = np.asarray(dyn(domain).start)
    end = np.asarray(dyn(domain).end)
    return float(np.min(start)), float(np.max(end))


def apply_window_shift(data: Data, spec: WindowShiftSpec, rng: np.random.Generator) -> Data:
    """Crop a longer source window down to analysis-length, shifted in time.

    Shift selection:
      - If `spec.fixed_shift_ms is None` (default), draw shift uniformly from
        `[-max_shift_ms, +max_shift_ms]` using `rng` — reproducible per seed.
      - Else use exactly `spec.fixed_shift_ms` and ignore `rng`.

    Uses `Data.slice(new_start, new_end, reset_origin=False)` so timestamps
    stay absolute and downstream transforms (CreateBinnedSpikeCountsArray)
    use the sliced domain.
    """
    t0, _ = _domain_bounds(dyn(data).spikes.domain)
    analysis_s = spec.analysis_window_ms * 1e-3
    source_s = spec.source_window_ms * 1e-3
    if spec.fixed_shift_ms is not None:
        shift_ms = float(spec.fixed_shift_ms)
    else:
        shift_ms = float(rng.uniform(-spec.max_shift_ms, spec.max_shift_ms))
    shift_s = shift_ms * 1e-3
    extra = source_s - analysis_s
    # (Validation of source vs max_shift already enforced in __post_init__;
    # here we only guard against a fixed shift magnitude larger than extra/2,
    # which __post_init__ also rules out, but assert-in-depth.)
    assert extra >= 2 * abs(shift_s) - 1e-9, (
        f"WindowShift: source={spec.source_window_ms} too short for "
        f"shift={shift_ms}ms + analysis={spec.analysis_window_ms}ms"
    )
    # Snap new_start to a 1 ms grid relative to t0 to avoid sub-bin
    # fractional offsets that would later trip downstream bin-count
    # assertions (a binned recon over [new_start, new_end] yields a
    # bin count that's compared to round((abs_end-abs_start)*1000) — the
    # two only match when (extra/2 + shift_s) is an integer ms). Hardcoded
    # to 1 ms because every input source in this codebase bins at 1 ms;
    # if a non-1ms input bin is ever introduced, plumb input_bin_ms into
    # WindowShiftSpec and replace the hardcoded constant.
    offset_s = round((extra / 2.0 + shift_s) * 1000.0) / 1000.0
    new_start = t0 + offset_s
    new_end = new_start + analysis_s
    return data.slice(new_start, new_end, reset_origin=False)


SPIKE_TIME_FNS: dict[type, Any] = {
    JitterSpec: apply_jitter,
    DropSpec: apply_drop,
    AddSpec: apply_add,
    DropAndAddSpec: apply_drop_and_add,
    WindowShiftSpec: apply_window_shift,
}


class SpikeTimeCorruption:
    def __init__(self, spec: SpikeTimeSpec, seed: int):
        if type(spec) not in SPIKE_TIME_FNS:
            raise ValueError(f"Unknown spike-time spec: {type(spec)}")
        self.spec = spec
        self.seed = int(seed)

    def __repr__(self) -> str:
        return f"SpikeTimeCorruption(spec={self.spec!r}, seed={self.seed})"

    def __call__(self, data: Data) -> Data:
        rng = np.random.default_rng(self.seed)
        fn = SPIKE_TIME_FNS[type(self.spec)]
        return fn(data, self.spec, rng)


def apply_permute_neurons(
    counts: np.ndarray, spec: PermuteNeuronsSpec, rng: np.random.Generator
) -> np.ndarray:
    del spec
    n_units = counts.shape[-1]
    perm = rng.permutation(n_units)
    return counts[..., perm]


def apply_drop_neurons(
    counts: np.ndarray, spec: DropNeuronsSpec, rng: np.random.Generator
) -> np.ndarray:
    """Drop-and-replace a random subset of neurons with surviving-unit copies.

    For each "dropped" unit ``u`` we replace its full spike train with a
    bit-identical copy of a randomly-chosen *surviving* (non-dropped)
    unit's spike train. Donors are sampled with replacement, so multiple
    dropped slots can receive copies of the same donor unit. This
    destroys the dropped units' identity entirely (their original
    pattern is gone) while keeping the population's overall activity
    structure populated by real neural traces.

    When ``round(fraction * n_units) < 1`` (e.g. fraction=0.05 with 10
    units) this is a no-op rather than an error: the caller asked for a
    zero-strength corruption and gets one. Returns a copy of the input
    in that case so callers' "always a fresh array" expectation holds.

    Raises ``ValueError`` when ``fraction`` would leave no surviving
    units to draw donors from (e.g. ``fraction=1.0``).
    """
    n_units = counts.shape[-1]
    n_drop = int(round(spec.fraction * n_units))
    if n_drop < 1:
        # No-op: the spec resolves to "drop nothing" given the unit count.
        # Returning a copy keeps callers' "always a fresh array" expectation.
        return counts.copy()
    if n_drop >= n_units:
        raise ValueError("DropNeuronsSpec.fraction must leave at least one surviving unit")
    idx = rng.choice(n_units, size=n_drop, replace=False)
    survivors = np.setdiff1d(np.arange(n_units), idx, assume_unique=False)
    # Draw one donor per dropped unit, with replacement — multiple
    # dropped slots may end up holding copies of the same surviving
    # unit's spike train.
    donors = rng.choice(survivors, size=n_drop, replace=True)
    out = counts.copy()
    # ``counts`` is expected to have units on axis -1 ([..., T, U]).
    # Copy each donor's full spike train into the corresponding dropped
    # slot, overwriting the dropped unit's original trace completely.
    for u_dropped, u_donor in zip(idx, donors, strict=True):
        out[..., u_dropped] = counts[..., u_donor]
    return out


UNIT_AXIS_FNS: dict[type, Any] = {
    PermuteNeuronsSpec: apply_permute_neurons,
    DropNeuronsSpec: apply_drop_neurons,
}


class UnitAxisCorruption:
    def __init__(self, spec: UnitAxisSpec, seed: int):
        if type(spec) not in UNIT_AXIS_FNS:
            raise ValueError(f"Unknown unit-axis spec: {type(spec)}")
        self.spec = spec
        self.seed = int(seed)

    def __repr__(self) -> str:
        return f"UnitAxisCorruption(spec={self.spec!r}, seed={self.seed})"

    def __call__(self, data: Data) -> Data:
        if not hasattr(data, "binned_spike_counts"):
            raise ValueError(
                "UnitAxisCorruption requires binned_spike_counts on the Data "
                "object; apply CreateBinnedSpikeCountsArray first."
            )
        rng = np.random.default_rng(self.seed)
        fn = UNIT_AXIS_FNS[type(self.spec)]
        old_rts = dyn(data).binned_spike_counts
        new_counts = fn(np.asarray(old_rts.counts), self.spec, rng)
        from temporaldata import RegularTimeSeries

        new_data = _clone_data(data)
        dyn(new_data).binned_spike_counts = RegularTimeSeries(
            sampling_rate=old_rts.sampling_rate,
            domain=old_rts.domain,
            counts=new_counts,
        )
        gm = getattr(old_rts, "gap_mask", None)
        if gm is not None:
            dyn(new_data).binned_spike_counts.gap_mask = gm
        return new_data


@dataclass
class CorruptionSweep:
    name: str
    specs: list[AnySpec] = field(default_factory=list)
    x_values: list[float] = field(default_factory=list)
    x_label: str = ""
    source_window_sec_override: float | None = None

    def __post_init__(self) -> None:
        if len(self.specs) != len(self.x_values):
            raise ValueError("specs and x_values must have the same length.")
