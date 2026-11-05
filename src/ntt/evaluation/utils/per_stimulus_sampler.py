"""Per-stimulus sampler geometry for stimulus-aligned evaluation."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
from torch.utils.data import Sampler

from ntt.core.data.components.behavior_transforms import (
    extend_interval_ends_with_post_event_window,
)

type StimulusGeometry = tuple[str, float, float, Any]
type StimulusBounds = tuple[float, float]


class PerStimulusSampler(Sampler[int]):
    """Yield one stable integer index per stimulus window."""

    def __init__(
        self,
        dataset,
        *,
        session_ids: list[str],
        interval_source: str = "stimuli",
        grouping_field: str = "image_id",
        interval_start_lag_ms: float = 0.0,
        interval_end_lag_ms: float | None = None,
        fixed_window_duration_ms: float | None = None,
        cross_stim_padding_ms: float = 20.0,
        label_extension_ms: float | None = None,
    ) -> None:
        """Cache per-stimulus window geometry across the requested sessions.

        Exactly one of ``interval_end_lag_ms`` or ``fixed_window_duration_ms``
        must be provided:

        - ``interval_end_lag_ms``: window = ``[stim_start + start_lag,
          stim_end + end_lag]``, with ``effective_end`` clamped at the next
          stim start plus ``cross_stim_padding_ms``. Stimulus-length-dependent.
        - ``fixed_window_duration_ms``: window = ``[stim_start + start_lag,
          stim_start + start_lag + fixed_window_duration_ms]``. The window
          length does not depend on stimulus duration. We assert that the
          overlap into the next stimulus is at most ``cross_stim_padding_ms``;
          callers should size ``fixed_window_duration_ms`` to fit within
          ``(next_stim_start - this_stim_start) + cross_stim_padding_ms``.
        """
        if (interval_end_lag_ms is None) == (fixed_window_duration_ms is None):
            raise ValueError(
                "PerStimulusSampler: exactly one of `interval_end_lag_ms` or "
                "`fixed_window_duration_ms` must be set (the other must be None). "
                f"Got interval_end_lag_ms={interval_end_lag_ms!r}, "
                f"fixed_window_duration_ms={fixed_window_duration_ms!r}."
            )
        if fixed_window_duration_ms is not None and fixed_window_duration_ms <= 0:
            raise ValueError(
                "PerStimulusSampler: `fixed_window_duration_ms` must be > 0, "
                f"got {fixed_window_duration_ms!r}."
            )
        self.dataset = dataset
        self.session_ids = list(session_ids)
        self.interval_source = interval_source
        self.grouping_field = grouping_field
        self.interval_start_lag_s = interval_start_lag_ms / 1000.0
        self.interval_end_lag_s = (
            None if interval_end_lag_ms is None else interval_end_lag_ms / 1000.0
        )
        self.fixed_window_duration_s = (
            None if fixed_window_duration_ms is None else fixed_window_duration_ms / 1000.0
        )
        self.cross_stim_padding_s = cross_stim_padding_ms / 1000.0
        # Label extension defaults: in lag mode, fall back to `interval_end_lag_ms`
        # (preserves prior behavior). In fixed-window mode, `interval_end_lag_ms`
        # is not defined, so an explicit `label_extension_ms` is required.
        if label_extension_ms is None:
            if interval_end_lag_ms is None:
                raise ValueError(
                    "PerStimulusSampler: `label_extension_ms` must be set explicitly "
                    "when using `fixed_window_duration_ms` (there is no "
                    "`interval_end_lag_ms` to fall back to)."
                )
            resolved_label_extension_ms = interval_end_lag_ms
        else:
            resolved_label_extension_ms = label_extension_ms
        self.label_extension_s = resolved_label_extension_ms / 1000.0
        self._geometries: list[StimulusGeometry] = []
        self._stim_bounds: list[StimulusBounds] = []
        self._label_bounds: list[StimulusBounds] = []
        for session_id in self.session_ids:
            session_geometries, session_stim_bounds, session_label_bounds = self._resolve_session(
                session_id
            )
            self._geometries.extend(session_geometries)
            self._stim_bounds.extend(session_stim_bounds)
            self._label_bounds.extend(session_label_bounds)

    def _resolve_session(
        self, session_id: str
    ) -> tuple[list[StimulusGeometry], list[StimulusBounds], list[StimulusBounds]]:
        """Resolve and validate one session's effective stimulus windows."""
        data = self.dataset._get_data_object(session_id)
        if hasattr(data, self.interval_source):
            interval_entry = getattr(data, self.interval_source)
        elif hasattr(data, "behavioral_intervals"):
            interval_entry = data.behavioral_intervals[self.interval_source]
        else:
            raise AttributeError(
                f"Session '{session_id}' is missing interval source '{self.interval_source}'."
            )

        starts_raw = (
            interval_entry["start"] if isinstance(interval_entry, dict) else interval_entry.start
        )
        ends_raw = (
            interval_entry["end"] if isinstance(interval_entry, dict) else interval_entry.end
        )
        groups_raw = (
            interval_entry[self.grouping_field]
            if isinstance(interval_entry, dict)
            else getattr(interval_entry, self.grouping_field)
        )

        starts = np.asarray(starts_raw, dtype=np.float64)
        ends = np.asarray(ends_raw, dtype=np.float64)
        groups = np.asarray(groups_raw)

        if starts.ndim != 1 or ends.ndim != 1 or groups.ndim != 1:
            raise ValueError(
                f"Session '{session_id}' interval arrays must be 1-D; got "
                f"starts={starts.shape}, ends={ends.shape}, groups={groups.shape}."
            )
        if len(starts) != len(ends) or len(starts) != len(groups):
            raise ValueError(
                f"Session '{session_id}' interval arrays differ in length: "
                f"starts={len(starts)}, ends={len(ends)}, groups={len(groups)}."
            )

        order = np.argsort(starts, kind="stable")
        starts = starts[order]
        ends = ends[order]
        groups = groups[order]

        for idx in range(1, len(starts)):
            if starts[idx] < ends[idx - 1]:
                raise ValueError(
                    "overlap: "
                    f"session={session_id}, pair=({idx - 1}, {idx}), "
                    f"ends[{idx - 1}]={ends[idx - 1]}, starts[{idx}]={starts[idx]}"
                )

        geometries: list[StimulusGeometry] = []
        stim_bounds: list[StimulusBounds] = []
        label_bounds: list[StimulusBounds] = []
        label_ends = extend_interval_ends_with_post_event_window(
            starts,
            ends,
            end_offset=self.label_extension_s,
        )
        for idx, start in enumerate(starts):
            effective_start = float(start + self.interval_start_lag_s)
            if idx > 0:
                effective_start = max(effective_start, float(ends[idx - 1]))

            if self.fixed_window_duration_s is not None:
                # Fixed-length window anchored at stim_start + start_lag.
                # `effective_start` may have been clamped against the previous
                # stim's end above; we still measure the window from the raw
                # `start + start_lag` to keep length deterministic.
                anchor_start = float(start + self.interval_start_lag_s)
                effective_end = anchor_start + self.fixed_window_duration_s
                if idx + 1 < len(starts):
                    next_start = float(starts[idx + 1])
                    overlap = effective_end - next_start
                    # Allow a tiny numerical slack so a window that lands
                    # exactly on the padding boundary doesn't trip the check.
                    if overlap > self.cross_stim_padding_s + 1e-9:
                        raise ValueError(
                            "fixed_window overlap with next stimulus exceeds "
                            "cross_stim_padding_ms: "
                            f"session={session_id}, pair=({idx}, {idx + 1}), "
                            f"this_stim_start={float(start):.6f}, "
                            f"next_stim_start={next_start:.6f}, "
                            f"window_end={effective_end:.6f}, "
                            f"overlap_s={overlap:.6f}, "
                            f"cross_stim_padding_s={self.cross_stim_padding_s:.6f}. "
                            "Either shorten `fixed_window_duration_ms` or raise "
                            "`cross_stim_padding_ms` (currently capped intentionally)."
                        )
            else:
                assert self.interval_end_lag_s is not None  # guarded by __init__
                effective_end = float(ends[idx] + self.interval_end_lag_s)
                if idx + 1 < len(starts):
                    effective_end = min(
                        effective_end,
                        float(starts[idx + 1] + self.cross_stim_padding_s),
                    )

            group_value: Any = groups[idx]
            if isinstance(group_value, np.generic):
                group_value = group_value.item()

            geometries.append((session_id, effective_start, effective_end, group_value))
            stim_bounds.append((float(start), float(ends[idx])))
            label_bounds.append((float(start), float(label_ends[idx])))

        if len(geometries) == 0:
            self._raise_for_zero_intervals(session_id)

        return geometries, stim_bounds, label_bounds

    def _recording_id_for_session(self, session_id: str) -> str:
        """Resolve the dataset recording id that backs a sampler session id."""
        return session_id

    def _dataset_ignore_splits_flag(self, recording_id: str) -> bool | None:
        """Best-effort lookup of a recording's ignore_splits flag."""
        if hasattr(self.dataset, "_should_ignore_splits"):
            return bool(self.dataset._should_ignore_splits(recording_id))

        config: Any | None = None
        if hasattr(self.dataset, "_get_data_config"):
            config = self.dataset._get_data_config(recording_id)
        elif hasattr(self.dataset, "recording_dict"):
            config = self.dataset.recording_dict.get(recording_id, {}).get("config")

        if config is None or not hasattr(config, "get"):
            return None

        ignore_splits = config.get("ignore_splits")
        if ignore_splits is None:
            return None
        return bool(ignore_splits)

    def _raise_for_zero_intervals(self, session_id: str) -> None:
        """Raise a loud, actionable error when a session resolves no intervals."""
        recording_id = self._recording_id_for_session(session_id)
        ignore_splits = self._dataset_ignore_splits_flag(recording_id)
        if ignore_splits is False:
            raise ValueError(
                f"PerStimulusSampler resolved zero intervals for session "
                f"{session_id!r}. This is almost certainly because the recording's "
                f"`ignore_splits` config flag is `False` and the requested "
                f"stimulus intervals fall outside the active split (train/val/test). "
                f"Set `ignore_splits=True` for this recording in the data config "
                f"(see `src/ntt/core/data/components/config.py:135`), or run with "
                f"a split that contains the stimuli."
            )
        raise ValueError(
            f"PerStimulusSampler resolved zero intervals for session "
            f"{session_id!r} (interval_source={self.interval_source!r}). "
            f"Either the recording has no stimuli under this source, or there's "
            f"a config / metadata mismatch."
        )

    def __iter__(self) -> Iterator[int]:
        """Yield integer indices into the cached geometry table."""
        return iter(range(len(self)))

    def __len__(self) -> int:
        """Return the number of cached stimulus windows."""
        return len(self._geometries)

    def window_geometries(self) -> list[tuple[float, float]]:
        """Return effective `(start, end)` pairs in iteration order."""
        return [(start, end) for _, start, end, _ in self._geometries]

    def geometry_for(self, idx: int) -> StimulusGeometry:
        """Return the full cached geometry tuple for one sampler index."""
        return self._geometries[idx]

    def stim_bounds_for(self, idx: int) -> StimulusBounds:
        """Return the raw `(stim_start, stim_end)` bounds for one sampler index."""
        return self._stim_bounds[idx]

    def label_bounds_for(self, idx: int) -> StimulusBounds:
        """Return the `(stim_start, extended_end)` bounds used for token labels."""
        return self._label_bounds[idx]
