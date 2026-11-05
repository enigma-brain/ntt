"""Windowed (session-aggregated) implementations for structure-preservation metrics.

PopulationStructurePreservation and PerUnitStructurePreservation buffer per-window
latent/original arrays (capped at ``max_samples``) and finalize by feeding the
concatenated buffers through the same ``_compute_*_result`` functions the
non-windowed ``compute()`` uses. Those result functions and all the RDM/RSA
helpers stay in :mod:`ntt.evaluation.metrics.structure_preservation`; the mixins
reach them via local import to avoid an import cycle.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from ...core.types import MetricResult
from ._shared import MetricMixinBase, metric_requests_by_key


class PopulationStructurePreservationWindowedMixin(MetricMixinBase):
    """Session-aggregated population (time-window RDM) structure preservation."""

    MIN_WINDOWS: ClassVar[int]  # defined by the concrete metric

    aggregation_fallback_policy = "session_aggregated"

    def _accumulate_window(
        self,
        session_state: dict[str, Any],
        window_intermediates,
    ) -> None:
        from ..structure_preservation import (
            _append_window_array,
            _coerce_population_window_matrix,
            _extract_original_window_per_unit,
            _make_ragged_window_result,
            _validate_max_samples,
        )

        if "terminal_result" in session_state:
            return

        requests = metric_requests_by_key(self)
        latent = window_intermediates.get(requests["latent_windows"])
        original_per_unit = _extract_original_window_per_unit(window_intermediates)
        if latent is None or original_per_unit is None:
            session_state["terminal_result"] = self._make_result(raw_value=float("nan"))
            return

        latent_array = _coerce_population_window_matrix(
            "PopulationStructurePreservation.latent_windows",
            latent,
        )
        if latent_array is None:
            session_state["terminal_result"] = _make_ragged_window_result(self)
            return
        if latent_array.shape[0] != 1:
            raise ValueError(
                "PopulationStructurePreservation._accumulate_window expected a single "
                f"latent window, got shape {latent_array.shape}."
            )
        original_array = original_per_unit.reshape(1, -1)
        if original_array.shape[0] != 1:
            raise ValueError(
                "PopulationStructurePreservation._accumulate_window expected a single "
                f"original window, got shape {original_array.shape}."
            )

        max_samples = _validate_max_samples(self.name, self.params["max_samples"])
        _append_window_array(session_state, "latent_windows", latent_array, max_samples)
        _append_window_array(session_state, "original_windows", original_array, max_samples)
        _append_window_array(
            session_state,
            "original_windows_per_unit",
            original_per_unit,
            max_samples,
        )
        session_state["n_windows_seen"] = int(session_state.get("n_windows_seen", 0)) + 1

    def _finalize_session_state(self, session_state: dict[str, Any]) -> MetricResult:
        from ..structure_preservation import _compute_population_structure_preservation_result

        if "terminal_result" in session_state:
            return session_state["terminal_result"]
        if int(session_state.get("n_windows_seen", 0)) == 0:
            raise ValueError(
                "PopulationStructurePreservation._finalize_session_state called with "
                "no accumulated windows."
            )

        latent_buffer = session_state.get("latent_windows_buffer")
        original_buffer = session_state.get("original_windows_buffer")
        original_per_unit_buffer = session_state.get("original_windows_per_unit_buffer")
        if not latent_buffer or not original_buffer or not original_per_unit_buffer:
            raise ValueError(
                "PopulationStructurePreservation missing buffered windows at finalize."
            )

        return _compute_population_structure_preservation_result(
            self,
            np.concatenate(latent_buffer, axis=0),
            np.concatenate(original_buffer, axis=0),
            np.concatenate(original_per_unit_buffer, axis=0),
        )


class PerUnitStructurePreservationWindowedMixin(MetricMixinBase):
    """Session-aggregated per-unit (unit RDM) structure preservation."""

    MIN_WINDOWS: ClassVar[int]  # defined by the concrete metric

    aggregation_fallback_policy = "session_aggregated"

    def _accumulate_window(
        self,
        session_state: dict[str, Any],
        window_intermediates,
    ) -> None:
        from ..structure_preservation import (
            _append_window_array,
            _coerce_per_unit_latent_windows,
            _extract_original_window_per_unit,
            _make_ragged_window_result,
            _validate_max_samples,
        )

        if "terminal_result" in session_state:
            return

        requests = metric_requests_by_key(self)
        latent = window_intermediates.get(requests["latent_windows_per_unit"])
        original = _extract_original_window_per_unit(window_intermediates)
        if latent is None or original is None:
            session_state["terminal_result"] = self._make_result(raw_value=float("nan"))
            return

        latent_array = _coerce_per_unit_latent_windows(
            "PerUnitStructurePreservation.latent_windows_per_unit",
            latent,
        )
        if latent_array is None:
            session_state["terminal_result"] = _make_ragged_window_result(self)
            return
        if latent_array.shape[0] != 1:
            raise ValueError(
                "PerUnitStructurePreservation._accumulate_window expected a single "
                f"latent window, got shape {latent_array.shape}."
            )
        if latent_array.shape[1] != original.shape[1]:
            raise ValueError(
                "PerUnitStructurePreservation._accumulate_window requires the same "
                "unit count for latent and original windows, got "
                f"{latent_array.shape[1]} vs {original.shape[1]}."
            )

        max_samples = _validate_max_samples(self.name, self.params["max_samples"])
        _append_window_array(session_state, "latent_windows_per_unit", latent_array, max_samples)
        _append_window_array(session_state, "original_windows_per_unit", original, max_samples)
        session_state["n_windows_seen"] = int(session_state.get("n_windows_seen", 0)) + 1

    def _finalize_session_state(self, session_state: dict[str, Any]) -> MetricResult:
        from ..structure_preservation import _compute_per_unit_structure_preservation_result

        if "terminal_result" in session_state:
            return session_state["terminal_result"]
        if int(session_state.get("n_windows_seen", 0)) == 0:
            raise ValueError(
                "PerUnitStructurePreservation._finalize_session_state called with "
                "no accumulated windows."
            )

        latent_buffer = session_state.get("latent_windows_per_unit_buffer")
        original_buffer = session_state.get("original_windows_per_unit_buffer")
        if not latent_buffer or not original_buffer:
            raise ValueError("PerUnitStructurePreservation missing buffered windows at finalize.")

        return _compute_per_unit_structure_preservation_result(
            self,
            np.concatenate(latent_buffer, axis=0),
            np.concatenate(original_buffer, axis=0),
        )
