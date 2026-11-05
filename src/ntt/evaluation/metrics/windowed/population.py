"""Windowed (session-aggregated) implementations for population-structure metrics.

PopulationDimensionality, SubspaceOverlap and PopulationCouplingPreservation
accumulate population-vector moments per window and finalize from the pooled
covariance. The covariance/PCA/coupling helpers are shared with the non-windowed
``compute()`` and stay in :mod:`ntt.evaluation.metrics.population`; the mixins
reach them via local import to avoid an import cycle.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...core.types import MetricResult
from ._shared import MetricMixinBase, metric_requests_by_key


class PopulationDimensionalityWindowedMixin(MetricMixinBase):
    """Session-aggregated participation-ratio difference from pooled covariance."""

    aggregation_fallback_policy = "session_aggregated"

    def _accumulate_window(
        self,
        session_state: dict[str, Any],
        window_intermediates,
    ) -> None:
        from ..population import _accumulate_population_moments, _coerce_population_vectors

        requests = metric_requests_by_key(self)
        pop_orig = _coerce_population_vectors(
            "PopulationDimensionality.population_vectors_original",
            window_intermediates.get(requests["population_vectors_original"]),
        )
        pop_recon = _coerce_population_vectors(
            "PopulationDimensionality.population_vectors_reconstructed",
            window_intermediates.get(requests["population_vectors_reconstructed"]),
        )
        if pop_orig.shape != pop_recon.shape:
            raise ValueError(
                "PopulationDimensionality._accumulate_window requires matching original "
                f"and reconstructed shapes, got {pop_orig.shape} vs {pop_recon.shape}."
            )

        _accumulate_population_moments(session_state, "population_vectors_original", pop_orig)
        _accumulate_population_moments(
            session_state,
            "population_vectors_reconstructed",
            pop_recon,
        )
        session_state["n_windows_seen"] = int(session_state.get("n_windows_seen", 0)) + 1

    def _finalize_session_state(self, session_state: dict[str, Any]) -> MetricResult:
        from ..population import (
            _covariance_from_session_state,
            _participation_ratio_from_covariance,
        )

        if int(session_state.get("n_windows_seen", 0)) == 0:
            raise ValueError(
                "PopulationDimensionality._finalize_session_state called with no "
                "accumulated windows."
            )

        cov_orig, _ = _covariance_from_session_state(session_state, "population_vectors_original")
        cov_recon, _ = _covariance_from_session_state(
            session_state,
            "population_vectors_reconstructed",
        )
        pr_orig = _participation_ratio_from_covariance(cov_orig)
        pr_recon = _participation_ratio_from_covariance(cov_recon)

        return self._make_result(
            raw_value=float(np.abs(pr_orig - pr_recon)),
            metadata={
                "dimensionality_original": pr_orig,
                "dimensionality_reconstructed": pr_recon,
                "n_windows_seen": int(session_state["n_windows_seen"]),
            },
        )


class SubspaceOverlapWindowedMixin(MetricMixinBase):
    """Session-aggregated principal-subspace overlap from pooled covariance."""

    aggregation_fallback_policy = "session_aggregated"

    def _accumulate_window(
        self,
        session_state: dict[str, Any],
        window_intermediates,
    ) -> None:
        from ..population import _accumulate_population_moments, _coerce_population_vectors

        requests = metric_requests_by_key(self)
        pop_orig = _coerce_population_vectors(
            "SubspaceOverlap.population_vectors_original",
            window_intermediates.get(requests["population_vectors_original"]),
        )
        pop_recon = _coerce_population_vectors(
            "SubspaceOverlap.population_vectors_reconstructed",
            window_intermediates.get(requests["population_vectors_reconstructed"]),
        )
        if pop_orig.shape != pop_recon.shape:
            raise ValueError(
                "SubspaceOverlap._accumulate_window requires matching original and "
                f"reconstructed shapes, got {pop_orig.shape} vs {pop_recon.shape}."
            )

        _accumulate_population_moments(session_state, "population_vectors_original", pop_orig)
        _accumulate_population_moments(
            session_state,
            "population_vectors_reconstructed",
            pop_recon,
        )
        session_state["n_windows_seen"] = int(session_state.get("n_windows_seen", 0)) + 1

    def _finalize_session_state(self, session_state: dict[str, Any]) -> MetricResult:
        from ..population import (
            _covariance_from_session_state,
            _subspace_overlap_from_covariances,
        )

        if int(session_state.get("n_windows_seen", 0)) == 0:
            raise ValueError(
                "SubspaceOverlap._finalize_session_state called with no accumulated windows."
            )

        cov_orig, n_samples_orig = _covariance_from_session_state(
            session_state,
            "population_vectors_original",
        )
        cov_recon, n_samples_recon = _covariance_from_session_state(
            session_state,
            "population_vectors_reconstructed",
        )
        if cov_orig is None or cov_recon is None:
            raise ValueError(
                "SubspaceOverlap requires at least two accumulated samples on both original "
                f"and reconstructed sides, got {n_samples_orig} and {n_samples_recon}."
            )

        n_components = min(
            self.params["pca_n_components"],
            cov_orig.shape[0],
            cov_recon.shape[0],
        )
        if n_components < 1:
            raise ValueError("SubspaceOverlap requires at least one PCA component.")
        max_supported_components = min(
            n_samples_orig,
            cov_orig.shape[0],
            n_samples_recon,
            cov_recon.shape[0],
        )
        if n_components > max_supported_components:
            raise ValueError(
                "SubspaceOverlap cannot fit PCA with "
                f"n_components={n_components} when only {max_supported_components} "
                "components are supported by the accumulated sample/feature counts."
            )

        overlap, meta = _subspace_overlap_from_covariances(
            cov_orig,
            cov_recon,
            max_components=n_components,
            measure=self.params["subspace_overlap_measure"],
        )
        meta["n_windows_seen"] = int(session_state["n_windows_seen"])
        return self._make_result(raw_value=overlap, metadata=meta)


class PopulationCouplingPreservationWindowedMixin(MetricMixinBase):
    """Session-aggregated population-coupling correlation from pooled covariance."""

    aggregation_fallback_policy = "session_aggregated"

    def _accumulate_window(
        self,
        session_state: dict[str, Any],
        window_intermediates,
    ) -> None:
        from ..population import _accumulate_population_moments, _coerce_population_vectors

        requests = metric_requests_by_key(self)
        pop_orig = _coerce_population_vectors(
            "PopulationCouplingPreservation.population_vectors_original",
            window_intermediates.get(requests["population_vectors_original"]),
        )
        pop_recon = _coerce_population_vectors(
            "PopulationCouplingPreservation.population_vectors_reconstructed",
            window_intermediates.get(requests["population_vectors_reconstructed"]),
        )
        if pop_orig.shape != pop_recon.shape:
            raise ValueError(
                "PopulationCouplingPreservation._accumulate_window requires matching "
                f"original and reconstructed shapes, got {pop_orig.shape} vs {pop_recon.shape}."
            )

        _accumulate_population_moments(session_state, "population_vectors_original", pop_orig)
        _accumulate_population_moments(
            session_state,
            "population_vectors_reconstructed",
            pop_recon,
        )
        session_state["n_windows_seen"] = int(session_state.get("n_windows_seen", 0)) + 1

    def _finalize_session_state(self, session_state: dict[str, Any]) -> MetricResult:
        from ..population import (
            _concordance_correlation,
            _covariance_from_session_state,
            _population_couplings_from_covariance,
        )

        if int(session_state.get("n_windows_seen", 0)) == 0:
            raise ValueError(
                "PopulationCouplingPreservation._finalize_session_state called with no "
                "accumulated windows."
            )

        cov_orig, _ = _covariance_from_session_state(session_state, "population_vectors_original")
        cov_recon, _ = _covariance_from_session_state(
            session_state,
            "population_vectors_reconstructed",
        )
        n_units_orig = int(np.asarray(session_state["population_vectors_original_sum"]).shape[0])
        n_units_recon = int(
            np.asarray(session_state["population_vectors_reconstructed_sum"]).shape[0]
        )
        if n_units_orig != n_units_recon:
            raise ValueError(
                "PopulationCouplingPreservation accumulated different unit counts across "
                f"original and reconstructed sides: {n_units_orig} vs {n_units_recon}."
            )

        coupling_orig = _population_couplings_from_covariance(cov_orig, n_units_orig)
        coupling_recon = _population_couplings_from_covariance(cov_recon, n_units_recon)

        valid_mask = ~np.isnan(coupling_orig) & ~np.isnan(coupling_recon)
        if np.sum(valid_mask) > 2:
            xo, xr = coupling_orig[valid_mask], coupling_recon[valid_mask]
            pearson = float(np.corrcoef(xo, xr)[0, 1])
            mode = self.params["coupling_agreement"]
            if mode == "ccc":
                preservation = float(_concordance_correlation(xo, xr))
            elif mode == "pearson":
                preservation = pearson
            else:
                raise ValueError(f"Unknown coupling_agreement: {mode!r}")
        else:
            preservation, pearson = float("nan"), float("nan")

        return self._make_result(
            raw_value=preservation,
            metadata={
                "coupling_original": coupling_orig,
                "coupling_reconstructed": coupling_recon,
                "coupling_pearson": pearson,
                "n_windows_seen": int(session_state["n_windows_seen"]),
            },
        )
