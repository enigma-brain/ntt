"""Population Structure metrics.

NOTE (RATES compatibility): All metrics in this module support DataType.RATES and actually
prefer it for population-level analysis. These are good metrics to use with rate-based
tokenizers like LDNS.
"""

import warnings
from typing import Any

import numpy as np

from ..core.types import (
    DataType,
    MetricCategory,
    MetricDirection,
    MetricLevel,
    MetricResult,
)
from .base import BaseMetric, IntermediateRequest, register_metric
from .windowed.population import (
    PopulationCouplingPreservationWindowedMixin,
    PopulationDimensionalityWindowedMixin,
    SubspaceOverlapWindowedMixin,
)


def _coerce_population_vectors(name: str, value: Any) -> np.ndarray:
    """Validate population vectors as a finite 2D float64 array."""
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {array.shape}.")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty, got shape {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or inf values.")
    return array


def _accumulate_population_moments(
    session_state: dict[str, Any],
    prefix: str,
    population_vectors: np.ndarray,
) -> None:
    """Accumulate first and second moments for row-wise population vectors."""
    feature_count = population_vectors.shape[1]
    sum_key = f"{prefix}_sum"
    sum_outer_key = f"{prefix}_sum_outer"
    sample_count_key = f"{prefix}_n_samples"

    if sum_key not in session_state:
        session_state[sum_key] = np.zeros(feature_count, dtype=np.float64)
        session_state[sum_outer_key] = np.zeros((feature_count, feature_count), dtype=np.float64)
        session_state[sample_count_key] = 0
    elif session_state[sum_key].shape != (feature_count,):
        raise ValueError(
            f"{prefix} feature count changed across windows: expected "
            f"{session_state[sum_key].shape[0]}, got {feature_count}."
        )

    session_state[sum_key] += population_vectors.sum(axis=0)
    session_state[sum_outer_key] += population_vectors.T @ population_vectors
    session_state[sample_count_key] += population_vectors.shape[0]


def _covariance_from_vectors(population_vectors: np.ndarray) -> np.ndarray | None:
    """Compute the unbiased covariance used by the contiguous path."""
    if population_vectors.shape[0] < 2:
        return None
    mean = population_vectors.mean(axis=0)
    centered = population_vectors - mean
    covariance = centered.T @ centered / (population_vectors.shape[0] - 1)
    covariance = np.atleast_2d(covariance).astype(np.float64, copy=False)
    covariance = 0.5 * (covariance + covariance.T)
    if not np.all(np.isfinite(covariance)):
        raise ValueError("Covariance contains NaN or inf values.")
    return covariance


def _covariance_from_session_state(
    session_state: dict[str, Any],
    prefix: str,
) -> tuple[np.ndarray | None, int]:
    """Recover an unbiased covariance matrix from accumulated moments."""
    sample_count_key = f"{prefix}_n_samples"
    n_samples = int(session_state.get(sample_count_key, 0))
    if n_samples == 0:
        raise ValueError(f"No accumulated samples available for {prefix}.")

    sum_key = f"{prefix}_sum"
    sum_outer_key = f"{prefix}_sum_outer"
    if sum_key not in session_state or sum_outer_key not in session_state:
        raise ValueError(f"Missing accumulated moment state for {prefix}.")

    if n_samples < 2:
        return None, n_samples

    sum_vector = np.asarray(session_state[sum_key], dtype=np.float64)
    sum_outer = np.asarray(session_state[sum_outer_key], dtype=np.float64)
    if sum_outer.shape != (sum_vector.shape[0], sum_vector.shape[0]):
        raise ValueError(
            f"Accumulated covariance state for {prefix} is inconsistent: "
            f"sum shape {sum_vector.shape}, sum_outer shape {sum_outer.shape}."
        )
    mean = sum_vector / n_samples
    covariance = (sum_outer - n_samples * np.outer(mean, mean)) / (n_samples - 1)
    covariance = 0.5 * (covariance + covariance.T)
    if not np.all(np.isfinite(covariance)):
        raise ValueError(f"Accumulated covariance for {prefix} contains NaN or inf values.")
    return covariance, n_samples


def _participation_ratio_from_covariance(covariance: np.ndarray | None) -> float:
    """Compute participation ratio from covariance eigenvalues."""
    if covariance is None:
        return float("nan")
    eigenvalues = np.linalg.eigvalsh(np.atleast_2d(covariance))
    positive_eigenvalues = eigenvalues[eigenvalues > 0]
    if positive_eigenvalues.size == 0:
        return float("nan")
    numerator = np.sum(positive_eigenvalues) ** 2
    denominator = np.sum(positive_eigenvalues**2)
    return float(numerator / denominator)


def _principal_subspace_from_covariance(
    covariance: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return PCA-style components and explained-variance ratios from covariance."""
    eigenvalues, eigenvectors = np.linalg.eigh(np.atleast_2d(covariance))
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    total_variance = float(np.sum(eigenvalues))
    if total_variance > 0:
        explained_variance_ratio = eigenvalues[:n_components] / total_variance
    else:
        explained_variance_ratio = np.full(n_components, np.nan, dtype=np.float64)

    components = eigenvectors[:, :n_components].T
    return components, explained_variance_ratio


def _population_couplings_from_covariance(
    covariance: np.ndarray | None,
    n_units: int,
) -> np.ndarray:
    """Compute per-unit population coupling from the full covariance matrix."""
    couplings = np.full(n_units, np.nan, dtype=np.float64)
    if covariance is None:
        return couplings

    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.shape != (n_units, n_units):
        raise ValueError(
            "Population coupling covariance has inconsistent shape: "
            f"expected {(n_units, n_units)}, got {covariance.shape}."
        )

    row_sums = covariance.sum(axis=1)
    total_sum = float(covariance.sum())
    for unit_idx in range(n_units):
        unit_variance = float(covariance[unit_idx, unit_idx])
        population_covariance = float(row_sums[unit_idx] - covariance[unit_idx, unit_idx])
        population_variance = float(
            total_sum - 2.0 * row_sums[unit_idx] + covariance[unit_idx, unit_idx]
        )
        if unit_variance > 0 and population_variance > 0:
            couplings[unit_idx] = population_covariance / np.sqrt(
                unit_variance * population_variance
            )

    return couplings


def _subspace_overlap_from_covariances(
    cov_orig: np.ndarray,
    cov_recon: np.ndarray,
    *,
    max_components: int,
    measure: str,
) -> tuple[float, dict[str, Any]]:
    """Agreement between the leading PCA subspaces of original vs reconstructed population.

    'chordal'  = 1 - sqrt(mean(sin^2 θ)) over the principal angles θ. This expands
                 resolution near perfect overlap, where the mean-cosine measure saturates
                 (the data's effective rank exceeds k, so all top-k directions are real
                 signal that good reconstructions preserve).
    'mean_cosine' = mean canonical correlation (mean cos θ).
    Both are higher-better in [0, 1].
    """
    k = min(max(1, int(max_components)), int(cov_orig.shape[0]))
    comp_orig, var_orig = _principal_subspace_from_covariance(cov_orig, k)
    comp_recon, var_recon = _principal_subspace_from_covariance(cov_recon, k)
    s = np.clip(np.linalg.svd(comp_orig @ comp_recon.T, compute_uv=False), -1.0, 1.0)
    angles = np.arccos(s)
    mean_cosine = float(np.mean(np.cos(angles)))
    chordal = float(1.0 - np.sqrt(np.mean(np.sin(angles) ** 2)))
    if measure == "chordal":
        overlap = chordal
    elif measure == "mean_cosine":
        overlap = mean_cosine
    else:
        raise ValueError(f"Unknown subspace_overlap_measure: {measure!r}")
    return overlap, {
        "principal_angles": angles,
        "subspace_k": k,
        "subspace_overlap_mean_cosine": mean_cosine,
        "subspace_overlap_chordal": chordal,
        "variance_explained_original": var_orig,
        "variance_explained_reconstructed": var_recon,
    }


def _concordance_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Lin's concordance correlation: penalizes scale/location drift, not just rank."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2:
        return float("nan")
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    cov = ((x - mx) * (y - my)).mean()
    denom = vx + vy + (mx - my) ** 2
    if denom <= 0:
        return float("nan")
    return float(2.0 * cov / denom)


@register_metric
class PopulationDimensionality(PopulationDimensionalityWindowedMixin, BaseMetric):
    """Participation ratio as measure of effective dimensionality."""

    name = "dimensionality_error"
    category = MetricCategory.POPULATION_STRUCTURE
    direction = MetricDirection.LOWER_IS_BETTER
    level = MetricLevel.POPULATION
    value_range = (0.0, None)
    _required_intermediates = ["population_vectors_original", "population_vectors_reconstructed"]
    _default_params = {"population_bin_size_ms": 10.0}

    # Works with any binned representation - RATES compatible
    supported_data_types = {DataType.SPIKE_TIMES, DataType.SPIKES, DataType.RATES}
    preferred_data_type = DataType.RATES  # Population metrics work well with rates
    aggregation_fallback_policy = "session_aggregated"

    def get_intermediate_requests(self, data_type: DataType) -> list[IntermediateRequest]:
        pop_params = {"population_bin_size_ms": self.params["population_bin_size_ms"]}
        return [
            IntermediateRequest.from_dict("population_vectors_original", pop_params),
            IntermediateRequest.from_dict("population_vectors_reconstructed", pop_params),
        ]

    def compute(self, intermediates: dict[str, Any]) -> MetricResult:
        pop_orig = _coerce_population_vectors(
            "PopulationDimensionality.population_vectors_original",
            intermediates["population_vectors_original"],
        )
        pop_recon = _coerce_population_vectors(
            "PopulationDimensionality.population_vectors_reconstructed",
            intermediates["population_vectors_reconstructed"],
        )
        if pop_orig.shape != pop_recon.shape:
            raise ValueError(
                "PopulationDimensionality requires original and reconstructed population "
                f"vectors to have identical shape, got {pop_orig.shape} vs {pop_recon.shape}."
            )

        pr_orig = _participation_ratio_from_covariance(_covariance_from_vectors(pop_orig))
        pr_recon = _participation_ratio_from_covariance(_covariance_from_vectors(pop_recon))

        error = np.abs(pr_orig - pr_recon)

        return self._make_result(
            raw_value=error,
            metadata={
                "dimensionality_original": pr_orig,
                "dimensionality_reconstructed": pr_recon,
            },
        )


@register_metric
class SubspaceOverlap(SubspaceOverlapWindowedMixin, BaseMetric):
    """Overlap between top PCA subspaces."""

    name = "subspace_overlap"
    category = MetricCategory.POPULATION_STRUCTURE
    direction = MetricDirection.HIGHER_IS_BETTER
    level = MetricLevel.POPULATION
    # Both supported reductions (chordal, mean_cosine) are higher-better in [0, 1].
    value_range = (0.0, 1.0)
    _required_intermediates = ["population_vectors_original", "population_vectors_reconstructed"]
    _default_params = {
        "pca_n_components": 20,
        "population_bin_size_ms": 10.0,
        "subspace_overlap_measure": "chordal",
    }
    aggregation_fallback_policy = "session_aggregated"

    def get_intermediate_requests(self, data_type: DataType) -> list[IntermediateRequest]:
        pop_params = {"population_bin_size_ms": self.params["population_bin_size_ms"]}
        return [
            IntermediateRequest.from_dict("population_vectors_original", pop_params),
            IntermediateRequest.from_dict("population_vectors_reconstructed", pop_params),
        ]

    # Works with any binned representation - RATES compatible
    supported_data_types = {DataType.SPIKE_TIMES, DataType.SPIKES, DataType.RATES}
    preferred_data_type = DataType.RATES

    def compute(self, intermediates: dict[str, Any]) -> MetricResult:
        pop_orig = _coerce_population_vectors(
            "SubspaceOverlap.population_vectors_original",
            intermediates["population_vectors_original"],
        )
        pop_recon = _coerce_population_vectors(
            "SubspaceOverlap.population_vectors_reconstructed",
            intermediates["population_vectors_reconstructed"],
        )
        if pop_orig.shape != pop_recon.shape:
            raise ValueError(
                "SubspaceOverlap requires original and reconstructed population vectors "
                f"to have identical shape, got {pop_orig.shape} vs {pop_recon.shape}."
            )
        if pop_orig.shape[0] < 2:
            raise ValueError(
                "SubspaceOverlap requires at least two population-vector samples to fit PCA."
            )

        n_components = min(
            self.params["pca_n_components"],
            pop_orig.shape[1],
            pop_recon.shape[1],
        )
        if n_components < 1:
            raise ValueError("SubspaceOverlap requires at least one PCA component.")
        max_supported_components = min(
            pop_orig.shape[0],
            pop_orig.shape[1],
            pop_recon.shape[0],
            pop_recon.shape[1],
        )
        if n_components > max_supported_components:
            raise ValueError(
                "SubspaceOverlap cannot fit PCA with "
                f"n_components={n_components} when only {max_supported_components} "
                "components are supported by the available sample/feature counts."
            )

        cov_orig = _covariance_from_vectors(pop_orig)
        cov_recon = _covariance_from_vectors(pop_recon)
        if cov_orig is None or cov_recon is None:
            raise ValueError(
                "SubspaceOverlap could not compute covariance because fewer than two samples "
                "were provided."
            )
        overlap, meta = _subspace_overlap_from_covariances(
            cov_orig,
            cov_recon,
            max_components=n_components,
            measure=self.params["subspace_overlap_measure"],
        )
        return self._make_result(raw_value=overlap, metadata=meta)


# Population bin sizes already warned about, so per-session/per-window metric
# construction does not flood identical coarse-bin warnings.
_WARNED_COUPLING_BIN_SIZES: set[float] = set()


@register_metric
class PopulationCouplingPreservation(PopulationCouplingPreservationWindowedMixin, BaseMetric):
    """Agreement between original and reconstructed population coupling values.

    Per neuron, coupling = correlation between that neuron and the population sum
    (excluding itself); this metric reports the agreement (CCC by default, or Pearson)
    between the original and reconstructed coupling vectors.

    Resolution caveat — pin to 1 ms. At coarse population bins (>=5 ms) population
    coupling is dominated by shared rate co-modulation, which a rate-matched Poisson
    baseline reproduces almost perfectly: the Poisson floor saturates at 0.97-0.999
    (1 ms: ~0.6-0.85 in motor cortex), leaving no headroom above the baseline and a
    near-degenerate normalization denominator. The default is therefore 1 ms, and the
    constructor warns above 5 ms. (Strongly stimulus-driven areas such as V4 stay
    saturated even at 1 ms; the ``insufficient_baseline_separation`` gate in
    aggregation/comparison.py is the backstop there.)
    """

    name = "population_coupling"
    category = MetricCategory.POPULATION_STRUCTURE
    direction = MetricDirection.HIGHER_IS_BETTER
    level = MetricLevel.POPULATION
    value_range = (-1.0, 1.0)
    _required_intermediates = ["population_vectors_original", "population_vectors_reconstructed"]
    _default_params = {"population_bin_size_ms": 1.0, "coupling_agreement": "ccc"}
    aggregation_fallback_policy = "session_aggregated"

    # Above this population bin (ms) the Poisson floor saturates; warn the caller.
    _max_recommended_bin_size_ms = 5.0

    def __init__(self, **params: Any):
        super().__init__(**params)
        bin_ms = self.params["population_bin_size_ms"]
        if bin_ms > self._max_recommended_bin_size_ms and bin_ms not in _WARNED_COUPLING_BIN_SIZES:
            _WARNED_COUPLING_BIN_SIZES.add(bin_ms)
            warnings.warn(
                f"population_coupling at population_bin_size_ms={bin_ms:g} (>"
                f"{self._max_recommended_bin_size_ms:g} ms): at coarse bins the "
                "rate-matched Poisson floor saturates (0.97-0.999), so the normalized "
                "score is near-degenerate and dominated by amplification noise. Prefer "
                "1 ms; see the PopulationCouplingPreservation docstring.",
                UserWarning,
                stacklevel=2,
            )

    def get_intermediate_requests(self, data_type: DataType) -> list[IntermediateRequest]:
        pop_params = {"population_bin_size_ms": self.params["population_bin_size_ms"]}
        return [
            IntermediateRequest.from_dict("population_vectors_original", pop_params),
            IntermediateRequest.from_dict("population_vectors_reconstructed", pop_params),
        ]

    # Works with any binned representation - RATES compatible
    supported_data_types = {DataType.SPIKE_TIMES, DataType.SPIKES, DataType.RATES}
    preferred_data_type = DataType.RATES

    def compute(self, intermediates: dict[str, Any]) -> MetricResult:
        pop_orig = _coerce_population_vectors(
            "PopulationCouplingPreservation.population_vectors_original",
            intermediates["population_vectors_original"],
        )
        pop_recon = _coerce_population_vectors(
            "PopulationCouplingPreservation.population_vectors_reconstructed",
            intermediates["population_vectors_reconstructed"],
        )
        if pop_orig.shape != pop_recon.shape:
            raise ValueError(
                "PopulationCouplingPreservation requires original and reconstructed "
                "population vectors to have identical shape, got "
                f"{pop_orig.shape} vs {pop_recon.shape}."
            )

        coupling_orig = _population_couplings_from_covariance(
            _covariance_from_vectors(pop_orig),
            pop_orig.shape[1],
        )
        coupling_recon = _population_couplings_from_covariance(
            _covariance_from_vectors(pop_recon),
            pop_recon.shape[1],
        )

        # Agreement between coupling values
        valid_mask = ~np.isnan(coupling_orig) & ~np.isnan(coupling_recon)
        if np.sum(valid_mask) > 2:
            xo, xr = coupling_orig[valid_mask], coupling_recon[valid_mask]
            pearson = float(np.corrcoef(xo, xr)[0, 1])
            mode = self.params["coupling_agreement"]
            if mode == "ccc":
                preservation = _concordance_correlation(xo, xr)
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
            },
        )
