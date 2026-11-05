"""Structure preservation metrics for tokenizer latent spaces.

Two complementary metrics measure how well the latent representation preserves
the relational geometry of original spike data:

**PopulationStructurePreservation** (population level)
    Treats each time window as a single high-dimensional vector containing all
    neurons' spike counts (or latent features) concatenated together.  Builds an
    N×N RDM over windows and asks: "Do windows that were similar across the whole
    population remain similar in the latent space?"  Works with any tokenizer.

**PerUnitStructurePreservation** (inter-unit level)
    Treats each unit as a single vector containing its activity across all time
    windows.  Builds a U×U RDM over units and asks: "Do units that had similar
    temporal activity patterns in the original space remain similar in the latent
    space?"  Only works with per-neuron tokenizers where the latent has shape
    ``[N_windows, n_units, latent_dim]``.

Both provide RSA (primary score) and k-NN neighbourhood overlap.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal, cast

import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import pdist, squareform

from ..core.types import DataType, MetricCategory, MetricDirection, MetricLevel, MetricResult
from .base import BaseMetric, IntermediateRequest, register_metric
from .windowed.structure_preservation import (
    PerUnitStructurePreservationWindowedMixin,
    PopulationStructurePreservationWindowedMixin,
)

# ---------------------------------------------------------------------------
# Type aliases for the configurable distance metrics
# ---------------------------------------------------------------------------
InputMetric = Literal[
    "jaccard",
    "hamming",
    "correlation",
    "cosine",
    "euclidean",
]
LatentMetric = Literal["euclidean", "cosine", "correlation"]

# ---------------------------------------------------------------------------
# Core computation functions
# ---------------------------------------------------------------------------


def smooth_spike_data(
    X: np.ndarray,
    sigma: float,
    unit_time_bins: int | None = None,
) -> np.ndarray:
    """Gaussian-smooth spike data along the time axis.

    For sparse binary spike trains (1ms bins, ~1% firing rate), raw distance
    metrics like Jaccard or correlation are dominated by the 99% of zeros and
    produce nearly uniform RDMs.  Smoothing converts the binary signal into a
    continuous firing-rate estimate at a behaviourally relevant timescale
    (e.g. σ=25 bins ≈ 25ms at 1kHz), making pairwise distances much more
    informative.

    Parameters
    ----------
    X : np.ndarray, shape ``[N, D]``
        Data matrix.  *D* can be a flat concatenation of multiple units'
        time bins (population case) or a single unit's time bins.
    sigma : float
        Standard deviation of the Gaussian kernel **in bins**.
    unit_time_bins : int | None
        If provided, *D* is treated as ``n_units × unit_time_bins`` and
        smoothing is applied **within** each unit segment so that it does
        not bleed across unit boundaries.  If ``None``, the whole row is
        smoothed as one signal.

    Returns
    -------
    np.ndarray, same shape as *X*, dtype float64.
    """
    X = np.asarray(X, dtype=np.float64)
    if unit_time_bins is not None and unit_time_bins < X.shape[1]:
        n_units = X.shape[1] // unit_time_bins
        usable = n_units * unit_time_bins
        # Reshape → (N, n_units, T), smooth per-unit, flatten back
        X_rs = X[:, :usable].reshape(X.shape[0], n_units, unit_time_bins)
        X_rs = gaussian_filter1d(X_rs, sigma=sigma, axis=2)
        out = X.copy()
        out[:, :usable] = X_rs.reshape(X.shape[0], -1)
        return out
    return gaussian_filter1d(X, sigma=sigma, axis=1)


def compute_rdm(X: np.ndarray, metric: str) -> np.ndarray:
    """Compute a representational dissimilarity matrix.

    Parameters
    ----------
    X : np.ndarray, shape ``[N, D]``
        Data matrix (N samples, D features).
    metric : str
        Distance metric (``"jaccard"``, ``"hamming"``, ``"correlation"``,
        ``"cosine"``, ``"euclidean"``).

    Returns
    -------
    np.ndarray, shape ``[N, N]``
        Symmetric pairwise distance matrix.
    """
    X_2d = X.reshape(X.shape[0], -1).astype(np.float64)

    if metric == "jaccard":
        X_bool = X_2d > 0
        dists = pdist(X_bool, metric="jaccard")
        dists = np.nan_to_num(dists, nan=0.0)
    elif metric == "cosine":
        norms = np.linalg.norm(X_2d, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        X_normed = X_2d / norms
        dists = pdist(X_normed, metric="cosine")
        dists = np.nan_to_num(dists, nan=0.0)
    elif metric == "correlation":
        dists = pdist(X_2d, metric="correlation")
        dists = np.nan_to_num(dists, nan=0.0)
    elif metric == "hamming":
        X_bool = X_2d > 0
        dists = pdist(X_bool, metric="hamming")
    elif metric == "euclidean":
        dists = pdist(X_2d, metric="euclidean")
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return squareform(dists)


def rsa_score(
    rdm_input: np.ndarray,
    rdm_latent: np.ndarray,
    method: Literal["spearman", "kendall"] = "spearman",
) -> tuple[float, float]:
    """Representational Similarity Analysis.

    Computes rank correlation between the upper triangles of two RDMs.

    Returns
    -------
    (correlation, p_value)
    """
    idx = np.triu_indices(rdm_input.shape[0], k=1)
    v_input = rdm_input[idx]
    v_latent = rdm_latent[idx]

    if method == "spearman":
        r, p = stats.spearmanr(v_input, v_latent)
    elif method == "kendall":
        r, p = stats.kendalltau(v_input, v_latent)
    else:
        raise ValueError(f"Unknown method: {method}")

    return float(cast(float, r)), float(cast(float, p))


def neighborhood_preservation(
    rdm_input: np.ndarray,
    rdm_latent: np.ndarray,
    k_values: list[int] | None = None,
) -> dict[int, float]:
    """k-NN neighbourhood overlap between two spaces.

    For each k, computes the mean fraction of each point's k nearest
    neighbours in the input space that are also among its k nearest
    neighbours in the latent space.

    Returns
    -------
    dict mapping k -> mean overlap fraction in [0, 1].
    """
    N = rdm_input.shape[0]
    if k_values is None:
        max_k = min(N - 1, 50)
        k_values = sorted(
            set([1, 2, 3, 5] + list(np.unique(np.geomspace(1, max_k, 10).astype(int))))
        )
        k_values = [k for k in k_values if 0 < k < N]

    rdm_in = rdm_input.copy()
    rdm_lat = rdm_latent.copy()
    np.fill_diagonal(rdm_in, np.inf)
    np.fill_diagonal(rdm_lat, np.inf)

    nn_input = np.argsort(rdm_in, axis=1)
    nn_latent = np.argsort(rdm_lat, axis=1)

    results: dict[int, float] = {}
    for k in k_values:
        overlaps = []
        for i in range(N):
            set_in = set(nn_input[i, :k])
            set_lat = set(nn_latent[i, :k])
            overlaps.append(len(set_in & set_lat) / k)
        results[k] = float(np.mean(overlaps))

    return results


_RAGGED_LATENT_WINDOWS_REASON = (
    "Latent windows have unequal token counts (from token overlap trimming), "
    "cannot form uniform matrix for RSA. Fix: set trim_token_overlap=False or "
    "drop_overlapping_windows=True."
)


def _make_ragged_window_result(metric: BaseMetric) -> MetricResult:
    """Return the existing fail-soft result for ragged latent windows."""
    return metric._make_result(
        raw_value=float("nan"),
        metadata={"reason": _RAGGED_LATENT_WINDOWS_REASON},
    )


def _coerce_metric_array(name: str, value: Any, ndim: int) -> np.ndarray:
    """Validate metric inputs as finite float64 arrays with the expected rank."""
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}D array, got shape {array.shape}.")
    if any(size == 0 for size in array.shape):
        raise ValueError(f"{name} must be non-empty, got shape {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or inf values.")
    return array


def _stack_uniform_window_list(values: list[Any]) -> np.ndarray | None:
    """Stack list-valued windows when every element has the same shape."""
    if not values:
        return None

    arrays = [np.asarray(value, dtype=np.float64) for value in values]
    first_shape = arrays[0].shape
    if any(array.shape != first_shape for array in arrays[1:]):
        return None
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("Uniform window list contains NaN or inf values.")
    return np.stack(arrays, axis=0)


def _coerce_uniform_window_array(
    name: str,
    value: Any,
    ndim: int,
) -> np.ndarray | None:
    """Coerce arrays directly, or stack uniform list-valued windows first."""
    if isinstance(value, list):
        stacked = _stack_uniform_window_list(value)
        if stacked is None:
            return None
        value = stacked
    return _coerce_metric_array(name, value, ndim)


def _coerce_population_window_matrix(
    name: str,
    value: Any,
) -> np.ndarray | None:
    """Normalize any dense per-window representation to a 2D [N_windows, D] matrix."""
    if isinstance(value, list):
        stacked = _stack_uniform_window_list(value)
        if stacked is None:
            return None
        value = stacked

    array = np.asarray(value, dtype=np.float64)
    if array.ndim < 2:
        raise ValueError(f"{name} must be at least 2D, got shape {array.shape}.")
    if any(size == 0 for size in array.shape):
        raise ValueError(f"{name} must be non-empty, got shape {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or inf values.")
    return array.reshape(array.shape[0], -1)


def _coerce_per_unit_latent_windows(
    name: str,
    value: Any,
) -> np.ndarray | None:
    """Normalize dense per-unit latents to the existing [N_windows, U, D_flat] form."""
    if isinstance(value, list):
        stacked = _stack_uniform_window_list(value)
        if stacked is None:
            return None
        value = stacked

    array = np.asarray(value, dtype=np.float64)
    if any(size == 0 for size in array.shape):
        raise ValueError(f"{name} must be non-empty, got shape {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or inf values.")
    if array.ndim == 3:
        return array
    if array.ndim == 4:
        return np.transpose(array, (0, 2, 1, 3)).reshape(array.shape[0], array.shape[2], -1)
    raise ValueError(f"{name} must be a 3D or 4D array, got shape {array.shape}.")


def _validate_max_samples(metric_name: str, max_samples: int) -> int:
    """Reject impossible bounded-sample settings loudly."""
    if not isinstance(max_samples, int):
        raise TypeError(
            f"{metric_name} max_samples must be an int, got {type(max_samples).__name__}."
        )
    if max_samples < 1:
        raise ValueError(f"{metric_name} max_samples must be >= 1, got {max_samples}.")
    return max_samples


def _subsample_to_max_samples(
    max_samples: int,
    *arrays: np.ndarray | None,
) -> tuple[np.ndarray | None, ...]:
    """Subsample a seeded-random ``max_samples`` windows, consistently across arrays.

    Uses a fixed-seed RNG so the choice is reproducible, then sorts the indices so
    temporal order is preserved. The random spread (rather than the first
    ``max_samples`` contiguous windows) samples across the whole session, which gives
    these RDM metrics more representative variety than the session's opening alone.

    Only the non-windowed ``compute()`` path can reach this with more than
    ``max_samples`` windows; the windowed ``_finalize_session_state`` path pre-caps
    its buffer to ``max_samples`` during accumulation, so this is a no-op there.
    """
    sample_count: int | None = None
    for array in arrays:
        if array is None:
            continue
        if sample_count is None:
            sample_count = array.shape[0]
        elif array.shape[0] != sample_count:
            raise ValueError(
                "All arrays passed to _subsample_to_max_samples must share the "
                f"same sample count, got {sample_count} and {array.shape[0]}."
            )

    if sample_count is None or sample_count <= max_samples:
        return arrays

    rng = np.random.default_rng(42)
    idx = rng.choice(sample_count, size=max_samples, replace=False)
    idx.sort()

    subsampled: list[np.ndarray | None] = []
    for array in arrays:
        subsampled.append(None if array is None else array[idx])
    return tuple(subsampled)


def _extract_original_window_per_unit(window_intermediates) -> np.ndarray | None:
    """Extract a single original-data window as [1, n_units, window_time]."""
    original = window_intermediates.original
    if original is None:
        return None

    spikes = original.counts_per_bin()
    if spikes is None:
        return None

    array = _coerce_metric_array("original_window", spikes, 2)
    return array[None, :, :]


def _append_window_array(
    session_state: dict[str, Any],
    key: str,
    window_array: np.ndarray,
    max_samples: int,
) -> None:
    """Append a single-window array while enforcing consistent trailing shape."""
    if window_array.shape[0] != 1:
        raise ValueError(
            f"{key} accumulation expects a single window with leading dimension 1, "
            f"got shape {window_array.shape}."
        )

    list_key = f"{key}_buffer"
    shape_key = f"{key}_shape"
    trailing_shape = window_array.shape[1:]
    if shape_key not in session_state:
        session_state[shape_key] = trailing_shape
        session_state[list_key] = []
    elif session_state[shape_key] != trailing_shape:
        raise ValueError(
            f"{key} shape changed across windows: expected {session_state[shape_key]}, "
            f"got {trailing_shape}."
        )

    if len(session_state[list_key]) < max_samples:
        session_state[list_key].append(window_array.copy())


def _compute_population_structure_preservation_result(
    metric: PopulationStructurePreservationWindowedMixin,
    latent: np.ndarray | list | None,
    original: np.ndarray | list | None,
    original_per_unit: np.ndarray | list | None,
) -> MetricResult:
    """Shared contiguous/finalize implementation for population structure preservation."""
    if latent is None or original is None:
        return metric._make_result(raw_value=float("nan"))

    latent_array = _coerce_population_window_matrix(
        "PopulationStructurePreservation.latent_windows",
        latent,
    )
    original_array = _coerce_population_window_matrix(
        "PopulationStructurePreservation.original_windows",
        original,
    )
    if latent_array is None or original_array is None:
        return _make_ragged_window_result(metric)
    if latent_array.shape[0] != original_array.shape[0]:
        raise ValueError(
            "PopulationStructurePreservation requires the same number of latent and "
            f"original windows, got {latent_array.shape[0]} vs {original_array.shape[0]}."
        )

    original_per_unit_array: np.ndarray | None = None
    if original_per_unit is not None:
        original_per_unit_array = _coerce_uniform_window_array(
            "PopulationStructurePreservation.original_windows_per_unit",
            original_per_unit,
            3,
        )
        if original_per_unit_array is None:
            return _make_ragged_window_result(metric)
        if original_per_unit_array.shape[0] != latent_array.shape[0]:
            raise ValueError(
                "PopulationStructurePreservation original_windows_per_unit must match the "
                "latent window count, got "
                f"{original_per_unit_array.shape[0]} vs {latent_array.shape[0]}."
            )
        expected_flat_features = (
            original_per_unit_array.shape[1] * original_per_unit_array.shape[2]
        )
        if original_array.shape[1] != expected_flat_features:
            raise ValueError(
                "PopulationStructurePreservation original window flattening is inconsistent: "
                f"{original_array.shape[1]} vs expected {expected_flat_features}."
            )

    N = latent_array.shape[0]
    if N < metric.MIN_WINDOWS:
        return metric._make_result(
            raw_value=float("nan"),
            metadata={"reason": f"too few windows ({N} < {metric.MIN_WINDOWS})"},
        )

    max_samples = _validate_max_samples(metric.name, metric.params["max_samples"])
    latent_array, original_array, original_per_unit_array = _subsample_to_max_samples(
        max_samples,
        latent_array,
        original_array,
        original_per_unit_array,
    )
    if latent_array is None or original_array is None:
        raise ValueError(
            "PopulationStructurePreservation truncation unexpectedly produced missing arrays."
        )
    N = latent_array.shape[0]

    input_metric: str = metric.params["input_metric"]
    latent_metric: str = metric.params["latent_metric"]
    comparison_method: Literal["spearman", "kendall"] = metric.params["comparison_method"]
    sigma: int = metric.params["smoothing_sigma_bins"]

    input_for_rdm = original_array
    if sigma > 0:
        if original_per_unit_array is not None:
            smoothed = smooth_spike_data(
                original_per_unit_array.reshape(N, -1),
                sigma,
                unit_time_bins=original_per_unit_array.shape[2],
            )
            input_for_rdm = smoothed
        else:
            input_for_rdm = smooth_spike_data(original_array, sigma)

    input_metric_label = f"{input_metric} (σ={sigma})" if sigma > 0 else input_metric

    rdm_input = compute_rdm(input_for_rdm, input_metric)
    rdm_latent = compute_rdm(latent_array, latent_metric)

    rsa_r, rsa_p = rsa_score(rdm_input, rdm_latent, method=comparison_method)
    knn = neighborhood_preservation(rdm_input, rdm_latent)

    return metric._make_result(
        raw_value=rsa_r,
        metadata={
            "rsa_p_value": rsa_p,
            "knn_overlap": knn,
            "knn_mean": float(np.mean(list(knn.values()))),
            "n_samples": N,
            "input_metric": input_metric_label,
            "latent_metric": latent_metric,
            "rdm_input": rdm_input,
            "rdm_latent": rdm_latent,
        },
    )


def _compute_per_unit_structure_preservation_result(
    metric: PerUnitStructurePreservationWindowedMixin,
    latent: np.ndarray | list | None,
    original: np.ndarray | list | None,
) -> MetricResult:
    """Shared contiguous/finalize implementation for per-unit structure preservation."""
    if latent is None or original is None:
        return metric._make_result(raw_value=float("nan"))

    latent_array = _coerce_per_unit_latent_windows(
        "PerUnitStructurePreservation.latent_windows_per_unit",
        latent,
    )
    original_array = _coerce_uniform_window_array(
        "PerUnitStructurePreservation.original_windows_per_unit",
        original,
        3,
    )
    if latent_array is None or original_array is None:
        return _make_ragged_window_result(metric)
    if latent_array.shape[:2] != original_array.shape[:2]:
        raise ValueError(
            "PerUnitStructurePreservation requires matching window/unit dimensions for "
            "latent and original inputs, got "
            f"{latent_array.shape[:2]} vs {original_array.shape[:2]}."
        )

    N, n_units = latent_array.shape[0], latent_array.shape[1]
    if N < metric.MIN_WINDOWS:
        return metric._make_result(
            raw_value=float("nan"),
            metadata={"reason": f"too few windows ({N} < {metric.MIN_WINDOWS})"},
        )

    max_samples = _validate_max_samples(metric.name, metric.params["max_samples"])
    latent_array, original_array = _subsample_to_max_samples(
        max_samples,
        latent_array,
        original_array,
    )
    if latent_array is None or original_array is None:
        raise ValueError(
            "PerUnitStructurePreservation truncation unexpectedly produced missing arrays."
        )
    N = latent_array.shape[0]

    input_metric: str = metric.params["input_metric"]
    latent_metric: str = metric.params["latent_metric"]
    comparison_method: Literal["spearman", "kendall"] = metric.params["comparison_method"]
    sigma: int = metric.params["smoothing_sigma_bins"]

    if sigma > 0:
        original_smoothed = np.zeros_like(original_array)
        for unit_idx in range(n_units):
            original_smoothed[:, unit_idx, :] = smooth_spike_data(
                original_array[:, unit_idx, :],
                sigma,
            )
        original_array = original_smoothed

    original_per_unit = original_array.transpose(1, 0, 2).reshape(n_units, -1)
    latent_per_unit = latent_array.transpose(1, 0, 2).reshape(n_units, -1)

    input_metric_label = f"{input_metric} (σ={sigma})" if sigma > 0 else input_metric

    rdm_input = compute_rdm(original_per_unit, input_metric)
    rdm_latent = compute_rdm(latent_per_unit, latent_metric)

    rsa_r, rsa_p = rsa_score(rdm_input, rdm_latent, method=comparison_method)
    knn = neighborhood_preservation(rdm_input, rdm_latent)

    return metric._make_result(
        raw_value=rsa_r,
        metadata={
            "rsa_p_value": rsa_p,
            "n_samples": N,
            "n_units": n_units,
            "input_metric": input_metric_label,
            "latent_metric": latent_metric,
            "rdm_input": rdm_input,
            "rdm_latent": rdm_latent,
            "knn_overlap": knn,
            "knn_mean": float(np.mean(list(knn.values()))),
        },
    )


# ---------------------------------------------------------------------------
# Registered metric
# ---------------------------------------------------------------------------


@register_metric
class PopulationStructurePreservation(PopulationStructurePreservationWindowedMixin, BaseMetric):
    """Population-level structure preservation between original data and latent space.

    Each time window is represented as a single vector containing all neurons'
    data (original) or all neurons' latent features (latent), flattened together.
    An N×N Representational Dissimilarity Matrix (RDM) is computed over windows
    in both spaces, and RSA measures whether the pairwise distance structure is
    preserved.

    This answers: "Do windows that were similar across the whole neural population
    remain similar in the latent space?"

    Primary score: RSA (Spearman rank correlation between RDMs).
    Metadata includes k-NN neighbourhood overlap.

    Works with any tokenizer that produces per-window latents.
    Requires ``EvalData.latent`` to be populated; returns NaN otherwise.

    See Also
    --------
    PerUnitStructurePreservation : Per-neuron variant that decomposes the
        analysis by unit.
    """

    name = "population_structure_preservation"
    category = MetricCategory.LATENT_QUALITY
    direction = MetricDirection.HIGHER_IS_BETTER
    level = MetricLevel.POPULATION
    value_range = (-1.0, 1.0)
    _required_intermediates: ClassVar[list[str]] = [
        "latent_windows",
        "original_windows",
        "original_windows_per_unit",  # for per-unit smoothing (None when unavailable)
    ]
    _default_params: ClassVar[dict[str, Any]] = {
        "input_metric": "correlation",
        "latent_metric": "euclidean",
        "comparison_method": "spearman",
        "max_samples": 200,
        "smoothing_sigma_bins": 25,  # Gaussian σ in bins; 0 = no smoothing
    }
    aggregation_fallback_policy = "session_aggregated"

    supported_data_types = {DataType.SPIKE_TIMES, DataType.SPIKES, DataType.RATES}
    preferred_data_type = DataType.SPIKES

    MIN_WINDOWS = 10  # Minimum windows for meaningful RDMs

    def get_intermediate_requests(self, data_type: DataType) -> list[IntermediateRequest]:
        return [
            IntermediateRequest.from_dict("latent_windows", {}),
            IntermediateRequest.from_dict("original_windows", {}),
            IntermediateRequest.from_dict("original_windows_per_unit", {}),
        ]

    def compute(self, intermediates: dict[str, Any]) -> MetricResult:
        return _compute_population_structure_preservation_result(
            self,
            intermediates["latent_windows"],
            intermediates["original_windows"],
            intermediates.get("original_windows_per_unit"),
        )


@register_metric
class PerUnitStructurePreservation(PerUnitStructurePreservationWindowedMixin, BaseMetric):
    """Inter-unit structure preservation between original spiking and latent space.

    Builds a U×U RDM where each unit is represented by its activity across all
    time windows (flattened into a single vector).  Compares the RDM in the
    original spike space to the RDM in the latent space via RSA.

    This answers: "Do units that had similar temporal activity patterns in the
    original space remain similar in the latent space?"

    Only works with **per-neuron tokenizers** (e.g., compressed sensing with
    ``compress_mode="per_neuron"``) where the latent preserves a unit axis,
    e.g. ``[N_windows, n_units, latent_dim]`` or
    ``[N_windows, T_tokens, n_units, latent_dim]``. Returns NaN when the
    latent cannot be decomposed per unit (e.g., population tokenizers with
    2D latents).

    Primary score: RSA (Spearman correlation between the U×U RDMs).
    Metadata includes the RDMs (for visualization) and k-NN neighbourhood overlap.

    See Also
    --------
    PopulationStructurePreservation : Compares time windows (not units).
        Asks "Do time windows that were similar remain similar?"
    """

    name = "per_unit_structure_preservation"
    category = MetricCategory.LATENT_QUALITY
    direction = MetricDirection.HIGHER_IS_BETTER
    level = MetricLevel.PER_UNIT
    value_range = (-1.0, 1.0)
    _required_intermediates: ClassVar[list[str]] = [
        "latent_windows_per_unit",
        "original_windows_per_unit",
    ]
    _default_params: ClassVar[dict[str, Any]] = {
        "input_metric": "correlation",
        "latent_metric": "euclidean",
        "comparison_method": "spearman",
        "max_samples": 200,
        "smoothing_sigma_bins": 25,  # Gaussian σ in bins; 0 = no smoothing
    }
    aggregation_fallback_policy = "session_aggregated"

    supported_data_types = {DataType.SPIKE_TIMES, DataType.SPIKES, DataType.RATES}
    preferred_data_type = DataType.SPIKES

    MIN_WINDOWS = 10

    def get_intermediate_requests(self, data_type: DataType) -> list[IntermediateRequest]:
        return [
            IntermediateRequest.from_dict("latent_windows_per_unit", {}),
            IntermediateRequest.from_dict("original_windows_per_unit", {}),
        ]

    def compute(self, intermediates: dict[str, Any]) -> MetricResult:
        return _compute_per_unit_structure_preservation_result(
            self,
            intermediates["latent_windows_per_unit"],
            intermediates["original_windows_per_unit"],
        )
