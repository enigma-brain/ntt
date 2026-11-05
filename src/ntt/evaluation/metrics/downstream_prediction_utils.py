"""Utilities for downstream prediction metrics: phase filtering, sample reshaping."""

import logging
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

from ntt.core.data.components.behavior_transforms import MOVEMENT_PHASE_LABELS

# NOTE: do NOT move this import to module level. It triggers a circular import:
# evaluation.config → config.metric_config → metrics.__init__ → downstream_prediction
# → downstream_prediction_utils → evaluation.utils.__init__ → validation_collation
# → evaluation.config (still initializing). The root cause is that __init__.py files
# eagerly re-export everything; a proper fix would be to make them lean.
# from ntt.evaluation.utils.alignment import resample_behavioral_data  # lazy-imported below


def flatten_latents(latents: Any) -> np.ndarray:
    """Convert flat/windowed latent inputs to a 2D ``[n_samples, latent_dim]`` array."""
    if isinstance(latents, list):
        flat_windows = []
        for window in latents:
            arr = np.asarray(window)
            if arr.ndim == 0:
                continue
            if arr.ndim == 1:
                arr = arr[:, None]
            else:
                arr = arr.reshape(arr.shape[0], -1)
            flat_windows.append(arr)
        if not flat_windows:
            return np.empty((0, 0), dtype=np.float64)
        return np.concatenate(flat_windows, axis=0)

    arr = np.asarray(latents)
    if arr.ndim == 0:
        return np.empty((0, 0), dtype=np.float64)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim == 2:
        return arr
    if arr.ndim == 4:
        n_windows, t_tokens, u_units, d_latent = arr.shape
        return arr.reshape(n_windows * t_tokens, u_units * d_latent)
    return arr.reshape(arr.shape[0] * arr.shape[1], -1)


def _canonicalize_latents_with_timestamps(
    latents: Any,
    token_timestamps: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Normalize latent layouts to ``[n_samples, n_features]`` using timestamp count.

    Timestamp cardinality resolves otherwise-ambiguous layouts such as ``[N, F]``
    versus raw discrete token grids ``[W, T]``.
    """
    timestamps = np.asarray(token_timestamps)
    n_timestamps = int(len(timestamps))

    def _empty_metadata(layout: str) -> dict[str, Any]:
        return {
            "latent_layout": layout,
            "window_token_counts": [],
            "is_discrete_index_like": False,
        }

    def _reshape_window_tokens(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 1:
            return arr[:, None]
        return arr.reshape(arr.shape[0], -1)

    def _reshape_single_window_sample(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 1:
            return arr[None, :]
        return arr.reshape(1, -1)

    if isinstance(latents, list):
        windows = [np.asarray(window) for window in latents if np.asarray(window).ndim != 0]
        if not windows:
            return np.empty((0, 0), dtype=np.float64), _empty_metadata("empty")

        token_counts_by_first_axis = [int(window.shape[0]) for window in windows]
        if sum(token_counts_by_first_axis) == n_timestamps:
            X = np.concatenate([_reshape_window_tokens(window) for window in windows], axis=0)
            is_discrete_index_like = bool(
                all(window.ndim == 1 for window in windows)
                and all(np.issubdtype(window.dtype, np.integer) for window in windows)
            )
            metadata = {
                "latent_layout": "ragged_window_tokens",
                "window_token_counts": token_counts_by_first_axis,
                "is_discrete_index_like": is_discrete_index_like,
            }
            return X, metadata

        if len(windows) == n_timestamps:
            X = np.concatenate(
                [_reshape_single_window_sample(window) for window in windows], axis=0
            )
            is_discrete_index_like = bool(
                all(window.size == 1 for window in windows)
                and all(np.issubdtype(window.dtype, np.integer) for window in windows)
            )
            metadata = {
                "latent_layout": "ragged_single_token_windows",
                "window_token_counts": [1] * len(windows),
                "is_discrete_index_like": is_discrete_index_like,
            }
            return X, metadata

        raise ValueError(
            f"Could not align ragged latent windows with {n_timestamps} timestamps: "
            f"sum(first_axis_sizes)={sum(token_counts_by_first_axis)}, "
            f"n_windows={len(windows)}"
        )

    arr = np.asarray(latents)
    if arr.ndim == 0:
        return np.empty((0, 0), dtype=np.float64), _empty_metadata("empty")

    if arr.ndim == 1:
        if n_timestamps == arr.shape[0]:
            metadata = {
                "latent_layout": "one_dimensional_tokens",
                "window_token_counts": None,
                "is_discrete_index_like": bool(np.issubdtype(arr.dtype, np.integer)),
            }
            return arr[:, None], metadata
        if n_timestamps == 1:
            metadata = {
                "latent_layout": "one_dimensional_feature_vector",
                "window_token_counts": [1],
                "is_discrete_index_like": False,
            }
            return arr[None, :], metadata
        raise ValueError(
            f"Could not align 1D latents of length {arr.shape[0]} with {n_timestamps} timestamps"
        )

    if arr.ndim == 2:
        n_rows, n_cols = arr.shape
        if n_timestamps == n_rows:
            metadata = {
                "latent_layout": "sample_feature_matrix",
                "window_token_counts": [1] * n_rows,
                "is_discrete_index_like": bool(
                    np.issubdtype(arr.dtype, np.integer) and n_cols == 1
                ),
            }
            return arr, metadata
        if n_timestamps == n_rows * n_cols:
            metadata = {
                "latent_layout": "token_grid_scalar_feature",
                "window_token_counts": [n_cols] * n_rows,
                "is_discrete_index_like": bool(np.issubdtype(arr.dtype, np.integer)),
            }
            return arr.reshape(n_rows * n_cols, 1), metadata
        raise ValueError(
            f"Could not align 2D latents of shape {arr.shape} with {n_timestamps} timestamps"
        )

    if arr.ndim == 3:
        n_windows, n_tokens_or_features, n_features = arr.shape
        if n_timestamps == n_windows:
            metadata = {
                "latent_layout": "window_feature_tensor",
                "window_token_counts": [1] * n_windows,
                "is_discrete_index_like": False,
            }
            return arr.reshape(n_windows, n_tokens_or_features * n_features), metadata
        if n_timestamps == n_windows * n_tokens_or_features:
            metadata = {
                "latent_layout": "windowed_token_features",
                "window_token_counts": [n_tokens_or_features] * n_windows,
                "is_discrete_index_like": False,
            }
            return arr.reshape(n_windows * n_tokens_or_features, n_features), metadata
        raise ValueError(
            f"Could not align 3D latents of shape {arr.shape} with {n_timestamps} timestamps"
        )

    if arr.ndim == 4:
        n_windows, t_tokens, u_units, d_latent = arr.shape
        if n_timestamps != n_windows * t_tokens:
            raise ValueError(
                f"Could not align 4D latents of shape {arr.shape} with {n_timestamps} timestamps"
            )
        metadata = {
            "latent_layout": "windowed_unit_latents",
            "window_token_counts": [t_tokens] * n_windows,
            "is_discrete_index_like": False,
        }
        return arr.reshape(n_windows * t_tokens, u_units * d_latent), metadata

    if n_timestamps == arr.shape[0]:
        metadata = {
            "latent_layout": f"{arr.ndim}d_sample_features",
            "window_token_counts": [1] * arr.shape[0],
            "is_discrete_index_like": False,
        }
        return arr.reshape(arr.shape[0], -1), metadata

    if n_timestamps == arr.shape[0] * arr.shape[1]:
        metadata = {
            "latent_layout": f"{arr.ndim}d_windowed_features",
            "window_token_counts": [arr.shape[1]] * arr.shape[0],
            "is_discrete_index_like": False,
        }
        return arr.reshape(arr.shape[0] * arr.shape[1], -1), metadata

    raise ValueError(f"Could not align latent shape {arr.shape} with {n_timestamps} timestamps")


def filter_by_behavioral_field(
    X: np.ndarray, behavioral_data: dict[str, np.ndarray], field: str, include_values: list
):
    """Filter X and behavioral_data to keep only time points with values in include_values."""

    if field == "movement_phase":
        invalid_phases = [p for p in include_values if p not in MOVEMENT_PHASE_LABELS]
        if invalid_phases:
            raise ValueError(
                f"Unknown phase name(s): {invalid_phases}. "
                f"Valid phases: {list(MOVEMENT_PHASE_LABELS.keys())}"
            )

        assert "movement_phase" in behavioral_data, (
            "behavioral_data must contain 'movement_phase' to filter by phases"
        )
        if isinstance(include_values[0], str):
            include_values = [MOVEMENT_PHASE_LABELS[p] for p in include_values]

    include_array = np.array(include_values)

    if field not in behavioral_data:
        raise ValueError(
            f"behavioral_data does not contain '{field}'. Available: "
            f"{list(behavioral_data.keys())}"
        )
    field_values = behavioral_data[field]
    assert len(field_values) == len(X), f"{field} length {len(field_values)} != X length {len(X)}"

    mask = np.isin(field_values, include_array)

    filtered_X = X[mask]
    if len(filtered_X) == 0:
        raise ValueError(f"No time points found for {field} in include_values {include_array}")
    filtered_behavioral = {k: v[mask] for k, v in behavioral_data.items()}
    return filtered_X, filtered_behavioral


def _normalize_include_values(
    field: str,
    include_values: list[Any],
) -> np.ndarray:
    """Normalize filter values to the representation stored in behavioral arrays.

    This exists because downstream preprocessing needs the exact same filter
    logic for both behavioral arrays and the parallel timestamp array. The
    public helper ``filter_by_behavioral_field`` only returns filtered X and
    behavioral data, so this private helper centralizes the field-specific
    value normalization used to build a reusable boolean mask.
    """
    if field == "movement_phase":
        invalid_phases = [p for p in include_values if p not in MOVEMENT_PHASE_LABELS]
        if invalid_phases:
            raise ValueError(
                f"Unknown phase name(s): {invalid_phases}. "
                f"Valid phases: {list(MOVEMENT_PHASE_LABELS.keys())}"
            )
        if include_values and isinstance(include_values[0], str):
            include_values = [MOVEMENT_PHASE_LABELS[p] for p in include_values]
    return np.asarray(include_values)


def _behavioral_filter_mask(
    behavioral_data: dict[str, np.ndarray],
    field: str,
    include_values: list[Any],
) -> np.ndarray:
    """Return the boolean mask implied by ``filter_by_behavioral_field``.

    This is used inside downstream preprocessing so we can apply one shared
    mask to latents, behavior, and timestamps, instead of filtering behavior
    and timestamps through two different code paths.
    """
    if field not in behavioral_data:
        raise ValueError(
            f"behavioral_data does not contain '{field}'. Available: "
            f"{list(behavioral_data.keys())}"
        )
    normalized_values = _normalize_include_values(field, include_values)
    return np.isin(behavioral_data[field], normalized_values)


def chunk_by_temporal_window(
    latents: np.ndarray,
    timestamps: np.ndarray,
    token_bin_size_ms: float,
    chunk_window_ms: float = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Group tokens into fixed-duration temporal windows by concatenation.


    Replaces ``reshape_to_min_num_features``. The primary parameter is a
    temporal window in ms rather than a feature count, so the prediction
    task is comparable across tokenizers with different bin sizes.

    Both token assignment and (later) behavioral alignment use the same
    index-space grid: chunk *j* starts at token index
    ``int(chunk_window_ms * j / token_bin_size_ms)``.  This ensures
    identical temporal coverage regardless of tokenizer phase.

    Parameters
    ----------
    latents : np.ndarray
        Shape ``(T, U*d)`` — must already be 2-D (flattened).
    timestamps : np.ndarray
        Shape ``(T,)`` — timestamp of each token.
    token_bin_size_ms : float
        Temporal extent of one token in ms.
    chunk_window_ms : float
        Desired temporal window per chunk in ms (default 100).

    Returns
    -------
    chunked_latents : np.ndarray
        ``(n_windows, tokens_per_chunk * feature_dim)``
    chunk_timestamps : np.ndarray
        Timestamp of the first token in each chunk, shape ``(n_windows,)``.
    chunk_indices : np.ndarray
        Original chunk index *j* for each window, shape ``(n_windows,)``.
        Used by ``align_behavioral_to_chunks`` to locate behavioral data.
    tokens_per_chunk : int
        Number of tokens concatenated per chunk.
    """
    assert latents.ndim == 2, f"Expected 2-D latents (T, features), got shape {latents.shape}"
    assert len(timestamps) == len(latents), (
        f"timestamps length {len(timestamps)} != latents length {len(latents)}"
    )

    # Each behavioral chunk tiles the latent stream into a whole number of
    # tokens, so chunk_window_ms must be an integer multiple of the tokenizer's
    # token_bin_size_ms. Two distinct ways this fails — keep them separate so
    # the skip reason is actionable rather than a generic error:
    #
    #   1. Sub-token: the tokenizer is coarser than the requested chunk, so a
    #      chunk can't span even one token. Common for large-window tokenizers
    #      (e.g. a 1000ms-token model with a 200ms behavioral chunk) — behavior
    #      simply can't be decoded at this temporal resolution.
    #   2. Misaligned: chunk is large enough but not a whole multiple, so chunks
    #      would straddle token boundaries.
    if chunk_window_ms + 1e-6 < token_bin_size_ms:
        raise ValueError(
            f"behavioral decoding not applicable: chunk_window_ms "
            f"({chunk_window_ms}) is smaller than the tokenizer's token_bin_size_ms "
            f"({token_bin_size_ms}), so no chunk spans a whole token. This "
            f"tokenizer's temporal resolution is too coarse for a "
            f"{chunk_window_ms}ms behavioral window — set chunk_window_ms to a "
            f"multiple of {token_bin_size_ms}ms (coarser decoding) or use a "
            f"finer-binned tokenizer."
        )

    remainder = chunk_window_ms % token_bin_size_ms
    if remainder > 1e-6 and abs(token_bin_size_ms - remainder) > 1e-6:
        raise ValueError(
            f"behavioral decoding misaligned: chunk_window_ms ({chunk_window_ms}) "
            f"must be a multiple of the tokenizer's token_bin_size_ms "
            f"({token_bin_size_ms}) so each chunk spans whole tokens "
            f"(remainder={remainder}ms)."
        )

    tokens_per_chunk = round(chunk_window_ms / token_bin_size_ms)

    T, feature_dim = latents.shape
    if T < tokens_per_chunk:
        raise ValueError(
            f"Not enough tokens ({T}) to form even one chunk "
            f"(tokens_per_chunk={tokens_per_chunk} from "
            f"chunk_window_ms={chunk_window_ms}, "
            f"token_bin_size_ms={token_bin_size_ms})"
        )

    # Number of complete windows: largest j such that
    # start_idx(j) + tokens_per_chunk <= T
    n_windows = int((T - tokens_per_chunk) * token_bin_size_ms / chunk_window_ms) + 1

    # Vectorised token gathering
    js = np.arange(n_windows)
    start_indices = (chunk_window_ms * js / token_bin_size_ms).astype(int)

    # Safety: clamp last window (floating-point edge case)
    assert start_indices[-1] + tokens_per_chunk <= T, (
        f"Last window overflows: start={start_indices[-1]}, "
        f"tokens_per_chunk={tokens_per_chunk}, T={T}"
    )

    offsets = np.arange(tokens_per_chunk)
    # (n_windows, tokens_per_chunk)
    all_indices = start_indices[:, None] + offsets[None, :]
    # Gather and reshape
    gathered = latents[all_indices]  # (n_windows, tokens_per_chunk, feature_dim)
    chunked_latents = gathered.reshape(n_windows, tokens_per_chunk * feature_dim)
    chunk_timestamps = timestamps[start_indices]

    return chunked_latents, chunk_timestamps, js, tokens_per_chunk


class FeatureReducer:
    """Reduce feature dimensionality after temporal chunking.


    Wraps truncation, PCA, and sparse random projection (SRP) behind a
    uniform ``fit`` / ``transform`` interface.  Instantiated inside each
    CV fold so that PCA is fit only on training data.

    Parameters
    ----------
    method : str
        One of ``"none"``, ``"truncate"``, ``"pca"``, ``"srp"``.
    n_features : int
        Target number of features after reduction.
    random_state : int
        Seed for SRP (ignored by other methods).
    """

    METHODS = ("none", "truncate", "pca", "srp")

    def __init__(
        self,
        method: str = "none",
        n_features: int | None = 32,
        random_state: int = 42,
    ):
        if method not in self.METHODS:
            raise ValueError(
                f"Unknown feature_reduction method '{method}'. Choose from {self.METHODS}"
            )
        if method != "none" and n_features is None:
            raise ValueError(
                f"n_features must be set when feature_reduction='{method}'. "
                f"Add n_features to your metric config."
            )
        self.method = method
        self.n_features = n_features
        self.random_state = random_state
        self._fitted = False
        self._transformer: Any = None  # sklearn transformer for pca/srp

    def fit(self, X: np.ndarray) -> "FeatureReducer":
        """Fit the reducer on training data."""
        import warnings

        if self.method == "none" or self.method == "truncate":
            # Stateless — nothing to fit
            pass
        elif self.method == "pca":
            from sklearn.decomposition import PCA

            n_components = min(self.n_features, X.shape[1], X.shape[0])
            self._transformer = PCA(n_components=n_components)
            self._transformer.fit(X)
        elif self.method == "srp":
            from sklearn.random_projection import SparseRandomProjection

            n_components = min(self.n_features, X.shape[1])
            self._transformer = SparseRandomProjection(
                n_components=n_components,
                random_state=self.random_state,
            )
            self._transformer.fit(X)

        if self.method != "none" and X.shape[1] < self.n_features:
            warnings.warn(
                f"FeatureReducer: input has {X.shape[1]} features, less than "
                f"target n_features={self.n_features}. Output will have "
                f"{X.shape[1]} features (no padding).",
                stacklevel=2,
            )
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project X to reduced dimensionality."""
        assert self._fitted, "FeatureReducer.fit() must be called before transform()"

        if self.method == "none":
            return X
        elif self.method == "truncate":
            n = min(self.n_features, X.shape[1])
            return X[:, :n]
        elif self.method in ("pca", "srp"):
            assert self._transformer is not None
            return self._transformer.transform(X)
        else:
            raise RuntimeError(f"Unexpected method: {self.method}")

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one call."""
        return self.fit(X).transform(X)


def align_behavioral_to_chunks(
    chunk_indices: np.ndarray,
    chunk_window_ms: float,
    behavioral_data_input_res: dict[str, np.ndarray],
    sampling_rate: float,
    target_field: str,
    mode: str,
    grouping_field: str | None = "trial_id",
    classification_majority_threshold: float = 0.75,
    behavioral_bin_size_ms: float | None = None,
    excluded_label_values: list[int | float] | None = None,
    nan_drop_threshold: float = 0.25,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, dict[str, Any]]:
    """Align input-resolution behavioral data to pre-built temporal chunks.


    Uses the same index-space grid as ``chunk_by_temporal_window``:
    behavioral chunk *j* starts at index
    ``int(chunk_window_ms * j / input_bin_size_ms)`` and spans
    ``K_input_res`` input-resolution bins.  This ensures identical
    behavioral targets across tokenizers regardless of token bin size.

    No trial-boundary drop: when grouping_field changes within a chunk,
    the majority grouping value is used for CV group assignment.

    Classification mode uses NaN-aware majority vote:
    1. If >nan_drop_threshold of labels are NaN → drop chunk
    2. Majority computed over non-NaN labels only
    3. If majority label in excluded_label_values → drop chunk
    4. If majority fraction < classification_majority_threshold → drop chunk

    Args:
        chunk_indices: Original chunk index *j* for each (possibly filtered)
            chunk, shape ``(n_chunks,)``.  Produced by
            ``chunk_by_temporal_window`` and carried through
            ``preprocess_downstream_data``.
        chunk_window_ms: Temporal window per chunk in ms (same value used
            for token chunking).
        behavioral_data_input_res: All behavioral fields at input resolution.
        sampling_rate: Input sampling rate in Hz.
        target_field: Which behavioral field to use as target.
        mode: ``"regression"`` or ``"classification"``.
        grouping_field: Field for grouping (default: ``"trial_id"``).
            When the field changes within a chunk, the majority value is used.
        classification_majority_threshold: Min fraction for majority class
            (default 0.75).
        behavioral_bin_size_ms: Optional downsampling of behavioral targets.
        excluded_label_values: Classification only. Label values to exclude
            (e.g. [-1] for intertrial fill). Chunks whose majority label
            is in this list are dropped.
        nan_drop_threshold: Classification only. Drop chunk if more than
            this fraction of labels are NaN (default 0.25).

    Returns:
        ``(y, grouping_values, keep_mask, metadata)`` where:

        - y: For regression: ``(n_kept, K * target_dim)`` flattened
          behavioral targets.  For classification: ``(n_kept,)`` majority
          class labels.
        - grouping_values: Per-kept-chunk grouping field values, or ``None``
          if *grouping_field* is ``None``.
        - keep_mask: Boolean array over ALL input chunks — caller applies
          it to X.
        - metadata: dict with provenance info.
    """
    assert mode in ("regression", "classification"), f"Unknown mode: {mode}"
    assert target_field in behavioral_data_input_res, (
        f"target_field '{target_field}' not found in behavioral_data_input_res. "
        f"Available: {sorted(behavioral_data_input_res.keys())}"
    )

    n_chunks = len(chunk_indices)
    n_behavioral = len(next(iter(behavioral_data_input_res.values())))
    target_data = behavioral_data_input_res[target_field]

    # Compute behavioral bin parameters
    input_bin_size_ms = 1000.0 / sampling_rate
    if behavioral_bin_size_ms is None:
        behavioral_bin_size_ms = input_bin_size_ms
    elif behavioral_bin_size_ms < input_bin_size_ms:
        log.warning(
            "behavioral_bin_size_ms (%.1f) < input bin size (%.1fms), clamping to %.1fms",
            behavioral_bin_size_ms,
            input_bin_size_ms,
            input_bin_size_ms,
        )
        behavioral_bin_size_ms = input_bin_size_ms

    downsample_factor = int(round(behavioral_bin_size_ms / input_bin_size_ms))
    assert abs(downsample_factor * input_bin_size_ms - behavioral_bin_size_ms) < 1e-6, (
        f"behavioral_bin_size_ms ({behavioral_bin_size_ms}) must be an integer "
        f"multiple of input bin size ({input_bin_size_ms}ms)"
    )

    K_input_res = int(round(chunk_window_ms / input_bin_size_ms))
    K = -(-K_input_res // downsample_factor)  # ceil division

    assert K > 0, (
        f"K=0: chunk_window_ms={chunk_window_ms}, behavioral_bin_size_ms={behavioral_bin_size_ms}"
    )

    # Get grouping field data
    grouping_data = None
    if grouping_field is not None:
        grouping_data = behavioral_data_input_res.get(grouping_field)

    # Per-chunk alignment
    keep_mask = np.ones(n_chunks, dtype=bool)
    y_chunks: list[np.ndarray | None] = []
    grouping_chunks: list[Any | None] = []
    metadata_counts: dict[str, Any] = {
        "n_chunks_total": n_chunks,
        "n_chunks_dropped_incomplete": 0,
        "n_chunks_dropped_low_majority": 0,
        "n_chunks_dropped_nan_labels": 0,
        "n_chunks_dropped_excluded_label": 0,
        "K_per_chunk": K,
        "K_input_res_per_chunk": K_input_res,
        "behavioral_bin_size_ms": behavioral_bin_size_ms,
        "chunk_window_ms": chunk_window_ms,
        "downsample_factor": downsample_factor,
    }

    excluded_set = set(excluded_label_values) if excluded_label_values else set()

    for i in range(n_chunks):
        j = int(chunk_indices[i])
        start_idx = int(chunk_window_ms * j / input_bin_size_ms)
        end_idx = start_idx + K_input_res

        # Check: enough behavioral bins?
        if end_idx > n_behavioral:
            keep_mask[i] = False
            metadata_counts["n_chunks_dropped_incomplete"] += 1
            y_chunks.append(None)
            grouping_chunks.append(None)
            continue

        # Grouping: use majority value (no trial-boundary drop)
        chunk_group_value = None
        if grouping_data is not None:
            chunk_grouping = grouping_data[start_idx:end_idx]
            unique_groups, group_counts = np.unique(chunk_grouping, return_counts=True)
            chunk_group_value = unique_groups[group_counts.argmax()]

        # Extract behavioral bins (with optional downsampling)
        chunk_behavioral = target_data[start_idx:end_idx:downsample_factor]
        assert len(chunk_behavioral) == K, (
            f"Expected {K} bins, got {len(chunk_behavioral)} "
            f"(start={start_idx}, end={end_idx}, factor={downsample_factor})"
        )

        if mode == "regression":
            # Flatten to 1D per chunk (works for both 1D and multi-dim targets)
            y_chunks.append(chunk_behavioral.flatten())
        else:
            # Classification: NaN-aware majority vote
            if chunk_behavioral.ndim > 1:
                # Multi-dim: treat each row as a tuple
                from collections import Counter

                rows = [tuple(r) for r in chunk_behavioral]
                nan_count = sum(1 for r in rows if any(np.isnan(v) for v in r))
                nan_fraction = nan_count / len(rows)
                if nan_fraction > nan_drop_threshold:
                    keep_mask[i] = False
                    metadata_counts["n_chunks_dropped_nan_labels"] += 1
                    y_chunks.append(None)
                    grouping_chunks.append(None)
                    continue
                non_nan_rows = [r for r in rows if not any(np.isnan(v) for v in r)]
                if not non_nan_rows:
                    keep_mask[i] = False
                    metadata_counts["n_chunks_dropped_nan_labels"] += 1
                    y_chunks.append(None)
                    grouping_chunks.append(None)
                    continue
                counts = Counter(non_nan_rows)
                majority_val, majority_count = counts.most_common(1)[0]
                majority_fraction = majority_count / len(non_nan_rows)
                if excluded_set and majority_val in excluded_set:
                    keep_mask[i] = False
                    metadata_counts["n_chunks_dropped_excluded_label"] += 1
                    y_chunks.append(None)
                    grouping_chunks.append(None)
                    continue
                if majority_fraction < classification_majority_threshold:
                    keep_mask[i] = False
                    metadata_counts["n_chunks_dropped_low_majority"] += 1
                    y_chunks.append(None)
                    grouping_chunks.append(None)
                    continue
                y_chunks.append(np.array(majority_val))
            else:
                # 1-D labels: NaN-aware majority vote
                nan_mask = (
                    np.isnan(chunk_behavioral)
                    if np.issubdtype(chunk_behavioral.dtype, np.floating)
                    else np.zeros(len(chunk_behavioral), dtype=bool)
                )
                nan_fraction = nan_mask.sum() / len(chunk_behavioral)
                if nan_fraction > nan_drop_threshold:
                    keep_mask[i] = False
                    metadata_counts["n_chunks_dropped_nan_labels"] += 1
                    y_chunks.append(None)
                    grouping_chunks.append(None)
                    continue
                non_nan_vals = chunk_behavioral[~nan_mask]
                if len(non_nan_vals) == 0:
                    keep_mask[i] = False
                    metadata_counts["n_chunks_dropped_nan_labels"] += 1
                    y_chunks.append(None)
                    grouping_chunks.append(None)
                    continue
                unique_vals, counts = np.unique(non_nan_vals, return_counts=True)
                majority_idx = counts.argmax()
                majority_val = unique_vals[majority_idx]
                majority_fraction = counts[majority_idx] / len(non_nan_vals)
                if excluded_set and majority_val in excluded_set:
                    keep_mask[i] = False
                    metadata_counts["n_chunks_dropped_excluded_label"] += 1
                    y_chunks.append(None)
                    grouping_chunks.append(None)
                    continue
                if majority_fraction < classification_majority_threshold:
                    keep_mask[i] = False
                    metadata_counts["n_chunks_dropped_low_majority"] += 1
                    y_chunks.append(None)
                    grouping_chunks.append(None)
                    continue
                y_chunks.append(majority_val)

        grouping_chunks.append(chunk_group_value)

    # Build outputs from kept chunks
    valid_y = [y for y in y_chunks if y is not None]

    if len(valid_y) == 0:
        raise ValueError(f"No valid chunks remain after alignment. Dropped: {metadata_counts}")

    if mode == "regression":
        y = np.stack(valid_y, axis=0)
    else:
        y = np.array(valid_y)

    metadata_counts["n_chunks_kept"] = int(keep_mask.sum())

    # Build grouping_values for kept chunks
    grouping_values: np.ndarray | None = None
    if grouping_data is not None:
        valid_grouping = [g for g in grouping_chunks if g is not None]
        grouping_values = np.array(valid_grouping)

    return y, grouping_values, keep_mask, metadata_counts


def _compute_chunk_token_indices(
    chunk_indices: np.ndarray,
    chunk_window_ms: float,
    token_bin_size_ms: float,
    tokens_per_chunk: int,
) -> np.ndarray:
    """Compute the token indices for each chunk, shape ``(n_chunks, tokens_per_chunk)``.


    Uses the same index-space grid as ``chunk_by_temporal_window``.
    """
    start_indices = (chunk_window_ms * chunk_indices / token_bin_size_ms).astype(int)
    if tokens_per_chunk > 1:
        offsets = np.arange(tokens_per_chunk)
        return start_indices[:, None] + offsets[None, :]
    else:
        return start_indices[:, None]


def _apply_chunk_threshold_filter(
    token_mask: np.ndarray,
    all_token_indices: np.ndarray,
    chunk_drop_threshold: float,
) -> np.ndarray:
    """Drop chunks where the fraction of failing tokens exceeds the threshold.


    Args:
        token_mask: Per-token boolean mask, shape ``(T,)``. True = passes.
        all_token_indices: Token indices per chunk, shape ``(n_chunks, tokens_per_chunk)``.
        chunk_drop_threshold: Fraction of failing tokens above which chunk is dropped.
            E.g. 0.25 means drop if >25% of tokens fail.

    Returns:
        Boolean keep mask, shape ``(n_chunks,)``.
    """
    per_chunk_pass = token_mask[all_token_indices]  # (n_chunks, tokens_per_chunk)
    fail_fraction = 1.0 - per_chunk_pass.mean(axis=1)
    return fail_fraction <= chunk_drop_threshold


def preprocess_downstream_data(
    latents: Any,
    behavioral_data_token_res: dict[str, np.ndarray] | None,
    token_timestamps: np.ndarray | None,
    *,
    token_bin_size_ms: float,
    drop_invalid: bool = False,
    include_phases: list[str] | None = None,
    filter_field: str | None = None,
    filter_values: list[Any] | None = None,
    chunk_window_ms: float = 1000,
    chunk_drop_threshold: float = 0.25,
    behavioral_data_input_res: dict[str, np.ndarray] | None = None,
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray] | None,
    dict[str, np.ndarray] | None,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    """Apply the common preprocessing steps used by downstream metrics.


    Handles: flatten latents, chunk X via ``chunk_by_temporal_window``,
    then filter whole chunks using token-resolution behavioral fields
    with a configurable threshold (default 25%: a chunk is dropped if
    more than 25% of its tokens fail a filter).

    Applied to ALL chunks (both train and val folds). But regression,
    which uses separate train and val time filtering, passes empty
    filters (no filtering) to this preprocessing function.

    The order is chunk-first, then filter whole chunks:

    1. Flatten latents, validate lengths
    2. Build per-token boolean masks from filtering criteria
    3. Call ``chunk_by_temporal_window`` to build temporal chunks
    4. For each chunk, check fraction of failing tokens against threshold
    5. Apply chunk-level keep mask to X, chunk_timestamps, chunk_indices
    6. Pass through ``behavioral_data_input_res`` unchanged
       (Note on 6.: preprocess_downstream_data returns (X, ..., chunk_indices, ...).
        X is chunked + chunk-filtered. chunk_indices[i] = the original chunk index j
        in the pre-filter grid of the kept X[i].
        - The caller then calls
           align_behavioral_to_chunks(chunk_indices=...,
                                      behavioral_data_input_res=...)
           (downstream_prediction.py:392).
           That function uses j together with chunk_window_ms / input_bin_size_ms to
           compute, for each kept chunk, exactly which slice of the original
           input-res array to pull its target and grouping values from.
        )

    Args:
        chunk_drop_threshold: Fraction of failing tokens above which a chunk
            is dropped. Default 0.25 (drop if >25% fail). Applies to phase,
            validity, and custom field filters independently.

    Returns:
        ``(X, behavioral_data_token_res, behavioral_data_input_res,
        chunk_timestamps, chunk_indices, metadata)``

        - ``chunk_indices``: Original chunk index *j* for each kept chunk,
          needed by ``align_behavioral_to_chunks``.
    """
    if behavioral_data_token_res is None:
        raise ValueError("behavioral_data_token_res is required for downstream metrics")
    if token_timestamps is None:
        raise ValueError("token_timestamps are required for downstream metrics")

    timestamps = np.asarray(token_timestamps)
    try:
        X, metadata = _canonicalize_latents_with_timestamps(latents, timestamps)
    except ValueError as e:
        raise ValueError(f"Temporal misalignment before preprocessing: {e}") from e

    if len(X) == 0:
        raise ValueError("No latent samples available for downstream metrics")
    if len(timestamps) != len(X):
        raise ValueError(
            f"Temporal misalignment before preprocessing: {len(X)} latents vs "
            f"{len(timestamps)} timestamps"
        )
    for key, values in behavioral_data_token_res.items():
        if len(values) != len(X):
            raise ValueError(
                f"Behavioral field '{key}' length {len(values)} != latent length {len(X)}"
            )

    behavior = {k: np.asarray(v) for k, v in behavioral_data_token_res.items()}
    # metadata from _canonicalize_latents_with_timestamps contains latent_layout etc.
    T = len(X)

    # ── Build per-token boolean masks for each filter ──────────────
    # Each filter gets its own mask. At chunk level, each is checked
    # independently against chunk_drop_threshold.
    filter_masks: dict[str, np.ndarray] = {}

    if drop_invalid:
        if "trial_is_valid" not in behavior:
            raise ValueError("drop_invalid=True requires behavioral field 'trial_is_valid'")
        filter_masks["invalid"] = behavior["trial_is_valid"] != 0
        metadata["n_tokens_invalid"] = int(T - np.count_nonzero(filter_masks["invalid"]))
    else:
        metadata["n_tokens_invalid"] = 0

    if include_phases is not None:
        filter_masks["phase"] = _behavioral_filter_mask(behavior, "movement_phase", include_phases)
        metadata["include_phases"] = list(include_phases)

    if filter_field is not None:
        if filter_values is None:
            raise ValueError("filter_field requires filter_values")
        filter_masks["field"] = _behavioral_filter_mask(behavior, filter_field, filter_values)
        metadata["filter_field"] = filter_field
        metadata["filter_values"] = list(filter_values)

    # ── Chunk X via temporal windowing ───────────────────────────
    X, chunk_timestamps, chunk_indices, tokens_per_chunk = chunk_by_temporal_window(
        X,
        timestamps,
        token_bin_size_ms=token_bin_size_ms,
        chunk_window_ms=chunk_window_ms,
    )
    metadata["chunk_window_ms"] = chunk_window_ms
    metadata["token_bin_size_ms"] = token_bin_size_ms
    metadata["tokens_per_chunk"] = tokens_per_chunk
    metadata["chunk_drop_threshold"] = chunk_drop_threshold

    # ── Apply chunk-level threshold filtering ──────────────────────
    # Each filter is applied independently: a chunk is dropped if the
    # fraction of failing tokens exceeds chunk_drop_threshold.
    n_chunks = len(X)
    all_token_indices = _compute_chunk_token_indices(
        chunk_indices,
        chunk_window_ms,
        token_bin_size_ms,
        tokens_per_chunk,
    )

    chunk_keep = np.ones(n_chunks, dtype=bool)
    n_dropped_by: dict[str, int] = {}

    for filter_name, token_mask in filter_masks.items():
        filter_keep = _apply_chunk_threshold_filter(
            token_mask,
            all_token_indices,
            chunk_drop_threshold,
        )
        n_dropped = int(np.count_nonzero(chunk_keep & ~filter_keep))
        n_dropped_by[filter_name] = n_dropped
        chunk_keep &= filter_keep

    total_dropped = int(n_chunks - np.count_nonzero(chunk_keep))
    metadata["n_chunks_dropped_by_filter"] = total_dropped
    metadata["n_chunks_dropped_by_filter_detail"] = n_dropped_by

    X = X[chunk_keep]
    chunk_timestamps = chunk_timestamps[chunk_keep]
    chunk_indices = chunk_indices[chunk_keep]

    if len(X) == 0:
        raise ValueError("No samples remain after downstream preprocessing")

    # ── Build filtered token-res behavioral data for fallback path ────
    # When behavioral_data_input_res is None, the metric needs token-res
    # behavioral data that matches X (filtered + chunked). Take the value
    # at the first token in each kept chunk (nearest-neighbor).
    filtered_behavior_token_res: dict[str, np.ndarray] | None = None
    if behavioral_data_input_res is None:
        first_token_indices = (chunk_window_ms * chunk_indices / token_bin_size_ms).astype(int)
        filtered_behavior_token_res = {k: v[first_token_indices] for k, v in behavior.items()}
    else:
        # When input-res data is available, return None for token-res
        # (metrics use align_behavioral_to_chunks instead)
        filtered_behavior_token_res = None

    return (
        X,
        filtered_behavior_token_res,
        behavioral_data_input_res,
        chunk_timestamps,
        chunk_indices,
        metadata,
    )


def build_eval_bin_mask(
    chunk_indices: np.ndarray,
    chunk_window_ms: float,
    K_per_chunk: int,
    behavioral_bin_size_ms: float,
    behavioral_data_input_res: dict[str, np.ndarray],
    sampling_rate: float,
    *,
    eval_include_phases: list[str] | None = None,
    eval_drop_invalid: bool = True,
    eval_filter_field: str | None = None,
    eval_filter_values: list[Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a per-bin boolean mask for eval-time filtering at behavioral resolution.


    For each chunk, computes which of its K behavioral-resolution bins pass
    the eval filters (phase, validity, custom field). Returns a flat mask
    over all bins (n_chunks * K_per_chunk).

    Args:
        chunk_indices: Original chunk index *j* for each chunk.
        chunk_window_ms: Temporal window per chunk in ms.
        K_per_chunk: Number of behavioral-resolution bins per chunk.
        behavioral_bin_size_ms: Resolution of predictions in ms.
        behavioral_data_input_res: All behavioral fields at input resolution.
        sampling_rate: Input sampling rate in Hz.
        eval_include_phases: Phases to keep. None = no phase filter.
        eval_drop_invalid: Whether to exclude bins with trial_is_valid == 0.
        eval_filter_field: Optional custom field filter.
        eval_filter_values: Values to keep for eval_filter_field.

    Returns:
        ``(bin_mask, stats)`` where bin_mask is shape ``(n_chunks * K_per_chunk,)``
        and stats is a dict with filtering statistics.
    """
    input_bin_size_ms = 1000.0 / sampling_rate
    downsample_factor = int(round(behavioral_bin_size_ms / input_bin_size_ms))
    n_chunks = len(chunk_indices)
    n_behavioral = len(next(iter(behavioral_data_input_res.values())))
    total_bins = n_chunks * K_per_chunk

    bin_mask = np.ones(total_bins, dtype=bool)
    stats: dict[str, Any] = {
        "n_eval_bins_total": total_bins,
        "eval_prediction_resolution_ms": behavioral_bin_size_ms,
    }

    # Precompute the input-res index for each behavioral bin across all chunks.
    # For chunk j, bin k: input_res_index = chunk_start + k * downsample_factor
    # where chunk_start = int(chunk_window_ms * j / input_bin_size_ms)
    bin_input_indices = np.empty(total_bins, dtype=int)
    for ci, j in enumerate(chunk_indices):
        chunk_start = int(chunk_window_ms * int(j) / input_bin_size_ms)
        for k in range(K_per_chunk):
            idx = chunk_start + k * downsample_factor
            bin_input_indices[ci * K_per_chunk + k] = min(idx, n_behavioral - 1)

    n_excluded_by: dict[str, int] = {}

    if eval_include_phases is not None:
        phase_data = behavioral_data_input_res.get("movement_phase")
        assert phase_data is not None, (
            "eval_include_phases requires 'movement_phase' in behavioral_data_input_res"
        )
        phase_vals = _normalize_include_values("movement_phase", eval_include_phases)
        phase_at_bins = phase_data[bin_input_indices]
        phase_keep = np.isin(phase_at_bins, phase_vals)
        n_excluded_by["phase"] = int(np.count_nonzero(bin_mask & ~phase_keep))
        bin_mask &= phase_keep

    if eval_drop_invalid:
        valid_data = behavioral_data_input_res.get("trial_is_valid")
        assert valid_data is not None, (
            "eval_drop_invalid=True requires 'trial_is_valid' in behavioral_data_input_res"
        )
        valid_at_bins = valid_data[bin_input_indices]
        valid_keep = valid_at_bins != 0
        n_excluded_by["invalid"] = int(np.count_nonzero(bin_mask & ~valid_keep))
        bin_mask &= valid_keep

    if eval_filter_field is not None:
        assert eval_filter_values is not None, "eval_filter_field requires eval_filter_values"
        field_data = behavioral_data_input_res.get(eval_filter_field)
        assert field_data is not None, (
            f"eval_filter_field '{eval_filter_field}' not found in behavioral_data_input_res"
        )
        field_vals = _normalize_include_values(eval_filter_field, eval_filter_values)
        field_at_bins = field_data[bin_input_indices]
        field_keep = np.isin(field_at_bins, field_vals)
        n_excluded_by["field"] = int(np.count_nonzero(bin_mask & ~field_keep))
        bin_mask &= field_keep

    stats["n_eval_bins_retained"] = int(bin_mask.sum())
    stats["pct_eval_bins_retained"] = float(bin_mask.sum() / total_bins) if total_bins > 0 else 0.0
    stats["n_eval_bins_excluded_by"] = n_excluded_by

    return bin_mask, stats
