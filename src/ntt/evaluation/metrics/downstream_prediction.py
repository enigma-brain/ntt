"""Downstream Prediction metrics for behavioral labels using neural latents."""

import logging
import warnings
from typing import Any, ClassVar, Literal, Protocol, cast

import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import LinAlgWarning
from sklearn.linear_model import RidgeClassifierCV, RidgeCV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    StratifiedGroupKFold,
)
from sklearn.preprocessing import StandardScaler

_linalg_warned = False


def _emit_linalg_warning_once(caught: list) -> None:
    """Re-emit the first LinAlgWarning exactly once across the entire process."""
    global _linalg_warned  # noqa: PLW0603
    if _linalg_warned or not caught:
        return
    _linalg_warned = True
    w = caught[0]
    warnings.warn_explicit(
        "Ill-conditioned matrix in RidgeCV fit.",
        category=LinAlgWarning,
        filename=w.filename,
        lineno=w.lineno,
    )


from ..core.types import (
    DataType,
    MetricCategory,
    MetricDirection,
    MetricLevel,
    MetricResult,
)
from .base import BaseMetric, register_metric
from .downstream_prediction_utils import (
    FeatureReducer,
    align_behavioral_to_chunks,
    build_eval_bin_mask,
    preprocess_downstream_data,
)


def required_behavioral_fields(metric: Any) -> frozenset[str]:
    """Behavioral fields a downstream decoding metric cannot compute without.

    Mirrors the metrics' own hard checks: the target field, ``movement_phase``
    when any phase filter is set, ``trial_is_valid`` when invalid trials are
    dropped, and any filter field. Used to decide whether a session's
    behavior can actually be consumed (and so whether ``max_eval_seconds``
    may be bypassed for it).
    """
    fields = {metric.target_field}
    phase_filters = ("include_phases", "train_include_phases", "eval_include_phases")
    if any(getattr(metric, name, None) is not None for name in phase_filters):
        fields.add("movement_phase")
    drop_flags = ("drop_invalid", "train_drop_invalid", "eval_drop_invalid")
    if any(getattr(metric, name, False) for name in drop_flags):
        fields.add("trial_is_valid")
    for name in ("filter_field", "train_filter_field", "eval_filter_field"):
        if getattr(metric, name, None) is not None:
            fields.add(getattr(metric, name))
    return frozenset(fields)


def _normalize_chunk_window_ms(value: "float | list[float]") -> list[float]:
    """Normalize chunk_window_ms to a list of floats.


    Accepts a scalar, list, tuple, or OmegaConf ListConfig. Returns a list of floats.
    """
    # Handle any iterable-like (list, tuple, OmegaConf ListConfig) but not str
    if not isinstance(value, (int, float)):
        values = list(value)
        assert len(values) > 0, "chunk_window_ms list must not be empty"
        return [float(v) for v in values]
    return [float(value)]


class _ChunkWindowMetric(Protocol):
    """A downstream metric whose ``chunk_window_ms`` can be swept."""

    chunk_window_ms: float


def _run_chunk_window_sweep(
    metric: _ChunkWindowMetric,
    chunk_values: list[float],
    intermediates: dict[str, Any],
    compute_fn,
) -> MetricResult:
    """Run a downstream metric for multiple chunk_window_ms values.


    For each chunk size, the full computation (preprocessing, CV, etc.) is
    repeated. Regression and classification report the smallest successful
    chunk size >= 200 ms. Metadata retains each chunk size's result or error.

    Args:
        metric: The metric instance (used to set ``chunk_window_ms``).
        chunk_values: List of chunk_window_ms values to try.
        intermediates: The intermediates dict from the eval pipeline.
        compute_fn: Callable that performs the single-chunk-size computation.
            Signature: ``(intermediates) -> MetricResult``.

    Returns:
        MetricResult for the selected chunk size, with sweep info in metadata.
    """
    sweep_results: dict[float, dict[str, Any]] = {}
    best_result: MetricResult | None = None
    primary_chunk_ms = float("inf")

    for cw_ms in chunk_values:
        metric.chunk_window_ms = cw_ms
        try:
            result = compute_fn(intermediates)
            if not np.isfinite(result.raw_value):
                raise ValueError(f"Non-finite score for chunk_window_ms={cw_ms}")
            # IMPORTANT: store a *non-cyclic* snapshot of per-chunk metadata.
            # `result.metadata` will later receive `chunk_window_sweep` itself,
            # so saving it by reference here creates a self-referential structure
            # that breaks JSON serialization and/or drops fields.
            per_chunk_meta = dict(result.metadata) if isinstance(result.metadata, dict) else {}
            per_chunk_meta.pop("chunk_window_sweep", None)
            sweep_results[cw_ms] = {
                "raw_value": result.raw_value,
                # Include per-chunk-size metadata so plotting can show correct
                # sample/chunk counts for each chunk window.
                "metadata": per_chunk_meta,
                "status": "ok",
            }
            if 200 <= cw_ms < primary_chunk_ms:
                primary_chunk_ms = cw_ms
                best_result = result
        except (ValueError, AssertionError) as e:
            sweep_results[cw_ms] = {
                "raw_value": None,
                "status": "error",
                "error": str(e),
            }

    if best_result is None:
        errors = {
            k: v.get("error", "unknown")
            for k, v in sweep_results.items()
            if v["status"] == "error"
        }
        raise ValueError(f"No valid eligible chunk_window_ms values in sweep: {errors}")

    best_result.metadata["chunk_window_sweep"] = sweep_results
    best_result.metadata["chunk_window_sweep_values"] = chunk_values
    best_chunk_ms = max(
        (cw for cw, result in sweep_results.items() if result["status"] == "ok"),
        key=lambda cw: sweep_results[cw]["raw_value"],
    )
    best_result.metadata["best_chunk_window_ms"] = best_chunk_ms
    best_result.metadata["best_raw_value"] = sweep_results[best_chunk_ms]["raw_value"]
    best_result.metadata["primary_chunk_window_ms"] = primary_chunk_ms
    metric.chunk_window_ms = primary_chunk_ms
    return best_result


logger = logging.getLogger(__name__)


def compute_groups(
    values: np.ndarray,
    transition: Literal["change", "decrease", "increase", "from_zero"] = "change",
) -> np.ndarray:
    """Compute group assignments based on transition type.

    Shared by both regression and classification downstream metrics.

    Args:
        values: Array of values to use for detecting group boundaries
        transition: How to detect boundaries:
            - 'change': New group when value changes
            - 'decrease': New group when value decreases
            - 'increase': New group when value increases
            - 'from_zero': New group when transitioning from <=0 to >0

    Returns:
        Array of integer group assignments (same length as values)
    """
    if len(values) == 0:
        return np.array([], dtype=int)

    values = np.asarray(values)
    prev = values[:-1]
    curr = values[1:]

    if transition == "change":
        boundaries = curr != prev
    elif transition == "decrease":
        boundaries = curr < prev
    elif transition == "increase":
        boundaries = curr > prev
    elif transition == "from_zero":
        boundaries = (prev <= 0) & (curr > 0)
    else:
        raise ValueError(f"Unknown transition type: {transition!r}")

    return np.concatenate([[0], np.cumsum(boundaries)]).astype(int)


def _ensure_2d_targets(targets: np.ndarray) -> np.ndarray:
    """Return targets as shape ``[n_samples, n_outputs]``."""
    arr = np.asarray(targets, dtype=np.float32)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim == 2:
        return arr
    raise ValueError(f"Expected 1D or 2D targets, got shape {arr.shape}")


def _channel_names_for_target(target_field: str, n_outputs: int) -> list[str]:
    if n_outputs == 1:
        return [target_field]
    return [f"{target_field}_{idx}" for idx in range(n_outputs)]


def _compute_r2_by_channel(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_field: str,
) -> dict[str, float]:
    y_true_2d = _ensure_2d_targets(y_true)
    y_pred_2d = _ensure_2d_targets(y_pred)
    if y_true_2d.shape != y_pred_2d.shape:
        raise ValueError(
            f"Expected matching target/prediction shapes, got "
            f"{y_true_2d.shape} and {y_pred_2d.shape}"
        )
    return {
        channel_name: float(r2_score(y_true_2d[:, idx], y_pred_2d[:, idx]))
        for idx, channel_name in enumerate(
            _channel_names_for_target(target_field, y_true_2d.shape[1])
        )
    }


def _build_sample_trials(
    *,
    y_val: np.ndarray,
    y_pred: np.ndarray,
    timestamps: np.ndarray,
    trial_ids: np.ndarray,
    sample_trial_count: int,
) -> list[dict[str, Any]]:
    """Return a small set of validation/test trial trajectories for inspection."""
    if sample_trial_count <= 0 or len(y_val) == 0:
        return []

    sample_trials: list[dict[str, Any]] = []
    unique_trial_ids = np.unique(trial_ids)
    rng = np.random.default_rng(42)
    n_to_sample = min(sample_trial_count, len(unique_trial_ids))
    sampled_trial_ids = rng.choice(unique_trial_ids, size=n_to_sample, replace=False)
    for trial_id in sampled_trial_ids:
        mask = trial_ids == trial_id
        sample_trials.append(
            {
                "trial_id": trial_id.item() if hasattr(trial_id, "item") else trial_id,
                "timestamps": np.asarray(timestamps[mask]).tolist(),
                "y_true": np.asarray(y_val[mask]).tolist(),
                "y_pred": np.asarray(y_pred[mask]).tolist(),
            }
        )
    return sample_trials


def _build_concatenated_trials(
    *,
    y_val: np.ndarray,
    y_pred: np.ndarray,
    timestamps: np.ndarray,
    trial_ids: np.ndarray,
    n_trials: int = 5,
) -> dict[str, Any] | None:
    """Return ~n_trials consecutive trials concatenated for trajectory plotting."""
    unique_trial_ids = np.unique(trial_ids)
    if len(unique_trial_ids) < 2:
        return None

    n_to_use = min(n_trials, len(unique_trial_ids))

    trial_first_ts = []
    for tid in unique_trial_ids:
        mask = trial_ids == tid
        trial_first_ts.append(timestamps[mask].min())
    sorted_order = np.argsort(trial_first_ts)
    sorted_trial_ids = unique_trial_ids[sorted_order]

    start_idx = max(0, len(sorted_trial_ids) // 2 - n_to_use // 2)
    selected_ids = sorted_trial_ids[start_idx : start_idx + n_to_use]

    concat_y_true = []
    concat_y_pred = []
    concat_ts = []
    boundaries = [0]
    selected_trial_id_list = []

    for tid in selected_ids:
        mask = trial_ids == tid
        trial_ts = timestamps[mask]
        sort_idx = np.argsort(trial_ts)
        concat_y_true.append(y_val[mask][sort_idx])
        concat_y_pred.append(y_pred[mask][sort_idx])
        concat_ts.append(trial_ts[sort_idx])
        boundaries.append(boundaries[-1] + int(mask.sum()))
        selected_trial_id_list.append(tid.item() if hasattr(tid, "item") else tid)

    return {
        "y_true": np.concatenate(concat_y_true, axis=0).tolist(),
        "y_pred": np.concatenate(concat_y_pred, axis=0).tolist(),
        "timestamps": np.concatenate(concat_ts, axis=0).tolist(),
        "trial_boundaries": boundaries,
        "trial_ids": selected_trial_id_list,
    }


class TemporalBehaviorDecoder(nn.Module):
    """Lightweight temporal decoder from latent trajectories to behavior."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        kernel_size: int = 9,
        dropout: float = 0.1,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve sequence length")

        padding = kernel_size // 2
        self.input_norm = nn.GroupNorm(1, in_channels)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                hidden_channels,
                hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=hidden_channels,
            ),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.net(self.input_norm(latents))


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        warnings.warn(
            "CUDA requested for behavioral_temporal_decoder, falling back to CPU.", stacklevel=2
        )
        return torch.device("cpu")
    return torch.device(device_name)


def _split_grouped_train_val(
    groups: np.ndarray,
    *,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic grouped train/val split."""
    n_groups = int(np.max(groups)) + 1
    if n_groups < 2:
        raise ValueError(
            f"Need at least 2 contiguous groups for grouped train/val splitting, got {n_groups}"
        )

    dummy = np.zeros(len(groups), dtype=np.float32)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_seed)
    try:
        train_idx, val_idx = next(gss.split(dummy, groups=groups))
    except StopIteration as e:
        raise ValueError(f"Could not create grouped train/val split (n_groups={n_groups})") from e

    if min(len(train_idx), len(val_idx)) == 0:
        raise ValueError("Grouped train/val split produced an empty subset")

    return (
        np.sort(train_idx.astype(int)),
        np.sort(val_idx.astype(int)),
    )


def _build_temporal_sequence_dataset(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    split_idx: np.ndarray,
    *,
    token_timestamps: np.ndarray,
) -> dict[str, Any]:
    """Build full grouped sequences for one split without crossing boundaries."""
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    position_sequences: list[np.ndarray] = []
    selected_positions = np.sort(np.asarray(split_idx, dtype=int))
    selected_group_ids = np.unique(groups[selected_positions])
    n_groups_split_by_time_gap = 0

    for group_id in selected_group_ids:
        group_positions = selected_positions[groups[selected_positions] == group_id]
        contiguous_sequences = _split_positions_by_time_contiguity(
            group_positions,
            token_timestamps,
        )
        if len(contiguous_sequences) > 1:
            n_groups_split_by_time_gap += 1
        for sequence_positions in contiguous_sequences:
            group_features = X[sequence_positions]
            group_targets = y[sequence_positions]
            inputs.append(group_features.T.astype(np.float32))
            targets.append(group_targets.T.astype(np.float32))
            position_sequences.append(sequence_positions)

    if not inputs:
        raise ValueError("No temporal sequences remain after grouped splitting.")

    return {
        "inputs": inputs,
        "targets": targets,
        "position_sequences": position_sequences,
        "selected_positions": selected_positions,
        "n_groups_total": int(len(selected_group_ids)),
        "n_groups_split_by_time_gap": int(n_groups_split_by_time_gap),
        "n_sequences": int(len(inputs)),
    }


def _split_positions_by_time_contiguity(
    token_positions: np.ndarray,
    token_timestamps: np.ndarray,
    *,
    gap_factor: float = 1.5,
) -> list[np.ndarray]:
    """Split grouped token positions wherever timestamps reveal a temporal gap."""
    token_positions = np.asarray(token_positions, dtype=int)
    if len(token_positions) <= 1:
        return [token_positions]

    timestamps = np.asarray(token_timestamps)[token_positions]
    diffs = np.diff(timestamps)
    if np.any(diffs <= 0):
        raise ValueError(
            "Temporal decoder requires strictly increasing token_timestamps within each group"
        )

    reference_delta = float(np.median(diffs))
    if not np.isfinite(reference_delta) or reference_delta <= 0:
        return [token_positions]

    gap_threshold = max(reference_delta * gap_factor, reference_delta + 1e-9)
    split_points = np.flatnonzero(diffs > gap_threshold) + 1
    return [segment for segment in np.split(token_positions, split_points) if len(segment) > 0]


def _sequence_arrays_to_tensors(
    sequences: list[np.ndarray],
    *,
    device: torch.device,
) -> list[torch.Tensor]:
    return [
        torch.from_numpy(sequence[None, ...]).to(device=device, dtype=torch.float32)
        for sequence in sequences
    ]


def _pad_sequences_to_batch(
    sequences: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length [1, C, T] tensors into a single [B, C, T_max] batch.

    Returns:
        batch: Tensor of shape [B, C, T_max], zero-padded.
        mask: Bool tensor of shape [B, T_max], True for valid positions.
    """
    lengths = [seq.shape[-1] for seq in sequences]
    max_len = max(lengths)
    C = sequences[0].shape[1]
    device = sequences[0].device

    batch = torch.zeros(len(sequences), C, max_len, device=device, dtype=torch.float32)
    mask = torch.zeros(len(sequences), max_len, device=device, dtype=torch.bool)
    for i, seq in enumerate(sequences):
        t = seq.shape[-1]
        batch[i, :, :t] = seq[0]
        mask[i, :t] = True
    return batch, mask


def _masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute MSE loss only over valid (unpadded) positions.

    Args:
        pred: [B, C, T]
        target: [B, C, T]
        mask: [B, T] — True for valid positions
    """
    # Expand mask to [B, C, T]
    mask_expanded = mask.unsqueeze(1).expand_as(pred)
    diff = (pred - target) ** 2
    return diff[mask_expanded].mean()


def _predict_temporal_sequences(
    model: TemporalBehaviorDecoder,
    inputs: list[torch.Tensor],
    *,
    target_mean: np.ndarray,
    target_std: np.ndarray,
) -> list[np.ndarray]:
    """Run the temporal decoder on grouped sequences and de-standardize predictions."""
    if not inputs:
        return []

    device = inputs[0].device
    mean = torch.as_tensor(target_mean, device=device, dtype=torch.float32).view(1, -1, 1)
    std = torch.as_tensor(target_std, device=device, dtype=torch.float32).view(1, -1, 1)

    lengths = [inp.shape[-1] for inp in inputs]
    batch, _ = _pad_sequences_to_batch(inputs)

    model.eval()
    with torch.no_grad():
        pred_batch = model(batch)
        pred_batch = pred_batch * std + mean

    # Unpad each sequence back to its original length
    return [
        pred_batch[i, :, :length].cpu().numpy().astype(np.float32)
        for i, length in enumerate(lengths)
    ]


def _flatten_temporal_sequence_predictions(
    predictions: list[np.ndarray],
    position_sequences: list[np.ndarray],
    *,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map grouped-sequence predictions back onto flat token positions."""
    if not predictions:
        raise ValueError("No temporal sequence predictions available")

    n_outputs = predictions[0].shape[0]
    flattened = np.zeros((sequence_length, n_outputs), dtype=np.float32)
    covered = np.zeros(sequence_length, dtype=bool)

    for pred_sequence, token_positions in zip(predictions, position_sequences, strict=True):
        token_pred = pred_sequence.T
        if token_pred.shape[0] != len(token_positions):
            raise ValueError(
                "Temporal decoder produced a sequence length that does not match "
                "the grouped token positions"
            )
        if np.any(covered[token_positions]):
            raise ValueError("Temporal decoder predicted overlapping grouped token positions")
        flattened[token_positions] = token_pred
        covered[token_positions] = True

    return flattened, covered


@register_metric
class BehavioralRegressionMetric(BaseMetric):
    """Predict continuous behavioral variables from neural latents using Ridge regression.

    Uses RidgeCV with internal cross-validation to select the best alpha.
    Returns validation R² score from cross-validation.

    Args:
        target_field: Name of behavioral field to predict (e.g., 'cursor_pos', 'cursor_vel')
        chunk_window_ms: Chunk tokens into chunks of this size. One chunk is the predictor for
                         one sample.Must be a multiple of token bin size.
                        **Allows to pass a list of values to sweep over.**
        behavioral_bin_size_ms: On what resolution to predict the target field.
        grouping_field: Field to use for trial/group boundaries in cross-validation splits.
            Common choices: 'trial_id', 'movement_phase'. Cannot be None (regression
            requires explicit grouping to avoid data leakage).
        grouping_transition: How to detect group boundaries from grouping_field values:
            - 'change': New group when value changes (default, use with trial_id)
            - 'decrease': New group when value decreases (use with movement_phase)
            - 'increase': New group when value increases
            - 'from_zero': New group when transitioning from <=0 to >0
        alphas: Regularization strengths to test (default: 0.1 to 1e12)
        n_splits: Number of outer crossvalidation splits. Default: None, which means 3 splits.
        cv: Cross-validation strategy for alpha selection inside RidgeCV.
            None (default) = efficient leave-one-out via GCV (fast, closed-form).
            int = k-fold CV (slower, but useful for very large datasets).
        standardize: If True, standardize the latents to have mean 0 and variance 1.
                 (Standardizes the validation split by training set statistics, I believe.)

        Filtering parameters: See `docs/eval_metrics.md` . For regression, we filter twice,
        at token-chunk level and at behavioral bin level.

        Chunk size sweep - how it works:
        - Scalar: single run (incl. inner (cv) and outer crossvalidation)
        - List with 1 element: single run
        - List with 2+ elements: runs the full computation (preprocessing, CV, etc.) for each
                chunk size independently. Reports the smallest successful chunk size
                >= 200 ms.
            Metadata includes:
            - chunk_window_sweep: dict mapping each tried chunk_ms → {raw_value, status}
              (or {error, status} for failures)
            - chunk_window_sweep_values: the original list
            - chunk_window_ms: reflects the selected chunk size
    """

    name = "behavioral_regression"
    category = MetricCategory.DOWNSTREAM_DECODING
    direction = MetricDirection.HIGHER_IS_BETTER
    level = MetricLevel.POPULATION

    # This metric needs latent representations, not raw spikes
    supported_data_types: set[DataType] = {DataType.SPIKES, DataType.SPIKE_TIMES, DataType.RATES}

    grouping_transition: Literal["change", "decrease", "increase", "from_zero"]
    _required_intermediates = [
        "latents",
        "behavioral_data_token_res",
        "token_timestamps",
        "behavioral_data_input_res",
        "sampling_rate",
        "token_bin_size_ms",
    ]

    def __init__(
        self,
        target_field: str = "cursor_pos",
        grouping_field: str | None = "trial_id",
        grouping_transition: Literal["change", "decrease", "increase", "from_zero"] = "change",
        alphas: tuple[float, ...] = (
            0.1,
            1.0,
            10.0,
            100.0,
            1e3,
            1e4,
            1e5,
            1e6,
            1e7,
            1e8,
            1e9,
            1e10,
            1e11,
            1e12,
        ),
        cv: int | None = None,
        n_splits: int | None = None,
        standardize: bool = False,
        # Train-time chunk-level filtering (applied to all chunks via preprocess)
        train_drop_invalid: bool = True,
        train_include_phases: list[str] | None = None,
        train_filter_field: str | None = None,
        train_filter_values: list[Any] | None = None,
        train_chunk_drop_threshold: float = 0.25,
        # Eval-time bin-level filtering (applied post-prediction)
        eval_drop_invalid: bool = True,
        eval_include_phases: list[str] | None = None,
        eval_filter_field: str | None = None,
        eval_filter_values: list[Any] | None = None,
        # Shared
        chunk_window_ms: float | list[float] = 1000,
        n_features: int | None = None,
        feature_reduction: str = "none",
        sample_trial_count: int = 4,
        behavioral_bin_size_ms: float | None = None,
    ):
        if cv is not None and cv < 2:
            raise ValueError(f"cv must be None (for efficient LOO/GCV) or >= 2, got cv={cv}")
        self.target_field = target_field
        self.grouping_field = grouping_field
        self.grouping_transition = grouping_transition
        self.alphas = alphas
        self.cv = cv
        self.n_splits = n_splits
        self.standardize = standardize
        # Train-time params
        self.train_drop_invalid = train_drop_invalid
        self.train_include_phases = train_include_phases
        self.train_filter_field = train_filter_field
        self.train_filter_values = train_filter_values
        self.train_chunk_drop_threshold = train_chunk_drop_threshold
        # Eval-time params
        self.eval_drop_invalid = eval_drop_invalid
        self.eval_include_phases = eval_include_phases
        self.eval_filter_field = eval_filter_field
        self.eval_filter_values = eval_filter_values
        # Shared
        self._chunk_window_ms_values = _normalize_chunk_window_ms(chunk_window_ms)
        self.chunk_window_ms = self._chunk_window_ms_values[0]
        self.n_features = n_features
        self.feature_reduction = feature_reduction
        self.sample_trial_count = sample_trial_count
        self.behavioral_bin_size_ms = behavioral_bin_size_ms

    def compute(self, intermediates: dict[str, Any]) -> MetricResult:
        """Dispatch to single chunk size or sweep over multiple.


        When ``chunk_window_ms`` was configured as a list, runs the full
        computation for each value and returns the smallest valid chunk >= 200 ms,
        with per-chunk-size results stored in ``metadata['chunk_window_sweep']``.
        """
        if len(self._chunk_window_ms_values) > 1:
            return _run_chunk_window_sweep(
                self,
                self._chunk_window_ms_values,
                intermediates,
                self._compute_single_chunk_size,
            )
        self.chunk_window_ms = self._chunk_window_ms_values[0]
        return self._compute_single_chunk_size(intermediates)

    def _compute_single_chunk_size(self, intermediates: dict[str, Any]) -> MetricResult:
        """Compute regression R² from latents to behavioral variable.


        Two-level filtering:
        - Train: chunk-level with 25% threshold (train_* params), applied per fold
        - Eval: bin-level at behavioral resolution (eval_* params), post-prediction

        Flow:
        1. Preprocess: chunk all tokens (no filtering)
        2. Align behavioral targets to all chunks
        3. CV split on all chunks
        4. Per fold: train filter → fit → predict on all val → eval filter → R2

        Returns:
            MetricResult with validation R² as raw_value
        """
        from .downstream_prediction_utils import (
            _apply_chunk_threshold_filter,
            _behavioral_filter_mask,
            _compute_chunk_token_indices,
        )

        token_bin_size_ms = intermediates.get("token_bin_size_ms")
        if token_bin_size_ms is None:
            raise ValueError(
                "token_bin_size_ms is required for downstream metrics but was not "
                "provided in intermediates. Ensure the tokenizer reports token_bin_size_ms."
            )

        behavioral_data_token_res = intermediates["behavioral_data_token_res"]
        behavioral_data_input_res = intermediates.get("behavioral_data_input_res")

        target_available = (
            behavioral_data_token_res is not None
            and self.target_field in behavioral_data_token_res
        ) or (
            behavioral_data_input_res is not None
            and self.target_field in behavioral_data_input_res
        )
        if not target_available:
            available = set()
            if behavioral_data_token_res:
                available.update(behavioral_data_token_res.keys())
            if behavioral_data_input_res:
                available.update(behavioral_data_input_res.keys())
            raise ValueError(
                f"Behavioral field '{self.target_field}' not found in data. "
                f"Available: {sorted(available)}"
            )

        # 1. Preprocessing: chunk X (NO filtering — filtering is per-fold)
        (
            X_all,
            behavior_token_res,
            behavior_input_res,
            chunk_timestamps_all,
            chunk_indices_all,
            preprocessing_metadata,
        ) = preprocess_downstream_data(
            intermediates["latents"],
            behavioral_data_token_res,
            intermediates.get("token_timestamps"),
            token_bin_size_ms=token_bin_size_ms,
            # No filters — regression filters per-fold
            drop_invalid=False,
            include_phases=None,
            filter_field=None,
            filter_values=None,
            chunk_window_ms=self.chunk_window_ms,
            behavioral_data_input_res=behavioral_data_input_res,
        )

        # 2. Align behavioral data to ALL chunks (no filtering in align)
        if behavior_input_res is not None:
            y_all, grouping_values_all, align_keep_mask, align_metadata = (
                align_behavioral_to_chunks(
                    chunk_indices=chunk_indices_all,
                    chunk_window_ms=self.chunk_window_ms,
                    behavioral_data_input_res=behavior_input_res,
                    sampling_rate=intermediates["sampling_rate"],
                    target_field=self.target_field,
                    mode="regression",
                    grouping_field=self.grouping_field,
                    behavioral_bin_size_ms=self.behavioral_bin_size_ms,
                )
            )
            preprocessing_metadata.update(align_metadata)

            # Apply keep_mask from alignment (drops incomplete chunks)
            X_all = X_all[align_keep_mask]
            chunk_timestamps_all = chunk_timestamps_all[align_keep_mask]
            chunk_indices_all = chunk_indices_all[align_keep_mask]
        else:
            # No input-res data — extract y from token-res behavioral data.
            assert behavior_token_res is not None and self.target_field in behavior_token_res, (
                f"target_field '{self.target_field}' not in behavioral_data_token_res. "
                f"Available: {sorted((behavior_token_res or {}).keys())}"
            )
            y_all = np.asarray(behavior_token_res[self.target_field])
            grouping_values_all = None
            if self.grouping_field is not None:
                assert self.grouping_field in behavior_token_res, (
                    f"Grouping field '{self.grouping_field}' not in behavioral_data_token_res. "
                    f"Available: {sorted(behavior_token_res.keys())}"
                )
                grouping_values_all = np.asarray(behavior_token_res[self.grouping_field])

        # 3. Build train-filter token masks (for chunk-level filtering per fold)
        if behavioral_data_token_res is None:
            raise ValueError("behavioral_data_token_res is required for train-filter token masks")
        behavior_tok = {k: np.asarray(v) for k, v in behavioral_data_token_res.items()}
        len(intermediates["token_timestamps"])
        train_token_masks: dict[str, np.ndarray] = {}
        if self.train_drop_invalid:
            assert "trial_is_valid" in behavior_tok, (
                "train_drop_invalid=True requires 'trial_is_valid'"
            )
            train_token_masks["invalid"] = behavior_tok["trial_is_valid"] != 0
        if self.train_include_phases is not None:
            train_token_masks["phase"] = _behavioral_filter_mask(
                behavior_tok,
                "movement_phase",
                self.train_include_phases,
            )
        if self.train_filter_field is not None:
            assert self.train_filter_values is not None, (
                "train_filter_field requires train_filter_values"
            )
            train_token_masks["field"] = _behavioral_filter_mask(
                behavior_tok,
                self.train_filter_field,
                self.train_filter_values,
            )

        # Precompute token indices for ALL chunks (needed for train filtering)
        tokens_per_chunk = preprocessing_metadata["tokens_per_chunk"]
        all_token_indices = _compute_chunk_token_indices(
            chunk_indices_all,
            self.chunk_window_ms,
            token_bin_size_ms,
            tokens_per_chunk,
        )

        # Determine groups for CV splits (using all chunks, before any filtering)
        if grouping_values_all is not None:
            groups = compute_groups(grouping_values_all, self.grouping_transition)
        else:
            raise ValueError(
                "grouping_field is None — a grouping field (e.g. 'trial_id') is needed "
                "to group samples into groups, otherwise the validation will be too "
                "similar to train set."
            )

        # Check for sufficient unique values
        y_check = y_all.flatten() if y_all.ndim > 1 else y_all
        unique_values = np.unique(y_check[np.isfinite(y_check)])
        if len(unique_values) < 2:
            raise ValueError(
                f"Target '{self.target_field}' has < 2 unique values "
                f"(n_unique={len(unique_values)})"
            )

        n_groups = np.max(groups) + 1
        if n_groups < 2:
            raise ValueError(
                f"Target '{self.target_field}': only {n_groups} contiguous segment(s), "
                f"need >= 2 for sequence-aware split"
            )

        n_splits = self.n_splits if self.n_splits is not None else min(3, n_groups)
        use_single_split = n_splits == 1

        if not use_single_split:
            n_splits = min(n_splits, n_groups)
            # \ Can be made a lot clearer by just checking n_groups.
            if n_splits < 2:
                raise ValueError(
                    f"Target '{self.target_field}': only {n_groups} group(s) found. "
                    f"Need at least 2 for cross-validation."
                )
            splitter = GroupKFold(n_splits=n_splits)
            splits = list(splitter.split(X_all, y_all, groups=groups))
        else:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            splits = list(gss.split(X_all, y_all, groups=groups))

        # Get K_per_chunk and behavioral_bin_size_ms from align metadata
        K_per_chunk = preprocessing_metadata.get("K_per_chunk")
        behavioral_bin_size_ms = preprocessing_metadata.get("behavioral_bin_size_ms")

        fold_r2s: list[float] = []
        fold_alphas: list[float] = []
        fold_train_samples: list[int] = []
        fold_val_samples: list[int] = []
        fold_val_bins_predicted: list[int] = []
        fold_val_bins_retained: list[int] = []
        fold_groups_train: list[int] = []
        fold_groups_val: list[int] = []
        fold_train_chunks_dropped: list[int] = []
        fold_train_chunks_total: list[int] = []

        # Store last fold's val data for sample trial plots
        last_fold_val_data: dict | None = None

        for train_idx, val_idx in splits:
            # ── Train: apply chunk-level filters ──────────────────
            train_keep = np.ones(len(train_idx), dtype=bool)
            for _mask_name, token_mask in train_token_masks.items():
                filter_keep = _apply_chunk_threshold_filter(
                    token_mask,
                    all_token_indices[train_idx],
                    self.train_chunk_drop_threshold,
                )
                train_keep &= filter_keep

            # Also drop chunks with NaN/Inf targets (all-or-nothing)
            y_train_all = y_all[train_idx]
            if y_train_all.ndim > 1:
                train_finite = np.all(np.isfinite(y_train_all), axis=1)
            else:
                train_finite = np.isfinite(y_train_all)
            train_keep &= train_finite

            # Also drop chunks with non-finite X
            X_train_all = X_all[train_idx]
            train_x_finite = np.all(np.isfinite(X_train_all), axis=1)
            train_keep &= train_x_finite

            fold_train_chunks_total.append(len(train_idx))
            fold_train_chunks_dropped.append(int((~train_keep).sum()))

            X_train = X_train_all[train_keep]
            y_train = y_train_all[train_keep]

            if len(X_train) == 0:
                continue  # Skip fold if no train samples survive filtering

            # ── Val: get all val chunks (unfiltered at chunk level) ──
            X_val = X_all[val_idx]
            y_val = y_all[val_idx]
            val_chunk_indices = chunk_indices_all[val_idx]
            val_chunk_timestamps = chunk_timestamps_all[val_idx]

            # Feature reduction (fit on train fold only)
            reducer = FeatureReducer(
                method=self.feature_reduction,
                n_features=self.n_features,
            )
            X_train = reducer.fit_transform(X_train)
            X_val = reducer.transform(X_val)
            preprocessing_metadata["feature_reduction"] = self.feature_reduction
            preprocessing_metadata["n_features_after_reduction"] = X_train.shape[1]

            if self.standardize:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)

            # Fit RidgeCV on filtered training segments
            if self.cv is not None:
                train_cv = min(self.cv, len(y_train))
                if train_cv < 2:
                    continue  # Skip fold if not enough training samples for RidgeCV
                model = RidgeCV(alphas=self.alphas, cv=train_cv)
            else:
                if len(y_train) < 2:
                    continue
                model = RidgeCV(alphas=self.alphas)  # efficient LOO/GCV
            with warnings.catch_warnings():
                warnings.simplefilter("once", LinAlgWarning)
                model.fit(X_train, y_train)

            y_pred = model.predict(X_val)

            # ── Reshape to per-bin and apply eval filtering ──────
            if K_per_chunk is not None and K_per_chunk > 1:
                # Reshape chunk-level [N, K*D] → per-bin [N*K, D]
                y_val_expanded, y_pred_expanded, ts_expanded = (
                    self._expand_chunked_regression_series(
                        y_true=y_val,
                        y_pred=y_pred,
                        timestamps=val_chunk_timestamps,
                        k_per_chunk=K_per_chunk,
                        behavioral_bin_size_ms=behavioral_bin_size_ms,
                    )
                )
            else:
                y_val_expanded = y_val if y_val.ndim > 1 else y_val[:, None]
                y_pred_expanded = y_pred if y_pred.ndim > 1 else y_pred[:, None]

            # Build eval bin mask
            has_eval_filters = (
                self.eval_include_phases is not None
                or self.eval_drop_invalid
                or self.eval_filter_field is not None
            )
            if has_eval_filters and behavior_input_res is not None:
                eval_bin_mask, eval_stats = build_eval_bin_mask(
                    chunk_indices=val_chunk_indices,
                    chunk_window_ms=self.chunk_window_ms,
                    K_per_chunk=K_per_chunk or 1,
                    behavioral_bin_size_ms=behavioral_bin_size_ms or token_bin_size_ms,
                    behavioral_data_input_res=behavior_input_res,
                    sampling_rate=intermediates["sampling_rate"],
                    eval_include_phases=self.eval_include_phases,
                    eval_drop_invalid=self.eval_drop_invalid,
                    eval_filter_field=self.eval_filter_field,
                    eval_filter_values=self.eval_filter_values,
                )
                # Also exclude non-finite bins
                finite_mask = np.all(np.isfinite(y_val_expanded), axis=1) & np.all(
                    np.isfinite(y_pred_expanded), axis=1
                )
                eval_bin_mask &= finite_mask

                fold_val_bins_predicted.append(len(eval_bin_mask))
                fold_val_bins_retained.append(int(eval_bin_mask.sum()))

                y_val_filtered = y_val_expanded[eval_bin_mask]
                y_pred_filtered = y_pred_expanded[eval_bin_mask]
            else:
                # No eval filters — use all bins, just exclude non-finite
                finite_mask = np.all(np.isfinite(y_val_expanded), axis=1) & np.all(
                    np.isfinite(y_pred_expanded), axis=1
                )
                y_val_filtered = y_val_expanded[finite_mask]
                y_pred_filtered = y_pred_expanded[finite_mask]
                fold_val_bins_predicted.append(len(y_val_expanded))
                fold_val_bins_retained.append(len(y_val_filtered))

            if len(y_val_filtered) < 2:
                continue  # Skip fold if too few bins survive

            fold_r2 = r2_score(y_val_filtered, y_pred_filtered)

            fold_r2s.append(float(fold_r2))
            fold_alphas.append(float(cast(float, model.alpha_)))
            fold_train_samples.append(len(y_train))
            fold_val_samples.append(len(y_val_filtered))
            fold_groups_train.append(len(np.unique(groups[train_idx])))
            fold_groups_val.append(len(np.unique(groups[val_idx])))

            # Keep last fold's data for sample trial visualisation
            # (use expanded+filtered data for accurate plots)
            last_fold_val_data = {
                "y_val": y_val,
                "y_pred": y_pred,
                "val_idx": val_idx,
                "val_chunk_timestamps": val_chunk_timestamps,
            }

        if len(fold_r2s) == 0:
            raise ValueError(
                f"Target '{self.target_field}': no valid folds produced "
                f"(n_groups={n_groups}, n_splits={n_splits})"
            )

        # Average across folds
        mean_r2 = float(np.mean(fold_r2s))

        # Build sample trials from last fold for visualisation
        sample_trials: list[dict] = []
        sample_trials_concatenated: dict | None = None
        if last_fold_val_data is not None:
            lf = last_fold_val_data
            val_trial_ids = (
                grouping_values_all[lf["val_idx"]]
                if grouping_values_all is not None
                else groups[lf["val_idx"]]
            )
            sample_trials = self._build_sample_trials(
                y_val=lf["y_val"],
                y_pred=lf["y_pred"],
                timestamps=lf["val_chunk_timestamps"],
                trial_ids=val_trial_ids,
                k_per_chunk=K_per_chunk,
                behavioral_bin_size_ms=behavioral_bin_size_ms,
            )
            sample_trials_concatenated = self._build_concatenated_trials(
                y_val=lf["y_val"],
                y_pred=lf["y_pred"],
                timestamps=lf["val_chunk_timestamps"],
                trial_ids=val_trial_ids,
                n_trials=5,
                k_per_chunk=K_per_chunk,
                behavioral_bin_size_ms=behavioral_bin_size_ms,
            )

        n_valid_folds = len(fold_r2s)

        # Start from preprocessing_metadata (chunk_window_ms, token_bin_size_ms,
        # tokens_per_chunk, K_per_chunk, etc.) then overwrite misleading counters.
        metadata = dict(preprocessing_metadata)

        # Remove counters from preprocess/align that are misleading for regression
        # (preprocess does no filtering; align counters like n_chunks_dropped_low_majority
        # don't apply to regression)
        for stale_key in [
            "n_chunks_dropped_by_filter",  # always 0 (preprocess doesn't filter for regression)
            "n_chunks_dropped_by_filter_detail",  # always {} for regression
            "n_chunks_dropped_trial_change",  # vestigial (trial boundary drop removed)
            "n_chunks_dropped_low_majority",  # classification-only concept
            "n_chunks_dropped_nan_labels",  # classification-only concept
            "n_chunks_dropped_excluded_label",  # classification-only concept
            "chunk_drop_threshold",  # from preprocess; train threshold reported below
            "n_tokens_invalid",  # always 0 (preprocess uses drop_invalid=False)
        ]:
            metadata.pop(stale_key, None)

        total_val_bins_predicted = sum(fold_val_bins_predicted)
        total_val_bins_retained = sum(fold_val_bins_retained)

        metadata.update(
            {
                "target_field": self.target_field,
                # Cross-validation info
                "n_splits": n_valid_folds,
                "fold_r2s": [float(r) for r in fold_r2s],
                "fold_alphas": fold_alphas,
                "r2_std": float(np.std(fold_r2s)) if n_valid_folds >= 3 else None,
                "n_unique_values": len(unique_values),
                "baseline_r2": 0.0,
                "sample_trials": sample_trials,
                # Per-fold stats
                "mean_samples_train": int(np.mean(fold_train_samples))
                if fold_train_samples
                else 0,
                "mean_samples_val": int(np.mean(fold_val_samples)) if fold_val_samples else 0,
                "mean_groups_train": int(np.mean(fold_groups_train)) if fold_groups_train else 0,
                "mean_groups_val": int(np.mean(fold_groups_val)) if fold_groups_val else 0,
                # Train-time chunk-level filtering (per fold + mean)
                "train_chunks_total_per_fold": fold_train_chunks_total,
                "train_chunks_dropped_per_fold": fold_train_chunks_dropped,
                "mean_train_chunks_total": int(np.mean(fold_train_chunks_total))
                if fold_train_chunks_total
                else 0,
                "mean_train_chunks_dropped": int(np.mean(fold_train_chunks_dropped))
                if fold_train_chunks_dropped
                else 0,
                "train_chunk_drop_threshold": self.train_chunk_drop_threshold,
                "train_include_phases": self.train_include_phases,
                "train_drop_invalid": self.train_drop_invalid,
                # Eval-time bin-level filtering (per fold + mean)
                "eval_bins_predicted_per_fold": fold_val_bins_predicted,
                "eval_bins_retained_per_fold": fold_val_bins_retained,
                "mean_eval_bins_predicted": int(np.mean(fold_val_bins_predicted))
                if fold_val_bins_predicted
                else 0,
                "mean_eval_bins_retained": int(np.mean(fold_val_bins_retained))
                if fold_val_bins_retained
                else 0,
                "pct_eval_bins_retained": (
                    total_val_bins_retained / total_val_bins_predicted
                    if total_val_bins_predicted > 0
                    else 0.0
                ),
                "eval_include_phases": self.eval_include_phases,
                "eval_drop_invalid": self.eval_drop_invalid,
                # Grouping info
                "grouping_field": self.grouping_field,
                "grouping_transition": self.grouping_transition,
                "standardize": self.standardize,
            }
        )
        if sample_trials_concatenated is not None:
            metadata["sample_trials_concatenated"] = sample_trials_concatenated

        return self._make_result(
            raw_value=float(mean_r2),
            metadata=metadata,
        )

    def _build_sample_trials(
        self,
        *,
        y_val: np.ndarray,
        y_pred: np.ndarray,
        timestamps: np.ndarray,
        trial_ids: np.ndarray,
        k_per_chunk: int | None = None,
        behavioral_bin_size_ms: float | None = None,
    ) -> list[dict[str, Any]]:
        """Return a small set of validation-trial trajectories for inspection."""
        if self.sample_trial_count <= 0 or len(y_val) == 0:
            return []

        sample_trials: list[dict[str, Any]] = []
        unique_trial_ids = np.unique(trial_ids)
        rng = np.random.default_rng(42)
        n_to_sample = min(self.sample_trial_count, len(unique_trial_ids))
        sampled_trial_ids = rng.choice(unique_trial_ids, size=n_to_sample, replace=False)
        for trial_id in sampled_trial_ids:
            mask = trial_ids == trial_id
            trial_y_true, trial_y_pred, trial_ts = self._expand_chunked_regression_series(
                y_true=np.asarray(y_val[mask]),
                y_pred=np.asarray(y_pred[mask]),
                timestamps=np.asarray(timestamps[mask]),
                k_per_chunk=k_per_chunk,
                behavioral_bin_size_ms=behavioral_bin_size_ms,
            )
            sample_trials.append(
                {
                    "trial_id": trial_id.item() if hasattr(trial_id, "item") else trial_id,
                    "timestamps": trial_ts.tolist(),
                    "y_true": trial_y_true.tolist(),
                    "y_pred": trial_y_pred.tolist(),
                }
            )
        return sample_trials

    def _build_concatenated_trials(
        self,
        *,
        y_val: np.ndarray,
        y_pred: np.ndarray,
        timestamps: np.ndarray,
        trial_ids: np.ndarray,
        n_trials: int = 5,
        k_per_chunk: int | None = None,
        behavioral_bin_size_ms: float | None = None,
    ) -> dict[str, Any] | None:
        """Return ~n_trials consecutive validation trials concatenated for trajectory plotting.


        Picks a contiguous block of n_trials from the validation set (sorted by
        timestamp) so the resulting trajectory shows realistic temporal flow.

        Returns a dict with:
            - y_true: concatenated ground truth, shape (T_total, D)
            - y_pred: concatenated predictions, shape (T_total, D)
            - timestamps: concatenated timestamps, shape (T_total,)
            - trial_boundaries: list of ints, indices where each trial starts
            - trial_ids: list of trial ids corresponding to each segment

        Returns None if fewer than 2 trials are available.
        """
        unique_trial_ids = np.unique(trial_ids)
        if len(unique_trial_ids) < 2:
            return None

        n_to_use = min(n_trials, len(unique_trial_ids))

        # Pick a contiguous block: find trials ordered by their first timestamp
        trial_first_ts = []
        for tid in unique_trial_ids:
            mask = trial_ids == tid
            trial_first_ts.append(timestamps[mask].min())
        sorted_order = np.argsort(trial_first_ts)
        sorted_trial_ids = unique_trial_ids[sorted_order]

        # Pick starting position deterministically (middle of the sequence)
        start_idx = max(0, len(sorted_trial_ids) // 2 - n_to_use // 2)
        selected_ids = sorted_trial_ids[start_idx : start_idx + n_to_use]

        concat_y_true = []
        concat_y_pred = []
        concat_ts = []
        boundaries = [0]
        selected_trial_id_list = []

        for tid in selected_ids:
            mask = trial_ids == tid
            trial_ts = timestamps[mask]
            # Sort within trial by timestamp
            sort_idx = np.argsort(trial_ts)
            trial_y_true, trial_y_pred, expanded_trial_ts = self._expand_chunked_regression_series(
                y_true=np.asarray(y_val[mask][sort_idx]),
                y_pred=np.asarray(y_pred[mask][sort_idx]),
                timestamps=np.asarray(trial_ts[sort_idx]),
                k_per_chunk=k_per_chunk,
                behavioral_bin_size_ms=behavioral_bin_size_ms,
            )
            concat_y_true.append(trial_y_true)
            concat_y_pred.append(trial_y_pred)
            concat_ts.append(expanded_trial_ts)
            boundaries.append(boundaries[-1] + len(expanded_trial_ts))
            selected_trial_id_list.append(tid.item() if hasattr(tid, "item") else tid)

        return {
            "y_true": np.concatenate(concat_y_true, axis=0).tolist(),
            "y_pred": np.concatenate(concat_y_pred, axis=0).tolist(),
            "timestamps": np.concatenate(concat_ts, axis=0).tolist(),
            "trial_boundaries": boundaries,
            "trial_ids": selected_trial_id_list,
        }

    def _expand_chunked_regression_series(
        self,
        *,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        timestamps: np.ndarray,
        k_per_chunk: int | None,
        behavioral_bin_size_ms: float | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reshape chunk-level regression outputs back to per-bin trajectories.

        Reshape: [N_chunks, k_per_chunk * D]
                 → [N_chunks, k_per_chunk, D]
                 → [N_chunks * k_per_chunk, D]

        Downstream regression flattens each chunk's behavioral targets into a
        single vector of shape ``(k_per_chunk * n_coords,)``.
        This  is the reverse op, plus flattening along time, recovering the original
        dimensionality of the data in the last dim, e.g. separate x, y columns.
        Timestamps are expanded from one per chunk to one per bin,
        spaced by ``behavioral_bin_size_ms`` from each chunk's start time.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        timestamps = np.asarray(timestamps)

        if y_true.ndim == 1:
            y_true = y_true[:, None]
        if y_pred.ndim == 1:
            y_pred = y_pred[:, None]
        if len(y_true) == 0:
            return y_true, y_pred, timestamps

        # No chunking — nothing to reshape
        if k_per_chunk is None or k_per_chunk <= 1:
            return y_true, y_pred, timestamps

        # Chunking is active — these must hold
        assert behavioral_bin_size_ms is not None, (
            f"k_per_chunk={k_per_chunk} but behavioral_bin_size_ms is None"
        )
        assert y_pred.shape[1] == y_true.shape[1], (
            f"y_true and y_pred feature dims differ: {y_true.shape[1]} vs {y_pred.shape[1]}"
        )
        assert y_true.shape[1] % k_per_chunk == 0, (
            f"y_true feature dim {y_true.shape[1]} not divisible by k_per_chunk={k_per_chunk}"
        )

        target_dim = y_true.shape[1] // k_per_chunk
        if target_dim == 0:
            return y_true, y_pred, timestamps

        expanded_true = y_true.reshape(len(y_true), k_per_chunk, target_dim).reshape(
            -1, target_dim
        )
        expanded_pred = y_pred.reshape(len(y_pred), k_per_chunk, target_dim).reshape(
            -1, target_dim
        )
        within_chunk_offsets = np.arange(k_per_chunk, dtype=np.float64) * (
            behavioral_bin_size_ms / 1000.0
        )
        expanded_ts = (timestamps[:, None] + within_chunk_offsets[None, :]).reshape(-1)
        return expanded_true, expanded_pred, expanded_ts


@register_metric
class BehavioralTemporalDecoderMetric(BaseMetric):
    """Predict behavior from full grouped latent sequences using a temporal CNN."""

    name = "behavioral_temporal_decoder"
    category = MetricCategory.DOWNSTREAM_DECODING
    direction = MetricDirection.HIGHER_IS_BETTER
    level = MetricLevel.POPULATION
    value_range = (None, 1.0)
    supported_data_types: set[DataType] = {DataType.SPIKES, DataType.SPIKE_TIMES, DataType.RATES}

    grouping_transition: Literal["change", "decrease", "increase", "from_zero"]
    # Input-resolution behavioral data is available via intermediates["behavioral_data_input_res"].
    # See align_behavioral_to_chunks() and build_eval_bin_mask() in downstream_prediction_utils.py,
    # and docs/eval_metrics.md for the filtering/alignment API.
    _required_intermediates = [
        "latents",
        "behavioral_data_token_res",
        "token_timestamps",
        "token_bin_size_ms",
    ]
    _default_params: ClassVar[dict[str, Any]] = {
        "target_field": "cursor_pos",
        "grouping_field": "trial_id",
        "grouping_transition": "change",
        "hidden_channels": 32,
        "kernel_size": 5,
        "dropout": 0.1,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "num_epochs": 50,
        "eval_every": 5,
        "early_stopping": True,
        "early_stopping_patience": 5,
        "random_seed": 42,
        "device": "auto",
        "drop_invalid": True,
        "include_phases": None,
        "filter_field": None,
        "filter_values": None,
        "sample_trial_count": 4,
    }

    def __init__(
        self,
        target_field: str = "cursor_pos",
        grouping_field: str | None = "trial_id",
        grouping_transition: Literal["change", "decrease", "increase", "from_zero"] = "change",
        hidden_channels: int = 32,
        kernel_size: int = 5,
        dropout: float = 0.1,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        num_epochs: int = 50,
        eval_every: int = 5,
        early_stopping: bool = True,
        early_stopping_patience: int = 5,
        random_seed: int = 42,
        device: str = "auto",
        drop_invalid: bool = True,
        include_phases: list[str] | None = None,
        filter_field: str | None = None,
        filter_values: list[Any] | None = None,
        sample_trial_count: int = 4,
    ):
        super().__init__(
            target_field=target_field,
            grouping_field=grouping_field,
            grouping_transition=grouping_transition,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            num_epochs=num_epochs,
            eval_every=eval_every,
            early_stopping=early_stopping,
            early_stopping_patience=early_stopping_patience,
            random_seed=random_seed,
            device=device,
            drop_invalid=drop_invalid,
            include_phases=include_phases,
            filter_field=filter_field,
            filter_values=filter_values,
            sample_trial_count=sample_trial_count,
        )
        self.target_field = target_field
        self.grouping_field = grouping_field
        self.grouping_transition = grouping_transition
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.eval_every = eval_every
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.random_seed = random_seed
        self.device = device
        self.drop_invalid = drop_invalid
        self.include_phases = include_phases
        self.filter_field = filter_field
        self.filter_values = filter_values
        self.sample_trial_count = sample_trial_count

    def compute(self, intermediates: dict[str, Any]) -> MetricResult:
        token_bin_size_ms = intermediates.get("token_bin_size_ms")
        if token_bin_size_ms is None:
            raise ValueError(
                "token_bin_size_ms is required for behavioral_temporal_decoder but was not "
                "provided in intermediates. Ensure the tokenizer reports token_bin_size_ms."
            )

        behavioral_data = intermediates["behavioral_data_token_res"]
        if behavioral_data is None or self.target_field not in behavioral_data:
            available = list(behavioral_data.keys()) if behavioral_data else []
            raise ValueError(
                f"Behavioral field '{self.target_field}' not found in data. Available: {available}"
            )
        if self.grouping_field is None:
            raise ValueError(
                "behavioral_temporal_decoder requires grouping_field to avoid crossing "
                "sequence boundaries during train/val splitting."
            )

        # chunk_window_ms=token_bin_size_ms → 1 token per chunk, equivalent to
        # the pre-chunking code path (flat token-level preprocessing).
        (
            X,
            behavioral_data,
            _behavior_input_res,
            token_timestamps,
            _chunk_indices,
            preprocessing_metadata,
        ) = preprocess_downstream_data(
            intermediates["latents"],
            behavioral_data,
            intermediates.get("token_timestamps"),
            token_bin_size_ms=token_bin_size_ms,
            drop_invalid=self.drop_invalid,
            include_phases=self.include_phases,
            filter_field=self.filter_field,
            filter_values=self.filter_values,
            chunk_window_ms=token_bin_size_ms,
        )
        if preprocessing_metadata.get("is_discrete_index_like", False):
            latent_layout = preprocessing_metadata.get("latent_layout", "unknown")
            raise ValueError(
                "behavioral_temporal_decoder requires continuous latent features; "
                f"received discrete token indices with latent_layout='{latent_layout}'"
            )
        assert behavioral_data is not None
        y = _ensure_2d_targets(behavioral_data[self.target_field])

        if self.grouping_field not in behavioral_data:
            raise ValueError(
                f"Grouping field '{self.grouping_field}' not found in behavioral_data. "
                f"target_field='{self.target_field}', "
                f"available: {list(behavioral_data.keys())}"
            )
        grouping_values = np.asarray(behavioral_data[self.grouping_field])

        x_finite_mask = np.all(np.isfinite(X), axis=1)
        y_finite_mask = np.all(np.isfinite(y), axis=1)
        valid_mask = x_finite_mask & y_finite_mask

        n_dropped = int(len(valid_mask) - np.count_nonzero(valid_mask))
        if not np.any(valid_mask):
            raise ValueError(f"Target '{self.target_field}' has no finite samples after filtering")
        if n_dropped:
            X = X[valid_mask]
            y = y[valid_mask]
            token_timestamps = np.asarray(token_timestamps)[valid_mask]
            grouping_values = grouping_values[valid_mask]

        groups = compute_groups(grouping_values, self.grouping_transition)
        per_channel_unique = [len(np.unique(y[:, col])) for col in range(y.shape[1])]
        if any(n < 2 for n in per_channel_unique):
            constant_channels = [i for i, n in enumerate(per_channel_unique) if n < 2]
            raise ValueError(
                f"Target '{self.target_field}' has < 2 unique values in "
                f"channel(s) {constant_channels} "
                f"(per_channel_unique={per_channel_unique})"
            )

        train_idx, val_idx = _split_grouped_train_val(
            groups,
            random_seed=self.random_seed,
        )
        train_data = _build_temporal_sequence_dataset(
            X,
            y,
            groups,
            train_idx,
            token_timestamps=np.asarray(token_timestamps),
        )
        val_data = _build_temporal_sequence_dataset(
            X,
            y,
            groups,
            val_idx,
            token_timestamps=np.asarray(token_timestamps),
        )

        # Check for single-token sequences (e.g. tokenizers with compress_t=True
        # that produce one token per window). Temporal decoding is meaningless
        # when every sequence has length 1 — skip the metric entirely.
        train_seq_lengths = [seq.shape[-1] for seq in train_data["inputs"]]
        if max(train_seq_lengths) <= 1:
            raise ValueError(
                "behavioral_temporal_decoder: all training sequences have length <= 1 "
                "(likely a single-token-per-window tokenizer with compress_t=True). "
                "Temporal decoding requires sequences with more than one time step. "
                "Skipping metric."
            )

        target_mean = y[train_idx].mean(axis=0).astype(np.float32)
        target_std = y[train_idx].std(axis=0).astype(np.float32).clip(min=1e-6)
        train_targets_std = [
            ((target - target_mean[:, None]) / target_std[:, None]).astype(np.float32)
            for target in train_data["targets"]
        ]

        requested_device = self.device
        fit_device = _resolve_device(requested_device)

        # Enable gradients explicitly: this metric may run inside Lightning's
        # validation loop which wraps callbacks in torch.no_grad().
        with torch.enable_grad():
            torch.manual_seed(self.random_seed)
            if fit_device.type == "cuda":
                torch.cuda.manual_seed_all(self.random_seed)

            model = TemporalBehaviorDecoder(
                in_channels=train_data["inputs"][0].shape[0],
                out_channels=y.shape[1],
                hidden_channels=self.hidden_channels,
                kernel_size=self.kernel_size,
                dropout=self.dropout,
            ).to(fit_device)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )

            train_input_tensors = _sequence_arrays_to_tensors(
                train_data["inputs"],
                device=fit_device,
            )
            train_target_tensors = _sequence_arrays_to_tensors(
                train_targets_std,
                device=fit_device,
            )
            val_input_tensors = _sequence_arrays_to_tensors(
                val_data["inputs"],
                device=fit_device,
            )

            # Pre-pad all training sequences into a single batch for vectorized
            # forward/backward passes (major speedup vs per-sequence loop).
            train_input_batch, train_mask = _pad_sequences_to_batch(train_input_tensors)
            train_target_batch, _ = _pad_sequences_to_batch(train_target_tensors)

            best_state = None
            best_val_r2 = -np.inf
            best_epoch = -1
            final_train_mse = float("nan")
            evals_without_improvement = 0
            stopped_epoch = self.num_epochs - 1

            for epoch in range(self.num_epochs):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                train_pred = model(train_input_batch)
                train_loss = _masked_mse_loss(train_pred, train_target_batch, train_mask)
                train_loss.backward()
                optimizer.step()
                final_train_mse = float(train_loss.item())

                should_eval = epoch == self.num_epochs - 1 or (epoch + 1) % self.eval_every == 0
                if not should_eval:
                    continue

                val_sequence_pred = _predict_temporal_sequences(
                    model,
                    val_input_tensors,
                    target_mean=target_mean,
                    target_std=target_std,
                )
                val_agg_pred, val_covered = _flatten_temporal_sequence_predictions(
                    val_sequence_pred,
                    val_data["position_sequences"],
                    sequence_length=len(X),
                )
                val_selected = val_data["selected_positions"]
                val_eval_mask = val_covered[val_selected]
                if not np.all(val_eval_mask):
                    raise ValueError(
                        "Validation split has uncovered tokens after temporal sequence decoding"
                    )

                val_truth = y[val_selected][val_eval_mask]
                val_pred_eval = val_agg_pred[val_selected][val_eval_mask]
                val_r2 = float(r2_score(val_truth, val_pred_eval))
                if val_r2 > best_val_r2:
                    best_val_r2 = val_r2
                    best_epoch = epoch
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                    evals_without_improvement = 0
                else:
                    evals_without_improvement += 1

                if (
                    self.early_stopping
                    and evals_without_improvement >= self.early_stopping_patience
                ):
                    stopped_epoch = epoch
                    logger.debug(
                        "behavioral_temporal_decoder[%s]: early stopping at epoch %d "
                        "(no val R² improvement for %d eval checks). "
                        "train_mse=%.6f, best_val_r2=%.6f (epoch %d)",
                        self.target_field,
                        epoch,
                        self.early_stopping_patience,
                        final_train_mse,
                        best_val_r2,
                        best_epoch,
                    )
                    break

            if not (
                self.early_stopping and evals_without_improvement >= self.early_stopping_patience
            ):
                stopped_epoch = self.num_epochs - 1
                logger.debug(
                    "behavioral_temporal_decoder[%s]: training completed all %d epochs. "
                    "train_mse=%.6f, best_val_r2=%.6f (epoch %d)",
                    self.target_field,
                    self.num_epochs,
                    final_train_mse,
                    best_val_r2,
                    best_epoch,
                )

        # Free training tensors to release GPU memory before final inference.
        del train_input_tensors, train_target_tensors
        if fit_device.type == "cuda":
            torch.cuda.empty_cache()

        if best_state is None:
            raise RuntimeError(
                "Temporal decoder training finished without a validation checkpoint"
            )
        model.load_state_dict(best_state)

        val_sequence_pred = _predict_temporal_sequences(
            model,
            val_input_tensors,
            target_mean=target_mean,
            target_std=target_std,
        )
        val_agg_pred, val_covered = _flatten_temporal_sequence_predictions(
            val_sequence_pred,
            val_data["position_sequences"],
            sequence_length=len(X),
        )
        val_selected = val_data["selected_positions"]
        val_eval_mask = val_covered[val_selected]
        if not np.all(val_eval_mask):
            raise ValueError("Validation split has uncovered tokens after restoring best model")
        val_truth = y[val_selected][val_eval_mask]
        val_pred_eval = val_agg_pred[val_selected][val_eval_mask]
        final_val_r2 = float(r2_score(val_truth, val_pred_eval))

        val_trial_ids = grouping_values[val_selected][val_eval_mask]
        val_timestamps = np.asarray(token_timestamps)[val_selected][val_eval_mask]
        sample_trials = _build_sample_trials(
            y_val=val_truth,
            y_pred=val_pred_eval,
            timestamps=val_timestamps,
            trial_ids=val_trial_ids,
            sample_trial_count=self.sample_trial_count,
        )
        sample_trials_concatenated = _build_concatenated_trials(
            y_val=val_truth,
            y_pred=val_pred_eval,
            timestamps=val_timestamps,
            trial_ids=val_trial_ids,
            n_trials=5,
        )

        metadata = {
            "target_field": self.target_field,
            "grouping_field": self.grouping_field,
            "grouping_transition": self.grouping_transition,
            "requested_device": requested_device,
            "fit_device": str(fit_device),
            "random_seed": self.random_seed,
            "sequence_mode": "full_group_sequences",
            "behavior_alignment": "prealigned_token_timestamps",
            "hidden_channels": self.hidden_channels,
            "kernel_size": self.kernel_size,
            "receptive_field_tokens": self.kernel_size,
            "dropout": self.dropout,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "num_epochs": self.num_epochs,
            "eval_every": self.eval_every,
            "early_stopping": self.early_stopping,
            "early_stopping_patience": self.early_stopping_patience,
            "stopped_epoch": stopped_epoch,
            "best_epoch": best_epoch,
            "best_val_r2": float(best_val_r2),
            "val_r2": final_val_r2,
            "evaluation_split": "val",
            "final_train_mse": final_train_mse,
            "n_samples_train": len(train_idx),
            "n_samples_val": len(val_idx),
            "n_samples_val_covered": int(np.count_nonzero(val_eval_mask)),
            "n_groups_train": train_data["n_groups_total"],
            "n_groups_val": val_data["n_groups_total"],
            "n_groups_train_split_by_time_gap": train_data["n_groups_split_by_time_gap"],
            "n_groups_val_split_by_time_gap": val_data["n_groups_split_by_time_gap"],
            "n_sequences_train": train_data["n_sequences"],
            "n_sequences_val": val_data["n_sequences"],
            "per_channel_unique_values": per_channel_unique,
            "per_channel_r2": _compute_r2_by_channel(
                val_truth,
                val_pred_eval,
                self.target_field,
            ),
            "n_samples_dropped_non_finite": n_dropped,
            "n_nan_or_inf_x": int((~x_finite_mask).sum()),
            "n_nan_or_inf_y": int((~y_finite_mask).sum()),
            "sample_trials": sample_trials,
            **preprocessing_metadata,
        }
        if sample_trials_concatenated is not None:
            metadata["sample_trials_concatenated"] = sample_trials_concatenated

        return self._make_result(
            raw_value=final_val_r2,
            metadata=metadata,
        )


@register_metric
class BehavioralClassificationMetric(BaseMetric):
    """Predict discrete behavioral variables from neural latents using Ridge classification.

    Uses RidgeClassifierCV with internal cross-validation to select the best alpha.
    Returns validation accuracy from cross-validation.

    Args:
        target_field: Name of behavioral field to predict (e.g., 'trial_target_id', 'trial_result')
        chunk_window_ms: Chunk tokens into chunks of this size. One chunk is the predictor for
                         one sample.Must be a multiple of token bin size.
                        **Allows to pass a list of values to sweep over.**
        behavioral_bin_size_ms: On what resolution to predict the target field.
                                I don't think this is used for classification.
        grouping_field: Field to use for trial/group boundaries in cross-validation splits.
            If None, uses target_field (contiguous same-label blocks become groups).
            Common choices: 'trial_id', 'movement_phase'.
        grouping_transition: How to detect group boundaries from grouping_field values:
            - 'change': New group when value changes (default, use with trial_id)
            - 'decrease': New group when value decreases (use with movement_phase)
            - 'increase': New group when value increases
            - 'from_zero': New group when transitioning from <=0 to >0
        standardize: If True, standardize the latents to have mean 0 and variance 1.
                 (Standardizes the validation split by training set statistics, I believe.)
        alphas: Regularization strengths to test (default: 0.1 to 1e7)
        n_splits: Number of outer crossvalidation splits. Default: None, which means 3 splits.
        cv: Cross-validation strategy for alpha selection inside RidgeClassifierCV.
            None (default) = efficient leave-one-out via GCV (fast, closed-form).
            int = k-fold CV (slower, but useful for very large datasets).
        balance_classes: If True, weight classes inversely proportional to their frequency.
            This helps with imbalanced datasets but may reduce overall accuracy. Default: False.


        Filtering parameters: See `docs/eval_metrics.md` . For classification, we filter once,
        at token-chunk level.
        Can filter by:
            - phase (include_phases)
            - filtering out invalid trials (drop_invalid)
            - custom field (filter_field, filter_values)
            - label values (i.e. the prediction target, only include some classes)
            chunk_drop_threshold has an effect on all of them.

        Chunk size sweep - how it works:
        - Scalar: single run (incl. inner (cv) and outer crossvalidation)
        - List with 1 element: single run
        - List with 2+ elements: runs the full computation (preprocessing, CV, etc.) for each
                chunk size independently. Reports the smallest successful chunk size
                >= 200 ms.
            Metadata includes:
            - chunk_window_sweep: dict mapping each tried chunk_ms → {raw_value, status}
              (or {error, status} for failures)
            - chunk_window_sweep_values: the original list
            - chunk_window_ms: reflects the selected chunk size

        Returns:
            MetricResult with validation accuracy as raw_value
    """

    name = "behavioral_classification"
    category = MetricCategory.DOWNSTREAM_DECODING
    direction = MetricDirection.HIGHER_IS_BETTER
    level = MetricLevel.POPULATION
    _required_intermediates = [
        "latents",
        "behavioral_data_token_res",
        "token_timestamps",
        "behavioral_data_input_res",
        "sampling_rate",
        "token_bin_size_ms",
    ]

    supported_data_types: set[DataType] = {DataType.SPIKES, DataType.SPIKE_TIMES, DataType.RATES}

    grouping_field: str | None
    grouping_transition: Literal["change", "decrease", "increase", "from_zero"]

    def __init__(
        self,
        target_field: str = "trial_target_id",
        grouping_field: str | None = "trial_id",
        grouping_transition: Literal["change", "decrease", "increase", "from_zero"] = "change",
        alphas: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6, 1e7),
        cv: int | None = None,
        n_splits: int | None = None,
        standardize: bool = False,
        balance_classes: bool = False,
        drop_invalid: bool = True,
        include_phases: list[str] | None = None,
        filter_field: str | None = None,
        filter_values: list[Any] | None = None,
        chunk_window_ms: float | list[float] = 1000,
        chunk_drop_threshold: float = 0.25,
        n_features: int | None = None,
        feature_reduction: str = "none",
        behavioral_bin_size_ms: float | None = None,
        excluded_label_values: list[int | float] | None = None,
        nan_drop_threshold: float = 0.25,
        classification_majority_threshold: float = 0.75,
    ):
        if cv is not None and cv < 2:
            raise ValueError(f"cv must be None (for efficient LOO/GCV) or >= 2, got cv={cv}")
        self.target_field = target_field
        self.grouping_field = grouping_field
        self.grouping_transition = grouping_transition
        self.alphas = alphas
        self.cv = cv
        self.n_splits = n_splits
        self.standardize = standardize
        self.balance_classes = balance_classes
        self.drop_invalid = drop_invalid
        self.include_phases = include_phases
        self.filter_field = filter_field
        self.filter_values = filter_values
        self._chunk_window_ms_values = _normalize_chunk_window_ms(chunk_window_ms)
        self.chunk_window_ms = self._chunk_window_ms_values[0]
        self.chunk_drop_threshold = chunk_drop_threshold
        self.n_features = n_features
        self.feature_reduction = feature_reduction
        self.behavioral_bin_size_ms = behavioral_bin_size_ms
        self.excluded_label_values = excluded_label_values
        self.nan_drop_threshold = nan_drop_threshold
        self.classification_majority_threshold = classification_majority_threshold

    def compute(self, intermediates: dict[str, Any]) -> MetricResult:
        """Dispatch to single chunk size or sweep over multiple.


        When ``chunk_window_ms`` was configured as a list, runs the full
        computation for each value and returns the smallest valid chunk >= 200 ms,
        with per-chunk-size results stored in ``metadata['chunk_window_sweep']``.
        """
        if len(self._chunk_window_ms_values) > 1:
            return _run_chunk_window_sweep(
                self,
                self._chunk_window_ms_values,
                intermediates,
                self._compute_single_chunk_size,
            )
        self.chunk_window_ms = self._chunk_window_ms_values[0]
        return self._compute_single_chunk_size(intermediates)

    def _compute_single_chunk_size(self, intermediates: dict[str, Any]) -> MetricResult:
        """Compute classification accuracy from latents to behavioral labels.

        Args:
            intermediates: Dict containing:
                - latents: Neural latent representations, shape (n_samples, latent_dim)
                - behavioral_data: Dict of behavioral arrays
                - token_timestamps: Timestamps for each token, shape (n_samples,)

        ``token_timestamps`` length and behavioral alignment are validated in
        ``preprocess_downstream_data`` before classification starts.

        Returns:
            MetricResult with validation accuracy as raw_value
        """
        token_bin_size_ms = intermediates.get("token_bin_size_ms")
        if token_bin_size_ms is None:
            raise ValueError(
                "token_bin_size_ms is required for downstream metrics but was not "
                "provided in intermediates. Ensure the tokenizer reports token_bin_size_ms."
            )

        behavioral_data_token_res = intermediates["behavioral_data_token_res"]
        behavioral_data_input_res = intermediates.get("behavioral_data_input_res")

        target_available = (
            behavioral_data_token_res is not None
            and self.target_field in behavioral_data_token_res
        ) or (
            behavioral_data_input_res is not None
            and self.target_field in behavioral_data_input_res
        )
        if not target_available:
            available = set()
            if behavioral_data_token_res:
                available.update(behavioral_data_token_res.keys())
            if behavioral_data_input_res:
                available.update(behavioral_data_input_res.keys())
            raise ValueError(
                f"Behavioral field '{self.target_field}' not found in data. "
                f"Available: {sorted(available)}"
            )

        # 1. Preprocessing: filter + chunk X (unified filtering for classification)
        (
            X,
            behavior_token_res,
            behavior_input_res,
            chunk_timestamps,
            chunk_indices,
            preprocessing_metadata,
        ) = preprocess_downstream_data(
            intermediates["latents"],
            behavioral_data_token_res,
            intermediates.get("token_timestamps"),
            token_bin_size_ms=token_bin_size_ms,
            drop_invalid=self.drop_invalid,
            include_phases=self.include_phases,
            filter_field=self.filter_field,
            filter_values=self.filter_values,
            chunk_window_ms=self.chunk_window_ms,
            chunk_drop_threshold=self.chunk_drop_threshold,
            behavioral_data_input_res=behavioral_data_input_res,
        )

        # 2. Align behavioral data to chunks (NaN-aware majority vote)
        if behavior_input_res is not None:
            y, grouping_values, keep_mask, align_metadata = align_behavioral_to_chunks(
                chunk_indices=chunk_indices,
                chunk_window_ms=self.chunk_window_ms,
                behavioral_data_input_res=behavior_input_res,
                sampling_rate=intermediates["sampling_rate"],
                target_field=self.target_field,
                mode="classification",
                grouping_field=self.grouping_field,
                classification_majority_threshold=self.classification_majority_threshold,
                behavioral_bin_size_ms=self.behavioral_bin_size_ms,
                excluded_label_values=self.excluded_label_values,
                nan_drop_threshold=self.nan_drop_threshold,
            )
            preprocessing_metadata.update(align_metadata)

            # Apply keep_mask from alignment
            assert len(keep_mask) == len(X), (
                f"Classification keep_mask length {len(keep_mask)} != X length {len(X)} "
                f"for target_field='{self.target_field}'. "
                f"preprocess returned n_chunks={len(chunk_timestamps)}, "
                f"chunk_indices length={len(chunk_indices)}"
            )
            X = X[keep_mask]
            chunk_timestamps = chunk_timestamps[keep_mask]
        else:
            # No input-res data — extract y from token-res behavioral data.
            # This path is used when the full pipeline metadata is not available.
            assert behavior_token_res is not None and self.target_field in behavior_token_res, (
                f"target_field '{self.target_field}' not in behavioral_data_token_res. "
                f"Available: {sorted((behavior_token_res or {}).keys())}"
            )
            y = np.asarray(behavior_token_res[self.target_field])
            # Get grouping values
            grouping_values = None
            if self.grouping_field is not None:
                assert self.grouping_field in behavior_token_res, (
                    f"Grouping field '{self.grouping_field}' not in behavioral_data_token_res. "
                    f"Available: {sorted(behavior_token_res.keys())}"
                )
                grouping_values = np.asarray(behavior_token_res[self.grouping_field])

        # Multi-dimensional targets: check if we can treat combinations as discrete classes
        if y.ndim > 1:
            # Find unique row combinations
            unique_combos, inverse_indices = np.unique(y, axis=0, return_inverse=True)
            n_unique_combos = len(unique_combos)
            n_samples = len(y)
            combo_threshold = min(15, n_samples // 10)

            if n_unique_combos <= combo_threshold:
                # Few unique combinations - treat each combo as a discrete class
                # Use inverse_indices as 1D labels (integers 0 to n_unique_combos-1)
                y_combo = inverse_indices
                result = self._classify_1d_target(X, y_combo, grouping_values)
                # Add info about the combo mapping to metadata
                result.metadata["multidim_mode"] = "combinations"
                result.metadata["n_unique_combinations"] = n_unique_combos
                result.metadata["combination_threshold"] = combo_threshold
                result.metadata["unique_combinations"] = unique_combos.tolist()
                # Include a compact record of any preprocessing that was applied.
                result.metadata.update(preprocessing_metadata)
                return result
            else:
                # Many unique combinations - classify each dimension independently
                result = self._compute_multidim(X, y, grouping_values, n_unique_combos)
                # Include a compact record of any preprocessing that was applied.
                result.metadata.update(preprocessing_metadata)
                return result

        result = self._classify_1d_target(X, y, grouping_values)
        # Include a compact record of any preprocessing that was applied.
        result.metadata.update(preprocessing_metadata)
        return result

    def _compute_multidim(
        self,
        X: np.ndarray,
        y: np.ndarray,
        grouping_values: np.ndarray | None,
        n_unique_combos: int | None = None,
    ) -> MetricResult:
        """Classify each dimension of a multi-dimensional target independently."""
        dim_results: list[tuple[int, MetricResult]] = []
        dim_errors: dict[int, str] = {}
        for dim_idx in range(y.shape[1]):
            try:
                result = self._classify_1d_target(X, y[:, dim_idx], grouping_values)
                dim_results.append((dim_idx, result))
            except (ValueError, AssertionError) as e:
                dim_errors[dim_idx] = str(e)

        if not dim_results:
            raise ValueError(
                f"All {y.shape[1]} dimensions failed classification for "
                f"'{self.target_field}'. Per-dimension errors: {dim_errors}"
            )

        mean_acc = np.mean([float(r.raw_value) for _, r in dim_results])
        metadata = {
            "target_field": self.target_field,
            "multidim_mode": "per_dimension",
            "n_dimensions": y.shape[1],
            "n_valid_dimensions": len(dim_results),
            "n_failed_dimensions": len(dim_errors),
            "per_dimension_accuracy": {idx: float(r.raw_value) for idx, r in dim_results},
            "per_dimension": {idx: r.metadata for idx, r in dim_results},
            **({"per_dimension_errors": dim_errors} if dim_errors else {}),
        }
        if n_unique_combos is not None:
            metadata["n_unique_combinations"] = n_unique_combos
            metadata["combination_threshold"] = min(15, len(y) // 10)
        return self._make_result(raw_value=float(mean_acc), metadata=metadata)

    def _classify_1d_target(
        self,
        X: np.ndarray,
        y: np.ndarray,
        grouping_values: np.ndarray | None,
    ) -> MetricResult:
        """Run classification for a single 1D target variable."""
        # sklearn rejects float targets as 'continuous'. When there are few
        # unique values (e.g. trial_target_dir stored as radians), map them
        # to integer class labels so RidgeClassifierCV accepts them.
        if np.issubdtype(y.dtype, np.floating):
            unique_vals, y = np.unique(y, return_inverse=True)

        # Check for sufficient unique classes and minimum samples per class
        unique_classes, class_counts = np.unique(y, return_counts=True)

        # Filter classes with at least 8 samples (reduced from 10 to avoid
        # aggressive dropping when chunk size × label-extension timings
        # conspire to put classes near the boundary — e.g. Cowley at 200ms
        # chunks where 11–12 reps per image can dip below 10 after
        # alignment-dependent chunk survival).
        MIN_SAMPLES_PER_CLASS = 8
        valid_class_mask = class_counts >= MIN_SAMPLES_PER_CLASS
        valid_classes = unique_classes[valid_class_mask]

        if len(valid_classes) < 2:
            class_counts_dict = dict(
                zip(unique_classes.tolist(), class_counts.tolist(), strict=True)
            )
            raise ValueError(
                f"Target '{self.target_field}' has < 2 classes with >= "
                f"{MIN_SAMPLES_PER_CLASS} samples. Class counts: {class_counts_dict}"
            )

        # Filter data to only include valid classes
        data_mask = np.isin(y, valid_classes)
        X = X[data_mask]
        y = y[data_mask]

        # Determine grouping for cross-validation splits
        # If grouping_values provided, apply same filter and use for grouping
        # Otherwise, use target labels (y) for grouping
        if grouping_values is not None:
            grouping_values = grouping_values[data_mask]
            source_values = grouping_values
        else:
            source_values = y

        # Compute groups based on transition type
        groups = compute_groups(source_values, self.grouping_transition)

        n_groups = np.max(groups) + 1
        if n_groups < 2:
            raise ValueError(
                f"Target '{self.target_field}': only {n_groups} trial(s) found "
                f"(grouping_field='{self.grouping_field}', "
                f"transition='{self.grouping_transition}'). "
                f"Need >= 2 for trial-aware split."
            )

        # Use StratifiedGroupKFold (or single GroupShuffleSplit) for accuracy estimate

        n_splits = self.n_splits if self.n_splits is not None else min(3, n_groups)
        use_single_split = n_splits == 1

        if not use_single_split:
            n_splits = min(n_splits, n_groups)
            if n_splits < 2:
                raise ValueError(
                    f"Target '{self.target_field}': only {n_groups} trial(s) found. "
                    f"Need at least 2 for cross-validation."
                )
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
            splits = list(splitter.split(X, y, groups=groups))
        else:
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            splits = list(gss.split(X, y, groups=groups))

        fold_accuracies = []
        fold_chance_accuracies = []
        fold_alphas: list[float] = []
        total_train_samples = 0
        total_val_samples = 0
        total_trials_train = 0
        total_trials_val = 0
        all_train_classes = set()
        all_val_classes = set()
        per_class_accuracies: dict[Any, list[float]] = {}
        aggregated_train_class_counts: dict[int, list[int]] = {}
        aggregated_val_class_counts: dict[int, list[int]] = {}

        for train_idx, val_idx in splits:
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Only evaluate on classes seen in training
            train_classes = set(np.unique(y_train))
            val_classes = set(np.unique(y_val))
            overlap_classes = train_classes & val_classes

            if len(overlap_classes) == 0:
                continue  # Skip fold if no class overlap

            # Filter val to only overlapping classes
            val_mask = np.isin(y_val, list(overlap_classes))
            if val_mask.sum() == 0:
                continue

            X_val_filtered = X_val[val_mask]
            y_val_filtered = y_val[val_mask]

            # Feature reduction (fit on train fold only)
            reducer = FeatureReducer(
                method=self.feature_reduction,
                n_features=self.n_features,
            )
            X_train = reducer.fit_transform(X_train)
            X_val_filtered = reducer.transform(X_val_filtered)

            if self.standardize:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_val_filtered = scaler.transform(X_val_filtered)

            # Fit and evaluate
            if self.cv is not None:
                cls_cv = min(self.cv, len(y_train))
                if cls_cv < 2:
                    continue
                model = RidgeClassifierCV(
                    alphas=self.alphas,
                    cv=cls_cv,
                    class_weight="balanced" if self.balance_classes else None,
                )
            else:
                if len(y_train) < 2:
                    continue
                model = RidgeClassifierCV(
                    alphas=self.alphas,
                    class_weight="balanced" if self.balance_classes else None,
                )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", LinAlgWarning)
                model.fit(X_train, y_train)
            _emit_linalg_warning_once(caught)
            y_pred = model.predict(X_val_filtered)
            fold_acc = accuracy_score(y_val_filtered, y_pred)

            # Chance accuracy for this fold (based on filtered val set)
            unique_val, counts_val = np.unique(y_val_filtered, return_counts=True)
            fold_chance = counts_val.max() / counts_val.sum()

            # Per-class accuracy for this fold
            for cls in overlap_classes:
                cls_mask = y_val_filtered == cls
                if cls_mask.sum() > 0:
                    cls_acc = accuracy_score(y_val_filtered[cls_mask], y_pred[cls_mask])
                    if cls not in per_class_accuracies:
                        per_class_accuracies[cls] = []
                    per_class_accuracies[cls].append(cls_acc)

            # Track class counts for this fold
            train_class_counts = dict(zip(*np.unique(y_train, return_counts=True), strict=True))
            val_class_counts = dict(
                zip(*np.unique(y_val_filtered, return_counts=True), strict=True)
            )
            for cls in train_class_counts:
                cls_key = int(cls) if hasattr(cls, "item") else cls
                if cls_key not in aggregated_train_class_counts:
                    aggregated_train_class_counts[cls_key] = []
                aggregated_train_class_counts[cls_key].append(train_class_counts[cls])
            for cls in val_class_counts:
                cls_key = int(cls) if hasattr(cls, "item") else cls
                if cls_key not in aggregated_val_class_counts:
                    aggregated_val_class_counts[cls_key] = []
                aggregated_val_class_counts[cls_key].append(val_class_counts[cls])

            fold_alphas.append(float(cast(float, model.alpha_)))
            fold_accuracies.append(fold_acc)
            fold_chance_accuracies.append(fold_chance)
            total_train_samples += len(y_train)
            total_val_samples += len(y_val_filtered)
            total_trials_train += len(np.unique(groups[train_idx]))
            total_trials_val += len(np.unique(groups[val_idx]))
            all_train_classes.update(train_classes)
            all_val_classes.update(val_classes)

        if len(fold_accuracies) == 0:
            raise ValueError(
                f"Target '{self.target_field}': no valid folds with class overlap found "
                f"(n_groups={n_groups}, n_splits={n_splits})"
            )

        # Average across folds
        mean_accuracy = np.mean(fold_accuracies)
        mean_chance = np.mean(fold_chance_accuracies)
        class_overlap = all_train_classes & all_val_classes

        # Compute mean per-class accuracy
        mean_per_class_accuracy = {
            int(cls) if hasattr(cls, "item") else cls: float(np.mean(accs))
            for cls, accs in per_class_accuracies.items()
        }

        # Compute mean class counts across folds
        mean_train_class_counts = {
            cls: int(np.mean(counts)) for cls, counts in aggregated_train_class_counts.items()
        }
        mean_val_class_counts = {
            cls: int(np.mean(counts)) for cls, counts in aggregated_val_class_counts.items()
        }

        return self._make_result(
            raw_value=float(mean_accuracy),
            metadata={
                "target_field": self.target_field,
                "balance_classes": self.balance_classes,
                "standardize": self.standardize,
                # Data dimensions
                "n_samples_total": len(X),
                "latent_dim": X.shape[1] if X.ndim > 1 else 1,
                "n_trials_total": n_groups,
                # Cross-validation info
                "n_splits": len(fold_accuracies),
                "fold_alphas": fold_alphas,
                "fold_accuracies": [float(a) for a in fold_accuracies],
                "accuracy_std": (
                    float(np.std(fold_accuracies)) if len(fold_accuracies) >= 3 else None
                ),
                "mean_samples_train": total_train_samples // len(fold_accuracies),
                "mean_samples_val": total_val_samples // len(fold_accuracies),
                "mean_trials_train": total_trials_train // len(fold_accuracies),
                "mean_trials_val": total_trials_val // len(fold_accuracies),
                # Class info
                "n_unique_classes": len(valid_classes),
                "n_classes_overlap": len(class_overlap),
                "train_class_counts": mean_train_class_counts,
                "val_class_counts": mean_val_class_counts,
                # Accuracy metrics
                "chance_accuracy": float(mean_chance),
                "above_chance": float(mean_accuracy - mean_chance),
                "per_class_accuracy": mean_per_class_accuracy,
                # Grouping info
                "grouping_field": self.grouping_field,
                "grouping_transition": self.grouping_transition,
                # Feature reduction
                "feature_reduction": self.feature_reduction,
                "n_features_target": self.n_features,
            },
        )
