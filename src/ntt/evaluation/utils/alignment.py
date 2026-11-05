"""Utilities for temporal alignment and resampling of behavioral data."""

import numpy as np

# Default token-level filtering fields.
# These are the shared fields that downstream metrics commonly filter on.
# If metric configs make the required set easy to gather centrally, prefer
# building the actual field list from those configs rather than hardcoding.
DEFAULT_FILTER_FIELDS = [
    "movement_phase",
    "trial_id",
    "trial_target_id",
    "trial_result",
    "trial_is_valid",
    "image_id",
]

# Fields known to be categorical (nearest-neighbor resampling is valid).
# Any field resampled by resample_fields_to_token_timestamps must either
# appear here or have an integer dtype.
KNOWN_CATEGORICAL_FIELDS = [
    "movement_phase",
    "trial_id",
    "trial_target_id",
    "trial_target_dir",
    "trial_result",
    "trial_is_valid",
    "image_id",
]


def resample_fields_to_token_timestamps(
    behavioral_data: dict[str, np.ndarray],
    source_timestamps: np.ndarray,
    target_timestamps: np.ndarray,
    fields: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Resample selected behavioral fields to token timestamps.


    This helper is for the shared upstream filtering path. It resamples only
    the fields needed for token-level filtering using nearest-neighbor
    interpolation. Fields not in ``fields`` (or not present in
    behavioral_data) are skipped.

    Nearest-neighbor resampling is only valid for categorical/integer fields.
    A ``ValueError`` is raised if a requested field is not in
    ``KNOWN_CATEGORICAL_FIELDS`` and does not have an integer dtype.

    Args:
        behavioral_data: Dict mapping field names to arrays at source resolution.
        source_timestamps: Timestamps for behavioral_data entries.
        target_timestamps: Timestamps to resample to (e.g. token centers).
        fields: Which fields to resample. Defaults to DEFAULT_FILTER_FIELDS.

    Returns:
        Dict of resampled token-level filtering fields only.
    """
    if fields is None:
        fields = list(DEFAULT_FILTER_FIELDS)

    source_timestamps = np.asarray(source_timestamps)
    target_timestamps = np.asarray(target_timestamps)

    if len(target_timestamps) == 0:
        return {
            field: np.empty((0, *data.shape[1:]), dtype=data.dtype)
            for field, data in behavioral_data.items()
            if field in fields
        }

    assert len(source_timestamps) > 0, (
        "Cannot resample with empty source_timestamps when target is non-empty."
    )

    result = {}
    for field, data in behavioral_data.items():
        if field not in fields:
            continue

        # Guard: nearest-neighbor is only valid for categorical/integer fields
        if field not in KNOWN_CATEGORICAL_FIELDS and not np.issubdtype(data.dtype, np.integer):
            raise ValueError(
                f"Field '{field}' is not in KNOWN_CATEGORICAL_FIELDS and has "
                f"non-integer dtype '{data.dtype}'. Nearest-neighbor resampling "
                f"is only valid for categorical fields. If '{field}' is "
                f"categorical, add it to KNOWN_CATEGORICAL_FIELDS in alignment.py."
            )

        assert len(data) == len(source_timestamps), (
            f"Field '{field}' length {len(data)} != source_timestamps length "
            f"{len(source_timestamps)}"
        )
        # Nearest-neighbor interpolation
        insertion = np.searchsorted(source_timestamps, target_timestamps)
        insertion = np.clip(insertion, 1, len(source_timestamps) - 1)
        left_dist = np.abs(target_timestamps - source_timestamps[insertion - 1])
        right_dist = np.abs(target_timestamps - source_timestamps[insertion])
        indices = np.where(left_dist <= right_dist, insertion - 1, insertion)
        result[field] = data[indices]

    return result


def resample_behavioral_data(
    behavioral_data: dict[str, np.ndarray],
    source_timestamps: np.ndarray,
    target_timestamps: np.ndarray,
    categorical_fields: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Resample behavioral data to match target timestamps.

    Args:
        behavioral_data: Dict mapping field names to numpy arrays.
            Each array should have the same length as source_timestamps.
        source_timestamps: Timestamps corresponding to the current behavioral data.
        target_timestamps: Timestamps to resample the data to.
        categorical_fields: List of field names that should be treated as categorical
            (using nearest-neighbor interpolation). If None, defaults to
             'movement_phase', 'trial_target_id', 'trial_result' (for perich-miller).

    Returns:
        Dict of resampled behavioral data.
    """
    source_timestamps = np.asarray(source_timestamps)
    target_timestamps = np.asarray(target_timestamps)

    if categorical_fields is None:
        categorical_fields = [
            "movement_phase",
            "trial_id",
            "trial_target_id",
            "trial_result",
        ]

    if len(target_timestamps) == 0:
        return {
            field: np.empty((0, *data.shape[1:]), dtype=data.dtype)
            for field, data in behavioral_data.items()
        }

    if len(source_timestamps) == 0:
        raise ValueError(
            "Cannot resample behavioral data with empty source_timestamps "
            "when target_timestamps is non-empty."
        )

    resampled_data = {}

    for field, data in behavioral_data.items():
        assert len(data) == len(source_timestamps), (
            f"Behavioral field '{field}' has length {len(data)} but "
            f"source_timestamps has length {len(source_timestamps)}. "
            f"All fields must be aligned to source_timestamps."
        )

        if field in categorical_fields or np.issubdtype(data.dtype, np.integer):
            # Nearest neighbor for categorical/integer data
            # Find indices of source_timestamps closest to target_timestamps
            insertion = np.searchsorted(source_timestamps, target_timestamps)
            insertion = np.clip(insertion, 1, len(source_timestamps) - 1)
            left_dist = np.abs(target_timestamps - source_timestamps[insertion - 1])
            right_dist = np.abs(target_timestamps - source_timestamps[insertion])
            indices = np.where(left_dist <= right_dist, insertion - 1, insertion)
            resampled_data[field] = data[indices]
        else:
            # Linear interpolation for continuous data
            if data.ndim == 1:
                resampled_data[field] = np.interp(target_timestamps, source_timestamps, data)
            else:
                # Handle multi-dimensional data (e.g. cursor_pos)
                resampled_dim = []
                for d in range(data.shape[1]):
                    resampled_dim.append(
                        np.interp(target_timestamps, source_timestamps, data[:, d])
                    )
                resampled_data[field] = np.stack(resampled_dim, axis=1)

    return resampled_data
