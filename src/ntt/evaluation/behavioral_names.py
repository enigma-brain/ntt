"""Shared naming constants for behavioral evaluation metrics and plots."""

REGRESSION_RESULT_NAMES = frozenset({"behavioral_regression", "behavioral_temporal_decoder"})
REGRESSION_PREFIXES = (
    "behavioral_regression_",
    "behavioral_temporal_decoder_",
)
BEHAVIORAL_NAMES = REGRESSION_RESULT_NAMES | {"behavioral_classification"}
