"""Shared metric-selection helpers for reconstruction-side eval passes."""

from __future__ import annotations

from ntt.evaluation.core.types import MetricCategory

# Categories that are never reconstruction metrics on an
# ``(original, reconstructed)`` or ``(clean, corrupted)`` pair.
_NON_RECON_CATEGORIES: frozenset[MetricCategory] = frozenset(
    {
        MetricCategory.DOWNSTREAM_ENCODING,
        MetricCategory.DOWNSTREAM_DECODING,
    }
)

# Prefix-matched non-reconstruction metrics.
_NON_RECON_METRIC_PREFIXES: tuple[str, ...] = (
    "behavioral_classification",
    "behavioral_regression",
    "behavioral_temporal_decoder",
)

# Exact-name non-reconstruction metrics. These may live in categories such as
# POPULATION / SINGLE_UNIT, so category checks alone are insufficient.
_NON_RECON_METRIC_NAMES: frozenset[str] = frozenset(
    {
        "per_unit_structure_preservation",
        "population_structure_preservation",
        "repeats_snr",
    }
)

# Interval-windowed evaluation currently leads to windows being stitched
# together in a way that leads to misalignment between original and
# reconstruction. SNR and repeats-based metrics would be better served by their
# own script. Here we skip all other metrics and window concatenation altogether
# to prevent erroneously computing other metrics on mismatched
# original/reconstruction.
# (note: compression and behavioral decoding metrics should actually
#  work just fine; generally all simple metrics that don't use the
# reconstruction.)
_INTERVAL_WINDOWING_ALLOWED_METRIC_NAMES: frozenset[str] = frozenset(
    {
        "repeats_snr",
        # Compression metric: reads only latents, never the reconstruction, so
        # it is unaffected by the original/recon misalignment described above.
        "token_dims_per_second",
    }
)

# latent_robustness-specific (M2) exclusions: these are meta-metrics, not primitive
# clean-reconstruction-vs-corrupted-reconstruction scores.
_LATENT_ROBUSTNESS_EXTRA_EXCLUDED_NAMES: frozenset[str] = frozenset(
    {
        "robustness_input",
        "repeats_baseline",
    }
)


def is_reconstruction_metric(
    metric_name: str,
    category: MetricCategory | None = None,
) -> bool:
    """Return True iff the metric is a reconstruction-side metric."""
    if category in _NON_RECON_CATEGORIES:
        return False
    if metric_name in _NON_RECON_METRIC_NAMES:
        return False
    return not any(metric_name.startswith(prefix) for prefix in _NON_RECON_METRIC_PREFIXES)


def is_latent_robustness_reconstruction_metric(
    metric_name: str,
    category: MetricCategory | None = None,
) -> bool:
    """Return True iff the metric should participate in the latent_robustness (M2) sweep."""
    if metric_name in _LATENT_ROBUSTNESS_EXTRA_EXCLUDED_NAMES:
        return False
    return is_reconstruction_metric(metric_name, category)


def get_interval_windowing_allowed_metric_names() -> frozenset[str]:
    """Return the metrics that are explicitly supported on interval windows."""
    return _INTERVAL_WINDOWING_ALLOWED_METRIC_NAMES
