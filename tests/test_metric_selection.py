from pathlib import Path

import yaml

from ntt.evaluation.core.types import MetricCategory
from ntt.evaluation.utils.metric_selection import (
    get_interval_windowing_allowed_metric_names,
    is_latent_robustness_reconstruction_metric,
    is_reconstruction_metric,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _declared_metric_names(config_path: Path) -> list[str]:
    """Metric names declared via ``defaults: - metrics/<name>@metrics.<name>``."""
    data = yaml.safe_load(config_path.read_text())
    names: list[str] = []
    for entry in data.get("defaults", []):
        if isinstance(entry, str) and entry.startswith("metrics/"):
            names.append(entry.split("/", 1)[1].split("@", 1)[0])
    return names


def test_is_reconstruction_metric_excludes_shared_non_recon_metrics():
    assert not is_reconstruction_metric(
        "behavioral_regression_cursor_pos",
        MetricCategory.DOWNSTREAM_DECODING,
    )
    assert not is_reconstruction_metric(
        "population_structure_preservation",
        MetricCategory.POPULATION_STRUCTURE,
    )
    assert not is_reconstruction_metric("repeats_snr", MetricCategory.POPULATION_STRUCTURE)
    assert is_reconstruction_metric("acg_correlation", MetricCategory.SINGLE_UNIT_STATISTICS)


def test_is_latent_robustness_reconstruction_metric_keeps_structure_preservation_excluded() -> (
    None
):
    assert not is_latent_robustness_reconstruction_metric(
        "population_structure_preservation",
        MetricCategory.POPULATION_STRUCTURE,
    )
    assert not is_latent_robustness_reconstruction_metric(
        "per_unit_structure_preservation",
        MetricCategory.POPULATION_STRUCTURE,
    )


def test_is_latent_robustness_reconstruction_metric_excludes_meta_metrics() -> None:
    assert not is_latent_robustness_reconstruction_metric(
        "robustness_input",
        MetricCategory.POPULATION_STRUCTURE,
    )
    assert not is_latent_robustness_reconstruction_metric(
        "repeats_baseline",
        MetricCategory.POPULATION_STRUCTURE,
    )


def test_per_stimulus_compression_config_metrics_survive_filter():
    """Every metric a per-stimulus pipeline config declares must be runnable
    on the interval-windowed path; otherwise the config silently drops it.

    Guards the ``cowley_per_stimulus_with_compression.yaml`` variant, whose
    whole purpose is to report ``token_dims_per_second`` alongside SNR.
    """
    config_path = (
        _REPO_ROOT / "config/evaluation/pipelines/cowley_per_stimulus_with_compression.yaml"
    )
    declared = _declared_metric_names(config_path)
    assert "token_dims_per_second" in declared  # guard against config drift
    allowed = get_interval_windowing_allowed_metric_names()
    dropped = [name for name in declared if name not in allowed]
    assert not dropped, (
        f"per-stimulus config {config_path.name} declares metrics that the "
        f"interval-windowing filter drops: {dropped}"
    )
