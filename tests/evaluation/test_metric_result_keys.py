"""Metric result keys must be unique.

A duplicate key means one metric's number is reported under another metric's name,
which is indistinguishable from a correct run everywhere downstream — the failure
mode that let the M2 correlogram variants blend silently. Two layers are tested:
the runtime guard in the pipeline, and a check over every eval job's *composed*
metric set so a colliding config is caught in CI rather than after a GPU run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ntt.evaluation.config import instantiate_metrics_from_config
from ntt.evaluation.core.types import MetricResult
from ntt.evaluation.pipeline import _store_result

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "config"


def _result(value: float = 1.0) -> MetricResult:
    return MetricResult(name="m", category=None, direction=None, level=None, raw_value=value)


# ── runtime guard ──────────────────────────────────────────────────────────────
def test_store_result_accepts_distinct_keys():
    results: dict[str, MetricResult] = {}
    _store_result(results, "a", _result(1.0))
    _store_result(results, "b", _result(2.0), expanded_from="a")
    assert [r.raw_value for r in results.values()] == [1.0, 2.0]


def test_store_result_rejects_a_duplicate_display_name():
    results = {"acg_correlation": _result(0.5)}
    with pytest.raises(ValueError, match="Duplicate metric result key 'acg_correlation'"):
        _store_result(results, "acg_correlation", _result(0.9))
    assert results["acg_correlation"].raw_value == 0.5  # original not clobbered


def test_store_result_names_the_expansion_that_shadowed_a_display_name():
    """The error has to say *which* mechanism collided, or it is unactionable."""
    results = {"spike_count_correlation_5ms": _result(0.5)}
    with pytest.raises(ValueError, match="multi-scale expansion of 'spike_count_correlation'"):
        _store_result(
            results,
            "spike_count_correlation_5ms",
            _result(0.9),
            expanded_from="spike_count_correlation",
        )


# ── composed-config check ──────────────────────────────────────────────────────
# The realised metric set is what matters, and it only exists after Hydra composes
# the defaults lists — a per-file scan misses it, because the multi-bin default lives
# in config/evaluation/metrics/<name>.yaml while the suffixed display names are added
# by a different file.
EVAL_JOBS = (
    "eval",
    "eval_hill_climb",
    "eval_light",
    "eval_robustness_latent",
    "eval_robustness_input",
)


def _composed_metrics(job: str) -> dict:
    """The ``metrics`` mapping a job actually runs with, composed like the job does."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(config_dir=str(CONFIG), version_base=None):
        cfg = compose(config_name="main", overrides=[f"job={job}"])
    # resolve=False: we only want the metrics mapping, and resolving pulls in
    # environment-dependent path interpolations we do not need.
    raw = OmegaConf.to_container(cfg, resolve=False)

    def walk(node):
        if isinstance(node, dict):
            if isinstance(node.get("metrics"), dict):
                yield node["metrics"]
            for value in node.values():
                yield from walk(value)

    blocks = list(walk(raw))
    assert blocks, f"job={job} composed without a metrics block"
    return blocks[0]


def _predicted_expansions(display_name: str, metric) -> set[str]:
    """Names ``expand_results`` would synthesise: one per NON-primary bin.

    Mirrors ``SpikeCountCorrelation`` — the SMALLEST bin size is the primary (its
    ``compute`` sorts the bins), reported under ``display_name`` itself.
    """
    bins = getattr(metric, "params", {}).get("bin_sizes_ms")
    if not isinstance(bins, (list, tuple)) or len(bins) < 2:
        return set()
    primary = min(bins)
    out = set()
    for bin_ms in bins:
        if bin_ms == primary:
            continue
        suffix = f"_{int(bin_ms)}ms" if bin_ms == int(bin_ms) else f"_{bin_ms}ms"
        out.add(f"{display_name}{suffix}")
    return out


@pytest.fixture(autouse=True)
def _project_root(monkeypatch):
    # config/main.yaml interpolates ${oc.env:PROJECT_ROOT}; composition touches it.
    monkeypatch.setenv("PROJECT_ROOT", str(REPO))


@pytest.mark.parametrize("job", EVAL_JOBS)
def test_composed_eval_config_has_no_colliding_result_keys(job: str):
    """No metric may report under a name another metric already owns.

    Multi-scale metrics synthesise ``{display_name}_{n}ms`` for every non-primary
    bin, which can shadow a display name declared directly — which is why
    ``config/evaluation/temporal/*.yaml`` hand-null their default-bin entries
    ("Null out default-bin metrics to avoid name collisions"). This turns that
    convention into an enforced invariant for every eval job we actually run.
    """
    block = _composed_metrics(job)
    live = {name: cfg for name, cfg in block.items() if isinstance(cfg, dict)}
    pairs = instantiate_metrics_from_config(live)
    declared = {display_name for display_name, _ in pairs}

    collisions = []
    seen_expansions: dict[str, str] = {}
    for display_name, metric in pairs:
        for synthesized in _predicted_expansions(display_name, metric):
            if synthesized in declared:
                collisions.append(
                    f"{display_name!r} expands to {synthesized!r}, also declared directly"
                )
            if synthesized in seen_expansions:
                collisions.append(
                    f"{display_name!r} and {seen_expansions[synthesized]!r} both expand "
                    f"to {synthesized!r}"
                )
            seen_expansions[synthesized] = display_name

    assert not collisions, f"job={job} colliding metric result keys:\n  " + "\n  ".join(collisions)


def test_at_least_one_eval_job_actually_exercises_multi_bin_expansion():
    """Anti-vacuity: if no composed job has a multi-bin metric, the checks above pass
    without testing anything (the failure mode that hid the M2 collision for weeks)."""
    total = 0
    for job in EVAL_JOBS:
        live = {n: c for n, c in _composed_metrics(job).items() if isinstance(c, dict)}
        for display_name, metric in instantiate_metrics_from_config(live):
            total += len(_predicted_expansions(display_name, metric))
    assert total > 0, "no composed eval job expands a multi-bin metric"


def test_the_collision_check_catches_a_planted_collision():
    """Guards the guard: proves the prediction logic can actually fire."""
    block = {
        "spike_count_correlation": {
            "metric_class": "spike_count_correlation",
            "bin_sizes_ms": [1.0, 5.0],
        },
        "spike_count_correlation_5ms": {
            "metric_class": "spike_count_correlation",
            "bin_sizes_ms": [5.0],
        },
    }
    pairs = instantiate_metrics_from_config(block)
    predicted: set[str] = set()
    for display_name, metric in pairs:
        predicted |= _predicted_expansions(display_name, metric)
    assert predicted & set(block), "planted collision not detected"
