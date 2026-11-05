"""Resolve a metric's higher/lower-is-better direction from the eval source of truth.

Direction is a property of the *metric*, not of the tokenizer, and it is already
declared exactly once per metric: ``BaseMetric.direction`` on each class in
``METRIC_REGISTRY``.  This module makes that declaration reachable from the
*column names* of the aggregated wide table (and from any other consumer that
only has a string key), so nothing downstream has to keep a hand-maintained
copy that silently rots.

Why a resolver and not a dict: wide-table columns are spelled with *display
names*, which are a config-level concept (``config/evaluation/metrics/*.yaml``
plus inline ``metrics:`` blocks map ``display_name -> metric_class``) and often
differ from the registered class name -- ``isi_cv2_error`` is class
``cv2_error``, ``participation_ratio_error`` is class ``dimensionality_error``,
and so on.  So the mapping has two hops, config then registry, and both hops
are tracked source.

Column shapes handled (four writers, see ``scripts/analysis/build_metrics_table.py``):

* reconstruction: ``{category}/{display_name}``
* multi-scale expansion: ``{display_name}_{n}ms`` (``metrics/spike_detection.py``)
* M1 input robustness: ``robustness/robustness_input[_per_dim]/{corruption_label}``
* M2 latent robustness: ``robustness_latent/{name}/eps_{value}``

``config_root`` note: ``config/`` is *not* shipped in the wheel
(``pyproject.toml`` packages only ``src``), so an installed-package caller gets
an empty display-name index and falls back to bare class names.  Every real
consumer (the analysis scripts, the dashboard, the tests) runs from a checkout,
which is the same assumption ``ntt.evaluation.generalization.build_tier_index``
and ``scripts/analysis/build_metrics_table.py`` already make.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from .core.types import MetricDirection
from .metrics.base import METRIC_REGISTRY

logger = logging.getLogger(__name__)

__all__ = [
    "NON_REGISTRY_BLOCK_DIRECTIONS",
    "build_display_name_index",
    "default_config_root",
    "direction_map_for_columns",
    "resolve_direction",
]


# Blocks whose columns come from downstream models rather than from a
# ``BaseMetric``, so there is no registry entry to consult:
#   encoding/{nc_r2,rate_pearson,rate_pearson_psth,temp_pearson,temp_pearson_psth}
#   decoding/test_r2
# All are R² or Pearson r -- higher-is-better by definition, which is exactly
# what ``aggregation/category_scores.py`` already hardcodes when it scores them
# (``direction="higher"`` for the poyo/ and ridge/ entries).
NON_REGISTRY_BLOCK_DIRECTIONS: dict[str, MetricDirection] = {
    "encoding": MetricDirection.HIGHER_IS_BETTER,
    "decoding": MetricDirection.HIGHER_IS_BETTER,
}

# The M1 report emits two NMSE variants of the same metric class: the token-level
# ``robustness_input`` and the per-latent-dimension ``robustness_input_per_dim``
# (``jobs/eval_robustness_input.py``).  Only the first is a registered name.
_M1_PREFIXES: dict[str, str] = {
    "robustness_input_per_dim/": "robustness_input",
    "robustness_input/": "robustness_input",
}

# Multi-scale suffix appended by ``BaseMetric.expand_results`` and by the
# per-bin display names in ``config/evaluation/temporal/sweep.yaml``.
_BIN_SUFFIX_RE = re.compile(r"_(?:\d+(?:\.\d+)?)ms$")

# M2 keys are ``{name}/eps_{value}``; the value is a bare float ("0", "0.01", "1").
_EPS_SUFFIX_RE = re.compile(r"/eps_[^/]+$")


@cache  # constant; cached because resolve_direction is called per column
def default_config_root() -> Path:
    """Repo ``config/`` directory, derived from this file's location.

    ``src/ntt/evaluation/metric_directions.py`` -> ``<repo>/config``.
    """
    return Path(__file__).resolve().parents[3] / "config"


def _collect_metrics_blocks(node: Any, out: dict[str, dict[str, str]], source: str) -> None:
    """Recursively record every ``metrics: {display_name: {metric_class: ...}}`` block.

    Recursion is needed because ``metrics:`` is not always top level: the
    Lightning callback presets nest it under
    ``<callback>.eval_config.metrics`` (``config/trainer/callbacks/*.yaml``,
    ``config/experiment/ldns_odoherty_sweep.yaml``).
    """
    if isinstance(node, dict):
        block = node.get("metrics")
        if isinstance(block, dict):
            for display_name, cfg in block.items():
                # ``null`` disables an inherited metric (see temporal/sweep.yaml,
                # which nulls the default-bin entries before adding per-bin ones).
                if not isinstance(cfg, dict):
                    continue
                metric_class = cfg.get("metric_class")
                if isinstance(metric_class, str) and isinstance(display_name, str):
                    out.setdefault(display_name, {})[metric_class] = source
        for value in node.values():
            _collect_metrics_blocks(value, out, source)
    elif isinstance(node, list):
        for value in node:
            _collect_metrics_blocks(value, out, source)


@cache  # a full config scan; the dashboard resolves hundreds of columns per run
def build_display_name_index(config_root: Path) -> dict[str, str]:
    """Map ``display_name -> metric_class`` over every tracked YAML under ``config_root``.

    Two spellings are recognised, matching what Hydra actually composes:

    1. ``config/evaluation/metrics/<display_name>.yaml`` with a top-level
       ``metric_class:`` -- the filename *is* the display name, because every
       defaults-list entry that pulls such a file targets ``@metrics.<stem>``.
    2. any nested ``metrics:`` mapping, whose keys are display names.

    Returns an empty index (with a warning) when ``config_root`` is absent, so
    an installed-wheel caller degrades to bare-class-name resolution instead of
    crashing.

    Raises:
        ValueError: if one display name is bound to two different metric
            classes anywhere in the tree -- that would make direction (and the
            eval config itself) ambiguous, so fail loud rather than pick one.
    """
    if not config_root.is_dir():
        logger.warning(
            "config root %s not found; metric directions will fall back to bare "
            "registry class names (display-name aliases will not resolve)",
            config_root,
        )
        return {}

    # display_name -> {metric_class: first config file that said so}
    candidates: dict[str, dict[str, str]] = {}
    for path in sorted(config_root.rglob("*.yaml")):
        text = path.read_text()
        # Only ~40 of the ~325 config files declare metrics at all; the substring
        # pre-filter keeps the scan off the YAML parser for the other ~285.
        if "metric_class" not in text:
            continue
        source = str(path.relative_to(config_root))
        try:
            doc = yaml.safe_load(text)
        except yaml.YAMLError:
            logger.warning("skipping unparseable config %s", path)
            continue
        if not isinstance(doc, dict):
            continue
        if isinstance(doc.get("metric_class"), str):
            candidates.setdefault(path.stem, {}).setdefault(doc["metric_class"], source)
        _collect_metrics_blocks(doc, candidates, source)

    conflicts = {name: bound for name, bound in candidates.items() if len(bound) > 1}
    if conflicts:
        detail = "; ".join(
            f"{name}: " + ", ".join(f"{cls} ({src})" for cls, src in sorted(bound.items()))
            for name, bound in sorted(conflicts.items())
        )
        raise ValueError(f"Display name(s) bound to conflicting metric_class values: {detail}")

    return {name: next(iter(bound)) for name, bound in candidates.items()}


def _direction_of_class(metric_class: str) -> MetricDirection | None:
    cls = METRIC_REGISTRY.get(metric_class)
    return cls.direction if cls is not None else None


def _resolve_metric_name(name: str, index: dict[str, str]) -> MetricDirection | None:
    """Direction for a single metric name: display name, then bin-stripped, then class.

    Display names win over class names at every step because they are what the
    configs (and therefore the report keys) actually declare; the bare-class
    fallback covers metrics whose display name happens to be the class name.
    """
    stripped = _BIN_SUFFIX_RE.sub("", name)
    candidates = [name] if stripped == name else [name, stripped]
    for candidate in candidates:  # display-name hop first for both spellings
        metric_class = index.get(candidate)
        if metric_class is not None:
            direction = _direction_of_class(metric_class)
            if direction is not None:
                return direction
    for candidate in candidates:  # then the registry directly
        direction = _direction_of_class(candidate)
        if direction is not None:
            return direction
    return None


def _match_m1(metric_part: str) -> str | None:
    """Return the metric class behind an M1 ``robustness_input[_per_dim]/<label>`` key.

    Longest prefix first, so ``robustness_input_per_dim/`` is not swallowed by a
    future shorter entry.
    """
    for prefix in sorted(_M1_PREFIXES, key=len, reverse=True):
        if metric_part.startswith(prefix):
            return _M1_PREFIXES[prefix]
    return None


def resolve_direction(
    block: str,
    metric_part: str,
    *,
    config_root: Path | str | None = None,
) -> MetricDirection | None:
    """Direction for one wide-table column, split into ``block`` and ``metric_part``.

    ``block`` is the text before the first ``/`` of the column name (e.g.
    ``temporal_fidelity``, ``robustness``, ``robustness_latent``, ``encoding``);
    ``metric_part`` is everything after it, which may itself contain slashes.

    Returns ``None`` -- never a guess -- when the metric cannot be traced back
    to a registered class. Callers should render that as "no direction known".
    """
    known = NON_REGISTRY_BLOCK_DIRECTIONS.get(block)
    if known is not None:
        return known

    root = default_config_root() if config_root is None else Path(config_root)
    index = build_display_name_index(root)

    m1_class = _match_m1(metric_part)
    if m1_class is not None:
        return _direction_of_class(m1_class)

    if _EPS_SUFFIX_RE.search(metric_part):
        # M2 (latent-noise) keys are ``{name}/eps_{value}`` where ``{name}`` is the
        # metric display name.
        return _resolve_metric_name(_EPS_SUFFIX_RE.sub("", metric_part), index)

    return _resolve_metric_name(metric_part, index)


def direction_map_for_columns(
    columns: Iterable[str],
    *,
    config_root: Path | str | None = None,
) -> dict[str, MetricDirection]:
    """Resolve a batch of ``{block}/{metric_part}`` column names.

    Unresolvable columns are omitted rather than mapped to ``None``, so
    ``column in result`` reads as "direction is known".
    """
    out: dict[str, MetricDirection] = {}
    for column in columns:
        if "/" not in column:
            continue
        block, metric_part = column.split("/", 1)
        direction = resolve_direction(block, metric_part, config_root=config_root)
        if direction is not None:
            out[column] = direction
    return out
