"""Per-pair helper used by robustness-style metrics.

Evaluates a list of ``(display_name, BaseMetric)`` pairs against
a single (clean, noisy) ``EvalData`` pair and returns a dict of
status and results for each metric.

You can pass a ``shared_cache`` that already holds the results
computed from the clean data (stored under keys ending in ``_original``),
so those expensive computations are not repeated. Metadata in the
cache that belongs to neither clean nor noisy — latents,
behavioral data, timestamps, sampling rate, and similar (the
"passthrough" keys in ``core/intermediates.py``) — is reused too.

Results are keyed by *display name*, not by ``metric.name``: a config may
instantiate the same metric class several times under different display
names (e.g. ``acg_correlation`` / ``acg_correlation_5ms`` /
``acg_correlation_10ms``), and keying by class name would silently collapse
those variants onto one another.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, overload

from ..core.types import EvalData

if TYPE_CHECKING:
    from ..config import EvaluationConfig
    from ..core.intermediates import IntermediatesCache
    from .base import BaseMetric


@overload
def evaluate_pair(
    clean: EvalData,
    noisy: EvalData,
    metrics: list[tuple[str, BaseMetric]],
    config: EvaluationConfig,
    *,
    shared_cache: IntermediatesCache | None = None,
    return_cache: Literal[False] = False,
) -> dict[str, dict[str, Any]]: ...


@overload
def evaluate_pair(
    clean: EvalData,
    noisy: EvalData,
    metrics: list[tuple[str, BaseMetric]],
    config: EvaluationConfig,
    *,
    shared_cache: IntermediatesCache | None = None,
    return_cache: Literal[True],
) -> tuple[dict[str, dict[str, Any]], IntermediatesCache]: ...


def evaluate_pair(
    clean: EvalData,
    noisy: EvalData,
    metrics: list[tuple[str, BaseMetric]],
    config: EvaluationConfig,
    *,
    shared_cache: IntermediatesCache | None = None,
    return_cache: bool = False,
) -> dict[str, dict[str, Any]] | tuple[dict[str, dict[str, Any]], IntermediatesCache]:
    """Run each ``(display_name, metric)`` pair against ``(clean, noisy)``.

    ``metrics`` mirrors :attr:`EvaluationPipeline.metrics`: the display name
    is the configured key (``config/evaluation/metrics/*.yaml`` filename or
    the key under a ``metrics:`` block), which for most metrics equals
    ``metric.name`` but differs whenever one class is configured several
    times with different parameters.

    If ``shared_cache`` is provided, its ``_original``-side and passthrough
    entries are reused via :meth:`IntermediatesCache.replace_reconstructed`;
    otherwise a fresh ``IntermediatesCache`` is built for this pair.

    Returns a dict mapping each ``display_name`` to either
    ``{"status": "ok", "result": MetricResult}`` on success or
    ``{"status": "skipped", "reason": str}`` on failure. Failures in
    intermediate gathering and in ``metric.compute`` are both caught so
    one broken metric does not abort the full sweep.

    When ``return_cache`` is True, the per-pair :class:`IntermediatesCache`
    is returned alongside ``results``. This lets callers retrieve already-
    computed intermediates (e.g. ``acg_original`` / ``acg_reconstructed``)
    for cross-window aggregation without recomputing them.
    """
    from ..core.intermediates import (
        IntermediatesCache,
    )  # local import to avoid circular at module load

    if shared_cache is not None:
        cache = shared_cache.replace_reconstructed(noisy)
    else:
        cache = IntermediatesCache(original=clean, reconstructed=noisy)

    display_names = [display_name for display_name, _ in metrics]
    duplicates = sorted({n for n in display_names if display_names.count(n) > 1})
    if duplicates:
        # Results are keyed by display name: a repeated name would silently overwrite
        # another metric's result. Refuse before computing anything.
        raise ValueError(f"evaluate_pair: duplicate display names {duplicates}")

    results: dict[str, dict[str, Any]] = {}
    for display_name, metric in metrics:
        # Use preferred_data_type for request resolution; the per-pair
        # helper intentionally does NOT implement the full pipeline's
        # DataType-conversion dance (see pipeline.py) — callers should
        # pass EvalData already in a compatible representation.
        target_type = metric.preferred_data_type
        try:
            requests = metric.get_intermediate_requests(target_type)
            intermediates = {req.key: cache.get(req) for req in requests}
            result = metric.compute(intermediates)
            results[display_name] = {"status": "ok", "result": result}
        except Exception as exc:  # noqa: BLE001 — want loud but non-fatal per metric
            results[display_name] = {"status": "skipped", "reason": str(exc)}
    if return_cache:
        return results, cache
    return results
