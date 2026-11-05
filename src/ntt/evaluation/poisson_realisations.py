"""Parallel Poisson-realisation runner.

Realisations 2..N of the Poisson comparison are independent given the
stashed originals: they share read-only inputs and produce independent
metric reports. This module factors out the realisation loop into a
free function used by both the in-eval Poisson comparison
(``ntt.jobs.eval``) and the standalone Poisson-baseline job
(``ntt.jobs.poisson_baseline``).

Parallelism uses ``ProcessPoolExecutor`` with the ``fork`` start method:
on Linux, fork gives the children copy-on-write access to the parent's
mmap-backed numpy originals, so there is no per-worker memory blow-up.

Fork + CUDA contract
--------------------
The parent process is allowed to have CUDA initialized at the time
``run_poisson_realisations`` is called (the real callers run a full
GPU pipeline in realisation 1 first). To stay safe under fork, the
runner takes a ``pipeline_factory`` — a picklable zero-arg callable —
*instead of* a constructed pipeline. Each fork child invokes the
factory to build its own (CPU-resident) pipeline.

In addition the runner defensively walks ``originals`` immediately
before submitting tasks and moves any CUDA-resident tensor it finds
to CPU (logging the path). This catches leaks the callers missed: any
CUDA tensor crossing the fork-pickler is reconstructed in the worker
via ``torch.cuda._lazy_init()``, which fails in a forked child with
``Cannot re-initialize CUDA in forked subprocess`` and breaks the pool.
"""

from __future__ import annotations

import dataclasses
import gc
import logging
import multiprocessing as mp
import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np
import torch

log = logging.getLogger(__name__)


def _strip_cuda_tensors(obj: Any, path: str, found: list[str]) -> Any:
    """Recursively strip CUDA tensors from a nested structure.

    Returns a new structure where any ``torch.Tensor`` with ``.is_cuda``
    has been replaced by its ``.cpu()`` copy. Recurses into ``dict``,
    ``list``, ``tuple``, and ``@dataclass`` instances; opaque to Pydantic
    models, ``NamedTuple``, and arbitrary custom objects — CUDA tensors
    held inside those will not be stripped here, and callers are
    responsible for surfacing them on CPU before submit (e.g. via a
    ``.tensor_free_copy()`` method on the type). Records each replaced
    tensor's path in ``found`` for logging.
    """
    if isinstance(obj, torch.Tensor):
        if obj.is_cuda:
            found.append(path)
            return obj.cpu()
        return obj
    if isinstance(obj, dict):
        return {k: _strip_cuda_tensors(v, f"{path}[{k!r}]", found) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_cuda_tensors(v, f"{path}[{i}]", found) for i, v in enumerate(obj)]
    if isinstance(obj, tuple):
        return tuple(_strip_cuda_tensors(v, f"{path}[{i}]", found) for i, v in enumerate(obj))
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        updates: dict[str, Any] = {}
        for f in dataclasses.fields(obj):
            value = getattr(obj, f.name)
            new_value = _strip_cuda_tensors(value, f"{path}.{f.name}", found)
            if new_value is not value:
                updates[f.name] = new_value
        return dataclasses.replace(obj, **updates) if updates else obj
    return obj


def _make_originals_fork_safe(originals: dict[str, list]) -> dict[str, list]:
    """Walk ``originals`` and move every CUDA tensor to CPU; log the paths.

    The runner calls this just before submitting tasks. If anything is moved,
    a single warning lists the offending paths so callers can fix the source
    rather than relying on this defensive cleanup forever.
    """
    found: list[str] = []
    safe = _strip_cuda_tensors(originals, "originals", found)
    if found:
        sample = found[:8] + (["…"] if len(found) > 8 else [])
        log.warning(
            "run_poisson_realisations: stripped %d CUDA tensor(s) from originals "
            "before fork-pool submit. First paths: %s. Source these and call "
            "tensor_free_copy() / .cpu() at construction time to avoid this overhead.",
            len(found),
            sample,
        )
    return safe


def build_lightweight_pipeline(eval_config) -> Any:
    """Build the spike-only, no-correlogram pipeline used by realisations 2..N
    and by the perfect-ceiling pass.

    Filters DOWNSTREAM_PREDICTION metrics (realisation 0 captures them; their
    cross-realisation variance is not informative here) and correlogram metrics
    (``acg_correlation``, ``ccg_correlation``, ``connectivity_f1`` — expensive
    O(N·S) Phy computations whose variance is dominated by spike-count noise).

    Module-level so it can be passed as ``functools.partial(...)`` to the
    parallel runner and pickled across fork.
    """
    from ntt.evaluation.core.types import MetricCategory
    from ntt.evaluation.pipeline import EvaluationPipeline

    full = EvaluationPipeline(config=eval_config)
    spike_only = [
        (n, m) for n, m in full.metrics if m.category != MetricCategory.DOWNSTREAM_DECODING
    ]
    correlogram_names = {"acg_correlation", "ccg_correlation", "connectivity_f1"}
    lightweight = [(n, m) for n, m in spike_only if m.name not in correlogram_names]
    return EvaluationPipeline(metrics=lightweight, config=eval_config)


def _resolve_n_workers(n_workers: int | None, n_realisations_to_run: int) -> int:
    """Compute effective worker count.

    None → ``min(n_realisations_to_run, SLURM_CPUS_PER_TASK or os.cpu_count())``.
    """
    if n_workers is not None:
        return max(1, int(n_workers))
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    cpu_budget = int(slurm_cpus) if slurm_cpus else (os.cpu_count() or 1)
    return max(1, min(n_realisations_to_run, cpu_budget))


def _run_one_realisation(
    seed: int,
    originals: dict[str, list],
    pipeline_factory: Callable[[], Any],
    bin_size_seconds: float,
    bin_size_ms: float,
    poisson_rate_ms: float,
) -> tuple[int, dict[str, list[tuple[Any, dict[str, Any]]]]]:
    """Run one realisation; return (seed, reports). Used as the worker fn.

    Builds the metric pipeline locally via ``pipeline_factory()`` so workers
    never receive CUDA-tainted state from the parent (see module docstring).
    """
    from ntt.contrib.tokenizer.poisson_baseline import (
        poisson_baseline_reconstruction,
    )

    pipeline = pipeline_factory()

    reports: dict[str, list[tuple[Any, dict[str, Any]]]] = {}
    for session_id, entries in originals.items():
        for segment, original_data, metadata in entries:
            original_np = original_data.binned_spikes
            assert original_np is not None
            poisson_spikes = poisson_baseline_reconstruction(
                np.asarray(original_np),
                bin_size_ms,
                poisson_rate_ms,
                seed=seed,
            )
            poisson_data = type(original_data)(
                binned_spikes=poisson_spikes,
                bin_size_seconds=bin_size_seconds,
                duration=original_data.duration,
                data_type=original_data.data_type,
            )
            report = pipeline.run(original_data, poisson_data)
            reports.setdefault(session_id, []).append(
                (segment, {"report": report, "metadata": dict(metadata)})
            )
    return seed, reports


def run_poisson_realisations(
    *,
    originals: dict[str, list],
    pipeline_factory: Callable[[], Any],
    bin_size_seconds: float,
    bin_size_ms: float,
    poisson_rate_ms: float,
    base_seed: int,
    n_realisations_to_run: int,
    n_workers: int | None = None,
) -> list[dict[str, list[tuple[Any, dict[str, Any]]]]]:
    """Run ``n_realisations_to_run`` Poisson realisations.

    Each realisation ``i`` (1-indexed externally) uses seed
    ``base_seed + i``. Returns a list of session-report dicts in seed
    order regardless of completion order.

    Args:
        originals: ``{session_id: [(segment, eval_data, metadata), ...]}``.
            The eval_data objects' ``binned_spikes`` arrays should be
            mmap-backed for memory efficiency under fork.
        pipeline_factory: picklable zero-arg callable returning an object with
            ``.run(original, poisson)`` → metric report. Each fork child calls
            this to build its own CPU-resident pipeline. The factory MUST NOT
            close over CUDA tensors (use e.g. ``functools.partial`` over a
            module-level builder); see module docstring for the rationale.
        bin_size_seconds, bin_size_ms: tokenizer's spike binning.
        poisson_rate_ms: Poisson rate-estimation bin size.
        base_seed: realisations use seeds ``base_seed+1 ... base_seed+N``.
        n_realisations_to_run: ``N - 1`` if N is total; pass 0 to skip.
        n_workers: parallelism. None → auto. ``<=1`` → serial fallback.

    Returns:
        ``[reports_for_seed_(base+1), reports_for_seed_(base+2), ...]``.
    """
    if n_realisations_to_run <= 0:
        return []

    seeds = [base_seed + i for i in range(1, n_realisations_to_run + 1)]

    n_workers_effective = _resolve_n_workers(n_workers, n_realisations_to_run)

    # Defensive: scrub CUDA tensors out of `originals` before any IPC, so a
    # missed strip upstream doesn't crash the fork-pool with a cryptic
    # BrokenProcessPool / Cannot re-initialize CUDA in forked subprocess.
    originals = _make_originals_fork_safe(originals)

    if n_workers_effective <= 1 or n_realisations_to_run <= 1:
        log.info(
            "run_poisson_realisations: serial path (%d realisation(s)).",
            n_realisations_to_run,
        )
        results = []
        for seed in seeds:
            _, reports = _run_one_realisation(
                seed=seed,
                originals=originals,
                pipeline_factory=pipeline_factory,
                bin_size_seconds=bin_size_seconds,
                bin_size_ms=bin_size_ms,
                poisson_rate_ms=poisson_rate_ms,
            )
            results.append(reports)
            gc.collect()
        return results

    log.info(
        "run_poisson_realisations: parallel path (%d realisation(s), n_workers=%d).",
        n_realisations_to_run,
        n_workers_effective,
    )
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=n_workers_effective, mp_context=ctx) as pool:
        futures = [
            pool.submit(
                _run_one_realisation,
                seed,
                originals,
                pipeline_factory,
                bin_size_seconds,
                bin_size_ms,
                poisson_rate_ms,
            )
            for seed in seeds
        ]
        completed: dict[int, dict] = {}
        for fut in futures:
            seed, reports = fut.result()  # re-raises worker exceptions
            completed[seed] = reports
    return [completed[s] for s in seeds]
