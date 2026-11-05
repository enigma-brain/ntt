"""Shared helpers for resolving a Poisson-baseline reference run.

Used by both ``NTTEvalRobustnessInputJob`` (M1) and
``NTTEvalRobustnessLatentJob`` (M2). Each job parses the Poisson reference's
``results_per_session.csv`` in its own format (long vs wide), but the path
resolution, the ``run_meta.json`` rate-bin lookup and the neuron-selection
check are identical, so they live here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from ntt.evaluation.neuron_selection import (
    NeuronSelection,
    check_neuron_selection,
    describe_neuron_selection,
    warn_padded_mode,
)
from ntt.evaluation.utils.paths import resolve_artifact_path

if TYPE_CHECKING:
    from ntt.core.data.datamodule import NTTDataModule
    from ntt.core.model.tokenizer import TokenizerSpec
    from ntt.evaluation.config import EvaluationConfig


def resolve_poisson_reference_paths(
    reference: str | Path,
    *,
    mode: str,
) -> tuple[Path, Path]:
    """Map a user-supplied Poisson reference to ``(csv_path, run_meta_path)``.

    ``reference`` may be either:
      - a results directory containing ``results_per_session.csv`` and
        (optionally) ``run_meta.json``; or
      - the ``results_per_session.csv`` file itself, in which case
        ``run_meta.json`` is looked up alongside it.

    Raises ``ValueError`` when ``reference`` is neither, or the CSV is missing.
    ``mode`` (e.g. ``"M1"`` / ``"M2"``) is surfaced in error messages.
    """
    if str(reference).startswith("hf://"):
        from ntt.hub import parse_hf_uri

        # Check the path without its @<revision>. Only the named file is downloaded, so
        # a CSV's sibling run_meta.json would be missing.
        _, name, _ = parse_hf_uri(str(reference))
        if name.rstrip("/").endswith(".csv"):
            raise ValueError(
                f"{mode} poisson_reference {reference}: on the Hugging Face Hub, point at the "
                "results folder (holding results_per_session.csv and run_meta.json), not the CSV."
            )
    resolved = Path(resolve_artifact_path(str(reference)))
    if resolved.is_dir():
        csv_path = resolved / "results_per_session.csv"
        run_meta_path = resolved / "run_meta.json"
    elif resolved.name == "results_per_session.csv":
        csv_path = resolved
        run_meta_path = resolved.parent / "run_meta.json"
    else:
        raise ValueError(
            f"{mode} poisson_reference must point to a results directory or "
            f"results_per_session.csv, got {resolved}"
        )
    if not csv_path.is_file():
        raise ValueError(f"{mode} poisson_reference missing results_per_session.csv at {csv_path}")
    return csv_path, run_meta_path


def load_poisson_rate_bin_ms(
    run_meta_path: Path,
    *,
    mode: str,
    log,
) -> float | None:
    """Return ``rate_estimation_bin_ms`` from a Poisson run's ``run_meta.json``.

    Returns ``None`` when the file is absent (the field is optional). On parse or type
    failures, logs a *verbose* warning and returns ``None`` rather than crashing the
    comparison; the bin size is metadata only, but silently dropping it makes downstream
    sanity checks confusing — fail-loud-but-don't-crash.
    """
    if not run_meta_path.is_file():
        return None
    try:
        meta = json.loads(run_meta_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        log.warning(
            "%s poisson_reference: failed to read/parse run_meta.json at %s "
            "(%s: %s); poisson_rate_bin_ms will be None.",
            mode,
            run_meta_path,
            type(exc).__name__,
            exc,
        )
        return None

    raw = meta.get("rate_estimation_bin_ms")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        log.warning(
            "%s poisson_reference: run_meta.json at %s has non-numeric "
            "rate_estimation_bin_ms=%r (%s); poisson_rate_bin_ms will be None.",
            mode,
            run_meta_path,
            raw,
            type(exc).__name__,
        )
        return None


def warn_if_no_session_overlap(
    *,
    model_sessions: set[str],
    poisson_sessions: set[str],
    ceiling_sessions: set[str],
    mode: str,
    log,
) -> set[str]:
    """Compute the three-way session intersection; warn loudly if empty.

    When the intersection is empty, the per-session comparison artifact is
    empty and the aggregate comparison silently compares aggregates over
    disjoint session sets — a meaningless number. Surface that.

    Returns the intersection (so callers don't redo the set algebra).
    """
    shared = set(model_sessions) & set(poisson_sessions) & set(ceiling_sessions)
    if not shared:
        log.warning(
            "%s poisson_comparison: no overlap between model sessions (%d), "
            "poisson_reference sessions (%d), and ceiling sessions (%d). "
            "per_session_comparisons will be empty; the aggregate comparison "
            "will combine aggregates over disjoint session sets and is not "
            "meaningful. Check that the poisson_reference run was evaluated "
            "on the same sessions as the model.",
            mode,
            len(model_sessions),
            len(poisson_sessions),
            len(ceiling_sessions),
        )
    return shared


def check_reference_neuron_selection(
    run_meta_path: Path,
    neuron_selection: NeuronSelection | None,
    *,
    mode: str,
    strict: bool,
) -> None:
    """Check that the Poisson reference run scored the same units as this run.

    Reads ``neuron_selection`` from the reference's ``run_meta.json``; a missing file or
    record counts as a mismatch. ``strict`` raises, otherwise only warns (padded mode).
    """
    reference = None
    if run_meta_path.is_file():
        reference = json.loads(run_meta_path.read_text()).get("neuron_selection")
    check_neuron_selection(
        reference, neuron_selection, what=f"{mode} poisson_reference", strict=strict
    )


def record_neuron_selection(
    eval_datamodule: NTTDataModule,
    *,
    split: Literal["val", "test"],
    eval_config: EvaluationConfig,
    spec: TokenizerSpec,
    evaluate_all_neurons: bool,
    poisson_reference: str | Path | None,
    mode: str,
) -> NeuronSelection | None:
    """Describe the units ``eval_datamodule`` scores, warn in padded mode, and check them
    against the Poisson reference run (if any) before the slow evaluation starts."""
    # The robustness passes score the loaded units as they are (no metric-time cap).
    selection = describe_neuron_selection(
        eval_datamodule, split, eval_config, metric_time_cap=False
    )
    if not evaluate_all_neurons:
        warn_padded_mode(
            selection,
            spec_n_neurons=spec.n_neurons,
            max_neurons=eval_config.max_neurons,
            what=mode,
        )
    if poisson_reference is not None:
        _, run_meta_path = resolve_poisson_reference_paths(poisson_reference, mode=mode)
        check_reference_neuron_selection(
            run_meta_path, selection, mode=mode, strict=evaluate_all_neurons
        )
    return selection
