"""Comparability checks for aggregated category scores.

Three checks:

- Poisson recompute (``check_reference_match``): does a tokenizer's recon report match
  the Poisson reference it is normalized against (sessions, aggregation strategy, run
  settings)? Raises on a recorded difference.
- Score parts (``raise_on_mixed_score_parts``): is every optional part of a score
  (input- or latent-corruption robustness, POYO, ridge) in all tokenizers' scores or in
  none? Raises otherwise.
- Cross-tokenizer (``find_mixed_member_rows``): were the tokenizers in one table
  averaged the same way (members, sessions, aggregation strategy, run settings,
  Poisson normalization, generalization path)? Raises, unless ``--common-parts-only``
  is passed (differing chunk sizes always raise, except for velocity, which only warns).
  ``provenance_record`` flattens one ``category_scores.json`` into the fields it compares.

Pure library — no I/O. Imported by ``category_scores``, so it must not import it back.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, cast

from ntt.evaluation.behavioral_names import REGRESSION_PREFIXES, REGRESSION_RESULT_NAMES
from ntt.evaluation.neuron_selection import NeuronSelection, neuron_selection_mismatches

log = logging.getLogger(__name__)

# build_aggregation_tables.py / plot-CLI flag: build every tokenizer's score only from
# the parts all tokenizers have, and accept aspects that could not be checked with a
# warning instead of an error. All tokenizers stay in the table.
COMMON_PARTS_ONLY_FLAG = "--common-parts-only"


def _finite_float(value: object) -> float | None:
    """Coerce a report ``raw_value`` to a finite float, or None if it's absent,
    non-numeric, or non-finite — so a malformed report excludes rather than crashes."""
    if value is None:
        return None
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


# ── Poisson recompute: model report vs Poisson reference ─────────────────────


def _session_ids(section: object) -> list[str] | None:
    """Returns the session IDs from a ``per_session`` dict or list, sorted. Returns ``None`` if
    there are none."""
    if isinstance(section, (dict, list)):
        return sorted(str(session_id) for session_id in section)
    return None


def _reference_session_ids(poisson_baseline: dict) -> tuple[list[str] | None, list[str] | None]:
    """Session IDs for the Poisson reference (floor and ceiling). ``None`` if not yet saved."""
    reference_metadata = poisson_baseline.get("metadata") or {}
    return (
        _session_ids(poisson_baseline.get("per_session"))
        or _session_ids(reference_metadata.get("sessions")),
        _session_ids(poisson_baseline.get("perfect_per_session")),
    )


def poisson_norm_session_ids(
    recon_report: dict, poisson_baseline: dict | None, source: str
) -> list[str] | None:
    """Which recording sessions the Poisson comparison used (for both the model and the reference),
    or ``None`` if that information was not saved.

    - ``recomputed``: the sessions that appear in all of the following saved lists: the model's
      ``per_session``, the reference's ``per_session`` (or ``metadata.sessions``), and
      ``perfect_per_session``.
    - ``precomputed``: the evaluation job only runs the Poisson comparison on the model's
      sessions and saves them as the keys of ``poisson_per_session_per_segment``. ``None`` if
      that field does not exist.
    """
    model_sessions = _session_ids(recon_report.get("per_session"))
    if model_sessions is None:
        return None
    if source == "precomputed":
        poisson_sessions = _session_ids(recon_report.get("poisson_per_session_per_segment"))
        if poisson_sessions is None:
            return None
        return sorted(set(model_sessions) & set(poisson_sessions))
    assert poisson_baseline is not None, "recomputed comparison without a Poisson reference"
    recorded = [set(ids) for ids in _reference_session_ids(poisson_baseline) if ids is not None]
    if not recorded:
        return None
    return sorted(set(model_sessions).intersection(*recorded))


def _reference_mismatch(recon_report: dict, poisson_baseline: dict) -> tuple[list[str], list[str]]:
    """Compare the saved averaging settings between the model report and its Poisson reference
    (floor and ceiling): sessions, strategy, bin size, window length, time cap, split, and the
    units each session was scored on (``neuron_selection``).

    Returns ``(differences, not_checkable)``: fields where the saved values disagree, and
    fields that exist on one side but not the other.

    During evaluation, ``NTTEvalJob`` first filters the reference down to the model's sessions
    and re-averages it the same way. This function does not re-average — it only checks what
    was saved. So any difference it finds means the two sides were averaged differently, and
    comparing them is not valid. A field that is missing on one side counts as not checkable,
    not as a match or a difference.
    """
    reference_metadata = poisson_baseline.get("metadata") or {}
    tokenizer_config = recon_report.get("tokenizer_config") or {}
    model_eval_config = (recon_report.get("aggregated") or {}).get("config") or {}
    differences: list[str] = []
    not_checkable: list[str] = []

    poisson_sessions, perfect_sessions = _reference_session_ids(poisson_baseline)
    sessions_by_role = {
        "model": _session_ids(recon_report.get("per_session")),
        "poisson_reference": poisson_sessions,
        "perfect_reference": perfect_sessions,
    }
    session_list_file = {
        "model": "the recon results.json (per_session)",
        "poisson_reference": "the Poisson reference (per_session / metadata.sessions)",
        "perfect_reference": "the Poisson reference (perfect_per_session)",
    }
    for role, session_ids in sessions_by_role.items():
        if session_ids is None:
            not_checkable.append(
                f"{role} session list, because {session_list_file[role]} does not save it"
            )
    recorded = {role: ids for role, ids in sessions_by_role.items() if ids is not None}
    if len({tuple(ids) for ids in recorded.values()}) > 1:
        shared = set.intersection(*(set(ids) for ids in recorded.values()))
        for role, session_ids in recorded.items():
            not_shared = sorted(set(session_ids) - shared)
            shown = not_shared[:10]
            if len(not_shared) > 10:
                shown.append(f"... (+{len(not_shared) - 10} more)")
            differences.append(
                f"  sessions: {role}: n={len(session_ids)}, not shared by all: {shown}"
            )

    # (field, where the model records it, where the reference records it). A key that
    # is present with value None (e.g. max_eval_seconds=None = "no cap") is recorded.
    reference_strategy_source = (
        poisson_baseline if "aggregation_strategy" in poisson_baseline else reference_metadata
    )
    # The reference's window is the one it was scored on; so is the model's, when saved (else
    # fall back to the tokenizer config's, which only says what it was built for).
    model_window = (
        {"window_length_seconds": recon_report["scored_window_seconds"]}
        if "scored_window_seconds" in recon_report
        else tokenizer_config
    )
    settings = (
        ("aggregation_strategy", recon_report, reference_strategy_source),
        ("bin_size_ms", tokenizer_config, reference_metadata),
        ("window_length_seconds", model_window, reference_metadata),
        ("max_eval_seconds", model_eval_config, reference_metadata),
        ("split", recon_report, reference_metadata),
    )
    for name, model_source, reference_source in settings:
        missing_in = [
            side
            for side, source in (("model", model_source), ("reference", reference_source))
            if name not in source
        ]
        if missing_in:
            missing_files = [
                "the recon results.json" if side == "model" else "the Poisson reference"
                for side in missing_in
            ]
            verb = "does" if len(missing_files) == 1 else "do"
            not_checkable.append(
                f"{name}, because {' and '.join(missing_files)} {verb} not save it"
            )
        elif model_source[name] != reference_source[name]:
            differences.append(
                f"  {name}: model={model_source[name]!r}, "
                f"poisson_reference={reference_source[name]!r}"
            )

    # The units each session was scored on (per-session unit-id hashes).
    model_units = recon_report.get("neuron_selection")
    reference_units = reference_metadata.get("neuron_selection")
    if model_units is None or reference_units is None:
        missing_files = [
            name
            for name, units in (
                ("the recon results.json", model_units),
                ("the Poisson reference", reference_units),
            )
            if units is None
        ]
        verb = "does" if len(missing_files) == 1 else "do"
        not_checkable.append(
            f"neuron_selection, because {' and '.join(missing_files)} {verb} not save it"
        )
    else:
        differences.extend(
            f"  neuron_selection: {mismatch}"
            for mismatch in neuron_selection_mismatches(
                NeuronSelection.model_validate(reference_units),
                NeuronSelection.model_validate(model_units),
            )
        )
    return differences, not_checkable


class UncheckedComparabilityError(ValueError):
    """Tokenizers' scores were averaged differently, or it could not be checked whether
    they were (the files do not save some aspect). Raised unless ``--common-parts-only``
    is passed, which turns it into a warning."""


_HOW_TO_CONTINUE = (
    f"Pass {COMMON_PARTS_ONLY_FLAG} to continue with a warning instead "
    f"(NTTAggregationJob: common_parts_only=true)."
)


class ReferenceMismatchError(ValueError):
    """The reconstruction report and its Poisson reference were computed on different sessions or
    settings, so they are not comparable as-is."""


def check_reference_match(
    recon_report: dict, poisson_baseline: dict, *, common_parts_only: bool = False
) -> None:
    """The Poisson-recompute check (see ``_reference_mismatch``).

    Raises ``ReferenceMismatchError`` for recorded differences. For aspects the files
    do not save, raises ``UncheckedComparabilityError`` — or, with
    ``common_parts_only``, logs a "could NOT check" warning instead."""
    differences, not_checkable = _reference_mismatch(recon_report, poisson_baseline)
    if differences:
        raise ReferenceMismatchError(
            "Poisson recompute: model and reference were averaged DIFFERENTLY, so the "
            "normalized scores would not be comparable. Differing aspects:\n"
            + "\n".join(differences)
            + "\nUse a Poisson reference evaluated on the same sessions and settings."
        )
    if not_checkable and not common_parts_only:
        raise UncheckedComparabilityError(
            "Poisson recompute: could NOT check model vs reference for: "
            + "; ".join(not_checkable)
            + ". "
            + _HOW_TO_CONTINUE
        )
    if not_checkable:
        log.warning(
            "WARNING: Poisson recompute: could NOT check model vs reference for: "
            + "; ".join(not_checkable)
        )


# ── Cross-tokenizer comparability check ───────────────────────────────────────

# This session is left out of ridge regression decoding only. Its test data come in ~3 s
# pieces, so tokenizers with long input windows get far fewer chunks (the wanx_B family,
# published as conv-vae-1000to1: 5 training chunks versus 49 for 1 s models), making their
# scores not comparable.
RIDGE_EXCLUDED_SESSIONS: frozenset[str] = frozenset(
    {"perich_miller_population_2018/c_20131029_random_target_reaching"}
)


# Version number for the rules used to compute each table row. Increase it when those rules
# change. It only determines which rows are recomputed: on a rerun, rows whose version differs
# from this one are recomputed from the eval results already downloaded. This does not rerun
# an eval or say whether old eval results are still valid.
COMPARABILITY_VERSION = 9

# Value of an input-dependent field for a tokenizer that has no such input at all (no
# latent-corruption robustness run, say). That difference already shows up in the member
# fields; it is neither a second difference nor something that "could not be checked".
NO_INPUT = "<no input>"

_POISSON_NORMALIZED_COLUMNS = ("single_neuron_recon", "population_structure")
# Scores built from several metrics, whose per-metric list is saved in the diagnostics.
_COLUMNS_WITH_MEMBERS = (
    *_POISSON_NORMALIZED_COLUMNS,
    "latent_structure_and_robustness",
    "decoding",  # ridge family
)
_RECON_COLUMNS = (*_COLUMNS_WITH_MEMBERS, "generalization", "compression")
_LATENT_COLUMNS = ("latent_structure_and_robustness",)

# Run settings that change metric values, compared generically across tokenizers.
# Looked up by name in each input's recorded config (see ``recorded_settings``).
RECORDED_SETTING_KEYS: tuple[str, ...] = (
    "split",
    "bin_size_ms",
    "window_length_seconds",  # the tokenizer's own; not compared, see NOT_COMPARED_SETTINGS
    "scored_window_seconds",
    "max_eval_seconds",
    "max_windows",
    "max_neurons",
    "evaluate_all_neurons",
    "neuron_subsample_seed",
    "conversion_bin_size_ms",
    "split_windows_on_interval_start",
    "max_window_duration_after_stim_end_ms",
    "poisson_rate_bin_ms",
    "poisson_n_realisations",
    "corruptions",
    "n_replicas",
    "epsilons",
    "n_samples_per_epsilon",
    "noise_distribution",
    "n_eval_windows",
    "metrics",
)


# Recorded, but different values across tokenizers do not change how scores were computed: the
# tokenizer's own window (``scored_window_seconds`` is what was scored).
NOT_COMPARED_SETTINGS = frozenset({"window_length_seconds"})


@dataclass(frozen=True)
class ComparabilityAspect:
    """One per-row field that says HOW a score was built."""

    affects: tuple[str, ...]  # Which columns in the results table it changes
    saved_by: str  # The file that is supposed to save this (noted for reference).


_CATEGORY_SCORES_JSON = "category_scores.json"
_RECON_FILE = "the recon results.json"
_M1_FILES = (
    "the input-corruption robustness run (eval_report.json + run_meta.json + "
    "results_per_session.csv)"
)
_M2_FILES = (
    "the latent-corruption robustness run (eval_report.json + run_meta.json + "
    "results_per_session.csv)"
)

# Every comparability field, flat strings so they fit a CSV cell (``provenance_record``
# writes them). ``<input>_settings`` holds a JSON dict that ``find_mixed_member_rows``
# also compares key by key.
COMPARABILITY_ASPECTS: dict[str, ComparabilityAspect] = {
    # (1) metrics in the column
    "m1_provenance": ComparabilityAspect(_LATENT_COLUMNS, _CATEGORY_SCORES_JSON),
    "m2_provenance": ComparabilityAspect(_LATENT_COLUMNS, _CATEGORY_SCORES_JSON),
    "decoding_provenance": ComparabilityAspect(("decoding",), _CATEGORY_SCORES_JSON),
    "single_neuron_recon_members": ComparabilityAspect(
        ("single_neuron_recon",), _CATEGORY_SCORES_JSON
    ),
    "population_structure_members": ComparabilityAspect(
        ("population_structure",), _CATEGORY_SCORES_JSON
    ),
    "latent_structure_and_robustness_members": ComparabilityAspect(
        _LATENT_COLUMNS, _CATEGORY_SCORES_JSON
    ),
    "decoding_members": ComparabilityAspect(("decoding",), _CATEGORY_SCORES_JSON),
    # (2) session set of each input
    "recon_sessions": ComparabilityAspect(_RECON_COLUMNS, _RECON_FILE),
    "m1_sessions": ComparabilityAspect(_LATENT_COLUMNS, _M1_FILES),
    "m2_sessions": ComparabilityAspect(_LATENT_COLUMNS, _M2_FILES),
    "decoding_sessions": ComparabilityAspect(
        ("decoding",),
        "the decoding cohort record (metadata.cohort.found)",
    ),
    "encoding_sessions": ComparabilityAspect(
        ("downstream_encoding",),
        "downstream_encoding_results.json (test.per_session)",
    ),
    # (2b) Poisson normalization: which sessions (model ∩ reference) and which reference
    "poisson_norm_sessions": ComparabilityAspect(
        _POISSON_NORMALIZED_COLUMNS,
        "the recon results.json (poisson_per_session_per_segment)",
    ),
    "poisson_norm_source": ComparabilityAspect(_POISSON_NORMALIZED_COLUMNS, _CATEGORY_SCORES_JSON),
    # Metrics scored against another reference than the Poisson floor (metric=reference).
    "metric_references": ComparabilityAspect(_POISSON_NORMALIZED_COLUMNS, _CATEGORY_SCORES_JSON),
    # (4) averaging across sessions
    "recon_aggregation_strategy": ComparabilityAspect(_RECON_COLUMNS, _RECON_FILE),
    "m1_aggregation_strategy": ComparabilityAspect(_LATENT_COLUMNS, _M1_FILES),
    "m2_aggregation_strategy": ComparabilityAspect(_LATENT_COLUMNS, _M2_FILES),
    # (3) the settings from the saved run, and the settings used for this job
    "recon_settings": ComparabilityAspect(_RECON_COLUMNS, _RECON_FILE),
    "recon_chunk_window_ms": ComparabilityAspect(("decoding",), _RECON_FILE),
    "m1_settings": ComparabilityAspect(_LATENT_COLUMNS, _M1_FILES),
    "m2_settings": ComparabilityAspect(_LATENT_COLUMNS, _M2_FILES),
    "m2_metric": ComparabilityAspect(_LATENT_COLUMNS, _CATEGORY_SCORES_JSON),
    "m2_epsilon": ComparabilityAspect(_LATENT_COLUMNS, _CATEGORY_SCORES_JSON),
    "score_clip": ComparabilityAspect(_POISSON_NORMALIZED_COLUMNS, _CATEGORY_SCORES_JSON),
    "compression_metric": ComparabilityAspect(("compression",), _CATEGORY_SCORES_JSON),
    # (5) generalization: precomputed in the recon eval vs back-filled by this job
    "generalization_source": ComparabilityAspect(("generalization",), _CATEGORY_SCORES_JSON),
}

# Also written per row; not compared itself (it says whether the rest exists).
COMPARABILITY_COLUMNS: tuple[str, ...] = ("comparability_version", *COMPARABILITY_ASPECTS)


def _name_list_fingerprint(names: list[str]) -> str:
    """``"<n> #<hash>"`` of a name list: equal iff the same names (order-free)."""
    digest = hashlib.sha1("\n".join(sorted(names)).encode()).hexdigest()[:8]
    return f"{len(names)} #{digest}"


def report_session_summary(
    report: dict | None, *, sibling_session_ids: list[str] | None = None
) -> dict | None:
    """Session ids, session count and cross-session aggregation strategy of an input
    report, as far as it records them. ``results.json`` carries
    ``per_session``/``n_sessions``/``aggregation_strategy``; a robustness
    ``eval_report.json`` saves ``sessions`` and ``aggregation_strategy`` in its ``config``
    (``add_robustness_report_fields``); when it has none, session ids come from the run's
    ``results_per_session.csv`` (``sibling_session_ids``). ``None`` if there is no report.
    Downstream inputs use ``DownstreamScoreView.session_summary``."""
    if not isinstance(report, dict):
        return None
    raw_config = report.get("config")
    config = raw_config if isinstance(raw_config, dict) else {}
    session_ids = _session_ids(
        report.get("per_session") or report.get("sessions") or config.get("sessions")
    )
    if session_ids is None:
        session_ids = sibling_session_ids
    n_sessions = report.get("n_sessions")
    if n_sessions is None and session_ids is not None:
        n_sessions = len(session_ids)
    return {
        "session_ids": session_ids,
        "n_sessions": n_sessions,
        "aggregation_strategy": report.get(
            "aggregation_strategy", config.get("aggregation_strategy")
        ),
    }


# Settings that may be saved only inside a nested config of a source, as
# {setting: container}: a robustness run can keep ``metrics`` in run_meta.json's
# ``eval_config`` instead of in the report config.
NESTED_SETTINGS = {"metrics": "eval_config"}


def recorded_settings(*config_sources: dict | None) -> dict | None:
    """The ``RECORDED_SETTING_KEYS`` found in the given config dicts (first source that
    has a key wins; a key no source has on top is looked up in its nested container, see
    ``NESTED_SETTINGS``). ``None`` if no source was given at all, so "nothing saved" stays
    distinguishable from "saved, none of these keys"."""
    sources = [source for source in config_sources if isinstance(source, dict)]
    if not sources:
        return None
    settings: dict = {}
    for key in RECORDED_SETTING_KEYS:
        for source in sources:
            if key in source:
                settings[key] = source[key]
                break
        else:
            container = NESTED_SETTINGS.get(key)
            for source in sources if container else ():
                nested = source.get(container)
                if isinstance(nested, dict) and key in nested:
                    settings[key] = nested[key]
                    break
    return settings


def derive_scored_window(*sources: dict | None) -> float | None:
    """The window a report scored, derived from its window settings, or ``None`` if they do
    not tell.

    The all-neuron path (``neuron_selection.policy == "eval_unit_selection"``,
    ``evaluate_all_neurons`` true) builds its datamodule after the eval window is applied, so
    ``eval_config.eval_window_seconds`` (a number) is what it scored. The padded path loads the
    tokenizer's own window whatever that says (M2 rebuilds it at the eval window), so there the
    window is known only if the two are the same number. Each of the sources (a results.json /
    run_meta.json / report config) is searched for the three fields, first hit wins;
    ``eval_window_seconds`` may sit in a nested ``eval_config``.
    """

    def find(key: str) -> object:
        for source in sources:
            if not isinstance(source, dict):
                continue
            if key in source:
                return source[key]
            nested = source.get("eval_config")
            if isinstance(nested, dict) and key in nested:
                return nested[key]
        return None

    selection = find("neuron_selection")
    window = find("eval_window_seconds")
    if not isinstance(window, int | float) or isinstance(window, bool):
        return None
    if (
        isinstance(selection, dict)
        and selection.get("policy") == "eval_unit_selection"
        and find("evaluate_all_neurons") is True
    ):
        return float(window)
    tokenizer_config = find("tokenizer_config")
    own = (
        tokenizer_config.get("window_length_seconds")
        if isinstance(tokenizer_config, dict)
        else None
    )
    if isinstance(own, int | float) and not isinstance(own, bool) and float(own) == float(window):
        return float(window)
    return None


def with_scored_window(settings: dict | None, *sources: dict | None) -> dict | None:
    """``settings`` with ``scored_window_seconds`` filled in from ``sources`` when the report
    does not record it (see ``derive_scored_window``); unchanged when it does or when the
    sources do not tell. ``None`` (no settings at all) passes through."""
    if settings is None or "scored_window_seconds" in settings:
        return settings
    derived = derive_scored_window(*sources)
    if derived is None:
        return settings
    return {**settings, "scored_window_seconds": derived}


def selected_chunk_windows(report: dict) -> dict[str, list[float] | None]:
    """Read each session’s saved regression chunk sizes."""
    segments = report.get("per_session_per_segment")
    if segments is None:
        segments = {sid: [entry] for sid, entry in (report.get("per_session") or {}).items()}
    if not segments:
        segments = {"aggregated": [report.get("aggregated", report)]}
    selected: dict[str, list[float] | None] = {}
    for session, reports in segments.items():
        by_metric: dict[str, list[float | None]] = defaultdict(list)
        for entry in reports:
            for metric, result in entry.get("results", {}).items():
                name = result.get("name")
                is_regression = (
                    name in REGRESSION_RESULT_NAMES
                    if isinstance(name, str)
                    else metric.startswith(REGRESSION_PREFIXES)
                )
                if not is_regression:
                    continue
                metadata = result.get("metadata") or {}
                size = metadata.get("primary_chunk_window_ms", metadata.get("chunk_window_ms"))
                if not metadata.get("parameter_metadata_checked") and (
                    metadata.get("n_segments", 1) > 1 or metadata.get("n_sessions", 1) > 1
                ):
                    # Averaged metadata does not show which chunk sizes were used.
                    size = None
                by_metric[metric].append(_finite_float(size))
        for metric, sizes in by_metric.items():
            selected[f"{session}/{metric}"] = (
                None if any(size is None for size in sizes) else cast(list[float], sizes)
            )
    return selected


def input_geometry(tokenizer_config: dict | None) -> dict | None:
    """Only the eval-input geometry from a tokenizer config. Other tokenizer fields
    (token bin, latent type, ...) are what is being compared, not eval settings."""
    if not isinstance(tokenizer_config, dict):
        return None
    return {
        key: tokenizer_config[key]
        for key in ("bin_size_ms", "window_length_seconds")
        if key in tokenizer_config
    }


def add_robustness_report_fields(
    report: Any,
    run_meta: dict,
    eval_config: Any,
    *,
    split: str,
    session_ids: list[str],
    aggregation_strategy: str,
    session_weights: dict[str, float] | None = None,
) -> None:
    """What a robustness job saves in the ``config`` section of its ``eval_report.json``, so you
    can check it later: the session ids, the averaging strategy (including the weights, if weighted
    averaging was used), and the ``RECORDED_SETTING_KEYS``. These are taken from ``run_meta``
    first. ``eval_config`` is the job's evaluation config (a pydantic
    model) or None. Sets a new dict on ``report.config``: the per-session reports may share
    the old one."""
    eval_config_dict = eval_config.model_dump(mode="json") if eval_config is not None else None
    fields: dict = {
        "sessions": sorted(str(session_id) for session_id in session_ids),
        "aggregation_strategy": aggregation_strategy,
        **(
            recorded_settings(
                {"split": split},
                run_meta,
                eval_config_dict,
                input_geometry(run_meta.get("tokenizer_config")),
            )
            or {}
        ),
    }
    if session_weights is not None:
        fields["session_weights"] = dict(session_weights)
    report.config = {**report.config, **fields}


def _is_position_regression_key(metric_name: str) -> bool:
    return metric_name.startswith("behavioral_regression") and metric_name.endswith("cursor_pos")


def provenance_record(cs: dict) -> dict[str, str | None]:
    """Pull every comparability field out of one ``category_scores.json``.

    ``None`` = not saved; ``NO_INPUT`` = the tokenizer has no such input.
    ``decoding_provenance`` is ``"poyo=<mean> (<n> sessions);ridge=absent"``-style, or
    ``None`` when no decoding family contributed. ``<col>_members`` and session ids are
    fingerprints (count + short hash of the name list); the full lists stay in the JSON.
    """
    diagnostics = cs.get("diagnostics") or {}
    meta = cs.get("metadata") or {}
    prov = meta.get("robustness_normalization") or {}
    dec_prov = (diagnostics.get("decoding") or {}).get("provenance") or {}
    version = meta.get("comparability_version")
    rec: dict[str, str | None] = {
        "comparability_version": None if version is None else str(version),
        "generalization_source": meta.get("generalization_source"),
        "m1_provenance": prov.get("m1"),
        "m2_provenance": prov.get("m2"),
        "decoding_provenance": (
            ";".join(f"{k}={v}" for k, v in sorted(dec_prov.items())) if dec_prov else None
        ),
    }
    for col in _COLUMNS_WITH_MEMBERS:
        per_metric = (diagnostics.get(col) or {}).get("per_metric")
        # Position regression is not part of this list. Whether it counts is decided by
        # the position rule (marked with an asterisk), and whether a tokenizer reported it
        # at all is compared in ``recon_chunk_window_ms``.
        names = [
            name
            for name in per_metric or ()
            if not _is_position_regression_key(name.removeprefix("ridge/"))
        ]
        rec[f"{col}_members"] = _name_list_fingerprint(names) if per_metric else None

    decoding_metrics = (diagnostics.get("decoding") or {}).get("per_metric") or {}
    rec["position_regression_included"] = str(
        any(
            key.startswith("ridge/behavioral_regression") and key.endswith("cursor_pos")
            for key in decoding_metrics
        )
    )
    rec["position_regression_excluded"] = str("position_regression" in dec_prov)
    selected = meta.get("recon_chunk_window_ms")
    rec["recon_chunk_window_ms"] = (
        json.dumps(selected, sort_keys=True)
        if selected
        else NO_INPUT
        if dec_prov.get("ridge") in ("absent", "off")
        else None
    )

    input_paths = meta.get("inputs") or {}
    input_sessions = meta.get("input_sessions") or {}
    input_settings = meta.get("input_settings") or {}
    for input_name in ("recon", "m1", "m2", "decoding", "encoding"):
        has_input = input_paths.get(input_name) is not None
        summary = input_sessions.get(input_name) or {}
        session_ids = summary.get("session_ids")
        if not has_input:
            sessions_value = NO_INPUT
        elif session_ids is not None:
            sessions_value = _name_list_fingerprint(session_ids)
        elif input_name == "encoding" and summary.get("n_sessions") is not None:
            # The encoding report saves only a count for each tokenizer.
            sessions_value = f"n={summary['n_sessions']}"
        else:
            sessions_value = None
        rec[f"{input_name}_sessions"] = sessions_value
        if input_name in ("recon", "m1", "m2"):
            strategy = summary.get("aggregation_strategy")
            rec[f"{input_name}_aggregation_strategy"] = (
                NO_INPUT if not has_input else None if strategy is None else str(strategy)
            )
            settings = input_settings.get(input_name)
            rec[f"{input_name}_settings"] = (
                NO_INPUT
                if not has_input
                else None
                if settings is None
                else json.dumps(settings, sort_keys=True)
            )

    poisson_sessions = meta.get("poisson_norm_sessions")
    if not poisson_sessions:
        rec["poisson_norm_sessions"] = None
        rec["poisson_norm_source"] = None
    else:
        ids = poisson_sessions.get("session_ids")
        rec["poisson_norm_sessions"] = None if ids is None else _name_list_fingerprint(ids)
        source = poisson_sessions.get("source")
        precomputed = "precomputed by the recon eval job"
        if meta.get("poisson_reference_kind") == "external":
            precomputed += " against an external Poisson reference"
        rec["poisson_norm_source"] = (
            f"recomputed against {meta.get('poisson_reference')}"
            if source == "recomputed"
            else precomputed
        )
    references = meta.get("metric_references")
    rec["metric_references"] = (
        NO_INPUT
        if input_paths.get("recon") is None
        else None
        if references is None
        else ";".join(f"{k}={v}" for k, v in sorted(references.items())) or "none"
    )
    for setting in ("m2_metric", "m2_epsilon", "score_clip", "compression_metric"):
        value = meta.get(setting)
        rec[setting] = None if value is None else str(value)
    downstream = meta.get("downstream") or {}
    for input_name in ("encoding", "decoding"):
        record = downstream.get(input_name) or {}
        if input_name == "encoding":
            rec["encoding_split"] = record.get("split")
        else:
            for key in ("complete", "mixed_provenance"):
                value = record.get(key)
                rec[f"decoding_{key}"] = None if value is None else str(value)
    return rec


# Where each downstream score was read from (``provenance_record``). Not compared like
# the comparability fields: mixed decoding provenance only warns. ``decoding_complete`` false
# explains an empty decoding score.
DOWNSTREAM_PROVENANCE_COLUMNS: tuple[str, ...] = (
    "encoding_split",
    "decoding_complete",
    "decoding_mixed_provenance",
)


def _is_unknown(value: object) -> bool:
    """Not saved: ``None`` (or NaN, which is how ``pd.read_csv`` returns an empty cell)."""
    return value is None or (isinstance(value, float) and value != value)


def _decoding_families(decoding_provenance: str) -> list[tuple[str, str]]:
    """``"poyo=0.41 (12 sessions);ridge=absent"`` -> ``[("poyo", "0.41 ..."), ...]``."""
    return [tuple(part.split("=", 1)) for part in decoding_provenance.split(";")]  # type: ignore[misc]


def _decoding_family_status(family_value: str) -> str:
    """``absent`` (no data), ``off`` (dropped), or ``included``."""
    return family_value if family_value in ("absent", "off") else "included"


def _comparable_value(field_name: str, value: str) -> str:
    """Reduce a field to what must match across tokenizers, dropping per-tokenizer
    numbers and the reason a member was left out (shown as detail instead)."""
    if field_name == "decoding_provenance":
        # "poyo=0.41 (12 sessions);ridge=absent" -> "poyo=included;ridge=absent";
        # the session set itself is checked via decoding_sessions. Position regression
        # left out by the position rule (marked with an asterisk) is not a difference.
        return ";".join(
            f"{family_name}={_decoding_family_status(family_value)}"
            for family_name, family_value in _decoding_families(value)
            if family_name != "position_regression"
        )
    if field_name in ("m1_provenance", "m2_provenance"):
        # Both reasons exclude that part from the average, so they have the same effect.
        if value.startswith("excluded"):
            return "excluded"
        if value in ("components_mean", "included"):
            return "included"
        return value  # "off"
    if field_name == "recon_chunk_window_ms":
        # Compare only which "<session>/<metric>" keys are present; the chunk sizes are
        # compared key by key. Leave out sessions that are not part of the ridge score.
        keys = json.loads(value)
        return json.dumps(
            sorted(key for key in keys if key.rsplit("/", 1)[0] not in RIDGE_EXCLUDED_SESSIONS)
        )
    if field_name == "recon_settings.metrics":
        metrics = json.loads(value)
        if not isinstance(metrics, dict):
            return value
        for metric_name, parameters in metrics.items():
            if isinstance(parameters, dict) and (
                metric_name.startswith(REGRESSION_PREFIXES)
                or parameters.get("metric_class") in REGRESSION_RESULT_NAMES
            ):
                parameters.pop("chunk_window_ms", None)
        return json.dumps(metrics, sort_keys=True)
    if field_name.endswith("_settings"):
        return "saved"  # compared key by key, see _expand_settings
    return value


def _expand_settings(row: dict) -> dict[str, object]:
    """``{"recon_settings.max_eval_seconds": "600", ...}`` from a row's settings JSON.
    A key the row's report did not save is absent here (unknown, not "equal")."""
    expanded: dict[str, object] = {}
    for input_name in ("recon", "m1", "m2"):
        raw = row.get(f"{input_name}_settings")
        if _is_unknown(raw) or raw == NO_INPUT:
            continue
        for key, value in json.loads(str(raw)).items():
            expanded[f"{input_name}_settings.{key}"] = json.dumps(value, sort_keys=True)
    return expanded


def _comparability_findings(records: list[dict]) -> tuple[list[str], list[str]]:
    """``(differing_lines, unchecked_lines)``: Returns two things — which fields have different
    values across tokenizers, and which fields or rows could not be compared. Requires at least two
    rows in the results table."""
    if len(records) < 2:
        return [], []
    expanded_rows = [{**row, **_expand_settings(row)} for row in records]
    fields: dict[str, ComparabilityAspect] = dict(COMPARABILITY_ASPECTS)
    # To check whether tokenizer scores in the results table can be compared, compare the saved
    # selected regression chunk sizes for the same session and metric. If the sizes differ,
    # fail the check. If a size is missing, report it as unknown. Do not include position
    # regression if it was left out of a tokenizer’s combined decoding score. Ignore chunk
    # sizes that were configured but not selected.
    for row in expanded_rows:
        raw = row.get("recon_chunk_window_ms")
        if _is_unknown(raw) or raw == NO_INPUT:
            continue
        for key, sizes in json.loads(str(raw)).items():
            field = f"recon_chunk_window_ms.{key}"
            row[field] = None if sizes is None else json.dumps(sizes)
            fields[field] = COMPARABILITY_ASPECTS["recon_chunk_window_ms"]
    # Compare chunk sizes for each session and metric. If a size is missing, report that they
    # cannot be compared. The whole ``recon_chunk_window_ms`` field compares only which
    # sessions and metrics have a chunk size, see ``_comparable_value``.
    setting_fields = sorted(
        field_name
        for field_name in {"recon_settings.metrics"}
        | {k for row in expanded_rows for k in row if "_settings." in k}
        if field_name.split(".", 1)[1] not in NOT_COMPARED_SETTINGS
    )
    for setting_field in setting_fields:
        fields[setting_field] = COMPARABILITY_ASPECTS[setting_field.split(".", 1)[0]]

    differing_lines: list[str] = []
    unchecked_lines: list[str] = []
    for field_name, aspect in fields.items():
        labels_by_value: dict[str, list[str]] = defaultdict(list)
        unknown_tokenizers: list[str] = []
        is_setting_key = "_settings." in field_name
        for row in expanded_rows:
            if all(_finite_float(row.get(column)) is None for column in aspect.affects):
                continue
            comparing_chunk_sizes = field_name.startswith("recon_chunk_window_ms.")
            tokenizer_has_no_regression_scores = row.get(
                "recon_chunk_window_ms"
            ) == NO_INPUT or any(
                part in {"ridge=absent", "ridge=off"}
                for part in str(row.get("decoding_provenance", "")).split(";")
            )
            # Skip tokenizers with no regression scores when comparing chunk sizes.
            if comparing_chunk_sizes and tokenizer_has_no_regression_scores:
                continue
            # A row whose chunk sizes are saved per session but that has no result for this
            # session and metric contributed nothing there: there is no chunk size to compare.
            # (A report saved without a per-session breakdown keys them "aggregated/<metric>";
            # its missing session keys mean "not broken down", so they stay unknown.)
            if (
                comparing_chunk_sizes
                and field_name not in row
                and not _is_unknown(row.get("recon_chunk_window_ms"))
                and not any(
                    key.startswith("aggregated/")
                    for key in json.loads(str(row["recon_chunk_window_ms"]))
                )
            ):
                continue
            if (
                comparing_chunk_sizes
                and field_name.rsplit("/", 1)[-1].startswith("behavioral_regression")
                and field_name.endswith("cursor_pos")
                and str(row.get("position_regression_excluded")) == "True"
            ):
                continue
            if is_setting_key:
                settings_value = row.get(field_name.split(".", 1)[0])
                # If the input does not exist, or if no settings were saved for it (logged once as
                # `<input>_settings`): skip those keys.
                if settings_value == NO_INPUT or _is_unknown(settings_value):
                    continue
            tokenizer = str(row.get("tokenizer"))
            value = row.get(field_name)
            if value == NO_INPUT:
                continue
            if _is_unknown(value):
                unknown_tokenizers.append(tokenizer)
                continue
            comparable = _comparable_value(field_name, str(value))
            # Show the full saved value (for example, why it was left out) if it was shortened.
            label = tokenizer if comparable == str(value) else f"{tokenizer} ({value})"
            labels_by_value[comparable].append(label)
        if len(labels_by_value) > 1:
            detail = " | ".join(
                f"{value}: {sorted(labels)}" for value, labels in sorted(labels_by_value.items())
            )
            differing_lines.append(
                f"  {field_name} (affects {', '.join(aspect.affects)}): {detail}"
            )
        # A settings key nobody saved is not known to exist; skip it. Every other field
        # is expected, so a missing value is always reported.
        if unknown_tokenizers and (labels_by_value or not is_setting_key):
            unchecked_lines.append(
                f"  {field_name} for {sorted(unknown_tokenizers)}, because "
                f"{aspect.saved_by} does not save it"
            )

    return differing_lines, unchecked_lines


def find_mixed_member_rows(records: list[dict]) -> list[str]:
    """Checks that all tokenizers in one table had their scores computed the same way (same metrics
    included, same sessions, same averaging method, same run settings, same Poisson
    normalization, same way generalization scores were obtained).

    Returns up to two warning blocks (empty list if everything matches):

    1. ``WARNING ... averaged DIFFERENTLY`` — fields where the saved values actually differ
       across tokenizers. For example, one tokenizer's latent score averages 4 metrics, but
       another tokenizer's latent score averages only 3 because one run was missing.
    2. ``WARNING: could NOT check`` — fields that some files do not have, plus which rows are
       missing. A missing value is never treated as different or equal; it is
       simply unknown.

    ``records``: flat rows, each with a ``tokenizer``, scores, and ``COMPARABILITY_COLUMNS``. A
    row only counts toward a given field if the row has a score that depends on that field.
    """
    differing_lines, unchecked_lines = _comparability_findings(records)
    lines: list[str] = []
    if differing_lines:
        lines += [
            "WARNING: tokenizers in this table were averaged DIFFERENTLY, so their scores "
            "are not directly comparable. Differing aspects (value: tokenizers):",
            *differing_lines,
        ]
    if unchecked_lines:
        lines += [
            "WARNING: could NOT check whether these tokenizers were averaged the same way:",
            *unchecked_lines,
        ]
    return lines


def raise_on_mixed_chunk_sizes(records: list[dict]) -> None:
    differing_lines, _ = _comparability_findings(records)
    # Velocity regression is not checked: the eval already used 200 ms when it had it and
    # 1000 ms only otherwise, so its score is taken as it is.
    chunk_size_differences = [
        line
        for line in differing_lines
        if line.lstrip().startswith("recon_chunk_window_ms.")
        and "/behavioral_regression_cursor_vel " not in line
    ]
    if chunk_size_differences:
        raise ValueError(
            "Cannot compare tokenizer scores with different selected regression chunk sizes:\n"
            + "\n".join(chunk_size_differences)
        )


def raise_if_not_comparable(records: list[dict], *, common_parts_only: bool) -> None:
    """Raise ``UncheckedComparabilityError`` with a list of what differs across tokenizers and
    what could not be compared.
    Chunk sizes are left to ``raise_on_mixed_chunk_sizes``. When ``common_parts_only`` is set,
    do nothing: ``find_mixed_member_rows`` already printed a warning."""
    differing_lines, unchecked_lines = _comparability_findings(records)
    differing_lines = [
        line for line in differing_lines if not line.lstrip().startswith("recon_chunk_window_ms.")
    ]
    if common_parts_only or not (differing_lines or unchecked_lines):
        return
    lines: list[str] = []
    if differing_lines:
        lines += ["Tokenizers in this table were averaged DIFFERENTLY:", *differing_lines]
    if unchecked_lines:
        lines += [
            "Could NOT check whether these tokenizers were averaged the same way:",
            *unchecked_lines,
        ]
    raise UncheckedComparabilityError("\n".join([*lines, _HOW_TO_CONTINUE]))


# ── Parts of a score that must be the same for every tokenizer ───────────────

# The optional parts of a composite score: (provenance column, score column it is part
# of). A part some tokenizers have and others lack makes that score a mean over a
# different set of parts — not comparable, so the table builder and loader raise.
SCORE_PARTS: dict[str, tuple[str, str]] = {
    "m1": ("m1_provenance", "latent_structure_and_robustness"),
    "m2": ("m2_provenance", "latent_structure_and_robustness"),
    "poyo": ("decoding_provenance", "decoding"),
    "ridge": ("decoding_provenance", "decoding"),
}


# Names of the score parts in messages.
_PART_NAMES = {
    "m1": "input-corruption robustness",
    "m2": "latent-corruption robustness",
    "poyo": "POYO",
    "ridge": "ridge",
}


class MixedScorePartsError(ValueError):
    """Some tokenizers' scores include a component (for example POYO) that the other tokenizers'
    scores do not have."""


def _part_is_included(part: str, row: dict) -> bool | None:
    """Whether ``part`` went into this row's score; ``None`` if not recorded."""
    provenance_column, _ = SCORE_PARTS[part]
    value = row.get(provenance_column)
    if _is_unknown(value):
        return None
    if provenance_column == "decoding_provenance":
        statuses = {
            name: _decoding_family_status(family_value)
            for name, family_value in _decoding_families(str(value))
        }
        return statuses.get(part) == "included"
    return _comparable_value(provenance_column, str(value)) == "included"


def find_mixed_score_parts(records: list[dict]) -> dict[str, dict[str, list[str]]]:
    """``{part: {"included": [tokenizers], "left_out": [tokenizers]}}`` for every part
    that is in some tokenizers' score but not in others'. Only rows whose score
    column (e.g. ``decoding`` for ``poyo``) is present count."""
    mixed: dict[str, dict[str, list[str]]] = {}
    for part, (_, score_column) in SCORE_PARTS.items():
        included: list[str] = []
        left_out: list[str] = []
        for row in records:
            if _finite_float(row.get(score_column)) is None:
                continue
            status = _part_is_included(part, row)
            if status is None:
                continue
            (included if status else left_out).append(str(row.get("tokenizer")))
        if included and left_out:
            mixed[part] = {"included": sorted(included), "left_out": sorted(left_out)}
    return mixed


def raise_on_mixed_score_parts(records: list[dict], *, how_to_fix: str) -> None:
    """Raise ``MixedScorePartsError`` naming each part some tokenizers have and others
    lack."""
    mixed = find_mixed_score_parts(records)
    if not mixed:
        return
    lines = [
        f"  {_PART_NAMES[part]} (`{part}`, part of {SCORE_PARTS[part][1]}): in the score of "
        f"{who['included']}, but not of {who['left_out']}"
        for part, who in mixed.items()
    ]
    raise MixedScorePartsError(
        "Tokenizers' scores are built from different parts, so they are not comparable:\n"
        + "\n".join(lines)
        + f"\n{how_to_fix}"
    )
