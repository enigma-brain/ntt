"""Which units of each session an evaluation used, and checks that two runs used the same.

A model eval is normalized against a Poisson reference (in-run, a ``job=poisson_baseline``
artifact, or a robustness run of the Poisson tokenizer). The normalization is only valid
when both sides scored the same units, without zero-padding on one side only. Each run
records a :class:`NeuronSelection` (per session: unit counts and a hash of the selected
unit ids), and consumers compare records with :func:`check_neuron_selection`.

The selection itself is :func:`ntt.core.data.components.transforms.select_eval_units`;
:func:`all_neuron_data_config` builds the data config every full-neuron path loads with.
"""

from __future__ import annotations

import copy
import hashlib
import logging
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pydantic as pdt

from ntt.core.data.components.transforms import EvalUnitSelectionConfig, select_eval_units

if TYPE_CHECKING:
    from ntt.core.data.datamodule import NTTDataModule, NTTDataModuleConfig
    from ntt.core.model.tokenizer import TokenizerSpec
    from ntt.evaluation.config import EvaluationConfig

log = logging.getLogger(__name__)


class NeuronSelectionMismatchError(ValueError):
    """Two runs that are compared did not score the same units."""


class SessionUnits(pdt.BaseModel, frozen=True):
    """The units one session contributed to the metrics."""

    n_units: int  # units in the recording
    n_selected: int  # real units scored
    n_padded: int  # zero-filled units scored alongside them (padded mode only)
    unit_ids_sha256: str  # hash of the scored real units' ids, in order


class NeuronSelection(pdt.BaseModel, frozen=True):
    """How an evaluation chose its units, and what it got per session."""

    # "eval_unit_selection": select_eval_units at load time (full-neuron evaluation).
    # "unit_dropout_first_n": padded mode — the first n_neurons units, zero-padded.
    policy: Literal["eval_unit_selection", "unit_dropout_first_n"]
    max_neurons: int | None
    seed: int
    padded_to: int | None
    sessions: dict[str, SessionUnits]


def all_neuron_data_config(
    data_config: NTTDataModuleConfig,
    eval_config: EvaluationConfig,
    tokenizer_spec: TokenizerSpec | None = None,
) -> NTTDataModuleConfig:
    """Data config for full-neuron evaluation: every unit, capped by ``select_eval_units``.

    ``tokenizer_spec``, if given, sets the tokenizer fields (window, bin size, input
    type, ...) first. Build the datamodule as ``NTTDataModule(config)`` without a spec
    argument: passing a spec re-applies its ``n_neurons`` and would undo this.
    """
    config = copy.deepcopy(data_config)
    if tokenizer_spec is not None:
        config.update_from_tokenizer_spec(tokenizer_spec)
    config.n_neurons = False
    config.pad_neurons_to_fixed = False
    config.eval_unit_selection = EvalUnitSelectionConfig(
        max_neurons=eval_config.max_neurons, seed=eval_config.neuron_subsample_seed
    )
    return config


def _unit_ids_sha256(unit_ids: Any) -> str:
    text = "\n".join(u.decode() if isinstance(u, bytes) else str(u) for u in unit_ids)
    return hashlib.sha256(text.encode()).hexdigest()


def describe_neuron_selection(
    datamodule: NTTDataModule,
    split: Literal["val", "test"],
    eval_config: EvaluationConfig,
    *,
    metric_time_cap: bool = True,
) -> NeuronSelection | None:
    """The units ``datamodule``'s ``split`` feeds to the metrics, per session.

    Mirrors the transforms (``EvalUnitSelection``, or deterministic ``UnitDropout`` +
    ``PadNeuronsToFixed``) and, with ``metric_time_cap``, the metric-time
    ``eval_config.max_neurons`` cap in ``run_validation_evaluation_per_session``, which
    acts on the loaded rows including padding (the robustness passes have no such cap).
    Returns None (with a warning) when the unit count is drawn per window (fractional or
    distribution ``n_neurons``), which no record can describe.
    """
    config = datamodule.config
    dataset = datamodule.get_dataset(split)
    assert dataset is not None, f"call datamodule.setup() before describing the {split} split"
    metric_cap = eval_config.max_neurons if metric_time_cap else None
    seed = eval_config.neuron_subsample_seed
    unit_selection = config.eval_unit_selection
    n_neurons = datamodule.n_neurons_to_sample
    padded_to = datamodule.pad_neurons_to_fixed
    if unit_selection is None and n_neurons is not None and not isinstance(n_neurons, int):
        log.warning(
            "Not recording the neuron selection: n_neurons=%r draws a different unit count "
            "per window.",
            n_neurons,
        )
        return None
    if unit_selection is not None:
        policy: Literal["eval_unit_selection", "unit_dropout_first_n"] = "eval_unit_selection"
        max_neurons = unit_selection.max_neurons
        seed = unit_selection.seed
    else:
        policy = "unit_dropout_first_n"
        max_neurons = metric_cap

    sessions: dict[str, SessionUnits] = {}
    for session_id in dataset.get_session_ids():
        unit_ids = np.asarray(dataset.get_recording_data(session_id).units.id)  # type: ignore[attr-defined]
        n_units = len(unit_ids)
        if unit_selection is not None:
            loaded = select_eval_units(
                n_units, unit_selection.max_neurons, unit_selection.seed, session_id
            )
            n_pad = 0
        else:
            n_loaded = n_units
            if isinstance(n_neurons, int):  # deterministic UnitDropout keeps the first n
                n_loaded = min(n_units, n_neurons)
                if config.max_neurons is not None:
                    n_loaded = min(n_loaded, config.max_neurons)
            loaded = np.arange(n_loaded)
            n_pad = max(0, (padded_to or 0) - n_loaded)
        # The metric-time cap picks rows of the loaded array, padding included.
        rows = select_eval_units(len(loaded) + n_pad, metric_cap, seed, session_id)
        real_rows = rows[rows < len(loaded)]
        selected = loaded[real_rows]
        sessions[session_id] = SessionUnits(
            n_units=n_units,
            n_selected=len(selected),
            n_padded=len(rows) - len(real_rows),
            unit_ids_sha256=_unit_ids_sha256(unit_ids[selected]),
        )
    return NeuronSelection(
        policy=policy,
        max_neurons=max_neurons,
        seed=seed,
        padded_to=padded_to if unit_selection is None else None,
        sessions=sessions,
    )


def _describe_units(units: SessionUnits) -> str:
    padded = f" + {units.n_padded} zero-padded" if units.n_padded else ""
    return (
        f"{units.n_selected}/{units.n_units} units{padded} "
        f"(ids sha256 {units.unit_ids_sha256[:12]})"
    )


def neuron_selection_mismatches(
    reference: NeuronSelection, consumer: NeuronSelection
) -> list[str]:
    """Sessions both records cover whose scored units differ. Session-set differences
    are left to the callers' own session checks."""
    mismatches = []
    for session_id in sorted(set(reference.sessions) & set(consumer.sessions)):
        ref_units = reference.sessions[session_id]
        con_units = consumer.sessions[session_id]
        if ref_units != con_units:
            mismatches.append(
                f"{session_id}: {_describe_units(con_units)} vs reference "
                f"{_describe_units(ref_units)}"
            )
    return mismatches


def check_neuron_selection(
    reference: NeuronSelection | dict | None,
    consumer: NeuronSelection | dict | None,
    *,
    what: str,
    strict: bool,
) -> None:
    """Raise ``NeuronSelectionMismatchError`` (``strict``) or log a warning when
    ``consumer`` and ``reference`` did not score the same units, or a record is missing."""
    if isinstance(reference, dict):
        reference = NeuronSelection.model_validate(reference)
    if isinstance(consumer, dict):
        consumer = NeuronSelection.model_validate(consumer)
    if reference is None or consumer is None:
        missing = "reference" if reference is None else "this run"
        problems = [
            f"the {missing} has no neuron_selection record (written before it was recorded, "
            "or its unit count is drawn per window), so it cannot be shown to use the same units"
        ]
    else:
        problems = neuron_selection_mismatches(reference, consumer)
    if not problems:
        return
    shown = problems[:10] + ([f"... (+{len(problems) - 10} more)"] if len(problems) > 10 else [])
    message = (
        f"{what}: this run and its reference did not score the same units:\n  "
        + "\n  ".join(shown)
        + "\nNormalized scores against this reference are not comparable. Rerun the "
        "reference with the same selection (evaluate_all_neurons, eval_config.max_neurons, "
        "eval_config.neuron_subsample_seed)."
    )
    if strict:
        raise NeuronSelectionMismatchError(message)
    log.warning(message)


def warn_padded_mode(
    selection: NeuronSelection | None,
    *,
    spec_n_neurons: Any,
    max_neurons: int | None,
    what: str,
) -> None:
    """Loud warning for padded mode (``evaluate_all_neurons=false``)."""
    lines = [
        f"{what}: PADDED MODE (evaluate_all_neurons=false). Each model is scored on its own "
        f"first n_neurons={spec_n_neurons!r} units in file order, so models with different "
        "n_neurons are scored on different units and are not comparable."
    ]
    if selection is not None:
        padded = sorted(s for s, u in selection.sessions.items() if u.n_padded)
        if padded:
            lines.append(
                f"{len(padded)}/{len(selection.sessions)} sessions are zero-padded; the padded "
                f"units enter every per-unit and population metric (e.g. {padded[:3]})."
            )
    if (
        max_neurons is not None
        and isinstance(spec_n_neurons, int)
        and not isinstance(spec_n_neurons, bool)
        and spec_n_neurons <= max_neurons
    ):
        lines.append(
            f"eval_config.max_neurons={max_neurons} has no effect: the model already sees "
            f"at most {spec_n_neurons} units."
        )
    log.warning("\n  ".join(lines))


def fixed_n_neurons(spec: TokenizerSpec) -> int | None:
    """The spec's fixed int ``n_neurons``, or None (``n_neurons=False`` is a bool, not a
    count, though ``isinstance(False, int)`` holds)."""
    n_neurons = spec.n_neurons
    if isinstance(n_neurons, int) and not isinstance(n_neurons, bool):
        return n_neurons
    return None


def require_direct_all_neuron_eval(spec: TokenizerSpec, *, what: str) -> None:
    """The robustness passes call the tokenizer on all selected units at once (no neuron
    covering), which a spec with a fixed neuron axis (``compress_n`` with an int
    ``n_neurons``, e.g. cosmos, ldns) cannot take."""
    if spec.compress_n and fixed_n_neurons(spec) is not None:
        raise ValueError(
            f"{what}: full-neuron evaluation is not supported for {type(spec).__name__} "
            f"(compress_n with a fixed n_neurons={spec.n_neurons}): this pass has no neuron "
            "covering. Set job.evaluate_all_neurons=false to run it in padded mode (each "
            "model scored on its own first n_neurons units; under job=eval_all: "
            "job.input_robustness.evaluate_all_neurons=false / "
            "job.latent_robustness.evaluate_all_neurons=false)."
        )
