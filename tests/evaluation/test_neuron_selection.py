"""Shared eval unit selection: the selection function, its transform, the record and checks.

Every eval path (model eval, Poisson baseline, robustness passes) selects units through
``select_eval_units`` and records the result as a ``NeuronSelection``; these tests pin
that the transform and the record agree and that mismatched runs are caught.
"""

import logging
from types import SimpleNamespace

import numpy as np
import pytest
from temporaldata import ArrayDict, Data, Interval, IrregularTimeSeries

from ntt.core.data.components.config import SelectionConfig
from ntt.core.data.components.transforms import (
    EvalUnitSelection,
    EvalUnitSelectionConfig,
    select_eval_units,
    session_seed,
)
from ntt.core.data.datamodule import NTTDataModuleConfig
from ntt.evaluation.config import EvaluationConfig
from ntt.evaluation.neuron_selection import (
    NeuronSelectionMismatchError,
    _unit_ids_sha256,
    all_neuron_data_config,
    check_neuron_selection,
    describe_neuron_selection,
    neuron_selection_mismatches,
)

SMALL = "toy_brainset/small"  # 5 units
_SELECTIONS = [SelectionConfig(selection=[{"brainset": "toy_brainset", "sessions": ["small"]}])]
LARGE = "toy_brainset/large"  # 50 units


def _session(session_id: str, n_units: int) -> Data:
    """A 2 s toy recording where unit i fires i + 1 spikes."""
    unit_index = np.concatenate([np.full(i + 1, i) for i in range(n_units)])
    timestamps = np.sort(np.random.default_rng(0).uniform(0.0, 2.0, len(unit_index)))
    return Data(
        spikes=IrregularTimeSeries(
            timestamps=timestamps, unit_index=unit_index, domain=Interval(0.0, 2.0)
        ),
        units=ArrayDict(id=np.array([f"{session_id}/u{i}" for i in range(n_units)])),
        session=Data(id=session_id),
        domain=Interval(0.0, 2.0),
    )


class _ToyDataset:
    def __init__(self, sessions: dict[str, int]):
        self._sessions = sessions

    def get_session_ids(self) -> list[str]:
        return list(self._sessions)

    def get_recording_data(self, session_id: str) -> Data:
        return _session(session_id, self._sessions[session_id])


def _toy_datamodule(
    *,
    eval_unit_selection: EvalUnitSelectionConfig | None = None,
    n_neurons: int | None = None,
    padded_to: int | None = None,
    sessions: dict[str, int] | None = None,
) -> SimpleNamespace:
    """The datamodule attributes ``describe_neuron_selection`` reads."""
    dataset = _ToyDataset(sessions or {SMALL: 5, LARGE: 50})
    return SimpleNamespace(
        config=SimpleNamespace(eval_unit_selection=eval_unit_selection, max_neurons=None),
        get_dataset=lambda split: dataset,
        n_neurons_to_sample=n_neurons,
        pad_neurons_to_fixed=padded_to,
    )


# ── select_eval_units ───────────────────────────────────────────────


def test_select_eval_units_keeps_all_units_up_to_the_cap():
    np.testing.assert_array_equal(select_eval_units(5, 20, 42, SMALL), np.arange(5))
    np.testing.assert_array_equal(select_eval_units(20, 20, 42, SMALL), np.arange(20))
    np.testing.assert_array_equal(select_eval_units(50, None, 42, LARGE), np.arange(50))


def test_select_eval_units_is_a_seeded_sorted_random_subset():
    selected = select_eval_units(50, 20, 42, LARGE)
    assert len(selected) == 20 and len(set(selected)) == 20
    assert np.all(np.diff(selected) > 0)
    np.testing.assert_array_equal(selected, select_eval_units(50, 20, 42, LARGE))
    assert not np.array_equal(selected, np.arange(20)), "must not be the first N units"
    assert not np.array_equal(selected, select_eval_units(50, 20, 42, "other/session"))
    assert not np.array_equal(selected, select_eval_units(50, 20, 7, LARGE))


def test_select_eval_units_matches_the_existing_metric_time_cap():
    """Same draw as the metric-time cap and job=poisson_baseline references."""
    rng = np.random.default_rng(session_seed(42, LARGE))
    np.testing.assert_array_equal(
        select_eval_units(50, 20, 42, LARGE), np.sort(rng.choice(50, size=20, replace=False))
    )


# ── EvalUnitSelection transform ─────────────────────────────────────


def test_transform_keeps_the_selected_units_and_relabels_spikes():
    data = _session(LARGE, 50)
    out = EvalUnitSelection(EvalUnitSelectionConfig(max_neurons=20, seed=42))(data)
    selected = select_eval_units(50, 20, 42, LARGE)
    np.testing.assert_array_equal(out.units.id, [f"{LARGE}/u{i}" for i in selected])
    # unit i fires i + 1 spikes; relabelled index j is original unit selected[j]
    counts = np.bincount(out.spikes.unit_index, minlength=20)
    np.testing.assert_array_equal(counts, selected + 1)


def test_transform_leaves_small_sessions_untouched():
    out = EvalUnitSelection(EvalUnitSelectionConfig(max_neurons=20))(_session(SMALL, 5))
    assert len(out.units) == 5 and len(out.spikes.timestamps) == 15


# ── all_neuron_data_config ──────────────────────────────────────────


def test_all_neuron_data_config_is_independent_of_the_model():
    base = NTTDataModuleConfig(
        root=".", train_selections=_SELECTIONS, n_neurons=32, pad_neurons_to_fixed=32
    )
    eval_config = EvaluationConfig(max_neurons=200, neuron_subsample_seed=3)
    config = all_neuron_data_config(base, eval_config)
    assert config.n_neurons is False and config.pad_neurons_to_fixed is False
    assert config.eval_unit_selection == EvalUnitSelectionConfig(max_neurons=200, seed=3)
    assert base.n_neurons == 32, "the input config must not be modified"
    other = all_neuron_data_config(base.model_copy(update={"n_neurons": 128}), eval_config)
    assert other.model_dump(exclude={"tokenizer_spec"}) == config.model_dump(
        exclude={"tokenizer_spec"}
    )


# ── describe_neuron_selection ───────────────────────────────────────


def test_record_matches_what_the_transform_loads():
    selection_config = EvalUnitSelectionConfig(max_neurons=20, seed=42)
    record = describe_neuron_selection(
        _toy_datamodule(eval_unit_selection=selection_config),  # type: ignore[arg-type]
        "test",
        EvaluationConfig(max_neurons=20),
    )
    assert record is not None and record.policy == "eval_unit_selection"
    for session_id, n_units in ((SMALL, 5), (LARGE, 50)):
        loaded = EvalUnitSelection(selection_config)(_session(session_id, n_units))
        units = record.sessions[session_id]
        assert units.n_selected == len(loaded.units) and units.n_padded == 0
        assert units.n_units == n_units
        assert units.unit_ids_sha256 == _unit_ids_sha256(loaded.units.id)


def test_two_models_with_different_n_neurons_get_the_same_record_in_full_neuron_mode():
    """The record must not depend on the model's n_neurons."""
    eval_config = EvaluationConfig(max_neurons=20)
    base = NTTDataModuleConfig(root=".", train_selections=_SELECTIONS)
    records = []
    for n_neurons in (8, 32):
        config = all_neuron_data_config(
            base.model_copy(update={"n_neurons": n_neurons}), eval_config
        )
        records.append(
            describe_neuron_selection(
                _toy_datamodule(eval_unit_selection=config.eval_unit_selection),  # type: ignore[arg-type]
                "test",
                eval_config,
            )
        )
    assert records[0] == records[1]


def test_padded_mode_record_shows_first_n_and_padding():
    record = describe_neuron_selection(
        _toy_datamodule(n_neurons=32, padded_to=32),  # type: ignore[arg-type]
        "test",
        EvaluationConfig(max_neurons=200),
    )
    assert record is not None and record.policy == "unit_dropout_first_n"
    assert record.sessions[SMALL].n_selected == 5 and record.sessions[SMALL].n_padded == 27
    assert record.sessions[LARGE].n_selected == 32 and record.sessions[LARGE].n_padded == 0
    first_32 = describe_neuron_selection(
        _toy_datamodule(eval_unit_selection=EvalUnitSelectionConfig(max_neurons=32)),  # type: ignore[arg-type]
        "test",
        EvaluationConfig(max_neurons=32),
    )
    assert first_32 is not None
    # Padded first-32 differs from the full-neuron random 32 on both sessions.
    assert len(neuron_selection_mismatches(first_32, record)) == 2


def test_metric_time_cap_on_padded_rows_counts_padding():
    """The metric-time cap picks rows of the loaded (padded) array, padding included."""
    record = describe_neuron_selection(
        _toy_datamodule(n_neurons=32, padded_to=32),  # type: ignore[arg-type]
        "test",
        EvaluationConfig(max_neurons=10),
    )
    assert record is not None
    small = record.sessions[SMALL]
    rows = select_eval_units(32, 10, 42, SMALL)
    assert small.n_selected == int((rows < 5).sum())
    assert small.n_padded == int((rows >= 5).sum())


def test_record_without_metric_time_cap_keeps_all_loaded_rows():
    """The robustness passes have no metric-time cap: the loaded rows are scored as is."""
    record = describe_neuron_selection(
        _toy_datamodule(n_neurons=32, padded_to=32),  # type: ignore[arg-type]
        "test",
        EvaluationConfig(max_neurons=10),
        metric_time_cap=False,
    )
    assert record is not None
    assert (record.sessions[SMALL].n_selected, record.sessions[SMALL].n_padded) == (5, 27)
    assert (record.sessions[LARGE].n_selected, record.sessions[LARGE].n_padded) == (32, 0)


# ── check_neuron_selection ──────────────────────────────────────────


def _record(**kwargs):
    return describe_neuron_selection(
        _toy_datamodule(**kwargs),  # type: ignore[arg-type]
        "test",
        EvaluationConfig(max_neurons=20),
    )


def test_check_accepts_identical_records_and_ignores_extra_reference_sessions():
    full = _record(eval_unit_selection=EvalUnitSelectionConfig(max_neurons=20))
    subset = _record(
        eval_unit_selection=EvalUnitSelectionConfig(max_neurons=20), sessions={LARGE: 50}
    )
    check_neuron_selection(full, subset, what="test", strict=True)
    check_neuron_selection(full.model_dump(), subset.model_dump(), what="test", strict=True)


def test_check_rejects_different_units():
    full = _record(eval_unit_selection=EvalUnitSelectionConfig(max_neurons=20))
    padded = _record(n_neurons=32, padded_to=32)
    with pytest.raises(NeuronSelectionMismatchError, match="toy_brainset/small"):
        check_neuron_selection(full, padded, what="test", strict=True)
    other_seed = _record(eval_unit_selection=EvalUnitSelectionConfig(max_neurons=20, seed=1))
    with pytest.raises(NeuronSelectionMismatchError, match="toy_brainset/large"):
        check_neuron_selection(full, other_seed, what="test", strict=True)


def test_check_rejects_a_missing_record_and_warns_when_not_strict(caplog):
    full = _record(eval_unit_selection=EvalUnitSelectionConfig(max_neurons=20))
    with pytest.raises(NeuronSelectionMismatchError, match="no neuron_selection record"):
        check_neuron_selection(None, full, what="test", strict=True)
    with caplog.at_level(logging.WARNING):
        check_neuron_selection(full, _record(n_neurons=32, padded_to=32), what="M1", strict=False)
    assert "did not score the same units" in caplog.text


# ── datamodule wiring ───────────────────────────────────────────────


def _datamodule(**overrides):
    from ntt.core.data.datamodule import NTTDataModule

    config = NTTDataModuleConfig(
        root=".",
        train_selections=_SELECTIONS,
        window_length_seconds=1.0,
        bin_size_ms=1.0,
        input_type="binary_spikes",
        **overrides,
    )
    return NTTDataModule(config)


def test_datamodule_uses_eval_unit_selection_on_eval_splits_only():
    from ntt.core.data.components.transforms import UnitDropout

    dm = _datamodule(
        n_neurons=False,
        pad_neurons_to_fixed=False,
        eval_unit_selection=EvalUnitSelectionConfig(max_neurons=20),
    )
    for split in ("val", "test"):
        kinds = [type(t) for t in dm._build_transforms(split).transforms]
        assert EvalUnitSelection in kinds and UnitDropout not in kinds
    assert EvalUnitSelection not in [type(t) for t in dm._build_transforms("train").transforms]


def test_datamodule_rejects_eval_unit_selection_with_a_neuron_count_or_padding():
    with pytest.raises(ValueError, match="eval_unit_selection requires n_neurons=False"):
        _datamodule(
            n_neurons=32,
            pad_neurons_to_fixed=False,
            eval_unit_selection=EvalUnitSelectionConfig(max_neurons=20),
        )._build_transforms("test")
    with pytest.raises(ValueError, match="eval_unit_selection requires n_neurons=False"):
        _datamodule(
            n_neurons=False,
            pad_neurons_to_fixed=32,
            eval_unit_selection=EvalUnitSelectionConfig(max_neurons=20),
        )._build_transforms("test")


def test_all_neuron_data_config_takes_the_spec_fields_but_not_its_neuron_count():
    from ntt.contrib.tokenizer.poisson_baseline import PoissonBaselineTokenizerSpec

    base = NTTDataModuleConfig(
        root=".",
        train_selections=_SELECTIONS,
        window_length_seconds=5.0,
        bin_size_ms=10.0,
        input_type="rates",
        n_neurons=32,
    )
    spec = PoissonBaselineTokenizerSpec(
        window_length_seconds=1.0, bin_size_ms=1.0, rate_estimation_bin_ms=10.0, n_neurons=7
    )
    config = all_neuron_data_config(base, EvaluationConfig(max_neurons=20), spec)
    assert (config.window_length_seconds, config.bin_size_ms) == (1.0, 1.0)
    assert config.input_type == spec.input_type
    assert config.n_neurons is False and config.pad_neurons_to_fixed is False


# ── robustness passes: no neuron covering ───────────────────────────


def test_robustness_passes_reject_fixed_width_compress_n_specs():
    from ntt.evaluation.neuron_selection import require_direct_all_neuron_eval

    with pytest.raises(ValueError, match="job.evaluate_all_neurons=false"):
        require_direct_all_neuron_eval(
            SimpleNamespace(compress_n=True, n_neurons=32),  # type: ignore[arg-type]
            what="M1",
        )
    # Neuron-count agnostic compress_n (perceiver) and compress_t specs are fine.
    require_direct_all_neuron_eval(
        SimpleNamespace(compress_n=True, n_neurons=False),  # type: ignore[arg-type]
        what="M1",
    )
    require_direct_all_neuron_eval(
        SimpleNamespace(compress_n=False, n_neurons=32),  # type: ignore[arg-type]
        what="M1",
    )
