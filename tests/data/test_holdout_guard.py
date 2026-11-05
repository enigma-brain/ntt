"""Tests for the guard that stops training on held-out evaluation data."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from hydra.utils import instantiate
from omegaconf import ListConfig
from torch_brain.data import Dataset

from ntt.core.components.types import DataType
from ntt.core.data.components import holdout_guard
from ntt.core.data.components.config import SelectionConfig
from ntt.core.data.components.holdout_guard import (
    HOLDOUT_FILES,
    check_no_holdout_in_train,
    find_holdout_leaks,
    holdout_dir,
    load_holdouts,
)

REPO = Path(__file__).resolve().parents[2]
_DATA_ROOT_CANDIDATES = [
    os.environ.get("NTT_TEST_DATA_ROOT", ""),
    os.path.join(os.environ.get("PROJECT_ROOT", "."), "data", "brainsets", "processed"),
]
DATA_ROOT = next((p for p in _DATA_ROOT_CANDIDATES if p and Path(p).exists()), None)

# --- find_holdout_leaks on hand-made recordings -------------------------------------

HOLDOUTS = {
    "ntt_full_val_xsession": ["bsA/s_held"],
    "ntt_full_test_xsubject": ["bsA/rat2_s1"],
    "ntt_full_val_xsource": ["bsC/x1"],
}
HOLDOUT_SUBJECTS = {"bsA/rat2_s1": "rat2"}
CLEAN_TRAIN = {"bsA/s_ok": "rat1", "bsB/s_ok": "mouse1"}


def test_clean_training_set_has_no_leaks():
    assert find_holdout_leaks(CLEAN_TRAIN, HOLDOUTS, HOLDOUT_SUBJECTS) == []


def test_held_out_session_is_reported():
    leaks = find_holdout_leaks({**CLEAN_TRAIN, "bsA/s_held": "rat1"}, HOLDOUTS, HOLDOUT_SUBJECTS)
    assert leaks == ["bsA/s_held: held-out session (ntt_full_val_xsession)"]


def test_other_session_of_held_out_animal_is_reported():
    # bsA/rat2_s2 is not listed anywhere, but its animal rat2 is held out.
    leaks = find_holdout_leaks({**CLEAN_TRAIN, "bsA/rat2_s2": "rat2"}, HOLDOUTS, HOLDOUT_SUBJECTS)
    assert leaks == ["bsA/rat2_s2: animal 'rat2' is held out (ntt_full_test_xsubject)"]


def test_same_animal_id_in_other_brainset_is_not_a_leak():
    assert find_holdout_leaks({**CLEAN_TRAIN, "bsB/m": "rat2"}, HOLDOUTS, HOLDOUT_SUBJECTS) == []


def test_held_out_dataset_is_reported():
    leaks = find_holdout_leaks({**CLEAN_TRAIN, "bsC/x2": ""}, HOLDOUTS, HOLDOUT_SUBJECTS)
    assert leaks == ["dataset bsC: held out (ntt_full_val_xsource), 1 training sessions"]


def test_missing_holdout_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_holdouts(tmp_path)


def _write_holdout_files(directory, entry):
    for name in HOLDOUT_FILES:
        (directory / f"{name}.yaml").write_text(yaml.safe_dump({"selection": [entry]}))


def test_holdout_entry_without_sessions_raises(tmp_path):
    """A brainset-only entry must not silently become the id 'brainset/None'."""
    _write_holdout_files(tmp_path, {"brainset": "bsA"})
    with pytest.raises(ValueError, match="must list sessions explicitly"):
        load_holdouts(tmp_path)


def test_holdout_entry_with_single_session(tmp_path):
    _write_holdout_files(tmp_path, {"brainset": "bsA", "session": "s1"})
    assert load_holdouts(tmp_path)["ntt_full_val_xsession"] == ["bsA/s1"]


def test_holdout_dir_uses_project_root(tmp_path, monkeypatch):
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    assert holdout_dir() == tmp_path / "config" / "data" / "selections" / "composite"


def _check_with_fake_files(tmp_path, monkeypatch, train_ids, subjects):
    """Run check_no_holdout_in_train on HOLDOUTS, reading animals from ``subjects``."""
    monkeypatch.setattr(holdout_guard, "load_holdouts", lambda: HOLDOUTS)
    monkeypatch.setattr(holdout_guard, "read_subject", lambda root, rid: subjects[rid])
    check_no_holdout_in_train(
        SimpleNamespace(root=tmp_path, recording_dict=dict.fromkeys(train_ids))
    )


def test_missing_held_out_file_is_skipped(tmp_path, monkeypatch):
    """A partial data root without the held-out session's file must not crash setup."""
    _check_with_fake_files(tmp_path, monkeypatch, ["bsA/rat2_s2"], {"bsA/rat2_s2": "rat2"})


def test_present_held_out_file_is_checked(tmp_path, monkeypatch):
    (tmp_path / "bsA").mkdir()
    (tmp_path / "bsA" / "rat2_s1.h5").touch()
    with pytest.raises(ValueError, match="animal 'rat2' is held out"):
        _check_with_fake_files(
            tmp_path, monkeypatch, ["bsA/rat2_s2"], {"bsA/rat2_s1": "rat2", "bsA/rat2_s2": "rat2"}
        )


def test_holdout_ids_are_strings(tmp_path):
    """Unquoted integer session ids must come out as strings."""
    for name in HOLDOUT_FILES:
        ids = "1234567890123" if name == "ntt_full_test_xsubject" else "s1"
        sessions = f"      - {ids}\n"
        (tmp_path / f"{name}.yaml").write_text(
            "_target_: ntt.core.data.components.config.SelectionConfig\n"
            f"selection:\n  - brainset: bsA\n    sessions:\n{sessions}"
        )
    holdouts = load_holdouts(tmp_path)
    assert "bsA/1234567890123" in holdouts["ntt_full_test_xsubject"]


# --- every training data config ---------------------------------------------------

TRAINING_DATA_CONFIGS = [
    "ntt_full",
    "ntt_medium",
    "ntt_small",
    "ntt_nuclr",
    "ntt_debug",
    "odoherty_only",
    "perich_miller_only",
    "ntt_reaching_strict_train",
    "ntt_reaching_strict_eval_xsession",
    "ntt_reaching_strict_eval_xsubject",
]


@pytest.mark.skipif(DATA_ROOT is None, reason="brainsets data directory not available")
@pytest.mark.parametrize("config_name", TRAINING_DATA_CONFIGS)
def test_training_data_config_has_no_holdout_leaks(config_name):
    with initialize_config_dir(config_dir=str(REPO / "config" / "data"), version_base=None):
        cfg = compose(config_name=config_name)
    selections = instantiate(cfg.train_selections)
    dataset = Dataset(
        DATA_ROOT,
        config=ListConfig([s.model_dump() for s in selections]),
        keep_files_open=False,
    )
    check_no_holdout_in_train(dataset)


# --- training-time eval callbacks: validation hold-outs ok, test hold-outs never -------


def _callback_selections(node):
    """Yield every list found under a ``selections`` key, at any depth."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "selections" and isinstance(value, list):
                yield from value
            else:
                yield from _callback_selections(value)
    elif isinstance(node, list):
        for item in node:
            yield from _callback_selections(item)


CALLBACK_CONFIG_FILES = sorted(
    str(p.relative_to(REPO))
    for p in [
        *(REPO / "config" / "trainer" / "callbacks").glob("*.yaml"),
        *(REPO / "config" / "experiment").glob("*.yaml"),
    ]
    if "selections:" in p.read_text()
)


def test_callback_config_files_found():
    assert "config/trainer/callbacks/behavioral_dataset_eval.yaml" in CALLBACK_CONFIG_FILES


@pytest.mark.parametrize(
    "path",
    [
        pytest.param(
            p,
            marks=pytest.mark.xfail(
                strict=True,
                reason="on main: callback evaluates on loco_20170301_05 (ntt_full_test_xsubject)",
            ),
        )
        if p == "config/experiment/ldns_odoherty_sweep.yaml"
        else p
        for p in CALLBACK_CONFIG_FILES
    ],
)
def test_callback_selections_use_no_test_holdout(path):
    test_holdouts = {
        rid: name for name, ids in load_holdouts().items() if "_test_" in name for rid in ids
    }
    raw_selections = list(_callback_selections(yaml.safe_load((REPO / path).read_text())))
    assert raw_selections, f"no callback selections parsed from {path}"
    used = []
    for raw in raw_selections:
        raw = {k: v for k, v in raw.items() if k != "_target_"}
        for sel in SelectionConfig(**raw).selection:
            sessions = sel.sessions or [sel.session]
            assert None not in sessions, f"{path}: callback selection without explicit sessions"
            used += [f"{sel.brainset}/{s}" for s in sessions]
    leaks = [f"{rid} ({test_holdouts[rid]})" for rid in used if rid in test_holdouts]
    assert not leaks, f"{path}: training-time eval uses TEST held-out sessions: {leaks}"


# --- the guard runs in NTTDataModule.setup() ------------------------------------------


def _datamodule(allow: bool):
    from ntt.core.data.datamodule import NTTDataModule, NTTDataModuleConfig

    held_out = SelectionConfig(
        selection=[
            {
                "brainset": "perich_miller_population_2018",
                "sessions": ["t_20130820_random_target_reaching"],
            }
        ]
    )
    config = NTTDataModuleConfig(
        root=DATA_ROOT,
        train_selections=[held_out],
        window_length_seconds=1.0,
        bin_size_ms=False,
        n_neurons=False,
        input_type=DataType.SPIKE_TIMES,
        allow_holdout_in_train=allow,
    )
    return NTTDataModule(config)


@pytest.mark.skipif(DATA_ROOT is None, reason="brainsets data directory not available")
def test_setup_fit_raises_on_held_out_train_session():
    with pytest.raises(ValueError, match="t_20130820_random_target_reaching: held-out session"):
        _datamodule(allow=False).setup("fit")


@pytest.mark.skipif(DATA_ROOT is None, reason="brainsets data directory not available")
def test_allow_holdout_in_train_skips_the_check():
    dm = _datamodule(allow=True)
    dm.setup("fit")
    assert dm.train_dataset is not None
