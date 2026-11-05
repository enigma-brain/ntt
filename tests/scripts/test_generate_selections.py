"""Tests for the selection generators in scripts/.

The shipped files under config/data/selections/ are the canonical split, so the
generators must never write over them by accident. Every test points the generators at
``tmp_path`` and never touches the real selections directory.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tiered(tmp_path, monkeypatch):
    module = _load("generate_tiered_selections")
    selections = tmp_path / "selections"
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "SELECTIONS_DIR", selections)
    monkeypatch.setattr(module, "COMPOSITE_DIR", selections / "composite")
    return module


@pytest.fixture
def scaled(tmp_path, monkeypatch):
    module = _load("generate_scaled_selections")
    selections = tmp_path / "selections"
    selections.mkdir()
    monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setattr(module, "SELECTIONS_DIR", selections)
    return module


def _synthetic_subject_map(module):
    return {
        source: {f"s{i:02d}": f"subj{i % 4}" for i in range(12)} for source in module.ALL_SOURCES
    }


def test_default_data_root_is_the_documented_layout(monkeypatch):
    monkeypatch.delenv("NTT_DATA_ROOT", raising=False)
    for name in ("generate_tiered_selections", "generate_scaled_selections"):
        module = _load(name)
        root = module.DATA_ROOT if hasattr(module, "DATA_ROOT") else module.DATA_DIR
        assert root == SCRIPTS.parent / "data"


def test_tiered_empty_data_root_exits_before_any_write(tiered, tmp_path, monkeypatch):
    written = []
    monkeypatch.setattr(tiered, "write_selection_file", lambda *args, **kw: written.append(args))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit) as exc:
        tiered.main(["--data-root", str(empty)])
    assert "IBL_brainwide_2023" in str(exc.value)
    assert written == []
    assert not tiered.SELECTIONS_DIR.exists()


def test_tiered_refuses_to_overwrite_without_force(tiered, monkeypatch):
    tiered.SELECTIONS_DIR.mkdir()
    (tiered.SELECTIONS_DIR / "IBL_brainwide_2023_train.yaml").write_text("canonical\n")
    monkeypatch.setattr(
        tiered, "load_subject_map", lambda *a, **k: pytest.fail("read data before refusing")
    )
    with pytest.raises(SystemExit) as exc:
        tiered.main([])
    assert "--force" in str(exc.value)
    assert (tiered.SELECTIONS_DIR / "IBL_brainwide_2023_train.yaml").read_text() == "canonical\n"


def test_tiered_force_writes_a_new_split(tiered, monkeypatch):
    tiered.SELECTIONS_DIR.mkdir()
    train = tiered.SELECTIONS_DIR / "IBL_brainwide_2023_train.yaml"
    train.write_text("canonical\n")
    monkeypatch.setattr(tiered, "load_subject_map", lambda *a, **k: _synthetic_subject_map(tiered))
    tiered.main(["--force"])
    assert train.read_text() != "canonical\n"
    assert (tiered.COMPOSITE_DIR / "ntt_full_val_xsession.yaml").is_file()


def test_scaled_empty_data_root_exits_before_any_write(scaled, tmp_path):
    (scaled.SELECTIONS_DIR / "toy_train.yaml").write_text(
        "selection:\n  - brainset: toy\n    exclude_sessions: []\n"
    )
    medium = scaled.SELECTIONS_DIR / "toy_train_medium.yaml"
    medium.write_text("x\n")
    # The existing subset trips the overwrite guard first; --force reaches the pool check.
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit) as exc:
        scaled.main(["--force", "--data-root", str(empty)])
    assert "toy" in str(exc.value)
    assert medium.read_text() == "x\n"


def test_scaled_refuses_to_overwrite_without_force(scaled):
    medium = scaled.SELECTIONS_DIR / "toy_train_medium.yaml"
    medium.write_text("canonical\n")
    with pytest.raises(SystemExit) as exc:
        scaled.main([])
    assert "--force" in str(exc.value)
    assert medium.read_text() == "canonical\n"
