"""Every data config composes, and every selection reference points at a file.

Guards against deleting or renaming a selection YAML that a config still uses
(Hydra only fails on it when that config is composed). Config-only: needs no
data on disk.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from hydra import compose, initialize_config_dir

from ntt.core.data.components.config import SelectionConfig

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "config"
SELECTIONS = CONFIG / "data" / "selections"

DATA_CONFIGS = sorted(p.stem for p in (CONFIG / "data").glob("*.yaml"))
# Active (uncommented) defaults entries such as
# `- selections@val_selections.0: composite/ntt_full_val_xsession`
# or `- override /data/selections@train_selections.0: foo`.
SELECTION_REF = re.compile(r"^\s*-\s+(?:override\s+)?/?(?:data/)?selections@[\w.]+:\s*([\w/]+)")
CONFIG_FILES = sorted(CONFIG.rglob("*.yaml"))


@pytest.mark.parametrize("name", DATA_CONFIGS)
def test_data_config_composes(name: str) -> None:
    with initialize_config_dir(config_dir=str(CONFIG), version_base=None):
        cfg = compose(config_name=None, overrides=[f"+data={name}", "+paths.data_dir=x"])
    assert "data" in cfg


def test_selection_references_exist() -> None:
    missing = [
        f"{path.relative_to(REPO)}:{lineno}: {match.group(1)}"
        for path in CONFIG_FILES
        for lineno, line in enumerate(path.read_text().splitlines(), start=1)
        if (match := SELECTION_REF.match(line))
        and not (SELECTIONS / f"{match.group(1)}.yaml").is_file()
    ]
    assert not missing, "selection references without a file:\n" + "\n".join(missing)


@pytest.mark.parametrize(
    "path", sorted(SELECTIONS.rglob("*.yaml")), ids=lambda p: str(p.relative_to(SELECTIONS))
)
def test_selection_file_is_valid(path: Path) -> None:
    doc = yaml.safe_load(path.read_text())
    doc.pop("_target_", None)
    SelectionConfig(**doc)
