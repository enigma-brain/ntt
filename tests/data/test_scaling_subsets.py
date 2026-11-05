"""Data-scaling train subsets must be nested and disjoint from eval holdouts.

``<brainset>_train_{medium,small}.yaml`` are explicit session lists that must
satisfy ``small ⊂ medium ⊂ full train``, where full train
(``<brainset>_train.yaml``) is "every session of the brainset except its
``exclude_sessions``". Config-only, except the size check, which is skipped
when the processed data is not on disk.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]
SELECTIONS = REPO / "config/data/selections"
COMPOSITES = SELECTIONS / "composite"

BRAINSETS = sorted(
    p.name.removesuffix("_train_medium.yaml") for p in SELECTIONS.glob("*_train_medium.yaml")
)


def _entries(path: Path) -> list[dict]:
    return yaml.safe_load(path.read_text())["selection"]


def _explicit_sessions(path: Path, brainset: str) -> set[str]:
    # str(): numeric session IDs (e.g. 13-digit ids) load as ints.
    entries = [e for e in _entries(path) if e["brainset"] == brainset]
    return {str(s) for e in entries for s in e.get("sessions") or []}


def _full_train_excludes(brainset: str) -> set[str]:
    entries = _entries(SELECTIONS / f"{brainset}_train.yaml")
    assert len(entries) == 1 and entries[0]["brainset"] == brainset
    return {str(s) for s in entries[0].get("exclude_sessions") or []}


def _eval_holdouts(brainset: str) -> dict[str, str]:
    """Explicit val/test sessions of ``brainset`` across ntt_full composites -> source file."""
    holdouts: dict[str, str] = {}
    for path in sorted(COMPOSITES.glob("**/ntt_full_*.yaml")):
        if path.name.startswith(("ntt_full_val_", "ntt_full_test_")):
            for s in _explicit_sessions(path, brainset):
                holdouts[s] = str(path.relative_to(REPO))
    return holdouts


def test_brainsets_discovered():
    assert len(BRAINSETS) >= 8


@pytest.mark.parametrize("brainset", BRAINSETS)
def test_small_subset_of_medium(brainset):
    small = _explicit_sessions(SELECTIONS / f"{brainset}_train_small.yaml", brainset)
    medium = _explicit_sessions(SELECTIONS / f"{brainset}_train_medium.yaml", brainset)
    assert small <= medium, f"small ⊄ medium: {sorted(small - medium)}"


@pytest.mark.parametrize("scale", ["medium", "small"])
@pytest.mark.parametrize("brainset", BRAINSETS)
def test_subset_not_in_full_train_excludes(brainset, scale):
    subset = _explicit_sessions(SELECTIONS / f"{brainset}_train_{scale}.yaml", brainset)
    leaked = subset & _full_train_excludes(brainset)
    assert not leaked, f"{scale} contains sessions excluded from full train: {sorted(leaked)}"


@pytest.mark.parametrize("scale", ["medium", "small"])
@pytest.mark.parametrize("brainset", BRAINSETS)
def test_subset_not_in_eval_composites(brainset, scale):
    subset = _explicit_sessions(SELECTIONS / f"{brainset}_train_{scale}.yaml", brainset)
    holdouts = _eval_holdouts(brainset)
    leaked = {s: holdouts[s] for s in sorted(subset & holdouts.keys())}
    assert not leaked, f"{scale} contains val/test holdout sessions: {leaked}"


FULL_CONFIGS = ["ntt_full"]
SCALED_CONFIGS = {f"ntt_{scale}": "ntt_full" for scale in ("medium", "small")}


def _compose_data(name: str) -> dict:
    with initialize_config_dir(config_dir=str(REPO / "config"), version_base=None):
        cfg = compose(config_name=None, overrides=[f"+data={name}", "+paths.data_dir=x"])
    return OmegaConf.to_container(cfg.data, resolve=False)  # type: ignore[return-value]


def _selection_entries(block: dict) -> list[dict]:
    return [e for k, sel in block.items() if k != "_target_" for e in sel["selection"]]


def _explicit(entry: dict) -> set[str]:
    sessions = entry.get("sessions")
    if sessions is None and entry.get("session") is not None:
        sessions = [entry["session"]]
    assert sessions is not None, f"expected an explicit session list: {entry}"
    return {str(s) for s in sessions}


@pytest.mark.parametrize("name", [*FULL_CONFIGS, *SCALED_CONFIGS])
def test_composed_train_disjoint_from_val_test(name):
    """Session-level train ∩ (val ∪ test) = ∅, without needing data on disk.

    Exclude-based train entries ("all but ...") are disjoint iff every val/test
    session of that brainset is in their ``exclude_sessions``.
    """
    data = _compose_data(name)
    held_out: dict[str, set[str]] = {}
    for split in ("val", "test"):
        for e in _selection_entries(data[f"{split}_selections"]):
            held_out.setdefault(e["brainset"], set()).update(_explicit(e))
    overlap = {}
    for e in _selection_entries(data["train_selections"]):
        bs = e["brainset"]
        if "exclude_sessions" in e:
            hit = held_out.get(bs, set()) - {str(s) for s in e["exclude_sessions"] or []}
        else:
            hit = held_out.get(bs, set()) & _explicit(e)
        if hit:
            overlap[bs] = sorted(hit)
    assert not overlap, f"{name}: train sessions also in val/test: {overlap}"


@pytest.mark.parametrize("name", SCALED_CONFIGS)
def test_scaled_val_test_match_full_config(name):
    scaled, full = _compose_data(name), _compose_data(SCALED_CONFIGS[name])
    for split in ("val_selections", "test_selections"):
        assert scaled[split] == full[split], f"{name} {split} differs from {SCALED_CONFIGS[name]}"


# Size check reuses the generator's full-train definition, which needs data on disk.
_SPEC = importlib.util.spec_from_file_location(
    "generate_scaled_selections", REPO / "scripts/generate_scaled_selections.py"
)
assert _SPEC is not None and _SPEC.loader is not None
generator = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(generator)


@pytest.mark.parametrize("brainset", BRAINSETS)
def test_subset_sizes_match_scaling_fractions(brainset):
    if not any((generator.DATA_DIR / brainset).glob("*.h5")):
        pytest.skip(f"processed {brainset} not on disk")
    if not generator.has_training_data(brainset):
        pytest.skip(f"{brainset} is eval-only")
    n_full = len(generator.full_train_sessions(brainset))
    for scale, fraction in generator.SCALES.items():
        subset = _explicit_sessions(SELECTIONS / f"{brainset}_train_{scale}.yaml", brainset)
        assert len(subset) == generator.scaled_size(fraction, n_full), (scale, n_full)
