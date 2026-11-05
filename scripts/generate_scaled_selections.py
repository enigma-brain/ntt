#!/usr/bin/env -S uv run python
"""Generate the data-scaling train subsets ``<brainset>_train_{medium,small}.yaml``.

Full train is derived from the configs, so the subsets cannot drift from it:
  full train = sessions on disk (``<data_dir>/<brainset>/*.h5``)
               - ``exclude_sessions`` of ``<brainset>_train.yaml``
               - every explicit session in ``composite/**/ntt_full_{val,test}_*.yaml``

Per brainset (sorted pools, fresh ``random.Random(42)`` per draw):
  medium = max(1, round(0.6 * |full train|)) sampled from full train
  small  = max(1, round(0.3 * |full train|)) sampled from medium
so small ⊂ medium ⊂ full train. Only the train_{medium,small} files are written;
val selections and the ``ntt_{medium,small}*.yaml`` data configs are left alone.
Invariants are checked by ``tests/data/test_scaling_subsets.py``.

The files in config/data/selections/ are the canonical published split. This script
refuses to overwrite the existing subsets unless --force is given, and regenerating with
--force creates new subsets that do not reproduce the shipped ones.

Usage:
    python scripts/generate_scaled_selections.py --force [--data-root DIR]

The data root is ``--data-root``, else ``NTT_DATA_ROOT``, else ``<repo>/data``
(``<data root>/<brainset>/*.h5``).
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
SELECTIONS_DIR = REPO / "config" / "data" / "selections"
DATA_DIR = Path(os.environ.get("NTT_DATA_ROOT", REPO / "data"))

SEED = 42
SCALES = {"medium": 0.6, "small": 0.3}


def scaled_size(fraction: float, n_full: int) -> int:
    return max(1, round(fraction * n_full))


def _sessions(path: Path, brainset: str, key: str = "sessions") -> set[str]:
    # str(): numeric session IDs (tolias) load as ints.
    entries = yaml.safe_load(path.read_text())["selection"]
    return {str(s) for e in entries if e["brainset"] == brainset for s in e.get(key) or []}


def has_training_data(brainset: str) -> bool:
    """False for eval-only sources (e.g. c4), whose train selection is ``sessions: []``."""
    (entry,) = yaml.safe_load((SELECTIONS_DIR / f"{brainset}_train.yaml").read_text())["selection"]
    return "exclude_sessions" in entry


def eval_holdouts(brainset: str) -> set[str]:
    """Explicit sessions of ``brainset`` in any ntt_full val/test composite."""
    holdouts: set[str] = set()
    for path in (SELECTIONS_DIR / "composite").glob("**/ntt_full_*.yaml"):
        if path.name.startswith(("ntt_full_val_", "ntt_full_test_")):
            holdouts |= _sessions(path, brainset)
    return holdouts


def full_train_sessions(brainset: str, data_dir: Path = DATA_DIR) -> list[str]:
    on_disk = {p.stem for p in (data_dir / brainset).glob("*.h5")}
    if not on_disk:
        raise ValueError(f"No .h5 files found in {data_dir / brainset}")
    excluded = _sessions(SELECTIONS_DIR / f"{brainset}_train.yaml", brainset, "exclude_sessions")
    return sorted(on_disk - excluded - eval_holdouts(brainset))


def write_selection_yaml(path: Path, brainset: str, sessions: list[str], header: list[str]):
    lines = [f"# {h}" for h in header]
    lines += ["_target_: ntt.core.data.components.config.SelectionConfig", "selection:"]
    lines += [f"  - brainset: {brainset}", "    sessions:"]
    lines += [f"      - {s}" for s in sorted(sessions)]
    path.write_text("\n".join(lines) + "\n")


def scaled_brainsets() -> list[str]:
    """Brainsets with a train subset file on disk, sorted."""
    return [
        path.name.removesuffix("_train_medium.yaml")
        for path in sorted(SELECTIONS_DIR.glob("*_train_medium.yaml"))
    ]


def check_pools(data_dir: Path) -> None:
    """Exit unless every brainset with training data has .h5 sessions under ``data_dir``."""
    missing = [
        b
        for b in scaled_brainsets()
        if has_training_data(b) and not any((data_dir / b).glob("*.h5"))
    ]
    if missing:
        raise SystemExit(
            "No .h5 sessions found for: " + ", ".join(missing) + ". Nothing was written; "
            "point --data-root (or NTT_DATA_ROOT) at a directory holding <brainset>/*.h5."
        )


def refuse_to_overwrite(force: bool) -> None:
    """Exit before writing when the canonical subset files exist and ``force`` is off."""
    existing = sorted(SELECTIONS_DIR.glob("*_train_medium.yaml")) + sorted(
        SELECTIONS_DIR.glob("*_train_small.yaml")
    )
    if existing and not force:
        raise SystemExit(
            f"{len(existing)} scaled selection files already exist under "
            f"{SELECTIONS_DIR}. They hold the canonical published split; "
            "regenerating creates new subsets that do not reproduce it. "
            "Pass --force to overwrite them."
        )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Directory holding <brainset>/*.h5 (default: NTT_DATA_ROOT, else <repo>/data).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the canonical scaled selection files with newly drawn subsets.",
    )
    args = parser.parse_args(argv)
    data_dir = DATA_DIR if args.data_root is None else args.data_root
    refuse_to_overwrite(args.force)
    check_pools(data_dir)
    print(f"{'Brainset':<40} {'full':>5} {'medium':>7} {'small':>6}")
    for brainset in scaled_brainsets():
        if not has_training_data(brainset):
            print(f"{brainset:<40} {'-':>5} {'-':>7} {'-':>6}  (eval-only, left unchanged)")
            continue
        full = full_train_sessions(brainset, data_dir)
        medium = sorted(random.Random(SEED).sample(full, scaled_size(SCALES["medium"], len(full))))
        small = sorted(random.Random(SEED).sample(medium, scaled_size(SCALES["small"], len(full))))
        for scale, sessions, parent in [
            ("medium", medium, f"{brainset}_train.yaml"),
            ("small", small, f"{brainset}_train_medium.yaml"),
        ]:
            source = "full train" if scale == "medium" else "medium"
            header = [
                f"Train ({scale}): {len(sessions)}/{len(full)} full-train sessions "
                f"({100 * len(sessions) / len(full):.0f}%; target {SCALES[scale]:.0%}).",
                f"Seed={SEED} sample from {source}; must stay a subset of {parent}.",
                "Generated by scripts/generate_scaled_selections.py "
                "(full train = on-disk - exclude_sessions - ntt_full val/test composites).",
            ]
            write_selection_yaml(
                SELECTIONS_DIR / f"{brainset}_train_{scale}.yaml", brainset, sessions, header
            )
        print(f"{brainset:<40} {len(full):>5} {len(medium):>7} {len(small):>6}")


if __name__ == "__main__":
    main()
