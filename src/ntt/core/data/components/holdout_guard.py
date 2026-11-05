"""Stop training runs from using the held-out evaluation data.

The held-out sessions are defined in
``config/data/selections/composite/ntt_full_{val,test}_*.yaml``:

- ``xsession``: these sessions must not be trained on.
- ``xsubject``: these sessions, and every other session of the same animal, must not be trained on.
- ``xsource``: no session of these brainsets may be trained on.

(The within-session time hold-out, ``xtime``, is NOT checked here; see the data leakage check in
``NTTDataset.get``.)
"""

import os
from pathlib import Path

import h5py
import rootutils
import yaml
from temporaldata import Data
from torch_brain.data import Dataset

from ntt.core.data.components.config import SelectionConfig
from ntt.core.data.components.dynamic import dyn
from ntt.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def holdout_dir() -> Path:
    """The repo's ``config/data/selections/composite``.

    The repo root is ``$PROJECT_ROOT`` if set (as ``config/paths`` already requires), else
    the ``.project-root`` directory above this file.
    """
    if os.environ.get("PROJECT_ROOT"):
        root = Path(os.environ["PROJECT_ROOT"])
    else:
        try:
            root = rootutils.find_root(__file__, indicator=".project-root")
        except FileNotFoundError as e:
            raise FileNotFoundError(
                "Cannot find the held-out split files: set PROJECT_ROOT to the ntt repo root."
            ) from e
    return root / "config" / "data" / "selections" / "composite"


HOLDOUT_FILES = tuple(
    f"ntt_full_{split}_{tier}"
    for split in ("val", "test")
    for tier in ("xsession", "xsubject", "xsource")
)


def load_holdouts(directory: Path | None = None) -> dict[str, list[str]]:
    """Return ``{held-out file name: ["brainset/session", ...]}`` for all held-out files."""
    directory = directory or holdout_dir()
    holdouts = {}
    for name in HOLDOUT_FILES:
        raw = yaml.safe_load((directory / f"{name}.yaml").read_text())
        raw.pop("_target_", None)
        # SelectionConfig turns unquoted integer session ids into strings.
        selection = SelectionConfig(**raw).selection
        recording_ids = []
        for sel in selection:
            sessions = sel.sessions or ([sel.session] if sel.session else [])
            if sel.subject or sel.subjects or sel.exclude_sessions or not sessions:
                raise ValueError(
                    f"{name}.yaml: held-out files must list sessions explicitly "
                    f"(found no sessions, or subject/exclude_sessions, for {sel.brainset})."
                )
            recording_ids += [f"{sel.brainset}/{s}" for s in sessions]
        if not recording_ids:
            raise ValueError(f"{name}.yaml lists no sessions")
        holdouts[name] = recording_ids
    return holdouts


def _h5_path(root: str | Path, recording_id: str) -> Path:
    return Path(root) / f"{recording_id}.h5"


def read_subject(root: str | Path, recording_id: str) -> str:
    """Read the animal id of one recording from its data file (lazy, no spikes loaded)."""
    with h5py.File(_h5_path(root, recording_id), "r") as f:
        return str(dyn(Data.from_hdf5(f, lazy=True)).subject.id)


def find_holdout_leaks(
    train_subjects: dict[str, str],
    holdouts: dict[str, list[str]],
    holdout_subjects: dict[str, str],
) -> list[str]:
    """List every way the training recordings overlap the held-out data.

    Args:
        train_subjects: ``{"brainset/session": animal id}`` for every training recording
            (animal id only needed for brainsets that have ``xsubject`` hold-outs).
        holdouts: output of :func:`load_holdouts`.
        holdout_subjects: ``{"brainset/session": animal id}`` for the ``xsubject`` held-out
            sessions of brainsets that are in the training data. Sessions missing here (no
            data file) are skipped.

    Returns:
        One human-readable line per problem; empty if there is no overlap.
    """
    train_brainsets = {rid.split("/")[0] for rid in train_subjects}
    leaks = []
    for name, held_out in holdouts.items():
        # Held-out sessions (all tiers).
        leaks += [f"{rid}: held-out session ({name})" for rid in held_out if rid in train_subjects]

        if "xsubject" in name:
            held_animals = {
                (rid.split("/")[0], holdout_subjects[rid])
                for rid in held_out
                if rid.split("/")[0] in train_brainsets and rid in holdout_subjects
            }
            for rid, subject in train_subjects.items():
                brainset = rid.split("/")[0]
                if (brainset, subject) in held_animals and rid not in held_out:
                    leaks.append(f"{rid}: animal '{subject}' is held out ({name})")

        if "xsource" in name:
            for brainset in sorted({rid.split("/")[0] for rid in held_out} & train_brainsets):
                n = sum(rid.split("/")[0] == brainset for rid in train_subjects)
                leaks.append(f"dataset {brainset}: held out ({name}), {n} training sessions")
    return leaks


def check_no_holdout_in_train(dataset: Dataset) -> None:
    """Raise if the training ``dataset`` uses any held-out session, animal or dataset."""
    holdouts = load_holdouts()
    train_ids = list(dataset.recording_dict)
    xsubject_ids = [rid for name, ids in holdouts.items() if "xsubject" in name for rid in ids]
    subject_brainsets = {rid.split("/")[0] for rid in xsubject_ids} & {
        rid.split("/")[0] for rid in train_ids
    }
    # Animal ids are only needed (and only read) for brainsets with held-out animals.
    train_subjects = {
        rid: read_subject(dataset.root, rid) if rid.split("/")[0] in subject_brainsets else ""
        for rid in train_ids
    }
    held_out_in_train_brainsets = [
        rid for rid in xsubject_ids if rid.split("/")[0] in subject_brainsets
    ]
    # A partial data root may not have every held-out session; its animal can then not be read.
    missing = [
        rid for rid in held_out_in_train_brainsets if not _h5_path(dataset.root, rid).exists()
    ]
    if missing:
        log.warning(
            f"{len(missing)} xsubject held-out sessions are not in {dataset.root}, so their "
            f"animals are not checked: {missing}"
        )
    holdout_subjects = {
        rid: read_subject(dataset.root, rid)
        for rid in held_out_in_train_brainsets
        if rid not in missing
    }
    leaks = find_holdout_leaks(train_subjects, holdouts, holdout_subjects)
    if leaks:
        raise ValueError(
            f"The training data uses held-out evaluation data ({len(leaks)} problems):\n  - "
            + "\n  - ".join(leaks)
            + f"\nThe held-out files are in {holdout_dir()}. Remove these from the training "
            "selection. Only runs that are not tokenizer training (e.g. downstream decoding "
            "on held-out sessions) may set `allow_holdout_in_train: true` in the data config."
        )
