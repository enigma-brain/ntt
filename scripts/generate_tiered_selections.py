#!/usr/bin/env python3
"""Generate tiered eval selection YAML files for ntt_full.

Tiers:
  1. xtime    — within-session temporal eval (same sessions as train, uses valid_domain)
  2. xsession — held-out sessions from subjects seen in training
  3. xsubject — sessions from subjects NOT in training
  4. xsource  — sessions from sources entirely excluded from training
                (c4, cowley, wingert)

The files in config/data/selections/ are the canonical published split. This script
refuses to overwrite them unless --force is given, and regenerating with --force creates
a new split that does not reproduce the shipped one.

Usage:
    python scripts/generate_tiered_selections.py --force [--data-root DIR]

Reads h5 files from the data root (``--data-root``, else ``NTT_DATA_ROOT``, else
``<repo>/data``, the layout ``<data root>/<brainset>/*.h5``) to determine subject IDs,
then generates:
  - Per-source selection YAMLs in config/data/selections/
  - Composite selection YAMLs in config/data/selections/composite/
"""

import argparse
import os
import random
from collections import defaultdict
from pathlib import Path

import h5py
from temporaldata import Data

from ntt.evaluation.excluded_brainsets import EXCLUDED_BRAINSET_SESSIONS

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("NTT_DATA_ROOT", REPO_ROOT / "data"))
SELECTIONS_DIR = REPO_ROOT / "config" / "data" / "selections"
COMPOSITE_DIR = SELECTIONS_DIR / "composite"

SEED = 42

# ──────────────────────────────────────────────────────────────────────────────
# Configuration: which subjects go to which tier, target holdout counts
# ──────────────────────────────────────────────────────────────────────────────

# Sources and their cross-subject subjects (to be fully removed from train)
XSUBJECT_SUBJECTS = {
    "perich_miller_population_2018": ["j", "t"],
    "odoherty_sabes_nonhuman_2017": ["loco"],
}

# Cross-session holdout targets (total val+test, per source)
# Only subjects NOT in XSUBJECT_SUBJECTS are eligible.
XSESSION_TARGETS = {
    "IBL_brainwide_2023": 12,
    "allen_visual_behaviour_neuropixels": 10,
    "allen_visual_coding_neuropixels": 0,  # all sessions are xsubject (1 per specimen)
    "churchland_shenoy_neural_2012": 4,
    "flint_slutzky_accurate_2012": 2,
    "odoherty_sabes_nonhuman_2017": 7,
    "perich_miller_population_2018": 10,
}

# Cross-source: entirely held out from training — not in XSESSION_TARGETS.
# Excluded brainsets (EXCLUDED_BRAINSET_SESSIONS) stay in these lists on purpose:
# they still draw from the shared seeded RNG, so the splits of the sources after
# them are unchanged, but they are never written to any selection file.
XSOURCE_SOURCES = [
    "c4_dataset_cerebellum_2025",
    "cowley_v4_compact_2024",
    "tostado_marcos_birdsong_2024",  # excluded, see comment above
    "wingert_auditory_cortex_2026",
]

ALL_SOURCES = [
    "IBL_brainwide_2023",
    "allen_visual_behaviour_neuropixels",
    "allen_visual_coding_neuropixels",
    "c4_dataset_cerebellum_2025",
    "churchland_shenoy_neural_2012",
    "cowley_v4_compact_2024",
    "flint_slutzky_accurate_2012",
    "odoherty_sabes_nonhuman_2017",
    "perich_miller_population_2018",
    "tostado_marcos_birdsong_2024",  # excluded, see comment above XSOURCE_SOURCES
    "wingert_auditory_cortex_2026",
]
WRITTEN_SOURCES = [s for s in ALL_SOURCES if s not in EXCLUDED_BRAINSET_SESSIONS]

# The composites are the canonical eval split. Per-source eval files are written
# only for sources with a consumer of their own (comment); every other per-source
# eval file would duplicate a composite. Value = which holdout the files hold:
#   xsession -> {source}_{val,test}, xsubject -> {source}_xsubject_{val,test},
#   xsource  -> {source}_{val,test}.
PER_SOURCE_EVAL_FILES = {
    "IBL_brainwide_2023": "xsession",  # ntt_nuclr
    "allen_visual_behaviour_neuropixels": "xsession",  # ntt_nuclr
    "allen_visual_coding_neuropixels": "xsubject",  # ntt_nuclr (no xsession holdouts)
    "cowley_v4_compact_2024": "xsource",  # cowley_* eval configs
    "odoherty_sabes_nonhuman_2017": "xsession",  # odoherty_only
    "perich_miller_population_2018": "xsession",  # perich_miller_only
}


# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Load session→subject mapping from h5 files
# ──────────────────────────────────────────────────────────────────────────────


def load_subject_map(data_root: Path | None = None):
    """Returns {source: {session_id: subject_id}}."""
    data_root = DATA_ROOT if data_root is None else data_root
    result = {}
    for source in ALL_SOURCES:
        source_dir = data_root / source
        sessions = {}
        for f in sorted(source_dir.glob("*.h5")):
            with h5py.File(f, "r") as hf:
                d = Data.from_hdf5(hf, lazy=True)
                sessions[f.stem] = str(d.subject.id)
        result[source] = sessions
    return result


def check_pools(subject_map):
    """Exit unless every source that gets written has at least one session."""
    missing = [source for source in WRITTEN_SOURCES if not subject_map.get(source)]
    if missing:
        raise SystemExit(
            "No .h5 sessions found for: " + ", ".join(missing) + ". Nothing was written; "
            "point --data-root (or NTT_DATA_ROOT) at a directory holding <brainset>/*.h5."
        )


def existing_selection_files():
    """The tiered selection files already on disk (train files and ntt_full composites)."""
    return sorted(SELECTIONS_DIR.glob("*_train.yaml")) + sorted(
        COMPOSITE_DIR.glob("ntt_full_*.yaml")
    )


def refuse_to_overwrite(force: bool):
    """Exit before writing when the canonical selection files exist and ``force`` is off."""
    existing = existing_selection_files()
    if existing and not force:
        raise SystemExit(
            f"{len(existing)} selection files already exist under "
            f"{SELECTIONS_DIR}. They hold the canonical published "
            "split; regenerating creates a new split that does not reproduce it. "
            "Pass --force to overwrite them."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Assign sessions to tiers
# ──────────────────────────────────────────────────────────────────────────────


def assign_tiers(subject_map):
    """Returns dicts keyed by source, each containing lists of session IDs:
    xsubject_val, xsubject_test, xsession_val, xsession_test, train, xsource_val, xsource_test
    """
    rng = random.Random(SEED)

    assignments = {}

    for source in ALL_SOURCES:
        sess_subj = subject_map[source]  # {session: subject}
        all_sessions = sorted(sess_subj.keys())

        a = {
            "xsubject_val": [],
            "xsubject_test": [],
            "xsession_val": [],
            "xsession_test": [],
            "train": [],
            "xsource_val": [],
            "xsource_test": [],
        }

        # ── Cross-source ──
        if source in XSOURCE_SOURCES:
            # All sessions go to eval (50/50 val/test), none in train.
            all_sessions_copy = list(all_sessions)
            rng.shuffle(all_sessions_copy)
            mid = len(all_sessions_copy) // 2
            a["xsource_val"] = sorted(all_sessions_copy[:mid])
            a["xsource_test"] = sorted(all_sessions_copy[mid:])
            assignments[source] = a
            continue

        # ── Cross-subject ──
        xsubj_subjects = set(XSUBJECT_SUBJECTS.get(source, []))
        xsubj_sessions = sorted([s for s in all_sessions if sess_subj[s] in xsubj_subjects])

        # For allen_visual_coding_neuropixels: every specimen has 1 session,
        # so ALL sessions are cross-subject candidates. But we only hold out 4
        # (the existing val/test). The rest stay in train.
        # Special handling: pick 4 cross-subject sessions from single-session subjects
        if source == "allen_visual_coding_neuropixels":
            # All 58 subjects have 1 session each. Pick 4 for xsubject eval.
            candidates = list(all_sessions)
            rng.shuffle(candidates)
            xsubj_sessions = sorted(candidates[:4])

        # For IBL: zfm-02368 has only 1 session, already in test. Keep it.
        if source == "IBL_brainwide_2023":
            single_session_subjects = set()
            subj_counts = defaultdict(int)
            for _s, subj in sess_subj.items():
                subj_counts[subj] += 1
            single_session_subjects = {subj for subj, cnt in subj_counts.items() if cnt == 1}
            # Pick 1 single-session subject for xsubject test (keep zfm-02368 flavor)
            single_sess_sessions = sorted(
                [s for s in all_sessions if sess_subj[s] in single_session_subjects]
            )
            rng.shuffle(single_sess_sessions)
            xsubj_sessions = sorted(single_sess_sessions[:2])

        # Split xsubject into val/test (roughly 50/50)
        rng.shuffle(xsubj_sessions)
        mid = len(xsubj_sessions) // 2
        a["xsubject_val"] = sorted(xsubj_sessions[:mid])
        a["xsubject_test"] = sorted(xsubj_sessions[mid:])

        # ── Cross-session ──
        # Pool: sessions from subjects NOT in xsubject, with >1 session in pool
        xsubj_session_set = set(xsubj_sessions)
        pool_sessions = [s for s in all_sessions if s not in xsubj_session_set]
        # Group by subject
        subj_to_pool = defaultdict(list)
        for s in pool_sessions:
            subj_to_pool[sess_subj[s]].append(s)

        # Only subjects with >1 session can provide cross-session holdout
        eligible_subjects = {
            subj: sessions for subj, sessions in subj_to_pool.items() if len(sessions) > 1
        }

        target = XSESSION_TARGETS.get(source, 0)
        if target > 0 and eligible_subjects:
            # Collect one candidate per eligible subject, then fill from remaining
            # Strategy: shuffle all eligible sessions, pick target count
            eligible_sessions = []
            for subj in sorted(eligible_subjects):
                eligible_sessions.extend(eligible_subjects[subj])
            rng.shuffle(eligible_sessions)
            xsess_holdout = sorted(eligible_sessions[:target])
        else:
            xsess_holdout = []

        # Split xsession into val/test (roughly 50/50)
        rng.shuffle(xsess_holdout)
        mid = len(xsess_holdout) // 2
        a["xsession_val"] = sorted(xsess_holdout[:mid])
        a["xsession_test"] = sorted(xsess_holdout[mid:])

        # ── Train ──
        held_out = xsubj_session_set | set(xsess_holdout)
        a["train"] = sorted([s for s in all_sessions if s not in held_out])

        assignments[source] = a

    return assignments


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Write YAML selection files
# ──────────────────────────────────────────────────────────────────────────────


def yaml_selection(brainset, sessions, ignore_splits=True):
    """Format a single selection block."""
    lines = []
    lines.append(f"  - brainset: {brainset}")
    lines.append("    sessions:")
    for s in sessions:
        lines.append(f"      - {s}")
    return "\n".join(lines)


def yaml_selection_exclude(brainset, exclude_sessions):
    """Format a selection block using exclude_sessions."""
    lines = []
    lines.append(f"  - brainset: {brainset}")
    lines.append("    exclude_sessions:")
    for s in exclude_sessions:
        lines.append(f"      - {s}")
    return "\n".join(lines)


def write_selection_file(path, comment, selections_content, ignore_splits=True):
    """Write a complete selection YAML file."""
    lines = [f"# {comment}"]
    lines.append("_target_: ntt.core.data.components.config.SelectionConfig")
    lines.append("selection:")
    lines.append(selections_content)
    if ignore_splits:
        lines.append("config:")
        lines.append("  ignore_splits: true")
    lines.append("")  # trailing newline
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"  wrote {path.relative_to(REPO_ROOT)}")


def write_per_source_files(assignments, subject_map):
    """Write per-source train files, plus the eval files in PER_SOURCE_EVAL_FILES."""

    for source in WRITTEN_SOURCES:
        a = assignments[source]
        eval_kind = PER_SOURCE_EVAL_FILES.get(source)

        if source in XSOURCE_SOURCES:
            # Cross-source: all sessions go to val/test, none in train.
            if eval_kind == "xsource" and a["xsource_val"]:
                write_selection_file(
                    SELECTIONS_DIR / f"{source}_val.yaml",
                    f"Cross-source val: {source} (entirely held out from training).",
                    yaml_selection(source, a["xsource_val"]),
                )
            if eval_kind == "xsource" and a["xsource_test"]:
                write_selection_file(
                    SELECTIONS_DIR / f"{source}_test.yaml",
                    f"Cross-source test: {source} (entirely held out from training).",
                    yaml_selection(source, a["xsource_test"]),
                )
            continue

        held_out = set()
        for key in ("xsubject_val", "xsubject_test", "xsession_val", "xsession_test"):
            held_out.update(a[key])

        # ── Train (exclude_sessions) ──
        write_selection_file(
            SELECTIONS_DIR / f"{source}_train.yaml",
            "Train: all sessions except cross-session and cross-subject eval holdouts.",
            yaml_selection_exclude(source, sorted(held_out)),
            ignore_splits=False,
        )

        # ── Val (cross-session) ──
        if eval_kind == "xsession" and a["xsession_val"]:
            write_selection_file(
                SELECTIONS_DIR / f"{source}_val.yaml",
                "Val: cross-session held-out sessions (subjects seen in train).",
                yaml_selection(source, a["xsession_val"]),
            )

        # ── Test (cross-session) ──
        if eval_kind == "xsession" and a["xsession_test"]:
            write_selection_file(
                SELECTIONS_DIR / f"{source}_test.yaml",
                "Test: cross-session held-out sessions (subjects seen in train).",
                yaml_selection(source, a["xsession_test"]),
            )

        # ── Cross-subject val ──
        if eval_kind == "xsubject" and a["xsubject_val"]:
            write_selection_file(
                SELECTIONS_DIR / f"{source}_xsubject_val.yaml",
                "Cross-subject val: sessions from subjects not in training.",
                yaml_selection(source, a["xsubject_val"]),
            )

        # ── Cross-subject test ──
        if eval_kind == "xsubject" and a["xsubject_test"]:
            write_selection_file(
                SELECTIONS_DIR / f"{source}_xsubject_test.yaml",
                "Cross-subject test: sessions from subjects not in training.",
                yaml_selection(source, a["xsubject_test"]),
            )


def write_composite_files(assignments, subject_map):
    """Write composite selection files that combine all sources per tier."""

    COMPOSITE_DIR.mkdir(parents=True, exist_ok=True)

    # ── xtime (within-session): train sessions, NO ignore_splits ──
    # Uses exclude_sessions approach per brainset
    blocks = []
    for source in WRITTEN_SOURCES:
        if source in XSOURCE_SOURCES:
            continue
        a = assignments[source]
        held_out = sorted(
            set(a["xsubject_val"] + a["xsubject_test"] + a["xsession_val"] + a["xsession_test"])
        )
        blocks.append(yaml_selection_exclude(source, held_out))

    write_selection_file(
        COMPOSITE_DIR / "ntt_full_val_xtime.yaml",
        "Tier 1 — Within-session temporal eval. Same sessions as train; use with\n"
        "# split='valid' to evaluate on the held-out temporal portion (valid_domain).\n"
        "# NOTE: Do NOT set ignore_splits for this tier.",
        "\n".join(blocks),
        ignore_splits=False,
    )

    # ── xsession (cross-session val) ──
    blocks = []
    for source in WRITTEN_SOURCES:
        if source in XSOURCE_SOURCES:
            continue
        a = assignments[source]
        if a["xsession_val"]:
            blocks.append(yaml_selection(source, a["xsession_val"]))
    if blocks:
        write_selection_file(
            COMPOSITE_DIR / "ntt_full_val_xsession.yaml",
            "Tier 2 — Cross-session val: held-out sessions from subjects seen in training.",
            "\n".join(blocks),
        )

    # ── xsession (cross-session test) ──
    blocks = []
    for source in WRITTEN_SOURCES:
        if source in XSOURCE_SOURCES:
            continue
        a = assignments[source]
        if a["xsession_test"]:
            blocks.append(yaml_selection(source, a["xsession_test"]))
    if blocks:
        write_selection_file(
            COMPOSITE_DIR / "ntt_full_test_xsession.yaml",
            "Tier 2 — Cross-session test: held-out sessions from subjects seen in training.",
            "\n".join(blocks),
        )

    # ── xsubject (cross-subject val) ──
    blocks = []
    for source in WRITTEN_SOURCES:
        if source in XSOURCE_SOURCES:
            continue
        a = assignments[source]
        if a["xsubject_val"]:
            blocks.append(yaml_selection(source, a["xsubject_val"]))
    if blocks:
        write_selection_file(
            COMPOSITE_DIR / "ntt_full_val_xsubject.yaml",
            "Tier 3 — Cross-subject val: sessions from subjects not in training.",
            "\n".join(blocks),
        )

    # ── xsubject (cross-subject test) ──
    blocks = []
    for source in WRITTEN_SOURCES:
        if source in XSOURCE_SOURCES:
            continue
        a = assignments[source]
        if a["xsubject_test"]:
            blocks.append(yaml_selection(source, a["xsubject_test"]))
    if blocks:
        write_selection_file(
            COMPOSITE_DIR / "ntt_full_test_xsubject.yaml",
            "Tier 3 — Cross-subject test: sessions from subjects not in training.",
            "\n".join(blocks),
        )

    # ── xsource (cross-source): combine all xsource datasets ──
    for split in ("val", "test"):
        blocks = []
        for source in WRITTEN_SOURCES:
            if source not in XSOURCE_SOURCES:
                continue
            a = assignments[source]
            key = f"xsource_{split}"
            if a[key]:
                blocks.append(yaml_selection(source, a[key]))
        if blocks:
            source_names = ", ".join(s for s in WRITTEN_SOURCES if s in XSOURCE_SOURCES)
            write_selection_file(
                COMPOSITE_DIR / f"ntt_full_{split}_xsource.yaml",
                f"Tier 4 — Cross-source {split}: sources entirely held out from training.\n"
                f"# Sources: {source_names}",
                "\n".join(blocks),
            )


# ──────────────────────────────────────────────────────────────────────────────
# Step 4: Print summary
# ──────────────────────────────────────────────────────────────────────────────


def print_summary(assignments, subject_map):
    """Print a summary of the tier assignments."""
    print("\n" + "=" * 70)
    print("TIER ASSIGNMENT SUMMARY")
    print("=" * 70)

    totals = {
        "train": 0,
        "xsession_val": 0,
        "xsession_test": 0,
        "xsubject_val": 0,
        "xsubject_test": 0,
        "xsource_val": 0,
        "xsource_test": 0,
    }

    for source in ALL_SOURCES:
        a = assignments[source]
        print(f"\n{source}:")

        if source in XSOURCE_SOURCES:
            nv = len(a["xsource_val"])
            nt = len(a["xsource_test"])
            total = nv + nt
            print(f"  xsource val: {nv}, xsource test: {nt}")
            print(f"  total: {total} (expected: {len(subject_map[source])})")
            assert total == len(subject_map[source]), f"Session count mismatch for {source}!"
            totals["xsource_val"] += nv
            totals["xsource_test"] += nt
            continue

        n_train = len(a["train"])
        n_xsess_val = len(a["xsession_val"])
        n_xsess_test = len(a["xsession_test"])
        n_xsubj_val = len(a["xsubject_val"])
        n_xsubj_test = len(a["xsubject_test"])
        total = n_train + n_xsess_val + n_xsess_test + n_xsubj_val + n_xsubj_test

        print(f"  train: {n_train}")
        if n_xsess_val or n_xsess_test:
            print(f"  xsession val: {n_xsess_val}, test: {n_xsess_test}")
        if n_xsubj_val or n_xsubj_test:
            # Show subjects
            xsubj_subjects_val = {subject_map[source][s] for s in a["xsubject_val"]}
            xsubj_subjects_test = {subject_map[source][s] for s in a["xsubject_test"]}
            print(
                f"  xsubject val: {n_xsubj_val} ({', '.join(sorted(xsubj_subjects_val))}), "
                f"test: {n_xsubj_test} ({', '.join(sorted(xsubj_subjects_test))})"
            )
        print(f"  total: {total} (expected: {len(subject_map[source])})")
        assert total == len(subject_map[source]), f"Session count mismatch for {source}!"

        totals["train"] += n_train
        totals["xsession_val"] += n_xsess_val
        totals["xsession_test"] += n_xsess_test
        totals["xsubject_val"] += n_xsubj_val
        totals["xsubject_test"] += n_xsubj_test

    print(f"\n{'=' * 70}")
    print("TOTALS:")
    print(f"  Train:         {totals['train']}")
    print(f"  xsession val:  {totals['xsession_val']}")
    print(f"  xsession test: {totals['xsession_test']}")
    print(f"  xsubject val:  {totals['xsubject_val']}")
    print(f"  xsubject test: {totals['xsubject_test']}")
    print(f"  xsource val:   {totals['xsource_val']}")
    print(f"  xsource test:  {totals['xsource_test']}")
    total_eval = (
        totals["xsession_val"]
        + totals["xsession_test"]
        + totals["xsubject_val"]
        + totals["xsubject_test"]
        + totals["xsource_val"]
        + totals["xsource_test"]
    )
    print(f"  Total eval:    {total_eval}")
    print(f"  Grand total:   {totals['train'] + total_eval}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


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
        help="Overwrite the canonical selection files with a newly generated split.",
    )
    args = parser.parse_args(argv)

    refuse_to_overwrite(args.force)

    print("Loading subject map from h5 files...")
    subject_map = load_subject_map(args.data_root)
    check_pools(subject_map)

    print("\nAssigning sessions to tiers (seed=42)...")
    assignments = assign_tiers(subject_map)

    print("\nWriting per-source selection files...")
    write_per_source_files(assignments, subject_map)

    print("\nWriting composite selection files...")
    write_composite_files(assignments, subject_map)

    print_summary(assignments, subject_map)


if __name__ == "__main__":
    main()
