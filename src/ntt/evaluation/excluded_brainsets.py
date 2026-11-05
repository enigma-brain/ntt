"""Registry of brainsets excluded from production train/val/test selections.

When a brainset is removed from the canonical selections after the fact
(typically because its data is judged unfit for evaluation), eval
reports produced *before* the removal still contain its per-session
entries. Loaders that consume such reports use this registry to
detect those entries and warn (or filter) — see
``EvalArtifact.load_json`` and ``EvalArtifact.without_brainsets``.

Adding a new excluded brainset:
    1. Add an entry to ``EXCLUDED_BRAINSET_SESSIONS`` mapping the brainset
       name to the frozen set of session IDs that ever appeared in any
       active selection (train/val/test) before exclusion.
    2. Remove the brainset from every selection YAML, but keep it in
       ``XSOURCE_SOURCES`` and ``ALL_SOURCES`` in
       ``scripts/generate_tiered_selections.py`` so it still consumes its part
       of the shared random-number stream and preserves later datasets' splits.

This registry is the record of the excluded session IDs; no selection YAML
backs it, so an excluded brainset is never loaded even if its data is still
on disk (selections name brainsets explicitly).
"""

from __future__ import annotations

from collections.abc import Iterable

# Brainset name → all session IDs that ever appeared in active val/test/train
# selections for that brainset. Union across all splits (some sessions appear
# in multiple splits).
EXCLUDED_BRAINSET_SESSIONS: dict[str, frozenset[str]] = {
    # Excluded: data not clean enough for evaluation.
    # All 9 sessions on disk (the cross-source val/test holdouts).
    "tostado_marcos_birdsong_2024": frozenset(
        {
            "finch_z_r12r13_21_20210626_held_in_calib",
            "finch_z_r12r13_21_20210626_held_in_minival",
            "finch_z_r12r13_21_20210627_held_in_calib",
            "finch_z_r12r13_21_20210627_held_in_minival",
            "finch_z_r12r13_21_20210628_held_in_calib",
            "finch_z_r12r13_21_20210628_held_in_minival",
            "finch_z_r12r13_21_20210630_held_out_calib",
            "finch_z_r12r13_21_20210701_held_out_calib",
            "finch_z_r12r13_21_20210705_held_out_calib",
        }
    ),
}

# Flat union, useful for quick membership tests.
ALL_EXCLUDED_SESSIONS: frozenset[str] = frozenset().union(*EXCLUDED_BRAINSET_SESSIONS.values())


def find_excluded_sessions(session_ids: Iterable[str]) -> dict[str, list[str]]:
    """Group session IDs that belong to excluded brainsets.

    Returns a mapping ``{brainset_name: [matching_session_ids]}``, restricted
    to brainsets that actually have a match. Sessions not in any excluded
    brainset are silently dropped — callers checking ``if find_excluded_sessions(...)``
    get a truthy/falsy answer for "does this report contain anything excluded".
    """
    found: dict[str, list[str]] = {}
    sids = set(session_ids)
    for brainset, members in EXCLUDED_BRAINSET_SESSIONS.items():
        hits = sorted(sids & members)
        if hits:
            found[brainset] = hits
    return found


def sessions_for_brainsets(brainsets: Iterable[str]) -> set[str]:
    """Return the union of excluded session IDs across the given brainsets.

    Unknown brainset names are silently ignored — used by callers that pass
    a user-supplied filter list and don't want to error on typos.
    """
    out: set[str] = set()
    for b in brainsets:
        out |= EXCLUDED_BRAINSET_SESSIONS.get(b, frozenset())
    return out
