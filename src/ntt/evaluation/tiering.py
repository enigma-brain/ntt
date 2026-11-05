"""Tier membership resolution for evaluation aggregation.

A tier is defined by a composite selection YAML. Two block styles are
supported per ``selection`` entry:

- inclusive: ``brainset`` + ``sessions`` -> the listed ``brainset/session``
  ids belong to the tier.
- exclusive: ``brainset`` + ``exclude_sessions`` -> any evaluated
  ``brainset/<name>`` belongs to the tier UNLESS ``<name>`` is in the
  exclude list. This is how the ``xtime`` tier (training-set complement)
  is defined; it cannot be enumerated to a fixed set, so membership is a
  predicate resolved against the sessions actually evaluated.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TierMembership:
    """Which sessions belong to a tier.

    ``include`` holds explicit ``brainset/session`` ids. ``exclude_brainsets``
    maps a brainset to the set of session *names* held out of the tier; any
    evaluated session in that brainset whose name is not listed belongs.
    """

    name: str
    include: frozenset[str] = frozenset()
    exclude_brainsets: Mapping[str, frozenset[str]] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not self.include and not self.exclude_brainsets

    def has_explicit_include(self) -> bool:
        return bool(self.include)

    def matches(self, session_id: str) -> bool:
        if session_id in self.include:
            return True
        brainset, _, name = session_id.partition("/")
        if brainset in self.exclude_brainsets:
            return name not in self.exclude_brainsets[brainset]
        return False


def read_explicit_sessions(yaml_path: str | Path) -> set[str]:
    """Return the explicit ``brainset/session`` ids listed in a composite
    selection YAML.

    Only reads ``sessions:`` blocks; ``exclude_sessions:`` blocks name a
    complement (not an enumerable set) and are ignored here.
    """
    cfg = cast(DictConfig, OmegaConf.load(yaml_path))
    return {
        f"{sel.get('brainset', '')}/{s}"
        for sel in cfg.get("selection", [])
        for s in (sel.get("sessions", []) or [])
    }


def parse_tier_membership(name: str, yaml_path: str | Path) -> TierMembership:
    """Parse a composite selection YAML into a :class:`TierMembership`."""
    cfg = cast(DictConfig, OmegaConf.load(yaml_path))
    include = read_explicit_sessions(yaml_path)
    exclude_brainsets: dict[str, frozenset[str]] = {}
    for sel in cfg.get("selection", []):
        brainset = sel.get("brainset", "")
        excluded = sel.get("exclude_sessions", None)
        if excluded is not None:
            # Composites currently list each brainset once; union guards
            # against a brainset appearing in multiple selection entries.
            exclude_brainsets[brainset] = exclude_brainsets.get(brainset, frozenset()) | frozenset(
                str(s) for s in excluded
            )
    return TierMembership(
        name=name,
        include=frozenset(include),
        exclude_brainsets=exclude_brainsets,
    )


def assign_sessions_to_tiers(
    session_ids: Iterable[str],
    memberships: Mapping[str, TierMembership],
) -> dict[str, set[str]]:
    """Assign each evaluated session to at most one tier.

    Explicit-``include`` matches take priority over exclude-based matches, so
    an OOD session is never double-counted into the exclude-based xtime tier.
    Remaining ambiguities are resolved deterministically (sorted tier name)
    with a warning.
    """
    assignment: dict[str, set[str]] = {name: set() for name in memberships}
    for sid in session_ids:
        matched = [name for name, m in memberships.items() if m.matches(sid)]
        if not matched:
            continue
        include_matches = [name for name in matched if memberships[name].has_explicit_include()]
        candidates = include_matches or matched
        if len(candidates) > 1:
            chosen = sorted(candidates)[0]
            log.warning(
                "Session %s matched multiple tiers %s; assigning to %r.",
                sid,
                sorted(candidates),
                chosen,
            )
        else:
            chosen = candidates[0]
        assignment[chosen].add(sid)
    return assignment
