"""Tests for ntt.evaluation.metric_directions — direction resolution from source.

These replace the coverage that a hand-generated ``METRIC_DIRECTIONS`` dict used
to (partially) provide.  The dict silently covered only ~51% of the production
wide table — every ``robustness_latent/`` column was missing — precisely because
nothing tested it against a table that had those columns.  Hence
``test_every_sample_wide_column_resolves`` below, plus the M2 block that was
added to the shared dashboard fixture for it to bite on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import scripts.analysis.dashboard as D
import yaml

from ntt.evaluation.core.types import MetricDirection
from ntt.evaluation.metric_directions import (
    build_display_name_index,
    default_config_root,
    direction_map_for_columns,
    resolve_direction,
)
from ntt.evaluation.metrics.base import METRIC_REGISTRY

REPO = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO / "config"
WIDE = REPO / "tests/data/dashboard/SAMPLE_wide_metrics_table.csv"
EVAL_REPORTS = REPO / "tests/data/eval_reports"


def _value_columns(df: pd.DataFrame) -> list[str]:
    """The wide columns the dashboard actually plots (see build_metric_registry)."""
    return [c for c in df.columns if "/" in c and "::" not in c and not c.startswith("gen_tier/")]


def test_config_root_default_points_at_the_repo_config():
    assert default_config_root() == CONFIG_ROOT


# ── 1. config integrity ────────────────────────────────────────────────────
def test_every_config_display_name_maps_to_a_registered_class():
    index = build_display_name_index(CONFIG_ROOT)
    assert index, "no display names found — the config scan is broken"
    unregistered = {display: cls for display, cls in index.items() if cls not in METRIC_REGISTRY}
    assert not unregistered, f"display names bound to unregistered classes: {unregistered}"


def test_display_names_are_unambiguous_across_the_config_tree():
    # build_display_name_index raises on conflict; assert independently here so a
    # regression points at the offending config rather than at every other test.
    seen: dict[str, tuple[str, Path]] = {}
    conflicts: list[str] = []

    def visit(node, path: Path) -> None:
        if isinstance(node, dict):
            block = node.get("metrics")
            if isinstance(block, dict):
                for display, cfg in block.items():
                    if not isinstance(cfg, dict):
                        continue
                    cls = cfg.get("metric_class")
                    if not isinstance(cls, str):
                        continue
                    prev = seen.setdefault(display, (cls, path))
                    if prev[0] != cls:
                        conflicts.append(f"{display}: {prev[0]} ({prev[1]}) vs {cls} ({path})")
            for value in node.values():
                visit(value, path)
        elif isinstance(node, list):
            for value in node:
                visit(value, path)

    for path in sorted(CONFIG_ROOT.rglob("*.yaml")):
        doc = yaml.safe_load(path.read_text())
        if not isinstance(doc, dict):
            continue
        if isinstance(doc.get("metric_class"), str):
            prev = seen.setdefault(path.stem, (doc["metric_class"], path))
            if prev[0] != doc["metric_class"]:
                conflicts.append(f"{path.stem}: {prev[0]} ({prev[1]}) vs {doc['metric_class']}")
        visit(doc, path)

    assert not conflicts, "; ".join(conflicts)


# ── 2. artifact round-trip: the static resolver == the artifact's own field ──
@pytest.mark.parametrize("report", ["eval_report.json", "per_session.json"])
def test_resolver_matches_direction_recorded_in_eval_artifacts(report):
    """The load-bearing test.

    ``generalization.py`` reads ``MetricResult.direction`` straight off the
    report; the dashboard has only the column name.  Both must agree, or a
    metric would be ranked one way in the generalization table and arrowed the
    other way in the dashboard.
    """
    results = json.loads((EVAL_REPORTS / report).read_text())["results"]
    assert results

    mismatches = []
    for name, result in results.items():
        resolved = resolve_direction(result["category"], name, config_root=CONFIG_ROOT)
        expected = MetricDirection(result["direction"])
        if resolved is not expected:
            mismatches.append((name, result["category"], expected, resolved))
    assert not mismatches, f"{len(mismatches)} direction mismatches: {mismatches[:5]}"


# ── 3. wide-table completeness ─────────────────────────────────────────────
def test_every_sample_wide_column_resolves():
    df = pd.read_csv(WIDE, nrows=1)
    columns = _value_columns(df)
    resolved = direction_map_for_columns(columns, config_root=CONFIG_ROOT)
    missing = sorted(set(columns) - set(resolved))
    assert not missing, f"{len(missing)}/{len(columns)} wide columns have no direction: {missing}"


def test_sample_wide_table_still_has_an_m2_block():
    # Guards the fixture itself: without robustness_latent/ columns the
    # completeness test above passes vacuously, which is how the old
    # hand-written map hid a 50% coverage hole.
    df = pd.read_csv(WIDE, nrows=1)
    m2 = [c for c in _value_columns(df) if c.startswith("robustness_latent/")]
    assert len(m2) >= 10, f"M2 block missing from the fixture (found {len(m2)} columns)"


def test_m2_resolves_both_class_name_and_display_name_spellings():
    # M2 results can be keyed by metric *class* name or by display name, so the
    # same metric can be spelled either way and both must resolve — see the
    # comment in resolve_direction.
    for spelling in ("connectivity_f1", "functional_connectivity_f1"):
        assert (
            resolve_direction("robustness_latent", f"{spelling}/eps_0.1", config_root=CONFIG_ROOT)
            is MetricDirection.HIGHER_IS_BETTER
        )
    for spelling in ("dimensionality_error", "participation_ratio_error"):
        assert (
            resolve_direction("robustness_latent", f"{spelling}/eps_1", config_root=CONFIG_ROOT)
            is MetricDirection.LOWER_IS_BETTER
        )


def test_key_shapes_resolve():
    r = lambda block, part: resolve_direction(block, part, config_root=CONFIG_ROOT)  # noqa: E731
    # recon, display name == class name
    assert r("temporal_fidelity", "spike_detection_f1") is MetricDirection.HIGHER_IS_BETTER
    # recon, display name != class name (isi_cv2_error -> cv2_error)
    assert r("single_unit_statistics", "isi_cv2_error") is MetricDirection.LOWER_IS_BETTER
    # multi-scale suffix that no config declares (expand_results invents it)
    assert (
        r("population_structure", "population_coupling_preservation_10ms")
        is MetricDirection.HIGHER_IS_BETTER
    )
    # M1, both the token-level and the per-dim NMSE variant
    assert r("robustness", "robustness_input/permute_neurons") is MetricDirection.LOWER_IS_BETTER
    assert (
        r("robustness", "robustness_input_per_dim/jitter|k_ms=1.0|p=0.2")
        is MetricDirection.LOWER_IS_BETTER
    )
    # non-registry blocks: R²/Pearson, higher by definition
    assert r("encoding", "nc_r2") is MetricDirection.HIGHER_IS_BETTER
    assert r("decoding", "test_r2") is MetricDirection.HIGHER_IS_BETTER


# ── 4. never guess ─────────────────────────────────────────────────────────
def test_unknown_metrics_resolve_to_none():
    assert resolve_direction("madeup", "not_a_real_metric", config_root=CONFIG_ROOT) is None
    # a plausible-looking name is still not a guess: no 'error' -> lower heuristic
    assert resolve_direction("temporal_fidelity", "made_up_error", config_root=CONFIG_ROOT) is None
    assert resolve_direction("robustness_latent", "nope/eps_0.1", config_root=CONFIG_ROOT) is None
    assert direction_map_for_columns(["madeup/not_a_real_metric"]) == {}


def test_dashboard_renders_no_arrow_for_unknown_metrics():
    assert D._direction_arrow("madeup/not_a_real_metric", "madeup") == ""
