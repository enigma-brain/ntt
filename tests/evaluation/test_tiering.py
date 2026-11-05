"""Tests for ntt.evaluation.tiering — tier membership resolution."""

from __future__ import annotations

from pathlib import Path

from ntt.evaluation.tiering import (
    TierMembership,
    assign_sessions_to_tiers,
    parse_tier_membership,
    read_explicit_sessions,
)

REPO = Path(__file__).resolve().parents[2]
VAL_XTIME = REPO / "config/data/selections/composite/ntt_full_val_xtime.yaml"
VAL_XSESSION = REPO / "config/data/selections/composite/ntt_full_val_xsession.yaml"


def test_exclude_based_membership_is_not_empty():
    m = parse_tier_membership("xtime", VAL_XTIME)
    assert not m.is_empty()
    assert m.include == frozenset()  # xtime has no explicit sessions
    assert "IBL_brainwide_2023" in m.exclude_brainsets


def test_exclude_based_matches_training_session_not_excluded():
    m = parse_tier_membership("xtime", VAL_XTIME)
    # A training session (not in the exclude list) -> belongs to xtime.
    assert m.matches("IBL_brainwide_2023/csh_zad_011_20200321_csh_zad_011")
    # An excluded OOD session -> does NOT belong to xtime.
    assert not m.matches("IBL_brainwide_2023/csh_zad_022_20200526_csh_zad_022")
    # A session whose brainset is not in the xtime selection -> no match.
    assert not m.matches("some_unknown_brainset/whatever")


def test_inclusive_membership_matches_only_listed_sessions():
    m = parse_tier_membership("xsession", VAL_XSESSION)
    assert m.exclude_brainsets == {}
    assert m.matches("IBL_brainwide_2023/csh_zad_022_20200526_csh_zad_022")
    assert not m.matches("IBL_brainwide_2023/csh_zad_011_20200321_csh_zad_011")


def test_assign_prefers_include_over_exclude():
    # csh_zad_022 is in xsession.include AND would match xtime by exclusion's
    # complement IF it were not in xtime's exclude list. It IS in xtime's
    # exclude list, so it only matches xsession. Verify single, correct tier.
    memberships = {
        "xtime": parse_tier_membership("xtime", VAL_XTIME),
        "xsession": parse_tier_membership("xsession", VAL_XSESSION),
    }
    evaluated = {
        "IBL_brainwide_2023/csh_zad_011_20200321_csh_zad_011",  # train -> xtime
        "IBL_brainwide_2023/csh_zad_022_20200526_csh_zad_022",  # held-out -> xsession
    }
    assignment = assign_sessions_to_tiers(evaluated, memberships)
    assert assignment["xtime"] == {"IBL_brainwide_2023/csh_zad_011_20200321_csh_zad_011"}
    assert assignment["xsession"] == {"IBL_brainwide_2023/csh_zad_022_20200526_csh_zad_022"}


def test_include_priority_breaks_overlap():
    # Construct an artificial overlap: a session that is BOTH explicitly
    # included in tier "ood" and would match exclude-based "xtime".
    sid = "bs/train_sess"
    xtime = TierMembership(
        name="xtime",
        include=frozenset(),
        exclude_brainsets={"bs": frozenset()},  # excludes nothing -> matches all "bs/*"
    )
    ood = TierMembership(name="ood", include=frozenset({sid}), exclude_brainsets={})
    assignment = assign_sessions_to_tiers({sid}, {"xtime": xtime, "ood": ood})
    # Explicit include wins; session is NOT double-counted into xtime.
    assert assignment["ood"] == {sid}
    assert assignment["xtime"] == set()


def test_assign_warns_and_picks_sorted_first_on_tie():
    # Two exclude-based tiers both match the same session (no explicit include
    # to break the tie) -> deterministic sorted-first assignment.
    sid = "bs/sess"
    tier_b = TierMembership(name="b", include=frozenset(), exclude_brainsets={"bs": frozenset()})
    tier_a = TierMembership(name="a", include=frozenset(), exclude_brainsets={"bs": frozenset()})
    assignment = assign_sessions_to_tiers({sid}, {"b": tier_b, "a": tier_a})
    assert assignment["a"] == {sid}  # sorted-first wins
    assert assignment["b"] == set()


def test_empty_exclude_list_matches_every_session_in_brainset():
    # exclude_sessions: [] -> an empty frozenset entry -> matches ALL sessions
    # in that brainset (distinct from a missing brainset key, which matches none).
    m = TierMembership(name="t", include=frozenset(), exclude_brainsets={"bs": frozenset()})
    assert m.matches("bs/anything")
    assert not m.matches("other_bs/anything")


def test_parse_membership_handles_duplicate_brainset_entries(tmp_path):
    # Two selection entries for the same brainset with exclude_sessions are
    # unioned, not overwritten.
    yaml_path = tmp_path / "dup.yaml"
    yaml_path.write_text(
        "_target_: ntt.core.data.components.config.SelectionConfig\n"
        "selection:\n"
        "  - brainset: bs\n"
        "    exclude_sessions:\n"
        "      - a\n"
        "  - brainset: bs\n"
        "    exclude_sessions:\n"
        "      - b\n"
    )
    m = parse_tier_membership("xtime", yaml_path)
    assert m.exclude_brainsets["bs"] == frozenset({"a", "b"})
    # Both excluded sessions are correctly held out.
    assert not m.matches("bs/a")
    assert not m.matches("bs/b")
    assert m.matches("bs/c")


def test_eval_light_sessions_populate_xtime_tier():
    """The eval_light xtime sessions land in the xtime tier (an exclude-based xtime
    selection must not be dropped, which would skip generalization)."""
    tier_dir = REPO / "config/data/selections/composite"
    memberships = {
        "xtime": parse_tier_membership("xtime", tier_dir / "ntt_full_val_xtime.yaml"),
        "xsession": parse_tier_membership("xsession", tier_dir / "ntt_full_val_xsession.yaml"),
        "xsubject": parse_tier_membership("xsubject", tier_dir / "ntt_full_val_xsubject.yaml"),
        "xsource": parse_tier_membership("xsource", tier_dir / "ntt_full_val_xsource.yaml"),
    }
    eval_light_xtime = [
        "IBL_brainwide_2023/csh_zad_011_20200321_csh_zad_011",
        "odoherty_sabes_nonhuman_2017/indy_20160407_02",
        "perich_miller_population_2018/c_20131029_random_target_reaching",
        "flint_slutzky_accurate_2012/monkey_c_e4_reaching",
        "churchland_shenoy_neural_2012/jenkins_20090918_center_out_reaching",
    ]
    eval_light_ood = [
        "IBL_brainwide_2023/csh_zad_022_20200526_csh_zad_022",  # xsession
        "odoherty_sabes_nonhuman_2017/loco_20170210_03",  # xsubject
        "c4_dataset_cerebellum_2025/hausser_19-01-24_YC013_probe1",  # xsource
    ]
    assignment = assign_sessions_to_tiers(eval_light_xtime + eval_light_ood, memberships)
    assert set(eval_light_xtime).issubset(assignment["xtime"])
    # None of the OOD sessions leaked into xtime.
    assert assignment["xtime"].isdisjoint(eval_light_ood)


TEST_XTIME = REPO / "config/data/selections/composite/ntt_full_test_xtime.yaml"


def _ood_session_ids(*paths) -> set[str]:
    return set().union(*(read_explicit_sessions(p) for p in paths))


def test_test_split_xtime_classifies_correctly():
    m = parse_tier_membership("xtime", TEST_XTIME)
    assert not m.is_empty()
    # A training session belongs to xtime on the test split too.
    assert m.matches("IBL_brainwide_2023/csh_zad_011_20200321_csh_zad_011")
    # Every test-OOD session must be excluded from xtime (no leakage into base).
    composite = REPO / "config/data/selections/composite"
    ood = _ood_session_ids(
        composite / "ntt_full_test_xsession.yaml",
        composite / "ntt_full_test_xsubject.yaml",
        composite / "ntt_full_test_xsource.yaml",
    )
    for sid in ood:
        assert not m.matches(sid), f"{sid} wrongly classified as xtime"


def test_test_tier_set_defines_xtime():
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(REPO / "config/tier/ntt_full_test.yaml")
    assert "xtime" in cfg
    assert str(cfg["xtime"]).endswith("ntt_full_test_xtime.yaml")
