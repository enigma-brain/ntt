"""Tests for ntt.evaluation.generalization."""

from __future__ import annotations

import math

import pytest

from ntt.evaluation.generalization import (
    GeneralizationResult,
    compute_from_per_session_reports,
    compute_from_session_tiers,
)


def _entry(*metrics: tuple[str, float, str]) -> dict:
    """One session's report: (name, value, direction) triples."""
    return {
        "results": {
            name: {
                "raw_value": value,
                "direction": direction,
                "category": "single_unit_statistics",
            }
            for name, value, direction in metrics
        }
    }


def _score(per_session: dict, session_tiers: dict, **kwargs) -> GeneralizationResult:
    kwargs.setdefault("ood_tiers", ("xsession",))
    kwargs.setdefault("score_tiers", kwargs["ood_tiers"])
    result = compute_from_session_tiers(per_session, session_tiers, **kwargs)
    assert result is not None
    return result


def _rows(result) -> dict:
    return {(r["tier"], r["metric"]): r for r in result.per_metric}


# One baseline and one held-out session from the same brainset: the matched and the
# tier-mean baselines agree, so these pin the retention math itself.
ONE_PAIR = {"bsA/t": "xtime", "bsA/s": "xsession"}


def test_higher_is_better_retention_math():
    """For higher-is-better, retention = tier / xtime."""
    per_session = {
        "bsA/t": _entry(("acc", 0.8, "higher")),
        "bsA/s": _entry(("acc", 0.6, "higher")),
    }
    out = _score(per_session, ONE_PAIR)
    assert _rows(out)[("xsession", "acc")]["retention"] == pytest.approx(0.75)


def test_lower_is_better_retention_math():
    """For lower-is-better, retention = xtime / tier."""
    per_session = {"bsA/t": _entry(("err", 0.1, "lower")), "bsA/s": _entry(("err", 0.4, "lower"))}
    out = _score(per_session, ONE_PAIR)
    assert _rows(out)[("xsession", "err")]["retention"] == pytest.approx(0.25)


def test_retention_clipped_to_unit_interval():
    """OOD better than ID still caps at 1."""
    per_session = {
        "bsA/t": _entry(("acc", 0.5, "higher")),
        "bsA/s": _entry(("acc", 0.9, "higher")),
    }
    row = _rows(_score(per_session, ONE_PAIR))[("xsession", "acc")]
    assert row["retention_raw"] == pytest.approx(1.8)
    assert row["retention"] == 1.0


def test_excluded_metrics_dropped():
    per_session = {
        "bsA/t": _entry(("acc", 0.8, "higher"), ("n_skipped", 5, "lower")),
        "bsA/s": _entry(("acc", 0.6, "higher"), ("n_skipped", 8, "lower")),
    }
    metric_names = {r["metric"] for r in _score(per_session, ONE_PAIR).per_metric}
    assert metric_names == {"acc"}


def _with_categories(value: float) -> dict:
    """``acc`` (reconstruction) plus one metric of each non-reconstruction category."""
    return {
        "results": {
            name: {"raw_value": value, "direction": "higher", "category": category}
            for name, category in [
                ("acc", "temporal_fidelity"),
                ("population_structure_preservation", "latent_quality"),
                ("behavioral_regression_cursor_vel", "downstream_decoding"),
                ("encoding_r2", "downstream_encoding"),
            ]
        }
    }


def test_only_reconstruction_metrics_are_scored():
    per_session = {"bsA/t": _with_categories(0.8), "bsA/s": _with_categories(0.6)}
    out = _score(per_session, ONE_PAIR)
    assert {r["metric"] for r in out.per_metric} == {"acc"}
    assert out.to_dict()["excluded_categories"] == [
        "downstream_decoding",
        "downstream_encoding",
        "latent_quality",
    ]
    everything = _score(per_session, ONE_PAIR, excluded_categories=())
    assert len(everything.per_metric) == 4


def test_per_session_reports_score_reconstruction_metrics_only():
    per_session = {"bsA/t": _with_categories(0.8), "bsA/s": _with_categories(0.6)}
    result, _ = compute_from_per_session_reports(
        per_session,
        explicit_tiers={("bsA", "s"): "xsession"},
        xtime_excludes={"bsA": {"s"}},
        ood_tiers=("xsession",),
        score_tiers=("xsession",),
    )
    assert result is not None
    assert {r["metric"] for r in result.per_metric} == {"acc"}


def test_no_baseline_sessions_returns_none():
    per_session = {"bsA/s": _entry(("acc", 0.6, "higher"))}
    assert compute_from_session_tiers(per_session, {"bsA/s": "xsession"}) is None


def test_summary_geometric_mean():
    """All-tier retentions: geometric mean is the score, mean is mean_retention."""
    per_session = {
        "bsA/t": _entry(("a", 1.0, "higher"), ("b", 1.0, "higher")),
        "bsA/s": _entry(("a", 0.5, "higher"), ("b", 0.25, "higher")),
    }
    out = _score(per_session, ONE_PAIR)
    assert out.summary["mean_retention"] == pytest.approx(0.375)
    assert out.summary["generalization_score"] == pytest.approx(
        math.exp((math.log(0.5) + math.log(0.25)) / 2)
    )
    assert out.summary["n_retention_terms"] == 2


def test_geometric_mean_floor_clamps_small_retention():
    """A retention smaller than retention_floor gets floored before log-mean."""
    per_session = {
        "bsA/t": _entry(("acc", 1.0, "higher")),
        "bsA/s": _entry(("acc", 1e-6, "higher")),
    }
    out = _score(per_session, ONE_PAIR, retention_floor=1e-3)
    row = _rows(out)[("xsession", "acc")]
    # The per-metric value stays unfloored; only the geometric mean floors it.
    assert row["retention_raw"] == pytest.approx(1e-6)
    assert row["retention"] == pytest.approx(1e-6)
    assert out.summary["generalization_score"] == pytest.approx(1e-3)
    assert out.summary["mean_retention"] == pytest.approx(1e-6)


def test_to_dict_round_trip():
    import json

    per_session = {
        "bsA/t": _entry(("acc", 0.8, "higher")),
        "bsA/s": _entry(("acc", 0.6, "higher")),
    }
    payload = _score(per_session, ONE_PAIR).to_dict()
    json.dumps(payload)  # serializable
    assert payload["baseline_tier"] == "xtime"
    assert payload["ood_tiers"] == ["xsession"]
    assert payload["summary"]["baseline_mode"] == "brainset_matched"
    assert "per_metric" in payload


# -- Brainset-matched baseline ----------------------------------------

# Two brainsets. bsB's error is 10x bsA's in every tier, as for a high-rate dataset.
TWO_BRAINSETS = {
    "bsA/t": _entry(("err", 1.0, "lower")),
    "bsB/t": _entry(("err", 10.0, "lower")),
    "bsA/s": _entry(("err", 2.0, "lower")),
    "bsB/s": _entry(("err", 10.0, "lower")),
}
TWO_BRAINSET_TIERS = {"bsA/t": "xtime", "bsB/t": "xtime", "bsA/s": "xsession", "bsB/s": "xsession"}


def test_matched_compares_each_session_with_its_brainsets_baseline():
    # bsA: 1/2 = 0.5, bsB: 10/10 = 1.0 -> mean 0.75.
    row = _rows(_score(TWO_BRAINSETS, TWO_BRAINSET_TIERS))[("xsession", "err")]
    assert row["retention"] == pytest.approx(0.75)
    assert row["tier_n_matched"] == 2


def test_tier_mean_mode_keeps_the_pooled_formula():
    # Ratio of tier means: 5.5 / 6.0.
    out = _score(TWO_BRAINSETS, TWO_BRAINSET_TIERS, baseline_mode="tier_mean")
    assert _rows(out)[("xsession", "err")]["retention"] == pytest.approx(5.5 / 6.0)
    assert out.summary["baseline_mode"] == "tier_mean"


def test_matched_is_not_moved_by_the_tiers_brainset_mix():
    # A tier without the high-error brainset: the pooled baseline (5.5) makes bsA's
    # doubled error look like an improvement; the matched baseline does not.
    per_session = {k: v for k, v in TWO_BRAINSETS.items() if k != "bsB/s"}
    tiers = {k: v for k, v in TWO_BRAINSET_TIERS.items() if k != "bsB/s"}
    assert _rows(_score(per_session, tiers))[("xsession", "err")]["retention"] == pytest.approx(
        0.5
    )
    pooled = _score(per_session, tiers, baseline_mode="tier_mean")
    assert _rows(pooled)[("xsession", "err")]["retention"] == 1.0


def test_matched_falls_back_to_the_pooled_baseline_without_a_same_brainset_session():
    per_session = {**TWO_BRAINSETS, "bsC/s": _entry(("err", 11.0, "lower"))}
    tiers = {**TWO_BRAINSET_TIERS, "bsC/s": "xsource"}
    row = _rows(_score(per_session, tiers, ood_tiers=("xsource",)))[("xsource", "err")]
    assert row["retention"] == pytest.approx(5.5 / 11.0)
    assert row["tier_n_matched"] == 0


def test_matched_clips_each_session_before_averaging():
    # bsA improves (raw 2.0, clipped to 1), bsB halves (0.5): mean of clipped 0.75,
    # mean of raw 1.25. An improvement on one session cannot offset a loss on another.
    per_session = {
        "bsA/t": _entry(("acc", 0.5, "higher")),
        "bsB/t": _entry(("acc", 1.0, "higher")),
        "bsA/s": _entry(("acc", 1.0, "higher")),
        "bsB/s": _entry(("acc", 0.5, "higher")),
    }
    row = _rows(_score(per_session, TWO_BRAINSET_TIERS))[("xsession", "acc")]
    assert row["retention"] == pytest.approx(0.75)
    assert row["retention_raw"] == pytest.approx(1.25)


def test_matched_averages_several_baseline_sessions_of_a_brainset():
    per_session = {
        "bsA/t1": _entry(("acc", 0.4, "higher")),
        "bsA/t2": _entry(("acc", 0.8, "higher")),
        "bsA/s": _entry(("acc", 0.3, "higher")),
    }
    tiers = {"bsA/t1": "xtime", "bsA/t2": "xtime", "bsA/s": "xsession"}
    assert _rows(_score(per_session, tiers))[("xsession", "acc")]["retention"] == pytest.approx(
        0.5
    )


def test_matched_skips_a_session_whose_baseline_is_zero():
    # bsA's baseline error is exactly 0, so its ratio is undefined: the term is left out
    # rather than scored 0. bsB: 1/2 = 0.5.
    per_session = {
        "bsA/t": _entry(("err", 0.0, "lower")),
        "bsB/t": _entry(("err", 1.0, "lower")),
        "bsA/s": _entry(("err", 0.005, "lower")),
        "bsB/s": _entry(("err", 2.0, "lower")),
    }
    row = _rows(_score(per_session, TWO_BRAINSET_TIERS))[("xsession", "err")]
    assert row["retention"] == pytest.approx(0.5)
    assert row["tier_n_skipped"] == 1
    assert row["tier_n_sessions"] == 2


def test_matched_skips_a_session_whose_baseline_is_negative():
    # A negative baseline correlation would invert the ratio: bsA's held-out -0.5 (worse)
    # would score 1 and +0.3 (better) 0. The session is left out; bsB: 0.4/0.8 = 0.5.
    per_session = {
        "bsA/t": _entry(("corr", -0.1, "higher")),
        "bsB/t": _entry(("corr", 0.8, "higher")),
        "bsA/s": _entry(("corr", -0.5, "higher")),
        "bsB/s": _entry(("corr", 0.4, "higher")),
    }
    row = _rows(_score(per_session, TWO_BRAINSET_TIERS))[("xsession", "corr")]
    assert row["retention"] == pytest.approx(0.5)
    assert row["tier_n_skipped"] == 1


def test_matched_skips_a_session_whose_brainset_baseline_lacks_the_metric():
    # bsA has an xtime session, but without "acc": its held-out session is not compared
    # with the pooled baseline of other brainsets (bsB's 10 would give 1.0).
    per_session = {
        "bsA/t": _entry(("other", 1.0, "higher")),
        "bsB/t": _entry(("acc", 10.0, "lower"), ("other", 1.0, "higher")),
        "bsA/s": _entry(("acc", 2.0, "lower"), ("other", 1.0, "higher")),
        "bsB/s": _entry(("acc", 20.0, "lower"), ("other", 1.0, "higher")),
    }
    row = _rows(_score(per_session, TWO_BRAINSET_TIERS))[("xsession", "acc")]
    assert row["retention"] == pytest.approx(0.5)
    assert row["tier_n_skipped"] == 1
    assert row["tier_n_matched"] == 1


def test_matched_drops_the_term_when_every_baseline_is_zero():
    per_session = {
        "bsA/t": _entry(("err", 0.0, "lower"), ("acc", 0.8, "higher")),
        "bsA/s": _entry(("err", 0.005, "lower"), ("acc", 0.6, "higher")),
    }
    out = _score(per_session, ONE_PAIR)
    assert set(_rows(out)) == {("xsession", "acc")}
    # The tier-mean formula keeps its zero-baseline rule: retention 0.
    pooled = _score(per_session, ONE_PAIR, baseline_mode="tier_mean")
    assert _rows(pooled)[("xsession", "err")]["retention"] == 0.0


def test_unknown_baseline_mode_raises():
    per_session = {
        "bsA/t": _entry(("acc", 0.8, "higher")),
        "bsA/s": _entry(("acc", 0.6, "higher")),
    }
    with pytest.raises(ValueError, match="baseline_mode"):
        _score(per_session, ONE_PAIR, baseline_mode="pooled")


# -- Scored tiers -----------------------------------------------------

THREE_TIERS = {"bsA/t": "xtime", "bsA/s": "xsession", "bsA/u": "xsubject", "bsB/v": "xsource"}
THREE_TIER_SESSIONS = {
    "bsA/t": _entry(("acc", 0.8, "higher")),
    "bsA/s": _entry(("acc", 0.6, "higher")),  # 0.75
    "bsA/u": _entry(("acc", 0.4, "higher")),  # 0.5
    "bsB/v": _entry(("acc", 0.2, "higher")),  # 0.25, vs the pooled xtime mean
}


def test_score_covers_xsource_only_while_rows_cover_every_tier():
    out = compute_from_session_tiers(THREE_TIER_SESSIONS, THREE_TIERS)
    assert out is not None
    assert {r["tier"] for r in out.per_metric} == {"xsession", "xsubject", "xsource"}
    assert set(out.summary["tier_scores"]) == {"xsession", "xsubject", "xsource"}
    assert out.summary["score_tiers"] == ["xsource"]
    assert out.summary["n_retention_terms"] == 1
    assert out.summary["generalization_score"] == pytest.approx(0.25)
    every_tier = compute_from_session_tiers(
        THREE_TIER_SESSIONS, THREE_TIERS, score_tiers=("xsession", "xsubject", "xsource")
    )
    assert every_tier is not None
    assert every_tier.summary["generalization_score"] == pytest.approx(
        (0.75 * 0.5 * 0.25) ** (1 / 3)
    )


def test_score_tiers_outside_ood_tiers_raise():
    with pytest.raises(ValueError, match="score_tiers"):
        compute_from_session_tiers(THREE_TIER_SESSIONS, THREE_TIERS, ood_tiers=("xsession",))


def _ps_entry(
    metric: str, value: float, direction: str = "higher", category: str = "single_unit_statistics"
) -> dict:
    return {
        "results": {metric: {"raw_value": value, "direction": direction, "category": category}}
    }


def test_per_session_reports_basic_retention():
    # bsA/sess_id -> xtime (brainset in xtime_excludes, session not excluded)
    # bsA/sess_ood -> xsession (explicit pair wins over xtime fallback)
    explicit = {("bsA", "sess_ood"): "xsession"}
    xtime_excludes = {"bsA": set()}
    per_session = {
        "bsA/sess_id": _ps_entry("acc", 0.8),
        "bsA/sess_ood": _ps_entry("acc", 0.6),
    }
    result, table = compute_from_per_session_reports(
        per_session,
        explicit_tiers=explicit,
        xtime_excludes=xtime_excludes,
        ood_tiers=("xsession",),
        score_tiers=("xsession",),
    )
    assert result is not None
    # retention(higher) = tier/xtime = 0.6/0.8 = 0.75; single term -> geomean 0.75
    assert result.summary["generalization_score"] == pytest.approx(0.75)
    assert result.summary["n_retention_terms"] == 1
    tiers = {row["tier"] for row in table}
    assert tiers == {"xtime", "xsession"}


def test_per_session_reports_baseline_empty_returns_none():
    # Only an OOD session, no xtime baseline -> not computable.
    explicit = {("bsA", "s1"): "xsession"}
    result, table = compute_from_per_session_reports(
        {"bsA/s1": _ps_entry("acc", 0.6)},
        explicit_tiers=explicit,
        xtime_excludes={},
        ood_tiers=("xsession",),
        score_tiers=("xsession",),
    )
    assert result is None
    assert any(row["tier"] == "xsession" for row in table)


def test_per_session_reports_no_ood_terms_returns_none():
    # Only xtime sessions, no OOD tier -> zero retention terms.
    xtime_excludes = {"bsA": set()}
    per_session = {"bsA/s1": _ps_entry("acc", 0.8), "bsA/s2": _ps_entry("acc", 0.7)}
    result, _ = compute_from_per_session_reports(
        per_session,
        explicit_tiers={},
        xtime_excludes=xtime_excludes,
        ood_tiers=("xsession",),
        score_tiers=("xsession",),
    )
    assert result is None


def test_per_session_reports_matches_tier_reports_math():
    # Same numbers as test_higher_is_better_retention_math -> same retention.
    explicit = {("bsA", "sess_ood"): "xsession"}
    xtime_excludes = {"bsA": set()}
    per_session = {
        "bsA/sess_id": _ps_entry("acc", 0.8),
        "bsA/sess_ood": _ps_entry("acc", 0.6),
    }
    result, _ = compute_from_per_session_reports(
        per_session,
        explicit_tiers=explicit,
        xtime_excludes=xtime_excludes,
        ood_tiers=("xsession",),
        score_tiers=("xsession",),
    )
    rows = {(r["tier"], r["metric"]): r for r in result.per_metric}
    assert rows[("xsession", "acc")]["retention"] == pytest.approx(0.75)


def test_per_session_reports_passes_the_baseline_mode_through():
    explicit = {("bsA", "s"): "xsession", ("bsB", "s"): "xsession"}
    xtime_excludes = {"bsA": {"s"}, "bsB": {"s"}}
    kwargs = {
        "explicit_tiers": explicit,
        "xtime_excludes": xtime_excludes,
        "ood_tiers": ("xsession",),
        "score_tiers": ("xsession",),
    }
    matched, _ = compute_from_per_session_reports(TWO_BRAINSETS, **kwargs)
    pooled, _ = compute_from_per_session_reports(
        TWO_BRAINSETS, baseline_mode="tier_mean", **kwargs
    )
    assert matched is not None and pooled is not None
    assert matched.summary["generalization_score"] == pytest.approx(0.75)
    assert pooled.summary["generalization_score"] == pytest.approx(5.5 / 6.0)
