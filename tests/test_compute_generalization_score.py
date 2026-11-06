import json
from pathlib import Path

import pytest

from ntt.evaluation.generalization import (
    DEFAULT_RETENTION_FLOOR,
    aggregate_metric_values,
    build_tier_index,
    compute_generalization,
    load_observations,
)


def _write_report(path: Path, *, high: float, low: float) -> None:
    path.write_text(
        json.dumps(
            {
                "results": {
                    "higher_metric": {
                        "raw_value": high,
                        "direction": "higher",
                        "category": "temporal_fidelity",
                    },
                    "lower_metric": {
                        "raw_value": low,
                        "direction": "lower",
                        "category": "single_unit_statistics",
                    },
                    "token_dims_per_second": {
                        "raw_value": 100.0,
                        "direction": "lower",
                        "category": "compression",
                    },
                    # Not reconstruction metrics: never observed.
                    "population_structure_preservation": {
                        "raw_value": high,
                        "direction": "higher",
                        "category": "latent_quality",
                    },
                    "behavioral_regression_cursor_vel": {
                        "raw_value": high,
                        "direction": "higher",
                        "category": "downstream_decoding",
                    },
                }
            }
        )
    )


def _write_config(root: Path) -> None:
    composite = root / "data" / "selections" / "composite"
    composite.mkdir(parents=True)
    (composite / "ntt_full_val_xtime.yaml").write_text(
        """
selection:
  - brainset: brainset_a
    exclude_sessions:
      - xsession_1
      - xsubject_1
      - xsource_1
"""
    )
    (composite / "ntt_full_val_xsession.yaml").write_text(
        """
selection:
  - brainset: brainset_a
    sessions:
      - xsession_1
"""
    )
    (composite / "ntt_full_val_xsubject.yaml").write_text(
        """
selection:
  - brainset: brainset_a
    sessions:
      - xsubject_1
"""
    )
    (composite / "ntt_full_val_xsource.yaml").write_text(
        """
selection:
  - brainset: brainset_a
    sessions:
      - xsource_1
"""
    )


def test_generalization_retention_from_per_session_reports(tmp_path):
    config_root = tmp_path / "config"
    report_root = tmp_path / "per_session"
    report_root.mkdir()
    _write_config(config_root)

    _write_report(report_root / "brainset_a__xtime_1.json", high=0.8, low=2.0)
    _write_report(report_root / "brainset_a__xsession_1.json", high=0.4, low=4.0)
    _write_report(report_root / "brainset_a__xsubject_1.json", high=1.0, low=1.0)
    _write_report(report_root / "brainset_a__xsource_1.json", high=0.2, low=8.0)

    explicit, xtime = build_tier_index(
        config_root,
        split="val",
        selection_family="ntt_full",
    )
    observations, sessions = load_observations(
        report_root,
        explicit,
        xtime,
        excluded_brainsets=set(),
        excluded_metrics={"token_dims_per_second"},
    )
    metric_rows, lookup = aggregate_metric_values(observations)
    retention_rows, summary = compute_generalization(lookup, baseline_tier="xtime")

    assert {row["tier"] for row in sessions} == {"xtime", "xsession", "xsubject", "xsource"}
    assert {row["metric"] for row in metric_rows} == {"higher_metric", "lower_metric"}

    by_metric_tier = {(row["metric"], row["tier"]): row for row in retention_rows}
    assert by_metric_tier[("higher_metric", "xsession")]["retention"] == 0.5
    assert by_metric_tier[("lower_metric", "xsession")]["retention"] == 0.5

    # Improvements over xtime are clipped to 1.0.
    assert by_metric_tier[("higher_metric", "xsubject")]["retention_raw"] == 1.25
    assert by_metric_tier[("higher_metric", "xsubject")]["retention"] == 1.0
    assert by_metric_tier[("lower_metric", "xsubject")]["retention_raw"] == 2.0
    assert by_metric_tier[("lower_metric", "xsubject")]["retention"] == 1.0

    assert summary["tier_scores"]["xsession"]["n_metrics"] == 2
    # rows and tier scores cover every OOD tier; the score only xsource
    assert summary["score_tiers"] == ["xsource"]
    assert summary["n_retention_terms"] == 2
    assert summary["retention_floor"] == DEFAULT_RETENTION_FLOOR
    assert 0.0 <= summary["generalization_score"] <= 1.0


def test_geometric_score_uses_floor_without_changing_retention_rows():
    lookup = {
        ("xtime", "higher_metric"): {
            "direction": "higher",
            "category": "temporal_fidelity",
            "value": 1.0,
            "n_sessions": 1,
        },
        ("xsession", "higher_metric"): {
            "direction": "higher",
            "category": "temporal_fidelity",
            "value": 0.0,
            "n_sessions": 1,
        },
    }

    retention_rows, summary = compute_generalization(
        lookup,
        baseline_tier="xtime",
        ood_tiers=("xsession",),
        score_tiers=("xsession",),
        retention_floor=1e-3,
    )

    assert retention_rows[0]["retention"] == 0.0
    assert summary["tier_scores"]["xsession"]["geometric_mean_retention"] == pytest.approx(1e-3)
    assert summary["generalization_score"] == pytest.approx(1e-3)


def test_excluded_brainset_is_not_scored(tmp_path):
    config_root = tmp_path / "config"
    report_root = tmp_path / "per_session"
    report_root.mkdir()
    _write_config(config_root)

    _write_report(report_root / "brainset_a__xtime_1.json", high=0.8, low=2.0)
    _write_report(report_root / "brainset_a__xsession_1.json", high=0.4, low=4.0)
    tostado_name = "tostado_marcos_birdsong_2024__finch_z_r12r13_21_20210626_held_in_calib.json"
    _write_report(
        report_root / tostado_name,
        high=1.0,
        low=1.0,
    )

    explicit, xtime = build_tier_index(
        config_root,
        split="val",
        selection_family="ntt_full",
    )
    observations, sessions = load_observations(
        report_root,
        explicit,
        xtime,
        excluded_brainsets={"tostado_marcos_birdsong_2024"},
        excluded_metrics=set(),
    )

    assert any(
        session["brainset"] == "tostado_marcos_birdsong_2024" and session["tier"] == "excluded"
        for session in sessions
    )
    assert all(
        not obs.session_key.startswith("tostado_marcos_birdsong_2024__") for obs in observations
    )


def test_unquoted_numeric_session_id_lands_in_xsession_not_xtime(tmp_path):
    """YAML reads an unquoted 1234567890123 as an int; results.json carries it as a string.

    A held-out session with such an id must be classified xsession, not fall through to
    the xtime baseline.
    """
    from ntt.evaluation.generalization import SessionId, classify_session

    config_root = tmp_path / "config"
    _write_config(config_root)
    composite = config_root / "data/selections/composite"
    (composite / "ntt_full_val_xsession.yaml").write_text(
        "selection:\n  - brainset: brainset_b\n    sessions:\n      - 1234567890123\n"
    )
    (composite / "ntt_full_val_xtime.yaml").write_text(
        "selection:\n  - brainset: brainset_b\n    exclude_sessions:\n      - 1234567890123\n"
    )
    explicit, xtime = build_tier_index(config_root, split="val", selection_family="ntt_full")
    session = SessionId(brainset="brainset_b", session="1234567890123")
    assert classify_session(session, explicit, xtime) == "xsession"


def test_session_in_an_ood_tier_and_in_xtime_warns(tmp_path, caplog):
    from ntt.evaluation.generalization import SessionId, classify_session

    config_root = tmp_path / "config"
    _write_config(config_root)
    # xsession_1 is listed in xsession but (wrongly) not excluded from xtime.
    xtime_yaml = config_root / "data/selections/composite/ntt_full_val_xtime.yaml"
    xtime_yaml.write_text(
        "selection:\n  - brainset: brainset_a\n    exclude_sessions:\n      - xsubject_1\n"
    )
    explicit, xtime = build_tier_index(config_root, split="val", selection_family="ntt_full")
    session = SessionId(brainset="brainset_a", session="xsession_1")
    with caplog.at_level("WARNING"):
        assert classify_session(session, explicit, xtime) == "xsession"
    assert "listed in the 'xsession' tier but not excluded from xtime" in caplog.text
