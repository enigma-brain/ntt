"""Tests for scripts/analysis/build_metrics_table.py."""

from __future__ import annotations

import statistics

import pytest
from scripts.analysis.build_metrics_table import flatten_tokenizer_metrics, ridge_normalized_score

from ntt.evaluation.aggregation.downstream_records import DownstreamScoreView


def _recon():
    return {
        "aggregated": {
            "results": {
                "acg_correlation": {"category": "single_unit_statistics", "raw_value": 0.9},
                "ccg_correlation": {"category": "population_structure", "raw_value": 0.4},
            }
        },
        "per_session": {
            "bsA/s1": {
                "results": {
                    "acg_correlation": {"raw_value": 0.8},
                    "ccg_correlation": {"raw_value": 0.3},
                }
            },
            "bsA/s2": {
                "results": {
                    "acg_correlation": {"raw_value": 1.0},
                    "ccg_correlation": {"raw_value": 0.5},
                }
            },
        },
        "comparison_vs_poisson": {
            "acg_correlation": {"status": "ok", "normalized_score": 0.7},
            # ccg has no valid norm -> no ::norm column
        },
    }


def test_recon_mean_std_n_and_norm():
    out = flatten_tokenizer_metrics(_recon(), None, None, None)
    # mean over [0.8, 1.0] = 0.9; sample std (ddof=1) = 0.1414...; n=2
    assert out["single_unit_statistics/acg_correlation"] == pytest.approx(0.9)
    assert out["single_unit_statistics/acg_correlation::std"] == pytest.approx(
        0.1414213562, abs=1e-6
    )
    assert out["single_unit_statistics/acg_correlation::n"] == 2
    assert out["single_unit_statistics/acg_correlation::norm"] == pytest.approx(0.7)
    # ccg present, no norm column
    assert out["population_structure/ccg_correlation"] == pytest.approx(0.4)
    assert "population_structure/ccg_correlation::norm" not in out


_EXCLUDED = "perich_miller_population_2018/c_20131029_random_target_reaching"
_RIDGE = "behavioral_regression_cursor_vel"


def _ridge_recon() -> dict:
    """Ridge numbers from a real eval results.json (a calm_transformer model);
    ``normalized_score`` is what eval stored, over all 3 sessions."""
    model = {
        _EXCLUDED: -0.14594649411331564,
        "perich_miller_population_2018/c_20131220_center_out_reaching": 0.24856350638640912,
        "perich_miller_population_2018/j_20160407_center_out_reaching": 0.19807035836652312,
    }
    poisson = {
        _EXCLUDED: 0.2339960957175762,
        "perich_miller_population_2018/c_20131220_center_out_reaching": 0.33052711236114823,
        "perich_miller_population_2018/j_20160407_center_out_reaching": 0.293698300944339,
    }
    return {
        "aggregation_strategy": "mean",
        "aggregated": {
            "results": {
                _RIDGE: {
                    "name": "behavioral_regression",
                    "category": "downstream_decoding",
                    "raw_value": 0.10022912354653886,
                }
            }
        },
        "per_session": {s: {"results": {_RIDGE: {"raw_value": v}}} for s, v in model.items()},
        "poisson_per_session_per_segment": {
            s: [{"results": {_RIDGE: {"raw_value": v}}}] for s, v in poisson.items()
        },
        "comparison_vs_poisson": {
            _RIDGE: {
                "status": "ok",
                "direction": "higher",
                "ceiling_value": 1.0,
                "value_range": [None, 1.0],
                "normalized_score": -0.26031363221372955,
            }
        },
    }


def test_ridge_norm_over_all_sessions_matches_eval():
    """With no session left out, the recomputed norm is eval's stored value."""
    recon = _ridge_recon()
    stored = recon["comparison_vs_poisson"][_RIDGE]["normalized_score"]
    assert ridge_normalized_score(_RIDGE, recon, excluded=frozenset()) == pytest.approx(stored)


def test_ridge_columns_leave_out_excluded_session():
    """The wide table's ridge columns (value, std, n, norm) use only the sessions the decoding
    score uses, on both the model and the Poisson side."""
    recon = _ridge_recon()
    out = flatten_tokenizer_metrics(recon, None, None, None)
    base = f"downstream_decoding/{_RIDGE}"
    model = [0.24856350638640912, 0.19807035836652312]
    assert out[base] == pytest.approx(statistics.fmean(model))
    assert out[f"{base}::n"] == 2
    assert out[f"{base}::std"] == pytest.approx(statistics.stdev(model))
    # (0.22332 - 0.31211) / (1 - 0.31211): both means over the two kept sessions
    assert out[f"{base}::norm"] == pytest.approx(-0.12908477178608382)


def test_robustness_uses_metadata_std():
    m1 = {
        "results": {
            "robustness_input/drop_neurons|fraction=0.1": {
                "raw_value": 0.25,
                "metadata": {"std": 0.05, "n_sessions": 20},
            },
        }
    }
    out = flatten_tokenizer_metrics(None, m1, None, None)
    assert out["robustness/robustness_input/drop_neurons|fraction=0.1"] == pytest.approx(0.25)
    assert out["robustness/robustness_input/drop_neurons|fraction=0.1::std"] == pytest.approx(0.05)
    assert out["robustness/robustness_input/drop_neurons|fraction=0.1::n"] == 20


def test_m2_latent_noise_surfaces_under_its_own_prefix():
    """M2 latent-noise metrics reach the wide table, namespaced away from M1."""
    m2 = {
        "results": {
            "acg_correlation/eps_0.1": {
                "raw_value": 0.72,
                "metadata": {"std": 0.03, "n_sessions": 12, "epsilon": 0.1},
            },
        },
    }
    out = flatten_tokenizer_metrics(None, None, None, None, m2=m2)
    assert out["robustness_latent/acg_correlation/eps_0.1"] == pytest.approx(0.72)
    assert out["robustness_latent/acg_correlation/eps_0.1::std"] == pytest.approx(0.03)
    assert out["robustness_latent/acg_correlation/eps_0.1::n"] == 12


def test_m1_and_m2_do_not_collide():
    same_key = "acg_correlation/eps_0.1"
    m1 = {"results": {same_key: {"raw_value": 0.11, "metadata": {}}}}
    m2 = {
        "results": {same_key: {"raw_value": 0.99, "metadata": {}}},
    }
    out = flatten_tokenizer_metrics(None, m1, None, None, m2=m2)
    assert out[f"robustness/{same_key}"] == pytest.approx(0.11)
    assert out[f"robustness_latent/{same_key}"] == pytest.approx(0.99)


def _view(aggregated, per_session, split="test"):
    return DownstreamScoreView(
        split=split,
        aggregated=aggregated,
        per_session=per_session,
        complete=True,
        provenance={},
    )


def test_encoding_cross_session_std_from_per_session_means():
    enc = _view(
        {"nc_r2": {"value": 0.3, "std": 0.04, "n": 2}},
        {"PE_1": {"nc_r2": 0.2}, "PE_2": {"nc_r2": 0.4}},
    )
    out = flatten_tokenizer_metrics(None, None, enc, None)
    assert out["encoding/split"] == "test"
    assert out["encoding/nc_r2"] == pytest.approx(0.3)
    assert out["encoding/nc_r2::std"] == pytest.approx(0.1414213562, abs=1e-6)  # std([0.2,0.4])
    assert out["encoding/nc_r2::n"] == 2


def test_decoding_records_whether_its_cohort_is_complete():
    """The wide table keeps an incomplete cohort's numbers, so it must say they are
    partial (the decoding score refuses them)."""
    partial = DownstreamScoreView(
        split="test",
        aggregated={},
        per_session={"s1": {"r2": 0.8}},
        complete=False,
        provenance={"missing": ["s2"]},
    )
    assert flatten_tokenizer_metrics(None, None, None, partial)["decoding/complete"] is False
    assert flatten_tokenizer_metrics(None, None, None, _view({}, {"s1": {"r2": 0.8}}))[
        "decoding/complete"
    ]


def test_decoding_mean_std_n():
    dec = _view({}, {"s1": {"r2": 0.6}, "s2": {"r2": 0.8}, "s3": {"r2": None}})
    out = flatten_tokenizer_metrics(None, None, None, dec)
    assert out["decoding/test_r2"] == pytest.approx(0.7)
    assert out["decoding/test_r2::std"] == pytest.approx(0.1414213562, abs=1e-6)
    assert out["decoding/test_r2::n"] == 2


def test_missing_blocks_produce_no_columns():
    out = flatten_tokenizer_metrics(None, None, None, None)
    assert out == {}


def test_single_session_std_is_none():
    recon = {
        "aggregated": {
            "results": {
                "acg_correlation": {"category": "single_unit_statistics", "raw_value": 0.8}
            }
        },
        "per_session": {"bsA/s1": {"results": {"acg_correlation": {"raw_value": 0.8}}}},
        "comparison_vs_poisson": {},
    }
    out = flatten_tokenizer_metrics(recon, None, None, None)
    assert out["single_unit_statistics/acg_correlation"] == pytest.approx(0.8)
    assert out["single_unit_statistics/acg_correlation::std"] is None
    assert out["single_unit_statistics/acg_correlation::n"] == 1


def test_repeats_surface_in_wide_table():
    recon = {
        "aggregated": {
            "results": {
                "repeats_snr": {"category": "latent_quality", "raw_value": 0.7},
                "repeats_baseline": {"category": "latent_quality", "raw_value": 0.9},
            }
        },
        "per_session": {},
        "comparison_vs_poisson": {},
    }
    row = flatten_tokenizer_metrics(recon, None, None, None)
    assert row["latent_quality/repeats_snr"] == 0.7
    assert row["latent_quality/repeats_baseline"] == 0.9


def test_build_table_end_to_end_local(tmp_path):
    import json

    from scripts.analysis.build_metrics_table import build_table

    recon = {
        "aggregated": {
            "results": {
                "acg_correlation": {"category": "single_unit_statistics", "raw_value": 0.9}
            }
        },
        "per_session": {
            "bsA/s1": {"results": {"acg_correlation": {"raw_value": 0.8}}},
            "bsA/s2": {"results": {"acg_correlation": {"raw_value": 1.0}}},
        },
        "comparison_vs_poisson": {"acg_correlation": {"status": "ok", "normalized_score": 0.7}},
    }
    rf = tmp_path / "results.json"
    rf.write_text(json.dumps(recon))
    sheet = tmp_path / "tracker.csv"
    sheet.write_text(f"tokenizer,mlflow_recon\nwanx,{rf}\n")

    df = build_table(str(sheet))
    assert list(df["tokenizer"]) == ["wanx"]
    row = df.set_index("tokenizer").loc["wanx"]
    assert row["single_unit_statistics/acg_correlation"] == 0.9
    assert row["single_unit_statistics/acg_correlation::n"] == 2
    assert row["single_unit_statistics/acg_correlation::norm"] == 0.7
