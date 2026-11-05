"""build_generalization_tables: reconstruction metrics only, tiers over shared metrics,
parity of the composite with the aggregation job, score check."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest
import yaml
from scripts.analysis import build_generalization_tables as bgt


def _selection_yamls(config_root: Path) -> None:
    """bsA/sess_ood = xsession, bsA/sess_sub = xsubject, bsB/sess_src = xsource (val and
    test lists); bsA/sess_id = xtime."""
    root = config_root / "data" / "selections" / "composite"
    root.mkdir(parents=True, exist_ok=True)
    tiers = {
        "xsession": [{"brainset": "bsA", "sessions": ["sess_ood"]}],
        "xsubject": [{"brainset": "bsA", "sessions": ["sess_sub"]}],
        "xsource": [{"brainset": "bsB", "sessions": ["sess_src"]}],
    }
    for split in ("val", "test"):
        for tier, sel in tiers.items():
            (root / f"ntt_full_{split}_{tier}.yaml").write_text(yaml.safe_dump({"selection": sel}))
    xtime = [
        {"brainset": "bsA", "exclude_sessions": ["sess_ood", "sess_sub"]},
        {"brainset": "bsB", "exclude_sessions": ["sess_src"]},
    ]
    (root / "ntt_full_val_xtime.yaml").write_text(yaml.safe_dump({"selection": xtime}))


def _results(**metrics: tuple[float, str]) -> dict:
    return {
        "results": {
            name: {"raw_value": v, "direction": "higher", "category": cat}
            for name, (v, cat) in metrics.items()
        }
    }


SU, LQ, DD = "single_unit_statistics", "latent_quality", "downstream_decoding"


def _recon(ood: bool = True) -> dict:
    """acc (recon, all tiers): retention .75 / .5 / .25; only_ses (recon, xsession only): .5;
    rsa (latent_quality) and dec (downstream_decoding): out of the tier view."""
    per_session = {
        "bsA/sess_id": _results(acc=(0.8, SU), only_ses=(0.9, SU), rsa=(0.5, LQ), dec=(0.6, DD)),
    }
    if ood:
        per_session["bsA/sess_ood"] = _results(
            acc=(0.6, SU), only_ses=(0.45, SU), rsa=(0.1, LQ), dec=(0.3, DD)
        )
        per_session["bsA/sess_sub"] = _results(acc=(0.4, SU), rsa=(0.25, LQ), dec=(0.3, DD))
        per_session["bsB/sess_src"] = _results(acc=(0.2, SU), rsa=(0.05, LQ))
    return {"per_session": per_session}


@pytest.fixture
def config_root(tmp_path: Path) -> Path:
    root = tmp_path / "config"
    _selection_yamls(root)
    return root


def _tiers(config_root: Path):
    from ntt.evaluation.generalization import build_tier_index

    return build_tier_index(config_root, split="test", selection_family="ntt_full")


def _rows(config_root: Path, recon: dict):
    explicit, xtime = _tiers(config_root)
    return bgt.tokenizer_rows(
        "T", recon, explicit_tiers=explicit, xtime_excludes=xtime, baseline_tier="xtime"
    )


def test_tiers_use_reconstruction_metrics_shared_by_all_tiers(config_root):
    tier, _ = _rows(config_root, _recon())
    # only `acc` is a reconstruction metric with a retention in all three tiers
    assert tier["gen_tier/xsession"] == pytest.approx(0.75)
    assert tier["gen_tier/xsubject"] == pytest.approx(0.5)
    assert tier["gen_tier/xsource"] == pytest.approx(0.25)
    for t in ("xsession", "xsubject", "xsource"):
        assert tier[f"gen_tier/{t}::n"] == 1
        assert tier[f"gen_tier/{t}::mean"] == pytest.approx(tier[f"gen_tier/{t}"])


def test_per_metric_rows_are_reconstruction_metrics_only(config_root):
    _, metrics = _rows(config_root, _recon())
    by_name = {m["metric"]: m for m in metrics}
    assert set(by_name) == {"acc", "only_ses"}  # no latent_quality, no downstream_*
    assert by_name["only_ses"]["ret_xsession"] == pytest.approx(0.5)
    assert "ret_xsubject" not in by_name["only_ses"]  # NaN once framed
    assert by_name["acc"]["category"] == SU


def test_composite_score_equals_the_aggregation_jobs_backfill(config_root, tmp_path):
    from ntt.jobs.aggregation import NTTAggregationJob

    recon = _recon()
    job = NTTAggregationJob(
        recon_path=str(tmp_path / "unused.json"),
        output_path=str(tmp_path / "out"),
        gen_config_root=str(config_root),
        gen_split="test",
    )
    backfilled = json.loads(json.dumps(recon))
    assert job._backfill_generalization(backfilled)
    tier, _ = _rows(config_root, recon)
    shipped = backfilled["generalization"]["summary"]["generalization_score"]
    assert tier["generalization_score"] == shipped
    # the composite is scored over xsource and the reconstruction terms only: acc
    # (0.25); rsa (latent_quality) and dec (downstream_decoding) are out
    assert shipped == pytest.approx(0.25)


def test_report_without_generalization_gives_an_empty_row(config_root):
    tier, metrics = _rows(config_root, _recon(ood=False))
    assert tier == {"tokenizer": "T"} and metrics == []


def test_check_scores_names_mismatches_and_gaps():
    by_tier = pd.DataFrame(
        [
            {"tokenizer": "A", "generalization_score": 0.5},
            {"tokenizer": "B", "generalization_score": 0.4},
            {"tokenizer": "C", "generalization_score": float("nan")},
        ]
    )
    shipped = pd.DataFrame(
        {"tokenizer": ["A", "B", "C", "D"], "generalization": [0.5, 0.41, 0.3, 0.2]}
    )
    assert bgt.check_scores(by_tier, shipped) == ["B", "C", "D"]
    only_a = bgt.check_scores(by_tier[by_tier.tokenizer == "A"], shipped[shipped.tokenizer == "A"])
    assert only_a == []


def _write_inputs(tmp_path: Path, names: list[str]) -> Path:
    rows = []
    for name in names:
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps(_recon()))
        rows.append({"tokenizer": name, "results_json": str(p)})
    out = tmp_path / "inputs.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def test_main_writes_both_tables_and_checks(config_root, tmp_path):
    inputs = _write_inputs(tmp_path, ["A", "B"])
    out = tmp_path / "out"
    argv = [
        "--inputs",
        str(inputs),
        "--outdir",
        str(out),
        "--gen-split",
        "test",
        "--config-root",
        str(config_root),
    ]
    assert bgt.main(argv) == 0
    tiers = pd.read_csv(out / "generalization_by_tier.csv")
    assert list(tiers.columns) == bgt.TIER_COLUMNS and len(tiers) == 2
    metrics = pd.read_csv(out / "generalization_by_metric.csv")
    assert list(metrics.columns) == bgt.METRIC_COLUMNS and len(metrics) == 4
    good = tmp_path / "good.csv"
    score = float(tiers["generalization_score"].iloc[0])
    pd.DataFrame({"tokenizer": ["A", "B"], "generalization": [score] * 2}).to_csv(
        good, index=False
    )
    assert bgt.main([*argv, "--check-scores", str(good)]) == 0
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"tokenizer": ["A", "B"], "generalization": [0.1, 0.1]}).to_csv(bad, index=False)
    assert bgt.main([*argv, "--check-scores", str(bad)]) == 1
    assert not math.isnan(score)


def test_duplicate_tokenizers_are_refused(config_root, tmp_path):
    inputs = _write_inputs(tmp_path, ["A", "A"])
    argv = ["--inputs", str(inputs), "--outdir", str(tmp_path / "o")]
    assert bgt.main([*argv, "--config-root", str(config_root)]) == 2
