"""Tests for ntt.evaluation.radar."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402

import pytest
from matplotlib.figure import Figure

from ntt.evaluation.core.types import (
    EvaluationReport,
    MetricCategory,
    MetricDirection,
    MetricLevel,
    MetricResult,
)
from ntt.evaluation.radar import (
    DEFAULT_EVAL_JOB_SCALARS,
    CategorySpec,
    RadarBundle,
    collect_category_scores,
    inject_scalar_metrics,
    plot_radar,
    tokenizers_with_data,
)


def _mr(
    name: str,
    value: float,
    *,
    category=MetricCategory.SINGLE_UNIT_STATISTICS,
    direction=MetricDirection.HIGHER_IS_BETTER,
):
    return MetricResult(
        name=name,
        category=category,
        direction=direction,
        level=MetricLevel.POPULATION,
        raw_value=value,
    )


def _report(*pairs):
    return EvaluationReport(results={r.name: r for r in pairs})


def _write_results_json(
    path: Path,
    *,
    aggregated_acc: float,
    compression_rate: float | None = None,
    gen_score: float | None = None,
):
    payload = {
        "aggregated": _report(_mr("acc", aggregated_acc)).to_dict(),
        "poisson_baseline": _report(_mr("acc", 0.1)).to_dict(),
        "perfect_ceiling": _report(_mr("acc", 1.0)).to_dict(),
    }
    if compression_rate is not None:
        payload["compression"] = {
            "compression_rate": compression_rate,
            "latent_dim_reduction": compression_rate,
        }
    if gen_score is not None:
        payload["generalization"] = {
            "baseline_tier": "xtime",
            "ood_tiers": ["xsession"],
            "summary": {
                "generalization_score": gen_score,
                "mean_retention": gen_score,
                "n_retention_terms": 1,
                "retention_floor": 0.001,
                "tier_scores": {},
            },
            "per_metric": [],
        }
    path.write_text(json.dumps(payload))


@pytest.fixture
def bundle_root(tmp_path):
    root = tmp_path / "runs"
    for name, acc, c, g in [
        ("tok_a", 0.7, 20.0, 0.8),
        ("tok_b", 0.5, 5.0, 0.5),
    ]:
        d = root / name
        d.mkdir(parents=True)
        _write_results_json(
            d / "results.json",
            aggregated_acc=acc,
            compression_rate=c,
            gen_score=g,
        )
    return root


def test_radar_bundle_from_local_dir_loads_two_tokenizers(bundle_root):
    bundle = RadarBundle.from_local_dir(bundle_root)
    assert set(bundle.tokenizers) == {"tok_a", "tok_b"}
    assert bundle.baseline is not None
    assert bundle.ceiling is not None
    assert "acc" in bundle.tokenizers["tok_a"].results


def test_radar_bundle_filters_by_tokenizer_subset(bundle_root):
    bundle = RadarBundle.from_local_dir(bundle_root, tokenizers=["tok_a"])
    assert set(bundle.tokenizers) == {"tok_a"}


def test_inject_scalar_metrics_attaches_to_each_report(bundle_root):
    bundle = RadarBundle.from_local_dir(bundle_root)
    raw = {
        n: json.loads((bundle.paths[n] / "results.json").read_text()) for n in bundle.tokenizers
    }
    inject_scalar_metrics(bundle, raw, DEFAULT_EVAL_JOB_SCALARS)
    for name in bundle.tokenizers:
        assert "compression/rate" in bundle.tokenizers[name].results
        assert "generalization/score" in bundle.tokenizers[name].results
    assert "compression/rate" in bundle.baseline.results
    assert "compression/rate" in bundle.ceiling.results


def test_inject_scalar_metrics_skips_missing_paths(bundle_root):
    payload_b = json.loads((bundle_root / "tok_b" / "results.json").read_text())
    payload_b.pop("generalization", None)
    (bundle_root / "tok_b" / "results.json").write_text(json.dumps(payload_b))

    bundle = RadarBundle.from_local_dir(bundle_root)
    raw = {
        n: json.loads((bundle.paths[n] / "results.json").read_text()) for n in bundle.tokenizers
    }
    inject_scalar_metrics(bundle, raw, DEFAULT_EVAL_JOB_SCALARS)
    assert "generalization/score" in bundle.tokenizers["tok_a"].results
    assert "generalization/score" not in bundle.tokenizers["tok_b"].results


def test_collect_category_scores_returns_per_tokenizer_dataframe(bundle_root):
    bundle = RadarBundle.from_local_dir(bundle_root)
    spec = CategorySpec({"All": ["acc"]})
    scores = collect_category_scores(bundle, spec, normalize=True)
    assert set(scores.index) == {"tok_a", "tok_b"}
    assert list(scores.columns) == ["All"]


def test_plot_radar_returns_figure(bundle_root):
    bundle = RadarBundle.from_local_dir(bundle_root)
    spec = CategorySpec({"All": ["acc"]})
    scores = collect_category_scores(bundle, spec, normalize=True)
    fig = plot_radar(scores, title="smoke", mode="normed")
    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 1  # at least one polar axis


def test_tokenizers_with_data_filters_by_required_metrics(bundle_root):
    bundle = RadarBundle.from_local_dir(bundle_root)
    raw = {
        n: json.loads((bundle.paths[n] / "results.json").read_text()) for n in bundle.tokenizers
    }
    inject_scalar_metrics(bundle, raw, DEFAULT_EVAL_JOB_SCALARS)
    toks = tokenizers_with_data(
        bundle,
        ["compression/rate", "generalization/score"],
        require="all",
    )
    assert set(toks) == {"tok_a", "tok_b"}
