"""Tests for scripts/analysis/data.py and the radar adapter in plot_radar.py.

Focus: the consolidation onto ``ntt.evaluation.aggregation.tracker_io`` (run-URL
parsing, artifact loading, Sheets resolution) and the shared family palette, plus
the pure ``scores_to_radar_frame`` adapter that feeds ``radar.plot_radar``.
"""

from __future__ import annotations

import json
import math

import pandas as pd
import pytest
from scripts.analysis import data
from scripts.analysis.data import (
    _SCORE_ALIASES,
    COMPOSITE_SCORES,
    extract_metric,
    family_color,
    infer_family,
    load_scores,
)
from scripts.analysis.plot_radar import scores_to_radar_frame

# ── composite-score column rename ─────────────────────────────────────────────


def test_mlflow_cache_tracks_run_identity(tmp_path, monkeypatch):
    calls = []

    def load(uri):
        calls.append(uri)
        return {"scores": {"decoding": 0.1 if "/abc/" in uri else 0.9}}

    monkeypatch.setattr(data.tracker_io, "load_json_artifact", load)
    old_url = "http://host/#/experiments/1/runs/abc"
    new_url = "http://host/#/experiments/1/runs/def"
    assert data._load_from_mlflow("tok", old_url, tmp_path, False)["scores"]["decoding"] == 0.1
    assert data._load_from_mlflow("tok", new_url, tmp_path, False)["scores"]["decoding"] == 0.9
    assert data._load_from_mlflow("tok", new_url, tmp_path, False)["scores"]["decoding"] == 0.9
    assert len(calls) == 2


def test_composite_score_names():
    assert "population_structure" in COMPOSITE_SCORES
    assert "latent_structure_and_robustness" in COMPOSITE_SCORES
    assert "decoding" in COMPOSITE_SCORES
    assert "population_stats" not in COMPOSITE_SCORES
    assert "latent_rob" not in _SCORE_ALIASES


# ── extract_metric ───────────────────────────────────────────────────────────


def _scores_doc():
    return {
        "scores": {
            "single_neuron_recon": 0.42,
            "population_structure": -0.1,
            "compression": float("inf"),  # non-finite -> None
        },
        "diagnostics": {
            "single_neuron_recon": {
                "per_metric": {"spike_f1": {"normalized": 0.8}},
            },
        },
    }


def test_extract_metric_composite_and_alias():
    doc = _scores_doc()
    assert extract_metric(doc, "single_neuron_recon") == pytest.approx(0.42)
    assert extract_metric(doc, "sn_recon") == pytest.approx(0.42)  # alias resolves


def test_extract_metric_per_metric_path():
    assert extract_metric(_scores_doc(), "sn_recon/spike_f1") == pytest.approx(0.8)


def test_extract_metric_missing_and_non_finite_return_none():
    doc = _scores_doc()
    assert extract_metric(doc, "latent_structure_and_robustness") is None  # absent key
    assert extract_metric(doc, "compression") is None  # inf -> None
    assert extract_metric(doc, "sn_recon/does_not_exist") is None


# ── infer_family ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name,family",
    [
        ("wanx_A_vae_bce_dw_k1_noncausal", "wanx_A_vae"),
        ("cosmos_tokenizer_DV4x8x8", "cosmos"),
        ("LDNS_vanilla", "LDNS"),
        ("some_unknown_thing", "some_unknown_thing"),  # falls through to itself
    ],
)
def test_infer_family(name, family):
    assert infer_family(name) == family


# ── family_color ─────────────────────────────────────────────────────────────


def test_family_color_known_is_stable():
    seen: dict[str, str] = {}
    assert family_color("cosmos", seen) == data._FAMILY_COLORS["cosmos"]
    # Known families do not consume fallback slots.
    assert seen == {}


def test_family_color_unknown_assigns_and_reuses_fallback():
    seen: dict[str, str] = {}
    first = family_color("mystery", seen)
    assert seen["mystery"] == first
    # Same family -> same colour on a second call.
    assert family_color("mystery", seen) == first
    # A different unknown family gets a different fallback slot.
    assert family_color("other", seen) != first


# ── scores_to_radar_frame (radar adapter) ─────────────────────────────────────


def _scores_df(rows):
    """Build a load_scores-shaped frame (tokenizer + the 7 composite columns)."""
    records = []
    for name, vals in rows.items():
        rec = {"tokenizer": name}
        for col in COMPOSITE_SCORES:
            rec[col] = vals.get(col, float("nan"))
        records.append(rec)
    return pd.DataFrame(records)


def test_scores_to_radar_frame_shape_order_and_clip():
    df = _scores_df(
        {
            "tok_a": dict.fromkeys(COMPOSITE_SCORES, 0.5),
            "tok_b": dict.fromkeys(COMPOSITE_SCORES, 2.0),  # above clip
        }
    )
    frame = scores_to_radar_frame(df, ["tok_a", "tok_b"], score_clip=1.0)
    # one row per tokenizer, seven category columns
    assert frame.shape == (2, len(COMPOSITE_SCORES))
    # row order preserved as requested
    assert [idx.split("  ")[0] for idx in frame.index] == ["tok_a", "tok_b"]
    # values clipped into [0, 1]
    assert (frame.to_numpy() <= 1.0).all()
    assert frame.iloc[1].max() == pytest.approx(1.0)
    # legend annotates axis count
    assert "(7/7 axes)" in frame.index[0]


def test_scores_to_radar_frame_missing_axes_floored_to_zero():
    vals = {c: float("nan") for c in COMPOSITE_SCORES}
    vals["single_neuron_recon"] = 0.6
    df = _scores_df({"tok": vals})
    frame = scores_to_radar_frame(df, ["tok"], score_clip=1.0)
    assert "(1/7 axes)" in frame.index[0]
    # the one present axis carries its value; all others floored to 0
    assert frame.to_numpy().max() == pytest.approx(0.6)
    assert frame.to_numpy().min() == pytest.approx(0.0)
    assert not any(math.isnan(v) for v in frame.to_numpy().ravel())


def test_scores_to_radar_frame_skips_unknown_then_raises_when_empty():
    df = _scores_df({"tok_a": dict.fromkeys(COMPOSITE_SCORES, 0.5)})
    # one good + one missing -> only the good one is kept
    frame = scores_to_radar_frame(df, ["tok_a", "nope"], score_clip=1.0)
    assert frame.shape[0] == 1
    # all-missing -> explicit error rather than an empty plot
    with pytest.raises(ValueError):
        scores_to_radar_frame(df, ["nope"], score_clip=1.0)


# ── load_scores routes through tracker_io ──────────────────────────────────────


def test_load_scores_routes_through_tracker_io_and_caches(tmp_path, monkeypatch):
    """load_scores must resolve the run URL + artifact URI via tracker_io,
    download exactly once, write the cache, and reuse it on the next call."""
    tracker = tmp_path / "tracker.csv"
    tracker.write_text(
        "tokenizer,mlflow_aggregation\n"
        "wanx_A_vae_x,https://mlflow.example/#/experiments/474/runs/abc123def\n"
    )
    scores_payload = {
        "scores": {c: (0.3 if c == "single_neuron_recon" else None) for c in COMPOSITE_SCORES}
    }

    calls: list[str] = []

    def fake_load_json_artifact(uri: str) -> dict:
        calls.append(uri)
        return scores_payload

    # data.py calls tracker_io.load_json_artifact at attribute-access time.
    monkeypatch.setattr(data.tracker_io, "load_json_artifact", fake_load_json_artifact)
    # avoid touching any real MLflow server
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example")

    cache = tmp_path / "cache"
    df = load_scores(tracker, cache_dir=cache)

    assert list(df["tokenizer"]) == ["wanx_A_vae_x"]
    assert df.iloc[0]["family"] == "wanx_A_vae"
    assert df.iloc[0]["single_neuron_recon"] == pytest.approx(0.3)
    # missing scores become NaN, not None
    assert math.isnan(df.iloc[0]["population_structure"])
    # downloaded by run id; the cache is still keyed on the mlflow-artifacts URI
    assert calls == ["runs:/abc123def/category_scores.json"]
    # cache file written after the successful download
    assert len(list(cache.glob("wanx_a_vae_x-*.json"))) == 1

    # Second call must hit the cache, not download again.
    load_scores(tracker, cache_dir=cache)
    assert len(calls) == 1


def test_load_scores_skips_rows_without_run_url(tmp_path, monkeypatch):
    tracker = tmp_path / "tracker.csv"
    tracker.write_text(
        "tokenizer,mlflow_aggregation\n"
        "good,https://mlflow.example/#/experiments/1/runs/aaa\n"
        "blank,\n"
    )
    monkeypatch.setattr(data.tracker_io, "load_json_artifact", lambda uri: {"scores": {}})
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example")
    df = load_scores(tracker)
    assert list(df["tokenizer"]) == ["good"]


def test_load_scores_mlflow_warns_loudly_about_unreadable_artifacts(tmp_path, monkeypatch, capsys):
    """The MLflow route must be as loud as the local one about rows it drops.

    A row with a valid ``mlflow_aggregation`` URL whose artifact will not load — bad
    credentials, a deleted run, a network blip — was dropped with zero output unless
    ``--verbose``, which all four CLIs default off. A short paper table with no
    explanation is exactly the failure this tooling exists to prevent.
    """
    tracker = tmp_path / "tracker.csv"
    tracker.write_text(
        "tokenizer,mlflow_aggregation\n"
        "good,https://mlflow.example/#/experiments/1/runs/aaa\n"
        "gone,https://mlflow.example/#/experiments/1/runs/bbb\n"
    )

    def _load(uri: str) -> dict:
        if "bbb" in uri:
            raise RuntimeError("RESOURCE_DOES_NOT_EXIST")
        return {"scores": {}}

    monkeypatch.setattr(data.tracker_io, "load_json_artifact", _load)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example")

    df = load_scores(tracker)  # verbose stays off, as every CLI leaves it
    captured = capsys.readouterr()

    assert list(df["tokenizer"]) == ["good"]
    assert "WARNING" in captured.err
    assert "gone" in captured.err
    # on stderr, so `build_latex_table ... > paper.tex` still shows it
    assert "WARNING" not in captured.out


def test_load_scores_mlflow_does_not_warn_about_rows_with_no_run_url(
    tmp_path, monkeypatch, capsys
):
    """A blank ``mlflow_aggregation`` is the normal not-yet-aggregated state.

    Warning on it would fire on every tracker with a pending row and train people to
    ignore the warning that means something.
    """
    tracker = tmp_path / "tracker.csv"
    tracker.write_text(
        "tokenizer,mlflow_aggregation\n"
        "good,https://mlflow.example/#/experiments/1/runs/aaa\n"
        "pending,\n"
    )
    monkeypatch.setattr(data.tracker_io, "load_json_artifact", lambda uri: {"scores": {}})
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example")

    load_scores(tracker)
    assert "WARNING" not in capsys.readouterr().err


def test_load_scores_verbose_progress_goes_to_stderr(tmp_path, monkeypatch, capsys):
    """`build_latex_table -v --scores-dir ... > paper.tex` must still emit valid LaTeX.

    The per-row "loaded <tokenizer>" trace went to stdout, so with -v it landed as
    line 1 of the .tex.
    """
    _no_mlflow(monkeypatch)
    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()
    (scores_dir / "tok_a.json").write_text(json.dumps(_local_scores_doc()))
    tracker = tmp_path / "tracker.csv"
    tracker.write_text("tokenizer\ntok_a\n")

    load_scores(tracker, scores_dir=scores_dir, verbose=True)
    captured = capsys.readouterr()

    assert "loaded tok_a" in captured.err
    assert captured.out == ""


def test_load_scores_raises_without_aggregation_column(tmp_path, monkeypatch):
    tracker = tmp_path / "tracker.csv"
    tracker.write_text("tokenizer\nwanx\n")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example")
    with pytest.raises(ValueError, match="mlflow_aggregation"):
        load_scores(tracker)


# ── load_scores from a local scores dir (build_aggregation_tables.py output) ───


def _local_scores_doc():
    """A category_scores.json as build_aggregation_tables.py persists it.

    Key names and nesting mirror the real artifacts written to
    <outdir>/_work/*/with_m2/category_scores.json.
    """
    return {
        "scores": {
            "single_neuron_recon": 0.8,
            "population_structure": 0.7,
            "latent_structure_and_robustness": 0.6,
            "decoding": 0.5,
            "downstream_encoding": 0.4,
            "generalization": 0.3,
            "compression": 1.25,
        },
        "diagnostics": {
            "single_neuron_recon": {
                "score": 0.8,
                "n_metrics": 1,
                "per_metric": {
                    "spike_detection_f1": {
                        "normalized": 0.8,
                        "clipped": False,
                        "raw_value": 0.42,
                        "direction": "higher",
                    }
                },
            }
        },
        "metadata": {"generalization_source": "precomputed"},
    }


def _no_mlflow(monkeypatch):
    """Any MLflow touch in local mode is a contract violation, not a slow path."""

    def _boom(*_a, **_k):
        raise AssertionError("local scores_dir mode must not contact MLflow")

    monkeypatch.setattr(data.mlflow, "set_tracking_uri", _boom)
    monkeypatch.setattr(data.tracker_io, "load_json_artifact", _boom)
    monkeypatch.setattr(data.tracker_io, "artifact_uri", _boom)


def test_load_scores_from_local_scores_dir(tmp_path, monkeypatch):
    """The single-pass runner's output must feed the plotting/LaTeX tools with no
    MLflow round-trip, including the per-metric diagnostics that plot_correlation
    and plot_family read."""
    _no_mlflow(monkeypatch)
    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()
    (scores_dir / "wanx_a_vae_x.json").write_text(json.dumps(_local_scores_doc()))

    tracker = tmp_path / "tracker.csv"
    # No mlflow_aggregation column at all: that is the whole point.
    tracker.write_text(
        "tokenizer,latent_type,tokenization_regime,variant\n"
        "wanx_A_vae_x,continuous,population,v1\n"
    )

    df = load_scores(tracker, scores_dir=scores_dir)

    assert list(df["tokenizer"]) == ["wanx_A_vae_x"]
    # the same row assembly as the MLflow path: family + metadata + the 7 columns
    assert df.iloc[0]["family"] == "wanx_A_vae"
    assert df.iloc[0]["latent_type"] == "continuous"
    assert df.iloc[0]["tokenization_regime"] == "population"
    assert df.iloc[0]["variant"] == "v1"
    assert df.iloc[0]["single_neuron_recon"] == pytest.approx(0.8)
    assert df.iloc[0]["compression"] == pytest.approx(1.25)
    # per-metric diagnostics survive: plot_correlation / plot_family read these
    assert extract_metric(df.iloc[0]["_scores_data"], "sn_recon/spike_detection_f1") == (
        pytest.approx(0.8)
    )


def test_load_scores_local_slug_matches_the_runner_filename(tmp_path, monkeypatch):
    """The runner names files by the same slug data.py derives, so punctuation-heavy
    tokenizer names must round-trip."""
    _no_mlflow(monkeypatch)
    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()
    (scores_dir / "wanx_a_vae_b1e_3_v2.json").write_text(json.dumps(_local_scores_doc()))

    tracker = tmp_path / "tracker.csv"
    tracker.write_text("tokenizer\nwanx_A_vae.b1e-3 v2\n")

    df = load_scores(tracker, scores_dir=scores_dir)
    assert list(df["tokenizer"]) == ["wanx_A_vae.b1e-3 v2"]


def test_load_scores_local_skips_rows_with_no_json(tmp_path, monkeypatch):
    """A tokenizer the runner failed on writes no <slug>.json; it must be skipped,
    not crash the plot, and must not take the rest of the cohort with it."""
    _no_mlflow(monkeypatch)
    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()
    (scores_dir / "good.json").write_text(json.dumps(_local_scores_doc()))

    tracker = tmp_path / "tracker.csv"
    tracker.write_text("tokenizer\ngood\nmissing\n")

    df = load_scores(tracker, scores_dir=scores_dir)
    assert list(df["tokenizer"]) == ["good"]


def test_load_scores_local_warns_loudly_about_skipped_tokenizers(tmp_path, monkeypatch, capsys):
    """A short table must never be silent.

    scores/ is allowed to lag category_scores.csv by a run (the runner re-scores rather
    than deleting CSV rows), so a tracker row can resolve to no JSON. The skip was
    verbose-only, and all four CLIs default verbose off — so a paper table built through
    --scores-dir could quietly omit tokenizers with nothing anywhere saying so.
    """
    _no_mlflow(monkeypatch)
    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()
    (scores_dir / "tok_b.json").write_text(json.dumps(_local_scores_doc()))

    tracker = tmp_path / "tracker.csv"
    tracker.write_text("tokenizer\ntok_a\ntok_b\n")

    df = load_scores(tracker, scores_dir=scores_dir)
    assert list(df["tokenizer"]) == ["tok_b"]

    captured = capsys.readouterr()
    # stderr, not stdout: `build_latex_table ... > paper_table_agg.tex` is documented,
    # and a warning on stdout would land inside the .tex file.
    assert "WARNING" in captured.err
    assert "tok_a" in captured.err
    assert "tok_a" not in captured.out


def test_load_scores_local_raises_when_nothing_matched(tmp_path, monkeypatch):
    """An empty result is a user error worth naming, and the message must point at
    the local route rather than telling them to run the MLflow batch runner."""
    _no_mlflow(monkeypatch)
    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()
    tracker = tmp_path / "tracker.csv"
    tracker.write_text("tokenizer\nnobody\n")

    with pytest.raises(RuntimeError, match="scores_dir"):
        load_scores(tracker, scores_dir=scores_dir)


def test_load_scores_raises_when_only_some_scores_include_m2(tmp_path, monkeypatch):
    """A table whose latent scores mix 'with M2' and 'without M2' must not be plotted."""
    _no_mlflow(monkeypatch)
    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()
    for tokenizer, m2 in (("tok_a", "included"), ("tok_b", "excluded_no_report")):
        doc = _local_scores_doc()
        doc["metadata"]["robustness_normalization"] = {"m1": "components_mean", "m2": m2}
        (scores_dir / f"{tokenizer}.json").write_text(json.dumps(doc))
    tracker = tmp_path / "tracker.csv"
    tracker.write_text("tokenizer\ntok_a\ntok_b\n")

    from ntt.evaluation.aggregation.comparability import MixedScorePartsError

    with pytest.raises(MixedScorePartsError, match="--common-parts-only") as excinfo:
        load_scores(tracker, scores_dir=scores_dir)
    assert "latent-corruption robustness (`m2`, part of latent_structure_and_robustness)" in str(
        excinfo.value
    )
    assert "in the score of ['tok_a'], but not of ['tok_b']" in str(excinfo.value)


def test_load_scores_raises_when_it_cannot_check_unless_common_parts_only(
    tmp_path, monkeypatch, capsys
):
    """Scores without recorded comparability fields cannot be compared: an error that
    names the flag, and with common_parts_only a warning."""
    _no_mlflow(monkeypatch)
    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()
    for tokenizer in ("tok_a", "tok_b"):
        (scores_dir / f"{tokenizer}.json").write_text(json.dumps(_local_scores_doc()))
    tracker = tmp_path / "tracker.csv"
    tracker.write_text("tokenizer\ntok_a\ntok_b\n")

    from ntt.evaluation.aggregation.comparability import UncheckedComparabilityError

    with pytest.raises(UncheckedComparabilityError) as excinfo:
        load_scores(tracker, scores_dir=scores_dir)
    assert "because" in str(excinfo.value) and "does not save it" in str(excinfo.value)
    assert "Pass --common-parts-only to continue with a warning instead" in str(excinfo.value)

    df = load_scores(tracker, scores_dir=scores_dir, common_parts_only=True)
    assert sorted(df["tokenizer"]) == ["tok_a", "tok_b"]
    assert "WARNING: could NOT check" in capsys.readouterr().err


def test_load_scores_mlflow_route_needs_a_tracking_server(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    tracker = tmp_path / "tracker.csv"
    tracker.write_text("tokenizer,mlflow_aggregation\ntok,http://h/#/experiments/1/runs/abc\n")

    def _boom(*_a, **_k):
        raise AssertionError("must not download")

    monkeypatch.setattr(data.tracker_io, "load_json_artifact", _boom)
    with pytest.raises(ValueError, match="MLFLOW_TRACKING_URI"):
        load_scores(tracker)


def test_load_scores_names_the_server_of_links_to_another_one(tmp_path, monkeypatch):
    # Never read from the server a link names: credentials go wherever the request goes.
    from mlflow.tracking._tracking_service import utils

    # mlflow_uri= below sets the process-wide URI, which outranks the env var in later tests.
    monkeypatch.setattr(utils, "_tracking_uri", None)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://new.example/")
    tracker = tmp_path / "tracker.csv"
    tracker.write_text(
        "tokenizer,mlflow_aggregation\n"
        "tok_a,https://new.example/#/experiments/1/runs/aaa\n"
        "tok_b,https://old.example/#/experiments/1/runs/bbb\n"
        "tok_c,\n"
    )
    calls = []

    def _load(uri):
        calls.append(uri)
        return {"scores": {}}

    monkeypatch.setattr(data.tracker_io, "load_json_artifact", _load)
    with pytest.raises(ValueError, match=r"\['https://old.example'\].*--mlflow-uri"):
        load_scores(tracker)
    assert calls == []

    # Explicitly naming the server reads every run from it.
    df = load_scores(tracker, mlflow_uri="https://old.example", common_parts_only=True)
    assert list(df["tokenizer"]) == ["tok_a", "tok_b"]
    assert len(calls) == 2
