"""Tests for scripts/analysis/build_aggregation_tables.py.

Covers the pure, network-free parts: the score-key contract, the record flattener
(incl. M2 provenance), the delta table, and the download-once artifact cache.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from scripts.analysis import build_aggregation_tables as bat


# ── score-key contract ─────────────────────────────────────────────────────────
def test_seven_keys_match_category_scores_dataclass():
    """SEVEN must stay in sync with the CategoryScores fields.

    A drifted key name does not raise — it produces a silently empty column — so the
    contract is asserted here rather than left to runtime.
    """
    from dataclasses import fields

    from ntt.evaluation.aggregation.category_scores import CategoryScores

    actual = {f.name for f in fields(CategoryScores)} - {
        "diagnostics",
        "poisson_norm_sessions",
        "metric_references",
    }
    assert set(bat.SEVEN) | set(bat.DECODING_BREAKDOWNS) == actual
    assert len(bat.SEVEN) == 7


def test_score_extra_carries_every_comparability_field():
    """Every field find_mixed_member_rows compares must be a column of the CSV, or the
    check silently sees it as "<not recorded>" on every resumed row."""
    from ntt.evaluation.aggregation.comparability import COMPARABILITY_ASPECTS

    assert set(COMPARABILITY_ASPECTS) <= set(bat.SCORE_EXTRA)


# ── score_record ───────────────────────────────────────────────────────────────
def _cs(latent=0.5, m2_prov="included", m2_raw=0.77):
    """A category_scores.json in which everything the comparability check reads is
    recorded and the same for every tokenizer (only a recon input), so tables built
    from it pass the check."""
    return {
        "scores": dict.fromkeys(bat.SEVEN, 0.1) | {"latent_structure_and_robustness": latent},
        "diagnostics": {
            "single_neuron_recon": {"per_metric": {"spike_f1": {}}},
            "population_structure": {"per_metric": {"corr": {}}},
            "latent_structure_and_robustness": {
                "per_metric": {
                    "m1:components_mean": {"raw_value": 0.2},
                    "m2:acg_correlation/eps_0.1": {"raw_value": m2_raw},
                }
            },
            "decoding": {
                "per_metric": {"ridge/x": {}},
                "provenance": {"poyo": "absent", "ridge": "0.1 (1 target)"},
            },
        },
        "metadata": {
            "comparability_version": bat.COMPARABILITY_VERSION,
            "recon_chunk_window_ms": {"s1/ridge": [200]},
            "generalization_source": "precomputed",
            "robustness_normalization": {"m1": "components_mean", "m2": m2_prov},
            "inputs": {"recon": "results.json"},
            "input_sessions": {
                "recon": {"session_ids": ["s1"], "n_sessions": 1, "aggregation_strategy": "mean"}
            },
            "input_settings": {"recon": {"metrics": {}}},
            "poisson_norm_sessions": {"source": "precomputed", "session_ids": ["s1"]},
            "metric_references": {},
            "m2_metric": "acg_correlation",
            "m2_epsilon": 0.1,
            "score_clip": -1.0,
            "compression_metric": "compression_rate",
        },
    }


def test_score_record_carries_m2_provenance_and_raw_value():
    rec = bat.score_record("tok_a", _cs())
    assert rec["tokenizer"] == "tok_a"
    assert rec["latent_structure_and_robustness"] == pytest.approx(0.5)
    assert rec["m2_provenance"] == "included"
    assert rec["m2_raw_acg"] == pytest.approx(0.77)
    assert rec["m2_metric_key"] == "m2:acg_correlation/eps_0.1"
    assert rec["generalization_source"] == "precomputed"


def test_score_record_without_m2_member_leaves_m2_fields_absent():
    cs = _cs()
    cs["diagnostics"]["latent_structure_and_robustness"]["per_metric"] = {
        "m1:components_mean": {"raw_value": 0.2}
    }
    cs["metadata"]["robustness_normalization"]["m2"] = "excluded_no_report"
    rec = bat.score_record("tok_b", cs)
    assert rec["m2_provenance"] == "excluded_no_report"
    assert "m2_raw_acg" not in rec


def test_score_record_tolerates_empty_payload():
    rec = bat.score_record("tok_c", {})
    assert rec["tokenizer"] == "tok_c"
    assert all(rec[k] is None for k in bat.SEVEN)
    assert rec["m2_provenance"] is None


# ── delta table ────────────────────────────────────────────────────────────────
def test_write_delta_computes_shift(tmp_path: Path):
    with_m2 = [bat.score_record("a", _cs(latent=0.48))]
    without = [bat.score_record("a", _cs(latent=0.33, m2_prov="excluded_no_report"))]
    out = tmp_path / "delta.csv"
    bat.write_delta(with_m2, without, out)
    df = pd.read_csv(out)
    assert df.loc[0, "delta"] == pytest.approx(0.15)
    assert df.loc[0, "latent_bucket_with_m2"] == pytest.approx(0.48)
    assert df.loc[0, "latent_bucket_without_m2"] == pytest.approx(0.33)


def test_write_delta_none_when_a_side_is_missing(tmp_path: Path):
    out = tmp_path / "delta.csv"
    bat.write_delta([bat.score_record("a", _cs(latent=0.4))], [], out)
    df = pd.read_csv(out)
    assert pd.isna(df.loc[0, "delta"])


# ── artifact cache ─────────────────────────────────────────────────────────────
def test_cache_refetches_when_source_run_changes(tmp_path: Path, monkeypatch):
    cache = bat.ArtifactCache(tmp_path / "cache", bat.RunLog(tmp_path / "log.txt"))
    calls = []

    def download(uri, dst_path):
        calls.append(uri)
        local = Path(dst_path) / "results.json"
        local.write_text(json.dumps({"source": uri}))
        return str(local)

    monkeypatch.setattr("mlflow.artifacts.download_artifacts", download)
    old_uri = "mlflow-artifacts:/1/abc/artifacts/results.json"
    new_uri = "mlflow-artifacts:/1/def/artifacts/results.json"
    old = cache.fetch("tok", "recon", old_uri)
    cache = bat.ArtifactCache(cache.root, cache.log)
    new = cache.fetch("tok", "recon", new_uri)
    assert json.loads(new.read_text()) == {"source": new_uri}
    assert old != new
    assert cache.fetch("tok", "recon", new_uri) == new
    assert calls == [old_uri, new_uri]


def test_cache_reuses_manifest_entry_without_downloading(tmp_path: Path, monkeypatch):
    log = bat.RunLog(tmp_path / "log.txt")
    cache = bat.ArtifactCache(tmp_path / "cache", log)
    payload = tmp_path / "results.json"
    payload.write_text("{}")
    cache.manifest["tok|recon"] = {
        "source": bat.tracker_io.artifact_cache_key(
            "mlflow-artifacts:/1/2/artifacts/results.json"
        ),
        "path": str(payload),
    }

    def _boom(*a, **k):  # any download attempt is a failure of the cache contract
        raise AssertionError("cache miss: should not download a manifest hit")

    monkeypatch.setattr("mlflow.artifacts.download_artifacts", _boom)
    assert cache.fetch("tok", "recon", "mlflow-artifacts:/1/2/artifacts/results.json") == payload


def test_cache_redownloads_when_recorded_path_vanished(tmp_path: Path, monkeypatch):
    log = bat.RunLog(tmp_path / "log.txt")
    cache = bat.ArtifactCache(tmp_path / "cache", log)
    source = bat.tracker_io.artifact_cache_key("mlflow-artifacts:/1/2/artifacts/results.json")
    cache.manifest["tok|recon"] = {"source": source, "path": str(tmp_path / "gone.json")}

    fresh = tmp_path / "fresh.json"
    fresh.write_text("{}")
    monkeypatch.setattr("mlflow.artifacts.download_artifacts", lambda *a, **k: str(fresh))
    assert cache.fetch("tok", "recon", "mlflow-artifacts:/1/2/artifacts/results.json") == fresh
    # ...and the manifest is repaired on disk for the next run
    assert json.loads(cache.manifest_path.read_text())["tok|recon"]["path"] == str(fresh)


def test_cache_retries_then_raises(tmp_path: Path, monkeypatch):
    log = bat.RunLog(tmp_path / "log.txt")
    cache = bat.ArtifactCache(tmp_path / "cache", log, attempts=3)
    calls = {"n": 0}

    def _flaky(*a, **k):
        calls["n"] += 1
        raise RuntimeError("Response ended prematurely")

    monkeypatch.setattr("mlflow.artifacts.download_artifacts", _flaky)
    with pytest.raises(RuntimeError, match="prematurely"):
        cache.fetch("tok", "recon", "mlflow-artifacts:/1/2/artifacts/results.json")
    assert calls["n"] == 3


def test_cache_survives_corrupt_manifest(tmp_path: Path):
    root = tmp_path / "cache"
    root.mkdir()
    (root / "manifest.json").write_text("{not json")
    cache = bat.ArtifactCache(root, bat.RunLog(tmp_path / "log.txt"))
    assert cache.manifest == {}


# ── metadata-rows writer ────────────────────────────────────────────────────────
def test_write_metadata_rows_preserves_metadata_columns_and_order(tmp_path: Path):
    fields = ["tokenizer", "variant", "latent_type", "tokenization_regime", "mlflow_recon"]
    rows = [
        {
            "tokenizer": "a",
            "variant": "v1",
            "latent_type": "continuous",
            "tokenization_regime": "population",
            "mlflow_recon": "url",
        }
    ]
    out = tmp_path / "metadata_rows.csv"
    bat.write_metadata_rows(fields, rows, out)
    df = pd.read_csv(out)
    assert list(df.columns) == fields
    assert df.loc[0, "latent_type"] == "continuous"


def test_write_metadata_rows_empty_rows_still_writes_header(tmp_path: Path):
    out = tmp_path / "metadata_rows.csv"
    bat.write_metadata_rows(["tokenizer", "variant"], [], out)
    assert list(pd.read_csv(out).columns) == ["tokenizer", "variant"]


# ── wide writer ────────────────────────────────────────────────────────────────
def test_write_wide_puts_tokenizer_first_and_sorts_metrics(tmp_path: Path):
    out = tmp_path / "wide.csv"
    bat.write_wide([{"tokenizer": "a", "z/m": 1.0, "a/m": 2.0}], out)
    assert list(pd.read_csv(out).columns) == ["tokenizer", "a/m", "z/m"]


# ── provenance columns ─────────────────────────────────────────────────────────
def _cs_with_decoding_provenance():
    cs = _cs()
    cs["diagnostics"]["decoding"] = {
        "provenance": {"poyo": "0.79 (2 sessions)", "ridge": "0.25 (2 targets)"},
        "per_metric": {},
    }
    return cs


def test_score_record_carries_m1_and_decoding_provenance():
    """M2 had a provenance column and M1 did not, so M1's warn-and-exclude path was
    invisible in the CSV people report from — as was the decoding column's poyo/ridge
    split, which decides whether the ridge family contributed at all."""
    rec = bat.score_record("tok_d", _cs_with_decoding_provenance())

    assert rec["m1_provenance"] == "components_mean"
    assert rec["decoding_provenance"] == "poyo=0.79 (2 sessions);ridge=0.25 (2 targets)"
    # Both must be in SCORE_EXTRA or write_scores drops them from the CSV.
    assert "m1_provenance" in bat.SCORE_EXTRA
    assert "decoding_provenance" in bat.SCORE_EXTRA


def test_score_record_keeps_provenance_columns_when_the_payload_has_none():
    """Emitted unconditionally, so the CSV header is stable across cohorts — a
    conditional key would silently drop the column when no row has decoding."""
    rec = bat.score_record("tok_e", {})
    assert rec["m1_provenance"] is None
    assert rec["decoding_provenance"] is None


def test_write_scores_emits_the_provenance_columns(tmp_path: Path):
    out = tmp_path / "scores.csv"
    bat.write_scores([bat.score_record("tok_d", _cs_with_decoding_provenance())], out)
    df = pd.read_csv(out)
    assert df.loc[0, "m1_provenance"] == "components_mean"
    # the ';' separator and the parenthesised counts must survive the CSV round-trip
    assert df.loc[0, "decoding_provenance"] == "poyo=0.79 (2 sessions);ridge=0.25 (2 targets)"


# ── failure bookkeeping ────────────────────────────────────────────────────────
def test_process_variants_returns_none_when_a_variant_raises(tmp_path: Path, monkeypatch):
    """A failed variant must not yield a record.

    The old handler appended a placeholder row of nulls and then fell through to
    write_scores / done.add(slug) / n_ok += 1, so the summary printed '0 failed',
    the shipped CSV carried a null row, and every later run skipped the tokenizer
    because _existing() seeds done from the output file.
    """

    def _boom(paths, out_dir, *, wire_m2, config_root, **_):
        raise RuntimeError("aggregation exploded")

    monkeypatch.setattr(bat, "run_variant", _boom)
    log = bat.RunLog(tmp_path / "log.txt")

    result = bat.process_variants(
        "tok", {}, tmp_path / "work", None, [("with_m2", True)], tmp_path / "config", log
    )

    assert result is None
    assert "AGG with_m2 FAILED" in (tmp_path / "log.txt").read_text()


def test_process_variants_returns_a_record_per_variant(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(bat, "run_variant", lambda *a, **k: _cs())
    log = bat.RunLog(tmp_path / "log.txt")

    result = bat.process_variants(
        "tok",
        {},
        tmp_path / "work",
        None,
        [("with_m2", True), ("no_m2", False)],
        tmp_path / "config",
        log,
    )

    assert result is not None
    assert [label for label, _ in result] == ["with_m2", "no_m2"]
    assert all(rec["tokenizer"] == "tok" for _, rec in result)


def test_process_variants_discards_earlier_variants_when_a_later_one_fails(
    tmp_path: Path, monkeypatch
):
    """All-or-nothing across variants: with --compare-m2 the old loop could leave a
    real row in category_scores.csv and a null row in category_scores_no_m2.csv for
    the same tokenizer, so the delta table differenced a score against a null."""
    seen: list[bool] = []

    def _second_one_explodes(paths, out_dir, *, wire_m2, config_root, **_):
        seen.append(wire_m2)
        if not wire_m2:
            raise RuntimeError("no_m2 exploded")
        return _cs()

    monkeypatch.setattr(bat, "run_variant", _second_one_explodes)

    result = bat.process_variants(
        "tok",
        {},
        tmp_path / "work",
        None,
        [("with_m2", True), ("no_m2", False)],
        tmp_path / "config",
        bat.RunLog(tmp_path / "log.txt"),
    )

    assert seen == [True, False]  # the first variant did run and its record is dropped
    assert result is None


# ── persisted per-tokenizer category_scores.json ───────────────────────────────
def test_process_variants_persists_the_with_m2_scores_json(tmp_path: Path, monkeypatch):
    """run_variant already produced a full category_scores.json per tokenizer and
    dropped it in _work/. Persisting it as <scores_dir>/<slug>.json is what lets
    load_scores(scores_dir=...) feed the plots with no MLflow round-trip."""
    monkeypatch.setattr(
        bat,
        "run_variant",
        lambda paths, out_dir, *, wire_m2, config_root, **_: (
            _cs(latent=0.48) if wire_m2 else _cs(latent=0.33)
        ),
    )
    scores_dir = tmp_path / "scores"

    result = bat.process_variants(
        "tok",
        {},
        tmp_path / "work",
        scores_dir,
        [("with_m2", True), ("no_m2", False)],
        tmp_path / "config",
        bat.RunLog(tmp_path / "log.txt"),
    )

    assert result is not None
    written = json.loads((scores_dir / "tok.json").read_text())
    # the with_m2 variant, not no_m2 — one file per tokenizer, the reported one
    assert written["scores"]["latent_structure_and_robustness"] == pytest.approx(0.48)
    # full payload, diagnostics included (the reason this is JSON and not the flat CSV)
    assert written["diagnostics"]["latent_structure_and_robustness"]["per_metric"]
    assert not (scores_dir / "tok_no_m2.json").exists()


def test_process_variants_writes_no_scores_json_when_a_variant_fails(tmp_path: Path, monkeypatch):
    """A failed tokenizer is recorded nowhere. A leftover <slug>.json would
    make load_scores read a stale
    payload for a tokenizer that has no row in category_scores.csv."""
    scores_dir = tmp_path / "scores"
    scores_dir.mkdir()
    stale = scores_dir / "tok.json"
    stale.write_text('{"scores": {"decoding": 0.99}}')  # from an earlier, successful run

    def _second_one_explodes(paths, out_dir, *, wire_m2, config_root, **_):
        if not wire_m2:
            raise RuntimeError("no_m2 exploded")
        return _cs()

    monkeypatch.setattr(bat, "run_variant", _second_one_explodes)

    result = bat.process_variants(
        "tok",
        {},
        tmp_path / "work",
        scores_dir,
        [("with_m2", True), ("no_m2", False)],
        tmp_path / "config",
        bat.RunLog(tmp_path / "log.txt"),
    )

    assert result is None
    assert not stale.exists()


def test_persisted_json_filename_matches_what_load_scores_looks_up(tmp_path: Path, monkeypatch):
    """Writer and reader must share one slug rule.

    The tracker's `tokenizer` values carry capitals and dots (`wanx_A_vae...`), which
    data.load_scores lowercases and punctuation-folds to find the file. Slugging on
    only one side writes `wanx_A_vae_x.json` and looks up `wanx_a_vae_x.json` — the
    reader just skips the row, with no error anywhere.
    """
    from scripts.analysis import data

    monkeypatch.setattr(bat, "run_variant", lambda *a, **k: _cs())
    scores_dir = tmp_path / "scores"
    tokenizer = "wanx_A_vae.b1e-3"

    bat.process_variants(
        tokenizer,
        {},
        tmp_path / "work",
        scores_dir,
        [("with_m2", True)],
        tmp_path / "config",
        bat.RunLog(tmp_path / "log.txt"),
    )

    tracker = tmp_path / "tracker.csv"
    tracker.write_text(f"tokenizer\n{tokenizer}\n")
    monkeypatch.setattr(data.mlflow, "set_tracking_uri", lambda *a, **k: None)
    assert list(data.load_scores(tracker, scores_dir=scores_dir)["tokenizer"]) == [tokenizer]


def test_process_variants_without_a_scores_dir_writes_nothing(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(bat, "run_variant", lambda *a, **k: _cs())
    bat.process_variants(
        "tok",
        {},
        tmp_path / "work",
        None,
        [("with_m2", True)],
        tmp_path / "config",
        bat.RunLog(tmp_path / "log.txt"),
    )
    assert not (tmp_path / "scores").exists()


def test_run_log_appends_across_runs(tmp_path: Path):
    """RunLog rewrote the whole file on every call, so resuming a run truncated the
    previous run's trail — the record you need to find why a row failed."""
    path = tmp_path / "run_log.txt"
    bat.RunLog(path)("first run line")
    bat.RunLog(path)("second run line")

    text = path.read_text()
    assert "first run line" in text
    assert "second run line" in text


# ── end-to-end resume bookkeeping (main) ───────────────────────────────────────
def _tracker_csv(path: Path, slugs: list[str]) -> Path:
    """A minimal tracker: only `mlflow_recon`, which is the one required column."""
    lines = ["tokenizer,mlflow_recon"]
    lines += [
        f"{slug},https://mlflow.internal/#/experiments/1/runs/abc12{i}"
        for i, slug in enumerate(slugs)
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


def _stub_download(uri, dst_path):
    local = Path(dst_path) / "results.json"
    local.write_text("{}")
    return str(local)


@pytest.mark.parametrize("key", ["m1", "m2", "encoding", "decoding"])
@pytest.mark.parametrize("failure", ["download", "invalid_json"])
def test_optional_artifact_failure_preserves_recon_scores(tmp_path, monkeypatch, key, failure):
    outdir = tmp_path / "out"
    sheet = _tracker_csv(tmp_path / "tracker.csv", ["tok"])
    monkeypatch.setattr(
        bat.tracker_io,
        "row_to_source_uris",
        lambda _, **__: {"recon": "runs:/recon", key: f"runs:/{key}"},
    )

    def download(uri, dst_path):
        if uri == f"runs:/{key}" and failure == "download":
            raise FileNotFoundError(f"{key} artifact absent")
        dest = Path(dst_path)
        if uri == "runs:/decoding":
            (dest / "run_result.json").write_text("{bad json")
            return str(dest)
        local = dest / "results.json"
        payload = {
            "scored_window_seconds": 1.0,
            "comparison_vs_poisson": {},
            "compression": {"compression_rate": 0.5},
        }
        local.write_text(json.dumps(payload) if uri == "runs:/recon" else "{bad json")
        return str(local)

    monkeypatch.setattr("mlflow.artifacts.download_artifacts", download)
    monkeypatch.setattr(sys, "argv", ["builder", "--sheet", str(sheet), "--outdir", str(outdir)])
    bat.main()
    scores = pd.read_csv(outdir / "category_scores.csv")
    assert list(scores["tokenizer"]) == ["tok"]
    assert scores.iloc[0]["compression"] == 0.5
    assert (outdir / "scores" / "tok.json").exists()
    log = (outdir / "run_log.txt").read_text()
    assert f"EXCLUDING {key}" in log
    assert "1 processed, 0 resumed-skip, 0 no-recon, 0 failed" in log


def test_required_recon_download_failure_emits_no_scores(tmp_path, monkeypatch):
    outdir = tmp_path / "out"
    sheet = _tracker_csv(tmp_path / "tracker.csv", ["tok"])

    def download(*args, **kwargs):
        raise FileNotFoundError("recon artifact absent")

    monkeypatch.setattr("mlflow.artifacts.download_artifacts", download)
    monkeypatch.setattr(sys, "argv", ["builder", "--sheet", str(sheet), "--outdir", str(outdir)])
    bat.main()
    assert not (outdir / "category_scores.csv").exists()
    assert "1 failed" in (outdir / "run_log.txt").read_text()


def test_a_failed_tokenizer_is_counted_and_retried_on_the_next_run(
    tmp_path: Path, monkeypatch, capsys
):
    """An aggregation failure is recorded as a failure, end to end.

    Without that, an aggregation failure would leave a seven-null placeholder row in
    category_scores.csv, a real wide row next to it, `0 failed` in the summary, and
    the slug in `done` forever after — because `_existing()` reseeds `done` from the
    output CSVs, so every later run skipped it.
    """
    outdir = tmp_path / "out"
    cache = tmp_path / "cache"
    config_root = tmp_path / "config"
    config_root.mkdir()
    # The failing slug is FIRST so the run continues past it: its wide row must not
    # survive into the write_wide() that the next (successful) tokenizer triggers.
    sheet = _tracker_csv(tmp_path / "tracker.csv", ["tok_bad", "tok_ok"])

    monkeypatch.setattr("mlflow.artifacts.download_artifacts", _stub_download)
    monkeypatch.setattr(bat, "flatten_tokenizer_metrics", lambda *a, **k: {"raw/metric": 1.0})

    failing = {"tok_bad"}

    def _run_variant(paths, out_dir, *, wire_m2, config_root, **_):
        if Path(out_dir).parent.name in failing:
            raise RuntimeError("aggregation exploded")
        return _cs()

    monkeypatch.setattr(bat, "run_variant", _run_variant)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_aggregation_tables.py",
            "--sheet",
            str(sheet),
            "--outdir",
            str(outdir),
            "--cache",
            str(cache),
            "--config-root",
            str(config_root),
        ],
    )

    bat.main()
    capsys.readouterr()

    scores = pd.read_csv(outdir / "category_scores.csv")
    wide = pd.read_csv(outdir / "wide_metrics_table.csv")
    assert list(scores["tokenizer"]) == ["tok_ok"]  # no null placeholder row
    assert list(wide["tokenizer"]) == ["tok_ok"]  # and no orphaned wide row
    assert (
        "DONE: 1 processed, 0 resumed-skip, 0 no-recon, 1 failed"
        in (outdir / "run_log.txt").read_text()
    )

    # Second run, with the transient failure gone: the tokenizer must be retried.
    failing.clear()
    bat.main()
    capsys.readouterr()

    scores = pd.read_csv(outdir / "category_scores.csv")
    wide = pd.read_csv(outdir / "wide_metrics_table.csv")
    assert sorted(scores["tokenizer"]) == ["tok_bad", "tok_ok"]
    assert sorted(wide["tokenizer"]) == ["tok_bad", "tok_ok"]
    assert scores["latent_structure_and_robustness"].notna().all()

    log_text = (outdir / "run_log.txt").read_text()
    assert "DONE: 1 processed, 1 resumed-skip, 0 no-recon, 0 failed" in log_text
    # ...and the first run's trail is still there to explain why the row was missing
    assert "DONE: 1 processed, 0 resumed-skip, 0 no-recon, 1 failed" in log_text


def test_compare_m2_success_path_routes_each_variant_to_its_own_table(
    tmp_path: Path, monkeypatch, capsys
):
    """Guards the success path across the variant-loop extraction: each variant's
    record must still reach its own sink, or the delta table differences a score
    against itself."""
    outdir = tmp_path / "out"
    config_root = tmp_path / "config"
    config_root.mkdir()
    sheet = _tracker_csv(tmp_path / "tracker.csv", ["tok_ok"])

    monkeypatch.setattr("mlflow.artifacts.download_artifacts", _stub_download)
    monkeypatch.setattr(bat, "flatten_tokenizer_metrics", lambda *a, **k: {"raw/metric": 1.0})
    monkeypatch.setattr(
        bat,
        "run_variant",
        lambda paths, out_dir, *, wire_m2, config_root, **_: (
            _cs(latent=0.48) if wire_m2 else _cs(latent=0.33, m2_prov="excluded_no_report")
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_aggregation_tables.py",
            "--sheet",
            str(sheet),
            "--outdir",
            str(outdir),
            "--cache",
            str(tmp_path / "cache"),
            "--config-root",
            str(config_root),
            "--compare-m2",
        ],
    )

    bat.main()
    capsys.readouterr()

    with_m2 = pd.read_csv(outdir / "category_scores.csv")
    without = pd.read_csv(outdir / "category_scores_no_m2.csv")
    delta = pd.read_csv(outdir / "category_scores_m2_delta.csv")
    assert with_m2.loc[0, "latent_structure_and_robustness"] == pytest.approx(0.48)
    assert without.loc[0, "latent_structure_and_robustness"] == pytest.approx(0.33)
    assert without.loc[0, "m2_provenance"] == "excluded_no_report"
    assert delta.loc[0, "delta"] == pytest.approx(0.15)
    assert (
        "DONE: 1 processed, 0 resumed-skip, 0 no-recon, 0 failed"
        in (outdir / "run_log.txt").read_text()
    )


def test_resume_rescores_a_missing_scores_json(tmp_path: Path, monkeypatch, capsys):
    """`scores/<slug>.json` is an output, so `done` must require it.

    Resuming an outdir built before this existed — or one whose scores/ was deleted —
    would otherwise skip every row as already-complete and leave the directory empty,
    and load_scores(scores_dir=...) would then raise 'No scores could be loaded' on a
    cohort whose CSVs are perfectly fine.
    """
    outdir = tmp_path / "out"
    config_root = tmp_path / "config"
    config_root.mkdir()
    sheet = _tracker_csv(tmp_path / "tracker.csv", ["tok_ok"])

    monkeypatch.setattr("mlflow.artifacts.download_artifacts", _stub_download)
    monkeypatch.setattr(bat, "flatten_tokenizer_metrics", lambda *a, **k: {"raw/metric": 1.0})
    monkeypatch.setattr(bat, "run_variant", lambda *a, **k: _cs())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_aggregation_tables.py",
            "--sheet",
            str(sheet),
            "--outdir",
            str(outdir),
            "--cache",
            str(tmp_path / "cache"),
            "--config-root",
            str(config_root),
        ],
    )

    bat.main()
    capsys.readouterr()
    assert (outdir / "scores" / "tok_ok.json").exists()

    # Simulate the pre-existing-outdir case: CSVs intact, no scores/.
    (outdir / "scores" / "tok_ok.json").unlink()
    bat.main()
    capsys.readouterr()

    assert (outdir / "scores" / "tok_ok.json").exists()
    # ...and the CSV did not gain a duplicate row on the way
    assert list(pd.read_csv(outdir / "category_scores.csv")["tokenizer"]) == ["tok_ok"]
    assert list(pd.read_csv(outdir / "wide_metrics_table.csv")["tokenizer"]) == ["tok_ok"]


# ── resume must never destroy existing output rows ─────────────────────────────
def _complete_outdir(tmp_path: Path, monkeypatch, capsys, slugs: list[str]) -> tuple[Path, Path]:
    """Run the builder to completion over `slugs`; return (outdir, sheet)."""
    outdir = tmp_path / "out"
    config_root = tmp_path / "config"
    config_root.mkdir(exist_ok=True)
    sheet = _tracker_csv(tmp_path / "tracker.csv", slugs)
    monkeypatch.setattr("mlflow.artifacts.download_artifacts", _stub_download)
    monkeypatch.setattr(bat, "flatten_tokenizer_metrics", lambda *a, **k: {"raw/metric": 1.0})

    def _run_variant(paths, out_dir, *, wire_m2, config_root, **_):
        # The real run_variant reads its return value back out of this file, so the
        # stub has to leave one behind too.
        cs = _cs()
        (Path(out_dir) / "category_scores.json").write_text(json.dumps(cs))
        return cs

    monkeypatch.setattr(bat, "run_variant", _run_variant)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_aggregation_tables.py",
            "--sheet",
            str(sheet),
            "--outdir",
            str(outdir),
            "--cache",
            str(tmp_path / "cache"),
            "--config-root",
            str(config_root),
        ],
    )
    bat.main()
    capsys.readouterr()
    assert len(pd.read_csv(outdir / "category_scores.csv")) == len(slugs)
    return outdir, sheet


def test_limit_run_does_not_truncate_existing_csv_rows(tmp_path: Path, monkeypatch, capsys):
    """`--limit 1` on a complete outdir must not shrink the headline table.

    Resume must not drop preloaded CSV rows it does not reprocess (those whose
    scores/<slug>.json is missing). With --limit (which docs advertise
    as the smoke test) or a `break`, the rest would never be re-appended — a 3-row
    category_scores.csv would come back with 1 row, silently, for any outdir with
    an empty scores/.
    """
    outdir, sheet = _complete_outdir(tmp_path, monkeypatch, capsys, ["tok_a", "tok_b", "tok_c"])

    # No scores/, so all three must reprocess.
    shutil.rmtree(outdir / "scores")

    monkeypatch.setattr(sys, "argv", sys.argv + ["--limit", "1"])
    bat.main()
    capsys.readouterr()

    assert sorted(pd.read_csv(outdir / "category_scores.csv")["tokenizer"]) == [
        "tok_a",
        "tok_b",
        "tok_c",
    ]
    assert sorted(pd.read_csv(outdir / "wide_metrics_table.csv")["tokenizer"]) == [
        "tok_a",
        "tok_b",
        "tok_c",
    ]


def test_a_failing_reprocess_keeps_the_existing_good_csv_row(tmp_path: Path, monkeypatch, capsys):
    """A tokenizer that fails on a re-run must keep the row it already had.

    Same root cause as above: the preloaded row was dropped up front and only put back
    on success, so a transient failure deleted a good row from the shipped table.
    """
    outdir, sheet = _complete_outdir(tmp_path, monkeypatch, capsys, ["tok_a", "tok_b"])
    shutil.rmtree(outdir / "scores")

    # tok_a fails, tok_b succeeds. The success is what triggers write_wide/write_scores,
    # so tok_a's already-good row has to survive that write.
    def _only_tok_a_explodes(paths, out_dir, *, wire_m2, config_root, **_):
        if Path(out_dir).parent.name == "tok_a":
            raise RuntimeError("aggregation exploded")
        return _cs()

    monkeypatch.setattr(bat, "run_variant", _only_tok_a_explodes)
    bat.main()
    capsys.readouterr()

    assert sorted(pd.read_csv(outdir / "category_scores.csv")["tokenizer"]) == ["tok_a", "tok_b"]
    assert sorted(pd.read_csv(outdir / "wide_metrics_table.csv")["tokenizer"]) == [
        "tok_a",
        "tok_b",
    ]


def test_compare_m2_over_a_plain_outdir_keeps_the_existing_scores_json(
    tmp_path: Path, monkeypatch, capsys
):
    """scores/ mirrors category_scores.csv, so reconciliation must key on THAT file.

    Keying the keep-set on the three-way intersection (wide n main n no_m2) meant that
    adding --compare-m2 to an outdir built without it collapsed the set to empty and
    deleted every scores/*.json at startup — while logging 'no row in
    category_scores.csv' about rows that were right there.
    """
    outdir, sheet = _complete_outdir(tmp_path, monkeypatch, capsys, ["tok_a", "tok_b", "tok_c"])
    assert len(list((outdir / "scores").glob("*.json"))) == 3

    monkeypatch.setattr(sys, "argv", sys.argv + ["--compare-m2", "--limit", "1"])
    bat.main()
    capsys.readouterr()

    assert sorted(p.name for p in (outdir / "scores").glob("*.json")) == [
        "tok_a.json",
        "tok_b.json",
        "tok_c.json",
    ]
    assert sorted(pd.read_csv(outdir / "category_scores.csv")["tokenizer"]) == [
        "tok_a",
        "tok_b",
        "tok_c",
    ]
    assert "no row in category_scores.csv" not in (outdir / "run_log.txt").read_text()


def test_summary_warns_about_csv_rows_with_no_scores_json(tmp_path: Path, monkeypatch, capsys):
    """The one-directional guarantee permits a CSV row with no JSON. The run that
    creates that state has to say so, or a later --scores-dir table is quietly short."""
    outdir, sheet = _complete_outdir(tmp_path, monkeypatch, capsys, ["tok_a", "tok_b", "tok_c"])
    shutil.rmtree(outdir / "scores")

    monkeypatch.setattr(sys, "argv", sys.argv + ["--limit", "1"])
    bat.main()
    capsys.readouterr()

    log_text = (outdir / "run_log.txt").read_text()
    assert "WARNING" in log_text
    assert "--scores-dir" in log_text
    # names the tokenizers a --scores-dir reader would silently drop
    assert "tok_b" in log_text.split("WARNING")[-1]
    assert "tok_c" in log_text.split("WARNING")[-1]


def test_a_score_record_raise_costs_one_tokenizer_not_the_batch(
    tmp_path: Path, monkeypatch, capsys
):
    """score_record() walks an untrusted payload, so it belongs inside the same
    logged-and-continue try as run_variant. Outside it, one malformed
    category_scores.json aborted the whole run — after 23 good tokenizers."""
    outdir = tmp_path / "out"
    config_root = tmp_path / "config"
    config_root.mkdir()
    sheet = _tracker_csv(tmp_path / "tracker.csv", ["tok_bad", "tok_ok"])

    monkeypatch.setattr("mlflow.artifacts.download_artifacts", _stub_download)
    monkeypatch.setattr(bat, "flatten_tokenizer_metrics", lambda *a, **k: {"raw/metric": 1.0})
    monkeypatch.setattr(bat, "run_variant", lambda *a, **k: _cs())

    real = bat.score_record

    def _explodes_on_tok_bad(slug, cs):
        if slug == "tok_bad":
            raise TypeError("string indices must be integers")
        return real(slug, cs)

    monkeypatch.setattr(bat, "score_record", _explodes_on_tok_bad)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_aggregation_tables.py",
            "--sheet",
            str(sheet),
            "--outdir",
            str(outdir),
            "--cache",
            str(tmp_path / "cache"),
            "--config-root",
            str(config_root),
        ],
    )

    bat.main()  # must not raise
    capsys.readouterr()

    assert list(pd.read_csv(outdir / "category_scores.csv")["tokenizer"]) == ["tok_ok"]
    assert (
        "DONE: 1 processed, 0 resumed-skip, 0 no-recon, 1 failed"
        in (outdir / "run_log.txt").read_text()
    )


def test_missing_scores_json_is_rescored_not_recovered_from_work(
    tmp_path: Path, monkeypatch, capsys
):
    """A missing ``scores/<slug>.json`` must be RE-SCORED, never copied out of ``_work/``.

    ``_work/<tok>/with_m2/category_scores.json`` is whatever the scorer wrote the day
    that directory was last touched. Recovering it marked the tokenizer complete and so
    suppressed the very re-score that a scorer change exists to deliver: a ``_work/``
    document written by an older scorer would carry stale decoding numbers (e.g. POYO-only
    instead of family-weighted) into the shipped table with `0 processed` and no
    warning. Aggregation is JSON-in/JSON-out off cached artifacts — re-scoring the whole
    24-tokenizer cohort is seconds, not hours.

    ``_work/`` here holds a deliberately different ``decoding`` value from the one the
    current scorer produces, so a copy and a re-score are distinguishable by number.
    """
    outdir, sheet = _complete_outdir(tmp_path, monkeypatch, capsys, ["tok_a", "tok_b"])
    shutil.rmtree(outdir / "scores")  # _work/ deliberately left in place

    stale = _cs()
    stale["scores"]["decoding"] = 0.8483  # POYO-only, as an older scorer wrote it
    for tok in ("tok_a", "tok_b"):
        (outdir / "_work" / tok / "with_m2" / "category_scores.json").write_text(json.dumps(stale))

    # Record rather than raise: process_variants catches broad `Exception`, so an
    # AssertionError thrown in here would be swallowed and merely counted as a failure.
    rescored: list[str] = []

    def _record_rescore(paths, out_dir, *, wire_m2, config_root, **_):
        rescored.append(Path(out_dir).parent.name)
        cs = _cs()
        cs["scores"]["decoding"] = 0.5412  # family-weighted poyo+ridge, post-fix
        (Path(out_dir) / "category_scores.json").write_text(json.dumps(cs))
        return cs

    monkeypatch.setattr(bat, "run_variant", _record_rescore)
    bat.main()
    capsys.readouterr()

    assert sorted(rescored) == ["tok_a", "tok_b"]
    assert "DONE: 2 processed, 0 resumed-skip" in (outdir / "run_log.txt").read_text()

    assert sorted(p.name for p in (outdir / "scores").glob("*.json")) == [
        "tok_a.json",
        "tok_b.json",
    ]
    # Both the headline table and the persisted documents carry the CURRENT scorer's
    # number, not the one that was sitting in _work/.
    scores = pd.read_csv(outdir / "category_scores.csv")
    assert list(scores["decoding"]) == [pytest.approx(0.5412)] * 2
    for tok in ("tok_a", "tok_b"):
        served = json.loads((outdir / "scores" / f"{tok}.json").read_text())
        assert served["scores"]["decoding"] == pytest.approx(0.5412)


def test_stale_scores_json_is_dropped_when_the_wide_flatten_fails(
    tmp_path: Path, monkeypatch, capsys
):
    """scores/ must not carry a tokenizer that category_scores.csv does not.

    _discard_scores_json only ran inside process_variants, but main() bails earlier on a
    download failure and on a wide-flatten failure. Force-redo a tokenizer the documented
    way (delete its CSV rows) and fail the flatten: scores/ kept tok_a while the CSV had
    only tok_b, and load_scores(scores_dir=...) then served a tokenizer in no table.
    """
    outdir, sheet = _complete_outdir(tmp_path, monkeypatch, capsys, ["tok_a", "tok_b"])

    # The documented force-redo: delete the tokenizer's rows from the output CSVs.
    for csv in ("category_scores.csv", "wide_metrics_table.csv"):
        df = pd.read_csv(outdir / csv)
        df[df["tokenizer"] != "tok_a"].to_csv(outdir / csv, index=False)

    def _boom(*a, **k):
        raise RuntimeError("wide flatten exploded")

    monkeypatch.setattr(bat, "flatten_tokenizer_metrics", _boom)
    bat.main()
    capsys.readouterr()

    assert list(pd.read_csv(outdir / "category_scores.csv")["tokenizer"]) == ["tok_b"]
    assert [p.name for p in (outdir / "scores").glob("*.json")] == ["tok_b.json"]


def test_colliding_tracker_slugs_are_warned_about(tmp_path: Path, monkeypatch, capsys):
    """Two tracker names folding to one slug overwrite each other's JSON, and the reader
    then hands the same _scores_data to both rows. Cheap to detect, invisible otherwise."""
    outdir = tmp_path / "out"
    config_root = tmp_path / "config"
    config_root.mkdir()
    sheet = _tracker_csv(tmp_path / "tracker.csv", ["tok.a", "tok-a"])  # both -> "tok_a"

    monkeypatch.setattr("mlflow.artifacts.download_artifacts", _stub_download)
    monkeypatch.setattr(bat, "flatten_tokenizer_metrics", lambda *a, **k: {"raw/metric": 1.0})
    monkeypatch.setattr(bat, "run_variant", lambda *a, **k: _cs())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_aggregation_tables.py",
            "--sheet",
            str(sheet),
            "--outdir",
            str(outdir),
            "--cache",
            str(tmp_path / "cache"),
            "--config-root",
            str(config_root),
        ],
    )
    bat.main()
    capsys.readouterr()

    log_text = (outdir / "run_log.txt").read_text()
    assert "slug collision" in log_text
    assert "tok_a" in log_text


def test_runner_output_feeds_load_scores_without_mlflow(tmp_path: Path, monkeypatch, capsys):
    """End to end: without --log-mlflow there is no mlflow_aggregation column, and
    <outdir>/scores/<slug>.json is what feeds build_latex_table and the three plotters.
    """
    from scripts.analysis import data

    outdir = tmp_path / "out"
    config_root = tmp_path / "config"
    config_root.mkdir()
    sheet = _tracker_csv(tmp_path / "tracker.csv", ["tok_bad", "tok_ok"])

    monkeypatch.setattr("mlflow.artifacts.download_artifacts", _stub_download)
    monkeypatch.setattr(bat, "flatten_tokenizer_metrics", lambda *a, **k: {"raw/metric": 1.0})

    def _run_variant(paths, out_dir, *, wire_m2, config_root, **_):
        if Path(out_dir).parent.name == "tok_bad":
            raise RuntimeError("aggregation exploded")
        return _cs(latent=0.48)

    monkeypatch.setattr(bat, "run_variant", _run_variant)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_aggregation_tables.py",
            "--sheet",
            str(sheet),
            "--outdir",
            str(outdir),
            "--cache",
            str(tmp_path / "cache"),
            "--config-root",
            str(config_root),
        ],
    )
    bat.main()
    capsys.readouterr()

    def _boom(*a, **k):
        raise AssertionError("the local route must not contact MLflow")

    monkeypatch.setattr(data.mlflow, "set_tracking_uri", _boom)
    monkeypatch.setattr(data.tracker_io, "load_json_artifact", _boom)

    df = data.load_scores(sheet, scores_dir=outdir / "scores")

    # only the tokenizer that actually succeeded, matching category_scores.csv
    assert list(df["tokenizer"]) == ["tok_ok"]
    assert list(pd.read_csv(outdir / "category_scores.csv")["tokenizer"]) == ["tok_ok"]
    assert df.iloc[0]["latent_structure_and_robustness"] == pytest.approx(0.48)
    # per-metric diagnostics reached the plotters, not just the seven scalars
    assert df.iloc[0]["_scores_data"]["diagnostics"]["latent_structure_and_robustness"][
        "per_metric"
    ]["m2:acg_correlation/eps_0.1"]["raw_value"] == pytest.approx(0.77)
    # ...and the failed tokenizer left no stale JSON for a later reader to trust
    assert not (outdir / "scores" / "tok_bad.json").exists()


# ── cross-tokenizer comparability check, end to end (real NTTAggregationJob) ──────
def _comparability_download(uri: str, dst_path: str) -> str:
    """Tiny artifacts for two tokenizers: recon runs `rec_a`/`rec_b` differ in their
    session set and max_eval_seconds; only tok_a has an M1 run (`m1_a`)."""
    run = uri.split("/")[2]
    name = uri.rsplit("/", 1)[1]
    local = Path(dst_path) / name
    if run.startswith("rec_"):
        sessions = ["s1", "s2"] if run == "rec_a" else ["s1", "s3"]
        payload = {
            "scored_window_seconds": 1.0,
            "comparison_vs_poisson": {},
            "compression": {"compression_rate": 0.5},
            "per_session": dict.fromkeys(sessions, {}),
            "aggregation_strategy": "mean",
            "split": "val",
            "aggregated": {
                "results": {},
                "config": {"max_eval_seconds": 600.0 if run == "rec_a" else None},
            },
        }
        local.write_text(json.dumps(payload))
    elif name == "eval_report.json":
        components = (
            "robustness_input/window_shift|max_shift_ms=3.0",
            "robustness_input/jitter|k_ms=1.0|p=0.2",
            "robustness_input/drop_and_add|fraction=0.02",
        )
        report = {
            "results": {c: {"raw_value": 0.5} for c in components},
            "config": {"n_replicas": 10, "scored_window_seconds": 1.0},
        }
        local.write_text(json.dumps(report))
    elif name == "results_per_session.csv":
        local.write_text("session_id,metric\ns1,x\n")
    else:
        raise FileNotFoundError(uri)
    return str(local)


def _comparability_uris(row: dict, **_) -> dict:
    letter = row["tokenizer"][-1]
    uris = {"recon": f"mlflow-artifacts:/1/rec_{letter}/artifacts/results.json"}
    if letter == "a":
        uris["m1"] = "mlflow-artifacts:/1/m1_a/artifacts/eval_report.json"
    return uris


def _run_builder(tmp_path: Path, monkeypatch, slugs: list[str], *extra_args: str) -> str:
    sheet = _tracker_csv(tmp_path / "tracker.csv", slugs)
    config_root = tmp_path / "config"
    config_root.mkdir(exist_ok=True)
    monkeypatch.setattr("mlflow.artifacts.download_artifacts", _comparability_download)
    monkeypatch.setattr(
        "mlflow.artifacts.list_artifacts",
        _fake_listing(["results_per_session.csv"]),
    )
    monkeypatch.setattr(bat.tracker_io, "row_to_source_uris", _comparability_uris)
    monkeypatch.setattr(bat, "flatten_tokenizer_metrics", lambda *a, **k: {"raw/metric": 1.0})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "builder",
            "--sheet",
            str(sheet),
            "--outdir",
            str(tmp_path / "out"),
            "--cache",
            str(tmp_path / "cache"),
            "--config-root",
            str(config_root),
            *extra_args,
        ],
    )
    bat.main()
    return (tmp_path / "out" / "run_log.txt").read_text()


def test_builder_raises_when_some_aspect_cannot_be_checked(tmp_path, monkeypatch, capsys):
    from ntt.evaluation.aggregation.comparability import UncheckedComparabilityError

    # The M1 run saves no cross-session aggregation strategy.
    with pytest.raises(UncheckedComparabilityError) as excinfo:
        _run_builder(tmp_path, monkeypatch, ["tok_a", "tok_b"])
    message = str(excinfo.value)
    assert "m1_aggregation_strategy for ['tok_a']" in message
    assert "Pass --common-parts-only to continue with a warning instead" in message
    capsys.readouterr()


def test_builder_warns_when_tokenizers_were_averaged_differently(tmp_path, monkeypatch, capsys):
    log = _run_builder(tmp_path, monkeypatch, ["tok_a", "tok_b"], "--common-parts-only")
    assert "WARNING: could NOT check" in log and "m1_aggregation_strategy for ['tok_a']" in log
    warning = log.split("WARNING: tokenizers in this table were averaged DIFFERENTLY")[1]
    warning = warning.split("WARNING: could NOT check")[0]
    assert "recon_sessions (affects" in warning
    assert "recon_settings.max_eval_seconds" in warning and "600.0: ['tok_a']" in warning
    # tok_b has no latent score at all (no M1, M2 or structure metric), so the latent
    # bucket is not compared for it.
    assert "m1_provenance" not in warning
    # tok_a's M1 sessions came from the run's results_per_session.csv: no note for them.
    assert "m1_sessions for" not in log
    capsys.readouterr()


def test_builder_rescores_rows_without_a_comparability_version(tmp_path, monkeypatch, capsys):
    _run_builder(tmp_path, monkeypatch, ["tok_a"])
    # Turn the finished row into one without a comparability version.
    scores_csv = tmp_path / "out" / "category_scores.csv"
    df = pd.read_csv(scores_csv)
    # Keep the recorded tracker links so that only the missing comparability version causes a
    # rescore.
    df[[c for c in df.columns if c not in ("tokenizer", "source_uris", *bat.SEVEN)]] = None
    df.to_csv(scores_csv, index=False)

    # --common-parts-only: these fixtures' M1 run saves no aggregation strategy (see above).
    log = _run_builder(tmp_path, monkeypatch, ["tok_a", "tok_b"], "--common-parts-only")
    assert "RESCORE tok_a: scored with comparability version none" in log
    assert "tok_a: SKIP (already done)" not in log
    capsys.readouterr()


def test_resume_rescores_a_tokenizer_whose_tracker_link_changed(tmp_path, monkeypatch, capsys):
    """A stored score must not survive the tracker pointing at a different run."""
    compression_by_run = {"aaa111": 0.5, "bbb222": 0.9}  # old, new run id

    def download(uri: str, dst_path: str) -> str:
        run_id = uri.split("/")[2]
        local = Path(dst_path) / "results.json"
        payload = {
            "scored_window_seconds": 1.0,
            "comparison_vs_poisson": {},
            "compression": {"compression_rate": compression_by_run[run_id]},
        }
        local.write_text(json.dumps(payload))
        return str(local)

    def write_tracker(run_id: str) -> Path:
        sheet = tmp_path / "tracker.csv"
        sheet.write_text(
            f"tokenizer,mlflow_recon\ntok,https://mlflow.internal/#/experiments/1/runs/{run_id}\n"
        )
        return sheet

    monkeypatch.setattr("mlflow.artifacts.download_artifacts", download)
    monkeypatch.setattr(bat, "flatten_tokenizer_metrics", lambda *a, **k: {"raw/metric": 1.0})
    (tmp_path / "config").mkdir()

    def run() -> str:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "builder",
                "--sheet",
                str(tmp_path / "tracker.csv"),
                "--outdir",
                str(tmp_path / "out"),
                "--cache",
                str(tmp_path / "cache"),
                "--config-root",
                str(tmp_path / "config"),
            ],
        )
        bat.main()
        capsys.readouterr()
        return (tmp_path / "out" / "run_log.txt").read_text()

    write_tracker("aaa111")
    run()
    scores = pd.read_csv(tmp_path / "out" / "category_scores.csv")
    assert scores.loc[0, "compression"] == 0.5

    # Unchanged tracker -> resumed, not rescored.
    assert "tok: SKIP (already done)" in run()

    write_tracker("bbb222")
    log = run()
    assert "RESCORE tok: tracker links changed for ['recon']" in log
    scores = pd.read_csv(tmp_path / "out" / "category_scores.csv")
    assert list(scores["tokenizer"]) == ["tok"]
    assert scores.loc[0, "compression"] == 0.9
    assert "bbb222" in scores.loc[0, "source_uris"]


def test_resume_rescores_a_row_with_local_paths(tmp_path, monkeypatch, capsys):
    """Re-running an evaluation into the same folder keeps the tracker cell but changes the
    files, so a local row is scored again on every run (an MLflow row with unchanged links
    is not)."""
    _boom_mlflow(monkeypatch)
    monkeypatch.setattr(bat, "flatten_tokenizer_metrics", lambda *a, **k: {"raw/metric": 1.0})
    report = tmp_path / "outputs" / "eval" / "results.json"
    report.parent.mkdir(parents=True)
    (tmp_path / "tracker.csv").write_text("tokenizer,mlflow_recon\ntok,outputs/eval\n")
    (tmp_path / "config").mkdir()
    argv = ["builder", "--sheet", str(tmp_path / "tracker.csv"), "--outdir", str(tmp_path / "out")]
    argv += ["--cache", str(tmp_path / "cache"), "--config-root", str(tmp_path / "config")]
    monkeypatch.setattr(sys, "argv", argv)

    def run(compression: float) -> str:
        payload = {
            "scored_window_seconds": 1.0,
            "comparison_vs_poisson": {},
            "compression": {"compression_rate": compression},
        }
        report.write_text(json.dumps(payload))
        bat.main()
        capsys.readouterr()
        return (tmp_path / "out" / "run_log.txt").read_text()

    run(0.5)
    log = run(0.9)  # the evaluation re-ran into the same folder

    assert "RESCORE tok: local outputs are always re-scored" in log
    assert "tok: SKIP (already done)" not in log
    scores = pd.read_csv(tmp_path / "out" / "category_scores.csv")
    assert list(scores["tokenizer"]) == ["tok"]
    assert scores.loc[0, "compression"] == 0.9


# ── score parts some tokenizers lack: error, or --common-parts-only ──────────────
def _parts_download(uri: str, dst_path: str) -> str:
    """Both tokenizers have a structure-preservation metric (so both have a latent
    score); only tok_a has an M2 run, at the ε the scorer reads."""
    run = uri.split("/")[2]
    local = Path(dst_path) / uri.rsplit("/", 1)[1]
    if run.startswith("rec_"):
        structure = {"population_structure_preservation": {"raw_value": 0.4}}
        payload = {
            "scored_window_seconds": 1.0,
            "comparison_vs_poisson": {},
            "compression": {"compression_rate": 0.5},
            "aggregated": {"results": structure},
        }
    elif run == "m2_a":
        payload = {
            "config": {"scored_window_seconds": 1.0},
            "results": {
                "acg_correlation/eps_0.1": {"raw_value": 0.9, "metadata": {"epsilon": 0.1}}
            },
        }
    else:
        raise FileNotFoundError(uri)
    local.write_text(json.dumps(payload))
    return str(local)


def _parts_uris(row: dict, **_) -> dict:
    letter = row["tokenizer"][-1]
    uris = {"recon": f"mlflow-artifacts:/1/rec_{letter}/artifacts/results.json"}
    if letter == "a":
        uris["m2"] = "mlflow-artifacts:/1/m2_a/artifacts/eval_report.json"
    return uris


def _run_parts_builder(tmp_path: Path, monkeypatch, *extra_args: str) -> pd.DataFrame:
    sheet = _tracker_csv(tmp_path / "tracker.csv", ["tok_a", "tok_b"])
    (tmp_path / "config").mkdir(exist_ok=True)
    monkeypatch.setattr("mlflow.artifacts.download_artifacts", _parts_download)
    monkeypatch.setattr(bat.tracker_io, "row_to_source_uris", _parts_uris)
    monkeypatch.setattr(bat, "flatten_tokenizer_metrics", lambda *a, **k: {"raw/metric": 1.0})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "builder",
            "--sheet",
            str(sheet),
            "--outdir",
            str(tmp_path / "out"),
            "--cache",
            str(tmp_path / "cache"),
            "--config-root",
            str(tmp_path / "config"),
            *extra_args,
        ],
    )
    bat.main()
    return pd.read_csv(tmp_path / "out" / "category_scores.csv").set_index("tokenizer")


def test_tokenizer_lacking_a_part_gets_that_score_empty_and_a_warning(
    tmp_path, monkeypatch, capsys
):
    # The aspects these fixtures cannot check would stop the run; that check is not tested here.
    monkeypatch.setattr(bat, "raise_if_not_comparable", lambda *a, **k: None)
    scores = _run_parts_builder(tmp_path, monkeypatch)
    assert scores.loc["tok_a", "latent_structure_and_robustness"] == pytest.approx(0.65)
    assert pd.isna(scores.loc["tok_b", "latent_structure_and_robustness"])
    assert scores.loc["tok_b", "emptied_scores"] == "latent_structure_and_robustness (no m2)"
    saved = json.loads((tmp_path / "out" / "scores" / "tok_b.json").read_text())
    assert saved["scores"]["latent_structure_and_robustness"] is None
    log = (tmp_path / "out" / "run_log.txt").read_text()
    assert "WARNING: latent_structure_and_robustness left empty for ['tok_b']" in log

    # A resumed run scores tok_b again, in case the other tokenizers changed.
    _run_parts_builder(tmp_path, monkeypatch)
    log = (tmp_path / "out" / "run_log.txt").read_text()
    assert "RESCORE tok_b: latent_structure_and_robustness (no m2) was left empty" in log
    capsys.readouterr()


def test_common_parts_only_leaves_m2_out_of_every_score(tmp_path, monkeypatch, capsys):
    scores = _run_parts_builder(tmp_path, monkeypatch, "--common-parts-only")
    # Both tokenizers stay; M2 is off for both, and tok_a's score is recomputed without it
    # (structure alone: max(0.4, 0) = 0.4, instead of mean(0.4, 0.9) = 0.65).
    assert sorted(scores.index) == ["tok_a", "tok_b"]
    assert list(scores["m2_provenance"]) == ["off", "off"]
    assert list(scores["left_out_parts"]) == ["m2", "m2"]
    assert scores.loc["tok_a", "latent_structure_and_robustness"] == pytest.approx(0.4)
    log = (tmp_path / "out" / "run_log.txt").read_text()
    assert "all tokenizers stay in the table; ['m2'] is left out of every tokenizer's score" in log

    # A later run without the flag notices the rows were scored without M2.
    monkeypatch.setattr(bat, "raise_if_not_comparable", lambda *a, **k: None)
    _run_parts_builder(tmp_path, monkeypatch)
    log = (tmp_path / "out" / "run_log.txt").read_text()
    assert "RESCORE tok_a: scored without m2; rescoring all parts" in log
    capsys.readouterr()


class _RecordingJob:
    """Stands in for NTTAggregationJob: records its kwargs, writes an empty score file."""

    seen: dict = {}

    def __init__(self, **kwargs):
        _RecordingJob.seen = kwargs
        self._out = Path(kwargs["output_path"])

    def run(self):
        self._out.mkdir(parents=True, exist_ok=True)
        (self._out / "category_scores.json").write_text("{}")


def _run_variant(tmp_path: Path, **kwargs) -> dict:
    return bat.run_variant(
        {"recon": tmp_path / "results.json"},
        tmp_path / "out",
        wire_m2=True,
        config_root=tmp_path,
        left_out_parts=frozenset(),
        common_parts_only=False,
        **kwargs,
    )


def test_run_variant_passes_gen_split_to_the_job(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(bat, "NTTAggregationJob", _RecordingJob)
    _run_variant(tmp_path, gen_split="test")
    assert _RecordingJob.seen["gen_split"] == "test"


def test_run_variant_gen_split_defaults_to_val(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(bat, "NTTAggregationJob", _RecordingJob)
    _run_variant(tmp_path)
    assert _RecordingJob.seen["gen_split"] == "val"


def test_process_variants_forwards_gen_split(tmp_path: Path, monkeypatch):
    seen = []

    def _record(*args, gen_split, **kwargs):
        seen.append(gen_split)
        return _cs()

    monkeypatch.setattr(bat, "run_variant", _record)
    bat.process_variants(
        "tok",
        {},
        tmp_path / "work",
        None,
        [("with_m2", True)],
        tmp_path / "config",
        bat.RunLog(tmp_path / "log.txt"),
        gen_split="test",
    )
    assert seen == ["test"]


def test_cli_gen_split_reaches_the_scoring(tmp_path: Path, monkeypatch):
    tracker = tmp_path / "tracker.csv"
    tracker.write_text(
        "tokenizer,mlflow_recon\ntok,https://mlflow.example/#/experiments/1/runs/abc\n"
    )
    recon = tmp_path / "results.json"
    recon.write_text("{}")
    monkeypatch.setattr(bat.ArtifactCache, "fetch", lambda self, slug, key, uri: recon)
    monkeypatch.setattr("mlflow.artifacts.download_artifacts", _no_download)
    monkeypatch.setattr(bat, "flatten_tokenizer_metrics", lambda *a, **k: {})
    seen = []

    def _process_variants(*args, gen_split="val", **kwargs):
        seen.append(gen_split)
        return None  # "failed": nothing is written, the end-of-run checks see no rows

    monkeypatch.setattr(bat, "process_variants", _process_variants)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_aggregation_tables.py",
            "--sheet",
            str(tracker),
            "--outdir",
            str(tmp_path / "out"),
            "--gen-split",
            "test",
        ],
    )
    bat.main()
    assert seen == ["test"]


def test_resume_rescores_generalization_rebuilt_on_another_split():
    """A row whose generalization was rebuilt on another split than --gen-split, or before the
    split was recorded, is scored again; a precomputed block does not depend on the flag."""
    recs = [
        {"tokenizer": "on_val", "generalization_source": "recomputed_from_per_session(val)"},
        {"tokenizer": "on_test", "generalization_source": "recomputed_from_per_session(test)"},
        {"tokenizer": "unrecorded", "generalization_source": "recomputed_from_per_session"},
        {"tokenizer": "precomputed", "generalization_source": "precomputed"},
    ]
    done = {rec["tokenizer"] for rec in recs}
    lines: list[str] = []
    bat._drop_rows_on_another_gen_split(done, recs, "test", lines.append)
    assert done == {"on_test", "precomputed"}
    assert (
        "RESCORE on_val: generalization was recomputed_from_per_session(val), --gen-split is test"
        in lines
    )


def test_resume_rescores_rows_scored_with_another_comparability_version():
    recs = [
        {"tokenizer": "other", "comparability_version": str(bat.COMPARABILITY_VERSION + 1)},
        {"tokenizer": "current", "comparability_version": str(bat.COMPARABILITY_VERSION)},
        {"tokenizer": "unrecorded", "comparability_version": None},
    ]
    done = {rec["tokenizer"] for rec in recs}
    lines: list[str] = []
    bat._drop_rows_scored_by_another_comparability_version(done, recs, lines.append)
    assert done == {"current"}
    assert [line.split(":")[0] for line in lines] == ["RESCORE other", "RESCORE unrecorded"]


def _no_download(*args, **kwargs):
    raise OSError("no network in tests")


def _fake_download(calls: list[str]):
    def _download(uri, dst_path):
        calls.append(uri)
        target = Path(dst_path) / uri.rsplit("/", 1)[1]
        target.write_text("{}")
        return str(target)

    return _download


def _fake_listing(names: list[str]):
    def _list(artifact_uri):
        return [SimpleNamespace(path=name) for name in names]

    return _list


def _run_files_cache(tmp_path: Path) -> bat.ArtifactCache:
    return bat.ArtifactCache(tmp_path / "cache", bat.RunLog(tmp_path / "cache_log.txt"))


def test_fetch_run_files_gets_run_meta_next_to_a_recon_report(tmp_path: Path, monkeypatch):
    report = tmp_path / "results.json"
    report.write_text("{}")
    calls: list[str] = []
    monkeypatch.setattr("mlflow.artifacts.download_artifacts", _fake_download(calls))
    monkeypatch.setattr("mlflow.artifacts.list_artifacts", _fake_listing(["run_meta.json"]))
    bat.fetch_run_files(
        report,
        "mlflow-artifacts:/1/abc/artifacts/results.json",
        _run_files_cache(tmp_path),
        bat.RunLog(tmp_path / "l.txt"),
    )
    assert calls == ["mlflow-artifacts:/1/abc/artifacts/run_meta.json"]
    assert (tmp_path / "run_meta.json").is_file()


def test_fetch_run_files_gets_both_robustness_files(tmp_path: Path, monkeypatch):
    report = tmp_path / "eval_report.json"
    report.write_text("{}")
    calls: list[str] = []
    monkeypatch.setattr("mlflow.artifacts.download_artifacts", _fake_download(calls))
    monkeypatch.setattr(
        "mlflow.artifacts.list_artifacts",
        _fake_listing(["eval_report.json", "results_per_session.csv", "run_meta.json"]),
    )
    bat.fetch_run_files(
        report,
        "mlflow-artifacts:/1/abc/artifacts/eval_report.json",
        _run_files_cache(tmp_path),
        bat.RunLog(tmp_path / "l.txt"),
    )
    assert [c.rsplit("/", 1)[1] for c in calls] == ["results_per_session.csv", "run_meta.json"]


def test_fetch_run_files_ignores_other_artifacts(tmp_path: Path, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr("mlflow.artifacts.download_artifacts", _fake_download(calls))
    bat.fetch_run_files(
        tmp_path / "downstream_encoding_results.json",
        "mlflow-artifacts:/1/abc/artifacts/downstream_encoding_results.json",
        _run_files_cache(tmp_path),
        bat.RunLog(tmp_path / "l.txt"),
    )
    assert calls == []


# ── collected table checks ────────────────────────────────────────────────────
def test_run_table_checks_runs_every_check_and_returns_the_first_failure(tmp_path, monkeypatch):
    def _fail(message):
        def check(*_a, **_k):
            raise ValueError(message)

        return check

    monkeypatch.setattr(bat, "raise_on_mixed_score_parts", _fail("parts"))
    monkeypatch.setattr(bat, "raise_if_not_comparable", _fail("not comparable"))

    checks, first = bat.run_table_checks(
        [],
        None,
        common_parts_only=False,
        log=bat.RunLog(tmp_path / "log.txt"),
    )

    assert str(first) == "parts"
    assert [(c["check"], c["passed"]) for c in checks] == [
        ("mixed_chunk_sizes", True),
        ("mixed_members", True),
        ("mixed_score_parts", False),
        ("not_comparable", False),
    ]
    assert checks[3]["message"] == "not comparable"


def test_table_checks_json_records_a_passing_table(tmp_path, monkeypatch, capsys):
    _run_parts_builder(tmp_path, monkeypatch, "--common-parts-only")
    doc = json.loads((tmp_path / "out" / "table_checks.json").read_text())
    assert doc["passed"] is True
    assert [c["check"] for c in doc["checks"]] == [
        "mixed_chunk_sizes",
        "mixed_members",
        "mixed_score_parts",
        "not_comparable",
    ]
    capsys.readouterr()


def test_a_failing_table_is_recorded_before_the_first_failure_is_raised(
    tmp_path, monkeypatch, capsys
):
    from ntt.evaluation.aggregation.comparability import UncheckedComparabilityError

    with pytest.raises(UncheckedComparabilityError):
        _run_parts_builder(tmp_path, monkeypatch)

    doc = json.loads((tmp_path / "out" / "table_checks.json").read_text())
    assert doc["passed"] is False
    failed = {c["check"]: c["message"] for c in doc["checks"] if not c["passed"]}
    assert "Could NOT check" in failed["not_comparable"]
    log = (tmp_path / "out" / "run_log.txt").read_text()
    assert "CHECK FAILED not_comparable" in log
    capsys.readouterr()


# ── --log-mlflow ──────────────────────────────────────────────────────────────
def _run_clean_builder(
    tmp_path: Path, monkeypatch, *extra_args: str, sheet: Path | None = None
) -> Path:
    """Two tokenizers scored as _cs(), deterministic and passing every table check."""
    sheet = sheet or _tracker_csv(tmp_path / "tracker.csv", ["tok_a", "tok_b"])
    (tmp_path / "config").mkdir(exist_ok=True)
    monkeypatch.setattr("mlflow.artifacts.download_artifacts", _stub_download)
    monkeypatch.setattr(bat, "flatten_tokenizer_metrics", lambda *a, **k: {"raw/metric": 1.0})
    monkeypatch.setattr(bat, "run_variant", lambda *a, **k: _cs())
    out = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "builder",
            "--sheet",
            str(sheet),
            "--outdir",
            str(out),
            "--cache",
            str(tmp_path / "cache"),
            "--config-root",
            str(tmp_path / "config"),
            *extra_args,
        ],
    )
    bat.main()
    return out


def _agg_urls(out: Path) -> dict[str, str]:
    df = pd.read_csv(out / "tracker_with_agg.csv", keep_default_na=False, dtype=str)
    return dict(zip(df["tokenizer"], df["mlflow_aggregation"], strict=True))


def _run_of(url: str):
    from mlflow.tracking import MlflowClient

    from ntt.evaluation.aggregation import tracker_io

    ids = tracker_io.parse_run_url(url)
    assert ids is not None, url
    return MlflowClient().get_run(ids[1])


def test_log_mlflow_publishes_every_row_tagged_passed(tmp_path, monkeypatch, capsys, mlflow_store):
    from mlflow.tracking import MlflowClient

    out = _run_clean_builder(
        tmp_path, monkeypatch, "--log-mlflow", "--mlflow-experiment", "agg-pass"
    )

    urls = _agg_urls(out)
    assert sorted(urls) == ["tok_a", "tok_b"]
    for tokenizer, url in urls.items():
        run = _run_of(url)
        assert run.data.tags["ntt.agg.table_checks"] == "passed"
        assert run.data.tags["ntt.agg.tokenizer"] == tokenizer
        artifacts = {a.path for a in MlflowClient().list_artifacts(run.info.run_id)}
        assert artifacts == {"category_scores.json", "table_checks.json"}
    assert "2 published, 0 reused, 0 failed" in (out / "run_log.txt").read_text()
    capsys.readouterr()


def test_log_mlflow_publishes_a_failing_table_and_tags_the_failure(
    tmp_path, monkeypatch, capsys, mlflow_store
):
    from ntt.evaluation.aggregation.comparability import UncheckedComparabilityError

    with pytest.raises(UncheckedComparabilityError):
        _run_parts_builder(
            tmp_path, monkeypatch, "--log-mlflow", "--mlflow-experiment", "agg-fail"
        )

    out = tmp_path / "out"
    doc = json.loads((out / "table_checks.json").read_text())
    failed = [c["check"] for c in doc["checks"] if not c["passed"]]
    assert "not_comparable" in failed
    urls = _agg_urls(out)
    assert sorted(urls) == ["tok_a", "tok_b"]
    for url in urls.values():
        assert _run_of(url).data.tags["ntt.agg.table_checks"] == "failed:" + ",".join(failed)
    capsys.readouterr()


def test_rerunning_the_builder_reuses_the_published_runs(
    tmp_path, monkeypatch, capsys, mlflow_store
):
    from mlflow.tracking import MlflowClient

    args = ("--log-mlflow", "--mlflow-experiment", "agg-twice")
    first = _agg_urls(_run_clean_builder(tmp_path, monkeypatch, *args))
    second = _agg_urls(_run_clean_builder(tmp_path, monkeypatch, *args))

    assert second == first
    experiment = MlflowClient().get_experiment_by_name("agg-twice")
    assert experiment is not None
    assert len(MlflowClient().search_runs([experiment.experiment_id])) == 2
    assert "0 published, 2 reused, 0 failed" in (tmp_path / "out" / "run_log.txt").read_text()
    capsys.readouterr()


def test_a_changed_table_verdict_gets_new_runs(tmp_path, monkeypatch, capsys, mlflow_store):
    # Same scores: a reused run would still carry the earlier build's verdict.
    args = ("--log-mlflow", "--mlflow-experiment", "agg-verdict")
    first = _agg_urls(_run_clean_builder(tmp_path, monkeypatch, *args))
    failing = [{"check": "mixed_score_parts", "passed": False, "message": "m"}]
    with monkeypatch.context() as patched:
        patched.setattr(bat, "run_table_checks", lambda *_a, **_k: (failing, None))
        second = _agg_urls(_run_clean_builder(tmp_path, monkeypatch, *args))
    third = _agg_urls(_run_clean_builder(tmp_path, monkeypatch, *args))

    assert set(second.values()).isdisjoint(first.values())
    assert set(third.values()).isdisjoint(second.values())
    for url in second.values():
        assert _run_of(url).data.tags["ntt.agg.table_checks"] == "failed:mixed_score_parts"
    for url in third.values():
        assert _run_of(url).data.tags["ntt.agg.table_checks"] == "passed"
    capsys.readouterr()


def test_one_failed_publish_costs_that_row_only(tmp_path, monkeypatch, capsys, mlflow_store):
    real_publish = bat.publish_category_scores

    def _flaky(path, **kwargs):
        if kwargs["run_name"] == "tok_a":
            raise ConnectionError("server went away")
        return real_publish(path, **kwargs)

    monkeypatch.setattr(bat, "publish_category_scores", _flaky)
    out = _run_clean_builder(
        tmp_path, monkeypatch, "--log-mlflow", "--mlflow-experiment", "agg-flaky"
    )

    urls = _agg_urls(out)
    assert urls["tok_a"] == ""
    assert "/runs/" in urls["tok_b"]
    log = (out / "run_log.txt").read_text()
    assert "PUBLISH FAILED tok_a: ConnectionError" in log
    assert "1 published, 0 reused, 1 failed" in log
    capsys.readouterr()


def test_without_log_mlflow_nothing_is_published(tmp_path, monkeypatch, capsys):
    def _boom(*_a, **_k):
        raise AssertionError("published without --log-mlflow")

    monkeypatch.setattr(bat, "publish_category_scores", _boom)
    out = _run_clean_builder(tmp_path, monkeypatch)
    assert not (out / "tracker_with_agg.csv").exists()
    assert json.loads((out / "table_checks.json").read_text())["passed"] is True
    capsys.readouterr()


def test_log_mlflow_without_a_tracking_uri_fails_before_any_download(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr("mlflow.artifacts.download_artifacts", _no_download)
    sheet = _tracker_csv(tmp_path / "tracker.csv", ["tok_a"])
    monkeypatch.setattr(
        sys,
        "argv",
        ["builder", "--sheet", str(sheet), "--outdir", str(tmp_path / "out"), "--log-mlflow"],
    )
    with pytest.raises(SystemExit) as excinfo:
        bat.main()
    assert excinfo.value.code == 2
    assert "MLFLOW_TRACKING_URI" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_a_run_without_log_mlflow_drops_an_earlier_runs_links(
    tmp_path, monkeypatch, capsys, mlflow_store
):
    out = _run_clean_builder(
        tmp_path, monkeypatch, "--log-mlflow", "--mlflow-experiment", "agg-stale"
    )
    assert (out / "tracker_with_agg.csv").exists()
    _run_clean_builder(tmp_path, monkeypatch)
    # Its links point at the runs of the earlier scores; a reader must not find them.
    assert not (out / "tracker_with_agg.csv").exists()
    capsys.readouterr()


def test_rerunning_on_the_last_tracker_with_agg_keeps_it(
    tmp_path, monkeypatch, capsys, mlflow_store
):
    out = _run_clean_builder(
        tmp_path, monkeypatch, "--log-mlflow", "--mlflow-experiment", "agg-input"
    )
    tracker = out / "tracker_with_agg.csv"
    before = tracker.read_text()

    _run_clean_builder(tmp_path, monkeypatch, sheet=tracker)
    assert tracker.read_text() == before
    _run_clean_builder(
        tmp_path, monkeypatch, "--log-mlflow", "--mlflow-experiment", "agg-input", sheet=tracker
    )
    assert tracker.read_text() == before
    capsys.readouterr()


def test_a_run_that_stops_before_the_checks_leaves_no_earlier_table_checks(
    tmp_path, monkeypatch, capsys
):
    out = _run_clean_builder(tmp_path, monkeypatch)
    assert json.loads((out / "table_checks.json").read_text())["passed"] is True

    def _crash(*_a, **_k):
        raise RuntimeError("crashed before the checks")

    monkeypatch.setattr(bat, "run_table_checks", _crash)
    with pytest.raises(RuntimeError):
        _run_clean_builder(tmp_path, monkeypatch)
    assert not (out / "table_checks.json").exists()
    capsys.readouterr()


def test_tracker_with_agg_replaces_stale_links_and_matches_stripped_names(tmp_path):
    fieldnames = ["tokenizer", "mlflow_recon", "mlflow_aggregation"]
    rows = [
        {"tokenizer": " tok_a ", "mlflow_recon": "r", "mlflow_aggregation": "old-a"},
        {"tokenizer": "tok_b", "mlflow_recon": "r", "mlflow_aggregation": "old-b"},
    ]
    bat.write_tracker_with_agg(fieldnames, rows, {"tok_a": "new-a"}, tmp_path / "t.csv")
    df = pd.read_csv(tmp_path / "t.csv", keep_default_na=False, dtype=str)
    assert list(df.columns) == fieldnames
    assert list(df["mlflow_aggregation"]) == ["new-a", ""]


def test_tracker_with_agg_appends_the_column(tmp_path):
    bat.write_tracker_with_agg(
        ["tokenizer"], [{"tokenizer": "tok_a"}], {"tok_a": "u"}, tmp_path / "t.csv"
    )
    df = pd.read_csv(tmp_path / "t.csv", keep_default_na=False, dtype=str)
    assert list(df.columns) == ["tokenizer", "mlflow_aggregation"]
    assert list(df["mlflow_aggregation"]) == ["u"]


# ── provenance and defaults ───────────────────────────────────────────────────
def test_run_variant_passes_ckpt_uri_and_leaves_m2_settings_to_the_job(tmp_path, monkeypatch):
    monkeypatch.setattr(bat, "NTTAggregationJob", _RecordingJob)
    _run_variant(tmp_path, ckpt_uri="mlflow-artifacts:/9/ck/artifacts/last.ckpt")
    assert _RecordingJob.seen["ckpt_uri"] == "mlflow-artifacts:/9/ck/artifacts/last.ckpt"
    assert "m2_metric" not in _RecordingJob.seen
    assert "m2_epsilon" not in _RecordingJob.seen


def test_cli_ckpt_mlflow_uri_reaches_the_scoring(tmp_path: Path, monkeypatch):
    tracker = tmp_path / "tracker.csv"
    tracker.write_text(
        "tokenizer,mlflow_recon,ckpt_mlflow_uri\n"
        "tok_a,https://mlflow.example/#/experiments/1/runs/abc,"
        " mlflow-artifacts:/9/ck/artifacts/last.ckpt \n"
        "tok_b,https://mlflow.example/#/experiments/1/runs/def,\n"
    )
    recon = tmp_path / "results.json"
    recon.write_text("{}")
    monkeypatch.setattr(bat.ArtifactCache, "fetch", lambda self, slug, key, uri: recon)
    monkeypatch.setattr("mlflow.artifacts.download_artifacts", _no_download)
    monkeypatch.setattr(bat, "flatten_tokenizer_metrics", lambda *a, **k: {})
    seen = {}

    def _process_variants(slug, *args, ckpt_uri=None, **kwargs):
        seen[slug] = ckpt_uri
        return None  # "failed": nothing is written, the end-of-run checks see no rows

    monkeypatch.setattr(bat, "process_variants", _process_variants)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_aggregation_tables.py",
            "--sheet",
            str(tracker),
            "--outdir",
            str(tmp_path / "out"),
        ],
    )
    bat.main()
    assert seen == {"tok_a": "mlflow-artifacts:/9/ck/artifacts/last.ckpt", "tok_b": None}


def test_both_read_routes_give_the_same_frame(tmp_path, monkeypatch, capsys, mlflow_store):
    """The plotting tools see the same frame from --scores-dir and from the published runs."""
    import mlflow.artifacts
    from scripts.analysis import data

    real_download = mlflow.artifacts.download_artifacts
    out = _run_clean_builder(
        tmp_path, monkeypatch, "--log-mlflow", "--mlflow-experiment", "agg-parity"
    )
    monkeypatch.setattr("mlflow.artifacts.download_artifacts", real_download)

    tracker = out / "tracker_with_agg.csv"
    local = data.load_scores(tracker, scores_dir=out / "scores")
    remote = data.load_scores(tracker)

    pd.testing.assert_frame_equal(local, remote)
    capsys.readouterr()


def test_main_fetches_the_recon_run_meta_before_scoring(tmp_path: Path, monkeypatch):
    """main() fetches run_meta.json next to the recon report: it supplies the input geometry
    and window settings a recon results.json does not save."""
    recon_uri = "mlflow-artifacts:/1/abc/artifacts/results.json"
    sheet = _tracker_csv(tmp_path / "tracker.csv", ["tok"])
    monkeypatch.setattr(bat.tracker_io, "row_to_source_uris", lambda _, **__: {"recon": recon_uri})
    calls: list[str] = []
    monkeypatch.setattr("mlflow.artifacts.download_artifacts", _fake_download(calls))
    monkeypatch.setattr("mlflow.artifacts.list_artifacts", _fake_listing(["run_meta.json"]))
    monkeypatch.setattr(bat, "flatten_tokenizer_metrics", lambda *a, **k: {"raw/metric": 1.0})
    recon_dirs: list[Path] = []

    def _run_variant(paths, out_dir, **_):
        recon_dirs.append(Path(paths["recon"]).parent)
        return _cs()

    monkeypatch.setattr(bat, "run_variant", _run_variant)
    argv = ["builder", "--sheet", str(sheet), "--outdir", str(tmp_path / "out")]
    monkeypatch.setattr(sys, "argv", [*argv, "--cache", str(tmp_path / "cache")])

    bat.main()

    assert calls == [recon_uri, "mlflow-artifacts:/1/abc/artifacts/run_meta.json"]
    assert recon_dirs and all((d / "run_meta.json").is_file() for d in recon_dirs)


# ── local paths in tracker cells ───────────────────────────────────────────────
def _boom_mlflow(monkeypatch):
    def _boom(*args, **kwargs):
        raise AssertionError("a local cell must not touch MLflow")

    for name in ("download_artifacts", "list_artifacts"):
        monkeypatch.setattr(f"mlflow.artifacts.{name}", _boom)
    monkeypatch.setattr("mlflow.get_tracking_uri", _boom)


def test_cache_fetch_resolves_a_local_directory_in_place(tmp_path: Path, monkeypatch):
    _boom_mlflow(monkeypatch)
    report = tmp_path / "eval" / "results.json"
    report.parent.mkdir()
    report.write_text("{}")
    cache = bat.ArtifactCache(tmp_path / "cache", bat.RunLog(tmp_path / "log.txt"))
    assert cache.fetch("tok", "recon", str(report.parent)) == report
    assert not (tmp_path / "cache" / "manifest.json").exists()


def test_fetch_run_files_leaves_a_local_report_alone(tmp_path: Path, monkeypatch):
    _boom_mlflow(monkeypatch)
    report = tmp_path / "results.json"
    report.write_text("{}")
    bat.fetch_run_files(
        report, str(report), _run_files_cache(tmp_path), bat.RunLog(tmp_path / "l.txt")
    )


def test_load_snr_report_unwraps_a_results_json(tmp_path: Path):
    inner = {"results": {"repeats_snr": {"raw_value": 1.5}}}
    wrapped = tmp_path / "results.json"
    wrapped.write_text(json.dumps({"aggregated": inner, "per_session": {}}))
    compact = tmp_path / "eval_report.json"
    compact.write_text(json.dumps(inner))
    assert bat.load_snr_report(wrapped) == inner
    assert bat.load_snr_report(compact) == inner


def test_main_reads_local_and_mlflow_rows_in_one_tracker(tmp_path: Path, monkeypatch):
    """A local cell is read in place with no MLflow call; the mlflow row next to it is
    still downloaded; each records its own source (the local one its absolute path)."""
    local_eval = tmp_path / "outputs" / "eval"
    local_eval.mkdir(parents=True)
    (local_eval / "results.json").write_text("{}")
    (tmp_path / "tracker.csv").write_text(
        "tokenizer,mlflow_recon\n"
        "loc,outputs/eval\n"
        "rem,https://mlflow.internal/#/experiments/1/runs/abc12\n"
    )
    calls: list[str] = []
    monkeypatch.setattr("mlflow.artifacts.download_artifacts", _fake_download(calls))
    monkeypatch.setattr("mlflow.artifacts.list_artifacts", _fake_listing([]))
    monkeypatch.setattr(bat, "flatten_tokenizer_metrics", lambda *a, **k: {"raw/metric": 1.0})
    recon_paths: dict[str, Path] = {}

    def _run_variant(paths, out_dir, **_):
        recon_paths[Path(out_dir).parent.name] = Path(paths["recon"])
        return _cs()

    monkeypatch.setattr(bat, "run_variant", _run_variant)
    argv = ["builder", "--sheet", str(tmp_path / "tracker.csv"), "--outdir", str(tmp_path / "out")]
    monkeypatch.setattr(sys, "argv", [*argv, "--cache", str(tmp_path / "cache")])

    bat.main()

    assert calls == ["mlflow-artifacts:/1/abc12/artifacts/results.json"], (
        "only the mlflow row downloads"
    )
    assert local_eval / "results.json" in recon_paths.values()
    scores = pd.read_csv(tmp_path / "out" / "category_scores.csv").set_index("tokenizer")
    assert json.loads(str(scores.loc["loc", "source_uris"])) == {"recon": str(local_eval)}
    assert "mlflow-artifacts:/1/abc12" in str(scores.loc["rem", "source_uris"])


def test_main_reports_a_missing_local_path_for_that_row_only(tmp_path: Path, monkeypatch):
    (tmp_path / "tracker.csv").write_text("tokenizer,mlflow_recon\nbad,outputs/nowhere\n")
    _boom_mlflow(monkeypatch)
    monkeypatch.setattr(bat, "run_variant", lambda *a, **k: _cs())
    argv = ["builder", "--sheet", str(tmp_path / "tracker.csv"), "--outdir", str(tmp_path / "out")]
    monkeypatch.setattr(sys, "argv", [*argv, "--cache", str(tmp_path / "cache")])
    try:
        bat.main()
    except SystemExit:
        pass
    log = (tmp_path / "out" / "run_log.txt").read_text()
    assert "bad" in log and str(tmp_path / "outputs" / "nowhere") in log
