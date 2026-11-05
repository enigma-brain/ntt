"""Tests for ntt.evaluation.aggregation.tracker_io."""

from __future__ import annotations

import json
from unittest.mock import ANY, patch

from ntt.evaluation.aggregation import tracker_io as tio


def test_parse_run_url_extracts_exp_and_run():
    url = "https://mlflow.x/#/experiments/474/runs/abc123def4567890abc123def4567890"
    assert tio.parse_run_url(url) == ("474", "abc123def4567890abc123def4567890")


def test_parse_run_url_returns_none_on_non_match():
    assert tio.parse_run_url("https://example.com/nope") is None
    assert tio.parse_run_url("") is None


def test_scores_slug_folds_case_and_punctuation():
    """Both build_aggregation_tables.py (writer) and scripts/analysis/data.py (reader)
    resolve <slug>.json through this, so the rule lives here rather than twice."""
    assert tio.scores_slug("wanx_A_vae.b1e-3 v2") == "wanx_a_vae_b1e_3_v2"
    assert tio.scores_slug("01_control_noncausal") == "01_control_noncausal"
    # leading/trailing separator runs are trimmed, not turned into empty segments
    assert tio.scores_slug("--tok--") == "tok"
    assert tio.scores_slug("") == ""


def test_artifact_uri_and_decoding_dir_uri():
    assert (
        tio.artifact_uri("474", "run", "results.json")
        == "mlflow-artifacts:/474/run/artifacts/results.json"
    )
    assert tio.decoding_dir_uri("474", "run") == "mlflow-artifacts:/474/run/artifacts"


def test_row_to_source_uris_full_row():
    row = {
        "mlflow_recon": "x/experiments/1/runs/aaaa",
        "mlflow_latent_input_corruption": "x/experiments/2/runs/bbbb",
        "mlflow_encoding": "x/experiments/3/runs/cccc",
        "mlflow_decoding": "x/experiments/4/runs/dddd",
    }
    out = tio.row_to_source_uris(row)
    assert out == {
        "recon": "mlflow-artifacts:/1/aaaa/artifacts/results.json",
        "m1": "mlflow-artifacts:/2/bbbb/artifacts/eval_report.json",
        "encoding": "mlflow-artifacts:/3/cccc/artifacts/downstream_encoding_results.json",
        "decoding": "mlflow-artifacts:/4/dddd/artifacts",
    }


def test_row_to_source_uris_maps_m2():
    """M2 resolves from the current tracker column ``mlflow_latent_latent_corruption``.

    Worth pinning: ``row.get(missing) or ""`` makes an unmatched column name
    indistinguishable from an empty cell, so a name mismatch drops M2 silently instead
    of raising.
    """
    row = {
        "mlflow_recon": "x/experiments/1/runs/aaaa",
        "mlflow_latent_input_corruption": "x/experiments/1/runs/eeee",
        "mlflow_latent_latent_corruption": "x/experiments/1/runs/bbbb",
    }
    uris = tio.row_to_source_uris(row)
    # De-normalized latent bucket: M1/M2 read the raw eval_report.json (no Poisson).
    assert uris["m1"].endswith("/1/eeee/artifacts/eval_report.json")
    assert uris["m2"].endswith("/1/bbbb/artifacts/eval_report.json")
    # The Poisson-reference run dir is not wired (the bucket isn't normalized).
    assert "poisson_reference_run" not in uris


def test_row_to_source_uris_recon_only():
    out = tio.row_to_source_uris({"mlflow_recon": "x/experiments/1/runs/aaaa"})
    assert out == {"recon": "mlflow-artifacts:/1/aaaa/artifacts/results.json"}


def test_row_to_source_uris_none_without_recon():
    assert tio.row_to_source_uris({"mlflow_encoding": "x/experiments/3/runs/cccc"}) is None


def test_read_tracker_rows_local_csv(tmp_path):
    f = tmp_path / "t.csv"
    f.write_text("tokenizer,mlflow_recon\nwanx,url1\ncosmos,url2\n")
    fieldnames, rows = tio.read_tracker_rows(str(f))
    assert fieldnames == ["tokenizer", "mlflow_recon"]
    assert rows == [
        {"tokenizer": "wanx", "mlflow_recon": "url1"},
        {"tokenizer": "cosmos", "mlflow_recon": "url2"},
    ]


def test_load_json_artifact_local(tmp_path):
    f = tmp_path / "d.json"
    f.write_text(json.dumps({"k": 1}))
    assert tio.load_json_artifact(str(f)) == {"k": 1}


def test_load_json_artifact_mlflow_uri_uses_dst_path(tmp_path):
    local = tmp_path / "dl.json"
    local.write_text(json.dumps({"m": True}))
    with patch("mlflow.artifacts.download_artifacts", return_value=str(local)) as mock_dl:
        out = tio.load_json_artifact("mlflow-artifacts:/1/r/artifacts/results.json")
    assert out == {"m": True}
    mock_dl.assert_called_once_with("mlflow-artifacts:/1/r/artifacts/results.json", dst_path=ANY)


def test_load_json_artifact_downloads_runs_uris(tmp_path):
    local = tmp_path / "category_scores.json"
    local.write_text(json.dumps({"a": 1}))
    with patch("mlflow.artifacts.download_artifacts", return_value=str(local)) as mock_dl:
        out = tio.load_json_artifact("runs:/abc/category_scores.json")
    assert out == {"a": 1}
    mock_dl.assert_called_once_with("runs:/abc/category_scores.json", dst_path=ANY)


# ── local paths in tracker cells ────────────────────────────────────────────────
import pytest  # noqa: E402


def _touch(path, text="{}"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_row_to_source_uris_local_cell_records_its_absolute_path(tmp_path):
    out = tio.row_to_source_uris({"mlflow_recon": str(tmp_path / "eval")})
    assert out == {"recon": str(tmp_path / "eval")}


def test_row_to_source_uris_relative_cell_resolves_against_base_dir(tmp_path):
    row = {"mlflow_recon": "outputs/eval", "mlflow_encoding": "./outputs/encoding"}
    out = tio.row_to_source_uris(row, base_dir=tmp_path)
    assert out == {
        "recon": str(tmp_path / "outputs/eval"),
        "encoding": str(tmp_path / "outputs/encoding"),
    }


def test_row_to_source_uris_relative_cell_without_base_dir_uses_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tio.row_to_source_uris({"mlflow_recon": "eval"})
    assert out == {"recon": str(tmp_path / "eval")}


def test_row_to_source_uris_mixes_mlflow_and_local_cells(tmp_path):
    row = {
        "mlflow_recon": "x/experiments/1/runs/aaaa",
        "mlflow_latent_input_corruption": str(tmp_path / "eval_m1"),
        "mlflow_latent_latent_corruption": "x/experiments/2/runs/bbbb",
        "mlflow_latent_snr": str(tmp_path / "eval_snr" / "results.json"),
        "mlflow_encoding": "x/experiments/3/runs/cccc",
        "mlflow_decoding": str(tmp_path / "decoding"),
    }
    assert tio.row_to_source_uris(row) == {
        "recon": "mlflow-artifacts:/1/aaaa/artifacts/results.json",
        "m1": str(tmp_path / "eval_m1"),
        "m2": "mlflow-artifacts:/2/bbbb/artifacts/eval_report.json",
        "snr": str(tmp_path / "eval_snr" / "results.json"),
        "encoding": "mlflow-artifacts:/3/cccc/artifacts/downstream_encoding_results.json",
        "decoding": str(tmp_path / "decoding"),
    }


def test_row_to_source_uris_still_ignores_an_unparseable_url():
    row = {"mlflow_recon": "x/experiments/1/runs/aaaa", "mlflow_encoding": "https://example.com/n"}
    assert set(tio.row_to_source_uris(row) or {}) == {"recon"}
    assert tio.row_to_source_uris({"mlflow_recon": "https://example.com/n"}) is None


def test_is_local_source():
    assert tio.is_local_source("/data/eval")
    assert not tio.is_local_source("mlflow-artifacts:/1/a/artifacts/results.json")
    assert not tio.is_local_source("runs:/abc/results.json")


# Layouts as the eval jobs write them: one flat output dir per axis.
@pytest.mark.parametrize(
    ("axis", "subdir", "report"),
    [
        ("recon", "eval", "results.json"),
        ("m1", "eval_m1", "eval_report.json"),
        ("m2", "eval_m2", "eval_report.json"),
        ("snr", "eval_snr", "results.json"),
        ("encoding", "encoding", "downstream_encoding_results.json"),
    ],
)
def test_resolve_local_artifact_directory_and_file(tmp_path, axis, subdir, report):
    expected = _touch(tmp_path / "output" / subdir / report)
    _touch(tmp_path / "output" / subdir / "run_meta.json")
    assert tio.resolve_local_artifact(str(tmp_path / "output" / subdir), axis) == expected
    # the job's output dir above it is enough while the match is unique
    assert tio.resolve_local_artifact(str(tmp_path / "output"), axis) == expected
    # the report file itself is used as given
    assert tio.resolve_local_artifact(str(expected), axis) == expected


def test_resolve_local_artifact_decoding_finds_the_cohort_record(tmp_path):
    parent = tmp_path / "output" / "decoding" / "ntt_downstream_decoding_tok"
    cell = _touch(
        parent / "ntt_downstream_decoding_tok__tok__c_2013_center_out" / "run_result.json"
    )
    record = _touch(
        parent / "ntt_downstream_decoding_tok__tok__cohort" / "downstream_decoding_results.json"
    )
    assert tio.resolve_local_artifact(str(tmp_path / "output"), "decoding") == record
    assert tio.resolve_local_artifact(str(record.parent), "decoding") == record
    assert cell.is_file()
    # a benchmark cell's own run record (same file name) is not the cohort's
    _touch(cell.parent / "csv_logs" / "version_0" / "downstream_decoding_results.json")
    assert tio.resolve_local_artifact(str(tmp_path / "output"), "decoding") == record


def test_resolve_local_artifact_snr_prefers_eval_report_next_to_results(tmp_path):
    report = _touch(tmp_path / "eval_snr" / "eval_report.json")
    _touch(tmp_path / "eval_snr" / "results.json")
    assert tio.resolve_local_artifact(str(tmp_path / "eval_snr"), "snr") == report


def test_resolve_local_artifact_directory_itself_wins_over_nested_match(tmp_path):
    top = _touch(tmp_path / "results.json")
    _touch(tmp_path / "older" / "results.json")
    assert tio.resolve_local_artifact(str(tmp_path), "recon") == top


def test_resolve_local_artifact_ambiguous_names_every_match(tmp_path):
    a = _touch(tmp_path / "eval" / "results.json")
    b = _touch(tmp_path / "eval_snr" / "results.json")
    with pytest.raises(ValueError, match="more than one") as exc:
        tio.resolve_local_artifact(str(tmp_path), "recon")
    assert str(a) in str(exc.value) and str(b) in str(exc.value)


def test_resolve_local_artifact_missing_path_names_the_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="nope"):
        tio.resolve_local_artifact(str(tmp_path / "nope"), "recon")


def test_resolve_local_artifact_empty_directory_names_what_was_looked_for(tmp_path):
    with pytest.raises(FileNotFoundError, match="eval_report.json") as exc:
        tio.resolve_local_artifact(str(tmp_path), "m1")
    assert str(tmp_path) in str(exc.value)


@pytest.mark.parametrize("junk", ["n/a", "-", "TODO", "none"])
def test_placeholder_cells_are_ignored_not_read_as_paths(junk):
    from ntt.evaluation.aggregation.tracker_io import row_to_source_uris

    run = "https://mlflow.example.org/#/experiments/1/runs/abc"
    uris = row_to_source_uris({"mlflow_recon": run, "mlflow_encoding": junk})
    assert "encoding" not in uris
    assert row_to_source_uris({"mlflow_recon": junk}) is None
