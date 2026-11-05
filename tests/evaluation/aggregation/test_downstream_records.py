"""Tests for ntt.evaluation.aggregation.downstream_records.

The ``tests/data/downstream_records/*.json`` records are real writer output, copied
unchanged from ntt-downstream ``tests/fixtures/results_v2/``.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, patch

import pytest

from ntt.evaluation.aggregation import downstream_records as dr

FIXTURES = Path(__file__).parents[2] / "data" / "downstream_records"


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


# ── v2 records ───────────────────────────────────────────────────────────────


def test_encoding_run_reads_the_test_block():
    view = dr.load_downstream_record(str(FIXTURES / "encoding_run.json"), "encoding")
    assert view.split == "test"
    assert view.complete is True
    # test (0.19), not val_final (0.21) or the periodic val blocks.
    assert view.aggregated["nc_r2"]["value"] == pytest.approx(0.19)
    assert view.aggregated["nc_r2"]["n"] == 1
    assert view.aggregated["nc_r2"]["aggregated_over"] == "sessions:mean_of_means"
    assert view.per_session == {"PE_201021": {"nc_r2": 0.19, "rate_pearson": 0.28}}
    assert (
        view.provenance["tokenizer_checkpoint"] == "mlflow-artifacts:/1/abc/artifacts/model.ckpt"
    )
    assert view.provenance["ntt_version"] == "0.0.0-fixture"
    assert view.provenance["ntt_path"] == "/fixture/ntt/src/ntt/__init__.py"
    assert view.provenance["ntt_commit"] == "0" * 40
    assert view.provenance["git_sha"] == "fixture-git-sha"
    assert view.provenance["final_eval"] == {"status": "ok"}


def test_encoding_run_without_test_block_has_no_score_and_keeps_the_diagnostic():
    view = dr.load_downstream_record(
        str(FIXTURES / "encoding_run_final_eval_failed.json"), "encoding"
    )
    assert view.split == "test"
    assert view.aggregated == {}
    assert view.per_session == {}
    assert view.provenance["final_eval"] == {
        "status": "failed",
        "error": "RuntimeError('cuda oom')",
    }


def test_complete_decoding_cohort():
    view = dr.load_downstream_record(str(FIXTURES / "decoding_cohort.json"), "decoding")
    assert view.split == "test"
    assert view.complete is True
    assert view.aggregated["r2"]["value"] == pytest.approx(0.83)
    assert view.aggregated["r2"]["aggregated_over"] == "sessions:mean"
    assert view.per_session == {
        "c_20131003_center_out_reaching": {"r2": 0.85},
        "t_20130819_center_out_reaching": {"r2": 0.81},
    }
    assert view.provenance["missing"] == []
    assert view.provenance["mixed_provenance"] is False


def test_incomplete_decoding_cohort_is_not_complete_and_names_the_missing_sessions():
    view = dr.load_downstream_record(str(FIXTURES / "decoding_cohort_incomplete.json"), "decoding")
    assert view.complete is False
    assert view.provenance["missing"] == ["j_20160405_center_out_reaching"]
    assert view.provenance["found"] == [
        "c_20131003_center_out_reaching",
        "t_20130819_center_out_reaching",
    ]


def test_session_summary_lists_the_sessions_behind_the_view():
    view = dr.load_downstream_record(str(FIXTURES / "decoding_cohort.json"), "decoding")
    assert view.session_summary("r2") == {
        "session_ids": ["c_20131003_center_out_reaching", "t_20130819_center_out_reaching"],
        "n_sessions": 2,
        "aggregation_strategy": "sessions:mean",
    }


def test_session_summary_counts_only_sessions_with_a_value_for_the_metric():
    """A session whose metric is null did not contribute to the score, so it must not
    count toward the comparability check's session set."""
    record = _fixture("encoding_run")
    record["metrics"]["test"]["per_session"]["S2"] = {"nc_r2": None, "rate_pearson": 0.3}
    summary = dr.view_from_record(record, "encoding").session_summary("nc_r2")
    assert summary["session_ids"] == ["PE_201021"]
    assert summary["n_sessions"] == 1


def test_schema_version_mismatch_raises():
    record = _fixture("encoding_run") | {"schema_version": 3}
    with pytest.raises(ValueError, match="schema_version 3"):
        dr.view_from_record(record, "encoding")


def test_task_mismatch_raises():
    with pytest.raises(ValueError, match="'decoding' record"):
        dr.view_from_record(_fixture("decoding_cohort"), "encoding")


def test_decoding_run_record_is_refused_with_a_pointer_to_the_cohort():
    """ntt scores decoding from the benchmark parent's cohort, never one cell's record."""
    with pytest.raises(ValueError, match="kind 'run'.*cohort"):
        dr.view_from_record(_fixture("decoding_run"), "decoding")


def test_last_checkpoint_encoding_score_warns(caplog):
    record = _fixture("encoding_run")
    record["metadata"]["final_eval"]["ckpt_kind"] = "last"
    with caplog.at_level(logging.WARNING):
        view = dr.view_from_record(record, "encoding")
    assert view.provenance["final_eval"]["ckpt_kind"] == "last"
    assert any("not val-selected" in r.message for r in caplog.records)


def test_mixed_provenance_cohort_warns_but_is_read(caplog):
    record = _fixture("decoding_cohort")
    record["metadata"]["cohort"]["mixed_provenance"] = True
    with caplog.at_level(logging.WARNING):
        view = dr.view_from_record(record, "decoding")
    assert view.aggregated["r2"]["value"] == pytest.approx(0.83)
    assert any("mixed_provenance" in r.message for r in caplog.records)


def test_a_directory_holding_the_cohort_record_reads_it(tmp_path):
    (tmp_path / dr.DECODING_RECORD).write_text((FIXTURES / "decoding_cohort.json").read_text())
    view = dr.load_downstream_record(str(tmp_path), "decoding")
    assert view.split == "test"


def test_mlflow_uri_is_downloaded(tmp_path):
    local = tmp_path / dr.ENCODING_RECORD
    local.write_text((FIXTURES / "encoding_run.json").read_text())
    uri = f"mlflow-artifacts:/3/cccc/artifacts/{dr.ENCODING_RECORD}"
    with patch("mlflow.artifacts.download_artifacts", return_value=str(local)) as dl:
        view = dr.load_downstream_record(uri, "encoding")
    dl.assert_called_once_with(uri, dst_path=ANY)
    assert view.split == "test"


# ── resolution ───────────────────────────────────────────────────────────────


def _listing(*names: str) -> list[SimpleNamespace]:
    return [SimpleNamespace(path=name) for name in names]


def test_resolve_encoding_keeps_the_record_uri():
    uri = f"mlflow-artifacts:/3/cccc/artifacts/{dr.ENCODING_RECORD}"
    with patch("mlflow.artifacts.list_artifacts") as ls:
        assert dr.resolve_downstream_uri(uri, "encoding") == uri
    ls.assert_not_called()


def test_resolve_decoding_picks_the_cohort_record_at_the_parent_root():
    uri = "mlflow-artifacts:/4/dddd/artifacts"
    with patch(
        "mlflow.artifacts.list_artifacts",
        return_value=_listing(dr.DECODING_RECORD, "T__tokA__sess"),
    ) as ls:
        assert dr.resolve_downstream_uri(uri, "decoding") == f"{uri}/{dr.DECODING_RECORD}"
    ls.assert_called_once_with(artifact_uri=uri)


def test_resolve_decoding_without_a_cohort_record_raises():
    uri = "mlflow-artifacts:/4/dddd/artifacts"
    with (
        patch("mlflow.artifacts.list_artifacts", return_value=_listing("T__tokA__sess")),
        pytest.raises(FileNotFoundError, match="cohort record was not uploaded"),
    ):
        dr.resolve_downstream_uri(uri, "decoding")


@pytest.mark.parametrize(
    ("uri", "task"),
    [
        ("/local/run/downstream_encoding_results.json", "encoding"),
        ("/local/benchmark_parent", "decoding"),
        (f"mlflow-artifacts:/4/dddd/artifacts/{dr.DECODING_RECORD}", "decoding"),
    ],
)
def test_resolve_leaves_local_paths_and_explicit_files_alone(uri, task):
    with patch("mlflow.artifacts.list_artifacts") as ls:
        assert dr.resolve_downstream_uri(uri, task) == uri
    ls.assert_not_called()


def test_decoding_parent_dir_without_its_cohort_record_raises(tmp_path):
    """A benchmark parent uploads per-cell records but the cohort record is what is read."""
    cell = tmp_path / "T__tokA__sess" / "csv_logs" / "version_0"
    cell.mkdir(parents=True)
    (cell / dr.DECODING_RECORD).write_text((FIXTURES / "decoding_run.json").read_text())
    with pytest.raises(ValueError, match="without downstream_decoding_results.json"):
        dr.load_downstream_record(str(tmp_path), "decoding")
