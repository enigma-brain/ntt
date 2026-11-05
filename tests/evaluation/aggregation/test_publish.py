"""Tests for ntt.evaluation.aggregation.publish (against a real local MLflow store)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient

from ntt.evaluation.aggregation import publish, tracker_io
from ntt.evaluation.aggregation.publish import publish_category_scores


def _scores_json(path: Path, sn_recon: float = 0.8) -> Path:
    doc = {
        "scores": {"single_neuron_recon": sn_recon, "decoding": None},
        "diagnostics": {"single_neuron_recon": {"n_metrics": 3}, "population_structure": None},
        "metadata": {"ckpt_uri": "mlflow-artifacts:/1/r/artifacts/last.ckpt"},
    }
    path.write_text(json.dumps(doc))
    return path


def _run(url: str):
    ids = tracker_io.parse_run_url(url)
    assert ids is not None, url
    return MlflowClient().get_run(ids[1])


def test_publish_logs_scores_artifacts_and_tags(tmp_path, mlflow_store):
    # The builder's file is scores/<slug>.json; the run must still carry category_scores.json.
    src = _scores_json(tmp_path / "tok_a.json")
    checks = tmp_path / "table_checks.json"
    checks.write_text("{}")

    published = publish_category_scores(
        src, experiment="pub-logs", run_name="tok_a", tags={"x": "y"}, artifacts=[checks]
    )

    assert not published.reused
    assert published.url.startswith(mlflow_store.rstrip("/") + "/#/experiments/")
    run = _run(published.url)
    assert run.info.run_name == "tok_a"
    assert run.info.status == "FINISHED"
    assert run.data.metrics["single_neuron_recon"] == pytest.approx(0.8)
    assert math.isnan(run.data.metrics["decoding"])
    assert run.data.metrics["n_metrics_sn_recon"] == 3
    assert "n_metrics_population_structure" not in run.data.metrics
    assert run.data.tags["x"] == "y"
    assert run.data.tags[publish.TOKENIZER_TAG] == "tok_a"
    assert run.data.tags[publish.CKPT_URI_TAG] == "mlflow-artifacts:/1/r/artifacts/last.ckpt"
    assert len(run.data.tags[publish.SCORES_SHA256_TAG]) == 64
    artifacts = {a.path for a in MlflowClient().list_artifacts(run.info.run_id)}
    assert artifacts == {"category_scores.json", "table_checks.json"}


def test_publishing_the_same_file_again_reuses_the_run(tmp_path, mlflow_store):
    src = _scores_json(tmp_path / "tok_a.json")
    first = publish_category_scores(src, experiment="pub-reuse", run_name="tok_a")
    second = publish_category_scores(src, experiment="pub-reuse", run_name="tok_a")

    assert second.reused and second.url == first.url
    experiment = MlflowClient().get_experiment_by_name("pub-reuse")
    assert experiment is not None
    assert len(MlflowClient().search_runs([experiment.experiment_id])) == 1


def test_a_changed_file_or_another_tokenizer_gets_a_new_run(tmp_path, mlflow_store):
    src = _scores_json(tmp_path / "tok_a.json")
    first = publish_category_scores(src, experiment="pub-new", run_name="tok_a")
    other_tokenizer = publish_category_scores(src, experiment="pub-new", run_name="tok_b")
    _scores_json(src, sn_recon=0.9)
    rescored = publish_category_scores(src, experiment="pub-new", run_name="tok_a")

    assert not other_tokenizer.reused and not rescored.reused
    assert len({first.url, other_tokenizer.url, rescored.url}) == 3


def test_other_tags_or_artifacts_get_a_new_run(tmp_path, mlflow_store):
    # Same scores, another table verdict: the reused run would carry the old one.
    src = _scores_json(tmp_path / "tok_a.json")
    checks = tmp_path / "table_checks.json"
    checks.write_text('{"passed": false}')
    kwargs = {"experiment": "pub-verdict", "run_name": "tok_a", "artifacts": [checks]}
    failed = publish_category_scores(src, tags={"verdict": "failed:x"}, **kwargs)
    passed = publish_category_scores(src, tags={"verdict": "passed"}, **kwargs)
    checks.write_text('{"passed": true}')
    new_checks = publish_category_scores(src, tags={"verdict": "passed"}, **kwargs)
    again = publish_category_scores(src, tags={"verdict": "passed"}, **kwargs)

    assert not passed.reused and not new_checks.reused
    assert len({failed.url, passed.url, new_checks.url}) == 3
    assert again.reused and again.url == new_checks.url
    assert _run(passed.url).data.tags["verdict"] == "passed"


def test_a_failing_cleanup_does_not_hide_the_upload_error(tmp_path, mlflow_store, monkeypatch):
    src = _scores_json(tmp_path / "tok_a.json")

    def _drop(*_a, **_k):
        raise ConnectionError("upload failed")

    def _down(*_a, **_k):
        raise TimeoutError("cleanup failed")

    monkeypatch.setattr(MlflowClient, "log_artifact", _drop)
    monkeypatch.setattr(MlflowClient, "set_terminated", _down)
    with pytest.raises(ConnectionError, match="upload failed"):
        publish_category_scores(src, experiment="pub-cleanup", run_name="tok_a")


def test_a_publish_that_failed_midway_is_never_reused(tmp_path, mlflow_store, monkeypatch):
    src = _scores_json(tmp_path / "tok_a.json")

    def _drop(*_a, **_k):
        raise ConnectionError("server went away")

    with monkeypatch.context() as patched:
        patched.setattr(MlflowClient, "log_artifact", _drop)
        with pytest.raises(ConnectionError):
            publish_category_scores(src, experiment="pub-midway", run_name="tok_a")

    retried = publish_category_scores(src, experiment="pub-midway", run_name="tok_a")

    assert not retried.reused
    experiment = MlflowClient().get_experiment_by_name("pub-midway")
    assert experiment is not None
    runs = MlflowClient().search_runs([experiment.experiment_id])
    statuses = sorted(r.info.status for r in runs)
    assert statuses == ["FAILED", "FINISHED"]
    failed = next(r for r in runs if r.info.status == "FAILED")
    assert publish.PUBLISH_SHA256_TAG not in failed.data.tags


def test_run_url_is_what_parse_run_url_reads():
    url = publish.run_url("https://mlflow.example/", "7", "0123abcd")
    assert url == "https://mlflow.example/#/experiments/7/runs/0123abcd"
    assert tracker_io.parse_run_url(url) == ("7", "0123abcd")
    assert tracker_io.run_url_tracking_uri(url) == "https://mlflow.example"
