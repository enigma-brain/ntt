"""Tests for the MLflow plot-upload gate.

Almost no experiment sets ``output_path``, so gating the plot upload on it would
leave plots generated locally but never uploaded to MLflow. The
upload is instead gated on a *resolvable* output dir (output_path OR the
Hydra run dir), mirroring ``_save_results``.

``_upload_plots_to_mlflow`` reads only ``self._get_output_dir`` /
``self.mlflow_logging`` / module globals, so it's exercised with a lightweight
stand-in ``self``.
"""

from __future__ import annotations

from types import SimpleNamespace

from ntt.jobs.eval import NTTEvalJob


class _FakeRunCtx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _make_self(output_dir, *, tracking_uri=None):
    return SimpleNamespace(
        _get_output_dir=lambda: output_dir,
        mlflow_logging=SimpleNamespace(tracking_uri=tracking_uri),
    )


def test_plots_uploaded_when_output_dir_resolves(tmp_path, monkeypatch):
    recorder: dict = {}
    # The helper does a local ``import mlflow``; patch the real module's funcs.
    monkeypatch.setattr(
        "mlflow.start_run", lambda run_id: recorder.__setitem__("run_id", run_id) or _FakeRunCtx()
    )
    monkeypatch.setattr(
        "mlflow.log_artifacts",
        lambda path, artifact_path=None: recorder.__setitem__("uploaded", (path, artifact_path)),
    )
    job = _make_self(tmp_path)

    uploaded = NTTEvalJob._upload_plots_to_mlflow(job, tmp_path, "run123")

    assert uploaded is True
    assert recorder["run_id"] == "run123"
    assert recorder["uploaded"] == (str(tmp_path), "plots")


def test_plots_not_uploaded_without_mlflow_run():
    job = _make_self("/some/dir")
    assert NTTEvalJob._upload_plots_to_mlflow(job, "/some/dir", None) is False


def test_plots_not_uploaded_when_no_output_dir_resolves():
    # output_path unset AND no Hydra run → _get_output_dir() is None → skip,
    # so we never upload the cwd fallback.
    job = _make_self(None)
    assert NTTEvalJob._upload_plots_to_mlflow(job, ".", "run123") is False
