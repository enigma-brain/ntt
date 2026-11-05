"""Tests for eval MLflow checkpoint upload.

Covers:
- Synthesizing a Lightning-compatible ckpt from a live tokenizer and
  round-tripping it via ``load_tokenizer_from_checkpoint_no_lightning``.
- ``resolve_checkpoint_for_eval`` dispatch for all three source cases
  (MLflow URI, local path, spec-only).
- ``MLflowEvalLogger.log`` uploading the checkpoint under ``checkpoints/``.

The full end-to-end eval-job smoke (running
``uv run ntt job=eval tokenizer@job.tokenizer_config=contrib/compressed_sensing``)
is the manual acceptance criterion and is not replicated here.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from ntt.contrib.tokenizer.compressed_sensing import (
    CompressedSensingTokenizerSpec,
)
from ntt.core.model.checkpoint_io import (
    resolve_checkpoint_for_eval,
    save_live_tokenizer_as_checkpoint,
)
from ntt.core.model.tokenizer import (
    _extract_tokenizer_state_dict,
    create_tokenizer,
    load_tokenizer_from_checkpoint_no_lightning,
)
from ntt.evaluation.core.types import (
    EvaluationReport,
    MetricCategory,
    MetricDirection,
    MetricLevel,
    MetricResult,
)
from ntt.evaluation.mlflow_logging import MLflowEvalLogger
from ntt.jobs.eval import _should_log_eval_run_to_mlflow


def _cs_spec(**overrides) -> CompressedSensingTokenizerSpec:
    """A tiny compressed-sensing spec cheap enough for test use."""
    defaults = {
        "n_neurons": 4,
        "window_length_seconds": 0.5,
        "bin_size_ms": 10.0,
        "compress_mode": "per_neuron",
        "latent_dim": 8,
        "trainable": False,
        "random_seed": 123,
        "fista_iterations": 5,
    }
    defaults.update(overrides)
    return CompressedSensingTokenizerSpec(**defaults)


def test_extract_tokenizer_state_dict_strips_lightning_prefix():
    tokenizer_weight = torch.tensor([1.0])
    checkpoint = {
        "state_dict": {
            "tokenizer.encoder.weight": tokenizer_weight,
            "unrelated.weight": torch.tensor([2.0]),
        }
    }

    extracted = _extract_tokenizer_state_dict(checkpoint)

    assert set(extracted) == {"encoder.weight"}
    assert extracted["encoder.weight"] is tokenizer_weight


def test_extract_tokenizer_state_dict_accepts_raw_state_dict():
    raw_weight = torch.tensor([1.0])
    raw_state_dict = {"encoder.weight": raw_weight}

    extracted = _extract_tokenizer_state_dict(raw_state_dict)

    assert extracted is raw_state_dict


class TestSaveLiveTokenizerAsCheckpoint:
    def test_roundtrip_compressed_sensing_spec(self, tmp_path: Path):
        spec = _cs_spec(latent_dim=12)
        tokenizer = create_tokenizer(spec)

        ckpt_path = tmp_path / "synth.ckpt"
        save_live_tokenizer_as_checkpoint(tokenizer, spec, ckpt_path)

        assert ckpt_path.exists()
        reloaded = load_tokenizer_from_checkpoint_no_lightning(ckpt_path)
        assert reloaded.spec.KIND == "compressed_sensing"
        assert reloaded.spec.latent_dim == 12

        # State dict keys survived the tokenizer. prefix reprefixing roundtrip.
        orig_keys = set(tokenizer.state_dict().keys())
        reloaded_keys = set(reloaded.state_dict().keys())
        assert orig_keys == reloaded_keys

        # Weights match (deterministic random_seed=123).
        for k, v in tokenizer.state_dict().items():
            torch.testing.assert_close(v, reloaded.state_dict()[k])

    def test_ckpt_contains_expected_fields(self, tmp_path: Path):
        spec = _cs_spec()
        tokenizer = create_tokenizer(spec)

        ckpt_path = tmp_path / "synth.ckpt"
        save_live_tokenizer_as_checkpoint(tokenizer, spec, ckpt_path)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert "hyper_parameters" in ckpt
        assert "state_dict" in ckpt
        assert "pytorch-lightning_version" in ckpt
        assert ckpt["hyper_parameters"]["tokenizer_spec"] == spec
        # EvalOnlyMSELossSpec by default.
        assert ckpt["hyper_parameters"]["loss_spec"].KIND == "eval_only_mse"
        # Every state_dict key must be tokenizer.-prefixed.
        for key in ckpt["state_dict"]:
            assert key.startswith("tokenizer."), key


class TestResolveCheckpointForEval:
    def _live_tokenizer(self):
        spec = _cs_spec()
        return create_tokenizer(spec), spec

    def test_mlflow_uri_without_copy_returns_none(self, tmp_path: Path):
        tokenizer, spec = self._live_tokenizer()
        result = resolve_checkpoint_for_eval(
            ckpt_path="mlflow-artifacts:/runs/abc/artifacts/best.ckpt",
            live_tokenizer=tokenizer,
            live_spec=spec,
            always_copy=False,
            tmpdir=tmp_path,
        )
        assert result is None

    @patch("mlflow.artifacts.download_artifacts")
    def test_mlflow_uri_with_copy_downloads(self, mock_download, tmp_path: Path):
        tokenizer, spec = self._live_tokenizer()

        fake_local = tmp_path / "downloaded.ckpt"
        fake_local.write_bytes(b"fake ckpt bytes")
        mock_download.return_value = str(fake_local)

        result = resolve_checkpoint_for_eval(
            ckpt_path="mlflow-artifacts:/runs/abc/artifacts/best.ckpt",
            live_tokenizer=tokenizer,
            live_spec=spec,
            always_copy=True,
            tmpdir=tmp_path,
        )
        assert result == fake_local
        mock_download.assert_called_once()

    def test_local_path_returned_as_is(self, tmp_path: Path):
        tokenizer, spec = self._live_tokenizer()
        local = tmp_path / "existing.ckpt"
        local.write_bytes(b"x")

        result = resolve_checkpoint_for_eval(
            ckpt_path=str(local),
            live_tokenizer=tokenizer,
            live_spec=spec,
            always_copy=False,
            tmpdir=tmp_path,
        )
        assert result == local

    def test_missing_local_path_skips_upload(self, tmp_path: Path):
        tokenizer, spec = self._live_tokenizer()
        result = resolve_checkpoint_for_eval(
            ckpt_path=str(tmp_path / "does_not_exist.ckpt"),
            live_tokenizer=tokenizer,
            live_spec=spec,
            always_copy=False,
            tmpdir=tmp_path,
        )
        assert result is None

    def test_spec_only_synthesizes_and_roundtrips(self, tmp_path: Path):
        tokenizer, spec = self._live_tokenizer()
        result = resolve_checkpoint_for_eval(
            ckpt_path=None,
            live_tokenizer=tokenizer,
            live_spec=spec,
            always_copy=False,
            tmpdir=tmp_path,
        )
        assert result is not None
        assert result.name == "model.ckpt"
        assert result.parent == tmp_path

        # Round-trip.
        reloaded = load_tokenizer_from_checkpoint_no_lightning(result)
        assert reloaded.spec.KIND == "compressed_sensing"


class TestMLflowLoggerUploadsCheckpoint:
    @patch("ntt.evaluation.mlflow_logging.mlflow")
    def test_checkpoint_logged_under_checkpoints_artifact_path(self, mock_mlflow, tmp_path: Path):
        mock_run = MagicMock()
        mock_run.info.run_id = "eval_run_ckpt"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        report = EvaluationReport(
            results={
                "f1": MetricResult(
                    name="f1",
                    raw_value=np.float64(0.9),
                    direction=MetricDirection.HIGHER_IS_BETTER,
                    category=MetricCategory.SINGLE_UNIT_STATISTICS,
                    level=MetricLevel.POPULATION,
                    metadata={},
                )
            },
            category_scores={},
            config={},
        )
        ckpt = tmp_path / "model.ckpt"
        ckpt.write_bytes(b"ckpt")

        logger = MLflowEvalLogger(experiment_name="test-eval")
        logger.log(report=report, checkpoint_local_path=ckpt)

        artifact_calls = mock_mlflow.log_artifact.call_args_list
        ckpt_calls = [c for c in artifact_calls if c.kwargs.get("artifact_path") == "checkpoints"]
        assert len(ckpt_calls) == 1
        assert ckpt_calls[0].args[0] == str(ckpt)

    @patch("ntt.evaluation.mlflow_logging.mlflow")
    def test_no_checkpoint_path_skips_upload(self, mock_mlflow):
        mock_run = MagicMock()
        mock_run.info.run_id = "eval_run_no_ckpt"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        report = EvaluationReport(results={}, category_scores={}, config={})

        logger = MLflowEvalLogger(experiment_name="test-eval")
        logger.log(report=report, checkpoint_local_path=None)

        artifact_calls = mock_mlflow.log_artifact.call_args_list
        ckpt_calls = [c for c in artifact_calls if c.kwargs.get("artifact_path") == "checkpoints"]
        assert ckpt_calls == []


def test_snr_without_repeats_snr_skips_mlflow_logging():
    report = EvaluationReport(results={}, category_scores={}, config={})
    assert _should_log_eval_run_to_mlflow(report, run_name_suffix="__snr") is False


def test_non_snr_or_present_repeats_snr_still_logs_to_mlflow():
    report = EvaluationReport(
        results={
            "repeats_snr": MetricResult(
                name="repeats_snr",
                raw_value=np.float64(0.1),
                direction=MetricDirection.HIGHER_IS_BETTER,
                category=MetricCategory.POPULATION_STRUCTURE,
                level=MetricLevel.POPULATION,
                metadata={},
            )
        },
        category_scores={},
        config={},
    )
    assert _should_log_eval_run_to_mlflow(report, run_name_suffix="__snr") is True
    assert _should_log_eval_run_to_mlflow(report, run_name_suffix="__main") is True


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
