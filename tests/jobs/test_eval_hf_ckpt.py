"""``hf://`` checkpoint paths resolve to a local ``.ckpt`` in the eval jobs (no network)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ntt.core.model.checkpoint_io import resolve_checkpoint_for_eval
from ntt.core.model.tokenizer import create_tokenizer
from ntt.jobs.eval import NTTEvalJob
from ntt.jobs.eval_robustness_input import NTTEvalRobustnessInputJob
from ntt.jobs.eval_robustness_latent import NTTEvalRobustnessLatentJob
from tests.test_hub import _HF_URI, _cs_spec, hf_hub  # noqa: F401  (hf_hub is a fixture)


@pytest.mark.parametrize(
    "job_cls", [NTTEvalJob, NTTEvalRobustnessInputJob, NTTEvalRobustnessLatentJob]
)
def test_eval_job_resolves_hf_ckpt(job_cls, hf_hub):  # noqa: F811
    resolved = Path(job_cls._resolve_ckpt_path(_HF_URI))

    assert resolved.suffix == ".ckpt"
    assert resolved.is_file()


def test_eval_hf_ckpt_with_mlflow_upload_on(hf_hub, tmp_path: Path, monkeypatch):  # noqa: F811
    from unittest import mock

    from ntt.core.model import checkpoint_io

    spec = _cs_spec()
    fake_log = mock.Mock()
    monkeypatch.setattr(checkpoint_io, "log", fake_log)

    result = resolve_checkpoint_for_eval(
        _HF_URI, create_tokenizer(spec), spec, always_copy=True, tmpdir=tmp_path
    )

    assert result is None
    # The generic local-path branch would warn "does not exist"; the hf:// branch must not.
    fake_log.warning.assert_not_called()


def test_tokenizer_checkpoint_caption_keeps_hf_uri():
    job = NTTEvalJob.model_construct(ckpt_path=_HF_URI + "@v1.0")

    assert job._tokenizer_checkpoint_caption() == _HF_URI + "@v1.0"
