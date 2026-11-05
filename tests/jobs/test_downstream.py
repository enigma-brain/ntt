"""Tests for the downstream-eval launcher (``NTTDownstreamJob``).

These lock the *command contract*: the launcher's whole job is to assemble a
correct argv + environment and shell into the vendored submodule. Everything
here runs without the submodule checked out and without a GPU — ``dry_run``
short-circuits before ``subprocess.run`` and downgrades preflight to warnings,
and the builders are pure functions of the job config.
"""

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any

import pydantic as pdt
import pytest

from ntt.jobs import downstream as ds
from ntt.jobs import downstream_env
from ntt.jobs.downstream import NTTDownstreamJob, _is_uri
from tests.test_hub import _HF_URI, hf_hub  # noqa: F401  (hf_hub is a fixture)


def _R(p: str) -> str:
    """Mirror ``_abspath`` resolution for a local path, for golden assertions."""
    return str(Path(p).expanduser().resolve())


def _decoding_job(**kw: Any) -> NTTDownstreamJob:
    base: dict[str, Any] = {
        "task": "decoding",
        "mode": "single-run",
        "ckpt": "/abs/model.ckpt",
        "session": "sess_x",
        "dry_run": True,
    }
    base.update(kw)
    return NTTDownstreamJob(**base)


def _roots(job: NTTDownstreamJob) -> tuple[Path, Path]:
    host = job._host_root()
    return host, (host / job.submodule_dir).resolve()


def _flag(cmd: list[str], flag: str) -> str:
    """Return the value following ``flag`` in an argv list."""
    return cmd[cmd.index(flag) + 1]


def _assert_cli(cmd: list[str], sub: Path, verb: str) -> None:
    """Pin the contract invocation: uv-run the submodule's console script."""
    assert cmd[:2] == ["uv", "run"]
    assert _flag(cmd, "--project") == str(sub)
    # The launcher owns the env (host-ntt overlay); --frozen would re-sync it to
    # the lock and silently swap the host ntt back for the pinned one.
    assert "--no-sync" in cmd
    assert "--frozen" not in cmd
    assert f"ntt-downstream-{verb}" in cmd


def _positionals(cmd: list[str]) -> list[str]:
    """The trailing raw key=value Hydra passthrough tokens."""
    return [t for t in cmd if "=" in t and not t.startswith("--")]


def _sweep_cfg(cmd: list[str]) -> dict:
    """Load the sweep YAML the builder wrote, from the command's own flags."""
    from omegaconf import OmegaConf

    path = Path(_flag(cmd, "--config-dir")) / f"{_flag(cmd, '--config-name')}.yaml"
    assert path.is_file(), f"sweep config not written: {path}"
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


# ── _is_uri ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "value,expected",
    [
        ("mlflow-artifacts:/123/model.ckpt", True),
        ("mlflow:/foo/bar", True),
        ("s3://bucket/model.ckpt", True),
        ("/abs/model.ckpt", False),
        ("relative/model.ckpt", False),
        ("model.ckpt", False),
    ],
)
def test_is_uri(value, expected):
    assert _is_uri(value) is expected


# ── _abspath ─────────────────────────────────────────────────────────────
def test_abspath_resolves_local_and_passes_uris_through():
    job = _decoding_job()
    assert job._abspath(None) is None
    # URIs are returned untouched (resolved inside the submodule's loader).
    assert job._abspath("mlflow-artifacts:/r/model.ckpt") == "mlflow-artifacts:/r/model.ckpt"
    # Local paths become absolute.
    assert job._abspath("/abs/p") == _R("/abs/p")
    resolved_rel = job._abspath("rel/p")
    assert resolved_rel is not None and Path(resolved_rel).is_absolute()


# ── _benchmark_tag ───────────────────────────────────────────────────────
def test_benchmark_tag_from_ckpt_root():
    job = _decoding_job(mode="benchmark", ckpt=None, ckpt_root="/a/my_toks", session=None)
    assert job._benchmark_tag() == "ntt_downstream_my_toks"


# ── _load_dotenv ─────────────────────────────────────────────────────────
def test_load_dotenv_parses_and_existing_keys_win(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# a comment line",
                "export FOO=bar",
                'QUOTED="hello world"',
                "SINGLE='x'",
                "EXISTING=fromfile",
                "noequalshere",
                "",
            ]
        )
    )
    env = {"EXISTING": "fromenv"}
    NTTDownstreamJob._load_dotenv(env_file, env)
    assert env["FOO"] == "bar"
    assert env["QUOTED"] == "hello world"
    assert env["SINGLE"] == "x"
    assert env["EXISTING"] == "fromenv"  # user env wins over the file
    assert "noequalshere" not in env


def test_load_dotenv_missing_file_is_noop(tmp_path):
    env: dict[str, str] = {}
    NTTDownstreamJob._load_dotenv(tmp_path / "nope.env", env)
    assert env == {}


# ── _clean_env ───────────────────────────────────────────────────────────
def test_clean_env_strips_leaky_vars_and_sets_splits_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("VIRTUAL_ENV", "/some/venv")
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/x")
    monkeypatch.setenv("PYTHONPATH", "/x")
    monkeypatch.setenv("SLURM_SUBMIT_DIR", "/host/repo")
    monkeypatch.setenv("KEEP_ME", "1")
    job = _decoding_job()
    env = job._clean_env(tmp_path, source_dotenv=None)
    for stripped in ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT", "PYTHONPATH", "SLURM_SUBMIT_DIR"):
        assert stripped not in env
    assert env["KEEP_ME"] == "1"
    assert env["NTT_SPLITS_DIR"] == str(tmp_path / "config" / "data" / "selections")


def test_clean_env_merges_dotenv_with_user_env_winning(monkeypatch, tmp_path):
    # The .sh orchestrators self-source .env; the direct-python paths rely on
    # _clean_env wiring _load_dotenv in to pull MLflow creds. Pin that wiring.
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://from-env")
    env_file = tmp_path / ".env"
    env_file.write_text("MLFLOW_TRACKING_URI=http://from-file\nNEW_KEY=val\n")
    job = _decoding_job()
    env = job._clean_env(tmp_path, source_dotenv=env_file)
    assert env["NEW_KEY"] == "val"  # fresh key pulled from .env
    assert env["MLFLOW_TRACKING_URI"] == "http://from-env"  # exported value wins


# ── _selection_sessions ──────────────────────────────────────────────────
def test_selection_sessions_flattens_shipped_cohort():
    job = _decoding_job()
    host = job._host_root()
    sessions = job._selection_sessions(host, "decoding/ntt_decoding_calib_base_train")
    assert sessions == [
        "c_20151103_center_out_reaching",
        "c_20131217_random_target_reaching",
    ]


def test_selection_sessions_missing_file_raises():
    job = _decoding_job()
    host = job._host_root()
    with pytest.raises(FileNotFoundError):
        job._selection_sessions(host, "decoding/does_not_exist")


def test_selection_sessions_rejects_multi_brainset(tmp_path):
    sel_dir = tmp_path / "config" / "data" / "selections"
    sel_dir.mkdir(parents=True)
    (sel_dir / "multi.yaml").write_text(
        "selection:\n  - brainset: a\n    sessions: [s1]\n  - brainset: b\n    sessions: [s2]\n"
    )
    job = _decoding_job()
    with pytest.raises(ValueError, match="single-brainset"):
        job._selection_sessions(tmp_path, "multi")


# ── cross-task / mutual-exclusivity validation ───────────────────────────
@pytest.mark.parametrize(
    "field,value",
    [("mode", "calibrate"), ("model_cfg", "poyo_1.3M"), ("session", "s"), ("mlflow", False)],
)
def test_encoding_rejects_decoding_only_fields(field, value):
    with pytest.raises(pdt.ValidationError, match="decoding-only"):
        NTTDownstreamJob(task="encoding", ckpt="/abs/m.ckpt", **{field: value})


# recipe must be non-default — the guard only fires on non-default overrides.
@pytest.mark.parametrize(
    "field,value", [("recipe", "cowley_v4_perceiver_mse"), ("loss_mode", "mse")]
)
def test_decoding_rejects_encoding_only_fields(field, value):
    with pytest.raises(pdt.ValidationError, match="encoding-only"):
        NTTDownstreamJob(task="decoding", ckpt="/abs/m.ckpt", **{field: value})


def test_rejects_both_ckpt_and_ckpt_root():
    with pytest.raises(pdt.ValidationError, match="mutually exclusive"):
        NTTDownstreamJob(task="decoding", mode="benchmark", ckpt="/a/m.ckpt", ckpt_root="/a/dir")


def test_default_task_configs_are_not_misscoped():
    # The fields each shipped config sets must not trip the cross-task guard.
    NTTDownstreamJob(task="decoding", mode="single-run", ckpt="/a/m.ckpt", session="s")
    NTTDownstreamJob(task="encoding", ckpt="/a/m.ckpt", recipe="cowley_v4_base", loss_mode="mse")


# ── numeric coercion (narrowed to gpus / session) ────────────────────────
def test_gpus_numeric_index_is_coerced_to_str():
    # job.gpus=0 arrives from Hydra as an int; the single GPU index must parse.
    job = _decoding_job(mode="calibrate", session=None, gpus=0)
    assert job.gpus == "0"


def test_session_numeric_id_is_coerced_to_str():
    job = _decoding_job(session=12345)
    assert job.session == "12345"


@pytest.mark.parametrize("field,value", [("model_cfg", 11), ("data_root", 7), ("out", 3)])
def test_numeric_on_strict_string_field_is_rejected(field, value):
    # The narrowed coercion leaves typos on other string fields caught at load.
    with pytest.raises(pdt.ValidationError, match="string"):
        _decoding_job(**{field: value})


def test_overriding_cross_task_field_to_its_default_is_tolerated():
    # The idempotency trade-off: an explicit override *to* the default is a no-op
    # and must not be rejected (this is what keeps round-trips working).
    NTTDownstreamJob(task="encoding", ckpt="/a/m.ckpt", mode="single-run", session=None)


@pytest.mark.parametrize("task,extra", [("decoding", {}), ("encoding", {})])
def test_job_survives_model_dump_round_trip(task, extra):
    # model_dump() emits every field; reconstructing must not falsely trip the
    # cross-task guard on the other task's defaulted fields.
    job = NTTDownstreamJob(task=task, ckpt="/abs/m.ckpt", **extra)
    dump = job.model_dump()
    assert NTTDownstreamJob(**dump).task == task
    assert NTTDownstreamJob.model_validate(dump).task == task


@pytest.mark.parametrize("cfg_name", ["downstream_decoding", "downstream_encoding"])
def test_shipped_job_configs_instantiate(cfg_name):
    # Guards against a mis-scoped key being added to a real config — the field
    # sets a config ships must pass the validator through Hydra instantiate.
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    root = _decoding_job()._host_root()
    cfg = OmegaConf.load(root / "config" / "job" / f"{cfg_name}.yaml")
    job = instantiate(cfg)
    assert isinstance(job, NTTDownstreamJob)


# ── golden command: decoding / single-run ────────────────────────────────
def test_single_run_uses_decode_console_script():
    job = _decoding_job(
        ckpt="/abs/model.ckpt", session="sess_x", data_root="/abs/data", out="/abs/out"
    )
    host, sub = _roots(job)
    cmd, _, out = job._build_decoding(host, sub)

    _assert_cli(cmd, sub, "decode")
    assert _flag(cmd, "--tokenizer-kind") == "frozen"
    assert _flag(cmd, "--tokenizer-ckpt") == _R("/abs/model.ckpt")
    assert _flag(cmd, "--data-root") == _R("/abs/data")
    assert _flag(cmd, "--output-root") == _R("/abs/out")
    # arch defaults are OWNED by configs/cli/decode.yaml now — never emitted here.
    assert not any(t.startswith("model=") for t in cmd)
    assert not any(t.startswith("tokenizer.unit_embedding_mode=") for t in cmd)
    assert not any(t.startswith("tokenizer.sentinels=") for t in cmd)
    assert "session=sess_x" in cmd
    assert cmd.count("wandb.enabled=false") == 1
    # job.dry_run=True must forward the launcher's GPU-free preview flag.
    assert "--dry-run" in cmd
    assert out == _R("/abs/out")


def test_single_run_knobs_become_hydra_passthrough():
    job = _decoding_job(
        epochs=1,
        batch_size=8,
        seed=7,
        num_workers=2,
        model_cfg="poyo_11.8M",
        unit_embedding_mode="concat",
        sentinels=True,
        extra=["max_steps=200"],
    )
    host, sub = _roots(job)
    cmd, _, _ = job._build_decoding(host, sub)
    assert _flag(cmd, "--seed") == "7"  # promoted flag, not a hydra token
    for token in (
        "epochs=1",
        "batch_size=8",
        "num_workers=2",
        "max_steps=200",
        "model=poyo_11.8M",
        "tokenizer.unit_embedding_mode=concat",
        "tokenizer.sentinels=true",
    ):
        assert token in cmd
    # user extra is LAST so it can override anything (Hydra last-wins).
    assert _positionals(cmd)[-1] == "max_steps=200"
    assert cmd.index("wandb.enabled=false") < cmd.index("max_steps=200")


def test_single_run_flags_precede_positionals():
    # argparse safety: every --flag must come before the raw key=value tokens.
    job = _decoding_job(extra=["max_steps=200"])
    host, sub = _roots(job)
    cmd, _, _ = job._build_decoding(host, sub)
    first_positional = cmd.index(_positionals(cmd)[0])
    assert all(cmd.index(t) < first_positional for t in cmd if t.startswith("--"))


def test_single_run_uri_ckpt_passes_through_unchanged():
    # The contract resolves mlflow URIs inside the submodule's loader; the
    # launcher must forward them verbatim (no _abspath mangling).
    job = _decoding_job(ckpt="mlflow-artifacts:/r/model.ckpt")
    cmd, _, _ = job._build_decoding(*_roots(job))
    assert _flag(cmd, "--tokenizer-ckpt") == "mlflow-artifacts:/r/model.ckpt"


# ── golden command: decoding / calibrate ─────────────────────────────────
def test_calibrate_command_and_sweep_config(tmp_path):
    out = tmp_path / "calib"
    job = _decoding_job(
        mode="calibrate",
        ckpt="/abs/m.ckpt",
        session=None,
        data_root="/abs/data",
        out=str(out),
        gpus="0",
        batch_size=16,
        num_workers=2,
        model_cfg="poyo_1.3M",
        unit_embedding_mode="add",
        sentinels=False,
        mlflow="auto",
    )
    host, sub = _roots(job)
    cmd, _, desc = job._build_decoding(host, sub)

    _assert_cli(cmd, sub, "calibrate")
    assert _flag(cmd, "--tokenizer-kind") == "frozen"
    assert _flag(cmd, "--tokenizer-ckpt") == _R("/abs/m.ckpt")
    assert _flag(cmd, "--data-root") == _R("/abs/data")
    assert _flag(cmd, "--output-root") == str(out)
    assert "--dry-run" in cmd

    cfg = _sweep_cfg(cmd)
    # Bucket B cohorts, flattened from this repo's shipped selections.
    assert "c_20151103_center_out_reaching" in cfg["train_sessions"]
    assert cfg["other_sessions"]
    # non-promoted knobs travel in the file (keys = engine argparse dests).
    assert cfg["gpus"] == "0"
    assert cfg["batch_size"] == 16
    assert cfg["num_workers"] == 2
    assert cfg["model_cfg"] == "poyo_1.3M"
    assert cfg["unit_embedding_mode"] == "add"
    assert cfg["sentinels"] is False
    assert cfg["mlflow"] == "auto"
    # promoted values must NOT be duplicated into the file.
    for absent in ("ckpt", "tokenizer_kind", "calib_root", "seeds", "data_root", "dry_run"):
        assert absent not in cfg
    assert str(out) in desc


def test_calibrate_seed_is_promoted_flag(tmp_path):
    job = _decoding_job(
        mode="calibrate", ckpt="/abs/m.ckpt", session=None, out=str(tmp_path), seed=3
    )
    cmd, _, _ = job._build_decoding(*_roots(job))
    assert _flag(cmd, "--seed") == "3"  # launcher maps → one-element seeds list


def test_calibrate_accepts_uri_ckpt(tmp_path):
    # URIs pass through verbatim per the contract — the old [[ -f $CKPT ]]
    # mirror is gone.
    job = _decoding_job(
        mode="calibrate", ckpt="mlflow-artifacts:/r/model.ckpt", session=None, out=str(tmp_path)
    )
    cmd, _, _ = job._build_decoding(*_roots(job))
    assert _flag(cmd, "--tokenizer-ckpt") == "mlflow-artifacts:/r/model.ckpt"


def test_calibrate_extra_rides_the_command_line(tmp_path):
    job = _decoding_job(
        mode="calibrate",
        ckpt="/abs/m.ckpt",
        session=None,
        out=str(tmp_path),
        extra=["ft_max_steps=10"],
    )
    cmd, _, _ = job._build_decoding(*_roots(job))
    assert "ft_max_steps=10" in cmd  # positional → engine per-cell extra
    assert "extra" not in _sweep_cfg(cmd)


def test_calibrate_custom_selection_and_multi_gpu_reach_sweep_config(tmp_path):
    # the cohort plumbing is the load-bearing part of the file: a custom
    # eval_selection must override the default, and a space-separated GPU
    # string must survive the YAML round-trip (_resolve_gpus splits on space).
    # NOTE: root_dir=tmp_path makes _host_root() return tmp_path, so the DEFAULT
    # train selection won't exist there — pass a custom train_selection too so the
    # test is self-contained (train_sessions is flattened first in the builder).
    sel_dir = tmp_path / "config" / "data" / "selections"
    sel_dir.mkdir(parents=True)
    (sel_dir / "custom_train.yaml").write_text(
        "selection:\n  - brainset: b\n    sessions: [s_train_1]\n"
    )
    (sel_dir / "custom_eval.yaml").write_text(
        "selection:\n  - brainset: b\n    sessions: [s_eval_1, s_eval_2]\n"
    )
    job = _decoding_job(
        mode="calibrate",
        ckpt="/abs/m.ckpt",
        session=None,
        out=str(tmp_path / "calib"),
        root_dir=str(tmp_path),
        train_selection="custom_train",
        eval_selection="custom_eval",
        gpus="0 1",
    )
    cmd, _, _ = job._build_decoding(*_roots(job))
    cfg = _sweep_cfg(cmd)
    assert cfg["train_sessions"] == ["s_train_1"]
    assert cfg["other_sessions"] == ["s_eval_1", "s_eval_2"]
    assert cfg["gpus"] == "0 1"


def test_numeric_gpus_coerces_all_the_way_into_sweep_config(tmp_path):
    job = _decoding_job(
        mode="calibrate", ckpt="/abs/m.ckpt", session=None, out=str(tmp_path), gpus=0
    )
    cmd, _, _ = job._build_decoding(*_roots(job))
    assert _sweep_cfg(cmd)["gpus"] == "0"


# ── golden command: decoding / benchmark ─────────────────────────────────
def test_benchmark_command_and_sweep_config():
    job = _decoding_job(
        mode="benchmark",
        ckpt=None,
        ckpt_root="/abs/my_toks",
        session=None,
        data_root="/abs/data",
        seed=5,
        epochs=2,
        model_cfg="poyo_1.3M",
    )
    host, sub = _roots(job)
    cmd, _, desc = job._build_decoding(host, sub)

    _assert_cli(cmd, sub, "benchmark")
    assert _flag(cmd, "--data-root") == _R("/abs/data")
    assert _flag(cmd, "--seed") == "5"
    assert "--tokenizer-ckpt" not in cmd  # intentionally ignored by the contract
    assert "--dry-run" in cmd

    cfg = _sweep_cfg(cmd)
    assert cfg["ckpt_root"] == _R("/abs/my_toks")
    assert cfg["comparison_tag"] == "ntt_downstream_my_toks"
    assert cfg["sessions"]  # flattened shipped benchmark cohort
    assert cfg["epochs"] == 2
    assert cfg["model_cfg"] == "poyo_1.3M"
    for absent in ("seed", "data_root", "dry_run"):
        assert absent not in cfg  # promoted/flag-owned values stay off the file
    assert "mlflow" not in cfg  # unset: the submodule's default (auto) applies
    assert desc.endswith("experiments_benchmark/ntt_downstream_my_toks/")


def test_benchmark_custom_selection_and_knobs_reach_sweep_config(tmp_path):
    # mirror the calibrate symmetry: a custom eval_selection must override the
    # default benchmark cohort, and every non-promoted knob must reach the file.
    sel_dir = tmp_path / "config" / "data" / "selections"
    sel_dir.mkdir(parents=True)
    (sel_dir / "custom_bench.yaml").write_text(
        "selection:\n  - brainset: b\n    sessions: [s_bench_1, s_bench_2]\n"
    )
    job = _decoding_job(
        mode="benchmark",
        ckpt=None,
        ckpt_root="/abs/my_toks",
        session=None,
        root_dir=str(tmp_path),
        eval_selection="custom_bench",
        gpus="0 1",
        batch_size=16,
        num_workers=2,
        unit_embedding_mode="add",
        sentinels=False,
        mlflow=False,
    )
    cmd, _, _ = job._build_decoding(*_roots(job))
    cfg = _sweep_cfg(cmd)
    assert cfg["sessions"] == ["s_bench_1", "s_bench_2"]
    assert cfg["gpus"] == "0 1"
    assert cfg["batch_size"] == 16
    assert cfg["num_workers"] == 2
    assert cfg["unit_embedding_mode"] == "add"
    assert cfg["sentinels"] is False
    assert cfg["mlflow"] is False


def test_single_run_rejects_job_mlflow():
    """single-run has no sweep: it logs to MLflow only with mlflow.enabled=true in job.extra."""
    job = _decoding_job(mlflow=False)
    with pytest.raises(ValueError, match="job.mlflow applies to mode=calibrate"):
        job._build_decoding(*_roots(job))


def test_benchmark_requires_ckpt_or_ckpt_root():
    job = _decoding_job(mode="benchmark", ckpt=None, session=None)
    with pytest.raises(ValueError, match="job.ckpt or job.ckpt_root"):
        job.run()


def test_benchmark_ckpt_and_ckpt_root_stay_exclusive():
    with pytest.raises(pdt.ValidationError, match="mutually exclusive"):
        _decoding_job(mode="benchmark", ckpt="/a/m.ckpt", ckpt_root="/a/dir", session=None)


def _bench_job(tmp_path: Path, **kw: Any) -> NTTDownstreamJob:
    """A single-ckpt benchmark job rooted in ``tmp_path`` (with a one-session cohort)."""
    sel_dir = tmp_path / "config" / "data" / "selections"
    sel_dir.mkdir(parents=True, exist_ok=True)
    (sel_dir / "bench.yaml").write_text("selection:\n  - brainset: b\n    sessions: [s1]\n")
    return _decoding_job(
        mode="benchmark", session=None, root_dir=str(tmp_path), eval_selection="bench", **kw
    )


def test_benchmark_hf_ckpt_builds_ckpt_root_layout(hf_hub, tmp_path):  # noqa: F811
    download, _, _ = hf_hub
    job = _bench_job(tmp_path, ckpt=_HF_URI + "@v1.0", dry_run=False)
    cmd, _, desc = job._build_decoding(*_roots(job))

    root = Path(_sweep_cfg(cmd)["ckpt_root"])
    assert root.parent == tmp_path / "results" / "downstream" / "benchmark_roots"
    # the <name> dir is the Hub model name: it becomes the cohort row name
    link = root / "conv-vae-100to1" / "checkpoints" / "model.ckpt"
    assert [p.name for p in root.iterdir()] == ["conv-vae-100to1"]
    assert link.is_symlink() and link.resolve().is_file()
    assert link.resolve().name == "model.ckpt" and "v1.0" in link.resolve().parts
    assert _sweep_cfg(cmd)["comparison_tag"] == f"ntt_downstream_{root.name}"
    assert desc.endswith(f"experiments_benchmark/ntt_downstream_{root.name}/")
    download.assert_called_once()


def test_benchmark_hf_ckpt_dry_run_downloads_nothing(hf_hub, tmp_path):  # noqa: F811
    download, _, _ = hf_hub
    job = _bench_job(tmp_path, ckpt=_HF_URI)
    cmd, _, _ = job._build_decoding(*_roots(job))
    assert Path(_sweep_cfg(cmd)["ckpt_root"]).name.startswith("conv-vae-100to1-")
    assert not (tmp_path / "results" / "downstream" / "benchmark_roots").exists()
    download.assert_not_called()


def test_benchmark_hf_tags_are_deterministic_and_distinct():
    def tag(uri: str) -> str:
        return _decoding_job(mode="benchmark", ckpt=uri, session=None)._benchmark_tag()

    assert tag(_HF_URI + "@v1") == tag(_HF_URI + "@v1")
    other_name = _HF_URI.replace("conv-vae-100to1", "conv-vae-1000to1")
    assert len({tag(_HF_URI), tag(_HF_URI + "@v1"), tag(_HF_URI + "@v2"), tag(other_name)}) == 4


def test_benchmark_local_ckpt_uses_file_stem_as_name(tmp_path):
    ckpt = tmp_path / "mytok.ckpt"
    ckpt.write_bytes(b"x")
    job = _bench_job(tmp_path, ckpt=str(ckpt), dry_run=False)
    cmd, _, _ = job._build_decoding(*_roots(job))
    link = Path(_sweep_cfg(cmd)["ckpt_root"]) / "mytok" / "checkpoints" / "model.ckpt"
    assert link.resolve() == ckpt.resolve()


def test_benchmark_rejects_non_hf_uri_ckpt():
    job = _decoding_job(mode="benchmark", ckpt="mlflow-artifacts:/1/m.ckpt", session=None)
    with pytest.raises(ValueError, match="hf:// URI or a local"):
        job.run()


# ── golden command: encoding ─────────────────────────────────────────────
def test_encoding_command_uses_encode_console_script():
    job = NTTDownstreamJob(task="encoding", ckpt="/abs/m.ckpt", out="/abs/enc", dry_run=True)
    host, sub = _roots(job)
    cmd, _, out = job._build_encoding(host, sub)

    _assert_cli(cmd, sub, "encode")
    assert _flag(cmd, "--tokenizer-ckpt") == _R("/abs/m.ckpt")
    assert _flag(cmd, "--output-root") == _R("/abs/enc")
    # default recipe → the curated configs/cli/encode.yaml entrypoint (no flag).
    assert "--config-name" not in cmd
    # the launcher owns checkpoint_path + enabled=true; ntt still owns these:
    assert "model.loss_mode=mse" in cmd
    assert "++tokenizer.checkpoint_first=true" in cmd
    assert "trainer.tokenizer_mode=frozen" in cmd
    assert "--dry-run" in cmd
    assert out == _R("/abs/enc")


def test_encoding_recipe_loss_seed_and_data_root():
    job = NTTDownstreamJob(
        task="encoding",
        ckpt="/abs/m.ckpt",
        recipe="cowley_v4_perceiver_mse",
        loss_mode="flow_matching",
        seed=7,
        data_root="/abs/data",
        dry_run=True,
    )
    host, sub = _roots(job)
    cmd, _, _ = job._build_encoding(host, sub)
    assert _flag(cmd, "--config-name") == "cowley_v4_perceiver_mse"
    assert _flag(cmd, "--seed") == "7"
    assert _flag(cmd, "--data-root") == _R("/abs/data")
    assert "model.loss_mode=flow_matching" in cmd


# ── encoding: path-form recipes (external Hydra configs) ──────────────────
def _encoding_job(**kw: Any) -> NTTDownstreamJob:
    base: dict[str, Any] = {"task": "encoding", "ckpt": "/abs/m.ckpt", "dry_run": True}
    base.update(kw)
    return NTTDownstreamJob(**base)


def test_path_form_recipe_missing_file_raises_with_resolved_path(tmp_path):
    # A path-form recipe that does not exist must fail loudly, ntt-side, naming
    # the resolved path (the check runs under dry_run too).
    missing = tmp_path / "abs" / "my_recipe.yaml"
    job = _encoding_job(recipe=str(missing))
    host, sub = _roots(job)
    with pytest.raises(ValueError, match=str(missing.resolve())):
        job._build_encoding(host, sub)


def test_path_form_recipe_absolute_emits_config_dir_and_name(tmp_path):
    recipe = tmp_path / "abs" / "my_recipe.yaml"
    recipe.parent.mkdir(parents=True)
    recipe.write_text("defaults:\n  - /cowley_v4_base\n  - _self_\n")
    job = _encoding_job(recipe=str(recipe))
    host, sub = _roots(job)
    cmd, _, _ = job._build_encoding(host, sub)
    assert _flag(cmd, "--config-dir") == str(recipe.parent.resolve())
    assert _flag(cmd, "--config-name") == "my_recipe"


def test_path_form_recipe_relative_resolves_against_host_root(tmp_path):
    # Relative path-form recipes resolve against host_root (the ntt repo root),
    # NOT the cwd — pin that by overriding root_dir.
    rel = "config/downstream/encoding/x.yaml"
    recipe = tmp_path / rel
    recipe.parent.mkdir(parents=True)
    recipe.write_text("defaults:\n  - /cowley_v4_base\n  - _self_\n")
    job = _encoding_job(recipe=rel, root_dir=str(tmp_path))
    host, sub = _roots(job)
    cmd, _, _ = job._build_encoding(host, sub)
    assert _flag(cmd, "--config-dir") == str((tmp_path / rel).parent.resolve())
    assert _flag(cmd, "--config-name") == "x"


def test_path_form_recipe_yml_extension_and_stem_stripping(tmp_path):
    recipe = tmp_path / "dir" / "overlay.yml"
    recipe.parent.mkdir(parents=True)
    recipe.write_text("defaults:\n  - /cowley_v4_base\n  - _self_\n")
    job = _encoding_job(recipe=str(recipe))
    cmd, _, _ = job._build_encoding(*_roots(job))
    assert _flag(cmd, "--config-dir") == str(recipe.parent.resolve())
    assert _flag(cmd, "--config-name") == "overlay"  # .yml stripped


def test_bare_default_recipe_emits_no_config_flags():
    # Bare name: the default recipe goes through configs/cli/encode.yaml.
    job = _encoding_job()
    cmd, _, _ = job._build_encoding(*_roots(job))
    assert "--config-dir" not in cmd
    assert "--config-name" not in cmd


def test_bare_nondefault_recipe_emits_config_name_only():
    job = _encoding_job(recipe="cowley_v4_perceiver_mse")
    cmd, _, _ = job._build_encoding(*_roots(job))
    assert "--config-dir" not in cmd
    assert _flag(cmd, "--config-name") == "cowley_v4_perceiver_mse"


def test_dotted_bare_recipe_stays_bare():
    # a dot is not a path marker — only "/" or a .yaml/.yml suffix is.
    job = _encoding_job(recipe="cowley.v4.base")
    cmd, _, _ = job._build_encoding(*_roots(job))
    assert "--config-dir" not in cmd
    assert _flag(cmd, "--config-name") == "cowley.v4.base"


def test_shipped_example_overlay_builds():
    # The shipped example file (host-root relative) must build into a path-form
    # command: --config-name example_overlay + a --config-dir ending in the
    # config/downstream/encoding folder.
    job = _encoding_job(recipe="config/downstream/encoding/example_overlay.yaml")
    host, sub = _roots(job)
    assert (host / "config/downstream/encoding/example_overlay.yaml").is_file()
    cmd, _, _ = job._build_encoding(host, sub)
    assert _flag(cmd, "--config-name") == "example_overlay"
    assert _flag(cmd, "--config-dir").endswith("config/downstream/encoding")


def test_encoding_flags_precede_positionals_and_extra_is_last():
    # argparse safety + Hydra last-wins: flags first, user extra strictly last.
    job = NTTDownstreamJob(
        task="encoding",
        ckpt="/abs/m.ckpt",
        recipe="cowley_v4_perceiver_mse",
        seed=7,
        data_root="/abs/data",
        extra=["model.depth=4"],
        dry_run=True,
    )
    cmd, _, _ = job._build_encoding(*_roots(job))
    first_positional = cmd.index(_positionals(cmd)[0])
    assert all(cmd.index(t) < first_positional for t in cmd if t.startswith("--"))
    assert cmd[-1] == "model.depth=4"
    assert cmd.index("trainer.tokenizer_mode=frozen") < cmd.index("model.depth=4")


def test_encoding_accepts_seed_and_tokenizer_free_loss():
    # seed is shared now (encode gained a global seed in the contract), and
    # loss validation is owned by the submodule — poisson must NOT raise here.
    job = NTTDownstreamJob(
        task="encoding", ckpt="/abs/m.ckpt", loss_mode="poisson", seed=1, dry_run=True
    )
    assert job.run()["cmd"]


def test_encoding_without_tracking_uri_warns_not_raises(monkeypatch, caplog):
    # The submodule's logger setup owns whether a URI is required; we warn and
    # let the child be authoritative rather than mirror its branch logic.
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    job = NTTDownstreamJob(task="encoding", ckpt="/abs/m.ckpt", dry_run=False)
    host, sub = _roots(job)
    with caplog.at_level(logging.WARNING, logger="ntt.jobs.downstream"):
        cmd, _, _ = job._build_encoding(host, sub)  # must NOT raise
    assert any("MLFLOW_TRACKING_URI" in r.message for r in caplog.records)
    assert "ntt-downstream-encode" in cmd


def test_encoding_with_tracking_uri_does_not_warn(monkeypatch, caplog):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://tracking")
    job = NTTDownstreamJob(task="encoding", ckpt="/abs/m.ckpt", dry_run=False)
    host, sub = _roots(job)
    with caplog.at_level(logging.WARNING, logger="ntt.jobs.downstream"):
        cmd, _, _ = job._build_encoding(host, sub)
    # assert the build actually ran, so the no-warning check isn't vacuous.
    assert "ntt-downstream-encode" in cmd
    assert not any("MLFLOW_TRACKING_URI" in r.message for r in caplog.records)


def test_encoding_dry_run_does_not_warn(monkeypatch, caplog):
    # dry_run short-circuits the URI check entirely (no child will run).
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    job = NTTDownstreamJob(task="encoding", ckpt="/abs/m.ckpt", dry_run=True)
    host, sub = _roots(job)
    with caplog.at_level(logging.WARNING, logger="ntt.jobs.downstream"):
        cmd, _, _ = job._build_encoding(host, sub)
    assert "ntt-downstream-encode" in cmd
    assert not any("MLFLOW_TRACKING_URI" in r.message for r in caplog.records)


# ── run() dry-run dispatch (preflight lenient, no subprocess) ─────────────
def test_run_dry_run_dispatches_and_returns_locals():
    job = _decoding_job()  # submodule not initialised; dry_run -> preflight warns
    result = job.run()
    assert isinstance(result, dict)
    assert "cmd" in result and isinstance(result["cmd"], list)
    assert "ntt-downstream-decode" in result["cmd"]


def test_run_dry_run_encoding_command_uses_no_sync():
    job = NTTDownstreamJob(task="encoding", ckpt="/abs/model.ckpt", dry_run=True)
    cmd = job.run()["cmd"]
    assert "--no-sync" in cmd
    assert "--frozen" not in cmd


# ── _ensure_env: the host-ntt overlay on the submodule's env ─────────────
_SUB_PYPROJECT = """\
[project]
name = "ntt-downstream"
[tool.uv]
override-dependencies = ["pytorch-brain; python_version > '99'", "einops>=0.6.0,<0.7"]
"""
_SUB_LOCK = """\
version = 1
[[package]]
name = "mlflow"
version = "3.13.0"
[[package]]
name = "torch"
version = "2.8.0+cu126"
[[package]]
name = "ntt"
version = "0.1.0"
"""
# A deliberate downstream override (einops<0.7) that `uv pip check` always flags.
_OVERRIDE_ONLY_CHECK = (
    "Found 1 incompatibility\n"
    "The package `vector-quantize-pytorch` requires `einops>=0.8.0`, but `0.6.1` is installed\n"
)


class _FakeUv:
    """Stands in for ``subprocess.run``: a submodule venv in some state.

    ``ntt_file`` is what ``import ntt`` resolves to (None: venv missing);
    ``check`` is the ``uv pip check`` (returncode, stdout). The ``uv pip install``
    call applies ``after_build`` to the state, simulating the overlay.
    """

    _LISTED = (
        {"name": "mlflow", "version": "3.14.0"},
        {"name": "torch", "version": "2.8.0+cu126"},
        {"name": "ntt", "version": "0.1.2"},
    )

    def __init__(
        self,
        python: Path,
        ntt_file: str | None,
        check=(0, ""),
        after_build=None,
        listed=_LISTED,
        install=(0, ""),
    ):
        self.python, self.ntt_file, self.check = python, ntt_file, check
        self.after_build = after_build or {}
        self.listed = list(listed)  # what `uv pip list` reports
        self.install = install  # (returncode, stderr) of `uv pip install`
        self.calls: list[tuple[list[str], dict]] = []

    def __call__(self, cmd, **kw):
        cmd = [str(c) for c in cmd]
        self.calls.append((cmd, kw))
        out, err, rc = "", "", 0
        if cmd[0] == str(self.python):
            if self.ntt_file is None:
                rc = 1
            else:
                out = self.ntt_file + "\n"
        elif cmd[:3] == ["uv", "pip", "check"]:
            rc, out = self.check
        elif cmd[:3] == ["uv", "pip", "list"]:
            out = json.dumps(self.listed)
        elif cmd[:2] == ["uv", "sync"]:
            self.python.parent.mkdir(parents=True, exist_ok=True)
            self.python.touch()
        elif cmd[:3] == ["uv", "pip", "install"]:
            rc, err = self.install
            if rc == 0:
                for k, v in self.after_build.items():
                    setattr(self, k, v)
        if kw.get("check") and rc:
            raise subprocess.CalledProcessError(rc, cmd)
        return subprocess.CompletedProcess(cmd, rc, stdout=out, stderr=err)

    def argvs(self) -> list[list[str]]:
        return [c for c, _ in self.calls]

    def builds(self) -> list[list[str]]:
        return [
            c for c in self.argvs() if c[:2] == ["uv", "sync"] or c[:3] == ["uv", "pip", "install"]
        ]


@pytest.fixture
def overlay(tmp_path, monkeypatch):
    """Sibling host + submodule checkouts (the eval's layout: never ``<sub>/..``)."""
    host = tmp_path / "ntt"
    (host / "src" / "ntt").mkdir(parents=True)
    (host / "pyproject.toml").write_text('[project]\nname = "ntt"\n')
    sub = tmp_path / "ntt_downstream"
    sub.mkdir()
    (sub / "pyproject.toml").write_text(_SUB_PYPROJECT)
    (sub / "uv.lock").write_text(_SUB_LOCK)
    monkeypatch.setenv("VIRTUAL_ENV", "/host/ntt/.venv")  # must never reach the children

    def make(state: str = "healthy", **fake_kw):
        python = sub / ".venv" / "bin" / "python"
        good = str(host / "src" / "ntt" / "__init__.py")
        job_kw = fake_kw.pop("job_kw", {})
        job = NTTDownstreamJob(
            task="encoding",
            ckpt="/abs/model.ckpt",
            root_dir=str(host),
            submodule_dir=str(sub),
            out=str(tmp_path / "out"),
            **job_kw,
        )
        if state == "healthy":
            python.parent.mkdir(parents=True)
            python.touch()
            (sub / ".venv" / "ntt-overlay.sha256").write_text(job._env_fingerprint(sub, host))
            fake = _FakeUv(python, good, **fake_kw)
        elif state == "missing":
            fake = _FakeUv(python, None, after_build={"ntt_file": good}, **fake_kw)
        else:  # "locked": the venv has the lock's ntt
            python.parent.mkdir(parents=True)
            python.touch()
            locked = "/x/ntt_downstream/.venv/lib/python3.13/site-packages/ntt/__init__.py"
            fake = _FakeUv(python, locked, after_build={"ntt_file": good}, **fake_kw)
        monkeypatch.setattr("ntt.jobs.downstream.subprocess.run", fake)
        return job, host, sub, fake

    return make


def test_ensure_env_healthy_does_not_build(overlay, caplog):
    job, host, sub, fake = overlay("healthy")
    with caplog.at_level(logging.INFO, logger="ntt.jobs.downstream"):
        record = job._ensure_env(sub, host)
    assert fake.builds() == []
    assert record["built"] is False
    assert record["ntt_file"] == str(host / "src" / "ntt" / "__init__.py")
    assert any("env OK" in r.message for r in caplog.records)
    assert any(str(host / "src" / "ntt") in r.message for r in caplog.records)


def test_ensure_env_tolerates_pip_check_findings_on_overridden_packages(overlay):
    job, host, sub, fake = overlay("healthy", check=(1, _OVERRIDE_ONLY_CHECK))
    job._ensure_env(sub, host)
    assert fake.builds() == []


def test_ensure_env_wrong_ntt_syncs_then_installs_host(overlay):
    job, host, sub, fake = overlay("locked")
    record = job._ensure_env(sub, host)
    sync, install = fake.builds()
    assert sync[:2] == ["uv", "sync"]
    assert _flag(sync, "--project") == str(sub)
    assert "--frozen" in sync
    assert _flag(sync, "--no-install-package") == "ntt"
    assert _flag(sync, "--python") == "3.13"
    assert install[:3] == ["uv", "pip", "install"]
    assert _flag(install, "--python") == str(sub / ".venv" / "bin" / "python")
    assert _flag(install, "-e") == str(host)
    assert "--no-sources" in install
    assert _flag(install, "--resolution") == "lowest-direct"
    assert "--no-deps" not in install  # the host's dependency floors must be honoured
    assert record["built"] is True
    for cmd, kw in fake.calls:
        # cwd=<sub> is load-bearing for the install: uv pip reads the submodule's
        # [tool.uv] override-dependencies from the cwd's pyproject.toml.
        assert kw["cwd"] == sub, cmd
        assert "VIRTUAL_ENV" not in kw["env"], cmd
    assert (sub / ".venv" / "ntt-overlay.sha256").read_text() == job._env_fingerprint(sub, host)


def test_ensure_env_builds_a_missing_venv(overlay):
    job, host, sub, fake = overlay("missing")
    job._ensure_env(sub, host)
    assert [c[:2] for c in fake.builds()] == [["uv", "sync"], ["uv", "pip"]]


def test_ensure_env_rebuilds_when_lock_changed_since_build(overlay):
    # --no-sync means nothing else re-syncs after a submodule bump: the stamp must.
    job, host, sub, fake = overlay("healthy")
    (sub / "uv.lock").write_text(_SUB_LOCK + "# bumped\n")
    job._ensure_env(sub, host)
    assert len(fake.builds()) == 2


def test_ensure_env_verify_raises_and_never_builds(overlay):
    job, host, sub, fake = overlay("locked", job_kw={"downstream_env": "verify"})
    with pytest.raises(RuntimeError, match="downstream_env=verify") as exc:
        job._ensure_env(sub, host)
    assert fake.builds() == []
    assert "python -m ntt.jobs.downstream_env" in str(exc.value)  # the build helper


def test_ensure_env_verify_passes_on_a_healthy_env(overlay):
    job, host, sub, fake = overlay("healthy", job_kw={"downstream_env": "verify"})
    assert job._ensure_env(sub, host)["built"] is False


def test_ensure_env_pip_check_failure_after_build_raises_with_hint(overlay):
    bad = (1, "The package `ntt` requires `mlflow>=9`, but `3.14.0` is installed\n")
    job, host, sub, fake = overlay("locked", check=bad)
    with pytest.raises(RuntimeError, match="bump downstream's ntt pin") as exc:
        job._ensure_env(sub, host)
    assert "mlflow>=9" in str(exc.value)
    assert len(fake.builds()) == 2


def test_dry_run_makes_no_subprocess_calls(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise AssertionError(f"subprocess.run called under dry_run: {a}")

    monkeypatch.setattr("ntt.jobs.downstream.subprocess.run", boom)
    NTTDownstreamJob(task="encoding", ckpt="/abs/model.ckpt", dry_run=True).run()
    _decoding_job().run()


def test_preflight_verify_without_venv_gives_overlay_commands(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "pyproject.toml").touch()
    job = NTTDownstreamJob(
        task="encoding", ckpt="/abs/model.ckpt", submodule_dir=str(sub), downstream_env="verify"
    )
    with pytest.raises(RuntimeError, match="python -m ntt.jobs.downstream_env"):
        job._preflight(sub, job._host_root())
    # auto: a missing venv is not a preflight error (_ensure_env builds it)
    NTTDownstreamJob(task="encoding", ckpt="/abs/model.ckpt")._preflight(sub, tmp_path)


def test_run_writes_env_provenance_next_to_outputs(overlay):
    job, host, sub, fake = overlay("locked")
    job.run()
    record = json.loads((Path(job.out) / "downstream_env.json").read_text())
    assert record["ntt_file"] == str(host / "src" / "ntt" / "__init__.py")
    assert record["changes_vs_lock"]["mlflow"] == ["3.13.0", "3.14.0"]
    assert record["changes_vs_lock"]["ntt"] == ["0.1.0", "0.1.2"]
    assert "torch" not in record["changes_vs_lock"]
    assert record["built"] is True
    # the child launch is the last call, after the env is ready
    assert "ntt-downstream-encode" in fake.argvs()[-1]


# ── `uv pip check`: only the two findings the overrides cause on purpose ──
_KNOWN_CHECK = (
    "Checked 269 packages in 4ms\n"
    "Found 2 incompatibilities\n"
    "The package `ntt` requires `pytorch-brain`, but it's not installed\n"
    "The package `vector-quantize-pytorch` requires `einops>=0.8.0`, but `0.6.1` is installed\n"
)


def test_ensure_env_tolerates_the_two_known_override_findings(overlay):
    job, host, sub, fake = overlay(
        "healthy", check=(1, _KNOWN_CHECK), job_kw={"downstream_env": "verify"}
    )
    assert job._ensure_env(sub, host)["built"] is False


def test_ensure_env_raises_when_the_host_needs_what_an_override_prevents(overlay):
    # The host raised its einops floor to 0.8; downstream's override keeps 0.6.1.
    # Before: the finding was dropped because einops is overridden -> silent 0.6.1.
    check = (
        1,
        _KNOWN_CHECK + "The package `ntt` requires `einops>=0.8.0`, but `0.6.1` is installed\n",
    )
    job, host, sub, fake = overlay("locked", check=check)
    with pytest.raises(RuntimeError, match="bump downstream's ntt pin") as exc:
        job._ensure_env(sub, host)
    assert "`ntt` requires `einops>=0.8.0`" in str(exc.value)
    assert "vector-quantize-pytorch" not in str(exc.value)  # the known pair stays tolerated


@pytest.mark.parametrize(
    "output",
    [
        "error: failed to query the environment\n",
        _KNOWN_CHECK + "something uv has never printed before\n",
        "The package `ntt` is broken in a new way\n",
    ],
)
def test_ensure_env_unparseable_pip_check_fails_closed(overlay, output):
    job, host, sub, fake = overlay(
        "healthy", check=(1, output), job_kw={"downstream_env": "verify"}
    )
    with pytest.raises(RuntimeError, match="uv pip check"):
        job._ensure_env(sub, host)


def test_ensure_env_raises_on_a_non_tolerated_finding(overlay):
    # torch-brain is not overridden: a real conflict even on an otherwise healthy env.
    check = (
        1,
        _KNOWN_CHECK + "The package `torch-brain` requires `numpy>=3`, but `2.1.0` is installed\n",
    )
    job, host, sub, fake = overlay("healthy", check=check, job_kw={"downstream_env": "verify"})
    with pytest.raises(RuntimeError, match="torch-brain") as exc:
        job._ensure_env(sub, host)
    assert "einops" not in str(exc.value)


# ── new host dependencies / build failures ───────────────────────────────
def test_ensure_env_raises_on_a_host_dependency_not_in_the_lock(overlay):
    # lowest-direct would install the OLDEST release of an unpinned new package.
    listed = [*_FakeUv._LISTED, {"name": "tabulate", "version": "0.3"}]
    job, host, sub, fake = overlay("locked")
    fake.after_build["listed"] = listed  # the install adds it
    with pytest.raises(RuntimeError, match="not in downstream's lock") as exc:
        job._ensure_env(sub, host)
    assert "tabulate 0.3" in str(exc.value)
    assert "bump downstream's ntt pin" in str(exc.value)


def test_ensure_env_steady_state_also_refuses_a_new_dependency(overlay):
    # The second launch must not sail through on the stamp alone.
    listed = [*_FakeUv._LISTED, {"name": "tabulate", "version": "0.3"}]
    job, host, sub, fake = overlay("healthy", listed=listed, job_kw={"downstream_env": "verify"})
    with pytest.raises(RuntimeError, match="tabulate"):
        job._ensure_env(sub, host)


def test_ensure_env_build_failure_carries_hint_and_stderr_tail(overlay):
    stderr = (
        "Resolved 12 packages\n"
        "  × No solution found when resolving dependencies:\n"
        "  ╰─▶ ntt depends on mlflow>=99\n"
    )
    job, host, sub, fake = overlay("locked", install=(1, stderr))
    with pytest.raises(RuntimeError, match="bump downstream's ntt pin") as exc:
        job._ensure_env(sub, host)
    assert "mlflow>=99" in str(exc.value)
    assert "uv pip install" in str(exc.value)
    assert not (sub / ".venv" / "ntt-overlay.sha256").exists()


def test_ensure_env_warns_when_a_tokenizer_numerics_package_moves(overlay, caplog):
    listed = [*_FakeUv._LISTED, {"name": "einops", "version": "0.8.1"}]
    job, host, sub, fake = overlay("healthy", listed=listed)
    (sub / "uv.lock").write_text(_SUB_LOCK + '[[package]]\nname = "einops"\nversion = "0.6.1"\n')
    (sub / ".venv" / "ntt-overlay.sha256").write_text(job._env_fingerprint(sub, host))
    with caplog.at_level(logging.WARNING, logger="ntt.jobs.downstream"):
        job._ensure_env(sub, host)
    assert any("einops" in r.message and r.levelno == logging.WARNING for r in caplog.records)


def test_core_packages_cover_tokenizer_numerics():
    assert {"vector-quantize-pytorch", "einops", "flash-attn"} <= ds._CORE_PACKAGES


# ── the stamp: an unstamped env is never trusted ─────────────────────────
def test_ensure_env_auto_rebuilds_an_unstamped_env(overlay):
    job, host, sub, fake = overlay("healthy")
    (sub / ".venv" / "ntt-overlay.sha256").unlink()
    assert job._ensure_env(sub, host)["built"] is True
    assert (sub / ".venv" / "ntt-overlay.sha256").read_text() == job._env_fingerprint(sub, host)


def test_ensure_env_verify_refuses_an_unstamped_env_with_the_helper(overlay):
    job, host, sub, fake = overlay("healthy", job_kw={"downstream_env": "verify"})
    (sub / ".venv" / "ntt-overlay.sha256").unlink()
    with pytest.raises(RuntimeError, match="no overlay stamp") as exc:
        job._ensure_env(sub, host)
    assert "python -m ntt.jobs.downstream_env" in str(exc.value)
    assert fake.builds() == []


def test_main_builds_and_stamps_the_env(overlay, capsys):
    job, host, sub, fake = overlay("locked")
    downstream_env.main(["--root-dir", str(host), "--submodule-dir", str(sub)])
    assert len(fake.builds()) == 2
    assert (sub / ".venv" / "ntt-overlay.sha256").is_file()
    assert json.loads(capsys.readouterr().out)["built"] is True


# ── the build lock: concurrent `auto` launches on one shared checkout ─────
def test_ensure_env_waits_for_a_build_in_progress_then_rechecks(overlay, monkeypatch):
    job, host, sub, fake = overlay("locked")
    lock = sub / ds._BUILD_LOCK
    lock.mkdir()
    sleeps = []

    def other_job_finishes(seconds):  # the lock holder builds, stamps and releases
        sleeps.append(seconds)
        fake.ntt_file = str(host / "src" / "ntt" / "__init__.py")
        (sub / ".venv" / "ntt-overlay.sha256").write_text(job._env_fingerprint(sub, host))
        lock.rmdir()

    monkeypatch.setattr(ds.time, "sleep", other_job_finishes)
    record = job._ensure_env(sub, host)
    assert len(sleeps) == 1
    assert fake.builds() == []  # the re-check under the lock saw the finished build
    assert record["built"] is False
    assert not lock.exists()


def test_ensure_env_breaks_a_stale_lock(overlay, monkeypatch):
    job, host, sub, fake = overlay("locked")
    lock = sub / ds._BUILD_LOCK
    lock.mkdir()
    (lock / "owner").write_text("deadhost pid=1")
    old = ds.time.time() - ds._LOCK_STALE_S - 1
    os.utime(lock, (old, old))
    monkeypatch.setattr(ds.time, "sleep", lambda s: pytest.fail("waited on a stale lock"))
    assert job._ensure_env(sub, host)["built"] is True
    assert not lock.exists()


def test_ensure_env_lock_timeout_names_the_lock_and_owner(overlay, monkeypatch):
    job, host, sub, fake = overlay("locked")
    lock = sub / ds._BUILD_LOCK
    lock.mkdir()
    (lock / "owner").write_text("node42 pid=7")
    monkeypatch.setattr(ds, "_LOCK_TIMEOUT_S", 0)
    monkeypatch.setattr(ds.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="node42 pid=7") as exc:
        job._ensure_env(sub, host)
    assert str(lock) in str(exc.value)
    assert fake.builds() == []
    assert lock.exists()  # never remove a live lock we do not own


def test_ensure_env_releases_the_lock_when_the_build_fails(overlay):
    job, host, sub, fake = overlay("locked", install=(1, "boom\n"))
    with pytest.raises(RuntimeError):
        job._ensure_env(sub, host)
    assert not (sub / ds._BUILD_LOCK).exists()


# ── config validation runs before the (possibly multi-minute) env build ──
def test_run_validates_the_command_before_touching_the_env(overlay):
    job, host, sub, fake = overlay("locked", job_kw={"recipe": "missing/recipe.yaml"})
    with pytest.raises(ValueError, match="does not exist"):
        job.run()
    assert fake.calls == []


def test_downstream_ckpt_materializes_hf(hf_hub):  # noqa: F811
    download, _, _ = hf_hub
    job = _decoding_job(ckpt=_HF_URI, dry_run=False)
    resolved = job._ckpt_path()
    assert resolved is not None
    assert resolved.endswith(".ckpt") and Path(resolved).is_file()
    download.assert_called_once()


def test_downstream_ckpt_hf_not_downloaded_in_dry_run(hf_hub):  # noqa: F811
    download, _, _ = hf_hub
    job = _encoding_job(ckpt=_HF_URI)
    assert job._ckpt_path() == _HF_URI
    host, sub = _roots(job)
    cmd, _, _ = job._build_encoding(host, sub)
    assert cmd[cmd.index("--tokenizer-ckpt") + 1] == _HF_URI
    download.assert_not_called()


def test_downstream_abspath_leaves_uris_alone():
    job = _decoding_job()
    assert job._abspath("mlflow-artifacts:/r/model.ckpt") == "mlflow-artifacts:/r/model.ckpt"


@pytest.mark.parametrize("field", ["data_root", "out", "ckpt_root"])
def test_downstream_rejects_hf_outside_ckpt(field):
    kw = {field: _HF_URI}
    if field == "ckpt_root":
        kw.update(ckpt=None, mode="benchmark")
    with pytest.raises(
        ValueError, match=f"job.{field}=.*hf:// paths are accepted only for job.ckpt"
    ):
        _decoding_job(**kw)
