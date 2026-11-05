"""Contract smoke test: run the REAL ntt-downstream console scripts, GPU-free.

These execute the commands NTTDownstreamJob builds against the pinned
submodule, using the contract's own preview flags (--print-resolved-config for
single runs — full Hydra composition; --dry-run for sweeps — the engine's plan
preview). A flag/key drift in the submodule fails HERE instead of at a user's
first real run. Skipped wherever the submodule venv is not
built (e.g. CI).
"""

import subprocess
from pathlib import Path

import pytest

from ntt.jobs.downstream import NTTDownstreamJob

_HOST = Path(NTTDownstreamJob(task="encoding", ckpt="/dev/null", dry_run=True)._host_root())
_SUB = _HOST / "downstream"

pytestmark = pytest.mark.skipif(
    not (_SUB / ".venv").is_dir(),
    reason="downstream submodule venv not built (one real ntt job=downstream_* launch builds it)",
)


def _execute(
    job: NTTDownstreamJob, *, swap_preview: bool = False
) -> subprocess.CompletedProcess[str]:
    if job.task == "encoding":
        cmd, env, _ = job._build_encoding(_HOST, _SUB)
    else:
        cmd, env, _ = job._build_decoding(_HOST, _SUB)
    if swap_preview:
        # single runs: exercise full Hydra composition, not just argv assembly.
        cmd[cmd.index("--dry-run")] = "--print-resolved-config"
    proc = subprocess.run(cmd, cwd=_SUB, env=env, capture_output=True, text=True)
    assert proc.returncode == 0, (
        f"contract drift?\ncmd: {cmd}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    return proc


def test_decode_composes(tmp_path):
    ckpt = tmp_path / "tok.ckpt"
    ckpt.touch()
    _execute(
        NTTDownstreamJob(
            task="decoding",
            mode="single-run",
            ckpt=str(ckpt),
            session="sess_x",
            out=str(tmp_path / "out"),
            dry_run=True,
        ),
        swap_preview=True,
    )


def test_encode_composes(tmp_path):
    ckpt = tmp_path / "tok.ckpt"
    ckpt.touch()
    _execute(
        NTTDownstreamJob(task="encoding", ckpt=str(ckpt), out=str(tmp_path / "out"), dry_run=True),
        swap_preview=True,
    )


def test_encode_external_recipe_composes(tmp_path):
    # The SHIPPED external overlay (host-root relative) must compose against the
    # submodule's internal tree: Hydra adds config/downstream/encoding/ to its
    # search path, so `/cowley_v4_base` resolves inside the submodule and the
    # overlay's trainer.max_epochs=50 lands in the resolved config.
    ckpt = tmp_path / "tok.ckpt"
    ckpt.touch()
    proc = _execute(
        NTTDownstreamJob(
            task="encoding",
            ckpt=str(ckpt),
            recipe="config/downstream/encoding/example_overlay.yaml",
            out=str(tmp_path / "out"),
            dry_run=True,
        ),
        swap_preview=True,
    )
    assert "max_epochs: 50" in proc.stdout, (
        f"external overlay did not compose against the internal tree:\n{proc.stdout}"
    )


def test_encode_cowley_ldns_recipe_composes(tmp_path):
    # cowley_ldns.yaml reproduces the original ldns_baseline_10_ms run; it must
    # still compose against the pinned submodule (no `- model:` group).
    ckpt = tmp_path / "tok.ckpt"
    ckpt.touch()
    proc = _execute(
        NTTDownstreamJob(
            task="encoding",
            ckpt=str(ckpt),
            recipe="config/downstream/encoding/cowley_ldns.yaml",
            loss_mode="flow_matching",
            data_root=str(tmp_path / "data"),
            out=str(tmp_path / "out"),
            dry_run=True,
        ),
        swap_preview=True,
    )
    assert "compress_n_target_slice: 20" in proc.stdout, proc.stdout


def test_calibrate_dry_run_plans(tmp_path):
    ckpt = tmp_path / "tok.ckpt"
    ckpt.touch()
    _execute(
        NTTDownstreamJob(
            task="decoding",
            mode="calibrate",
            ckpt=str(ckpt),
            out=str(tmp_path / "calib"),
            dry_run=True,
        )
    )


def test_benchmark_dry_run_plans(tmp_path):
    link = tmp_path / "roots" / "tokA" / "checkpoints" / "model.ckpt"
    link.parent.mkdir(parents=True)
    link.touch()
    _execute(
        NTTDownstreamJob(
            task="decoding",
            mode="benchmark",
            ckpt_root=str(tmp_path / "roots"),
            dry_run=True,
        )
    )
