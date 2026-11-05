"""Tests for scripts/eval/published_commands.py (published eval protocol -> ntt commands)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HF = "hf://the-enigma-project/ntt-tokenizers"


@pytest.fixture
def pc():
    script_dir = REPO_ROOT / "scripts" / "eval"
    sys.path.insert(0, str(script_dir))
    try:
        import published_commands

        importlib.reload(published_commands)
    finally:
        sys.path.pop(0)
    return published_commands


def _commands(pc, model, axes, revision="v1.0"):
    return pc.commands(
        model, axes, data_dir=Path("/data"), output_dir=Path("/out"), revision=revision
    )


def test_intrinsic_axes_are_one_eval_all_run_with_those_passes(pc):
    [cmd] = _commands(pc, "conv-vae-100to1", ["recon", "snr"])

    assert cmd[0] == "job=published_eval"
    assert f"job.ckpt_path={HF}/conv-vae-100to1@v1.0" in cmd
    assert "paths.data_dir=/data" in cmd
    assert "job.output_path=/out/conv-vae-100to1/eval" in cmd
    assert {
        "job.main.enabled=true",
        "job.snr.enabled=true",
        "job.input_robustness.enabled=false",
        "job.latent_robustness.enabled=false",
    } <= set(cmd)


def test_a_model_with_its_own_eval_config_uses_it(pc):
    [cmd] = _commands(pc, "ldns", ["m1"])
    assert cmd[0] == "job=published_eval_ldns"


def test_no_revision_means_the_hub_default_branch(pc):
    [cmd] = _commands(pc, "conv-vae-100to1", ["recon"], revision=None)
    assert f"job.ckpt_path={HF}/conv-vae-100to1" in cmd


def test_encoding_uses_the_model_recipe_and_no_mlflow(pc):
    [cmd] = _commands(pc, "conv-fsq-100to1", ["encoding"])

    assert cmd[0] == "job=downstream_encoding"
    assert "job.recipe=config/downstream/encoding/published_conv-fsq-100to1.yaml" in cmd
    assert "job.loss_mode=cross_entropy" in cmd
    assert "job.out=/out/conv-fsq-100to1/encoding" in cmd
    [extra] = [a for a in cmd if a.startswith("job.extra=")]
    assert '"data.dataset_path=/data/cowley_v4_compact_2024"' in extra
    assert '"trainer.logging=false"' in extra


def test_models_without_an_encoding_score_get_no_encoding_command(pc):
    assert _commands(pc, "pca-1000to1", ["encoding"]) == []


@pytest.mark.parametrize(
    ("model", "gpus", "extra"),
    [
        ("conv-vae-100to1", "job.gpus=0", None),
        ("ldns-perceiver", 'job.gpus="0 1 2 3"', None),
        ("conv-vae-1000to1", "job.gpus=0", 'job.extra=["tokenizer.fixed_window_seconds=1.0"]'),
        # population encoders read neurons by position: every session unit, in a fixed order
        ("ldns", "job.gpus=0", 'job.extra=["tokenizer.full_window_units=true"]'),
        ("wavelet-fsq", "job.gpus=0", 'job.extra=["tokenizer.full_window_units=true"]'),
    ],
)
def test_decoding_is_the_benchmark_with_the_model_settings(pc, model, gpus, extra):
    [cmd] = _commands(pc, model, ["decoding"])

    assert cmd[:2] == ["job=downstream_decoding", "job.mode=benchmark"]
    assert f"job.ckpt={HF}/{model}@v1.0" in cmd
    assert "job.data_root=/data" in cmd
    assert {"job.epochs=400", "job.model_cfg=poyo_1.3M_no_crossattn", gpus} <= set(cmd)
    assert (extra in cmd) if extra else not any(a.startswith("job.extra=") for a in cmd)
    # Off even when MLFLOW_TRACKING_URI is set (the sweep's own default would then log).
    assert "job.mlflow=false" in cmd


def test_with_mlflow_the_downstream_runs_log_to_the_users_server(pc):
    enc, dec = pc.commands(
        "conv-vae-100to1",
        ["encoding", "decoding"],
        data_dir=Path("/data"),
        output_dir=Path("/out"),
        revision=None,
        mlflow=True,
    )
    assert not any("trainer.logging" in arg for arg in enc)
    assert "job.mlflow=true" in dec


def test_unknown_model_names_the_known_ones(pc):
    with pytest.raises(ValueError, match="conv-vae-100to1"):
        _commands(pc, "no-such-model", ["recon"])


def test_every_referenced_config_exists_and_every_eval_config_is_a_model(pc):
    models = pc.load_protocol()
    for name, entry in models.items():
        if entry["encoding"]:
            recipe = f"{entry['encoding']['recipe']}.yaml"
            assert (REPO_ROOT / "config/downstream/encoding" / recipe).is_file(), name
    # Flat names: Hydra names the run's log file after the job choice, so a "/" breaks it.
    eval_configs = {p.stem for p in (REPO_ROOT / "config/job").glob("published_eval_*.yaml")}
    assert {c.removeprefix("published_eval_") for c in eval_configs} <= set(models)
    assert not (REPO_ROOT / "config/job/published_eval").exists()


def _decoding_output(downstream: Path, model: str, revision: str | None) -> Path:
    from ntt.jobs.downstream import hub_benchmark_tag

    uri = f"{HF}/{model}" + (f"@{revision}" if revision else "")
    folder = downstream / "experiments_benchmark" / hub_benchmark_tag(uri)
    folder.mkdir(parents=True)
    return folder


def test_hub_benchmark_tag_names_the_folder_a_decoding_run_wrote():
    """The folder a real decoding benchmark of this checkpoint wrote."""
    from ntt.jobs.downstream import hub_benchmark_tag

    sha = "64d8bb47951d4d1b06765a74db0e7ae8aec36436"
    tag = hub_benchmark_tag(f"{HF}/conv-vae-100to1@{sha}")
    assert tag == "ntt_downstream_conv-vae-100to1-bc22d371"


def test_tracker_rows_point_at_each_axis_output_and_split_by_table(pc, tmp_path):
    out, downstream = tmp_path / "results", tmp_path / "downstream"
    decoding = _decoding_output(downstream, "conv-vae-100to1", "v1.0")

    for folder in ("eval", "eval_snr", "eval_m1", "eval_m2", "encoding"):
        (out / "conv-vae-100to1" / folder).mkdir(parents=True)
    (out / "pca-1000to1" / "eval").mkdir(parents=True)

    trackers = pc.tracker_rows(
        ["conv-vae-100to1", "pca-1000to1", "ldns"], out, downstream, revision="v1.0"
    )

    assert set(trackers) == {"200ms", "1000ms"}  # ldns has no results: left out
    [row] = trackers["200ms"]
    assert row == {
        "tokenizer": "conv-vae-100to1",
        "mlflow_recon": str(out / "conv-vae-100to1/eval"),
        "mlflow_latent_snr": str(out / "conv-vae-100to1/eval_snr"),
        "mlflow_latent_input_corruption": str(out / "conv-vae-100to1/eval_m1"),
        "mlflow_latent_latent_corruption": str(out / "conv-vae-100to1/eval_m2"),
        "mlflow_encoding": str(out / "conv-vae-100to1/encoding"),
        "mlflow_decoding": str(decoding),
    }
    [pca] = trackers["1000ms"]
    assert pca["mlflow_encoding"] == ""  # no encoding score
    assert pca["mlflow_decoding"] == ""  # no decoding output found


def test_tracker_takes_the_decoding_output_of_the_requested_revision_only(pc, tmp_path):
    """Another revision's decoding output is never put in the row."""
    out, downstream = tmp_path / "results", tmp_path / "downstream"
    (out / "conv-vae-100to1/eval").mkdir(parents=True)
    old = _decoding_output(downstream, "conv-vae-100to1", "v1.0")

    def decoding_cell(revision):
        rows = pc.tracker_rows(["conv-vae-100to1"], out, downstream, revision=revision)
        return rows["200ms"][0]["mlflow_decoding"]

    assert decoding_cell("v1.0") == str(old)
    assert decoding_cell("v1.1") == ""  # not run for v1.1
    assert decoding_cell(None) == ""  # main is not v1.0
    new = _decoding_output(downstream, "conv-vae-100to1", "v1.1")
    assert decoding_cell("v1.1") == str(new)


def test_tracker_does_not_take_a_longer_model_names_decoding_output(pc, tmp_path):
    """wavelet-fsq must not pick up wavelet-fsq-8-5-5-5-levels' benchmark folder."""
    downstream = tmp_path / "downstream"
    _decoding_output(downstream, "wavelet-fsq-8-5-5-5-levels", None)
    (tmp_path / "results/wavelet-fsq/eval").mkdir(parents=True)

    [row] = pc.tracker_rows(["wavelet-fsq"], tmp_path / "results", downstream, revision=None)[
        "200ms"
    ]
    assert row["mlflow_decoding"] == ""


def test_tracker_names_only_outputs_that_exist(pc, tmp_path):
    out = tmp_path / "results"
    (out / "conv-vae-100to1" / "eval").mkdir(parents=True)  # recon only

    [row] = pc.tracker_rows(["conv-vae-100to1"], out, tmp_path / "downstream", revision=None)[
        "200ms"
    ]
    assert row["mlflow_recon"] == str(out / "conv-vae-100to1/eval")
    assert row["mlflow_latent_snr"] == row["mlflow_encoding"] == ""


def _main(pc, monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["published_commands.py", *argv])
    pc.main()


def test_main_defaults_to_v1_0_and_a_results_folder_per_revision(pc, monkeypatch, capsys):
    _main(pc, monkeypatch, "conv-vae-100to1", "--axes", "recon", "--data-dir", "/data")
    out = capsys.readouterr().out
    assert f"job.ckpt_path={HF}/conv-vae-100to1@v1.0" in out
    assert f"job.output_path={Path('results/v1.0').resolve()}/conv-vae-100to1/eval" in out


def test_main_refuses_the_main_branch(pc, monkeypatch):
    """main can change under the same name, so its outputs could not be told apart."""
    with pytest.raises(SystemExit):
        _main(pc, monkeypatch, "conv-vae-100to1", "--data-dir", "/data", "--revision", "main")


@pytest.mark.parametrize(
    "extra,folder",
    [
        (["--revision", "v2.0"], "results/v2.0"),
        (
            ["--revision", "64d8bb47951d4d1b06765a74db0e7ae8aec36436"],
            "results/64d8bb47951d4d1b06765a74db0e7ae8aec36436",
        ),
        (["--revision", "v2.0", "--output-dir", "mine"], "mine"),
    ],
)
def test_main_output_folder_follows_the_revision_unless_given(
    pc, monkeypatch, capsys, extra, folder
):
    _main(pc, monkeypatch, "conv-vae-100to1", "--axes", "recon", "--data-dir", "/data", *extra)
    out = capsys.readouterr().out
    assert f"job.output_path={Path(folder).resolve()}/conv-vae-100to1/eval" in out
