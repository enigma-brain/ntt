"""Focused tests for the ``job=eval_all`` orchestration helpers."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from ntt.jobs.eval_all import NTTEvalAllJob

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


def test_eval_all_secondary_overrides_suffix_run_name_and_output_path():
    job = NTTEvalAllJob()
    job._task_overrides = [
        "job=eval_all",
        "paths.run_name=my_eval",
        "job.output_path=results/my_eval",
        "job.compare_poisson=true",
        "job.input_robustness.enabled=false",
    ]
    job._primary_run_name = "my_eval"
    job._primary_output_path = "results/my_eval"

    passthrough = job._passthrough_overrides()
    overrides = job._build_secondary_overrides(
        passthrough,
        job="eval_robustness_input",
        evaluation="metrics/robustness_input",
        data="robustness_fast_eval",
        run_name_suffix="_m1",
    )

    assert "job=eval_robustness_input" in overrides
    assert "evaluation@job.eval_config=metrics/robustness_input" in overrides
    assert "data@job.data_config=robustness_fast_eval" in overrides
    assert "paths.run_name='my_eval_m1'" in overrides
    assert "job.output_path='results/my_eval_m1'" in overrides
    assert "job.compare_poisson=true" in overrides
    assert all(not override.startswith("job.input_robustness.") for override in overrides)


@pytest.mark.parametrize("run_name", ["my_eval", "'my_eval'", '"my_eval"'])
def test_eval_all_secondary_run_name_drops_the_users_quotes(run_name):
    """A quoted ``paths.run_name='my_eval'`` gives ``my_eval_m1``, not ``'my_eval'_m1``."""
    from ntt.jobs.eval_all import NTTEvalAllJob

    job = NTTEvalAllJob()
    job._task_overrides = ["job=eval_all", f"paths.run_name={run_name}"]

    overrides = job._build_secondary_overrides(
        job._passthrough_overrides(),
        job="eval_robustness_input",
        evaluation="metrics/robustness_input",
        data="robustness_fast_eval",
        run_name_suffix="_m1",
    )

    assert "paths.run_name='my_eval_m1'" in overrides


def test_eval_all_secondary_run_name_null_is_not_the_string_none():
    """``paths.run_name=null`` falls back to the captured run name, not ``'None_m1'``."""
    from ntt.jobs.eval_all import NTTEvalAllJob

    job = NTTEvalAllJob()
    job._task_overrides = ["job=eval_all", "paths.run_name=null"]
    job._primary_run_name = "captured"

    overrides = job._build_secondary_overrides(
        job._passthrough_overrides(),
        job="eval_robustness_input",
        evaluation="metrics/robustness_input",
        data="robustness_fast_eval",
        run_name_suffix="_m1",
    )

    assert "paths.run_name='captured_m1'" in overrides


def test_eval_robustness_input_job_composes_with_defaults():
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="main", overrides=["job=eval_robustness_input"])

    assert "data_config" in cfg.job
    assert "eval_config" in cfg.job
    assert cfg.job.data_config is not None
    assert cfg.job.eval_config is not None
    assert list(cfg.job.eval_config.metrics.keys()) == ["robustness_input"]


def test_input_latent_robustness_eval_resolution_comes_from_metric_yaml_not_mode_override():
    """Eval resolution lives in the metric YAMLs.

    input_robustness (M1) ``max_eval_seconds`` and latent_robustness (M2)
    ``n_eval_windows`` / ``n_samples_per_epsilon`` must live in the metric YAMLs, NOT in
    ``eval_all._MODE_OVERRIDES``. Injecting it per-mode there would
    shadow the YAML silently — forcing latent_robustness (M2) to 64 windows
    despite the YAML's 300, and pinning input_robustness (M1) to 64 s in fast mode.
    ``_MODE_OVERRIDES`` may only
    carry genuinely mode-specific settings (the SNR data config).
    """
    import yaml

    from ntt.jobs.eval_all import _MODE_OVERRIDES

    for mode, cfg in _MODE_OVERRIDES.items():
        assert set(cfg) == {"snr_data"}, (mode, sorted(cfg))

    metrics_dir = CONFIG_DIR / "evaluation" / "metrics"
    input_robustness = yaml.safe_load((metrics_dir / "robustness_input.yaml").read_text())
    latent_robustness = yaml.safe_load((metrics_dir / "robustness_latent.yaml").read_text())
    assert input_robustness["max_eval_seconds"] == 300
    assert latent_robustness["robustness_latent"]["n_eval_windows"] == 300


def _instantiate_job(overrides: list[str]):
    from hydra.core.hydra_config import HydraConfig
    from hydra.utils import instantiate

    from ntt.jobs.eval_all import _compose

    # `_compose` sets the process-wide HydraConfig (as `@hydra.main` would); clear it so
    # later tests that expect no active Hydra context still see none.
    try:
        return instantiate(_compose(overrides, label="test")).job
    finally:
        HydraConfig.instance().cfg = None


def test_compression_n_neurons_is_a_plain_override_of_job_eval():
    """Declared in config/job/eval.yaml: no ``++`` needed, default None."""
    base = ["job=eval", "job.ckpt_path=/tmp/fake.ckpt", "paths.run_name=t"]

    assert _instantiate_job(base).compression_n_neurons is None
    assert _instantiate_job([*base, "job.compression_n_neurons=256"]).compression_n_neurons == 256


def test_compression_n_neurons_and_poisson_reach_only_the_main_pass(monkeypatch):
    """``job.compression_n_neurons`` and the Poisson comparison (settings and reference) are for
    the reconstruction (main) pass: the SNR pass is scored without them, and the M1/M2
    robustness passes score raw values (their own Poisson references are a standalone-job
    option)."""
    import ntt.jobs.eval_all as eval_all

    user = [
        "job=eval_all",
        "job.ckpt_path=/tmp/fake.ckpt",
        "paths.run_name=t",
        "job.compression_n_neurons=256",
        "job.compare_poisson=true",
        "job.poisson_rate_bin_ms=10.0",
        "++job.poisson_reference=/tmp/reference.json",
    ]
    job = _instantiate_job(user)
    assert job.compression_n_neurons == 256
    job._task_overrides = user

    passes: dict[str, list[str]] = {}

    def fake_run_one(overrides, label, **_kw):
        passes[label] = list(overrides)
        return 0.0

    monkeypatch.setattr(eval_all, "_run_one", fake_run_one)
    job.run()

    assert set(passes) == {"main", "snr", "m1", "m2"}
    main = _instantiate_job(passes["main"])
    assert (main.compression_n_neurons, main.compare_poisson, main.poisson_rate_bin_ms) == (
        256,
        True,
        10.0,
    )
    assert main.poisson_reference == "/tmp/reference.json"
    for label in ("snr", "m1", "m2"):
        main_only = [o for o in passes[label] if "compression_n_neurons" in o or "poisson" in o]
        assert not main_only, label
        _instantiate_job(passes[label])  # composes and validates without the fields


def test_eval_all_defaults_to_fast_mode():
    job = NTTEvalAllJob()

    assert job.mode == "fast"
    assert job._mode_cfg()["snr_data"] == "cowley_snr_fast_eval"


@pytest.mark.parametrize(
    ("extra", "main_upload"),
    [
        ([], True),
        (["job.mlflow_logging.upload_checkpoint=false"], False),
        (["++job.mlflow_logging.upload_checkpoint=false"], False),
    ],
)
def test_upload_checkpoint_follows_eval_all_for_main_and_is_off_for_other_passes(
    monkeypatch, extra, main_upload
):
    """The main pass takes eval_all's resolved upload_checkpoint (default true, or the
    user's override in any Hydra form); SNR / M1 / M2 never upload. Each pass gets the key
    exactly once, so a user override cannot collide with the value eval_all sets."""
    import ntt.jobs.eval_all as eval_all

    user = [
        "job=eval_all",
        "job.ckpt_path=/tmp/fake.ckpt",
        "paths.run_name=t",
        "job.mlflow_logging.enabled=true",
        *extra,
    ]
    job = _instantiate_job(user)
    job._task_overrides = user

    passes: dict[str, list[str]] = {}

    def fake_run_one(overrides, label, **_kw):
        passes[label] = list(overrides)
        return 0.0

    monkeypatch.setattr(eval_all, "_run_one", fake_run_one)
    job.run()

    assert set(passes) == {"main", "snr", "m1", "m2"}
    for label, overrides in passes.items():
        assert sum("mlflow_logging.upload_checkpoint=" in o for o in overrides) == 1, label
        expected = main_upload if label == "main" else False
        assert _instantiate_job(overrides).mlflow_logging.upload_checkpoint is expected, label


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        ([], {"main": True, "snr": True, "m1": True, "m2": True}),
        (
            ["job.evaluate_all_neurons=false"],
            {"main": False, "snr": False, "m1": False, "m2": False},
        ),
        (
            [
                "job.input_robustness.evaluate_all_neurons=false",
                "job.latent_robustness.evaluate_all_neurons=false",
            ],
            {"main": True, "snr": True, "m1": False, "m2": False},
        ),
    ],
)
def test_evaluate_all_neurons_reaches_every_pass_with_per_pass_overrides(
    monkeypatch, extra, expected
):
    """Full-neuron evaluation is on by default for every pass, and one pass can be run
    padded (e.g. M1/M2 for cosmos / ldns) while the main recon stays on all neurons."""
    import ntt.jobs.eval_all as eval_all

    user = ["job=eval_all", "job.ckpt_path=/tmp/fake.ckpt", "paths.run_name=t", *extra]
    job = _instantiate_job(user)
    job._task_overrides = user

    passes: dict[str, list[str]] = {}

    def fake_run_one(overrides, label, **_kw):
        passes[label] = list(overrides)
        return 0.0

    monkeypatch.setattr(eval_all, "_run_one", fake_run_one)
    job.run()

    for label, value in expected.items():
        assert sum("evaluate_all_neurons" in o for o in passes[label]) == 1, passes[label]
        assert _instantiate_job(passes[label]).evaluate_all_neurons is value, label


def test_standalone_eval_jobs_default_to_all_neurons():
    for job_name in ("eval", "eval_robustness_input", "eval_robustness_latent"):
        job = _instantiate_job(
            [f"job={job_name}", "job.ckpt_path=/tmp/f.ckpt", "paths.run_name=t"]
        )
        assert job.evaluate_all_neurons is True, job_name


def test_standalone_eval_caps_units_like_eval_all_main():
    """job=eval caps at 200 units, as eval_all's main pass and the Poisson baselines do."""
    job = _instantiate_job(["job=eval", "job.ckpt_path=/tmp/f.ckpt", "paths.run_name=t"])
    assert job.eval_config["max_neurons"] == 200


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        ([], None),
        (["job.eval_window_seconds=native"], "native"),
        (["job.eval_window_seconds=4.0"], 4.0),
    ],
)
def test_eval_window_reaches_every_pass(monkeypatch, extra, expected):
    """``job.eval_window_seconds`` (a number, or native) sets every pass's eval window."""
    import ntt.jobs.eval_all as eval_all

    user = ["job=eval_all", "job.ckpt_path=/tmp/fake.ckpt", "paths.run_name=t", *extra]
    job = _instantiate_job(user)
    job._task_overrides = user

    passes: dict[str, list[str]] = {}

    def fake_run_one(overrides, label, **_kw):
        passes[label] = list(overrides)
        return 0.0

    monkeypatch.setattr(eval_all, "_run_one", fake_run_one)
    job.run()

    assert set(passes) == {"main", "snr", "m1", "m2"}
    for label, overrides in passes.items():
        assert sum("eval_window_seconds" in o for o in overrides) == (expected is not None), label
        window = _instantiate_job(overrides).eval_config.get("eval_window_seconds")
        assert window == expected, label


def test_poisson_reference_reaches_the_main_pass_as_a_plain_override(monkeypatch):
    """``job=eval_all ... job.poisson_reference=<file>`` (no ``++``): the main pass composes
    ``job=eval``, so that config must declare the key."""
    import ntt.jobs.eval_all as eval_all

    user = [
        "job=eval_all",
        "job.ckpt_path=/tmp/fake.ckpt",
        "paths.run_name=t",
        "job.compare_poisson=true",
        "job.poisson_reference=/tmp/reference.json",
        "job.snr.enabled=false",
        "job.input_robustness.enabled=false",
        "job.latent_robustness.enabled=false",
    ]
    job = _instantiate_job(user)
    job._task_overrides = user
    passes: dict[str, list[str]] = {}

    def fake_run_one(overrides, label, **_kw):
        passes[label] = list(overrides)
        return 0.0

    monkeypatch.setattr(eval_all, "_run_one", fake_run_one)
    job.run()

    assert _instantiate_job(passes["main"]).poisson_reference == "/tmp/reference.json"


def _pass_jobs(monkeypatch, user: list[str]) -> dict:
    """Run eval_all with ``_run_one`` stubbed; return each pass's instantiated job."""
    import ntt.jobs.eval_all as eval_all

    job = _instantiate_job(user)
    job._task_overrides = user
    passes: dict[str, list[str]] = {}

    def fake_run_one(overrides, label, **_kw):
        passes[label] = list(overrides)
        return 0.0

    monkeypatch.setattr(eval_all, "_run_one", fake_run_one)
    job.run()
    return {label: _instantiate_job(overrides) for label, overrides in passes.items()}


_RUN = ["job.ckpt_path=/tmp/fake.ckpt", "paths.run_name=t", "job.output_path=/tmp/out"]
_HUB_REFERENCE = (
    "hf://the-enigma-project/ntt-tokenizers/poisson_references/recon_bin{}ms_rate10ms_window0.1s.json"
    "@64d8bb47951d4d1b06765a74db0e7ae8aec36436"
)
_PUBLISHED_CLI = [
    "job.split=test",
    "job.mode=precision",
    "job.compare_poisson=true",
    "job.poisson_rate_bin_ms=10.0",
    "job.poisson_n_realisations=5",
]


def test_fields_set_in_a_config_file_reach_every_pass(monkeypatch):
    """A job config extending eval_all sets its fields in yaml, not on the command line;
    the passes must still get them (they are composed from command-line overrides)."""
    jobs = _pass_jobs(monkeypatch, ["job=published_eval", *_RUN])

    assert set(jobs) == {"main", "snr", "m1", "m2"}
    assert all(j.split == "test" for j in jobs.values())
    main = jobs["main"]
    assert (main.compare_poisson, main.poisson_rate_bin_ms, main.poisson_n_realisations) == (
        True,
        10.0,
        5,
    )


def test_an_output_path_set_in_a_config_file_reaches_every_pass(monkeypatch):
    """The main pass writes to the config file's output_path; the others to suffixed paths."""
    import ntt.jobs.eval_all as eval_all

    user = ["job=eval_all", "job.ckpt_path=/tmp/fake.ckpt", "paths.run_name=t"]
    job = _instantiate_job(user)
    job._task_overrides = user
    job.output_path = "/tmp/from_config"  # as if set in a yaml extending eval_all
    passes: dict[str, list[str]] = {}

    def fake_run_one(overrides, label, **_kw):
        passes[label] = list(overrides)
        return 0.0

    monkeypatch.setattr(eval_all, "_run_one", fake_run_one)
    job.run()

    assert {label: _instantiate_job(o).output_path for label, o in passes.items()} == {
        "main": "/tmp/from_config",
        "snr": "/tmp/from_config_snr",
        "m1": "/tmp/from_config_m1",
        "m2": "/tmp/from_config_m2",
    }


@pytest.mark.parametrize("path", ["/tmp/a,b", "/tmp/a[1]", "/tmp/a=b", "/tmp/it's", "/tmp/a b"])
def test_an_output_path_with_hydra_special_characters_reaches_every_pass(monkeypatch, path):
    from ntt.jobs.eval_all import _override_value

    run = ["job.ckpt_path=/tmp/fake.ckpt", "paths.run_name=t"]
    jobs = _pass_jobs(
        monkeypatch, ["job=eval_all", *run, f"job.output_path={_override_value(path)}"]
    )

    assert {label: j.output_path for label, j in jobs.items()} == {
        "main": path,
        "snr": f"{path}_snr",
        "m1": f"{path}_m1",
        "m2": f"{path}_m2",
    }


@pytest.mark.parametrize("plus", ["", "++"])
def test_main_only_settings_reach_the_main_pass_only_in_every_override_form(monkeypatch, plus):
    """``++job.compare_poisson=true`` must be stripped from SNR like the plain form; otherwise
    SNR would compare against a Poisson baseline it builds itself."""
    main_only = [
        f"{plus}job.compare_poisson=true",
        f"{plus}job.poisson_reference=/tmp/reference.json",
        f"{plus}job.poisson_rate_bin_ms=10.0",
        f"{plus}job.poisson_n_realisations=5",
        f"{plus}job.compression_n_neurons=256",
    ]
    jobs = _pass_jobs(monkeypatch, ["job=eval_all", *_RUN, *main_only])

    main = jobs["main"]
    assert (main.compare_poisson, main.poisson_reference, main.compression_n_neurons) == (
        True,
        "/tmp/reference.json",
        256,
    )
    defaults = _instantiate_job(["job=eval", *_RUN])
    for field in ("compare_poisson", "poisson_reference", "poisson_rate_bin_ms"):
        assert getattr(jobs["snr"], field) == getattr(defaults, field), field
    for label in ("m1", "m2"):  # standalone robustness jobs: their own optional reference
        assert getattr(jobs[label], "poisson_reference", None) is None, label


@pytest.mark.parametrize(
    ("config", "cli"),
    [
        ("published_eval", [*_PUBLISHED_CLI, f"job.poisson_reference={_HUB_REFERENCE.format(1)}"]),
        (
            "published_eval_ldns",
            [
                *_PUBLISHED_CLI,
                f"job.poisson_reference={_HUB_REFERENCE.format(10)}",
                "job.eval_window_seconds=native",
                "job.input_robustness.evaluate_all_neurons=false",
                "job.latent_robustness.evaluate_all_neurons=false",
            ],
        ),
    ],
)
def test_config_file_and_command_line_give_the_same_passes(monkeypatch, config, cli):
    from_config = _pass_jobs(monkeypatch, [f"job={config}", *_RUN])
    from_cli = _pass_jobs(monkeypatch, ["job=eval_all", *_RUN, *cli])

    assert set(from_config) == set(from_cli)
    for label in from_cli:
        assert from_config[label].model_dump() == from_cli[label].model_dump(), label


def test_command_line_only_runs_forward_nothing_extra():
    """Fields given on the command line are already passed through; nothing is added."""
    user = ["job=eval_all", *_RUN, "job.split=test", "job.compare_poisson=true"]
    job = _instantiate_job(user)
    job._task_overrides = user

    assert job._passthrough_overrides() == [o for o in user if o != "job=eval_all"]


def test_fields_that_cannot_be_forwarded_from_a_config_file_raise():
    user = ["job=eval_all", *_RUN]
    job = _instantiate_job(user)
    job._task_overrides = user
    job.tier_selections = {"xsource": "some/selection"}

    with pytest.raises(ValueError, match="job.tier_selections"):
        job._passthrough_overrides()


def test_override_value_round_trips_through_hydra():
    from hydra.core.override_parser.overrides_parser import OverridesParser

    from ntt.jobs.eval_all import _override_value

    parser = OverridesParser.create()
    for value in ("hf://o/r/f.json@abc", "a,b=c", "it's", r"C:\x\y", "trailing\\", "null", ""):
        parsed = parser.parse_override(f"job.x={_override_value(value)}").value()
        assert parsed == value, value


@pytest.mark.parametrize("prefix", ["", "++"])  # "+" cannot add a key that exists
def test_a_field_on_the_command_line_is_not_forwarded_twice(monkeypatch, prefix):
    """A value set both in the config file and on the command line reaches each pass once."""
    user = ["job=published_eval", *_RUN, f"{prefix}job.split=val"]
    job = _instantiate_job(user)
    job._task_overrides = user
    forwarded = [o for o in job._passthrough_overrides() if "job.split=" in o]
    assert forwarded == [f"{prefix}job.split=val"]
