#!/usr/bin/env python
"""Combined evaluation job for standard eval + SNR + robustness passes.

Canonical user-facing entrypoint:

    uv run ntt job=eval_all ...
"""

from __future__ import annotations

import os
import re
import time
import warnings
from pathlib import Path
from typing import Any, Literal

import pydantic as pdt
import rootutils
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

# Project root: locate via .project-root, set PROJECT_ROOT for paths/default.yaml
# to resolve before any ntt.* import touches it.
ROOT = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
os.environ.setdefault("PROJECT_ROOT", str(ROOT))

from ntt.contrib import TokenizerSpecKind  # noqa: E402
from ntt.evaluation.config.schema import AggregationStrategy  # noqa: E402
from ntt.evaluation.mlflow_logging import MLflowLoggingConfig  # noqa: E402
from ntt.jobs.base import BaseJob, Locals  # noqa: E402
from ntt.utils import RankedLogger, log_config_tree_debug, print_config_tree  # noqa: E402

log = RankedLogger(__name__, rank_zero_only=True)


class EvalAllPassConfig(pdt.BaseModel):
    """Configuration for one eval_all pass."""

    enabled: bool = True
    evaluation: str
    data: str | None = None
    # None = the eval_all job's evaluate_all_neurons.
    evaluate_all_neurons: bool | None = None


class EvalAllMainPassConfig(EvalAllPassConfig):
    evaluation: str = "hill_climb"
    data: str = "ntt_eval_light"


class EvalAllSNRPassConfig(EvalAllPassConfig):
    evaluation: str = "pipelines/cowley_per_stimulus"
    data: str | None = None


class EvalAllInputRobustnessPassConfig(EvalAllPassConfig):
    evaluation: str = "metrics/robustness_input"
    data: str = "robustness_fast_eval"


class EvalAllLatentRobustnessPassConfig(EvalAllPassConfig):
    evaluation: str = "metrics/robustness_latent"
    data: str = "robustness_fast_eval"


# Per-mode settings. Only values that GENUINELY differ by mode belong here.
# input_robustness (M1) / latent_robustness (M2) eval resolution (``max_eval_seconds``,
# ``n_eval_windows``, ``n_samples_per_epsilon``) is intentionally NOT here: it comes from the
# metric YAMLs (config/evaluation/metrics/robustness_{input,latent}.yaml),
# the single source of truth. It stays in the metric YAMLs: injecting it
# per-mode here would silently shadow the YAML values.
#
# The same holds for the per-mode corruption knobs (input_corruption_max_eval_seconds,
# latent_corruption_n_eval_windows, latent_corruption_n_samples_per_epsilon): they stay in
# the metric YAMLs and are NOT mirrored here (regression guard in test_eval_all.py).
_MODE_OVERRIDES = {
    "fast": {
        "snr_data": "cowley_snr_fast_eval",
    },
    "precision": {
        "snr_data": "cowley_v4_compact_val",
    },
}

_DEFAULT_MODE = "fast"

_EVAL_GROUP_PREFIXES = (
    "evaluation@job.eval_config=",
    "evaluation=",
    "evaluation@_global_=",
)
_DATA_GROUP_PREFIXES = (
    "data@job.data_config=",
    "data=",
    "data@_global_=",
)
_JOB_GROUP_PREFIXES = ("job=", "job@_global_=")
# Set by eval_all for each pass, so a user override of these keys is removed from every
# pass in all three Hydra forms (key=, +key=, ++key=). upload_checkpoint still follows the
# user: the main pass gets eval_all's resolved ``mlflow_logging.upload_checkpoint``, which
# includes any command-line override; the SNR / M1 / M2 passes always get false.
_PASS_MANAGED_KEYS = (
    "job.mlflow_logging.run_name_suffix",
    "job.mlflow_logging.metric_prefix",
    "job.mlflow_logging.upload_checkpoint",
)
_PASS_MANAGED_OVERRIDE_PREFIXES = tuple(
    f"{plus}{key}=" for key in _PASS_MANAGED_KEYS for plus in ("", "+", "++")
)
# Passes are composed from command-line overrides, so a field set in a config file that
# extends eval_all (e.g. config/job/published_eval/) would not reach them. Each of these is
# passed on as an override when it differs from its default and is not on the command line.
_FORWARDED_FIELDS = (
    "ckpt_path",
    "output_path",
    "ckpt_weights_only",
    "seed",
    "device",
    "split",
    "aggregation_strategy",
    "compare_poisson",
    "poisson_reference",
    "poisson_rate_bin_ms",
    "poisson_seed",
    "poisson_n_realisations",
    "poisson_use_all_neurons",
    "compression_n_neurons",
    "generate_plots",
    "plot_seconds",
    "plot_sessions",
)
_FORWARDED_MLFLOW_FIELDS = ("enabled", "experiment_name", "tracking_uri", "run_name")
# These cannot be written as a single override; set them on the command line.
_COMMAND_LINE_ONLY_FIELDS = ("tokenizer_config", "tier_selections")
# The Poisson comparison and compression score belong to the reconstruction (main) pass.
# poisson_reference is the reconstruction reference (a poisson_baseline.json): M1/M2 scores
# use raw values; their own optional Poisson references go to the standalone robustness jobs.
_MAIN_ONLY_KEYS = (
    "job.compare_poisson",
    "job.poisson_reference",
    "job.poisson_rate_bin_ms",
    "job.poisson_seed",
    "job.poisson_n_realisations",
    "job.poisson_use_all_neurons",
    "job.compression_n_neurons",
)
_MAIN_ONLY_PREFIXES = tuple(f"{plus}{key}=" for key in _MAIN_ONLY_KEYS for plus in ("", "+", "++"))
_ROBUSTNESS_INCOMPATIBLE_PREFIXES = _MAIN_ONLY_PREFIXES + tuple(
    f"{plus}{key}="
    for key in ("job.generate_plots", "job.plot_seconds", "job.plot_sessions")
    for plus in ("", "+", "++")
)
_EVALUATE_ALL_NEURONS_PREFIXES = tuple(
    f"{plus}job.evaluate_all_neurons=" for plus in ("", "+", "++")
)
_EVAL_ALL_SPECIFIC_PREFIXES = (
    *(f"{plus}job.eval_window_seconds=" for plus in ("", "+", "++")),  # set per pass instead
    "job.mode=",
    "job.main.",
    "job.snr.",
    "job.input_robustness.",
    "job.latent_robustness.",
)
_SNR_FORBIDDEN_CAPS = ("max_windows", "max_bytes", "max_eval_seconds")
_HILL_CLIMB_MAIN_DEFAULTS = {
    "job.eval_config.max_eval_seconds": "600",
    "job.eval_config.max_neurons": "200",
}


def _read_override(overrides: list[str], key: str) -> str | None:
    found: str | None = None
    for override in overrides:
        if override.startswith(f"{key}="):
            found = override.split("=", 1)[1]
    return found


def _strip_prefixed_overrides(overrides: list[str], prefixes: tuple[str, ...]) -> list[str]:
    return [override for override in overrides if not override.startswith(prefixes)]


def _has_override(overrides: list[str], key: str) -> bool:
    return any(override.startswith((f"{key}=", f"+{key}=", f"++{key}=")) for override in overrides)


def _override_value(value: Any) -> str:
    """Write a field value as the right-hand side of a Hydra override."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return repr(value)
    value = str(getattr(value, "value", value))  # enums by their value
    # Inside Hydra quotes, backslashes are escapes only before a quote or at the end.
    value = re.sub(r"\\+(?='|$)", lambda m: m.group(0) * 2, value)
    return "'" + value.replace("'", "\\'") + "'"


def _compose(overrides: list[str], label: str) -> Any:
    """Compose Hydra config for one pass or wrapper entrypoint."""
    log.debug(f"[{label}] Composing Hydra config with {len(overrides)} overrides.")
    config_dir = str(ROOT / "config")
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(
            config_name="main",
            overrides=overrides,
            return_hydra_config=True,
        )
        HydraConfig.instance().set_config(cfg)
    OmegaConf.set_readonly(cfg, False)
    OmegaConf.set_struct(cfg, False)
    with open_dict(cfg):
        if "hydra" in cfg:
            del cfg["hydra"]
    return cfg


def _assert_no_session_capping(cfg: Any) -> None:
    """Reject session-capping params on the SNR pass."""
    eval_cfg = cfg.job.eval_config if "job" in cfg and cfg.job is not None else None
    if eval_cfg is None:
        return
    offending = {}
    for cap in _SNR_FORBIDDEN_CAPS:
        val = eval_cfg.get(cap) if hasattr(eval_cfg, "get") else getattr(eval_cfg, cap, None)
        if val is not None:
            offending[cap] = val
    if offending:
        raise ValueError(
            f"SNR eval pass rejects session-capping params {sorted(offending)} "
            "(they split a session into segments, which breaks RepeatsSNR's "
            f"per-stim repeat aggregation). Got: {offending}."
        )


def _run_one(overrides: list[str], label: str, *, validate_no_caps: bool = False) -> float:
    """Instantiate and run one sub-job."""
    from ntt import settings as ntt_settings

    cfg = _compose(overrides, label=label)
    if validate_no_caps:
        _assert_no_session_capping(cfg)
    setting_obj: ntt_settings.MainSettings = instantiate(cfg)

    if setting_obj.ignore_warnings:
        log.warning(f"[{label}] Disabling python warnings (ignore_warnings=true).")
        warnings.filterwarnings("ignore")
    if setting_obj.print_cfg:
        log_config_tree_debug(cfg, logger=log, intro_str=f"[{label}] Hydra config")
    if setting_obj.job is None:
        log.warning(f"[{label}] No job configured - skipping.")
        return 0.0

    setting_obj.job.process_config(cfg)
    t0 = time.time()
    with setting_obj.job as runner:
        runner.run()
    return time.time() - t0


def run_eval_all_from_cli_overrides(overrides: list[str]) -> int:
    """Compose ``job=eval_all`` from overrides and run it."""
    from ntt import settings as ntt_settings

    cfg = _compose(overrides, label="eval_all")
    setting: ntt_settings.MainSettings = instantiate(cfg)

    if setting.ignore_warnings:
        log.warning("Disabling python warnings!")
        warnings.filterwarnings("ignore")
    if setting.print_schema:
        schema = ntt_settings.MainSettings.model_json_schema()
        omegaconf_schema = DictConfig(schema)
        print_config_tree(
            omegaconf_schema,
            format="json",
            intro_str="MainSettings JSON Schema:",
        )
    if setting.print_cfg:
        log_config_tree_debug(cfg, logger=log, intro_str="Hydra config")

    if setting.job is None:
        log.info("No job configured. Exiting.")
        return 0

    setting.job.process_config(cfg)
    with setting.job as runner:
        runner.run()
    return 0


class NTTEvalAllJob(BaseJob, strict=False, frozen=False, extra="forbid"):
    """Hydra job that orchestrates main eval + SNR + input/latent robustness passes."""

    KIND: Literal["eval_all"] = "eval_all"

    ckpt_path: str | Path | None = None
    ckpt_weights_only: bool = False
    tokenizer_config: TokenizerSpecKind = None

    seed: int = 42
    device: str = "auto"
    output_path: str | Path | None = None
    split: Literal["val", "test"] = "val"
    aggregation_strategy: AggregationStrategy = AggregationStrategy.MEAN
    tier_selections: dict[str, str] | None = None

    compare_poisson: bool = False
    poisson_reference: str | Path | None = None
    poisson_rate_bin_ms: float | None = None
    poisson_seed: int = 42
    poisson_n_realisations: int = 5
    poisson_use_all_neurons: bool = False

    # Forwarded to every pass (each pass config can override it).
    evaluate_all_neurons: bool = True

    # Eval window of every pass, in seconds, or "native" = the tokenizer's own window.
    # None = the evaluation yamls' (1.0 s) default.
    eval_window_seconds: float | Literal["native"] | None = None

    # Forwarded to the main pass as NTTEvalJob.compression_n_neurons (via the passthrough
    # overrides); stripped from the SNR and robustness passes.
    compression_n_neurons: int | None = pdt.Field(default=None, ge=1)

    mlflow_logging: MLflowLoggingConfig = pdt.Field(default_factory=MLflowLoggingConfig)

    generate_plots: bool = False
    plot_seconds: float = 600.0
    plot_sessions: int = 10

    mode: Literal["fast", "precision"] = _DEFAULT_MODE
    main: EvalAllMainPassConfig = pdt.Field(default_factory=EvalAllMainPassConfig)
    snr: EvalAllSNRPassConfig = pdt.Field(default_factory=EvalAllSNRPassConfig)
    input_robustness: EvalAllInputRobustnessPassConfig = pdt.Field(
        default_factory=EvalAllInputRobustnessPassConfig
    )
    latent_robustness: EvalAllLatentRobustnessPassConfig = pdt.Field(
        default_factory=EvalAllLatentRobustnessPassConfig
    )

    _task_overrides: list[str] = pdt.PrivateAttr(default_factory=list)
    _primary_run_name: str | None = pdt.PrivateAttr(default=None)
    _primary_output_path: str | None = pdt.PrivateAttr(default=None)

    def process_config(self, cfg: DictConfig | None = None) -> None:
        """Capture the current Hydra task overrides for re-composition."""
        try:
            self._task_overrides = list(HydraConfig.get().overrides.task)
        except Exception:
            self._task_overrides = []

        if cfg is not None:
            paths = getattr(cfg, "paths", None)
            if paths is not None:
                self._primary_run_name = getattr(paths, "run_name", None)
            job_cfg = getattr(cfg, "job", None)
            if job_cfg is not None:
                output_path = getattr(job_cfg, "output_path", None)
                if output_path is not None:
                    self._primary_output_path = str(output_path)

    def _passthrough_overrides(self) -> list[str]:
        passthrough = []
        for override in self._task_overrides:
            if override in {"job=eval_all", "job@_global_=eval_all"}:
                continue
            if override.startswith(_EVAL_ALL_SPECIFIC_PREFIXES):
                continue
            passthrough.append(override)
        return passthrough + self._config_file_overrides()

    def _config_file_overrides(self) -> list[str]:
        """Overrides for the fields set in a config file rather than on the command line."""
        cli = self._task_overrides
        default_mlflow = MLflowLoggingConfig()
        changed = [
            (f"job.{name}", getattr(self, name))
            for name in _FORWARDED_FIELDS
            if getattr(self, name) != type(self).model_fields[name].default
        ] + [
            (f"job.mlflow_logging.{name}", getattr(self.mlflow_logging, name))
            for name in _FORWARDED_MLFLOW_FIELDS
            if getattr(self.mlflow_logging, name) != getattr(default_mlflow, name)
        ]
        unsupported = [
            f"job.{name}"
            for name in _COMMAND_LINE_ONLY_FIELDS
            if getattr(self, name) is not None
            and not any(f"job.{name}" in override for override in cli)
        ]
        if self.mlflow_logging.tags and not any("job.mlflow_logging.tags" in o for o in cli):
            unsupported.append("job.mlflow_logging.tags")
        if unsupported:
            raise ValueError(
                f"{', '.join(unsupported)} set in a config file would not reach the eval "
                "passes; set it on the command line instead."
            )
        return [
            f"{key}={_override_value(value)}"
            for key, value in changed
            if not _has_override(cli, key)
        ]

    def _with_evaluate_all_neurons(
        self, overrides: list[str], pass_config: EvalAllPassConfig
    ) -> list[str]:
        """Set the pass's ``job.evaluate_all_neurons`` explicitly, replacing any passthrough
        value, so a pass never falls back to its job yaml's default."""
        value = pass_config.evaluate_all_neurons
        if value is None:
            value = self.evaluate_all_neurons
        overrides = _strip_prefixed_overrides(overrides, _EVALUATE_ALL_NEURONS_PREFIXES)
        return [*overrides, f"job.evaluate_all_neurons={str(value).lower()}"]

    def _with_eval_window(self, overrides: list[str]) -> list[str]:
        """Set every pass's ``eval_config.eval_window_seconds`` to the job's, if given."""
        if self.eval_window_seconds is None:
            return overrides
        return [*overrides, f"++job.eval_config.eval_window_seconds={self.eval_window_seconds}"]

    def _mode_cfg(self) -> dict[str, Any]:
        if self.mode not in _MODE_OVERRIDES:
            raise ValueError(
                f"Unknown job.mode={self.mode!r}; expected one of {sorted(_MODE_OVERRIDES)}."
            )
        return _MODE_OVERRIDES[self.mode]

    def _build_main_overrides(self, passthrough: list[str]) -> list[str]:
        overrides = _strip_prefixed_overrides(
            passthrough,
            _EVAL_GROUP_PREFIXES + _DATA_GROUP_PREFIXES + _JOB_GROUP_PREFIXES,
        )
        overrides = _strip_prefixed_overrides(overrides, _PASS_MANAGED_OVERRIDE_PREFIXES)
        overrides.append("job=eval")
        overrides.append(f"evaluation@job.eval_config={self.main.evaluation}")
        overrides.append(f"data@job.data_config={self.main.data}")
        if self.main.evaluation == "hill_climb":
            for key, value in _HILL_CLIMB_MAIN_DEFAULTS.items():
                if not _has_override(overrides, key):
                    overrides.append(f"++{key}={value}")
        overrides.append("job.mlflow_logging.run_name_suffix=__main")
        upload = "true" if self.mlflow_logging.upload_checkpoint else "false"
        overrides.append(f"job.mlflow_logging.upload_checkpoint={upload}")
        return overrides

    def _build_secondary_overrides(
        self,
        passthrough: list[str],
        *,
        job: str,
        evaluation: str,
        data: str,
        run_name_suffix: str,
        strip_prefixes: tuple[str, ...] = (),
    ) -> list[str]:
        primary_run_name = _read_override(passthrough, "paths.run_name")
        if primary_run_name is not None:
            # The parsed value: a quoted run name would otherwise keep its quotes.
            parsed = (
                OverridesParser.create()
                .parse_override(f"paths.run_name={primary_run_name}")
                .value()
            )
            primary_run_name = None if parsed is None else str(parsed)
        primary_run_name = primary_run_name or self._primary_run_name
        # The parsed value: a forwarded config-file path reaches the passthrough quoted.
        primary_output_path = (
            (str(self.output_path) if self.output_path is not None else None)
            or _read_override(passthrough, "job.output_path")
            or self._primary_output_path
        )

        overrides = _strip_prefixed_overrides(
            passthrough,
            _EVAL_GROUP_PREFIXES + _DATA_GROUP_PREFIXES + _JOB_GROUP_PREFIXES,
        )
        overrides = _strip_prefixed_overrides(overrides, _PASS_MANAGED_OVERRIDE_PREFIXES)
        if strip_prefixes:
            overrides = _strip_prefixed_overrides(overrides, strip_prefixes)
        overrides = [
            override
            for override in overrides
            if not override.startswith(("paths.run_name=", "job.output_path="))
        ]
        overrides.append(f"job={job}")
        overrides.append(f"evaluation@job.eval_config={evaluation}")
        overrides.append(f"data@job.data_config={data}")
        if primary_run_name is not None:
            run_name = _override_value(f"{primary_run_name}{run_name_suffix}")
            overrides.append(f"paths.run_name={run_name}")
        if primary_output_path is not None:
            output_path = _override_value(f"{primary_output_path}{run_name_suffix}")
            overrides.append(f"job.output_path={output_path}")
        return overrides

    def run(self) -> Locals:
        """Run the requested eval passes sequentially."""
        if (
            not self.main.enabled
            and not self.snr.enabled
            and not self.input_robustness.enabled
            and not self.latent_robustness.enabled
        ):
            raise ValueError("All eval_all passes are disabled; nothing to do.")

        passthrough = self._passthrough_overrides()
        if self.poisson_reference is not None and (
            self.input_robustness.enabled or self.latent_robustness.enabled
        ):
            log.info(
                "job.poisson_reference is the reconstruction reference: the robustness passes "
                "run without it (use job=eval_robustness_input/latent for M1/M2 references)."
            )
        mode_cfg = self._mode_cfg()
        timings: dict[str, float] = {}

        if self.main.enabled:
            log.info("=== Pass 1/4: main eval ===")
            main_overrides = self._with_eval_window(
                self._with_evaluate_all_neurons(self._build_main_overrides(passthrough), self.main)
            )
            timings["main"] = _run_one(main_overrides, label="main")
        else:
            log.info("=== Pass 1/4: main eval - skipped ===")

        snr_data = self.snr.data or str(mode_cfg["snr_data"])
        if self.snr.enabled:
            snr_overrides = self._build_secondary_overrides(
                passthrough,
                job="eval",
                evaluation=self.snr.evaluation,
                data=snr_data,
                run_name_suffix="_snr",
                strip_prefixes=_MAIN_ONLY_PREFIXES,
            )
            snr_overrides = self._with_eval_window(
                self._with_evaluate_all_neurons(snr_overrides, self.snr)
            )
            snr_overrides += [
                "job.mlflow_logging.run_name_suffix=__snr",
                "job.mlflow_logging.metric_prefix=eval_snr",
                "job.mlflow_logging.upload_checkpoint=false",
            ]
            log.info(
                f"=== Pass 2/4: SNR eval (evaluation={self.snr.evaluation}, data={snr_data}) ==="
            )
            timings["snr"] = _run_one(snr_overrides, label="snr", validate_no_caps=True)
        else:
            log.info("=== Pass 2/4: SNR eval - skipped ===")

        if self.input_robustness.enabled:
            input_robustness_overrides = self._build_secondary_overrides(
                passthrough,
                job="eval_robustness_input",
                evaluation=self.input_robustness.evaluation,
                data=str(self.input_robustness.data),
                run_name_suffix="_m1",
                strip_prefixes=_ROBUSTNESS_INCOMPATIBLE_PREFIXES,
            )
            input_robustness_overrides = self._with_eval_window(
                self._with_evaluate_all_neurons(input_robustness_overrides, self.input_robustness)
            )
            # input_robustness (M1) max_eval_seconds comes from the metric YAML
            # (config/evaluation/metrics/robustness_input.yaml) — not overridden
            # per-mode here. See _MODE_OVERRIDES note above.
            input_robustness_overrides += [
                f"++job.generate_plots={str(self.generate_plots).lower()}",
            ]
            # Pass-specific MLflow naming so the input_robustness (M1) run is
            # distinguishable from the main / SNR runs. Without this pass-specific prefix,
            # it inherits the default ``metric_prefix="eval"`` and collides with the main pass.
            input_robustness_overrides += [
                "job.mlflow_logging.run_name_suffix=__robustness_input",
                "job.mlflow_logging.metric_prefix=eval_m1",
                "job.mlflow_logging.upload_checkpoint=false",
            ]
            log.info(
                "=== Pass 3/4: input-corruption robustness (M1) "
                f"(evaluation={self.input_robustness.evaluation}, "
                f"data={self.input_robustness.data}, mode={self.mode}) ==="
            )
            timings["m1"] = _run_one(input_robustness_overrides, label="m1")
        else:
            log.info("=== Pass 3/4: input-corruption robustness (M1) - skipped ===")

        if self.latent_robustness.enabled:
            latent_robustness_overrides = self._build_secondary_overrides(
                passthrough,
                job="eval_robustness_latent",
                evaluation=self.latent_robustness.evaluation,
                data=str(self.latent_robustness.data),
                run_name_suffix="_m2",
                strip_prefixes=_ROBUSTNESS_INCOMPATIBLE_PREFIXES,
            )
            latent_robustness_overrides = self._with_eval_window(
                self._with_evaluate_all_neurons(
                    latent_robustness_overrides, self.latent_robustness
                )
            )
            # latent_robustness (M2) n_eval_windows / n_samples_per_epsilon come from the metric
            # YAML (config/evaluation/metrics/robustness_latent.yaml) — not overridden
            # per-mode here. See _MODE_OVERRIDES note above.
            latent_robustness_overrides += [
                f"++job.generate_plots={str(self.generate_plots).lower()}",
            ]
            # Pass-specific MLflow naming, see input_robustness comment above.
            latent_robustness_overrides += [
                "job.mlflow_logging.run_name_suffix=__robustness_latent",
                "job.mlflow_logging.metric_prefix=eval_m2",
                "job.mlflow_logging.upload_checkpoint=false",
            ]
            log.info(
                "=== Pass 4/4: latent-corruption robustness (M2) "
                f"(evaluation={self.latent_robustness.evaluation}, "
                f"data={self.latent_robustness.data}, mode={self.mode}) ==="
            )
            timings["m2"] = _run_one(latent_robustness_overrides, label="m2")
        else:
            log.info("=== Pass 4/4: latent-corruption robustness (M2) - skipped ===")

        print("=== Wallclock summary ===", flush=True)
        for key, value in timings.items():
            print(f"  {key:<6s} : {value:6.1f} s", flush=True)
        if timings:
            print(f"  total  : {sum(timings.values()):6.1f} s", flush=True)
        return locals()
