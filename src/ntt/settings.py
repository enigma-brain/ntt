"""Settings for the NTT application.

This module provides a settings layer between configuration (Hydra) and
runtime execution (Jobs). The Settings classes handle:
- Environment variable overrides
- Secrets management
- Configuration validation
- Schema generation

Example:
    Basic usage with Hydra config:
        cfg = hydra.compose(...)
        settings = MainSettings.model_validate(cfg)
        with settings.job as runner:
            runner.run()

    With environment variable override:
        export NTT_JOB__SEED=42
        settings = MainSettings.model_validate(cfg)
        # settings.job.seed will be 42 (overridden from env)
"""

from pathlib import Path

import pydantic as pdt
import pydantic_settings as pdts

from ntt.jobs import JobKind
from ntt.utils import NTTBaseModel


class PathSettings(NTTBaseModel, strict=False, frozen=True, extra="forbid"):
    """Path configuration settings.

    Maps to config/paths/default.yaml structure.

    Parameters:
        root_dir (Path): Path to root directory (typically from PROJECT_ROOT env var)
        data_dir (Path): Path to data directory
        experiment_name (str): Name of larger experiment (e.g., "ntt")
        run_name (str): Required interpretable name for current run
        log_dir (Path): Path to logging directory
        output_dir (Path): Path to output directory (created dynamically by Hydra)
        work_dir (Path): Path to working directory (Hydra runtime cwd)

    Example:
        paths = PathSettings(
            root_dir="/path/to/project",
            data_dir="/path/to/project/data",
            experiment_name="ntt",
            run_name="my-experiment",
            log_dir="/path/to/project/logs",
            output_dir="/path/to/project/logs/train/my-experiment",
            work_dir="/path/to/project",
        )
    """

    root_dir: Path
    data_dir: Path
    experiment_name: str
    run_name: str
    log_dir: Path
    output_dir: Path
    work_dir: Path


class MainSettings(pdts.BaseSettings):
    """Main settings of the NTT application.

    Parameters:
        job (jobs.BaseJob | None): The job to run (TrainingJob, etc.). Optional.
        paths (PathSettings): Path configuration
        print_cfg (bool): Whether to print configuration tree (optional)
        print_schema (bool): Whether to print settings JSON schema (optional)
        ignore_warnings (bool): Whether to ignore Python warnings (optional)

    Additional fields from Hydra config are allowed and will not be validated.

    Environment Variables:
        NTT_JOB__SEED: Random seed for reproducibility
        NTT_JOB__CKPT_PATH: Path to checkpoint for resuming
        NTT_JOB__DATA_CONFIG__*: Override data configuration
        NTT_JOB__TRAINER__*: Override trainer settings
        NTT_PATHS__ROOT_DIR: Override root directory
        NTT_PATHS__DATA_DIR: Override data directory
        NTT_PRINT_CFG: Whether to print config tree
        NTT_PRINT_SCHEMA: Whether to print settings schema
        NTT_IGNORE_WARNINGS: Whether to ignore warnings

    Example:
        From Hydra config:
            cfg = hydra.compose(config_name="train")
            settings = MainSettings.model_validate(cfg)
            with settings.job as runner:
                runner.run()

        With environment overrides:
            export NTT_JOB__SEED=42
            export NTT_JOB__DATA_CONFIG__BATCH_SIZE=64
            export NTT_PATHS__DATA_DIR=/custom/data/path
            settings = MainSettings.model_validate(cfg)
            # settings.job.seed == 42
            # settings.job.data_config.batch_size == 64
            # settings.paths.data_dir == Path("/custom/data/path")

        Print schema (no job required):
            ntt print_schema=true job=null
    """

    model_config = pdts.SettingsConfigDict(
        env_prefix="NTT_",
        env_nested_delimiter="__",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",  # Allow additional fields from Hydra config
    )

    job: JobKind = pdt.Field(
        default=None,
        description="The job to execute (e.g., TrainingJob). If None, no job runs.",
    )
    paths: PathSettings
    print_cfg: bool = False
    print_schema: bool = False
    ignore_warnings: bool = False
