"""Configuration management for evaluation."""

from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from .metric_config import MetricInstanceConfig, instantiate_metrics_from_config
from .schema import AggregationStrategy, EvaluationConfig, LightningEvalConfig

DEFAULT_CONFIG = EvaluationConfig()

__all__ = [
    "AggregationStrategy",
    "EvaluationConfig",
    "LightningEvalConfig",
    "MetricInstanceConfig",
    "DEFAULT_CONFIG",
    "instantiate_metrics_from_config",
    "load_default_config",
    "validate_config",
    "validate_lightning_config",
    "config_to_pipeline_kwargs",
]


def load_default_config() -> DictConfig:
    """Load default configuration from YAML."""
    if OmegaConf is None:
        return cast(DictConfig, {})
    config_path = Path(__file__).parent / "defaults.yaml"
    if config_path.exists():
        return cast(DictConfig, OmegaConf.load(config_path))
    else:
        import os

        raise AssertionError(
            f"Default configuration file {config_path} "
            f"does not exist. cwd: {os.getcwd()}. "
            "Note: the config might be in config/evaluation/default.yaml"
        )
    return OmegaConf.create({})


def validate_config(cfg: DictConfig | dict) -> EvaluationConfig:
    """Validate configuration using Pydantic.

    Recursively materialises any OmegaConf containers into plain dicts /
    lists before handing off to Pydantic, including the case where ``cfg``
    is a plain ``dict`` whose nested values are still ``ListConfig`` /
    ``DictConfig`` (which happens when a Hydra config is passed through a
    Pydantic field typed as ``dict[str, Any]`` — Pydantic coerces the
    top-level container but leaves the nested OmegaConf objects intact).
    Without this, Pydantic sub-models such as
    :class:`RobustnessInputConfig` see ``ListConfig`` instances where they
    expect ``list`` and reject otherwise-valid sweep specs.
    """
    config_dict: Any
    if OmegaConf is not None:
        config_dict = OmegaConf.to_container(OmegaConf.create(cfg), resolve=True)
    else:
        config_dict = cfg

    if isinstance(config_dict, dict) and "evaluation" in config_dict:
        config_dict = config_dict["evaluation"]
    # Strip keys that are not EvaluationConfig fields (e.g. lightning_eval)
    # so the YAML can contain settings consumed by other subsystems without
    # causing a Pydantic validation error.
    if isinstance(config_dict, dict):
        valid_keys = set(EvaluationConfig.model_fields)
        config_dict = {k: v for k, v in config_dict.items() if k in valid_keys}

    return EvaluationConfig(**config_dict)


def validate_lightning_config(cfg: DictConfig | dict) -> LightningEvalConfig:
    """Validate Lightning evaluation configuration."""
    config_dict: Any
    if OmegaConf is not None and isinstance(cfg, DictConfig):
        config_dict = OmegaConf.to_container(cfg, resolve=True)
    else:
        config_dict = cfg

    if isinstance(config_dict, dict) and "lightning_eval" in config_dict:
        config_dict = config_dict["lightning_eval"]
    elif (
        isinstance(config_dict, dict)
        and "evaluation" in config_dict
        and isinstance(config_dict["evaluation"], dict)
        and "lightning_eval" in config_dict["evaluation"]
    ):
        config_dict = config_dict["evaluation"]["lightning_eval"]

    return LightningEvalConfig(**config_dict)


def config_to_pipeline_kwargs(config: EvaluationConfig) -> dict[str, Any]:
    """Convert Pydantic config to kwargs for EvaluationPipeline."""
    return config.model_dump()
