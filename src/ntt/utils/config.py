import json
import logging
from typing import Literal

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.rank_zero import (
    rank_zero_only,  # type: ignore
)
from rich.console import Console
from rich.syntax import Syntax


@rank_zero_only
def print_config_tree(
    cfg: DictConfig,
    resolve: bool = True,
    format: Literal["yaml", "json"] = "yaml",
    intro_str: str = "",
) -> None:
    """Prints the contents of a DictConfig as a tree structure using the Rich library.

    :param cfg: A DictConfig composed by Hydra.
    :param resolve: Whether to resolve reference fields of DictConfig. Default is ``True``.
    :param format: Output format, either "yaml" or "json". Default is "yaml".
    """
    if format == "json":
        # Convert to Python object first, then to JSON for pretty printing
        cfg_dict = OmegaConf.to_container(cfg, resolve=resolve)
        cfg_str = json.dumps(cfg_dict, indent=4)
    elif format == "yaml":
        cfg_str = OmegaConf.to_yaml(cfg, resolve=resolve)
    else:
        raise ValueError(f"Unsupported format: {format}. Choose from ['yaml', 'json'].")
    if intro_str:
        print(f"{intro_str}\n\n")
    Console().print(Syntax(cfg_str, format))


@rank_zero_only
def log_config_tree_debug(
    cfg: DictConfig,
    logger: logging.Logger | logging.LoggerAdapter,
    resolve: bool = True,
    format: Literal["yaml", "json"] = "yaml",
    intro_str: str = "",
) -> None:
    """Log a DictConfig dump at debug level instead of printing to stdout.

    :param cfg: A DictConfig composed by Hydra.
    :param logger: Logger instance with a ``debug`` method.
    :param resolve: Whether to resolve reference fields of DictConfig. Default is ``True``.
    :param format: Output format, either "yaml" or "json". Default is "yaml".
    :param intro_str: Optional prefix string for the logged dump.
    """
    if format == "json":
        cfg_dict = OmegaConf.to_container(cfg, resolve=resolve)
        cfg_str = json.dumps(cfg_dict, indent=4)
    elif format == "yaml":
        cfg_str = OmegaConf.to_yaml(cfg, resolve=resolve)
    else:
        raise ValueError(f"Unsupported format: {format}. Choose from ['yaml', 'json'].")
    if intro_str:
        logger.debug(f"{intro_str}\n{cfg_str}")
    else:
        logger.debug(cfg_str)
