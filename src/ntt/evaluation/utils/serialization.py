"""Serialize and save evaluation results to JSON."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, ListConfig

from ntt.evaluation.aggregation.comparison import MetricComparison


def json_default(obj: Any) -> Any:
    """JSON serializer for numpy/torch types.

    Used as the ``default`` argument for :func:`json.dump` across the
    evaluation pipeline.  Raises :class:`TypeError` for unknown types
    instead of silently converting to strings.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.generic):  # other numpy scalars (bool_, str_, etc.)
        return obj.item()
    if isinstance(obj, DictConfig):
        from omegaconf import OmegaConf

        return OmegaConf.to_container(obj, resolve=True)
    if isinstance(obj, ListConfig):
        from omegaconf import OmegaConf

        return OmegaConf.to_container(obj, resolve=True)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


_json_default = json_default


def replace_non_finite_floats(obj: Any) -> Any:
    """Recursively replace NaN / +Inf / -Inf floats with ``None``.

    CPython's ``json.dump`` emits these as the bare tokens ``NaN``, ``Infinity``,
    ``-Infinity``, which are not valid per RFC 8259 and break strict parsers
    (``jq``, ``pandas.read_json(strict=True)``, browsers' ``JSON.parse``).
    Apply this to a payload immediately before ``json.dump`` to keep the
    written file conformant. The in-memory dict is left untouched.
    """
    import math

    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if isinstance(obj, dict):
        return {k: replace_non_finite_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [replace_non_finite_floats(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(replace_non_finite_floats(v) for v in obj)
    return obj


def save_tokenizer_details(
    spec: Any,
    output_dir: str | Path,
    name: str,
    *,
    display_name: str | None = None,
    timestamp: str | None = None,
) -> Path:
    """Save tokenizer spec parameters to a standalone JSON file.

    Args:
        spec: A pydantic TokenizerSpec (or subclass) instance.
        output_dir: Directory to write into (created if needed).
        name: Short identifier, e.g. ``"cs_tokenizer"``.
        display_name: The annotated display name used in plots, if any.
        timestamp: Optional timestamp string; generated if not provided.

    Returns:
        Path to the written JSON file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    data: dict[str, Any] = {
        "timestamp": timestamp,
        "name": name,
        "spec_class": type(spec).__name__,
        "parameters": spec.model_dump(),
    }
    if display_name is not None:
        data["display_name"] = display_name

    output_file = output_dir / f"tokenizer_details_{name}_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2, default=json_default)
    return output_file


def save_normalized_comparisons(
    comparisons: dict[str, MetricComparison],
    output_dir: str | Path,
    *,
    model_name: str | None = None,
    baseline_name: str | None = None,
    timestamp: str | None = None,
) -> Path:
    """Save normalized comparison scores to a JSON file.

    This includes the ceiling values needed to reconstruct the radar chart
    when replotting from saved results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    data: dict[str, Any] = {
        "timestamp": timestamp,
        "model_name": model_name,
        "baseline_name": baseline_name,
        "comparisons": {name: comp.to_dict() for name, comp in comparisons.items()},
    }

    output_file = output_dir / f"normalized_comparisons_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2, default=json_default)
    return output_file


def load_normalized_comparisons(
    path: str | Path,
) -> tuple[dict[str, MetricComparison], str | None, str | None]:
    """Load normalized comparisons from a JSON file.

    Returns:
        Tuple of (comparisons_dict, model_name, baseline_name).
    """
    with open(path) as f:
        data = json.load(f)

    comparisons = {
        name: MetricComparison.from_dict(comp_dict)
        for name, comp_dict in data["comparisons"].items()
    }
    return comparisons, data.get("model_name"), data.get("baseline_name")
