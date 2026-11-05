"""Helpers for persisting evaluation config provenance."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf

from ntt.evaluation.utils.serialization import replace_non_finite_floats

log = logging.getLogger(__name__)


def _omegaconf_to_container(value: Any) -> Any:
    """``model_dump`` fallback for OmegaConf containers held inside a pydantic model.

    Specs pickled into checkpoints before ``TokenizerSpec`` converted Hydra's nested
    configs can still hold a ``ListConfig`` / ``DictConfig``. Any other unknown type
    still fails the dump.
    """
    if isinstance(value, DictConfig | ListConfig):
        return OmegaConf.to_container(value, resolve=True)
    raise TypeError(f"Unable to serialize unknown type: {type(value)}")


def config_to_jsonable(obj: Any, *, name: str) -> Any:
    """Convert a config object to JSON-compatible containers.

    Warns and returns ``None`` when ``obj`` is ``None`` rather than raising, so a
    missing config never aborts an eval at the (post-compute) provenance step.
    """
    from types import SimpleNamespace

    if obj is None:
        log.warning("%s is unavailable for eval run metadata; recording null.", name)
        return None
    if hasattr(obj, "model_dump"):
        value = obj.model_dump(mode="json", by_alias=True, fallback=_omegaconf_to_container)
    elif isinstance(obj, DictConfig | ListConfig):
        try:
            value = OmegaConf.to_container(obj, resolve=True)
        except Exception as exc:
            log.warning("%s could not be resolved for run metadata: %s", name, exc)
            return None
    elif isinstance(obj, SimpleNamespace):
        # SimpleNamespace turns up in unit-test fixtures that stub the
        # eval / data configs. ``vars()`` is the canonical SimpleNamespace
        # dict view.
        value = vars(obj)
    else:
        value = obj
    return replace_non_finite_floats(value)


def hydra_task_overrides() -> list[str]:
    """Return current Hydra task overrides, warning when Hydra is unavailable."""
    try:
        from hydra.core.hydra_config import HydraConfig

        return list(HydraConfig.get().overrides.task)
    except Exception as exc:
        log.warning("Hydra task overrides unavailable for eval run metadata: %s", exc)
        return []


def composed_config_yaml(cfg: Any | None) -> str | None:
    """Return the resolved composed Hydra config as YAML.

    Warns and returns ``None`` when resolution fails (e.g. a lazy or missing
    interpolation). Provenance capture runs before the job context, so an
    unhandled raise here would abort an otherwise-valid eval before compute.
    """
    if cfg is None:
        return None
    try:
        return OmegaConf.to_yaml(cfg, resolve=True)
    except Exception as exc:
        log.warning("Composed Hydra config could not be resolved for provenance: %s", exc)
        return None


def config_subtree_to_jsonable(cfg: Any | None, dotted_paths: list[str]) -> Any | None:
    """Return the first available resolved config subtree for provenance."""
    if cfg is None:
        return None
    for dotted_path in dotted_paths:
        current = cfg
        found = True
        for part in dotted_path.split("."):
            try:
                current = current[part]
            except (KeyError, TypeError):
                current = getattr(current, part, None)
            if current is None:
                found = False
                break
        if found:
            return config_to_jsonable(current, name=dotted_path)
    return None


def _scalar_window_seconds(data_config: Any) -> float | None:
    """Return scalar ``window_length_seconds`` from a data config payload."""
    if not isinstance(data_config, dict):
        return None
    value = data_config.get("window_length_seconds")
    if isinstance(value, int | float):
        return float(value)
    return None


def write_config_provenance_sidecar(
    *,
    run_meta_path: Path,
    eval_config: Any,
    data_config: Any,
    hydra_overrides: list[str],
    composed_config_yaml_text: str | None,
) -> dict[str, Any]:
    """Write ``config_snapshot.yaml`` and return fields for ``run_meta.json``.

    Warns and skips the sidecar (recording a null snapshot path) when the
    composed config YAML is unavailable, rather than raising — the provenance
    step runs after compute and must not abort an otherwise-complete eval.
    """
    snapshot_name: str | None
    if composed_config_yaml_text is None:
        log.warning(
            "composed_config_yaml_text is unavailable for eval run metadata; "
            "skipping config_snapshot.yaml sidecar."
        )
        snapshot_name = None
    else:
        snapshot_path = run_meta_path.parent / "config_snapshot.yaml"
        snapshot_path.write_text(composed_config_yaml_text)
        snapshot_name = snapshot_path.name
    eval_payload = config_to_jsonable(eval_config, name="eval_config")
    data_payload = config_to_jsonable(data_config, name="data_config")
    if isinstance(eval_payload, dict) and "eval_window_seconds" not in eval_payload:
        eval_window_seconds = _scalar_window_seconds(data_payload)
        if eval_window_seconds is None:
            log.warning(
                "Could not derive eval_window_seconds for eval run metadata from data_config."
            )
        else:
            eval_payload["eval_window_seconds"] = eval_window_seconds
    return {
        "eval_config": eval_payload,
        "data_config": data_payload,
        "hydra_overrides": list(hydra_overrides),
        "composed_config_snapshot_path": snapshot_name,
    }
