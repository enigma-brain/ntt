"""Path resolution helpers — MLflow URIs + per-job partial output dir."""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


def resolve_artifact_path(path_str: str) -> Path:
    """Resolve a path string to a local Path.

    Accepts:
    - ``/abs/path`` or ``relative/path`` → returned as Path unchanged.
    - ``mlflow-artifacts:/...`` or ``mlflow:/...`` → downloaded via
      ``mlflow.artifacts.download_artifacts`` and returned as the local
      Path of the downloaded file or directory.
    - ``hf://<owner>/<repo>/<path>[@<revision>]`` → downloaded from the Hugging
      Face Hub (``ntt.hub.download_hub_path``).
    """
    if path_str.startswith("hf://"):
        from ntt.hub import download_hub_path

        log.info(f"Downloading from the Hugging Face Hub: {path_str}")
        return download_hub_path(path_str)
    if path_str.startswith("mlflow-artifacts:") or path_str.startswith("mlflow:"):
        import mlflow.artifacts

        log.info(f"Downloading MLflow artifact: {path_str}")
        return Path(mlflow.artifacts.download_artifacts(path_str))
    return Path(path_str)


def resolve_partial_output_dir(output_path: str | Path | None) -> Path:
    """Resolve where ``model_eval_partial.json`` (the early durability
    write) should land.

    Priority:
    1. ``output_path`` if explicitly provided — returned as ``Path``.
    2. Hydra's per-run output dir via ``HydraConfig.get().runtime.output_dir``
       — always unique per launch, so concurrent eval jobs (e.g., parallel
       sweep jobs) never collide.
    3. ``Path.cwd()`` with a ``log.warning``: if both above fail (e.g.,
       direct programmatic instantiation outside Hydra) we still write
       something, but warn that concurrent jobs in the same CWD will
       overwrite each other.
    """
    if output_path is not None:
        return Path(str(output_path))
    try:
        from hydra.core.hydra_config import HydraConfig

        return Path(HydraConfig.get().runtime.output_dir)
    except (ImportError, ValueError):
        log.warning(
            "model_eval_partial.json: no output_path configured and HydraConfig "
            "unavailable; falling back to CWD. Concurrent jobs in the same CWD "
            "will overwrite each other's partial results."
        )
        return Path.cwd()
