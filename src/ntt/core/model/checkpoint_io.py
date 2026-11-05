"""Materialize Lightning-compatible checkpoints for MLflow upload.

Two concerns live here:

1. Synthesizing a ``.ckpt`` file from an in-memory tokenizer when no
   training checkpoint exists (e.g. untrained spec-only tokenizers like
   ``contrib/compressed_sensing``).
2. Dispatching, for a given eval job state, to the right local ``.ckpt``
   path that ``mlflow.log_artifact`` can upload — whether that path is a
   pre-existing local file, a downloaded MLflow artifact, or a freshly
   synthesized checkpoint, or an ``hf://`` Hub checkpoint (never uploaded).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import lightning
import torch

from ntt.utils import RankedLogger

if TYPE_CHECKING:
    from ntt.core.model.loss import LossSpec
    from ntt.core.model.tokenizer import BaseTokenizer, TokenizerSpec

log = RankedLogger(__name__, rank_zero_only=True)


def save_live_tokenizer_as_checkpoint(
    tokenizer: BaseTokenizer,
    spec: TokenizerSpec,
    path: str | Path,
    loss_spec: LossSpec | None = None,
) -> Path:
    """Write a Lightning-compatible ``.ckpt`` from a live tokenizer.

    The resulting file round-trips through ``load_tokenizer_from_checkpoint``
    with no special-case handling: its ``hyper_parameters`` mirror
    ``NTTLitModule.__init__`` and its ``state_dict`` keys are prefixed with
    ``tokenizer.`` so Lightning's loader finds the weights.

    Note: the saved ckpt embeds pydantic-model objects in
    ``hyper_parameters`` (via pickle), so ``torch.load(weights_only=True)``
    will refuse it. The standard loaders default to ``weights_only=False``.

    Args:
        tokenizer: In-memory tokenizer whose weights to save.
        spec: Spec that was used to build ``tokenizer``.
        path: Destination ``.ckpt`` path.
        loss_spec: Optional loss spec to embed in ``hyper_parameters``.
            Defaults to ``EvalOnlyMSELossSpec`` — a dummy that returns
            zero during training. Appropriate for eval-only artifacts
            where training state is irrelevant.

    Returns:
        The path written.
    """
    # Late import to avoid pulling in the contrib loss registry just to
    # save a ckpt (callers who supply their own ``loss_spec`` never hit this).
    from ntt.contrib.loss.eval_only_mse import EvalOnlyMSELossSpec

    if loss_spec is None:
        loss_spec = EvalOnlyMSELossSpec()

    path = Path(path)

    tokenizer_state = tokenizer.state_dict()
    prefixed_state_dict = {f"tokenizer.{k}": v for k, v in tokenizer_state.items()}

    checkpoint = {
        "hyper_parameters": {
            "tokenizer_spec": spec,
            "loss_spec": loss_spec,
            "optimizer_gen": None,
            "scheduler_config": None,
            "compile_config": None,
            "viz_every_n_steps": None,
            "disable_val_viz": False,
        },
        "state_dict": prefixed_state_dict,
        "pytorch-lightning_version": lightning.__version__,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    return path


def resolve_checkpoint_for_eval(
    ckpt_path: str | Path | None,
    live_tokenizer: BaseTokenizer,
    live_spec: TokenizerSpec,
    always_copy: bool,
    tmpdir: Path,
) -> Path | None:
    """Return a local ``.ckpt`` path ready to upload to MLflow, or None.

    Dispatches on the four eval source cases described in the
    ``checkpoint_io`` module docstring:

    - ``ckpt_path`` is an MLflow URI (``mlflow-artifacts:/...`` or
      ``mlflow:/...``):
        - ``always_copy=False`` → ``None`` (tag-only; caller skips upload).
        - ``always_copy=True``  → download via
          ``mlflow.artifacts.download_artifacts`` and return that local path.
    - ``ckpt_path`` is an ``hf://`` Hub URI → ``None``: no checkpoint artifact is
      uploaded (the Hub URI is the provenance).
    - ``ckpt_path`` is a local filesystem path → return as-is (MLflow
      reads bytes directly; no copy).
    - ``ckpt_path`` is ``None`` → synthesize a fresh ckpt in ``tmpdir``
      from the live tokenizer via ``save_live_tokenizer_as_checkpoint``.

    The returned path's ``.name`` becomes the artifact filename under
    ``checkpoints/`` in the MLflow run.
    """
    if ckpt_path is not None:
        path_str = str(ckpt_path)
        if path_str.startswith("hf://"):
            # Hub checkpoints are materialised locally by the job; nothing to upload.
            log.info(f"Hub checkpoint {path_str}; skipping MLflow checkpoint upload")
            return None
        is_mlflow_uri = path_str.startswith("mlflow-artifacts:") or path_str.startswith("mlflow:")

        if is_mlflow_uri:
            if not always_copy:
                return None

            import mlflow.artifacts

            log.info(f"Downloading MLflow artifact for re-upload: {path_str}")
            downloaded = Path(mlflow.artifacts.download_artifacts(path_str, dst_path=str(tmpdir)))
            if downloaded.is_dir():
                ckpt_files = list(downloaded.glob("*.ckpt"))
                if len(ckpt_files) == 1:
                    return ckpt_files[0]
                if len(ckpt_files) == 0:
                    raise FileNotFoundError(
                        f"No .ckpt file found in downloaded artifact: {downloaded}"
                    )
                raise ValueError(
                    f"Multiple .ckpt files in downloaded artifact: {[f.name for f in ckpt_files]}"
                )
            return downloaded

        local = Path(ckpt_path)
        if not local.exists():
            log.warning(
                f"Local ckpt_path does not exist; skipping MLflow checkpoint upload: {local}"
            )
            return None
        return local

    # Spec-only case: synthesize from the live tokenizer.
    synth_path = tmpdir / "model.ckpt"
    log.info(f"Synthesizing checkpoint for spec-only tokenizer → {synth_path}")
    return save_live_tokenizer_as_checkpoint(live_tokenizer, live_spec, synth_path)
