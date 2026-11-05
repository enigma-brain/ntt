"""NTT model components."""

import logging
from typing import IO, Any

import torch
from lightning.fabric.utilities.types import _MAP_LOCATION_TYPE, _PATH

from ntt.core.model.litmodule import NTTLitModule
from ntt.core.model.tokenizer.base import BaseTokenizer

__all__ = [
    # Lightning module
    "NTTLitModule",
    # Checkpoint loading
    "load_tokenizer_from_checkpoint",
]

log = logging.getLogger(__name__)

# Some checkpoint keys are safe to ignore when we only need the tokenizer for
# inference:
# - ``loss_fn.`` covers loss-module buffers that are never consulted at eval time.
# - ``tokenizer.sigreg.`` covers training-only SIGReg buffers in checkpoints
#   whose tokenizer does not instantiate that helper.
# - ``tokenizer.ar_regularizer.`` covers training-only AR-regularizer buffers
#   (e.g. the CALM tokenizer) that are likewise absent at eval time.
# Anything outside these prefixes indicates a real architecture mismatch and
# must still fail loudly rather than silently loading wrong weights.
_KNOWN_DRIFT_PREFIXES: tuple[str, ...] = (
    "loss_fn.",
    "tokenizer.sigreg.",
    "tokenizer.ar_regularizer.",
)


def load_tokenizer_from_checkpoint(
    checkpoint_path: _PATH | IO,
    map_location: _MAP_LOCATION_TYPE = None,
    weights_only: bool | None = None,
    **kwargs: Any,
) -> BaseTokenizer | None:
    r"""Primary way of loading a tokenizer from a checkpoint, using Lightning and
    `ntt.core.model.NTTLitModule`. When Lightning saves a checkpoint it stores the arguments passed
    to ``NTTLitModule.__init__``  in the checkpoint under ``"hyper_parameters"``.

    Any arguments specified through \*\*kwargs will override stored args.

    Args:
        checkpoint_path: Path to checkpoint. This can also be a URL, or file-like object
        map_location:
            If your checkpoint saved a GPU model and you now load on CPUs
            or a different number of GPUs, use this to map to the new setup.
            The behaviour is the same as in :func:`torch.load`.
        weights_only: If ``True``, restricts loading to
            ``state_dicts`` of plain ``torch.Tensor`` and other
            primitive types. If loading a checkpoint from a trusted
            source that contains an ``nn.Module``, use
            ``weights_only=False``. If loading from an untrusted
            source, we recommend ``weights_only=True``.
        \**kwargs: Any extra keyword args needed to init the
            datamodule. Can also be used to override saved
            hyperparameter values.

    Return:
        :class:`BaseTokenizer` instance with loaded weights and hyperparameters (if available).

    Note:
        ``load_from_checkpoint`` is a **class** method. You must use your :class:`NTTLitModule`
        **class** to call it instead of the :class:`NTTLitModule` instance, or a
        ``TypeError`` will be raised.
    """
    # Use NTTLitModule's built-in loading. ``strict=False`` lets us tolerate
    # loss-module drift (e.g. new buffers registered after the checkpoint was
    # written). The post-load check below enforces that any skipped key is
    # in fact loss-only — a tokenizer-side mismatch would silently load wrong
    # weights, so it must raise.
    lit_module = NTTLitModule.load_from_checkpoint(
        checkpoint_path,
        map_location=map_location,
        weights_only=weights_only,
        strict=False,
        **kwargs,
    )

    _verify_only_known_drift_keys_skipped(
        checkpoint_path,
        lit_module,
        weights_only=weights_only,
    )

    tokenizer = lit_module.tokenizer
    if tokenizer is not None:
        tokenizer.eval()
    return tokenizer  # type: ignore[return-value]


def _verify_only_known_drift_keys_skipped(
    checkpoint_path: _PATH | IO,
    lit_module: NTTLitModule,
    weights_only: bool | None,
) -> None:
    """Compare the checkpoint's state-dict keys to the loaded module's keys.

    Raises if any missing/unexpected key is outside ``_KNOWN_DRIFT_PREFIXES``.
    Warns once if any drift-prefix keys are present (so the load is auditable).
    """
    ckpt = torch.load(
        checkpoint_path,
        map_location="cpu",  # we only inspect keys; placement is irrelevant
        weights_only=False if weights_only is None else weights_only,
    )
    ckpt_keys = set(ckpt["state_dict"].keys())
    model_keys = set(lit_module.state_dict().keys())

    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys

    def _allowed(k: str) -> bool:
        return any(k.startswith(p) for p in _KNOWN_DRIFT_PREFIXES)

    bad_missing = sorted(k for k in missing if not _allowed(k))
    bad_unexpected = sorted(k for k in unexpected if not _allowed(k))

    if bad_missing or bad_unexpected:
        raise RuntimeError(
            f"Checkpoint state-dict mismatch outside known-drift prefixes "
            f"{_KNOWN_DRIFT_PREFIXES}:\n"
            f"  missing in checkpoint: {bad_missing}\n"
            f"  unexpected in checkpoint: {bad_unexpected}\n"
            f"This usually means the tokenizer architecture differs from the "
            f"one the checkpoint was trained with."
        )

    if missing or unexpected:
        log.warning(
            "Loaded tokenizer with non-strict matching on loss module: "
            "skipped %d missing key(s), %d unexpected key(s).",
            len(missing),
            len(unexpected),
        )
