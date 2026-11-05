# Import base classes and registry
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from ntt.core.model.tokenizer.ar_regularizer import ARRegularizer
from ntt.core.model.tokenizer.base import (
    BaseTokenizer,
    CapabilityValidationError,
    TokenizerSpec,
    create_tokenizer,
    get_tokenizer_class,
    list_tokenizers,
    register_tokenizer,
)

__all__ = [
    # Base classes
    "TokenizerSpec",
    "BaseTokenizer",
    # Exceptions
    "CapabilityValidationError",
    # Registry functions
    "register_tokenizer",
    "create_tokenizer",
    "get_tokenizer_class",
    "list_tokenizers",
    # Checkpoint loading
    "load_tokenizer_from_checkpoint_no_lightning",
    # AR Regularizer
    "ARRegularizer",
]


def _extract_tokenizer_state_dict(
    checkpoint: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Extract tokenizer weights from a Lightning checkpoint or raw state dict."""
    state_dict = checkpoint.get("state_dict", checkpoint)
    tokenizer_state = {
        key.removeprefix("tokenizer."): value
        for key, value in state_dict.items()
        if key.startswith("tokenizer.")
    }
    return tokenizer_state or state_dict


def load_tokenizer_from_checkpoint_no_lightning(
    ckpt_path: str | Path,
    spec: TokenizerSpec | None = None,
    device: str = "cpu",
    allow_spec_update: bool = False,
) -> BaseTokenizer:
    """Load a tokenizer from a Lightning checkpoint.

    Args:
        ckpt_path: Path to the ``.ckpt`` file saved by Lightning.
        spec: TokenizerSpec to use.  If ``None``, the spec is extracted
            from the checkpoint's ``hyper_parameters.tokenizer_spec``.
        device: Device to load weights onto (default ``"cpu"``).
        allow_spec_update: If ``True``, allow the spec to be updated with new fields.
        (Can be useful if the spec was saved with a different version of NTT.)

    Returns:
        Tokenizer with loaded weights, in eval mode.

    Raises:
        ValueError: If *spec* is ``None`` and the checkpoint does not
            contain a ``tokenizer_spec`` in its hyper-parameters.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    if spec is None:
        hparams = ckpt.get("hyper_parameters", {})
        spec_dict = hparams.get("tokenizer_spec")
        if spec_dict is None:
            raise ValueError(
                "No tokenizer_spec found in checkpoint hyper_parameters. "
                "Provide the spec explicitly."
            )
        # Ensure contrib tokenizers are registered so KIND lookup works
        import ntt.contrib.tokenizer  # noqa: F401  # isort: skip
        from ntt.contrib import TokenizerSpecKind  # isort: skip
        from pydantic import TypeAdapter  # isort: skip

        # convert to dict first so re-validation applies current defaults
        # for any fields the checkpoint lacks.
        if hasattr(spec_dict, "model_dump") and allow_spec_update:
            spec_dict = spec_dict.model_dump(by_alias=True, exclude_none=True)

        adapter: TypeAdapter[TokenizerSpecKind] = TypeAdapter(TokenizerSpecKind)
        validated_spec = adapter.validate_python(spec_dict)
        if validated_spec is None:
            raise ValueError("tokenizer_spec in checkpoint hyper_parameters is null.")
        spec = validated_spec

    tokenizer = create_tokenizer(spec)

    # The state dict is loaded manually rather than through Lightning's loader.
    tokenizer_state = _extract_tokenizer_state_dict(ckpt)
    tokenizer.load_state_dict(tokenizer_state)
    tokenizer.eval()
    return tokenizer
