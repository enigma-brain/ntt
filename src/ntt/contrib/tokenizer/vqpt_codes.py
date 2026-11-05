"""Codes per latent position for ``vector_quantize_pytorch`` (VQPT) quantizers.

The wanneural and cosmos VQPT bottlenecks wrap a lucidrains quantizer chosen by
``vqpt_type`` and built from ``vqpt_kwargs``. ``get_vocab_size()`` on those specs gives
the size of ONE code (one codebook, one stage). This module gives how many such codes
the quantizer emits per latent position. It works from the kwargs alone, so it needs
neither the quantizer nor the library.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

# vqpt_type -> the kwargs whose product is the number of codes per position.
# Every other multiplier key must be absent or 1 (see _MULTIPLIER_KEYS).
_CODE_FACTORS: dict[str, tuple[str, ...]] = {
    "vq": ("heads",),
    "fsq": ("num_codebooks",),
    "lfq": ("num_codebooks",),
    "sim_vq": (),
    "latent_quantize": ("num_codebooks",),
    "residual_vq": ("num_quantizers",),
    "residual_fsq": ("num_quantizers",),
    "residual_lfq": ("num_quantizers",),
    "residual_sim_vq": ("num_quantizers",),
    "grouped_residual_vq": ("groups", "num_quantizers"),
    "grouped_residual_fsq": ("groups", "num_quantizers"),
    "grouped_residual_lfq": ("groups", "num_quantizers"),
}
_ALIASES = {"rvq": "residual_vq"}
_REQUIRED = {"num_quantizers"}
_MULTIPLIER_KEYS = ("heads", "num_codebooks", "groups", "num_quantizers")


def vqpt_codes_per_token(vqpt_type: str, vqpt_kwargs: Mapping[str, Any]) -> int:
    """Number of codes a VQPT quantizer emits per latent position.

    Matches the index layout of the lucidrains quantizers (v1.30). For example,
    ``residual_fsq`` with ``num_quantizers=4`` emits ``(B, T*N, 4)`` indices, which is
    4 codes per position. ``grouped_residual_*`` emits ``groups * num_quantizers``.
    ``latent_quantize`` counts ``num_codebooks`` like FSQ/LFQ (lucidrains v1.30.1 cannot
    construct ``num_codebooks > 1`` yet, so that case is spec-level only).

    Raises:
        NotImplementedError: When the count cannot be determined. This covers an
            unknown ``vqpt_type``, a residual type without ``num_quantizers``, or a
            multiplier kwarg this type does not support (e.g. ``num_codebooks`` on a
            residual quantizer). It never falls back to a default count.
    """
    kind = _ALIASES.get(vqpt_type, vqpt_type)
    if kind not in _CODE_FACTORS:
        raise NotImplementedError(f"codes per token unknown for vqpt_type={vqpt_type!r}")
    factors = _CODE_FACTORS[kind]
    for key in _MULTIPLIER_KEYS:
        if key not in factors and vqpt_kwargs.get(key, 1) != 1:
            raise NotImplementedError(
                f"codes per token unknown for vqpt_type={vqpt_type!r} with "
                f"{key}={vqpt_kwargs[key]!r}"
            )
    codes = 1
    for key in factors:
        if key in _REQUIRED and key not in vqpt_kwargs:
            raise NotImplementedError(
                f"codes per token unknown: vqpt_type={vqpt_type!r} needs {key!r} in vqpt_kwargs"
            )
        value = vqpt_kwargs.get(key, 1)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise NotImplementedError(
                f"codes per token unknown: vqpt_kwargs[{key!r}]={value!r} is not a positive int"
            )
        codes *= value
    return codes


def vqpt_vocab_size(vqpt_kwargs: Mapping[str, Any]) -> int:
    """Size of ONE code of a VQPT quantizer (one codebook, one stage).

    ``codebook_size`` when given, else ``prod(levels)``.

    Raises:
        NotImplementedError: ``codebook_size`` is a list/tuple (``ResidualVQ`` with a
            different codebook per stage), so there is no single vocabulary size.
        ValueError: neither ``codebook_size`` nor ``levels`` is given.
    """
    cs = vqpt_kwargs.get("codebook_size")
    levels = vqpt_kwargs.get("levels")
    if isinstance(cs, list | tuple):
        raise NotImplementedError(
            f"per-stage codebook sizes {list(cs)} have no single vocabulary size"
        )
    if cs is not None:
        return int(cs)
    if levels is not None:
        return int(math.prod(levels))
    raise ValueError(
        "vqpt_kwargs must include either `codebook_size` or `levels` to derive a vocabulary size."
    )
