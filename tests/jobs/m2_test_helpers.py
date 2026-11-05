"""Shared helpers for isolated latent_robustness (M2) metric tests."""

from __future__ import annotations

from ntt.evaluation.core.intermediates import IntermediatesCache
from ntt.evaluation.core.types import EvalData


def build_window_intermediates_cache(
    *,
    original: EvalData,
    reconstructed: EvalData,
    **kwargs,
) -> IntermediatesCache:
    """Construct a single-pair intermediates cache for metric tests."""
    return IntermediatesCache(
        original=original,
        reconstructed=reconstructed,
        **kwargs,
    )
