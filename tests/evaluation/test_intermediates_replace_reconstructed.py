"""Test :meth:`IntermediatesCache.replace_reconstructed`.

Covers core logic: original-side cache entries and passthrough metadata
survive the swap; reconstructed-side entries are dropped; the new cache
has the new ``reconstructed`` EvalData; the old cache is untouched.
"""

import numpy as np

from ntt.evaluation.core.intermediates import IntermediatesCache
from ntt.evaluation.core.types import DataType, EvalData


def _tiny_eval_data(seed: int) -> EvalData:
    rng = np.random.default_rng(seed)
    return EvalData(
        binned_spikes=rng.integers(0, 2, size=(3, 10)).astype(np.float32),
        bin_size_seconds=0.001,
        data_type=DataType.SPIKES,
    )


def test_replace_reconstructed_keeps_original_drops_reconstructed():
    clean = _tiny_eval_data(0)
    recon = _tiny_eval_data(1)
    new_recon = _tiny_eval_data(2)

    cache = IntermediatesCache(
        original=clean,
        reconstructed=recon,
        latents=np.arange(4),  # passthrough metadata
    )
    # Prime one original-side key and one reconstructed-side key manually.
    cache._cache["binned_spikes_original"] = np.ones((3, 10))
    cache._cache["binned_spikes_reconstructed"] = np.zeros((3, 10))

    new_cache = cache.replace_reconstructed(new_recon)

    # Original survives, reconstructed is dropped, passthrough survives.
    assert "binned_spikes_original" in new_cache._cache
    assert "binned_spikes_reconstructed" not in new_cache._cache
    assert "latents" in new_cache._cache

    # New cache points at the new reconstructed data.
    assert new_cache.reconstructed is new_recon
    assert new_cache.original is clean

    # Old cache untouched.
    assert cache.reconstructed is recon
    assert "binned_spikes_reconstructed" in cache._cache
