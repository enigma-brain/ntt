"""Auxiliary tests for RepeatsSNR metric (edge cases, shape invariants)."""

import numpy as np
import pytest

from ntt.evaluation.core.intermediates import IntermediatesCache
from ntt.evaluation.core.types import EvalData
from ntt.evaluation.metrics.repeats import RepeatsSNR


def _make_cache(latents, ids, tokens_per_window=1):
    image_id_token_res = np.repeat(ids, tokens_per_window)
    window_group_id_token_res = np.repeat(np.arange(len(ids), dtype=np.int32), tokens_per_window)
    dummy = np.zeros((2, max(len(ids), 2)), dtype=np.float32)
    eval_data = EvalData(
        binned_spikes=dummy,
        bin_size_seconds=0.001,
        duration=dummy.shape[1] * 0.001,
        n_units=2,
    )
    return IntermediatesCache(
        original=eval_data,
        reconstructed=eval_data,
        latents=latents,
        behavioral_data_token_res={
            "image_id": image_id_token_res,
            "window_group_id_token_res": window_group_id_token_res,
        },
    )


def test_per_dim_shape_matches_token_shape():
    """Per-dim arrays have size D and reshape cleanly back to token_shape."""
    S, R, T, D_latent = 6, 4, 3, 5
    rng = np.random.default_rng(0)
    mu = rng.normal(size=(S, T, D_latent))
    windows = []
    ids = []
    for s in range(S):
        for _ in range(R):
            windows.append(mu[s] + 0.1 * rng.normal(size=(T, D_latent)))
            ids.append(s)
    latents = np.stack(windows, axis=0)  # [N, T, D_latent]

    cache = _make_cache(latents, np.array(ids), tokens_per_window=T)
    metric = RepeatsSNR()
    req = metric.get_intermediate_requests(data_type=None)[0]
    inter = cache.get(req)
    result = metric.compute({"stimulus_grouped_latents": inter})

    D_flat = T * D_latent
    assert result.metadata["sigma2_sig_raw"].shape == (D_flat,)
    assert result.metadata["token_shape"] == (T, D_latent)
    reshaped = result.metadata["sigma2_sig_raw"].reshape(result.metadata["token_shape"])
    assert reshaped.shape == (T, D_latent)


def test_clip_at_zero_for_no_signal_dim():
    """A dimension with zero signal but nonzero noise should clip to 0."""
    S, R = 20, 5
    rng = np.random.default_rng(42)
    # D=2: dim0 has real signal, dim1 is pure noise (mu_s[:, 1] = 0).
    mu = np.zeros((S, 2))
    mu[:, 0] = rng.normal(0, 2.0, size=S)
    windows = []
    ids = []
    for s in range(S):
        for _ in range(R):
            windows.append(mu[s] + rng.normal(0, 1.0, size=2))
            ids.append(s)
    latents = np.stack(windows, axis=0)[:, None, :]

    cache = _make_cache(latents, np.array(ids))
    metric = RepeatsSNR()
    req = metric.get_intermediate_requests(data_type=None)[0]
    inter = cache.get(req)
    result = metric.compute({"stimulus_grouped_latents": inter})

    # dim1 (index 1) must be clipped at 0.
    assert result.metadata["sigma2_sig_debiased"][1] >= 0.0
    # raw may be negative-looking positive; but debiased is exactly 0 with high probability
    # because mean of per-stim within-repeat variance / n_bar > between-stim var of the means.
    # We only assert non-negative.


def test_single_stimulus_raises():
    """S=1: can't estimate between-stimulus variance → clear error."""
    R, D = 5, 3
    rng = np.random.default_rng(0)
    latents = rng.normal(size=(R, 1, D))
    ids = np.zeros(R, dtype=int)

    cache = _make_cache(latents, ids)
    metric = RepeatsSNR()
    req = metric.get_intermediate_requests(data_type=None)[0]
    inter = cache.get(req)
    with pytest.raises(ValueError, match=r"(?i)stimul"):
        metric.compute({"stimulus_grouped_latents": inter})


def test_all_single_repeat_raises():
    """R=1 everywhere: noise variance undefined → clear error."""
    S, D = 5, 3
    rng = np.random.default_rng(0)
    latents = rng.normal(size=(S, 1, D))
    ids = np.arange(S)

    cache = _make_cache(latents, ids)
    metric = RepeatsSNR()
    req = metric.get_intermediate_requests(data_type=None)[0]
    inter = cache.get(req)
    with pytest.raises(ValueError, match=r"(?i)repeat"):
        metric.compute({"stimulus_grouped_latents": inter})
