"""Core tests for the repeats-based SNR metric (Metric A).

The tests use small synthetic repeats with known signal and noise.
"""

import warnings

import numpy as np
import pytest
import torch

from ntt.evaluation.core.intermediates import IntermediatesCache
from ntt.evaluation.core.types import EvalData
from ntt.evaluation.metrics.base import IntermediateRequest
from ntt.evaluation.metrics.repeats import RepeatsSNR

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _build_cache_from_grouped_latents(
    latents_flat: np.ndarray,
    image_ids_per_window: np.ndarray,
    tokens_per_window: int = 1,
) -> IntermediatesCache:
    """Build a minimal IntermediatesCache that feeds stimulus_grouped_latents.

    Args:
        latents_flat: [N_windows, T_tokens, D_latent].
        image_ids_per_window: [N_windows] — one label per window.
        tokens_per_window: number of tokens per window (used to expand behavioral
            fields to token resolution).

    """
    N = latents_flat.shape[0]

    # Behavioral data is at token resolution: one value per token, same across
    # all tokens within a window.
    image_id_token_res = np.repeat(image_ids_per_window, tokens_per_window)
    # Window group ids the collation pipeline normally attaches — one int per
    # token telling you which window that token belongs to.
    window_group_id_token_res = np.repeat(np.arange(N, dtype=np.int32), tokens_per_window)

    # Minimal EvalData: we don't actually need spike data for this intermediate
    # since only latents + behavioral_data_token_res are consumed.
    n_units = 2
    total_bins = max(N, 2)
    dummy_spikes = np.zeros((n_units, total_bins), dtype=np.float32)
    eval_data = EvalData(
        binned_spikes=dummy_spikes,
        bin_size_seconds=0.001,
        duration=total_bins * 0.001,
        n_units=n_units,
    )

    cache = IntermediatesCache(
        original=eval_data,
        reconstructed=eval_data,
        latents=latents_flat,
        behavioral_data_token_res={
            "image_id": image_id_token_res,
            "window_group_id_token_res": window_group_id_token_res,
        },
    )
    return cache


def _build_cache_from_ragged_latents(
    latents_list: list[np.ndarray],
    image_ids_per_window: np.ndarray,
) -> IntermediatesCache:
    """Build cache with ragged (variable-length) windows as a list of arrays.

    Each element: [T_i, D_latent]. Behavioral data is built per-window to
    match each window's token count.

    """
    N = len(latents_list)
    assert len(image_ids_per_window) == N

    image_id_chunks = []
    wid_chunks = []
    for i, arr in enumerate(latents_list):
        T_i = arr.shape[0]
        image_id_chunks.append(np.full(T_i, image_ids_per_window[i]))
        wid_chunks.append(np.full(T_i, i, dtype=np.int32))
    image_id_token_res = np.concatenate(image_id_chunks)
    window_group_id_token_res = np.concatenate(wid_chunks)

    n_units = 2
    total_bins = max(N, 2)
    dummy_spikes = np.zeros((n_units, total_bins), dtype=np.float32)
    eval_data = EvalData(
        binned_spikes=dummy_spikes,
        bin_size_seconds=0.001,
        duration=total_bins * 0.001,
        n_units=n_units,
    )

    cache = IntermediatesCache(
        original=eval_data,
        reconstructed=eval_data,
        latents=latents_list,  # list triggers ragged-handling path
        behavioral_data_token_res={
            "image_id": image_id_token_res,
            "window_group_id_token_res": window_group_id_token_res,
        },
    )
    return cache


def _generate_synthetic(
    rng: np.random.Generator,
    S: int,
    R: int | list[int],
    D: int,
    sigma_sig: float,
    sigma_noise: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate per-(stimulus, repeat) observations.

    Returns:
        latents_flat: [N_windows, 1, D] — one token per window.
        image_ids_per_window: [N_windows] with stimulus label.

    """
    mu = rng.normal(0.0, sigma_sig, size=(S, D))
    if isinstance(R, int):
        R_per = [R] * S
    else:
        R_per = R
    all_latents = []
    all_ids = []
    for s in range(S):
        for _ in range(R_per[s]):
            z = mu[s] + rng.normal(0.0, sigma_noise, size=(D,))
            all_latents.append(z)
            all_ids.append(s)
    latents_flat = np.stack(all_latents, axis=0)[:, None, :]  # [N, 1, D]
    return latents_flat, np.array(all_ids)


# ----------------------------------------------------------------------
# Test 1: Ground-truth recovery (large-sample limit)
# ----------------------------------------------------------------------


def test_snr_ground_truth_recovery_large_sample():
    """With S=30, R=20, SNR_debiased should recover sigma_sig^2/sigma_noise^2.

    Averaged over multiple seeds to reduce flakiness; within 15%.
    """
    S, R, D = 30, 20, 5
    sigma_sig, sigma_noise = 2.0, 1.0
    true_ratio = (sigma_sig**2) / (sigma_noise**2)

    debiased_values = []
    raw_values = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        latents, ids = _generate_synthetic(rng, S, R, D, sigma_sig, sigma_noise)
        cache = _build_cache_from_grouped_latents(latents, ids, tokens_per_window=1)

        metric = RepeatsSNR()
        req = metric.get_intermediate_requests(data_type=None)[0]
        inter = cache.get(req)
        result = metric.compute({"stimulus_grouped_latents": inter})

        debiased_values.append(result.raw_value)
        raw_values.append(result.metadata["snr_raw"])

    mean_debiased = np.mean(debiased_values)
    mean_raw = np.mean(raw_values)

    # Within 15% of the true ratio for the debiased estimate.
    assert abs(mean_debiased - true_ratio) / true_ratio < 0.15, (
        f"mean_debiased={mean_debiased}, true={true_ratio}"
    )
    # Raw should also be close here because bias = noise/(R*sigma_noise^2) is small.
    assert abs(mean_raw - true_ratio) / true_ratio < 0.20


# ----------------------------------------------------------------------
# Test 2: Bias correction matters with few repeats
# ----------------------------------------------------------------------


def test_snr_bias_correction_with_few_repeats():
    """With R=3, the raw estimator overestimates the true SNR and debiasing
    recovers it; debiased is both closer to truth and unbiased in magnitude."""
    S, R, D = 30, 3, 5
    sigma_sig, sigma_noise = 1.0, 1.0
    true_ratio = (sigma_sig**2) / (sigma_noise**2)  # == 1.0

    raw_values = []
    debiased_values = []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        latents, ids = _generate_synthetic(rng, S, R, D, sigma_sig, sigma_noise)
        cache = _build_cache_from_grouped_latents(latents, ids, tokens_per_window=1)

        metric = RepeatsSNR()
        req = metric.get_intermediate_requests(data_type=None)[0]
        inter = cache.get(req)
        result = metric.compute({"stimulus_grouped_latents": inter})

        raw_values.append(result.metadata["snr_raw"])
        debiased_values.append(result.raw_value)

    mean_raw = float(np.mean(raw_values))
    mean_debiased = float(np.mean(debiased_values))

    # Ground-truth magnitude: with R=3 the bias = noise/(R·σ_noise²) is large,
    # so the raw estimator overestimates the true ratio by a clear margin while
    # the debiased estimator recovers it (within 20% — looser than the
    # large-sample test because R=3 leaves more variance).
    assert mean_raw > true_ratio * 1.15, (
        f"raw should be clearly upward-biased; mean_raw={mean_raw}, true={true_ratio}"
    )
    assert abs(mean_debiased - true_ratio) / true_ratio < 0.20, (
        f"debiased should recover the true ratio; mean_debiased={mean_debiased}, true={true_ratio}"
    )
    # And debiased is closer to the truth than raw.
    assert abs(mean_debiased - true_ratio) < abs(mean_raw - true_ratio)


# ----------------------------------------------------------------------
# Test 3: Ragged repeats
# ----------------------------------------------------------------------


def test_snr_ragged_repeats():
    """Per-stimulus R ∈ {3, 5, 10} (random); metric runs, primary positive & finite."""
    S, D = 20, 4
    rng = np.random.default_rng(0)
    R_per = rng.choice([3, 5, 10], size=S).tolist()
    latents, ids = _generate_synthetic(rng, S, R_per, D, sigma_sig=1.0, sigma_noise=1.0)

    cache = _build_cache_from_grouped_latents(latents, ids, tokens_per_window=1)
    metric = RepeatsSNR()
    req = metric.get_intermediate_requests(data_type=None)[0]
    inter = cache.get(req)
    result = metric.compute({"stimulus_grouped_latents": inter})

    assert np.isfinite(result.raw_value)
    assert result.raw_value > 0

    # arithmetic mean check (the ragged-aware per-stimulus debias does not rely
    # on a single aggregate repeat count, so there is no harmonic mean).
    expected_arithmetic = float(np.mean(R_per))
    assert result.metadata["n_repeats_mean_arithmetic"] == pytest.approx(
        expected_arithmetic, rel=1e-10
    )
    assert result.metadata["n_stimuli"] == S
    assert result.metadata["n_repeats_min"] == min(R_per)
    assert result.metadata["n_repeats_max"] == max(R_per)


# ----------------------------------------------------------------------
# Test 4: D-mismatch fallback (cut to smallest)
# ----------------------------------------------------------------------


def test_snr_token_count_mismatch_fallback():
    """10 windows T=20, 2 windows T=15 → UserWarning; metadata records the cut."""
    D_latent = 3
    rng = np.random.default_rng(42)

    # 6 stimuli × 2 repeats = 12 windows. Give 2 of them T=15, rest T=20.
    S = 6
    R = 2
    windows = []
    ids = []
    for s in range(S):
        for r in range(R):
            T_i = 15 if (s == 0 and r == 0) or (s == 1 and r == 0) else 20
            windows.append(rng.normal(size=(T_i, D_latent)))
            ids.append(s)

    cache = _build_cache_from_ragged_latents(windows, np.array(ids))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        metric = RepeatsSNR()
        req = metric.get_intermediate_requests(data_type=None)[0]
        inter = cache.get(req)

    user_warnings = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(user_warnings) >= 1, "Expected a UserWarning about the token-count cut."

    cut_info = inter["n_tokens_cut_info"]
    assert cut_info is not None
    assert cut_info["n_tokens_min"] == 15
    assert cut_info["n_tokens_max"] == 20
    assert cut_info["n_tokens_used"] == 15
    # None should end up with zero tokens here.
    assert cut_info["n_windows_dropped"] == 0

    # Metric still runs to completion:
    result = metric.compute({"stimulus_grouped_latents": inter})
    assert np.isfinite(result.raw_value)
    assert result.metadata["n_tokens_per_stimulus_repeat_used"] == 15
    assert result.metadata["n_tokens_cut_info"] == cut_info


# ----------------------------------------------------------------------
# Test 5: Integration with intermediate cache
# ----------------------------------------------------------------------


def test_snr_integration_with_cache():
    """End-to-end via IntermediatesCache; check metadata keys and approximate value."""
    S, R, D = 15, 8, 3
    sigma_sig, sigma_noise = 1.5, 0.8
    rng = np.random.default_rng(123)
    latents, ids = _generate_synthetic(rng, S, R, D, sigma_sig, sigma_noise)

    cache = _build_cache_from_grouped_latents(latents, ids, tokens_per_window=1)
    metric = RepeatsSNR()

    # Pull via the intermediate request (what the pipeline would do).
    req = metric.get_intermediate_requests(data_type=None)[0]
    assert isinstance(req, IntermediateRequest)
    assert req.key == "stimulus_grouped_latents"
    assert req.params_dict == {"grouping_field": "image_id"}

    inter = cache.get(req)
    # Cache hit on repeat call.
    inter2 = cache.get(req)
    assert inter is inter2

    # Shape checks on the intermediate itself.
    assert inter["latents"].shape == (S, R, 1 * D)
    assert inter["repeat_mask"].shape == (S, R)
    assert inter["repeat_mask"].all()
    assert inter["stimulus_ids"].shape == (S,)
    np.testing.assert_array_equal(np.sort(inter["stimulus_ids"]), np.arange(S))
    assert inter["repeat_counts"].tolist() == [R] * S
    assert inter["token_shape"] == (1, D)
    assert inter["n_tokens_cut_info"] is None

    # Compute metric.
    result = metric.compute({"stimulus_grouped_latents": inter})
    assert result.metadata["n_tokens_per_stimulus_repeat_used"] == 1

    # Metadata keys per spec.
    expected_keys = {
        "snr_raw",
        "sigma2_sig_raw",
        "sigma2_sig_debiased",
        "sigma2_noise",
        "n_stimuli",
        "n_repeats_mean_arithmetic",
        "n_repeats_min",
        "n_repeats_max",
        "grouping_field",
        "token_shape",
        "n_tokens_per_stimulus_repeat_used",
        "n_tokens_cut_info",
    }
    assert expected_keys.issubset(result.metadata.keys())

    # Per-dim arrays have size D (here token_shape = (1, D) so flat D = 3).
    assert result.metadata["sigma2_sig_raw"].shape == (D,)
    assert result.metadata["sigma2_sig_debiased"].shape == (D,)
    assert result.metadata["sigma2_noise"].shape == (D,)

    # Roughly matches true ratio.
    true_ratio = sigma_sig**2 / sigma_noise**2
    assert abs(result.raw_value - true_ratio) / true_ratio < 0.5  # single seed, lenient


# ----------------------------------------------------------------------
# Test: Ragged debias under homoscedastic noise
# ----------------------------------------------------------------------


def test_ragged_homoscedastic_debias():
    """Debias must recover the true SNR ratio under ragged repeats + homoscedastic noise.

    Setup: same S, ragged N_s (mix of 3 and 30), but uniform noise std across
    all stimuli (homoscedastic). Under the pooled homoscedastic debias
    ``sigma2_sig_raw - sigma2_noise * mean_s(1/N_s)`` should recover
    ``sigma_sig^2 / sigma_noise^2`` in expectation. The raw (un-debiased)
    SNR should be clearly biased upward because mean_s(1/N_s) > 0.

    """
    S_half, D = 30, 4
    S = 2 * S_half
    # Choose sigma_sig < sigma_noise so the raw (un-debiased) bias
    # sigma_noise^2 * mean_s(1/N_s) is a visibly large fraction of true_ratio.
    sigma_sig = 0.5
    # Homoscedastic: same noise std for all stimuli. Repeats remain ragged.
    sigma_noise = 1.0
    sigma_noise_by_stim = np.array([sigma_noise] * S)
    N_by_stim = np.array([3] * S_half + [30] * S_half)

    # Theoretical: per-dim debiased SNR = sigma_sig^2 / sigma_noise^2
    true_ratio = (sigma_sig**2) / (sigma_noise**2)

    debiased_values = []
    raw_values = []
    n_replicates = 200
    for seed in range(n_replicates):
        rng = np.random.default_rng(seed + 10_000)
        mu = rng.normal(0.0, sigma_sig, size=(S, D))
        all_z = []
        all_ids = []
        for s in range(S):
            for _ in range(int(N_by_stim[s])):
                z = mu[s] + rng.normal(0.0, sigma_noise_by_stim[s], size=(D,))
                all_z.append(z)
                all_ids.append(s)
        latents_flat = np.stack(all_z, axis=0)[:, None, :]  # [N, 1, D]
        ids = np.array(all_ids)

        cache = _build_cache_from_grouped_latents(latents_flat, ids, tokens_per_window=1)
        metric = RepeatsSNR()
        req = metric.get_intermediate_requests(data_type=None)[0]
        inter = cache.get(req)
        result = metric.compute({"stimulus_grouped_latents": inter})

        debiased_values.append(result.raw_value)
        raw_values.append(result.metadata["snr_raw"])

    mean_debiased = float(np.mean(debiased_values))
    mean_raw = float(np.mean(raw_values))

    # Debiased must be within 5% of theoretical.
    assert abs(mean_debiased - true_ratio) / true_ratio < 0.05, (
        f"mean_debiased={mean_debiased:.4f}, true={true_ratio:.4f}, "
        f"rel_err={abs(mean_debiased - true_ratio) / true_ratio:.4f}"
    )
    # Raw must be visibly biased upward (test has teeth).
    assert (mean_raw - true_ratio) / true_ratio > 0.15, (
        f"Expected raw SNR to be >15% above truth, got mean_raw={mean_raw:.4f}, "
        f"true={true_ratio:.4f}"
    )
