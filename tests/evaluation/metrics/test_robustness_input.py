"""Core tests for :class:`InputCorruptionRobustness` (M1-a).

Covers three minimal cases (now expressed in the squared-NMSE
formulation: numerator and denominator are both squared, so E = 2 under
i.i.d., not 1):

1. Zero-corruption replicas → all per-label ``mean_token_nmse`` and
   ``mean_per_dim_nmse`` are 0.
2. ``metadata["sigma_joint"]`` still matches ``np.std`` of replica-0
   latents (kept for diagnostic continuity).
3. Two corruption labels with constant per-replica offsets → per-label
   ``mean_token_nmse`` and ``mean_per_dim_nmse`` match the analytic
   values: for offset Δ, ``joint_nmse = Δ² / σ_token²`` and
   ``per_dim_nmse = Δ² · mean_d (1/σ_d²)``.

The metric receives a plain ``dict`` of intermediates (not an
``IntermediatesCache``); ``compute()`` indexes it directly via
``intermediates["replica_latents"]`` etc., so we mirror that here.
"""

from __future__ import annotations

import numpy as np

from ntt.evaluation.metrics.robustness_input import InputCorruptionRobustness
from ntt.evaluation.metrics.robustness_latent_stats import LatentStatsCache


def _make_metric() -> InputCorruptionRobustness:
    """Construct an input_robustness (M1) metric with a fresh σ cache."""
    return InputCorruptionRobustness(
        latent_stats_cache=LatentStatsCache(on_disk_root=None),
    )


def test_zero_corruption_gives_zero_distances():
    """Identical replicas → every per-label ``mean_joint`` is exactly 0."""
    N, K, T_tok, D = 4, 3, 5, 6
    # All zeros; replica-0 == replica-k for all k.
    replica_latents = np.zeros((N, K + 1, T_tok, D), dtype=np.float64)
    replica_metadata = {"corruption_labels": ["A", "B", "C"]}

    intermediates = {
        "replica_latents": replica_latents,
        "replica_metadata": replica_metadata,
    }

    metric = _make_metric()
    result = metric.compute(intermediates)

    assert set(result.raw_value.keys()) == {"A", "B", "C"}
    for label, stats in result.raw_value.items():
        assert stats["mean_token_nmse"] == 0.0, f"label={label}: {stats}"
        assert stats["std_token_nmse"] == 0.0
        assert stats["mean_per_dim_nmse"] == 0.0
        assert stats["std_per_dim_nmse"] == 0.0


def test_sigma_joint_matches_np_std_of_clean_latents():
    """``metadata["sigma_joint"]`` = ``replica0.std()`` (population std)."""
    rng = np.random.default_rng(42)
    N, K, T_tok, D = 8, 2, 7, 4
    clean = rng.standard_normal((N, T_tok, D)).astype(np.float64) * 3.7  # known scale
    # Replicas 1..K differ from clean (any non-zero perturbation works).
    perturbed = clean[None, :, :, :].repeat(K, axis=0) + rng.standard_normal((K, N, T_tok, D))
    # Stack into [N, K+1, T_tok, D].
    replica_latents = np.concatenate(
        [clean[:, None, :, :], np.transpose(perturbed, (1, 0, 2, 3))], axis=1
    )
    replica_metadata = {"corruption_labels": ["x", "y"]}

    intermediates = {
        "replica_latents": replica_latents,
        "replica_metadata": replica_metadata,
    }

    metric = _make_metric()
    result = metric.compute(intermediates)

    expected_sigma = float(clean.std())
    assert np.isclose(result.metadata["sigma_joint"], expected_sigma)


def test_per_label_aggregation_with_known_constant_offsets():
    """Two labels A, B with constant offsets Δa, Δb → analytic NMSEs.

    For a constant offset Δ added to every element of a corrupted replica
    relative to replica-0:
      diff = Δ everywhere → mean_{t,d}(diff²) = Δ²
      mean_token_nmse  = Δ² / σ_token²
      mean_per_dim_nmse = Δ² · mean_d (1/σ_d²)

    where σ_token² = mean_d σ_d² and σ_d² is the per-dim variance of the
    clean latents (per-dim mean subtracted).
    """
    rng = np.random.default_rng(7)
    N, T_tok, D = 6, 5, 3
    clean = rng.standard_normal((N, T_tok, D)).astype(np.float64)

    delta_a = 0.25
    delta_b = -1.5
    replica_a = clean + delta_a
    replica_b = clean + delta_b

    replica_latents = np.stack([clean, replica_a, replica_b], axis=1)  # [N, 3, T_tok, D]
    replica_metadata = {"corruption_labels": ["A", "B"]}

    intermediates = {
        "replica_latents": replica_latents,
        "replica_metadata": replica_metadata,
    }

    metric = _make_metric()
    result = metric.compute(intermediates)

    # σ_per_dim from the cache uses per-dim mean (np.std(axis=0)) over the
    # flattened (N·T_tok, D) clean tokens — same shape the metric used.
    clean_flat = clean.reshape(-1, D)
    sigma_per_dim = clean_flat.std(axis=0)
    sigma_per_dim_sq = sigma_per_dim**2
    sigma_token_sq = float(sigma_per_dim_sq.mean())

    expected_a_joint = float(delta_a**2 / sigma_token_sq)
    expected_b_joint = float(delta_b**2 / sigma_token_sq)
    expected_a_pd = float(delta_a**2 * (1.0 / sigma_per_dim_sq).mean())
    expected_b_pd = float(delta_b**2 * (1.0 / sigma_per_dim_sq).mean())

    assert np.isclose(result.raw_value["A"]["mean_token_nmse"], expected_a_joint)
    assert np.isclose(result.raw_value["B"]["mean_token_nmse"], expected_b_joint)
    assert np.isclose(result.raw_value["A"]["mean_per_dim_nmse"], expected_a_pd)
    assert np.isclose(result.raw_value["B"]["mean_per_dim_nmse"], expected_b_pd)


def test_variable_T_tok_per_window_is_handled():
    """Windows with different ``T_tok`` are accepted (list-of-arrays input).

    Builds a session of three windows with T_tok in {3, 5, 8}, all sharing
    K+1=2 (one clean + one corrupted) and D=2. Replica 1 is replica-0
    plus a constant offset Δ. The per-window ``d`` (per-element RMS over
    T_w and D, then divided by ``sigma_joint``) equals ``|Δ| / sigma_joint``
    regardless of T_w — that's the point of the per-element RMS.

    """
    rng = np.random.default_rng(11)
    D = 2
    delta = 0.4
    T_per_window = [3, 5, 8]

    windows: list[np.ndarray] = []
    for T_w in T_per_window:
        clean_w = rng.standard_normal((T_w, D)).astype(np.float64)
        corrupt_w = clean_w + delta
        windows.append(np.stack([clean_w, corrupt_w], axis=0))  # [2, T_w, D]

    intermediates = {
        "replica_latents": windows,  # explicit list — mixed T_w
        "replica_metadata": {"corruption_labels": ["A"]},
    }

    metric = _make_metric()
    result = metric.compute(intermediates)

    # For constant offset Δ across (t, d):
    # - joint NMSE sums the squared change across all D latent dims for
    #   each token.
    # - per-dim NMSE divides by σ_d² separately in each dimension, then
    #   averages those per-dimension normalized values.
    sigma_token_sq = result.metadata["sigma_token_sq"]
    sigma_per_dim_sq = np.asarray(result.metadata["sigma_per_dim"]) ** 2
    expected_joint = float(D * delta**2 / sigma_token_sq)
    expected_pd = float(delta**2 * (1.0 / sigma_per_dim_sq).mean())
    assert np.isclose(result.raw_value["A"]["mean_token_nmse"], expected_joint, atol=1e-9)
    assert np.isclose(result.raw_value["A"]["mean_per_dim_nmse"], expected_pd, atol=1e-9)
    assert sigma_token_sq > 0.0
