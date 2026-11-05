"""Core tests for ``RepeatsReconstructionBaseline``.

Covers:
  - no (stim_a, stim_b) pair is ever produced across different stimuli;
  - identical repeats produce a trivially-zero "mse_identity" metric of 0;
  - pair/stimulus counting matches combinatorics.

"""

from __future__ import annotations

from typing import Any

import numpy as np

from ntt.evaluation.core.intermediates import IntermediatesCache
from ntt.evaluation.core.types import (
    DataType,
    EvalData,
    MetricCategory,
    MetricDirection,
    MetricLevel,
    MetricResult,
)
from ntt.evaluation.metrics.base import BaseMetric
from ntt.evaluation.metrics.repeats_baseline import (
    RepeatsReconstructionBaseline,
    _iter_stimulus_pairs,
)


class _MseIdentityMetric(BaseMetric):
    """Trivial pair metric: MSE between reconstructed vs original counts.

    When both inputs are identical (copies) this returns 0 exactly.
    Not registered - used only as a test fixture injected into the
    baseline.
    """

    name = "mse_identity"
    category = MetricCategory.POPULATION_STRUCTURE
    direction = MetricDirection.LOWER_IS_BETTER
    level = MetricLevel.POPULATION
    _default_params: dict[str, Any] = {}
    _required_intermediates = [
        "native_binned_counts_original",
        "native_binned_counts_reconstructed",
    ]
    supported_data_types = {DataType.RATES}
    preferred_data_type = DataType.RATES

    def get_intermediate_requests(self, data_type):
        """Ask for native binned counts on both sides."""
        from ntt.evaluation.metrics.base import IntermediateRequest

        return [
            IntermediateRequest(key="native_binned_counts_original"),
            IntermediateRequest(key="native_binned_counts_reconstructed"),
        ]

    def compute(self, intermediates):
        """Return MSE between the original-side and reconstructed-side counts."""
        a = np.asarray(intermediates["native_binned_counts_original"], dtype=np.float64)
        b = np.asarray(intermediates["native_binned_counts_reconstructed"], dtype=np.float64)
        mse = float(np.mean((a - b) ** 2))
        return MetricResult(
            name=self.name,
            category=self.category,
            direction=self.direction,
            level=self.level,
            raw_value=mse,
        )


def _make_synthetic_cache(
    n_stim: int = 3,
    n_repeats: int = 3,
    n_units: int = 2,
    n_input_bins_per_window: int = 10,
    n_token_bins_per_window: int = 4,
    identical_repeats: bool = True,
    seed: int = 0,
) -> IntermediatesCache:
    """Build an IntermediatesCache populated with synthetic repeat data."""
    rng = np.random.default_rng(seed)
    total_windows = n_stim * n_repeats
    latent_dim = 3
    per_window_latents = [
        rng.standard_normal((n_token_bins_per_window, latent_dim)).astype(np.float64)
        for _ in range(total_windows)
    ]

    stim_per_window = []
    base_recon_per_stim = {
        s: rng.standard_normal((n_units, n_input_bins_per_window)).astype(np.float64)
        for s in range(n_stim)
    }
    per_window_recon = []
    for s in range(n_stim):
        for _ in range(n_repeats):
            stim_per_window.append(s)
            if identical_repeats:
                per_window_recon.append(base_recon_per_stim[s].copy())
            else:
                per_window_recon.append(
                    base_recon_per_stim[s] + rng.standard_normal(base_recon_per_stim[s].shape)
                )

    grouping_values = np.repeat(
        np.asarray(stim_per_window, dtype=np.int64), n_token_bins_per_window
    )
    window_group_ids = np.repeat(np.arange(total_windows, dtype=np.int64), n_token_bins_per_window)
    behavioral = {
        "image_id": grouping_values,
        "window_group_id_token_res": window_group_ids,
    }

    recon_stacked = np.concatenate(per_window_recon, axis=1)
    original = EvalData(
        binned_counts=np.zeros_like(recon_stacked),
        bin_size_seconds=0.01,
        data_type=DataType.RATES,
    )
    reconstructed = EvalData(
        binned_counts=recon_stacked,
        bin_size_seconds=0.01,
        data_type=DataType.RATES,
    )

    return IntermediatesCache(
        original=original,
        reconstructed=reconstructed,
        latents=per_window_latents,
        behavioral_data_token_res=behavioral,
    )


def test_iter_stimulus_pairs_stays_within_stimulus():
    """Pairs never cross stimulus boundaries."""
    repeat_mask = np.array(
        [
            [True, True, True],
            [True, True, False],
            [True, True, True],
        ]
    )
    pairs = _iter_stimulus_pairs(repeat_mask)
    # stim 0: C(3,2)=3; stim 1: C(2,2)=1; stim 2: C(3,2)=3 -> total 7.
    assert len(pairs) == 7
    for s, i, j in pairs:
        assert 0 <= s < repeat_mask.shape[0]
        assert i < j
        assert repeat_mask[s, i]
        assert repeat_mask[s, j]


def test_identical_repeats_give_zero_mse_and_expected_counts():
    """Identical repeats -> zero MSE; counts match combinatorics."""
    cache = _make_synthetic_cache(
        n_stim=3,
        n_repeats=3,
        identical_repeats=True,
        seed=42,
    )
    metric = RepeatsReconstructionBaseline(
        recon_metrics=[_MseIdentityMetric()],
        grouping_field="image_id",
    )
    requests = metric.get_intermediate_requests(DataType.RATES)
    intermediates = {r.key: cache.get(r) for r in requests}
    result = metric.compute(intermediates)
    raw = result.raw_value
    assert "mse_identity" in raw
    # 3 stimuli x C(3,2) = 9 ordered pairs (i<j).
    assert result.metadata["total_pairs"] == 9
    assert result.metadata["n_stimuli"] == 3
    assert raw["mse_identity"]["mean"] == 0.0
    for _s, per_s in raw["mse_identity"]["per_stimulus"].items():
        assert per_s["mean"] == 0.0
        assert per_s["n_pairs"] == 3


def test_nonidentical_repeats_give_positive_mse():
    """Noisy repeats -> strictly positive MSE aggregate."""
    cache = _make_synthetic_cache(
        n_stim=3,
        n_repeats=3,
        identical_repeats=False,
        seed=7,
    )
    metric = RepeatsReconstructionBaseline(
        recon_metrics=[_MseIdentityMetric()],
        grouping_field="image_id",
    )
    requests = metric.get_intermediate_requests(DataType.RATES)
    intermediates = {r.key: cache.get(r) for r in requests}
    result = metric.compute(intermediates)
    raw = result.raw_value
    assert raw["mse_identity"]["mean"] > 0.0
