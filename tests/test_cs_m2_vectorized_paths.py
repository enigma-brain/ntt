"""Direct-correctness tests for the vectorized latent_robustness (M2) paths.

1. ``MutualInformation._accumulate_window`` counts per-unit joint and marginal
   values with ``np.bincount``; the tests compare its state with a naive
   ``collections.Counter`` reference.

2. ``NTTEvalRobustnessLatentJob._reconstruct_tokens_with_fixed_input_chunking``
   decodes all chunks of a window in one batch-merged ``tokenizer.reconstruct``
   call; the tests compare it with decoding the chunks one by one.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
import torch

from ntt.contrib.tokenizer.compressed_sensing import (
    CompressedSensingTokenizer,
    CompressedSensingTokenizerSpec,
)
from ntt.evaluation.metrics.information import MutualInformation
from ntt.jobs.eval_robustness_latent import NTTEvalRobustnessLatentJob

# ---------------------------------------------------------------------------
# 1) MutualInformation accumulation
# ---------------------------------------------------------------------------


class _StubIntermediates:
    """Minimal stand-in for ``IntermediatesCache.get(IntermediateRequest)``.

    ``MutualInformation._accumulate_window`` looks up
    ``binned_counts_original`` / ``binned_counts_reconstructed`` and expects a
    ``dict[bin_size_ms -> ndarray[n_units, T]]``. We keep a small map keyed
    on the request's ``key`` attribute and return the cached dict.
    """

    def __init__(self, counts_orig, counts_recon):
        self._data = {
            "binned_counts_original": counts_orig,
            "binned_counts_reconstructed": counts_recon,
        }

    def get(self, request):
        return self._data[request.key]


def _make_synthetic_counts(rng, n_units, T, max_value):
    """Random Poisson-like integer counts shaped ``[n_units, T]``."""
    return rng.integers(low=0, high=max_value + 1, size=(n_units, T)).astype(np.float64)


def _naive_state(windows):
    """Per-bin, per-unit joint/marginal Counters over every window, built value by value."""
    state: dict = {}
    for orig_by_bin, recon_by_bin in windows:
        for bin_size, orig in orig_by_bin.items():
            recon = recon_by_bin[bin_size]
            bin_state = state.setdefault(
                bin_size,
                {
                    "joint": [Counter() for _ in range(orig.shape[0])],
                    "orig": [Counter() for _ in range(orig.shape[0])],
                    "recon": [Counter() for _ in range(orig.shape[0])],
                    "total": np.zeros(orig.shape[0], dtype=np.int64),
                },
            )
            for unit in range(orig.shape[0]):
                x = np.round(orig[unit]).astype(int)
                y = np.round(recon[unit]).astype(int)
                bin_state["joint"][unit].update(zip(x.tolist(), y.tolist(), strict=True))
                bin_state["orig"][unit].update(x.tolist())
                bin_state["recon"][unit].update(y.tolist())
                bin_state["total"][unit] += x.size
    return state


@pytest.mark.parametrize(
    ("n_units", "T", "max_value", "n_windows", "n_bin_sizes"),
    [
        (4, 50, 3, 1, 1),  # tiny
        (12, 100, 5, 3, 5),  # 5 bin sizes x 3 windows
        (96, 200, 4, 2, 5),  # cowley-shape, 5 bin sizes
        (3, 30, 0, 1, 1),  # all-zero edge case
    ],
)
def test_mutual_information_accumulate_window_matches_naive_counts(
    n_units, T, max_value, n_windows, n_bin_sizes
):
    rng = np.random.default_rng(seed=42 + n_units + T)
    bin_sizes = sorted([1.0, 5.0, 10.0, 50.0, 100.0][:n_bin_sizes])

    metric = MutualInformation()
    state: dict = {}
    windows = []
    for _ in range(n_windows):
        counts_orig = {bs: _make_synthetic_counts(rng, n_units, T, max_value) for bs in bin_sizes}
        counts_recon = {bs: _make_synthetic_counts(rng, n_units, T, max_value) for bs in bin_sizes}
        windows.append((counts_orig, counts_recon))
        metric._accumulate_window(state, _StubIntermediates(counts_orig, counts_recon))

    expected = _naive_state(windows)
    assert state["n_windows_seen"] == n_windows
    assert set(state["per_bin_state"]) == set(expected)
    for bin_size, got in state["per_bin_state"].items():
        want = expected[bin_size]
        assert got["joint_counts_per_unit"] == [dict(c) for c in want["joint"]]
        assert got["orig_counts_per_unit"] == [dict(c) for c in want["orig"]]
        assert got["recon_counts_per_unit"] == [dict(c) for c in want["recon"]]
        np.testing.assert_array_equal(got["total_samples_per_unit"], want["total"])


def test_mutual_information_accumulate_rejects_negative_counts():
    """A negative count fails loudly instead of being binned."""
    metric = MutualInformation()
    counts_orig = {1.0: np.array([[0.0, -1.0, 2.0]])}
    counts_recon = {1.0: np.array([[0.0, 1.0, 2.0]])}
    intermediates = _StubIntermediates(counts_orig, counts_recon)
    with pytest.raises(ValueError, match="non-negative"):
        metric._accumulate_window({}, intermediates)


# ---------------------------------------------------------------------------
# 2) Chunked decode
# ---------------------------------------------------------------------------


def _make_cs_tokenizer(n_neurons, latent_dim=10, signal_dim_ms=100):
    spec = CompressedSensingTokenizerSpec(
        window_length_seconds=signal_dim_ms / 1000.0,
        bin_size_ms=1.0,
        n_neurons=n_neurons,
        compress_mode="per_neuron",
        latent_dim=latent_dim,
        fista_iterations=50,  # keep tests fast; equivalence is iteration-count-free
        fista_lambda=0.01,
        nonneg=True,
        trainable=False,
        random_seed=0,
        matrix_type="gaussian",
        solver="fista",
    )
    return CompressedSensingTokenizer(spec).eval()


@pytest.mark.parametrize(
    ("B", "n_chunks", "n_units"),
    [(1, 1, 5), (2, 1, 8), (3, 10, 12), (2, 7, 8)],
)
def test_reconstruct_chunked_merged_matches_chunk_by_chunk_cs(B, n_chunks, n_units):
    """Batch-merged chunked decode reproduces decoding the chunks one by one."""
    torch.manual_seed(123)
    tokenizer = _make_cs_tokenizer(n_neurons=n_units)
    spec = tokenizer.spec

    # tokens shape mirrors what the latent_robustness (M2) collator emits for CS: per-chunk
    # latent shape is [B, 1, U, m]; ``n_chunks`` chunks concatenate along
    # dim 1 → [B, n_chunks, U, m].
    tokens = torch.randn(B, n_chunks, n_units, spec.latent_dim)

    # A stub job: ``__init__`` is never called, the method only needs ``self``.
    job = NTTEvalRobustnessLatentJob.__new__(NTTEvalRobustnessLatentJob)

    out_new = job._reconstruct_tokens_with_fixed_input_chunking(tokenizer=tokenizer, tokens=tokens)
    chunk_len = tokens.shape[1] // n_chunks
    out_sequential = torch.cat(
        [
            tokenizer.reconstruct(tokens[:, i * chunk_len : (i + 1) * chunk_len]).reconstruction
            for i in range(n_chunks)
        ],
        dim=1,
    )

    # Same FISTA, same A, same y → tensors should be element-wise equal.
    # CS FISTA is fully deterministic given fixed seed and inputs; we use
    # ``allclose`` with a tiny tolerance to absorb any float-reassociation
    # from BLAS path selection between batch sizes.
    torch.testing.assert_close(
        out_new.reconstruction,
        out_sequential,
        rtol=1e-5,
        atol=1e-5,
    )
    # And the rebuilt time axis length matches the sum of chunks.
    expected_t = n_chunks * (int(spec.window_length_seconds * 1000.0 / spec.bin_size_ms))
    assert out_new.reconstruction.shape[1] == expected_t
