"""Core tests for the TokenDimsPerSecond compression metric.

Covers the four key paths of the metric:
1. Declared ``token_bin_size_ms`` present → used directly.
2. Only ``token_timestamps`` present → inferred via median diff.
3. Per-neuron latent shape ``[N_windows, T, U, D]`` → dims multiply.
4. Neither declared bin nor timestamps → fails loud with a clear message.
"""

import numpy as np
import pytest

from ntt.evaluation.metrics.compression import TokenDimsPerSecond


def test_declared_bin_size_population_tokenizer():
    """Declared bin size: 8 dims / 20 ms = 400 dims/s."""
    latents = np.random.randn(5, 10, 8).astype(np.float32)  # [N, T, D]
    intermediates = {
        "latents": latents,
        "token_bin_size_ms": 20.0,
        "token_timestamps": None,
    }
    result = TokenDimsPerSecond().compute(intermediates)
    assert result.raw_value == pytest.approx(8 / 0.020)
    assert result.metadata["source"] == "declared"
    assert result.metadata["token_bin_size_ms"] == 20.0
    assert result.metadata["latent_dim"] == 8
    assert result.metadata["n_token_dims_per_token"] == 8
    assert result.metadata["is_per_neuron"] is False


def test_inferred_bin_size_from_timestamps():
    """No declared bin — infer from token_timestamps median diff (1 ms here)."""
    latents = np.zeros((3, 6, 4), dtype=np.float32)  # D_latent = 4
    # 1 ms spacing → 4 / 0.001 = 4000 dims/s.
    timestamps = np.arange(0, 18, dtype=np.float64) * 0.001
    intermediates = {
        "latents": latents,
        "token_bin_size_ms": None,
        "token_timestamps": timestamps,
    }
    result = TokenDimsPerSecond().compute(intermediates)
    assert result.raw_value == pytest.approx(4 / 0.001)
    assert result.metadata["source"] == "inferred"
    assert result.metadata["token_bin_size_ms"] == pytest.approx(1.0)
    assert result.metadata["latent_dim"] == 4


def test_per_neuron_tokenizer_shape():
    """Per-neuron latents [N, T, U=6, D=4] with 5 ms bin → 24 / 0.005 = 4800 dims/s."""
    latents = np.zeros((2, 3, 6, 4), dtype=np.float32)  # [N, T, U, D]
    intermediates = {
        "latents": latents,
        "token_bin_size_ms": 5.0,
        "token_timestamps": None,
    }
    result = TokenDimsPerSecond().compute(intermediates)
    assert result.raw_value == pytest.approx(24 / 0.005)
    assert result.metadata["n_token_dims_per_token"] == 24
    assert result.metadata["latent_dim"] == 4
    assert result.metadata["is_per_neuron"] is True
    assert result.metadata["per_token_shape"] == [6, 4]


def test_fail_loud_when_neither_bin_nor_timestamps():
    """Must raise ValueError when bin size cannot be determined."""
    latents = np.zeros((2, 3, 4), dtype=np.float32)
    intermediates = {
        "latents": latents,
        "token_bin_size_ms": None,
        "token_timestamps": None,
    }
    with pytest.raises(ValueError, match="token_bin_size_ms.*token_timestamps"):
        TokenDimsPerSecond().compute(intermediates)
