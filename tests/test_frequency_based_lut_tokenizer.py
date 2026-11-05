"""Tests for FrequencyBasedLUTTokenizer edge cases (zero vectors, sparse vocab, ...)."""

import warnings

import pytest
import torch

from ntt.contrib.tokenizer.frequency_based_lut import (
    FrequencyBasedLUTTokenizer,
    FrequencyBasedLUTTokenizerSpec,
)
from ntt.core.model.tokenizer import create_tokenizer


def _make_spec(**kwargs) -> FrequencyBasedLUTTokenizerSpec:
    """Small spec for fast tests: 10-bin windows at 1ms."""
    defaults = {
        "vocab_size": 8,
        "n_neurons": 1,
        "window_length_seconds": 0.01,  # 10 bins at 1ms
        "bin_size_ms": 1.0,
    }
    defaults.update(kwargs)
    return FrequencyBasedLUTTokenizerSpec(**defaults)


def _make_fitted_tokenizer(spec=None) -> FrequencyBasedLUTTokenizer:
    """Return a tokenizer already fitted on 4 distinct 10-bin patterns."""
    spec = spec or _make_spec()
    tok = create_tokenizer(spec)
    # Four distinct patterns: spike-at-0, spike-at-1, spike-at-2, all-zero
    data = torch.zeros(4, 10, 1)
    data[0, 0, 0] = 1.0
    data[1, 1, 0] = 1.0
    data[2, 2, 0] = 1.0
    # data[3] stays all-zero (silent period)
    tok.train()
    tok._accumulate_frequencies(data)
    tok.fit_from_running_stats()
    tok.eval()
    return tok


# ---------------------------------------------------------------------------
# Zero-vector cosine similarity
# ---------------------------------------------------------------------------


class TestZeroVectorEncoding:
    def test_all_zero_pattern_returns_valid_token(self):
        """Silent (all-zero) inputs must not produce NaN and must return a valid index."""
        tok = _make_fitted_tokenizer()
        silent = torch.zeros(1, 10, 1)
        tokens = tok.encode(silent)
        assert tokens.shape == (1, 1, 1, 1)
        assert not torch.isnan(tokens.float()).any()
        assert tokens.item() >= 0

    def test_mixed_zero_nonzero_batch_no_nan(self):
        """A batch where some samples are silent must not produce NaN anywhere."""
        tok = _make_fitted_tokenizer()
        x = torch.zeros(4, 10, 1)
        x[0, 0, 0] = 1.0  # only first sample non-zero
        tokens = tok.encode(x)
        assert tokens.shape == (4, 1, 1, 1)
        assert not torch.isnan(tokens.float()).any()


# ---------------------------------------------------------------------------
# Sparse vocab (fewer unique patterns than vocab_size)
# ---------------------------------------------------------------------------


class TestSparseVocab:
    def test_sparse_vocab_emits_warning(self):
        """When fewer unique patterns than vocab_size are seen, a UserWarning is raised."""
        spec = _make_spec(vocab_size=100)  # only 4 unique patterns
        tok = create_tokenizer(spec)
        data = torch.zeros(4, 10, 1)
        data[0, 0, 0] = 1.0
        tok.train()
        tok._accumulate_frequencies(data)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tok.fit_from_running_stats()
        assert any("clipped" in str(w.message).lower() for w in caught)

    def test_sparse_vocab_tokenizer_still_fits(self):
        """Even with sparse data the tokenizer must mark itself fitted and be usable."""
        spec = _make_spec(vocab_size=100)
        tok = create_tokenizer(spec)
        data = torch.zeros(2, 10, 1)
        data[0, 0, 0] = 1.0
        tok.train()
        tok._accumulate_frequencies(data)
        tok.fit_from_running_stats()
        tok.eval()
        assert tok.is_fit.item()
        tokens = tok.encode(data)
        assert tokens.shape == (2, 1, 1, 1)


# ---------------------------------------------------------------------------
# Design fix: fit_from_running_stats protocol
# ---------------------------------------------------------------------------


class TestFitFromRunningStats:
    def test_encode_before_fit_returns_zeros_not_raises(self):
        """encode() must return zero tokens before fitting, not raise RuntimeError."""
        spec = _make_spec()
        tok = create_tokenizer(spec)
        tok.eval()
        x = torch.zeros(2, 10, 1)
        tokens = tok.encode(x)
        assert tokens.shape == (2, 1, 1, 1)
        assert (tokens == 0).all()

    def test_decode_before_fit_does_not_raise(self):
        """decode() on zero tokens must not raise before fitting."""
        spec = _make_spec()
        tok = create_tokenizer(spec)
        tok.eval()
        zero_tokens = torch.zeros(2, 1, 1, 1, dtype=torch.long)
        recon = tok.decode(zero_tokens)
        assert recon.shape == (2, 10, 1)

    def test_training_forward_accumulates_counts(self):
        """forward() in training mode must populate _pattern_counts before fitting."""
        spec = _make_spec()
        tok = create_tokenizer(spec)
        tok.train()
        data = torch.zeros(2, 10, 1)
        data[0, 0, 0] = 1.0
        tok(data)
        assert len(tok._pattern_counts) > 0

    def test_fit_from_running_stats_returns_true_first_call(self):
        """fit_from_running_stats() must return True the first time it fits."""
        spec = _make_spec()
        tok = create_tokenizer(spec)
        tok.train()
        data = torch.zeros(2, 10, 1)
        tok._accumulate_frequencies(data)
        result = tok.fit_from_running_stats()
        assert result is True
        assert tok.is_fit.item()

    def test_fit_from_running_stats_is_idempotent(self):
        """fit_from_running_stats() must return False when called again after fitting."""
        tok = _make_fitted_tokenizer()
        result = tok.fit_from_running_stats()
        assert result is False

    def test_clears_pattern_counts_after_fit(self):
        """_pattern_counts must be cleared after fitting to free memory."""
        spec = _make_spec()
        tok = create_tokenizer(spec)
        tok.train()
        data = torch.zeros(2, 10, 1)
        tok._accumulate_frequencies(data)
        tok.fit_from_running_stats()
        assert len(tok._pattern_counts) == 0


# ---------------------------------------------------------------------------
# Design fix: Lightning optimizer compatibility (dummy parameter)
# ---------------------------------------------------------------------------


class TestDummyParameter:
    def test_has_at_least_one_parameter(self):
        """Tokenizer must have at least one nn.Parameter for Lightning's optimizer."""
        spec = _make_spec()
        tok = create_tokenizer(spec)
        assert len(list(tok.parameters())) > 0


# ---------------------------------------------------------------------------
# Round-trip correctness
# ---------------------------------------------------------------------------


class TestRoundtrip:
    def test_in_vocab_pattern_round_trips(self):
        """A pattern that was seen during training should reconstruct with low error."""
        tok = _make_fitted_tokenizer()
        x = torch.zeros(1, 10, 1)
        x[0, 0, 0] = 1.0  # spike-at-bin-0 pattern
        tokens = tok.encode(x)
        recon = tok.decode(tokens)
        assert recon.shape == (1, 10, 1)
        assert torch.allclose(recon, x, atol=0.01)

    def test_forward_output_shape(self):
        """forward() must return TokenizerOutput with correctly shaped tensors."""
        tok = _make_fitted_tokenizer()
        x = torch.zeros(3, 10, 1)
        output = tok(x)
        assert output.tokens is not None
        assert output.reconstruction is not None
        assert output.reconstruction.shape == (3, 10, 1)
        assert output.tokens.shape == (3, 1, 1, 1)


@pytest.mark.parametrize(
    "batch,time,units,window,neurons", [(1, 1, 1, 1, 1), (1, 1, 1, 3, 2), (2, 3, 2, 3, 2)]
)
def test_continuous_patterns_decode_perturbations(batch, time, units, window, neurons):
    """Decoding a perturbed pattern must move every value to its exact spike position.

    What this checks, step by step:

    1. The vocabulary table is filled with distinct numbers, so every pattern
       position is individually traceable through decode().
    2. Tokens are expanded to continuous patterns via tokens_to_latents(); those
       patterns must equal the raw table rows (no reordering on lookup).
    3. decode() must accept tokens and patterns interchangeably and give the
       same reconstruction for both.
    4. A *different* offset is added at every pattern position before decoding.
       decode() has to unpack the flat [B, T', U', window*neurons] pattern into
       (window, neurons) blocks and interleave the blocks back into [B, T, U]
       spikes; the test asserts each offset reappears at exactly one
       predictable spike position. A swapped permute/reshape order in decode()
       would place offsets at wrong positions and fail here.
    5. A freshly constructed tokenizer loaded from state_dict decodes the same
       patterns identically (vocab table survives saving).
    """
    tok = FrequencyBasedLUTTokenizer(
        _make_spec(
            window_length_seconds=window / 1000,
            n_neurons=neurons,
        )
    ).eval()
    tok.vocab_patterns.copy_(
        torch.arange(tok.vocab_patterns.numel()).reshape_as(tok.vocab_patterns)
    )
    tok.is_fit.fill_(True)
    tokens = torch.arange(batch * time * units).reshape(batch, time, units, 1) % tok.vocab_size
    latents = tok.tokens_to_latents(tokens)
    assert latents.is_floating_point()
    assert latents.shape == (batch, time, units, window * neurons)
    assert not tok.spec.has_latent_dim  # Native tokens are scalar IDs.
    torch.testing.assert_close(latents, tok.vocab_patterns[tokens[..., 0]])
    clean = tok.decode(tokens)
    torch.testing.assert_close(tok.decode(tokens[..., 0]), clean)
    torch.testing.assert_close(tok.decode(latents), clean)
    # A distinct perturbation at each block position catches incorrect unflattening.
    delta = torch.arange(latents.numel(), dtype=latents.dtype).reshape_as(latents) / 10 + 0.25
    expected_delta = (
        delta.reshape(batch, time, units, window, neurons)
        .permute(0, 1, 3, 2, 4)
        .reshape(batch, time * window, units * neurons)
    )
    torch.testing.assert_close(
        tok.reconstruct(latents + delta).reconstruction, clean + expected_delta
    )
    restored = FrequencyBasedLUTTokenizer(tok.spec)
    restored.load_state_dict(tok.state_dict(), strict=True)
    torch.testing.assert_close(restored.decode(latents), clean)
