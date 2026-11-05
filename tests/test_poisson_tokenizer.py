"""Tests for PoissonBaselineTokenizer and standalone reconstruction helper."""

import numpy as np
import torch

from ntt.contrib.tokenizer.poisson_baseline import (
    PoissonBaselineTokenizer,
    PoissonBaselineTokenizerSpec,
    poisson_baseline_reconstruction,
)
from ntt.core.components.types import BinnedSpikesInput


def _make_batch(spike_matrix: torch.Tensor, bin_size_ms: float = 1.0) -> BinnedSpikesInput:
    """Helper to wrap a (batch, time, units) tensor into BinnedSpikesInput."""
    if spike_matrix.ndim == 2:
        spike_matrix = spike_matrix.unsqueeze(0)
    B, T, U = spike_matrix.shape
    duration = T * bin_size_ms / 1000.0
    return BinnedSpikesInput(
        spike_matrix=spike_matrix,
        pad_and_gap_mask=torch.ones_like(spike_matrix, dtype=torch.int32),
        sampling_rate=1000.0 / bin_size_ms,
        unit_ids=torch.arange(U, dtype=torch.int32).unsqueeze(0).expand(B, -1),
        unit_ids_pad_mask=torch.ones(B, U, dtype=torch.int32),
        duration_sec=torch.full((B,), duration),
        session_id=tuple("test" for _ in range(B)),
        subject_id=tuple("test" for _ in range(B)),
        species=tuple("test" for _ in range(B)),
        absolute_start=torch.zeros(B, dtype=torch.float64),
        absolute_end=torch.full((B,), duration, dtype=torch.float64),
    )


class TestPoissonBaselineTokenizer:
    """Tests for the nn.Module tokenizer."""

    def test_spec_uses_rate_estimation_bin_as_token_bin_size(self):
        """Poisson token size should track the configured rate-estimation bin."""
        spec_10 = PoissonBaselineTokenizerSpec(
            rate_estimation_bin_ms=10.0,
            bin_size_ms=1.0,
        )
        spec_100 = PoissonBaselineTokenizerSpec(
            rate_estimation_bin_ms=100.0,
            bin_size_ms=1.0,
        )

        assert spec_10.token_bin_size_ms == 10.0
        assert spec_100.token_bin_size_ms == 100.0

    def test_encode_decode_roundtrip_preserves_rate(self):
        """Encode→decode should give expected counts that match input rate."""
        spec = PoissonBaselineTokenizerSpec(rate_estimation_bin_ms=10.0, bin_size_ms=1.0)
        tok = PoissonBaselineTokenizer(spec)

        # Constant rate: 1 spike per 10ms = 100 Hz
        spikes = torch.zeros(1, 100, 5)
        for t in range(0, 100, 10):
            spikes[0, t, :] = 1
        batch = _make_batch(spikes)

        z = tok.encode(batch)  # rates in Hz
        assert z.shape == (1, 10, 5, 1)  # 100 time bins / 10 ratio, [B, T', U, D=1]
        # Each rate bin has 1 spike / 10ms = 100 Hz
        assert torch.allclose(z, torch.full_like(z, 100.0))

        recon = tok.decode(z)
        assert recon.shape == (1, 100, 5)
        # Expected count per fine bin: 100 Hz * 0.001 s = 0.1
        assert torch.allclose(recon, torch.full_like(recon, 0.1))

    def test_forward_returns_tokenizer_output(self):
        """forward() should return TokenizerOutput with correct shapes."""
        spec = PoissonBaselineTokenizerSpec(rate_estimation_bin_ms=5.0, bin_size_ms=1.0)
        tok = PoissonBaselineTokenizer(spec)
        spikes = torch.zeros(2, 50, 3)
        spikes[:, ::5, :] = 1
        batch = _make_batch(spikes)

        out = tok.forward(batch)
        assert out.tokens.shape == (2, 10, 3, 1)  # 50/5, [B, T', U, D=1]
        assert out.reconstruction.shape == (2, 50, 3)
        assert out.aux_loss is None

    def test_reconstruct_produces_integer_counts(self):
        """reconstruct() should produce non-negative integer spike counts."""
        spec = PoissonBaselineTokenizerSpec(rate_estimation_bin_ms=10.0, bin_size_ms=1.0, seed=123)
        tok = PoissonBaselineTokenizer(spec)
        spikes = torch.zeros(1, 100, 4)
        spikes[:, ::5, :] = 1  # 200 Hz
        batch = _make_batch(spikes)

        z = tok.encode(batch)
        recon_out = tok.reconstruct(z)
        sampled = recon_out.reconstruction
        assert (sampled >= 0).all()
        assert torch.all(sampled == sampled.floor())

    def test_seed_reproducibility(self):
        """Same seed → the same sequence of reconstructions from a fresh tokenizer."""
        spec = PoissonBaselineTokenizerSpec(rate_estimation_bin_ms=10.0, bin_size_ms=1.0, seed=42)
        spikes = torch.zeros(1, 100, 3)
        spikes[:, ::3, :] = 1
        batch = _make_batch(spikes)

        def two_draws():
            tok = PoissonBaselineTokenizer(spec)
            z = tok.encode(batch)
            return tok.reconstruct(z).reconstruction, tok.reconstruct(z).reconstruction

        (a1, a2), (b1, b2) = two_draws(), two_draws()
        assert torch.equal(a1, b1) and torch.equal(a2, b2)

    def test_successive_reconstructions_draw_new_noise(self):
        """Each call (one per batch in an eval) continues the random stream; reusing one
        stream for every batch makes the noise add up across batches instead of averaging."""
        spec = PoissonBaselineTokenizerSpec(rate_estimation_bin_ms=10.0, bin_size_ms=1.0, seed=42)
        tok = PoissonBaselineTokenizer(spec)
        z = torch.full((32, 10, 20, 1), 10.0)  # 32 windows of 0.1 s, 20 units at 10 Hz
        r1 = tok.reconstruct(z).reconstruction
        r2 = tok.reconstruct(z).reconstruction
        assert not torch.equal(r1, r2)

    def test_rate_error_over_many_batches_is_sampling_noise(self):
        """600 s at 10 Hz scored as total counts / duration, batch by batch (as in an eval)."""
        spec = PoissonBaselineTokenizerSpec(rate_estimation_bin_ms=10.0, bin_size_ms=1.0, seed=42)
        tok = PoissonBaselineTokenizer(spec)
        n_units, rate, seconds = 50, 10.0, 600.0
        z = torch.full((32, 10, n_units, 1), rate)  # one batch = 32 windows of 0.1 s
        total = torch.zeros(n_units)
        for _ in range(int(seconds / 3.2)):
            total += tok.reconstruct(z).reconstruction.sum(dim=(0, 1))
        mean_abs_error = (total / seconds - rate).abs().mean().item()
        independent = 0.8 * np.sqrt(rate / seconds)  # E|N(0, rate/T)| = sqrt(2/pi) * sd
        assert mean_abs_error < 2 * independent

    def test_variable_n_units(self):
        """Tokenizer should handle different numbers of units."""
        spec = PoissonBaselineTokenizerSpec(rate_estimation_bin_ms=10.0, bin_size_ms=1.0)
        tok = PoissonBaselineTokenizer(spec)

        for n_units in [1, 5, 50]:
            spikes = torch.zeros(1, 100, n_units)
            spikes[:, ::10, :] = 1
            batch = _make_batch(spikes)
            out = tok.forward(batch)
            assert out.reconstruction.shape == (1, 100, n_units)

    def test_invalid_ratio_raises(self):
        """Non-integer ratio should raise ValueError."""
        import pytest

        with pytest.raises(ValueError, match="integer multiple"):
            spec = PoissonBaselineTokenizerSpec(rate_estimation_bin_ms=7.0, bin_size_ms=3.0)
            PoissonBaselineTokenizer(spec)

    def test_reconstruct_from_input_preserves_odd_length_10ms(self):
        """Module-path reconstruction should pad back to the input length."""
        spec = PoissonBaselineTokenizerSpec(
            rate_estimation_bin_ms=10.0,
            bin_size_ms=1.0,
            seed=123,
        )
        tok = PoissonBaselineTokenizer(spec)
        spikes = torch.zeros(1, 997, 3)
        batch = _make_batch(spikes)

        out = tok.reconstruct_from_input(batch, return_latents=True)

        assert out["latents"].shape == (1, 99, 3, 1)
        assert out["reconstruction"].shape == (1, 997, 3)
        assert torch.equal(out["reconstruction"][:, 990:, :], torch.zeros(1, 7, 3))

    def test_match_input_length_within_tolerance_pads(self):
        """Mismatch within one rate-estimation bin should pad silently."""
        spec = PoissonBaselineTokenizerSpec(rate_estimation_bin_ms=10.0, bin_size_ms=1.0)
        tok = PoissonBaselineTokenizer(spec)
        # tol_bins = round(rate_estimation_bin_ms / bin_size_ms) = round(10.0 / 1.0) = 10
        tol_bins = 10
        recon = torch.zeros(1, 990, 5)
        out = tok._match_input_length(recon, target_time_bins=990 + tol_bins)
        assert out.shape[1] == 990 + tol_bins

    def test_match_input_length_beyond_tolerance_raises(self):
        """Mismatch beyond one rate-estimation bin should raise rather than fabricate."""
        import pytest

        spec = PoissonBaselineTokenizerSpec(rate_estimation_bin_ms=10.0, bin_size_ms=1.0)
        tok = PoissonBaselineTokenizer(spec)
        # tol_bins = round(10.0 / 1.0) = 10; 997 - 500 = 497 >> 10
        recon = torch.zeros(1, 500, 5)
        with pytest.raises(ValueError, match="differs from input length"):
            tok._match_input_length(recon, target_time_bins=997)

    def test_reconstruct_from_input_preserves_odd_length_100ms(self):
        """The same padding contract should hold for very coarse Poisson bins."""
        spec = PoissonBaselineTokenizerSpec(
            rate_estimation_bin_ms=100.0,
            bin_size_ms=1.0,
            seed=123,
        )
        tok = PoissonBaselineTokenizer(spec)
        spikes = torch.zeros(1, 998, 2)
        batch = _make_batch(spikes)

        out = tok.reconstruct_from_input(batch, return_latents=False)

        assert out.reconstruction.shape == (1, 998, 2)
        assert torch.equal(out.reconstruction[:, 900:, :], torch.zeros(1, 98, 2))


class TestPoissonBaselineReconstruction:
    """Tests for the standalone numpy helper."""

    def test_shape_preserved(self):
        """Output should have same shape as input."""
        spike_counts = np.zeros((5, 100), dtype=np.int32)
        spike_counts[:, ::10] = 1
        result = poisson_baseline_reconstruction(
            spike_counts, bin_size_ms=1.0, rate_estimation_bin_ms=10.0, seed=42
        )
        assert result.shape == spike_counts.shape

    def test_output_dtype_matches_input(self):
        """Output dtype should match input dtype."""
        spike_counts = np.zeros((3, 100), dtype=np.int32)
        spike_counts[:, ::10] = 1
        result = poisson_baseline_reconstruction(
            spike_counts, bin_size_ms=1.0, rate_estimation_bin_ms=10.0, seed=42
        )
        assert result.dtype == spike_counts.dtype

    def test_nonnegative_output(self):
        """Poisson samples are always non-negative."""
        spike_counts = np.zeros((4, 200), dtype=np.int32)
        spike_counts[:, ::5] = 2
        result = poisson_baseline_reconstruction(
            spike_counts, bin_size_ms=1.0, rate_estimation_bin_ms=10.0, seed=42
        )
        assert (result >= 0).all()

    def test_seed_reproducibility(self):
        """Same seed → same output."""
        spike_counts = np.zeros((3, 100), dtype=np.int32)
        spike_counts[:, ::5] = 1
        r1 = poisson_baseline_reconstruction(
            spike_counts, bin_size_ms=1.0, rate_estimation_bin_ms=10.0, seed=42
        )
        r2 = poisson_baseline_reconstruction(
            spike_counts, bin_size_ms=1.0, rate_estimation_bin_ms=10.0, seed=42
        )
        np.testing.assert_array_equal(r1, r2)

    def test_different_seeds_differ(self):
        """Different seeds → different output (with high probability)."""
        spike_counts = np.zeros((3, 1000), dtype=np.int32)
        spike_counts[:, ::2] = 1  # High rate for obvious differences
        r1 = poisson_baseline_reconstruction(
            spike_counts, bin_size_ms=1.0, rate_estimation_bin_ms=10.0, seed=1
        )
        r2 = poisson_baseline_reconstruction(
            spike_counts, bin_size_ms=1.0, rate_estimation_bin_ms=10.0, seed=2
        )
        assert not np.array_equal(r1, r2)

    def test_invalid_ratio_raises(self):
        """Non-integer ratio should raise ValueError."""
        import pytest

        spike_counts = np.zeros((2, 100), dtype=np.int32)
        with pytest.raises(ValueError, match="integer multiple"):
            poisson_baseline_reconstruction(
                spike_counts, bin_size_ms=3.0, rate_estimation_bin_ms=7.0
            )

    def test_rate_preservation(self):
        """Mean output rate should approximately match input rate (large sample)."""
        rng = np.random.default_rng(0)
        # High rate for tight convergence
        spike_counts = rng.poisson(0.5, size=(10, 10000)).astype(np.int32)
        result = poisson_baseline_reconstruction(
            spike_counts, bin_size_ms=1.0, rate_estimation_bin_ms=10.0, seed=42
        )
        # Mean rate should be close (within ~5% for this many samples)
        np.testing.assert_allclose(result.mean(), spike_counts.mean(), rtol=0.05)
