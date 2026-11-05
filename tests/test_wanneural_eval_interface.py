"""Tests that the WanNeural tokenizer fits the eval pipeline interface."""

import pytest
import torch

from ntt.contrib.tokenizer.wanneural.wanneural import (
    WanNeuralTokenizer,
    WanNeuralTokenizerSpec,
)
from ntt.core.components.types import (
    DataType,
    ReconstructionOutput,
    TokenizerOutput,
)
from tests.fixtures import make_dummy_binned_input


def _make_spec(**overrides) -> WanNeuralTokenizerSpec:
    """Create a minimal WanNeural spec for testing."""
    defaults = {
        "n_neurons": 8,
        "window_length_seconds": 0.2,
        "bin_size_ms": 1.0,
        "dim": 16,
        "z_dim": 4,
        "num_levels": 3,
        "dim_mult": (1, 2, 4),
        "num_res_blocks": 1,
        "attn_levels": (),
        "num_attn_heads": 1,
        "temporal_patch_size": 1,
        "temporal_downsample": (True, True),
        "token_bin_size_ms": 4.0,  # 200 bins / (2*2) compression = 50 steps, 200ms/50 = 4ms
    }
    defaults.update(overrides)
    return WanNeuralTokenizerSpec(**defaults)


class TestEncodeShape:
    """encode() must return [B, T, U, D] 4D tensor."""

    def test_fsq_encode_returns_4d(self):
        spec = _make_spec(bottleneck_type="fsq", q_levels=(8, 5, 5, 5))
        tok = WanNeuralTokenizer(spec)
        x = make_dummy_binned_input(n_neurons=8, time_bins=200, batch_size=2)
        z = tok.encode(x)
        assert z.ndim == 4, f"Expected 4D, got {z.ndim}D: {z.shape}"
        B, T, U, D = z.shape
        assert B == 2
        assert T == spec.num_latent_time_steps
        assert U == spec.n_neurons
        assert D == spec.z_dim

    def test_vae_encode_returns_4d(self):
        spec = _make_spec(bottleneck_type="vae")
        tok = WanNeuralTokenizer(spec)
        x = make_dummy_binned_input(n_neurons=8, time_bins=200, batch_size=2)
        z = tok.encode(x)
        assert z.ndim == 4, f"Expected 4D, got {z.ndim}D: {z.shape}"
        B, T, U, D = z.shape
        assert B == 2
        assert T == spec.num_latent_time_steps
        assert U == spec.n_neurons
        assert D == spec.z_dim

    def test_bsq_encode_returns_4d(self):
        spec = _make_spec(bottleneck_type="bsq", z_dim=8)
        tok = WanNeuralTokenizer(spec)
        x = make_dummy_binned_input(n_neurons=8, time_bins=200, batch_size=2)
        z = tok.encode(x)
        assert z.ndim == 4, f"Expected 4D, got {z.ndim}D: {z.shape}"
        B, T, U, D = z.shape
        assert B == 2
        assert T == spec.num_latent_time_steps
        assert U == spec.n_neurons
        assert D == spec.z_dim


class TestDecodeRoundTrip:
    """decode() must accept [B, T, U, D] from encode() and return [B, T, N]."""

    @pytest.mark.parametrize("bottleneck_type", ["fsq", "vae", "bsq"])
    def test_decode_accepts_4d(self, bottleneck_type):
        extra = {"z_dim": 8} if bottleneck_type == "bsq" else {}
        spec = _make_spec(bottleneck_type=bottleneck_type, **extra)
        tok = WanNeuralTokenizer(spec)
        x = make_dummy_binned_input(n_neurons=8, time_bins=200, batch_size=2)
        z = tok.encode(x)
        recon = tok.decode(z)
        assert recon.shape == (2, 200, 8), f"Expected (2, 200, 8), got {recon.shape}"


class TestReconstruct:
    """reconstruct() must accept 4D latents and return ReconstructionOutput."""

    @pytest.mark.parametrize("bottleneck_type", ["fsq", "vae", "bsq"])
    def test_reconstruct_returns_reconstruction_output(self, bottleneck_type):
        extra = {"z_dim": 8} if bottleneck_type == "bsq" else {}
        spec = _make_spec(bottleneck_type=bottleneck_type, **extra)
        tok = WanNeuralTokenizer(spec)
        x = make_dummy_binned_input(n_neurons=8, time_bins=200, batch_size=2)
        z = tok.encode(x)
        result = tok.reconstruct(z)
        assert isinstance(result, ReconstructionOutput)
        assert result.reconstruction.shape == (2, 200, 8)
        assert result.data_type == DataType.SPIKES
        assert result.bin_size_seconds == pytest.approx(0.001)


class TestForward:
    """forward() must return TokenizerOutput with properly shaped fields."""

    def test_fsq_forward_output(self):
        spec = _make_spec(bottleneck_type="fsq", q_levels=(8, 5, 5, 5))
        tok = WanNeuralTokenizer(spec)
        x = make_dummy_binned_input(n_neurons=8, time_bins=200, batch_size=2)
        out = tok.forward(x)
        assert isinstance(out, TokenizerOutput)
        # reconstruction: (B, T, N)
        assert out.reconstruction.shape == (2, 200, 8)
        # tokens: integer indices (B, T_comp * N) for discrete bottleneck
        assert out.tokens is not None
        assert out.tokens.shape == (2, spec.num_latent_time_steps * 8)
        assert out.tokens.dtype in (torch.int32, torch.int64, torch.long)
        # latents: continuous 4D (B, T, N, D)
        assert out.latents is not None
        assert out.latents.ndim == 4
        assert out.latents.shape == (2, spec.num_latent_time_steps, 8, spec.z_dim)

    def test_vae_forward_output(self):
        spec = _make_spec(bottleneck_type="vae")
        tok = WanNeuralTokenizer(spec)
        x = make_dummy_binned_input(n_neurons=8, time_bins=200, batch_size=2)
        out = tok.forward(x)
        assert isinstance(out, TokenizerOutput)
        assert out.reconstruction.shape == (2, 200, 8)
        # VAE: tokens field is None (no discrete tokens)
        assert out.tokens is None
        # latents: continuous 4D
        assert out.latents is not None
        assert out.latents.ndim == 4
        assert out.latents.shape == (2, spec.num_latent_time_steps, 8, spec.z_dim)
        # aux_loss: named dict with KL divergence
        assert out.aux_loss is not None
        assert isinstance(out.aux_loss, dict)
        assert "kl" in out.aux_loss


class TestReconstructFromInput:
    """reconstruct_from_input() is the eval pipeline entry point."""

    @pytest.mark.parametrize("bottleneck_type", ["fsq", "vae", "bsq"])
    def test_returns_dict_with_latents(self, bottleneck_type):
        extra = {"z_dim": 8} if bottleneck_type == "bsq" else {}
        spec = _make_spec(bottleneck_type=bottleneck_type, **extra)
        tok = WanNeuralTokenizer(spec)
        x = make_dummy_binned_input(n_neurons=8, time_bins=200, batch_size=2)
        result = tok.reconstruct_from_input(x, return_latents=True)

        assert isinstance(result, dict)
        assert "reconstruction" in result
        assert "latents" in result
        assert "token_timestamps" in result
        assert "token_start_time" in result
        assert "token_end_time" in result

        # Reconstruction shape
        assert result["reconstruction"].shape == (2, 200, 8)

        # Latents: 4D [B, T, U, D]
        latents = result["latents"]
        assert latents.ndim == 4
        T_comp = spec.num_latent_time_steps
        assert latents.shape == (2, T_comp, 8, spec.z_dim)

        # Timestamps: [B, T'] where T' is the time dimension of latents
        assert result["token_timestamps"].shape == (2, T_comp)
        assert result["token_start_time"].shape == (2, T_comp)
        assert result["token_end_time"].shape == (2, T_comp)

    @pytest.mark.parametrize("bottleneck_type", ["fsq", "vae", "bsq"])
    def test_returns_reconstruction_output_without_latents(self, bottleneck_type):
        extra = {"z_dim": 8} if bottleneck_type == "bsq" else {}
        spec = _make_spec(bottleneck_type=bottleneck_type, **extra)
        tok = WanNeuralTokenizer(spec)
        x = make_dummy_binned_input(n_neurons=8, time_bins=200, batch_size=2)
        result = tok.reconstruct_from_input(x, return_latents=False)
        assert isinstance(result, ReconstructionOutput)


class TestTimestamps:
    """Token timestamps must be absolute and consistent with token_bin_size_ms."""

    def test_timestamp_values(self):
        spec = _make_spec(bottleneck_type="fsq", token_bin_size_ms=4.0)
        tok = WanNeuralTokenizer(spec)
        x = make_dummy_binned_input(n_neurons=8, time_bins=200, batch_size=1)
        result = tok.reconstruct_from_input(x, return_latents=True)

        T_comp = spec.num_latent_time_steps
        ts = result["token_timestamps"]
        starts = result["token_start_time"]
        ends = result["token_end_time"]

        # absolute_start=0 for dummy input, so first token center = 0.5 * bin_size
        token_bin_sec = spec.token_bin_size_ms / 1000.0
        expected_centers = torch.arange(T_comp).double() * token_bin_sec + 0.5 * token_bin_sec
        torch.testing.assert_close(ts[0], expected_centers, atol=1e-10, rtol=0)

        # starts < centers < ends
        assert (starts < ts).all()
        assert (ts < ends).all()

        # ends - starts == token_bin_size_sec for all tokens
        torch.testing.assert_close(
            ends - starts,
            torch.full_like(ends, token_bin_sec),
            atol=1e-10,
            rtol=0,
        )
