"""Tests for independent encoder/decoder architecture configuration."""

from __future__ import annotations

import pytest

from ntt.contrib.tokenizer.wanneural.wanneural import (
    WanNeuralTokenizer,
    WanNeuralTokenizerSpec,
)
from tests.fixtures import make_dummy_binned_input


def _base_kwargs(**overrides) -> dict:
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
        "token_bin_size_ms": 4.0,
        "bottleneck_type": "vae",
    }
    defaults.update(overrides)
    return defaults


class TestDefaultsMirrorEncoder:
    """With no dec_* overrides, effective values match encoder."""

    def test_all_effective_values_inherit_encoder(self):
        spec = WanNeuralTokenizerSpec(**_base_kwargs())
        assert spec.effective_dec_dim == spec.dim
        assert spec.effective_dec_dim_mult == spec.dim_mult
        assert spec.effective_dec_num_levels == spec.num_levels
        assert spec.effective_dec_num_res_blocks == spec.num_res_blocks
        assert spec.effective_dec_attn_levels == spec.attn_levels
        assert spec.effective_dec_num_attn_heads == spec.num_attn_heads
        assert spec.effective_dec_stage_shortcut == spec.stage_shortcut
        assert spec.effective_dec_mid_attn == spec.mid_attn
        assert spec.effective_dec_neuron_kernel_size == spec.neuron_kernel_size
        assert spec.effective_dec_k_t == spec.k_t
        assert spec.effective_dec_k_t_up == spec.k_t_down
        assert spec.effective_dec_s_t_up == spec.s_t_down
        assert spec.effective_dec_k_t_init == spec.k_t_init
        assert spec.effective_dec_s_t_init == spec.s_t_init
        assert spec.effective_dec_temporal_upsample == spec.temporal_downsample


class TestAsymmetricArchitectures:
    """Encoder and decoder can differ in every non-compression parameter."""

    def test_different_base_dim(self):
        spec = WanNeuralTokenizerSpec(**_base_kwargs(dec_dim=32))
        tok = WanNeuralTokenizer(spec)
        x = make_dummy_binned_input(n_neurons=8, time_bins=200, batch_size=2)
        out = tok.forward(x)
        assert out.reconstruction.shape == (2, 200, 8)

    def test_different_dim_mult_same_compression(self):
        # Encoder: 3 levels (1,2,4) with 2 downsamples -> compression 4
        # Decoder: 3 levels (1,3,5) with 2 upsamples -> compression 4 (same)
        spec = WanNeuralTokenizerSpec(
            **_base_kwargs(
                dec_dim_mult=(1, 3, 5),
                dec_temporal_upsample=(True, True),
            )
        )
        tok = WanNeuralTokenizer(spec)
        x = make_dummy_binned_input(n_neurons=8, time_bins=200, batch_size=2)
        out = tok.forward(x)
        assert out.reconstruction.shape == (2, 200, 8)

    def test_different_num_levels(self):
        # Encoder: 3 levels, 2 downsamples -> compression 4
        # Decoder: 4 levels, 2 upsamples -> compression 4 (different depth,
        # same total expansion).
        spec = WanNeuralTokenizerSpec(
            **_base_kwargs(
                dec_num_levels=4,
                dec_dim_mult=(1, 2, 4, 4),
                dec_temporal_upsample=(True, True, False),
            )
        )
        tok = WanNeuralTokenizer(spec)
        x = make_dummy_binned_input(n_neurons=8, time_bins=200, batch_size=2)
        out = tok.forward(x)
        assert out.reconstruction.shape == (2, 200, 8)

    def test_different_num_res_blocks(self):
        spec = WanNeuralTokenizerSpec(**_base_kwargs(num_res_blocks=1, dec_num_res_blocks=3))
        tok = WanNeuralTokenizer(spec)
        x = make_dummy_binned_input(n_neurons=8, time_bins=200, batch_size=2)
        out = tok.forward(x)
        assert out.reconstruction.shape == (2, 200, 8)

    def test_different_attn_levels(self):
        spec = WanNeuralTokenizerSpec(
            **_base_kwargs(attn_levels=(), dec_attn_levels=(1, 2), dec_num_attn_heads=2)
        )
        tok = WanNeuralTokenizer(spec)
        x = make_dummy_binned_input(n_neurons=8, time_bins=200, batch_size=2)
        out = tok.forward(x)
        assert out.reconstruction.shape == (2, 200, 8)

    def test_different_mid_attn_and_stage_shortcut(self):
        spec = WanNeuralTokenizerSpec(
            **_base_kwargs(
                mid_attn=True,
                stage_shortcut=True,
                dec_mid_attn=False,
                dec_stage_shortcut=False,
            )
        )
        tok = WanNeuralTokenizer(spec)
        x = make_dummy_binned_input(n_neurons=8, time_bins=200, batch_size=2)
        out = tok.forward(x)
        assert out.reconstruction.shape == (2, 200, 8)

    def test_different_kernel_sizes(self):
        spec = WanNeuralTokenizerSpec(
            **_base_kwargs(
                k_t=3,
                neuron_kernel_size=1,
                dec_k_t=5,
                dec_k_t_up=5,
                dec_neuron_kernel_size=3,
                dec_k_t_init=5,
            )
        )
        tok = WanNeuralTokenizer(spec)
        x = make_dummy_binned_input(n_neurons=8, time_bins=200, batch_size=2)
        out = tok.forward(x)
        assert out.reconstruction.shape == (2, 200, 8)


class TestCompressionMismatchRejected:
    """Validator rejects configs where decoder expansion != encoder compression."""

    def test_decoder_upsample_mismatch_rejected(self):
        # Encoder compresses by 4, decoder upsamples by only 2.
        with pytest.raises(ValueError, match="temporal expansion"):
            WanNeuralTokenizerSpec(**_base_kwargs(dec_temporal_upsample=(True, False)))

    def test_decoder_stride_mismatch_rejected(self):
        # Encoder compresses by 4 (stride 2 twice). Decoder strides 3 twice -> 9.
        with pytest.raises(ValueError, match="temporal expansion"):
            WanNeuralTokenizerSpec(**_base_kwargs(dec_s_t_up=3))

    def test_decoder_dim_mult_wrong_length_rejected(self):
        with pytest.raises(ValueError, match="dec_dim_mult"):
            WanNeuralTokenizerSpec(**_base_kwargs(dec_dim_mult=(1, 2), dec_num_levels=3))
