"""Tests for WanNeural block construction and spec validation."""

from __future__ import annotations

import pytest
import torch.nn as nn
from pydantic import ValidationError

from ntt.contrib.tokenizer.wanneural.blocks import AttentionBlock, RMSNorm2d
from ntt.contrib.tokenizer.wanneural.wanneural import (
    WanNeuralTokenizer,
    WanNeuralTokenizerSpec,
)


def _base_spec(**overrides) -> dict:
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
    }
    defaults.update(overrides)
    return defaults


class TestMMDVAEARValidation:
    """ar_weight > 0 must be rejected for all continuous bottlenecks (vae, mmd_vae,
    diffusion_prior)."""

    def test_mmd_vae_with_ar_weight_raises(self):
        with pytest.raises(ValidationError, match="ar_weight>0 requires a discrete"):
            WanNeuralTokenizerSpec(**_base_spec(bottleneck_type="mmd_vae", ar_weight=0.1))

    def test_mmd_vae_without_ar_weight_passes(self):
        # ar_weight=0 (default) should not raise
        spec = WanNeuralTokenizerSpec(**_base_spec(bottleneck_type="mmd_vae"))
        assert spec.bottleneck_type == "mmd_vae"

    def test_vae_with_ar_weight_still_raises(self):
        # Existing vae check must not regress
        with pytest.raises(ValidationError, match="ar_weight>0 requires a discrete"):
            WanNeuralTokenizerSpec(**_base_spec(bottleneck_type="vae", ar_weight=0.1))

    def test_diffusion_prior_with_ar_weight_still_raises(self):
        # diffusion_prior is in the exclusion list as well
        with pytest.raises(ValidationError, match="ar_weight>0 requires a discrete"):
            WanNeuralTokenizerSpec(**_base_spec(bottleneck_type="diffusion_prior", ar_weight=0.1))


class TestAttentionBlockFFNNorm:
    """norm2 (FFN pre-norm) must use the same norm family as norm (attn pre-norm)."""

    def test_ffn_norm2_is_groupnorm_when_norm_type_group(self):
        blk = AttentionBlock(
            channels=16,
            num_heads=2,
            use_ffn=True,
            norm_type="group",
            num_groups=8,
        )
        assert isinstance(blk.norm, nn.GroupNorm), (
            f"norm should be GroupNorm, got {type(blk.norm).__name__}"
        )
        assert isinstance(blk.norm2, nn.GroupNorm), (
            f"norm2 should be GroupNorm when norm_type='group', got {type(blk.norm2).__name__}"
        )

    def test_ffn_norm2_is_rmsnorm_when_norm_type_rms(self):
        blk = AttentionBlock(
            channels=16,
            num_heads=2,
            use_ffn=True,
            norm_type="rms",
        )
        assert isinstance(blk.norm2, RMSNorm2d), (
            f"norm2 should be RMSNorm2d when norm_type='rms', got {type(blk.norm2).__name__}"
        )


class TestTemporalMaskCaching:
    """_temporal_mask result must be identical across calls for the same (t, device)."""

    def test_same_object_returned_on_repeat_call(self):
        import torch

        blk = AttentionBlock(channels=8, num_heads=1, axis="temporal", block_size=4)
        device = torch.device("cpu")
        mask1 = blk._temporal_mask(12, device)
        mask2 = blk._temporal_mask(12, device)
        assert mask1 is mask2, "Expected the same Tensor object to be returned from cache"

    def test_different_t_produces_different_masks(self):
        import torch

        blk = AttentionBlock(channels=8, num_heads=1, axis="temporal", block_size=4)
        device = torch.device("cpu")
        mask8 = blk._temporal_mask(8, device)
        mask12 = blk._temporal_mask(12, device)
        assert mask8 is not mask12
        assert mask8.shape == (8, 8)
        assert mask12.shape == (12, 12)
