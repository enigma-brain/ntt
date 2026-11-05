"""Tests for _strip_input_padding and _drop_tokens_outside_bounds / _strip_token_padding."""

import numpy as np
import pytest
import torch

from ntt.evaluation.utils.validation_collation import (
    _drop_tokens_outside_bounds,
    _strip_input_padding,
    _strip_token_padding,
)


class TestStripInputPadding:
    """Tests for trailing padding stripping and interior gap detection."""

    def test_trailing_time_padding(self):
        """Last 10 time columns are all-False → stripped to (10, 40)."""
        n_units, n_time = 10, 50
        n_pad_time = 10
        original = torch.randn(n_units, n_time)
        reconstructed = torch.randn(n_units, n_time)
        mask = torch.ones(n_units, n_time, dtype=torch.bool)
        mask[:, n_time - n_pad_time :] = False

        orig_out, recon_out, mask_out = _strip_input_padding(original, reconstructed, mask)

        assert orig_out.shape == (n_units, n_time - n_pad_time)
        assert recon_out.shape == (n_units, n_time - n_pad_time)
        assert mask_out.shape == (n_units, n_time - n_pad_time)
        assert mask_out.all()
        # Content should match the unpadded portion
        torch.testing.assert_close(orig_out, original[:, :40])
        torch.testing.assert_close(recon_out, reconstructed[:, :40])

    def test_trailing_unit_padding(self):
        """Last 3 unit rows are all-False → stripped to (7, 50)."""
        n_units, n_time = 10, 50
        n_pad_units = 3
        original = torch.randn(n_units, n_time)
        reconstructed = torch.randn(n_units, n_time)
        mask = torch.ones(n_units, n_time, dtype=torch.bool)
        mask[n_units - n_pad_units :, :] = False

        orig_out, recon_out, mask_out = _strip_input_padding(original, reconstructed, mask)

        assert orig_out.shape == (n_units - n_pad_units, n_time)
        assert recon_out.shape == (n_units - n_pad_units, n_time)
        assert mask_out.shape == (n_units - n_pad_units, n_time)
        assert mask_out.all()
        torch.testing.assert_close(orig_out, original[:7, :])
        torch.testing.assert_close(recon_out, reconstructed[:7, :])

    def test_both_time_and_unit_padding(self):
        """Both trailing time and unit padding → both stripped."""
        n_units, n_time = 10, 50
        n_pad_units = 3
        n_pad_time = 10
        original = torch.randn(n_units, n_time)
        reconstructed = torch.randn(n_units, n_time)
        mask = torch.ones(n_units, n_time, dtype=torch.bool)
        mask[n_units - n_pad_units :, :] = False
        mask[:, n_time - n_pad_time :] = False

        orig_out, recon_out, mask_out = _strip_input_padding(original, reconstructed, mask)

        assert orig_out.shape == (n_units - n_pad_units, n_time - n_pad_time)
        assert recon_out.shape == (n_units - n_pad_units, n_time - n_pad_time)
        assert mask_out.shape == (n_units - n_pad_units, n_time - n_pad_time)
        assert mask_out.all()

    def test_no_padding(self):
        """All-True mask → returned unchanged, same shape."""
        n_units, n_time = 10, 50
        original = torch.randn(n_units, n_time)
        reconstructed = torch.randn(n_units, n_time)
        mask = torch.ones(n_units, n_time, dtype=torch.bool)

        orig_out, recon_out, mask_out = _strip_input_padding(original, reconstructed, mask)

        assert orig_out.shape == (n_units, n_time)
        assert recon_out.shape == (n_units, n_time)
        assert mask_out.shape == (n_units, n_time)
        assert mask_out.all()
        torch.testing.assert_close(orig_out, original)
        torch.testing.assert_close(recon_out, reconstructed)

    def test_interior_gap_warns(self):
        """Interior False values after stripping trailing padding → warning (not raise)."""
        n_units, n_time = 10, 50
        original = torch.randn(n_units, n_time)
        reconstructed = torch.randn(n_units, n_time)
        mask = torch.ones(n_units, n_time, dtype=torch.bool)
        # Add trailing padding (will be stripped)
        mask[:, 45:] = False
        # Add an interior gap (should trigger warning, not raise)
        mask[:, 20:23] = False

        # Should not raise; interior gaps now produce a logger.warning
        orig_out, recon_out, mask_out = _strip_input_padding(original, reconstructed, mask)
        # Trailing padding stripped, but interior gap still present in data
        assert orig_out.shape[1] == 45
        assert not mask_out.all()


class TestDropTokensOutsideBounds:
    """Tests for _drop_tokens_outside_bounds."""

    def _make_tokens(self, n_tokens: int = 50, token_bin_size: float = 0.02):
        """Create tokens with linearly spaced start/end times."""
        tokens = torch.randn(n_tokens, 8)  # [T, D]
        starts = torch.arange(n_tokens, dtype=torch.float64) * token_bin_size
        ends = starts + token_bin_size
        return tokens, starts, ends

    def test_upper_bound_strips_trailing_tokens(self):
        """Last 10 tokens have end_time > upper_bound → 40 remain."""
        tokens, starts, ends = self._make_tokens(50, 0.02)
        # tokens span [0, 1.0). Set upper_bound so last 10 are out.
        upper_bound = float(ends[39])  # 0.8

        trimmed, mask = _drop_tokens_outside_bounds(
            tokens,
            starts,
            ends,
            upper_bound=upper_bound,
        )

        assert mask.sum().item() == 40
        assert trimmed.shape[0] == 40
        assert mask[:40].all()
        assert not mask[40:].any()
        torch.testing.assert_close(trimmed, tokens[:40])

    def test_lower_bound_strips_leading_tokens(self):
        """First 5 tokens have start_time < lower_bound → 45 remain."""
        tokens, starts, ends = self._make_tokens(50, 0.02)
        lower_bound = float(starts[5])  # 0.1

        trimmed, mask = _drop_tokens_outside_bounds(
            tokens,
            starts,
            ends,
            lower_bound=lower_bound,
        )

        assert mask.sum().item() == 45
        assert trimmed.shape[0] == 45
        assert not mask[:5].any()
        assert mask[5:].all()
        torch.testing.assert_close(trimmed, tokens[5:])

    def test_both_bounds(self):
        """Both lower and upper bounds applied simultaneously."""
        tokens, starts, ends = self._make_tokens(50, 0.02)
        lower_bound = float(starts[5])  # drop first 5
        upper_bound = float(ends[39])  # drop last 10

        trimmed, mask = _drop_tokens_outside_bounds(
            tokens,
            starts,
            ends,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

        assert mask.sum().item() == 35
        assert trimmed.shape[0] == 35
        torch.testing.assert_close(trimmed, tokens[5:40])

    def test_no_filtering_needed(self):
        """All tokens within bounds → unchanged, all-True mask."""
        tokens, starts, ends = self._make_tokens(50, 0.02)

        trimmed, mask = _drop_tokens_outside_bounds(
            tokens,
            starts,
            ends,
            lower_bound=0.0,
            upper_bound=2.0,
        )

        assert mask.all()
        assert trimmed.shape == tokens.shape
        torch.testing.assert_close(trimmed, tokens)

    def test_no_bounds_returns_unchanged(self):
        """Both bounds None → returns input unchanged."""
        tokens, starts, ends = self._make_tokens(50, 0.02)

        trimmed, mask = _drop_tokens_outside_bounds(
            tokens,
            starts,
            ends,
        )

        assert mask.all()
        torch.testing.assert_close(trimmed, tokens)

    def test_batched_input_raises(self):
        """Batched input (3D tokens) raises ValueError."""
        tokens = torch.randn(2, 50, 8)  # [B, T, D] — batched
        starts = torch.arange(50, dtype=torch.float64) * 0.02
        ends = starts + 0.02

        with pytest.raises(ValueError, match="[Bb]atch"):
            _drop_tokens_outside_bounds(tokens, starts, ends, upper_bound=0.5)

    def test_float_tolerance_upper(self):
        """Upper bound with float precision: token exactly at boundary kept."""
        tokens, starts, ends = self._make_tokens(10, 0.02)
        # Set upper_bound to exactly match last token end time
        # With tolerance, this token should be kept
        upper_bound = float(ends[-1])

        trimmed, mask = _drop_tokens_outside_bounds(
            tokens,
            starts,
            ends,
            upper_bound=upper_bound,
        )

        assert mask.all(), "Token at exact boundary should be kept (tolerance)"

    def test_float_tolerance_lower(self):
        """Lower bound with float precision: token exactly at boundary kept."""
        tokens, starts, ends = self._make_tokens(10, 0.02)
        lower_bound = float(starts[0])

        trimmed, mask = _drop_tokens_outside_bounds(
            tokens,
            starts,
            ends,
            lower_bound=lower_bound,
        )

        assert mask.all(), "Token at exact boundary should be kept (tolerance)"


class TestStripTokenPadding:
    """Tests for _strip_token_padding wrapper."""

    def test_basic_stripping(self):
        """Latents + timestamps + behavioral dict → all trimmed consistently."""
        n_tokens = 50
        token_bin_size = 0.02
        n_valid = 40

        latents = torch.randn(n_tokens, 8)
        starts = torch.arange(n_tokens, dtype=torch.float64) * token_bin_size
        ends = starts + token_bin_size
        absolute_end = float(ends[n_valid - 1])  # keep first 40

        behavioral_data = {
            "velocity": np.random.randn(n_tokens, 2),
            "labels": np.random.randint(0, 5, size=(n_tokens,)),
        }

        result = _strip_token_padding(
            latents=latents,
            token_start_times=starts,
            token_end_times=ends,
            absolute_end=absolute_end,
            behavioral_data=behavioral_data,
        )

        assert result["latents"].shape[0] == n_valid
        assert result["token_start_times"].shape[0] == n_valid
        assert result["token_end_times"].shape[0] == n_valid
        assert result["behavioral_data"]["velocity"].shape[0] == n_valid
        assert result["behavioral_data"]["labels"].shape[0] == n_valid

        # Content preserved
        torch.testing.assert_close(result["latents"], latents[:n_valid])
        np.testing.assert_array_equal(
            result["behavioral_data"]["velocity"],
            behavioral_data["velocity"][:n_valid],
        )

    def test_no_behavioral_data(self):
        """Works without behavioral data."""
        n_tokens = 20
        latents = torch.randn(n_tokens, 4)
        starts = torch.arange(n_tokens, dtype=torch.float64) * 0.02
        ends = starts + 0.02

        result = _strip_token_padding(
            latents=latents,
            token_start_times=starts,
            token_end_times=ends,
            absolute_end=float(ends[-1]),
        )

        assert result["latents"].shape[0] == n_tokens
        assert result["behavioral_data"] is None

    def test_no_trimming_needed(self):
        """When all tokens are within bounds, nothing changes."""
        n_tokens = 20
        latents = torch.randn(n_tokens, 4)
        starts = torch.arange(n_tokens, dtype=torch.float64) * 0.02
        ends = starts + 0.02

        result = _strip_token_padding(
            latents=latents,
            token_start_times=starts,
            token_end_times=ends,
            absolute_end=2.0,  # well beyond all tokens
        )

        assert result["latents"].shape[0] == n_tokens
        torch.testing.assert_close(result["latents"], latents)
