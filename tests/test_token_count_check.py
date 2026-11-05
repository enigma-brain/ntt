"""Tests for _dbg_check_token_count_via_bins_per_token: it must agree with
the token count that _strip_token_padding keeps, under the keep rule that was
applied (token end times, or center times when only those exist)."""

import pytest
import torch

from ntt.evaluation.utils.eval_collation_helpers import (
    _dbg_check_token_count_via_bins_per_token,
    _strip_token_padding,
)


class TestDbgCheckTokenCountViaBinsPerToken:
    """The sanity check must agree with what ``_strip_token_padding`` keeps.

    Stripping keeps token ``i`` iff its end time ``start + (i + 1) * token_dur``
    is ``<= absolute_end + 1e-6``, i.e. only whole tokens. A truncated window
    whose leftover is >= half a bin must therefore not round up.
    """

    def test_truncated_window_10ms_bins_leftover_over_half_bin(self):
        """98.827 ms truncated window (seen in production in flint monkey_c_e4's test
        domain), 10 ms bins: stripping keeps 9 whole tokens, rounding expected 10."""
        start = 1234.5
        _dbg_check_token_count_via_bins_per_token(
            n_valid_tokens=9,
            absolute_start=start,
            absolute_end=start + 0.098827,
            token_bin_size_ms=10.0,
            sampling_rate=100.0,
        )

    def test_truncated_window_1ms_bins(self):
        start = 1234.5
        _dbg_check_token_count_via_bins_per_token(
            n_valid_tokens=98,
            absolute_start=start,
            absolute_end=start + 0.098827,
            token_bin_size_ms=1.0,
            sampling_rate=1000.0,
        )

    def test_truncated_window_multi_bin_tokens(self):
        """20 ms tokens over 1 ms bins: 98 whole bins -> 4 whole tokens."""
        start = 1234.5
        _dbg_check_token_count_via_bins_per_token(
            n_valid_tokens=4,
            absolute_start=start,
            absolute_end=start + 0.098827,
            token_bin_size_ms=20.0,
            sampling_rate=1000.0,
        )

    @pytest.mark.parametrize(
        ("token_bin_size_ms", "sampling_rate", "expected"),
        [(10.0, 100.0, 10), (1.0, 1000.0, 100), (20.0, 1000.0, 5)],
    )
    def test_exact_multiple_window(self, token_bin_size_ms, sampling_rate, expected):
        """Float error just below an exact multiple must not drop a token."""
        start = 1234.5
        end = start + 0.1  # (end - start) * rate is slightly below the integer
        _dbg_check_token_count_via_bins_per_token(
            n_valid_tokens=expected,
            absolute_start=start,
            absolute_end=end,
            token_bin_size_ms=token_bin_size_ms,
            sampling_rate=sampling_rate,
        )

    def test_genuine_mismatch_still_raises(self):
        start = 1234.5
        with pytest.raises(AssertionError, match="bins_per_token sanity check failed"):
            _dbg_check_token_count_via_bins_per_token(
                n_valid_tokens=8,
                absolute_start=start,
                absolute_end=start + 0.098827,
                token_bin_size_ms=10.0,
                sampling_rate=100.0,
            )

    @pytest.mark.parametrize("token_bin_size_ms", [1.0, 10.0])
    @pytest.mark.parametrize("duration_sec", [0.0915, 0.0949, 0.0951, 0.098827, 0.0999, 0.1])
    def test_agrees_with_strip_token_padding(self, token_bin_size_ms, duration_sec):
        """Count kept by stripping base-tokenizer timestamps must pass the check."""
        sampling_rate = 1000.0 / token_bin_size_ms
        start = 1234.5
        n_tokens = 100  # more than fit: the tail is padding to strip
        token_dur = token_bin_size_ms / 1000.0
        idx = torch.arange(n_tokens, dtype=torch.float64)
        starts = start + idx * token_dur
        ends = start + (idx + 1) * token_dur

        result = _strip_token_padding(
            latents=torch.randn(n_tokens, 4),
            token_start_times=starts,
            token_end_times=ends,
            absolute_end=start + duration_sec,
        )
        _dbg_check_token_count_via_bins_per_token(
            n_valid_tokens=result["latents"].shape[0],
            absolute_start=start,
            absolute_end=start + duration_sec,
            token_bin_size_ms=token_bin_size_ms,
            sampling_rate=sampling_rate,
        )


class TestDbgCheckCenterTimesOnly:
    """Tokenizers that report only center times (e.g. the perceiver's
    reconstruct_from_input) are stripped on centers: token ``i`` is kept iff
    ``(i + 0.5) * token_dur <= duration + 1e-6``, so a trailing partial token
    whose center fits is kept."""

    @staticmethod
    def _kept_on_centers(duration_sec: float, token_ms: float) -> int:
        start = 1234.5678
        token_s = token_ms / 1000.0
        n_tokens = round(0.1 / token_s)
        centers = start + (torch.arange(n_tokens, dtype=torch.float64) + 0.5) * token_s
        result = _strip_token_padding(
            latents=torch.zeros(n_tokens, 3, 1),
            token_start_times=centers,
            token_end_times=centers,
            absolute_end=start + duration_sec,
        )
        return result["latents"].shape[0]

    @pytest.mark.parametrize("token_ms", [1.0, 10.0])
    @pytest.mark.parametrize("duration_sec", [0.098827, 0.0949, 0.0951, 0.0995, 0.1])
    def test_agrees_with_center_time_stripping(self, token_ms, duration_sec):
        start = 1234.5678
        _dbg_check_token_count_via_bins_per_token(
            n_valid_tokens=self._kept_on_centers(duration_sec, token_ms),
            absolute_start=start,
            absolute_end=start + duration_sec,
            token_bin_size_ms=token_ms,
            sampling_rate=1000.0 / token_ms,
            stripped_on_center_times=True,
        )

    def test_center_rule_keeps_partial_token(self):
        """98.827 ms at 10 ms tokens: the 10th token's center (95 ms) fits."""
        assert self._kept_on_centers(0.098827, 10.0) == 10
        start = 1234.5678
        _dbg_check_token_count_via_bins_per_token(
            n_valid_tokens=10,
            absolute_start=start,
            absolute_end=start + 0.098827,
            token_bin_size_ms=10.0,
            sampling_rate=100.0,
            stripped_on_center_times=True,
        )
        with pytest.raises(AssertionError, match="bins_per_token sanity check failed"):
            _dbg_check_token_count_via_bins_per_token(
                n_valid_tokens=9,
                absolute_start=start,
                absolute_end=start + 0.098827,
                token_bin_size_ms=10.0,
                sampling_rate=100.0,
                stripped_on_center_times=True,
            )
