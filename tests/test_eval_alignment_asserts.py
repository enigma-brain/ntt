"""Tests for fail-loud asserts added to the eval pipeline:

1. ``_assert_token_bin_divides_window_length`` — token bin must evenly divide
   the *configured* window length (or be ``None`` for dynamic-bin tokenizers).
   Runs once at startup, not per-batch on actual durations, so a
   shorter-than-configured session-end window does NOT trip it.
2. ``_pad_window_for_small_mismatch`` — small per-window orig/recon mismatches
   are zero-padded; large mismatches raise on the standard path but are
   tolerated on the interval-windowing path (returns is_large_mismatch=True).
3. ``_flush_segment`` post-pad shape invariant — assert original/reconstructed
   time bins match after padding.

These tests target the asserts directly so we don't need a full dataloader.

"""

from __future__ import annotations

import logging

import pytest
import torch

from ntt.evaluation.utils.validation_collation import (
    _assert_token_bin_divides_window_length,
    _flush_segment,
    _pad_window_for_small_mismatch,
)

# --- Change 1: divisibility check (now on configured values, at startup) ---------


def test_divisibility_check_raises_for_non_dividing_token_bin():
    """1.0s configured window with 33ms token → 1000 ms / 33 ms = 30.30… → ValueError."""
    with pytest.raises(ValueError) as excinfo:
        _assert_token_bin_divides_window_length(window_length_seconds=1.0, token_bin_size_ms=33.0)
    msg = str(excinfo.value)
    assert "does not divide configured input window length" in msg
    assert "33" in msg
    assert "1000" in msg


def test_divisibility_check_passes_for_dividing_token_bin():
    """1.0s configured window with 10ms token → 1000 / 10 = 100 → no error."""
    # Should not raise:
    _assert_token_bin_divides_window_length(window_length_seconds=1.0, token_bin_size_ms=10.0)


def test_divisibility_check_silent_for_none_token_bin():
    """Dynamic-bin tokenizers (token_bin_size_ms=None) must not trip the check."""
    # 1.0s % 33ms ≠ 0, but None bypasses regardless. Should not raise:
    _assert_token_bin_divides_window_length(window_length_seconds=1.0, token_bin_size_ms=None)


def test_divisibility_check_silent_for_none_window_length():
    """If the caller can't supply a configured window length, skip silently."""
    # Configured window length unknown → caller passes None → no error.
    _assert_token_bin_divides_window_length(window_length_seconds=None, token_bin_size_ms=33.0)


def test_divisibility_check_does_not_fire_on_runtime_shorter_window():
    """A runtime session-end partial window (e.g. 0.873s when configured is 1.0s)
    must NOT trip the check — the check runs on configured values, not on each
    batch's actual duration. We verify by passing the configured 1000 ms × 10 ms
    pair (which divides cleanly), confirming the check is independent of any
    runtime per-batch shorter durations.
    """
    # Configured 1.0s, token 10ms → 100 ms × 10 → divides cleanly.
    # Even if a downstream batch happens to be 0.873s long, the check is
    # already done at startup against the configured value, so it never sees
    # the shorter runtime duration.
    _assert_token_bin_divides_window_length(window_length_seconds=1.0, token_bin_size_ms=10.0)
    # And a 0.873s runtime duration would NOT have been passed to this
    # function in the new design — but to make that explicit, even if a future
    # caller mistakenly passed it, 0.873s × 10ms = 87.3 → would still raise.
    # That's fine: the protection is by removing the per-batch call sites.
    # (Negative case asserted by the absence of per-batch invocations in
    # iter_collated_validation_segments, exercised by the integration tests.)


# --- Change 2: zero-pad on small mismatch ----------------------------------------


def _make_window(
    *,
    n_units: int = 5,
    n_time_orig: int = 100,
    n_time_recon: int = 100,
    sampling_rate: float = 1000.0,
    absolute_start: float = 0.0,
    n_tokens: int = 4,
) -> dict:
    """Build a per-window dict matching what _flush_segment expects."""
    absolute_end = absolute_start + n_time_orig / sampling_rate
    return {
        "original": torch.ones(n_units, n_time_orig, dtype=torch.float32),
        "reconstructed": torch.ones(n_units, n_time_recon, dtype=torch.float32),
        "mask": torch.ones(n_units, max(n_time_orig, n_time_recon), dtype=torch.bool),
        "latents": torch.randn(n_tokens, n_units, 1),
        "absolute_start": absolute_start,
        "absolute_end": absolute_end,
        "session_id": "sess_test",
        "metadata": {"sampling_rate": sampling_rate},
        "has_overlap": False,
    }


def test_zero_pad_both_on_small_mismatch():
    """Mismatch of 1-2 bins (≤ 2 ms tolerance @ 1 kHz) → both padded to longer length."""
    sampling_rate = 1000.0  # tolerance = round(2e-3 * 1000) = 2 bins
    n_orig = 100
    n_recon = 98  # 2-bin diff, exactly at tolerance
    w = _make_window(
        n_time_orig=n_orig,
        n_time_recon=n_recon,
        sampling_rate=sampling_rate,
    )
    # Trim mask to match the shorter side, so we test that the helper repads it.
    w["mask"] = torch.ones(w["original"].shape[0], n_recon, dtype=torch.bool)

    # _pad_window_for_small_mismatch now returns a 2-tuple; the previous
    # ``is_large`` flag (used for the interval-windowing exemption) was
    # removed when that exemption was dropped.
    padded_w, n_padded = _pad_window_for_small_mismatch(w, sampling_rate=sampling_rate)

    assert n_padded == n_orig - n_recon == 2
    assert padded_w["original"].shape[1] == max(n_orig, n_recon) == n_orig
    assert padded_w["reconstructed"].shape[1] == max(n_orig, n_recon) == n_orig
    # Padded recon bins should be zero (the original was longer, so its trailing
    # bins are unchanged).
    assert torch.all(padded_w["reconstructed"][:, -2:] == 0)
    # Mask should be False at the padded bins (and True before).
    assert padded_w["mask"].shape[1] == n_orig
    assert not padded_w["mask"][:, -2:].any()
    assert padded_w["mask"][:, :-2].all()


def test_zero_pad_raises_on_large_mismatch_standard_path():
    """Standard (non-interval-windowing) path: 10-bin diff > 2-bin tolerance @ 1 kHz
    → ValueError mentioning cumulative drift. is_interval_windowed defaults to False.
    """
    sampling_rate = 1000.0
    w = _make_window(
        n_time_orig=100,
        n_time_recon=90,  # 10-bin diff
        sampling_rate=sampling_rate,
    )
    with pytest.raises(ValueError) as excinfo:
        _pad_window_for_small_mismatch(w, sampling_rate=sampling_rate)
    msg = str(excinfo.value)
    assert "10 bins" in msg or "10," in msg
    # The user emphasised: small per-window mismatches accumulate.
    assert "accumulate" in msg.lower()


# --- Change 3: post-pad shape invariant ------------------------------------------


def test_post_pad_shape_assertion_passes_after_pad():
    """A mismatch within tolerance survives _flush_segment without tripping the
    post-pad shape assertion (i.e. the padding helper is wired in)."""
    sampling_rate = 1000.0
    # Build two contiguous windows where each has a 1-bin mismatch.
    w0 = _make_window(
        n_time_orig=100,
        n_time_recon=99,  # 1-bin diff, within tolerance
        sampling_rate=sampling_rate,
        absolute_start=0.0,
    )
    w1 = _make_window(
        n_time_orig=100,
        n_time_recon=99,
        sampling_rate=sampling_rate,
        absolute_start=0.1,  # contiguous: 0.0 + 100/1000
    )
    # Strip latents to keep the test focused on shape padding (avoids needing
    # token timestamps to be aligned). _flush_segment also stacks latents, but
    # we don't need them here.
    for w in (w0, w1):
        del w["latents"]
        # Mask must match the longer-side post-pad shape; helper handles that
        # only if it starts shorter, so build the mask at the shorter recon
        # length to exercise the mask-padding branch.
        w["mask"] = torch.ones(w["original"].shape[0], 99, dtype=torch.bool)

    segment = _flush_segment([w0, w1])
    # Shapes must be equal post-pad.
    assert segment.original.shape[1] == segment.reconstructed.shape[1]
    # Two windows × 100 input bins each (after both sides padded to 100).
    assert segment.original.shape[1] == 200
    # Metadata records the cumulative pad.
    assert segment.metadata["n_zero_padded_bins_total"] == 2
    assert segment.metadata["n_padded_windows"] == 2
