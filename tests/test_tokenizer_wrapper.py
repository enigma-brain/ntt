"""Tests for encode_with_timestamps() method on BaseTokenizer."""

from typing import ClassVar, Literal

import pytest
import torch

from ntt.core.components.types import (
    BinnedSpikesInput,
    DataType,
    LatentType,
    TokenizerOutput,
)
from ntt.core.model.tokenizer.base import BaseTokenizer, TokenizerSpec

# ---------------------------------------------------------------------------
# Minimal concrete tokenizer for testing encode_with_timestamps
# ---------------------------------------------------------------------------


class _TestTokenizerSpec(TokenizerSpec):
    """Minimal spec for testing."""

    KIND: Literal["_test_encode_ts"] = "_test_encode_ts"
    latent_type: LatentType = LatentType.CONTINUOUS
    compress_n: bool = False
    compress_t: bool = False
    variable_n: bool = True
    variable_t: bool = True
    variable_dt: bool = True
    input_type: DataType = DataType.SPIKES
    output_type: DataType = DataType.SPIKES


class _TestTokenizer(BaseTokenizer):
    """Minimal BaseTokenizer subclass for testing encode_with_timestamps.

    encode() returns a tensor of specified shape (controlled by n_token_bins).
    decode() is a no-op. forward() is a no-op.
    """

    spec_class: ClassVar[type[_TestTokenizerSpec]] = _TestTokenizerSpec

    def __init__(self, spec: _TestTokenizerSpec, n_token_bins: int, n_units: int):
        super().__init__(spec)
        self._n_token_bins = n_token_bins
        self._n_units = n_units

    def encode(self, x):
        B = x.spike_matrix.shape[0] if x.spike_matrix.ndim == 3 else 1
        return torch.zeros(B, self._n_token_bins, self._n_units, 1)

    def decode(self, tokens):
        return tokens

    def forward(self, x):
        return TokenizerOutput(tokens=self.encode(x), reconstruction=x.spike_matrix, aux_loss=None)


class _TestTokenizerWithTimestamps(_TestTokenizer):
    """Like _TestTokenizer but encode() returns a dict with custom timestamps."""

    def __init__(self, spec, n_token_bins, n_units, custom_timestamps):
        super().__init__(spec, n_token_bins, n_units)
        self._custom_timestamps = custom_timestamps

    def encode(self, x):
        B = x.spike_matrix.shape[0] if x.spike_matrix.ndim == 3 else 1
        return {
            "latents": torch.zeros(B, self._n_token_bins, self._n_units, 1),
            "token_timestamps": self._custom_timestamps,
        }


def _make_batch(
    B: int,
    T: int,
    U: int,
    sampling_rate: float,
    absolute_starts: list[float],
) -> BinnedSpikesInput:
    """Create a minimal BinnedSpikesInput for testing."""
    assert len(absolute_starts) == B
    spike_matrix = torch.zeros(B, T, U, dtype=torch.int32)
    pad_and_gap_mask = torch.ones(B, T, U, dtype=torch.int32)
    abs_start = torch.tensor(absolute_starts, dtype=torch.float64)
    # padded_end = absolute_start + T / sampling_rate
    abs_end = abs_start + T / sampling_rate
    return BinnedSpikesInput(
        spike_matrix=spike_matrix,
        pad_and_gap_mask=pad_and_gap_mask,
        sampling_rate=sampling_rate,
        unit_ids=torch.arange(U).unsqueeze(0).expand(B, -1),
        unit_ids_pad_mask=torch.ones(B, U, dtype=torch.int32),
        duration_sec=torch.tensor([T / sampling_rate] * B),
        session_id=tuple(f"sess_{i}" for i in range(B)),
        subject_id=tuple(f"subj_{i}" for i in range(B)),
        species=tuple(["mouse"] * B),
        absolute_start=abs_start,
        absolute_end=abs_end,
    )


def _make_tokenizer(token_bin_size_ms: float, n_token_bins: int, n_units: int) -> _TestTokenizer:
    """Create a _TestTokenizer with the given token_bin_size_ms."""
    spec = _TestTokenizerSpec(
        window_length_seconds=1.0,  # dummy
        token_bin_size_ms=token_bin_size_ms,
    )
    return _TestTokenizer(spec, n_token_bins=n_token_bins, n_units=n_units)


class TestEncodeWithTimestampsComputed:
    """When tokenizer does NOT return timestamps, method computes them."""

    def test_center_start_end_times(self):
        """Verify center, start, end timestamps computed from spec + absolute time."""
        B, T, U = 1, 100, 5
        sampling_rate = 1000.0  # 1 ms bins
        token_bin_size_ms = 20.0  # each token covers 20 ms -> 5 tokens
        absolute_start = 10.0  # seconds

        batch = _make_batch(B, T, U, sampling_rate, [absolute_start])

        # Mock encode: returns [B, T', U, D] where T' = T / (token_bin_size_ms / (1000/sr))
        n_token_bins = int(T / (token_bin_size_ms / 1000.0 * sampling_rate))  # 100/20 = 5
        tokenizer = _make_tokenizer(token_bin_size_ms, n_token_bins, U)

        latents, meta = tokenizer.encode_with_timestamps(batch)

        assert latents.shape == (B, n_token_bins, U, 1)
        token_bin_size_sec = token_bin_size_ms / 1000.0

        # Check timestamps shape
        assert meta["token_timestamps"].shape == (B, n_token_bins)
        assert meta["token_start_time"].shape == (B, n_token_bins)
        assert meta["token_end_time"].shape == (B, n_token_bins)

        # Check values for first (only) batch element
        for i in range(n_token_bins):
            expected_center = absolute_start + (i + 0.5) * token_bin_size_sec
            expected_start = absolute_start + i * token_bin_size_sec
            expected_end = absolute_start + (i + 1) * token_bin_size_sec
            assert meta["token_timestamps"][0, i].item() == pytest.approx(
                expected_center, abs=1e-10
            )
            assert meta["token_start_time"][0, i].item() == pytest.approx(
                expected_start, abs=1e-10
            )
            assert meta["token_end_time"][0, i].item() == pytest.approx(expected_end, abs=1e-10)

    def test_float64_precision(self):
        """Timestamps must be float64."""
        batch = _make_batch(1, 100, 5, 1000.0, [10.0])
        tokenizer = _make_tokenizer(20.0, n_token_bins=5, n_units=5)

        _, meta = tokenizer.encode_with_timestamps(batch)
        assert meta["token_timestamps"].dtype == torch.float64
        assert meta["token_start_time"].dtype == torch.float64
        assert meta["token_end_time"].dtype == torch.float64


class TestEncodeWithTimestampsPassthrough:
    """When tokenizer DOES return timestamps, method passes them through."""

    def test_passthrough_existing_timestamps(self):
        """If encode returns a dict with token_timestamps, use them as-is."""
        B, T, U = 1, 100, 5
        token_bin_size_ms = 20.0
        n_token_bins = 5
        batch = _make_batch(B, T, U, 1000.0, [10.0])

        custom_timestamps = torch.arange(n_token_bins, dtype=torch.float64).unsqueeze(0) * 999.0

        spec = _TestTokenizerSpec(
            window_length_seconds=1.0,
            token_bin_size_ms=token_bin_size_ms,
        )
        tokenizer = _TestTokenizerWithTimestamps(
            spec,
            n_token_bins=n_token_bins,
            n_units=U,
            custom_timestamps=custom_timestamps,
        )

        latents, meta = tokenizer.encode_with_timestamps(batch)

        assert latents.shape == (B, n_token_bins, U, 1)
        assert torch.equal(meta["token_timestamps"], custom_timestamps)
        # When tokenizer provides timestamps, start/end are NOT computed
        assert "token_start_time" not in meta
        assert "token_end_time" not in meta


class TestEncodeWithTimestampsPaddedExtent:
    """Timestamps cover full padded extent, not just real data."""

    def test_timestamps_cover_padded_extent(self):
        """spike_matrix may be padded beyond real data; timestamps cover all of it."""
        B, U = 1, 5
        sampling_rate = 1000.0
        token_bin_size_ms = 10.0
        absolute_start = 5.0

        # spike_matrix has 100 time bins (100 ms), but let's say real data
        # was only 80 ms — doesn't matter, method uses spike_matrix shape.
        T = 100  # total padded time bins
        batch = _make_batch(B, T, U, sampling_rate, [absolute_start])

        n_token_bins = int(T / (token_bin_size_ms / 1000.0 * sampling_rate))  # 100/10 = 10
        tokenizer = _make_tokenizer(token_bin_size_ms, n_token_bins, U)

        _, meta = tokenizer.encode_with_timestamps(batch)

        padded_end = absolute_start + T / sampling_rate  # 5.0 + 0.1 = 5.1

        # Last token_end should reach padded_end
        assert meta["token_end_time"][0, -1].item() == pytest.approx(padded_end, abs=1e-10)
        # First token_start should be absolute_start
        assert meta["token_start_time"][0, 0].item() == pytest.approx(absolute_start, abs=1e-10)
        # Total number of tokens covers the full padded duration
        assert meta["token_timestamps"].shape[1] == n_token_bins


class TestEncodeWithTimestampsBatched:
    """Batched: B=2 with different absolute_start values."""

    def test_per_sample_timestamps_differ(self):
        """Different absolute_start → different timestamps per batch element."""
        B, T, U = 2, 60, 3
        sampling_rate = 1000.0
        token_bin_size_ms = 20.0
        starts = [10.0, 20.0]

        batch = _make_batch(B, T, U, sampling_rate, starts)

        n_token_bins = int(T / (token_bin_size_ms / 1000.0 * sampling_rate))  # 60/20 = 3
        tokenizer = _make_tokenizer(token_bin_size_ms, n_token_bins, U)

        _, meta = tokenizer.encode_with_timestamps(batch)

        token_bin_size_sec = token_bin_size_ms / 1000.0
        for b, start in enumerate(starts):
            for i in range(n_token_bins):
                expected_center = start + (i + 0.5) * token_bin_size_sec
                expected_start = start + i * token_bin_size_sec
                expected_end = start + (i + 1) * token_bin_size_sec
                assert meta["token_timestamps"][b, i].item() == pytest.approx(
                    expected_center, abs=1e-10
                )
                assert meta["token_start_time"][b, i].item() == pytest.approx(
                    expected_start, abs=1e-10
                )
                assert meta["token_end_time"][b, i].item() == pytest.approx(
                    expected_end, abs=1e-10
                )

        # Timestamps for batch 0 and batch 1 should differ
        assert not torch.equal(meta["token_timestamps"][0], meta["token_timestamps"][1])
