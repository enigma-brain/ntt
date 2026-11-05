"""End-to-end tests for token timestamp alignment through the validation pipeline.

Constructs small known datasets where every expected timestamp can be hand-computed,
runs them through iter_collated_validation_segments with a PoissonBaselineTokenizer (temporal
compression via rate binning), and verifies that token timestamps (center, start, end)
match expected absolute times at every stage.

These tests are overly extensive — the full pipeline tests (e.g.
test_full_pipeline_even_division) are supersets of the earlier unit-level tests
in the same class. For review, skip to eg the second test / generally the full
pipeline versions.

"""

from __future__ import annotations

import functools

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from torch_brain.data.dataset import DatasetIndex

from ntt.contrib.tokenizer.poisson_baseline import (
    PoissonBaselineTokenizer,
    PoissonBaselineTokenizerSpec,
)
from ntt.core.components.types import (
    BinnedSpikesInput,
)
from ntt.core.data.datamodule import collate
from ntt.evaluation.utils.validation_collation import (
    CollatedValidationSegment,
    _drop_tokens_outside_bounds,
    _flush_segment,
    _strip_token_padding,
    _trim_token_overlap,
    iter_collated_validation_segments,
)
from tests.fixtures import WindowedSpikeDataset

# ---------------------------------------------------------------------------
# PoissonBaselineTokenizer factory
# ---------------------------------------------------------------------------


def _make_poisson_tokenizer(
    token_bin_size_ms: float,
    sampling_rate: float,
) -> PoissonBaselineTokenizer:
    """Create a PoissonBaselineTokenizer with the given token bin size and sampling rate.

    The input bin_size_ms is derived from the sampling rate (1 / sampling_rate * 1000),
    and rate_estimation_bin_ms is set to the desired token_bin_size_ms, giving a compression
    factor of rate_estimation_bin_ms / bin_size_ms.

    """
    input_bin_size_ms = 1000.0 / sampling_rate
    spec = PoissonBaselineTokenizerSpec(
        window_length_seconds=1.0,  # dummy, not used at runtime
        bin_size_ms=input_bin_size_ms,
        rate_estimation_bin_ms=token_bin_size_ms,
    )
    return PoissonBaselineTokenizer(spec)


# ---------------------------------------------------------------------------
# Dataset helper for overlapping windows
# ---------------------------------------------------------------------------


class OverlappingWindowDataset(torch.utils.data.Dataset):
    """Dataset that returns windows from a spike tensor, indexed by DatasetIndex.

    Supports overlapping windows: the DatasetIndex specifies the absolute
    start and end time, and the dataset slices the spike tensor accordingly.
    """

    def __init__(
        self,
        spikes: torch.Tensor,
        sampling_rate: float,
        session_id: str = "test_session",
    ):
        """
        Args:
            spikes: Full spike tensor of shape (units, total_time_bins)
            sampling_rate: Sampling rate in Hz
            session_id: Session identifier
        """
        self.spikes = spikes
        self.sampling_rate = sampling_rate
        self.session_id = session_id

    def __len__(self) -> int:
        # Length is determined by the sampler, not by the dataset itself.
        # Return a large number; actual windowing is done via DatasetIndex.
        return 10000

    def __getitem__(self, index: DatasetIndex) -> BinnedSpikesInput:
        """Return a window of data based on the DatasetIndex timestamps."""
        assert isinstance(index, DatasetIndex), f"Expected DatasetIndex, got {type(index)}"

        start_bin = int(round(index.start * self.sampling_rate))
        end_bin = int(round(index.end * self.sampling_rate))

        # spikes is (units, time)
        window = self.spikes[:, start_bin:end_bin]  # (units, T_window)
        spike_matrix = window.T.int()  # (T_window, units)

        n_units = spike_matrix.shape[1]
        pad_and_gap_mask = torch.ones_like(spike_matrix, dtype=torch.int32)
        unit_ids = torch.arange(n_units, dtype=torch.int32)
        unit_ids_pad_mask = torch.ones(n_units, dtype=torch.int32)
        duration_sec = torch.tensor(
            (end_bin - start_bin) / self.sampling_rate, dtype=torch.float64
        )

        abs_start = torch.tensor(index.start, dtype=torch.float64)
        abs_end = torch.tensor(index.end, dtype=torch.float64)

        return BinnedSpikesInput(
            spike_matrix=spike_matrix,
            pad_and_gap_mask=pad_and_gap_mask,
            sampling_rate=self.sampling_rate,
            unit_ids=unit_ids,
            unit_ids_pad_mask=unit_ids_pad_mask,
            duration_sec=duration_sec,
            session_id=self.session_id,
            subject_id="test_subject",
            species="test_species",
            absolute_start=abs_start,
            absolute_end=abs_end,
        )


# ---------------------------------------------------------------------------
# Helper to make a DataLoader with specific window boundaries
# ---------------------------------------------------------------------------


def _make_dataloader(
    spikes: torch.Tensor,
    sampling_rate: float,
    windows: list[tuple[float, float]],
    session_id: str = "test_session",
    batch_size: int = 1,
) -> DataLoader:
    """Create a DataLoader from spike tensor with specific windows.
    To create overlap, need to specify overlapping windows.
    """
    dataset = OverlappingWindowDataset(spikes, sampling_rate, session_id)
    indices = [DatasetIndex(session_id, start, end) for start, end in windows]
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=indices,
        collate_fn=collate,
    )


def _make_tokenizer_fun(tokenizer: PoissonBaselineTokenizer):
    """Create a tokenizer_fun wrapping
    ``tokenizer.reconstruct_from_input(return_latents=True)``."""
    return functools.partial(tokenizer.reconstruct_from_input, return_latents=True)


# ===========================================================================
# Test Case A: Token bin divides evenly into window
# ===========================================================================


class TestTimestampAlignmentEvenDivision:
    """Case A: 250ms data at 1kHz, 50ms tokens, 100ms windows, 50ms overlap.

    Windows: [0-100ms], [50-150ms], [100-200ms], [150-250ms]
    Tokens per window: 100 / 50 = 2 tokens exactly

    Window 1 [0.0 - 0.1s]:
        Token 0: start=0.000, end=0.050, center=0.025
        Token 1: start=0.050, end=0.100, center=0.075
        -> No overlap trim (first window)
        -> Both tokens survive

    Window 2 [0.05 - 0.15s]:
        Token 0: start=0.050, end=0.100, center=0.075
        Token 1: start=0.100, end=0.150, center=0.125
        -> After overlap trim (prev_end=0.100): token 0 start < 0.100, dropped
        -> Token 1 survives: start=0.100, end=0.150

    Window 3 [0.10 - 0.20s]:
        Token 0: start=0.100, end=0.150, center=0.125
        Token 1: start=0.150, end=0.200, center=0.175
        -> After overlap trim (prev_end=0.150): token 0 start < 0.150, dropped
        -> Token 1 survives: start=0.150, end=0.200

    Window 4 [0.15 - 0.25s]:
        Token 0: start=0.150, end=0.200, center=0.175
        Token 1: start=0.200, end=0.250, center=0.225
        -> After overlap trim (prev_end=0.200): token 0 start < 0.200, dropped
        -> Token 1 survives: start=0.200, end=0.250

    After concatenation: 5 tokens covering [0.000, 0.250]s with no gaps.

    """

    SAMPLING_RATE = 1000.0  # Hz
    N_UNITS = 5
    TOTAL_BINS = 250  # 250ms
    TOKEN_BIN_SIZE_MS = 50.0
    WINDOW_SIZE_S = 0.100  # 100ms
    OVERLAP_S = 0.050  # 50ms

    def _make_windows(self) -> list[tuple[float, float]]:
        """Generate overlapping window boundaries."""
        total_duration = self.TOTAL_BINS / self.SAMPLING_RATE  # 0.250s
        step = self.WINDOW_SIZE_S - self.OVERLAP_S  # 0.050s
        windows = []
        start = 0.0
        while start + self.WINDOW_SIZE_S <= total_duration + 1e-9:
            windows.append((start, start + self.WINDOW_SIZE_S))
            start += step
        return windows

    def _make_pipeline_components(self):
        """Create spike data, tokenizer, and dataloader."""
        spikes = torch.zeros((self.N_UNITS, self.TOTAL_BINS))
        # Put a distinctive spike pattern so we can verify data integrity
        # Todo: How can this be verified with a rate tokenizer? We don't have good reconstruction.
        spikes[0, 25] = 1  # At 25ms
        spikes[1, 125] = 1  # At 125ms
        spikes[2, 225] = 1  # At 225ms

        tokenizer = _make_poisson_tokenizer(self.TOKEN_BIN_SIZE_MS, self.SAMPLING_RATE)

        windows = self._make_windows()
        loader = _make_dataloader(spikes, self.SAMPLING_RATE, windows, batch_size=1)

        return spikes, tokenizer, loader, windows

    def test_encode_with_timestamps_per_window(self):
        """Verify encode_with_timestamps returns correct timestamps per window."""
        spikes, tokenizer, loader, windows = self._make_pipeline_components()

        for i, batch in enumerate(loader):
            batch: BinnedSpikesInput
            latents, metadata = tokenizer.encode_with_timestamps(batch)

            window_start, window_end = windows[i]
            B = latents.shape[0]
            T_prime = latents.shape[1]

            assert B == 1
            assert T_prime == 2, f"Window {i}: expected 2 tokens, got {T_prime}"

            token_bin_sec = self.TOKEN_BIN_SIZE_MS / 1000.0

            expected_starts = torch.tensor(
                [[window_start + j * token_bin_sec for j in range(T_prime)]],
                dtype=torch.float64,
            )
            expected_ends = torch.tensor(
                [[window_start + (j + 1) * token_bin_sec for j in range(T_prime)]],
                dtype=torch.float64,
            )
            expected_centers = torch.tensor(
                [[window_start + (j + 0.5) * token_bin_sec for j in range(T_prime)]],
                dtype=torch.float64,
            )

            torch.testing.assert_close(
                metadata["token_start_time"],
                expected_starts,
                atol=1e-9,
                rtol=0,
                msg=f"Window {i}: token start times mismatch",
            )
            torch.testing.assert_close(
                metadata["token_end_time"],
                expected_ends,
                atol=1e-9,
                rtol=0,
                msg=f"Window {i}: token end times mismatch",
            )
            torch.testing.assert_close(
                metadata["token_timestamps"],
                expected_centers,
                atol=1e-9,
                rtol=0,
                msg=f"Window {i}: token center times mismatch",
            )

    def test_strip_token_padding_no_effect_even_division(self):
        """With even division, no tokens should be stripped by padding removal."""
        spikes, tokenizer, loader, windows = self._make_pipeline_components()

        for i, batch in enumerate(loader):
            latents, metadata = tokenizer.encode_with_timestamps(batch)

            # Per-window unbatch (batch_size=1, so idx=0)
            window_latents = latents[0]  # [T', U, D]
            window_starts = metadata["token_start_time"][0]  # [T']
            window_ends = metadata["token_end_time"][0]  # [T']

            _, window_end = windows[i]

            result = _strip_token_padding(
                window_latents,
                window_starts,
                window_ends,
                absolute_end=window_end,
            )

            # All tokens should survive — no padding needed for even division
            assert result["latents"].shape[0] == 2, (
                f"Window {i}: expected 2 tokens after strip, got {result['latents'].shape[0]}"
            )
            torch.testing.assert_close(
                result["token_start_times"],
                window_starts,
                atol=1e-9,
                rtol=0,
            )
            torch.testing.assert_close(
                result["token_end_times"],
                window_ends,
                atol=1e-9,
                rtol=0,
            )

    def test_overlap_trimming_removes_correct_tokens(self):
        """Verify that overlap trimming removes exactly the expected tokens.

        Note: this tests the same scenario as test_full_pipeline_even_division
        but at the level of individual functions (_strip_token_padding,
        _trim_token_overlap) rather than the full pipeline. The full pipeline
        test is strictly a superset.
        """
        spikes, tokenizer, loader, windows = self._make_pipeline_components()

        prev_last_token_end: float | None = None
        all_starts = []
        all_ends = []

        for i, batch in enumerate(loader):
            latents, metadata = tokenizer.encode_with_timestamps(batch)

            window_latents = latents[0]
            window_starts = metadata["token_start_time"][0]
            window_ends = metadata["token_end_time"][0]

            _, window_end = windows[i]

            # Strip token padding first (should be no-op for even division)
            strip_result = _strip_token_padding(
                window_latents,
                window_starts,
                window_ends,
                absolute_end=window_end,
            )

            # Trim overlap
            trim_result = _trim_token_overlap(
                latents=strip_result["latents"],
                token_start_times=strip_result["token_start_times"],
                token_end_times=strip_result["token_end_times"],
                prev_last_token_end_time=prev_last_token_end,
            )

            trimmed_starts = trim_result["token_start_times"]
            trimmed_ends = trim_result["token_end_times"]

            if i == 0:
                # First window: both tokens survive
                assert trimmed_starts.shape[0] == 2, (
                    f"Window 0: expected 2 tokens, got {trimmed_starts.shape[0]}"
                )
            else:
                # Subsequent windows: first token overlaps, only second survives
                assert trimmed_starts.shape[0] == 1, (
                    f"Window {i}: expected 1 token after overlap trim, "
                    f"got {trimmed_starts.shape[0]}"
                )

            all_starts.append(trimmed_starts)
            all_ends.append(trimmed_ends)

            if trimmed_ends.numel() > 0:
                prev_last_token_end = float(trimmed_ends[-1].item())

        # Concatenate all tokens
        concat_starts = torch.cat(all_starts, dim=0)
        concat_ends = torch.cat(all_ends, dim=0)

        # Should have exactly 5 tokens: 2 from window 0 + 1 each from windows 1-3
        assert concat_starts.shape[0] == 5, (
            f"Expected 5 total tokens, got {concat_starts.shape[0]}"
        )

        # Expected absolute timestamps
        expected_starts = torch.tensor([0.000, 0.050, 0.100, 0.150, 0.200], dtype=torch.float64)
        expected_ends = torch.tensor([0.050, 0.100, 0.150, 0.200, 0.250], dtype=torch.float64)

        torch.testing.assert_close(
            concat_starts,
            expected_starts,
            atol=1e-9,
            rtol=0,
            msg="Concatenated token start times do not match expected values",
        )
        torch.testing.assert_close(
            concat_ends,
            expected_ends,
            atol=1e-9,
            rtol=0,
            msg="Concatenated token end times do not match expected values",
        )

    def test_full_pipeline_even_division(self):
        """Full end-to-end test through iter_collated_validation_segments."""
        spikes, tokenizer, loader, windows = self._make_pipeline_components()

        segments = list(
            iter_collated_validation_segments(
                loader,
                _make_tokenizer_fun(tokenizer),
                token_bin_size_ms=self.TOKEN_BIN_SIZE_MS,
                device=torch.device("cpu"),
                trim_token_overlap=True,
                drop_overlapping_windows=False,
            )
        )

        assert len(segments) == 1, f"Expected 1 segment, got {len(segments)}"
        seg = segments[0]

        # Token timestamps must be present
        assert seg.token_start_times is not None, "token_start_times is None"
        assert seg.token_end_times is not None, "token_end_times is None"
        assert seg.token_timestamps is not None, "token_timestamps is None"

        # 5 tokens total
        n_tokens = seg.token_start_times.shape[0]
        assert n_tokens == 5, f"Expected 5 tokens, got {n_tokens}"
        assert seg.token_timestamps.shape[0] == 5

        # Exact expected timestamps
        expected_starts = torch.tensor([0.000, 0.050, 0.100, 0.150, 0.200], dtype=torch.float64)
        expected_ends = torch.tensor([0.050, 0.100, 0.150, 0.200, 0.250], dtype=torch.float64)
        expected_centers = (expected_starts + expected_ends) / 2.0

        torch.testing.assert_close(
            seg.token_start_times,
            expected_starts,
            atol=1e-9,
            rtol=0,
            msg="Full pipeline: token start times mismatch",
        )
        torch.testing.assert_close(
            seg.token_end_times,
            expected_ends,
            atol=1e-9,
            rtol=0,
            msg="Full pipeline: token end times mismatch",
        )
        torch.testing.assert_close(
            seg.token_timestamps,
            expected_centers,
            atol=1e-9,
            rtol=0,
            msg="Full pipeline: token center times mismatch",
        )

        # Monotonically increasing
        assert torch.all(seg.token_start_times[1:] > seg.token_start_times[:-1]), (
            "Token start times are not strictly monotonically increasing"
        )
        assert torch.all(seg.token_end_times[1:] > seg.token_end_times[:-1]), (
            "Token end times are not strictly monotonically increasing"
        )

        # No gaps: each token's end == next token's start
        gaps = seg.token_start_times[1:] - seg.token_end_times[:-1]
        torch.testing.assert_close(
            gaps,
            torch.zeros_like(gaps),
            atol=1e-9,
            rtol=0,
            msg="Gaps detected between consecutive tokens",
        )

        # Coverage: first token starts at 0.0, last ends at 0.250
        assert abs(float(seg.token_start_times[0]) - 0.0) < 1e-9
        assert abs(float(seg.token_end_times[-1]) - 0.250) < 1e-9

        # Latents should be stacked: 4 windows, but with variable token counts
        # after trimming, they get stacked per-window in _flush_segment.
        # Since window 0 has 2 tokens and windows 1-3 have 1 token each,
        # they can't be stacked uniformly — check that latents is present.
        assert seg.latents is not None, "Latents should be present"

    def test_full_pipeline_no_overlap(self):
        """Test with non-overlapping windows (simpler case, no trimming needed).

        Windows: [0-100ms], [100-200ms], [200-250ms (only 50ms)]
        Last window has only 50ms = 1 token (50ms token bins).

        """
        spikes = torch.zeros((self.N_UNITS, self.TOTAL_BINS))
        tokenizer = _make_poisson_tokenizer(self.TOKEN_BIN_SIZE_MS, self.SAMPLING_RATE)

        # Non-overlapping windows
        windows = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.25)]
        loader = _make_dataloader(spikes, self.SAMPLING_RATE, windows, batch_size=1)

        segments = list(
            iter_collated_validation_segments(
                loader,
                _make_tokenizer_fun(tokenizer),
                token_bin_size_ms=self.TOKEN_BIN_SIZE_MS,
                device=torch.device("cpu"),
            )
        )

        assert len(segments) == 1
        seg = segments[0]

        # Window 1: 2 tokens (0-50ms, 50-100ms)
        # Window 2: 2 tokens (100-150ms, 150-200ms)
        # Window 3: 1 token (200-250ms)
        # No overlap → no trimming → 5 tokens total
        assert seg.token_start_times.shape[0] == 5
        assert seg.token_timestamps is not None

        expected_starts = torch.tensor([0.000, 0.050, 0.100, 0.150, 0.200], dtype=torch.float64)
        expected_ends = torch.tensor([0.050, 0.100, 0.150, 0.200, 0.250], dtype=torch.float64)
        expected_centers = (expected_starts + expected_ends) / 2.0

        torch.testing.assert_close(
            seg.token_start_times,
            expected_starts,
            atol=1e-9,
            rtol=0,
        )
        torch.testing.assert_close(
            seg.token_end_times,
            expected_ends,
            atol=1e-9,
            rtol=0,
        )
        torch.testing.assert_close(
            seg.token_timestamps,
            expected_centers,
            atol=1e-9,
            rtol=0,
        )


# ===========================================================================
# Test Case B: Token bin does NOT divide evenly into window
# ===========================================================================


class TestTimestampAlignmentUnevenDivision:
    """Case B: 250ms data at 1kHz, 30ms tokens, 100ms windows, 50ms overlap.

    100ms window / 30ms token = 3 full tokens + 10ms remainder (dropped).
    Tokens per window cover [0, 90ms), leaving a 10ms gap at the end.

    Window 1 [0.0 - 0.1s]:
        Token 0: start=0.000, end=0.030
        Token 1: start=0.030, end=0.060
        Token 2: start=0.060, end=0.090
        -> Padding strip: all 3 end <= 0.100, all survive
        -> No overlap trim (first window)

    Window 2 [0.05 - 0.15s]:
        Token 0: start=0.050, end=0.080
        Token 1: start=0.080, end=0.110
        Token 2: start=0.110, end=0.140
        -> Padding strip: all 3 end <= 0.150, all survive
        -> Overlap trim (prev_end=0.090): token 0 start (0.050) < 0.090, dropped
                                           token 1 start (0.080) < 0.090, dropped
        -> Token 2 survives: start=0.110, end=0.140

    Window 3 [0.10 - 0.20s]:
        Token 0: start=0.100, end=0.130
        Token 1: start=0.130, end=0.160
        Token 2: start=0.160, end=0.190
        -> Padding strip: all 3 end <= 0.200, all survive
        -> Overlap trim (prev_end=0.140): token 0 start (0.100) < 0.140, dropped
                                           token 1 start (0.130) < 0.140, dropped
        -> Token 2 survives: start=0.160, end=0.190

    Window 4 [0.15 - 0.25s]:
        Token 0: start=0.150, end=0.180
        Token 1: start=0.180, end=0.210
        Token 2: start=0.210, end=0.240
        -> Padding strip: all 3 end <= 0.250, all survive
        -> Overlap trim (prev_end=0.190): token 0 start (0.150) < 0.190, dropped
                                           token 1 start (0.180) < 0.190, dropped
        -> Token 2 survives: start=0.210, end=0.240

    After concatenation: tokens at [0.000, 0.030, 0.060, 0.110, 0.160, 0.210]
    with gaps at [0.090-0.110], [0.140-0.160], [0.190-0.210] due to remainder bins.

    """

    SAMPLING_RATE = 1000.0
    N_UNITS = 5
    TOTAL_BINS = 250
    TOKEN_BIN_SIZE_MS = 30.0
    WINDOW_SIZE_S = 0.100
    OVERLAP_S = 0.050

    def _make_windows(self) -> list[tuple[float, float]]:
        """Generate overlapping window boundaries."""
        total_duration = self.TOTAL_BINS / self.SAMPLING_RATE
        step = self.WINDOW_SIZE_S - self.OVERLAP_S
        windows = []
        start = 0.0
        while start + self.WINDOW_SIZE_S <= total_duration + 1e-9:
            windows.append((start, start + self.WINDOW_SIZE_S))
            start += step
        return windows

    def _make_pipeline_components(self):
        """Create spike data, tokenizer, and dataloader."""
        spikes = torch.zeros((self.N_UNITS, self.TOTAL_BINS))
        tokenizer = _make_poisson_tokenizer(self.TOKEN_BIN_SIZE_MS, self.SAMPLING_RATE)
        windows = self._make_windows()
        loader = _make_dataloader(spikes, self.SAMPLING_RATE, windows, batch_size=1)
        return spikes, tokenizer, loader, windows

    def test_tokens_per_window_uneven(self):
        """Each 100ms window with 30ms tokens should produce exactly 3 tokens."""
        spikes, tokenizer, loader, windows = self._make_pipeline_components()

        for i, batch in enumerate(loader):
            latents, metadata = tokenizer.encode_with_timestamps(batch)

            T_prime = latents.shape[1]
            assert T_prime == 3, f"Window {i}: expected 3 tokens (100 // 30 = 3), got {T_prime}"

    def test_strip_token_padding_uneven(self):
        """With uneven division, all 3 tokens still fit within window bounds.

        100ms window, 30ms tokens -> tokens end at 30, 60, 90ms < 100ms,
        so no tokens should be stripped.

        """
        spikes, tokenizer, loader, windows = self._make_pipeline_components()

        for i, batch in enumerate(loader):
            latents, metadata = tokenizer.encode_with_timestamps(batch)

            window_latents = latents[0]
            window_starts = metadata["token_start_time"][0]
            window_ends = metadata["token_end_time"][0]

            _, window_end = windows[i]

            result = _strip_token_padding(
                window_latents,
                window_starts,
                window_ends,
                absolute_end=window_end,
            )

            assert result["latents"].shape[0] == 3, (
                f"Window {i}: all 3 tokens should survive strip, got {result['latents'].shape[0]}"
            )

    def test_overlap_trimming_uneven(self):
        """Verify overlap trimming with uneven token division.

        Note: this tests the same scenario as test_full_pipeline_uneven_division
        but at the level of individual functions (_strip_token_padding,
        _trim_token_overlap) rather than the full pipeline. The full pipeline
        test is strictly a superset.

        """
        spikes, tokenizer, loader, windows = self._make_pipeline_components()

        prev_last_token_end: float | None = None
        all_starts = []
        all_ends = []

        for i, batch in enumerate(loader):
            latents, metadata = tokenizer.encode_with_timestamps(batch)

            window_latents = latents[0]
            window_starts = metadata["token_start_time"][0]
            window_ends = metadata["token_end_time"][0]

            _, window_end = windows[i]

            strip_result = _strip_token_padding(
                window_latents,
                window_starts,
                window_ends,
                absolute_end=window_end,
            )

            trim_result = _trim_token_overlap(
                latents=strip_result["latents"],
                token_start_times=strip_result["token_start_times"],
                token_end_times=strip_result["token_end_times"],
                prev_last_token_end_time=prev_last_token_end,
            )

            trimmed_starts = trim_result["token_start_times"]
            trimmed_ends = trim_result["token_end_times"]

            if i == 0:
                # First window: all 3 tokens survive
                assert trimmed_starts.shape[0] == 3, (
                    f"Window 0: expected 3 tokens, got {trimmed_starts.shape[0]}"
                )
            else:
                # Subsequent windows: only the last token survives
                # (first two have start_time < prev_last_token_end)
                assert trimmed_starts.shape[0] == 1, (
                    f"Window {i}: expected 1 token after overlap trim, "
                    f"got {trimmed_starts.shape[0]}"
                )

            all_starts.append(trimmed_starts)
            all_ends.append(trimmed_ends)

            if trimmed_ends.numel() > 0:
                prev_last_token_end = float(trimmed_ends[-1].item())

        concat_starts = torch.cat(all_starts, dim=0)
        concat_ends = torch.cat(all_ends, dim=0)

        # 3 from window 0 + 1 each from windows 1-3 = 6 tokens
        assert concat_starts.shape[0] == 6, (
            f"Expected 6 total tokens, got {concat_starts.shape[0]}"
        )

        expected_starts = torch.tensor(
            [0.000, 0.030, 0.060, 0.110, 0.160, 0.210], dtype=torch.float64
        )
        expected_ends = torch.tensor(
            [0.030, 0.060, 0.090, 0.140, 0.190, 0.240], dtype=torch.float64
        )

        torch.testing.assert_close(
            concat_starts,
            expected_starts,
            atol=1e-9,
            rtol=0,
            msg="Uneven: concatenated token start times mismatch",
        )
        torch.testing.assert_close(
            concat_ends,
            expected_ends,
            atol=1e-9,
            rtol=0,
            msg="Uneven: concatenated token end times mismatch",
        )

    def test_full_pipeline_uneven_division(self):
        """Full end-to-end test with uneven token division."""
        spikes, tokenizer, loader, windows = self._make_pipeline_components()

        with pytest.raises(
            ValueError,
            match="does not divide configured input window length",
        ):
            list(
                iter_collated_validation_segments(
                    loader,
                    _make_tokenizer_fun(tokenizer),
                    token_bin_size_ms=self.TOKEN_BIN_SIZE_MS,
                    window_length_seconds=self.WINDOW_SIZE_S,
                    device=torch.device("cpu"),
                    trim_token_overlap=True,
                    drop_overlapping_windows=False,
                )
            )

    def test_uneven_no_overlap_windows(self):
        """Non-overlapping windows with uneven division fail at the upfront divisibility check."""
        spikes = torch.zeros((self.N_UNITS, self.TOTAL_BINS))
        tokenizer = _make_poisson_tokenizer(self.TOKEN_BIN_SIZE_MS, self.SAMPLING_RATE)

        windows = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.25)]
        loader = _make_dataloader(spikes, self.SAMPLING_RATE, windows, batch_size=1)

        with pytest.raises(
            ValueError,
            match="does not divide configured input window length",
        ):
            list(
                iter_collated_validation_segments(
                    loader,
                    _make_tokenizer_fun(tokenizer),
                    token_bin_size_ms=self.TOKEN_BIN_SIZE_MS,
                    window_length_seconds=self.WINDOW_SIZE_S,
                    device=torch.device("cpu"),
                )
            )


# ===========================================================================
# Test: tokenizer_fun returns latents + timestamps directly
# ===========================================================================


class TestTimestampAlignmentTokenizerFunPath:
    """Test timestamps when tokenizer_fun returns latents and timestamps directly.

    The tokenizer_fun returns a dict with "reconstruction", "latents",
    "token_start_time", and "token_end_time", as
    ``reconstruct_from_input(return_latents=True)`` does.

    """

    SAMPLING_RATE = 1000.0
    N_UNITS = 3
    TOTAL_BINS = 200  # 200ms
    TOKEN_BIN_SIZE_MS = 50.0

    def test_tokenizer_fun_timestamps(self):
        """Verify timestamps via _make_tokenizer_fun match direct encode_with_timestamps."""
        spikes = torch.zeros((self.N_UNITS, self.TOTAL_BINS))
        tokenizer = _make_poisson_tokenizer(self.TOKEN_BIN_SIZE_MS, self.SAMPLING_RATE)

        # Non-overlapping windows
        windows = [(0.0, 0.1), (0.1, 0.2)]
        loader = _make_dataloader(spikes, self.SAMPLING_RATE, windows, batch_size=1)

        segments = list(
            iter_collated_validation_segments(
                loader,
                _make_tokenizer_fun(tokenizer),
                token_bin_size_ms=self.TOKEN_BIN_SIZE_MS,
                device=torch.device("cpu"),
            )
        )

        assert len(segments) == 1

        # 100ms / 50ms = 2 tokens per window, 2 windows = 4 tokens
        seg = segments[0]
        assert seg.token_start_times is not None
        assert seg.token_end_times is not None
        assert seg.token_timestamps is not None
        assert seg.token_start_times.shape[0] == 4

        expected_starts = torch.tensor([0.000, 0.050, 0.100, 0.150], dtype=torch.float64)
        expected_ends = torch.tensor([0.050, 0.100, 0.150, 0.200], dtype=torch.float64)
        expected_centers = (expected_starts + expected_ends) / 2.0

        torch.testing.assert_close(
            seg.token_start_times,
            expected_starts,
            atol=1e-9,
            rtol=0,
            msg="tokenizer_fun path: start times mismatch",
        )
        torch.testing.assert_close(
            seg.token_end_times,
            expected_ends,
            atol=1e-9,
            rtol=0,
            msg="tokenizer_fun path: end times mismatch",
        )
        torch.testing.assert_close(
            seg.token_timestamps,
            expected_centers,
            atol=1e-9,
            rtol=0,
            msg="tokenizer_fun path: center times mismatch",
        )


# ===========================================================================
# Test: flush_segment gap and contiguity checks
# ===========================================================================


class TestFlushSegmentTimestampIntegrity:
    """Test that _flush_segment correctly concatenates timestamps."""

    # def test_flush_preserves_monotonicity(self):
    #     """Concatenated token times from _flush_segment should be monotonic.

    #
    #     Superfluous?! Dont see the point of this test. Dropping.
    #     """
    #     N_UNITS = 3
    #     T_tokens = 2

    #     window_buffer = []
    #     for i in range(3):
    #         t_start = i * 0.050
    #         window_buffer.append({
    #             "original": torch.zeros(N_UNITS, 50),
    #             "reconstructed": torch.zeros(N_UNITS, 50),
    #             "mask": torch.ones(N_UNITS, 50, dtype=torch.bool),
    #             "latents": torch.randn(T_tokens, N_UNITS, 1),
    #             "session_id": "test",
    #             "metadata": {"sampling_rate": 1000.0},
    #             "absolute_start": i * 0.050,
    #             "absolute_end": (i + 1) * 0.050,
    #             "token_start_times": torch.tensor(
    #                 [t_start, t_start + 0.025], dtype=torch.float64
    #             ),
    #             "token_end_times": torch.tensor(
    #                 [t_start + 0.025, t_start + 0.050], dtype=torch.float64
    #             ),
    #             "token_timestamps": torch.tensor(
    #                 [t_start + 0.0125, t_start + 0.0375], dtype=torch.float64
    #             ),
    #         })

    #     seg = _flush_segment(window_buffer)

    #     assert seg.token_start_times.shape[0] == 6
    #     assert seg.token_timestamps.shape[0] == 6
    #     assert torch.all(seg.token_start_times[1:] > seg.token_start_times[:-1])
    #     assert torch.all(seg.token_end_times[1:] > seg.token_end_times[:-1])
    #     assert torch.all(seg.token_timestamps[1:] > seg.token_timestamps[:-1])

    #     # Center times should be midpoints of start/end
    #     expected_centers = (seg.token_start_times + seg.token_end_times) / 2.0
    #     torch.testing.assert_close(
    #         seg.token_timestamps, expected_centers, atol=1e-9, rtol=0,
    #     )

    #     # No gaps between consecutive tokens
    #     gaps = seg.token_start_times[1:] - seg.token_end_times[:-1]
    #     torch.testing.assert_close(
    #         gaps, torch.zeros(5, dtype=torch.float64), atol=1e-9, rtol=0,
    #     )

    def test_flush_detects_temporal_gap_in_windows(self):
        """_flush_segment concatenates across a gap between adjacent windows and
        records it in the segment metadata (the segment is kept, not dropped).
        """
        N_UNITS = 3

        window_buffer = [
            {
                "original": torch.zeros(N_UNITS, 50),
                "reconstructed": torch.zeros(N_UNITS, 50),
                "mask": torch.ones(N_UNITS, 50, dtype=torch.bool),
                "session_id": "test",
                "metadata": {"sampling_rate": 1000.0},
                "absolute_start": 0.0,
                "absolute_end": 0.050,
            },
            {
                "original": torch.zeros(N_UNITS, 50),
                "reconstructed": torch.zeros(N_UNITS, 50),
                "mask": torch.ones(N_UNITS, 50, dtype=torch.bool),
                "session_id": "test",
                "metadata": {"sampling_rate": 1000.0},
                "absolute_start": 0.060,  # Gap of 10ms!
                "absolute_end": 0.110,
            },
        ]

        result = _flush_segment(window_buffer)
        assert result.original is not None
        assert result.original.shape == (N_UNITS, 100)
        assert result.metadata["n_gaps"] == 1
        assert result.metadata["gap_seconds"] == pytest.approx(0.010)


# ===========================================================================
# Test: edge case — single token per window
# ===========================================================================


# class TestSingleTokenPerWindow:
#     """Edge case: window size barely fits one token.

#     50ms window, 50ms tokens -> exactly 1 token per window.

#     # So what is this supposed to test?
#     # I consider this superfluous as well. It is also not reproducing actual
#     #  data properties for which it might make sense to test this.
#     """

#     def test_single_token_per_window_timestamps(self):
#         """Verify timestamps when each window produces exactly one token.

#
#         """
#         SAMPLING_RATE = 1000.0
#         N_UNITS = 3
#         TOTAL_BINS = 200
#         TOKEN_BIN_SIZE_MS = 50.0

#         spikes = torch.zeros((N_UNITS, TOTAL_BINS))
#         tokenizer = _make_poisson_tokenizer(TOKEN_BIN_SIZE_MS, SAMPLING_RATE)

#         # Non-overlapping 50ms windows
#         windows = [(i * 0.05, (i + 1) * 0.05) for i in range(4)]
#         loader = _make_dataloader(spikes, SAMPLING_RATE, windows, batch_size=1)

#         segments = list(
#             iter_collated_validation_segments(
#                 loader,
#                 _make_tokenizer_fun(tokenizer),
#                 token_bin_size_ms=TOKEN_BIN_SIZE_MS,
#                 device=torch.device("cpu"),
#             )
#         )

#         assert len(segments) == 1
#         seg = segments[0]

#         assert seg.token_start_times.shape[0] == 4
#         assert seg.token_timestamps is not None

#         expected_starts = torch.tensor(
#             [0.000, 0.050, 0.100, 0.150], dtype=torch.float64
#         )
#         expected_ends = torch.tensor(
#             [0.050, 0.100, 0.150, 0.200], dtype=torch.float64
#         )
#         expected_centers = (expected_starts + expected_ends) / 2.0

#         torch.testing.assert_close(
#             seg.token_start_times, expected_starts, atol=1e-9, rtol=0,
#         )
#         torch.testing.assert_close(
#             seg.token_end_times, expected_ends, atol=1e-9, rtol=0,
#         )
#         torch.testing.assert_close(
#             seg.token_timestamps, expected_centers, atol=1e-9, rtol=0,
#         )
