"""Tests for validation_collation.py shape handling."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset
from torch_brain.data.collate import pad, pad2d
from torch_brain.data.dataset import DatasetIndex

from ntt.contrib.tokenizer.cosmos.cosmos_2d import CosmosTokenizer2DSpec
from ntt.contrib.tokenizer.identity_tokenizer import IdentityTokenizerSpec
from ntt.core.components.types import (
    BinnedSpikesInput,
    DataType,
    RatesInput,
    ReconstructionOutput,
)
from ntt.core.data.datamodule import collate
from ntt.evaluation.utils.validation_collation import (
    _iter_collated_validation_segments_with_per_stimulus_sampling,
    iter_collated_validation_segments,
)
from tests.fixtures import SimpleDataset


class _RatesStimulusDataset(Dataset):
    """Single-session synthetic rates dataset for per-stimulus SNR tests."""

    def __init__(self) -> None:
        self.session_id = "test_rates_session"
        self.sampling_rate = 1000.0
        self.n_units = 32
        self._recording_objects = {
            self.session_id: SimpleNamespace(
                stimuli=SimpleNamespace(
                    start=np.asarray([0.0, 1.0], dtype=np.float64),
                    end=np.asarray([0.9, 1.9], dtype=np.float64),
                    image_id=np.asarray([7, 11], dtype=np.int64),
                )
            )
        }
        total_bins = 2000
        self._rates = torch.linspace(
            0.0,
            1.0,
            steps=total_bins * self.n_units,
            dtype=torch.float32,
        ).reshape(total_bins, self.n_units)

    def __len__(self) -> int:
        return 1

    def _get_data_object(self, recording_id: str):
        return self._recording_objects[recording_id]

    def __getitem__(self, index: DatasetIndex | int) -> RatesInput:
        if isinstance(index, DatasetIndex):
            start_bin = int(round(float(index.start) * self.sampling_rate))
            end_bin = int(round(float(index.end) * self.sampling_rate))
            abs_start = torch.tensor(float(index.start), dtype=torch.float64)
            abs_end = torch.tensor(float(index.end), dtype=torch.float64)
        else:
            start_bin = 0
            end_bin = self._rates.shape[0]
            abs_start = torch.tensor(0.0, dtype=torch.float64)
            abs_end = torch.tensor(end_bin / self.sampling_rate, dtype=torch.float64)

        rate_matrix = self._rates[start_bin:end_bin]
        duration_sec = torch.tensor(
            (end_bin - start_bin) / self.sampling_rate,
            dtype=torch.float64,
        )
        return RatesInput(
            rate_matrix=rate_matrix,
            pad_and_gap_mask=torch.zeros_like(rate_matrix, dtype=torch.int32),
            sampling_rate=self.sampling_rate,
            unit_ids=torch.arange(self.n_units, dtype=torch.int64),
            unit_ids_pad_mask=torch.ones(self.n_units, dtype=torch.int64),
            duration_sec=duration_sec,
            session_id=self.session_id,
            subject_id="test_subject",
            species="test_species",
            absolute_start=abs_start,
            absolute_end=abs_end,
            behavioral_data=None,
            behavioral_intervals=None,
        )


class _LabelExtensionStimulusDataset(Dataset):
    """Single-session stimulus fixture for post-stim label extension tests."""

    def __init__(self) -> None:
        self.session_id = "label_extension_session"
        self.sampling_rate = 100.0
        self.n_units = 4
        self._recording_objects = {
            self.session_id: SimpleNamespace(
                stimuli=SimpleNamespace(
                    start=np.asarray([0.1, 0.4], dtype=np.float64),
                    end=np.asarray([0.2, 0.5], dtype=np.float64),
                    image_id=np.asarray([7, 11], dtype=np.int64),
                )
            )
        }
        self._spikes = torch.zeros((90, self.n_units), dtype=torch.int32)

    def __len__(self) -> int:
        return 1

    def _get_data_object(self, recording_id: str):
        return self._recording_objects[recording_id]

    def __getitem__(self, index: DatasetIndex | int) -> BinnedSpikesInput:
        if isinstance(index, DatasetIndex):
            session_id = index.recording_id
            start = float(index.start)
            end = float(index.end)
        else:
            session_id = self.session_id
            start = 0.0
            end = 0.9

        start_bin = int(round(start * self.sampling_rate))
        end_bin = int(round(end * self.sampling_rate))
        spike_matrix = self._spikes[start_bin:end_bin].contiguous()
        duration_sec = torch.tensor(
            (end_bin - start_bin) / self.sampling_rate,
            dtype=torch.float64,
        )
        return BinnedSpikesInput(
            spike_matrix=spike_matrix,
            pad_and_gap_mask=torch.ones_like(spike_matrix, dtype=torch.int32),
            sampling_rate=self.sampling_rate,
            unit_ids=torch.arange(self.n_units, dtype=torch.int64),
            unit_ids_pad_mask=torch.ones(self.n_units, dtype=torch.int32),
            duration_sec=duration_sec,
            session_id=session_id,
            subject_id="test_subject",
            species="test_species",
            absolute_start=torch.tensor(start, dtype=torch.float64),
            absolute_end=torch.tensor(end, dtype=torch.float64),
            behavioral_data=None,
            behavioral_intervals=None,
        )


class _VariableDurationStimulusDataset(Dataset):
    """Single-session fixture with unequal stimulus-window durations."""

    def __init__(self) -> None:
        self.session_id = "variable_duration_session"
        self.sampling_rate = 100.0
        self.n_units = 2
        self._recording_objects = {
            self.session_id: SimpleNamespace(
                stimuli=SimpleNamespace(
                    start=np.asarray([0.0, 0.5], dtype=np.float64),
                    end=np.asarray([0.1, 0.8], dtype=np.float64),
                    image_id=np.asarray([7, 11], dtype=np.int64),
                )
            )
        }
        self._spikes = torch.zeros((100, self.n_units), dtype=torch.int32)

    def __len__(self) -> int:
        return 1

    def _get_data_object(self, recording_id: str):
        return self._recording_objects[recording_id]

    def __getitem__(self, index: DatasetIndex | int) -> BinnedSpikesInput:
        if isinstance(index, DatasetIndex):
            session_id = index.recording_id
            start = float(index.start)
            end = float(index.end)
        else:
            session_id = self.session_id
            start = 0.0
            end = 1.0

        start_bin = int(round(start * self.sampling_rate))
        end_bin = int(round(end * self.sampling_rate))
        spike_matrix = self._spikes[start_bin:end_bin].contiguous()
        duration_sec = torch.tensor(
            (end_bin - start_bin) / self.sampling_rate,
            dtype=torch.float64,
        )
        return BinnedSpikesInput(
            spike_matrix=pad2d(spike_matrix),
            pad_and_gap_mask=pad2d(torch.ones_like(spike_matrix, dtype=torch.int32)),
            sampling_rate=self.sampling_rate,
            unit_ids=torch.arange(self.n_units, dtype=torch.int64),
            unit_ids_pad_mask=torch.ones(self.n_units, dtype=torch.int32),
            duration_sec=duration_sec,
            session_id=session_id,
            subject_id="test_subject",
            species="test_species",
            absolute_start=torch.tensor(start, dtype=torch.float64),
            absolute_end=torch.tensor(end, dtype=torch.float64),
            behavioral_data=None,
            behavioral_intervals=None,
        )


class _TwoSessionDataset(Dataset):
    """Two-session synthetic spikes dataset for session-boundary trim tests."""

    def __init__(self) -> None:
        self.sampling_rate = 1000.0
        self.n_units = 4
        self._session_ids = ("session_a", "session_b")
        total_bins = 3000
        self._spikes = {
            session_id: torch.zeros((total_bins, self.n_units), dtype=torch.int32)
            for session_id in self._session_ids
        }

    def __len__(self) -> int:
        return len(self._session_ids)

    def __getitem__(self, index: DatasetIndex | int) -> BinnedSpikesInput:
        if isinstance(index, DatasetIndex):
            session_id = index.recording_id
            start = float(index.start)
            end = float(index.end)
        else:
            session_id = self._session_ids[int(index)]
            start = 0.0
            end = 3.0

        start_bin = int(round(start * self.sampling_rate))
        end_bin = int(round(end * self.sampling_rate))
        spike_matrix = self._spikes[session_id][start_bin:end_bin].contiguous()
        duration_sec = torch.tensor(
            (end_bin - start_bin) / self.sampling_rate,
            dtype=torch.float64,
        )
        return BinnedSpikesInput(
            spike_matrix=spike_matrix,
            pad_and_gap_mask=torch.ones_like(spike_matrix, dtype=torch.int32),
            sampling_rate=self.sampling_rate,
            unit_ids=torch.arange(self.n_units, dtype=torch.int64),
            unit_ids_pad_mask=torch.ones(self.n_units, dtype=torch.int32),
            duration_sec=duration_sec,
            session_id=session_id,
            subject_id=f"subject_{session_id}",
            species="test_species",
            absolute_start=torch.tensor(start, dtype=torch.float64),
            absolute_end=torch.tensor(end, dtype=torch.float64),
            behavioral_data=None,
            behavioral_intervals=None,
        )


def test_collation_shape_multiple_windows():
    """Test that multiple windows are concatenated along time dimension."""
    n_units = 5
    time_bins_per_window = 1000
    n_windows = 3
    dataset = SimpleDataset(
        n_units=n_units, time_bins_per_window=time_bins_per_window, n_windows=n_windows
    )

    # Create indices for sequential sampling
    indices = [
        DatasetIndex("test_session", 0.0, 1.0),
        DatasetIndex("test_session", 1.0, 2.0),
        DatasetIndex("test_session", 2.0, 3.0),
    ]
    loader = DataLoader(dataset, batch_size=1, sampler=indices, collate_fn=collate)

    def identity_tokenizer(batch: BinnedSpikesInput) -> dict:
        """Return the same spike matrix as reconstruction."""
        return {"reconstruction": batch.spike_matrix}

    segments = list(
        iter_collated_validation_segments(
            loader,
            identity_tokenizer,
            device=torch.device("cpu"),
        )
    )

    assert len(segments) == 1
    segment = segments[0]

    # After concatenating 3 windows, should have shape (units, time_bins * 3)
    expected_time_bins = time_bins_per_window * n_windows
    assert segment.original.shape == (n_units, expected_time_bins), (
        f"Expected shape ({n_units}, {expected_time_bins}), got {segment.original.shape}"
    )
    assert segment.reconstructed.shape == (n_units, expected_time_bins), (
        f"Expected shape ({n_units}, {expected_time_bins}), got {segment.reconstructed.shape}"
    )


class _InteriorGapDataset(Dataset):
    """Two-window single-session fixture; one window carries an interior gap.

    The gap is a contiguous block of ``pad_and_gap_mask == False`` time bins in
    the middle of the window (valid bins on both sides), exactly the shape that
    ``CreateBinnedSpikeCountsArray`` produces for a session whose valid domain
    has two disjoint intervals. Trailing bins stay valid so ``_strip_input_padding``
    cannot strip the gap away as trailing padding.
    """

    def __init__(self, n_units: int, time_bins_per_window: int, gap_window_idx: int):
        self.n_units = n_units
        self.time_bins_per_window = time_bins_per_window
        self.gap_window_idx = gap_window_idx
        self.sampling_rate = 1000.0
        self.session_id = "test_session"

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: DatasetIndex | int) -> BinnedSpikesInput:
        if hasattr(index, "start"):
            window_idx = int(index.start * self.sampling_rate / self.time_bins_per_window)
            abs_start = torch.tensor(float(index.start), dtype=torch.float64)
            abs_end = torch.tensor(float(index.end), dtype=torch.float64)
        else:
            window_idx = int(index)
            dur = self.time_bins_per_window / self.sampling_rate
            abs_start = torch.tensor(index * dur, dtype=torch.float64)
            abs_end = torch.tensor((index + 1) * dur, dtype=torch.float64)

        spike_matrix = torch.ones((self.time_bins_per_window, self.n_units), dtype=torch.int32)
        pad_and_gap_mask = torch.ones_like(spike_matrix, dtype=torch.int32)
        if window_idx == self.gap_window_idx:
            lo = self.time_bins_per_window // 3
            hi = 2 * self.time_bins_per_window // 3
            pad_and_gap_mask[lo:hi, :] = 0  # interior gap; trailing bins stay valid

        return BinnedSpikesInput(
            spike_matrix=spike_matrix,
            pad_and_gap_mask=pad_and_gap_mask,
            sampling_rate=self.sampling_rate,
            unit_ids=torch.arange(self.n_units, dtype=torch.int32),
            unit_ids_pad_mask=torch.ones(self.n_units, dtype=torch.int32),
            duration_sec=torch.tensor(self.time_bins_per_window / self.sampling_rate),
            session_id=self.session_id,
            subject_id="test_subject",
            species="test_species",
            absolute_start=abs_start,
            absolute_end=abs_end,
            behavioral_data=None,
        )


def test_collation_skips_window_with_interior_gap():
    """A window whose pad_and_gap_mask has an interior gap must be dropped, not
    processed with zero-filled gap bins masquerading as real zero-count data."""
    n_units = 5
    time_bins_per_window = 1000
    dataset = _InteriorGapDataset(
        n_units=n_units, time_bins_per_window=time_bins_per_window, gap_window_idx=1
    )

    indices = [
        DatasetIndex("test_session", 0.0, 1.0),
        DatasetIndex("test_session", 1.0, 2.0),
    ]
    loader = DataLoader(dataset, batch_size=1, sampler=indices, collate_fn=collate)

    def identity_tokenizer(batch: BinnedSpikesInput) -> dict:
        return {"reconstruction": batch.spike_matrix}

    segments = list(
        iter_collated_validation_segments(
            loader,
            identity_tokenizer,
            device=torch.device("cpu"),
        )
    )

    assert len(segments) == 1
    segment = segments[0]
    # Only the clean window (0) survives; the gapped window (1) is skipped, so the
    # concatenated trace spans exactly one window, not two.
    assert segment.original.shape == (n_units, time_bins_per_window), (
        f"Gapped window should be skipped; expected ({n_units}, "
        f"{time_bins_per_window}), got {segment.original.shape}"
    )


def test_collation_handles_reconstruction_output():
    """iter_collated_validation_segments should handle ReconstructionOutput from tokenizer_fun."""
    n_units = 5
    time_bins_per_window = 1000
    n_windows = 2
    dataset = SimpleDataset(
        n_units=n_units, time_bins_per_window=time_bins_per_window, n_windows=n_windows
    )

    indices = [
        DatasetIndex("test_session", 0.0, 1.0),
        DatasetIndex("test_session", 1.0, 2.0),
    ]
    loader = DataLoader(dataset, batch_size=1, sampler=indices, collate_fn=collate)

    def recon_output_tokenizer(batch: BinnedSpikesInput) -> ReconstructionOutput:
        """Return a ReconstructionOutput instead of a dict."""
        return ReconstructionOutput(
            reconstruction=batch.spike_matrix,
            data_type=DataType.SPIKES,
            bin_size_seconds=0.001,
        )

    segments = list(
        iter_collated_validation_segments(
            loader,
            recon_output_tokenizer,
            device=torch.device("cpu"),
        )
    )

    assert len(segments) == 1
    segment = segments[0]
    expected_time_bins = time_bins_per_window * n_windows
    assert segment.original.shape == (n_units, expected_time_bins)
    assert segment.reconstructed.shape == (n_units, expected_time_bins)


def test_collation_max_session_windows_caps_first_windows() -> None:
    n_units = 5
    time_bins_per_window = 1000
    n_windows = 3
    dataset = SimpleDataset(
        n_units=n_units,
        time_bins_per_window=time_bins_per_window,
        n_windows=n_windows,
    )
    indices = [
        DatasetIndex("test_session", 0.0, 1.0),
        DatasetIndex("test_session", 1.0, 2.0),
        DatasetIndex("test_session", 2.0, 3.0),
    ]
    loader = DataLoader(dataset, batch_size=1, sampler=indices, collate_fn=collate)

    def identity_tokenizer(batch: BinnedSpikesInput) -> dict:
        return {"reconstruction": batch.spike_matrix}

    segments = list(
        iter_collated_validation_segments(
            loader,
            identity_tokenizer,
            device=torch.device("cpu"),
            max_windows=1,
            max_session_windows=2,
            preserve_overlap_state_on_partial_flush=True,
        )
    )

    assert len(segments) == 2
    assert [int(segment.segment_index) for segment in segments] == [0, 1]
    assert all(segment.session_id == "test_session" for segment in segments)
    assert all(segment.original.shape == (n_units, time_bins_per_window) for segment in segments)


def test_collation_resets_overlap_state_on_session_change_after_partial_flush() -> None:
    dataset = _TwoSessionDataset()
    indices = [
        DatasetIndex("session_a", 0.0, 1.0),
        DatasetIndex("session_a", 1.0, 2.0),
        DatasetIndex("session_a", 2.0, 3.0),
        DatasetIndex("session_b", 0.0, 1.0),
        DatasetIndex("session_b", 1.0, 2.0),
        DatasetIndex("session_b", 2.0, 3.0),
    ]
    loader = DataLoader(dataset, batch_size=1, sampler=indices, collate_fn=collate)

    def tokenizer_with_latents(batch: BinnedSpikesInput) -> dict:
        batch_size = batch.spike_matrix.shape[0]
        token_centers = (
            torch.tensor(
                [0.05 + 0.1 * i for i in range(10)],
                dtype=torch.float64,
            )
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        token_starts = token_centers - 0.05
        token_ends = token_centers + 0.05
        latents = torch.zeros((batch_size, 10, 1), dtype=torch.float32)
        return {
            "reconstruction": batch.spike_matrix,
            "latents": latents,
            "token_timestamps": token_centers,
            "token_start_time": token_starts,
            "token_end_time": token_ends,
        }

    segments = list(
        iter_collated_validation_segments(
            loader,
            tokenizer_with_latents,
            device=torch.device("cpu"),
            max_windows=1,
            max_session_windows=2,
            preserve_overlap_state_on_partial_flush=True,
        )
    )

    assert len(segments) == 4
    assert [segment.session_id for segment in segments] == [
        "session_a",
        "session_a",
        "session_b",
        "session_b",
    ]
    first_b = segments[2]
    assert int(first_b.segment_index) == 2
    assert first_b.segment_absolute_start == pytest.approx(0.0)
    assert first_b.original.shape == (dataset.n_units, 1000)
    assert first_b.reconstructed is not None
    assert first_b.reconstructed.shape == (dataset.n_units, 1000)
    assert first_b.latents is not None
    assert first_b.latents.shape[-2:] == (10, 1)
    assert first_b.token_timestamps is not None
    assert tuple(first_b.token_timestamps.shape) == (10,)


def test_collation_with_timestamps_in_tokenizer_output():
    """When tokenizer_fun returns token timestamps, segments should have token_start_times and
    token_end_times."""
    n_units = 5
    time_bins_per_window = 1000
    n_windows = 2
    token_bin_size_ms = 100.0  # 100ms per token → 10 tokens per 1s window

    dataset = SimpleDataset(
        n_units=n_units, time_bins_per_window=time_bins_per_window, n_windows=n_windows
    )
    indices = [
        DatasetIndex("test_session", 0.0, 1.0),
        DatasetIndex("test_session", 1.0, 2.0),
    ]
    loader = DataLoader(dataset, batch_size=1, sampler=indices, collate_fn=collate)

    def tokenizer_with_timestamps(batch: BinnedSpikesInput) -> dict:
        """Return reconstruction, latents, and token timestamps.

        Mimics the output of reconstruct_from_input(return_latents=True).

        """
        B = batch.spike_matrix.shape[0]
        T = batch.spike_matrix.shape[1]
        token_bin_size_sec = token_bin_size_ms / 1000.0
        n_tokens = int(T * (1.0 / batch.sampling_rate) / token_bin_size_sec)

        abs_start = batch.absolute_start.to(dtype=torch.float64)  # [B]
        indices_t = torch.arange(n_tokens, dtype=torch.float64)  # [T']

        token_start_time = abs_start[:, None] + indices_t[None, :] * token_bin_size_sec
        token_end_time = abs_start[:, None] + (indices_t[None, :] + 1) * token_bin_size_sec
        token_timestamps = abs_start[:, None] + (indices_t[None, :] + 0.5) * token_bin_size_sec

        return {
            "reconstruction": batch.spike_matrix,
            "latents": torch.randn(B, n_tokens, n_units, 4),
            "token_timestamps": token_timestamps,
            "token_start_time": token_start_time,
            "token_end_time": token_end_time,
        }

    segments = list(
        iter_collated_validation_segments(
            loader,
            tokenizer_with_timestamps,
            token_bin_size_ms=token_bin_size_ms,
            device=torch.device("cpu"),
        )
    )

    assert len(segments) == 1
    segment = segments[0]

    # Token timestamps should be present
    assert segment.token_start_times is not None, "token_start_times should be set"
    assert segment.token_end_times is not None, "token_end_times should be set"
    assert segment.token_timestamps is not None, "token_timestamps (centers) should be set"

    # 10 tokens per window × 2 windows = 20 tokens
    expected_tokens = 20
    assert segment.token_start_times.shape == (expected_tokens,), (
        f"Expected {expected_tokens} token start times, got {segment.token_start_times.shape}"
    )
    assert segment.token_end_times.shape == (expected_tokens,), (
        f"Expected {expected_tokens} token end times, got {segment.token_end_times.shape}"
    )
    assert segment.token_timestamps.shape == (expected_tokens,)

    # Token times should be monotonically increasing
    assert torch.all(segment.token_start_times[1:] >= segment.token_start_times[:-1])

    # Center times should be midpoints of start/end
    expected_centers = (segment.token_start_times + segment.token_end_times) / 2.0
    torch.testing.assert_close(
        segment.token_timestamps,
        expected_centers,
        atol=1e-9,
        rtol=0,
    )

    # First token should start at 0.0, last token should end at 2.0
    assert abs(float(segment.token_start_times[0]) - 0.0) < 1e-6
    assert abs(float(segment.token_end_times[-1]) - 2.0) < 1e-6

    # Latents should also be present — stacked across windows
    assert segment.latents is not None
    assert segment.latents.shape[0] == n_windows  # stacked along window dim


def test_per_stimulus_snr_iterator_accepts_rates_batches_for_cosmos_spec():
    """The per-stimulus SNR path must support rate-input tokenizers like cosmos_2d."""
    dataset = _RatesStimulusDataset()
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate)
    spec = CosmosTokenizer2DSpec(
        window_length_seconds=1.0,
        bin_size_ms=1.0,
        n_neurons=dataset.n_units,
        input_type=DataType.RATES,
        output_type=DataType.RATES,
    )

    def tokenizer_with_latents(batch: RatesInput) -> dict[str, torch.Tensor]:
        batch_size = batch.rate_matrix.shape[0]
        token_start_time = batch.absolute_start[:, None] + 0.25
        token_end_time = batch.absolute_start[:, None] + 0.75
        token_timestamps = (token_start_time + token_end_time) / 2.0
        return {
            "reconstruction": batch.rate_matrix.clone(),
            "latents": torch.zeros((batch_size, 1, 1, 3), dtype=torch.float32),
            "token_timestamps": token_timestamps,
            "token_start_time": token_start_time,
            "token_end_time": token_end_time,
        }

    segments = list(
        _iter_collated_validation_segments_with_per_stimulus_sampling(
            loader,
            tokenizer_with_latents,
            tokenizer_spec=spec,
            interval_source="stimuli",
            grouping_field="image_id",
            interval_end_lag_ms=None,
            fixed_window_duration_ms=1000.0,
            label_extension_ms=100.0,
            device=torch.device("cpu"),
        )
    )

    assert len(segments) == 1
    segment = segments[0]
    assert segment.session_id == dataset.session_id
    assert segment.latents is not None
    assert segment.latents.shape == (2, 1, 1, 3)
    assert segment.token_timestamps is not None
    assert segment.token_timestamps.shape == (2,)
    np.testing.assert_array_equal(
        segment.behavioral_data_token_res["image_id"],  # type: ignore[index]
        np.asarray([7.0, 11.0]),
    )


@pytest.mark.parametrize(
    ("label_extension_ms", "expected_third_label"),
    [(100.0, np.nan), (300.0, 7.0)],
)
def test_per_stimulus_labels_extend_past_stim_end_and_clamp_at_next_stim(
    label_extension_ms: float,
    expected_third_label: float,
):
    """Per-stim labels should keep the post-stim response bin, but never cross into the next
    stim."""
    dataset = _LabelExtensionStimulusDataset()
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate)
    spec = IdentityTokenizerSpec(window_length_seconds=1.0, bin_size_ms=100.0)

    def tokenizer_with_100ms_tokens(batch: BinnedSpikesInput) -> dict[str, torch.Tensor]:
        batch_size = batch.spike_matrix.shape[0]
        n_input_bins = batch.spike_matrix.shape[1]
        n_tokens = n_input_bins // 10
        token_offsets = torch.arange(n_tokens, dtype=torch.float64)
        token_start = batch.absolute_start[:, None] + token_offsets[None, :] * 0.1
        token_end = token_start + 0.1
        token_timestamps = (token_start + token_end) / 2.0
        return {
            "reconstruction": batch.spike_matrix.to(torch.float32),
            "latents": torch.zeros((batch_size, n_tokens, 1), dtype=torch.float32),
            "token_timestamps": token_timestamps,
            "token_start_time": token_start,
            "token_end_time": token_end,
        }

    segments = list(
        _iter_collated_validation_segments_with_per_stimulus_sampling(
            loader,
            tokenizer_with_100ms_tokens,
            tokenizer_spec=spec,
            interval_source="stimuli",
            grouping_field="image_id",
            interval_end_lag_ms=300.0,
            label_extension_ms=label_extension_ms,
            max_windows=1,
            device=torch.device("cpu"),
        )
    )

    assert len(segments) == 2
    first_labels = segments[0].behavioral_data_token_res["image_id"]  # type: ignore[index]
    np.testing.assert_array_equal(first_labels[:2], np.asarray([7.0, 7.0]))
    if np.isnan(expected_third_label):
        assert np.isnan(first_labels[2])
    else:
        assert first_labels[2] == pytest.approx(expected_third_label)


def test_per_stimulus_snr_variable_windows_trim_token_padding_before_labeling():
    """Variable-duration SNR windows should trim padded-tail tokens per sample."""
    dataset = _VariableDurationStimulusDataset()
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate)
    spec = IdentityTokenizerSpec(window_length_seconds=1.0, bin_size_ms=100.0)

    def tokenizer_with_padded_tail_tokens(batch: BinnedSpikesInput) -> dict[str, torch.Tensor]:
        batch_size = batch.spike_matrix.shape[0]
        n_input_bins = batch.spike_matrix.shape[1]
        n_tokens = n_input_bins // 10
        token_offsets = torch.arange(n_tokens, dtype=torch.float64)
        token_start = batch.absolute_start[:, None] + token_offsets[None, :] * 0.1
        token_end = token_start + 0.1
        token_timestamps = (token_start + token_end) / 2.0
        return {
            "reconstruction": batch.spike_matrix.to(torch.float32),
            "latents": torch.zeros((batch_size, n_tokens, 1), dtype=torch.float32),
            "token_timestamps": token_timestamps,
            "token_start_time": token_start,
            "token_end_time": token_end,
        }

    segments = list(
        _iter_collated_validation_segments_with_per_stimulus_sampling(
            loader,
            tokenizer_with_padded_tail_tokens,
            tokenizer_spec=spec,
            interval_source="stimuli",
            grouping_field="image_id",
            interval_end_lag_ms=100.0,
            device=torch.device("cpu"),
        )
    )

    assert len(segments) == 1
    segment = segments[0]
    assert isinstance(segment.latents, list)
    assert [item.shape[0] for item in segment.latents] == [2, 4]
    np.testing.assert_array_equal(
        segment.behavioral_data_token_res["image_id"],  # type: ignore[index]
        np.asarray([7.0, 7.0, 11.0, 11.0, 11.0, 11.0]),
    )


def test_per_stimulus_snr_fixed_windows_keep_full_token_set_without_padding_trim():
    """Fixed-duration SNR windows should take the assert-only branch."""
    dataset = _LabelExtensionStimulusDataset()
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate)
    spec = IdentityTokenizerSpec(window_length_seconds=1.0, bin_size_ms=100.0)

    def tokenizer_with_two_tokens(batch: BinnedSpikesInput) -> dict[str, torch.Tensor]:
        batch_size = batch.spike_matrix.shape[0]
        n_input_bins = batch.spike_matrix.shape[1]
        n_tokens = n_input_bins // 10
        token_offsets = torch.arange(n_tokens, dtype=torch.float64)
        token_start = batch.absolute_start[:, None] + token_offsets[None, :] * 0.1
        token_end = token_start + 0.1
        token_timestamps = (token_start + token_end) / 2.0
        return {
            "reconstruction": batch.spike_matrix.to(torch.float32),
            "latents": torch.zeros((batch_size, n_tokens, 1), dtype=torch.float32),
            "token_timestamps": token_timestamps,
            "token_start_time": token_start,
            "token_end_time": token_end,
        }

    segments = list(
        _iter_collated_validation_segments_with_per_stimulus_sampling(
            loader,
            tokenizer_with_two_tokens,
            tokenizer_spec=spec,
            interval_source="stimuli",
            grouping_field="image_id",
            interval_end_lag_ms=None,
            fixed_window_duration_ms=200.0,
            label_extension_ms=100.0,
            device=torch.device("cpu"),
        )
    )

    assert len(segments) == 1
    segment = segments[0]
    assert isinstance(segment.latents, torch.Tensor)
    assert segment.latents.shape == (2, 2, 1)
    np.testing.assert_array_equal(
        segment.behavioral_data_token_res["image_id"],  # type: ignore[index]
        np.asarray([7.0, 7.0, 11.0, 11.0]),
    )


def test_collation_with_latents_and_timestamps_in_tokenizer_output():
    """When tokenizer_fun returns latents and token timestamps,
    segments should have token timestamps."""
    n_units = 3
    time_bins_per_window = 500
    n_windows = 2
    token_bin_size_ms = 50.0  # 50ms per token → 10 tokens per 0.5s window

    dataset = SimpleDataset(
        n_units=n_units, time_bins_per_window=time_bins_per_window, n_windows=n_windows
    )
    indices = [
        DatasetIndex("test_session", 0.0, 0.5),
        DatasetIndex("test_session", 0.5, 1.0),
    ]
    loader = DataLoader(dataset, batch_size=1, sampler=indices, collate_fn=collate)

    def tokenizer_with_latents_and_timestamps(batch: BinnedSpikesInput) -> dict:
        """Return reconstruction, latents, and token timestamps.

        Mimics the output of reconstruct_from_input(return_latents=True).

        """
        B = batch.spike_matrix.shape[0]
        T = batch.spike_matrix.shape[1]
        token_bin_size_sec = token_bin_size_ms / 1000.0
        n_tokens = int(T * (1.0 / batch.sampling_rate) / token_bin_size_sec)

        abs_start = batch.absolute_start.to(dtype=torch.float64)  # [B]
        indices_t = torch.arange(n_tokens, dtype=torch.float64)  # [T']

        token_start_time = abs_start[:, None] + indices_t[None, :] * token_bin_size_sec
        token_end_time = abs_start[:, None] + (indices_t[None, :] + 1) * token_bin_size_sec
        token_timestamps = abs_start[:, None] + (indices_t[None, :] + 0.5) * token_bin_size_sec

        return {
            "reconstruction": batch.spike_matrix,
            "latents": torch.randn(B, n_tokens, n_units, 2),
            "token_timestamps": token_timestamps,
            "token_start_time": token_start_time,
            "token_end_time": token_end_time,
        }

    segments = list(
        iter_collated_validation_segments(
            loader,
            tokenizer_with_latents_and_timestamps,
            token_bin_size_ms=token_bin_size_ms,
            device=torch.device("cpu"),
        )
    )

    assert len(segments) == 1
    segment = segments[0]

    # Should have token timestamps
    assert segment.token_start_times is not None
    assert segment.token_end_times is not None
    assert segment.token_timestamps is not None

    # 0.5s per window / 0.05s per token = 10 tokens per window × 2 windows = 20 tokens
    expected_tokens = 20
    assert segment.token_start_times.shape == (expected_tokens,)
    assert segment.token_timestamps.shape == (expected_tokens,)

    # Center times should be midpoints
    expected_centers = (segment.token_start_times + segment.token_end_times) / 2.0
    torch.testing.assert_close(
        segment.token_timestamps,
        expected_centers,
        atol=1e-9,
        rtol=0,
    )

    # Times should be monotonic and span [0, 1.0]
    assert abs(float(segment.token_start_times[0]) - 0.0) < 1e-6
    assert abs(float(segment.token_end_times[-1]) - 1.0) < 1e-6


def test_interval_windowing_respects_max_eval_seconds_cap() -> None:
    """When cumulative window_duration crosses max_eval_seconds, the
    accumulated buffer should flush as an ``is_partial=True`` segment and
    further windows for that session should be skipped.

    """
    n_units = 2
    sampling_rate = 10.0  # 10 Hz → 1.0 s / window
    n_time_bins = 10

    def make_window(abs_start: float, scale: int) -> BinnedSpikesInput:
        return BinnedSpikesInput(
            spike_matrix=scale * torch.ones(n_time_bins, n_units, dtype=torch.int64),
            pad_and_gap_mask=torch.ones(n_time_bins, n_units, dtype=torch.bool),
            sampling_rate=sampling_rate,
            unit_ids=torch.tensor([0, 1]),
            unit_ids_pad_mask=torch.tensor([1, 1], dtype=torch.bool),
            duration_sec=torch.tensor(1.0),
            session_id="capped_session",
            subject_id="subject",
            species="mouse",
            absolute_start=torch.tensor(abs_start, dtype=torch.float64),
            absolute_end=torch.tensor(abs_start + 1.0, dtype=torch.float64),
        )

    # Three 1.0 s windows. With max_eval_seconds=1.5, the cap should trigger
    # after the second window (cumulative=2.0 ≥ 1.5); the third should be
    # skipped.
    samples = [make_window(0.0, 1), make_window(1.0, 2), make_window(2.0, 3)]
    loader = DataLoader(samples, batch_size=1, collate_fn=collate)

    def identity_tokenizer(batch: BinnedSpikesInput) -> dict:
        batch_size, n_bins, _ = batch.spike_matrix.shape
        token_starts = torch.stack(
            [
                torch.arange(n_bins, dtype=torch.float64) / sampling_rate
                + float(batch.absolute_start[i].item())
                for i in range(batch_size)
            ]
        )
        token_ends = token_starts + (1.0 / sampling_rate)
        token_centers = (token_starts + token_ends) / 2.0
        latents = torch.stack(
            [torch.arange(n_bins, dtype=torch.float32).unsqueeze(-1) for _ in range(batch_size)]
        )
        return {
            "reconstruction": batch.spike_matrix,
            "latents": latents,
            "token_start_time": token_starts,
            "token_end_time": token_ends,
            "token_timestamps": token_centers,
        }

    segments = list(
        iter_collated_validation_segments(
            loader,
            identity_tokenizer,
            max_eval_seconds=1.5,
            device=torch.device("cpu"),
        )
    )

    # Exactly one segment, flushed at the cap as partial; the third window
    # must NOT appear in it. The buffered data spans windows 1 + 2 only, so
    # original.shape[1] == 2 × n_time_bins (not 3 × n_time_bins).
    assert len(segments) == 1, f"Expected one is_partial segment, got {len(segments)}"
    segment = segments[0]
    assert segment.is_partial is True
    assert segment.session_id == "capped_session"
    assert segment.original is not None
    assert segment.original.shape[1] == 2 * n_time_bins, (
        f"Cap should include windows 1+2 and drop window 3; got time dim "
        f"{segment.original.shape[1]} (expected {2 * n_time_bins})."
    )


def test_collation_adds_window_group_id_token_res_to_behavioral_data():
    n_time_bins = 10
    n_units = 3
    n_windows = 2
    batches = []
    for window_idx in range(n_windows):
        spike_matrix = torch.ones((1, n_time_bins, n_units), dtype=torch.int32)
        behavioral_data = [
            {
                "cursor_pos": np.full((n_time_bins, 1), float(window_idx), dtype=np.float64),
            }
        ]
        batches.append(
            BinnedSpikesInput(
                spike_matrix=spike_matrix,
                pad_and_gap_mask=torch.ones_like(spike_matrix),
                sampling_rate=10.0,
                unit_ids=torch.arange(n_units).unsqueeze(0),
                unit_ids_pad_mask=torch.ones(1, n_units),
                duration_sec=torch.tensor([1.0]),
                session_id="test_session",
                subject_id="test_subject",
                species="test_species",
                absolute_start=torch.tensor([float(window_idx)], dtype=torch.float64),
                absolute_end=torch.tensor([float(window_idx + 1)], dtype=torch.float64),
                behavioral_data=behavioral_data,
            )
        )

    loader = DataLoader(batches, batch_size=None)

    def tokenizer_with_window_timestamps(batch: BinnedSpikesInput) -> dict:
        batch_size, n_bins, _ = batch.spike_matrix.shape
        token_start_time = (
            batch.absolute_start[:, None]
            + torch.arange(n_bins, dtype=torch.float64)[None, :] * 0.1
        )
        token_end_time = token_start_time + 0.1
        token_timestamps = token_start_time + 0.05
        return {
            "reconstruction": batch.spike_matrix,
            "latents": torch.randn(batch_size, n_bins, n_units, 2),
            "token_timestamps": token_timestamps,
            "token_start_time": token_start_time,
            "token_end_time": token_end_time,
        }

    segments = list(
        iter_collated_validation_segments(
            loader,
            tokenizer_with_window_timestamps,
            token_bin_size_ms=100.0,
            device=torch.device("cpu"),
        )
    )

    assert len(segments) == 1
    segment = segments[0]
    assert segment.behavioral_data_token_res is not None
    assert "window_group_id_token_res" in segment.behavioral_data_token_res
    np.testing.assert_array_equal(
        segment.behavioral_data_token_res["window_group_id_token_res"],
        np.array([0] * n_time_bins + [1] * n_time_bins, dtype=np.int32),
    )


def _make_multiwindow_loader(
    behavioral_field_name: str,
    n_windows: int = 3,
    n_time_bins: int = 10,
    n_units: int = 3,
) -> DataLoader:
    """Build a single-session DataLoader with one behavioral field per window."""
    batches = []
    for window_idx in range(n_windows):
        spike_matrix = torch.ones((1, n_time_bins, n_units), dtype=torch.int32)
        behavioral_data = [
            {
                behavioral_field_name: np.full(
                    (n_time_bins, 1), float(window_idx), dtype=np.float64
                ),
            }
        ]
        batches.append(
            BinnedSpikesInput(
                spike_matrix=spike_matrix,
                pad_and_gap_mask=torch.ones_like(spike_matrix),
                sampling_rate=10.0,
                unit_ids=torch.arange(n_units).unsqueeze(0),
                unit_ids_pad_mask=torch.ones(1, n_units),
                duration_sec=torch.tensor([1.0]),
                session_id="test_session",
                subject_id="test_subject",
                species="test_species",
                absolute_start=torch.tensor([float(window_idx)], dtype=torch.float64),
                absolute_end=torch.tensor([float(window_idx + 1)], dtype=torch.float64),
                behavioral_data=behavioral_data,
            )
        )
    return DataLoader(batches, batch_size=None)


def _identity_tokenizer_with_timestamps(batch: BinnedSpikesInput) -> dict:
    batch_size, n_bins, n_units = batch.spike_matrix.shape
    token_start_time = (
        batch.absolute_start[:, None] + torch.arange(n_bins, dtype=torch.float64)[None, :] * 0.1
    )
    token_end_time = token_start_time + 0.1
    token_timestamps = token_start_time + 0.05
    return {
        "reconstruction": batch.spike_matrix,
        "latents": torch.randn(batch_size, n_bins, n_units, 2),
        "token_timestamps": token_timestamps,
        "token_start_time": token_start_time,
        "token_end_time": token_end_time,
    }


def test_max_eval_seconds_bypassed_when_decoding_target_field_present():
    """When ``decoding_target_fields`` matches a key in the window's behavioral
    data, ``max_eval_seconds`` must be silently bypassed for that session."""
    loader = _make_multiwindow_loader(behavioral_field_name="cursor_pos", n_windows=3)

    segments = list(
        iter_collated_validation_segments(
            loader,
            _identity_tokenizer_with_timestamps,
            token_bin_size_ms=100.0,
            device=torch.device("cpu"),
            max_eval_seconds=0.5,  # would normally fire after 1st window (1s)
            decoding_target_fields={"cursor_pos"},
        )
    )

    assert len(segments) == 1, f"Expected 1 segment when bypassed, got {len(segments)}"
    segment = segments[0]
    assert not segment.is_partial, (
        "Segment was flagged is_partial=True even though decoding_target_fields "
        "matched the behavioral data — cap should have been bypassed."
    )
    # All 3 windows × 10 bins survived
    assert segment.original.shape[1] == 30


def test_max_eval_seconds_applied_when_decoding_target_field_absent():
    """When the registered ``decoding_target_fields`` are not present in the
    session's behavioral data, ``max_eval_seconds`` must still cap the session."""
    loader = _make_multiwindow_loader(behavioral_field_name="other_field", n_windows=3)

    segments = list(
        iter_collated_validation_segments(
            loader,
            _identity_tokenizer_with_timestamps,
            token_bin_size_ms=100.0,
            device=torch.device("cpu"),
            max_eval_seconds=0.5,  # fires after the first 1s window
            decoding_target_fields={"cursor_pos"},
        )
    )

    assert len(segments) == 1, f"Expected 1 segment when capped, got {len(segments)}"
    segment = segments[0]
    assert segment.is_partial, (
        "Segment should be flagged is_partial=True when max_eval_seconds fires "
        "and the session lacks any decoding target field."
    )
    # Only the first window's bins survived; subsequent windows skipped
    assert segment.original.shape[1] == 10


def test_max_eval_seconds_skips_tokenizer_calls_after_cap():
    """Once a session is capped, subsequent batches for that same session
    should fast-skip — the tokenizer must not be invoked on capped batches."""
    n_windows = 5
    loader = _make_multiwindow_loader(behavioral_field_name="other_field", n_windows=n_windows)

    call_count = {"n": 0}

    def counting_tokenizer(batch: BinnedSpikesInput) -> dict:
        call_count["n"] += 1
        return _identity_tokenizer_with_timestamps(batch)

    segments = list(
        iter_collated_validation_segments(
            loader,
            counting_tokenizer,
            token_bin_size_ms=100.0,
            device=torch.device("cpu"),
            max_eval_seconds=0.5,  # fires after the first 1s window
        )
    )

    assert len(segments) == 1, f"Expected 1 segment when capped, got {len(segments)}"
    # First batch (window 0) tokenizes and triggers the cap; remaining 4 batches
    # belong to the same already-capped session and must be fast-skipped.
    assert call_count["n"] == 1, (
        f"Tokenizer was called {call_count['n']} times for {n_windows} batches "
        "after a 1-window cap — fast-path is not skipping capped-session batches."
    )


def _make_binned_sample(
    session_id: str,
    n_units: int,
    time_bins: int = 1000,
) -> BinnedSpikesInput:
    spike_matrix = torch.randint(0, 2, (time_bins, n_units), dtype=torch.int32)
    pad_and_gap_mask = torch.ones_like(spike_matrix)
    return BinnedSpikesInput(
        spike_matrix=spike_matrix,
        pad_and_gap_mask=pad_and_gap_mask,
        sampling_rate=1000.0,
        unit_ids=torch.arange(n_units, dtype=torch.int32),
        unit_ids_pad_mask=torch.ones(n_units, dtype=torch.int32),
        duration_sec=torch.tensor(time_bins / 1000.0),
        session_id=session_id,
        subject_id="test_subject",
        species="test_species",
        absolute_start=torch.tensor(0.0, dtype=torch.float64),
        absolute_end=torch.tensor(time_bins / 1000.0, dtype=torch.float64),
        behavioral_data=None,
    )


def test_collation_accepts_per_sample_ragged_batch_latents():
    """Batched outputs may carry a per-sample latent list when widths differ."""
    spike_matrix = torch.zeros((2, 1000, 130), dtype=torch.int32)
    spike_matrix[0, :, :64] = torch.randint(0, 2, (1000, 64), dtype=torch.int32)
    spike_matrix[1] = torch.randint(0, 2, (1000, 130), dtype=torch.int32)
    unit_ids = torch.arange(130, dtype=torch.int32).unsqueeze(0).expand(2, -1).clone()
    unit_ids_pad_mask = torch.zeros((2, 130), dtype=torch.int32)
    unit_ids_pad_mask[0, :64] = 1
    unit_ids_pad_mask[1, :] = 1
    batch = BinnedSpikesInput(
        spike_matrix=spike_matrix,
        pad_and_gap_mask=torch.ones_like(spike_matrix),
        sampling_rate=1000.0,
        unit_ids=unit_ids,
        unit_ids_pad_mask=unit_ids_pad_mask,
        duration_sec=torch.tensor([1.0, 1.0]),
        session_id=("session_small", "session_large"),
        subject_id=("test_subject", "test_subject"),
        species=("test_species", "test_species"),
        absolute_start=torch.tensor([0.0, 0.0], dtype=torch.float64),
        absolute_end=torch.tensor([1.0, 1.0], dtype=torch.float64),
        behavioral_data=None,
    )
    loader = DataLoader([batch], batch_size=None)

    def tokenizer_with_ragged_latents(batch: BinnedSpikesInput) -> dict:
        token_start_time = torch.arange(10, dtype=torch.float64)[None, :] * 0.1
        token_start_time = token_start_time.repeat(2, 1)
        token_end_time = token_start_time + 0.1
        token_timestamps = token_start_time + 0.05
        return {
            "reconstruction": batch.spike_matrix.float(),
            "latents": [
                torch.zeros((10, 64, 3), dtype=torch.float32),
                torch.zeros((10, 192, 3), dtype=torch.float32),
            ],
            "token_timestamps": token_timestamps,
            "token_start_time": token_start_time,
            "token_end_time": token_end_time,
        }

    segments = list(
        iter_collated_validation_segments(
            loader,
            tokenizer_with_ragged_latents,
            token_bin_size_ms=100.0,
            device=torch.device("cpu"),
        )
    )

    assert len(segments) == 2
    by_session = {segment.session_id: segment for segment in segments}
    assert by_session["session_small"].latents is not None
    assert by_session["session_large"].latents is not None
    assert by_session["session_small"].latents.shape == (1, 10, 64, 3)
    assert by_session["session_large"].latents.shape == (1, 10, 192, 3)


def test_extract_metadata_converts_sampling_rate_to_python_scalar():
    """Collated batches arrive with
    ``sampling_rate`` as a torch.Tensor. ``_extract_metadata`` must convert it
    to a Python float so segment.metadata never carries a tensor — otherwise
    the tensor rides in the dict, gets shipped to fork-pool workers, and
    crashes them with "Cannot re-initialize CUDA in forked subprocess" if it
    happens to be CUDA-resident.
    """
    from ntt.evaluation.utils.validation_collation import _extract_metadata

    batch = BinnedSpikesInput(
        spike_matrix=torch.zeros(1, 10, 5, dtype=torch.int8),
        pad_and_gap_mask=torch.ones(1, 10, 5, dtype=torch.bool),
        sampling_rate=torch.tensor([100.0]),  # tensor — what the collator emits
        unit_ids=torch.zeros(1, 5, dtype=torch.long),
        unit_ids_pad_mask=torch.ones(1, 5, dtype=torch.bool),
        duration_sec=torch.tensor([0.1]),
        session_id="test",
        subject_id="x",
        species="y",
        absolute_start=torch.zeros(1),
        absolute_end=torch.full((1,), 0.1),
    )

    meta = _extract_metadata(batch, index=0)

    assert "sampling_rate" in meta
    assert not isinstance(meta["sampling_rate"], torch.Tensor), (
        f"sampling_rate must be a Python scalar, got {type(meta['sampling_rate'])}"
    )
    assert isinstance(meta["sampling_rate"], float)
    assert meta["sampling_rate"] == 100.0


def test_aggregate_metric_stack_propagates_ndarray_metadata():
    """Per-unit ``realisation_std`` must survive metadata aggregation.

    Per-unit metrics (``spike_count_correlation``, ``binned_rate_difference``,
    ``temporal_precision``, ``spike_detection_f1``, ...) must keep
    ``realisation_std`` in the aggregated metadata, like
    scalar metrics (``subspace_overlap``). A metadata aggregator whose numeric-scalar
    gate (``isinstance(v, (int, float, np.number))``) would silently drop any
    np.ndarray-valued key.
    For PER_UNIT metrics, ``_average_realisation_reports`` produces
    ``realisation_std`` as an ndarray (one std per unit), so the key
    would be dropped during ``aggregate_segment_reports`` and the user
    would see effectively-zero (or absent) cross-realisation variance.
    """
    from dataclasses import dataclass

    from ntt.evaluation.config.schema import AggregationStrategy
    from ntt.evaluation.core.types import (
        MetricCategory,
        MetricDirection,
        MetricLevel,
        MetricResult,
    )
    from ntt.evaluation.utils.validation_collation import (
        CollatedValidationSegment,
        aggregate_segment_reports,
    )

    @dataclass
    class _Report:
        # Minimal stand-in for EvaluationReport — only the fields touched
        # by aggregate_segment_reports are needed.
        results: dict
        skipped_metrics: dict
        config: dict
        original_data_type: DataType | None = None
        reconstructed_data_type: DataType | None = None

    # Build 3 segments. Each has the same per-unit metric, with a different
    # realisation_std array — this is the post-_average_realisation_reports
    # state for a per-unit metric.
    n_units = 6
    segments_with_reports = []
    for seg_idx in range(3):
        per_unit_value = np.full(n_units, 0.5 + 0.01 * seg_idx)
        per_unit_realisation_std = np.full(n_units, 0.1 + 0.05 * seg_idx)

        result = MetricResult(
            name="firing_rate_error",
            raw_value=per_unit_value,
            category=MetricCategory.SINGLE_UNIT_STATISTICS,
            direction=MetricDirection.LOWER_IS_BETTER,
            level=MetricLevel.PER_UNIT,
            metadata={
                "realisation_std": per_unit_realisation_std,
                "n_realisations": 5,
            },
        )

        report = _Report(
            results={"firing_rate_error": result},
            skipped_metrics={},
            config={},
        )

        seg = CollatedValidationSegment(
            session_id="sess0",
            original=None,
            reconstructed=None,
            pad_and_gap_mask=None,
            metadata={"sampling_rate": 1000.0, "subject_id": "x", "species": "y"},
            segment_index=seg_idx,
            is_partial=False,
            original_shape=(n_units, 200),
        )
        segments_with_reports.append((seg, {"report": report, "metadata": {}}))

    aggregated = aggregate_segment_reports(
        segments_with_reports, strategy=AggregationStrategy.MEAN
    )
    assert "firing_rate_error" in aggregated.results
    md = aggregated.results["firing_rate_error"].metadata

    # The ndarray-valued realisation_std must survive aggregation as a
    # numeric scalar (matching the scalar-collapse contract used for
    # raw_value via _scalar_metric_value).
    assert "realisation_std" in md, (
        f"realisation_std dropped from per-unit metric metadata: keys={list(md)}"
    )
    assert isinstance(md["realisation_std"], float), (
        f"realisation_std should be float after aggregation, got {type(md['realisation_std'])}"
    )
    # Element-wise mean across 3 segments of 0.10/0.15/0.20, then scalar
    # collapse via nanmean → 0.15.
    assert md["realisation_std"] == pytest.approx(0.15)


def test_aggregate_metric_stack_preserves_scalar_realisation_std():
    """Sanity: scalar realisation_std (population/scalar metrics) propagates
    alongside the ndarray-valued case."""
    from dataclasses import dataclass

    from ntt.evaluation.config.schema import AggregationStrategy
    from ntt.evaluation.core.types import (
        MetricCategory,
        MetricDirection,
        MetricLevel,
        MetricResult,
    )
    from ntt.evaluation.utils.validation_collation import (
        CollatedValidationSegment,
        aggregate_segment_reports,
    )

    @dataclass
    class _Report:
        results: dict
        skipped_metrics: dict
        config: dict
        original_data_type: DataType | None = None
        reconstructed_data_type: DataType | None = None

    segments_with_reports = []
    for seg_idx, (val, std) in enumerate([(0.95, 0.01), (0.96, 0.02), (0.97, 0.03)]):
        result = MetricResult(
            name="subspace_overlap",
            raw_value=val,
            category=MetricCategory.POPULATION_STRUCTURE,
            direction=MetricDirection.HIGHER_IS_BETTER,
            level=MetricLevel.POPULATION,
            metadata={"realisation_std": std, "n_realisations": 5},
        )
        report = _Report(
            results={"subspace_overlap": result},
            skipped_metrics={},
            config={},
        )
        seg = CollatedValidationSegment(
            session_id="sess0",
            original=None,
            reconstructed=None,
            pad_and_gap_mask=None,
            metadata={"sampling_rate": 1000.0, "subject_id": "x", "species": "y"},
            segment_index=seg_idx,
            is_partial=False,
            original_shape=(3, 200),
        )
        segments_with_reports.append((seg, {"report": report, "metadata": {}}))

    aggregated = aggregate_segment_reports(
        segments_with_reports, strategy=AggregationStrategy.MEAN
    )
    md = aggregated.results["subspace_overlap"].metadata
    assert md["realisation_std"] == pytest.approx(0.02)  # mean of 0.01/0.02/0.03


def _make_binned_sample(
    session_id: str,
    n_units: int,
    time_bins: int = 1000,
) -> BinnedSpikesInput:
    spike_matrix = torch.randint(0, 2, (time_bins, n_units), dtype=torch.int32)
    pad_and_gap_mask = torch.ones_like(spike_matrix)
    return BinnedSpikesInput(
        spike_matrix=spike_matrix,
        pad_and_gap_mask=pad_and_gap_mask,
        sampling_rate=1000.0,
        unit_ids=torch.arange(n_units, dtype=torch.int32),
        unit_ids_pad_mask=torch.ones(n_units, dtype=torch.int32),
        duration_sec=torch.tensor(time_bins / 1000.0),
        session_id=session_id,
        subject_id="test_subject",
        species="test_species",
        absolute_start=torch.tensor(0.0, dtype=torch.float64),
        absolute_end=torch.tensor(time_bins / 1000.0, dtype=torch.float64),
        behavioral_data=None,
    )


def test_collation_accepts_per_sample_ragged_batch_latents():
    """Batched outputs may carry a per-sample latent list when widths differ."""
    spike_matrix = torch.zeros((2, 1000, 130), dtype=torch.int32)
    spike_matrix[0, :, :64] = torch.randint(0, 2, (1000, 64), dtype=torch.int32)
    spike_matrix[1] = torch.randint(0, 2, (1000, 130), dtype=torch.int32)
    unit_ids = torch.arange(130, dtype=torch.int32).unsqueeze(0).expand(2, -1).clone()
    unit_ids_pad_mask = torch.zeros((2, 130), dtype=torch.int32)
    unit_ids_pad_mask[0, :64] = 1
    unit_ids_pad_mask[1, :] = 1
    batch = BinnedSpikesInput(
        spike_matrix=spike_matrix,
        pad_and_gap_mask=torch.ones_like(spike_matrix),
        sampling_rate=1000.0,
        unit_ids=unit_ids,
        unit_ids_pad_mask=unit_ids_pad_mask,
        duration_sec=torch.tensor([1.0, 1.0]),
        session_id=("session_small", "session_large"),
        subject_id=("test_subject", "test_subject"),
        species=("test_species", "test_species"),
        absolute_start=torch.tensor([0.0, 0.0], dtype=torch.float64),
        absolute_end=torch.tensor([1.0, 1.0], dtype=torch.float64),
        behavioral_data=None,
    )
    loader = DataLoader([batch], batch_size=None)

    def tokenizer_with_ragged_latents(batch: BinnedSpikesInput) -> dict:
        token_start_time = torch.arange(10, dtype=torch.float64)[None, :] * 0.1
        token_start_time = token_start_time.repeat(2, 1)
        token_end_time = token_start_time + 0.1
        token_timestamps = token_start_time + 0.05
        return {
            "reconstruction": batch.spike_matrix.float(),
            "latents": [
                torch.zeros((10, 64, 3), dtype=torch.float32),
                torch.zeros((10, 192, 3), dtype=torch.float32),
            ],
            "token_timestamps": token_timestamps,
            "token_start_time": token_start_time,
            "token_end_time": token_end_time,
        }

    segments = list(
        iter_collated_validation_segments(
            loader,
            tokenizer_with_ragged_latents,
            token_bin_size_ms=100.0,
            device=torch.device("cpu"),
        )
    )

    assert len(segments) == 2
    by_session = {segment.session_id: segment for segment in segments}
    assert by_session["session_small"].latents is not None
    assert by_session["session_large"].latents is not None
    assert by_session["session_small"].latents.shape == (1, 10, 64, 3)
    assert by_session["session_large"].latents.shape == (1, 10, 192, 3)


def test_concatenate_segments_handles_none_original_and_reconstructed():
    """Concatenating tokenize-only / interval-windowed segments must not crash.

    Tokenize-only tokenizers emit ``reconstructed=None``; the interval-windowed
    path emits both ``original=None`` and ``reconstructed=None``. When
    ``concatenate_sessions=True`` folds >1 such segment together, the combine
    step must tolerate the ``None`` sides (mirroring the latents/behavioral
    guards) rather than calling ``torch.cat`` on a list of ``None``.
    """
    from ntt.evaluation.utils.validation_collation import (
        CollatedValidationSegment,
        _concatenate_segments,
    )

    def _seg(idx: int) -> CollatedValidationSegment:
        return CollatedValidationSegment(
            session_id="sessA",
            original=None,
            reconstructed=None,
            pad_and_gap_mask=None,
            metadata={"sampling_rate": 1000.0},
            segment_index=idx,
            is_partial=False,
            latents=torch.zeros(2, 3),
            token_timestamps=torch.arange(2, dtype=torch.float32),
        )

    combined = _concatenate_segments([_seg(0), _seg(1)])

    assert combined.original is None
    assert combined.reconstructed is None
    # The non-None sides combine normally.
    assert combined.latents is not None and combined.latents.shape[0] == 4


def test_pad_window_returns_3tuple_for_none_recon_when_interval_windowed():
    """reconstructed=None must still return the (window, n, is_large) 3-tuple under
    is_interval_windowed, matching the caller's unpacking (latents-only with timestamps)."""
    import torch

    from ntt.evaluation.utils.validation_collation import _pad_window_for_small_mismatch

    window = {"original": torch.zeros(1, 10, 2), "reconstructed": None}
    result = _pad_window_for_small_mismatch(
        window, sampling_rate=1000.0, is_interval_windowed=True
    )
    assert len(result) == 3
    w, n_padded, is_large = result
    assert n_padded == 0 and is_large is False
    # and the 2-tuple shape is preserved when not interval-windowed
    r2 = _pad_window_for_small_mismatch(window, sampling_rate=1000.0, is_interval_windowed=False)
    assert len(r2) == 2


class _UnitTaggedDataset(Dataset):
    """Binned counts encode the unit id: ``spikes[t, u] == unit_ids[u]``.

    A collated segment that mixes unit subsets therefore has a row whose value
    changes over time. ``n_units`` is per session (keyed by recording id);
    ``unit_ids_per_window`` (keyed by ``(session, window start)``) lets a test
    emulate per-window UnitDropout subsets.
    """

    def __init__(
        self,
        n_units: dict[str, int],
        bins_per_window: int = 10,
        unit_ids_per_window: dict[tuple[str, float], list[int]] | None = None,
    ) -> None:
        self.n_units = n_units
        self.bins_per_window = bins_per_window
        self.unit_ids_per_window = unit_ids_per_window or {}

    def __len__(self) -> int:
        return len(self.n_units)

    def __getitem__(self, index: DatasetIndex) -> BinnedSpikesInput:
        session = str(index.recording_id)
        start, end = float(index.start), float(index.end)
        unit_ids = torch.tensor(
            self.unit_ids_per_window.get((session, start), list(range(self.n_units[session]))),
            dtype=torch.int64,
        )
        spike_matrix = unit_ids.to(torch.int32).unsqueeze(0).expand(self.bins_per_window, -1)
        # pad/pad2d-wrapped like the real transform output (neuron_split_collate needs .obj).
        return BinnedSpikesInput(
            spike_matrix=pad2d(spike_matrix.contiguous()),
            pad_and_gap_mask=pad2d(torch.ones_like(spike_matrix, dtype=torch.int32)),
            sampling_rate=self.bins_per_window / (end - start),
            unit_ids=pad(unit_ids),
            unit_ids_pad_mask=pad(torch.ones(len(unit_ids), dtype=torch.int32)),
            duration_sec=torch.tensor(end - start, dtype=torch.float64),
            session_id=session,
            subject_id="subject",
            species="test_species",
            absolute_start=torch.tensor(start, dtype=torch.float64),
            absolute_end=torch.tensor(end, dtype=torch.float64),
            behavioral_data=None,
            behavioral_intervals=None,
        )


def _collate_unit_tagged(
    dataset: _UnitTaggedDataset,
    n_windows: int,
    collate_fn,
    *,
    batch_size: int = 1,
    stride: float = 1.0,
) -> list:
    """Collate ``n_windows`` 1-s windows (start ``i * stride``) per session, sessions in order."""
    indices = [
        DatasetIndex(session, i * stride, i * stride + 1.0)
        for session in dataset.n_units
        for i in range(n_windows)
    ]
    loader = DataLoader(dataset, batch_size=batch_size, sampler=indices, collate_fn=collate_fn)
    return list(
        iter_collated_validation_segments(
            loader,
            lambda batch: {"reconstruction": batch.spike_matrix},
            device=torch.device("cpu"),
        )
    )


def _assert_segments_cover_units_without_mixing(
    segments: list, n_units: int, n_windows: int, bins_per_window: int = 10
) -> None:
    covered_bins = torch.zeros(n_units, dtype=torch.int64)
    for segment in segments:
        rows = segment.original
        # Each row must be one unit for the whole segment (no unit-subset mixing).
        assert torch.equal(rows, rows[:, :1].expand_as(rows)), (
            f"segment {segment.segment_index} mixes unit subsets over time"
        )
        covered_bins[rows[:, 0].long()] += rows.shape[1]
    assert covered_bins.tolist() == [n_windows * bins_per_window] * n_units


@pytest.mark.parametrize("n_units", [6, 8], ids=["unequal_chunks", "equal_chunks"])
@pytest.mark.parametrize(
    ("batch_size", "stride"),
    [(1, 1.0), (2, 1.0), (1, 0.5)],
    ids=["bs1", "bs2", "overlapping_windows"],
)
def test_collation_neuron_split_multiwindow_session_keeps_chunks_apart(
    n_units: int, batch_size: int, stride: float
):
    """neuron_split_collate emits [w0·A, w0·B, w1·A, w1·B, ...].

    The same-extent guard must flush w0·A on w0·B without appending w1·A to w0·B.
    Unequal chunk sizes would crash torch.cat (ldns_perceiver: 257 vs 512); equal
    sizes would silently concatenate different unit subsets into one segment.
    Chunk segments are flushed before any overlap trim, so overlapping windows
    keep every chunk whole.
    """
    from ntt.core.data.components.collate import make_neuron_split_collate_fn

    n_windows = 3
    segments = _collate_unit_tagged(
        _UnitTaggedDataset({"split_session": n_units}),
        n_windows,
        make_neuron_split_collate_fn(max_neurons=4),
        batch_size=batch_size,
        stride=stride,
    )

    _assert_segments_cover_units_without_mixing(segments, n_units, n_windows)


def test_collation_without_neuron_split_still_concatenates_changing_unit_subsets():
    """Scope guard for the neuron-split handling: without a split, per-window unit
    subsets (uniform UnitDropout under eval max_neurons) keep collating into
    one segment."""
    dataset = _UnitTaggedDataset(
        {"plain": 3},
        unit_ids_per_window={("plain", 1.0): [3, 4, 5], ("plain", 2.0): [0, 4, 5]},
    )

    segments = _collate_unit_tagged(dataset, n_windows=3, collate_fn=collate)

    assert len(segments) == 1
    assert segments[0].original.shape == (3, 30)


def test_collation_neuron_split_flag_resets_on_session_change():
    """A split session must not put the next (unsplit) session into
    flush-on-unit-change mode: its changing per-window subsets stay one segment."""
    from ntt.core.data.components.collate import make_neuron_split_collate_fn

    dataset = _UnitTaggedDataset(
        {"split_session": 6, "plain": 3},
        unit_ids_per_window={("plain", 1.0): [3, 4, 5], ("plain", 2.0): [0, 4, 5]},
    )

    segments = _collate_unit_tagged(
        dataset, n_windows=3, collate_fn=make_neuron_split_collate_fn(max_neurons=4)
    )

    split_segments = [s for s in segments if s.session_id == "split_session"]
    plain_segments = [s for s in segments if s.session_id == "plain"]
    _assert_segments_cover_units_without_mixing(split_segments, n_units=6, n_windows=3)
    assert len(plain_segments) == 1
    assert plain_segments[0].original.shape == (3, 30)


if __name__ == "__main__":
    print("Running test_collation_shape_multiple_windows...")
    test_collation_shape_multiple_windows()
    print("Passed\n")

    print("Running test_collation_handles_reconstruction_output...")
    test_collation_handles_reconstruction_output()
    print("Passed\n")

    print("Running test_collation_with_timestamps_in_tokenizer_output...")
    test_collation_with_timestamps_in_tokenizer_output()
    print("Passed\n")

    print("Running test_collation_with_latents_and_timestamps_in_tokenizer_output...")
    test_collation_with_latents_and_timestamps_in_tokenizer_output()
    print("Passed\n")

    print("All tests passed!")


def test_aggregate_metric_stack_preserves_parameters_and_rejects_mismatches():
    from dataclasses import replace

    from ntt.evaluation.config.schema import AggregationStrategy
    from ntt.evaluation.core.types import (
        EvaluationReport,
        MetricCategory,
        MetricDirection,
        MetricLevel,
        MetricResult,
    )
    from ntt.evaluation.utils.validation_collation import (
        _aggregate_evaluation_reports,
        _aggregate_metric_stack,
    )

    template = MetricResult(
        name="behavioral_regression",
        category=MetricCategory.DOWNSTREAM_DECODING,
        direction=MetricDirection.HIGHER_IS_BETTER,
        level=MetricLevel.POPULATION,
        raw_value=0.2,
    )
    metadata = {
        "chunk_window_ms": 200,
        "ridge_alpha": 1,
        "mean_samples_train": 10,
        "best_chunk_window_ms": 200,
    }
    result = _aggregate_metric_stack(
        template,
        [0.2, 0.4],
        [1, 1],
        [metadata, {**metadata, "mean_samples_train": 20, "best_chunk_window_ms": 1000}],
        AggregationStrategy.MEAN,
        count_key="n_sessions",
        count=2,
        metadata_fields_that_must_match={"ridge_alpha"},
    )
    assert result.raw_value == pytest.approx(0.3)
    assert result.metadata["chunk_window_ms"] == 200
    assert isinstance(result.metadata["chunk_window_ms"], int)
    assert result.metadata["mean_samples_train"] == 15
    assert "best_chunk_window_ms" not in result.metadata
    assert metadata["best_chunk_window_ms"] == 200
    for second in (
        {**metadata, "chunk_window_ms": 1000},
        {**metadata, "ridge_alpha": 2},
        {"mean_samples_train": 20},
    ):
        with pytest.raises(ValueError, match="parameter .* differs or is missing"):
            _aggregate_metric_stack(
                template,
                [0.2, 0.4],
                [1, 1],
                [metadata, second],
                AggregationStrategy.MEAN,
                count_key="n_sessions",
                count=2,
                metadata_fields_that_must_match={"ridge_alpha"},
            )

    for config in ({"metrics": None}, {"metrics": {"behavioral_regression": {"n_splits": 3}}}):
        reports = [
            EvaluationReport(
                results={
                    "behavioral_regression": replace(
                        template, metadata={"chunk_window_ms": 200, "n_splits": folds}
                    )
                },
                config=config,
            )
            for folds in (2, 3)
        ]
        combined = _aggregate_evaluation_reports(reports, [1, 1], AggregationStrategy.MEAN)
        assert combined.results["behavioral_regression"].metadata["chunk_window_ms"] == 200
        assert combined.results["behavioral_regression"].metadata["n_splits"] == 2.5
