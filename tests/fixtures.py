"""Test fixtures and utilities for creating toy data."""

from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch_brain.data.dataset import DatasetIndex

from ntt.core.components.types import BinnedSpikesInput
from ntt.core.data.datamodule import collate


class WindowedSpikeDataset(Dataset):
    """Dataset that returns BinnedSpikesInput from a full spike tensor.

    The spike tensor is stored as (units, time) and windowed along time.
    Returns BinnedSpikesInput with shape (time, units) per the training convention.
    """

    def __init__(
        self,
        spikes: torch.Tensor,
        sampling_rate: float,
        session_id: str,
        length: int,
        window_size_samples: int | None = None,
    ):
        """Initialize the dataset.

        Args:
            spikes: Full spike tensor of shape (units, time)
            sampling_rate: Sampling rate in Hz
            session_id: Session identifier
            length: Number of windows in the dataset
            window_size_samples: Size of each window in samples. If None, inferred
                from length and total time.
        """
        self.spikes = spikes
        self.sampling_rate = sampling_rate
        self.session_id = session_id
        self.length = length
        self.window_size_samples = window_size_samples or (spikes.shape[1] // length)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: DatasetIndex | int) -> BinnedSpikesInput:
        if isinstance(index, DatasetIndex):
            start = int(round(index.start * self.sampling_rate))
            end = int(round(index.end * self.sampling_rate))
        else:
            start = index * self.window_size_samples
            end = start + self.window_size_samples

        # spikes is (units, time), window along time axis
        window = self.spikes[:, start:end]  # Shape: (units, time)
        # Transpose to (time, units) for BinnedSpikesInput
        spike_matrix = window.T.int()  # Shape: (time, units)

        n_units = spike_matrix.shape[1]
        pad_and_gap_mask = torch.ones_like(spike_matrix, dtype=torch.int32)
        unit_ids = torch.arange(n_units, dtype=torch.int32)
        unit_ids_pad_mask = torch.ones(n_units, dtype=torch.int32)
        duration_sec = torch.tensor((end - start) / self.sampling_rate)

        # Compute absolute start/end from the DatasetIndex (or integer index)
        if isinstance(index, DatasetIndex):
            abs_start = torch.tensor(index.start, dtype=torch.float64)
            abs_end = torch.tensor(index.end, dtype=torch.float64)
        else:
            abs_start = torch.tensor(start / self.sampling_rate, dtype=torch.float64)
            abs_end = torch.tensor(end / self.sampling_rate, dtype=torch.float64)

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
            behavioral_data=None,
        )


class SimpleDataset(Dataset):
    """Simple dataset that creates spike patterns based on window index.

    Returns BinnedSpikesInput with shape (time, units).
    """

    def __init__(self, n_units: int, time_bins_per_window: int, n_windows: int):
        self.n_units = n_units
        self.time_bins_per_window = time_bins_per_window
        self.n_windows = n_windows
        self.sampling_rate = 1000.0  # 1 kHz
        self.session_id = "test_session"

    def __len__(self) -> int:
        return self.n_windows

    @classmethod
    def with_stimuli(
        cls,
        *,
        session_id: str,
        stim_starts: list[float],
        stim_ends: list[float],
        image_ids: list[int],
        n_units: int,
        sampling_rate: float,
        duration: float,
        ignore_splits: bool = False,
        split: str | None = None,
        stim_in_split: str | None = None,
    ) -> "SimpleDataset":
        """Construct a fixture dataset with one session exposing stimulus intervals."""
        time_bins_per_window = int(round(duration * sampling_rate))
        dataset = cls(n_units=n_units, time_bins_per_window=time_bins_per_window, n_windows=1)
        dataset.sampling_rate = float(sampling_rate)
        dataset.session_id = session_id
        dataset.split = split
        dataset.recording_dict = {
            session_id: {
                "config": {
                    "ignore_splits": ignore_splits,
                }
            }
        }
        split_filters_out_stimuli = (
            split is not None
            and stim_in_split is not None
            and split != stim_in_split
            and not ignore_splits
        )
        stimuli = SimpleNamespace(
            start=np.asarray([] if split_filters_out_stimuli else stim_starts, dtype=np.float64),
            end=np.asarray([] if split_filters_out_stimuli else stim_ends, dtype=np.float64),
            image_id=np.asarray([] if split_filters_out_stimuli else image_ids),
        )
        dataset._recording_objects = {session_id: SimpleNamespace(stimuli=stimuli)}
        return dataset

    def _get_data_object(self, recording_id: str):
        """Return the synthetic session object for interval-aware tests."""
        return self._recording_objects[recording_id]

    def _get_data_config(self, recording_id: str) -> dict:
        """Mirror the NTTDataset config accessor for split-aware sampler tests."""
        return self.recording_dict[recording_id]["config"]

    def _should_ignore_splits(self, recording_id: str) -> bool:
        """Mirror the NTTDataset split-bypass accessor for split-aware sampler tests."""
        return self._get_data_config(recording_id).get("ignore_splits", False)

    def __getitem__(self, index: DatasetIndex | int) -> BinnedSpikesInput:
        spike_matrix = torch.zeros((self.time_bins_per_window, self.n_units), dtype=torch.int32)

        # Get window index from DatasetIndex or int
        if hasattr(index, "start"):
            window_idx = int(index.start * self.sampling_rate / self.time_bins_per_window)
        else:
            window_idx = index

        # Add a distinctive pattern based on window index
        pattern_start = (window_idx * 10) % (self.time_bins_per_window - 5)
        spike_matrix[pattern_start : pattern_start + 5, :] = 1

        pad_and_gap_mask = torch.ones_like(spike_matrix, dtype=torch.int32)
        unit_ids = torch.arange(self.n_units, dtype=torch.int32)
        unit_ids_pad_mask = torch.ones(self.n_units, dtype=torch.int32)
        duration_sec = torch.tensor(self.time_bins_per_window / self.sampling_rate)

        # Compute absolute start/end from the DatasetIndex or integer window index
        if hasattr(index, "start"):
            abs_start = torch.tensor(float(index.start), dtype=torch.float64)
            abs_end = torch.tensor(float(index.end), dtype=torch.float64)
        else:
            window_duration = self.time_bins_per_window / self.sampling_rate
            abs_start = torch.tensor(index * window_duration, dtype=torch.float64)
            abs_end = torch.tensor((index + 1) * window_duration, dtype=torch.float64)

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
            behavioral_data=None,
        )


def make_spike_tensor(
    n_units: int,
    total_bins: int,
    spike_positions: list[int] | None = None,
    random_spikes: bool = False,
    seed: int = 42,
) -> torch.Tensor:
    """Create a spike tensor of shape (units, time).

    Args:
        n_units: Number of units
        total_bins: Total number of time bins
        spike_positions: List of time bin indices where spikes occur for all units.
            If None, uses default positions [100, 500, 1500].
        random_spikes: If True, generate random sparse spikes instead
        seed: Random seed for random spike generation

    Returns:
        Spike tensor of shape (n_units, total_bins)
    """
    spikes = torch.zeros((n_units, total_bins))

    if random_spikes:
        rng = torch.Generator().manual_seed(seed)
        spikes = (torch.rand(n_units, total_bins, generator=rng) > 0.95).float()
    else:
        if spike_positions is None:
            spike_positions = [100, 500, 1500]
        # Clip positions to valid range
        spike_positions = [p for p in spike_positions if p < total_bins]
        spikes[:, spike_positions] = 1.0

    return spikes


def make_dataloader(
    spikes: torch.Tensor,
    sampling_rate: float,
    n_windows: int = 3,
    batch_size: int = 1,
    session_id: str = "test_session",
) -> DataLoader:
    """Create a DataLoader from a spike tensor.

    Args:
        spikes: Spike tensor of shape (units, time)
        sampling_rate: Sampling rate in Hz
        n_windows: Number of windows to create
        batch_size: Batch size for the DataLoader
        session_id: Session identifier

    Returns:
        DataLoader yielding BinnedSpikesInput batches
    """
    total_time = spikes.shape[1] / sampling_rate
    window_size = total_time / n_windows

    # Create indices for sequential sampling
    indices = [
        DatasetIndex(session_id, i * window_size, (i + 1) * window_size) for i in range(n_windows)
    ]

    dataset = WindowedSpikeDataset(spikes, sampling_rate, session_id, len(indices))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=indices,  # type: ignore[arg-type]
        collate_fn=collate,
    )
    return loader


def make_identity_tokenizer():
    """Create a tokenizer that returns input as reconstruction."""

    def tokenizer(batch: BinnedSpikesInput) -> dict[str, torch.Tensor]:
        return {"reconstruction": batch.spike_matrix}

    return tokenizer


def make_shifted_tokenizer(shift_bins: int):
    """Create a tokenizer that shifts spikes by a fixed number of bins.

    Args:
        shift_bins: Number of bins to shift (positive = forward in time)

    Returns:
        Tokenizer function
    """

    def tokenizer(batch: BinnedSpikesInput) -> dict[str, torch.Tensor]:
        spike_matrix = batch.spike_matrix  # Shape: (batch, time, units) or (time, units)
        shifted = torch.roll(spike_matrix, shifts=shift_bins, dims=-2)
        return {"reconstruction": shifted}

    return tokenizer


def make_zero_tokenizer():
    """Create a tokenizer that returns all zeros (no spikes)."""

    def tokenizer(batch: BinnedSpikesInput) -> dict[str, torch.Tensor]:
        return {"reconstruction": torch.zeros_like(batch.spike_matrix)}

    return tokenizer


def make_dummy_binned_input(
    n_neurons: int = 5,
    time_bins: int = 100,
    bin_size_ms: float = 1.0,
    batch_size: int = 1,
    session_id: str = "__test__",
) -> BinnedSpikesInput:
    """Create a minimal BinnedSpikesInput for testing.

    Args:
        n_neurons: Number of units
        time_bins: Number of time bins
        bin_size_ms: Bin size in milliseconds
        batch_size: Batch dimension size
        session_id: Session identifier string

    Returns:
        BinnedSpikesInput with zero spike data
    """
    sampling_rate = 1000.0 / bin_size_ms
    spike_matrix = torch.zeros((batch_size, time_bins, n_neurons), dtype=torch.int32)
    duration_val = time_bins * bin_size_ms / 1000.0
    return BinnedSpikesInput(
        spike_matrix=spike_matrix,
        pad_and_gap_mask=torch.ones_like(spike_matrix, dtype=torch.int32),
        sampling_rate=sampling_rate,
        unit_ids=torch.arange(n_neurons, dtype=torch.int32).unsqueeze(0).expand(batch_size, -1),
        unit_ids_pad_mask=torch.ones((batch_size, n_neurons), dtype=torch.int32),
        duration_sec=torch.full((batch_size,), duration_val),
        session_id=tuple([session_id] * batch_size),
        subject_id=tuple(["test_subject"] * batch_size),
        species=tuple(["test_species"] * batch_size),
        absolute_start=torch.zeros(batch_size, dtype=torch.float64),
        absolute_end=torch.full((batch_size,), duration_val, dtype=torch.float64),
        behavioral_data=None,
    )
