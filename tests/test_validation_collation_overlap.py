"""Tests for the validation collation overlap removal logic."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from torch_brain.data.dataset import DatasetIndex

from ntt.core.components.types import BinnedSpikesInput
from ntt.core.data.datamodule import collate
from ntt.evaluation.utils.validation_collation import (
    _trim_input_overlap,
    _trim_token_overlap,
    iter_collated_validation_segments,
)
from tests.fixtures import WindowedSpikeDataset


class WindowSampler(Sampler):
    def __init__(self, indices: list[DatasetIndex]):
        self._indices = indices

    def __iter__(self):
        yield from self._indices

    def __len__(self) -> int:
        return len(self._indices)


def test_collation_drops_overlap_window() -> None:
    session_id = "session_1"
    sampling_rate = 1000.0
    duration_seconds = 2.5
    total_bins = int(duration_seconds * sampling_rate)
    full_spikes = torch.zeros((total_bins, 1), dtype=torch.int8)
    full_spikes[[100, 1100, 1600, 2400], 0] = 1

    indices = [
        DatasetIndex(session_id, 0.0, 1.0),
        DatasetIndex(session_id, 1.0, 2.0),
        DatasetIndex(session_id, 1.5, 2.5),
    ]
    sampler = WindowSampler(indices)
    # WindowedSpikeDataset from tests.fixtures expects spikes shaped as (units, time)
    dataset = WindowedSpikeDataset(full_spikes.T, sampling_rate, session_id, len(indices))
    loader = DataLoader(dataset, sampler=sampler, batch_size=1, collate_fn=collate)

    def tokenizer_fun(batch: BinnedSpikesInput) -> dict[str, torch.Tensor]:
        return {"reconstruction": batch.spike_matrix}

    segments = list(
        iter_collated_validation_segments(
            loader,
            tokenizer_fun,
            drop_overlapping_windows=False,
        )
    )
    assert len(segments) == 1
    collated = segments[0].original
    assert collated.shape == full_spikes.t().shape  # (units, time) vs (time, units)
    assert torch.equal(collated, full_spikes.t())


# ── Unit tests for _trim_input_overlap ──────────────────────────────────────


def test_trim_input_overlap_basic() -> None:
    """Two windows: [0.0, 1.0]s and [0.5, 1.5]s at 1kHz → 500 bins trimmed."""
    n_units = 3
    sampling_rate = 1000.0

    # Window 1 covers [0.0, 1.0]s → 1000 bins
    # Window 2 covers [0.5, 1.5]s → 1000 bins, overlap = 500 bins
    time_bins_w2 = 1000
    original = torch.randn(n_units, time_bins_w2)
    reconstructed = torch.randn(n_units, time_bins_w2)
    mask = torch.ones(n_units, time_bins_w2, dtype=torch.bool)

    trimmed_orig, trimmed_recon, trimmed_mask, trimmed_start = _trim_input_overlap(
        original=original,
        reconstructed=reconstructed,
        mask=mask,
        current_absolute_start=0.5,
        prev_absolute_end=1.0,
        sampling_rate=sampling_rate,
    )

    # 500 bins should be trimmed from the head
    assert trimmed_orig.shape == (n_units, 500)
    assert trimmed_recon.shape == (n_units, 500)
    assert trimmed_mask.shape == (n_units, 500)

    # The remaining data should be the second half of the original window
    assert torch.equal(trimmed_orig, original[:, 500:])
    assert torch.equal(trimmed_recon, reconstructed[:, 500:])

    # trimmed_start should approximately equal prev_absolute_end
    assert abs(trimmed_start - 1.0) < 1e-6


def test_trim_input_overlap_no_overlap() -> None:
    """No overlap: window1 [0.0, 1.0]s, window2 [1.0, 2.0]s → nothing trimmed."""
    n_units = 2
    time_bins = 1000
    original = torch.randn(n_units, time_bins)
    reconstructed = torch.randn(n_units, time_bins)
    mask = torch.ones(n_units, time_bins, dtype=torch.bool)

    trimmed_orig, trimmed_recon, trimmed_mask, trimmed_start = _trim_input_overlap(
        original=original,
        reconstructed=reconstructed,
        mask=mask,
        current_absolute_start=1.0,
        prev_absolute_end=1.0,
        sampling_rate=1000.0,
    )

    assert trimmed_orig.shape == (n_units, time_bins)
    assert torch.equal(trimmed_orig, original)
    assert trimmed_start == 1.0


def test_trim_input_overlap_tiny_gap() -> None:
    """Tiny negative overlap (gap) → nothing trimmed."""
    n_units = 2
    time_bins = 1000
    original = torch.randn(n_units, time_bins)
    reconstructed = torch.randn(n_units, time_bins)
    mask = torch.ones(n_units, time_bins, dtype=torch.bool)

    trimmed_orig, _, _, trimmed_start = _trim_input_overlap(
        original=original,
        reconstructed=reconstructed,
        mask=mask,
        current_absolute_start=1.001,
        prev_absolute_end=1.0,
        sampling_rate=1000.0,
    )

    # No trimming: prev_end < current_start
    assert trimmed_orig.shape == (n_units, time_bins)
    assert trimmed_start == 1.001


def test_trim_input_overlap_trimmed_start_sanity() -> None:
    """After trimming, trimmed_start ≈ prev_absolute_end (within 1 bin)."""
    sampling_rate = 1000.0
    n_units = 1
    time_bins = 2000

    original = torch.randn(n_units, time_bins)
    reconstructed = torch.randn(n_units, time_bins)
    mask = torch.ones(n_units, time_bins, dtype=torch.bool)

    # Window starts at 0.3s, prev ended at 0.7s → overlap = 0.4s = 400 bins
    _, _, _, trimmed_start = _trim_input_overlap(
        original=original,
        reconstructed=reconstructed,
        mask=mask,
        current_absolute_start=0.3,
        prev_absolute_end=0.7,
        sampling_rate=sampling_rate,
    )

    bin_duration = 1.0 / sampling_rate
    assert abs(trimmed_start - 0.7) < bin_duration + 1e-9


# ── Unit tests for _trim_token_overlap ──────────────────────────────────────


def test_trim_token_overlap_basic() -> None:
    """Tokens overlapping with previous window are removed."""
    # Window1 tokens cover [0.0, 1.0]s with 100ms bins → 10 tokens
    # Window2 tokens cover [0.5, 1.5]s with 100ms bins → 10 tokens
    # Token start times for window2: [0.5, 0.6, ..., 1.4]
    # Token end times for window2:   [0.6, 0.7, ..., 1.5]
    # prev_last_token_end_time = 1.0 (window1's last token ends at 1.0s)
    # Tokens with start_time < 1.0 should be dropped → first 5 tokens dropped
    n_tokens = 10
    latent_dim = 4
    latents = torch.randn(n_tokens, latent_dim)
    token_start_times = torch.tensor(
        [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4], dtype=torch.float64
    )
    token_end_times = torch.tensor(
        [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5], dtype=torch.float64
    )

    result = _trim_token_overlap(
        latents=latents,
        token_start_times=token_start_times,
        token_end_times=token_end_times,
        prev_last_token_end_time=1.0,
    )

    # Tokens with start >= 1.0 survive: indices 5..9 → 5 tokens
    assert result["latents"].shape == (5, latent_dim)
    assert torch.equal(result["latents"], latents[5:])
    assert result["token_start_times"].shape == (5,)
    assert torch.allclose(result["token_start_times"], token_start_times[5:])
    assert torch.allclose(result["token_end_times"], token_end_times[5:])


def test_trim_token_overlap_no_overlap() -> None:
    """No overlap → nothing trimmed."""
    n_tokens = 5
    latent_dim = 3
    latents = torch.randn(n_tokens, latent_dim)
    token_start_times = torch.tensor([1.0, 1.1, 1.2, 1.3, 1.4], dtype=torch.float64)
    token_end_times = torch.tensor([1.1, 1.2, 1.3, 1.4, 1.5], dtype=torch.float64)

    result = _trim_token_overlap(
        latents=latents,
        token_start_times=token_start_times,
        token_end_times=token_end_times,
        prev_last_token_end_time=1.0,
    )

    assert result["latents"].shape == (n_tokens, latent_dim)
    assert torch.equal(result["latents"], latents)


def test_trim_token_overlap_with_behavioral_data() -> None:
    """Behavioral data arrays are trimmed with the same mask."""
    n_tokens = 8
    latent_dim = 2
    latents = torch.randn(n_tokens, latent_dim)
    token_start_times = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=torch.float64)
    token_end_times = token_start_times + 0.1

    behavioral_data = {
        "velocity": np.random.randn(n_tokens, 2),
        "hand_pos": np.random.randn(n_tokens, 3),
    }

    result = _trim_token_overlap(
        latents=latents,
        token_start_times=token_start_times,
        token_end_times=token_end_times,
        prev_last_token_end_time=0.5,
        behavioral_data=behavioral_data,
    )

    # Tokens with start >= 0.5: indices 5, 6, 7 → 3 tokens
    assert result["latents"].shape == (3, latent_dim)
    assert result["behavioral_data"]["velocity"].shape == (3, 2)
    assert result["behavioral_data"]["hand_pos"].shape == (3, 3)
    # Check the actual data matches
    assert np.array_equal(result["behavioral_data"]["velocity"], behavioral_data["velocity"][5:])


def test_trim_token_overlap_disabled() -> None:
    """When prev_last_token_end_time is None, no trimming occurs."""
    n_tokens = 5
    latent_dim = 3
    latents = torch.randn(n_tokens, latent_dim)
    token_start_times = torch.tensor([0.5, 0.6, 0.7, 0.8, 0.9], dtype=torch.float64)
    token_end_times = token_start_times + 0.1

    result = _trim_token_overlap(
        latents=latents,
        token_start_times=token_start_times,
        token_end_times=token_end_times,
        prev_last_token_end_time=None,
    )

    assert result["latents"].shape == (n_tokens, latent_dim)
    assert torch.equal(result["latents"], latents)
