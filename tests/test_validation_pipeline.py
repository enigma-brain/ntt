import os
import sys
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_brain.data.dataset import DatasetIndex

from ntt.contrib.tokenizer.poisson_baseline import poisson_baseline_reconstruction
from ntt.core.components.types import BinnedSpikesInput
from ntt.core.data.datamodule import collate
from ntt.core.model.tokenizer.base import DataType
from ntt.evaluation.config import EvaluationConfig
from ntt.evaluation.core.types import EvalData  # noqa: F401
from ntt.evaluation.metrics.spike_detection import SmoothSE
from ntt.evaluation.utils.validation_collation import (
    iter_collated_validation_segments,
    run_validation_evaluation_per_session,
)
from tests.convolution_helpers import gaussian_distance
from tests.fixtures import WindowedSpikeDataset


def _make_loader(full_spikes: torch.Tensor, sampling_rate: float) -> DataLoader:
    indices = [
        DatasetIndex("session", 0.0, 1.0),
        DatasetIndex("session", 1.0, 2.0),
        DatasetIndex("session", 2.0, 3.0),
    ]
    dataset = WindowedSpikeDataset(full_spikes, sampling_rate, "session", len(indices))
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=indices,  # type: ignore[arg-type]
        collate_fn=collate,
    )
    return loader


def _make_shifted_tokenizer(
    shift_bins: int,
) -> Callable[[BinnedSpikesInput], dict[str, torch.Tensor]]:
    """This 'tokenizer' returns the shifted spike sequence as reconstruction."""

    def fn(batch: BinnedSpikesInput) -> dict[str, torch.Tensor]:
        spike_matrix = batch.spike_matrix  # Shape: (batch, time, units) or (time, units)
        shifted = torch.roll(spike_matrix, shifts=shift_bins, dims=-2)  # Shift along time dim
        return {"reconstruction": shifted}

    return fn


def _make_rate_tokenizer(
    full_spikes: torch.Tensor,
    sampling_rate: float,
    tokenizer_bin_size_ms: float = 1000,
) -> Callable[[BinnedSpikesInput], dict[str, torch.Tensor]]:
    """Returns Poisson spikes sampled given the original's rate in a given bin size.

    Parameters:
    - full_spikes: (units, time) - the full spike train (unused, kept for API compat)
    - sampling_rate: sampling rate in Hz, used to derive bin_size_ms
    - tokenizer_bin_size_ms: The bin size to compute rates at

    Returns:
    - tokenizer: Callable[[BinnedSpikesInput], dict[str, torch.Tensor]]
    """

    bin_size_ms = 1000.0 / sampling_rate

    def fn(batch: BinnedSpikesInput) -> dict[str, torch.Tensor]:
        spike_matrix = batch.spike_matrix  # Shape: (batch, time, units) or (time, units)
        has_batch = spike_matrix.ndim == 3

        if has_batch:
            # (batch, time, units) -> process each batch item
            results = []
            for b in range(spike_matrix.shape[0]):
                item = spike_matrix[b].numpy().T  # (units, time)
                recon = poisson_baseline_reconstruction(
                    item,
                    bin_size_ms=bin_size_ms,
                    rate_estimation_bin_ms=tokenizer_bin_size_ms,
                    seed=0,
                )
                results.append(torch.from_numpy(recon.T))  # back to (time, units)
            reconstruction = torch.stack(results)
        else:
            item = spike_matrix.numpy().T  # (units, time)
            recon = poisson_baseline_reconstruction(
                item,
                bin_size_ms=bin_size_ms,
                rate_estimation_bin_ms=tokenizer_bin_size_ms,
                seed=0,
            )
            reconstruction = torch.from_numpy(recon.T)  # back to (time, units)

        return {"reconstruction": reconstruction.to(spike_matrix.dtype)}

    return fn


def _make_fake_rate_tokenizer(
    tokenizer_bin_size_ms: float,
    sampling_rate: float,
) -> Callable[[BinnedSpikesInput], dict[str, torch.Tensor]]:
    """Fake rate tokenizer that bins spikes and places them at the start of each bin.

    This tokenizer:
    1. Bins the input spikes into windows of size tokenizer_bin_size_ms
    2. Counts total spikes in each window
    3. Places that many spikes in the first k milliseconds of the window

    This creates a rate-matched reconstruction but with different timing.
    Fully vectorized - no loops over batch, units, or time within bins.
    """

    def fn(batch: BinnedSpikesInput) -> dict[str, torch.Tensor]:
        spike_matrix = batch.spike_matrix  # Shape: (batch, time, units) or (time, units)

        # Handle empty batches
        if spike_matrix.numel() == 0:
            return {"reconstruction": torch.zeros_like(spike_matrix)}

        bin_size_samples = int(tokenizer_bin_size_ms * sampling_rate / 1000.0)
        # Time is the second-to-last dimension in BinnedSpikesInput
        time_dim = spike_matrix.shape[-2]
        n_bins = time_dim // bin_size_samples

        if n_bins == 0:
            return {"reconstruction": torch.zeros_like(spike_matrix)}

        if len(spike_matrix.shape) == 3:
            b, t, u = spike_matrix.shape
        else:
            assert len(spike_matrix.shape) == 2, "Expected 2 or 3 dimensions"
            t, u = spike_matrix.shape
            b = 1
            spike_matrix = spike_matrix.unsqueeze(0)

        # (b, t, u) -> (b, n_bins, bin_size_samples, u)
        truncated_t = n_bins * bin_size_samples
        truncated = spike_matrix[:, :truncated_t, :]
        bin_view = truncated.view(b, n_bins, bin_size_samples, u)
        counts = bin_view.sum(dim=2)  # (b, n_bins, u)
        reconstruction_view = torch.zeros_like(bin_view)

        max_count = torch.max(counts).item() if counts.numel() > 0 else 0
        for i in range(int(max_count)):
            mask = counts > i  # shape: (b, n_bins, u)
            reconstruction_view[:, :, i, :] = mask.float()

        # Reshape back to (b, truncated_t, u)
        reconstruction = reconstruction_view.view(b, truncated_t, u)

        # Pad back to original time dimension if needed
        if truncated_t < t:
            padding = torch.zeros(b, t - truncated_t, u, dtype=reconstruction.dtype)
            reconstruction = torch.cat([reconstruction, padding], dim=1)

        # Remove batch dimension if input didn't have one
        if len(batch.spike_matrix.shape) == 2:
            reconstruction = reconstruction.squeeze(0)

        return {"reconstruction": reconstruction}

    return fn


def test_validation_pipeline_integration_smoothed_se() -> None:
    sampling_rate = 1000.0
    total_bins = 3000
    n_units = 5
    spikes = torch.zeros((n_units, total_bins))  # Shape: (units, time)
    spikes[:, [100, 500, 1500]] = 1.0  # Set spikes at time bins 100, 500, 1500 for all units

    loader = _make_loader(spikes, sampling_rate)

    eval_config = EvaluationConfig(
        metrics={
            "smoothed_squared_error": {"metric_class": "smoothed_squared_error"},
        },
    )

    shift_tokenizer = _make_shifted_tokenizer(shift_bins=5)
    reports_shift = run_validation_evaluation_per_session(
        loader,
        shift_tokenizer,
        data_type=DataType.SPIKES,
        bin_size_seconds=1.0 / sampling_rate,
        eval_config=eval_config,
        device=torch.device("cpu"),
    )
    session_reports_shift = reports_shift["session"]
    report_shift = session_reports_shift[0][1]["report"]
    assert "smoothed_squared_error" in report_shift.results
    shift_values = report_shift.results["smoothed_squared_error"].raw_value
    expected_shift = gaussian_distance(sigma_ms=10.0, shift_ms=5.0)
    # metric now normalizes by spike count and returns per-unit array
    assert np.allclose(shift_values, expected_shift, rtol=1e-2)

    rate_tokenizer = _make_rate_tokenizer(spikes, sampling_rate)
    reports_rate = run_validation_evaluation_per_session(
        loader,
        rate_tokenizer,
        data_type=DataType.SPIKES,
        bin_size_seconds=1.0 / sampling_rate,
        eval_config=eval_config,
        device=torch.device("cpu"),
    )
    session_reports_rate = reports_rate["session"]
    report_rate = session_reports_rate[0][1]["report"]
    assert "smoothed_squared_error" in report_rate.results
    rate_values = report_rate.results["smoothed_squared_error"].raw_value
    # the rate of the sample data is just 3 spikes per second,
    # resampling that will lead to a large MSE (per-unit array)
    assert np.mean(rate_values) > 1.5  # on the order of num spikes per unit, = 3


def test_binned_rate_difference_integration() -> None:
    """Integration test for BinnedRateDifference with oscillating pattern.

    Tests both value correctness and metadata presence.
    """
    from ntt.evaluation.metrics.spike_detection import BinnedRateDifference

    sampling_rate = 1000.0  # 1ms resolution
    n_units = 2

    # Create one cycle of oscillating pattern: 0,1,2,3,4,5,4,3,2,1 spikes per 5ms (50ms total)
    spike_pattern = [0, 1, 2, 3, 4, 5, 4, 3, 2, 1]
    cycle_length = 50  # ms

    # Build one cycle
    one_cycle = torch.zeros((n_units, cycle_length))
    for i, n_spikes in enumerate(spike_pattern):
        start_idx = i * 5
        if n_spikes > 0:
            one_cycle[:, start_idx : start_idx + n_spikes] = 1

    # Tile to 3 seconds (3000 bins)
    n_repeats = 3000 // cycle_length
    spikes = one_cycle.repeat(1, n_repeats)

    loader = _make_loader(spikes, sampling_rate)

    # Use fake rate tokenizer with 50ms bins
    fake_tokenizer = _make_fake_rate_tokenizer(
        tokenizer_bin_size_ms=50.0,
        sampling_rate=sampling_rate,
    )

    eval_config = EvaluationConfig(
        metrics={
            "binned_rate_difference": {"metric_class": "binned_rate_difference"},
        },
    )

    # Run validation
    reports = run_validation_evaluation_per_session(
        loader,
        fake_tokenizer,
        data_type=DataType.SPIKES,
        bin_size_seconds=1.0 / sampling_rate,
        eval_config=eval_config,
        device=torch.device("cpu"),
    )

    session_reports = reports["session"]
    report = session_reports[0][1]["report"]

    # Check metric exists
    assert "binned_rate_difference" in report.results
    result = report.results["binned_rate_difference"]

    # Check all metadata fields are present
    assert "mean_rate_original_hz" in result.metadata
    assert "mean_rate_reconstructed_hz" in result.metadata
    assert "bin_size_ms" in result.metadata
    assert "rate_unit" in result.metadata
    assert result.metadata["rate_unit"] == "Hz"

    # Check actual values match expected (from test_metrics.py oscillating test)
    # Default metric bin size is 5.0ms (from BinnedRateDifference.__init__)
    # Expected differences: [5,4,3,2,1,5,4,3,2,1] spikes per 5ms bin
    # MAE = 3.0 spikes / 0.005s = 600 Hz per unit
    assert result.raw_value is not None
    assert isinstance(result.raw_value, np.ndarray)
    assert np.allclose(result.raw_value, 600.0, rtol=1e-2)

    # Also check metadata
    assert result.metadata["bin_size_ms"] == 5.0
    assert result.metadata["rate_unit"] == "Hz"


def test_spike_detection_f1_integration() -> None:
    """Integration test for SpikeDetectionF1 with 5ms shifted spikes.

    Tests the metric with two different tolerance windows:
    1. 10ms tolerance: should match all spikes (F1=1.0)
    2. 4ms tolerance: should match no spikes (F1=0.0) since shift is 5ms
    """
    sampling_rate = 1000.0  # 1ms resolution
    n_units = 5
    total_bins = 3000

    # Create spikes at specific times
    spikes = torch.zeros((n_units, total_bins))
    spikes[:, [100, 500, 1500, 2000, 2500]] = 1.0
    loader = _make_loader(spikes, sampling_rate)

    # Shift by 5ms
    shift_tokenizer = _make_shifted_tokenizer(shift_bins=5)

    # Test 1: 10ms tolerance - should match all spikes
    eval_config_match = EvaluationConfig(
        metrics={
            "spike_detection_f1": {
                "metric_class": "spike_detection_f1",
                "spike_match_tolerance_ms": 10.0,
            },
        },
    )

    reports_match = run_validation_evaluation_per_session(
        loader,
        shift_tokenizer,
        data_type=DataType.SPIKES,
        bin_size_seconds=1.0 / sampling_rate,
        eval_config=eval_config_match,
        device=torch.device("cpu"),
    )

    session_reports_match = reports_match["session"]
    report_match = session_reports_match[0][1]["report"]

    assert "spike_detection_f1" in report_match.results
    result_match = report_match.results["spike_detection_f1"]

    # All spikes should match: TP=5 per unit, FP=0, FN=0
    # F1 = 1.0, precision = 1.0, recall = 1.0
    assert result_match.raw_value is not None
    # raw_value is per-unit array
    f1_values = np.array(result_match.raw_value)
    assert f1_values.shape == (n_units,)
    assert np.allclose(f1_values, 1.0)

    # Check metadata
    assert "precision" in result_match.metadata
    assert "recall" in result_match.metadata
    precision = np.array(result_match.metadata["precision"])
    recall = np.array(result_match.metadata["recall"])
    assert np.allclose(precision, 1.0)
    assert np.allclose(recall, 1.0)

    # Test 2: 4ms tolerance - should match no spikes
    eval_config_no_match = EvaluationConfig(
        metrics={
            "spike_detection_f1": {
                "metric_class": "spike_detection_f1",
                "spike_match_tolerance_ms": 4.0,
            },
        },
    )

    reports_no_match = run_validation_evaluation_per_session(
        loader,
        shift_tokenizer,
        data_type=DataType.SPIKES,
        bin_size_seconds=1.0 / sampling_rate,
        eval_config=eval_config_no_match,
        device=torch.device("cpu"),
    )

    session_reports_no_match = reports_no_match["session"]
    report_no_match = session_reports_no_match[0][1]["report"]

    assert "spike_detection_f1" in report_no_match.results
    result_no_match = report_no_match.results["spike_detection_f1"]

    # No spikes should match: TP=0, FP=5 per unit, FN=5 per unit
    # F1 = 0.0, precision = 0.0, recall = 0.0
    assert result_no_match.raw_value is not None
    f1_values_no_match = np.array(result_no_match.raw_value)
    assert f1_values_no_match.shape == (n_units,)
    assert np.allclose(f1_values_no_match, 0.0)

    # Check metadata
    precision_no_match = np.array(result_no_match.metadata["precision"])
    recall_no_match = np.array(result_no_match.metadata["recall"])
    assert np.allclose(precision_no_match, 0.0)
    assert np.allclose(recall_no_match, 0.0)


class MultiSessionSpikeDataset(torch.utils.data.Dataset):
    """Dataset that serves windows from multiple sessions with different neuron counts."""

    def __init__(
        self,
        session_spikes: dict[str, torch.Tensor],
        sampling_rate: float,
    ):
        """
        Args:
            session_spikes: Dict mapping session_id to spike tensor of shape (units, time).
            sampling_rate: Sampling rate in Hz.
        """
        self.session_spikes = session_spikes
        self.sampling_rate = sampling_rate

    def __len__(self) -> int:
        # Not meaningful since we use a sampler; return sum of sessions
        return sum(s.shape[1] for s in self.session_spikes.values())

    def __getitem__(self, index: DatasetIndex) -> BinnedSpikesInput:
        session_id = index.recording_id
        spikes = self.session_spikes[session_id]
        start = int(round(index.start * self.sampling_rate))
        end = int(round(index.end * self.sampling_rate))

        window = spikes[:, start:end]  # (units, time)
        spike_matrix = window.T.int()  # (time, units)
        n_units = spike_matrix.shape[1]

        return BinnedSpikesInput(
            spike_matrix=spike_matrix,
            pad_and_gap_mask=torch.ones_like(spike_matrix, dtype=torch.int32),
            sampling_rate=self.sampling_rate,
            unit_ids=torch.arange(n_units, dtype=torch.int32),
            unit_ids_pad_mask=torch.ones(n_units, dtype=torch.int32),
            duration_sec=torch.tensor((end - start) / self.sampling_rate),
            session_id=session_id,
            subject_id="test_subject",
            species="test_species",
            absolute_start=torch.tensor(index.start, dtype=torch.float64),
            absolute_end=torch.tensor(index.end, dtype=torch.float64),
        )


def test_end_to_end_multisession_with_overlap() -> None:
    """End-to-end integration test: 2 sessions, different neuron counts, overlapping windows.

    Exercises the full pipeline: DataLoader -> run_validation_evaluation_per_session()
    -> EvaluationReport per session.

    Checks:
    - One report group per session
    - Reports contain metric results
    - No exceptions during the entire flow
    - Timestamps in collated segments are monotonically increasing (overlap handled)
    """
    sampling_rate = 1000.0

    # Session A: 3 neurons, 3 seconds of data
    n_units_a = 3
    total_bins_a = 3000
    spikes_a = torch.zeros((n_units_a, total_bins_a))
    spikes_a[:, [50, 200, 800, 1500, 2200, 2800]] = 1.0

    # Session B: 7 neurons (different count forces padding when batched), 2 seconds
    n_units_b = 7
    total_bins_b = 2000
    spikes_b = torch.zeros((n_units_b, total_bins_b))
    spikes_b[:, [100, 600, 1200, 1800]] = 1.0

    dataset = MultiSessionSpikeDataset(
        session_spikes={"session_a": spikes_a, "session_b": spikes_b},
        sampling_rate=sampling_rate,
    )

    # Windows: session_a has overlapping windows, session_b has contiguous windows
    indices = [
        # Session A: 3 windows with overlap between 2nd and 3rd
        DatasetIndex("session_a", 0.0, 1.0),
        DatasetIndex("session_a", 1.0, 2.0),
        DatasetIndex("session_a", 1.5, 2.5),  # overlaps with previous by 0.5s
        # note: window [2.5, 3.0] is NOT covered — that's fine, the pipeline handles partial
        # coverage
        # Session B: 2 contiguous windows
        DatasetIndex("session_b", 0.0, 1.0),
        DatasetIndex("session_b", 1.0, 2.0),
    ]

    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=indices,  # type: ignore[arg-type]
        collate_fn=collate,
    )

    eval_config = EvaluationConfig(
        metrics={
            "smoothed_squared_error": {"metric_class": "smoothed_squared_error"},
        },
    )

    def identity_tokenizer(batch):
        return {"reconstruction": batch.spike_matrix}

    reports = run_validation_evaluation_per_session(
        loader,
        identity_tokenizer,
        data_type=DataType.SPIKES,
        bin_size_seconds=1.0 / sampling_rate,
        eval_config=eval_config,
        device=torch.device("cpu"),
        drop_overlapping_windows=False,
    )

    # --- Assertions ---
    # One report group per session
    assert "session_a" in reports, f"Missing session_a, got keys: {list(reports.keys())}"
    assert "session_b" in reports, f"Missing session_b, got keys: {list(reports.keys())}"
    assert len(reports) == 2, f"Expected 2 sessions, got {len(reports)}"

    for session_id, segment_list in reports.items():
        assert len(segment_list) >= 1, f"No segments for session {session_id}"
        for _segment, data in segment_list:
            report = data["report"]
            # Report has the metric we asked for
            assert "smoothed_squared_error" in report.results, (
                f"Missing smoothed_squared_error in report for {session_id}"
            )
            result = report.results["smoothed_squared_error"]
            assert result.raw_value is not None
            # Identity tokenizer → reconstruction == original → SSE should be 0 (per-unit)
            assert np.allclose(result.raw_value, 0.0, atol=1e-6), (
                f"Expected ~0 SSE for identity tokenizer, got {result.raw_value} "
                f"in session {session_id}"
            )

    # Check session A segment shape: overlap should have been trimmed
    # Session A covers [0.0, 2.5]s with overlap → after trimming, 2500 bins
    # Note: segment.original is nulled by strip_original() after metric
    # computation; we read the preserved shape from ``original_shape``.
    seg_a = reports["session_a"][0][0]
    assert seg_a.original_shape is not None, "original_shape missing on segment"
    assert seg_a.original_shape[0] == n_units_a, (
        f"Session A units: expected {n_units_a}, got {seg_a.original_shape[0]}"
    )
    assert seg_a.original_shape[1] == 2500, (
        f"Session A time bins: expected 2500 (overlap trimmed), got {seg_a.original_shape[1]}"
    )

    # Check session B segment shape: contiguous, no overlap
    seg_b = reports["session_b"][0][0]
    assert seg_b.original_shape is not None, "original_shape missing on segment"
    assert seg_b.original_shape[0] == n_units_b, (
        f"Session B units: expected {n_units_b}, got {seg_b.original_shape[0]}"
    )
    assert seg_b.original_shape[1] == total_bins_b, (
        f"Session B time bins: expected {total_bins_b}, got {seg_b.original_shape[1]}"
    )


def test_end_to_end_interleaved_sessions() -> None:
    """End-to-end test with interleaved session windows (A, B, A, B ordering).

    Tests that the pipeline correctly handles session changes mid-stream:
    when windows arrive in interleaved order, the pipeline must flush segments
    at session boundaries and produce separate reports per session.
    """
    sampling_rate = 1000.0

    # Session A: 3 neurons, 2 seconds
    n_units_a = 3
    spikes_a = torch.zeros((n_units_a, 2000))
    spikes_a[:, [100, 500, 1500]] = 1.0

    # Session B: 7 neurons, 2 seconds
    n_units_b = 7
    spikes_b = torch.zeros((n_units_b, 2000))
    spikes_b[:, [200, 800, 1800]] = 1.0

    dataset = MultiSessionSpikeDataset(
        session_spikes={"session_a": spikes_a, "session_b": spikes_b},
        sampling_rate=sampling_rate,
    )

    # Interleaved order: A, B, A, B — each session change triggers a flush.
    # This means each session will produce 2 separate segments (not concatenated).
    indices = [
        DatasetIndex("session_a", 0.0, 1.0),
        DatasetIndex("session_b", 0.0, 1.0),
        DatasetIndex("session_a", 1.0, 2.0),
        DatasetIndex("session_b", 1.0, 2.0),
    ]

    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=indices,  # type: ignore[arg-type]
        collate_fn=collate,
    )

    eval_config = EvaluationConfig(
        metrics={
            "smoothed_squared_error": {"metric_class": "smoothed_squared_error"},
        },
    )

    def identity_tokenizer(batch):
        return {"reconstruction": batch.spike_matrix}

    reports = run_validation_evaluation_per_session(
        loader,
        identity_tokenizer,
        data_type=DataType.SPIKES,
        bin_size_seconds=1.0 / sampling_rate,
        eval_config=eval_config,
        device=torch.device("cpu"),
    )

    # Both sessions should have reports
    assert "session_a" in reports, f"Missing session_a, got keys: {list(reports.keys())}"
    assert "session_b" in reports, f"Missing session_b, got keys: {list(reports.keys())}"

    # Due to interleaving, each session gets 2 segments (flushed at each session change)
    assert len(reports["session_a"]) == 2, (
        f"Session A: expected 2 segments (interleaved), got {len(reports['session_a'])}"
    )
    assert len(reports["session_b"]) == 2, (
        f"Session B: expected 2 segments (interleaved), got {len(reports['session_b'])}"
    )

    # Check unit counts are correct (different per session)
    # Note: segment.original is nulled by strip_original() after metric
    # computation; we read the preserved shape from ``original_shape``.
    for seg, _data in reports["session_a"]:
        assert seg.original_shape is not None, "original_shape missing on segment"
        assert seg.original_shape[0] == n_units_a, (
            f"Session A units: expected {n_units_a}, got {seg.original_shape[0]}"
        )
    for seg, _data in reports["session_b"]:
        assert seg.original_shape is not None, "original_shape missing on segment"
        assert seg.original_shape[0] == n_units_b, (
            f"Session B units: expected {n_units_b}, got {seg.original_shape[0]}"
        )

    # Identity tokenizer → SSE should be 0 for all segments
    for session_id in ("session_a", "session_b"):
        for _seg, data in reports[session_id]:
            report = data["report"]
            sse = report.results["smoothed_squared_error"].raw_value
            assert np.allclose(sse, 0.0, atol=1e-6), (
                f"Expected ~0 SSE for identity tokenizer in {session_id}, got {sse}"
            )


if __name__ == "__main__":
    test_validation_pipeline_integration_smoothed_se()
    test_binned_rate_difference_integration()
    test_spike_detection_f1_integration()
    test_end_to_end_multisession_with_overlap()
    test_end_to_end_interleaved_sessions()
