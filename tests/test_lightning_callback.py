"""Tests for TokenizerEvalCallback Lightning integration."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch_brain.data.dataset import DatasetIndex

from ntt.core.components.types import BinnedSpikesInput
from ntt.evaluation.config.schema import (
    EvaluationConfig,
    LightningEvalConfig,
)
from ntt.evaluation.core.types import DataType, EvalData
from ntt.evaluation.lightning import TokenizerEvalCallback
from ntt.evaluation.pipeline import EvaluationPipeline
from tests.fixtures import (
    SimpleDataset,
    WindowedSpikeDataset,
    make_dataloader,
    make_identity_tokenizer,
    make_shifted_tokenizer,
    make_spike_tensor,
)


@dataclass
class MockTokenizerSpec:
    """Mock tokenizer spec for testing."""

    bin_size_ms: float = 1.0
    window_length_seconds: float = 1.0
    output_type: DataType = DataType.SPIKES


class MockTokenizerOutput:
    """Mock tokenizer output."""

    def __init__(self, reconstruction: torch.Tensor, aux_loss: torch.Tensor | None = None):
        self.reconstruction = reconstruction
        self.aux_loss = aux_loss
        self.tokens = reconstruction  # For compatibility


class MockTokenizer(torch.nn.Module):
    """Mock tokenizer that applies a simple transformation."""

    def __init__(self, transform_fn=None):
        super().__init__()
        self.spec = MockTokenizerSpec()
        self.transform_fn = transform_fn or (lambda x: x)
        # Need at least one parameter for Lightning
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, batch: BinnedSpikesInput) -> MockTokenizerOutput:
        spike_matrix = batch.spike_matrix
        if spike_matrix.dim() == 2:
            spike_matrix = spike_matrix.unsqueeze(0)
        reconstruction = self.transform_fn(spike_matrix)
        return MockTokenizerOutput(reconstruction=reconstruction)


class MockLitModule(pl.LightningModule):
    """Mock Lightning module for testing the callback."""

    def __init__(self, tokenizer: MockTokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.tokenizer_spec = tokenizer.spec

    def forward(self, batch):
        return self.tokenizer(batch)

    def training_step(self, batch, batch_idx):
        output = self.tokenizer(batch)
        loss = torch.mean(output.reconstruction.float())
        return loss

    def validation_step(self, batch, batch_idx):
        output = self.tokenizer(batch)
        loss = torch.mean(output.reconstruction.float())
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def get_eval_data(self, batch: BinnedSpikesInput) -> tuple[EvalData, EvalData]:
        """Return EvalData for evaluation."""
        output = self.tokenizer(batch)

        # Handle batch dimension
        spike_matrix = batch.spike_matrix
        if spike_matrix.dim() == 3:
            spike_matrix = spike_matrix[0]  # Take first batch item

        reconstruction = output.reconstruction
        if reconstruction.dim() == 3:
            reconstruction = reconstruction[0]

        # BinnedSpikesInput is (time, units), EvalData expects (units, time)
        original = EvalData.from_tensor(
            spike_matrix.T,  # Transpose to (units, time)
            data_type=DataType.SPIKES,
            bin_size_seconds=self.tokenizer_spec.bin_size_ms / 1000.0,
        )
        reconstructed = EvalData.from_tensor(
            reconstruction.T,  # Transpose to (units, time)
            data_type=DataType.SPIKES,
            bin_size_seconds=self.tokenizer_spec.bin_size_ms / 1000.0,
        )
        return original, reconstructed


class TestTokenizerEvalCallbackInit:
    """Tests for callback initialization."""

    def test_default_config(self):
        """Test callback initializes with default configs."""
        callback = TokenizerEvalCallback()
        assert callback.eval_config is not None
        assert callback.lightning_config is not None
        assert callback.pipeline is not None

    def test_custom_eval_config(self):
        """Test callback accepts custom evaluation config."""
        config = EvaluationConfig(
            metrics={
                "spike_detection_f1": {"metric_class": "spike_detection_f1"},
                "temporal_precision": {"metric_class": "temporal_precision"},
                "spike_count_correlation": {"metric_class": "spike_count_correlation"},
                "smoothed_squared_error": {"metric_class": "smoothed_squared_error"},
                "binned_rate_difference": {"metric_class": "binned_rate_difference"},
            },
        )
        callback = TokenizerEvalCallback(eval_config=config)
        assert callback.eval_config is not None

    def test_custom_lightning_config(self):
        """Test callback accepts custom lightning config."""
        config = LightningEvalConfig(
            eval_every_n_batches=50,
            full_eval_at_training_end=False,
        )
        callback = TokenizerEvalCallback(lightning_config=config)
        assert callback.lightning_config.eval_every_n_batches == 50
        assert callback.lightning_config.full_eval_at_training_end is False

    def test_behavioral_metrics_are_filtered_from_tokenizer_callback_pipeline(self):
        """Tokenizer callback should not schedule downstream behavioral metrics."""
        callback = TokenizerEvalCallback(
            eval_config=EvaluationConfig(
                metrics={
                    "spike_count_correlation": {"metric_class": "spike_count_correlation"},
                    "behavioral_regression_cursor_pos": {
                        "metric_class": "behavioral_regression",
                        "target_field": "cursor_pos",
                    },
                },
            )
        )

        assert [name for name, _metric in callback.pipeline.metrics] == ["spike_count_correlation"]


class TestCallbackGetEvalData:
    """Tests for _get_eval_data method."""

    def test_get_eval_data_via_module_method(self):
        """Test that callback uses module's get_eval_data method."""
        tokenizer = MockTokenizer()
        module = MockLitModule(tokenizer)

        callback = TokenizerEvalCallback()

        # Create a batch
        n_units = 5
        time_bins = 100
        spike_matrix = torch.zeros((1, time_bins, n_units), dtype=torch.int32)
        spike_matrix[0, 10:15, :] = 1

        batch = BinnedSpikesInput(
            spike_matrix=spike_matrix[0],  # Remove batch dim for single item
            pad_and_gap_mask=torch.ones_like(spike_matrix[0], dtype=torch.int32),
            sampling_rate=1000.0,
            unit_ids=torch.arange(n_units, dtype=torch.int32),
            unit_ids_pad_mask=torch.ones(n_units, dtype=torch.int32),
            duration_sec=torch.tensor(time_bins / 1000.0),
            session_id="test",
            subject_id="test",
            species="test",
            absolute_start=torch.tensor(0.0, dtype=torch.float64),
            absolute_end=torch.tensor(time_bins / 1000.0, dtype=torch.float64),
        )

        result = callback._get_eval_data(module, batch)
        assert result is not None
        original, reconstructed = result

        assert original.n_units == n_units
        assert reconstructed.n_units == n_units
        assert original.binned_spikes is not None
        assert reconstructed.binned_spikes is not None

    def test_get_eval_data_fallback(self):
        """Test fallback when module has no get_eval_data."""
        tokenizer = MockTokenizer()
        # Create a module without get_eval_data
        module = pl.LightningModule()
        module.tokenizer = tokenizer
        module.tokenizer_spec = tokenizer.spec

        callback = TokenizerEvalCallback()

        n_units = 5
        time_bins = 100
        spike_matrix = torch.zeros((1, time_bins, n_units), dtype=torch.int32)
        spike_matrix[0, 10:15, :] = 1

        batch = BinnedSpikesInput(
            spike_matrix=spike_matrix,  # Include batch dim
            pad_and_gap_mask=torch.ones_like(spike_matrix, dtype=torch.int32),
            sampling_rate=1000.0,
            unit_ids=torch.arange(n_units, dtype=torch.int32),
            unit_ids_pad_mask=torch.ones(n_units, dtype=torch.int32),
            duration_sec=torch.tensor([time_bins / 1000.0]),
            session_id="test",
            subject_id="test",
            species="test",
            absolute_start=torch.tensor([0.0], dtype=torch.float64),
            absolute_end=torch.tensor([time_bins / 1000.0], dtype=torch.float64),
        )

        result = callback._get_eval_data(module, batch)
        assert result is not None
        original, reconstructed = result
        assert original.n_units == n_units


class TestCallbackQuickEval:
    """Tests for quick evaluation functionality."""

    def test_on_train_batch_end_counts_batches(self):
        """Test that batch counter increments."""
        callback = TokenizerEvalCallback(
            lightning_config=LightningEvalConfig(eval_every_n_batches=5)
        )
        tokenizer = MockTokenizer()
        module = MockLitModule(tokenizer)
        trainer = MagicMock(spec=pl.Trainer)
        trainer.logger = None

        # Create minimal batch
        batch = _make_minimal_batch()

        # Call on_train_batch_end multiple times
        for i in range(3):
            callback.on_train_batch_end(trainer, module, {}, batch, i)

        assert callback.batch_count == 3

    def test_quick_eval_runs_at_interval(self):
        """Test that quick eval runs at configured interval."""
        callback = TokenizerEvalCallback(
            eval_config=EvaluationConfig(
                metrics={
                    "spike_detection_f1": {"metric_class": "spike_detection_f1"},
                    "temporal_precision": {"metric_class": "temporal_precision"},
                    "spike_count_correlation": {"metric_class": "spike_count_correlation"},
                    "smoothed_squared_error": {"metric_class": "smoothed_squared_error"},
                    "binned_rate_difference": {"metric_class": "binned_rate_difference"},
                },
            ),
            lightning_config=LightningEvalConfig(eval_every_n_batches=2),
        )

        tokenizer = MockTokenizer()
        module = MockLitModule(tokenizer)
        trainer = MagicMock(spec=pl.Trainer)
        trainer.logger = MagicMock()
        trainer.global_step = 0

        batch = _make_minimal_batch()

        # First batch - no eval
        callback.on_train_batch_end(trainer, module, {}, batch, 0)
        trainer.logger.log_metrics.assert_not_called()

        # Second batch - should eval
        callback.on_train_batch_end(trainer, module, {}, batch, 1)
        assert trainer.logger.log_metrics.called


class TestCallbackEpochEval:
    """Tests for epoch-end evaluation."""

    def test_epoch_end_eval_disabled(self):
        """Test that epoch eval doesn't run when disabled."""
        callback = TokenizerEvalCallback(
            lightning_config=LightningEvalConfig(eval_on_epoch_end=False)
        )

        tokenizer = MockTokenizer()
        module = MockLitModule(tokenizer)
        trainer = MagicMock(spec=pl.Trainer)
        trainer.val_dataloaders = None
        trainer.logger = MagicMock()

        # Should not crash and should not log anything
        callback.on_validation_epoch_end(trainer, module)
        trainer.logger.log_metrics.assert_not_called()

    def test_epoch_eval_uses_transfer_batch_to_device(self):
        """Test that epoch eval uses Lightning's transfer_batch_to_device for batch movement."""
        callback = TokenizerEvalCallback(
            eval_config=EvaluationConfig(
                metrics={
                    "spike_count_correlation": {"metric_class": "spike_count_correlation"},
                    "smoothed_squared_error": {"metric_class": "smoothed_squared_error"},
                },
            ),
            lightning_config=LightningEvalConfig(
                eval_on_epoch_end=True,
                quick_eval_n_batches=1,
            ),
        )

        tokenizer = MockTokenizer()
        module = MockLitModule(tokenizer)
        trainer = MagicMock(spec=pl.Trainer)
        trainer.logger = MagicMock()
        trainer.global_step = 0

        # Create a small dataloader
        n_units = 5
        time_bins = 100
        batches = []
        for _ in range(2):
            spike_matrix = torch.zeros((1, time_bins, n_units), dtype=torch.int32)
            spike_matrix[0, 10:15, :] = 1
            batch = BinnedSpikesInput(
                spike_matrix=spike_matrix,
                pad_and_gap_mask=torch.ones_like(spike_matrix, dtype=torch.int32),
                sampling_rate=1000.0,
                unit_ids=torch.arange(n_units, dtype=torch.int32).unsqueeze(0),
                unit_ids_pad_mask=torch.ones(1, n_units, dtype=torch.int32),
                duration_sec=torch.tensor([time_bins / 1000.0]),
                session_id=("test",),
                subject_id=("test",),
                species=("test",),
                absolute_start=torch.tensor([0.0], dtype=torch.float64),
                absolute_end=torch.tensor([time_bins / 1000.0], dtype=torch.float64),
            )
            batches.append(batch)

        trainer.val_dataloaders = [batches]

        # Spy on transfer_batch_to_device to verify it gets called
        original_transfer = module.transfer_batch_to_device
        transfer_calls = []

        def tracked_transfer(batch, device, dataloader_idx=0):
            transfer_calls.append((type(batch).__name__, device, dataloader_idx))
            return original_transfer(batch, device, dataloader_idx)

        module.transfer_batch_to_device = tracked_transfer

        callback._run_quick_epoch_eval(trainer, module, n_batches=1)

        # Verify transfer_batch_to_device was called (not a custom method)
        assert len(transfer_calls) >= 1, "transfer_batch_to_device should have been called"
        assert transfer_calls[0][0] == "BinnedSpikesInput"

    def test_epoch_eval_uses_quick_metric_subset_when_enabled(self):
        """Epoch-end quick eval should use run_quick with the configured metric subset."""
        callback = TokenizerEvalCallback(
            eval_config=EvaluationConfig(
                metrics={
                    "spike_count_correlation": {"metric_class": "spike_count_correlation"},
                    "behavioral_regression_cursor_pos": {
                        "metric_class": "behavioral_regression",
                        "target_field": "cursor_pos",
                    },
                },
            ),
            lightning_config=LightningEvalConfig(
                eval_on_epoch_end=True,
                quick_eval_n_batches=1,
                quick_metrics_only=True,
                quick_metrics=["spike_count_correlation"],
            ),
        )

        tokenizer = MockTokenizer()
        module = MockLitModule(tokenizer)
        trainer = MagicMock(spec=pl.Trainer)
        trainer.logger = MagicMock()
        trainer.global_step = 0

        spike_matrix = torch.zeros((1, 100, 5), dtype=torch.int32)
        spike_matrix[0, 10:15, :] = 1
        batch = BinnedSpikesInput(
            spike_matrix=spike_matrix,
            pad_and_gap_mask=torch.ones_like(spike_matrix, dtype=torch.int32),
            sampling_rate=1000.0,
            unit_ids=torch.arange(5, dtype=torch.int32).unsqueeze(0),
            unit_ids_pad_mask=torch.ones(1, 5, dtype=torch.int32),
            duration_sec=torch.tensor([0.1]),
            session_id=("test",),
            subject_id=("test",),
            species=("test",),
            absolute_start=torch.tensor([0.0], dtype=torch.float64),
            absolute_end=torch.tensor([0.1], dtype=torch.float64),
        )
        trainer.val_dataloaders = [[batch]]

        report = MagicMock()
        report.results = {"spike_count_correlation": MagicMock(raw_value=0.5)}
        report.category_scores = {}
        callback.pipeline.run_quick = MagicMock(return_value=report)
        callback.pipeline.run = MagicMock()

        callback._run_quick_epoch_eval(trainer, module, n_batches=1)

        callback.pipeline.run_quick.assert_called()
        callback.pipeline.run.assert_not_called()
        _, kwargs = callback.pipeline.run_quick.call_args
        assert kwargs["metric_names"] == ["spike_count_correlation"]


class TestCallbackFullSessionEval:
    """Tests for full session-collated evaluation."""

    def test_full_eval_at_training_end_disabled(self):
        """Test that full eval doesn't run when disabled."""
        callback = TokenizerEvalCallback(
            lightning_config=LightningEvalConfig(full_eval_at_training_end=False)
        )

        trainer = MagicMock(spec=pl.Trainer)
        module = MagicMock()
        trainer.logger = MagicMock()

        callback.on_fit_end(trainer, module)
        trainer.logger.log_metrics.assert_not_called()

    def test_full_eval_with_no_val_dataloader(self):
        """Test graceful handling when no validation dataloader."""
        callback = TokenizerEvalCallback(
            lightning_config=LightningEvalConfig(full_eval_at_training_end=True)
        )

        trainer = MagicMock(spec=pl.Trainer)
        trainer.val_dataloaders = None
        module = MagicMock()
        module.tokenizer_spec = MockTokenizerSpec()

        # Should not crash
        callback._run_full_session_eval(trainer, module)

    def test_full_eval_is_reconstruction_only(self):
        """Full session eval should not request or forward latent-aware behavioral inputs."""
        callback = TokenizerEvalCallback(
            lightning_config=LightningEvalConfig(full_eval_at_training_end=True)
        )

        trainer = MagicMock(spec=pl.Trainer)
        trainer.val_dataloaders = [object()]
        trainer.logger = MagicMock()
        trainer.global_step = 0

        module = MagicMock()
        module.tokenizer_spec = MockTokenizerSpec()
        module.device = torch.device("cpu")
        module.transfer_batch_to_device.side_effect = lambda batch, device, dataloader_idx=0: batch
        module.tokenizer = MagicMock()
        module.tokenizer.reconstruct_from_input.return_value = {
            "reconstruction": torch.zeros((1, 4, 2), dtype=torch.float32),
        }

        segment = SimpleNamespace(
            session_id="session_A",
            original=torch.ones((2, 4), dtype=torch.float32),
            reconstructed=torch.ones((2, 4), dtype=torch.float32),
            behavioral_data={"cursor_pos": np.array([[0.0, 1.0], [1.0, 2.0]])},
            latents=None,
            token_timestamps=None,
            token_start_times=None,
            token_end_times=None,
        )

        report = MagicMock()
        report.results = {"spike_count_correlation": MagicMock(raw_value=0.5)}
        report.category_scores = {}
        callback.pipeline.run = MagicMock(return_value=report)

        def fake_iter(dataloader, tokenizer_fn, **kwargs):
            tokenizer_fn("fake_batch")
            yield segment

        with patch("ntt.evaluation.lightning.iter_collated_validation_segments", fake_iter):
            callback._run_full_session_eval(trainer, module)

        _, reconstruct_kwargs = module.tokenizer.reconstruct_from_input.call_args
        assert reconstruct_kwargs == {}

        _, run_kwargs = callback.pipeline.run.call_args
        assert run_kwargs == {"session_id": "session_A"}


class TestCallbackPipelineParity:
    """Tests that callback produces same results as direct pipeline."""

    def test_identity_tokenizer_perfect_reconstruction(self):
        """Test that identity tokenizer gives near-zero error."""
        # Create data with known spikes
        sampling_rate = 1000.0
        n_units = 3
        time_bins = 1000
        spikes = make_spike_tensor(n_units, time_bins, spike_positions=[100, 300, 500, 700, 900])

        # Create EvalData directly
        original = EvalData.from_tensor(
            spikes,  # Already (units, time)
            data_type=DataType.SPIKES,
            bin_size_seconds=1.0 / sampling_rate,
        )

        # Use identity: reconstructed == original
        reconstructed = EvalData.from_tensor(
            spikes.clone(),
            data_type=DataType.SPIKES,
            bin_size_seconds=1.0 / sampling_rate,
        )

        # Run pipeline directly
        pipeline = EvaluationPipeline(
            config=EvaluationConfig(
                metrics={
                    "spike_detection_f1": {"metric_class": "spike_detection_f1"},
                    "temporal_precision": {"metric_class": "temporal_precision"},
                    "spike_count_correlation": {"metric_class": "spike_count_correlation"},
                    "smoothed_squared_error": {"metric_class": "smoothed_squared_error"},
                    "binned_rate_difference": {"metric_class": "binned_rate_difference"},
                },
            )
        )
        report = pipeline.run(original, reconstructed)

        # Check smoothed_squared_error is near zero for perfect reconstruction
        if "smoothed_squared_error" in report.results:
            sse = report.results["smoothed_squared_error"].raw_value
            assert np.allclose(sse, 0.0, atol=1e-6), f"Expected ~0 SSE for identity, got {sse}"

        # Check spike detection F1 is 1.0
        if "spike_detection_f1" in report.results:
            f1 = report.results["spike_detection_f1"].raw_value
            if hasattr(f1, "__len__"):
                f1 = np.mean(f1)
            assert np.allclose(f1, 1.0), f"Expected F1=1.0 for identity, got {f1}"

    def test_shifted_tokenizer_produces_known_error(self):
        """Test that shifted tokenizer produces expected error."""
        # Create data with known spikes - avoid edges to prevent wrapping issues
        sampling_rate = 1000.0
        n_units = 3
        time_bins = 1000
        # Spike positions away from edges
        spikes = make_spike_tensor(n_units, time_bins, spike_positions=[200, 400, 600, 800])

        # Shift by 5ms (non-wrapping: pad with zeros)
        shift_bins = 5
        shifted_spikes = torch.zeros_like(spikes)
        shifted_spikes[:, shift_bins:] = spikes[:, :-shift_bins]

        # Create EvalData for original
        original = EvalData.from_tensor(
            spikes,
            data_type=DataType.SPIKES,
            bin_size_seconds=1.0 / sampling_rate,
        )
        reconstructed = EvalData.from_tensor(
            shifted_spikes,
            data_type=DataType.SPIKES,
            bin_size_seconds=1.0 / sampling_rate,
        )

        # Run pipeline with tolerance > shift to test spike matching
        pipeline = EvaluationPipeline(
            config=EvaluationConfig(
                metrics={
                    "spike_detection_f1": {
                        "metric_class": "spike_detection_f1",
                        "spike_match_tolerance_ms": 10.0,
                    },
                    "temporal_precision": {
                        "metric_class": "temporal_precision",
                        "spike_match_tolerance_ms": 10.0,
                    },
                    "spike_count_correlation": {"metric_class": "spike_count_correlation"},
                    "smoothed_squared_error": {"metric_class": "smoothed_squared_error"},
                    "binned_rate_difference": {"metric_class": "binned_rate_difference"},
                },
                # spike_match_tolerance_ms is set per-metric in the metrics dict above
            )
        )
        report = pipeline.run(original, reconstructed)

        # With 5ms shift and 10ms tolerance, F1 should be 1.0
        if "spike_detection_f1" in report.results:
            f1 = report.results["spike_detection_f1"].raw_value
            if hasattr(f1, "__len__"):
                f1 = np.mean(f1)
            msg = f"Expected F1=1.0 with 5ms shift and 10ms tolerance, got {f1}"
            assert np.allclose(f1, 1.0), msg

        # smoothed_squared_error should be > 0 but bounded (per-unit array)
        if "smoothed_squared_error" in report.results:
            sse = report.results["smoothed_squared_error"].raw_value
            assert np.all(sse > 0), f"Expected SSE > 0 for shifted data, got {sse}"
            assert np.all(sse < 1.0), f"Expected SSE < 1.0 for small shift, got {sse}"

    def test_callback_vs_direct_pipeline_consistency(self):
        """Test that callback produces same metrics as direct pipeline call."""
        # Create simple data
        sampling_rate = 1000.0
        n_units = 3
        time_bins = 1000
        spikes = make_spike_tensor(n_units, time_bins, spike_positions=[100, 300, 500, 700, 900])

        # Create EvalData
        original = EvalData.from_tensor(
            spikes,
            data_type=DataType.SPIKES,
            bin_size_seconds=1.0 / sampling_rate,
        )
        reconstructed = EvalData.from_tensor(
            spikes.clone(),  # Identity
            data_type=DataType.SPIKES,
            bin_size_seconds=1.0 / sampling_rate,
        )

        # Run pipeline directly
        eval_config = EvaluationConfig(
            metrics={
                "spike_detection_f1": {"metric_class": "spike_detection_f1"},
                "temporal_precision": {"metric_class": "temporal_precision"},
                "spike_count_correlation": {"metric_class": "spike_count_correlation"},
                "smoothed_squared_error": {"metric_class": "smoothed_squared_error"},
                "binned_rate_difference": {"metric_class": "binned_rate_difference"},
            },
        )
        pipeline = EvaluationPipeline(config=eval_config)
        direct_report = pipeline.run(original, reconstructed)

        # Create callback with same config
        callback = TokenizerEvalCallback(eval_config=eval_config)

        # Run through callback's pipeline
        callback_report = callback.pipeline.run(original, reconstructed)

        # Compare results
        for name in direct_report.results:
            if name in callback_report.results:
                direct_val = direct_report.results[name].raw_value
                callback_val = callback_report.results[name].raw_value

                if hasattr(direct_val, "__len__"):
                    np.testing.assert_array_almost_equal(
                        direct_val,
                        callback_val,
                        err_msg=f"Metric {name} differs between direct and callback",
                    )
                else:
                    assert np.isclose(direct_val, callback_val), (
                        f"Metric {name}: direct={direct_val}, callback={callback_val}"
                    )


def _make_minimal_batch(n_units: int = 5, time_bins: int = 100) -> BinnedSpikesInput:
    """Create a minimal batch for testing."""
    spike_matrix = torch.zeros((time_bins, n_units), dtype=torch.int32)
    spike_matrix[10:15, :] = 1

    return BinnedSpikesInput(
        spike_matrix=spike_matrix,
        pad_and_gap_mask=torch.ones_like(spike_matrix, dtype=torch.int32),
        sampling_rate=1000.0,
        unit_ids=torch.arange(n_units, dtype=torch.int32),
        unit_ids_pad_mask=torch.ones(n_units, dtype=torch.int32),
        duration_sec=torch.tensor(time_bins / 1000.0),
        session_id="test",
        subject_id="test",
        species="test",
        absolute_start=torch.tensor(0.0, dtype=torch.float64),
        absolute_end=torch.tensor(time_bins / 1000.0, dtype=torch.float64),
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
