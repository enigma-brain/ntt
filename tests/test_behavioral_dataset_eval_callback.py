"""Tests for BehavioralDatasetEvalCallback."""

from dataclasses import dataclass, field
from itertools import islice
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import pytorch_lightning as pl
import torch

from ntt.core.components.types import BinnedSpikesInput, DataType
from ntt.core.data.components.config import (
    BrainsetSelection,
    SelectionConfig,
    SelectionsModifier,
)
from ntt.evaluation.config.schema import AggregationStrategy, EvaluationConfig
from ntt.evaluation.lightning import (
    BehavioralDatasetEvalCallback,
    _report_to_metrics,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_segment(n_units: int = 10, n_time: int = 100):
    """Create a minimal fake segment with .original having a proper shape."""
    seg = MagicMock()
    seg.original = np.zeros((n_units, n_time))
    return seg


def _make_selections() -> list[SelectionConfig]:
    return [
        SelectionConfig(
            selection=[
                BrainsetSelection(
                    brainset="perich_miller_population_2018",
                    sessions=["c_20131022_center_out_reaching"],
                )
            ],
            config=SelectionsModifier(ignore_splits=True),
        ),
    ]


def _make_behavioral_data(n_samples: int, n_classes: int = 4, seed: int = 42):
    """Create synthetic behavioral data dict for n_samples tokens."""
    rng = np.random.default_rng(seed)
    return {
        "cursor_pos": rng.normal(size=(n_samples, 2)),
        "cursor_vel": rng.normal(size=(n_samples, 2)),
        "trial_id": np.repeat(np.arange(max(1, n_samples // 10)), 10)[:n_samples],
        "trial_target_id": rng.integers(0, n_classes, size=n_samples).astype(float),
        "movement_phase": np.ones(n_samples),  # all reach_period
        "trial_is_valid": np.ones(n_samples, dtype=int),
        "trial_result": rng.integers(0, 3, size=n_samples).astype(float),
    }


def _make_batched_input(
    batch_size: int = 4,
    time_bins: int = 200,
    n_neurons: int = 8,
    session_id: str = "test/session_a",
    seed: int = 42,
) -> BinnedSpikesInput:
    """Create a BinnedSpikesInput with behavioral data attached."""
    rng = np.random.default_rng(seed)
    spike_matrix = torch.from_numpy(
        rng.poisson(0.01, size=(batch_size, time_bins, n_neurons)).astype(np.int32)
    )
    duration_val = time_bins / 1000.0

    # Per-window behavioral data (list of dicts, one per batch item)
    n_tokens_per_window = 10  # typical for a tokenizer with compression
    behavioral_data = [
        _make_behavioral_data(n_tokens_per_window, seed=seed + i) for i in range(batch_size)
    ]

    return BinnedSpikesInput(
        spike_matrix=spike_matrix,
        pad_and_gap_mask=torch.ones_like(spike_matrix, dtype=torch.int32),
        sampling_rate=1000.0,
        unit_ids=torch.arange(n_neurons, dtype=torch.int32).unsqueeze(0).expand(batch_size, -1),
        unit_ids_pad_mask=torch.ones((batch_size, n_neurons), dtype=torch.int32),
        duration_sec=torch.full((batch_size,), duration_val),
        session_id=tuple([session_id] * batch_size),
        subject_id=tuple(["test_subject"] * batch_size),
        species=tuple(["test_species"] * batch_size),
        absolute_start=torch.arange(batch_size, dtype=torch.float64) * duration_val,
        absolute_end=(torch.arange(batch_size, dtype=torch.float64) + 1) * duration_val,
        behavioral_data=behavioral_data,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Init tests
# ---------------------------------------------------------------------------


class TestBehavioralDatasetEvalCallbackInit:
    def test_requires_selections(self):
        with pytest.raises(ValueError, match="requires explicit selections"):
            BehavioralDatasetEvalCallback(selections=None)

    def test_requires_selections_nonempty(self):
        with pytest.raises(ValueError, match="requires explicit selections"):
            BehavioralDatasetEvalCallback(selections=[])

    def test_every_n_epochs_must_be_positive(self):
        with pytest.raises(ValueError, match="every_n_epochs must be >= 1"):
            BehavioralDatasetEvalCallback(selections=_make_selections(), every_n_epochs=0)

    def test_max_batches_must_be_positive(self):
        with pytest.raises(ValueError, match="max_batches must be >= 1"):
            BehavioralDatasetEvalCallback(selections=_make_selections(), max_batches=0)

    def test_max_sessions_must_be_positive(self):
        with pytest.raises(ValueError, match="max_sessions must be >= 1"):
            BehavioralDatasetEvalCallback(selections=_make_selections(), max_sessions=0)

    def test_default_construction(self):
        cb = BehavioralDatasetEvalCallback(selections=_make_selections())
        assert cb.eval_on_epoch_end is True
        assert cb.every_n_epochs == 1
        assert cb.run_on_fit_end is True
        assert cb.compute_poisson_baseline is True
        assert cb.max_batches is None
        assert cb._poisson_baseline_metrics is None

    def test_custom_config(self):
        eval_config = EvaluationConfig(compute_baseline=True)
        cb = BehavioralDatasetEvalCallback(
            eval_config=eval_config,
            selections=_make_selections(),
            log_prefix="custom_prefix",
            max_batches=5,
            max_sessions=2,
            aggregation_strategy=AggregationStrategy.MEDIAN,
            every_n_epochs=3,
            compute_poisson_baseline=False,
        )
        assert cb.eval_config.compute_baseline is True
        assert cb.log_prefix == "custom_prefix"
        assert cb.max_batches == 5
        assert cb.max_sessions == 2
        assert cb.aggregation_strategy == AggregationStrategy.MEDIAN
        assert cb.every_n_epochs == 3
        assert cb.compute_poisson_baseline is False

    def test_string_aggregation_strategy_is_normalized(self):
        cb = BehavioralDatasetEvalCallback(
            selections=_make_selections(),
            aggregation_strategy="mean",
        )
        assert cb.aggregation_strategy == AggregationStrategy.MEAN


# ---------------------------------------------------------------------------
# Epoch gating tests
# ---------------------------------------------------------------------------


class TestBehavioralDatasetEvalCallbackGating:
    def _make_callback_and_mocks(self, **kwargs):
        cb = BehavioralDatasetEvalCallback(selections=_make_selections(), **kwargs)
        trainer = MagicMock(spec=pl.Trainer)
        trainer.current_epoch = 0
        module = MagicMock(spec=pl.LightningModule)
        # Patch _run_eval so no real eval runs
        cb._run_eval = MagicMock()  # type: ignore[method-assign]
        return cb, trainer, module

    def test_epoch_end_runs_at_correct_interval(self):
        cb, trainer, module = self._make_callback_and_mocks(every_n_epochs=2)

        trainer.current_epoch = 0  # epoch 1 → not multiple of 2
        cb.on_validation_epoch_end(trainer, module)
        assert cb._run_eval.call_count == 0

        trainer.current_epoch = 1  # epoch 2 → multiple of 2
        cb.on_validation_epoch_end(trainer, module)
        assert cb._run_eval.call_count == 1

        trainer.current_epoch = 2  # epoch 3 → not multiple of 2
        cb.on_validation_epoch_end(trainer, module)
        assert cb._run_eval.call_count == 1

        trainer.current_epoch = 3  # epoch 4 → multiple of 2
        cb.on_validation_epoch_end(trainer, module)
        assert cb._run_eval.call_count == 2

    def test_eval_on_epoch_end_disabled(self):
        cb, trainer, module = self._make_callback_and_mocks(eval_on_epoch_end=False)
        cb.on_validation_epoch_end(trainer, module)
        cb._run_eval.assert_not_called()

    def test_fit_end_runs_when_enabled(self):
        cb, trainer, module = self._make_callback_and_mocks(run_on_fit_end=True)
        cb.on_fit_end(trainer, module)
        cb._run_eval.assert_called_once()

    def test_fit_end_skipped_when_disabled(self):
        cb, trainer, module = self._make_callback_and_mocks(run_on_fit_end=False)
        cb.on_fit_end(trainer, module)
        cb._run_eval.assert_not_called()

    def test_skips_when_no_tokenizer_spec(self):
        cb = BehavioralDatasetEvalCallback(selections=_make_selections())
        trainer = MagicMock(spec=pl.Trainer)
        module = MagicMock(spec=pl.LightningModule)
        # Remove tokenizer_spec
        del module.tokenizer_spec
        # Should not raise, just warn and return
        cb._run_eval(trainer, module, prefix="test")


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestBehavioralDatasetEvalCallbackErrorHandling:
    def test_eval_failure_hits_exit_barrier(self):
        """In DDP mode, the exit barrier must fire even when _run_eval_inner raises."""
        cb = BehavioralDatasetEvalCallback(selections=_make_selections())
        trainer = MagicMock(spec=pl.Trainer)
        module = MagicMock(spec=pl.LightningModule)
        module.tokenizer_spec = MagicMock()

        # Make inner eval raise
        cb._run_eval_inner = MagicMock(side_effect=RuntimeError("datamodule exploded"))

        with (
            patch("torch.distributed.is_available", return_value=True),
            patch("torch.distributed.is_initialized", return_value=True),
        ):
            # Exception should still propagate
            with pytest.raises(RuntimeError, match="datamodule exploded"):
                cb._run_eval(trainer, module, prefix="test")

        # Both barriers must have been called despite the exception
        barrier_calls = [call.args[0] for call in trainer.strategy.barrier.call_args_list]
        assert "behavioral_eval_entry" in barrier_calls, "entry barrier was not called"
        assert "behavioral_eval_done" in barrier_calls, (
            "exit barrier was not called (deadlock risk)"
        )

    def test_eval_failure_propagates_single_gpu(self):
        """In single-GPU mode, exceptions from _run_eval_inner propagate to the caller."""
        cb = BehavioralDatasetEvalCallback(selections=_make_selections())
        trainer = MagicMock(spec=pl.Trainer)
        module = MagicMock(spec=pl.LightningModule)
        module.tokenizer_spec = MagicMock()

        # Make inner eval raise
        cb._run_eval_inner = MagicMock(side_effect=RuntimeError("datamodule exploded"))

        with (
            patch("torch.distributed.is_available", return_value=False),
            patch("torch.distributed.is_initialized", return_value=False),
        ):
            with pytest.raises(RuntimeError, match="datamodule exploded"):
                cb._run_eval(trainer, module, prefix="test")

    def test_empty_reports_warns_and_returns(self):
        """When no session reports are produced, should warn instead of raise."""
        cb = BehavioralDatasetEvalCallback(
            selections=_make_selections(),
            compute_poisson_baseline=False,
        )

        @dataclass
        class FakeSpec:
            bin_size_ms: float = 1.0
            input_type: DataType = DataType.SPIKES
            token_bin_size_ms: float = 10.0

        trainer = MagicMock(spec=pl.Trainer)
        module = MagicMock(spec=pl.LightningModule)
        module.tokenizer_spec = FakeSpec()

        # Mock the datamodule to return an empty dataloader
        fake_dm = MagicMock()
        fake_dm.val_dataloader.return_value = iter([])
        cb._cached_datamodule = fake_dm

        # Should not raise
        cb._run_eval_inner(trainer, module, prefix="test", is_distributed=False)
        # Logger should not have been called with metrics
        trainer.logger.log_metrics.assert_not_called()

    def test_subset_eval_requests_latents_via_behavioral_callback(self):
        """Behavioral callback should remain the latent-aware evaluation path."""
        cb = BehavioralDatasetEvalCallback(
            selections=_make_selections(),
            compute_poisson_baseline=False,
        )

        @dataclass
        class FakeSpec:
            bin_size_ms: float = 1.0
            input_type: DataType = DataType.SPIKES
            token_bin_size_ms: float = 10.0

        trainer = MagicMock(spec=pl.Trainer)
        trainer.logger = MagicMock()
        trainer.global_step = 0

        module = MagicMock(spec=pl.LightningModule)
        module.tokenizer_spec = FakeSpec()
        module.device = torch.device("cpu")
        module.transfer_batch_to_device.side_effect = lambda batch, device, dataloader_idx=0: batch
        module.tokenizer = MagicMock()
        module.tokenizer.reconstruct_from_input.return_value = {
            "reconstruction": torch.zeros((1, 4, 2), dtype=torch.float32),
            "latents": torch.zeros((1, 2, 3), dtype=torch.float32),
        }

        fake_dm = MagicMock()
        fake_dm.val_dataloader.return_value = ["fake_batch"]
        cb._cached_datamodule = fake_dm

        fake_report = MagicMock()
        fake_report.results = {}

        fake_seg = _fake_segment(n_units=10, n_time=100)

        def fake_run_validation(dataloader, tokenizer_fun, **kwargs):
            tokenizer_fun("fake_batch")
            return {"session_a": [(fake_seg, fake_report)]}

        with (
            patch(
                "ntt.evaluation.lightning.run_validation_evaluation_per_session",
                side_effect=fake_run_validation,
            ) as mock_eval,
            patch(
                "ntt.evaluation.lightning.aggregate_segment_reports",
                return_value=fake_report,
            ),
        ):
            cb._run_eval_inner(trainer, module, prefix="eval", is_distributed=False)

        _, reconstruct_kwargs = module.tokenizer.reconstruct_from_input.call_args
        assert reconstruct_kwargs["return_latents"] is True
        assert mock_eval.called


# ---------------------------------------------------------------------------
# Poisson baseline tests
# ---------------------------------------------------------------------------


class TestPoissonBaselineCaching:
    def test_baseline_computed_once_and_cached(self):
        """Poisson baseline should be computed on first call, then reused."""
        cb = BehavioralDatasetEvalCallback(
            selections=_make_selections(),
            compute_poisson_baseline=True,
        )
        assert cb._poisson_baseline_metrics is None

        # Mock _compute_poisson_baseline_reports to track calls
        fake_baseline_report = MagicMock()
        fake_baseline_report.results = {}
        fake_baseline_reports = {
            "session_a": [(_fake_segment(), {"report": fake_baseline_report})]
        }
        cb._compute_poisson_baseline_reports = MagicMock(return_value=fake_baseline_reports)

        # Mock the rest of the eval pipeline
        with patch("ntt.evaluation.lightning.run_validation_evaluation_per_session") as mock_eval:
            # Setup mock returns
            fake_report = MagicMock()
            fake_report.results = {}
            mock_eval.return_value = {"session_a": [(_fake_segment(), {"report": fake_report})]}

            @dataclass
            class FakeSpec:
                bin_size_ms: float = 1.0
                input_type: DataType = DataType.SPIKES
                token_bin_size_ms: float = 10.0

            trainer = MagicMock(spec=pl.Trainer)
            trainer.logger = MagicMock()
            trainer.global_step = 100
            module = MagicMock(spec=pl.LightningModule)
            module.tokenizer_spec = FakeSpec()
            module.tokenizer = MagicMock()
            module.device = torch.device("cpu")

            fake_dm = MagicMock()
            fake_dm.val_dataloader.return_value = iter([])
            cb._cached_datamodule = fake_dm

            # First call: should compute baseline
            cb._run_eval_inner(trainer, module, prefix="eval", is_distributed=False)
            assert cb._compute_poisson_baseline_reports.call_count == 1
            assert cb._poisson_baseline_metrics is not None

            # Second call: should reuse cached baseline
            cb._run_eval_inner(trainer, module, prefix="eval", is_distributed=False)
            assert cb._compute_poisson_baseline_reports.call_count == 1  # still 1

    def test_baseline_disabled(self):
        """When compute_poisson_baseline=False, baseline is never computed."""
        cb = BehavioralDatasetEvalCallback(
            selections=_make_selections(),
            compute_poisson_baseline=False,
        )
        assert cb._poisson_baseline_metrics is None

        with patch("ntt.evaluation.lightning.run_validation_evaluation_per_session") as mock_eval:
            fake_report = MagicMock()
            fake_report.results = {}
            mock_eval.return_value = {"session_a": [(_fake_segment(), {"report": fake_report})]}

            @dataclass
            class FakeSpec:
                bin_size_ms: float = 1.0
                input_type: DataType = DataType.SPIKES
                token_bin_size_ms: float = 10.0

            trainer = MagicMock(spec=pl.Trainer)
            trainer.logger = MagicMock()
            trainer.global_step = 0
            module = MagicMock(spec=pl.LightningModule)
            module.tokenizer_spec = FakeSpec()
            module.tokenizer = MagicMock()
            module.device = torch.device("cpu")

            fake_dm = MagicMock()
            fake_dm.val_dataloader.return_value = iter([])
            cb._cached_datamodule = fake_dm

            cb._run_eval_inner(trainer, module, prefix="eval", is_distributed=False)
            assert cb._poisson_baseline_metrics is None

    def test_delta_metrics_logged_when_baseline_available(self):
        """When baseline is cached, delta metrics should be logged."""
        cb = BehavioralDatasetEvalCallback(
            selections=_make_selections(),
            compute_poisson_baseline=False,  # We'll set the cache manually
        )

        # Pre-populate the cache
        cb._poisson_baseline_metrics = {
            "behavioral_classification_trial_target_id": 0.25,
        }

        with patch("ntt.evaluation.lightning.run_validation_evaluation_per_session") as mock_eval:
            # Build a fake report with a matching metric
            from ntt.evaluation.core.types import (
                MetricCategory,
                MetricDirection,
                MetricLevel,
                MetricResult,
            )

            fake_result = MetricResult(
                name="behavioral_classification",
                category=MetricCategory.DOWNSTREAM_DECODING,
                direction=MetricDirection.HIGHER_IS_BETTER,
                level=MetricLevel.POPULATION,
                raw_value=0.45,
            )
            fake_report = MagicMock()
            fake_report.results = {
                "behavioral_classification_trial_target_id": fake_result,
            }
            mock_eval.return_value = {"session_a": [(_fake_segment(), {"report": fake_report})]}

            @dataclass
            class FakeSpec:
                bin_size_ms: float = 1.0
                input_type: DataType = DataType.SPIKES
                token_bin_size_ms: float = 10.0

            trainer = MagicMock(spec=pl.Trainer)
            trainer.logger = MagicMock()
            trainer.global_step = 500
            module = MagicMock(spec=pl.LightningModule)
            module.tokenizer_spec = FakeSpec()
            module.tokenizer = MagicMock()
            module.device = torch.device("cpu")

            fake_dm = MagicMock()
            fake_dm.val_dataloader.return_value = iter([])
            cb._cached_datamodule = fake_dm

            cb._run_eval_inner(trainer, module, prefix="eval", is_distributed=False)

            # Check what was logged
            logged = trainer.logger.log_metrics.call_args[0][0]
            assert "eval/behavioral_classification_trial_target_id" in logged
            assert logged["eval/behavioral_classification_trial_target_id"] == 0.45
            assert "eval/poisson_baseline/behavioral_classification_trial_target_id" in logged
            assert (
                logged["eval/poisson_baseline/behavioral_classification_trial_target_id"] == 0.25
            )
            assert "eval/delta_to_poisson/behavioral_classification_trial_target_id" in logged
            assert (
                abs(
                    logged["eval/delta_to_poisson/behavioral_classification_trial_target_id"]
                    - 0.20
                )
                < 1e-6
            )


# ---------------------------------------------------------------------------
# _report_to_metrics utility tests
# ---------------------------------------------------------------------------


class TestReportToMetrics:
    def test_scalar_values(self):
        from ntt.evaluation.core.types import (
            EvaluationReport,
            MetricCategory,
            MetricDirection,
            MetricLevel,
            MetricResult,
        )

        result = MetricResult(
            name="test_metric",
            category=MetricCategory.DOWNSTREAM_DECODING,
            direction=MetricDirection.HIGHER_IS_BETTER,
            level=MetricLevel.POPULATION,
            raw_value=0.75,
        )
        report = EvaluationReport(results={"my_metric": result})
        metrics = _report_to_metrics(report, "prefix")
        assert metrics == {"prefix/my_metric": 0.75}

    def test_array_values_averaged(self):
        from ntt.evaluation.core.types import (
            EvaluationReport,
            MetricCategory,
            MetricDirection,
            MetricLevel,
            MetricResult,
        )

        result = MetricResult(
            name="test_metric",
            category=MetricCategory.DOWNSTREAM_DECODING,
            direction=MetricDirection.HIGHER_IS_BETTER,
            level=MetricLevel.POPULATION,
            raw_value=np.array([0.5, 0.7, 0.9]),
        )
        report = EvaluationReport(results={"my_metric": result})
        metrics = _report_to_metrics(report, "prefix")
        assert abs(metrics["prefix/my_metric"] - 0.7) < 1e-6

    def test_nan_values_skipped(self):
        from ntt.evaluation.core.types import (
            EvaluationReport,
            MetricCategory,
            MetricDirection,
            MetricLevel,
            MetricResult,
        )

        result = MetricResult(
            name="test_metric",
            category=MetricCategory.DOWNSTREAM_DECODING,
            direction=MetricDirection.HIGHER_IS_BETTER,
            level=MetricLevel.POPULATION,
            raw_value=float("nan"),
        )
        report = EvaluationReport(results={"my_metric": result})
        metrics = _report_to_metrics(report, "prefix")
        assert "prefix/my_metric" not in metrics

    def test_empty_array_skipped(self):
        from ntt.evaluation.core.types import (
            EvaluationReport,
            MetricCategory,
            MetricDirection,
            MetricLevel,
            MetricResult,
        )

        result = MetricResult(
            name="test_metric",
            category=MetricCategory.DOWNSTREAM_DECODING,
            direction=MetricDirection.HIGHER_IS_BETTER,
            level=MetricLevel.POPULATION,
            raw_value=np.array([]),
        )
        report = EvaluationReport(results={"my_metric": result})
        metrics = _report_to_metrics(report, "prefix")
        assert "prefix/my_metric" not in metrics


# ---------------------------------------------------------------------------
# Datamodule config cloning test
# ---------------------------------------------------------------------------


class TestDatamoduleCloning:
    def test_deterministic_transforms_set(self):
        """Cloned datamodule config should NOT override deterministic_transforms.

        The callback does not force
        ``deterministic_transforms=True`` because ``NTTDataModule`` already
        forces deterministic unit dropout for val/test splits. The cloned
        config should therefore preserve the base value (here: False) and
        only override ``val_selections``, ``use_train_for_val``, and
        ``dataloader``.
        """
        cb = BehavioralDatasetEvalCallback(selections=_make_selections())

        # Create a mock base datamodule with a config
        @dataclass
        class FakeDataLoaderConfig:
            drop_last: bool = True

            def model_copy(self, update=None):
                new = FakeDataLoaderConfig(**{**self.__dict__, **(update or {})})
                return new

        @dataclass
        class FakeDataModuleConfig:
            dataloader: FakeDataLoaderConfig = field(default_factory=FakeDataLoaderConfig)
            val_selections: Any = None
            use_train_for_val: bool = True
            deterministic_transforms: bool = False

            def model_copy(self, deep=False, update=None):
                d = {**self.__dict__, **(update or {})}
                return FakeDataModuleConfig(**d)

        from ntt.core.data.datamodule import NTTDataModule

        # Create a real-looking mock that passes isinstance by patching the check
        base_dm = MagicMock()
        base_dm.config = FakeDataModuleConfig()

        trainer = MagicMock(spec=pl.Trainer)
        trainer.datamodule = base_dm

        module = MagicMock(spec=pl.LightningModule)
        module.tokenizer_spec = None

        # Patch both isinstance (to accept our mock) and NTTDataModule constructor
        with (
            patch("ntt.evaluation.lightning.NTTDataModule") as MockDM,
            patch("ntt.evaluation.lightning.isinstance", side_effect=lambda obj, cls: True),
        ):
            mock_instance = MagicMock()
            MockDM.return_value = mock_instance

            cb._get_or_build_datamodule(trainer, module, _make_selections())

            # Check the config passed to NTTDataModule
            call_kwargs = MockDM.call_args[1]
            # deterministic_transforms is NOT overridden by the callback
            # (NTTDataModule handles deterministic val/test unit dropout itself).
            assert call_kwargs["config"].deterministic_transforms is False
            assert call_kwargs["config"].use_train_for_val is False
            assert call_kwargs["config"].dataloader.drop_last is False
