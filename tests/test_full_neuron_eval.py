"""Integration tests for evaluate_all_neurons feature.

Tests the full pipeline: DataLoader -> NeuronCoveringWrapper -> collation -> metrics.
Uses the same test fixtures as test_validation_pipeline.py.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_brain.data.dataset import DatasetIndex

from ntt.core.components.types import BinnedSpikesInput
from ntt.core.data.datamodule import collate
from ntt.core.model.tokenizer.base import DataType
from ntt.evaluation.config import EvaluationConfig
from ntt.evaluation.utils.neuron_covering import NeuronCoveringWrapper
from ntt.evaluation.utils.validation_collation import (
    run_validation_evaluation_per_session,
)
from tests.fixtures import WindowedSpikeDataset


def _make_loader(
    full_spikes: torch.Tensor, sampling_rate: float, session_id: str = "session"
) -> DataLoader:
    """Create DataLoader from a full spike tensor (units, time)."""
    n_windows = 3
    indices = [DatasetIndex(session_id, float(i), float(i + 1)) for i in range(n_windows)]
    dataset = WindowedSpikeDataset(full_spikes, sampling_rate, session_id, len(indices))
    return DataLoader(dataset, batch_size=2, sampler=indices, collate_fn=collate)


def _identity_tokenizer_fn(batch: BinnedSpikesInput) -> dict:
    """Identity tokenizer: reconstruction = input (as float)."""
    return {"reconstruction": batch.spike_matrix.float(), "latents": None}


class TestFullNeuronEvalIntegration:
    def test_compress_n_covering_runs_full_pipeline(self):
        """NeuronCoveringWrapper produces valid results through full eval pipeline."""
        n_units = 20  # More than n_neurons=8
        n_neurons = 8
        sampling_rate = 100.0
        time_bins = 300  # 3 seconds at 100 Hz

        rng = torch.Generator().manual_seed(42)
        full_spikes = torch.bernoulli(torch.full((n_units, time_bins), 0.05), generator=rng)

        loader = _make_loader(full_spikes, sampling_rate)

        wrapper = NeuronCoveringWrapper(
            tokenizer_fn=_identity_tokenizer_fn,
            n_neurons=n_neurons,
            seed=42,
        )

        eval_config = EvaluationConfig(
            metrics={"Firing Rate Error": {"metric_class": "firing_rate_error"}}
        )

        reports = run_validation_evaluation_per_session(
            loader,
            wrapper,
            data_type=DataType.SPIKES,
            bin_size_seconds=1.0 / sampling_rate,
            eval_config=eval_config,
        )

        assert len(reports) > 0
        assert "session" in reports
        for _segment, report in reports["session"]:
            assert "Firing Rate Error" in report["report"].results

    def test_compress_t_passthrough_runs_full_pipeline(self):
        """All-neuron eval without covering wrapper runs the full pipeline."""
        n_units = 20
        sampling_rate = 100.0
        time_bins = 300

        rng = torch.Generator().manual_seed(42)
        full_spikes = torch.bernoulli(torch.full((n_units, time_bins), 0.05), generator=rng)

        loader = _make_loader(full_spikes, sampling_rate)

        eval_config = EvaluationConfig(
            metrics={"Firing Rate Error": {"metric_class": "firing_rate_error"}}
        )

        reports = run_validation_evaluation_per_session(
            loader,
            _identity_tokenizer_fn,
            data_type=DataType.SPIKES,
            bin_size_seconds=1.0 / sampling_rate,
            eval_config=eval_config,
        )

        assert len(reports) > 0
        assert "session" in reports

    def test_stashed_original_matches_subsampled_metric_view(self, tmp_path):
        """When ``eval_config.max_neurons`` caps neurons in-place,
        the on-disk ``stash_originals_dir`` array must match the post-cap view
        the metrics were computed on (not the pre-cap ``segment.original``).

        Otherwise multi-realisation Poisson averaging blows up later in
        ``NTTEvalJob._average_realisation_reports`` with
        ``ValueError: all input arrays must have the same shape`` because
        realisation 0's per-unit arrays have length ``max_neurons`` while
        realisations 1+ (which reload the stash) have length ``n_units``.
        """
        n_units = 20
        max_neurons = 8
        assert n_units > max_neurons, "trigger the post-load subsample cap"
        sampling_rate = 100.0
        time_bins = 300

        rng = torch.Generator().manual_seed(42)
        full_spikes = torch.bernoulli(torch.full((n_units, time_bins), 0.1), generator=rng)

        loader = _make_loader(full_spikes, sampling_rate)

        eval_config = EvaluationConfig(
            metrics={"Firing Rate Error": {"metric_class": "firing_rate_error"}},
            max_neurons=max_neurons,
            neuron_subsample_seed=42,
        )

        stash_dir = tmp_path / "stash"
        stash_dir.mkdir()

        reports = run_validation_evaluation_per_session(
            loader,
            _identity_tokenizer_fn,
            data_type=DataType.SPIKES,
            bin_size_seconds=1.0 / sampling_rate,
            eval_config=eval_config,
            stash_originals_dir=str(stash_dir),
        )

        assert "session" in reports
        for _segment, data in reports["session"]:
            metric = data["report"].results["Firing Rate Error"]
            # The post-cap metric view sees max_neurons units.
            assert metric.raw_value.shape == (max_neurons,), (
                f"metric saw shape {metric.raw_value.shape}, expected ({max_neurons},)"
            )

            # The stash on disk must match what the metric saw — not the
            # uncapped n_units view from segment.original.
            stash_path = data["metadata"]["_original_stash_path"]
            stashed = np.load(stash_path)
            assert stashed.shape[0] == max_neurons, (
                f"stash has {stashed.shape[0]} units, expected {max_neurons}; "
                f"realisations 1+ would produce shape-({stashed.shape[0]},) "
                f"per-unit arrays that cannot stack with realisation 0's "
                f"shape-({max_neurons},) arrays."
            )
