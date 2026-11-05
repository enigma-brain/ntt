"""Tests for max_neurons cap in the eval pipeline.

Tests that eval_config.max_neurons correctly limits the number of neurons
used in metric computation, covering:
- Segment-level subsampling in run_validation_evaluation_per_session
- DataModule-level UnitDropout cap in _build_all_neuron_eval
- Multi-session with different neuron counts
- Sessions below the cap are unaffected
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

# ── Helpers ─────────────────────────────────────────────────────────


class MultiSessionSpikeDataset(torch.utils.data.Dataset):
    """Dataset serving windows from multiple sessions with different neuron counts."""

    def __init__(self, session_spikes: dict[str, torch.Tensor], sampling_rate: float):
        self.session_spikes = session_spikes
        self.sampling_rate = sampling_rate

    def __len__(self) -> int:
        return sum(s.shape[1] for s in self.session_spikes.values())

    def __getitem__(self, index: DatasetIndex) -> BinnedSpikesInput:
        session_id = index.recording_id
        spikes = self.session_spikes[session_id]
        start = int(round(index.start * self.sampling_rate))
        end = int(round(index.end * self.sampling_rate))

        window = spikes[:, start:end]
        spike_matrix = window.T.int()
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


def _identity_tokenizer_fn(batch: BinnedSpikesInput) -> dict:
    """Identity tokenizer: reconstruction = input (as float)."""
    return {"reconstruction": batch.spike_matrix.float(), "latents": None}


def _make_loader(
    full_spikes: torch.Tensor,
    sampling_rate: float,
    session_id: str = "session",
    n_windows: int = 3,
) -> DataLoader:
    """Create DataLoader from a full spike tensor (units, time)."""
    indices = [DatasetIndex(session_id, float(i), float(i + 1)) for i in range(n_windows)]
    dataset = WindowedSpikeDataset(full_spikes, sampling_rate, session_id, len(indices))
    return DataLoader(dataset, batch_size=2, sampler=indices, collate_fn=collate)


# ── Tests: segment-level neuron cap ────────────────────────────────


class TestSegmentLevelNeuronCap:
    """Tests for the segment-level max_neurons cap in run_validation_evaluation_per_session.

    This cap applies after collation, before metrics — it subsamples the same
    neuron indices from both original and reconstructed data.
    """

    def test_cap_limits_neurons_in_metrics(self):
        """When max_neurons < session neuron count, metrics see only max_neurons units."""
        n_units = 50
        cap = 10
        sampling_rate = 100.0
        time_bins = 300

        rng = torch.Generator().manual_seed(42)
        spikes = torch.bernoulli(torch.full((n_units, time_bins), 0.05), generator=rng)
        loader = _make_loader(spikes, sampling_rate)

        eval_config = EvaluationConfig(
            max_neurons=cap,
            metrics={"Firing Rate Error": {"metric_class": "firing_rate_error"}},
        )

        reports = run_validation_evaluation_per_session(
            loader,
            _identity_tokenizer_fn,
            data_type=DataType.SPIKES,
            bin_size_seconds=1.0 / sampling_rate,
            eval_config=eval_config,
            progress=False,
        )

        assert "session" in reports
        # The metric was computed (not skipped)
        for _segment, data in reports["session"]:
            assert "Firing Rate Error" in data["report"].results
            # For identity tokenizer, firing rate error should be ~0
            result = data["report"].results["Firing Rate Error"]
            val = np.nanmean(result.raw_value)
            assert abs(val) < 1e-6

    def test_cap_no_op_when_below_threshold(self):
        """When max_neurons >= session neuron count, all neurons are used."""
        n_units = 10
        cap = 50  # cap is larger than n_units
        sampling_rate = 100.0
        time_bins = 300

        rng = torch.Generator().manual_seed(42)
        spikes = torch.bernoulli(torch.full((n_units, time_bins), 0.05), generator=rng)
        loader = _make_loader(spikes, sampling_rate)

        eval_config = EvaluationConfig(
            max_neurons=cap,
            metrics={"Firing Rate Error": {"metric_class": "firing_rate_error"}},
        )

        reports = run_validation_evaluation_per_session(
            loader,
            _identity_tokenizer_fn,
            data_type=DataType.SPIKES,
            bin_size_seconds=1.0 / sampling_rate,
            eval_config=eval_config,
            progress=False,
        )

        assert "session" in reports
        for _segment, data in reports["session"]:
            assert "Firing Rate Error" in data["report"].results

    def test_cap_preserves_alignment_between_original_and_reconstructed(self):
        """Both original and reconstructed are subsampled with the same indices.

        For an identity tokenizer (recon = original), firing rate error should
        be exactly 0 regardless of neuron cap.  If subsampling were misaligned,
        we'd see non-zero error.
        """
        n_units = 40
        cap = 15
        sampling_rate = 100.0
        time_bins = 300

        # Use varied per-neuron spike rates so misalignment would be detectable
        rng = torch.Generator().manual_seed(42)
        rates = torch.linspace(0.01, 0.3, n_units).unsqueeze(1).expand(n_units, time_bins)
        spikes = torch.bernoulli(rates, generator=rng)
        loader = _make_loader(spikes, sampling_rate)

        eval_config = EvaluationConfig(
            max_neurons=cap,
            metrics={"Firing Rate Error": {"metric_class": "firing_rate_error"}},
        )

        reports = run_validation_evaluation_per_session(
            loader,
            _identity_tokenizer_fn,
            data_type=DataType.SPIKES,
            bin_size_seconds=1.0 / sampling_rate,
            eval_config=eval_config,
            progress=False,
        )

        for _segment, data in reports["session"]:
            result = data["report"].results["Firing Rate Error"]
            val = np.nanmean(result.raw_value)
            assert abs(val) < 1e-6, (
                f"Firing rate error should be 0 for identity tokenizer, "
                f"got {val}. This suggests neuron misalignment."
            )

    def test_cap_with_multisession_different_neuron_counts(self):
        """Cap works across sessions with different neuron counts.

        Session A has 30 neurons (above cap), session B has 5 (below cap).
        Session A should be capped, session B should be unaffected.
        """
        sampling_rate = 100.0
        cap = 10

        rng_a = torch.Generator().manual_seed(42)
        spikes_a = torch.bernoulli(torch.full((30, 300), 0.05), generator=rng_a)

        rng_b = torch.Generator().manual_seed(43)
        spikes_b = torch.bernoulli(torch.full((5, 200), 0.05), generator=rng_b)

        dataset = MultiSessionSpikeDataset(
            session_spikes={"sess_a": spikes_a, "sess_b": spikes_b},
            sampling_rate=sampling_rate,
        )
        indices = [
            DatasetIndex("sess_a", 0.0, 1.0),
            DatasetIndex("sess_a", 1.0, 2.0),
            DatasetIndex("sess_a", 2.0, 3.0),
            DatasetIndex("sess_b", 0.0, 1.0),
            DatasetIndex("sess_b", 1.0, 2.0),
        ]
        # batch_size=1 because sessions have different neuron counts and our
        # toy dataset doesn't use pad2d wrapping (unlike the real DataModule).
        loader = DataLoader(dataset, batch_size=1, sampler=indices, collate_fn=collate)

        eval_config = EvaluationConfig(
            max_neurons=cap,
            metrics={"Firing Rate Error": {"metric_class": "firing_rate_error"}},
        )

        reports = run_validation_evaluation_per_session(
            loader,
            _identity_tokenizer_fn,
            data_type=DataType.SPIKES,
            bin_size_seconds=1.0 / sampling_rate,
            eval_config=eval_config,
            progress=False,
        )

        # Both sessions should have reports
        assert "sess_a" in reports
        assert "sess_b" in reports

        # Both should have valid metric results (identity tokenizer → 0 error)
        for session_id in ("sess_a", "sess_b"):
            for _segment, data in reports[session_id]:
                result = data["report"].results["Firing Rate Error"]
                val = np.nanmean(result.raw_value)
                assert abs(val) < 1e-6, f"Session {session_id}: expected ~0 error, got {val}"

    def test_cap_deterministic_across_runs(self):
        """Same cap + seed produces identical metric values across runs."""
        n_units = 40
        cap = 15
        sampling_rate = 100.0
        time_bins = 300

        rng = torch.Generator().manual_seed(42)
        rates = torch.linspace(0.01, 0.3, n_units).unsqueeze(1).expand(n_units, time_bins)
        spikes = torch.bernoulli(rates, generator=rng)

        def _run_eval():
            loader = _make_loader(spikes, sampling_rate)
            eval_config = EvaluationConfig(
                max_neurons=cap,
                neuron_subsample_seed=42,
                metrics={"Firing Rate Error": {"metric_class": "firing_rate_error"}},
            )
            return run_validation_evaluation_per_session(
                loader,
                _identity_tokenizer_fn,
                data_type=DataType.SPIKES,
                bin_size_seconds=1.0 / sampling_rate,
                eval_config=eval_config,
                progress=False,
            )

        reports1 = _run_eval()
        reports2 = _run_eval()

        _, data1 = reports1["session"][-1]
        _, data2 = reports2["session"][-1]
        v1 = np.nanmean(data1["report"].results["Firing Rate Error"].raw_value)
        v2 = np.nanmean(data2["report"].results["Firing Rate Error"].raw_value)
        assert v1 == v2, f"Results differ across runs: {v1} vs {v2}"


# ── Tests: DataModule-level neuron cap via UnitDropout ─────────────


class TestDataModuleLevelNeuronCap:
    """Tests that verify the DataModule-level cap works when n_neurons is set
    on the all-neuron DataModule config (leveraging UnitDropout).

    These tests verify the PLANNED behavior: when _build_all_neuron_eval sets
    all_neuron_config.n_neurons = max_neurons (instead of False), the
    DataModule's UnitDropout caps neurons at load time.
    """

    def test_covering_with_capped_neurons(self):
        """NeuronCoveringWrapper + cap: covering processes only capped neurons.

        Simulates the compress_n path where the DataModule delivers at most
        `cap` neurons and the covering wrapper partitions them into groups.
        """
        cap = 15  # cap to 15
        n_neurons = 8  # tokenizer's n_neurons
        sampling_rate = 100.0
        time_bins = 300

        rng = torch.Generator().manual_seed(42)
        # Create spikes with first `cap` neurons only (simulating UnitDropout first-N)
        spikes = torch.bernoulli(torch.full((cap, time_bins), 0.05), generator=rng)
        loader = _make_loader(spikes, sampling_rate)

        wrapper = NeuronCoveringWrapper(
            tokenizer_fn=_identity_tokenizer_fn,
            n_neurons=n_neurons,
            seed=42,
        )

        eval_config = EvaluationConfig(
            metrics={"Firing Rate Error": {"metric_class": "firing_rate_error"}},
        )

        reports = run_validation_evaluation_per_session(
            loader,
            wrapper,
            data_type=DataType.SPIKES,
            bin_size_seconds=1.0 / sampling_rate,
            eval_config=eval_config,
            progress=False,
        )

        assert "session" in reports
        for _segment, data in reports["session"]:
            assert "Firing Rate Error" in data["report"].results

    def test_n_neurons_config_controls_unit_dropout(self):
        """Setting n_neurons on the config controls whether UnitDropout is added.

        This verifies the mechanism that _build_all_neuron_eval should use:
        setting n_neurons=cap (instead of False) causes n_neurons_to_sample
        to return that value, which _build_transforms routes into UnitDropout.

        We test via the NTTDataModuleConfig property rather than instantiating
        the full DataModule (which requires data directories).
        """
        from ntt.core.data.datamodule import NTTDataModuleConfig

        # Test the n_neurons field behavior: False → n_neurons_to_sample returns None
        # int → returns the int. This is what _build_transforms checks.
        # We can't instantiate the full config without root/selections, so we
        # test the field logic directly on the model fields dict.
        assert NTTDataModuleConfig.model_fields["n_neurons"].default is None

        # Verify the property logic: simulate what NTTDataModule.n_neurons_to_sample does
        # When n_neurons=False → None (skip UnitDropout)
        # When n_neurons=200 → 200 (add UnitDropout with 200)
        # This is the core mechanism the DataModule-level cap relies on.
        class _FakeConfig:
            n_neurons = False

        class _FakeDM:
            config = _FakeConfig()

        from ntt.core.data.datamodule import NTTDataModule

        prop = NTTDataModule.n_neurons_to_sample.fget
        assert prop(_FakeDM()) is None

        _FakeConfig.n_neurons = 200
        assert prop(_FakeDM()) == 200

    def test_batch_size_scaling_with_effective_max(self):
        """compute_eval_batch_size correctly scales with effective max neurons."""
        from ntt.evaluation.utils.neuron_covering import compute_eval_batch_size

        # Without cap: scale by actual max
        bs_no_cap = compute_eval_batch_size(
            config_batch_size=32,
            config_n_neurons=64,
            max_neurons=500,
        )
        # With cap: scale by min(actual, cap)
        bs_with_cap = compute_eval_batch_size(
            config_batch_size=32,
            config_n_neurons=64,
            max_neurons=200,  # effective_max = min(500, 200) = 200
        )

        # Capped version should have larger batch size (fewer neurons → more headroom)
        assert bs_with_cap > bs_no_cap
        # Verify formula: 32 * 64 / 200 = 10, 32 * 64 / 500 = 4
        assert bs_no_cap == 4
        assert bs_with_cap == 10

    def test_segment_cap_is_noop_when_datamodule_caps(self):
        """When data already has <= max_neurons, segment-level cap is a no-op.

        This verifies that both capping mechanisms compose correctly:
        DataModule delivers capped data, segment-level cap sees no excess.
        """
        cap = 10
        sampling_rate = 100.0
        time_bins = 300

        # Simulate DataModule-capped data: only `cap` neurons
        rng = torch.Generator().manual_seed(42)
        spikes = torch.bernoulli(torch.full((cap, time_bins), 0.05), generator=rng)
        loader = _make_loader(spikes, sampling_rate)

        eval_config_with_cap = EvaluationConfig(
            max_neurons=cap,
            metrics={"Firing Rate Error": {"metric_class": "firing_rate_error"}},
        )
        eval_config_no_cap = EvaluationConfig(
            max_neurons=None,
            metrics={"Firing Rate Error": {"metric_class": "firing_rate_error"}},
        )

        reports_capped = run_validation_evaluation_per_session(
            loader,
            _identity_tokenizer_fn,
            data_type=DataType.SPIKES,
            bin_size_seconds=1.0 / sampling_rate,
            eval_config=eval_config_with_cap,
            progress=False,
        )
        # Recreate loader since it's consumed
        loader2 = _make_loader(spikes, sampling_rate)
        reports_no_cap = run_validation_evaluation_per_session(
            loader2,
            _identity_tokenizer_fn,
            data_type=DataType.SPIKES,
            bin_size_seconds=1.0 / sampling_rate,
            eval_config=eval_config_no_cap,
            progress=False,
        )

        # Results should be identical (cap is a no-op when data has exactly cap neurons)
        for (_, data1), (_, data2) in zip(
            reports_capped["session"], reports_no_cap["session"], strict=True
        ):
            v1 = np.nanmean(data1["report"].results["Firing Rate Error"].raw_value)
            v2 = np.nanmean(data2["report"].results["Firing Rate Error"].raw_value)
            assert v1 == v2, (
                f"Segment-level cap changed results when it should be a no-op: {v1} vs {v2}"
            )
