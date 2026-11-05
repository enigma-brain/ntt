"""Tests for full-neuron evaluation utilities."""

import numpy as np
import pytest
import torch

from ntt.core.components.types import BinnedSpikesInput
from ntt.evaluation.utils.neuron_covering import (
    CompressNLatentStrategy,
    NeuronCoveringWrapper,
    build_neuron_groups,
    compute_eval_batch_size,
    make_oom_safe_tokenize_fn,
    oom_safe_forward,
)


class TestComputeEvalBatchSize:
    def test_same_neuron_count_keeps_batch_size(self):
        assert (
            compute_eval_batch_size(config_batch_size=16, config_n_neurons=64, max_neurons=64)
            == 16
        )

    def test_double_neurons_halves_batch(self):
        assert (
            compute_eval_batch_size(config_batch_size=16, config_n_neurons=64, max_neurons=128)
            == 8
        )

    def test_many_neurons_clamps_to_one(self):
        assert (
            compute_eval_batch_size(config_batch_size=4, config_n_neurons=64, max_neurons=500) == 1
        )

    def test_fewer_neurons_increases_batch(self):
        assert (
            compute_eval_batch_size(config_batch_size=8, config_n_neurons=64, max_neurons=32) == 16
        )

    def test_n_neurons_false_returns_original(self):
        """When tokenizer uses all neurons (n_neurons=False), no scaling."""
        assert (
            compute_eval_batch_size(config_batch_size=16, config_n_neurons=None, max_neurons=500)
            == 16
        )


class TestBuildNeuronGroups:
    def test_exact_multiple(self):
        """128 neurons, groups of 64 -> 2 groups, no fillers."""
        groups, primary_masks = build_neuron_groups(n_total=128, n_per_group=64, seed=42)
        assert len(groups) == 2
        assert all(len(g) == 64 for g in groups)
        assert all(len(m) == 64 for m in primary_masks)
        # All neurons are primary (no fillers)
        assert all(all(m) for m in primary_masks)
        # Every neuron appears exactly once across primary slots
        all_primary = []
        for g, m in zip(groups, primary_masks, strict=True):
            all_primary.extend(g[i] for i in range(len(g)) if m[i])
        assert sorted(all_primary) == list(range(128))

    def test_partial_last_group(self):
        """200 neurons, groups of 64 -> 4 groups, last has 8 primary + 56 fillers."""
        groups, primary_masks = build_neuron_groups(n_total=200, n_per_group=64, seed=42)
        assert len(groups) == 4
        assert all(len(g) == 64 for g in groups)
        # Count primary neurons per group
        primary_counts = [sum(m) for m in primary_masks]
        # First 3 groups: 64 primary each, last: 8 primary
        assert sum(primary_counts) == 200  # every neuron primary exactly once
        assert primary_counts[-1] == 200 - 3 * 64  # == 8

    def test_every_neuron_covered(self):
        """Every neuron index appears as primary exactly once."""
        groups, primary_masks = build_neuron_groups(n_total=500, n_per_group=64, seed=0)
        primary_indices = []
        for g, m in zip(groups, primary_masks, strict=True):
            primary_indices.extend(g[i] for i in range(len(g)) if m[i])
        assert sorted(primary_indices) == list(range(500))

    def test_fewer_neurons_than_group_size(self):
        """30 neurons, group size 64 -> 1 group, 30 primary + 34 fillers."""
        groups, primary_masks = build_neuron_groups(n_total=30, n_per_group=64, seed=42)
        assert len(groups) == 1
        assert len(groups[0]) == 64
        assert sum(primary_masks[0]) == 30

    def test_deterministic(self):
        """Same seed -> same groups."""
        g1, m1 = build_neuron_groups(n_total=200, n_per_group=64, seed=42)
        g2, m2 = build_neuron_groups(n_total=200, n_per_group=64, seed=42)
        assert g1 == g2
        assert m1 == m2

    def test_different_seeds_differ(self):
        """Different seeds -> different permutations."""
        g1, _ = build_neuron_groups(n_total=200, n_per_group=64, seed=42)
        g2, _ = build_neuron_groups(n_total=200, n_per_group=64, seed=99)
        assert g1 != g2


def _make_identity_tokenizer_fn(n_neurons: int):
    """Fake compress_n tokenizer: returns input as reconstruction."""

    def fn(batch: BinnedSpikesInput) -> dict:
        return {"reconstruction": batch.spike_matrix.float(), "latents": None}

    return fn


def _make_latent_tokenizer_fn(latent_tokens: int = 2, latent_dim: int = 3):
    """Fake tokenizer that returns per-group latents plus timestamps."""

    def fn(batch: BinnedSpikesInput) -> dict:
        B = batch.spike_matrix.shape[0]
        n_units = batch.spike_matrix.shape[2]
        latents = torch.arange(
            B * latent_tokens * n_units * latent_dim,
            dtype=torch.float32,
        ).reshape(B, latent_tokens, n_units, latent_dim)
        indices = torch.arange(latent_tokens, dtype=torch.float64)
        abs_start = batch.absolute_start.to(dtype=torch.float64)
        token_start_time = abs_start[:, None] + indices[None, :] * 0.1
        token_end_time = token_start_time + 0.1
        token_timestamps = token_start_time + 0.05
        return {
            "reconstruction": batch.spike_matrix.float(),
            "latents": latents,
            "token_timestamps": token_timestamps,
            "token_start_time": token_start_time,
            "token_end_time": token_end_time,
        }

    return fn


def _make_batch(n_units: int, time_bins: int = 100) -> BinnedSpikesInput:
    """Create a minimal BinnedSpikesInput for testing."""
    spike_matrix = torch.randint(0, 2, (1, time_bins, n_units), dtype=torch.int32)
    return BinnedSpikesInput(
        spike_matrix=spike_matrix,
        pad_and_gap_mask=torch.ones_like(spike_matrix),
        sampling_rate=1000.0,
        unit_ids=torch.arange(n_units, dtype=torch.int32).unsqueeze(0),
        unit_ids_pad_mask=torch.ones(1, n_units, dtype=torch.int32),
        duration_sec=torch.tensor([time_bins / 1000.0]),
        session_id=("test_session",),
        subject_id=("test_subject",),
        species=("test_species",),
        absolute_start=torch.tensor([0.0], dtype=torch.float64),
        absolute_end=torch.tensor([time_bins / 1000.0], dtype=torch.float64),
        behavioral_data=None,
    )


class TestNeuronCoveringWrapper:
    def test_output_shape_matches_input(self):
        """Wrapper returns reconstruction with same neuron count as input."""
        n_units = 200
        n_neurons = 64
        batch = _make_batch(n_units)
        wrapper = NeuronCoveringWrapper(
            tokenizer_fn=_make_identity_tokenizer_fn(n_neurons),
            n_neurons=n_neurons,
            seed=42,
        )
        result = wrapper(batch)
        assert "reconstruction" in result
        recon = result["reconstruction"]
        assert recon.shape == (1, 100, n_units)

    def test_identity_reconstruction_preserves_data(self):
        """With identity tokenizer, each neuron's data matches original."""
        n_units = 130
        n_neurons = 64
        batch = _make_batch(n_units)
        wrapper = NeuronCoveringWrapper(
            tokenizer_fn=_make_identity_tokenizer_fn(n_neurons),
            n_neurons=n_neurons,
            seed=42,
        )
        result = wrapper(batch)
        recon = result["reconstruction"]
        torch.testing.assert_close(recon, batch.spike_matrix.float(), atol=0, rtol=0)

    def test_fewer_neurons_than_group_size(self):
        """Session with fewer neurons than n_neurons still works."""
        n_units = 30
        n_neurons = 64
        batch = _make_batch(n_units)
        wrapper = NeuronCoveringWrapper(
            tokenizer_fn=_make_identity_tokenizer_fn(n_neurons),
            n_neurons=n_neurons,
            seed=42,
        )
        result = wrapper(batch)
        recon = result["reconstruction"]
        assert recon.shape == (1, 100, n_units)

    def test_latents_are_none(self):
        """Wrapper sets latents to None for compress_n."""
        batch = _make_batch(200)
        wrapper = NeuronCoveringWrapper(
            tokenizer_fn=_make_identity_tokenizer_fn(64),
            n_neurons=64,
            seed=42,
        )
        result = wrapper(batch)
        assert result.get("latents") is None

    def test_concatenates_latents_from_all_groups(self):
        """Latents are concatenated across all groups, including the padded last group."""
        batch = _make_batch(130)
        wrapper = NeuronCoveringWrapper(
            tokenizer_fn=_make_latent_tokenizer_fn(latent_tokens=2, latent_dim=3),
            n_neurons=64,
            seed=42,
        )

        result = wrapper(batch)

        assert isinstance(result["latents"], torch.Tensor)
        assert result["latents"].shape == (1, 2, 64 * 3, 3)
        assert result["token_timestamps"].shape == (1, 2)
        assert result["token_start_time"].shape == (1, 2)
        assert result["token_end_time"].shape == (1, 2)

    def test_returns_ragged_latents_for_mixed_group_counts(self):
        """Mixed batches keep per-sample latent tensors when group counts differ."""
        spike_matrix = torch.randint(0, 2, (2, 100, 130), dtype=torch.int32)
        pad_and_gap_mask = torch.ones_like(spike_matrix)
        unit_ids = torch.arange(130, dtype=torch.int32).unsqueeze(0).expand(2, -1).clone()
        unit_ids_pad_mask = torch.zeros(2, 130, dtype=torch.int32)
        unit_ids_pad_mask[0, :64] = 1
        unit_ids_pad_mask[1, :130] = 1

        batch = BinnedSpikesInput(
            spike_matrix=spike_matrix,
            pad_and_gap_mask=pad_and_gap_mask,
            sampling_rate=1000.0,
            unit_ids=unit_ids,
            unit_ids_pad_mask=unit_ids_pad_mask,
            duration_sec=torch.tensor([0.1, 0.1]),
            session_id=("session_a", "session_b"),
            subject_id=("test_subject", "test_subject"),
            species=("test_species", "test_species"),
            absolute_start=torch.tensor([0.0, 0.0], dtype=torch.float64),
            absolute_end=torch.tensor([0.1, 0.1], dtype=torch.float64),
            behavioral_data=None,
        )
        wrapper = NeuronCoveringWrapper(
            tokenizer_fn=_make_latent_tokenizer_fn(latent_tokens=2, latent_dim=3),
            n_neurons=64,
            seed=42,
        )

        result = wrapper(batch)

        assert isinstance(result["latents"], list)
        assert result["latents"][0] is not None
        assert result["latents"][1] is not None
        assert result["latents"][0].shape == (2, 64, 3)
        assert result["latents"][1].shape == (2, 64 * 3, 3)

    def test_exclude_padded_group_drops_last_group_latents(self):
        """Optional strategy omits latents from the padded covering group."""
        batch = _make_batch(130)
        wrapper = NeuronCoveringWrapper(
            tokenizer_fn=_make_latent_tokenizer_fn(latent_tokens=2, latent_dim=3),
            n_neurons=64,
            seed=42,
            latent_strategy=CompressNLatentStrategy.EXCLUDE_PADDED_GROUP,
        )

        result = wrapper(batch)

        assert isinstance(result["latents"], torch.Tensor)
        assert result["latents"].shape == (1, 2, 64 * 2, 3)
        assert result["token_timestamps"].shape == (1, 2)

    def test_deterministic_across_calls(self):
        """Same wrapper produces same results on same input."""
        batch = _make_batch(200)
        wrapper = NeuronCoveringWrapper(
            tokenizer_fn=_make_identity_tokenizer_fn(64),
            n_neurons=64,
            seed=42,
        )
        r1 = wrapper(batch)
        r2 = wrapper(batch)
        torch.testing.assert_close(r1["reconstruction"], r2["reconstruction"])


class TestOomSafeForward:
    def test_normal_forward_succeeds(self):
        """When forward pass works, returns result normally."""

        def fn(batch):
            return {"reconstruction": batch.spike_matrix.float()}

        batch = _make_batch(64, time_bins=100)
        result, final_bs = oom_safe_forward(fn, batch, batch_size=1)
        assert result["reconstruction"].shape == (1, 100, 64)
        assert final_bs == 1

    def test_oom_halves_batch_size(self):
        """Simulates OOM by raising on large batches."""
        call_count = 0

        def fn(batch):
            nonlocal call_count
            call_count += 1
            if batch.spike_matrix.shape[0] > 2:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate ...")
            return {"reconstruction": batch.spike_matrix.float()}

        # Build a batch of 4 samples
        spike_matrix = torch.randint(0, 2, (4, 100, 64), dtype=torch.int32)
        batch = BinnedSpikesInput(
            spike_matrix=spike_matrix,
            pad_and_gap_mask=torch.ones_like(spike_matrix),
            sampling_rate=1000.0,
            unit_ids=torch.arange(64).unsqueeze(0).expand(4, -1),
            unit_ids_pad_mask=torch.ones(4, 64, dtype=torch.int32),
            duration_sec=torch.full((4,), 0.1),
            session_id=("s",) * 4,
            subject_id=("sub",) * 4,
            species=("sp",) * 4,
            absolute_start=torch.tensor([0.0, 0.1, 0.2, 0.3], dtype=torch.float64),
            absolute_end=torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64),
            behavioral_data=None,
        )

        result, final_bs = oom_safe_forward(fn, batch, batch_size=4)
        assert result["reconstruction"].shape == (4, 100, 64)
        assert final_bs <= 2  # Had to reduce from 4


class TestMakeOomSafeTokenizeFn:
    @staticmethod
    def _batch_of(n: int) -> BinnedSpikesInput:
        spike_matrix = torch.randint(0, 2, (n, 10, 8), dtype=torch.int32)
        return BinnedSpikesInput(
            spike_matrix=spike_matrix,
            pad_and_gap_mask=torch.ones_like(spike_matrix),
            sampling_rate=1000.0,
            unit_ids=torch.arange(8).unsqueeze(0).expand(n, -1),
            unit_ids_pad_mask=torch.ones(n, 8, dtype=torch.int32),
            duration_sec=torch.full((n,), 0.01),
            session_id=("s",) * n,
            subject_id=("sub",) * n,
            species=("sp",) * n,
            absolute_start=torch.arange(n, dtype=torch.float64) * 0.01,
            absolute_end=(torch.arange(n, dtype=torch.float64) + 1) * 0.01,
            behavioral_data=None,
        )

    def test_returns_unwrapped_output_when_batch_fits(self):
        """No OOM -> exactly fn(batch), not a re-packed dict (no behaviour change)."""
        sentinel = object()
        safe_fn = make_oom_safe_tokenize_fn(lambda batch: sentinel)
        assert safe_fn(self._batch_of(4)) is sentinel

    def test_oom_splits_batch_and_remembers_working_size(self):
        """After an OOM the batch is chunked, and later batches start at the
        size that fit instead of OOMing on the full batch again."""
        sizes_seen: list[int] = []

        def fn(batch):
            n = batch.spike_matrix.shape[0]
            sizes_seen.append(n)
            if n > 2:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory.")
            return {"reconstruction": batch.spike_matrix.float()}

        safe_fn = make_oom_safe_tokenize_fn(fn)
        first, second = self._batch_of(8), self._batch_of(8)

        out1 = safe_fn(first)
        n_calls_first = len(sizes_seen)
        out2 = safe_fn(second)

        torch.testing.assert_close(out1["reconstruction"], first.spike_matrix.float())
        torch.testing.assert_close(out2["reconstruction"], second.spike_matrix.float())
        assert sizes_seen[n_calls_first:] == [2, 2, 2, 2]  # no retry at 8 or 4

    def test_oom_at_batch_size_one_is_raised(self):
        def fn(batch):
            raise torch.cuda.OutOfMemoryError("CUDA out of memory.")

        with pytest.raises(torch.cuda.OutOfMemoryError):
            make_oom_safe_tokenize_fn(fn)(self._batch_of(1))
