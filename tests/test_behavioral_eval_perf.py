"""Tests for behavioral eval performance improvements.

Covers:
- compute_groups vectorization (equivalence with reference impl for all transitions)
- Batched TCN training (padded batch helpers, masked loss)
- Auto device resolution
- TCN YAML config instantiation
"""

import numpy as np
import pytest
import torch

from ntt.evaluation.metrics.downstream_prediction import (
    _masked_mse_loss,
    _pad_sequences_to_batch,
    _resolve_device,
    compute_groups,
)

# ── Reference implementation for compute_groups ──────────────────────────────


def _compute_groups_reference(values, transition="change"):
    """Original pure-Python loop implementation, kept as ground truth."""
    if len(values) == 0:
        return np.array([], dtype=int)
    groups = np.zeros(len(values), dtype=int)
    group_id = 0
    groups[0] = group_id
    for i in range(1, len(values)):
        prev_val = values[i - 1]
        curr_val = values[i]
        is_new_group = False
        if transition == "change":
            is_new_group = curr_val != prev_val
        elif transition == "decrease":
            is_new_group = curr_val < prev_val
        elif transition == "increase":
            is_new_group = curr_val > prev_val
        elif transition == "from_zero":
            is_new_group = (prev_val <= 0) and (curr_val > 0)
        if is_new_group:
            group_id += 1
        groups[i] = group_id
    return groups


# ── compute_groups tests ─────────────────────────────────────────────────────


class TestComputeGroups:
    @pytest.mark.parametrize("transition", ["change", "decrease", "increase", "from_zero"])
    def test_matches_reference_random(self, transition):
        rng = np.random.default_rng(42)
        values = rng.integers(0, 5, size=500)
        expected = _compute_groups_reference(values, transition)
        result = compute_groups(values, transition)
        np.testing.assert_array_equal(result, expected)

    def test_empty_array(self):
        result = compute_groups(np.array([]))
        assert len(result) == 0

    def test_single_element(self):
        result = compute_groups(np.array([7]))
        np.testing.assert_array_equal(result, [0])

    def test_all_same_values(self):
        result = compute_groups(np.array([3, 3, 3, 3, 3]))
        np.testing.assert_array_equal(result, [0, 0, 0, 0, 0])

    def test_all_different_values(self):
        result = compute_groups(np.array([1, 2, 3, 4, 5]))
        np.testing.assert_array_equal(result, [0, 1, 2, 3, 4])

    @pytest.mark.parametrize("transition", ["change", "decrease", "increase", "from_zero"])
    def test_matches_reference_large(self, transition):
        """Stress test with a large array."""
        rng = np.random.default_rng(123)
        values = rng.integers(-2, 5, size=100_000)
        expected = _compute_groups_reference(values, transition)
        result = compute_groups(values, transition)
        np.testing.assert_array_equal(result, expected)

    def test_from_zero_specific(self):
        values = np.array([0, 0, 1, 2, 0, -1, 0, 3, 0, 1])
        expected = _compute_groups_reference(values, "from_zero")
        result = compute_groups(values, "from_zero")
        np.testing.assert_array_equal(result, expected)

    def test_decrease_specific(self):
        values = np.array([5, 4, 4, 3, 5, 2])
        expected = _compute_groups_reference(values, "decrease")
        result = compute_groups(values, "decrease")
        np.testing.assert_array_equal(result, expected)

    def test_unknown_transition_raises(self):
        with pytest.raises(ValueError, match="Unknown transition type"):
            compute_groups(np.array([1, 2, 3]), "bogus")


# ── Batched TCN helpers ──────────────────────────────────────────────────────


class TestPadSequencesToBatch:
    def test_uniform_lengths(self):
        seqs = [torch.randn(1, 3, 10), torch.randn(1, 3, 10)]
        batch, mask = _pad_sequences_to_batch(seqs)
        assert batch.shape == (2, 3, 10)
        assert mask.shape == (2, 10)
        assert mask.all()

    def test_variable_lengths(self):
        seqs = [torch.randn(1, 2, 5), torch.randn(1, 2, 10), torch.randn(1, 2, 3)]
        batch, mask = _pad_sequences_to_batch(seqs)
        assert batch.shape == (3, 2, 10)
        assert mask.shape == (3, 10)
        # First seq: 5 valid, 5 padded
        assert mask[0, :5].all()
        assert not mask[0, 5:].any()
        # Second seq: all valid
        assert mask[1].all()
        # Third seq: 3 valid, 7 padded
        assert mask[2, :3].all()
        assert not mask[2, 3:].any()

    def test_preserves_data(self):
        data = torch.tensor([[[1.0, 2.0, 3.0]]])
        seqs = [data, torch.randn(1, 1, 5)]
        batch, _ = _pad_sequences_to_batch(seqs)
        torch.testing.assert_close(batch[0, :, :3], data[0])
        # Padded positions should be zero
        assert (batch[0, :, 3:] == 0).all()


class TestMaskedMseLoss:
    def test_no_padding_equals_regular_mse(self):
        pred = torch.randn(4, 2, 10)
        target = torch.randn(4, 2, 10)
        mask = torch.ones(4, 10, dtype=torch.bool)
        masked = _masked_mse_loss(pred, target, mask)
        regular = torch.nn.functional.mse_loss(pred, target)
        torch.testing.assert_close(masked, regular)

    def test_ignores_padded_positions(self):
        pred = torch.zeros(1, 1, 5)
        target = torch.zeros(1, 1, 5)
        # Only first 2 positions valid, rest are garbage
        pred[0, 0, :2] = torch.tensor([1.0, 2.0])
        target[0, 0, :2] = torch.tensor([1.0, 2.0])
        pred[0, 0, 2:] = 999.0  # should be ignored
        mask = torch.tensor([[True, True, False, False, False]])
        loss = _masked_mse_loss(pred, target, mask)
        assert loss.item() == pytest.approx(0.0)


# ── Device resolution ────────────────────────────────────────────────────────


class TestResolveDevice:
    def test_auto_returns_valid_device(self):
        device = _resolve_device("auto")
        assert device.type in ("cpu", "cuda")

    def test_cpu_explicit(self):
        device = _resolve_device("cpu")
        assert device.type == "cpu"

    def test_cuda_fallback(self):
        # If CUDA not available, should fall back gracefully
        device = _resolve_device("cuda")
        assert device.type in ("cpu", "cuda")
