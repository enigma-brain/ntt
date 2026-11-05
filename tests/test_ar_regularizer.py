"""Tests for the reusable AR Regularizer module."""

import pytest
import torch

from ntt.core.model.tokenizer.ar_regularizer import ARRegularizer


class TestARRegularizerShapes:
    def test_scalar_loss_output(self):
        ar = ARRegularizer(embed_dim=32, vocab_size=64, num_heads=4, horizon=1)
        z_seq = torch.randn(4, 10, 32)
        token_indices = torch.randint(0, 64, (4, 10))
        loss = ar(z_seq, token_indices)
        assert loss.shape == ()
        assert loss.requires_grad

    def test_various_shapes(self):
        ar = ARRegularizer(embed_dim=16, vocab_size=32, num_heads=2, horizon=1)
        for B, T, C in [(1, 5, 16), (8, 20, 16), (2, 3, 16)]:
            z = torch.randn(B, T, C)
            idx = torch.randint(0, 32, (B, T))
            loss = ar(z, idx)
            assert loss.shape == ()


class TestARRegularizerCausality:
    def test_future_independence(self):
        """Changing tokens at t+1 should not affect prediction at t."""
        ar = ARRegularizer(embed_dim=16, vocab_size=32, num_heads=2, horizon=1)
        ar.eval()
        z_seq = torch.randn(1, 10, 16)
        logits1 = ar.get_logits(z_seq)
        z_perturbed = z_seq.clone()
        z_perturbed[:, 5:] = torch.randn(1, 5, 16)
        logits2 = ar.get_logits(z_perturbed)
        assert torch.allclose(logits1[:, :5], logits2[:, :5], atol=1e-5)
        assert not torch.allclose(logits1[:, 5:], logits2[:, 5:], atol=1e-3)


class TestARRegularizerHorizon:
    def test_horizon_1(self):
        ar = ARRegularizer(embed_dim=16, vocab_size=32, num_heads=2, horizon=1)
        z = torch.randn(2, 10, 16)
        idx = torch.randint(0, 32, (2, 10))
        loss = ar(z, idx)
        assert loss.item() > 0

    def test_horizon_4(self):
        ar = ARRegularizer(embed_dim=16, vocab_size=32, num_heads=2, horizon=4)
        z = torch.randn(2, 10, 16)
        idx = torch.randint(0, 32, (2, 10))
        loss = ar(z, idx)
        assert loss.item() > 0


class TestARRegularizerLearning:
    def test_loss_decreases_on_repeating_pattern(self):
        torch.manual_seed(42)
        ar = ARRegularizer(embed_dim=16, vocab_size=8, num_heads=2, horizon=1)
        ar.train()
        optimizer = torch.optim.Adam(ar.parameters(), lr=1e-3)
        pattern = torch.arange(8).repeat(5)[:20]
        token_indices = pattern.unsqueeze(0).expand(4, -1)
        z_seq = torch.randn(4, 20, 16)
        losses = []
        for _ in range(50):
            loss = ar(z_seq, token_indices)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        assert losses[-1] < losses[0]


def test_ar_regularizer_accepts_dropout_arg():
    """ARRegularizer should accept a dropout kwarg and pass it to the encoder layer."""
    reg = ARRegularizer(embed_dim=16, vocab_size=32, num_heads=2, horizon=1, dropout=0.2)
    # TransformerEncoderLayer stores dropout on its Dropout modules.
    assert reg.attn_layer.dropout1.p == 0.2
    assert reg.attn_layer.dropout2.p == 0.2


def test_ar_regularizer_default_dropout_zero():
    reg = ARRegularizer(embed_dim=16, vocab_size=32, num_heads=2, horizon=1)
    assert reg.attn_layer.dropout1.p == 0.0
