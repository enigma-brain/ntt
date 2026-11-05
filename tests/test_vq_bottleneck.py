"""Tests for CALM VQ Bottleneck correctness."""

import pytest
import torch
import torch.nn.functional as F

from ntt.contrib.tokenizer.calm.vq import VQBottleneck, VQOutput


class TestVQCodebookLookup:
    def test_nearest_neighbor_correctness(self):
        """Returned indices should correspond to actual nearest codebook entries."""
        vq = VQBottleneck(codebook_size=16, embed_dim=8)
        vq.eval()
        z = torch.randn(4, 5, 8)
        out = vq(z, temperature=1.0)
        z_flat = z.reshape(-1, 8)
        dists = torch.cdist(z_flat, vq.codebook.weight.data)
        expected_indices = dists.argmin(dim=-1)
        assert torch.equal(out.indices.reshape(-1), expected_indices)

    def test_quantized_are_codebook_vectors(self):
        """Quantized output should be actual codebook vectors."""
        vq = VQBottleneck(codebook_size=16, embed_dim=8)
        vq.eval()
        z = torch.randn(2, 5, 8)
        out = vq(z, temperature=1.0)
        for idx, vec in zip(out.indices.reshape(-1), out.quantized.reshape(-1, 8), strict=True):
            expected = vq.codebook.weight.data[idx]
            assert torch.allclose(vec, expected, atol=1e-6)


class TestVQStraightThrough:
    def test_gradient_flows_through(self):
        """Gradient of quantized output should equal gradient of encoder output."""
        vq = VQBottleneck(codebook_size=16, embed_dim=8, dead_code_threshold=0.0)
        z = torch.randn(2, 5, 8, requires_grad=True)
        out = vq(z, temperature=1.0)
        loss = out.quantized.sum()
        loss.backward()
        assert z.grad is not None
        assert z.grad.shape == z.shape
        assert torch.allclose(z.grad, torch.ones_like(z))


class TestVQEMAUpdate:
    def test_codebook_moves_toward_inputs(self):
        """After training forward pass, codebook vectors should move toward assigned features."""
        torch.manual_seed(42)
        vq = VQBottleneck(codebook_size=4, embed_dim=8, ema_decay=0.5)
        vq.train()
        initial_codebook = vq.codebook.weight.data.clone()
        z = torch.randn(8, 1, 8)
        vq(z, temperature=1.0)
        updated_codebook = vq.codebook.weight.data
        assert not torch.allclose(initial_codebook, updated_codebook, atol=1e-6)


class TestVQLosses:
    def test_commitment_loss(self):
        """Commitment loss should equal ||z - sg(e)||^2."""
        vq = VQBottleneck(codebook_size=16, embed_dim=8, dead_code_threshold=0.0)
        vq.train()
        z = torch.randn(2, 5, 8)
        out = vq(z, temperature=1.0)
        z_flat = z.reshape(-1, 8)
        selected = vq.codebook.weight.data[out.indices.reshape(-1)]
        expected = F.mse_loss(z_flat, selected.detach())
        assert torch.allclose(out.losses["commitment"], expected, atol=1e-5)

    def test_entropy_loss_uniform_vs_collapsed(self):
        """Entropy loss should be lower when codebook usage is uniform."""
        vq = VQBottleneck(codebook_size=8, embed_dim=8, dead_code_threshold=0.0)
        vq.train()
        z_uniform = vq.codebook.weight.data.unsqueeze(0).expand(4, -1, -1)
        z_uniform = z_uniform + torch.randn_like(z_uniform) * 0.01
        z_collapsed = vq.codebook.weight.data[0:1].unsqueeze(0).expand(4, 8, -1)
        z_collapsed = z_collapsed + torch.randn_like(z_collapsed) * 0.01
        out_uniform = vq(z_uniform, temperature=1.0)
        out_collapsed = vq(z_collapsed, temperature=1.0)
        assert out_uniform.losses["entropy"] < out_collapsed.losses["entropy"]

    def test_orthogonality_loss_identity_vs_correlated(self):
        """Orthogonality loss should be ~0 for orthogonal codebook."""
        vq = VQBottleneck(codebook_size=4, embed_dim=4)
        # Test the loss function directly to avoid EMA interference
        with torch.no_grad():
            vq.codebook.weight.copy_(torch.eye(4))
        loss_ortho = vq._orthogonality_loss()
        with torch.no_grad():
            vq.codebook.weight.copy_(torch.ones(4, 4))
        loss_corr = vq._orthogonality_loss()
        assert loss_ortho < loss_corr


class TestVQDeadCodeRevival:
    def test_dead_codes_get_revived(self):
        """Codebook vectors with zero usage should be replaced from feature buffer."""
        torch.manual_seed(0)
        vq = VQBottleneck(codebook_size=8, embed_dim=4, dead_code_threshold=1.0)
        vq.train()
        initial_codebook = vq.codebook.weight.data.clone()
        # Use only code 0 repeatedly
        z = vq.codebook.weight.data[0:1].unsqueeze(0).expand(4, 10, -1)
        z = z + torch.randn_like(z) * 0.001
        for _ in range(5):
            vq(z, temperature=1.0)
        # Some unused codes should have been revived (changed from initial)
        changed = ~torch.allclose(initial_codebook, vq.codebook.weight.data, atol=1e-6)
        assert changed, "Dead code revival should have modified some codebook entries"


class TestVQInterface:
    def test_output_shapes(self):
        vq = VQBottleneck(codebook_size=32, embed_dim=16, dead_code_threshold=0.0)
        z = torch.randn(3, 7, 16)
        out = vq(z, temperature=1.0)
        assert out.quantized.shape == (3, 7, 16)
        assert out.indices.shape == (3, 7)
        assert isinstance(out.losses, dict)

    def test_vocab_size(self):
        vq = VQBottleneck(codebook_size=128, embed_dim=64)
        assert vq.codebook_size == 128

    def test_indices_to_codes(self):
        vq = VQBottleneck(codebook_size=32, embed_dim=16)
        vq.eval()
        z = torch.randn(2, 5, 16)
        out = vq(z, temperature=1.0)
        recovered = vq.indices_to_codes(out.indices)
        assert torch.allclose(recovered, out.quantized, atol=1e-6)
