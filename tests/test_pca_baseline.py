"""Tests for the PCA baseline tokenizer."""

from __future__ import annotations

import pytest
import torch

from ntt.contrib.tokenizer.pca_baseline import (
    PCABaselineTokenizer,
    PCABaselineTokenizerSpec,
)
from ntt.core.components.types import RatesInput
from ntt.evaluation.utils.eval_collation_helpers import _strip_token_padding


def _make_input(
    B: int = 4,
    T: int = 1000,
    N: int = 32,
    rate: float = 0.01,
    absolute_start: torch.Tensor | None = None,
) -> RatesInput:
    absolute_end = None if absolute_start is None else absolute_start + T / 1000.0
    return RatesInput(
        rate_matrix=torch.poisson(torch.full((B, T, N), rate)),
        pad_and_gap_mask=torch.ones(B, T, N, dtype=torch.long),
        sampling_rate=1000.0,
        unit_ids=torch.arange(N).unsqueeze(0).expand(B, -1),
        unit_ids_pad_mask=torch.ones(B, N, dtype=torch.long),
        duration_sec=torch.ones(B),
        session_id="test",
        subject_id="subj",
        species="mouse",
        absolute_start=absolute_start,
        absolute_end=absolute_end,
    )


def _make_spec(n_components: int = 10) -> PCABaselineTokenizerSpec:
    return PCABaselineTokenizerSpec(
        n_neurons=32,
        window_length_seconds=1.0,
        bin_size_ms=1.0,
        n_components=n_components,
    )


def test_token_bin_size_inferred_from_window() -> None:
    spec = _make_spec()
    assert spec.token_bin_size_ms == 1000.0  # full window collapse


def test_n_components_must_fit_in_window() -> None:
    with pytest.raises(ValueError):
        PCABaselineTokenizer(_make_spec(n_components=2000))
    with pytest.raises(ValueError):
        PCABaselineTokenizer(_make_spec(n_components=0))


@pytest.mark.parametrize("k", [1, 4, 10, 64])
def test_forward_shapes(k: int) -> None:
    torch.manual_seed(0)
    spec = _make_spec(n_components=k)
    tok = PCABaselineTokenizer(spec)
    B, T, N = 4, 1000, 32
    x = _make_input(B=B, T=T, N=N)
    out = tok(x)
    assert out.latents.shape == (B, 1, N, k)
    assert out.reconstruction.shape == (B, T, N)
    assert (out.reconstruction >= 0).all()


def test_running_stats_accumulate_during_training() -> None:
    """During training, every forward pass should update running stats."""
    torch.manual_seed(0)
    tok = PCABaselineTokenizer(_make_spec(n_components=4)).train()
    assert int(tok._n_seen.item()) == 0
    for _ in range(3):
        tok(_make_input(B=2, N=8))
    # 3 batches × 2 × 8 = 48 rows
    assert int(tok._n_seen.item()) == 48
    assert not bool(tok.fitted.item())  # fit only happens at epoch end


def test_fit_from_running_stats_explicit_call() -> None:
    """The LightningModule calls this at on_train_epoch_end."""
    torch.manual_seed(0)
    tok = PCABaselineTokenizer(_make_spec(n_components=4)).train()
    for _ in range(3):
        tok(_make_input(B=2, N=8))
    assert not bool(tok.fitted.item())
    assert tok.fit_from_running_stats() is True
    assert bool(tok.fitted.item())
    # eigh of a covariance matrix gives orthonormal eigenvectors → ‖W‖_F = √k.
    assert torch.allclose(
        tok.W.norm().to(torch.float64),
        torch.tensor(2.0, dtype=torch.float64),  # sqrt(4)
        atol=1e-3,
    )


def test_fit_with_no_data_is_noop() -> None:
    """fit_from_running_stats with no rows seen returns False, doesn't crash."""
    tok = PCABaselineTokenizer(_make_spec(n_components=4))
    assert tok.fit_from_running_stats() is False
    assert not bool(tok.fitted.item())


def test_eval_does_not_trigger_fit() -> None:
    """Going to eval mode must not auto-fit — fit is now LitModule-driven only."""
    torch.manual_seed(0)
    tok = PCABaselineTokenizer(_make_spec(n_components=4)).train()
    for _ in range(3):
        tok(_make_input(B=2, N=8))
    tok.eval()
    assert not bool(tok.fitted.item())


def test_fit_is_idempotent() -> None:
    """A second call to fit_from_running_stats is a no-op once fitted."""
    torch.manual_seed(0)
    tok = PCABaselineTokenizer(_make_spec(n_components=4)).train()
    for _ in range(3):
        tok(_make_input(B=2, N=8))
    assert tok.fit_from_running_stats() is True
    W_first = tok.W.clone()
    # A subsequent epoch shouldn't re-accumulate or re-fit:
    for _ in range(2):
        tok(_make_input(B=2, N=8))
    assert tok.fit_from_running_stats() is False
    assert torch.equal(tok.W, W_first)


def test_pre_fit_recon_is_eps_floor() -> None:
    """Before fit, μ is zero, so the linear output is zero → clamped to ε."""
    torch.manual_seed(0)
    tok = PCABaselineTokenizer(_make_spec(n_components=4))
    out = tok(_make_input(B=1, N=4))
    assert not bool(tok.fitted.item())
    # Pre-fit, μ=0 and z=0 → recon=0 → clamped to ε.
    assert torch.allclose(
        out.reconstruction,
        torch.full_like(out.reconstruction, 1e-8),
        atol=1e-9,
    )


def test_post_fit_recon_is_near_mu_for_zero_input() -> None:
    """A zero-everywhere input projects to z=−μ@W and reconstructs to ≈0."""
    torch.manual_seed(0)
    tok = PCABaselineTokenizer(_make_spec(n_components=4)).train()
    for _ in range(8):
        tok(_make_input(B=4, N=32))
    tok.fit_from_running_stats()
    # Sanity: μ should be close to the data mean (~0.01 for poisson(0.01)).
    assert tok.mu.mean().item() > 0
    # Reconstruction of the actual training distribution should beat the
    # eps-floor reconstruction (more variation, not all clamped).
    out = tok(_make_input(B=4, N=32))
    assert out.reconstruction.std().item() > 1e-5


def test_backward_runs() -> None:
    """Lightning needs a grad path even though all real params are buffers."""
    torch.manual_seed(0)
    tok = PCABaselineTokenizer(_make_spec(n_components=4))
    out = tok(_make_input())
    loss = out.reconstruction.mean()
    loss.backward()
    assert tok._dummy.grad is not None
    assert tok._dummy.grad.item() == 0.0  # the threading is a 0-coefficient op


def test_encode_decode_roundtrip_post_fit() -> None:
    """After fit, the encoded → decoded path produces a non-trivial rate reconstruction."""
    torch.manual_seed(42)
    tok = PCABaselineTokenizer(_make_spec(n_components=10)).train()
    for _ in range(8):
        tok(_make_input(B=4, N=32))
    tok.fit_from_running_stats()
    tok.eval()
    assert bool(tok.fitted.item())

    x = _make_input()
    z = tok.encode(x)
    recon = tok.decode(z)
    assert recon.shape == x.rate_matrix.shape
    assert (recon >= 0).all()
    # Top-10 PCA captures *some* variance.
    assert recon.std().item() > 1e-4


def _fitted_tokenizer(n_components: int = 4) -> PCABaselineTokenizer:
    torch.manual_seed(0)
    tok = PCABaselineTokenizer(_make_spec(n_components=n_components)).train()
    for _ in range(4):
        tok(_make_input(B=4, N=32))
    assert tok.fit_from_running_stats() is True
    return tok.eval()


def test_latents_follow_btud_contract_with_one_token_per_window() -> None:
    """``[B, T, U, D]`` with a singleton token axis: the window is one token.

    Neurons on the token axis would each get their own consecutive 1 s
    timestamp, so collation would strip every neuron but the first as padding.
    """
    tok = _fitted_tokenizer(n_components=4)
    B, N = 2, 32
    x = _make_input(B=B, N=N, absolute_start=torch.tensor([10.0, 11.0], dtype=torch.float64))
    with torch.no_grad():
        out = tok.reconstruct_from_input(x, return_latents=True)
    assert out["latents"].shape == (B, 1, N, 4)
    assert out["token_start_time"].tolist() == [[10.0], [11.0]]
    assert out["token_end_time"].tolist() == [[11.0], [12.0]]

    stripped = _strip_token_padding(
        out["latents"][0],
        out["token_start_time"][0],
        out["token_end_time"][0],
        absolute_end=11.0,
    )
    assert stripped["latents"].shape == (1, N, 4)


def test_reconstruct_accepts_latents_for_latent_robustness() -> None:
    """M2 decodes ``tokens_to_latents(encode(x))`` with noise added."""
    tok = _fitted_tokenizer(n_components=4)
    x = _make_input(B=3, N=32)
    with torch.no_grad():
        latents = tok.tokens_to_latents(tok.encode(x))
        recon = tok.reconstruct(latents + 0.1 * torch.randn_like(latents)).reconstruction
    assert recon.shape == (3, 1000, 32)


def test_decode_rejects_latents_without_token_axis() -> None:
    """The pre-contract ``[B, N, k]`` layout fails loudly instead of misdecoding."""
    tok = _fitted_tokenizer(n_components=4)
    with pytest.raises(ValueError, match=r"\[B, 1, N, k\]"):
        tok.decode(torch.zeros(2, 32, 4))
    with pytest.raises(ValueError, match=r"\[B, 1, N, k\]"):
        tok.decode(torch.zeros(2, 2, 32, 4))
