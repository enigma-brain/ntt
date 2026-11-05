from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError
from torch_brain.data.collate import pad, pad2d

from ntt.contrib.tokenizer.perceiver.perceiver_tokenizer import (
    MaskingConfig,
    PerceiverTokenizer,
    PerceiverTokenizerSpec,
)
from ntt.core.components.types import BinnedSpikesInput
from ntt.core.data.components.collate import neuron_split_collate

D_EMBED = 8
N_UNITS_PER_SESSION = 16
NUM_SESSIONS = 4


def _write_embeddings_file(
    tmp_path: Path,
    *,
    n_sessions: int = NUM_SESSIONS,
    n_units: int = N_UNITS_PER_SESSION,
    d_embed: int = D_EMBED,
    seed: int = 1234,
) -> Path:
    """Write a synthetic embeddings .pt file matching the format _load_embeddings expects.

    Each session gets n_units rows of d_embed-dim random embeddings.
    Sessions are named ``session_0`` … ``session_{n_sessions-1}``.
    Unit_ids encode ``session/local_index`` so the loader's lexsort works.
    """
    gen = torch.Generator().manual_seed(seed)
    embs = torch.randn(n_sessions * n_units, d_embed, generator=gen)
    session_ids = []
    unit_ids = []
    for s in range(n_sessions):
        for u in range(n_units):
            session_ids.append(f"session_{s}")
            unit_ids.append(f"session_{s}/{u}")
    path = tmp_path / "synthetic_embeddings.pt"
    torch.save(
        {
            "embs": embs,
            "unit_ids": np.asarray(unit_ids),
            "session_ids": np.asarray(session_ids),
        },
        path,
    )
    return path


@pytest.fixture(scope="module")
def embeddings_file(tmp_path_factory) -> Path:
    """Module-scoped temp embeddings file shared across tests."""
    return _write_embeddings_file(tmp_path_factory.mktemp("perceiver_embs"))


def _make_batched_input(
    batch_size: int = 2,
    n_units: int = 12,
    n_bins: int = 8,
    *,
    seed: int = 2026,
    real_neurons_per_sample: list[int] | None = None,
) -> BinnedSpikesInput:
    """Build a batched BinnedSpikesInput with deterministic spikes.

    ``real_neurons_per_sample``: if given, the per-sample number of real
    (non-padded) neurons; padded slots use unit_ids_pad_mask=0.
    """
    gen = torch.Generator().manual_seed(seed)
    spike_matrix = torch.poisson(torch.full((batch_size, n_bins, n_units), 0.4), generator=gen)
    if real_neurons_per_sample is None:
        unit_mask = torch.ones(batch_size, n_units, dtype=torch.long)
    else:
        assert len(real_neurons_per_sample) == batch_size
        unit_mask = torch.zeros(batch_size, n_units, dtype=torch.long)
        for b, n_real in enumerate(real_neurons_per_sample):
            unit_mask[b, :n_real] = 1
    return BinnedSpikesInput(
        spike_matrix=spike_matrix,
        pad_and_gap_mask=torch.ones_like(spike_matrix),
        sampling_rate=torch.full((batch_size,), 100.0),
        unit_ids=torch.arange(n_units).expand(batch_size, -1).contiguous(),
        unit_ids_pad_mask=unit_mask,
        duration_sec=torch.full((batch_size,), n_bins / 100.0),
        session_id=tuple(f"session_{i}" for i in range(batch_size)),
        subject_id=tuple(f"subject_{i}" for i in range(batch_size)),
        species=("mouse",) * batch_size,
        absolute_start=torch.zeros(batch_size),
        absolute_end=torch.full((batch_size,), n_bins / 100.0),
    )


def _make_unbatched_input(n_units: int = 10, n_bins: int = 6) -> BinnedSpikesInput:
    spike_matrix = torch.arange(n_bins * n_units).reshape(n_bins, n_units)
    return BinnedSpikesInput(
        spike_matrix=pad2d(spike_matrix),
        pad_and_gap_mask=pad2d(torch.ones_like(spike_matrix)),
        sampling_rate=100.0,
        unit_ids=pad(torch.arange(n_units)),
        unit_ids_pad_mask=pad(torch.ones(n_units, dtype=torch.long)),
        duration_sec=torch.tensor(n_bins / 100.0),
        session_id="session_0",
        subject_id="subject_0",
        species="mouse",
        absolute_start=torch.tensor(0.0),
        absolute_end=torch.tensor(n_bins / 100.0),
    )


def _make_tokenizer(embeddings_file: Path, **overrides) -> PerceiverTokenizer:
    """Construct a PerceiverTokenizer via its real __init__ using a synthetic .pt file."""
    spec = PerceiverTokenizerSpec(
        hidden_dim=16,
        latent_tokens=4,
        latent_dim=4,
        d_embed=D_EMBED,
        num_heads=4,
        mlp_ratio=1,
        num_encoder_N_blocks=1,
        num_encoder_M_blocks=1,
        num_decoder_blocks=1,
        use_s4=False,
        grad_checkpointing=False,
        use_layernorm=True,
        window_length_seconds=0.08,
        bin_size_ms=10.0,
        embeddings_path=str(embeddings_file),
        **overrides,
    )
    return PerceiverTokenizer(spec)


# ---------------------------------------------------------------------------
# Collate / masking unit tests (no tokenizer construction needed)
# ---------------------------------------------------------------------------


def test_neuron_split_collate_bounds_units_and_preserves_counts():
    sample = _make_unbatched_input(n_units=10, n_bins=6)

    batch = neuron_split_collate([sample], max_neurons=4)

    assert batch.spike_matrix.shape == (3, 6, 4)
    assert batch.unit_ids_pad_mask.sum(dim=1).tolist() == [4, 4, 2]
    assert batch.spike_matrix[0, :, :4].equal(torch.arange(60).reshape(6, 10)[:, :4])
    assert batch.spike_matrix[2, :, 2:].sum() == 0


def test_neuron_only_masking_with_all_loss_keeps_decoder_target_full():
    torch.manual_seed(1)
    spikes = torch.randn(2, 10, 8).abs()
    embeddings = torch.randn(2, 10, 8)
    real_mask = torch.ones(2, 10, dtype=torch.bool)
    real_mask[0, 8:] = False
    cfg = MaskingConfig(neuron_ratio=0.5, temporal_ratio=0.0, loss_on_all_neurons=True)

    masked = cfg.apply(spikes, embeddings, real_mask)

    assert masked.enc_spikes.shape[1] < spikes.shape[1]
    assert masked.targets.data_ptr() == spikes.data_ptr()
    assert torch.equal(masked.loss_mask, real_mask.float().unsqueeze(-1).expand_as(spikes))
    assert masked.loss_mask[0, 8:].sum() == 0


def test_neuron_only_masking_loss_on_all_neurons_false_restricts_loss_to_masked():
    """Default ``loss_on_all_neurons=False`` (MAE-style): loss only at masked neurons."""
    torch.manual_seed(11)
    B, N, L = 2, 10, 6
    spikes = torch.randn(B, N, L).abs()
    embeddings = torch.randn(B, N, 8)
    real_mask = torch.ones(B, N, dtype=torch.bool)
    cfg = MaskingConfig(neuron_ratio=0.5, temporal_ratio=0.0, loss_on_all_neurons=False)

    masked = cfg.apply(spikes, embeddings, real_mask)

    # Loss should be nonzero only at masked (held-out) neurons; zero at visible.
    # masked.loss_mask is (B, N, L) float; per-neuron sum == 0 ↔ that neuron is visible.
    per_neuron_loss = masked.loss_mask.sum(dim=-1)  # (B, N)
    held_out = per_neuron_loss > 0  # (B, N)
    n_held = held_out.sum(dim=1)
    n_expected = (real_mask.sum(dim=1).float() * 0.5).to(torch.long).clamp(min=1)
    assert torch.equal(n_held, n_expected)


def test_combined_neuron_and_temporal_masking_loss_mask_partition():
    """temporal_ratio > 0: loss covers both masked neurons (all time) and visible
    neurons at temporally-zeroed bins. Verifies the scatter_add_ path in apply()."""
    torch.manual_seed(31)
    B, N, L = 2, 8, 6
    spikes = torch.randn(B, N, L).abs()
    embeddings = torch.randn(B, N, 8)
    real_mask = torch.ones(B, N, dtype=torch.bool)
    cfg = MaskingConfig(neuron_ratio=0.25, temporal_ratio=0.5, loss_on_all_neurons=False)

    masked = cfg.apply(spikes, embeddings, real_mask)

    # No NaN/Inf in loss_mask
    assert torch.isfinite(masked.loss_mask).all()
    # Loss mask values are exactly 0 or 1 (no double-count from scatter_add_)
    unique_vals = masked.loss_mask.unique()
    assert ((unique_vals == 0) | (unique_vals == 1)).all(), (
        f"loss_mask should be 0/1 with no accumulation, got values {unique_vals.tolist()}"
    )
    # At least one bin has loss (sanity)
    assert masked.loss_mask.sum() > 0


def test_combined_masking_handles_variable_neuron_counts():
    """Variable n_real per sample: scatter_add_ in apply() must not pollute neuron 0."""
    torch.manual_seed(41)
    B, N, L = 3, 10, 6
    spikes = torch.randn(B, N, L).abs()
    embeddings = torch.randn(B, N, 8)
    real_mask = torch.zeros(B, N, dtype=torch.bool)
    real_mask[0, :10] = True  # full
    real_mask[1, :6] = True  # half
    real_mask[2, :3] = True  # quarter (still > 1 so masking can run)
    cfg = MaskingConfig(neuron_ratio=0.5, temporal_ratio=0.3, loss_on_all_neurons=False)

    masked = cfg.apply(spikes, embeddings, real_mask)

    assert torch.isfinite(masked.loss_mask).all()
    # Padded neurons (mask=False) must have loss_mask == 0 everywhere.
    padded = ~real_mask
    assert (masked.loss_mask * padded.float().unsqueeze(-1)).sum() == 0
    # Loss mask is exactly 0/1 even with variable n_real.
    unique_vals = masked.loss_mask.unique()
    assert ((unique_vals == 0) | (unique_vals == 1)).all()


def test_masking_rejects_all_padded_sample():
    """A batch element with n_real==0 must surface as a ValueError, not silent corruption."""
    torch.manual_seed(51)
    B, N, L = 2, 6, 5
    spikes = torch.randn(B, N, L).abs()
    embeddings = torch.randn(B, N, 8)
    real_mask = torch.ones(B, N, dtype=torch.bool)
    real_mask[0] = False  # all-padded sample 0
    cfg = MaskingConfig(neuron_ratio=0.5, temporal_ratio=0.0)

    with pytest.raises(ValueError, match="at least 1 real neuron"):
        cfg.apply(spikes, embeddings, real_mask)


# ---------------------------------------------------------------------------
# Constructor / spec validation
# ---------------------------------------------------------------------------


def test_perceiver_spec_rejects_empty_embeddings_path():
    """embeddings_path is required; instantiating without it raises at validation."""
    with pytest.raises(ValidationError):
        PerceiverTokenizerSpec(d_embed=8)  # missing embeddings_path


def test_perceiver_spec_migrates_neuron_mask_ratio(tmp_path: Path):
    """A spec dict with ``neuron_mask_ratio`` loads as ``masking`` with no temporal masking."""
    spec = PerceiverTokenizerSpec.model_validate(
        {"d_embed": 8, "embeddings_path": str(tmp_path / "e.pt"), "neuron_mask_ratio": 0.3}
    )
    assert spec.masking.neuron_ratio == 0.3
    assert spec.masking.temporal_ratio == 0.0


def test_perceiver_constructs_with_real_init(embeddings_file: Path):
    """Smoke test: the real __init__ runs end-to-end on a synthetic embeddings file."""
    tok = _make_tokenizer(embeddings_file)
    assert tok.spec.d_embed == D_EMBED
    assert tok._all_embs.shape == (NUM_SESSIONS * N_UNITS_PER_SESSION, D_EMBED)
    assert "session_0" in tok._session_offset


def test_perceiver_decode_raises_without_embeddings(embeddings_file: Path):
    """decode(tokens) and reconstruct(tokens) signal the embedding-required contract."""
    tok = _make_tokenizer(embeddings_file)
    z = torch.zeros(2, 4, 4, 8)
    with pytest.raises(NotImplementedError, match="reconstruct_from_input"):
        tok.decode(z)
    with pytest.raises(NotImplementedError, match="reconstruct_from_input"):
        tok.reconstruct(z)


# ---------------------------------------------------------------------------
# Embedding lookup: unit-id range guard
# ---------------------------------------------------------------------------


def test_get_embeddings_rejects_real_unit_ids_beyond_the_session(embeddings_file: Path):
    """A real unit whose id is >= the session's embedding count must raise.

    Clamping it would map the unit onto the session's last
    embedding row, so a caller passing dataset-global instead of local unit ids
    trains on the wrong embeddings without any error.
    """
    tok = _make_tokenizer(embeddings_file)
    batch = _make_batched_input(batch_size=2, n_units=N_UNITS_PER_SESSION + 4)
    with pytest.raises(ValueError, match="session_0"):
        tok._get_embeddings(batch)


def test_get_embeddings_rejects_negative_real_unit_ids(embeddings_file: Path):
    tok = _make_tokenizer(embeddings_file)
    batch = _make_batched_input(batch_size=2, n_units=4)
    batch = batch._replace(unit_ids=batch.unit_ids - 1)
    with pytest.raises(ValueError, match="unit id"):
        tok._get_embeddings(batch)


def test_get_embeddings_allows_out_of_range_ids_on_padded_slots(embeddings_file: Path):
    """Padded slots (unit_ids_pad_mask == 0) are masked out, so any id is fine there."""
    tok = _make_tokenizer(embeddings_file)
    n_real = N_UNITS_PER_SESSION
    batch = _make_batched_input(
        batch_size=2, n_units=n_real + 4, real_neurons_per_sample=[n_real, n_real - 3]
    )
    e, mask = tok._get_embeddings(batch)
    assert e.shape == (2, n_real + 4, D_EMBED)
    assert mask.sum(dim=1).tolist() == [n_real, n_real - 3]


# ---------------------------------------------------------------------------
# Regularizer tests
# ---------------------------------------------------------------------------


def test_curvature_td_regularizer_does_not_penalize_linear_ramp(embeddings_file: Path):
    tok = _make_tokenizer(
        embeddings_file,
        latent_beta=0.0,
        latent_td_beta=1.0,
        latent_td_k=1,
        latent_td_order=2,
    )
    ramp = torch.arange(4, dtype=torch.float32).view(1, 1, 1, 4)
    jitter = torch.tensor([0.0, 1.0, 0.0, 1.0]).view(1, 1, 1, 4)

    assert torch.allclose(tok._compute_latent_loss(ramp), torch.tensor(0.0))
    assert torch.allclose(tok._compute_latent_loss(jitter), torch.tensor(4.0))


def test_td_regularizer_accumulates_across_k_hops(embeddings_file: Path):
    """latent_td_k > 1 should accumulate contributions across multiple lags."""
    tok = _make_tokenizer(
        embeddings_file,
        latent_beta=0.0,
        latent_td_beta=1.0,
        latent_td_k=3,
        latent_td_order=1,
    )
    # Constant signal: zero contribution at every hop.
    constant = torch.full((1, 1, 1, 10), 2.5)
    assert torch.allclose(tok._compute_latent_loss(constant), torch.tensor(0.0))
    # Non-constant signal: positive total > positive per-hop.
    nonconst = torch.linspace(0.0, 1.0, 10).view(1, 1, 1, 10)
    loss_k3 = tok._compute_latent_loss(nonconst).item()
    tok_k1 = _make_tokenizer(
        embeddings_file,
        latent_beta=0.0,
        latent_td_beta=1.0,
        latent_td_k=1,
        latent_td_order=1,
    )
    loss_k1 = tok_k1._compute_latent_loss(nonconst).item()
    assert loss_k3 > loss_k1 > 0


# ---------------------------------------------------------------------------
# End-to-end forward / backward
# ---------------------------------------------------------------------------


def _make_session_aware_batch(embeddings_file: Path) -> BinnedSpikesInput:
    """Batch whose session_ids exist in the synthetic embeddings file."""
    batch = _make_batched_input(batch_size=2, n_units=N_UNITS_PER_SESSION, n_bins=8)
    return batch


def test_full_loss_backprop_with_neuron_only_encoder_masking_and_sigreg(
    embeddings_file: Path,
):
    torch.manual_seed(2)
    batch = _make_session_aware_batch(embeddings_file)
    tok = _make_tokenizer(
        embeddings_file,
        latent_beta=1e-6,
        latent_td_beta=1e-4,
        latent_td_k=1,
        latent_td_order=2,
        use_sigreg_loss=True,
        latent_sigreg_num_slices=8,
        latent_sigreg_n_points=5,
        latent_sigreg_center=False,
        latent_sigreg_sync_slices=True,
        latent_sigreg_seed=42,
        masking={
            "neuron_ratio": 0.5,
            "temporal_ratio": 0.0,
            "loss_on_all_neurons": True,
        },
    )
    tok.train()

    out = tok(batch)

    assert out.precomputed_recon_loss is not None
    assert out.precomputed_recon_loss.requires_grad
    assert out.aux_loss is not None and out.aux_loss.requires_grad
    assert out.reconstruction.shape == (2, 8, N_UNITS_PER_SESSION)
    (out.precomputed_recon_loss + out.aux_loss).backward()

    grads = [p.grad for p in tok.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum().item() for g in grads) > 0
    # Latent queries should receive gradient since SIGReg acts on z.
    assert tok.count_wrapper.ae_net.latent_queries.grad is not None
    assert tok.count_wrapper.ae_net.latent_queries.grad.abs().sum() > 0


def test_full_loss_backprop_with_combined_neuron_and_temporal_masking(
    embeddings_file: Path,
):
    """End-to-end check that combined neuron + temporal masking trains cleanly."""
    torch.manual_seed(7)
    batch = _make_session_aware_batch(embeddings_file)
    tok = _make_tokenizer(
        embeddings_file,
        latent_beta=0.0,
        latent_td_beta=0.0,
        masking={
            "neuron_ratio": 0.3,
            "temporal_ratio": 0.5,
            "loss_on_all_neurons": False,
        },
    )
    tok.train()

    out = tok(batch)

    assert out.precomputed_recon_loss is not None and out.precomputed_recon_loss.requires_grad
    out.precomputed_recon_loss.backward()
    grads = [p.grad for p in tok.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


def test_compute_compression_runs_on_the_dummy_validation_session(embeddings_file: Path):
    """Compression's shape-only dummy batch uses the ``"__validation__"``
    session, which no embeddings table holds; ldns_perceiver must still record compression."""
    from ntt.evaluation.compression import compute_compression

    tok = _make_tokenizer(embeddings_file)
    tok.eval()

    result = compute_compression(tok.spec, tok)

    assert np.isfinite(result.compression_rate) and result.compression_rate > 0


def test_reconstruct_from_input_round_trip(embeddings_file: Path):
    """reconstruct_from_input(x) must produce shape-correct rates and optional latents."""
    batch = _make_session_aware_batch(embeddings_file)
    tok = _make_tokenizer(embeddings_file)
    tok.eval()

    result = tok.reconstruct_from_input(batch)
    assert result.reconstruction.shape == (2, 8, N_UNITS_PER_SESSION)

    # return_latents=True should give a dict with latents and timestamps.
    result_dict = tok.reconstruct_from_input(
        batch, return_latents=True, return_latent_timestamps=True
    )
    assert isinstance(result_dict, dict)
    assert "reconstruction" in result_dict
    assert "latents" in result_dict
    assert "token_timestamps" in result_dict
    # latents reshaped to (B, T, M, D)
    assert result_dict["latents"].shape == (2, 8, 4, 4)

    # return_latent_timestamps=False omits the timestamps key.
    result_dict_no_ts = tok.reconstruct_from_input(
        batch, return_latents=True, return_latent_timestamps=False
    )
    assert "token_timestamps" not in result_dict_no_ts


def test_encode_with_timestamps_returns_canonical_layout(embeddings_file: Path):
    """encode_with_timestamps emits ``[B, T'=L, U=M, D=D_lat]``, not raw ``[B, M, D_lat, L]``.

    Downstream consumers assume the last axis is the invariant latent-embedding
    dim (so an input_projection sized off a dummy probe still matches at run
    time), and axis 1 is time (so per-token timestamps align).
    """
    batch = _make_session_aware_batch(embeddings_file)  # n_bins=8, batch=2
    tok = _make_tokenizer(embeddings_file)  # hidden_dim=16, latent_tokens=4, latent_dim=4
    tok.eval()

    with torch.no_grad():
        latents, metadata = tok.encode_with_timestamps(batch)

    # Canonical (B, T'=L, U=M, D=D_lat). The last axis must be D_lat (4),
    # not L (8), so downstream code that reads latents.shape[-1] as the token
    # embedding dim sees the right size.
    assert latents.shape == (2, 8, 4, 4)
    # Per-time-bin timestamps, one per T' entry.
    assert metadata["token_timestamps"].shape == (2, 8)
    # bin_size_ms=10.0 → 10 ms centers starting from absolute_start=0.
    expected_centers = torch.tensor([0.005 + 0.010 * i for i in range(8)])
    torch.testing.assert_close(metadata["token_timestamps"][0], expected_centers.double())


def test_encode_with_timestamps_no_bottleneck_uses_hidden_dim(embeddings_file: Path):
    """Without a latent bottleneck (``latent_dim=None``), the last axis is ``hidden_dim``.

    Guards the case that motivated this fix: earlier code returned
    ``latents.shape[-1] = L`` from a raw ``[B, M, hidden_dim, L]`` layout,
    which varies with the input window length. With the canonical layout the
    last axis reports the constant ``hidden_dim`` even when the bottleneck is
    disabled — matching how LDNS-Perceiver checkpoints operate.
    """
    batch = _make_session_aware_batch(embeddings_file)  # n_bins=8
    # ``_make_tokenizer`` hard-codes ``latent_dim=4`` so build the spec directly
    # to exercise the ``latent_dim is None`` code path.
    spec = PerceiverTokenizerSpec(
        hidden_dim=16,
        latent_tokens=4,
        latent_dim=None,
        d_embed=D_EMBED,
        num_heads=4,
        mlp_ratio=1,
        num_encoder_N_blocks=1,
        num_encoder_M_blocks=1,
        num_decoder_blocks=1,
        use_s4=False,
        grad_checkpointing=False,
        use_layernorm=True,
        window_length_seconds=0.08,
        bin_size_ms=10.0,
        embeddings_path=str(embeddings_file),
    )
    tok = PerceiverTokenizer(spec)
    tok.eval()

    with torch.no_grad():
        latents, _ = tok.encode_with_timestamps(batch)

    # hidden_dim=16, latent_tokens=4, L=8.
    assert latents.shape == (2, 8, 4, 16)
