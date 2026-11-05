"""Causal-decoder + variable-window support for the Perceiver-S4 tokenizer.

Covers:
  - DWConvTimeMixer(causal=True): output[..., t] must not depend on input[..., >t].
  - PerceiverNeuralAE(decoder_causal=True): decode output at time t must not
    depend on the latent z at any future time t' > t (true decoder causality).
  - PerceiverTokenizer constructs with a Dist window_length_seconds (variable
    windows) instead of crashing on spec.window_length_bins.
"""

from pathlib import Path

import numpy as np
import torch

from ntt.contrib.tokenizer.perceiver.blocks import DWConvTimeMixer, PerceiverNeuralAE
from ntt.contrib.tokenizer.perceiver.perceiver_tokenizer import (
    PerceiverTokenizer,
    PerceiverTokenizerSpec,
)
from ntt.core.components.dist import UniformDist


def test_causal_dwconv_does_not_leak_future():
    """With causal=True, output[..., t] is independent of input[..., t'] for t' > t."""
    torch.manual_seed(0)
    C, L = 8, 16
    mixer = DWConvTimeMixer(C, kernel_size=3, bidirectional=False, causal=True).eval()

    x = torch.randn(2, C, L)
    y_ref, _ = mixer(x)

    # Perturb the LAST timestep only; every earlier output must be unchanged.
    t_future = L - 1
    x_pert = x.clone()
    x_pert[..., t_future] += 10.0
    y_pert, _ = mixer(x_pert)

    assert torch.allclose(y_ref[..., :t_future], y_pert[..., :t_future], atol=1e-6), (
        "Causal DWConv leaked information from a future timestep into the past."
    )


def test_bidirectional_dwconv_does_leak_future():
    """Control: the default (bidirectional, centered) mixer DOES use the future."""
    torch.manual_seed(0)
    C, L = 8, 16
    mixer = DWConvTimeMixer(C, kernel_size=3, bidirectional=True).eval()

    x = torch.randn(2, C, L)
    y_ref, _ = mixer(x)

    t_future = L - 1
    x_pert = x.clone()
    x_pert[..., t_future] += 10.0
    y_pert, _ = mixer(x_pert)

    # The bin just before t_future must change (centered/backward kernel sees t+1).
    assert not torch.allclose(y_ref[..., t_future - 1], y_pert[..., t_future - 1], atol=1e-6), (
        "Non-causal mixer unexpectedly ignored the future timestep."
    )


def test_decoder_causal_output_independent_of_future_latent():
    """decoder_causal=True ⇒ decode output at t is independent of z at t' > t.

    Built with use_layernorm=True so normalization is per-timestep (InstanceNorm
    over L would itself be a non-causal, global-over-time statistic).
    """
    torch.manual_seed(0)
    B, N, M, C, L = 1, 5, 4, 16, 12
    d_embed = 8
    ae = PerceiverNeuralAE(
        C=C,
        d_embed=d_embed,
        L=L,
        M=M,
        num_heads=4,
        num_encoder_N_blocks=1,
        num_encoder_M_blocks=1,
        num_decoder_blocks=2,
        use_s4=False,  # decoder already forces DWConv; keep encoder cheap too
        grad_checkpointing=False,
        use_layernorm=True,
        decoder_causal=True,
    ).eval()

    z = torch.randn(B, M, C, L)
    e = torch.randn(B, N, d_embed)

    with torch.no_grad():
        out_ref = ae.decode(z, e)
        t_future = L - 1
        z_pert = z.clone()
        z_pert[..., t_future] += 10.0
        out_pert = ae.decode(z_pert, e)

    assert torch.allclose(out_ref[..., :t_future], out_pert[..., :t_future], atol=1e-5), (
        "Causal decoder leaked a future latent timestep into earlier outputs."
    )


def test_s4_kernel_presizes_to_lmax_not_first_window():
    """With L set, the encoder S4 kernel pre-initializes to l_max=L on the first
    forward, even when that forward is a SHORTER window — so later max-length
    windows never re-trigger kernel growth (_setup_C)."""
    torch.manual_seed(0)
    B, N, M, C = 1, 4, 4, 8
    d_embed, L_max, L_short = 8, 16, 4
    ae = PerceiverNeuralAE(
        C=C,
        d_embed=d_embed,
        L=L_max,
        M=M,
        num_heads=4,
        num_encoder_N_blocks=1,
        num_encoder_M_blocks=1,
        num_decoder_blocks=1,
        use_s4=True,
        grad_checkpointing=False,
        use_layernorm=True,
    ).eval()

    x = torch.randn(B, N, L_short)
    e = torch.randn(B, N, d_embed)
    with torch.no_grad():
        ae.encode(x, e)

    # Reach into the first encoder N-block's S4 kernel cache.
    kernel = ae.encoder_N_blocks[0].time_mixer.conv.kernel
    assert int(kernel.l_kernel.item()) >= L_max, (
        f"S4 kernel cached length {int(kernel.l_kernel.item())} < l_max {L_max}; "
        f"it sized to the first (short) window and will re-init on longer ones."
    )


def _write_embeddings_file(tmp_path: Path, *, d_embed: int = 8) -> Path:
    gen = torch.Generator().manual_seed(7)
    n_sessions, n_units = 2, 16
    embs = torch.randn(n_sessions * n_units, d_embed, generator=gen)
    session_ids, unit_ids = [], []
    for s in range(n_sessions):
        for u in range(n_units):
            session_ids.append(f"session_{s}")
            unit_ids.append(f"session_{s}/{u}")
    path = tmp_path / "embs.pt"
    torch.save(
        {
            "embs": embs,
            "unit_ids": np.asarray(unit_ids),
            "session_ids": np.asarray(session_ids),
        },
        path,
    )
    return path


def test_tokenizer_constructs_with_variable_window(tmp_path):
    """A Dist window_length_seconds must not crash construction (window_length_bins
    raises on Dist; init should resolve via effective_window_length_seconds)."""
    emb_path = _write_embeddings_file(tmp_path)
    spec = PerceiverTokenizerSpec(
        window_length_seconds=UniformDist(low=0.1, high=4.0, return_int=False),
        bin_size_ms=10.0,
        hidden_dim=16,
        latent_tokens=4,
        latent_dim=4,
        d_embed=8,
        num_heads=4,
        num_encoder_N_blocks=1,
        num_encoder_M_blocks=1,
        num_decoder_blocks=1,
        use_s4=False,
        grad_checkpointing=False,
        use_layernorm=True,
        decoder_causal=True,
        embeddings_path=str(emb_path),
    )
    tok = PerceiverTokenizer(spec)
    assert tok is not None
