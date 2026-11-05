"""Tests for the raster visualisation path in NTTLitModule.

Continuous-bottleneck tokenizers (vae / mmd_vae / diffusion_prior, and the PCA
baseline) set `tokens = None`. `_visualize_sample` must not plot
`output.reconstruction` verbatim for them: it holds post-sigmoid probabilities in
(0, 1), and `plot_population_raster` treats any value > 0 as a spike, so every bin
would render as a spike and the raster would be a solid block. The plotted
reconstruction must be binarised through the tokenizer's spike output instead.
"""

import torch

from ntt.contrib.loss.bce import BCELossSpec
from ntt.contrib.tokenizer.wanneural.wanneural import (
    WanNeuralTokenizer,
    WanNeuralTokenizerSpec,
)
from ntt.core.components.types import BinnedSpikesInput
from ntt.core.model.litmodule import NTTLitModule

N_UNITS = 6
T_BINS = 40


def _spec(**overrides) -> WanNeuralTokenizerSpec:
    """A minimal continuous-bottleneck spec: VAE bottleneck, sigmoid binary-spike head."""
    cfg = {
        "n_neurons": N_UNITS,
        "window_length_seconds": T_BINS / 1000.0,
        "bin_size_ms": 1.0,
        "dim": 8,
        "z_dim": 4,
        "num_levels": 1,
        "dim_mult": [1],
        "num_res_blocks": 1,
        "attn_levels": [],
        "num_attn_heads": 1,
        "mid_attn": False,
        "dec_dim": 8,
        "dec_dim_mult": [1],
        "dec_num_res_blocks": 1,
        "temporal_patch_size": 10,
        "temporal_downsample": [],
        "k_t": 1,
        "k_t_init": 1,
        "s_t_init": 1,
        "k_t_down": 2,
        "s_t_down": 2,
        "causal": True,
        "bottleneck_type": "vae",
        "beta": 0.0,
        "output_type": "binary_spikes",
        "output_activation": "sigmoid",
        "reconstruct_sample": "round",
        "input_type": "binary_spikes",
        # Required whenever compress_t is true: temporal_patch_size=10 at 1 ms
        # bins means each token spans 10 ms, and it must divide the window.
        "token_bin_size_ms": 10.0,
        "stage_shortcut": True,
        "neuron_kernel_size": 1,
    }
    cfg.update(overrides)
    return WanNeuralTokenizerSpec(**cfg)


def _batch() -> BinnedSpikesInput:
    torch.manual_seed(0)
    spikes = (torch.rand(1, T_BINS, N_UNITS) < 0.2).to(torch.int32)
    return BinnedSpikesInput(
        spike_matrix=spikes,
        pad_and_gap_mask=torch.ones_like(spikes),
        # Post-collation this is a per-sample Tensor, and _visualize_sample
        # indexes it (`batch.sampling_rate[0]`), so a bare float will not do.
        sampling_rate=torch.full((1,), 1000.0),
        unit_ids=torch.arange(N_UNITS, dtype=torch.int32).unsqueeze(0),
        unit_ids_pad_mask=torch.ones((1, N_UNITS), dtype=torch.int32),
        duration_sec=torch.full((1,), T_BINS / 1000.0),
        session_id=("__test__",),
        subject_id=("s",),
        species=("sp",),
        absolute_start=torch.zeros(1, dtype=torch.float64),
        absolute_end=torch.full((1,), T_BINS / 1000.0, dtype=torch.float64),
        behavioral_data=None,
    )


def test_vae_bottleneck_produces_no_tokens():
    """Precondition: a VAE bottleneck yields no tokens and a continuous reconstruction."""
    tok = WanNeuralTokenizer(_spec()).eval()
    with torch.no_grad():
        out = tok(_batch())
    assert out.tokens is None, "VAE bottleneck is expected to yield tokens=None"
    assert out.reconstruction is not None
    # The raw forward reconstruction is continuous and strictly positive under
    # sigmoid — plotting it directly is what produced the solid-block raster.
    assert out.reconstruction.dtype.is_floating_point
    assert (out.reconstruction > 0).all()


def test_visualised_reconstruction_is_binarised(monkeypatch):
    """The tensor handed to the raster plotter must be the model's spike output.

    With tokens=None the plotted reconstruction must still
    go through `reconstruct_sample`, not be raw sigmoid probabilities.
    """
    tok_spec = _spec()
    module = NTTLitModule(tokenizer_spec=tok_spec, loss_spec=BCELossSpec(distance_weight_window=1))
    module.configure_model()
    module.eval()

    batch = _batch()
    with torch.no_grad():
        output = module.tokenizer(batch)
    assert output.tokens is None

    captured = {}

    def _fake_raster(segment, bin_size_seconds):
        captured["recon"] = torch.as_tensor(segment.reconstructed)
        import matplotlib.figure

        return matplotlib.figure.Figure()

    monkeypatch.setattr("ntt.evaluation.plotting.plot_population_raster", _fake_raster)
    monkeypatch.setattr(NTTLitModule, "_log_figure", lambda self, fig, tag: None)

    module._visualize_sample(batch, output, "train")

    recon = captured["recon"].float()
    assert torch.equal(recon, recon.round()), (
        "plotted reconstruction is not binarised — the viz path is bypassing "
        f"reconstruct_sample; got {recon.unique()[:8].tolist()}"
    )
    assert recon.max() <= 1.0, "binary_spikes output must not exceed 1"
    # A solid block (every bin a spike) is the exact failure signature.
    assert (recon > 0).float().mean() < 1.0, "every bin is a spike — raster is a solid block"


def test_reconstruct_from_output_matches_reconstruct_cap():
    """`reconstruct_from_output` must apply the same max_count as `reconstruct`.

    Using only `reconstruct_max_count` would let a SPIKES tokenizer with an
    unbounded activation emit counts > 1 there while `reconstruct` caps at 1.
    """
    tok = WanNeuralTokenizer(_spec(output_activation="softplus")).eval()
    with torch.no_grad():
        out = tok(_batch())
        from_output = tok.reconstruct_from_output(out).reconstruction
    assert from_output.max() <= 1, (
        f"binary_spikes must clamp to 1, got max {int(from_output.max())}"
    )
