"""Discrete bits per token on real tokenizer modules (tiny CPU configs, no checkpoints).

``compute_compression`` charges each ``[t, u]`` latent position
``codes_per_token * log2(vocab)`` bits, where ``vocab`` is the per-code alphabet
(``spec.get_vocab_size()``) and ``codes_per_token`` is the number of such codes the
quantizer emits per position (``spec.get_codes_per_token()``). These tests pin that count
against the quantizers that actually ship: wanneural FSQ / BSQ / VQPT, cosmos FSQ2d /
VQPT2d and CaLM VQ.
"""

from __future__ import annotations

import math

import pytest
import torch

from ntt.contrib.tokenizer.calm.calm_tokenizer import CalmTokenizer, CalmTokenizerSpec
from ntt.contrib.tokenizer.cosmos.cosmos_2d import CosmosTokenizer2D, CosmosTokenizer2DSpec
from ntt.contrib.tokenizer.wanneural.bottleneck import VQPTBottleneck
from ntt.contrib.tokenizer.wanneural.wanneural import WanNeuralTokenizer, WanNeuralTokenizerSpec
from ntt.core.model.tokenizer.base import BaseTokenizer, TokenizerSpec
from ntt.core.model.tokenizer.validation import make_dummy_input
from ntt.evaluation.compression import compute_compression


def _wanneural_spec(**overrides: object) -> WanNeuralTokenizerSpec:
    kwargs: dict[str, object] = {
        "n_neurons": 8,
        "window_length_seconds": 0.2,
        "bin_size_ms": 1.0,
        "dim": 16,
        "z_dim": 8,
        "num_levels": 3,
        "dim_mult": (1, 2, 4),
        "num_res_blocks": 1,
        "attn_levels": (),
        "num_attn_heads": 1,
        "temporal_patch_size": 1,
        "temporal_downsample": (True, True),
        "token_bin_size_ms": 4.0,
    }
    kwargs.update(overrides)
    return WanNeuralTokenizerSpec(**kwargs)  # type: ignore[arg-type]


def _cosmos_spec(**overrides: object) -> CosmosTokenizer2DSpec:
    kwargs: dict[str, object] = {
        "n_neurons": 32,
        "window_length_seconds": 1.0,
        "bin_size_ms": 1.0,
        "channels": 16,
        "channels_mult": [1, 2, 2],
        "num_res_blocks": 1,
        "z_channels": 16,
        "neuron_compression": 16,
        "temporal_compression": 16,
        "patch_h": 4,
        "patch_w": 4,
    }
    kwargs.update(overrides)
    return CosmosTokenizer2DSpec(**kwargs)  # type: ignore[arg-type]


def _calm_spec(**overrides: object) -> CalmTokenizerSpec:
    kwargs: dict[str, object] = {
        "n_neurons": 8,
        "window_length_seconds": 0.2,
        "bin_size_ms": 10.0,
        "patch_length": 5,
        "embed_dim": 32,
        "num_encoder_layers": 1,
        "num_decoder_layers": 1,
        "num_heads": 4,
        "codebook_size": 2048,
        "bottleneck_type": "vq",
        "ar_weight": 0.0,
    }
    kwargs.update(overrides)
    return CalmTokenizerSpec(**kwargs)  # type: ignore[arg-type]


_RESIDUAL_FSQ = {"num_quantizers": 4, "levels": [8, 5, 5, 5]}
_RESIDUAL_LFQ = {"num_quantizers": 4, "codebook_size": 1024}

# (id, spec factory, tokenizer class, expected bits per [t, u] position)
REAL_CASES = [
    (
        "wanneural-bsq-z8",
        lambda: _wanneural_spec(bottleneck_type="bsq"),
        WanNeuralTokenizer,
        8.0,
    ),
    (
        "wanneural-bsq-z8-codebook4",
        lambda: _wanneural_spec(bottleneck_type="bsq", bsq_codebook_dim=4),
        WanNeuralTokenizer,
        4.0,
    ),
    (
        "wanneural-fsq-75555",
        lambda: _wanneural_spec(bottleneck_type="fsq", q_levels=(7, 5, 5, 5, 5)),
        WanNeuralTokenizer,
        math.log2(4375),
    ),
    (
        "wanneural-vqpt-residual_fsq-q4",
        lambda: _wanneural_spec(
            bottleneck_type="vqpt", vqpt_type="residual_fsq", vqpt_kwargs=_RESIDUAL_FSQ
        ),
        WanNeuralTokenizer,
        4 * math.log2(1000),
    ),
    (
        "wanneural-vqpt-residual_lfq-q4",
        lambda: _wanneural_spec(
            bottleneck_type="vqpt", vqpt_type="residual_lfq", vqpt_kwargs=_RESIDUAL_LFQ
        ),
        WanNeuralTokenizer,
        40.0,
    ),
    (
        "cosmos-fsq-32x32",
        lambda: _cosmos_spec(bottleneck_type="fsq", fsq_levels=[32, 32]),
        CosmosTokenizer2D,
        10.0,
    ),
    (
        "cosmos-vqpt-residual_fsq-q2",
        lambda: _cosmos_spec(
            bottleneck_type="vqpt",
            vqpt_type="residual_fsq",
            vqpt_kwargs={"num_quantizers": 2, "levels": [8, 5, 5, 5]},
        ),
        CosmosTokenizer2D,
        2 * math.log2(1000),
    ),
    ("calm-vq-2048", _calm_spec, CalmTokenizer, 11.0),
]


@pytest.mark.parametrize(
    ("make_spec", "tok_cls", "expected_bits"),
    [pytest.param(*case[1:], id=case[0]) for case in REAL_CASES],
)
def test_bits_per_token_on_real_tokenizers(make_spec, tok_cls, expected_bits):
    spec = make_spec()
    tok = tok_cls(spec).eval()

    result = compute_compression(spec, tok)

    assert result.bits_per_token == pytest.approx(expected_bits)
    assert result.codes_per_token * result.bits_per_dim == pytest.approx(expected_bits)
    assert result.output_bits == pytest.approx(result.t_tokens * result.u_units * expected_bits)


def _random_binned_input(spec: TokenizerSpec, batch_size: int = 4):
    torch.manual_seed(0)
    x = make_dummy_input(spec, batch_size=batch_size)
    rand = (torch.rand(x.spike_matrix.shape) < 0.3).to(x.spike_matrix.dtype)
    return x._replace(spike_matrix=rand)


@pytest.mark.parametrize(
    "make_spec",
    [
        pytest.param(
            lambda: _wanneural_spec(bottleneck_type="fsq", q_levels=(7, 5, 5, 5, 5)), id="fsq"
        ),
        pytest.param(lambda: _wanneural_spec(bottleneck_type="bsq"), id="bsq"),
        pytest.param(
            lambda: _wanneural_spec(bottleneck_type="bsq", bsq_codebook_dim=4), id="bsq-cb4"
        ),
        pytest.param(
            lambda: _wanneural_spec(
                bottleneck_type="vqpt", vqpt_type="residual_fsq", vqpt_kwargs=_RESIDUAL_FSQ
            ),
            id="residual_fsq",
        ),
        pytest.param(
            lambda: _wanneural_spec(
                bottleneck_type="vqpt", vqpt_type="residual_lfq", vqpt_kwargs=_RESIDUAL_LFQ
            ),
            id="residual_lfq",
        ),
    ],
)
def test_encode_vectors_carry_exactly_one_index_tuple(make_spec):
    """Ties ``bits_per_token`` to what the tokenizer really emits.

    1. The D floats ``encode()`` returns per position are a function of ONE index tuple
       (distinct rounded vectors == distinct index tuples), so a position holds no more
       information than its indices.
    2. ``bits_per_token`` equals the measured index count per position (from
       ``forward().tokens``) times ``log2(vocab)``. This catches over-counting (e.g. the
       old per-dim rule, ``D * log2(vocab)``) and a spec whose code count is wrong.

    A future tokenizer whose D values are independent symbols fails (1).
    """
    spec = make_spec()
    tok: BaseTokenizer = WanNeuralTokenizer(spec).eval()
    x = _random_binned_input(spec)

    with torch.no_grad():
        z = tok.encode(x)
        tokens = tok.forward(x).tokens
    assert tokens is not None
    B, T, U, D = z.shape
    codes = tokens.numel() // (B * T * U)
    assert codes == spec.get_codes_per_token()

    # Round away straight-through float noise, then count distinct code vectors.
    vectors = torch.round(z.reshape(-1, D).double() * 1e4)
    n_vectors = torch.unique(vectors, dim=0).shape[0]
    n_tuples = torch.unique(tokens.reshape(-1, codes), dim=0).shape[0]
    bits_per_token = compute_compression(spec, tok).bits_per_token

    assert n_tuples > 1, "degenerate probe: every position got the same code"
    assert n_vectors == n_tuples
    assert bits_per_token == pytest.approx(codes * math.log2(spec.get_vocab_size()))


# ── Spec-level codes per token vs. what the lucidrains quantizer really emits ──

VQPT_CASES = [
    ("vq", {"codebook_size": 64}, 1),
    ("vq", {"codebook_size": 64, "heads": 2, "separate_codebook_per_head": True}, 2),
    ("fsq", {"levels": [8, 5, 5, 5]}, 1),
    ("fsq", {"levels": [8, 5, 5, 5], "num_codebooks": 2}, 2),
    ("lfq", {"codebook_size": 256}, 1),
    ("lfq", {"codebook_size": 16, "num_codebooks": 2}, 2),
    ("residual_vq", {"codebook_size": 64, "num_quantizers": 3}, 3),
    ("residual_fsq", {"levels": [8, 5, 5, 5], "num_quantizers": 4}, 4),
    ("residual_lfq", {"codebook_size": 1024, "num_quantizers": 4}, 4),
    ("grouped_residual_vq", {"codebook_size": 64, "num_quantizers": 2, "groups": 2}, 4),
    ("grouped_residual_fsq", {"levels": [8, 5, 5, 5], "num_quantizers": 2, "groups": 2}, 4),
    ("grouped_residual_lfq", {"codebook_size": 16, "num_quantizers": 2, "groups": 2}, 4),
    ("sim_vq", {"codebook_size": 64}, 1),
    ("residual_sim_vq", {"codebook_size": 64, "num_quantizers": 3}, 3),
    ("latent_quantize", {"levels": [8, 5, 5, 5]}, 1),
]


@pytest.mark.parametrize(
    ("vqpt_type", "vqpt_kwargs", "expected"),
    [pytest.param(t, kw, n, id=f"{t}-{n}") for t, kw, n in VQPT_CASES],
)
def test_wanneural_vqpt_codes_per_token_matches_the_quantizer_indices(
    vqpt_type, vqpt_kwargs, expected
):
    spec = _wanneural_spec(bottleneck_type="vqpt", vqpt_type=vqpt_type, vqpt_kwargs=vqpt_kwargs)
    bottleneck = VQPTBottleneck(z_dim=spec.z_dim, vq_type=vqpt_type, vq_kwargs=vqpt_kwargs).eval()
    B, T, N = 2, 3, 4

    with torch.no_grad():
        indices = bottleneck.get_indices(torch.randn(B, spec.z_dim, T, N))

    assert spec.get_codes_per_token() == expected
    assert indices.numel() == B * T * N * expected


@pytest.mark.parametrize(
    ("vqpt_type", "vqpt_kwargs", "expected"),
    [
        ("rvq", {"codebook_size": 64, "num_quantizers": 3}, 3),
        ("residual_fsq", {"levels": [8, 5, 5, 5], "num_quantizers": 2}, 2),
        ("fsq", {"levels": [8, 5, 5, 5]}, 1),
    ],
)
def test_cosmos_vqpt_codes_per_token(vqpt_type, vqpt_kwargs, expected):
    spec = _cosmos_spec(bottleneck_type="vqpt", vqpt_type=vqpt_type, vqpt_kwargs=vqpt_kwargs)

    assert spec.get_codes_per_token() == expected


def test_latent_quantize_counts_one_code_per_codebook():
    """lucidrains ``LatentQuantize(num_codebooks=n)`` emits one index per codebook.

    Spec-level only: lucidrains v1.30.1 cannot construct ``num_codebooks > 1`` (its
    implicit-codebook setup fails in ``indices_to_codes``), so there is no quantizer to
    compare the layout with, unlike the other ``VQPT_CASES``.
    """
    spec = _wanneural_spec(
        bottleneck_type="vqpt",
        vqpt_type="latent_quantize",
        vqpt_kwargs={"levels": [4, 4, 4, 4], "num_codebooks": 2},
    )

    assert spec.get_codes_per_token() == 2


@pytest.mark.parametrize(
    "make_spec",
    [
        lambda kw: _wanneural_spec(
            bottleneck_type="vqpt", vqpt_type="residual_vq", vqpt_kwargs=kw
        ),
        lambda kw: _cosmos_spec(bottleneck_type="vqpt", vqpt_type="rvq", vqpt_kwargs=kw),
    ],
    ids=["wanneural", "cosmos"],
)
def test_per_stage_codebook_sizes_have_no_single_vocab(make_spec):
    """``ResidualVQ(codebook_size=(64, 256))`` has a different alphabet per stage, so
    there is no one ``vocab_size``: raise NotImplementedError (compression gives NaN and
    a warning) instead of a ``TypeError`` from ``int(tuple)``."""
    spec = make_spec({"codebook_size": [64, 256], "num_quantizers": 2})

    with pytest.raises(NotImplementedError, match="per-stage"):
        spec.get_vocab_size()


@pytest.mark.parametrize(
    ("vqpt_type", "vqpt_kwargs"),
    [
        ("residual_fsq", {"levels": [8, 5, 5, 5]}),  # num_quantizers missing
        ("residual_fsq", {"levels": [8, 5, 5, 5], "num_quantizers": 2, "num_codebooks": 2}),
        ("not_a_quantizer", {"codebook_size": 64}),
    ],
)
def test_vqpt_codes_per_token_refuses_to_guess(vqpt_type, vqpt_kwargs):
    """Configs whose code count is unknown must raise, never default to some count."""
    spec = _wanneural_spec(bottleneck_type="vqpt", vqpt_type=vqpt_type, vqpt_kwargs=vqpt_kwargs)

    with pytest.raises(NotImplementedError):
        spec.get_codes_per_token()


@pytest.mark.parametrize("bottleneck_type", ["vq", "fsq", "bsq"])
def test_calm_emits_one_code_per_token(bottleneck_type):
    assert _calm_spec(bottleneck_type=bottleneck_type).get_codes_per_token() == 1


def test_continuous_spec_has_no_codes_per_token():
    spec = _wanneural_spec(bottleneck_type="vae")

    with pytest.raises(NotImplementedError):
        spec.get_codes_per_token()
