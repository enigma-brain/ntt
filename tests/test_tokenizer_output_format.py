"""Tests for tokenizer output format conformance with has_*_dim attributes.

For tokenizers supporting latent-space evaluation, the continuous latents
returned by the eval paths (encode_with_timestamps, reconstruct_from_input)
must:
1. Be floating-point 4D tensors [B, T, U, D] (batch, time, units, latent_dim).
2. Have T = 1 (singleton) if has_time_dim=False.
3. Have U = 1 (singleton) if has_unit_dim=False.

has_latent_dim describes the native tokens returned by encode(), so:
4. encode() tokens have D = 1 (singleton) if and only if has_latent_dim=False.

D is checked on the tokens because it is the axis tokens_to_latents() expands:
a discrete tokenizer's [B, T, U, 1] token IDs become [B, T, U, C] code vectors.
T and U are checked on the latents because those are in the canonical layout
for every tokenizer, whereas encode() may use a tokenizer-native layout
(e.g. Perceiver's [B, M, D_lat, L]).
"""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

import ntt.contrib.tokenizer  # noqa: F401 -- populate the tokenizer registry
from ntt.core.components.types import BinnedSpikesInput, DataType, LatentType, RatesInput
from ntt.core.model.tokenizer.base import BaseTokenizer, get_tokenizer_class, list_tokenizers

_N_UNITS = 5
_BATCH_SIZE = 2


def _resolve_n_units(spec) -> int:
    if spec.n_neurons is False:
        return _N_UNITS
    return int(spec.n_neurons)


def _make_dummy_input(spec, batch_size: int):
    """Create dummy binned input matching the tokenizer's declared input_type."""
    if spec.input_type == DataType.SPIKE_TIMES:
        pytest.skip(f"{spec.KIND} declares unbinned input_type={spec.input_type}")

    n_units = _resolve_n_units(spec)
    n_bins = spec.window_length_bins
    data = torch.rand(batch_size, n_bins, n_units)
    unit_ids = torch.arange(n_units).unsqueeze(0).expand(batch_size, -1)
    common = {
        "pad_and_gap_mask": torch.ones_like(data),
        "sampling_rate": 1000.0 / float(spec.bin_size_ms),
        "unit_ids": unit_ids,
        "unit_ids_pad_mask": torch.ones(batch_size, n_units),
        "duration_sec": torch.full((batch_size,), float(spec.window_length_seconds)),
        "session_id": ("__validation__",) * batch_size,
        "subject_id": ("test_subject",) * batch_size,
        "species": ("test_species",) * batch_size,
        "absolute_start": torch.zeros(batch_size, dtype=torch.float64),
        "absolute_end": torch.full(
            (batch_size,), float(spec.window_length_seconds), dtype=torch.float64
        ),
    }
    if spec.input_type == DataType.RATES:
        return RatesInput(rate_matrix=data, **common)
    if spec.input_type == DataType.SPIKES:
        return BinnedSpikesInput(spike_matrix=(data > 0.8).to(torch.int32), **common)
    raise AssertionError(f"Unhandled input_type={spec.input_type!r} for KIND={spec.KIND!r}")


def _make_perceiver_embeddings(tmp_path, *, n_units: int, d_embed: int) -> str:
    path = tmp_path / "perceiver_embeddings.pt"
    torch.save(
        {
            "embs": torch.randn(n_units, d_embed),
            "unit_ids": np.asarray([f"__validation__/{idx}" for idx in range(n_units)]),
            "session_ids": np.asarray(["__validation__"] * n_units),
        },
        path,
    )
    return str(path)


def _make_test_spec(kind: str, tmp_path):
    """Return a small valid spec for a registered tokenizer KIND."""
    # The registry is typed with the base TokenizerSpec, which does not know the
    # per-KIND fields; pydantic validates the kwargs below at runtime.
    spec_cls: Any = get_tokenizer_class(kind)[0]
    if kind == "calm":
        return spec_cls(
            n_neurons=_N_UNITS,
            window_length_seconds=1.0,
            bin_size_ms=10.0,
            patch_length=10,
            embed_dim=16,
            num_encoder_layers=1,
            num_decoder_layers=1,
            num_heads=2,
            codebook_size=16,
            bottleneck_type="vq",
            ar_weight=0.0,
        )
    if kind == "compressed_sensing":
        return spec_cls(
            window_length_seconds=1.0,
            bin_size_ms=10.0,
            latent_dim=8,
            compress_mode="per_neuron",
        )
    if kind == "conv_vae":
        return spec_cls(
            n_neurons=_N_UNITS,
            window_length_seconds=0.1,
            bin_size_ms=1.0,
            hidden_channels=8,
            latent_channels=4,
            channel_multipliers=(1,),
            downsample_factors=(2,),
            num_res_blocks=1,
        )
    if kind == "cosmos_2d":
        return spec_cls(
            n_neurons=_N_UNITS,
            window_length_seconds=0.1,
            bin_size_ms=1.0,
            channels=8,
            channels_mult=(1,),
            num_res_blocks=1,
            z_channels=8,
            neuron_compression=1,
            temporal_compression=1,
            patch_w=1,
            latent_channels=4,
        )
    if kind == "frequency_based_lut":
        return spec_cls(
            vocab_size=8,
            n_neurons=1,
            window_length_seconds=0.1,
            bin_size_ms=1.0,
        )
    if kind == "fsqffn":
        return spec_cls(
            n_neurons=1,
            window_length_seconds=0.1,
            bin_size_ms=1.0,
            q_levels=(4, 4),
            ffn_model_dim=16,
            ffn_multiple_of=16,
        )
    if kind == "identity":
        return spec_cls(window_length_seconds=1.0, bin_size_ms=10.0)
    if kind == "ldns":
        return spec_cls(
            n_neurons=_N_UNITS,
            window_length_seconds=1.0,
            bin_size_ms=10.0,
            latent_dim=8,
            hidden_dim=16,
            num_encoder_blocks=1,
            num_decoder_blocks=0,
        )
    if kind == "pca_baseline":
        return spec_cls(
            n_neurons=_N_UNITS,
            window_length_seconds=0.1,
            bin_size_ms=1.0,
            n_components=2,
        )
    if kind == "perceiver_s4":
        d_embed = 4
        return spec_cls(
            n_neurons=_N_UNITS,
            window_length_seconds=0.1,
            bin_size_ms=10.0,
            hidden_dim=4,
            latent_tokens=2,
            d_embed=d_embed,
            num_heads=1,
            mlp_ratio=1,
            num_encoder_N_blocks=1,
            num_encoder_M_blocks=1,
            num_decoder_blocks=1,
            use_s4=False,
            grad_checkpointing=False,
            embeddings_path=_make_perceiver_embeddings(
                tmp_path, n_units=_N_UNITS, d_embed=d_embed
            ),
        )
    if kind == "poisson_baseline":
        return spec_cls(
            window_length_seconds=1.0,
            bin_size_ms=10.0,
            rate_estimation_bin_ms=100.0,
        )
    if kind == "simple_vae":
        return spec_cls(
            window_length_seconds=1.0,
            bin_size_ms=10.0,
            latent_dim=8,
        )
    if kind == "wanneural":
        return spec_cls(
            n_neurons=_N_UNITS,
            window_length_seconds=0.2,
            bin_size_ms=1.0,
            dim=16,
            z_dim=8,
            num_levels=3,
            dim_mult=(1, 2, 4),
            num_res_blocks=1,
            attn_levels=(),
            num_attn_heads=1,
            temporal_patch_size=1,
            temporal_downsample=(True, True),
            token_bin_size_ms=4.0,
        )
    raise AssertionError(f"No test spec factory for registered tokenizer KIND={kind!r}")


def _encode_tokens(tokenizer: BaseTokenizer, x) -> torch.Tensor:
    """Native encode() output, unpacked the same way the base eval paths do."""
    tokens, _provided = tokenizer._unpack_encode_result(tokenizer.encode(x))
    return tokens


def _assert_eval_latents(
    kind: str, source: str, latents: torch.Tensor, tokens: torch.Tensor, spec
) -> None:
    """Check eval latents and encode() tokens against the has_*_dim flags.

    See the module docstring for why T/U are checked on the latents and D on
    the tokens.
    """
    assert latents.is_floating_point(), (
        f"{source} for tokenizer KIND={kind!r} must return continuous floating-point "
        "latents. Latent-space evals add numeric perturbations to these values; "
        "integer token IDs are not a valid latent representation."
    )
    assert latents.ndim == 4, (
        f"{source} for tokenizer KIND={kind!r} must return 4D latents "
        f"[B, T, U, D], got shape {tuple(latents.shape)}"
    )
    for axis, flag in ((1, "has_time_dim"), (2, "has_unit_dim")):
        if not getattr(spec, flag):
            assert latents.shape[axis] == 1, (
                f"{source} for {kind!r}: {flag}=False requires axis {axis} to be 1, "
                f"got {latents.shape[axis]}"
            )
    # shape[-1], not shape[3]: encode() tokens need not be 4D, and an
    # IndexError here would escape the xfail(raises=AssertionError) markers.
    if spec.has_latent_dim:
        assert tokens.shape[-1] > 1, (
            f"encode(x) for {kind!r}: has_latent_dim=True requires the token "
            f"latent axis to be > 1, got tokens of shape {tuple(tokens.shape)}"
        )
    else:
        assert tokens.shape[-1] == 1, (
            f"encode(x) for {kind!r}: has_latent_dim=False requires the token "
            f"latent axis to be 1, got tokens of shape {tuple(tokens.shape)}"
        )


@pytest.mark.parametrize(
    ("latents_shape", "tokens_shape", "match"),
    [
        ((2, 3, 1, 16), (2, 3, 1, 1), "has_time_dim=False requires axis 1"),
        ((2, 1, 4, 16), (2, 1, 4, 1), "has_unit_dim=False requires axis 2"),
        ((2, 1, 1, 16), (2, 1, 1, 16), "has_latent_dim=False requires the token"),
    ],
)
def test_absent_axes_must_be_singletons(latents_shape, tokens_shape, match):
    spec = SimpleNamespace(has_time_dim=False, has_unit_dim=False, has_latent_dim=False)
    # Discrete-style pass: D=1 token IDs expanded to D=16 code vectors.
    _assert_eval_latents("test", "test", torch.zeros(2, 1, 1, 16), torch.zeros(2, 1, 1, 1), spec)
    with pytest.raises(AssertionError, match=match):
        _assert_eval_latents(
            "test", "test", torch.zeros(latents_shape), torch.zeros(tokens_shape), spec
        )


def test_present_latent_axis_must_not_be_singleton():
    spec = SimpleNamespace(has_time_dim=True, has_unit_dim=True, has_latent_dim=True)
    # Continuous-style pass: encode() already returns the D=16 latents.
    latents = torch.zeros(2, 3, 4, 16)
    _assert_eval_latents("test", "test", latents, latents, spec)
    with pytest.raises(AssertionError, match="has_latent_dim=True requires the token"):
        _assert_eval_latents("test", "test", latents, torch.zeros(2, 3, 4, 1), spec)


def test_eval_latents_must_be_continuous():
    spec = SimpleNamespace(has_time_dim=True, has_unit_dim=True, has_latent_dim=True)
    tokens = torch.zeros(2, 3, 4, 1, dtype=torch.long)
    with pytest.raises(AssertionError, match="continuous floating-point"):
        _assert_eval_latents("test", "test", tokens, tokens, spec)


def test_tokenizer_output_format_specs_cover_registered_tokenizers(tmp_path):
    """The test cases must be keyed from the production tokenizer registry."""
    covered = {_make_test_spec(kind, tmp_path).KIND for kind in list_tokenizers()}
    assert covered == set(list_tokenizers())


_LATENT_OUTPUT_XFAIL_KINDS: set[str] = {
    "identity",  # example pass-through tokenizer returns dummy integer tokens, not latents.
}


@pytest.mark.parametrize("kind", list_tokenizers())
def test_encode_with_timestamps_returns_4d_continuous_latents(kind, tmp_path, request):
    """Eval-facing encode path returns continuous [B, T, U, D] latents."""
    if kind in _LATENT_OUTPUT_XFAIL_KINDS:
        request.node.add_marker(
            pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason=(
                    f"tokenizer KIND={kind!r} does not yet return continuous 4D "
                    "latents on the eval-facing encode path (latent contract)"
                ),
            )
        )

    spec = _make_test_spec(kind, tmp_path)
    _spec_cls, tokenizer_cls = get_tokenizer_class(spec.KIND)
    tokenizer = tokenizer_cls(spec)
    tokenizer.eval()

    x = _make_dummy_input(spec, _BATCH_SIZE)
    with torch.no_grad():
        tokens = _encode_tokens(tokenizer, x)
        latents, metadata = tokenizer.encode_with_timestamps(x)

    _assert_eval_latents(spec.KIND, "encode_with_timestamps(x)", latents, tokens, spec)
    assert metadata["token_timestamps"].shape == latents.shape[:2]
    assert metadata["token_start_time"].shape == latents.shape[:2]
    assert metadata["token_end_time"].shape == latents.shape[:2]


_RECONSTRUCT_FROM_INPUT_LATENTS_XFAIL_KINDS: set[str] = {
    *_LATENT_OUTPUT_XFAIL_KINDS,
}


@pytest.mark.parametrize("kind", list_tokenizers())
def test_reconstruct_from_input_returns_4d_continuous_latents(kind, tmp_path, request):
    """Eval-facing reconstruct path returns continuous [B, T, U, D] latents."""
    if kind in _RECONSTRUCT_FROM_INPUT_LATENTS_XFAIL_KINDS:
        request.node.add_marker(
            pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason=(
                    f"tokenizer KIND={kind!r} does not yet return continuous 4D "
                    "latents on the eval-facing encode path (latent contract)"
                ),
            )
        )

    spec = _make_test_spec(kind, tmp_path)
    _spec_cls, tokenizer_cls = get_tokenizer_class(spec.KIND)
    tokenizer = tokenizer_cls(spec)
    tokenizer.eval()

    x = _make_dummy_input(spec, _BATCH_SIZE)
    with torch.no_grad():
        tokens = _encode_tokens(tokenizer, x)
        output = tokenizer.reconstruct_from_input(x, return_latents=True)

    assert isinstance(output, dict), "return_latents=True must return a dict"
    latents = output["latents"]
    _assert_eval_latents(
        spec.KIND,
        "reconstruct_from_input(x, return_latents=True)['latents']",
        latents,
        tokens,
        spec,
    )
    assert output["token_timestamps"].shape == latents.shape[:2]
    if "token_start_time" in output:
        assert output["token_start_time"].shape == latents.shape[:2]
    if "token_end_time" in output:
        assert output["token_end_time"].shape == latents.shape[:2]


_DISCRETE_TOKENS_TO_LATENTS_XFAIL_KINDS: set[str] = {
    "identity",  # tokens_to_latents is identity over dummy integer token IDs.
}


@pytest.mark.parametrize("kind", list_tokenizers())
def test_discrete_tokens_to_latents_returns_continuous_values(kind, tmp_path, request):
    """Discrete tokenizers must convert token IDs before latent-space evals add noise."""
    spec = _make_test_spec(kind, tmp_path)
    if spec.latent_type != LatentType.DISCRETE:
        pytest.skip(f"tokenizer KIND={kind!r} is not discrete")

    if kind in _DISCRETE_TOKENS_TO_LATENTS_XFAIL_KINDS:
        request.node.add_marker(
            pytest.mark.xfail(
                strict=True,
                reason=(
                    f"tokenizer KIND={kind!r} does not convert discrete token IDs "
                    "to continuous latents"
                ),
            )
        )

    _spec_cls, tokenizer_cls = get_tokenizer_class(spec.KIND)
    tokenizer = tokenizer_cls(spec)
    tokenizer.eval()

    x = _make_dummy_input(spec, _BATCH_SIZE)
    with torch.no_grad():
        tokens = _encode_tokens(tokenizer, x)
        latents = tokenizer.tokens_to_latents(tokens)

    assert latents.is_floating_point(), (
        f"discrete tokenizer KIND={spec.KIND!r} must expose continuous latents from "
        "tokens_to_latents(tokens), not integer token IDs. Latent-space robustness "
        "evaluation adds noise to the output of tokens_to_latents(tokens); adding "
        "Gaussian noise to integer code labels is not a valid latent perturbation."
    )


_M2_EVAL_LATENTS_XFAIL_KINDS: set[str] = {
    *_LATENT_OUTPUT_XFAIL_KINDS,
}


@pytest.mark.parametrize("kind", list_tokenizers())
def test_tokenizer_m2_eval_latents_reconstruct(kind, tmp_path, request):
    """M2 must decode the continuous 4D latents produced by its clean eval path."""
    if kind in _M2_EVAL_LATENTS_XFAIL_KINDS:
        request.node.add_marker(
            pytest.mark.xfail(
                strict=True,
                reason=(
                    f"tokenizer KIND={kind!r} does not implement the M2 "
                    "continuous 4D latent reconstruction contract"
                ),
            )
        )

    spec = _make_test_spec(kind, tmp_path)
    _spec_cls, tokenizer_cls = get_tokenizer_class(spec.KIND)
    tokenizer = tokenizer_cls(spec)
    tokenizer.eval()

    x = _make_dummy_input(spec, _BATCH_SIZE)
    with torch.no_grad():
        tokens = _encode_tokens(tokenizer, x)
        if spec.KIND == "poisson_baseline":
            # M2 uses a Poisson-specific clean path to make the sampled clean
            # reconstruction deterministic, then adds noise to tokens_to_latents(tokens).
            eval_tokens, _metadata = tokenizer.encode_with_timestamps(x)
            clean_reconstruction = tokenizer.decode(eval_tokens)
            latents = tokenizer.tokens_to_latents(eval_tokens)
        else:
            output = tokenizer.reconstruct_from_input(x, return_latents=True)
            assert isinstance(output, dict) and "latents" in output, (
                f"reconstruct_from_input(x, return_latents=True) for tokenizer "
                f"KIND={spec.KIND!r} must return a 'latents' entry for M2."
            )
            clean_reconstruction = output["reconstruction"]
            latents = output["latents"]

        _assert_eval_latents(spec.KIND, "M2 clean eval path latents", latents, tokens, spec)

        if spec.KIND == "perceiver_s4":
            # @register_tokenizer is typed type[BaseTokenizer] -> type[BaseTokenizer],
            # which erases the subclass, so pyright cannot see Perceiver's members.
            perceiver: Any = tokenizer
            embeddings, _mask = perceiver._get_embeddings(x)
            reconstructed = perceiver.count_wrapper.decode(
                latents.permute(0, 2, 3, 1).contiguous(),
                embeddings,
            ).transpose(-1, -2)
        elif spec.KIND == "calm":
            noisy_latents = latents + 0.01 * torch.randn_like(latents)
            reconstructed = tokenizer.reconstruct(noisy_latents).reconstruction
        else:
            reconstructed = tokenizer.reconstruct(latents).reconstruction

    assert reconstructed.shape == clean_reconstruction.shape, (
        f"M2 latent reconstruction shape mismatch for KIND={spec.KIND!r}: "
        f"clean reconstruction shape={tuple(clean_reconstruction.shape)}, "
        f"decode/reconstruct from M2 latents shape={tuple(reconstructed.shape)}"
    )
