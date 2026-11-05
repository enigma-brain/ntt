import pydantic as pdt
import pytest

from ntt.contrib.tokenizer.calm.calm_tokenizer import CalmTokenizerSpec
from ntt.contrib.tokenizer.compressed_sensing import CompressedSensingTokenizerSpec
from ntt.contrib.tokenizer.identity_tokenizer import IdentityTokenizerSpec
from ntt.contrib.tokenizer.simple_vae import SimpleVAETokenizerSpec
from ntt.evaluation.utils.window_chunking import requires_fixed_input_window


def test_fixed_input_field_is_gone():
    """The redundant requires_fixed_input_window field was removed (extra=forbid)."""
    with pytest.raises(pdt.ValidationError):
        SimpleVAETokenizerSpec(
            window_length_seconds=1.0,
            bin_size_ms=10,
            latent_dim=16,
            hidden_dims=(32,),
            requires_fixed_input_window=False,  # type: ignore[call-arg]  # field removed on purpose
        )


def test_variable_t_specs_are_not_fixed_input():
    spec = IdentityTokenizerSpec(window_length_seconds=1.0, bin_size_ms=10)
    assert spec.variable_t is True
    assert requires_fixed_input_window(spec) is False


def test_non_variable_t_specs_require_fixed_input():
    spec = SimpleVAETokenizerSpec(
        window_length_seconds=1.0, bin_size_ms=10, latent_dim=16, hidden_dims=(32,)
    )
    assert spec.variable_t is False
    assert requires_fixed_input_window(spec) is True


def test_cs_per_time_bin_opts_out_via_variable_t():
    spec = CompressedSensingTokenizerSpec(
        window_length_seconds=1.0, bin_size_ms=10, compress_mode="per_time_bin", n_neurons=50
    )
    assert spec.variable_t is True
    assert requires_fixed_input_window(spec) is False


def test_cs_per_neuron_still_requires_fixed_input():
    spec = CompressedSensingTokenizerSpec(
        window_length_seconds=1.0, bin_size_ms=10, compress_mode="per_neuron", n_neurons=50
    )
    assert spec.variable_t is False
    assert requires_fixed_input_window(spec) is True


def test_calm_is_variable_input_validated_against_eval_window():
    """calm is variable_t=True and is NOT fixed-input chunked.

    calm needs patch_length-multiple windows, but per the eval-window-as-
    single-source-of-truth design that constraint is enforced by
    ``assert_tokenizer_accepts_eval_window`` (window_contract.py) feeding the
    eval window directly, not by ``requires_fixed_input_window`` chunking.
    """
    spec = CalmTokenizerSpec(bin_size_ms=1.0)
    assert spec.variable_t is True
    assert requires_fixed_input_window(spec) is False
