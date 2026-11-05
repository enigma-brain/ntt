"""FSQFFN's continuous representation must preserve discrete decoding."""

from unittest.mock import patch

import pytest
import torch

from ntt.contrib.tokenizer.fsqffn.fsqffn import FSQFFNTokenizer, FSQFFNTokenizerSpec


def _tokenizer(q_levels=(4, 4), q_dim=None):
    return FSQFFNTokenizer(
        FSQFFNTokenizerSpec(
            window_length_seconds=0.008,
            bin_size_ms=1.0,
            q_levels=q_levels,
            q_dim=q_dim,
            ffn_model_dim=8,
            ffn_multiple_of=8,
        )
    ).eval()


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("q_levels,q_dim", [((4,), None), ((4, 4), None), ((4, 4), 3)])
def test_continuous_decode_matches_original_integer_decode(batch_size, q_levels, q_dim):
    tokenizer = _tokenizer(q_levels, q_dim)
    tokens = torch.arange(batch_size).reshape(batch_size, 1)
    width = q_dim or len(q_levels)
    with torch.no_grad():
        # Original decoder computation, independent of the new conversion method.
        codes = tokenizer.quantizer.indices_to_codes(tokens[:, 0], project_out=True)
        expected = tokenizer.output_act(tokenizer.decoder(codes)).unsqueeze(-1)
        latents = tokenizer.tokens_to_latents(tokens)
        assert latents.shape == (batch_size, 1, 1, width)
        assert latents.is_floating_point()
        torch.testing.assert_close(latents[:, 0, 0, :], codes)
        torch.testing.assert_close(tokenizer.decode(tokens), expected)
        torch.testing.assert_close(tokenizer.reconstruct(latents).reconstruction, expected)


def test_noisy_latents_reach_decoder_without_lookup_or_quantization():
    tokenizer = _tokenizer(q_dim=3)
    with torch.no_grad():
        latents = tokenizer.tokens_to_latents(torch.tensor([[1]]))
        noisy = latents + torch.tensor([0.13, -0.27, 0.41])
        expected = tokenizer.output_act(tokenizer.decoder(noisy[:, 0, 0, :])).unsqueeze(-1)
        with (
            patch.object(
                tokenizer.quantizer, "indices_to_codes", side_effect=AssertionError("lookup")
            ),
            patch.object(
                tokenizer.quantizer, "forward", side_effect=AssertionError("quantization")
            ),
            patch.object(tokenizer.decoder, "forward", wraps=tokenizer.decoder.forward) as decode,
        ):
            actual = tokenizer.reconstruct(noisy).reconstruction
        torch.testing.assert_close(decode.call_args.args[0], noisy[:, 0, 0, :], rtol=0, atol=0)
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("shape", [(1, 2), (1, 2, 1, 2), (1, 1, 2, 2), (1, 1, 1, 3)])
def test_decode_rejects_noncanonical_continuous_shapes(shape):
    with pytest.raises(ValueError, match="Expected continuous latents"):
        _tokenizer().decode(torch.zeros(shape))
