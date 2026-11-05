"""Tests for :mod:`ntt.evaluation.utils.window_chunking`.

The wrapper pre-chunks fixed-input tokenizers and passes through variable-T
tokenizers; outputs match the tokenizer's natural return type, just with
time-axis fields concatenated across chunks.
"""

import pytest
import torch

from ntt.contrib.tokenizer.calm.calm_tokenizer import CalmTokenizerSpec
from ntt.contrib.tokenizer.identity_tokenizer import IdentityTokenizerSpec
from ntt.contrib.tokenizer.simple_vae import SimpleVAETokenizerSpec
from ntt.core.components.types import BinnedSpikesInput, DataType, ReconstructionOutput
from ntt.evaluation.utils.window_chunking import (
    _concat_chunk_outputs,
    chunked_tokenizer_fn,
    requires_fixed_input_window,
)


def _chunk(latents=None, extra=None):
    out = {"reconstruction": torch.zeros(1, 2, 3)}
    if latents is not None:
        out["latents"] = latents
    if extra is not None:
        out["extra"] = extra
    return out


def test_partial_latents_presence_raises():
    chunk_a = _chunk(latents=torch.zeros(1, 1, 4))
    chunk_b = _chunk(latents=None)  # no 'latents' key at all
    with pytest.raises(ValueError, match="present in some chunks"):
        _concat_chunk_outputs([chunk_a, chunk_b])


def test_divergent_extra_key_raises():
    with pytest.raises(ValueError, match="differs across chunks"):
        _concat_chunk_outputs([_chunk(extra="a"), _chunk(extra="b")])


def test_consistent_extra_key_passes_through():
    merged = _concat_chunk_outputs([_chunk(extra="same"), _chunk(extra="same")])
    assert merged["extra"] == "same"


def test_all_present_latents_still_concatenate():
    chunk_a = _chunk(latents=torch.zeros(1, 1, 4))
    chunk_b = _chunk(latents=torch.ones(1, 1, 4))
    merged = _concat_chunk_outputs([chunk_a, chunk_b])
    assert merged["latents"].shape == (1, 2, 4)


def _make_batch(
    t_bins: int,
    *,
    batch_size: int = 1,
    units: int = 4,
    sampling_rate: float = 1000.0,
) -> BinnedSpikesInput:
    """Uniform-T batch — chunking helper sees only the tensor shape."""
    spike = torch.zeros((batch_size, t_bins, units), dtype=torch.int32)
    mask = torch.zeros((batch_size, t_bins, units), dtype=torch.int32)
    duration = t_bins / sampling_rate
    durations = torch.tensor([duration] * batch_size, dtype=torch.float64)
    return BinnedSpikesInput(
        spike_matrix=spike,
        pad_and_gap_mask=mask,
        sampling_rate=torch.tensor([sampling_rate] * batch_size),
        unit_ids=torch.zeros((batch_size, units), dtype=torch.int64),
        unit_ids_pad_mask=torch.ones((batch_size, units), dtype=torch.int64),
        duration_sec=durations,
        session_id=tuple(["S"] * batch_size),
        subject_id=tuple(["s"] * batch_size),
        species=tuple(["x"] * batch_size),
        absolute_start=torch.zeros(batch_size, dtype=torch.float64),
        absolute_end=durations.clone(),
        behavioral_data=None,
        behavioral_intervals=None,
    )


def test_detection_derives_from_variable_t():
    """variable_t=False -> fixed-input; variable_t=True -> not fixed-input."""
    fixed = SimpleVAETokenizerSpec(
        window_length_seconds=1.0,
        bin_size_ms=10,
        latent_dim=16,
        hidden_dims=(32,),
    )
    assert fixed.variable_t is False
    assert requires_fixed_input_window(fixed) is True

    variable = IdentityTokenizerSpec(window_length_seconds=1.0, bin_size_ms=10)
    assert variable.variable_t is True
    assert requires_fixed_input_window(variable) is False


def test_calm_is_not_treated_as_fixed_input():
    spec = CalmTokenizerSpec(window_length_seconds=2.0, bin_size_ms=10.0, n_neurons=8)
    assert requires_fixed_input_window(spec) is False


def test_variable_passthrough_returns_batched_output_unchanged():
    """Variable-T tokenizers see the batch once; output is the tokenizer's
    natural batched return value."""
    spec = IdentityTokenizerSpec(window_length_seconds=1.0, bin_size_ms=1)
    batch = _make_batch(700, batch_size=2)
    calls: list[int] = []

    def fn(b):
        calls.append(b.spike_matrix.shape[1])
        return {"reconstruction": torch.zeros_like(b.spike_matrix, dtype=torch.float32)}

    wrapped = chunked_tokenizer_fn(fn, spec=spec)
    out = wrapped(batch)
    # Exactly one call, full batch.
    assert calls == [700]
    # Batched output (not a list).
    assert isinstance(out, dict)
    assert out["reconstruction"].shape == (2, 700, 4)


def test_fixed_chunks_along_time_and_concatenates():
    """Fixed-T tokenizer: input [B=2, T=300, U=1] with T_chunk=100 → three
    chunks per sample, called per chunk, concatenated along time axis 1."""
    spec = SimpleVAETokenizerSpec(
        window_length_seconds=0.1,
        bin_size_ms=1,
        latent_dim=16,
        hidden_dims=(32,),
    )
    batch = _make_batch(300, batch_size=2, units=1)
    seen_chunk_shapes: list[tuple[int, ...]] = []
    seen_chunk_starts: list[float] = []

    def fn(b):
        seen_chunk_shapes.append(tuple(b.spike_matrix.shape))
        seen_chunk_starts.append(float(b.absolute_start[0].item()))
        chunk_batch = b.spike_matrix.shape[0]
        return {
            "reconstruction": torch.zeros_like(b.spike_matrix, dtype=torch.float32),
            "latents": torch.zeros((chunk_batch, 1, 1, 3), dtype=torch.float32),
            "token_timestamps": torch.zeros((chunk_batch, 1), dtype=torch.float64),
            "token_start_time": torch.zeros((chunk_batch, 1), dtype=torch.float64),
            "token_end_time": torch.zeros((chunk_batch, 1), dtype=torch.float64),
        }

    wrapped = chunked_tokenizer_fn(fn, spec=spec)
    out = wrapped(batch)

    # 3 chunks, each shape (2, 100, 1).
    assert seen_chunk_shapes == [(2, 100, 1), (2, 100, 1), (2, 100, 1)]
    # Chunk absolute_start is offset by k * T_chunk / sampling_rate.
    assert seen_chunk_starts == pytest.approx([0.0, 0.1, 0.2])

    # Output is batched (not a list); time axis 1 concatenated.
    assert isinstance(out, dict)
    assert out["reconstruction"].shape == (2, 300, 1)
    assert out["latents"].shape == (2, 3, 1, 3)
    assert out["token_timestamps"].shape == (2, 3)


def test_fixed_drops_trailing_remainder():
    """Input not divisible by T_chunk → drop the tail; reconstruction
    shorter than input."""
    spec = SimpleVAETokenizerSpec(
        window_length_seconds=0.1,
        bin_size_ms=1,
        latent_dim=16,
        hidden_dims=(32,),
    )
    batch = _make_batch(250, batch_size=1, units=1)

    def fn(b):
        return {"reconstruction": torch.zeros_like(b.spike_matrix, dtype=torch.float32)}

    wrapped = chunked_tokenizer_fn(fn, spec=spec)
    out = wrapped(batch)

    # floor(250 / 100) = 2 chunks → 200 bins kept, 50 dropped.
    assert out["reconstruction"].shape == (1, 200, 1)


def test_fixed_input_shorter_than_chunk_raises():
    """If the input window is shorter than one chunk we cannot tokenize."""
    spec = SimpleVAETokenizerSpec(
        window_length_seconds=0.1,
        bin_size_ms=1,
        latent_dim=16,
        hidden_dims=(32,),
    )
    batch = _make_batch(50, batch_size=1, units=1)

    def fn(b):
        return {"reconstruction": torch.zeros_like(b.spike_matrix, dtype=torch.float32)}

    wrapped = chunked_tokenizer_fn(fn, spec=spec)
    with pytest.raises(ValueError) as exc_info:
        wrapped(batch)
    assert "shorter than one tokenizer chunk" in str(exc_info.value)


def test_fixed_supports_reconstruction_output():
    """The wrapper preserves the wrapped function's output type (here a
    ``ReconstructionOutput`` NamedTuple instead of a dict)."""
    spec = SimpleVAETokenizerSpec(
        window_length_seconds=0.1,
        bin_size_ms=1,
        latent_dim=16,
        hidden_dims=(32,),
    )
    batch = _make_batch(300, batch_size=2, units=1)

    def fn(b):
        return ReconstructionOutput(
            reconstruction=torch.zeros_like(b.spike_matrix, dtype=torch.float32),
            data_type=DataType.SPIKES,
            bin_size_seconds=spec.bin_size_seconds,
        )

    wrapped = chunked_tokenizer_fn(fn, spec=spec)
    out = wrapped(batch)
    assert isinstance(out, ReconstructionOutput)
    assert out.reconstruction.shape == (2, 300, 1)


def test_fixed_exact_match_no_chunking_work():
    """Input length already equals T_chunk → wrapper calls tokenize_fn
    exactly once with the original batch."""
    spec = SimpleVAETokenizerSpec(
        window_length_seconds=0.1,
        bin_size_ms=1,
        latent_dim=16,
        hidden_dims=(32,),
    )
    batch = _make_batch(100, batch_size=1, units=1)
    seen_call_shapes: list[tuple[int, ...]] = []

    def fn(b):
        seen_call_shapes.append(tuple(b.spike_matrix.shape))
        return {"reconstruction": torch.zeros_like(b.spike_matrix, dtype=torch.float32)}

    wrapped = chunked_tokenizer_fn(fn, spec=spec)
    out = wrapped(batch)
    assert seen_call_shapes == [(1, 100, 1)]
    assert out["reconstruction"].shape == (1, 100, 1)


def test_variable_shape_error_points_to_variable_t_flag():
    """If a tokenizer treated as variable raises a shape error, the wrapper
    re-raises with a hint pointing at the ``variable_t`` capability flag."""
    spec = IdentityTokenizerSpec(window_length_seconds=1.0, bin_size_ms=1)
    batch = _make_batch(140, batch_size=1)

    def fn(_batch):
        raise RuntimeError("size mismatch: expected 100 bins, got 140")

    wrapped = chunked_tokenizer_fn(fn, spec=spec)
    with pytest.raises(RuntimeError) as exc_info:
        wrapped(batch)

    message = str(exc_info.value)
    assert "variable_t" in message
    assert "ntt.evaluation.utils.window_chunking" in message
    assert "identity" in message
