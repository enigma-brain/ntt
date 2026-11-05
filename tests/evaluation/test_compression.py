"""Tests for ntt.evaluation.compression."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Literal, NamedTuple
from unittest.mock import MagicMock

import pytest
import torch

from ntt.contrib.tokenizer.identity_tokenizer import IdentityTokenizer, IdentityTokenizerSpec
from ntt.core.components.dist import Dist, UniformDist
from ntt.core.components.types import LatentType
from ntt.evaluation.compression import compute_compression


@dataclass
class _SpecStub:
    """Minimal stand-in for a TokenizerSpec.

    The real spec is a Pydantic frozen model with many fields; tests only
    need the attributes compute_compression actually reads.
    """

    latent_type: LatentType
    bin_size_ms: float | None
    effective_window_length_seconds: float
    # False is the real spec's "unset" sentinel; read to decide whether a sampled
    # neuron count could leave compression_rate non-deterministic.
    n_neurons: int | Dist | Literal[False] = False
    _vocab_size: int | None = None
    _codes_per_token: int | None = 1

    def get_vocab_size(self) -> int:
        if self._vocab_size is None:
            raise NotImplementedError("vocab unknown")
        return self._vocab_size

    def get_codes_per_token(self) -> int:
        if self._codes_per_token is None:
            raise NotImplementedError("codes per token unknown")
        return self._codes_per_token


class _DummyInput(NamedTuple):
    """Minimal stand-in for a binned input.

    A ``NamedTuple`` (like the real ``BinnedSpikesInput``) so Lightning's
    ``move_data_to_device`` relocates its tensors — plain objects are left
    untouched by that utility.
    """

    spike_matrix: torch.Tensor


def _patch_dummy_input(monkeypatch, n_neurons: int = 32) -> None:
    # 200 ms @ 1 ms bins → 200 time bins.
    # ``**_kw`` swallows the ``window_length_seconds`` override compute_compression now
    # passes to pin the dummy input to a deterministic window.
    monkeypatch.setattr(
        "ntt.evaluation.compression.make_dummy_input",
        lambda _spec, **_kw: _DummyInput(spike_matrix=torch.zeros((1, 200, n_neurons))),
    )


def _make_tokenizer(latents):
    tok = MagicMock()
    tok.encode.return_value = latents
    return tok


def test_compute_compression_discrete_counts_bits_per_code_not_per_dim(monkeypatch):
    """An FSQ-style latent: the D=2 values of one ``[t, u]`` position are ONE code from a
    vocab-1024 alphabet, so the position carries log2(1024) = 10 bits, not 2 * 10.

    The old rule charged ``log2(vocab)`` to every latent dim and inflated output_bits by
    D for every vector-coded quantizer (FSQ/BSQ/LFQ and their residual variants).
    """
    _patch_dummy_input(monkeypatch, n_neurons=4)
    spec = _SpecStub(
        latent_type=LatentType.DISCRETE,
        bin_size_ms=1.0,
        effective_window_length_seconds=0.2,
        _vocab_size=1024,  # log2 = 10
        _codes_per_token=1,
    )
    latents = torch.zeros((1, 5, 1, 2))  # B, T, U, latent_dim
    tok = _make_tokenizer(latents)

    result = compute_compression(spec, tok)  # type: ignore[arg-type]

    assert result.n_neurons == 4
    assert result.t_tokens == 5
    assert result.u_units == 1
    assert result.latent_dim == 2
    assert result.bits_per_dim == 10.0  # log2(1024), i.e. bits per code
    assert result.codes_per_token == 1
    assert result.bits_per_token == 10.0
    # input bits = log2(1+1) * (200 ms / 1 ms) bins * 4 neurons = 800
    assert result.input_bits == pytest.approx(800.0)
    # output bits = 5 * 1 * (1 code * 10 bits) = 50
    assert result.output_bits == pytest.approx(50.0)
    assert result.compression_rate == pytest.approx(16.0)
    # input_elements = 1 * 200 * 4 = 800; output_elements = 1 * 5 * 1 * 2 = 10
    assert result.input_elements == 800
    assert result.output_elements == 10
    assert result.latent_dim_reduction == pytest.approx(80.0)


def test_compute_compression_discrete_multi_code_token_is_independent_of_latent_dim(
    monkeypatch,
):
    """A residual quantizer emits ``codes_per_token`` codes per position (one per stage);
    the bits are ``codes_per_token * log2(vocab)`` whatever the latent width D is."""
    _patch_dummy_input(monkeypatch, n_neurons=4)
    spec = _SpecStub(
        latent_type=LatentType.DISCRETE,
        bin_size_ms=1.0,
        effective_window_length_seconds=0.2,
        _vocab_size=1000,  # per-stage prod([8, 5, 5, 5])
        _codes_per_token=4,  # num_quantizers
    )
    tok = _make_tokenizer(torch.zeros((1, 5, 1, 64)))  # D = 64 projected-out channels

    result = compute_compression(spec, tok)  # type: ignore[arg-type]

    assert result.codes_per_token == 4
    assert result.bits_per_token == pytest.approx(4 * math.log2(1000))
    assert result.output_bits == pytest.approx(5 * 1 * 4 * math.log2(1000))


def test_compute_compression_discrete_unknown_codes_per_token_is_nan_and_warns(
    monkeypatch, caplog
):
    """A discrete spec that cannot say how many codes it emits per position must not
    fall back to the per-dim rule: the bit count is NaN and a warning says why."""
    _patch_dummy_input(monkeypatch, n_neurons=4)
    spec = _SpecStub(
        latent_type=LatentType.DISCRETE,
        bin_size_ms=1.0,
        effective_window_length_seconds=0.2,
        _vocab_size=1024,
        _codes_per_token=None,  # get_codes_per_token raises
    )
    tok = _make_tokenizer(torch.zeros((1, 5, 1, 2)))

    with caplog.at_level(logging.WARNING):
        result = compute_compression(spec, tok)  # type: ignore[arg-type]

    assert result.bits_per_dim == 10.0
    assert math.isnan(result.codes_per_token)
    assert math.isnan(result.bits_per_token)
    assert math.isnan(result.output_bits)
    assert math.isnan(result.compression_rate)
    assert "codes per token" in caplog.text


def test_compute_compression_discrete_unknown_vocab_propagates_nan(monkeypatch, caplog):
    """Discrete spec whose get_vocab_size raises → bits_per_dim is NaN, with a warning.

    output_bits and compression_rate inherit the NaN; latent_dim_reduction
    is element-only and stays real.
    """
    _patch_dummy_input(monkeypatch, n_neurons=4)
    spec = _SpecStub(
        latent_type=LatentType.DISCRETE,
        bin_size_ms=1.0,
        effective_window_length_seconds=0.2,
        _vocab_size=None,  # get_vocab_size raises
    )
    tok = _make_tokenizer(torch.zeros((1, 5, 1, 2)))

    with caplog.at_level(logging.WARNING):
        result = compute_compression(spec, tok)  # type: ignore[arg-type]

    assert "vocabulary size unknown" in caplog.text
    assert math.isnan(result.bits_per_dim)
    assert math.isnan(result.bits_per_token)
    assert math.isnan(result.output_bits)
    assert math.isnan(result.compression_rate)
    # element-based stays real
    assert result.input_elements == 800
    assert result.output_elements == 10
    assert result.latent_dim_reduction == pytest.approx(80.0)


def test_compute_compression_continuous_assumes_bf16(monkeypatch):
    """Continuous spec → bits_per_dim = 16 (bf16 convention)."""
    _patch_dummy_input(monkeypatch, n_neurons=4)
    spec = _SpecStub(
        latent_type=LatentType.CONTINUOUS,
        bin_size_ms=1.0,
        effective_window_length_seconds=0.2,
    )
    tok = _make_tokenizer(torch.zeros((1, 5, 1, 2)))

    result = compute_compression(spec, tok)  # type: ignore[arg-type]

    assert result.bits_per_dim == 16.0
    # Every continuous latent value is its own 16-bit symbol.
    assert result.codes_per_token == 2
    assert result.bits_per_token == 32.0
    assert result.output_bits == pytest.approx(5 * 1 * 2 * 16.0)


def test_compute_compression_tokenizer_returns_dict(monkeypatch):
    """tokenizer.encode() may return {"latents": tensor} — handled."""
    _patch_dummy_input(monkeypatch, n_neurons=4)
    spec = _SpecStub(
        latent_type=LatentType.CONTINUOUS,
        bin_size_ms=1.0,
        effective_window_length_seconds=0.2,
    )
    tok = _make_tokenizer({"latents": torch.zeros((1, 5, 1, 2))})

    result = compute_compression(spec, tok)  # type: ignore[arg-type]

    assert result.t_tokens == 5
    assert result.latent_dim == 2


def test_compression_result_to_dict_returns_python_primitives(monkeypatch):
    """to_dict produces a plain-Python dict (no torch/numpy types).

    NaN is allowed in the output and is the documented convention for
    "unknown" — downstream consumers filter NaN via math.isnan. This test
    asserts the dict shape and that NaN flows through unchanged on the
    discrete-unknown-vocab path; portable JSON serialization is not
    promised here (json.dumps emits NaN; downstream readers are Python).
    """
    import json

    _patch_dummy_input(monkeypatch, n_neurons=4)
    spec = _SpecStub(
        latent_type=LatentType.DISCRETE,
        bin_size_ms=1.0,
        effective_window_length_seconds=0.2,
        _vocab_size=256,
    )
    tok = _make_tokenizer(torch.zeros((1, 5, 1, 2)))

    result = compute_compression(spec, tok)  # type: ignore[arg-type]
    payload = result.to_dict()

    # All values are Python primitives (no torch.Tensor leaks); the convention tag is
    # the one string.
    assert payload["convention"] == "per_code_per_bin"
    for key, value in payload.items():
        if key == "convention":
            continue
        assert isinstance(value, (int, float)), (
            f"{key!r} has non-primitive type {type(value).__name__}"
        )

    # CPython json.dumps accepts NaN; downstream callers handle it.
    json.dumps(payload)

    assert set(payload).issuperset(
        {
            "input_bits",
            "output_bits",
            "compression_rate",
            "input_elements",
            "output_elements",
            "latent_dim_reduction",
            "bits_per_dim",
            "bits_per_token",
            "codes_per_token",
            "bits_per_neuron_per_window",
            "n_neurons",
            "t_tokens",
            "u_units",
            "latent_dim",
        }
    )


def test_compression_result_nan_round_trips_through_to_dict(monkeypatch):
    """NaN-producing path: bits_per_dim/output_bits/compression_rate are NaN
    in the dict; element-only fields stay real."""
    _patch_dummy_input(monkeypatch, n_neurons=4)
    spec = _SpecStub(
        latent_type=LatentType.DISCRETE,
        bin_size_ms=1.0,
        effective_window_length_seconds=0.2,
        _vocab_size=None,
    )
    tok = _make_tokenizer(torch.zeros((1, 5, 1, 2)))

    payload = compute_compression(spec, tok).to_dict()  # type: ignore[arg-type]

    assert math.isnan(payload["bits_per_dim"])
    assert math.isnan(payload["bits_per_token"])
    assert math.isnan(payload["output_bits"])
    assert math.isnan(payload["compression_rate"])
    # element-only stays real
    assert payload["input_elements"] == 800
    assert payload["latent_dim_reduction"] == pytest.approx(80.0)


class _DeviceRecordingTokenizer(torch.nn.Module):
    """Records the device of the input it receives in ``encode``.

    Mirrors real tokenizers (e.g. compressed_sensing) whose weights live on
    the module's device: encode must be handed an input on that same device.
    We use the ``meta`` device as a GPU-free stand-in for cuda. (``meta`` is a
    wildcard for arithmetic, so it can't be detected by a raising matmul —
    hence we assert on the *input device* directly.)
    """

    def __init__(self, n_neurons: int, latent_dim: int) -> None:
        super().__init__()
        self.register_buffer("weight", torch.zeros(n_neurons, latent_dim))
        self.received_device: torch.device | None = None

    def encode(self, dummy_input):  # type: ignore[no-untyped-def]
        self.received_device = dummy_input.spike_matrix.device
        # Emit latents on the tokenizer's own device (B, T, U, latent_dim).
        return self.weight.new_zeros(1, 5, 1, self.weight.shape[-1])


def test_compute_compression_moves_dummy_input_to_tokenizer_device(monkeypatch):
    """Dummy input must be moved to the tokenizer's device before encode().

    The dummy input is built on CPU; the tokenizer's buffer lives on ``meta``.
    Without the device-move fix, encode receives a CPU input (which, on a real
    cuda tokenizer, triggers ``Expected all tensors to be on the same device``).
    """
    _patch_dummy_input(monkeypatch, n_neurons=4)
    spec = _SpecStub(
        latent_type=LatentType.CONTINUOUS,
        bin_size_ms=1.0,
        effective_window_length_seconds=0.2,
    )
    tok = _DeviceRecordingTokenizer(n_neurons=4, latent_dim=3).to("meta")

    result = compute_compression(spec, tok)  # type: ignore[arg-type]

    assert tok.received_device == torch.device("meta")
    assert result.latent_dim == 3


def test_compute_compression_cpu_tokenizer_unaffected(monkeypatch):
    """A CPU tokenizer still receives a CPU input — the device move is a no-op."""
    _patch_dummy_input(monkeypatch, n_neurons=4)
    spec = _SpecStub(
        latent_type=LatentType.CONTINUOUS,
        bin_size_ms=1.0,
        effective_window_length_seconds=0.2,
    )
    tok = _DeviceRecordingTokenizer(n_neurons=4, latent_dim=3)  # stays on cpu

    result = compute_compression(spec, tok)  # type: ignore[arg-type]

    assert tok.received_device == torch.device("cpu")
    assert result.latent_dim == 3


# ── Variable-window determinism ────────────────────────────────────────────────
# These use a REAL spec and a real make_dummy_input: the defect lived in the seam
# between the two, so a mocked dummy input cannot see it.


class _PatchTokenizer(torch.nn.Module):
    """Emits one token per 100 ms patch and one latent per input unit.

    That is the shape every shipping tokenizer has — ``t_tokens`` tracks the length of
    the input it is handed — which is what makes ``output_bits`` follow the dummy
    window while ``input_bits`` follows the spec's resolved window.
    """

    def __init__(self, latent_dim: int = 8, patch_bins: int = 100) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.patch_bins = patch_bins
        self.seen_bins: list[int] = []

    def encode(self, x):  # type: ignore[no-untyped-def]
        n_bins, n_units = int(x.spike_matrix.shape[1]), int(x.spike_matrix.shape[2])
        self.seen_bins.append(n_bins)
        return torch.zeros(1, max(1, n_bins // self.patch_bins), n_units, self.latent_dim)


def _dist_window_spec(low: float = 1.0, high: float = 5.0):
    return IdentityTokenizerSpec(
        n_neurons=32,
        window_length_seconds=UniformDist(low=low, high=high),
        bin_size_ms=1.0,
        vocab_size=256,  # one log2(256) = 8-bit code per [t, u] position
    )


def test_compression_rate_is_identical_across_repeated_calls_for_a_dist_window():
    """``compression_rate`` is reported as an architectural constant, so it must not be
    a readout of one random draw.

    ``input_bits`` resolves the window via ``effective_window_length_seconds``
    (``Dist.max``) while ``make_dummy_input`` sampled it, so numerator and denominator
    described different-length windows. That is why the 04e/07a replicate pair — with
    byte-identical configs — scored 0.7512 vs 0.4650.
    """
    spec = _dist_window_spec()
    tok = _PatchTokenizer()

    rates = [compute_compression(spec, tok).compression_rate for _ in range(5)]

    assert len(set(rates)) == 1, f"compression_rate varies across calls: {rates}"


def test_compression_builds_the_dummy_input_at_the_effective_window():
    """The dummy input is built at ``Dist.max``, the same window the numerator uses."""
    spec = _dist_window_spec(low=1.0, high=5.0)
    tok = _PatchTokenizer()

    result = compute_compression(spec, tok)

    assert tok.seen_bins == [5000]  # 5.0 s @ 1 ms bins, never a 1–5 s draw
    assert result.t_tokens == 50  # 5000 bins / 100-bin patches
    # input_bits = log2(1+1) * 5000 bins * 32 ; output_bits = 50 * 32 * (1 code * log2(256))
    assert result.input_bits == pytest.approx(160_000.0)
    assert result.output_bits == pytest.approx(12_800.0)
    assert result.compression_rate == pytest.approx(160_000.0 / 12_800.0)


def test_compression_fixed_window_spec_is_unchanged():
    """A float window is never sampled, so pinning the dummy window must leave the
    numbers of non-Dist specs untouched."""
    spec = IdentityTokenizerSpec(
        n_neurons=32,
        window_length_seconds=0.2,
        bin_size_ms=1.0,
        vocab_size=256,
    )
    tok = _PatchTokenizer()

    result = compute_compression(spec, tok)

    assert tok.seen_bins == [200]
    assert result.n_neurons == 32
    assert result.t_tokens == 2
    assert result.input_bits == pytest.approx(6400.0)  # log2(2) * 200 bins * 32
    assert result.output_bits == pytest.approx(512.0)  # 2 * 32 * (1 code * 8 bits)
    assert result.compression_rate == pytest.approx(6400.0 / 512.0)
    # latent_dim_reduction is deliberately untouched: both of its terms come from the
    # same dummy tensor, so it was internally consistent all along.
    assert result.input_elements == 6400  # 1 * 200 * 32
    assert result.output_elements == 512  # 1 * 2 * 32 * 8
    assert result.latent_dim_reduction == pytest.approx(12.5)


def test_compression_rate_is_invariant_to_the_window_when_the_token_rate_is_fixed():
    """Two specs differing *only* in window length score the same rate.

    With a fixed token rate (one token per 100 ms) ``t_tokens`` scales with the window,
    so the window cancels out of ``input_bits / output_bits``, leaving
    ``100 / bits_per_token`` — here 100/8 = 12.5.

    Any spread in the rate across specs with a fixed token rate would therefore be an
    artefact of the dummy window draw rather than of the tokenizer.
    """
    tok = _PatchTokenizer()

    rates = [
        compute_compression(_dist_window_spec(low=0.5, high=high), tok).compression_rate
        for high in (2.0, 5.0, 9.0)
    ]

    assert rates == [pytest.approx(100.0 / 8)] * 3, rates


class _FixedUnitTokenizer(torch.nn.Module):
    """Emits a fixed number of latent units, unlike the input's neuron count."""

    def __init__(self, u_units: int) -> None:
        super().__init__()
        self.u_units = u_units

    def encode(self, x):  # type: ignore[no-untyped-def]
        return torch.zeros(1, 4, self.u_units, 8)


def test_compression_warns_when_a_sampled_neuron_count_cannot_cancel(caplog):
    """``n_neurons`` is still drawn by make_dummy_input when ``spec.n_neurons`` is a
    Dist. It cancels out of the ratio only because ``u_units == n_neurons`` for every
    shipping tokenizer (``input_bits`` ∝ n_neurons, ``output_bits`` ∝ u_units). If a
    tokenizer breaks that, the rate is a random variable again — say so rather than
    assume the cancellation holds."""
    spec = IdentityTokenizerSpec(
        n_neurons=UniformDist(low=8, high=64, return_int=True),
        window_length_seconds=0.2,
        bin_size_ms=1.0,
        vocab_size=256,
    )
    tok = _FixedUnitTokenizer(u_units=7)

    with caplog.at_level(logging.WARNING):
        compute_compression(spec, tok)

    assert "u_units" in caplog.text
    assert "compression_rate" in caplog.text


def test_compression_does_not_warn_when_the_neuron_count_cancels(caplog):
    """The shipping case: u_units == n_neurons, so the draw cancels and is silent."""
    spec = IdentityTokenizerSpec(
        n_neurons=UniformDist(low=8, high=64, return_int=True),
        window_length_seconds=0.2,
        bin_size_ms=1.0,
        vocab_size=256,
    )
    tok = _PatchTokenizer()  # u_units == the input's unit count

    with caplog.at_level(logging.WARNING):
        compute_compression(spec, tok)

    assert "u_units" not in caplog.text


# ── Input bits are counted per bin ─────────────────────────────────────────────


@pytest.mark.parametrize("n_neurons", [1, 7, 32])
def test_input_bits_are_per_bin_count_entropy_times_number_of_bins(n_neurons):
    """A 10 ms bin holds a count in 0..10, i.e. log2(11) bits; 1 s holds 100 such bins.

    The old formula multiplied the per-bin entropy by the number of *milliseconds*, so
    every bin_size_ms != 1 tokenizer had its input bits (and rate) inflated by bin_size_ms.
    """
    spec = IdentityTokenizerSpec(
        n_neurons=n_neurons,
        window_length_seconds=1.0,
        bin_size_ms=10.0,
        vocab_size=256,
    )
    tok = _PatchTokenizer(patch_bins=10)

    result = compute_compression(spec, tok)

    assert tok.seen_bins == [100]
    assert result.bits_per_neuron_per_window == pytest.approx(math.log2(11) * 100)
    assert result.input_bits == pytest.approx(math.log2(11) * 100 * n_neurons)


@pytest.mark.parametrize(
    ("window_s", "bin_ms", "dummy_bins"),
    [(1.0, 3.0, 333), (2.01, 1.0, 2009)],  # int()-floored: 333.33 and 2009.9999999999998
)
def test_input_bits_count_the_bins_the_dummy_input_really_has(window_s, bin_ms, dummy_bins):
    """Numerator and denominator must describe the same window: the input side counts
    the time bins of the dummy input the tokenizer encoded, not ``window_ms / bin_ms``."""
    spec = IdentityTokenizerSpec(
        n_neurons=2, window_length_seconds=window_s, bin_size_ms=bin_ms, vocab_size=256
    )
    tok = _PatchTokenizer()

    result = compute_compression(spec, tok)

    assert tok.seen_bins == [dummy_bins]
    assert result.bits_per_neuron_per_window == pytest.approx(math.log2(bin_ms + 1) * dummy_bins)
    assert result.input_bits == pytest.approx(math.log2(bin_ms + 1) * dummy_bins * 2)


@pytest.mark.parametrize("bin_ms", [0.5, 0.999, 0.0, -1.0])
def test_bins_shorter_than_1ms_are_rejected(monkeypatch, bin_ms):
    """The input convention assumes at most one spike per ms (a bin holds 0..bin_ms
    spikes), which has no meaning below 1 ms. Fail loudly instead of a made-up number."""
    _patch_dummy_input(monkeypatch, n_neurons=4)
    spec = _SpecStub(
        latent_type=LatentType.CONTINUOUS,
        bin_size_ms=bin_ms,
        effective_window_length_seconds=0.2,
    )

    with pytest.raises(ValueError, match="bin_size_ms"):
        compute_compression(spec, _make_tokenizer(torch.zeros((1, 5, 1, 2))))  # type: ignore[arg-type]


def test_input_bits_at_1ms_bins_are_one_bit_per_ms():
    """1 ms bins: one bit per bin and bin == ms, so the per-bin rule changes nothing."""
    spec = IdentityTokenizerSpec(
        n_neurons=5, window_length_seconds=2.5, bin_size_ms=1.0, vocab_size=256
    )

    result = compute_compression(spec, _PatchTokenizer())

    assert result.bits_per_neuron_per_window == pytest.approx(2500.0)
    assert result.input_bits == pytest.approx(2500.0 * 5)


# ── Neuron count for specs without a fixed n_neurons ──────────────────────────


def _unfixed_neuron_spec():
    # n_neurons=False: the spec takes any population size (e.g. the LDNS-Perceiver).
    return IdentityTokenizerSpec(window_length_seconds=0.2, bin_size_ms=1.0, vocab_size=256)


def test_n_neurons_override_sets_the_dummy_population_and_is_recorded(caplog):
    spec = _unfixed_neuron_spec()
    assert spec.n_neurons is False
    tok = _PatchTokenizer()

    with caplog.at_level(logging.WARNING):
        result = compute_compression(spec, tok, n_neurons=256)

    assert caplog.text == ""  # the caller stated the count: nothing to warn about
    assert result.n_neurons == 256
    assert result.to_dict()["n_neurons"] == 256
    assert result.u_units == 256  # the tokenizer really saw 256 units
    assert result.input_bits == pytest.approx(200.0 * 256)  # log2(2) * 200 bins * 256


def test_n_neurons_override_changes_input_bits_for_a_spec_without_fixed_neurons():
    spec = _unfixed_neuron_spec()

    small = compute_compression(spec, _PatchTokenizer(), n_neurons=16)
    large = compute_compression(spec, _PatchTokenizer(), n_neurons=256)

    assert large.input_bits == pytest.approx(16 * small.input_bits)
    assert (small.n_neurons, large.n_neurons) == (16, 256)


def test_unfixed_neuron_count_without_override_warns_and_names_the_count(caplog):
    spec = _unfixed_neuron_spec()

    with caplog.at_level(logging.WARNING):
        result = compute_compression(spec, _PatchTokenizer())

    assert "n_neurons" in caplog.text
    assert str(result.n_neurons) in caplog.text


def test_fixed_neuron_spec_without_override_does_not_warn_about_neurons(caplog):
    spec = IdentityTokenizerSpec(
        n_neurons=32, window_length_seconds=0.2, bin_size_ms=1.0, vocab_size=256
    )

    with caplog.at_level(logging.WARNING):
        result = compute_compression(spec, _PatchTokenizer())

    assert result.n_neurons == 32
    assert "n_neurons" not in caplog.text


def test_n_neurons_override_of_a_fixed_neuron_spec_is_ignored_with_a_warning(caplog):
    """A spec with a fixed int neuron count is built for that count (for compress-n
    tokenizers it sets U); the override is ignored and the record keeps the spec's."""
    spec = IdentityTokenizerSpec(
        n_neurons=32, window_length_seconds=0.2, bin_size_ms=1.0, vocab_size=256
    )
    tok = _PatchTokenizer()

    with caplog.at_level(logging.WARNING):
        result = compute_compression(spec, tok, n_neurons=64)

    assert result.n_neurons == 32
    assert result.u_units == 32
    assert "ignored" in caplog.text


def test_n_neurons_override_equal_to_the_fixed_count_is_silent(caplog):
    spec = IdentityTokenizerSpec(
        n_neurons=32, window_length_seconds=0.2, bin_size_ms=1.0, vocab_size=256
    )

    with caplog.at_level(logging.WARNING):
        result = compute_compression(spec, _PatchTokenizer(), n_neurons=32)

    assert result.n_neurons == 32
    assert caplog.text == ""


def _dist_neuron_spec():
    return IdentityTokenizerSpec(
        n_neurons=UniformDist(low=10, high=20, return_int=True),
        window_length_seconds=0.2,
        bin_size_ms=1.0,
        vocab_size=256,
    )


def test_n_neurons_override_inside_a_dist_range_is_used():
    result = compute_compression(_dist_neuron_spec(), _PatchTokenizer(), n_neurons=20)

    assert result.n_neurons == 20


@pytest.mark.parametrize("n_neurons", [9, 21, 500])
def test_n_neurons_override_outside_a_dist_range_raises(n_neurons):
    with pytest.raises(ValueError, match="outside"):
        compute_compression(_dist_neuron_spec(), _PatchTokenizer(), n_neurons=n_neurons)


@pytest.mark.parametrize("n_neurons", [0, -3])
def test_n_neurons_override_must_be_positive(n_neurons):
    with pytest.raises(ValueError, match="n_neurons"):
        compute_compression(_unfixed_neuron_spec(), _PatchTokenizer(), n_neurons=n_neurons)


# ── encode() outputs with fewer than 4 dims ───────────────────────────────────


def test_real_identity_tokenizer_counts_every_token_index():
    """IdentityTokenizer.encode returns ``[B, T]`` indices, one per time bin. Each is a
    code of log2(vocab) bits, so 200 bins x 10 bits = 2000 output bits (rate 0.4).

    Padding the missing dims at the FRONT put T into the latent axis, which the per-code
    rule does not count: 10 output bits, rate 80, with no warning.
    """
    spec = IdentityTokenizerSpec(
        n_neurons=4, window_length_seconds=0.2, bin_size_ms=1.0, vocab_size=1024
    )
    tok = IdentityTokenizer(spec)

    result = compute_compression(spec, tok)

    assert (result.t_tokens, result.u_units, result.latent_dim) == (200, 1, 1)
    assert result.output_bits == pytest.approx(200 * 10.0)
    assert result.input_bits == pytest.approx(800.0)  # 200 bins * 4 neurons * 1 bit
    assert result.compression_rate == pytest.approx(0.4)


@pytest.mark.parametrize(
    ("shape", "expected_tu"),
    [((1, 5), (5, 1)), ((1, 5, 3), (5, 3)), ((1,), (1, 1))],
)
def test_discrete_short_latents_pad_the_trailing_dims(monkeypatch, shape, expected_tu):
    """A discrete ``[B, T]`` / ``[B, T, U]`` index tensor has no latent axis: the missing
    dims are U and D, so every given position counts as one token."""
    _patch_dummy_input(monkeypatch, n_neurons=4)
    spec = _SpecStub(
        latent_type=LatentType.DISCRETE,
        bin_size_ms=1.0,
        effective_window_length_seconds=0.2,
        _vocab_size=1024,
    )
    tok = _make_tokenizer(torch.zeros(shape, dtype=torch.long))

    result = compute_compression(spec, tok)  # type: ignore[arg-type]

    assert (result.t_tokens, result.u_units) == expected_tu
    assert result.latent_dim == 1
    assert result.output_bits == pytest.approx(expected_tu[0] * expected_tu[1] * 10.0)
