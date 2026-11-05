"""Compression metrics for tokenizers.

Computes input→output bit and element ratios for a single tokenizer using a
dummy input that matches its spec. Designed to be cheap enough to run once
at the start of every eval job.

``compression_rate = input_bits / output_bits``. The two sides use these conventions.

Input bits (count entropy per bin):

- A bin of ``bin_size_ms`` holds a spike count in ``0..bin_size_ms`` (at most one
  spike per ms), so it carries ``log2(bin_size_ms + 1)`` bits.
- ``input_bits = log2(bin_size_ms + 1) * (window_ms / bin_size_ms) * n_neurons``.
  That is the per-bin entropy times the number of bins times the neuron count.
  ``bits_per_neuron_per_window`` is the same product without ``n_neurons``.
  The number of bins is the time-axis length of the dummy input the tokenizer encodes
  (``floor(window_ms / bin_size_ms)``), so both sides describe the same window.
- Bins shorter than 1 ms raise ``ValueError``: "at most one spike per ms" has no
  meaning there.
- At 1 ms bins this is 1 bit per ms. A coarser bin carries fewer bits per ms
  (10 ms: ``log2(11) / 10`` = 0.35). Unbinned specs (``bin_size_ms=False``) are
  counted at 1 ms.
- ``n_neurons`` is the dummy input's neuron count and is recorded in the result.
  Specs without a fixed count (``n_neurons=False``) should get it from the
  caller, via ``compute_compression(..., n_neurons=...)``.

Output bits (information per latent position):

- ``encode()`` returns ``[B, T, U, D]``. Each of the ``T * U`` positions carries
  ``bits_per_token`` bits, and ``output_bits = T * U * bits_per_token``.
- Discrete: ``bits_per_token = codes_per_token * log2(vocab_size)``.
  ``vocab_size`` is the size of ONE code (``spec.get_vocab_size()``).
  ``codes_per_token`` is the number of such codes per position
  (``spec.get_codes_per_token()``). This is 1 for FSQ/BSQ/LFQ/VQ and
  ``num_quantizers`` for residual quantizers.
  The D values of a position are NOT independent symbols. An FSQ position stores
  one code as ``D`` floats, so ``D`` does not enter the count.
  ``bits_per_dim`` keeps its name but means bits per code
  (``log2(vocab_size)``) here.
- Continuous: each of the ``D`` values is a bf16 number (16 bits), so
  ``bits_per_dim = 16``, ``codes_per_token = D`` and ``bits_per_token = 16 * D``.
  If a future spec ships fp32 latents, lift this from the spec instead of the
  constant.
- Unknowns give NaN, never a guessed count. A discrete spec whose
  ``get_vocab_size()`` or ``get_codes_per_token()`` raises ``NotImplementedError``
  gets NaN ``bits_per_token``. The NaN propagates into ``output_bits`` and
  ``compression_rate``, and a warning names the unknown.
  ``latent_dim_reduction`` (element-only) stays real.

In every case ``bits_per_token == codes_per_token * bits_per_dim``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from typing import Any

import torch

from ntt.core.components.dist import Dist
from ntt.core.components.types import LatentType
from ntt.core.model.tokenizer.base import BaseTokenizer, TokenizerSpec
from ntt.core.model.tokenizer.validation import make_dummy_input

log = logging.getLogger(__name__)

# The bit-count convention of every record this module writes (``CompressionResult.convention``):
# output bits per CODE (not per latent value), input bits per BIN (not per ms).
COMPRESSION_CONVENTION = "per_code_per_bin"


@dataclass(frozen=True)
class CompressionResult:
    """Single-tokenizer compression summary.

    All fields are JSON-serialisable. ``bits_per_dim``, ``codes_per_token``,
    ``bits_per_token``, ``output_bits`` and ``compression_rate`` may be ``math.nan``
    when the spec is discrete but its vocabulary size or code count is unknown.
    The module docstring gives the conventions.

    Fields:
        bits_per_dim: Bits per symbol. ``log2(vocab_size)`` (bits per code) for
            discrete specs, 16 for continuous ones.
        codes_per_token: Symbols per ``[t, u]`` latent position. Integer-valued
            float: codes per position for discrete specs, ``latent_dim`` for
            continuous ones.
        bits_per_token: ``codes_per_token * bits_per_dim``. Bits per latent position.
        n_neurons: Neuron count of the dummy input that both sides were measured on.
        convention: ``COMPRESSION_CONVENTION``, the bit-count convention of this record.
    """

    input_bits: float
    output_bits: float
    compression_rate: float
    input_elements: int
    output_elements: int
    latent_dim_reduction: float
    bits_per_dim: float
    codes_per_token: float
    bits_per_token: float
    bits_per_neuron_per_window: float
    n_neurons: int
    t_tokens: int
    u_units: int
    latent_dim: int
    convention: str = COMPRESSION_CONVENTION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_token_bits(spec: TokenizerSpec, latent_dim: int) -> tuple[float, float, float]:
    """Return ``(bits_per_dim, codes_per_token, bits_per_token)`` for one latent position."""
    if spec.latent_type != LatentType.DISCRETE:
        return 16.0, float(latent_dim), 16.0 * latent_dim
    try:
        bits_per_code = math.log2(spec.get_vocab_size())
    except NotImplementedError as exc:
        log.warning(
            "compression: vocabulary size unknown for discrete spec %s (%s); output_bits "
            "and compression_rate are NaN.",
            type(spec).__name__,
            exc,
        )
        bits_per_code = math.nan
    try:
        codes_per_token = float(spec.get_codes_per_token())
    except NotImplementedError as exc:
        log.warning(
            "compression: codes per token unknown for discrete spec %s (%s); output_bits "
            "and compression_rate are NaN. Implement get_codes_per_token() on the spec.",
            type(spec).__name__,
            exc,
        )
        codes_per_token = math.nan
    return bits_per_code, codes_per_token, codes_per_token * bits_per_code


def _input_n_neurons(dummy_input: Any) -> tuple[int, tuple[int, ...], bool]:
    """Return (n_neurons, shape, is_binned) from a dummy input's data tensor.

    ``is_binned`` is True when ``shape`` is a real ``[B, T_bins, N]`` matrix.
    """
    if hasattr(dummy_input, "spike_matrix"):
        shape = tuple(dummy_input.spike_matrix.shape)
    elif hasattr(dummy_input, "rate_matrix"):
        shape = tuple(dummy_input.rate_matrix.shape)
    else:
        # Unknown shape — fall back to 1 to avoid div-by-zero downstream.
        return 1, (1, 1), False
    return shape[-1], shape, len(shape) >= 2


def _infer_tokenizer_device(tokenizer: BaseTokenizer) -> torch.device:
    """Best-effort device of a tokenizer's tensors (params → buffers → cpu).

    Defensive against non-``nn.Module`` stand-ins (e.g. test mocks whose
    ``parameters()`` isn't iterable): any failure falls back to CPU, which
    makes the subsequent device move a harmless no-op.
    """
    import itertools

    try:
        for t in itertools.chain(tokenizer.parameters(), tokenizer.buffers()):
            if isinstance(t, torch.Tensor):
                return t.device
    except (TypeError, AttributeError):
        pass
    return torch.device("cpu")


def _latents_from_encode(out: Any) -> torch.Tensor:
    if isinstance(out, dict):
        latents = out.get("latents")
        if latents is None:
            latents = out.get("tokens")
        if latents is None:
            raise ValueError("tokenizer.encode() returned a dict without 'latents' or 'tokens'")
        return latents
    return out


def _resolve_n_neurons_override(spec: TokenizerSpec, n_neurons: int | None) -> int | None:
    """The neuron count to build the dummy input with (``None``: the spec decides).

    See ``compute_compression``'s ``n_neurons`` argument for the rules.
    """
    if n_neurons is None:
        return None
    if isinstance(n_neurons, bool) or not isinstance(n_neurons, int) or n_neurons < 1:
        raise ValueError(f"compression: n_neurons must be an int >= 1, got {n_neurons!r}")
    fixed = spec.n_neurons
    if isinstance(fixed, Dist):
        low, high = fixed.min, fixed.max
        if not low <= n_neurons <= high:
            raise ValueError(
                f"compression: n_neurons={n_neurons} is outside the spec's n_neurons range "
                f"[{low}, {high}]"
            )
        return n_neurons
    if fixed is False:  # no fixed count (False is an int too, so check it first)
        return n_neurons
    if n_neurons != fixed:
        log.warning(
            "compression: n_neurons=%d ignored: spec %s fixes n_neurons=%d, the count its "
            "tokenizer is built for.",
            n_neurons,
            type(spec).__name__,
            fixed,
        )
    return None


def compute_compression(
    spec: TokenizerSpec,
    tokenizer: BaseTokenizer,
    *,
    n_neurons: int | None = None,
) -> CompressionResult:
    """Compute compression metrics for one tokenizer using a dummy forward.

    The dummy input is generated from the spec; no real data is required.

    Args:
        spec: The tokenizer's spec.
        tokenizer: The tokenizer, already on its eval device.
        n_neurons: Neuron count of the dummy input, for specs WITHOUT a fixed int count.
            Must be >= 1. Used when the spec has ``n_neurons=False`` (no count; the
            rate depends on it) or a ``Dist`` (must lie in ``[Dist.min, Dist.max]``,
            else ``ValueError``). Ignored, with a warning, when the spec fixes an int
            count: the tokenizer is built for that count. ``None`` lets the spec decide
            (its fixed count, a draw from its ``Dist``, or the dummy default of 8).
            The count used is recorded as ``result.n_neurons``.
    """
    # One window length on BOTH sides of the ratio. ``input_bits`` resolves the spec's
    # window deterministically (a Dist becomes its max), so the dummy input that
    # ``output_bits`` is measured on must be built at that same length. Letting
    # make_dummy_input sample it instead made numerator and denominator describe
    # different-length windows, and ``compression_rate`` a readout of one random draw:
    # the 04e/07a replicate pair, with byte-identical configs, scored 0.7512 vs 0.4650.
    window_length_seconds = float(spec.effective_window_length_seconds)
    # spec.bin_size_ms is `float | Literal[False]`; False is the "unbinned spike
    # times" sentinel, counted at 1 ms. Below 1 ms the convention (at most one spike
    # per ms, so a bin holds 0..bin_size_ms spikes) has no meaning: refuse.
    bin_size_ms = 1.0 if spec.bin_size_ms is False else float(spec.bin_size_ms)
    if bin_size_ms < 1.0:
        raise ValueError(
            f"compression: bin_size_ms must be >= 1 ms (the input-bits convention counts "
            f"at most one spike per ms), got {bin_size_ms}"
        )
    requested_n_neurons = n_neurons
    n_neurons = _resolve_n_neurons_override(spec, n_neurons)
    dummy_input = make_dummy_input(
        spec, window_length_seconds=window_length_seconds, n_neurons=n_neurons
    )
    # The tokenizer may live on GPU (it is moved to device before eval); the
    # dummy input is built on CPU. Move it onto the tokenizer's device so
    # device-sensitive ops (e.g. the compressed-sensing sensing-matrix matmul)
    # don't hit a CPU-vs-CUDA mismatch. Lightning's move_data_to_device walks
    # the NamedTuple input and relocates every tensor.
    from lightning.fabric.utilities.apply_func import move_data_to_device

    dummy_input = move_data_to_device(dummy_input, _infer_tokenizer_device(tokenizer))
    # From here on, n_neurons is the count the dummy input really has.
    n_neurons, input_shape, is_binned = _input_n_neurons(dummy_input)
    input_elements = 1
    for d in input_shape:
        input_elements *= int(d)

    if requested_n_neurons is None and spec.n_neurons is False:
        log.warning(
            "compression: spec %s has no fixed n_neurons; input_bits and compression_rate "
            "use the dummy input's n_neurons=%d. Pass n_neurons= to compute_compression "
            "to set the population size.",
            type(spec).__name__,
            n_neurons,
        )

    # Per-bin count entropy times the number of bins (module docstring). For a binned
    # dummy input, count the time bins it really has (make_dummy_input floors
    # window_ms / bin_ms), so input_bits describe the window the tokenizer encoded.
    n_bins = float(input_shape[-2]) if is_binned else window_length_seconds * 1000.0 / bin_size_ms
    bits_per_neuron_per_window = math.log2(bin_size_ms + 1) * n_bins
    input_bits = bits_per_neuron_per_window * n_neurons

    with torch.no_grad():
        latents = _latents_from_encode(tokenizer.encode(dummy_input))

    output_elements = 1
    for d in latents.shape:
        output_elements *= int(d)

    # Most tokenizers emit [B, T_tokens, U, latent_dim]; some emit fewer dims (the identity
    # tokenizer returns [B, T]), which are padded to four:
    # - Discrete: pad at the END. A ``[B, T]`` or ``[B, T, U]`` index tensor (e.g. the
    #   identity tokenizer's) has no latent axis, and every given position is a token.
    #   Front-padding would move T into latent_dim, which the per-code rule does not count.
    # - Continuous: pad at the front (unchanged). Every value is counted, so the
    #   split only affects the t_tokens / u_units / latent_dim fields.
    if latents.ndim < 4:
        pad = (1,) * (4 - latents.ndim)
        if spec.latent_type == LatentType.DISCRETE:
            shape4 = tuple(latents.shape) + pad
        else:
            shape4 = pad + tuple(latents.shape)
    else:
        shape4 = tuple(latents.shape[:4])
    _, t_tokens, u_units, latent_dim = shape4

    # ``n_neurons`` is still drawn by make_dummy_input when ``spec.n_neurons`` is a Dist.
    # It cancels out of compression_rate only because ``u_units == n_neurons`` for every
    # shipping tokenizer (input_bits ∝ n_neurons, output_bits ∝ u_units). Don't assume
    # that silently: if a tokenizer breaks it, the rate is a random variable again.
    if requested_n_neurons is None and isinstance(spec.n_neurons, Dist) and u_units != n_neurons:
        log.warning(
            "compression_rate is NOT deterministic for this spec: n_neurons is a Dist "
            "(drawn as %d for the dummy input) and u_units=%d != n_neurons, so the draw "
            "does not cancel between input_bits and output_bits. Resolve spec.n_neurons "
            "to a fixed value before quoting compression_rate.",
            n_neurons,
            u_units,
        )

    bits_per_dim, codes_per_token, bits_per_token = _resolve_token_bits(spec, latent_dim)
    output_bits = t_tokens * u_units * bits_per_token

    if math.isnan(output_bits) or output_bits == 0.0:
        compression_rate = math.nan
    else:
        compression_rate = input_bits / output_bits
    latent_dim_reduction = input_elements / output_elements if output_elements > 0 else math.nan

    return CompressionResult(
        input_bits=float(input_bits),
        output_bits=float(output_bits),
        compression_rate=float(compression_rate),
        input_elements=int(input_elements),
        output_elements=int(output_elements),
        latent_dim_reduction=float(latent_dim_reduction),
        bits_per_dim=float(bits_per_dim),
        codes_per_token=float(codes_per_token),
        bits_per_token=float(bits_per_token),
        bits_per_neuron_per_window=float(bits_per_neuron_per_window),
        n_neurons=int(n_neurons),
        t_tokens=int(t_tokens),
        u_units=int(u_units),
        latent_dim=int(latent_dim),
    )
