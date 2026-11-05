"""Inference-cost metrics (FLOPs, latency, params) for tokenizers.

Computes encoder/decoder FLOPs (via ``torch.utils.flop_counter.FlopCounterMode``),
wall-clock latency, and parameter counts for a single tokenizer using a dummy
input that matches its spec — mirroring the
``src/ntt/evaluation/compression.py::compute_compression`` pattern, but for
compute cost rather than compression ratio.

Spec provenance (hard rule): every spec fed into this module must originate
from a real checkpoint's ``hyper_parameters['tokenizer_spec']`` — never a
YAML default. ``resolve_spec_from_checkpoint`` is the sanctioned entry point;
it loads the checkpoint, extracts the real spec, and applies a narrow set of
overrides (``window_length_seconds``, ``n_neurons``) via full pydantic
re-validation (``spec.__class__(**{**spec.model_dump(), **overrides})``) so
every other field — ``bin_size_ms``, ``token_bin_size_ms``, bottleneck
config, everything — carries through untouched.

FLOPs coverage caveat: ``FlopCounterMode`` only counts matmul/conv/attention
style ops it can hook via ``__torch_dispatch__``. It does **not** count
``torch.fft.rfft``/``irfft`` (used by LDNS's S4 kernel) or ``torch.cdist``
(used by some VQ-family bottlenecks' nearest-neighbor search — e.g. SimVQ's
``vector_quantize_pytorch/sim_vq.py``, reachable from more than one family).
``_FFTFlopTracker`` and ``_CdistFlopTracker`` trace the real
``torch.fft.rfft``/``irfft``/``torch.cdist`` calls a call actually dispatches
and report the total as a manual supplement rather than baking a guess into
the counted total. Both are applied unconditionally around every measured
encoder AND decoder call, for every family — each tracker starts at 0 FLOPs
and only accrues if the real forward path dispatches the traced op, so a
family that never calls ``torch.fft.*``/``torch.cdist`` simply measures a 0
supplement (a traced fact), and there is no per-family allowlist to keep in
sync as new families or bottleneck variants are added.
"""

from __future__ import annotations

import contextlib
import functools
import itertools
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch.utils.flop_counter import FlopCounterMode

from ntt.core.components.dist import Dist
from ntt.core.model.tokenizer import _extract_tokenizer_state_dict
from ntt.core.model.tokenizer.base import BaseTokenizer, TokenizerSpec
from ntt.core.model.tokenizer.validation import make_dummy_input
from ntt.evaluation.compression import _latents_from_encode

# rfft/irfft FLOPs approximation: a real-valued length-N FFT costs about half
# of a complex FFT of the same length (5*N*log2(N)), i.e. ~2.5*N*log2(N).
# This is an order-of-magnitude estimate, not exact.
_REAL_FFT_FLOPS_PER_ELEMENT_LOG2N = 2.5

# Buffers that exist on a tokenizer module purely for fitting/EMA-training
# bookkeeping and are never read by encode()/decode() at inference time
# (traced/confirmed per family below, not assumed) -- excluded from
# `inference_params` in `family_extra_config`, which total_params/
# learnable_params otherwise conflate them into. Matched against a buffer's
# leaf name (the part after the last '.') so nested submodule paths (e.g.
# "vq._ema_count") still match. Families not listed here have no known
# inference-irrelevant buffers, so inference_params == total_params for them.
#
# Keys are lowercase and looked up via family.lower() (see
# `compute_inference_cost` below) -- this module's own docstring
# (scripts/benchmark_inference_cost.py) documents the manifest's family
# column using "PCA" (uppercase) while "wanx_A"/"calm"/"cosmos" use other
# casings; a case-sensitive dict would match none of them correctly for a
# caller that followed that documented convention, silently reporting
# inference_params == total_params (no exclusion, no error) for a
# mis-cased family string. The lookup is therefore case-insensitive on both
# sides rather than requiring callers to spell the family exactly.
_INFERENCE_IRRELEVANT_BUFFER_NAMES: dict[str, frozenset[str]] = {
    # pca_baseline: _n_seen/_sum_T/_sum_outer_TT are running-covariance
    # accumulators used only inside .fit(); `fitted` is a control-flow flag
    # with no computational contribution. encode()/decode() (pca_baseline.py)
    # only ever read W/mu (+ the required-by-Lightning `_dummy` nn.Parameter,
    # which IS in the graph via `recon = recon + 0.0 * self._dummy.sum()`
    # and is therefore NOT excluded here).
    "pca": frozenset({"_n_seen", "_sum_T", "_sum_outer_TT", "fitted"}),
    # calm (VQ bottleneck): encode() reaches VQBottleneck.forward(), but the
    # harness unconditionally puts the tokenizer in evaluation mode. The EMA
    # mutations are confined to forward()'s `if self.training:` branch
    # (calm/vq.py), so these training-only buffers are inference-irrelevant.
    "calm": frozenset(
        {"_ema_count", "_ema_weight", "_usage_count", "_feature_buffer", "_buffer_ptr", "_eye"}
    ),
    # wanx_A (published as conv-*; FSQ bottleneck only -- bsq/vqpt variants not traced, so this
    # only fires when the FSQuantizer submodule is actually present):
    # implicit_codebook is precomputed once at __init__ but never read by
    # FSQuantizer.forward()/quantize()/indices_to_codes() (bottleneck.py).
    "wanx_a": frozenset({"implicit_codebook"}),
}


@dataclass(frozen=True)
class InferenceCostResult:
    """Single-tokenizer inference-cost summary (FLOPs, latency, params).

    All fields are JSON-serialisable. ``*_flops_manual_supplement`` fields
    are 0 for families where ``FlopCounterMode`` coverage is adequate (see
    module docstring); ``family_extra_config`` carries family-specific
    diagnostic fields that don't fit the fixed schema above.
    """

    family: str
    variant: str
    n_neurons: int
    window_length_seconds: float
    bin_size_ms: float
    batch_size: int
    native_or_forced: str
    total_params: int
    learnable_params: int
    encoder_flops_counted: int
    encoder_flops_manual_supplement: int
    encoder_latency_ms_mean: float
    encoder_latency_ms_std: float
    decoder_flops_counted: int
    decoder_flops_manual_supplement: int
    decoder_latency_ms_mean: float
    decoder_latency_ms_std: float
    shape_ok: bool
    shape_detail: str
    family_extra_config: dict

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _assert_no_dist_fields(spec: TokenizerSpec) -> None:
    """Assert no field on *spec* is (still) a ``Dist`` instance.

    Insurance against a spec whose ``n_neurons``/``window_length_seconds``/
    ``bin_size_ms`` (or any other field) is a distribution rather than a
    literal after the override step below. Real checkpoints seen so far have
    always carried literals, so this is not expected to fire.
    """
    for field_name in spec.__class__.model_fields:
        value = getattr(spec, field_name)
        if isinstance(value, Dist):
            raise AssertionError(
                f"{spec.__class__.__name__}.{field_name} is a Dist ({value!r}) "
                "after override — inference-cost benchmarking requires fully "
                "resolved (literal) specs."
            )


def _needs_token_bin_size_reset(spec_obj: TokenizerSpec, overrides: dict[str, Any]) -> bool:
    """True if *overrides* must force TokenizerSpec's token_bin_size_ms to recompute.

    Mirrors the exact predicate TokenizerSpec._set_default_token_bin_size uses
    for its window-derived branch (base.py lines ~196-198): compress_t=True
    and has_time_dim=False ("whole window -> one token"). Evaluated against
    *spec_obj*, the originally-loaded checkpoint spec, BEFORE overrides are
    applied -- compress_t/has_time_dim never change under override in
    practice, but this keeps the check anchored to the real spec's declared
    capabilities rather than to the caller's override dict.

    Returns False (no reset) for every other family:
      - compress_t=False: token_bin_size_ms = bin_size_ms, independent of
        window_length_seconds by construction (e.g. ldns, identity_tokenizer,
        perceiver, compressed_sensing per_time_bin mode).
      - compress_t=True, has_time_dim=True: ambiguous per the base validator;
        every such family instead sets token_bin_size_ms explicitly or via
        its own shadowing validator (wanneural, cosmos, calm, conv_vae,
        poisson_baseline, frequency_based_lut) -- none of these are
        window-derived by the base class's formula, so a window override
        must never perturb token_bin_size_ms for them.

    Also False if the caller already supplied an explicit token_bin_size_ms
    override themselves (that override must win untouched) or did not touch
    window_length_seconds at all (nothing to recompute).

    Known gaps, intentionally out of scope here:
      - ``FrequencyBasedLUTTokenizerSpec`` declares has_time_dim=True (so this
        predicate is False for it) but its own before-validator computes
        ``token_bin_size_ms = window_length_seconds * 1000.0`` via
        ``setdefault``, which is subject to the same staleness via a
        different code path. That path is not handled here.
      - A ``bin_size_ms`` override (rather than ``window_length_seconds``)
        would leave ``token_bin_size_ms`` stale for compress_t=False families
        (``token_bin_size_ms = bin_size_ms``, same short-circuit mechanism,
        different upstream field). No current caller overrides
        ``bin_size_ms``, so this is not handled — worth revisiting if one
        ever does.
    """
    return (
        "window_length_seconds" in overrides
        and "token_bin_size_ms" not in overrides
        and spec_obj.compress_t
        and not spec_obj.has_time_dim
    )


@functools.lru_cache(maxsize=1)
def _load_ckpt(ckpt_path: str) -> dict:
    """Cached ``torch.load`` shared by ``resolve_spec_from_checkpoint`` and
    ``load_tokenizer_weights_from_checkpoint``.

    A single manifest row calls the former once and the latter once per
    swept ``batch_size`` — without this, a row sweeping several batch sizes
    against a large checkpoint re-deserializes the same file from disk that
    many times. Safe to share: nothing downstream mutates entries of the
    returned dict (``load_state_dict`` copies values into existing
    parameters/buffers, it doesn't write back into the source dict), and
    the cache key is the checkpoint path string, so different rows still
    load their own files.

    ``maxsize=1`` deliberately: every call within a single manifest row
    shares that row's ``ckpt_path``, so a single-entry cache serves the
    row's own repeated batch-size sweep in full. A larger ``maxsize`` would
    let several full (potentially large) checkpoint state dicts pile up in
    RAM at once across a manifest sweep for no benefit under that pattern.
    The cost is paid only if a manifest interleaves rows for two different
    checkpoints instead of grouping them (nothing enforces grouping) — the
    interleaved row re-deserializes its checkpoint from disk on every
    switch instead of hitting a warm cache. That's a performance
    regression, not a correctness one (the returned dict is always the
    correct one for its ``ckpt_path``); revisit if that access pattern
    becomes common.
    """
    return torch.load(ckpt_path, map_location="cpu", weights_only=False)


def resolve_spec_from_checkpoint(
    ckpt_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> TokenizerSpec:
    """Load a checkpoint's real tokenizer spec and apply narrow overrides.

    Args:
        ckpt_path: Path to a Lightning ``.ckpt`` file.
        overrides: Fields to override on the resolved spec — intended for
            ``window_length_seconds``, ``n_neurons`` (and, in principle,
            other spec fields; ``batch_size`` is NOT a spec field and must
            never be passed here — it is a ``compute_inference_cost()``
            argument). Every other field on the real checkpoint's spec is
            carried through untouched.

    Returns:
        A fully-resolved ``TokenizerSpec`` instance — never a plain dict,
        never a YAML default — reconstructed via
        ``spec.__class__(**{**spec.model_dump(), **overrides})`` so that
        pydantic validators (e.g. ``token_bin_size_ms`` inference) re-run
        against the overridden values.

    Raises:
        ValueError: If the checkpoint has no ``tokenizer_spec`` in its
            ``hyper_parameters``.
        AssertionError: If any field on the reconstructed spec is a ``Dist``.

    Note:
        This resolves the *architecture* only. ``create_tokenizer(spec)``
        built from this spec still carries freshly (randomly) initialized
        weights — call ``load_tokenizer_weights_from_checkpoint`` afterward
        to load the checkpoint's real trained weights, which matters even
        for buffer-only families (e.g. a PCA baseline's unfitted buffers
        take a different, near-zero-FLOPs code path than its fitted ones).
    """
    overrides = overrides or {}

    ckpt = _load_ckpt(str(ckpt_path))
    hparams = ckpt.get("hyper_parameters", {})
    spec_obj = hparams.get("tokenizer_spec")
    if spec_obj is None:
        raise ValueError(f"No tokenizer_spec found in checkpoint hyper_parameters: {ckpt_path}")

    if not isinstance(spec_obj, TokenizerSpec):
        # Plain dict (or dict-like) rather than an already-instantiated spec —
        # validate it through the same TypeAdapter[TokenizerSpecKind] pattern
        # used by load_tokenizer_from_checkpoint_no_lightning.
        import ntt.contrib.tokenizer  # noqa: F401  # isort: skip — registers contrib tokenizers
        from ntt.contrib import TokenizerSpecKind  # isort: skip
        from pydantic import TypeAdapter  # isort: skip

        spec_dict = spec_obj
        if hasattr(spec_dict, "model_dump"):
            spec_dict = spec_dict.model_dump(by_alias=True, exclude_none=True)
        adapter: TypeAdapter[TokenizerSpecKind] = TypeAdapter(TokenizerSpecKind)
        spec_obj = cast(TokenizerSpec, adapter.validate_python(spec_dict))

    # Reconstruct via model_dump() + overrides (NOT model_copy(update=...),
    # which skips pydantic validators) so every non-overridden field carries
    # through untouched and validators re-run against the new values.
    dump = spec_obj.model_dump()

    # For "one token per window" specs (compress_t and not has_time_dim), the
    # base validator's _set_default_token_bin_size infers
    # token_bin_size_ms = window_length_seconds * 1000 and, once set, that
    # inferred value is indistinguishable from an explicitly-set one on
    # model_dump() (it was written via object.__setattr__, so it's still in
    # model_fields_set). The validator early-returns whenever
    # token_bin_size_ms is not None, so reconstructing with the dumped value
    # present skips re-inference entirely and silently carries over the
    # *old* window's token_bin_size_ms when window_length_seconds is
    # overridden. Clear it here so the validator re-derives it against the
    # new window. See _needs_token_bin_size_reset's docstring for the exact
    # predicate and its known out-of-scope gaps.
    if _needs_token_bin_size_reset(spec_obj, overrides):
        dump["token_bin_size_ms"] = None

    merged = {**dump, **overrides}
    resolved_spec = spec_obj.__class__(**merged)

    _assert_no_dist_fields(resolved_spec)
    return resolved_spec


def load_tokenizer_weights_from_checkpoint(
    ckpt_path: str | Path,
    tokenizer: BaseTokenizer,
) -> tuple[bool, str]:
    """Best-effort load of a checkpoint's real tokenizer weights into *tokenizer*.

    ``resolve_spec_from_checkpoint`` + ``create_tokenizer`` only reproduce the
    checkpoint's *architecture*; the resulting module carries freshly
    initialized (random) weights until this is called. That matters for
    buffer-bearing families in particular — a fitted PCA projection, for
    example, cannot be reproduced from the spec alone.

    Mirrors the state-dict extraction in
    ``load_tokenizer_from_checkpoint_no_lightning``
    (``src/ntt/core/model/tokenizer/__init__.py``): strips the ``"tokenizer."``
    prefix Lightning's ``state_dict`` wraps tokenizer weights in, falling back
    to the raw state dict if no such keys are found.

    A strict load is tried first; on failure (expected when a forced
    ``n_neurons``/``window_length_seconds`` override resizes weight-bearing
    tensors), a *shape-filtered* non-strict load is attempted instead: only
    entries whose shape matches the freshly-constructed module's current
    parameter/buffer shape are handed to ``load_state_dict(..., strict=False)``.
    This matters because ``strict=False`` only suppresses missing/unexpected
    **key** errors — a shape mismatch on a key present in both state dicts
    still raises unconditionally, which would otherwise make the "best effort"
    fallback load nothing at all. The outcome is always returned — never
    silently swallowed — so callers can flag "forced" rows as not-real-weights
    when a strict load fails.

    Returns:
        ``(weights_loaded, detail)``: ``weights_loaded`` is True only for a
        fully strict load (no missing/unexpected keys, no shape mismatch);
        ``detail`` is a human-readable description of what happened,
        including how many of the checkpoint's tensors were shape-compatible
        and actually loaded in the fallback case.
    """
    ckpt = _load_ckpt(str(ckpt_path))
    tokenizer_state = _extract_tokenizer_state_dict(ckpt)

    try:
        tokenizer.load_state_dict(tokenizer_state, strict=True)
        return True, "loaded (strict, no missing/unexpected keys, no shape mismatch)"
    except RuntimeError as e:
        model_sd = tokenizer.state_dict()
        compatible = {
            k: v
            for k, v in tokenizer_state.items()
            if k in model_sd and model_sd[k].shape == v.shape
        }
        incompatible_shape = [k for k in tokenizer_state if k in model_sd and k not in compatible]
        try:
            result = tokenizer.load_state_dict(compatible, strict=False)
            return False, (
                f"shape-filtered non-strict load after strict failure ({e}); "
                f"loaded {len(compatible)}/{len(tokenizer_state)} checkpoint tensors "
                f"(shape mismatch, excluded from load: {incompatible_shape}); "
                f"missing_keys={list(result.missing_keys)} "
                f"unexpected_keys={list(result.unexpected_keys)}"
            )
        except Exception as e2:
            return False, f"load_state_dict failed entirely: {type(e2).__name__}: {e2}"


def _time_calls(
    fn,
    device: torch.device,
    warmup: int,
    repeats: int,
) -> tuple[float, float]:
    """Warmup + timed loop; returns (mean_ms, std_ms)."""
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1 to compute a mean latency, got {repeats}")
    is_cuda = device.type == "cuda"

    with torch.no_grad():
        for _ in range(warmup):
            fn()
        if is_cuda:
            torch.cuda.synchronize()

        durations_ms: list[float] = []
        for _ in range(repeats):
            if is_cuda:
                torch.cuda.synchronize()
            start = time.perf_counter()
            fn()
            if is_cuda:
                torch.cuda.synchronize()
            durations_ms.append((time.perf_counter() - start) * 1000.0)

    mean_ms = statistics.mean(durations_ms)
    std_ms = statistics.pstdev(durations_ms) if len(durations_ms) > 1 else 0.0
    return mean_ms, std_ms


@contextlib.contextmanager
def _no_op_wanx_aux_loss():
    """Harness-only patch: elide wanx_A's training-only MMD/KL aux loss.

    wanx_A is the internal family name of the conv-* models published on the
    Hugging Face Hub.

    ``MMDVAEBottleneck.encode()`` and ``VAEBottleneck.encode()``
    (``contrib/tokenizer/wanneural/bottleneck.py``) unconditionally compute
    an ``aux_loss`` (MMD via an O(n^2) RBF kernel, or KL divergence) that
    ``WanNeural.encode()`` simply discards at inference time — it exists
    only to be summed into the training loss elsewhere. Left alone, that
    discarded-but-still-computed term inflates measured *latency* (though
    not counted FLOPs, since it's elementwise/broadcast compute invisible
    to ``FlopCounterMode``) for the ``mmd_vae``/``vae`` bottleneck variants.

    Rather than editing model source (which would need auditing every
    training-path caller of these two ``encode()`` methods), this patches
    the two specific expensive leaves — the RBF kernel and the KL
    divergence — to trivial no-ops, scoped strictly to this context and
    restored on exit. It is inert for every family/bottleneck that doesn't
    use these two classes, so it is applied unconditionally around every
    measured call rather than gated on ``family == "wanx_A"`` — the
    manifest's ``family`` column is free text a caller could mislabel, and
    gating on it would mean a mislabeled row silently measures a different
    thing than what gets reported.

    The import is optional: ``ntt.contrib.tokenizer.wanneural.bottleneck``
    transitively imports the whole contrib tokenizer package (see
    ``ntt/contrib/tokenizer/__init__.py``), which pulls in ``pytorch_wavelets``
    for cosmos (the wavelet-* models) — an optional dependency that can be
    absent in some environments. If the import fails, this becomes a true no-op (nothing
    patched) rather than taking down measurement for every other family
    (PCA, LDNS, cosmos, ...) that has nothing to do with wanneural.

    Yields a status string recorded in ``family_extra_config`` for
    provenance: ``"patched"`` (both classes patched for the scope of this
    call) or ``"import_unavailable"`` (wanneural.bottleneck could not be
    imported; no-op).
    """
    try:
        from ntt.contrib.tokenizer.wanneural.bottleneck import (
            DiagonalGaussianDistribution,
            MMDVAEBottleneck,
        )
    except ImportError:
        yield "import_unavailable"
        return

    orig_kernel = MMDVAEBottleneck._compute_kernel
    orig_kl = DiagonalGaussianDistribution.kl

    def _noop_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], y.shape[0], device=x.device, dtype=x.dtype)

    def _noop_kl(self, other=None, mask=None) -> torch.Tensor:
        return torch.zeros(1, device=self.parameters.device, dtype=self.parameters.dtype)

    MMDVAEBottleneck._compute_kernel = _noop_kernel
    DiagonalGaussianDistribution.kl = _noop_kl
    try:
        yield "patched"
    finally:
        MMDVAEBottleneck._compute_kernel = orig_kernel
        DiagonalGaussianDistribution.kl = orig_kl


class _FFTFlopTracker:
    """Traces real ``torch.fft.rfft``/``irfft`` calls to estimate their FLOPs.

    ``FlopCounterMode`` does not hook FFT ops at all, so families whose real
    forward path recomputes FFTs every call (LDNS's S4 kernel) need a manual
    supplement. Rather than hardcoding layer/channel counts from the spec
    (fragile — depends on internal architecture details), this monkeypatches
    ``torch.fft.rfft``/``irfft`` for the duration of a single real call and
    accumulates FLOPs from the *actual* tensor shapes and FFT lengths used —
    which already accounts for batch size, channel/head multiplicity, and
    however many rfft/irfft calls the code path exercises.

    Uses the standard approximation for a real-valued length-N FFT:
    ``~2.5 * N * log2(N)`` flops (half of the ``5*N*log2(N)`` complex-FFT
    approximation) per length-N transform, times however many independent
    transforms are packed into the leading (non-transformed) dimensions of
    the input tensor. This is an order-of-magnitude estimate, not exact.
    """

    def __init__(self) -> None:
        self.total_flops: float = 0.0
        self._orig_rfft: Any = None
        self._orig_irfft: Any = None

    def _flops_for(self, leading_elements: int, n: int) -> float:
        if n <= 1:
            return 0.0
        return leading_elements * _REAL_FFT_FLOPS_PER_ELEMENT_LOG2N * n * math.log2(n)

    def __enter__(self) -> _FFTFlopTracker:
        self._orig_rfft = torch.fft.rfft
        self._orig_irfft = torch.fft.irfft

        def wrapped_rfft(input, n=None, dim=-1, norm=None, **kwargs):
            out = self._orig_rfft(input, n=n, dim=dim, norm=norm, **kwargs)
            length = n if n is not None else input.shape[dim]
            leading = input.numel() // max(1, input.shape[dim])
            self.total_flops += self._flops_for(leading, length)
            return out

        def wrapped_irfft(input, n=None, dim=-1, norm=None, **kwargs):
            out = self._orig_irfft(input, n=n, dim=dim, norm=norm, **kwargs)
            # irfft's output length along `dim` is the real-signal length N;
            # that N is the transform size that drives its FLOPs cost.
            length = n if n is not None else out.shape[dim]
            leading = out.numel() // max(1, out.shape[dim])
            self.total_flops += self._flops_for(leading, length)
            return out

        torch.fft.rfft = wrapped_rfft
        torch.fft.irfft = wrapped_irfft
        return self

    def __exit__(self, *exc_info) -> None:
        torch.fft.rfft = self._orig_rfft
        torch.fft.irfft = self._orig_irfft


class _CdistFlopTracker:
    """Traces real ``torch.cdist`` calls to estimate their FLOPs.

    ``FlopCounterMode`` does not hook ``torch.cdist`` — it dispatches to a
    native fused kernel rather than decomposing into ``mm``/``addmm`` at the
    level ``FlopCounterMode`` intercepts, so calm's VQ nearest-neighbor
    search (``contrib/tokenizer/calm/vq.py``'s
    ``torch.cdist(z_flat, codebook.weight)``) was measured as reporting 0
    FLOPs despite doing real, non-negligible compute. Empirically, at the
    controlled-grid point (100k query vectors x 2048 codebook entries x 384
    dims) this is ~157B FLOPs — comparable to or larger than calm's already
    -counted encoder total, not a minor correction.

    Monkeypatches ``torch.cdist`` for the duration of a single real call and
    accumulates FLOPs from the actual tensor shapes, using the standard
    pairwise-distance identity (``2 * batch * N * M * D``, the same cost an
    equivalent ``x @ y.T``-based computation would incur). Like
    ``_FFTFlopTracker``, this is an order-of-magnitude estimate — exact for
    the mm-trick Euclidean (``p=2``) path, approximate for other ``p``.
    """

    def __init__(self) -> None:
        self.total_flops: float = 0.0
        self._orig_cdist: Any = None

    def __enter__(self) -> _CdistFlopTracker:
        self._orig_cdist = torch.cdist

        def wrapped_cdist(
            x1, x2, p=2.0, compute_mode="use_mm_for_euclid_dist_if_necessary", **kwargs
        ):
            out = self._orig_cdist(x1, x2, p=p, compute_mode=compute_mode, **kwargs)
            # torch.cdist requires >=2D inputs, but guard anyway rather than
            # assume every caller respects that — an unbatched x2 (R, M)
            # against a batched x1 (B, P, M) is valid (batch broadcasts from
            # x1), so take the max of both shapes' implied batch count
            # rather than reading it off x1 alone.
            if x1.dim() >= 2 and x2.dim() >= 2:
                n, d = x1.shape[-2], x1.shape[-1]
                m = x2.shape[-2]
                batch = max(x1.numel() // max(1, n * d), x2.numel() // max(1, m * d))
                self.total_flops += 2.0 * batch * n * m * d
            return out

        torch.cdist = wrapped_cdist
        return self

    def __exit__(self, *exc_info) -> None:
        torch.cdist = self._orig_cdist


def _measure_flops_with_supplements(fn: Any) -> tuple[Any, int, int]:
    """Run *fn* once, measuring ``FlopCounterMode``-counted FLOPs plus
    whatever manual FFT/cdist supplement the call actually triggers.

    Both supplement trackers (``_FFTFlopTracker``, ``_CdistFlopTracker``)
    are entered unconditionally — each starts at 0 FLOPs and only accrues
    if the real forward path dispatches the traced op — so this needs no
    per-family allowlist and is identical whether *fn* wraps an
    ``encode()`` or a ``decode()`` call. Using one shared implementation for
    both sides (rather than duplicating the encoder's tracker-list +
    ``ExitStack`` pattern for the decoder) is what keeps them from silently
    drifting apart — see the module docstring for the ``torch.cdist`` case
    this covers on the decoder side as well.

    Returns ``(fn(), counted_flops, manual_supplement_flops)``.

    Not thread-safe: the trackers monkeypatch global ``torch.fft.rfft/irfft``
    and ``torch.cdist`` for the duration of the call. Fine for this harness's
    sequential sweep; if a concurrent caller is ever added, serialize calls
    to this function (or replace the monkeypatching with a non-global
    dispatch mechanism) first.
    """
    trackers: list[_FFTFlopTracker | _CdistFlopTracker] = [
        _FFTFlopTracker(),
        _CdistFlopTracker(),
    ]
    with torch.no_grad(), contextlib.ExitStack() as stack:
        for tracker in trackers:
            stack.enter_context(tracker)
        with FlopCounterMode(display=False) as fc:
            result = fn()
    counted_flops = int(fc.get_total_flops())
    manual_supplement_flops = int(sum(tracker.total_flops for tracker in trackers))
    return result, counted_flops, manual_supplement_flops


def _shape_parity_check(dummy_input: Any, recon: Any) -> tuple[bool, str]:
    """Compare `recon`'s (time, neuron, batch) axes against the input's."""
    if hasattr(dummy_input, "spike_matrix"):
        input_shape = tuple(dummy_input.spike_matrix.shape)  # (B, T, N)
    elif hasattr(dummy_input, "rate_matrix"):
        input_shape = tuple(dummy_input.rate_matrix.shape)
    else:
        input_shape = None

    if input_shape is None or not hasattr(recon, "shape"):
        return False, (
            f"could not determine input/recon shapes for parity check "
            f"(input_shape={input_shape}, recon type={type(recon).__name__})"
        )

    recon_shape = tuple(recon.shape)
    if len(recon_shape) < 3 or len(input_shape) < 3:
        return False, (
            f"recon shape {recon_shape} vs input shape {input_shape}: "
            f"fewer than 3 dims, cannot verify batch/time/neuron parity"
        )

    batch_ok = recon_shape[0] == input_shape[0]
    time_ok = recon_shape[1] == input_shape[1]
    neuron_ok = recon_shape[-1] == input_shape[-1]
    ok = batch_ok and time_ok and neuron_ok
    detail = (
        f"recon shape {recon_shape} vs input shape {input_shape} "
        f"(batch_ok={batch_ok}, time_ok={time_ok}, neuron_ok={neuron_ok})"
    )
    return ok, detail


def _assert_hardware_pin(tokenizer: BaseTokenizer) -> torch.device:
    """Hard-fail (no CPU fallback) unless *tokenizer* is fully resident on cuda:0.

    Per the pinned single-GPU re-run's hardware rules: CUDA must be available,
    the tokenizer is moved to ``cuda:0`` explicitly, and every parameter and
    buffer must actually be on a cuda device afterward — checked, not assumed.
    Any failure raises immediately; it is never caught-and-silently-downgraded
    to CPU.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "compute_inference_cost: torch.cuda.is_available() is False — "
            "this benchmark run requires GPU execution with no CPU fallback. "
            "Aborting rather than silently continuing on CPU."
        )
    print(f"[compute_inference_cost] cuda:0 = {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda:0")
    tokenizer.to(device)

    bad_params = [name for name, p in tokenizer.named_parameters() if not p.is_cuda]
    bad_buffers = [name for name, b in tokenizer.named_buffers() if not b.is_cuda]
    if bad_params or bad_buffers:
        raise RuntimeError(
            "compute_inference_cost: tokenizer.to('cuda:0') did not move every "
            f"parameter/buffer onto cuda — non-cuda params={bad_params}, "
            f"non-cuda buffers={bad_buffers}. Aborting (no CPU fallback)."
        )
    return device


def compute_inference_cost(
    spec: TokenizerSpec,
    tokenizer: BaseTokenizer,
    family: str,
    batch_size: int = 1,
    warmup: int = 3,
    repeats: int = 10,
    native_or_forced: str = "",
    steady_state_warmup_calls: int = 1,
) -> InferenceCostResult:
    """Compute FLOPs/latency/params inference-cost metrics for one tokenizer.

    The dummy input is generated from *spec* (via ``make_dummy_input``); no
    real data is required. ``tokenizer.eval()`` is always called first —
    ``create_tokenizer()`` returns a module in training mode, and some
    families (VQ codebook updates, LDNS training-only paths) have measurably
    different behavior under ``.train()``.

    Hardware pin (hard rule, no CPU fallback): this function asserts CUDA is
    available, moves *tokenizer* onto ``cuda:0`` explicitly, and asserts every
    parameter/buffer actually landed on cuda before doing anything else. Any
    failure raises immediately rather than silently falling back to CPU.

    Steady-state warm-up (hard rule): some families cache first-call-only
    state in a buffer that gets rebuilt (expensively) the first time a call
    implies a longer sequence than what's cached — e.g. LDNS's S4 kernel
    (``SSMKernelDPLR._setup_C``, keyed off the ``l_kernel`` buffer). Left alone, that
    one-time rebuild gets captured by ``FlopCounterMode`` as if it were
    steady-state per-call cost (empirically: ~17x FLOPs inflation for LDNS at
    window=5.0s vs. the correct ~2x). To avoid this for every family (not
    just LDNS — this is the more correct definition of steady-state inference
    cost in general, matching standard ML benchmarking practice),
    ``steady_state_warmup_calls`` untimed encode()+decode() calls are run
    before both the ``FlopCounterMode``-wrapped measurement calls and the
    timed latency loop.

    Args:
        spec: Fully-resolved tokenizer spec (see ``resolve_spec_from_checkpoint``).
        tokenizer: Tokenizer instance built from *spec* via ``create_tokenizer``.
        family: Tokenizer family name (e.g. ``"compressed_sensing"``,
            ``"ldns"``, ``"wanx_A"`` (published as conv-*)) — used only to look up
            ``_INFERENCE_IRRELEVANT_BUFFER_NAMES`` (case-insensitively);
            does not affect FLOPs/latency measurement mechanics, which run
            identically regardless of family (see
            ``_measure_flops_with_supplements``).
        batch_size: Batch dimension for the dummy input.
        warmup: Number of untimed warmup calls before each timed loop.
        repeats: Number of timed calls per timed loop.
        native_or_forced: Free-form description of whether
            (window_length_seconds, n_neurons) matches this checkpoint's
            real trained values ("native") or is a forced/synthetic
            override ("forced"). Caller-supplied; not inferred here.
        steady_state_warmup_calls: Number of untimed encode()+decode() calls
            to run before any measurement (FLOPs or latency). Defaults to 1
            so that one-time cache rebuilds stay out of the measurement. Set
            to 0 only to measure cold-start behavior deliberately — never for
            a real measurement.

    Returns:
        An ``InferenceCostResult`` with FLOPs, latency, params, and a
        shape-parity flag/detail.
    """
    # Guards direct callers (e.g. tests constructing a spec in-memory rather
    # than via resolve_spec_from_checkpoint) with the module's own clear
    # assertion instead of a confusing TypeError several lines further down.
    _assert_no_dist_fields(spec)

    tokenizer.eval()

    device = _assert_hardware_pin(tokenizer)

    dummy_input = make_dummy_input(spec, batch_size=batch_size)
    from lightning.fabric.utilities.apply_func import move_data_to_device

    dummy_input = move_data_to_device(dummy_input, device)

    # Case-insensitive: the manifest docstring (scripts/benchmark_inference_cost.py)
    # documents family labels with inconsistent casing across families
    # ("ldns", "wanx_A", "cosmos", "calm", "PCA") -- the one remaining
    # family-keyed lookup in this module (_INFERENCE_IRRELEVANT_BUFFER_NAMES)
    # normalizes to lowercase on both sides so a caller's casing choice can
    # never silently skip an exclusion rule. FLOPs supplements do not key
    # off family at all -- see _measure_flops_with_supplements.
    family_key = family.lower()

    # Elides wanx_A's discarded-but-still-computed MMD/KL aux loss for the
    # entire measured region (warmup + FLOPs + latency, encoder + decoder).
    # See _no_op_wanx_aux_loss docstring — no-op for every other family.
    with _no_op_wanx_aux_loss() as wanx_aux_loss_patch_status:
        # --- Steady-state warm-up: untimed encode()+decode(), BEFORE any
        # measurement (FLOPs or latency). See docstring: this is what keeps a
        # family's one-time cache-rebuild (LDNS's S4 kernel, in particular) out
        # of the measured FLOPs/latency entirely, for every family uniformly.
        with torch.no_grad():
            for _ in range(steady_state_warmup_calls):
                warm_encode_out = tokenizer.encode(dummy_input)
                warm_tokens = _latents_from_encode(warm_encode_out)
                warm_latents = tokenizer.tokens_to_latents(warm_tokens)
                tokenizer.decode(warm_latents)

        # --- Encoder: FLOPs (single call) ---
        encode_out, encoder_flops_counted, encoder_flops_manual_supplement = (
            _measure_flops_with_supplements(lambda: tokenizer.encode(dummy_input))
        )

        tokens = _latents_from_encode(encode_out)

        # --- Encoder: latency (separate warmup + timed loop) ---
        def _encode_once() -> None:
            tokenizer.encode(dummy_input)

        encoder_latency_ms_mean, encoder_latency_ms_std = _time_calls(
            _encode_once, device, warmup, repeats
        )

        # --- Decoder: FLOPs (single call) ---
        # Same helper as the encoder above (not a separate hand-rolled
        # tracker ladder) so encoder and decoder can never drift apart on
        # which manual supplements they check for.
        def _decode_once():
            # Include token-to-latent conversion in decoder cost, as when
            # discrete decoders performed their own codebook lookup.
            return tokenizer.decode(tokenizer.tokens_to_latents(tokens))

        recon, decoder_flops_counted, decoder_flops_manual_supplement = (
            _measure_flops_with_supplements(_decode_once)
        )

        # --- Decoder: latency (separate warmup + timed loop) ---

        decoder_latency_ms_mean, decoder_latency_ms_std = _time_calls(
            _decode_once, device, warmup, repeats
        )

    # --- Shape-parity check ---
    shape_ok, shape_detail = _shape_parity_check(dummy_input, recon)

    # --- Params ---
    total_params = sum(
        t.numel() for t in itertools.chain(tokenizer.parameters(), tokenizer.buffers())
    )
    learnable_params = sum(p.numel() for p in tokenizer.parameters())

    # inference_params: total_params minus any buffers confirmed (per
    # _INFERENCE_IRRELEVANT_BUFFER_NAMES, traced per family, not assumed)
    # to be fitting/training-only bookkeeping never touched by encode()/
    # decode(). For families not in that dict, inference_params ==
    # total_params -- there is no blanket assumption that "unlisted" means
    # "verified clean," only that no exclusion has been traced yet.
    excluded_buffer_names = _INFERENCE_IRRELEVANT_BUFFER_NAMES.get(family_key, frozenset())
    inference_irrelevant_params = sum(
        buf.numel()
        for name, buf in tokenizer.named_buffers()
        if name.rsplit(".", 1)[-1] in excluded_buffer_names
    )
    inference_params = total_params - inference_irrelevant_params

    # --- Resolved n_neurons / window_length_seconds / bin_size_ms for reporting ---
    if hasattr(dummy_input, "spike_matrix"):
        n_neurons_resolved = int(dummy_input.spike_matrix.shape[-1])
    else:
        n_neurons_resolved = int(cast(int, spec.n_neurons)) if spec.n_neurons is not False else 0

    window_length_seconds = float(spec.effective_window_length_seconds)
    bin_size_ms = 1.0 if spec.bin_size_ms is False else float(spec.bin_size_ms)

    family_extra_config = {
        "KIND": spec.KIND,
        "compress_n": spec.compress_n,
        "compress_t": spec.compress_t,
        "token_bin_size_ms": spec.token_bin_size_ms,
        # FFT and cdist supplement trackers run unconditionally for every
        # family (see _measure_flops_with_supplements) -- this is a measured
        # fact, not a family-based claim. A 0 supplement means the traced
        # ops were checked for and did not fire, not that they weren't
        # checked.
        "flops_manual_supplement_method": "fft+cdist traced unconditionally",
        "steady_state_warmup_calls": int(steady_state_warmup_calls),
        "device": str(device),
        "inference_params": int(inference_params),
        "inference_irrelevant_params_excluded": int(inference_irrelevant_params),
        "wanx_aux_loss_patch_status": wanx_aux_loss_patch_status,
        # Provenance record of the precision flags this measurement actually
        # ran under (read here, not passed in by the caller, so a driver bug
        # cannot write a value the measurement didn't really use). wanx_A
        # (conv-heavy, cudnn) and calm/pca/compressed_sensing (matmul-heavy)
        # otherwise run on different implicit TF32 paths, which confounds
        # cross-family latency comparisons -- see the harness docstring.
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
    }

    return InferenceCostResult(
        family=family,
        variant=getattr(spec, "KIND", family),
        n_neurons=n_neurons_resolved,
        window_length_seconds=window_length_seconds,
        bin_size_ms=bin_size_ms,
        batch_size=int(batch_size),
        native_or_forced=native_or_forced,
        total_params=int(total_params),
        learnable_params=int(learnable_params),
        encoder_flops_counted=encoder_flops_counted,
        encoder_flops_manual_supplement=encoder_flops_manual_supplement,
        encoder_latency_ms_mean=float(encoder_latency_ms_mean),
        encoder_latency_ms_std=float(encoder_latency_ms_std),
        decoder_flops_counted=decoder_flops_counted,
        decoder_flops_manual_supplement=decoder_flops_manual_supplement,
        decoder_latency_ms_mean=float(decoder_latency_ms_mean),
        decoder_latency_ms_std=float(decoder_latency_ms_std),
        shape_ok=bool(shape_ok),
        shape_detail=shape_detail,
        family_extra_config=family_extra_config,
    )
