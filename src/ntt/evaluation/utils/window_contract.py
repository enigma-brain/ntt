"""Eval-window compatibility contract for tokenizers.

The eval/analysis window is the single source of truth. Every tokenizer must
accept it or fail loud:

- variable_t=True (quantized-variable, e.g. calm, poisson): the eval window must
  be a multiple of the token quantum (``token_bin_size_ms``). Fed directly.
- variable_t=False (rigid-fixed, e.g. fsqffn, CS): the eval window must be a
  positive integer multiple of the native ``effective_window_length_seconds``
  (k==1 → fed directly; k>1 → chunked into k pieces by ``chunked_tokenizer_fn``).
- anything else: raise IncompatibleEvalWindowError.

``eval_window_seconds`` may also be ``"native"``: score at the spec's own
``window_length_seconds`` (see ``resolve_eval_window_seconds``). Whatever window is scored is
recorded (``eval_window_record``) and the aggregation compares it across tokenizers.
"""

from __future__ import annotations

from typing import Any

from ntt.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

NATIVE = "native"
_TOLERANCE_MS = 1e-6


class IncompatibleEvalWindowError(ValueError):
    """Raised when a tokenizer cannot be evaluated at the configured eval window."""


def _is_multiple(value_ms: float, unit_ms: float, *, tolerance_ms: float = _TOLERANCE_MS) -> bool:
    if unit_ms <= 0:
        return False
    remainder = value_ms - round(value_ms / unit_ms) * unit_ms
    return abs(remainder) <= tolerance_ms


def assert_tokenizer_accepts_eval_window(spec, eval_window_seconds: float) -> None:
    """Validate that ``spec`` can be evaluated at ``eval_window_seconds``.

    Raises:
        IncompatibleEvalWindowError: with an actionable message naming the
            tokenizer KIND, its requirement, and the eval window.
    """
    eval_window_ms = float(eval_window_seconds) * 1000.0

    if spec.variable_t:
        quantum_ms = spec.token_bin_size_ms
        if quantum_ms is None:
            return  # dynamic-bin tokenizer: no quantum constraint
        if not _is_multiple(eval_window_ms, float(quantum_ms)):
            raise IncompatibleEvalWindowError(
                f"Tokenizer KIND={spec.KIND!r} accepts windows that are multiples of its "
                f"{float(quantum_ms) / 1000.0:g} s token quantum, but the eval window is "
                f"{eval_window_seconds:g} s (not a multiple). Choose an eval window that is a "
                f"multiple of {float(quantum_ms) / 1000.0:g} s."
            )
        return

    native_s = float(spec.effective_window_length_seconds)
    native_ms = native_s * 1000.0
    if eval_window_ms + _TOLERANCE_MS < native_ms or not _is_multiple(eval_window_ms, native_ms):
        raise IncompatibleEvalWindowError(
            f"Tokenizer KIND={spec.KIND!r} has a fixed input window of {native_s:g} s; the eval "
            f"window ({eval_window_seconds:g} s) must be a positive integer multiple of it. "
            f"Set a compatible eval window or evaluate this tokenizer separately."
        )


def apply_eval_window(data_config, spec, eval_window_seconds: float):
    """Force ``data_config`` to the authoritative eval window and validate ``spec``.

    Must be called BEFORE the datamodule is constructed, because
    ``NTTDataModuleConfig.update_from_tokenizer_spec(unset_only=True)`` will not
    overwrite an explicitly-set ``window_length_seconds``. Returns ``data_config``.
    """
    assert_tokenizer_accepts_eval_window(spec, eval_window_seconds)
    data_config.window_length_seconds = float(eval_window_seconds)
    return data_config


def resolve_eval_window_seconds(requested: float | str, spec) -> float:
    """The window (seconds) to score at: a number as given, or ``"native"`` = the spec's own
    ``window_length_seconds``.

    ``"native"`` needs one fixed window to resolve to, so a distribution-valued
    ``window_length_seconds`` (calm's ``UniformDist(0.1, 2.0)``) raises: pick it explicitly.
    """
    if requested != NATIVE:
        return float(requested)
    native = getattr(spec, "window_length_seconds", None)
    if isinstance(native, bool) or not isinstance(native, int | float):
        raise IncompatibleEvalWindowError(
            f"eval_window_seconds='native' needs a tokenizer with a fixed window_length_seconds, "
            f"but {getattr(spec, 'KIND', type(spec).__name__)!r} has {native!r}. Set "
            f"eval_window_seconds to a number."
        )
    return float(native)


def eval_window_record(spec, requested: float | str, scored_seconds: float) -> dict[str, Any]:
    """What a run saves about its window: the scored one, what was asked for, and the spec's own
    (``None`` if it is a distribution)."""
    native = getattr(spec, "window_length_seconds", None)
    return {
        "scored_window_seconds": float(scored_seconds),
        "eval_window_requested": requested,
        "spec_window_length_seconds": float(native) if isinstance(native, int | float) else None,
    }


def warn_if_off_native_window(spec, scored_seconds: float) -> None:
    """Warn when a tokenizer with one fixed training window is scored at another.

    No warning for distribution-valued windows (no single native window) or for rigid
    (``variable_t=False``) tokenizers, which are fed their native window in chunks.
    """
    native = getattr(spec, "window_length_seconds", None)
    if isinstance(native, bool) or not isinstance(native, int | float) or not spec.variable_t:
        return
    if abs(float(native) - float(scored_seconds)) > _TOLERANCE_MS / 1000.0:
        log.warning(
            "Tokenizer %r was built for %g s windows but is scored at %g s: scores are not "
            "comparable with runs at its native window. To score at %g s set "
            "job.eval_window_seconds=native (job=eval_all) or "
            "++job.eval_config.eval_window_seconds=native (the pass jobs).",
            getattr(spec, "KIND", type(spec).__name__),
            float(native),
            float(scored_seconds),
            float(native),
        )


def check_datamodule_window(datamodule, expected_seconds: float, *, what: str) -> float:
    """Fail loud if ``datamodule`` will load another window than the one being scored.
    Returns the window the datamodule loads."""
    actual = datamodule.config.window_length_seconds
    if not isinstance(actual, int | float) or abs(float(actual) - expected_seconds) > 1e-9:
        raise RuntimeError(
            f"{what}: the datamodule loads windows of {actual!r} s, not the eval window "
            f"{expected_seconds:g} s."
        )
    return float(actual)
