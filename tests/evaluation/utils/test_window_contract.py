from __future__ import annotations

from types import SimpleNamespace

import pytest

from ntt.evaluation.utils.window_contract import (
    IncompatibleEvalWindowError,
    assert_tokenizer_accepts_eval_window,
)


def _spec(*, variable_t, token_bin_size_ms=None, window_s=None, kind="x"):
    return SimpleNamespace(
        KIND=kind,
        variable_t=variable_t,
        token_bin_size_ms=token_bin_size_ms,
        effective_window_length_seconds=window_s,
    )


def test_variable_t_multiple_of_quantum_ok():
    assert_tokenizer_accepts_eval_window(
        _spec(variable_t=True, token_bin_size_ms=100.0, kind="calm"), 1.0
    )


def test_variable_t_non_multiple_raises():
    with pytest.raises(IncompatibleEvalWindowError, match="calm"):
        assert_tokenizer_accepts_eval_window(
            _spec(variable_t=True, token_bin_size_ms=100.0, kind="calm"), 0.15
        )


def test_variable_t_dynamic_bin_skips():
    assert_tokenizer_accepts_eval_window(
        _spec(variable_t=True, token_bin_size_ms=None, kind="dyn"), 1.0
    )


def test_fixed_exact_match_ok():
    assert_tokenizer_accepts_eval_window(_spec(variable_t=False, window_s=1.0, kind="fsqffn"), 1.0)


def test_fixed_integer_multiple_ok():
    assert_tokenizer_accepts_eval_window(_spec(variable_t=False, window_s=0.1, kind="cs"), 1.0)


def test_fixed_non_divisor_raises():
    with pytest.raises(IncompatibleEvalWindowError, match="fsqffn"):
        assert_tokenizer_accepts_eval_window(
            _spec(variable_t=False, window_s=1.0, kind="fsqffn"), 1.5
        )


def test_fixed_native_larger_than_eval_raises():
    with pytest.raises(IncompatibleEvalWindowError):
        assert_tokenizer_accepts_eval_window(
            _spec(variable_t=False, window_s=2.0, kind="big"), 1.0
        )


# ── "native" eval window, record, warning ───────────────────────────


def _fixed_window_spec(window_s, *, variable_t=True, kind="ldns"):
    return SimpleNamespace(KIND=kind, variable_t=variable_t, window_length_seconds=window_s)


def test_native_resolves_to_the_spec_window_and_numbers_pass_through():
    from ntt.evaluation.utils.window_contract import resolve_eval_window_seconds

    assert resolve_eval_window_seconds("native", _fixed_window_spec(4.0)) == 4.0
    assert resolve_eval_window_seconds(1.0, _fixed_window_spec(4.0)) == 1.0


def test_native_with_a_distribution_window_raises():
    from ntt.core.components.dist import UniformDist
    from ntt.evaluation.utils.window_contract import resolve_eval_window_seconds

    calm_like = _fixed_window_spec(UniformDist(low=0.1, high=2.0), kind="calm")
    with pytest.raises(IncompatibleEvalWindowError, match="calm"):
        resolve_eval_window_seconds("native", calm_like)
    assert resolve_eval_window_seconds(1.0, calm_like) == 1.0


def test_record_has_the_scored_requested_and_spec_windows():
    from ntt.evaluation.utils.window_contract import eval_window_record

    assert eval_window_record(_fixed_window_spec(4.0), "native", 4.0) == {
        "scored_window_seconds": 4.0,
        "eval_window_requested": "native",
        "spec_window_length_seconds": 4.0,
    }
    assert (
        eval_window_record(_fixed_window_spec(object()), 1.0, 1.0)["spec_window_length_seconds"]
        is None
    )


def test_warning_fires_for_a_fixed_window_tokenizer_scored_elsewhere(caplog):
    from ntt.evaluation.utils.window_contract import warn_if_off_native_window

    with caplog.at_level("WARNING"):
        warn_if_off_native_window(_fixed_window_spec(4.0, kind="ldns_perceiver"), 1.0)
    assert "ldns_perceiver" in caplog.text and "built for 4 s" in caplog.text
    assert "scored at 1 s" in caplog.text


@pytest.mark.parametrize(
    "spec, scored",
    [
        (_fixed_window_spec(4.0), 4.0),  # native
        (_fixed_window_spec(object(), kind="calm"), 1.0),  # distribution window
        (_fixed_window_spec(0.1, variable_t=False, kind="cs"), 1.0),  # rigid, chunked
    ],
)
def test_no_warning_at_native_or_without_a_single_native_window(caplog, spec, scored):
    from ntt.evaluation.utils.window_contract import warn_if_off_native_window

    with caplog.at_level("WARNING"):
        warn_if_off_native_window(spec, scored)
    assert caplog.text == ""


def test_datamodule_window_check_fails_loud_on_a_mismatch():
    from ntt.evaluation.utils.window_contract import check_datamodule_window

    dm = SimpleNamespace(config=SimpleNamespace(window_length_seconds=4.0))
    check_datamodule_window(dm, 4.0, what="x")
    with pytest.raises(RuntimeError, match="not the eval window 1 s"):
        check_datamodule_window(dm, 1.0, what="x")


def test_schema_rejects_nan_and_warning_names_both_knobs(caplog):
    from ntt.evaluation.config.schema import EvaluationConfig
    from ntt.evaluation.utils.window_contract import warn_if_off_native_window

    with pytest.raises(ValueError):
        EvaluationConfig(eval_window_seconds=float("nan"))
    with caplog.at_level("WARNING"):
        warn_if_off_native_window(_fixed_window_spec(4.0), 1.0)
    assert "job.eval_window_seconds=native" in caplog.text
    assert "++job.eval_config.eval_window_seconds=native" in caplog.text
