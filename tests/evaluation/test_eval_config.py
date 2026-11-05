from __future__ import annotations

from ntt.evaluation.config import EvaluationConfig


def test_eval_window_seconds_defaults_to_one():
    assert EvaluationConfig().eval_window_seconds == 1.0


def test_eval_window_seconds_is_overridable():
    assert EvaluationConfig(eval_window_seconds=2.0).eval_window_seconds == 2.0
