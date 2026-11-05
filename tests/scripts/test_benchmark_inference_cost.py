"""Tests for scripts/benchmark_inference_cost.py's batch-size handling.

Covers two edge cases of ``process_row``'s batch-size sweep: duplicate batch
sizes must not corrupt the tokenizer-build-failure recovery path (no value-based
``list.index`` lookup), and non-positive batch sizes must be validated rather
than failing confusingly downstream.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


def _load_module():
    script_dir = Path(__file__).resolve().parents[2] / "scripts"
    sys.path.insert(0, str(script_dir))
    try:
        import importlib

        import benchmark_inference_cost as bic

        importlib.reload(bic)
    finally:
        sys.path.pop(0)
    return bic


bic = _load_module()


class _FakeTokenizer:
    def eval(self):
        return self


class _FakeResult:
    """Mimics the subset of InferenceCostResult that process_row reads
    directly (for its progress-line print) as well as via to_dict()."""

    shape_ok = True
    encoder_flops_counted = 0
    encoder_flops_manual_supplement = 0
    decoder_flops_counted = 0
    decoder_flops_manual_supplement = 0
    encoder_latency_ms_mean = 0.0
    decoder_latency_ms_mean = 0.0

    def to_dict(self):
        return {
            "shape_ok": True,
            "encoder_flops_counted": 0,
            "encoder_flops_manual_supplement": 0,
            "decoder_flops_counted": 0,
            "decoder_flops_manual_supplement": 0,
            "encoder_latency_ms_mean": 0.0,
            "decoder_latency_ms_mean": 0.0,
        }


def test_duplicate_batch_sizes_do_not_corrupt_failure_recovery(monkeypatch):
    """batch_sizes with a repeated value must not re-slice from the first
    occurrence when a later iteration fails — that would both duplicate
    already-succeeded rows and mis-report their batch size."""
    monkeypatch.setattr(bic, "resolve_spec_from_checkpoint", lambda path, overrides: object())

    call_count = {"n": 0}

    def fake_create_tokenizer(spec):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated build failure on the second iteration")
        return _FakeTokenizer()

    monkeypatch.setattr(bic, "create_tokenizer", fake_create_tokenizer)
    monkeypatch.setattr(
        bic, "load_tokenizer_weights_from_checkpoint", lambda path, tok: (True, "ok")
    )
    monkeypatch.setattr(bic, "compute_inference_cost", lambda *a, **k: _FakeResult())

    row = pd.Series(
        {
            "family": "ldns",
            "variant": "v",
            "checkpoint_path": "/fake/path.ckpt",
            "batch_sizes": "2;2;4",
        }
    )
    rows = bic.process_row(
        row, default_batch_sizes=[1], warmup=0, repeats=0, excluded_families=frozenset()
    )

    # One row per swept batch size (3), regardless of the duplicate value.
    assert len(rows) == 3
    assert sum(1 for r in rows if r.get("shape_ok") is True) == 1
    assert sum(1 for r in rows if r.get("shape_ok") is False) == 2


@pytest.mark.parametrize("raw", ["0", "-1", "1;0;4"])
def test_non_positive_batch_size_raises(raw):
    with pytest.raises(ValueError, match="non-positive"):
        bic._parse_batch_sizes(raw, default=[1])


@pytest.mark.parametrize("raw", [";", ";;", " ; "])
def test_all_blank_batch_sizes_raises(raw):
    """A manifest 'batch_sizes' cell like ';' parses to an empty list after
    filtering blank entries -- that must fail loudly rather than silently
    producing zero measurements for the row (distinct from a fully blank
    cell, which legitimately means 'use the CLI default')."""
    with pytest.raises(ValueError, match="no batch sizes"):
        bic._parse_batch_sizes(raw, default=[1])


def test_blank_batch_sizes_cell_uses_default():
    """An actually-blank cell (None/NaN/'') is the documented 'use
    --batch-sizes' signal and must not raise."""
    assert bic._parse_batch_sizes("", default=[1, 4]) == [1, 4]
    assert bic._parse_batch_sizes(None, default=[1, 4]) == [1, 4]


def test_positive_batch_sizes_parse_normally():
    assert bic._parse_batch_sizes("1;4;16", default=[1]) == [1, 4, 16]


@pytest.mark.parametrize(
    "option,value",
    [("--repeats", "0"), ("--repeats", "-1"), ("--warmup", "-1")],
)
def test_timing_count_cli_rejects_values_below_boundary(option, value):
    with pytest.raises(SystemExit) as exc_info:
        bic._build_parser().parse_args([option, value])

    assert exc_info.value.code == 2


def test_timing_count_cli_accepts_boundary_values():
    args = bic._build_parser().parse_args(["--repeats", "1", "--warmup", "0"])

    assert args.repeats == 1
    assert args.warmup == 0


def _manifest_row(batch_sizes: str = "1") -> pd.Series:
    return pd.Series(
        {
            "family": "ldns",
            "variant": "v",
            "checkpoint_path": "/fake/path.ckpt",
            "batch_sizes": batch_sizes,
        }
    )


def _patch_successful_setup(monkeypatch):
    monkeypatch.setattr(bic, "resolve_spec_from_checkpoint", lambda path, overrides: object())
    monkeypatch.setattr(bic, "create_tokenizer", lambda spec: _FakeTokenizer())
    monkeypatch.setattr(
        bic, "load_tokenizer_weights_from_checkpoint", lambda path, tok: (True, "ok")
    )


def test_oom_records_failure_cleans_up_and_continues(monkeypatch):
    _patch_successful_setup(monkeypatch)
    calls: list[int] = []

    def fake_compute(*args, **kwargs):
        calls.append(kwargs["batch_size"])
        if len(calls) == 1:
            raise bic.torch.cuda.OutOfMemoryError("simulated OOM")
        return _FakeResult()

    empty_cache_calls: list[None] = []
    monkeypatch.setattr(bic, "compute_inference_cost", fake_compute)
    monkeypatch.setattr(bic.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(bic.torch.cuda, "empty_cache", lambda: empty_cache_calls.append(None))

    rows = bic.process_row(
        _manifest_row("1;2"),
        default_batch_sizes=[1],
        warmup=0,
        repeats=1,
        excluded_families=frozenset(),
    )

    assert calls == [1, 2]
    assert len(rows) == 2
    assert rows[0]["shape_ok"] is False
    assert "OutOfMemoryError" in rows[0]["shape_detail"]
    assert rows[1]["shape_ok"] is True
    assert len(empty_cache_calls) == 2


def test_successful_measurement_cleans_up_cuda_cache(monkeypatch):
    _patch_successful_setup(monkeypatch)
    monkeypatch.setattr(bic, "compute_inference_cost", lambda *args, **kwargs: _FakeResult())
    empty_cache_calls: list[None] = []
    monkeypatch.setattr(bic.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(bic.torch.cuda, "empty_cache", lambda: empty_cache_calls.append(None))

    rows = bic.process_row(
        _manifest_row(),
        default_batch_sizes=[1],
        warmup=0,
        repeats=1,
        excluded_families=frozenset(),
    )

    assert rows[0]["shape_ok"] is True
    assert len(empty_cache_calls) == 1


def test_tokenizer_load_failure_cleans_up_cuda_cache(monkeypatch):
    monkeypatch.setattr(bic, "resolve_spec_from_checkpoint", lambda path, overrides: object())
    monkeypatch.setattr(bic, "create_tokenizer", lambda spec: _FakeTokenizer())

    def fail_load(path, tokenizer):
        raise RuntimeError("simulated load failure")

    empty_cache_calls: list[None] = []
    monkeypatch.setattr(bic, "load_tokenizer_weights_from_checkpoint", fail_load)
    monkeypatch.setattr(bic.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(bic.torch.cuda, "empty_cache", lambda: empty_cache_calls.append(None))

    rows = bic.process_row(
        _manifest_row(),
        default_batch_sizes=[1],
        warmup=0,
        repeats=1,
        excluded_families=frozenset(),
    )

    assert rows[0]["shape_ok"] is False
    assert "tokenizer build" in rows[0]["shape_detail"]
    assert len(empty_cache_calls) == 1
