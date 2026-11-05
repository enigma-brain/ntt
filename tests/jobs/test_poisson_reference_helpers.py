"""Core tests for the M1/M2 Poisson-reference helpers + the _to_scalar guard.

Covers the loud-failure paths of ``ntt.jobs._poisson_reference`` and
``ntt.evaluation.aggregation.comparison._to_scalar`` so the warnings stay
wired up. These are the post-hoc safety nets — exhaustive coverage of the
loaders themselves lives in ``tests/test_poisson_reference.py``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest

from ntt.evaluation.aggregation.comparison import _to_scalar
from ntt.jobs._poisson_reference import (
    load_poisson_rate_bin_ms,
    resolve_poisson_reference_paths,
    warn_if_no_session_overlap,
)


@pytest.fixture
def warn_log(caplog):
    """Capture WARNING-level records from any logger used by the helpers."""
    caplog.set_level(logging.WARNING)
    return caplog


def test_resolve_paths_directory(tmp_path: Path) -> None:
    csv = tmp_path / "results_per_session.csv"
    csv.write_text("session_id,metric\n")
    csv_path, meta_path = resolve_poisson_reference_paths(tmp_path, mode="M1")
    assert csv_path == csv
    assert meta_path == tmp_path / "run_meta.json"


def test_resolve_paths_csv_file(tmp_path: Path) -> None:
    csv = tmp_path / "results_per_session.csv"
    csv.write_text("session_id,metric\n")
    csv_path, meta_path = resolve_poisson_reference_paths(csv, mode="M2")
    assert csv_path == csv
    assert meta_path == tmp_path / "run_meta.json"


def test_resolve_paths_rejects_bogus(tmp_path: Path) -> None:
    bogus = tmp_path / "something_else.txt"
    bogus.write_text("x")
    with pytest.raises(ValueError, match="must point to a results directory"):
        resolve_poisson_reference_paths(bogus, mode="M1")


def test_resolve_paths_missing_csv(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing results_per_session.csv"):
        resolve_poisson_reference_paths(tmp_path, mode="M1")


def test_rate_bin_ms_happy(tmp_path: Path, warn_log) -> None:
    meta = tmp_path / "run_meta.json"
    meta.write_text(json.dumps({"rate_estimation_bin_ms": 25.0}))
    out = load_poisson_rate_bin_ms(meta, mode="M1", log=logging.getLogger("test"))
    assert out == 25.0
    assert not warn_log.records


def test_rate_bin_ms_absent_file_returns_none_no_warning(warn_log) -> None:
    out = load_poisson_rate_bin_ms(
        Path("/definitely/does/not/exist.json"),
        mode="M1",
        log=logging.getLogger("test"),
    )
    assert out is None
    assert not warn_log.records


def test_rate_bin_ms_corrupt_json_warns(tmp_path: Path, warn_log) -> None:
    """A corrupt run_meta.json must produce a warning, not be swallowed silently."""
    bad = tmp_path / "run_meta.json"
    bad.write_text("{not valid json")
    out = load_poisson_rate_bin_ms(bad, mode="M1", log=logging.getLogger("test"))
    assert out is None
    msgs = [r.getMessage() for r in warn_log.records]
    assert any("JSONDecodeError" in m and "M1" in m for m in msgs), msgs


def test_rate_bin_ms_nonnumeric_value_warns(tmp_path: Path, warn_log) -> None:
    bad = tmp_path / "run_meta.json"
    bad.write_text(json.dumps({"rate_estimation_bin_ms": "not-a-number"}))
    out = load_poisson_rate_bin_ms(bad, mode="M2", log=logging.getLogger("test"))
    assert out is None
    msgs = [r.getMessage() for r in warn_log.records]
    assert any("non-numeric" in m and "M2" in m for m in msgs), msgs


def test_session_overlap_empty_warns(warn_log) -> None:
    """Disjoint session sets must warn: the aggregate comparison would be meaningless."""
    out = warn_if_no_session_overlap(
        model_sessions={"a", "b"},
        poisson_sessions={"c", "d"},
        ceiling_sessions={"a", "b"},
        mode="M2",
        log=logging.getLogger("test"),
    )
    assert out == set()
    msgs = [r.getMessage() for r in warn_log.records]
    assert any("no overlap" in m and "M2" in m for m in msgs), msgs


def test_session_overlap_present_silent(warn_log) -> None:
    out = warn_if_no_session_overlap(
        model_sessions={"a", "b"},
        poisson_sessions={"a", "b"},
        ceiling_sessions={"a"},
        mode="M1",
        log=logging.getLogger("test"),
    )
    assert out == {"a"}
    assert not warn_log.records


def test_to_scalar_array_warns(warn_log) -> None:
    """A multi-element array reaching the normalizer means
    per-(metric, corruption, strength) bucketing was lost upstream."""
    out = _to_scalar(np.array([1.0, 2.0, 3.0]))
    assert out == pytest.approx(2.0)
    msgs = [r.getMessage() for r in warn_log.records]
    assert any("per-bucket granularity is lost" in m for m in msgs), msgs


def test_to_scalar_single_element_array_no_warning(warn_log) -> None:
    out = _to_scalar(np.array([7.5]))
    assert out == pytest.approx(7.5)
    assert not warn_log.records


def test_to_scalar_scalar_no_warning(warn_log) -> None:
    out = _to_scalar(3.0)
    assert out == pytest.approx(3.0)
    assert not warn_log.records
