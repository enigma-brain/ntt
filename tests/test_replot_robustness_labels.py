"""Tests for the tokenizer-label helpers in ``scripts/replot_robustness.py``.

Focused on the Poisson disambiguation logic — bare ``poisson`` legend
entries are useless when comparing two Poisson runs at different
``rate_estimation_bin_ms``, so the label rewriter pulls that knob out
of the run dir's ``run_meta.json`` and appends ``_<N>ms``.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "replot_robustness.py"
_SPEC = importlib.util.spec_from_file_location("ntt_replot_robustness", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

_build_input_robustness_payload_from_csv = _MODULE._build_input_robustness_payload_from_csv
_decorate_poisson_label = _MODULE._decorate_poisson_label
_format_rate_bin_ms_for_label = _MODULE._format_rate_bin_ms_for_label
_load_m2_results_frame = _MODULE._load_m2_results_frame
_tokenizer_label_for = _MODULE._tokenizer_label_for


# ── _format_rate_bin_ms_for_label ─────────────────────────────────────


@pytest.mark.parametrize(
    ("rate_bin_ms", "expected"),
    [
        (10, "10"),
        (10.0, "10"),
        (5, "5"),
        (12.5, "12.5"),
    ],
)
def test_format_rate_bin_ms_for_label(rate_bin_ms, expected):
    """Integer-valued floats drop the decimal; non-integers keep 1 dp."""
    assert _format_rate_bin_ms_for_label(rate_bin_ms) == expected


# ── _decorate_poisson_label ───────────────────────────────────────────


def test_decorate_poisson_label_uses_run_meta_json(tmp_path: Path):
    """When ``run_meta.json`` carries ``rate_estimation_bin_ms``, the
    bare ``poisson`` label is rewritten to ``poisson_<N>ms``.

    """
    run_dir = tmp_path / "robustness_input_poisson_20260427_120000"
    run_dir.mkdir()
    (run_dir / "run_meta.json").write_text(json.dumps({"rate_estimation_bin_ms": 25.0}))
    assert _decorate_poisson_label("poisson", run_dir) == "poisson_25ms"


def test_decorate_poisson_label_passthrough_for_non_poisson(tmp_path: Path):
    """Non-Poisson labels are returned unchanged regardless of metadata."""
    run_dir = tmp_path / "robustness_input_vae_20260427_120000"
    run_dir.mkdir()
    (run_dir / "run_meta.json").write_text(json.dumps({"rate_estimation_bin_ms": 25.0}))
    assert _decorate_poisson_label("vae", run_dir) == "vae"


def test_decorate_poisson_label_falls_back_when_meta_missing(tmp_path: Path):
    """Missing metadata: keep the bare ``poisson`` label rather than
    crashing — disambiguation is best-effort.

    """
    run_dir = tmp_path / "robustness_input_poisson_20260427_120000"
    run_dir.mkdir()
    # No run_meta.json, no .hydra/.
    assert _decorate_poisson_label("poisson", run_dir) == "poisson"


# ── End-to-end via _tokenizer_label_for ───────────────────────────────


def test_tokenizer_label_for_poisson_with_run_meta(tmp_path: Path):
    """End-to-end: a Poisson run dir whose run_meta.json declares the
    rate bin produces a ``poisson_<N>ms`` label.

    """
    run_dir = tmp_path / "robustness_input_poisson_20260427_143453"
    run_dir.mkdir()
    (run_dir / "run_meta.json").write_text(json.dumps({"rate_estimation_bin_ms": 10.0}))
    # The 1ms input bin contributes no suffix (see _bin_size_suffix_for:
    # only attaches a suffix when ``bin_size_ms`` is in run_meta.json).
    assert _tokenizer_label_for(run_dir) == "poisson_10ms"


def test_build_input_robustness_payload_from_csv_accepts_raw_value_column(tmp_path: Path):
    """Replotting should accept the renamed ``raw_value`` column in input_robustness (M1) CSVs."""
    csv_path = tmp_path / "results.csv"
    csv_path.write_text(
        "session_id,segment_index,corruption,metric,sub_label,raw_value,status\n"
        "sess0,0,drop|fraction=0.1,acg_correlation,,0.9,ok\n"
    )

    payload = _build_input_robustness_payload_from_csv(csv_path, metric_name="acg_correlation")

    assert payload["drop"]["strengths"] == [0.1]
    assert payload["drop"]["means"] == [0.9]


def test_load_m2_results_frame_accepts_raw_value_jsonl(tmp_path: Path):
    """JSONL fallback for latent_robustness (M2) replotting should accept the renamed
    ``raw_value`` key."""
    results_dir = tmp_path / "m2_run"
    results_dir.mkdir()
    (results_dir / "results.jsonl").write_text(
        '{"session_id":"sess0","segment_index":0,"epsilon":0.1,"k":0,"metric":"acg_correlation","raw_value":0.9,"status":"ok"}\n'
    )

    df = _load_m2_results_frame(results_dir)

    assert list(df["acg_correlation"]) == [0.9]


def test_load_m2_results_frame_accepts_session_level_jsonl_without_segment_index(
    tmp_path: Path,
):
    results_dir = tmp_path / "m2_run"
    results_dir.mkdir()
    (results_dir / "results.jsonl").write_text(
        '{"session_id":"sess0","epsilon":0.1,"metric":"acg_correlation","raw_value":0.9,"status":"ok"}\n'
        '{"session_id":"sess0","epsilon":0.1,"metric":"ccg_correlation","raw_value":0.8,"status":"ok"}\n'
    )

    df = _load_m2_results_frame(results_dir)

    assert "segment_index" not in df.columns
    assert "k" not in df.columns
    assert list(df["acg_correlation"]) == [0.9]
    assert list(df["ccg_correlation"]) == [0.8]
