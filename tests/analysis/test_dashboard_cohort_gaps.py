"""What a cohort lacks is shown as lacking: absent optional files get a notice, a score no
model of the cohort has is not a radar axis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import scripts.analysis.dashboard as D
from bokeh.models import Column, ColumnDataSource, CustomJS, Div, Label, Tabs

REPO = Path(__file__).resolve().parents[2]
FIX = REPO / "tests/data/dashboard"
REQUIRED = {
    "category_scores.csv": "SAMPLE_category_scores_aggregate.csv",
    "wide_metrics_table.csv": "SAMPLE_wide_metrics_table.csv",
    "metadata_rows.csv": "SAMPLE_metadata_rows.csv",
}


def _required_only(tmp_path: Path, **overrides: pd.DataFrame) -> Path:
    d = tmp_path / "cohort"
    d.mkdir()
    for dst, src in REQUIRED.items():
        overrides.get(dst, pd.read_csv(FIX / src)).to_csv(d / dst, index=False)
    return d


def _tab_text(view, title):
    tabs = next(m for m in view.references() if isinstance(m, Tabs))
    child = next(p for p in tabs.tabs if p.title == title).child
    return " ".join(m.text for m in child.references() if isinstance(m, Div))


def test_absent_optional_files_get_a_notice(tmp_path):
    page = D.build_page([D.load_cohort("200 ms", _required_only(tmp_path))])
    for view in (m for m in page.references() if isinstance(m, Column) and m.name):
        if view.name not in ("simple_view", "dev_view"):
            continue
        gen = _tab_text(view, "Generalization")
        assert "generalization_by_tier.csv" in gen and "not supplied" in gen
        cav = _tab_text(view, "Caveats")
        assert "caveats.csv" in cav and "not supplied" in cav
        assert "No model-specific caveats" not in cav


def test_supplied_but_empty_caveats_still_say_none(tmp_path):
    d = _required_only(tmp_path)
    pd.DataFrame(columns=["tokenizer", "footnote", "notes"]).to_csv(d / "caveats.csv", index=False)
    cohort = D.load_cohort("200 ms", d)
    view = next(
        m
        for m in D.build_page([cohort]).references()
        if isinstance(m, Column) and m.name == "dev_view"
    )
    assert "No model-specific caveats" in _tab_text(view, "Caveats")


def _no_encoding(tmp_path: Path) -> D.CohortData:
    agg = pd.read_csv(FIX / REQUIRED["category_scores.csv"])
    agg["downstream_encoding"] = float("nan")  # the 1000 ms-token cohort: no encoding
    return D.load_cohort("1000 ms", _required_only(tmp_path, **{"category_scores.csv": agg}))


def test_a_score_no_model_has_is_not_a_radar_axis(tmp_path):
    cohort = _no_encoding(tmp_path)
    f = cohort.frame
    panel, cb = D._build_radar_tab(f, D._build_full_source(f), *D._build_filter_widgets(f))
    labels = [m.text for m in panel.child.references() if isinstance(m, Label)]
    assert len(labels) == 6 and "Encoding" not in labels
    src = next(
        m
        for m in panel.child.references()
        if isinstance(m, ColumnDataSource) and "v_single_neuron_recon" in m.data
    )
    assert "v_downstream_encoding" not in src.data
    assert all(len(xs) == 7 for xs in src.data["xs"])  # 6 axes + closing vertex
    assert '"downstream_encoding"' not in cb.code
    notes = " ".join(m.text for m in panel.child.references() if isinstance(m, Div))
    assert "Not in this cohort" in notes and "Encoding" in notes


def test_a_score_no_model_has_is_not_pickable(tmp_path):
    cohort = _no_encoding(tmp_path)
    values = {e["value"] for e in cohort.registry}
    assert "downstream_encoding" not in values and "decoding" in values


def test_generalization_without_tables_has_no_failing_callback(tmp_path):
    cohort = D.load_cohort("200 ms", _required_only(tmp_path))
    f = cohort.frame
    _, cb = D._build_generalization_tab(f, D._build_full_source(f), *D._build_filter_widgets(f))
    assert isinstance(cb, CustomJS) and "gen_tier/" not in cb.code
