"""Caveats: † marker, hover text, cohort-wide banner, Caveats tab, escaping."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import scripts.analysis.dashboard as D
from bokeh.models import Column, ColumnDataSource, CustomJS, Div, HoverTool, Tabs

REPO = Path(__file__).resolve().parents[2]
FIX = REPO / "tests/data/dashboard"
VAE = "wanx_A_vae_bce_dw_k1_lightdec_noncausal_anneal1e-6_sigreg_b1e-3_z32"
COSMOS = "cosmos_public_lc2_rfsq"
NASTY = "compression: not computed (eval: \"Session '__validation__'\" <b>x</b> -> & y)"


@pytest.fixture
def frame():
    return D.load_dashboard_frame(
        FIX / "SAMPLE_category_scores_aggregate.csv",
        FIX / "SAMPLE_wide_metrics_table.csv",
        FIX / "SAMPLE_metadata_rows.csv",
        FIX / "SAMPLE_generalization_by_tier.csv",
    )


@pytest.fixture
def caveats_csv(tmp_path):
    p = tmp_path / "caveats.csv"
    pd.DataFrame(
        [
            {"tokenizer": VAE, "footnote": "shared window note", "notes": ""},
            {"tokenizer": COSMOS, "footnote": "shared window note", "notes": "provenance detail"},
            {"tokenizer": COSMOS, "footnote": NASTY, "notes": "provenance detail"},
            {"tokenizer": "Poisson @ 10 ms", "footnote": "Poisson reference", "notes": ""},
        ]
    ).to_csv(p, index=False)
    return p


def test_load_caveats_lifts_shared_footnotes(caveats_csv, frame):
    cav = D.load_caveats(caveats_csv, frame["tokenizer"])
    assert cav.cohort_wide == ["shared window note"]
    assert cav.per_model == {COSMOS: [NASTY]}  # VAE has only the shared one
    assert cav.notes == {COSMOS: "provenance detail"}
    assert "Poisson @ 10 ms" not in cav.per_model  # not in this cohort


def test_missing_caveats_file_is_empty(frame):
    assert D.load_caveats(None, frame["tokenizer"]) == D.NO_CAVEATS
    assert D.load_caveats(Path("/nonexistent.csv"), frame["tokenizer"]) == D.NO_CAVEATS


def test_one_model_cohort_lifts_nothing(tmp_path):
    p = tmp_path / "c.csv"
    pd.DataFrame([{"tokenizer": "A", "footnote": "x", "notes": ""}]).to_csv(p, index=False)
    assert D.load_caveats(p, ["A"]) == D.Caveats({"A": ["x"]}, [], {})


def test_apply_caveats_marks_labels(caveats_csv, frame):
    cav = D.load_caveats(caveats_csv, frame["tokenizer"])
    df = D.apply_caveats(frame, cav).set_index("tokenizer")
    assert df.loc[COSMOS, "label"] == f"{COSMOS} †" and df.loc[COSMOS, "caveat_mark"] == "†"
    assert df.loc[VAE, "label"] == VAE and df.loc[VAE, "caveats"] == "—"
    assert NASTY in df.loc[COSMOS, "caveats"]


def test_frame_without_caveats_has_plain_labels(frame):
    assert list(frame["label"]) == list(frame["tokenizer"])
    assert set(frame["caveats"]) == {"—"}


def _view(layout, name):
    return next(m for m in layout.references() if isinstance(m, Column) and m.name == name)


def test_views_show_marker_banner_and_tab(caveats_csv, frame):
    cav = D.load_caveats(caveats_csv, frame["tokenizer"])
    layout = D.build_dashboard(frame, D.build_metric_registry(frame), caveats=cav)
    for name in ("simple_view", "dev_view"):
        view = _view(layout, name)
        sources = [m for m in view.references() if isinstance(m, ColumnDataSource)]
        names = {str(v) for s in sources for v in s.data.get("tokenizer", [])}
        assert any(n.endswith(" †") for n in names), name
        divs = " ".join(m.text for m in view.references() if isinstance(m, Div))
        assert "shared window note" in divs  # banner
        tabs = next(m for m in view.references() if isinstance(m, Tabs))
        assert [p.title for p in tabs.tabs] == [
            "Radar",
            "Scatter",
            "Ranking",
            "Generalization",
            "Caveats",
            "Models",
            "Glossary",
        ]
        cav_tab = next(p for p in tabs.tabs if p.title == "Caveats").child.text
        assert "&lt;b&gt;x&lt;/b&gt;" in cav_tab and "<b>x</b>" not in cav_tab  # escaped
        assert ("provenance detail" in cav_tab) is (name == "dev_view")  # notes: dev only


def test_footnote_text_never_reaches_js_code(caveats_csv, frame):
    cav = D.load_caveats(caveats_csv, frame["tokenizer"])
    layout = D.build_dashboard(frame, D.build_metric_registry(frame), caveats=cav)
    code = " ".join(m.code for m in layout.references() if isinstance(m, CustomJS))
    assert "__validation__" not in code


def test_hover_tooltips_carry_caveats(caveats_csv, frame):
    cav = D.load_caveats(caveats_csv, frame["tokenizer"])
    layout = D.build_dashboard(frame, D.build_metric_registry(frame), caveats=cav)
    tips = [m.tooltips for m in layout.references() if isinstance(m, HoverTool)]
    assert tips and all("@caveats" in str(t) for t in tips)
