# tests/analysis/test_dashboard_simple.py
from pathlib import Path

import pytest
import scripts.analysis.dashboard as D

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests/data/dashboard"
AGG = FIXTURES / "SAMPLE_category_scores_aggregate.csv"
WIDE = FIXTURES / "SAMPLE_wide_metrics_table.csv"
META = FIXTURES / "SAMPLE_metadata_rows.csv"
GEN = FIXTURES / "SAMPLE_generalization_by_tier.csv"


def _layout_models(root):
    """Models reachable through visible layout children/child only (NOT CustomJS args)."""
    seen, stack = [], [root]
    while stack:
        m = stack.pop()
        seen.append(m)
        stack.extend(getattr(m, "children", []) or [])
        child = getattr(m, "child", None)
        if child is not None:
            stack.append(child)
    return seen


@pytest.fixture(scope="module")
def frame():
    return D.load_dashboard_frame(AGG, WIDE, META, GEN)


VAE = "wanx_A_vae_bce_dw_k1_lightdec_noncausal_anneal1e-6_sigreg_b1e-3_z32"


def test_simple_frame_keeps_every_complete_model(frame):
    simple = D.derive_simple_frame(frame)
    # no regime filter any more: the population cosmos model is in the simple view too
    assert set(simple["family"]) == {"wanx_A_vae", "cosmos"}


def test_simple_frame_added_columns(frame):
    simple = D.derive_simple_frame(frame).set_index("tokenizer")
    for col in ["family_score", "family_pretty", "is_family_best", "display_name"]:
        assert col in simple.columns
    row = simple.loc[VAE]
    assert row["family_pretty"] == "VAE"
    assert bool(row["is_family_best"]) is True
    assert row["display_name"] == "VAE #1"
    assert simple.loc["cosmos_public_lc2_rfsq", "display_name"] == "Cosmos #1"


def test_score_no_model_has_is_na_not_incomplete(tmp_path):
    import pandas as pd
    from bokeh.models import ColumnDataSource

    agg = pd.read_csv(AGG)
    agg["downstream_encoding"] = float("nan")  # e.g. the 1000 ms-token cohort: no encoding
    p = tmp_path / "agg.csv"
    agg.to_csv(p, index=False)
    f = D.load_dashboard_frame(p, WIDE, META, GEN)
    assert bool(f["is_complete"].all())
    radar_panel, _ = D._build_radar_tab(f, D._build_full_source(f), *D._build_filter_widgets(f))
    src = next(
        m
        for m in radar_panel.child.references()
        if isinstance(m, ColumnDataSource) and "v_single_neuron_recon" in m.data
    )
    assert "v_downstream_encoding" not in src.data  # N/A for the cohort: no radar axis
    agg.loc[0, "compression"] = float("nan")  # one model missing a cohort axis IS incomplete
    agg.to_csv(p, index=False)
    f = D.load_dashboard_frame(p, WIDE, META, GEN).set_index("tokenizer")
    assert bool(f.loc[agg.loc[0, "tokenizer"], "is_complete"]) is False
    assert bool(f.loc[agg.loc[1, "tokenizer"], "is_complete"]) is True


def test_family_pretty_covers_the_public_cohort_families():
    from scripts.analysis.data import FAMILY_PRETTY

    assert {"wanx_B", "cosmos", "PCA", "LDNS"} <= set(FAMILY_PRETTY)


def test_simple_registry_restricts_wide_metrics(frame):
    simple = D.derive_simple_frame(frame)
    reg = D.build_metric_registry(simple, simple=True)
    wide = {e["value"] for e in reg if e["group"] != "Composite"}
    # only allow-listed wide metrics that actually exist in the frame
    assert wide <= set(D.SIMPLE_METRICS)
    assert "encoding/nc_r2" in wide  # present in the sample wide table
    # a non-shortlisted raw metric is excluded even though it exists in the columns
    assert "single_unit_statistics/isi_divergence" not in wide
    # composites still lead
    assert reg[0]["group"] == "Composite"


def test_simple_radar_overlay_lists_families(frame):
    from bokeh.models import MultiChoice

    simple = D.derive_simple_frame(frame)
    full = D._build_full_source(simple)
    fam_w, reg_w, lat_w = D._build_filter_widgets(simple)
    panel, _cb = D._build_radar_tab(simple, full, fam_w, reg_w, lat_w, simple=True)
    mcs = [m for m in panel.child.references() if isinstance(m, MultiChoice)]
    overlay = next(m for m in mcs if m.title and "overlay" in m.title.lower())
    # options are pretty family names, not raw tokenizer strings
    assert "VAE" in overlay.options
    assert not any("wanx_A" in o for o in overlay.options)


def test_simple_scatter_uses_display_names_no_symlog(frame):
    from bokeh.models import CheckboxGroup

    simple = D.derive_simple_frame(frame)
    reg = D.build_metric_registry(simple, simple=True)
    full = D._build_full_source(simple)
    fam_w, reg_w, lat_w = D._build_filter_widgets(simple)
    panel, cb = D._build_scatter_tab(simple, reg, full, fam_w, reg_w, lat_w, simple=True)
    # the scatter's own VIEW source carries display names, not raw tokenizers
    names = list(cb.args["scatter_src"].data["tokenizer"])
    assert any(str(n).startswith("VAE #") for n in names)
    assert not any("wanx_A" in str(n) for n in names)
    # the symlog "Scale" toggle is absent from the VISIBLE layout in simple mode
    # (walk layout children only — the inert widget still lives in CustomJS args)
    cbs = [m for m in _layout_models(panel.child) if isinstance(m, CheckboxGroup)]
    assert not any("log" in lbl.lower() for c in cbs for lbl in c.labels)


def test_simple_ranking_factors_are_display_names(frame):
    from bokeh.models import FactorRange

    simple = D.derive_simple_frame(frame)
    reg = D.build_metric_registry(simple, simple=True)
    full = D._build_full_source(simple)
    fam_w, reg_w, lat_w = D._build_filter_widgets(simple)
    panel, _cb = D._build_ranking_tab(simple, reg, full, fam_w, reg_w, lat_w, simple=True)
    frs = [m for m in panel.child.references() if isinstance(m, FactorRange)]
    factors = [str(f) for fr in frs for f in fr.factors]
    assert any(f.startswith("VAE #") for f in factors)
    assert not any("wanx_A" in f for f in factors)


def _gm(simple):
    """Minimal per-metric generalization frame for the sample cohort tokenizer."""
    import pandas as pd

    return pd.DataFrame(
        {
            "tokenizer": [simple.iloc[0]["tokenizer"]],
            "metric": ["spike_detection_f1"],
            "category": ["temporal_fidelity"],
            "ret_xsession": [0.9],
            "ret_xsubject": [0.8],
            "ret_xsource": [0.7],
        }
    )


def _hover_labels(panel):
    """Tooltip field labels across a tab's HoverTools — handles both the list form
    (scatter/ranking) and the HTML-string form (generalization)."""
    from bokeh.models import HoverTool

    labels = []
    for m in panel.child.references():
        if isinstance(m, HoverTool):
            if isinstance(m.tooltips, list):
                labels += [t[0] for t in m.tooltips]
            elif isinstance(m.tooltips, str) and "full_model_tag" in m.tooltips:
                labels.append("full_model_tag")
    return labels


def test_simple_family_labels_are_pretty_and_full_tag_present(frame):
    """In the simple view the scatter/ranking/generalization VIEW sources carry the
    pretty family name (matching the radar tab) plus a ``full_tag`` column holding the
    raw tokenizer, and each hover exposes a ``full_model_tag`` field."""
    simple = D.derive_simple_frame(frame)
    reg = D.build_metric_registry(simple, simple=True)
    full = D._build_full_source(simple)
    fam_w, reg_w, lat_w = D._build_filter_widgets(simple)
    raw = set(simple["tokenizer"])

    sp, scb = D._build_scatter_tab(simple, reg, full, fam_w, reg_w, lat_w, simple=True)
    rp, rcb = D._build_ranking_tab(simple, reg, full, fam_w, reg_w, lat_w, simple=True)
    gp, gcb = D._build_generalization_tab(
        simple, full, fam_w, reg_w, lat_w, _gm(simple), simple=True
    )

    # pretty family labels (no raw stems) in the sources that drive legend + hover
    assert set(scb.args["scatter_src"].data["family"]) == {"VAE", "Cosmos"}
    assert set(gcb.args["pts_src"].data["family"]) == {"VAE", "Cosmos"}

    # full_tag column carries the raw tokenizer in all three view sources
    for src_key, cb in (("scatter_src", scb), ("rank_src", rcb), ("pts_src", gcb)):
        data = cb.args[src_key].data
        assert "full_tag" in data
        assert set(data["full_tag"]) == raw

    # every hover exposes the full_model_tag field
    for panel in (sp, rp, gp):
        assert "full_model_tag" in _hover_labels(panel)


def test_dev_family_labels_stay_raw_and_no_full_tag(frame):
    """Dev mode is unchanged: raw family strings, no full_tag column, no
    full_model_tag hover field (the dev view already shows the raw tokenizer name)."""
    reg = D.build_metric_registry(frame)
    full = D._build_full_source(frame)
    fam_w, reg_w, lat_w = D._build_filter_widgets(frame)

    sp, scb = D._build_scatter_tab(frame, reg, full, fam_w, reg_w, lat_w, simple=False)
    rp, rcb = D._build_ranking_tab(frame, reg, full, fam_w, reg_w, lat_w, simple=False)
    gp, gcb = D._build_generalization_tab(
        frame, full, fam_w, reg_w, lat_w, _gm(frame), simple=False
    )

    assert "wanx_A_vae" in set(scb.args["scatter_src"].data["family"])
    for src_key, cb in (("scatter_src", scb), ("rank_src", rcb), ("pts_src", gcb)):
        assert "full_tag" not in cb.args[src_key].data
    for panel in (sp, rp, gp):
        assert "full_model_tag" not in _hover_labels(panel)


def test_simple_gen_metric_select_restricted(frame):
    import pandas as pd
    from bokeh.models import Select

    simple = D.derive_simple_frame(frame)
    full = D._build_full_source(simple)
    fam_w, reg_w, lat_w = D._build_filter_widgets(simple)
    gm = pd.DataFrame(
        {
            "tokenizer": [simple.iloc[0]["tokenizer"]] * 2,
            "metric": ["spike_detection_f1", "isi_divergence"],
            "category": ["temporal_fidelity", "single_unit_statistics"],
            "ret_xsession": [0.9, 0.8],
            "ret_xsubject": [0.8, 0.7],
            "ret_xsource": [0.7, 0.6],
        }
    )
    panel, _cb = D._build_generalization_tab(simple, full, fam_w, reg_w, lat_w, gm, simple=True)
    sel = next(m for m in panel.child.references() if isinstance(m, Select))
    flat = []
    for v in sel.options.values() if isinstance(sel.options, dict) else [sel.options]:
        flat += [t[0] if isinstance(t, tuple) else t for t in v]
    assert "__all__" in flat
    assert "spike_detection_f1" in flat  # shortlisted recon metric -> kept
    assert "isi_divergence" not in flat  # not shortlisted -> dropped


def test_simple_glossary_only_shortlist(frame):
    from bokeh.models import Div

    panel = D._build_glossary_tab(D._SIMPLE_GLOSSARY_MD)
    assert isinstance(panel.child, Div)
    text = panel.child.text
    assert "spike detection f1" in text or "spike_detection_f1" in text
    assert "Generalization" in text
    # not-shown raw metrics are absent from the simple glossary
    assert "isi_divergence" not in text
    assert "subspace_overlap" not in text


def test_dual_layout_toggle(frame):
    from bokeh.models import CheckboxGroup, Column, Tabs

    reg = D.build_metric_registry(frame)
    layout = D.build_dashboard(frame, reg)
    tabs = [m for m in layout.references() if isinstance(m, Tabs)]
    assert len(tabs) == 2  # simple + dev
    dev_view = next(
        m for m in layout.references() if isinstance(m, Column) and m.name == "dev_view"
    )
    simple_view = next(
        m for m in layout.references() if isinstance(m, Column) and m.name == "simple_view"
    )
    assert dev_view.visible is False
    assert simple_view.visible is True
    toggle = next(
        m for m in layout.references() if isinstance(m, CheckboxGroup) and m.name == "dev_toggle"
    )
    assert toggle.active == []


def test_build_glossary_tab_custom_title():
    panel = D._build_glossary_tab(D._MODEL_GLOSSARY_MD, title="Models")
    from bokeh.models import Div

    assert panel.title == "Models"
    assert isinstance(panel.child, Div)
    assert "wanx_A_vqpt" in panel.child.text  # a dev-only stem present


def test_model_glossary_scope():
    dev = D._build_glossary_tab(D._MODEL_GLOSSARY_MD, title="Models").child.text
    simple = D._build_glossary_tab(D._SIMPLE_MODEL_GLOSSARY_MD, title="Models").child.text
    # both cover every family of the public cohorts, NOT the no-score baselines
    stems = ["wanx_A_vae", "wanx_A_dprior", "wanx_B", "cosmos", "compressed_sensing", "calm"]
    for stem in [*stems, "PCA", "LDNS"]:
        assert stem in dev
    for excluded in ["Poisson", "unigram", "bpe"]:
        assert excluded not in dev
    pretty = ["VQ-PT", "CaLM", "Diffusion-Prior", "Cosmos", "PCA", "LDNS", "VAE-B"]
    for name in pretty:
        assert name in simple
    for off in ["Poisson", "unigram"]:
        assert off not in simple


def test_models_tab_present_both_views(frame):
    from bokeh.models import Column, Tabs

    reg = D.build_metric_registry(frame)
    layout = D.build_dashboard(frame, reg)
    for name in ("simple_view", "dev_view"):
        view = next(m for m in layout.references() if isinstance(m, Column) and m.name == name)
        tabs = next(m for m in view.references() if isinstance(m, Tabs))
        titles = [p.title for p in tabs.tabs]
        assert titles == [
            "Radar",
            "Scatter",
            "Ranking",
            "Generalization",
            "Caveats",
            "Models",
            "Glossary",
        ]


# ── "include partial evaluations" filter ─────────────────────────────────────────


def test_is_complete_column_from_load_frame(frame, tmp_path):
    import pandas as pd

    # both sample tokenizers have all 7 composites filled
    assert "is_complete" in frame.columns
    assert bool(frame["is_complete"].all())
    # blanking one composite cell makes that tokenizer incomplete
    agg = pd.read_csv(AGG)
    agg.loc[0, "generalization"] = None
    tmp_agg = tmp_path / "agg_partial.csv"
    agg.to_csv(tmp_agg, index=False)
    f2 = D.load_dashboard_frame(tmp_agg, WIDE, META, GEN)
    by_tok = f2.set_index("tokenizer")
    assert bool(by_tok.loc[agg.loc[0, "tokenizer"], "is_complete"]) is False
    assert bool(by_tok.loc[agg.loc[1, "tokenizer"], "is_complete"]) is True


def test_derive_simple_frame_is_complete_only(frame):
    # baseline: the complete VAE tokenizer is in the simple cohort
    assert "VAE" in set(D.derive_simple_frame(frame)["family_pretty"])
    # mark it incomplete -> derive_simple_frame drops it before ranking (no vanished-family gaps)
    f2 = frame.copy()
    f2.loc[f2["family"] == "wanx_A_vae", "is_complete"] = False
    simple2 = D.derive_simple_frame(f2)
    assert list(simple2["family"]) == ["cosmos"]


def test_partial_toggle_defaults_dev_shown_simple_hidden(frame):
    from bokeh.models import CheckboxGroup, Column

    # factory defaults: dev ON (include partial), simple OFF (complete-only)
    assert D._build_partial_toggle(simple=False).active == [0]
    assert D._build_partial_toggle(simple=True).active == []

    reg = D.build_metric_registry(frame)
    layout = D.build_dashboard(frame, reg)
    dev = next(m for m in layout.references() if isinstance(m, Column) and m.name == "dev_view")
    simple = next(
        m for m in layout.references() if isinstance(m, Column) and m.name == "simple_view"
    )

    def visible_toggles(view):
        return [
            m
            for m in _layout_models(view)
            if isinstance(m, CheckboxGroup) and "include partial evaluations" in m.labels
        ]

    dev_t = visible_toggles(dev)
    assert len(dev_t) == 1 and dev_t[0].active == [0]  # shown + ON in dev
    assert visible_toggles(simple) == []  # hidden in simple


def test_partial_filter_wired_into_every_tab(frame):
    # the shared guard lives in _FILTER_JS (applies to all tabs at once)
    assert "incl_partial" in D._FILTER_JS
    assert "d['is_complete']" in D._FILTER_JS
    # and the shared widget reaches every tab's callback
    reg = D.build_metric_registry(frame)
    full = D._build_full_source(frame)
    fam_w, reg_w, lat_w = D._build_filter_widgets(frame)
    partial_w = D._build_partial_toggle(simple=False)
    _, scb = D._build_scatter_tab(
        frame, reg, full, fam_w, reg_w, lat_w, simple=False, partial_w=partial_w
    )
    _, rcb = D._build_ranking_tab(
        frame, reg, full, fam_w, reg_w, lat_w, simple=False, partial_w=partial_w
    )
    _, rdcb = D._build_radar_tab(
        frame, full, fam_w, reg_w, lat_w, simple=False, partial_w=partial_w
    )
    _, gcb = D._build_generalization_tab(
        frame, full, fam_w, reg_w, lat_w, None, simple=False, partial_w=partial_w
    )
    for cb in (scb, rcb, rdcb, gcb):
        assert cb.args.get("partial_w") is partial_w
