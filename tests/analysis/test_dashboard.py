from pathlib import Path

import pytest
import scripts.analysis.dashboard as D

REPO = Path(__file__).resolve().parents[2]
# Tracked sample fixtures live under tests/data/ (see .gitignore !tests/data/**/*.csv);
# each is a 2-tokenizer slice of the corresponding build_aggregation_tables.py output.
FIXTURES = REPO / "tests/data/dashboard"
AGG = FIXTURES / "SAMPLE_category_scores_aggregate.csv"
WIDE = FIXTURES / "SAMPLE_wide_metrics_table.csv"
META = FIXTURES / "SAMPLE_metadata_rows.csv"
GEN = FIXTURES / "SAMPLE_generalization_by_tier.csv"
GEN_METRIC = FIXTURES / "SAMPLE_generalization_by_metric.csv"


@pytest.fixture(scope="module")
def frame():
    return D.load_dashboard_frame(AGG, WIDE, META, GEN)


def test_frame_has_two_sample_rows(frame):
    assert len(frame) == 2
    assert set(frame["tokenizer"]) == {
        "wanx_A_vae_bce_dw_k1_lightdec_noncausal_anneal1e-6_sigreg_b1e-3_z32",
        "cosmos_public_lc2_rfsq",
    }


def test_derived_columns_present(frame):
    for col in ["family", "color", "latent_type", "regime_short"]:
        assert col in frame.columns


def test_family_and_color(frame):
    row = frame.set_index("tokenizer").loc["cosmos_public_lc2_rfsq"]
    assert row["family"] == "cosmos"
    # colour comes from the tab10/tab20 palette, assigned by sorted family index
    fams = sorted(frame["family"].unique())
    expected = D._FAMILY_PALETTE[fams.index("cosmos") % len(D._FAMILY_PALETTE)]
    assert row["color"] == expected
    # the two sample families get distinct colours
    assert frame["color"].nunique() == frame["family"].nunique()


def test_regime_short(frame):
    by_tok = frame.set_index("tokenizer")
    assert (
        "per-neuron"
        in by_tok.loc[
            "wanx_A_vae_bce_dw_k1_lightdec_noncausal_anneal1e-6_sigreg_b1e-3_z32", "regime_short"
        ]
    )
    assert "population" in by_tok.loc["cosmos_public_lc2_rfsq", "regime_short"]


def test_composite_and_wide_columns_present(frame):
    assert "single_neuron_recon" in frame.columns  # composite
    assert "encoding/nc_r2" in frame.columns  # wide mean
    assert "latent_type" in frame.columns
    assert set(frame["latent_type"]) <= {"continuous", "discrete", "unknown"}


def test_regime_short_helper_handles_missing():
    assert D._regime_short("") == "unknown"
    assert D._regime_short(None) == "unknown"


def test_registry_composites_first(frame):
    reg = D.build_metric_registry(frame)
    assert reg[0]["group"] == "Composite"
    composite_values = {e["value"] for e in reg if e["group"] == "Composite"}
    assert "single_neuron_recon" in composite_values
    assert composite_values <= set(D.COMPOSITE_SCORES)


def test_registry_includes_wide_metric(frame):
    reg = D.build_metric_registry(frame)
    by_value = {e["value"]: e for e in reg}
    assert "encoding/nc_r2" in by_value
    assert by_value["encoding/nc_r2"]["group"] == "encoding"


def test_registry_excludes_suffix_columns(frame):
    reg = D.build_metric_registry(frame)
    assert not any("::" in e["value"] for e in reg)


def test_registry_has_std_flag(frame):
    reg = D.build_metric_registry(frame)
    # At least one wide metric in the sample has a ::std column.
    assert any(e["has_std"] for e in reg)


def test_registry_options_optgroup_shape(frame):
    reg = D.build_metric_registry(frame)
    opts = D.registry_options(reg)
    assert "Composite" in opts
    assert all(isinstance(v, list) for v in opts.values())
    # entries are (value, label) tuples
    assert all(len(t) == 2 for t in opts["Composite"])


def test_build_dashboard_returns_layout(frame):
    from bokeh.models import Tabs

    reg = D.build_metric_registry(frame)
    layout = D.build_dashboard(frame, reg)
    tabs = [m for m in layout.references() if isinstance(m, Tabs)]
    assert len(tabs) == 2  # simple + dev
    for t in tabs:
        assert [p.title for p in t.tabs] == [
            "Radar",
            "Scatter",
            "Ranking",
            "Generalization",
            "Caveats",
            "Models",
            "Glossary",
        ]


def test_glossary_tab_has_nonempty_content(frame):
    from bokeh.models import Column, Div, Tabs

    reg = D.build_metric_registry(frame)
    layout = D.build_dashboard(frame, reg)
    dev_view = next(
        m for m in layout.references() if isinstance(m, Column) and m.name == "dev_view"
    )
    tabs = next(m for m in dev_view.references() if isinstance(m, Tabs))
    glossary = next(p for p in tabs.tabs if p.title == "Glossary")
    assert isinstance(glossary.child, Div)
    text = glossary.child.text
    assert text
    assert "nc_r2" in text
    assert "participation_ratio" in text


def test_metric_glossaries_match_main_scoring():
    # whitespace-normalized: the markdown wraps phrases across lines
    full = " ".join(D._GLOSSARY_MD.read_text(encoding="utf-8").split())
    simple = " ".join(D._SIMPLE_GLOSSARY_MD.read_text(encoding="utf-8").split())
    assert "Scored over the reconstruction metrics only" in full  # the score's scope
    for text in (full, simple):
        assert "Without latent-quality metrics" not in text  # no scope toggle any more
        assert "reconstruction metrics only" in text and "all three tiers share" in text
    assert "its own brainset" in full  # the matched baseline
    assert (
        "Compression (log, relative)" in full and "lowest and highest rate of all models" in simple
    )
    assert D.SCORE_LABELS["decoding"] in full and "mean R²" not in simple


def test_md_to_html_basic_constructs():
    html = D._md_to_html("# Title\n\nSome **bold** and `code`.\n\n- one\n- two\n\n---\n")
    assert "<h1>Title</h1>" in html
    assert "<strong>bold</strong>" in html
    assert "<code>code</code>" in html
    assert "<ul>" in html and "<li>one</li>" in html
    assert "<hr>" in html


def test_view_sources_initialized_nonempty(frame):
    from bokeh.models import ColumnDataSource

    reg = D.build_metric_registry(frame)
    layout = D.build_dashboard(frame, reg)
    sources = [m for m in layout.references() if isinstance(m, ColumnDataSource)]
    # at least the full source + 3 view sources, and the scatter view has points
    assert any(len(s.data.get("x", [])) == 2 for s in sources)  # 2 sample tokenizers


def test_color_latent_column(frame):
    # the "Colour by: Latent type" toggle reads a precomputed per-row color column
    assert "color_latent" in frame.columns
    by_tok = frame.set_index("tokenizer")
    cont = by_tok.loc[
        "wanx_A_vae_bce_dw_k1_lightdec_noncausal_anneal1e-6_sigreg_b1e-3_z32", "color_latent"
    ]
    disc = by_tok.loc["cosmos_public_lc2_rfsq", "color_latent"]
    assert cont == D._LATENT_COLORS["continuous"]
    assert disc == D._LATENT_COLORS["discrete"]
    assert cont != disc


def test_direction_arrows_from_authoritative_map():
    # composites are all higher-is-better (ntt.evaluation.radar)
    assert D._direction_arrow("single_neuron_recon", "Composite") == "↑"
    # wide metrics resolve against BaseMetric.direction (NOT a name heuristic):
    # a per-dim robustness NMSE has no 'error' in its name but is lower-is-better
    assert (
        D._direction_arrow("robustness/robustness_input_per_dim/permute_neurons", "robustness")
        == "↓"
    )
    assert D._direction_arrow("temporal_fidelity/binned_rate_difference_5ms", "tf") == "↓"
    assert D._direction_arrow("encoding/nc_r2", "encoding") == "↑"
    # M2 (latent-noise) columns resolve an arrow too, like every other axis.
    assert (
        D._direction_arrow("robustness_latent/spike_detection_f1/eps_0.1", "robustness_latent")
        == "↑"
    )
    assert (
        D._direction_arrow("robustness_latent/binned_rate_difference/eps_0.1", "robustness_latent")
        == "↓"
    )
    # unknown metric -> no arrow (never guess)
    assert D._direction_arrow("madeup/not_a_real_metric", "madeup") == ""


def test_every_wide_axis_label_carries_a_direction_arrow(frame):
    # The dashboard's own registry is the real consumer: if any pickable metric
    # resolves to no direction, its axis label silently loses the ↑/↓ the top
    # banner promises. (Composites are handled by the group=="Composite" branch.)
    missing = [
        e["value"]
        for e in D.build_metric_registry(frame)
        if e["group"] != "Composite" and not D._direction_arrow(e["value"], e["group"])
    ]
    assert not missing, f"metrics with no direction arrow: {missing}"


def test_norm_columns_available_for_poisson_view(frame):
    # the scatter/ranking "Poisson-norm" toggle reads ::norm columns from the frame;
    # they must be present (and they hold the Poisson normalized_score)
    norm_cols = [c for c in frame.columns if c.endswith("::norm")]
    assert norm_cols, "no ::norm columns — the Poisson-normalized view would be empty"


def test_export_html_is_self_contained(frame, tmp_path):
    reg = D.build_metric_registry(frame)
    layout = D.build_dashboard(frame, reg)
    out = D.export_html(layout, tmp_path / "dash.html")
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    # offline: BokehJS inlined, no CDN reference
    assert "cdn.bokeh.org" not in html
    # interactivity shipped as JS, and data embedded
    assert "CustomJS" in html
    assert "cosmos_public_lc2_rfsq" in html
    # INLINE bokehjs makes a large file
    assert len(html) > 300_000


def test_gen_tier_columns_merged(frame):
    for tier in ("xsession", "xsubject", "xsource"):
        assert f"gen_tier/{tier}" in frame.columns
    row = frame.set_index("tokenizer").loc["cosmos_public_lc2_rfsq"]
    assert row["gen_tier/xsource"] == 0.50
    assert row["gen_tier/xsession::n"] == 48


def test_gen_lane_deterministic_and_bounded(frame):
    assert "gen_lane" in frame.columns
    assert frame["gen_lane"].between(-0.22, 0.22).all()
    # stable per tokenizer across a re-load
    again = D.load_dashboard_frame(AGG, WIDE, META, GEN)
    a = frame.set_index("tokenizer")["gen_lane"]
    b = again.set_index("tokenizer")["gen_lane"]
    assert (a.sort_index().values == b.sort_index().values).all()


def test_load_frame_without_gen_csv_still_works():
    # 3-arg call (no gen csv) must not break; gen columns simply absent
    f = D.load_dashboard_frame(AGG, WIDE, META)
    assert "gen_tier/xsource" not in f.columns
    assert "gen_lane" in f.columns  # lane is always present (computed from tokenizer)


def test_registry_excludes_gen_tier_columns(frame):
    reg = D.build_metric_registry(frame)
    assert not any(e["value"].startswith("gen_tier/") for e in reg)


def test_generalization_tab_present_with_box_and_points(frame):
    from bokeh.models import ColumnDataSource, Tabs

    reg = D.build_metric_registry(frame)
    layout = D.build_dashboard(frame, reg)
    tabs = next(m for m in layout.references() if isinstance(m, Tabs))
    titles = [p.title for p in tabs.tabs]
    assert "Generalization" in titles
    # the box source has exactly the 3 tiers; the points source has 2 toks x 3 tiers
    srcs = [m for m in layout.references() if isinstance(m, ColumnDataSource)]
    box = [s for s in srcs if set(s.data.get("tier", [])) == {"xsession", "xsubject", "xsource"}]
    assert box, "no per-tier box source found"
    assert any(len(s.data.get("y", [])) == 6 for s in srcs)  # 2 sample toks x 3 tiers


@pytest.fixture(scope="module")
def gen_by_metric():
    return D.load_generalization_by_metric(GEN_METRIC)


def test_load_generalization_by_metric(gen_by_metric):
    assert gen_by_metric is not None
    assert {
        "tokenizer",
        "metric",
        "category",
        "ret_xsession",
        "ret_xsubject",
        "ret_xsource",
    } <= set(gen_by_metric.columns)
    assert D.load_generalization_by_metric(None) is None


def test_generalization_tab_has_metric_select(frame, gen_by_metric):
    from bokeh.models import Select, Tabs

    reg = D.build_metric_registry(frame)
    layout = D.build_dashboard(frame, reg, gen_by_metric=gen_by_metric)
    tabs = next(m for m in layout.references() if isinstance(m, Tabs))
    assert "Generalization" in [p.title for p in tabs.tabs]
    selects = [m for m in layout.references() if isinstance(m, Select)]

    # the gen-tab metric Select has the "__all__" default among its option values.
    # options may be a flat list[(value,label)|str] or a dict {group: [(value,label),...]}
    # (Bokeh optgroup form) — scan both shapes for the value.
    def _opt_values(opts):
        items = opts.values() if isinstance(opts, dict) else [opts]
        for lst in items:
            for o in lst:
                yield o[0] if isinstance(o, tuple) else o

    vals = [v for s in selects for v in _opt_values(s.options)]
    assert "__all__" in vals


def test_build_dashboard_without_gen_by_metric_still_works(frame):
    # gen_by_metric defaults to None -> tab builds, no metric Select required
    reg = D.build_metric_registry(frame)
    layout = D.build_dashboard(frame, reg)
    from bokeh.models import Tabs

    tabs = next(m for m in layout.references() if isinstance(m, Tabs))
    assert "Generalization" in [p.title for p in tabs.tabs]


# ── block display labels (§ robustness vs robustness_latent) ───────────────────
def test_registry_carries_display_label_separate_from_group_key(frame):
    reg = D.build_metric_registry(frame)
    wide = [e for e in reg if e["group"] != "Composite"]
    # `group` stays the raw block key (logic branches on it); `group_label` is display-only
    assert all("group_label" in e for e in wide)
    rob = next(e for e in wide if e["group"] == "robustness")  # present in the fixture
    assert rob["group"] == "robustness"
    assert rob["group_label"] == "input corruption, M1"
    assert "input corruption, M1" in D._axis_label(rob)


def test_registry_options_are_keyed_by_display_label(frame):
    reg = D.build_metric_registry(frame)
    opts = D.registry_options(reg)
    assert "Composite" in opts  # composites have no group_label -> fall back to `group`
    assert "input corruption, M1" in opts
    assert "robustness" not in opts


# ── M2 comparability flag ──────────────────────────────────────────────────────
def test_m2_comparable_flags_non_included_provenance(tmp_path):
    import pandas as pd

    agg = pd.read_csv(AGG)
    agg["m2_provenance"] = ["included", "excluded_missing_key"]
    tmp_agg = tmp_path / "agg_m2.csv"
    agg.to_csv(tmp_agg, index=False)
    f = D.load_dashboard_frame(tmp_agg, WIDE, META, GEN).set_index("tokenizer")
    assert bool(f.loc[agg.loc[0, "tokenizer"], "m2_comparable"]) is True
    assert bool(f.loc[agg.loc[1, "tokenizer"], "m2_comparable"]) is False


def test_ranking_bars_are_unhatched_for_an_unaffected_default_metric(frame):
    from bokeh.models import ColumnDataSource

    # the ranking tab opens on single_neuron_recon, which the M2 caveat does not touch —
    # hatching there would smear the warning across metrics it does not apply to
    reg = D.build_metric_registry(frame)
    layout = D.build_dashboard(frame, reg)
    hatched = [
        s for s in layout.references() if isinstance(s, ColumnDataSource) and "hatch" in s.data
    ]
    assert hatched, "ranking source should carry a hatch column"
    assert all(set(s.data["hatch"]) <= {" "} for s in hatched)


def test_guide_banner_explains_the_warning_marker(frame):
    from bokeh.models import Div

    reg = D.build_metric_registry(frame)
    layout = D.build_dashboard(frame, reg)
    text = " ".join(m.text for m in layout.references() if isinstance(m, Div) and m.text)
    assert D.M2_CAVEAT in text


def test_ranking_shows_metric_coverage_against_the_cohort(frame):
    from bokeh.models import Div

    # the wide table is a union across tokenizer families, so a short bar list is usually
    # coverage, not a broken load — say so rather than leaving the reader to guess
    reg = D.build_metric_registry(frame)
    layout = D.build_dashboard(frame, reg)
    texts = [m.text for m in layout.references() if isinstance(m, Div) and m.text]
    assert any("2 of 2 filtered tokenizers have this metric" in t for t in texts)


def test_scatter_reports_n_against_the_cohort(tmp_path):
    """The scatter's r/p line needs >=3 finite points, which the 2-row fixture cannot
    give — so widen it to 3 tokenizers here rather than skip the assertion."""
    import pandas as pd
    from bokeh.models import Div

    paths = {}
    for key, src in (("agg", AGG), ("wide", WIDE), ("meta", META), ("gen", GEN)):
        t = pd.read_csv(src)
        extra = t.iloc[[0]].copy()
        extra["tokenizer"] = "duplicate_for_n3"
        out = tmp_path / f"{key}.csv"
        pd.concat([t, extra], ignore_index=True).to_csv(out, index=False)
        paths[key] = out

    f3 = D.load_dashboard_frame(paths["agg"], paths["wide"], paths["meta"], paths["gen"])
    assert len(f3) == 3
    layout = D.build_dashboard(f3, D.build_metric_registry(f3))
    # match the stats line specifically — "Pearson r" alone also hits the glossary/banner
    stats = [
        m.text
        for m in layout.references()
        if isinstance(m, Div) and m.text and m.text.startswith("<b>n = ")
    ]
    assert stats, "scatter should render its r/p stats line"
    # n carries a denominator, so a sparse metric reads as coverage, not a broken load
    assert all("n = 3 of 3" in t for t in stats)


def test_registry_skips_non_numeric_wide_columns(frame):
    values = {e["value"] for e in D.build_metric_registry(frame)}
    assert "encoding/split" not in values
    assert "decoding/complete" not in values
    assert "encoding/nc_r2" in values


def test_frame_carries_only_the_columns_the_views_use(frame):
    assert "source_uris" not in frame.columns
    assert "decoding_members" not in frame.columns
    assert "generalization_source" not in frame.columns
    assert "m2_provenance" in frame.columns


def test_m2_hatch_keys_on_excluded_provenance_only(tmp_path):
    import pandas as pd

    agg = pd.read_csv(AGG)
    agg["m2_provenance"] = ["off", "excluded_no_report"]
    tmp_agg = tmp_path / "agg_m2.csv"
    agg.to_csv(tmp_agg, index=False)
    f = D.load_dashboard_frame(tmp_agg, WIDE, META, GEN).set_index("tokenizer")
    assert bool(f.loc[agg.loc[0, "tokenizer"], "m2_comparable"]) is True  # off = M2 off for all
    assert bool(f.loc[agg.loc[1, "tokenizer"], "m2_comparable"]) is False


def test_generalization_callback_attached_once_per_filter(frame):
    from bokeh.models import MultiChoice

    view = D._build_view(frame, D.build_metric_registry(frame), simple=False)
    fam_w = next(
        m for m in view.references() if isinstance(m, MultiChoice) and m.title == "Family"
    )
    callbacks = fam_w.js_property_callbacks["change:value"]
    assert len(callbacks) == len({id(c) for c in callbacks})


def test_tokenizer_missing_from_side_tables_still_renders(tmp_path):
    import pandas as pd

    paths = {}
    for key, src in (("wide", WIDE), ("meta", META), ("gen", GEN)):
        t = pd.read_csv(src)
        out = tmp_path / f"{key}.csv"
        t[t["tokenizer"] != "cosmos_public_lc2_rfsq"].to_csv(out, index=False)
        paths[key] = out
    f = D.load_dashboard_frame(AGG, paths["wide"], paths["meta"], paths["gen"])
    assert set(f["tokenizer"]) == {
        "wanx_A_vae_bce_dw_k1_lightdec_noncausal_anneal1e-6_sigreg_b1e-3_z32",
        "cosmos_public_lc2_rfsq",
    }
    row = f.set_index("tokenizer").loc["cosmos_public_lc2_rfsq"]
    assert row["latent_type"] == "unknown"
    D.build_dashboard(f, D.build_metric_registry(f))  # must not raise


def test_log_minmax_edges():
    import math

    import pandas as pd

    out = D._log_minmax(pd.Series([0.1, 1.0, 10.0, None, 0.0, -2.0]))
    assert list(out[:3]) == [0.0, 0.5, 1.0]
    assert all(math.isnan(v) for v in out[3:])
    assert list(D._log_minmax(pd.Series([4.0, 4.0]))) == [1.0, 1.0]
    assert list(D._log_minmax(pd.Series([9.0]))) == [1.0]
    assert D._log_minmax(pd.Series([None, None], dtype=float)).isna().all()


def test_frame_has_cohort_relative_compression(frame):
    by_tok = frame.set_index("tokenizer")["compression_radar"]
    assert by_tok["cosmos_public_lc2_rfsq"] == 0.0  # the lower rate of the two
    assert by_tok["wanx_A_vae_bce_dw_k1_lightdec_noncausal_anneal1e-6_sigreg_b1e-3_z32"] == 1.0


def test_radar_places_mapped_compression_but_shows_raw(frame):
    import math

    from bokeh.models import ColumnDataSource, Label

    full = D._build_full_source(frame)
    fam_w, reg_w, lat_w = D._build_filter_widgets(frame)
    panel, _ = D._build_radar_tab(frame, full, fam_w, reg_w, lat_w)
    src = next(
        m
        for m in panel.child.references()
        if isinstance(m, ColumnDataSource) and "v_compression" in m.data
    )
    k = D.COMPOSITE_SCORES.index("compression")
    raw = frame.set_index("tokenizer")["compression"]
    mapped = frame.set_index("tokenizer")["compression_radar"]
    assert len(src.data["tokenizer"]) == 2
    for name, xs, ys, shown in zip(
        src.data["tokenizer"],
        src.data["xs"],
        src.data["ys"],
        src.data["v_compression"],
        strict=True,
    ):
        assert math.hypot(xs[k], ys[k]) == pytest.approx(mapped[name])
        assert shown == D._fmt(raw[name])
    labels = [m.text for m in panel.child.references() if isinstance(m, Label)]
    assert "Compression (log, relative)" in labels


def test_family_score_uses_the_radar_scale(frame):
    simple = D.derive_simple_frame(frame).set_index("tokenizer")
    for _, row in simple.iterrows():
        vals = [min(1.0, max(0.0, row[D._radar_col(simple, c)])) for c in D.COMPOSITE_SCORES]
        assert row["family_score"] == pytest.approx(sum(vals) / len(vals))


def test_generalization_tab_has_no_scope_toggle(frame):
    from bokeh.models import RadioButtonGroup

    full = D._build_full_source(frame)
    panel, cb = D._build_generalization_tab(frame, full, *D._build_filter_widgets(frame))
    assert not any(isinstance(m, RadioButtonGroup) for m in panel.child.references())
    assert "scope_w" not in cb.args and "SCOPE" not in cb.code


def test_generalization_tab_states_what_the_tiers_pool(frame):
    from bokeh.models import Div

    def note(simple):
        df = D.derive_simple_frame(frame) if simple else frame
        full = D._build_full_source(df)
        panel, _ = D._build_generalization_tab(
            df, full, *D._build_filter_widgets(df), simple=simple
        )
        return " ".join(m.text for m in panel.child.references() if isinstance(m, Div))

    dev = note(False)
    assert "reconstruction metrics" in dev.lower() and "all three tiers share" in dev
    # no metric-count range (it varies per model; the hover shows each model's n)
    assert "48–50" not in dev and "per tokenizer)" not in dev
    assert "shortlist" in note(True)
