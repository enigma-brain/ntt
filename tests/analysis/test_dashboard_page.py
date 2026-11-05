"""One page with every cohort in the same plots (marker per cohort), decoding hidden for a
flagged cohort, provenance footer, CLI."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pandas as pd
import pytest
import scripts.analysis.dashboard as D
from bokeh.models import Column, ColumnDataSource, Div, RadioButtonGroup

REPO = Path(__file__).resolve().parents[2]
FIX = REPO / "tests/data/dashboard"
NAMES = {
    "category_scores.csv": "SAMPLE_category_scores_aggregate.csv",
    "wide_metrics_table.csv": "SAMPLE_wide_metrics_table.csv",
    "metadata_rows.csv": "SAMPLE_metadata_rows.csv",
    "generalization_by_tier.csv": "SAMPLE_generalization_by_tier.csv",
    "generalization_by_metric.csv": "SAMPLE_generalization_by_metric.csv",
}
SUFFIX = "_1s"


def _cohort_dir(tmp_path: Path, name: str, suffix: str = "", edit=None) -> Path:
    d = tmp_path / name
    d.mkdir()
    for dst, src in NAMES.items():
        t = pd.read_csv(FIX / src)
        t["tokenizer"] = t["tokenizer"] + suffix
        if edit is not None:
            t = edit(dst, t)
        t.to_csv(d / dst, index=False)
    return d


def _two(tmp_path, *, hide_decoding=False, edit=None):
    return [
        D.load_cohort("200 ms", _cohort_dir(tmp_path, "a")),
        D.load_cohort(
            "1000 ms",
            _cohort_dir(tmp_path, "b", suffix=SUFFIX, edit=edit),
            hide_decoding=hide_decoding,
        ),
    ]


@pytest.fixture
def two(tmp_path):
    return _two(tmp_path)


def _is_second(name) -> bool:
    return str(name).split(" ")[0].endswith(SUFFIX) or "(1000 ms)" in str(name)


def _view(page, name):
    views = [m for m in page.references() if isinstance(m, Column) and m.name == name]
    assert len(views) == 1, name
    return views[0]


def test_one_page_holds_every_cohort(two):
    page = D.build_page(two)
    dev = _view(page, "dev_view")
    _view(page, "simple_view")
    assert not any(
        isinstance(m, RadioButtonGroup) and m.name == "cohort_select" for m in page.references()
    )
    names = {
        str(t)
        for m in dev.references()
        if isinstance(m, ColumnDataSource)
        for t in m.data.get("tokenizer", [])
    }
    assert any(n.endswith(SUFFIX) for n in names) and any(not n.endswith(SUFFIX) for n in names)


def test_markers_follow_the_cohort(two):
    dev = _view(D.build_page(two), "dev_view")
    marked = [
        m
        for m in dev.references()
        if isinstance(m, ColumnDataSource) and {"x", "marker"} <= set(m.data)
    ]
    assert len(marked) >= 2  # scatter + generalization points
    for src in marked:
        for name, marker in zip(src.data["tokenizer"], src.data["marker"], strict=True):
            assert marker == ("square" if _is_second(name) else "circle"), name


def test_ranking_marks_the_second_cohorts_bars(two):
    dev = _view(D.build_page(two), "dev_view")
    rank = next(
        m for m in dev.references() if isinstance(m, ColumnDataSource) and "hatch" in m.data
    )
    for name, size in zip(rank.data["tokenizer"], rank.data["msize"], strict=True):
        assert (size > 0) is _is_second(name), name


def test_radar_dashes_the_second_cohort(two):
    # drop the first cohort's cosmos model: the default dev overlay (first tokenizer of the
    # first two families) then holds one model of each cohort
    f = D.merge_cohorts(two).frame
    f = f[f["tokenizer"] != "cosmos_public_lc2_rfsq"].reset_index(drop=True)
    panel, _ = D._build_radar_tab(f, D._build_full_source(f), *D._build_filter_widgets(f))
    radar = next(
        m
        for m in panel.child.references()
        if isinstance(m, ColumnDataSource) and {"xs", "dash"} <= set(m.data)
    )
    shown = dict(zip(radar.data["tokenizer"], radar.data["dash"], strict=True))
    assert sorted(shown.values()) == ["dashed", "solid"]
    for name, dash in shown.items():
        assert dash == ("dashed" if _is_second(name) else "solid"), name


def _add_decoding_breakdown(name, table):
    if name == "category_scores.csv":
        table = table.assign(decoding_poyo=0.8, decoding_linear=0.2)
    return table


def test_hidden_decoding_cohort_is_left_out_of_decoding(tmp_path):
    merged = D.merge_cohorts(_two(tmp_path, hide_decoding=True, edit=_add_decoding_breakdown))
    f = merged.frame
    second = f["tokenizer"].str.endswith(SUFFIX)
    dec_cols = [
        "decoding",
        "decoding_poyo",
        "decoding_linear",
        *[c for c in f.columns if c.startswith(("decoding/", "downstream_decoding/"))],
    ]
    assert len(dec_cols) > 3
    assert f.loc[second, dec_cols].isna().all().all()
    assert f.loc[~second, "decoding"].notna().all()
    assert any("1000 ms models" in t and "decoding" in t for t in merged.caveats.cohort_wide)


def test_compression_scale_spans_all_cohorts(tmp_path):
    def tenfold(dst, t):
        if dst == "category_scores.csv":
            t["compression"] = t["compression"] * 10
        return t

    f = D.merge_cohorts(_two(tmp_path, edit=tenfold)).frame.set_index("tokenizer")
    assert f["compression_radar"].min() == 0.0 and f["compression_radar"].max() == 1.0
    assert f.loc["cosmos_public_lc2_rfsq", "compression_radar"] == 0.0
    vae = "wanx_A_vae_bce_dw_k1_lightdec_noncausal_anneal1e-6_sigreg_b1e-3_z32"
    assert f.loc[vae + SUFFIX, "compression_radar"] == 1.0
    assert 0.0 < f.loc[vae, "compression_radar"] < 1.0


def test_completeness_stays_per_cohort(tmp_path):
    def no_encoding(dst, t):
        if dst == "category_scores.csv":
            t["downstream_encoding"] = float("nan")
        return t

    f = D.merge_cohorts(_two(tmp_path, edit=no_encoding)).frame
    assert bool(f["is_complete"].all())  # encoding is N/A for the second cohort, not missing


def test_simple_names_tell_cohorts_of_one_family_apart(two):
    simple = D.derive_simple_frame(D.merge_cohorts(two).frame)
    assert "VAE" in set(simple["family_pretty"])
    assert "VAE (1000 ms)" in set(simple["family_pretty"])
    assert simple["display_name"].is_unique


def test_every_family_of_a_later_cohort_names_its_cohort(tmp_path):
    # a family only the second cohort has (here: the second cohort's cosmos models alone)
    def only_cosmos(dst, t):
        return t[t["tokenizer"].str.startswith("cosmos")] if "tokenizer" in t else t

    first = D.load_cohort(
        "200 ms",
        _cohort_dir(tmp_path, "a", edit=lambda d, t: t[~t["tokenizer"].str.startswith("cosmos")]),
    )
    second = D.load_cohort("1000 ms", _cohort_dir(tmp_path, "b", suffix=SUFFIX, edit=only_cosmos))
    f = D.merge_cohorts([first, second]).frame
    assert set(f.loc[f["cohort"] == "1000 ms", "family_label"]) == {"Cosmos (1000 ms)"}
    assert set(f.loc[f["cohort"] == "200 ms", "family_label"]) == {"VAE"}


def test_best_per_family_scored_over_axes_every_cohort_has(tmp_path):
    def no_encoding(dst, t):
        if dst == "category_scores.csv":
            t["downstream_encoding"] = float("nan")
        return t

    merged = D.merge_cohorts(_two(tmp_path, hide_decoding=True, edit=no_encoding))
    simple = D.derive_simple_frame(merged.frame)
    common = [c for c in D.COMPOSITE_SCORES if c not in ("decoding", "downstream_encoding")]
    for _, row in simple.iterrows():
        vals = [min(1.0, max(0.0, row[D._radar_col(simple, c)])) for c in common]
        assert row["family_score"] == pytest.approx(sum(vals) / len(vals)), row["tokenizer"]


def test_a_cohort_wide_caveat_names_its_cohort(tmp_path):
    cohorts = _two(tmp_path)
    b = tmp_path / "b"
    toks = pd.read_csv(b / "category_scores.csv")["tokenizer"]
    pd.DataFrame({"tokenizer": toks, "footnote": "own table", "notes": ""}).to_csv(
        b / "caveats.csv", index=False
    )
    cohorts[1] = D.load_cohort("1000 ms", b)
    merged = D.merge_cohorts(cohorts)
    assert merged.caveats.cohort_wide == ["1000 ms models: own table"]
    assert merged.caveats.per_model == {}
    text = " ".join(m.text for m in D.build_page(cohorts).references() if isinstance(m, Div))
    assert "1000 ms models: own table" in text


def test_radar_skips_a_missing_vertex(tmp_path):
    def drop_one(dst, t):
        if dst == "category_scores.csv":
            t.loc[0, "compression"] = float("nan")
        return t

    cohort = D.load_cohort("200 ms", _cohort_dir(tmp_path, "c", edit=drop_one))
    f = cohort.frame
    panel, _ = D._build_radar_tab(f, D._build_full_source(f), *D._build_filter_widgets(f))
    src = next(
        m for m in panel.child.references() if isinstance(m, ColumnDataSource) and "xs" in m.data
    )
    lengths = sorted(len(xs) for xs in src.data["xs"])
    assert lengths == [7, 8]  # 6 or 7 axes + the closing vertex
    assert all(math.isfinite(v) for xs in src.data["xs"] for v in xs)


def _layout_models(root):
    """Models reachable through visible layout children/child only (not CustomJS args)."""
    seen, stack = [], [root]
    while stack:
        m = stack.pop()
        seen.append(m)
        stack.extend(getattr(m, "children", []) or [])
        child = getattr(m, "child", None)
        if child is not None:
            stack.append(child)
    return seen


def test_cohort_filter_in_both_views(two):
    from bokeh.models import CheckboxGroup, CustomJS

    page = D.build_page(two)
    for name in ("simple_view", "dev_view"):
        view = _view(page, name)
        boxes = [
            m
            for m in _layout_models(view)
            if isinstance(m, CheckboxGroup) and m.labels == ["200 ms", "1000 ms"]
        ]
        assert len(boxes) == 1 and boxes[0].active == [0, 1], name
        filtering = [m for m in view.references() if isinstance(m, CustomJS) and "fam_w" in m.args]
        assert len(filtering) == 4  # radar, scatter, ranking, generalization
        assert all(m.args["coh_w"] is boxes[0] for m in filtering)
    assert "coh_w" in D._FILTER_JS and "d['cohort']" in D._FILTER_JS


def test_single_cohort_page_has_no_cohort_filter():
    from bokeh.models import CheckboxGroup

    frame = D.load_dashboard_frame(
        FIX / NAMES["category_scores.csv"],
        FIX / NAMES["wide_metrics_table.csv"],
        FIX / NAMES["metadata_rows.csv"],
    )
    page = D.build_dashboard(frame, D.build_metric_registry(frame))
    shown = [
        m
        for m in _layout_models(page)
        if isinstance(m, CheckboxGroup) and m.name == "cohort_filter" and m.visible
    ]
    assert shown == []


def test_duplicate_cohort_names_are_refused(two):
    with pytest.raises(ValueError, match="duplicate"):
        D.build_page([two[0], two[0]])


def test_load_cohort_requires_the_three_tables(tmp_path):
    d = _cohort_dir(tmp_path, "c")
    (d / "metadata_rows.csv").unlink()
    with pytest.raises(FileNotFoundError, match="metadata_rows.csv"):
        D.load_cohort("x", d)


def test_cli_writes_an_offline_page(tmp_path):
    a = _cohort_dir(tmp_path, "a")
    b = _cohort_dir(tmp_path, "b", suffix=SUFFIX)
    caveat = {"tokenizer": "cosmos_public_lc2_rfsq", "footnote": "a <caveat>", "notes": ""}
    pd.DataFrame([caveat]).to_csv(a / "caveats.csv", index=False)
    prov = tmp_path / "prov.json"
    prov.write_text(
        json.dumps({"ntt_commit": "abc123<x>", "cohorts": {"200 ms": {"aggregate_job": "uuid-1"}}})
    )
    out = tmp_path / "page.html"
    argv = ["--cohort", "200 ms", str(a), "--cohort", "1000 ms", str(b)]
    argv += ["--hide-decoding", "1000 ms", "--out", str(out), "--provenance", str(prov)]
    assert D.main(argv) == 0
    text = out.read_text(encoding="utf-8")
    # offline: no external <script src> / <link href> (BokehJS itself is inlined)
    assert "cdn.bokeh.org" not in text
    assert not re.search(r'<script[^>]+src="http', text)
    assert not re.search(r'<link[^>]+href="http', text)
    # escaped by us (&lt;) and once more by Bokeh's embedded-JSON escaping (&amp;lt;): the
    # footer shows the literal "abc123<x>", never markup
    assert "uuid-1" in text and "abc123&amp;lt;x&amp;gt;" in text


def test_texts_match_the_one_page_design(two):
    model_glossary = D._MODEL_GLOSSARY_MD.read_text(encoding="utf-8")
    assert "compared only with the other 1000 ms" not in model_glossary
    page = D.build_page(two)
    guides = [m.text for m in page.references() if isinstance(m, Div) and "<b>Radar</b>" in m.text]
    assert len(guides) == 2  # simple + dev guide lines
    # a cohort may lack axes (skipped on the radar), so the guide does not promise seven
    assert all("7 composite scores" not in g for g in guides)


def test_provenance_is_optional(tmp_path):
    a = _cohort_dir(tmp_path, "a")
    out = tmp_path / "page.html"
    assert D.main(["--cohort", "200 ms", str(a), "--out", str(out)]) == 0
    assert "Provenance" not in out.read_text(encoding="utf-8")  # no footer without it
    # any JSON object works (no assumption about how the inputs were produced)
    html = D._provenance_html({"source": "my laptop", "nested": {"run": 3}})
    assert "my laptop" in html and "nested · run" in html


def test_cli_refuses_hiding_decoding_of_an_unknown_cohort(tmp_path):
    a = _cohort_dir(tmp_path, "a")
    argv = ["--cohort", "200 ms", str(a), "--hide-decoding", "1000 ms"]
    with pytest.raises(SystemExit):
        D.main([*argv, "--out", str(tmp_path / "p.html")])
