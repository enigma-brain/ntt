"""Interactive Bokeh dashboard substrate for tokenizer-eval results.

Pure-Python data layer + Bokeh builders + standalone-HTML export.  Reuses the
analysis-tool vocabulary (composite names, labels, family inference, palette)
so the dashboard stays consistent with the matplotlib CLI tools.

Needs the ``dashboard`` dependency group (bokeh), which a plain ``uv sync`` installs.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

# `scripts.analysis.*` is a repo directory, not an installed package. It imports only when
# the repo root is on sys.path — which happens implicitly under pytest, and when the CWD
# *is* the repo root, but not for `python scripts/analysis/x.py` from anywhere else. Adding
# it explicitly makes this runnable from any working directory. The imports below must
# therefore follow this block; `noqa: I001` keeps the import sorter from hoisting them
# above it and breaking that.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: I001
import pandas as pd
from bokeh.embed import file_html
from bokeh.layouts import column, row
from bokeh.models import (
    CheckboxGroup,
    ColumnDataSource,
    CustomJS,
    Div,
    FactorRange,
    FixedTicker,
    HoverTool,
    Label,
    MultiChoice,
    RadioButtonGroup,
    Range1d,
    Select,
    TabPanel,
    Tabs,
    Whisker,
)
from bokeh.plotting import figure
from bokeh.resources import INLINE

from scripts.analysis.data import (
    COMPOSITE_SCORES,
    DECODING_BREAKDOWNS,
    SCORE_LABELS,
    SCORE_SHORT,
    SIMPLE_METRICS,
    WIDE_METRIC_LABELS,
    block_label,
    family_pretty,
    infer_family,
)

from ntt.evaluation.core.types import MetricDirection
from ntt.evaluation.metric_directions import resolve_direction

# When a tokenizer's M2 (latent-noise) eval yields no usable normalization key, the latent
# bucket is still scored — on its remaining, harsher members. The number that comes out
# reads plausibly but is NOT comparable to its peers', so it gets marked rather than shown
# bare. `m2_provenance` (from category_scores.csv) records this per tokenizer; only the
# composite below is affected, since the raw `robustness_latent/*` columns are simply
# absent when there is no M2 data.
M2_SENSITIVE_SCORE = "latent_structure_and_robustness"
M2_CAVEAT = "M2 missing — this latent score is not comparable to the others"
_M2_HATCH = "/"  # bar hatch for flagged tokenizers (" " = no hatch)

# Family colour palette: tab10 (dark, saturated) first — so up to 10 families get
# the classic matplotlib qualitative colours — then tab20's lighter variants for
# any beyond 10. Assigned deterministically by sorted family name.
_FAMILY_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
    "#c49c94",
    "#f7b6d2",
    "#c7c7c7",
    "#dbdb8d",
    "#9edae5",
]

# Colors for the "Colour by: Latent type" toggle (scatter/ranking).
_LATENT_COLORS = {
    "continuous": "#1f77b4",
    "discrete": "#d62728",
    "unknown": "#9e9e9e",
}


def _regime_short(regime) -> str:
    """Compact label for the regime filter widget.

    '100ms→1tok (per-neuron, 1ms input bin)' -> 'per-neuron · 100ms→1tok'
    '1000ms→500tok (population, ...)'        -> 'population · 1000ms→500tok'
    """
    if not isinstance(regime, str) or not regime.strip():
        return "unknown"
    head = regime.split("(")[0].strip()
    if "population" in regime:
        kind = "population"
    elif "per-neuron" in regime or "per neuron" in regime:
        kind = "per-neuron"
    else:
        kind = ""
    return f"{kind} · {head}" if kind else head


def _gen_lane(tokenizer: str) -> float:
    """Deterministic per-tokenizer x-offset in [-0.22, 0.22] for the
    Generalization tab's jittered points. Computed once here (Python) and read
    by the tab's CustomJS via the ``gen_lane`` column, so the initial render and
    every filtered re-render place each tokenizer in the same lane (no JS/Python
    hash to keep in sync). FNV-1a-ish 32-bit hash -> stable, well-spread."""
    h = 2166136261
    for ch in str(tokenizer):
        h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
    return ((h % 1000) / 1000.0 - 0.5) * 0.44


# The only category_scores.csv columns the views read; the rest (provenance, source URIs,
# per-category settings) would be embedded in every view's data source for nothing.
AGG_COLUMNS = ["tokenizer", *COMPOSITE_SCORES, *DECODING_BREAKDOWNS, "m2_provenance"]
# Metric-picker group of the two family scores the decoding composite averages.
DECODING_BREAKDOWN_GROUP = "Decoding breakdown"


# Radar position column per composite. Compression is a rate (input bits / latent bits),
# unbounded above (up to ~60 in the public cohorts), so the radar shows it on a log scale
# min-max mapped over all models on the page; hovers keep the raw rate.
RADAR_COLUMN = {"compression": "compression_radar"}
RADAR_LABELS = {**SCORE_SHORT, "compression": "Compression (log, relative)"}


def _radar_col(df: pd.DataFrame, score: str) -> str:
    col = RADAR_COLUMN.get(score)
    return col if col is not None and col in df.columns else score


def _log_minmax(values: pd.Series) -> pd.Series:
    """log10, then min-max to [0, 1] over this frame's values (all models shown).
    Missing or non-positive -> NaN; a single distinct value -> 1.0."""
    v = pd.to_numeric(values, errors="coerce").astype(float)
    logv = pd.Series(np.log10(v.where(v > 0)), index=v.index)
    lo, hi = logv.min(), logv.max()
    if pd.isna(lo):
        return logv
    if hi == lo:
        return logv.where(logv.isna(), 1.0)
    return (logv - lo) / (hi - lo)


def _gen_tiers(gen: pd.DataFrame) -> pd.DataFrame:
    """``tokenizer`` + the per-tier columns ``gen_tier/<tier>[::mean|::n]`` of
    ``generalization_by_tier.csv`` (as written by build_generalization_tables.py)."""
    return gen.loc[:, ["tokenizer", *[c for c in gen.columns if c.startswith("gen_tier/")]]]


# Point marker per cohort, in --cohort order (the first cohort's are circles).
COHORT_MARKERS = ("circle", "square", "triangle", "diamond", "inverted_triangle")
_MARKER_SYMBOL = {
    "circle": "●",
    "square": "■",
    "triangle": "▲",
    "diamond": "◆",
    "inverted_triangle": "▼",
}


def _cohort_legend(df: pd.DataFrame) -> str:
    """' | ● 200 ms · ■ 1000 ms' when several cohorts share the page, else ''."""
    pairs = df.loc[:, ["cohort", "marker"]].drop_duplicates()
    if len(pairs) < 2:
        return ""
    items = " · ".join(
        f"{_MARKER_SYMBOL.get(m, m)} {html.escape(str(c))}"
        for c, m in zip(pairs["cohort"], pairs["marker"], strict=True)
    )
    return f" &nbsp;|&nbsp; <b>Cohorts</b>: {items} (radar: 2nd cohort dashed)"


CAVEAT_MARK = "†"


@dataclass(frozen=True)
class Caveats:
    """Footnotes of one cohort. ``per_model`` holds the model-specific ones (a footnote
    every model carries is lifted into ``cohort_wide`` and shown once, as a banner);
    ``notes`` holds provenance notes, shown in dev mode only."""

    per_model: dict[str, list[str]]
    cohort_wide: list[str]
    notes: dict[str, str]
    supplied: bool = True  # False: no caveats.csv was given (unknown, not "none")


NO_CAVEATS = Caveats({}, [], {}, supplied=False)


def load_caveats(csv: str | Path | None, tokenizers: Iterable[str]) -> Caveats:
    """Read ``caveats.csv`` (``tokenizer, footnote[, notes]``, one row per footnote; a row
    with an empty footnote carries notes only). Rows of other tokenizers are ignored."""
    toks = list(dict.fromkeys(str(t) for t in tokenizers))
    if csv is None or not Path(csv).exists():
        return NO_CAVEATS
    cav = pd.read_csv(csv, dtype=str, keep_default_na=False)
    if "notes" not in cav.columns:
        cav["notes"] = ""
    per: dict[str, list[str]] = {}
    notes: dict[str, str] = {}
    for tok, footnote, note in zip(cav["tokenizer"], cav["footnote"], cav["notes"], strict=True):
        if tok not in toks:
            continue
        if footnote.strip() and footnote not in per.setdefault(tok, []):
            per[tok].append(footnote)
        if note.strip():
            notes[tok] = note
    shared = (
        [f for f in per.get(toks[0], []) if all(f in per.get(t, []) for t in toks)]
        if len(toks) > 1
        else []
    )
    per_model = {t: [f for f in fs if f not in shared] for t, fs in per.items()}
    return Caveats({t: fs for t, fs in per_model.items() if fs}, shared, notes)


def apply_caveats(df: pd.DataFrame, caveats: Caveats) -> pd.DataFrame:
    """Add ``caveat_mark``, ``caveats`` (hover text) and ``label`` (tokenizer + mark)."""
    out = df.copy()
    fns = out["tokenizer"].map(lambda t: caveats.per_model.get(str(t), []))
    out["caveat_mark"] = fns.map(lambda f: CAVEAT_MARK if f else "")
    out["caveats"] = fns.map(lambda f: " ".join(f"{CAVEAT_MARK} {x}" for x in f) if f else "—")
    out["label"] = out["tokenizer"].astype(str) + out["caveat_mark"].map(
        lambda m: f" {m}" if m else ""
    )
    return out


def cohort_scores(df: pd.DataFrame) -> list[str]:
    """The composites some tokenizer of ``df`` (one cohort) has, in radar order. A score no
    model has (e.g. encoding for the 1000 ms-token cohort) is N/A for the cohort: not
    an axis, not pickable, not required for completeness."""
    return [c for c in COMPOSITE_SCORES if c in df.columns and df[c].notna().any()]


def _is_metric_column(series: pd.Series) -> bool:
    """A wide-table column is a pickable metric only if it is numeric (not bool):
    ``encoding/split`` (a string) and ``decoding/complete`` (a flag) are not."""
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def load_dashboard_frame(
    aggregate_csv: str | Path,
    wide_csv: str | Path,
    metadata_rows_csv: str | Path,
    gen_tiers_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Merge the result CSVs into one tidy per-tokenizer DataFrame.

    All four are outputs of ``scripts/analysis/build_aggregation_tables.py``:
    ``category_scores.csv``, ``wide_metrics_table.csv``, ``metadata_rows.csv``
    (the tracker rows, carrying the per-tokenizer metadata the result tables do
    not) and ``generalization_by_tier.csv``.

    ``gen_tiers_csv`` (optional) adds the per-tier generalization columns
    ``gen_tier/<tier>`` (+ ``::mean``/``::n``) for the Generalization tab.
    """
    agg = pd.read_csv(aggregate_csv)
    wide = pd.read_csv(wide_csv)
    metadata = pd.read_csv(metadata_rows_csv)
    agg = agg.loc[:, [c for c in AGG_COLUMNS if c in agg.columns]]
    metric_cols = [c for c in wide.columns if c != "tokenizer" and _is_metric_column(wide[c])]
    wide = wide.loc[:, ["tokenizer", *metric_cols]]

    meta_cols = [
        c
        for c in ["tokenizer", "latent_type", "tokenization_regime", "variant"]
        if c in metadata.columns
    ]
    meta = metadata.loc[:, meta_cols].drop_duplicates(subset="tokenizer")

    df = agg.merge(wide, on="tokenizer", how="left")
    df = df.merge(meta, on="tokenizer", how="left")

    df["family"] = df["tokenizer"].map(infer_family)
    df["latent_type"] = df["latent_type"] if "latent_type" in df.columns else "unknown"
    df["latent_type"] = df["latent_type"].fillna("unknown").replace("", "unknown")
    df["tokenization_regime"] = (
        df["tokenization_regime"] if "tokenization_regime" in df.columns else ""
    )
    df["tokenization_regime"] = df["tokenization_regime"].fillna("").replace("", "unknown")
    df["regime_short"] = df["tokenization_regime"].map(_regime_short)

    # Deterministic family colours: sorted families mapped onto the tab10/tab20
    # palette — stable across runs, distinct between families.
    fams = sorted(df["family"].unique())
    color_map = {f: _FAMILY_PALETTE[i % len(_FAMILY_PALETTE)] for i, f in enumerate(fams)}
    df["color"] = df["family"].map(color_map.get)
    df["color_latent"] = df["latent_type"].map(
        lambda v: _LATENT_COLORS.get(str(v), _LATENT_COLORS["unknown"])
    )

    if gen_tiers_csv is not None and Path(gen_tiers_csv).exists():
        gen = _gen_tiers(pd.read_csv(gen_tiers_csv))
        df = df.merge(gen, on="tokenizer", how="left")
    df["gen_lane"] = df["tokenizer"].map(_gen_lane)

    # `m2_provenance` rides in from category_scores.csv. "excluded_*" means this tokenizer's
    # latent bucket was scored without its M2 member while its peers kept it — see
    # M2_SENSITIVE_SCORE. "off" (M2 disabled for the whole table) is comparable.
    df["m2_comparable"] = ~df["m2_provenance"].fillna("").astype(str).str.startswith("excluded_")

    if "compression" in df.columns:
        df["compression_radar"] = _log_minmax(df["compression"])

    # A tokenizer is "complete" when it has every composite that SOME tokenizer of this
    # cohort has: a score no model has (e.g. encoding for the 1000 ms-token cohort) is N/A
    # for the cohort, not missing. Drives the simple view and the dev "include partial
    # evaluations" filter.
    comp_cols = cohort_scores(df)
    df["is_complete"] = df[comp_cols].notna().all(axis=1) if comp_cols else False
    # Cohort columns (merge_cohorts overwrites them when several cohorts share a page).
    df["cohort"] = ""
    df["marker"] = COHORT_MARKERS[0]
    df["dash"] = "solid"
    df["family_label"] = df["family"].map(family_pretty)
    # label/caveats columns always exist; build_dashboard re-applies the real caveats
    return apply_caveats(df, NO_CAVEATS)


def derive_simple_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Filter ``df`` to the simple view (complete-only, see ``is_complete``) and add the
    columns the simple view needs:

    - ``family_score``  — mean of the radar axes every cohort has (the family-best heuristic)
    - ``family_pretty`` — pretty family label
    - ``is_family_best``— True for the top-``family_score`` row within each family
    - ``display_name``  — ``"<pretty family> #<rank>"`` (rank 1 = family-best)
    """
    # The simple view is permanently complete-only (the "include partial evaluations"
    # toggle is hidden + off): drop incomplete tokenizers BEFORE ranking, so best-per-family
    # and the "#1/#2" ranks are computed over complete tokenizers only (no vanished families).
    sub = df[df["is_complete"]].copy() if "is_complete" in df.columns else df.copy()
    comp = [c for c in COMPOSITE_SCORES if c in sub.columns]
    if "cohort" in sub.columns and sub["cohort"].nunique() > 1:
        # Rank on the axes every cohort has: a cohort with fewer axes (no encoding, decoding
        # hidden) would otherwise be averaged over a different, easier set.
        comp = [c for c in comp if all(g[c].notna().any() for _, g in sub.groupby("cohort"))]
    # Same scale as the radar: compression mapped, every axis clamped to [0, 1].
    radar = pd.DataFrame({c: sub[_radar_col(sub, c)].clip(0.0, 1.0) for c in comp})
    sub["family_score"] = radar.mean(axis=1, skipna=True) if comp else float("nan")
    # family_label = pretty family name, plus the cohort for families present in several
    # cohorts (merge_cohorts), so best-per-family never mixes cohorts.
    sub["family_pretty"] = (
        sub["family_label"] if "family_label" in sub.columns else sub["family"].map(family_pretty)
    )
    # rank within family by family_score desc; NaN scores sort last (pandas default).
    sub = sub.sort_values(
        ["family_pretty", "family_score"],
        ascending=[True, False],
        kind="stable",
        na_position="last",
    )
    rank = sub.groupby("family_pretty").cumcount() + 1
    sub["is_family_best"] = rank == 1
    sub["display_name"] = sub["family_pretty"] + " #" + rank.astype(str)
    if "caveat_mark" in sub.columns:
        sub["display_name"] += sub["caveat_mark"].map(lambda m: f" {m}" if m else "")
    return sub.reset_index(drop=True)


def load_generalization_by_metric(csv: str | Path | None) -> pd.DataFrame | None:
    """Load the long per-metric generalization table (one row per tokenizer x metric:
    ``tokenizer, metric, category, ret_xsession, ret_xsubject, ret_xsource``).
    Returns ``None`` when ``csv`` is ``None`` or missing — the Generalization tab
    then omits the metric selector and shows only the aggregate view."""
    if csv is None or not Path(csv).exists():
        return None
    return pd.read_csv(csv)


def build_metric_registry(df: pd.DataFrame, simple: bool = False) -> list[dict]:
    """One entry per pickable metric (composites first, then wide blocks).

    ``simple=True`` restricts the wide entries to ``SIMPLE_METRICS`` (the
    PI-friendly shortlist); composites are always kept.
    """
    entries: list[dict] = []
    for c in cohort_scores(df):
        if c in df.columns:
            entries.append(
                {
                    "value": c,
                    "label": SCORE_LABELS.get(c, c),
                    "group": "Composite",
                    "has_std": False,
                }
            )
    for c in DECODING_BREAKDOWNS:
        if c in df.columns and df[c].notna().any():
            entries.append(
                {
                    "value": c,
                    "label": SCORE_LABELS[c],
                    "group": DECODING_BREAKDOWN_GROUP,
                    "has_std": False,
                }
            )
    allow = set(SIMPLE_METRICS) if simple else None
    for c in df.columns:
        if "/" not in c or "::" in c or c in COMPOSITE_SCORES or c.startswith("gen_tier/"):
            continue
        if allow is not None and c not in allow:
            continue
        block, metric = c.split("/", 1)
        entries.append(
            {
                "value": c,
                "label": WIDE_METRIC_LABELS.get(c, metric.replace("_", " ")),
                # `group` stays the raw block key (a stable identifier other code branches
                # on); `group_label` is the display string. Keeping them apart means
                # renaming a block for readers cannot change any logic.
                "group": block,
                "group_label": block_label(block),
                "has_std": f"{c}::std" in df.columns,
            }
        )
    return entries


def registry_options(registry: list[dict]) -> dict[str, list[str | tuple[str, str]]]:
    """Group registry entries into Bokeh Select optgroup options, keyed by display name."""
    groups: dict[str, list[str | tuple[str, str]]] = {}
    for e in registry:
        groups.setdefault(e.get("group_label") or e["group"], []).append((e["value"], e["label"]))
    return groups


def _direction_arrow(value: str, group: str) -> str:
    """'↑' higher-is-better / '↓' lower-is-better, resolved from the AUTHORITATIVE
    source (``BaseMetric.direction`` on the registered metric class, reached through
    the config display-name index — not a heuristic). Returns '' (no arrow) when a
    metric's direction is unknown. The 7 composites are all higher-is-better
    (ntt.evaluation.radar), and so are the decoding breakdown scores.

    The block is taken from ``value`` rather than from ``group``: ``group`` is a
    display grouping that callers pass loosely, whereas the resolver keys off the
    column's real block (only ``encoding``/``decoding`` actually depend on it)."""
    if group in ("Composite", DECODING_BREAKDOWN_GROUP):
        return "↑"
    if "/" not in value:
        return ""
    block, part = value.split("/", 1)
    d = resolve_direction(block, part)
    if d is MetricDirection.HIGHER_IS_BETTER:
        return "↑"
    return "↓" if d is MetricDirection.LOWER_IS_BETTER else ""


def _axis_label(entry: dict) -> str:
    """Label for axes/titles: block category for wide metrics + a direction arrow
    ('↑'/'↓' = higher/lower is better; explained in the top banner). Composites keep
    their already-descriptive label (the category is the composite itself), as do the
    decoding breakdown scores."""
    group = entry.get("group", "")
    base = (
        entry["label"]
        if group in ("Composite", DECODING_BREAKDOWN_GROUP)
        else f"{entry['label']} ({entry.get('group_label') or group.replace('_', ' ')})"
    )
    arrow = _direction_arrow(entry["value"], group)
    return f"{base} {arrow}" if arrow else base


# Shared JS prelude: from the filter widgets, build `visible` = row indices.
_FILTER_JS = """
const d = full.data;
const N = d['tokenizer'].length;
const fam_sel = fam_w.value, reg_sel = reg_w.value;
const lat_sel = lat_w.active.map(i => lat_w.labels[i]);
// "include partial evaluations": checked => keep tokenizers missing >=1 composite.
const incl_partial = partial_w.active.includes(0);
const coh_sel = coh_w.active.map(i => coh_w.labels[i]);
// "selected = included": widgets default to ALL options, so the default shows
// everything; removing a value excludes it (clearing a filter shows nothing).
const visible = [];
for (let i = 0; i < N; i++) {
  if (!fam_sel.includes(d['family'][i])) continue;
  if (!reg_sel.includes(d['regime_short'][i])) continue;
  if (!lat_sel.includes(d['latent_type'][i])) continue;
  if (!incl_partial && !d['is_complete'][i]) continue;
  if (!coh_sel.includes(d['cohort'][i])) continue;
  visible.push(i);
}
"""


def _build_full_source(df: pd.DataFrame) -> ColumnDataSource:
    """One source holding every tokenizer x every column (CustomJS filters it)."""
    # ColumnDataSource keeps pandas NaN as-is; the CustomJS checks isNaN.
    return ColumnDataSource(df)


def _build_filter_widgets(
    df: pd.DataFrame,
) -> tuple[MultiChoice, MultiChoice, CheckboxGroup]:
    # Default to ALL options selected ("selected = included" — see _FILTER_JS), so the
    # default cohort is everything and the user removes values to exclude them.
    fam_opts = sorted(df["family"].unique())
    reg_opts = sorted(df["regime_short"].unique())
    fam_w = MultiChoice(title="Family", options=fam_opts, value=list(fam_opts), width=300)
    reg_w = MultiChoice(title="Regime", options=reg_opts, value=list(reg_opts), width=340)
    lat_labels = sorted(df["latent_type"].unique())
    lat_w = CheckboxGroup(labels=lat_labels, active=list(range(len(lat_labels))))
    return fam_w, reg_w, lat_w


def _build_cohort_toggle(df: pd.DataFrame) -> CheckboxGroup:
    """The Cohort filter (reads via ``_FILTER_JS``): every cohort ticked; hidden when the
    page holds one cohort."""
    labels = [str(c) for c in dict.fromkeys(df["cohort"])] if "cohort" in df else [""]
    return CheckboxGroup(
        labels=labels,
        active=list(range(len(labels))),
        name="cohort_filter",
        visible=len(labels) > 1,
    )


def _build_partial_toggle(simple: bool = False) -> CheckboxGroup:
    """The "include partial evaluations" filter (reads via ``_FILTER_JS``).

    Checked (active ``[0]``) keeps tokenizers missing ≥1 composite. Dev defaults ON
    and shows the toggle; the simple view forces it OFF (complete-only) and hides it.
    """
    return CheckboxGroup(
        labels=["include partial evaluations"],
        active=([] if simple else [0]),
        width=220,
    )


def _finite(v) -> bool:
    try:
        return v is not None and not (isinstance(v, float) and math.isnan(v))
    except TypeError:
        return False


def _finite_str(v) -> bool:
    """True when ``v`` is a non-empty, non-NaN string-able category label."""
    return isinstance(v, str) and bool(v.strip())


def _fmt(v) -> str:
    """Format a metric value for hover display; non-finite -> em dash."""
    return f"{float(v):.3f}" if _finite(v) else "—"


try:  # exact two-sided p-value for the scatter tab's INITIAL label (JS computes updates)
    from scipy.stats import pearsonr as _scipy_pearsonr
except Exception:  # pragma: no cover - scipy is normally present in this env
    _scipy_pearsonr = None


def _fmt_p(p) -> str:
    """Format a p-value: scientific for tiny values, 4 dp otherwise; '—' if NaN."""
    try:
        p = float(p)
    except (TypeError, ValueError):
        return "—"
    if p != p:  # NaN
        return "—"
    return f"{p:.2e}" if p < 1e-4 else f"{p:.4f}"


# JS helper rendering [{label, color}, ...] as an HTML swatch legend. Kept in
# sync with the Python `_legend_html` mirror (used for the initial render before
# any interaction, so the exported HTML's legends are not empty).
_LEGEND_JS_FN = """
function legendHTML(items){
  let h = '<div style="font:11px/1.5 sans-serif; max-height:480px; overflow:auto;">';
  for (const it of items){
    h += '<div style="margin-bottom:2px; padding-left:16px; text-indent:-16px;">'
       + '<span style="display:inline-block;width:11px;height:11px;vertical-align:middle;'
       + 'background:' + it.color + ';margin-right:5px;border:1px solid #999;"></span>'
       + '<span style="vertical-align:middle; word-break:break-all;">'
       + it.label + '</span></div>';
  }
  return h + '</div>';
}
"""


# JS computing the exact two-sided Pearson p-value (regularized incomplete beta —
# matches scipy.stats.pearsonr), so the scatter tab shows p next to r with no kernel.
_PVAL_JS_FN = """
function _gammaln(x){
  const c=[76.18009172947146,-86.50532032941677,24.01409824083091,
           -1.231739572450155,0.1208650973866179e-2,-0.5395239384953e-5];
  let y=x, tmp=x+5.5; tmp-=(x+0.5)*Math.log(tmp); let ser=1.000000000190015;
  for(let j=0;j<6;j++){ y+=1; ser+=c[j]/y; }
  return -tmp+Math.log(2.5066282746310005*ser/x);
}
function _betacf(a,b,x){
  const MAXIT=200, EPS=3e-12, FPMIN=1e-300;
  let qab=a+b, qap=a+1, qam=a-1, c=1, d=1-qab*x/qap;
  if(Math.abs(d)<FPMIN) d=FPMIN; d=1/d; let h=d;
  for(let m=1;m<=MAXIT;m++){
    let m2=2*m, aa=m*(b-m)*x/((qam+m2)*(a+m2));
    d=1+aa*d; if(Math.abs(d)<FPMIN) d=FPMIN;
    c=1+aa/c; if(Math.abs(c)<FPMIN) c=FPMIN; d=1/d; h*=d*c;
    aa=-(a+m)*(qab+m)*x/((a+m2)*(qap+m2));
    d=1+aa*d; if(Math.abs(d)<FPMIN) d=FPMIN;
    c=1+aa/c; if(Math.abs(c)<FPMIN) c=FPMIN; d=1/d;
    let del=d*c; h*=del; if(Math.abs(del-1)<EPS) break;
  }
  return h;
}
function _betai(a,b,x){
  if(x<=0) return 0; if(x>=1) return 1;
  const bt=Math.exp(_gammaln(a+b)-_gammaln(a)-_gammaln(b)+a*Math.log(x)+b*Math.log(1-x));
  return (x<(a+1)/(a+b+2)) ? bt*_betacf(a,b,x)/a : 1-bt*_betacf(b,a,1-x)/b;
}
function pearsonP(r,n){
  if(n<3 || isNaN(r)) return NaN;
  if(r>=1 || r<=-1) return 0;
  const df=n-2, t2=r*r*df/(1-r*r);
  return _betai(df/2, 0.5, df/(df+t2));   // two-sided
}
function fmtP(p){ return isNaN(p) ? "—" : (p<1e-4 ? p.toExponential(2) : p.toFixed(4)); }
"""


def _legend_html(items: list[tuple[str, str]]) -> str:
    """HTML swatch legend from (label, color) pairs — mirrors JS `legendHTML`."""
    rows = "".join(
        '<div style="margin-bottom:2px; padding-left:16px; text-indent:-16px;">'
        '<span style="display:inline-block;width:11px;height:11px;vertical-align:middle;'
        f'background:{color};margin-right:5px;border:1px solid #999;"></span>'
        f'<span style="vertical-align:middle; word-break:break-all;">{label}</span></div>'
        for label, color in items
    )
    return f'<div style="font:11px/1.5 sans-serif; max-height:480px; overflow:auto;">{rows}</div>'


def _dedup_items(pairs) -> list[tuple[str, str]]:
    """Deduplicate (label, color) pairs by label, preserving first-seen order.

    Casts both elements to ``str`` (pandas row values arrive loosely typed).
    """
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for label, color in pairs:
        lab = str(label)
        if lab not in seen:
            seen.add(lab)
            out.append((lab, str(color)))
    return out


# Repo root: dashboard.py lives at <repo>/scripts/analysis/, so parents[2] is the
# repo root (mirrors how the notebook computes REPO).
_REPO = Path(__file__).resolve().parents[2]
_GLOSSARY_MD = _REPO / "docs" / "metric_glossary.md"
_SIMPLE_GLOSSARY_MD = _REPO / "docs" / "metric_glossary_simple.md"
_MODEL_GLOSSARY_MD = _REPO / "docs" / "model_glossary.md"
_SIMPLE_MODEL_GLOSSARY_MD = _REPO / "docs" / "model_glossary_simple.md"


def _md_to_html(md: str) -> str:
    """Tiny self-contained Markdown→HTML converter (no external dependency).

    Handles only the subset the glossary uses: ATX headings (#..####),
    unordered bullet lists (- ...), paragraphs, and the inline spans
    **bold**, `code`, and ⚠ (passed through). Horizontal rules (---) become
    <hr>. This intentionally avoids adding the `markdown` package as a dep.
    """
    import html as _html
    import re as _re

    def inline(text: str) -> str:
        # Escape first, then re-introduce our inline markup on the escaped text.
        text = _html.escape(text)
        # `code` -> <code> (escaped already; backticks survive escaping)
        text = _re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
        # **bold** -> <strong>
        text = _re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
        return text

    out: list[str] = []
    para: list[str] = []
    in_list = False

    def flush_para() -> None:
        if para:
            out.append("<p>" + inline(" ".join(para)) + "</p>")
            para.clear()

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            out.append("</ul>")
            in_list = False

    for raw_line in md.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            flush_para()
            close_list()
            continue
        if stripped.startswith("#"):
            flush_para()
            close_list()
            level = len(stripped) - len(stripped.lstrip("#"))
            level = min(max(level, 1), 6)
            text = stripped[level:].strip()
            out.append(f"<h{level}>{inline(text)}</h{level}>")
            continue
        if stripped in ("---", "***", "___"):
            flush_para()
            close_list()
            out.append("<hr>")
            continue
        if stripped.startswith("- ") or stripped.startswith("* "):
            flush_para()
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append("<li>" + inline(stripped[2:].strip()) + "</li>")
            continue
        # plain text line: accumulate into the current paragraph
        if in_list:
            # continuation of the preceding bullet (wrapped line)
            out[-1] = out[-1][:-5] + " " + inline(stripped) + "</li>"
        else:
            para.append(stripped)

    flush_para()
    close_list()
    return "\n".join(out)


def _build_glossary_tab(md_path: Path = _GLOSSARY_MD, title: str = "Glossary") -> TabPanel:
    """Static, offline-safe Glossary tab built from a markdown file (dev or simple).

    The markdown is read and converted to HTML at *build* time and embedded in a
    scrollable ``Div`` — no runtime fetch, CSP-safe in the exported HTML. Falls
    back to a short notice (never raises) if the glossary file is missing.
    """
    if md_path.exists():
        body = _md_to_html(md_path.read_text(encoding="utf-8"))
    else:
        body = f"<p><em>Glossary file not found</em> (expected at <code>{md_path}</code>).</p>"
    html = (
        '<div style="font:13px/1.55 sans-serif; color:#222; max-width:100%; '
        'padding:4px 16px 16px 4px;">' + body + "</div>"
    )
    # Natural height + stretch_width: the long glossary uses the full tab width and
    # scrolls within the central container's own (capped) scroll — no inner scrollbar,
    # so we avoid a double scrollbar on this tab.
    div = Div(text=html, sizing_mode="stretch_width")
    return TabPanel(child=div, title=title)


def _build_scatter_tab(
    df: pd.DataFrame,
    registry: list[dict],
    full: ColumnDataSource,
    fam_w: MultiChoice,
    reg_w: MultiChoice,
    lat_w: CheckboxGroup,
    simple: bool = False,
    partial_w: CheckboxGroup | None = None,
    coh_w: CheckboxGroup | None = None,
) -> tuple[TabPanel, CustomJS]:
    partial_w = partial_w if partial_w is not None else _build_partial_toggle(simple)
    coh_w = coh_w if coh_w is not None else _build_cohort_toggle(df)
    opts = registry_options(registry)
    namecol = "display_name" if simple else "label"
    # Legend + hover "family" field use the pretty family name in the simple view
    # (matching the radar tab); dev mode keeps the raw family string.
    famcol = "family_pretty" if simple else "family"
    label_map = {e["value"]: _axis_label(e) for e in registry}
    label_map_js = json.dumps(label_map)
    x0, y0 = "single_neuron_recon", "decoding"
    xsel = Select(title="X metric", options=opts, value=x0)
    ysel = Select(title="Y metric", options=opts, value=y0)

    # initial view (Python mirror of the JS update for default metrics)
    xs, ys, names, fams, lats, cols, tags, cavs = [], [], [], [], [], [], [], []
    mks: list[str] = []
    for _, r in df.iterrows():
        if _finite(r[x0]) and _finite(r[y0]):
            xs.append(r[x0])
            ys.append(r[y0])
            names.append(r[namecol])
            fams.append(r[famcol])
            lats.append(r["latent_type"])
            cols.append(r["color"])
            tags.append(r["tokenizer"])  # raw full name, shown on hover in simple mode
            cavs.append(r["caveats"])
            mks.append(str(r["marker"]))
    src_data: dict[str, list[Any]] = {
        "x": xs,
        "y": ys,
        "tokenizer": names,
        "family": fams,
        "latent_type": lats,
        "color": cols,
        "caveats": cavs,
        "marker": mks,
    }
    if simple:
        src_data["full_tag"] = tags
    src = ColumnDataSource(cast(Any, src_data))

    p = figure(
        height=460,
        width=740,
        title="Metric vs metric (one point per tokenizer)",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    p.xaxis.axis_label = label_map.get(x0, x0)
    p.yaxis.axis_label = label_map.get(y0, y0)
    p.scatter(
        "x",
        "y",
        source=src,
        marker="marker",
        size=11,
        color="color",
        line_color="white",
        alpha=0.85,
    )
    tooltips = [("tokenizer", "@tokenizer")]
    if simple:  # surface the raw full model name so points stay identifiable in simple mode
        tooltips.append(("full_model_tag", "@full_tag"))
    tooltips += [
        ("family", "@family"),
        ("latent", "@latent_type"),
        ("x", "@x{0.000}"),
        ("y", "@y{0.000}"),
        ("caveats", "@caveats"),
    ]
    p.add_tools(HoverTool(tooltips=tooltips))
    # "n = X of Y": the wide table is a union across tokenizer families, so many metrics
    # cover only a subset. Without the denominator a sparse metric reads as a load bug.
    if _scipy_pearsonr is not None and len(xs) >= 3:
        _r0, _p0 = _scipy_pearsonr(xs, ys)
        rlabel = Div(
            text=f"<b>n = {len(xs)} of {len(df)}</b>  &nbsp; Pearson r = {_r0:.3f}"
            f"  &nbsp; p = {_fmt_p(_p0)}"
        )
    else:
        rlabel = Div(text="")

    # colour-by toggle: Family (0) | Latent type (1); legend matches the encoding
    colorby = RadioButtonGroup(labels=["Family", "Latent type"], active=0)
    logsel = CheckboxGroup(labels=["log X", "log Y"], active=[])
    valsel = RadioButtonGroup(labels=["Raw", "Poisson-norm"], active=0)
    legend_title = Div(text="<b>Family</b>")
    legend = Div(text=_legend_html(_dedup_items(zip(fams, cols, strict=True))), width=150)

    cb = CustomJS(
        args={
            "full": full,
            "scatter_src": src,
            "xsel": xsel,
            "ysel": ysel,
            "rlabel": rlabel,
            "legend": legend,
            "legend_title": legend_title,
            "colorby": colorby,
            "logsel": logsel,
            "valsel": valsel,
            "p": p,
            "fam_w": fam_w,
            "reg_w": reg_w,
            "lat_w": lat_w,
            "partial_w": partial_w,
            "coh_w": coh_w,
        },
        code=_LEGEND_JS_FN
        + _PVAL_JS_FN
        + _FILTER_JS
        + f"const LABELS = {label_map_js};\n"
        + f"const SIMPLE = {str(simple).lower()}; const NAMECOL = '{namecol}';\n"
        + f"const FAMCOL = '{famcol}';\n"
        + """
        const xc = xsel.value, yc = ysel.value;
        // Poisson-normalized view: plot the ::norm column when it exists for the metric
        const norm = (valsel.active === 1);
        const xcol = (norm && (xc+"::norm" in d)) ? xc+"::norm" : xc;
        const ycol = (norm && (yc+"::norm" in d)) ? yc+"::norm" : yc;
        const byFam = SIMPLE ? true : (colorby.active === 0);
        const colCol = byFam ? 'color' : 'color_latent';
        const labCol = byFam ? FAMCOL : 'latent_type';
        const xs=[], ys=[], names=[], fams=[], lats=[], labs=[], cols=[], tags=[], cv=[];
        const mk=[];
        for (const i of visible) {
          const x = d[xcol][i], y = d[ycol][i];
          if (x==null || y==null || isNaN(x) || isNaN(y)) continue;
          xs.push(x); ys.push(y); names.push(d[NAMECOL][i]);
          fams.push(d[FAMCOL][i]); lats.push(d['latent_type'][i]);
          labs.push(d[labCol][i]); cols.push(d[colCol][i]);
          tags.push(d['tokenizer'][i]); cv.push(d['caveats'][i]); mk.push(d['marker'][i]);
        }
        // Pearson r/p on the plotted (raw or Poisson-norm) pairs; symlog doesn't change it
        function pear(a,b){ const n=a.length; if(n<2) return NaN;
          let ma=0,mb=0; for(let i=0;i<n;i++){ma+=a[i];mb+=b[i];} ma/=n; mb/=n;
          let nu=0,da=0,db=0; for(let i=0;i<n;i++){const u=a[i]-ma,v=b[i]-mb;
            nu+=u*v; da+=u*u; db+=v*v;} return (da>0&&db>0)?nu/Math.sqrt(da*db):NaN; }
        const r = pear(xs, ys);
        const pv = pearsonP(r, xs.length);
        // denominator = the filtered cohort, so a sparse metric reads as coverage,
        // not as a broken load (the wide table is a union across tokenizer families).
        rlabel.text = "<b>n = " + xs.length + " of " + visible.length + "</b>" +
          (isNaN(r) ? "" : ("  &nbsp; Pearson r = " + r.toFixed(3) +
           "  &nbsp; p = " + fmtP(pv)));
        // optional symlog display transform: sign(v)*log10(1+|v|). Compresses
        // extreme magnitudes while keeping zeros and negatives (no points dropped).
        const logX = SIMPLE ? false : logsel.active.includes(0);
        const logY = SIMPLE ? false : logsel.active.includes(1);
        const symlog = v => Math.sign(v) * Math.log10(1 + Math.abs(v));
        const px=[], py=[], pn=[], pf=[], pl=[], pb=[], pc=[], pg=[], pcv=[], pk=[];
        for (let k=0;k<xs.length;k++){
          px.push(logX ? symlog(xs[k]) : xs[k]);
          py.push(logY ? symlog(ys[k]) : ys[k]);
          pn.push(names[k]); pf.push(fams[k]);
          pl.push(lats[k]); pb.push(labs[k]); pc.push(cols[k]);
          pg.push(tags[k]); pcv.push(cv[k]); pk.push(mk[k]);
        }
        const out = {x:px, y:py, tokenizer:pn, family:pf, latent_type:pl, color:pc,
                     caveats:pcv, marker:pk};
        if (SIMPLE) out['full_tag'] = pg;
        scatter_src.data = out;
        const seen = {}, items = [];
        for (let j=0;j<pb.length;j++){ const f=pb[j];
          if(!(f in seen)){ seen[f]=1; items.push({label:f, color:pc[j]}); } }
        legend.text = legendHTML(items);
        legend_title.text = byFam ? "<b>Family</b>" : "<b>Latent type</b>";
        function axlbl(metric, col, isLog){
          let s = (LABELS[metric] || metric);
          if (norm) s = col.endsWith("::norm")
            ? s.replace(/\\s[↑↓]\\s*$/, "") + "  (Poisson-norm. ↑)"
            : s + "  (no Poisson ref)";
          return s + (isLog ? "  (symlog₁₀)" : "");
        }
        if (p.below.length) p.below[0].axis_label = axlbl(xc, xcol, logX);
        if (p.left.length) p.left[0].axis_label = axlbl(yc, ycol, logY);
        """,
    )
    for w in (xsel, ysel):
        w.js_on_change("value", cb)
    valsel.js_on_change("active", cb)
    if not simple:
        colorby.js_on_change("active", cb)
        logsel.js_on_change("active", cb)
        legend_box = column(
            Div(text="<b>Colour by</b>"),
            colorby,
            Div(text="<b>Values</b>"),
            valsel,
            Div(text="<b>Scale</b>"),
            logsel,
            legend_title,
            legend,
        )
    else:
        legend_box = column(Div(text="<b>Values</b>"), valsel, legend_title, legend)
    return (
        TabPanel(child=row(column(row(xsel, ysel), rlabel, p), legend_box), title="Scatter"),
        cb,
    )


def _build_ranking_tab(
    df: pd.DataFrame,
    registry: list[dict],
    full: ColumnDataSource,
    fam_w: MultiChoice,
    reg_w: MultiChoice,
    lat_w: CheckboxGroup,
    simple: bool = False,
    partial_w: CheckboxGroup | None = None,
    coh_w: CheckboxGroup | None = None,
) -> tuple[TabPanel, CustomJS]:
    partial_w = partial_w if partial_w is not None else _build_partial_toggle(simple)
    coh_w = coh_w if coh_w is not None else _build_cohort_toggle(df)
    opts = registry_options(registry)
    namecol = "display_name" if simple else "label"
    # Legend uses the pretty family name in the simple view (matching radar); dev keeps raw.
    famcol = "family_pretty" if simple else "family"
    label_map = {e["value"]: _axis_label(e) for e in registry}
    m0 = "single_neuron_recon"
    label0 = label_map.get(m0, m0)
    msel = Select(title="Metric", options=opts, value=m0)

    rows = [
        (
            r[namecol],
            r[m0],
            r["color"],
            r[famcol],
            r["tokenizer"],
            not bool(r.get("m2_comparable", True)),
            r["caveats"],
            str(r["marker"]),
        )
        for _, r in df.iterrows()
        if _finite(r[m0])
    ]
    rows.sort(key=lambda t: t[1], reverse=True)
    names = [str(t[0]) for t in rows]
    src_data: dict[str, list[Any]] = {
        "tokenizer": names,
        "value": [t[1] for t in rows],
        "color": [t[2] for t in rows],
        "upper": [t[1] for t in rows],
        "lower": [t[1] for t in rows],
        # hatch only where the caveat actually applies — i.e. the flagged tokenizers,
        # and only while the affected composite is the metric on screen.
        "hatch": [_M2_HATCH if (t[5] and m0 == M2_SENSITIVE_SCORE) else " " for t in rows],
        "caveats": [str(t[6]) for t in rows],
        "marker": [t[7] for t in rows],
        # the first cohort's circles are not drawn; other cohorts get their marker
        "msize": [0 if t[7] == "circle" else 9 for t in rows],
    }
    if simple:  # raw full name, shown on hover in simple mode
        src_data["full_tag"] = [str(t[4]) for t in rows]
    src = ColumnDataSource(cast(Any, src_data))
    p = figure(
        height=460,
        width=820,
        y_range=FactorRange(factors=list(reversed(names))),
        title=f"Ranked tokenizers — {label0}",
        tools="pan,wheel_zoom,reset,save",
    )
    p.xaxis.axis_label = label0
    p.hbar(
        y="tokenizer",
        right="value",
        left=0,
        height=0.78,
        source=src,
        color="color",
        hatch_pattern="hatch",
        hatch_color="#222",
        hatch_alpha=0.55,
        hatch_scale=9,
    )
    p.scatter(
        x="value",
        y="tokenizer",
        marker="marker",
        size="msize",
        source=src,
        fill_color="#222",
        line_color="white",
    )
    p.add_layout(
        Whisker(
            base="tokenizer",
            upper="upper",
            lower="lower",
            source=src,
            dimension="width",
            line_color="#444",
        )
    )
    rank_tooltips = [("tokenizer", "@tokenizer")]
    if simple:  # surface the raw full model name so bars stay identifiable in simple mode
        rank_tooltips.append(("full_model_tag", "@full_tag"))
    rank_tooltips += [("value", "@value{0.000}"), ("caveats", "@caveats")]
    p.add_tools(HoverTool(tooltips=rank_tooltips))

    # colour-by toggle: Family (0) | Latent type (1); legend matches the encoding
    colorby = RadioButtonGroup(labels=["Family", "Latent type"], active=0)
    logsel = CheckboxGroup(labels=["log scale"], active=[])
    valsel = RadioButtonGroup(labels=["Raw", "Poisson-norm"], active=0)
    legend_title = Div(text="<b>Family</b>")
    legend = Div(text=_legend_html(_dedup_items((t[3], t[2]) for t in rows)), width=150)
    # Metric coverage, plus the hatch explanation when the affected metric is shown.
    # Kept in sync with the JS below, which rebuilds it on every widget change.
    caveat = Div(
        text='<div style="font:11px sans-serif; color:#777; margin-top:4px;">'
        f"{len(rows)} of {len(df)} filtered tokenizers have this metric.</div>",
        width=820,
    )

    label_map_js = json.dumps(label_map)
    cb = CustomJS(
        args={
            "full": full,
            "rank_src": src,
            "msel": msel,
            "p": p,
            "legend": legend,
            "legend_title": legend_title,
            "caveat": caveat,
            "colorby": colorby,
            "logsel": logsel,
            "valsel": valsel,
            "fam_w": fam_w,
            "reg_w": reg_w,
            "lat_w": lat_w,
            "partial_w": partial_w,
            "coh_w": coh_w,
        },
        code=_LEGEND_JS_FN
        + _FILTER_JS
        + f"const LABELS = {label_map_js};\n"
        + f"const SIMPLE = {str(simple).lower()}; const NAMECOL = '{namecol}';\n"
        + f"const FAMCOL = '{famcol}';\n"
        + f"const M2_SENSITIVE = {json.dumps(M2_SENSITIVE_SCORE)};\n"
        + f"const M2_CAVEAT = {json.dumps(M2_CAVEAT)};\n"
        + f"const M2_HATCH = {json.dumps(_M2_HATCH)};\n"
        + """
        const c = msel.value;
        const norm = (valsel.active === 1);
        const col = (norm && (c+"::norm" in d)) ? c+"::norm" : c;
        const sdc = c + "::std";                 // std is for the raw metric only
        const useStd = !col.endsWith("::norm");
        const logOn = SIMPLE ? false : logsel.active.includes(0);
        const symlog = v => Math.sign(v) * Math.log10(1 + Math.abs(v));
        const byFam = SIMPLE ? true : (colorby.active === 0);
        const colCol = byFam ? 'color' : 'color_latent';
        const labCol = byFam ? FAMCOL : 'latent_type';
        const rows = [];
        for (const i of visible) {
          const v = d[col][i];
          if (v==null || isNaN(v)) continue;   // symlog keeps zeros/negatives
          let s = (useStd && sdc in d) ? d[sdc][i] : null;
          if (s==null || isNaN(s)) s = 0;
          rows.push({name:d[NAMECOL][i], val:v, color:d[colCol][i],
                     std:s, lab:d[labCol][i], tag:d['tokenizer'][i], cav:d['caveats'][i],
                     mk:d['marker'][i],
                     flag: ('m2_comparable' in d) ? !d['m2_comparable'][i] : false});
        }
        rows.sort((a,b)=>b.val-a.val);
        const names = rows.map(r=>r.name);
        const vals = rows.map(r => logOn ? symlog(r.val) : r.val);
        // The M2 caveat is about the latent composite only — hatch nothing on other metrics.
        const affected = (c === M2_SENSITIVE);
        // whiskers are linear-only (symlog would make them asymmetric)
        const out = {tokenizer:names, value:vals,
          color:rows.map(r=>r.color),
          upper:rows.map((r,j) => logOn ? vals[j] : r.val+r.std),
          lower:rows.map((r,j) => logOn ? vals[j] : r.val-r.std),
          hatch:rows.map(r => (affected && r.flag) ? M2_HATCH : ' '),
          caveats:rows.map(r => r.cav), marker:rows.map(r => r.mk),
          msize:rows.map(r => r.mk === 'circle' ? 0 : 9)};
        if (SIMPLE) out['full_tag'] = rows.map(r=>r.tag);
        rank_src.data = out;
        const nflag = affected ? rows.filter(r=>r.flag).length : 0;
        // Coverage first: the wide table is a union across tokenizer families, so a short
        // bar list usually means "this metric covers a subset", not a broken load.
        let note = '<div style="font:11px sans-serif; color:#777; margin-top:4px;">'
          + rows.length + ' of ' + visible.length + ' filtered tokenizers have this metric.'
          + '</div>';
        if (nflag) note += '<div style="font:11px sans-serif; color:#8a5a00;'
          + ' background:#fff8e1; border-left:3px solid #e0a800; padding:6px 9px;'
          + ' margin-top:4px;">&#9888; ' + nflag + ' hatched bar'
          + (nflag === 1 ? '' : 's') + ': ' + M2_CAVEAT + '.</div>';
        caveat.text = note;
        p.y_range.factors = names.slice().reverse();
        let lbl = (LABELS[c] || c);
        if (norm) lbl = col.endsWith("::norm")
          ? lbl.replace(/\\s[↑↓]\\s*$/, "") + "  (Poisson-norm. ↑)"
          : lbl + "  (no Poisson ref)";
        lbl = lbl + (logOn ? "  (symlog₁₀)" : "");
        p.title.text = "Ranked tokenizers — " + lbl;
        if (p.below.length) p.below[0].axis_label = lbl;
        const seen = {}, items = [];
        for (const rr of rows){ if(!(rr.lab in seen)){ seen[rr.lab]=1;
          items.push({label:rr.lab, color:rr.color}); } }
        legend.text = legendHTML(items);
        legend_title.text = byFam ? "<b>Family</b>" : "<b>Latent type</b>";
        """,
    )
    msel.js_on_change("value", cb)
    valsel.js_on_change("active", cb)
    if not simple:
        colorby.js_on_change("active", cb)
        logsel.js_on_change("active", cb)
        legend_box = column(
            Div(text="<b>Colour by</b>"),
            colorby,
            Div(text="<b>Values</b>"),
            valsel,
            Div(text="<b>Scale</b>"),
            logsel,
            legend_title,
            legend,
        )
    else:
        legend_box = column(Div(text="<b>Values</b>"), valsel, legend_title, legend)
    return TabPanel(child=row(column(msel, p, caveat), legend_box), title="Ranking"), cb


def _build_radar_tab(
    df: pd.DataFrame,
    full: ColumnDataSource,
    fam_w: MultiChoice,
    reg_w: MultiChoice,
    lat_w: CheckboxGroup,
    simple: bool = False,
    partial_w: CheckboxGroup | None = None,
    coh_w: CheckboxGroup | None = None,
) -> tuple[TabPanel, CustomJS]:
    partial_w = partial_w if partial_w is not None else _build_partial_toggle(simple)
    coh_w = coh_w if coh_w is not None else _build_cohort_toggle(df)
    # Axes = the composites some model of this cohort has; a score no model has (e.g.
    # encoding for the 1000 ms-token cohort) is N/A, not 0, so it gets no spoke.
    comps = cohort_scores(df) or list(COMPOSITE_SCORES)
    absent = [c for c in COMPOSITE_SCORES if c not in comps]
    n = len(comps)
    angles = [math.pi / 2 - 2 * math.pi * k / n for k in range(n)]

    p = figure(
        height=520,
        width=640,
        match_aspect=True,
        x_range=Range1d(start=-1.25, end=1.25),
        y_range=Range1d(start=-1.25, end=1.25),
        title="Composite scores (radar)",
        tools="save,reset",
    )
    p.axis.visible = False
    p.grid.visible = False
    # static rings + spokes + axis labels
    for rr in (0.25, 0.5, 0.75, 1.0):
        p.circle(x=0, y=0, radius=rr, fill_color=None, line_color="#ddd")
    for k, a in enumerate(angles):
        p.line([0, math.cos(a)], [0, math.sin(a)], line_color="#ddd")
        lx, ly = 1.12 * math.cos(a), 1.12 * math.sin(a)
        p.add_layout(
            Label(
                x=lx,
                y=ly,
                text=RADAR_LABELS.get(comps[k], comps[k]),
                text_align="center",
                text_font_size="9pt",
            )
        )

    def verts(row):
        xs, ys = [], []
        for k, a in enumerate(angles):
            v = row[_radar_col(df, comps[k])]  # position; hovers show the raw value
            if not _finite(v):
                continue  # missing: no vertex (drawing it at 0 would read as a score of 0)
            r = max(0.0, min(1.0, float(v)))
            xs.append(r * math.cos(a))
            ys.append(r * math.sin(a))
        return xs + xs[:1], ys + ys[:1]

    # In simple mode the radar overlays one polygon per FAMILY (its best tokenizer,
    # is_family_best=True), labelled by the pretty family name. In dev mode it overlays
    # individual tokenizers (today's behaviour).
    if simple:
        best = df[df["is_family_best"]].sort_values("family_score", ascending=False)
        overlay_keys = [str(v) for v in best["family_pretty"]]  # widget options
        default_overlay = overlay_keys[:3]  # top 3 families
        render_rows = best[best["family_pretty"].isin(default_overlay)]
        keycol = "family_pretty"
    else:
        default_overlay = []
        for fam in sorted(df["family"].unique()):
            default_overlay.append(sorted(df.loc[df["family"] == fam, "tokenizer"])[0])
            if len(default_overlay) == 2:
                break
        overlay_keys = sorted(df["tokenizer"].unique())
        render_rows = df[df["tokenizer"].isin(default_overlay)]
        keycol = "tokenizer"

    data: dict[str, list[Any]] = {
        "xs": [],
        "ys": [],
        "color": [],
        "tokenizer": [],
        "caveats": [],
        "dash": [],
    }
    for c in comps:
        data[f"v_{c}"] = []
    init_items: list[tuple[str, str]] = []
    for _, r in render_rows.iterrows():
        xs, ys = verts(r)
        data["xs"].append(xs)
        data["ys"].append(ys)
        data["color"].append(r["color"])
        data["tokenizer"].append(str(r[keycol] if simple else r["label"]))
        data["caveats"].append(str(r["caveats"]))
        data["dash"].append(str(r["dash"]))
        # The radar always shows all 7 axes, so the caveat is marked on the one value it
        # applies to rather than on the whole polygon.
        flagged = not bool(r.get("m2_comparable", True))
        for c in comps:
            v = _fmt(r[c])
            data[f"v_{c}"].append(f"{v} ⚠" if flagged and c == M2_SENSITIVE_SCORE else v)
        init_items.append((str(r[keycol] if simple else r["label"]), str(r["color"])))
    # cast: the dict is a valid runtime CDS payload; the stub's DataDictLike is
    # an invariant dict that rejects a `dict[str, list[Any]]` variable (a dict
    # *literal* would be checked element-wise, but this one is built up dynamically).
    src = ColumnDataSource(cast(Any, data))
    patch_r = p.patches(
        "xs",
        "ys",
        source=src,
        fill_color="color",
        fill_alpha=0.13,
        line_color="color",
        line_width=2,
        line_dash="dash",
    )
    # color-matched name + every composite value, scoped to the polygons only
    # (so it never fires on the static ring/spoke glyphs -> no "tokenizer: ???").
    tooltip_rows = "".join(
        "<div>" + SCORE_SHORT.get(c, c) + ": <b>@{v_" + c + "}</b></div>" for c in comps
    )
    tooltip_html = (
        '<div style="font:11px sans-serif; padding:3px;">'
        '<div style="font-weight:bold; color:@color;">@tokenizer</div>'
        + tooltip_rows
        + '<div style="color:#8a5a00;">@caveats</div>'
        + "</div>"
    )
    p.add_tools(HoverTool(renderers=[patch_r], tooltips=tooltip_html))
    legend = Div(text=_legend_html(init_items), width=190)
    overlay_w = MultiChoice(
        title="Radar tokenizers (overlay)" if not simple else "Radar families (overlay)",
        options=overlay_keys,
        value=default_overlay,
        width=560,
    )

    comp_js = "[" + ",".join(f'"{c}"' for c in comps) + "]"
    ang_js = "[" + ",".join(repr(a) for a in angles) + "]"
    cb = CustomJS(
        args={
            "full": full,
            "radar_src": src,
            "legend": legend,
            "overlay_w": overlay_w,
            "fam_w": fam_w,
            "reg_w": reg_w,
            "lat_w": lat_w,
            "partial_w": partial_w,
            "coh_w": coh_w,
        },
        code=_LEGEND_JS_FN
        + _FILTER_JS
        + f"const COMP = {comp_js}; const ANG = {ang_js};\n"
        + f"const RCOL = {json.dumps({c: _radar_col(df, c) for c in comps})};\n"
        + f"const SIMPLE = {str(simple).lower()};\n"
        + f"const M2_SENSITIVE = {json.dumps(M2_SENSITIVE_SCORE)};\n"
        + """
        const ov = overlay_w.value;
        const xss=[], yss=[], cols=[], names=[], cvs=[], dsh=[], vcols={};
        for (const c of COMP) vcols[c] = [];
        for (const i of visible) {
          if (SIMPLE && !d['is_family_best'][i]) continue;
          const key = SIMPLE ? d['family_pretty'][i] : d['tokenizer'][i];
          if (ov.length && !ov.includes(key)) continue;
          const xs=[], ys=[];
          const flagged = ('m2_comparable' in d) ? !d['m2_comparable'][i] : false;
          for (let k=0;k<COMP.length;k++) {
            const raw = d[COMP[k]][i];
            const pos = d[RCOL[COMP[k]]][i];
            if (pos!=null && !isNaN(pos)) {  // missing: no vertex (not drawn at 0)
              const r = Math.max(0, Math.min(1, pos));
              xs.push(r*Math.cos(ANG[k])); ys.push(r*Math.sin(ANG[k]));
            }
            let txt = (raw==null || isNaN(raw)) ? '—' : Number(raw).toFixed(3);
            if (flagged && COMP[k] === M2_SENSITIVE) txt += ' \\u26a0';
            vcols[COMP[k]].push(txt);
          }
          if (xs.length) { xs.push(xs[0]); ys.push(ys[0]); }
          xss.push(xs); yss.push(ys);
          cols.push(d['color'][i]); names.push(SIMPLE ? key : d['label'][i]);
          cvs.push(d['caveats'][i]); dsh.push(d['dash'][i]);
        }
        const out = {xs:xss, ys:yss, color:cols, tokenizer:names, caveats:cvs, dash:dsh};
        for (const c of COMP) out['v_'+c] = vcols[c];
        radar_src.data = out;
        legend.text = legendHTML(names.map((nm,j)=>({label:nm, color:cols[j]})));
        """,
    )
    overlay_w.js_on_change("value", cb)
    legend_box = column(Div(text="<b>Tokenizers</b>"), legend)
    absent_note = Div(
        text=(
            '<div style="font:11px sans-serif; color:#777;">Not in this cohort (no model '
            "has it, so no axis): " + ", ".join(SCORE_SHORT.get(c, c) for c in absent) + "</div>"
        )
        if absent
        else ""
    )
    return TabPanel(child=column(overlay_w, absent_note, row(p, legend_box)), title="Radar"), cb


# JS: linear-interpolation quantile over a SORTED numeric array (matches numpy
# 'linear'); plus Tukey 1.5*IQR whiskers clamped to the in-fence data extremes.
_QUANTILE_JS_FN = """
function _q(sorted, q){
  const n = sorted.length;
  if (n === 0) return NaN;
  if (n === 1) return sorted[0];
  const pos = (n - 1) * q, base = Math.floor(pos), rest = pos - base;
  return (base + 1 < n) ? sorted[base] + rest * (sorted[base+1] - sorted[base]) : sorted[base];
}
function boxStats(vals){
  const s = vals.slice().sort((a,b)=>a-b);
  const q1 = _q(s, 0.25), q2 = _q(s, 0.5), q3 = _q(s, 0.75);
  const iqr = q3 - q1, loFence = q1 - 1.5*iqr, hiFence = q3 + 1.5*iqr;
  let lo = q1, hi = q3;
  for (const v of s){ if (v >= loFence){ lo = v; break; } }
  for (let k=s.length-1;k>=0;k--){ if (s[k] <= hiFence){ hi = s[k]; break; } }
  return {q1, q2, q3, lo, hi};
}
"""

_GEN_TIERS = ("xsession", "xsubject", "xsource")


def _build_generalization_tab(
    df: pd.DataFrame,
    full: ColumnDataSource,
    fam_w: MultiChoice,
    reg_w: MultiChoice,
    lat_w: CheckboxGroup,
    gen_by_metric: pd.DataFrame | None = None,
    simple: bool = False,
    partial_w: CheckboxGroup | None = None,
    coh_w: CheckboxGroup | None = None,
) -> tuple[TabPanel, CustomJS]:
    """Box-plot of per-tier OOD retention with jittered family-coloured points and a
    hover trajectory line. A metric ``Select`` (default ``__all__`` = geomean over all
    metrics, i.e. the aggregate view) switches the y-values to a single metric's
    per-tier retention, read from an embedded per-metric source. All recompute is
    client-side CustomJS, honouring the shared family/regime/latent filters."""
    partial_w = partial_w if partial_w is not None else _build_partial_toggle(simple)
    coh_w = coh_w if coh_w is not None else _build_cohort_toggle(df)
    namecol = "display_name" if simple else "label"
    # Legend + hover "family" field use the pretty family name in the simple view
    # (matching the radar tab); dev mode keeps the raw family string.
    famcol = "family_pretty" if simple else "family"
    simple_bare = {m.split("/", 1)[1] for m in SIMPLE_METRICS}  # e.g. spike_detection_f1
    tiers = list(_GEN_TIERS)
    tier_x = {t: i for i, t in enumerate(tiers)}
    have_cols = all(f"gen_tier/{t}" in df.columns for t in tiers)
    if not have_cols:
        notice = Div(
            text='<p style="font:12px sans-serif; color:#777;">No generalization tables for '
            "this cohort: <code>generalization_by_tier.csv</code> not supplied (or empty).</p>"
        )
        return TabPanel(child=notice, title="Generalization"), CustomJS(code="")

    def _rows():
        for _, r in df.iterrows():
            if not have_cols:
                return
            vals = {t: r.get(f"gen_tier/{t}") for t in tiers}
            if not any(_finite(v) for v in vals.values()):
                continue
            yield r, vals

    # initial (default = aggregate) render, mirrored by the JS aggregate path
    pmk: list[str] = []
    px, py, pt, pn, pf, pl, pc, pm, pnn, pmet, ptag, pcav = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for r, vals in _rows():
        lane = float(r["gen_lane"])
        for t in tiers:
            v = vals[t]
            if not _finite(v):
                continue
            px.append(tier_x[t] + lane)
            py.append(float(cast(Any, v)))
            pt.append(t)
            pn.append(str(r[namecol]))
            pf.append(str(r[famcol]))
            pl.append(str(r["latent_type"]))
            pc.append(str(r["color"]))
            pm.append(_fmt(r.get(f"gen_tier/{t}::mean")))
            nv = r.get(f"gen_tier/{t}::n")
            pnn.append("—" if not _finite(nv) else str(int(cast(Any, nv))))
            pmet.append("(all, geomean)")
            ptag.append(str(r["tokenizer"]))  # raw full name, shown on hover in simple mode
            pcav.append(str(r["caveats"]))
            pmk.append(str(r["marker"]))
    pts_data: dict[str, list[Any]] = {
        "x": px,
        "y": py,
        "tier": pt,
        "tokenizer": pn,
        "family": pf,
        "latent_type": pl,
        "color": pc,
        "mean": pm,
        "n": pnn,
        "metric": pmet,
        "caveats": pcav,
        "marker": pmk,
    }
    if simple:
        pts_data["full_tag"] = ptag
    pts_src = ColumnDataSource(cast(Any, pts_data))

    def _box_init():
        xs, q1s, q2s, q3s, los, his, names = [], [], [], [], [], [], []
        for t in tiers:
            vals = sorted(
                float(cast(Any, v))
                for _, vv in _rows()
                for k, v in vv.items()
                if k == t and _finite(v)
            )
            if not vals:
                continue
            import numpy as _np

            q1, q2, q3 = (float(_np.quantile(vals, q)) for q in (0.25, 0.5, 0.75))
            iqr = q3 - q1
            lo = next((v for v in vals if v >= q1 - 1.5 * iqr), q1)
            hi = next((v for v in reversed(vals) if v <= q3 + 1.5 * iqr), q3)
            xs.append(float(tier_x[t]))
            q1s.append(q1)
            q2s.append(q2)
            q3s.append(q3)
            los.append(lo)
            his.append(hi)
            names.append(t)
        return {
            "x": xs,
            "x_lo": [v - 0.25 for v in xs],
            "x_hi": [v + 0.25 for v in xs],
            "q1": q1s,
            "q2": q2s,
            "q3": q3s,
            "lo": los,
            "hi": his,
            "tier": names,
        }

    box_src = ColumnDataSource(cast(Any, _box_init()))
    line_src = ColumnDataSource({"x": [], "y": []})

    # per-metric source (embedded) + the metric Select; absent gen_by_metric => no selector
    if gen_by_metric is not None and not gen_by_metric.empty:
        gm = gen_by_metric
        metric_src = ColumnDataSource(
            cast(
                Any,
                {
                    "tokenizer": [str(v) for v in gm["tokenizer"]],
                    "metric": [str(v) for v in gm["metric"]],
                    "ret_xsession": [
                        float(v) if _finite(v) else float("nan") for v in gm["ret_xsession"]
                    ],
                    "ret_xsubject": [
                        float(v) if _finite(v) else float("nan") for v in gm["ret_xsubject"]
                    ],
                    "ret_xsource": [
                        float(v) if _finite(v) else float("nan") for v in gm["ret_xsource"]
                    ],
                },
            )
        )
        # Bokeh optgroup options are a dict {group: [(value,label),...]}; the ungrouped
        # default option lives under the "" key so it renders above the category groups.
        groups: dict[str, list[tuple[str, str]]] = {}
        for _, mr in gm[["metric", "category"]].drop_duplicates().iterrows():
            if simple and str(mr["metric"]) not in simple_bare:
                continue
            cat = str(mr["category"]) if _finite_str(mr["category"]) else "other"
            label = str(mr["metric"]).replace("_", " ")
            groups.setdefault(cat, []).append((str(mr["metric"]), label))
        grouped: dict[str, list] = {"": [("__all__", "All metrics (geomean)")]}
        for cat in sorted(groups):
            grouped[cat.replace("_", " ")] = sorted(groups[cat], key=lambda t: t[1])
        msel = Select(title="Metric", value="__all__", options=cast(Any, grouped))
    else:
        metric_src = ColumnDataSource(
            cast(
                Any,
                {
                    "tokenizer": [],
                    "metric": [],
                    "ret_xsession": [],
                    "ret_xsubject": [],
                    "ret_xsource": [],
                },
            )
        )
        msel = Select(
            title="Metric", value="__all__", options=[("__all__", "All metrics (geomean)")]
        )

    p = figure(
        height=460,
        width=720,
        x_range=Range1d(start=-0.5, end=len(tiers) - 0.5),
        y_range=Range1d(start=0.0, end=1.05),
        title="Generalization retention by OOD tier",
        tools="pan,wheel_zoom,box_zoom,reset,save",
    )
    p.yaxis.axis_label = "Retention vs xtime baseline ↑"
    p.xaxis.ticker = FixedTicker(ticks=list(range(len(tiers))))
    p.xaxis.major_label_overrides = dict(enumerate(tiers))
    p.xgrid.visible = False

    p.vbar(
        x="x",
        width=0.5,
        bottom="q1",
        top="q3",
        source=box_src,
        fill_color="#dfe6ef",
        line_color="#5b6b7f",
        fill_alpha=0.6,
    )
    p.segment(x0="x", y0="lo", x1="x", y1="q1", source=box_src, line_color="#5b6b7f")
    p.segment(x0="x", y0="q3", x1="x", y1="hi", source=box_src, line_color="#5b6b7f")
    p.segment(
        x0="x_lo", y0="q2", x1="x_hi", y1="q2", source=box_src, line_color="#1a1a1a", line_width=2
    )
    p.line(x="x", y="y", source=line_src, line_color="#444", line_width=2, line_alpha=0.7)
    scatter_r = p.scatter(
        "x",
        "y",
        source=pts_src,
        marker="marker",
        size=10,
        color="color",
        line_color="white",
        alpha=0.9,
    )

    tag_row = "<div>full_model_tag: <b>@full_tag</b></div>" if simple else ""
    tooltip_html = (
        '<div style="font:11px sans-serif; padding:3px;">'
        '<div style="font-weight:bold; color:@color;">@tokenizer</div>'
        + tag_row
        + "<div>family: <b>@family</b></div>"
        "<div>latent: <b>@latent_type</b></div>"
        "<div>tier: <b>@tier</b></div>"
        "<div>metric: <b>@metric</b></div>"
        "<div>retention: <b>@y{0.000}</b></div>"
        "<div>mean (agg): <b>@mean</b> · n: <b>@n</b></div>"
        '<div style="color:#8a5a00;">@caveats</div>'
        "</div>"
    )
    hover = HoverTool(renderers=[scatter_r], tooltips=tooltip_html)
    p.add_tools(hover)

    legend = Div(text=_legend_html(_dedup_items(zip(pf, pc, strict=True))), width=170)

    tiers_js = "[" + ",".join(f'"{t}"' for t in tiers) + "]"
    tierx_js = json.dumps(tier_x)
    main_cb = CustomJS(
        args={
            "full": full,
            "pts_src": pts_src,
            "box_src": box_src,
            "metric_src": metric_src,
            "msel": msel,
            "p": p,
            "legend": legend,
            "fam_w": fam_w,
            "reg_w": reg_w,
            "lat_w": lat_w,
            "partial_w": partial_w,
            "coh_w": coh_w,
        },
        code=_LEGEND_JS_FN
        + _QUANTILE_JS_FN
        + _FILTER_JS
        + f"const TIERS = {tiers_js}; const TX = {tierx_js};\n"
        + f"const NAMECOL = '{namecol}';\n"
        + f"const SIMPLE = {str(simple).lower()}; const FAMCOL = '{famcol}';\n"
        + """
        const fin = v => (v != null && !isNaN(v));
        const M = msel.value, AGG = (M === "__all__");
        // per-metric lookup: tokenizer -> [ret_xsession, ret_xsubject, ret_xsource]
        const mmap = {};
        if (!AGG) {
          const md = metric_src.data;
          for (let k=0;k<md['metric'].length;k++){
            if (md['metric'][k] === M)
              mmap[md['tokenizer'][k]] =
                [md['ret_xsession'][k], md['ret_xsubject'][k], md['ret_xsource'][k]];
          }
        }
        const px=[], py=[], pt=[], pn=[], pf=[], pl=[], pc=[], pm=[], pnn=[], pmet=[], ptag=[];
        const pcav=[], pmk=[];
        const perTier = {}; for (const t of TIERS) perTier[t] = [];
        const items = [], seenFam = {};
        for (const i of visible) {
          const lane = d['gen_lane'][i], rets = AGG ? null : mmap[d['tokenizer'][i]];
          let any = false;
          for (let ti=0; ti<TIERS.length; ti++) {
            const t = TIERS[ti];
            let y, meanStr='—', nStr='—', metStr;
            if (AGG) {
              y = d['gen_tier/'+t][i];
              if (!fin(y)) continue;
              const mv = d['gen_tier/'+t+'::mean'][i];
              meanStr = fin(mv) ? Number(mv).toFixed(3) : '—';
              const nv = d['gen_tier/'+t+'::n'][i];
              nStr = fin(nv) ? String(Math.round(nv)) : '—';
              metStr = '(all, geomean)';
            } else {
              if (!rets) continue;
              y = rets[ti];
              if (!fin(y)) continue;
              metStr = M;
            }
            any = true; perTier[t].push(y);
            px.push(TX[t] + lane); py.push(y); pt.push(t);
            pn.push(d[NAMECOL][i]); pf.push(d[FAMCOL][i]);
            pl.push(d['latent_type'][i]); pc.push(d['color'][i]);
            pm.push(meanStr); pnn.push(nStr); pmet.push(metStr);
            ptag.push(d['tokenizer'][i]); pcav.push(d['caveats'][i]); pmk.push(d['marker'][i]);
          }
          if (any && !(d[FAMCOL][i] in seenFam)) {
            seenFam[d[FAMCOL][i]] = 1;
            items.push({label: d[FAMCOL][i], color: d['color'][i]});
          }
        }
        const out = {x:px, y:py, tier:pt, tokenizer:pn, family:pf,
                     latent_type:pl, color:pc, mean:pm, n:pnn, metric:pmet, caveats:pcav,
                     marker:pmk};
        if (SIMPLE) out['full_tag'] = ptag;
        pts_src.data = out;
        const bx=[], q1=[], q2=[], q3=[], lo=[], hi=[], bt=[];
        for (const t of TIERS) {
          if (!perTier[t].length) continue;
          const s = boxStats(perTier[t]);
          bx.push(TX[t]); q1.push(s.q1); q2.push(s.q2); q3.push(s.q3);
          lo.push(s.lo); hi.push(s.hi); bt.push(t);
        }
        box_src.data = {x:bx, x_lo:bx.map(v=>v-0.25), x_hi:bx.map(v=>v+0.25),
                        q1:q1, q2:q2, q3:q3, lo:lo, hi:hi, tier:bt};
        legend.text = legendHTML(items);
        const lbl = AGG ? "Retention vs xtime baseline ↑"
                        : (M.replace(/_/g,' ') + " retention vs xtime ↑");
        if (p.left.length) p.left[0].axis_label = lbl;
        p.title.text = AGG ? "Generalization retention by OOD tier"
                           : ("Generalization — " + M.replace(/_/g,' '));
        """,
    )
    # The shared filters are wired once, for every tab, in _build_view.
    msel.js_on_change("value", main_cb)

    hover_cb = CustomJS(
        args={"pts_src": pts_src, "line_src": line_src},
        code="""
        const idx = cb_data.index.indices;
        if (!idx || !idx.length) { line_src.data = {x:[], y:[]}; line_src.change.emit(); return; }
        const d = pts_src.data, tok = d['tokenizer'][idx[0]];
        const order = {xsession:0, xsubject:1, xsource:2}, triples = [];
        for (let k=0;k<d['tokenizer'].length;k++){
          if (d['tokenizer'][k] === tok)
            triples.push([order[d['tier'][k]], d['x'][k], d['y'][k]]);
        }
        triples.sort((a,b)=>a[0]-b[0]);
        line_src.data = {x: triples.map(t=>t[1]), y: triples.map(t=>t[2])};
        line_src.change.emit();
        """,
    )
    hover.callback = hover_cb

    legend_box = column(Div(text="<b>Family</b>"), legend)
    # What the tiers pool (first principles, briefly; the glossary has the full text). The
    # count varies per model (its metric set), so it is in the hover, not here.
    note = Div(
        text='<div style="font:11px/1.5 sans-serif; color:#777; max-width:720px;">'
        "Does the tokenizer's <b>reconstruction</b> hold up out of distribution? "
        "Reconstruction metrics only (single-unit, temporal, population structure); each "
        "tokenizer's tiers are compared over the metrics all three tiers share. "
        "All metrics = geometric mean of their retention vs the xtime baseline. "
        "The Generalization score uses xsource (unseen datasets) only; xsession and "
        "xsubject are shown for reference."
        + (" The metric menu is a shortlist; dev mode lists every metric." if simple else "")
        + "</div>"
    )
    return TabPanel(child=row(column(msel, note, p), legend_box), title="Generalization"), main_cb


_BANNER_STYLE = (
    "font:12px/1.5 sans-serif; color:#5a4300; background:#fff8e1; "
    "border-left:3px solid #e0a800; padding:6px 10px; margin-top:6px;"
)


def _cohort_banner_html(footnotes: list[str]) -> str:
    """Footnotes every model of the cohort carries, shown once above the tabs."""
    if not footnotes:
        return ""
    items = "".join(f"<li>{html.escape(f)}</li>" for f in footnotes)
    return (
        f'<div style="{_BANNER_STYLE}"><b>Applies to every model of a cohort:</b>'
        f'<ul style="margin:2px 0 0 0;">{items}</ul></div>'
    )


def _caveats_tab_html(df: pd.DataFrame, caveats: Caveats, *, simple: bool) -> str:
    """Per-model footnotes (+ provenance notes in dev mode), escaped."""
    if not caveats.supplied:
        return (
            '<p style="font:12px sans-serif; color:#777;">No caveats shown for this cohort: '
            "<code>caveats.csv</code> not supplied.</p>"
        )
    namecol = "display_name" if simple else "tokenizer"
    head = "<tr><th>Model</th><th>Caveats</th>" + ("" if simple else "<th>Notes</th>") + "</tr>"
    rows = []
    for _, r in df.iterrows():
        tok = str(r["tokenizer"])
        fns = caveats.per_model.get(tok, [])
        note = "" if simple else caveats.notes.get(tok, "")
        if not fns and not note:
            continue
        cells = "<br>".join(f"{CAVEAT_MARK} {html.escape(f)}" for f in fns)
        rows.append(
            f"<tr><td>{html.escape(str(r[namecol]))}</td><td>{cells}</td>"
            + ("" if simple else f"<td>{html.escape(note)}</td>")
            + "</tr>"
        )
    table = (
        '<table style="font:12px/1.5 sans-serif; border-collapse:collapse;">'
        + head
        + "".join(rows)
        + "</table>"
        if rows
        else "<p>No model-specific caveats.</p>"
    )
    return _cohort_banner_html(caveats.cohort_wide) + table


def _build_view(
    df: pd.DataFrame,
    registry: list[dict],
    gen_by_metric: pd.DataFrame | None = None,
    *,
    simple: bool = False,
    caveats: Caveats = NO_CAVEATS,
) -> Any:
    """Assemble one dashboard view (filters + Radar|Scatter|Ranking|Generalization|
    Caveats|Models|Glossary). ``simple`` selects the PI-friendly behaviours."""
    full = _build_full_source(df)
    fam_w, reg_w, lat_w = _build_filter_widgets(df)
    # One shared "include partial evaluations" toggle drives the complete-only filter on
    # every tab (see _FILTER_JS). Dev: ON + shown; simple: OFF (complete-only) + hidden.
    partial_w = _build_partial_toggle(simple)
    coh_w = _build_cohort_toggle(df)  # hidden for a single cohort

    radar_panel, radar_cb = _build_radar_tab(
        df, full, fam_w, reg_w, lat_w, simple=simple, partial_w=partial_w, coh_w=coh_w
    )
    scatter_panel, scatter_cb = _build_scatter_tab(
        df, registry, full, fam_w, reg_w, lat_w, simple=simple, partial_w=partial_w, coh_w=coh_w
    )
    ranking_panel, ranking_cb = _build_ranking_tab(
        df, registry, full, fam_w, reg_w, lat_w, simple=simple, partial_w=partial_w, coh_w=coh_w
    )
    gen_panel, gen_cb = _build_generalization_tab(
        df,
        full,
        fam_w,
        reg_w,
        lat_w,
        gen_by_metric,
        simple=simple,
        partial_w=partial_w,
        coh_w=coh_w,
    )

    for w in (fam_w, reg_w):
        w.js_on_change("value", radar_cb, scatter_cb, ranking_cb, gen_cb)
    lat_w.js_on_change("active", radar_cb, scatter_cb, ranking_cb, gen_cb)
    partial_w.js_on_change("active", radar_cb, scatter_cb, ranking_cb, gen_cb)
    coh_w.js_on_change("active", radar_cb, scatter_cb, ranking_cb, gen_cb)

    title = "NTT tokenizer results" + ("" if not simple else " — overview")
    subtitle = (
        "Interactive comparison of tokenizer-evaluation metrics — filter the cohort, "
        "then pick metrics per tab. Fully self-contained; works offline."
        if not simple
        else "Best tokenizer per family. Filter, then pick a metric per tab. "
        "Switch to dev mode for every tokenizer and metric."
    )
    header = Div(
        text=(
            '<div style="font:600 22px/1.25 sans-serif; color:#1a1a1a;">' + title + "</div>"
            '<div style="font:13px/1.45 sans-serif; color:#666; margin-top:2px;">'
            + subtitle
            + "</div>"
        )
    )
    if simple:
        guide = Div(
            text=(
                '<div style="font:11px/1.5 sans-serif; color:#777; margin-top:4px;">'
                "<b>Radar</b>: composite scores (best per family) · "
                "<b>Scatter</b>: any metric vs any metric · "
                "<b>Ranking</b>: sorted bars · "
                "<b>Generalization</b>: out-of-distribution retention · "
                "<b>Glossary</b>: metric definitions. &nbsp;|&nbsp; Axis arrows: "
                "<b>↑</b> higher-is-better, <b>↓</b> lower-is-better. &nbsp;|&nbsp; "
                "<b>Values</b>: Raw or <b>Poisson-normalized</b> (0 = Poisson baseline, "
                "or every neuron at the population mean rate for firing-rate error; "
                "1 = ceiling) where available. &nbsp;|&nbsp; "
                "<b>⚠ / hatched bars</b>: " + M2_CAVEAT + ". &nbsp;|&nbsp; "
                "<b>†</b>: model-specific caveat (hover, or the Caveats tab)."
                + _cohort_legend(df)
                + "</div>"
            )
        )
    else:
        guide = Div(
            text=(
                '<div style="font:11px/1.5 sans-serif; color:#777; margin-top:4px;">'
                "<b>Radar</b>: composite scores · <b>Scatter</b>: any metric vs any "
                "metric (+ Pearson r/p) · <b>Ranking</b>: sorted bars with cross-session "
                "std · <b>Generalization</b>: OOD retention (box + points) across "
                "xsession/xsubject/xsource, hover a point for its trajectory · "
                "<b>Glossary</b>: metric definitions. &nbsp;|&nbsp; Axis arrows: "
                "<b>↑</b> higher-is-better, <b>↓</b> lower-is-better. &nbsp;|&nbsp; "
                "<b>Values</b>: Raw or <b>Poisson-normalized</b> (0 = Poisson baseline, "
                "or every neuron at the population mean rate for firing-rate error; "
                "1 = ceiling) where available. &nbsp;|&nbsp; "
                "<b>Scale</b>: a <b>symlog</b> axis (signed log₁₀, keeps zeros & "
                "negatives) to compress extreme values. &nbsp;|&nbsp; "
                "<b>⚠ / hatched bars</b>: " + M2_CAVEAT + ". &nbsp;|&nbsp; "
                "<b>†</b>: model-specific caveat (hover, or the Caveats tab)."
                + _cohort_legend(df)
                + "</div>"
            )
        )
    filters_label = Div(text='<div style="font:600 12px sans-serif; color:#444;">Filters</div>')
    lat_box = column(Div(text="<b>Latent type</b>"), lat_w)
    coh_box = column(Div(text="<b>Cohort</b>"), coh_w, visible=coh_w.visible)
    if simple:
        # The "include partial evaluations" toggle is hidden here — simple is complete-only.
        filters = row(fam_w, lat_box, coh_box)
    else:
        partial_box = column(Div(text="<b>Evaluations</b>"), partial_w)
        filters = row(fam_w, reg_w, lat_box, coh_box, partial_box)
    model_panel = _build_glossary_tab(
        _SIMPLE_MODEL_GLOSSARY_MD if simple else _MODEL_GLOSSARY_MD, title="Models"
    )
    glossary_panel = _build_glossary_tab(_SIMPLE_GLOSSARY_MD if simple else _GLOSSARY_MD)
    caveats_panel = TabPanel(
        child=Div(text=_caveats_tab_html(df, caveats, simple=simple), sizing_mode="stretch_width"),
        title="Caveats",
    )
    tabs = Tabs(
        tabs=[
            radar_panel,
            scatter_panel,
            ranking_panel,
            gen_panel,
            caveats_panel,
            model_panel,
            glossary_panel,
        ],
        sizing_mode="stretch_width",
    )
    banner = Div(text=_cohort_banner_html(caveats.cohort_wide), sizing_mode="stretch_width")
    return column(header, guide, banner, filters_label, filters, tabs, sizing_mode="stretch_width")


REQUIRED_FILES = ("category_scores.csv", "wide_metrics_table.csv", "metadata_rows.csv")
# Blanked for a cohort whose behavior decoding is not comparable with the others' (e.g. the
# 1000 ms-token models decode from 1 s chunks).
DECODING_COLUMN_PREFIXES = ("decoding/", "downstream_decoding/")


@dataclass
class CohortData:
    """One cohort's inputs (``merge_cohorts`` puts several on one page)."""

    name: str
    frame: pd.DataFrame
    registry: list[dict]
    gen_by_metric: pd.DataFrame | None
    caveats: Caveats
    hide_decoding: bool = False


def load_cohort(name: str, directory: str | Path, *, hide_decoding: bool = False) -> CohortData:
    """Load a cohort directory (fixed file names, see ``REQUIRED_FILES``; the two
    generalization tables and ``caveats.csv`` are optional)."""
    d = Path(directory)
    missing = [f for f in REQUIRED_FILES if not (d / f).is_file()]
    if missing:
        raise FileNotFoundError(f"cohort {name!r}: {d} lacks {missing}")
    frame = load_dashboard_frame(
        d / "category_scores.csv",
        d / "wide_metrics_table.csv",
        d / "metadata_rows.csv",
        d / "generalization_by_tier.csv",
    )
    return CohortData(
        name=name,
        frame=frame,
        registry=build_metric_registry(frame),
        gen_by_metric=load_generalization_by_metric(d / "generalization_by_metric.csv"),
        caveats=load_caveats(d / "caveats.csv", frame["tokenizer"]),
        hide_decoding=hide_decoding,
    )


def merge_cohorts(cohorts: Sequence[CohortData]) -> CohortData:
    """Put several cohorts on one page: one frame with a ``cohort`` column and a marker /
    radar dash per cohort (first cohort: circles, solid). Decoding columns are blanked for
    ``hide_decoding`` cohorts. Completeness stays per cohort; the compression radar scale
    and the family colours span all cohorts."""
    if not cohorts:
        raise ValueError("no cohorts")
    names = [c.name for c in cohorts]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate cohort names: {names}")
    if len(cohorts) == 1 and not cohorts[0].hide_decoding:
        return cohorts[0]
    multi = len(cohorts) > 1
    frames, notes = [], []
    for k, c in enumerate(cohorts):
        f = c.frame.copy()
        f["cohort"] = c.name
        f["marker"] = COHORT_MARKERS[k % len(COHORT_MARKERS)]
        f["dash"] = "solid" if k == 0 else "dashed"
        if c.hide_decoding:
            cols = [
                x
                for x in f.columns
                if x in ("decoding", *DECODING_BREAKDOWNS)
                or x.startswith(DECODING_COLUMN_PREFIXES)
            ]
            f[cols] = float("nan")
            notes.append(
                f"{c.name} models: left out of the decoding score and the decoding metrics "
                "(decoded from a different chunk size; not comparable)"
            )
        # completeness per cohort: a score no model of the cohort has is N/A for it
        comp_cols = cohort_scores(f)
        f["is_complete"] = f[comp_cols].notna().all(axis=1) if comp_cols else False
        frames.append(f)
    df = pd.concat(frames, ignore_index=True, sort=False)
    dup = sorted(set(df.loc[df["tokenizer"].duplicated(), "tokenizer"]))
    if dup:
        raise ValueError(f"tokenizers in more than one cohort: {dup}")
    fams = sorted(df["family"].unique())
    color_map = {f: _FAMILY_PALETTE[i % len(_FAMILY_PALETTE)] for i, f in enumerate(fams)}
    df["color"] = df["family"].map(color_map.get)
    if "compression" in df.columns:
        df["compression_radar"] = _log_minmax(df["compression"])
    # every family of a later cohort carries the cohort, so the simple view never shows a
    # later-cohort model under a bare family name
    df["family_label"] = [
        family_pretty(f) + (f" ({c})" if c != names[0] else "")
        for f, c in zip(df["family"], df["cohort"], strict=True)
    ]
    gms = [c.gen_by_metric for c in cohorts if c.gen_by_metric is not None]
    caveats = Caveats(
        per_model={t: fs for c in cohorts for t, fs in c.caveats.per_model.items()},
        cohort_wide=[
            *[
                f"{c.name} models: {fn}" if multi else fn
                for c in cohorts
                for fn in c.caveats.cohort_wide
            ],
            *notes,
        ],
        notes={t: n for c in cohorts for t, n in c.caveats.notes.items()},
        supplied=any(c.caveats.supplied for c in cohorts),
    )
    return CohortData(
        name=" + ".join(names),
        frame=df,
        registry=build_metric_registry(df),
        gen_by_metric=pd.concat(gms, ignore_index=True) if gms else None,
        caveats=caveats,
    )


def _cohort_views(cohort: CohortData) -> tuple[Any, Any]:
    """The simple and dev views (own sources, filters and callbacks)."""
    df = apply_caveats(cohort.frame, cohort.caveats)
    simple_df = derive_simple_frame(df)
    simple_view = _build_view(
        simple_df,
        build_metric_registry(simple_df, simple=True),
        cohort.gen_by_metric,
        simple=True,
        caveats=cohort.caveats,
    )
    dev_view = _build_view(
        df, cohort.registry, cohort.gen_by_metric, simple=False, caveats=cohort.caveats
    )
    simple_view.name, dev_view.name = "simple_view", "dev_view"
    return simple_view, dev_view


def _provenance_html(provenance: dict | None) -> str:
    """Collapsible page footer listing the inputs (nested keys joined with " · ")."""
    if not provenance:
        return ""

    def flat(d: dict, prefix: str = ""):
        for k, v in d.items():
            if isinstance(v, dict):
                yield from flat(v, f"{prefix}{k} · ")
            else:
                yield f"{prefix}{k}", v

    rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td><code>{html.escape(str(v))}</code></td></tr>"
        for k, v in flat(provenance)
    )
    return (
        '<details style="font:11px sans-serif; color:#666; margin-top:12px;">'
        f"<summary>Provenance</summary><table>{rows}</table></details>"
    )


def build_page(cohorts: Sequence[CohortData], provenance: dict | None = None):
    """The standalone page: every cohort in the same plots (marker per cohort, see
    ``merge_cohorts``), a simple view and a dev view behind a page-wide toggle (``#dev``
    deep link in ``export_html``)."""
    simple_view, dev_view = _cohort_views(merge_cohorts(cohorts))
    simple_view.visible, dev_view.visible = True, False
    toggle = CheckboxGroup(labels=["dev mode"], active=[], name="dev_toggle", width=110)
    toggle.js_on_change(
        "active",
        CustomJS(
            args={"simple_view": simple_view, "dev_view": dev_view, "toggle": toggle},
            code="const on = toggle.active.includes(0);"
            " simple_view.visible = !on; dev_view.visible = on;",
        ),
    )
    # discreet, low-contrast dev toggle parked top-right
    top = row(
        Div(text="", sizing_mode="stretch_width"),
        column(toggle, styles={"opacity": "0.45", "font-size": "11px"}),
        sizing_mode="stretch_width",
    )
    footer = Div(text=_provenance_html(provenance), sizing_mode="stretch_width")
    return column(
        top,
        simple_view,
        dev_view,
        footer,
        sizing_mode="stretch_width",
        styles={
            "font-family": "sans-serif",
            "padding": "16px 20px",
            "max-width": "min(1500px, 92vw)",
            "margin": "0 auto",
            "box-sizing": "border-box",
            "max-height": "96vh",
            "overflow": "auto",
            "background": "#fcfcfc",
            "border": "1px solid #eee",
            "border-radius": "8px",
        },
    )


def build_dashboard(
    df: pd.DataFrame,
    registry: list[dict],
    gen_by_metric: pd.DataFrame | None = None,
    caveats: Caveats | None = None,
):
    """Single-cohort page: a PI-friendly Simple view (default) and the full Dev view,
    flipped by a discreet toggle (and the ``#dev`` URL deep-link wired in
    ``export_html``)."""
    return build_page([CohortData("", df, registry, gen_by_metric, caveats or NO_CAVEATS)])


def export_html(layout, path: str | Path, title: str = "NTT tokenizer results") -> Path:
    """Write a standalone, offline, self-contained HTML for `layout`.

    Also attaches a DocumentReady callback so opening the page at ``…#dev`` starts in
    dev mode (it ticks the named ``dev_toggle``, whose own callback flips the views)."""
    from bokeh.document import Document
    from bokeh.events import DocumentReady

    path = Path(path)
    doc = layout.document or Document()
    if layout not in doc.roots:
        doc.add_root(layout)
    doc.js_on_event(
        DocumentReady,
        CustomJS(
            code="if (window.location.hash === '#dev') {"
            " const d = Bokeh.documents && Bokeh.documents[0];"
            " const t = d && d.get_model_by_name('dev_toggle');"
            " if (t) { t.active = [0]; } }"
        ),
    )
    html = file_html(doc, resources=INLINE, title=title)
    path.write_text(html, encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the standalone tokenizer-results dashboard.")
    ap.add_argument(
        "--cohort",
        nargs=2,
        action="append",
        required=True,
        metavar=("NAME", "DIR"),
        help="a cohort directory (repeatable; all cohorts share the plots, one marker each)",
    )
    ap.add_argument(
        "--hide-decoding",
        action="append",
        default=[],
        metavar="NAME",
        help="a cohort whose decoding scores are not comparable with the others' (repeatable)",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--provenance",
        type=Path,
        help="optional: any JSON object (e.g. code version, where the inputs came from), "
        "shown as a collapsible footer; omit it for no footer",
    )
    ap.add_argument("--title", default="NTT tokenizer results")
    args = ap.parse_args(argv)
    names = [name for name, _ in args.cohort]
    unknown = sorted(set(args.hide_decoding) - set(names))
    if unknown:
        ap.error(f"--hide-decoding names no --cohort: {unknown}")
    cohorts = [
        load_cohort(name, d, hide_decoding=name in args.hide_decoding) for name, d in args.cohort
    ]
    provenance = json.loads(args.provenance.read_text()) if args.provenance else None
    out = export_html(build_page(cohorts, provenance), args.out, title=args.title)
    for c in cohorts:
        n_simple = len(derive_simple_frame(apply_caveats(c.frame, c.caveats)))
        print(
            f"{c.name}: {len(c.frame)} tokenizers, {n_simple} in the simple view, "
            f"{len(c.caveats.per_model)} with caveats, {len(c.caveats.cohort_wide)} cohort-wide"
        )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
