"""Shared data-loading helpers for NTT analysis tools.

Loads one ``category_scores.json`` per tracker row, written by
``scripts/analysis/build_aggregation_tables.py``, from either of two places:

* **MLflow** (default) — the ``mlflow_aggregation`` run URL, from the
  ``tracker_with_agg.csv`` that ``build_aggregation_tables.py --log-mlflow`` writes. The
  tracking server is ``MLFLOW_TRACKING_URI`` (or ``mlflow_uri=``); links to another server
  are an error.
* **Local** (``scores_dir=``) — ``<scores_dir>/<slug>.json`` as persisted by
  ``build_aggregation_tables.py``: the same files the MLflow runs were published from.

Both routes converge on the same row assembly, so the consumers
(``build_latex_table``, ``plot_radar``, ``plot_family``, ``plot_correlation``)
see an identical frame — per-metric diagnostics included — either way.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from ntt.evaluation.aggregation import tracker_io
from ntt.evaluation.aggregation.comparability import (
    COMMON_PARTS_ONLY_FLAG,
    find_mixed_member_rows,
    provenance_record,
    raise_if_not_comparable,
    raise_on_mixed_chunk_sizes,
    raise_on_mixed_score_parts,
)

# ── Composite score columns ────────────────────────────────────────────────────
COMPOSITE_SCORES: list[str] = [
    "single_neuron_recon",
    "population_structure",
    "decoding",
    "downstream_encoding",
    "latent_structure_and_robustness",
    "generalization",
    "compression",
]

# The two family scores the ``decoding`` composite averages (equal weight). Pickable and
# labelled like the composites, but not radar axes and not part of completeness.
DECODING_BREAKDOWNS: list[str] = ["decoding_poyo", "decoding_linear"]

# Human-readable labels for composite scores and the decoding breakdown.
SCORE_LABELS: dict[str, str] = {
    "single_neuron_recon": "SN Recon (Poisson-norm.)",
    "population_structure": "Population Structure (Poisson-norm.)",
    "decoding": "Decoding (POYO + linear)",
    "decoding_poyo": "POYO decoding (cursor-velocity R²)",
    "decoding_linear": "Linear decoding (ridge R²)",
    "downstream_encoding": "Downstream Encoding (nc_r²)",
    "latent_structure_and_robustness": "Latent Structure & Robustness",
    "generalization": "Generalization",
    "compression": "Compression",
}

# Short labels used on radar axes.
SCORE_SHORT: dict[str, str] = {
    "single_neuron_recon": "SN Recon",
    "population_structure": "Pop. Structure",
    "decoding": "Decoding (POYO + lin.)",
    "downstream_encoding": "Encoding",
    "latent_structure_and_robustness": "Latent Struct.+Rob.",
    "generalization": "Generalization",
    "compression": "Compression",
}

# Display names for wide-table columns whose metric name alone does not say what they are.
WIDE_METRIC_LABELS: dict[str, str] = {
    "decoding/test_r2": SCORE_LABELS["decoding_poyo"],
    "downstream_decoding/behavioral_classification_trial_target_id": (
        "Linear decoding: trial-target accuracy"
    ),
}

# Display names for wide-table block prefixes ({block}/{metric}), used for the metric
# dropdown's optgroup headings and the category suffix on axis labels. Only blocks whose
# raw name is genuinely ambiguous are listed; anything absent falls back to the block name
# with underscores spaced out, so a new block needs no entry here.
#
# The two robustness blocks are the reason this map exists: "robustness" vs
# "robustness_latent" gives a reader no way to tell which perturbs the *input* and which
# perturbs the *latent*, and they are separate eval suites (M1 / M2).
BLOCK_LABELS: dict[str, str] = {
    "robustness": "input corruption, M1",
    "robustness_latent": "latent noise, M2",
}


def block_label(block: str) -> str:
    """Display name for a wide-table block prefix."""
    return BLOCK_LABELS.get(block, block.replace("_", " "))


# ── Metric-path aliases ────────────────────────────────────────────────────────
# Maps short alias → canonical composite score column name.
_SCORE_ALIASES: dict[str, str] = {
    "sn_recon": "single_neuron_recon",
    "encoding": "downstream_encoding",
    "gen": "generalization",
    "comp": "compression",
    # identity entries so canonical names also resolve through the dict
    **{v: v for v in COMPOSITE_SCORES},
}

# Per-metric path prefix → diagnostics column name.
_DIAG_PREFIX_MAP: dict[str, str] = {
    "sn_recon": "single_neuron_recon",
    "single_neuron_recon": "single_neuron_recon",
    "population_structure": "population_structure",
    "latent_structure_and_robustness": "latent_structure_and_robustness",
}

# ── Tokenizer family inference ─────────────────────────────────────────────────
_FAMILY_PATTERNS: list[tuple[str, str]] = [
    (r"^wanx_A_vae_", "wanx_A_vae"),
    (r"^wanx_A_mmd_", "wanx_A_mmd"),
    (r"^wanx_A_bsq_", "wanx_A_bsq"),
    (r"^wanx_A_fsq_", "wanx_A_fsq"),
    (r"^wanx_A_vqpt_", "wanx_A_vqpt"),
    (r"^wanx_A_dprior_", "wanx_A_dprior"),
    (r"^wanx_B_", "wanx_B"),
    (r"^cosmos_", "cosmos"),
    (r"^compressed_sensing_", "compressed_sensing"),
    (r"^calm_", "calm"),
    (r"^PCA_", "PCA"),
    (r"^LDNS", "LDNS"),
    (r"^unigram_", "unigram"),
    (r"^ppe_", "ppe"),
    (r"^bpe_", "bpe"),
    (r"^Poisson", "Poisson"),
]


def infer_family(name: str) -> str:
    for pat, family in _FAMILY_PATTERNS:
        if re.match(pat, name):
            return family
    return name


# ── Simple ("PI-friendly") view config ──────────────────────────────────────────
# Pretty, family-level labels for the simple view (the wanx_A_* families are the same
# WAN architecture under different bottlenecks). Families absent here fall back to the
# raw family string.
FAMILY_PRETTY: dict[str, str] = {
    "wanx_A_vae": "VAE",
    "wanx_A_mmd": "MMD-VAE",
    "wanx_A_bsq": "BSQ",
    "wanx_A_fsq": "FSQ",
    "wanx_A_vqpt": "VQ-PT",
    "wanx_A_dprior": "Diffusion-Prior",
    "compressed_sensing": "Compressed Sensing",
    "calm": "CaLM",
    "wanx_B": "VAE-B",
    "cosmos": "Cosmos",
    "PCA": "PCA",
    "LDNS": "LDNS",
}


def family_pretty(family: str) -> str:
    return FAMILY_PRETTY.get(family, family)


# Raw {block}/{metric} columns surfaced in the simple view's metric pickers (the most
# explainable one or two per block). Composites are always kept on top of these. Raw
# robustness perturbations and the Ridge behavioral probes are intentionally omitted
# (unreadable names / covered by the latent_structure_and_robustness composite + decoding).
SIMPLE_METRICS: list[str] = [
    "single_unit_statistics/firing_rate_error",
    "single_unit_statistics/acg_correlation",
    "temporal_fidelity/spike_detection_f1",
    "temporal_fidelity/spike_count_correlation",
    "population_structure/population_coupling_preservation",
    "population_structure/participation_ratio_error",
    "latent_quality/population_structure_preservation",
    "encoding/nc_r2",
    "decoding/test_r2",
]


# ── Family palette ─────────────────────────────────────────────────────────────
# Single source of truth for family → colour, shared by every plotter.
_FAMILY_COLORS: dict[str, str] = {
    "wanx_A_vae": "#4C72B0",
    "wanx_A_mmd": "#DD8452",
    "wanx_A_bsq": "#55A868",
    "wanx_A_fsq": "#C44E52",
    "wanx_A_vqpt": "#8172B3",
    "wanx_A_dprior": "#937860",
    "wanx_B": "#DA8BC3",
    "cosmos": "#8C8C8C",
    "compressed_sensing": "#CCB974",
    "calm": "#64B5CD",
    "PCA": "#2CA02C",
    "LDNS": "#17BECF",
    "unigram": "#BCBD22",
    "ppe": "#FF7F0E",
    "bpe": "#9467BD",
    "Poisson": "#E377C2",
}

_FALLBACK_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def family_color(family: str, seen: dict[str, str]) -> str:
    """Resolve a family to a stable colour, assigning fallbacks for unknowns.

    ``seen`` accumulates fallback assignments so repeated calls within one
    figure stay consistent.
    """
    if family in _FAMILY_COLORS:
        return _FAMILY_COLORS[family]
    if family not in seen:
        idx = len(seen) % len(_FALLBACK_COLORS)
        seen[family] = _FALLBACK_COLORS[idx]
    return seen[family]


# ── Metric extraction ──────────────────────────────────────────────────────────


def extract_metric(data: dict[str, Any], path: str) -> float | None:
    """Resolve a metric path to a float from a ``category_scores.json`` dict.

    Path formats::

        "single_neuron_recon"               # composite score (or alias e.g. "sn_recon")
        "sn_recon/spike_f1"                 # per-metric within single_neuron_recon
        "population_structure/participation_ratio"     # per-metric within population_structure
        "latent_structure_and_robustness/robustness_input/jitter|k_ms=1.0|p=0.2"  # name has /
    """
    slash = path.find("/")
    if slash != -1:
        prefix = path[:slash]
        metric_name = path[slash + 1 :]
        col = _DIAG_PREFIX_MAP.get(prefix, prefix)
        diag = (data.get("diagnostics") or {}).get(col)
        if not isinstance(diag, dict):
            return None
        entry = (diag.get("per_metric") or {}).get(metric_name)
        if not isinstance(entry, dict):
            return None
        v = entry.get("normalized")
    else:
        canonical = _SCORE_ALIASES.get(path, path)
        v = (data.get("scores") or {}).get(canonical)

    if v is None:
        return None
    try:
        f = float(v)
        return None if not math.isfinite(f) else f
    except (TypeError, ValueError):
        return None


def metric_label(path: str) -> str:
    if "/" not in path:
        canonical = _SCORE_ALIASES.get(path, path)
        return SCORE_LABELS.get(canonical, canonical.replace("_", " ").title())
    prefix, metric_name = path.split("/", 1)
    col = _DIAG_PREFIX_MAP.get(prefix, prefix)
    col_label = SCORE_LABELS.get(col, col)
    return f"{col_label}\n{metric_name}"


# ── Data loading ───────────────────────────────────────────────────────────────


def _load_from_scores_dir(
    tokenizer: str, scores_dir: Path, verbose: bool
) -> dict[str, Any] | None:
    """Read ``<scores_dir>/<slug>.json``, or ``None`` if it is absent/unreadable.

    A miss is a skip so the rest of the cohort still plots, but it is **not** benign:
    ``scores/`` is allowed to lag ``category_scores.csv`` by a run, so a tokenizer that
    is in the headline table can be missing here. :func:`load_scores` therefore reports
    every skip unconditionally — see ``_warn_skipped``.
    """
    slug = tracker_io.scores_slug(tokenizer)
    if not slug:
        return None
    path = scores_dir / f"{slug}.json"
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        if verbose:
            print(f"  WARN: {tokenizer}: {path.name}: {exc}", file=sys.stderr)
        return None


def _warn_skipped(skipped: list[str], scores_dir: Path | None) -> None:
    """Announce tokenizers either route could not resolve. Never silent.

    A table or plot that quietly omits rows is the failure this tooling exists to
    prevent, and ``verbose`` is off by default in all four CLIs — so this does not
    check it. It goes to stderr because ``build_latex_table ... > paper_table_agg.tex``
    is a documented invocation and a warning on stdout would land inside the ``.tex``.

    ``scores_dir`` is the local route's directory, or ``None`` on the MLflow route —
    which drops rows just as silently, and for reasons (expired credentials, a deleted
    run, a network blip) a reader is even less likely to guess at.
    """
    if not skipped:
        return
    shown = ", ".join(skipped[:10]) + (
        f", … (+{len(skipped) - 10} more)" if len(skipped) > 10 else ""
    )
    where = (
        f"had no readable <slug>.json under {scores_dir}"
        if scores_dir is not None
        else "have an mlflow_aggregation URL whose category_scores.json could not be loaded"
    )
    fix = (
        "Re-run build_aggregation_tables.py to score them."
        if scores_dir is not None
        else "Check MLflow credentials and that those runs still exist."
    )
    print(
        f"WARNING: {len(skipped)} tracker row(s) {where} and are MISSING from this "
        f"output: {shown}. {fix}",
        file=sys.stderr,
    )


def _load_from_mlflow(
    tokenizer: str, agg_url: str, cache_dir: Path | None, verbose: bool
) -> dict[str, Any] | None:
    """Download the row's ``category_scores.json`` artifact, honouring the cache."""
    ids = tracker_io.parse_run_url(agg_url)
    if ids is None:
        return None
    exp_id, run_id = ids
    slug = tracker_io.scores_slug(tokenizer) or run_id[:12]

    uri = tracker_io.artifact_uri(exp_id, run_id, "category_scores.json")
    source = tracker_io.artifact_cache_key(uri)
    local_path = (cache_dir / f"{slug}-{source}.json") if cache_dir else None
    if local_path and local_path.exists():
        with open(local_path) as f:
            return json.load(f)

    try:
        # By run id, which resolves on any tracking store; an mlflow-artifacts:/ URI only
        # resolves against an HTTP server. `uri` stays the cache key, so caches stay valid.
        scores_data = tracker_io.load_json_artifact(f"runs:/{run_id}/category_scores.json")
    except Exception as exc:
        if verbose:
            print(f"  WARN: {tokenizer}: {exc}", file=sys.stderr)
        return None
    if local_path:
        # Cache is an optimization; a write failure must not drop the row.
        try:
            local_path.write_text(json.dumps(scores_data, indent=2))
        except OSError as exc:
            if verbose:
                print(f"  WARN: could not cache {tokenizer}: {exc}", file=sys.stderr)
    return scores_data


def _raise_on_links_to_another_server(agg_urls: pd.Series) -> None:
    """Runs are read from the configured server, never from the one a link names: the
    link is sheet data, and the MLflow credentials go wherever the request goes. A link
    to another server would only fail as an unreadable row, so name the server here."""
    active = mlflow.get_tracking_uri().rstrip("/")
    servers = {tracker_io.run_url_tracking_uri(str(url).strip()) for url in agg_urls.dropna()}
    others = sorted(s for s in servers if s and s != active)
    if others:
        raise ValueError(
            f"The tracker links runs on {others}, but the MLflow route reads from {active} "
            f"(MLFLOW_TRACKING_URI). Set MLFLOW_TRACKING_URI to the server of the links, or "
            f"pass mlflow_uri / --mlflow-uri to read every run from that one server."
        )


def load_scores(
    tracker_csv: Path | str,
    mlflow_uri: str | None = None,
    cache_dir: Path | None = None,
    verbose: bool = False,
    scores_dir: Path | str | None = None,
    common_parts_only: bool = False,
) -> pd.DataFrame:
    """Load ``category_scores.json`` for every row in the tracker.

    ``tracker_csv`` may be a local file path or a Google Sheets URL
    (``https://docs.google.com/spreadsheets/d/{ID}/edit#gid={GID}``).
    The sheet must be publicly accessible or shared with "Anyone with the link."

    Returns a DataFrame with one row per tokenizer, columns:
    ``tokenizer``, ``family``, ``latent_type``, ``tokenization_regime``,
    ``variant``, ``_scores_data`` (raw dict), plus the 7 composite score
    columns (float, ``nan`` when absent). Rows whose scores cannot be resolved
    are skipped.

    Parameters
    ----------
    mlflow_uri:
        Tracking server for the MLflow route; default ``MLFLOW_TRACKING_URI``. Without it,
        run links to a server other than ``MLFLOW_TRACKING_URI`` raise; with it, every run
        is read from ``mlflow_uri``, whichever server its link names.
    cache_dir:
        If given, downloaded JSONs are saved as ``<cache_dir>/<slug>-<source>.json``
        and reused on subsequent runs to avoid repeated MLflow downloads.
        MLflow route only.
    scores_dir:
        If given, ``category_scores.json`` is read from
        ``<scores_dir>/<slug>.json`` — the layout written by
        ``scripts/analysis/build_aggregation_tables.py`` into ``<outdir>/scores``
        — and MLflow is never contacted. The tracker then needs only a
        ``tokenizer`` column, not ``mlflow_aggregation``.
    common_parts_only:
        Report aspects of the scores that cannot be compared across tokenizers (not
        saved by the eval files) as a warning instead of raising. Scores that include
        a part (input- or latent-corruption robustness, POYO, ridge) only for some
        tokenizers always raise: rebuild them with
        ``build_aggregation_tables.py --common-parts-only``.
    """
    local_root = Path(scores_dir) if scores_dir is not None else None
    if local_root is None:
        if mlflow_uri:
            mlflow.set_tracking_uri(mlflow_uri)
        elif not os.environ.get("MLFLOW_TRACKING_URI"):
            raise ValueError(
                "The MLflow route needs a tracking server: set MLFLOW_TRACKING_URI (e.g. "
                "`set -a && source .env && set +a`) or pass mlflow_uri / --mlflow-uri. To "
                "read a build's scores from disk instead, pass scores_dir / --scores-dir."
            )

    source = tracker_io.to_csv_export_url(str(tracker_csv))
    if verbose and source != str(tracker_csv):
        print(f"  Google Sheets → {source}", file=sys.stderr)
    tracker = pd.read_csv(source)

    if local_root is None and "mlflow_aggregation" not in tracker.columns:
        raise ValueError(
            "Tracker CSV has no 'mlflow_aggregation' column.\n"
            "Either read the tracker_with_agg.csv of a build_aggregation_tables.py "
            "--log-mlflow run, or pass scores_dir=<outdir>/scores from a "
            "scripts/analysis/build_aggregation_tables.py run."
        )
    if local_root is None and not mlflow_uri:
        _raise_on_links_to_another_server(tracker["mlflow_aggregation"])

    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for _, row in tracker.iterrows():
        tokenizer = str(row.get("tokenizer", "")).strip()

        # The two routes differ only in where the payload comes from; everything
        # below is shared so both frames are byte-for-byte the same shape.
        if local_root is not None:
            scores_data = _load_from_scores_dir(tokenizer, local_root, verbose)
            if scores_data is None and tokenizer:
                skipped.append(tokenizer)
        else:
            agg_url = str(row.get("mlflow_aggregation", "") or "").strip()
            scores_data = _load_from_mlflow(tokenizer, agg_url, cache_dir, verbose)
            # A row with no run URL is the normal not-yet-aggregated state; warning on
            # it would fire on every tracker with a pending row. A row that HAS a URL
            # and still resolved to nothing is a genuine, silently dropped result.
            if scores_data is None and tokenizer and tracker_io.parse_run_url(agg_url) is not None:
                skipped.append(tokenizer)
        if scores_data is None:
            continue

        entry: dict[str, Any] = {
            "tokenizer": tokenizer,
            "family": infer_family(tokenizer),
            "latent_type": str(row.get("latent_type", "") or ""),
            "tokenization_regime": str(row.get("tokenization_regime", "") or ""),
            "variant": str(row.get("variant", "") or ""),
            "_scores_data": scores_data,
            # Copy the status flags into the table rows to mark the scores that get asterisks.
            **{
                key: value
                for key, value in provenance_record(scores_data).items()
                if key.startswith("position_regression_")
            },
        }
        for col in COMPOSITE_SCORES:
            v = (scores_data.get("scores") or {}).get(col)
            entry[col] = float(v) if v is not None else float("nan")

        rows.append(entry)
        if verbose:
            # stderr, not stdout: `build_latex_table -v ... > paper.tex` is a documented
            # invocation and this trace would otherwise be line 1 of the .tex.
            print(f"  loaded {tokenizer}", file=sys.stderr)

    _warn_skipped(skipped, local_root)
    # Warn when the loaded tokenizers were averaged differently. Unconditional and on
    # stderr, for the same reasons as _warn_skipped.
    rows_with_provenance = [
        {**row_entry, **provenance_record(row_entry["_scores_data"])} for row_entry in rows
    ]
    raise_on_mixed_chunk_sizes(rows_with_provenance)
    for line in find_mixed_member_rows(rows_with_provenance):
        print(line, file=sys.stderr)
    # A part (input- or latent-corruption robustness, POYO, ridge) in some tokenizers'
    # scores but not others' makes the table wrong, not just imprecise: refuse to plot it.
    raise_on_mixed_score_parts(
        rows_with_provenance,
        how_to_fix=(
            f"Rebuild the scores with build_aggregation_tables.py {COMMON_PARTS_ONLY_FLAG} "
            f"to build every tokenizer's score only from the parts all tokenizers have "
            f"(all tokenizers stay in the table)."
        ),
    )
    raise_if_not_comparable(rows_with_provenance, common_parts_only=common_parts_only)

    if not rows:
        if local_root is not None:
            raise RuntimeError(
                f"No scores could be loaded from scores_dir={local_root}. Check that "
                "build_aggregation_tables.py has been run and that its <slug>.json "
                "filenames match the tracker's 'tokenizer' column."
            )
        raise RuntimeError(
            "No scores could be loaded. Check MLflow credentials and that "
            "build_aggregation_tables.py --log-mlflow has published the scores."
        )

    return pd.DataFrame(rows)
