#!/usr/bin/env python3
"""Build a LaTeX results table from aggregated category scores.

Produces a ``table*`` with one row per tokenizer (grouped by family),
matching the style of the paper's main results table.

Usage::

    uv run python -m scripts.analysis.build_latex_table \\
        /path/to/tracker_with_agg.csv \\
        --output paper_table_agg.tex

    # write to stdout (pipe into a file):
    uv run python -m scripts.analysis.build_latex_table tracker.csv

    # suppress bold (e.g. for appendix sub-table):
    uv run python -m scripts.analysis.build_latex_table tracker.csv --no-bold

    # sort families by median SN Recon rather than by best member:
    uv run python -m scripts.analysis.build_latex_table tracker.csv --sort-family median
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd
from scripts.analysis.data import COMPOSITE_SCORES, load_scores

# ── Column display config ──────────────────────────────────────────────────────

_COL_HEADERS = {
    "single_neuron_recon": r"\makecell{SN\\Recon}",
    "population_structure": r"\makecell{Pop.\\Structure}",
    "decoding": r"\makecell{Decod-\\ing}",
    "downstream_encoding": r"\makecell{Encod-\\ing}",
    "latent_structure_and_robustness": r"\makecell{Latent\\Struct.+Rob.}",
    "generalization": r"\makecell{General-\\ization}",
    "compression": r"\makecell{Compres-\\sion}",
}

# Columns where higher = better for bolding purposes.
_HIGHER_IS_BETTER = {
    "single_neuron_recon",
    "population_structure",
    "decoding",
    "downstream_encoding",
    "latent_structure_and_robustness",
    "generalization",
    "compression",
}

# Primary sort column within families and for family ordering.
_PRIMARY_SORT = "single_neuron_recon"

# Default caption. This ships into the paper, so every clause is a claim about
# what ``ntt.evaluation.aggregation.category_scores`` actually computes under the
# production job's defaults (``ntt/jobs/aggregation.py``: ``score_clip=-1.0``,
# ``include_m2=True``, ``compression_metric="compression_rate"``) — note the
# pure-library default is ``include_m2=False``, which the paper cohort does not
# use. ``tests/analysis/test_build_latex_table.py`` pins the wording. Do not edit
# without re-reading the scorers.
DEFAULT_CAPTION = (
    r"\textbf{All scored tokenizers across all seven evaluation framework "
    r"dimensions.} \emph{SN Recon} and \emph{Pop.\ Structure} are family-weighted, "
    r"Poisson-normalized aggregates ($0=$ Poisson floor, $1=$ the "
    r"perfect-reconstruction ceiling; below $0$ is worse than Poisson, with "
    r"per-metric scores clipped at $-1$; for firing-rate error, $0$ is instead every "
    r"unit given the population's mean rate). \emph{Decoding} is raw $R^2$ pooling the "
    r"POYO sessions and the in-repo ridge decoders as two equal families (not a "
    r"flat mean over sessions); \emph{Encoding} is the mean noise-ceiling-corrected "
    r"$R^2$. \emph{Latent Struct.+Rob.}\ is an equally weighted mean over the "
    r"members it can compute, each mapped onto $[0,1]$: the two RSA "
    r"structure-preservation correlations, M1 input-corruption retention, and M2 "
    r"latent-noise ACG correlation. \emph{Generalization} is the geometric mean, over "
    r"the reconstruction metrics, of the retention on datasets unseen in training "
    r"relative to the in-distribution sessions, clipped to $[0,1]$. \emph{Compression} "
    r"is the raw bit-ratio (input bits / output bits), not a normalized score: "
    r"higher means more compression, and a value below $1$ means the latent is "
    r"larger than the input. Rows sorted by SN Recon descending within each "
    r"family. Dashes: unavailable data."
)


# ── LaTeX helpers ──────────────────────────────────────────────────────────────


def _tex_escape_name(s: str) -> str:
    """Escape a tokenizer slug for LaTeX body text."""
    # Replace hyphens with \mbox{-} (prevents bad line-breaks in narrow columns).
    s = s.replace("-", r"\mbox{-}")
    # Escape underscores.
    s = s.replace("_", r"\_")
    return s


def _fmt_score(v: float | None) -> str:
    """Format a score value for the table cell.

    Every column gets three decimals, including ``compression`` — which is a
    raw bit-ratio (``input_bits / output_bits``) rather than a [0,1] score, but
    whose observed spread is narrow enough (~0.39-1.50) that fewer decimals
    would collapse distinct tokenizers onto the same cell. There is
    deliberately no per-column branch here: one that returned the default
    string was dead code asserting a difference that did not exist.
    """
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    return f"{v:.3f}"


def _bold(s: str) -> str:
    return r"\textbf{" + s + "}"


# ── Core builder ──────────────────────────────────────────────────────────────


def build_table(
    df: pd.DataFrame,
    bold: bool = True,
    sort_family_by: str = "best",  # "best" | "median"
    caption: str | None = None,
    label: str = "tab:all_results_agg",
) -> str:
    """Return the full LaTeX ``table*`` string."""

    has_position = any(
        str(value) == "True" for value in df.get("position_regression_included", [])
    )
    # ── Find global best per column (for bolding) ──────────────────────────────
    best: dict[str, float] = {}
    if bold:
        for col in COMPOSITE_SCORES:
            vals = [r[col] for _, r in df.iterrows() if not math.isnan(r[col])]
            if vals:
                best[col] = max(vals) if col in _HIGHER_IS_BETTER else min(vals)

    # ── Group and sort ─────────────────────────────────────────────────────────
    families: dict[str, list[pd.Series]] = {}
    for _, row in df.iterrows():
        families.setdefault(row["family"], []).append(row)

    def _family_sort_key(fam: str) -> float:
        vals = [r[_PRIMARY_SORT] for r in families[fam] if not math.isnan(r[_PRIMARY_SORT])]
        if not vals:
            return -math.inf
        return max(vals) if sort_family_by == "best" else sum(vals) / len(vals)

    sorted_families = sorted(families.keys(), key=_family_sort_key, reverse=True)

    # ── Build rows ─────────────────────────────────────────────────────────────
    body_lines: list[str] = []
    n_cols = 1 + len(COMPOSITE_SCORES)  # tokenizer + 7 scores

    for fam_idx, fam in enumerate(sorted_families):
        members = sorted(
            families[fam],
            key=lambda r: r[_PRIMARY_SORT] if not math.isnan(r[_PRIMARY_SORT]) else -math.inf,
            reverse=True,
        )

        # Family header row.
        fam_display = _tex_escape_name(fam)
        body_lines.append(rf"\multicolumn{{{n_cols}}}{{l}}{{\textit{{{fam_display}}}}} \\[2pt]")

        for row in members:
            tok = _tex_escape_name(row["tokenizer"])
            # Optionally show variant/latent_type as a tiny annotation.
            variant = str(row.get("tokenization_regime", "") or "").strip()
            tok_cell = tok
            if variant and variant not in ("nan", ""):
                tok_cell += r" {\tiny \textit{" + variant + r"}}"

            cells: list[str] = [tok_cell]
            for col in COMPOSITE_SCORES:
                v = row[col]
                if math.isnan(v):
                    cells.append("--")
                    continue
                s = _fmt_score(v)
                if bold and col in best and abs(v - best[col]) < 1e-9:
                    s = _bold(s)
                # Asterisk only if position regression excluded here and included for another
                # tokenizer at 200 ms.
                if (
                    col == "decoding"
                    and has_position
                    and str(row.get("position_regression_excluded")) == "True"
                ):
                    s += r"$^{*}$"
                cells.append(s)

            body_lines.append(" & ".join(cells) + r" \\")

        # Thin separator between families (except after the last one).
        if fam_idx < len(sorted_families) - 1:
            body_lines.append(r"\addlinespace[4pt]")

    # ── Assemble ───────────────────────────────────────────────────────────────
    col_spec = "l" + "r" * len(COMPOSITE_SCORES)
    header_cells = ["Tokenizer"] + [_COL_HEADERS[c] for c in COMPOSITE_SCORES]
    header_row = " & ".join(header_cells) + r" \\"

    if caption is None:
        caption = DEFAULT_CAPTION

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\setlength{\tabcolsep}{4pt}",
        r"\small",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        header_row,
        r"\midrule",
        *body_lines,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


# ── CLI ────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a LaTeX results table from aggregated category scores.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "tracker_csv",
        help="Tracker CSV; needs an mlflow_aggregation column unless --scores-dir is given",
    )
    parser.add_argument("--output", "-o", help="Output .tex path (default: stdout)")
    parser.add_argument(
        "--sort-family",
        choices=["best", "median"],
        default="best",
        help="Order families by best member (default) or median SN Recon",
    )
    parser.add_argument("--no-bold", action="store_true", help="Skip bolding best values")
    parser.add_argument("--caption", default=None, help="Override default table caption")
    parser.add_argument("--label", default="tab:all_results_agg", help="LaTeX label")
    parser.add_argument("--cache-dir", help="Cache dir for downloaded category_scores.json files")
    parser.add_argument(
        "--scores-dir",
        help="Read category_scores.json from <dir>/<slug>.json instead of MLflow "
        "(build_aggregation_tables.py writes this as <outdir>/scores)",
    )
    parser.add_argument(
        "--common-parts-only",
        action="store_true",
        help="warn instead of failing when it cannot be checked whether all tokenizers "
        "were averaged the same way (see build_aggregation_tables.py --common-parts-only)",
    )
    parser.add_argument("--mlflow-uri", default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    load_kwargs: dict = {
        "tracker_csv": args.tracker_csv,
        "cache_dir": args.cache_dir,
        "verbose": args.verbose,
    }
    if args.mlflow_uri:
        load_kwargs["mlflow_uri"] = args.mlflow_uri
    if args.scores_dir:
        load_kwargs["scores_dir"] = args.scores_dir
    load_kwargs["common_parts_only"] = args.common_parts_only

    print("Loading scores …", file=sys.stderr)
    df = load_scores(**load_kwargs)
    print(f"  {len(df)} tokenizers loaded", file=sys.stderr)

    tex = build_table(
        df,
        bold=not args.no_bold,
        sort_family_by=args.sort_family,
        caption=args.caption,
        label=args.label,
    )

    if args.output:
        Path(args.output).write_text(tex)
        print(f"Wrote: {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(tex)


if __name__ == "__main__":
    main()
