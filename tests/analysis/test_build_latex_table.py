"""Tests for the LaTeX results-table formatter.

Focus: ``compression`` is a raw bit-ratio (``input_bits / output_bits``), not a
``[0, 1]`` score. It is nevertheless formatted and bolded exactly like its six
peers — three decimals, per-column comparison — and these tests pin both halves
of that, because each has a plausible-looking wrong answer.

Also pins ``DEFAULT_CAPTION``, which ships into the paper. Untested prose drifts
from what the scorers do, so the claims are asserted here in both directions.

Fixture values track a real 24-tokenizer cohort, where
``compression_rate`` runs 0.39-1.50.
"""

from __future__ import annotations

import pandas as pd
from scripts.analysis.build_latex_table import DEFAULT_CAPTION, _fmt_score, build_table
from scripts.analysis.data import COMPOSITE_SCORES, SCORE_LABELS

# Two adjacent real cohort values (01_control_noncausal, 05c_nuclr_w2p0) and the
# cohort maximum (06d_nuclr_w0p25).
COHORT_LO = 0.390625
COHORT_NEXT = 0.4340277777777778
COHORT_HI = 1.5024038461538463

# ── _fmt_score ────────────────────────────────────────────────────────────────


def test_fmt_score_uses_three_decimals():
    assert _fmt_score(0.4237) == "0.424"
    assert _fmt_score(0.8) == "0.800"


def test_fmt_score_missing_values_are_dashes():
    assert _fmt_score(None) == "--"
    assert _fmt_score(float("nan")) == "--"


def test_three_decimals_keep_adjacent_compression_rates_distinct():
    """Coarser formatting would merge real, differently-performing tokenizers.

    The cohort spans 0.39-1.50, so one decimal collapses 0.391 and 0.434 — two
    distinct tokenizers — onto the same ``0.4`` cell. Three decimals is not a
    cosmetic choice here; it is the difference between a readable column and a
    misleading one.
    """
    assert _fmt_score(COHORT_LO) != _fmt_score(COHORT_NEXT)
    assert _fmt_score(COHORT_LO) == "0.391"
    assert _fmt_score(COHORT_NEXT) == "0.434"


# ── build_table bolding ───────────────────────────────────────────────────────


def _table_df():
    """Two tokenizers whose per-column winners are deliberately split."""
    peers = [c for c in COMPOSITE_SCORES if c != "compression"]
    tok_a = {"tokenizer": "tok_a", "family": "fam", "compression": COHORT_LO}
    tok_b = {"tokenizer": "tok_b", "family": "fam", "compression": COHORT_HI}
    tok_a["single_neuron_recon"] = 0.9
    tok_b["single_neuron_recon"] = 0.2
    for c in peers:
        if c == "single_neuron_recon":
            continue
        tok_a[c] = 0.1
        tok_b[c] = 0.8
    return pd.DataFrame([tok_a, tok_b])


def test_bolding_is_per_column_and_compression_does_not_leak():
    """``best`` is keyed per column, so the ratio never competes with a score.

    1.502 is the largest number anywhere in the table, yet the winners in the
    six normalized columns are unaffected — which is only true because the
    comparison is within-column.
    """
    tex = build_table(_table_df(), bold=True)

    # compression: the cohort max wins its own column, the cohort min does not.
    assert r"\textbf{1.502}" in tex
    assert r"\textbf{0.391}" not in tex
    # the normalized columns keep their own winners, split across both rows.
    assert r"\textbf{0.900}" in tex  # tok_a wins single_neuron_recon
    assert r"\textbf{0.800}" in tex  # tok_b wins the other five
    assert r"\textbf{0.200}" not in tex
    assert r"\textbf{0.100}" not in tex


def test_no_bold_emits_plain_cells():
    tex = build_table(_table_df(), bold=False)
    assert r"\textbf{1.502}" not in tex
    # The whole cell, not a substring: `"1.502" in tex` also matches \textbf{1.502},
    # so it survives a regression that puts the bolding back.
    assert r"& 1.502 \\" in tex


# ── default caption ───────────────────────────────────────────────────────────
#
# The caption is the only place a reader learns what the seven numbers mean, so
# a wrong clause is a wrong paper. Each assertion below is checked against the
# scorer it describes in ``ntt/evaluation/aggregation/category_scores.py``.

# Clauses that a caption must not make, because each contradicts the scorer:
#   1. ``score_decoding_merged`` pools POYO sessions and ridge targets as two
#      equal families via ``family_weighted_mean`` — not a flat mean, and the
#      fused "Decoding and Encoding" clause also mislabelled encoding, which is
#      ``overall.nc_r2.mean`` (noise-ceiling-corrected), not plain R².
#   2. ``score_latent_structure_and_robustness`` is an equally weighted mean over
#      four members under the job defaults (two RSA structure-preservation
#      correlations + M1 + M2); input corruption is one of them, not the column.
#   3. ``_score_compression`` is a scalar pass-through of ``compression_rate``,
#      the raw ``input_bits / output_bits`` ratio — nothing normalizes it.
_RETIRED_FALSE_CLAIMS = (
    r"\emph{Decoding} and \emph{Encoding} report mean $R^2$",
    r"a soft retention score under input perturbations",
    r"the normalized compression score",
)


def test_default_caption_does_not_reinstate_the_retired_false_claims():
    for claim in _RETIRED_FALSE_CLAIMS:
        assert claim not in DEFAULT_CAPTION


def test_default_caption_describes_decoding_as_two_equal_families():
    """``score_decoding_merged`` family-weights POYO vs ridge; the old caption
    called the column a flat mean R², which would hand ~12 POYO sessions 12/14
    of it."""
    assert "two equal families" in DEFAULT_CAPTION
    assert "not a flat mean over sessions" in DEFAULT_CAPTION
    # Encoding is noise-ceiling-corrected R² (``overall.nc_r2.mean``), and is no
    # longer described in the same breath as decoding.
    assert "noise-ceiling-corrected" in DEFAULT_CAPTION


def test_default_caption_names_the_latent_bucket_members():
    """The bucket is a mean over members, not an input-perturbation score."""
    assert "structure-preservation correlations" in DEFAULT_CAPTION
    assert "M1 input-corruption retention" in DEFAULT_CAPTION
    # M2 is ON by default in the production job (include_m2=True), so the
    # shipped table includes it even though the pure library defaults it off.
    assert "M2 latent-noise ACG correlation" in DEFAULT_CAPTION


def test_default_caption_calls_compression_a_raw_unnormalized_ratio():
    """The cohort runs 0.391-1.502 with most values below 1, so the caption must
    not let a reader assume a normalized score or a ratio above 1."""
    assert "raw bit-ratio (input bits / output bits)" in DEFAULT_CAPTION
    assert "not a normalized score" in DEFAULT_CAPTION
    assert "below $1$ means the latent is larger than the input" in DEFAULT_CAPTION


def test_default_caption_states_the_poisson_scale_and_its_clip():
    """``score_clip=-1.0`` is the job default, so the composite cannot fall below
    -1 however bad the tokenizer is."""
    assert "$0=$ Poisson floor" in DEFAULT_CAPTION
    assert "below $0$ is worse than Poisson" in DEFAULT_CAPTION
    assert r"clipped at $-1$" in DEFAULT_CAPTION
    # firing_rate_error is scored against a population-mean-rate reference instead.
    assert "for firing-rate error, $0$ is instead every unit given the population's mean rate" in (
        DEFAULT_CAPTION
    )


def test_build_table_emits_the_default_caption_and_honours_an_override():
    assert r"\caption{" + DEFAULT_CAPTION + "}" in build_table(_table_df())
    override = build_table(_table_df(), caption="Custom.")
    assert r"\caption{Custom.}" in override
    assert "raw bit-ratio" not in override


def test_decoding_axis_label_is_not_a_flat_mean():
    """The radar/plot axis label made the same claim the caption did."""
    assert SCORE_LABELS["decoding"] == "Decoding (POYO + linear)"
