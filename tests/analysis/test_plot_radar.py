"""Radar-frame shaping tests for the ``compression`` axis.

``compression`` is a raw bit-ratio (``input_bits / output_bits``), not a
``[0, 1]`` score like its six peers. These tests pin the presentation contract:
it gets its own ceiling-based scale rather than being folded into the shared
``score_clip``.

Fixture values track a real 24-tokenizer cohort, where
``compression_rate`` runs 0.39-1.50 (21 of 24 below 1.0 — the latent is larger
than the input). A ceiling calibrated to some other range flattens the axis,
which is exactly what this module guards against.
"""

from __future__ import annotations

import pandas as pd
import pytest
from scripts.analysis.data import COMPOSITE_SCORES
from scripts.analysis.plot_radar import scores_to_radar_frame

# Observed extremes of the real cohort (01_control_noncausal / 06d_nuclr_w0p25).
COHORT_MIN = 0.390625
COHORT_MAX = 1.5024038461538463


def _frame(compression_values):
    rows = []
    for i, comp in enumerate(compression_values):
        row: dict[str, object] = dict.fromkeys(COMPOSITE_SCORES, 0.5)
        row["compression"] = comp
        row["tokenizer"] = f"tok{i}"
        rows.append(row)
    return pd.DataFrame(rows)


def _compression_column(out: pd.DataFrame) -> str:
    cols = [c for c in out.columns if "ompress" in c]
    assert len(cols) == 1, f"expected exactly one compression axis, got {cols}"
    return cols[0]


def test_compression_axis_is_not_pinned_at_one():
    """compression is a bit-ratio, not a [0,1] score.

    Folding it into the global --clip pinned every tokenizer's compression axis
    at exactly 1.0 — an informationless axis reading as 'all tokenizers perfect'.
    """
    df = _frame([0.5, 1.5])
    out = scores_to_radar_frame(df, ["tok0", "tok1"], score_clip=1.0)

    col = _compression_column(out)
    values = list(out[col])
    assert values[0] != values[1], "distinct compression rates must give distinct axis values"
    # 0.5/2 = 0.25, 1.5/2 = 0.75 with the default ceiling
    assert values[0] == pytest.approx(0.25)
    assert values[1] == pytest.approx(0.75)


def test_default_ceiling_spreads_the_real_cohort_across_the_axis():
    """The default ceiling must be calibrated to the data it will actually plot.

    A ceiling of 50 (the placeholder in ``ntt.evaluation.radar``) maps this
    cohort onto 0.008-0.030 — visually flat. Its default must separate real
    tokenizers.
    """
    df = _frame([COHORT_MIN, COHORT_MAX])
    out = scores_to_radar_frame(df, ["tok0", "tok1"], score_clip=1.0)
    lo, hi = list(out[_compression_column(out)])

    # 0.390625/2 and 1.5024038.../2, by hand.
    assert lo == pytest.approx(0.1953125)
    assert hi == pytest.approx(0.7512019230769231)
    # The cohort must occupy a usable share of the axis, not hug the hub.
    assert hi - lo > 0.4


def test_compression_above_ceiling_clamps_to_one():
    """A rate past the ceiling saturates rather than overflowing the ring."""
    df = _frame([3.0])  # 3/2 = 1.5, above score_clip
    out = scores_to_radar_frame(df, ["tok0"], score_clip=1.0)
    col = _compression_column(out)
    assert list(out[col])[0] == pytest.approx(1.0)


def test_compression_ceiling_is_configurable():
    """A wider ceiling rescales the axis: 1/4 = 0.25, 3/4 = 0.75."""
    df = _frame([1.0, 3.0])
    out = scores_to_radar_frame(df, ["tok0", "tok1"], score_clip=1.0, compression_ceiling=4.0)
    col = _compression_column(out)
    assert list(out[col]) == pytest.approx([0.25, 0.75])


def test_compression_shares_the_score_clip_ceiling():
    """The radial axis runs to ``score_clip``; compression must respect it too.

    ``plot_radar`` is called with ``clip_range=(0.0, args.clip)``, so an axis
    value above ``score_clip`` would overflow the ring. 1.5/2 = 0.75 exceeds a
    ``score_clip`` of 0.5 and is clipped there, exactly as the peer scores are.
    """
    df = _frame([1.5])
    out = scores_to_radar_frame(df, ["tok0"], score_clip=0.5)
    col = _compression_column(out)
    assert list(out[col])[0] == pytest.approx(0.5)
    # ... and every peer axis (raw 0.5) is untouched by the compression scaling.
    peers = [c for c in out.columns if c != col]
    assert list(out.loc[out.index[0], peers]) == pytest.approx([0.5] * len(peers))


def test_compression_axis_label_states_the_ceiling():
    """The axis is a fraction of a ceiling, so the label must name the ceiling."""
    df = _frame([1.0])
    out = scores_to_radar_frame(df, ["tok0"], score_clip=1.0)
    assert _compression_column(out) == "Compression (max 2x)"
    # A non-default, non-integer ceiling is reflected verbatim.
    out = scores_to_radar_frame(df, ["tok0"], score_clip=1.0, compression_ceiling=1.6)
    assert _compression_column(out) == "Compression (max 1.6x)"


def test_non_positive_ceiling_is_rejected():
    """The ceiling is a divisor; 0 would be a ZeroDivisionError mid-loop."""
    df = _frame([1.0])
    with pytest.raises(ValueError, match="compression_ceiling must be positive"):
        scores_to_radar_frame(df, ["tok0"], compression_ceiling=0.0)


def test_missing_compression_still_floors_to_zero():
    """A NaN compression is a missing axis, not a 0-rate one; it must not count."""
    df = _frame([float("nan")])
    out = scores_to_radar_frame(df, ["tok0"], score_clip=1.0)
    col = _compression_column(out)
    assert list(out[col])[0] == pytest.approx(0.0)
    assert "(6/7 axes)" in out.index[0]
