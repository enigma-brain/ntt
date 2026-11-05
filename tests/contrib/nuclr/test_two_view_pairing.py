"""CPU tests for two-view pairing logic used by NuCLR contrastive learning.

Covers:
- ``compute_match_idx`` (data/two_view_types.py): the leakage-critical mapping
  from view1 unit rows to view2 unit rows by stable ``unit_number``. The match
  must index the RIGHT rows regardless of unit ordering between views.
- ``TwoViewNTTDataset._sample_view2_start`` (data/two_view_datamodule.py): the
  view-2 window start is sampled within ``max_time_diff`` of view1 and clamped to
  valid sampling intervals. Tested via a lightweight stand-in supplying only the
  attributes the unbound method reads (``_intervals``, ``max_time_diff``,
  ``window_length``).
"""

import random
import types

import numpy as np

from ntt.contrib.nuclr.data.two_view_datamodule import TwoViewNTTDataset
from ntt.contrib.nuclr.data.two_view_types import compute_match_idx


# --------------------------------------------------------------------------- #
# compute_match_idx
# --------------------------------------------------------------------------- #
def test_match_idx_identical_units_is_identity() -> None:
    """When both views hold the same units in the same order, the match is the
    diagonal: row i of view1 pairs with row i of view2."""
    ids = np.array([10, 20, 30, 40])
    idx = compute_match_idx(ids, ids.copy())

    assert idx.shape == (2, 4)
    # Row 0 indexes view1, row 1 indexes view2; np.where yields sorted pairs.
    assert idx[0].tolist() == [0, 1, 2, 3]
    assert idx[1].tolist() == [0, 1, 2, 3]


def test_match_idx_partial_overlap_keeps_only_shared_units() -> None:
    """Only units present in both views are matched, at their respective rows."""
    ids1 = np.array([10, 20, 30])  # rows 0,1,2
    ids2 = np.array([20, 30, 40])  # rows 0,1,2
    idx = compute_match_idx(ids1, ids2)

    # Shared units are 20 and 30.
    # 20: view1 row 1 <-> view2 row 0; 30: view1 row 2 <-> view2 row 1.
    pairs = set(zip(idx[0].tolist(), idx[1].tolist(), strict=True))
    assert pairs == {(1, 0), (2, 1)}
    assert idx.shape == (2, 2)
    # Every matched pair must reference the SAME underlying unit_number.
    for v1_row, v2_row in pairs:
        assert ids1[v1_row] == ids2[v2_row]


def test_match_idx_no_overlap_is_empty() -> None:
    """Disjoint unit sets -> a (2, 0) match (no positives, no leakage)."""
    idx = compute_match_idx(np.array([1, 2, 3]), np.array([4, 5, 6]))
    assert idx.shape == (2, 0)


def test_match_idx_respects_differing_unit_orderings() -> None:
    """The leakage-critical invariant: when the two views order their shared
    units differently, the match must still pair rows that share a unit_number,
    NOT rows that merely share a position."""
    ids1 = np.array([30, 10, 20])  # rows: 30->0, 10->1, 20->2
    ids2 = np.array([10, 20, 30])  # rows: 10->0, 20->1, 30->2
    idx = compute_match_idx(ids1, ids2)

    pairs = set(zip(idx[0].tolist(), idx[1].tolist(), strict=True))
    # 30: v1 row 0 <-> v2 row 2; 10: v1 row 1 <-> v2 row 0; 20: v1 row 2 <-> v2 row 1.
    assert pairs == {(0, 2), (1, 0), (2, 1)}
    # Positional pairing would have been {(0,0),(1,1),(2,2)} -- which is WRONG.
    assert pairs != {(0, 0), (1, 1), (2, 2)}
    # Cross-check: matched rows always reference the same unit number.
    for v1_row, v2_row in pairs:
        assert ids1[v1_row] == ids2[v2_row]


def test_match_idx_handles_duplicate_unit_ids() -> None:
    """If a unit_number appears more than once, every (v1, v2) co-occurrence is
    matched (broadcast equality semantics)."""
    ids1 = np.array([5, 5, 7])
    ids2 = np.array([5, 7])
    idx = compute_match_idx(ids1, ids2)

    pairs = set(zip(idx[0].tolist(), idx[1].tolist(), strict=True))
    # Both view1 rows holding unit 5 pair with view2 row 0; unit 7 pairs row 2<->1.
    assert pairs == {(0, 0), (1, 0), (2, 1)}


# --------------------------------------------------------------------------- #
# _sample_view2_start
# --------------------------------------------------------------------------- #
def _make_dataset_stub(interval_starts, interval_ends, window_length, max_time_diff):
    """Minimal stand-in exposing only what ``_sample_view2_start`` reads.

    ``self._intervals[recording_id]`` must have ``.start`` / ``.end`` arrays.
    """
    interval = types.SimpleNamespace(
        start=np.asarray(interval_starts, dtype=float),
        end=np.asarray(interval_ends, dtype=float),
    )
    stub = types.SimpleNamespace(
        _intervals={"rec": interval},
        window_length=float(window_length),
        max_time_diff=float(max_time_diff),
    )
    return stub


def test_sample_view2_start_within_max_time_diff_and_interval() -> None:
    """Sampled start lies within [v1 - max_time_diff, v1 + max_time_diff] AND
    leaves room for a full window inside the interval."""
    # One long interval [0, 100]; window 5s; max_time_diff 10s; view1_start=50.
    stub = _make_dataset_stub([0.0], [100.0], window_length=5.0, max_time_diff=10.0)
    v1 = 50.0
    for seed in range(50):
        random.seed(seed)
        start2 = TwoViewNTTDataset._sample_view2_start(stub, "rec", v1)
        # Within max_time_diff of view1.
        assert v1 - 10.0 <= start2 <= v1 + 10.0
        # A full window fits inside the interval.
        assert 0.0 <= start2
        assert start2 + 5.0 <= 100.0


def test_sample_view2_start_clamped_to_interval_bounds() -> None:
    """Near the right edge of an interval, the window must not run past the end;
    the sampled start is capped at ``end - window_length`` even though
    ``view1 + max_time_diff`` would allow later."""
    # Interval [0, 20]; window 5s -> latest legal start is 15.
    # view1_start=18, max_time_diff=10 -> target_hi=28, but clamp to 15.
    stub = _make_dataset_stub([0.0], [20.0], window_length=5.0, max_time_diff=10.0)
    v1 = 18.0
    for seed in range(50):
        random.seed(seed)
        start2 = TwoViewNTTDataset._sample_view2_start(stub, "rec", v1)
        assert start2 + 5.0 <= 20.0  # window stays inside interval
        assert start2 >= v1 - 10.0  # still respects lower bound (= 8.0)
        assert start2 <= 15.0  # capped at end - window_length


def test_sample_view2_start_picks_only_in_range_interval() -> None:
    """With two intervals where only one overlaps the target range, the sample
    must come from the in-range interval."""
    # Interval A [0, 10] is far from view1; interval B [100, 130] is in range.
    stub = _make_dataset_stub([0.0, 100.0], [10.0, 130.0], window_length=5.0, max_time_diff=10.0)
    v1 = 110.0  # target range [100, 120]; only interval B overlaps it
    for seed in range(50):
        random.seed(seed)
        start2 = TwoViewNTTDataset._sample_view2_start(stub, "rec", v1)
        # Must land inside interval B's valid start range, never interval A.
        assert 100.0 <= start2
        assert start2 + 5.0 <= 130.0
        assert v1 - 10.0 <= start2 <= v1 + 10.0


def test_sample_view2_start_fallback_when_no_valid_range() -> None:
    """When no interval can fit a window within max_time_diff, fall back to the
    view1 start (documented behaviour)."""
    # Window 50s but interval only [0, 10] -> end - window_length = -40 < start.
    stub = _make_dataset_stub([0.0], [10.0], window_length=50.0, max_time_diff=5.0)
    v1 = 5.0
    random.seed(0)
    start2 = TwoViewNTTDataset._sample_view2_start(stub, "rec", v1)
    assert start2 == v1
