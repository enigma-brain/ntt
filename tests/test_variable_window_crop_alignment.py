"""Tests for VariableWindowCrop's patch-alignment contract.

The class documents that sampled crop lengths are "snapped to the nearest valid
length that is a multiple of ``patch_length * bin_size_ms`` milliseconds, ensuring
the downstream patchification layer always receives an evenly divisible time axis".

The contract must also hold when the crop is clamped to a shorter
available interval — an unaligned `min(crop_sec, full_length)` would, with
`variable_t`, produce a window whose bin count is not a multiple
of the tokenizer's temporal compression, and `pool_nuclr_latents` (which
deliberately requires exact divisibility) would raise mid-training:

    ValueError: NuCLR latent time 43 must divide evenly into input time 4297.

Only truncated crops are affected, so such a failure is rare per step but
near-certain over a long run.
"""

import pytest
import torch

from ntt.core.components.dist import UniformDist
from ntt.core.data.components.transforms import VariableWindowCrop

BIN_MS = 1.0
PATCH = 100  # bins; a representative tokenizer temporal compression (25 * 2 * 2)
PATCH_SEC = PATCH * BIN_MS / 1000.0
TOL = 1e-9  # seconds; comparison slack for durations


class _FakeData:
    """Minimal stand-in exposing the start/end/slice surface the transform uses."""

    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end
        self.sliced_to: tuple[float, float] | None = None

    def slice(self, start: float, end: float) -> "_FakeData":
        out = _FakeData(start, end)
        out.sliced_to = (start, end)
        return out


def _crop(full_length_sec: float, dist_lo: float, dist_hi: float, seed: int) -> float:
    """Run one crop and return its duration in seconds.

    The seed goes into the *distribution*, not into `np.random`: `UniformDist`
    owns a private `np.random.default_rng(seed=self.seed)`, which the global
    `np.random.seed` does not reach. Seeding globally would leave the
    sampled crop length non-deterministic, so the truncation path these tests
    target would only be covered by chance. `torch.manual_seed` is still needed, but
    only fixes the offset of the crop within the interval.
    """
    torch.manual_seed(seed)
    t = VariableWindowCrop(
        dist=UniformDist(low=dist_lo, high=dist_hi, seed=seed),
        bin_size_ms=BIN_MS,
        patch_length=PATCH,
    )
    out = t(_FakeData(0.0, full_length_sec))
    return out.end - out.start


def _is_patch_aligned(length_sec: float) -> bool:
    bins = length_sec * 1000.0 / BIN_MS
    return abs(round(bins / PATCH) * PATCH - bins) < 1e-6


@pytest.mark.parametrize("seed", range(12))
def test_truncated_crop_is_still_patch_aligned(seed):
    """The failure mode: sampled length exceeds the available interval.

    4.297 s is chosen because clamping to it yields T=4297 bins, which is not a
    multiple of PATCH=100 — the unaligned case described in the module docstring.
    The 4.4-5.0 s distribution guarantees every seed samples past the interval
    and so exercises the truncation branch; a wider distribution would leave that
    to chance. The largest whole-patch fit below 4.297 s is 42 patches = 4.2 s,
    which is therefore the expected result for every seed.
    """
    length = _crop(full_length_sec=4.297, dist_lo=4.4, dist_hi=5.0, seed=seed)
    assert length <= 4.297 + TOL, "crop must not exceed the available interval"
    assert _is_patch_aligned(length), (
        f"truncated crop {length:.6f}s = {length * 1000:.1f} bins is not a multiple "
        f"of {PATCH} bins"
    )
    assert length == pytest.approx(4.2), "expected the largest whole-patch fit"


@pytest.mark.parametrize("seed", range(12))
def test_alignment_holds_whether_or_not_the_crop_is_truncated(seed):
    """Broad sweep: a wide distribution against an awkward interval.

    Some seeds truncate and some do not; the alignment invariant must hold for
    both. Guaranteed coverage of the truncation branch itself lives in
    `test_truncated_crop_is_still_patch_aligned`.
    """
    length = _crop(full_length_sec=4.297, dist_lo=1.0, dist_hi=5.0, seed=seed)
    assert length <= 4.297 + TOL
    assert _is_patch_aligned(length)


@pytest.mark.parametrize("seed", range(12))
def test_untruncated_crop_still_aligned(seed):
    """Crops stay aligned when no truncation occurs."""
    length = _crop(full_length_sec=30.0, dist_lo=1.0, dist_hi=5.0, seed=seed)
    assert _is_patch_aligned(length)
    assert PATCH_SEC - TOL <= length <= 5.0 + TOL


@pytest.mark.parametrize(
    ("full_length", "expected"),
    [(1.0009, 1.0), (2.5001, 2.5), (3.9999, 3.9), (4.297, 4.2), (4.9999, 4.9)],
)
def test_alignment_across_awkward_interval_lengths(full_length, expected):
    """Whatever the interval, the emitted crop is the largest whole-patch fit.

    The 4.9-5.0 s distribution exceeds every interval here, so all cases truncate.
    """
    length = _crop(full_length_sec=full_length, dist_lo=4.9, dist_hi=5.0, seed=0)
    assert length <= full_length + TOL
    assert _is_patch_aligned(length)
    assert length == pytest.approx(expected)


def test_exactly_aligned_interval_is_not_short_changed():
    """Float division must not shave a patch off an interval that divides exactly.

    `0.3 / 0.1 == 2.9999999999999996`, so flooring the raw quotient would emit
    2 patches instead of 3.
    """
    length = _crop(full_length_sec=0.3, dist_lo=4.9, dist_hi=5.0, seed=0)
    assert length == pytest.approx(0.3)


def test_interval_shorter_than_one_patch_is_not_over_sliced():
    """No aligned crop exists below one patch; emit the interval, never more."""
    full = PATCH_SEC / 2  # 50 ms, half a patch
    length = _crop(full_length_sec=full, dist_lo=1.0, dist_hi=5.0, seed=0)
    assert length == pytest.approx(full)
