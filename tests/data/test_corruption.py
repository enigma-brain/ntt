"""Focused tests for the corruption transforms in
``src/ntt/core/data/components/corruption.py``.

Covers the drop-and-replace semantics for ``apply_drop_neurons``: each
dropped unit's spike train is overwritten with a copy of a randomly
chosen surviving unit's full spike train (sampling with replacement).
"""

from __future__ import annotations

import numpy as np
import pytest

from ntt.core.data.components.corruption import (
    DropNeuronsSpec,
    apply_drop_neurons,
)


def test_apply_drop_neurons_replaces_with_donor_unit():
    """Each dropped unit gets a bit-identical copy of some surviving unit.

    Builds a [T, U] counts array where each unit ``u`` has a unique
    signature (it fires only at bin ``u``) so any donor copy is
    immediately identifiable. After applying ``apply_drop_neurons``
    with ``fraction=0.5``, every output column must equal the input
    column of *some* surviving unit (its own column for survivors,
    a donor's column for dropped units).
    """
    T, U = 6, 6
    counts = np.zeros((T, U), dtype=np.int64)
    for u in range(U):
        # unit u fires exactly once, only at time bin u — unique signature.
        counts[u, u] = 1

    spec = DropNeuronsSpec(fraction=0.5)
    rng = np.random.default_rng(0)
    out = apply_drop_neurons(counts, spec, rng)

    assert out.shape == counts.shape

    # Every output column must match some input column (the donor's, or
    # the unit's own for survivors). With unique per-unit signatures
    # this also confirms the donor was a *surviving* unit.
    input_cols = [counts[:, u] for u in range(U)]
    for u in range(U):
        out_col = out[:, u]
        matches = [
            donor_u for donor_u, col in enumerate(input_cols) if np.array_equal(out_col, col)
        ]
        assert matches, f"output unit {u} does not match any input unit's spike train"

    # fraction=0.5 with U=6 → round(3) units dropped. Identify dropped
    # vs surviving by where the output column equals its own input
    # column. Survivors must be unchanged; dropped units' donors must
    # be drawn from the surviving set.
    unchanged = [u for u in range(U) if np.array_equal(out[:, u], counts[:, u])]
    changed = [u for u in range(U) if not np.array_equal(out[:, u], counts[:, u])]
    # With unique-signature donors the probability of a dropped unit
    # randomly receiving its own pattern is zero (donors come from the
    # surviving set only), so |changed| equals the actual drop count.
    assert len(changed) == 3, f"expected 3 dropped units (fraction=0.5 of U=6); got {len(changed)}"
    assert len(unchanged) == 3

    # Each dropped unit's spike train must equal *some* surviving unit's
    # spike train.
    for u_dropped in changed:
        donor_matches = [
            u_surv for u_surv in unchanged if np.array_equal(out[:, u_dropped], counts[:, u_surv])
        ]
        assert donor_matches, (
            f"dropped unit {u_dropped} did not receive a copy of any surviving unit's spike train"
        )


def test_apply_drop_neurons_no_op_when_fraction_too_small():
    """``round(fraction * n_units) < 1`` must return a fresh copy unchanged."""
    counts = np.arange(20, dtype=np.int64).reshape(10, 2)
    spec = DropNeuronsSpec(fraction=0.1)  # round(0.2) = 0 → no-op
    rng = np.random.default_rng(0)
    out = apply_drop_neurons(counts, spec, rng)
    assert out is not counts  # fresh array
    np.testing.assert_array_equal(out, counts)


def test_apply_drop_neurons_deterministic_for_fixed_seed():
    """Same seed → same dropped-unit indices and same donor assignment."""
    T, U = 20, 4
    rng_data = np.random.default_rng(123)
    counts = (rng_data.random((T, U)) > 0.5).astype(np.int64)

    spec = DropNeuronsSpec(fraction=0.5)
    out_a = apply_drop_neurons(counts, spec, np.random.default_rng(7))
    out_b = apply_drop_neurons(counts, spec, np.random.default_rng(7))
    np.testing.assert_array_equal(out_a, out_b)


def test_apply_drop_neurons_raises_when_no_survivors():
    """``fraction=1.0`` leaves no donor pool → ValueError."""
    T, U = 10, 4
    counts = np.zeros((T, U), dtype=np.int64)
    spec = DropNeuronsSpec(fraction=1.0)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="at least one surviving unit"):
        apply_drop_neurons(counts, spec, rng)
