"""Tests for BinnedSpikesInput absolute_start/end fields and collation.

Seems mostly unnecessary.
"""

import torch

from ntt.core.components.types import BinnedSpikesInput
from ntt.core.data.datamodule import collate

_DEFAULT_ABSOLUTE_START = torch.tensor(100.5, dtype=torch.float64)
_DEFAULT_ABSOLUTE_END = torch.tensor(101.5, dtype=torch.float64)


def _make_binned_input(
    absolute_start=_DEFAULT_ABSOLUTE_START,
    absolute_end=_DEFAULT_ABSOLUTE_END,
    session_id="sess_0",
):
    """Create a minimal single-sample BinnedSpikesInput for testing."""
    n_time, n_units = 10, 3
    spike_matrix = torch.zeros((n_time, n_units), dtype=torch.int32)
    return BinnedSpikesInput(
        spike_matrix=spike_matrix,
        pad_and_gap_mask=torch.ones_like(spike_matrix, dtype=torch.int32),
        sampling_rate=1000.0,
        unit_ids=torch.arange(n_units, dtype=torch.int32),
        unit_ids_pad_mask=torch.ones(n_units, dtype=torch.int32),
        duration_sec=torch.tensor(0.01),
        session_id=session_id,
        subject_id="subj_0",
        species="mouse",
        absolute_start=absolute_start,
        absolute_end=absolute_end,
    )


def test_creation_with_absolute_times():
    """BinnedSpikesInput can be created with absolute_start and absolute_end."""
    t0 = torch.tensor(100.5, dtype=torch.float64)
    t1 = torch.tensor(101.5, dtype=torch.float64)
    inp = _make_binned_input(absolute_start=t0, absolute_end=t1)
    assert inp.absolute_start is not None
    assert inp.absolute_end is not None
    assert inp.absolute_start.dtype == torch.float64
    assert float(inp.absolute_start) == 100.5
    assert float(inp.absolute_end) == 101.5


def test_creation_requires_absolute_times():
    """BinnedSpikesInput requires absolute_start/end constructor args."""
    n_time, n_units = 5, 2
    try:
        BinnedSpikesInput(
            spike_matrix=torch.zeros((n_time, n_units), dtype=torch.int32),
            pad_and_gap_mask=torch.ones((n_time, n_units), dtype=torch.int32),
            sampling_rate=1000.0,
            unit_ids=torch.arange(n_units, dtype=torch.int32),
            unit_ids_pad_mask=torch.ones(n_units, dtype=torch.int32),
            duration_sec=torch.tensor(0.005),
            session_id="s",
            subject_id="sub",
            species="rat",
        )
        raise AssertionError("Should have raised TypeError for missing absolute_start/end")
    except TypeError as e:
        assert "absolute_start" in str(e)
        assert "absolute_end" in str(e)


def test_collation_with_absolute_times():
    """Collating two samples with absolute_start/end stacks them correctly."""
    s0 = _make_binned_input(
        absolute_start=torch.tensor(10.0, dtype=torch.float64),
        absolute_end=torch.tensor(11.0, dtype=torch.float64),
    )
    s1 = _make_binned_input(
        absolute_start=torch.tensor(20.0, dtype=torch.float64),
        absolute_end=torch.tensor(21.0, dtype=torch.float64),
    )
    batch = collate([s0, s1])

    assert batch.absolute_start is not None
    assert batch.absolute_end is not None
    assert batch.absolute_start.shape == (2,)
    assert batch.absolute_end.shape == (2,)
    assert float(batch.absolute_start[0]) == 10.0
    assert float(batch.absolute_start[1]) == 20.0
    assert float(batch.absolute_end[0]) == 11.0
    assert float(batch.absolute_end[1]) == 21.0


def test_collation_none_raises():
    """Collating with None absolute_start/end should raise."""
    s0 = _make_binned_input(absolute_start=torch.tensor(10.0, dtype=torch.float64))
    s1 = _make_binned_input(
        absolute_start=None,  # type: ignore[arg-type]
        absolute_end=None,  # type: ignore[arg-type]
    )
    try:
        collate([s0, s1])
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "absolute_start" in str(e)


def test_creation_with_required_kwargs_and_behavioral_default():
    """BinnedSpikesInput supports behavioral_data default with required absolutes."""
    n_time, n_units = 5, 2
    inp = BinnedSpikesInput(
        spike_matrix=torch.zeros((n_time, n_units), dtype=torch.int32),
        pad_and_gap_mask=torch.ones((n_time, n_units), dtype=torch.int32),
        sampling_rate=1000.0,
        unit_ids=torch.arange(n_units, dtype=torch.int32),
        unit_ids_pad_mask=torch.ones(n_units, dtype=torch.int32),
        duration_sec=torch.tensor(0.005),
        session_id="s",
        subject_id="sub",
        species="rat",
        absolute_start=torch.tensor(0.0, dtype=torch.float64),
        absolute_end=torch.tensor(0.005, dtype=torch.float64),
    )
    assert float(inp.absolute_start) == 0.0
    assert float(inp.absolute_end) == 0.005
    assert inp.behavioral_data is None
