"""Tests that absolute_start is correctly preserved through the data pipeline.

``NTTDataset.get()`` does not set ``sample._absolute_start`` by hand; the pipeline
relies on ``temporaldata.Data.slice()`` to propagate ``absolute_start``
automatically.  These tests verify that contract.
"""

import numpy as np
import pytest
import torch
from temporaldata import ArrayDict, Data, Interval, IrregularTimeSeries

from ntt.core.components.types import DataType
from ntt.core.data.components.transforms import (
    CreateBinnedSpikeCountsArray,
    DataToModelInput,
)


def _make_data(
    domain_start: float,
    domain_end: float,
    spike_times: np.ndarray | None = None,
    spike_units: np.ndarray | None = None,
    n_units: int = 2,
) -> Data:
    """Create a temporaldata.Data object with spikes, units, session, subject."""
    domain = Interval(domain_start, domain_end)
    units = ArrayDict(
        unit_number=np.arange(n_units),
        id=np.array([f"unit_{i}" for i in range(n_units)]),
    )
    if spike_times is None:
        spike_times = np.array([0.25, 0.75, 1.25, 1.75, 2.5])
        spike_units = np.array([0, 1, 0, 1, 0])

    spikes = IrregularTimeSeries(
        timestamps=spike_times,
        unit_index=spike_units,
        domain=domain,
    )
    session = type("Session", (), {"id": "sess_1"})()
    subject = type("Subject", (), {"id": "subj_1", "species": "macaque"})()

    data = Data(domain=domain)
    data.units = units
    data.spikes = spikes
    data.session = session
    data.subject = subject
    return data


class TestAbsoluteStartSlicing:
    """Verify that Data.slice() propagates absolute_start correctly."""

    def test_unsliced_data_has_zero_absolute_start(self):
        data = _make_data(0.0, 3.0)
        assert data.absolute_start == 0.0

    def test_single_slice_sets_absolute_start(self):
        data = _make_data(0.0, 3.0)
        sliced = data.slice(1.0, 2.0)
        assert sliced.absolute_start == pytest.approx(1.0)

    def test_domain_is_reset_after_slice(self):
        data = _make_data(0.0, 3.0)
        sliced = data.slice(1.0, 2.0)
        # Domain should be reset to [0, 1] after slice with reset_origin=True.
        # ``domain.start`` / ``domain.end`` are 1-element arrays; use ``.item()``.
        assert float(sliced.domain.start.item()) == pytest.approx(0.0)
        assert float(sliced.domain.end.item()) == pytest.approx(1.0)

    def test_double_slice_accumulates_absolute_start(self):
        data = _make_data(0.0, 10.0, spike_times=np.array([5.5]), spike_units=np.array([0]))
        first = data.slice(3.0, 8.0)
        assert first.absolute_start == pytest.approx(3.0)
        second = first.slice(1.0, 3.0)
        assert second.absolute_start == pytest.approx(4.0)

    def test_absolute_start_with_nonzero_origin(self):
        """If _absolute_start is already nonzero, slice adds to it."""
        data = _make_data(0.0, 5.0, spike_times=np.array([1.5, 3.5]), spike_units=np.array([0, 1]))
        data._absolute_start = 100.0  # simulate pre-existing offset
        sliced = data.slice(1.0, 3.0)
        assert sliced.absolute_start == pytest.approx(101.0)


class TestDataToModelInputAbsoluteStart:
    """Verify that DataToModelInput reads absolute_start from sliced data."""

    def test_absolute_start_in_binned_output(self):
        """After slicing + binning, absolute_start in output == slice start."""
        data = _make_data(0.0, 3.0)
        sliced = data.slice(1.0, 2.0)

        binner = CreateBinnedSpikeCountsArray(bin_size_ms=500)
        sliced = binner(sliced)

        transform = DataToModelInput(mode=DataType.SPIKES)
        result = transform(sliced)

        assert hasattr(result, "absolute_start")
        abs_start = float(result.absolute_start)
        assert abs_start == pytest.approx(1.0), (
            f"Expected absolute_start=1.0 (slice start), got {abs_start}"
        )

    def test_absolute_end_equals_start_plus_duration(self):
        data = _make_data(0.0, 3.0)
        sliced = data.slice(1.0, 2.0)

        binner = CreateBinnedSpikeCountsArray(bin_size_ms=500)
        sliced = binner(sliced)

        transform = DataToModelInput(mode=DataType.SPIKES)
        result = transform(sliced)

        abs_start = float(result.absolute_start)
        abs_end = float(result.absolute_end)
        duration = float(result.duration_sec)
        assert abs_end == pytest.approx(abs_start + duration)

    def test_absolute_start_with_offset_slice(self):
        """Slicing from t=2.5 should give absolute_start=2.5."""
        data = _make_data(0.0, 4.0, spike_times=np.array([2.6, 3.1]), spike_units=np.array([0, 1]))
        sliced = data.slice(2.5, 3.5)

        binner = CreateBinnedSpikeCountsArray(bin_size_ms=500)
        sliced = binner(sliced)

        transform = DataToModelInput(mode=DataType.SPIKES)
        result = transform(sliced)

        assert float(result.absolute_start) == pytest.approx(2.5)
        assert float(result.absolute_end) == pytest.approx(3.5)
