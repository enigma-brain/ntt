"""Tests for ``NTTDataset.get_recording_data`` leak-check behavior.

Eval runs with the data-leakage check enabled (``disable_data_leakage_check=false``)
and the light-eval val set is split into a split-respecting ``xtime`` block and an OOD
block carrying ``ignore_splits=true``. ``NTTDataset.get()`` already skips the leak
check for ``ignore_splits`` recordings, but the whole-session path
``get_recording_data()`` (used by ``n_neurons_per_session`` / ``extract_spike_data``)
must also skip it: the base ``torch_brain`` implementation runs the check
unconditionally, but OOD sessions are evaluated over their full recording, so most
events fall outside ``valid_domain`` and an unguarded check would raise a false-positive
"Data leakage detected" ``AssertionError``.

These tests pin that ``get_recording_data`` must honor ``ignore_splits`` while
still enforcing the check for split-respecting recordings.
"""

import numpy as np
import pytest
from temporaldata import ArrayDict, Data, Interval, IrregularTimeSeries

from ntt.core.data.components.dataset import NTTDataset

SESSION_ID = "toy_brainset/toy_session"


def _make_session(valid_mask_interval: Interval) -> Data:
    """A 10 s toy recording with one spike per second and a ``valid`` split mask.

    Spikes sit at 0.5, 1.5, ..., 9.5 s. ``valid_domain`` is the held-out tail
    [8, 10) s; ``valid_mask_interval`` controls which spikes are marked valid
    (normally identical to ``valid_domain``; deliberately narrower to simulate a
    genuine leak).
    """
    timestamps = np.arange(0.5, 10.0, 1.0)
    spikes = IrregularTimeSeries(
        timestamps=timestamps,
        unit_index=(np.arange(len(timestamps)) % 2),
        domain=Interval(0.0, 10.0),
    )
    spikes.add_split_mask("valid", valid_mask_interval)
    units = ArrayDict(
        id=np.array(["u0", "u1"]),
        unit_number=np.array([0, 1]),
    )
    return Data(
        spikes=spikes,
        units=units,
        valid_domain=Interval(8.0, 10.0),
        domain=Interval(0.0, 10.0),
    )


def _make_dataset(*, ignore_splits: bool, valid_mask_interval: Interval) -> NTTDataset:
    """Build a minimal ``NTTDataset`` over one in-memory toy session.

    Bypasses ``__init__`` (which expects on-disk brainset HDF5s) and wires only the
    attributes ``get_recording_data`` touches: an in-memory data object, the
    recording config carrying ``ignore_splits``, the active split, and the
    leak-check flag (ON, as in eval).
    """
    ds = NTTDataset.__new__(NTTDataset)
    ds.keep_files_open = True
    ds._data_objects = {SESSION_ID: _make_session(valid_mask_interval)}
    ds.recording_dict = {SESSION_ID: {"config": {"ignore_splits": ignore_splits}}}
    ds.split = "valid"
    ds._check_for_data_leakage_flag = True
    ds.unit_id_prefix_fn = lambda data: "toy_brainset/"
    return ds


def test_ignore_splits_recording_skips_leak_check():
    """OOD path: ignore_splits=true bypasses the check and keeps the full recording."""
    ds = _make_dataset(ignore_splits=True, valid_mask_interval=Interval(8.0, 10.0))
    data = ds.get_recording_data(SESSION_ID)  # must NOT raise
    # Full recording retained (all 10 spikes), not trimmed to valid_domain.
    assert len(data.spikes.timestamps) == 10


def test_split_respecting_recording_slices_to_valid_domain():
    """xtime path: ignore_splits=false slices to valid_domain and passes the check."""
    ds = _make_dataset(ignore_splits=False, valid_mask_interval=Interval(8.0, 10.0))
    data = ds.get_recording_data(SESSION_ID)  # must NOT raise
    # Only the held-out tail [8, 10) survives: spikes at 8.5 and 9.5 s.
    assert len(data.spikes.timestamps) == 2
    assert data.spikes.valid_mask.all()


def test_split_respecting_recording_still_enforces_leak_check():
    """The fix must not disable the check globally: a genuine leak still raises."""
    # valid_domain is [8, 10) but only [8, 9) is marked valid, so the spike at
    # 9.5 s survives the valid_domain slice with valid_mask=False -> leakage.
    ds = _make_dataset(ignore_splits=False, valid_mask_interval=Interval(8.0, 9.0))
    with pytest.raises(AssertionError, match="leakage"):
        ds.get_recording_data(SESSION_ID)
