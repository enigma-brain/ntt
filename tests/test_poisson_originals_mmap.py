"""Tests for the mmap-backed loading of stashed Poisson originals."""

from __future__ import annotations

import numpy as np

from ntt.jobs.eval import _load_stashed_original


def test_load_stashed_original_returns_readonly_mmap_view(tmp_path):
    arr = np.ones((100, 50), dtype=np.float32)
    path = tmp_path / "spikes.npy"
    np.save(path, arr)

    loaded = _load_stashed_original(str(path))

    # Read-only mmap view: both flags must be False
    assert loaded.flags.writeable is False, (
        "Loaded array is writeable; should be a read-only mmap view"
    )
    assert loaded.flags.owndata is False, (
        "Loaded array owns its data; should be a memory-mapped view"
    )
    # Content matches
    np.testing.assert_array_equal(loaded, arr)


def test_load_stashed_original_view_survives_dict_packing(tmp_path):
    """Putting the view in a dict and reading from it preserves the view."""
    arr = np.arange(100, dtype=np.int32)
    path = tmp_path / "x.npy"
    np.save(path, arr)

    loaded = _load_stashed_original(str(path))
    holder = {"x": loaded}
    assert holder["x"].flags.writeable is False
    assert holder["x"].flags.owndata is False
