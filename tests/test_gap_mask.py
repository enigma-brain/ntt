import numpy as np
import torch
from temporaldata import Interval

from ntt.core.components.types import DataType
from ntt.core.data.components.transforms import CreateBinnedSpikeCountsArray, DataToModelInput


class DummyUnits:
    def __init__(self, unit_number: np.ndarray):
        self.unit_number = unit_number

    def __len__(self) -> int:
        return len(self.unit_number)


def _make_dummy_data():
    domain = Interval(np.array([0.0, 1.5]), np.array([1.0, 2.0]))
    units = DummyUnits(unit_number=np.array([0]))
    spikes = type(
        "DummySpikes",
        (),
        {
            "unit_index": np.array([0, 0, 0]),
            "timestamps": np.array([0.25, 1.25, 1.75]),
        },
    )()
    session = type("DummySession", (), {"id": "session_1"})()
    subject = type("DummySubject", (), {"id": "subject_1", "species": "macaque"})()
    return type(
        "DummyData",
        (),
        {
            "domain": domain,
            "absolute_start": 0.0,
            "units": units,
            "spikes": spikes,
            "session": session,
            "subject": subject,
        },
    )()


def test_binned_counts_zero_fill_and_gap_mask():
    data = _make_dummy_data()
    transform = CreateBinnedSpikeCountsArray(bin_size_ms=500)
    data = transform(data)

    counts = data.binned_spike_counts.counts
    assert counts.shape == (4, 1)
    assert counts[:, 0].tolist() == [1, 0, 0, 1]

    gap_mask = data.binned_spike_counts.gap_mask
    assert gap_mask.tolist() == [True, True, False, True]


def test_data_to_model_input_uses_gap_mask():
    data = _make_dummy_data()
    transform = CreateBinnedSpikeCountsArray(bin_size_ms=500)
    data = transform(data)

    model_input = DataToModelInput(mode=DataType.SPIKES)(data)
    pad_and_gap_mask = model_input.pad_and_gap_mask
    assert hasattr(pad_and_gap_mask, "obj")
    mask = pad_and_gap_mask.obj
    assert isinstance(mask, torch.Tensor)
    assert mask.shape == (4, 1)
    assert mask[:, 0].tolist() == [True, True, False, True]
