import numpy as np
import torch

from ntt.core.components.types import BinnedSpikesInput, DataType, RatesInput
from ntt.evaluation.core.types import EvalData


def test_from_binned_input_routes_rates_input_to_binned_counts() -> None:
    rate_matrix = torch.tensor(
        [
            [
                [0.0, 1.0, 2.0],
                [3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0],
            ]
        ],
        dtype=torch.float32,
    )
    batch = RatesInput(
        rate_matrix=rate_matrix,
        pad_and_gap_mask=torch.ones_like(rate_matrix, dtype=torch.int32),
        sampling_rate=20.0,
        unit_ids=torch.tensor([[10, 11, 12]], dtype=torch.int32),
        unit_ids_pad_mask=torch.ones((1, 3), dtype=torch.int32),
        duration_sec=torch.tensor([0.15], dtype=torch.float32),
        session_id="session_a",
        subject_id="subject_a",
        species="macaque",
        absolute_start=torch.tensor([0.0], dtype=torch.float64),
        absolute_end=torch.tensor([0.15], dtype=torch.float64),
    )

    data = EvalData.from_binned_input(batch, batch_index=0)

    assert data.data_type == DataType.RATES
    assert data.binned_counts is not None
    assert data.rates_hz is None
    np.testing.assert_array_equal(data.binned_counts, rate_matrix[0].numpy().T)


def test_from_binned_batch_collated_routes_rates_input_to_binned_counts() -> None:
    rate_matrix = torch.tensor(
        [
            [[1.0, 10.0], [2.0, 20.0]],
            [[3.0, 30.0], [4.0, 40.0]],
            [[5.0, 50.0], [6.0, 60.0]],
        ],
        dtype=torch.float32,
    )
    batch = RatesInput(
        rate_matrix=rate_matrix,
        pad_and_gap_mask=torch.ones_like(rate_matrix, dtype=torch.int32),
        sampling_rate=10.0,
        unit_ids=torch.tensor([[0, 1], [0, 1], [0, 1]], dtype=torch.int32),
        unit_ids_pad_mask=torch.ones((3, 2), dtype=torch.int32),
        duration_sec=torch.tensor([0.2, 0.2, 0.2], dtype=torch.float32),
        session_id=("session_a", "session_a", "session_b"),
        subject_id=("subject", "subject", "subject"),
        species=("macaque", "macaque", "macaque"),
        absolute_start=torch.tensor([0.0, 0.2, 1.0], dtype=torch.float64),
        absolute_end=torch.tensor([0.2, 0.4, 1.2], dtype=torch.float64),
    )

    session_data = EvalData.from_binned_batch_collated(batch)

    assert session_data["session_a"].binned_counts is not None
    assert session_data["session_a"].rates_hz is None
    np.testing.assert_array_equal(
        session_data["session_a"].binned_counts,
        np.array([[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        session_data["session_b"].binned_counts,
        np.array([[5.0, 6.0], [50.0, 60.0]], dtype=np.float32),
    )


def test_from_binned_batch_collated_keeps_spikes_as_binned_spikes() -> None:
    spike_matrix = torch.tensor(
        [
            [[0, 1], [1, 0]],
            [[1, 1], [0, 0]],
        ],
        dtype=torch.int32,
    )
    batch = BinnedSpikesInput(
        spike_matrix=spike_matrix,
        pad_and_gap_mask=torch.ones_like(spike_matrix, dtype=torch.int32),
        sampling_rate=1000.0,
        unit_ids=torch.tensor([[0, 1], [0, 1]], dtype=torch.int32),
        unit_ids_pad_mask=torch.ones((2, 2), dtype=torch.int32),
        duration_sec=torch.tensor([0.002, 0.002], dtype=torch.float32),
        session_id=("session_a", "session_a"),
        subject_id=("subject", "subject"),
        species=("macaque", "macaque"),
        absolute_start=torch.tensor([0.0, 0.002], dtype=torch.float64),
        absolute_end=torch.tensor([0.002, 0.004], dtype=torch.float64),
    )

    session_data = EvalData.from_binned_batch_collated(batch)

    assert session_data["session_a"].binned_spikes is not None
    assert session_data["session_a"].binned_counts is None
