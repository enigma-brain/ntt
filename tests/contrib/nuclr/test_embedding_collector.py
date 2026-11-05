"""CPU tests for ``EmbeddingCollector`` (embedding_collector.py).

The collector accumulates per-unit embeddings across update() calls (summing
duplicates), and ``compute()`` returns the per-unit MEAN embeddings sorted by
unit id, as float64. ``state_dict``/``load_state_dict`` support the resume path
used by extract_embeddings.py.

Note (API): ``compute()`` calls ``reset()`` internally, so it is a one-shot
drain -- a second compute() with no further updates raises RuntimeError.
"""

import numpy as np
import pytest
import torch

from ntt.contrib.nuclr.embedding_collector import EmbeddingCollector

CPU = torch.device("cpu")


def test_compute_means_and_sorts_by_unit_id() -> None:
    """Embeddings accumulated across calls are mean-reduced per unit and the
    output rows are sorted by unit id, with float64 dtype and correct shape."""
    c = EmbeddingCollector(device=CPU)

    # Unit 2 seen twice (mean of the two rows), units 0 and 1 once each.
    c.update(torch.tensor([[1.0, 1.0], [2.0, 2.0]]), np.array([2, 0]))
    c.update(torch.tensor([[3.0, 3.0]]), np.array([2]))
    c.update(torch.tensor([[5.0, 5.0]]), np.array([1]))

    emb, unit_ids = c.compute()

    assert unit_ids.tolist() == [0, 1, 2]  # sorted
    assert emb.shape == (3, 2)
    assert emb.dtype == torch.float64
    # unit 0 -> [2,2]; unit 1 -> [5,5]; unit 2 -> mean([1,1],[3,3]) = [2,2]
    expected = torch.tensor([[2.0, 2.0], [5.0, 5.0], [2.0, 2.0]], dtype=torch.float64)
    assert torch.allclose(emb, expected)


def test_unit_ids_aligned_with_embedding_rows() -> None:
    """Each returned row corresponds to the same-index unit id, regardless of
    the order units first appeared in."""
    c = EmbeddingCollector(device=CPU)
    # Distinct constant embeddings keyed to the unit id so we can verify alignment.
    c.update(torch.tensor([[30.0]]), np.array([30]))
    c.update(torch.tensor([[10.0]]), np.array([10]))
    c.update(torch.tensor([[20.0]]), np.array([20]))

    emb, unit_ids = c.compute()

    assert unit_ids.tolist() == [10, 20, 30]
    for row, uid in zip(emb.tolist(), unit_ids.tolist(), strict=True):
        assert row[0] == float(uid)  # row value encodes its unit id


def test_update_accepts_tensor_unit_ids() -> None:
    """unit_ids whose elements expose ``.item()`` (e.g. a torch tensor / numpy
    scalar array) are keyed by their python value, not the wrapper object."""
    c = EmbeddingCollector(device=CPU)
    c.update(torch.tensor([[1.0], [2.0]]), torch.tensor([7, 7]))
    c.update(torch.tensor([[3.0]]), torch.tensor([7]))

    emb, unit_ids = c.compute()
    assert unit_ids.tolist() == [7]
    # mean of 1, 2, 3
    assert torch.allclose(emb, torch.tensor([[2.0]], dtype=torch.float64))


def test_state_dict_load_state_dict_roundtrip() -> None:
    """state_dict() -> load_state_dict() into a fresh collector reproduces the
    same compute() output (the resume path in extract_embeddings.py)."""
    src = EmbeddingCollector(device=CPU)
    src.update(torch.tensor([[1.0, 4.0], [2.0, 8.0]]), np.array([5, 9]))
    src.update(torch.tensor([[3.0, 6.0]]), np.array([5]))

    state = src.state_dict()
    # state captures running sums/counts/dim BEFORE compute drains them.
    assert state["dim"] == 2
    assert state["counts"] == {5: 2, 9: 1}

    fresh = EmbeddingCollector(device=CPU)
    fresh.load_state_dict(state)

    emb_resumed, ids_resumed = fresh.compute()
    emb_src, ids_src = src.compute()

    assert ids_resumed.tolist() == ids_src.tolist()
    assert torch.allclose(emb_resumed, emb_src)
    # And it is the correct mean: unit 5 -> mean([1,4],[3,6]) = [2,5]; unit 9 -> [2,8]
    assert torch.allclose(emb_resumed, torch.tensor([[2.0, 5.0], [2.0, 8.0]], dtype=torch.float64))


def test_resume_then_continue_accumulating() -> None:
    """After load_state_dict, further update() calls keep accumulating onto the
    restored sums (units already seen are summed, not overwritten)."""
    src = EmbeddingCollector(device=CPU)
    src.update(torch.tensor([[2.0]]), np.array([1]))  # unit 1 sum=2, count=1
    fresh = EmbeddingCollector(device=CPU)
    fresh.load_state_dict(src.state_dict())

    fresh.update(torch.tensor([[4.0]]), np.array([1]))  # unit 1 sum=6, count=2
    emb, unit_ids = fresh.compute()
    assert unit_ids.tolist() == [1]
    assert torch.allclose(emb, torch.tensor([[3.0]], dtype=torch.float64))  # mean


def test_reset_clears_state_and_compute_raises_when_empty() -> None:
    """reset() empties the collector; compute() on an empty collector raises."""
    c = EmbeddingCollector(device=CPU)
    c.update(torch.tensor([[1.0]]), np.array([0]))
    c.reset()

    # Internal state cleared.
    assert c._emb_sums == {}
    assert c._counts == {}
    assert c._dim is None

    with pytest.raises(RuntimeError):
        c.compute()


def test_compute_is_one_shot_drain() -> None:
    """compute() resets internally, so a second call with no new updates raises
    (documents the drain semantics callers must account for)."""
    c = EmbeddingCollector(device=CPU)
    c.update(torch.tensor([[1.0]]), np.array([0]))
    c.compute()
    with pytest.raises(RuntimeError):
        c.compute()
