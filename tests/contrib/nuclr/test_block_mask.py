"""Behavioral tests for the block-diagonal mask helper.

Verifies the mask via flex_attention output on crafted inputs rather than
inspecting BlockMask internals (`to_dense()` returns block-level granularity
that doesn't reflect element-level patterns at the default BLOCK_SIZE).
"""

from __future__ import annotations

import pytest
import torch
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from ntt.contrib.nuclr.nn._block_mask import build_block_diagonal_mask

# Only the flex_attention-kernel tests need CUDA; the pure mask-builder
# validation tests run on CPU so they execute in CI.
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="flex_attention kernels require CUDA",
)


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda")


def _attend_with_values(
    q_total: int,
    kv_total: int,
    values: torch.Tensor,
    mask,
    device: torch.device,
) -> torch.Tensor:
    """Run flex_attention with q=k=ones and the given values, returning
    the per-position output (collapsed to a 1-D tensor by averaging dim D).

    With q=k=ones, all unmasked positions get equal attention weight, so
    each output row is the mean of the values at positions allowed by the
    mask. This makes the test independent of any softmax-temperature
    subtleties.
    """
    dim = 4
    q = torch.ones(1, 1, q_total, dim, device=device, dtype=torch.float32)
    k = torch.ones(1, 1, kv_total, dim, device=device, dtype=torch.float32)
    v = values.view(1, 1, kv_total, 1).expand(1, 1, kv_total, dim).contiguous()
    out = flex_attention(q, k, v, block_mask=mask)
    return out.mean(dim=-1).squeeze(0).squeeze(0)


@requires_cuda
def test_block_diagonal_self_attention(device: torch.device) -> None:
    """Self-attention with q_seqlen=[2, 3]:
    positions 0,1 form one block (size 2); positions 2,3,4 the other (size 3).
    With distinguishing values [10, 10, 100, 100, 100], block-0 outputs ≈ 10
    and block-1 outputs ≈ 100.
    """
    mask = build_block_diagonal_mask(q_seqlen=[2, 3], device=device)
    values = torch.tensor([10.0, 10.0, 100.0, 100.0, 100.0], device=device)
    out = _attend_with_values(q_total=5, kv_total=5, values=values, mask=mask, device=device)

    torch.testing.assert_close(
        out[:2], torch.full((2,), 10.0, device=device), atol=1e-4, rtol=1e-4
    )
    torch.testing.assert_close(
        out[2:], torch.full((3,), 100.0, device=device), atol=1e-4, rtol=1e-4
    )


@requires_cuda
def test_block_diagonal_cross_attention(device: torch.device) -> None:
    """Cross-attention with q_seqlen=[2, 3], kv_seqlen=[5, 4]:
    Q-block 0 (positions 0,1) attends to KV positions 0..4;
    Q-block 1 (positions 2,3,4) attends to KV positions 5..8.
    """
    mask = build_block_diagonal_mask(q_seqlen=[2, 3], kv_seqlen=[5, 4], device=device)
    values = torch.tensor(
        [
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,  # KV block 0
            50.0,
            50.0,
            50.0,
            50.0,
        ],  # KV block 1
        device=device,
    )
    out = _attend_with_values(q_total=5, kv_total=9, values=values, mask=mask, device=device)

    torch.testing.assert_close(out[:2], torch.full((2,), 1.0, device=device), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(
        out[2:], torch.full((3,), 50.0, device=device), atol=1e-4, rtol=1e-4
    )


@requires_cuda
def test_block_diagonal_single_block(device: torch.device) -> None:
    """Single-block mask should let every position attend to every other."""
    mask = build_block_diagonal_mask(q_seqlen=[4], device=device)
    values = torch.tensor([1.0, 2.0, 3.0, 4.0], device=device)
    out = _attend_with_values(q_total=4, kv_total=4, values=values, mask=mask, device=device)
    # Mean of [1,2,3,4] is 2.5; every output position should be 2.5.
    torch.testing.assert_close(out, torch.full((4,), 2.5, device=device), atol=1e-4, rtol=1e-4)


def test_block_diagonal_mismatched_lengths_raises() -> None:
    with pytest.raises(ValueError, match="matching length"):
        build_block_diagonal_mask(
            q_seqlen=[2, 3],
            kv_seqlen=[5, 4, 1],
            device="cpu",
        )


def test_block_diagonal_negative_segment_raises() -> None:
    """Negative segment lengths are invalid and must raise."""
    with pytest.raises(ValueError, match="negative"):
        build_block_diagonal_mask(q_seqlen=[2, -1, 3], device="cpu")
    with pytest.raises(ValueError, match="negative"):
        build_block_diagonal_mask(
            q_seqlen=[2, 3],
            kv_seqlen=[5, -1],
            device="cpu",
        )


def test_block_diagonal_zero_length_segment_tolerated() -> None:
    """A zero-length segment is representable, not an error.

    A unit that is
    silent in a window legitimately yields ``spike_seqlen == 0`` (see
    ``ntt_to_nuclr_format``). The mask builder must accept it — the matching
    query rows simply attend to an empty key-set, and ``flex_attention``
    returns zeros for such fully-masked rows.
    """
    mask = build_block_diagonal_mask(
        q_seqlen=[2, 2],
        kv_seqlen=[3, 0],
        device="cpu",
    )
    assert isinstance(mask, BlockMask)
