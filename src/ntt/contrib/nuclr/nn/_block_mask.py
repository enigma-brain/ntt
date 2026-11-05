"""Block-diagonal mask construction for flex_attention.

Replaces ``xformers.ops.fmha.BlockDiagonalMask.from_seqlens`` with an upstream
PyTorch primitive: builds a ``BlockMask`` whose ``mask_mod`` returns True iff
the query and key indices fall within the same segment.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask, create_block_mask


def build_block_diagonal_mask(
    q_seqlen: list[int],
    kv_seqlen: list[int] | None = None,
    *,
    device: torch.device | str,
) -> BlockMask:
    """Construct a block-diagonal mask for ``flex_attention``.

    Args:
        q_seqlen: Number of query positions per segment.
        kv_seqlen: Number of key/value positions per segment. ``None`` ⇒
            self-attention (``kv_seqlen = q_seqlen``).
        device: Device on which to build the cumulative-length tensors.

    Returns:
        ``BlockMask`` enforcing per-segment isolation.

    Raises:
        ValueError: If segment lists differ in length or contain negative
            entries.

    Note:
        Zero-length segments are allowed. A unit that is silent in a window
        legitimately yields a ``kv_seqlen`` entry of 0 (see
        ``ntt_to_nuclr_format``); its query rows attend to an empty key-set and
        ``flex_attention`` returns zeros for those fully-masked rows (verified
        finite in forward and backward), so the unit's embedding is formed from
        the remaining self/spatial-attention paths instead of crashing.
    """
    if kv_seqlen is None:
        kv_seqlen = q_seqlen
    if len(q_seqlen) != len(kv_seqlen):
        raise ValueError(
            f"q_seqlen and kv_seqlen must have matching length; "
            f"got {len(q_seqlen)} and {len(kv_seqlen)}."
        )
    if any(n < 0 for n in q_seqlen) or any(n < 0 for n in kv_seqlen):
        raise ValueError(
            "Segment lengths must be non-negative; "
            f"got q_seqlen={q_seqlen}, kv_seqlen={kv_seqlen}."
        )

    q_total = sum(q_seqlen)
    kv_total = sum(kv_seqlen)

    # seg_q[i] = segment id (0..len(q_seqlen)-1) that contains position i.
    # We use repeat_interleave + indexing in mask_mod rather than the more
    # natural searchsorted-on-cumsum form because Inductor (the backend
    # behind torch.compile(flex_attention)) cannot currently lower
    # `torch.searchsorted` inside a mask_mod subgraph.
    n_seg = len(q_seqlen)
    seg_ids = torch.arange(n_seg, dtype=torch.int64, device=device)
    seg_q = torch.repeat_interleave(
        seg_ids,
        torch.tensor(q_seqlen, dtype=torch.int64, device=device),
    )
    seg_kv = torch.repeat_interleave(
        seg_ids,
        torch.tensor(kv_seqlen, dtype=torch.int64, device=device),
    )

    def mask_mod(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        return seg_q[q_idx] == seg_kv[kv_idx]

    return create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=q_total,
        KV_LEN=kv_total,
        device=device,
    )
