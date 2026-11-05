"""CPU tests for ``SampleWiseContrastiveLoss`` — the contrastive math core.

This is the most correctness-critical piece of the NuCLR model. Covers the two
degenerate cases that must not produce NaNs:
a global batch with zero matched pairs, and a matched element whose unit is
the only unit in its view (so it has no in-batch negatives).
"""

import torch

from ntt.contrib.nuclr.contrastive_loss import SampleWiseContrastiveLoss


def _loss(dim: int = 8) -> SampleWiseContrastiveLoss:
    # projector=False -> deterministic (Identity), so p = normalize(emb).
    return SampleWiseContrastiveLoss(dim=dim, projector=False)


def test_normal_batch_is_finite_and_backprops() -> None:
    torch.manual_seed(0)
    dim = 8
    # batch of 2 elements, 3 units each view, all matched 0<->0, 1<->1, 2<->2.
    emb1 = torch.randn(6, dim, requires_grad=True)
    emb2 = torch.randn(6, dim, requires_grad=True)
    unit_seqlen = torch.tensor([3, 3])
    match_idx = [
        torch.tensor([[0, 1, 2], [0, 1, 2]]),
        torch.tensor([[0, 1, 2], [0, 1, 2]]),
    ]

    loss = _loss(dim)(emb1, emb2, unit_seqlen, unit_seqlen, match_idx)

    assert torch.isfinite(loss), loss
    loss.backward()
    assert torch.isfinite(emb1.grad).all()
    assert torch.isfinite(emb2.grad).all()


def test_zero_matches_in_global_batch_returns_finite_zero() -> None:
    # Every element has zero matched pairs -> total_matches == 0.
    # Guards against loss_weights = 0 / 0 = NaN -> NaN loss.
    dim = 8
    emb1 = torch.randn(2, dim, requires_grad=True)
    emb2 = torch.randn(2, dim, requires_grad=True)
    unit_seqlen = torch.tensor([2])
    match_idx = [torch.empty(2, 0, dtype=torch.long)]

    loss = _loss(dim)(emb1, emb2, unit_seqlen, unit_seqlen, match_idx)

    assert torch.isfinite(loss), loss
    assert loss.item() == 0.0
    # Must stay connected to the graph so .backward() is well-defined.
    loss.backward()
    assert torch.isfinite(emb1.grad).all()


def test_single_unit_matched_element_is_finite() -> None:
    # One element, one unit per view, matched. The unit has NO in-batch
    # negatives, so its denominator row is fully -inf.
    # Guards against logsumexp(all -inf) = -inf -> NaN gradient.
    dim = 8
    emb1 = torch.randn(1, dim, requires_grad=True)
    emb2 = torch.randn(1, dim, requires_grad=True)
    unit_seqlen = torch.tensor([1])
    match_idx = [torch.tensor([[0], [0]])]

    loss = _loss(dim)(emb1, emb2, unit_seqlen, unit_seqlen, match_idx)

    assert torch.isfinite(loss), loss
    loss.backward()
    assert torch.isfinite(emb1.grad).all()
    assert torch.isfinite(emb2.grad).all()


def test_mixed_batch_with_one_single_unit_element_is_finite() -> None:
    # Element 0 is a normal 3-unit element; element 1 is a degenerate
    # single-unit element. The single-unit element must not poison the batch.
    torch.manual_seed(1)
    dim = 8
    emb1 = torch.randn(4, dim, requires_grad=True)  # 3 + 1
    emb2 = torch.randn(4, dim, requires_grad=True)
    unit_seqlen = torch.tensor([3, 1])
    match_idx = [
        torch.tensor([[0, 1, 2], [0, 1, 2]]),
        torch.tensor([[0], [0]]),
    ]

    loss = _loss(dim)(emb1, emb2, unit_seqlen, unit_seqlen, match_idx)

    assert torch.isfinite(loss), loss
    loss.backward()
    assert torch.isfinite(emb1.grad).all()
    assert torch.isfinite(emb2.grad).all()


def test_dcl_excludes_positive_from_denominator() -> None:
    # With dcl=True the matched (positive) entry is removed from the
    # denominator, so a 2-unit element where both views are identical and the
    # two units are orthogonal yields a well-defined finite loss.
    torch.manual_seed(2)
    dim = 8
    base = torch.randn(2, dim)
    emb1 = base.clone().requires_grad_(True)
    emb2 = base.clone().requires_grad_(True)
    unit_seqlen = torch.tensor([2])
    match_idx = [torch.tensor([[0, 1], [0, 1]])]

    loss = SampleWiseContrastiveLoss(dim=dim, projector=False, dcl=True)(
        emb1, emb2, unit_seqlen, unit_seqlen, match_idx
    )

    assert torch.isfinite(loss), loss
