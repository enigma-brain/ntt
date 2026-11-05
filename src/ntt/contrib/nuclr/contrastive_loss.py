"""Sample-wise contrastive loss for neuron identity learning.

Ported from nuclr ``src/losses/sw_contrastive.py`` with correct defaults
(``tau=0.2``, ``full_denom=True``, ``dcl=True``) and ``AllReduce`` for
DDP-aware loss weighting across GPUs.

Adapted from the NuCLR reference implementation, used under the Apache-2.0
license: https://github.com/nerdslab/nuclr (Arora et al., 2025, NeurIPS;
https://arxiv.org/abs/2512.01199).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .distributed import AllReduce


class SampleWiseContrastiveLoss(nn.Module):
    """Per-sample contrastive loss with DCL and AllReduce for DDP.

    Computes contrastive loss between identity embeddings from two views.
    Uses ``match_idx`` to identify positive pairs (same neuron in both views).
    Loss weights are synchronized across GPUs via ``AllReduce``.

    Args:
        dim: Embedding dimension (for projector input/output).
        tau: Temperature for similarity scaling.
        dcl: Use decoupled contrastive learning variant (mask positives from denominator).
        projector: Use learnable projector head (recommended).
        full_denom: Use full denominator (cross-view + within-view negatives).
    """

    def __init__(
        self,
        dim: int = 256,
        tau: float = 0.2,
        dcl: bool = True,
        projector: bool = True,
        full_denom: bool = True,
    ):
        super().__init__()
        self.tau = tau
        self.dcl = dcl
        self.full_denom = full_denom

        if projector:
            self.projector = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, dim),
            )
        else:
            self.projector = nn.Identity()

    def forward(
        self,
        emb1: Tensor,
        emb2: Tensor,
        unit_seqlen_v1: Tensor,
        unit_seqlen_v2: Tensor,
        match_idx: list[Tensor],
    ) -> Tensor:
        """Compute contrastive loss between two views.

        Args:
            emb1: Identity embeddings from view 1 ``(total_units_v1, dim)``.
            emb2: Identity embeddings from view 2 ``(total_units_v2, dim)``.
            unit_seqlen_v1: Units per batch element in view 1 ``(batch,)``.
            unit_seqlen_v2: Units per batch element in view 2 ``(batch,)``.
            match_idx: List of ``(2, M_b)`` tensors per batch element,
                where row 0 indexes into view1's units and row 1 indexes
                into view2's units for matched neurons.

        Returns:
            Scalar contrastive loss.
        """
        p1 = F.normalize(self.projector(emb1), dim=-1)
        p2 = F.normalize(self.projector(emb2), dim=-1)

        B = len(match_idx)
        loss_numers = emb1.new_zeros(B)
        loss_denoms = emb1.new_zeros(B)
        offset1, offset2 = 0, 0

        # Materialize seqlens once (single host sync) instead of calling
        # ``.item()`` inside the loop, which forces a device->host sync per
        # iteration.
        n1s = unit_seqlen_v1.tolist()
        n2s = unit_seqlen_v2.tolist()

        for b in range(B):
            n1, n2 = n1s[b], n2s[b]

            if match_idx[b].size(1) > 0:
                loss_numers[b], loss_denoms[b] = self._std_contrastive_loss(
                    x1=p1[offset1 : offset1 + n1],
                    x2=p2[offset2 : offset2 + n2],
                    match_idx=match_idx[b],
                )

            offset1 += n1
            offset2 += n2

        # Loss weight is num_pos_pairs / total_pos_pairs (summed across GPUs).
        # Cast to float before AllReduce so the division is unambiguously
        # floating-point regardless of backend dtype handling.
        num_matches = torch.tensor(
            [m.size(1) for m in match_idx],
            dtype=torch.long,
            device=emb1.device,
        )
        total_matches: Tensor = AllReduce.apply(num_matches.sum().float())  # type: ignore

        if total_matches == 0:
            # No positive pairs anywhere in the (global) batch this step: there
            # is no contrastive signal, and loss_numers/denoms were never
            # populated (so they are disconnected from the autograd graph).
            # Return a graph-connected zero so ``.backward()`` stays well-defined
            # instead of computing 0 / 0 -> NaN.
            return p1.sum() * 0.0 + p2.sum() * 0.0

        # NOTE (DDP): ``total_matches`` is the *mean* per-rank match count (the
        # AllReduce above divides by world_size before summing), so these
        # per-rank weights are inflated by a factor of world_size. That factor
        # is exactly cancelled by the AllReduce inside ``_weighted_avg`` below
        # (which divides the summed weighted loss by world_size again), yielding
        # the correct global match-weighted average. Removing either AllReduce
        # would silently break the DDP weighting.
        loss_weights = num_matches / total_matches

        # Compute weighted-averaged loss (synchronized across GPUs)
        loss_numer = self._weighted_avg(loss_numers, loss_weights)
        loss_denom = self._weighted_avg(loss_denoms, loss_weights)
        loss = loss_numer + loss_denom

        return loss

    def _weighted_avg(self, x: Tensor, weights: Tensor) -> Tensor:
        ans = (x * weights).sum()
        ans = AllReduce.apply(ans)
        return ans  # type: ignore

    def _std_contrastive_loss(
        self,
        x1: Tensor,
        x2: Tensor,
        match_idx: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Standard contrastive loss computation.

        Args:
            x1: Normalized embeddings from view 1 ``(n1, dim)``.
            x2: Normalized embeddings from view 2 ``(n2, dim)``.
            match_idx: ``(2, M)`` indices of matched neurons.

        Returns:
            Tuple of (numerator_loss, denominator_loss).
        """
        # Numerator: average similarity of positive pairs.
        # Shifted by 1/tau so the reported value is non-negative
        # (cosine sim ≤ 1 ⇒ sim/tau ≤ 1/tau). The constant offset does not
        # affect gradients; kept to match the upstream NuCLR reporting.
        sim12 = (x1 @ x2.T) / self.tau
        numer = sim12[match_idx[0], match_idx[1]].mean()
        numer = (1 / self.tau) - numer

        # Work on a copy so the in-place DCL mask below cannot alias the matmul
        # output that is reused (directly and transposed) for the denominator.
        sim12 = sim12.clone()

        # Denominator: log-sum-exp over negatives
        if self.dcl:
            sim12[match_idx[0], match_idx[1]] = float("-inf")

        if self.full_denom:
            sim11 = (x1 @ x1.T).fill_diagonal_(float("-inf"))
            sim1 = torch.cat((sim12, sim11), dim=1)[match_idx[0]]

            sim22 = (x2 @ x2.T).fill_diagonal_(float("-inf"))
            sim2 = torch.cat((sim12.T, sim22), dim=1)[match_idx[1]]
        else:
            sim1 = sim12[match_idx[0]]
            sim2 = sim12.T[match_idx[1]]

        denom1 = self._masked_logsumexp_mean(sim1)
        denom2 = self._masked_logsumexp_mean(sim2)

        denom = (denom1 + denom2) / 2

        return numer, denom

    @staticmethod
    def _masked_logsumexp_mean(sim: Tensor) -> Tensor:
        """Mean over rows of ``logsumexp(negatives) - log K``, NaN-safe.

        A matched unit that is the only unit in its batch element has no
        in-batch negatives, so its similarity row is entirely ``-inf``.
        ``logsumexp`` of an all-``-inf`` row is ``-inf`` with a NaN gradient
        (``0 * softmax(-inf)``), and masking the row out *after* ``logsumexp``
        with ``torch.where`` does not help — the NaN is produced inside the
        ``logsumexp`` Jacobian. Such rows carry no contrastive signal, so drop
        them before the reduction; if every row is empty, return a
        graph-connected zero.
        """
        k = sim.size(1)
        sim = sim[torch.isfinite(sim).any(dim=1)]
        if sim.size(0) == 0:
            return sim.sum() * 0.0
        return torch.mean(sim.logsumexp(1) - math.log(k))
