"""Focal loss for sparse binary spike prediction.

Note: this function expects **raw logits**, not probabilities. Since the
tokenizer output has already been through sigmoid activation, we invert
it back to logits via ``torch.logit()`` before calling the function.

Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.

Focal loss implementation adapted from torchvision (BSD-3-Clause license):
https://github.com/pytorch/vision/blob/main/torchvision/ops/focal_loss.py
Original: https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
"""

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from ntt.core.components.types import DataType
from ntt.core.model.loss.base import BaseLoss, LossSpec, register_loss


class FocalLossSpec(LossSpec, frozen=True):  # pyright: ignore[reportCallIssue,reportGeneralTypeIssues]
    """Spec for Focal loss.

    Parameters:
        alpha: Weighting factor for the positive class (spikes).
        gamma: Focusing parameter. Higher values downweight easy examples more.
    """

    KIND: Literal["focal"] = "focal"
    alpha: float = 0.25
    gamma: float = 2.0
    supported_latents: list = []
    supported_outputs: list[DataType] = [DataType.SPIKES]


@register_loss
class FocalLoss(BaseLoss):
    """Focal loss for sparse binary spike prediction."""

    spec_class = FocalLossSpec
    spec: FocalLossSpec

    def __init__(self, spec: FocalLossSpec):
        super().__init__(spec)

    def compute_loss(
        self,
        input_data: Tensor,
        reconstruction: Tensor,
        target: Tensor,
        sample_weights: Tensor | None,
    ) -> Tensor:
        # The tokenizer emits probabilities, so the sigmoid is inverted here.
        # Note: clamp + logit(eps=1e-6) saturates values near 0/1 to ±13.8,
        # which may cause gradient cliffs.
        pred = reconstruction.clamp(0.0, 1.0)
        target = target.clamp(0.0, 1.0)
        logits = torch.logit(pred, eps=1e-6)

        # Focal loss: CE * (1 - p_t)^gamma, optionally weighted by alpha
        ce_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * target + (1 - p) * (1 - target)
        loss = ce_loss * ((1 - p_t) ** self.spec.gamma)

        if self.spec.alpha >= 0:
            alpha_t = self.spec.alpha * target + (1 - self.spec.alpha) * (1 - target)
            loss = alpha_t * loss

        if sample_weights is not None:
            loss = loss * sample_weights
        return loss
