"""Huber (Smooth L1) loss for tokenizer reconstruction."""

from typing import Literal

import torch.nn.functional as F
from torch import Tensor

from ntt.core.components.types import DataType
from ntt.core.model.loss.base import BaseLoss, LossSpec, register_loss


class HuberLossSpec(LossSpec, frozen=True):
    """Spec for Huber loss.

    Parameters:
        delta: Threshold at which loss transitions from quadratic to linear.
    """

    KIND: Literal["huber"] = "huber"
    delta: float = 1.0
    supported_latents: list = []
    supported_outputs: list[DataType] = [DataType.RATES]


@register_loss
class HuberLoss(BaseLoss):
    """Huber (Smooth L1) loss for reconstruction."""

    spec_class = HuberLossSpec
    spec: HuberLossSpec

    def __init__(self, spec: HuberLossSpec):
        super().__init__(spec)

    def compute_loss(
        self,
        input_data: Tensor,
        reconstruction: Tensor,
        target: Tensor,
        sample_weights: Tensor | None,
    ) -> Tensor:
        loss = F.huber_loss(reconstruction, target, reduction="none", delta=self.spec.delta)
        if sample_weights is not None:
            loss = loss * sample_weights
        return loss
