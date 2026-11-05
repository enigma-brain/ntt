"""Binary Cross-Entropy loss for binary spike prediction."""

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from ntt.core.components.types import DataType
from ntt.core.model.loss.base import BaseLoss, LossSpec, register_loss


class BCELossSpec(LossSpec, frozen=True):  # pyright: ignore[reportCallIssue,reportGeneralTypeIssues]
    """Spec for Binary Cross-Entropy loss."""

    KIND: Literal["bce"] = "bce"
    supported_latents: list = []
    # Rates are permitted in addition to spikes: sigmoid-bounded rates in [0, 1]
    # are the continuous form of the Bernoulli targets BCE models.
    supported_outputs: list[DataType] = [DataType.SPIKES, DataType.RATES]


@register_loss
class BCELoss(BaseLoss):
    """Binary Cross-Entropy loss for binary spike reconstruction."""

    spec_class = BCELossSpec

    def __init__(self, spec: BCELossSpec):
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
        loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        if sample_weights is not None:
            loss = loss * sample_weights
        return loss
