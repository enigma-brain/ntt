"""L1 (Mean Absolute Error) loss for tokenizer reconstruction."""

from typing import Literal

from torch import Tensor

from ntt.core.components.types import DataType
from ntt.core.model.loss.base import BaseLoss, LossSpec, register_loss


class L1LossSpec(LossSpec, frozen=True):
    """Spec for L1 / MAE loss."""

    KIND: Literal["l1"] = "l1"
    supported_latents: list = []
    supported_outputs: list[DataType] = [DataType.RATES]


@register_loss
class L1Loss(BaseLoss):
    """L1 (Mean Absolute Error) loss for reconstruction."""

    spec_class = L1LossSpec

    def __init__(self, spec: L1LossSpec):
        super().__init__(spec)

    def compute_loss(
        self,
        input_data: Tensor,
        reconstruction: Tensor,
        target: Tensor,
        sample_weights: Tensor | None,
    ) -> Tensor:
        loss = (reconstruction - target).abs()
        if sample_weights is not None:
            loss = loss * sample_weights
        return loss
