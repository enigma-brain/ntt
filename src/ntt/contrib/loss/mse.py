"""Mean Squared Error loss for tokenizer reconstruction."""

from typing import Literal

from torch import Tensor

from ntt.core.model.loss.base import BaseLoss, LossSpec, register_loss


class MSELossSpec(LossSpec, frozen=True):  # pyright: ignore[reportCallIssue,reportGeneralTypeIssues]
    """Spec for Mean Squared Error loss.

    Computes element-wise squared error between reconstruction and target.
    Supports all output types (rates, spikes, etc.).
    """

    KIND: Literal["mse"] = "mse"
    supported_latents: list = []
    supported_outputs: None = None


@register_loss
class MSELoss(BaseLoss):
    """Mean Squared Error loss for reconstruction.

    Computes MSE between the input spike data and the reconstruction,
    respecting the pad mask to only compute loss on real data.
    """

    spec_class = MSELossSpec

    def __init__(self, spec: MSELossSpec):
        super().__init__(spec)

    def compute_loss(
        self,
        input_data: Tensor,
        reconstruction: Tensor,
        target: Tensor,
        sample_weights: Tensor | None,
    ) -> Tensor:
        loss = (reconstruction - target) ** 2
        if sample_weights is not None:
            loss = loss * sample_weights
        return loss
