"""Poisson Negative Log-Likelihood loss for neural spike data.

This loss is appropriate for predicting firing rates from spike count data,
as neural spike counts follow a Poisson distribution.
"""

from typing import Literal

import torch.nn.functional as F
from torch import Tensor

from ntt.core.components.types import DataType
from ntt.core.model.loss.base import BaseLoss, LossSpec, register_loss


class PoissonNLLLossSpec(LossSpec, frozen=True):  # pyright: ignore[reportCallIssue,reportGeneralTypeIssues]
    """Spec for Poisson Negative Log-Likelihood loss.

    The Poisson NLL loss computes: -target * log(input) + input + log(target!)
    This is appropriate when the model predicts firing rates and the targets
    are spike counts.

    Parameters:
        log_input: If True, the input is expected to be log(rate) rather than rate
        full: If True, include the Stirling approximation term: log(target!)
        eps: Small value added to rates for numerical stability
    """

    KIND: Literal["poisson_nll"] = "poisson_nll"
    log_input: bool = False
    full: bool = False
    eps: float = 1e-8
    supported_latents: list = []
    supported_outputs: list[DataType] = [DataType.RATES, DataType.SPIKES]


@register_loss
class PoissonNLLLoss(BaseLoss):
    """Poisson Negative Log-Likelihood loss for spike rate prediction.

    Computes the Poisson NLL between predicted firing rates and observed
    spike counts. The Poisson distribution models spike counts:
    P(k|λ) = λ^k * e^(-λ) / k!, with NLL: -k * log(λ) + λ + log(k!).
    """

    spec_class = PoissonNLLLossSpec
    spec: PoissonNLLLossSpec

    def __init__(self, spec: PoissonNLLLossSpec):
        super().__init__(spec)

    def compute_loss(
        self,
        input_data: Tensor,
        reconstruction: Tensor,
        target: Tensor,
        sample_weights: Tensor | None,
    ) -> Tensor:
        rates = reconstruction.clamp(min=self.spec.eps)
        loss = F.poisson_nll_loss(
            rates,
            target,
            log_input=self.spec.log_input,
            full=self.spec.full,
            reduction="none",
        )
        if sample_weights is not None:
            loss = loss * sample_weights
        return loss
