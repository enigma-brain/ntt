"""Eval-only MSE loss for tokenizers that skip reconstruction during training.

Returns zero during training (when reconstruction is meaningless) and
computes real MSE during evaluation, giving val/recon_loss progress
tracking without affecting training gradients.
"""

from typing import Literal

import torch
from torch import Tensor

from ntt.core.components.types import (
    BinnedSpikesInput,
    RatesInput,
    SpikeTimesInput,
    TokenizerOutput,
)
from ntt.core.model.loss.base import BaseLoss, LossSpec, register_loss


class EvalOnlyMSELossSpec(LossSpec, frozen=True):  # pyright: ignore[reportCallIssue,reportGeneralTypeIssues]
    """Spec for eval-only MSE loss.

    Use with tokenizers that skip reconstruction during training but
    produce real reconstructions during evaluation (e.g., compressed
    sensing with trainable sensing matrix).
    """

    KIND: Literal["eval_only_mse"] = "eval_only_mse"
    supported_latents: list = []
    supported_outputs: None = None


@register_loss
class EvalOnlyMSELoss(BaseLoss):
    """MSE loss that only computes during evaluation.

    During training (self.training=True): returns 0.
    During evaluation (self.training=False): computes masked MSE via template.
    """

    spec_class = EvalOnlyMSELossSpec

    def __init__(self, spec: EvalOnlyMSELossSpec):
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

    def forward(
        self,
        input: SpikeTimesInput | BinnedSpikesInput | RatesInput,
        output: TokenizerOutput,
        pad_mask: Tensor | None = None,
    ) -> Tensor:
        if self.training:
            device = output.reconstruction.device if output.reconstruction is not None else None
            return torch.tensor(0.0, device=device, requires_grad=True)
        return super().forward(input, output, pad_mask)
