"""Global firing rate loss for spike count calibration.

Penalizes the mismatch between total predicted rate and total observed
spike count per neuron over the full time window.  This constrains the
*integral* of the predicted rate curve without interfering with its
per-bin shape, making it a natural complement to Poisson NLL.

Loss per neuron: (sum(pred_rates) - sum(target_spikes))^2
"""

from typing import Literal

from torch import Tensor

from ntt.core.components.types import (
    BinnedSpikesInput,
    DataType,
    RatesInput,
    SpikeTimesInput,
    TokenizerOutput,
)
from ntt.core.model.loss.base import BaseLoss, LossSpec, register_loss


class GlobalRateLossSpec(LossSpec, frozen=True):  # pyright: ignore[reportCallIssue,reportGeneralTypeIssues]
    """Spec for global firing rate loss.

    Computes the squared error between total predicted rate and total
    observed spike count per neuron, averaged over neurons and batch.

    Parameters:
        normalize: If True, divide per-neuron sums by the number of valid
            timesteps so the loss is in units of (rate/bin)^2 rather than
            (count)^2.  Recommended when mixing windows of different lengths.
    """

    KIND: Literal["global_rate"] = "global_rate"
    normalize: bool = False
    supported_latents: list = []
    supported_outputs: list[DataType] = [DataType.RATES, DataType.SPIKES]


@register_loss
class GlobalRateLoss(BaseLoss):
    """Global firing rate loss — MSE on integrated rates vs spike counts.

    Operates per-neuron over the time dimension, then reduces over (B, N).
    Overrides ``forward()`` because this is a sequence-level metric that
    sums over time before computing the error.
    """

    spec_class = GlobalRateLossSpec
    spec: GlobalRateLossSpec

    def __init__(self, spec: GlobalRateLossSpec):
        if spec.distance_weight_window is not None:
            raise ValueError(
                "distance_weight_window is not compatible with GlobalRateLoss "
                "(sequence-level metric; per-element sample weights are not applicable)."
            )
        super().__init__(spec)

    def compute_loss(
        self,
        input_data: Tensor,
        reconstruction: Tensor,
        target: Tensor,
        sample_weights: Tensor | None,
    ) -> Tensor:
        raise NotImplementedError("GlobalRateLoss overrides forward() directly.")

    def forward(
        self,
        input: SpikeTimesInput | BinnedSpikesInput | RatesInput,
        output: TokenizerOutput,
        pad_mask: Tensor | None = None,
    ) -> Tensor:
        input_data, input_pad_mask = self._extract_input_data(input)
        if pad_mask is None:
            pad_mask = input_pad_mask

        reconstruction = output.reconstruction
        assert reconstruction is not None, "TokenizerOutput.reconstruction is None"

        pred = reconstruction.clamp(min=0.0)
        target = input_data.float()

        # Note: we do NOT apply gaussian_sigma here. The target for global
        # rate is raw spike counts — smoothing doesn't change the integral
        # (peak-normalized Gaussian preserves the sum), but using raw counts
        # is conceptually cleaner.

        # Zero out padded positions before summation
        if pad_mask is not None:
            mask = pad_mask.float().to(pred.device)
            pred = pred * mask
            target = target * mask

        # Per-neuron sums over time: (B, T, N) -> (B, N)
        pred_rate = pred.sum(dim=1)
        target_rate = target.sum(dim=1)

        if self.spec.normalize and pad_mask is not None:
            real_counts = mask.sum(dim=1).clamp(min=1.0)  # (B, N)
            pred_rate = pred_rate / real_counts
            target_rate = target_rate / real_counts

        # Squared error on global rate per neuron
        loss = (pred_rate - target_rate) ** 2  # (B, N)

        if self.spec.reduction == "mean":
            return loss.mean()
        elif self.spec.reduction == "sum":
            return loss.sum()
        else:  # none
            return loss
