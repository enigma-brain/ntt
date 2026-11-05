"""1D Wasserstein (Earth Mover's) distance loss for spike train comparison.

Computes the Wasserstein-p distance between predicted and target spike
trains via cumulative distribution comparison along the time axis.
In 1D this has a closed-form solution — no Sinkhorn iterations needed.

For unnormalized inputs (default), the CDF difference captures both
spike count errors and temporal displacement errors.  For normalized
inputs, it is a pure timing metric.

Reference: Vallender, "Calculation of the Wasserstein distance between
probability distributions on the line", Theory of Probability & Its
Applications, 1974.
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


class WassersteinLossSpec(LossSpec, frozen=True):  # pyright: ignore[reportCallIssue,reportGeneralTypeIssues]
    """Spec for 1D Wasserstein loss.

    Parameters:
        normalize: If True, normalize both signals to sum to 1 (pure timing
            metric). If False (default), use unnormalized cumulative sums
            which penalize both count and timing errors.
        p: Order of the Wasserstein distance. 1 = W1 (L1 of CDFs),
            2 = W2 (L2 of CDFs).
    """

    KIND: Literal["wasserstein"] = "wasserstein"
    normalize: bool = False
    p: int = 1
    supported_latents: list = []
    supported_outputs: list[DataType] = [DataType.SPIKES, DataType.RATES]


@register_loss
class WassersteinLoss(BaseLoss):
    """1D Wasserstein distance between predicted and target spike trains.

    Computed per-neuron over the time dimension via cumulative sum
    comparison, then reduced over (B, N). Overrides ``forward()`` because
    the Wasserstein distance is a sequence-level metric.

    Supports ``gaussian_sigma`` for smoothed target comparison.
    Raises if ``distance_weight_window`` is set (incompatible).

    .. note:: Normalized Wasserstein variant

        For p > 1 (e.g. W2), the loss computes ``(mean_t |d_t|^p)^(1/p)``
        rather than the standard ``(sum_t |d_t|^p)^(1/p)``.  This is a
        time-normalized variant that keeps the loss scale independent of
        sequence length.  For p = 1 (W1), normalization is a simple scalar
        factor and the ordering of inputs is preserved.
    """

    spec_class = WassersteinLossSpec
    spec: WassersteinLossSpec

    def __init__(self, spec: WassersteinLossSpec):
        if spec.distance_weight_window is not None:
            raise ValueError(
                "distance_weight_window is not compatible with Wasserstein loss "
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
        raise NotImplementedError("WassersteinLoss overrides forward() directly.")

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

        # Apply Gaussian smoothing to targets if configured
        if self._smooth_targets is not None:
            target = self._smooth_targets(target)

        # Zero out padded positions before cumsum
        if pad_mask is not None:
            mask = pad_mask.float().to(pred.device)
            pred = pred * mask
            target = target * mask

        # Normalize to probability distributions if requested
        if self.spec.normalize:
            eps = 1e-8
            pred_sum = pred.sum(dim=1, keepdim=True)
            target_sum = target.sum(dim=1, keepdim=True)
            pred = pred / (pred_sum + eps)
            target = target / (target_sum + eps)

        # Cumulative sums along time: (B, T, N)
        cdf_pred = pred.cumsum(dim=1)
        cdf_target = target.cumsum(dim=1)

        # Per-element CDF difference
        diff = cdf_pred - cdf_target
        if self.spec.p == 1:
            elem_loss = diff.abs()
        else:
            elem_loss = diff.abs().pow(self.spec.p)

        # Sum over time, respecting mask
        if pad_mask is not None:
            # Count real timesteps per (b, n) for normalization
            real_counts = mask.sum(dim=1)  # (B, N)
            real_counts = real_counts.clamp(min=1.0)
            loss_per_neuron = (elem_loss * mask).sum(dim=1) / real_counts  # (B, N)
        else:
            T = elem_loss.shape[1]
            loss_per_neuron = elem_loss.sum(dim=1) / T  # (B, N)

        # Take p-th root for W_p when p > 1
        if self.spec.p > 1:
            loss_per_neuron = loss_per_neuron.pow(1.0 / self.spec.p)

        if self.spec.reduction == "mean":
            return loss_per_neuron.mean()
        elif self.spec.reduction == "sum":
            return loss_per_neuron.sum()
        else:  # none
            return loss_per_neuron
