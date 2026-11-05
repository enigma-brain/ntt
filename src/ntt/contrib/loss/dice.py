"""Dice (Soft F1) loss with Tversky generalization for sparse spike detection.

Dice loss is a set-level metric that is inherently invariant to class
imbalance, making it well-suited for 1ms-binned spike data where >95%
of bins are zero.

The Tversky generalization allows asymmetric weighting of false positives
vs false negatives via alpha/beta parameters:
    Tversky = TP / (TP + alpha*FP + beta*FN)
    alpha = beta = 0.5  =>  standard Dice (F1)

Reference: Sudre et al., "Generalised Dice overlap as a deep learning
loss function for highly unbalanced segmentations", DLMIA 2017.
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
from ntt.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class DiceLossSpec(LossSpec, frozen=True):  # pyright: ignore[reportCallIssue,reportGeneralTypeIssues]
    """Spec for Dice / Tversky loss.

    Parameters:
        eps: Additive smoothing constant (1.0 = smooth Dice).
        alpha: Weight for false positives in Tversky denominator.
        beta: Weight for false negatives in Tversky denominator.
            alpha = beta = 0.5 gives standard Dice (equivalent to F1).
    """

    KIND: Literal["dice"] = "dice"
    eps: float = 1.0
    alpha: float = 0.5
    beta: float = 0.5
    supported_latents: list = []
    supported_outputs: list[DataType] = [DataType.SPIKES]


@register_loss
class DiceLoss(BaseLoss):
    """Dice (Soft F1 / Tversky) loss for sparse binary spike detection.

    Computes the Tversky index per-neuron over the time dimension, then
    reduces over (B, N). Overrides ``forward()`` because Dice is a
    set-level metric — it cannot produce a meaningful per-element (B, T, N)
    loss tensor.

    Supports ``gaussian_sigma`` for soft-target Dice (temporal tolerance).
    Raises if ``distance_weight_window`` is set (incompatible with set-level loss).

    .. note:: RATES output compatibility

        When used with rate-valued outputs, predictions and targets must
        be in [0, 1] for the Tversky FP/FN terms (``1 - pred``,
        ``1 - target``) to be meaningful. This holds naturally at small
        bin sizes (e.g. 1ms) where expected spike counts per bin are << 1.
        At larger bin sizes or with high-firing neurons, values may exceed
        1 and the loss will produce incorrect gradients. A warning is
        emitted at construction time when ``output_type=RATES``.
    """

    spec_class = DiceLossSpec
    spec: DiceLossSpec

    def __init__(self, spec: DiceLossSpec):
        if spec.distance_weight_window is not None:
            raise ValueError(
                "distance_weight_window is not compatible with Dice loss "
                "(set-level metric; per-element sample weights are not applicable)."
            )
        if DataType.RATES in spec.supported_outputs:
            log.warning(
                "Dice loss with RATES output assumes predictions and targets "
                "are in [0, 1]. This holds at small bin sizes (e.g. 1ms) but "
                "may produce incorrect gradients at larger bin sizes where "
                "rates exceed 1.0."
            )
        super().__init__(spec)

    def compute_loss(
        self,
        input_data: Tensor,
        reconstruction: Tensor,
        target: Tensor,
        sample_weights: Tensor | None,
    ) -> Tensor:
        # Not used — forward() is overridden. Kept for ABC compliance.
        raise NotImplementedError("DiceLoss overrides forward() directly.")

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
        target = input_data.float().clamp(min=0.0)

        # Apply Gaussian smoothing to targets if configured
        if self._smooth_targets is not None:
            target = self._smooth_targets(target)

        # Zero out padded positions before summation
        if pad_mask is not None:
            mask = pad_mask.float().to(pred.device)
            pred = pred * mask
            target = target * mask

        # Per-neuron sums over time: (B, T, N) -> (B, N)
        tp = (pred * target).sum(dim=1)
        fp = (pred * (1.0 - target)).sum(dim=1)
        fn = ((1.0 - pred) * target).sum(dim=1)

        eps = self.spec.eps
        alpha = self.spec.alpha
        beta = self.spec.beta

        # Tversky index: TP / (TP + alpha*FP + beta*FN)
        tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
        loss = 1.0 - tversky  # (B, N)

        if self.spec.reduction == "mean":
            return loss.mean()
        elif self.spec.reduction == "sum":
            return loss.sum()
        else:  # none
            return loss
