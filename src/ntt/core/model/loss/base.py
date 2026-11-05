"""Base classes and registry for loss functions."""

import abc
from typing import ClassVar, Literal

import pydantic as pdt
import torch
from torch import Tensor, nn

from ntt.core.components.types import (
    BinnedSpikesInput,
    DataType,
    LatentType,
    RatesInput,
    SpikeTimesInput,
    TokenizerOutput,
)
from ntt.utils import NTTBaseModel, RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class LossSpec(abc.ABC, NTTBaseModel, strict=False, frozen=True, extra="forbid"):  # pyright: ignore[reportCallIssue,reportGeneralTypeIssues]
    """Base class for loss function configuration.

    Similar to TokenizerSpec, this defines the configuration for a loss function.
    Subclasses must define a KIND field with a Literal type to identify the loss.

    Example:
        class MSELossSpec(LossSpec, frozen=True):
            KIND: Literal["mse"] = "mse"

    Parameters:
        KIND (str): Discriminator field identifying the loss type. Subclasses must
            override with a Literal type (e.g., KIND: Literal["mse"] = "mse")
        reduction: How to reduce the loss over elements ('mean', 'sum', or 'none')

    Capability Metadata:
        supported_latents: List of latent types this loss supports, or None for all.
            An empty list means the loss doesn't operate on latents.
        supported_outputs: List of output data types this loss supports, or None for all.
            An empty list means the loss doesn't operate on outputs.
    """

    # Discriminator field - subclasses must override with a Literal type
    KIND: str

    # Common loss parameter
    reduction: Literal["mean", "sum", "none"] = pdt.Field(
        default="mean",
        description="How to reduce the loss over elements ('mean', 'sum', or 'none').",
    )

    # Capability metadata
    supported_latents: list[LatentType] | None = pdt.Field(
        default=None,
        description="Latent types this loss supports. None = all, [] = none.",
    )
    supported_outputs: list[DataType] | None = pdt.Field(
        default=None,
        description="Output data types this loss supports. None = all, [] = none.",
    )

    # Target transform parameters (optional)
    gaussian_sigma: float | None = pdt.Field(
        default=None,
        description="If set, smooth targets with Gaussian kernel of this sigma (in bins).",
    )
    distance_weight_window: int | None = pdt.Field(
        default=None,
        description=(
            "If set, apply distance-weighted penalty reduction within this window (in bins)."
        ),
    )


class BaseLoss(abc.ABC, nn.Module):
    """Base class for all loss function implementations.

    Uses a template method pattern: ``forward()`` is concrete and handles
    data extraction, target transforms, masking, and reduction. Subclasses
    implement ``compute_loss()`` for per-element loss math only.

    Example::

        @register_loss
        class MSELoss(BaseLoss):
            spec_class = MSELossSpec

            def __init__(self, spec: MSELossSpec):
                super().__init__(spec)

            def compute_loss(self, input_data, reconstruction, target, sample_weights):
                loss = (reconstruction - target) ** 2
                if sample_weights is not None:
                    loss = loss * sample_weights
                return loss
    """

    spec_class: ClassVar[type[LossSpec]]

    def __init__(self, spec: LossSpec):
        super().__init__()
        if not isinstance(spec, self.spec_class):
            raise TypeError(
                f"spec must be an instance of {self.spec_class.__name__}, "
                f"got {type(spec).__name__}"
            )
        self._spec = spec

        # Conditionally instantiate target transforms
        # Use getattr so checkpoints whose spec lacks these fields still load
        self._smooth_targets: nn.Module | None = None
        self._distance_weights: nn.Module | None = None
        gaussian_sigma = getattr(spec, "gaussian_sigma", None)
        distance_weight_window = getattr(spec, "distance_weight_window", None)
        if gaussian_sigma is not None:
            from ntt.core.model.loss.components import GaussianSmoothTargets

            self._smooth_targets = GaussianSmoothTargets(sigma=gaussian_sigma)
        if distance_weight_window is not None:
            from ntt.core.model.loss.components import DistanceWeightedMask

            self._distance_weights = DistanceWeightedMask(window_size=distance_weight_window)

    @property
    def spec(self) -> LossSpec:
        """Get the loss specification."""
        return self._spec

    def get_component_losses(self) -> dict[str, "torch.Tensor"] | None:
        """Return per-component losses for logging (e.g., from MultiLoss).

        Default implementation returns None, meaning this is a single loss
        with no sub-components to log separately.
        """
        return None

    def _extract_input_data(
        self, input: SpikeTimesInput | BinnedSpikesInput | RatesInput
    ) -> tuple[Tensor, Tensor | None]:
        """Extract data tensor and pad mask from the input batch."""
        if isinstance(input, BinnedSpikesInput):
            return input.spike_matrix.float(), input.pad_and_gap_mask
        elif isinstance(input, RatesInput):
            return input.rate_matrix.float(), input.pad_and_gap_mask
        elif isinstance(input, SpikeTimesInput):
            return input.spike_timestamps.float(), input.spike_pad_mask
        else:
            raise TypeError(f"Unsupported input type: {type(input)}")

    def _prepare_targets_and_weights(
        self, target: Tensor, pad_mask: Tensor | None
    ) -> tuple[Tensor, Tensor | None]:
        """Apply active target transforms and return (target, sample_weights).

        Order matters: distance weights are computed from the *original* (hard)
        targets so that proximity is measured against true spike positions.
        Gaussian smoothing is applied afterwards, so ``compute_loss`` receives
        smoothed targets but original-position-based weights.
        """
        sample_weights = None
        if self._distance_weights is not None:
            sample_weights = self._distance_weights(target)
        if self._smooth_targets is not None:
            target = self._smooth_targets(target)
        return target, sample_weights

    def _apply_mask_and_reduce(self, loss: Tensor, pad_mask: Tensor | None) -> Tensor:
        """Apply pad mask and reduction to per-element loss."""
        if pad_mask is not None:
            mask = pad_mask.float().to(loss.device)
            masked_loss = loss * mask
            num_real = mask.sum()

            if num_real > 0:
                if self.spec.reduction == "mean":
                    return masked_loss.sum() / num_real
                elif self.spec.reduction == "sum":
                    return masked_loss.sum()
                else:  # none
                    return masked_loss
            else:
                return torch.tensor(0.0, device=loss.device, requires_grad=True)
        else:
            if self.spec.reduction == "mean":
                return loss.mean()
            elif self.spec.reduction == "sum":
                return loss.sum()
            else:  # none
                return loss

    @abc.abstractmethod
    def compute_loss(
        self,
        input_data: Tensor,
        reconstruction: Tensor,
        target: Tensor,
        sample_weights: Tensor | None,
    ) -> Tensor:
        """Compute per-element loss. Subclasses must implement this.

        Args:
            input_data: Raw input data tensor (B, T, N).
            reconstruction: Model reconstruction tensor (B, T, N).
            target: Possibly transformed target tensor (B, T, N).
                Same as input_data unless target transforms are active.
            sample_weights: Per-element weights from distance weighting,
                or None if no distance weighting is active.

        Returns:
            Per-element loss tensor (B, T, N). Do NOT apply reduction.
        """
        raise NotImplementedError

    def forward(
        self,
        input: SpikeTimesInput | BinnedSpikesInput | RatesInput,
        output: TokenizerOutput,
        pad_mask: Tensor | None = None,
    ) -> Tensor:
        """Template method: extract data, transform targets, compute loss, reduce.

        Args:
            input: Original input data batch.
            output: Tokenizer output containing reconstruction.
            pad_mask: Boolean mask where True = real data. If None, uses input's mask.

        Returns:
            Loss tensor (scalar if reduction != "none").
        """
        input_data, input_pad_mask = self._extract_input_data(input)
        if pad_mask is None:
            pad_mask = input_pad_mask

        reconstruction = output.reconstruction
        assert reconstruction is not None, "TokenizerOutput.reconstruction is None"
        target, sample_weights = self._prepare_targets_and_weights(input_data, pad_mask)

        loss = self.compute_loss(input_data, reconstruction, target, sample_weights)
        return self._apply_mask_and_reduce(loss, pad_mask)


# Global registry mapping loss names to (SpecClass, LossClass)
_LOSS_REGISTRY: dict[str, tuple[type[LossSpec], type[BaseLoss]]] = {}


def register_loss(loss_class: type[BaseLoss]) -> type[BaseLoss]:
    """Decorator to register a loss implementation.

    Usage:
        @register_loss
        class MSELoss(BaseLoss):
            spec_class = MSELossSpec  # must have KIND: Literal["mse"] = "mse"

            def __init__(self, spec: MSELossSpec):
                super().__init__(spec)
            ...

    The decorator expects:
    1. A spec_class class attribute pointing to a LossSpec subclass
    2. The spec_class must have a `KIND` field with a Literal type default value

    The registration name is automatically inferred from the spec_class's KIND field.

    Returns:
        The decorated loss class
    """
    # Get the spec class from the loss class
    if not hasattr(loss_class, "spec_class"):
        raise ValueError(
            f"Loss class {loss_class.__name__} must define 'spec_class' class attribute"
        )

    spec_class = loss_class.spec_class
    if not issubclass(spec_class, LossSpec):
        raise TypeError(f"spec_class must be a subclass of LossSpec, got {spec_class}")

    # Get the KIND from the spec's model fields
    kind_field = spec_class.model_fields.get("KIND")
    if kind_field is None:
        raise ValueError(f"Spec class {spec_class.__name__} must define a 'KIND' field")

    name = kind_field.default
    if name is None or not isinstance(name, str):
        raise ValueError(
            f"Spec class {spec_class.__name__} must have a string default for KIND field"
        )

    if name in _LOSS_REGISTRY:
        log.warning(f"Loss '{name}' is already registered!")

    _LOSS_REGISTRY[name] = (spec_class, loss_class)
    return loss_class


def get_loss_class(name: str) -> tuple[type[LossSpec], type[BaseLoss]]:
    """Get loss spec and implementation classes by name.

    Args:
        name: Registered loss name

    Returns:
        Tuple of (spec_class, loss_class)

    Raises:
        KeyError: If loss name not found in registry
    """
    if name not in _LOSS_REGISTRY:
        available = ", ".join(_LOSS_REGISTRY.keys())
        raise KeyError(f"Loss '{name}' not found. Available losses: {available}")
    return _LOSS_REGISTRY[name]


def create_loss(spec: LossSpec) -> BaseLoss:
    """Factory function to create a loss from a spec.

    The spec must be registered via @register_loss decorator.
    Uses the spec's `KIND` field to look up the loss class.

    Args:
        spec: LossSpec instance (must be a registered type)

    Returns:
        Instantiated loss function

    Raises:
        KeyError: If loss KIND not found in registry

    Example:
        spec = MSELossSpec(reduction="mean")
        loss = create_loss(spec)
    """
    KIND = spec.KIND

    if KIND not in _LOSS_REGISTRY:
        available = ", ".join(_LOSS_REGISTRY.keys())
        raise KeyError(
            f"Loss KIND '{KIND}' not found in registry. "
            f"Available losses: {available}. "
            f"Did you forget to import the loss implementation?"
        )

    _spec_class, loss_class = _LOSS_REGISTRY[KIND]
    return loss_class(spec)


def list_losses() -> list[str]:
    """List all registered loss names.

    Returns:
        List of registered loss names
    """
    return list(_LOSS_REGISTRY.keys())
