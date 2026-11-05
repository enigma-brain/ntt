"""Composite loss that combines multiple loss functions with weights.

Enables configurations like Dice + Wasserstein, or BCE + spectral loss,
each with independent weights and target transforms.  Component losses
are stored as detached tensors for logging via ``get_component_losses()``.

Example YAML:
    _target_: ntt.contrib.loss.MultiLossSpec
    KIND: multi
    reduction: mean
    components:
      - loss:
          KIND: dice
          eps: 1.0
        weight: 1.0
        name: dice
      - loss:
          KIND: wasserstein
          p: 1
        weight: 0.1
        name: wasserstein
"""

from typing import Literal

import torch
from torch import Tensor

from ntt.core.components.types import (
    BinnedSpikesInput,
    DataType,
    RatesInput,
    SpikeTimesInput,
    TokenizerOutput,
)
from ntt.core.model.loss.base import BaseLoss, LossSpec, create_loss, register_loss
from ntt.utils import NTTBaseModel, RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class MultiLossComponent(NTTBaseModel, frozen=True, extra="forbid"):  # pyright: ignore[reportCallIssue,reportGeneralTypeIssues]
    """A single component in a MultiLoss.

    Parameters:
        loss: Child loss configuration as a dict. Validated at runtime
            against the LossSpecKind discriminated union.
        weight: Multiplicative weight for this component's loss value.
        name: Display name for logging. Defaults to the child's KIND.
    """

    loss: dict  # Validated at runtime to avoid circular imports with LossSpecKind
    weight: float = 1.0
    name: str | None = None


class MultiLossSpec(LossSpec, frozen=True):  # pyright: ignore[reportCallIssue,reportGeneralTypeIssues]
    """Spec for composite multi-loss.

    Combines multiple loss functions with configurable weights.
    The ``reduction`` field is enforced on all children for consistent
    scale across components.

    Parameters:
        components: List of loss components with weights and optional names.
    """

    KIND: Literal["multi"] = "multi"
    components: list[MultiLossComponent]

    # MultiLoss delegates output-type validation to its children.
    # Set to None so the litmodule's top-level check passes; actual
    # validation happens in MultiLoss.__init__().
    supported_outputs: None = None
    supported_latents: None = None


@register_loss
class MultiLoss(BaseLoss):
    """Composite loss combining multiple child losses with weights.

    Each child is a fully independent ``BaseLoss`` instance with its own
    target transforms (``gaussian_sigma``, ``distance_weight_window``).
    The parent's ``reduction`` is forced onto all children for consistent
    scale.

    Per-component losses (detached) are available via
    ``get_component_losses()`` for logging.
    """

    spec_class = MultiLossSpec
    spec: MultiLossSpec

    def __init__(self, spec: MultiLossSpec):
        if spec.gaussian_sigma is not None:
            raise ValueError(
                "gaussian_sigma on MultiLossSpec is not supported. "
                "Set it on individual component loss specs instead."
            )
        if spec.distance_weight_window is not None:
            raise ValueError(
                "distance_weight_window on MultiLossSpec is not supported. "
                "Set it on individual component loss specs instead."
            )
        super().__init__(spec)

        # Lazy import to break circular dependency with __init__.py
        from pydantic import TypeAdapter

        from ntt.contrib.loss import LossSpecKind

        adapter = TypeAdapter(LossSpecKind)

        self._components: list[tuple[str, float, BaseLoss]] = []
        self._last_component_losses: dict[str, Tensor] = {}

        for comp in spec.components:
            # Validate the dict into a concrete LossSpec subclass
            child_spec = adapter.validate_python(comp.loss)
            if child_spec is None:
                raise ValueError(f"Component loss resolved to None. Check the config: {comp.loss}")

            # Force child reduction to match parent
            if child_spec.reduction != spec.reduction:
                log.info(
                    f"Overriding reduction '{child_spec.reduction}' -> '{spec.reduction}' "
                    f"for child loss '{child_spec.KIND}'"
                )
                child_spec = child_spec.model_copy(update={"reduction": spec.reduction})

            child_loss = create_loss(child_spec)
            name = comp.name or child_spec.KIND
            self._components.append((name, comp.weight, child_loss))
            # Register as submodule for device/state management
            self.add_module(f"loss_{name}", child_loss)

        if not self._components:
            raise ValueError("MultiLoss requires at least one component.")

        # Validate that children have compatible supported_outputs
        self._validate_children_compat()

    def _validate_children_compat(self) -> None:
        """Check that all children agree on supported output types."""
        constrained: set[DataType] | None = None
        for name, _, child in self._components:
            child_supported = child.spec.supported_outputs
            if child_supported is not None and len(child_supported) > 0:
                child_set = set(child_supported)
                if constrained is None:
                    constrained = child_set
                else:
                    overlap = constrained & child_set
                    if not overlap:
                        raise ValueError(
                            f"MultiLoss children have incompatible supported_outputs. "
                            f"No overlap found when adding child '{name}' "
                            f"(supports {child_supported})."
                        )
                    constrained = overlap

    def compute_loss(
        self,
        input_data: Tensor,
        reconstruction: Tensor,
        target: Tensor,
        sample_weights: Tensor | None,
    ) -> Tensor:
        raise NotImplementedError("MultiLoss overrides forward() directly.")

    def forward(
        self,
        input: SpikeTimesInput | BinnedSpikesInput | RatesInput,
        output: TokenizerOutput,
        pad_mask: Tensor | None = None,
    ) -> Tensor:
        assert output.reconstruction is not None, "TokenizerOutput.reconstruction is None"
        total_loss = torch.tensor(0.0, device=output.reconstruction.device, requires_grad=True)
        self._last_component_losses = {}

        for name, weight, child in self._components:
            child_loss = child(input, output, pad_mask)
            self._last_component_losses[name] = child_loss.detach()
            total_loss = total_loss + weight * child_loss

        return total_loss

    def get_component_losses(self) -> dict[str, Tensor] | None:
        """Return per-component losses (detached) from the last forward pass."""
        if not self._last_component_losses:
            return None
        return self._last_component_losses
