"""Learning rate scheduler configuration and builder.

This module provides a flexible LR scheduler configuration framework adapted from
torchtitan: https://github.com/pytorch/torchtitan/blob/main/torchtitan/components/lr_scheduler.py
(BSD-3-Clause; see ``LICENSES/torchtitan-BSD-3-Clause.txt``).

The scheduler supports warmup-stable-decay (WSD) schedules with multiple decay types.
"""

import functools
import math
from typing import Literal

import pydantic as pdt
from lightning.pytorch.utilities.types import LRSchedulerConfigType
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from ntt.utils import NTTBaseModel, RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class LRSchedulerConfig(NTTBaseModel, strict=False, frozen=True, extra="forbid"):
    """Configuration for learning rate scheduler with warmup-stable-decay support.

    Combines scheduling parameters from torchtitan with Lightning integration fields.
    Supports linear warmup followed by an optional stable phase, then decay using
    one of several decay functions.

    The Warmup-Stable-Decay (WSD) schedule is described in https://arxiv.org/abs/2404.06395.

    Attributes:
        warmup_steps: Number of warmup steps for linear warmup phase.
        decay_ratio: Proportion of training steps for decay phase. If None, decay
            begins immediately after warmup. If set, learning rate stays stable after
            warmup and only decays during the last decay_ratio portion of training.
        decay_type: Type of decay function ('linear', 'sqrt', or 'cosine').
        min_lr_factor: Minimum learning rate as a factor of initial LR. The decay
            range is scaled from 1 to min_lr_factor.
        name: Optional name for the scheduler (Lightning integration).
        interval: When to step the scheduler - 'step' or 'epoch' (Lightning integration).
        frequency: How often to step the scheduler (Lightning integration).
        monitor: Accepted so that serialized scheduler configs round-trip; ignored.
        strict: Accepted so that serialized scheduler configs round-trip; ignored.

    Note:
        ReduceLROnPlateau is not supported, so `monitor` and `strict` have no effect in
        build_lr_scheduler.

    Example YAML:
        ```yaml
        lr_scheduler:
          warmup_steps: 200
          decay_ratio: 0.2
          decay_type: cosine
          min_lr_factor: 0.1
          interval: step
        ```
    """

    # Scheduling parameters (from torchtitan)
    warmup_steps: int = pdt.Field(
        default=200,
        ge=0,
        description="Steps for lr scheduler warmup, normally 1/5 of total training steps.",
    )
    decay_ratio: float | None = pdt.Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Proportion of training steps for decay phase. If None, decay begins "
        "immediately after warmup. Otherwise, LR stays stable after warmup and decays "
        "during the last decay_ratio portion of training (WSD schedule).",
    )
    decay_type: Literal["linear", "sqrt", "cosine"] = pdt.Field(
        default="linear",
        description="Learning rate decay type: 'linear' (linear decay), 'sqrt' (1 minus "
        "square root curve), or 'cosine' (cosine annealing).",
    )
    min_lr_factor: float = pdt.Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum learning rate as a factor of initial LR. Decay range is scaled "
        "from 1 to min_lr_factor to ensure LR doesn't drop below optimizer.lr * min_lr_factor.",
    )

    # Lightning integration fields
    name: str | None = pdt.Field(
        default=None,
        description="Optional name for the scheduler.",
    )
    interval: Literal["step", "epoch"] = pdt.Field(
        default="step",
        description="When to step the scheduler: 'step' (after each batch) or 'epoch'.",
    )
    frequency: int = pdt.Field(
        default=1,
        ge=1,
        description="How often to step the scheduler (every N intervals).",
    )
    # Accepted but unused: the config forbids extra keys, so serialized configs that
    # carry these fields must still validate.
    monitor: str | None = pdt.Field(
        default=None,
        description="Metric to monitor (ignored; ReduceLROnPlateau is not supported).",
    )
    strict: bool = pdt.Field(
        default=True,
        description="Whether the monitored metric must exist (ignored).",
    )


def build_lr_scheduler(
    optimizer: Optimizer,
    config: LRSchedulerConfig,
    training_steps: int,
) -> LRSchedulerConfigType:
    """Build a learning rate scheduler for PyTorch Lightning.

    Creates a LambdaLR scheduler with warmup-stable-decay schedule and returns it
    in a format compatible with Lightning's configure_optimizers().

    Adapted from torchtitan:
    https://github.com/pytorch/torchtitan/blob/main/torchtitan/components/lr_scheduler.py

    TODO: Multiple schedulers are currently not supported. This function handles
    a single optimizer/scheduler pair. For multi-optimizer setups, call this
    function multiple times.

    Args:
        optimizer: The optimizer to schedule.
        config: LRSchedulerConfig with scheduling and Lightning integration parameters.
        training_steps: Total number of training steps.

    Returns:
        LRSchedulerConfigType dict compatible with Lightning's configure_optimizers().

    Example:
        ```python
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        config = LRSchedulerConfig(warmup_steps=100, decay_type="cosine")
        scheduler_config = build_lr_scheduler(optimizer, config, training_steps=1000)

        # In LightningModule.configure_optimizers:
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}
        ```
    """
    warmup_steps = config.warmup_steps

    if warmup_steps > training_steps:
        log.warning(
            f"Warmup steps ({warmup_steps}) exceed total training steps ({training_steps}). "
            f"Adjusting warmup steps to {training_steps}."
        )
        warmup_steps = training_steps

    if config.decay_ratio is not None:
        decay_steps = round(training_steps * config.decay_ratio)
        if warmup_steps + decay_steps > training_steps:
            log.warning(
                f"Warmup ({warmup_steps}) + decay ({decay_steps}) steps exceed "
                f"total training steps ({training_steps}). "
                f"Adjusting decay steps to {training_steps - warmup_steps}."
            )
            decay_steps = training_steps - warmup_steps
    else:
        decay_steps = training_steps - warmup_steps

    # Add a virtual last step to prevent the learning rate from dropping to 0
    stable_steps = training_steps + 1 - warmup_steps - decay_steps

    def _linear_warmup_stable_decay(
        current_step: int,
        warmup_steps: int,
        stable_steps: int,
        decay_steps: int,
        decay_type: str,
        min_lr_factor: float,
    ) -> float:
        """Compute LR multiplier for warmup-stable-decay schedule.

        Computes linear warmup followed by stable learning rate for a while,
        then some type of decay.

        Per LambdaLR requirement, this is accomplished by returning a multiplicative
        factor ranging from 0 to 1 to adjust the learning rate.

        Decay types:
        - 'linear': decays linearly from 1 to 0 over the decay period.
        - 'sqrt': decays as 1 minus the square root of the decay progress.
        - 'cosine': follows a cosine curve (half-period).

        If min_lr_factor is specified, the decay range is scaled from 1 to min_lr_factor.

        Args:
            current_step: Current training step (0-indexed).
            warmup_steps: Number of warmup steps.
            stable_steps: Number of steps at stable LR after warmup.
            decay_steps: Number of decay steps.
            decay_type: Type of decay ('linear', 'sqrt', 'cosine').
            min_lr_factor: Minimum LR as factor of initial LR.

        Returns:
            Learning rate multiplier for the current step.
        """
        warmup_stable_steps = warmup_steps + stable_steps

        if current_step < warmup_steps:
            # Linear warmup (0-indexed step, hence + 1 adjustments)
            current_step += 1
            if warmup_steps == 0:
                return 1.0
            return float(current_step / warmup_steps)

        if current_step < warmup_stable_steps:
            # Stable phase
            return 1.0

        # Decay phase (0-indexed step, hence + 1 adjustments)
        current_step += 1
        if decay_steps == 0:
            return min_lr_factor

        progress = float(current_step - warmup_stable_steps) / decay_steps

        if decay_type == "linear":
            curr_adjustment = 1 - progress
        elif decay_type == "sqrt":
            curr_adjustment = 1 - math.sqrt(progress)
        elif decay_type == "cosine":
            curr_adjustment = 0.5 * (1.0 + math.cos(math.pi * progress))
        else:
            raise ValueError(f"Unknown decay_type: {decay_type}")

        return min_lr_factor + (1 - min_lr_factor) * curr_adjustment

    lr_lambda = functools.partial(
        _linear_warmup_stable_decay,
        warmup_steps=warmup_steps,
        stable_steps=stable_steps,
        decay_steps=decay_steps,
        decay_type=config.decay_type,
        min_lr_factor=config.min_lr_factor,
    )

    scheduler = LambdaLR(optimizer, lr_lambda)

    # reduce_on_plateau, monitor and strict are fixed: ReduceLROnPlateau is not supported.
    return LRSchedulerConfigType(
        scheduler=scheduler,
        name=config.name,
        interval=config.interval,
        frequency=config.frequency,
        reduce_on_plateau=False,
        monitor=None,
        strict=True,
    )
