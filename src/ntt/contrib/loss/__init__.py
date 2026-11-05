"""User-contributed loss function implementations.

Contributors should place their loss function implementations here.

Each loss file should:
1. Define a LossSpec subclass with KIND: Literal["name"] = "name"
2. Define a BaseLoss subclass with spec_class = YourLossSpec
3. Register using @register_loss decorator

Example structure:
    from typing import Literal

    from ntt.core.model.loss.base import BaseLoss, LossSpec, register_loss

    class MyLossSpec(LossSpec, frozen=True):
        KIND: Literal["my_loss"] = "my_loss"
        weight: float = 1.0

    @register_loss
    class MyLoss(BaseLoss):
        spec_class = MyLossSpec

        def __init__(self, spec: MyLossSpec):
            super().__init__(spec)

        def compute_loss(self, input_data, reconstruction, target, sample_weights):
            ...

Usage:
    from ntt.core.model.loss import create_loss
    from ntt.contrib.loss import MyLossSpec

    spec = MyLossSpec(weight=2.0)
    loss = create_loss(spec)
"""

import typing as T

from pydantic import Discriminator

from .bce import BCELossSpec
from .dice import DiceLossSpec
from .eval_only_mse import EvalOnlyMSELossSpec
from .focal import FocalLossSpec
from .global_rate import GlobalRateLossSpec
from .huber import HuberLossSpec
from .l1 import L1LossSpec
from .mse import MSELossSpec
from .multi import MultiLossSpec
from .poisson_nll import PoissonNLLLossSpec
from .sigreg import SIGReg
from .wasserstein import WassersteinLossSpec

# Discriminated union of all loss specs
# This allows automatic deserialization based on the "KIND" field
# Usage: LossSpecKind.model_validate({"KIND": "mse", ...})
LossSpecKind = T.Annotated[
    T.Union[  # noqa: UP007
        MSELossSpec,
        PoissonNLLLossSpec,
        EvalOnlyMSELossSpec,
        L1LossSpec,
        HuberLossSpec,
        BCELossSpec,
        DiceLossSpec,
        FocalLossSpec,
        GlobalRateLossSpec,
        WassersteinLossSpec,
        MultiLossSpec,
        None,
    ],
    Discriminator("KIND"),
]

__all__ = [
    "BCELossSpec",
    "DiceLossSpec",
    "EvalOnlyMSELossSpec",
    "FocalLossSpec",
    "GlobalRateLossSpec",
    "HuberLossSpec",
    "L1LossSpec",
    "MSELossSpec",
    "PoissonNLLLossSpec",
    "LossSpecKind",
    "MultiLossSpec",
    "SIGReg",
    "WassersteinLossSpec",
]
