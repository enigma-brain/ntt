"""Two-view batch types for NuCLR contrastive learning."""

from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor

from ntt.core.components.types import RatesInput, SpikeTimesInput

# View type is either SpikeTimesInput (spike encoder) or RatesInput (binned encoder)
ViewType = SpikeTimesInput | RatesInput


class TwoViewBatch(NamedTuple):
    """Batch containing two views of the same neural population.

    ``view1`` and ``view2`` are ``SpikeTimesInput`` (spike encoder) or
    ``RatesInput`` (binned encoder) objects (padded, batched).
    ``match_idx`` is a list of ``(2, M_b)`` tensors, one per batch element,
    indicating which unit indices in view1 correspond to which in view2.
    """

    view1: ViewType
    view2: ViewType
    match_idx: list[Tensor]  # list of (2, M_b) per batch element


def compute_match_idx(
    unit_ids_1: np.ndarray,
    unit_ids_2: np.ndarray,
) -> np.ndarray:
    """Find matching unit indices between two views.

    Args:
        unit_ids_1: Unit IDs for view 1 (from ``data.units.unit_number``).
        unit_ids_2: Unit IDs for view 2 (from ``data.units.unit_number``).

    Returns:
        ``(2, M)`` array where row 0 indexes into view1's units and
        row 1 indexes into view2's units.
    """
    match = unit_ids_1[:, None] == unit_ids_2[None, :]
    idx = np.array(np.where(match))  # (2, M)
    return idx


def match_idx_from_tensors(
    unit_ids_1: Tensor,
    mask_1: Tensor,
    unit_ids_2: Tensor,
    mask_2: Tensor,
) -> Tensor:
    """Compute match_idx from batched unit_ids tensors (post-collation).

    For use when match_idx needs to be computed after DataToModelInput
    and collation, e.g., when the dataset doesn't compute it.

    Args:
        unit_ids_1: ``(num_units_1,)`` unit IDs from view 1.
        mask_1: ``(num_units_1,)`` pad mask (1 = real, 0 = pad).
        unit_ids_2: ``(num_units_2,)`` unit IDs from view 2.
        mask_2: ``(num_units_2,)`` pad mask (1 = real, 0 = pad).

    Returns:
        ``(2, M)`` tensor of matching indices.
    """
    real_ids_1 = unit_ids_1[mask_1.bool()]
    real_ids_2 = unit_ids_2[mask_2.bool()]

    match = real_ids_1[:, None] == real_ids_2[None, :]
    idx1, idx2 = torch.where(match)
    return torch.stack([idx1, idx2], dim=0)
