"""Typing helpers for third-party objects whose attributes are only known at runtime.

``temporaldata.Data`` and its containers (``ArrayDict``, ``IrregularTimeSeries``, ``Interval``,
...) expose their fields (``spikes``, ``units``, ``timestamps``, ...) through dynamic attributes,
so static checkers cannot see them. ``torch_brain``'s ``pad``/``pad2d``/``track_mask*`` return
placeholder objects that only become tensors in the collate step, while the model-input types in
:mod:`ntt.core.components.types` are annotated with the collated ``Tensor`` type.
"""

from typing import Any, cast

from torch import Tensor
from torch_brain.data.collate import pad, pad2d, track_mask, track_mask2d


def dyn(obj: object) -> Any:
    """View ``obj`` through an untyped lens for runtime-defined attributes."""
    return obj


def pad_field(obj: Any) -> Tensor:
    """``torch_brain.data.collate.pad``, typed as the collated tensor."""
    return cast(Tensor, pad(obj))


def pad2d_field(obj: Any) -> Tensor:
    """``torch_brain.data.collate.pad2d``, typed as the collated tensor."""
    return cast(Tensor, pad2d(obj))


def track_mask_field(obj: Any) -> Tensor:
    """``torch_brain.data.collate.track_mask``, typed as the collated tensor."""
    return cast(Tensor, track_mask(obj))


def track_mask2d_field(obj: Any) -> Tensor:
    """``torch_brain.data.collate.track_mask2d``, typed as the collated tensor."""
    return cast(Tensor, track_mask2d(obj))
