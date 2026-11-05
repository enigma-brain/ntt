"""The data type a tokenizer's reconstruction declares, carried through the eval.

``reconstruct()`` declares what it returns (``ReconstructionOutput.data_type``) and
``reconstruct_from_input(return_latents=True)`` puts it in its output dict under
:data:`RECONSTRUCTION_DATA_TYPE_KEY`. It can differ from both ``spec.input_type``
and ``spec.output_type`` (e.g. a RATES-input tokenizer that samples spikes), so the
eval labels the reconstruction with it rather than with either spec field.
"""

from collections.abc import Callable
from typing import Any

from ntt.core.components.types import DataType, ReconstructionOutput

RECONSTRUCTION_DATA_TYPE_KEY = "reconstruction_data_type"


def declared_reconstruction_data_type(output: Any) -> DataType | None:
    """The reconstruction data type a tokenizer output declares, or None."""
    if isinstance(output, dict):
        return output.get(RECONSTRUCTION_DATA_TYPE_KEY)
    if isinstance(output, ReconstructionOutput):
        return output.data_type
    return None


class ReconstructionTypeRecorder:
    """Wraps a tokenizer function and records the reconstruction type its outputs declare.

    Raises if two calls declare different types: one eval run labels all of its
    reconstructions with a single type.
    """

    def __init__(self, fn: Callable[[Any], Any]):
        self._fn = fn
        self.declared: DataType | None = None

    def __call__(self, batch: Any) -> Any:
        output = self._fn(batch)
        declared = declared_reconstruction_data_type(output)
        if declared is not None:
            if self.declared is not None and declared != self.declared:
                raise ValueError(
                    f"Tokenizer outputs declare different {RECONSTRUCTION_DATA_TYPE_KEY}: "
                    f"{self.declared} and {declared}."
                )
            self.declared = declared
        return output
