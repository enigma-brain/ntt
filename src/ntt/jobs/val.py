"""Validation job — runs trainer.validate with callbacks."""

import typing as T
from pathlib import Path

from ntt.jobs.base import Locals
from ntt.jobs.lightning import NTTLightningJob
from ntt.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class NTTValJob(
    NTTLightningJob,
    strict=False,
    frozen=False,
    extra="forbid",
    arbitrary_types_allowed=True,
):
    """Validation job — calls ``trainer.validate``.

    Loads a trained model from a checkpoint and runs the validation loop,
    executing any configured callbacks (e.g. visualization).
    """

    KIND: T.Literal["val"] = "val"

    # Required — must point to a trained checkpoint. Pydantic deliberately makes the
    # field required again here, which pyright reads as a missing default.
    ckpt_path: str | Path  # pyright: ignore[reportGeneralTypeIssues]

    def run(self) -> Locals:
        """Run validation."""
        self.trainer.validate(self.model, self.datamodule, ckpt_path=self.ckpt_path)
        return locals()
