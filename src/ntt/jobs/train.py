"""Training job."""

import typing as T
from collections.abc import Iterable

from torch.nn import Parameter
from torch.optim import Optimizer

from ntt.core.model.components.lr_scheduler import LRSchedulerConfig
from ntt.core.model.litmodule import NTTLitModule
from ntt.jobs.base import Locals
from ntt.jobs.lightning import NTTLightningJob
from ntt.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class NTTTrainJob(
    NTTLightningJob,
    strict=False,
    frozen=False,
    extra="forbid",
    arbitrary_types_allowed=True,
):
    """Training job — calls ``trainer.fit``."""

    KIND: T.Literal["train"] = "train"

    optimizer_gen: T.Callable[[Iterable[Parameter]], Optimizer]
    scheduler_config: LRSchedulerConfig | None = None
    viz_every_n_steps: int | None = (
        None  # TODO: Need better way to pass args to litmodule without hardcoding in job class
    )
    disable_val_viz: bool = False

    def _build_model(self) -> NTTLitModule:
        assert self.tokenizer_config is not None
        assert self.loss_config is not None
        return NTTLitModule(
            tokenizer_spec=self.tokenizer_config,
            loss_spec=self.loss_config,
            optimizer_gen=self.optimizer_gen,
            scheduler_config=self.scheduler_config,
            compile_config=self.compile_config,
            viz_every_n_steps=self.viz_every_n_steps,
            disable_val_viz=self.disable_val_viz,
        )

    def run(self) -> Locals:
        """Run training."""
        self.trainer.fit(
            self.model, self.datamodule, ckpt_path=self.ckpt_path, weights_only=self.weights_only
        )
        return locals()
