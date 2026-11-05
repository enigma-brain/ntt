"""Base class for Lightning Trainer-based jobs (train, validate, etc.)."""

import typing as T
from pathlib import Path

import lightning as L
import pydantic as pdt

from ntt.contrib import LossSpecKind
from ntt.core.model.components.compile import CompileConfig
from ntt.core.model.litmodule import NTTLitModule
from ntt.jobs.ntt_base import NTTBaseJob
from ntt.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class NTTLightningJob(
    NTTBaseJob,
    strict=False,
    frozen=False,
    extra="forbid",
    arbitrary_types_allowed=True,
):
    """Base class for jobs that use a Lightning Trainer.

    Extends ``NTTBaseJob`` with loss config, model creation,
    compilation, and checkpoint loading.  Subclasses implement
    ``run()`` to call the appropriate Trainer method.
    """

    pretrained_ckpt_path: str | Path | None = None
    loss_config: LossSpecKind
    compile_config: CompileConfig | None = None
    watch_freq: int | None = None
    trainer: L.Trainer = pdt.Field(
        json_schema_extra={"type": "object", "description": "Lightning Trainer"}
    )

    _model: NTTLitModule | None = pdt.PrivateAttr(default=None)

    def _build_model(self) -> NTTLitModule:
        """Build the LitModule.

        Override in subclasses to pass additional arguments (e.g. optimizer, scheduler).
        """
        assert self.tokenizer_config is not None
        assert self.loss_config is not None
        return NTTLitModule(
            tokenizer_spec=self.tokenizer_config,
            loss_spec=self.loss_config,
            compile_config=self.compile_config,
        )

    def __enter__(self) -> T.Self:
        """Enter job context — seeds RNG, creates datamodule and model."""
        super().__enter__()

        log.info("Instantiating model...")
        assert self.tokenizer_config is not None
        assert self.loss_config is not None
        self._model = self._build_model()

        if self.pretrained_ckpt_path is not None:
            log.info(f"Loading pretrained weights from: {self.pretrained_ckpt_path}")
            self._load_pretrained_weights(self.pretrained_ckpt_path)

        if self.watch_freq is not None:
            self._setup_wandb_watch()

        return self

    @property
    def model(self) -> NTTLitModule:
        """Get the model instance."""
        if self._model is None:
            raise RuntimeError("Model not instantiated. Use the job within a context manager.")
        return self._model

    def _setup_wandb_watch(self) -> None:
        """Register wandb model watching on any WandbLogger attached to the trainer."""
        from lightning.pytorch.loggers import WandbLogger

        for logger in self.trainer.loggers:
            if isinstance(logger, WandbLogger):
                logger.watch(
                    model=self.model,
                    log="all",
                    log_freq=T.cast(int, self.watch_freq),
                    log_graph=True,
                )

    def _load_pretrained_weights(self, ckpt_path: str | Path) -> None:
        """Load pretrained weights into the model (weight-only, strict=False).

        This is used for multi-stage training (e.g., Stage 2 adapter training of LDNS)
        where some weights come from a previous training run but new modules
        are initialized fresh. The model's `configure_model` is called first
        to ensure the tokenizer and loss modules exist before loading.

        Args:
            ckpt_path: Path to a Lightning checkpoint file.
        """
        import torch

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)

        # Ensure tokenizer is created before loading weights (idempotent)
        self.model.configure_model()

        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            log.info(f"Missing keys in checkpoint (expected for new modules): {missing}")
        if unexpected:
            log.warning(f"Unexpected keys in checkpoint: {unexpected}")
