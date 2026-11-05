"""Training job for NuCLR contrastive learning.

Wires ``TwoViewNTTDataModule`` and ``NuclrContrastiveLitModule`` for
pure contrastive training (no reconstruction).

NuCLR is a standalone model trained inside NTT to support tokenizers (mainly
the Perceiver); it is not itself a tokenizer. The model lives under
``ntt.contrib.nuclr``, but the training job lives here alongside the other jobs
so it can be registered eagerly in ``JobKind`` like every other job (it cannot
extend ``NTTLightningJob``/``NTTBaseJob`` because those require ``tokenizer_config``
and ``loss_config``).

Encoder variant is chosen automatically based on ``data_config.bin_size_ms``:
- ``False`` / ``None`` → spike-based perceiver encoder (cross-attention to raw spike times)
- ``<float>`` → binned encoder (linear read-in of spike count bins, self-attention only)

When spike-based, the pipeline delivers raw spike times (``SpikeTimesInput``).
When binned, the pipeline bins spikes via ``CreateBinnedSpikeCountsArray`` and
delivers non-binarized counts (``RatesInput``); the adapter reshapes to the
encoder's expected ``(U*L, bins_per_latent)`` format.
"""

import typing as T

import lightning as L
import pydantic as pdt

from ntt.contrib.nuclr.data.two_view_datamodule import TwoViewNTTDataModule
from ntt.contrib.nuclr.litmodule import NuclrContrastiveLitModule
from ntt.core.components.types import DataType
from ntt.core.data.datamodule import NTTDataModuleConfig
from ntt.jobs.base import BaseJob, Locals
from ntt.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class NuclrContrastiveTrainJob(
    BaseJob,
    strict=False,
    frozen=False,
    extra="forbid",
    arbitrary_types_allowed=True,
):
    """Training job for NuCLR contrastive model.

    Uses ``TwoViewNTTDataModule`` for paired-view data and
    ``NuclrContrastiveLitModule`` for contrastive-only training.

    Encoder variant is selected by ``data_config.bin_size_ms``:
    - ``False`` → spike-based encoder
    - ``<float>`` (e.g. ``20.0``) → binned encoder with that bin size
    """

    KIND: T.Literal["nuclr_contrastive"] = "nuclr_contrastive"

    seed: int | None = None
    ckpt_path: str | T.Any | None = None
    data_config: NTTDataModuleConfig
    trainer: L.Trainer = pdt.Field(
        json_schema_extra={"type": "object", "description": "Lightning Trainer"}
    )

    # Two-view data params
    max_time_diff: float = 30.0
    val_two_view: bool = True

    # Encoder params
    ctx_duration: float = 10.0
    latent_step: float = 1.0
    dim: int = 256
    cross_heads: int = 4
    self_heads: int = 4
    dim_head: int = 64
    atn_dropout: float = 0.0
    lin_dropout: float = 0.2
    t_layers: int = 2
    st_layers: int = 2
    rot_ratio: float = 0.5

    # Loss params
    tau: float = 0.2
    dcl: bool = True
    full_denom: bool = True
    projector: bool = True

    # Optimizer params
    lr: float = 0.0005
    weight_decay: float = 0.01
    max_epochs: int = 300

    _model: NuclrContrastiveLitModule | None = pdt.PrivateAttr(default=None)
    _datamodule: TwoViewNTTDataModule | None = pdt.PrivateAttr(default=None)

    @property
    def _bin_size_seconds(self) -> float | None:
        """Bin size in seconds, or None for spike-based encoder."""
        bms = self.data_config.bin_size_ms
        if bms is False or bms is None:
            return None
        return float(bms) / 1000.0

    def __enter__(self) -> T.Self:
        """Enter training context — creates TwoView data + NuclrContrastiveLitModule."""
        if self.seed is not None:
            log.info(f"Setting random seed ({self.seed})")
            L.seed_everything(self.seed, workers=True)

        bin_size = self._bin_size_seconds
        if bin_size is not None:
            # Binned encoder: pipeline bins spikes → RatesInput (non-binarized counts)
            self.data_config.input_type = DataType.RATES
            log.info(f"Using binned encoder (bin_size={bin_size:.4f}s)")
        else:
            # Spike encoder: pipeline delivers raw spike times
            self.data_config.input_type = DataType.SPIKE_TIMES
            log.info("Using spike-based encoder")

        log.info("Instantiating TwoViewNTTDataModule...")
        self._datamodule = TwoViewNTTDataModule(
            config=self.data_config,
            tokenizer_spec=None,
            max_time_diff=self.max_time_diff,
            val_two_view=self.val_two_view,
        )

        log.info("Instantiating NuclrContrastiveLitModule...")
        self._model = NuclrContrastiveLitModule(
            ctx_duration=self.ctx_duration,
            latent_step=self.latent_step,
            bin_size=bin_size,
            dim=self.dim,
            cross_heads=self.cross_heads,
            self_heads=self.self_heads,
            dim_head=self.dim_head,
            atn_dropout=self.atn_dropout,
            lin_dropout=self.lin_dropout,
            t_layers=self.t_layers,
            st_layers=self.st_layers,
            rot_ratio=self.rot_ratio,
            tau=self.tau,
            dcl=self.dcl,
            full_denom=self.full_denom,
            projector=self.projector,
            lr=self.lr,
            weight_decay=self.weight_decay,
            max_epochs=self.max_epochs,
        )

        return super().__enter__()

    @property
    def model(self) -> NuclrContrastiveLitModule:
        if self._model is None:
            raise RuntimeError("Model not instantiated. Use within a context manager.")
        return self._model

    @property
    def datamodule(self) -> TwoViewNTTDataModule:
        if self._datamodule is None:
            raise RuntimeError("DataModule not instantiated. Use within a context manager.")
        return self._datamodule

    def run(self) -> Locals:
        """Run contrastive training."""
        self.trainer.fit(self.model, self.datamodule, ckpt_path=self.ckpt_path)
        return locals()
