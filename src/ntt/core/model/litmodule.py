from __future__ import annotations

import logging
import math
import os
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, cast

import lightning as L
import matplotlib.pyplot as plt
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.utilities.parsing import AttributeDict
from lightning.pytorch.utilities.types import LRSchedulerConfigType
from pydantic import BaseModel
from torch import Tensor, nn
from torch.optim import Optimizer
from torchmetrics.regression import PearsonCorrCoef

# Bin sizes (in milliseconds) at which to report Pearson correlation.
# A bin size is skipped when it is finer than the tokenizer's base resolution
# or not an integer multiple of it (mirrors intermediates.py rebinning).
PEARSON_BIN_SIZES_MS: tuple[float, ...] = (1.0, 5.0, 10.0)

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

from ntt.core.components.types import (
    BinnedSpikesInput,
    RatesInput,
    ReconstructionOutput,
    SpikeTimesInput,
    TokenizerInputType,
    TokenizerOutput,
)
from ntt.core.model.components.compile import CompileConfig, compile_model
from ntt.core.model.components.lr_scheduler import LRSchedulerConfig, build_lr_scheduler
from ntt.core.model.loss import BaseLoss, LossSpec, create_loss
from ntt.core.model.tokenizer import BaseTokenizer, TokenizerSpec, create_tokenizer
from ntt.evaluation.core.types import EvalData
from ntt.utils.base_model import loggable_flat_dump


class NTTLitModule(L.LightningModule):
    """Lightning module wrapper for NTT tokenizers.

    This module handles training, validation, and testing of tokenizers.
    It uses the factory pattern to instantiate the appropriate tokenizer
    and loss function from their respective specs.

    The litmodule computes reconstruction loss using the configured LossSpec.
    Tokenizers may also return auxiliary losses (e.g., KL divergence) which
    are added to the total loss.

    Args:
        tokenizer_spec: Configuration for the tokenizer to use.
        loss_spec: Configuration for the reconstruction loss function.
        optimizer_gen: Callable that takes model parameters and returns an optimizer.
        scheduler_config: Optional learning rate scheduler configuration.
        compile_config: Optional configuration for torch.compile.
    """

    def __init__(
        self,
        tokenizer_spec: TokenizerSpec,
        loss_spec: LossSpec,
        optimizer_gen: Callable[[Iterable[nn.Parameter]], Optimizer] | None = None,
        scheduler_config: LRSchedulerConfig | None = None,
        compile_config: CompileConfig | None = None,
        viz_every_n_steps: int | None = None,
        disable_val_viz: bool = False,
    ):
        super().__init__()
        # Store hyperparameters
        self.save_hyperparameters()
        # Tokenizer and loss will be created in configure_model for efficient initialization
        self.tokenizer: nn.Module | None = None
        self.loss_fn: BaseLoss | None = None
        # Store hyperparams
        self.tokenizer_spec = tokenizer_spec
        self.loss_spec = loss_spec
        self.optimizer_gen = optimizer_gen
        self.scheduler_config = scheduler_config
        self.compile_config = compile_config
        self.viz_every_n_steps = viz_every_n_steps
        self.disable_val_viz = disable_val_viz

        # Pearson correlation metrics, one per target bin size (in ms).
        # See ``_update_pearson`` for the rebinning logic.
        self.train_pearson = nn.ModuleDict(
            {
                self._bin_key(bs): PearsonCorrCoef(sync_on_compute=False)
                for bs in PEARSON_BIN_SIZES_MS
            }
        )
        self.val_pearson = nn.ModuleDict(
            {
                self._bin_key(bs): PearsonCorrCoef(sync_on_compute=False)
                for bs in PEARSON_BIN_SIZES_MS
            }
        )

    @property
    def hparams_initial(self) -> AttributeDict:  # type: ignore[override]
        """Hyperparameters as seen by the loggers, one param per spec field.

        Lightning logs ``hparams_initial`` (``loggers/utilities.py::_log_hyperparams``)
        but checkpoints ``hparams`` (``checkpoint_connector.py::dump_checkpoint``).
        Expanding the specs here therefore makes every knob individually
        queryable in MLflow/W&B without changing what ``load_from_checkpoint``
        reconstructs the module from. See :func:`loggable_flat_dump` for why the
        default behaviour clips the specs to 250 characters.
        """
        params = dict(super().hparams_initial)
        for key in ("tokenizer_spec", "loss_spec", "scheduler_config", "compile_config"):
            value = params.get(key)
            if isinstance(value, BaseModel):
                params[key] = loggable_flat_dump(value)
        return AttributeDict(params)

    @staticmethod
    def _bin_key(bin_size_ms: float) -> str:
        """Stable ModuleDict/log-tag key for a bin size in ms (e.g. 1.0 -> '1ms')."""
        if float(bin_size_ms).is_integer():
            return f"{int(bin_size_ms)}ms"
        return f"{str(bin_size_ms).replace('.', '_')}ms"

    def configure_model(self):
        """Configure model for efficient initialization.

        This method enables efficient initialization of model weights for
        distributed training (e.g., FSDP). See Lightning docs:
        https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/fsdp.html#speed-up-model-initialization

        The tokenizer and loss function are created here rather than in __init__
        to support lazy initialization patterns required by some distributed strategies.

        Note: Spec migration for checkpoints that lack current spec fields is
        handled automatically by NTTBaseModel.__setstate__ during unpickling.
        """
        # Initialize tokenizer if not already done
        if self.tokenizer is not None:
            return

        # Create tokenizer from spec
        self.tokenizer = create_tokenizer(self.tokenizer_spec)

        # Compile model if config provided
        if self.compile_config is not None:
            self.tokenizer = compile_model(self.tokenizer, self.compile_config)

        # Create loss function from spec
        self.loss_fn = create_loss(self.loss_spec)

        # Validate loss ↔ tokenizer compatibility
        self._validate_loss_tokenizer_compat()

    def _validate_loss_tokenizer_compat(self) -> None:
        """Check that the loss function is compatible with the tokenizer output."""
        supported = self.loss_spec.supported_outputs
        output_type = self.tokenizer_spec.output_type
        if supported is not None and len(supported) > 0 and output_type not in supported:
            log.warning(
                f"Loss '{self.loss_spec.KIND}' supports outputs {supported}, "
                f"but tokenizer output_type is '{output_type}'. "
                f"This may be fine if the reconstruction lives in a different "
                f"domain than the post-sampling output (e.g. rates vs spikes)."
            )

    def forward(self, x: SpikeTimesInput | BinnedSpikesInput) -> TokenizerOutput:
        """Forward pass through tokenizer.

        Args:
            x: Input neural data

        Returns:
            TokenizerOutput containing tokens, reconstruction, and optional aux_loss
        """
        assert self.tokenizer is not None, "Call `configure_model` first!"
        return self.tokenizer(x)

    def training_step(self, batch: SpikeTimesInput | BinnedSpikesInput, batch_idx: int):
        """Training step.

        Args:
            batch: Training batch from datamodule
            batch_idx: Batch index

        Returns:
            Loss tensor
        """
        assert self.tokenizer is not None, "Call `configure_model` first!"
        assert self.loss_fn is not None, "Call `configure_model` first!"

        # Forward pass through tokenizer. A tokenizer may return either a
        # `TokenizerOutput` or a `(modified_batch, TokenizerOutput)` tuple to
        # replace the litmodule's local batch reference for the downstream
        # loss / metric path (see BaseTokenizer.forward docstring) — used
        # e.g. by WanNeuralTokenizer's tile-batched curvature path to swap a
        # 1s batch with a B*n_tiles tile-batch so shapes match the per-tile
        # reconstruction.
        result = self.tokenizer(batch)
        if not isinstance(result, TokenizerOutput):
            batch, output = result
        else:
            output = result

        # Use the tokenizer's effective mask if it modified positions during
        # training (e.g. neuron dropout), otherwise fall back to the batch mask.
        effective_mask = cast(BaseTokenizer, self.tokenizer).get_effective_pad_mask()
        if effective_mask is not None:
            pad_mask = effective_mask
        elif isinstance(batch, SpikeTimesInput):
            pad_mask = batch.spike_pad_mask
        else:
            pad_mask = batch.pad_and_gap_mask

        if output.precomputed_recon_loss is not None:
            recon_loss = output.precomputed_recon_loss
        else:
            recon_loss = self.loss_fn(batch, output, pad_mask)
            # Per-component losses are only valid for the path that actually
            # ran loss_fn — skip when the tokenizer supplied a precomputed
            # loss, otherwise get_component_losses() returns stale values
            # from whichever batch last invoked loss_fn.
            component_losses = self.loss_fn.get_component_losses()
            if component_losses is not None:
                for name, val in component_losses.items():
                    if val.ndim == 0:
                        self.log(f"train/loss_{name}", val, prog_bar=False)

        # Total loss = reconstruction loss + auxiliary losses (if any)
        total_loss = recon_loss
        if output.aux_loss is not None:
            if isinstance(output.aux_loss, dict):
                aux_total = sum(output.aux_loss.values())
                for name, value in output.aux_loss.items():
                    self.log(f"train/aux/{name}", value, prog_bar=False)
            else:
                aux_total = output.aux_loss
            total_loss = total_loss + aux_total
            self.log("train/aux_loss", aux_total, prog_bar=False)
        if output.aux_losses:
            for name, value in output.aux_losses.items():
                self.log(f"train/aux/{name}", value, prog_bar=False)
        self._log_sigreg_metrics("train")

        # Log metrics
        self.log("train/recon_loss", recon_loss, prog_bar=True)
        self.log("train/loss", total_loss, prog_bar=True)

        # Update and log Pearson correlation at each feasible bin size
        updated_keys = self._update_pearson(self.train_pearson, batch, output)
        for key in updated_keys:
            metric = self.train_pearson[key]
            assert isinstance(metric, PearsonCorrCoef)
            self.log(f"train/pearson_{key}", metric.compute(), prog_bar=False)

        # Visualize raster at configured frequency
        if self.viz_every_n_steps is not None and self.global_step % self.viz_every_n_steps == 0:
            self._visualize_sample(batch, output, "train")

        return total_loss

    def validation_step(self, batch: SpikeTimesInput | BinnedSpikesInput, batch_idx: int):
        """Validation step.

        Args:
            batch: Validation batch from datamodule
            batch_idx: Batch index
        """
        assert self.tokenizer is not None, "Call `configure_model` first!"
        assert self.loss_fn is not None, "Call `configure_model` first!"

        # Forward pass through tokenizer. See training_step for the
        # (modified_batch, output) tuple-return convention.
        result = self.tokenizer(batch)
        if not isinstance(result, TokenizerOutput):
            batch, output = result
        else:
            output = result

        # Use the tokenizer's effective mask if it modified positions during
        # training (e.g. neuron dropout), otherwise fall back to the batch mask.
        effective_mask = cast(BaseTokenizer, self.tokenizer).get_effective_pad_mask()
        if effective_mask is not None:
            pad_mask = effective_mask
        elif isinstance(batch, SpikeTimesInput):
            pad_mask = batch.spike_pad_mask
        else:
            pad_mask = batch.pad_and_gap_mask

        if output.precomputed_recon_loss is not None:
            recon_loss = output.precomputed_recon_loss
        else:
            recon_loss = self.loss_fn(batch, output, pad_mask)
            component_losses = self.loss_fn.get_component_losses()
            if component_losses is not None:
                for name, val in component_losses.items():
                    if val.ndim == 0:
                        self.log(f"val/loss_{name}", val, prog_bar=False)

        # Total loss = reconstruction loss + auxiliary losses (if any)
        total_loss = recon_loss
        if output.aux_loss is not None:
            if isinstance(output.aux_loss, dict):
                aux_total = sum(output.aux_loss.values())
                for name, value in output.aux_loss.items():
                    self.log(f"val/aux/{name}", value, prog_bar=False)
            else:
                aux_total = output.aux_loss
            total_loss = total_loss + aux_total
            self.log("val/aux_loss", aux_total, prog_bar=False)
        if output.aux_losses:
            for name, value in output.aux_losses.items():
                self.log(f"val/aux/{name}", value, prog_bar=False)
        self._log_sigreg_metrics("val")

        # Log metrics
        self.log("val/recon_loss", recon_loss, prog_bar=True)
        self.log("val/loss", total_loss, prog_bar=True)

        # Update and log Pearson correlation at each feasible bin size
        updated_keys = self._update_pearson(self.val_pearson, batch, output)
        for key in updated_keys:
            metric = self.val_pearson[key]
            assert isinstance(metric, PearsonCorrCoef)
            self.log(f"val/pearson_{key}", metric.compute(), prog_bar=False)

        # Visualize first batch of each validation loop
        if self.viz_every_n_steps is not None and not self.disable_val_viz and batch_idx == 0:
            self._visualize_sample(batch, output, "val")

    def _log_sigreg_metrics(self, stage: str) -> None:
        sigreg_loss = getattr(self.tokenizer, "_latest_sigreg_loss", None)
        if sigreg_loss is None:
            return
        self.log(f"{stage}/sigreg_loss", sigreg_loss, prog_bar=False)
        sigreg_full_loss = getattr(self.tokenizer, "_latest_sigreg_full_loss", None)
        if sigreg_full_loss is not None:
            self.log(f"{stage}/sigreg_full_loss", sigreg_full_loss, prog_bar=False)

    def _update_pearson(
        self,
        metrics: nn.ModuleDict,
        batch: TokenizerInputType,
        output: TokenizerOutput,
    ) -> list[str]:
        """Update per-bin-size PearsonCorrCoef metrics from batch input and output.

        For each bin size in ``PEARSON_BIN_SIZES_MS`` that is a non-strict integer
        multiple of the tokenizer's base bin size (``1 / sampling_rate``), rebins
        the prediction, target, and pad mask along the time axis and updates the
        corresponding metric. Bin sizes finer than the base resolution, or not an
        integer multiple of it, are silently skipped (mirrors
        ``intermediates._compute_binned_counts_from_binned``).

        The rebinned mask is ``True`` only where *all* sub-bins are valid, so
        padded/gapped samples are never folded into valid bins.

        Args:
            metrics: ModuleDict of PearsonCorrCoef keyed by ``_bin_key``.
            batch: Model input (BinnedSpikesInput or RatesInput).
            output: TokenizerOutput from the forward pass.

        Returns:
            List of ``_bin_key`` values for metrics that were updated this step.
        """
        if isinstance(batch, SpikeTimesInput):
            raise NotImplementedError(
                "Pearson correlation is not implemented for SpikeTimesInput."
            )

        # Both BinnedSpikesInput and RatesInput have pad_mask and a 2D matrix
        if isinstance(batch, BinnedSpikesInput):
            target = batch.spike_matrix.float()
        elif isinstance(batch, RatesInput):
            target = batch.rate_matrix.float()
        else:
            raise TypeError(f"Unsupported input type: {type(batch)}")

        mask = batch.pad_and_gap_mask.bool()
        pred = output.reconstruction
        assert pred is not None, "reconstruction is None"

        # After batching, ``sampling_rate`` is a Tensor of shape (batch,); samples
        # within a batch share a sampling rate (see _visualize_sample for a prior
        # use of this assumption). The NamedTuple annotation declares it ``float``,
        # which is true only at the pre-collate stage; runtime check via isinstance.
        sampling_rate: float | Tensor = batch.sampling_rate
        if isinstance(sampling_rate, Tensor):
            sampling_rate = float(sampling_rate.flatten()[0].item())  # pyright: ignore[reportAttributeAccessIssue]
        base_bin_size_ms = 1000.0 / float(sampling_rate)

        updated: list[str] = []
        batch_size, n_time, n_units = pred.shape
        for bin_size_ms in PEARSON_BIN_SIZES_MS:
            # Skip bin sizes finer than base resolution
            if bin_size_ms + 1e-9 < base_bin_size_ms:
                continue
            ratio = bin_size_ms / base_bin_size_ms
            ratio_int = int(round(ratio))
            if not math.isclose(ratio, ratio_int, abs_tol=1e-9) or ratio_int < 1:
                continue

            if ratio_int == 1:
                p, t, m = pred, target, mask
            else:
                n_bins = n_time // ratio_int
                if n_bins == 0:
                    continue
                trunc = n_bins * ratio_int
                p = pred[:, :trunc].reshape(batch_size, n_bins, ratio_int, n_units).sum(dim=2)
                t = target[:, :trunc].reshape(batch_size, n_bins, ratio_int, n_units).sum(dim=2)
                # A rebinned sample is valid only if every sub-bin was valid,
                # so padding never contaminates aggregated bins.
                m = mask[:, :trunc].reshape(batch_size, n_bins, ratio_int, n_units).all(dim=2)

            if not m.any():
                continue

            key = self._bin_key(bin_size_ms)
            metric = metrics[key]
            assert isinstance(metric, PearsonCorrCoef)
            # Cast pred to float32: bf16-mixed training can produce
            # low-precision values that collapse variance and trigger
            # spurious NaN inside PearsonCorrCoef. Target is already float
            # (cast at the top of this function).
            metric.update(p[m].float(), t[m])
            updated.append(key)

        return updated

    @torch.no_grad()
    def _visualize_sample(
        self,
        batch: TokenizerInputType,
        output: TokenizerOutput,
        stage: str,
    ) -> None:
        """Plot a population raster for the first sample in a batch and log it.

        Uses ``reconstruct_from_input`` (encode → reconstruct with
        inference-time post-processing) so the reconstruction goes through
        the same pipeline as evaluation — including Poisson sampling for
        rate-output tokenizers.

        Args:
            batch: Model input.
            output: TokenizerOutput from the forward pass (unused;
                reconstruction is obtained via ``reconstruct_from_input``
                instead).
            stage: "train" or "val", used as the logging tag prefix.

        Raises:
            NotImplementedError: If batch is SpikeTimesInput.
        """
        from ntt.evaluation.plotting import plot_population_raster
        from ntt.evaluation.utils.validation_collation import CollatedValidationSegment

        if isinstance(batch, SpikeTimesInput):
            raise NotImplementedError(
                "Raster visualization is not implemented for SpikeTimesInput."
            )

        # Trim trailing padding from VariableWindowCrop so the raster doesn't
        # show zero-input garbage after the real sample ends. Mid-sample gap
        # bins are preserved (they're already zero in spike_matrix).
        if batch.pad_and_gap_mask is not None:
            mask_T = batch.pad_and_gap_mask[0].any(dim=-1)  # [T] bool
            if mask_T.any():
                valid_T = int(torch.nonzero(mask_T).max().item()) + 1
            else:
                valid_T = int(mask_T.shape[0])
        else:
            matrix = (
                batch.spike_matrix if isinstance(batch, BinnedSpikesInput) else batch.rate_matrix
            )
            valid_T = int(matrix.shape[1])

        if isinstance(batch, BinnedSpikesInput):
            # (batch, time, units) -> (units, time) for first sample
            original = batch.spike_matrix[0, :valid_T].T.float().detach().cpu()
        elif isinstance(batch, RatesInput):
            original = batch.rate_matrix[0, :valid_T].T.float().detach().cpu()
        else:
            raise TypeError(f"Unsupported input type: {type(batch)}")

        # Always route the plotted reconstruction through the tokenizer's
        # spike-generation criterion (output activation + `reconstruct_sample`),
        # so the raster shows what the model actually emits.
        #
        # This applies to every tokenizer, including continuous-bottleneck ones
        # (vae / mmd_vae / diffusion_prior, and the PCA baseline) that set
        # `tokens = None`. Plotting `output.reconstruction` verbatim would show
        # post-sigmoid probabilities in (0, 1), and since plot_population_raster
        # treats any value > 0 as a spike, EVERY bin would render as a spike
        # and the raster would be a solid block.
        #
        # `reconstruct_from_output` is preferred over `reconstruct_from_input`:
        # it binarises exactly the tensor the loss saw, whereas
        # reconstruct_from_input runs a fresh encode+decode that re-samples
        # attention drop-path and the VAE reparameterisation, so the plotted
        # reconstruction would not be the one that produced the logged loss.
        assert self.tokenizer is not None, "Call `configure_model` first!"
        recon_output = None
        tokenizer: Any = self.tokenizer  # reconstruction entry points vary by tokenizer
        with torch.no_grad():
            if hasattr(tokenizer, "reconstruct_from_output"):
                recon_output = tokenizer.reconstruct_from_output(output)
            elif hasattr(tokenizer, "reconstruct_from_input"):
                recon_output = tokenizer.reconstruct_from_input(batch)
            elif output.tokens is not None:
                recon_output = tokenizer.reconstruct(output.tokens)
        if recon_output is not None:
            reconstructed = recon_output.reconstruction[0, :valid_T].T.float().detach().cpu()
        else:
            # Reachable for any tokenizer that exposes none of the reconstruction
            # entry points above and leaves `reconstruction` unset, so this is a
            # runtime configuration error rather than an internal invariant.
            if output.reconstruction is None:
                raise ValueError(
                    f"Cannot visualise a sample: {type(self.tokenizer).__name__} exposes "
                    "neither `reconstruct_from_output` nor `reconstruct_from_input`, "
                    "produced no tokens to `reconstruct` from, and its output carries no "
                    "`reconstruction`."
                )
            reconstructed = output.reconstruction[0, :valid_T].T.float().detach().cpu()
        pad_mask = (
            batch.pad_and_gap_mask[0, :valid_T].T.detach().cpu()
            if batch.pad_and_gap_mask is not None
            else None
        )

        session_id = batch.session_id if isinstance(batch.session_id, str) else batch.session_id[0]
        bin_size_seconds = (1.0 / batch.sampling_rate[0]).item()  # type: ignore[index]  # after batching, sampling_rate is a Tensor

        segment = CollatedValidationSegment(
            session_id=session_id,
            original=original,
            reconstructed=reconstructed,
            pad_and_gap_mask=pad_mask,
            metadata={},
            segment_index=0,
            is_partial=False,
        )

        fig = plot_population_raster(segment, bin_size_seconds)
        self._log_figure(fig, f"{stage}/raster")
        plt.close(fig)

    def _log_figure(self, fig: Figure, tag: str) -> None:
        """Log a matplotlib figure to the active logger.

        Supports WandB (via logger.log_image), TensorBoard (via
        experiment.add_figure), and MLflow (via experiment.log_figure).
        Silently skips if the logger doesn't support figure logging.
        """
        # Typed as Any: the supported loggers expose different, logger-specific APIs.
        logger: Any = self.logger
        if logger is None:
            return
        # WandB: WandbLogger has log_image on the logger itself
        if hasattr(logger, "log_image"):
            logger.log_image(key=tag, images=[fig], step=self.global_step)
        # TensorBoard: SummaryWriter has add_figure
        elif hasattr(logger.experiment, "add_figure"):
            logger.experiment.add_figure(tag, fig, global_step=self.global_step)
        # MLflow: MlflowClient has log_figure
        elif hasattr(logger.experiment, "log_figure"):
            logger.experiment.log_figure(
                logger.run_id,
                fig,
                f"{tag}_step_{self.global_step}.png",
            )

    def on_train_epoch_end(self) -> None:
        for metric in self.train_pearson.values():
            assert isinstance(metric, PearsonCorrCoef)
            metric.reset()
        # Allow fit-once-then-frozen tokenizers (e.g. PCA baseline) to fit
        # at epoch boundary, after seeing the full training set.
        tokenizer: Any = self.tokenizer  # only some tokenizers define fit_from_running_stats
        if tokenizer is not None and hasattr(tokenizer, "fit_from_running_stats"):
            if tokenizer.fit_from_running_stats():
                self._resave_last_checkpoints()

    def _resave_last_checkpoints(self) -> None:
        """Rewrite each ModelCheckpoint's last.ckpt after an epoch-end fit.

        Lightning runs callback hooks before this module's ``on_train_epoch_end``, so
        ModelCheckpoint has already written ``last.ckpt`` (on the epoch's last step, or at
        epoch end) from the *unfitted* model. Without this, a 1-epoch run ends with every
        saved checkpoint unfitted and the fit only in memory.
        """
        for callback in self.trainer.checkpoint_callbacks:
            if not isinstance(callback, ModelCheckpoint):
                continue
            if not callback.save_last or not callback.dirpath:
                continue
            last = callback.last_model_path or os.path.join(
                callback.dirpath, callback.CHECKPOINT_NAME_LAST + callback.FILE_EXTENSION
            )
            self.trainer.save_checkpoint(last)

    def on_validation_epoch_end(self) -> None:
        for metric in self.val_pearson.values():
            assert isinstance(metric, PearsonCorrCoef)
            metric.reset()

    def configure_optimizers(
        self,
    ) -> Optimizer | dict[str, Optimizer | LRSchedulerConfigType]:
        """Configure optimizers.

        Returns:
            Optimizer or tuple of (optimizer, scheduler)
        """
        assert self.tokenizer is not None, "Call `configure_model` first!"
        assert self.optimizer_gen is not None, "optimizer_gen must be set!"
        optimizer = self.optimizer_gen(self.tokenizer.parameters())
        scheduler = None
        if self.scheduler_config is not None:
            train_steps = int(self.trainer.estimated_stepping_batches)
            scheduler = build_lr_scheduler(optimizer, self.scheduler_config, train_steps)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer

    def get_eval_data(self, batch: BinnedSpikesInput) -> tuple[EvalData, EvalData]:
        """Get EvalData for evaluation callback.

        Uses ``reconstruct_from_input`` (encode → reconstruct) which applies
        proper inference-time post-processing (e.g. Poisson/Bernoulli/ZIP
        sampling) and returns a ``ReconstructionOutput`` with correct metadata.

        Args:
            batch: Input batch from dataloader (BinnedSpikesInput)

        Returns:
            Tuple of (original, reconstructed) EvalData objects
        """
        # Import here to avoid circular imports
        from ntt.evaluation.core.types import EvalData

        assert self.tokenizer is not None, "Call `configure_model` first!"

        # return_latents=False (default) yields a ReconstructionOutput, not the latents dict
        result = cast(
            ReconstructionOutput, cast(BaseTokenizer, self.tokenizer).reconstruct_from_input(batch)
        )

        original = EvalData.from_binned_input(batch, batch_index=0)

        # ReconstructionOutput.reconstruction: (batch, time, units)
        # EvalData expects (units, time)
        reconstructed = EvalData.from_tensor(
            tensor=result.reconstruction[0].T,
            data_type=result.data_type,
            bin_size_seconds=result.bin_size_seconds,
            duration=self.tokenizer_spec.effective_window_length_seconds,
        )
        return original, reconstructed
