"""PyTorch Lightning integration for evaluation."""

from collections.abc import Callable
from itertools import islice, zip_longest
from typing import Any, cast

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback

from ntt.core.components.types import BinnedSpikesInput, RatesInput
from ntt.core.data.components.config import SelectionConfig
from ntt.core.data.datamodule import NTTDataModule
from ntt.utils import RankedLogger

from .config.schema import AggregationStrategy, EvaluationConfig, LightningEvalConfig
from .core.types import (  # MetricCategory used for report aggregation
    DataType,
    EvalData,
    EvaluationReport,
    MetricCategory,
)
from .pipeline import EvaluationPipeline
from .utils import (
    aggregate_segment_reports,
    iter_collated_validation_segments,
    run_validation_evaluation_per_session,
)

log = RankedLogger(__name__, rank_zero_only=True)


def _aggregate_pre_aggregated(
    per_session_agg: dict[str, tuple[EvaluationReport, float]],
    strategy: AggregationStrategy,
) -> EvaluationReport:
    """Aggregate pre-aggregated per-session (report, weight) pairs across sessions.

    This is the second level of the two-level aggregation: each session has already
    been reduced from segments to a single EvaluationReport with a duration weight.
    """
    from .utils.validation_collation import _aggregate_evaluation_reports

    reports = [r for r, _ in per_session_agg.values()]
    weights = [w for _, w in per_session_agg.values()]
    return _aggregate_evaluation_reports(reports, weights, strategy)


def _report_to_metrics(report: EvaluationReport, prefix: str) -> dict[str, float]:
    """Convert an EvaluationReport into scalar logger metrics."""
    metrics_to_log: dict[str, float] = {}
    for name, result in report.results.items():
        value = result.raw_value
        if hasattr(value, "__len__"):
            if len(value) == 0:  # type: ignore[arg-type]
                continue
            value = np.nanmean(value)
        if np.isscalar(value) and not np.isnan(value):
            metrics_to_log[f"{prefix}/{name}"] = float(cast(Any, value))
    return metrics_to_log


def _metadata_sampling_rate(metadata: dict[str, Any]) -> float | None:
    """Extract a scalar sampling rate from segment metadata."""
    rate = metadata.get("sampling_rate")
    if rate is None:
        return None
    if isinstance(rate, torch.Tensor):
        flat = rate.flatten()
        if flat.numel() == 0:
            return None
        return float(flat[0].item())
    return float(rate)


class BehavioralDatasetEvalCallback(Callback):
    """Run latent-aware evaluation on an explicit validation-style subset.

    The subset is defined by explicit selections and loaded through a dedicated
    datamodule, so the evaluated sessions are guaranteed by construction rather
    than filtered from the generic validation dataloader.

    In DDP, sessions are distributed across ranks (round-robin).  Each rank
    runs inference + metrics on its assigned sessions, then per-session reports
    are gathered to rank 0 for final aggregation and logging.
    """

    def __init__(
        self,
        eval_config: EvaluationConfig | None = None,
        selections: list[SelectionConfig] | None = None,
        log_prefix: str = "behavioral_dataset_eval",
        max_batches: int | None = None,
        max_windows_per_session: int | None = None,
        max_sessions: int | None = None,
        aggregation_strategy: AggregationStrategy | str = AggregationStrategy.MEAN,
        eval_on_epoch_end: bool = True,
        every_n_epochs: int = 1,
        run_on_fit_end: bool = True,
        compute_poisson_baseline: bool = True,
    ):
        """
        Notes on caps:
            ``max_batches`` truncates each rank's dataloader via ``islice`` — a
            per-rank cap on total batches fetched. On DDP each rank applies it
            to its own sharded datamodule, so per-session coverage scales with
            world_size (a session on a 4-GPU run sees ~4x the windows it would
            on 1 GPU at the same ``max_batches``). Use it to bound runtime.

            ``max_windows_per_session`` is passed through to the segment
            collator as a *segment flush threshold*, not a total cap: a session
            with N windows and ``max_windows_per_session=k`` is chopped into
            ~N/k segments, each scored independently and then averaged. Use it
            to bound per-segment memory. Unset in all current configs.
        """
        super().__init__()
        if not selections:
            raise ValueError("BehavioralDatasetEvalCallback requires explicit selections.")
        if every_n_epochs < 1:
            raise ValueError("every_n_epochs must be >= 1")
        if max_batches is not None and max_batches < 1:
            raise ValueError("max_batches must be >= 1 when provided")
        if max_sessions is not None and max_sessions < 1:
            raise ValueError("max_sessions must be >= 1 when provided")
        if isinstance(aggregation_strategy, str):
            aggregation_strategy = AggregationStrategy(aggregation_strategy)

        self.eval_config = eval_config or EvaluationConfig()
        self.selections = selections
        self.log_prefix = log_prefix
        self.max_batches = max_batches
        self.max_windows_per_session = max_windows_per_session
        self.max_sessions = max_sessions
        if isinstance(aggregation_strategy, str):  # Hydra passes raw string from YAML
            aggregation_strategy = AggregationStrategy(aggregation_strategy)
        self.aggregation_strategy = aggregation_strategy
        self.eval_on_epoch_end = eval_on_epoch_end
        self.every_n_epochs = every_n_epochs
        self.run_on_fit_end = run_on_fit_end
        self.compute_poisson_baseline = compute_poisson_baseline

        self._cached_datamodule: NTTDataModule | None = None
        self._poisson_baseline_metrics: dict[str, float] | None = None

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Run explicit-subset evaluation at validation epoch end."""
        if not self.eval_on_epoch_end:
            return
        current_epoch = trainer.current_epoch
        if (current_epoch + 1) % self.every_n_epochs != 0:
            return
        self._run_eval(trainer, pl_module, prefix=self.log_prefix)

    def on_fit_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Run explicit-subset evaluation at training end."""
        if not self.run_on_fit_end:
            return
        self._run_eval(trainer, pl_module, prefix=f"{self.log_prefix}_final")

    # ── Helpers ──────────────────────────────────────────────────────

    def _get_or_build_datamodule(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        selections: list[SelectionConfig],
    ) -> NTTDataModule:
        """Build and cache a dedicated datamodule for the given selections.

        Clones the trainer's base datamodule config, overriding val_selections.
        The val/test split path in NTTDataModule already forces deterministic
        unit dropout, so no extra flag is needed here. The result is cached so
        that subsequent epochs reuse the same datamodule without rebuilding.
        """
        if self._cached_datamodule is not None:
            return self._cached_datamodule

        base_datamodule = getattr(trainer, "datamodule", None)
        if not isinstance(base_datamodule, NTTDataModule):
            raise TypeError(
                "BehavioralDatasetEvalCallback requires trainer.datamodule to be an NTTDataModule."
            )

        dataloader_cfg = base_datamodule.config.dataloader.model_copy(update={"drop_last": False})
        eval_config = base_datamodule.config.model_copy(
            deep=True,
            update={
                "val_selections": selections,
                "use_train_for_val": False,
                "dataloader": dataloader_cfg,
            },
        )

        dm = NTTDataModule(
            config=eval_config,
            tokenizer_spec=getattr(pl_module, "tokenizer_spec", None),
        )
        dm.setup("validate")
        self._cached_datamodule = dm
        return dm

    @staticmethod
    def _extract_session_ids(selections: list[SelectionConfig]) -> list[str]:
        """Collect session IDs, interleaved across selection groups.

        Each SelectionConfig typically corresponds to one subject/brainset.
        Round-robin interleaving ensures that ``max_sessions`` picks from
        all subjects rather than exhausting one before starting the next.
        """
        groups: list[list[str]] = []
        for sel in selections:
            group: list[str] = []
            for bs in sel.selection:
                if bs.sessions:
                    group.extend(bs.sessions)
            if group:
                groups.append(group)
        # Round-robin across groups so max_sessions picks from all subjects
        return list(
            dict.fromkeys(s for chunk in zip_longest(*groups) for s in chunk if s is not None)
        )

    @staticmethod
    def _filter_selections_to_sessions(
        selections: list[SelectionConfig],
        keep: set[str],
    ) -> list[SelectionConfig]:
        """Return a copy of *selections* containing only sessions in *keep*."""
        from ntt.core.data.components.config import BrainsetSelection

        filtered: list[SelectionConfig] = []
        for sel in selections:
            new_bs_list: list[BrainsetSelection] = []
            for bs in sel.selection:
                if bs.sessions:
                    kept = [s for s in bs.sessions if s in keep]
                    if kept:
                        new_bs_list.append(BrainsetSelection(brainset=bs.brainset, sessions=kept))
                else:
                    new_bs_list.append(bs)
            if new_bs_list:
                filtered.append(SelectionConfig(selection=new_bs_list, config=sel.config))
        return filtered

    @staticmethod
    def _gather_and_merge(
        local_reports: dict,
        world_size: int,
    ) -> dict:
        """All-gather per-rank report dicts and merge into one."""
        import torch.distributed as dist

        gathered: list[dict | None] = [None] * world_size
        dist.all_gather_object(gathered, local_reports)
        merged: dict = {}
        for r in gathered:
            if r:
                merged.update(r)
        return merged

    def _run_per_session(
        self,
        datamodule: NTTDataModule,
        tokenizer_fn: Callable[[Any], dict[str, Any]],
        spec: Any,
        bin_size_seconds: float | None,
        token_bin_size_ms: float | None,
    ) -> dict:
        """Run tokenizer_fn over the datamodule's val dataloader.

        Batches are capped by ``max_batches`` and collated into per-session
        windows (capped by ``max_windows_per_session``).  Returns a dict
        mapping session IDs to lists of (segment, report) tuples.
        """
        dataloader: Any = datamodule.val_dataloader()
        if self.max_batches is not None:
            dataloader = islice(dataloader, self.max_batches)

        return run_validation_evaluation_per_session(
            dataloader,
            tokenizer_fn,
            data_type=spec.input_type,
            reconstructed_data_type=getattr(spec, "output_type", None),
            bin_size_seconds=bin_size_seconds,
            token_bin_size_ms=token_bin_size_ms,
            window_length_seconds=getattr(spec, "window_length_seconds", None),
            eval_config=self.eval_config,
            max_windows=self.max_windows_per_session,
            device=None,
            progress=False,
        )

    # ── Main eval ────────────────────────────────────────────────────

    def _run_eval(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        prefix: str,
    ) -> None:
        """Run latent-aware evaluation on the explicit subset.

        DDP barriers bracket the entire call so that rank desync from prior
        callbacks or from asymmetric eval times cannot cause NCCL timeouts.
        """
        if not hasattr(pl_module, "tokenizer_spec"):
            log.warning("Module has no tokenizer_spec, skipping dataset-specific evaluation")
            return

        import torch.distributed as dist

        is_distributed = dist.is_available() and dist.is_initialized()

        # Barrier at entry: prior callbacks (e.g. TokenizerEvalCallback) may
        # take asymmetric time across ranks.  Without this, the fastest rank
        # enters our all_gather while the slowest is still in a prior callback,
        # causing an NCCL timeout.
        if is_distributed:
            trainer.strategy.barrier("behavioral_eval_entry")

        try:
            self._run_eval_inner(trainer, pl_module, prefix, is_distributed)
        finally:
            if is_distributed:
                trainer.strategy.barrier("behavioral_eval_done")

    def _run_eval_inner(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        prefix: str,
        is_distributed: bool,
    ) -> None:
        """Core evaluation logic.

        1. Shard sessions across ranks (round-robin; single-GPU is trivially
           world_size=1, rank=0).
        2. Each rank builds a datamodule for its sessions and runs inference.
        3. Optionally computes Poisson baseline on the same sessions.
        4. All-gathers reports to rank 0 for aggregation and logging.

        Local eval is wrapped in try/except so that a failing rank still
        participates in the DDP collectives with empty data, preventing
        NCCL deadlocks.
        """
        import torch.distributed as dist

        spec: Any = pl_module.tokenizer_spec
        bin_size_seconds = spec.bin_size_ms / 1000.0 if spec.bin_size_ms else None
        token_bin_size_ms = getattr(spec, "token_bin_size_ms", None)
        world_size = dist.get_world_size() if is_distributed else 1
        rank = dist.get_rank() if is_distributed else 0

        # ── Determine this rank's sessions (works for single-GPU too) ─
        all_sessions = self._extract_session_ids(self.selections)
        if self.max_sessions is not None:
            all_sessions = all_sessions[: self.max_sessions]

        my_sessions = [s for i, s in enumerate(all_sessions) if i % world_size == rank]
        my_selections = self._filter_selections_to_sessions(self.selections, set(my_sessions))

        # ── Build tokenizer function ─────────────────────────────────
        def model_tokenizer_fn(batch: Any) -> dict[str, Any]:
            batch = pl_module.transfer_batch_to_device(batch, pl_module.device, dataloader_idx=0)
            tokenizer: Any = pl_module.tokenizer
            with torch.no_grad():
                if hasattr(tokenizer, "reconstruct_from_input"):
                    return tokenizer.reconstruct_from_input(batch, return_latents=True)
                output = tokenizer(batch)
                return {
                    "reconstruction": output.reconstruction,
                    "latents": output.latents,
                }

        # ── Run eval on this rank's sessions ─────────────────────────
        # Wrapped in try/except: if a rank's session fails (e.g. missing
        # behavioral fields), it must still reach the DDP collectives below
        # with empty data to avoid NCCL deadlock.
        per_session_agg: dict[str, tuple[EvaluationReport, float]] = {}
        need_poisson = self.compute_poisson_baseline and self._poisson_baseline_metrics is None
        poisson_agg: dict[str, tuple[EvaluationReport, float]] = {}
        dm = None

        try:
            if my_selections:
                dm = self._get_or_build_datamodule(trainer, pl_module, my_selections)
                session_reports = self._run_per_session(
                    dm, model_tokenizer_fn, spec, bin_size_seconds, token_bin_size_ms
                )
            else:
                session_reports = {}

            # ── Pre-aggregate per session before gathering ───────────
            for sid, segment_list in session_reports.items():
                weight = sum(float(seg.original.shape[1]) for seg, _ in segment_list)
                per_session_agg[sid] = (
                    aggregate_segment_reports(segment_list, self.aggregation_strategy),
                    weight,
                )

            # ── Poisson baseline (local computation, pre-aggregated) ─
            if need_poisson:
                poisson_reports = (
                    self._compute_poisson_baseline_reports(
                        dm, spec, bin_size_seconds, token_bin_size_ms
                    )
                    if dm is not None
                    else {}
                )
                for sid, segment_list in poisson_reports.items():
                    weight = sum(float(seg.original.shape[1]) for seg, _ in segment_list)
                    poisson_agg[sid] = (
                        aggregate_segment_reports(segment_list, self.aggregation_strategy),
                        weight,
                    )
        except Exception:
            import traceback

            _local_error: str | None = traceback.format_exc()
            log.warning(
                f"[rank: {rank}] BehavioralDatasetEvalCallback: local eval failed, "
                f"contributing empty data to gather.\n{_local_error}"
            )
            per_session_agg = {}
            poisson_agg = {}
        else:
            _local_error = None

        # ── DDP gather ────────────────────────────────────────────────
        # Barrier first: each rank's local eval takes a different amount of
        # time (different sessions / different sizes).  Without a barrier the
        # fastest rank's all_gather fires while the slowest rank is still
        # computing, and NCCL times out (default 30 min).  The barrier
        # resynchronises all ranks before any collective data transfer.
        if is_distributed:
            trainer.strategy.barrier("behavioral_eval_pre_gather")
            per_session_agg = self._gather_and_merge(per_session_agg, world_size)
            if need_poisson:
                poisson_agg = self._gather_and_merge(poisson_agg, world_size)
            # Mark Poisson as computed on ALL ranks so need_poisson stays
            # consistent across ranks on the next invocation.
            if need_poisson:
                self._poisson_baseline_metrics = {}
            if rank != 0:
                if _local_error is not None:
                    raise RuntimeError(f"[rank: {rank}] Behavioral eval failed:\n{_local_error}")
                return  # only rank 0 logs

        # Re-raise after collectives so the run fails visibly.
        if _local_error is not None:
            raise RuntimeError(f"[rank: {rank}] Behavioral eval failed:\n{_local_error}")

        # ── Poisson baseline aggregation (rank 0 only in DDP) ────────
        if need_poisson:
            if poisson_agg:
                aggregated_poisson = _aggregate_pre_aggregated(
                    poisson_agg, self.aggregation_strategy
                )
                baseline_metrics = _report_to_metrics(aggregated_poisson, self.log_prefix)
                prefix_str = f"{self.log_prefix}/"
                self._poisson_baseline_metrics = {
                    k[len(prefix_str) :]: v for k, v in baseline_metrics.items()
                }
            else:
                self._poisson_baseline_metrics = {}

        if not per_session_agg:
            log.warning(
                "BehavioralDatasetEvalCallback: no session reports produced. "
                "Check that the explicit selections resolve to available sessions."
            )
            return

        aggregated = _aggregate_pre_aggregated(per_session_agg, self.aggregation_strategy)
        metrics_to_log = _report_to_metrics(aggregated, prefix)
        metrics_to_log[f"{prefix}/n_sessions"] = float(len(per_session_agg))

        # ── Log Poisson baseline and deltas ──────────────────────────
        if self._poisson_baseline_metrics:
            for metric_name, baseline_val in self._poisson_baseline_metrics.items():
                metrics_to_log[f"{prefix}/poisson_baseline/{metric_name}"] = baseline_val
                model_key = f"{prefix}/{metric_name}"
                if model_key in metrics_to_log:
                    metrics_to_log[f"{prefix}/delta_to_poisson/{metric_name}"] = (
                        metrics_to_log[model_key] - baseline_val
                    )

        if trainer.logger is not None and metrics_to_log:
            trainer.logger.log_metrics(metrics_to_log, step=trainer.global_step)

    def _compute_poisson_baseline_reports(
        self,
        datamodule: NTTDataModule,
        spec: Any,
        bin_size_seconds: float | None,
        token_bin_size_ms: float | None,
    ) -> dict:
        """Run Poisson baseline on a datamodule, return raw per-session reports.

        The Poisson tokenizer is a non-trainable baseline that preserves
        firing rates but destroys fine temporal structure.  Returns raw
        reports (not aggregated) so the caller can gather across DDP ranks
        before aggregating.
        """
        from ntt.contrib.tokenizer.poisson_baseline import (
            PoissonBaselineTokenizer,
            PoissonBaselineTokenizerSpec,
        )

        bin_size_ms = float(spec.bin_size_ms) if spec.bin_size_ms else 1.0
        rate_bin_ms = token_bin_size_ms if token_bin_size_ms else 10.0

        # Resolve to a concrete float — variable_t specs (e.g. CALM trained with
        # a UniformDist window) expose a Dist here, which PoissonBaselineTokenizerSpec
        # rejects (window_length_seconds: float).
        poisson_spec = PoissonBaselineTokenizerSpec(
            bin_size_ms=bin_size_ms,
            window_length_seconds=spec.effective_window_length_seconds,
            rate_estimation_bin_ms=rate_bin_ms,
        )
        poisson_tokenizer = PoissonBaselineTokenizer(poisson_spec)
        poisson_tokenizer.eval()

        def poisson_fn(batch: Any) -> dict[str, Any]:
            with torch.no_grad():
                return poisson_tokenizer.reconstruct_from_input(batch, return_latents=True)  # type: ignore[return-value]

        return self._run_per_session(datamodule, poisson_fn, spec, bin_size_seconds, rate_bin_ms)


class TokenizerEvalCallback(Callback):
    """
    PyTorch Lightning callback for evaluating tokenizer reconstructions.

    Does not load or propagate latents or behavioral data.

    Supports two evaluation modes:

    1. Quick Evaluation (per-batch):
       - Runs every N training batches or at validation epoch end
       - Only computes fast metrics (spike detection, single unit)
       - Evaluates a single batch at a time

    2. Full Session-Collated Evaluation:
       - Aggregates validation windows into longer per-session segments
       - Runs all metrics including slower pairwise/population metrics
       - Can be triggered every N epochs or only at training end
       - Session count can be limited for speed

    Usage:
        callback = TokenizerEvalCallback(
            eval_config=EvaluationConfig(),
            lightning_config=LightningEvalConfig(
                full_eval_at_training_end=True,
                max_sessions_for_full_eval=5,
            ),
        )
        trainer = pl.Trainer(callbacks=[callback])
    """

    def __init__(
        self,
        eval_config: EvaluationConfig | None = None,
        lightning_config: LightningEvalConfig | None = None,
    ):
        super().__init__()
        self.eval_config = eval_config or EvaluationConfig()
        self.lightning_config = lightning_config or LightningEvalConfig()
        self.pipeline = EvaluationPipeline(config=self.eval_config)
        self.pipeline.metrics = [
            (name, metric)
            for name, metric in self.pipeline.metrics
            if metric.category != MetricCategory.DOWNSTREAM_DECODING
        ]
        self.batch_count = 0

        # Determine quick eval metric names
        self.quick_metrics = self.lightning_config.quick_metrics

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Run quick evaluation every N batches during training.

        Uses within-batch collation to aggregate items from the same session,
        producing more reliable metric estimates than single-item evaluation.
        """
        self.batch_count += 1

        if self.batch_count % self.lightning_config.eval_every_n_batches != 0:
            return

        # Get collated data for all sessions in this batch
        with torch.no_grad():
            session_data = self._get_eval_data_collated(pl_module, batch)

        if session_data is None or len(session_data) == 0:
            return

        # Run evaluation for each session and aggregate results
        all_results: dict[str, list[float]] = {}

        for _session_id, (original, reconstructed) in session_data.items():
            if self.lightning_config.quick_metrics_only:
                report = self.pipeline.run_quick(
                    original, reconstructed, metric_names=self.quick_metrics
                )
            else:
                report = self.pipeline.run(original, reconstructed)

            # Collect results for aggregation
            for name, result in report.results.items():
                value = (
                    result.normalized_value
                    if result.normalized_value is not None
                    else result.raw_value
                )
                if hasattr(value, "__len__"):
                    if len(value) == 0:  # type: ignore[arg-type]
                        continue
                    value = np.nanmean(value)
                if np.isscalar(value) and np.isfinite(value):
                    all_results.setdefault(name, []).append(float(cast(Any, value)))

        # Log aggregated metrics
        if all_results and trainer.logger is not None:
            metrics_to_log = {}
            for name, values in all_results.items():
                if values:
                    metrics_to_log[f"train_eval/{name}"] = float(np.nanmean(values))
            if metrics_to_log:
                trainer.logger.log_metrics(metrics_to_log, step=trainer.global_step)

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Run evaluation at end of validation epoch if configured."""
        # Quick eval on epoch end with multi-batch accumulation
        if self.lightning_config.eval_on_epoch_end:
            self._run_quick_epoch_eval(
                trainer, pl_module, n_batches=self.lightning_config.quick_eval_n_batches
            )

        # Full session-collated eval every N epochs
        if self.lightning_config.full_eval_every_n_epochs is not None:
            current_epoch = trainer.current_epoch
            if (current_epoch + 1) % self.lightning_config.full_eval_every_n_epochs == 0:
                log.info(f"Running full session-collated evaluation at epoch {current_epoch + 1}")
                self._run_full_session_eval(trainer, pl_module, prefix="full_eval")

    def on_fit_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Run full session-collated evaluation at end of training."""
        if not self.lightning_config.full_eval_at_training_end:
            return

        log.info("Running full session-collated evaluation at training end")
        self._run_full_session_eval(trainer, pl_module, prefix="final_eval")

    def _run_quick_epoch_eval(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        n_batches: int = 3,
    ) -> None:
        """Run quick evaluation on validation data with within-batch collation.

        Accumulates multiple batches and collates items from the same session
        to produce longer segments for more reliable metric estimates.

        Args:
            trainer: Lightning trainer
            pl_module: Lightning module
            n_batches: Number of batches to accumulate (default: 3)
        """
        val_dataloaders = trainer.val_dataloaders
        if val_dataloaders is None or len(val_dataloaders) == 0:
            return

        # Get the dataloader
        if isinstance(val_dataloaders, list):
            dataloader = val_dataloaders[0]
        else:
            dataloader = val_dataloaders

        # Accumulate session data across multiple batches
        accumulated_originals: dict[str, list[np.ndarray]] = {}
        accumulated_recons: dict[str, list[np.ndarray]] = {}
        accumulated_kinds: dict[str, str] = {}
        bin_size_seconds: float | None = None
        batch_count = 0

        spec = getattr(pl_module, "tokenizer_spec", None)
        if spec is not None:
            bin_size_seconds = spec.bin_size_ms / 1000.0 if spec.bin_size_ms else None

        def _array_and_kind(data: EvalData) -> tuple[np.ndarray | None, str | None]:
            if data.binned_spikes is not None:
                return data.binned_spikes, "binned_spikes"
            if data.binned_counts is not None:
                return data.binned_counts, "binned_counts"
            if data.rates_hz is not None:
                return data.rates_hz, "rates_hz"
            return None, None

        for batch in dataloader:
            if batch_count >= n_batches:
                break

            batch = pl_module.transfer_batch_to_device(batch, pl_module.device, dataloader_idx=0)

            with torch.no_grad():
                session_data = self._get_eval_data_collated(pl_module, batch)

            if session_data is None:
                continue

            # Accumulate per-session data
            for session_id, (original, reconstructed) in session_data.items():
                orig_array, kind = _array_and_kind(original)
                recon_array, recon_kind = _array_and_kind(reconstructed)
                if orig_array is not None:
                    accumulated_originals.setdefault(session_id, []).append(orig_array)
                if recon_array is not None:
                    accumulated_recons.setdefault(session_id, []).append(recon_array)
                if kind is not None:
                    accumulated_kinds.setdefault(session_id, kind)
                if recon_kind is not None and kind is not None and recon_kind != kind:
                    log.warning(
                        "Mismatched EvalData kinds for session %s: original=%s reconstructed=%s",
                        session_id,
                        kind,
                        recon_kind,
                    )
                # Get bin_size from first valid data
                if bin_size_seconds is None and original.bin_size_seconds is not None:
                    bin_size_seconds = original.bin_size_seconds

            batch_count += 1

        if not accumulated_originals:
            return

        # Create collated EvalData for each session and run evaluation
        all_results: dict[str, list[float]] = {}
        all_category_scores: dict[MetricCategory, list[float]] = {}
        n_dropped = 0

        for session_id in accumulated_originals:
            if session_id not in accumulated_recons:
                continue

            # Concatenate along time axis (axis=1 for (units, time) format).
            # Skip sessions where neuron counts vary across windows (e.g. due to
            # UnitDropout sampling different subsets per window).
            orig_arrays = accumulated_originals[session_id]
            recon_arrays = accumulated_recons[session_id]
            if (
                len({a.shape[0] for a in orig_arrays}) > 1
                or len({a.shape[0] for a in recon_arrays}) > 1
            ):
                continue
            orig_collated = np.concatenate(orig_arrays, axis=1)
            recon_collated = np.concatenate(recon_arrays, axis=1)

            duration = orig_collated.shape[1] * bin_size_seconds if bin_size_seconds else None
            kind = accumulated_kinds.get(session_id, "binned_spikes")
            if kind == "binned_spikes":
                original_kwargs: dict[str, Any] = {"binned_spikes": orig_collated}
                recon_kwargs: dict[str, Any] = {"binned_spikes": recon_collated}
                data_type = DataType.SPIKES
            elif kind == "binned_counts":
                original_kwargs = {"binned_counts": orig_collated}
                recon_kwargs = {"binned_counts": recon_collated}
                data_type = DataType.RATES
            else:
                original_kwargs = {"rates_hz": orig_collated}
                recon_kwargs = {"rates_hz": recon_collated}
                data_type = DataType.RATES

            original = EvalData(
                **original_kwargs,
                bin_size_seconds=bin_size_seconds,
                duration=duration,
                data_type=data_type,
            )
            reconstructed = EvalData(
                **recon_kwargs,
                bin_size_seconds=bin_size_seconds,
                duration=duration,
                data_type=data_type,
            )

            if self.lightning_config.quick_metrics_only:
                report = self.pipeline.run_quick(
                    original, reconstructed, metric_names=self.quick_metrics
                )
            else:
                report = self.pipeline.run(original, reconstructed)

            # Poisson baseline comparison
            if self.lightning_config.compare_to_poisson and bin_size_seconds is not None:
                self._log_poisson_comparison(
                    original, report, bin_size_seconds, all_results, prefix="epoch_eval"
                )

            # A metric that never yields a finite value simply never appears in
            # MLflow, which is indistinguishable from "the callback is off".
            # Name every metric we asked for but did not get, and why.
            if report.skipped_metrics:
                for metric_name, reason in report.skipped_metrics.items():
                    log.warning(
                        "epoch_eval: metric '%s' skipped for session %s: %s",
                        metric_name,
                        session_id,
                        reason,
                    )
                    n_dropped += 1

            # Collect results
            for name, result in report.results.items():
                value = result.raw_value
                if hasattr(value, "__len__"):
                    if len(value) == 0:  # type: ignore[arg-type]
                        log.warning(
                            "epoch_eval: metric '%s' produced an empty array for session %s",
                            name,
                            session_id,
                        )
                        n_dropped += 1
                        continue
                    value = np.nanmean(value)
                if np.isscalar(value) and np.isfinite(value):
                    all_results.setdefault(name, []).append(float(cast(Any, value)))
                else:
                    log.warning(
                        "epoch_eval: metric '%s' is non-finite (%s) for session %s — "
                        "not logged. A degenerate reconstruction (all-zero or "
                        "saturated) is the usual cause.",
                        name,
                        value,
                        session_id,
                    )
                    n_dropped += 1

            # Collect category scores
            if report.category_scores:
                for cat, score in report.category_scores.items():
                    if np.isfinite(score):
                        all_category_scores.setdefault(cat, []).append(float(score))

        # Log aggregated metrics
        if all_results and trainer.logger is not None:
            metrics_to_log = {}
            for name, values in all_results.items():
                if values:
                    metrics_to_log[f"epoch_eval/{name}"] = float(np.nanmean(values))

            # Log category scores
            for cat, scores in all_category_scores.items():
                if scores:
                    metrics_to_log[f"epoch_eval/category_{cat.value}"] = float(np.nanmean(scores))

            metrics_to_log["epoch_eval/n_sessions"] = float(len(accumulated_originals))
            metrics_to_log["epoch_eval/n_batches"] = float(batch_count)
            # Numeric alarm: a non-zero value here means metrics were requested
            # but silently produced nothing, visible in MLflow without log-diving.
            metrics_to_log["epoch_eval/n_metrics_dropped"] = float(n_dropped)
            metrics_to_log["epoch_eval/n_metrics_logged"] = float(len(all_results))
            trainer.logger.log_metrics(metrics_to_log, step=trainer.global_step)
        elif trainer.logger is not None:
            log.warning(
                "epoch_eval produced NO usable metrics (%d sessions, %d batches, "
                "%d metric results dropped). Check the reconstruction is not degenerate.",
                len(accumulated_originals),
                batch_count,
                n_dropped,
            )

    def _run_full_session_eval(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        prefix: str = "full_eval",
    ) -> None:
        """Run full session-collated evaluation over the validation set.

        This method:
        1. Iterates over the validation dataloader
        2. Aggregates windows into per-session segments via collation
        3. Runs the full evaluation pipeline on each session
        4. Aggregates metrics using the configured aggregation strategy
        5. Logs aggregated (and optionally per-session) metrics
        """
        val_dataloaders = trainer.val_dataloaders
        if val_dataloaders is None or len(val_dataloaders) == 0:
            log.warning("No validation dataloaders available for full evaluation")
            return

        # Get the dataloader (handle both single and list cases)
        if isinstance(val_dataloaders, list):
            dataloader = val_dataloaders[0]
        else:
            dataloader = val_dataloaders

        # Get tokenizer spec for data type info
        if not hasattr(pl_module, "tokenizer_spec"):
            log.warning("Module has no tokenizer_spec, skipping full session eval")
            return

        spec: Any = pl_module.tokenizer_spec
        bin_size_seconds = spec.bin_size_ms / 1000.0 if spec.bin_size_ms else None

        # Create tokenizer function for collation
        def tokenizer_fn(batch):
            with torch.no_grad():
                batch = pl_module.transfer_batch_to_device(
                    batch, pl_module.device, dataloader_idx=0
                )
                tokenizer: Any = pl_module.tokenizer
                if hasattr(tokenizer, "reconstruct_from_input"):
                    return tokenizer.reconstruct_from_input(batch)
                output = tokenizer(batch)
            return {"reconstruction": output.reconstruction}

        # Run per-session evaluation
        session_count = 0
        max_sessions = self.lightning_config.max_sessions_for_full_eval
        all_results: dict[str, list[float]] = {}
        all_weights: list[float] = []  # For weighted aggregation
        session_ids: list[str] = []

        agg_strategy = self.lightning_config.session_aggregation_strategy
        log.info(
            f"Starting full session-collated evaluation "
            f"(max_sessions={max_sessions or 'all'}, aggregation={agg_strategy.value})"
        )

        try:
            for segment in iter_collated_validation_segments(
                dataloader,
                tokenizer_fn,
                window_length_seconds=getattr(spec, "window_length_seconds", None),
                device=pl_module.device,
                max_windows=self.eval_config.max_windows,
                max_bytes=self.eval_config.max_bytes,
                max_eval_seconds=self.eval_config.max_eval_seconds,
            ):
                if max_sessions is not None and session_count >= max_sessions:
                    log.info(f"Reached max_sessions limit ({max_sessions}), stopping")
                    break

                # Create EvalData from segment tensors
                # Segment tensors are already in (units, time) format
                assert segment.original is not None and segment.reconstructed is not None
                original = EvalData.from_tensor(
                    segment.original,
                    data_type=spec.output_type,
                    bin_size_seconds=bin_size_seconds,
                )
                reconstructed = EvalData.from_tensor(
                    segment.reconstructed,
                    data_type=spec.output_type,
                    bin_size_seconds=bin_size_seconds,
                )

                report = self.pipeline.run(
                    original,
                    reconstructed,
                    session_id=segment.session_id,
                )

                # Poisson baseline comparison
                if self.lightning_config.compare_to_poisson and bin_size_seconds is not None:
                    self._log_poisson_comparison(
                        original, report, bin_size_seconds, all_results, prefix=prefix
                    )

                # Compute weight for this session (duration * n_units as proxy for data amount)
                weight = float(segment.original.shape[1])  # time bins
                if bin_size_seconds is not None:
                    weight *= bin_size_seconds  # Convert to duration
                all_weights.append(weight)
                session_ids.append(segment.session_id)

                # Collect results for aggregation
                session_metrics = {}
                for name, result in report.results.items():
                    value = result.raw_value
                    # Handle array values - convert to scalar
                    if hasattr(value, "__len__"):
                        if len(value) == 0:  # type: ignore[arg-type]
                            continue  # Skip empty arrays
                        value = np.nanmean(value)
                    # Check for nan (the value is a scalar here)
                    if np.isscalar(value) and np.isfinite(value):
                        all_results.setdefault(name, []).append(float(cast(Any, value)))
                        session_metrics[name] = float(cast(Any, value))

                # Log per-session metrics if configured
                if self.lightning_config.log_per_session_metrics and trainer.logger:
                    per_session_log = {
                        f"{prefix}/session/{segment.session_id}/{k}": v
                        for k, v in session_metrics.items()
                    }
                    trainer.logger.log_metrics(per_session_log, step=trainer.global_step)

                if report.skipped_metrics:
                    for metric_name, reason in report.skipped_metrics.items():
                        log.info(
                            f"Metric '{metric_name}' skipped for session "
                            f"{segment.session_id}: {reason}"
                        )

                session_count += 1
                max_str = str(max_sessions) if max_sessions else "?"
                log.info(f"Evaluated session {segment.session_id} ({session_count}/{max_str})")

        except Exception as e:
            log.error(f"Error during full session evaluation: {e}", exc_info=True)
            raise

        # Log aggregated metrics using configured strategy
        if all_results and trainer.logger is not None:
            metrics_to_log = self._aggregate_session_metrics(
                all_results, all_weights, agg_strategy, prefix
            )
            metrics_to_log[f"{prefix}/n_sessions"] = float(session_count)
            trainer.logger.log_metrics(metrics_to_log, step=trainer.global_step)

        log.info(f"Full session evaluation complete: {session_count} sessions evaluated")

        # DDP barrier: ranks process different val batches (DistributedSampler)
        # and finish at different times.  Without this, the next callback's
        # collectives (e.g. BehavioralDatasetEvalCallback.all_gather) start
        # while some ranks are still here, causing NCCL timeouts.
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            trainer.strategy.barrier("tokenizer_full_eval_done")

    def _aggregate_session_metrics(
        self,
        all_results: dict[str, list[float]],
        weights: list[float],
        strategy: AggregationStrategy,
        prefix: str,
    ) -> dict[str, float]:
        """Aggregate per-session metrics using the specified strategy.

        Args:
            all_results: Dict mapping metric names to lists of per-session values
            weights: List of weights for each session (e.g., duration)
            strategy: Aggregation strategy to use
            prefix: Metric prefix for logging

        Returns:
            Dict of aggregated metrics ready for logging
        """
        metrics_to_log = {}
        weights_arr = np.array(weights) if weights else None

        for name, values in all_results.items():
            if not values:
                continue

            values_arr = np.array(values)

            if strategy == AggregationStrategy.MEAN:
                agg_value = float(np.nanmean(values_arr))
            elif strategy == AggregationStrategy.MEDIAN:
                agg_value = float(np.nanmedian(values_arr))
            elif strategy == AggregationStrategy.WEIGHTED_MEAN:
                if weights_arr is not None and len(weights_arr) == len(values_arr):
                    # Mask out NaN values for weighted average
                    valid_mask = ~np.isnan(values_arr)
                    if valid_mask.any():
                        agg_value = float(
                            np.average(values_arr[valid_mask], weights=weights_arr[valid_mask])
                        )
                    else:
                        agg_value = float("nan")
                else:
                    # Fallback to simple mean if weights don't match
                    agg_value = float(np.nanmean(values_arr))
            else:
                agg_value = float(np.nanmean(values_arr))

            metrics_to_log[f"{prefix}/{name}"] = agg_value
            metrics_to_log[f"{prefix}/{name}_std"] = float(np.nanstd(values_arr))

        return metrics_to_log

    def _get_eval_data_collated(
        self, pl_module: pl.LightningModule, batch: Any
    ) -> dict[str, tuple[EvalData, EvalData]] | None:
        """
        Extract original and reconstructed data, collated by session.

        This method collates all batch items from the same session into longer
        segments, which produces more reliable metric estimates than evaluating
        single batch items.

        Args:
            pl_module: The Lightning module with tokenizer
            batch: Input batch (BinnedSpikesInput)

        Returns:
            Dict mapping session_id to (original, reconstructed) EvalData tuples,
            or None if extraction fails.
        """
        if not hasattr(pl_module, "tokenizer") or not hasattr(pl_module, "tokenizer_spec"):
            log.warning("Module has no tokenizer/tokenizer_spec. Cannot extract evaluation data.")
            return None

        spec: Any = pl_module.tokenizer_spec

        tokenizer: Any = pl_module.tokenizer
        with torch.no_grad():
            if hasattr(tokenizer, "reconstruct_from_input"):
                result = tokenizer.reconstruct_from_input(batch)
                reconstruction = result.reconstruction
            else:
                output = tokenizer(batch)
                reconstruction = output.reconstruction

        # Use batch collation to get longer segments per session
        if isinstance(batch, BinnedSpikesInput | RatesInput) and hasattr(batch, "session_id"):
            try:
                session_data = EvalData.from_binned_batch_collated(
                    batch, reconstructed=reconstruction
                )
                return session_data  # type: ignore[return-value]
            except Exception as e:
                log.warning(f"Batch collation failed: {e}, falling back to single item")

        # Fallback: return single item (first in batch)
        bin_size_seconds = spec.bin_size_ms / 1000.0 if spec.bin_size_ms else None

        if isinstance(batch, BinnedSpikesInput | RatesInput):
            original = EvalData.from_binned_input(batch, batch_index=0)
        else:
            original_tensor = batch.get("input", batch) if isinstance(batch, dict) else None
            if isinstance(original_tensor, torch.Tensor):
                original = EvalData.from_tensor(
                    original_tensor[0].T,  # type: ignore[index]
                    data_type=spec.output_type,
                    bin_size_seconds=bin_size_seconds,
                    duration=spec.effective_window_length_seconds,
                )
            else:
                return None

        reconstructed = EvalData.from_tensor(
            reconstruction[0].T,
            data_type=spec.output_type,
            bin_size_seconds=bin_size_seconds,
            duration=spec.effective_window_length_seconds,
        )

        # Return as dict with unknown session
        session_id = getattr(batch, "session_id", ["unknown"])[0]
        if isinstance(session_id, torch.Tensor):
            session_id = str(session_id.item())
        return {str(session_id): (original, reconstructed)}

    def _get_eval_data(
        self, pl_module: pl.LightningModule, batch: Any
    ) -> tuple[EvalData, EvalData] | None:
        """
        Extract original and reconstructed spike data from module and batch.

        This is a convenience wrapper around _get_eval_data_collated that returns
        the session with the most data (longest time segment).

        For proper evaluation with all sessions, use _get_eval_data_collated directly.
        """
        # First try the module's get_eval_data method
        if hasattr(pl_module, "get_eval_data"):
            try:
                return cast(Any, pl_module).get_eval_data(batch)
            except Exception as e:
                log.warning(f"get_eval_data failed: {e}")

        # Use collated extraction and pick the session with most data
        session_data = self._get_eval_data_collated(pl_module, batch)
        if session_data is None or len(session_data) == 0:
            return None

        # Return the session with the longest duration (most data)
        best_session = max(session_data.items(), key=lambda item: item[1][0].duration or 0)
        return best_session[1]

    def _log_poisson_comparison(
        self,
        original: EvalData,
        model_report: EvaluationReport,
        bin_size_seconds: float,
        all_results: dict[str, list[float]],
        prefix: str,
    ) -> None:
        """Compute Poisson floor and log normalized comparison scores."""
        from ntt.contrib.tokenizer.poisson_baseline import poisson_baseline_reconstruction

        from .aggregation.comparison import (
            compute_normalized_comparisons,
            with_alternative_floors,
        )

        if original.binned_spikes is None:
            return

        bin_size_ms = bin_size_seconds * 1000.0
        rate_bin_ms = self.lightning_config.poisson_rate_estimation_bin_ms
        poisson_spikes = poisson_baseline_reconstruction(
            original.binned_spikes, bin_size_ms, rate_bin_ms, seed=42
        )
        poisson_data = EvalData(
            binned_spikes=poisson_spikes,
            bin_size_seconds=bin_size_seconds,
            duration=original.duration,
            data_type=DataType.SPIKES,
        )
        baseline_report = self.pipeline.run(original, poisson_data)
        ceiling_report = self.pipeline.run(original, original)
        comparisons = compute_normalized_comparisons(
            model_report,
            baseline_report,
            ceiling_report,
        )
        # firing_rate_error against its population-mean-rate reference, as in the eval job.
        comparisons, _ = with_alternative_floors(
            comparisons, {"session": [model_report.results]}, "mean"
        )
        for metric_name, comparison in comparisons.items():
            if comparison.normalized_score is None:
                continue
            all_results.setdefault(
                f"comparison_vs_poisson/{metric_name}",
                [],
            ).append(comparison.normalized_score)

    def _log_metrics(self, trainer: pl.Trainer, report: Any, prefix: str) -> None:
        """Log metrics to configured loggers."""
        if trainer.logger is None:
            return

        metrics_to_log = {}

        for name, result in report.results.items():
            value = result.raw_value

            # Handle array values - convert to scalar
            if hasattr(value, "__len__"):
                if len(value) == 0:
                    continue  # Skip empty arrays
                value = np.nanmean(value)

            # Check for nan (the value is a scalar here)
            if np.isscalar(value) and np.isfinite(value):
                metrics_to_log[f"{prefix}/{name}"] = float(cast(Any, value))

        if metrics_to_log:
            trainer.logger.log_metrics(metrics_to_log, step=trainer.global_step)


class EvalMixin:
    """
    Mixin for LightningModule to provide evaluation interface.

    Usage:
        class MyTokenizer(pl.LightningModule, EvalMixin):
            def __init__(self, ...):
                super().__init__()
                self.output_type = DataType.SPIKES
                self.bin_size = 0.001

            def forward(self, x):
                # Your forward pass
                return reconstructed
    """

    # These should be overridden by the implementing class
    output_type: DataType = DataType.SPIKES
    bin_size: float = 0.001  # in seconds

    def get_reconstruction(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get original and reconstructed tensors from a batch.

        Override this method in your model.

        Args:
            batch: Input batch from dataloader

        Returns:
            (original_tensor, reconstructed_tensor) both of shape (batch, units, time)
        """
        raise NotImplementedError(
            "Implement get_reconstruction() to return (original, reconstructed) tensors"
        )

    def get_eval_data(self, batch: Any) -> tuple[EvalData, EvalData]:
        """
        Get EvalData objects for evaluation.

        Default implementation uses tokenizer spec if available, otherwise
        uses get_reconstruction() and converts tensors.
        Override for custom behavior.
        """
        # Try to use tokenizer spec directly
        if hasattr(self, "tokenizer") and hasattr(self.tokenizer, "spec"):  # type: ignore[attr-defined]
            spec = self.tokenizer.spec  # type: ignore[attr-defined]
            original_tensor, reconstructed_tensor = self.get_reconstruction(batch)

            bin_size_seconds = spec.bin_size_ms / 1000.0 if spec.bin_size_ms else None

            original = EvalData.from_tensor(
                original_tensor[0].T,
                data_type=spec.output_type,
                bin_size_seconds=bin_size_seconds,
                duration=spec.effective_window_length_seconds,
            )

            reconstructed = EvalData.from_tensor(
                reconstructed_tensor[0].T,
                data_type=spec.output_type,
                bin_size_seconds=bin_size_seconds,
                duration=spec.effective_window_length_seconds,
            )

            return original, reconstructed

        # Fallback to manual conversion
        original_tensor, reconstructed_tensor = self.get_reconstruction(batch)
        bin_size_seconds = getattr(self, "bin_size_seconds", getattr(self, "bin_size", 0.001))

        # Take first item in batch for evaluation
        original = EvalData.from_tensor(
            original_tensor[0].T,
            data_type=self.output_type,
            bin_size_seconds=bin_size_seconds,
        )
        reconstructed = EvalData.from_tensor(
            reconstructed_tensor[0].T,
            data_type=self.output_type,
            bin_size_seconds=bin_size_seconds,
        )

        return original, reconstructed


def create_eval_callback_from_config(cfg: Any) -> TokenizerEvalCallback:
    """
    Create evaluation callback from Hydra config.

    Usage with Hydra:
        @hydra.main(config_path="conf", config_name="config")
        def main(cfg):
            eval_callback = create_eval_callback_from_config(cfg)
            trainer = pl.Trainer(callbacks=[eval_callback])
    """
    from .config import validate_config, validate_lightning_config

    eval_config = validate_config(cfg)
    lightning_config = validate_lightning_config(cfg)

    return TokenizerEvalCallback(
        eval_config=eval_config,
        lightning_config=lightning_config,
    )
