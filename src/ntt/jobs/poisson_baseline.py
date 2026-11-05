"""Standalone Poisson-baseline job.

Runs the Poisson floor + perfect ceiling computation independent of any
specific tokenizer, and emits an EvalArtifact (kind="poisson_baseline")
that other eval runs can consume via the ``poisson_reference`` field of
NTTEvalJob — skipping their own Poisson computation when the metadata
matches.
"""

from __future__ import annotations

import gc
import shutil
import tempfile
import typing as T
from functools import partial
from pathlib import Path

import pydantic as pdt
import torch

from ntt.core.data.datamodule import NTTDataModule, NTTDataModuleConfig
from ntt.evaluation.artifact import (
    EvalArtifact,
    compute_decoding_bypass_fields,
    compute_per_session_weights,
    make_metadata,
)
from ntt.evaluation.config import EvaluationConfig, validate_config
from ntt.evaluation.config.schema import AggregationStrategy
from ntt.evaluation.core.types import MetricResult
from ntt.evaluation.mlflow_logging import MLflowLoggingConfig
from ntt.evaluation.neuron_selection import all_neuron_data_config, describe_neuron_selection
from ntt.evaluation.pipeline import EvaluationPipeline
from ntt.evaluation.poisson_realisations import (
    build_lightweight_pipeline,
    run_poisson_realisations,
)
from ntt.evaluation.utils import (
    aggregate_segment_reports,
    aggregate_session_reports,
    run_validation_evaluation_per_session,
)
from ntt.jobs.base import BaseJob
from ntt.jobs.eval import NTTEvalJob, _load_stashed_original
from ntt.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class NTTPoissonBaselineJob(
    BaseJob,
    strict=False,
    frozen=False,
    extra="forbid",
    arbitrary_types_allowed=True,
):
    """Standalone Poisson baseline computation.

    Produces ``poisson_baseline.json`` (an EvalArtifact with
    ``kind="poisson_baseline"``) containing per-session aggregated Poisson
    and perfect-ceiling metric reports plus full metadata for matching.
    """

    KIND: T.Literal["poisson_baseline"] = "poisson_baseline"

    data_config: NTTDataModuleConfig
    eval_config: dict[str, T.Any] | None = None

    device: str = "auto"
    output_path: str | Path | None = None
    split: T.Literal["val", "test"] = "val"
    aggregation_strategy: AggregationStrategy = AggregationStrategy.MEAN

    # Poisson configuration — must match the spec a consumer will use
    bin_size_ms: float
    window_length_seconds: float
    rate_estimation_bin_ms: float
    n_realisations: int = 5
    seed: int = 42
    n_workers: int | None = None
    use_all_neurons: bool = True

    mlflow_logging: MLflowLoggingConfig = MLflowLoggingConfig()

    _device: torch.device | None = pdt.PrivateAttr(default=None)
    _eval_config: EvaluationConfig | None = pdt.PrivateAttr(default=None)

    def run(self) -> Path:
        """Execute the job and return the artifact path."""
        from ntt.contrib.tokenizer.poisson_baseline import (
            PoissonBaselineTokenizer,
            PoissonBaselineTokenizerSpec,
        )

        # ── Setup ───────────────────────────────────────────────────
        self._eval_config = validate_config(self.eval_config or {})
        self._device = (
            torch.device(self.device)
            if self.device != "auto"
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        bin_size_seconds = self.bin_size_ms / 1000.0

        poisson_spec = PoissonBaselineTokenizerSpec(
            window_length_seconds=self.window_length_seconds,
            bin_size_ms=self.bin_size_ms,
            rate_estimation_bin_ms=self.rate_estimation_bin_ms,
            seed=self.seed,
        )
        poisson_tokenizer = PoissonBaselineTokenizer(poisson_spec).to(self._device)
        poisson_tokenizer.eval()

        _tokenizer = poisson_tokenizer  # capture for closure (linter clarity)

        @torch.no_grad()
        def _poisson_fn(batch):
            return _tokenizer.reconstruct_from_input(batch, return_latents=True)

        # ── Build dataloader ────────────────────────────────────────
        if self.use_all_neurons:
            # Same units as a full-neuron model eval (shared EvalUnitSelection).
            dm = NTTDataModule(
                all_neuron_data_config(self.data_config, self._eval_config, poisson_spec)
            )
        else:
            dm = NTTDataModule(self.data_config, poisson_spec)
        stage = "validate" if self.split == "val" else "test"
        dm.setup(stage)
        neuron_selection = describe_neuron_selection(dm, self.split, self._eval_config)
        dataloader = dm.val_dataloader() if self.split == "val" else dm.test_dataloader()

        # ── Realisation 1 (full pipeline + stash) ───────────────────
        stash_dir = tempfile.mkdtemp(prefix="ntt_poisson_originals_")
        log.info(f"Realisation 1/{self.n_realisations}: full pipeline (stash={stash_dir})")
        try:
            first_reports = run_validation_evaluation_per_session(
                dataloader,
                _poisson_fn,
                data_type=poisson_spec.input_type,
                reconstructed_data_type=poisson_spec.output_type,
                bin_size_seconds=bin_size_seconds,
                token_bin_size_ms=poisson_spec.token_bin_size_ms,
                eval_config=self._eval_config,
                device=self._device,
                stash_originals_dir=stash_dir,
            )
        except Exception:
            shutil.rmtree(stash_dir, ignore_errors=True)
            raise

        del poisson_tokenizer, _poisson_fn
        gc.collect()
        torch.cuda.empty_cache()

        # ── Reload originals as mmap views ──────────────────────────
        from ntt.evaluation.core.types import EvalData

        originals: dict[str, list] = {}
        for sid, segs in first_reports.items():
            entries = []
            for segment, data in segs:
                stash_path = data["metadata"].get("_original_stash_path")
                if stash_path is None:
                    raise RuntimeError(f"Missing stash path for session {sid}")
                original_np = _load_stashed_original(stash_path)
                eval_data = EvalData(
                    binned_spikes=original_np,
                    bin_size_seconds=bin_size_seconds,
                    duration=original_np.shape[1] * bin_size_seconds,
                    data_type=poisson_spec.input_type,
                )
                # Drop CUDA tensors from the segment before it crosses the
                # fork-pool IPC boundary; downstream aggregation only needs
                # `original_shape`, which tensor_free_copy preserves.
                entries.append((segment.tensor_free_copy(), eval_data, dict(data["metadata"])))
            originals[sid] = entries

        # ── Build lightweight pipeline (drop downstream + correlograms) ─
        # Parent uses this for the perfect-ceiling pass below; workers build
        # their own via the factory (see poisson_realisations module docstring
        # for the fork+CUDA contract).
        lightweight_pipeline = build_lightweight_pipeline(self._eval_config)
        pipeline_factory = partial(build_lightweight_pipeline, self._eval_config)

        # ── Realisations 2..N (parallel) ─────────────────────────────
        extra = run_poisson_realisations(
            originals=originals,
            pipeline_factory=pipeline_factory,
            bin_size_seconds=bin_size_seconds,
            bin_size_ms=self.bin_size_ms,
            poisson_rate_ms=self.rate_estimation_bin_ms,
            base_seed=self.seed,
            n_realisations_to_run=self.n_realisations - 1,
            n_workers=self.n_workers,
        )
        all_reports = [first_reports, *extra]

        # ── Average across realisations ─────────────────────────────
        averaged = NTTEvalJob._average_session_reports_across_realisations(
            all_reports, MetricResult
        )
        del all_reports
        gc.collect()

        # ── Perfect ceiling ──────────────────────────────────────────
        n_ceiling = sum(len(v) for v in originals.values())
        log.info(f"Computing perfect ceiling over {n_ceiling} segments...")
        perfect_reports: dict[str, list] = {}
        for sid, entries in originals.items():
            for segment, eval_data, metadata in entries:
                report = lightweight_pipeline.run(eval_data, eval_data)
                perfect_reports.setdefault(sid, []).append(
                    (segment, {"report": report, "metadata": dict(metadata)})
                )

        shutil.rmtree(stash_dir, ignore_errors=True)

        # ── Aggregate ───────────────────────────────────────────────
        poisson_per_session = {
            sid: aggregate_segment_reports(segs, self.aggregation_strategy).to_dict()
            for sid, segs in averaged.items()
        }
        perfect_per_session = {
            sid: aggregate_segment_reports(segs, self.aggregation_strategy).to_dict()
            for sid, segs in perfect_reports.items()
        }
        poisson_aggregated = aggregate_session_reports(
            averaged, strategy=self.aggregation_strategy
        ).to_dict()
        perfect_aggregated = aggregate_session_reports(
            perfect_reports, strategy=self.aggregation_strategy
        ).to_dict()

        # ── Build artifact ──────────────────────────────────────────
        sessions = sorted(originals.keys())
        # Bypass fields are derived from the *full* metric list (not the
        # lightweight one) — the full set knows which downstream-prediction
        # behavioral fields to surface for max_eval_seconds bypass logic.
        full_pipeline = EvaluationPipeline(config=self._eval_config)
        decoding_bypass_fields = compute_decoding_bypass_fields(full_pipeline.metrics)
        meta = make_metadata(
            kind="poisson_baseline",
            bin_size_ms=self.bin_size_ms,
            window_length_seconds=self.window_length_seconds,
            rate_estimation_bin_ms=self.rate_estimation_bin_ms,
            split=self.split,
            max_eval_seconds=getattr(self._eval_config, "max_eval_seconds", None),
            decoding_bypass_fields=decoding_bypass_fields,
            sessions=sessions,
            n_realisations=self.n_realisations,
            seed=self.seed,
            aggregation_strategy=self.aggregation_strategy.value,
            neuron_selection=neuron_selection,
        )
        # Per-session weights — computed from realisation 0's segment shapes
        # (identical for poisson and ceiling since they use the same
        # originals). Required for a downstream consumer using
        # WEIGHTED_MEAN to match the inline-eval aggregation exactly.
        per_session_weights = compute_per_session_weights(first_reports)

        artifact = EvalArtifact(
            metadata=meta,
            per_session=poisson_per_session,
            aggregated=poisson_aggregated,
            perfect_per_session=perfect_per_session,
            perfect_aggregated=perfect_aggregated,
            per_session_weights=per_session_weights,
        )
        out_dir = Path(str(self.output_path or "."))
        out_path = out_dir / "poisson_baseline.json"
        artifact.dump_json(out_path)
        log.info(f"Wrote Poisson baseline artifact to {out_path}")

        # ── MLflow logging (optional) ────────────────────────────────
        # Log the artifact so future eval runs can reference it via
        # ``mlflow-artifacts:/<run_id>/poisson_baseline.json`` (resolved by
        # NTTEvalJob.poisson_reference). Also log baseline + ceiling metric
        # values flat for browsing in the UI, and structural matching fields
        # as tags (so users can query by bin/window/split/sessions).
        if self.mlflow_logging.enabled:
            try:
                self._log_to_mlflow(out_path, meta, poisson_aggregated, perfect_aggregated)
            except Exception:
                log.warning("Failed to log Poisson baseline to MLflow", exc_info=True)
        return out_path

    def _log_to_mlflow(
        self,
        artifact_path: Path,
        meta,
        poisson_aggregated: dict,
        perfect_aggregated: dict,
    ) -> None:
        """Upload the artifact + flat metrics + structural tags to MLflow."""
        import mlflow

        cfg = self.mlflow_logging
        if cfg.tracking_uri is not None:
            mlflow.set_tracking_uri(cfg.tracking_uri)
        mlflow.set_experiment(cfg.experiment_name)

        with mlflow.start_run(run_name=cfg.run_name) as run:
            run_id = run.info.run_id

            # Flat metrics (raw_value of each metric, prefixed for clarity)
            for metric_name, result in poisson_aggregated.get("results", {}).items():
                value = result.get("raw_value")
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"poisson/{metric_name}", float(value))
            for metric_name, result in perfect_aggregated.get("results", {}).items():
                value = result.get("raw_value")
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"perfect/{metric_name}", float(value))

            # Structural tags (consumers can query by these)
            mlflow.set_tags(
                {
                    "ntt.eval_artifact_kind": "poisson_baseline",
                    "ntt.poisson.bin_size_ms": str(meta.bin_size_ms),
                    "ntt.poisson.window_length_seconds": str(meta.window_length_seconds),
                    "ntt.poisson.rate_estimation_bin_ms": str(meta.rate_estimation_bin_ms),
                    "ntt.poisson.split": meta.split,
                    "ntt.poisson.max_eval_seconds": str(meta.max_eval_seconds),
                    "ntt.poisson.n_realisations": str(meta.n_realisations),
                    "ntt.poisson.seed": str(meta.seed),
                    "ntt.poisson.n_sessions": str(len(meta.sessions)),
                }
            )

            # Upload the artifact itself
            mlflow.log_artifact(str(artifact_path))
            log.info(
                f"Poisson baseline logged to MLflow run {run_id} (artifact: poisson_baseline.json)"
            )
