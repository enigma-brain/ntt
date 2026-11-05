"""Aggregation job: compute category scores from eval result JSONs."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from datetime import UTC
from pathlib import Path
from typing import Any, Literal

import mlflow
import pydantic as pdt

from ntt.evaluation.aggregation.category_scores import (
    ENCODING_METRIC,
    CategoryScores,
    ColumnDiagnostics,
    compute_category_scores,
)
from ntt.evaluation.aggregation.comparability import (
    COMPARABILITY_VERSION,
    input_geometry,
    recorded_settings,
    report_session_summary,
    selected_chunk_windows,
    with_scored_window,
)
from ntt.evaluation.aggregation.downstream_records import (
    DownstreamScoreView,
    load_downstream_record,
)
from ntt.evaluation.aggregation.publish import DEFAULT_EXPERIMENT, publish_category_scores
from ntt.evaluation.aggregation.tracker_io import (
    load_json_artifact as _load_json,
)
from ntt.evaluation.generalization import (
    build_tier_index,
    compute_from_per_session_reports,
)
from ntt.evaluation.mlflow_logging import MLflowLoggingConfig
from ntt.jobs.base import BaseJob
from ntt.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)

# Stamped into ``category_scores.json`` metadata as ``schema_version``.
AGGREGATION_SCHEMA_VERSION = 3

# `generalization_source` of a score this job rebuilt from per-session results. The split it
# was rebuilt from is appended, e.g. "recomputed_from_per_session(test)", so the
# comparability check sees a val/test mix and a resumed table build can rescore rows built
# on the other split. A precomputed block does not depend on `gen_split`.
RECOMPUTED_GENERALIZATION = "recomputed_from_per_session"


def recomputed_generalization_source(gen_split: str) -> str:
    """``generalization_source`` of a score rebuilt from per-session results on ``gen_split``."""
    return f"{RECOMPUTED_GENERALIZATION}({gen_split})"


class NTTAggregationJob(BaseJob, strict=True):
    """Aggregate eval result JSONs into seven category scores."""

    # Must be a real pydantic field (not a ClassVar) so it can serve as the
    # `JobKind` union discriminator, as it does on every other job.
    KIND: Literal["aggregation"] = "aggregation"

    # Paths to input eval result JSONs — local file paths or mlflow-artifacts:/ URIs.
    # encoding_path / decoding_path are downstream records (load_downstream_record): an
    # encoding run's downstream_encoding_results.json, a decoding benchmark parent's
    # artifact dir (its cohort record sits at the root).
    recon_path: str | None = None
    m1_path: str | None = None
    m2_path: str | None = None
    encoding_path: str | None = None
    decoding_path: str | None = None

    # Poisson reference for normalization of the RECON buckets (single_neuron_recon,
    # population_structure). The latent_structure_and_robustness bucket is NOT
    # Poisson-normalized (it reads raw M1/M2/structure values and applies bounded
    # [0,1] transforms), so it ignores these.
    # - None: use precomputed comparison_vs_poisson from the report (error if missing)
    # - set + force_recompute_poisson=False: precomputed if present, else recompute
    # - set + force_recompute_poisson=True: always recompute from reference
    poisson_reference: str | None = None
    force_recompute_poisson: bool = False

    score_clip: float | None = -1.0  # recon buckets only; the latent bucket never clips
    compression_metric: str = "compression_rate"
    # A decoding cohort with missing cells scores None by default; true scores it over
    # the found cells (the missing ones stay in metadata.downstream.decoding).
    allow_incomplete_decoding_cohort: bool = False

    # latent_structure_and_robustness — M2 latent-noise robustness. ON by default
    # (team decision: ACG-under-noise at ε=0.1 discriminates cleanly across the
    # tokenizer set). These are GLOBAL knobs, fixed across ALL tokenizers in a
    # comparison (never per-tokenizer): m2_metric/m2_epsilon select one
    # ACG-under-noise operating point so every tokenizer is bounded at the same ε.
    # Tokenizers lacking that row are warned-and-excluded (bucket scores on M1+structure).
    include_m2: bool = True
    m2_metric: str = "acg_correlation"
    m2_epsilon: float = 0.1
    # Exclude a part from the score by marking it "off", the same way --common-parts-only already
    # excludes parts that only some tokenizers have.
    include_m1: bool = True
    include_poyo: bool = True
    include_ridge: bool = True
    # Poisson recompute: if the reference details (sessions, averaging, etc.) were not saved, this
    # raises an error. When common_parts_only is true, it raises a warning instead.
    common_parts_only: bool = False

    # Generalization fallback: recompute the score from per_session data when the
    # recon report lacks a precomputed `generalization` block. On by default.
    generalization_fallback: bool = True
    gen_config_root: str = "config"
    gen_selection_family: str = "ntt_full"
    gen_split: Literal["val", "test"] = "val"
    gen_baseline_tier: str = "xtime"

    ckpt_uri: str | None = None
    output_path: str

    mlflow_logging: MLflowLoggingConfig = pdt.Field(
        default_factory=lambda: MLflowLoggingConfig(experiment_name=DEFAULT_EXPERIMENT)
    )

    def run(self) -> dict[str, Any]:
        if (
            self.mlflow_logging.enabled
            and not self.mlflow_logging.tracking_uri
            and not os.environ.get("MLFLOW_TRACKING_URI")
        ):
            # Checked before scoring: without a server MLflow logs to ./mlruns, a run no
            # tracker can link to.
            raise ValueError(
                "mlflow_logging.enabled needs a tracking server: set "
                "mlflow_logging.tracking_uri or MLFLOW_TRACKING_URI"
            )
        recon = _load_json(self.recon_path) if self.recon_path else None
        if recon is not None and "generalization" in recon:
            summary = (recon["generalization"] or {}).get("summary") or {}
            if "score_tiers" not in summary:
                # A generalization block without score_tiers was scored by another formula:
                # treat it as absent so one cohort never mixes formulas.
                log.warning(
                    "generalization block in %s has no score_tiers; dropping it "
                    "(rescored from per_session when generalization_fallback is on)",
                    self.recon_path,
                )
                del recon["generalization"]
        had_precomputed = recon is not None and "generalization" in recon
        gen_source = "precomputed" if had_precomputed else "absent"
        if (
            self.generalization_fallback
            and recon is not None
            and not had_precomputed
            and isinstance(recon.get("per_session"), dict)
            and recon["per_session"]
        ):
            if self._backfill_generalization(recon):
                gen_source = recomputed_generalization_source(self.gen_split)
        # M1/M2 robustness eval reports are optional: a tokenizer may not have a
        # robustness run, or the artifact may be missing. A load failure here must
        # not crash the batch — warn and yield None so the resolver's
        # fail-loud-exclude path leaves the latent bucket on its remaining members.
        m1 = self._load_optional_json(self.m1_path, "m1") if self.m1_path else None
        m2 = self._load_optional_json(self.m2_path, "m2") if self.m2_path else None
        encoding = (
            load_downstream_record(self.encoding_path, "encoding") if self.encoding_path else None
        )
        decoding = (
            load_downstream_record(self.decoding_path, "decoding") if self.decoding_path else None
        )
        poisson = _load_json(self.poisson_reference) if self.poisson_reference else None

        scores = compute_category_scores(
            recon_report=recon,
            m1_report=m1,
            m2_report=m2,
            encoding=encoding,
            decoding=decoding,
            poisson_baseline=poisson,
            force_recompute_poisson=self.force_recompute_poisson,
            score_clip=self.score_clip,
            compression_metric=self.compression_metric,
            include_m2=self.include_m2,
            m2_metric=self.m2_metric,
            m2_epsilon=self.m2_epsilon,
            include_m1=self.include_m1,
            include_poyo=self.include_poyo,
            include_ridge=self.include_ridge,
            common_parts_only=self.common_parts_only,
            allow_incomplete_decoding_cohort=self.allow_incomplete_decoding_cohort,
        )

        out_dir = Path(self.output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        output_file = out_dir / "category_scores.json"
        output_dict = _build_output_dict(scores, self, gen_source)
        # How each score was averaged, so tables can warn when two tokenizers were
        # averaged differently (see find_mixed_member_rows):
        # - poisson_norm_sessions: model ∩ Poisson reference, behind the two
        #   Poisson-normalized columns;
        # - input_sessions: session set + cross-session aggregation strategy per input;
        # - input_settings: the recorded run settings that change metric values.
        # A robustness eval_report.json saves its sessions and settings in its config; the
        # run's results_per_session.csv and run_meta.json next to it supply what the config
        # lacks (session ids, settings).
        m1_session_ids, m1_run_meta = _robustness_run_files(
            self.m1_path if m1 is not None else None
        )
        m2_session_ids, m2_run_meta = _robustness_run_files(
            self.m2_path if m2 is not None else None
        )
        recon_settings = None
        if recon is not None:
            # results.json wraps the eval config; a compact eval_report.json has it on top.
            recon_eval_config = (recon.get("aggregated") or {}).get("config") or recon.get(
                "config"
            )
            recon_run_meta = _run_meta(self.recon_path)
            recon_settings = with_scored_window(
                recorded_settings(
                    recon,
                    recon_eval_config,
                    _run_geometry(recon.get("tokenizer_config"), recon_run_meta),
                ),
                recon,
                recon_run_meta,
                recon_eval_config,
            )
        metadata = output_dict["metadata"]
        metadata["comparability_version"] = COMPARABILITY_VERSION
        metadata["recon_chunk_window_ms"] = (
            selected_chunk_windows(recon) if recon is not None else None
        )
        metadata["poisson_norm_sessions"] = scores.poisson_norm_sessions
        # Metrics scored against another reference than the Poisson floor.
        metadata["metric_references"] = scores.metric_references
        metadata["input_sessions"] = {
            "recon": report_session_summary(recon),
            "m1": report_session_summary(m1, sibling_session_ids=m1_session_ids),
            "m2": report_session_summary(m2, sibling_session_ids=m2_session_ids),
            "decoding": None if decoding is None else decoding.session_summary("r2"),
            "encoding": None if encoding is None else encoding.session_summary(ENCODING_METRIC),
        }
        # Where each downstream score was read from: split, cohort completeness, and
        # the record's own provenance (tokenizer checkpoint, ntt version/path/commit, ...).
        metadata["downstream"] = {
            "encoding": _downstream_provenance(encoding),
            "decoding": _downstream_provenance(decoding),
        }
        m1_settings = _robustness_settings(m1, m1_run_meta)
        m2_settings = _robustness_settings(m2, m2_run_meta)
        metadata["input_settings"] = {
            "recon": recon_settings,
            "m1": m1_settings,
            "m2": m2_settings,
        }
        # The Poisson reference the recon was normalized by: in-run or external, and its window.
        metadata["poisson_reference_kind"] = (recon or {}).get("poisson_reference_kind")
        metadata["poisson_reference_window_seconds"] = (recon or {}).get(
            "poisson_reference_window_seconds"
        )

        with open(output_file, "w") as f:
            json.dump(output_dict, f, indent=2)

        if self.mlflow_logging.enabled:
            _log_to_mlflow(output_file, self)

        return {"output_path": str(output_file), "scores": output_dict["scores"]}

    @staticmethod
    def _load_optional_json(path: str, label: str) -> dict | None:
        """Load an optional M1/M2 robustness eval report; warn and return None on
        any failure.

        The M1/M2 ``eval_report.json`` artifacts may be absent (no robustness run
        for this tokenizer, or the artifact wasn't uploaded). Never raises.
        """
        try:
            return _load_json(path)
        except Exception as exc:
            log.warning(
                "%s robustness report unavailable (%s): %s — excluding %s from "
                "latent_structure_and_robustness.",
                label,
                type(exc).__name__,
                exc,
                label,
            )
            return None

    def _backfill_generalization(self, recon: dict) -> bool:
        """Recompute generalization from recon['per_session'] and inject it.

        Returns True iff a trustworthy score was computed and back-filled.
        Never raises — every failure path warns and returns False.
        """
        try:
            explicit, xtime_excludes = build_tier_index(
                Path(self.gen_config_root),
                split=self.gen_split,
                selection_family=self.gen_selection_family,
            )
            result, table = compute_from_per_session_reports(
                recon["per_session"],
                explicit_tiers=explicit,
                xtime_excludes=xtime_excludes,
                baseline_tier=self.gen_baseline_tier,
            )
        except Exception as exc:
            log.warning("generalization fallback skipped (%s): %s", type(exc).__name__, exc)
            return False

        tiers = dict(Counter(row["tier"] for row in table))
        if result is None:
            log.warning(
                "generalization fallback skipped: no usable baseline/OOD metrics "
                "(tiers=%s); set gen_split/gen_selection_family or disable "
                "generalization_fallback",
                tiers,
            )
            return False

        summary = result.summary
        log.info(
            "generalization recomputed from per_session: score=%.4f "
            "(%d retention terms; tiers=%s)",
            summary["generalization_score"],
            summary["n_retention_terms"],
            tiers,
        )
        recon["generalization"] = result.to_dict()
        return True


def _downstream_provenance(view: DownstreamScoreView | None) -> dict | None:
    if view is None:
        return None
    return {
        "split": view.split,
        "complete": view.complete,
        **view.provenance,
    }


_MLFLOW_URI_PREFIXES = ("mlflow-artifacts:", "mlflow:")


def _run_meta(report_path: str | None) -> dict | None:
    """``run_meta.json`` from the folder of a local report (robustness ``eval_report.json``
    or recon ``results.json``). A missing file gives ``None`` (reported later as "could NOT
    check"); an MLflow URI is not looked into."""
    if not report_path or report_path.startswith(_MLFLOW_URI_PREFIXES):
        return None
    run_meta_file = Path(report_path).parent / "run_meta.json"
    return json.loads(run_meta_file.read_text()) if run_meta_file.is_file() else None


def _robustness_run_files(report_path: str | None) -> tuple[list[str] | None, dict | None]:
    """``(session ids from results_per_session.csv, run_meta.json)`` from the folder of a
    local robustness ``eval_report.json``; ``None`` for a missing file (see ``_run_meta``)."""
    if not report_path or report_path.startswith(_MLFLOW_URI_PREFIXES):
        return None, None
    session_ids = None
    sessions_csv = Path(report_path).parent / "results_per_session.csv"
    if sessions_csv.is_file():
        with open(sessions_csv, newline="") as f:
            session_ids = sorted({row["session_id"] for row in csv.DictReader(f)})
    return session_ids, _run_meta(report_path)


def _robustness_settings(report: dict | None, run_meta: dict | None = None) -> dict | None:
    """Recorded settings of a robustness run: from the ``config`` of its ``eval_report.json``
    (``add_robustness_report_fields``), then from its ``run_meta.json`` for what the config
    lacks. A report that does not record ``scored_window_seconds`` gets it derived from its
    window settings; it stays unknown when they do not tell. ``None`` without a report."""
    if report is None:
        return None
    report_config = report.get("config") or None
    settings = recorded_settings(
        report_config,
        run_meta,
        _run_geometry((run_meta or {}).get("tokenizer_config"), run_meta),
    )
    return with_scored_window(settings, report_config, run_meta)


def _run_geometry(tokenizer_config: dict | None, run_meta: dict | None) -> dict | None:
    """The eval-input geometry (bin size, window) of a run: each value from its tokenizer
    config, else from the ``data_config`` in its ``run_meta.json`` (an eval whose tokenizer
    config failed to serialize saves only the tokenizer ``KIND``; its data config holds the
    same values)."""
    geometry = input_geometry(tokenizer_config)
    fallback = input_geometry((run_meta or {}).get("data_config"))
    if fallback is None:
        return geometry
    return {**fallback, **(geometry or {})}


def _build_output_dict(
    scores: CategoryScores, job: NTTAggregationJob, generalization_source: str
) -> dict[str, Any]:
    from datetime import datetime

    def _diag_to_dict(d: ColumnDiagnostics | None, clip: float | None) -> dict | None:
        if d is None:
            return None
        return {
            "score": d.score,
            "n_metrics": d.n_metrics,
            "normalization": d.normalization,
            "score_clip": clip,
            # How the column resolved: the decoding family means (poyo/ridge), the
            # latent bucket's M1/M2 resolution. {} for columns that record none.
            "provenance": d.provenance,
            "per_metric": {
                name: {
                    "normalized": ms.normalized,
                    "clipped": ms.clipped,
                    "raw_value": ms.raw_value,
                    "direction": ms.direction,
                }
                for name, ms in d.per_metric.items()
            },
        }

    diag = scores.diagnostics
    lsr_diag = diag.get("latent_structure_and_robustness")
    robustness_normalization = lsr_diag.provenance if lsr_diag is not None else {}
    return {
        "scores": {
            "single_neuron_recon": scores.single_neuron_recon,
            "population_structure": scores.population_structure,
            "latent_structure_and_robustness": scores.latent_structure_and_robustness,
            "decoding": scores.decoding,
            # The two family scores ``decoding`` averages (not composite columns).
            "decoding_poyo": scores.decoding_poyo,
            "decoding_linear": scores.decoding_linear,
            "downstream_encoding": scores.downstream_encoding,
            "generalization": scores.generalization,
            "compression": scores.compression,
        },
        "diagnostics": {
            "single_neuron_recon": _diag_to_dict(diag.get("single_neuron_recon"), job.score_clip),
            "population_structure": _diag_to_dict(
                diag.get("population_structure"), job.score_clip
            ),
            # The latent bucket uses bounded [0,1] transforms, not Poisson + clip.
            "latent_structure_and_robustness": _diag_to_dict(lsr_diag, None),
            "decoding": _diag_to_dict(diag.get("decoding"), None),
            "downstream_encoding": _diag_to_dict(diag.get("downstream_encoding"), None),
            "generalization": None,
            "compression": None,
        },
        "metadata": {
            # Raw M1/M2/structure values of latent_structure_and_robustness are mapped onto
            # [0,1] via bounded transforms (``latent_bucket_transforms``).
            "schema_version": AGGREGATION_SCHEMA_VERSION,
            "ckpt_uri": job.ckpt_uri,
            "inputs": {
                "recon": job.recon_path,
                "m1": job.m1_path,
                "m2": job.m2_path,
                "encoding": job.encoding_path,
                "decoding": job.decoding_path,
            },
            "poisson_reference": job.poisson_reference,
            "robustness_normalization": robustness_normalization,
            "latent_bucket_transforms": {
                "m1": "retention=1/(1+d)",
                "structure": "max(corr,0)",
                "m2": "max(acg,0)",
            },
            "include_m2": job.include_m2,
            "m2_metric": job.m2_metric,
            "m2_epsilon": job.m2_epsilon,
            "score_clip": job.score_clip,
            "compression_metric": job.compression_metric,
            "generalization_source": generalization_source,
            "computed_at": datetime.now(UTC).isoformat(),
        },
    }


def _log_to_mlflow(output_file: Path, job: NTTAggregationJob) -> None:
    """Publish the written ``category_scores.json``. A ``tracking_uri`` in the job's own
    config is honoured; otherwise the run goes to ``MLFLOW_TRACKING_URI``."""
    cfg = job.mlflow_logging
    if cfg.tracking_uri:
        mlflow.set_tracking_uri(cfg.tracking_uri)
    published = publish_category_scores(
        output_file,
        experiment=cfg.experiment_name,
        run_name=cfg.run_name or Path(job.output_path).name,
        tags=cfg.tags,
    )
    log.info(
        "category scores %s MLflow run %s",
        "already in" if published.reused else "logged to",
        published.url,
    )
