"""Standalone evaluation job.

Loads a trained tokenizer from a Lightning checkpoint, runs the
evaluation pipeline on validation (or test) data, and prints/saves
results.  Does NOT require a Lightning Trainer, loss function, or
optimizer.
"""

import gc
import json
import math
import tempfile
import time
import typing as T
import warnings
from pathlib import Path

import numpy as np
import pydantic as pdt
import torch

# ── Warning filters ──────────────────────────────────────────────────
# DataLoader worker count: not actionable (we can't change SLURM CPU
# allocation mid-run), and fires every time a dataloader is created.
warnings.filterwarnings(
    "once",
    message=r".*DataLoader will create \d+ worker processes.*",
    category=UserWarning,
)
# NumPy nanmean/nanstd on all-NaN slices during Poisson realisation
# averaging — expected for metrics that are skipped on some sessions.
warnings.filterwarnings("ignore", message=r".*Mean of empty slice.*")
warnings.filterwarnings("ignore", message=r".*Degrees of freedom <= 0 for slice.*")
# Matplotlib tight_layout sometimes can't fit all decorations — cosmetic only.
warnings.filterwarnings("ignore", message=r".*Tight layout not applied.*")

from ntt.contrib import TokenizerSpecKind
from ntt.core.data.datamodule import NTTDataModule, NTTDataModuleConfig
from ntt.core.model import load_tokenizer_from_checkpoint
from ntt.core.model.tokenizer import BaseTokenizer, TokenizerSpec, create_tokenizer
from ntt.evaluation.aggregation.comparison import (
    POPULATION_MEAN_RATE_ERROR_KEY,
    MetricComparison,
    compute_normalized_comparisons,
    extract_normalized_scores,
    with_alternative_floors,
)
from ntt.evaluation.aggregation.summary import aggregate_by_category
from ntt.evaluation.compression import CompressionResult, compute_compression
from ntt.evaluation.config import EvaluationConfig, validate_config
from ntt.evaluation.config.schema import AggregationStrategy
from ntt.evaluation.core.types import EvaluationReport, MetricCategory
from ntt.evaluation.generalization import compute_from_session_tiers
from ntt.evaluation.mlflow_logging import MLflowLoggingConfig
from ntt.evaluation.neuron_selection import (
    NeuronSelection,
    all_neuron_data_config,
    check_neuron_selection,
    describe_neuron_selection,
    fixed_n_neurons,
    warn_padded_mode,
)
from ntt.evaluation.tiering import (
    TierMembership,
    assign_sessions_to_tiers,
    parse_tier_membership,
)
from ntt.evaluation.utils import (
    aggregate_segment_reports,
    aggregate_session_reports,
    run_validation_evaluation_per_session,
)
from ntt.evaluation.utils.config_provenance import (
    composed_config_yaml,
    config_subtree_to_jsonable,
    config_to_jsonable,
    hydra_task_overrides,
    write_config_provenance_sidecar,
)
from ntt.evaluation.utils.detailed_results import write_metrics_detailed_results_jsonl
from ntt.evaluation.utils.neuron_covering import (
    CompressNLatentStrategy,
    make_oom_safe_tokenize_fn,
)
from ntt.evaluation.utils.serialization import (
    _json_default as _eval_json_default,
)
from ntt.evaluation.utils.serialization import (
    replace_non_finite_floats,
)
from ntt.evaluation.utils.window_contract import check_datamodule_window
from ntt.jobs.base import Locals
from ntt.jobs.ntt_base import NTTBaseJob
from ntt.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def _should_log_eval_run_to_mlflow(
    report: EvaluationReport,
    *,
    run_name_suffix: str | None,
) -> bool:
    """Return whether this eval run should create an MLflow run.

    Skip the SNR side-pass when ``repeats_snr`` never materialized into the
    aggregated report (for example because the interval-windowed path was
    skipped entirely). In that case creating an MLflow run is misleading:
    there is no SNR result to upload.
    """
    if run_name_suffix == "__snr" and "repeats_snr" not in report.results:
        return False
    return True


def _maybe_log_rate_override(consumer_rate: float | None, baseline_rate: float | None) -> None:
    """Emit a warning when the consumer's configured Poisson rate differs
    from the baseline's, before the consumer silently adopts the baseline.

    Silent when the consumer left the rate unset (None) or when the rates
    already agree. Helps users notice that a hand-set
    ``poisson_rate_bin_ms`` is being overridden by the loaded baseline
    (which is required behaviour — the rate must match the baseline — but
    deserves a visible signal rather than a silent assignment).
    """
    if consumer_rate is None:
        return
    if baseline_rate is None:
        return
    if float(consumer_rate) == float(baseline_rate):
        return
    log.warning(
        f"Overriding configured poisson_rate_bin_ms={consumer_rate} "
        f"with baseline rate_estimation_bin_ms={baseline_rate} "
        f"(must match the baseline's rate)."
    )


def _validate_poisson_reference_hard_fields(
    poisson_reference: str | Path,
    *,
    bin_size_ms: float,
    window_length_seconds: float,
    split: T.Literal["val", "test"],
    max_eval_seconds: float | None,
    decoding_bypass_fields: list[str],
    n_realisations: int,
    seed: int,
    poisson_rate_bin_ms: float | None = None,
    neuron_selection: NeuronSelection | None = None,
    strict_neuron_selection: bool = True,
):
    """Hard-field-only validation of a poisson_reference artifact.

    Loads the artifact and runs ``validate_against`` with the consumer
    metadata. ``sessions`` is populated from the baseline itself so the
    session-set superset check trivially holds — the real consumer-vs-
    baseline session check is deferred to ``_run_poisson_comparison``,
    which knows the consumer's actual session list only after
    ``run_validation_evaluation_per_session`` has produced it.

    Used as a fail-fast hook before the (slow) tokenizer eval, so an
    incompatible baseline raises in seconds instead of after 60-100 minutes
    of compute.

    Returns the loaded artifact. Raises ``ValueError`` on hard-field
    mismatch; emits a rate-override warning when the consumer's
    configured ``poisson_rate_bin_ms`` differs from the baseline's
    (mirrors the late-path adoption logic).
    """
    from ntt.evaluation.artifact import EvalArtifact, make_metadata

    artifact = EvalArtifact.load_json(poisson_reference)
    _maybe_log_rate_override(poisson_rate_bin_ms, artifact.metadata.rate_estimation_bin_ms)
    consumer_meta = make_metadata(
        kind="poisson_baseline",
        bin_size_ms=bin_size_ms,
        window_length_seconds=window_length_seconds,
        rate_estimation_bin_ms=artifact.metadata.rate_estimation_bin_ms,
        split=split,
        max_eval_seconds=max_eval_seconds,
        decoding_bypass_fields=decoding_bypass_fields,
        sessions=list(artifact.metadata.sessions),
        n_realisations=n_realisations,
        seed=seed,
        neuron_selection=neuron_selection,
    )
    artifact.validate_against(consumer_meta, strict_neuron_selection=strict_neuron_selection)
    return artifact


def _load_stashed_original(stash_path: str) -> np.ndarray:
    """Load a stashed-original spike array as a read-only mmap view.

    The numpy file at ``stash_path`` was written by
    ``run_validation_evaluation_per_session`` (via ``stash_originals_dir``)
    during Poisson realisation 1. We map it read-only so that:

    1. The 32 GB total stash residency does not double into Python ndarray
       copies (saves ~25 GB peak RSS in the cs eval case).
    2. The OS page cache can evict pages under memory pressure.
    3. Children spawned via fork (parallel realisations) share the
       backing pages without copy-on-write overhead.
    """
    return np.load(stash_path, mmap_mode="r")


def _segment_time_bins(segment: T.Any) -> float:
    """A segment's weight in ``aggregate_segment_reports``: its number of time bins."""
    shape = (
        segment.original_shape
        if segment.original_shape is not None
        else tuple(segment.original.shape)
    )
    return float(shape[1])


class NTTEvalJob(
    NTTBaseJob,
    strict=False,
    frozen=False,
    extra="forbid",
    arbitrary_types_allowed=True,
):
    """Standalone evaluation job.

    Loads a tokenizer from a checkpoint, runs the full evaluation
    pipeline on validation or test data, and reports results.

    TODO: Consolidate arguments with base job
    """

    KIND: T.Literal["eval"] = "eval"
    applies_eval_window: T.ClassVar[bool] = True

    # Provide exactly one: ckpt_path (trained tokenizer) or tokenizer_config (untrained)
    ckpt_path: str | Path | None = None
    ckpt_weights_only: bool = False
    tokenizer_config: TokenizerSpecKind = None
    data_config: NTTDataModuleConfig

    # Evaluation config — accepts a raw dict so Hydra can pass the full
    # evaluation YAML (which includes ``lightning_eval`` keys).  Validated
    # and stripped in ``__enter__`` via ``validate_config()``.
    eval_config: dict[str, T.Any] | None = None

    # Optional
    device: str = "auto"
    output_path: str | Path | None = None
    split: T.Literal["val", "test"] = "val"
    aggregation_strategy: AggregationStrategy = AggregationStrategy.MEAN

    # Poisson baseline comparison
    compare_poisson: bool = False
    # Required whenever compare_poisson=True and no poisson_reference is supplied.
    poisson_rate_bin_ms: float | None = None
    poisson_seed: int = 42
    poisson_n_realisations: int = 5
    # Parallel realisations: None → auto from SLURM_CPUS_PER_TASK or os.cpu_count()
    poisson_n_workers: int | None = None
    # Pre-computed Poisson baseline to consume instead of recomputing.
    # Path, mlflow-artifacts:/ URI or hf://<owner>/<repo>/<file>[@<revision>] URI to a
    # poisson_baseline.json produced by NTTPoissonBaselineJob. Validated against this
    # eval's metadata at load time.
    poisson_reference: str | None = None
    # Use all neurons for Poisson baseline (instead of tokenizer's n_neurons).
    # The Poisson baseline doesn't have a fixed architecture, so it can use
    # all available neurons, giving a fairer comparison.
    poisson_use_all_neurons: bool = False

    # Neuron count of the dummy input that compute_compression measures on, for specs
    # without a fixed int n_neurons: n_neurons=False (whose compression rate depends on
    # it) or a Dist (must lie in its range). Ignored with a warning for a spec that fixes
    # an int count. None keeps the spec's own count (or the dummy default of 8).
    compression_n_neurons: int | None = pdt.Field(default=None, ge=1)

    # Full-neuron evaluation
    evaluate_all_neurons: bool = True
    compress_n_latent_strategy: CompressNLatentStrategy = CompressNLatentStrategy.ALL_GROUPS

    # MLflow logging
    mlflow_logging: MLflowLoggingConfig = MLflowLoggingConfig()

    # Tiered evaluation — map tier name to composite selection yaml path.
    # When set, per-session results are also aggregated per tier.  Sessions
    # are matched by ``brainset/session_name`` from the composite yamls.
    tier_selections: dict[str, str] | None = None

    # Plot generation
    generate_plots: bool = False
    plot_seconds: float = 600.0
    plot_sessions: int = 10

    # Private runtime state
    _tokenizer: BaseTokenizer | None = pdt.PrivateAttr(default=None)
    _device: torch.device | None = pdt.PrivateAttr(default=None)
    _eval_config: EvaluationConfig | None = pdt.PrivateAttr(default=None)
    _spec: TokenizerSpec | None = pdt.PrivateAttr(default=None)
    _plot_poisson_label: str | None = pdt.PrivateAttr(default=None)
    _poisson_rate_bin_ms: float | None = pdt.PrivateAttr(default=None)
    _compression: CompressionResult | None = pdt.PrivateAttr(default=None)
    _hydra_overrides: list[str] = pdt.PrivateAttr(default_factory=list)
    _composed_config_yaml: str | None = pdt.PrivateAttr(default=None)
    _hydra_data_config: T.Any | None = pdt.PrivateAttr(default=None)
    # Units each session contributes to the metrics (set when the eval data is built).
    _neuron_selection: NeuronSelection | None = pdt.PrivateAttr(default=None)
    # Which Poisson reference normalised this run ("in_run" or "external") and the window it
    # was computed at; saved in results.json.
    _poisson_reference_kind: str | None = pdt.PrivateAttr(default=None)
    _poisson_reference_window_seconds: float | None = pdt.PrivateAttr(default=None)

    def process_config(self, cfg: T.Any | None = None) -> None:
        """Capture the full Hydra config for run metadata provenance."""
        self._hydra_overrides = hydra_task_overrides()
        self._composed_config_yaml = composed_config_yaml(cfg)
        self._hydra_data_config = config_subtree_to_jsonable(
            cfg,
            ["job.data_config", "data_config", "data"],
        )

    def _provenance_composed_config_yaml(self) -> str:
        """Return the composed config snapshot, warning on direct non-Hydra use."""
        if self._composed_config_yaml is not None:
            return self._composed_config_yaml
        log.warning(
            "Full composed Hydra config unavailable for eval run metadata; "
            "writing a job-scoped snapshot instead."
        )
        return f"job:\n  KIND: {self.KIND}\n  hydra_config_unavailable: true\n"

    def _provenance_data_config(self) -> T.Any:
        """Return the effective runtime data config used by the dataloader."""
        return self.data_config

    @pdt.model_validator(mode="after")
    def _check_ckpt_or_config(self) -> T.Self:
        has_ckpt = self.ckpt_path is not None
        has_config = self.tokenizer_config is not None
        if has_ckpt == has_config:
            raise ValueError(
                "Provide exactly one of ckpt_path (trained) or"
                " tokenizer_config (untrained), not both."
            )
        return self

    # ── Tier helpers ─────────────────────────────────────────────────

    @staticmethod
    def _parse_tier_membership(name: str, yaml_path: str | Path) -> TierMembership:
        """Parse a composite selection yaml into a :class:`TierMembership`.

        Supports both explicit ``sessions`` tiers and ``exclude_sessions``
        tiers (e.g. ``xtime``, the training-set complement).
        """
        return parse_tier_membership(name, yaml_path)

    def _build_tier_map(self) -> dict[str, TierMembership] | None:
        """Build tier_name -> TierMembership from ``tier_selections``."""
        if not self.tier_selections:
            return None
        tier_map: dict[str, TierMembership] = {}
        for tier_name, yaml_path in self.tier_selections.items():
            membership = self._parse_tier_membership(tier_name, yaml_path)
            if membership.is_empty():
                log.warning(f"Tier '{tier_name}': no sessions resolvable from {yaml_path}.")
                continue
            tier_map[tier_name] = membership
            n_inc = len(membership.include)
            n_excl_bs = len(membership.exclude_brainsets)
            log.info(
                f"Tier '{tier_name}': {n_inc} explicit sessions, "
                f"{n_excl_bs} exclude-based brainsets from {yaml_path}"
            )
        return tier_map if tier_map else None

    @staticmethod
    def _aggregate_by_tier(
        session_reports: dict[str, list],
        tier_map: dict[str, TierMembership],
        strategy: AggregationStrategy,
    ) -> dict[str, EvaluationReport]:
        """Aggregate per-session reports into per-tier reports.

        Each evaluated session is assigned to at most one tier (explicit
        ``include`` matches take priority over exclude-based matches).
        Sessions not matching any tier are skipped, so referencing a
        full-suite composite while evaluating a smaller subset works as a
        natural filter.
        """
        assignment = assign_sessions_to_tiers(session_reports.keys(), tier_map)
        tier_reports: dict[str, EvaluationReport] = {}
        for tier_name in tier_map:
            tier_session_reports = {
                sid: session_reports[sid]
                for sid in assignment.get(tier_name, set())
                if session_reports.get(sid)
            }
            if not tier_session_reports:
                log.warning(f"Tier '{tier_name}': no matching sessions found in eval results")
                continue
            tier_reports[tier_name] = aggregate_session_reports(
                tier_session_reports, strategy=strategy
            )
            log.info(f"Tier '{tier_name}': aggregated {len(tier_session_reports)} sessions")
        return tier_reports

    def _compute_generalization(
        self,
        per_session: dict[str, dict],
        tier_map: dict[str, TierMembership] | None,
    ) -> dict | None:
        """Compute the generalization retention score from per-session reports.

        ``per_session`` has the shape of ``results.json["per_session"]``. Scored per
        session, as the aggregation job rescores it from ``results.json``. Computed
        independently of where results are written, so the score reaches both
        ``results.json`` and MLflow even when ``output_path`` is unset. Returns
        ``None`` (logging the reason) when tiers aren't configured or no usable
        baseline/OOD metrics were found.
        """
        if not tier_map:
            log.info(
                "Skipping generalization score: tier_selections not configured. "
                "Set tier_selections.{xtime,xsession,xsubject,xsource} to enable."
            )
            return None
        assignment = assign_sessions_to_tiers(per_session.keys(), tier_map)
        session_tiers = {sid: tier for tier, sids in assignment.items() for sid in sids}
        try:
            result = compute_from_session_tiers(per_session, session_tiers, baseline_tier="xtime")
        except Exception as e:
            log.warning(f"Generalization score computation failed: {e}")
            return None
        if result is None:
            log.warning(
                "Skipping generalization score: no usable baseline ('xtime') or OOD metrics "
                f"(configured tiers: {sorted(tier_map)})."
            )
            return None
        return result.to_dict()

    # ── Context management ──────────────────────────────────────────

    @staticmethod
    def _resolve_ckpt_path(ckpt_path: str | Path) -> str | Path:
        """Download MLflow artifact if path starts with ``mlflow-artifacts:/``,
        and resolve a directory artifact down to its single .ckpt file."""
        from ntt.evaluation.utils.paths import resolve_artifact_path

        if str(ckpt_path).startswith("hf://"):
            from ntt.hub import materialize_checkpoint

            return materialize_checkpoint(str(ckpt_path))
        path_str = str(ckpt_path)
        if not (path_str.startswith("mlflow-artifacts:") or path_str.startswith("mlflow:")):
            return ckpt_path
        resolved = resolve_artifact_path(path_str)
        # MLflow may download a directory containing the .ckpt file
        if resolved.is_dir():
            ckpt_files = list(resolved.glob("*.ckpt"))
            if len(ckpt_files) == 1:
                return ckpt_files[0]
            elif len(ckpt_files) == 0:
                raise FileNotFoundError(f"No .ckpt file found in downloaded artifact: {resolved}")
            else:
                raise ValueError(
                    f"Multiple .ckpt files in downloaded artifact: {[f.name for f in ckpt_files]}"
                )
        return resolved

    def _resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def __enter__(self) -> T.Self:
        # Device
        self._device = self._resolve_device()
        log.info(f"Using device: {self._device}")

        # Tokenizer — either load from checkpoint or create from spec.
        # Must happen BEFORE super().__enter__() so the datamodule gets
        # a valid tokenizer spec (the base class passes it to NTTDataModule).
        if self.ckpt_path is not None:
            ckpt_path = self._resolve_ckpt_path(self.ckpt_path)
            log.info(f"Loading tokenizer from {ckpt_path}")
            self._tokenizer = load_tokenizer_from_checkpoint(
                ckpt_path,
                map_location=str(self._device),
                weights_only=self.ckpt_weights_only,
            )
            if self._tokenizer is None:
                raise ValueError(f"Checkpoint {self.ckpt_path} did not contain a tokenizer!")
            self._tokenizer.to(self._device)
            self._spec = self._tokenizer.spec
            self.tokenizer_config = T.cast(TokenizerSpecKind, self._spec)
        else:
            assert self.tokenizer_config is not None  # guaranteed by validator
            log.info(f"Creating untrained tokenizer from spec ({self.tokenizer_config.KIND})")
            self._spec = self.tokenizer_config
            self._tokenizer = create_tokenizer(self._spec)
            self._tokenizer.to(self._device)

        # Evaluation always uses deterministic transforms (fixed neuron subset,
        # no random shuffling) for reproducible results.
        if not self.data_config.deterministic_transforms:
            self.data_config.deterministic_transforms = True
            log.info("Eval job: set deterministic_transforms=True for reproducibility")

        # Eval window is the single source of truth: force the datamodule's
        # window to it and validate the tokenizer accepts it (fail loud
        # otherwise). MUST happen BEFORE super().__enter__() builds the
        # datamodule, because update_from_tokenizer_spec(unset_only=True) will
        # not overwrite an already-set window_length_seconds.
        from ntt.evaluation.utils.window_contract import apply_eval_window

        apply_eval_window(
            self.data_config, self.tokenizer_config, self._resolve_eval_window_seconds()
        )

        # Base handles seeding + datamodule creation (with a valid tokenizer_config)
        super().__enter__()

        # Parse eval config (strip lightning_eval, etc.)
        if self.eval_config is not None:
            self._eval_config = validate_config(self.eval_config)
        else:
            self._eval_config = EvaluationConfig()

        # Data
        stage = "validate" if self.split == "val" else "test"
        self.datamodule.setup(stage)

        return self

    # ── Full-neuron evaluation ─────────────────────────────────────

    def _build_all_neuron_eval(self) -> tuple:
        """Build dataloader and tokenizer_fn for full-neuron evaluation.

        Creates a separate DataModule with n_neurons=False and wraps the
        tokenizer for compress_n architectures.

        Returns:
            (dataloader, tokenizer_fn, snr_tokenizer_fn) ready for
            run_validation_evaluation_per_session.
        """
        from ntt.core.data.utils import n_neurons_per_session
        from ntt.evaluation.utils.neuron_covering import (
            NeuronCoveringWrapper,
            compute_eval_batch_size,
            oom_safe_forward,
        )
        from ntt.evaluation.utils.window_contract import apply_eval_window

        assert self._spec is not None
        assert self._tokenizer is not None
        assert self._device is not None
        assert self._eval_config is not None

        # DataModule with all neurons, capped at eval_config.max_neurons by the
        # shared EvalUnitSelection (seeded random subset per session) at load
        # time, so original and reconstruction see the same units and every
        # model and Poisson baseline see the same units for a session.
        all_neuron_config = all_neuron_data_config(self.data_config, self._eval_config)
        stage = "validate" if self.split == "val" else "test"

        config_n_neurons = fixed_n_neurons(self._spec)
        # compress_n specs with n_neurons=False (e.g. the perceiver) take any number
        # of units, so they need no covering.
        direct = not self._spec.compress_n or self._spec.n_neurons is False

        # Get max neuron count from the main datamodule's dataset
        # (which shares the same sessions) to avoid creating a throwaway DM.
        ds = self.datamodule.val_dataset if self.split == "val" else self.datamodule.test_dataset
        assert ds is not None, "datamodule.setup() must have created the eval dataset"
        neuron_counts = n_neurons_per_session(ds)
        max_neurons_in_dataset = max(neuron_counts.values())

        neuron_cap = self._eval_config.max_neurons
        if neuron_cap is not None:
            log.info(
                "Full-neuron eval: at most %d units per session, a seeded random subset "
                "where there are more (max in dataset: %d)",
                neuron_cap,
                max_neurons_in_dataset,
            )

        effective_max = (
            min(max_neurons_in_dataset, neuron_cap)
            if neuron_cap is not None
            else max_neurons_in_dataset
        )

        if direct:
            # compress_t: scale batch size for effective neuron count
            adjusted_bs = compute_eval_batch_size(
                config_batch_size=self.data_config.dataloader.batch_size,
                config_n_neurons=config_n_neurons,
                max_neurons=effective_max,
            )
            cap_note = (
                f", capped from {max_neurons_in_dataset}"
                if neuron_cap is not None and neuron_cap < max_neurons_in_dataset
                else ""
            )
            log.info(
                "Full-neuron eval (direct): batch_size %d -> %d (effective_max_neurons=%d%s)",
                self.data_config.dataloader.batch_size,
                adjusted_bs,
                effective_max,
                cap_note,
            )
            all_neuron_config.dataloader = all_neuron_config.dataloader.model_copy(
                update={"batch_size": adjusted_bs}
            )

        apply_eval_window(all_neuron_config, self._spec, self._resolve_eval_window_seconds())
        all_neuron_dm = NTTDataModule(all_neuron_config)
        self._check_scored_window(all_neuron_dm, "all-neuron eval")
        all_neuron_dm.setup(stage)
        self._neuron_selection = describe_neuron_selection(
            all_neuron_dm, self.split, self._eval_config
        )

        if direct:
            tokenizer = self._tokenizer

            @torch.no_grad()
            def _raw_recon(b):
                return tokenizer.reconstruct_from_input(b, return_latents=True)

            @torch.no_grad()
            def _compress_t_fn(batch) -> dict:
                result, _ = oom_safe_forward(
                    _raw_recon,
                    batch,
                    batch_size=adjusted_bs,
                )
                return result

            tok_fn = _compress_t_fn

            @torch.no_grad()
            def _snr_compress_t_fn(batch) -> dict:
                result, _ = oom_safe_forward(
                    _raw_recon,
                    batch,
                    batch_size=adjusted_bs,
                )
                return result

            snr_tok_fn = _snr_compress_t_fn
        else:
            # compress_n: wrap with NeuronCoveringWrapper
            assert config_n_neurons is not None, (
                "compress_n tokenizer must have a fixed int n_neurons"
            )
            cap_note = (
                f", capped from {max_neurons_in_dataset}"
                if neuron_cap is not None and neuron_cap < max_neurons_in_dataset
                else ""
            )
            log.info(
                "Full-neuron eval (compress_n): covering with n_neurons=%d "
                "(effective_max_neurons=%d%s, ~%d groups/session)",
                config_n_neurons,
                effective_max,
                cap_note,
                (effective_max + config_n_neurons - 1) // config_n_neurons,
            )
            tokenizer = self._tokenizer

            @torch.no_grad()
            def _inner_fn(batch) -> dict:
                # return_latents=True yields the latents dict
                return T.cast(dict, tokenizer.reconstruct_from_input(batch, return_latents=True))

            wrapper = NeuronCoveringWrapper(
                tokenizer_fn=_inner_fn,
                n_neurons=config_n_neurons,
                seed=self.data_config.seed,
                latent_strategy=self.compress_n_latent_strategy,
            )
            snr_wrapper = NeuronCoveringWrapper(
                tokenizer_fn=_inner_fn,
                n_neurons=config_n_neurons,
                seed=self.data_config.seed,
                latent_strategy=self.compress_n_latent_strategy,
            )

            @torch.no_grad()
            def _compress_n_fn(batch) -> dict:
                return wrapper(batch)

            tok_fn = _compress_n_fn
            snr_tok_fn = snr_wrapper

        dataloader = (
            all_neuron_dm.val_dataloader()
            if self.split == "val"
            else all_neuron_dm.test_dataloader()
        )

        return dataloader, tok_fn, snr_tok_fn

    # ── Run ──────────────────────────────────────────────────────────

    def _compute_compression(self) -> None:
        """Compute compression once into ``self._compression``.

        Skipped on failure (exotic spec / encode error); the rest of the eval is
        unaffected.
        """
        assert self._spec is not None
        assert self._tokenizer is not None
        try:
            compression = compute_compression(
                self._spec, self._tokenizer, n_neurons=self.compression_n_neurons
            )
            self._compression = compression
            rate_str = (
                "nan (vocab size or codes per token unknown)"
                if math.isnan(compression.compression_rate)
                else f"{compression.compression_rate:.2f}"
            )
            log.info(
                f"Compression: rate={rate_str}, "
                f"bits_per_token={compression.bits_per_token:.2f}, "
                f"n_neurons={compression.n_neurons}, "
                f"latent_dim_reduction={compression.latent_dim_reduction:.2f}"
            )
        except Exception as e:
            log.warning(f"Compression computation failed: {e}")
            self._compression = None

    def run(self) -> Locals:
        assert self._tokenizer is not None
        assert self._device is not None
        assert self._eval_config is not None
        assert self._spec is not None

        self._plot_poisson_label = None

        spec = self._spec
        bin_size_seconds = spec.bin_size_ms / 1000.0 if spec.bin_size_ms else None
        tokenizer = self._tokenizer
        device = self._device
        eval_window_seconds = self._resolve_eval_window_seconds()

        self._compute_compression()

        log.info("Running evaluation...")

        if self.evaluate_all_neurons:
            dataloader, tokenizer_fun, snr_tokenizer_fun = self._build_all_neuron_eval()
        else:
            dataloader = (
                self.datamodule.val_dataloader()
                if self.split == "val"
                else self.datamodule.test_dataloader()
            )

            @torch.no_grad()
            def _raw_tokenize(batch: T.Any) -> dict:
                # return_latents=True yields the latents dict
                return T.cast(dict, tokenizer.reconstruct_from_input(batch, return_latents=True))

            # Split batches that OOM (e.g. the perceiver: a per-neuron S4 over
            # batch_size x units sequences); batches that fit are unchanged.
            tokenizer_fun = make_oom_safe_tokenize_fn(_raw_tokenize)
            snr_tokenizer_fun = tokenizer_fun
            self._neuron_selection = describe_neuron_selection(
                self.datamodule, self.split, self._eval_config
            )
            warn_padded_mode(
                self._neuron_selection,
                spec_n_neurons=spec.n_neurons,
                max_neurons=self._eval_config.max_neurons,
                what="eval",
            )

        # Fail-fast (after the data is set up, so this run's neuron selection is
        # known): validate the poisson_reference's hard fields here so an
        # incompatible baseline kills the job in seconds rather than after
        # the slow tokenizer eval. Session-set ⊆ check stays in
        # _run_poisson_comparison (consumer sessions are only known then).
        if self.poisson_reference is not None:
            from ntt.evaluation.artifact import compute_decoding_bypass_fields
            from ntt.evaluation.pipeline import EvaluationPipeline

            _validate_poisson_reference_hard_fields(
                self.poisson_reference,
                bin_size_ms=float(spec.bin_size_ms),
                window_length_seconds=eval_window_seconds,
                split=self.split,
                max_eval_seconds=getattr(self._eval_config, "max_eval_seconds", None),
                decoding_bypass_fields=compute_decoding_bypass_fields(
                    EvaluationPipeline(config=self._eval_config).metrics
                ),
                n_realisations=self.poisson_n_realisations,
                seed=self.poisson_seed,
                poisson_rate_bin_ms=self.poisson_rate_bin_ms,
                neuron_selection=self._neuron_selection,
                strict_neuron_selection=self.evaluate_all_neurons,
            )
            log.info(
                f"Poisson reference at {self.poisson_reference} passed early hard-field validation"
            )

        session_reports = run_validation_evaluation_per_session(
            dataloader,
            (
                snr_tokenizer_fun
                if self._eval_config.interval_source is not None
                else tokenizer_fun
            ),
            data_type=spec.input_type,
            reconstructed_data_type=spec.output_type,
            bin_size_seconds=bin_size_seconds,
            token_bin_size_ms=spec.token_bin_size_ms,
            window_length_seconds=eval_window_seconds,
            eval_config=self._eval_config,
            device=device,
            tokenizer_spec=spec,
        )

        aggregated = aggregate_session_reports(session_reports, strategy=self.aggregation_strategy)

        # ── Early durability write ──────────────────────────────────
        # Persist the model-eval results here, before the high-OOM-risk
        # Poisson comparison phase. If anything below this point crashes,
        # this artifact still preserves the model eval (the most expensive
        # piece — typically 60-100+ minutes of compute).
        try:
            from ntt.evaluation.artifact import (
                EvalArtifact,
                compute_decoding_bypass_fields,
                make_metadata,
            )
            from ntt.evaluation.pipeline import EvaluationPipeline
            from ntt.evaluation.utils.paths import resolve_partial_output_dir

            model_per_session = {
                sid: aggregate_segment_reports(segs, self.aggregation_strategy).to_dict()
                for sid, segs in session_reports.items()
            }
            partial_meta = make_metadata(
                kind="model_eval",
                bin_size_ms=float(spec.bin_size_ms),
                window_length_seconds=eval_window_seconds,
                rate_estimation_bin_ms=None,
                split=self.split,
                max_eval_seconds=getattr(self._eval_config, "max_eval_seconds", None),
                # Computed from this eval's own pipeline so the metadata is
                # structurally consistent — even though kind="model_eval"
                # artifacts are not currently consumed via poisson_reference.
                decoding_bypass_fields=compute_decoding_bypass_fields(
                    EvaluationPipeline(config=self._eval_config).metrics
                ),
                sessions=sorted(session_reports.keys()),
                n_realisations=None,
                seed=None,
                neuron_selection=self._neuron_selection,
            )
            partial_path = resolve_partial_output_dir(self.output_path) / "model_eval_partial.json"
            EvalArtifact(
                metadata=partial_meta,
                per_session=model_per_session,
                aggregated=aggregated.to_dict(),
            ).dump_json(partial_path)
            log.info(f"Persisted partial model-eval results to {partial_path}")
        except Exception:  # noqa: BLE001
            log.warning(
                "Failed to persist partial model-eval results; "
                "the final results.json save at end of run still captures the "
                "same data. This early write only matters when later phases crash.",
                exc_info=True,
            )

        # ── Poisson baseline comparison ──────────────────────────────
        comparisons_vs_poisson: dict[str, MetricComparison] | None = None
        # {metric: {session_id: reference error}} for the metrics in ALTERNATIVE_FLOORS.
        alternative_floor_per_session: dict[str, dict[str, float]] = {}
        poisson_aggregated = None
        perfect_aggregated = None
        poisson_session_reports: dict | None = None
        if self.compare_poisson and self.evaluate_all_neurons and not self.poisson_use_all_neurons:
            log.info(
                "evaluate_all_neurons=True but poisson_use_all_neurons=False — "
                "forcing poisson_use_all_neurons=True so the Poisson baseline "
                "uses the same neuron population as the model evaluation."
            )
            self.poisson_use_all_neurons = True
        if self.compare_poisson and bin_size_seconds is not None:
            # Free the main eval dataloader (persistent workers) and move the
            # trained tokenizer off GPU before Poisson allocates its own.
            # All segment fields (including original) are stripped by
            # run_validation_evaluation_per_session; Poisson stashes
            # originals to disk for resampling.
            tokenizer.cpu()
            del dataloader
            gc.collect()
            torch.cuda.empty_cache()

            (
                comparisons_vs_poisson,
                poisson_aggregated,
                perfect_aggregated,
                poisson_session_reports,
            ) = self._run_poisson_comparison(session_reports, bin_size_seconds, aggregated)
            # firing_rate_error is scored against a population-mean-rate reference, not the
            # Poisson floor. Done here, so results.json, the printed table, MLflow and the
            # plots all show the same score.
            comparisons_vs_poisson, alternative_floor_per_session = with_alternative_floors(
                comparisons_vs_poisson,
                {
                    sid: [data["report"].results for _, data in segs]
                    for sid, segs in session_reports.items()
                },
                self.aggregation_strategy,
                segment_weights={
                    sid: [_segment_time_bins(segment) for segment, _ in segs]
                    for sid, segs in session_reports.items()
                },
            )

            # Move the trained tokenizer back to device for downstream plots.
            tokenizer.to(device)

        # ── Tier aggregation ────────────────────────────────────────
        tier_map = self._build_tier_map()
        tier_reports: dict[str, EvaluationReport] | None = None
        tiered_per_session: dict[str, dict] = {}
        if tier_map:
            tier_reports = self._aggregate_by_tier(
                session_reports, tier_map, self.aggregation_strategy
            )
            tiered_per_session = {
                sid: aggregate_segment_reports(segments, self.aggregation_strategy).to_dict()
                for sid, segments in session_reports.items()
                if segments
            }

        # Generalization retention score — computed once here (independent of
        # output_path) so it reaches both the local results.json and MLflow.
        generalization_block = self._compute_generalization(tiered_per_session, tier_map)

        self._print_results(
            session_reports,
            aggregated,
            comparisons_vs_poisson,
            poisson_aggregated,
            perfect_aggregated,
        )

        # Save locally whenever an output dir resolves (output_path or the
        # Hydra run dir). The written files are mirrored to MLflow below so
        # nothing computed is lost to the local-save gate.
        saved_artifacts: list[Path] = []
        if self._get_output_dir() is not None:
            saved_artifacts = self._save_results(
                aggregated,
                session_reports,
                comparisons_vs_poisson,
                poisson_aggregated,
                perfect_aggregated,
                poisson_session_reports,
                tier_reports=tier_reports,
                generalization_block=generalization_block,
                alternative_floor_per_session=alternative_floor_per_session,
            )

        # ── MLflow logging ──────────────────────────────────────────
        mlflow_run_id: str | None = None
        if self.mlflow_logging.enabled:
            if not _should_log_eval_run_to_mlflow(
                aggregated,
                run_name_suffix=self.mlflow_logging.run_name_suffix,
            ):
                log.info(
                    "Skipping MLflow logging for this eval pass because no "
                    "repeats_snr result was computed."
                )
            else:
                try:
                    from ntt.core.model.checkpoint_io import resolve_checkpoint_for_eval
                    from ntt.evaluation.mlflow_logging import (
                        MLflowEvalLogger,
                        resolve_eval_run_name,
                    )

                    with tempfile.TemporaryDirectory(prefix="ntt_eval_ckpt_") as tmpdir:
                        ckpt_local_path = resolve_checkpoint_for_eval(
                            ckpt_path=self.ckpt_path,
                            live_tokenizer=self._tokenizer,
                            live_spec=self._spec,
                            always_copy=self.mlflow_logging.always_copy_checkpoint,
                            tmpdir=Path(tmpdir),
                        )

                        resolved_run_name = resolve_eval_run_name(
                            explicit=self.mlflow_logging.run_name,
                            ckpt_path=self.ckpt_path,
                            spec_kind=self._spec.KIND,
                            tracking_uri=self.mlflow_logging.tracking_uri,
                        )
                        mlflow_logger = MLflowEvalLogger(
                            experiment_name=self.mlflow_logging.experiment_name,
                            tracking_uri=self.mlflow_logging.tracking_uri,
                            run_name=resolved_run_name,
                            run_name_suffix=self.mlflow_logging.run_name_suffix,
                            metric_prefix=self.mlflow_logging.metric_prefix,
                        )
                        mlflow_run_id = mlflow_logger.log(
                            report=aggregated,
                            per_session_reports={
                                sid: aggregate_segment_reports(segments, self.aggregation_strategy)
                                for sid, segments in session_reports.items()
                                if segments
                            },
                            tier_reports=tier_reports,
                            comparisons=comparisons_vs_poisson,
                            generalization=generalization_block,
                            extra_artifacts=saved_artifacts,
                            checkpoint_path=(str(self.ckpt_path) if self.ckpt_path else None),
                            tags={
                                "ntt.eval.split": self.split,
                                "ntt.eval.aggregation_strategy": self.aggregation_strategy.value,
                                **self.mlflow_logging.tags,
                            },
                            checkpoint_local_path=(
                                ckpt_local_path if self.mlflow_logging.upload_checkpoint else None
                            ),
                        )
                        log.info(f"Eval results logged to MLflow run: {mlflow_run_id}")
                except Exception:
                    log.warning("Failed to log eval results to MLflow", exc_info=True)

        # ── Plot generation ──────────────────────────────────────────
        if self.generate_plots:
            plots_output_dir = self._generate_plots(
                tokenizer,
                bin_size_seconds,
                session_reports,
                comparisons_vs_poisson,
                aggregated,
                poisson_aggregated,
                perfect_aggregated,
                poisson_session_reports,
            )
            self._upload_plots_to_mlflow(plots_output_dir, mlflow_run_id)

        return locals()

    def _upload_plots_to_mlflow(self, plots_output_dir: Path, mlflow_run_id: str | None) -> bool:
        """Upload generated plots to the MLflow run. Returns True if uploaded.

        Gated on a *resolvable* output dir (``_get_output_dir()``), not on
        ``output_path`` directly — mirroring ``_save_results`` so plots reach
        MLflow even when ``output_path`` is unset. Skips (rather than uploading
        the cwd fallback) when no output dir resolves.
        """
        if mlflow_run_id is None:
            return False
        if self._get_output_dir() is None:
            log.info("Skipping MLflow plot upload: no resolvable output directory.")
            return False
        try:
            import mlflow

            if self.mlflow_logging.tracking_uri:
                mlflow.set_tracking_uri(self.mlflow_logging.tracking_uri)
            with mlflow.start_run(run_id=mlflow_run_id):
                mlflow.log_artifacts(str(plots_output_dir), artifact_path="plots")
            log.info(f"Plots logged to MLflow run: {mlflow_run_id}")
            return True
        except Exception:
            log.warning("Failed to log plots to MLflow", exc_info=True)
            return False

    def _get_output_dir(self) -> Path | None:
        """Resolve the local results directory.

        Uses ``output_path`` when set, otherwise falls back to the Hydra run
        directory so eval artifacts (results.json, run_meta.json, ...) always
        land somewhere and are never lost just because ``output_path`` wasn't
        set explicitly. Returns ``None`` only outside a Hydra run with no
        ``output_path`` (e.g. unit tests), in which case local saving is skipped.
        """
        if self.output_path is not None:
            output_dir = Path(self.output_path)
        else:
            try:
                from hydra.core.hydra_config import HydraConfig

                output_dir = Path(HydraConfig.get().runtime.output_dir)
            except Exception:
                return None

        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _tokenizer_legend_short_name(self) -> str:
        """Short legend / dict key for plots, e.g. ``wanneural (33ms)``."""
        assert self._spec is not None
        spec = self._spec
        if spec.bin_size_ms is False:
            return spec.KIND
        return f"{spec.KIND} ({float(spec.bin_size_ms):g}ms)"

    def _tokenizer_checkpoint_caption(self) -> str | None:
        """Resolved checkpoint path for behavioral figure suptitles; ``None`` if untrained."""
        if self.ckpt_path is None:
            return None
        if str(self.ckpt_path).startswith("hf://"):
            return str(self.ckpt_path)
        return str(Path(self.ckpt_path).resolve())

    @staticmethod
    def _to_eval_data(original: torch.Tensor | np.ndarray, bin_size_seconds: float):
        from ntt.evaluation.core.types import DataType, EvalData

        if isinstance(original, torch.Tensor):
            original_np = original.cpu().numpy()
        else:
            original_np = np.asarray(original)

        duration = original_np.shape[1] * bin_size_seconds
        return EvalData(
            binned_spikes=original_np,
            bin_size_seconds=bin_size_seconds,
            duration=duration,
            data_type=DataType.SPIKES,
        )

    # ── Poisson comparison ──────────────────────────────────────────

    def _run_poisson_comparison(
        self,
        session_reports: dict,
        bin_size_seconds: float,
        model_aggregated: T.Any,
    ) -> tuple[dict[str, MetricComparison], T.Any, T.Any, dict]:
        """Run Poisson floor and perfect ceiling comparisons.

        Uses a full ``PoissonBaselineTokenizer`` through the standard
        ``run_validation_evaluation_per_session`` pipeline for the first
        realisation (to get latents, behavioral data, and all metrics).
        Additional realisations only re-run the Poisson sampling + metrics
        on the already-collated segments, avoiding redundant dataloader
        iteration and collation.

        When ``poisson_n_realisations > 1``, averages metric values across
        realisations and stores the cross-realisation std in each
        ``MetricResult.metadata["realisation_std"]``.

        Returns:
            (comparisons, poisson_aggregated, perfect_aggregated, poisson_session_reports)
        """
        eval_window_seconds = self._resolve_eval_window_seconds()
        # ── Reference path ───────────────────────────────────────────
        # If a precomputed baseline is configured, load + validate + filter +
        # re-aggregate, and return without running any Poisson computation.
        if self.poisson_reference is not None:
            from ntt.evaluation.artifact import (
                EvalArtifact,
                aggregate_per_session_dicts,
                compute_decoding_bypass_fields,
                make_metadata,
                reconstruct_session_reports_from_per_session,
            )
            from ntt.evaluation.pipeline import EvaluationPipeline
            from ntt.evaluation.utils.paths import resolve_artifact_path

            resolved_reference = Path(resolve_artifact_path(str(self.poisson_reference)))
            if resolved_reference.is_dir():
                candidate = resolved_reference / "poisson_baseline.json"
                if not candidate.is_file():
                    raise ValueError(
                        "poisson_reference points to a directory but no "
                        f"poisson_baseline.json was found in {resolved_reference}"
                    )
                resolved_reference = candidate

            artifact = EvalArtifact.load_json(resolved_reference)
            # Adopt the baseline's rate (required to match it). Warn first if
            # the consumer had a different rate configured — silent overwrite
            # is the spec but it shouldn't be invisible.
            _maybe_log_rate_override(
                self.poisson_rate_bin_ms, artifact.metadata.rate_estimation_bin_ms
            )
            self._poisson_rate_bin_ms = artifact.metadata.rate_estimation_bin_ms
            self._poisson_reference_kind = "external"
            self._poisson_reference_window_seconds = artifact.metadata.window_length_seconds
            consumer_sessions = sorted(session_reports.keys())
            # Compute the consumer's bypass-fields from its own EvaluationPipeline
            # — NOT by copying the baseline's value into our metadata, which
            # would make the validate_against hard-field check tautological.
            assert self._eval_config is not None and self._spec is not None
            consumer_bypass_fields = compute_decoding_bypass_fields(
                EvaluationPipeline(config=self._eval_config).metrics
            )
            consumer_meta = make_metadata(
                kind="poisson_baseline",
                bin_size_ms=float(self._spec.bin_size_ms),
                window_length_seconds=eval_window_seconds,
                rate_estimation_bin_ms=self._poisson_rate_bin_ms,
                split=self.split,
                max_eval_seconds=getattr(self._eval_config, "max_eval_seconds", None),
                decoding_bypass_fields=consumer_bypass_fields,
                sessions=consumer_sessions,
                n_realisations=self.poisson_n_realisations,
                seed=self.poisson_seed,
                neuron_selection=self._neuron_selection,
            )
            artifact.validate_against(
                consumer_meta, strict_neuron_selection=self.evaluate_all_neurons
            )
            log.info(
                f"Loaded Poisson reference from {self.poisson_reference} "
                f"({len(artifact.metadata.sessions)} baseline sessions; "
                f"{len(consumer_sessions)} consumer sessions)"
            )
            if artifact.perfect_per_session is None:
                raise ValueError(
                    f"Poisson reference at {self.poisson_reference} is missing "
                    f"perfect_per_session — was it produced by NTTPoissonBaselineJob?"
                )
            consumer_set = set(consumer_sessions)
            filtered_poisson = {
                sid: r for sid, r in artifact.per_session.items() if sid in consumer_set
            }
            filtered_perfect = {
                sid: r for sid, r in artifact.perfect_per_session.items() if sid in consumer_set
            }
            # Filter weights to consumer subset; None → equal-weight fallback
            # inside aggregate_per_session_dicts (with a warning when
            # WEIGHTED_MEAN is in use).
            filtered_weights = (
                {sid: w for sid, w in artifact.per_session_weights.items() if sid in consumer_set}
                if artifact.per_session_weights is not None
                else None
            )
            poisson_aggregated = aggregate_per_session_dicts(
                filtered_poisson, self.aggregation_strategy, weights=filtered_weights
            )
            perfect_aggregated = aggregate_per_session_dicts(
                filtered_perfect, self.aggregation_strategy, weights=filtered_weights
            )
            perfect_aggregated = self._inject_known_perfect_ceiling_metrics(
                perfect_aggregated, model_aggregated
            )
            comparisons = compute_normalized_comparisons(
                model_aggregated, poisson_aggregated, perfect_aggregated
            )
            # Reconstruct a synthetic per-session/per-segment structure so
            # downstream consumers (results.json save, behavioral plots) get
            # non-empty data. Each session collapses to a single virtual
            # segment — segment-granular features are degraded vs the inline
            # path.
            reconstructed_session_reports = reconstruct_session_reports_from_per_session(
                filtered_poisson
            )
            log.info("Poisson reference applied; skipping Poisson computation.")
            return (
                comparisons,
                poisson_aggregated,
                perfect_aggregated,
                reconstructed_session_reports,
            )

        from ntt.contrib.tokenizer.poisson_baseline import (
            PoissonBaselineTokenizer,
            PoissonBaselineTokenizerSpec,
            format_poisson_tokenizer_display_name,
        )
        from ntt.evaluation.core.types import MetricResult
        from ntt.evaluation.pipeline import EvaluationPipeline

        assert self._eval_config is not None
        assert self._spec is not None
        assert self._device is not None

        n_realisations = self.poisson_n_realisations
        bin_size_ms = bin_size_seconds * 1000.0

        spec_bin = self._spec.bin_size_ms
        if spec_bin is False:
            raise ValueError(
                "Poisson comparison requires a tokenizer spec with fixed bin_size_ms."
            )
        if self.poisson_rate_bin_ms is not None:
            poisson_rate_ms = float(self.poisson_rate_bin_ms)
        else:
            token_bin_size_ms = getattr(self._spec, "token_bin_size_ms", None)
            if token_bin_size_ms is None:
                raise ValueError(
                    "compare_poisson=True requires either job.poisson_rate_bin_ms "
                    "or a tokenizer spec with token_bin_size_ms."
                )
            try:
                poisson_rate_ms = float(token_bin_size_ms)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "compare_poisson=True could not infer poisson_rate_bin_ms "
                    f"from tokenizer token_bin_size_ms={token_bin_size_ms!r}."
                ) from exc

        log.info(
            f"Running Poisson floor + perfect ceiling comparison "
            f"({n_realisations} Poisson realisation(s))..."
        )

        # --- Realisation 0: full pipeline (dataloader + collation + metrics) ---
        # Resolve window_length to a concrete float — variable_t tokenizers (e.g.
        # CALM trained with a UniformDist window) expose a Dist here, which the
        # Poisson spec rejects. effective_window_length_seconds returns Dist.max
        # for distributions, the value itself for floats.
        poisson_spec = PoissonBaselineTokenizerSpec(
            window_length_seconds=eval_window_seconds,
            bin_size_ms=self._spec.bin_size_ms,
            rate_estimation_bin_ms=poisson_rate_ms,
            seed=self.poisson_seed,
        )
        self._plot_poisson_label = format_poisson_tokenizer_display_name(poisson_spec)
        self._poisson_rate_bin_ms = poisson_rate_ms
        self._poisson_reference_kind = "in_run"
        self._poisson_reference_window_seconds = eval_window_seconds
        poisson_tokenizer = PoissonBaselineTokenizer(poisson_spec).to(self._device)
        poisson_tokenizer.eval()

        @torch.no_grad()
        def _poisson_fn(
            batch: T.Any,
            *,
            _poisson_tokenizer: PoissonBaselineTokenizer = poisson_tokenizer,
        ) -> T.Any:
            return _poisson_tokenizer.reconstruct_from_input(batch, return_latents=True)

        # Build dataloader for Poisson — optionally with all neurons
        if self.poisson_use_all_neurons:
            from ntt.evaluation.utils.window_contract import apply_eval_window

            assert self._eval_config is not None
            poisson_data_config = all_neuron_data_config(
                self.data_config, self._eval_config, poisson_spec
            )
            apply_eval_window(
                poisson_data_config, poisson_spec, self._resolve_eval_window_seconds()
            )
            poisson_dm = NTTDataModule(poisson_data_config)
            check_datamodule_window(
                poisson_dm, self._resolve_eval_window_seconds(), what="in-run Poisson baseline"
            )
            stage = "validate" if self.split == "val" else "test"
            poisson_dm.setup(stage)
            # The normalization is only valid if the Poisson baseline scores the
            # same units as the model.
            check_neuron_selection(
                self._neuron_selection,
                describe_neuron_selection(poisson_dm, self.split, self._eval_config),
                what="in-run Poisson baseline",
                strict=self.evaluate_all_neurons,
            )
            dataloader = (
                poisson_dm.val_dataloader()
                if self.split == "val"
                else poisson_dm.test_dataloader()
            )
            log.info("Poisson baseline using all neurons (separate datamodule)")
        else:
            dataloader = (
                self.datamodule.val_dataloader()
                if self.split == "val"
                else self.datamodule.test_dataloader()
            )
        # Stash original numpy arrays to disk so they survive strip_original()
        # inside run_validation_evaluation_per_session.  We need them for
        # Poisson resampling (realisations 1+) and the perfect ceiling.
        _originals_tmpdir = tempfile.mkdtemp(prefix="ntt_poisson_originals_")
        log.info(
            f"Poisson realisation 1/{n_realisations}: full pipeline "
            f"(dataloader + collation + all metrics inc. correlograms/downstream)..."
        )
        _t0 = time.perf_counter()
        try:
            first_reports = run_validation_evaluation_per_session(
                dataloader,
                _poisson_fn,
                data_type=poisson_spec.input_type,
                reconstructed_data_type=poisson_spec.output_type,
                bin_size_seconds=bin_size_seconds,
                token_bin_size_ms=poisson_spec.token_bin_size_ms,
                window_length_seconds=eval_window_seconds,
                eval_config=self._eval_config,
                device=self._device,
                stash_originals_dir=_originals_tmpdir,
            )
        except Exception:
            import shutil

            shutil.rmtree(_originals_tmpdir, ignore_errors=True)
            raise
        _n_segments_total = sum(len(v) for v in first_reports.values())
        log.info(
            f"Poisson realisation 1/{n_realisations} complete in {time.perf_counter() - _t0:.1f}s "
            f"({len(first_reports)} sessions, {_n_segments_total} segments)."
        )

        # Free the Poisson tokenizer — realisations 1+ use numpy-only
        # poisson_baseline_reconstruction, so we don't need the model on GPU.
        del poisson_tokenizer, _poisson_fn
        gc.collect()
        torch.cuda.empty_cache()

        # --- Realisations 1..N-1: reuse collated segments, only resample ---
        #
        # Downstream prediction metrics use latents/behavioral data, not the
        # reconstructed spikes — their values are invariant across Poisson
        # re-samplings and trivially perfect for the ceiling.  Skip them in
        # realisations 1+ and the ceiling for speed; realisation 0 (the full
        # pipeline run above) already has them.
        #
        # Correlograms (ACG, CCG, connectivity) are similarly skipped:
        # expensive O(N·S) Phy computations whose cross-realisation variance
        # is dominated by spike-count noise.
        from functools import partial

        from ntt.evaluation.poisson_realisations import (
            build_lightweight_pipeline,
            run_poisson_realisations,
        )

        # Parent uses lightweight_pipeline for the perfect-ceiling pass below.
        # Workers build their own via pipeline_factory — see the
        # poisson_realisations module docstring for the fork+CUDA contract.
        lightweight_pipeline = build_lightweight_pipeline(self._eval_config)
        pipeline_factory = partial(build_lightweight_pipeline, self._eval_config)

        all_realisation_reports = [first_reports]

        # Load stashed originals from disk for resampling and ceiling.
        # segment.original was stripped (tensor freed) by the pipeline;
        # we reload the numpy arrays from the temp dir.
        _first_report_originals: dict[str, list[tuple[T.Any, T.Any, dict]]] = {}
        for session_id, segment_list in first_reports.items():
            entries = []
            for segment, data in segment_list:
                stash_path = data["metadata"].get("_original_stash_path")
                if stash_path is None:
                    raise RuntimeError(
                        f"No stashed original for {session_id} segment "
                        f"{segment.segment_index}. This is a bug — "
                        f"stash_originals_dir should have been set."
                    )
                original_np = _load_stashed_original(stash_path)
                original_data = self._to_eval_data(original_np, bin_size_seconds)
                metadata = dict(data["metadata"])
                # Drop CUDA tensors from the segment before it crosses the
                # fork-pool IPC boundary; downstream aggregation only needs
                # `original_shape`, which tensor_free_copy preserves.
                entries.append((segment.tensor_free_copy(), original_data, metadata))
            _first_report_originals[session_id] = entries

        extra_realisation_reports = run_poisson_realisations(
            originals=_first_report_originals,
            pipeline_factory=pipeline_factory,
            bin_size_seconds=bin_size_seconds,
            bin_size_ms=bin_size_ms,
            poisson_rate_ms=poisson_rate_ms,
            base_seed=self.poisson_seed,
            n_realisations_to_run=n_realisations - 1,
            n_workers=self.poisson_n_workers,
        )
        all_realisation_reports.extend(extra_realisation_reports)
        gc.collect()
        torch.cuda.empty_cache()

        # Average across realisations — then release per-realisation data
        poisson_reports = self._average_session_reports_across_realisations(
            all_realisation_reports, MetricResult
        )
        del all_realisation_reports
        gc.collect()
        torch.cuda.empty_cache()

        # Perfect ceiling: original vs original (no latents needed).
        # Reuses the lightweight pipeline — downstream prediction metrics and
        # correlograms are already captured from realisation 0, and the
        # ceiling is trivially perfect for them.
        # Uses _first_report_originals (loaded from disk) for the ceiling.
        _n_segments_ceiling = sum(len(v) for v in _first_report_originals.values())
        log.info(
            f"Computing perfect ceiling (original vs original) over "
            f"{_n_segments_ceiling} segments..."
        )
        _t_ceil = time.perf_counter()
        ceiling_pipeline = lightweight_pipeline
        perfect_reports: dict[str, list[tuple[T.Any, dict[str, T.Any]]]] = {}
        for session_id, entries in _first_report_originals.items():
            for segment, original_data, metadata in entries:
                perfect_report = ceiling_pipeline.run(original_data, original_data)
                perfect_reports.setdefault(session_id, []).append(
                    (segment, {"report": perfect_report, "metadata": metadata})
                )
        log.info(f"Perfect ceiling complete in {time.perf_counter() - _t_ceil:.1f}s.")
        del _first_report_originals
        # Clean up stashed numpy files from disk
        import shutil

        shutil.rmtree(_originals_tmpdir, ignore_errors=True)
        gc.collect()
        torch.cuda.empty_cache()

        log.info("Aggregating Poisson and perfect-ceiling reports + computing comparisons...")
        poisson_aggregated = aggregate_session_reports(
            poisson_reports, strategy=self.aggregation_strategy
        )
        perfect_aggregated = aggregate_session_reports(
            perfect_reports, strategy=self.aggregation_strategy
        )
        perfect_aggregated = self._inject_known_perfect_ceiling_metrics(
            perfect_aggregated,
            model_aggregated,
        )
        comparisons = compute_normalized_comparisons(
            model_aggregated,
            poisson_aggregated,
            perfect_aggregated,
        )

        log.info("Poisson baseline comparison complete.")
        return comparisons, poisson_aggregated, perfect_aggregated, poisson_reports

    @staticmethod
    def _average_session_reports_across_realisations(
        realisation_session_reports: list[dict],
        MetricResult: type,
    ) -> dict:
        """Average session reports across N Poisson realisations.

        Each entry in *realisation_session_reports* has the standard structure
        ``{session_id: [(segment, {"report": EvaluationReport, ...}), ...]}``.
        Returns a single dict with the same structure, where each report's
        metric values are averaged across realisations.
        """
        if len(realisation_session_reports) == 1:
            return realisation_session_reports[0]

        base = realisation_session_reports[0]
        averaged: dict[str, list[tuple[T.Any, dict[str, T.Any]]]] = {}

        for session_id, segment_list in base.items():
            averaged[session_id] = []
            for seg_idx, (segment, data) in enumerate(segment_list):
                reports = []
                for realisation in realisation_session_reports:
                    if session_id in realisation and seg_idx < len(realisation[session_id]):
                        _, r_data = realisation[session_id][seg_idx]
                        reports.append(r_data["report"])

                avg_report = NTTEvalJob._average_realisation_reports(reports, MetricResult)
                averaged[session_id].append(
                    (segment, {"report": avg_report, "metadata": dict(data.get("metadata", {}))})
                )

        return averaged

    @staticmethod
    def _average_realisation_reports(
        reports: list,
        MetricResult: type,
    ) -> T.Any:
        """Average metric values across multiple Poisson realisation reports.

        For each metric, computes the mean and std of ``raw_value`` across
        realisations and stores ``realisation_std`` in metadata.  When only
        one report is provided, returns it unchanged.
        """
        if len(reports) == 1:
            return reports[0]

        base = reports[0]
        averaged_results: dict[str, T.Any] = {}

        for metric_name, base_result in base.results.items():
            values = []
            for report in reports:
                if metric_name not in report.results:
                    continue
                v = report.results[metric_name].raw_value
                if isinstance(v, np.ndarray):
                    values.append(v)
                else:
                    values.append(float(v))

            if not values:
                continue

            if isinstance(values[0], np.ndarray):
                stacked = np.stack(values, axis=0)
                mean_val = np.nanmean(stacked, axis=0)
                std_val = np.nanstd(stacked, axis=0)
            else:
                arr = np.array(values)
                mean_val = float(np.nanmean(arr))
                std_val = float(np.nanstd(arr))

            metadata = dict(base_result.metadata)
            metadata["realisation_std"] = std_val
            metadata["n_realisations"] = len(values)

            # Preserve per-realisation fold results (fold_accuracies, fold_r2s)
            # so downstream plotting can compute cross-realisation x fold std.
            for fold_key in ("fold_accuracies", "fold_r2s"):
                per_real_folds = []
                for report in reports:
                    r = report.results.get(metric_name)
                    if r is not None:
                        folds = r.metadata.get(fold_key)
                        if folds is not None:
                            per_real_folds.append(folds)
                if per_real_folds:
                    metadata[f"per_realisation_{fold_key}"] = per_real_folds

            averaged_results[metric_name] = MetricResult(
                name=base_result.name,
                category=base_result.category,
                direction=base_result.direction,
                level=base_result.level,
                raw_value=mean_val,
                normalized_value=base_result.normalized_value,
                metadata=metadata,
                value_range=base_result.value_range,
                source_data_type=base_result.source_data_type,
            )

        from ntt.evaluation.core.types import EvaluationReport

        return EvaluationReport(
            results=averaged_results,
            category_scores=base.category_scores,
            config=base.config,
            skipped_metrics=base.skipped_metrics,
            original_data_type=base.original_data_type,
            reconstructed_data_type=base.reconstructed_data_type,
        )

    @staticmethod
    def _inject_known_perfect_ceiling_metrics(
        perfect_aggregated,
        model_aggregated,
    ):
        """Insert analytically known ceiling=1 metrics into the aggregated report.

        Structure-preservation metrics, downstream decoding metrics, and
        correlogram metrics (acg_correlation, ccg_correlation, connectivity_f1)
        have a known perfect score of 1, while the ``original vs original``
        ceiling computation does not work for them (either analytically or
        because they are excluded from the lightweight ceiling pipeline for
        performance). Injecting the ceiling here keeps them in normalized
        comparisons and the radar plot.

        This has to fill in values into a ceiling report structre, that's why
        it's somewhat complex.
        """
        from ntt.evaluation.core.types import MetricDirection

        # Metrics analytically known to be perfect (1.0) when comparing
        # original vs original, but excluded from the lightweight ceiling
        # pipeline for performance reasons.
        _known_perfect_metric_names = {
            "population_structure_preservation",
            "per_unit_structure_preservation",
            "acg_correlation",
            "ccg_correlation",
            "connectivity_f1",
        }

        completed_ceiling_results = dict(perfect_aggregated.results)
        changed = False

        for display_name, model_result in model_aggregated.results.items():
            if display_name in perfect_aggregated.results:
                continue

            is_known_perfect = (
                model_result.category == MetricCategory.DOWNSTREAM_DECODING
                or model_result.name in _known_perfect_metric_names
                or display_name in _known_perfect_metric_names
            )
            if not is_known_perfect:
                continue

            # Perfect ceiling: 1.0 for higher-is-better, 0.0 for lower-is-better
            perfect_value = (
                1.0 if model_result.direction == MetricDirection.HIGHER_IS_BETTER else 0.0
            )

            completed_ceiling_results[display_name] = type(model_result)(
                name=model_result.name,
                category=model_result.category,
                direction=model_result.direction,
                level=model_result.level,
                raw_value=perfect_value,
                normalized_value=model_result.normalized_value,
                metadata={
                    "synthetic_ceiling": True,
                    "ceiling_reason": "known_perfect_score",
                },
                value_range=model_result.value_range,
                source_data_type=model_result.source_data_type,
            )
            changed = True

        if not changed:
            return perfect_aggregated

        perfect_aggregated.results = completed_ceiling_results
        perfect_aggregated.category_scores = aggregate_by_category(completed_ceiling_results)
        return perfect_aggregated

    # ── Plot generation ──────────────────────────────────────────────

    def _generate_plots(
        self,
        tokenizer: BaseTokenizer,
        bin_size_seconds: float | None,
        session_reports: dict,
        comparisons_vs_poisson: dict[str, MetricComparison] | None,
        model_aggregated: T.Any,
        poisson_aggregated: T.Any,
        perfect_aggregated: T.Any,
        poisson_session_reports: dict | None = None,
    ) -> Path:
        """Generate per-session reconstruction plots and global comparison plots."""
        import matplotlib.pyplot as plt

        from ntt.evaluation.plotting import get_validation_segment, plot_reconstruction_summary

        assert self._datamodule is not None
        assert self._device is not None

        output_dir = self._get_output_dir() or Path(".")
        output_dir.mkdir(parents=True, exist_ok=True)
        session_plots_dir = output_dir / "sessions"
        model_label = self._tokenizer_legend_short_name()
        poisson_label = self._plot_poisson_label or "Poisson"
        ckpt_caption = self._tokenizer_checkpoint_caption()

        # ── Per-session reconstruction plots ─────────────────────────
        session_ids = list(session_reports.keys())
        n_plot = min(self.plot_sessions, len(session_ids))
        log.info(
            f"Generating reconstruction plots for {n_plot}/{len(session_ids)} sessions "
            f"({self.plot_seconds}s of data each)..."
        )

        for i, session_id in enumerate(session_ids[:n_plot]):
            # Derive folder: brainset/session_name
            if "/" in session_id:
                brainset, session_name = session_id.split("/", 1)
            else:
                brainset, session_name = session_id, session_id
            session_dir = session_plots_dir / brainset / session_name
            session_dir.mkdir(parents=True, exist_ok=True)

            try:
                segment, seg_bin_size = get_validation_segment(
                    tokenizer,
                    self._datamodule,
                    session_id=session_id,
                    seconds=self.plot_seconds,
                    device=self._device,
                )
                figs = plot_reconstruction_summary(segment, seg_bin_size)
                for name, fig in figs.items():
                    fig_path = session_dir / f"{name}.png"
                    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                log.info(f"  [{i + 1}/{n_plot}] {session_id} → {session_dir}")
            except Exception as e:
                log.warning(f"  [{i + 1}/{n_plot}] {session_id}: plot failed — {e}")

        # ── Global comparison plots (Poisson baseline) ───────────────
        if comparisons_vs_poisson:
            from ntt.evaluation.visualization import plot_normalized_comparison_radar

            fig_radar = plot_normalized_comparison_radar(comparisons_vs_poisson)
            radar_path = output_dir / "normalized_comparison_radar.png"
            fig_radar.savefig(radar_path, dpi=150, bbox_inches="tight")
            log.info(f"  Saved {radar_path}")
            plt.close(fig_radar)

        if model_aggregated is not None and poisson_aggregated is not None:
            from ntt.evaluation.visualization import plot_metric_bars, plot_metric_dumbbell

            fig_bars = plot_metric_bars(
                {model_label: model_aggregated, poisson_label: poisson_aggregated}
            )
            bars_path = output_dir / "metric_bars.png"
            fig_bars.savefig(bars_path, dpi=150, bbox_inches="tight")
            log.info(f"  Saved {bars_path}")
            plt.close(fig_bars)

            fig_dumbbell = plot_metric_dumbbell(
                model_aggregated,
                poisson_aggregated,
                model_name=model_label,
                baseline_name=poisson_label,
            )
            dumbbell_path = output_dir / "metric_dumbbell.png"
            fig_dumbbell.savefig(dumbbell_path, dpi=150, bbox_inches="tight")
            log.info(f"  Saved {dumbbell_path}")
            plt.close(fig_dumbbell)

        # ── Behavioral plots ─────────────────────────────────────────
        from ntt.evaluation.behavioral_plotting import generate_all_behavioral_plots

        tokenizer_session_reports: dict[str, dict] = {model_label: session_reports}
        if poisson_session_reports is not None:
            tokenizer_session_reports[poisson_label] = poisson_session_reports
        try:
            generate_all_behavioral_plots(
                tokenizer_session_reports,
                output_dir=output_dir,
                dpi=150,
                trained_checkpoint_caption=ckpt_caption,
            )
        except Exception as e:
            log.warning(f"  Behavioral plots failed: {e}")

        # ── Structure preservation plots ──────────────────────────────
        self._generate_structure_plots(session_reports, output_dir)

        return output_dir

    def _generate_structure_plots(
        self,
        session_reports: dict,
        output_dir: Path,
    ) -> None:
        """Generate structure preservation plots from session reports."""
        import matplotlib.pyplot as plt

        from ntt.evaluation.visualization import (
            plot_per_unit_structure_preservation,
            plot_structure_preservation,
        )

        plot_specs: list[tuple[str, T.Callable, str]] = [
            (
                "population_structure_preservation",
                plot_structure_preservation,
                "population_structure",
            ),
            (
                "per_unit_structure_preservation",
                plot_per_unit_structure_preservation,
                "per_unit_structure",
            ),
        ]

        for session_id, segments in session_reports.items():
            if not segments:
                continue
            _segment, info = segments[0]
            report = info.get("report")
            if report is None:
                continue

            # Place plots in the per-session folder: sessions/brainset/session_name/
            if "/" in session_id:
                brainset, session_name = session_id.split("/", 1)
            else:
                brainset, session_name = session_id, session_id
            session_dir = output_dir / "sessions" / brainset / session_name
            session_dir.mkdir(parents=True, exist_ok=True)

            for metric_name, plot_fn, suffix in plot_specs:
                result = report.results.get(metric_name)
                if result is None:
                    continue
                # Skip NaN results (metric ran but had no data)
                raw = result.raw_value
                if np.isscalar(raw) and np.isnan(raw):
                    continue
                if result.metadata.get("rdm_input") is None:
                    continue
                try:
                    fig = plot_fn(result)
                    fig_path = session_dir / f"{suffix}.png"
                    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
                    log.info(f"  Saved {fig_path}")
                    plt.close(fig)
                except Exception as e:
                    log.warning(f"  Structure plot ({suffix}) failed for {session_id}: {e}")

    # ── Reporting ────────────────────────────────────────────────────

    def _print_results(  # type: ignore[no-untyped-def]
        self,
        session_reports,
        aggregated,
        comparisons_vs_poisson: dict[str, MetricComparison] | None = None,
        poisson_aggregated=None,
        perfect_aggregated=None,
    ) -> None:
        n_sessions = len(session_reports)
        print(f"\n{'=' * 60}")
        print("EVALUATION RESULTS")
        print(f"{'=' * 60}")
        if self.ckpt_path is not None:
            print(f"Checkpoint:  {self.ckpt_path}")
        else:
            print(f"Tokenizer:   {self.tokenizer_config.KIND} (untrained)")  # type: ignore[union-attr]
        print(f"Sessions:    {n_sessions}")
        print(f"Aggregation: {self.aggregation_strategy.value}")
        print(f"{'-' * 60}")

        for name, result in aggregated.results.items():
            value = result.raw_value
            if hasattr(value, "__len__"):
                value = float(np.nanmean(value))
            direction = result.direction.value
            print(f"  {name:40s} {value:>10.4f}  ({direction} is better)")

        if aggregated.skipped_metrics:
            print("\nSkipped:")
            for name, reason in aggregated.skipped_metrics.items():
                print(f"  {name}: {reason}")

        if comparisons_vs_poisson:
            print(f"\n{'-' * 60}")
            n_real = self.poisson_n_realisations
            header = (
                "NORMALIZED COMPARISON vs POISSON BASELINE  "
                f"(0 = Poisson, 1 = perfect reconstruction, {n_real} realisations; "
                "firing_rate_error: 0 = population-mean-rate reference)"
            )
            print(header)
            print(f"{'-' * 60}")
            for name, comparison in sorted(comparisons_vs_poisson.items()):
                score = comparison.normalized_score
                if score is None:
                    continue
                marker = "+" if score > 0.0 else "-" if score < 0.0 else "="
                print(f"  [{marker}] {name:40s} {score:>8.3f}")

            invalid_comparisons = {
                name: comparison.reason
                for name, comparison in sorted(comparisons_vs_poisson.items())
                if comparison.normalized_score is None
            }
            if invalid_comparisons:
                print("\nComparison metrics skipped:")
                for name, reason in invalid_comparisons.items():
                    print(f"  {name}: {reason}")

        # Show Poisson baseline metric values with std when available
        if poisson_aggregated is not None and self.poisson_n_realisations > 1:
            print(f"\n{'-' * 60}")
            print(
                "POISSON BASELINE METRICS  "
                f"(mean +/- std, {self.poisson_n_realisations} realisations)"
            )
            print(f"{'-' * 60}")
            for name, result in poisson_aggregated.results.items():
                value = result.raw_value
                if hasattr(value, "__len__"):
                    value = float(np.nanmean(value))
                std = result.metadata.get("realisation_std")
                if std is not None:
                    if hasattr(std, "__len__"):
                        std = float(np.nanmean(std))
                    print(f"  {name:40s} {value:>10.4f} +/- {std:.4f}")
                else:
                    print(f"  {name:40s} {value:>10.4f}")

        if perfect_aggregated is not None:
            print(f"\n{'-' * 60}")
            print("PERFECT RECONSTRUCTION CEILING")
            print(f"{'-' * 60}")
            for name, result in perfect_aggregated.results.items():
                value = result.raw_value
                if hasattr(value, "__len__"):
                    value = float(np.nanmean(value))
                print(f"  {name:40s} {value:>10.4f}")

        print(f"{'=' * 60}\n")

    def _save_results(  # type: ignore[no-untyped-def]
        self,
        aggregated,
        session_reports,
        comparisons_vs_poisson: dict[str, MetricComparison] | None = None,
        poisson_aggregated=None,
        perfect_aggregated=None,
        poisson_session_reports=None,
        tier_reports: dict[str, EvaluationReport] | None = None,
        generalization_block: dict | None = None,
        alternative_floor_per_session: dict[str, dict[str, float]] | None = None,
    ) -> list[Path]:
        source = (
            {"checkpoint": str(self.ckpt_path)}
            if self.ckpt_path is not None
            else {"tokenizer": self.tokenizer_config.KIND}  # type: ignore[union-attr]
        )
        # Serialize tokenizer config for reproducibility
        tokenizer_config: dict[str, T.Any] = {}
        if self._spec is not None:
            try:
                tokenizer_config = config_to_jsonable(self._spec, name="tokenizer_config")
            except Exception as e:
                log.warning(f"Failed to serialize tokenizer config: {e}. Saving KIND only.")
                tokenizer_config = {"KIND": getattr(self._spec, "KIND", "unknown")}

        output: dict[str, T.Any] = {
            **source,
            "tokenizer_config": tokenizer_config,
            "n_sessions": len(session_reports),
            "aggregation_strategy": self.aggregation_strategy.value,
            "split": self.split,
            "evaluate_all_neurons": self.evaluate_all_neurons,
            # The window actually scored (and the spec's own); the aggregation compares it.
            **self._eval_window_record(),
            # Units each session was scored on; the aggregation checks it against the
            # Poisson reference's.
            "neuron_selection": (
                self._neuron_selection.model_dump() if self._neuron_selection else None
            ),
            "aggregated": aggregated.to_dict(),
            # Per-session: both aggregated and per-segment results
            "per_session": {
                sid: aggregate_segment_reports(segments, self.aggregation_strategy).to_dict()
                for sid, segments in session_reports.items()
                if segments
            },
            "per_session_per_segment": {
                sid: [data["report"].to_dict() for _, data in segments]
                for sid, segments in session_reports.items()
                if segments
            },
        }
        # Each session's reference error for a metric scored against another reference
        # than the Poisson floor (see with_alternative_floors).
        for metric_name, per_session in (alternative_floor_per_session or {}).items():
            for sid, reference_error in per_session.items():
                metadata = output["per_session"][sid]["results"][metric_name]["metadata"]
                metadata[POPULATION_MEAN_RATE_ERROR_KEY] = reference_error
        if comparisons_vs_poisson is not None:
            output["comparison_vs_poisson"] = {
                name: comparison.to_dict() for name, comparison in comparisons_vs_poisson.items()
            }
            output["normalized_scores_vs_poisson"] = extract_normalized_scores(
                comparisons_vs_poisson
            )
            output["poisson_n_realisations"] = self.poisson_n_realisations
            output["poisson_reference_kind"] = self._poisson_reference_kind
            output["poisson_reference_window_seconds"] = self._poisson_reference_window_seconds
            if self._poisson_rate_bin_ms is not None:
                output["poisson_rate_bin_ms"] = self._poisson_rate_bin_ms
        if poisson_aggregated is not None:
            output["poisson_baseline"] = poisson_aggregated.to_dict()
        if poisson_session_reports is not None:
            output["poisson_per_session_per_segment"] = {
                sid: [data["report"].to_dict() for _, data in segments]
                for sid, segments in poisson_session_reports.items()
                if segments
            }
        if perfect_aggregated is not None:
            output["perfect_ceiling"] = perfect_aggregated.to_dict()
        if tier_reports:
            output["per_tier"] = {tier: report.to_dict() for tier, report in tier_reports.items()}
        # Generalization retention score is computed once in run() and passed in,
        # so it lands in both results.json and MLflow regardless of output_path.
        if generalization_block is not None:
            output["generalization"] = generalization_block
        if self._compression is not None:
            output["compression"] = self._compression.to_dict()
        output_dir = self._get_output_dir()
        if output_dir is None:
            raise ValueError("Cannot save evaluation results without an output directory.")

        saved_artifacts: list[Path] = []
        json_path = output_dir / "results.json"
        # Replace NaN/Inf with null at the JSON boundary so results.json stays
        # RFC-8259 conformant for strict downstream parsers (jq, pandas, JS).
        with json_path.open("w") as f:
            json.dump(
                replace_non_finite_floats(output),
                f,
                indent=2,
                default=_eval_json_default,
            )
        saved_artifacts.append(json_path)
        log.info(f"Results saved to {json_path}")

        # Write a long-form detailed results file for plotting per-sweep-parameter values.
        # This is an *additional* output; existing `results.json` / `results.csv` remain unchanged.
        detailed_path = output_dir / "metrics_detailed_results.jsonl"
        run_meta: dict[str, T.Any] = {
            "aggregation_strategy": self.aggregation_strategy.value,
            "checkpoint": str(self.ckpt_path) if self.ckpt_path is not None else None,
            "tokenizer_kind": getattr(self.tokenizer_config, "KIND", None),  # type: ignore[union-attr]
            "tokenizer_config": tokenizer_config,
            "n_sessions": len(session_reports),
            "evaluate_all_neurons": self.evaluate_all_neurons,
            **self._eval_window_record(),
            "neuron_selection": (
                self._neuron_selection.model_dump() if self._neuron_selection else None
            ),
        }
        run_meta.update(
            write_config_provenance_sidecar(
                run_meta_path=output_dir / "run_meta.json",
                eval_config=self._eval_config,
                data_config=self._provenance_data_config(),
                hydra_overrides=self._hydra_overrides,
                composed_config_yaml_text=self._provenance_composed_config_yaml(),
            )
        )
        if self.compare_poisson:
            run_meta["compare_poisson"] = True
            run_meta["poisson_n_realisations"] = int(self.poisson_n_realisations)
            if self._poisson_rate_bin_ms is not None:
                run_meta["poisson_rate_bin_ms"] = float(self._poisson_rate_bin_ms)
            run_meta["poisson_use_all_neurons"] = bool(self.poisson_use_all_neurons)

        run_meta_path = output_dir / "run_meta.json"
        with run_meta_path.open("w") as f:
            json.dump(
                replace_non_finite_floats(run_meta),
                f,
                indent=2,
                default=_eval_json_default,
            )
        saved_artifacts.append(run_meta_path)
        log.info(f"Run metadata saved to {run_meta_path}")

        write_metrics_detailed_results_jsonl(
            session_reports=session_reports,
            output_path=detailed_path,
            run_metadata=run_meta,
        )
        log.info(f"Detailed metric results saved to {detailed_path}")

        # Also save per-session DataFrame as CSV
        from ntt.evaluation.aggregation.results_dataframe import (
            save_results_csv,
            session_reports_to_dataframe,
        )

        csv_path = output_dir / "results.csv"
        df = session_reports_to_dataframe(session_reports)
        save_results_csv(df, csv_path)
        log.info(f"Results DataFrame saved to {csv_path}")

        # Save normalized scores as a separate CSV for easy programmatic access
        if comparisons_vs_poisson is not None:
            from ntt.evaluation.aggregation.results_dataframe import (
                normalized_comparisons_to_dataframe,
            )

            norm_df = normalized_comparisons_to_dataframe(comparisons_vs_poisson)
            norm_csv_path = output_dir / "normalized_scores.csv"
            save_results_csv(norm_df, norm_csv_path)
            log.info(f"Normalized scores saved to {norm_csv_path}")

        return saved_artifacts
