"""Standalone input_robustness (M1) evaluation job.

Mirrors :mod:`ntt.jobs.eval_robustness_latent` but implements the input_robustness (M1)
sweep: per configured corruption, build a
replica-aware dataloader that emits ``[K+1, T, U]`` batches (one clean
sample + ``K`` corrupted replicas), run the tokenizer once per batch,
and feed the resulting ``replica_latents`` /
``replica_reconstructions`` / ``replica_metadata`` passthroughs into
the standard ``EvaluationPipeline.run`` so the
``InputCorruptionRobustness`` metric can score normalized latent
distances per corruption.

Output: ``results.jsonl`` (streamed), ``results.csv``
(wide, one row per (session, corruption_label)), and ``run_meta.json``.

Driven by ``ntt job=eval_robustness_input ...``.
"""

from __future__ import annotations

import gc
import json
import re
import traceback
import typing as T
import warnings
from pathlib import Path

import matplotlib

# Force a headless backend so the job runs on cluster nodes without a display.
matplotlib.use("Agg")

import numpy as np
import pydantic as pdt
import torch

# ── Warning filters (match eval.py) ─────────────────────────────────
warnings.filterwarnings(
    "once",
    message=r".*DataLoader will create \d+ worker processes.*",
    category=UserWarning,
)
warnings.filterwarnings("ignore", message=r".*Mean of empty slice.*")
warnings.filterwarnings("ignore", message=r".*Degrees of freedom <= 0 for slice.*")

from ntt.contrib import TokenizerSpecKind
from ntt.core.data.components.corruption import (
    AddSpec,
    DropAndAddSpec,
    DropNeuronsSpec,
    DropSpec,
    JitterSpec,
    PermuteNeuronsSpec,
    WindowShiftSpec,
)
from ntt.core.data.components.transforms_auxiliary import (
    build_transform_with_replica_corruption,
)
from ntt.core.data.datamodule import NTTDataModule, NTTDataModuleConfig
from ntt.core.data.replica import build_replica_dataloader
from ntt.core.model import load_tokenizer_from_checkpoint
from ntt.core.model.tokenizer import BaseTokenizer, TokenizerSpec, create_tokenizer
from ntt.evaluation.aggregation.comparability import add_robustness_report_fields
from ntt.evaluation.aggregation.comparison import (
    compute_normalized_comparisons,
    extract_normalized_scores,
)
from ntt.evaluation.artifact import aggregate_per_session_dicts
from ntt.evaluation.config import EvaluationConfig, validate_config
from ntt.evaluation.config.schema import (
    AggregationStrategy,
    CorruptionParams,
    resolve_window_shift_windows,
)
from ntt.evaluation.core.types import EvaluationReport, MetricResult
from ntt.evaluation.mlflow_logging import (
    MLflowLoggingConfig,
    sanitize_metric_name_for_mlflow,
)
from ntt.evaluation.neuron_selection import (
    NeuronSelection,
    all_neuron_data_config,
    require_direct_all_neuron_eval,
)
from ntt.evaluation.plotting.io import save_fig_multi
from ntt.evaluation.plotting.robustness import (
    plot_input_corruption_a,
    plot_input_corruption_c,
    plot_input_corruption_per_dim_distribution,
)
from ntt.evaluation.utils.config_provenance import (
    composed_config_yaml,
    config_subtree_to_jsonable,
    config_to_jsonable,
    hydra_task_overrides,
    write_config_provenance_sidecar,
)
from ntt.evaluation.utils.paths import resolve_artifact_path
from ntt.evaluation.utils.replica_collation import (
    run_validation_evaluation_per_session_replicas,
)
from ntt.evaluation.utils.window_chunking import (
    chunked_tokenizer_fn,
)
from ntt.jobs._poisson_reference import (
    load_poisson_rate_bin_ms,
    record_neuron_selection,
    resolve_poisson_reference_paths,
    warn_if_no_session_overlap,
)
from ntt.jobs.base import Locals
from ntt.jobs.ntt_base import NTTBaseJob
from ntt.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def _window_shift_default_analysis_ms(eval_window_seconds: float) -> float:
    """Default input_robustness (M1) window-shift analysis span = the authoritative eval window.

    The eval window is the single source of truth (see window_contract.py); the
    window_shift analysis window defaults to it so input_robustness (M1) robustness tracks the same
    window as the rest of the eval. A per-corruption ``analysis_window_ms`` still
    overrides this default.
    """
    return float(eval_window_seconds) * 1000.0


def _params_to_spec(params: CorruptionParams, *, eval_window_seconds: float) -> T.Any:
    """Convert a :class:`CorruptionParams` to its concrete corruption spec.

    The Pydantic ``CorruptionParams`` schema is a discriminated-by-``type``
    union; the dataclass specs in
    :mod:`ntt.core.data.components.corruption` are the actual things the
    ``ReplicaAware*Corruption`` transforms consume.

    """
    t = params.type
    if t == "jitter":
        return JitterSpec(k_ms=float(params.k_ms or 0.0), p=float(params.p or 0.0))
    if t == "drop":
        return DropSpec(fraction=float(params.fraction or 0.0))
    if t == "add":
        return AddSpec(fraction=float(params.fraction or 0.0))
    if t == "drop_and_add":
        return DropAndAddSpec(fraction=float(params.fraction or 0.0))
    if t == "window_shift":
        max_shift_ms = float(params.max_shift_ms or 0.0)
        default_analysis_ms = _window_shift_default_analysis_ms(eval_window_seconds)
        analysis_ms, source_ms = resolve_window_shift_windows(
            max_shift_ms=max_shift_ms,
            analysis_window_ms=(
                params.analysis_window_ms
                if params.analysis_window_ms is not None
                else default_analysis_ms
            ),
            source_window_ms=params.source_window_ms,
        )
        return WindowShiftSpec(
            max_shift_ms=max_shift_ms,
            analysis_window_ms=analysis_ms,
            source_window_ms=source_ms,
        )
    if t == "drop_neurons":
        return DropNeuronsSpec(
            fraction=float(params.fraction or 0.0),
            deterministic_per_replica=bool(params.deterministic_per_replica),
        )
    if t == "permute_neurons":
        return PermuteNeuronsSpec(
            deterministic_per_replica=bool(params.deterministic_per_replica),
        )
    raise ValueError(f"Unsupported CorruptionParams.type: {t!r}")


def _corruption_label(params: CorruptionParams) -> str:
    """Return a stable human-readable label for one corruption spec.

    Falls back to ``params.name`` when set, otherwise a short
    ``type[/key=value]`` string.
    """
    if params.name:
        return params.name
    bits = [params.type]
    for key in ("k_ms", "p", "fraction", "max_shift_ms"):
        v = getattr(params, key, None)
        if v is not None:
            bits.append(f"{key}={v}")
    return "|".join(bits)


# Per-type strength field used as the x-axis for plot_input_corruption_a.
# Permute_neurons has no continuous strength → we use a sentinel of 1.0
# so it still draws a single point on a curve.
_STRENGTH_FIELD: dict[str, str] = {
    "jitter": "k_ms",
    "drop": "fraction",
    "add": "fraction",
    "drop_and_add": "fraction",
    "window_shift": "max_shift_ms",
    "drop_neurons": "fraction",
    "permute_neurons": "",  # sentinel
}

_ROBUSTNESS_INPUT_CORE_SCORE_LABEL = "aggregate_non_population"
_ROBUSTNESS_INPUT_CORE_SCORE_METRIC = "robustness_input_core_avg"
_ROBUSTNESS_INPUT_CORE_SCORE_COMPONENTS = (
    "window_shift|max_shift_ms=3.0",
    "jitter|k_ms=1.0|p=0.2",
    "drop_and_add|fraction=0.02",
)


def _corruption_strength(params: CorruptionParams) -> float:
    """Return the scalar "strength" for one corruption spec, used as the
    x-axis value in M1-a curves.

    For ``permute_neurons`` (no continuous strength) the strength is
    fixed to 1.0 so the spec still appears as a single-point curve.
    """
    field = _STRENGTH_FIELD.get(params.type, "")
    if not field:
        return 1.0
    v = getattr(params, field, None)
    if v is None:
        return 1.0
    return float(v)


class NTTEvalRobustnessInputJob(
    NTTBaseJob,
    strict=False,
    frozen=False,
    extra="forbid",
    arbitrary_types_allowed=True,
):
    """input_robustness (M1) evaluation job.

    For each ``cfg.eval_config.robustness_input.corruptions`` entry,
    builds a replica-aware dataloader (``K+1`` rows per source window),
    runs the standard validation pipeline through
    :func:`run_validation_evaluation_per_session_replicas`, and
    surfaces the per-corruption input_robustness (M1) metric values into ``results.jsonl``
    / ``results.csv``.
    """

    KIND: T.Literal["eval_robustness_input"] = "eval_robustness_input"
    applies_eval_window: T.ClassVar[bool] = True

    # Same as NTTEvalRobustnessLatentJob: exactly one of ckpt_path /
    # tokenizer_config.
    ckpt_path: str | Path | None = None
    ckpt_weights_only: bool = False
    tokenizer_config: TokenizerSpecKind = None
    data_config: NTTDataModuleConfig

    # Evaluation config — accepts a raw dict so Hydra can pass the full
    # evaluation YAML (including the ``robustness_input`` section).
    eval_config: dict[str, T.Any] | None = None

    device: str = "auto"
    output_path: str | Path | None = None
    split: T.Literal["val", "test"] = "val"
    aggregation_strategy: AggregationStrategy = AggregationStrategy.MEAN
    evaluate_all_neurons: bool = True
    generate_plots: bool = True
    poisson_reference: str | Path | None = None

    # MLflow logging — defaults to disabled. When enabled, the input_robustness (M1) job logs
    # its own run (separate from the main eval pass) tagged with the same
    # ``ntt.eval.checkpoint_path`` so the runs are discoverable from one
    # another. Secondary pass: ``upload_checkpoint`` defaults False here.
    mlflow_logging: MLflowLoggingConfig = MLflowLoggingConfig(
        upload_checkpoint=False,
        run_name_suffix="__m1",
        metric_prefix="eval_m1",
    )

    # Private runtime state
    _tokenizer: BaseTokenizer | None = pdt.PrivateAttr(default=None)
    _device: torch.device | None = pdt.PrivateAttr(default=None)
    _eval_config: EvaluationConfig | None = pdt.PrivateAttr(default=None)
    _spec: TokenizerSpec | None = pdt.PrivateAttr(default=None)
    _hydra_overrides: list[str] = pdt.PrivateAttr(default_factory=list)
    _composed_config_yaml: str | None = pdt.PrivateAttr(default=None)
    _hydra_data_config: T.Any | None = pdt.PrivateAttr(default=None)
    # Units each session is scored on (set when the eval data is built).
    _neuron_selection: NeuronSelection | None = pdt.PrivateAttr(default=None)

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
            "Full composed Hydra config unavailable for input_robustness (M1) run metadata; "
            "writing a job-scoped snapshot instead."
        )
        return f"job:\n  KIND: {self.KIND}\n  hydra_config_unavailable: true\n"

    def _provenance_data_config(self) -> T.Any:
        """Return the effective runtime data config used by the dataloader."""
        return self.data_config

    @pdt.model_validator(mode="after")
    def _check_ckpt_or_config(self) -> T.Self:
        """Require exactly one of ``ckpt_path`` / ``tokenizer_config``."""
        has_ckpt = self.ckpt_path is not None
        has_config = self.tokenizer_config is not None
        if has_ckpt == has_config:
            raise ValueError(
                "Provide exactly one of ckpt_path (trained) or"
                " tokenizer_config (untrained), not both."
            )
        return self

    # ── Context management (mirrors NTTEvalRobustnessLatentJob.__enter__) ──

    def _resolve_device(self) -> torch.device:
        """Resolve ``self.device`` to a concrete :class:`torch.device`"""
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    @staticmethod
    def _resolve_ckpt_path(ckpt_path: str | Path) -> str | Path:
        """Resolve MLflow checkpoint artifacts down to a concrete ``.ckpt`` file."""
        if str(ckpt_path).startswith("hf://"):
            from ntt.hub import materialize_checkpoint

            return materialize_checkpoint(str(ckpt_path))
        path_str = str(ckpt_path)
        if not (path_str.startswith("mlflow-artifacts:") or path_str.startswith("mlflow:")):
            return ckpt_path
        resolved = resolve_artifact_path(path_str)
        if resolved.is_dir():
            ckpt_files = list(resolved.glob("*.ckpt"))
            if len(ckpt_files) == 1:
                return ckpt_files[0]
            if len(ckpt_files) == 0:
                raise FileNotFoundError(f"No .ckpt file found in downloaded artifact: {resolved}")
            raise ValueError(
                f"Multiple .ckpt files in downloaded artifact: {[f.name for f in ckpt_files]}"
            )
        return resolved

    def __enter__(self) -> T.Self:
        """Set up device, tokenizer, datamodule, and parsed eval config."""
        self._device = self._resolve_device()
        log.info(f"Using device: {self._device}")

        if self.ckpt_path is not None:
            # Resolve ``mlflow-artifacts:/`` URIs to a local path before
            # handing to Lightning, which otherwise tries to open the URI as
            # a literal filesystem path.
            local_ckpt = self._resolve_ckpt_path(self.ckpt_path)
            log.info(f"Loading tokenizer from {local_ckpt}")
            self._tokenizer = load_tokenizer_from_checkpoint(
                str(local_ckpt),
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

        # Match eval.py: force deterministic transforms for reproducibility.
        if not self.data_config.deterministic_transforms:
            self.data_config.deterministic_transforms = True
            log.info("Eval (robustness_input) job: set deterministic_transforms=True")

        # Eval window is the single source of truth: force the datamodule window
        # and validate the tokenizer accepts it. MUST precede super().__enter__()
        # which builds the datamodule (see eval.py for rationale).
        from ntt.evaluation.utils.window_contract import apply_eval_window

        apply_eval_window(
            self.data_config, self.tokenizer_config, self._resolve_eval_window_seconds()
        )

        super().__enter__()

        if self.eval_config is not None:
            self._eval_config = validate_config(self.eval_config)
        else:
            self._eval_config = EvaluationConfig()

        stage = "validate" if self.split == "val" else "test"
        self.datamodule.setup(stage)

        return self

    # ── Run ───────────────────────────────────────────────────────────

    def run(self) -> Locals:
        """Execute the input_robustness (M1) sweep and write per-D11 outputs."""
        assert self._tokenizer is not None
        assert self._device is not None
        assert self._eval_config is not None
        assert self._spec is not None

        ri_cfg = self._eval_config.robustness_input
        if ri_cfg is None:
            log.info(
                "eval_config.robustness_input is None — nothing to do. "
                "Set `evaluation.robustness_input` to enable the input_robustness (M1) sweep."
            )
            return locals()

        spec = self._spec
        bin_size_seconds = spec.bin_size_ms / 1000.0 if spec.bin_size_ms else None
        if bin_size_seconds is None:
            raise ValueError("robustness_input job requires a tokenizer with fixed bin_size_ms.")
        tokenizer = self._tokenizer
        device = self._device

        # Per-corruption-iteration raster capture state. Reset for every
        # corruption: maps source-window-index → captured spike_matrix (clean
        # at index 0, corrupted at indices 1..K). Populated by ``_tokenizer_fn``
        # below for the source windows we picked for raster export.
        # Outer mapping: session_id → window_idx_within_session → np.ndarray
        raster_capture: dict[str, dict[int, np.ndarray]] = {}
        # Per-session counter to assign incrementing window indices to source
        # windows as they are seen by the dataloader. Reset per corruption
        # iteration since each corruption rebuilds its own dataloader.
        per_session_window_counter: dict[str, int] = {}
        # Set of source-window indices (within a session) to capture. We
        # capture the same indices in every session.
        raster_window_idx_set: set[int] = set()
        capture_enabled: bool = self.generate_plots and bool(ri_cfg.plot_examples)

        @torch.no_grad()
        def _raw_tokenize(batch: T.Any) -> dict:
            # M1-a only consumes ``replica_latents``; running the decoder
            # would be wasted compute and forces tokenizers without a
            # decoder out of the input_robustness (M1) path. ``encode_with_timestamps`` returns
            # post-``tokens_to_latents`` continuous latents (codebook
            # vectors for VQ, identity for VAE/FSQ/BSQ) plus the per-token
            # timestamp metadata; the chunking + replica iterator both
            # tolerate a missing ``reconstruction`` key.
            latents, metadata = tokenizer.encode_with_timestamps(batch)
            return {"latents": latents, **metadata}

        # Wrap once at setup: for fixed-input tokenizers the wrapper
        # pre-chunks each batch along time; for variable-input tokenizers it
        # is a near no-op pass-through.
        _wrapped_tokenize = chunked_tokenizer_fn(_raw_tokenize, spec=spec)

        @torch.no_grad()
        def _tokenizer_fn(batch: T.Any) -> dict:
            """Encode + reconstruct in one pass, returning the tokenizer's
            natural batched output.

            Side effect: when this batch's source-window-within-session matches
            an entry in ``raster_window_idx_set``, copy the per-replica
            ``spike_matrix`` into ``raster_capture`` for later raster export.
            """
            if capture_enabled and raster_window_idx_set:
                # batch.session_id may be a single str or a per-row list.
                sid_field = getattr(batch, "session_id", None)
                if isinstance(sid_field, (list, tuple)):
                    sid = str(sid_field[0]) if sid_field else None
                else:
                    sid = str(sid_field) if sid_field is not None else None
                if sid is not None:
                    w_idx = per_session_window_counter.get(sid, 0)
                    per_session_window_counter[sid] = w_idx + 1
                    if w_idx in raster_window_idx_set:
                        sm = getattr(batch, "spike_matrix", None)
                        if sm is not None:
                            raster_capture.setdefault(sid, {})[w_idx] = sm.detach().cpu().numpy()
            # _raw_tokenize returns the latents dict, which the chunking wrapper passes through
            return T.cast(dict, _wrapped_tokenize(batch))

        split = self.split
        (
            _eval_dm,
            base_dataset,
            base_sampler,
            base_transform,
        ) = self._build_eval_data_components()

        seed = self.seed if self.seed is not None else 0

        output_dir = self._get_output_dir()
        jsonl_path = output_dir / "results.jsonl" if output_dir is not None else None
        csv_rows: list[dict[str, T.Any]] = []
        metric_schemas: dict[str, dict[str, T.Any]] = {}
        skipped_counter: dict[str, int] = {}
        total_pairs = 0

        # Decide raster selection up-front (deterministic in ``seed``).
        # We pick the first ``n_raster_examples_per_session`` source-window
        # indices in each session. (The first window seen for a session
        # gets index 0, then 1, then 2, ...) Per the task spec we cap at 3.
        # ``seed`` is preserved for the future case where we randomize
        # the chosen indices instead of taking the first few.
        n_raster_examples_per_session = 3
        if capture_enabled:
            raster_window_idx_set = set(range(n_raster_examples_per_session))

        # Per-corruption captured rasters across all corruptions:
        # session_id → window_idx → corruption_label → corrupted_spikes [T, U]
        # Plus a separate clean_spikes dict: session_id → window_idx → [T, U]
        #
        captured_corrupted: dict[str, dict[int, dict[str, np.ndarray]]] = {}
        captured_clean: dict[str, dict[int, np.ndarray]] = {}
        # Per-corruption latent distances: session_id → window_idx → label → float
        # Note: we currently only have segment-level (mean over windows) latent
        # distances available from the input_robustness (M1) metric; raster annotations will use
        # the corruption-level mean as a fallback. The segment-aware design
        # below threads the per-segment scalar through.
        captured_d_latent: dict[str, dict[int, dict[str, float]]] = {}
        # Per-dim diagnostics (consumed by ``plot_input_corruption_per_dim_
        # distribution``): for each session, σ_d² (a [D] vector, taken from
        # the input_robustness (M1) metric metadata; same across all corruptions for a session)
        # and the per-(label, replica-aggregated) NMSE_d breakdowns.
        # Multiplying NMSE_d × σ_d² recovers the unnormalised per-dim
        # MSE the per-dim distribution plot wants.
        captured_sigma_per_dim_sq: dict[str, list[float]] = {}
        captured_per_dim_nmse_breakdown: dict[str, dict[str, list[float]]] = {}

        jsonl_fh = jsonl_path.open("w") if jsonl_path is not None else None
        skipped_specs: list[dict[str, T.Any]] = []

        def _store_result_row(rec: dict[str, T.Any]) -> None:
            csv_rows.append(rec)
            if jsonl_fh is not None:
                if "value" in rec:
                    jsonl_rec = dict(rec)
                    jsonl_rec["raw_value"] = jsonl_rec.pop("value")
                else:
                    jsonl_rec = rec
                jsonl_fh.write(json.dumps(jsonl_rec) + "\n")

        try:
            for c_idx, params in enumerate(ri_cfg.corruptions):
                label = _corruption_label(params)
                log.info(f"[{c_idx + 1}/{len(ri_cfg.corruptions)}] corruption={label}")

                # Reset per-corruption raster capture buffers so each
                # iteration's session-window-counter starts at 0 (matches
                # the freshly-rebuilt replica_dataloader).
                raster_capture.clear()
                per_session_window_counter.clear()

                corruption_spec = _params_to_spec(
                    params, eval_window_seconds=self._resolve_eval_window_seconds()
                )
                replica_transform = build_transform_with_replica_corruption(
                    base_transform, corruption_spec
                )

                # WindowShift needs the dataloader to provide a SOURCE
                # window of length ``source_window_ms`` (longer than the
                # tokenizer's training-time analysis window, by
                # ``2 * max_shift_ms``). The default sampler produces
                # windows of ``self.datamodule.window_length`` (typically
                # 1.0 s = the tokenizer's training T). For WindowShift
                # specs we build a fresh ``SequentialFixedWindowSampler``
                # with the spec's ``source_window_ms`` so the
                # post-shift analysis sub-window lands at exactly the
                # tokenizer's training T. Non-WindowShift specs reuse
                # the standard sampler (window_length = training T).
                if isinstance(corruption_spec, WindowShiftSpec):
                    from torch_brain.data.sampler import (
                        SequentialFixedWindowSampler,
                    )

                    spec_sampler = SequentialFixedWindowSampler(
                        sampling_intervals=base_dataset.get_sampling_intervals(),
                        window_length=float(corruption_spec.source_window_ms) / 1000.0,
                        drop_short=True,
                    )
                else:
                    spec_sampler = base_sampler

                effective_window_seconds = None
                if isinstance(corruption_spec, WindowShiftSpec):
                    effective_window_seconds = float(corruption_spec.analysis_window_ms) / 1000.0

                replica_dataloader = build_replica_dataloader(
                    base_dataset=base_dataset,
                    base_sampler=spec_sampler,
                    transform=replica_transform,
                    n_replicas=int(ri_cfg.n_replicas),
                    max_windows=self._eval_config.max_windows,
                    max_eval_seconds=self._eval_config.max_eval_seconds,
                    effective_window_seconds=effective_window_seconds,
                    base_seed=seed,
                )

                # Run the per-spec sweep. We catch failures here (rather
                # than letting them abort the whole job) for tokenizer /
                # spec incompatibilities at runtime — e.g. fixed-length
                # encoders that crash when WindowShift trims the
                # analysis window below their training length, or
                # backend-specific decode errors. The spec is explicitly
                # skipped and recorded.
                #
                # No-op corruption specs (e.g. ``drop_neurons fraction=0.05``
                # with too few units to drop a single one, or
                # ``jitter k_ms=0 p=0``) are NOT skipped — the underlying
                # transforms no-op the data instead of raising, so
                # they produce real (zero-distance) data points that
                # anchor the strength curves at zero. See
                # ``apply_drop_neurons`` for the no-op semantics.
                #
                # We catch ``Exception`` (not BaseException) so KeyboardInterrupt
                # still propagates.
                try:
                    reports = run_validation_evaluation_per_session_replicas(
                        replica_dataloader,
                        _tokenizer_fn,
                        n_replicas=int(ri_cfg.n_replicas),
                        data_type=spec.input_type,
                        reconstructed_data_type=spec.output_type,
                        bin_size_seconds=bin_size_seconds,
                        token_bin_size_ms=spec.token_bin_size_ms,
                        window_length_seconds=(
                            effective_window_seconds
                            if effective_window_seconds is not None
                            else self._resolve_eval_window_seconds()
                        ),
                        eval_config=self._eval_config,
                        device=device,
                        progress=True,
                        corruption_label=label,
                    )
                except Exception as exc:
                    reason = f"{type(exc).__name__}: {exc}"
                    log.warning(
                        f"  skipping spec {label!r}: {reason}",
                        exc_info=True,
                    )
                    skipped_specs.append({"corruption": label, "reason": reason})
                    skipped_counter[label] = skipped_counter.get(label, 0) + 1
                    if jsonl_fh is not None:
                        jsonl_fh.write(
                            json.dumps(
                                {
                                    "corruption": label,
                                    "status": "skipped",
                                    "reason": reason,
                                }
                            )
                            + "\n"
                        )
                    continue

                # Stream one JSONL row per (session, corruption, metric).
                for session_id, entries in reports.items():
                    for segment, payload in entries:
                        report = payload["report"]
                        for metric_name, mres in report.results.items():
                            output_metric_name = metric_name
                            self._capture_metric_schema(metric_schemas, metric_name, mres)
                            raw = mres.raw_value
                            # Pull metric-level metadata once. We want
                            # ``n_windows`` (the actual count the input_robustness (M1) metric
                            # averaged over for this (session, corruption))
                            # to flow into every result row so it survives
                            # to results.csv / results.jsonl.
                            mres_meta = getattr(mres, "metadata", None) or {}
                            n_windows_value = mres_meta.get("n_windows")
                            n_replicas_used_value = mres_meta.get("n_replicas_used")
                            # Stash per-dim diagnostics from input_robustness (M1) (only) so the
                            # per-dim distribution plot can reconstruct
                            # unnormalised per-dim MSE = NMSE_d · σ_d²
                            # without needing reports kept alive across
                            # corruption iterations.
                            if metric_name == "robustness_input":
                                meta = mres_meta
                                # ``sigma_per_dim`` in the metric metadata is σ
                                # (std), not σ². We square it once below to
                                # produce ``captured_sigma_per_dim_sq``.
                                sigma_per_dim = meta.get("sigma_per_dim")
                                if (
                                    sigma_per_dim is not None
                                    and session_id not in captured_sigma_per_dim_sq
                                ):
                                    spd_sq_arr = np.asarray(sigma_per_dim, dtype=float) ** 2
                                    captured_sigma_per_dim_sq[session_id] = spd_sq_arr.tolist()
                                if isinstance(raw, dict):
                                    sess_bd = captured_per_dim_nmse_breakdown.setdefault(
                                        session_id, {}
                                    )
                                    for sub_label, sub_val in raw.items():
                                        if not isinstance(sub_val, dict):
                                            continue
                                        bd = sub_val.get("per_dim_nmse_breakdown")
                                        if bd is not None:
                                            sess_bd[sub_label] = list(bd)
                            # input_robustness (M1)'s raw_value is a dict keyed by corruption
                            # label, with mean_token_nmse/mean_per_dim_nmse
                            # entries (plus their per-replica stds and a per-dim
                            # breakdown). For all other metrics we fall back to
                            # a scalar/array representation.
                            base_row: dict[str, T.Any] = {
                                "session_id": session_id,
                                "segment_index": int(segment.segment_index),
                                "corruption": label,
                                "metric": output_metric_name,
                            }
                            if isinstance(raw, dict):
                                # input_robustness (M1)'s raw_value is keyed by corruption label;
                                # each sub_val carries both metric A
                                # (``mean_token_nmse``, the canonical M1-a value)
                                # and metric B (``mean_per_dim_nmse``, the
                                # per-dim variant). We emit one CSV/JSONL row
                                # per (sub_label, variant): metric A under the
                                # original metric name (so existing M1-a plot
                                # consumers keep working), metric B under
                                # ``<metric>_per_dim`` (single underscore marks
                                # it as a variant of the base metric rather than
                                # a distinct metric name).
                                # Tuple form: (output_metric_name, mean_key, std_key).
                                # The std_key surfaces the metric's replica-spread
                                # (``per_label_std``-equivalent) onto each row so
                                # consumers don't have to recompute it.
                                m1_variants = (
                                    (metric_name, "mean_token_nmse", "std_token_nmse"),
                                    (
                                        f"{metric_name}_per_dim",
                                        "mean_per_dim_nmse",
                                        "std_per_dim_nmse",
                                    ),
                                )
                                for sub_label, sub_val in raw.items():
                                    if isinstance(sub_val, dict):
                                        # Per-sub_label n_replicas_used (when the
                                        # metric reports it as a dict). Falls
                                        # back to the metric-level scalar.
                                        if isinstance(n_replicas_used_value, dict):
                                            row_n_replicas_used = n_replicas_used_value.get(
                                                sub_label
                                            )
                                        else:
                                            row_n_replicas_used = n_replicas_used_value
                                        for variant_metric, variant_key, std_key in m1_variants:
                                            self._capture_metric_schema(
                                                metric_schemas,
                                                variant_metric,
                                                mres,
                                            )
                                            scalar = sub_val.get(variant_key)
                                            scalar_std = sub_val.get(std_key)
                                            rec = {
                                                **base_row,
                                                "metric": variant_metric,
                                                "sub_label": sub_label,
                                                "value": (
                                                    float(scalar) if scalar is not None else None
                                                ),
                                                # std across replicas (per
                                                # corruption sub-label) — the
                                                # actual replica-spread the
                                                # input_robustness (M1) metric computed.
                                                "value_std_replicas": (
                                                    float(scalar_std)
                                                    if scalar_std is not None
                                                    else None
                                                ),
                                                # number of windows the input_robustness
                                                # (M1) metric averaged over for this
                                                # (session, corruption) — this
                                                # is the ground truth of how
                                                # much data the metric saw.
                                                "n_windows": (
                                                    int(n_windows_value)
                                                    if n_windows_value is not None
                                                    else None
                                                ),
                                                "n_replicas_used": (
                                                    int(row_n_replicas_used)
                                                    if row_n_replicas_used is not None
                                                    else None
                                                ),
                                                "status": "ok",
                                            }
                                            _store_result_row(rec)
                                            total_pairs += 1
                                    else:
                                        rec = {
                                            **base_row,
                                            "sub_label": sub_label,
                                            "value": (
                                                float(sub_val)
                                                if hasattr(sub_val, "__float__")
                                                else None
                                            ),
                                            "n_windows": (
                                                int(n_windows_value)
                                                if n_windows_value is not None
                                                else None
                                            ),
                                            "status": "ok",
                                        }
                                        _store_result_row(rec)
                                        total_pairs += 1
                            else:
                                if hasattr(raw, "__len__"):
                                    try:
                                        value = float(np.nanmean(np.asarray(raw)))
                                    except (TypeError, ValueError):
                                        value = None
                                elif raw is None:
                                    value = None
                                else:
                                    try:
                                        value = float(raw)
                                    except (TypeError, ValueError):
                                        value = None
                                rec = {
                                    **base_row,
                                    "sub_label": None,
                                    "value": value,
                                    "n_windows": (
                                        int(n_windows_value)
                                        if n_windows_value is not None
                                        else None
                                    ),
                                    "status": "ok",
                                }
                                _store_result_row(rec)
                                total_pairs += 1

                # Fold per-corruption raster captures into the global
                # stores. ``raster_capture[sid][w_idx]`` is a [K+1, T, U]
                # spike_matrix; row 0 = clean, rows 1..K = corrupted.
                # We pick replica 1 as the representative corrupted
                # spikes for the M1-c plot.
                if capture_enabled:
                    for sid, win_map in raster_capture.items():
                        for w_idx, sm in win_map.items():
                            # ``sm`` has shape [K+1, T, U] but may have a
                            # leading batch dim 1 (when collated as a single
                            # batched window). Squeeze gracefully.
                            arr = np.asarray(sm)
                            if arr.ndim == 4 and arr.shape[0] == 1:
                                arr = arr[0]
                            if arr.ndim != 3 or arr.shape[0] < 2:
                                # Need at least 1 clean + 1 corrupted row.
                                continue
                            captured_clean.setdefault(sid, {})[w_idx] = arr[0]
                            captured_corrupted.setdefault(sid, {}).setdefault(w_idx, {})[label] = (
                                arr[1]
                            )
                            # Latent distance annotation: use the
                            # corruption-mean from the input_robustness (M1) metric in
                            # ``reports`` (mean across replicas+windows).
                            d_lat = _segment_d_latent_for_session(reports, sid)
                            if d_lat is not None:
                                captured_d_latent.setdefault(sid, {}).setdefault(w_idx, {})[
                                    label
                                ] = d_lat

                if jsonl_fh is not None:
                    jsonl_fh.flush()
                # Free replica dataloader + reports between corruptions to
                # bound peak memory.
                del replica_dataloader
                del reports
                gc.collect()
        finally:
            if jsonl_fh is not None:
                jsonl_fh.close()

        csv_rows = _append_core_robustness_score_rows(csv_rows)
        if jsonl_path is not None and jsonl_path.exists():
            with jsonl_path.open("a") as fh:
                for row in csv_rows:
                    if row.get("metric") != _ROBUSTNESS_INPUT_CORE_SCORE_METRIC:
                        continue
                    fh.write(json.dumps(row) + "\n")

        if output_dir is not None:
            self._write_csv(csv_rows, output_dir / "results.csv")
            self._write_per_session_csv(csv_rows, output_dir / "results_per_session.csv")
            per_session_reports, aggregated_report, per_session_weights = (
                self._build_reports_from_m1_rows(csv_rows, metric_schemas)
            )
            self._write_per_session_report_artifacts(
                per_session_reports, output_dir / "per_session"
            )
            poisson_metrics: dict[str, float] = {}
            if self.poisson_reference is not None:
                poisson_payload, poisson_metrics = self._write_poisson_comparison_artifact(
                    output_dir=output_dir,
                    per_session_reports=per_session_reports,
                    aggregated_report=aggregated_report,
                    metric_schemas=metric_schemas,
                    per_session_weights=per_session_weights,
                )
            else:
                poisson_payload = None
            run_meta = self._write_run_meta(
                output_dir / "run_meta.json",
                ri_cfg=ri_cfg,
                skipped_counter=skipped_counter,
                skipped_specs=skipped_specs,
                total_pairs=total_pairs,
                spec=self._spec,
                checkpoint=self.ckpt_path,
                eval_config=self._eval_config,
                data_config=self._provenance_data_config(),
                hydra_overrides=self._hydra_overrides,
                composed_config_yaml=self._provenance_composed_config_yaml(),
                evaluate_all_neurons=self.evaluate_all_neurons,
                neuron_selection=self._neuron_selection,
                eval_window=self._eval_window_record(),
            )
            weighted = self.aggregation_strategy == AggregationStrategy.WEIGHTED_MEAN
            add_robustness_report_fields(
                aggregated_report,
                run_meta,
                self._eval_config,
                split=self.split,
                session_ids=list(per_session_reports),
                aggregation_strategy=self.aggregation_strategy.value,
                session_weights=per_session_weights if weighted else None,
            )
            self._write_eval_report_json(aggregated_report, output_dir / "eval_report.json")
            # Per-dim diagnostics file (consumed by per-dim distribution plot
            # + replotters). Always written; small (D × n_sessions × n_specs).
            with (output_dir / "per_dim_breakdown.json").open("w") as f:
                json.dump(
                    {
                        "sigma_per_dim_sq": captured_sigma_per_dim_sq,
                        "per_dim_nmse_breakdown": captured_per_dim_nmse_breakdown,
                    },
                    f,
                )
            log.info(f"Results written to {output_dir}")

            # ── Eval-time figures ─────────────────────────────────
            if self.generate_plots:
                self._emit_figures(
                    output_dir=output_dir,
                    csv_rows=csv_rows,
                    ri_cfg=ri_cfg,
                    bin_size_seconds=bin_size_seconds,
                    captured_clean=captured_clean,
                    captured_corrupted=captured_corrupted,
                    captured_d_latent=captured_d_latent,
                    captured_sigma_per_dim_sq=captured_sigma_per_dim_sq,
                    captured_per_dim_nmse_breakdown=captured_per_dim_nmse_breakdown,
                )

            # ── MLflow logging ─────────────────────────────────────
            if self.mlflow_logging.enabled:
                self._log_to_mlflow(
                    output_dir=output_dir,
                    ri_cfg=ri_cfg,
                    aggregated_report=aggregated_report,
                    per_session_reports=per_session_reports,
                    extra_metrics=poisson_metrics,
                )

        return locals()

    def _build_eval_data_components(
        self,
    ) -> tuple[NTTDataModule, T.Any, T.Any, T.Any]:
        """Build the dataset/sampler/transform tuple used for input_robustness (M1) evaluation."""
        eval_dm = self.datamodule
        if self.evaluate_all_neurons:
            eval_dm = self._build_all_neuron_datamodule()
        if self._neuron_selection is None:
            assert self._eval_config is not None and self._spec is not None
            self._neuron_selection = record_neuron_selection(
                eval_dm,
                split=self.split,
                eval_config=self._eval_config,
                spec=self._spec,
                evaluate_all_neurons=self.evaluate_all_neurons,
                poisson_reference=self.poisson_reference,
                mode="input_robustness (M1)",
            )

        base_dataset = eval_dm.get_dataset(self.split)
        base_sampler = eval_dm.get_sampler(self.split)
        if base_dataset is None or base_sampler is None:
            raise RuntimeError(
                f"datamodule.{self.split}_dataset / {self.split}_sampler is None; setup() ran?"
            )
        base_transform = eval_dm._build_transforms(self.split)
        return eval_dm, base_dataset, base_sampler, base_transform

    def _build_all_neuron_datamodule(self) -> NTTDataModule:
        """Build an evaluation datamodule that mirrors the main eval all-neuron path:
        every unit, capped at ``eval_config.max_neurons`` by the shared selection."""
        from ntt.core.data.utils import n_neurons_per_session
        from ntt.evaluation.utils.window_contract import apply_eval_window

        assert self._spec is not None
        assert self._eval_config is not None
        require_direct_all_neuron_eval(self._spec, what="input_robustness (M1)")

        all_neuron_config = all_neuron_data_config(self.data_config, self._eval_config, self._spec)
        stage = "validate" if self.split == "val" else "test"

        ds = self.datamodule.val_dataset if self.split == "val" else self.datamodule.test_dataset
        assert ds is not None, "datamodule.setup() must have created the eval dataset"
        neuron_counts = n_neurons_per_session(ds)
        log.info(
            "input_robustness (M1) full-neuron eval: at most %s units per session, a seeded "
            "random subset where there are more (max in dataset: %d)",
            self._eval_config.max_neurons,
            max(neuron_counts.values()),
        )

        apply_eval_window(all_neuron_config, self._spec, self._resolve_eval_window_seconds())
        dm = NTTDataModule(all_neuron_config)
        self._check_scored_window(dm, "input_robustness (M1) all-neuron eval")
        dm.setup(stage)
        return dm

    @staticmethod
    def _capture_metric_schema(
        metric_schemas: dict[str, dict[str, T.Any]],
        metric_name: str,
        metric_result: MetricResult,
    ) -> None:
        if metric_name in metric_schemas:
            return
        metric_schemas[metric_name] = {
            "name": metric_name,
            "category": metric_result.category,
            "direction": metric_result.direction,
            "level": metric_result.level,
            "value_range": metric_result.value_range,
            "source_data_type": metric_result.source_data_type,
        }

    @staticmethod
    def _metric_result_from_schema(
        schema: dict[str, T.Any],
        *,
        name: str,
        raw_value: float,
        metadata: dict[str, T.Any] | None = None,
    ) -> MetricResult:
        return MetricResult(
            name=name,
            category=schema["category"],
            direction=schema["direction"],
            level=schema["level"],
            raw_value=float(raw_value),
            normalized_value=None,
            metadata=metadata or {},
            value_range=tuple(schema["value_range"]),
            source_data_type=schema.get("source_data_type"),
        )

    @staticmethod
    def _row_metric_key(metric: str, sub_label: str | None) -> str:
        return metric if not sub_label else f"{metric}/{sub_label}"

    @staticmethod
    def _zero_strength_m1_label(label: str) -> str | None:
        if label == _ROBUSTNESS_INPUT_CORE_SCORE_LABEL:
            return None
        if label.startswith("permute_neurons"):
            return None
        if label.startswith("window_shift|"):
            return re.sub(r"max_shift_ms=[^|]+", "max_shift_ms=0.0", label)
        if label.startswith("jitter|"):
            return re.sub(r"k_ms=[^|]+", "k_ms=0.0", label)
        if "|fraction=" in label:
            return re.sub(r"fraction=[^|]+", "fraction=0.0", label)
        return None

    def _build_reports_from_m1_rows(
        self,
        rows: list[dict[str, T.Any]],
        metric_schemas: dict[str, dict[str, T.Any]],
    ) -> tuple[dict[str, EvaluationReport], EvaluationReport, dict[str, float]]:
        assert self._eval_config is not None
        grouped: dict[str, dict[str, list[float]]] = {}
        meta_counts: dict[str, dict[str, dict[str, T.Any]]] = {}
        for row in rows:
            if row.get("status") != "ok":
                continue
            session_id = row.get("session_id")
            metric = row.get("metric")
            if session_id is None or metric is None:
                continue
            value = row.get("value")
            if value is None:
                continue
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                continue
            sub_label = row.get("sub_label")
            metric_key = self._row_metric_key(str(metric), str(sub_label) if sub_label else None)
            grouped.setdefault(str(session_id), {}).setdefault(metric_key, []).append(value_f)
            meta_counts.setdefault(str(session_id), {}).setdefault(
                metric_key,
                {
                    "corruption": row.get("corruption"),
                    "sub_label": sub_label,
                    "n_rows": 0,
                    "segment_indices": set(),
                    # Track per-metric-call provenance from each row so
                    # the per_session metadata can surface the metric's
                    # own n_windows / replica-spread instead of dropping
                    # them.
                    "n_windows_values": [],
                    "value_std_replicas_values": [],
                    "n_replicas_used_values": [],
                },
            )
            meta_counts[str(session_id)][metric_key]["n_rows"] += 1
            try:
                meta_counts[str(session_id)][metric_key]["segment_indices"].add(
                    int(row["segment_index"])
                )
            except (KeyError, TypeError, ValueError):
                pass
            row_n_windows = row.get("n_windows")
            if row_n_windows is not None:
                try:
                    meta_counts[str(session_id)][metric_key]["n_windows_values"].append(
                        int(row_n_windows)
                    )
                except (TypeError, ValueError):
                    pass
            row_value_std = row.get("value_std_replicas")
            if row_value_std is not None:
                try:
                    meta_counts[str(session_id)][metric_key]["value_std_replicas_values"].append(
                        float(row_value_std)
                    )
                except (TypeError, ValueError):
                    pass
            row_n_replicas_used = row.get("n_replicas_used")
            if row_n_replicas_used is not None:
                try:
                    meta_counts[str(session_id)][metric_key]["n_replicas_used_values"].append(
                        int(row_n_replicas_used)
                    )
                except (TypeError, ValueError):
                    pass

        per_session_reports: dict[str, EvaluationReport] = {}
        for session_id, metrics in grouped.items():
            report_metrics: dict[str, MetricResult] = {}
            for metric_key, values in metrics.items():
                base_metric = metric_key.split("/", 1)[0]
                schema = metric_schemas.get(base_metric)
                if schema is None:
                    continue
                counts = meta_counts[session_id][metric_key]
                metadata: dict[str, T.Any] = {
                    "n_rows": int(counts["n_rows"]),
                    "n_segments": int(len(counts["segment_indices"])),
                    "corruption": counts.get("corruption"),
                    "sub_label": counts.get("sub_label"),
                    "aggregation_strategy": "mean",
                }
                # Surface the metric's own n_windows / replica-spread.
                # When multiple rows contribute (multiple metric calls
                # per metric_key, e.g. one per flushed segment), record
                # the per-row list and the summary stats.
                n_windows_values = list(counts.get("n_windows_values", []))
                if n_windows_values:
                    metadata["n_windows_per_row"] = n_windows_values
                    metadata["n_windows_total"] = int(sum(n_windows_values))
                value_std_replicas_values = list(counts.get("value_std_replicas_values", []))
                if value_std_replicas_values:
                    metadata["value_std_replicas_per_row"] = value_std_replicas_values
                    metadata["value_std_replicas_mean"] = float(
                        np.nanmean(value_std_replicas_values)
                    )
                n_replicas_used_values = list(counts.get("n_replicas_used_values", []))
                if n_replicas_used_values:
                    metadata["n_replicas_used_per_row"] = n_replicas_used_values
                report_metrics[metric_key] = self._metric_result_from_schema(
                    schema,
                    name=metric_key,
                    raw_value=float(np.mean(values)),
                    metadata=metadata,
                )
            per_session_reports[session_id] = EvaluationReport(results=report_metrics)

        # Per-session weights = unique segment count per session (union across
        # metrics). Mirrors the segment-duration weighting used by the standard
        # eval job's poisson reference path on `main`. Honors the job's
        # ``aggregation_strategy`` so WEIGHTED_MEAN actually weights by segment
        # count instead of falling back to equal weights.
        per_session_weights: dict[str, float] = {}
        for session_id, by_metric in meta_counts.items():
            seg_idxs: set[int] = set()
            for entry in by_metric.values():
                seg_idxs.update(entry.get("segment_indices", set()))
            per_session_weights[str(session_id)] = float(max(1, len(seg_idxs)))

        aggregated_report = aggregate_per_session_dicts(
            {sid: report.to_dict() for sid, report in per_session_reports.items()},
            self.aggregation_strategy,
            weights=per_session_weights,
        )
        return per_session_reports, aggregated_report, per_session_weights

    def _build_m1_ceiling_reports(
        self,
        per_session_reports: dict[str, EvaluationReport],
    ) -> dict[str, EvaluationReport]:
        """Build per-session ceiling reports for M1 normalization. The
        ceiling is pinned to the theoretical optimum (M1 NMSE = 0) for
        every corruption metric.

        Mirrors ``poisson_post_hoc._m1_ceiling_report``; see that docstring
        for the rationale (avoiding the σ₀-noise-driven small denominator
        on DiffusionPriorBottleneck tokenizers, and bounding the
        normalized score at ≤ 1).
        """
        out: dict[str, EvaluationReport] = {}
        for session_id, report in per_session_reports.items():
            ceiling_metrics: dict[str, MetricResult] = {}
            for metric_name, metric_result in report.results.items():
                if "/" not in metric_name:
                    continue
                _, label = metric_name.split("/", 1)
                zero_label = self._zero_strength_m1_label(label)
                if zero_label is None:
                    continue
                ceiling_metrics[metric_name] = self._metric_result_from_schema(
                    {
                        "category": metric_result.category,
                        "direction": metric_result.direction,
                        "level": metric_result.level,
                        "value_range": metric_result.value_range,
                        "source_data_type": metric_result.source_data_type,
                    },
                    name=metric_name,
                    raw_value=0.0,
                    metadata={"ceiling_source_metric": "theoretical_zero"},
                )
            if ceiling_metrics:
                out[session_id] = EvaluationReport(results=ceiling_metrics)
        return out

    def _load_poisson_m1_reference(
        self,
        metric_schemas: dict[str, dict[str, T.Any]],
    ) -> tuple[dict[str, EvaluationReport], EvaluationReport, float | None]:
        assert self.poisson_reference is not None

        csv_path, run_meta_path = resolve_poisson_reference_paths(
            self.poisson_reference, mode="input_robustness (M1)"
        )

        rows: list[dict[str, T.Any]] = []
        import csv

        with csv_path.open() as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                # ``segment_index`` is dropped here because the per-session
                # CSV is already aggregated over segments. ``_build_reports_…``
                # derives its per-session weight from the number of distinct
                # segment indices, so this collapses every poisson session to
                # weight=1. The model-side aggregation still uses real
                # per-session segment weights, so aggregate-level
                # normalization mixes a weighted model with an unweighted
                # baseline. This is benign while every session contributes a
                # single segment (the current setup); revisit when multi-
                # segment-per-session evaluation is enabled.
                rows.append(
                    {
                        "session_id": row.get("session_id"),
                        "metric": row.get("metric"),
                        "sub_label": row.get("sub_label"),
                        "corruption": row.get("corruption"),
                        "value": row.get("value_mean"),
                        "status": "ok",
                        "segment_index": None,
                    }
                )
        per_session_reports, aggregated_report, _per_session_weights = (
            self._build_reports_from_m1_rows(rows, metric_schemas)
        )
        if rows and not per_session_reports:
            log.warning(
                "input_robustness (M1) poisson_reference at %s loaded %d CSV "
                "rows but produced zero "
                "per-session reports — likely a metric_schemas mismatch (rows are "
                "skipped when the metric's base name is not in the live schema "
                "registry). Downstream comparison artifacts will be empty.",
                csv_path,
                len(rows),
            )
        poisson_rate_bin_ms = load_poisson_rate_bin_ms(run_meta_path, mode="M1", log=log)
        return per_session_reports, aggregated_report, poisson_rate_bin_ms

    @staticmethod
    def _write_eval_report_json(report: EvaluationReport, path: Path) -> None:
        path.write_text(json.dumps(report.to_dict(), indent=2))

    @staticmethod
    def _write_per_session_report_artifacts(
        per_session_reports: dict[str, EvaluationReport],
        output_dir: Path,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for session_id, report in per_session_reports.items():
            safe_sid = session_id.replace("/", "__")
            (output_dir / f"{safe_sid}.json").write_text(json.dumps(report.to_dict(), indent=2))

    def _write_poisson_comparison_artifact(
        self,
        *,
        output_dir: Path,
        per_session_reports: dict[str, EvaluationReport],
        aggregated_report: EvaluationReport,
        metric_schemas: dict[str, dict[str, T.Any]],
        per_session_weights: dict[str, float] | None = None,
    ) -> tuple[dict[str, T.Any], dict[str, float]]:
        assert self._eval_config is not None
        poisson_per_session, poisson_aggregated, poisson_rate_bin_ms = (
            self._load_poisson_m1_reference(metric_schemas)
        )
        ceiling_per_session = self._build_m1_ceiling_reports(per_session_reports)

        per_session_comparisons: dict[str, dict[str, T.Any]] = {}
        normalized_by_session_metric: dict[str, list[float]] = {}
        extra_metrics: dict[str, float] = {}
        shared_sessions = sorted(
            warn_if_no_session_overlap(
                model_sessions=set(per_session_reports),
                poisson_sessions=set(poisson_per_session),
                ceiling_sessions=set(ceiling_per_session),
                mode="M1",
                log=log,
            )
        )
        for session_id in shared_sessions:
            comparisons = compute_normalized_comparisons(
                per_session_reports[session_id],
                poisson_per_session[session_id],
                ceiling_per_session[session_id],
            )
            if not comparisons:
                continue
            per_session_comparisons[session_id] = {
                name: comp.to_dict() for name, comp in comparisons.items()
            }
            safe_sid = session_id.replace("/", "__")
            for metric_name, comp in comparisons.items():
                if comp.normalized_score is None or comp.status != "ok":
                    continue
                normalized_by_session_metric.setdefault(metric_name, []).append(
                    float(comp.normalized_score)
                )
                extra_metrics[
                    sanitize_metric_name_for_mlflow(
                        f"{self.mlflow_logging.metric_prefix}/normalized/per_session/"
                        f"{safe_sid}/{metric_name}"
                    )
                ] = float(comp.normalized_score)

        # Aggregate model / poisson / ceiling over the SHARED session subset
        # so all three describe the same population (using each side's full
        # aggregate would bias the aggregate-first comparison whenever the
        # model and poisson session sets differ, e.g. when the poisson run
        # was evaluated on extra sessions the model didn't see).
        # Per-session normalization likewise restricts to ``shared_sessions``.
        shared_set = set(shared_sessions)
        model_shared = {
            sid: report for sid, report in per_session_reports.items() if sid in shared_set
        }
        poisson_shared = {
            sid: report for sid, report in poisson_per_session.items() if sid in shared_set
        }
        ceiling_shared = {
            sid: report for sid, report in ceiling_per_session.items() if sid in shared_set
        }
        # Keep using job-configured strategy + segment-count weights, but
        # restrict weights to the shared session set as well.
        shared_model_weights = (
            {sid: w for sid, w in per_session_weights.items() if sid in shared_set}
            if per_session_weights is not None
            else None
        )
        model_aggregated_shared = aggregate_per_session_dicts(
            {sid: report.to_dict() for sid, report in model_shared.items()},
            self.aggregation_strategy,
            weights=shared_model_weights,
        )
        poisson_aggregated_shared = aggregate_per_session_dicts(
            {sid: report.to_dict() for sid, report in poisson_shared.items()},
            self.aggregation_strategy,
            weights=shared_model_weights,
        )
        ceiling_aggregated = aggregate_per_session_dicts(
            {sid: report.to_dict() for sid, report in ceiling_shared.items()},
            self.aggregation_strategy,
            weights=shared_model_weights,
        )
        aggregate_comparisons = compute_normalized_comparisons(
            model_aggregated_shared,
            poisson_aggregated_shared,
            ceiling_aggregated,
        )
        for metric_name, score in extract_normalized_scores(aggregate_comparisons).items():
            extra_metrics[
                sanitize_metric_name_for_mlflow(
                    f"{self.mlflow_logging.metric_prefix}/normalized/{metric_name}"
                )
            ] = float(score)

        normalized_per_session_mean = {
            metric_name: float(np.mean(values))
            for metric_name, values in normalized_by_session_metric.items()
            if values
        }
        for metric_name, score in normalized_per_session_mean.items():
            extra_metrics[
                sanitize_metric_name_for_mlflow(
                    f"{self.mlflow_logging.metric_prefix}/normalized_per_session_mean/{metric_name}"
                )
            ] = float(score)

        # Two normalized-score flavors are kept side by side; each answers a
        # different question (see docs/robustness_metrics.md "Poisson
        # normalization" for the full breakdown):
        #   • normalized_scores_vs_poisson — aggregate model / poisson /
        #     ceiling across sessions FIRST, then normalize the aggregates
        #     once. Matches what the main reconstruction-eval pipeline does
        #     (run_validation_evaluation_per_session → eval.py).
        #     "If I aggregate everything first, how much better than Poisson?"
        #   • normalized_scores_from_per_session_mean — normalize each
        #     session against ITS OWN poisson + ceiling, then take an
        #     unweighted mean across sessions. "How does the model compare
        #     on a typical session?" Less sensitive to one large session
        #     dominating the aggregate.
        # The two collapse to the same value when sessions are equally
        # weighted and the metric is linear in the underlying counts; they
        # diverge when sessions differ in size, the metric is nonlinear, or
        # only some sessions overlap between model and poisson runs.
        payload = {
            "poisson_reference": str(self.poisson_reference),
            "poisson_rate_bin_ms": poisson_rate_bin_ms,
            "aggregate_vs_poisson": {
                name: comp.to_dict() for name, comp in aggregate_comparisons.items()
            },
            "normalized_scores_vs_poisson": extract_normalized_scores(aggregate_comparisons),
            "per_session": per_session_comparisons,
            "normalized_scores_from_per_session_mean": normalized_per_session_mean,
        }
        (output_dir / "poisson_comparison.json").write_text(json.dumps(payload, indent=2))
        return payload, extra_metrics

    # ── Helpers ─────────────────────────────────────────────────────

    def _get_output_dir(self) -> Path | None:
        """Return ``output_path`` as a ``Path`` (created if missing), or None."""
        if self.output_path is None:
            return None
        out = Path(self.output_path)
        out.mkdir(parents=True, exist_ok=True)
        return out

    @staticmethod
    def _write_csv(rows: list[dict[str, T.Any]], path: Path) -> None:
        """Write the long CSV: one row per (session, corruption, metric, sub_label)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            path.write_text("")
            return
        output_rows: list[dict[str, T.Any]] = []
        for row in rows:
            out_row = dict(row)
            if "value" in out_row:
                out_row["raw_value"] = out_row.pop("value")
            output_rows.append(out_row)
        try:
            import pandas as pd

            df = pd.DataFrame(output_rows)
            df.to_csv(path, index=False)
        except ImportError:
            import csv

            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(output_rows[0].keys()))
                writer.writeheader()
                writer.writerows(output_rows)

    @staticmethod
    def _write_per_session_csv(rows: list[dict[str, T.Any]], path: Path) -> None:
        """Write one input_robustness (M1) summary row per (session, corruption,
        metric, sub_label).

        The raw ``results.csv`` is segment-level long-form. This helper writes
        an explicit per-session summary that averages over all segment rows for
        the same sweep cell so downstream tooling does not need to rediscover
        that aggregation each time.
        """
        grouped: dict[tuple[str, str, str, str], list[float]] = {}
        segment_ids: dict[tuple[str, str, str, str], set[int]] = {}
        # Per-group accumulators for the metric-side provenance columns the
        # row writer carries (see ``_store_result_row``). Kept here so
        # ``results_per_session.csv`` exposes them as explicit columns
        # rather than forcing consumers to grep ``results.csv``.
        n_windows_acc: dict[tuple[str, str, str, str], list[int]] = {}
        replica_std_acc: dict[tuple[str, str, str, str], list[float]] = {}
        n_replicas_used_acc: dict[tuple[str, str, str, str], list[int]] = {}
        for row in rows:
            if row.get("status") != "ok":
                continue
            session_id = row.get("session_id")
            corruption = row.get("corruption")
            metric = row.get("metric")
            sub_label = row.get("sub_label") or ""
            value = row.get("value")
            if session_id is None or corruption is None or metric is None or value is None:
                continue
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                continue
            key = (str(session_id), str(corruption), str(metric), str(sub_label))
            grouped.setdefault(key, []).append(value_f)
            try:
                seg_idx = int(row["segment_index"])
            except (KeyError, TypeError, ValueError):
                seg_idx = -1
            segment_ids.setdefault(key, set()).add(seg_idx)
            row_n_windows = row.get("n_windows")
            if row_n_windows is not None:
                try:
                    n_windows_acc.setdefault(key, []).append(int(row_n_windows))
                except (TypeError, ValueError):
                    pass
            row_value_std = row.get("value_std_replicas")
            if row_value_std is not None:
                try:
                    replica_std_acc.setdefault(key, []).append(float(row_value_std))
                except (TypeError, ValueError):
                    pass
            row_n_replicas_used = row.get("n_replicas_used")
            if row_n_replicas_used is not None:
                try:
                    n_replicas_used_acc.setdefault(key, []).append(int(row_n_replicas_used))
                except (TypeError, ValueError):
                    pass

        summary_rows: list[dict[str, T.Any]] = []
        for (session_id, corruption, metric, sub_label), values in sorted(grouped.items()):
            key = (session_id, corruption, metric, sub_label)
            nw_vals = n_windows_acc.get(key, [])
            rs_vals = replica_std_acc.get(key, [])
            nru_vals = n_replicas_used_acc.get(key, [])
            summary_rows.append(
                {
                    "session_id": session_id,
                    "corruption": corruption,
                    "metric": metric,
                    "sub_label": (sub_label or None),
                    "value_mean": float(np.mean(values)),
                    # std across the result-rows that contributed to this
                    # group. With a single contributing row this is 0.0 by
                    # construction — the meaningful spread is
                    # ``value_std_replicas_mean`` below, which is the
                    # average of the metric's own replica-spread std.
                    "value_std": float(np.std(values)),
                    "n_rows": int(len(values)),
                    "n_segments": int(
                        len(
                            {
                                x
                                for x in segment_ids[(session_id, corruption, metric, sub_label)]
                                if x >= 0
                            }
                        )
                    ),
                    # Number of windows the input_robustness (M1) metric averaged over for
                    # this (session, corruption, metric, sub_label). Sum
                    # across contributing rows when more than one metric
                    # call landed in the group. ``None`` if the source
                    # rows did not carry it.
                    "n_windows_total": (int(sum(nw_vals)) if nw_vals else None),
                    "n_windows_per_row": (nw_vals if nw_vals else None),
                    # Replica-spread std the input_robustness (M1) metric computed (one std
                    # per contributing row). Mean across rows is the
                    # representative number; the list is preserved for
                    # diagnostics.
                    "value_std_replicas_mean": (float(np.nanmean(rs_vals)) if rs_vals else None),
                    "value_std_replicas_per_row": (rs_vals if rs_vals else None),
                    "n_replicas_used_per_row": (nru_vals if nru_vals else None),
                }
            )
        NTTEvalRobustnessInputJob._write_csv(summary_rows, path)

    @staticmethod
    def _write_run_meta(
        path: Path,
        ri_cfg: T.Any,
        skipped_counter: dict[str, int],
        total_pairs: int,
        skipped_specs: list[dict[str, T.Any]] | None = None,
        spec: T.Any | None = None,
        checkpoint: str | Path | None = None,
        eval_config: T.Any | None = None,
        data_config: T.Any | None = None,
        hydra_overrides: list[str] | None = None,
        composed_config_yaml: str | None = None,
        evaluate_all_neurons: bool | None = None,
        neuron_selection: NeuronSelection | None = None,
        eval_window: dict[str, T.Any] | None = None,
    ) -> dict[str, T.Any]:
        """Write ``run_meta.json`` per D11 and return its contents.

        ``skipped_specs`` is the list of fully-skipped specs (one entry
        per spec the runner aborted on, with ``corruption`` + ``reason``).
        It is surfaced so fully-empty specs (e.g. ``window_shift`` when
        the tokenizer rejects the analysis window) are visible without
        trawling logs.

        ``spec`` (optional) carries the resolved :class:`TokenizerSpec`
        so we can persist tokenizer-identifying knobs (``KIND`` plus
        spec-specific fields like ``rate_estimation_bin_ms`` for
        Poisson) needed by the replot CLI to disambiguate variants in
        joint legends. The metadata also carries the checkpoint path when
        present plus the full serialized tokenizer config, matching the
        main eval's run-identification payload.

        """
        meta = {
            "n_replicas": int(ri_cfg.n_replicas),
            "n_corruptions": len(ri_cfg.corruptions),
            "corruptions": [_corruption_label(p) for p in ri_cfg.corruptions],
            "skipped_per_metric": dict(skipped_counter),
            "skipped_specs": list(skipped_specs) if skipped_specs else [],
            "total_pairs": int(total_pairs),
            "checkpoint": str(checkpoint) if checkpoint is not None else None,
        }
        # Units each session was scored on; a robustness comparison checks it against
        # the Poisson reference run's.
        meta["evaluate_all_neurons"] = evaluate_all_neurons
        meta.update(eval_window or {})  # the window actually scored, and the spec's own
        meta["neuron_selection"] = (
            neuron_selection.model_dump() if neuron_selection is not None else None
        )
        meta.update(
            write_config_provenance_sidecar(
                run_meta_path=path,
                eval_config=eval_config,
                data_config=data_config,
                hydra_overrides=hydra_overrides if hydra_overrides is not None else [],
                composed_config_yaml_text=composed_config_yaml,
            )
        )
        if spec is not None:
            tokenizer_config: dict[str, T.Any] = {}
            try:
                tokenizer_config = config_to_jsonable(spec, name="tokenizer_config")
            except Exception as e:
                log.warning(f"Failed to serialize tokenizer config: {e}. Saving KIND only.")
                tokenizer_config = {"KIND": getattr(spec, "KIND", "unknown")}
            meta["tokenizer_config"] = tokenizer_config
            kind = getattr(spec, "KIND", None)
            if kind is not None:
                meta["tokenizer_kind"] = str(kind)
            # Surface tokenizer-specific knobs that ought to be visible
            # in joint plot legends. Currently only Poisson's
            # rate_estimation_bin_ms.
            rate_bin = getattr(spec, "rate_estimation_bin_ms", None)
            if rate_bin is not None:
                try:
                    meta["rate_estimation_bin_ms"] = float(rate_bin)
                except (TypeError, ValueError):
                    pass
        with path.open("w") as f:
            json.dump(meta, f, indent=2)
        return meta

    # ── MLflow logging ──────────────────────────────────────────────

    def _log_to_mlflow(
        self,
        *,
        output_dir: Path,
        ri_cfg: T.Any,
        aggregated_report: EvaluationReport | None = None,
        per_session_reports: dict[str, EvaluationReport] | None = None,
        extra_metrics: dict[str, float] | None = None,
    ) -> None:
        """Log this input_robustness (M1) run to MLflow.

        Reads the just-written ``results.jsonl`` and emits a step-indexed
        sweep curve per (session, corruption-family, metric). Uploads only
        compact artifacts by default; raw replot/debug sidecars and
        eval-time figures remain local unless explicitly requested via
        ``mlflow_logging.upload_artifacts_for_replotting``. The run is
        tagged with ``ntt.eval.checkpoint_path`` to link back to the main
        pass; the checkpoint bytes are not re-uploaded.
        """
        from ntt.evaluation.mlflow_logging import (
            log_robustness_run_to_mlflow,
        )

        # Build label → (family, strength) lookup from this run's specs.
        label_to_type_strength: dict[str, tuple[str, float]] = {
            _corruption_label(p): (p.type, _corruption_strength(p)) for p in ri_cfg.corruptions
        }
        label_to_type_strength[_ROBUSTNESS_INPUT_CORE_SCORE_LABEL] = ("aggregate", 0.0)

        jsonl_path = output_dir / "results.jsonl"
        if not jsonl_path.exists():
            log.warning(f"input_robustness (M1) MLflow logging: {jsonl_path} not found; skipping.")
            return

        # Keep the default artifact bundle compact; raw replot/debug
        # sidecars are uploaded only when explicitly requested.
        artifacts: list[Path] = []
        for p in [
            output_dir / "results.csv",
            output_dir / "run_meta.json",
            output_dir / "config_snapshot.yaml",
            output_dir / "eval_report.json",
        ]:
            if p.exists():
                artifacts.append(p)
        include_jsonl_artifact = False
        if self.mlflow_logging.upload_artifacts_for_replotting:
            include_jsonl_artifact = jsonl_path.exists()
            for p in [
                output_dir / "results_per_session.csv",
                output_dir / "poisson_comparison.json",
            ]:
                if p.exists():
                    artifacts.append(p)
            per_session_dir = output_dir / "per_session"
            if per_session_dir.is_dir():
                for p in per_session_dir.rglob("*.json"):
                    artifacts.append(p)
            # M1-a (core curves) — multiple file extensions per save_fig_multi.
            curves_dir = output_dir / "curves"
            if curves_dir.is_dir():
                for p in curves_dir.glob("input-corruption-a*"):
                    artifacts.append(p)
            # M1-c (example rasters) when produced.
            rasters_dir = output_dir / "rasters"
            if rasters_dir.is_dir():
                for p in rasters_dir.rglob("*"):
                    if p.is_file():
                        artifacts.append(p)

        # Resolve checkpoint for upload only when explicitly requested
        # (standalone input_robustness (M1) runs that want the ckpt bytes in MLflow).
        try:
            import tempfile

            from ntt.core.model.checkpoint_io import resolve_checkpoint_for_eval

            with tempfile.TemporaryDirectory(prefix="ntt_m1_ckpt_") as tmpdir:
                ckpt_local_path: Path | None = None
                if self.mlflow_logging.upload_checkpoint:
                    assert self._tokenizer is not None and self._spec is not None
                    ckpt_local_path = resolve_checkpoint_for_eval(
                        ckpt_path=self.ckpt_path,
                        live_tokenizer=self._tokenizer,
                        live_spec=self._spec,
                        always_copy=self.mlflow_logging.always_copy_checkpoint,
                        tmpdir=Path(tmpdir),
                    )
                run_id = log_robustness_run_to_mlflow(
                    mode="m1",
                    config=self.mlflow_logging,
                    jsonl_path=jsonl_path,
                    report=aggregated_report,
                    per_session_reports=per_session_reports,
                    label_to_type_strength=label_to_type_strength,
                    checkpoint_path=(str(self.ckpt_path) if self.ckpt_path else None),
                    checkpoint_local_path=ckpt_local_path,
                    artifact_paths=artifacts,
                    artifact_root=output_dir,
                    include_jsonl_artifact=include_jsonl_artifact,
                    extra_tags={"ntt.eval.split": self.split},
                    extra_metrics=extra_metrics,
                    run_metadata=json.loads((output_dir / "run_meta.json").read_text()),
                )
            log.info(f"input_robustness (M1) results logged to MLflow run: {run_id}")
        except Exception:
            log.warning("Failed to log input_robustness (M1) results to MLflow", exc_info=True)

    # ── Plotting ────────────────────────────────────────────────────

    def _emit_figures(
        self,
        *,
        output_dir: Path,
        csv_rows: list[dict[str, T.Any]],
        ri_cfg: T.Any,
        bin_size_seconds: float,
        captured_clean: dict[str, dict[int, np.ndarray]],
        captured_corrupted: dict[str, dict[int, dict[str, np.ndarray]]],
        captured_d_latent: dict[str, dict[int, dict[str, float]]],
        captured_sigma_per_dim_sq: dict[str, list[float]] | None = None,
        captured_per_dim_nmse_breakdown: dict[str, dict[str, list[float]]] | None = None,
    ) -> None:
        """Emit input_robustness (M1) aggregate curves (always) and M1-c rasters
        (when ``plot_examples``).

        Each plotting block is wrapped in try/except so a single failure
        does not abort the eval — the failure is logged with the offending
        path/window plus the traceback.
        """
        import matplotlib.pyplot as plt

        # ---- aggregate curves: input-corruption-a (joint NMSE) -----
        try:
            payload_a = self._build_curve_payload(csv_rows, ri_cfg, metric_name="robustness_input")
            if payload_a:
                fig = plot_input_corruption_a(payload_a)
                save_fig_multi(fig, output_dir / "curves" / "input-corruption-a")
                plt.close(fig)
            else:
                log.warning(
                    "input-corruption-a: no rows matched metric='robustness_input';"
                    " skipping aggregate curve."
                )
        except Exception:
            log.warning(
                "input-corruption-a plotting failed at"
                f" {output_dir / 'curves' / 'input-corruption-a'}\n"
                f"{traceback.format_exc()}"
            )

        # ---- per-dim distribution diagnostic --------------------
        # One panel per session, showing how the unnormalised per-dim MSE
        # distributes across latent dims (sorted by decreasing variance)
        # for the median-strength jitter spec. Helps spot off-manifold
        # corruptions where MSE_d stays flat as σ_d² drops orders of
        # magnitude.
        try:
            if captured_sigma_per_dim_sq and captured_per_dim_nmse_breakdown:
                # Pick the middle jitter spec (median strength) — tends to
                # be in the dynamic range of "real but not adversarial".
                jitter_specs = [p for p in ri_cfg.corruptions if p.type == "jitter"]
                jitter_specs.sort(key=_corruption_strength)
                if jitter_specs:
                    middle = jitter_specs[len(jitter_specs) // 2]
                    middle_label = _corruption_label(middle)
                    for sid, sigma_sq in captured_sigma_per_dim_sq.items():
                        nmse_d = captured_per_dim_nmse_breakdown.get(sid, {}).get(middle_label)
                        if nmse_d is None:
                            continue
                        fig = plot_input_corruption_per_dim_distribution(
                            sigma_per_dim_sq=sigma_sq,
                            per_dim_nmse_breakdown=nmse_d,
                            corruption_label=middle_label,
                            session_id=sid,
                        )
                        save_fig_multi(
                            fig,
                            output_dir
                            / "curves"
                            / f"input-corruption-per-dim-dist_{_safe_filename(sid)}",
                        )
                        plt.close(fig)
                else:
                    log.warning(
                        "input-corruption-per-dim-dist: no jitter specs found;"
                        " skipping per-dim distribution plot."
                    )
            else:
                log.warning(
                    "input-corruption-per-dim-dist: missing captured per-dim"
                    " diagnostics; skipping."
                )
        except Exception:
            log.warning(
                f"input-corruption-per-dim-dist plotting failed:\n{traceback.format_exc()}"
            )

        # ---- aggregate curves: input-corruption-a-per-dim (per-dim NMSE) -----
        # Same plot shape as M1-a above, but driven by the per-dim NMSE
        # variant. Off-manifold corruptions (e.g. neuron permutation) on
        # population tokenizers tend to grow this metric faster than the
        # joint variant. Stored under a distinct file name so replotters
        # can pick either curve.
        try:
            payload_a_pd = self._build_curve_payload(
                csv_rows, ri_cfg, metric_name="robustness_input_per_dim"
            )
            if payload_a_pd:
                fig = plot_input_corruption_a(payload_a_pd)
                save_fig_multi(fig, output_dir / "curves" / "input-corruption-a-per-dim")
                plt.close(fig)
            else:
                log.warning(
                    "input-corruption-a-per-dim: no rows matched"
                    " metric='robustness_input_per_dim'; skipping aggregate curve."
                )
        except Exception:
            log.warning(
                "input-corruption-a-per-dim plotting failed at"
                f" {output_dir / 'curves' / 'input-corruption-a-per-dim'}\n"
                f"{traceback.format_exc()}"
            )

        # ---- rasters: input-corruption-c (eval-time only) -----------
        if not ri_cfg.plot_examples:
            return

        rasters_dir = output_dir / "rasters"
        for sid, win_map in captured_clean.items():
            for w_idx, clean_arr in win_map.items():
                base_path = rasters_dir / f"input-corruption-c_{_safe_filename(sid)}_w{w_idx}"
                try:
                    corrupted_map = captured_corrupted.get(sid, {}).get(w_idx, {})
                    if not corrupted_map:
                        log.warning(
                            f"input-corruption-c: no corrupted rasters captured for"
                            f" session={sid} window={w_idx}; skipping {base_path}."
                        )
                        continue
                    d_latent_map = captured_d_latent.get(sid, {}).get(w_idx, {})
                    fig = plot_input_corruption_c(
                        window_idx=w_idx,
                        session_id=sid,
                        clean_spikes=clean_arr,
                        corrupted_spikes_per_label=corrupted_map,
                        latent_distance_per_label=d_latent_map,
                        bin_size_ms=float(bin_size_seconds * 1000.0),
                        output_stem=base_path,
                    )
                    plt.close(fig)
                except Exception:
                    log.warning(
                        f"input-corruption-c plotting failed at {base_path}\n"
                        f"{traceback.format_exc()}"
                    )

    def _build_curve_payload(
        self,
        csv_rows: list[dict[str, T.Any]],
        ri_cfg: T.Any,
        metric_name: str,
    ) -> dict[str, dict[str, list[float]]]:
        """Collapse ``csv_rows`` into the per-corruption-type curve payload.

        For each corruption type seen in ``ri_cfg.corruptions`` we collect
        the (strength, value) pairs across the sweep, then aggregate
        (mean, std) across sessions+sub_labels for each strength.
        Returns ``{type → {strengths, means, stds}}``; types with no rows
        are omitted.
        """
        # corruption_label → (type, strength)
        label_to_type: dict[str, tuple[str, float]] = {}
        for params in ri_cfg.corruptions:
            label_to_type[_corruption_label(params)] = (
                params.type,
                _corruption_strength(params),
            )

        # type → strength → list[value]
        bucket: dict[str, dict[float, list[float]]] = {}
        for row in csv_rows:
            if row.get("metric") != metric_name:
                continue
            v = row.get("value")
            if v is None:
                continue
            label = row.get("corruption")
            if label is None:
                continue
            tp = label_to_type.get(label)
            if tp is None:
                continue
            t, strength = tp
            bucket.setdefault(t, {}).setdefault(float(strength), []).append(float(v))

        payload: dict[str, dict[str, list[float]]] = {}
        for t, strengths_map in bucket.items():
            xs = sorted(strengths_map.keys())
            means = [float(np.mean(strengths_map[x])) for x in xs]
            stds = [
                float(np.std(strengths_map[x], ddof=0)) if len(strengths_map[x]) > 1 else 0.0
                for x in xs
            ]
            # No synthetic zero-strength prepend: for stochastic
            # encoders, replica-0 and replica-k for a true no-op spec
            # may still differ (sampling in latent space), and the
            # empirical zero point captures that noise floor. The
            # default sweep includes a real ``jitter k_ms=0 p=0`` head
            # spec named "zero" so this point comes from data.
            payload[t] = {"strengths": list(xs), "means": means, "stds": stds}
        return payload


def _segment_d_latent_for_session(
    reports: dict[str, list[tuple[T.Any, dict[str, T.Any]]]],
    session_id: str,
) -> float | None:
    """Return the corruption-mean ``mean_token_nmse`` from the input_robustness
    (M1) metric for ``session_id``.

    Falls back to ``None`` if the input_robustness (M1) metric did not run /
    produce a usable raw_value mapping.
    """
    entries = reports.get(session_id) or []
    for _segment, payload in entries:
        report = payload.get("report")
        if report is None:
            continue
        results = getattr(report, "results", {}) or {}
        mres = results.get("robustness_input")
        if mres is None:
            continue
        raw = getattr(mres, "raw_value", None)
        if not isinstance(raw, dict):
            continue
        # raw is a mapping ``replica_label → {mean_token_nmse, ...}``; we
        # average mean_token_nmse across replicas to get a single scalar
        # representative for the corruption applied in this iteration.
        joints: list[float] = []
        for sub_val in raw.values():
            if isinstance(sub_val, dict):
                mj = sub_val.get("mean_token_nmse")
                if mj is not None:
                    joints.append(float(mj))
        if joints:
            return float(np.mean(joints))
    return None


def _safe_filename(name: str) -> str:
    """Sanitize ``name`` for use as a filesystem path component."""
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name)


def _append_core_robustness_score_rows(
    rows: list[dict[str, T.Any]],
) -> list[dict[str, T.Any]]:
    """Append one aggregate input_robustness (M1) score row per (session, segment).

    The score averages the canonical ``robustness_input`` values for the
    three non-population corruption settings requested by the user:
    3 ms window shift, 1 ms jitter at p=0.2, and 2% drop+add.
    """
    grouped: dict[tuple[str, int], dict[str, list[float]]] = {}
    keys_with_robustness_input: set[tuple[str, int]] = set()
    for row in rows:
        if row.get("metric") != "robustness_input":
            continue
        if row.get("status") != "ok":
            continue
        try:
            seen_key = (str(row.get("session_id")), int(row["segment_index"]))
        except (KeyError, TypeError, ValueError):
            seen_key = None
        if seen_key is not None:
            keys_with_robustness_input.add(seen_key)
        label = row.get("corruption")
        if label not in _ROBUSTNESS_INPUT_CORE_SCORE_COMPONENTS:
            continue
        value = row.get("value")
        if value is None:
            continue
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            continue
        try:
            segment_index = int(row["segment_index"])
        except (KeyError, TypeError, ValueError):
            continue
        key = (str(row.get("session_id")), segment_index)
        grouped.setdefault(key, {}).setdefault(str(label), []).append(value_f)

    derived_rows: list[dict[str, T.Any]] = []
    needed = set(_ROBUSTNESS_INPUT_CORE_SCORE_COMPONENTS)
    keys_scored: set[tuple[str, int]] = set()
    for (session_id, segment_index), per_label in grouped.items():
        if set(per_label) != needed:
            continue
        values = [
            float(np.mean(per_label[label])) for label in _ROBUSTNESS_INPUT_CORE_SCORE_COMPONENTS
        ]
        keys_scored.add((session_id, segment_index))
        derived_rows.append(
            {
                "session_id": session_id,
                "segment_index": segment_index,
                "corruption": _ROBUSTNESS_INPUT_CORE_SCORE_LABEL,
                "metric": _ROBUSTNESS_INPUT_CORE_SCORE_METRIC,
                "sub_label": None,
                "value": float(np.mean(values)),
                "status": "ok",
            }
        )

    # The core components are matched by exact float-formatted label string
    # (e.g. ``drop_and_add|fraction=0.02``). A non-default config silently yields
    # zero core-score rows; warn rather than drop the headline metric in silence.
    unscored = sorted(keys_with_robustness_input - keys_scored)
    if unscored:
        log.warning(
            "robustness_input_core_avg could not be computed for %d "
            "(session, segment) group(s) with robustness_input results: the core "
            "components %s were not all present (exact-label match). This usually "
            "means a non-default corruption config (e.g. a drop fraction other "
            "than 0.02). Affected: %s",
            len(unscored),
            sorted(_ROBUSTNESS_INPUT_CORE_SCORE_COMPONENTS),
            unscored,
        )
    return rows + derived_rows
