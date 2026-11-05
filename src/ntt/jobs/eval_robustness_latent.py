"""Standalone latent_robustness (M2) evaluation job.

Known limitations
-----------------
This job is windowed but not fully streaming. For each session it
collates the capped window set (``n_eval_windows`` / ``max_eval_seconds``)
and holds all clean latents in memory before corrupting and scoring them
per window: clean latents and reconstructions are computed once, then each
window draws ``k`` corrupted latents and accumulates windowed
reconstruction metrics against the clean reconstruction. Peak memory is
therefore bounded by the whole capped window set rather than a single
window, which can bottleneck large tokenizers or high window counts.

Tokenizers that sample spikes in ``reconstruct`` (calm with
``reconstruct_sample`` bernoulli/poisson, the Poisson baseline) are decoded to
expected values; M2 samples the clean and every noisy reconstruction of a window
from one shared uniform field, so the score reflects the latents, not sampling noise.


Mirrors :mod:`ntt.jobs.eval` (``NTTEvalJob``) but implements the latent_robustness (M2)
sweep described in design doc §D7: for each window and each
(ε, k) pair, draw Gaussian/uniform latent noise scaled **per latent
dim** (``ξ_d ~ N(0, (ε·σ_d)²)``), decode ``clean_latents + ξ_k`` via
``tokenizer.reconstruct()``, and evaluate configured reconstruction
metrics against the clean reconstruction using a shared
:class:`IntermediatesCache` (to reuse clean-side intermediates across
k draws). Earlier versions scaled jointly by a single σ_joint, which
under-perturbed low-σ_d dims and over-perturbed high-σ_d dims; per-dim
scaling makes ε mean "fraction of each dim's natural std" uniformly.

Output per D11: ``results.jsonl`` (streamed final scores, one row per
``(session, metric, ε)``), ``results.csv`` (wide, one row per
``(session, ε)``), and ``run_meta.json``.

Note on the design-doc phrasing "``@hydra.main(...)`` config
path/name": the existing :mod:`ntt.jobs.eval` is *not* itself a
``@hydra.main`` function — it is a Pydantic ``NTTBaseJob`` subclass
that is instantiated and run by the top-level ``ntt.scripts:main``
(which *is* the ``@hydra.main`` function).  This file mirrors that
actual shape: it defines :class:`NTTEvalRobustnessLatentJob` and is
launched via ``ntt job=eval_robustness_latent ...``.
"""

from __future__ import annotations

import functools
import gc
import hashlib
import json
import traceback
import types
import typing as T
import warnings
from dataclasses import dataclass
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
from ntt.core.components.types import BinnedSpikesInput, DataType, ReconstructionOutput
from ntt.core.data.datamodule import NTTDataModule, NTTDataModuleConfig
from ntt.core.model import load_tokenizer_from_checkpoint
from ntt.core.model.tokenizer import BaseTokenizer, TokenizerSpec, create_tokenizer
from ntt.evaluation.aggregation.comparability import add_robustness_report_fields
from ntt.evaluation.aggregation.comparison import (
    compute_normalized_comparisons,
    extract_normalized_scores,
)
from ntt.evaluation.artifact import aggregate_per_session_dicts
from ntt.evaluation.config import EvaluationConfig, validate_config
from ntt.evaluation.config.schema import AggregationStrategy
from ntt.evaluation.core.types import (
    EvalData,
    EvaluationReport,
    MetricResult,
)
from ntt.evaluation.metrics.base import BaseMetric
from ntt.evaluation.metrics.robustness_latent import evaluate_pair
from ntt.evaluation.metrics.robustness_latent_stats import LatentStats, LatentStatsCache
from ntt.evaluation.mlflow_logging import MLflowLoggingConfig, sanitize_metric_name_for_mlflow
from ntt.evaluation.neuron_selection import (
    NeuronSelection,
    all_neuron_data_config,
    require_direct_all_neuron_eval,
)
from ntt.evaluation.pipeline import EvaluationPipeline
from ntt.evaluation.plotting.io import save_fig_multi
from ntt.evaluation.plotting.robustness import (
    plot_latent_corruption_a,
    plot_latent_corruption_b,
)
from ntt.evaluation.utils.config_provenance import (
    composed_config_yaml,
    config_subtree_to_jsonable,
    config_to_jsonable,
    hydra_task_overrides,
    write_config_provenance_sidecar,
)
from ntt.evaluation.utils.metric_selection import (
    is_latent_robustness_reconstruction_metric,
)
from ntt.evaluation.utils.paths import resolve_artifact_path
from ntt.evaluation.utils.reconstruction_type import ReconstructionTypeRecorder
from ntt.evaluation.utils.validation_collation import iter_collated_validation_segments
from ntt.evaluation.utils.window_chunking import requires_fixed_input_window
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

_BOOTSTRAP_N_SAMPLES = 200
_MIN_BOOTSTRAP_WINDOWS = 5
_DEFAULT_M2_EVAL_WINDOW_SECONDS = 1.0
_M2_SESSION_GAP_TOLERANCE_SECONDS = 1e-12


# ── Windowed-mean-then-correlate aggregation helpers ───────────────
# Some reconstruction-side metrics (currently ``acg_correlation`` and
# ``ccg_correlation``) shouldn't be summarised as "per-window correlate,
# mean across windows": that double-aggregation is statistically less
# informative than first averaging the per-window correlogram estimates
# (each a noisy estimate of the same underlying ACG/CCG) and then
# correlating the windowed-mean estimates ONCE per (sid, ε, k). The
# helpers below are the per-metric aggregators used post-loop in
# ``run_evaluation``; the registry below maps metric names to the
# (intermediate keys, aggregator) needed to drive the buffering+post-loop
# pass.


def _per_row_pearson_then_mean(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson per row of (rows × len) arrays, then nanmean across rows.

    Used by the latent_robustness (M2) 'average first, then correlate' aggregation to
    summarise the windowed-mean ACG (rows = units) or CCG (rows = pairs)
    into a single scalar. Rows with any-NaN or zero-variance on either
    side are excluded from the mean.
    """
    n_rows = a.shape[0]
    rs = np.full(n_rows, np.nan, dtype=np.float64)
    for i in range(n_rows):
        ai, bi = a[i], b[i]
        if np.any(np.isnan(ai)) or np.any(np.isnan(bi)):
            continue
        if np.std(ai) > 0 and np.std(bi) > 0:
            rs[i] = float(np.corrcoef(ai, bi)[0, 1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return float(np.nanmean(rs))


def _aggregate_acg_mean_then_correlate(
    clean_windows: list[list[np.ndarray]],
    noisy_windows: list[list[np.ndarray]],
) -> float:
    """Mean ACGs across windows, Pearson per unit, mean across units.

    ``clean_windows`` / ``noisy_windows`` are lists of per-window outputs
    of the ACG intermediate (each itself a list of shape-(L,) arrays, one
    per unit). Per-window outputs are validated to share the same unit
    count and ACG length — fail loud rather than silently truncate.
    """
    if not clean_windows or not noisy_windows:
        return float("nan")
    n_units = len(clean_windows[0])
    if n_units == 0:
        return float("nan")
    acg_len = clean_windows[0][0].shape[0]
    for w in clean_windows + noisy_windows:
        if len(w) != n_units:
            raise ValueError(
                f"ACG per-window unit count mismatch: expected {n_units}, got {len(w)}"
            )
        for u_arr in w:
            if u_arr.shape[0] != acg_len:
                raise ValueError(
                    f"ACG per-unit length mismatch: expected {acg_len}, got {u_arr.shape[0]}"
                )
    clean_stack = np.stack([np.stack(w, axis=0) for w in clean_windows], axis=0)
    noisy_stack = np.stack([np.stack(w, axis=0) for w in noisy_windows], axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        clean_mean = np.nanmean(clean_stack, axis=0)
        noisy_mean = np.nanmean(noisy_stack, axis=0)
    return _per_row_pearson_then_mean(clean_mean, noisy_mean)


def _aggregate_ccg_mean_then_correlate(
    clean_windows: list[dict[tuple[int, int], np.ndarray]],
    noisy_windows: list[dict[tuple[int, int], np.ndarray]],
) -> float:
    """Mean CCGs across windows per pair, Pearson per pair, mean across pairs.

    Pairs are intersected across all windows (clean + noisy). In practice
    CCG pair selection is deterministic on ``n_units`` (seeded — see
    ``IntermediatesCache._select_pairs``), so all windows of one session
    share the same pair set; the intersection is a defensive no-op there.
    """
    if not clean_windows or not noisy_windows:
        return float("nan")
    common_pairs = set(clean_windows[0].keys())
    for w in clean_windows[1:] + noisy_windows:
        common_pairs &= set(w.keys())
    if not common_pairs:
        return float("nan")
    pairs = sorted(common_pairs)
    ccg_len = clean_windows[0][pairs[0]].shape[0]
    for w in clean_windows + noisy_windows:
        for p in pairs:
            if w[p].shape[0] != ccg_len:
                raise ValueError(
                    f"CCG per-pair length mismatch for pair {p}: "
                    f"expected {ccg_len}, got {w[p].shape[0]}"
                )
    clean_stack = np.stack(
        [np.stack([w[p] for p in pairs], axis=0) for w in clean_windows], axis=0
    )
    noisy_stack = np.stack(
        [np.stack([w[p] for p in pairs], axis=0) for w in noisy_windows], axis=0
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        clean_mean = np.nanmean(clean_stack, axis=0)
        noisy_mean = np.nanmean(noisy_stack, axis=0)
    return _per_row_pearson_then_mean(clean_mean, noisy_mean)


@dataclass(frozen=True)
class _MeanThenCorrelateRegistration:
    """Registry entry: which intermediate keys to buffer, and how to aggregate."""

    orig_key: str
    recon_key: str
    aggregator: T.Callable[[list[T.Any], list[T.Any]], float]


_MEAN_THEN_CORRELATE_REGISTRY: dict[str, _MeanThenCorrelateRegistration] = {
    "acg_correlation": _MeanThenCorrelateRegistration(
        orig_key="acg_original",
        recon_key="acg_reconstructed",
        aggregator=_aggregate_acg_mean_then_correlate,
    ),
    "ccg_correlation": _MeanThenCorrelateRegistration(
        orig_key="ccg_original",
        recon_key="ccg_reconstructed",
        aggregator=_aggregate_ccg_mean_then_correlate,
    ),
}


@dataclass
class _MeanThenCorrelateSpec:
    """Per-metric-instance spec built from the registry at run time.

    Holds the resolved ``IntermediateRequest`` objects (which encode the
    metric's params, e.g. correlogram bin size / max lag) so we can fetch
    the per-window arrays from the cache.
    """

    metric_name: str
    orig_req: T.Any  # IntermediateRequest
    recon_req: T.Any  # IntermediateRequest
    aggregator: T.Callable[[list[T.Any], list[T.Any]], float]


# Exact Poisson inverse CDF up to this per-bin rate; exp(-rate) underflows near 745.
# Above it a Cornish-Fisher quantile stands in (off by at most one count for
# u >= 2**-53, the resolution of float64 torch.rand; ~0.2% of u).
_M2_EXACT_POISSON_MAX_RATE = 500.0
M2_SAMPLING_KEY = "m2_sampling"


@functools.cache
def _poisson_cdf_at_exact_cap() -> tuple[float, ...]:
    """CDF of Poisson(_M2_EXACT_POISSON_MAX_RATE) at k = 0, 1, ..., K.

    Built with the same float64 recursion as the inverse-CDF walk in
    ``_sample_with_shared_uniforms``, so a lookup gives that walk's count exactly. It
    stops where the pmf underflows to zero, as the walk does; K is then the count.
    """
    rate = _M2_EXACT_POISSON_MAX_RATE
    pmf = float(torch.exp(torch.tensor(-rate, dtype=torch.float64)))
    cdf = [pmf]
    k = 0
    while pmf != 0.0:
        k += 1
        pmf = pmf * rate / k
        cdf.append(cdf[-1] + pmf)
    return tuple(cdf)


class _ExpectedValueTokenizer:
    """A spike-sampling tokenizer seen through its expected values.

    ``reconstruct`` returns what the tokenizer's own ``reconstruct`` samples from:
    spike probabilities (Bernoulli) or expected counts (Poisson). Everything else is
    the wrapped tokenizer's. latent_robustness (M2) then samples with shared draws.
    """

    def __init__(self, tokenizer: T.Any, kind: T.Literal["bernoulli", "poisson"]):
        self._tokenizer = tokenizer
        self._kind = kind

    def __getattr__(self, name: str) -> T.Any:
        return getattr(self._tokenizer, name)

    def reconstruct(self, latents: torch.Tensor) -> ReconstructionOutput:
        expected = self._tokenizer.decode(latents)
        if self._kind == "bernoulli":
            expected = expected.clamp(0.0, 1.0)
        else:
            expected = expected.clamp(min=0.0)
        bin_size_ms = getattr(self._tokenizer.spec, "bin_size_ms", None)
        return ReconstructionOutput(
            reconstruction=expected,
            data_type=DataType.SPIKES,
            bin_size_seconds=bin_size_ms / 1000.0 if bin_size_ms else None,
        )


class NTTEvalRobustnessLatentJob(
    NTTBaseJob,
    strict=False,
    frozen=False,
    extra="forbid",
    arbitrary_types_allowed=True,
):
    """latent_robustness (M2) evaluation job.

    Sweeps noise scales ``epsilons`` × ``n_samples_per_epsilon`` over
    each evaluated window, decoding ``clean_latents + ξ`` through the
    tokenizer and measuring reconstruction-side metrics against the
    clean reconstruction.  Reads ``cfg.eval_config.robustness_latent``
    (a :class:`RobustnessLatentConfig`); if ``None`` the job exits
    early.
    """

    KIND: T.Literal["eval_robustness_latent"] = "eval_robustness_latent"
    applies_eval_window: T.ClassVar[bool] = True

    # Same as NTTEvalJob: exactly one of ckpt_path / tokenizer_config
    ckpt_path: str | Path | None = None
    ckpt_weights_only: bool = False
    tokenizer_config: TokenizerSpecKind = None
    data_config: NTTDataModuleConfig

    # Evaluation config — accepts a raw dict so Hydra can pass the full
    # evaluation YAML (including the ``robustness_latent`` section).
    eval_config: dict[str, T.Any] | None = None

    device: str = "auto"
    output_path: str | Path | None = None
    split: T.Literal["val", "test"] = "val"
    evaluate_all_neurons: bool = True
    generate_plots: bool = True
    poisson_reference: str | Path | None = None

    # MLflow logging — defaults to disabled. Secondary pass: when enabled,
    # opens its own MLflow run tagged with the same checkpoint as the main
    # pass; checkpoint bytes are not re-uploaded.
    mlflow_logging: MLflowLoggingConfig = MLflowLoggingConfig(
        upload_checkpoint=False,
        run_name_suffix="__m2",
        metric_prefix="eval_m2",
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
            "Full composed Hydra config unavailable for latent_robustness (M2) run metadata; "
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

    # ── Context management (mirrors NTTEvalJob.__enter__) ──────────

    def _resolve_device(self) -> torch.device:
        """Resolve ``self.device`` to a concrete :class:`torch.device`."""
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
        """Set up device, tokenizer, datamodule, and parsed eval config.

        Mirrors :meth:`NTTEvalJob.__enter__` verbatim except that no
        MLflow-artifact download helper is used (deviation: we accept
        only local ``ckpt_path``; add parity later if needed).
        """
        self._device = self._resolve_device()
        log.info(f"Using device: {self._device}")

        if self.ckpt_path is not None:
            # Resolve ``mlflow-artifacts:/`` URIs to a local path before
            # handing to Lightning.
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
            log.info("Eval (robustness_latent) job: set deterministic_transforms=True")

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
        """Execute the latent_robustness (M2) sweep and write outputs."""
        assert self._tokenizer is not None
        assert self._device is not None
        assert self._eval_config is not None
        assert self._spec is not None

        rl_cfg = self._eval_config.robustness_latent
        if rl_cfg is None:
            log.info(
                "eval_config.robustness_latent is None — nothing to do. "
                "Set `evaluation.robustness_latent` to enable the latent_robustness (M2) sweep."
            )
            return locals()

        spec = self._spec
        bin_size_seconds = spec.bin_size_ms / 1000.0 if spec.bin_size_ms else None
        if bin_size_seconds is None:
            raise ValueError("robustness_latent job requires a tokenizer with fixed bin_size_ms.")
        tokenizer = self._tokenizer
        device = self._device
        eval_window_length_seconds = self._resolve_eval_window_length_seconds(spec)
        if self._eval_config.max_windows is not None or self._eval_config.max_bytes is not None:
            raise ValueError(
                "latent_robustness (M2) does not accept evaluation.max_windows "
                "or evaluation.max_bytes. "
                "Use robustness_latent.n_eval_windows and/or evaluation.max_eval_seconds."
            )
        if rl_cfg.max_decode_batch_size is not None and int(rl_cfg.max_decode_batch_size) < 1:
            raise ValueError("robustness_latent.max_decode_batch_size must be >= 1 when provided.")

        log.info(
            "Running plain validation pass to collect bounded clean windows "
            "(eval_window_length_seconds=%.3f)...",
            eval_window_length_seconds,
        )
        dataloader, _tokenizer_fn = self._build_eval_loader_and_tokenizer_fn(
            eval_window_length_seconds=eval_window_length_seconds
        )
        # Clean and noisy sides are both reconstructions: label them with the type
        # the tokenizer outputs declare, else spec.output_type (never input_type).
        _tokenizer_fn = ReconstructionTypeRecorder(_tokenizer_fn)
        seed = self.seed if self.seed is not None else 0
        eps_k_pairs = self._iter_eps_k_pairs(rl_cfg)
        eps_to_k_draws: dict[float, list[int]] = {}
        for eps_value, k_draw in eps_k_pairs:
            eps_to_k_draws.setdefault(float(eps_value), []).append(int(k_draw))

        selected_segments, session_order_seen, processed_window_ids_by_session = (
            self._collect_selected_window_segments(
                dataloader=dataloader,
                tokenizer_fn=_tokenizer_fn,
                rl_cfg=rl_cfg,
                spec=spec,
                device=device,
                bin_size_seconds=bin_size_seconds,
                eval_window_length_seconds=eval_window_length_seconds,
            )
        )
        if not selected_segments:
            raise ValueError(
                "latent_robustness (M2) did not collect any validation windows "
                "after applying caps."
            )
        reconstruction_data_type = _tokenizer_fn.declared or spec.output_type
        sampling_kind = self._m2_sampling_kind(tokenizer)
        m2_sampling = self._m2_sampling_label(tokenizer)
        if m2_sampling == "independent":
            log.warning(
                "KIND=%r samples spikes in reconstruct(), which latent_robustness (M2) "
                "does not take over: clean and noisy decodes use independent draws, so "
                "the M2 score includes sampling noise.",
                getattr(spec, "KIND", None),
            )
        decode_tokenizer = (
            tokenizer
            if sampling_kind is None
            else _ExpectedValueTokenizer(tokenizer, sampling_kind)
        )
        log.info(
            "Collected %d selected window(s) across %d session(s).",
            len(selected_segments),
            len(processed_window_ids_by_session),
        )

        stats = self._compute_sigma(selected_segments, seed=seed)
        log.info(
            "σ_joint = %.4f (σ_per_dim shape = %s)",
            stats.sigma_joint,
            tuple(stats.sigma_per_dim.shape),
        )

        full_pipeline = EvaluationPipeline(config=self._eval_config)
        # Keep the configured DISPLAY name alongside each instance: one metric
        # class may be configured several times with different parameters
        # (e.g. acg_correlation / acg_correlation_5ms / acg_correlation_10ms).
        # Every result key and bucket key below must therefore use the display
        # name — keying by ``metric.name`` collapses the variants onto a single
        # accumulator. Selection still filters on ``metric.name`` because the
        # exclusion lists in ``metric_selection`` are class-name based.
        recon_metrics: list[tuple[str, BaseMetric]] = [
            (display_name, metric)
            for display_name, metric in full_pipeline.metrics
            if is_latent_robustness_reconstruction_metric(metric.name, metric.category)
        ]
        session_aggregated_metrics = [
            (display_name, metric)
            for display_name, metric in recon_metrics
            if metric.aggregation_fallback_policy == "session_aggregated"
        ]
        per_window_metrics = [
            (display_name, metric)
            for display_name, metric in recon_metrics
            if metric.aggregation_fallback_policy == "per_window"
        ]
        log.info(
            "Evaluating %d latent_robustness (M2) reconstruction metric(s): "
            "%d session_aggregated, %d per_window.",
            len(recon_metrics),
            len(session_aggregated_metrics),
            len(per_window_metrics),
        )

        output_dir = self._get_output_dir()
        jsonl_path = output_dir / "results.jsonl" if output_dir is not None else None
        csv_rows: list[dict[str, T.Any]] = []
        metric_schemas: dict[str, dict[str, T.Any]] = {}
        skipped_counter: dict[str, int] = {}
        total_pairs = 0
        session_states: dict[tuple[str, str, float, int], dict[str, T.Any]] = {}
        per_window_scalars: dict[tuple[str, str, float, int], list[float]] = {}
        mse_match_miss_scalars: dict[tuple[str, str, float, int], list[float]] = {}
        bucket_errors: dict[tuple[str, str, float, int], str] = {}
        token_distance_records: list[dict[str, T.Any]] = []
        token_distance_skip_keys: set[str] = set()

        n_raster_examples_per_session = 3
        raster_selected_segment_ids: set[tuple[str, int]] = set()
        if self.generate_plots and rl_cfg.plot_examples:
            per_session_raster_count: dict[str, int] = {}
            for segment in selected_segments:
                sid = str(segment.session_id)
                if per_session_raster_count.get(sid, 0) >= n_raster_examples_per_session:
                    continue
                raster_selected_segment_ids.add((sid, int(segment.segment_index)))
                per_session_raster_count[sid] = per_session_raster_count.get(sid, 0) + 1
        captured_rasters: dict[str, dict[int, dict[str, T.Any]]] = {}

        jsonl_fh = jsonl_path.open("w") if jsonl_path is not None else None
        report_config = self._eval_config.model_dump() if self._eval_config is not None else {}
        report_config[M2_SAMPLING_KEY] = m2_sampling
        report_original_data_type = None
        report_reconstructed_data_type = None
        per_session_reports: dict[str, EvaluationReport] = {}
        aggregated_report: EvaluationReport | None = None
        poisson_metrics: dict[str, float] = {}

        def _mark_metric_skipped(metric_name: str) -> None:
            skipped_counter[metric_name] = skipped_counter.get(metric_name, 0) + 1

        def _update_raster_metrics(
            session_id: str, eps: float, k_draw: int, metric: str, value: float
        ):
            if k_draw != 0 or session_id not in captured_rasters:
                return
            for payload in captured_rasters[session_id].values():
                payload.setdefault("metric_per_eps", {}).setdefault(float(eps), {})[metric] = value

        def _score_std(values: list[float]) -> float:
            if not values:
                return float("nan")
            if len(values) == 1:
                return 0.0
            return float(np.std(np.asarray(values, dtype=float), ddof=1))

        def _write_final_jsonl_row(
            *,
            session_id: str,
            epsilon: float,
            metric_name: str,
            sample_rows: list[dict[str, T.Any]],
        ) -> None:
            if jsonl_fh is None:
                return

            row: dict[str, T.Any] = {
                "session_id": session_id,
                "epsilon": float(epsilon),
                "metric": metric_name,
            }
            ok_values: list[float] = []
            reasons: list[str] = []
            for sample_row in sample_rows:
                if sample_row.get("status") == "ok":
                    ok_values.append(float(sample_row["raw_value"]))
                else:
                    reasons.append(str(sample_row.get("reason", "unknown failure")))

            if ok_values:
                row["status"] = "ok"
                row["raw_value"] = float(np.mean(np.asarray(ok_values, dtype=float)))
                row["metadata"] = {
                    "n_samples_requested": len(sample_rows),
                    "n_samples_ok": len(ok_values),
                    "n_samples_skipped": len(sample_rows) - len(ok_values),
                    "score_std": _score_std(ok_values),
                }
            else:
                row["status"] = "skipped"
                if reasons:
                    unique_reasons = sorted(set(reasons))
                    if len(unique_reasons) == 1:
                        row["reason"] = unique_reasons[0]
                    else:
                        joined = "; ".join(unique_reasons)
                        row["reason"] = (
                            f"all random samples failed ({len(sample_rows)} draws): {joined}"
                        )
                else:
                    row["reason"] = "no successful random samples"
            jsonl_fh.write(json.dumps(row) + "\n")

        def _finalize_session(session_id: str | None) -> None:
            if session_id is None:
                return

            for eps_value, k_draws in eps_to_k_draws.items():
                sample_rows_by_metric: dict[str, list[dict[str, T.Any]]] = {}

                for k_draw in k_draws:
                    wide_row: dict[str, T.Any] = {
                        "session_id": session_id,
                        "epsilon": float(eps_value),
                        "k": int(k_draw),
                    }

                    for display_name, metric in session_aggregated_metrics:
                        bucket_key = (display_name, session_id, float(eps_value), int(k_draw))
                        row: dict[str, T.Any] = {
                            "session_id": session_id,
                            "epsilon": float(eps_value),
                            "k": int(k_draw),
                            "metric": display_name,
                        }
                        if bucket_key in bucket_errors:
                            row["status"] = "skipped"
                            row["reason"] = bucket_errors[bucket_key]
                            _mark_metric_skipped(display_name)
                        else:
                            state = session_states.pop(bucket_key, None)
                            if state is None or state.get("n_windows_seen", 0) == 0:
                                row["status"] = "skipped"
                                row["reason"] = "no windows accumulated"
                                _mark_metric_skipped(display_name)
                            else:
                                try:
                                    result = metric._finalize_session_state(state)
                                except Exception as exc:
                                    row["status"] = "skipped"
                                    row["reason"] = str(exc)
                                    _mark_metric_skipped(display_name)
                                else:
                                    self._capture_metric_schema(
                                        metric_schemas, display_name, result
                                    )
                                    scalar = self._metric_result_scalar(result)
                                    if scalar is None:
                                        row["status"] = "skipped"
                                        row["reason"] = (
                                            "finalized metric did not yield a finite scalar"
                                        )
                                        _mark_metric_skipped(display_name)
                                    else:
                                        row["status"] = "ok"
                                        row["raw_value"] = float(scalar)
                                        row["metadata"] = dict(result.metadata)
                                        wide_row[display_name] = float(scalar)
                                        _update_raster_metrics(
                                            session_id,
                                            eps_value,
                                            k_draw,
                                            display_name,
                                            float(scalar),
                                        )
                        sample_rows_by_metric.setdefault(display_name, []).append(row)

                    for display_name, _metric in per_window_metrics:
                        bucket_key = (display_name, session_id, float(eps_value), int(k_draw))
                        row = {
                            "session_id": session_id,
                            "epsilon": float(eps_value),
                            "k": int(k_draw),
                            "metric": display_name,
                        }
                        if bucket_key in bucket_errors:
                            row["status"] = "skipped"
                            row["reason"] = bucket_errors[bucket_key]
                            _mark_metric_skipped(display_name)
                        else:
                            scalars = per_window_scalars.pop(bucket_key, [])
                            if not scalars:
                                row["status"] = "skipped"
                                row["reason"] = "no per-window scalars"
                                _mark_metric_skipped(display_name)
                            else:
                                mean_value = float(np.mean(np.asarray(scalars, dtype=float)))
                                if len(scalars) >= _MIN_BOOTSTRAP_WINDOWS:
                                    bootstrap_variance = self._bootstrap_variance(
                                        scalars,
                                        seed=seed,
                                    )
                                    n_bootstrap_samples = _BOOTSTRAP_N_SAMPLES
                                else:
                                    bootstrap_variance = float("nan")
                                    n_bootstrap_samples = 0
                                row["status"] = "ok"
                                row["raw_value"] = mean_value
                                row["metadata"] = {
                                    "n_windows": len(scalars),
                                    "bootstrap_variance": bootstrap_variance,
                                    "n_bootstrap_samples": n_bootstrap_samples,
                                }
                                wide_row[display_name] = mean_value
                                _update_raster_metrics(
                                    session_id,
                                    eps_value,
                                    k_draw,
                                    display_name,
                                    mean_value,
                                )
                        sample_rows_by_metric.setdefault(display_name, []).append(row)

                    mse_key = ("mse_match_miss", session_id, float(eps_value), int(k_draw))
                    mse_row = {
                        "session_id": session_id,
                        "epsilon": float(eps_value),
                        "k": int(k_draw),
                        "metric": "mse_match_miss",
                    }
                    if mse_key in bucket_errors:
                        mse_row["status"] = "skipped"
                        mse_row["reason"] = bucket_errors[mse_key]
                        _mark_metric_skipped("mse_match_miss")
                    else:
                        mse_scalars = mse_match_miss_scalars.pop(mse_key, [])
                        if not mse_scalars:
                            mse_row["status"] = "skipped"
                            mse_row["reason"] = "no per-window scalars"
                            _mark_metric_skipped("mse_match_miss")
                        else:
                            mse_mean = float(np.mean(np.asarray(mse_scalars, dtype=float)))
                            if len(mse_scalars) >= _MIN_BOOTSTRAP_WINDOWS:
                                mse_bootstrap_variance = self._bootstrap_variance(
                                    mse_scalars,
                                    seed=seed,
                                )
                                mse_n_bootstrap_samples = _BOOTSTRAP_N_SAMPLES
                            else:
                                mse_bootstrap_variance = float("nan")
                                mse_n_bootstrap_samples = 0
                            mse_row["status"] = "ok"
                            mse_row["raw_value"] = mse_mean
                            mse_row["metadata"] = {
                                "n_windows": len(mse_scalars),
                                "bootstrap_variance": mse_bootstrap_variance,
                                "n_bootstrap_samples": mse_n_bootstrap_samples,
                            }
                            wide_row["mse_match_miss"] = mse_mean
                            _update_raster_metrics(
                                session_id,
                                eps_value,
                                k_draw,
                                "mse_match_miss",
                                mse_mean,
                            )
                    sample_rows_by_metric.setdefault("mse_match_miss", []).append(mse_row)

                    csv_rows.append(wide_row)

                for display_name, _metric in session_aggregated_metrics:
                    _write_final_jsonl_row(
                        session_id=session_id,
                        epsilon=float(eps_value),
                        metric_name=display_name,
                        sample_rows=sample_rows_by_metric.get(display_name, []),
                    )
                for display_name, _metric in per_window_metrics:
                    _write_final_jsonl_row(
                        session_id=session_id,
                        epsilon=float(eps_value),
                        metric_name=display_name,
                        sample_rows=sample_rows_by_metric.get(display_name, []),
                    )
                _write_final_jsonl_row(
                    session_id=session_id,
                    epsilon=float(eps_value),
                    metric_name="mse_match_miss",
                    sample_rows=sample_rows_by_metric.get("mse_match_miss", []),
                )
            if jsonl_fh is not None:
                jsonl_fh.flush()

        try:
            latent_type_name = getattr(getattr(spec, "latent_type", None), "name", None)
            processed_window_count = 0
            total_window_evaluations = len(selected_segments) * len(eps_k_pairs)
            current_sid: str | None = None
            current_session_has_windows = False
            current_window_time_bins: int | None = None
            vocab_size: int | None = None
            if rl_cfg.corruption_target == "codes":
                if not hasattr(tokenizer, "quantize"):
                    raise NotImplementedError(
                        "corruption_target='codes' requires tokenizer.quantize(latents) -> codes."
                    )
                vocab_size = int(tokenizer.get_vocab_size())

            for segment in selected_segments:
                sid = str(segment.session_id)
                if current_sid is None:
                    current_sid = sid
                    current_session_has_windows = False
                    current_window_time_bins = None
                elif sid != current_sid:
                    if current_session_has_windows:
                        _finalize_session(current_sid)
                        gc.collect()
                    current_sid = sid
                    current_session_has_windows = False
                    current_window_time_bins = None

                if segment.latents is None or segment.reconstructed is None:
                    log.warning("Skipping selected window without latents/reconstruction.")
                    continue

                clean_latents_t = self._segment_latents_tensor(segment).to(device)
                clean_recon_t = segment.reconstructed
                shared_uniforms: torch.Tensor | None = None
                if sampling_kind is not None:
                    # The collector stored expected values; sample them here from the
                    # draws this window's noisy reconstructions share.
                    if segment.segment_absolute_start is None:
                        raise ValueError("Shared M2 draws require segment_absolute_start.")
                    shared_uniforms = self._m2_shared_uniforms(
                        tuple(clean_recon_t.shape),
                        seed=seed,
                        session_id=sid,
                        window_start=float(segment.segment_absolute_start),
                    )
                    clean_recon_t = self._sample_with_shared_uniforms(
                        clean_recon_t, shared_uniforms, kind=sampling_kind
                    )
                clean_time_bins = int(clean_recon_t.shape[-1])
                if current_window_time_bins is None:
                    current_window_time_bins = clean_time_bins
                else:
                    assert clean_time_bins == current_window_time_bins, (
                        "latent_robustness (M2) selected windows must stay "
                        "fixed-size within a session: "
                        f"expected {current_window_time_bins} bins, got {clean_time_bins}."
                    )
                clean_recon_data = EvalData.from_tensor(
                    tensor=clean_recon_t,
                    data_type=reconstruction_data_type,
                    bin_size_seconds=bin_size_seconds,
                    duration=clean_time_bins * bin_size_seconds,
                )
                sampling_rate = self._extract_sampling_rate(segment)
                if report_original_data_type is None:
                    report_original_data_type = clean_recon_data.data_type
                    report_reconstructed_data_type = clean_recon_data.data_type
                else:
                    assert report_original_data_type == clean_recon_data.data_type
                    assert report_reconstructed_data_type == clean_recon_data.data_type

                clean_cache = self._build_clean_cache(
                    clean_original=clean_recon_data,
                    clean_reconstructed=clean_recon_data,
                    segment=segment,
                    sampling_rate=sampling_rate,
                    token_bin_size_ms=spec.token_bin_size_ms,
                )
                # Clean side cut to a short noisy decode's length, built once per window
                # and length so the clean-side intermediates are reused across draws.
                # To avoid rebuilding the clean metric cache for every noise draw.
                # Draws producing the same length reuse the same entry; different
                # lengths need different entries. If every draw produces the same
                # length, the dictionary contains only one entry.
                # Memory is per window, with at most 32 shortened lengths under the
                # tolerance below. Separate CPU copies cost at most
                # n_lengths * n_units * clean_time_bins * bytes_per_element:
                # 32 * 100 * 1000 * 4 = 12.2 MiB for float32, plus cache objects and
                # any computed metric intermediates (metric-dependent extra memory).
                # CPU-backed slices share the clean array's storage instead; the two
                # cache fields also reference the same shortened EvalData object.
                clean_cache_by_n_time_bins: dict[int, tuple[EvalData, T.Any]] = {}
                capture_this = (sid, int(segment.segment_index)) in raster_selected_segment_ids
                noisy_per_eps_capture: dict[float, list[np.ndarray]] = {}
                clean_codes_t: torch.Tensor | None = None
                if rl_cfg.corruption_target == "codes":
                    clean_codes_t = T.cast(T.Any, tokenizer).quantize(clean_latents_t)
                    if not isinstance(clean_codes_t, torch.Tensor):
                        raise TypeError(
                            "tokenizer.quantize(latents) must return a torch.Tensor for "
                            "codes corruption."
                        )
                    if clean_codes_t.is_floating_point():
                        raise TypeError(
                            "tokenizer.quantize(latents) returned floating codes; "
                            "latent_robustness (M2) codes corruption "
                            "expects discrete index-like codes."
                        )
                elif latent_type_name == "DISCRETE":
                    skip_key = f"token_distance:{type(tokenizer).__name__}"
                    try:
                        clean_codes_t = self._token_distance_tokens_from_latents(
                            tokenizer,
                            clean_latents_t,
                        )
                    except Exception as exc:
                        if skip_key not in token_distance_skip_keys:
                            log.info(
                                "Skipping token-distance metadata for %s: %s",
                                type(tokenizer).__name__,
                                exc,
                            )
                            token_distance_skip_keys.add(skip_key)

                window_index = int(segment.segment_index)
                current_session_has_windows = True

                for eps_value, k_draws in eps_to_k_draws.items():
                    expected_replica_count = int(rl_cfg.n_samples_per_epsilon)
                    assert len(k_draws) == expected_replica_count, (
                        "Per-epsilon replica batch must match n_samples_per_epsilon: "
                        f"expected {expected_replica_count}, got {len(k_draws)}."
                    )

                    corrupted_code_replicas: list[torch.Tensor] | None = None
                    code_swap_masks: list[np.ndarray] | None = None
                    if rl_cfg.corruption_target == "latents":
                        corrupted_latent_replicas = [
                            clean_latents_t
                            + self._draw_noise(
                                shape=tuple(clean_latents_t.shape),
                                eps=float(eps_value),
                                sigma_per_dim=np.asarray(stats.sigma_per_dim),
                                distribution=rl_cfg.noise_distribution,
                                seed=seed,
                                k=int(k_draw),
                                session_id=sid,
                                window_index=window_index,
                                device=device,
                                dtype=clean_latents_t.dtype,
                            )
                            for k_draw in k_draws
                        ]
                        corrupted_replica_batch, replica_lengths = self._stack_window_tensors(
                            corrupted_latent_replicas,
                            name="corrupted latent replica",
                        )
                        expected_replica_length = int(clean_latents_t.shape[0])
                        assert (
                            replica_lengths == [expected_replica_length] * expected_replica_count
                        ), (
                            "Replica stack unexpectedly changed the per-window leading dimension: "
                            f"expected {[expected_replica_length] * expected_replica_count}, "
                            f"got {replica_lengths}."
                        )
                        decoded_replica_batch = self._decode_reconstruction_batch(
                            tokenizer=decode_tokenizer,
                            batch_tensor=corrupted_replica_batch,
                            corruption_target="latents",
                            segment=segment,
                            clean_recon_t=clean_recon_t,
                            sampling_rate=sampling_rate,
                            max_decode_batch_size=rl_cfg.max_decode_batch_size,
                        )
                        decoded_replicas = self._split_window_tensor(
                            decoded_replica_batch,
                            replica_lengths,
                        )

                        if isinstance(clean_codes_t, torch.Tensor):
                            skip_key = f"token_distance:{type(tokenizer).__name__}"
                            try:
                                corrupted_codes_batch = self._token_distance_tokens_from_latents(
                                    tokenizer,
                                    corrupted_replica_batch,
                                )
                                # Split inside the try: for grouped/structured codes
                                # (e.g. VQPT grouped-residual-FSQ) the re-quantized
                                # batch's leading dim is the group count (2), not the
                                # replica count, so this split raises. token-distance
                                # is measurement-only, so the failure must be skipped
                                # here (as the except branch intends) rather than
                                # crash the whole latent_robustness (M2) job.
                                corrupted_code_replicas = self._split_window_tensor(
                                    corrupted_codes_batch,
                                    replica_lengths,
                                )
                            except Exception as exc:
                                if skip_key not in token_distance_skip_keys:
                                    log.info(
                                        "Skipping token-distance metadata for %s: %s",
                                        type(tokenizer).__name__,
                                        exc,
                                    )
                                    token_distance_skip_keys.add(skip_key)
                    elif rl_cfg.corruption_target == "codes":
                        assert clean_codes_t is not None
                        assert vocab_size is not None
                        clean_codes_np = clean_codes_t.detach().cpu().numpy()
                        corrupted_code_replicas_np: list[np.ndarray] = []
                        code_swap_masks = []
                        for k_draw in k_draws:
                            corrupted_codes_window, swap_mask = self._draw_corruption_for_codes(
                                clean_codes_np,
                                eps=float(eps_value),
                                vocab_size=vocab_size,
                                seed=seed,
                                k=int(k_draw),
                                session_id=sid,
                                window_index=window_index,
                            )
                            corrupted_code_replicas_np.append(corrupted_codes_window)
                            code_swap_masks.append(swap_mask)
                        corrupted_code_replicas = [
                            torch.as_tensor(
                                corrupted_codes_window,
                                device=clean_codes_t.device,
                                dtype=clean_codes_t.dtype,
                            )
                            for corrupted_codes_window in corrupted_code_replicas_np
                        ]
                        corrupted_replica_batch, replica_lengths = self._stack_window_tensors(
                            corrupted_code_replicas,
                            name="corrupted code replica",
                        )
                        expected_replica_length = int(clean_codes_t.shape[0])
                        assert (
                            replica_lengths == [expected_replica_length] * expected_replica_count
                        ), (
                            "Replica stack unexpectedly changed the per-window leading dimension: "
                            f"expected {[expected_replica_length] * expected_replica_count}, "
                            f"got {replica_lengths}."
                        )
                        decoded_replica_batch = self._decode_reconstruction_batch(
                            tokenizer=decode_tokenizer,
                            batch_tensor=corrupted_replica_batch,
                            corruption_target="codes",
                            segment=segment,
                            clean_recon_t=clean_recon_t,
                            sampling_rate=sampling_rate,
                            max_decode_batch_size=rl_cfg.max_decode_batch_size,
                        )
                        decoded_replicas = self._split_window_tensor(
                            decoded_replica_batch,
                            replica_lengths,
                        )
                    else:
                        raise ValueError(
                            "Unsupported robustness_latent.corruption_target="
                            f"{rl_cfg.corruption_target!r}."
                        )

                    assert len(decoded_replicas) == expected_replica_count, (
                        "Per-epsilon decode must return one reconstruction batch "
                        "per requested replica: "
                        f"expected {expected_replica_count}, got {len(decoded_replicas)}."
                    )
                    if corrupted_code_replicas is not None:
                        assert len(corrupted_code_replicas) == expected_replica_count, (
                            "Per-epsilon corruption batch must keep exactly one code tensor per "
                            f"requested replica; expected {expected_replica_count}, got "
                            f"{len(corrupted_code_replicas)}."
                        )

                    for replica_offset, (k_draw, decoded_replica) in enumerate(
                        zip(k_draws, decoded_replicas, strict=True)
                    ):
                        processed_window_count += 1
                        log.info(
                            "[%d/%d] %s (window_idx=%d)",
                            processed_window_count,
                            total_window_evaluations,
                            sid,
                            window_index,
                        )

                        noisy_recon_t_2d = self._to_units_time(decoded_replica)
                        if shared_uniforms is not None:
                            assert sampling_kind is not None
                            noisy_recon_t_2d = noisy_recon_t_2d[:, : shared_uniforms.shape[-1]]
                            noisy_recon_t_2d = self._sample_with_shared_uniforms(
                                noisy_recon_t_2d,
                                shared_uniforms[:, : noisy_recon_t_2d.shape[-1]],
                                kind=sampling_kind,
                            )
                        n_noisy_time_bins = int(noisy_recon_t_2d.shape[-1])
                        # Some tokenizers (notably Cosmos with db4 wavelet
                        # patcher) produce a noisy decode that is a few bins
                        # SHORTER than the clean reconstruction. The clean
                        # path runs the tokenizer's own `forward()` whose
                        # output is trimmed to ``window_length_bins``; the
                        # noisy path runs ``reconstruct(noisy_latents)`` on
                        # latents whose token-timestamp strip already
                        # discarded the wavelet-overhang tokens, so the
                        # inverse DWT yields ``2*W_lat - (filter_len - 2)``
                        # bins for one db4 level. Tolerate a small wavelet
                        # boundary shortage by truncating clean down to the
                        # shared length; fail loud if the shortage is large
                        # (likely a real alignment bug).
                        _COSMOS_WAVELET_TOLERANCE_BINS = 32
                        compare_clean_recon_data = clean_recon_data
                        compare_clean_cache = clean_cache
                        compare_time_bins = clean_time_bins
                        if n_noisy_time_bins < clean_time_bins:
                            shortage = clean_time_bins - n_noisy_time_bins
                            if shortage > _COSMOS_WAVELET_TOLERANCE_BINS:
                                raise AssertionError(
                                    f"noisy reconstruction shorter than clean: "
                                    f"{n_noisy_time_bins} < {clean_time_bins} "
                                    f"(shortage {shortage} > tolerance "
                                    f"{_COSMOS_WAVELET_TOLERANCE_BINS}); "
                                    f"KIND={getattr(spec, 'KIND', '<unknown>')!r}"
                                )
                            log.warning(
                                "latent_robustness (M2) noisy reconstruction "
                                "shorter than clean by %d bins "
                                "(noisy=%d, clean=%d) — within wavelet-boundary tolerance. "
                                "Truncating clean to %d bins for the metric pair. "
                                "KIND=%r",
                                shortage,
                                n_noisy_time_bins,
                                clean_time_bins,
                                n_noisy_time_bins,
                                getattr(spec, "KIND", "<unknown>"),
                            )
                            compare_time_bins = n_noisy_time_bins
                            # If we haven’t already prepared clean data for this length,
                            # prepare and store it now.
                            if n_noisy_time_bins not in clean_cache_by_n_time_bins:
                                truncated_clean = EvalData.from_tensor(
                                    tensor=clean_recon_t[:, :n_noisy_time_bins],
                                    data_type=reconstruction_data_type,
                                    bin_size_seconds=bin_size_seconds,
                                    duration=n_noisy_time_bins * bin_size_seconds,
                                )
                                # Truncate both sides: metrics compare the cache's
                                # ``original`` against the noisy ``reconstructed``.
                                clean_cache_by_n_time_bins[n_noisy_time_bins] = (
                                    truncated_clean,
                                    self._build_clean_cache(
                                        clean_original=truncated_clean,
                                        clean_reconstructed=truncated_clean,
                                        segment=segment,
                                        sampling_rate=sampling_rate,
                                        token_bin_size_ms=spec.token_bin_size_ms,
                                    ),
                                )
                            compare_clean_recon_data, compare_clean_cache = (
                                clean_cache_by_n_time_bins[n_noisy_time_bins]
                            )
                        elif n_noisy_time_bins > clean_time_bins:
                            noisy_recon_t_2d = noisy_recon_t_2d[:, :clean_time_bins]

                        noisy_data = EvalData.from_tensor(
                            tensor=noisy_recon_t_2d,
                            data_type=reconstruction_data_type,
                            bin_size_seconds=bin_size_seconds,
                            duration=noisy_recon_t_2d.shape[-1] * bin_size_seconds,
                        )

                        if per_window_metrics:
                            pair_results, pair_cache = evaluate_pair(
                                clean=compare_clean_recon_data,
                                noisy=noisy_data,
                                metrics=per_window_metrics,
                                config=self._eval_config,
                                shared_cache=compare_clean_cache,
                                return_cache=True,
                            )
                        else:
                            pair_results = {}
                            pair_cache = compare_clean_cache.replace_reconstructed(noisy_data)

                        for display_name, metric in session_aggregated_metrics:
                            bucket_key = (display_name, sid, float(eps_value), int(k_draw))
                            if bucket_key in bucket_errors:
                                continue
                            try:
                                metric._accumulate_window(
                                    session_states.setdefault(bucket_key, {}), pair_cache
                                )
                            except Exception as exc:
                                bucket_errors.setdefault(bucket_key, str(exc))

                        for display_name, _metric in per_window_metrics:
                            bucket_key = (display_name, sid, float(eps_value), int(k_draw))
                            if bucket_key in bucket_errors:
                                continue
                            entry = pair_results.get(display_name)
                            if entry is None:
                                bucket_errors.setdefault(bucket_key, "missing per-window result")
                                continue
                            if entry["status"] != "ok":
                                bucket_errors.setdefault(
                                    bucket_key,
                                    str(entry.get("reason", "unknown per-window metric failure")),
                                )
                                continue
                            metric_result = entry["result"]
                            self._capture_metric_schema(
                                metric_schemas, display_name, metric_result
                            )
                            scalar = self._metric_result_scalar(metric_result)
                            if scalar is None:
                                bucket_errors.setdefault(
                                    bucket_key,
                                    "per-window metric did not yield a finite scalar",
                                )
                                continue
                            per_window_scalars.setdefault(bucket_key, []).append(float(scalar))

                        mse_bucket_key = ("mse_match_miss", sid, float(eps_value), int(k_draw))
                        mse_match_miss = self._compute_mse_match_miss(
                            clean_recon_t, noisy_recon_t_2d
                        )
                        mse_match_miss_scalars.setdefault(mse_bucket_key, []).append(
                            float(mse_match_miss)
                        )

                        if rl_cfg.corruption_target == "latents" and isinstance(
                            clean_codes_t, torch.Tensor
                        ):
                            if corrupted_code_replicas is not None:
                                corrupted_codes = corrupted_code_replicas[replica_offset]
                                fraction_changed = float(
                                    (corrupted_codes != clean_codes_t).float().mean().item()
                                )
                                token_distance_records.append(
                                    {
                                        "session_id": sid,
                                        "window_index": window_index,
                                        "eps": float(eps_value),
                                        "k": int(k_draw),
                                        "fraction_changed": fraction_changed,
                                        "n_codes": int(clean_codes_t.numel()),
                                    }
                                )
                        elif rl_cfg.corruption_target == "codes":
                            assert code_swap_masks is not None
                            swap_mask = code_swap_masks[replica_offset]
                            token_distance_records.append(
                                {
                                    "session_id": sid,
                                    "window_index": window_index,
                                    "eps": float(eps_value),
                                    "k": int(k_draw),
                                    "fraction_changed": float(np.mean(swap_mask)),
                                    "n_codes": int(swap_mask.size),
                                }
                            )

                        total_pairs += 1
                        if capture_this and k_draw == 0:
                            noisy_per_eps_capture.setdefault(float(eps_value), []).append(
                                _units_time_to_TU(noisy_recon_t_2d)
                            )

                    del corrupted_replica_batch
                    del decoded_replica_batch
                    del decoded_replicas
                    if rl_cfg.corruption_target == "latents":
                        del corrupted_latent_replicas
                    else:
                        del clean_codes_np
                        del corrupted_code_replicas_np
                    if corrupted_code_replicas is not None:
                        del corrupted_code_replicas
                    if code_swap_masks is not None:
                        del code_swap_masks

                if capture_this:
                    captured_rasters.setdefault(sid, {})[window_index] = {
                        "clean": _units_time_to_TU(clean_recon_t),
                        "noisy_per_eps": noisy_per_eps_capture,
                        "metric_per_eps": {},
                    }
                segment.strip_original()
                del clean_latents_t
                del clean_recon_t
                del clean_recon_data
                del clean_cache
                if clean_codes_t is not None:
                    del clean_codes_t

            if current_session_has_windows:
                _finalize_session(current_sid)
                gc.collect()
        finally:
            if jsonl_fh is not None:
                jsonl_fh.close()

        # ── Step 6: CSV + run_meta.json ────────────────────────────
        if output_dir is not None:
            self._write_csv(csv_rows, output_dir / "results.csv")
            self._write_per_session_csv(csv_rows, output_dir / "results_per_session.csv")
            per_session_reports, aggregated_report = self._build_reports_from_m2_rows(
                csv_rows,
                metric_schemas,
                report_config=report_config,
                original_data_type=report_original_data_type,
                reconstructed_data_type=report_reconstructed_data_type,
            )
            self._write_per_session_report_artifacts(
                per_session_reports, output_dir / "per_session"
            )
            poisson_metrics: dict[str, float] = {}
            if self.poisson_reference is not None:
                _poisson_payload, poisson_metrics = self._write_poisson_comparison_artifact(
                    output_dir=output_dir,
                    per_session_reports=per_session_reports,
                    aggregated_report=aggregated_report,
                    metric_schemas=metric_schemas,
                )
            run_meta = self._write_run_meta(
                output_dir / "run_meta.json",
                stats=stats,
                skipped_counter=skipped_counter,
                total_pairs=total_pairs,
                rl_cfg=rl_cfg,
                spec=self._spec,
                tokenizer=tokenizer,
                checkpoint=self.ckpt_path,
                token_distance_records=token_distance_records,
                eval_window_length_seconds=eval_window_length_seconds,
                eval_config=self._eval_config,
                data_config=self._provenance_data_config(),
                hydra_overrides=self._hydra_overrides,
                composed_config_yaml=self._provenance_composed_config_yaml(),
                evaluate_all_neurons=self.evaluate_all_neurons,
                neuron_selection=self._neuron_selection,
                eval_window=self._eval_window_record(),
            )
            # Sessions, cross-session aggregation (always a plain mean, see
            # _build_reports_from_m2_rows) and run settings go into the report itself
            # (also its MLflow copy), for the aggregation's comparability check.
            add_robustness_report_fields(
                aggregated_report,
                run_meta,
                self._eval_config,
                split=self.split,
                session_ids=list(per_session_reports),
                aggregation_strategy=AggregationStrategy.MEAN.value,
            )
            self._write_eval_report_json(aggregated_report, output_dir / "eval_report.json")
            log.info(f"Results written to {output_dir}")

            # ── Eval-time figures ───────────────────────────────
            if self.generate_plots:
                self._emit_figures(
                    output_dir=output_dir,
                    rl_cfg=rl_cfg,
                    bin_size_seconds=bin_size_seconds,
                    captured_rasters=captured_rasters,
                )

            # ── MLflow logging ──────────────────────────────────
            if self.mlflow_logging.enabled:
                self._log_to_mlflow(
                    output_dir=output_dir,
                    rl_cfg=rl_cfg,
                    aggregated_report=aggregated_report,
                    per_session_reports=per_session_reports,
                    extra_metrics=poisson_metrics,
                )

        return locals()

    @staticmethod
    def _iter_eps_k_pairs(rl_cfg: T.Any) -> list[tuple[float, int]]:
        return [
            (float(eps), int(k))
            for eps in rl_cfg.epsilons
            for k in range(rl_cfg.n_samples_per_epsilon)
        ]

    @staticmethod
    def _metric_result_scalar(result: MetricResult) -> float | None:
        raw = result.raw_value
        if hasattr(raw, "__len__"):
            arr = np.asarray(raw, dtype=float)
            if arr.size == 0:
                return None
            value = float(np.nanmean(arr))
        else:
            try:
                value = float(raw)
            except (TypeError, ValueError):
                return None
        if not np.isfinite(value):
            return None
        return value

    @staticmethod
    def _bootstrap_variance(
        per_window_scores: list[float],
        *,
        n_bootstrap_samples: int = _BOOTSTRAP_N_SAMPLES,
        seed: int = 0,
    ) -> float:
        arr = np.asarray(per_window_scores, dtype=float)
        if arr.size == 0:
            return float("nan")
        rng = np.random.default_rng(seed)
        means = rng.choice(arr, size=(n_bootstrap_samples, arr.size), replace=True).mean(axis=1)
        return float(means.var(ddof=1)) if means.size > 1 else 0.0

    @staticmethod
    def _window_segment_end_seconds(segment: T.Any, bin_size_seconds: float) -> float:
        if segment.segment_absolute_start is None:
            raise ValueError("Segment is missing segment_absolute_start.")
        if segment.original is not None:
            time_bins = int(segment.original.shape[-1])
        elif segment.original_shape is not None:
            time_bins = int(segment.original_shape[-1])
        else:
            raise ValueError("Segment is missing original/original_shape; cannot infer duration.")
        return float(segment.segment_absolute_start) + float(time_bins) * float(bin_size_seconds)

    def _resolve_eval_window_length_seconds(self, spec: TokenizerSpec) -> float:
        """Resolve the latent_robustness (M2) evaluation window length (seconds).

        The eval/analysis window is the single source of truth: latent_robustness (M2) uses the
        authoritative window resolved from ``eval_config.eval_window_seconds``
        (see :meth:`NTTBaseJob._resolve_eval_window_seconds`), defaulting to
        1.0 s. ``__enter__`` already fed this window to the datamodule and
        validated — via ``apply_eval_window`` /
        ``assert_tokenizer_accepts_eval_window`` — that ``spec`` can be
        evaluated at it (raising ``IncompatibleEvalWindowError`` otherwise).
        Fixed-input tokenizers whose native window is shorter are handled by
        ``chunked_tokenizer_fn`` in the collator; the window is never
        forced to a hardcoded length.

        ``spec`` is unused; the window comes from the eval config alone.
        """
        return self._resolve_eval_window_seconds()

    def _collect_selected_window_segments(
        self,
        *,
        dataloader: T.Any,
        tokenizer_fn: T.Callable[[T.Any], dict],
        rl_cfg: T.Any,
        spec: TokenizerSpec,
        device: torch.device,
        bin_size_seconds: float,
        eval_window_length_seconds: float | None = None,
    ) -> tuple[list[T.Any], list[str], dict[str, list[int]]]:
        """Collect the bounded latent_robustness (M2) evaluation window prefix per session."""
        assert self._eval_config is not None
        selected_segments: list[T.Any] = []
        session_order_seen: list[str] = []
        processed_window_ids_by_session: dict[str, list[int]] = {}
        if eval_window_length_seconds is None:
            eval_window_length_seconds = self._resolve_eval_window_length_seconds(spec)
        expected_time_bins = int(
            round(float(eval_window_length_seconds) / float(bin_size_seconds))
        )

        current_session_id: str | None = None
        current_session_segments: list[T.Any] = []
        current_session_invalid = False
        previous_window_end_seconds: float | None = None

        for segment in iter_collated_validation_segments(
            dataloader,
            tokenizer_fn,
            tokenizer_spec=spec,
            token_bin_size_ms=spec.token_bin_size_ms,
            window_length_seconds=float(eval_window_length_seconds),
            device=device,
            max_windows=1,
            max_bytes=None,
            max_session_windows=rl_cfg.n_eval_windows,
            max_eval_seconds=self._eval_config.max_eval_seconds,
            # Keep overlap trimming active in the collector, but latent_robustness (M2) later
            # rejects any trimmed result that is shorter than the required
            # fixed eval window.
            drop_overlapping_windows=False,
            preserve_overlap_state_on_partial_flush=True,
        ):
            sid = str(segment.session_id)
            if current_session_id is None:
                current_session_id = sid
                session_order_seen.append(sid)
            elif sid != current_session_id:
                if not current_session_invalid:
                    selected_segments.extend(current_session_segments)
                    processed_window_ids_by_session[current_session_id] = [
                        int(seg.segment_index) for seg in current_session_segments
                    ]
                current_session_id = sid
                current_session_segments = []
                current_session_invalid = False
                previous_window_end_seconds = None
                session_order_seen.append(sid)

            if segment.original is None or segment.reconstructed is None:
                raise ValueError(
                    "latent_robustness (M2) selected window is missing original "
                    "or reconstructed tensors."
                )
            original_time_bins = int(segment.original.shape[-1])
            reconstructed_time_bins = int(segment.reconstructed.shape[-1])
            if original_time_bins != reconstructed_time_bins:
                raise ValueError(
                    "latent_robustness (M2) selected window "
                    "original/reconstructed lengths differ: "
                    f"{original_time_bins} vs {reconstructed_time_bins}."
                )
            if original_time_bins < expected_time_bins:
                log.warning(
                    "Skipping session %s window %d for latent_robustness (M2): "
                    "partial window (%d < %d bins).",
                    sid,
                    int(segment.segment_index),
                    original_time_bins,
                    expected_time_bins,
                )
                continue
            if original_time_bins > expected_time_bins:
                raise ValueError(
                    "latent_robustness (M2) selected window is longer than "
                    f"expected: {original_time_bins} > "
                    f"{expected_time_bins} bins."
                )

            assert segment.segment_absolute_start is not None
            window_start_seconds = float(segment.segment_absolute_start)
            if previous_window_end_seconds is not None:
                gap_seconds = window_start_seconds - previous_window_end_seconds
                if gap_seconds > _M2_SESSION_GAP_TOLERANCE_SECONDS:
                    log.warning(
                        "Skipping session %s for latent_robustness (M2): detected "
                        "%.6fs gap between selected windows.",
                        sid,
                        gap_seconds,
                    )
                    current_session_invalid = True
            previous_window_end_seconds = self._window_segment_end_seconds(
                segment, bin_size_seconds
            )
            if current_session_invalid:
                continue
            current_session_segments.append(segment)

        if current_session_id is not None and not current_session_invalid:
            selected_segments.extend(current_session_segments)
            processed_window_ids_by_session[current_session_id] = [
                int(seg.segment_index) for seg in current_session_segments
            ]

        return selected_segments, session_order_seen, processed_window_ids_by_session

    @staticmethod
    def _codes_corruption_seed(
        seed: int,
        eps: float,
        k: int,
        *,
        session_id: str,
        window_index: int,
    ) -> int:
        h = hashlib.blake2b(
            (
                "codes::"
                f"{int(seed)}::{float(eps):.12g}::{int(k)}::"
                f"{session_id}::{int(window_index)}"
            ).encode(),
            digest_size=8,
        ).digest()
        return int.from_bytes(h, byteorder="big", signed=False) % (2**31 - 1)

    @staticmethod
    def _stack_window_tensors(
        window_tensors: list[torch.Tensor],
        *,
        name: str,
    ) -> tuple[torch.Tensor, list[int]]:
        if not window_tensors:
            raise ValueError(f"No {name} tensors provided.")

        reference_shape = tuple(window_tensors[0].shape[1:])
        bad_shapes = [
            tuple(tensor.shape)
            for tensor in window_tensors
            if tuple(tensor.shape[1:]) != reference_shape
        ]
        if bad_shapes:
            raise ValueError(
                f"{name} tensors must share all non-window dimensions; "
                f"expected trailing shape {reference_shape}, got {bad_shapes}."
            )

        lengths = [int(tensor.shape[0]) for tensor in window_tensors]
        return torch.cat(window_tensors, dim=0), lengths

    @staticmethod
    def _split_window_tensor(stacked: torch.Tensor, lengths: list[int]) -> list[torch.Tensor]:
        return list(torch.split(stacked, lengths, dim=0))

    @staticmethod
    def _make_codes_rng(
        seed: int,
        eps: float,
        k: int,
        *,
        session_id: str,
        window_index: int,
    ) -> np.random.Generator:
        return np.random.default_rng(
            NTTEvalRobustnessLatentJob._codes_corruption_seed(
                seed=seed,
                eps=eps,
                k=k,
                session_id=session_id,
                window_index=window_index,
            )
        )

    @staticmethod
    def _draw_corruption_for_codes_with_rng(
        clean_codes: np.ndarray,
        *,
        eps: float,
        vocab_size: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not (0.0 <= float(eps) <= 1.0):
            raise ValueError(f"codes corruption eps must be in [0, 1], got {eps}.")
        swap_mask = rng.random(clean_codes.shape) < float(eps)
        replacements = rng.integers(0, vocab_size, size=clean_codes.shape, dtype=clean_codes.dtype)
        corrupted = np.where(swap_mask, replacements, clean_codes)
        return corrupted, swap_mask

    @staticmethod
    def _draw_corruption_for_codes(
        clean_codes: np.ndarray,
        *,
        eps: float,
        vocab_size: int,
        seed: int,
        k: int,
        session_id: str,
        window_index: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        return NTTEvalRobustnessLatentJob._draw_corruption_for_codes_with_rng(
            clean_codes,
            eps=eps,
            vocab_size=vocab_size,
            rng=NTTEvalRobustnessLatentJob._make_codes_rng(
                seed=seed,
                eps=eps,
                k=k,
                session_id=session_id,
                window_index=window_index,
            ),
        )

    def _build_eval_loader_and_tokenizer_fn(
        self,
        *,
        eval_window_length_seconds: float | None = None,
    ) -> tuple[T.Any, T.Callable[[T.Any], dict]]:
        """Build the dataloader/tokenizer pair used for the clean latent_robustness (M2) pass.

        For fixed-input-window tokenizers (e.g. CS-100ms), the validation
        iterator wraps this raw reconstruct function with
        ``chunked_tokenizer_fn`` once at entry. Variable-T tokenizers go
        through a (near) no-op pass-through. When latent_robustness (M2) needs a source window
        length different from the tokenizer-native datamodule window, this
        method rebuilds the eval datamodule at the requested window length
        so the collator sees the correct source windows to chunk/stitch.
        """
        assert self._tokenizer is not None
        assert self._spec is not None

        tokenizer = self._tokenizer
        if eval_window_length_seconds is None:
            eval_window_length_seconds = self._resolve_eval_window_length_seconds(self._spec)
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
                mode="latent_robustness (M2)",
            )
        stage = "validate" if self.split == "val" else "test"
        requested_window_length_seconds = float(eval_window_length_seconds)
        if (
            abs(float(eval_dm.window_length) - requested_window_length_seconds)
            > _M2_SESSION_GAP_TOLERANCE_SECONDS
        ):
            eval_dm_config = eval_dm.config.model_copy(deep=True)
            eval_dm_config.window_length_seconds = requested_window_length_seconds
            log.info(
                "latent_robustness (M2) eval: rebuilding %s datamodule at "
                "%.3f s windows (was %.3f s).",
                stage,
                requested_window_length_seconds,
                float(eval_dm.window_length),
            )
            eval_dm = NTTDataModule(eval_dm_config)
            eval_dm.setup(stage)

        dataloader = eval_dm.val_dataloader() if self.split == "val" else eval_dm.test_dataloader()

        sampling_kind = self._m2_sampling_kind(tokenizer)
        if sampling_kind is not None:
            expected_tokenizer = _ExpectedValueTokenizer(tokenizer, sampling_kind)

            @torch.no_grad()
            def _tokenizer_fn(batch: T.Any) -> dict:
                # reconstruct_from_input(return_latents=True) without the sampling:
                # M2 samples the clean and noisy reconstructions itself, from shared
                # draws (see _m2_shared_uniforms).
                # encode_with_timestamps already applies tokens_to_latents.
                latents, token_metadata = tokenizer.encode_with_timestamps(batch)
                out: dict = {
                    "reconstruction": expected_tokenizer.reconstruct(latents).reconstruction,
                    "reconstruction_data_type": DataType.SPIKES,
                    "latents": latents,
                }
                if token_metadata is not None:
                    out.update(token_metadata)
                return out
        else:

            @torch.no_grad()
            def _tokenizer_fn(batch: T.Any) -> dict:
                # return_latents=True yields the latents dict
                return T.cast(dict, tokenizer.reconstruct_from_input(batch, return_latents=True))

        return dataloader, _tokenizer_fn

    def _reconstruct_noisy_latents(
        self,
        tokenizer: T.Any,
        noisy_latents: torch.Tensor,
        segment: T.Any,
        clean_recon_t: torch.Tensor,
        sampling_rate: float,
    ) -> T.Any:
        """Decode noisy latents for latent_robustness (M2), including
        embedding-aware perceiver support.
        """
        kind = getattr(tokenizer.spec, "KIND", None)
        if kind == "perceiver_s4":
            return self._reconstruct_noisy_latents_perceiver(
                tokenizer=tokenizer,
                noisy_latents=noisy_latents,
                segment=segment,
                clean_recon_t=clean_recon_t,
                sampling_rate=sampling_rate,
            )
        try:
            return self._reconstruct_tokens_with_fixed_input_chunking(
                tokenizer=tokenizer,
                tokens=noisy_latents,
            )
        except Exception as exc:
            # latent_robustness (M2) contract: ``tokenizer.reconstruct`` must accept the continuous
            # tensor returned by ``tokens_to_latents`` (plus Gaussian noise).
            # Some tokenizers violate this; surface a clean, actionable error
            # here instead of the raw downstream traceback so the sheet runner
            # skips this tokenizer's latent_robustness (M2) pass and moves on to the next
            # checkpoint.
            kind_str = kind or "<unknown>"
            raise RuntimeError(
                f"latent_robustness (M2) decode failed for tokenizer KIND={kind_str!r}: "
                f"{type(exc).__name__}: {exc}\n"
                "This usually means the tokenizer's `reconstruct()` does not "
                "accept the continuous-codes tensor returned by "
                "`tokens_to_latents(tokens)` (the latent_robustness (M2) round-trip contract). "
                "See the latent contract in `src/ntt/contrib/README.md` "
                "('`decode()` must accept the same 4D format') and "
                "`test_tokenizer_m2_eval_latents_reconstruct` in "
                "`tests/test_tokenizer_output_format.py`."
            ) from exc

    @staticmethod
    def _fixed_input_chunk_token_length(spec: TokenizerSpec) -> int:
        """Return the native token count per fixed-input tokenizer window.

        The collator-side chunking wrapper concatenates fixed-input tokenizer
        outputs along token axis 1 when the eval window spans multiple native
        tokenizer windows. latent_robustness (M2)'s later ``tokenizer.reconstruct(...)`` calls must
        invert that split locally: decode each native token window and stitch
        the reconstructed time bins back together.
        """
        token_bin_size_ms = getattr(spec, "token_bin_size_ms", None)
        if token_bin_size_ms is None:
            raise ValueError(
                "Fixed-input tokenizer decode requires spec.token_bin_size_ms to "
                f"split chunk-stitched tokens, but KIND={spec.KIND!r} has None."
            )
        native_window_ms = float(spec.effective_window_length_seconds) * 1000.0
        ratio = native_window_ms / float(token_bin_size_ms)
        nearest = round(ratio)
        if nearest <= 0 or abs(ratio - nearest) > 1e-6:
            raise ValueError(
                "Fixed-input tokenizer decode requires an integral native token count, "
                f"but KIND={spec.KIND!r} has effective_window_length_seconds="
                f"{spec.effective_window_length_seconds} and token_bin_size_ms="
                f"{token_bin_size_ms} (ratio={ratio})."
            )
        return int(nearest)

    def _reconstruct_tokens_with_fixed_input_chunking(
        self,
        *,
        tokenizer: T.Any,
        tokens: torch.Tensor,
    ) -> T.Any:
        """Decode tokens, splitting chunk-stitched fixed-input windows when needed."""
        spec = tokenizer.spec
        if not requires_fixed_input_window(spec):
            return tokenizer.reconstruct(tokens)

        if tokens.ndim < 2:
            return tokenizer.reconstruct(tokens)

        native_chunk_tokens = self._fixed_input_chunk_token_length(spec)
        total_tokens = int(tokens.shape[1])
        if total_tokens < native_chunk_tokens:
            raise ValueError(
                f"Fixed-input tokenizer KIND={spec.KIND!r} received only {total_tokens} "
                f"token bins, shorter than one native chunk ({native_chunk_tokens})."
            )
        if total_tokens == native_chunk_tokens:
            return tokenizer.reconstruct(tokens)
        if total_tokens % native_chunk_tokens != 0:
            raise ValueError(
                f"Fixed-input tokenizer KIND={spec.KIND!r} received {total_tokens} token bins, "
                f"which is not an integer multiple of the native chunk size "
                f"{native_chunk_tokens}."
            )

        # Merged-chunk path: instead of decoding ``n_chunks`` slices
        # sequentially we collapse the chunk axis into the batch axis and
        # call ``tokenizer.reconstruct`` once. For fixed-input tokenizers
        # whose ``reconstruct`` is independent across batch elements (e.g.
        # CS — same sensing matrix and FISTA solver per element) this is
        # exactly equivalent to the sequential loop but removes the
        # 6-7x Python / kernel-launch overhead measured for CS FISTA
        # (measured 978 ms → 160 ms at n_chunks=10).
        n_chunks = total_tokens // native_chunk_tokens
        B = int(tokens.shape[0])
        other_dims = tuple(tokens.shape[2:])
        tokens_merged = tokens.reshape(B * n_chunks, native_chunk_tokens, *other_dims)
        decoded_merged = tokenizer.reconstruct(tokens_merged)
        recon_merged = decoded_merged.reconstruction
        if not isinstance(recon_merged, torch.Tensor):
            raise TypeError(
                "Tokenizer reconstruction output must expose `.reconstruction` as a torch.Tensor."
            )
        if recon_merged.shape[0] != B * n_chunks:
            raise ValueError(
                "Merged-chunk reconstruct: expected batch dim "
                f"{B * n_chunks}, got {recon_merged.shape[0]}; tokenizer "
                f"KIND={spec.KIND!r} may not support batch-merged chunked decode. "
                "Inspect tokenizer.reconstruct contract."
            )
        # [B*n_chunks, T_per_chunk, U, ...] -> [B, n_chunks * T_per_chunk, U, ...].
        T_per_chunk = recon_merged.shape[1]
        recon_tail = tuple(recon_merged.shape[2:])
        reconstruction = recon_merged.reshape(B, n_chunks * T_per_chunk, *recon_tail)
        return types.SimpleNamespace(reconstruction=reconstruction)

    def _reconstruct_noisy_latents_perceiver(
        self,
        tokenizer: T.Any,
        noisy_latents: torch.Tensor,
        segment: T.Any,
        clean_recon_t: torch.Tensor,
        sampling_rate: float,
    ) -> T.Any:
        """Decode perceiver noisy latents using session/unit embeddings from segment metadata."""
        unit_ids = segment.metadata["unit_ids"]
        if not isinstance(unit_ids, list):
            raise TypeError(
                "Expected segment.metadata['unit_ids'] to be a list, got "
                f"{type(unit_ids).__name__}"
            )
        if not unit_ids:
            raise ValueError(
                "Perceiver latent_robustness (M2) decode requires non-empty "
                "segment.metadata['unit_ids']."
            )
        if noisy_latents.ndim != 4:
            raise ValueError(
                "Perceiver latent_robustness (M2) expects noisy latents with "
                f"ndim=4 [N,L,M,D], got {tuple(noisy_latents.shape)}"
            )

        batch_size = noisy_latents.shape[0]
        n_units = len(unit_ids)
        device = noisy_latents.device
        window_duration_seconds = float(clean_recon_t.shape[-1]) / float(sampling_rate)

        # Perceiver decode materializes tensors of shape roughly
        # ``[batch_windows, n_units, token_length]`` internally. Decoding the
        # whole stack of clean + corrupted samples at once can exceed GPU memory
        # even though each sample decodes independently, so we chunk only across
        # that leading sample axis.
        chunk_size = min(batch_size, 8)
        unit_ids_base = torch.tensor(unit_ids, dtype=torch.long, device=device).unsqueeze(0)
        recon_chunks: list[torch.Tensor] = []
        noisy_latents_decode = noisy_latents.permute(0, 2, 3, 1).contiguous()

        for start in range(0, batch_size, chunk_size):
            end = min(start + chunk_size, batch_size)
            chunk_n = end - start
            unit_ids_t = unit_ids_base.expand(chunk_n, -1).contiguous()
            unit_pad_mask_t = torch.ones((chunk_n, n_units), dtype=torch.long, device=device)
            absolute_start_t = torch.zeros((chunk_n,), dtype=torch.float64, device=device)
            absolute_end_t = torch.full(
                (chunk_n,),
                fill_value=window_duration_seconds,
                dtype=torch.float64,
                device=device,
            )
            dummy_input = BinnedSpikesInput(
                spike_matrix=torch.zeros((chunk_n, 1, n_units), dtype=torch.int32, device=device),
                pad_and_gap_mask=torch.zeros(
                    (chunk_n, 1, n_units), dtype=torch.int32, device=device
                ),
                sampling_rate=sampling_rate,
                unit_ids=unit_ids_t,
                unit_ids_pad_mask=unit_pad_mask_t,
                duration_sec=absolute_end_t - absolute_start_t,
                session_id=tuple(segment.session_id for _ in range(chunk_n)),
                subject_id=tuple(segment.metadata["subject_id"] for _ in range(chunk_n)),
                species=tuple(segment.metadata["species"] for _ in range(chunk_n)),
                absolute_start=absolute_start_t,
                absolute_end=absolute_end_t,
                behavioral_data=None,
                behavioral_intervals=None,
            )
            embeddings, _mask = tokenizer._get_embeddings(dummy_input)
            # PerceiverTokenizer.decode(tokens) cannot take embeddings (it raises);
            # decode through the count wrapper like reconstruct_from_input does.
            # Rates come out (B, units, time); return (B, time, units), the
            # layout _to_units_time expects (and reconstruct_from_input returns).
            rates = tokenizer.count_wrapper.decode(noisy_latents_decode[start:end], embeddings)
            recon_chunks.append(rates.transpose(-1, -2))

        reconstruction = torch.cat(recon_chunks, dim=0)
        assert reconstruction.shape[0] == batch_size, (
            "Perceiver chunked decode lost windows: "
            f"expected {batch_size}, got {reconstruction.shape[0]}."
        )
        return types.SimpleNamespace(reconstruction=reconstruction)

    def _decode_reconstruction_batch(
        self,
        *,
        tokenizer: T.Any,
        batch_tensor: torch.Tensor,
        corruption_target: str,
        segment: T.Any,
        clean_recon_t: torch.Tensor,
        sampling_rate: float | None,
        max_decode_batch_size: int | None,
    ) -> torch.Tensor:
        """Decode a stacked batch of latent/code perturbations, optionally in chunks."""
        batch_size = int(batch_tensor.shape[0])
        chunk_size = batch_size if max_decode_batch_size is None else int(max_decode_batch_size)
        recon_chunks: list[torch.Tensor] = []

        for start in range(0, batch_size, chunk_size):
            end = min(start + chunk_size, batch_size)
            batch_chunk = batch_tensor[start:end]
            with torch.no_grad():
                if corruption_target == "latents":
                    decoded = self._reconstruct_noisy_latents(
                        tokenizer=tokenizer,
                        noisy_latents=batch_chunk,
                        segment=segment,
                        clean_recon_t=clean_recon_t,
                        # only the perceiver decode path reads the sampling rate
                        sampling_rate=T.cast(float, sampling_rate),
                    )
                elif corruption_target == "codes":
                    decoded = self._reconstruct_tokens_with_fixed_input_chunking(
                        tokenizer=tokenizer,
                        tokens=batch_chunk,
                    )
                else:
                    raise ValueError(
                        f"Unsupported corruption_target={corruption_target!r} for decode."
                    )
            reconstruction = decoded.reconstruction
            if not isinstance(reconstruction, torch.Tensor):
                raise TypeError(
                    "Tokenizer reconstruction output must expose `.reconstruction` "
                    "as a torch.Tensor."
                )
            recon_chunks.append(reconstruction)

        if not recon_chunks:
            raise ValueError("decode batch was empty.")
        return torch.cat(recon_chunks, dim=0)

    @staticmethod
    def _is_poisson_baseline_tokenizer(tokenizer: T.Any) -> bool:
        """Whether the tokenizer is the Poisson baseline (it samples in ``reconstruct``)."""
        return type(tokenizer).__name__ == "PoissonBaselineTokenizer"

    @classmethod
    def _m2_sampling_kind(cls, tokenizer: T.Any) -> T.Literal["bernoulli", "poisson"] | None:
        """How the tokenizer's ``reconstruct`` samples spikes, or None if it is deterministic.

        For these tokenizers latent_robustness (M2) decodes expected values and samples
        the clean and noisy reconstructions itself from shared draws.
        """
        if cls._is_poisson_baseline_tokenizer(tokenizer):
            return "poisson"
        spec = tokenizer.spec
        if getattr(spec, "KIND", None) == "calm":
            mode = getattr(spec, "reconstruct_sample", None)
            if mode in ("bernoulli", "poisson"):
                return mode
        return None

    @classmethod
    def _m2_sampling_label(cls, tokenizer: T.Any) -> str | None:
        """How M2 treats the tokenizer's sampling, recorded in the report and run_meta.

        ``shared_uniforms:<kind>`` when M2 samples with shared draws; ``independent``
        for a tokenizer that samples in ``reconstruct`` but that M2 does not take over
        (each decode draws afresh, so the score includes sampling noise); None for
        deterministic decoding.
        """
        kind = cls._m2_sampling_kind(tokenizer)
        if kind is not None:
            return f"shared_uniforms:{kind}"
        spec = tokenizer.spec
        if getattr(spec, "reconstruct_sample", None) in ("bernoulli", "poisson"):
            return "independent"
        if getattr(spec, "KIND", None) == "conv_vae":  # Poisson-samples outside training
            return "independent"
        return None

    @staticmethod
    def _m2_shared_uniforms(
        shape: tuple[int, ...],
        *,
        seed: int,
        session_id: str,
        window_start: float,
    ) -> torch.Tensor:
        """Uniform draws in [0, 1) shared by a window's clean and noisy reconstructions.

        Keyed on the window's identity, not on eps or the replica, so with zero latent
        noise every noisy sample equals the clean one and M2 measures how the latents
        move the reconstruction, not sampling noise. Reproducible and independent of
        decode batching; distinct across windows, sessions and seeds.
        """
        digest = hashlib.blake2b(
            f"m2_shared_draws::{int(seed)}::{session_id}::{float(window_start):.6f}".encode(),
            digest_size=8,
        ).digest()
        generator = torch.Generator()
        generator.manual_seed(int.from_bytes(digest, byteorder="big") % (2**63 - 1))
        return torch.rand(shape, generator=generator, dtype=torch.float64)

    @staticmethod
    def _sample_with_shared_uniforms(
        expected: torch.Tensor,
        uniforms: torch.Tensor,
        *,
        kind: T.Literal["bernoulli", "poisson"],
    ) -> torch.Tensor:
        """Sample spikes from expected values with given uniforms (common random numbers).

        Bernoulli: a spike where ``u < p``. Poisson: the inverse CDF at ``u``, so for a
        fixed ``u`` the count never decreases as the rate grows. Non-finite expected
        values raise, as ``torch.bernoulli`` / ``torch.poisson`` do.
        """
        u = uniforms.to(device=expected.device, dtype=torch.float64)
        expected64 = expected.to(torch.float64)
        if not bool(torch.isfinite(expected64).all()):
            raise ValueError(
                "latent_robustness (M2) decoder output has non-finite values; "
                "cannot sample spikes from it."
            )
        if kind == "bernoulli":
            return (u < expected64.clamp(0.0, 1.0)).to(torch.float32)

        rate = expected64.clamp(min=0.0)
        flat_rate, flat_u = rate.reshape(-1), u.reshape(-1)
        counts = torch.zeros_like(flat_rate)
        large = flat_rate > _M2_EXACT_POISSON_MAX_RATE
        # Walk the CDF only for the cells still above it (most cells stop at k=0).
        pmf = torch.exp(-flat_rate)
        idx = ((flat_u >= pmf) & ~large).nonzero().squeeze(1)
        r_rate, r_u, r_pmf = flat_rate[idx], flat_u[idx], pmf[idx]
        r_cdf = r_pmf.clone()
        k = 0
        while idx.numel() > 0:
            k += 1
            r_pmf = r_pmf * r_rate / k
            r_cdf = r_cdf + r_pmf
            # The float CDF can stall just below u in the far tail; stop there.
            settled = (r_u < r_cdf) | (r_pmf == 0)
            counts[idx[settled]] = k
            if bool(settled.any()):
                keep = ~settled
                idx, r_rate, r_u = idx[keep], r_rate[keep], r_u[keep]
                r_pmf, r_cdf = r_pmf[keep], r_cdf[keep]
        if bool(large.any()):
            # Cornish-Fisher Poisson quantile with continuity correction, monotone in
            # the rate for fixed u here. Floored at the exact count at the cap, so the
            # count stays monotone across the switch from exact to approximate.
            l_rate, l_u = flat_rate[large], flat_u[large]
            z = torch.special.ndtri(l_u.clamp(2**-53, 1.0 - 2**-53))
            quantile = l_rate + l_rate.sqrt() * z + (z * z - 1.0) / 6.0 - 0.5
            cap_cdf = torch.tensor(_poisson_cdf_at_exact_cap(), dtype=torch.float64)
            cap_counts = torch.searchsorted(cap_cdf.to(l_u.device), l_u, right=True)
            cap_counts = cap_counts.clamp(max=len(cap_cdf) - 1).to(torch.float64)
            counts[large] = torch.maximum(torch.ceil(quantile), cap_counts)
        counts = counts.reshape(rate.shape)
        return counts.to(torch.float32)

    def _build_all_neuron_datamodule(self) -> NTTDataModule:
        """Build an evaluation datamodule that mirrors the main eval all-neuron path:
        every unit, capped at ``eval_config.max_neurons`` by the shared selection."""
        from ntt.core.data.utils import n_neurons_per_session
        from ntt.evaluation.utils.window_contract import apply_eval_window

        assert self._spec is not None
        assert self._eval_config is not None
        require_direct_all_neuron_eval(self._spec, what="latent_robustness (M2)")

        all_neuron_config = all_neuron_data_config(self.data_config, self._eval_config, self._spec)
        stage = "validate" if self.split == "val" else "test"

        ds = self.datamodule.val_dataset if self.split == "val" else self.datamodule.test_dataset
        assert ds is not None, "datamodule.setup() must have created the eval dataset"
        neuron_counts = n_neurons_per_session(ds)
        log.info(
            "latent_robustness (M2) full-neuron eval: at most %s units per session, a seeded "
            "random subset where there are more (max in dataset: %d)",
            self._eval_config.max_neurons,
            max(neuron_counts.values()),
        )

        apply_eval_window(all_neuron_config, self._spec, self._resolve_eval_window_seconds())
        dm = NTTDataModule(all_neuron_config)
        self._check_scored_window(dm, "latent_robustness (M2) all-neuron eval")
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
    def _epsilon_metric_key(metric: str, epsilon: float) -> str:
        return f"{metric}/eps_{epsilon:g}"

    def _build_reports_from_m2_rows(
        self,
        rows: list[dict[str, T.Any]],
        metric_schemas: dict[str, dict[str, T.Any]],
        *,
        report_config: dict[str, T.Any],
        original_data_type: T.Any,
        reconstructed_data_type: T.Any,
    ) -> tuple[dict[str, EvaluationReport], EvaluationReport]:
        from ntt.evaluation.aggregation.summary import create_report

        grouped: dict[str, dict[str, list[float]]] = {}
        meta_counts: dict[str, dict[str, dict[str, T.Any]]] = {}
        for row in rows:
            session_id = row.get("session_id")
            epsilon = row.get("epsilon")
            if session_id is None or epsilon is None:
                continue
            try:
                eps_f = float(epsilon)
            except (TypeError, ValueError):
                continue
            for metric_name in metric_schemas:
                value = row.get(metric_name)
                if value is None:
                    continue
                try:
                    value_f = float(value)
                except (TypeError, ValueError):
                    continue
                metric_key = self._epsilon_metric_key(metric_name, eps_f)
                grouped.setdefault(str(session_id), {}).setdefault(metric_key, []).append(value_f)
                meta_counts.setdefault(str(session_id), {}).setdefault(
                    metric_key,
                    {"epsilon": eps_f, "n_rows": 0},
                )
                meta_counts[str(session_id)][metric_key]["n_rows"] += 1

        per_session_reports: dict[str, EvaluationReport] = {}
        for session_id, metrics in grouped.items():
            report_metrics: dict[str, MetricResult] = {}
            for metric_key, values in metrics.items():
                base_metric = metric_key.split("/", 1)[0]
                schema = metric_schemas.get(base_metric)
                if schema is None:
                    continue
                counts = meta_counts[session_id][metric_key]
                report_metrics[metric_key] = self._metric_result_from_schema(
                    schema,
                    name=metric_key,
                    raw_value=float(np.mean(values)),
                    metadata={
                        "epsilon": counts["epsilon"],
                        "n_rows": int(counts["n_rows"]),
                        "aggregation_strategy": "mean",
                    },
                )
            per_session_report = create_report(report_metrics, config=report_config)
            per_session_report.original_data_type = original_data_type
            per_session_report.reconstructed_data_type = reconstructed_data_type
            per_session_reports[session_id] = per_session_report

        aggregated_report = aggregate_per_session_dicts(
            {sid: report.to_dict() for sid, report in per_session_reports.items()},
            AggregationStrategy.MEAN,
        )
        return per_session_reports, aggregated_report

    def _build_m2_ceiling_reports(
        self,
        per_session_reports: dict[str, EvaluationReport],
    ) -> dict[str, EvaluationReport]:
        out: dict[str, EvaluationReport] = {}
        for session_id, report in per_session_reports.items():
            ceiling_metrics: dict[str, MetricResult] = {}
            for metric_name, metric_result in report.results.items():
                base_metric = metric_name.split("/", 1)[0]
                zero_metric_name = self._epsilon_metric_key(base_metric, 0.0)
                zero_metric = report.results.get(zero_metric_name)
                if zero_metric is None:
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
                    raw_value=float(zero_metric.raw_value),
                    metadata={"ceiling_source_metric": zero_metric_name},
                )
            if ceiling_metrics:
                out[session_id] = EvaluationReport(results=ceiling_metrics)
        return out

    def _load_poisson_m2_reference(
        self,
        metric_schemas: dict[str, dict[str, T.Any]],
    ) -> tuple[dict[str, EvaluationReport], EvaluationReport, float | None]:
        assert self.poisson_reference is not None

        csv_path, run_meta_path = resolve_poisson_reference_paths(
            self.poisson_reference, mode="latent_robustness (M2)"
        )

        rows: list[dict[str, T.Any]] = []
        import csv

        with csv_path.open() as fh:
            reader = csv.DictReader(fh)
            csv_columns = set(reader.fieldnames or [])
            missing_metric_cols = sorted(set(metric_schemas) - csv_columns)
            if missing_metric_cols:
                log.warning(
                    "M2 poisson_reference at %s is missing CSV columns for %d "
                    "current metric(s): %s. Those metrics will be silently "
                    "skipped in the comparison (row.get returns None). Likely "
                    "the poisson_reference was produced with a different "
                    "metric set than the current model eval.",
                    csv_path,
                    len(missing_metric_cols),
                    missing_metric_cols,
                )
            for row in reader:
                rec: dict[str, T.Any] = {
                    "session_id": row.get("session_id"),
                    "epsilon": row.get("epsilon"),
                }
                for metric_name in metric_schemas:
                    rec[metric_name] = row.get(metric_name)
                rows.append(rec)
        # ``_build_reports_from_m2_rows`` requires three keyword arguments —
        # ``report_config`` (the eval.report config dict), ``original_data_type``
        # and ``reconstructed_data_type`` (the EvalData type tags the model run
        # was scored against). All three describe the
        # *model* run, not the poisson reference. The poisson reports built
        # here feed only into ``compute_normalized_comparisons``, which reads
        # ``raw_value`` / ``category`` / ``direction`` / ``level`` — not the
        # report config or data types — so passing ``{}`` and ``None`` is safe
        # and avoids fabricating model-side metadata for the baseline.
        per_session_reports, aggregated_report = self._build_reports_from_m2_rows(
            rows,
            metric_schemas,
            report_config={},
            original_data_type=None,
            reconstructed_data_type=None,
        )
        poisson_rate_bin_ms = load_poisson_rate_bin_ms(run_meta_path, mode="M2", log=log)
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
    ) -> tuple[dict[str, T.Any], dict[str, float]]:
        poisson_per_session, poisson_aggregated, poisson_rate_bin_ms = (
            self._load_poisson_m2_reference(metric_schemas)
        )
        ceiling_per_session = self._build_m2_ceiling_reports(per_session_reports)

        per_session_comparisons: dict[str, dict[str, T.Any]] = {}
        normalized_by_session_metric: dict[str, list[float]] = {}
        extra_metrics: dict[str, float] = {}
        shared_sessions = sorted(
            warn_if_no_session_overlap(
                model_sessions=set(per_session_reports),
                poisson_sessions=set(poisson_per_session),
                ceiling_sessions=set(ceiling_per_session),
                mode="M2",
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
        model_aggregated_shared = aggregate_per_session_dicts(
            {sid: report.to_dict() for sid, report in model_shared.items()},
            AggregationStrategy.MEAN,
        )
        poisson_aggregated_shared = aggregate_per_session_dicts(
            {sid: report.to_dict() for sid, report in poisson_shared.items()},
            AggregationStrategy.MEAN,
        )
        ceiling_aggregated = aggregate_per_session_dicts(
            {sid: report.to_dict() for sid, report in ceiling_shared.items()},
            AggregationStrategy.MEAN,
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
    def _write_per_session_csv(rows: list[dict[str, T.Any]], path: Path) -> None:
        """Write one latent_robustness (M2) summary row per (session, epsilon).

        The in-memory ``rows`` store one wide row per ``(session, epsilon, k)``.
        This helper averages over ``k`` to produce explicit per-session
        robustness curves for downstream analysis.
        """
        if not rows:
            path.write_text("")
            return

        try:
            import pandas as pd
        except ImportError:
            return

        df = pd.DataFrame(rows)
        if df.empty or "session_id" not in df.columns or "epsilon" not in df.columns:
            path.write_text("")
            return

        value_cols = [
            c
            for c in df.columns
            if c not in {"session_id", "segment_index", "epsilon", "k"}
            and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not value_cols:
            path.write_text("")
            return

        summary = df.groupby(["session_id", "epsilon"], as_index=False)[value_cols].mean(
            numeric_only=True
        )
        counts = (
            df.groupby(["session_id", "epsilon"], as_index=False)
            .size()
            .rename(columns={"size": "n_rows"})
        )
        summary = summary.merge(counts, on=["session_id", "epsilon"], how="left")
        summary.to_csv(path, index=False)

    @staticmethod
    def _segment_latents_tensor(segment: T.Any) -> torch.Tensor:
        """Return segment.latents as a single tensor.

        Supports the ``torch.Tensor`` case (stacked across windows) and the
        ``list[Tensor]`` fallback (which we concatenate along dim 0).
        """
        lat = segment.latents
        if isinstance(lat, torch.Tensor):
            return lat
        if isinstance(lat, list):
            if not lat:
                raise ValueError("Empty latents list on segment.")
            return torch.cat(list(lat), dim=0)
        raise TypeError(f"Unsupported latents type: {type(lat).__name__}")

    @staticmethod
    def _token_distance_tokens_from_latents(
        tokenizer: T.Any,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """Derive discrete token IDs from a latent tensor for token-distance metadata."""
        if hasattr(tokenizer, "quantize"):
            tokens = tokenizer.quantize(latents)
            if not isinstance(tokens, torch.Tensor):
                raise TypeError(
                    "tokenizer.quantize(latents) must return a torch.Tensor for "
                    "token-distance metadata."
                )
            if tokens.is_floating_point():
                raise TypeError(
                    "tokenizer.quantize(latents) returned floating codes; "
                    "latent_robustness (M2) token-distance "
                    "metadata expects discrete index-like codes."
                )
            return tokens

        if getattr(getattr(tokenizer, "spec", None), "KIND", None) == "wanneural":
            bottleneck = getattr(tokenizer, "bottleneck", None)
            if bottleneck is None or not hasattr(bottleneck, "get_indices"):
                raise AttributeError(
                    "WanNeural tokenizer is missing bottleneck.get_indices() for "
                    "token-distance metadata."
                )
            if latents.ndim != 4:
                raise ValueError(
                    "WanNeural token-distance fallback expects latent tensor "
                    "shape [B, T, N, D]; "
                    f"got {tuple(latents.shape)}."
                )
            latent_bctn = latents.permute(0, 3, 1, 2).contiguous()
            tokens = bottleneck.get_indices(latent_bctn)
            if not isinstance(tokens, torch.Tensor):
                raise TypeError(
                    "WanNeural bottleneck.get_indices() must return a torch.Tensor for "
                    "token-distance metadata."
                )
            if tokens.is_floating_point():
                raise TypeError(
                    "WanNeural bottleneck.get_indices() returned floating codes; "
                    "latent_robustness (M2) "
                    "token-distance metadata expects discrete index-like codes."
                )
            return tokens

        raise AttributeError(
            f"{type(tokenizer).__name__} cannot derive discrete tokens from latents: "
            "tokenizer lacks quantize(latents) and no supported bottleneck fallback exists."
        )

    @staticmethod
    def _to_units_time(recon_t: torch.Tensor) -> torch.Tensor:
        """Normalize reconstruction tensor shape to ``(units, total_time)``,
        stitched across the window axis to match the per-segment clean
        reconstruction.

        The tokenizer's ``reconstruct()`` returns shape ``[N_windows, T, U]``
        when fed stacked window latents. The clean ``segment.reconstructed``
        is built by ``validation_collation`` as the per-window slices
        concatenated along the time axis, i.e. ``(U, N_windows * T)``. We
        replicate that here so ``EvalData.from_tensor`` consumes a 2-D
        ``(U, total_time)`` payload that matches clean.
        """
        # ``reconstruct()`` from the base tokenizer uses ``decode()`` which
        # typically returns [N_windows, T, U]. We transpose to
        # [N_windows, U, T] then reshape to [U, N_windows*T] so that the
        # noisy reconstruction aligns with the clean session stitch.
        if recon_t.ndim == 3:
            r = recon_t.transpose(-1, -2).contiguous()  # [N, U, T]
            r = r.permute(1, 0, 2).contiguous()  # [U, N, T]
            U, N, T = r.shape
            return r.reshape(U, N * T)
        if recon_t.ndim == 2:
            # Assume (T, U) — transpose to (U, T)
            return recon_t.transpose(-1, -2).contiguous()
        raise ValueError(f"Unexpected reconstruction ndim: {recon_t.ndim}")

    @staticmethod
    def _extract_sampling_rate(segment: T.Any) -> float | None:
        """Pull a scalar sampling rate from ``segment.metadata`` (if present)."""
        meta = getattr(segment, "metadata", None) or {}
        rate = meta.get("sampling_rate")
        if rate is None:
            return None
        if isinstance(rate, torch.Tensor):
            if rate.ndim > 0:
                rate = rate.flatten()[0]
            return float(rate.item())
        return float(rate)

    def _build_clean_cache(
        self,
        clean_original: EvalData,
        clean_reconstructed: EvalData,
        segment: T.Any,
        sampling_rate: float | None,
        token_bin_size_ms: float | None,
    ):
        """Build the shared clean-side :class:`IntermediatesCache`.

        Populates ``original=clean_spikes`` and ``reconstructed=clean_recon``
        along with ``latents`` / ``token_timestamps`` / ``sampling_rate`` /
        ``token_bin_size_ms`` / ``behavioral_data_*`` passthroughs sourced
        from ``segment``. ``replace_reconstructed(noisy)`` is used inside
        :func:`evaluate_pair` to reuse the clean-side cached intermediates
        across many noise draws.
        """
        from ntt.evaluation.core.intermediates import IntermediatesCache

        # Convert latents to numpy per-pipeline convention.
        lat = segment.latents
        if isinstance(lat, torch.Tensor):
            latents_np = lat.detach().cpu().numpy()
        elif isinstance(lat, list):
            latents_np = [
                li.detach().cpu().numpy() if isinstance(li, torch.Tensor) else li for li in lat
            ]
        else:
            latents_np = None

        token_timestamps_np = None
        if segment.token_timestamps is not None:
            token_timestamps_np = segment.token_timestamps.detach().cpu().numpy()

        assert self._eval_config is not None
        return IntermediatesCache(
            original=clean_original,
            reconstructed=clean_reconstructed,
            latents=latents_np,
            behavioral_data_token_res=segment.behavioral_data_token_res,
            behavioral_data_input_res=segment.behavioral_data_input_res,
            token_timestamps=token_timestamps_np,
            sampling_rate=sampling_rate,
            segment_absolute_start=segment.segment_absolute_start,
            token_bin_size_ms=token_bin_size_ms,
        )

    def _compute_sigma(self, segments: list, seed: int) -> LatentStats:
        """Compute σ via :class:`LatentStatsCache`.

        Cache key includes: tokenizer checkpoint path + mtime + size, dataset
        selection name, split, session ids (sorted), windowing params, seed.
        """
        assert self._spec is not None

        ckpt_stat: dict[str, T.Any] = {"path": str(self.ckpt_path) if self.ckpt_path else None}
        if self.ckpt_path is not None:
            try:
                p = Path(self.ckpt_path)
                if p.exists():
                    st = p.stat()
                    ckpt_stat["mtime"] = int(st.st_mtime)
                    ckpt_stat["size"] = int(st.st_size)
            except OSError:
                pass

        session_ids = sorted({seg.session_id for seg in segments})
        dataset_selection_name = getattr(self.data_config, "selection_name", None) or getattr(
            self.data_config, "name", None
        )

        key_dict = {
            "ckpt": ckpt_stat,
            "tokenizer_kind": getattr(self._spec, "KIND", None),
            "dataset_selection": dataset_selection_name,
            "split": self.split,
            "session_ids": session_ids,
            "n_segments": len(segments),
            "bin_size_ms": getattr(self._spec, "bin_size_ms", None),
            "token_bin_size_ms": getattr(self._spec, "token_bin_size_ms", None),
            "seed": int(seed),
        }

        def _collect_latents() -> np.ndarray:
            """Concat all latents across segments, flattened to ``(N, D)``."""
            chunks: list[np.ndarray] = []
            for seg in segments:
                if seg.latents is None:
                    continue
                if isinstance(seg.latents, torch.Tensor):
                    arr = seg.latents.detach().cpu().numpy()
                elif isinstance(seg.latents, list):
                    arr_list = [
                        li.detach().cpu().numpy() if isinstance(li, torch.Tensor) else li
                        for li in seg.latents
                    ]
                    arr = np.concatenate(arr_list, axis=0) if arr_list else None
                else:
                    arr = None
                if arr is None or arr.size == 0:
                    continue
                # Flatten all leading dims against a trailing latent-dim.
                # We assume the last axis is the latent-dim D.
                D = arr.shape[-1]
                chunks.append(arr.reshape(-1, D))
            if not chunks:
                raise ValueError("No latents collected across selected segments.")
            return np.concatenate(chunks, axis=0)

        on_disk_root: Path | None = None
        if self._eval_config is not None and self._eval_config.latent_stats_on_disk:
            on_disk_root = Path(".ntt_cache") / "latent_stats"

        cache = LatentStatsCache(on_disk_root=on_disk_root)
        return cache.get_or_compute(key_dict, _collect_latents)

    @staticmethod
    def _draw_noise(
        shape: tuple[int, ...],
        eps: float,
        sigma_per_dim: np.ndarray,
        distribution: str,
        seed: int,
        k: int,
        session_id: str,
        window_index: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Draw a latent-shaped noise tensor, deterministic in ``(seed, eps, k, window)``.

        Noise is scaled **per latent dim**: ``ξ_d ~ N(0, (ε·σ_d)²)`` (or
        Uniform(−ε·σ_d, ε·σ_d)). σ_per_dim must have length equal to the
        last axis of ``shape`` — it broadcasts as ``[..., D]``.

        Earlier versions scaled jointly with a single σ_joint, which
        under-perturbed dims with σ_d ≪ σ_joint and over-perturbed dims
        with σ_d ≫ σ_joint. Per-dim scaling makes ε meaningful as
        "fraction of each dim's natural standard deviation", uniformly.
        """
        return NTTEvalRobustnessLatentJob._draw_noise_from_generator(
            shape=shape,
            eps=eps,
            sigma_per_dim=sigma_per_dim,
            distribution=distribution,
            generator=NTTEvalRobustnessLatentJob._make_noise_generator(
                seed=seed,
                eps=eps,
                k=k,
                session_id=session_id,
                window_index=window_index,
            ),
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _make_noise_generator(
        seed: int,
        eps: float,
        k: int,
        *,
        session_id: str,
        window_index: int,
    ) -> torch.Generator:
        # Hash (seed, eps, k, session_id, window_index) -> int32 noise seed.
        h = hashlib.blake2b(
            (
                f"{int(seed)}::{float(eps):.12g}::{int(k)}::{session_id}::{int(window_index)}"
            ).encode(),
            digest_size=8,
        ).digest()
        noise_seed = int.from_bytes(h, byteorder="big", signed=False) % (2**31 - 1)
        return torch.Generator(device="cpu").manual_seed(noise_seed)

    @staticmethod
    def _draw_noise_from_generator(
        shape: tuple[int, ...],
        eps: float,
        sigma_per_dim: np.ndarray,
        distribution: str,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if distribution == "gaussian":
            xi = torch.randn(shape, generator=generator, dtype=torch.float32)
        elif distribution == "uniform":
            xi = torch.rand(shape, generator=generator, dtype=torch.float32) * 2 - 1
        else:
            raise ValueError(f"Unknown noise_distribution: {distribution!r}")

        # Broadcast σ_per_dim along leading axes — last axis = D.
        sigma_per_dim_arr = np.asarray(sigma_per_dim, dtype=np.float32)
        if sigma_per_dim_arr.ndim != 1:
            raise ValueError(f"sigma_per_dim must be 1D [D]; got shape {sigma_per_dim_arr.shape}")
        D = sigma_per_dim_arr.shape[0]
        if shape[-1] != D:
            raise ValueError(
                f"Latent shape last axis ({shape[-1]}) must equal sigma_per_dim length ({D})."
            )
        sigma_t = torch.from_numpy(sigma_per_dim_arr)  # [D], float32
        return (xi * eps * sigma_t).to(device=device, dtype=dtype)

    @staticmethod
    def _compute_mse_match_miss(clean_recon: torch.Tensor, noisy_recon: torch.Tensor) -> float:
        """Simple numpy MSE between clean and noisy reconstructions.

        Matches input_robustness (M1)'s ``mse_match_miss`` definition (mean squared difference
        over binned counts).  Broadcasts shapes after squeezing batch.
        """
        a = clean_recon.detach().cpu().numpy()
        b = noisy_recon.detach().cpu().numpy()
        if a.ndim == 3:
            a = a[0]
        if b.ndim == 3:
            b = b[0]
        # Align on the smaller shape in case of trailing pad differences.
        min_u = min(a.shape[0], b.shape[0])
        min_t = min(a.shape[1], b.shape[1])
        a = a[:min_u, :min_t]
        b = b[:min_u, :min_t]
        diff = a.astype(np.float64) - b.astype(np.float64)
        return float(np.mean(diff * diff))

    @staticmethod
    def _write_csv(rows: list[dict[str, T.Any]], path: Path) -> None:
        """Write the wide CSV: one row per ``(session, epsilon)`` with mean/std over ``k``.

        Columns: ``{metric}_mean``, ``{metric}_std``, ``mse_match_miss_mean``,
        ``mse_match_miss_std``. Uses pandas if available, else csv module.
        """
        if not rows:
            path.write_text("")
            return
        try:
            import pandas as pd

            df = pd.DataFrame(rows)
            if df.empty or "session_id" not in df.columns or "epsilon" not in df.columns:
                path.write_text("")
                return
            value_cols = [
                c
                for c in df.columns
                if c not in ("session_id", "segment_index", "epsilon", "k")
                and pd.api.types.is_numeric_dtype(df[c])
            ]
            if not value_cols:
                path.write_text("")
                return
            agg = df.groupby(["session_id", "epsilon"], as_index=False)[value_cols].agg(
                ["mean", "std"]
            )
            # Flatten MultiIndex columns -> "{col}_mean"/"{col}_std"
            # (pandas-stubs types ``columns`` as Index[str]; it is a MultiIndex after agg([...]))
            flat_columns = agg.columns.to_flat_index()  # pyright: ignore[reportAttributeAccessIssue]
            agg.columns = [
                "_".join([c for c in col if c]).rstrip("_") if isinstance(col, tuple) else col
                for col in flat_columns
            ]
            agg.to_csv(path, index=False)
        except ImportError:
            import csv

            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

    @staticmethod
    def _aggregate_token_distance(
        records: list[dict[str, T.Any]] | None,
    ) -> list[dict[str, T.Any]]:
        """Aggregate per-window token-distance records into a compact summary.

        Returns one entry per ``(session_id, eps)`` with the mean and std of
        ``fraction_changed`` over the windows x replicas seen for that pair,
        plus ``n_codes`` and the number of records aggregated. The raw
        per-window/per-replica records are intentionally NOT persisted.
        """
        if not records:
            return []
        grouped: dict[tuple[str, float], list[dict[str, T.Any]]] = {}
        for rec in records:
            grouped.setdefault((str(rec["session_id"]), float(rec["eps"])), []).append(rec)
        summary: list[dict[str, T.Any]] = []
        for session_id, eps in sorted(grouped):
            recs = grouped[(session_id, eps)]
            fractions = np.asarray([float(r["fraction_changed"]) for r in recs], dtype=float)
            n_codes_values = {int(r["n_codes"]) for r in recs}
            summary.append(
                {
                    "session_id": session_id,
                    "eps": eps,
                    "mean_fraction_changed": float(fractions.mean()),
                    "std_fraction_changed": float(fractions.std(ddof=1))
                    if fractions.size > 1
                    else 0.0,
                    # n_codes is fixed per session; record the value(s) seen.
                    "n_codes": int(max(n_codes_values)),
                    "n_records": int(fractions.size),
                }
            )
        return summary

    @staticmethod
    def _write_run_meta(
        path: Path,
        stats: LatentStats,
        skipped_counter: dict[str, int],
        total_pairs: int,
        rl_cfg: T.Any,
        eval_window_length_seconds: float = _DEFAULT_M2_EVAL_WINDOW_SECONDS,
        spec: T.Any | None = None,
        tokenizer: T.Any | None = None,
        checkpoint: str | Path | None = None,
        token_distance_records: list[dict[str, T.Any]] | None = None,
        eval_config: T.Any | None = None,
        data_config: T.Any | None = None,
        hydra_overrides: list[str] | None = None,
        composed_config_yaml: str | None = None,
        evaluate_all_neurons: bool | None = None,
        neuron_selection: NeuronSelection | None = None,
        eval_window: dict[str, T.Any] | None = None,
    ) -> dict[str, T.Any]:
        """Write ``run_meta.json`` per D11 and return its contents.

        ``spec`` (optional) is the resolved :class:`TokenizerSpec`; when
        present we persist tokenizer-identifying knobs (``KIND`` and
        spec-specific fields like ``rate_estimation_bin_ms`` for
        Poisson) so the replot CLI can disambiguate variants in joint
        legends.

        ``tokenizer`` (optional) is the runtime tokenizer instance.
        When it exposes a ``reconstruction_lipschitz_bound`` property
        (currently ``CompressedSensingTokenizer`` only), the value is
        persisted so the replot CLI can append it to the legend label
        as a stability annotation. The metadata also carries the
        checkpoint path when present plus the full serialized tokenizer
        config, matching the main eval's run-identification payload.
        """
        meta = {
            "sigma_joint": float(stats.sigma_joint),
            "sigma_per_dim": np.asarray(stats.sigma_per_dim).tolist(),
            "skipped_per_metric": dict(skipped_counter),
            "total_pairs": int(total_pairs),
            "epsilons": list(rl_cfg.epsilons),
            "n_samples_per_epsilon": int(rl_cfg.n_samples_per_epsilon),
            "noise_distribution": str(rl_cfg.noise_distribution),
            "n_eval_windows": rl_cfg.n_eval_windows,
            "checkpoint": str(checkpoint) if checkpoint is not None else None,
            "eval_window_length_seconds": float(eval_window_length_seconds),
        }
        # Units each session was scored on; a robustness comparison checks it against
        # the Poisson reference run's.
        meta["evaluate_all_neurons"] = evaluate_all_neurons
        meta.update(eval_window or {})  # the window actually scored, and the spec's own
        meta["neuron_selection"] = (
            neuron_selection.model_dump() if neuron_selection is not None else None
        )
        if tokenizer is not None:
            meta[M2_SAMPLING_KEY] = NTTEvalRobustnessLatentJob._m2_sampling_label(tokenizer)
        if composed_config_yaml is not None:
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
            rate_bin = getattr(spec, "rate_estimation_bin_ms", None)
            if rate_bin is not None:
                try:
                    meta["rate_estimation_bin_ms"] = float(rate_bin)
                except (TypeError, ValueError):
                    pass
        if tokenizer is not None:
            # Optional, tokenizer-specific stability metadata. Read via
            # getattr so any tokenizer that doesn't expose the property
            # silently skips it (no hasattr+access race in @property).
            lip = getattr(tokenizer, "reconstruction_lipschitz_bound", None)
            if lip is not None:
                try:
                    meta["reconstruction_lipschitz_bound"] = float(lip)
                except (TypeError, ValueError):
                    pass
        # Token-distance metadata: persist a per-(session, eps) aggregate of
        # ``fraction_changed`` (mean/std over windows x replicas), not the raw
        # per-record list. Only populated when the corruption produced
        # code-level distance records (discrete tokenizers / codes target).
        token_distance_summary = NTTEvalRobustnessLatentJob._aggregate_token_distance(
            token_distance_records
        )
        if token_distance_summary:
            meta["token_distance"] = token_distance_summary
        with path.open("w") as f:
            json.dump(meta, f, indent=2)
        return meta

    # ── Plotting ────────────────────────────────────────────────────

    # ── MLflow logging ──────────────────────────────────────────────

    def _log_to_mlflow(
        self,
        *,
        output_dir: Path,
        rl_cfg: T.Any,
        aggregated_report: EvaluationReport | None = None,
        per_session_reports: dict[str, EvaluationReport] | None = None,
        extra_metrics: dict[str, float] | None = None,
    ) -> None:
        """Log this latent_robustness (M2) run to MLflow.

        Reads ``results.jsonl`` and emits a step-indexed sweep curve per
        (session, metric) keyed by epsilon. Uploads only compact artifacts by
        default; raw replotting sidecars remain local unless explicitly
        requested via ``mlflow_logging.upload_artifacts_for_replotting``.
        The run is tagged with
        ``ntt.eval.checkpoint_path`` so it links back to the main pass; the
        checkpoint bytes are not re-uploaded.
        """
        from ntt.evaluation.mlflow_logging import (
            log_robustness_run_to_mlflow,
        )

        jsonl_path = output_dir / "results.jsonl"
        if not jsonl_path.exists():
            log.warning(
                f"latent_robustness (M2) MLflow logging: {jsonl_path} not found; skipping."
            )
            return

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
            replot_csv = output_dir / "results_per_session.csv"
            if replot_csv.exists():
                artifacts.append(replot_csv)

        try:
            import tempfile

            from ntt.core.model.checkpoint_io import resolve_checkpoint_for_eval

            with tempfile.TemporaryDirectory(prefix="ntt_m2_ckpt_") as tmpdir:
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
                    mode="m2",
                    config=self.mlflow_logging,
                    jsonl_path=jsonl_path,
                    report=aggregated_report,
                    per_session_reports=per_session_reports,
                    checkpoint_path=(str(self.ckpt_path) if self.ckpt_path else None),
                    checkpoint_local_path=ckpt_local_path,
                    artifact_paths=artifacts,
                    artifact_root=output_dir,
                    include_jsonl_artifact=include_jsonl_artifact,
                    extra_tags={"ntt.eval.split": self.split},
                    extra_metrics=extra_metrics,
                    run_metadata=json.loads((output_dir / "run_meta.json").read_text()),
                )
            log.info(f"latent_robustness (M2) results logged to MLflow run: {run_id}")
        except Exception:
            log.warning("Failed to log latent_robustness (M2) results to MLflow", exc_info=True)

    def _emit_figures(
        self,
        *,
        output_dir: Path,
        rl_cfg: T.Any,
        bin_size_seconds: float,
        captured_rasters: dict[str, dict[int, dict[str, T.Any]]],
    ) -> None:
        """Emit M2-a (always) and M2-b rasters (when ``plot_examples``).

        Each plotting block is wrapped in try/except so a single failure
        does not abort the eval — the failure is logged with the offending
        path/window plus the traceback.
        """
        import matplotlib.pyplot as plt

        # ---- aggregate curve: latent-corruption-a ------------------
        try:
            csv_path = output_dir / "results.csv"
            if csv_path.exists() and csv_path.stat().st_size > 0:
                fig = plot_latent_corruption_a(csv_path)
                save_fig_multi(fig, output_dir / "curves" / "latent-corruption-a")
                plt.close(fig)
            else:
                log.warning(
                    f"latent-corruption-a: results.csv missing/empty at"
                    f" {csv_path}; skipping aggregate curve."
                )
        except Exception:
            log.warning(f"latent-corruption-a plotting failed:\n{traceback.format_exc()}")

        # ---- rasters: latent-corruption-b (eval-time only) ---------
        if not rl_cfg.plot_examples:
            return

        rasters_dir = output_dir / "rasters"
        eps_per_panel = list(rl_cfg.epsilons)
        for sid, by_seg_idx in captured_rasters.items():
            for w_idx, payload in by_seg_idx.items():
                base_path = rasters_dir / f"latent-corruption-b_{_safe_filename(sid)}_w{w_idx}"
                try:
                    clean = payload.get("clean")
                    noisy_per_eps = payload.get("noisy_per_eps") or {}
                    metric_per_eps = payload.get("metric_per_eps") or {}
                    if clean is None or not noisy_per_eps:
                        log.warning(
                            f"latent-corruption-b: missing capture for"
                            f" session={sid} window={w_idx}; skipping {base_path}."
                        )
                        continue
                    fig = plot_latent_corruption_b(
                        window_idx=w_idx,
                        session_id=sid,
                        clean_spikes=clean,
                        noisy_spikes_per_eps=noisy_per_eps,
                        eps_per_panel=eps_per_panel,
                        bin_size_ms=float(bin_size_seconds * 1000.0),
                        metric_per_eps=metric_per_eps or None,
                        output_stem=base_path,
                    )
                    plt.close(fig)
                except Exception:
                    log.warning(
                        f"latent-corruption-b plotting failed at {base_path}\n"
                        f"{traceback.format_exc()}"
                    )


def _units_time_to_TU(arr) -> np.ndarray:
    """Convert a tensor in ``(units, time)`` (or batched) layout to a
    numpy ``[T, U]`` raster array.

    ``plot_spike_raster`` expects ``[T, U]`` (time, units). The latent_robustness (M2) job's
    ``clean_recon_t`` and ``noisy_recon_t_2d`` use the
    ``EvalData.from_tensor`` ``(units, time)`` convention, so we
    transpose.
    """
    if isinstance(arr, torch.Tensor):
        a = arr.detach().cpu().numpy()
    else:
        a = np.asarray(arr)
    if a.ndim == 3:
        a = a[0]
    if a.ndim != 2:
        raise ValueError(f"_units_time_to_TU: expected 2-D after squeeze, got {a.shape}")
    # Heuristic: if the array looks like (U, T) where U < T, transpose;
    # else assume already (T, U). The latent_robustness (M2) path always feeds (U, T) here so
    # this transpose is the right answer.
    return a.T


def _safe_filename(name: str) -> str:
    """Sanitize ``name`` for use as a filesystem path component."""
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name)
