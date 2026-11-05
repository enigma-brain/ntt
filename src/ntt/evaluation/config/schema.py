"""Pydantic configuration schemas for evaluation."""

from enum import StrEnum
from itertools import product
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class CorruptionParams(BaseModel):
    """Parameters for a single input-level corruption spec (M1 replica fan-out).

    A union-style schema covering jitter / drop / add / drop_and_add /
    window_shift / drop_neurons / permute_neurons. Fields not relevant to a
    given ``type`` stay ``None``. ``deterministic_per_replica`` only makes
    sense for unit-axis corruptions (drop_neurons, permute_neurons); it
    causes a per-replica-only seed so the same mask/permutation applies to
    every window for a given replica (needed for ACG/CCG averaging).
    ``name`` is an optional human-readable label overriding the auto-label
    used for aggregation in input_robustness (M1).
    """

    type: Literal[
        "jitter",
        "drop",
        "add",
        "drop_and_add",
        "window_shift",
        "drop_neurons",
        "permute_neurons",
    ]
    # union of optional param fields; unused ones default None
    k_ms: float | None = None
    p: float | None = None
    fraction: float | None = None
    max_shift_ms: float | None = None
    analysis_window_ms: float | None = None
    source_window_ms: float | None = None
    deterministic_per_replica: bool = False
    name: str | None = None


_CORRUPTION_SWEEPABLE_FIELDS: tuple[str, ...] = (
    "k_ms",
    "p",
    "fraction",
    "max_shift_ms",
    "analysis_window_ms",
    "source_window_ms",
)


def _expand_corruption_entry(entry: Any) -> list[Any]:
    """Expand one raw corruption entry with list-valued sweep fields.

    Example:
      ``{type: window_shift, max_shift_ms: [25, 5, 1, 0]}``
    becomes four scalar dicts before nested ``CorruptionParams`` validation.
    When multiple sweep fields are lists, we take the Cartesian product.
    """
    if isinstance(entry, CorruptionParams):
        return [entry]
    if not isinstance(entry, dict):
        return [entry]

    sweep_fields = [
        field_name
        for field_name in _CORRUPTION_SWEEPABLE_FIELDS
        if isinstance(entry.get(field_name), list)
    ]
    if not sweep_fields:
        return [entry]

    sweep_values: list[list[Any]] = []
    for field_name in sweep_fields:
        values = entry[field_name]
        if len(values) == 0:
            raise ValueError(f"corruption field '{field_name}' may not be an empty list")
        sweep_values.append(list(values))

    expanded: list[dict[str, Any]] = []
    for combo in product(*sweep_values):
        item = dict(entry)
        for field_name, value in zip(sweep_fields, combo, strict=True):
            item[field_name] = value
        expanded.append(item)
    return expanded


DEFAULT_M1_ANALYSIS_WINDOW_MS = 1000.0


def resolve_window_shift_windows(
    *,
    max_shift_ms: float,
    analysis_window_ms: float | None = None,
    source_window_ms: float | None = None,
) -> tuple[float, float]:
    """Resolve the effective analysis/source windows for input_robustness (M1) window shift."""
    analysis_ms = (
        float(DEFAULT_M1_ANALYSIS_WINDOW_MS)
        if analysis_window_ms is None
        else float(analysis_window_ms)
    )
    source_ms = (
        analysis_ms + 2.0 * abs(float(max_shift_ms))
        if source_window_ms is None
        else float(source_window_ms)
    )
    return analysis_ms, source_ms


def default_m1_corruptions_for_tests() -> list[CorruptionParams]:
    """Return the default list of input_robustness (M1) corruptions — test/replay use only.

    Production input_robustness (M1) runs should configure corruptions explicitly via the
    eval-config yaml (see ``config/evaluation/metrics/robustness_input.yaml``)
    rather than rely on this fallback. The function is the
    canonical sweep used by:
      - ``tests/jobs/test_eval_robustness_input.py`` (sanity-check the
        default sweep covers each corruption type),
      - the ``RobustnessInputConfig.corruptions`` ``default_factory``,
        which keeps ``RobustnessInputConfig()`` instantiable but is not
        the recommended path for production runs.

    Covers: window_shift (3 non-fixed specs), jitter (4 (k_ms, p) combos),
    drop (3 fractions), add (3 fractions), drop_neurons (2 fractions,
    fixed-mode), permute_neurons (single, fixed-mode). Plus zero-strength
    specs for each corruption, as sanity check and to get zero points in
    the plots.
    """
    specs: list[CorruptionParams] = []

    # window_shift: max_shift_ms in [25, 5, 1]. The tokenizer should
    # always see an analysis window of the standard training-time
    # length (1000 ms). To allow a ±max_shift_ms shift on top of that,
    # the source window the dataloader provides must be longer:
    # ``source = analysis + 2*max_shift_ms``. The input_robustness (M1) job rebuilds the
    # val sampler with the appropriate source-length per WindowShift
    # spec, so each tokenizer (CS, LDNS, wan-A i.e. conv-*) always receives its
    # training-time T at the encoder while the analysis sub-window
    # walks across the source.
    for max_shift_ms in (25.0, 5.0, 1.0, 0.0):
        analysis_ms, source_ms = resolve_window_shift_windows(
            max_shift_ms=max_shift_ms,
        )
        specs.append(
            CorruptionParams(
                type="window_shift",
                max_shift_ms=max_shift_ms,
                analysis_window_ms=analysis_ms,
                source_window_ms=source_ms,
            )
        )

    # jitter: (k_ms, p) in (1, 0.5), (3, 0.5), (10, 0.5), (30, 0.5)
    for k_ms in (0.0, 1.0, 3.0, 10.0, 30.0):
        specs.append(CorruptionParams(type="jitter", k_ms=k_ms, p=0.5))

    # We skip drop and add. We only use drop+add.
    # # drop: fraction in [0.05, 0.2, 0.5]
    # for fraction in (0.05, 0.2, 0.5):
    #     specs.append(CorruptionParams(type="drop", fraction=fraction))

    # # add: fraction in [0.05, 0.2, 0.5]
    # for fraction in (0.05, 0.2, 0.5):
    #     specs.append(CorruptionParams(type="add", fraction=fraction))

    # drop_and_add: fraction in [0.05, 0.1, 0.2, 0.4]. Combined corruption
    # that drops `fraction` of spikes then adds back the same fraction at
    # random unit/time positions — exposes whether the tokenizer is
    # sensitive to mass-rebalancing vs. raw spike count changes alone.
    for fraction in (0.0, 0.05, 0.15, 0.5):
        specs.append(CorruptionParams(type="drop_and_add", fraction=fraction))

    # drop_neurons: fraction in [0.0, 0.05, 0.11, 0.25], fixed-mode
    # (deterministic_per_replica, so that we can concatenate windows
    # per replica ID and have the same neurons aligned).
    # 0.05 is a no-op for tokenizers with
    # group_size=10 (CS, LDNS via NeuronCoveringWrapper) — the eval
    # job's per-spec try/except logs and skips with status=skipped on
    # those, while large-group tokenizers (wan-A vae sees full session
    # n_units) get the low-strength data point. 0.11 is the smallest
    # fraction that always corrupts at the smallest plausible group
    # (1.1 → 1 dropped); 0.25 is a strong-corruption anchor.
    for fraction in (0.0, 0.05, 0.11, 0.25):
        specs.append(
            CorruptionParams(
                type="drop_neurons",
                fraction=fraction,
                deterministic_per_replica=True,
            )
        )

    # permute_neurons: single entry, fixed-mode
    specs.append(CorruptionParams(type="permute_neurons", deterministic_per_replica=True))

    return specs


class RobustnessInputConfig(BaseModel):
    """Top-level config for input_robustness (M1).

    ``n_replicas`` is the number of corrupted replicas per window
    (plus one clean replica, so the replica axis has length ``K+1``).
    ``corruptions`` defines the sweep; default comes from
    ``default_m1_corruptions_for_tests()``. Session-duration capping for input_robustness (M1) uses
    the standard top-level ``EvaluationConfig.max_eval_seconds`` knob.
    """

    n_replicas: int = 10
    corruptions: list[CorruptionParams] = Field(default_factory=default_m1_corruptions_for_tests)
    plot_examples: bool = True

    @model_validator(mode="before")
    @classmethod
    def _expand_list_sweeps(cls, data: Any) -> Any:
        """Allow compact list syntax inside ``corruptions`` entries."""
        if not isinstance(data, dict):
            return data
        corruptions = data.get("corruptions")
        if not isinstance(corruptions, list):
            return data

        expanded: list[Any] = []
        for entry in corruptions:
            expanded.extend(_expand_corruption_entry(entry))

        updated = dict(data)
        updated["corruptions"] = expanded
        return updated


class RobustnessLatentConfig(BaseModel):
    """Top-level config for latent_robustness (M2).

    ``epsilons`` is the noise scale sweep (units of sigma_joint).
    ``n_samples_per_epsilon`` is the number of noise draws per epsilon.
    ``noise_distribution`` picks gaussian or uniform noise.
    ``n_eval_windows`` caps the number of first contiguous windows processed
    per session; ``None`` = all.
    """

    epsilons: list[float] = [0.01, 0.03, 0.1, 0.3, 1.0]
    n_samples_per_epsilon: int = 10
    noise_distribution: Literal["gaussian", "uniform"] = "gaussian"
    n_eval_windows: int | None = 64
    max_decode_batch_size: int | None = None
    corruption_target: Literal["latents", "codes"] = "latents"
    plot_examples: bool = True


class RepeatsBaselineConfig(BaseModel):
    """Config for the repeats reconstruction baseline metric.

    ``skip_metrics`` lists registered metric names that the repeats
    baseline should skip (e.g. ACG/CCG which are ill-defined on pairs
    of short stimulus windows).
    """

    skip_metrics: list[str] = ["acg_correlation", "ccg_correlation"]


class AggregationStrategy(StrEnum):
    """Strategy for aggregating metrics across sessions.

    When running full session-collated evaluation, metrics are computed
    per-session and then aggregated. This enum controls how that aggregation
    is performed.
    """

    MEAN = "mean"
    MEDIAN = "median"
    WEIGHTED_MEAN = "weighted_mean"  # Weight by session duration or unit count


class EvaluationConfig(BaseModel):
    """Pydantic configuration for evaluation pipeline.

    Metric parameters (e.g. spike_match_tolerance_ms, bin_sizes_ms) live on
    each metric class via ``_default_params`` and can be overridden per-metric
    in the ``metrics`` dict.  This config holds only pipeline infrastructure.
    """

    # Data type handling
    allow_lossy_conversion: bool = Field(
        default=True, description="Whether to allow lossy data type conversions"
    )
    conversion_bin_size_ms: float = Field(
        default=1.0, description="Bin size for data type conversions"
    )
    allow_stochastic_conversion: bool = Field(
        default=False,
        description=(
            "Allow stochastic data conversions (e.g., Poisson sampling from RATES "
            "to SPIKE_TIMES). Disabled by default for deterministic evaluation."
        ),
    )
    stochastic_conversion_seed: int | None = Field(
        default=42,
        description="Random seed for stochastic conversions (reproducibility).",
    )

    # Validation collation / segmenting
    max_windows: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Flush a validation segment after accumulating this many validation windows. "
            "None disables window-count based flushing, and will aggregate all validation windows."
        ),
    )
    max_bytes: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Flush a validation segment after accumulating approximately this many bytes "
            "in the collation buffers. None disables byte-size based flushing. "
            "This can act as a soft limit to prevent memory bloat."
        ),
    )
    interval_source: str | None = Field(
        default=None,
        description=(
            "Optional sparse interval source name for the per-stimulus SNR path. "
            "When set, repeats-SNR uses PerStimulusSampler-driven stimulus windows."
        ),
    )
    interval_start_lag_ms: float = Field(
        default=0.0,
        description=(
            "Signed offset from each stimulus start, in milliseconds, for the "
            "PerStimulusSampler SNR path."
        ),
    )
    interval_end_lag_ms: float | None = Field(
        default=None,
        description=(
            "Signed offset from each stimulus end, in milliseconds, for the "
            "PerStimulusSampler SNR path. Mutually exclusive with "
            "`fixed_window_duration_ms`: exactly one of the two must be set."
        ),
    )
    fixed_window_duration_ms: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "If set, the PerStimulusSampler uses a fixed-length window of "
            "`[stim_start + interval_start_lag_ms, stim_start + interval_start_lag_ms "
            "+ fixed_window_duration_ms]` for each stimulus, regardless of "
            "stimulus duration. Overlap with the next stimulus must be at most "
            "`cross_stim_padding_ms`. Mutually exclusive with "
            "`interval_end_lag_ms`."
        ),
    )
    cross_stim_padding_ms: float = Field(
        default=20.0,
        ge=0.0,
        description=(
            "Additional real-data padding, in milliseconds, that each "
            "PerStimulusSampler window may borrow past the next stimulus onset "
            "before zero-padding fallback is applied."
        ),
    )
    label_extension_ms: float | None = Field(
        default=None,
        description=(
            "Optional post-stimulus label extension, in milliseconds, for the "
            "PerStimulusSampler SNR path. None reuses interval_end_lag_ms."
        ),
    )
    max_eval_seconds: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Maximum duration (in seconds) of data to evaluate per session. "
            "Once reached, remaining windows for that session are skipped. "
            "None = evaluate the full session."
        ),
    )
    max_neurons: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Maximum number of neurons to include per session. "
            "For the all-neuron eval path (evaluate_all_neurons=True), this is "
            "applied at data-loading time (select_eval_units: a random subset seeded "
            "by neuron_subsample_seed and the session id, the same for every model "
            "and Poisson baseline), bounding both tokenizer memory and metric cost. "
            "In padded mode, neurons are subsampled the same way at "
            "metric-computation time. "
            "Sessions with fewer neurons are unaffected. None = all neurons."
        ),
    )
    neuron_subsample_seed: int = Field(
        default=42,
        description="Random seed for neuron subsampling when max_neurons is set.",
    )

    eval_window_seconds: float | Literal["native"] = Field(
        default=1.0,
        description=(
            "Authoritative analysis/eval window in seconds, or 'native' = the tokenizer "
            "spec's own window_length_seconds (an error for a distribution-valued window). "
            "Fed to the datamodule "
            "(batch length) and the collation window; tokenizers must accept it (see "
            "assert_tokenizer_accepts_eval_window) or the eval fails loud. The window "
            "actually scored is saved as scored_window_seconds. The input_robustness (M1) "
            "window_shift analysis window defaults to this window."
        ),
    )

    @field_validator("eval_window_seconds")
    @classmethod
    def _eval_window_positive(cls, value: float | str) -> float | str:
        if value != "native" and not (0 < float(value) < float("inf")):  # also rejects NaN
            raise ValueError("eval_window_seconds must be a finite number > 0 or 'native'")
        return value

    # Metric selection
    metrics: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Dict of metric configurations keyed by display name. "
            "Each value must have a 'metric_class' key referencing a registered "
            "metric name, plus optional parameter overrides. "
            "None = run all registered metrics with default parameters."
        ),
    )

    compute_baseline: bool = Field(
        default=False, description="Whether to compute baseline metrics during evaluation."
    )

    # Robustness (M1 / latent_robustness (M2) / repeats baseline) sections.
    # These are top-level config sections; metrics input_robustness (M1) / repeats_baseline are
    # registered by name. latent_robustness (M2) is NOT a metric entry — it is consumed by
    # the separate `eval_robustness_latent.py` job.
    robustness_input: RobustnessInputConfig | None = None
    robustness_latent: RobustnessLatentConfig | None = None
    repeats_baseline: RepeatsBaselineConfig | None = None
    latent_stats_on_disk: bool = False

    class Config:
        extra = "forbid"


class LightningEvalConfig(BaseModel):
    """Configuration for Lightning integration.

    This config controls when and how evaluation is run during training:

    Quick Evaluation:
        - Runs every `eval_every_n_batches` training batches or at epoch end
        - Uses within-batch collation to aggregate items from same session
        - At epoch end, accumulates `quick_eval_n_batches` batches for longer segments
        - Only computes fast metrics (spike_detection, single_unit by default)
        - Useful for monitoring during training

    Full Session-Collated Evaluation:
        - Aggregates validation windows into longer per-session segments
        - Runs all metrics including slower pairwise/population metrics
        - Can be triggered every N epochs or only at training end
        - Session count can be limited for speed
    """

    # Quick evaluation during training (per-batch)
    eval_every_n_batches: int = Field(
        default=999999,
        ge=1,
        description=(
            "Run quick evaluation every N training batches. "
            "Set to a very high number to effectively disable."
        ),
    )
    eval_on_epoch_end: bool = Field(
        default=False,
        description="Run evaluation at end of each validation epoch.",
    )
    quick_eval_n_batches: int = Field(
        default=3,
        ge=1,
        le=100,
        description=(
            "Number of batches to accumulate for epoch-end evaluation. "
            "More batches = longer segments = more reliable metrics, but slower."
        ),
    )
    quick_metrics_only: bool = Field(
        default=True,
        description="During quick (per-batch) evaluation, only compute fast metrics.",
    )
    quick_metrics: list[str] | None = Field(
        default=None,
        description=(
            "Display names of metrics to include in quick evaluation. "
            "Must match keys in EvaluationConfig.metrics. "
            "None = use all pipeline metrics."
        ),
    )

    # Full session-collated evaluation
    full_eval_every_n_epochs: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Run full session-collated evaluation every N epochs. "
            "None = disabled during training (only at end if full_eval_at_training_end=True)."
        ),
    )
    full_eval_at_training_end: bool = Field(
        default=True,
        description="Run full session-collated evaluation at end of training.",
    )
    max_sessions_for_full_eval: int | None = Field(
        default=5,
        ge=1,
        description=(
            "Maximum number of sessions to evaluate during full evaluation. "
            "None = evaluate all sessions (can be slow)."
        ),
    )

    # Aggregation across sessions
    session_aggregation_strategy: AggregationStrategy = Field(
        default=AggregationStrategy.MEAN,
        description=(
            "Strategy for aggregating metrics across sessions. "
            "'mean': simple mean, 'median': median, 'weighted_mean': weighted by duration/units."
        ),
    )
    log_per_session_metrics: bool = Field(
        default=False,
        description="Log individual session metrics in addition to aggregated metrics.",
    )

    # Batching / subsampling
    max_units_for_pairwise: int = Field(
        default=50,
        description="Maximum units before subsampling for pairwise metrics.",
    )

    # Poisson baseline comparison
    compare_to_poisson: bool = Field(
        default=False,
        description=(
            "Run a PoissonBaselineTokenizer on the same data and log "
            "normalized comparison scores using Poisson as floor and "
            "perfect reconstruction as ceiling."
        ),
    )
    poisson_rate_estimation_bin_ms: float = Field(
        default=10.0,
        description="Rate estimation bin size (ms) for the Poisson baseline tokenizer.",
    )

    class Config:
        extra = "forbid"
