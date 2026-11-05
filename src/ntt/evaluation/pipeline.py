"""Main evaluation pipeline for spike train tokenizers."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)

from .aggregation.comparison import MetricComparison, compute_normalized_comparisons
from .aggregation.summary import create_report
from .config import DEFAULT_CONFIG, EvaluationConfig, instantiate_metrics_from_config
from .core.converters import DataConverter
from .core.intermediates import IntermediatesCache
from .core.types import DataType, EvalData, EvaluationReport, MetricCategory, MetricResult
from .metrics.base import (
    BaseMetric,
    get_all_metrics,
)


def _store_result(
    results: dict[str, MetricResult],
    key: str,
    value: MetricResult,
    *,
    expanded_from: str | None = None,
) -> None:
    """Write one metric result, refusing to overwrite an existing key.

    Result keys have to be unique: a silent overwrite means one metric's number is
    reported under another metric's name, which is indistinguishable from a correct
    run everywhere downstream. Two ways a collision can arise:

    * a display name declared twice -- impossible from config (they are mapping
      keys) but possible when a caller passes explicit ``(display_name, metric)``
      pairs;
    * multi-scale expansion, which synthesises ``{display_name}_{n}ms`` names that
      can collide with a display name the config declares directly. The
      ``config/evaluation/temporal/*.yaml`` presets null out their default-bin
      entries by hand for precisely this reason ("Null out default-bin metrics to
      avoid name collisions") -- an unenforced convention that this makes an
      invariant.

    Raises rather than recording a skip: a collision is a *configuration* defect
    that affects every window identically, not a per-window computation failure,
    so it must not be downgraded into ``skipped_metrics`` alongside compute errors.
    """
    if key in results:
        origin = (
            f"the multi-scale expansion of {expanded_from!r}"
            if expanded_from
            else "a metric display name"
        )
        raise ValueError(
            f"Duplicate metric result key {key!r}, from {origin}. Two metrics would "
            f"report under one name and the later would silently overwrite the "
            f"earlier. Fix the eval config: rename the colliding display name, or "
            f"null out the default-bin entry that expands into it."
        )
    results[key] = value


class EvaluationPipeline:
    """
    Main evaluation pipeline for spike train tokenizers.

    Handles data type conversion automatically based on metric requirements.

    Usage:
        pipeline = EvaluationPipeline()
        report = pipeline.run(original_spikes, reconstructed_spikes)
    """

    def __init__(
        self,
        metrics: list[BaseMetric] | list[tuple[str, BaseMetric]] | None = None,
        config: EvaluationConfig = DEFAULT_CONFIG,
    ):
        """
        Initialize the evaluation pipeline.

        Args:
            metrics: Metrics to compute. Accepts either:
                - list[BaseMetric]: display names default to metric.name
                - list[tuple[str, BaseMetric]]: (display_name, metric_instance) pairs
                If None, uses config to determine metrics.
            config: Evaluation configuration.
        """
        self.config = config
        self.converter = DataConverter(
            default_bin_size_seconds=self.config.conversion_bin_size_ms / 1000.0,
            allow_stochastic_conversion=self.config.allow_stochastic_conversion,
            stochastic_seed=self.config.stochastic_conversion_seed,
        )

        # Determine which metrics to use -- stored as (display_name, metric) tuples
        self.metrics: list[tuple[str, BaseMetric]]
        if metrics is not None:
            if metrics and isinstance(metrics[0], tuple):
                self.metrics = cast("list[tuple[str, BaseMetric]]", metrics)
            else:
                self.metrics = [(m.name, m) for m in cast("list[BaseMetric]", metrics)]
        else:
            self.metrics = self._get_configured_metrics()

    def _get_configured_metrics(self) -> list[tuple[str, BaseMetric]]:
        """Get metrics based on configuration.

        Returns (display_name, metric_instance) tuples. When ``config.metrics``
        is set, display names come from the dict keys and metrics are
        instantiated with per-metric parameters. When ``config.metrics`` is
        None, all registered metrics are used with default parameters.
        """
        if self.config.metrics is not None:
            return instantiate_metrics_from_config(self.config.metrics)
        return [(m.name, m) for m in get_all_metrics()]

    def _can_convert_pair(
        self,
        original: EvalData,
        reconstructed: EvalData | None,
        target_type: DataType,
    ) -> bool:
        """Check if both datasets can be converted to target type.

        ``reconstructed`` may be ``None`` for tokenize-only tokenizers; in
        that case only the original side is checked.
        """
        self._require_data_type(original)
        if not self.converter.can_convert_data(original, target_type):
            return False
        if reconstructed is None:
            return True
        self._require_data_type(reconstructed)
        return self.converter.can_convert_data(reconstructed, target_type)

    def _select_target_type(
        self,
        original: EvalData,
        reconstructed: EvalData | None,
        metric: BaseMetric,
    ) -> DataType:
        """Select the best target data type for a metric.

        ``reconstructed`` may be ``None`` for tokenize-only tokenizers; in
        that case selection considers only the original side and the
        metric's preferred / supported types.
        """
        original_type = self._require_data_type(original)
        reconstructed_type = (
            self._require_data_type(reconstructed) if reconstructed is not None else None
        )
        candidates: list[DataType] = []
        candidate_pool: list[DataType] = [metric.preferred_data_type, original_type]
        if reconstructed_type is not None:
            candidate_pool.append(reconstructed_type)
        candidate_pool.extend(sorted(metric.supported_data_types, key=lambda dt: dt.value))
        for candidate in candidate_pool:
            if candidate not in candidates:
                candidates.append(candidate)

        for candidate in candidates:
            if candidate not in metric.supported_data_types:
                continue
            if self._can_convert_pair(original, reconstructed, candidate):
                return candidate

        if (
            reconstructed_type is not None
            and original_type == reconstructed_type
            and original_type in metric.supported_data_types
        ):
            return original_type
        if reconstructed_type is None and original_type in metric.supported_data_types:
            return original_type

        side_str = (
            f"{original_type.value}/{reconstructed_type.value}"
            if reconstructed_type is not None
            else f"{original_type.value}/None"
        )
        raise ValueError(
            f"Metric {metric.name} cannot convert from {side_str} to any supported type."
        )

    def _get_target_bin_size(
        self,
        original: EvalData,
        reconstructed: EvalData | None,
        target_type: DataType,
    ) -> float | None:
        """Pick a bin size when converting to a binned representation.

        If either of the original or reconstructed data has target_type,
        return their bin size if available.
        Else return None.

        ``reconstructed`` may be ``None`` for tokenize-only tokenizers.
        """
        original_type = self._require_data_type(original)
        if target_type == original_type and original.bin_size_seconds is not None:
            return original.bin_size_seconds
        if reconstructed is not None:
            reconstructed_type = self._require_data_type(reconstructed)
            if target_type == reconstructed_type and reconstructed.bin_size_seconds is not None:
                return reconstructed.bin_size_seconds
        return None

    def _require_data_type(self, data: EvalData) -> DataType:
        """Ensure EvalData has a data type."""
        if data.data_type is None:
            raise ValueError("EvalData.data_type must be set for evaluation.")
        return data.data_type

    def run(
        self,
        original: EvalData,
        reconstructed: EvalData | None,
        compute_baseline: bool | None = None,
        baseline: EvalData | None = None,
        poisson_seed: int | None = None,
        latents: np.ndarray | list[np.ndarray] | None = None,
        behavioral_data_token_res: dict[str, np.ndarray] | None = None,
        behavioral_data_input_res: dict[str, np.ndarray] | None = None,
        token_timestamps: np.ndarray | None = None,
        token_start_times: np.ndarray | None = None,
        token_end_times: np.ndarray | None = None,
        sampling_rate: float | None = None,
        segment_absolute_start: float | None = None,
        token_bin_size_ms: float | None = None,
        session_id: str | None = None,
        replica_latents: "np.ndarray | list[np.ndarray] | None" = None,
        replica_reconstructions: "np.ndarray | list[np.ndarray] | None" = None,
        replica_originals: "np.ndarray | list[np.ndarray] | None" = None,
        replica_metadata: dict | None = None,
    ) -> EvaluationReport:
        """
        Run evaluation comparing reconstructed to original spike trains.

        Args:
            original: Original spike data
            reconstructed: Reconstructed spike data
            compute_baseline: Whether to compute Poisson baseline (default from config)
            baseline: Pre-computed baseline (skips generation if provided)
            poisson_seed: Random seed for Poisson baseline (default from config)
            latents: Optional latent representations from tokenizer,
                     a list of per-window arrays. each: (n_samples, latent_dim)
            behavioral_data_token_res: Categorical behavioral fields at token resolution
                (for phase/trial filtering in downstream metrics)
            behavioral_data_input_res: All behavioral fields at input resolution
                (for resolution-independent regression/classification targets)
            token_timestamps: Center timestamps for each token (n_samples,).
                Required when latents are present.
            token_start_times: Start times for each token (n_samples,).
                Required when latents are present.
            token_end_times: End times for each token (n_samples,).
                Required when latents are present.
            sampling_rate: Input sampling rate in Hz (e.g. 1000 for 1ms bins).
            segment_absolute_start: Absolute start time of the segment (seconds).
            token_bin_size_ms: Temporal extent of each token in ms.
            session_id: Optional session identifier (e.g.
                "odoherty_sabes_nonhuman_2017/indy_20160407_02").  When
                provided, metrics with ``limit_to_datasets`` set are skipped
                if the session_id doesn't match.

        Returns:
            EvaluationReport with all metrics and category scores

        """
        t_pipeline_start = time.perf_counter()
        results = {}
        skipped_metrics = {}

        # Group metrics by their computed target data type to minimize conversions
        metrics_by_type: dict[DataType, list[tuple[str, BaseMetric]]] = {}
        for display_name, metric in self.metrics:
            try:
                target_type = self._select_target_type(original, reconstructed, metric)
            except ValueError as exc:
                skipped_metrics[display_name] = str(exc)
                continue
            metrics_by_type.setdefault(target_type, []).append((display_name, metric))

        # Token-resolution behavioral data is used for filtering (phase, trial).
        # Input-resolution behavioral data is used for regression/classification targets.

        # Process each group
        for target_type, type_metrics in metrics_by_type.items():
            t_group_start = time.perf_counter()
            metric_names_str = ", ".join(dn for dn, _ in type_metrics)
            logger.debug(
                "Processing %d metrics for %s: [%s]",
                len(type_metrics),
                target_type.value,
                metric_names_str,
            )

            # Convert data once per target type.
            # Convert original and reconstructed independently so that a
            # failure on one side (e.g. dense reconstruction failing the
            # SPIKE_TIMES density check) doesn't block metrics that only
            # need the other side (e.g. behavioral metrics use latents).
            t_conv_start = time.perf_counter()
            try:
                target_bin_size = self._get_target_bin_size(original, reconstructed, target_type)
            except ValueError as e:
                for display_name, _metric in type_metrics:
                    skipped_metrics[display_name] = str(e)
                continue

            try:
                conv_original: EvalData = self.converter.convert(
                    original, target_type, target_bin_size
                )
            except ValueError as e:
                for display_name, _metric in type_metrics:
                    skipped_metrics[display_name] = f"Original conversion error: {e}"
                continue

            conv_reconstructed: EvalData | None
            if reconstructed is None:
                # Tokenize-only mode — no reconstruction tensor exists.
                # IntermediatesCache accepts ``None`` and any metric whose
                # intermediate request reaches ``_require_reconstructed``
                # is caught + skipped by the per-metric ``try/except``
                # block below.
                conv_reconstructed = None
            else:
                try:
                    conv_reconstructed = self.converter.convert(
                        reconstructed,
                        target_type,
                        target_bin_size,
                    )
                except ValueError as e:
                    # Reconstructed conversion failed (e.g. too dense for
                    # SPIKE_TIMES).  Fall back to the original target type so
                    # metrics that don't need reconstructed spike_times (like
                    # behavioral decoding from latents) can still run.
                    logger.warning(
                        "Reconstructed data conversion to %s failed: %s. "
                        "Metrics needing reconstructed data in this format "
                        "will be skipped.",
                        target_type.value,
                        e,
                    )
                    conv_reconstructed = reconstructed

            t_conv_elapsed = time.perf_counter() - t_conv_start
            if t_conv_elapsed > 1.0:
                logger.debug(
                    "Data conversion to %s took %.1fs",
                    target_type.value,
                    t_conv_elapsed,
                )

            # Build cache for this data type

            cache = IntermediatesCache(
                conv_original,
                conv_reconstructed,
                latents=latents,
                behavioral_data_token_res=behavioral_data_token_res,
                behavioral_data_input_res=behavioral_data_input_res,
                token_timestamps=token_timestamps,
                sampling_rate=sampling_rate,
                segment_absolute_start=segment_absolute_start,
                token_bin_size_ms=token_bin_size_ms,
                replica_latents=replica_latents,
                replica_reconstructions=replica_reconstructions,
                replica_originals=replica_originals,
                replica_metadata=replica_metadata,
            )

            # Validate compatibility and fetch intermediates for all metrics.
            # Separate into sequential (run now) and parallelizable (run in
            # a thread pool).  Downstream prediction metrics are CPU-heavy
            # sklearn fits that release the GIL during BLAS calls, so they
            # benefit from concurrent execution.
            ready: list[tuple[str, BaseMetric, dict[str, Any]]] = []
            parallel: list[tuple[str, BaseMetric, dict[str, Any]]] = []

            for display_name, metric in type_metrics:
                # Skip metrics restricted to other datasets
                if session_id is not None and not metric.should_run_for_session(session_id):
                    skipped_metrics[display_name] = (
                        f"Skipped: limit_to_datasets={metric.limit_to_datasets} "
                        f"does not match session '{session_id}'"
                    )
                    continue

                orig_ok, orig_reason = metric.is_compatible(
                    self._require_data_type(original),
                    original.bin_size_seconds,
                    allow_conversion=self.config.allow_lossy_conversion,
                    data=original,
                )
                if reconstructed is None:
                    # Tokenize-only mode: the recon-side compatibility check
                    # is meaningless. Metrics that actually need the recon
                    # side will be filtered when their intermediates
                    # ``_require_reconstructed`` raises (caught below).
                    recon_ok, recon_reason = True, None
                else:
                    recon_ok, recon_reason = metric.is_compatible(
                        self._require_data_type(reconstructed),
                        reconstructed.bin_size_seconds,
                        # Alignment of original and reconstructed data is assumed here.
                        window_length_seconds=(original.duration or reconstructed.duration),
                        allow_conversion=self.config.allow_lossy_conversion,
                        data=reconstructed,
                    )
                if not orig_ok or not recon_ok:
                    skipped_metrics[display_name] = orig_reason or recon_reason
                    continue

                t_inter_start = time.perf_counter()
                requests = metric.get_intermediate_requests(target_type)
                try:
                    intermediates = {req.key: cache.get(req) for req in requests}
                except Exception as e:
                    skipped_metrics[display_name] = f"Intermediate error: {str(e)}"
                    continue
                t_inter_elapsed = time.perf_counter() - t_inter_start
                if t_inter_elapsed > 1.0:
                    req_keys = [r.key for r in requests]
                    logger.debug(
                        "  Intermediates for '%s' took %.1fs: %s",
                        display_name,
                        t_inter_elapsed,
                        req_keys,
                    )

                if metric.category == MetricCategory.DOWNSTREAM_DECODING:
                    parallel.append((display_name, metric, intermediates))
                else:
                    ready.append((display_name, metric, intermediates))

            # Run non-parallel metrics sequentially
            for display_name, metric, intermediates in ready:
                try:
                    t_metric_start = time.perf_counter()
                    result = metric.compute(intermediates)
                    t_metric_elapsed = time.perf_counter() - t_metric_start
                    if t_metric_elapsed > 1.0:
                        logger.debug(
                            "  Metric '%s' compute took %.1fs",
                            display_name,
                            t_metric_elapsed,
                        )
                    result.source_data_type = target_type
                    expanded = metric.expand_results(result, display_name)
                    for extra_result in expanded.values():
                        extra_result.source_data_type = target_type
                except Exception as e:
                    skipped_metrics[display_name] = f"Computation error: {str(e)}"
                    continue
                # Stored outside the try: a duplicate key is a config defect and must
                # propagate, not be recorded as this metric's "computation error".
                _store_result(results, display_name, result)
                for extra_name, extra_result in expanded.items():
                    _store_result(results, extra_name, extra_result, expanded_from=display_name)

            # Run parallelizable metrics concurrently
            if parallel:
                t_parallel_start = time.perf_counter()
                self._run_parallel_metrics(parallel, target_type, results, skipped_metrics)
                t_parallel_elapsed = time.perf_counter() - t_parallel_start
                par_names = [dn for dn, _, _ in parallel]
                logger.debug(
                    "  Parallel metrics took %.1fs: %s",
                    t_parallel_elapsed,
                    par_names,
                )

            t_group_elapsed = time.perf_counter() - t_group_start
            logger.debug(
                "Data type group %s: %d metrics in %.1fs",
                target_type.value,
                len(type_metrics),
                t_group_elapsed,
            )

        t_pipeline_elapsed = time.perf_counter() - t_pipeline_start
        logger.debug("Pipeline total: %.1fs (%d metrics)", t_pipeline_elapsed, len(results))

        # Create report
        report = create_report(results, config=self.config.model_dump())
        report.skipped_metrics = skipped_metrics
        report.original_data_type = original.data_type
        report.reconstructed_data_type = (
            reconstructed.data_type if reconstructed is not None else None
        )

        return report

    @staticmethod
    def _run_parallel_metrics(
        metrics: list[tuple[str, BaseMetric, dict[str, Any]]],
        target_type: DataType,
        results: dict[str, Any],
        skipped_metrics: dict[str, str],
    ) -> None:
        """Run a batch of metrics concurrently using threads.

        Downstream prediction metrics (sklearn fits) release the GIL during
        BLAS/LAPACK calls, so threading gives real speedup here.
        """
        with ThreadPoolExecutor(max_workers=len(metrics)) as pool:
            futures = {
                pool.submit(metric.compute, intermediates): (display_name, metric)
                for display_name, metric, intermediates in metrics
            }
            for future in as_completed(futures):
                display_name, metric = futures[future]
                try:
                    result = future.result()
                    result.source_data_type = target_type
                    expanded = metric.expand_results(result, display_name)
                    for extra_result in expanded.values():
                        extra_result.source_data_type = target_type
                except Exception as e:
                    skipped_metrics[display_name] = f"Computation error: {str(e)}"
                    continue
                # See the sequential path: collisions propagate, compute errors skip.
                _store_result(results, display_name, result)
                for extra_name, extra_result in expanded.items():
                    _store_result(results, extra_name, extra_result, expanded_from=display_name)

    def run_quick(
        self,
        original: EvalData,
        reconstructed: EvalData,
        metric_names: list[str] | None = None,
    ) -> EvaluationReport:
        """
        Run quick evaluation with subset of metrics.

        Intended for use during training loops.

        Args:
            original: Original spike data
            reconstructed: Reconstructed spike data
            metric_names: Display names of metrics to run.
                None = run all pipeline metrics (no filtering).
        """
        if metric_names is None:
            return self.run(original, reconstructed)

        name_set = set(metric_names)

        # Temporarily override metrics
        original_metrics = self.metrics
        self.metrics = [(n, m) for n, m in original_metrics if n in name_set]

        try:
            report = self.run(original, reconstructed)
        finally:
            self.metrics = original_metrics

        return report

    def compare_models(
        self,
        original: EvalData,
        reconstructions: dict[str, EvalData],
        baseline_name: str | None = None,
    ) -> tuple[dict[str, EvaluationReport], dict[str, dict[str, MetricComparison]] | None]:
        """
        Compare multiple model reconstructions on the same original data.

        Args:
            original: Original spike data
            reconstructions: Dict mapping model names to their reconstructions
            baseline_name: If provided, must be a key in ``reconstructions``.
                Floor/ceiling-normalized comparisons will be computed for all
                other models against this baseline.

        Returns:
            Tuple of (reports, normalized_comparisons).
            ``reports`` maps model names to their evaluation reports.
            ``normalized_comparisons`` maps model names to per-metric comparison
            objects (None if no baseline_name was given).
        """
        if baseline_name is not None and baseline_name not in reconstructions:
            raise ValueError(
                f"baseline_name '{baseline_name}' not found in reconstructions. "
                f"Available: {list(reconstructions.keys())}"
            )

        reports = {}
        for model_name, reconstructed in reconstructions.items():
            reports[model_name] = self.run(original, reconstructed)

        comparisons = None
        if baseline_name is not None:
            baseline_report = reports[baseline_name]
            ceiling_report = self.run(original, original)
            comparisons = {}
            for model_name, report in reports.items():
                if model_name == baseline_name:
                    continue
                comparisons[model_name] = compute_normalized_comparisons(
                    report,
                    baseline_report,
                    ceiling_report,
                )

        return reports, comparisons
