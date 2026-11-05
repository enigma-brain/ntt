"""
Spike tokenizer evaluation framework.

This module provides tools for evaluating neural spike train reconstructions,
comparing original data against tokenizer outputs using various metrics.

Data Flow Overview
------------------
The evaluation pipeline processes data through several stages:

1. **Training Data Input** (BinnedSpikesInput); SpikeTimesInput is not supported
   - Shape: (batch, time, units)
   - Source: DataLoader yields BinnedSpikesInput NamedTuples
   - Contains: spike_matrix, pad_and_gap_mask, sampling_rate, session metadata

2. **Tokenizer Processing**
   - Input: BinnedSpikesInput
   - Output: reconstruction tensor, same shape as input

3. **Validation Collation** (iter_collated_validation_segments)
   - Iterates over dataloader batches
   - Calls tokenizer to get reconstructions
   - Transposes from (time, units) to (units, time) for eval
   - Buffers and concatenates windows by session
   - Handles overlapping windows and partial segments
   - Yields CollatedValidationSegment objects

4. **Evaluation Data** (EvalData)
   - Shape: (units, time) - note the transpose from training!
   - Supports multiple representations: spike_times, binned_spikes, rates
   - Created via EvalData.from_tensor() or within the pipeline

5. **Intermediates Cache** (IntermediatesCache)
   - Lazily computes shared intermediate results
   - Stores: ISIs, spike matches, correlograms, binned counts, firing rates
   - One cache per (original, reconstructed) pair

6. **Metrics Computation**
   - Metrics request required intermediates from cache
   - DataConverter handles format conversions (spikes ↔ rates ↔ spike_times)
   - Results collected into MetricResult objects

7. **Evaluation Report** (EvaluationReport)
   - Aggregates all metric results
   - Includes category scores and skipped metrics
   - Serializable via to_dict()

Dimension Conventions
---------------------
- Training: (batch, time, units) - matches PyTorch batch-first convention
- Evaluation: (units, time) - natural for per-unit metric iteration

The validation_collation module handles the transpose automatically when
extracting data from BinnedSpikesInput.

Quick Start
-----------
For validation during training:

    from ntt.evaluation import run_validation_evaluation_per_session
    from ntt.evaluation.config import EvaluationConfig

    # Pass the tokenizer's reconstruct_from_input method directly.
    # It encodes the input, decodes with inference-time post-processing,
    # and returns a ReconstructionOutput with type metadata.
    reports = run_validation_evaluation_per_session(
        val_loader,
        tokenizer.reconstruct_from_input,
        data_type=DataType.SPIKES,
        bin_size_seconds=0.001,
        eval_config=EvaluationConfig(),
    )

For standalone evaluation:

    from ntt.evaluation import EvaluationPipeline, EvalData

    pipeline = EvaluationPipeline()
    original = EvalData.from_tensor(orig_tensor, DataType.SPIKES, bin_size_seconds=0.001)
    reconstructed = EvalData.from_tensor(recon_tensor, DataType.SPIKES, bin_size_seconds=0.001)
    report = pipeline.run(original, reconstructed)

Key Classes
-----------
- EvalData: Container for neural data in evaluation (units, time) format
- EvaluationPipeline: Main pipeline for computing metrics
- IntermediatesCache: Lazy computation and caching of shared intermediates
- DataConverter: Handles conversions between data representations
- CollatedValidationSegment: Container for concatenated validation windows

See Also
--------
- ntt.core.components.types.BinnedSpikesInput: Training data structure
- ntt.evaluation.config: Configuration options
- ntt.evaluation.metrics: Available metrics
"""

from .aggregation.results_dataframe import (
    load_results_csv,
    report_to_dataframe,
    save_results_csv,
    session_reports_to_dataframe,
)
from .analysis import compute_acgs, compute_ccgs, compute_isis
from .core.types import EvalData, EvaluationReport, MetricCategory, MetricResult
from .pipeline import EvaluationPipeline
from .plotting import (
    DisplayMode,
    get_validation_segment,
    plot_acg_comparison,
    plot_firing_rate_comparison,
    plot_per_unit_raster,
    plot_population_raster,
    plot_population_raster_correspondence,
    plot_reconstruction_summary,
)
from .radar import (
    DEFAULT_EVAL_JOB_SCALARS,
    CategorySpec,
    RadarBundle,
    ScalarMetric,
    collect_category_scores,
    inject_scalar_metrics,
    plot_radar,
    tokenizers_with_data,
)
from .utils import (
    CollatedValidationSegment,
    iter_collated_validation_segments,
    run_validation_evaluation_per_session,
)
from .visualization import (
    plot_metric_dumbbell,
    plot_normalized_comparison_radar,
    plot_per_unit_structure_preservation,
    plot_structure_preservation,
)

__all__ = [
    "EvaluationPipeline",
    "EvalData",
    "MetricResult",
    "EvaluationReport",
    "MetricCategory",
    "CollatedValidationSegment",
    "DisplayMode",
    "compute_acgs",
    "compute_ccgs",
    "compute_isis",
    "get_validation_segment",
    "iter_collated_validation_segments",
    "plot_acg_comparison",
    "plot_firing_rate_comparison",
    "plot_per_unit_raster",
    "plot_population_raster",
    "plot_population_raster_correspondence",
    "plot_reconstruction_summary",
    "run_validation_evaluation_per_session",
    "session_reports_to_dataframe",
    "report_to_dataframe",
    "save_results_csv",
    "load_results_csv",
    "plot_metric_dumbbell",
    "plot_normalized_comparison_radar",
    "plot_per_unit_structure_preservation",
    "plot_structure_preservation",
    "DEFAULT_EVAL_JOB_SCALARS",
    "CategorySpec",
    "RadarBundle",
    "ScalarMetric",
    "collect_category_scores",
    "inject_scalar_metrics",
    "plot_radar",
    "tokenizers_with_data",
]
