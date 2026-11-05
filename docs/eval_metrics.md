# Evaluation Metrics Reference

Complete reference for the evaluation metrics in the NTT evaluation pipeline.
Each metric belongs to a category and declares which data types it supports.

**Data types:** `SPIKE_TIMES` (continuous), `SPIKES` (binary bins), `RATES` (continuous bins).

**Naming convention:** Each metric has a *registry name* (Python `name` attribute, used
internally) and a *display name* (the key in the YAML config / evaluation report). These
are usually the same, but the display name can differ — e.g., registry `cv_error` has
display name `isi_cv_error` for clarity. Headings below show the **display name**,
with the registry name noted where it differs.

---

## Spike Detection (5 metrics)

Metrics that assess whether individual spikes are correctly reconstructed.

### `spike_detection_f1`

Precision, recall, and F1 for spike detection within a timing tolerance window.
Each original spike is greedily matched to the nearest reconstructed spike within
`spike_match_tolerance_ms`. Unmatched original spikes are false negatives; unmatched
reconstructed spikes are false positives.


|                    |                                    |
| ------------------ | ---------------------------------- |
| **Level**          | per-unit                           |
| **Direction**      | higher is better                   |
| **Range**          | [0, 1]                             |
| **Data types**     | SPIKE_TIMES, SPIKES                |
| **Default params** | `spike_match_tolerance_ms: 1.0`    |
| **Metadata**       | precision, recall, TP/FP/FN counts |


### `temporal_precision`

Median absolute timing error (ms) for matched spikes. Uses a wider default tolerance
than F1 so it produces useful gradients even for coarsely binned data. A perfect
reconstruction scores 0.0 ms.


|                    |                                 |
| ------------------ | ------------------------------- |
| **Level**          | per-unit                        |
| **Direction**      | lower is better                 |
| **Range**          | [0, +inf)                       |
| **Data types**     | SPIKE_TIMES, SPIKES             |
| **Default params** | `spike_match_tolerance_ms: 5.0` |
| **Metadata**       | IQR of timing errors            |


### `spike_count_correlation`

Pearson correlation of binned spike counts at multiple timescales. Primary value uses
the finest bin size; coarser timescales are reported in metadata. This tells you how
well temporal fluctuations in firing rate are preserved.


|                    |                                          |
| ------------------ | ---------------------------------------- |
| **Level**          | per-unit                                 |
| **Direction**      | higher is better                         |
| **Range**          | [-1, 1]                                  |
| **Data types**     | SPIKE_TIMES, SPIKES, RATES               |
| **Default params** | `bin_sizes_ms: [1.0, 10.0, 50.0, 100.0]` |
| **Metadata**       | correlations at each bin size            |


### `smoothed_squared_error`

Summed squared error between Gaussian-smoothed spike trains, divided by original
spike count (so the value represents "fraction of spikes mismatched"). Uses L2-normalized
kernel so each unmatched spike contributes ~1.0 to the error.


|                    |                                                          |
| ------------------ | -------------------------------------------------------- |
| **Level**          | per-unit                                                 |
| **Direction**      | lower is better                                          |
| **Range**          | [0, +inf)                                                |
| **Data types**     | SPIKES only                                              |
| **Default params** | `causal: false, filter_type: "gaussian", sigma_ms: 10.0` |
| **Notes**          | Only the Gaussian filter is supported; the exponential filter is not supported.        |


### `binned_rate_difference`

Per-unit mean absolute difference between binned spike rates (Hz). Bins spikes at
the specified resolution and computes MAE over time for each unit. Optionally
normalizes by the original mean rate.


|                    |                                                      |
| ------------------ | ---------------------------------------------------- |
| **Level**          | per-unit                                             |
| **Direction**      | lower is better                                      |
| **Range**          | [0, +inf)                                            |
| **Data types**     | SPIKES only                                          |
| **Default params** | `bin_size_ms: 5.0, normalize: false`                 |
| **Metadata**       | per-unit mean rates (original & reconstructed) in Hz |


---

## Single Unit (7 metrics)

Metrics that assess preservation of individual neuron temporal statistics.

### `isi_divergence`

Wasserstein distance between original and reconstructed inter-spike-interval (ISI)
distributions. Captures differences in firing regularity, burstiness, and rate.
KS statistic is reported in metadata as a complementary non-parametric measure.


|                    |                           |
| ------------------ | ------------------------- |
| **Level**          | per-unit                  |
| **Direction**      | lower is better           |
| **Range**          | [0, +inf)                 |
| **Data types**     | SPIKE_TIMES, SPIKES       |
| **Default params** | `isi_smoothing_kernel: 1` |
| **Metadata**       | KS statistic              |


### `acg_correlation`

Pearson correlation between original and reconstructed autocorrelograms (ACGs).
The ACG captures a neuron's temporal firing patterns (refractory period, bursting,
oscillations). Supports all data types: uses optimized Phy algorithm for spike times,
FFT-based cross-correlation for rates/coarse bins.


|                    |                                                                                        |
| ------------------ | -------------------------------------------------------------------------------------- |
| **Level**          | per-unit                                                                               |
| **Direction**      | higher is better                                                                       |
| **Range**          | [-1, 1]                                                                                |
| **Data types**     | SPIKE_TIMES, SPIKES, RATES                                                             |
| **Default params** | `correlogram_max_lag_ms: 100.0, correlogram_bin_size_ms: 1.0, acg_smoothing_kernel: 1` |


### `isi_cv_error`

*Registry name: `cv_error`*

Absolute error in ISI coefficient of variation (CV = std/mean of ISIs). CV measures
the overall regularity of a neuron's firing: CV < 1 is more regular than Poisson,
CV > 1 is more bursty.


|                    |                                          |
| ------------------ | ---------------------------------------- |
| **Level**          | per-unit                                 |
| **Direction**      | lower is better                          |
| **Range**          | [0, +inf)                                |
| **Data types**     | SPIKE_TIMES, SPIKES                      |
| **Default params** | `isi_smoothing_kernel: 1`                |
| **Metadata**       | CV values for original and reconstructed |


### `isi_cv2_error`

*Registry name: `cv2_error`*

Absolute error in CV2, a local ISI irregularity measure. CV2 = mean of
`2 * |ISI[n+1] - ISI[n]| / (ISI[n+1] + ISI[n])`. Unlike global CV, CV2 is
insensitive to slow firing rate changes, measuring only local spike-to-spike
variability.


|                    |                           |
| ------------------ | ------------------------- |
| **Level**          | per-unit                  |
| **Direction**      | lower is better           |
| **Range**          | [0, +inf)                 |
| **Data types**     | SPIKE_TIMES, SPIKES       |
| **Default params** | `isi_smoothing_kernel: 1` |


### `firing_rate_error`

Per-unit absolute error in mean firing rate (Hz). Computes |rate_orig - rate_recon|
for each unit. A simple but important sanity check: if the tokenizer systematically
over- or under-predicts spikes, this metric catches it.


|                    |                            |
| ------------------ | -------------------------- |
| **Level**          | per-unit                   |
| **Direction**      | lower is better            |
| **Range**          | [0, +inf)                  |
| **Data types**     | SPIKE_TIMES, SPIKES, RATES |
| **Default params** | (none)                     |
| **Metadata**       | original and reconstructed rates (Hz) |

Its normalized score is taken against a population-mean-rate reference (every unit given
the segment's mean rate), not the rate-matched Poisson floor, which keeps each unit's rate
and so has an error of only sampling noise; see
[`metric_glossary.md`](metric_glossary.md#composite-scores-the-7-radar-axes).


### `fano_factor_error`

Per-unit absolute error in Fano factor (variance / mean of spike counts).
FF = 1 for Poisson, FF < 1 for regular firing, FF > 1 for bursty firing.
Captures whether the reconstruction preserves the variability structure
of each neuron's firing at a configurable timescale.


|                    |                            |
| ------------------ | -------------------------- |
| **Level**          | per-unit                   |
| **Direction**      | lower is better            |
| **Range**          | [0, +inf)                  |
| **Data types**     | SPIKE_TIMES, SPIKES, RATES |
| **Default params** | `fano_bin_size_ms: 100.0`  |
| **Metadata**       | Fano factors for original and reconstructed |


### `per_unit_structure_preservation`

RSA (Representational Similarity Analysis) between per-unit temporal activity patterns
in original spike space vs. latent space. Builds a U x U RDM (units x units) and
computes Spearman correlation between input and latent RDMs.

Answers: "Do neurons with similar temporal activity patterns remain similar in the
latent space?"

Only works with per-neuron tokenizers (latent shape `[N_windows, n_units, latent_dim]`).


|                    |                                                                                                                                      |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------ |
| **Level**          | per-unit                                                                                                                             |
| **Direction**      | higher is better                                                                                                                     |
| **Range**          | [-1, 1]                                                                                                                              |
| **Data types**     | SPIKE_TIMES, SPIKES, RATES                                                                                                           |
| **Default params** | `input_metric: "correlation", latent_metric: "euclidean", comparison_method: "spearman", max_samples: 200, smoothing_sigma_bins: 25` |
| **Metadata**       | RSA p-value, k-NN overlap, RDMs                                                                                                      |


---

## Pairwise (2 metrics)

Metrics that assess preservation of inter-neuron interactions.

### `ccg_correlation`

Pearson correlation between original and reconstructed cross-correlograms (CCGs).
CCGs capture the precise timing relationships between neuron pairs (synaptic
connections, common input, oscillatory coupling).


|                    |                                                                                         |
| ------------------ | --------------------------------------------------------------------------------------- |
| **Level**          | per-pair                                                                                |
| **Direction**      | higher is better                                                                        |
| **Range**          | [-1, 1]                                                                                 |
| **Data types**     | SPIKE_TIMES, SPIKES, RATES                                                              |
| **Default params** | `correlogram_max_lag_ms: 100.0, correlogram_bin_size_ms: 1.0, max_pairs_to_compute: 20` |
| **Metadata**       | per-pair correlation values                                                             |


### `functional_connectivity_f1`

*Registry name: `connectivity_f1`*

F1 score for detecting functionally connected pairs. A pair is "significant" if
its CCG has a z-score peak above threshold. Compares which pairs are significant
in original vs. reconstructed data.


|                    |                                                                                                                          |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| **Level**          | population                                                                                                               |
| **Direction**      | higher is better                                                                                                         |
| **Range**          | [0, 1]                                                                                                                   |
| **Data types**     | SPIKE_TIMES, SPIKES, RATES                                                                                               |
| **Default params** | `ccg_significance_threshold: 3.0, correlogram_max_lag_ms: 100.0, correlogram_bin_size_ms: 1.0, max_pairs_to_compute: 20` |
| **Metadata**       | precision, recall, connection counts                                                                                     |


---

## Population (4 metrics)

Metrics that assess preservation of population-level neural structure.

### `participation_ratio_error`

*Registry name: `dimensionality_error`*

Absolute error in participation ratio (effective dimensionality).
PR = (sum of eigenvalues)^2 / (sum of squared eigenvalues) from the population
covariance matrix. Measures how many dimensions the population activity occupies.


|                    |                                                      |
| ------------------ | ---------------------------------------------------- |
| **Level**          | population                                           |
| **Direction**      | lower is better                                      |
| **Range**          | [0, +inf)                                            |
| **Data types**     | SPIKE_TIMES, SPIKES, RATES                           |
| **Default params** | `population_bin_size_ms: 10.0`                       |
| **Metadata**       | dimensionality values for original and reconstructed |


### `subspace_overlap`

Agreement between the top PCA subspaces of original and reconstructed population
activity, measuring whether the dominant directions of population variability are
preserved. The default `chordal` measure (`1 - sqrt(mean(sin^2 theta))` over the
principal angles) expands resolution near perfect overlap, where the
alternative `mean_cosine` measure saturates because the effective rank exceeds `k`;
select either measure via `subspace_overlap_measure`. Both reductions are higher-better in [0, 1].


|                    |                                                                                  |
| ------------------ | -------------------------------------------------------------------------------- |
| **Level**          | population                                                                       |
| **Direction**      | higher is better                                                                 |
| **Range**          | [0, 1]                                                                            |
| **Data types**     | SPIKE_TIMES, SPIKES, RATES                                                       |
| **Default params** | `pca_n_components: 20, population_bin_size_ms: 10.0, subspace_overlap_measure: chordal` |
| **Metadata**       | principal angles, `subspace_k`, both measures, explained variance ratios         |


### `population_coupling_preservation`

*Registry name: `population_coupling`*

Agreement (concordance correlation by default, or Pearson via `coupling_agreement`)
between the per-neuron population-coupling vectors of the original and reconstructed
data. For each neuron, coupling = correlation between that neuron and the population
sum (excluding itself). Measures whether the relative participation of each neuron in
population dynamics is preserved.

**Pinned to 1 ms.** At population bins ≥ 5 ms, coupling is dominated by shared rate
co-modulation, which a rate-matched Poisson baseline reproduces almost perfectly — the
Poisson floor saturates at 0.97–0.999, leaving no headroom and a near-degenerate
normalization denominator (so good tokenizers score hugely negative for ~0.01
differences). At 1 ms the floor drops (≈0.6–0.85 in motor cortex), restoring headroom.
The metric therefore defaults to 1 ms and **warns above 5 ms**; the normalization layer
additionally gates any axis whose floor↔ceiling separation is below 0.1
(`insufficient_baseline_separation`). Strongly stimulus-driven areas (e.g. V4) stay
saturated even at 1 ms, where that gate is the backstop.


|                    |                                                       |
| ------------------ | ----------------------------------------------------- |
| **Level**          | population                                            |
| **Direction**      | higher is better                                      |
| **Range**          | [-1, 1]                                               |
| **Data types**     | SPIKE_TIMES, SPIKES, RATES                            |
| **Default params** | `population_bin_size_ms: 1.0, coupling_agreement: ccc` |
| **Metadata**       | per-neuron coupling values, `coupling_pearson`        |


### `population_structure_preservation`

RSA between population activity patterns in original spike space vs. latent space.
Builds an N x N RDM (time windows x time windows) and computes Spearman correlation
between input and latent RDMs.

Answers: "Do time windows that were similar across the whole neural population remain
similar in the latent space?"


|                    |                                                                                                                                      |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------ |
| **Level**          | population                                                                                                                           |
| **Direction**      | higher is better                                                                                                                     |
| **Range**          | [-1, 1]                                                                                                                              |
| **Data types**     | SPIKE_TIMES, SPIKES, RATES                                                                                                           |
| **Default params** | `input_metric: "correlation", latent_metric: "euclidean", comparison_method: "spearman", max_samples: 200, smoothing_sigma_bins: 25` |
| **Metadata**       | RSA p-value, k-NN overlap, RDMs                                                                                                      |


---

## Information (2 metrics)

### `mutual_information`

Normalized mutual information (NMI) between original and reconstructed spike counts.
Uses the exact discrete distribution of integer counts (no histogram binning) with
Miller-Madow bias correction. Primary value is NMI = MI / H(original), representing
the fraction of original information preserved: 0 = no information, 1 = perfect.

Computes at multiple timescales. Returns NaN when the estimate is unreliable (too few
samples, zero original entropy, or sparse contingency table).

|                    |                                                              |
| ------------------ | ------------------------------------------------------------ |
| **Level**          | per-unit                                                     |
| **Direction**      | higher is better                                             |
| **Range**          | [0, 1]                                                       |
| **Data types**     | SPIKE_TIMES, SPIKES, RATES                                   |
| **Default params** | `bin_sizes_ms: [1.0, 5.0, 10.0, 50.0, 100.0], min_samples: 50` |
| **Metadata**       | NMI/MI/H(orig)/H(recon)/bias per timescale                   |


### `token_dims_per_second`

Intrinsic compression rate of the tokenizer: the number of scalar latent
dimensions emitted per second of input. Computed as
`n_token_dims_per_token / token_bin_size_seconds`, where
`n_token_dims_per_token` is `latent_dim` for population tokenizers or
`U * latent_dim` for per-neuron tokenizers. The token bin size is taken from
the tokenizer's declared `token_bin_size_ms` when available, otherwise
inferred as the median diff of `token_timestamps` (`source` in metadata
records which path was used).

Lower-is-better reflects higher compression, but note this is only one side
of the rate–distortion trade-off and must be read together with reconstruction
quality. No stimulus alignment required.

|                    |                                                            |
| ------------------ | ---------------------------------------------------------- |
| **Level**          | population                                                 |
| **Direction**      | lower is better                                            |
| **Range**          | [0, +inf)                                                  |
| **Data types**     | SPIKE_TIMES, SPIKES, RATES                                 |
| **Default params** | none                                                       |
| **Metadata**       | token_bin_size_ms, latent_dim, n_token_dims_per_token, source, is_per_neuron, per_token_shape |


---

## Downstream Prediction (2 metrics)

Metrics that assess whether latent representations support behavioral decoding.

### `behavioral_regression`

Ridge regression R^2 from latent representations to continuous behavioral variables
(e.g., cursor position). Uses GroupShuffleSplit for trial-aware train/validation
splitting. RidgeCV selects the best regularization strength internally.


|                    |                                                                 |
| ------------------ | --------------------------------------------------------------- |
| **Level**          | population                                                      |
| **Direction**      | higher is better                                                |
| **Data types**     | SPIKE_TIMES, SPIKES, RATES                                      |
| **Default params** | `target_field: "cursor_pos", grouping_field: "trial_id", cv: 5` |
| **Metadata**       | best alpha, train/val splits, sample trial trajectories         |


### `behavioral_classification`

Ridge classification accuracy from latent representations to discrete behavioral
labels (e.g., reach target direction). Uses StratifiedGroupKFold for balanced
trial-aware cross-validation. Reports mean accuracy across folds.


|                    |                                                                                              |
| ------------------ | -------------------------------------------------------------------------------------------- |
| **Level**          | population                                                                                   |
| **Direction**      | higher is better                                                                             |
| **Data types**     | SPIKE_TIMES, SPIKES, RATES                                                                   |
| **Default params** | `target_field: "trial_target_id", grouping_field: "trial_id", cv: 5, balance_classes: false` |
| **Metadata**       | per-class accuracy, chance level, class counts, fold accuracies                              |


### Filtering

Downstream metrics apply two-level filtering to select relevant data for decoding.

**Regression — chunk-level (applied to all chunks before CV split):**

| Filter | Drop condition | Threshold |
|---|---|---|
| Phase inclusion | >25% of tokens NOT in `train_include_phases` | 0.25 |
| Trial validity | >25% of tokens with `trial_is_valid == 0` | 0.25 |
| Custom field | >25% of tokens NOT in `train_filter_values` | 0.25 |
| NaN/Inf targets | ANY NaN/Inf in chunk targets | all-or-nothing |

**Regression — eval-time (per-bin at behavioral resolution, post-prediction):**

| Filter | Exclude condition |
|---|---|
| Phase inclusion | bin's phase NOT in `eval_include_phases` |
| Trial validity | `trial_is_valid == 0` at that bin |
| Custom field | bin's value NOT in `eval_filter_values` |

**Classification — chunk-level (applied to all chunks before CV split):**

| Filter | Drop condition | Threshold |
|---|---|---|
| Phase inclusion | >25% of tokens NOT in `include_phases` | 0.25 |
| Trial validity | >25% of tokens with `trial_is_valid == 0` | 0.25 |
| Custom field | >25% of tokens NOT in `filter_values` | 0.25 |
| NaN labels | >25% of labels are NaN | 0.25 |
| Excluded labels | majority label in `excluded_label_values` | — |
| Low majority | majority fraction (among non-NaN) < 0.75 | 0.75 |

Trial-boundary drop (chunk spanning multiple trials) is not applied; instead, majority grouping value is used for CV group assignment.

### Default decoded variables

The default evaluation config decodes the following behavioral variables:

| Metric name | Type | Target variable | Phases | Notes |
|---|---|---|---|---|
| `behavioral_regression_cursor_pos` | regression | `cursor_pos` | reach, random | |
| `behavioral_regression_cursor_vel` | regression | `cursor_vel` | reach, random | |
| `behavioral_classification_trial_target_id` | classification | `trial_target_id` | reach, random | |
| `behavioral_classification_movement_phase` | classification | `movement_phase` | all | **uses class balancing** |
| `behavioral_classification_trial_result` | classification | `trial_result` | return only | filters to successful (0) and failed (2) trials |

Not all metrics run on all session types. Metrics that require `reach_period` or
`random_period` will only produce results for sessions that contain those phases
(center-out reaching sessions have reach periods, random-target sessions have
random periods). Invalid trials are excluded.
We decode reaching target and trajectory only during the reach phase
as done in POYO.  `trial_result` classification requires sessions with both
successful and unsuccessful trials — sessions with only successful trials will
skip this metric.

## Robustness (latent quality)
For details see also `docs/robustness_metrics.md`. Included in job=eval_all.

### Robustness to input-space corruptions
Measures how much input corruptions move the latents, reported as latent MSE normalized by clean token variance. Lower means more robust.
### Robustness to latent-space corruptions
Measures how decoding degrades when per-dimension latent noise is added at scale epsilon * sigma_d, where sigma_d is per-dimension latent variance. Direction (lower vs higher is better) equals direction of the reconstruction metric used.

## SNR (latent quality)
Included in job=eval_all.

Evaluates signal-to-noise ratio based on data containing repeats. Measures the variance divided between stimulus-mean tokens, by the within-stimulus variance. (Signal variance divided by noise variance.)


---

## Data Type Compatibility Summary


| Metric                              | SPIKE_TIMES | SPIKES | RATES |
| ----------------------------------- | ----------- | ------ | ----- |
| `spike_detection_f1`                | yes         | yes    | no    |
| `temporal_precision`                | yes         | yes    | no    |
| `spike_count_correlation`           | yes         | yes    | yes   |
| `smoothed_squared_error`            | no          | yes    | no    |
| `binned_rate_difference`            | no          | yes    | no    |
| `isi_divergence`                    | yes         | yes    | no    |
| `acg_correlation`                   | yes         | yes    | yes   |
| `isi_cv_error`                      | yes         | yes    | no    |
| `isi_cv2_error`                     | yes         | yes    | no    |
| `firing_rate_error`                 | yes         | yes    | yes   |
| `fano_factor_error`                 | yes         | yes    | yes   |
| `per_unit_structure_preservation`   | yes         | yes    | yes   |
| `ccg_correlation`                   | yes         | yes    | yes   |
| `functional_connectivity_f1`        | yes         | yes    | yes   |
| `participation_ratio_error`         | yes         | yes    | yes   |
| `subspace_overlap`                  | yes         | yes    | yes   |
| `population_coupling_preservation`  | yes         | yes    | yes   |
| `population_structure_preservation` | yes         | yes    | yes   |
| `mutual_information`                | yes         | yes    | yes   |
| `behavioral_regression`             | yes         | yes    | yes   |
| `behavioral_classification`         | yes         | yes    | yes   |


**Summary:** The metrics marked `no` under RATES are SPIKES/SPIKE_TIMES only (all in
spike detection and single unit categories). Rate-based tokenizers (e.g., LDNS) lose
coverage of spike-level timing fidelity and ISI statistics.

---

## Shared Intermediates

Metrics share precomputed intermediates via a caching system. Understanding which
intermediates are shared helps when configuring metric subsets.


| Intermediate                                            | Used by                                                     | Notes                                           |
| ------------------------------------------------------- | ----------------------------------------------------------- | ----------------------------------------------- |
| `spike_matches`                                         | spike_detection_f1, temporal_precision                      | Parameterized by tolerance_ms                   |
| `firing_rates_original/reconstructed`                   | firing_rate_error                                           | Mean rate per unit                              |
| `isi_original/reconstructed`                            | isi_divergence, isi_cv_error, isi_cv2_error                 | Shared across all ISI metrics                   |
| `acg_original/reconstructed`                            | acg_correlation                                             | Extracted from full correlogram matrix          |
| `ccg_original/reconstructed`                            | ccg_correlation, functional_connectivity_f1                 | Shared; parameterized by max_pairs              |
| `binned_counts_original/reconstructed`                  | spike_count_correlation, mutual_information, fano_factor_error | Parameterized by bin_sizes_ms                 |
| `population_vectors_original/reconstructed`             | participation_ratio_error, subspace_overlap, population_coupling_preservation | Shared; parameterized by population_bin_size_ms |
| `binned_spikes_original/reconstructed`                  | smoothed_squared_error, binned_rate_difference              | Direct passthrough of input data                |
| `latent_windows` / `original_windows`                   | population_structure_preservation                           | Window-level representations                    |
| `latent_windows_per_unit` / `original_windows_per_unit` | per_unit_structure_preservation                             | Per-neuron window-level representations         |
| `latents` / `behavioral_data` / `token_timestamps`      | behavioral_regression, behavioral_classification            | Direct passthrough                              |
