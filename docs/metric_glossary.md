# Metric Glossary

Every metric shown in the local dashboard (`scripts/analysis/dashboard.py`), with what it measures, its range/units, and
whether **higher** or **lower** is better.

Two tiers:

- **7 composite scores** — the radar axes. Each aggregates many raw metrics onto a
  roughly `[0, 1]` scale where higher is better.
- **~88 raw `{block}/{metric}` columns** — the per-metric values before composition,
  grouped into blocks. Each bare column is a cross-session mean; the Ranking-tab
  whisker is the cross-session spread. Raw columns keep their natural direction
  (correlations, F1, R² higher-better; errors and divergences lower-better) — read
  the direction tag on each.

Timescale suffixes (`_5ms`, `_10ms`, `_25ms`, `_50ms`) give the bin size used for the
statistic; most bare (unsuffixed) columns use a 1 ms bin (per-metric exceptions are
noted below — e.g. `binned_rate_difference` uses 5 ms). For `acg_correlation` and the
CCG metrics the suffix also scales the lag window (noted below).

## Composite scores (the 7 radar axes)

The two Poisson-normalized composites rescale each contributing metric against a
rate-matched Poisson floor (0 = baseline) and a perfect-reconstruction ceiling
(1 = ceiling), so higher is always better regardless of the metric's native scale.
They are **family-weighted**: a metric's bin-size and strength variants collapse to a
single family of weight 1, so a metric is not over-counted through its `_5ms`/`_10ms`
variants.

One metric uses a different floor. The Poisson floor keeps each unit's own mean rate, so
its `firing_rate_error` is only sampling noise (about 0.1 Hz), and every real tokenizer
would sit at the −1 clip. `firing_rate_error` is instead scored against a
**population-mean-rate reference**, which gives every unit the mean rate of its segment
(keeping the population's average rate, dropping each unit's own rate). Its error per
segment is `R_seg = mean_u |r_u − mean_u' r_u'|` over the same units as the model's error.
`R_seg` is aggregated over segments and sessions exactly like the model's error `E`, and the
score is taken once at the aggregate level: `1 − E / R` (1 = perfect, 0 = no better than
giving every unit the population mean rate).

### `single_neuron_recon` — "SN Recon (Poisson-norm.)"
Family-weighted mean of the Poisson-normalized scores of the `temporal_fidelity` and
`single_unit_statistics` metrics (0 = Poisson baseline, 1 = perfect reconstruction),
clipped below at −1 before averaging. `firing_rate_error` enters with its score against the
population-mean-rate reference (see above).
**Range:** roughly `[-1, 1]`. **Higher is better.**

### `population_structure` — "Population Structure (Poisson-norm.)"
Same family-weighted, Poisson-normalized construction, over the `population_structure`
block. Excludes `token_dims_per_second` (a throughput metric) and the unbounded
`repeats_snr` / `repeats_baseline` diagnostics (still shown as raw columns).
**Range:** roughly `[-1, 1]`. **Higher is better.**

### `decoding` — "Decoding (POYO + linear)"
Held-out **R²** of two decoder families, each carrying total weight 1: the POYO
decoder's per-session `test_r2` (averaged over sessions) and the in-process ridge
behavioral regressors (`downstream_decoding/behavioral_regression_*`, averaged over
targets); the score is the mean of the two family means. Non-finite values are dropped.
Position regression counts only where its selected chunk size is 200 ms. Behavioral
classification probes (accuracy, not R²) do not contribute.
**Range:** `(-∞, 1]`. **Higher is better.**

The two family means are reported next to the composite, so either can be read on its own:

- **`decoding_poyo`** — "POYO decoding (cursor-velocity R²)": the POYO family mean, the
  mean over sessions of the POYO decoder's cursor-velocity test R². Equal to the
  wide-table `decoding/test_r2`.
- **`decoding_linear`** — "Linear decoding (ridge R²)": the ridge family mean, the mean
  over the regression targets that enter the composite (cursor velocity, and cursor
  position where its chunk size is 200 ms).

Each is empty when its family contributes nothing, and `decoding` is their mean (or the one
that is present).

### `downstream_encoding` — "Downstream Encoding (nc_r²)"
Mean noise-ceiling-corrected R² (`nc_r2`) from the downstream stimulus-encoding run
(Cowley-V4 / Wingert).
**Range:** `(-∞, 1]` (can be negative). **Higher is better.**

### `latent_structure_and_robustness` — "Latent Structure & Robustness"
Bounded `[0, 1]` family-weighted mean over up to four equally-weighted members, each
mapped onto `[0, 1]`:

- **Latent structure preservation** — two members: `max(corr, 0)` of the raw RSA
  correlation of `population_structure_preservation` and
  `per_unit_structure_preservation` (the latter only for per-neuron tokenizers; a
  non-positive correlation pins to 0).
- **Input-corruption retention (M1)** — `1 / (1 + d)`, where `d` is the mean
  latent-NMSE over three fixed input corruptions (`window_shift|max_shift_ms=3.0`,
  `jitter|k_ms=1.0|p=0.2`, `drop_and_add|fraction=0.02`); `d = 0` gives 1.
- **Latent-noise robustness (M2)** — `max(acg_correlation, 0)` at the configured
  latent-noise level.

The score is the mean over whichever members are present.
**Range:** `[0, 1]`. **Higher is better.**

### `generalization` — "Generalization"
How well the tokenizer's *reconstruction* holds up out-of-distribution relative to the
in-distribution `xtime` baseline. For each reconstruction metric and each OOD
tier (`xsession`, `xsubject`, `xsource`) a retention term is formed: each held-out
session is compared with the `xtime` session(s) of its own brainset (`session/baseline`
for higher-better metrics, `baseline/session` for lower-better), clipped to `[0, 1]`,
and the term is the mean over the tier's sessions. `xsource` sessions, whose brainsets
have no `xtime` session, are compared with the mean over all `xtime` sessions. A session
is left out when its baseline is not positive (the ratio is undefined or inverted), or
when its brainset's `xtime` sessions lack the metric. The
composite is the **geometric mean** over the `xsource` terms only (one per metric), each
floored at `1e-3`: it measures generalization to datasets unseen in training. In the
public eval, reconstruction does not measurably degrade on held-out sessions and subjects
of the training datasets (before clipping, their retention is centred above 1), and a
model's retention on one such session does not predict its retention on another, so
`xsession` and `xsubject` would only add noise to the score. Scored over the
reconstruction metrics only (single-unit statistics, temporal
fidelity, population structure), which every dataset has. The `latent_quality` metrics
are left out: they are signed correlations near zero, so their ratio to the baseline is
close to a 0-or-1 coin flip. The downstream metrics are left out too: decoding exists
only for datasets with behaviour, so it would weight the score towards them.
`token_dims_per_second` and `n_skipped` are also excluded.
**Range:** `[0, 1]`. **Higher is better.**

**Generalization tab.** It shows the same reconstruction metrics only, per tier and per
metric, for all three tiers (`xsession` and `xsubject` for reference; only `xsource`
enters the score). Each tokenizer's tiers are compared over the metrics all three tiers
share.
Because each `xsession` and `xsubject` session is compared with its own brainset, those
tiers' different dataset mixes do not set their retention (`xsource` has no same-brainset
baseline). In the public eval's test split each tier holds only one session per brainset,
so tier retentions are noisy, and the tiers should not be ranked against each other
from these numbers.

### `compression` — "Compression"
`compression_rate = input_bits / output_bits`: the input's modeled bit-budget over
the latent output's bit-budget. `input_bits` = `log2(bin_size_ms + 1)` bits per neuron
per ms × window length × neuron count; `output_bits` = `t_tokens · u_units ·
latent_dim · bits_per_dim`, with `bits_per_dim` = 16 for continuous tokenizers and
`log2(vocab_size)` for discrete ones. **NaN** when a discrete vocab size is unknown or
`output_bits` = 0.
**Range:** `[0, ∞)`; values **> 1 mean the latent carries fewer bits than the input**.
**Higher is better.** One side of a rate–distortion trade-off — a tokenizer that
discards information also compresses well, so read it alongside the reconstruction
scores. On the radar: **Compression (log, relative)** — log₁₀ of the rate, min–max
scaled over all models shown (0 = the lowest rate, 1 = the highest); hovers, ranking
and scatter show the raw rate.

---

## Block: `single_unit_statistics`

Per-unit statistics of each neuron's spike train, original vs reconstructed.

- **`acg_correlation`** — Pearson correlation between the original and reconstructed
  per-unit **auto-correlogram** (a unit's spikes relative to itself across time lags),
  averaged across units. Range `[-1, 1]`. **Higher is better.**
- **`acg_correlation_5ms`, `acg_correlation_10ms`** — same, with a 5 ms / 10 ms
  histogram bin. The bin also scales the max lag window (100× the bin: 5 ms → ±500 ms,
  10 ms → ±1000 ms). Range `[-1, 1]`. **Higher is better.**
- **`firing_rate_error`** — per-unit absolute error in mean firing rate,
  `|rate_orig − rate_recon|`. Range `[0, ∞)`, Hz. **Lower is better.** Its normalized
  score is `1 − E / R` against the population-mean-rate reference, not the Poisson floor
  (see the composite scores above).
- **`fano_factor_error`** — per-unit absolute error in the Fano factor
  (variance / mean of binned spike counts). Range `[0, ∞)`, dimensionless.
  **Lower is better.**
- **`fano_factor_error_5ms`, `_10ms`, `_25ms`, `_50ms`** — same, at a 5/10/25/50 ms
  count bin. **Lower is better.**
- **`isi_cv_error`** — per-unit absolute error in the inter-spike-interval
  **coefficient of variation** (std(ISI) / mean(ISI)). Range `[0, ∞)`, dimensionless.
  **Lower is better.**
- **`isi_cv2_error`** — per-unit absolute error in **CV2**, the mean local ISI
  irregularity `2·|ISIₙ₊₁ − ISIₙ| / (ISIₙ₊₁ + ISIₙ)`. Range `[0, ∞)`, dimensionless.
  **Lower is better.**
- **`isi_divergence`** — Wasserstein (earth-mover) distance between the original and
  reconstructed per-unit ISI distributions. Range `[0, ∞)`, seconds.
  **Lower is better.**

---

## Block: `temporal_fidelity`

Time-aligned, per-bin or per-spike comparison of original vs reconstructed spike
trains.

- **`binned_rate_difference`** — per-unit mean absolute error over time between
  original and reconstructed binned spike rates. The bare column uses a 5 ms bin
  (identical to `binned_rate_difference_5ms`). Range `[0, ∞)`, Hz. **Lower is better.**
- **`binned_rate_difference_5ms`, `_10ms`, `_25ms`, `_50ms`** — same, at the given bin
  size. **Lower is better.**
- **`mutual_information`** — normalized mutual information between original and
  reconstructed integer spike counts, `NMI = MI / H(original)` (Miller–Madow
  corrected), finest bin. Range `[0, 1]`. **Higher is better.**
- **`mutual_information_5ms`, `_10ms`, `_25ms`, `_50ms`** — same, at the given count
  bin size. **Higher is better.**
- **`smoothed_squared_error`** — per-unit summed squared error between the
  Gaussian-smoothed original and reconstructed spike trains, normalized by the
  original spike count (a Van-Rossum-style distance, ~1.0 per fully mismatched spike).
  The bare column uses a 1 ms Gaussian std. Range `[0, ∞)`, dimensionless.
  **Lower is better.**
- **`smoothed_squared_error_5ms`, `_10ms`, `_50ms`, `_100ms`** — same, where the
  timescale is the Gaussian smoothing-kernel width (std, ms). **Lower is better.**
- **`spike_count_correlation`** — per-unit Pearson correlation between original and
  reconstructed binned spike-count series (finest bin). Range `[-1, 1]`.
  **Higher is better.**
- **`spike_count_correlation_5ms`, `_10ms`, `_25ms`, `_50ms`** — same, at the given
  count bin size. **Higher is better.**
- **`spike_detection_f1`** — per-unit F1 for matched spikes; an original spike is a
  true positive if a reconstructed spike falls within the match tolerance
  (default 1.0 ms), greedy nearest-match. Range `[0, 1]`. **Higher is better.**
- **`spike_detection_f1_strict`** — same, with a tighter 0.5 ms tolerance.
  Range `[0, 1]`. **Higher is better.**
- **`temporal_precision`** — median absolute timing error over matched
  original→reconstructed spike pairs (5 ms match tolerance). The tolerance must exceed
  the 1 ms conversion bin size, or matched pairs collapse to zero error on the shared
  bin grid. Range `[0, ∞)`, ms; 0 = perfect timing. **Lower is better.**

---

## Block: `population_structure`

Pairwise and population-level structure across units. For the population metrics
(`participation_ratio_error`, `subspace_overlap`, `population_coupling_preservation`)
the suffix sets the bin size for the population activity vectors; the bare column uses
a 1 ms bin. For the CCG metrics (`ccg_correlation`, `functional_connectivity_f1`) the
bare column uses a 1 ms bin / ±100 ms lag, and the suffix scales both the
cross-correlogram bin size and the lag window (5 ms → ±500 ms, 10 ms → ±1000 ms). CCG
metrics run over a capped random subset of neuron pairs.

- **`ccg_correlation`** — mean Pearson correlation, over a capped random subset of
  neuron pairs, between the original and reconstructed **cross-correlograms** (one
  unit's spikes relative to another's across lags — pairwise temporal interaction).
  Range `[-1, 1]`. **Higher is better.**
- **`ccg_correlation_5ms`, `ccg_correlation_10ms`** — same, with a 5/10 ms CCG bin and
  scaled lag window. Range `[-1, 1]`. **Higher is better.**
- **`functional_connectivity_f1`** — F1 for recovering the set of significantly
  connected pairs: a pair is connected if its CCG peak exceeds a z-score threshold
  (default 3.0); original-connected pairs are ground truth. Range `[0, 1]`.
  **Higher is better.**
- **`participation_ratio_error`** — absolute difference between the **participation
  ratio** (effective dimensionality, `(Σλ)² / Σλ²` over the positive covariance
  eigenvalues λ) of the original vs reconstructed population activity. Range `[0, ∞)`,
  dimensionless; 0 = identical effective dimensionality. **Lower is better.**
- **`participation_ratio_error_5ms`, `_10ms`** — same, at a 5/10 ms population bin.
  **Lower is better.**
- **`population_coupling_preservation`** — agreement across units between each unit's
  **population coupling** (its correlation with the population sum excluding itself) in
  the original vs the reconstruction. The agreement statistic defaults to Lin's
  concordance correlation, which penalizes scale and location drift as well as rank.
  The bare column uses a 1 ms population bin; at bins ≥ 5 ms the rate-matched Poisson
  floor saturates (0.97–0.999), leaving little headroom. Range `[-1, 1]`.
  **Higher is better** (despite the name, it is a correlation).
- **`population_coupling_preservation_5ms`, `_10ms`** — same, at a 5/10 ms population
  bin. Coarse-bin coupling sits near the saturated Poisson floor — read with that
  caveat. **Higher is better.**
- **`subspace_overlap`** — agreement between the top PCA subspaces (default 20
  components) of the original vs reconstructed population covariance, via their
  principal angles θ. Default measure **chordal**: `1 − sqrt(mean(sin²θ))`.
  Range `[0, 1]`; 1 = identical subspaces. **Higher is better.**
- **`subspace_overlap_5ms`, `_10ms`** — same, at a 5/10 ms population bin.
  **Higher is better.**

---

## Block: `latent_quality`

Properties of the tokenizer's latent representation itself. The two RSA metrics are not
part of the generalization score or the Generalization tab, which use reconstruction
metrics only.

- **`population_structure_preservation`** — RSA: Spearman rank correlation between the
  upper triangles of two window×window representational dissimilarity matrices (RDMs),
  one from the original population activity and one from the latent — do windows that
  were similar in activity stay similar in latent space. Range `[-1, 1]`.
  **Higher is better.**
- **`per_unit_structure_preservation`** — same RSA over a unit×unit RDM (each unit a
  vector of its activity across windows) — do units with similar temporal patterns
  stay similar in latent space. Range `[-1, 1]`. **Higher is better.** Defined only for
  per-neuron tokenizers whose latent keeps a per-window unit axis; population
  tokenizers emit a flat per-window latent with no unit axis and return NaN.
- **`token_dims_per_second`** — the latent **rate**:
  `n_token_dims_per_token / token_bin_size_seconds`, i.e. scalar latent dimensions
  emitted per second of neural signal. Range `[0, ∞)`, dims·s⁻¹. **Lower is better**
  (lower = more compression). One side of a rate–distortion trade-off; read with the
  reconstruction scores. Excluded from the generalization score.

---

## Block: `robustness`

Stability of the latent under input perturbations. Each entry is a **normalized latent
divergence**: the mean squared distance between a perturbed replica's latent tokens and
the clean replica's, normalized by token variance (replica 0 is the clean reference).
0 = the latent is unchanged; larger = the latent moved more. **Lower is better.** The
M1 member of `latent_structure_and_robustness` converts the three canonical corruptions
(`window_shift`, `jitter`, `drop_and_add`) to a higher-is-better retention `1/(1+d)`.

Two families, differing only in the normalizer:

- **`robustness/robustness_input/{perturbation}`** — joint normalized MSE: a single
  denominator (the per-token joint variance to the mean token).
- **`robustness/robustness_input_per_dim/{perturbation}`** — per-dimension normalized
  MSE: each latent dim divided by its own variance, then averaged over dims.
  Corruptions that disproportionately move low-variance dims read larger here than in
  the joint metric. Diagnostic-only (does not feed `latent_structure_and_robustness`).

Perturbations (the same set under both families):

- **`drop_and_add|fraction=0.0 / 0.02 / 0.05`** — drop a fraction of spikes and re-add
  an equal expected fraction elsewhere (rate-preserving). `fraction=0.0` is a no-op
  baseline.
- **`drop_neurons|fraction=0.0 / 0.15`** — overwrite a fraction of units' spike trains
  with copies of randomly chosen surviving units (donors sampled with replacement),
  destroying the dropped units' identity while keeping real traces. `0.0` is the
  baseline.
- **`jitter|k_ms=0.0 / 1.0 / 3.0|p=0.2`** — each spike, with probability 0.2, is
  temporally jittered by a draw of scale `k_ms`. `k_ms=0.0` is the baseline.
- **`permute_neurons`** — randomly permute the unit axis (tests unit-identity
  sensitivity).
- **`window_shift|max_shift_ms=0.0 / 1.0 / 3.0 / 5.0`** — crop the analysis window
  shifted by a draw uniform in `[−max_shift_ms, +max_shift_ms]`. `0.0` is the centered
  baseline.

The `=0.0` entries are zero-strength baselines (divergence ≈ 0).

---

## Block: `encoding`

Downstream **stimulus-encoding** run (stimulus → neural activity): a Cowley-V4 /
Wingert encoding model trained on the frozen tokenizer's latents, evaluated per neuron
then aggregated. All five metrics are higher-better.

- **`encoding/nc_r2`** — **noise-ceiling-corrected R²** (Pospisil & Bair ER estimator):
  the unbiased fraction of **explainable** (across-repeat) stimulus-driven variance the
  model captures. Range `(-∞, 1]` (can be negative; can slightly exceed 1 from
  estimation noise; NaN if the denominator ≈ 0). **Higher is better.** Source of the
  `downstream_encoding` composite.
- **`encoding/rate_pearson`** — cross-image Pearson r between predicted and observed
  **time-summed** rates per neuron, on the single-trial loader. Range `[-1, 1]`.
  **Higher is better.**
- **`encoding/temp_pearson`** — per-(image, neuron) **temporal** Pearson r between
  predicted and observed time-courses, averaged over images. Range `[-1, 1]`.
  **Higher is better.**
- **`encoding/rate_pearson_psth`** — `rate_pearson` on the repeat-averaged (PSTH)
  responses. Range `[-1, 1]`. **Higher is better.**
- **`encoding/temp_pearson_psth`** — `temp_pearson` on the repeat-averaged (PSTH)
  responses. Range `[-1, 1]`. **Higher is better.**

`rate_*` correlate across stimuli on time-collapsed rates (tuning fidelity); `temp_*`
correlate within a stimulus across time (temporal-dynamics fidelity). The `_psth`
variants use the mean response across repeats of an image, removing trial noise from
the target.

---

## Block: `downstream_decoding`

In-process behavioral decoding probes (neural latent → behavior): lightweight Ridge
probes run inside the recon-eval pipeline. All higher-better. The two
`behavioral_regression` R²s are merged with the POYO `test_r2` into the `decoding`
composite; the two `behavioral_classification` accuracies are not (accuracy, not R²).

- **`downstream_decoding/behavioral_regression_cursor_pos`** — RidgeCV regression R²
  from chunked latents → cursor position, trial-aware GroupKFold, mean R² across folds.
  Range `(-∞, 1]` (negative = worse than predicting the mean). **Higher is better.**
- **`downstream_decoding/behavioral_regression_cursor_vel`** — same, target cursor
  velocity. Range `(-∞, 1]`. **Higher is better.**
- **`downstream_decoding/behavioral_classification_trial_target_id`** — "Linear
  decoding: trial-target accuracy": RidgeClassifierCV accuracy from chunked latents → discrete reach target id,
  StratifiedGroupKFold, mean accuracy across folds. Range `[0, 1]`
  (chance ≈ 1/n_classes). **Higher is better.**
- **`downstream_decoding/behavioral_classification_movement_phase`** — same, target
  movement phase (inverse-frequency class balancing). Range `[0, 1]`.
  **Higher is better.**

---

## Block: `decoding`

- **`decoding/test_r2`** — "POYO decoding (cursor-velocity R²)": held-out **test R²** of
  a **POYO decoder** trained on the frozen tokenizer's latents to decode continuous
  behavior (default cursor velocity), meaned across sessions. This is the separate POYO
  decoding run, distinct from the in-process `downstream_decoding/behavioral_*` Ridge
  probes. Range `(-∞, 1]`. **Higher is better.** One of the two sources of the merged
  `decoding` composite; the same value as the `decoding_poyo` score.
