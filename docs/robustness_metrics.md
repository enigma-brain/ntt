# Robustness Metrics

We use two robustness evaluations for tokenizers:

- `input_robustness` (M1) checks whether small input corruptions move
  the latent representation.
- `latent_robustness` (M2) checks whether small latent perturbations
  degrade decoding.

## Metric definitions

### input_robustness: input-corruption robustness (M1)

For each clean window, we create corrupted replicas of the input, encode all replicas, and compare each corrupted latent sequence `L_k` to the clean latent sequence `L_0`.

Canonical score:

`input_robustness = mean_{t,d} (L_k[t,d] - L_0[t,d])^2 / sigma_token^2`

i.e., normalized MSE between aligned tokens in one window, averaged over windows. Here `sigma_token^2 = mean_d sigma_d^2` is the clean latent **token variance**, estimated from clean replica-0 tokens across the evaluated windows. This is a normalized MSE, so the unit is "latent token variance". Lower is better.

Notes:

- The computation is token-based, so variable numbers of tokens per window are handled naturally.
- The job also records a per-dimension normalized variant, the metric returns two values, one normalizing per-token-dim distances by the per-token-dim variance.

### latent_robustness: latent-corruption robustness (M2)

We start from clean latents and add noise before decoding. Noise is scaled per latent dimension:

- Gaussian: `xi_d ~ N(0, (epsilon * sigma_d)^2)`
- Uniform: `xi_d ~ U(-epsilon * sigma_d, epsilon * sigma_d)`

Here `sigma_d` is the clean standard deviation of latent dimension `d`. So `epsilon` means "fraction of each dimension's natural scale", not a single global latent scale.

We then decode `clean_latents + xi` and evaluate reconstruction metrics, typically against the input spikes (`input_vs_recon`) or against the clean reconstruction (`recon_vs_recon`). Curves that remain at close-to-optimal metric values for higher epsilons in metric-vs-`epsilon` curves mean a more robust decoder.

## Running

Both are Hydra jobs under `src/ntt/jobs/`. Set `paths.run_name=<name>`. For untrained tokenizers, pass `tokenizer@job.tokenizer_config=contrib/<name>` and omit `job.ckpt_path`.

### input_robustness (M1)

```bash
uv run ntt job=eval_robustness_input \
  paths.run_name=robustness_input_tokenizer_demo \
  job.ckpt_path=path/to/checkpoints/last.ckpt \
  job.output_path=results/robustness_input_tokenizer_demo
```

Untrained `cs` example (both scripts, _input and _latent, have the
same main parameters):

```bash
uv run ntt job=eval_robustness_latent \
  paths.run_name=robustness_input_cs_demo \
  tokenizer@job.tokenizer_config=contrib/compressed_sensing \
  job.output_path=results/robustness_input_cs_demo
```

### latent_robustness (M2)

Example with an untrained tokenizer:

```bash
uv run ntt job=eval_robustness_input \
  paths.run_name=robustness_latent_cs_demo \
  tokenizer@job.tokenizer_config=contrib/compressed_sensing \
  job.output_path=results/robustness_latent_cs_demo
```

Default latent_robustness (M2) sweep:

- `epsilons = [0.01, 0.03, 0.1, 0.3, 1.0]`
- `n_samples_per_epsilon = 10`
- `noise_distribution = gaussian`

## Replot CLI

`scripts/replot_robustness.py` rebuilds joint comparison plots from existing run directories.

```bash
# input_robustness (M1): corruption strength vs normalized latent distance
python scripts/replot_robustness.py input-corruption-a \
  results/robustness_input_tokenizer_demo \
  results/robustness_input_cs_demo

# input_robustness (M1): corruption strength vs selected reconstruction metrics
python scripts/replot_robustness.py input-corruption-b \
  results/robustness_input_tokenizer_demo \
  results/robustness_input_cs_demo \
  --metric acg_correlation \
  --metric binned_rate_difference

# Discover all available reconstruction metrics in the CSVs
python scripts/replot_robustness.py input-corruption-b \
  results/robustness_input_tokenizer_demo \
  results/robustness_input_cs_demo \
  --metric ALL

# latent_robustness (M2): epsilon vs reconstruction metrics
python scripts/replot_robustness.py latent-corruption-a \
  results/robustness_latent_tokenizer_demo \
  results/robustness_latent_cs_demo
```

Notes:

- `latent-corruption-a` defaults to `--metric acg_correlation --metric ccg_correlation`
- `input-corruption-b` defaults to `--metric acg_correlation --metric binned_rate_difference`
- `--out-dir` can override the output location for any subcommand

## Poisson normalization (`poisson_comparison.json`)

When `job.poisson_reference=<path/to/poisson_run>` is passed to either M1 or M2, the job loads the Poisson tokenizer's prior eval results from that directory and writes a `poisson_comparison.json` next to the per-session reports. The artifact contains **two** normalized-score flavors per metric, and they answer different questions:

| Field | What it is | Computed how |
|---|---|---|
| `normalized_scores_vs_poisson` | "If I aggregate everything first, how much better is the model than Poisson?" | Aggregate model / Poisson / ceiling reports across sessions, then normalize the aggregates with `compute_normalized_comparisons`. **One number per metric.** |
| `normalized_scores_from_per_session_mean` | "On the average session, how much better is the model than Poisson?" | Normalize per session (against that session's Poisson and zero-strength ceiling), then take the unweighted mean across sessions. **One number per metric.** |

The two collapse to the same value when every session contributes equally and the metric is linear in the underlying counts; they diverge when sessions differ in size, when the metric is nonlinear, or when only some sessions overlap between the model and Poisson runs. Both are kept because each has cleaner semantics for a different use case:

- Use `normalized_scores_vs_poisson` when you want a single bottom-line score for the whole dataset.
- Use `normalized_scores_from_per_session_mean` when you want "how does the model compare on a typical session?" — less sensitive to one large session dominating the aggregate.

Note: in the main reconstruction-eval pipeline (`run_validation_evaluation_per_session`), Poisson normalization is computed **aggregate-then-normalize** only — matching the first column above.

### Ceiling per metric

For both M1 and M2, the **ceiling** in the normalization formula is the model's own zero-corruption performance:

- M1 looks up the same metric at `max_shift_ms=0.0` / `k_ms=0.0` / `fraction=0.0` via `_zero_strength_m1_label`.
- M2 looks up the same metric at `epsilon=0` via `_epsilon_metric_key(metric, 0.0)`.

So `0.0` = matches Poisson, `1.0` = matches the zero-corruption ceiling.

### Per-(metric, corruption, strength) granularity

Each `(metric, corruption_type, strength)` bucket is its own metric in the report (e.g. `cs_corr/window_shift|max_shift_ms=10.0`, `acg_correlation/eps_0.1`), so both normalized-score flavors give one value per fully-qualified bucket — no collapse across corruption types or strengths. The `_to_scalar` helper in `comparison.py` will warn if a future caller ever hands in a multi-element array, because that would silently collapse buckets.

### Post-hoc normalization for existing runs

If a model run and a Poisson run were evaluated **independently** (neither was launched with `job.poisson_reference=...`), `src/ntt/evaluation/aggregation/poisson_post_hoc.py` provides read-only helpers to compute the comparison after the fact for plotting / analysis:

- `normalize_m1_results_against_poisson(model_results_dir=…, poisson_results_dir=…, aggregation_order="aggregate_first"|"per_session_then_mean")` — recompute for M1. Reads `per_session/*.json` from each run; does not require an existing `poisson_comparison.json`.
- `normalize_m2_results_against_poisson(…)` — same for M2.
- `compare_snr_against_poisson(…)` — SNR is a ratio that is unbounded above when noise is low, so there is no natural ceiling for `(model − poisson) / (ceiling − poisson)`. This helper returns the **direction-corrected raw delta** `model − poisson` per metric instead. Reads the main eval's `results.csv` (per-segment) and `results.json` (aggregate).

None of these write anywhere; the existing results dirs are never modified. `aggregation_order` defaults to `aggregate_first` (matching the main reconstruction-eval pipeline default); switch to `per_session_then_mean` for the per-session-normalize-then-average flavor. Both the post-hoc helpers and the live M1/M2 job paths aggregate model / poisson / ceiling over the **shared session subset** before comparing, so when sessions overlap perfectly post-hoc results match the live artifact bit-equal; when they don't overlap perfectly, both honestly report the shared-population comparison (and log a warning that some sessions were dropped).
