# Evaluation Pipeline Usage Guide

This guide explains how to run non-downstream evaluations.

Use `job=eval` for reconstruction, behavioral regression and classification, and latent structure-preservation metrics. Use `job=eval_all` to run those evaluations plus SNR, input-corruption robustness, and latent-corruption robustness. Set `job.output_path=results/my_eval` to save results in `results/my_eval`. The additional results go in `results/my_eval_snr`, `results/my_eval_m1` (input corruption), and `results/my_eval_m2` (latent corruption).

To choose sessions, metrics, and metric settings, follow the [custom evaluation guide](guides/custom_evaluation.md).

Downstream decoding and encoding run separately. See the [downstream evaluation guide](downstream_eval.md). The [aggregation guide](eval_aggregation.md) describes final aggregation and generalization scores. See [metric definitions](eval_metrics.md) and [robustness metrics](robustness_metrics.md) for details on individual metrics.

## Quick start

Run commands from the repository root. To evaluate a cohort of tokenizers on every axis, follow the [eval-battery runbook](eval_battery_runbook.md); agents use the `ntt-eval-battery` skill in `.claude/skills/`.

### Evaluate one checkpoint

```bash
uv run ntt job=eval_all \
    paths.data_dir=/absolute/path/to/brainsets/preprocessed \
    paths.run_name=my_eval \
    job.ckpt_path=/path/to/model.ckpt \
    job.output_path=results/my_eval
```

Set `paths.data_dir` to the directory that contains the preprocessed brainset folders, such as `perich_miller_population_2018/` and `cowley_v4_compact_2024/`. The directory may be named `processed` or `preprocessed`; use its actual path.

### Evaluate a CSV list

Create `cohort.csv`, or export a spreadsheet tab as CSV:

```csv
tokenizer,ckpt_uri,tokenizer_config
model_a,/absolute/path/model.ckpt,
model_b,mlflow-artifacts:/123/run_id/artifacts/checkpoints/model.ckpt,
poisson-10ms,,contrib/poisson_baseline
cs-100ms,,contrib/compressed_sensing
```

Each row needs a unique `tokenizer` name made only of letters, digits, underscores, or hyphens. Fill exactly one of `ckpt_uri` and `tokenizer_config`. Config names are relative to `config/tokenizer/` and omit `.yaml`. The examples use 10 ms bins for Poisson rate estimation and 100 ms CS windows, with 1 ms CS input bins. To use different settings for a row, create a separate tokenizer config and refer to it in the CSV.

```bash
uv run python scripts/run_eval_cohort.py cohort.csv \
    --data-dir /absolute/path/to/brainsets/preprocessed
```

Rows run one after another, and the script stops if an evaluation fails. Results go to `results/<tokenizer>`. Use `--job eval` for the standalone metric job, `--output-dir` to choose another results directory, and repeated `--override` arguments for settings shared by all rows. See the [custom evaluation guide](guides/custom_evaluation.md) for a regression-only example.

### Save results to MLflow

Export the tracking server address and credentials in the shell that submits the job, or add the exports to your shell startup file:

```bash
export MLFLOW_TRACKING_URI=https://your-mlflow-server
export MLFLOW_TRACKING_USERNAME=your-username
export MLFLOW_TRACKING_PASSWORD=your-password
```

A username and password are needed only if the server uses that authentication method. To log each CSV evaluation, run:

```bash
uv run python scripts/run_eval_cohort.py cohort.csv \
    --data-dir /absolute/path/to/brainsets/preprocessed \
    --override job.mlflow_logging.enabled=true \
    --override job.mlflow_logging.experiment_name=ntt-eval
```

For one evaluation, add `job.mlflow_logging.enabled=true` and `job.mlflow_logging.experiment_name=ntt-eval` to its command.

## Run standalone metrics

```bash
# Mode 1: Trained tokenizer from checkpoint
uv run ntt job=eval \
  paths.run_name=my_eval \
  job.ckpt_path=/path/to/model.ckpt \
  job.output_path=results/eval_output

# Mode 2: Untrained tokenizer from spec (e.g. Poisson baseline)
uv run ntt job=eval \
  paths.run_name=poisson_eval \
  tokenizer@job.tokenizer_config=contrib/poisson_baseline \
  job.output_path=results/eval_output

# Mode 2b: Untrained Compressed Sensing tokenizer + Poisson comparison + plots
# (~2 min end-to-end on the ntt_debug sandbox; no checkpoint required)
uv run ntt job=eval \
  paths.run_name=cs_baseline \
  tokenizer@job.tokenizer_config=contrib/compressed_sensing \
  data@job.data_config=ntt_debug \
  job.compare_poisson=true \
  job.generate_plots=true \
  job.output_path=results/cs_baseline

# Full evaluation with different data + Poisson comparison + plots:
uv run ntt job=eval \
  paths.run_name=my_eval \
  job.ckpt_path=/path/to/model.ckpt \
  data@job.data_config=ntt_small \
  job.output_path=results/eval_output \
  job.compare_poisson=true \
  job.generate_plots=true
```


## What the eval job provides

`NTTEvalJob` (invoked via `uv run ntt job=eval ...`) does:

1. **Loads** a tokenizer from a Lightning `.ckpt` file **or** creates one from a spec
2. **Runs** all evaluation metrics per validation session
3. **Aggregates** results (mean/median/weighted_mean across sessions)
4. **Prints** a summary table to stdout
5. **Saves** JSON + CSV results into an output directory (when `output_path` is set)
6. **Compares** against a Poisson baseline (when `compare_poisson=true`)
7. **Generates** reconstruction and comparison plots (when `generate_plots=true`)

### CLI arguments

| Argument | Default | Description |
|---|---|---|
| `paths.run_name` | required | Name for this eval run (used for output directory) |
| `job.ckpt_path` | `null` | Path to `.ckpt` file (one required) |
| `tokenizer@job.tokenizer_config` | `null` | Tokenizer spec for untrained tokenizer (one required) |
| `data@job.data_config` | `ntt_eval_light` | Data config — see [Hydra overrides](hydra_overrides.md) for syntax |
| `evaluation@job.eval_config` | `hill_climb` | Metric config (usually leave as default); `hill_climb` = `default` plus metrics at several bin sizes, same as `eval_all`'s main pass |
| `job.output_path` | `null` | Directory where eval artifacts are saved |
| `job.device` | `auto` | `auto`, `cpu`, `cuda`, `cuda:0` |
| `job.split` | `val` | `val` or `test` |
| `job.aggregation_strategy` | `mean` | `mean`, `median`, `weighted_mean` |
| `job.seed` | `42` | Random seed |
| `job.compare_poisson` | `false` | Run Poisson baseline and compute normalized comparison scores |
| `job.poisson_rate_bin_ms` | `null` | Rate estimation bin size for the Poisson comparison (ms). Without an override it matches the tokenizer's token duration |
| `job.poisson_seed` | `42` | Random seed for Poisson sampling |
| `job.generate_plots` | `false` | Save reconstruction + comparison plots as PNGs |
| `job.plot_seconds` | `600.0` | Seconds of data to use for reconstruction plots |
| `job.mlflow_logging.enabled` | `false` | Log eval results (metrics, tags, JSON artifacts, **and the model checkpoint**) to MLflow |
| `job.mlflow_logging.experiment_name` | `"ntt-eval"` | MLflow experiment to log under |
| `job.mlflow_logging.always_copy_checkpoint` | `false` | When the source ckpt is already an `mlflow-artifacts:/` URI, also copy its bytes into the eval run's artifacts. Default leaves them where they are (linked via `ntt.eval.training_run_id` tag). |

> **Note:** `job.ckpt_path` and `job.tokenizer_config` are mutually exclusive — provide
> exactly one. Use `job.ckpt_path` for trained tokenizers (the spec is extracted from the
> checkpoint). Use `tokenizer@job.tokenizer_config` for untrained tokenizers (e.g.
> `contrib/poisson_baseline`, `contrib/identity`). MLflow artifact URIs are supported
> for `job.ckpt_path` (e.g. `mlflow-artifacts:/...`).

### MLflow checkpoint upload

When `job.mlflow_logging.enabled=true`, `job=eval` and the main pass of `job=eval_all` upload a
Lightning-compatible `.ckpt` as an artifact of the eval run at
`checkpoints/<name>.ckpt`. The artifact can be reloaded via
`load_tokenizer_from_checkpoint("mlflow-artifacts:/<eval_run_id>/artifacts/checkpoints/<name>.ckpt")`.

Behavior by source:

| Source | Default behavior | With `always_copy_checkpoint=true` |
|---|---|---|
| `ckpt_path=mlflow-artifacts:/...` | Tag-only (`ntt.eval.training_run_id` links training run). | Bytes re-uploaded into the eval run. |
| `ckpt_path=/local/path.ckpt` | Uploaded as-is (filename preserved). | Same. |
| `tokenizer_config=<spec>` (no ckpt) | Synthesized from the live tokenizer → `checkpoints/model.ckpt`. | Same. |

The synthesized-for-spec-only ckpt captures the spec in
`hyper_parameters.tokenizer_spec` and the tokenizer weights under
`state_dict`, so it round-trips through the standard loader.

The SNR, input-robustness and latent-robustness passes of `job=eval_all` never upload a
checkpoint. `job.mlflow_logging.upload_checkpoint=false` skips the upload in either job.

When `job.output_path` is set, the eval job treats it as an output directory and writes:

```text
<output_path>/
  results.json
  results.csv
  improvement_radar.png        # when compare_poisson=true
  metric_bars.png              # when compare_poisson=true
  sessions/                    # when generate_plots=true
    <brainset>/
      <session_name>/
        per_unit_raster.png
        population_raster.png
        firing_rate.png
        acg.png
```

### Generalization retention score

After an eval run has produced per-session reports, compute split-wise
generalization retention with:

```bash
uv run python scripts/compute_generalization_score.py \
  mlflow-artifacts:/<experiment_id>/<run_id>/artifacts/per_session \
  --output-dir outputs/generalization_score
```

The input can also be a local `per_session/` directory. The script reads the
current `config/data/selections/composite/ntt_full_*` split definitions and
assigns sessions to:

- `xtime`: held-out time from held-in sessions
- `xsession`: held-out sessions from held-in subjects/sources
- `xsubject`: held-out subjects from held-in sources
- `xsource`: held-out data sources

Use `split=val` while you are still developing, and only switch to `split=test` for the final reported numbers. The held-out sessions are defined in `config/data/selections/composite/ntt_full_{val,test}_*` (see the README section "Selecting a Dataset").

The score asks whether the tokenizer's reconstruction holds up, so it uses the
reconstruction metrics only: the `latent_quality` metrics (signed correlations near
zero) and the `downstream_*` metrics (only some datasets have them) are left out.
For each such metric present in both `xtime` and a held-out tier, it compares every
held-out session with the `xtime` session(s) of its own brainset, using
direction-aware retention:

- higher-is-better metrics: `session_value / xtime_value`
- lower-is-better metrics: `xtime_value / session_value`

Each session's retention is clipped to `[0, 1]` by default, so an improvement
over `xtime` counts as full retention but does not compensate for degradation
elsewhere. A tier's retention for the metric is the mean over its sessions.

Matching by brainset keeps the comparison within one recording setup and
firing-rate range: most metrics scale with firing rate, and the tiers hold
different brainsets, so a pooled baseline would score the tiers' dataset mix
rather than the distribution shift. A session whose brainset has no `xtime`
session (`xsource`) is compared with the mean over all `xtime` sessions. A
session is left out when its `xtime` value is not positive: a ratio to 0 is
undefined, and a ratio to a negative correlation inverts (a worse session would
score higher). It is also left out when its brainset's `xtime` sessions lack the
metric.
`--baseline-mode tier_mean` instead compares the tier's mean value with the mean
over all `xtime` sessions. The global
`generalization_score` is the geometric mean across the `xsource` retention terms
only (`score_tiers`): on held-out sessions and subjects of the training datasets,
reconstruction does not measurably degrade and a model's retention does not carry
over from one session to another, so those tiers are reported per tier and per
metric but left out of the score. A small retention floor, `1e-3` by default, is applied only inside the
geometric aggregation so one exactly-zero term does not collapse the whole score
to zero. The per-metric `retention` values remain unfloored in the output table.

Outputs:

```text
<output-dir>/
  session_tiers.csv              # per report file: matched tier/excluded/unmatched
  tier_metric_values.csv         # per-tier mean raw metric values
  metric_retention.csv           # direction-aware per-metric retention terms
  generalization_summary.json    # machine-readable summary and metadata
  generalization_summary.md      # compact human-readable summary
```

Useful options:

| Option | Default | Purpose |
|---|---:|---|
| `--config-root` | `config` | Root containing `data/selections/composite` |
| `--selection-family` | `ntt_full` | Selection YAML family to use |
| `--split` | `val` | Eval split, `val` or `test` |
| `--baseline-tier` | `xtime` | Reference tier for retention ratios |
| `--baseline-mode` | `brainset_matched` | `brainset_matched` or `tier_mean` (see above) |
| `--retention-floor` | `1e-3` | Floor used only for geometric aggregation |
| `--no-clip-improvements` | `false` | Allow retention values above 1 |
| `--exclude-brainset` | `tostado_marcos_birdsong_2024` | Exclude historical datasets from scoring |

### Examples

```bash
# Trained tokenizer (checkpoint mode)
uv run ntt job=eval \
  paths.run_name=ldns_eval \
  job.ckpt_path=/path/to/model.ckpt \
  job.output_path=results/ldns_eval \
  job.device=auto \
  job.split=val

# Untrained Poisson baseline (spec mode)
uv run ntt job=eval \
  paths.run_name=poisson_eval \
  tokenizer@job.tokenizer_config=contrib/poisson_baseline \
  job.output_path=results/poisson_eval

# Untrained Compressed Sensing baseline (spec mode, with comparison + plots)
uv run ntt job=eval \
  paths.run_name=cs_baseline \
  tokenizer@job.tokenizer_config=contrib/compressed_sensing \
  data@job.data_config=ntt_debug \
  job.compare_poisson=true \
  job.generate_plots=true \
  job.output_path=results/cs_baseline
```

> **Note on sandbox-only behavioural skips:** with the small `ntt_debug` data config, some
> behavioural decoders can be skipped (e.g. `trial_result` needs both successful and
> unsuccessful trials). Regression folds split trial groups within each session, so the
> number of sessions does not decide whether a metric runs: labels, valid samples and
> chunk sizes do. All five behavioural metrics run on the full public dataset
> configurations.

## Changing the dataset

The default eval data config is `ntt_eval_light` (20 sessions, the same as the main pass of `eval_all`); `ntt_debug` is the small, fast sandbox. To evaluate on more
data, swap the config group — note this uses `group@package=choice` syntax, not dot-path:

```bash
# Use the small (30%) dataset split
uv run ntt job=eval \
  paths.run_name=my_eval \
  job.ckpt_path=/path/to/model.ckpt \
  data@job.data_config=ntt_small

# Use the full production dataset
uv run ntt job=eval \
  paths.run_name=my_eval \
  job.ckpt_path=/path/to/model.ckpt \
  data@job.data_config=ntt_full
```

Available data configs: `ntt_debug`, `ntt_small`, `ntt_medium`, `ntt_full`.

> **Common mistake:** `++job.data_config=ntt_small` will NOT work — it sets the field to
> the string `"ntt_small"` instead of composing the YAML file. You must use the
> `data@job.data_config=...` syntax. See [hydra_overrides.md](hydra_overrides.md) for details.

## Running the battery on the test split

`job.split=test` propagates from `job=eval_all` to every pass (main, SNR, M1, M2) and to
`job=poisson_baseline`. The public eval data configs (`ntt_eval_light`, `cowley_snr_fast_eval`,
`cowley_v4_compact_val`, `robustness_fast_eval`) define `test_selections`:

- `ntt_eval_light`: the 7 xtime sessions read through their `test_domain`, plus
  `composite/ntt_eval_light_ood_test` (the same 13 brainset x tier cells as the val OOD block,
  with sessions taken from the `ntt_full_test_*` composites, full recording).
- SNR: `cowley_snr_fast_eval` -> `PE_201014`; `cowley_v4_compact_val` (precision) -> `cowley_v4_compact_2024_test_snr`,
  22 of the 25 `cowley_v4_compact_2024_test` sessions. PE_210303, RA_211025 and WE_190927 are dropped because they have
  stimulus starts < 180 ms apart, which the per-stimulus sampler's 200 ms window (20 ms padding cap) rejects.
- M1/M2: `robustness_fast_eval` already consists of test-composite sessions.

```bash
uv run ntt job=eval_all job.split=test paths.run_name=my_test_eval job.ckpt_path=/path/to/model.ckpt
```

Notes:
- The Poisson baseline must be computed on the same split (`job=poisson_baseline job.split=test`);
  a val-split baseline is not comparable to test-split results.
- When aggregating, set `job.gen_split=test` (aggregation job). Tier classification looks sessions up in
  `composite/ntt_full_test_*`, so out-of-distribution test sessions must come from those composites,
  otherwise they are silently dropped from the generalization score.

## Available metrics
For three additional latent quality metrics, see separate section below.

The main evaluation pipeline can run the following metrics. This list leaves out the behavioral metrics (`behavioral_regression`, `behavioral_classification`) and `token_dims_per_second`; see [`eval_metrics.md`](eval_metrics.md) for all of them. The default `hill_climb` metric config used by `job=eval` and the main pass of `job=eval_all` adds several of these metrics at more bin sizes.

### Spike detection
- `spike_detection_f1` — F1 for matched spikes (1ms tolerance)
- `temporal_precision` — median timing error (ms) for matched spikes
- `spike_count_correlation` — correlation of binned counts across timescales
- `smoothed_squared_error` — per-unit SSE of Gaussian-smoothed spike trains
- `binned_rate_difference` — per-unit mean abs difference in binned rates (Hz)

### Single unit
- `isi_divergence` — Wasserstein distance between ISI distributions
- `acg_correlation` — Pearson correlation of autocorrelograms
- `isi_cv_error` — absolute error in ISI coefficient of variation
- `isi_cv2_error` — absolute error in CV2 local irregularity
- `firing_rate_error` — per-unit absolute error in mean firing rate (Hz)
- `fano_factor_error` — per-unit absolute error in Fano factor (var/mean)

### Pairwise
- `ccg_correlation` — Pearson correlation of cross-correlograms
- `functional_connectivity_f1` — F1 for detecting functionally connected pairs

### Population
- `participation_ratio_error` — absolute error in participation ratio
- `subspace_overlap` — overlap between top PCA subspaces
- `population_coupling_preservation` — correlation of per-unit population coupling

### Information
- `mutual_information` — MI between original and reconstructed counts

### Structure preservation
- `population_structure_preservation` — RDM correlation of population activity patterns
- `per_unit_structure_preservation` — kNN preservation of per-unit response profiles

## Full evaluation with plots + Poisson comparison

The eval job supports Poisson comparison and plot generation via CLI flags:

```bash
uv run ntt job=eval \
  paths.run_name=full_eval \
  job.ckpt_path=/path/to/model.ckpt \
  job.output_path=results/eval \
  job.compare_poisson=true \
  job.generate_plots=true \
  job.plot_seconds=600.0
```

For an untrained tokenizer:

```bash
uv run ntt job=eval \
  paths.run_name=full_eval_cs \
  tokenizer@job.tokenizer_config=contrib/compressed_sensing \
  job.output_path=results/eval_cs \
  job.compare_poisson=true \
  job.generate_plots=true \
  job.plot_seconds=600.0
```

With `compare_poisson=true`, `results.json` includes `comparison_vs_poisson`
metadata per metric plus `normalized_scores_vs_poisson`, where `0` means
Poisson-level performance and `1` means perfect-reconstruction performance.
`firing_rate_error` is the exception: its entry is scored against a population-mean-rate
reference (`baseline_kind: "population_mean_rate"`, score `1 − E / R`; see
[`metric_glossary.md`](metric_glossary.md#composite-scores-the-7-radar-axes)), keeps the
Poisson floor value in `poisson_baseline_value`, and each session's reference error is saved
as `population_mean_rate_error_hz` in that session's `firing_rate_error` metadata.
With `generate_plots=true`, PNG files are saved into the same output directory,
and per-session reconstruction plots are organized under `<output_path>/sessions/`.

### Re-plotting results jointly across multiple tokenizers

```bash
uv run python scripts/replot_from_eval_results.py \
  results/tokenizer_A/ \
  results/tokenizer_B/ \
  --output-dir results/comparison_plots
```

If the source `results.json` files were created with
`job.compare_poisson=true`, the replotter automatically includes the embedded
Poisson baseline from the first such run. You do not need a separate Poisson
directory for the standard eval plots.

When the passed dirs come from `job=eval_all`, the
replotter also looks for sibling robustness dirs named `<run>_m1` and
`<run>_m2`. If found, it additionally writes the joint M1-a and M2-a plots
under `<output_dir>/robustness/`.

### Building blocks for custom scripts

You can also use the building blocks directly in a script:

```python
# Load tokenizer
from ntt.core.model import load_tokenizer_from_checkpoint
tokenizer = load_tokenizer_from_checkpoint("/path/to/model.ckpt", map_location="cpu")

# Run evaluation per session
from ntt.evaluation.utils import (
    run_validation_evaluation_per_session,
    aggregate_session_reports,
)
session_reports = run_validation_evaluation_per_session(
    val_loader, tokenizer.reconstruct_from_input,
    data_type=spec.input_type, bin_size_seconds=...,
    eval_config=EvaluationConfig(), device=device,
)
aggregated = aggregate_session_reports(session_reports)

# Poisson baseline (per-segment)
from ntt.contrib.tokenizer.poisson_baseline import poisson_baseline_reconstruction
from ntt.evaluation.pipeline import EvaluationPipeline
from ntt.evaluation.aggregation.comparison import compute_normalized_comparisons
pipeline = EvaluationPipeline()
poisson_recon = poisson_baseline_reconstruction(
    original_np, bin_size_ms=..., rate_estimation_bin_ms=..., ...
)
poisson_report = pipeline.run(original_data, poisson_data)
perfect_report = pipeline.run(original_data, original_data)
comparisons = compute_normalized_comparisons(model_report, poisson_report, perfect_report)

# Reconstruction visualization
from ntt.evaluation.plotting import (
    get_validation_segment,        # load a segment for plotting
    plot_per_unit_raster,          # per-unit original vs reconstructed
    plot_population_raster,        # population raster (side by side)
    plot_firing_rate_comparison,   # firing rate scatter
    plot_acg_comparison,           # autocorrelogram grid
    plot_reconstruction_summary,   # all four above at once
)
segment, bin_size_seconds = get_validation_segment(tokenizer, datamodule, device=device)
figs = plot_reconstruction_summary(segment, bin_size_seconds=bin_size_seconds)

# Model comparison visualization
from ntt.evaluation.visualization import (
    plot_metric_dumbbell,          # raw model vs baseline comparison
    plot_metric_bars,              # grouped bar chart
    plot_metric_comparison,        # box plot for a single metric
    plot_normalized_comparison_radar,  # radar: 0 = Poisson, 1 = perfect
    plot_structure_preservation,   # RDM + kNN preservation
)
```



## Latent-quality metrics (RepeatsSNR, input corruption robustness, latent corruption robustness)

These three metrics use their own evaluation passes. The recommended combined
entrypoint is `job=eval_all`, which runs the standard eval, RepeatsSNR,
input_robustness (M1), and latent_robustness (M2) sequentially. You can switch
each pass on or off and override its data / evaluation config from the CLI.

```bash
uv run ntt job=eval_all \
  paths.run_name=my_eval \
  job.ckpt_path=/path/to/model.ckpt \
  job.output_path=results/my_eval
# defaults to `job.mode=fast`. Both modes use the robustness YAML settings
#   (300 s of data per session for input-corruption, 300 windows for
#   latent-corruption); the modes differ in the SNR data they select:
#   a single short Cowley session in fast mode, the full Cowley validation
#   set with `job.mode=precision`
```

Or, for tokenizers that do not need training:
```bash
uv run ntt job=eval_all \
  paths.run_name=my_eval \
  tokenizer@job.tokenizer_config=contrib/poisson_baseline \
  'job.poisson_rate_bin_ms=100.0' \
  job.output_path=results/my_eval
```

`job.compare_poisson=true` only adds a Poisson baseline comparison **to the
main eval pass and the SNR pass**. The two robustness metrics do not compute
an extra Poisson baseline internally. To get Poisson robustness curves, run
the robustness
pass with the Poisson tokenizer itself and set the bin size explicitly:

```bash
uv run ntt job=eval_all \
  paths.run_name=poisson_robustness_100ms \
  tokenizer@job.tokenizer_config=contrib/poisson_baseline \
  '++job.tokenizer_config.rate_estimation_bin_ms=100.0' \
  job.main.enabled=false \
  job.output_path=results/poisson_robustness_100ms
```

You can also run just one robustness pass the same way, for example
`job.latent_robustness.enabled=false` to keep only input-corruption robustness
(M1), or `job.input_robustness.enabled=false` to keep only latent-corruption
robustness (M2).

`RepeatsSNR` needs repeats-containing data and so far only runs on Cowley.
The combined job accepts the same common tokenizer / checkpoint / output flags
as `job=eval`, plus pass-specific config fields:

| Flag | Default | Purpose |
|---|---|---|
| `job.mode=fast` / `precision` | `fast` | Shared runtime preset for SNR / input_robustness / latent_robustness |
| `job.main.enabled=false` | `true` | Skip the default-metrics pass |
| `job.snr.enabled=false` | `true` | Skip the SNR pass |
| `job.input_robustness.enabled=false` | `true` | Skip the input-corruption robustness (M1) pass |
| `job.latent_robustness.enabled=false` | `true` | Skip the latent-corruption robustness (M2) pass |
| `job.snr.evaluation=<name>` | `pipelines/cowley_per_stimulus` | SNR-pass evaluation config |
| `job.snr.data=<name>` | mode-dependent | SNR-pass data config |
| `job.input_robustness.evaluation=<name>` | `metrics/robustness_input` | Input-corruption robustness (M1) evaluation config |
| `job.input_robustness.data=<name>` | `robustness_fast_eval` | Input-corruption robustness (M1) data config |
| `job.latent_robustness.evaluation=<name>` | `metrics/robustness_latent` | Latent-corruption robustness (M2) evaluation config |
| `job.latent_robustness.data=<name>` | `robustness_fast_eval` | Latent-corruption robustness (M2) data config |

`robustness_fast_eval` covers 4 held-out sessions from the test composites.

The passes share the tokenizer; when `job.output_path` is set, the extra passes
write to sibling directories like `<output_path>_snr`, `<output_path>_m1`, and
`<output_path>_m2`.


### Re-plotting

To compare multiple tokenizers on a single figure (or re-render after editing
plot code without re-running the eval), use `scripts/replot_robustness.py`. It
reads the `results.csv` files written by the input_robustness (M1) /
latent_robustness (M2) jobs and produces multi-tokenizer overlay curves. See
`docs/robustness_metrics.md` (Replotting) for the
subcommand list and example invocations. Per-window rasters
(`input-corruption-c_*`, `latent-corruption-b_*`, the annotated raster file)
are eval-time only — they need raw spike data which the eval doesn't persist;
re-run the relevant job with `plot_examples=true` to regenerate.


## Loading tokenizers from checkpoints

Two options:

```python
# Option 1: Via Lightning (full, standard)
from ntt.core.model import load_tokenizer_from_checkpoint
tokenizer = load_tokenizer_from_checkpoint("/path/to/model.ckpt")

# Option 2: Lightweight (no Lightning overhead)
from ntt.core.model.tokenizer import load_tokenizer_from_checkpoint_no_lightning
tokenizer = load_tokenizer_from_checkpoint_no_lightning("/path/to/model.ckpt")
```

Both return a `BaseTokenizer` (nn.Module) ready for `.eval()` and inference.
