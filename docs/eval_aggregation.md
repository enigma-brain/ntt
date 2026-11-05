# Evaluation aggregation: summary scores and tables for many tokenizers

## What this does

Reads each tokenizer's row from the input table (a local CSV file; a Google Sheet is also supported) → downloads evaluation files from the MLflow runs linked in that row → computes seven summary scores → writes one JSON file per tokenizer → combines all tokenizers into CSV tables → optionally makes plots or a LaTeX table.

## How to run

**Input: a tracker table.** One row per tokenizer, as a local CSV file with the columns listed below; other columns are ignored. A Google Sheet with the same columns is an optional alternative: pass its link instead of the CSV path (copy the link while the right tab is open, so that it contains `gid=...`; otherwise the script reads the first tab; the Sheet must be readable through its link without signing in). Write the table by hand or with `scripts/eval/build_cohort.py` (see [the eval-battery runbook](eval_battery_runbook.md#6-the-cohort-table)).

**Columns** (each link is an MLflow run link):

| Column | Needed? | Used for |
|---|---|---|
| `tokenizer` | required | name of the row; the tokenizer family is guessed from it |
| `mlflow_recon` | required | reconstruction results; also gives compression and generalization |
| `mlflow_latent_input_corruption` | optional | input-corruption robustness |
| `mlflow_latent_latent_corruption` | optional | latent-corruption robustness |
| `mlflow_encoding` | optional | downstream encoding: the encoding run |
| `mlflow_decoding` | optional | downstream decoding: the benchmark's parent run for this tokenizer |
| `mlflow_latent_snr` | optional | repeat SNR; goes into the wide raw-metrics table only, not into the seven scores |

**Local outputs instead of MLflow links.** Any of these cells may hold a local path instead of a run link, so a tracker can be scored without MLflow access, and rows with links and rows with paths can share one table. A cell is either the report file itself or a directory, which is searched for the file below (the directory itself first, otherwise exactly one match somewhere below it; several matches are an error, so point the cell at one). Relative paths are resolved against the folder of the CSV (against the current directory when the tracker is a Google Sheet link). Placeholder cells (`-`, `n/a`, `none`, `TODO`, …) count as empty. Files are read in place, and the scores record the absolute path as the source. A row with a local cell is scored again on every run, because re-running an evaluation into the same folder changes its files but not its path. For the published tokenizers, `scripts/eval/published_commands.py --tracker <dir>` writes such a table from finished outputs (see [`pretrained_models.md`](pretrained_models.md) §2).

| Column | Directory is searched for |
|---|---|
| `mlflow_recon` | `results.json` (with `run_meta.json` beside it) |
| `mlflow_latent_input_corruption`, `mlflow_latent_latent_corruption` | `eval_report.json` (with `results_per_session.csv` and `run_meta.json` beside it) |
| `mlflow_latent_snr` | `eval_report.json`, else `results.json` (what the SNR pass writes) |
| `mlflow_encoding` | `downstream_encoding_results.json` |
| `mlflow_decoding` | `downstream_decoding_results.json`, the benchmark's cohort record: the file itself, or the `decoding/<run>/<run>__cohort/` folder of a job's output (the per-cell records beside it are not it) |

The reconstruction results must already contain the Poisson baseline comparison (`comparison_vs_poisson`); otherwise that tokenizer fails.

The `run_meta.json` next to a report supplies what the report does not save itself: the input geometry (bin size, window) of a recon `results.json` whose tokenizer config was not serialized, the run settings of a robustness `eval_report.json` without them in its `config`, and the window settings that `scored_window_seconds` is derived from.

**The scored window.** Every report must carry the window the model was scored on (`scored_window_seconds`), because it is compared across tokenizers. A report that does not record it gets it derived from its window settings (the `eval_window_seconds`, `neuron_selection`, `evaluate_all_neurons` and tokenizer window in its config or `run_meta.json`): the all-neuron path scored `eval_window_seconds`, the padded path scored the tokenizer's own window, so there the two have to be the same number. If the settings do not determine the window, it stays unknown and the comparability check reports that it could not compare it; rerun the eval that wrote the report to record it.

**Before running:**
- Set `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` (for example in `~/.bashrc`). Not needed when every cell is a local path.
- Downloaded evaluation files (about 25 MB per tokenizer) are kept in `<outdir>/../artifact_cache`, so a second run does not download them again.

**Run** from the repository root:

```bash
uv run python scripts/analysis/build_aggregation_tables.py \
    --sheet <tracker.csv> \
    --outdir <outdir>  --gen-split test  --limit 2
```

`--limit 2` scores only two tokenizers, for a first test. For a new `--outdir` in a different folder, pass `--cache <other-outdir>/../artifact_cache` to reuse the downloaded files.

For test-split evals, pass `--gen-split test`. When a reconstruction report has no precomputed generalization score, the script rebuilds it from per-session results on the `--gen-split` split. The default `val` does not match a test-split eval.

**Time:** the first run takes about 9 minutes per tokenizer, mostly to download files. That estimate was measured on 3 tokenizers, so 48 tokenizers would take about 7 hours. Later runs reuse the downloaded files and take seconds. A second run skips tokenizers it has already processed, unless their links in the input table have changed.

**When it stops with an error:** the script writes the tables, then stops if tokenizers were evaluated differently (for example on different sessions), or if it cannot verify that all tokenizers were evaluated the same way. If some tokenizers are missing score parts that others have (input-corruption robustness, latent-corruption robustness, POYO decoding, or ridge decoding), the script does not stop: it leaves that score empty for the tokenizers missing the part and prints a warning. The `emptied_scores` column names the score and the missing part. With `--common-parts-only`, it instead leaves missing parts out of every tokenizer's score and reports the differences and the unverifiable cases as warnings. All tokenizers remain in the table.

## Output files

Each run writes the tables, one JSON file per tokenizer under `<outdir>/scores/`, a run log and the table checks to `<outdir>`. [Outputs and what each is for](#outputs-and-what-each-is-for) lists every file.

## The seven summary scores

- `single_neuron_recon`: how well single-neuron activity is reconstructed (temporal fidelity, single-neuron statistics), compared with a Poisson baseline. Metrics measured at several bin sizes or strengths share one weight.
- `population_structure`: same as above but for population-structure metrics.
- `latent_structure_and_robustness`: mean of up to four parts, each between 0 and 1: two structure-preservation correlations (negative values set to 0), input-corruption robustness as `1/(1+d)` where `d` is the mean latent error under three corruptions, and latent-corruption robustness as the autocorrelogram correlation at noise level ε = 0.1 (negative set to 0).
- `decoding`: POYO test R² averaged over sessions and ridge R² averaged over targets, then the mean of those two. Ridge always leaves out `perich_miller_population_2018/c_20131029_random_target_reaching`, whose test data come in pieces of about 3 seconds. Mixed ridge chunk sizes stop the build, except for velocity with 1000 ms alongside 200 ms (warning). Can be negative. Empty when the decoding benchmark is missing sessions.
  Position regression counts only when the tokenizer has a 200 ms position score. Otherwise it is left out, and the LaTeX table marks the score with an asterisk if another tokenizer in the table includes it.
  Labelled "Decoding (POYO + linear)". Its two parts are saved next to it as `decoding_poyo` ("POYO decoding (cursor-velocity R²)", the POYO mean over sessions, equal to the wide table's `decoding/test_r2`) and `decoding_linear` ("Linear decoding (ridge R²)", the ridge mean over targets); `decoding` is their mean. They are not composite scores: they are not radar axes and do not count toward a tokenizer being complete. When `decoding` is left empty because another tokenizer's score has a part this one lacks, the part this one has stays filled in.
- `downstream_encoding`: mean noise-ceiling R² of the encoding models on the test split. Can be negative.
- `generalization`: between 0.001 and 1. Reconstruction metrics only. Datasets unseen in training (`xsource`) only. For each metric and held-out session: the session's value ÷ the mean within-time value of the training datasets (flipped when lower is better), kept between 0 and 1, then averaged over the tier's sessions. The score is the geometric mean of these ratios over the metrics, each at least 0.001. Held-out sessions and subjects of the training datasets are reported per tier but not scored: they show no measurable loss. See the [metric glossary](metric_glossary.md) for details.
- `compression`: input bits ÷ output bits; no upper limit.

The scores are on different scales. Compare tokenizers only within the same score.

## Plotting and table tools

Four separate scripts. Each reads `--scores-dir <outdir>/scores` and the input table as a CSV file (for tokenizer names). Without `-o`, plots open in a window.

| Script | What it creates | Extra inputs | Output |
|---|---|---|---|
| `uv run python -m scripts.analysis.plot_radar` | radar plot of the seven scores | tokenizer names | `-o` file (.pdf/.png/.svg) |
| `uv run python -m scripts.analysis.plot_family` | one score per tokenizer, grouped by family | a score or metric name | `-o` file |
| `uv run python -m scripts.analysis.plot_correlation` | scatter plot of two scores across tokenizers, with Pearson r | two score or metric names | `-o` file |
| `uv run python -m scripts.analysis.build_latex_table` | LaTeX table of the seven scores, one row per tokenizer | none | `-o` `.tex` file, or printed |

Example: `uv run python -m scripts.analysis.plot_radar table.csv tok_a tok_b --scores-dir <outdir>/scores -o radar.pdf`

---

# Reference

`scripts/analysis/` is the single source of truth for eval aggregation and plotting: how to
turn a tracker sheet of eval runs into the **7 category scores**, the **wide raw-metrics
table**, and the plots and tables built on them.

Two layers:

- **Builder** — [`build_aggregation_tables.py`](#a-one-pass-build--csvs-recommended) (tracker →
  CSVs, one pass; optionally [published to MLflow](#publishing-to-mlflow---log-mlflow)). Also
  [`build_metrics_table.py`](#wide-table-only), the wide-table builder used standalone.
- **Consumers** — the [plotting/table CLIs](#tools), which read one `category_scores.json` per
  tokenizer: `--scores-dir <outdir>/scores`, or the `mlflow_aggregation` column that
  `--log-mlflow` writes into `tracker_with_agg.csv`. Both are the same files.

The CLIs are a package (`scripts.analysis`), so invoke them with
`python -m scripts.analysis.<tool>` from the repo root.

> **Related:** [`docs/tokenizer-radar.md`](tokenizer-radar.md) documents the underlying
> `ntt.evaluation.radar.plot_radar` renderer that `plot_radar` here feeds. This doc covers the
> aggregate-score tooling built on top of `NTTAggregationJob`'s `category_scores.json`.

---

### Generating the tables

There is one entry point, `build_aggregation_tables.py`; MLflow runs are an optional output of
it.

#### A. One-pass build → CSVs (recommended)

`scripts/analysis/build_aggregation_tables.py` reads the tracker directly and produces both
the **wide raw-metrics table** and the **7 category scores** from a single download pass.
It writes plain CSVs; with `--log-mlflow` it also publishes one MLflow run per tokenizer.

```bash
set -a && source .env && set +a

uv run python scripts/analysis/build_aggregation_tables.py \
    --sheet tracker.csv \
    --outdir results/aggregation
```

Run it **from the repo root**. `--sheet` takes the local tracker CSV; it also accepts the link of a Google Sheet tab (`https://docs.google.com/spreadsheets/d/<ID>/edit?gid=<tab-id>`) with the same columns.

| Option | Purpose |
|---|---|
| `--outdir` | where the CSVs land (required) |
| `--cache` | artifact cache dir (default `<outdir>/../artifact_cache`) |
| `--config-root` | Hydra config root for generalization tiers (default: absolute repo `config/`) |
| `--common-parts-only` | leave score parts that only some tokenizers have out of every tokenizer's score and report the differences as warnings, instead of leaving the score empty for the tokenizers missing a part |
| `--compare-m2` | also score a second variant with M2 withheld, and emit the per-tokenizer delta — costs no extra downloads |
| `--limit N` | process at most N new rows — useful for a smoke test |
| `--allow-incomplete-decoding-cohort` | score a decoding cohort with missing sessions over the sessions it has, instead of leaving `decoding` empty |
| `--gen-split {val,test}` | split whose sessions the generalization score is rebuilt from when a recon report has no precomputed `generalization` block (default `val`); use `test` for test-split evals. The split is recorded in `generalization_source` (`recomputed_from_per_session(test)`), so a val/test mix is reported as not comparable, and a resume rescores rows rebuilt on the other split |

Properties worth knowing:

- **Download-once cache.** The `tokenizer|artifact` manifest records the artifact URI's
  identity together with the tracking server. An unchanged source costs no network I/O;
  a changed source is downloaded again into a separate directory. A cached entry is trusted only while its recorded path exists.
- **Resumable + atomic.** Every CSV is rewritten atomically after each row; a re-run skips
  tokenizers already present in **all** outputs, `scores/<slug>.json` included. A row with
  no `scores/<slug>.json` is **re-scored** — never recovered from `_work/`, whose document
  is whatever the scorer wrote when that directory was last touched, so reusing it would
  suppress a re-score with the current scorer. Re-scoring is cheap: the job is
  JSON-in/JSON-out over already-cached artifacts (about 0.2 s per tokenizer). To force a redo of specific tokenizers, delete their rows from the
  output CSVs and re-run — their `scores/<slug>.json` is dropped to match. To force a redo
  of the whole cohort after a scorer change, delete `scores/` (or the CSVs) and re-run.
- **A re-run never shrinks a table.** A record is swapped in only once every output for
  that tokenizer has been recomputed, so a `--limit` stop, an interruption, or a failing
  tokenizer leaves the rows already in the tables untouched.
- **Only `mlflow_recon` is required** per row. Rows without it are skipped; missing
  M1/M2/encoding/decoding artifacts are warned-and-excluded rather than fatal.
- **It feeds the plotting/LaTeX CLIs directly.** Each tokenizer's full
  `category_scores.json` is persisted under `<outdir>/scores/`, so the
  [tools](#tools) below can be pointed at it with `--scores-dir` — no MLflow
  round-trip and no `mlflow_aggregation` column needed. See
  [Feeding the tools from path (A)](#feeding-the-tools-from-path-a).

#### Publishing to MLflow (`--log-mlflow`)

With `--log-mlflow`, after writing the tables and running the table-level checks, the builder
publishes each tokenizer's `scores/<slug>.json` as an MLflow run (artifact `category_scores.json`,
the seven scores as metrics) in `--mlflow-experiment` (default `ntt-aggregation`) on
`MLFLOW_TRACKING_URI`. It then writes `<outdir>/tracker_with_agg.csv`: the tracker rows plus an
`mlflow_aggregation` column with each run's URL. Paste that column into the sheet; nothing writes
to the sheet.

```bash
set -a && source .env && set +a   # MLFLOW_TRACKING_URI and credentials of your own MLflow server

uv run python scripts/analysis/build_aggregation_tables.py \
    --sheet tracker.csv --outdir results/aggregation --log-mlflow
```

- **A table that fails a check is still published.** Every run carries the tag
  `ntt.agg.table_checks` (`passed`, or `failed:<check>,...`) and the table's `table_checks.json`;
  the builder still exits with the first failure afterwards.
- **Idempotent.** A run is identified by the sha256 of everything it carries: tokenizer,
  `category_scores.json`, tags and `table_checks.json` (tag `ntt.agg.publish_sha256`).
  Publishing the same again (a resume, a re-run of an unchanged table)
  reuses the run; a rescored tokenizer gets a new one, and so does every tokenizer when
  `table_checks.json` changes (another verdict, or a check message that changed with the
  table's rows), so a run never shows another build's checks.
- Without `MLFLOW_TRACKING_URI` the builder stops before downloading anything.
- **Publishing an existing build later.** Re-run the builder over (a copy of) its `--outdir`
  with the same `--sheet`, `--cache` and scoring flags (`--gen-split`, `--common-parts-only`,
  `--allow-*`, `--compare-m2`) as the build. A resumed row is published from the file already in
  `scores/`, but some rows are **rescored** with the current code instead: changed tracker links,
  rows with a local path, a different `--gen-split` (e.g. the default `val` on a `--gen-split test` build), and rows a
  `--common-parts-only` build scored with parts left out. So first run it **without**
  `--log-mlflow` and check that the log has no `RESCORE` lines other than `local outputs are always
  re-scored`, and that its `DONE:` line counts only those rows as processed; then add `--log-mlflow`. A read-only output has to be copied first.

#### Wide table only

`scripts/analysis/build_metrics_table.py` builds just the wide raw-metrics table. Path (A)
calls into it, so use this only when you want the wide table on its own.

```bash
uv run python scripts/analysis/build_metrics_table.py \
    tracker.csv --output wide_metrics_table.csv
```

Columns: bare `{block}/{metric}` = mean; `::std` = cross-session sample std (ddof=1, empty if
n<2); `::n` = count behind mean/std; `::norm` = Poisson `normalized_score` (recon metrics with
a valid comparison only; `firing_rate_error` is scored against its population-mean-rate
reference, as in `single_neuron_recon`). Ridge columns leave out `RIDGE_EXCLUDED_SESSIONS`; their `::norm` is
recomputed without those sessions for both the model and the Poisson baseline. Blocks are
`single_unit_statistics/`, `temporal_fidelity/`, `population_structure/`, `robustness/` (M1
input-corruption), `robustness_latent/` (M2 latent-noise), `encoding/`, `decoding/`.

#### Set MLflow credentials (reading artifacts, `--log-mlflow`, the MLflow route)

```bash
set -a && source .env && set +a   # from the repo root
```

#### Feeding the tools from path (A)

Path (A) already computes a full `category_scores.json` per tokenizer and persists it as
`<outdir>/scores/<slug>.json` (slug = the tokenizer name, lowercased, with runs of
non-alphanumerics collapsed to `_`). Point any tool at that directory and it never touches
MLflow:

```bash
uv run python -m scripts.analysis.build_latex_table \
    tracker.csv --scores-dir results/aggregation/scores -o paper_table_agg.tex

uv run python -m scripts.analysis.plot_family \
    tracker.csv decoding --scores-dir results/aggregation/scores
```

The tracker then needs only a `tokenizer` column — no `mlflow_aggregation`, no credentials.
The persisted JSON is the whole document, so per-metric paths
(`sn_recon/spike_detection_f1`, `latent_structure_and_robustness/m2:acg_correlation/eps_0.1`) resolve exactly as on
the MLflow route. A tokenizer the run failed on has no `<slug>.json` and is skipped, which
keeps the plots consistent with `category_scores.csv`.

#### (Optional) score cache for the plotting tools

Pass `--cache-dir /tmp/ntt_scores` to any tool to save downloaded `category_scores.json` files locally.
Files are named `<slug>-<source>.json`, where the source digest includes the artifact
URI and tracking server.
Subsequent runs skip the MLflow download and read from disk instead — useful when iterating on plots.
This applies to the MLflow route only; `--scores-dir` reads from disk already.

---

### Outputs and what each is for

Produced by `build_aggregation_tables.py` (path A) into `--outdir`:

| File | Shape | Contents | Use it for |
|---|---|---|---|
| `category_scores.csv` | tokenizer × 14 | The **7 category scores** and the decoding breakdown (`decoding_poyo`, `decoding_linear`), plus `generalization_source`, `m1_provenance`, `m2_provenance`, `m2_raw_acg`, `m2_metric_key`, `decoding_provenance` (the poyo/ridge family means, `family=mean (n)`, joined by `;`) | **The headline table.** Leaderboards, radar charts, the paper table. This is the one to report. |
| `wide_metrics_table.csv` | tokenizer × ~560 | Every **raw** metric, as `{block}/{metric}` = mean, plus `::std` (cross-session, ddof=1), `::n` (count), `::norm` (Poisson normalized score, recon only) | Metric-level analysis: correlations, redundancy, saturation checks, per-metric debugging. Not for headline numbers — these are un-pooled raw values. |
| `metadata_rows.csv` | tracker rows × 15 | The tracker input verbatim, incl. `latent_type`, `tokenization_regime`, `variant` | Joining metadata onto the tables (the result CSVs carry only `tokenizer`), and recording exactly which input produced these outputs. |
| `scores/<slug>.json` | one file per tokenizer | Each tokenizer's **full** `category_scores.json` — the 7 scores and the decoding breakdown, the per-metric `diagnostics` the CSVs flatten away, and `metadata` | Feeding the [tools](#tools) with `--scores-dir <outdir>/scores`, no MLflow round-trip. Written only once every variant for that tokenizer succeeded, and reconciled against the CSVs at startup, so **it never contains a tokenizer that `category_scores.csv` does not**. The converse can lag — see [When `scores/` lags the CSVs](#when-scores-lags-the-csvs). |
| `category_scores_no_m2.csv` | tokenizer × 14 | `--compare-m2` only: the same 7 scores with the M2 artifact withheld | Isolating M2's contribution to `latent_structure_and_robustness`. |
| `category_scores_m2_delta.csv` | tokenizer × 6 | `--compare-m2` only: `latent_bucket_without_m2`, `latent_bucket_with_m2`, their `delta`, and the M2 provenance columns | Quantifying per tokenizer how much of the latent bucket is M2. |
| `run_log.txt` | — | Per-row provenance: which artifacts resolved, M2 provenance, generalization source, failures | Auditing a run; diagnosing why a row is missing or a score is null. |
| `table_checks.json` | one file | Every table-level check (`mixed_chunk_sizes`, `mixed_members`, `mixed_score_parts`, `not_comparable`) with `passed` and its message; written even when a check fails | Automation: read this instead of parsing the builder's stderr |
| `tracker_with_agg.csv` | one row per tracker row | Only with `--log-mlflow`: the tracker rows plus `mlflow_aggregation` (empty for rows not published) | Pasting the column into the sheet; the tools' MLflow route |
| `artifact_cache/` | — | Downloaded MLflow artifacts + `manifest.json` | Reuse across runs. Regenerable; safe to delete to reclaim space. |

The wide table's column count is the **union** across tokenizers, so sparsity is expected —
different tokenizer families emit different metric sets.

#### When `scores/` lags the CSVs

The guarantee is deliberately one-directional. `scores/` never holds a tokenizer that
`category_scores.csv` does not — an orphan JSON is deleted at startup. The reverse is
allowed: a row can exist with no JSON yet, because the alternative would be deleting the
numbers you report in order to force agreement.

It happens when a `--limit` run, an interruption, or a failure stops before every row has
been scored. Both ends say so out loud:

- the builder's run summary ends with
  `WARNING: N tokenizer(s) are in category_scores.csv but have no scores/*.json …`,
  naming them;
- `load_scores(scores_dir=…)` — and so every CLI using `--scores-dir` — prints
  `WARNING: N tracker row(s) had no readable <slug>.json … and are MISSING from this
  output`, on **stderr**, regardless of `--verbose`, so piping a table to a file still
  shows it.

Re-running the builder closes the gap; the artifacts are already cached, so only the
scoring is repeated.

#### Reading `m2_provenance`

`category_scores.csv` carries the M2 resolution status per tokenizer. Check it before
reporting the latent bucket:

| Value | Meaning |
|---|---|
| `included` | M2 contributed at the configured ε |
| `excluded_no_report` | no M2 artifact for this row |
| `excluded_missing_key` | M2 report present but lacking the metric at that ε (empty report, or a different metric config) |
| `off` | `include_m2=False` |

**This matters for comparability.** The latent bucket is a mean over *available* members, and
M2 scores far higher than the others (mean ≈0.91 vs ≈0.42 for the bucket without it). A
tokenizer missing M2 is therefore not scored neutrally — it is scored on strictly harsher
members and lands materially lower. Rows that are not `included` are **not peer-comparable**
on `latent_structure_and_robustness`; re-run their M2 eval or mark them N/A.

`m1_provenance` reads the same way, over `components_mean` (M1 contributed the retention
of the three canonical corruptions), `excluded_no_report` (no M1 artifact for this row) and
`excluded_no_components` (M1 report present, but not all three canonical components had a
finite `raw_value` — usually a non-default M1 corruption config). The same comparability
caveat applies: a row without M1 is scored on its remaining members, not neutrally.

---

### Input: tracker CSV (or Google Sheets)

All tools take a local CSV path. A Google Sheets share URL is accepted as an optional alternative:

```bash
## local file
uv run python -m scripts.analysis.plot_radar tracker_with_agg.csv conv-vae-100to1

## Google Sheets (link-readable sheet)
uv run python -m scripts.analysis.plot_radar \
    "https://docs.google.com/spreadsheets/d/1ABC.../edit#gid=<tab-id>" \
    conv-vae-100to1
```

The `gid=` parameter in the Sheets URL selects the tab.

---

### Metric paths

All plot scripts accept **metric path** arguments that resolve against `category_scores.json`.

| Path form | Example | What it returns |
|---|---|---|
| Composite score (canonical) | `single_neuron_recon` | Top-level composite score |
| Composite score (alias) | `sn_recon`, `encoding`, `gen`, `comp` | Same as canonical |
| Decoding breakdown | `decoding_poyo`, `decoding_linear` | The two family scores `decoding` averages |
| Per-metric (normalized) | `sn_recon/spike_detection_f1` | Normalized per-metric value within that composite |
| Per-metric (long key) | `latent_structure_and_robustness/m1:components_mean` | M1 retention member (raw NMSE in `raw_value`, retention in `normalized`) |

The seven composite scores are: `single_neuron_recon`, `population_structure`,
`latent_structure_and_robustness`, `decoding`, `downstream_encoding`, `generalization`,
`compression`. The short aliases above (`sn_recon`, `encoding`, `gen`, `comp`) stand for
`single_neuron_recon`, `downstream_encoding`, `generalization` and `compression`; the other
composites are addressed by their canonical names.

---

### Composite-score sources

Each composite is built by `ntt.evaluation.aggregation.category_scores.compute_category_scores`.
The pooled columns use **per-named-metric family weighting**: each metric family (e.g.
`acg_correlation` across its time-bins, or a perturbation across its strengths) contributes a total
weight of 1 — bin/strength variants split that weight equally — so a metric with many bins doesn't
dominate. The recon comparison columns (`single_neuron_recon`, `population_structure`) are
Poisson-normalized (`0` = Poisson floor, `1` = ceiling); `latent_structure_and_robustness` instead
uses bounded `[0,1]` transforms on raw values (see its row below).

| Column | Source | Composition |
|---|---|---|
| `single_neuron_recon` | recon `results.json` | Poisson-normalized, family-weighted pool of `TEMPORAL_FIDELITY` + `SINGLE_UNIT_STATISTICS` metrics. `firing_rate_error` is scored `1 − E / R` against a population-mean-rate reference instead of the Poisson floor ([glossary](metric_glossary.md#composite-scores-the-7-radar-axes)); when the report's comparison is still against the Poisson floor, the aggregation recomputes it from the per-unit rates in `per_session_per_segment` (and fails if they are missing) |
| `population_structure` | recon `results.json` | Poisson-normalized, family-weighted pool of `POPULATION_STRUCTURE` metrics. **`subspace_overlap` is included** (it is not excluded); `token_dims_per_second`, `repeats_snr`, `repeats_baseline` are excluded |
| `latent_structure_and_robustness` | recon `results.json` + M1/M2 robustness `eval_report.json` | **Not Poisson-normalized** — a family-weighted mean of members mapped onto `[0,1]` by monotonic bounded transforms: structure-preservation (`population_structure_preservation`, `per_unit_structure_preservation`) via `max(corr, 0)` on raw RSA; M1 (input-corruption) via retention `1/(1+d)` where `d` is the **mean** raw latent-NMSE over the three canonical non-population corruptions (`window_shift\|max_shift_ms=3.0`, `jitter\|k_ms=1.0\|p=0.2`, `drop_and_add\|fraction=0.02`) — **all three or nothing**: if any is missing or non-finite the whole M1 member is warned-and-excluded, since a partial mean would not be comparable across tokenizers (`permute_neurons`, `*_per_dim/*` and zero-strength variants are never read); M2 (latent-noise) via `max(acg, 0)` at a chosen ε — **ON by default** (`include_m2`/`m2_metric`/`m2_epsilon`, GLOBAL knobs fixed across all tokenizers; default ε=0.1, see [ε note](#choosing-the-m2-ε-operating-point)). M2 is read from the tracker's `mlflow_latent_latent_corruption` column. The correlation-valued members (structure, M2) pin zero at correlation 0 — a non-positive correlation scores 0, not partial credit. A source missing its inputs is **warned-and-excluded**; the bucket scores on its remaining members. No `score_clip` (the transforms are bounded by construction) |
| `decoding` | decoding cohort record + recon `results.json` | **Raw R²** of POYO test R² (per session, `metrics.test.per_session[*].r2`; see [Downstream inputs](#downstream-inputs-encoding-and-decoding)) and the in-repo ridge-regression decoders `behavioral_regression` + `behavioral_temporal_decoder`, pooled as **two equal families**: the POYO sessions pool to one family and the ridge targets to a second, each carrying weight 1 (a flat mean would give ~12 POYO sessions 12/14 of the column and let 2 ridge targets barely register). The two families measure different things — a trained decoder's ceiling vs linear decodability of the latent — so each family mean is recorded separately in the column's `provenance` (`poyo`, `ridge`; `absent` when a family contributes nothing). Ridge entries are matched on `result.name`, since `aggregated.results` is keyed by the per-target *display* name (`behavioral_regression_cursor_vel`). For each ridge target, R² is averaged across sessions, excluding sessions in `RIDGE_EXCLUDED_SESSIONS` (`category_scores.py`). `behavioral_classification` is accuracy, not R², so it is excluded. The two family means are also saved as the scores `decoding_poyo` and `decoding_linear` (empty for a family that contributes nothing) |
| `downstream_encoding` | encoding `downstream_encoding_results.json` | Mean noise-ceiling R² of the val-selected checkpoint on the test split (`metrics.test.aggregated.nc_r2.value`), scalar pass-through; see [Downstream inputs](#downstream-inputs-encoding-and-decoding) |
| `generalization` | recon `results.json` | Scalar pass-through of the generalization score |
| `compression` | recon `results.json` | Scalar pass-through (`compression_rate`) — the **raw bit-ratio** `input_bits / output_bits`, *not* normalized to `[0, 1]` like the other six: unbounded above, higher is better, and **a value below 1 means the latent is larger than the input** (no compression). Bits are counted per code (output) and per bin (input; the record's `convention` is `per_code_per_bin`). Consumers rescale for display (see `plot_radar.py`'s `--compression-ceiling`); the stored value stays raw |

> **`repeats_snr` / `repeats_baseline`:** these are kept in the wide raw-metrics table (see
> `scripts/analysis/build_metrics_table.py`) but are **excluded from every composite score** — they
> are unbounded SNR-style ratios with no natural Poisson ceiling.

#### Downstream inputs (encoding and decoding)

ntt-downstream writes one schema-v2 record per result; its
[`docs/results.md`](https://github.com/enigma-brain/ntt-downstream/blob/main/docs/results.md)
is the authority on the format. `ntt.evaluation.aggregation.downstream_records` reads both
inputs through one loader (`load_downstream_record`), and scores only the record's `test`
block:

| Input | Tracker column points at | File read | Score |
|---|---|---|---|
| encoding | the encoding run | `downstream_encoding_results.json` (`kind: "run"`) | `metrics.test.aggregated.nc_r2.value` |
| decoding | the benchmark's parent run for this tokenizer | `downstream_decoding_results.json` at the parent's artifact root (`kind: "cohort"`) | `poyo/<session>` = `metrics.test.per_session[<session>].r2` |

- **No `test` block, no score.** An encoding record whose final eval failed or was disabled has
  no `test` block; `downstream_encoding` is then empty, with a warning that quotes the record's
  `metadata.final_eval`. A val block is never used in its place.
- **Incomplete decoding cohorts** (`metadata.cohort.complete = false`) give an empty
  `decoding` score and a warning naming the missing sessions. `allow_incomplete_decoding_cohort`
  (job config) / `--allow-incomplete-decoding-cohort` (table builder) scores it over the found
  sessions instead.
- **Encoding split.** Every encoding score records its split (`test`).
- **Warnings only:** a cohort whose cells disagree on git sha, ntt version or ntt commit
  (`mixed_provenance`), and an encoding test score from the last checkpoint rather than the
  val-selected one (`metadata.final_eval.ckpt_kind = "last"`).
- **Provenance.** `category_scores.json` records, under `metadata.downstream.{encoding,decoding}`,
  each input's `split`, `complete`, and the record's provenance
  (`tokenizer_checkpoint`, `ntt_version`, `ntt_path`, `ntt_commit`, `git_sha`; encoding
  `final_eval`; cohort `expected`/`found`/`missing`/`stale`/`mixed_provenance`).
  `category_scores.csv` carries `encoding_split`,
  `decoding_complete` and `decoding_mixed_provenance`; the wide table carries
  `encoding/split` and `decoding/complete`.

#### Choosing the M2 ε operating point

ε is a **global** knob: one value for every tokenizer, so all are bounded at the same noise
level. The default is `m2_epsilon=0.1`. Measured spread of `acg_correlation` across 48
tokenizers (`robustness_latent/acg_correlation/eps_*` in the wide table):

| ε | mean | median | std | IQR |
|---|---|---|---|---|
| 0.0 | 0.961 | 1.000 | 0.081 | 0.001 |
| 0.01 | 0.952 | 0.999 | 0.086 | 0.035 |
| 0.03 | 0.940 | 0.997 | 0.097 | 0.073 |
| **0.1** (default) | 0.907 | 0.960 | 0.104 | **0.183** |
| **0.3** | 0.840 | 0.816 | 0.114 | **0.196** |
| 1.0 | 0.708 | 0.745 | 0.169 | 0.040 |

ε=0 is fully saturated by construction (no noise injected) and ε=1.0 collapses again with the
distribution piling up. ε=0.1 is defensible but **not the most discriminative**: ε=0.3 has both
a wider IQR and a larger std. At ε=0.1 roughly half the field sits above 0.95, so the top of the
range is compressed. Since the whole sweep is in the wide table, check it against your data
before relying on the default.

---

### Tools

#### `plot_correlation.py` — scatter plot of two metrics

One point per tokenizer, coloured by family, with a Pearson-r annotation and regression line.

```bash
uv run python -m scripts.analysis.plot_correlation \
    tracker_with_agg.csv \
    single_neuron_recon decoding \
    --output plots/sn_recon_vs_decoding.pdf
```

```
positional arguments:
  tracker_csv    Tracker CSV (or Google Sheets URL)
  x_metric       X-axis metric path
  y_metric       Y-axis metric path

options:
  --output, -o   Save path (.pdf/.png/.svg); shows interactive window if omitted
  --no-annot     Skip Pearson-r / p-value annotation
  --labels       Label each point with the tokenizer name
  --cache-dir    Local cache for downloaded category_scores.json files
  --scores-dir   Read <dir>/<slug>.json locally instead of MLflow (path A output)
  --mlflow-uri   Override default MLflow tracking URI
  --verbose, -v  Print per-tokenizer load progress
```

**Example — per-metric sub-paths:**

```bash
uv run python -m scripts.analysis.plot_correlation \
    tracker_with_agg.csv \
    "sn_recon/spike_detection_f1" "population_structure/participation_ratio"
```

---

#### `plot_family.py` — strip plot grouped by tokenizer family

Metric value on the y-axis; families on the x-axis, sorted by median descending.
Each point is one tokenizer; a horizontal bar marks the family median.

```bash
uv run python -m scripts.analysis.plot_family \
    tracker_with_agg.csv \
    decoding \
    --output plots/family_decoding.pdf
```

```
positional arguments:
  tracker_csv    Tracker CSV (or Google Sheets URL)
  metric         Metric path

options:
  --output, -o   Save path; shows interactive window if omitted
  --no-median    Skip the per-family median bar
  --no-legend    Hide the colour legend
  --labels       Label each point with the tokenizer name
  --cache-dir    Local cache directory
  --scores-dir   Read <dir>/<slug>.json locally instead of MLflow (path A output)
  --mlflow-uri   Override MLflow tracking URI
  --verbose, -v
```

---

#### `plot_radar.py` — radar / spider plot

All seven composite scores on one chart for one or more tokenizers.
Metrics that are `null` are shown at zero; the legend notes how many axes had data.

```bash
## single tokenizer
uv run python -m scripts.analysis.plot_radar \
    tracker_with_agg.csv \
    conv-vae-100to1

## compare several tokenizers on one chart
uv run python -m scripts.analysis.plot_radar \
    tracker_with_agg.csv \
    conv-vae-100to1 \
    wavelet-rfsq-2-stages \
    --output plots/radar_comparison.pdf
```

```
positional arguments:
  tracker_csv    Tracker CSV (or Google Sheets URL)
  tokenizers     One or more tokenizer names

options:
  --output, -o   Save path; shows interactive window if omitted
  --fuzzy        Match tokenizer names by substring instead of exact match
  --clip         Axis upper bound (default: 1.0)
  --compression-ceiling
                 Bit-ratio mapped to a full compression axis
                 (default: 2.0; choose a value a little above
                 your cohort's maximum compression rate)
  --cache-dir    Local cache directory
  --scores-dir   Read <dir>/<slug>.json locally instead of MLflow (path A output)
  --mlflow-uri   Override MLflow tracking URI
  --verbose, -v
```

> **Note on the Compression axis:** the six other composites are already on a `[0, 1]`
> scale, but `compression` is a raw bit-ratio (`input_bits / output_bits`) — unbounded
> above, and **below 1 when the latent is larger than the input**. The radar divides it by
> `--compression-ceiling` before applying `--clip`, so a full-length compression spoke
> means "at or above the ceiling", and the axis label names the ceiling in use
> (e.g. `Compression (max 2x)`). Without that rescale every tokenizer's compression spoke
> would saturate at `--clip` and the axis would carry no information.
>
> **Size the ceiling for your cohort.** With the default `2.0`, a cohort whose
> `compression_rate` lies between 0.4 and 1.5 would be spread over roughly 0.20–0.75 of the
> axis. A ceiling far above the cohort's range (for example `50`) maps every tokenizer onto
> a sliver near the centre of the plot. Pick a ceiling a little above your cohort's maximum.
> The value in `category_scores.json` is unaffected and stays raw.

**Tip:** `--fuzzy` lets you use short substrings when you don't want to type the full slug:

```bash
uv run python -m scripts.analysis.plot_radar tracker_with_agg.csv \
    "conv-vae" wavelet-rfsq --fuzzy
```

---

#### `build_latex_table.py` — LaTeX results table

Generates a `table*` in the style of the paper's main results table:
seven metric columns, rows grouped by tokenizer family with `\multicolumn` section headers,
sorted by SN Recon descending within each family.
Best value per column is **bolded**.

```bash
uv run python -m scripts.analysis.build_latex_table \
    tracker_with_agg.csv \
    --output paper_table_agg.tex

## write to stdout and pipe:
uv run python -m scripts.analysis.build_latex_table tracker_with_agg.csv > paper_table_agg.tex
```

```
positional arguments:
  tracker_csv       Tracker CSV (or Google Sheets URL)

options:
  --output, -o      Save path (.tex); writes to stdout if omitted
  --sort-family     Order families by "best" member (default) or "median" SN Recon
  --no-bold         Skip bolding of best values
  --caption         Override the default table caption
  --label           Override the LaTeX \label (default: tab:all_results_agg)
  --cache-dir       Local cache directory
  --scores-dir      Read <dir>/<slug>.json locally instead of MLflow (path A output)
  --mlflow-uri      Override MLflow tracking URI
  --verbose, -v
```

> **Note on the Compression column:** unlike the other six columns, this one is **not** a
> normalized score. It is the raw bit-ratio (`input_bits / output_bits`) passed straight
> through from `category_scores.json` — higher is better, and **below 1 when the latent is
> larger than the input** (see the
> [composite-source table](#composite-score-sources)). It is printed with the same three
> decimals as the `[0, 1]` columns: the spread is often narrow enough that one decimal
> would collapse distinct tokenizers (`0.391` and `0.434` both become `0.4`). Bolding is
> computed per column, so the ratio never competes with the normalized scores.

---

### Common options reference

| Option | All tools | Description |
|---|---|---|
| `--cache-dir DIR` | ✓ | Cache `category_scores.json` locally; skipped on subsequent runs (MLflow route only) |
| `--scores-dir DIR` | ✓ | Read `category_scores.json` from `DIR/<slug>.json` — `build_aggregation_tables.py`'s `<outdir>/scores` — instead of MLflow. Makes `mlflow_aggregation` unnecessary |
| `--mlflow-uri URI` | ✓ | MLflow server for the MLflow route (default: `MLFLOW_TRACKING_URI`, and links to another server are an error); every run is read from it |
| `--output / -o` | ✓ | Save figure or `.tex` file; omit to show an interactive window (plots) or write to stdout (table) |
| `--verbose / -v` | ✓ | Print per-tokenizer load progress and warnings |
