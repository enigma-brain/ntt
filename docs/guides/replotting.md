# Re-plotting and Scaling Plots from Saved Eval Results

After running `uv run ntt job=eval ...` with `job.output_path=<dir>`, the eval job
saves all metric results to disk. You can regenerate and extend plots from these
saved results without re-running evaluation.

## Output files from the eval job

Each eval run writes:

```
<output_dir>/
  results.json                      # aggregated + per-session metric results
  results.csv                       # wide-format summary
  metrics_detailed_results.jsonl    # one record per metric per sweep point (NEW)
```

The JSONL file stores per-session, per-sweep-point results in a long-form format
that is easy to filter and plot from. Each line is a JSON object with
`record_type` = `"metric_result"` (base result) or `"sweep_point"` (one per
parameter value in a sweep, e.g. chunk_window_ms or bin_size_ms).

## Re-plotting: compare multiple tokenizers

Use `scripts/replot_from_eval_results.py` to compare multiple eval runs
side-by-side:

```bash
python scripts/replot_from_eval_results.py \
  results/tokenizer_A/ \
  results/tokenizer_B/ \
  results/tokenizer_C/ \
  --output-dir results/comparison_plots
```

This loads `results.json` (and `metrics_detailed_results.jsonl` when present)
from each directory and generates:

- **Behavioral plots** (regression R^2, classification accuracy) per session,
  with error bars from cross-validation folds
- **Comparison bar charts** and radar plots (when Poisson baseline data is present)
- **Scaling plots** (enabled by default, disable with `--no-scaling-plots`)

## Scaling plots

Scaling plots show how metric quality relates to tokenizer capacity
(token dimensions per second). They are generated automatically when
`--scaling-plots` is set (the default).

### Metric vs token dimensions

One scatter plot per metric, with each tokenizer as a point. For metrics that
have a chunk-window sweep, each sweep point appears separately.

The x-axis can be configured:

| `--dims-x-axis` value | Meaning |
|---|---|
| `token_dims_per_second` (default) | Fixed x per tokenizer regardless of chunk size |
| `token_dims_per_chunk` | x scales by `chunk_window_ms / 1000`, useful for comparing across chunk sizes |

Filter which metrics are plotted with `--dims-metrics`:

```bash
python scripts/replot_from_eval_results.py \
  results/tok_A/ results/tok_B/ \
  --dims-metrics acg_correlation firing_rate_error
```

### Metric vs metric (colored by token dims)

Scatter plots of one metric against another, with color encoding token
dimensions/sec. Useful for exploring the reconstruction-vs-downstream tradeoff.

Configure with:

| Flag | Default | Description |
|---|---|---|
| `--xy-x-metric` | `acg_correlation` | Metric for x-axis |
| `--xy-y-metrics` | `behavioral_regression_cursor_vel behavioral_classification_trial_target_id` | Metric(s) for y-axis (one plot each) |

```bash
python scripts/replot_from_eval_results.py \
  results/tok_A/ results/tok_B/ \
  --xy-x-metric acg_correlation \
  --xy-y-metrics behavioral_regression_cursor_vel subspace_overlap
```

### Per-session variants

All scaling plots also generate per-session versions under
`<output_dir>/scaling_plots_per_session/`. These avoid cross-session aggregation
artifacts and are important for behavioral decoding metrics where session-level
variance is high.

## Output directory structure

```
<output_dir>/
  behavioral/                              # behavioral bar charts
    <session>/
      behavioral_regression_*.png
      behavioral_classification_*.png
  scaling_plots/                           # aggregated scaling plots
    metric_vs_token_dims__<metric>.png
    scatter__<y_metric>__vs__<x_metric>.png
  scaling_plots_per_session/               # per-session scaling plots
    metric_vs_token_dims__<metric>__<session>.png
    scatter__<y>__vs__<x>__<session>.png
```

## Full CLI reference

```
python scripts/replot_from_eval_results.py [OPTIONS] RESULT_DIRS...

positional:
  result_dirs             Directories containing results.json (one per tokenizer)

options:
  --output-dir DIR        Where to save plots (default: scripts/eval_outputs/comparison_plots)
  --dpi INT               Figure resolution (default: 150)
  --no-scaling-plots      Disable scaling scatter plots
  --dims-metrics KEY...   Metric keys for metric-vs-token-dims plots (default: behavioral decoding metrics plus repeats_snr)
  --dims-x-axis CHOICE    token_dims_per_second | token_dims_per_chunk
  --xy-x-metric KEY       X-axis metric for metric-vs-metric plots
  --xy-y-metrics KEY...   Y-axis metric(s) for metric-vs-metric plots
```
