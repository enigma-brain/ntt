# Tokenizer radar plots

Multi-tokenizer comparison radars (a.k.a. spider plots) that overlay one
polygon per tokenizer on a polar chart, with each axis a metric or metric
category. Floor and ceiling rings mark the Poisson baseline and the
perfect-reconstruction ceiling — so `0` is "no better than a Poisson
process" and `1` is "indistinguishable from ground truth."

The data source is each tokenizer's `results.json` produced by
`NTTEvalJob`.

> **Related:** [`docs/eval_aggregation.md`](eval_aggregation.md) covers the
> `scripts/analysis/` tooling that aggregates `category_scores.json` into the
> seven paper columns; its `plot_radar` CLI shapes those scores into a frame and
> draws it with the `plot_radar` renderer documented here.

## Quickstart

```python
import json
from pathlib import Path

import matplotlib.pyplot as plt

from ntt.evaluation.radar import (
    CategorySpec, DEFAULT_EVAL_JOB_SCALARS, RadarBundle,
    collect_category_scores, inject_scalar_metrics, plot_radar,
)

root = Path("tokenizer_results_artifacts")
bundle = RadarBundle.from_local_dir(root)

# Pull the eval-job scalars (compression, generalization) into the bundle
# as first-class radar axes.
raw = {n: json.loads((bundle.paths[n] / "results.json").read_text())
       for n in bundle.tokenizers}
inject_scalar_metrics(bundle, raw, DEFAULT_EVAL_JOB_SCALARS)

# Categories: either roll up by MetricCategory, or define your own.
spec = CategorySpec({
    "Temporal fidelity": ["acg_correlation", "spike_detection_f1"],
    "Population structure": ["population_structure_preservation"],
    "Compression": ["compression/rate"],
    "Generalization": ["generalization/score"],
})
scores = collect_category_scores(bundle, spec, normalize=True)
fig = plot_radar(scores, title="Tokenizer comparison", mode="normed")
plt.savefig("tokenizer_radar.png", dpi=200, bbox_inches="tight")
```

## Where the data comes from

`RadarBundle.from_local_dir(root)` expects a tree like:

```
root/
  <tokenizer_name>/results.json
  _reference/poisson_baseline.json    # optional override
```

Each `results.json` is a stock `NTTEvalJob` output. Three keys are
required: `aggregated`, `poisson_baseline`, `perfect_ceiling`. Two more
become first-class radar inputs when present:

- `compression` — written by the eval job when `compute_compression` succeeds.
- `generalization` — written by the eval job when `tier_selections` is configured.

Use `scripts/analysis/aggregate_runs.py` to pull a set of runs into the
expected tree:

```bash
uv run --env-file .env python scripts/analysis/aggregate_runs.py \
    --manifest runs.csv \
    --output-dir tokenizer_results_artifacts
```

`runs.csv` columns: `tokenizer`, `results_uri`. URIs accept local paths
or `mlflow-artifacts:/...`. A `--default-poisson-reference` flag places
one shared baseline at `_reference/poisson_baseline.json` (paper-figure
canonical floor) — this is the only override `RadarBundle.from_local_dir`
honors; omit it and the radar uses each tokenizer's in-results baseline.

## Custom axes

`ScalarMetric` exposes any nested scalar from `results.json` via a dotted path:

```python
from ntt.evaluation.radar import ScalarMetric, inject_scalar_metrics

custom = [
    # The shipped defaults are in DEFAULT_EVAL_JOB_SCALARS — extend or override.
    ScalarMetric(
        "compression/rate",
        path="compression.compression_rate",
        direction="higher", floor=0.0, ceiling=100.0,  # tighter range for small figures
    ),
    ScalarMetric(
        "my_custom_axis",
        path="some.other.nested.key",
        direction="lower", floor=1.0, ceiling=0.0,    # lower-is-better
    ),
]
inject_scalar_metrics(bundle, raw, custom)
```

Range hints (`floor`, `ceiling`) drive how the radar normalizes a scalar
to `[0, 1]`. For lower-is-better metrics, set `direction="lower"` and
flip `(floor, ceiling)` so the "good" side maps to 1.

The shipped `DEFAULT_EVAL_JOB_SCALARS` uses a placeholder ceiling of
`1000` for compression scalars. Realistic compression ratios run ~2–50,
so the normalized polygon will sit near zero unless you override
`(floor, ceiling)` to a tighter range per-figure. Generalization scalars
are already in `[0, 1]` by construction.

## `normed` vs `raw`

`collect_category_scores(..., normalize=True)` returns Poisson-normalized
values: `0 = floor`, `1 = ceiling`, sign-oriented by `MetricDirection`.
`firing_rate_error` is left out of normalized scores here (with a warning): it is scored
against a population-mean-rate reference, not the Poisson floor, and that reference needs
the per-unit rates of `per_session_per_segment`, which the bundle does not load (see
[`metric_glossary.md`](metric_glossary.md#composite-scores-the-7-radar-axes)).
`normalize=False` returns the raw category means.

`plot_radar(..., mode="normed" | "raw")` appends a `(NORMED)` or `(RAW)`
suffix to the title so saved figures stay unambiguous.

When mixing raw values from heterogeneous units on a single
`mode="raw"` radar, the polygon will be lopsided (the largest-scale
axis dominates). The CAPS suffix in the title is there partly to flag
this so misleading mixed-unit charts can't be confused with the normed
equivalents.

## Schema notes

### Missing metrics

Metrics present in `CategorySpec` but absent from a tokenizer's report
are skipped with a per-(tokenizer, metric) warning. The category mean
is computed over whatever is present, so partial coverage degrades
gracefully rather than failing the whole radar.

### `inject_scalar_metrics` is idempotent on baseline/ceiling

The floor/ceiling synthetic entries on `bundle.baseline` / `bundle.ceiling`
are added the first time a scalar attaches; subsequent injections of the
same name do not overwrite. Override `(floor, ceiling)` by editing the
`ScalarMetric` definition, not by re-injecting.

### Tokenizers without a cached `results.json` don't load

`RadarBundle.from_local_dir` requires `<tokenizer>/results.json` to be
present. Rows in the aggregator's input CSV that fail to copy are
logged at error level but don't kill the batch; check the produced
`manifest.csv` to see which tokenizers actually made it through.

## See also

- `src/ntt/evaluation/radar.py` — module-level docstring contains a more
  compact quickstart.
- `src/ntt/jobs/eval.py` — where `compression` and `generalization`
  blocks are written into `results.json`.
- `scripts/analysis/aggregate_runs.py` — pulls cross-run results.json
  files into the expected tree.
