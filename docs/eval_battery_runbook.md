# Runbook — evaluating ntt tokenizers

How to score a trained ntt tokenizer on every evaluation axis, and how to scale that from
one checkpoint to a cohort.

> **This runbook is the reference body behind the `ntt-eval-battery` agent skill.**
> The skill (`.claude/skills/ntt-eval-battery/`) is a short
> spine that points at the sections below. If you change how the battery works, change it
> here — the skill and every downstream reference point back to this file.
>
> **This document describes the supported workflow.** Troubleshooting notes for checkpoint,
> data-config and environment issues live in
> `.claude/skills/ntt-eval-battery/references/known-pitfalls.md` — the "troubleshooting
> notes" from here on. See §10.

The battery is **six metric families across four job invocations**:

| axis | job | pass |
|---|---|---|
| reconstruction | `job=eval_all` | 1 |
| RepeatsSNR | `job=eval_all` | 2 |
| M1 input-corruption robustness | `job=eval_all` | 3 |
| M2 latent-corruption robustness | `job=eval_all` | 4 |
| encoding | `job=downstream_encoding` | — |
| decoding | `job=downstream_decoding job.mode=benchmark` | — |

**Every job here takes exactly one checkpoint.** There is no multi-tokenizer entrypoint in
ntt: a cohort is a loop you write. So **Part I** is one tokenizer end to end, and **Part
II** is scaling that up.

Read Part I first even with fifty checkpoints in hand. Part II's fan-out assumes a
verified working invocation, and producing one is the entire point of Part I — several
failure modes only surface after an expensive earlier stage succeeds, and discovering
them once beats discovering them once per cohort member.

## The tracker's four jobs

Both parts assume an experiment tracker, but nothing depends on *which* one. A tracker does
exactly four things here:

1. **Enumerates candidate runs** — which trainings exist, and their status (§6).
2. **Resolves checkpoint artifacts** — a URI per run whose bytes you can fetch (§6).
3. **Receives per-axis metrics** — so results are comparable across a cohort (§2–§4).
4. **Acts as the progress channel** for runs whose stdout is unreachable (§7).

For a single tokenizer only (3) really matters, and even that is optional — the jobs write
artifacts to disk regardless. Tracker dependence arrives with the cohort.

ntt wires **MLflow** natively, so MLflow is the worked example throughout, always inside a
marked block:

> **With MLflow:** one experiment holds the training runs, and `search_runs()` — or the
> REST `/api/2.0/mlflow/runs/search` endpoint — enumerates them **(1)**. Each run's
> artifact listing yields an `mlflow-artifacts:/…` checkpoint URI **(2)**.
> `job.mlflow_logging.*` on the eval jobs and `MLFLOW_TRACKING_URI` for the downstream
> submodule send metrics back **(3)**. The run table, filtered to the target experiment, is
> the progress view while a cluster job's stdout is unreachable **(4)**.

Sections whose mechanics change under a different tracker end with a note on what you then
have to supply yourself.

---

# Part I — Evaluating one tokenizer

> **Re-scoring a published tokenizer?** `scripts/eval/published_commands.py` writes the exact
> commands its published scores came from; see [`pretrained_models.md`](pretrained_models.md) §2.
> This runbook is for your own checkpoints.

## 0. Prerequisites

```bash
set -a && source .env && set +a   # tracker creds, GH_TOKEN — secrets; keep them out of
                                  # commits and logs
uv sync                            # per-worktree venv; a worktree without one silently
                                   # runs the MAIN repo's src
git submodule update --init --force downstream
uv run python -m ntt.jobs.downstream_env   # builds downstream's env with THIS checkout's
                                           # ntt overlaid
```

The downstream launcher owns the submodule's env: every launch checks it and, with
the default `job.downstream_env=auto`, builds it if needed, so the last line only saves the
first launch the work. Never `uv sync --project downstream --frozen` by hand: it swaps this
checkout's `ntt` back for the submodule's pinned tag. `docs/downstream_maintainer_internals.md` § "Step 2 — the submodule's venv (the
host-ntt overlay)" has the details; `docs/downstream_eval.md` § "Which `ntt` code the
evaluation uses" is the user-level summary.

### How the configs compose, and the one you must not forget

Every job here is composed by **Hydra**. `ntt job=<name>` selects `config/job/<name>.yaml`,
and other groups — `data`, `evaluation`, `local`, `paths` — compose in around it. Two
consequences for every command below:

- `paths.run_name` is `???` (mandatory), which is why every invocation sets it.
- **`++` is required for any key not already present** in the composed config; plain `=`
  on an unknown key hard-fails, and that is a feature.

**`local=<host>` sets `paths.data_dir`, and you should always pass it.** The files in
`config/local/` are one-line per-machine overlays naming the **brainsets root** — the
directory holding processed brainset directories:

| entry | `paths.data_dir` |
|---|---|
| `local=default` | empty overlay: changes nothing. Jobs get `${PROJECT_ROOT}/data` |
| `local=<your-host>` | a file you add, e.g. `config/local/myhost.yaml` |

Add a file for each machine; it needs nothing but a `# @package _global_` header and a
`paths.data_dir`:

```yaml
# @package _global_
paths:
  data_dir: /path/to/brainsets/processed
```

**Omitting `local=` falls back to `${PROJECT_ROOT}/data`.** The eval jobs use `paths=default`, so
without a `local=` overlay or an explicit `paths.data_dir` the data root is a directory inside the
checkout:

```console
$ ntt job=eval_all paths.run_name=probe --cfg all | grep data_dir
  data_dir: /path/to/ntt/data                                  # no local=
$ ntt job=eval_all local=myhost paths.run_name=probe --cfg all | grep data_dir
  data_dir: /path/to/brainsets/processed                       # local= wins
```

To keep a shared-cluster layout in one place, copy `config/paths/example_cluster.yaml`
(placeholder paths), edit it, and select it with `paths=<name>`.

`local` is applied after `job` in the root defaults list, so it wins — but only if you pass
it. **`--cfg all` resolves the config without running anything**, and checking `data_dir`
that way costs seconds against a job that takes hours. `PROJECT_ROOT` must be set for
`root_dir` to resolve; the `.project-root` marker at the repo root is what it points at.

### The checkpoint must load on this checkout

A tokenizer trained with a different version of ntt may use loss kinds or spec fields that
this checkout does not have, so the checkout has to carry the training code that produced
the checkpoint.

Verify it before launching any evals:

```bash
uv run python -c "
from ntt.core.model import load_tokenizer_from_checkpoint
tk = load_tokenizer_from_checkpoint('<one.ckpt>', map_location='cpu', weights_only=False)
print('OK', type(tk).__name__, tk.spec.KIND)"
```

`weights_only=False` is required — torch ≥ 2.6 defaults to blocking the pickled spec.

This is the cheapest check in the battery and it forecloses the most expensive failure. The
two shapes the failure takes, and what each means, are in the troubleshooting notes.

## 1. Decide the Poisson baseline

`job=eval_all` with `compare_poisson=true` consumes a precomputed `poisson_baseline.json`.
Its metadata check **raises** when the consumer's session set is a *superset* of the
baseline's, and on any mismatch in the hard fields — `bin_size_ms`,
`rate_estimation_bin_ms`, `split`, `max_eval_seconds`, `decoding_bypass_fields`. Two fields
are soft and only warn: `window_length_seconds` and `n_realisations`. Confirm the current
split with `grep -n "_HARD_FIELDS\|_SOFT_FIELDS" src/ntt/evaluation/artifact.py`. The
baseline keeps its own window (0.1 s by default) whatever window the model is scored on; most
metrics do not depend on it, but the two structure-preservation metrics compare windows with
each other, so their Poisson values are those of the baseline's window.

### The hard fields are not free parameters — they are your eval setup

This is the part worth internalising, because it determines whether one baseline can serve
your whole cohort. **You do not choose the hard fields directly.** Each is derived from
something else, so the way to make a baseline match is to reproduce that something else:

| hard field | derived from | consequence |
|---|---|---|
| `bin_size_ms` | the **tokenizer spec** (`spec.bin_size_ms`) | **A cohort can share one baseline only if all its tokenizers share a bin size.** Mixed bin sizes need one baseline each |
| `max_eval_seconds` | the **evaluation config** | Overriding the per-session cap invalidates the baseline |
| `decoding_bypass_fields` | the **metric suite** in the evaluation config | Changing which metrics run changes this field, so a different `evaluation` group invalidates the baseline — the least obvious of the five |
| `split` | `job.split` (default `val`) | Must match |
| `rate_estimation_bin_ms` | the **baseline job's** `rate_estimation_bin_ms` (default `10.0`) | Cannot mismatch: the consumer adopts the baseline's value, and only warns if `job.poisson_rate_bin_ms` was set to something else |
| `sessions` (not hard, but checked) | the **data config's** selections | The consumer's sessions must be a **subset** of the baseline's; a superset raises |

So the rule is short: **produce the baseline with the same `data` and `evaluation` configs,
the same `split`, and the cohort's `bin_size_ms`.** Then the hard fields agree by
construction and there is nothing to reconcile. `job=poisson_baseline` takes
`job.bin_size_ms` as a parameter (default `1.0`); it is not inferred from a tokenizer, so
pass it when the cohort's spec bin size differs.

Decide between three routes before you start, because recomputing is the most expensive
thing in this document and scoping is nearly free:

- **Reuse an existing baseline** — cheapest, if one exists at your scope. The references the
  published tokenizers are scored against are on the Hugging Face Hub:
  `job.poisson_reference=hf://the-enigma-project/ntt-tokenizers/poisson_references/<file>[@<revision>]`
  (the README there says which file goes with which bin size).
- **Scope the consumer** — filter the eval data config's selections down to the baseline's
  sessions. A config edit, at the cost of coverage. Record which sessions you dropped: it
  changes what the Poisson column means.
- **Recompute** — `ntt job=poisson_baseline local=<host>`, passing the **same
  `data@data_config` and `evaluation@eval_config` your eval will use.** Its defaults are
  `ntt_eval_light` + `hill_climb` precisely so that an untouched baseline matches an
  untouched `eval_all` main pass: `eval_all` injects `max_eval_seconds=600` and
  `max_neurons=200` into that pass (`grep -rn _HILL_CLIMB_MAIN_DEFAULTS src/`), which are the
  same caps the baseline job sets.

**Prefer not to touch the knobs below.** They exist, and they are the right lever if a
recomputation is genuinely too slow, but the default configuration is chosen to be
mutually consistent — every deviation is a field you now have to keep matched by hand.

| knob | effect if you do move it |
|---|---|
| `n_workers` | **Safe: changes no metadata.** Realisation 1 runs the full pipeline serially and the rest parallelise across this, defaulting to `min(n_realisations - 1, SLURM_CPUS_PER_TASK or os.cpu_count())` — so the CPU budget is the one lever with no compatibility cost, and the first thing to reach for |
| `n_realisations` | Linear in total work. **Soft** field: a smaller value only makes the baseline noisier and warns, so it stays usable |
| `window_length_seconds` | Inversely proportional to window count. **Soft**, but it changes what is being compared |
| `max_eval_seconds`, `max_neurons` | **Hard.** Must then match on the consumer side |

### Two validation points, and why the first one matters

The check runs twice, deliberately:

- **Fail-fast, before the slow eval.** Hard fields only, in seconds rather than after an
  hour of compute. Sessions are populated from the baseline itself here, so the session check
  trivially passes at this stage — that is intentional, not a bug.
- **The real session check, late.** Only after per-session evaluation has produced the
  consumer's actual session list can the subset check run. A session set that is too broad
  therefore fails *after* the main pass has done its work — one more reason to settle scope
  in this section rather than discovering it later.

## 2. Intrinsic evals — `job=eval_all`

**Use `eval_all`.** It is a thin wrapper that runs four standalone jobs in sequence, and its
value is that it pairs each pass with the right `evaluation` and `data` config, derives
consistent run names and output paths, and forwards the shared settings once. Assembling the
four by hand means getting all of that right yourself.

It is only a wrapper, though, so nothing is hidden — and when one pass fails you can re-run
just that one:

| pass | standalone job | `evaluation` config | default `data` config | output suffix | tracker suffix |
|---|---|---|---|---|---|
| 1 main | `job=eval` | `hill_climb` | `ntt_eval_light` | *(none)* | `__main` |
| 2 SNR | `job=eval` | `pipelines/cowley_per_stimulus` | `cowley_snr_fast_eval` | `_snr` | `__snr` |
| 3 M1 | `job=eval_robustness_input` | `metrics/robustness_input` | `robustness_fast_eval` | `_m1` | `__robustness_input` |
| 4 M2 | `job=eval_robustness_latent` | `metrics/robustness_latent` | `robustness_fast_eval` | `_m2` | `__robustness_latent` |

Note the two suffix schemes are different: on-disk outputs get `_snr`/`_m1`/`_m2` (the main
pass gets none), while tracker run names get the longer `__`-prefixed forms.

**To re-run one pass**, either disable the others and keep the wrapper's wiring —

```bash
uv run ntt job=eval_all local=<host> paths.run_name=$slug job.ckpt_path=<ckpt> \
  job.main.enabled=false job.snr.enabled=false job.latent_robustness.enabled=false
```

— or invoke the standalone job with the pairing from the table:

```bash
uv run ntt job=eval_robustness_input local=<host> \
  evaluation@job.eval_config=metrics/robustness_input \
  data@job.data_config=robustness_fast_eval \
  paths.run_name=${slug}_m1 job.ckpt_path=<ckpt>
```

The tracker-suffix column describes the `eval_all` wrapper. The standalone robustness jobs
default to `__m1` / `__m2`; add `job.mlflow_logging.run_name_suffix=__robustness_input`
(or `__robustness_latent`) to match the wrapper's run names.

The first form is safer, because the pairing is the thing that is easy to get wrong. Reach
for the second when you want to change something the wrapper does not expose per pass.

### The full invocation

One invocation covers all four intrinsic passes:

```bash
CUDA_VISIBLE_DEVICES=$GPU uv run ntt \
  job=eval_all local=<host> \
  paths.run_name=$slug \
  job.ckpt_path=<checkpoint> \
  job.main.data=<your main data config> \
  job.snr.data=cowley_snr_fast_eval \
  job.input_robustness.data=robustness_fast_eval \
  job.latent_robustness.data=robustness_fast_eval \
  job.output_path=$OUT/$slug \
  job.compare_poisson=true job.poisson_reference=<uri>
```

> **With MLflow:** append `job.mlflow_logging.enabled=true`,
> `job.mlflow_logging.experiment_name=<target>`, `job.mlflow_logging.run_name=$slug`. The
> four passes become four runs sharing the checkpoint-path tag, each with its own metric
> prefix, so they are discoverable from one another and do not collide. Their run names take
> the suffixes `__main`, `__snr`, `__robustness_input`, `__robustness_latent`.

### What has a default, and when to override it

Most of `config/job/eval_all.yaml` is already right. These are the fields worth knowing:

| field | default | override when |
|---|---|---|
| `job.ckpt_path` | `null` | **Always** — this or `tokenizer_config`, exactly one |
| `paths.run_name` | `???` | **Always** — mandatory, and it names the outputs |
| `local` | *unset* | **Always** — see §0; the default `data_dir` is a foreign path |
| `job.output_path` | `null` | Whenever you want outputs somewhere predictable, which is always for a cohort |
| `job.compare_poisson` | `false` | **Any run you will aggregate**: aggregation fails a tokenizer whose recon results lack `comparison_vs_poisson`. Then `poisson_reference` too (§1) |
| `job.split` | `val` | Evaluating on test. It is a **hard** baseline field (§1) |
| `job.mode` | `fast` | `precision` swaps the SNR pass onto `cowley_v4_compact_val`. Affects the SNR pass only |
| `job.<pass>.data` | per the table above | Your tokenizer needs padding (§2 note below), or a different session scope |
| `job.<pass>.enabled` | `true` | Re-running or skipping a pass |
| `job.mlflow_logging.enabled` | `false` | Logging to a tracker. Off by default, so results are local-only unless asked for |
| `job.evaluate_all_neurons` | `false` | You want the full-neuron pass (main/SNR only) |
| `job.generate_plots` | `false` | You want rasters; costs time |
| `job.seed`, `job.poisson_seed` | `42` | Rarely. Changing the Poisson seed does not invalidate a reference, but it does change what you compare |
| `job.device` | `auto` | Rarely — pin with `CUDA_VISIBLE_DEVICES` instead |

How to call it:

- **This eval is CPU-bound, not GPU-bound** (near-zero GPU utilisation, ~8.5 GB). Cap the
  BLAS pool — `OMP_NUM_THREADS` / `MKL_NUM_THREADS` around 20 — or a single run expands to
  most of the box's cores and starves everything else. The parallelism ceiling is cores,
  not VRAM.
- **M2 (latent robustness) is the longest pass**, so it dominates a full `eval_all` and
  sets the walltime to request on a cluster.
- The eval window is `job.eval_window_seconds` (seconds, or `native` = the tokenizer's own
  `window_length_seconds`; an error for a distribution-valued window). It reaches every pass.
  `++job.eval_config.eval_window_seconds` still hard-fails Hydra on `eval_all`. The default
  (1.0 s) is NOT the native window of ldns / perceiver (2.5-4 s): pass `native` for them. The
  window scored is saved as `scored_window_seconds` and compared by the aggregation. A report without it
  has it derived from its window settings by the aggregation; if they do not determine it,
  it stays unknown and the comparability check reports it as not checked.
- Set per-pass data configs via `job.<pass>.data`. There is no `job.data_config` at the
  `eval_all` level; `++job.data_config.<x>` fails with `extra_forbidden`.
- `++` is mandatory for any field not already in `config/job/eval_all.yaml`.

**If the spec has a sampled neuron count**, read the troubleshooting notes before launching:
it needs a manual change in each pass’s data config.

## 3. Encoding — `job=downstream_encoding`

```bash
CUDA_VISIBLE_DEVICES=$GPU uv run ntt \
  job=downstream_encoding local=<host> \
  paths.run_name=enc_$slug \
  job.ckpt=<checkpoint> \
  job.recipe=<recipe name, or a path to an overlay> \
  job.loss_mode=<the recipe's loss mode> \
  job.out=$OUT/$slug \
  'job.extra=["data.dataset_path=<dataset dir>","tags.model=<unique>"]'
```

Nearly every field in `config/job/downstream_encoding.yaml` defaults to `null`, meaning "let
the submodule decide". That is usually right; these are the exceptions:

| field | default | override when |
|---|---|---|
| `job.ckpt` | `null` | **Always** — local path or `mlflow-artifacts:/` URI |
| `job.recipe` | `cowley_v4_base` | A different dataset or architecture; Perceiver (Compress-N) tokenizers need a Perceiver recipe such as `cowley_v4_perceiver_mse` (`docs/downstream_encoding_advanced.md`). A bare name resolves inside the submodule; a path (contains `/` or ends `.yaml`) points at an external overlay |
| `job.loss_mode` | `null` → **`mse`** | **Whenever the recipe sets one** — see below, this is the biggest footgun here |
| `job.out` | `null` | **Always for a cohort** — runs collide on disk otherwise |
| `job.data_root` | `null` | Carefully; it is a *root*, not a dataset dir — see below |
| `job.gpus` | `null` | Never for encoding: it is read but never used. Pin with `CUDA_VISIBLE_DEVICES` |
| `job.dry_run` | `false` | Printing the resolved command and env without executing — cheap way to check an invocation |
| `job.root_dir` | `null` | Rarely; auto-discovered from the `.project-root` marker |

`job.dry_run=true` is the fastest way to see exactly what will be handed to the submodule,
which is worth doing once before a fan-out.

Three fields behave differently from how they read, so set them deliberately:

- **`job.loss_mode` is mandatory if the recipe sets one.** The launcher always appends
  `model.loss_mode=<job.loss_mode or "mse">`, so omitting it overrides the recipe's own
  choice with `mse` and produces a quietly different model. (`job.extra` is appended after
  it, so it can also be set there — but set it *somewhere*.)
- **`job.data_root` is not the dataset directory.** It is forwarded as `--data-root`, which
  the submodule maps onto its dataset-path override — a *root*, not a dataset dir. If the
  recipe requires a dataset path, pass it explicitly via `job.extra`.
- **`job.gpus` does nothing for encoding.** The encoding command builder never reads it, and
  it is tolerated rather than rejected, so there is no error. Pin with
  `CUDA_VISIBLE_DEVICES`.

Also:

- `job.recipe` accepts either a bare recipe name or a **path** (anything containing `/` or
  ending in `.yaml`/`.yml`), which lets an external overlay compose the submodule's internal
  config tree.
- Setting a decoding-only field on an encoding run hits an explicit guard rather than being
  silently ignored. The guard names the offending field, so the message is an accurate
  diagnosis rather than a downstream symptom.
- **Cost is set by the recipe's epochs × steps per epoch.** The rich progress bar's `it/s`
  is not the step rate, and extrapolating from it produces estimates wrong by multiples —
  derive rates from step counters instead.

> **With MLflow:** the submodule's encoding entry point needs a tracking URI only if it
> logs; set `MLFLOW_TRACKING_URI` in the environment or the submodule's `.env`. The launcher
> warns before the heavy boot if it is unset. To log elsewhere or not at all, pass
> `trainer.mlflow_tracking_uri=…` or `trainer.logging=false` via `job.extra`.

## 4. Decoding — `job=downstream_decoding`

`mode=benchmark` multiplexes the benchmark's sessions for one tokenizer. It discovers
cohorts from **local files**, so give the checkpoint a private one-tokenizer root:
`<ckpt_root>/<name>/checkpoints/model.ckpt`.

```bash
export WANDB_MODE=disabled
uv run ntt job=downstream_decoding job.mode=benchmark local=<host> \
  paths.run_name=$slug \
  job.ckpt_root=$tok_root \
  job.gpus=$GPU \
  job.data_root=<brainsets root> \
  job.eval_selection=decoding/ntt_decoding_benchmark_eval \
  job.epochs=<n>
```

`config/job/downstream_decoding.yaml` shares the launcher with encoding, so the same
"`null` means let the submodule decide" rule applies. What differs:

| field | default | override when |
|---|---|---|
| `job.mode` | `single-run` | **Always for the battery** — `benchmark` is the axis this runbook means. `calibrate` is a separate exercise |
| `job.ckpt_root` | `null` | **Always in benchmark mode** — a private one-tokenizer root (below) |
| `job.eval_selection` | `null` → mode default | Scoping to different sessions. In `benchmark` mode it defaults to the benchmark session set |
| `job.epochs` | `null` → recipe's value | Shortening a run deliberately; it is a direct cost lever |
| `job.gpus` | `null` → auto-discover | **Always** — and unlike encoding it *is* honoured, written into the sweep config |
| `job.data_root` | `null` → `./data/processed` in-submodule | Your brainsets live elsewhere, which is usual |
| `job.model_cfg` | `null` | Comparing decoder capacity: `poyo_1.3M`, `poyo_1.3M_no_crossattn`, `poyo_11.8M` |
| `job.batch_size`, `job.num_workers` | `null` | Fitting a specific GPU |
| `job.extra` | `[]` | Passthrough overrides; **appended last**, so they win |

Setting a decoding-only field on an encoding run — or the reverse — hits an explicit guard
rather than being silently ignored, so a misplaced field is an error, not a mystery.

- **Leave `CUDA_VISIBLE_DEVICES` unset here** — the engine pins a device per cell itself.
  Pass `job.gpus`, which the launcher writes into the sweep config. This is the opposite of
  encoding (§3).
- `WANDB_MODE=disabled` is required without a W&B key: the recipe attaches a W&B logger, and
  with no key every cell dies at import.
- **The submodule runs against this checkout's `ntt`**, not the tag it pins.
  Every launch records which `ntt` it imported in `downstream_env.json` in
  its output dir (`<sub>/experiments_benchmark/<tag>/` for benchmark):
  `grep -h '"ntt_file"' <out>/downstream_env.json` should name this checkout's `src/`.
- **Exit status is not a usable success signal here.** A benchmark exits non-zero if *any*
  of its cells fails, so a failure line often accompanies a perfectly good aggregate. Read
  the per-tokenizer cohort record (§5) instead: `metadata.cohort.complete`, `found` and
  `missing` say which cells finished.
- **Cost is set by sessions × epochs**, divided by how many cells run concurrently per GPU.
  `job.epochs` and the eval selection's session count are the levers.

> **With MLflow:** route the experiment through the launcher's own configuration or the
> environment, not through `job.extra` — the submodule's top-level parser rejects an
> experiment-name flag there with `unrecognized arguments` and exit 2, because that flag
> exists only on its inner orchestration parser.

## 5. What each axis produces

Exit code 0 is not verification. Each axis has a positive artifact, and knowing its name is
what makes §8's coverage audit mechanical rather than a judgement call.

| axis | artifact | notes |
|---|---|---|
| reconstruction | `results.json` | the main pass's report; also carries the Poisson comparison |
| RepeatsSNR | its own pass report | **no Poisson family** — see below |
| M1 | `eval_report.json` | raw robustness report |
| M2 | `eval_report.json` | raw robustness report |
| encoding | `downstream_encoding_results.json` | its `test` block is the score: the val-selected checkpoint on the **test** split. No `test` block means the final eval failed or was disabled (`metadata.final_eval` says which) |
| decoding | a parent run per tokenizer, holding the cohort record `downstream_decoding_results.json` at its artifact root, plus one child per session | judge by `metadata.cohort.complete` / `missing`, never the parent's status alone |

All of them land under `job.output_path` / `job.out` locally, and additionally as tracker
artifacts when logging is enabled.

**The two downstream artifacts are ntt-downstream's results records** (schema v2;
format in [downstream `docs/results.md`](https://github.com/enigma-brain/ntt-downstream/blob/main/docs/results.md)).
Aggregation refuses to put val and test encoding scores in one table
(`docs/eval_aggregation.md` § "Downstream inputs").

**The SNR pass carries no Poisson family**, and this is expected rather than a gap: its
inline-baseline path is not wired for the per-stimulus pipeline, so `eval_all` strips
Poisson from that pass entirely.

---

# Part II — Scaling to a cohort

Nothing in ntt evaluates more than one checkpoint per invocation, so everything below is a
pattern for driving Part I repeatedly. **These are worked examples, not a supported
interface.** Pick what fits your cluster and your patience; the only piece worth
standardising is the table in §6, because two other tools already read it.

## 6. The cohort table

One row per tokenizer. **You fill two columns; the battery fills the rest** as each axis
completes:

```csv
tokenizer,ckpt_uri,mlflow_recon,mlflow_latent_snr,mlflow_latent_input_corruption,mlflow_latent_latent_corruption,mlflow_encoding,mlflow_decoding
tok_alpha,/data/ckpts/tok_alpha/checkpoints/last.ckpt,,,,,,
tok_beta,mlflow-artifacts:/12/abc123/artifacts/checkpoints/last.ckpt,,,,,,
```

- `tokenizer` — display name. **Must be filesystem-safe and unique**, because it becomes a
  directory name and a filename stem downstream. The safe form is lowercase alphanumerics
  and underscores; anything else is rewritten, which means a name with a space or a dot
  will not match the name you chose.
- `ckpt_uri` — either an **absolute local path**, or an `mlflow-artifacts:/…` /
  `mlflow:/…` URI. The eval jobs resolve those two schemes themselves, so a URI is a
  first-class value here and needs no manual download. Other schemes (`s3://`, `gs://`,
  `http://`) are **not** resolved and will fail at load time.
- The `mlflow_*` columns start **empty** and take an MLflow run URL
  (`<server>/#/experiments/<id>/runs/<run_id>`) as each axis finishes. An empty cell is skipped-with-a-warning by the consumers, so a partially filled
  table is safe to hand downstream; a *missing column* is silently dropped instead, which is
  why the builder emits them all up front. `mlflow_decoding` takes the benchmark's
  **per-tokenizer parent run**, whose artifact root holds the cohort record (§5).

**Two columns really is the whole contract.** Nothing else is required to drive the
battery: §2's `job.ckpt_path` and §3's `job.ckpt` both accept either form of `ckpt_uri`
directly, and `tokenizer` supplies `paths.run_name` and `tags.model`. The one axis that
needs more is **decoding**, which discovers checkpoints from local files only (§4) — a URI
has to be materialised into `<ckpt_root>/<name>/checkpoints/model.ckpt` first, which is a
local-filesystem step whatever the table says.

### Writing the table by hand

Entirely reasonable for a handful of checkpoints — two columns in any editor or spreadsheet.
Rather than eyeballing it, validate it:

```bash
python scripts/eval/build_cohort.py --check cohort.csv
```

That reports, per row: an unsafe or empty `tokenizer`, a duplicate name, a local path that
does not exist, an unresolvable URI scheme, two rows pointing at the same checkpoint, and
any absent axis column. Exit status is 0 only if the table is usable, so it works as a gate
in a script. Everything it catches is otherwise discovered by whichever job runs first.

Two things to know if you are editing by hand or in a spreadsheet:

- **Quote a field containing a comma**, as normal CSV. CRLF line endings from a spreadsheet
  export are fine.
- **Absolute paths.** A relative path resolves against whatever directory the fan-out
  happens to run from, which is rarely where you wrote the table.

This is deliberately the same table the aggregation layer reads, so there is no second
format to keep in sync. Three caveats on that handoff:

- Aggregation needs MLflow. It only recognises run URLs of the form above; any other cell
  (an artifact URI, a local path) counts as empty, and an empty `mlflow_recon` skips the
  row. So a cell the coverage audit counts as filled can still be ignored by aggregation.
- Aggregation reads the checkpoint from an optional `ckpt_mlflow_uri` column, not
  `ckpt_uri`. Copy the column under that name if you want the checkpoint recorded in the
  scores (`replot_robustness.py` uses it).
- `scripts/analysis/aggregate_runs.py` — a simpler consumer — reads
  `tokenizer` plus a single `results_uri` column instead of the per-axis ones. If you feed
  that script, add `results_uri`. Extra columns are ignored by both, so one table can serve
  both.

### Filling the first two columns

Three routes, all producing the same two columns. `scripts/eval/build_cohort.py` is one
worked example of each:

```bash
# You have a directory of checkpoints
python scripts/eval/build_cohort.py --from-dir /data/ckpts --out cohort.csv

# You have a list — paste paths or URIs, one per line; "name,uri" to name them explicitly.
# Also accepts an existing cohort table, so a hand-edited one can be normalised and re-read.
python scripts/eval/build_cohort.py --from-list checkpoints.txt --out cohort.csv

# You want them discovered from a tracker experiment
python scripts/eval/build_cohort.py --from-mlflow <experiment> --out cohort.csv --verify
```

The point of the split is that the **checkpoint list is the common entrypoint**. A human can
produce one with `ls`, or by pasting from a tracker UI; an agent can produce the same thing
by querying the tracker. Everything downstream consumes the resulting table and neither
knows nor cares which route produced it.

`--verify` applies §0's load check to every row and exits non-zero if any checkpoint fails.
Running it once here is much cheaper than discovering a bad cohort mid-fan-out. It downloads
remote URIs, so it is opt-in.

**Checkpoint artifact layouts vary between runs** — `checkpoints/last.ckpt`, a top-level
`last.ckpt`, or a `last/` / `last-v1/` / `last-v2/` directory. Resolve per run rather than
assuming one. Note also that an artifact named `last` means whatever the training run's
checkpoint callback meant by it, which is not always the final epoch.

### Two decisions that are yours to make

Both should be recorded, because either choice is defensible and neither is discoverable
later from the table alone:

- **`FAILED` training runs.** Their last checkpoint is often a perfectly good tokenizer.
  `build_cohort.py` excludes them unless `--include-failed` is passed, and says so, to force
  the choice rather than making it silently.
- **Duplicate run names.** Reruns and replicates collide. The builder disambiguates with a
  short run-id suffix rather than dropping or overwriting a row, but whether the duplicates
  *are* replicates is a scientific call.

## 7. Fanning out

### A local loop

The simplest thing that works. `eval_all` is CPU-bound (§2), so pack several per GPU and cap
the BLAS pool, then let the cores be the limit:

```bash
export OMP_NUM_THREADS=20 MKL_NUM_THREADS=20
tail -n +2 cohort.csv | while IFS=, read -r tok ckpt _rest; do
  CUDA_VISIBLE_DEVICES=$((i++ % 2)) uv run ntt job=eval_all \
      paths.run_name="$tok" job.ckpt_path="$ckpt" job.output_path="$OUT/$tok" &
done; wait
```

Do not edit a launcher script while it is running: bash reads it incrementally by byte
offset. Under zsh, unquoted expansions are not word-split, so use an array for the loop
variable.

### A SLURM array

There is no SLURM launcher or submitit plugin in this repo; a SLURM run is a batch script that
calls `ntt` exactly as Part I does. The cohort table is already the array index. A generic
example, with placeholder paths and a placeholder paths profile
(`config/paths/example_cluster.yaml`, copied and edited for your site):

```bash
#!/bin/bash
#SBATCH --array=1-24
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
read -r tok ckpt _ < <(tail -n +2 cohort.csv | sed -n "${SLURM_ARRAY_TASK_ID}p" | tr ',' ' ')
uv run ntt job=eval_all paths=my_cluster \
    paths.data_dir=/shared/datasets/brainsets/processed \
    paths.run_name="$tok" job.ckpt_path="$ckpt" job.output_path="$OUT/$tok"
```

Three things are SLURM-specific enough to state:

- **`--cpus-per-task` is a real knob, and the only place SLURM is auto-detected.** The
  Poisson realisation pool sizes itself from `SLURM_CPUS_PER_TASK`, falling back to the
  machine's CPU count (`grep -n "SLURM_CPUS_PER_TASK" -B 4
  src/ntt/evaluation/poisson_realisations.py`). Since the intrinsic axes are CPU-bound, the
  CPU request matters more than the GPU count for them.
- **The downstream child is insulated from the inherited environment.** The launcher strips
  `SLURM_SUBMIT_DIR`, `PYTHONPATH`, `VIRTUAL_ENV`, the conda variables and a fixed list of
  `UV_*` variables before invoking the submodule, so the env the launcher built for it resolves — downstream's lock with this
  checkout's `ntt` overlaid (`grep -n "_STRIP_ENV" -A 14 src/ntt/jobs/downstream.py`). This is load-bearing: `PYTHONPATH` is not a way to influence
  the submodule. Other SLURM variables (`SLURM_JOB_ID`, `SLURM_CPUS_PER_TASK`, ...) are
  inherited.
- **Walltime and preemption interact with what is resumable.** Encoding writes defensive
  checkpoints and can resume; `eval_all` does not checkpoint, so a preempted intrinsic pass
  is lost work rather than delayed work. Size walltime for M2 and prefer partitions where a
  long job will not be requeued mid-pass.

### Concerns common to any cluster

- **Data is usually node-local.** A mounted data path may be present on some nodes only.
  Probe with a tiny task that just lists the mount, on the nodes you will use, or have the job
  stage what it needs. Staging only the decoding benchmark's sessions is a few hundred
  megabytes.
- **Progress lives in the tracker**, because runners typically sit in a multiplexer on the
  node and never write to the scheduler's log stream.

## 8. Verifying coverage

Queue logs lie in both directions: non-zero exits that are partial successes (§4), and runs
killed without ever writing a terminal status. So coverage is not read from logs — it is
read from the table.

Because the battery fills a cell per axis (§6), the audit reduces to **which cells are
empty**. That is mechanical, tracker-independent, and immune to both lies above. The skill
ships it as a runnable snippet —
`.claude/skills/ntt-eval-battery/references/coverage-audit.md`.

Two refinements the empty-cell test does not catch on its own:

- **Decoding presence is not decoding success.** A parent run exists even when children
  failed; read its cohort record's `metadata.cohort.complete` and `missing` (§5).
- **Orphaned in-progress runs** should be marked killed so the experiment does not mislead
  the next reader. Scope that filter carefully, and confirm the process is actually gone
  first.

Tracker HTTP flakiness is the dominant failure mode for long fan-outs — chunked-encoding
errors, SSL EOF, DNS. Configure retries and backoff before starting, not after.

> **With MLflow:** `MLFLOW_HTTP_REQUEST_MAX_RETRIES` is **hard-capped at 10**; a higher
> value raises on the very first request. Pair it with
> `MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR`.

## 9. Aggregating and interpreting the results

Out of scope here, and owned by its own document: **`docs/eval_aggregation.md`**, together
with `job=aggregation` and the scripts under `scripts/analysis/`.

The boundary is worth stating, because it is easy to blur: this runbook's deliverable is a
**complete, verified set of per-axis results** with the cohort table to locate them (§8).
Turning those into cross-cohort tables, category scores, and comparisons is a separate
concern with different failure modes.

---

# Part III

## 10. When the battery needs a workaround

Evaluating a tokenizer trained with another version of ntt can require something the
workflow above does not describe: using staged checkpoint copies, hand-setting a config value
that is not inferred, or working around a launcher that reverts an environment.

These cases are catalogued in
`.claude/skills/ntt-eval-battery/references/known-pitfalls.md`, with, for each one, what it
looks like and what to do about it. Record any workaround you apply alongside the results.
