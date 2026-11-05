---
name: ntt-eval-battery
description: Runs the full ntt tokenizer evaluation battery — reconstruction, RepeatsSNR, input- and latent-corruption robustness, encoding, and decoding — over one trained checkpoint or a whole cohort, locally or on a cluster. Use when scoring, evaluating, or benchmarking trained tokenizers; when running eval_all, downstream_encoding, or downstream_decoding; when building a cohort table or checkpoint list from an experiment tracker; when auditing which evaluation axes a cohort is missing; or when an eval fails on checkpoint loading, neuron/window padding, Poisson-baseline validation, or a partial-cell benchmark exit.
---

# Running the evaluation battery

Six metric families, four job invocations. **Every job takes exactly one checkpoint** —
there is no multi-tokenizer entrypoint in ntt, so a cohort is a loop you write.

| axis | job | pass |
|---|---|---|
| reconstruction | `job=eval_all` | 1 |
| RepeatsSNR | `job=eval_all` | 2 |
| M1 input-corruption robustness | `job=eval_all` | 3 |
| M2 latent-corruption robustness | `job=eval_all` | 4 |
| encoding | `job=downstream_encoding` | — |
| decoding | `job=downstream_decoding job.mode=benchmark` | — |

## Two inputs only the user can give you

**If either was not stated when this skill was invoked, ask before running anything.**
Neither is guessable, and getting either wrong fails expensively rather than obviously — a
wrong data path produces a confusing error hours in, and the wrong checkpoints produce
results that look fine and mean nothing.

**1. Where to run — this machine's data path.** The `local=<host>` entry naming the
brainsets root (runbook §0). Do not invent one, and do not fall back to omitting it:
the eval jobs (`eval_all`, `eval`, the robustness jobs) override the paths group to a config
with a fixed `data_dir`, so a missing `local=` silently reads from a path that is not yours.
If an entry under `config/local/` already fits, confirm which. If not, ask
for the path and add a one-line overlay. Either way, confirm the resolved value with
`--cfg all | grep data_dir` before launching, not after.

Ask for **one per machine** if the work spans several. The paths differ, and so does whether
the data is present at all — cluster storage is often node-local (runbook §7).

**2. What to run on — the checkpoints.** One of:

- one or more **tracker experiments** → `build_cohort.py --from-mlflow <experiment>`
- a **local directory** of checkpoints → `--from-dir <dir>`
- an explicit **list of names and paths** → `--from-list <file>`

Ask which, and for the identifiers. Do not scan the filesystem hoping to find checkpoints,
and do not pick an experiment because its name looks plausible.

Then **show the cohort table before launching anything** (runbook §6). Two calls in it
belong to the user: whether `FAILED` training runs are included, and whether duplicate run
names are replicates or reruns.

**Secondary, but settle it if unstated:** which axes are in scope (all six, or intrinsic
only), and whether a Poisson baseline already exists to reuse — recomputing one is the most
expensive single step in the battery, and its parameters must match the eval configs
(runbook §1).

---

**Two documents back this skill, and the split matters:**

- **`docs/eval_battery_runbook.md`** — the supported workflow. **Part I (§0–§5) is one
  tokenizer; Part II (§6–§9) scales to a cohort.** Read the named section before each step
  rather than improvising from the summary here.
- **`references/known-pitfalls.md`** — troubleshooting notes: symptoms, causes and what to do
  for the checkpoint, data-config and environment issues that come up when running the battery.

When you apply a Part A workaround from the troubleshooting notes, **say so in your report**,
so the result is read with it in mind.

## Published models: use the published protocol

To re-score a tokenizer from the Hugging Face repo (`the-enigma-project/ntt-tokenizers`) as its
published scores were computed, do not rebuild the commands from this skill. Run
`scripts/eval/published_commands.py <model> --data-dir <brainsets> --output-dir <out>
--revision <rev>`: it prints (or `--run`s) the exact commands from the protocol files
(`scripts/eval/published_protocol.yaml`, `config/job/published_eval*.yaml`,
`config/downstream/encoding/published_*.yaml`), with `hf://` checkpoints and the published
Poisson references. `--tracker <dir>` then writes the aggregation input from the outputs;
local paths need no tracker server. `docs/pretrained_models.md` §2 is the user-facing
version.

## Order of operations

**Do one tokenizer completely before touching the cohort.** This is not caution for its own
sake: several failure modes only surface after an expensive earlier stage succeeds, so
discovering them once beats discovering them once per cohort member. Part II's fan-out
assumes a verified working invocation, and producing one is what Part I is for.

0. **Have both inputs above.** Ask if you were not told; do not infer them.
1. **Preflight** — work `references/preflight-checklist.md` to completion.
2. **Prove the checkpoint loads** with this checkout. Runbook §0.
3. **Decide the Poisson baseline** — reuse, scope, or recompute. Runbook §1.
4. **Run all four intrinsic passes** on one checkpoint, then encoding, then decoding, and
   check each one's artifact. Runbook §2–§5.
5. **Only now build the cohort table** and fan out. Runbook §6–§7.
6. **Audit coverage** — which cells are empty. Runbook §8, and
   `references/coverage-audit.md`.
7. **Aggregate and interpret** — a separate concern with its own owner,
   `docs/eval_aggregation.md`. Runbook §9.

## The cohort table is the interface

One row per tokenizer. Two columns are filled by whoever starts the work; the battery fills
one cell per axis as runs complete (runbook §6):

```
tokenizer,ckpt_uri,mlflow_recon,mlflow_latent_snr,mlflow_latent_input_corruption,…
└─ human or agent fills these ─┘└──────── the battery fills these ────────┘
```

The **checkpoint list is the common entrypoint**: a human produces one with `ls` or by
pasting from a tracker UI, an agent produces the same thing by querying the tracker, and
nothing downstream cares which happened. `scripts/eval/build_cohort.py` does all three
(`--from-dir`, `--from-list`, `--from-mlflow`), `--verify` applies the §0 load check to every
row, and `--check <table>` validates a hand-written one (unsafe or duplicate names, missing
paths, unresolvable URI schemes, absent axis columns) before anything launches.

Two columns is the entire contract: `ckpt_uri` may be an absolute local path **or** an
`mlflow-artifacts:/` URI, which §2 and §3 resolve themselves. Only decoding needs it
materialised locally (runbook §4).

## What drives cost

Wall-clock varies by more than an order of magnitude with hardware and allocation, so plan
against the knobs, not remembered durations, and measure one run before committing to a
fan-out.

| axis | what sets it | knobs |
|---|---|---|
| `eval_all` (4 passes) | **CPU cores**, not GPU | `OMP_NUM_THREADS`/`MKL_NUM_THREADS` (~20, or one run eats the box), `max_eval_seconds`, `max_neurons`. M2 is the longest pass |
| encoding | recipe epochs × steps/epoch | recipe epochs, GPU class |
| decoding | sessions × epochs, cells per GPU | `job.epochs`, session count, `job.gpus` |
| Poisson baseline | realisations × sessions × windows/session | `n_workers` (CPU budget is the main lever), `n_realisations`, `window_length_seconds` |

**A progress bar's `it/s` is not the step rate.** Extrapolating from it produces estimates
wrong by multiples, especially for encoding. Derive rates from step counters.

## Non-negotiables

- **Always pass `local=<host>`.** The eval jobs default `paths.data_dir` to a fixed path
  that is not yours; `--cfg all` shows the resolved value in seconds. Runbook §0.
- **Prefer `job=eval_all` over the four standalone jobs**, because it pairs each pass with
  the right `evaluation`/`data` config. It is only a wrapper, so a single failed pass can be
  re-run with `job.<pass>.enabled=false` on the others. Runbook §2.
- **Exit code 0 is not verification.** Require the axis's positive artifact — runbook §5
  names each one.
- **A non-zero benchmark exit is not failure.** One failed cell out of many fails the whole
  invocation. Read the child aggregate and the parent's metric count. Runbook §4, §8.
- **Coverage comes from the table, not the logs.** Runbook §8.
- **Never edit a launcher script that is running.** Appending is safe; inserting makes bash
  resume mid-text and execute garbage.
- **Report what you worked around, not just what you ran.**

## Tracker independence

The runbook assumes a tracker but not *which* one, and for a single tokenizer barely needs
one. MLflow is the worked example, always inside a `> **With MLflow:**` block — those blocks
are what you replace. See "The tracker's four jobs" at the top of the runbook.

## Working practices

- Rest every "done" claim on a metric, an artifact, or a tracker row.
- Most blockers surface as one opaque exception hours in. Work the failure; do not guess.
- The cluster side (runbook §7) is self-contained. Read it together with Part B of the
  troubleshooting notes.
