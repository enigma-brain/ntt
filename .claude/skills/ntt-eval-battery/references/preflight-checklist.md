# Preflight checklist

Work every item to a **yes** before launching. Each one is a failure that otherwise surfaces
hours in, after real compute has been spent.

Create one todo per line.

**Stage 1 applies to a single tokenizer and is never skippable.** Stage 2 only applies once
you scale to a cohort — skip it entirely for a one-off eval.

---

# Stage 1 — before the first tokenizer

## Environment

- [ ] `set -a && source .env && set +a` — tracker credentials and any tokens loaded. Keep
      them out of commits and logs.
- [ ] `uv sync` run **in your checkout**, so the venv runs the `src` you are editing.
- [ ] `git submodule update --init --force downstream` and
      `uv run python -m ntt.jobs.downstream_env` (builds the submodule's env with this
      checkout's `ntt` overlaid). Needed for encoding and decoding only — skip if you are
      running intrinsic evals alone. Never `uv sync --project downstream --frozen` by hand
      (runbook §0).
- [ ] A `local=<host>` entry exists for this machine and resolves to the right brainsets
      root, checked with `--cfg all | grep data_dir` rather than assumed. Without it,
      `job=eval_all` points at a fixed path that is not yours (runbook §0).

## Checkpoint loading — the big one

- [ ] Your checkout includes the code that trained the checkpoint, or at least every spec
      field and module its config references (see `known-pitfalls.md` A4).
- [ ] The checkpoint loads, proven by running this and seeing `OK`:

      uv run python -c "
      from ntt.core.model import load_tokenizer_from_checkpoint
      tk = load_tokenizer_from_checkpoint('<one.ckpt>', map_location='cpu', weights_only=False)
      print('OK', type(tk).__name__, tk.spec.KIND)"

If it fails, the two shapes it takes have different causes and get misdiagnosed as each
other — see `known-pitfalls.md` A4.

## Does the checkpoint carry a training-only loss?

Only matters if you are running encoding; decoding and intrinsic evals are unaffected.

- [ ] Checked whether the loss spec references an absolute path to a training-only asset. If
      it does not, skip the rest of this block.
- [ ] If it does: applied the sanitisation workaround from `known-pitfalls.md` **A1** to a
      staged *copy*, never the original.
- [ ] **Noted that you did this**, so it appears in the final report.

## Poisson baseline

- [ ] Decided: reuse an existing baseline, scope the consumer to its sessions, or recompute.
- [ ] If recomputing: the **same `data` and `evaluation` configs** the eval will use, plus
      the same `split` and rate bin. The hard fields are derived from those, so matching the
      configs makes them agree by construction (runbook §1).
- [ ] For a cohort: all tokenizers share one `bin_size_ms`. They cannot share a baseline
      otherwise, because that field comes from the tokenizer spec.
- [ ] If scoping: the dropped sessions are **written down**. It changes what the Poisson
      column means.
- [ ] Hard fields match between baseline and consumer — `bin_size_ms`,
      `rate_estimation_bin_ms`, `split`, `max_eval_seconds`, `decoding_bypass_fields`. A
      consumer session set that is a *superset* of the baseline's raises.

## Distribution-valued spec fields

Inspect the spec rather than assuming this does not apply. This is described in
`known-pitfalls.md` **A2**.

- [ ] Is `n_neurons` a distribution? Then `pad_neurons_to_fixed: <ceiling>` is set in **all
      four** `eval_all` pass data configs — each pass composes a different one. Setting
      `n_neurons:` in a data config instead does nothing (A2).
- [ ] Noted for the report, with the fix recommendation.

## Resource plan

- [ ] `OMP_NUM_THREADS` / `MKL_NUM_THREADS` capped (~20). `eval_all` is CPU-bound; one
      uncapped run takes most of the box.
- [ ] Encoding: pinned with `CUDA_VISIBLE_DEVICES`. `job.gpus` is a no-op there.
- [ ] Decoding: pinned with `job.gpus`, and `CUDA_VISIBLE_DEVICES` **not** set — the engine
      pins per cell itself.
- [ ] `export WANDB_MODE=disabled` for decoding, or every cell dies at import.

## The gate

- [ ] All four intrinsic passes have run on **one** checkpoint, and each produced its
      artifact (runbook §5) — not merely exited 0.
- [ ] Encoding and decoding likewise, if they are in scope.

---

# Stage 2 — before fanning out to a cohort

## The cohort table

- [ ] `cohort.csv` exists with `tokenizer` and `ckpt_uri` filled, and the axis columns
      present-but-empty. An absent column is silently dropped downstream; an empty one is
      skipped with a warning.
- [ ] Every `tokenizer` value is unique and filesystem-safe.
- [ ] **Every** checkpoint loads, not just the one from Stage 1 —
      `build_cohort.py --verify` does this. A cohort-wide load failure found mid-fan-out is
      the worst available outcome.

## The two decisions that are the cohort owner's

- [ ] A recorded call on whether `FAILED` training runs are in or out. Either is defensible;
      silence is not.
- [ ] A recorded call on duplicate run names: replicates (keep both) or reruns (keep one).
- [ ] If an agent built the table, the human has **seen it** before anything launched.

## Fan-out plan

- [ ] Unique `tags.model` **and** unique output directory per encoding run, or runs collide
      in the tracker and on disk.
- [ ] Encoding/decoding jobs sharing **one** checkout: its downstream env is built once
      (`uv run python -m ntt.jobs.downstream_env`), and every job runs with
      `job.downstream_env=verify`, so none builds mid-fan-out.
- [ ] Tracker retry/backoff configured. HTTP flakiness is the dominant failure mode for long
      fan-outs.
- [ ] On a cluster: walltime sized for M2, the longest pass, and the resumability of each
      axis understood — a preempted `eval_all` is lost work, not delayed work.
- [ ] Launcher script will not be edited while running. Appending is safe; inserting is not.
