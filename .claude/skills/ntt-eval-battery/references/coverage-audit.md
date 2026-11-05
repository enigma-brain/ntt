# Coverage audit

Run this after the fan-out, before reporting anything. It answers one question: which
tokenizers are missing which axis?

## The rule

Audit the **cohort table**, never the logs. Queue logs lie in both directions:

- **Non-zero exit, real success.** A decoding benchmark fails its invocation if *any* cell
  fails. All-but-one cell succeeding still yields a usable aggregate.
- **Zero exit or no exit, real failure.** Runs killed by the scheduler, OOM, or a node
  reclaim never write a terminal status at all.

Because the battery fills one cell per axis as each run completes (runbook §6), coverage is
just **which cells are empty**. No tracker query, no per-axis predicate, nothing to keep in
sync with a tracker's API.

## The audit

```python
import csv

AXES = [
    ("recon",    "mlflow_recon"),
    ("snr",      "mlflow_latent_snr"),
    ("m1",       "mlflow_latent_input_corruption"),
    ("m2",       "mlflow_latent_latent_corruption"),
    ("encoding", "mlflow_encoding"),
    ("decoding", "mlflow_decoding"),
]

with open("cohort.csv", newline="") as f:
    rows = list(csv.DictReader(f))

total, ok = len(rows), True
for label, column in AXES:
    missing = sorted(r["tokenizer"] for r in rows if not (r.get(column) or "").strip())
    ok &= not missing
    print(f"{'OK ' if not missing else '!! '}{label:9s} {total - len(missing)}/{total}",
          missing or "")

print("\nCOMPLETE" if ok else "\nINCOMPLETE — do not report these results as final")
```

A column that is **absent** rather than empty reports every tokenizer as missing that axis,
which is the correct alarm: the consumers silently drop an axis whose column does not exist,
so an absent column is a worse failure than an empty one, not a lesser one.

## Two things the empty-cell test cannot catch

**Decoding presence is not decoding success.** A parent run exists even when its children
failed. For each decoding cell, read the cohort record at the parent's artifact root
(`downstream_decoding_results.json`, runbook §5): `metadata.cohort.complete` must be true, and
`missing` names any cell that did not finish.

A parent with a terminal status and zero metrics *seconds* after starting is **normal** — it
is a placeholder, re-opened to aggregate at the end. Judge liveness from the children.

**A filled cell can point at a dead run.** If a fan-out was interrupted, a cell may name a
run that never finished. Spot-check that each axis's artifact actually exists — runbook §5
names them per axis. For encoding, a `downstream_encoding_results.json` without a `test`
block is a run whose final eval failed: it gives no encoding score.

## Cleaning up orphans

Runs left in an in-progress state after their process is gone make the experiment misleading
to the next reader. Mark them killed — but **scope the filter carefully**, because it is easy
to catch a run that is still alive. Confirm the process is gone before touching anything;
when in doubt, leave it and say so.

## Reporting

State the fraction covered per axis and name every gap. If any axis is incomplete, say so
plainly rather than reporting the covered subset as if it were the cohort.

Two things also belong in the report, and both are easy to omit because the numbers look fine
without them:

- **Scope decisions that shrank coverage deliberately** — dropped Poisson sessions (runbook
  §1), excluded `FAILED` training runs (runbook §6).
- **Every Part A workaround you applied** from `known-pitfalls.md`. Those are workarounds, not
  steps of the supported workflow; a report that omits them overstates how routine the run was.

Hand the table itself on as the deliverable — `docs/eval_aggregation.md` consumes it directly
(runbook §9), so a verified table is the artifact, not a prose summary of one.
