# Downstream evaluation — maintainer internals

How the `ntt` ⇄ `ntt-downstream` integration is built. For running evals, see [downstream_eval.md](downstream_eval.md).

## How downstream is connected to ntt

`ntt-downstream` already depends on `ntt`. To avoid a circular dependency — and
to keep the heavy `torch==2.8.0+cu126` / `flash-attn==2.8.3` stack out of
`ntt`'s base environment — `ntt-downstream` is vendored as a **git submodule**
under `downstream` with its **own** venv. The launcher
(`src/ntt/jobs/downstream.py`, `NTTDownstreamJob`) **never imports
`ntt_downstream`**: it resolves your arguments, builds a clean child
environment, and runs the downstream entrypoints as a **subprocess** inside the
submodule's venv.

Concretely, every mode shells into one of the submodule's four **versioned
console scripts** in that venv, which the launcher overlays with **this
checkout's `ntt`** (see [Step 2](#step-2--the-submodules-venv-the-host-ntt-overlay)):

```bash
uv run --project downstream --no-sync ntt-downstream-{decode,encode,calibrate,benchmark} \
    <promoted --flags> [--config-name X | --config-dir D] [key=value …]
```

This is the submodule's **stable CLI contract** (`downstream/docs/cli.md`). The
launcher (ntt side) only forwards promoted `--flags` plus raw `key=value` Hydra
passthrough; the launcher *on the submodule side* owns the
promoted-flag→Hydra-key mapping and the contract's architecture defaults
(pinned in `configs/cli/decode.yaml`; the three values are enumerated under the
single-run [Arch bullet](downstream_eval.md#1-decoding--modesingle-run)). There is no bash and no
dependency on a mutable `.sh` env-var contract.

The launcher subclasses the lightweight `BaseJob` (not `NTTBaseJob`), so it
never builds a tokenizer or datamodule itself — all heavy work happens in the
child process.

> **Repin note.** After pulling a submodule bump (or a change to this repo's
> `pyproject.toml`), the next launch with `job.downstream_env=auto` rebuilds the
> submodule's venv (see [Step 2](#step-2--the-submodules-venv-the-host-ntt-overlay)).

## One-time setup — details

User-facing steps: [downstream_eval.md § Quick start](downstream_eval.md#quick-start).

### Step 1 — the git submodule

`ntt-downstream` is wired in as a submodule in `.gitmodules`:

```ini
[submodule "downstream"]
	path = downstream
	url = https://github.com/enigma-brain/ntt-downstream
	# The gitlink records the pinned commit. `branch` names the downstream …
	branch = <release tag>
```

The submodule is pinned to an **exact commit** — the gitlink the host repo
records is always the source of truth. The
`branch = <release tag>` line names the downstream release that pin belongs to, with a
**tag** (not a moving branch), so it is human-legible in `.gitmodules`. The pin is either
the tag's commit or, when both repos are released together, the commit just before it:
downstream's lock can only record ntt's release commit once that exists, so the tag lands
on the commit that adds it. The two differ only in downstream's `uv.lock`, which the
launcher does not read (Step 2). Because tags don't move, `git submodule update --remote`
stays reproducible: it always checks out the tag's commit. (Never point `branch` at a
mutable branch like `main`; that *would* let `--remote` silently walk the pin off the tip.)

- **Existing checkout:** `git submodule update --init downstream` clones it into
  `./downstream` and checks out the pinned commit. (Drop the path to init *all*
  submodules.)
- **Fresh clone:** add `--recurse-submodules` to `git clone` to pull it in one
  shot, or run the command above afterwards.
- **Bumping the pin later:** check out the new tagged release inside the
  submodule and record it in the host repo —
  `git -C downstream fetch --tags && git -C downstream checkout <new-tag> && git add downstream`,
  then update the `branch = <new-tag>` line in `.gitmodules` to match, and commit
  both. Keep `branch` pointed at a **tag**, never a mutable branch.

  The reverse pin (downstream → ntt) lives in `ntt-downstream`'s
  `pyproject.toml` under `[tool.uv.sources] ntt = { …, tag = "vX" }`. Launched
  from here, that pin is **not** what runs (the launcher overlays this checkout's
  `ntt`, see Step 2); it matters for standalone use of downstream and as the
  dependency baseline the overlay starts from. Bump it (+ `uv lock`) when the
  overlay check fails (see below) or downstream should track a newer ntt
  release. The full cross-bump cycle is: tag ntt → bump downstream's `ntt` tag +
  relock + tag downstream → repoint ntt's submodule at the new downstream tag.

After this step `downstream/pyproject.toml` exists; the launcher fails fast with
the `git submodule update --init downstream` hint if it doesn't.

### Step 2 — the submodule's venv (the host-ntt overlay)

The submodule runs against **this checkout's `ntt`** — the code that wrote your
checkpoints — not the `ntt` tag its lockfile pins (which can be older than the
checkpoint format). So the launcher owns `downstream/.venv`. Before every real launch
(`NTTDownstreamJob._ensure_env`) it checks, read-only:

- `downstream/.venv/bin/python` imports `ntt` from **this checkout's `src/`**;
- the launcher built it, from the current `pyproject.toml` here and the current
  `downstream/uv.lock`: a stamp, `downstream/.venv/ntt-overlay.sha256`, holds a
  hash of those three files. A missing stamp fails the check too (an env built
  by hand, or by another tool, can't be trusted to be current);
- `uv pip check` passes, apart from the two incompatibilities downstream's own
  `[tool.uv] override-dependencies` create on purpose, and nothing else:
  - a package *other than ntt* requiring an overridden package (for example
    vector-quantize-pytorch's `einops>=0.8.0` against the `einops<0.7`
    override; downstream's own lock ships that pair);
  - ntt requiring a package an override removes, which is then not installed
    (for example `pytorch-brain`, replaced by git `torch_brain`).

  If **ntt** requires a version an override prevents (say this checkout raises
  its floor to `einops>=0.8`), the override would silently win, so that fails.
  Output the launcher can't parse fails too;
- every package the overlay installed is in downstream's lock (see the
  new-dependency rule below).

If a check fails, `job.downstream_env=auto` (the default) builds the overlay and
checks again. The build, run from inside the submodule, is:

```bash
cd downstream
# The downstream lock minus ntt: ntt itself is installed from this checkout, below.
uv sync --project . --frozen --no-install-package ntt --python 3.13
# This checkout, editable, WITH its dependencies (its floors win over the lock).
uv pip install --python .venv/bin/python --no-sources --resolution lowest-direct -e ..  # .. = this checkout
```

To build it **outside a launch** (before a fan-out, or to see the result), run
`uv run python -m ntt.jobs.downstream_env` from this checkout (`--root-dir` /
`--submodule-dir` for another layout, `--verify` to check only). It runs the
same check-then-build as an `auto` launch, writes the stamp, and prints the
provenance record. The two commands above are for debugging only: they write
no stamp, so the next `auto` launch rebuilds and `verify` refuses the env.

- **Only what this checkout requires moves.** The packages already installed
  from the lock are preferences, and `--resolution lowest-direct` moves a
  locked dependency only when it is below this checkout's floor, and only to
  that floor. For example, mlflow
  (+ mlflow-skinny/-tracing), plotly, pydantic-settings and wandb can move up to this
  checkout's floor. torch, numpy,
  lightning, temporaldata, torch-brain, brainsets and the tokenizer numerics
  (vector-quantize-pytorch, einops, flash-attn) stay as locked; a move in one of
  those is logged as a warning.
- **New dependencies are refused.** For a dependency of this checkout that is
  **not in downstream's lock**, `lowest-direct` installs its *oldest* release
  (an unconstrained requirement resolves to a very old release, which may not
  build on Python 3.13). So the check fails when any installed package other
  than ntt is missing from the lock, naming it. The fix is the same as for a
  conflict (below).
- **`--no-sources`:** this repo's `[tool.uv.sources]` pin git revisions for
  ntt's own dev env (one is an unpinned branch). They must not override the
  versions downstream's lock already chose.
- **Run it from `downstream/`:** `uv pip` takes downstream's
  `override-dependencies` from the `pyproject.toml` in the working directory.
  Without them it installs PyPI `pytorch-brain` over the git `torch_brain`.
- **Provenance.** Every real launch logs the resolved `ntt.__file__` and the
  package changes vs. the lock at INFO, and writes both to
  **`downstream_env.json` in the run's output directory**: `job.out` for
  single-run, calibrate and encoding (or their default), and
  `downstream/experiments_benchmark/<tag>/` for benchmark. To confirm a run used
  this checkout: `grep -h '"ntt_file"' <out>/downstream_env.json` (or grep the
  job log for `ntt.__file__`).
- **Never `uv run --frozen` / `uv sync` the submodule by hand** once the overlay
  is in place: syncing to the lock swaps the overlay back for the pinned tag. The launcher's console-script
  calls use `uv run --no-sync`. The next `auto` launch repairs an env that was
  synced by accident.
- **When the build or the check after it fails**, this checkout's requirements
  conflict with the downstream lock beyond what the overlay may change (or add
  a package the lock lacks). The launcher raises with the failing check, or
  with the tail of uv's output if a build step failed, and does not patch the
  env in place. The fix is upstream: bump downstream's `ntt` pin, relock,
  release, and bump the submodule here.
- **Building the env needs network access** (PyPI and the public GitHub
  repos that downstream's lock references) and Python 3.13, which uv downloads
  if needed. On a cold
  uv cache it takes ~1–3 min.
**`job.downstream_env`: `auto` or `verify`.** Use `auto` by default. If many jobs (for example under a cluster scheduler) share one checkout, build once and launch them with `verify`. The check only reads files, so jobs can run it together; only a build writes to `downstream/.venv`.

Concurrent `auto` builds in one checkout use the lock directory `downstream/.venv.ntt-overlay.lock`. Creating a directory is atomic on network filesystems, unlike `flock`. The first job builds; the others wait up to 40 minutes, check again, and skip building if the environment is ready. A lock older than 30 minutes is treated as left by a killed build and removed. If a job times out, it reports the lock and its owner (host and process ID); remove the directory if no build is running.

`verify` only checks and reports the build command instead of building. For many jobs sharing one checkout, build once with `uv run python -m ntt.jobs.downstream_env` or one `auto` launch, then use `job.downstream_env=verify` for the other jobs. Jobs with separate checkouts can use `auto`.

> All four entrypoints pass checks against the real submodule, including a contract smoke test that composes encode.

Downstream documents the same split from its side: `downstream/docs/getting-started.md`
§ "Standalone vs. embedded in `ntt`" and `downstream/docs/cli.md` § "For ntt
integrators".

## Underlying command per setting

What the launcher runs in the submodule's environment for each setting.

### 1. Decoding — `mode=single-run`

- **Underlying command** (promoted flags first, then Hydra passthrough last):
  ```text
  uv run --project downstream --no-sync ntt-downstream-decode \
    --tokenizer-kind frozen --tokenizer-ckpt <abs ckpt | URI> \
    --data-root <abs data_root> --output-root <abs out> [--seed N] \
    session=<session> [epochs=N batch_size=N num_workers=N] \
    [model=<model_cfg> tokenizer.unit_embedding_mode=<mode> tokenizer.sentinels=<bool>] \
    wandb.enabled=false <job.extra …>
  ```

### 2. Decoding — `mode=calibrate`

- **Underlying command** — promoted flags and a generated sweep YAML file are passed through `--config-dir` and `--config-name`:
  ```text
  uv run --project downstream --no-sync ntt-downstream-calibrate \
    --tokenizer-kind frozen --tokenizer-ckpt <abs ckpt | URI> \
    --data-root <abs data_root> --output-root <abs out> \
    --config-dir <abs out> --config-name ntt_calibrate [--seed N] <job.extra …>
  ```
  The launcher writes `<out>/ntt_calibrate.yaml` with settings not passed as promoted flags: `train_sessions`, `other_sessions`, `gpus`, `batch_size`, `num_workers`, `model_cfg`, `unit_embedding_mode`, and `sentinels`. It writes this file even in `dry_run`, so you can rerun the printed command as-is.

### 3. Decoding — `mode=benchmark`

- **Underlying command** — promoted flags plus a generated sweep YAML:
  ```text
  uv run --project downstream --no-sync ntt-downstream-benchmark \
    --data-root <abs data_root> \
    --config-dir <repo>/results/downstream/sweep_configs --config-name <tag> [--seed N] <job.extra …>
  ```
  The launcher writes `results/downstream/sweep_configs/<tag>.yaml` with `ckpt_root`, `comparison_tag`, `sessions`, and any of these that you set: `gpus`, `epochs`, `batch_size`, `num_workers`, `model_cfg`, `unit_embedding_mode`, or `sentinels`. It does **not** pass `--tokenizer-ckpt` or `--tokenizer-kind`; benchmark gets the tokenizer cohort from `ckpt_root`.

### 4. Encoding — `job=downstream_encoding`

- **Underlying command** (promoted flags first, then frozen-tokenizer positionals, then `job.extra` last):
  ```text
  uv run --project downstream --no-sync ntt-downstream-encode \
    --tokenizer-ckpt <abs ckpt | URI> --output-root <abs out> \
    [--data-root <abs data_root>] [--seed N] [--config-name <recipe>] \
    model.loss_mode=<loss_mode> ++tokenizer.checkpoint_first=true \
    trainer.tokenizer_mode=frozen <job.extra …>
  ```
  The submodule maps `--tokenizer-ckpt` to `tokenizer.checkpoint_path` and `tokenizer.enabled=true`. It maps `--output-root` to `hydra.run.dir`. ntt sets the last three tokens—`model.loss_mode`, `++tokenizer.checkpoint_first=true`, and `trainer.tokenizer_mode=frozen`—to ensure the tokenizer stays frozen.

## What maps to what

| `ntt …` | Console script invocation (run in the submodule's venv) |
|---|---|
| `job=downstream_decoding job.mode=single-run` | `ntt-downstream-decode --tokenizer-kind frozen --tokenizer-ckpt … --data-root … --output-root … [--seed N] session=… [knobs…] wandb.enabled=false <extra>` |
| `job=downstream_decoding job.mode=calibrate` | `ntt-downstream-calibrate --tokenizer-kind frozen --tokenizer-ckpt … --data-root … --output-root … --config-dir <out> --config-name ntt_calibrate [--seed N] <extra>` (+ generated `<out>/ntt_calibrate.yaml`) |
| `job=downstream_decoding job.mode=benchmark` | `ntt-downstream-benchmark --data-root … --config-dir <repo>/results/downstream/sweep_configs --config-name <tag> [--seed N] <extra>` (+ generated `<tag>.yaml`) |
| `job=downstream_encoding` | `ntt-downstream-encode --tokenizer-ckpt … --output-root … [--data-root …] [--seed N] [--config-name <recipe>] model.loss_mode=<loss> ++tokenizer.checkpoint_first=true trainer.tokenizer_mode=frozen <extra>` |

Each is invoked as `uv run --project downstream --no-sync ntt-downstream-<verb> …`
(in the [overlaid venv](#step-2--the-submodules-venv-the-host-ntt-overlay)).
The launcher resolves all checkpoint / data / output paths to **absolute** (the
contract scripts `cd` into the submodule), and strips env vars that would leak
`ntt`'s venv or mis-direct the scripts (`VIRTUAL_ENV`, `UV_PROJECT_ENVIRONMENT`,
`PYTHONPATH`, `SLURM_SUBMIT_DIR`, …). It then sets `NTT_SPLITS_DIR` to this
repo's selections and merges the submodule's `.env` (your exported values win).

## The ntt ⇄ ntt-downstream boundary

This integration has a deliberate seam. Knowing which side of it a knob lives on
tells you whether it is **stable** (safe to script against; only changes on a
contract major bump) or **unversioned** (an escape hatch that may need updating
when the submodule pin moves).

### What is ntt-facing and stable

Versioned by the **submodule pin** (the contract is released as a git tag; ntt
records the gitlink). These are safe to depend on:

- **The four verbs** — `ntt-downstream-{decode,encode,calibrate,benchmark}`.
- **The `ntt` job fields** — everything in the knob tables above
  (`job.ckpt`, `job.session`, `job.recipe`, …). These are ntt's own surface.
- **The promoted flags** the job fields ride on: `job.ckpt` → `--tokenizer-ckpt`,
  `job.data_root` → `--data-root`, `job.out` → `--output-root`, `job.seed` →
  `--seed` (plus `--tokenizer-kind frozen`). The flag **names** are versioned;
  the Hydra keys they map to internally are not (and differ per verb).
- **`--dry-run` semantics** — the GPU-free preview every verb honours.
- **The results records** — `downstream_{encoding,decoding}_results.json`
  (schema v2, [downstream `docs/results.md`](https://github.com/enigma-brain/ntt-downstream/blob/main/docs/results.md)).
  ntt's aggregation reads only these (`ntt.evaluation.aggregation.downstream_records`);
  `run_result.json` and the CSVs are downstream-internal.

**Who owns the env:** ntt does. Launched from here, the submodule's venv is the
downstream lock minus `ntt`, plus this checkout's `ntt` installed editable
([Step 2](#step-2--the-submodules-venv-the-host-ntt-overlay)). The submodule's
own `ntt` pin applies only to standalone use of downstream.

### What is ntt-downstream-internal and unversioned

These are **not** part of the contract's stability promise. They are the
submodule's Hydra/engine surface, and can change between submodule pins:

- **Hydra config keys** passed as raw `key=value` passthrough — `model.*`,
  `trainer.*`, `data.*`, `tokenizer.*`, `session=`, `epochs=`, `batch_size=`, …
  (including the three tokens ntt itself appends to encode:
  `model.loss_mode=`, `++tokenizer.checkpoint_first=true`,
  `trainer.tokenizer_mode=frozen`).
- **Engine argparse dests** that become **keys in the generated sweep YAML** for
  calibrate/benchmark — `train_sessions`, `other_sessions`, `sessions`,
  `ckpt_root`, `comparison_tag`, `gpus`, `batch_size`, `num_workers`,
  `model_cfg`, `unit_embedding_mode`, `sentinels`, `epochs` (benchmark only).

### What crosses the boundary

Three things deliberately reach **across** the seam into internal surface — they
are **power-user escape hatches**, not contract surface:

- **`job.extra`** — appended verbatim as trailing Hydra `key=value` tokens
  (last, so Hydra last-wins). It targets internal config keys directly.
- **Path-form `job.recipe`** — an external Hydra config (see
  [External recipes](downstream_encoding_advanced.md#external-recipes-full-configs-outside-the-submodule)),
  emitted as `--config-dir`/`--config-name`. Hydra adds the dir to the search
  path so the file composes the submodule's internal tree; every key inside it
  is internal/unversioned surface.
- **The sweep-config file keys** — when you set `job.gpus`, `job.model_cfg`,
  etc. for calibrate/benchmark, they land in `ntt_calibrate.yaml` /
  `<tag>.yaml` keyed by engine argparse dests.

Both **fail loudly** when a key goes stale: a bad Hydra key raises Hydra's
unknown-key / struct error, and an unknown sweep-config dest makes the launcher
**exit 2** (`error: unknown calibrate sweep config key '…'`). They are **not**
covered by the stability promise — **expect to update them when the submodule
pin bumps.** Use `job.dry_run=true` (and the submodule's
`--print-resolved-config`) to check them before a real run.

> **Parallel sweeps.** The sweep YAML is keyed solely by its path. calibrate
> writes `<out>/ntt_calibrate.yaml`, so concurrent calibrate sweeps need
> **distinct `job.out` dirs**. benchmark writes
> `results/downstream/sweep_configs/<tag>.yaml` where `<tag>` derives from the
> `ckpt_root` **basename**, so concurrent benchmarks need **distinct
> `ckpt_root` basenames** (they also collide on the output dir otherwise).

## Notes

- `ntt-downstream` is used **unmodified** — the launcher only builds a command
  line + environment its versioned console scripts already honour.
- **Contract drift alarm.** `tests/jobs/test_downstream_contract.py` runs the
  **real** `ntt-downstream-*` entrypoints GPU-free (via `--dry-run` /
  `--print-resolved-config`). It is skipped when `downstream/.venv` is missing
  (e.g. CI), but locally it fails the moment a promoted flag or Hydra key drifts
  against the pinned submodule — catching it before a user's first real run.

## Why decoding reruns are not bit-for-bit identical

The decoding trainer uses `torch.use_deterministic_algorithms(False)`, `cudnn.benchmark = True`, and `Trainer(deterministic=False)`. The model also uses CUDA operations that can give slightly different results, including SDPA / flash-attn backward atomics and bf16 reductions. There is **no launcher or config flag** for bit-for-bit identical results. To get them, `ntt-downstream` would need to enable deterministic algorithms and cuDNN determinism, and stop using flash-attn. That would reduce speed and precision.
