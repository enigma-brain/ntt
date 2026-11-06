# Troubleshooting notes

Things that are easy to get wrong when running the evaluation battery, with the symptom, the
cause and what to do about each.

- **Part A — checkpoint and data-config issues.** Behaviours of the ntt code itself that
  need a specific setting or a staged workaround.
- **Part B — environment and third-party traps.** Tracker clients, containers and
  authentication.

The supported workflow is `docs/eval_battery_runbook.md`; these notes cover what that
workflow does not.

---

# Part A — checkpoint and data-config issues

## A1. Checkpoints carrying a training-only loss break downstream encoding

**Symptom.** Encoding dies with `FileNotFoundError` on a path that exists only
on the machine that trained the cohort — a frozen teacher checkpoint, an embedding table.
Intrinsic evals (`job=eval_all`) are unaffected, which makes this look machine-specific
rather than systematic.

**Why.** Eval never uses the training loss, and `load_tokenizer_from_checkpoint` skips it.
But the submodule's encoding wrapper (`encoding_wrapper.py` `_load_from_checkpoint`)
rebuilds the whole `NTTLitModule` with Lightning's `_load_state` and calls
`configure_model()`, which builds the loss from `loss_spec`, so it faithfully reconstructs a
loss object whose spec points at assets that are not there. Decoding is not affected: it
loads only the tokenizer (`frozen_ntt_tokenizer.py`).

**What to do.** Strip the loss from **staged copies** of the checkpoints. This needs
no source change and does not touch tokenizer weights; do it on copies, never on the
originals:

```python
ck = torch.load(p, map_location='cpu', weights_only=False)
ck['hyper_parameters']['loss_spec'] = MSELossSpec()          # dependency-free
for k in [k for k in ck['state_dict'] if k.startswith('loss_fn.')]:
    del ck['state_dict'][k]
torch.save(ck, p)
```

Deleting the `loss_fn.` weights is safe because the loader tolerates that prefix as known
drift — confirm with `grep -rn _KNOWN_DRIFT_PREFIXES src/`. For remote execution, upload
the **sanitised** copies and point the launcher at those URIs; sanitising on each node is a
cost paid repeatedly for nothing.


## A2. A distribution-valued `n_neurons` crashes collation

**Symptom.**

```
RuntimeError: Sizes of tensors must match except in dimension 1.
              Expected size 10 but got size 225
```

**Why.** `pad_neurons_to_fixed` is only inferred when the spec's `n_neurons` is an `int`
(`grep -n "Infer pad_neurons_to_fixed" -A 4 src/ntt/core/data/datamodule.py`). For a
distribution it stays `None`, so each batch pads to its own max and segment collation
fails.

**What to do.** Set `pad_neurons_to_fixed: <ceiling of the distribution>` in the
data config. It pads up and never truncates, so the value must be ≥ any count the sampler
can draw. Each `eval_all` pass composes a **different** data config, so all four need it.

Setting `n_neurons:` in the data config instead does nothing — see A3 for why.


## A3. The datamodule overwrites tokenizer fields; only the eval window is put back

**What happens.** The eval datamodule is built **with the spec**. This calls
`update_from_tokenizer_spec` with `unset_only=False`, which overwrites every tokenizer
field in the data config. So `n_neurons:` in a data config has no effect (A2).

Right after building the datamodule, the eval job restores the requested window length
and checks it (`src/ntt/jobs/ntt_base.py`, `_check_scored_window`). If the job would load a
different window length, it fails at the start. The run saves the window length it actually
evaluated as `scored_window_seconds`.

**What to do.** Nothing for the window. If `scored_window_seconds` differs from the requested
`eval_window_seconds`, report it as a bug; this is not expected.


## A4. Two shapes of checkpoint-load failure, two different causes

Both are fixed by evaluating with the code that wrote the checkpoint, but they look nothing
alike and get misdiagnosed as each other:

| symptom | cause |
|---|---|
| `pydantic … union_tag_invalid`, or an unknown-field error | the spec schema in your checkout lacks a field the checkpoint carries |
| `ModuleNotFoundError` raised by `torch.load` **itself**, before any pydantic error | the run pickled a config *object* — an auxiliary-objective config, say — whose defining module does not exist in your checkout |

The second one is the confusing one: it fails before any ntt validation code runs, so the
traceback points at torch and implicates nothing about specs.


---

# Part B — environment and third-party traps

## B3. Tracker HTTP retry ceilings

Long fan-outs fail on chunked-encoding errors, SSL EOF, and DNS far more often than on
anything in the science.

> **With MLflow:** `MLFLOW_HTTP_REQUEST_MAX_RETRIES` is **capped at 10** by default
> (`_MLFLOW_HTTP_REQUEST_MAX_RETRIES_LIMIT`) — a higher value raises `MlflowException` on the
> very first request, so "turn it up" makes things
> worse. Pair it with `MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR`.
>
> The Python client's `search_runs()` can also fail outright once a run has logged models.
> Use the REST API if you hit that.

## B3b. A client newer than the tracking server breaks artifact listing

**Symptom.** Listing a run's artifacts works at the top level and raises for any
subdirectory:

```
MlflowException: API request to endpoint /api/2.0/mlflow/logged-models/search
failed with error code 404 != 200
```

**Why.** In MLflow 3.x, `MlflowClient.list_artifacts` resolves through `runs:/` and adds
`_list_model_artifacts(path)` to the result. That helper short-circuits **only when `path`
is empty**, so the root listing succeeds and every nested one queries a `logged-models`
endpoint that older tracking servers do not implement. This is not intermittent: it breaks
every nested listing against such a server.

**What to do.** Go to the run's own artifact repository, which lists the same paths
without the model lookup:

```python
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository

info = client.get_run(run_id).info
ls = get_artifact_repository(info.artifact_uri, tracking_uri=client.tracking_uri).list_artifacts
```

`scripts/eval/build_cohort.py` already does this. Use `info.artifact_uri` to build artifact
URIs too, rather than assembling `mlflow-artifacts:/<exp>/<run>/artifacts/...` by hand —
the hand-built form is correct only for proxied artifact storage and silently fabricates
unusable URIs for an experiment backed by `s3://` or `file://`.

This is the same root cause as B3's `search_runs()` note: a client ahead of its server.

## B4. A container with its own `$HOME` does not see your Hugging Face token

`hf auth login` stores the token under `~/.cache/huggingface/`. A container started with a
separate home directory cannot read it, so `hf://` downloads of a private repo fail. Export
`HF_TOKEN` into the container's environment instead; never write the token into a job's
committed files.
