# Downstream encoding — advanced usage

Part of [downstream_eval.md](downstream_eval.md).


## Worked example — an encoding run with a frozen checkpoint and custom settings

A common setup uses a frozen tokenizer checkpoint, a non-default recipe, and `job.extra` to set internal Hydra keys such as `model.*`, `trainer.*`, and `data.*`.

```bash
uv run ntt paths.run_name=enc_eval job=downstream_encoding \
    job.ckpt=mlflow-artifacts:/123/model.ckpt \
    job.recipe=cowley_v4_perceiver_mse \
    job.loss_mode=flow_matching \
    job.seed=7 \
    'job.extra=["model.depth=4","trainer.max_epochs=50","data.batch_size=32"]' \
    job.dry_run=true
```

> **`job.extra` list syntax.** Since each value contains `=`, put double quotes around each value inside the Hydra list, then single quotes around the whole assignment for the shell:
> `'job.extra=["model.depth=4","trainer.max_epochs=50"]'`. The unquoted form
> `'job.extra=[model.depth=4]'` fails Hydra's override grammar
> (`no viable alternative at input '[model.depth='`). Quote each value.

With `job.dry_run=true`, the launcher prints the command it would run. This example was checked locally against the pinned submodule:

```text
uv run --project <repo>/downstream --no-sync ntt-downstream-encode \
  --tokenizer-ckpt mlflow-artifacts:/123/model.ckpt \
  --output-root <repo>/results/downstream/encoding \
  --seed 7 --config-name cowley_v4_perceiver_mse --dry-run \
  model.loss_mode=flow_matching ++tokenizer.checkpoint_first=true trainer.tokenizer_mode=frozen \
  model.depth=4 trainer.max_epochs=50 data.batch_size=32
```

Read the command from left to right:

| Token(s) | Origin | Stability |
|---|---|---|
| `ntt-downstream-encode` | One of downstream's four stable commands (`ntt-downstream-decode`, `-encode`, `-calibrate`, `-benchmark`) | **Stable contract** (versioned by the submodule pin) |
| `--tokenizer-ckpt`, `--output-root`, `--seed`, `--config-name` | Promoted flags from `job.ckpt`, `job.out`, `job.seed`, and `job.recipe` | **Stable contract** — flag names are versioned; the Hydra keys they map to are not |
| `model.loss_mode=flow_matching`, `++tokenizer.checkpoint_first=true`, `trainer.tokenizer_mode=frozen` | ntt-owned passthrough that makes the frozen tokenizer get used | **ntt-downstream-internal** Hydra keys, not covered by the stability promise |
| `model.depth=4`, `trainer.max_epochs=50`, `data.batch_size=32` | Your `job.extra`, added last; Hydra uses the last value when settings conflict | **ntt-downstream-internal** — a power-user escape hatch that fails loudly but is not versioned |

Use `job.dry_run=true` to preview the assembled command. On the submodule side, use `--print-resolved-config` to preview the fully resolved Hydra config before spending GPU time.

## External recipes: full configs outside the submodule

Use `job.extra` for a few settings. To keep an experiment recipe alongside your ntt checkout, set `job.recipe` to a **path**, not a name. It counts as a path if it contains `/` or ends in `.yaml` or `.yml`. The path points to a Hydra config outside the submodule. It can be absolute or relative to the **ntt repo root**, not the current working directory, because ntt's Hydra run directory makes the current directory unreliable.

ntt checks that the resolved path exists. If it does not, ntt reports the path. Then ntt passes `--config-dir <dir> --config-name <stem>`.

A starter overlay is at `config/downstream/encoding/example_overlay.yaml`:

```yaml
defaults:
  - /cowley_v4_base   # the submodule's baseline recipe (absolute = search-path root)
  - _self_

trainer:
  max_epochs: 50
data:
  batch_size: 32
```

Launch it with a recipe path relative to the ntt repo root:

```bash
uv run ntt paths.run_name=enc_eval job=downstream_encoding \
    job.ckpt=mlflow-artifacts:/123/model.ckpt \
    job.recipe=config/downstream/encoding/example_overlay.yaml
```

Two details make this work:

- **Hydra adds the folder to its search path.** ntt passes the recipe's parent folder as `--config-dir`. For a single run, Hydra adds that folder to the search path and keeps the submodule's `configs/encoding/` there too. So `defaults: [/cowley_v4_base, _self_]` finds `/cowley_v4_base` in the submodule, then applies the external file on top. The curated `configs/cli/encode.yaml` works the same way. Since `--config-dir` removes `configs/cli/` from the search path, ntt also always passes `--config-name` for path-form recipes.
- **All settings in the file are internal and unversioned.** Keys the overlay sets, such as `trainer.*` and `data.*`, are on the ntt-downstream-internal side of the contract boundary (see [What crosses the boundary](downstream_maintainer_internals.md#what-crosses-the-boundary)). A bad key causes an error, but these keys may need updating when the submodule pin changes.

> **`job.loss_mode` takes precedence.** After the recipe is composed, ntt adds `model.loss_mode=<job.loss_mode or mse>`. Set the loss with `job.loss_mode`, not in the file; ntt overrides any `model.loss_mode` set by the overlay.

## Keep the recipe portable: leave the data path as `???`

Do not put a machine-specific data path in the recipe. Leave `data.dataset_path` as `???`; Hydra requires the launch command to supply it. Set `job.data_root` when you launch. ntt maps it to `data.dataset_path` and adds it last, replacing `???`.

Use the **full brainset path**: the directory that contains the `.h5` recordings, not its parent. If the path is empty or wrong, ntt silently loads 0 sessions and then crashes with `estimated_stepping_batches=0`. Leave out the tokenizer block; the launcher adds the checkpoint from `job.ckpt`.

```yaml
# In the path-form overlay, set the recipe file here:
# config/downstream/encoding/<your_recipe>.yaml
defaults:
  - data: cowley_v4
  - backbone: hiera_small_112
  - model: image_encoding_diffusion   # flow/diffusion family; use image_encoding for cross_entropy
  - trainer: default
  - _self_

data:
  dataset_path: ???                    # required override — pass it via job.data_root at launch
  bin_size: 0.001
  response_window: 0.200
  subjects: "Pepe"
dataloader:
  batch_size: 64
model:
  loss_mode: "flow_matching"           # job.loss_mode still wins; kept for standalone use
  token_t_latent: 2                    # = response_window_ms / tokenizer_window_ms
  flow_time_sampling: "logit_normal"
trainer:
  lr: 5e-4
  n_epochs: 60
  # example inference-time flow sampling — euler / 5 steps / shift 3.0:
  tokenizer_eval: { flow_solver: euler, n_flow_steps: 5, flow_shift: 3.0 }
  final_eval:     { flow_solver: euler, n_flow_steps: 5 }   # final_eval otherwise defaults to heun
```

Launch with your cluster's brainset path:

```bash
uv run ntt paths.run_name=enc_eval job=downstream_encoding \
    job.ckpt=mlflow-artifacts:/<exp>/<run>/artifacts/checkpoints/model.ckpt \
    job.recipe=config/downstream/encoding/<your_recipe>.yaml \
    job.loss_mode=flow_matching \
    job.data_root=/abs/processed/cowley_v4_compact_2024
```

To check the recipe without a GPU, add `job.dry_run=true` to print the assembled child command. Then run that command with `--print-resolved-config`. Do not use `--dry-run`: it only echoes the command. The resolved config check confirms that `data.dataset_path` has a value and there is no `MissingMandatoryValue`.

## Version the whole invocation with an `experiment` config

The recipe stores the evaluation architecture. To also store the checkpoint, loss, data path, and run name, use ntt’s built-in `experiment` config group. (`config/main.yaml` includes `- experiment: null`.) Make a `# @package _global_` overlay at `config/experiment/<name>.yaml`. In it, select `downstream_encoding` and set the `job.*` fields:

```yaml
# @package _global_
# config/experiment/<name>.yaml
defaults:
  - override /job: downstream_encoding

job:
  ckpt: mlflow-artifacts:/<exp>/<run>/artifacts/checkpoints/model.ckpt
  recipe: config/downstream/encoding/<your_recipe>.yaml
  loss_mode: flow_matching
  data_root: /abs/processed/<brainset>      # fills the recipe's dataset_path=???

paths:
  run_name: <run_name>                       # required (paths.run_name defaults to ???)
```

Run the experiment by name, without entering the overrides again:

```bash
uv run ntt experiment=<name>
```

`experiment=<name>` adds the overlay to `config/main.yaml`. The usual `paths`, `hydra`, and logging defaults stay in place. The overlay only selects `downstream_encoding` and sets its fields. Tokenizer-training configs in `config/experiment/` use the same mechanism.
