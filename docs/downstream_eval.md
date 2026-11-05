# Downstream evaluation (`ntt job=downstream_decoding` / `job=downstream_encoding`)

Use the [`ntt-downstream`](https://github.com/enigma-brain/ntt-downstream) evaluations on a tokenizer checkpoint through the `ntt` command. Each evaluation trains a small model while keeping the tokenizer fixed. Choose one of four settings: test decoding on one session (`single-run`); test decoding on new sessions using a small number of training trials (`calibrate`); run single-run's decoding on each benchmark session, for each tokenizer in a folder (`benchmark`); or predict neural activity from stimuli using the fixed tokenizer (`encoding`).

Decoding uses POYO, a transformer that predicts behaviour, such as hand movement, from neural activity.

These evaluations **complement** `ntt job=eval`:

| Job | Measures | Cost |
|---|---|---|
| `job=eval` | Reconstruction, structure and compression measures (see [eval_usage.md](eval_usage.md)) | GPU or CPU; no training |
| `job=eval_all` | `job=eval` plus repeat SNR and input- and latent-corruption robustness | GPU or CPU; no training |
| `job=downstream_decoding` / `job=downstream_encoding` | Downstream task performance: behaviour **decoding** R² (POYO) and stimulus **encoding** correlation (Cowley V4 / Wingert datasets) | Hours, GPU/time-heavy; trains a small model on top of the fixed tokenizer |

For maintainer details—how the launcher is wired, version pins, environment setup and checks, and the `ntt` ⇄ `ntt-downstream` contract—see [downstream_maintainer_internals.md](downstream_maintainer_internals.md).

## Quick start

```bash
# 1. Run this once inside your ntt checkout to get the downstream code in `./downstream`
git submodule update --init downstream
#    If you do not have ntt yet, clone it with the downstream code included
git clone --recurse-submodules https://github.com/enigma-brain/ntt

# 2. Only needed for encoding runs if `MLFLOW_TRACKING_URI` or your MLflow login is not already set in your shell (for example, in ~/.bashrc): copy the example settings file and fill it in.
cp downstream/.env.example downstream/.env

# 3. One time: set up downstream's Python environment from this ntt checkout. This takes a few minutes.
#    Do not install anything inside downstream/ yourself.
uv run python -m ntt.jobs.downstream_env
```

Then run one of the four settings. Replace each `<...>` with your value. The shared values are explained below the commands.

```bash
# a) Decode from one tokenizer's latents: train and evaluate one model on one session
uv run ntt paths.run_name=<run_name> job=downstream_decoding job.mode=single-run \
  job.ckpt=<ckpt> \
  job.session=<session name, e.g. c_20150319_center_out_reaching> \
  job.data_root=<data_root> \
  job.out=<out>

# b) Decode in two stages: train a base model, then fine-tune it for each evaluation session using a small number of that session's trials
uv run ntt paths.run_name=<run_name> job=downstream_decoding job.mode=calibrate \
  job.ckpt=<ckpt> \
  job.data_root=<data_root> \
  job.out=<out>

# c) Run the same decoding evaluation separately for each tokenizer in a directory; provide the directory with ckpt_root
uv run ntt paths.run_name=<run_name> job=downstream_decoding job.mode=benchmark \
  job.ckpt_root=<folder holding tok_a/checkpoints/model.ckpt, tok_b/checkpoints/model.ckpt, …> \
  job.data_root=<data_root>

# c') Benchmark ONE tokenizer without building the folder: give job.ckpt (hf:// or local .ckpt) instead of ckpt_root
uv run ntt paths.run_name=<run_name> job=downstream_decoding job.mode=benchmark \
  job.ckpt=hf://the-enigma-project/ntt-tokenizers/conv-vae-100to1@<revision> \
  job.data_root=<data_root>

# d) Encoding with the Cowley V4 dataset (monkey visual-cortex responses to images): predict the frozen tokenizer's latent vectors from images, using latent-MSE loss by default
#    Encoding details: docs/downstream_encoding_advanced.md
uv run ntt paths.run_name=<run_name> job=downstream_encoding \
  job.recipe=<recipe name; default cowley_v4_base> \
  job.ckpt=<ckpt> \
  job.out=<out>

# Print the resolved command and environment without running the evaluation
uv run ntt paths.run_name=<run_name> job=downstream_decoding job.mode=single-run \
  job.ckpt=<ckpt> \
  job.session=<session name> \
  job.data_root=<data_root> \
  job.dry_run=true
```

- `<run_name>`: the name of the run’s log folder, under `logs/downstream_decoding/` or `logs/downstream_encoding/`.
- `<ckpt>`: the path to the tokenizer’s `.ckpt` file, or an `mlflow-artifacts:/` URI.
- `<data_root>`: the brainsets `processed/` folder. It contains folders such as `perich_miller_population_2018/`.
- `<out>`: optional folder for results. If omitted, results go to `results/downstream/` (single-run), `results/downstream/calibration/` (calibrate), or `results/downstream/encoding/` (encoding). Benchmark results always go to `downstream/experiments_benchmark/<tag>/`.

Poisson and compressed sensing do not have training checkpoints. For them, use the `checkpoints/model.ckpt` saved by the main eval (`job=eval` or `job=eval_all`) when MLflow logging is on ([eval_usage.md § MLflow checkpoint upload](eval_usage.md#mlflow-checkpoint-upload)). For `benchmark`, download it first into the `ckpt_root` layout: `mlflow artifacts download -u <URI> -d <ckpt_root>/<name>/checkpoints/`.

### Which command gives which score

The aggregation ([eval_aggregation.md](eval_aggregation.md)) uses these runs to make scores. Its input is a table with one row per tokenizer. The “tracker column” is where you put the MLflow link for the run.

| Score | Run | Tracker column |
|---|---|---|
| downstream decoding | POYO R²: `job=downstream_decoding job.mode=benchmark` with the default session list (the score is empty if any session did not finish); ridge R²: the main eval; the score is the mean of the two | `mlflow_decoding` (the benchmark's parent run) and `mlflow_recon` |
| `downstream_encoding` | `job=downstream_encoding` (default recipe `cowley_v4_base`): noise-ceiling R² on the test split | `mlflow_encoding` |
| `generalization` | not a downstream run: the main eval (`job=eval`, or the main pass of `job=eval_all`); how well the main eval's reconstruction metrics hold up on datasets unseen in training compared with the training datasets | `mlflow_recon` |

`calibrate` (the paper's Figure 4b/d) and `single-run` do not feed any of these scores.

### Setup details

- No MLflow server is needed. Encoding logs to MLflow unless you pass `'job.extra=["trainer.logging=false"]'`; with logging on and no `MLFLOW_TRACKING_URI` (shell or `downstream/.env`), it stops before training with a message saying so. Decoding sweeps use MLflow only when `MLFLOW_TRACKING_URI` is set; `job.mlflow=false` keeps them off even then. Either way the run writes its results record, and fails if it cannot.
- `job.downstream_env=auto` is the default. It builds the downstream environment if needed. `verify` only checks that environment. If many parallel jobs (for example on a cluster scheduler) use the same checkout, build the environment once with `uv run python -m ntt.jobs.downstream_env`, then use `verify`.

## Which `ntt` code the evaluation uses: your checkout or the pinned release

The quick-start `ntt` command runs downstream with the `ntt` code from your current checkout, including local changes. Here, `ntt` means this repository and its Python package.

To use downstream's pinned `ntt` release, for example to reproduce a result with fixed versions, use a separate clone of `ntt-downstream`. Run `uv sync` there, then run its `ntt-downstream-*` commands as described in `docs/getting-started.md`. Do not run `uv sync` in this checkout's `downstream/` folder: the `ntt` command manages that folder's environment, and `uv sync` would replace it.

## The four settings in detail

- `single-run`: Trains and tests one decoder on one session.
- `calibrate`: Measures how well decoding transfers to a new session when only a few trials from it are available. This is the paper’s measure of sample-efficient decoding (Figure 4b and 4d).
- `benchmark`: for each tokenizer in a folder, trains and tests a separate POYO decoder on each benchmark session, one session at a time. This works like `single-run`. The per-session results are combined into one record per tokenizer.
- `encoding`: Measures how well a model uses your tokenizer’s latent vectors to predict neural activity from stimuli.

### Entrypoints

There is one launcher class and two Hydra job configs. The decoding config has three `job.mode` choices, so the configs provide four ways to run an evaluation. Both use the same `NTTDownstreamJob` class. The config you choose sets the `task` and shows the settings for that task.

You do not run these console scripts yourself. The `ntt` command starts each one in downstream's environment.

| Config | `task` | `mode` | Console script (run in `downstream`'s venv) |
|---|---|---|---|
| `job=downstream_decoding` | `decoding` | `single-run` | `ntt-downstream-decode` |
| `job=downstream_decoding` | `decoding` | `calibrate` | `ntt-downstream-calibrate` |
| `job=downstream_decoding` | `decoding` | `benchmark` | `ntt-downstream-benchmark` |
| `job=downstream_encoding` | `encoding` | — | `ntt-downstream-encode` |

For decoding, POYO uses the frozen tokenizer's latent vectors to predict behaviour; the evaluation reports R². For encoding, the frozen tokenizer is used to predict neural activity from stimuli (Cowley V4 / Wingert).

#### 1. Decoding — `mode=single-run`

Trains and tests one POYO decoder on one session. This is one training run, not a sweep of many runs.

- **Requires:** `job.ckpt`, either a frozen-tokenizer `.ckpt` file or an `mlflow-artifacts:/` URI, and `job.session`.
- **Architecture:** Uses the same settings as the sweeps so R² scores can be compared: `model=poyo_1.3M_no_crossattn`, `unit_embedding_mode=add`, `sentinels=false`. These default settings are in the submodule contract config, `configs/cli/decode.yaml`. The ntt launcher passes them along only if you set the matching option below.
- **Output:** `job.out` (default `results/downstream/`) contains `csv_logs/version_*/downstream_decoding_results.json` (schema-v2 record; see [downstream `docs/results.md`](https://github.com/enigma-brain/ntt-downstream/blob/main/docs/results.md)), `run_result.json` (local sweep bookkeeping, not uploaded), and `metrics.csv`.

> `single-run` skips the GPU-pool sweep, but still uses the **full default epoch budget**. For a smoke test, set `job.epochs=1` or `'job.extra=["max_steps=200"]'`. Double-quote each element; see [the quoting note](downstream_encoding_advanced.md#worked-example--an-encoding-run-with-a-frozen-checkpoint-and-custom-settings).

#### 2. Decoding — `mode=calibrate`

For each base-train session, this trains a separate base decoder. It then fine-tunes each decoder on a small number of trials from every eval session. This measures how well decoding works with few fine-tuning trials, as shown in Figure 4b and 4d. The command is `ntt-downstream-calibrate`. It writes CSV files showing decoding R² (a measure of how well the decoder predicts) for different numbers of fine-tuning trials.

POYO first trains for 3,400 steps using latents from the frozen tokenizer and the base-train sessions. For each eval session, it then fine-tunes using 4, 8, 16, 32, 64, or 86 trials from that session. It skips counts above 70% of that session's valid trials. Fine-tuning runs for 800 steps, with seeds 42 and 43. It can update only the unit and session embeddings (`unit_identification`), or all decoder weights (`full`). The CSV files show decoding R² for each number of fine-tuning trials. Defaults for downstream training are in `configs/decoding/single_session/training.yaml` and `tokenizer_training.yaml`.

By default, two sessions are used to train base decoders. Each session has a separate decoder for each seed. ntt cannot yet train one base decoder jointly across sessions.

- **Requires:** `job.ckpt`. MLflow URIs are supported (see [URI support](#the-frozen-tokenizer)). If you have only a URI, use `mode=single-run`.
- **Sessions:** `job.train_selection` sets the `train_sessions` used to train base decoders. `job.eval_selection` sets the `other_sessions` used for evaluation, one session at a time. Both selections go into a sweep config loaded through `--config-dir`.
- **Steps and overrides:** `job.epochs` is ignored. The launcher warns because the sweep uses step counts instead. `job.model_cfg` is passed into the sweep config. Use `job.extra` for overrides to `train.py` for each evaluation cell, such as `ft_max_steps`.
- **Output:** `job.out` (default `results/downstream/calibration/`) holds the fine-tuning and calibration CSV files. Base-training checkpoints are saved under `downstream/outputs/`.

#### 3. Decoding — `mode=benchmark`

- **Requires:** `job.ckpt_root` or `job.ckpt` (not both). Downstream must find each checkpoint at `<root>/<name>/checkpoints/model.ckpt`; a symlink to this layout is fine.
  With `job.ckpt` (an `hf://` URI or a local `.ckpt` path) ntt builds a one-tokenizer root itself at `results/downstream/benchmark_roots/<name>-<hash>/`, with `<name>/checkpoints/model.ckpt` a symlink to the materialized checkpoint. `<name>` is the Hub model name (for a local file, its stem), so it is the tokenizer's label in the results. `<hash>` is 8 hex characters computed from repo, name and revision (or from the local path). `mlflow-artifacts:/` URIs are not supported here; put them under a `ckpt_root`. With `dry_run`, nothing is downloaded and no folder is created.
  Each subfolder name `<name>` becomes that tokenizer’s label in the results. No separate config file is needed. Each `model.ckpt` must load on its own, just as `job.ckpt` does in the other settings.
- **Sessions:** `job.eval_selection` becomes `sessions`. By default, these are the 12 sessions in `decoding/ntt_decoding_benchmark_eval`: 5 were in the tokenizer's training data, 1 is a new session from a monkey seen in training, and 6 are from a monkey never seen in training. Results are reported per session; the benchmark does not group them by these categories.
- **Tag and path:** The comparison tag is `ntt_downstream_<ckpt_root-basename>`. It sets the output path. If two roots have the same basename, they collide in both the sweep-config path and the engine output directory. Use distinct directory names. With `job.ckpt` the basename is `<name>-<hash>`, so different models and revisions get different tags.
- **Output:** Always `<downstream submodule>/experiments_benchmark/<tag>/` (the printed `outputs ->` line shows it). The engine sets this location, so `job.out` is ignored and a warning is logged; no other field redirects it (to collect results elsewhere, symlink `experiments_benchmark` to the target). Each tokenizer’s cells are combined into one cohort record at `<tag>__<tokenizer>__cohort/downstream_decoding_results.json`. This record is also uploaded to the tokenizer’s MLflow parent run at its artifact root. The aggregation tracker’s `mlflow_decoding` column points to that parent run (see [eval_aggregation.md § Downstream inputs](eval_aggregation.md#downstream-inputs-encoding-and-decoding)).

#### 4. Encoding — `job=downstream_encoding`

Evaluates stimulus encoding (Cowley V4 / Wingert). It trains a model to predict neural activity using the frozen tokenizer.

- **Requires:** `job.ckpt`, given as a local path or an `mlflow-artifacts:/` URI. Encoding logs to MLflow by default: set `MLFLOW_TRACKING_URI` in your environment or `downstream/.env`, or run without MLflow with `'job.extra=["trainer.logging=false"]'`.
- **Data and seed:** `job.data_root` becomes `--data-root`; the launcher maps it to the encoding data path. `job.seed` becomes `--seed`. The encode contract also sets a top-level `lightning.seed_everything`.
- **Loss:** `job.loss_mode` defaults to `mse`, which predicts tokenizer latents and works with any `compress_t` tokenizer. Other options are `flow_matching`, `diffusion`, `energy`, `mar`, and `cross_entropy`. ntt does not check whether a loss uses the tokenizer. The submodule decides whether the loss is valid and whether it uses `job.ckpt`. An unsuitable loss may be accepted or rejected by the child run.
- **`token_prediction_dim` — CE vs. flow (the head-width distinction):** If unset, `EncodingLitModule.setup()` derives `model.token_prediction_dim` from the tokenizer and loss:
  - **`cross_entropy`:** Predicts token indices through `ce_head=Linear(token_prediction_dim, V)`. Its width is not constrained by the latent shape, so the default is **`max(z_dim, 32)`**. It is separate from `z_dim` and never below 32. Setting it to a bare `z_dim`—for example, `z_dim=1` for CALM—would make `ce_head=Linear(1, V)`, forcing image features through one value and crushing the signal at the softmax. Set a different width with `job.extra: "++model.token_prediction_dim=128"`. At setup, a resolved CE width below 16 triggers a loud `WARNING` naming the `Linear(D, V)` bottleneck.
  - **`flow_matching`, `diffusion`, `mse`, and `energy`:** Regress the continuous latent, so the head width must match `z_dim`; that is the default. **`bce`** predicts `z_dim` bits directly, so its default is also `z_dim`. For any loss, an explicitly set `token_prediction_dim` takes precedence.
- **Recipe:** `job.recipe` defaults to `cowley_v4_base` and selects the data, backbone, and architecture. It is a bare config name resolved against the submodule's `configs/encoding/`. ntt does not check that the recipe exists; if it does not, the child reports a Hydra error. Compress-N (Perceiver) tokenizers need a Perceiver recipe such as `cowley_v4_perceiver_mse` and may need extra inputs, passed through `job.extra`. The default recipe uses `configs/cli/encode.yaml` and adds no `--config-name` flag. A non-default recipe adds `--config-name <recipe>`.
- **Output:** `job.out` defaults to `results/downstream/encoding/` and is used as `hydra.run.dir`. It contains checkpoints, `downstream_encoding_results.json`, and local `final_eval*.csv` reports. The results file is also uploaded to the MLflow run. Its `test` block reports the test split using the checkpoint selected on validation; see [downstream `docs/results.md`](https://github.com/enigma-brain/ntt-downstream/blob/main/docs/results.md).

## Comparing results

### Determinism / reproducibility (decoding)

These rules apply to all three decoding settings, which use the same training code:

- **Same seed means similar, not identical, results.** Reruns with the same setting, seed, config and hardware can differ slightly because some GPU computations are not exactly repeatable. No option makes them identical ([why](downstream_maintainer_internals.md#why-decoding-reruns-are-not-bit-for-bit-identical)). The default seed, `42`, sets the decoder’s starting weights, training-window order, data splits, fine-tune trials, dropout and data loading. You do not need `job.seed` for repeatability. `calibrate` runs every combination with seeds 42 and 43. Setting `job.seed=N` runs each combination once with seed N.
- **Compare results only within the same setting.** single-run uses fp32 (32-bit floating point) and PyTorch SDPA attention. benchmark uses bf16 (16-bit floating point) and flash-attn, so it needs an Ampere or newer GPU (on older GPUs every session fails with "FlashAttention only supports Ampere GPUs or newer") and glibc 2.32 or newer (on older systems such as RHEL 8, flash-attn fails to import with ``GLIBC_2.32' not found``). calibrate uses the settings in its training config. All use the same decoder architecture (`poyo_1.3M_no_crossattn` / `add` / no sentinels), but single-run and benchmark R² scores are not directly comparable.
- **Safe to change:** `job.num_workers` and `job.gpus`; they change speed only, and each run still uses one GPU. Do not force several GPUs per run with `'job.extra=["gpus=N"]'`. That changes the seeds used per GPU, number of steps and learning rate, so results no longer match one-GPU runs.
- With `temporal_split.mode=random`, eval trials are always selected using seed 42, so every run seed uses the same eval set. To change it, set `temporal_split.seed` via `job.extra`. The default `temporal` mode does not use this setting.

## Reference

### Where outputs land

At the end of each run, the launcher prints the full path to its output.

- **single-run** → `job.out` (default `results/downstream/` in this checkout). It contains `csv_logs/version_*/downstream_decoding_results.json`, `run_result.json`, and `metrics.csv`.
- **calibrate** → `job.out` (default `results/downstream/calibration/` in this checkout; the engine uses it as `CALIB_ROOT`). It contains fine-tune and calibration CSV files. Base-train checkpoints are saved under `downstream/outputs/`.
- **benchmark** → `<downstream submodule>/experiments_benchmark/<tag>/`, where `<tag>` is `ntt_downstream_<ckpt_root-basename>` (with `job.ckpt`, `ntt_downstream_<name>-<hash>`). The engine chooses this location, ignores `job.out`, and prints a warning. Each tokenizer has a `<tag>__<tokenizer>__cohort/downstream_decoding_results.json` file.
- **encoding** → `job.out` (default `results/downstream/encoding/` in this checkout; the engine uses it as `hydra.run.dir`). It contains checkpoints, `downstream_encoding_results.json`, and an MLflow run.

The `downstream_*_results.json` files are the only guaranteed results files. Their format is described in [downstream `docs/results.md`](https://github.com/enigma-brain/ntt-downstream/blob/main/docs/results.md).

### Knobs — full configuration reference

Set each field on the `ntt` command line as `job.<field>=<value>`. The values below are the launcher’s defaults. Usually, `null` (“None”) means the downstream script uses its own default. Find the defaults in `config/job/downstream_decoding.yaml` and `config/job/downstream_encoding.yaml`.

#### Selecting the evaluation

| Knob | Values | Default | Applies to | What it does |
|---|---|---|---|---|
| `task` | `decoding` \| `encoding` | set by config | — | Chooses the evaluation type. The config sets this, so you usually leave it alone. |
| `mode` | `single-run` \| `calibrate` \| `benchmark` | `single-run` | decoding | Chooses how to run a decoding evaluation. `job=downstream_encoding` rejects this setting with an error. |

#### The frozen tokenizer

| Knob | Values | Default | Applies to | What it does |
|---|---|---|---|---|
| `ckpt` | local `.ckpt` path, `hf://` URI **or** `mlflow-artifacts:/` URI | `null` | single-run, calibrate, encoding, benchmark (one tokenizer) | Frozen tokenizer checkpoint. Required in the first three modes. Benchmark takes it as an alternative to `ckpt_root` for a single tokenizer (`hf://` or local path only). You cannot use it with `ckpt_root`. |
| `ckpt_root` | local directory | `null` | benchmark (**required**) | Directory containing tokenizers at `<root>/<name>/checkpoints/model.ckpt`. A symlink into this layout also works. You cannot use it with `ckpt`. A single checkpoint can be given as `job.ckpt` instead. |

**URI support.** `mlflow-artifacts:/` and other `://` or `mlflow…` references pass through unchanged and are resolved by the submodule’s tokenizer loader. They work for **single-run**, **calibrate**, and **encoding**. **Benchmark** needs a `ckpt_root` directory; `job.ckpt` works there only for `hf://` URIs and local paths.

#### Data and cohort selection

| Knob | Values | Default | Applies to | What it does |
|---|---|---|---|---|
| `data_root` | path | `null` → `<submodule>/data/processed` | all | Root folder for processed `.h5` recordings. The path is made absolute. |
| `session` | recording id | `null` | decoding/single-run | Recording to decode. Required for single-run. |
| `train_selection` | selection name | `null` → `decoding/ntt_decoding_calib_base_train` | decoding/calibrate | An ntt selection in `config/data/selections/`. Its sessions form calibrate’s base-train group (the `train_sessions` key in the sweep config). |
| `eval_selection` | selection name | `null` → calibrate: `decoding/ntt_decoding_calib_eval`; benchmark: `decoding/ntt_decoding_benchmark_eval` | decoding/calibrate, decoding/benchmark | An ntt selection whose sessions form the evaluation group: `other_sessions` for calibrate or `sessions` for benchmark (keys in the sweep config). |

Selections are written as plain session lists in the generated sweep config. This means the groups set in this repository take priority over the submodule’s YAML defaults. Each selection must include sessions from only one brainset. See [Split / cohort source of truth](#split--cohort-source-of-truth).

#### Decoding training knobs

If you do not set a knob, it uses the downstream value. Check the “Applies to” columns: calibrate ignores some knobs and warns you.

| Knob | Values | Default | single-run | calibrate | benchmark | What it does |
|---|---|---|:--:|:--:|:--:|---|
| `epochs` | int | `null` | ✓ | — (warns) | ✓ | Maximum training epochs. calibrate uses steps instead, so it ignores `epochs`. Set `job.extra` instead, for example `'job.extra=["ft_max_steps=10"]'`. |
| `batch_size` | int | `null` | ✓ | ✓ | ✓ | Number of examples in each training step. |
| `seed` | int | `null` | ✓ | ✓ (maps to a one-element seeds list) | ✓ | Random-number seed, sent as the `--seed` flag in all three modes. In calibrate, it sets a sweep with one seed. |
| `num_workers` | int | `null` | ✓ | ✓ | ✓ | Number of data-loader worker processes. |
| `model_cfg` | `poyo_1.3M` \| `poyo_1.3M_no_crossattn` \| `poyo_11.8M` | `null` | ✓ (contract default `poyo_1.3M_no_crossattn`) | ✓ (forwarded into the sweep config) | ✓ | POYO decoder size or variant. |
| `unit_embedding_mode` | `none` \| `add` \| `concat` | `null` | ✓ (contract default `add`) | ✓ | ✓ | How POYO combines the tokenizer’s unit embedding. |
| `sentinels` | `true` \| `false` | `null` | ✓ (contract default `false`) | ✓ | ✓ | Whether POYO gets an extra marker token for each spiking unit, before that unit’s latents. These are called sentinel tokens. |

Notes:
- For single-run, the submodule’s contract sets the architecture defaults, not ntt. See the [Arch bullet](#1-decoding--modesingle-run). ntt passes these knobs only when you set them.
- single-run sends these knobs as trailing Hydra `key=value` tokens. calibrate and benchmark put them in the generated sweep YAML. The keys are engine argparse destinations; see [the boundary section](downstream_maintainer_internals.md#what-crosses-the-boundary).
- For other settings, use `job.extra`.

#### Encoding knobs

| Knob | Values | Default | What it does |
|---|---|---|---|
| `recipe` | A config name under `downstream/configs/encoding/*.yaml`, **or** a path to an external Hydra config | `cowley_v4_base` | Chooses the data, backbone, and model design. **Config name:** The default uses `configs/cli/encode.yaml`. For another name, ntt passes `--config-name` and looks for it using the submodule’s search path. ntt does not check whether the name exists; if it is missing, the child’s Hydra code reports an error. **Path:** A value with `/` in it or ending in `.yaml` or `.yml` is treated as a config file outside the submodule. The path can be absolute or relative to the ntt repo root, not to the current directory. ntt passes `--config-dir <dir> --config-name <stem>` and checks that the file exists. If it does not, ntt reports a clear error. See [External recipes](downstream_encoding_advanced.md#external-recipes-full-configs-outside-the-submodule). Compress-N tokenizers need a Perceiver recipe, such as `cowley_v4_perceiver_mse`. |
| `loss_mode` | `mse` \| `flow_matching` \| `diffusion` \| `energy` \| `mar` \| `cross_entropy` \| … | `null` → `mse` | Chooses the encoding loss and is passed as `model.loss_mode=`. `mse` trains the model to predict the tokenizer’s latent vectors and works with any `compress_t` tokenizer. ntt does not check whether a loss is valid; the submodule decides whether it can use that loss with the tokenizer. |

#### Shared knobs

| Knob | Values | Default | Applies to | What it does |
|---|---|---|---|---|
| `out` | path | `null` → mode-specific default | all except benchmark | Host-side directory for output, changed to an **absolute** path. Defaults: single-run `results/downstream/`, calibrate `results/downstream/calibration/`, encoding `results/downstream/encoding/`. **Ignored for benchmark**; a warning is shown, and benchmark uses a fixed location. |
| `gpus` | space-separated indices, e.g. `"0 1 2"` | `null` → auto-discover | calibrate, benchmark | GPUs used for sweeps; their indices go into the generated sweep config. single-run and encoding ignore this field and use GPUs visible to the process. Set `CUDA_VISIBLE_DEVICES` yourself for those modes. |
| `mlflow` | `auto` \| `true` \| `false` | `null` → `auto` | calibrate, benchmark | MLflow logging of the sweep, written into the sweep config. `auto` logs when `MLFLOW_TRACKING_URI` is set (shell or `downstream/.env`); `true` stops before any session runs if it is not set. single-run does not log to MLflow unless you pass `'job.extra=["mlflow.enabled=true"]'`, and rejects this field. |
| `extra` | list of strings | `[]` | all | Raw Hydra `key=value` overrides, added last, so they can replace values set by promoted fields. single-run and encoding add them unchanged to the command. calibrate and benchmark add them at the end as positional arguments; the engine passes these to `train.py` for each sweep cell. **Syntax:** put each element in double quotes; see [the quoting note](downstream_encoding_advanced.md#worked-example--an-encoding-run-with-a-frozen-checkpoint-and-custom-settings) and [the boundary section](downstream_maintainer_internals.md#what-crosses-the-boundary). |
| `dry_run` | `true` \| `false` | `false` | all | Prints the assembled command, environment variables set by the launcher (no secrets), and output path, then exits without running. It also requests a GPU-free `--dry-run` preview from each command. calibrate and benchmark still write the sweep YAML, so you can run the printed command later. Preflight issues are warnings instead of errors. |

#### Advanced knobs

| Knob | Values | Default | What it does |
|---|---|---|---|
| `root_dir` | path | `null` → find it automatically from `.project-root` | The host repository’s root folder. Used to find `config/data/selections/` and choose default output folders. Usually, you do not need to set it. |
| `submodule_dir` | path (relative to repo root) | `downstream` | Where the vendored `ntt-downstream` submodule is located. |
| `downstream_env` | `auto` \| `verify` | `auto` | Controls the submodule’s virtual environment (see [Step 2](downstream_maintainer_internals.md#step-2--the-submodules-venv-the-host-ntt-overlay)). With `auto`, ntt checks the environment and builds the host-ntt overlay if the check fails. If multiple runs in one checkout need to build it at the same time, they wait for a lock. With `verify`, ntt only checks; it raises an error if the check fails, including when the environment lacks the launcher’s stamp. For multiple runs from one shared checkout, build the environment once with `uv run python -m ntt.jobs.downstream_env`, then use `verify`. This setting is skipped during `dry_run`. |

### Split / cohort source of truth

`calibrate` and `benchmark` use session lists from `config/data/selections/decoding/`:

- `ntt_decoding_calib_base_train.yaml`: sessions used to train the calibration base model.
- `ntt_decoding_calib_eval.yaml`: sessions used to fine-tune and test each base decoder during calibration.
- `ntt_decoding_benchmark_eval.yaml`: 12 benchmark sessions. Inline labels show whether each session was in the tokenizer training data (`ntt_full` is the tokenizer train/validation/test split).

To use other session lists, set `job.train_selection=` and `job.eval_selection=` to names relative to `config/data/selections/`. Both lists must use the brainset `perich_miller_population_2018`. Single-session decoding looks for each session file at `${data_root}/${brainset}/${session}.h5`, so the launcher stops with an error if a list includes more than one brainset.

The launcher also sets `NTT_SPLITS_DIR` to this repository. This lets `${ntt_split:…}` overrides from multi-session or tier configs passed through `job.extra` find split files here instead of in the pinned `ntt` submodule. The shipped single-session cohorts are flattened into the generated sweep config, so they do not use `NTT_SPLITS_DIR`.

### Notes / limitations

- GPU memory per decoding run is about n_batch × T × W × (59,000 + 6 × D) bytes plus ~1 GB, with n_batch the batch size (default 128), T the tokenizer's tokens per second, W the window (1 s) and D the features per token (fitted on 40 GB A100s, ±20 %), e.g. Poisson @ 100 ms on 73 neurons: T = 730, so ≈ 6.5 GB.
