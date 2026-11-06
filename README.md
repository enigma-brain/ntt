# Neurons To Tokens (NTT) <img src="./docs/assets/ntt_logo_v4.png" align="right" width="120" />

> a.k.a. NeuroTicTokenization: a neurotic exploration of neural tokenization.

NTT releases two artifacts that together let any tokenization algorithm for spiking neural data be scored on equal footing:

- **A 10-brainset benchmark corpus** — 1,104 sessions, ~1,566 hours, ~297 GB processed — spanning species (mouse, macaque, ferret), brain areas (visual, motor, somatosensory, cerebellar, auditory), recording technologies (Utah arrays, Neuropixels), and behavioural paradigms. Held-in / held-out tier splits expose cross-instance and cross-task transfer behaviour.
- **A 22-metric evaluation suite** that scores any candidate tokenizer on reconstruction fidelity, latent-space structure, behaviour decoding, stimulus encoding, and compression. Run it on a trained checkpoint, or on an untrained baseline (Compressed Sensing, Poisson) shipped with the repo.

Both are wired up through a Hydra-driven CLI on top of PyTorch Lightning, with [`brainsets`](https://github.com/neuro-galaxy/brainsets) handling data ingestion. Tokenizers follow a Spec + Implementation pattern, so contributing a new one is a small register-and-go workflow.

## 📁 Project structure

```
├── src/ntt/                 # Main package source code
├── config/                  # Hydra configs (data, tokenizer, trainer, paths, …)
├── brainsets_pipelines/     # Per-dataset ingestion pipelines (run via `brainsets prepare`)
├── downstream/              # Git submodule: ntt-downstream (optional, for extrinsic eval)
├── tests/                   # Test suite
├── notebooks/               # Example notebooks (start with v0.ipynb)
├── docs/                    # Project documentation
├── pyproject.toml           # Project configuration
└── README.md
```

### 👩‍💻 Development setup

0. Ensure you have [`uv` installed](https://docs.astral.sh/uv/), then install the Python environment with uv:

```bash
uv sync
```

1. **Install pre-commit hooks**:

```bash
uv run pre-commit install
```


2. **Check code quality and run tests**:

```bash
# Linting and formatting
uv run ruff check .
uv run ruff format .

# Type checking
uv run pyright

# Testing
uv run pytest
uv run pytest --cov=src/ntt --cov-report=<...>  # with code coverage report
```

### 🚦 Next steps

Three paths into the codebase, depending on what you want to do first:

- **Benchmark a tokenizer** → jump to [Benchmarking a tokenizer](#benchmarking-a-tokenizer) below. The quick-start runs the untrained Compressed Sensing baseline against Poisson on a 4-session sandbox in roughly 2 minutes — no model training required.
- **Build a tokenizer from scratch** → open [`notebooks/v0.ipynb`](notebooks/v0.ipynb). End-to-end walkthrough: defining a `TokenizerSpec`, implementing the `BaseTokenizer`, registering it, wiring up `NTTDataModule`, and running a toy training loop. Companion: [`notebooks/example_usage.ipynb`](notebooks/example_usage.ipynb) for shorter snippets.
- **Train at scale** → set up data first ([`docs/public_dataset_creation.md`](docs/public_dataset_creation.md)), then launch via the `ntt` CLI as described below.

For a smoke test that runs end-to-end without downloading the full dataset, see the **[Quick-start sandbox](docs/public_dataset_creation.md#6-quick-start-sandbox)** section of the dataset guide. The debug config reads 4 sessions (~38 MB) of the `perich_miller_population_2018` brainset, whose full processed build is ~3 GB.

> **Optional - downstream evals.** The repo vendors the
> [`ntt-downstream`](https://github.com/enigma-brain/ntt-downstream) submodule
> under `downstream/` for behaviour-decoding / stimulus-encoding evals.
>
> It is **not** needed for general development of new tokenizers, but it is needed to run
> the full suite of evaluations we provide for tokenizers.
>
> To use it, pull the submodule and
> build its (separate, Python 3.13) venv — fresh clones can add
> `--recurse-submodules` to `git clone`, or in an existing checkout run
> `git submodule update --init downstream`.
>
> Full one-time setup:
> [`docs/downstream_eval.md`](docs/downstream_eval.md#quick-start).

## Benchmarking a tokenizer

The `ntt` CLI ships an evaluation job that scores a tokenizer on the 22-metric suite per session, aggregates across sessions, and (optionally) compares against a Poisson rate-matched baseline. It works on **trained checkpoints** *and* on **untrained tokenizer specs**, so a benchmark number can be produced with or without ever calling `.fit()`.

Quick reference: [`docs/eval_usage.md`](docs/eval_usage.md). Per-metric definitions: [`docs/eval_metrics.md`](docs/eval_metrics.md).

### Quick start: the untrained Compressed Sensing baseline

The Compressed Sensing tokenizer is **untrained by design** — a fixed Gaussian sensing matrix paired with FISTA reconstruction. Anyone with the quick-start sandbox built ([dataset guide §6](docs/public_dataset_creation.md#6-quick-start-sandbox), ~3 GB on disk) can produce a full evaluation report without a checkpoint:

```bash
uv run ntt job=eval \
  paths.run_name=cs_baseline \
  tokenizer@job.tokenizer_config=contrib/compressed_sensing \
  data@job.data_config=ntt_debug \
  job.compare_poisson=true \
  job.generate_plots=true \
  job.output_path=results/cs_baseline
```

End-to-end runtime is roughly 2 minutes on a single GPU (longer on CPU). Outputs land in `results/cs_baseline/`:

- `results.json` / `results.csv` — per-metric values aggregated across the validation sessions, plus normalized scores against the Poisson baseline (0 = Poisson, 1 = perfect reconstruction).
- `normalized_comparison_radar.png`, `metric_bars.png`, `metric_dumbbell.png` — comparison plots.
- `sessions/<brainset>/<session>/` — per-session reconstruction, ACG, firing-rate, and structure plots.

The 4-session sandbox is small enough that a couple of behavioural-decoder targets may skip (e.g. trial_result has only successful trials, cursor regression can fall short of the cross-validation fold count); these run cleanly on the full dataset configurations.

### Evaluate your own checkpoint

```bash
uv run ntt job=eval \
  paths.run_name=my_eval \
  job.ckpt_path=/path/to/model.ckpt \
  data@job.data_config=ntt_full \
  job.compare_poisson=true \
  job.generate_plots=true \
  job.output_path=results/my_eval
```

Swap `data@job.data_config=ntt_full` for `ntt_small` / `ntt_medium` / `ntt_debug` to evaluate on smaller subsets while keeping the same tier definitions. The full set of CLI flags (split selection, MLflow logging, tier aggregation, per-session caps) and Python building blocks for custom scripts are documented in [`docs/eval_usage.md`](docs/eval_usage.md).

### What the eval suite measures

The 22 metrics are organised into six families: **spike detection** (5), **single-unit statistics** (7), **pairwise interactions** (2), **population structure** (4), **information** (2 — including the intrinsic `token_dims_per_second` compression rate), and **downstream prediction** (2 behavioural decoders, when sessions support them). Each metric declares its supported data types (`SPIKE_TIMES` / `SPIKES` / `RATES`) and exposes its own side parameters; metrics that share intermediates (binned counts, ISIs, correlograms, …) compute them once per session.

Full per-metric reference: [`docs/eval_metrics.md`](docs/eval_metrics.md).

## Training a Tokenizer

The `ntt` command-line interface uses [Hydra](https://hydra.cc/) for configuration management. Training runs are launched via `uv run ntt` with configuration overrides.

### Basic Command

```bash
uv run ntt paths.run_name=my_experiment \
  tokenizer@job.tokenizer_config=contrib/simple_vae \
  loss@job.loss_config=contrib/mse
```

A training job needs a tokenizer and a loss. `experiment=simple_vae` bundles both with a data config and hyperparameters.

Training logs to the MLflow server in `MLFLOW_TRACKING_URI`. When it is unset, run metrics go to a local store at `logs/mlflow.db` (browse it with `mlflow ui --backend-store-uri sqlite:///logs/mlflow.db`). Logged checkpoints go to `mlruns/` in the directory you launched from.

To print the full configuration without running training, add Hydra’s `--cfg job` flag:

```bash
uv run ntt paths.run_name=test \
  tokenizer@job.tokenizer_config=contrib/simple_vae \
  loss@job.loss_config=contrib/mse --cfg job
```

### Configuration Structure

The configuration system is organized into composable groups under `config/`. The tree below shows the high-level layout — see the actual files for the full inventory:

```
config/
├── main.yaml                    # Entry point
├── job/                         # Training & evaluation jobs (train.yaml, eval.yaml, …)
├── tokenizer/contrib/           # Tokenizer specs (simple_vae, conv_vae, ldns, fsqffn,
│                                #  poisson_baseline, compressed_sensing, calm/, wanneural/, …)
├── data/                        # Data configs (ntt.yaml, ntt_debug.yaml,
│                                #  ntt_full.yaml, ntt_medium.yaml,
│                                #  ntt_small.yaml,
│                                #  plus eval-tier composites under selections/)
├── trainer/                     # Lightning Trainer (default, gpu, cpu, ddp, mps)
├── paths/                       # Per-machine path overrides
├── logger/                      # MLFlow / WandB / CSV
├── optimizer/, scheduler/       # Optimization configs
└── loss/                        # Loss specs
```

For the full set of CLI override patterns, see [`docs/hydra_overrides.md`](docs/hydra_overrides.md).

### Selecting a Tokenizer

Select the tokenizer config with the `tokenizer@job.tokenizer_config` group override (the loss is a separate required group, `loss@job.loss_config`):

```bash
# Use the simple VAE tokenizer (single-neuron toy)
uv run ntt paths.run_name=my_run \
  tokenizer@job.tokenizer_config=contrib/simple_vae loss@job.loss_config=contrib/mse

# Use the conv VAE tokenizer
uv run ntt paths.run_name=my_run \
  tokenizer@job.tokenizer_config=contrib/conv_vae loss@job.loss_config=contrib/mse

# Use the Poisson baseline (no learning — sanity / floor)
uv run ntt paths.run_name=my_run \
  tokenizer@job.tokenizer_config=contrib/poisson_baseline loss@job.loss_config=contrib/mse
```

You can also override individual tokenizer parameters:

```bash
uv run ntt paths.run_name=my_run \
  tokenizer@job.tokenizer_config=contrib/simple_vae loss@job.loss_config=contrib/mse \
  job.tokenizer_config.latent_dim=16 job.tokenizer_config.beta=0.5
```

### Selecting a Dataset

Select the data config with the `data@job.data_config` group override:

```bash
# Use the full public multi-dataset configuration
uv run ntt paths.run_name=my_run \
  tokenizer@job.tokenizer_config=contrib/simple_vae loss@job.loss_config=contrib/mse \
  data@job.data_config=ntt_full

# Use a debug subset for quick iteration
uv run ntt paths.run_name=my_run \
  tokenizer@job.tokenizer_config=contrib/simple_vae loss@job.loss_config=contrib/mse \
  data@job.data_config=ntt_debug
```

**Train only on training data:** held-out sessions are listed in `config/data/selections/composite/ntt_full_{val,test}_*.yaml` (the canonical published split, do not edit; regenerating with `scripts/generate_tiered_selections.py --force` creates a new split that does not reproduce it); training will stop with an error if it tries to use them.

`ntt_full` is the canonical public composition: 10 brainsets, 1,104 sessions, ~1,566 hours, ~297 GB processed. Seven held-in brainsets are used for training plus Tier-1/2/3 evaluation; three are reserved as Tier-4 (`xsource`) held-out evaluation sources.

| Dataset                              | Role     | Recording Type     | Brain Region        | Species  |
|--------------------------------------|----------|--------------------|---------------------|----------|
| `IBL_brainwide_2023`                 | held-in  | Neuropixels        | brainwide           | mouse    |
| `allen_visual_behaviour_neuropixels` | held-in  | Neuropixels        | visual cortex       | mouse    |
| `allen_visual_coding_neuropixels`    | held-in  | Neuropixels        | visual cortex + HPF | mouse    |
| `churchland_shenoy_neural_2012`      | held-in  | Utah array         | M1, PMd             | macaque  |
| `flint_slutzky_accurate_2012`        | held-in  | Utah array         | M1, PMd             | macaque  |
| `odoherty_sabes_nonhuman_2017`       | held-in  | Utah array         | M1, S1              | macaque  |
| `perich_miller_population_2018`      | held-in  | Utah array         | M1, PMd             | macaque  |
| `c4_dataset_cerebellum_2025`         | xsource  | Neuropixels        | cerebellum          | mouse + macaque |
| `cowley_v4_compact_2024`             | xsource  | Utah array         | V4                  | macaque  |
| `wingert_auditory_cortex_2026`       | xsource  | Utah array + Neuropixels | A1, PEG       | ferret   |

Companion configs sized for scaling experiments (same eval splits, smaller train coverage): `ntt_medium.yaml` (~60 % of held-in training sessions), `ntt_small.yaml` (~30 %).

### Reproducing the public dataset

The public dataset above can be rebuilt from scratch using the pipelines under `brainsets_pipelines/`. The build flow, per-brainset commands, sources and citations, and a quick-start sandbox on the ~3 GB `perich_miller_population_2018` build (run via `data@job.data_config=ntt_debug`) are all documented in:

→ Full guide: [`docs/public_dataset_creation.md`](docs/public_dataset_creation.md).

### Example: Full Training Command

```bash
uv run ntt \
  paths.run_name=vae_full_training \
  tokenizer@job.tokenizer_config=contrib/simple_vae \
  loss@job.loss_config=contrib/mse \
  data@job.data_config=ntt_full \
  job.trainer.max_epochs=10
```

### Output Configuration

A resolved configuration will include these key sections:

```yaml
paths:
  run_name: my_experiment              # Your run name
  output_dir: ${paths.log_dir}/train/${paths.run_name}  # Output directory

job:
  tokenizer_config:
    KIND: simple_vae                   # Tokenizer type
    latent_dim: 8
    hidden_dims: [64, 32]
    beta: 1.0

  data_config:
    train_selections: ...              # Training data splits
    val_selections: ...                # Validation data splits
    root: ${paths.data_dir}

  trainer:
    max_epochs: 6
    accelerator: auto
    precision: bf16-mixed
    gradient_clip_val: 1.0
```

### Logs and Output

Training outputs are saved to `logs/train/<run_name>/`. Each run creates a directory named after your `paths.run_name` value:

```
logs/
└── train/
    └── my_experiment/
        ├── checkpoints/
        │   ├── epoch0-step1000-val_loss0.45.ckpt
        │   └── last.ckpt -> epoch0-step1000-val_loss0.45.ckpt
        ├── logging.log
        ├── csv_metrics/           # If using CSV logger
        │   └── version_0/
        │       ├── hparams.yaml
        │       └── metrics.csv
        └── wandb/                  # If using W&B logger
            └── run-YYYYMMDD_HHMMSS-<id>/
                ├── files/
                └── logs/
```

The run name defaults to `<experiment_name>-<date>-<time>` if not specified. Checkpoints are saved based on `validation/loss` with early stopping enabled by default.

## 📜 Croissant manifest

`croissant.json` at the repo root is a [Croissant 1.0](https://docs.mlcommons.org/croissant/) (JSON-LD) metadata file describing the public dataset (10 brainsets, schema of the per-session HDF5 files, upstream URLs, and per-source licenses).

```bash
# Regenerate after editing anything under croissant/ or the selection YAMLs:
uv run python scripts/generate_croissant.py

# Validate against the Croissant 1.0 schema (mlcroissant is in the dev extra):
uv run --extra dev python -c "import mlcroissant as mlc; mlc.Dataset(jsonld='croissant.json'); print('ok')"
```

Inputs the generator reads:
- `croissant/upstream_sources.yaml` — per-brainset upstream URL, license, citation.
- `croissant/brainset_summary.csv` — per-brainset summary stats (role, sessions, units, minutes, species, recording tech, brain areas).
- `config/data/selections/composite/ntt_full_{val,test}_{xsession,xsubject}.yaml` — public split YAMLs (sha256-hashed into the manifest).

Dataset-level metadata (authors, URL, citation, version) lives at the top of `scripts/generate_croissant.py`.

## 📚 Further reading

| Topic | Where |
|---|---|
| Build the public dataset (10 brainsets, ~297 GB) + quick-start sandbox | [`docs/public_dataset_creation.md`](docs/public_dataset_creation.md) |
| Croissant 1.0 metadata manifest | [`croissant.json`](croissant.json) (regen: `scripts/generate_croissant.py`) |
| End-to-end "build your own tokenizer" walkthrough | [`notebooks/v0.ipynb`](notebooks/v0.ipynb) |
| Shorter usage snippets | [`notebooks/example_usage.ipynb`](notebooks/example_usage.ipynb) |
| Hydra override cheatsheet | [`docs/hydra_overrides.md`](docs/hydra_overrides.md) |
| Pretrained tokenizers on Hugging Face | [`docs/pretrained_models.md`](docs/pretrained_models.md) |
| Evaluation pipeline | [`docs/eval_usage.md`](docs/eval_usage.md) |
| Evaluation metrics reference | [`docs/eval_metrics.md`](docs/eval_metrics.md) |
| (Downstream) decoding/encoding eval | [`docs/downstream_eval.md`](docs/downstream_eval.md) |
| Running individual brainset pipelines | [`brainsets_pipelines/README.md`](brainsets_pipelines/README.md) |

## 🤝 Contributing

### Development Environment Setup

The repository ships a `Dockerfile` for a reproducible development container (CUDA-enabled PyTorch base image, `uv`, and a non-root user that matches your host UID/GID). Using it is optional; a plain `uv sync` works as well.

#### 1. Configure your `.env` file
Copy `.env.example` to `.env` and fill in the values you need. The container user variables are required for the build:

    echo -e "USER_ID=$(id -u)\nGROUP_ID=$(id -g)\nUSERNAME=$(whoami)" >> .env

#### 2. Build and run

    set -a && source .env && set +a
    docker build -t ntt-dev \
      --build-arg USER_ID --build-arg GROUP_ID --build-arg USERNAME .
    docker run -d --name ntt-dev --gpus all --shm-size=16g \
      --env-file .env -v "$PWD":/src/ntt -v ntt-venv:/src/ntt/.venv \
      -v /path/to/your/data:/data ntt-dev sleep infinity
    docker exec -it ntt-dev bash

Mount your data directory wherever your Hydra `paths` profile expects it (see [`docs/hydra_overrides.md`](docs/hydra_overrides.md)).

#### 3. VS Code Devcontainer (optional)

If you use VS Code, you can create a `.devcontainer/devcontainer.json` that builds the `Dockerfile` and adds persistent named volumes. Here is an example configuration:

```jsonc
{
  "name": "${localEnv:USER}-ntt",
  "build": {
    "dockerfile": "../Dockerfile",
    "context": "..",
    "args": {
      "USER_ID": "${localEnv:USER_ID}",
      "GROUP_ID": "${localEnv:GROUP_ID}",
      "USERNAME": "${localEnv:USER}"
    }
  },
  "workspaceFolder": "/src/ntt",
  "workspaceMount": "source=${localWorkspaceFolder},target=/src/ntt,type=bind",
  "mounts": [
    // Keep the image's virtual environment: the bind mount above would otherwise hide /src/ntt/.venv
    "source=ntt-venv-${localEnv:USER},target=/src/ntt/.venv,type=volume",
    // Persist shell history across rebuilds
    "source=ntt-bash-history-${localEnv:USER},target=/home/${localEnv:USER}/.commandhistory,type=volume",
    // Optional: persist Claude Code session history across rebuilds (only useful if you use Claude Code in the container)
    "source=ntt-claude-${localEnv:USER},target=/home/${localEnv:USER}/.claude,type=volume"
  ],
  "runArgs": ["--gpus", "all"],
  "remoteEnv": {
    "HISTFILE": "/home/${localEnv:USER}/.commandhistory/.bash_history"
  },
  "remoteUser": "${localEnv:USER}",
  "overrideCommand": true,
  "postCreateCommand": "uv run pre-commit install && echo 'Devcontainer ready!'"
}
```

The `workspaceMount` bind-mounts your checkout at `/src/ntt`, and because the Dockerfile installs the virtual environment at `/src/ntt/.venv`, the `ntt-venv-*` volume keeps it visible (Docker fills a new volume from the image's `.venv` on first use). The `~/.claude` mount is optional and can be dropped if you do not use Claude Code in the container. The Dockerfile does not pre-create the mount points, so Docker creates them as root-owned directories on first use; if the container user cannot write to a volume, fix ownership once with `sudo chown -R $(id -u):$(id -g) ~/.commandhistory ~/.claude` inside the container. The `HISTFILE` env var tells bash to write history into the `~/.commandhistory` volume directory.

Once your `.devcontainer/devcontainer.json` is in place, open the repo in VS Code and select **"Reopen in Container"**.

### Code Quality Standards

Run `uv run ruff check .`, `uv run ruff format .`, and `uv run pyright` before committing. Pre-commit hooks enforce these — install with `uv run pre-commit install`. CI re-runs the same checks on every PR.

## License

The code is released under the [MIT License](LICENSE). Some modules adapt third-party code, mostly under `src/ntt/contrib/`. Those parts keep their original license (Apache-2.0, BSD-3-Clause or MIT), as noted in each file's header; [`LICENSES/`](LICENSES/) holds the upstream license texts and lists which files use each. The pretrained weights, embeddings and model cards on Hugging Face are licensed CC-BY-NC-4.0 ([`docs/pretrained_models.md`](docs/pretrained_models.md)). Each brainset keeps the license of its upstream source ([`croissant/upstream_sources.yaml`](croissant/upstream_sources.yaml)).

## Citations & acknowledgements

The NTT dataset is built on the [**brainsets**](https://github.com/neuro-galaxy/brainsets) ingestion framework. If you use this dataset or any of the pipelines under `brainsets_pipelines/`, please cite:

```bibtex
@inproceedings{azabou2023unified,
    title={A Unified, Scalable Framework for Neural Population Decoding},
    author={Mehdi Azabou and Vinam Arora and Venkataramana Ganesh and Ximeng Mao and Santosh Nachimuthu and Michael Mendelson and Blake Richards and Matthew Perich and Guillaume Lajoie and Eva L. Dyer},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
}
```

Per-brainset upstream citations are listed in [`docs/public_dataset_creation.md`](docs/public_dataset_creation.md#7-citations); please cite the originating publication for every brainset you actually use.
