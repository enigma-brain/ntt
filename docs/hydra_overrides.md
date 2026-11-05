# Hydra CLI Override Cheatsheet

Practical guide to overriding NTT configs from the command line.

## Two kinds of overrides

Hydra has two override mechanisms and mixing them up is the #1 source of frustration:

| What you want | Syntax | Example |
|---|---|---|
| **Change a scalar value** (string, number, bool) | `key=value` or `++key=value` | `job.seed=123` |
| **Swap a config group** (a whole YAML file) | `group@package=choice` | `data@job.data_config=ntt_small` |

The `group@package` form tells Hydra *which YAML file to compose*. The dot-path `key=value` form sets a field *inside* an already-composed config. You cannot swap a config group with dot-path syntax.

## Config group reference

### Training (`job=train`)

The train job composes these groups (from `config/job/train.yaml`):

| Group | Package (where it lands) | Default | CLI override |
|---|---|---|---|
| `/tokenizer` | `job.tokenizer_config` | `???` (required) | `tokenizer@job.tokenizer_config=contrib/ldns` |
| `/loss` | `job.loss_config` | `???` (required) | `loss@job.loss_config=contrib/mse` |
| `/data` | `job.data_config` | `ntt_full` | `data@job.data_config=ntt_small` |
| `/optimizer` | `job.optimizer_gen` | `adamw` | `optimizer@job.optimizer_gen=adam` |
| `/scheduler` | `job.scheduler_config` | `linear_warmup_cosine_annealing` | `scheduler@job.scheduler_config=...` |
| `/compile` | `job.compile_config` | `default` | `compile@job.compile_config=...` |
| `/trainer` | `job.trainer` | `default` | `trainer=gpu` |

### Evaluation (`job=eval`)

The eval job composes these groups (from `config/job/eval.yaml`):

| Group | Package (where it lands) | Default | CLI override |
|---|---|---|---|
| `/tokenizer` | `job.tokenizer_config` | `null` | `tokenizer@job.tokenizer_config=contrib/poisson_baseline` |
| `/data` | `job.data_config` | `ntt_debug` | `data@job.data_config=ntt_small` |
| `/evaluation` | `job.eval_config` | `default` | `evaluation@job.eval_config=...` |

### Top-level (from `config/main.yaml`)

| Group | Default | CLI override |
|---|---|---|
| `/paths` | `default` | `paths.data_dir=/path/to/brainsets` |
| `/job` | `train` | `job=eval` |
| `/experiment` | `null` | `experiment=simple_vae` |
| `/debug` | `null` | `debug=default` |

## Examples

### Training

```bash
# Minimal training run
uv run ntt \
  tokenizer@job.tokenizer_config=contrib/simple_vae \
  loss@job.loss_config=contrib/mse

# Training with small data + custom batch size
uv run ntt \
  tokenizer@job.tokenizer_config=contrib/ldns \
  loss@job.loss_config=contrib/poisson_nll \
  data@job.data_config=ntt_small \
  job.data_config.dataloader.batch_size=128

# Use an experiment config (bundles tokenizer + loss + data + hyperparams)
uv run ntt experiment=simple_vae

# Resume from checkpoint
uv run ntt experiment=simple_vae job.ckpt_path=/path/to/last.ckpt

# GPU trainer
uv run ntt experiment=simple_vae trainer=gpu
```

### Evaluation

```bash
# Eval with default debug data
uv run ntt job=eval \
  paths.run_name=my_eval \
  job.ckpt_path=/path/to/model.ckpt

# Eval with a different dataset
uv run ntt job=eval \
  paths.run_name=my_eval \
  job.ckpt_path=/path/to/model.ckpt \
  data@job.data_config=ntt_small

# Full eval: different data + poisson comparison + plots
uv run ntt job=eval \
  paths.run_name=my_eval \
  job.ckpt_path=/path/to/model.ckpt \
  data@job.data_config=ntt_full \
  job.output_path=results/my_eval \
  job.compare_poisson=true \
  job.generate_plots=true

# Eval an untrained tokenizer (e.g. Poisson baseline)
uv run ntt job=eval \
  paths.run_name=poisson_baseline \
  tokenizer@job.tokenizer_config=contrib/poisson_baseline \
  data@job.data_config=ntt_small \
  job.output_path=results/poisson
```

### Mixing group swaps and value overrides

You can combine both in one command. Group swaps select *which* YAML to load, then value overrides change fields *within* the loaded config:

```bash
uv run ntt job=eval \
  paths.run_name=custom_eval \
  job.ckpt_path=/path/to/model.ckpt \
  data@job.data_config=ntt_small \
  job.data_config.dataloader.batch_size=4 \
  job.data_config.dataloader.num_workers=2 \
  job.split=test \
  job.seed=0
```

## Common mistakes

| Mistake | Why it fails | Fix |
|---|---|---|
| `++job.data_config=ntt_small` | Sets field to the *string* `"ntt_small"` | `data@job.data_config=ntt_small` |
| `job/data_config=ntt_small` | Looks for config group `job/data_config` which doesn't exist | `data@job.data_config=ntt_small` |
| `/data@job.data_config=ntt_small` | Leading `/` is invalid in CLI overrides | `data@job.data_config=ntt_small` |
| `job.tokenizer_config=contrib/ldns` | Sets the field to a string instead of composing | `tokenizer@job.tokenizer_config=contrib/ldns` |

## Available data configs

| Config | Description |
|---|---|
| `ntt_debug` | 1 dataset (perich_miller), fast iteration |
| `ntt_small` | 30% of training data, 10 datasets |
| `ntt_medium` | 60% of training data, 10 datasets |
| `ntt_full` | All training data, 10 datasets (production) |
