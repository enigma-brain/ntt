# Custom evaluations - evaluating a different metric set on specific sessions

Assume we want to create a linear-regression only evaluation, on an extended set of sessions than the default.

We will create new eval and data configs and override those when calling the eval job.

```bash
  uv run ntt job=eval \
    paths.data_dir=/absolute/path/to/brainsets/preprocessed \
    paths.run_name=regression \
    job.ckpt_path=/absolute/path/to/model.ckpt \
    data@job.data_config=custom_ridge_regression \
    evaluation@job.eval_config=custom_ridge_regression_eval \
    job.output_path=results/regression \
    job.compare_poisson=true
```
The last line also evaluates a Poisson baseline with the same token bin size.


#### Change the metrics set
First, create a new eval config at `config/evaluation/`, let's call it `config/evaluation/custom_ridge_regression_eval.yaml`, with contents:

```yaml
defaults:
  - metrics/behavioral_regression_cursor_pos@metrics.behavioral_regression_cursor_pos
  - metrics/behavioral_regression_cursor_vel@metrics.behavioral_regression_cursor_vel
  - _self_

metrics:
  behavioral_regression_cursor_pos:
    eval_drop_invalid: true
    chunk_window_ms: [200, 1000]
    n_features: null
    feature_reduction: none
    behavioral_bin_size_ms: 10

  behavioral_regression_cursor_vel:
    eval_drop_invalid: true
    chunk_window_ms: [200, 1000]
    n_features: null
    feature_reduction: none
    behavioral_bin_size_ms: 10
```

**`defaults`** tells Hydra which YAML files to load. Here, it loads the position and velocity metric definitions from `config/evaluation/metrics/`. For example:
  - metrics/behavioral_regression_cursor_pos@metrics.behavioral_regression_cursor_pos
  means: “Load config/evaluation/metrics/behavioral_regression_cursor_pos.yaml and use its settings for cursor-position regression.” Putting `_self_` last means: “Then apply the settings written below in this file.” So our chunk_window_ms: [200, 1000] replaces the value from the loaded metric file.

#### Change the data selection
For evaluating on more (eval) Perich-Miller sessions, we create a selection config in `config/data/selections/custom_ridge_regression_data_selection.yaml`, with contents:

```yaml
_target_: ntt.core.data.components.config.SelectionConfig
selection:
  # Cross-session: held-out sessions from subjects seen during training.
  - brainset: perich_miller_population_2018
    sessions:
      - c_20151104_center_out_reaching
      - c_20151106_center_out_reaching
      - m_20140303_center_out_reaching
  # Cross-subject: sessions from subjects not seen during training.
  - brainset: perich_miller_population_2018
    sessions:
      - j_20160406_center_out_reaching
      - t_20130822_random_target_reaching
      - t_20130906_random_target_reaching
config:
  ignore_splits: true # NTT defines its own train/val splits, linear regression and classification internally define their own train/val splits.
```

Then we need to also add a _data config_, which selects that split, using the defaults of data config _`_ntt`_ (`config/data/_ntt.yaml`) otherwise - create file `config/data/custom_ridge_regression.yaml`:

```yaml
defaults:
  - _ntt
  - selections@train_selections.0: perich_miller_population_2018_train
  - selections@val_selections.0: custom_ridge_regression_data_selection
```

Then we can run the command as stated above, or additionally enable MLFlow logging. To run it for every row of a CSV list (see [Evaluate a CSV list](../eval_usage.md#evaluate-a-csv-list)):

```bash
uv run python scripts/run_eval_cohort.py cohort.csv \
    --data-dir /absolute/path/to/brainsets/preprocessed \
    --job eval \
    --override data@job.data_config=custom_ridge_regression \
    --override evaluation@job.eval_config=custom_ridge_regression_eval \
    --override job.compare_poisson=true \
    --override job.mlflow_logging.enabled=true
```
