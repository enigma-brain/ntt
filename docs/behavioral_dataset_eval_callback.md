# Behavioral Dataset Eval Callback

This callback runs behavioral decoding metrics on an explicitly selected subset
during training, instead of relying on the ordinary validation split contents.

The implementation lives in `src/ntt/evaluation/lightning.py`.

## Recommended usage

Add the callback under `job.trainer.callbacks` in the experiment config that
should use it.

Example:

```yaml
job:
  trainer:
    callbacks:
      behavioral_dataset_eval:
        _target_: ntt.evaluation.lightning.BehavioralDatasetEvalCallback
        eval_config:
          _target_: ntt.evaluation.config.schema.EvaluationConfig
          allow_lossy_conversion: true
          conversion_bin_size_ms: 1.0
          compute_baseline: false
          metrics:
            behavioral_regression_cursor_pos:
              metric_class: behavioral_regression
              target_field: cursor_pos
              include_phases: [reach_period, random_period]
              drop_invalid: true
              min_num_features: null
        selections:
          - _target_: ntt.core.data.components.config.SelectionConfig
            selection:
              - brainset: perich_miller_population_2018
                sessions: [c_20131022_center_out_reaching]
            config:
              ignore_splits: true
        log_prefix: behavioral_dataset_eval
        max_batches: null
        max_sessions: null
        aggregation_strategy: mean
        eval_on_epoch_end: true
        every_n_epochs: 1
        run_on_fit_end: true
```

## How to run training

Once the callback is part of the experiment config, run training normally:

```bash
uv run ntt paths.run_name=my_run experiment=my_experiment
```

To inspect the composed config before running:

```bash
uv run ntt paths.run_name=my_run experiment=my_experiment print_cfg=true
```

## What to expect

The callback logs metrics under its configured prefix, by default:

- `behavioral_dataset_eval/...`

Typical outputs are written under:

- `logs/train/<run_name>/`

If CSV logging is enabled, callback metrics appear in:

- `logs/train/<run_name>/csv_metrics/version_*/metrics.csv`

## Notes

- `selections` controls exactly which sessions are evaluated.
- `eval_config.metrics` controls which behavioral decoding metrics are run.
- `ignore_splits: true` is useful when you want to evaluate the full listed
  sessions rather than the pre-defined validation subset within them.
