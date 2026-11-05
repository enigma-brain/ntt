# Documentation index

This directory holds the long-form NTT documentation. The repository [`README.md`](../README.md) is the entry point for installing, benchmarking, and training; the files here go deeper on individual topics.

## Datasets and benchmarks

- [`public_dataset_creation.md`](public_dataset_creation.md) — building the public 10-brainset dataset (~297 GB processed) end-to-end, with per-pipeline commands, citations, and a quick-start sandbox in §6 (~3 GB build; the debug config reads ~38 MB of it).
- [`pretrained_models.md`](pretrained_models.md) — loading the published tokenizers from Hugging Face (`ntt.hub.load_pretrained`) and re-running our evaluation on them with `hf://` checkpoint paths.
- [`eval_usage.md`](eval_usage.md) — evaluation-pipeline TL;DR, full CLI reference, and Python building blocks for custom scripts. Runs on a trained checkpoint or on an untrained tokenizer spec (Poisson, Compressed Sensing).
- [`eval_metrics.md`](eval_metrics.md) — per-metric reference for the evaluation suite: definitions, parameters, data-type compatibility, and the shared-intermediate cache.
- [`robustness_metrics.md`](robustness_metrics.md) — the input-corruption and latent-perturbation robustness evaluations (`input_robustness`, `latent_robustness`).
- [`eval_battery_runbook.md`](eval_battery_runbook.md) — runbook for scoring a trained tokenizer on every evaluation axis, from one checkpoint to a whole cohort.
- [`eval_aggregation.md`](eval_aggregation.md) — aggregating per-tokenizer evaluation results into the seven summary scores and comparison tables.
- [`tokenizer-radar.md`](tokenizer-radar.md) — multi-tokenizer radar plots built from evaluation results.
- [`metric_glossary.md`](metric_glossary.md) and [`metric_glossary_simple.md`](metric_glossary_simple.md) — glossaries of the composite scores and raw metrics shown in the local dashboard (full and simple views).
- [`model_glossary.md`](model_glossary.md) and [`model_glossary_simple.md`](model_glossary_simple.md) — what each tokenizer family is (full and simple views).
- [`downstream_eval.md`](downstream_eval.md) — **extrinsic** evaluation (`ntt job=downstream_decoding` / `job=downstream_encoding`): behaviour decoding (POYO) and stimulus encoding (Cowley V4 / Wingert) on a frozen tokenizer checkpoint, run through the vendored `ntt-downstream` submodule. Complements the intrinsic `job=eval` suite. Includes one-time submodule setup.

## Configuration and CLI

- [`hydra_overrides.md`](hydra_overrides.md) — Hydra override cheatsheet, including the `group@package=choice` vs. dot-path distinction and other common pitfalls.

## Training internals

- [`behavioral_dataset_eval_callback.md`](behavioral_dataset_eval_callback.md) — Lightning callback that runs behavioural decoding during training.
- [`wanneural-architecture.md`](wanneural-architecture.md) — `wanneural` tokenizer-family architecture notes.

## Other

- [`guides/replotting.md`](guides/replotting.md) — re-plotting evaluation results and scaling-experiment workflows.
- [`dev/datasets/`](dev/datasets/perich_miller.md) — dataset notes: behavioral labels and extraction commands for the brainsets used in behavioural evaluation.
