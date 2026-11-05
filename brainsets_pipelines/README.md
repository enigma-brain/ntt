# Brainsets Pipelines

Data ingestion pipelines for converting raw neural datasets into the brainsets format.

## Running a pipeline

```bash
# Run a pipeline (creates isolated uv environment from inline metadata)
brainsets prepare /path/to/brainsets_pipelines/<pipeline_name> --local --cores 8

# Process a single session (useful for testing)
brainsets prepare /path/to/brainsets_pipelines/<pipeline_name> --local -s 0

# Dev mode: skip isolated env, use current environment
brainsets prepare /path/to/brainsets_pipelines/<pipeline_name> --local --use-active-env

# Download only (no processing)
brainsets prepare /path/to/brainsets_pipelines/<pipeline_name> --local --download-only
```

## Key options

| Flag | Description |
|---|---|
| `--local` | Required for local pipeline directories |
| `--cores N` | Number of parallel workers (default: 4) |
| `-s INDEX` | Process single session by manifest index |
| `--use-active-env` | Skip isolated env setup (dev only) |
| `--download-only` | Only download raw data |
| `--raw-dir PATH` | Override raw data storage path |
| `--processed-dir PATH` | Override processed output path |
| `-v` | Verbose/debug output |

Pipelines may define additional arguments (e.g. `--extract-behaviour`, `--redownload`). Check each pipeline's `parser` for details.

## Dependencies

Each `pipeline.py` declares its own dependencies via inline metadata (PEP 723 style):

```python
# /// brainset-pipeline
# python-version = "3.11"
# dependencies = ["dandi==0.74.0"]
# ///
```

When running without `--use-active-env`, `brainsets prepare` uses `uv` to create an isolated environment with these dependencies automatically.
