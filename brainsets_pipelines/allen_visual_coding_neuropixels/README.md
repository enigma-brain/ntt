# Export notes

No pipeline-specific setup. Run via the standard `brainsets prepare` flow — the PEP-723 header in `pipeline.py` pins `python-version = "3.11"` and `allensdk>=2.16`, which `uv` resolves into an isolated env automatically.

```bash
brainsets prepare brainsets_pipelines/allen_visual_coding_neuropixels --local --cores 8 \
    --raw-dir <RAW>/allen_visual_coding_neuropixels \
    --processed-dir <DATA_DIR>/allen_visual_coding_neuropixels
```

See [`brainsets_pipelines/README.md`](../README.md) for the full CLI reference and [`docs/public_dataset_creation.md`](../../docs/public_dataset_creation.md) for the end-to-end public-dataset build guide.
