# Public NTT dataset — creation guide

This guide reproduces the **public release** of the NTT dataset: 10 brainsets, 1,104 sessions, 266,660 units, ~1,566 hours of recording. Three of the brainsets (`c4_dataset_cerebellum_2025`, `cowley_v4_compact_2024`, `wingert_auditory_cortex_2026`) are reserved as Tier-4 *xsource* held-out evaluation sources; the other seven are used for training plus Tier-1/2/3 evaluation.

For a small subset (the ~3 GB `perich_miller_population_2018` build; the debug config reads 4 of its sessions, ~38 MB) that trains in minutes on a CPU, jump to **[Quick-start sandbox](#6-quick-start-sandbox)**.

---

## 1. Overview

| Brainset | role | sessions | units | recording | processed size | source |
|---|---|---:|---:|---:|---:|---|
| `IBL_brainwide_2023` | held-in | 331 | 101,557 | 446 h | ~42 GB | DANDI:000409 |
| `allen_visual_behaviour_neuropixels` | held-in | 103 | 82,133 | 276 h | ~125 GB | AllenSDK |
| `allen_visual_coding_neuropixels` | held-in | 58 | 40,010 | 159 h | ~55 GB | AllenSDK |
| `churchland_shenoy_neural_2012` | held-in | 10 | 1,911 | 29 h | ~25 GB | DANDI:000070 |
| `flint_slutzky_accurate_2012` | held-in | 5 | 957 | 1 h | ~144 MB | CRCNS (anonymous) |
| `odoherty_sabes_nonhuman_2017` | held-in | 47 | 16,566 | 14 h | ~27 GB | Zenodo 3854034 |
| `perich_miller_population_2018` | held-in | 111 | 10,410 | 43 h | ~3 GB | DANDI:000688 |
| `c4_dataset_cerebellum_2025` | xsource | 321 | 4,960 | 387 h | ~9 GB | bundled URLs |
| `cowley_v4_compact_2024` | xsource | 50 | 4,774 | 122 h | ~6 GB | Figshare 30500090 |
| `wingert_auditory_cortex_2026` | xsource | 68 | 3,382 | 86 h | ~4 GB | Zenodo 18331549 |
| **TOTAL** | | **1,104** | **266,660** | **1,566 h** | **~297 GB** | |

Sizes are the on-disk footprint of *processed* H5 files — the raw downloads occupy more during conversion and can be deleted afterwards. Numbers are approximate.

---

## 2. Storage layout & paths config

Each pipeline writes one `<session>.h5` per session under a brainset-named directory:

```
<DATA_DIR>/
├── IBL_brainwide_2023/
│   ├── csh_zad_001_20200116_csh_zad_001.h5
│   └── ...
├── allen_visual_behaviour_neuropixels/
├── ...
└── wingert_auditory_cortex_2026/
```

`<DATA_DIR>` is whatever `paths.data_dir` resolves to in your Hydra config. Three ways to set it:

- **Default** (`config/paths/default.yaml`): resolves to `${PROJECT_ROOT}/data`. Drop the processed files under `<repo>/data/<brainset>/...` and nothing else needs to change.
- **CLI override**: `uv run ntt ... paths.data_dir=/abs/path/to/processed`.
- **Per-machine paths config**: copy `config/paths/example_cluster.yaml` to `config/paths/<machine>.yaml`, edit the `data_dir` line, then select with `paths=<machine>` on the CLI.

If you create a per-machine paths config, add it to `.gitignore` (or check in a sanitized copy) — these files frequently embed absolute home-dir or cluster paths and you may not want to share them.

---

## 3. One-time setup

The data ingestion pipelines live in `brainsets_pipelines/<name>/pipeline.py`. They are run via the `brainsets prepare` CLI, which spins up an isolated `uv` environment per pipeline based on the `# /// brainset-pipeline ... # ///` PEP-723 header at the top of each `pipeline.py`. See `brainsets_pipelines/README.md` for the full CLI reference.

Install the CLI (already a dependency of NTT, so this is implicit if you ran `uv sync`):

```bash
uv sync                       # installs brainsets in the project env
uv run brainsets --help       # sanity check
```

One per-pipeline note worth flagging: `flint_slutzky_accurate_2012` downloads from CRCNS using a hard-coded anonymous-login POST. No account needed, but the request can stall behind a corporate proxy that rewrites the form data.

---

## 4. Building each brainset

The pattern is the same for every pipeline. `<RAW>` is wherever you want to keep the originals (deletable after processing); `<DATA_DIR>` is the processed root that NTT will read from.

```bash
brainsets prepare brainsets_pipelines/<NAME> --local --cores 8 \
    --raw-dir <RAW>/<NAME> \
    --processed-dir <DATA_DIR>/<NAME>
```

Useful flags: `-s INDEX` runs a single manifest row (smoke test); `--download-only` fetches raw without processing; `-v` is verbose. Per-pipeline flags are visible via the pipeline's `parser` block.

| Brainset | Command (replace `<RAW>` and `<DATA_DIR>`) |
|---|---|
| `IBL_brainwide_2023` | `brainsets prepare brainsets_pipelines/IBL_brainwide_2023 --local --cores 8 --raw-dir <RAW>/IBL_brainwide_2023 --processed-dir <DATA_DIR>/IBL_brainwide_2023` |
| `allen_visual_behaviour_neuropixels` | `brainsets prepare brainsets_pipelines/allen_visual_behaviour_neuropixels --local --cores 8 --raw-dir <RAW>/allen_visual_behaviour_neuropixels --processed-dir <DATA_DIR>/allen_visual_behaviour_neuropixels` |
| `allen_visual_coding_neuropixels` | `brainsets prepare brainsets_pipelines/allen_visual_coding_neuropixels --local --cores 8 --raw-dir <RAW>/allen_visual_coding_neuropixels --processed-dir <DATA_DIR>/allen_visual_coding_neuropixels` |
| `churchland_shenoy_neural_2012` | `brainsets prepare brainsets_pipelines/churchland_shenoy_neural_2012 --local --cores 8 --raw-dir <RAW>/churchland_shenoy_neural_2012 --processed-dir <DATA_DIR>/churchland_shenoy_neural_2012` |
| `flint_slutzky_accurate_2012` | `brainsets prepare brainsets_pipelines/flint_slutzky_accurate_2012 --local --cores 4 --raw-dir <RAW>/flint_slutzky_accurate_2012 --processed-dir <DATA_DIR>/flint_slutzky_accurate_2012` |
| `odoherty_sabes_nonhuman_2017` | `brainsets prepare brainsets_pipelines/odoherty_sabes_nonhuman_2017 --local --cores 8 --raw-dir <RAW>/odoherty_sabes_nonhuman_2017 --processed-dir <DATA_DIR>/odoherty_sabes_nonhuman_2017` |
| `perich_miller_population_2018` | `brainsets prepare brainsets_pipelines/perich_miller_population_2018 --local --cores 8 --raw-dir <RAW>/perich_miller_population_2018 --processed-dir <DATA_DIR>/perich_miller_population_2018` |
| `c4_dataset_cerebellum_2025` | `brainsets prepare brainsets_pipelines/c4_dataset_cerebellum_2025 --local --cores 8 --raw-dir <RAW>/c4_dataset_cerebellum_2025 --processed-dir <DATA_DIR>/c4_dataset_cerebellum_2025` |
| `cowley_v4_compact_2024` | `brainsets prepare brainsets_pipelines/cowley_v4_compact_2024 --local --cores 8 --raw-dir <RAW>/cowley_v4_compact_2024 --processed-dir <DATA_DIR>/cowley_v4_compact_2024` |
| `wingert_auditory_cortex_2026` | `brainsets prepare brainsets_pipelines/wingert_auditory_cortex_2026 --local --cores 8 --raw-dir <RAW>/wingert_auditory_cortex_2026 --processed-dir <DATA_DIR>/wingert_auditory_cortex_2026` |

**Behaviour data for decoding.** For `perich_miller_population_2018`, add `--extract-behaviour` to the command shown in the table. Without it, the processed files will not contain cursor or trial data, so decoding (predicting the monkey's cursor movement from tokens) cannot run, and each session is split into train, valid, and test by time instead of by trial. If you already built it without the flag, rebuild it with `--extract-behaviour --reprocess`.

Per-brainset details (split logic, unit metadata fields, additional flags) are in each pipeline's `pipeline.py` and, where present, the pipeline's `README.md`.

---

## 5. Verifying the build

After every pipeline has finished, the shipped public configs should resolve without errors:

```bash
# Resolve the full public config and exit (no data I/O — useful right after the build)
uv run ntt paths.run_name=public_smoke print_schema=true \
    tokenizer@job.tokenizer_config=contrib/simple_vae \
    loss@job.loss_config=contrib/mse \
    data@job.data_config=ntt_full
```

For a live one-batch sanity check that actually opens H5 files, drop `print_schema=true` and add a small `job.trainer.max_epochs=1` plus `job.trainer.limit_train_batches=1` (or any other Lightning-trainer override exposed by `config/trainer/default.yaml`).

Three configs ship for the public dataset, all under `config/data/`:

| Config | Train coverage | Use case |
|---|---|---|
| `ntt_full.yaml` | all held-in training sessions | full reproduction |
| `ntt_medium.yaml` | ~60 % of held-in training sessions | scaling experiments |
| `ntt_small.yaml` | ~30 % of held-in training sessions | scaling experiments |

All three share the same Tier-2/3/4 validation and test selections (under `config/data/selections/composite/`), so models trained at different scales are directly comparable.

`config/data/selections/` holds the canonical published split: the per-source train files, the `composite/ntt_full_{val,test}_*.yaml` tiers and the `*_train_{medium,small}.yaml` subsets. They are produced by `scripts/generate_tiered_selections.py` and `scripts/generate_scaled_selections.py`, which read `<DATA_DIR>/<brainset>/*.h5` (`--data-root DIR`, else `NTT_DATA_ROOT`, else `<repo>/data`). Both refuse to overwrite existing selection files unless you pass `--force`, and regenerating with `--force` creates a new split that does not reproduce the shipped one. Keep the shipped files to compare against published results.

---

## 6. Quick-start sandbox

If you want to get started quickly and run *something* end-to-end without downloading 297 GB, the repo ships `config/data/ntt_debug.yaml`. It uses 2 train + 2 val sessions of `perich_miller_population_2018` (~38 MB on disk).

```bash
# 1. Build perich_miller_population_2018 only (≈ 3 GB processed).
#    The --processed-dir below is the *brainset* directory; paths.data_dir
#    in step 2 is its parent.
brainsets prepare brainsets_pipelines/perich_miller_population_2018 --local --cores 4 \
    --extract-behaviour \
    --raw-dir /tmp/ntt_sandbox/raw/perich_miller_population_2018 \
    --processed-dir /tmp/ntt_sandbox/processed/perich_miller_population_2018

# 2. Train with the debug config. paths.data_dir points at the parent of the
#    perich_miller_population_2018/ directory created in step 1.
uv run ntt paths.run_name=sandbox \
    paths.data_dir=/tmp/ntt_sandbox/processed \
    tokenizer@job.tokenizer_config=contrib/simple_vae \
    loss@job.loss_config=contrib/mse \
    data@job.data_config=ntt_debug \
    job.trainer.max_epochs=1 \
    job.trainer.val_check_interval=1.0
```

The four sessions actually used are pinned in `config/data/selections/perich_miller_sample_train.yaml` and `..._val.yaml`. If you want a *single*-session run (~10 MB), copy those files, drop one entry, and wire your copies into a new debug config — every selection YAML is a thin pointer list, no code change needed.

---

## 7. Citations

The NTT dataset is built on the **brainsets** ingestion framework. If you use this dataset (or any pipeline in `brainsets_pipelines/`), please cite:

```bibtex
@inproceedings{azabou2023unified,
    title={A Unified, Scalable Framework for Neural Population Decoding},
    author={Mehdi Azabou and Vinam Arora and Venkataramana Ganesh and Ximeng Mao and Santosh Nachimuthu and Michael Mendelson and Blake Richards and Matthew Perich and Guillaume Lajoie and Eva L. Dyer},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
}
```

Brainsets repository: <https://github.com/neuro-galaxy/brainsets>.

### Per-dataset citations

Cite the upstream publication for every brainset you use. The links below are the canonical data sources (also recorded as `BrainsetDescription.source` in each `pipeline.py`); follow them for the dataset's own citation guidance.

| Brainset | Source | Notes |
|---|---|---|
| `IBL_brainwide_2023` | <https://dandiarchive.org/dandiset/000409> | IBL "Brain-Wide Map" dataset; see DANDI page for the IBL collaboration's preferred citations. |
| `allen_visual_behaviour_neuropixels` | <https://allensdk.readthedocs.io/en/latest/visual_behavior_neuropixels.html> | Follow Allen Institute citation guidelines linked from the AllenSDK page. |
| `allen_visual_coding_neuropixels` | <https://allensdk.readthedocs.io/en/latest/visual_coding_neuropixels.html> | Associated paper: Siegle et al., *Nature* 2021, "Survey of spiking in the mouse visual system reveals functional hierarchy". |
| `churchland_shenoy_neural_2012` | <https://dandiarchive.org/dandiset/000070> | Churchland et al., *Nature* 2012, "Neural population dynamics during reaching". |
| `flint_slutzky_accurate_2012` | <https://portal.nersc.gov/project/crcns/download/dream/data_sets/Flint_2012> | Flint et al., *J. Neural Eng.* 2012; CRCNS dataset hosted via the DREAM project. |
| `odoherty_sabes_nonhuman_2017` | <https://zenodo.org/records/3854034> | Companion paper: Makin, O'Doherty, Cardoso, Sabes, *J. Neural Eng.* 2018. The pipeline downloads this 2020 version of Zenodo record 583331, which is what its `source` field still names. |
| `perich_miller_population_2018` | <https://dandiarchive.org/dandiset/000688> | Perich, Gallego, Miller, *Neuron* 2018, "A neural population mechanism for the rapid learning of novel motor associations". |
| `c4_dataset_cerebellum_2025` | `brainsets_pipelines/c4_dataset_cerebellum_2025/data_sources.json` | URLs bundled with the pipeline (UCL RDR, DOI [10.5522/04/23702850.v1](https://doi.org/10.5522/04/23702850.v1)). Companion paper: Beau et al., *Cell* 2025, "A deep-learning strategy to identify cell types across species from high-density extracellular recordings". |
| `cowley_v4_compact_2024` | <https://kilthub.cmu.edu/articles/dataset/30500090> | Cowley et al., *Nature* 2026; see also [`brainsets_pipelines/cowley_v4_compact_2024/README.md`](../brainsets_pipelines/cowley_v4_compact_2024/README.md). |
| `wingert_auditory_cortex_2026` | <https://zenodo.org/records/18331549> | Wingert, Parida, Norman-Haignere, David, *Nat. Neuroscience* 2026, DOI [10.1038/s41593-026-02216-0](https://doi.org/10.1038/s41593-026-02216-0); see also [`brainsets_pipelines/wingert_auditory_cortex_2026/README.md`](../brainsets_pipelines/wingert_auditory_cortex_2026/README.md). |
