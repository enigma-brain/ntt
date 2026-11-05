# Wingert Auditory Cortex Dataset (2026)

High-density extracellular recordings from ferret primary (A1) and secondary (PEG) auditory cortex during passive listening to natural sounds. From [Wingert, Parida, Norman-Haignere & David, *Nature Neuroscience* 2026](https://doi.org/10.1038/s41593-026-02216-0).

**Source**: [Zenodo 18331549](https://zenodo.org/records/18331549) (recordings, metadata, stimulus spectrograms)

## What's in each session file

Each session corresponds to one recording site (multi-unit probe insertion).

| Field | Type | Description |
|---|---|---|
| `spikes` | `IrregularTimeSeries` | Sorted single-unit spike times (session-absolute, seconds). `unit_index` maps to `units`. |
| `units` | `ArrayDict` | Unit metadata (see below) |
| `stimuli` | `Interval` | Per-presentation stimulus windows. One entry per epoch occurrence. |
| `stimuli.spectrogram_index` | `int64` | Index into the `spectrograms` lookup table |
| `stimuli.epoch_name` | `str` | Original NEMS epoch name (e.g. `STIM_seq0042`) |
| `spectrograms` | `ArrayDict` | Unique stimulus spectrograms (see below) |
| `spectrogram_fs` | `float` | Spectrogram sampling rate (100 Hz) |
| `spectrogram_f_min` | `float` | Minimum frequency (200 Hz) |
| `spectrogram_f_max` | `float` | Maximum frequency (20,000 Hz) |

### `units` fields

| Field | Description |
|---|---|
| `id` | Cell ID string (e.g. `PRN018a-003-1`) |
| `unit_number` | Sequential index (0, 1, 2, ...) |
| `brain_area` | `A1`, `PEG`, `AC`, `HC`, or `unknown` |
| `layer` | Cortical layer: `13` (L1/3), `44` (L4), `56` (L5/6), or `unknown` |
| `celltype` | `RS` (regular-spiking), `NS` (narrow-spiking), `RD` (regular-deep), `ND` (narrow-deep) |
| `depth` | Depth from cortical surface (micrometers) |
| `spike_width` | Spike waveform width (ms); narrow/broad threshold ~0.35-0.375 ms |
| `isolation` | Spike-sorting isolation quality (%) |
| `narrow` | Boolean: narrow-spiking (putative inhibitory) |
| `spike_count` | Total spike count in the session |

### `spectrograms` fields

| Field | Description |
|---|---|
| `id` | Epoch name (e.g. `STIM_seq0042`) |
| `data` | 3-D float32 array `(n_stimuli, 32, max_time_bins)` — cochleagram (log-spaced frequency, 200-20k Hz) |
| `length` | Per-stimulus time length in bins (only present if lengths vary) |

The cochleagram is the raw spectrogram as stored by the authors (gammatone filter bank, 32 channels, log-spaced). No log-compression or normalization is applied — apply these in your training pipeline as needed.

## Dataset structure

- **4 ferrets**: CLT (21 sites, 64-ch silicon probe), LMD (9 sites), PRN (33 sites), SLJ (5 sites, all Neuropixels)
- **68 recording sites** (one session per site), 9-248 units per site
- **Brain areas**: A1 (49 sites), PEG (15 sites), AC/HC/other (4 sites)
- **Stimuli**: 50-400 natural sound sequences per session, each ~18-22 seconds
- **Spectrograms**: 32-channel cochleagram, 2000-2200 time bins per stimulus (100 Hz)
- **Task**: Passive listening (awake, head-fixed, no behavioral task)
- **Spike sorting**: Kilosort2 with manual curation in phy

## Splits

Splits follow the authors' NEMS `split_using_epoch_occurrence_counts` logic, with a validation carve-out:

1. Count repetitions per unique stimulus name.
2. Compute the mean of the **distinct** repetition counts (e.g., mean of [1, 2, 30] = 11).
3. Stimuli with count **above** the mean → test (typically the 6 heavily-repeated test sequences).
4. Stimuli at or below the mean → estimation pool (train + val).
5. The estimation pool is further split ~85/15 into train/val by stimulus identity (seed 42).

| Split | Content |
|---|---|
| `train_domain` | ~85% of estimation stimuli (single-presentation + low-repeat) |
| `valid_domain` | ~15% of estimation stimuli (held out by stimulus identity) |
| `test_domain` | All intervals for high-repetition stimuli (typically 6 test sequences, 10-30 repeats) |

The split is **stimulus-based**: no sound appears in more than one split. The high-repetition test stimuli enable noise-ceiling estimation by averaging across repetitions.

## Running

```bash
# Single session (downloads ~4.5 GB recordings.zip on first run)
brainsets prepare brainsets_pipelines/wingert_auditory_cortex_2026 --local --use-active-env \
  -s 0 --raw-dir /path/to/raw --processed-dir /path/to/processed

# All sessions (parallel)
brainsets prepare brainsets_pipelines/wingert_auditory_cortex_2026 --local --use-active-env \
  --cores 8 --raw-dir /path/to/raw --processed-dir /path/to/processed
```

## Citation

Wingert, JC, Parida, S, Norman-Haignere, SV, David, SV. (2026). Convolutional neural network models describe the encoding subspace of local circuits in auditory cortex. *Nature Neuroscience*. DOI: [10.1038/s41593-026-02216-0](https://doi.org/10.1038/s41593-026-02216-0)

Data DOI: [10.5281/zenodo.18331549](https://doi.org/10.5281/zenodo.18331549). License: CC-BY-4.0.
