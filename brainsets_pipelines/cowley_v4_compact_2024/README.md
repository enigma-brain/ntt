# Cowley V4 Compact Models Dataset

Utah array recordings from macaque area V4 during rapid serial visual presentation (RSVP) of natural images. From [Cowley et al., 2026](https://www.nature.com/articles/s41586-026-10150-1).

**Source**: [Figshare](https://kilthub.cmu.edu/articles/dataset/30500090) (spike data) + [GitHub](https://github.com/cowleygroup/V4_compact_models) (images & code)

## What's in each session file

| Field | Type | Description |
|---|---|---|
| `spikes` | `IrregularTimeSeries` | Sorted single-unit spike times (session-absolute), `unit_index` maps to `units` |
| `units` | `ArrayDict` | Unit metadata: `id` (channel ID), `brain_area` (all "V4") |
| `stimuli` | `Interval` | Per-presentation stimulus windows (STIM_ON → STIM_OFF) |
| `stimuli.image_id` | `int64` | Original 1-indexed image ID from the experiment |
| `stimuli.image_index` | `int64` | Index into the `images` lookup table |
| `images` | `ArrayDict` | `pixels`: uint8 array (N, 112, 112, 3), `id`: string IDs |

7 early PE sessions (201014–201020) have no image files available; spikes and stimuli are still exported.

## Dataset structure

- **3 animals**: WE (Wakko, 7 sessions), PE (Pepe, 28 sessions), RA (Raul, 15 sessions)
- **50 sessions** total, 19–96 sorted units per session
- **RSVP task**: 6–8 images per fixation trial, ~100ms each, 100ms gray ISI
- **Only correct trials** (result == 150) are exported

## Splits

Splits are **image-based within each session**: the unique images in a session are randomly partitioned 70/15/15 (train/val/test) using a fixed seed (42). Every stimulus presentation interval is then assigned to the split of its image, so no image ever leaks across splits.

Each session file has `train_domain`, `valid_domain`, `test_domain` — all three are populated with the stimulus intervals belonging to their respective image partition.

### Author's original session-level splits

The authors designated entire sessions as train/val/test. We preserve this as the `author_split` attribute (a string: `"train"`, `"val"`, or `"test"`) on each Data object for downstream use:

| Split | Sessions | Purpose |
|---|---|---|
| train (44) | All others | Ensemble CNN training |
| val (1) | 210224 | Early stopping / hyperparameter tuning |
| test (5) | 190923, 201025, 210225, 210325, 211022 | Held-out evaluation |

**Note on the authors' evaluation protocol**: At test time, they split each test session's images 50/50 — first half fits a linear ridge regression from CNN embeddings to neural responses, second half evaluates noise-corrected R². This within-session split is a modeling choice, not encoded in our export.

## Recovering repeat-averaged responses

The authors' `responses_raw` arrays are spike counts in **100ms bins starting 50ms after stimulus onset** (to account for V4 latency). To reconstruct from our data:

```python
# For each stimulus presentation:
#   window = [stimuli.start + 0.050, stimuli.start + 0.150]  (seconds)
#   count spikes per unit in that window
# Group by stimuli.image_id to collect repeats
# Average across repeats → responses_repeat_averaged
```

Images have varying repeat counts per session (4–24 repeats depending on session).

## Trial code reference

| Code | Meaning |
|---|---|
| 10 | STIM_ON |
| 40 | STIM_OFF |
| >10000 | IMAGE_ID (subtract 10000; PE/RA sessions only) |
| 150 | Correct trial result |

WE sessions store image IDs in `trialParams.image1_id`–`image6_id` instead of trial codes.

## Running

```bash
# Single session
brainsets prepare brainsets_pipelines/cowley_v4_compact_2024 --local --use-active-env -s 201025 \
  --raw-dir /path/to/raw --processed-dir /path/to/processed

# All sessions (parallel)
brainsets prepare brainsets_pipelines/cowley_v4_compact_2024 --local --cores 8 \
  --raw-dir /path/to/raw --processed-dir /path/to/processed
```
