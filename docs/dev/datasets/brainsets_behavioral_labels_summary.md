# Brainsets: Behavioral Labels and Additional Data Fields

## Overview

This document summarizes the behavioral labels and additional data fields available across public brainsets, including dataset sizes, extraction commands, and data shapes.

---

## Dataset Sizes

| Dataset | Size | Sessions | Species | Task Type |
|---------|------|----------|---------|-----------|
| **IBL_brainwide_2023** | 89GB | ~600+ | Mouse | Decision-making |
| **odoherty_sabes_nonhuman_2017** | 25GB | ~100+ | Macaque | Reaching |
| **churchland_shenoy_neural_2012** | 14GB | ~200+ | Macaque | Reaching |
| **perich_miller_population_2018** | 2GB | ~200 | Macaque | Reaching |
| **flint_slutzky_accurate_2012** | 144MB | 5 | Macaque | Reaching |

---

## 1. Perich-Miller (2 GB, Macaque Reaching)

**Task:** Center-out reaching or random target acquisition

**Extraction command:**
```bash
brainsets prepare perich_miller_population_2018 \
  --raw-dir /path/to/raw \
  --processed-dir /path/to/processed \
  --extract-behaviour \
  --reprocess
```

### Available Labels/Data

#### Session-level
- `session.task`: Task type string ("REACHING")

#### With `--extract-behaviour`:

**Trials** (Interval object)
- `target_id`: Integer target location identifier
- `result`: String, "R" for success, other for failure
- `target_on_time`, `go_cue_time`, `stop_time`: Float timestamps
- `is_valid`: Boolean mask
- `num_attempted`: Integer (for random target task)
- Shape: Per trial (variable count per session)

**Cursor** (IrregularTimeSeries, ~250Hz)
- `pos`: 2D cursor position (x, y), shape `(n_timepoints, 2)`
- `vel`: 2D velocity, shape `(n_timepoints, 2)`
- `acc`: 2D acceleration, shape `(n_timepoints, 2)`

**Movement Phases** (Data object with Intervals)
- `hold_period`: Pre-movement hold intervals
- `reach_period`: Active reaching intervals
- `return_period`: Return to center intervals
- `random_period`: Inter-trial periods
- Shape: Intervals covering different task phases

**Cursor Outlier Segments** (Interval)
- Time intervals with erroneous cursor data

### Use Cases
- Trial-level conditioning (target direction)
- Movement phase classification
- Continuous cursor position/velocity decoding
- Success/failure prediction

---

## 2. Churchland-Shenoy (14GB, Macaque Reaching)

**Task:** Center-out reaching

**Extraction command:**
```bash
brainsets prepare churchland_shenoy_neural_2012 \
  --raw-dir /path/to/raw \
  --processed-dir /path/to/processed \
  --extract-behaviour \
  --reprocess
```

### Available Labels/Data

#### Session-level
- `session.task`: "REACHING"
- `trials`: Always extracted (even without `--extract-behaviour`)

#### With `--extract-behaviour`:

**Trials** (Interval)
- `target_on_time`, `go_cue_time`, `move_begins_time`, `move_ends_time`: Event timing
- `discard_trial`: Boolean flag
- `task_success`: Boolean flag
- `is_valid`: Validity mask

**Cursor** (IrregularTimeSeries)
- `pos`, `vel`: 2D cursor position/velocity

**Hand** (IrregularTimeSeries)
- `pos`, `vel`: 2D hand position/velocity

**Eye** (IrregularTimeSeries)
- `pos`, `vel`: 2D eye position/velocity

**Movement Phases** (Data with Intervals)
- `hold_period`, `reach_period`, `return_period`

**Cursor Outlier Segments**

### Use Cases
- Multi-effector decoding (hand, eye, cursor)
- Event timing prediction
- Movement phase classification

---

## 3. Flint-Slutzky (144MB, Macaque Reaching)

**Task:** Reaching with right hand

**Extraction command:**
```bash
brainsets prepare flint_slutzky_accurate_2012 \
  --raw-dir /path/to/raw \
  --processed-dir /path/to/processed \
  --extract-behaviour \
  --reprocess
```

### Available Labels/Data

#### Session-level
- `session.task`: "REACHING"

#### With `--extract-behaviour`:

**Trials** (Interval)
- Basic trial timing structure

**Hand** (IrregularTimeSeries)
- `pos`, `vel`: 2D hand position/velocity

### Use Cases
- Hand position decoding
- Trial structure analysis

---

## 4. O'Doherty-Sabes (25GB, Macaque Reaching)

**Task:** Self-paced reaching to grid targets

**Extraction command:**
```bash
brainsets prepare odoherty_sabes_nonhuman_2017 \
  --raw-dir /path/to/raw \
  --processed-dir /path/to/processed \
  --extract-behaviour \
  --reprocess
```

### Available Labels/Data

#### Session-level
- `session.task`: "REACHING"

#### With `--extract-behaviour`:

**Cursor** (IrregularTimeSeries, 250Hz)
- `pos`, `vel`, `acc`: 2D cursor kinematics
- `target_pos`: 2D target positions, shape `(n_timepoints, 2)`

**Finger** (IrregularTimeSeries)
- `pos`, `vel`: 3D or 6D finger position/velocity (varies by session)

**No Movement Segments** (Interval)
- Periods without active movement

### Use Cases
- Continuous cursor decoding
- Target-directed reaching analysis
- Multi-finger tracking

---

## 5. IBL Brainwide (89GB, Mouse Decision-Making)

**Task:** Visual decision-making task

**Status:** ⚠️ **No behavioral data currently extracted**

### Potential Data (in NWB, not extracted)

- Trial outcomes (correct/incorrect)
- Response times
- Sensory stimuli (visual contrast)
- Choice (left/right)
- Pose data (DeepLabCut body tracking)
- Comprehensive brain region labels

**Note:** Would require custom pipeline modification to extract.

### Use Cases
- Decision prediction
- Reaction time prediction
- Whole-brain activity mapping
- Sensorimotor integration

---

## Summary Table: Behavioral Data Types

| Dataset           | Cursor | Hand | Eye | Finger | Trials | Movement Phases | Target Info |
|-------------------|--------|------|-----|--------|--------|-----------------|-------------|
| Perich-Miller     | ✓ (2D) | ❌   | ❌   | ❌     | ✓      | ✓               | ✓(target_id)|
| Churchland-Shenoy | ✓ (2D) |✓ (2D)|✓(2D)| ❌     | ✓      | ✓               | ✓           |
| Flint-Slutzky     | ❌     | ✓ (2D)| ❌  | ❌     | ✓      | ❌              | ❌         |
| O'Doherty-Sabes   | ✓ (2D) | ❌    | ❌  | ✓(3-6D)| ❌     | ❌              | ✓ (target_pos) |
| IBL               | ⚠️     | ⚠️    | ⚠️  | ❌      | ⚠️     | ❌              | ⚠️          |

**Legend:**
- ✓ Available with `--extract-behaviour`
- ❌ Not available
- ⚠️ Exists in raw data but not extracted

---

## Data Types

### IrregularTimeSeries
- Time-varying data with irregular sampling
- Contains `timestamps` array
- Attributes vary by data type (pos, vel, acc, etc.)
- Automatically sliced with neural data based on time windows

### Interval
- Time intervals with start/end times
- Can have additional attributes per interval
- Used for trials, movement phases, outlier segments
- Not time-varying but defines temporal structure

### Shape Conventions
- **2D positions/velocities**: `(n_timepoints, 2)` for (x, y)
- **3D positions/velocities**: `(n_timepoints, 3)` for (x, y, z)
- **6D finger data**: `(n_timepoints, 6)` for multiple DOFs
- **Trials**: Variable number per session
- **Batch data**: After collation, adds batch dimension as first axis

---

## Implementation Notes

### Forwarding through the data pipeline
- `session.task` is forwarded through the data pipeline by the `DataToModelInput` transform and is available in the batch dict as `batch["task"]`.
- Other behavioral fields are not propagated into the batch by default.

### Data Storage
All behavioral data stored in HDF5 files when extracted with `--extract-behaviour` flag. Data objects use the `temporaldata` library format:
- Top-level attributes on `Data` object (e.g., `data.cursor`, `data.trials`)
- Preserved through `Data.slice()` operations
- Lost during conversion to dict in `DataToModelInput` unless explicitly extracted

### Repeated trial conditions

Whether a dataset repeats the same trial condition across trials depends on its task design:

| Dataset | Repeated conditions |
|---------|---------------------|
| `perich_miller_population_2018` | Center-out sessions repeat reaches to the same targets; random-target sessions do not |
| `churchland_shenoy_neural_2012` | Center-out reaching with multiple reach conditions, repeated across trials |
| `flint_slutzky_accurate_2012` | Reaching task; only basic trial timing is extracted |
| `IBL_brainwide_2023` | Decision-making task; trials with the same stimulus and choice combination are repeated conditions once behaviour is extracted |
| `odoherty_sabes_nonhuman_2017` | Self-paced random-target task, so reach conditions are not repeated |

See the Neural Latents benchmark dataset descriptions (https://neurallatents.github.io/datasets.html) for the task designs.
