
# Perich-Miller (2 GB, macaque reaching)

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

**Target IDs (Targets)**
*Center-out reaching trials:*  
The 8 targets are evenly spaced at 45° intervals around a circle:

          ↑ ID 0 (90°)
     ↖ ID 7 (135°)    ↗ ID 1 (45°)
    ← ID 6 (180°)      → ID 2 (0°)
     ↙ ID 5 (-135°)   ↘ ID 3 (-45°)
          ↓ ID 4 (-90°)

Inter-trial samples carry target ID -1, giving 9 classes in total.

*Random target trials:*
- target_id is a 2D array (n_trials, 4)

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

### Conventions used by POYO

POYO and POYO+ decode a single readout on this dataset, `cursor_velocity_2d` (`cursor.vel`), scored with R². There is no classification readout. Velocity is z-scored (mean 0, std 20).

Training loss weights are assigned per timestamp from interval membership, and they multiply when intervals overlap:

| Interval (center-out) | Weight |
|---|---|
| `movement_phases.random_period` | 1.0 |
| `movement_phases.hold_period` | 0.1 |
| `movement_phases.reach_period` | 5.0 |
| `movement_phases.return_period` | 1.0 |
| `movement_phases.invalid` | 0.1 |
| `cursor_outlier_segments` | 0.0 (excluded) |

For random-target reaching, `random_period` has weight 1.0, `hold_period` 0.1 and `cursor_outlier_segments` 0.0. Evaluation is restricted to `movement_phases.reach_period` for center-out and to `movement_phases.random_period` for random-target sessions. POYO does not read `is_valid` or `trials.result`.
