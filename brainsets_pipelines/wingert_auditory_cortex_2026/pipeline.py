# /// brainset-pipeline
# python-version = "3.11"
# dependencies = [
#   "h5py",
#   "requests",
# ]
# ///

import datetime
import io
import json
import logging
import re
import tarfile
import tempfile
import zipfile
from argparse import ArgumentParser, Namespace
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import requests
from brainsets import serialize_fn_map
from brainsets.descriptions import (
    BrainsetDescription,
    DeviceDescription,
    SessionDescription,
    SubjectDescription,
)
from brainsets.pipeline import BrainsetPipeline
from brainsets.taxonomy import RecordingTech, Sex, Species
from temporaldata import ArrayDict, Data, Interval, IrregularTimeSeries

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ZENODO_RECORD = "18331549"
ZENODO_BASE = f"https://zenodo.org/records/{ZENODO_RECORD}/files"
RECORDINGS_ZIP_SIZE = 4_558_475_493  # expected size in bytes

# Animal prefix → recording technology.
# CLT: 64-channel silicon electrode arrays (closest available: UTAH_ARRAY_SPIKES).
# LMD, PRN, SLJ: Neuropixels short NHP probes.
ANIMAL_RECORDING_TECH = {
    "CLT": RecordingTech.UTAH_ARRAY_SPIKES,
    "LMD": RecordingTech.NEUROPIXELS_SPIKES,
    "PRN": RecordingTech.NEUROPIXELS_SPIKES,
    "SLJ": RecordingTech.NEUROPIXELS_SPIKES,
}

# Fraction of single-presentation stimuli held out for validation.
VALID_FRACTION = 0.15
SPLIT_SEED = 42

parser = ArgumentParser()
parser.add_argument("--redownload", action="store_true")
parser.add_argument("--reprocess", action="store_true")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _download_file(url: str, dest: Path, desc: str = "") -> None:
    """Stream-download *url* to *dest* with progress logging."""
    logger.info(f"Downloading {desc or url} ...")
    resp = requests.get(url, stream=True, timeout=7200)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(dest, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
            fh.write(chunk)
            downloaded += len(chunk)
            if total > 0 and downloaded % (200 * 1024 * 1024) < 8 * 1024 * 1024:
                pct = 100 * downloaded / total
                logger.info(f"  {downloaded / 1e9:.2f} / {total / 1e9:.2f} GB ({pct:.0f}%)")
    logger.info(f"Saved {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")


def _parse_epoch_csv(text: str) -> pd.DataFrame:
    """Parse a NEMS epoch CSV.

    The CSV has a header row with the columns ``name``, ``start`` and ``end``
    (in any order).
    """
    lines = text.strip().split("\n")
    if not lines:
        return pd.DataFrame(columns=["name", "start", "end"])

    first = lines[0].strip().strip('"')
    # Detect header by checking if the first token is non-numeric
    has_header = not first[0].isdigit() and not first[0] == "-"

    if not has_header:
        raise ValueError("Epoch CSV must have a header row with name, start and end columns")
    df = pd.read_csv(io.StringIO(text), header=0)

    # Normalise to canonical columns
    df.columns = [c.strip() for c in df.columns]
    for col in ("start", "end", "name"):
        if col not in df.columns:
            raise ValueError(f"Epoch CSV missing column '{col}': {list(df.columns)}")

    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"] = pd.to_numeric(df["end"], errors="coerce")
    df["name"] = df["name"].astype(str).str.strip()
    return df[["name", "start", "end"]].dropna(subset=["start", "end"])


def _parse_nems_recording(tgz_path: Path) -> dict:
    """Parse a NEMS .tgz recording archive *without* the NEMS library.

    Returns a dict with keys:
        spike_times   – {cell_id: 1-D float64 array of spike times (seconds)}
        stim_specs    – {epoch_name: 2-D float32 array (freq_channels, time_bins)}
        stim_fs       – stimulus sampling rate (Hz)
        resp_fs       – response sampling rate (Hz)
        stim_chans    – list of frequency-channel labels
        epochs        – pd.DataFrame (start, end, name)
        meta          – recording-level metadata dict
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(tmpdir, filter="data")

        tmpdir = Path(tmpdir)
        subdirs = [d for d in tmpdir.iterdir() if d.is_dir()]
        rec_dir = subdirs[0] if subdirs else tmpdir
        recname = rec_dir.name

        # Helper: find a file by suffix pattern (handles name mismatches)
        def _find(suffix: str) -> Path | None:
            # Try exact name first, then glob
            exact = rec_dir / f"{recname}.{suffix}"
            if exact.exists():
                return exact
            matches = list(rec_dir.glob(f"*.{suffix}"))
            return matches[0] if matches else None

        # ---- recording metadata ----
        meta = {}
        mp = _find("meta.json")
        if mp:
            meta = json.loads(mp.read_text())

        # ---- response signal (spike times) ----
        resp_json = _find("resp.json")
        resp_meta = json.loads(resp_json.read_text()) if resp_json else {}
        resp_fs = float(resp_meta.get("fs", 100))
        segments = resp_meta.get("segments", [[0, 0]])
        recording_total_bins = int(segments[0][1])

        resp_h5 = _find("resp.h5")
        if not resp_h5:
            raise FileNotFoundError(f"Missing resp.h5 in {tgz_path}")
        spike_times: dict[str, np.ndarray] = {}
        with h5py.File(resp_h5, "r") as fh:
            for key in fh.keys():
                spike_times[key] = fh[key][:].astype(np.float64)

        # ---- epochs (from resp signal — shared across signals in NEMS) ----
        epochs = pd.DataFrame(columns=["name", "start", "end"])
        ep_path = _find("resp.epoch.csv")
        if ep_path:
            epochs = _parse_epoch_csv(ep_path.read_text())

        # ---- stimulus signal ----
        stim_json = _find("stim.json")
        stim_meta = json.loads(stim_json.read_text()) if stim_json else {}
        stim_fs = float(stim_meta.get("fs", 100))
        stim_chans = stim_meta.get("chans", []) or []
        signal_type = stim_meta.get("signal_type", "")

        stim_h5 = _find("stim.h5")
        stim_csv = _find("stim.csv")

        # Stim-specific epoch table (fall back to resp epochs)
        stim_ep = _find("stim.epoch.csv")
        stim_epochs_df = _parse_epoch_csv(stim_ep.read_text()) if stim_ep else epochs

        stim_specs: dict[str, np.ndarray] = {}

        if "TiledSignal" in signal_type and stim_h5 is not None:
            # Keys = epoch names, values = (channels, time)
            with h5py.File(stim_h5, "r") as fh:
                for key in fh.keys():
                    stim_specs[key] = fh[key][:].astype(np.float32)

        elif stim_h5 is not None:
            # Could be TiledSignal without explicit type label, or RasterizedSignal
            with h5py.File(stim_h5, "r") as fh:
                keys = list(fh.keys())
                if any(k.startswith("STIM_") for k in keys):
                    for key in keys:
                        stim_specs[key] = fh[key][:].astype(np.float32)
                else:
                    # RasterizedSignal stored as H5 — extract per-epoch
                    arr = fh[keys[0]][:].astype(np.float32)
                    if arr.shape[0] > arr.shape[1]:
                        arr = arr.T  # ensure (channels, time)
                    stim_specs = _extract_epoch_spectrograms(arr, stim_epochs_df, stim_fs)

        elif stim_csv is not None:
            # RasterizedSignal in CSV: disk layout is (time, channels)
            arr = np.loadtxt(stim_csv, delimiter=",").astype(np.float32).T
            stim_specs = _extract_epoch_spectrograms(arr, stim_epochs_df, stim_fs)

        else:
            logger.warning(f"No stimulus data found in {tgz_path}")

    return {
        "spike_times": spike_times,
        "stim_specs": stim_specs,
        "stim_fs": stim_fs,
        "resp_fs": resp_fs,
        "stim_chans": stim_chans,
        "epochs": epochs,
        "meta": meta,
        "recording_total_bins": recording_total_bins,
    }


def _extract_epoch_spectrograms(
    continuous: np.ndarray,
    epochs_df: pd.DataFrame,
    fs: float,
) -> dict[str, np.ndarray]:
    """Slice a continuous (channels, time) array into per-epoch spectrograms."""
    rows = epochs_df[epochs_df["name"].str.startswith("STIM_")]
    out: dict[str, np.ndarray] = {}
    for name in rows["name"].unique():
        bounds = rows[rows["name"] == name].iloc[0]
        s = int(round(bounds["start"] * fs))
        e = int(round(bounds["end"] * fs))
        e = min(e, continuous.shape[1])
        out[name] = continuous[:, s:e]
    return out


def _find_site_tgz(recordings_dir: Path, site_id: str) -> list[Path]:
    """Find the .tgz file for a site, handling naming mismatches.

    Tries, in order:
      1. Exact: ``{site_id}_*.tgz``  (covers 65/68 sites)
      2. Base prefix: strip ``-B`` / ``-C`` suffix and retry  (SLJ032a-B)
      3. Short prefix: first 6 chars  (PRN015a → PRN015b)
    """
    # 1. Exact
    matches = list(recordings_dir.glob(f"{site_id}_*.tgz"))
    if matches:
        return matches

    # 2. Strip sub-site suffix (e.g. SLJ032a-B → SLJ032a)
    if "-" in site_id:
        base = site_id.split("-")[0]
        matches = list(recordings_dir.glob(f"{base}_*.tgz"))
        if matches:
            return matches

    # 3. Short prefix — animal (3 letters) + number (3 digits)
    m = re.match(r"([A-Z]{3}\d{3})", site_id)
    if m:
        candidates = list(recordings_dir.glob(f"{m.group(1)}*_*.tgz"))
        if len(candidates) == 1:
            return candidates

    return []


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class Pipeline(BrainsetPipeline):
    """Wingert et al. (2026) ferret auditory cortex dataset.

    High-density recordings from primary (A1) and secondary (PEG) auditory
    cortex in passively listening ferrets during presentation of natural sounds.
    Each recording site is exported as one session.

    Source: https://zenodo.org/records/18331549
    Paper:  https://doi.org/10.1038/s41593-026-02216-0
    """

    brainset_id = "wingert_auditory_cortex_2026"
    parser = parser

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    @classmethod
    def get_manifest(cls, raw_dir: Path, args: Namespace | None) -> pd.DataFrame:
        """One row per recording site, derived from cell_list.csv."""
        raw_dir.mkdir(exist_ok=True, parents=True)
        cell_list_path = raw_dir / "cell_list.csv"

        if not cell_list_path.exists() or (args and args.redownload):
            _download_file(f"{ZENODO_BASE}/cell_list.csv", cell_list_path, "cell_list.csv")

        df = pd.read_csv(cell_list_path, index_col=0)

        rows = []
        for site_id, grp in df.groupby("siteid"):
            animal = str(site_id)[:3]
            area_mode = grp["area"].mode()
            area = str(area_mode.iloc[0]) if len(area_mode) > 0 else "unknown"
            rows.append(
                {
                    "session_id": str(site_id),
                    "site_id": str(site_id),
                    "animal": animal,
                    "area": area,
                    "n_units_celllist": len(grp),
                }
            )

        return pd.DataFrame(rows).set_index("session_id")

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download(self, manifest_item):
        self.update_status("DOWNLOADING")
        self.raw_dir.mkdir(exist_ok=True, parents=True)

        site_id = manifest_item.site_id

        # ---- recordings.zip (shared, ~4.5 GB) ----
        recordings_zip = self.raw_dir / "recordings.zip"
        recordings_dir = self.raw_dir / "recordings"

        # Fast path: if .tgz already extracted, skip zip handling entirely.
        tgz_matches = _find_site_tgz(recordings_dir, site_id) if recordings_dir.exists() else []
        if not tgz_matches:
            # Need to ensure zip is downloaded and extracted.
            zip_ok = (
                recordings_zip.exists()
                and recordings_zip.stat().st_size == RECORDINGS_ZIP_SIZE
                and not self.args.redownload
            )
            if not zip_ok:
                self.update_status("Downloading recordings.zip (~4.5 GB)")
                _download_file(f"{ZENODO_BASE}/recordings.zip", recordings_zip, "recordings.zip")

            if not recordings_dir.exists() or not any(recordings_dir.glob("*.tgz")):
                self.update_status("Extracting recordings.zip")
                with zipfile.ZipFile(recordings_zip) as zf:
                    zf.extractall(self.raw_dir)
                logger.info(f"Extracted recordings to {recordings_dir}")

            tgz_matches = _find_site_tgz(recordings_dir, site_id)

        # ---- cell_list.csv (for unit metadata) ----
        cell_list_path = self.raw_dir / "cell_list.csv"
        if not cell_list_path.exists():
            _download_file(f"{ZENODO_BASE}/cell_list.csv", cell_list_path, "cell_list.csv")

        # ---- Locate .tgz for this site ----
        if not tgz_matches:
            tgz_matches = _find_site_tgz(recordings_dir, site_id)
        if not tgz_matches:
            raise FileNotFoundError(f"No .tgz for site {site_id} in {recordings_dir}")

        return {
            "tgz_path": tgz_matches[0],
            "site_id": site_id,
            "animal": manifest_item.animal,
            "cell_list_path": cell_list_path,
        }

    # ------------------------------------------------------------------
    # Process
    # ------------------------------------------------------------------

    def process(self, download_output):
        self.processed_dir.mkdir(exist_ok=True, parents=True)

        site_id = download_output["site_id"]
        animal = download_output["animal"]
        tgz_path = Path(download_output["tgz_path"])
        cell_list_path = Path(download_output["cell_list_path"])

        store_path = self.processed_dir / f"{site_id}.h5"
        if store_path.exists() and not self.args.reprocess:
            self.update_status("Skipped Processing")
            return

        # ---- Parse NEMS recording ----
        self.update_status("Parsing recording")
        rec = _parse_nems_recording(tgz_path)

        spike_times_dict = rec["spike_times"]
        stim_specs = rec["stim_specs"]
        epochs = rec["epochs"]
        meta = rec["meta"]
        stim_fs = rec["stim_fs"]

        if not spike_times_dict:
            logger.warning(f"No spikes for {site_id}, skipping")
            return

        # ---- Cell metadata from cell_list.csv ----
        self.update_status("Loading unit metadata")
        cell_df = pd.read_csv(cell_list_path, index_col=0)
        site_cells = cell_df[cell_df["siteid"] == site_id]

        # ---- Spikes & unit metadata ----
        # Skip zero-spike units to keep unit_index contiguous, matching
        # the convention used in all other pipelines.
        all_cell_ids = sorted(spike_times_dict.keys())

        ts_parts: list[np.ndarray] = []
        idx_parts: list[np.ndarray] = []
        unit_meta: list[dict] = []

        for cid in all_cell_ids:
            t = spike_times_dict[cid]
            if len(t) == 0:
                continue

            ts_parts.append(t)
            idx_parts.append(np.full(len(t), len(unit_meta), dtype=np.int64))

            meta: dict = {
                "id": cid,
                "unit_number": len(unit_meta),
                "count": len(t),
                "brain_area": "unknown",
                "layer": "unknown",
                "celltype": "unknown",
                "depth": np.nan,
                "spike_width": np.nan,
                "isolation": np.nan,
                "narrow": False,
            }

            if cid in site_cells.index:
                row = site_cells.loc[cid]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                if pd.notna(row.get("area")):
                    meta["brain_area"] = str(row["area"])
                if pd.notna(row.get("layer")):
                    meta["layer"] = str(row["layer"])
                if pd.notna(row.get("celltype")):
                    meta["celltype"] = str(row["celltype"])
                if pd.notna(row.get("depth")):
                    meta["depth"] = float(row["depth"])
                if pd.notna(row.get("sw")):
                    meta["spike_width"] = float(row["sw"])
                if pd.notna(row.get("iso")):
                    meta["isolation"] = float(row["iso"])
                if pd.notna(row.get("narrow")):
                    meta["narrow"] = bool(row["narrow"])

            unit_meta.append(meta)

        if not ts_parts:
            logger.warning(f"All units empty for {site_id}, skipping")
            return

        timestamps = np.concatenate(ts_parts)
        unit_index = np.concatenate(idx_parts)

        spikes = IrregularTimeSeries(timestamps=timestamps, unit_index=unit_index, domain="auto")
        spikes.sort()

        unit_meta_df = pd.DataFrame(unit_meta)
        units = ArrayDict.from_dataframe(unit_meta_df, unsigned_to_long=True)

        # ---- Spectrogram lookup table ----
        self.update_status("Building spectrogram table")

        spectrograms = None
        name_to_idx: dict[str, int] = {}

        if stim_specs:
            unique_names = sorted(stim_specs.keys())
            name_to_idx = {n: i for i, n in enumerate(unique_names)}

            n_freq = stim_specs[unique_names[0]].shape[0]
            lengths = np.array([stim_specs[n].shape[1] for n in unique_names], dtype=np.int64)
            max_len = int(lengths.max())
            all_same = bool(np.all(lengths == lengths[0]))

            spec_arr = np.zeros((len(unique_names), n_freq, max_len), dtype=np.float32)
            for i, name in enumerate(unique_names):
                s = stim_specs[name]
                spec_arr[i, :, : s.shape[1]] = s

            kw: dict = {"id": np.array(unique_names), "data": spec_arr}
            if not all_same:
                kw["length"] = lengths
            spectrograms = ArrayDict(**kw)

            logger.info(
                f"{site_id}: {len(unique_names)} unique stimuli, "
                f"spectrogram shape per stim = ({n_freq}, {max_len})"
            )

        # ---- Stimulus intervals ----
        self.update_status("Building stimulus intervals")

        stim_rows = epochs[epochs["name"].str.startswith("STIM_")]

        if len(stim_rows) > 0 and name_to_idx:
            stim_starts = stim_rows["start"].values.astype(np.float64)
            stim_ends = stim_rows["end"].values.astype(np.float64)
            stim_names = np.array(stim_rows["name"].tolist(), dtype=str)

            # Drop any epoch without a matching spectrogram
            valid = np.array([n in name_to_idx for n in stim_names])
            if not valid.all():
                n_drop = int((~valid).sum())
                logger.warning(f"{site_id}: dropping {n_drop} epochs without spectrogram")
                stim_starts = stim_starts[valid]
                stim_ends = stim_ends[valid]
                stim_names = stim_names[valid]

            stimuli = Interval(start=stim_starts, end=stim_ends)
            stimuli.spectrogram_index = np.array(
                [name_to_idx[n] for n in stim_names], dtype=np.int64
            )
            stimuli.epoch_name = stim_names
        else:
            stim_starts = np.array([], dtype=np.float64)
            stim_ends = np.array([], dtype=np.float64)
            stim_names = np.array([], dtype=str)
            stimuli = Interval(start=stim_starts, end=stim_ends)

        # ---- Train / val / test split ----
        # Follows the NEMS split_using_epoch_occurrence_counts logic:
        #   1. Count how many times each unique STIM_ epoch appears.
        #   2. Group epoch names by their occurrence count.
        #   3. If >2 distinct counts: split at mean of distinct counts.
        #      If  2 distinct counts: lower → train, higher → test.
        #      If  1 distinct count:  first 80% → train, last 20% → test.
        #   4. Among train stimuli, carve out ~15% for validation
        #      (stimulus-based, so no stimulus leaks across splits).
        self.update_status("Creating splits")

        empty = Interval(
            start=np.array([], dtype=np.float64),
            end=np.array([], dtype=np.float64),
        )

        if len(stim_names) > 0:
            counts = pd.Series(stim_names).value_counts()
            distinct_counts = sorted(counts.unique())

            if len(distinct_counts) == 0:
                test_stim_set: set[str] = set()
            elif len(distinct_counts) == 1:
                # All stimuli have the same rep count — 80/20 by name
                sorted_names = sorted(counts.index)
                cutoff = int(len(sorted_names) * 0.8)
                test_stim_set = set(sorted_names[cutoff:])
            elif len(distinct_counts) == 2:
                # Two groups: lower count → train, higher → test
                test_stim_set = set(counts[counts > distinct_counts[0]].index)
            else:
                # >2 groups: split at mean of the distinct counts
                threshold = float(np.mean(distinct_counts))
                test_stim_set = set(counts[counts > threshold].index)

            # Test = all intervals for high-rep stimuli
            test_mask = np.array([n in test_stim_set for n in stim_names])

            # Among non-test stimuli, carve out ~15% for validation
            # (split by unique stimulus name to prevent leakage).
            est_mask = ~test_mask
            est_names = stim_names[est_mask]
            unique_est = np.array(sorted(set(est_names)))

            rng = np.random.default_rng(SPLIT_SEED)
            rng.shuffle(unique_est)
            n_val = max(1, int(round(VALID_FRACTION * len(unique_est))))
            val_stim_set = set(unique_est[:n_val])

            val_mask = np.array([n in val_stim_set for n in stim_names])
            train_mask = est_mask & ~val_mask

            train_domain = (
                Interval(start=stim_starts[train_mask], end=stim_ends[train_mask])
                if train_mask.any()
                else empty
            )
            valid_domain = (
                Interval(start=stim_starts[val_mask], end=stim_ends[val_mask])
                if val_mask.any()
                else empty
            )
            test_domain = (
                Interval(start=stim_starts[test_mask], end=stim_ends[test_mask])
                if test_mask.any()
                else empty
            )

            logger.info(
                f"{site_id}: train={train_mask.sum()} intervals, "
                f"val={val_mask.sum()} intervals, "
                f"test={test_mask.sum()} intervals "
                f"({len(test_stim_set)} test stimuli, "
                f"threshold: mean of distinct counts {distinct_counts})"
            )
        else:
            train_domain = empty
            valid_domain = empty
            test_domain = empty

        # ---- Descriptions ----
        brainset_description = BrainsetDescription(
            id="wingert_auditory_cortex_2026",
            origin_version="1.0",
            derived_version="1.0.0",
            source="https://zenodo.org/records/18331549",
            description=(
                "High-density recordings from ferret auditory cortex (A1/PEG) "
                "during passive listening to natural sounds. Sorted single-unit "
                "spike times and 32-channel cochleagram spectrograms (200-20 kHz, "
                "100 Hz). Wingert, Parida, Norman-Haignere & David, Nat Neurosci 2026."
            ),
        )

        subject = SubjectDescription(
            id=animal,
            species=Species.UNKNOWN,  # ferret not in taxonomy
            sex=Sex.UNKNOWN,
        )

        # No per-session date available; use dataset publication date.
        session_desc = SessionDescription(
            id=site_id,
            recording_date=datetime.datetime(2026, 1, 21),
        )

        rec_tech = ANIMAL_RECORDING_TECH.get(animal, RecordingTech.NEUROPIXELS_SPIKES)
        device_desc = DeviceDescription(id=f"{animal}_probe", recording_tech=rec_tech)

        # ---- Assemble & save ----
        self.update_status("Saving")

        data_kw: dict = {
            "brainset": brainset_description,
            "subject": subject,
            "session": session_desc,
            "device": device_desc,
            "spikes": spikes,
            "units": units,
            "stimuli": stimuli,
            "domain": spikes.domain,
        }
        if spectrograms is not None:
            data_kw["spectrograms"] = spectrograms

        data = Data(**data_kw)

        # Spectrogram metadata (scalar attributes)
        data.spectrogram_fs = float(stim_fs)
        data.spectrogram_f_min = float(meta.get("f_min", 200))
        data.spectrogram_f_max = float(meta.get("f_max", 20000))
        data.recording_total_bins = int(rec["recording_total_bins"])

        data.set_train_domain(train_domain)
        data.set_valid_domain(valid_domain)
        data.set_test_domain(test_domain)

        with h5py.File(store_path, "w") as fh:
            data.to_hdf5(fh, serialize_fn_map=serialize_fn_map)

        logger.info(
            f"Saved {site_id} → {store_path.name} "
            f"({store_path.stat().st_size / 1e6:.1f} MB, "
            f"{len(units)} units, {len(stim_specs)} unique stimuli)"
        )
