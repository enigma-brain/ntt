# /// brainset-pipeline
# python-version = "3.11"
# dependencies = [
#   "h5py",
#   "requests",
#   "scipy",
# ]
# ///

import datetime
import logging
from argparse import ArgumentParser
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import requests
import scipy.io
from brainsets import serialize_fn_map
from brainsets.descriptions import (
    BrainsetDescription,
    DeviceDescription,
    SessionDescription,
    SubjectDescription,
)
from brainsets.pipeline import BrainsetPipeline
from brainsets.taxonomy import RecordingTech, Sex, Species
from temporaldata import ArrayDict, Data, Interval, RegularTimeSeries

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GIN_BASE = "https://gin.g-node.org/paolo_papale/TVSD/raw/master"

# Recording dates per monkey (day_index 1-4 maps to these dates)
MONKEY_DATES = {
    "monkeyN": ["2022-01-11", "2022-01-12", "2022-01-13", "2022-01-14"],
    "monkeyF": ["2024-01-12", "2024-01-15", "2024-01-16", "2024-01-18"],
}

# Brain region assignments in hardware channel order (0-indexed ranges).
# The mapping permutation is applied to both data and these labels.
CHANNEL_REGIONS = {
    "monkeyN": [("V1", 0, 512), ("V4", 512, 768), ("IT", 768, 1024)],
    "monkeyF": [("V1", 0, 512), ("IT", 512, 832), ("V4", 832, 1024)],
}

parser = ArgumentParser()
parser.add_argument("--redownload", action="store_true")
parser.add_argument("--reprocess", action="store_true")


def download_file(url: str, dest: Path, redownload: bool = False) -> Path:
    """Download a file from GIN if not already present."""
    if dest.exists() and not redownload:
        logger.info(f"Already exists: {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {url} -> {dest}")
    resp = requests.get(url, stream=True, timeout=7200)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
            f.write(chunk)
    logger.info(f"Downloaded {dest} ({dest.stat().st_size / 1e9:.1f} GB)")
    return dest


def read_h5_strings(h5file: h5py.File, dataset: h5py.Dataset) -> list[str]:
    """Dereference HDF5 object references to extract MATLAB cell strings."""
    result = []
    for ref in dataset[:].flat:
        obj = h5file[ref]
        s = "".join(chr(c) for c in obj[:].flat)
        result.append(s)
    return result


def build_region_labels(monkey: str, mapping: np.ndarray) -> np.ndarray:
    """Assign brain region labels per channel, then apply mapping."""
    regions_hw = np.empty(1024, dtype="U3")
    for region, start, end in CHANNEL_REGIONS[monkey]:
        regions_hw[start:end] = region
    return regions_hw[mapping]


class Pipeline(BrainsetPipeline):
    brainset_id = "papale_tvsd_2024"
    parser = parser

    @classmethod
    def get_manifest(cls, raw_dir, args) -> pd.DataFrame:
        rows = []
        for monkey in ["monkeyN", "monkeyF"]:
            for day_idx, date_str in enumerate(MONKEY_DATES[monkey], start=1):
                session_id = f"{monkey}_day{day_idx}"
                rows.append(
                    {
                        "id": session_id,
                        "session_id": session_id,
                        "monkey": monkey,
                        "day_index": day_idx,
                        "date": date_str,
                    }
                )
        return pd.DataFrame(rows).set_index("id")

    def download(self, manifest_item):
        self.update_status("DOWNLOADING")
        monkey = manifest_item.monkey
        redownload = self.args.redownload
        monkey_dir = self.raw_dir / monkey

        trials_path = download_file(
            f"{GIN_BASE}/{monkey}/THINGS_MUA_trials.mat",
            monkey_dir / "THINGS_MUA_trials.mat",
            redownload=redownload,
        )
        things_imgs_path = download_file(
            f"{GIN_BASE}/{monkey}/_logs/things_imgs.mat",
            monkey_dir / "things_imgs.mat",
            redownload=redownload,
        )
        mapping_path = download_file(
            f"{GIN_BASE}/{monkey}/_logs/1024chns_mapping_20220105.mat",
            monkey_dir / "mapping.mat",
            redownload=redownload,
        )
        normmua_path = download_file(
            f"{GIN_BASE}/{monkey}/THINGS_normMUA.mat",
            monkey_dir / "THINGS_normMUA.mat",
            redownload=redownload,
        )

        return {
            "trials_path": trials_path,
            "things_imgs_path": things_imgs_path,
            "mapping_path": mapping_path,
            "normmua_path": normmua_path,
            "monkey": monkey,
            "day_index": manifest_item.day_index,
            "date": manifest_item.date,
            "session_id": manifest_item.session_id,
        }

    def process(self, download_output):
        monkey = download_output["monkey"]
        day_index = download_output["day_index"]
        date_str = download_output["date"]
        session_id = download_output["session_id"]

        self.processed_dir.mkdir(exist_ok=True, parents=True)
        store_path = self.processed_dir / f"{session_id}.h5"
        if store_path.exists() and not self.args.reprocess:
            self.update_status("Skipped Processing")
            return

        # --- Load channel mapping (regular .mat v5, use scipy) ---
        self.update_status("Loading mapping")
        mapping = (
            scipy.io.loadmat(download_output["mapping_path"])["mapping"].flatten() - 1
        )  # convert to 0-indexed

        # --- Load trial data (HDF5-based .mat v7.3) ---
        self.update_status("Loading trial data")
        with h5py.File(download_output["trials_path"], "r") as f:
            # h5py reverses MATLAB dimensions:
            #   MATLAB ALLMUA [1024, n_trials, 301] -> h5py [301, n_trials, 1024]
            #   MATLAB ALLMAT [n_trials, 6]         -> h5py [6, n_trials]
            allmat = f["ALLMAT"][:].T  # -> [n_trials, 6]
            tb = f["tb"][:].flatten()  # time base in ms

            # Filter to this recording day (column 5, 1-indexed day)
            day_mask = allmat[:, 5].astype(int) == day_index
            day_indices = np.sort(np.where(day_mask)[0])

            if len(day_indices) == 0:
                logger.warning(f"No trials found for {session_id}, skipping")
                return

            # Read trials in chunks to limit memory from h5py decompression
            # buffers. Convert to float32 immediately (source is float64).
            n_day_trials = len(day_indices)
            n_tp = f["ALLMUA"].shape[0]
            n_ch = f["ALLMUA"].shape[2]
            self.update_status(f"Loading {n_day_trials} trials from MUA array")
            allmua_day = np.empty((n_tp, n_day_trials, n_ch), dtype=np.float32)

            chunk_size = 500
            ds = f["ALLMUA"]
            for i in range(0, n_day_trials, chunk_size):
                chunk_idx = day_indices[i : i + chunk_size]
                allmua_day[:, i : i + len(chunk_idx), :] = ds[:, chunk_idx, :]

        allmat_day = allmat[day_mask]  # [n_day_trials, 6]

        # Apply channel mapping to data
        allmua_day = allmua_day[:, :, mapping]  # reorder channel dimension

        n_timepoints, n_trials, n_channels = allmua_day.shape
        logger.info(
            f"{session_id}: {n_trials} trials, {n_channels} channels, "
            f"{n_timepoints} timepoints, tb=[{tb[0]:.0f}, {tb[-1]:.0f}] ms"
        )

        # --- Load stimulus image info ---
        self.update_status("Loading stimulus info")
        with h5py.File(download_output["things_imgs_path"], "r") as f:
            train_classes = read_h5_strings(f, f["train_imgs/class"])
            test_classes = read_h5_strings(f, f["test_imgs/class"])

        # --- Load quality metrics (already in mapped channel order) ---
        self.update_status("Loading quality metrics")
        with h5py.File(download_output["normmua_path"], "r") as f:
            snr_max = f["SNR_max"][:].flatten().astype(np.float32)
            oracle = f["oracle"][:].flatten().astype(np.float32)
            # Per-day SNR: h5py shape [n_days, 1024]
            snr_day = f["SNR"][day_index - 1, :].astype(np.float32)

        # --- Build descriptions ---
        self.update_status("Building data object")

        brainset_description = BrainsetDescription(
            id="papale_tvsd_2024",
            origin_version="1.0.0",
            derived_version="1.0.0",
            source="https://gin.g-node.org/paolo_papale/TVSD",
            description=(
                "THINGS Ventral-Stream Spiking Dataset (TVSD). "
                "Multi-unit activity from V1, V4, and IT in macaque monkeys "
                "viewing ~22k natural images from the THINGS database."
            ),
        )

        subject = SubjectDescription(
            id=monkey,
            species=Species.MACACA_MULATTA,
            sex=Sex.UNKNOWN,
        )

        session_description = SessionDescription(
            id=session_id,
            recording_date=datetime.datetime.strptime(date_str, "%Y-%m-%d"),
        )

        device_description = DeviceDescription(
            id=f"{monkey}_blackrock_array",
            recording_tech=RecordingTech.UTAH_ARRAY,
        )

        # --- Build continuous timeline from concatenated trials ---
        self.update_status("Building timeline")

        # Reshape: [n_timepoints, n_trials, 1024] -> [n_trials * n_timepoints, 1024]
        mua_data = np.transpose(allmua_day, (1, 0, 2))  # [n_trials, n_timepoints, 1024]
        mua_data = mua_data.reshape(-1, n_channels).astype(np.float32)

        sampling_rate = 1000.0  # 1 kHz (1 ms bins)

        mua = RegularTimeSeries(
            mua=mua_data,
            sampling_rate=sampling_rate,
            domain="auto",
            domain_start=0.0,
        )

        # --- Build stimulus intervals ---
        self.update_status("Building stimulus intervals")

        trial_duration = n_timepoints / sampling_rate  # seconds per trial
        pre_stimulus = abs(tb[0]) / 1000.0  # baseline duration in seconds

        trial_starts = np.arange(n_trials, dtype=np.float64) * trial_duration
        trial_ends = trial_starts + trial_duration
        stimulus_onsets = trial_starts + pre_stimulus

        # ALLMAT columns: [trial_idx, train_idx, test_idx, repetition, ncount, day]
        train_idx = allmat_day[:, 1].astype(int)  # >0 for train trials
        test_idx = allmat_day[:, 2].astype(int)  # >0 for test trials
        is_test = test_idx > 0
        repetition = allmat_day[:, 3].astype(np.int64)

        # Map trial indices to THINGS concept class names
        image_class = np.empty(n_trials, dtype=object)
        image_index = np.zeros(n_trials, dtype=np.int64)
        for i in range(n_trials):
            if is_test[i]:
                image_class[i] = test_classes[test_idx[i] - 1]  # 1-indexed to 0-indexed
                image_index[i] = test_idx[i]
            else:
                image_class[i] = train_classes[train_idx[i] - 1]
                image_index[i] = train_idx[i]
        image_class = image_class.astype(str)

        stimuli = Interval(start=trial_starts, end=trial_ends)
        stimuli.stimulus_onset = stimulus_onsets
        stimuli.register_timekey("stimulus_onset")
        stimuli.image_class = image_class
        stimuli.image_index = image_index
        stimuli.is_test = is_test
        stimuli.repetition = repetition

        # --- Build units metadata ---
        self.update_status("Building unit metadata")

        regions = build_region_labels(monkey, mapping)

        units = ArrayDict(
            id=np.array([f"ch{i:04d}" for i in range(n_channels)]),
            unit_number=np.arange(n_channels),
            count=mua_data.sum(axis=0).astype(int),
            brain_area=regions,
            snr_max=snr_max,
            oracle=oracle,
            snr_day=snr_day,
        )

        # --- Build train/valid/test split ---
        self.update_status("Creating splits")

        # Test trials use the dataset's designated test images (shown multiple times)
        # Training trials are split 85/15 into train/valid
        train_trial_indices = np.where(~is_test)[0]
        test_trial_indices = np.where(is_test)[0]

        rng = np.random.RandomState(42)
        shuffled = train_trial_indices.copy()
        rng.shuffle(shuffled)
        n_valid = max(1, int(len(shuffled) * 0.15))
        valid_indices = np.sort(shuffled[:n_valid])
        train_indices = np.sort(shuffled[n_valid:])

        def indices_to_interval(idxs):
            return Interval(start=trial_starts[idxs], end=trial_ends[idxs])

        train_domain = indices_to_interval(train_indices)
        valid_domain = indices_to_interval(valid_indices)
        test_domain = indices_to_interval(test_trial_indices)

        # --- Assemble and save ---
        data = Data(
            brainset=brainset_description,
            subject=subject,
            session=session_description,
            device=device_description,
            mua=mua,
            units=units,
            stimuli=stimuli,
            domain=mua.domain,
        )

        data.set_train_domain(train_domain)
        data.set_valid_domain(valid_domain)
        data.set_test_domain(test_domain)

        self.update_status("Storing")
        with h5py.File(store_path, "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)

        logger.info(
            f"Saved {session_id} to {store_path} ({store_path.stat().st_size / 1e6:.0f} MB)"
        )
