# /// brainset-pipeline
# python-version = "3.11"
# dependencies = [
#   "h5py",
#   "requests",
#   "scipy",
#   "Pillow",
# ]
# ///

import datetime
import io
import logging
import zipfile
from argparse import ArgumentParser, Namespace
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
from PIL import Image
from temporaldata import ArrayDict, Data, Interval, IrregularTimeSeries

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FIGSHARE_ARTICLE_ID = "30500090"

# Animal mapping from the dataset readme (Ext. Data Table 1).
# Key: session date string (YYMMDD), Value: (animal_code, animal_name)
# WE = Wakko, PE = Pepe, RA = Raul
ANIMAL_MAP: dict[str, tuple[str, str]] = {}
for _d in ["190923", "190924", "190925", "190926", "190927", "190928", "190929"]:
    ANIMAL_MAP[_d] = ("WE", "Wakko")
for _d in [
    "201014",
    "201015",  # on Figshare but not in readme; PE by date range
    "201016",
    "201017",
    "201018",
    "201019",
    "201020",
    "201021",
    "201022",
    "201023",
    "201024",
    "201025",
    "210224",
    "210225",
    "210226",
    "210301",
    "210302",
    "210303",
    "210304",
    "210305",
    "210308",
    "210309",
    "210310",
    "210315",
    "210316",
    "210322",
    "210323",
    "210324",
    "210325",
    "210326",
]:
    ANIMAL_MAP[_d] = ("PE", "Pepe")
for _d in [
    "210620",
    "210621",
    "211008",
    "211013",
    "211014",
    "211015",
    "211018",
    "211022",
    "211025",
    "211026",
    "211027",
    "211028",
    "211103",
]:
    ANIMAL_MAP[_d] = ("RA", "Raul")

# Original session-level split assignments from the dataset readme and
# class_neural_data_train.py / class_neural_data_test.py in the authors' repo.
# "train" = model training, "val" = hyperparameter validation, "test" = held-out evaluation.
# These are not used for the actual train/val/test domains (we split by image instead),
# but we store them on each Data object as `author_split` for downstream use.
AUTHOR_SESSION_SPLIT: dict[str, str] = {}
for _d in [
    "190924",
    "190925",
    "190926",
    "190927",
    "190928",
    "190929",
    "201014",
    "201015",  # on Figshare but not in readme; train by default
    "201016",
    "201017",
    "201018",
    "201019",
    "201020",
    "201021",
    "201022",
    "201023",
    "201024",
    "210226",
    "210301",
    "210302",
    "210303",
    "210304",
    "210305",
    "210308",
    "210309",
    "210310",
    "210315",
    "210316",
    "210322",
    "210323",
    "210324",
    "210326",
    "210620",
    "210621",
    "211008",
    "211013",
    "211014",
    "211015",
    "211018",
    "211025",
    "211026",
    "211027",
    "211028",
    "211103",
]:
    AUTHOR_SESSION_SPLIT[_d] = "train"
AUTHOR_SESSION_SPLIT["210224"] = "val"
for _d in ["190923", "201025", "210225", "210325", "211022"]:
    AUTHOR_SESSION_SPLIT[_d] = "test"

# Image-based split ratios (applied per session).
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train, val, test
SPLIT_SEED = 42

# GitHub base URL for image zips
IMAGES_BASE = "https://github.com/cowleygroup/V4_compact_models/raw/main/data_V4_responses/images"

parser = ArgumentParser()
parser.add_argument("--redownload", action="store_true")
parser.add_argument("--reprocess", action="store_true")


def _date_str_to_datetime(date_yymmdd: str) -> datetime.datetime:
    """Convert YYMMDD string to datetime."""
    return datetime.datetime.strptime(f"20{date_yymmdd}", "%Y%m%d")


class Pipeline(BrainsetPipeline):
    """Export pipeline for the Cowley V4 compact models dataset.

    Utah array recordings from macaque V4 during rapid serial visual
    presentation of natural images. Contains spike times and stimulus images.
    Units correspond to electrode channels (one unit per channel, after spike sorting).
    """

    brainset_id = "cowley_v4_compact_2024"
    parser = parser

    @classmethod
    def get_manifest(cls, raw_dir: Path, args: Namespace | None) -> pd.DataFrame:
        """Build manifest from Figshare API, one row per .mat file."""
        url = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE_ID}/files?page_size=100"
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        files = resp.json()

        rows = []
        for f in files:
            name = f["name"]
            if not name.endswith(".mat"):
                continue
            date_str = name.replace("_data.mat", "")
            animal_code, animal_name = ANIMAL_MAP.get(date_str, ("UNK", "Unknown"))
            rows.append(
                {
                    "session_id": date_str,
                    "filename": name,
                    "download_url": f["download_url"],
                    "date_str": date_str,
                    "animal_code": animal_code,
                    "animal_name": animal_name,
                    "author_split": AUTHOR_SESSION_SPLIT.get(date_str, "train"),
                }
            )

        return pd.DataFrame(rows).set_index("session_id")

    def download(self, manifest_item):
        """Download .mat file from Figshare and image zip from GitHub."""
        self.update_status("DOWNLOADING")
        self.raw_dir.mkdir(exist_ok=True, parents=True)

        date_str = manifest_item.date_str
        mat_path = self.raw_dir / manifest_item.filename

        # --- Download .mat file ---
        if not mat_path.exists() or self.args.redownload:
            download_url = manifest_item.download_url
            # Convert ndownloader URL to Figshare API endpoint
            if "/ndownloader/files/" in download_url:
                file_id = download_url.split("/ndownloader/files/")[-1].split("?")[0]
                download_url = f"https://api.figshare.com/v2/file/download/{file_id}"

            self.update_status(f"Downloading {manifest_item.filename}")
            resp = requests.get(download_url, stream=True, timeout=7200)
            resp.raise_for_status()
            with open(mat_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                    f.write(chunk)
            logger.info(f"Downloaded {mat_path} ({mat_path.stat().st_size / 1e6:.0f} MB)")
        else:
            logger.info(f"Already exists: {mat_path}")

        # --- Download image zip from GitHub ---
        images_dir = self.raw_dir / "images" / date_str
        zip_url = f"{IMAGES_BASE}/images_{date_str}.zip"

        if not images_dir.exists() or self.args.redownload:
            self.update_status(f"Downloading images for {date_str}")
            try:
                resp = requests.get(zip_url, stream=True, timeout=7200, allow_redirects=True)
                resp.raise_for_status()
                zip_data = io.BytesIO(resp.content)
                images_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(zip_data) as zf:
                    for member in zf.namelist():
                        if member.endswith(".jpg") or member.endswith(".jpeg"):
                            # Extract to flat directory, stripping any subdirs
                            fname = Path(member).name
                            with zf.open(member) as src, open(images_dir / fname, "wb") as dst:
                                dst.write(src.read())
                logger.info(f"Extracted images to {images_dir}")
            except requests.exceptions.HTTPError:
                logger.warning(f"No image zip available for {date_str} (HTTP error)")
                images_dir = None
            except Exception as e:
                logger.warning(f"Failed to download images for {date_str}: {e}")
                images_dir = None
        else:
            logger.info(f"Images already exist: {images_dir}")

        return {
            "mat_path": mat_path,
            "images_dir": images_dir,
            "date_str": date_str,
            "animal_code": manifest_item.animal_code,
            "animal_name": manifest_item.animal_name,
            "author_split": manifest_item.author_split,
        }

    def process(self, download_output):
        """Process a single session .mat file into brainsets format."""
        self.processed_dir.mkdir(exist_ok=True, parents=True)

        date_str = download_output["date_str"]
        animal_code = download_output["animal_code"]
        animal_name = download_output["animal_name"]
        mat_path = download_output["mat_path"]
        images_dir = download_output["images_dir"]
        author_split = download_output["author_split"]

        session_id = f"{animal_code}_{date_str}"
        store_path = self.processed_dir / f"{session_id}.h5"

        if store_path.exists() and not self.args.reprocess:
            self.update_status("Skipped Processing")
            return

        # --- Load .mat ---
        self.update_status("Loading .mat file")
        mat = scipy.io.loadmat(str(mat_path), squeeze_me=True)
        trials = mat["data"]

        # --- Filter correct trials ---
        self.update_status("Filtering correct trials")
        correct_mask = np.array(
            [
                float(t["result"]) == 150 if not np.isnan(float(t["result"])) else False
                for t in trials
            ]
        )
        correct_trials = trials[correct_mask]
        n_correct = len(correct_trials)

        if n_correct == 0:
            logger.warning(f"No correct trials for {session_id}, skipping")
            return

        logger.info(f"{session_id}: {n_correct}/{len(trials)} correct trials")

        # --- Extract spikes using session-absolute times ---
        self.update_status("Extracting spikes")

        # Collect all unique channel IDs across correct trials
        all_channels = set()
        for t in correct_trials:
            st = np.atleast_2d(t["spikeTimes"])
            if st.size > 0:
                all_channels.update(st[:, 0].astype(int).tolist())
        channel_ids = np.array(sorted(all_channels))
        channel_to_idx = {ch: i for i, ch in enumerate(channel_ids)}

        timestamps_list: list[np.ndarray] = []
        unit_index_list: list[np.ndarray] = []

        for t in correct_trials:
            st = np.atleast_2d(t["spikeTimes"])
            if st.size == 0:
                continue
            # Column 2: session-absolute time
            spike_times = st[:, 2].astype(np.float64)
            spike_channels = st[:, 0].astype(int)
            spike_unit_idx = np.array([channel_to_idx[ch] for ch in spike_channels])

            timestamps_list.append(spike_times)
            unit_index_list.append(spike_unit_idx)

        if not timestamps_list:
            logger.warning(f"No spikes found for {session_id}, skipping")
            return

        timestamps = np.concatenate(timestamps_list)
        unit_index = np.concatenate(unit_index_list)

        spikes = IrregularTimeSeries(
            timestamps=timestamps,
            unit_index=unit_index,
            domain="auto",
        )
        spikes.sort()

        # --- Units metadata ---
        spike_counts = np.bincount(unit_index, minlength=len(channel_ids))
        units = ArrayDict(
            id=np.array([f"ch{ch:04d}" for ch in channel_ids]),
            unit_number=np.arange(len(channel_ids)),
            count=spike_counts,
            brain_area=np.array(["V4"] * len(channel_ids)),
        )

        # --- Extract stimulus intervals ---
        self.update_status("Extracting stimulus intervals")

        stim_starts = []
        stim_ends = []
        stim_image_ids = []

        for t in correct_trials:
            tc = np.atleast_2d(t["trialCodes"])
            if tc.size == 0:
                continue

            codes = tc[:, 0].astype(int)
            times_abs = tc[:, 2].astype(np.float64)

            # Get image IDs for this trial. Two formats exist:
            # 1. PE/RA sessions: IMAGE_ID codes (>10000) in trialCodes
            # 2. WE sessions: image1_id..image6_id fields in trialParams
            has_image_codes = np.any(codes > 10000)
            tp = t["trialParams"]

            if has_image_codes:
                # Extract image IDs from trialCodes in order of appearance
                trial_image_ids = (codes[codes > 10000] - 10000).tolist()
            elif "image1_id" in (tp.dtype.names or []):
                trial_image_ids = [int(tp[f"image{i}_id"]) for i in range(1, 7)]
            elif "imageList" in (tp.dtype.names or []):
                trial_image_ids = np.atleast_1d(tp["imageList"]).flatten().tolist()
            else:
                continue

            # Parse STIM_ON(10)/STIM_OFF(40) pairs
            stim_idx = 0
            i = 0
            while i < len(codes):
                if codes[i] == 10:  # STIM_ON
                    onset = times_abs[i]
                    offset = None
                    j = i + 1
                    while j < len(codes):
                        if codes[j] == 40:  # STIM_OFF
                            offset = times_abs[j]
                            break
                        j += 1
                    if offset is not None and stim_idx < len(trial_image_ids):
                        stim_starts.append(onset)
                        stim_ends.append(offset)
                        stim_image_ids.append(int(trial_image_ids[stim_idx]))
                        stim_idx += 1
                    i = j + 1
                else:
                    i += 1

        if not stim_starts:
            logger.warning(f"No stimulus events found for {session_id}")

        stim_starts = np.array(stim_starts, dtype=np.float64)
        stim_ends = np.array(stim_ends, dtype=np.float64)
        stim_image_ids = np.array(stim_image_ids, dtype=np.int64)

        logger.info(
            f"{session_id}: {len(stim_starts)} stimulus presentations, "
            f"{len(np.unique(stim_image_ids))} unique images"
        )

        # --- Load images ---
        image_data = {}
        if images_dir is not None and Path(images_dir).exists():
            self.update_status("Loading images")
            unique_image_ids = np.unique(stim_image_ids)
            for img_id in unique_image_ids:
                # Images are 1-indexed: image0001.jpg, image0002.jpg, ...
                img_path = Path(images_dir) / f"image{img_id:04d}.jpg"
                if img_path.exists():
                    img = Image.open(img_path)
                    image_data[int(img_id)] = np.array(img, dtype=np.uint8)
                else:
                    logger.warning(f"Image not found: {img_path}")

        # Build image lookup table
        if image_data:
            sorted_ids = sorted(image_data.keys())
            image_id_to_lookup_idx = {img_id: i for i, img_id in enumerate(sorted_ids)}
            pixels = np.stack([image_data[img_id] for img_id in sorted_ids])

            images = ArrayDict(
                id=np.array([f"img{img_id:06d}" for img_id in sorted_ids]),
                pixels=pixels,
            )

            # Map stimulus image_ids to lookup indices
            stim_image_index = np.array(
                [image_id_to_lookup_idx[int(iid)] for iid in stim_image_ids],
                dtype=np.int64,
            )
        else:
            images = None
            stim_image_index = stim_image_ids  # fallback: raw IDs

        # Build stimulus Interval
        stimuli = Interval(start=stim_starts, end=stim_ends)
        stimuli.image_index = stim_image_index
        stimuli.image_id = stim_image_ids

        # --- Image-based train/val/test split ---
        # Split unique images 70/15/15, then assign each stimulus interval
        # to the split of its image. This ensures no image leaks across splits.
        self.update_status("Creating image-based splits")

        unique_images = np.unique(stim_image_ids)
        rng = np.random.default_rng(SPLIT_SEED)
        rng.shuffle(unique_images)

        n_images = len(unique_images)
        n_train = int(np.round(SPLIT_RATIOS[0] * n_images))
        n_val = int(np.round(SPLIT_RATIOS[1] * n_images))
        # Remainder goes to test to avoid off-by-one from rounding
        train_images = set(unique_images[:n_train].tolist())
        val_images = set(unique_images[n_train : n_train + n_val].tolist())
        test_images = set(unique_images[n_train + n_val :].tolist())

        logger.info(
            f"{session_id}: image split — "
            f"{len(train_images)} train, {len(val_images)} val, {len(test_images)} test"
        )

        # Partition stimulus intervals by image split
        train_mask = np.array([int(iid) in train_images for iid in stim_image_ids])
        val_mask = np.array([int(iid) in val_images for iid in stim_image_ids])
        test_mask = np.array([int(iid) in test_images for iid in stim_image_ids])

        empty = Interval(
            start=np.array([], dtype=np.float64),
            end=np.array([], dtype=np.float64),
        )
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

        # --- Descriptions ---
        brainset_description = BrainsetDescription(
            id="cowley_v4_compact_2024",
            origin_version="1.0.0",
            derived_version="1.0.0",
            source="https://kilthub.cmu.edu/articles/dataset/30500090",
            description=(
                "Utah array recordings from macaque V4 during rapid serial visual "
                "presentation (RSVP) of natural images. Contains sorted single-unit "
                "spike times and 112x112 stimulus images."
            ),
        )

        subject = SubjectDescription(
            id=animal_name,
            species=Species.MACACA_MULATTA,
            sex=Sex.UNKNOWN,
        )

        session_description = SessionDescription(
            id=session_id,
            recording_date=_date_str_to_datetime(date_str),
        )

        device_description = DeviceDescription(
            id=f"{animal_code}_utah_array",
            recording_tech=RecordingTech.UTAH_ARRAY_SPIKES,
        )

        # --- Assemble and save ---
        self.update_status("Assembling Data object")

        data_kwargs = {
            "brainset": brainset_description,
            "subject": subject,
            "session": session_description,
            "device": device_description,
            "spikes": spikes,
            "units": units,
            "stimuli": stimuli,
            "domain": spikes.domain,
        }
        if images is not None:
            data_kwargs["images"] = images

        data = Data(**data_kwargs)

        # Store the authors' original session-level split designation
        data.author_split = author_split

        data.set_train_domain(train_domain)
        data.set_valid_domain(valid_domain)
        data.set_test_domain(test_domain)

        self.update_status("Storing")
        with h5py.File(store_path, "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)

        logger.info(
            f"Saved {session_id} to {store_path} ({store_path.stat().st_size / 1e6:.0f} MB)"
        )
