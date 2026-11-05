# /// brainset-pipeline
# python-version = "3.11"
# dependencies = [
#   "dandi==0.74.0",
# ]
# ///

import datetime
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from brainsets import serialize_fn_map
from brainsets.descriptions import (
    BrainsetDescription,
    DeviceDescription,
    SessionDescription,
    SubjectDescription,
)
from brainsets.pipeline import BrainsetPipeline
from brainsets.taxonomy import RecordingTech, Sex, Species
from brainsets.utils.dandi_utils import download_file, get_nwb_asset_list
from pynwb import NWBHDF5IO
from temporaldata import ArrayDict, Data, Interval, IrregularTimeSeries

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Threshold crossings sampling rate in Hz.
TX_SAMPLING_RATE = 30_000.0

parser = ArgumentParser()
parser.add_argument("--redownload", action="store_true")
parser.add_argument("--reprocess", action="store_true")
parser.add_argument("--extract-behaviour", action="store_true")
parser.set_defaults(extract_behaviour=False)


def _parse_asset_path(path: str) -> dict:
    """Parse a DANDI asset path into subject, split type, and date.

    Paths look like:
        sub-Finch-z-r12r13-21-held-in-calib/
            sub-Finch-z-r12r13-21-held-in-calib_ses-20210626.nwb
    """
    fname = path.split("/")[-1]  # sub-..._ses-YYYYMMDD.nwb
    dir_name = path.split("/")[0]  # sub-...-held-in-calib

    # Extract date from filename
    date = fname.split("_ses-")[1].replace(".nwb", "")

    # Extract subject and split type from directory name.
    # Directory: sub-Finch-z-r12r13-21-held-in-calib
    # Remove 'sub-' prefix
    dir_body = dir_name.replace("sub-", "", 1)

    # Split types are: held-in-calib, held-in-minival, held-out-calib
    for split in ("held-in-calib", "held-in-minival", "held-out-calib"):
        if dir_body.endswith(split):
            subject = dir_body[: -len(split) - 1]  # strip trailing '-{split}'
            return {"subject": subject, "split_type": split, "date": date}

    raise ValueError(f"Cannot parse split type from path: {path}")


class Pipeline(BrainsetPipeline):
    """FALCON Benchmark B1: zebra finch RA Neuropixels recordings during birdsong.

    Neuropixels recordings from the robust nucleus of the arcopallium (RA) in
    a zebra finch during singing.  Neural data is stored as threshold crossings
    (binary, 30 kHz) across 85 channels.  Behavioral data includes raw audio
    vocalizations (25 kHz) and per-trial spectrograms.

    Source: https://dandiarchive.org/dandiset/001046
    Paper:  https://arxiv.org/abs/2407.06244
    """

    brainset_id = "tostado_marcos_birdsong_2024"
    dandiset_id = "DANDI:001046/draft"
    parser = parser

    @classmethod
    def get_manifest(cls, raw_dir: Path, args: Namespace | None) -> pd.DataFrame:
        asset_list = get_nwb_asset_list(cls.dandiset_id)
        manifest_list = [{"path": x.path, "url": x.download_url} for x in asset_list]

        for m in manifest_list:
            parsed = _parse_asset_path(m["path"])
            subject = parsed["subject"].lower()
            split_type = parsed["split_type"].replace("-", "_")
            date = parsed["date"]
            m["session_id"] = f"{subject}_{split_type}_{date}"
            m["split_type"] = parsed["split_type"]

        manifest = pd.DataFrame(manifest_list).set_index("session_id")
        return manifest

    def download(self, manifest_item):
        self.update_status("DOWNLOADING")
        self.raw_dir.mkdir(exist_ok=True, parents=True)
        fpath = download_file(
            manifest_item.path,
            manifest_item.url,
            self.raw_dir,
            overwrite=self.args.redownload,
        )
        return {
            "fpath": fpath,
            "split_type": manifest_item.split_type.replace("-", "_"),
        }

    def process(self, download_output):
        fpath = download_output["fpath"]
        split_type = download_output["split_type"]

        self.update_status("Loading NWB")
        io = NWBHDF5IO(fpath, "r")
        nwbfile = io.read()

        self.processed_dir.mkdir(exist_ok=True, parents=True)

        brainset_description = BrainsetDescription(
            id="tostado_marcos_birdsong_2024",
            origin_version="dandi/001046/draft",
            derived_version="1.0.0",
            source="https://dandiarchive.org/dandiset/001046",
            description=(
                "Neuropixels recordings from the robust nucleus of the arcopallium (RA) "
                "in a zebra finch during singing. Neural data consists of threshold "
                "crossings from 85 channels at 30 kHz. Behavioral data includes raw "
                "audio vocalizations and per-trial spectrograms. Part of the FALCON "
                "Benchmark B1 dataset (Tostado-Marcos, Arneodo, Gentner & Gilja)."
            ),
        )

        self.update_status("Extracting Metadata")
        # Extract subject metadata manually since species (zebra finch) is not
        # in the standard taxonomy and the subject_id field contains split info.
        nwb_subject = nwbfile.subject
        raw_subject_id = nwb_subject.subject_id if nwb_subject else "unknown"
        # Strip split type from subject ID:
        # "Finch_z_r12r13_21_held_in_calib" -> "finch_z_r12r13_21"
        subject_id = raw_subject_id.lower()
        for suffix in ("_held_in_calib", "_held_in_minival", "_held_out_calib"):
            if subject_id.endswith(suffix):
                subject_id = subject_id[: -len(suffix)]
                break

        sex = Sex.UNKNOWN
        if nwb_subject and nwb_subject.sex:
            sex_str = nwb_subject.sex.upper()
            if sex_str == "M":
                sex = Sex.MALE
            elif sex_str == "F":
                sex = Sex.FEMALE

        subject = SubjectDescription(
            id=subject_id,
            species=Species.UNKNOWN,  # zebra finch not in taxonomy
            sex=sex,
        )

        recording_date = nwbfile.session_start_time.strftime("%Y%m%d")

        device_id = f"{subject_id}_{recording_date}"
        session_id = f"{device_id}_{split_type}"

        store_path = self.processed_dir / f"{session_id}.h5"
        if store_path.exists() and not self.args.reprocess:
            io.close()
            self.update_status("Skipped Processing")
            return

        session_description = SessionDescription(
            id=session_id,
            recording_date=datetime.datetime.strptime(recording_date, "%Y%m%d"),
        )

        device_description = DeviceDescription(
            id=device_id,
            recording_tech=RecordingTech.NEUROPIXELS_THRESHOLDCROSSINGS,
        )

        # Extract threshold crossings and convert to spike events.
        self.update_status("Extracting Spikes")
        spikes, units = extract_spikes_from_threshold_crossings(nwbfile)

        # Extract trial structure with spectrograms.
        self.update_status("Extracting Trials")
        trials, spectrograms = extract_trials(nwbfile)

        behaviour_data = {}
        if self.args.extract_behaviour:
            self.update_status("Extracting Behavior")
            vocalizations, eval_mask = extract_behaviour(nwbfile)
            behaviour_data["vocalizations"] = vocalizations
            behaviour_data["eval_mask"] = eval_mask

        io.close()

        data = Data(
            brainset=brainset_description,
            subject=subject,
            session=session_description,
            device=device_description,
            spikes=spikes,
            units=units,
            trials=trials,
            domain=spikes.domain,
            **behaviour_data,
        )
        if spectrograms is not None:
            data.spectrograms = spectrograms

        # Train / valid / test split: split trials 70/10/20.
        self.update_status("Creating splits")
        n_trials = len(trials)
        if n_trials > 3:
            train_trials, valid_trials, test_trials = trials.split(
                [0.7, 0.1, 0.2], shuffle=True, random_seed=42
            )
            data.set_train_domain(train_trials)
            data.set_valid_domain(valid_trials)
            data.set_test_domain(test_trials)
        else:
            # Very few trials (e.g. minival files with 2 trials): use time-based split.
            total_duration = data.domain.end[-1] - data.domain.start[0]
            t0 = data.domain.start[0]
            train_end = t0 + 0.7 * total_duration
            valid_end = train_end + 0.1 * total_duration
            data.set_train_domain(Interval(np.array([t0]), np.array([train_end])))
            data.set_valid_domain(Interval(np.array([train_end]), np.array([valid_end])))
            data.set_test_domain(Interval(np.array([valid_end]), np.array([data.domain.end[-1]])))

        self.update_status("Storing")
        with h5py.File(store_path, "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)

        logger.info(
            f"Saved {session_id} -> {store_path.name} "
            f"({store_path.stat().st_size / 1e6:.1f} MB, "
            f"{len(units)} units, {n_trials} trials)"
        )


def extract_spikes_from_threshold_crossings(nwbfile):
    """Convert binary threshold crossings (30 kHz, 85 channels) to spike events.

    Returns an IrregularTimeSeries of spike events and an ArrayDict of unit metadata.
    """
    tx_data = nwbfile.acquisition["tx"].data[:]  # (T, N_channels), binary float64
    tx_timestamps = nwbfile.acquisition["tx"].timestamps[:]  # (T,)
    n_channels = tx_data.shape[1]

    ts_parts = []
    idx_parts = []
    unit_meta = []

    for ch in range(n_channels):
        spike_mask = tx_data[:, ch] > 0.5  # binary: 0 or 1
        spike_times = tx_timestamps[spike_mask]

        unit_meta.append(
            {
                "id": f"channel_{ch:03d}",
                "unit_number": len(unit_meta),
                "count": len(spike_times),
                "type": int(RecordingTech.NEUROPIXELS_THRESHOLDCROSSINGS),
            }
        )

        if len(spike_times) > 0:
            ts_parts.append(spike_times)
            idx_parts.append(np.full(len(spike_times), len(unit_meta) - 1, dtype=np.int64))

    units = ArrayDict.from_dataframe(pd.DataFrame(unit_meta), unsigned_to_long=True)

    if ts_parts:
        timestamps = np.concatenate(ts_parts)
        unit_index = np.concatenate(idx_parts)
    else:
        timestamps = np.array([], dtype=np.float64)
        unit_index = np.array([], dtype=np.int64)

    # Domain spans the full threshold crossing recording.
    domain = Interval(
        start=np.array([tx_timestamps[0]]),
        end=np.array([tx_timestamps[-1] + 1.0 / TX_SAMPLING_RATE]),
    )

    spikes = IrregularTimeSeries(
        timestamps=timestamps,
        unit_index=unit_index,
        domain=domain,
    )
    spikes.sort()

    return spikes, units


def extract_trials(nwbfile):
    """Extract trial intervals and per-trial spectrograms from the NWB file."""
    trial_table = nwbfile.trials.to_dataframe()

    starts = trial_table["start_time"].values.astype(np.float64)
    stops = trial_table["stop_time"].values.astype(np.float64)
    trials = Interval(start=starts, end=stops)

    # Per-trial spectrograms stored in the trial table.
    spectrograms = None
    if "spectrogram_values" in trial_table.columns:
        spec_values = np.stack(trial_table["spectrogram_values"].values)  # (n_trials, freq, time)
        spec_freqs = np.stack(trial_table["spectrogram_frequencies"].values)  # (n_trials, freq)
        spec_times = np.stack(trial_table["spectrogram_times"].values)  # (n_trials, time)

        spectrograms = ArrayDict(
            data=spec_values.astype(np.float32),
            frequencies=spec_freqs.astype(np.float32),
            times=spec_times.astype(np.float32),
        )

        if "spectrogram_eval_mask" in trial_table.columns:
            spec_eval_mask = np.stack(trial_table["spectrogram_eval_mask"].values)
            spectrograms.eval_mask = spec_eval_mask

    return trials, spectrograms


def extract_behaviour(nwbfile):
    """Extract vocalization audio and evaluation mask."""
    voc_obj = nwbfile.acquisition["vocalizations"]
    voc_data = voc_obj.data[:].astype(np.float32)
    voc_ts = voc_obj.timestamps[:]

    vocalizations = IrregularTimeSeries(
        timestamps=voc_ts,
        audio=voc_data[:, np.newaxis] if voc_data.ndim == 1 else voc_data,
        domain="auto",
    )

    mask_obj = nwbfile.acquisition["eval_mask_audio"]
    mask_data = mask_obj.data[:]
    mask_ts = mask_obj.timestamps[:]

    eval_mask = IrregularTimeSeries(
        timestamps=mask_ts,
        mask=mask_data,
        domain="auto",
    )

    return vocalizations, eval_mask
