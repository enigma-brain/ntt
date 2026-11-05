# /// brainset-pipeline
# python-version = "3.11"
# dependencies = ["dandi==0.73.2"]
# ///

import datetime
import os
from argparse import ArgumentParser
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from brainsets import serialize_fn_map
from brainsets.descriptions import (
    BrainsetDescription,
    DeviceDescription,
    SessionDescription,
)
from brainsets.pipeline import BrainsetPipeline
from brainsets.taxonomy import RecordingTech
from brainsets.utils.dandi_utils import (
    download_file,
    extract_subject_from_nwb,
    get_nwb_asset_list,
)
from pynwb import NWBHDF5IO
from temporaldata import ArrayDict, Data, Interval, IrregularTimeSeries

parser = ArgumentParser()
parser.add_argument("--redownload", action="store_true")
parser.add_argument("--reprocess", action="store_true")
parser.add_argument(
    "--disable-quality-filters",
    dest="quality_filters",
    action="store_false",
    help=(
        "Disable the default IBL unit quality filters "
        "(presence_ratio, contamination). "
        "By default, only high-quality units are exported."
    ),
)
parser.add_argument(
    "--max-contamination",
    type=float,
    default=0.5,
    help="Maximum contamination value for a unit to be kept (default: 0.5)",
)
parser.add_argument(
    "--min-presence-ratio",
    type=float,
    default=0.9,
    help="Minimum presence ratio for a unit to be kept (default: 0.9)",
)
parser.set_defaults(extract_behaviour=False, quality_filters=True)


class Pipeline(BrainsetPipeline):
    brainset_id = "IBL_brainwide_2023"
    dandiset_id = "DANDI:000409"
    parser = parser

    @classmethod
    def get_manifest(cls, raw_dir, args) -> pd.DataFrame:
        asset_list = get_nwb_asset_list(cls.dandiset_id)
        manifest_list = [
            {"path": x.path, "url": x.download_url} for x in asset_list if "ecephys" in x.path
        ]

        # Add a session_id to each manifest_list item
        # session_id should look like: `university_sessionhash`
        for m in manifest_list:
            path = m["path"]
            university_code = "".join(
                [c for c in path.split("/")[0].split("-")[1] if not c.isdigit()]
            )
            session_hash = path.split("/")[1].split("ses-")[1]

            m["session_id"] = f"{university_code}_{session_hash}"

        manifest = pd.DataFrame(manifest_list).set_index("session_id")
        return manifest

    def download(self, manifest_item):
        self.update_status("DOWNLOADING")
        self.raw_dir.mkdir(exist_ok=True, parents=True)

        # Check if file or placeholder already exists
        asset_path = Path(manifest_item.path)
        fpath = self.raw_dir / asset_path

        if fpath.exists() and not self.args.redownload:
            return fpath

        fpath = download_file(
            manifest_item.path,
            manifest_item.url,
            self.raw_dir,
            overwrite=self.args.redownload,
        )
        return fpath

    def process(self, fpath):
        # Skip processing if a placeholder or non-HDF5 file is present
        # (occurs after we delete raw NWB files to save space).
        if not h5py.is_hdf5(fpath) and not self.args.reprocess:
            self.update_status("Skipped Processing")
            return

        # open file
        self.update_status("Loading NWB")
        io = NWBHDF5IO(fpath, "r")
        nwbfile = io.read()

        self.processed_dir.mkdir(exist_ok=True, parents=True)

        brainset_description = BrainsetDescription(
            id="IBL_brainwide_2023",
            origin_version="dandi/000409",
            derived_version="1.0.0",
            source="https://dandiarchive.org/dandiset/000409",
            description=(
                "The International Brain lab (IBL) aims to understand the neural basis of "
                "decision-making in the mouse by gathering a whole-brain activity map composed "
                "of electrophysiological recordings pooledfrom multiple laboratories. We have "
                "systematically recorded from nearly all major brain areas with Neuropixels "
                "probes, using a grid system for unbiased sampling and replicating each "
                "recording site in at least two laboratories. These data have been used to "
                "construct a brain-wide map of activityat single-spike cellular resolution "
                "during a decision-making task. In addition to the map, this dataset contains "
                "other information gathered during the task: sensory stimuli presented to the "
                "mouse; mousedecisions and response times; and mouse pose information from video "
                "recordings and DeepLabCut analysis."
            ),
        )

        print(f"Processing file: {fpath}")

        self.update_status("Extracting Metadata")
        # extract subject metadata
        # this dataset is from dandi, which has structured subject metadata, so we
        # can use the helper function extract_subject_from_nwb
        subject = extract_subject_from_nwb(nwbfile)

        # extract experiment metadata
        recording_date = nwbfile.session_start_time.strftime("%Y%m%d")
        device_id = f"{subject.id}_{recording_date}"
        session_id = f"{device_id}_{subject.id}"

        store_path = self.processed_dir / f"{session_id}.h5"
        if store_path.exists() and not self.args.reprocess:
            self.update_status("Skipped Processing")
            io.close()
            return

        # register session
        session_description = SessionDescription(
            id=session_id,
            recording_date=datetime.datetime.strptime(recording_date, "%Y%m%d"),
            task=None,
        )

        # register device
        device_description = DeviceDescription(
            id=device_id,
            recording_tech=RecordingTech.NEUROPIXELS_SPIKES,
        )

        self.update_status("Extracting Spikes")
        # extract spiking activity
        spikes, units = self.extract_spikes(
            nwbfile, recording_tech=RecordingTech.NEUROPIXELS_SPIKES, args=self.args
        )

        # close file
        io.close()

        # register session
        data = Data(
            brainset=brainset_description,
            subject=subject,
            session=session_description,
            device=device_description,
            # neural activity
            spikes=spikes,
            units=units,
            # domain
            domain="auto",
        )

        self.update_status("Creating splits")

        # Create time intervals for train/valid/test splits
        total_time = data.domain.end - data.domain.start
        train_end = data.domain.start + 0.7 * total_time  # 70% for training
        valid_end = train_end + 0.1 * total_time  # 10% for validation

        train_interval = Interval(data.domain.start, train_end)
        valid_interval = Interval(train_end, valid_end)
        test_interval = Interval(valid_end, data.domain.end)

        # Set domains based on time intervals
        data.set_train_domain(train_interval)
        data.set_valid_domain(valid_interval)
        data.set_test_domain(test_interval)

        self.update_status("Storing")
        with h5py.File(store_path, "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)

        # Normally would not delete the raw file, but this is a large dataset and we want to
        # save space. We create a placeholder file so the pipeline doesn't try to re-download
        # on subsequent runs
        self.update_status("Replacing raw file with placeholder")
        os.remove(fpath)
        # Create empty placeholder file
        with open(fpath, "w") as f:
            f.write("# Placeholder - original file processed and removed to save space\n")

    def extract_spikes(self, nwbfile, recording_tech, args):
        # spikes
        timestamps = []
        unit_index = []

        # units
        unit_meta = []

        # Get the spike_times_index which tells us the range of spikes for each unit
        spike_times_index = nwbfile.units.spike_times_index[:]

        # Access attributes directly from the units table
        # Build a helper function to safely get attribute values
        def get_unit_attribute(attr_name, i, default):
            """Safely get an attribute value for unit at index i."""
            try:
                if hasattr(nwbfile.units, attr_name):
                    attr_data = getattr(nwbfile.units, attr_name).data
                    if i < len(attr_data):
                        val = attr_data[i]
                        # Handle numpy types
                        if hasattr(val, "item"):
                            return val.item()
                        return val
            except (AttributeError, IndexError, KeyError):
                pass
            return default

        # Track which units pass quality filters
        units_kept = 0
        units_filtered = 0

        # Get total number of units
        n_units = len(spike_times_index)

        # Iterate over each unit index
        for i in range(n_units):
            # Extract spike times for this unit
            spiketimes = spike_times_index[i]

            # Extract quality metrics from the units table attributes
            try:
                contamination = float(get_unit_attribute("contamination", i, 1.0))
                presence_ratio = float(get_unit_attribute("presence_ratio", i, 0.0))
            except (ValueError, TypeError):
                # If metrics are unavailable, use values that will fail quality filters
                contamination = 1.0
                presence_ratio = 0.0

            # Apply quality filters if enabled
            if args.quality_filters:
                passes_quality = (
                    contamination <= args.max_contamination
                    and presence_ratio >= args.min_presence_ratio
                )
                if not passes_quality:
                    units_filtered += 1
                    continue

            units_kept += 1

            # Extract spikes - only track when present so unit_index stays in sync
            if len(spiketimes) > 0:
                timestamps.append(spiketimes)
                # Use len(unit_meta) as the unit index since we're building it incrementally
                unit_index.append([len(unit_meta)] * len(spiketimes))

            # Extract brain region information
            allen_location = str(get_unit_attribute("allen_location", i, ""))
            beryl_location = str(get_unit_attribute("beryl_location", i, ""))
            cosmos_location = str(get_unit_attribute("cosmos_location", i, ""))

            # Extract additional quality metrics
            firing_rate = float(get_unit_attribute("firing_rate", i, 0.0))
            isi_violations = float(
                get_unit_attribute("sliding_refractory_period_violation", i, 0.0)
            )
            drift = float(get_unit_attribute("drift", i, 0.0))

            # Build unit metadata
            unit_meta.append(
                {
                    "id": f"unit_{len(unit_meta)}",
                    "unit_number": len(unit_meta),
                    "unit_name": get_unit_attribute("unit_name", i, f"unit_{i}"),
                    "count": len(spiketimes),
                    "type": int(recording_tech),
                    # Brain region information
                    "allen_location": allen_location,
                    "beryl_location": beryl_location,
                    "cosmos_location": cosmos_location,
                    # Quality metrics
                    "contamination": contamination,
                    "presence_ratio": presence_ratio,
                    "firing_rate": firing_rate,
                    "isi_violations": isi_violations,
                    "drift": drift,
                }
            )

        self.update_status(f"Units kept: {units_kept}, Units filtered: {units_filtered}")

        # convert unit metadata to a Data object
        unit_meta_df = pd.DataFrame(unit_meta)  # list of dicts to dataframe
        units = ArrayDict.from_dataframe(
            unit_meta_df,
            unsigned_to_long=True,
        )

        # concatenate spikes
        if timestamps:
            timestamps = np.concatenate(timestamps)
            unit_index = np.concatenate(unit_index)
        else:
            timestamps = np.array([], dtype=float)
            unit_index = np.array([], dtype=int)

        # create spikes object
        spikes = IrregularTimeSeries(
            timestamps=timestamps,
            unit_index=unit_index,
            domain="auto",
        )

        # make sure to sort the spikes
        spikes.sort()

        return spikes, units
