# /// brainset-pipeline
# python-version = "3.11"
# dependencies = [
#   "h5py",
#   "requests",
# ]
# ///

import datetime
import json
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
from brainsets.taxonomy import RecordingTech, Species
from temporaldata import ArrayDict, Data, Interval, IrregularTimeSeries

parser = ArgumentParser()
parser.add_argument("--redownload", action="store_true")
parser.add_argument("--reprocess", action="store_true")
parser.set_defaults(extract_behaviour=False)


class Pipeline(BrainsetPipeline):
    """Export pipeline for the C4 Cerebellum 2025 dataset.

    This pipeline downloads h5 files from URLs and extracts spiking data
    organized by lab and session. Each downloaded h5 file contains multiple
    sessions under the 'datasets/' group.
    """

    brainset_id = "c4_dataset_cerebellum_2025"
    parser = parser

    @classmethod
    def _get_data_sources_path(cls) -> Path:
        """Return the path to data_sources.json."""
        return Path(__file__).parent / "data_sources.json"

    @classmethod
    def _load_data_sources(cls) -> dict[str, str]:
        """Load the data sources JSON file."""
        with open(cls._get_data_sources_path()) as f:
            return json.load(f)

    @classmethod
    def get_manifest(cls, raw_dir: Path, args: Namespace | None) -> pd.DataFrame:
        """Build a manifest from the data sources.

        Each row corresponds to a source h5 file that needs to be downloaded.
        """
        data_sources = cls._load_data_sources()
        manifest_rows: list[dict] = []

        for source_name, url in data_sources.items():
            manifest_rows.append(
                {
                    "source_name": source_name,
                    "url": url,
                    "session_id": source_name,  # Use source_name as session_id for now
                }
            )

        manifest = pd.DataFrame(manifest_rows).set_index("session_id")
        return manifest

    def download(self, manifest_item):
        """Download the h5 file for this data source from a remote URL.

        Handles Figshare ndownloader URLs by converting them to API endpoints.
        """
        self.update_status("DOWNLOADING")

        self.raw_dir.mkdir(exist_ok=True, parents=True)
        source_name = manifest_item.source_name
        url = manifest_item.url

        # Determine filename from source_name
        fpath = self.raw_dir / f"{source_name}.h5"

        if fpath.exists() and not self.args.redownload:
            # Verify existing file is valid h5
            try:
                with h5py.File(fpath, "r"):
                    pass
                return fpath
            except OSError:
                self.update_status(f"Existing file {fpath} appears corrupted, re-downloading")
                fpath.unlink()

        # Convert Figshare ndownloader URL to API endpoint if needed
        # Format: https://...ndownloader/files/{file_id} -> https://api.figshare.com/v2/file/download/{file_id}
        download_url = url
        if "/ndownloader/files/" in url:
            # Extract file ID from ndownloader URL
            file_id = url.split("/ndownloader/files/")[-1].split("?")[0].split("#")[0]
            # Use Figshare API endpoint (works for both figshare.com and institutional instances)
            download_url = f"https://api.figshare.com/v2/file/download/{file_id}"
            self.update_status(f"Converted ndownloader URL to API endpoint: {download_url}")

        self.update_status("DOWNLOADING")
        try:
            response = requests.get(download_url, stream=True, timeout=60 * 60)
            response.raise_for_status()  # Raise an exception for bad status codes

            block_size = 1024 * 1024  # 1 MB chunks
            with open(fpath, "wb") as file:
                for data in response.iter_content(block_size):
                    file.write(data)
        except requests.exceptions.RequestException as e:
            self.update_status(f"ERROR: Failed to download {source_name} from {download_url}: {e}")
            if fpath.exists():
                fpath.unlink()
            raise
        except Exception as e:
            self.update_status(f"ERROR: Unexpected error downloading {source_name}: {e}")
            if fpath.exists():
                fpath.unlink()
            raise

        # Verify the downloaded file is a valid h5 file
        try:
            with h5py.File(fpath, "r"):
                pass
        except OSError as e:
            self.update_status(f"ERROR: Downloaded file {fpath} is not a valid h5 file: {e}")
            if fpath.exists():
                fpath.unlink()
            raise ValueError(f"Downloaded file is not a valid h5 file: {fpath}") from e

        self.update_status(f"Successfully downloaded {source_name}")
        return fpath

    def process(self, fpath: Path):
        """Process a downloaded h5 file, extracting all sessions and neurons."""
        self.processed_dir.mkdir(exist_ok=True, parents=True)

        self.update_status("Loading h5 file")
        with h5py.File(fpath, "r") as h5_file:

            def _decode_scalar(val):
                """Decode bytes/arrays/scalars to Python primitives."""
                if isinstance(val, bytes):
                    return val.decode("utf-8")
                if isinstance(val, np.ndarray):
                    if val.shape == ():
                        return _decode_scalar(val.item())
                    if val.size > 0:
                        return _decode_scalar(val.flatten()[0])
                if hasattr(val, "item"):
                    try:
                        return val.item()
                    except Exception:
                        pass
                return val

            if "datasets" in h5_file:
                datasets_group = h5_file["datasets"]
                session_iter = [
                    ("datasets", session_name, datasets_group[session_name])
                    for session_name in datasets_group.keys()
                ]
            else:
                # Handle lisberger layout: neuron groups at root named lisberger_neuron_XXXX
                lisberger_groups = [
                    (name, grp)
                    for name, grp in h5_file.items()
                    if isinstance(grp, h5py.Group) and name.startswith("lisberger_neuron_")
                ]
                if not lisberger_groups:
                    self.update_status(f"ERROR: No recognizable groups in {fpath}")
                    return

                # Group neurons by their dataset_id to form sessions
                sessions_map: dict[str, list[tuple[str, h5py.Group]]] = {}
                for name, grp in lisberger_groups:
                    dataset_id_val = grp.get("dataset_id")
                    dataset_id = (
                        _decode_scalar(dataset_id_val[()])
                        if dataset_id_val is not None
                        else "unknown_session"
                    )
                    dataset_id = str(dataset_id)
                    sessions_map.setdefault(dataset_id, []).append((name, grp))

                session_iter = [
                    ("lisberger", session_name, neuron_list)
                    for session_name, neuron_list in sessions_map.items()
                ]

            # Iterate through sessions (either datasets layout or lisberger grouping)
            for layout_type, session_name, session_group in session_iter:
                if layout_type == "datasets":
                    neuron_items = [
                        (key, session_group[key])
                        for key in session_group.keys()
                        if key.isdigit() and isinstance(session_group[key], h5py.Group)
                    ]
                    if not neuron_items:
                        self.update_status(f"WARNING: No neurons found in session {session_name}")
                        continue
                    first_neuron_group = neuron_items[0][1]
                else:
                    neuron_items = session_group  # already list of (name, group)
                    first_neuron_group = neuron_items[0][1]

                # Determine lab_id from the first neuron
                lab_id_val = first_neuron_group.get("lab_id")
                if lab_id_val is None:
                    self.update_status(f"WARNING: lab_id not found in session {session_name}")
                    continue
                lab_id = str(_decode_scalar(lab_id_val[()]))

                session_id = f"{lab_id}_{session_name}"
                store_path = self.processed_dir / f"{session_id}.h5"

                if store_path.exists() and not self.args.reprocess:
                    self.update_status(f"Skipped Processing: {session_id}")
                    continue

                self.update_status(f"Processing session: {session_id}")

                session_lab_id = lab_id
                timestamps_list: list[np.ndarray] = []
                unit_index_list: list[np.ndarray] = []
                unit_meta: list[dict] = []

                for neuron_name, neuron_group in neuron_items:
                    # Get neuron_id
                    try:
                        neuron_id = (
                            int(neuron_name)
                            if neuron_name.isdigit()
                            else int(_decode_scalar(neuron_group.get("neuron_id")[()]))
                        )
                    except Exception:
                        continue

                    if "spike_indices" not in neuron_group:
                        continue

                    spike_indices = np.asarray(neuron_group["spike_indices"], dtype=np.int64)
                    if spike_indices.size == 0:
                        continue

                    # sampling_rate
                    if "sampling_rate" in neuron_group:
                        sampling_rate = float(_decode_scalar(neuron_group["sampling_rate"][()]))
                    else:
                        sampling_rate = 30000.0

                    spike_times = spike_indices.astype(float) / sampling_rate

                    lab_id_val = neuron_group.get("lab_id")
                    if lab_id_val is not None:
                        lab_id_val = str(_decode_scalar(lab_id_val[()]))

                    # optotagged_label
                    optotagged_label = None
                    if "optotagged_label" in neuron_group:
                        optotagged_label_val = neuron_group["optotagged_label"][()]
                        optotagged_label = _decode_scalar(optotagged_label_val)
                        if optotagged_label == "":
                            optotagged_label = None

                    # layer / phyllum_layer
                    layer = None
                    if "phyllum_layer" in neuron_group:
                        layer_val = _decode_scalar(neuron_group["phyllum_layer"][()])
                        # keep as string if not numeric
                        try:
                            layer = int(layer_val)
                        except (ValueError, TypeError):
                            layer = str(layer_val) if layer_val is not None else None

                    unit_number = len(unit_meta)
                    timestamps_list.append(spike_times)
                    unit_index_list.append(np.full_like(spike_times, unit_number, dtype=int))

                    unit_meta_dict = {
                        "id": str(neuron_id),
                        "unit_number": unit_number,
                        "neuron_id": int(neuron_id),
                        "lab_id": lab_id_val,
                        "count": int(spike_times.size),
                    }

                    if optotagged_label is not None:
                        unit_meta_dict["optotagged_label"] = str(optotagged_label)

                    if layer is not None:
                        unit_meta_dict["layer"] = layer

                    unit_meta.append(unit_meta_dict)

                if not timestamps_list:
                    self.update_status(f"No spikes found for session {session_id}")
                    continue

                if session_lab_id is None:
                    self.update_status(
                        f"ERROR: Could not determine lab_id for session {session_id}"
                    )
                    continue

                timestamps = np.concatenate(timestamps_list)
                unit_index = np.concatenate(unit_index_list)

                spikes = IrregularTimeSeries(
                    timestamps=timestamps,
                    unit_index=unit_index,
                    domain="auto",
                )
                spikes.sort()

                unit_meta_df = pd.DataFrame(unit_meta)
                units = ArrayDict.from_dataframe(unit_meta_df, unsigned_to_long=True)

                brainset_description = BrainsetDescription(
                    id="c4_dataset_cerebellum_2025",
                    origin_version="c4/2025",
                    derived_version="1.0.0",
                    source="C4 Cerebellum Dataset 2025",
                    description=(
                        "Cerebellar neuropixels recordings from multiple labs, "
                        "with optotagging labels and layer information where available."
                    ),
                )

                subject = SubjectDescription(
                    id=session_lab_id,
                    species=Species.MUS_MUSCULUS
                    if lab_id != "lisberger"
                    else Species.MACACA_MULATTA,
                )

                try:
                    date_part = session_name.split("_")[0]
                    if len(date_part.split("-")) == 3:
                        parts = date_part.split("-")
                        recording_date = datetime.datetime.strptime(
                            f"20{parts[0]}-{parts[1]}-{parts[2]}", "%Y-%m-%d"
                        )
                    else:
                        recording_date = datetime.datetime.now()
                except (ValueError, IndexError):
                    recording_date = datetime.datetime.now()

                session_description = SessionDescription(
                    id=session_id,
                    recording_date=recording_date,
                    task=None,
                )

                device_id = f"{session_lab_id}_{session_name}"
                device_description = DeviceDescription(
                    id=device_id,
                    recording_tech=RecordingTech.NEUROPIXELS_SPIKES,
                )

                data = Data(
                    brainset=brainset_description,
                    subject=subject,
                    session=session_description,
                    device=device_description,
                    spikes=spikes,
                    units=units,
                    domain="auto",
                )

                self.update_status("Creating splits")
                total_time = data.domain.end - data.domain.start
                train_end = data.domain.start + 0.7 * total_time
                valid_end = train_end + 0.1 * total_time

                train_interval = Interval(data.domain.start, train_end)
                valid_interval = Interval(train_end, valid_end)
                test_interval = Interval(valid_end, data.domain.end)

                data.set_train_domain(train_interval)
                data.set_valid_domain(valid_interval)
                data.set_test_domain(test_interval)

                self.update_status("Storing")
                with h5py.File(store_path, "w") as file:
                    data.to_hdf5(file, serialize_fn_map=serialize_fn_map)

                self.update_status(f"Completed: {session_id}")
