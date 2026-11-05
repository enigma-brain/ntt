# /// brainset-pipeline
# python-version = "3.11"
# dependencies = ["allensdk>=2.16", "pynwb>=2.5,<2.8", "hdmf>=3.14,<4"]
# ///

import datetime
from argparse import ArgumentParser, Namespace
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from allensdk.brain_observatory.behavior.behavior_project_cache import (
    VisualBehaviorNeuropixelsProjectCache,
)
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

parser.add_argument(
    "--disable-allen-quality-filters",
    dest="allen_quality_filters",
    action="store_false",
    help=(
        "Disable the default AllenSDK unit quality filters "
        "(isi_violations, amplitude_cutoff, presence_ratio). "
        "By default, only high-quality units are exported."
    ),
)
parser.set_defaults(extract_behaviour=False, allen_quality_filters=True)


class Pipeline(BrainsetPipeline):
    """Export pipeline for the Allen Visual Behavior Neuropixels dataset.

    This pipeline relies on the AllenSDK VisualBehaviorNeuropixelsProjectCache
    to download and cache NWB files locally. Only spiking activity and
    lightweight unit metadata are exported (no LFP or heavy analysis tables).
    """

    brainset_id = "allen_visual_behaviour_neuropixels"
    parser = parser

    # ---------------------------
    # Cache / manifest utilities
    # ---------------------------
    @classmethod
    def _get_manifest_path(cls, raw_dir: Path) -> Path:
        """Return the cache directory used by the AllenSDK cloud cache."""
        raw_dir = Path(raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        return raw_dir

    @classmethod
    def _get_cache(cls, raw_dir: Path) -> VisualBehaviorNeuropixelsProjectCache:
        """Create or load a VisualBehaviorNeuropixelsProjectCache bound to ``raw_dir``."""
        cache_dir = cls._get_manifest_path(raw_dir)
        return VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=cache_dir)

    # ---------------------------
    # BrainsetPipeline interface
    # ---------------------------
    @classmethod
    def get_manifest(
        cls,
        raw_dir: Path,
        args: Namespace | None,
    ) -> pd.DataFrame:
        """Build a manifest from the Visual Behavior Neuropixels ecephys session table.

        Each row corresponds to a single ecephys session NWB file.
        """
        cache = cls._get_cache(raw_dir)
        ecephys_sessions = cache.get_ecephys_session_table()

        manifest_rows: list[dict] = []
        for ecephys_session_id, row in ecephys_sessions.iterrows():
            mouse_id = int(row["mouse_id"])
            session_type = str(row.get("session_type", ""))
            experience_level = str(row.get("experience_level", ""))
            behavior_session_id = (
                int(row["behavior_session_id"]) if "behavior_session_id" in row else -1
            )

            # Create a human-readable, dataset-local session identifier.
            session_id = f"mouse{mouse_id}_session{int(ecephys_session_id)}"

            manifest_rows.append(
                {
                    "session_id": session_id,
                    "ecephys_session_id": int(ecephys_session_id),
                    "mouse_id": mouse_id,
                    "behavior_session_id": behavior_session_id,
                    "session_type": session_type,
                    "experience_level": experience_level,
                }
            )

        manifest = pd.DataFrame(manifest_rows).set_index("session_id")
        return manifest

    def download(self, manifest_item):
        """Ensure that the NWB file for this session is cached locally.

        The AllenSDK cache will download the session NWB file on demand and
        remember its location. We simply trigger this download and return the
        ecephys_session_id so that :meth:`process` can re-open the session.
        """
        self.update_status("DOWNLOADING (AllenSDK cache)")

        cache = self._get_cache(self.raw_dir)
        ecephys_session_id = int(manifest_item.ecephys_session_id)

        # Trigger download / verification of the spikes NWB file.
        # We do not keep the returned object here to keep the interface
        # consistent with other pipelines (which pass a lightweight handle
        # to ``process``).
        cache.get_ecephys_session(ecephys_session_id=ecephys_session_id)

        return ecephys_session_id

    def process(self, ecephys_session_id: int):
        """Process a single ecephys session into the brainsets format."""
        self.processed_dir.mkdir(exist_ok=True, parents=True)

        cache = self._get_cache(self.raw_dir)

        self.update_status("Loading session from AllenSDK cache")
        ecephys_session = cache.get_ecephys_session(ecephys_session_id=ecephys_session_id)

        # Prepare output path and short-circuit if already processed.
        sessions_table = cache.get_ecephys_session_table()
        mouse_id = int(sessions_table.loc[ecephys_session_id, "mouse_id"])
        recording_date = ecephys_session.metadata.get("session_start_time", None)
        if recording_date is None:
            # Fallback: try attribute access
            recording_date = getattr(
                ecephys_session, "session_start_time", datetime.datetime.now()
            )

        # Simple filename: specimen{specimen_id}_session{ecephys_session_id}.h5
        session_id = f"mouse{mouse_id}_session{ecephys_session_id}"
        store_path = self.processed_dir / f"{session_id}.h5"
        if store_path.exists() and not self.args.reprocess:
            self.update_status("Skipped Processing")
            return

        # -----------------
        # Dataset metadata
        # -----------------
        brainset_description = BrainsetDescription(
            id="allen_visual_behaviour_neuropixels",
            origin_version="allen/vbn",
            derived_version="1.0.0",
            source=("https://allensdk.readthedocs.io/en/latest/visual_behavior_neuropixels.html"),
            description=(
                "Neuropixels recordings from mice performing visual behavior tasks, "
                "with associated electrophysiology and lightweight unit metadata. "
            ),
        )

        subject = SubjectDescription(
            id=str(mouse_id),
            species=Species.MUS_MUSCULUS,
        )

        session_description = SessionDescription(
            id=session_id,
            recording_date=recording_date
            if isinstance(recording_date, datetime.datetime)
            else datetime.datetime.fromisoformat(str(recording_date)),
            task=None,
        )

        # Simple device_id based on specimen and date
        device_id = f"mouse{mouse_id}_{recording_date.strftime('%Y%m%d')}"
        device_description = DeviceDescription(
            id=device_id,
            recording_tech=RecordingTech.NEUROPIXELS_SPIKES,
        )

        # -----------------
        # Spikes & units
        # -----------------
        self.update_status("Extracting spikes and unit metadata")

        spike_times_series = ecephys_session.spike_times  # pandas Series: unit_id -> np.ndarray
        units_table = ecephys_session.get_units()

        # Optional Allen "default" quality filters (implemented locally for VBN).
        if self.args.allen_quality_filters:
            # Match the Visual Coding Neuropixels default thresholds.
            required_cols = {"isi_violations", "amplitude_cutoff", "presence_ratio"}
            if required_cols.issubset(set(units_table.columns)):
                units_table = units_table[
                    (units_table["isi_violations"] < 0.5)
                    & (units_table["amplitude_cutoff"] < 0.1)
                    & (units_table["presence_ratio"] > 0.9)
                ]

        timestamps_list: list[np.ndarray] = []
        unit_index_list: list[np.ndarray] = []
        unit_meta: list[dict] = []

        for unit_id, unit_row in units_table.iterrows():
            # Spike times for this unit (in seconds)
            unit_spike_times = np.asarray(spike_times_series[unit_id], dtype=float)
            if unit_spike_times.size == 0:
                continue

            timestamps_list.append(unit_spike_times)
            # Use len(unit_meta) so indices stay contiguous when units are skipped.
            unit_index_list.append(np.full_like(unit_spike_times, len(unit_meta), dtype=int))

            # Lightweight metadata per unit
            unit_meta.append(
                {
                    "id": str(unit_id),
                    "unit_number": len(unit_meta),
                    "ecephys_session_id": int(ecephys_session_id),
                    "count": int(unit_spike_times.size),
                    # Lightweight anatomical / quality information
                    "structure_acronym": str(unit_row.get("structure_acronym", "")),
                    "firing_rate": float(unit_row["firing_rate"])
                    if "firing_rate" in unit_row
                    else np.nan,
                    "probe_id": int(unit_row["ecephys_probe_id"])
                    if "ecephys_probe_id" in unit_row
                    else -1,
                    "ecephys_channel_id": int(unit_row["ecephys_channel_id"])
                    if "ecephys_channel_id" in unit_row
                    else -1,
                    # Quality metrics
                    "isi_violations": float(unit_row["isi_violations"])
                    if "isi_violations" in unit_row
                    else np.nan,
                    "amplitude_cutoff": float(unit_row["amplitude_cutoff"])
                    if "amplitude_cutoff" in unit_row
                    else np.nan,
                    "presence_ratio": float(unit_row["presence_ratio"])
                    if "presence_ratio" in unit_row
                    else np.nan,
                }
            )

        if not timestamps_list:
            self.update_status("No spikes found for this session")
            return

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

        # -----------------
        # Build Data object
        # -----------------
        data = Data(
            brainset=brainset_description,
            subject=subject,
            session=session_description,
            device=device_description,
            spikes=spikes,
            units=units,
            domain="auto",
        )

        # -----------------
        # Train/valid/test splits
        # -----------------
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

        # -----------------
        # Store to HDF5
        # -----------------
        self.update_status("Storing")
        with h5py.File(store_path, "w") as file:
            data.to_hdf5(file, serialize_fn_map=serialize_fn_map)
