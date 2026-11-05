import copy
import logging

import h5py
import numpy as np
from torch_brain.data import Dataset

from ntt.core.data.components.dynamic import dyn

logger = logging.getLogger(__name__)


class NTTDataset(Dataset):
    """Custom Dataset class for NTT neural data processing.

    This class extends the base Dataset from torch_brain to include
    NTT-specific functionality and configurations.
    """

    def __init__(self, *args, deep_validate_units: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._deep_validate_units = deep_validate_units
        self._validate_recording_units()

    def _validate_recording_units(self):
        """Validate that every recording's units meet the required contract.

        Runs once at init time. Cheap structural checks always run:
        - Missing ``units`` with ``id`` and ``unit_number`` fields
        - ``unit_number`` that is not contiguous 0-based (``np.arange(n)``)
        - Non-unique ``id`` values

        The expensive ``spikes.unit_index`` max-bound check only runs when
        ``deep_validate_units=True``. It materializes the entire per-recording
        ``unit_index`` array from disk (tens of MB per Neuropixels session),
        which adds minutes of network file system I/O on multi-hundred-session splits and is
        an exporter-correctness guard rather than something that should run on
        every training launch.
        """
        for recording_id in self.recording_dict:
            data = dyn(self._get_data_object(recording_id))

            if not hasattr(data, "units") or len(data.units) == 0:
                raise ValueError(f"{recording_id}: data is missing non-empty 'units' attribute.")

            if not (hasattr(data.units, "id") and hasattr(data.units, "unit_number")):
                raise ValueError(
                    f"{recording_id}: units missing required 'id' or 'unit_number' field. "
                )

            uid, un = np.asarray(data.units.id), np.asarray(data.units.unit_number)

            if len(np.unique(uid)) < len(uid):
                raise ValueError(f"{recording_id}: units.id is not unique.")

            if not np.array_equal(un, np.arange(len(un))):
                raise ValueError(f"{recording_id}: units.unit_number is not contiguous")

            if not self._deep_validate_units:
                continue

            # Check unit_index bounds if spikes exist (MUA-only datasets may not have spikes)
            if hasattr(data, "spikes"):
                if not hasattr(data.spikes, "unit_index"):
                    raise ValueError(f"{recording_id}: spikes is missing 'unit_index' attribute.")

                ui = data.spikes.__dict__.get("unit_index")
                ui_arr = ui[:] if isinstance(ui, h5py.Dataset) else ui
                if len(ui_arr) == 0 or int(ui_arr.max()) >= len(un):
                    raise ValueError(
                        f"{recording_id}: spikes.unit_index max exceeds number of units."
                    )

    def _get_data_config(self, recording_id: str):
        return self.recording_dict[recording_id]["config"]

    def _should_ignore_splits(self, recording_id: str) -> bool:
        recording_config = self._get_data_config(recording_id)
        return recording_config.get("ignore_splits", False)

    def get_sampling_intervals(self):
        r"""Returns a dictionary of sampling intervals for each session.

         This method adapts torch_brain.data.Dataset to support additional functionality:
         - Allows sessions to ignore split restrictions via `ignore_splits` config flag
         - Disables `sampling_intervals_modifier` for security reasons

         The sampling intervals represent the time ranges that can be sampled from each session.
         These intervals change based on the active split (train/val/test), unless `ignore_splits`
         is set to True for a recording, in which case the full domain is always used.

        Returns:
             dict: A dictionary mapping recording IDs to their corresponding sampling intervals.

         Raises:
             NotImplementedError: If a recording attempts to use `sampling_intervals_modifier`,
                 which has been disabled due to security concerns.
        """
        sampling_intervals_dict = {}
        for recording_id in self.recording_dict.keys():
            recording_config = self._get_data_config(recording_id)
            if self.split is None or self._should_ignore_splits(recording_id):
                sampling_domain = "domain"
            else:
                sampling_domain = f"{self.split}_domain"
            sampling_intervals = getattr(self._get_data_object(recording_id), sampling_domain)
            if (
                "sampling_intervals_modifier" in recording_config
                and recording_config["sampling_intervals_modifier"] is not None
            ):
                raise NotImplementedError(
                    "Custom sampling intervals modifier is disabled due to security concerns!"
                )
            sampling_intervals_dict[recording_id] = sampling_intervals
        return sampling_intervals_dict

    def get(self, recording_id: str, start: float, end: float):
        r"""Extract a slice from a recording.

        Adaptation of the original torch_brain Dataset.get method. The key difference
        is that the data leakage check is skipped if :obj:`ignore_splits` is set to
        :obj:`True` in the recording's config.

        Args:
            recording_id: The recording id of the slice. This is usually
                <brainset_id>/<session_id>
            start: The start time of the slice.
            end: The end time of the slice.

        Returns:
            A Data object containing all data for the specified recording between
            the given start and end times.
        """
        data = self._get_data_object(recording_id)
        sample = data.slice(start, end)

        if (
            self._check_for_data_leakage_flag
            and self.split is not None
            and not self._should_ignore_splits(recording_id)
        ):
            sample._check_for_data_leakage(self.split)

        self._update_data_with_prefixed_ids(sample)
        sample.config = self.recording_dict[recording_id]["config"]

        return sample

    def get_recording_data(self, recording_id: str):
        r"""Return the full data object for a recording, sliced to the active split.

        Adaptation of the original torch_brain ``Dataset.get_recording_data``. The
        base method runs the data-leakage check unconditionally, but recordings with
        :obj:`ignore_splits=True` are evaluated over their full domain (see
        :meth:`get_sampling_intervals`), so most events legitimately fall outside the
        split mask. Mirroring :meth:`get`, the leakage check is therefore skipped for
        ``ignore_splits`` recordings. Without this guard, whole-session consumers like
        ``n_neurons_per_session`` / ``extract_spike_data`` raise a spurious
        "data leakage" error on OOD sessions whenever the check is enabled.
        """
        data = self._get_data_object(recording_id)

        if self.split is not None:
            sampling_intervals = self.get_sampling_intervals()[recording_id]
            data = data.select_by_interval(sampling_intervals)
            if self._check_for_data_leakage_flag and not self._should_ignore_splits(recording_id):
                data._check_for_data_leakage(self.split)
        else:
            data = copy.deepcopy(data)

        self._update_data_with_prefixed_ids(data)
        return data
