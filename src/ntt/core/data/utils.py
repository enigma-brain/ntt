import numpy as np
import pandas as pd
from jaxtyping import Float, Int, Shaped
from temporaldata import Data
from torch_brain.data import Dataset
from tqdm.auto import tqdm

from ntt.core.data.components.dynamic import dyn
from ntt.core.data.components.transforms import CreateBinnedSpikeCountsArray


def n_neurons_per_session(dataset: Dataset) -> dict[str, int]:
    """Number of neurons per session in each dataset.

    Returns a nested dictionary mapping dataset split ('train', 'val', 'test')
    to a dictionary mapping session_id to number of neurons.

    Raises:
        RuntimeError: If datasets are not set up yet. Call setup() first.
    """
    result: dict[str, int] = {}
    session_ids = dataset.get_session_ids()
    for session_id in tqdm(session_ids, desc="Counting neurons per session", leave=False):
        recording_data = dataset.get_recording_data(session_id)
        num_units = len(recording_data.units)  # type: ignore[attr-defined]
        result[session_id] = num_units
    return result


def n_neurons_total(dataset: Dataset) -> int:
    """Total number of neurons available in each dataset.

    Returns a dictionary mapping dataset split ('train', 'val', 'test') to
    the total number of neurons across all sessions in that split.

    Raises:
        RuntimeError: If datasets are not set up yet. Call setup() first.
    """
    return sum(n_neurons_per_session(dataset).values())


def print_transforms(dataset: Dataset) -> None:
    """Print the transform pipeline for a specific dataset split."""
    transforms = dataset.transform
    if hasattr(transforms, "transforms"):
        # Compose object
        for i, transform in enumerate(transforms.transforms):  # type: ignore[attr-defined]
            print(f"  {i + 1}. {transform!r}")
    else:
        print(f"  {transforms}")


def extract_spike_data(dataset: Dataset, bin_size_ms: float | None = None) -> pd.DataFrame:
    """Extract spike data from the dataset into a DataFrame.

    Each row corresponds to a recording session and contains:
        - session_id: ID of the recording session
        - start: Relative start time of the recording session
        - end: Relative end time of the recording session
        - neuron_ids: Unique neuron IDs in the session
        - timestamps: Spike timestamps
        - neuron_indices: Indices of neuron id (in `neuron_ids`) for each spike
        - start_global: Global (virtual) start time of the recording session (i.e. cumulative time
          accross sessions)
        - end_global: Global (virtual) end time of the recording session (i.e. cumulative time
          accross sessions)
        - neuron_indices_global: Indices of neuron id for each spike (in
          `global_neuron_ids = dataset.get_unit_ids()`)

    Args:
        dataset: The Dataset to extract spike data from.
    Returns:
        A DataFrame with spike data for each recording session.

    NOTE: Global times are cumulative, where sessions are sorted alphabetically (see
      :py:func:`Dataset.get_session_ids`). They DO NOT transfer accross dataset splits!!!
      So, for the same datamodule, the global times in the train/val/test datasets are not
      in the same domain.
    NOTE: Global neuron indices map to the global neuron IDs based on the order returned by
      :py:func:`Dataset.get_unit_ids`. The DO transfer accross dataset splits! So a neuron
      specific tokenizer on the train dataset CAN be applied to the val/test datasets.
    TODO: Define pandera schema for the returned spikes data DataFrame.
    """
    spike_data = []
    # unique session ids in this dataset (sorted alphabetically)
    global_session_ids: list[str] = dataset.get_session_ids()
    # unique neuron ids (strings, with prefixes indicating sessions) across the entire dataset
    global_neuron_ids: list[np.str_] = dataset.get_unit_ids()
    # create mapping from global neuron IDs to "global indices"
    global_neuron_ids_to_indices: dict[str, int] = {
        gid.item(): gidx for gidx, gid in enumerate(global_neuron_ids)
    }
    # initialize global duration accumulator
    global_duration = 0.0
    desc = f"Processing sessions ({len(global_neuron_ids)} total neurons)"
    for session_id in tqdm(global_session_ids, desc=desc, leave=False):
        # get the underlying Data object for this section (filtered for `dataset.split`)
        recording_data: Data = dataset.get_recording_data(session_id)
        # add binned spike counts to the data
        binned_spike_counts: Int[np.ndarray, "n_time_bins n_units"] | None = None
        if bin_size_ms is not None:
            spike_binning_func = CreateBinnedSpikeCountsArray(bin_size_ms=bin_size_ms)
            recording_data = spike_binning_func(recording_data)
            binned_spike_counts = dyn(recording_data).binned_spike_counts.counts
        # extract the time "domain" (i.e. start to end time range) for the dataset
        domain = dyn(recording_data).domain
        start: float = domain.start.item()
        end: float = domain.end.item()
        duration = end - start
        # update global duration
        global_duration += duration
        # extract spikes data and units data
        spikes = dyn(recording_data).spikes
        units = dyn(recording_data).units
        # unique neuron IDs in this session.
        neuron_ids: Shaped[np.ndarray, " n_units"] = units.id
        assert neuron_ids.dtype.type == np.str_
        # spike timestamps for this session (all neurons)
        timestamps: Float[np.ndarray, " n_spikes"] = spikes.timestamps
        # index into `neurons_ids` corresponding to each spike
        neuron_idcs: Int[np.ndarray, " n_spikes"] = spikes.unit_index
        # subset of neuron *indices* from global map, as an indexable array
        global_neuron_indices_sel: Int[np.ndarray, " n_units"] = np.array(
            [global_neuron_ids_to_indices[nid] for nid in neuron_ids]
        )
        # map local idcs (i.e. into `neuron_ids`) to global idcs (i.e. into `global_neuron_ids`)
        neuron_idcs_global: Int[np.ndarray, " n_spikes"] = global_neuron_indices_sel[neuron_idcs]
        spike_data.append(
            {
                "session_id": session_id,
                "start": start,
                "end": end,
                "neuron_ids": neuron_ids,
                "timestamps": timestamps,
                "neuron_indices": neuron_idcs,
                "start_global": global_duration - duration,
                "end_global": global_duration,
                "neuron_indices_global": neuron_idcs_global,
                "binned_spike_counts": binned_spike_counts,
            }
        )
    return pd.DataFrame(spike_data)
