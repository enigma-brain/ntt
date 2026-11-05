"""Two-view DataModule for NuCLR contrastive learning.

Extends ``NTTDataModule`` to produce paired views for contrastive learning.
For each training sample, the datamodule provides two time windows from the
same recording, along with match indices indicating shared neurons.
"""

import random
from typing import cast

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torch_brain.data.dataset import DatasetIndex

from ntt.core.data.components.dataset import NTTDataset
from ntt.core.data.components.transforms import (
    CreateBinnedSpikeCountsArray,
    DataToModelInput,
    UnitDropout,
)
from ntt.core.data.datamodule import NTTDataModule, NTTDataModuleConfig, collate
from ntt.core.model.tokenizer import TokenizerSpec
from ntt.utils import Split

from .two_view_types import TwoViewBatch, compute_match_idx


class TwoViewNTTDataset(Dataset):
    """Dataset wrapper that produces paired views from the same recording.

    Wraps an ``NTTDataset`` (without transform) and applies transforms
    independently to each view. Computes match_idx from unit numbers
    after UnitDropout but before DataToModelInput.

    Args:
        base_dataset: NTTDataset with ``transform=None``.
        pre_model_transforms: Transforms to apply before match_idx computation
            (e.g., UnitDropout). Applied independently to each view.
        model_input_transform: ``DataToModelInput`` transform applied last.
        window_length: Length of each view's time window in seconds.
        max_time_diff: Maximum time offset between view1 and view2 starts.
    """

    def __init__(
        self,
        base_dataset: NTTDataset,
        pre_model_transforms: list,
        model_input_transform: DataToModelInput,
        window_length: float,
        max_time_diff: float,
    ):
        self.base = base_dataset
        self.pre_model_transforms = pre_model_transforms
        self.model_input_transform = model_input_transform
        self.window_length = window_length
        self.max_time_diff = max_time_diff

        # Cache sampling intervals for view2 start computation
        self._intervals = base_dataset.get_sampling_intervals()

    def __len__(self):
        # Delegate to base dataset for sampler compatibility
        return len(self.base)

    def __getitem__(self, index: DatasetIndex):
        """Get two views from the same recording.

        Args:
            index: DatasetIndex(recording_id, start, end) from sampler.

        Returns:
            Tuple of (view1, view2, match_idx) where views are SpikeTimesInput
            and match_idx is a (2, M) tensor of matching unit indices.
        """
        # View 1: use the sampler's window
        data1 = self.base.get(index.recording_id, index.start, index.end)

        # View 2: sample a nearby window from the same recording
        start2 = self._sample_view2_start(index.recording_id, index.start)
        end2 = start2 + self.window_length
        data2 = self.base.get(index.recording_id, start2, end2)

        # Apply pre-model transforms independently (e.g., UnitDropout)
        for t in self.pre_model_transforms:
            data1 = t(data1)
        for t in self.pre_model_transforms:
            data2 = t(data2)

        # Compute match_idx from unit numbers (after dropout, before DataToModelInput)
        unit_ids_1 = data1.units.unit_number  # pyright: ignore[reportAttributeAccessIssue]
        unit_ids_2 = data2.units.unit_number  # pyright: ignore[reportAttributeAccessIssue]
        match = compute_match_idx(unit_ids_1, unit_ids_2)
        match_idx = torch.as_tensor(match, dtype=torch.long)

        # Apply DataToModelInput
        view1 = self.model_input_transform(data1)
        view2 = self.model_input_transform(data2)

        return view1, view2, match_idx

    def _sample_view2_start(self, recording_id: str, view1_start: float) -> float:
        """Sample a valid start time for view2 near view1.

        Uses the recording's sampling intervals to ensure the window fits.
        """
        intervals = self._intervals[recording_id]
        interval_starts = np.asarray(intervals.start, dtype=float)
        interval_ends = np.asarray(intervals.end, dtype=float)

        # Target range: within max_time_diff of view1
        target_lo = view1_start - self.max_time_diff
        target_hi = view1_start + self.max_time_diff

        # Find valid start ranges within each interval
        valid_ranges = []
        for s, e in zip(interval_starts, interval_ends, strict=True):
            # Valid start range for a full window within this interval
            valid_lo = max(s, target_lo)
            valid_hi = min(e - self.window_length, target_hi)
            if valid_lo <= valid_hi:
                valid_ranges.append((valid_lo, valid_hi))

        if not valid_ranges:
            # Fallback: same start as view1
            return view1_start

        # Sample uniformly from valid ranges (weighted by range length)
        lengths = [hi - lo for lo, hi in valid_ranges]
        total = sum(lengths)
        r = random.random() * total
        cumulative = 0.0
        for (lo, hi), length in zip(valid_ranges, lengths, strict=True):
            cumulative += length
            if r <= cumulative:
                return random.uniform(lo, hi)

        # Shouldn't reach here, but just in case
        lo, hi = valid_ranges[-1]
        return random.uniform(lo, hi)

    def get_sampling_intervals(self):
        """Delegate to base dataset for sampler creation."""
        return self._intervals


def two_view_collate(batch):
    """Custom collate function for TwoViewNTTDataset.

    Collates view1 and view2 samples separately using NTT's collate
    (handles behavioral_data=None), and keeps match_idx as a list.

    Args:
        batch: List of (view1, view2, match_idx) tuples from dataset.

    Returns:
        TwoViewBatch with collated views and match_idx list.
    """
    views1 = [b[0] for b in batch]
    views2 = [b[1] for b in batch]
    match_idxs = [b[2] for b in batch]

    view1_batch = collate(views1)
    view2_batch = collate(views2)

    return TwoViewBatch(
        view1=view1_batch,
        view2=view2_batch,
        match_idx=match_idxs,
    )


class TwoViewNTTDataModule(NTTDataModule):
    """DataModule that produces paired views for NuCLR contrastive learning.

    Extends ``NTTDataModule``. For training, each sample contains two time
    windows (views) from the same recording. The second view is sampled
    within ``max_time_diff`` seconds of the first.

    For validation, behavior depends on ``val_two_view``:
    - True: produce two views (for computing val contrastive loss)
    - False: standard single-view (for reconstruction eval only)

    Args:
        config: Standard NTTDataModuleConfig.
        tokenizer_spec: Tokenizer spec for data config propagation.
        max_time_diff: Maximum time offset between view1 and view2.
        val_two_view: Whether to produce two views for validation.
    """

    def __init__(
        self,
        config: NTTDataModuleConfig,
        tokenizer_spec: TokenizerSpec | None = None,
        max_time_diff: float = 30.0,
        val_two_view: bool = True,
    ):
        super().__init__(config, tokenizer_spec)
        self.max_time_diff = max_time_diff
        self.val_two_view = val_two_view

    def _build_two_view_transforms(self, split: Split):
        """Split transform pipeline into pre-model and model-input components.

        Returns:
            Tuple of (pre_model_transforms, model_input_transform).
        """
        pre_model = []

        # UnitDropout (applied independently to each view). This is also where
        # the max_neurons cap is enforced for the two-view path, since
        # two_view_collate bypasses the base make_neuron_split_collate_fn
        # (see _build_dataloader).
        if (n_neurons := self.n_neurons_to_sample) is not None:
            pre_model.append(
                UnitDropout(
                    neuron_sel=n_neurons,
                    max_neurons=self.config.max_neurons,
                    reset_index=True,
                )
            )

        # Bin spikes if bin_size is specified (must come after UnitDropout,
        # before DataToModelInput, and before match_idx computation so that
        # binned_spike_counts is available on the Data object)
        if self.bin_size_ms is not None:
            pre_model.append(CreateBinnedSpikeCountsArray(bin_size_ms=self.bin_size_ms))

        # DataToModelInput (applied after match_idx computation)
        model_input = DataToModelInput(mode=self.input_type)

        return pre_model, model_input

    def _setup_two_view_split(
        self, split: Split, shuffle: bool = False
    ) -> tuple[TwoViewNTTDataset, Sampler]:
        """Set up a two-view dataset and sampler for the given split."""
        from torch import Generator
        from torch_brain.data.sampler import RandomFixedWindowSampler, SequentialFixedWindowSampler

        # Note: temporaldata uses 'valid' instead of 'val'
        dataset_split = "valid" if split == "val" else split

        # Create base dataset WITHOUT transforms (we apply them in TwoViewNTTDataset)
        base_dataset = NTTDataset(
            root=str(self.config.root),
            split=dataset_split,
            config=self.config.selections_to_omegaconf(split),
            transform=None,
        )

        # Build split transforms
        pre_model, model_input = self._build_two_view_transforms(split)

        # Wrap in TwoViewNTTDataset
        dataset = TwoViewNTTDataset(
            base_dataset=base_dataset,
            pre_model_transforms=pre_model,
            model_input_transform=model_input,
            window_length=self.window_length,
            max_time_diff=self.max_time_diff,
        )

        # Create sampler from sampling intervals
        sampling_intervals = base_dataset.get_sampling_intervals()
        if shuffle:
            sampler = RandomFixedWindowSampler(
                sampling_intervals=sampling_intervals,
                window_length=self.window_length,
                generator=Generator().manual_seed(self.config.seed),
            )
        else:
            sampler = SequentialFixedWindowSampler(
                sampling_intervals=sampling_intervals,
                window_length=self.window_length,
                drop_short=True,
            )

        return dataset, sampler

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets: two-view for train (and optionally val), single-view otherwise."""
        if stage in ("fit", None) and self.train_dataset is None:
            train_dataset, self.train_sampler = self._setup_two_view_split("train", shuffle=True)
            # TwoViewNTTDataset stands in for the NTTDataset slot on the base class
            self.train_dataset = cast(NTTDataset, train_dataset)
            self.check_train_holdout_leaks(train_dataset.base)

        if stage in ("validate", "fit", None) and self.val_dataset is None:
            if self.val_two_view:
                val_dataset, self.val_sampler = self._setup_two_view_split("val")
                self.val_dataset = cast(NTTDataset, val_dataset)
            else:
                self.val_dataset, self.val_sampler = self._setup_split("val")

        if stage in ("test", None) and self.test_dataset is None:
            # Test always uses single-view (standard eval)
            self.test_dataset, self.test_sampler = self._setup_split("test")

    def _build_dataloader(self, split: Split, **kwargs) -> DataLoader:
        """Build DataLoader with appropriate collate function."""
        self.check_setup(split, should_raise=True)
        dataset = self.get_dataset(split)
        assert dataset is not None  # check_setup(should_raise=True) guarantees this
        dl_kwargs = self.config.dataloader.model_dump()
        dl_kwargs = {k: v for k, v in dl_kwargs.items() if "hydra" not in k}
        dl_kwargs.update(kwargs)

        # Use two_view_collate for TwoViewNTTDataset, standard collate otherwise.
        # NOTE: the two-view path deliberately bypasses the base
        # make_neuron_split_collate_fn(max_neurons). The max_neurons cap is
        # instead enforced upstream by UnitDropout(max_neurons=...) in
        # _build_two_view_transforms, because two_view_collate must keep the
        # ragged per-element match_idx as a list (it cannot be neuron-split).
        if isinstance(dataset, TwoViewNTTDataset):
            collate_fn = two_view_collate
        else:
            collate_fn = collate

        return DataLoader(
            dataset=dataset,
            sampler=self.get_sampler(split),
            collate_fn=collate_fn,
            **dl_kwargs,
        )
