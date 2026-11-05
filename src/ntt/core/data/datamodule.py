from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import lightning as L
from lightning.pytorch.utilities.parsing import AttributeDict
from omegaconf import ListConfig
from pydantic import (
    BaseModel,
    Field,
    field_serializer,
    model_validator,
)
from temporaldata import Data
from torch import Generator
from torch.utils.data import DataLoader, Sampler
from torch_brain.data import Dataset
from torch_brain.data.collate import collate as _torch_brain_collate
from torch_brain.data.sampler import RandomFixedWindowSampler, SequentialFixedWindowSampler
from torch_brain.transforms import Compose

from ntt.core.components.dist import Dist
from ntt.core.components.types import (
    BinnedSpikesInput,
    DataType,
    RatesInput,
    SamplingStrategy,
    SpikeTimesInput,
)
from ntt.core.data.components.collate import make_neuron_split_collate_fn
from ntt.core.data.components.config import (
    DataLoaderConfig,
    SelectionConfig,
)
from ntt.core.data.components.dataset import NTTDataset
from ntt.core.data.components.holdout_guard import check_no_holdout_in_train
from ntt.core.data.components.transforms import (
    CreateBinnedSpikeCountsArray,
    DataToModelInput,
    EvalUnitSelection,
    EvalUnitSelectionConfig,
    PadNeuronsToFixed,
    UnitDropout,
    VariableWindowCrop,
)
from ntt.core.model.tokenizer import TokenizerSpec
from ntt.utils import (
    NTTMutableBaseModel,
    RankedLogger,
    Split,
    check_valid_split,
    fields_by_json_schema_extra,
    loggable_flat_dump,
)

log = RankedLogger(__name__, rank_zero_only=True)


def collate(batch: list) -> Any:
    """Collate wrapper that handles non-tensor metadata specially.

    1. behavioral_data: kept as list[dict] (one dict per batch element), not stacked.
    2. behavioral_intervals: kept as list[dict] (one dict per batch element), not stacked.

    All other fields (spike_matrix, pad_and_gap_mask, unit_ids, absolute_start,
    absolute_end, etc.) are handled
    by torch_brain's default collate (stacking along a new batch dimension).

    TODO: Integrate this in the collate map
    TODO: Make collate mode configurable

    """
    if not batch or not isinstance(batch[0], BinnedSpikesInput | RatesInput | SpikeTimesInput):
        return _torch_brain_collate(batch)

    behavior_list = [s.behavioral_data for s in batch]
    interval_list = [s.behavioral_intervals for s in batch]

    if isinstance(batch[0], BinnedSpikesInput):
        for field_name in ("absolute_start", "absolute_end"):
            # Contract: absolute_start/end must always be present.
            if any(getattr(s, field_name) is None for s in batch):
                raise ValueError(
                    f"Collation error: {field_name} must be present on every "
                    "BinnedSpikesInput sample."
                )

    # Strip non-collatable fields before passing to torch_brain collate.
    # Use {} (not None) as placeholder — torch collate can't handle None in NamedTuples.
    replace_kwargs: dict[str, Any] = {
        "behavioral_data": {},
        "behavioral_intervals": {},
    }
    batch_clean = [s._replace(**replace_kwargs) for s in batch]
    collated = cast(BinnedSpikesInput | SpikeTimesInput, _torch_brain_collate(batch_clean))

    # Restore non-collated metadata (None if no samples had it)
    has_behavior = any(b is not None for b in behavior_list)
    has_intervals = any(i is not None for i in interval_list)
    restore_kwargs: dict[str, Any] = {
        "behavioral_data": behavior_list if has_behavior else None,
        "behavioral_intervals": interval_list if has_intervals else None,
    }
    return collated._replace(**restore_kwargs)


class NTTDataModuleConfig(
    NTTMutableBaseModel,
    strict=False,
    frozen=False,
    extra="forbid",
    arbitrary_types_allowed=True,
):
    """Configuration schema for NTTDataModule.

    Attributes:
        root: Path to the root data directory
        tokenizer_spec: Optional TokenizerSpec to populate tokenizer fields from
        n_neurons: Number of neurons to sample (None=not set, False=disabled, int/Dist=enabled)
        window_length_seconds: Window length in seconds (None=not set)
        bin_size_ms: Bin size in milliseconds (None=not set, False=disabled)
        input_type: Type of neural data input (None=not set)
        train_selections: List of selection configs for training data
        val_selections: List of selection configs for validation data
            (defaults to train_selections)
        test_selections: List of selection configs for test data
            (defaults to val_selections)
        train_transforms: List of transform callables for training data
        val_transforms: List of transform callables for validation data
        test_transforms: List of transform callables for test data

    NOTE: Tokenizer fields (n_neurons, window_length_seconds, bin_size_ms, input_type) can be
    set directly or populated from a tokenizer_spec. Use `are_tokenizer_fields_set()` to
    check if all fields are configured, and `update_from_tokenizer_spec()` to override
    all tokenizer fields from a TokenizerSpec instance.

    TODO: Add NTTBaseModel/Dataclass for transforms to facilitate serializing/type checking
      transforms
    """

    root: str | Path = Field(
        ...,
        description="Path to the root data directory",
    )
    train_selections: list[SelectionConfig] = Field(
        ...,
        description="List of selection configs for training data",
        min_length=1,
    )
    val_selections: list[SelectionConfig] | None = Field(
        default=None,
        description="List of selection configs for validation data (defaults to train_selections)",
    )
    test_selections: list[SelectionConfig] | None = Field(
        default=None,
        description="List of selection configs for test data (defaults to val_selections)",
    )
    train_transforms: list[Callable[[Data], Any]] | None = Field(
        default=None,
        description="Optional list of additional transform callables for training data",
    )
    val_transforms: list[Callable[[Data], Any]] | None = Field(
        default=None,
        description="Optional list of additional transform callables for validation data",
    )
    test_transforms: list[Callable[[Data], Any]] | None = Field(
        default=None,
        description="Optional list of additional transform callables for test data",
    )
    dataloader: DataLoaderConfig = Field(
        default_factory=DataLoaderConfig,
        description="DataLoader configuration (batch size, workers, etc.)",
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility",
    )
    tokenizer_spec: TokenizerSpec | None = Field(
        default=None,
        description="Optional tokenizer specification (TokenizerSpec subclass instance). "
        "If provided, tokenizer fields will be populated from it.",
    )

    # Tokenizer fields - directly configurable data loading parameters
    # These can be set explicitly or populated from tokenizer_spec
    n_neurons: int | float | Dist | Literal[False] | None = Field(
        default=None,
        description="Number of neurons to sample. None=not set, False=disabled (use all), "
        "int=fixed count, float in (0,1]=min fraction, Dist=sample from distribution.",
        json_schema_extra={"is_tokenizer_field": True},
    )
    max_neurons: int | None = Field(
        default=None,
        description="Hard upper bound on the number of neurons to sample. "
        "Applied to limit memory usage on large sessions. "
        "None=no cap.",
    )
    window_length_seconds: float | Dist | None = Field(
        default=None,
        description="Window length in seconds for sampling (float or Dist). None=not set.",
        json_schema_extra={"is_tokenizer_field": True},
    )
    bin_size_ms: float | Literal[False] | None = Field(
        default=None,
        description="Bin size in milliseconds for spike binning. None=not set, False=disabled"
        " (use spike times).",
        json_schema_extra={"is_tokenizer_field": True},
    )
    input_type: DataType | None = Field(
        default=None,
        description="Type of neural data input (binary_spikes, rates, spike_times). None=not set.",
        json_schema_extra={"is_tokenizer_field": True},
    )

    # UnitDropout transform configuration. TODO: Should this be the tokenizer's responsibility?
    unit_dropout_exclude_zero: bool = Field(
        default=True,
        description="Whether to exclude zero-activity units from sampling.",
    )
    unit_dropout_sampling_strategy: SamplingStrategy = Field(
        default=SamplingStrategy.UNIFORM,
        description="Strategy for sampling units (uniform, linear_weighted, etc.).",
    )
    unit_dropout_sampling_temperature: float = Field(
        default=1.0,
        description="Temperature for weighted sampling (higher = more uniform).",
    )
    unit_dropout_sampling_weight_minimum: float = Field(
        default=1e-3,
        description="Minimum weight to add to all units to avoid zero weights.",
    )
    unit_dropout_sampling_use_global_stats: bool = Field(
        default=False,
        description="If True, uses global statistics for sampling weights instead of per-split "
        "statistics. NOTE: This is generally just an approximation and uses data from across "
        "splits, which may be considered unfair for evaluation. However, it can yield significant "
        "speedup by avoiding per-split weight computation.",
    )

    # Neuron padding option (not a tokenizer field, but a data loading option)
    pad_neurons_to_fixed: int | Literal[False] | None = Field(
        default=None,
        description="If set to an integer, pads neuron populations to this fixed size. "
        "Useful for models that require a fixed number of neurons (e.g., LDNS tokenizer). "
        "None=not set (will be inferred from tokenizer_spec.n_neurons if available), "
        "False=disabled (no padding).",
    )

    # Eval-only unit selection (replaces UnitDropout/padding on val and test)
    eval_unit_selection: EvalUnitSelectionConfig | None = Field(
        default=None,
        description="If set, val/test keep the units chosen by "
        "ntt.core.data.components.transforms.select_eval_units (all units, or a seeded "
        "random max_neurons per session) instead of UnitDropout. Requires n_neurons=False "
        "and no padding. Set by the eval jobs for full-neuron evaluation.",
    )

    # Use train data for validation
    use_train_for_val: bool = Field(
        default=False,
        description="If True, uses the training dataset and sampler for validation. "
        "Useful for overfitting tests where you want to validate on the same data.",
    )

    # Deterministic transforms control
    deterministic_transforms: bool = Field(
        default=False,
        description="If True, disables randomness in transforms. "
        "UnitDropout uses deterministic selection (first N neurons) and "
        "ShufflePopulation becomes a passthrough.",
    )

    # Train shuffle control
    shuffle_train: bool = Field(
        default=True,
        description="If True, shuffles the training dataset using RandomFixedWindowSampler. "
        "If False, uses SequentialFixedWindowSampler for deterministic ordering.",
    )

    # Data leakage check control
    disable_data_leakage_check: bool = Field(
        default=False,
        description="If True, disables the data leakage check in torch_brain Dataset. "
        "WARNING: Only enable this if you are absolutely certain there is no data leakage "
        "between splits. This is useful for debugging or when using custom split logic "
        "that the leakage checker doesn't understand.",
    )

    max_neurons: int | None = Field(
        default=None,
        description="Maximum neurons per binned batch element after transform construction. "
        "Samples with more neurons are split during collation to bound tokenizer memory.",
    )

    # Held-out data guard (see ntt.core.data.components.holdout_guard)
    allow_holdout_in_train: bool = Field(
        default=False,
        description="If False (default), setting up the train split raises an error when it "
        "contains a held-out session, a session of a held-out animal, or a held-out dataset "
        "(config/data/selections/composite/ntt_full_{val,test}_*.yaml). Set True ONLY for "
        "runs that are not tokenizer training, e.g. downstream decoding on held-out sessions.",
    )

    # Deep unit-index validation (expensive; exporter-correctness check)
    deep_validate_units: bool = Field(
        default=False,
        description="If True, `NTTDataset` will verify that "
        "`spikes.unit_index.max() < len(units)` for every recording at init time. "
        "This materializes the full per-recording `unit_index` array from disk and "
        "can add minutes of network file system I/O on large multi-session splits (notably the "
        "Neuropixels brainsets). Keep off for training; enable in CI / post-export "
        "smoke tests.",
    )

    @model_validator(mode="after")
    def apply_conditional_defaults(self) -> "NTTDataModuleConfig":
        """Apply conditional default values based on other fields.

        - val_selections defaults to train_selections if not provided
        - test_selections defaults to val_selections (after it's potentially set) if not provided
        - If tokenizer_spec is provided and tokenizer fields are None, populate from spec
        - If pad_neurons_to_fixed is None and tokenizer_spec has a fixed n_neurons, infer it
        """
        # Default val_selections to train_selections
        if self.val_selections is None:
            self.val_selections = self.train_selections

        # Default test_selections to val_selections
        if self.test_selections is None:
            self.test_selections = self.val_selections

        # Populate tokenizer fields from tokenizer_spec if spec is provided
        if self.tokenizer_spec is not None:
            self.update_from_tokenizer_spec(self.tokenizer_spec, unset_only=True)

            # Infer pad_neurons_to_fixed from tokenizer_spec.n_neurons if not explicitly set
            # Only infer if tokenizer spec has a fixed int n_neurons (not Dist or None)
            if self.pad_neurons_to_fixed is None:
                spec_n_neurons = getattr(self.tokenizer_spec, "n_neurons", None)
                if isinstance(spec_n_neurons, int):
                    self.pad_neurons_to_fixed = spec_n_neurons

        return self

    @property
    def bin_size_seconds(self) -> float | Literal[False] | None:
        """Get bin size in seconds (or False for unbinned)."""
        if isinstance(self.bin_size_ms, float):
            return self.bin_size_ms / 1000.0
        return self.bin_size_ms

    @field_serializer("train_transforms", "val_transforms", "test_transforms")
    def serialize_transforms(self, v: list[Callable] | None) -> list[str] | None:
        """TODO: Fix this to something that can reconstruct the transforms"""
        if v is None:
            return None
        return [repr(fn) for fn in v]

    def get_selections(self, split: Split) -> list[SelectionConfig]:
        """Get the selections for the specified data split.

        Args:
            split: One of 'train', 'val', or 'test'
        Returns:
            List of selection configs for the specified split
        """
        check_valid_split(split)
        return getattr(self, f"{split}_selections")

    def get_transforms(self, split: Split) -> list[Callable]:
        """Get the transforms for the specified data split.

        Args:
            split: One of 'train', 'val', or 'test'
        Returns:
            List of Callable for the specified split
        """
        check_valid_split(split)
        return getattr(self, f"{split}_transforms")

    def selections_to_omegaconf(self, split: Split) -> ListConfig:
        selections = self.get_selections(split)
        return ListConfig([s.model_dump() for s in selections])

    @property
    def tokenizer_fields(self) -> dict[str, Any]:
        """Get a dictionary of all tokenizer fields and their current values.

        Returns:
            Dictionary mapping tokenizer field names to their values.
        """
        return fields_by_json_schema_extra(self, "is_tokenizer_field")

    def update_from_tokenizer_spec(self, spec: TokenizerSpec, unset_only: bool = False) -> None:
        """Update this config from a TokenizerSpec instance.

        Sets `self.tokenizer_spec = spec` and overrides ALL tokenizer field values
        from the spec. For fields that can be disabled (n_neurons, bin_size_ms),
        the spec's None value is converted to False (disabled) since None in our
        config means "not set".

        Args:
            spec: TokenizerSpec instance to update from.
        """
        self.tokenizer_spec = spec
        for field_name, old_value in self.tokenizer_fields.items():
            if old_value is not None and unset_only:
                continue  # Skip already set fields
            if not hasattr(spec, field_name):
                raise ValueError(
                    f"TokenizerSpec of type {type(spec).__name__} "
                    f"does not have field '{field_name}' required by NTTDataModuleConfig."
                )
            new_value = getattr(spec, field_name)
            setattr(self, field_name, new_value)

        # Infer pad_neurons_to_fixed from spec if not explicitly set
        if self.pad_neurons_to_fixed is None:
            spec_n_neurons = getattr(spec, "n_neurons", None)
            if isinstance(spec_n_neurons, int):
                self.pad_neurons_to_fixed = spec_n_neurons

    def validate_tokenizer_fields_set(self) -> None:
        """Check if all tokenizer fields are set (not None).

        Returns:
            None
        Raises:
            ValueError: If any tokenizer field is still None.
        """
        unset_fields = [fn for fn, v in self.tokenizer_fields.items() if v is None]
        if unset_fields:
            raise ValueError(
                f"Tokenizer fields not fully set. Missing: {', '.join(unset_fields)}. "
                f"Either provide a tokenizer_spec or set all tokenizer fields in config."
            )


class NTTDataModule(L.LightningDataModule):
    """Lightning DataModule for NTT neural data processing.

    This module handles:
    - Dataset creation from brainset selections
    - Transform pipelines based on tokenizer spec or direct config fields
    - Samplers for windowed neural data
    - DataLoader configuration

    The module uses the following config fields (can be set directly or via tokenizer_spec):
    - window_length_seconds: Window length for sampling
    - bin_size_ms: Bin size for spike binning (False to disable)
    - n_neurons: Number of neurons to sample (False to disable)
    - input_type: Input type (binary_spikes, rates, spike_times)

    DataLoader settings (batch_size, num_workers, etc.) are configured via
    the `dataloader` field in NTTDataModuleConfig.

    TODO: Verify loading config from checkpoint
    """

    def __init__(
        self,
        config: NTTDataModuleConfig,
        tokenizer_spec: TokenizerSpec | None = None,
    ):
        """Initialize the NTTDataModule.

        Args:
            config: NTTDataModuleConfig containing data paths, selections,
                and dataloader settings.
            tokenizer_spec: Optional TokenizerSpec to update config from.
                If provided, overrides all tokenizer fields in config.
        """
        super().__init__()

        # If tokenizer_spec is provided, update config from it
        if tokenizer_spec is not None:
            config.update_from_tokenizer_spec(tokenizer_spec)
        # Verify all tokenizer fields are set
        config.validate_tokenizer_fields_set()

        # Store hyperparameters (exclude tokenizer_spec to avoid hparams
        # merge conflict with LitModule which also saves tokenizer_spec;
        # the spec's fields are already copied into config).
        self.save_hyperparameters(ignore=["tokenizer_spec"])
        self._config = config

        # Store the full tokenizer spec so _build_transforms can access
        # tokenizer-specific fields (e.g. patch_length for VariableWindowCrop).
        # This is separate from config.tokenizer_spec to avoid hparams conflicts
        # with Lightning's LightningModule which also stores the spec.
        self._tokenizer_spec = tokenizer_spec or config.tokenizer_spec

        # Dataset and sampler attributes (set during setup)
        self.train_dataset: NTTDataset | None = None
        self.val_dataset: NTTDataset | None = None
        self.test_dataset: NTTDataset | None = None
        self.train_sampler: Sampler | None = None
        self.val_sampler: Sampler | None = None
        self.test_sampler: Sampler | None = None

    # Selection / transform lists are thousands of characters long, identical
    # across every run of a sweep, and would each be clipped to 250 chars
    # anyway; drop them so the knobs a sweep actually varies stay readable.
    HPARAM_LOG_EXCLUDE: ClassVar[set[str]] = {
        "train_selections",
        "val_selections",
        "test_selections",
        "train_transforms",
        "val_transforms",
        "test_transforms",
        "tokenizer_spec",
    }

    @property
    def hparams_initial(self) -> AttributeDict:  # type: ignore[override]
        """Hyperparameters as seen by the loggers, one param per config field.

        Mirrors ``NTTLitModule.hparams_initial``: loggers read
        ``hparams_initial`` while checkpoints read ``hparams``, so expanding
        the config here does not affect the ``NTTDataModuleConfig(**self.hparams)``
        round-trip in :attr:`config`.
        """
        params = dict(super().hparams_initial)
        config = params.get("config")
        if isinstance(config, BaseModel):
            params["config"] = loggable_flat_dump(config, exclude=self.HPARAM_LOG_EXCLUDE)
        return AttributeDict(params)

    @classmethod
    def from_config(cls, **kwargs) -> "NTTDataModule":
        """Factory method for Hydra instantiation."""
        config = NTTDataModuleConfig(**kwargs)
        return cls(config=config)

    @property
    def config(self) -> NTTDataModuleConfig:
        """Reconstruct config from saved hyperparameters (cached).

        TODO: Verify loading from checkpoint using hparams
        TODO: Figure out how to deal with transforms fields (i.e.
          list of non-serializeable Callable)
        """
        if not hasattr(self, "_config"):
            self._config = NTTDataModuleConfig(**self.hparams)
        return self._config

    def get_dataset(self, split: Split) -> Dataset | None:
        check_valid_split(split)
        return getattr(self, f"{split}_dataset")

    def get_sampler(self, split: Split) -> Sampler | None:
        check_valid_split(split)
        return getattr(self, f"{split}_sampler")

    def check_setup(self, split: Split, should_raise: bool = False) -> bool:
        """Check if the dataset and sampler for the given split are set up."""
        if not (flag := (self.get_dataset(split) is not None)):
            msg = f"{split} dataset is `None`. Call `setup()` first!"
            if should_raise:
                raise RuntimeError(msg)
            log.warning(msg)
        return flag

    @property
    def n_neurons_to_sample(self) -> int | float | Dist | None:
        """Number of neurons sampled per batch.

        This is the number of neurons that will be in each batch after subsampling.
        Returns None if disabled (False) or the actual value if set.
        """
        value = self.config.n_neurons
        # The config uses False for "disabled"; the property exposes it as None.
        if value is False:
            return None
        return value

    @property
    def window_length(self) -> float:
        """Window length for the sampler (always a concrete float).

        When window_length_seconds is a Dist, returns Dist.max so the sampler
        fetches full-length windows. The VariableWindowCrop transform then
        randomly shortens each sample during training.
        """
        wl = self.config.window_length_seconds
        if isinstance(wl, Dist):
            return float(wl.max)
        return wl  # type: ignore[return-value]

    @property
    def bin_size_ms(self) -> float | None:
        """Bin size in milliseconds from config.

        Returns None if disabled (False) or the actual value if set.
        """
        value = self.config.bin_size_ms
        # The config uses False for "disabled"; the property exposes it as None.
        if value is False:
            return None
        return value

    @property
    def input_type(self) -> DataType:
        """Input type from config."""
        # input_type should never be None at this point (validated in __init__)
        return self.config.input_type  # type: ignore[return-value]

    @property
    def pad_neurons_to_fixed(self) -> int | None:
        """Target neuron count for padding.

        Returns None if disabled (False or None) or the target count if set.
        """
        value = self.config.pad_neurons_to_fixed
        if value is False or value is None:
            return None
        return value

    def _build_transforms(self, split: Split) -> Compose:
        """Build the transform pipeline based on tokenizer spec.

        The pipeline follows the order:
        1. UnitDropout (if n_neurons is specified)
        2. CreateBinnedSpikeCountsArray (if bin_size is specified)
        3. PadNeuronsToFixed (if pad_neurons_to_fixed is enabled)
        4. DataToModelInput (converts to model-expected format)
        4.b - Behavioral data extraction (for val and test splits)
        5. Additional user-specified transforms

        Args:
            additional_transforms: Optional list of additional transforms to append.

        Returns:
            Composed transform pipeline.
        """
        transforms: list[Callable] = []

        # 0. Variable window crop (train only, when using Dist window length)
        if split == "train" and isinstance(self.config.window_length_seconds, Dist):
            wl_dist = self.config.window_length_seconds
            bin_size = self.bin_size_ms
            assert bin_size is not None, (
                "bin_size_ms must be set when using variable window cropping"
            )
            spec = self._tokenizer_spec
            patch_length = getattr(spec, "patch_length", None) if spec else None
            if patch_length is None:
                # Tokenizers without a patch concept (e.g. Perceiver) only need
                # whole-bin alignment; default to 1 so crops snap to single bins.
                patch_length = 1
                log.info(
                    "tokenizer_spec has no patch_length; defaulting VariableWindowCrop "
                    "alignment to 1 bin (%.1f ms)",
                    bin_size,
                )
            transforms.append(
                VariableWindowCrop(
                    dist=wl_dist,
                    bin_size_ms=bin_size,
                    patch_length=patch_length,
                )
            )

        # 1. Unit selection: the eval jobs' shared selection, else unit dropout
        eval_unit_selection = self.config.eval_unit_selection
        if eval_unit_selection is not None and split in ("val", "test"):
            if self.n_neurons_to_sample is not None or self.pad_neurons_to_fixed is not None:
                raise ValueError(
                    "eval_unit_selection requires n_neurons=False and pad_neurons_to_fixed=False "
                    f"(got n_neurons={self.config.n_neurons!r}, "
                    f"pad_neurons_to_fixed={self.config.pad_neurons_to_fixed!r})."
                )
            transforms.append(EvalUnitSelection(eval_unit_selection))
        elif (n_neurons := self.n_neurons_to_sample) is not None:
            transforms.append(
                UnitDropout(
                    neuron_sel=n_neurons,
                    max_neurons=self.config.max_neurons,
                    reset_index=True,
                    exclude_zero=self.config.unit_dropout_exclude_zero,
                    sampling_strategy=self.config.unit_dropout_sampling_strategy,
                    sampling_temperature=self.config.unit_dropout_sampling_temperature,
                    sampling_weight_minimum=self.config.unit_dropout_sampling_weight_minimum,
                    sampling_use_global_stats=self.config.unit_dropout_sampling_use_global_stats,
                    seed=self.config.seed,
                    deterministic=(
                        self.config.deterministic_transforms or split in ("val", "test")
                    ),
                )
            )

        # 2. Bin spikes if bin_size is specified
        if self.bin_size_ms is not None:
            transforms.append(CreateBinnedSpikeCountsArray(bin_size_ms=self.bin_size_ms))

        # 3. Pad neurons to fixed size if enabled
        # This ensures models requiring fixed neuron counts work across sessions
        if (pad_target := self.pad_neurons_to_fixed) is not None:
            transforms.append(PadNeuronsToFixed(target_n_neurons=pad_target))

        # 4. Convert to model input format
        tokenizer_spec = self._tokenizer_spec or self.config.tokenizer_spec
        binarize_spikes = tokenizer_spec.binarize_spikes if tokenizer_spec is not None else True
        transforms.append(
            DataToModelInput(
                mode=self.input_type,
                extract_behavior=split in ("val", "test"),
                binarize_spikes=binarize_spikes,
            )
        )

        # 5. Add any additional transforms
        additional_transforms = self.config.get_transforms(split)
        if additional_transforms:
            transforms.extend(additional_transforms)

        return Compose(transforms)

    def _setup_split(
        self,
        split: Split,
        shuffle: bool = False,
    ) -> tuple[NTTDataset, Sampler]:
        # Note: :py:func:`temporaldata.Dataset.set_valid_domain` uses 'valid' instead of 'val'
        log.info(f"Setting up {split} dataset and sampler...")
        dataset_split = "valid" if split == "val" else split
        dataset = NTTDataset(
            root=str(self.config.root),
            split=dataset_split,
            config=self.config.selections_to_omegaconf(split),  # type: ignore
            transform=self._build_transforms(split),
            deep_validate_units=self.config.deep_validate_units,
        )

        # Disable data leakage check if configured
        if self.config.disable_data_leakage_check:
            dataset.disable_data_leakage_check()

        sampling_intervals = dataset.get_sampling_intervals()
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

    def check_train_holdout_leaks(self, train_dataset: Dataset) -> None:
        """Raise if the train split uses held-out sessions, animals or datasets."""
        if self.config.allow_holdout_in_train:
            log.warning("allow_holdout_in_train=True: NOT checking train data for held-out data.")
            return
        check_no_holdout_in_train(train_dataset)

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets and samplers for the specified stage.

        This method is called by Lightning before training/validation/testing.
        It creates the appropriate datasets, transforms, and samplers.

        Args:
            stage: One of 'fit', 'validate', 'test', or 'predict'.
                   If None, sets up all stages.
        """
        if (
            stage in ("fit", None) or (self.config.use_train_for_val and stage == "validate")
        ) and self.train_dataset is None:
            self.train_dataset, self.train_sampler = self._setup_split(
                "train", shuffle=self.config.shuffle_train
            )
            self.check_train_holdout_leaks(self.train_dataset)

        if stage in ("validate", "fit", None) and self.val_dataset is None:
            if self.config.use_train_for_val:
                self.val_dataset = self.train_dataset
                self.val_sampler = self.train_sampler
            else:
                self.val_dataset, self.val_sampler = self._setup_split("val")

        if stage in ("test", None) and self.test_dataset is None:
            self.test_dataset, self.test_sampler = self._setup_split("test")

    def _build_dataloader(
        self,
        split: Split,
        **kwargs,
    ) -> DataLoader:
        """Build a DataLoader for the specified split.

        Args:
            split: One of 'train', 'val', or 'test'.
            **kwargs: Additional DataLoader keyword arguments to override config.
        Returns:
            Configured DataLoader.
        """
        log.info(f"Building {split} dataloader...")
        self.check_setup(split, should_raise=True)
        dataset = self.get_dataset(split)
        dl_kwargs = self.config.dataloader.model_dump()
        dl_kwargs.update(kwargs)
        return DataLoader(
            dataset=dataset,  # type: ignore
            sampler=self.get_sampler(split),
            collate_fn=make_neuron_split_collate_fn(self.config.max_neurons),
            **dl_kwargs,
        )

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader."""
        return self._build_dataloader("train")

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader."""
        return self._build_dataloader("val")

    def test_dataloader(self) -> DataLoader:
        """Return the test DataLoader."""
        return self._build_dataloader("test", drop_last=False)

    def teardown(self, stage: str | None = None) -> None:
        """Clean up after training/validation/testing.

        Args:
            stage: The stage being torn down.
        """
        # Currently no cleanup needed, but can be extended
        pass
