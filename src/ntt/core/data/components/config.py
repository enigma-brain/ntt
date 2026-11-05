from typing import Any

from pydantic import (
    Field,
    field_validator,
    model_validator,
)

from ntt.utils import (
    NTTBaseModel,
    RankedLogger,
    fields_by_json_schema_extra,
)

log = RankedLogger(__name__, rank_zero_only=True)


class BrainsetSelection(NTTBaseModel, strict=False, frozen=True, extra="forbid"):
    """Schema for dataset selection configuration.

    Defines which sessions from which brainsets should be included in a dataset.
    Follows the torchbrain brainsets selection format.

    Example YAML:
        ```yaml
        brainset: BRAINSET_1
        sessions:
            - SESSION_1_A
            - SESSION_1_B
        subjects:
            - SUBJECT_1_A
        exclude_sessions:
            - SESSION_1_C
        ```

    Attributes:
        brainset: Name of the brainset to select from (e.g., 'IBL_brainwide_2023')
        session: Optional single session ID to include
        sessions: Optional list of session IDs to include from this brainset
        subject: Optional single subject ID to restrict the selection
        subjects: Optional list of subject IDs to restrict the selection
        exclude_sessions: Optional list of session IDs to omit

    TODO: Add comprehensive field validators
    """

    brainset: str = Field(
        ...,
        description="Name of the brainset to select from (e.g., 'IBL_brainwide_2023')",
    )
    session: str | None = Field(
        default=None,
        description="Single session ID to include for this brainset",
        json_schema_extra={"is_filter": True},
    )
    sessions: list[str] | None = Field(
        default=None,
        description="List of session IDs to include from this brainset",
        json_schema_extra={"is_filter": True},
    )
    subject: str | None = Field(
        default=None,
        description="Single subject ID to restrict the selection",
        json_schema_extra={"is_filter": True},
    )
    subjects: list[str] | None = Field(
        default=None,
        description="List of subject IDs to restrict the selection",
        json_schema_extra={"is_filter": True},
    )
    exclude_sessions: list[str] | None = Field(
        default=None,
        description="List of session IDs to omit from the selection",
    )

    @field_validator("session", "subject", mode="before")
    @classmethod
    def convert_single_id_to_string(cls, v: Any) -> str | None:
        """Convert numeric IDs to strings (YAML parses numeric-only values as int)."""
        if v is None:
            return None
        return str(v)

    @field_validator("sessions", "subjects", "exclude_sessions", mode="before")
    @classmethod
    def convert_list_ids_to_strings(cls, v: Any) -> list[str] | None:
        """Convert all list elements to strings (YAML parses numeric-only values as int)."""
        if v is None:
            return None
        return [str(item) for item in v]

    @property
    def filter_fields(self) -> dict[str, Any]:
        """Get a dictionary of all filter fields and their current values.

        Returns:
            Dictionary mapping filter field names to their values.
        """
        return fields_by_json_schema_extra(self, "is_filter")

    @model_validator(mode="after")
    def validate_singular_plural_exclusivity(self) -> "BrainsetSelection":
        """Ensure only one of session/sessions and subject/subjects is specified.

        Users should use either the singular or plural form, not both.
        """
        set_filters = [k for k, v in self.filter_fields.items() if v is not None]

        if len(set_filters) > 1:
            raise ValueError(
                f"Only one filter field can be specified at a time. "
                f"Found multiple: {', '.join(repr(f) for f in set_filters)}. "
                f"Available fields: {', '.join(repr(f) for f in self.filter_fields.keys())}."
            )

        return self


class SelectionsModifier(NTTBaseModel, strict=False, frozen=True, extra="forbid"):
    """Configuration options that modify how selections are processed.

    Currently only supports sampling_intervals_modifier, which is DISABLED
    for security reasons.

    Attributes:
        sampling_intervals_modifier: Code string that modifies sampling intervals.
            DISABLED: Must be None due to security concerns (uses exec()).
    """

    sampling_intervals_modifier: str | None = Field(
        default=None,
        description="Code string that modifies sampling intervals. "
        "DISABLED: Must be None due to security concerns.",
    )
    ignore_splits: bool = Field(
        default=False,
        description="If True, sessions will ignore split restrictions and use full domain.",
    )

    @field_validator("sampling_intervals_modifier")
    @classmethod
    def validate_no_code_execution(cls, v: str | None) -> str | None:
        """Ensure sampling_intervals_modifier is None for security.

        The torch_brain library uses exec() on the sampling_intervals_modifier,
        which is a security risk. We disable this feature until a safer
        alternative is implemented.

        See :py:func:`torch_brain.data.dataset.Dataset.get_sampling_intervals`
        """
        if v is not None and v.strip():
            raise ValueError(
                "sampling_intervals_modifier must be None due to security concerns. "
                "The current torch_brain implementation uses exec() which is unsafe. "
                "This feature is disabled until a safer alternative is implemented."
            )
        return None


class SelectionConfig(NTTBaseModel, strict=False, frozen=True, extra="forbid"):
    """Top-level wrapper for selection config for collection of brainsets.

    Example YAML:
        ```yaml
        selection:
          - brainset: BRAINSET_1
            sessions:
              - SESSION_1_A
              - SESSION_1_B
            exclude_sessions:
              - SESSION_1_C
          - brainset: BRAINSET_2
            session: SESSION_2_A
        config:
          # DISABLED for security: sampling_intervals_modifier contains arbitrary code
          # that gets eval'd in torch_brain.data.dataset.Dataset.get_sampling_intervals
          sampling_intervals_modifier: null
        ```

    Attributes:
        selection: List of dataset selections from different brainsets
        config: Optional configuration block for dataset-specific options
    """

    selection: list[BrainsetSelection] = Field(
        ...,
        description="List of dataset selections from different brainsets",
        min_length=1,
    )
    config: SelectionsModifier = Field(
        default_factory=SelectionsModifier,
        description="Optional configuration block for dataset-specific options",
    )


class DataLoaderConfig(NTTBaseModel, strict=False, frozen=True, extra="forbid"):
    """Configuration for PyTorch DataLoader.

    This config handles common DataLoader parameters with sensible defaults.
    The persistent_workers and prefetch_factor fields are automatically
    adjusted based on num_workers if needed.

    Example YAML:
        ```yaml
        dataloader:
          batch_size: 32
          num_workers: 8
          pin_memory: true
        ```

    Attributes:
        batch_size: Number of samples per batch.
        num_workers: Number of worker processes for data loading.
        pin_memory: Whether to use pinned memory for faster GPU transfer.
        drop_last: Whether to drop the last incomplete batch.
        persistent_workers: Whether to keep worker processes alive between epochs.
            Automatically set to False if num_workers == 0.
        prefetch_factor: Number of batches to prefetch per worker.
            Automatically set to None if num_workers == 0.
    """

    batch_size: int = Field(
        default=8,
        description="Number of samples per batch",
        gt=0,
    )
    num_workers: int = Field(
        default=4,
        description="Number of worker processes for data loading",
        ge=0,
    )
    pin_memory: bool = Field(
        default=True,
        description="Whether to use pinned memory for faster GPU transfer",
    )
    drop_last: bool = Field(
        default=True,
        description="Whether to drop the last incomplete batch",
    )
    persistent_workers: bool = Field(
        default=True,
        description="Whether to keep worker processes alive between epochs. "
        "Automatically set to False if num_workers == 0.",
    )
    prefetch_factor: int | None = Field(
        default=2,
        description="Number of batches to prefetch per worker. "
        "Automatically set to None if num_workers == 0.",
        ge=1,
    )

    @model_validator(mode="before")
    @classmethod
    def apply_worker_defaults(cls, data: dict) -> dict:
        """Adjust persistent_workers and prefetch_factor when num_workers == 0.

        Also caps num_workers at the number of available CPU cores to avoid
        the PyTorch DataLoader warning about excessive worker creation.
        """
        if not isinstance(data, dict):
            return data

        num_workers = data.get("num_workers", 4)
        if num_workers == 0:
            data["persistent_workers"] = False
            data["prefetch_factor"] = None
        elif num_workers > 0:
            import os

            try:
                available = len(os.sched_getaffinity(0))
            except AttributeError:
                # sched_getaffinity not available on all platforms (e.g. macOS)
                available = os.cpu_count() or 1
            if num_workers > available:
                data["num_workers"] = available
        return data
