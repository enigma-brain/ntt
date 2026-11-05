from typing import Literal, get_args

# Standard data splits used throughout the codebase (aligned with PyTorch Lightning)
Split = Literal["train", "val", "test"]
SPLITS: tuple[Split, ...] = get_args(Split)


def check_valid_split(split: str) -> None:
    """Utility function to check if a given split string is valid.

    :param split: The split string to check.
    :raises ValueError: If the split is not one of the valid options.
    """
    if split not in SPLITS:
        raise ValueError(f"Invalid split: {split}. Choose from {SPLITS}.")
