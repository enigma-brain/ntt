"""Brain region label extraction from brainsets.

Loads per-unit brain region metadata from brainsets HDF5 files for probe
evaluation.

Supported brainsets:
- IBL_brainwide_2023: uses ``units.cosmos_location`` field (pre-mapped
  to 10 coarse regions by the IBL brainsets export).
- allen_visual_coding_neuropixels: uses ``units.structure_acronym`` field
  with manual mapping to coarse regions.

Region mappings adapted from the NuCLR reference implementation, used under
the Apache-2.0 license: https://github.com/nerdslab/nuclr (Arora et al., 2025,
NeurIPS; https://arxiv.org/abs/2512.01199).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Target regions (coarse Allen CCF groupings)
# ---------------------------------------------------------------------------
IBL_TARGET_REGIONS = {
    "CB",
    "CNU",
    "CTXsp",
    "HB",
    "HPF",
    "HY",
    "Isocortex",
    "MB",
    "OLF",
    "TH",
}

# Values in cosmos_location that are not valid brain regions
_IBL_EXCLUDED_COSMOS = {"root", "void", ""}

# ---------------------------------------------------------------------------
# Allen Visual Coding region mapping: structure_acronym → coarse region
# Ported from nuclr/src/evaluators/allen_vc_2019.py ``structure2region``.
# ---------------------------------------------------------------------------
_ALLEN_VC_STRUCTURE_MAP: dict[str, str] = {
    # Thalamus
    "IntG": "TH",
    "POL": "TH",
    "PIL": "TH",
    "LGd": "TH",
    "LP": "TH",
    "LD": "TH",
    "LGv": "TH",
    "VPL": "TH",
    "VPM": "TH",
    "PoT": "TH",
    "TH": "TH",
    "MGv": "TH",
    "PP": "TH",
    "MGm": "TH",
    "IGL": "TH",
    "SGN": "TH",
    "PO": "TH",
    "RT": "TH",
    "MGd": "TH",
    # Visual areas
    "VISmma": "VIS",
    "VIS": "VIS",
    "VISam": "VIS",
    "VISl": "VIS",
    "VISal": "VIS",
    "VISrl": "VIS",
    "VISli": "VIS",
    "VISmmp": "VIS",
    "VISp": "VIS",
    "VISpm": "VIS",
    # Midbrain
    "MB": "MB",
    "APN": "MB",
    "MRN": "MB",
    "PPT": "MB",
    "NOT": "MB",
    "ZI": "MB",
    "Eth": "MB",
    "SCig": "MB",
    "OP": "MB",
    "SCiw": "MB",
    "SCop": "MB",
    "SCsg": "MB",
    "LT": "MB",
    # Hippocampus
    "HPF": "HPF",
    "CA2": "HPF",
    "DG": "HPF",
    "CA1": "HPF",
    "SUB": "HPF",
    "CA3": "HPF",
    "ProS": "HPF",
    "PRE": "HPF",
    "POST": "HPF",
    # Remove
    "VL": "Remove",
    "grey": "Remove",
}

# Allen VC fine-grained: banned regions with insufficient labels
_ALLEN_VC_BANNED_REGIONS = {"VIS", "VISmma", "VISli", "VISmmp"}


def _map_allen_vc_region(acronym: str) -> str | None:
    """Map Allen VC structure_acronym to coarse target region.

    Returns None if the region should be excluded.
    """
    region = _ALLEN_VC_STRUCTURE_MAP.get(acronym)
    if region is None:
        logger.warning("Unknown Allen VC structure acronym: %s", acronym)
        return None
    if region == "Remove":
        return None
    return region


def _discover_h5_sessions(
    brainset_dir: Path,
    sessions: list[str] | None = None,
) -> list[tuple[str, Path]]:
    """Find HDF5 session files in a brainset directory.

    Returns list of (session_id, h5_path) tuples.
    """
    h5_files = sorted(brainset_dir.glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No HDF5 files found in {brainset_dir}")

    result = []
    sessions_set = set(sessions) if sessions is not None else None
    for h5_path in h5_files:
        session_id = h5_path.stem
        if sessions_set is not None and session_id not in sessions_set:
            continue
        result.append((session_id, h5_path))

    return result


def _read_units_field(h5_path: Path, field: str) -> np.ndarray:
    """Read a single field from the units group in an HDF5 file.

    Byte-string arrays are decoded to Python str automatically.
    """
    with h5py.File(h5_path, "r") as f:
        if "units" not in f or field not in cast(h5py.Group, f["units"]):
            raise KeyError(f"Field 'units/{field}' not found in {h5_path}")
        data = cast(h5py.Dataset, cast(h5py.Group, f["units"])[field])[:]
        # HDF5 stores strings as bytes — decode to str
        if data.dtype.kind in ("S", "O"):
            data = np.array([x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in data])
        return data


def _units_has_field(h5_path: Path, field: str) -> bool:
    """Check if a field exists in the units group."""
    with h5py.File(h5_path, "r") as f:
        return "units" in f and field in cast(h5py.Group, f["units"])


def _assert_sequential_unit_numbers(
    h5_path: Path,
    session_id: str,
    expected_len: int,
) -> None:
    """Assert ``units/unit_number`` is a contiguous 0..N-1 sequence.

    The probe composite-key contract requires the H5's ``unit_number`` to
    match ``arange(N)`` so that callback-side keys (built from
    ``data.units.unit_number``) align with label-side keys (built from
    ``enumerate(regions)``). Raises ``ValueError`` on any mismatch.
    """
    if not _units_has_field(h5_path, "unit_number"):
        raise ValueError(
            f"{session_id}: 'units/unit_number' field is missing in {h5_path}; "
            f"probe composite-key contract requires it."
        )
    unit_numbers = _read_units_field(h5_path, "unit_number").astype(np.int64)
    expected = np.arange(expected_len, dtype=np.int64)
    if unit_numbers.shape != expected.shape or not np.array_equal(unit_numbers, expected):
        raise ValueError(
            f"{session_id}: 'units/unit_number' is not a contiguous 0..N-1 "
            f"sequence aligned with the regions field "
            f"(len={unit_numbers.size}, expected={expected_len}). "
            f"Probe composite keys would be inconsistent with the eval "
            f"callback; re-export the brainset or update the label builder."
        )


def get_unit_labels(
    data_root: str | Path,
    brainset: str,
    sessions: list[str] | None = None,
) -> pd.DataFrame:
    """Load unit brain region labels from brainsets HDF5 files.

    Args:
        data_root: Root directory containing brainsets.
        brainset: Brainset name (e.g. ``"IBL_brainwide_2023"``).
        sessions: Optional list of session IDs to include.

    Returns:
        DataFrame indexed by ``unit_id`` (str) with columns:
            - ``brain_region``: Coarse target region string
            - ``session_id``: Session identifier string
    """
    data_root = Path(data_root)
    brainset_dir = data_root / brainset
    session_files = _discover_h5_sessions(brainset_dir, sessions)
    logger.info("Found %d session files in %s", len(session_files), brainset)

    rows: list[dict[str, str]] = []
    for session_id, h5_path in session_files:
        regions, map_fn = _get_region_array(brainset, h5_path, session_id)
        if regions is None:
            continue
        assert map_fn is not None

        # Get unit IDs
        if _units_has_field(h5_path, "id"):
            unit_ids = _read_units_field(h5_path, "id")
        elif _units_has_field(h5_path, "unit_id"):
            unit_ids = _read_units_field(h5_path, "unit_id")
        else:
            logger.warning("No unit ID field found for %s", session_id)
            continue

        for uid, region_raw in zip(unit_ids, regions, strict=False):
            mapped = map_fn(str(region_raw))
            if mapped is not None:
                rows.append(
                    {
                        "unit_id": str(uid),
                        "brain_region": mapped,
                        "session_id": session_id,
                    }
                )

    if not rows:
        raise ValueError(f"No units with valid brain region labels found in {brainset}")

    df = pd.DataFrame(rows).set_index("unit_id")
    logger.info(
        "Loaded %d units across %d regions from %s",
        len(df),
        df["brain_region"].nunique(),
        brainset,
    )
    return df


def get_unit_number_to_label(
    data_root: str | Path,
    brainset: str,
    sessions: list[str] | None = None,
) -> pd.DataFrame:
    """Build label table indexed by composite key ``session_id/unit_number``.

    This is used by the online callback where unit IDs are the sequential
    ``unit_number`` assigned by the NTT data pipeline (not brainset IDs).

    Args:
        data_root: Root directory containing brainsets.
        brainset: Brainset name.
        sessions: Optional list of session IDs.

    Returns:
        DataFrame indexed by composite key with columns:
            - ``brain_region``: Coarse target region string
            - ``session_id``: Session identifier string
    """
    data_root = Path(data_root)
    brainset_dir = data_root / brainset
    session_files = _discover_h5_sessions(brainset_dir, sessions)

    rows: list[dict[str, str]] = []
    for session_id, h5_path in session_files:
        regions, map_fn = _get_region_array(brainset, h5_path, session_id)
        if regions is None:
            continue
        assert map_fn is not None

        # Composite keys here use `enumerate(regions)` (0..N-1) as the unit
        # index, but the eval callback looks them up using
        # `data.units.unit_number` from the brainsets pipeline. These only
        # agree if the H5 export stores `unit_number` as a contiguous
        # 0..N-1 sequence in the same order as the regions field. Fail loud
        # if a future export breaks that contract.
        _assert_sequential_unit_numbers(h5_path, session_id, len(regions))

        for unit_num, region_raw in enumerate(regions):
            mapped = map_fn(str(region_raw))
            if mapped is not None:
                rows.append(
                    {
                        "composite_key": f"{session_id}/{unit_num}",
                        "brain_region": mapped,
                        "session_id": session_id,
                    }
                )

    if not rows:
        raise ValueError(f"No units with valid brain region labels found in {brainset}")

    df = pd.DataFrame(rows).set_index("composite_key")
    logger.info(
        "Built composite label table: %d units across %d regions from %s",
        len(df),
        df["brain_region"].nunique(),
        brainset,
    )
    return df


def _get_region_array(
    brainset: str,
    h5_path: Path,
    session_id: str,
) -> tuple[np.ndarray | None, Callable[[str], str | None] | None]:
    """Get region labels and mapping function for a session.

    Returns (regions_array, map_fn) or (None, None) if unavailable.
    """
    if brainset == "IBL_brainwide_2023":
        if not _units_has_field(h5_path, "cosmos_location"):
            logger.warning("Missing cosmos_location in %s", session_id)
            return None, None
        regions = _read_units_field(h5_path, "cosmos_location")
        return regions, _map_ibl_cosmos
    elif brainset == "allen_visual_coding_neuropixels":
        if not _units_has_field(h5_path, "structure_acronym"):
            return None, None
        regions = _read_units_field(h5_path, "structure_acronym")
        return regions, _map_allen_vc_region
    else:
        raise ValueError(f"Unsupported brainset: {brainset}")


def _map_ibl_cosmos(region: str) -> str | None:
    """Filter cosmos_location values to valid target regions."""
    if region in _IBL_EXCLUDED_COSMOS or region not in IBL_TARGET_REGIONS:
        return None
    return region
