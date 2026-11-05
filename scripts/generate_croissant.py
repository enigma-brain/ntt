"""Generate the Croissant 1.0 metadata manifest for the NTT public dataset.

Reads:
  - croissant/upstream_sources.yaml  (one entry per public brainset)
  - croissant/brainset_summary.csv  (per-brainset summary stats)
  - config/data/selections/composite/ntt_full_{val,test}_{xsession,xsubject}.yaml

Computes:
  - sha256 of every in-repo FileObject (selection YAMLs, brainset summary CSV)

Emits:
  - croissant.json at the repo root.

Validation:
  - uv run --extra dev python -c "import mlcroissant as mlc; mlc.Dataset(jsonld='croissant.json')"
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CROISSANT_DIR = REPO_ROOT / "croissant"
SUMMARY_CSV = CROISSANT_DIR / "brainset_summary.csv"
SPLIT_DIR = REPO_ROOT / "config/data/selections/composite"
SPLIT_YAMLS = [
    SPLIT_DIR / f"ntt_full_{split}_{tier}.yaml"
    for split in ("test", "val")
    for tier in ("xsession", "xsubject")
]

# --------------------------------------------------------------------------- #
# Dataset-level metadata.                                                     #
# --------------------------------------------------------------------------- #
DATASET_NAME = "ntt-public"
DATASET_VERSION = "1.0.0"
DATASET_DATE_PUBLISHED = "2026-11-06"
DATASET_URL = "https://github.com/enigma-brain/ntt"
DATASET_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
PAPER_TITLE = (
    "Evaluating Neural Data Tokenizers: A Framework for Assessing Learned "
    "Representations of Spiking Activity"
)
# Author order as in the paper.
CREATOR_NAMES = [
    "Federico D'Agostino",
    "Alex Gilbert",
    "Susanne Keller",
    "Jaivardhan Kapoor",
    "Nicolas Reategui",
    "Vedang Lad",
    "Matthias Kümmerer",
    "Jai Bhagat",
    "Emin Orhan",
    "Hasan Atakan Bedel",
    "Nikos Karantzas",
    "Fabian H. Sinz",
    "Surya Ganguli",
    "Jakob H. Macke",
    "Sophia Sanborn",
    "Alexander S. Ecker",
    "Katrin Franke",
    "Matthias Bethge",
    "Andreas S. Tolias",
    "Konstantin F. Willeke",
]
_BIBTEX_AUTHORS = {
    "Matthias Kümmerer": 'K{\\"u}mmerer, Matthias',
}


def _bibtex_author(name: str) -> str:
    if name in _BIBTEX_AUTHORS:
        return _BIBTEX_AUTHORS[name]
    *given, family = name.split(" ")
    return f"{family}, {' '.join(given)}"


DATASET_CITATION = (
    "@inproceedings{dagostino2026evaluating,\n"
    f"  title = {{{PAPER_TITLE}}},\n"
    f"  author = {{{' and '.join(_bibtex_author(n) for n in CREATOR_NAMES)}}},\n"
    "  booktitle = {Advances in Neural Information Processing Systems "
    "(Datasets and Benchmarks Track)},\n"
    "  year = {2026},\n"
    f"  url = {{{DATASET_URL}}},\n"
    "}"
)
DATASET_DESCRIPTION = (
    "NTT-public is a composite neural-spike dataset for tokenization research, "
    "spanning 1,104 sessions and ~266k units across 10 publicly-released "
    "brainsets (7 held-in for training and 3 held-out as cross-source "
    "evaluation). The dataset is distributed as a recipe: the NTT repository "
    "ships per-brainset selection YAMLs and a brainsets-based ingestion "
    "pipeline that materializes per-session HDF5 files from canonical public "
    "upstream sources (DANDI, Zenodo, AllenSDK, CRCNS, UCL RDR, Kilthub). "
    "This manifest documents the schema of those HDF5 files and references "
    "the upstream sources, the in-repo split selection YAMLs, and a "
    "per-brainset summary table of the assembled dataset."
)
KEYWORDS = [
    "neuroscience",
    "neural recordings",
    "spike trains",
    "neural population activity",
    "tokenization",
    "neural foundation models",
    "Neuropixels",
    "Utah array",
]

# --------------------------------------------------------------------------- #
# JSON-LD @context — pinned to the mlcroissant 1.0 canonical block.           #
# --------------------------------------------------------------------------- #
CROISSANT_CONTEXT: dict[str, Any] = {
    "@language": "en",
    "@vocab": "https://schema.org/",
    "citeAs": "cr:citeAs",
    "column": "cr:column",
    "conformsTo": "dct:conformsTo",
    "cr": "http://mlcommons.org/croissant/",
    "rai": "http://mlcommons.org/croissant/RAI/",
    "data": {"@id": "cr:data", "@type": "@json"},
    "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
    "dct": "http://purl.org/dc/terms/",
    "equivalentProperty": "cr:equivalentProperty",
    "examples": {"@id": "cr:examples", "@type": "@json"},
    "extract": "cr:extract",
    "field": "cr:field",
    "fileProperty": "cr:fileProperty",
    "fileObject": "cr:fileObject",
    "fileSet": "cr:fileSet",
    "format": "cr:format",
    "includes": "cr:includes",
    "isLiveDataset": "cr:isLiveDataset",
    "jsonPath": "cr:jsonPath",
    "key": "cr:key",
    "md5": "cr:md5",
    "parentField": "cr:parentField",
    "path": "cr:path",
    "recordSet": "cr:recordSet",
    "references": "cr:references",
    "regex": "cr:regex",
    "repeated": "cr:repeated",
    "replace": "cr:replace",
    "samplingRate": "cr:samplingRate",
    "sc": "https://schema.org/",
    "separator": "cr:separator",
    "source": "cr:source",
    "subField": "cr:subField",
    "transform": "cr:transform",
}

# --------------------------------------------------------------------------- #
# Per-session H5 schema — uniform across brainsets with small variations.     #
# Encoded as a Python list because it ties tightly to the brainsets pipeline. #
# --------------------------------------------------------------------------- #
H5_FIELDS: list[dict[str, Any]] = [
    # spikes/* — the spike train itself.
    {
        "name": "spike_timestamps",
        "h5_path": "spikes/timestamps",
        "dataType": "sc:Float",
        "repeated": True,
        "description": "Spike timestamps in seconds. Vector of length N_spikes.",
    },
    {
        "name": "spike_unit_index",
        "h5_path": "spikes/unit_index",
        "dataType": "sc:Integer",
        "repeated": True,
        "description": "Index into units/* for each spike. Vector of length N_spikes.",
    },
    {
        "name": "spike_train_mask",
        "h5_path": "spikes/train_mask",
        "dataType": "sc:Boolean",
        "repeated": True,
        "description": "Train-split membership per spike. Vector of length N_spikes.",
    },
    {
        "name": "spike_valid_mask",
        "h5_path": "spikes/valid_mask",
        "dataType": "sc:Boolean",
        "repeated": True,
        "description": "Validation-split membership per spike. Vector of length N_spikes.",
    },
    {
        "name": "spike_test_mask",
        "h5_path": "spikes/test_mask",
        "dataType": "sc:Boolean",
        "repeated": True,
        "description": "Test-split membership per spike. Vector of length N_spikes.",
    },
    # spikes/domain/* — multi-interval (one entry per trial). MUST BE VECTOR.
    {
        "name": "spike_domain_start",
        "h5_path": "spikes/domain/start",
        "dataType": "sc:Float",
        "repeated": True,
        "description": (
            "Trial-interval starts (seconds). VECTOR of length N_intervals: one entry "
            "per trial for trial-structured brainsets (flint, churchland, perich), one "
            "entry spanning the full session for non-trial-structured brainsets. Misreading "
            "this as a scalar yields a 5-second 'session duration' for trial-structured data."
        ),
    },
    {
        "name": "spike_domain_end",
        "h5_path": "spikes/domain/end",
        "dataType": "sc:Float",
        "repeated": True,
        "description": (
            "Trial-interval ends (seconds). VECTOR of length N_intervals (paired "
            "with spike_domain_start)."
        ),
    },
    # units/* — per-unit metadata.
    {
        "name": "unit_id",
        "h5_path": "units/id",
        "dataType": "sc:Text",
        "repeated": True,
        "description": (
            "Per-unit identifier (bytes-encoded string in the source H5). Vector "
            "of length N_units."
        ),
    },
    {
        "name": "unit_number",
        "h5_path": "units/unit_number",
        "dataType": "sc:Integer",
        "repeated": True,
        "description": "Per-unit integer index. Vector of length N_units.",
    },
    {
        "name": "unit_brain_area",
        "h5_path": "units/<area_key>",
        "dataType": "sc:Text",
        "repeated": True,
        "description": (
            "Per-unit brain area label. The actual H5 key varies by brainset: "
            "area_name (default), structure_acronym (allen_visual_coding_neuropixels), "
            "allen_location | beryl_location | cosmos_location (IBL_brainwide_2023, in "
            "decreasing fineness). Brainsets without per-unit area data fall back to a "
            "default area string per RecordSet (see RecordSet.description). 11 IBL "
            "sessions have empty area strings even when the keys are present, so this "
            "field is nullable."
        ),
    },
    {
        "name": "unit_firing_rate",
        "h5_path": "units/firing_rate",
        "dataType": "sc:Float",
        "repeated": True,
        "description": (
            "Per-unit mean firing rate in Hz. Optional (not all brainsets carry this). Nullable."
        ),
    },
    {
        "name": "unit_contamination",
        "h5_path": "units/contamination",
        "dataType": "sc:Float",
        "repeated": True,
        "description": "Per-unit contamination QC metric (fraction). Optional. Nullable.",
    },
    {
        "name": "unit_isi_violations",
        "h5_path": "units/isi_violations",
        "dataType": "sc:Float",
        "repeated": True,
        "description": "Per-unit ISI-violation rate. Optional. Nullable.",
    },
    {
        "name": "unit_presence_ratio",
        "h5_path": "units/presence_ratio",
        "dataType": "sc:Float",
        "repeated": True,
        "description": (
            "Per-unit presence ratio (fraction of session bins with at least one "
            "spike). Optional. Nullable."
        ),
    },
    # Session-level attrs (h5 attrs, not datasets).
    {
        "name": "session_task",
        "h5_path": "session.attrs.task",
        "dataType": "sc:Text",
        "description": (
            "Behavioral task label (e.g. REACHING). Empty for passive/free-viewing sessions."
        ),
    },
    {
        "name": "device_recording_tech",
        "h5_path": "device.attrs.recording_tech",
        "dataType": "sc:Text",
        "description": (
            "Recording technology used for this session. Values include "
            "NEUROPIXELS_SPIKES, UTAH_ARRAY_SPIKES, UTAH_ARRAY_THRESHOLD_CROSSINGS."
        ),
    },
    {
        "name": "subject_species",
        "h5_path": "subject.attrs.species",
        "dataType": "sc:Text",
        "description": (
            "Subject species (e.g. MUS_MUSCULUS, MACACA_MULATTA, "
            "MACACA_FASCICULARIS). UNKNOWN for sessions without species metadata."
        ),
    },
    {
        "name": "brainset_description",
        "h5_path": "brainset.attrs.description",
        "dataType": "sc:Text",
        "description": "Per-brainset prose description carried in the H5 attrs.",
    },
    {
        "name": "brainset_source",
        "h5_path": "brainset.attrs.source",
        "dataType": "sc:Text",
        "description": (
            "Upstream source as written by the brainset's pipeline: usually the "
            "canonical URL (e.g. https://dandiarchive.org/dandiset/000409); "
            "c4_dataset_cerebellum_2025 stores a free-text label instead."
        ),
    },
    {
        "name": "brainset_origin_version",
        "h5_path": "brainset.attrs.origin_version",
        "dataType": "sc:Text",
        "description": (
            "Version identifier for the upstream source as written by the brainset's "
            "pipeline (e.g. dandi/000409, allen/vbn, 1.0.0). Format varies by brainset."
        ),
    },
    {
        "name": "brainset_derived_version",
        "h5_path": "brainset.attrs.derived_version",
        "dataType": "sc:Text",
        "description": "Version of the brainsets-pipeline output schema (e.g. 1.0.0).",
    },
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def slugify(s: str) -> str:
    return s.replace("_", "-").lower()


def make_in_repo_file_object(
    file_id: str,
    name: str,
    description: str,
    relative_path: str,
    encoding_format: str,
) -> dict[str, Any]:
    """FileObject for a file checked into the repo (selection YAML, EDA parquet/CSV)."""
    abs_path = REPO_ROOT / relative_path
    if not abs_path.exists():
        raise FileNotFoundError(f"missing in-repo file: {abs_path}")
    return {
        "@type": "cr:FileObject",
        "@id": file_id,
        "name": name,
        "description": description,
        "contentUrl": relative_path,
        "encodingFormat": encoding_format,
        "sha256": sha256_of(abs_path),
    }


def make_upstream_distribution(brainset_id: str, info: dict[str, Any]) -> list[dict[str, Any]]:
    """Build a FileSet describing the upstream source.

    Each upstream source is modeled as a FileSet (not FileObject) because:
      - We don't redistribute the bytes; we point at canonical public URLs.
      - FileObject would require sha256/md5, which we cannot compute without
        fetching every upstream asset.
      - FileSet semantically represents "a collection of files at the upstream"
        — a DANDI dandiset, a Zenodo record, or a UCL RDR article each *is* a
        collection of files.
    The upstream URL(s) are encoded in the description (human-readable) and
    in the schema.org `url` field (machine-readable).
    """
    license_url = info.get("license_url")
    license_label = info.get("license", "TODO")
    license_value = license_url if license_url else license_label

    content_urls = info.get("contentUrls")
    urls: list[str]
    if content_urls is not None:
        urls = list(content_urls)
    else:
        single = info.get("contentUrl")
        if single is None:
            raise ValueError(f"{brainset_id}: neither contentUrl nor contentUrls set")
        urls = [single]

    base_desc = info.get("description", "").strip()
    if len(urls) == 1:
        url_text = f"Canonical URL: {urls[0]}"
    else:
        url_list = "\n".join(f"  - {u}" for u in urls)
        url_text = f"Canonical URLs ({len(urls)} files at the upstream):\n{url_list}"

    fileset = {
        "@type": "cr:FileSet",
        "@id": f"upstream-{slugify(brainset_id)}",
        "name": f"{brainset_id}-upstream",
        "description": f"{base_desc}\n\n{url_text}",
        "encodingFormat": "application/x-hdf5"
        if any(u.endswith(".h5") for u in urls)
        else "text/html",
        "license": license_value,
        "url": urls[0] if len(urls) == 1 else urls,
        "includes": "*",
    }
    # schema.org `citation` = works this source should be cited by (BibTeX).
    # Distinct from the dataset-level `citeAs`, which cites NTT itself.
    citation = (info.get("citation") or "").strip()
    if not citation:
        raise ValueError(f"{brainset_id}: missing citation in upstream_sources.yaml")
    fileset["citation"] = citation
    return [fileset]


def make_brainset_fileset(brainset_id: str) -> dict[str, Any]:
    """FileSet describing the *processed* per-session H5 files users obtain by
    running the brainsets pipeline locally. The bytes are not redistributed by
    NTT — the FileSet documents the schema users will read from disk.

    The relationship to the upstream FileSet (`upstream-<brainset>`) is
    implicit (same brainset_id, different pipeline stage). Croissant's
    `containedIn` is reserved for FileObject parents (e.g., archive members),
    which doesn't apply here since both are FileSets.
    """
    return {
        "@type": "cr:FileSet",
        "@id": f"processed-{slugify(brainset_id)}",
        "name": f"{brainset_id}-processed",
        "description": (
            f"Per-session HDF5 files for {brainset_id}, produced by running "
            f"`brainsets prepare brainsets_pipelines/{brainset_id}` against the "
            f"upstream source ({slugify(brainset_id)}-upstream). Files live "
            f"under <DATA_DIR>/{brainset_id}/<session>.h5 on the user's local "
            f"disk after the recipe is run."
        ),
        "encodingFormat": "application/x-hdf5",
        "includes": f"{brainset_id}/*.h5",
    }


def make_brainset_recordset(
    brainset_id: str,
    summary_row: pd.Series,
) -> dict[str, Any]:
    """One RecordSet per brainset, declaring the H5 schema fields."""
    fileset_id = f"processed-{slugify(brainset_id)}"
    rs_id = f"recordset-{slugify(brainset_id)}"

    fields: list[dict[str, Any]] = []
    for spec in H5_FIELDS:
        fields.append(
            {
                "@type": "cr:Field",
                "@id": f"{rs_id}/{spec['name']}",
                "name": spec["name"],
                "description": spec["description"],
                "dataType": spec["dataType"],
                **({"repeated": True} if spec.get("repeated") else {}),
                "source": {
                    "fileSet": {"@id": fileset_id},
                    "extract": {"fileProperty": "fullpath"},
                },
            }
        )

    description = (
        f"{brainset_id}: {int(summary_row['n_sessions'])} sessions, "
        f"{int(summary_row['total_units'])} units, "
        f"{summary_row['total_minutes']:.1f} minutes of recording. "
        f"Species: {summary_row['species']}. "
        f"Recording tech: {summary_row['recording_tech']}. "
        f"Brain areas: {summary_row['brain_areas']}. "
        "Role: "
        + (
            "cross-source held-out evaluation"
            if summary_row["role"] == "xsource"
            else "training + within-source evaluation"
        )
        + "."
    )

    return {
        "@type": "cr:RecordSet",
        "@id": rs_id,
        "name": brainset_id,
        "description": description,
        "field": fields,
    }


def build_manifest() -> dict[str, Any]:
    # Load inputs.
    with (CROISSANT_DIR / "upstream_sources.yaml").open() as f:
        upstream = yaml.safe_load(f)
    summary_df = pd.read_csv(SUMMARY_CSV).set_index("brainset")
    unknown_roles = set(summary_df["role"]) - {"held-in", "xsource"}
    if unknown_roles:
        raise ValueError(f"{SUMMARY_CSV.name}: unknown role(s) {sorted(unknown_roles)}")

    # ----- Distribution: in-repo FileObjects + per-brainset upstream sources + processed
    # FileSets ----- #
    distribution: list[dict[str, Any]] = []

    # In-repo: the four public split YAMLs.
    for split_yaml in SPLIT_YAMLS:
        rel = split_yaml.relative_to(REPO_ROOT).as_posix()
        stem = split_yaml.stem
        distribution.append(
            make_in_repo_file_object(
                file_id=f"selection-{stem}",
                name=stem,
                description=(
                    f"Public split selection YAML ({stem}). Lists the per-brainset "
                    f"sessions that constitute this split. Consumed by NTT's Hydra "
                    f"data-config composition."
                ),
                relative_path=rel,
                encoding_format="application/x-yaml",
            )
        )

    # In-repo: per-brainset summary table.
    distribution.append(
        make_in_repo_file_object(
            file_id="brainset-summary-csv",
            name="brainset_summary",
            description=(
                "Per-brainset summary of the assembled public dataset: role (held-in "
                "or xsource held-out), n_sessions, total_units, total_minutes, "
                "species, recording_tech and brain_areas."
            ),
            relative_path=SUMMARY_CSV.relative_to(REPO_ROOT).as_posix(),
            encoding_format="text/csv",
        )
    )

    # Per-brainset upstream sources + processed FileSets.
    record_sets: list[dict[str, Any]] = []
    for brainset_id, info in upstream.items():
        distribution.extend(make_upstream_distribution(brainset_id, info))
        distribution.append(make_brainset_fileset(brainset_id))
        if brainset_id not in summary_df.index:
            raise KeyError(f"brainset {brainset_id} missing from {SUMMARY_CSV.name}")
        record_sets.append(make_brainset_recordset(brainset_id, summary_df.loc[brainset_id]))

    manifest: dict[str, Any] = {
        "@context": CROISSANT_CONTEXT,
        "@type": "sc:Dataset",
        "name": DATASET_NAME,
        "description": DATASET_DESCRIPTION,
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        "url": DATASET_URL,
        "license": DATASET_LICENSE_URL,
        "creator": [{"@type": "sc:Person", "name": name} for name in CREATOR_NAMES],
        "citeAs": DATASET_CITATION,
        "version": DATASET_VERSION,
        "datePublished": DATASET_DATE_PUBLISHED,
        "keywords": KEYWORDS,
        "isLiveDataset": False,
        "distribution": distribution,
        "recordSet": record_sets,
    }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "croissant.json",
        help="Where to write the manifest (default: croissant.json at repo root).",
    )
    args = parser.parse_args()

    manifest = build_manifest()
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=False, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    print(
        f"  - {len(manifest['distribution'])} distribution entries, "
        f"{len(manifest['recordSet'])} record sets"
    )


if __name__ == "__main__":
    main()
