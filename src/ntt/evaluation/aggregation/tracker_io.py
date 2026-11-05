"""Shared tracker-IO layer: sheet/CSV reading, run-URL parsing, artifact-URI
builders, and artifact loaders. Used by the aggregation batch runner and the
metrics-table tool so the input plumbing lives in exactly one place.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path

import mlflow
import mlflow.artifacts

from ntt.evaluation.aggregation.downstream_records import DECODING_RECORD, ENCODING_RECORD

# ── Tracker source (local CSV or Google Sheets) ─────────────────────────────────
_GSHEETS_RE = re.compile(r"https://docs\.google\.com/spreadsheets/d/([^/]+)")
_GSHEETS_GID_RE = re.compile(r"[#?&]gid=(\d+)")
_RUN_URL_RE = re.compile(r"/experiments/(\d+)/runs/([0-9a-f]+)")


def is_url(s: str) -> bool:
    return s.startswith(("http://", "https://"))


def to_csv_export_url(url: str) -> str:
    """Convert a Google-Sheets share URL to its CSV-export URL; pass others through."""
    m = _GSHEETS_RE.match(url)
    if not m:
        return url
    sheet_id = m.group(1)
    gid_m = _GSHEETS_GID_RE.search(url)
    export = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    if gid_m:
        export += f"&gid={gid_m.group(1)}"
    return export


def scores_slug(tokenizer: str) -> str:
    """Filename stem for one tokenizer's ``category_scores.json``.

    ``build_aggregation_tables.py`` names the files it persists with this, and
    ``scripts/analysis/data.py`` resolves them with it. Two copies of the rule would
    drift the moment a tokenizer name carried a capital or a dot, and the failure is
    silent — the reader just finds nothing and skips the row.
    """
    return re.sub(r"[^0-9A-Za-z]+", "_", tokenizer).strip("_").lower()


def read_tracker_rows(source: str) -> tuple[list[str], list[dict[str, str]]]:
    """Return (fieldnames, rows) from a local path or Google-Sheets/HTTP URL."""
    source = to_csv_export_url(source)
    if not is_url(source):
        with open(source, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            rows = [dict(r) for r in reader]
        return fieldnames, rows
    with urllib.request.urlopen(source) as resp:
        text = resp.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    return list(reader.fieldnames or []), [dict(r) for r in reader]


# ── Run-URL / artifact-URI helpers ──────────────────────────────────────────────
def parse_run_url(url: str) -> tuple[str, str] | None:
    if not url:
        return None
    m = _RUN_URL_RE.search(url)
    return (m.group(1), m.group(2)) if m else None


def run_url_tracking_uri(url: str) -> str | None:
    """The tracking server a run URL (``<server>/#/experiments/<id>/runs/<id>``) names."""
    server, sep, _ = url.partition("/#/")
    return server.rstrip("/") if sep and server else None


def artifact_uri(exp_id: str, run_id: str, filename: str) -> str:
    return f"mlflow-artifacts:/{exp_id}/{run_id}/artifacts/{filename}"


def artifact_cache_key(uri: str) -> str:
    """Identify an artifact together with the server resolving relative MLflow URIs."""
    source = json.dumps([mlflow.get_tracking_uri(), uri])
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def decoding_dir_uri(exp_id: str, run_id: str) -> str:
    return f"mlflow-artifacts:/{exp_id}/{run_id}/artifacts"


_MLFLOW_URI_SCHEMES = ("mlflow-artifacts:", "mlflow:", "runs:")

# What a local directory cell is searched for, per axis, in order of preference. The
# directory is what an eval job wrote (an MLflow run's artifact dir holds the same files):
# the reconstruction ``results.json``, the robustness ``eval_report.json`` (M1/M2), the
# v2 downstream records. The SNR pass writes only ``results.json`` locally, whose
# ``aggregated`` block is the ``eval_report.json`` MLflow holds.
LOCAL_REPORT_NAMES: dict[str, tuple[str, ...]] = {
    "recon": ("results.json",),
    "m1": ("eval_report.json",),
    "m2": ("eval_report.json",),
    "snr": ("eval_report.json", "results.json"),
    "encoding": (ENCODING_RECORD,),
    "decoding": (DECODING_RECORD,),
}


# Below a directory, decoding's record is the benchmark cohort's: every cell of the benchmark
# (``csv_logs/version_0/``) holds a ``kind: "run"`` record of the same name.
_NESTED_PATTERN = {"decoding": f"*__cohort/{DECODING_RECORD}"}


def is_local_source(uri: str) -> bool:
    """Whether a source URI from ``row_to_source_uris`` is a local path, not MLflow."""
    return not uri.startswith(_MLFLOW_URI_SCHEMES)


# Placeholder cells mean "no run yet"; anything else that is not a URL is a local path.
_PLACEHOLDER_CELLS = frozenset({"", "-", "--", "n/a", "na", "none", "null", "nan", "todo", "tbd"})


def _cell_source(
    cell: str, to_uri: Callable[[str, str], str], base_dir: Path | None
) -> str | None:
    """The source a tracker cell names: an MLflow artifact URI for a run link, else the
    absolute path of a local directory or file. ``None`` for an empty cell or a URL that
    is not a run link."""
    cell = cell.strip()
    if ids := parse_run_url(cell):
        return to_uri(*ids)
    if cell.lower() in _PLACEHOLDER_CELLS or is_url(cell) or cell.startswith(_MLFLOW_URI_SCHEMES):
        return None
    path = Path(cell).expanduser()
    if not path.is_absolute():
        path = (base_dir or Path.cwd()) / path
    return os.path.abspath(path)


def resolve_local_artifact(path: str, key: str) -> Path:
    """The file a local tracker cell stands for.

    A file is used as given. A directory is searched for the axis's report
    (``LOCAL_REPORT_NAMES``; decoding below the directory: ``*__cohort/``): the directory
    itself first, else anywhere below it, so a
    job's output dir works as well as the axis's own. Several matches are an error, never
    a pick. The report's side files (``run_meta.json``, ``results_per_session.csv``) are
    read from the directory it sits in.
    """
    target = Path(path)
    if target.is_file():
        return target
    if not target.is_dir():
        raise FileNotFoundError(f"{key}: local path {target} does not exist")
    names = LOCAL_REPORT_NAMES[key]
    for name in names:
        if (target / name).is_file():
            return target / name
    for name in names:
        found = sorted(target.rglob(_NESTED_PATTERN.get(key, name)))
        if len(found) > 1:
            listing = "\n  ".join(str(p) for p in found)
            raise ValueError(
                f"{key}: more than one {name} under {target}; point the cell at one:\n  {listing}"
            )
        if found:
            return found[0]
    raise FileNotFoundError(f"{key}: no {' or '.join(names)} in {target} or below")


def row_to_source_uris(row: dict[str, str], base_dir: Path | None = None) -> dict[str, str] | None:
    """Map a tracker row to its source URIs. recon is required.

    A cell holds an MLflow run link (mapped to the artifact URI) or a local path (its
    absolute path: a relative one resolves against ``base_dir``, default the cwd; whether
    it exists is checked at read time by ``resolve_local_artifact``).
    """

    def report(filename: str) -> Callable[[str, str], str]:
        return lambda exp_id, run_id: artifact_uri(exp_id, run_id, filename)

    def cell(column: str) -> str:
        return row.get(column) or ""

    recon = _cell_source(cell("mlflow_recon"), report("results.json"), base_dir)
    if recon is None:
        return None
    out: dict[str, str] = {"recon": recon}
    # M1/M2 read the raw robustness eval report (de-normalized latent bucket — the
    # bucket applies bounded transforms to raw values, no Poisson reference needed).
    # M2 latent-noise. The tracker column is `mlflow_latent_latent_corruption`; a
    # missing column is indistinguishable from an empty cell here, so an unmatched
    # name drops M2 silently rather than raising.
    # Downstream inputs name the v2 record (encoding) and the benchmark parent's artifact
    # dir (decoding, whose cohort record sits at its root).
    optional = {
        "m1": (cell("mlflow_latent_input_corruption"), report("eval_report.json")),
        "m2": (
            cell("mlflow_latent_latent_corruption"),
            report("eval_report.json"),
        ),
        "snr": (cell("mlflow_latent_snr"), report("eval_report.json")),
        "encoding": (cell("mlflow_encoding"), report(ENCODING_RECORD)),
        "decoding": (cell("mlflow_decoding"), decoding_dir_uri),
    }
    for key, (value, to_uri) in optional.items():
        if source := _cell_source(value, to_uri, base_dir):
            out[key] = source
    return out


# ── Artifact loaders (moved verbatim from jobs/aggregation.py) ───────────────────
def load_json_artifact(path: str) -> dict:
    if not path.startswith(("mlflow-artifacts:", "mlflow:", "runs:")):
        with open(path) as f:
            return json.load(f)
    tmp = tempfile.mkdtemp(prefix="ntt_agg_")
    try:
        local = mlflow.artifacts.download_artifacts(path, dst_path=tmp)
        with open(local) as f:
            return json.load(f)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
