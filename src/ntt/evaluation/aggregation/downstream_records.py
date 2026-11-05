"""Downstream (ntt-downstream) results records, read through one view.

ntt-downstream writes one schema-v2 record per encoding run
(``downstream_encoding_results.json``, ``kind: "run"``) and one cohort record per
decoding benchmark, at the per-tokenizer parent run's artifact root
(``downstream_decoding_results.json``, ``kind: "cohort"``). Downstream's
``docs/results.md`` is the authority on the format.

``load_downstream_record`` turns either into a ``DownstreamScoreView`` of the record's
authoritative ``test`` block; scoring and comparability read only that view.
"""

from __future__ import annotations

import json
import logging
import posixpath
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import mlflow
import mlflow.artifacts

from ntt.evaluation.aggregation.comparability import _finite_float

log = logging.getLogger(__name__)

Task = Literal["encoding", "decoding"]

SCHEMA_VERSION = 2
ENCODING_RECORD = "downstream_encoding_results.json"
DECODING_RECORD = "downstream_decoding_results.json"
_RECORD_FILE: dict[str, str] = {"encoding": ENCODING_RECORD, "decoding": DECODING_RECORD}
# ntt scores encoding per run and decoding per benchmark cohort.
_RECORD_KIND: dict[str, str] = {"encoding": "run", "decoding": "cohort"}
_MLFLOW_SCHEMES = ("mlflow-artifacts:", "mlflow:")


@dataclass(frozen=True)
class DownstreamScoreView:
    """One downstream input, normalized to the block its score is read from.

    - ``split``: the split the score comes from (``"test"``).
    - ``aggregated``: ``{metric: {"value", "std", "n", "aggregated_over", ...}}``.
      Empty means no score: encoding final eval failed or was disabled, or no cohort
      cell has a finite value.
    - ``per_session``: ``{session: {metric: value or None}}``.
    - ``complete``: ``False`` for a decoding cohort with missing cells. Always ``True``
      for encoding.
    - ``provenance``: what the record says about how it was made (tokenizer
      checkpoint, ntt version/path/commit, git sha; encoding ``final_eval``; cohort
      ``expected``/``found``/``missing``/``stale``/``mixed_provenance``).
    """

    split: str
    aggregated: dict[str, dict[str, Any]]
    per_session: dict[str, dict[str, float | None]]
    complete: bool
    provenance: dict[str, Any]

    def session_summary(self, metric: str) -> dict:
        """The sessions behind this view's ``metric``, in ``report_session_summary``'s
        shape: those with a finite value, else (no per-session values) only its count."""
        entry = self.aggregated.get(metric) or {}
        session_ids = (
            sorted(
                session
                for session, values in self.per_session.items()
                if _finite_float(values.get(metric)) is not None
            )
            or None
        )
        return {
            "session_ids": session_ids,
            "n_sessions": len(session_ids) if session_ids else entry.get("n"),
            "aggregation_strategy": entry.get("aggregated_over"),
        }


def load_downstream_record(path: str, task: Task) -> DownstreamScoreView:
    """Read one downstream input into a view.

    ``path`` is a local file or directory, or an MLflow URI: an encoding run's
    ``downstream_encoding_results.json``, or a decoding benchmark parent's artifact dir.
    An MLflow URI is resolved first (``resolve_downstream_uri``). Raises ``ValueError``
    on a record of the wrong schema version, task or kind.
    """
    if not path.startswith(_MLFLOW_SCHEMES):
        return _load_local(Path(path), task)
    uri = resolve_downstream_uri(path, task)
    tmp = tempfile.mkdtemp(prefix="ntt_agg_downstream_")
    try:
        local = mlflow.artifacts.download_artifacts(uri, dst_path=tmp)
        return _load_local(Path(local), task)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def resolve_downstream_uri(uri: str, task: Task) -> str:
    """The record file to download for a downstream input. A decoding parent's artifact
    dir resolves to its cohort record, so the benchmark's cell artifacts are not
    downloaded with it; this costs one MLflow listing call.

    Only rewrites the URIs ``tracker_io.row_to_source_uris`` builds; local paths and
    explicit record files pass through unchanged.
    """
    if not uri.startswith(_MLFLOW_SCHEMES) or task == "encoding":
        return uri
    if posixpath.basename(uri) == DECODING_RECORD:
        return uri
    parent = uri.rstrip("/")
    names = {
        posixpath.basename(info.path)
        for info in mlflow.artifacts.list_artifacts(artifact_uri=parent)
    }
    if DECODING_RECORD not in names:
        raise FileNotFoundError(_missing_cohort_record(parent, names))
    return f"{parent}/{DECODING_RECORD}"


def _missing_cohort_record(where: str, names: set[str]) -> str:
    hint = (
        " The cohort record was not uploaded: rebuild it by rerunning the benchmark sweep."
        if names
        else ""
    )
    return (
        f"{where} has no {DECODING_RECORD}. Check that the tracker's mlflow_decoding "
        f"points at the benchmark's parent run.{hint}"
    )


def view_from_record(record: dict, task: Task, where: str = "record") -> DownstreamScoreView:
    """Validate a schema-v2 record and return the view of its ``test`` block."""
    version = record.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"{where}: schema_version {version!r}; this reader reads {SCHEMA_VERSION} "
            "(see ntt-downstream docs/results.md)"
        )
    if record.get("task") != task:
        raise ValueError(
            f"{where}: a {record.get('task')!r} record where a {task!r} record was expected"
        )
    kind = _RECORD_KIND[task]
    if record.get("kind") != kind:
        hint = (
            " Point mlflow_decoding at the benchmark's per-tokenizer parent run: its "
            "artifact root holds the cohort record."
            if task == "decoding"
            else ""
        )
        raise ValueError(
            f"{where}: kind {record.get('kind')!r}; ntt scores {task} from "
            f"kind {kind!r} records.{hint}"
        )

    meta = record.get("metadata") or {}
    provenance: dict[str, Any] = {
        key: meta.get(key)
        for key in ("tokenizer_checkpoint", "ntt_version", "ntt_path", "ntt_commit", "git_sha")
    }
    if task == "encoding":
        provenance["final_eval"] = meta.get("final_eval")
        complete = True
    else:
        cohort = meta.get("cohort") or {}
        for key in ("expected", "found", "missing", "stale", "mixed_provenance"):
            provenance[key] = cohort.get(key)
        complete = cohort.get("complete") is True

    # A missing ``test`` block means "no test score"; never substitute a val block.
    test = (record.get("metrics") or {}).get("test") or {}
    view = DownstreamScoreView(
        split="test",
        aggregated=dict(test.get("aggregated") or {}),
        per_session=dict(test.get("per_session") or {}),
        complete=complete,
        provenance=provenance,
    )
    final_eval = provenance.get("final_eval") or {}
    if final_eval.get("ckpt_kind") == "last":
        log.warning(
            "%s: the encoding test score comes from the last checkpoint, so it is not "
            "val-selected (final_eval.ckpt_kind = 'last').",
            where,
        )
    if provenance.get("mixed_provenance"):
        log.warning(
            "%s: decoding cohort has mixed_provenance: its cells disagree on git sha, ntt "
            "version or ntt commit (see metadata.cohort.children).",
            where,
        )
    return view


def _load_local(path: Path, task: Task) -> DownstreamScoreView:
    if path.is_dir():
        record_file = path / _RECORD_FILE[task]
        if not record_file.is_file():
            raise ValueError(f"{path} is a directory without {record_file.name}")
        path = record_file
    return view_from_record(json.loads(path.read_text()), task, where=str(path))
