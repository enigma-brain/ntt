#!/usr/bin/env python3
"""Build the tokenizer results tables from a tracker sheet, in one pass.

Reads a tracker sheet (or a local CSV export of it), resolves each row's eval artifacts
from MLflow, and emits both the wide per-tokenizer raw-metrics table and the 7 category
scores from a single download pass.

The tracker is a Google Sheet (or a local CSV export of it) with one row per tokenizer: a
``tokenizer`` column, the MLflow run links ``mlflow_recon`` (required),
``mlflow_latent_input_corruption``, ``mlflow_latent_latent_corruption``, ``mlflow_encoding``
and ``mlflow_decoding``, and optionally ``ckpt_mlflow_uri`` (recorded as the scores'
``ckpt_uri``). The link columns are filled in by hand; no code writes them. Instead of a
run link, a cell may hold a local path (an eval output directory or its report file), read in
place without MLflow; relative paths resolve against the CSV's folder.

Outputs (under ``--outdir``):
  ``wide_metrics_table.csv``       tokenizer x all raw metrics, incl. ``robustness_latent/*`` (M2)
  ``category_scores.csv``          the 7 category scores, plus ``decoding_poyo`` and
                                   ``decoding_linear`` (the two parts ``decoding`` averages)
  ``category_scores_no_m2.csv``    only with ``--compare-m2``: the same scores with M2 withheld
  ``category_scores_m2_delta.csv`` only with ``--compare-m2``: per-tokenizer latent-bucket shift
  ``metadata_rows.csv``            the tracker rows verbatim, for the per-tokenizer metadata
                                   columns the result tables do not carry
  ``scores/<slug>.json``           each tokenizer's full ``category_scores.json``, per-metric
                                   diagnostics included — feed it to the plotting and LaTeX
                                   CLIs with ``--scores-dir`` (no MLflow round-trip)
  ``run_log.txt``                  full per-row provenance trail
  ``table_checks.json``            every table-level check's outcome (written even when one fails)
  ``tracker_with_agg.csv``         only with ``--log-mlflow``: the tracker rows plus an
                                   ``mlflow_aggregation`` column linking each tokenizer's run

Artifacts are downloaded ONCE into ``--cache`` (manifest-tracked), so the optional
second no-M2 variant costs zero network I/O, and re-runs resume for free.

Examples
--------
    # a tracker sheet, both variants
    uv run python scripts/analysis/build_aggregation_tables.py \
        --sheet "https://docs.google.com/spreadsheets/d/<ID>/edit?gid=<GID>" \
        --outdir "aggregation rebuttal/outputs" --compare-m2

    # a local CSV export of the tracker, scores only
    uv run python scripts/analysis/build_aggregation_tables.py \
        --sheet tracker.csv --outdir out/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

# `scripts.analysis.*` is a repo directory, not an installed package. It imports only when
# the repo root is on sys.path — which happens implicitly under pytest, and when the CWD
# *is* the repo root, but not for `python scripts/analysis/x.py` from anywhere else. Adding
# it explicitly makes this runnable from any working directory. The imports below must
# therefore follow this block; `noqa: I001` keeps the import sorter from hoisting them
# above it and breaking that.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: I001

from ntt.evaluation.aggregation import tracker_io
from ntt.evaluation.aggregation.comparability import (
    COMMON_PARTS_ONLY_FLAG,
    COMPARABILITY_COLUMNS,
    COMPARABILITY_VERSION,
    DOWNSTREAM_PROVENANCE_COLUMNS,
    find_mixed_member_rows,
    find_mixed_score_parts,
    provenance_record,
    raise_if_not_comparable,
    raise_on_mixed_chunk_sizes,
    raise_on_mixed_score_parts,
    SCORE_PARTS,
    report_session_summary,
)
from ntt.evaluation.aggregation.downstream_records import (
    load_downstream_record,
    resolve_downstream_uri,
)
from ntt.evaluation.aggregation.publish import DEFAULT_EXPERIMENT, publish_category_scores
from ntt.evaluation.mlflow_logging import MLflowLoggingConfig
from ntt.jobs.aggregation import (
    RECOMPUTED_GENERALIZATION,
    NTTAggregationJob,
    recomputed_generalization_source,
)
from scripts.analysis.build_metrics_table import flatten_tokenizer_metrics

# The 7 primary score keys, mirroring the CategoryScores fields in
# src/ntt/evaluation/aggregation/category_scores.py. Declared once and covered by a test
# that asserts the two stay in sync: a key that drifts from the dataclass does not raise,
# it just yields a silently empty column.
SEVEN = (
    "single_neuron_recon",
    "population_structure",
    "latent_structure_and_robustness",
    "decoding",
    "downstream_encoding",
    "generalization",
    "compression",
)
# The two family scores the ``decoding`` composite averages (``CategoryScores``): written
# right after ``decoding`` in category_scores.csv, but not composite columns themselves.
DECODING_BREAKDOWNS = ("decoding_poyo", "decoding_linear")
_AFTER_DECODING = SEVEN.index("decoding") + 1
_SCORE_COLUMNS = (*SEVEN[:_AFTER_DECODING], *DECODING_BREAKDOWNS, *SEVEN[_AFTER_DECODING:])

ARTIFACT_KEYS = ("recon", "m1", "m2", "snr", "encoding", "decoding")
SCORE_EXTRA = (
    "generalization_source",
    "m1_provenance",
    "m2_provenance",
    "m2_raw_acg",
    "m2_metric_key",
    "decoding_provenance",
    "position_regression_included",
    "position_regression_excluded",
    # The artifact addresses (incl. MLflow run ids) this row was scored from, as JSON;
    # a resume rescores the row when the tracker now points elsewhere.
    "source_uris",
    # Parts left out of every tokenizer's score by --common-parts-only (e.g. "m2").
    "left_out_parts",
    # Which of this tokenizer's scores were left empty because it lacks a part that the other
    # tokenizers' scores have, e.g. "decoding (no poyo)".
    "emptied_scores",
    # The remaining comparability fields (members, sessions, aggregation strategy, run
    # settings, Poisson normalization), from provenance_record; find_mixed_member_rows
    # compares them across rows. Generated from the one list so the two cannot drift.
    *(
        column
        for column in COMPARABILITY_COLUMNS
        if column
        not in ("generalization_source", "m1_provenance", "m2_provenance", "decoding_provenance")
    ),
    # Where the downstream scores were read from (encoding split, decoding cohort
    # provenance).
    *DOWNSTREAM_PROVENANCE_COLUMNS,
)


# ── logging ────────────────────────────────────────────────────────────────────
class RunLog:
    """Console + file log, flushed every line so a crash keeps the trail.

    Appends, so resuming a run does not truncate the previous run's trail — which is
    exactly the record you need to find out why a row is missing from the output.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", encoding="utf-8")

    def __call__(self, msg: str) -> None:
        print(msg, flush=True)
        self._fh.write(msg + "\n")
        self._fh.flush()


# ── artifact cache ─────────────────────────────────────────────────────────────
class ArtifactCache:
    """Download-once cache keyed by ``tokenizer|artifact``, tracked in a JSON manifest.

    Entries record the source identity (URI and tracking server); a changed source is
    fetched again.
    """

    def __init__(self, root: Path, log: RunLog, attempts: int = 4) -> None:
        self.root = root
        self.log = log
        self.attempts = attempts
        self.manifest_path = root / "manifest.json"
        self.manifest: dict[str, dict[str, str] | str] = {}
        if self.manifest_path.exists():
            try:
                self.manifest = json.loads(self.manifest_path.read_text())
            except json.JSONDecodeError:
                self.log(f"  (ignoring corrupt cache manifest at {self.manifest_path})")

    def _save(self) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.manifest, indent=2, sort_keys=True))
        os.replace(tmp, self.manifest_path)

    def fetch(self, slug: str, key: str, uri: str) -> Path:
        if tracker_io.is_local_source(uri):
            # A local tracker cell is read in place: nothing to download or cache.
            return tracker_io.resolve_local_artifact(uri, key)
        mkey = f"{slug}|{key}"
        source = tracker_io.artifact_cache_key(uri)
        cached = self.manifest.get(mkey)
        if (
            isinstance(cached, dict)
            and cached.get("source") == source
            and cached.get("path")
            and Path(cached["path"]).exists()
        ):
            return Path(cached["path"])

        import mlflow.artifacts

        # Separate run directories prevent a recursive decoding download from
        # retaining session files that only belonged to an earlier run.
        dest = self.root / slug / key / source
        dest.mkdir(parents=True, exist_ok=True)
        last: Exception | None = None
        for attempt in range(self.attempts):
            try:
                local = Path(mlflow.artifacts.download_artifacts(uri, dst_path=str(dest)))
            except Exception as exc:  # transient MLflow "Response ended prematurely"
                last = exc
                self.log(f"      retry {attempt + 1}/{self.attempts} {key} ({type(exc).__name__})")
                continue
            self.manifest[mkey] = {"source": source, "path": str(local)}
            self._save()
            return local
        assert last is not None
        raise last

    def record_missing(self, uri: str) -> None:
        """Remember that the MLflow run has no file at ``uri``, so it is not requested again."""
        self.manifest[f"missing|{uri}"] = {"source": uri}
        self._save()

    def is_missing(self, uri: str) -> bool:
        return f"missing|{uri}" in self.manifest


# Each eval run saves a few small files next to its report. The report itself can lack the
# session list or the run settings (robustness eval_report.json), or the input geometry (recon
# results.json whose tokenizer config failed to serialize). fetch_run_files downloads and
# caches them from the same MLflow run so that NTTAggregationJob can use them. If they are
# missing, aggregation fails with "could NOT check".
RUN_FILES = {
    "eval_report.json": ("results_per_session.csv", "run_meta.json"),
    "results.json": ("run_meta.json",),
}


def fetch_run_files(report_path: Path, report_uri: str, cache: ArtifactCache, log: RunLog) -> None:
    """Download the ``RUN_FILES`` of a report once, and save them next to the cached report.

    Skip ``results_per_session.csv`` if ``report_session_summary`` shows the report already
    lists its sessions. Record missing ``RUN_FILES`` in the cache manifest so later runs do
    not request them again.

    If the download fails, the failure is logged but not raised immediately. The check will
    print "could NOT check", and the build will exit with an error — or with a warning if you
    pass ``--common-parts-only``."""
    if tracker_io.is_local_source(report_uri):
        return  # a local report's side files sit next to it already
    import mlflow.artifacts

    run_artifacts_uri, _, report_name = report_uri.rpartition("/")
    # Side files still to fetch: neither cached nor marked missing.
    wanted = [
        name
        for name in RUN_FILES.get(report_name, ())
        if not (report_path.parent / name).exists()
        and not cache.is_missing(f"{run_artifacts_uri}/{name}")
    ]
    if "results_per_session.csv" in wanted:
        report = tracker_io.load_json_artifact(str(report_path))
        if report_session_summary(report)["session_ids"] is not None:
            wanted.remove("results_per_session.csv")
    if not wanted:
        return
    try:
        # Check the run's file list first to tell a missing file from a failed download.
        listed = mlflow.artifacts.list_artifacts(artifact_uri=run_artifacts_uri)
        present = {Path(info.path).name for info in listed}
        for name in wanted:
            uri = f"{run_artifacts_uri}/{name}"
            if name not in present:
                cache.record_missing(uri)
                log(f"    note: the run has no {name} next to {report_uri}")
                continue
            mlflow.artifacts.download_artifacts(uri, dst_path=str(report_path.parent))
    except Exception as exc:
        log(f"    note: could not fetch run files next to {report_uri} ({type(exc).__name__})")


def load_snr_report(path: Path) -> dict:
    """The SNR report: ``eval_report.json``, or the ``aggregated`` block of the
    ``results.json`` the SNR pass writes locally (the same report)."""
    report = tracker_io.load_json_artifact(str(path))
    if "results" not in report and isinstance(report.get("aggregated"), dict):
        return report["aggregated"]
    return report


# ── output writers ─────────────────────────────────────────────────────────────
def _atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def write_metadata_rows(fieldnames: list[str], rows: list[dict], path: Path) -> None:
    """Dump the tracker rows verbatim next to the results.

    Preserves the per-tokenizer metadata columns (`latent_type`,
    `tokenization_regime`, `variant`) that the score/wide tables do not carry, so
    downstream consumers get everything they need from one run, and the exact input
    is recorded alongside the output it produced.
    """
    df = pd.DataFrame(rows, columns=fieldnames) if rows else pd.DataFrame(columns=fieldnames)
    _atomic(df, path)


def write_wide(records: list[dict], path: Path) -> None:
    df = pd.DataFrame(records)
    if "tokenizer" in df.columns:
        df = df[["tokenizer", *sorted(c for c in df.columns if c != "tokenizer")]]
    _atomic(df, path)


def write_scores(records: list[dict], path: Path) -> None:
    df = pd.DataFrame(records)
    cols = [c for c in ("tokenizer", *_SCORE_COLUMNS, *SCORE_EXTRA) if c in df.columns]
    _atomic(df[cols], path)


def write_delta(with_m2: list[dict], without_m2: list[dict], path: Path) -> None:
    """Per-tokenizer latent-bucket shift attributable to including M2."""
    base = {r["tokenizer"]: r for r in without_m2}
    rows = []
    for r in with_m2:
        tok = r["tokenizer"]
        before = (base.get(tok) or {}).get("latent_structure_and_robustness")
        after = r.get("latent_structure_and_robustness")
        both_numeric = isinstance(before, int | float) and isinstance(after, int | float)
        rows.append(
            {
                "tokenizer": tok,
                "latent_bucket_without_m2": before,
                "latent_bucket_with_m2": after,
                "delta": (after - before) if both_numeric else None,
                "m2_provenance": r.get("m2_provenance"),
                "m2_raw_acg": r.get("m2_raw_acg"),
                "m2_metric_key": r.get("m2_metric_key"),
            }
        )
    _atomic(pd.DataFrame(rows), path)


def _existing(path: Path) -> tuple[list[dict], set[str]]:
    if not path.exists():
        return [], set()
    recs = pd.read_csv(path).to_dict("records")
    return recs, {r["tokenizer"] for r in recs if isinstance(r.get("tokenizer"), str)}


# ── aggregation ────────────────────────────────────────────────────────────────
def run_variant(
    paths: dict[str, Path],
    out_dir: Path,
    *,
    wire_m2: bool,
    config_root: Path,
    left_out_parts: frozenset[str],
    common_parts_only: bool,
    allow_incomplete_decoding_cohort: bool = False,
    gen_split: str = "val",
    ckpt_uri: str | None = None,
) -> dict[str, Any]:
    """Run NTTAggregationJob off cached local paths.

    The two variants differ only in whether they use latent-corruption robustness. The
    variant without it passes no ``m2_path`` and sets ``include_m2=False``, so the score
    is recorded as ``off``. This way you can see what latent-corruption robustness adds
    to the latent score, without the system logging "report absent".
    ``config_root`` must be absolute so the generalization tier index resolves
    regardless of CWD — an unresolvable config root degrades the generalization score
    with a warning rather than raising.
    ``left_out_parts`` (subset of ``SCORE_PARTS``: m1, m2, poyo, ridge) are left out of
    the score, as ``--common-parts-only`` does for every tokenizer; ``common_parts_only``
    also turns unrecorded Poisson-reference aspects into a warning instead of an error.
    ``gen_split`` is the split whose sessions the generalization tiers are rebuilt from
    when a recon report has no precomputed ``generalization`` block.
    ``ckpt_uri`` is the tracker row's checkpoint (``ckpt_mlflow_uri``), recorded in the
    scores' metadata. The M2 operating point is the job's default (``m2_metric``/
    ``m2_epsilon``), defined once there.
    """
    job = NTTAggregationJob(
        recon_path=str(paths["recon"]),
        m1_path=str(paths["m1"]) if "m1" in paths else None,
        m2_path=str(paths["m2"]) if (wire_m2 and "m2" in paths) else None,
        encoding_path=str(paths["encoding"]) if "encoding" in paths else None,
        decoding_path=str(paths["decoding"]) if "decoding" in paths else None,
        include_m2=wire_m2 and "m2" not in left_out_parts,
        include_m1="m1" not in left_out_parts,
        include_poyo="poyo" not in left_out_parts,
        include_ridge="ridge" not in left_out_parts,
        common_parts_only=common_parts_only,
        allow_incomplete_decoding_cohort=allow_incomplete_decoding_cohort,
        gen_config_root=str(config_root),
        gen_split=gen_split,
        ckpt_uri=ckpt_uri,
        output_path=str(out_dir),
        mlflow_logging=MLflowLoggingConfig(enabled=False),
    )
    job.run()
    return json.loads((out_dir / "category_scores.json").read_text())


def score_record(slug: str, cs: dict) -> dict:
    """Flatten a category_scores.json into one row, carrying provenance as evidence.

    The provenance columns say *how* a score resolved: ``m1_provenance``/``m2_provenance``
    whether each latent-bucket member contributed or was warned-and-excluded, and
    ``decoding_provenance`` the poyo/ridge family means behind the decoding column
    (``absent`` for a family that contributed nothing). They are always present — an
    all-null column is honest evidence, whereas a conditional key would drop the header
    entirely on a cohort where nothing resolved.
    The remaining comparability fields come from ``provenance_record``.
    """
    diagnostics = cs.get("diagnostics") or {}
    rec: dict[str, Any] = {
        "tokenizer": slug,
        **{c: cs.get("scores", {}).get(c) for c in _SCORE_COLUMNS},
        **provenance_record(cs),
    }
    lsr = diagnostics.get("latent_structure_and_robustness") or {}
    for name, ms in (lsr.get("per_metric") or {}).items():
        if name.startswith("m2:"):
            rec["m2_raw_acg"] = ms.get("raw_value")
            rec["m2_metric_key"] = name
            break
    return rec


def _scores_json_path(scores_json_dir: Path, tokenizer: str) -> Path:
    """Where one tokenizer's persisted ``category_scores.json`` lives.

    The stem comes from ``tracker_io.scores_slug`` — the *same* function
    ``scripts/analysis/data.py`` uses to resolve it — because a tokenizer name with a
    capital or a dot would otherwise be written under one name and looked up under
    another, and the reader's only symptom is a silently missing row.
    """
    return scores_json_dir / f"{tracker_io.scores_slug(tokenizer)}.json"


def _persist_scores_json(scores_json_dir: Path, tokenizer: str, cs: dict) -> None:
    """Write the tokenizer's full ``category_scores.json`` beside the CSVs.

    ``score_record`` flattens the payload to seven scalars plus provenance; this keeps
    the whole document — per-metric diagnostics included — which is what
    ``data.load_scores(scores_dir=...)`` needs to feed ``plot_correlation`` and
    ``plot_family``. Written atomically so a reader never sees a half-file.
    """
    scores_json_dir.mkdir(parents=True, exist_ok=True)
    dest = _scores_json_path(scores_json_dir, tokenizer)
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cs, indent=2))
    os.replace(tmp, dest)


def _discard_scores_json(scores_json_dir: Path, tokenizer: str) -> None:
    """Drop any JSON left by an earlier run of a now-failing tokenizer.

    A failed tokenizer is recorded nowhere (no wide row, no score row), so leaving a
    stale JSON would make the local route disagree with the CSVs: ``load_scores``
    would happily serve a payload for a tokenizer this run could not produce.
    """
    _scores_json_path(scores_json_dir, tokenizer).unlink(missing_ok=True)


def _reconcile_scores_dir(scores_json_dir: Path, scored: set[str], log: RunLog) -> set[str]:
    """Make ``scores/`` agree with ``category_scores.csv``, and report what it covers.

    ``scored`` must be the tokenizer set of ``category_scores.csv`` alone — that is the
    file ``scores/`` mirrors, one JSON per row. Keying this on the wider resume
    intersection (wide ∩ main ∩ no_m2) instead makes adding ``--compare-m2`` to an outdir
    built without it collapse the set to empty and delete every JSON, while logging that
    rows plainly present in ``category_scores.csv`` are missing from it.

    Two directions, deliberately asymmetric:

    * A JSON with no ``category_scores.csv`` row is **deleted**. That is the direction
      that produces wrong output — ``load_scores(scores_dir=...)`` would serve a
      tokenizer absent from the headline table — and it is what the documented
      force-redo (delete the row from the CSVs and re-run) has to mean.
    * A row with no JSON is left uncovered, so the tokenizer is **re-scored**. Deleting
      the row to force agreement would be destroying the very numbers people report,
      and recovering the document from ``_work/`` would be worse: that file is whatever
      the scorer wrote the day it was last touched, so copying it back marks the
      tokenizer complete and suppresses the re-score a scorer fix exists to deliver.
      Re-scoring is cheap — the job is JSON-in/JSON-out over already-cached artifacts.
    """
    keep = {tracker_io.scores_slug(t) for t in scored}
    for path in sorted(scores_json_dir.glob("*.json")):
        if path.stem not in keep:
            path.unlink()
            log(f"  (dropped scores/{path.name}: no row in category_scores.csv)")

    return {tok for tok in scored if _scores_json_path(scores_json_dir, tok).exists()}


def _is_present(value: Any) -> bool:
    """True when a record cell actually carries something.

    Preloaded records come back through ``pd.read_csv``, so "this row has no value for
    a column some other row has" arrives as NaN rather than a missing key.
    """
    if value is None:
        return False
    return not (isinstance(value, float) and value != value)


def _warn_on_slug_collisions(tokenizers: list[str], log: RunLog) -> None:
    """Two tracker names folding to one slug silently share a ``scores/<slug>.json``.

    The second write overwrites the first, and ``load_scores`` then hands the same
    ``_scores_data`` to both rows — identical per-metric diagnostics for two different
    tokenizers, with nothing anywhere saying so.
    """
    by_slug: dict[str, list[str]] = {}
    for tok in tokenizers:
        if tok:
            by_slug.setdefault(tracker_io.scores_slug(tok), []).append(tok)
    for slug, toks in sorted(by_slug.items()):
        if len(toks) > 1:
            log(
                f"WARNING: slug collision on '{slug}': {toks} share one scores/{slug}.json; "
                f"only the last one written survives, and --scores-dir readers will "
                f"attribute it to all of them."
            )


def _source_uris_json(uris: dict[str, str]) -> str:
    """The tracker links of one row, as compared on resume (stable key order)."""
    return json.dumps(uris, sort_keys=True)


def _drop_rows_with_changed_links(
    done: set[str],
    rows: list[dict],
    main_recs: list[dict],
    log: RunLog,
    base_dir: Path | None = None,
) -> None:
    """When a tokenizer's MLflow links in the tracker sheet have changed since its last scoring,
    remove it from ``done`` so it gets scored again. Also remove any tokenizer that has no
    stored links. Downloaded files are cached
    by link, so unchanged links are not re-downloaded.

    A row with a local path is always scored again: re-running an evaluation into the same
    folder keeps its path but changes its files. Local files are read in place, so this is cheap.
    """
    recorded = {
        rec["tokenizer"]: rec.get("source_uris")
        for rec in main_recs
        if isinstance(rec.get("tokenizer"), str)
    }
    for row in rows:
        slug = (row.get("tokenizer") or "").strip()
        uris = tracker_io.row_to_source_uris(row, base_dir=base_dir)
        if slug not in done or uris is None:
            continue
        before = recorded.get(slug)
        if any(tracker_io.is_local_source(uri) for uri in uris.values()):
            reason = "local outputs are always re-scored"
        elif not _is_present(before):
            reason = "no tracker links recorded"
        elif before != _source_uris_json(uris):
            old, new = json.loads(before), uris
            changed = sorted(k for k in old.keys() | new.keys() if old.get(k) != new.get(k))
            reason = f"tracker links changed for {changed}"
        else:
            continue
        done.discard(slug)
        log(f"RESCORE {slug}: {reason}")


def _drop_rows_scored_without_some_parts(
    done: set[str], main_recs: list[dict], log: RunLog
) -> None:
    """Remove from ``done`` each tokenizer whose stored results row was computed with some parts
    excluded. Every run scores all parts first, then decides fresh which parts to exclude."""
    for rec in main_recs:
        tokenizer = rec.get("tokenizer")
        if tokenizer in done and _is_present(rec.get("left_out_parts")):
            done.discard(tokenizer)
            log(
                f"RESCORE {tokenizer}: scored without {rec['left_out_parts']}; rescoring all parts"
            )
        elif tokenizer in done and _is_present(rec.get("emptied_scores")):
            done.discard(tokenizer)
            log(
                f"RESCORE {tokenizer}: {rec['emptied_scores']} was left empty; rescoring all parts"
            )


def _drop_rows_on_another_gen_split(
    done: set[str], main_recs: list[dict], gen_split: str, log: RunLog
) -> None:
    """Remove from ``done`` each tokenizer whose generalization score was rebuilt from
    per-session results on another split than ``--gen-split``. Rows with a precomputed
    generalization block do not depend on the flag."""
    wanted = recomputed_generalization_source(gen_split)
    for rec in main_recs:
        tokenizer = rec.get("tokenizer")
        source = rec.get("generalization_source")
        if (
            tokenizer in done
            and isinstance(source, str)
            and source.startswith(RECOMPUTED_GENERALIZATION)
            and source != wanted
        ):
            done.discard(tokenizer)
            log(f"RESCORE {tokenizer}: generalization was {source}, --gen-split is {gen_split}")


def _drop_rows_scored_by_another_comparability_version(
    done: set[str], main_recs: list[dict], log: RunLog
) -> None:
    """Remove each tokenizer from ``done`` whose row carries another ``COMPARABILITY_VERSION``
    than the current one (or none), so it is scored again."""
    for rec in main_recs:
        tokenizer = rec.get("tokenizer")
        version = rec.get("comparability_version")
        if tokenizer in done and (
            not _is_present(version) or int(float(version)) != COMPARABILITY_VERSION
        ):
            done.discard(tokenizer)
            shown = version if _is_present(version) else "none"
            log(
                f"RESCORE {tokenizer}: scored with comparability version {shown}, "
                f"current is {COMPARABILITY_VERSION}"
            )


def _empty_scores_with_missing_parts(
    recs: list[dict], scores_json_dir: Path | None, log: RunLog
) -> bool:
    """Empty a tokenizer's score when it lacks a part that other tokenizers' scores
    include (e.g. POYO in decoding), in ``recs`` and in its ``scores/<slug>.json``.
    Returns whether any score was emptied."""
    mixed = find_mixed_score_parts(recs)
    for part, who in mixed.items():
        column = SCORE_PARTS[part][1]
        for rec in recs:
            if rec["tokenizer"] not in who["left_out"]:
                continue
            rec[column] = None
            note = f"{column} (no {part})"
            before = rec.get("emptied_scores")
            rec["emptied_scores"] = f"{before}; {note}" if _is_present(before) else note
            # If this tokenizer has no `scores/<tokenizer>.json` file, there is nothing to empty
            # there. `main()` lists such tokenizers at the end.
            path = (
                None
                if scores_json_dir is None
                else _scores_json_path(scores_json_dir, rec["tokenizer"])
            )
            if path is not None and path.exists():
                cs = json.loads(path.read_text())
                cs["scores"][column] = None
                _persist_scores_json(scores_json_dir, rec["tokenizer"], cs)
        log(
            f"WARNING: {column} left empty for {who['left_out']}: their score has no {part}, "
            f"the other tokenizers' scores have it."
        )
    return bool(mixed)


def _replace_record(recs: list[dict], rec: dict) -> None:
    """Swap a freshly computed record in, dropping any preloaded copy of the same slug.

    Called only once every output for the tokenizer has succeeded. Dropping the stale
    copy up front instead — before knowing whether the re-run would produce anything —
    is what let a ``--limit`` stop, a ``break``, or one transient failure delete rows
    from the shipped table.
    """
    tok = rec.get("tokenizer")
    recs[:] = [r for r in recs if r.get("tokenizer") != tok]
    recs.append(rec)


# ── table-level checks ─────────────────────────────────────────────────────────
TABLE_CHECKS_FILE = "table_checks.json"

_SCORE_PARTS_HOW_TO_FIX = (
    f"Re-run with {COMMON_PARTS_ONLY_FLAG} to build every tokenizer's score only "
    f"from the parts all tokenizers have (all tokenizers stay in the table), or "
    f"add the missing runs to the input table (Google Sheet or CSV file)."
)


def run_table_checks(
    main_recs: list[dict],
    no_m2_recs: list[dict] | None,
    *,
    common_parts_only: bool,
    log: RunLog,
) -> tuple[list[dict[str, Any]], ValueError | None]:
    """Run every table-level check over the final table, collecting instead of raising.

    Each outcome is ``{"check", "passed", "message"}``; a passed check's message is a
    warning worth reading. Returns the outcomes and the first failure, which the caller
    raises once the outcomes are recorded (``table_checks.json``, the MLflow runs), so the
    exit is the same as when the checks raised one by one. Order: chunk sizes, members,
    mixed parts (the more specific problem; the flag fixes both), then scores averaged
    differently or that could not be checked. The table
    without latent-corruption robustness is only checked for chunk sizes: the rest would
    repeat these lines.
    """
    checks: list[dict[str, Any]] = []
    first_error: ValueError | None = None

    def collect(name: str, check: Callable[[], str | None]) -> None:
        nonlocal first_error
        try:
            message = check() or ""
        except ValueError as exc:
            checks.append({"check": name, "passed": False, "message": str(exc)})
            log(f"CHECK FAILED {name}: {exc}")
            first_error = first_error or exc
            return
        checks.append({"check": name, "passed": True, "message": message})
        if message:
            log(message)

    def mixed_members() -> str | None:
        lines = find_mixed_member_rows(main_recs)
        return "\n".join(["[category_scores.csv]", *lines]) if lines else None

    def no_m2_chunk_sizes() -> None:
        raise_on_mixed_chunk_sizes(no_m2_recs or [])

    collect("mixed_chunk_sizes", lambda: raise_on_mixed_chunk_sizes(main_recs))
    if no_m2_recs is not None:
        collect("mixed_chunk_sizes_no_m2", no_m2_chunk_sizes)
    collect("mixed_members", mixed_members)
    collect(
        "mixed_score_parts",
        lambda: raise_on_mixed_score_parts(main_recs, how_to_fix=_SCORE_PARTS_HOW_TO_FIX),
    )
    collect(
        "not_comparable",
        lambda: raise_if_not_comparable(main_recs, common_parts_only=common_parts_only),
    )
    return checks, first_error


def write_table_checks(checks: list[dict[str, Any]], path: Path) -> None:
    doc = {"passed": all(c["passed"] for c in checks), "checks": checks}
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2))
    os.replace(tmp, path)


# ── MLflow output (--log-mlflow) ───────────────────────────────────────────────
TRACKER_WITH_AGG_FILE = "tracker_with_agg.csv"
AGG_URL_COLUMN = "mlflow_aggregation"
TABLE_CHECKS_TAG = "ntt.agg.table_checks"


def table_checks_tag(checks: list[dict[str, Any]]) -> str:
    failed = [c["check"] for c in checks if not c["passed"]]
    return "failed:" + ",".join(failed) if failed else "passed"


def publish_table(
    main_recs: list[dict],
    scores_json_dir: Path,
    table_checks_path: Path,
    checks: list[dict[str, Any]],
    *,
    experiment: str,
    log: RunLog,
) -> dict[str, str]:
    """Publish each ``category_scores.csv`` row's ``scores/<slug>.json`` as an MLflow run;
    return tokenizer → run URL.

    Every row is published even when the table failed a check: the run's
    ``ntt.agg.table_checks`` tag and its ``table_checks.json`` say which. A row whose
    publish fails is logged and left out; the others continue.
    """
    tag = table_checks_tag(checks)
    urls: dict[str, str] = {}
    n_new = n_reused = n_failed = 0
    for rec in main_recs:
        tokenizer = rec.get("tokenizer")
        if not isinstance(tokenizer, str):
            continue
        try:
            published = publish_category_scores(
                _scores_json_path(scores_json_dir, tokenizer),
                experiment=experiment,
                run_name=tokenizer,
                tags={TABLE_CHECKS_TAG: tag},
                artifacts=[table_checks_path],
            )
        except Exception as exc:
            n_failed += 1
            log(f"  PUBLISH FAILED {tokenizer}: {type(exc).__name__}: {str(exc)[:200]}")
            continue
        urls[tokenizer] = published.url
        if published.reused:
            n_reused += 1
        else:
            n_new += 1
        log(f"  {'REUSED' if published.reused else 'PUBLISHED'} {tokenizer}: {published.url}")
    log(
        f"MLflow experiment {experiment}: {n_new} published, {n_reused} reused, "
        f"{n_failed} failed ({TABLE_CHECKS_TAG}={tag})"
    )
    return urls


def write_tracker_with_agg(
    fieldnames: list[str], rows: list[dict], urls: dict[str, str], path: Path
) -> None:
    """The tracker rows with ``mlflow_aggregation`` set to this run's URLs, to paste into
    the sheet. A row not published here gets an empty cell, never a stale link."""
    columns = fieldnames if AGG_URL_COLUMN in fieldnames else [*fieldnames, AGG_URL_COLUMN]
    out = [
        {**row, AGG_URL_COLUMN: urls.get((row.get("tokenizer") or "").strip(), "")} for row in rows
    ]
    write_metadata_rows(columns, out, path)


def process_variants(
    slug: str,
    paths: dict[str, Path],
    work: Path,
    scores_json_dir: Path | None,
    variants: list[tuple[str, bool]],
    config_root: Path,
    log: RunLog,
    left_out_parts: frozenset[str] = frozenset(),
    common_parts_only: bool = False,
    allow_incomplete_decoding_cohort: bool = False,
    gen_split: str = "val",
    ckpt_uri: str | None = None,
) -> list[tuple[str, dict]] | None:
    """Score every variant for one tokenizer, all-or-nothing.

    Returns ``[(label, record), ...]`` on full success, or ``None`` if any variant
    raised — in which case the caller must record the tokenizer nowhere and must NOT
    mark the slug done, so the next run retries it. Records are held locally until
    every variant has succeeded, so a late failure cannot leave a real row in one
    output table and nothing (or worse, a placeholder) in another.

    ``scores_json_dir``, when set, receives the ``with_m2`` variant's full
    ``category_scores.json`` as ``<slug>.json`` — the persistent form of what
    ``run_variant`` otherwise leaves in ``_work/``. It is written only once every
    variant has succeeded, and any pre-existing copy is deleted on failure, so this
    function never leaves behind a JSON for a tokenizer it returned ``None`` for.
    ``main()`` additionally discards before processing, covering the failure paths that
    return before ever reaching here.
    """
    out: list[tuple[str, dict]] = []
    reported: dict | None = None
    for label, wire in variants:
        vdir = work / slug / label
        vdir.mkdir(parents=True, exist_ok=True)
        # score_record() is inside the try on purpose: it walks an untrusted payload,
        # and a raise from it out here aborted the whole batch instead of costing this
        # one tokenizer — the same logged-and-continue contract run_variant already had.
        try:
            cs = run_variant(
                paths,
                vdir,
                wire_m2=wire,
                config_root=config_root,
                left_out_parts=left_out_parts,
                common_parts_only=common_parts_only,
                allow_incomplete_decoding_cohort=allow_incomplete_decoding_cohort,
                gen_split=gen_split,
                ckpt_uri=ckpt_uri,
            )
            if label == "with_m2":
                reported = cs
            record = score_record(slug, cs)
            # Recorded so a later run notices rows scored without some parts.
            record["left_out_parts"] = ";".join(sorted(left_out_parts)) or None
            out.append((label, record))
        except Exception as exc:
            log(f"    AGG {label} FAILED (retry next run): {type(exc).__name__}: {str(exc)[:200]}")
            log(traceback.format_exc(limit=3))
            if scores_json_dir is not None:
                _discard_scores_json(scores_json_dir, slug)
            return None
    if scores_json_dir is not None and reported is not None:
        _persist_scores_json(scores_json_dir, slug, reported)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--sheet",
        required=True,
        help=(
            "the eval tracker: a Google Sheet URL or a local CSV export of it, one row per "
            "tokenizer with the MLflow link columns mlflow_recon (required), "
            "mlflow_latent_input_corruption, mlflow_latent_latent_corruption, "
            "mlflow_encoding, mlflow_decoding, and optionally ckpt_mlflow_uri. Each link "
            "cell may instead be a local path to the eval output directory or its report "
            "file (relative paths: against the CSV's folder), read without MLflow"
        ),
    )
    ap.add_argument("--outdir", required=True, help="directory for the output CSVs")
    ap.add_argument(
        "--cache", default=None, help="artifact cache dir (default: <outdir>/../artifact_cache)"
    )
    ap.add_argument(
        "--config-root",
        default=str(REPO_ROOT / "config"),
        help="Hydra config root for generalization tiers (default: repo config/)",
    )
    ap.add_argument(
        "--compare-m2",
        action="store_true",
        help="also emit a no-M2 variant + delta table (no extra downloads)",
    )
    ap.add_argument("--limit", type=int, default=None, help="process at most N new rows")
    ap.add_argument(
        COMMON_PARTS_ONLY_FLAG,
        action="store_true",
        help=(
            "build every tokenizer's score only from the parts all tokenizers have "
            "(input-corruption robustness, latent-corruption robustness, POYO, ridge); all "
            "tokenizers stay in the table. Also reports "
            "aspects that cannot be checked (not saved by the eval files) as a warning. "
            "Without it, both are an error."
        ),
    )
    ap.add_argument(
        "--allow-incomplete-decoding-cohort",
        action="store_true",
        help=(
            "score a decoding cohort with missing cells over its found cells; without it, "
            "such a tokenizer's decoding score is empty"
        ),
    )
    ap.add_argument(
        "--gen-split",
        choices=("val", "test"),
        default="val",
        help=(
            "split whose sessions the generalization score is rebuilt from when a recon "
            "report has no precomputed generalization block (NTTAggregationJob.gen_split)"
        ),
    )
    ap.add_argument(
        "--log-mlflow",
        action="store_true",
        help=(
            "after the table checks, publish each tokenizer's category_scores.json as an "
            "MLflow run (to MLFLOW_TRACKING_URI), tagged with the checks' outcome, even when "
            "a check failed; and write tracker_with_agg.csv with an mlflow_aggregation "
            "column. Publishing an unchanged score in an unchanged table again reuses its run."
        ),
    )
    ap.add_argument(
        "--mlflow-experiment",
        default=DEFAULT_EXPERIMENT,
        help=f"MLflow experiment for --log-mlflow (default: {DEFAULT_EXPERIMENT})",
    )
    args = ap.parse_args()
    if args.log_mlflow and not os.environ.get("MLFLOW_TRACKING_URI"):
        ap.error(
            "--log-mlflow needs MLFLOW_TRACKING_URI, the server the runs go to "
            "(e.g. `set -a && source .env && set +a`)"
        )

    outdir = Path(args.outdir).resolve()
    cache_root = Path(args.cache).resolve() if args.cache else outdir.parent / "artifact_cache"
    config_root = Path(args.config_root).resolve()
    work = outdir / "_work"

    log = RunLog(outdir / "run_log.txt")
    cache = ArtifactCache(cache_root, log)
    # Both describe the table at the end of one run. An earlier run's copy must not
    # outlive a run that stops before the checks, or (the links) one without --log-mlflow:
    # its URLs point at the runs of the earlier scores. Unless it is this run's --sheet:
    # re-running on the last tracker_with_agg.csv keeps its links.
    sheet = tracker_io.to_csv_export_url(args.sheet)
    sheet_path = None if tracker_io.is_url(sheet) else Path(sheet).resolve()
    # Relative local paths in the tracker's cells resolve against the CSV's folder (the cwd
    # for a Google Sheet, which has none).
    tracker_dir = sheet_path.parent if sheet_path else None
    for previous in (outdir / TABLE_CHECKS_FILE, outdir / TRACKER_WITH_AGG_FILE):
        if previous != sheet_path:
            previous.unlink(missing_ok=True)

    wide_out = outdir / "wide_metrics_table.csv"
    scores_out = outdir / "category_scores.csv"
    nom2_out = outdir / "category_scores_no_m2.csv"
    delta_out = outdir / "category_scores_m2_delta.csv"
    metadata_rows_out = outdir / "metadata_rows.csv"
    # The per-tokenizer artifacts the plotting/LaTeX CLIs read via
    # data.load_scores(scores_dir=...). Created up front so the directory exists (and
    # is visibly empty) even on a run where every tokenizer fails.
    scores_json_dir = outdir / "scores"
    scores_json_dir.mkdir(parents=True, exist_ok=True)

    if not config_root.is_dir():
        log(
            f"WARNING: --config-root {config_root} is not a directory; "
            f"generalization tiers will not resolve and those scores will degrade."
        )

    fieldnames, rows = tracker_io.read_tracker_rows(args.sheet)
    write_metadata_rows(fieldnames, rows, metadata_rows_out)
    log(f"Tracker source : {args.sheet}")
    log(f"Rows in sheet  : {len(rows)}")
    log(f"Output dir     : {outdir}")
    log(f"Artifact cache : {cache_root}")
    log(f"Config root    : {config_root}")
    log(f"compare-m2     : {args.compare_m2}")
    log(f"gen-split      : {args.gen_split}")

    _warn_on_slug_collisions([(r.get("tokenizer") or "").strip() for r in rows], log)

    wide_recs, wide_done = _existing(wide_out)
    main_recs, main_done = _existing(scores_out)
    csv_done = wide_done & main_done
    nom2_recs: list[dict] = []
    if args.compare_m2:
        nom2_recs, nom2_done = _existing(nom2_out)
        csv_done &= nom2_done

    # scores/ mirrors category_scores.csv, so it is reconciled against `main_done` alone
    # — never the wider resume intersection, which would delete every JSON the moment a
    # sibling CSV is short. Completeness for *resume* purposes is then the intersection
    # of that coverage with the CSV agreement. The preloaded records are NOT filtered to
    # `done`: a record is replaced only once its tokenizer has been fully recomputed
    # (see _replace_record), so a --limit stop or a failure cannot delete rows
    # that are already in the shipped tables.
    covered = _reconcile_scores_dir(scores_json_dir, main_done, log)
    done = csv_done & covered
    _drop_rows_with_changed_links(done, rows, main_recs, log, base_dir=tracker_dir)
    _drop_rows_scored_without_some_parts(done, main_recs, log)
    _drop_rows_on_another_gen_split(done, main_recs, args.gen_split, log)
    _drop_rows_scored_by_another_comparability_version(done, main_recs, log)
    log(f"Already complete (resume): {len(done)}\n")

    n_ok = n_skip = n_resume = n_err = 0
    processed = 0
    # Parts (m1, m2, poyo, ridge) left out of every tokenizer's score. The first pass
    # scores every part; with --common-parts-only, a part some tokenizers lack is then
    # added here and every tokenizer is rescored without it (cheap: artifacts cached).
    left_out_parts: frozenset[str] = frozenset()
    is_first_pass = True
    while True:
        for i, row in enumerate(rows):
            slug = (row.get("tokenizer") or "").strip()
            uris = tracker_io.row_to_source_uris(row, base_dir=tracker_dir)
            tag = f"[{i + 1}/{len(rows)}]"

            if not slug or uris is None:
                n_skip += 1
                log(f"{tag} {slug or '(no tokenizer)'}: SKIP (no recon run)")
                continue
            if slug in done:
                n_resume += 1
                log(f"{tag} {slug}: SKIP (already done)")
                continue
            # --limit only affects the first pass. The --common-parts-only pass rescores every
            # tokenizer regardless.
            if is_first_pass and args.limit is not None and processed >= args.limit:
                log(f"{tag} {slug}: STOP (--limit {args.limit} reached)")
                break

            log(f"{tag} {slug}: artifacts={sorted(uris)}")
            # This tokenizer is being (re)processed, so it has no current scores/<slug>.json
            # until it succeeds again. Dropping it here rather than in process_variants
            # covers the two failure paths below (download, wide flatten) that return before
            # process_variants is ever reached — each of which would otherwise leave a stale JSON
            # for a tokenizer with no row in category_scores.csv.
            _discard_scores_json(scores_json_dir, slug)
            try:
                paths = {"recon": cache.fetch(slug, "recon", uris["recon"])}
                fetch_run_files(paths["recon"], uris["recon"], cache, log)
            except Exception as exc:
                n_err += 1
                log(
                    f"    DOWNLOAD FAILED (retry next run): {type(exc).__name__}: {str(exc)[:160]}"
                )
                continue

            # Optional inputs have independent failure boundaries at both download and
            # parsing. Only validated paths are passed to the job, so its second read
            # cannot turn an excluded optional artifact into a row-level failure.
            loaded = {}
            for key in ARTIFACT_KEYS[1:]:
                if key not in uris:
                    continue
                try:
                    if key in ("encoding", "decoding"):
                        path = cache.fetch(slug, key, resolve_downstream_uri(uris[key], key))
                        loaded[key] = load_downstream_record(str(path), key)
                    else:
                        path = cache.fetch(slug, key, uris[key])
                        fetch_run_files(path, uris[key], cache, log)
                        loaded[key] = (
                            load_snr_report(path)
                            if key == "snr"
                            else tracker_io.load_json_artifact(str(path))
                        )
                    paths[key] = path
                except Exception as exc:
                    log(f"    WARNING: EXCLUDING {key}: {type(exc).__name__}: {str(exc)[:160]}")

            try:
                recon = tracker_io.load_json_artifact(str(paths["recon"]))
                wide_rec = {
                    "tokenizer": slug,
                    **flatten_tokenizer_metrics(
                        recon,
                        loaded.get("m1"),
                        loaded.get("encoding"),
                        loaded.get("decoding"),
                        m2=loaded.get("m2"),
                        snr=loaded.get("snr"),
                    ),
                }
            except Exception as exc:
                n_err += 1
                log(f"    WIDE FAILED: {type(exc).__name__}: {str(exc)[:160]}")
                continue

            variants = [("with_m2", True)]
            if args.compare_m2:
                variants.append(("no_m2", False))
            sinks = {"with_m2": main_recs, "no_m2": nom2_recs}

            results = process_variants(
                slug,
                paths,
                work,
                scores_json_dir,
                variants,
                config_root,
                log,
                left_out_parts,
                args.common_parts_only,
                args.allow_incomplete_decoding_cohort,
                gen_split=args.gen_split,
                ckpt_uri=(row.get("ckpt_mlflow_uri") or "").strip() or None,
            )
            if results is None:
                n_err += 1
                # `wide_rec` was never committed, so the slug is recorded nowhere and any
                # row it already had in the tables is left exactly as it was.
                continue

            # Everything for this tokenizer succeeded: commit all of its records together.
            _replace_record(wide_recs, wide_rec)
            for label, rec in results:
                rec["source_uris"] = _source_uris_json(uris)
                _replace_record(sinks[label], rec)

            raise_on_mixed_chunk_sizes(main_recs)
            if args.compare_m2:
                raise_on_mixed_chunk_sizes(nom2_recs)
            write_wide(wide_recs, wide_out)
            write_scores(main_recs, scores_out)
            if args.compare_m2:
                write_scores(nom2_recs, nom2_out)
                write_delta(main_recs, nom2_recs, delta_out)

            done.add(slug)
            n_ok += 1
            processed += 1
            log(
                f"    OK  m1={main_recs[-1].get('m1_provenance')}  "
                f"m2={main_recs[-1].get('m2_provenance')}  "
                f"gen={main_recs[-1].get('generalization_source')}  "
                f"left_out={sorted(left_out_parts) or 'none'}"
            )

        if not args.common_parts_only:
            break
        missing_for_some = frozenset(find_mixed_score_parts(main_recs))
        if missing_for_some <= left_out_parts:
            break
        left_out_parts |= missing_for_some
        log(
            f"\n{COMMON_PARTS_ONLY_FLAG}: {sorted(left_out_parts)} is missing for some "
            f"tokenizers — all tokenizers stay in the table; {sorted(left_out_parts)} is "
            f"left out of every tokenizer's score. Rescoring all tokenizers.\n"
        )
        done.clear()
        is_first_pass = False

    # Empty each score missing a part found in other tokenizers' scores, including in the
    # `--compare-m2` table. Rewrite the output files if anything changed.
    changed = _empty_scores_with_missing_parts(main_recs, scores_json_dir, log)
    if args.compare_m2:
        changed |= _empty_scores_with_missing_parts(nom2_recs, None, log)
    if changed:
        write_wide(wide_recs, wide_out)
        write_scores(main_recs, scores_out)
        if args.compare_m2:
            write_scores(nom2_recs, nom2_out)
            write_delta(main_recs, nom2_recs, delta_out)

    log("\n" + "=" * 70)
    log(f"DONE: {n_ok} processed, {n_resume} resumed-skip, {n_skip} no-recon, {n_err} failed")
    for p in (wide_out, scores_out, nom2_out, delta_out, metadata_rows_out):
        if p.exists():
            df = pd.read_csv(p)
            log(f"  {p.name}: {df.shape[0]} rows x {df.shape[1]} cols")
    n_json = len(list(scores_json_dir.glob("*.json")))
    log(
        f"  scores/: {n_json} category_scores.json "
        f"(plot/LaTeX CLIs: --scores-dir {scores_json_dir})"
    )

    # scores/ is allowed to lag category_scores.csv (a row with no JSON is re-scored,
    # never deleted), but the run that leaves it lagging has to say so: a --scores-dir
    # table built now would silently omit these tokenizers.
    uncovered = sorted(
        r["tokenizer"]
        for r in main_recs
        if isinstance(r.get("tokenizer"), str)
        and not _scores_json_path(scores_json_dir, r["tokenizer"]).exists()
    )
    if uncovered:
        log(
            f"WARNING: {len(uncovered)} tokenizer(s) are in category_scores.csv but have no "
            f"scores/*.json, so --scores-dir output would omit them: {uncovered}. "
            f"Re-run to score them (their artifacts are already cached)."
        )

    mixed_cohorts = sorted(
        str(r["tokenizer"]) for r in main_recs if str(r.get("decoding_mixed_provenance")) == "True"
    )
    if mixed_cohorts:
        log(
            f"WARNING: decoding cohorts with mixed provenance (their cells disagree on git "
            f"sha, ntt version or ntt commit): {mixed_cohorts}"
        )

    if left_out_parts:
        log(
            f"Left out of every tokenizer's score ({COMMON_PARTS_ONLY_FLAG}): "
            f"{sorted(left_out_parts)}; all tokenizers stay in the table."
        )
    # Over the FULL final table (resumed rows included), not just this run's rows.
    # Different parts in different tokenizers' scores make the table wrong, not just
    # imprecise; so do scores averaged differently, and not being able to check. The
    # tables are already written, so the run log, the CSVs and table_checks.json show
    # what is wrong. Every check runs and is recorded before the first failure is raised.
    checks, first_error = run_table_checks(
        main_recs,
        nom2_recs if args.compare_m2 else None,
        common_parts_only=args.common_parts_only,
        log=log,
    )
    table_checks_path = outdir / TABLE_CHECKS_FILE
    write_table_checks(checks, table_checks_path)
    if args.log_mlflow:
        urls = publish_table(
            main_recs,
            scores_json_dir,
            table_checks_path,
            checks,
            experiment=args.mlflow_experiment,
            log=log,
        )
        write_tracker_with_agg(fieldnames, rows, urls, outdir / TRACKER_WITH_AGG_FILE)
        log(f"  {TRACKER_WITH_AGG_FILE}: paste its {AGG_URL_COLUMN} column into the tracker")
    if first_error is not None:
        raise first_error


if __name__ == "__main__":
    main()
