#!/usr/bin/env python
"""Build the cohort table that the evaluation battery consumes.

The cohort table is a CSV with one row per tokenizer. You fill two columns;
the battery fills the rest as each axis completes:

    tokenizer    (required) display name; must be filesystem-safe and unique
    ckpt_uri     (required) local path or mlflow-artifacts:/... URI
    mlflow_*     (empty here) one per evaluation axis, filled as runs finish

Once the axis columns are populated the same file is what the aggregation
layer reads, so there is no second format to maintain. See
``docs/eval_battery_runbook.md`` §6.

Three discovery sources, all producing the same two columns:

    --from-list FILE      one checkpoint per line, or a cohort CSV to re-read
    --from-dir DIR        glob a directory tree for checkpoints
    --from-mlflow EXP     query an MLflow experiment for one checkpoint per run
    --check FILE          validate an existing (e.g. hand-written) table and exit

``--verify`` loads every checkpoint before you launch anything. That is the
cheapest check in the battery and it forecloses the most expensive failure, so
prefer running it once here over discovering a bad cohort mid-fan-out.

Examples
--------
    # A human with a directory of checkpoints
    python scripts/eval/build_cohort.py --from-dir /data/ckpts --out cohort.csv

    # An agent (or human) pointing at a tracker experiment
    python scripts/eval/build_cohort.py --from-mlflow <experiment-name> \
        --out cohort.csv --verify
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from pathlib import Path

LOGGER = logging.getLogger("build_cohort")

# Axis columns, empty at build time and filled as each axis completes. These names
# are the ones the aggregation layer's row->URI mapping reads
# (``ntt.evaluation.aggregation.tracker_io.row_to_source_uris``); keep them in sync
# with it; tests/scripts/test_build_cohort.py checks every column is read.
AXIS_COLUMNS: tuple[str, ...] = (
    "mlflow_recon",
    "mlflow_latent_snr",
    "mlflow_latent_input_corruption",
    "mlflow_latent_latent_corruption",
    "mlflow_encoding",
    "mlflow_decoding",
)

BASE_COLUMNS: tuple[str, ...] = ("tokenizer", "ckpt_uri")

# Checkpoint artifact layouts seen in the wild, most specific first. Training runs
# do not agree on this, so resolve per run rather than assuming one.
_CKPT_CANDIDATES: tuple[str, ...] = (
    "checkpoints/last.ckpt",
    "last.ckpt",
)
_LAST_DIR_RE = re.compile(r"^(?:checkpoints/)?last(?:-v(\d+))?$")


def slugify(name: str) -> str:
    """Filesystem-safe tokenizer name.

    Mirrors the aggregation layer's ``scores_slug`` so a name chosen here survives
    round-tripping into per-tokenizer filenames without silently changing.
    """
    return re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_").lower()


def dedupe(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Make tokenizer names unique, appending a short discriminator on collision.

    Duplicate run names are common (reruns, replicates). Silently collapsing them
    loses cohort members; silently overwriting loses results. Both are worse than
    a mangled-but-distinct name, which is at least visible.
    """
    seen: dict[str, int] = {}
    out: list[dict[str, str]] = []
    for row in rows:
        name = row["tokenizer"]
        if name not in seen:
            seen[name] = 1
            out.append(row)
            continue
        disc = (row.get("_run_id") or "")[:8] or str(seen[name])
        seen[name] += 1
        new = f"{name}_{slugify(disc)}" if disc else f"{name}_{seen[name]}"
        LOGGER.warning("Duplicate tokenizer name %r -> %r", name, new)
        row = {**row, "tokenizer": new}
        out.append(row)
    return out


def is_uri(value: str) -> bool:
    """True for a non-local reference — ``mlflow-artifacts:/…``, ``s3://…``, and friends."""
    return "://" in value or value.startswith("mlflow")


def normalise_uri(value: str) -> str:
    """Absolutise a local path; leave a URI untouched.

    A relative path in a list file resolves against whoever happens to be the current
    directory at fan-out time, which is rarely where the list was written, and ``~`` is
    not expanded by every consumer. Both are silent failures later, so settle them here.
    """
    value = value.strip()
    if is_uri(value):
        return value
    return str(Path(value).expanduser().resolve())


def read_list(path: Path) -> list[dict[str, str]]:
    """Read checkpoints from a list file, or from a cohort CSV.

    Two accepted shapes, detected rather than configured:

    * **A cohort CSV** — anything whose header carries ``tokenizer`` and ``ckpt_uri``.
      Parsed as CSV so the names already chosen are preserved. This makes a table
      round-trip through ``--from-list`` instead of being silently renamed and, worse,
      having its URIs mangled by the trailing empty axis columns.
    * **A plain list** — one checkpoint per line, ``#`` comments and blanks ignored. A
      line may be a bare URI, or ``name,uri`` / ``name<TAB>uri`` to set the name
      explicitly. Parsed with the csv module so a quoted field containing a comma
      survives.
    """
    text = path.read_text()
    lines = [ln for ln in text.splitlines() if ln.split("#", 1)[0].strip()]
    if not lines:
        return []

    header = next(csv.reader([lines[0]]))
    if {"tokenizer", "ckpt_uri"} <= {h.strip() for h in header}:
        LOGGER.info("%s looks like a cohort table; reading it as CSV", path.name)
        with path.open(newline="") as f:
            return [
                {
                    "tokenizer": slugify((r.get("tokenizer") or "").strip()),
                    "ckpt_uri": normalise_uri(r.get("ckpt_uri") or ""),
                }
                for r in csv.DictReader(f)
                if (r.get("ckpt_uri") or "").strip()
            ]

    rows: list[dict[str, str]] = []
    for raw in lines:
        line = raw.split("#", 1)[0].strip()
        name, uri = _split_line(line)
        rows.append({"tokenizer": slugify(name), "ckpt_uri": normalise_uri(uri)})
    return rows


def _split_line(line: str) -> tuple[str, str]:
    """Split a list line into (name, uri), deriving the name when absent."""
    if "\t" in line:
        left, _, right = line.partition("\t")
        if left.strip() and right.strip():
            return left.strip(), right.strip()
    fields = [f.strip() for f in next(csv.reader([line]))]
    if len(fields) >= 2 and fields[0] and fields[1]:
        return fields[0], fields[1]
    uri = fields[0] if fields else line
    return name_from_path(uri), uri


def name_from_path(uri: str) -> str:
    """Derive a tokenizer name from a checkpoint path.

    Uses the nearest meaningful ancestor rather than the filename, because most
    layouts name the file ``last.ckpt`` and put the identity in a parent directory.
    """
    p = PurePathish(uri)
    parts = [q for q in p.parts if q not in ("/", "")]
    stem = p.stem
    if stem.startswith("last") or stem in {"model", "final"}:
        for anc in reversed(parts[:-1]):
            if anc not in {"checkpoints", "artifacts"} and not _LAST_DIR_RE.match(anc):
                return anc
    return stem


class PurePathish:
    """Minimal path view that works for both local paths and ``scheme:/`` URIs."""

    def __init__(self, uri: str) -> None:
        self._raw = uri
        body = uri.split("://", 1)[-1] if "://" in uri else uri.split(":/", 1)[-1]
        self.parts = tuple(p for p in body.split("/") if p)

    @property
    def stem(self) -> str:
        last = self.parts[-1] if self.parts else self._raw
        return last[: -len(".ckpt")] if last.endswith(".ckpt") else last


def scan_dir(root: Path, pattern: str) -> list[dict[str, str]]:
    """Find checkpoints under ``root`` matching ``pattern``."""
    rows: list[dict[str, str]] = []
    for p in sorted(root.glob(pattern)):
        if p.is_file():
            rows.append({"tokenizer": slugify(name_from_path(str(p))), "ckpt_uri": str(p)})
    return rows


def _artifact_lister(client, run_id: str) -> tuple[object, str]:
    """Return ``(list_artifacts, artifact_uri)`` for one run, bypassing the client wrapper.

    ``MlflowClient.list_artifacts`` resolves through ``runs:/`` and, for any non-empty
    path, also asks the tracker's ``logged-models`` search API — which tracking servers
    older than the client answer with a 404, so every nested listing raises. Going
    straight to the run's artifact repository lists the same paths without that lookup.

    The run's own ``artifact_uri`` comes back too, because building a URI by hand assumes
    the proxied-artifact layout and silently fabricates unusable URIs for an experiment
    stored on ``s3://`` or ``file://``.
    """
    from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository

    artifact_uri = client.get_run(run_id).info.artifact_uri
    repo = get_artifact_repository(artifact_uri, tracking_uri=client.tracking_uri)
    return repo.list_artifacts, artifact_uri


def resolve_ckpt_artifact(ls, run_id: str, ckpt_name: str | None = None) -> str | None:
    """Return the artifact path of a run's last-epoch checkpoint, or None.

    ``ls`` is a ``list_artifacts(path=None)`` callable (see ``_artifact_lister``).

    Layouts vary between runs, so this probes rather than assuming, in order of
    specificity and returning as soon as it has an answer. Ordering matters for cost as
    well as correctness: a run's artifact tree often holds thousands of raster images, and
    the exhaustive walk below would list every one of them. Healthy runs never reach it.

    When several ``last-vN`` variants exist the highest N wins, which is the most recently
    written one. Note that what a run *means* by ``last`` is whatever its checkpoint
    callback meant — not necessarily the final epoch.
    """
    if ckpt_name:
        # Listing the parent, not the root: an explicit name is usually nested.
        parent, _, _ = ckpt_name.rpartition("/")
        return ckpt_name if any(a.path == ckpt_name for a in ls(parent or None)) else None

    root = {a.path: a for a in ls()}

    # 1. A canonical file, at the root or one level down under checkpoints/.
    for cand in _CKPT_CANDIDATES:
        parent, _, _ = cand.rpartition("/")
        pool = root if not parent else {a.path: a for a in ls(parent)} if parent in root else {}
        if cand in pool and not pool[cand].is_dir:
            return cand

    # 2. last/ , last-v1/ , last-v2/ … — highest version wins.
    versioned = [
        (int(m.group(1) or 0), path)
        for path, art in root.items()
        if art.is_dir and (m := _LAST_DIR_RE.match(path))
    ]
    for _, dirpath in sorted(versioned, reverse=True):
        for a in ls(dirpath):
            if a.path.endswith(".ckpt"):
                return a.path

    # 3. Same, nested under checkpoints/.
    if "checkpoints" in root and root["checkpoints"].is_dir:
        nested = {a.path: a for a in ls("checkpoints")}
        versioned = [
            (int(m.group(1) or 0), path)
            for path, art in nested.items()
            if art.is_dir and (m := _LAST_DIR_RE.match(path))
        ]
        for _, dirpath in sorted(versioned, reverse=True):
            for a in ls(dirpath):
                if a.path.endswith(".ckpt"):
                    return a.path

    # 4. Exhaustive one-level walk. Only reached when nothing canonical exists, so the
    #    cost of listing large sibling directories is paid once, on an odd run.
    listing = dict(root)
    for top, art in list(root.items()):
        if art.is_dir:
            for a in ls(top):
                listing[a.path] = a

    ckpts = [p for p, a in listing.items() if p.endswith(".ckpt") and not a.is_dir]
    if len(ckpts) == 1:
        return ckpts[0]
    if ckpts:
        LOGGER.warning("run %s: %d checkpoints, none canonical: %s", run_id, len(ckpts), ckpts)
    elif any(a.is_dir for a in listing.values()):
        # Distinguish "searched and found nothing" from "nothing to search". A checkpoint
        # more than one level deep is invisible here and would otherwise look identical
        # to a run that never wrote one.
        LOGGER.warning(
            "run %s: no .ckpt within one level of the artifact root; if it is nested "
            "deeper, pass --ckpt-name with its exact path",
            run_id,
        )
    return None


def from_mlflow(
    experiment: str,
    *,
    include_failed: bool = False,
    ckpt_name: str | None = None,
) -> list[dict[str, str]]:
    """Enumerate an MLflow experiment's runs and resolve one checkpoint each."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    exp = client.get_experiment_by_name(experiment) or (
        client.get_experiment(experiment) if experiment.isdigit() else None
    )
    if exp is None:
        raise SystemExit(f"No such MLflow experiment: {experiment!r}")
    LOGGER.info("experiment %r -> id %s", experiment, exp.experiment_id)

    wanted = {"FINISHED"} | ({"FAILED"} if include_failed else set())
    rows: list[dict[str, str]] = []
    skipped: list[str] = []
    page = None
    while True:
        batch = client.search_runs(
            [exp.experiment_id],
            max_results=1000,
            page_token=page,
            # Deterministic, sweep-ordered output. Without this the row order is whatever
            # the server defaults to, which reads as scrambled in the aggregated table.
            order_by=["attributes.start_time ASC"],
        )
        for run in batch:
            status = run.info.status
            name = run.info.run_name or run.info.run_id
            if status not in wanted:
                skipped.append(f"{name} ({status})")
                continue
            ls, artifact_uri = _artifact_lister(client, run.info.run_id)
            art = resolve_ckpt_artifact(ls, run.info.run_id, ckpt_name)
            if art is None:
                skipped.append(f"{name} (no checkpoint artifact)")
                continue
            rows.append(
                {
                    "tokenizer": slugify(name),
                    "ckpt_uri": f"{artifact_uri.rstrip('/')}/{art}",
                    "_run_id": run.info.run_id,
                }
            )
        page = getattr(batch, "token", None)
        if not page:
            break

    for s in skipped:
        LOGGER.info("skipped %s", s)
    if not include_failed and any("(FAILED)" in s for s in skipped):
        LOGGER.warning(
            "FAILED runs were skipped. Their last checkpoint is often a perfectly good "
            "tokenizer — pass --include-failed if the cohort should contain them."
        )
    return rows


def verify(rows: list[dict[str, str]]) -> list[tuple[str, str]]:
    """Load every checkpoint. Returns [(tokenizer, error)] for the ones that fail."""
    from ntt.core.model import load_tokenizer_from_checkpoint

    failures: list[tuple[str, str]] = []
    for row in rows:
        uri = row["ckpt_uri"]
        try:
            local = _localise(uri)
            tk = load_tokenizer_from_checkpoint(local, map_location="cpu", weights_only=False)
            LOGGER.info("OK   %-40s %s", row["tokenizer"], type(tk).__name__)
        except Exception as e:  # noqa: BLE001 — reporting every failure beats stopping at one
            LOGGER.error("FAIL %-40s %s: %s", row["tokenizer"], type(e).__name__, e)
            failures.append((row["tokenizer"], f"{type(e).__name__}: {e}"))
    return failures


def _localise(uri: str) -> str:
    """Download a tracker URI to a local path; pass local paths through."""
    if uri.startswith(("mlflow-artifacts:", "mlflow:")):
        import mlflow.artifacts

        return str(mlflow.artifacts.download_artifacts(uri))
    return str(Path(uri).expanduser().resolve())


# Schemes the eval jobs know how to resolve. A local path is always fine; anything else
# has to be something ``resolve_artifact_path`` handles, or the job fails at load time with
# an error that says nothing about the table.
_RESOLVABLE_SCHEMES: tuple[str, ...] = ("mlflow-artifacts:", "mlflow:")


def check_rows(rows: list[dict[str, str]]) -> list[str]:
    """Return a list of problems with a cohort table. Empty means usable.

    This is the answer to "I wrote the table by hand — is it right?". Everything checked
    here is something that otherwise surfaces once a job is already running.
    """
    problems: list[str] = []
    if not rows:
        return ["table has no rows"]

    seen_names: dict[str, int] = {}
    seen_uris: dict[str, list[str]] = {}
    for i, row in enumerate(rows, start=2):  # +2: 1-based, and row 1 is the header
        name = (row.get("tokenizer") or "").strip()
        uri = (row.get("ckpt_uri") or "").strip()

        if not name:
            problems.append(f"row {i}: empty tokenizer")
        elif name != slugify(name):
            problems.append(
                f"row {i}: tokenizer {name!r} is not filesystem-safe; use {slugify(name)!r} "
                "— it becomes a directory name and a filename stem downstream"
            )
        seen_names[name] = seen_names.get(name, 0) + 1

        if not uri:
            problems.append(f"row {i}: empty ckpt_uri")
            continue
        seen_uris.setdefault(uri, []).append(name)

        if is_uri(uri):
            if not uri.startswith(_RESOLVABLE_SCHEMES):
                problems.append(
                    f"row {i}: {uri.split('://')[0]}:// URIs are not resolved by the eval "
                    f"jobs — use a local path or one of {', '.join(_RESOLVABLE_SCHEMES)}"
                )
        elif not Path(uri).expanduser().exists():
            problems.append(f"row {i}: local path does not exist: {uri}")

    problems += [f"duplicate tokenizer {n!r} ({c} rows)" for n, c in seen_names.items() if c > 1]
    for uri, names in seen_uris.items():
        if len(names) > 1:
            problems.append(f"same ckpt_uri used by {len(names)} rows ({', '.join(names)}): {uri}")
    return problems


def check_table(path: Path) -> int:
    """Validate an existing cohort table. Returns a process exit code."""
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = [h.strip() for h in (reader.fieldnames or [])]
        rows = [dict(r) for r in reader]

    problems: list[str] = []
    for col in BASE_COLUMNS:
        if col not in fieldnames:
            problems.append(f"missing required column {col!r}")
    missing_axes = [c for c in AXIS_COLUMNS if c not in fieldnames]
    problems += check_rows(rows)

    for p in problems:
        LOGGER.error("%s", p)
    # Absent axis columns are a warning, not an error: the battery can add them, but a
    # consumer silently drops an axis whose column does not exist, so say so now.
    if missing_axes:
        LOGGER.warning(
            "axis column(s) absent: %s. Consumers silently drop an axis with no column — "
            "add them empty rather than leaving them out.",
            ", ".join(missing_axes),
        )
    if problems:
        print(f"{path}: {len(problems)} problem(s) — not usable as-is")
        return 1
    print(f"{path}: OK, {len(rows)} row(s){', axis columns incomplete' if missing_axes else ''}")
    return 0


def write_cohort(rows: list[dict[str, str]], out: Path, *, axis_columns=AXIS_COLUMNS) -> None:
    """Write the cohort table, with the axis columns present but empty."""
    fieldnames = [*BASE_COLUMNS, *axis_columns]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--from-list", type=Path, help="file with one checkpoint per line, or a cohort CSV"
    )
    src.add_argument("--from-dir", type=Path, help="directory to scan for checkpoints")
    src.add_argument("--from-mlflow", type=str, help="MLflow experiment name or id")
    src.add_argument(
        "--check",
        type=Path,
        help="validate an existing cohort table (e.g. a hand-written one) and exit",
    )
    parser.add_argument("--out", type=Path, help="cohort CSV to write; required unless --check")
    parser.add_argument("--glob", default="**/*.ckpt", help="pattern for --from-dir")
    parser.add_argument(
        "--ckpt-name",
        default=None,
        help="exact artifact path to take from each run instead of probing layouts",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="include FAILED training runs (their last checkpoint is often usable)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="load every checkpoint; exit non-zero if any fails (downloads remote URIs)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if args.check:
        return check_table(args.check)

    if args.out is None:
        LOGGER.error("--out is required unless --check is used.")
        return 2

    if args.from_list:
        rows = read_list(args.from_list)
    elif args.from_dir:
        rows = scan_dir(args.from_dir, args.glob)
    else:
        rows = from_mlflow(
            args.from_mlflow, include_failed=args.include_failed, ckpt_name=args.ckpt_name
        )

    if not rows:
        LOGGER.error("No checkpoints found; nothing written.")
        return 1

    rows = dedupe(rows)
    _warn_on_opaque_names(rows)
    # Surface the same problems --check would, so a bad list is caught here rather than by
    # whichever job happens to run first.
    for problem in check_rows(rows):
        LOGGER.warning("%s", problem)

    # Write the table before the verify gate: a single unloadable checkpoint should not
    # discard the discovery work for every other row.
    write_cohort(rows, args.out)
    print(f"Wrote {args.out} with {len(rows)} tokenizer(s)")

    if args.verify:
        failures = verify(rows)
        if failures:
            LOGGER.error(
                "%d/%d checkpoints failed to load: %s. The table was still written, but the "
                "eval branch probably cannot load this cohort — see the runbook's "
                "prerequisites before launching.",
                len(failures),
                len(rows),
                ", ".join(t for t, _ in failures),
            )
            return 1

    print("Fill nothing else by hand: the battery populates the axis columns as it runs.")
    return 0


def _warn_on_opaque_names(rows: list[dict[str, str]]) -> None:
    """Warn when a derived name is a bare run id rather than something readable.

    Happens when tracker URIs are fed through --from-list: the identity of the run lives
    in the tracker, not the path, so the name falls back to the run-id hex. The CSV's
    ``tokenizer`` column is the cohort's identity, so silently renaming every row is worse
    than it looks — round-tripping --from-mlflow output through --from-list does exactly
    that.
    """
    opaque = [r["tokenizer"] for r in rows if re.fullmatch(r"[0-9a-f]{16,}", r["tokenizer"])]
    if opaque:
        LOGGER.warning(
            "%d row(s) named after a run id (%s...). Use 'name,uri' lines to set readable "
            "names — the tokenizer column is the cohort's identity downstream.",
            len(opaque),
            opaque[0][:12],
        )


if __name__ == "__main__":
    sys.exit(main())
