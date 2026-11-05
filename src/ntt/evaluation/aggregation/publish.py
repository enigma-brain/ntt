"""Publish a finished ``category_scores.json`` as an MLflow run.

The one place aggregation scores reach MLflow: ``build_aggregation_tables.py --log-mlflow``
and ``NTTAggregationJob``'s ``mlflow_logging`` both call ``publish_category_scores``. It logs
to the active tracking URI (``MLFLOW_TRACKING_URI``) and never sets one.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

DEFAULT_EXPERIMENT = "ntt-aggregation"
# The artifact name the MLflow read route (scripts/analysis/data.py) downloads.
SCORES_ARTIFACT = "category_scores.json"
TOKENIZER_TAG = "ntt.agg.tokenizer"
SCORES_SHA256_TAG = "ntt.agg.scores_sha256"
# Over everything the run carries (name, tags, every artifact): what a later call matches.
PUBLISH_SHA256_TAG = "ntt.agg.publish_sha256"
CKPT_URI_TAG = "ntt.agg.ckpt_uri"

_N_METRICS_NAMES = {
    "single_neuron_recon": "n_metrics_sn_recon",
    "population_structure": "n_metrics_population_structure",
    "latent_structure_and_robustness": "n_metrics_latent_structure_and_robustness",
}


@dataclass(frozen=True)
class Published:
    url: str
    # True: the experiment already had a run with exactly this content.
    reused: bool


def run_url(tracking_uri: str, experiment_id: str, run_id: str) -> str:
    """The run's UI URL, the form ``tracker_io.parse_run_url`` reads."""
    return f"{tracking_uri.rstrip('/')}/#/experiments/{experiment_id}/runs/{run_id}"


def publish_category_scores(
    scores_json: Path,
    *,
    experiment: str,
    run_name: str,
    tags: Mapping[str, str] | None = None,
    artifacts: Sequence[Path] = (),
) -> Published:
    """Log ``scores_json`` as a run named ``run_name`` in ``experiment``, unless the
    experiment already has a run with exactly this content: the same file, name, tags and
    artifacts. A run whose ``tags`` or ``artifacts`` (e.g. the table checks) differ is a
    new run, so a reused run never carries another build's verdict.

    The run carries the seven scores and the ``n_metrics_*`` counts as metrics (``None`` →
    NaN), the file as ``category_scores.json`` plus ``artifacts``, and ``tags`` plus the
    tokenizer, the file's sha256 and the checkpoint URI from its metadata.
    """
    content = scores_json.read_bytes()
    doc = json.loads(content)
    run_tags = {**(tags or {}), TOKENIZER_TAG: run_name}
    ckpt_uri = (doc.get("metadata") or {}).get("ckpt_uri")
    if ckpt_uri:
        run_tags[CKPT_URI_TAG] = ckpt_uri
    run_tags[SCORES_SHA256_TAG] = hashlib.sha256(content).hexdigest()
    digest = _publish_digest(run_name, run_tags, artifacts)
    tracking_uri = mlflow.get_tracking_uri()
    client = MlflowClient(tracking_uri)

    found = client.get_experiment_by_name(experiment)
    experiment_id = (
        found.experiment_id if found is not None else client.create_experiment(experiment)
    )
    for run in client.search_runs(
        [experiment_id], filter_string=f"tags.`{PUBLISH_SHA256_TAG}` = '{digest}'", max_results=1
    ):
        return Published(run_url(tracking_uri, experiment_id, run.info.run_id), reused=True)

    run_id = client.create_run(experiment_id, run_name=run_name, tags=run_tags).info.run_id
    try:
        for name, value in (doc.get("scores") or {}).items():
            client.log_metric(run_id, name, math.nan if value is None else float(value))
        diagnostics = doc.get("diagnostics") or {}
        for column, metric in _N_METRICS_NAMES.items():
            entry = diagnostics.get(column)
            if isinstance(entry, dict) and entry.get("n_metrics") is not None:
                client.log_metric(run_id, metric, entry["n_metrics"])
        with tempfile.TemporaryDirectory() as tmp:
            named = Path(tmp) / SCORES_ARTIFACT
            shutil.copyfile(scores_json, named)
            client.log_artifact(run_id, str(named))
        for path in artifacts:
            client.log_artifact(run_id, str(path))
        # Last: only a completely logged run can be found, and reused, by a later call.
        client.set_tag(run_id, PUBLISH_SHA256_TAG, digest)
    except BaseException:
        # Best effort: the server that failed the upload may fail this too, and its error
        # must not replace the one being raised.
        with contextlib.suppress(Exception):
            client.set_terminated(run_id, status="FAILED")
        raise
    client.set_terminated(run_id)
    return Published(run_url(tracking_uri, experiment_id, run_id), reused=False)


def _publish_digest(run_name: str, tags: Mapping[str, str], artifacts: Sequence[Path]) -> str:
    """sha256 over the run name, the tags (the scores file's sha256 among them) and each
    artifact's name and bytes."""
    h = hashlib.sha256(json.dumps([run_name, sorted(tags.items())]).encode())
    for path in artifacts:
        h.update(json.dumps(Path(path).name).encode())
        h.update(hashlib.sha256(Path(path).read_bytes()).digest())
    return h.hexdigest()
