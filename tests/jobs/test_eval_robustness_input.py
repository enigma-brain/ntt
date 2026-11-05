"""End-to-end smoke test for the input_robustness (M1) job pipeline.

Drives the per-session replica loop +
``NTTEvalRobustnessInputJob._write_csv`` / ``_write_run_meta`` against
a synthetic dataset + identity-style tokenizer so the test stays fast
and model-free, mirroring the structure of
``tests/test_validation_collation.py``.

This intentionally skips ``NTTBaseJob.__enter__`` / ``self.datamodule``
plumbing — those are exercised by integration runs. The smoke test
focuses on the contract that the runner produces ``results.jsonl`` and
``results.csv`` with expected columns when given a valid replica
dataloader.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader

from ntt.core.components.types import BinnedSpikesInput, DataType
from ntt.core.data.datamodule import collate
from ntt.core.data.replica.sampler import (
    ReplicaWindowIndex,
    WindowReplicaBatchSampler,
)
from ntt.evaluation.config import EvaluationConfig
from ntt.evaluation.config.metric_config import MetricInstanceConfig
from ntt.evaluation.core.types import MetricCategory, MetricDirection, MetricLevel
from ntt.evaluation.mlflow_logging import MLflowLoggingConfig
from ntt.evaluation.utils.replica_collation import (
    run_validation_evaluation_per_session_replicas,
)
from ntt.jobs.eval_robustness_input import (
    NTTEvalRobustnessInputJob,
    _append_core_robustness_score_rows,
    _corruption_label,
)

# Re-use the synthetic dataset from the replica iterator test.
from tests.evaluation.utils.test_replica_collation import (
    _identity_tokenizer_with_latents,
    _IdentityReplicaDataset,
    _SeqSourceSampler,
)


def _build_replica_loader(K: int):
    """Build a one-window replica loader with K corrupted replicas."""
    dataset = _IdentityReplicaDataset(n_units=3, time_bins_per_window=100)
    base_items = [
        ReplicaWindowIndex(recording_id="sess0", start=0.0, end=0.1, replica_id=0),
    ]
    batch_sampler = WindowReplicaBatchSampler(
        base_sampler=_SeqSourceSampler(base_items),
        n_replicas=K,
        max_windows=None,
    )
    return DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate,
    )


def test_eval_robustness_input_smoke_writes_jsonl_and_csv(tmp_path: Path):
    """Two corruption labels × 1 replica each → JSONL + CSV with expected cols.

    The "corruption labels" here are simulated by running the per-session
    driver twice (once per fake corruption) and feeding the resulting
    rows through the job's serializer helpers.
    """
    K = 1
    # Build an EvaluationConfig that registers the input_robustness (M1) metric.
    cfg = EvaluationConfig(
        metrics={
            "robustness_input": MetricInstanceConfig(
                metric_class="robustness_input",
            ),
        },
        compute_baseline=False,
    )

    rows: list[dict] = []
    for label in ["fake_drop_low", "fake_drop_high"]:
        loader = _build_replica_loader(K)
        reports = run_validation_evaluation_per_session_replicas(
            loader,
            _identity_tokenizer_with_latents,
            n_replicas=K,
            data_type=DataType.SPIKES,
            bin_size_seconds=0.001,
            token_bin_size_ms=1.0,
            eval_config=cfg,
            device=torch.device("cpu"),
            progress=False,
        )
        # Walk the reports the same way the job does.
        for session_id, entries in reports.items():
            for segment, payload in entries:
                report = payload["report"]
                for metric_name, mres in report.results.items():
                    raw = mres.raw_value
                    base = {
                        "session_id": session_id,
                        "segment_index": int(segment.segment_index),
                        "corruption": label,
                        "metric": metric_name,
                    }
                    if isinstance(raw, dict):
                        for sub_label, sub_val in raw.items():
                            scalar = (
                                sub_val.get("mean_token_nmse")
                                if isinstance(sub_val, dict)
                                else sub_val
                            )
                            rows.append(
                                {
                                    **base,
                                    "sub_label": sub_label,
                                    "value": (float(scalar) if scalar is not None else None),
                                    "status": "ok",
                                }
                            )
                    else:
                        try:
                            value = float(raw)
                        except (TypeError, ValueError):
                            value = None
                        rows.append(
                            {
                                **base,
                                "sub_label": None,
                                "value": value,
                                "status": "ok",
                            }
                        )

    assert rows, "expected at least one result row"
    # The input_robustness (M1) metric must appear at least once.
    assert any(r["metric"] == "robustness_input" for r in rows), (
        "input_robustness (M1) metric missing; saw metric names: "
        f"{sorted({r['metric'] for r in rows})}"
    )
    # Both corruption labels should be represented.
    assert {r["corruption"] for r in rows} == {"fake_drop_low", "fake_drop_high"}

    # Serialize via the job's CSV writer + write a JSONL file alongside.
    csv_path = tmp_path / "results.csv"
    NTTEvalRobustnessInputJob._write_csv(rows, csv_path)
    assert csv_path.exists() and csv_path.stat().st_size > 0

    jsonl_path = tmp_path / "results.jsonl"
    with jsonl_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    assert jsonl_path.exists() and jsonl_path.stat().st_size > 0

    # CSV columns should include the canonical row fields.
    header = csv_path.read_text().splitlines()[0].split(",")
    for required in ("session_id", "corruption", "metric", "raw_value", "status"):
        assert required in header, f"missing required column {required!r}; got {header}"


def test_per_session_csv_surfaces_n_windows_and_replica_std(tmp_path: Path):
    """The metric's n_windows and replica-spread std must reach results_per_session.csv.

    The input_robustness (M1) metric carries
    ``n_windows`` and ``per_label_std`` (equivalently
    ``std_token_nmse`` / ``std_per_dim_nmse`` per sub-label) in
    ``MetricResult.metadata`` / ``raw_value[label]``; the writer must
    keep them, rather than leaving ``results_per_session.csv`` with only a
    structural ``value_std`` (always 0 because the group has a single
    row).
    """
    rows = [
        {
            "session_id": "sess_a",
            "segment_index": 0,
            "corruption": "jitter|k_ms=1.0|p=0.2",
            "metric": "robustness_input",
            "sub_label": "jitter|k_ms=1.0|p=0.2",
            "value": 0.123,
            "value_std_replicas": 0.045,
            "n_windows": 64,
            "n_replicas_used": 10,
            "status": "ok",
        },
        {
            "session_id": "sess_a",
            "segment_index": 0,
            "corruption": "permute_neurons",
            "metric": "robustness_input",
            "sub_label": "permute_neurons",
            "value": 2.5,
            "value_std_replicas": 0.7,
            "n_windows": 64,
            "n_replicas_used": 10,
            "status": "ok",
        },
    ]

    path = tmp_path / "results_per_session.csv"
    NTTEvalRobustnessInputJob._write_per_session_csv(rows, path)

    text = path.read_text()
    header = text.splitlines()[0].split(",")
    for required in (
        "value_mean",
        "value_std",
        "n_windows_total",
        "n_windows_per_row",
        "value_std_replicas_mean",
        "value_std_replicas_per_row",
        "n_replicas_used_per_row",
    ):
        assert required in header, f"missing column {required!r}; got {header}"

    # Sanity-check the values for the jitter row.
    import csv

    with path.open() as fh:
        reader = csv.DictReader(fh)
        per_corruption = {row["corruption"]: row for row in reader}

    jitter = per_corruption["jitter|k_ms=1.0|p=0.2"]
    assert float(jitter["value_mean"]) == pytest.approx(0.123)
    assert int(jitter["n_windows_total"]) == 64
    assert float(jitter["value_std_replicas_mean"]) == pytest.approx(0.045)


def test_build_reports_from_m1_rows_surfaces_n_windows_in_metadata():
    """The post-hoc aggregator must expose the metric's n_windows / replica-std
    in the per-session metadata of the EvaluationReport, not just in results.csv."""
    job = NTTEvalRobustnessInputJob.model_construct(
        _eval_config=EvaluationConfig(
            metrics={
                "robustness_input": MetricInstanceConfig(
                    metric_class="robustness_input",
                ),
            },
            compute_baseline=False,
        ),
    )

    rows = [
        {
            "session_id": "sess_a",
            "segment_index": 0,
            "corruption": "jitter|k_ms=1.0|p=0.2",
            "metric": "robustness_input",
            "sub_label": "jitter|k_ms=1.0|p=0.2",
            "value": 0.5,
            "value_std_replicas": 0.05,
            "n_windows": 64,
            "n_replicas_used": 10,
            "status": "ok",
        },
    ]
    metric_schemas = {
        "robustness_input": {
            "category": MetricCategory.LATENT_QUALITY,
            "direction": MetricDirection.LOWER_IS_BETTER,
            "level": MetricLevel.POPULATION,
            "source_data_type": DataType.SPIKES,
            "value_range": (None, None),
        },
    }

    per_session_reports, _agg_report, _weights = job._build_reports_from_m1_rows(
        rows, metric_schemas
    )

    assert "sess_a" in per_session_reports
    sess_metrics = per_session_reports["sess_a"].results
    # Per-sub_label metric_key for the input_robustness (M1) row above.
    expected_key = "robustness_input/jitter|k_ms=1.0|p=0.2"
    assert expected_key in sess_metrics
    md = sess_metrics[expected_key].metadata
    assert md.get("n_windows_total") == 64
    assert md.get("n_windows_per_row") == [64]
    assert md.get("value_std_replicas_mean") == pytest.approx(0.05)
    assert md.get("value_std_replicas_per_row") == [0.05]
    assert md.get("n_replicas_used_per_row") == [10]


def test_corruption_label_uses_name_then_type():
    """``_corruption_label`` prefers ``params.name`` over the auto label."""
    from ntt.evaluation.config.schema import CorruptionParams

    p_named = CorruptionParams(type="drop", fraction=0.1, name="custom-label")
    assert _corruption_label(p_named) == "custom-label"

    p_auto = CorruptionParams(type="drop", fraction=0.1)
    auto = _corruption_label(p_auto)
    assert auto.startswith("drop") and "fraction=0.1" in auto


def test_eval_robustness_input_run_meta_writes_config_provenance(tmp_path: Path):
    """Core provenance fields should be written with a config snapshot sidecar."""
    meta_path = tmp_path / "run_meta.json"
    ri_cfg = SimpleNamespace(
        n_replicas=2,
        corruptions=[],
    )
    spec = SimpleNamespace(
        KIND="identity",
        model_dump=lambda mode: {"KIND": "identity", "token_bin_size_ms": 20.0},
    )

    NTTEvalRobustnessInputJob._write_run_meta(
        meta_path,
        ri_cfg=ri_cfg,
        skipped_counter={},
        total_pairs=3,
        spec=spec,
        checkpoint="model.ckpt",
        eval_config={"eval_window_seconds": 1.0, "robustness_input": {"n_eval_windows": 4}},
        data_config={"selection_name": "dev-val", "dataset_yaml": "datasets/dev.yaml"},
        hydra_overrides=["job=eval_robustness_input"],
        composed_config_yaml="job:\n  KIND: eval_robustness_input\n",
    )

    meta = json.loads(meta_path.read_text())
    assert meta["eval_config"]["eval_window_seconds"] == 1.0
    assert meta["data_config"]["selection_name"] == "dev-val"
    assert meta["hydra_overrides"] == ["job=eval_robustness_input"]
    assert meta["composed_config_snapshot_path"] == "config_snapshot.yaml"
    assert (tmp_path / "config_snapshot.yaml").read_text() == (
        "job:\n  KIND: eval_robustness_input\n"
    )


def test_eval_report_saves_what_the_comparability_check_compares(tmp_path: Path):
    """eval_report.json itself saves session ids, cross-session aggregation (with its
    weights when weighted) and run settings, so the aggregation reads them without
    the run's side files."""
    from ntt.evaluation.aggregation.comparability import (
        add_robustness_report_fields,
        report_session_summary,
    )
    from ntt.evaluation.config.schema import AggregationStrategy
    from ntt.evaluation.core.types import EvaluationReport
    from ntt.jobs.aggregation import _robustness_settings

    run_meta = NTTEvalRobustnessInputJob._write_run_meta(
        tmp_path / "run_meta.json",
        ri_cfg=SimpleNamespace(n_replicas=2, corruptions=[]),
        skipped_counter={},
        total_pairs=3,
        eval_window={"scored_window_seconds": 1.0},
        spec=SimpleNamespace(
            KIND="identity",
            model_dump=lambda **_: {"KIND": "identity", "bin_size_ms": 1.0},
        ),
    )
    job = NTTEvalRobustnessInputJob.model_construct(
        split="test", aggregation_strategy=AggregationStrategy.WEIGHTED_MEAN
    )
    job._eval_config = EvaluationConfig(compute_baseline=False, max_eval_seconds=300.0)
    report = EvaluationReport(results={})
    add_robustness_report_fields(
        report,
        run_meta,
        job._eval_config,
        split=job.split,
        session_ids=["s2", "s1"],
        aggregation_strategy=job.aggregation_strategy.value,
        session_weights={"s1": 3.0, "s2": 5.0},
    )
    job._write_eval_report_json(report, tmp_path / "eval_report.json")

    saved = json.loads((tmp_path / "eval_report.json").read_text())
    assert report_session_summary(saved) == {
        "session_ids": ["s1", "s2"],
        "n_sessions": 2,
        "aggregation_strategy": "weighted_mean",
    }
    assert saved["config"]["session_weights"] == {"s1": 3.0, "s2": 5.0}
    assert _robustness_settings(saved) == {
        "split": "test",
        "bin_size_ms": 1.0,
        "scored_window_seconds": 1.0,
        "max_eval_seconds": 300.0,
        "max_windows": None,
        "max_neurons": None,
        "evaluate_all_neurons": None,  # this test's run_meta does not pass it
        "neuron_subsample_seed": 42,
        "conversion_bin_size_ms": 1.0,
        "corruptions": [],
        "n_replicas": 2,
        "metrics": None,
    }


# ── Eval-time figure emission ─────────────────────────────────────────

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless for cluster runs.


def _make_input_job_skeleton() -> NTTEvalRobustnessInputJob:
    """Construct a minimally-validated job for testing the figure helpers
    in isolation. ``model_construct`` skips validation so the test does
    not need a real tokenizer/datamodule."""
    return NTTEvalRobustnessInputJob.model_construct()


def _write_text(path: Path, text: str = "x\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_eval_robustness_input_log_to_mlflow_uses_compact_artifacts_by_default(tmp_path: Path):
    """input_robustness (M1) should mirror latent_robustness (M2)'s compact-default artifact
    upload policy."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_text(out_dir / "results.jsonl", "{}\n")
    _write_text(out_dir / "results.csv")
    _write_text(out_dir / "results_per_session.csv")
    _write_text(out_dir / "run_meta.json", "{}\n")
    _write_text(out_dir / "eval_report.json", "{}\n")
    _write_text(out_dir / "poisson_comparison.json", "{}\n")
    _write_text(out_dir / "per_session" / "sess0.json", "{}\n")
    _write_text(out_dir / "curves" / "input-corruption-a.png")
    _write_text(out_dir / "rasters" / "input-corruption-c_sess0_w0.png")

    job = _make_input_job_skeleton()
    job.mlflow_logging = MLflowLoggingConfig(
        enabled=True,
        experiment_name="test-eval",
        upload_checkpoint=False,
        upload_artifacts_for_replotting=False,
    )
    job.ckpt_path = None
    job._tokenizer = None
    job._spec = None
    job.split = "val"

    captured: dict[str, object] = {}

    def _fake_log_robustness_run_to_mlflow(**kwargs):
        captured.update(kwargs)
        return "m1-run"

    with patch(
        "ntt.evaluation.mlflow_logging.log_robustness_run_to_mlflow",
        side_effect=_fake_log_robustness_run_to_mlflow,
    ):
        job._log_to_mlflow(
            output_dir=out_dir,
            ri_cfg=SimpleNamespace(corruptions=[]),
        )

    artifact_paths = [
        path.relative_to(out_dir).as_posix()
        for path in captured["artifact_paths"]  # type: ignore[index]
    ]
    assert sorted(artifact_paths) == [
        "eval_report.json",
        "results.csv",
        "run_meta.json",
    ]
    assert captured.get("include_jsonl_artifact") is False


def test_eval_robustness_input_log_to_mlflow_includes_replot_bundle_when_enabled(
    tmp_path: Path,
):
    """When explicitly requested, input_robustness (M1) should upload the extra replot/debug
    bundle."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _write_text(out_dir / "results.jsonl", "{}\n")
    _write_text(out_dir / "results.csv")
    _write_text(out_dir / "results_per_session.csv")
    _write_text(out_dir / "run_meta.json", "{}\n")
    _write_text(out_dir / "eval_report.json", "{}\n")
    _write_text(out_dir / "poisson_comparison.json", "{}\n")
    _write_text(out_dir / "per_session" / "sess0.json", "{}\n")
    _write_text(out_dir / "curves" / "input-corruption-a.png")
    _write_text(out_dir / "rasters" / "input-corruption-c_sess0_w0.png")

    job = _make_input_job_skeleton()
    job.mlflow_logging = MLflowLoggingConfig(
        enabled=True,
        experiment_name="test-eval",
        upload_checkpoint=False,
        upload_artifacts_for_replotting=True,
    )
    job.ckpt_path = None
    job._tokenizer = None
    job._spec = None
    job.split = "val"

    captured: dict[str, object] = {}

    def _fake_log_robustness_run_to_mlflow(**kwargs):
        captured.update(kwargs)
        return "m1-run"

    with patch(
        "ntt.evaluation.mlflow_logging.log_robustness_run_to_mlflow",
        side_effect=_fake_log_robustness_run_to_mlflow,
    ):
        job._log_to_mlflow(
            output_dir=out_dir,
            ri_cfg=SimpleNamespace(corruptions=[]),
        )

    artifact_paths = sorted(
        path.relative_to(out_dir).as_posix()
        for path in captured["artifact_paths"]  # type: ignore[index]
    )
    assert artifact_paths == [
        "curves/input-corruption-a.png",
        "eval_report.json",
        "per_session/sess0.json",
        "poisson_comparison.json",
        "rasters/input-corruption-c_sess0_w0.png",
        "results.csv",
        "results_per_session.csv",
        "run_meta.json",
    ]
    assert captured.get("include_jsonl_artifact") is True


def test_eval_robustness_input_emit_figures_writes_curves_and_rasters(tmp_path: Path):
    """``_emit_figures`` should leave a complete plot directory on disk.

    Builds a tiny set of csv_rows + a captured-raster mapping so both
    the aggregate curve (M1-a) and a single raster (M1-c) are emitted as
    PNG+SVG+PDF triplets.
    """
    from ntt.evaluation.config.schema import (
        CorruptionParams,
        RobustnessInputConfig,
    )

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    # Two drop-fraction strengths → one curve with two points.
    p_low = CorruptionParams(type="drop", fraction=0.05)
    p_high = CorruptionParams(type="drop", fraction=0.5)
    ri_cfg = RobustnessInputConfig(
        n_replicas=1,
        corruptions=[p_low, p_high],
        plot_examples=True,
    )
    label_low = _corruption_label(p_low)
    label_high = _corruption_label(p_high)
    csv_rows = [
        {
            "session_id": "sess0",
            "segment_index": 0,
            "corruption": label_low,
            "metric": "robustness_input",
            "sub_label": "replica_1",
            "value": 0.1,
            "status": "ok",
        },
        {
            "session_id": "sess0",
            "segment_index": 0,
            "corruption": label_high,
            "metric": "robustness_input",
            "sub_label": "replica_1",
            "value": 0.6,
            "status": "ok",
        },
    ]

    # Raster: one window per session with clean + per-label corrupted spikes.
    import numpy as np

    rng = np.random.default_rng(0)
    clean_TU = (rng.random((50, 4)) > 0.8).astype(np.float32)  # [T, U]
    corr_low = (rng.random((50, 4)) > 0.7).astype(np.float32)
    corr_high = (rng.random((50, 4)) > 0.5).astype(np.float32)
    captured_clean = {"sess0": {0: clean_TU}}
    captured_corrupted = {
        "sess0": {0: {label_low: corr_low, label_high: corr_high}},
    }
    captured_d_latent = {"sess0": {0: {label_low: 0.1, label_high: 0.6}}}

    job = _make_input_job_skeleton()
    job._emit_figures(
        output_dir=out_dir,
        csv_rows=csv_rows,
        ri_cfg=ri_cfg,
        bin_size_seconds=0.001,
        captured_clean=captured_clean,
        captured_corrupted=captured_corrupted,
        captured_d_latent=captured_d_latent,
    )

    # Curve A: PNG + SVG + PDF triplet.
    curves_dir = out_dir / "curves"
    for ext in ("png", "svg", "pdf"):
        assert (curves_dir / f"input-corruption-a.{ext}").exists(), (
            f"missing curves/input-corruption-a.{ext}"
        )

    # Raster M1-c: PNG + SVG + PDF triplet.
    rasters_dir = out_dir / "rasters"
    for ext in ("png", "svg", "pdf"):
        path = rasters_dir / f"input-corruption-c_sess0_w0.{ext}"
        assert path.exists(), f"missing {path}"


def test_eval_robustness_input_skips_rasters_when_plot_examples_false(tmp_path: Path):
    """``plot_examples=False`` should still emit the aggregate curve but
    no rasters.
    """
    from ntt.evaluation.config.schema import (
        CorruptionParams,
        RobustnessInputConfig,
    )

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    p = CorruptionParams(type="drop", fraction=0.1)
    ri_cfg = RobustnessInputConfig(n_replicas=1, corruptions=[p], plot_examples=False)
    csv_rows = [
        {
            "session_id": "sess0",
            "segment_index": 0,
            "corruption": _corruption_label(p),
            "metric": "robustness_input",
            "sub_label": "replica_1",
            "value": 0.3,
            "status": "ok",
        },
    ]

    job = _make_input_job_skeleton()
    job._emit_figures(
        output_dir=out_dir,
        csv_rows=csv_rows,
        ri_cfg=ri_cfg,
        bin_size_seconds=0.001,
        captured_clean={},
        captured_corrupted={},
        captured_d_latent={},
    )

    assert (out_dir / "curves" / "input-corruption-a.png").exists()
    assert not (out_dir / "rasters").exists()


def test_corruption_strength_helper():
    """``_corruption_strength`` reads the per-type strength field."""
    from ntt.evaluation.config.schema import CorruptionParams
    from ntt.jobs.eval_robustness_input import _corruption_strength

    assert _corruption_strength(CorruptionParams(type="drop", fraction=0.2)) == 0.2
    assert _corruption_strength(CorruptionParams(type="jitter", k_ms=5.0, p=0.5)) == 5.0
    assert _corruption_strength(CorruptionParams(type="window_shift", max_shift_ms=10.0)) == 10.0
    # permute_neurons has no continuous strength → sentinel 1.0.
    assert _corruption_strength(CorruptionParams(type="permute_neurons")) == 1.0


def test_append_core_robustness_score_rows_excludes_permutation():
    """The aggregate input_robustness (M1) score must ignore permutation rows."""
    rows = [
        {
            "session_id": "sess0",
            "segment_index": 0,
            "corruption": "window_shift|max_shift_ms=3.0",
            "metric": "robustness_input",
            "sub_label": "replica_1",
            "value": 0.2,
            "status": "ok",
        },
        {
            "session_id": "sess0",
            "segment_index": 0,
            "corruption": "jitter|k_ms=1.0|p=0.2",
            "metric": "robustness_input",
            "sub_label": "replica_1",
            "value": 0.4,
            "status": "ok",
        },
        {
            "session_id": "sess0",
            "segment_index": 0,
            "corruption": "drop_and_add|fraction=0.02",
            "metric": "robustness_input",
            "sub_label": "replica_1",
            "value": 0.8,
            "status": "ok",
        },
        {
            "session_id": "sess0",
            "segment_index": 0,
            "corruption": "permute_neurons",
            "metric": "robustness_input",
            "sub_label": "replica_1",
            "value": 10.0,
            "status": "ok",
        },
    ]

    augmented = _append_core_robustness_score_rows(rows)
    derived_rows = [
        row
        for row in augmented
        if row["metric"] == "robustness_input_core_avg"
        and row["corruption"] == "aggregate_non_population"
    ]

    assert len(derived_rows) == 1
    assert derived_rows[0]["value"] == pytest.approx((0.2 + 0.4 + 0.8) / 3.0)


def test_append_core_score_warns_when_components_dont_match(caplog):
    """A non-default corruption config (here drop fraction=0.05 instead of the
    hard-coded 0.02) makes the core-score component labels not match, so no
    aggregate row is produced. That must not happen silently — a warning is
    logged so the missing headline metric is visible."""
    import logging

    rows = [
        {
            "session_id": "sess0",
            "segment_index": 0,
            "corruption": "window_shift|max_shift_ms=3.0",
            "metric": "robustness_input",
            "sub_label": "replica_1",
            "value": 0.2,
            "status": "ok",
        },
        {
            "session_id": "sess0",
            "segment_index": 0,
            "corruption": "jitter|k_ms=1.0|p=0.2",
            "metric": "robustness_input",
            "sub_label": "replica_1",
            "value": 0.4,
            "status": "ok",
        },
        {
            "session_id": "sess0",
            "segment_index": 0,
            "corruption": "drop_and_add|fraction=0.05",  # non-default → no match
            "metric": "robustness_input",
            "sub_label": "replica_1",
            "value": 0.8,
            "status": "ok",
        },
    ]

    with caplog.at_level(logging.WARNING, logger="ntt.jobs.eval_robustness_input"):
        augmented = _append_core_robustness_score_rows(rows)

    derived_rows = [row for row in augmented if row["metric"] == "robustness_input_core_avg"]
    assert derived_rows == [], "no core score should be derivable from a non-matching config"
    assert any(
        "core" in rec.message.lower() and "robustness_input" in rec.message
        for rec in caplog.records
        if rec.levelno >= logging.WARNING
    ), (
        "expected a warning about the missing core score; got "
        f"{[r.message for r in caplog.records]}"
    )


def test_append_core_score_no_warning_on_complete_match(caplog):
    """When all three core components are present the aggregate is produced and
    no warning is logged (guards against the warning becoming noisy)."""
    import logging

    rows = [
        {
            "session_id": "sess0",
            "segment_index": 0,
            "corruption": label,
            "metric": "robustness_input",
            "sub_label": "replica_1",
            "value": v,
            "status": "ok",
        }
        for label, v in [
            ("window_shift|max_shift_ms=3.0", 0.2),
            ("jitter|k_ms=1.0|p=0.2", 0.4),
            ("drop_and_add|fraction=0.02", 0.8),
        ]
    ]

    with caplog.at_level(logging.WARNING, logger="ntt.jobs.eval_robustness_input"):
        augmented = _append_core_robustness_score_rows(rows)

    derived_rows = [row for row in augmented if row["metric"] == "robustness_input_core_avg"]
    assert len(derived_rows) == 1
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING], (
        "no warning expected when the core component set is complete"
    )


def test_params_to_spec_drop_and_add_returns_dropandaddspec():
    """``_params_to_spec`` should map ``drop_and_add`` → ``DropAndAddSpec``.

    Without this branch, the combined drop+add corruption is unreachable
    end-to-end (config validates but the runner raises). The default
    sweep also lists at least one ``drop_and_add`` entry so the
    combined corruption is exercised by the standard input_robustness (M1) run.
    """
    from ntt.core.data.components.corruption import DropAndAddSpec
    from ntt.evaluation.config.schema import (
        CorruptionParams,
        default_m1_corruptions_for_tests,
    )
    from ntt.jobs.eval_robustness_input import _params_to_spec

    spec = _params_to_spec(
        CorruptionParams(type="drop_and_add", fraction=0.1), eval_window_seconds=1.0
    )
    assert isinstance(spec, DropAndAddSpec)
    assert spec.fraction == 0.1

    # The test default sweep should include at least one drop_and_add
    # entry so the combined corruption is exercised by this test fixture.
    types = {p.type for p in default_m1_corruptions_for_tests()}
    assert "drop_and_add" in types


def test_params_to_spec_window_shift_fills_default_windows():
    """WindowShift should inherit the shared analysis default and infer
    the minimum valid source window when omitted."""
    from ntt.core.data.components.corruption import WindowShiftSpec
    from ntt.evaluation.config.schema import (
        DEFAULT_M1_ANALYSIS_WINDOW_MS,
        CorruptionParams,
    )
    from ntt.jobs.eval_robustness_input import _params_to_spec

    spec = _params_to_spec(
        CorruptionParams(type="window_shift", max_shift_ms=25.0),
        eval_window_seconds=1.0,
    )
    assert isinstance(spec, WindowShiftSpec)
    assert spec.analysis_window_ms == DEFAULT_M1_ANALYSIS_WINDOW_MS
    assert spec.source_window_ms == DEFAULT_M1_ANALYSIS_WINDOW_MS + 50.0


def test_params_to_spec_window_shift_analysis_window_tracks_eval_window():
    """The window_shift analysis window defaults to the authoritative eval window.

    The eval window is the single source of truth; the default window_shift
    analysis window tracks it (eval_window_seconds * 1000 ms). A per-corruption
    ``analysis_window_ms`` would still override this default.
    """
    from ntt.core.data.components.corruption import WindowShiftSpec
    from ntt.evaluation.config.schema import CorruptionParams
    from ntt.jobs.eval_robustness_input import _params_to_spec

    spec = _params_to_spec(
        CorruptionParams(type="window_shift", max_shift_ms=3.0),
        eval_window_seconds=1.0,
    )
    assert isinstance(spec, WindowShiftSpec)
    assert spec.analysis_window_ms == 1000.0
    assert spec.source_window_ms == 1000.0 + 6.0

    spec2 = _params_to_spec(
        CorruptionParams(type="window_shift", max_shift_ms=3.0),
        eval_window_seconds=2.0,
    )
    assert isinstance(spec2, WindowShiftSpec)
    assert spec2.analysis_window_ms == 2000.0
    assert spec2.source_window_ms == 2000.0 + 6.0


def test_window_shift_analysis_window_tracks_eval_window():
    """The default window_shift analysis span tracks the authoritative eval window."""
    from ntt.jobs.eval_robustness_input import _window_shift_default_analysis_ms

    assert _window_shift_default_analysis_ms(1.0) == 1000.0
    assert _window_shift_default_analysis_ms(2.0) == 2000.0


def test_robustness_input_config_expands_list_valued_corruption_sweeps():
    """Compact list syntax in YAML should expand to multiple corruption specs."""
    from ntt.evaluation.config.schema import RobustnessInputConfig

    cfg = RobustnessInputConfig(
        corruptions=[
            {"type": "window_shift", "max_shift_ms": [25.0, 5.0, 1.0, 0.0]},
            {"type": "drop_and_add", "fraction": 0.1},
        ]
    )

    assert [c.max_shift_ms for c in cfg.corruptions[:4]] == [25.0, 5.0, 1.0, 0.0]
    assert cfg.corruptions[4].type == "drop_and_add"
