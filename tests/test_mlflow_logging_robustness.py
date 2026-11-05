"""Core tests for the input_robustness (M1)/latent_robustness (M2) robustness JSONL → MLflow
flattener.

Synthetic JSONL records exercise: per-strength aggregation (mean across
segments/replicas), step ordering (sorted by strength/epsilon), and the
auxiliary x-axis series. Skipped/non-ok records are excluded from the
output series and counted in the summary scalars.
"""

from __future__ import annotations

import json
from pathlib import Path

from ntt.evaluation.mlflow_logging import (
    _robustness_metadata_params,
    extract_input_robustness_jsonl_summary_metrics,
    flatten_input_robustness_jsonl_to_metric_series,
    flatten_latent_robustness_jsonl_to_metric_series,
    remap_input_robustness_metric_keys_for_mlflow,
)
from ntt.evaluation.utils.config_provenance import (
    composed_config_yaml,
    config_subtree_to_jsonable,
    write_config_provenance_sidecar,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def test_run_meta_artifact_contains_config_provenance_fields(tmp_path: Path):
    """The on-disk run_meta artifact should carry config provenance fields."""
    run_meta_path = tmp_path / "run_meta.json"
    run_meta = {
        "tokenizer_config": {"KIND": "identity", "token_bin_size_ms": 20.0},
    }
    run_meta.update(
        write_config_provenance_sidecar(
            run_meta_path=run_meta_path,
            eval_config={"eval_window_seconds": 1.0},
            data_config={"root": "/tmp/data", "val_selections": []},
            hydra_overrides=["job=eval_robustness_input"],
            composed_config_yaml_text="job:\n  KIND: eval_robustness_input\n",
        )
    )
    run_meta_path.write_text(json.dumps(run_meta, indent=2))

    loaded = json.loads(run_meta_path.read_text())
    assert {
        "tokenizer_config",
        "data_config",
        "eval_config",
        "hydra_overrides",
        "composed_config_snapshot_path",
    }.issubset(loaded)
    assert (tmp_path / loaded["composed_config_snapshot_path"]).exists()


def test_config_provenance_missing_composed_config_warns_and_continues(tmp_path: Path, caplog):
    """A missing composed config must warn-and-skip, not abort the eval.

    The provenance step runs after compute, so a None composed config (or
    None eval/data config) should record null fields rather than raise.
    """
    run_meta_path = tmp_path / "run_meta.json"
    with caplog.at_level("WARNING"):
        fields = write_config_provenance_sidecar(
            run_meta_path=run_meta_path,
            eval_config=None,
            data_config=None,
            hydra_overrides=["job=eval_robustness_input"],
            composed_config_yaml_text=None,
        )

    # No sidecar written; provenance fields present but null where unavailable.
    assert fields["composed_config_snapshot_path"] is None
    assert fields["eval_config"] is None
    assert fields["data_config"] is None
    assert fields["hydra_overrides"] == ["job=eval_robustness_input"]
    assert not (tmp_path / "config_snapshot.yaml").exists()
    assert any(record.levelname == "WARNING" for record in caplog.records)


def test_robustness_params_derive_dataset_identity_from_real_data_config():
    """Searchable dataset params must come from fields the real data config
    actually carries.

    ``NTTDataModuleConfig`` has no ``selection_name``/``dataset_yaml`` fields,
    so the dataset identity is derived from ``val_selections`` (brainsets +
    session count) instead.
    """
    from ntt.core.data.components.config import SelectionConfig
    from ntt.core.data.datamodule import NTTDataModuleConfig

    sel = SelectionConfig(
        selection=[
            {"brainset": "cowley_v4_compact_2024", "sessions": ["PE_201018"]},
            {
                "brainset": "perich_miller_population_2018",
                "sessions": ["c_20131022_center_out_reaching", "x_2"],
            },
        ]
    )
    cfg = NTTDataModuleConfig(
        root="/tmp/data",
        train_selections=[sel],
        val_selections=[sel],
        disable_data_leakage_check=True,
    )
    data_config = cfg.model_dump(mode="json")

    params, missing = _robustness_metadata_params(
        run_metadata={
            "data_config": data_config,
            "eval_config": {"eval_window_seconds": 1.0},
            "tokenizer_config": {"KIND": "identity", "token_bin_size_ms": 20.0},
            "n_sessions": 1,
        },
        per_session_reports=None,
    )

    assert (
        params.get("data.val_brainsets") == "cowley_v4_compact_2024,perich_miller_population_2018"
    )
    assert params.get("data.n_val_sessions") == 3
    # The old never-populated keys must not be advertised (and so never warned about).
    assert "data.selection_name" not in missing
    assert "data.dataset_yaml" not in missing


def test_composed_config_yaml_tolerates_unresolvable_interpolation(caplog):
    """An unresolvable interpolation must warn-and-skip, not crash the eval.

    Provenance capture runs (via ``process_config``) before the job context,
    so an eager ``resolve=True`` that raises on a lazy/missing interpolation
    would abort an otherwise-valid eval before any compute. It must instead
    record ``None`` and warn.
    """
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"a": "${nonexistent_key}", "data": {"x": 1}})
    with caplog.at_level("WARNING"):
        result = composed_config_yaml(cfg)
    assert result is None
    assert any(record.levelname == "WARNING" for record in caplog.records)


def test_config_subtree_tolerates_unresolvable_interpolation(caplog):
    """A subtree with an unresolvable interpolation warns and returns None."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"data": {"x": "${nonexistent_key}"}})
    with caplog.at_level("WARNING"):
        result = config_subtree_to_jsonable(cfg, ["data"])
    assert result is None
    assert any(record.levelname == "WARNING" for record in caplog.records)


def test_m1_flattener_x_axis_matches_each_metrics_strengths(tmp_path):
    """The shared `_x_strength` axis must label every metric's steps correctly,
    even when metrics in the same family cover different strength sets.

    The per-family `_x_strength` must not be emitted from whichever
    group iterates first: a metric missing a strength (e.g. NaN-dropped) would
    otherwise get steps that the single shared axis mislabels.
    """
    jsonl = tmp_path / "results.jsonl"
    # Same "drop" family, two metrics with DIFFERENT strength coverage:
    #   mA: {0.1, 0.5}          (missing 0.25)
    #   mB: {0.1, 0.25, 0.5}
    value_to_strength = {10.0: 0.1, 50.0: 0.5, 1.0: 0.1, 2.0: 0.25, 5.0: 0.5}
    records = [
        ("drop_p01", "mA", 10.0),
        ("drop_p05", "mA", 50.0),
        ("drop_p01", "mB", 1.0),
        ("drop_p025", "mB", 2.0),
        ("drop_p05", "mB", 5.0),
    ]
    _write_jsonl(
        jsonl,
        [
            {
                "session_id": "S1",
                "segment_index": 0,
                "corruption": corruption,
                "metric": metric,
                "sub_label": None,
                "raw_value": value,
                "status": "ok",
            }
            for corruption, metric, value in records
        ],
    )
    label_lookup = {
        "drop_p01": ("drop", 0.1),
        "drop_p025": ("drop", 0.25),
        "drop_p05": ("drop", 0.5),
    }
    series, _ = flatten_input_robustness_jsonl_to_metric_series(
        jsonl, label_lookup, prefix="eval_m1"
    )

    x_map = {step: strength for name, strength, step in series if name.endswith("_x_strength")}
    # One shared axis covering the union of strengths.
    assert sorted(x_map.values()) == [0.1, 0.25, 0.5]

    # Every main-curve point sits at a step the shared axis labels with the
    # strength that actually produced that value.
    for name, value, step in series:
        if name.endswith("_x_strength"):
            continue
        assert x_map[step] == value_to_strength[value], (
            f"{name} value {value} (strength {value_to_strength[value]}) landed at "
            f"step {step} which the x-axis labels {x_map[step]}"
        )


def test_m1_flattener_groups_by_family_and_steps_by_strength(tmp_path):
    """Three drop-fraction levels collapse to one curve, sorted by strength."""
    jsonl = tmp_path / "results.jsonl"
    # Two segments per (corruption, metric) → mean across segments.
    records = [
        # "drop" family, p=0.1
        {
            "session_id": "S1",
            "segment_index": 0,
            "corruption": "drop_p01",
            "metric": "rec_loss",
            "sub_label": None,
            "raw_value": 1.0,
            "status": "ok",
        },
        {
            "session_id": "S1",
            "segment_index": 1,
            "corruption": "drop_p01",
            "metric": "rec_loss",
            "sub_label": None,
            "raw_value": 3.0,
            "status": "ok",
        },
        # "drop" family, p=0.5
        {
            "session_id": "S1",
            "segment_index": 0,
            "corruption": "drop_p05",
            "metric": "rec_loss",
            "sub_label": None,
            "raw_value": 5.0,
            "status": "ok",
        },
        # "drop" family, p=0.25 (out of order on disk, must be sorted on output)
        {
            "session_id": "S1",
            "segment_index": 0,
            "corruption": "drop_p025",
            "metric": "rec_loss",
            "sub_label": None,
            "raw_value": 4.0,
            "status": "ok",
        },
    ]
    _write_jsonl(jsonl, records)

    label_lookup = {
        "drop_p01": ("drop", 0.1),
        "drop_p025": ("drop", 0.25),
        "drop_p05": ("drop", 0.5),
    }
    series, scalars = flatten_input_robustness_jsonl_to_metric_series(
        jsonl, label_lookup, prefix="eval_m1"
    )

    # Expect 3 main curve points + 3 x-axis points = 6 total.
    main_series = [s for s in series if "/_x_" not in s[0]]
    x_series = [s for s in series if "/_x_" in s[0]]

    assert len(main_series) == 3, main_series
    # Sorted by strength: step 0 = 0.1 (mean of 1+3=2), step 1 = 0.25 (4),
    # step 2 = 0.5 (5).
    main_series.sort(key=lambda s: s[2])
    expected_name = "eval_m1/S1/drop/rec_loss"
    assert all(s[0] == expected_name for s in main_series)
    assert main_series[0] == (expected_name, 2.0, 0)
    assert main_series[1] == (expected_name, 4.0, 1)
    assert main_series[2] == (expected_name, 5.0, 2)

    # x-axis series records the strength at each step, named per (session, family).
    assert len(x_series) == 3
    x_series.sort(key=lambda s: s[2])
    assert x_series[0] == ("eval_m1/S1/drop/_x_strength", 0.1, 0)
    assert x_series[1] == ("eval_m1/S1/drop/_x_strength", 0.25, 1)
    assert x_series[2] == ("eval_m1/S1/drop/_x_strength", 0.5, 2)

    assert scalars == {
        "eval_m1/n_records": 4.0,
        "eval_m1/n_skipped": 0.0,
    }


def test_m1_flattener_skips_non_ok_and_unknown_labels(tmp_path):
    """Skipped specs and labels missing from the lookup are excluded."""
    jsonl = tmp_path / "results.jsonl"
    records = [
        {
            "session_id": "S1",
            "segment_index": 0,
            "corruption": "drop_p01",
            "metric": "rec_loss",
            "sub_label": None,
            "raw_value": 1.0,
            "status": "ok",
        },
        # Skipped record (status != ok) — counts toward n_skipped only.
        {"corruption": "drop_p05", "status": "skipped", "reason": "boom"},
        # Unknown label (not in lookup) — silently dropped from curves.
        {
            "session_id": "S1",
            "segment_index": 0,
            "corruption": "wat",
            "metric": "rec_loss",
            "sub_label": None,
            "raw_value": 9.0,
            "status": "ok",
        },
    ]
    _write_jsonl(jsonl, records)

    series, scalars = flatten_input_robustness_jsonl_to_metric_series(
        jsonl, {"drop_p01": ("drop", 0.1)}, prefix="eval_m1"
    )

    # Only the one valid record yields output.
    main_series = [s for s in series if "/_x_" not in s[0]]
    assert main_series == [("eval_m1/S1/drop/rec_loss", 1.0, 0)]
    # The unknown-label "ok" record is counted in n_records but produces no
    # series; the "skipped" record bumps n_skipped.
    assert scalars["eval_m1/n_skipped"] == 1.0
    assert scalars["eval_m1/n_records"] == 2.0


def test_m1_flattener_preserves_sub_label_in_metric_name(tmp_path):
    """sub_label (e.g. mean_joint flavour) becomes a trailing path component."""
    jsonl = tmp_path / "results.jsonl"
    records = [
        {
            "session_id": "S1",
            "segment_index": 0,
            "corruption": "drop_p01",
            "metric": "rec_loss",
            "sub_label": "mean_joint",
            "raw_value": 1.0,
            "status": "ok",
        },
        {
            "session_id": "S1",
            "segment_index": 0,
            "corruption": "drop_p01",
            "metric": "rec_loss",
            "sub_label": "std_joint",
            "raw_value": 0.5,
            "status": "ok",
        },
    ]
    _write_jsonl(jsonl, records)
    series, _ = flatten_input_robustness_jsonl_to_metric_series(
        jsonl, {"drop_p01": ("drop", 0.1)}, prefix="eval_m1"
    )
    names = sorted({s[0] for s in series if "/_x_" not in s[0]})
    assert names == [
        "eval_m1/S1/drop/rec_loss/mean_joint",
        "eval_m1/S1/drop/rec_loss/std_joint",
    ]


def test_m2_flattener_averages_replicas_and_steps_by_epsilon(tmp_path):
    """K replicas at the same epsilon collapse to a per-epsilon mean.

    The curve series uses ``step = int(round(epsilon * 10000))`` so
    consumers reading ``metric_history`` recover the epsilon directly
    from the step value (mlflow only accepts integer steps). No
    parallel ``_x_epsilon`` side-curve is emitted. Per-session
    per-epsilon scalars are logged from aggregated report artifacts
    elsewhere, not by this raw-row flattener.
    """
    jsonl = tmp_path / "results.jsonl"
    records = [
        # eps=0.1 with 3 replicas
        {
            "session_id": "S1",
            "segment_index": 0,
            "epsilon": 0.1,
            "k": 0,
            "metric": "mse",
            "raw_value": 1.0,
            "status": "ok",
        },
        {
            "session_id": "S1",
            "segment_index": 0,
            "epsilon": 0.1,
            "k": 1,
            "metric": "mse",
            "raw_value": 3.0,
            "status": "ok",
        },
        {
            "session_id": "S1",
            "segment_index": 0,
            "epsilon": 0.1,
            "k": 2,
            "metric": "mse",
            "raw_value": 5.0,
            "status": "ok",
        },
        # eps=0.5 with 1 replica
        {
            "session_id": "S1",
            "segment_index": 0,
            "epsilon": 0.5,
            "k": 0,
            "metric": "mse",
            "raw_value": 10.0,
            "status": "ok",
        },
    ]
    _write_jsonl(jsonl, records)

    series, scalars = flatten_latent_robustness_jsonl_to_metric_series(jsonl, prefix="eval_m2")
    main = [s for s in series if "/_x_" not in s[0]]
    x = [s for s in series if "/_x_" in s[0]]

    assert main == [
        ("eval_m2/S1/mse", 3.0, 1000),  # eps=0.1 → step=1000
        ("eval_m2/S1/mse", 10.0, 5000),  # eps=0.5 → step=5000
    ]
    assert x == []
    assert scalars["eval_m2/n_records"] == 4.0
    assert scalars["eval_m2/n_skipped"] == 0.0
    assert set(scalars) == {"eval_m2/n_records", "eval_m2/n_skipped"}


def test_m1_flattener_sanitizes_pipe_and_equals_in_sub_label(tmp_path):
    """input_robustness (M1)'s ``_corruption_label`` joins key=value with ``|`` (e.g.
    ``jitter|k_ms=10.0|p=0.5``); MLflow rejects those chars in metric
    names. The flattener must replace them so logging doesn't crash."""
    jsonl = tmp_path / "results.jsonl"
    records = [
        {
            "session_id": "S1",
            "segment_index": 0,
            "corruption": "jit",
            "metric": "robustness_input",
            "sub_label": "jitter|k_ms=10.0|p=0.5",
            "raw_value": 1.0,
            "status": "ok",
        },
    ]
    _write_jsonl(jsonl, records)
    series, _ = flatten_input_robustness_jsonl_to_metric_series(
        jsonl, {"jit": ("jitter", 10.0)}, prefix="eval_m1"
    )
    main = [s for s in series if "/_x_" not in s[0]]
    assert main, "no main series produced"
    name = main[0][0]
    # No illegal chars remain.
    assert "|" not in name and "=" not in name, name
    # Substituted with ``-``; no embedded `=`/`|` mean the suffix is reachable.
    assert name == "eval_m1/S1/jitter/robustness_input/jitter-k_ms-10.0-p-0.5"


def test_extract_input_robustness_jsonl_summary_metrics_emits_core_avg_scalars(tmp_path):
    """Derived core_avg rows should become compact overall + session metrics."""
    jsonl = tmp_path / "results.jsonl"
    records = [
        {
            "session_id": "brainset/A",
            "segment_index": 0,
            "corruption": "aggregate_non_population",
            "metric": "robustness_input_core_avg",
            "raw_value": 1.0,
            "status": "ok",
        },
        {
            "session_id": "brainset/A",
            "segment_index": 1,
            "corruption": "aggregate_non_population",
            "metric": "robustness_input_core_avg",
            "raw_value": 3.0,
            "status": "ok",
        },
        {
            "session_id": "brainset/B",
            "segment_index": 0,
            "corruption": "aggregate_non_population",
            "metric": "robustness_input_core_avg",
            "raw_value": 5.0,
            "status": "ok",
        },
    ]
    _write_jsonl(jsonl, records)

    metrics = extract_input_robustness_jsonl_summary_metrics(jsonl, prefix="eval_m1")

    assert metrics == {
        "eval_m1/joint/core_avg": 3.0,
        "eval_m1/session/brainset__A/joint/core_avg": 2.0,
        "eval_m1/session/brainset__B/joint/core_avg": 5.0,
    }


def test_remap_input_robustness_metric_keys_for_mlflow_shortens_layout():
    """Report-based input_robustness (M1) metrics should move under joint/per_dim/session
    namespaces."""
    metrics = {
        "eval_m1/n_skipped": 2.0,
        "eval_m1/robustness_input/jitter-k_ms-1.0-p-0.2": 0.4,
        "eval_m1/robustness_input_per_dim/jitter-k_ms-1.0-p-0.2": 0.6,
        "eval_m1/per_session/S1/n_skipped": 1.0,
        "eval_m1/per_session/S1/robustness_input/window_shift-max_shift_ms-3.0": 0.7,
        "eval_m1/per_session/S1/robustness_input_per_dim/window_shift-max_shift_ms-3.0": 0.8,
    }

    remapped = remap_input_robustness_metric_keys_for_mlflow(metrics, prefix="eval_m1")

    assert remapped == {
        "eval_m1/n_skipped": 2.0,
        "eval_m1/joint/jitter-k_ms-1.0-p-0.2": 0.4,
        "eval_m1/per_dim/jitter-k_ms-1.0-p-0.2": 0.6,
        "eval_m1/session/S1/n_skipped": 1.0,
        "eval_m1/session/S1/joint/window_shift-max_shift_ms-3.0": 0.7,
        "eval_m1/session/S1/per_dim/window_shift-max_shift_ms-3.0": 0.8,
    }


def test_m2_flattener_drops_nonfinite_values(tmp_path):
    """NaN/Inf values are dropped (MLflow rejects them)."""
    jsonl = tmp_path / "results.jsonl"
    records = [
        {
            "session_id": "S1",
            "segment_index": 0,
            "epsilon": 0.1,
            "k": 0,
            "metric": "mse",
            "raw_value": float("nan"),
            "status": "ok",
        },
        {
            "session_id": "S1",
            "segment_index": 0,
            "epsilon": 0.1,
            "k": 1,
            "metric": "mse",
            "raw_value": 2.0,
            "status": "ok",
        },
    ]
    _write_jsonl(jsonl, records)
    series, _ = flatten_latent_robustness_jsonl_to_metric_series(jsonl, prefix="eval_m2")
    main = [s for s in series if "/_x_" not in s[0]]
    # Only the finite value remains; the NaN was silently dropped.
    # eps=0.1 → step = int(round(0.1 * 10000)) = 1000.
    assert main == [("eval_m2/S1/mse", 2.0, 1000)]
