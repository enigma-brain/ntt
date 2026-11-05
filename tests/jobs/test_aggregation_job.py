"""Tests for jobs/aggregation.py."""

from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import pytest
import yaml

from ntt.evaluation.aggregation.comparability import COMPARABILITY_VERSION
from ntt.jobs.aggregation import AGGREGATION_SCHEMA_VERSION, NTTAggregationJob, _load_json

DOWNSTREAM_RECORDS = Path(__file__).parents[1] / "data" / "downstream_records"


def test_load_json_reads_local_file(tmp_path):
    data = {"key": "value", "number": 42}
    f = tmp_path / "data.json"
    f.write_text(json.dumps(data))
    result = _load_json(str(f))
    assert result == data


def test_load_json_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        _load_json(str(tmp_path / "nonexistent.json"))


def test_load_json_handles_mlflow_uri(tmp_path):
    data = {"mlflow": True}
    local_file = tmp_path / "downloaded.json"
    local_file.write_text(json.dumps(data))

    with patch("mlflow.artifacts.download_artifacts", return_value=str(local_file)) as mock_dl:
        result = _load_json("mlflow-artifacts:/474/run/artifacts/result.json")

    mock_dl.assert_called_once_with(
        "mlflow-artifacts:/474/run/artifacts/result.json", dst_path=ANY
    )
    assert result == data


def test_aggregation_job_fields_default_to_none():
    job = NTTAggregationJob(output_path="/tmp/out")
    assert job.recon_path is None
    assert job.m1_path is None
    assert job.m2_path is None
    assert job.encoding_path is None
    assert job.decoding_path is None
    assert job.poisson_reference is None
    assert job.force_recompute_poisson is False
    assert job.score_clip == pytest.approx(-1.0)
    assert job.compression_metric == "compression_rate"
    assert job.include_m2 is True  # M2 ON by default (team decision; ε=0.1)
    assert job.m2_metric == "acg_correlation"
    assert job.m2_epsilon == pytest.approx(0.1)
    assert job.ckpt_uri is None
    assert job.mlflow_logging.enabled is False


def _write_recon(tmp_path: Path, filename: str = "results.json") -> Path:
    data = {
        "scored_window_seconds": 1.0,
        "comparison_vs_poisson": {
            "spike_f1": {
                "name": "spike_f1",
                "category": "temporal_fidelity",
                "direction": "higher",
                "level": "population",
                "model_value": 0.74,
                "baseline_value": 0.0,
                "ceiling_value": 1.0,
                "normalized_score": 0.8,
                "value_range": [None, None],
                "status": "ok",
                "reason": None,
            }
        },
        "generalization": {
            "summary": {
                "generalization_score": 0.87,
                "baseline_mode": "brainset_matched",
                "score_tiers": ["xsource"],
            }
        },
        "compression": {"compression_rate": 8.0},
    }
    f = tmp_path / filename
    f.write_text(json.dumps(data))
    return f


def test_run_writes_category_scores_json(tmp_path):
    recon_file = _write_recon(tmp_path)
    job = NTTAggregationJob(
        recon_path=str(recon_file),
        output_path=str(tmp_path / "output"),
        ckpt_uri="mlflow-artifacts:/test/last.ckpt",
    )
    with job:
        result = job.run()

    out_file = tmp_path / "output" / "category_scores.json"
    assert out_file.exists()
    data = json.loads(out_file.read_text())

    assert data["scores"]["single_neuron_recon"] == pytest.approx(0.8)
    assert data["scores"]["generalization"] == pytest.approx(0.87)
    assert data["scores"]["compression"] == pytest.approx(8.0)
    assert data["scores"]["latent_structure_and_robustness"] is None
    assert data["scores"]["downstream_encoding"] is None

    assert data["metadata"]["schema_version"] == AGGREGATION_SCHEMA_VERSION
    assert data["metadata"]["ckpt_uri"] == "mlflow-artifacts:/test/last.ckpt"
    assert data["metadata"]["inputs"]["recon"] == str(recon_file)
    assert data["metadata"]["inputs"]["m1"] is None

    assert "single_neuron_recon" in data["diagnostics"]
    assert data["diagnostics"]["generalization"] is None  # scalar pass-through

    assert result["scores"]["single_neuron_recon"] == pytest.approx(0.8)


def test_run_all_nulls_with_no_inputs(tmp_path):
    job = NTTAggregationJob(output_path=str(tmp_path / "output"))
    with job:
        job.run()

    data = json.loads((tmp_path / "output" / "category_scores.json").read_text())
    for col_val in data["scores"].values():
        assert col_val is None


def test_run_creates_output_directory(tmp_path):
    recon_file = _write_recon(tmp_path)
    deep_out = tmp_path / "a" / "b" / "c"
    job = NTTAggregationJob(recon_path=str(recon_file), output_path=str(deep_out))
    with job:
        job.run()
    assert (deep_out / "category_scores.json").exists()


def test_run_logs_to_mlflow_when_enabled(tmp_path, mlflow_store):
    from mlflow.tracking import MlflowClient

    from ntt.evaluation.mlflow_logging import MLflowLoggingConfig

    recon_file = _write_recon(tmp_path)
    job = NTTAggregationJob(
        recon_path=str(recon_file),
        output_path=str(tmp_path / "output"),
        ckpt_uri="mlflow-artifacts:/test/last.ckpt",
        mlflow_logging=MLflowLoggingConfig(
            enabled=True, experiment_name="job-logs", run_name="tok_job"
        ),
    )
    with job:
        job.run()

    client = MlflowClient()
    experiment = client.get_experiment_by_name("job-logs")
    assert experiment is not None
    (run,) = client.search_runs([experiment.experiment_id])
    metrics = run.data.metrics
    assert metrics["single_neuron_recon"] == pytest.approx(0.8)
    assert "generalization" in metrics
    assert math.isnan(metrics["latent_structure_and_robustness"])  # None → nan
    assert run.info.run_name == "tok_job"
    assert run.data.tags["ntt.agg.ckpt_uri"] == "mlflow-artifacts:/test/last.ckpt"
    assert [a.path for a in client.list_artifacts(run.info.run_id)] == ["category_scores.json"]


def test_mlflow_logging_without_a_tracking_server_fails_before_scoring(tmp_path, monkeypatch):
    # MLflow would otherwise log to ./mlruns, a run no tracker can link to.
    from ntt.evaluation.mlflow_logging import MLflowLoggingConfig

    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    job = NTTAggregationJob(
        recon_path=str(_write_recon(tmp_path)),
        output_path=str(tmp_path / "output"),
        mlflow_logging=MLflowLoggingConfig(enabled=True),
    )
    with job, pytest.raises(ValueError, match="MLFLOW_TRACKING_URI"):
        job.run()
    assert not (tmp_path / "output").exists()


def test_job_yaml_and_field_defaults_agree():
    """config/job/aggregation.yaml restates the job's defaults; they must not drift."""
    yaml_path = Path(__file__).parents[2] / "config" / "job" / "aggregation.yaml"
    cfg = yaml.safe_load(yaml_path.read_text())
    cfg.pop("_target_")
    cfg.pop("output_path")  # required: "???"
    yaml_logging = cfg.pop("mlflow_logging")
    fields = NTTAggregationJob.model_fields
    for key, value in cfg.items():
        assert fields[key].default == value, key
    default_logging = fields["mlflow_logging"].get_default(call_default_factory=True)
    assert {key: getattr(default_logging, key) for key in yaml_logging} == yaml_logging


def test_aggregation_job_end_to_end_with_recon_and_m1(tmp_path):
    recon = {
        "scored_window_seconds": 1.0,
        "comparison_vs_poisson": {
            "spike_f1": {
                "name": "spike_f1",
                "category": "temporal_fidelity",
                "direction": "higher",
                "level": "population",
                "model_value": 0.74,
                "baseline_value": 0.0,
                "ceiling_value": 1.0,
                "normalized_score": 0.8,
                "value_range": [None, None],
                "status": "ok",
                "reason": None,
            },
            "participation_ratio": {
                "name": "participation_ratio",
                "category": "population_structure",
                "direction": "higher",
                "level": "population",
                "model_value": 0.5,
                "baseline_value": 0.0,
                "ceiling_value": 1.0,
                "normalized_score": 0.6,
                "value_range": [None, None],
                "status": "ok",
                "reason": None,
            },
        },
        "generalization": {
            "summary": {
                "generalization_score": 0.87,
                "baseline_mode": "brainset_matched",
                "score_tiers": ["xsource"],
            }
        },
        "compression": {"compression_rate": 8.0},
    }
    # De-normalized latent bucket: M1 arrives as a raw robustness eval report; the
    # bucket maps the canonical-aggregate NMSE d onto retention = 1/(1+d).
    m1 = {
        "results": {
            "robustness_input/window_shift|max_shift_ms=3.0": {"raw_value": 0.5},
            "robustness_input/jitter|k_ms=1.0|p=0.2": {"raw_value": 0.5},
            "robustness_input/drop_and_add|fraction=0.02": {"raw_value": 0.5},
        }
    }

    recon_file = tmp_path / "results.json"
    m1_file = tmp_path / "eval_report.json"
    recon_file.write_text(json.dumps(recon))
    m1_file.write_text(json.dumps(m1))

    job = NTTAggregationJob(
        recon_path=str(recon_file),
        m1_path=str(m1_file),
        output_path=str(tmp_path / "agg_out"),
        score_clip=-1.0,
    )
    with job:
        result = job.run()

    data = json.loads(Path(result["output_path"]).read_text())

    assert data["scores"]["single_neuron_recon"] == pytest.approx(0.8)
    assert data["scores"]["population_structure"] == pytest.approx(0.6)
    # d = mean(0.5, 0.5, 0.5) = 0.5 -> retention = 1/1.5
    assert data["scores"]["latent_structure_and_robustness"] == pytest.approx(1 / 1.5)
    assert data["scores"]["generalization"] == pytest.approx(0.87)
    assert data["scores"]["compression"] == pytest.approx(8.0)
    assert data["scores"]["downstream_encoding"] is None
    assert data["scores"]["decoding"] is None

    assert data["diagnostics"]["single_neuron_recon"]["n_metrics"] == 1
    assert data["diagnostics"]["latent_structure_and_robustness"]["n_metrics"] == 1

    assert data["metadata"]["schema_version"] == AGGREGATION_SCHEMA_VERSION
    assert data["metadata"]["robustness_normalization"]["m1"] == "components_mean"
    # M2 is ON by default but no m2_path here -> excluded; bucket = M1 retention only.
    assert data["metadata"]["include_m2"] is True
    assert data["metadata"]["robustness_normalization"]["m2"] == "excluded_no_report"
    assert data["metadata"]["inputs"]["recon"] == str(recon_file)
    assert data["metadata"]["inputs"]["m1"] == str(m1_file)


def test_run_includes_m2_by_default(tmp_path):
    # include_m2 defaults to True: a valid M2 report contributes to the bucket.
    recon_file = _write_recon(tmp_path)  # spike_f1 only; no structure-preservation metric
    m1 = {
        "results": {
            "robustness_input/window_shift|max_shift_ms=3.0": {"raw_value": 0.0},
            "robustness_input/jitter|k_ms=1.0|p=0.2": {"raw_value": 0.0},
            "robustness_input/drop_and_add|fraction=0.02": {"raw_value": 0.0},
        }
    }
    m2 = {
        "config": {"scored_window_seconds": 1.0},
        "results": {
            "acg_correlation/eps_0": {"raw_value": 0.9, "metadata": {"epsilon": 0.0}},
            "acg_correlation/eps_0.1": {"raw_value": 0.5, "metadata": {"epsilon": 0.1}},
        },
    }
    m1_file = tmp_path / "m1.json"
    m1_file.write_text(json.dumps(m1))
    m2_file = tmp_path / "m2.json"
    m2_file.write_text(json.dumps(m2))

    job = NTTAggregationJob(
        recon_path=str(recon_file),
        m1_path=str(m1_file),
        m2_path=str(m2_file),
        output_path=str(tmp_path / "out"),
    )
    with job:
        result = job.run()

    data = json.loads(Path(result["output_path"]).read_text())
    diag = data["diagnostics"]["latent_structure_and_robustness"]
    # M1 retention (d=0 -> 1.0) + M2 max(0.5, 0) = 0.5 -> mean 0.75
    assert data["scores"]["latent_structure_and_robustness"] == pytest.approx((1.0 + 0.5) / 2)
    assert data["metadata"]["robustness_normalization"]["m2"] == "included"
    assert any(k.startswith("m2:") for k in diag["per_metric"])


def test_run_tolerates_missing_m1_m2_artifact(tmp_path, caplog):
    """A bad m1_path/m2_path (missing eval_report.json) must not crash the job.

    The load warns and yields None; the resolver then excludes M1/M2 from the
    latent_structure_and_robustness column, leaving the rest of the run intact.
    """
    import logging

    recon_file = _write_recon(tmp_path)
    job = NTTAggregationJob(
        recon_path=str(recon_file),
        m1_path=str(tmp_path / "does_not_exist_m1.json"),
        m2_path=str(tmp_path / "does_not_exist_m2.json"),
        output_path=str(tmp_path / "out"),
        score_clip=-1.0,
    )
    with caplog.at_level(logging.WARNING), job:
        result = job.run()

    data = json.loads(Path(result["output_path"]).read_text())
    # Recon-derived columns still computed; robustness column is None (M1+M2 excluded).
    assert data["scores"]["single_neuron_recon"] == pytest.approx(0.8)
    assert data["scores"]["latent_structure_and_robustness"] is None
    assert any("robustness report unavailable" in r.message for r in caplog.records)


def _write_decoding_dir(tmp_path: Path, sessions: dict[str, float]) -> Path:
    """A decoding benchmark parent's artifact dir whose cohort record scores ``sessions``
    (POYO test R² per session)."""
    parent = _decoding_parent(tmp_path, "decoding_cohort")
    path = parent / "downstream_decoding_results.json"
    record = json.loads(path.read_text())
    mean = sum(sessions.values()) / len(sessions)
    record["metrics"]["test"] = {
        "aggregated": {
            "r2": {"value": mean, "n": len(sessions), "aggregated_over": "sessions:mean"}
        },
        "per_session": {sid: {"r2": r2} for sid, r2 in sessions.items()},
    }
    path.write_text(json.dumps(record))
    return parent


def _write_encoding(tmp_path: Path, nc_r2_mean: float) -> Path:
    """A schema-v2 encoding run record whose test ``nc_r2`` is ``nc_r2_mean``."""
    record = json.loads((DOWNSTREAM_RECORDS / "encoding_run.json").read_text())
    record["metrics"]["test"]["aggregated"]["nc_r2"]["value"] = nc_r2_mean
    path = tmp_path / "downstream_encoding_results.json"
    path.write_text(json.dumps(record))
    return path


def test_run_with_all_four_inputs(tmp_path):
    recon_file = _write_recon(tmp_path)
    # De-normalized latent bucket: raw M1 robustness report; d=0.5 -> retention 1/1.5.
    m1 = {
        "results": {
            "robustness_input/window_shift|max_shift_ms=3.0": {"raw_value": 0.5},
            "robustness_input/jitter|k_ms=1.0|p=0.2": {"raw_value": 0.5},
            "robustness_input/drop_and_add|fraction=0.02": {"raw_value": 0.5},
        }
    }
    m1_file = tmp_path / "eval_report.json"
    m1_file.write_text(json.dumps(m1))

    decoding_dir = _decoding_parent(tmp_path, "decoding_cohort")
    encoding_file = DOWNSTREAM_RECORDS / "encoding_run.json"

    job = NTTAggregationJob(
        recon_path=str(recon_file),
        m1_path=str(m1_file),
        decoding_path=str(decoding_dir),
        encoding_path=str(encoding_file),
        output_path=str(tmp_path / "all_out"),
        score_clip=-1.0,
    )
    with job:
        result = job.run()

    data = json.loads(Path(result["output_path"]).read_text())
    assert data["scores"]["single_neuron_recon"] == pytest.approx(0.8)
    assert data["scores"]["latent_structure_and_robustness"] == pytest.approx(1 / 1.5)
    assert data["scores"]["downstream_encoding"] == pytest.approx(0.19)
    assert data["scores"]["decoding"] == pytest.approx(0.83)
    # The breakdown sits next to the composite: POYO only here, no ridge targets.
    assert data["scores"]["decoding_poyo"] == pytest.approx(0.83)
    assert data["scores"]["decoding_linear"] is None
    assert data["diagnostics"]["downstream_encoding"] is None  # scalar pass-through
    # decoding now carries a per-metric diagnostic (POYO sessions); not None.
    assert data["diagnostics"]["decoding"]["n_metrics"] == 2
    assert data["metadata"]["downstream"]["encoding"]["split"] == "test"
    assert data["metadata"]["downstream"]["decoding"]["split"] == "test"


def _decoding_parent(tmp_path: Path, fixture: str) -> Path:
    """A decoding benchmark parent's artifact dir: the cohort record at its root."""
    parent = tmp_path / "decoding_parent"
    parent.mkdir()
    (parent / "downstream_decoding_results.json").write_text(
        (DOWNSTREAM_RECORDS / f"{fixture}.json").read_text()
    )
    return parent


def test_run_scores_downstream_records_and_records_their_provenance(tmp_path):
    job = NTTAggregationJob(
        encoding_path=str(DOWNSTREAM_RECORDS / "encoding_run.json"),
        decoding_path=str(_decoding_parent(tmp_path, "decoding_cohort")),
        output_path=str(tmp_path / "out"),
    )
    with job:
        result = job.run()
    data = json.loads(Path(result["output_path"]).read_text())

    assert data["scores"]["downstream_encoding"] == pytest.approx(0.19)  # test, not val_final
    assert data["scores"]["decoding"] == pytest.approx(0.83)
    downstream = data["metadata"]["downstream"]
    assert downstream["encoding"]["split"] == "test"
    assert downstream["encoding"]["ntt_version"] == "0.0.0-fixture"
    assert downstream["encoding"]["tokenizer_checkpoint"] == (
        "mlflow-artifacts:/1/abc/artifacts/model.ckpt"
    )
    assert downstream["decoding"]["complete"] is True
    assert downstream["decoding"]["mixed_provenance"] is False
    sessions = data["metadata"]["input_sessions"]
    assert sessions["encoding"]["session_ids"] == ["PE_201021"]
    assert sessions["decoding"]["n_sessions"] == 2


def test_incomplete_decoding_cohort_scores_none_unless_allowed(tmp_path):
    parent = _decoding_parent(tmp_path, "decoding_cohort_incomplete")

    def run(allow: bool) -> dict:
        job = NTTAggregationJob(
            decoding_path=str(parent),
            allow_incomplete_decoding_cohort=allow,
            output_path=str(tmp_path / f"out_{allow}"),
        )
        with job:
            return json.loads(Path(job.run()["output_path"]).read_text())

    refused = run(False)
    assert refused["scores"]["decoding"] is None
    assert refused["metadata"]["downstream"]["decoding"]["missing"] == [
        "j_20160405_center_out_reaching"
    ]
    allowed = run(True)
    assert allowed["scores"]["decoding"] == pytest.approx(0.83)
    assert allowed["metadata"]["downstream"]["decoding"]["complete"] is False


def test_failed_final_eval_gives_no_encoding_score(tmp_path):
    job = NTTAggregationJob(
        encoding_path=str(DOWNSTREAM_RECORDS / "encoding_run_final_eval_failed.json"),
        output_path=str(tmp_path / "out"),
    )
    with job:
        data = json.loads(Path(job.run()["output_path"]).read_text())
    assert data["scores"]["downstream_encoding"] is None
    assert data["metadata"]["downstream"]["encoding"]["final_eval"]["status"] == "failed"


def _write_selection_yamls(config_root: Path) -> None:
    """Hermetic tier YAMLs: bsA/sess_ood is xsource (the scored tier); other bsA sessions
    are xtime."""
    root = config_root / "data" / "selections" / "composite"
    root.mkdir(parents=True, exist_ok=True)
    (root / "ntt_full_val_xsession.yaml").write_text(yaml.safe_dump({"selection": []}))
    (root / "ntt_full_val_xsubject.yaml").write_text(yaml.safe_dump({"selection": []}))
    (root / "ntt_full_val_xsource.yaml").write_text(
        yaml.safe_dump({"selection": [{"brainset": "bsA", "sessions": ["sess_ood"]}]})
    )
    (root / "ntt_full_val_xtime.yaml").write_text(
        yaml.safe_dump({"selection": [{"brainset": "bsA", "exclude_sessions": ["sess_ood"]}]})
    )


def _recon_dict(*, with_generalization: bool, per_session_brainset: str = "bsA") -> dict:
    """A recon report with comparison_vs_poisson (required by the scorer) + per_session."""
    data = {
        "scored_window_seconds": 1.0,
        "comparison_vs_poisson": {
            "spike_f1": {
                "name": "spike_f1",
                "category": "temporal_fidelity",
                "direction": "higher",
                "level": "population",
                "model_value": 0.74,
                "baseline_value": 0.0,
                "ceiling_value": 1.0,
                "normalized_score": 0.8,
                "value_range": [None, None],
                "status": "ok",
                "reason": None,
            }
        },
        "compression": {"compression_rate": 8.0},
        "per_session": {
            f"{per_session_brainset}/sess_id": {
                "results": {
                    "acc": {
                        "raw_value": 0.8,
                        "direction": "higher",
                        "category": "single_unit_statistics",
                    }
                }
            },
            f"{per_session_brainset}/sess_ood": {
                "results": {
                    "acc": {
                        "raw_value": 0.6,
                        "direction": "higher",
                        "category": "single_unit_statistics",
                    }
                }
            },
        },
    }
    if with_generalization:
        data["generalization"] = {
            "summary": {
                "generalization_score": 0.87,
                "baseline_mode": "brainset_matched",
                "score_tiers": ["xsource"],
            }
        }
    return data


def _write_recon_json(tmp_path: Path, data: dict) -> Path:
    f = tmp_path / "results.json"
    f.write_text(json.dumps(data))
    return f


def test_generalization_backfilled_when_block_absent(tmp_path):
    config_root = tmp_path / "config"
    _write_selection_yamls(config_root)
    recon_file = _write_recon_json(tmp_path, _recon_dict(with_generalization=False))
    job = NTTAggregationJob(
        recon_path=str(recon_file),
        output_path=str(tmp_path / "out"),
        gen_config_root=str(config_root),
    )
    with job:
        job.run()
    data = json.loads((tmp_path / "out" / "category_scores.json").read_text())
    # retention(higher) = 0.6/0.8 = 0.75
    assert data["scores"]["generalization"] == pytest.approx(0.75)
    assert data["metadata"]["generalization_source"] == "recomputed_from_per_session(val)"


def test_generalization_precomputed_block_not_overwritten(tmp_path):
    config_root = tmp_path / "config"
    _write_selection_yamls(config_root)
    recon_file = _write_recon_json(tmp_path, _recon_dict(with_generalization=True))
    job = NTTAggregationJob(
        recon_path=str(recon_file),
        output_path=str(tmp_path / "out"),
        gen_config_root=str(config_root),
    )
    with job:
        job.run()
    data = json.loads((tmp_path / "out" / "category_scores.json").read_text())
    assert data["scores"]["generalization"] == pytest.approx(0.87)
    assert data["metadata"]["generalization_source"] == "precomputed"


@pytest.mark.parametrize(
    "stale_summary",
    [
        {"generalization_score": 0.87},  # pooled tier-mean baseline, every category
        {"generalization_score": 0.87, "baseline_mode": "brainset_matched"},  # every OOD tier
    ],
)
def test_generalization_block_without_score_tiers_is_rescored(tmp_path, stale_summary):
    # A block without score_tiers was scored by another formula and is not reused.
    config_root = tmp_path / "config"
    _write_selection_yamls(config_root)
    data = _recon_dict(with_generalization=False)
    data["generalization"] = {"summary": stale_summary}
    recon_file = _write_recon_json(tmp_path, data)
    job = NTTAggregationJob(
        recon_path=str(recon_file),
        output_path=str(tmp_path / "out"),
        gen_config_root=str(config_root),
    )
    with job:
        job.run()
    out = json.loads((tmp_path / "out" / "category_scores.json").read_text())
    assert out["scores"]["generalization"] == pytest.approx(0.75)
    assert out["metadata"]["generalization_source"] == "recomputed_from_per_session(val)"


def test_generalization_fallback_disabled_stays_none(tmp_path):
    config_root = tmp_path / "config"
    _write_selection_yamls(config_root)
    recon_file = _write_recon_json(tmp_path, _recon_dict(with_generalization=False))
    job = NTTAggregationJob(
        recon_path=str(recon_file),
        output_path=str(tmp_path / "out"),
        gen_config_root=str(config_root),
        generalization_fallback=False,
    )
    with job:
        job.run()
    data = json.loads((tmp_path / "out" / "category_scores.json").read_text())
    assert data["scores"]["generalization"] is None
    assert data["metadata"]["generalization_source"] == "absent"


def test_generalization_fallback_unmatched_sessions_warns_and_stays_none(tmp_path, caplog):
    import logging

    config_root = tmp_path / "config"
    _write_selection_yamls(config_root)
    # per_session brainset "bsZ" matches no tier YAML -> all unmatched.
    recon_file = _write_recon_json(
        tmp_path, _recon_dict(with_generalization=False, per_session_brainset="bsZ")
    )
    job = NTTAggregationJob(
        recon_path=str(recon_file),
        output_path=str(tmp_path / "out"),
        gen_config_root=str(config_root),
    )
    with caplog.at_level(logging.WARNING), job:
        job.run()
    data = json.loads((tmp_path / "out" / "category_scores.json").read_text())
    assert data["scores"]["generalization"] is None
    assert data["metadata"]["generalization_source"] == "absent"
    assert any("generalization fallback skipped" in r.message for r in caplog.records)


def test_hydra_config_instantiates_main_settings(tmp_path):
    """config/job/aggregation.yaml must survive full MainSettings validation.

    Composing with `--cfg job` only prints the resolved config; it never builds
    the pydantic model, which is how two hard failures shipped unnoticed.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.hydra_config import HydraConfig
    from hydra.utils import instantiate
    from omegaconf import open_dict

    from ntt.jobs.aggregation import NTTAggregationJob

    config_dir = Path(__file__).resolve().parents[2] / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="main",
            overrides=[
                "job=aggregation",
                f"job.output_path={tmp_path / 'out'}",
                "paths.run_name=test_aggregation_config",
            ],
            # `paths.output_dir`/`work_dir` interpolate `${hydra:...}`, which only
            # resolves once HydraConfig is populated. `@hydra.main` does this for
            # us at runtime; under bare `compose` the test must do it itself.
            return_hydra_config=True,
        )
        HydraConfig.instance().set_config(cfg)
    try:
        with open_dict(cfg):
            cfg.pop("hydra")
        settings = instantiate(cfg)
    finally:
        HydraConfig.instance().cfg = None
    assert isinstance(settings.job, NTTAggregationJob)
    assert settings.job.common_parts_only is False


def test_column_provenance_is_serialized_into_the_artifact(tmp_path):
    """Every non-null column's `provenance` must reach `category_scores.json`.

    1f4c0014 records each decoding family mean in the column's `provenance` so the next
    reader need not re-derive the poyo/ridge split, but `_diag_to_dict` dropped the field
    entirely: only the latent bucket's provenance survived, via the separate
    `metadata.robustness_normalization` hop, and docs/eval_aggregation.md documented a
    field that was never written.
    """
    recon = {
        "scored_window_seconds": 1.0,
        "comparison_vs_poisson": {
            "spike_f1": {
                "name": "spike_f1",
                "category": "temporal_fidelity",
                "direction": "higher",
                "level": "population",
                "model_value": 0.74,
                "baseline_value": 0.0,
                "ceiling_value": 1.0,
                "normalized_score": 0.8,
                "value_range": [None, None],
                "status": "ok",
                "reason": None,
            }
        },
        # Real recon reports wrap the raw metric rows in `aggregated.results`, keyed by the
        # per-target display name while `name` stays the bare decoder.
        "aggregated": {
            "results": {
                "behavioral_regression_cursor_pos": {
                    "metadata": {"chunk_window_ms": 200},
                    "name": "behavioral_regression",
                    "raw_value": 0.2,
                    "direction": "higher",
                },
                "behavioral_regression_cursor_vel": {
                    "name": "behavioral_regression",
                    "raw_value": 0.3,
                    "direction": "higher",
                },
                # accuracy, not R² — never part of the ridge family
                "behavioral_classification_movement_phase": {
                    "name": "behavioral_classification",
                    "raw_value": 0.72,
                    "direction": "higher",
                },
            }
        },
    }
    # The ridge scorer recalculates each target using the per-session values.
    recon |= {"aggregation_strategy": "mean", "per_session": {"s1": recon["aggregated"]}}
    m1 = {
        "results": {
            "robustness_input/window_shift|max_shift_ms=3.0": {"raw_value": 0.5},
            "robustness_input/jitter|k_ms=1.0|p=0.2": {"raw_value": 0.5},
            "robustness_input/drop_and_add|fraction=0.02": {"raw_value": 0.5},
        }
    }
    recon_file = tmp_path / "results.json"
    recon_file.write_text(json.dumps(recon))
    m1_file = tmp_path / "eval_report.json"
    m1_file.write_text(json.dumps(m1))
    decoding_dir = _write_decoding_dir(tmp_path, {"session_a": 0.82, "session_b": 0.76})

    job = NTTAggregationJob(
        recon_path=str(recon_file),
        m1_path=str(m1_file),
        decoding_path=str(decoding_dir),
        output_path=str(tmp_path / "prov_out"),
    )
    with job:
        result = job.run()
    data = json.loads(Path(result["output_path"]).read_text())

    # poyo family = mean(0.82, 0.76) = 0.79 over 2 sessions;
    # ridge family = mean(0.2, 0.3) = 0.25 over 2 targets (classification excluded).
    assert data["diagnostics"]["decoding"]["provenance"] == {
        "poyo": "0.79 (2 sessions)",
        "ridge": "0.25 (2 targets)",
    }
    assert data["scores"]["decoding"] == pytest.approx((0.79 + 0.25) / 2)

    # The latent bucket's provenance now rides in diagnostics too, not only in the
    # separate metadata.robustness_normalization hop.
    lsr = data["diagnostics"]["latent_structure_and_robustness"]
    assert lsr["provenance"] == {"m1": "components_mean", "m2": "excluded_no_report"}
    assert lsr["provenance"] == data["metadata"]["robustness_normalization"]

    # Every non-null column carries the field — {} where the scorer records none.
    for col, diag in data["diagnostics"].items():
        if diag is not None:
            assert isinstance(diag["provenance"], dict), col
    assert data["diagnostics"]["single_neuron_recon"]["provenance"] == {}


def test_run_writes_session_and_settings_fields_for_the_comparability_check(tmp_path):
    """The cross-tokenizer check can only compare what the job writes down."""
    recon = json.loads(_write_recon(tmp_path).read_text())
    recon |= {
        "per_session": {
            "s1": {
                "results": {
                    "behavioral_regression_cursor_pos": {
                        "name": "behavioral_regression",
                        "raw_value": 0.2,
                        "metadata": {"chunk_window_ms": 200},
                    }
                }
            },
            "s2": {
                "results": {
                    "behavioral_regression_cursor_pos": {
                        "name": "behavioral_regression",
                        "raw_value": 0.3,
                        "metadata": {"primary_chunk_window_ms": 1000},
                    }
                }
            },
        },
        "aggregation_strategy": "mean",
        "split": "val",
        "tokenizer_config": {"bin_size_ms": 1.0, "window_length_seconds": 1.0, "KIND": "x"},
        "aggregated": {
            "results": {},
            "config": {
                "max_eval_seconds": 600.0,
                "metrics": {
                    "behavioral_regression_cursor_pos": {
                        "metric_class": "behavioral_regression",
                        "chunk_window_ms": [200, 1000],
                    }
                },
            },
        },
    }
    recon_file = tmp_path / "results.json"
    recon_file.write_text(json.dumps(recon))
    # M1 eval_report.json saves the run settings in its config but no sessions; the run's
    # results_per_session.csv next to it lists them.
    m1_dir = tmp_path / "m1_run"
    m1_dir.mkdir()
    m1_config = {"n_replicas": 10, "corruptions": ["jitter"], "scored_window_seconds": 1.0}
    (m1_dir / "eval_report.json").write_text(json.dumps({"results": {}, "config": m1_config}))
    (m1_dir / "results_per_session.csv").write_text(
        "session_id,corruption,metric\ns1,a,b\ns1,c,d\ns2,a,b\n"
    )

    job = NTTAggregationJob(
        recon_path=str(recon_file),
        m1_path=str(m1_dir / "eval_report.json"),
        encoding_path=str(_write_encoding(tmp_path, nc_r2_mean=0.3)),
        output_path=str(tmp_path / "out"),
    )
    with job:
        result = job.run()
    meta = json.loads(Path(result["output_path"]).read_text())["metadata"]

    assert meta["recon_chunk_window_ms"] == {
        "s1/behavioral_regression_cursor_pos": [200.0],
        "s2/behavioral_regression_cursor_pos": [1000.0],
    }
    assert meta["comparability_version"] == COMPARABILITY_VERSION
    assert meta["input_sessions"]["recon"] == {
        "session_ids": ["s1", "s2"],
        "n_sessions": 2,
        "aggregation_strategy": "mean",
    }
    assert meta["input_sessions"]["m1"]["session_ids"] == ["s1", "s2"]
    assert meta["input_sessions"]["encoding"]["n_sessions"] == 1
    assert meta["input_sessions"]["m2"] is None
    assert meta["input_settings"]["recon"] == {
        "scored_window_seconds": 1.0,
        "split": "val",
        "bin_size_ms": 1.0,
        "window_length_seconds": 1.0,
        "max_eval_seconds": 600.0,
        "metrics": {
            "behavioral_regression_cursor_pos": {
                "metric_class": "behavioral_regression",
                "chunk_window_ms": [200, 1000],
            }
        },
    }
    # Only the listed run settings, not e.g. the fitted noise scale.
    assert meta["input_settings"]["m1"] == {
        "corruptions": ["jitter"],
        "n_replicas": 10,
        "scored_window_seconds": 1.0,
    }
    assert meta["input_settings"]["m2"] is None
    assert meta["poisson_norm_sessions"] == {"source": "precomputed", "session_ids": None}


def test_common_parts_only_reaches_the_poisson_recompute_check(tmp_path):
    """An aspect the Poisson reference does not save is an error; the job config
    common_parts_only=true turns it into a warning."""
    from ntt.evaluation.aggregation.comparability import UncheckedComparabilityError

    report = {
        "results": {
            "population_structure_preservation": {
                "name": "population_structure_preservation",
                "category": "temporal_fidelity",
                "direction": "higher",
                "level": "population",
                "raw_value": 0.6,
            }
        }
    }
    recon = tmp_path / "results.json"
    recon.write_text(
        json.dumps({"aggregated": report, "per_session": {"s1": {}}, "scored_window_seconds": 1.0})
    )
    # An older reference that saves no settings (no metadata block).
    reference = tmp_path / "poisson.json"
    reference.write_text(
        json.dumps({"aggregated": report, "perfect_aggregated": report, "per_session": {"s1": {}}})
    )
    kwargs = {
        "recon_path": str(recon),
        "poisson_reference": str(reference),
        "force_recompute_poisson": True,
        "output_path": str(tmp_path / "out"),
    }
    with pytest.raises(UncheckedComparabilityError, match="common_parts_only=true"):
        with NTTAggregationJob(**kwargs) as job:
            job.run()
    with NTTAggregationJob(**kwargs, common_parts_only=True) as job:
        job.run()
    assert (tmp_path / "out" / "category_scores.json").exists()


def _recon_with_tokenizer_config(tmp_path: Path, tokenizer_config: dict) -> Path:
    recon = json.loads(_write_recon(tmp_path).read_text())
    recon |= {"tokenizer_config": tokenizer_config, "aggregated": {"results": {}, "config": {}}}
    recon_file = tmp_path / "results.json"
    recon_file.write_text(json.dumps(recon))
    return recon_file


def _input_settings(tmp_path: Path, **paths: str) -> dict:
    job = NTTAggregationJob(
        **paths,
        encoding_path=str(_write_encoding(tmp_path, nc_r2_mean=0.3)),
        output_path=str(tmp_path / "out"),
    )
    with job:
        result = job.run()
    return json.loads(Path(result["output_path"]).read_text())["metadata"]["input_settings"]


def test_recon_geometry_falls_back_to_run_meta_data_config(tmp_path):
    """An eval can save only the tokenizer KIND (its config failed to serialize)."""
    recon_file = _recon_with_tokenizer_config(tmp_path, {"KIND": "cosmos_2d"})
    (tmp_path / "run_meta.json").write_text(
        json.dumps(
            {"data_config": {"bin_size_ms": 1.0, "window_length_seconds": 1.0, "dataset": "x"}}
        )
    )
    recon = _input_settings(tmp_path, recon_path=str(recon_file))["recon"]
    assert recon["bin_size_ms"] == 1.0
    assert recon["window_length_seconds"] == 1.0
    assert "dataset" not in recon


def test_recon_geometry_prefers_the_tokenizer_config(tmp_path):
    recon_file = _recon_with_tokenizer_config(
        tmp_path, {"KIND": "x", "bin_size_ms": 10.0, "window_length_seconds": 1.0}
    )
    (tmp_path / "run_meta.json").write_text(
        json.dumps({"data_config": {"bin_size_ms": 1.0, "window_length_seconds": 0.1}})
    )
    recon = _input_settings(tmp_path, recon_path=str(recon_file))["recon"]
    assert (recon["bin_size_ms"], recon["window_length_seconds"]) == (10.0, 1.0)


def test_recon_geometry_fills_only_the_missing_value(tmp_path):
    recon_file = _recon_with_tokenizer_config(tmp_path, {"KIND": "x", "bin_size_ms": 10.0})
    (tmp_path / "run_meta.json").write_text(
        json.dumps({"data_config": {"bin_size_ms": 1.0, "window_length_seconds": 1.0}})
    )
    recon = _input_settings(tmp_path, recon_path=str(recon_file))["recon"]
    assert (recon["bin_size_ms"], recon["window_length_seconds"]) == (10.0, 1.0)


def test_robustness_settings_come_from_run_meta_when_the_report_has_no_config(tmp_path):
    recon_file = _recon_with_tokenizer_config(
        tmp_path, {"KIND": "x", "bin_size_ms": 1.0, "window_length_seconds": 1.0}
    )
    m1_dir = tmp_path / "m1_run"
    m1_dir.mkdir()
    (m1_dir / "eval_report.json").write_text(json.dumps({"results": {}}))
    (m1_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "tokenizer_config": {"KIND": "cosmos_2d"},
                "data_config": {"bin_size_ms": 1.0, "window_length_seconds": 1.0},
                "scored_window_seconds": 1.0,
                "n_replicas": 10,
            }
        )
    )
    settings = _input_settings(
        tmp_path, recon_path=str(recon_file), m1_path=str(m1_dir / "eval_report.json")
    )
    assert settings["m1"] == {
        "bin_size_ms": 1.0,
        "window_length_seconds": 1.0,
        "scored_window_seconds": 1.0,
        "n_replicas": 10,
    }


def test_robustness_window_is_derived_from_the_report_config(tmp_path):
    """An M1 eval_report.json that does not record scored_window_seconds, with its window
    settings in the config, is scored with the derived window."""
    recon_file = _recon_with_tokenizer_config(
        tmp_path, {"KIND": "x", "bin_size_ms": 1.0, "window_length_seconds": 1.0}
    )
    m1_file = tmp_path / "eval_report.json"
    m1_file.write_text(
        json.dumps(
            {
                "results": {},
                "config": {
                    "n_replicas": 10,
                    "evaluate_all_neurons": True,
                    "neuron_selection": {"policy": "eval_unit_selection"},
                    "eval_window_seconds": 1.0,
                },
            }
        )
    )
    settings = _input_settings(tmp_path, recon_path=str(recon_file), m1_path=str(m1_file))
    assert settings["m1"] == {
        "n_replicas": 10,
        "evaluate_all_neurons": True,
        "scored_window_seconds": 1.0,
    }


def test_robustness_report_without_a_derivable_window_leaves_it_unknown(tmp_path):
    recon_file = _recon_with_tokenizer_config(tmp_path, {"KIND": "x"})
    m1_file = tmp_path / "eval_report.json"
    m1_file.write_text(json.dumps({"results": {}, "config": {"n_replicas": 10}}))
    settings = _input_settings(tmp_path, recon_path=str(recon_file), m1_path=str(m1_file))
    assert "scored_window_seconds" not in settings["m1"]


def test_recon_geometry_missing_from_the_tokenizer_config_stays_unknown(tmp_path):
    recon_file = _recon_with_tokenizer_config(tmp_path, {"KIND": "cosmos_2d"})
    recon = _input_settings(tmp_path, recon_path=str(recon_file))["recon"]
    assert "bin_size_ms" not in recon and "window_length_seconds" not in recon
