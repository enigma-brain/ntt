"""Tests for MLflow evaluation logging utilities."""

import json
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ntt.core.components.types import DataType
from ntt.evaluation.aggregation.comparison import MetricComparison
from ntt.evaluation.core.types import (
    EvaluationReport,
    MetricCategory,
    MetricDirection,
    MetricLevel,
    MetricResult,
)
from ntt.evaluation.mlflow_logging import (
    MLflowEvalLogger,
    MLflowLoggingConfig,
    flatten_report_to_metrics,
    log_robustness_run_to_mlflow,
    parse_training_run_id,
    remap_input_robustness_metric_keys_for_mlflow,
    remap_latent_robustness_metric_keys_for_mlflow,
    resolve_eval_run_name,
)


def test_remap_latent_robustness_metric_keys_moves_per_session_to_session():
    """latent_robustness (M2) per-session keys adopt the compact ``session/<sid>/`` layout while
    aggregate keys (the ones the Google sheet extracts) stay byte-identical."""
    metrics = {
        # aggregate — must be preserved exactly (sheet's _M2_EXACT_KEY)
        "eval_m2/smoothed_squared_error/eps_0.03": 0.5,
        "eval_m2/acg/eps_0.1": 0.1,
        # per-session — verbose layout to be rewritten
        "eval_m2/per_session/PE_201018/smoothed_squared_error/eps_0.03": 0.6,
        "eval_m2/per_session/PE_201018/acg/eps_0.1": 0.2,
    }
    out = remap_latent_robustness_metric_keys_for_mlflow(metrics, prefix="eval_m2")

    # aggregate untouched
    assert out["eval_m2/smoothed_squared_error/eps_0.03"] == 0.5
    assert out["eval_m2/acg/eps_0.1"] == 0.1
    # per_session -> session
    assert out["eval_m2/session/PE_201018/smoothed_squared_error/eps_0.03"] == 0.6
    assert out["eval_m2/session/PE_201018/acg/eps_0.1"] == 0.2
    # no verbose keys remain
    assert not any("per_session" in key for key in out)


def test_remap_latent_robustness_metric_keys_raises_on_conflict():
    """Two distinct values colliding on the same remapped key must fail loud."""
    metrics = {
        "eval_m2/per_session/PE_201018/acg/eps_0.1": 0.2,
        "eval_m2/session/PE_201018/acg/eps_0.1": 0.9,  # collides after remap
    }
    with pytest.raises(
        ValueError, match=r"Conflicting latent_robustness \(M2\) MLflow metric remap"
    ):
        remap_latent_robustness_metric_keys_for_mlflow(metrics, prefix="eval_m2")


def _make_result(
    name: str,
    value: float | np.ndarray,
    *,
    direction: MetricDirection = MetricDirection.HIGHER_IS_BETTER,
    category: MetricCategory = MetricCategory.SINGLE_UNIT_STATISTICS,
    level: MetricLevel = MetricLevel.POPULATION,
) -> MetricResult:
    return MetricResult(
        name=name,
        raw_value=value,
        direction=direction,
        category=category,
        level=level,
        metadata={},
    )


def _make_report(
    results: dict[str, MetricResult],
    category_scores: dict[MetricCategory, float] | None = None,
) -> EvaluationReport:
    return EvaluationReport(
        results=results,
        category_scores=category_scores or {},
        config={},
    )


class TestFlattenMetrics:
    def test_scalar_metric_logged_directly(self):
        report = _make_report(
            {
                "subspace_overlap": _make_result(
                    "subspace_overlap", 0.85, level=MetricLevel.POPULATION
                ),
            }
        )
        flat = flatten_report_to_metrics(report)
        assert flat["eval/subspace_overlap"] == pytest.approx(0.85)
        assert "eval/subspace_overlap/std" not in flat

    def test_per_unit_metric_produces_mean_and_std(self):
        values = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        report = _make_report(
            {
                "firing_rate_error": _make_result(
                    "firing_rate_error", values, level=MetricLevel.PER_UNIT
                ),
            }
        )
        flat = flatten_report_to_metrics(report)
        assert flat["eval/firing_rate_error"] == pytest.approx(np.mean(values))
        assert flat["eval/firing_rate_error/std"] == pytest.approx(np.std(values))

    def test_category_scores_are_included(self):
        report = _make_report(
            results={},
            category_scores={
                MetricCategory.TEMPORAL_FIDELITY: 0.72,
                MetricCategory.SINGLE_UNIT_STATISTICS: 0.65,
            },
        )
        flat = flatten_report_to_metrics(report)
        assert flat["eval/category/temporal_fidelity"] == pytest.approx(0.72)
        assert flat["eval/category/single_unit_statistics"] == pytest.approx(0.65)

    def test_skipped_count_is_logged(self):
        report = _make_report(results={})
        report.skipped_metrics = {"metric_a": "reason_a", "metric_b": "reason_b"}
        flat = flatten_report_to_metrics(report)
        assert flat["eval/n_skipped"] == 2

    def test_nan_values_are_excluded(self):
        report = _make_report(
            {
                "bad_metric": _make_result("bad_metric", float("nan")),
            }
        )
        flat = flatten_report_to_metrics(report)
        assert "eval/bad_metric" not in flat

    def test_inf_scalar_values_are_excluded(self):
        # MLflow accepts ±inf but it corrupts dashboards/aggregates, and the
        # series paths (_safe_float / np.isfinite) already drop it — the scalar
        # path must match. math.isnan(inf) is False, so a naive NaN-only guard
        # would let inf through.
        report = _make_report(
            {
                "pos_inf": _make_result("pos_inf", float("inf")),
                "neg_inf": _make_result("neg_inf", float("-inf")),
            }
        )
        flat = flatten_report_to_metrics(report)
        assert "eval/pos_inf" not in flat
        assert "eval/neg_inf" not in flat

    def test_inf_in_array_metric_is_excluded(self):
        # An array whose reduction is non-finite (e.g. all +inf → mean +inf)
        # must not reach MLflow on the mean or std keys.
        report = _make_report(
            {
                "arr_inf": _make_result(
                    "arr_inf",
                    np.array([float("inf"), float("inf")]),
                    level=MetricLevel.PER_UNIT,
                ),
            }
        )
        flat = flatten_report_to_metrics(report)
        assert "eval/arr_inf" not in flat
        assert "eval/arr_inf/std" not in flat


class TestRemapM1MetricKeys:
    def test_collision_does_not_abort_eval(self):
        # Both keys rewrite to ``eval_m1/joint/core_avg``. The remap runs
        # post-compute, so a conflict must NOT raise (it would abort an
        # otherwise-complete eval) — keep the first value, like
        # merge_input_robustness_metric_sources does.
        metrics = {
            "eval_m1/robustness_input_core_avg": 0.5,
            "eval_m1/robustness_input/core_avg": 0.9,
        }
        out = remap_input_robustness_metric_keys_for_mlflow(metrics, prefix="eval_m1")
        assert out["eval_m1/joint/core_avg"] == pytest.approx(0.5), (
            "first-wins: the leading value should be kept on collision"
        )

    def test_no_collision_when_values_match(self):
        metrics = {
            "eval_m1/robustness_input_core_avg": 0.5,
            "eval_m1/robustness_input/core_avg": 0.5,
        }
        out = remap_input_robustness_metric_keys_for_mlflow(metrics, prefix="eval_m1")
        assert out["eval_m1/joint/core_avg"] == pytest.approx(0.5)


class TestParseTrainingRunId:
    def test_extracts_run_id_from_mlflow_artifacts_uri(self):
        uri = "mlflow-artifacts:/runs/abc123def456/artifacts/checkpoints/best.ckpt"
        assert parse_training_run_id(uri) == "abc123def456"

    def test_extracts_run_id_from_mlflow_colon_uri(self):
        uri = "mlflow:/runs/xyz789/artifacts/model.ckpt"
        assert parse_training_run_id(uri) == "xyz789"

    def test_returns_none_for_local_path(self):
        assert parse_training_run_id("/data/checkpoints/epoch=5.ckpt") is None

    def test_returns_none_for_mlflow_uri_without_runs(self):
        uri = "mlflow-artifacts:/models/my-model/latest/model.ckpt"
        assert parse_training_run_id(uri) is None


class TestResolveEvalRunName:
    """Covers the four resolution paths plus the global-state contract.

    The resolver MUST NOT mutate the process-global MLflow tracking URI
    Calling ``mlflow.set_tracking_uri`` to scope the lookup would clobber
    whatever URI the caller had set up, so the resolver uses
    a scoped ``MlflowClient(tracking_uri=...)`` instead.
    """

    _SUFFIX_RE = r"-[0-9a-f]{6}$"

    def test_explicit_name_takes_precedence(self):
        name = resolve_eval_run_name(explicit="my-run", ckpt_path=None, spec_kind="vqvae")
        assert name.startswith("my-run-")
        assert re.search(self._SUFFIX_RE, name)

    def test_explicit_wins_even_when_ckpt_resolvable(self):
        # If an explicit name is given, the resolver must not construct a client.
        with patch("ntt.evaluation.mlflow_logging.MlflowClient") as mock_client_cls:
            name = resolve_eval_run_name(
                explicit="explicit-name",
                ckpt_path="mlflow-artifacts:/runs/run123/artifacts/best.ckpt",
                spec_kind="vqvae",
            )
        assert name.startswith("explicit-name-")
        mock_client_cls.assert_not_called()

    def test_falls_back_to_spec_kind_when_no_ckpt(self):
        name = resolve_eval_run_name(explicit=None, ckpt_path=None, spec_kind="identity")
        assert name.startswith("identity-")

    def test_falls_back_to_spec_kind_for_local_ckpt_path(self):
        # Local checkpoint paths don't carry a run id; resolver should not
        # construct a client at all.
        with patch("ntt.evaluation.mlflow_logging.MlflowClient") as mock_client_cls:
            name = resolve_eval_run_name(
                explicit=None,
                ckpt_path="/data/checkpoints/best.ckpt",
                spec_kind="vqvae",
            )
        assert name.startswith("vqvae-")
        mock_client_cls.assert_not_called()

    def test_uses_training_run_name_when_lookup_succeeds(self):
        mock_run = MagicMock()
        mock_run.info.run_name = "stately-trout-42"
        mock_client = MagicMock()
        mock_client.get_run.return_value = mock_run

        with (
            patch(
                "ntt.evaluation.mlflow_logging.MlflowClient",
                return_value=mock_client,
            ) as mock_client_cls,
            patch("ntt.evaluation.mlflow_logging.mlflow.set_tracking_uri") as mock_set_uri,
        ):
            name = resolve_eval_run_name(
                explicit=None,
                ckpt_path="mlflow-artifacts:/runs/abc123/artifacts/best.ckpt",
                spec_kind="vqvae",
                tracking_uri="http://example:5000",
            )

        assert name.startswith("stately-trout-42-")
        # Scoped client constructed with the supplied URI; module-level
        # `set_tracking_uri` must NOT have been called.
        mock_client_cls.assert_called_once_with(tracking_uri="http://example:5000")
        mock_set_uri.assert_not_called()
        mock_client.get_run.assert_called_once_with("abc123")

    def test_falls_back_to_spec_kind_when_lookup_fails(self):
        mock_client = MagicMock()
        mock_client.get_run.side_effect = RuntimeError("tracking server down")

        with patch(
            "ntt.evaluation.mlflow_logging.MlflowClient",
            return_value=mock_client,
        ):
            name = resolve_eval_run_name(
                explicit=None,
                ckpt_path="mlflow-artifacts:/runs/abc123/artifacts/best.ckpt",
                spec_kind="vqvae",
                tracking_uri="http://example:5000",
            )

        assert name.startswith("vqvae-")

    def test_falls_back_to_spec_kind_when_training_run_name_is_blank(self):
        # MLflow returns the run but `run_name` is None/empty — treat as a
        # missing name and continue to the spec_kind fallback.
        mock_run = MagicMock()
        mock_run.info.run_name = ""
        mock_client = MagicMock()
        mock_client.get_run.return_value = mock_run

        with patch(
            "ntt.evaluation.mlflow_logging.MlflowClient",
            return_value=mock_client,
        ):
            name = resolve_eval_run_name(
                explicit=None,
                ckpt_path="mlflow-artifacts:/runs/abc123/artifacts/best.ckpt",
                spec_kind="vqvae",
            )

        assert name.startswith("vqvae-")

    def test_does_not_mutate_global_tracking_uri(self):
        """The resolver must not call ``mlflow.set_tracking_uri``, which would
        clobber the caller's MLflow context; the scoped client leaves it untouched."""
        import mlflow as real_mlflow

        before = real_mlflow.get_tracking_uri()

        mock_run = MagicMock()
        mock_run.info.run_name = "training-run"
        mock_client = MagicMock()
        mock_client.get_run.return_value = mock_run

        with patch(
            "ntt.evaluation.mlflow_logging.MlflowClient",
            return_value=mock_client,
        ):
            resolve_eval_run_name(
                explicit=None,
                ckpt_path="mlflow-artifacts:/runs/abc123/artifacts/best.ckpt",
                spec_kind="vqvae",
                tracking_uri="http://should-not-stick:5000",
            )

        after = real_mlflow.get_tracking_uri()
        assert after == before, (
            f"resolve_eval_run_name leaked tracking URI: was {before!r}, now {after!r}"
        )

    def test_disambiguator_differs_across_calls(self):
        # Repeat calls with identical inputs should yield distinct names —
        # otherwise the MLflow UI gets duplicate run names on re-eval.
        names = {
            resolve_eval_run_name(explicit="same", ckpt_path=None, spec_kind="vqvae")
            for _ in range(20)
        }
        assert len(names) == 20


def _make_comparison(name: str, normalized_score: float) -> MetricComparison:
    return MetricComparison(
        name=name,
        category=MetricCategory.SINGLE_UNIT_STATISTICS,
        direction=MetricDirection.HIGHER_IS_BETTER,
        level=MetricLevel.POPULATION,
        model_value=0.5,
        baseline_value=0.0,
        ceiling_value=1.0,
        normalized_score=normalized_score,
        status="ok",
    )


class TestMLflowEvalLogger:
    @patch("ntt.evaluation.mlflow_logging.mlflow")
    def test_log_creates_run_and_logs_metrics(self, mock_mlflow):
        mock_run = MagicMock()
        mock_run.info.run_id = "eval_run_123"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        report = _make_report(
            {
                "f1": _make_result("f1", 0.9),
            }
        )

        logger = MLflowEvalLogger(experiment_name="test-eval")
        run_id = logger.log(report=report, checkpoint_path="/local/best.ckpt")

        assert run_id == "eval_run_123"
        mock_mlflow.set_experiment.assert_called_once_with("test-eval")
        mock_mlflow.log_metrics.assert_called_once()
        logged_metrics = mock_mlflow.log_metrics.call_args[0][0]
        assert "eval/f1" in logged_metrics

    @patch("ntt.evaluation.mlflow_logging.mlflow")
    def test_log_tags_include_checkpoint_and_split(self, mock_mlflow):
        mock_run = MagicMock()
        mock_run.info.run_id = "eval_run_456"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        report = _make_report({"f1": _make_result("f1", 0.9)})
        logger = MLflowEvalLogger(experiment_name="test-eval")
        logger.log(
            report=report,
            checkpoint_path="mlflow-artifacts:/runs/train_abc/artifacts/best.ckpt",
            tags={"ntt.eval.split": "val"},
        )

        mock_mlflow.set_tags.assert_called_once()
        tags = mock_mlflow.set_tags.call_args[0][0]
        expected_ckpt = "mlflow-artifacts:/runs/train_abc/artifacts/best.ckpt"
        assert tags["ntt.eval.checkpoint_path"] == expected_ckpt
        assert tags["ntt.eval.training_run_id"] == "train_abc"
        assert tags["ntt.eval.split"] == "val"

    @patch("ntt.evaluation.mlflow_logging.mlflow")
    def test_log_saves_report_json_artifact(self, mock_mlflow):
        mock_run = MagicMock()
        mock_run.info.run_id = "eval_run_789"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        report = _make_report({"f1": _make_result("f1", 0.9)})
        logger = MLflowEvalLogger(experiment_name="test-eval")
        logger.log(report=report)

        mock_mlflow.log_artifact.assert_called()
        artifact_calls = mock_mlflow.log_artifact.call_args_list
        paths = [c[0][0] for c in artifact_calls]
        assert any("eval_report.json" in p for p in paths)

    @patch("ntt.evaluation.mlflow_logging.mlflow")
    def test_log_includes_normalized_scores_from_comparisons(self, mock_mlflow):
        mock_run = MagicMock()
        mock_run.info.run_id = "eval_run_comp"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        report = _make_report({"isi": _make_result("isi", 0.5)})
        comparisons = {"isi": _make_comparison("isi", normalized_score=0.75)}

        logger = MLflowEvalLogger(experiment_name="test-eval")
        logger.log(report=report, comparisons=comparisons)

        logged_metrics = mock_mlflow.log_metrics.call_args[0][0]
        assert logged_metrics["eval/normalized/isi"] == pytest.approx(0.75)

    @patch("ntt.evaluation.mlflow_logging.mlflow")
    def test_log_per_session_reports(self, mock_mlflow):
        mock_run = MagicMock()
        mock_run.info.run_id = "eval_run_sess"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        agg_report = _make_report({"f1": _make_result("f1", 0.9)})
        session_report = _make_report({"f1": _make_result("f1", 0.85)})

        logger = MLflowEvalLogger(experiment_name="test-eval")
        logger.log(
            report=agg_report,
            per_session_reports={"session_A": session_report},
        )

        # Check per-session artifact was logged
        artifact_calls = mock_mlflow.log_artifact.call_args_list
        paths = [c[0][0] for c in artifact_calls]
        assert any("session_A.json" in p for p in paths)

        # Check n_sessions tag
        tags = mock_mlflow.set_tags.call_args[0][0]
        assert tags["ntt.eval.n_sessions"] == "1"

    @patch("ntt.evaluation.mlflow_logging.mlflow")
    def test_per_session_report_flattens_slash_in_sid(self, mock_mlflow):
        """Session IDs of the form ``brainset/session`` are flattened to
        ``brainset__session.json`` and uploaded under a flat ``per_session/``
        artifact root. This keeps the layout consistent with sibling
        artifacts (``comparisons/``, ``checkpoints/``) and avoids the
        FileNotFoundError that would result from writing through a
        non-existent intermediate directory."""
        mock_run = MagicMock()
        mock_run.info.run_id = "eval_run_nested"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        agg_report = _make_report({"f1": _make_result("f1", 0.9)})
        session_report = _make_report({"f1": _make_result("f1", 0.85)})
        sid = "perich_miller_population_2018/c_20131022_center_out_reaching"

        logger = MLflowEvalLogger(experiment_name="test-eval")
        # Must not raise FileNotFoundError.
        logger.log(
            report=agg_report,
            per_session_reports={sid: session_report},
        )

        artifact_calls = mock_mlflow.log_artifact.call_args_list
        session_calls = [
            c for c in artifact_calls if c[1].get("artifact_path", "").startswith("per_session")
        ]
        assert len(session_calls) == 1
        # Flat artifact root; brainset is encoded in the filename.
        assert session_calls[0][1]["artifact_path"] == "per_session"
        assert session_calls[0][0][0].endswith(
            "perich_miller_population_2018__c_20131022_center_out_reaching.json"
        )


class TestRobustnessMLflowLogging:
    @patch("ntt.evaluation.mlflow_logging.mlflow")
    @patch("ntt.evaluation.mlflow_logging.MlflowClient")
    def test_m1_logging_uses_compact_scalar_metrics_only(
        self, mock_client_cls, mock_mlflow, tmp_path
    ):
        jsonl_path = tmp_path / "results.jsonl"
        jsonl_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "session_id": "S1",
                            "segment_index": 0,
                            "corruption": "drop_p01",
                            "metric": "rec_loss",
                            "sub_label": None,
                            "raw_value": 1.0,
                            "status": "ok",
                        }
                    ),
                    json.dumps(
                        {
                            "session_id": "S1",
                            "segment_index": 0,
                            "corruption": "aggregate_non_population",
                            "metric": "robustness_input_core_avg",
                            "raw_value": 1.5,
                            "status": "ok",
                        }
                    ),
                ]
            )
            + "\n"
        )

        mock_run = MagicMock()
        mock_run.info.run_id = "eval_run_m1"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        config = MLflowLoggingConfig(
            enabled=True,
            experiment_name="test-eval",
            metric_prefix="eval_m1",
        )

        log_robustness_run_to_mlflow(
            mode="m1",
            config=config,
            jsonl_path=jsonl_path,
            label_to_type_strength={"drop_p01": ("drop", 0.1)},
        )

        mock_client.log_batch.assert_not_called()
        mock_client.log_metric.assert_not_called()
        mock_client_cls.assert_not_called()
        logged = {}
        for call in mock_mlflow.log_metrics.call_args_list:
            logged.update(call.args[0])
        assert logged["eval_m1/n_records"] == 2.0
        assert logged["eval_m1/joint/core_avg"] == 1.5
        assert logged["eval_m1/session/S1/joint/core_avg"] == 1.5
        assert not any("/_x_strength" in key for key in logged)
        assert not any(key.startswith("eval_m1/per_session/") for key in logged)
        assert not any(
            key.startswith("eval_m1/S1/") and key != "eval_m1/session/S1/joint/core_avg"
            for key in logged
        )

    @patch("ntt.evaluation.mlflow_logging.log")
    @patch("ntt.evaluation.mlflow_logging.mlflow")
    @patch("ntt.evaluation.mlflow_logging.MlflowClient")
    def test_m1_metric_merge_conflict_logs_trusted_value_and_tags_run(
        self, mock_client_cls, mock_mlflow, mock_log, tmp_path
    ):
        """A jsonl-vs-report divergence must not abort a completed eval.

        The jsonl (source of truth) value is the one logged; the run gets a
        searchable conflict tag and the full candidates land in the error log.
        """
        jsonl_path = tmp_path / "results.jsonl"
        jsonl_path.write_text(
            json.dumps(
                {
                    "session_id": "S1",
                    "segment_index": 0,
                    "corruption": "aggregate_non_population",
                    "metric": "robustness_input_core_avg",
                    "raw_value": 1.5,
                    "status": "ok",
                }
            )
            + "\n"
        )
        # Report disagrees with the jsonl on the same remapped key
        # (eval_m1/joint/core_avg): 9.9 vs the jsonl's 1.5.
        report = _make_report({"robustness_input_core_avg": _make_result("core_avg", 9.9)})

        mock_run = MagicMock()
        mock_run.info.run_id = "eval_run_m1"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        config = MLflowLoggingConfig(
            enabled=True,
            experiment_name="test-eval",
            metric_prefix="eval_m1",
        )

        log_robustness_run_to_mlflow(
            mode="m1",
            config=config,
            jsonl_path=jsonl_path,
            label_to_type_strength={},
            report=report,
        )

        logged = {}
        for call in mock_mlflow.log_metrics.call_args_list:
            logged.update(call.args[0])
        assert logged["eval_m1/joint/core_avg"] == 1.5  # jsonl wins
        mock_log.error.assert_called()
        tag_calls = {call.args[0]: call.args[1] for call in mock_mlflow.set_tag.call_args_list}
        assert "eval_m1/joint/core_avg" in tag_calls["ntt.eval.m1_metric_merge_conflicts"]

    @patch("ntt.evaluation.mlflow_logging.log")
    @patch("ntt.evaluation.mlflow_logging.mlflow")
    @patch("ntt.evaluation.mlflow_logging.MlflowClient")
    def test_m2_artifact_failure_does_not_abort_logged_metrics(
        self,
        mock_client_cls,
        mock_mlflow,
        mock_log,
        tmp_path,
    ):
        jsonl_path = tmp_path / "results.jsonl"
        jsonl_path.write_text(
            json.dumps(
                {
                    "session_id": "S1",
                    "segment_index": 0,
                    "epsilon": 0.03,
                    "k": 0,
                    "metric": "spike_count_correlation",
                    "raw_value": 0.1,
                    "status": "ok",
                }
            )
            + "\n"
        )
        artifact_path = tmp_path / "results.csv"
        artifact_path.write_text("x\n")

        mock_run = MagicMock()
        mock_run.info.run_id = "eval_run_m2_artifacts"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
        mock_mlflow.log_artifact.side_effect = RuntimeError("artifact upload failed")

        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        config = MLflowLoggingConfig(
            enabled=True,
            experiment_name="test-eval",
            metric_prefix="eval_m2",
        )

        run_id = log_robustness_run_to_mlflow(
            mode="m2",
            config=config,
            jsonl_path=jsonl_path,
            artifact_paths=[artifact_path],
            include_jsonl_artifact=False,
        )

        assert run_id == "eval_run_m2_artifacts"
        mock_mlflow.log_metrics.assert_called()
        # latent_robustness (M2) does not log the per-(session, metric) epsilon curve: it
        # would duplicate
        # the per-epsilon scalars logged via ``log_metrics``. ``log_batch`` is only
        # used by the step-series upload, so it must NOT be called for latent_robustness (M2).
        mock_client.log_batch.assert_not_called()
        mock_log.warning.assert_called()

    @patch("ntt.evaluation.mlflow_logging.mlflow")
    @patch("ntt.evaluation.mlflow_logging.MlflowClient")
    def test_m2_upload_uses_fresh_in_memory_eval_report_artifact(
        self,
        mock_client_cls,
        mock_mlflow,
        tmp_path,
    ):
        jsonl_path = tmp_path / "results.jsonl"
        jsonl_path.write_text("")

        stale_report_path = tmp_path / "eval_report.json"
        stale_report_path.write_text(
            json.dumps(
                {
                    "results": {},
                    "skipped_metrics": {},
                    "category_scores": None,
                    "config": {},
                    "original_data_type": None,
                    "reconstructed_data_type": None,
                }
            )
        )

        report = EvaluationReport(
            results={},
            config={"max_eval_seconds": 30.0},
            original_data_type=DataType.RATES,
            reconstructed_data_type=DataType.RATES,
        )

        mock_run = MagicMock()
        mock_run.info.run_id = "eval_run_m2_fresh_report"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = MagicMock()

        uploaded_report_payloads: list[dict] = []

        def _capture_log_artifact(path: str, artifact_path: str | None = None):
            if path.endswith("eval_report.json"):
                uploaded_report_payloads.append(json.loads(Path(path).read_text()))

        mock_mlflow.log_artifact.side_effect = _capture_log_artifact

        config = MLflowLoggingConfig(
            enabled=True,
            experiment_name="test-eval",
            metric_prefix="eval_m2",
        )

        log_robustness_run_to_mlflow(
            mode="m2",
            config=config,
            jsonl_path=jsonl_path,
            report=report,
            artifact_paths=[stale_report_path],
            artifact_root=tmp_path,
            include_jsonl_artifact=False,
        )

        assert len(uploaded_report_payloads) == 1
        uploaded = uploaded_report_payloads[0]
        assert uploaded["config"] == {"max_eval_seconds": 30.0}
        assert uploaded["original_data_type"] == DataType.RATES.value
        assert uploaded["reconstructed_data_type"] == DataType.RATES.value

    @patch("ntt.evaluation.mlflow_logging.log")
    @patch("ntt.evaluation.mlflow_logging.mlflow")
    @patch("ntt.evaluation.mlflow_logging.MlflowClient")
    def test_robustness_logging_surfaces_config_metadata_as_params(
        self,
        mock_client_cls,
        mock_mlflow,
        mock_log,
        tmp_path,
    ):
        jsonl_path = tmp_path / "results.jsonl"
        jsonl_path.write_text("")
        mock_run = MagicMock()
        mock_run.info.run_id = "eval_run_params"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = MagicMock()
        config = MLflowLoggingConfig(enabled=True, experiment_name="test-eval")

        log_robustness_run_to_mlflow(
            mode="m2",
            config=config,
            jsonl_path=jsonl_path,
            include_jsonl_artifact=False,
            per_session_reports={"S1": EvaluationReport(results={})},
            run_metadata={
                "eval_config": {
                    "eval_window_seconds": 1.0,
                    "robustness_latent": {"n_eval_windows": 5},
                },
                "data_config": {
                    "val_selections": [
                        {
                            "selection": [
                                {"brainset": "cowley_v4_compact_2024", "sessions": ["PE_201018"]},
                            ],
                        },
                    ],
                },
                "tokenizer_config": {
                    "KIND": "identity",
                    "token_bin_size_ms": 20.0,
                },
            },
        )

        mock_mlflow.log_params.assert_called_once()
        params = mock_mlflow.log_params.call_args.args[0]
        assert params["eval_window_seconds"] == 1.0
        assert params["n_eval_windows"] == 5
        assert params["data.val_brainsets"] == "cowley_v4_compact_2024"
        assert params["data.n_val_sessions"] == 1
        assert params["tokenizer.KIND"] == "identity"
        assert params["tokenizer.token_bin_size_ms"] == 20.0
        assert params["n_sessions"] == 1
        mock_log.warning.assert_not_called()

    @patch("ntt.evaluation.mlflow_logging.log")
    @patch("ntt.evaluation.mlflow_logging.mlflow")
    @patch("ntt.evaluation.mlflow_logging.MlflowClient")
    def test_robustness_logging_skips_missing_metadata_params_with_warning(
        self,
        mock_client_cls,
        mock_mlflow,
        mock_log,
        tmp_path,
    ):
        jsonl_path = tmp_path / "results.jsonl"
        jsonl_path.write_text("")
        mock_run = MagicMock()
        mock_run.info.run_id = "eval_run_missing_params"
        mock_mlflow.start_run.return_value.__enter__ = MagicMock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)
        mock_client_cls.return_value = MagicMock()
        config = MLflowLoggingConfig(enabled=True, experiment_name="test-eval")

        log_robustness_run_to_mlflow(
            mode="m2",
            config=config,
            jsonl_path=jsonl_path,
            include_jsonl_artifact=False,
            run_metadata={},
        )

        mock_mlflow.log_params.assert_not_called()
        mock_log.warning.assert_called()


def test_merge_conflict_keeps_trusted_value_and_reports_candidates():
    from ntt.evaluation.mlflow_logging import merge_input_robustness_metric_sources

    merged, conflicts = merge_input_robustness_metric_sources(
        [
            ("jsonl_summary", {"eval_m1/joint/core_avg": 1.0}),
            ("report_aggregate", {"eval_m1/joint/core_avg": 2.0}),
        ]
    )
    # First (most-trusted) source wins; both candidates are reported.
    assert merged["eval_m1/joint/core_avg"] == 1.0
    assert conflicts == {
        "eval_m1/joint/core_avg": [("jsonl_summary", 1.0), ("report_aggregate", 2.0)]
    }


def test_merge_close_values_are_not_a_conflict():
    from ntt.evaluation.mlflow_logging import merge_input_robustness_metric_sources

    merged, conflicts = merge_input_robustness_metric_sources(
        [
            ("jsonl_summary", {"eval_m1/joint/core_avg": 1.0}),
            (
                "report_aggregate",
                {
                    "eval_m1/joint/core_avg": 1.0 + 1e-12,
                    "eval_m1/session/s1/joint/core_avg": 3.0,
                },
            ),
        ]
    )
    assert not conflicts
    assert merged["eval_m1/session/s1/joint/core_avg"] == 3.0
    assert np.isclose(merged["eval_m1/joint/core_avg"], 1.0)


def test_merge_nan_vs_nan_is_not_a_conflict():
    from ntt.evaluation.mlflow_logging import merge_input_robustness_metric_sources

    merged, conflicts = merge_input_robustness_metric_sources(
        [
            ("jsonl_summary", {"eval_m1/joint/core_avg": float("nan")}),
            ("report_aggregate", {"eval_m1/joint/core_avg": float("nan")}),
        ]
    )
    assert not conflicts
    assert np.isnan(merged["eval_m1/joint/core_avg"])


def test_m1_multi_source_merge_disjoint_namespaces_ok():
    from ntt.evaluation.mlflow_logging import merge_input_robustness_metric_sources

    merged, conflicts = merge_input_robustness_metric_sources(
        [
            ("jsonl_summary", {"eval_m1/joint/core_avg": 1.0}),
            ("jsonl_series_scalars", {"eval_m1/n_skipped": 0.0}),
            ("report_aggregate", {"eval_m1/joint/core_avg": 1.0}),  # same value -> ok
            ("per_session_report/s1", {"eval_m1/session/s1/joint/core_avg": 2.0}),
        ]
    )
    assert not conflicts
    assert merged["eval_m1/joint/core_avg"] == 1.0
    assert merged["eval_m1/session/s1/joint/core_avg"] == 2.0
    assert merged["eval_m1/n_skipped"] == 0.0
