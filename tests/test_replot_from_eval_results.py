"""Tests for the replot_from_eval_results script."""

import json
from pathlib import Path

import pytest

from ntt.evaluation.aggregation.comparison import MetricComparison
from ntt.evaluation.core.types import (
    EvaluationReport,
    MetricCategory,
    MetricDirection,
    MetricLevel,
    MetricResult,
)


def _make_metric_result(name: str = "acg_correlation", value: float = 0.5) -> MetricResult:
    return MetricResult(
        name=name,
        category=MetricCategory.SINGLE_UNIT_STATISTICS,
        direction=MetricDirection.HIGHER_IS_BETTER,
        level=MetricLevel.POPULATION,
        raw_value=value,
        metadata={},
    )


def _make_results_json_data(
    *,
    include_comparisons: bool = False,
) -> dict:
    """Build a minimal results.json dict matching the eval job's _save_results format."""
    report = EvaluationReport(
        results={"acg_correlation": _make_metric_result()},
        category_scores={MetricCategory.SINGLE_UNIT_STATISTICS: 0.5},
        config={},
    )
    report_dict = report.to_dict()

    data = {
        "checkpoint": "/fake/path.ckpt",
        "tokenizer_config": {"KIND": "test"},
        "n_sessions": 1,
        "aggregation_strategy": "mean",
        "aggregated": report_dict,
        "per_session": {"session_a": report_dict},
        "per_session_per_segment": {"session_a": [report_dict]},
    }

    if include_comparisons:
        comparison = MetricComparison(
            name="acg_correlation",
            category=MetricCategory.SINGLE_UNIT_STATISTICS,
            direction=MetricDirection.HIGHER_IS_BETTER,
            level=MetricLevel.POPULATION,
            model_value=0.5,
            baseline_value=0.0,
            ceiling_value=1.0,
            normalized_score=0.5,
        )
        data["comparison_vs_poisson"] = {
            "acg_correlation": comparison.to_dict(),
        }
        data["normalized_scores_vs_poisson"] = {"acg_correlation": 0.5}

    return data


def _write_minimal_m1_results_csv(results_dir: Path) -> None:
    rows = [
        "session_id,segment_index,corruption,metric,sub_label,raw_value,status",
        "sess0,0,jitter|k_ms=0.0|p=0.0,robustness_input,replica_0,0.05,ok",
        "sess0,0,jitter|k_ms=1.0|p=0.5,robustness_input,replica_1,0.10,ok",
        "sess0,0,drop|fraction=0.1|p=1.0,robustness_input,replica_1,0.20,ok",
    ]
    (results_dir / "results.csv").write_text("\n".join(rows) + "\n")


def _write_minimal_m2_results_csv(results_dir: Path) -> None:
    rows = [
        "session_id,segment_index,epsilon,k,acg_correlation",
        "sess0,0,0.01,0,0.90",
        "sess0,0,0.10,0,0.70",
    ]
    (results_dir / "results.csv").write_text("\n".join(rows) + "\n")


class TestLoadSessionReports:
    def test_loads_per_session_reports(self):
        """load_session_reports should return session_reports keyed by session_id."""
        from scripts.replot_from_eval_results import load_session_reports

        data = _make_results_json_data()
        session_reports = load_session_reports(data)

        assert "session_a" in session_reports
        segments = session_reports["session_a"]
        assert len(segments) == 1
        _segment_stub, payload = segments[0]
        report = payload["report"]
        assert isinstance(report, EvaluationReport)
        assert "acg_correlation" in report.results

    def test_prefers_per_session_per_segment_when_available(self):
        """Should use per_session_per_segment (full detail) over per_session (aggregated)."""
        from scripts.replot_from_eval_results import load_session_reports

        data = _make_results_json_data()
        # Add a second segment so per_session_per_segment is distinguishable
        # from per_session (which only has one entry per session)
        data["per_session_per_segment"]["session_a"].append(
            data["per_session_per_segment"]["session_a"][0]
        )

        session_reports = load_session_reports(data)

        # Only the per_session_per_segment path yields 2 segments
        assert len(session_reports["session_a"]) == 2

    def test_falls_back_to_per_session_when_no_segments(self):
        """Should use per_session when per_session_per_segment is absent."""
        from scripts.replot_from_eval_results import load_session_reports

        data = _make_results_json_data()
        del data["per_session_per_segment"]

        session_reports = load_session_reports(data)

        assert "session_a" in session_reports
        segments = session_reports["session_a"]
        assert len(segments) == 1
        _, payload = segments[0]
        assert isinstance(payload["report"], EvaluationReport)

    def test_raises_on_missing_keys(self):
        """Should raise if neither per_session nor per_session_per_segment exists."""
        from scripts.replot_from_eval_results import load_session_reports

        with pytest.raises(ValueError, match="per_session"):
            load_session_reports({"aggregated": {}})


class TestLoadComparisonsFromResults:
    def test_returns_comparisons_when_present(self):
        from scripts.replot_from_eval_results import load_comparisons_from_results

        data = _make_results_json_data(include_comparisons=True)
        comparisons = load_comparisons_from_results(data)

        assert comparisons is not None
        assert "acg_correlation" in comparisons
        comp = comparisons["acg_correlation"]
        assert isinstance(comp, MetricComparison)
        assert comp.normalized_score == 0.5

    def test_returns_none_when_absent(self):
        from scripts.replot_from_eval_results import load_comparisons_from_results

        data = _make_results_json_data(include_comparisons=False)
        comparisons = load_comparisons_from_results(data)

        assert comparisons is None


class TestReplotIntegration:
    def test_main_generates_radar_when_comparisons_present(self, tmp_path: Path):
        """When results.json contains comparison_vs_poisson, the radar chart is generated."""
        from unittest.mock import patch

        from scripts.replot_from_eval_results import main

        data = _make_results_json_data(include_comparisons=True)
        results_dir = tmp_path / "eval_run"
        results_dir.mkdir()
        (results_dir / "results.json").write_text(json.dumps(data))

        output_dir = tmp_path / "plots"

        with patch(
            "sys.argv",
            ["replot", str(results_dir), "--output-dir", str(output_dir)],
        ):
            main()

        # The radar chart should exist
        assert (output_dir / "normalized_comparison_radar_cs.png").exists()

    def test_main_works_without_comparisons(self, tmp_path: Path):
        """When results.json has no comparisons, main() still succeeds (no radar)."""
        from unittest.mock import patch

        from scripts.replot_from_eval_results import main

        data = _make_results_json_data(include_comparisons=False)
        results_dir = tmp_path / "eval_run"
        results_dir.mkdir()
        (results_dir / "results.json").write_text(json.dumps(data))

        output_dir = tmp_path / "plots"

        with patch(
            "sys.argv",
            ["replot", str(results_dir), "--output-dir", str(output_dir)],
        ):
            main()

        # Should still produce metric_bars at minimum
        assert (output_dir / "metric_bars.png").exists()
        # No radar without comparisons
        assert not (output_dir / "normalized_comparison_radar_cs.png").exists()

    def test_main_auto_replots_wrapper_robustness_siblings(self, tmp_path: Path):
        """Sibling ``_m1`` / ``_m2`` dirs should trigger robustness joint plots."""
        from unittest.mock import patch

        from scripts.replot_from_eval_results import main

        data = _make_results_json_data(include_comparisons=False)
        results_dir = tmp_path / "eval_run"
        results_dir.mkdir()
        (results_dir / "results.json").write_text(json.dumps(data))

        m1_dir = tmp_path / "eval_run_m1"
        m1_dir.mkdir()
        _write_minimal_m1_results_csv(m1_dir)

        m2_dir = tmp_path / "eval_run_m2"
        m2_dir.mkdir()
        _write_minimal_m2_results_csv(m2_dir)

        output_dir = tmp_path / "plots"

        with patch(
            "sys.argv",
            ["replot", str(results_dir), "--output-dir", str(output_dir)],
        ):
            main()

        assert (output_dir / "metric_bars.png").exists()
        assert (output_dir / "robustness" / "joint_input-corruption-a.png").exists()
        assert (output_dir / "robustness" / "joint_latent-corruption-a.png").exists()

    def test_main_accepts_robustness_wrapper_without_results_file(self, tmp_path: Path):
        """An eval_all wrapper path may only exist through its ``_m1`` / ``_m2`` siblings."""
        from unittest.mock import patch

        from scripts.replot_from_eval_results import main

        results_dir = tmp_path / "eval_run"

        m1_dir = tmp_path / "eval_run_m1"
        m1_dir.mkdir()
        _write_minimal_m1_results_csv(m1_dir)

        m2_dir = tmp_path / "eval_run_m2"
        m2_dir.mkdir()
        _write_minimal_m2_results_csv(m2_dir)

        output_dir = tmp_path / "plots"

        with patch(
            "sys.argv",
            ["replot", str(results_dir), "--output-dir", str(output_dir)],
        ):
            main()

        assert (output_dir / "robustness" / "joint_input-corruption-a.png").exists()
        assert (output_dir / "robustness" / "joint_latent-corruption-a.png").exists()
