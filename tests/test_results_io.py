"""Tests for evaluation results serialization and DataFrame conversion."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from ntt.core.components.types import DataType
from ntt.evaluation.aggregation.results_dataframe import (
    load_results_csv,
    report_to_dataframe,
    save_results_csv,
    session_reports_to_dataframe,
)
from ntt.evaluation.core.types import (
    EvaluationReport,
    MetricCategory,
    MetricDirection,
    MetricLevel,
    MetricResult,
)

# ---------------------------------------------------------------------------
# MetricResult round-trip
# ---------------------------------------------------------------------------


class TestMetricResultRoundTrip:
    def test_scalar_value(self):
        """Scalar raw_value and all fields survive round-trip."""
        result = MetricResult(
            name="connectivity_f1",
            category=MetricCategory.POPULATION_STRUCTURE,
            direction=MetricDirection.HIGHER_IS_BETTER,
            level=MetricLevel.POPULATION,
            raw_value=0.85,
            normalized_value=0.9,
            metadata={"precision": 0.8, "recall": 0.9},
            value_range=(0.0, 1.0),
            source_data_type=DataType.SPIKES,
        )
        d = result.to_dict()
        restored = MetricResult.from_dict(d)

        assert restored.name == "connectivity_f1"
        assert restored.category == MetricCategory.POPULATION_STRUCTURE
        assert restored.direction == MetricDirection.HIGHER_IS_BETTER
        assert restored.level == MetricLevel.POPULATION
        assert restored.raw_value == 0.85
        assert restored.normalized_value == 0.9
        assert restored.metadata["precision"] == 0.8
        assert restored.value_range == (0.0, 1.0)
        assert restored.source_data_type == DataType.SPIKES

    def test_array_value_and_metadata(self):
        """Array raw_value and metadata with arrays survive round-trip."""
        result = MetricResult(
            name="spike_detection_f1",
            category=MetricCategory.TEMPORAL_FIDELITY,
            direction=MetricDirection.HIGHER_IS_BETTER,
            level=MetricLevel.PER_UNIT,
            raw_value=np.array([0.8, 0.9, 0.7]),
            metadata={
                "precision": np.array([0.85, 0.95, 0.75]),
                "recall": np.array([0.75, 0.85, 0.65]),
            },
        )
        d = result.to_dict()

        # Verify JSON-serializable (no numpy types)
        json.dumps(d)

        restored = MetricResult.from_dict(d)
        np.testing.assert_array_almost_equal(restored.raw_value, [0.8, 0.9, 0.7])
        np.testing.assert_array_almost_equal(restored.metadata["precision"], [0.85, 0.95, 0.75])

    def test_value_range_preserved(self):
        """value_range tuple with None survives round-trip."""
        result = MetricResult(
            name="temporal_precision",
            category=MetricCategory.TEMPORAL_FIDELITY,
            direction=MetricDirection.LOWER_IS_BETTER,
            level=MetricLevel.PER_UNIT,
            raw_value=1.5,
            value_range=(0.0, None),
        )
        d = result.to_dict()
        restored = MetricResult.from_dict(d)
        assert restored.value_range == (0.0, None)

    def test_json_roundtrip(self):
        """Full JSON serialize → deserialize preserves data."""
        result = MetricResult(
            name="acg_correlation",
            category=MetricCategory.SINGLE_UNIT_STATISTICS,
            direction=MetricDirection.HIGHER_IS_BETTER,
            level=MetricLevel.PER_UNIT,
            raw_value=np.array([0.3, -0.1, 0.5]),
            value_range=(-1.0, 1.0),
            source_data_type=DataType.SPIKE_TIMES,
        )
        json_str = json.dumps(result.to_dict())
        restored = MetricResult.from_dict(json.loads(json_str))
        np.testing.assert_array_almost_equal(restored.raw_value, [0.3, -0.1, 0.5])
        assert restored.value_range == (-1.0, 1.0)
        assert restored.source_data_type == DataType.SPIKE_TIMES


# ---------------------------------------------------------------------------
# EvaluationReport round-trip
# ---------------------------------------------------------------------------


class TestEvaluationReportRoundTrip:
    def _make_report(self) -> EvaluationReport:
        return EvaluationReport(
            results={
                "spike_detection_f1": MetricResult(
                    name="spike_detection_f1",
                    category=MetricCategory.TEMPORAL_FIDELITY,
                    direction=MetricDirection.HIGHER_IS_BETTER,
                    level=MetricLevel.PER_UNIT,
                    raw_value=np.array([0.8, 0.9]),
                    value_range=(0.0, 1.0),
                ),
                "acg_correlation": MetricResult(
                    name="acg_correlation",
                    category=MetricCategory.SINGLE_UNIT_STATISTICS,
                    direction=MetricDirection.HIGHER_IS_BETTER,
                    level=MetricLevel.PER_UNIT,
                    raw_value=0.45,
                    value_range=(-1.0, 1.0),
                ),
            },
            category_scores={
                MetricCategory.TEMPORAL_FIDELITY: 0.85,
                MetricCategory.SINGLE_UNIT_STATISTICS: 0.45,
            },
            config={"metric_count": 2},
            skipped_metrics={"mutual_information": "insufficient data"},
            original_data_type=DataType.SPIKES,
            reconstructed_data_type=DataType.SPIKES,
        )

    def test_full_roundtrip(self):
        """All fields survive to_dict → from_dict."""
        report = self._make_report()
        d = report.to_dict()

        # JSON-serializable
        json_str = json.dumps(d)
        restored = EvaluationReport.from_dict(json.loads(json_str))

        assert set(restored.results.keys()) == {"spike_detection_f1", "acg_correlation"}
        np.testing.assert_array_almost_equal(
            restored.results["spike_detection_f1"].raw_value, [0.8, 0.9]
        )
        assert restored.results["acg_correlation"].raw_value == 0.45
        assert restored.category_scores[MetricCategory.TEMPORAL_FIDELITY] == 0.85
        assert restored.skipped_metrics["mutual_information"] == "insufficient data"
        assert restored.original_data_type == DataType.SPIKES
        assert restored.reconstructed_data_type == DataType.SPIKES

    def test_result_without_name_uses_its_key(self):
        """A per-result dict missing name, metadata, value_range and data types loads."""
        d = {
            "results": {
                "some_metric": {
                    "raw_value": 0.75,
                    "normalized_value": None,
                    "category": "temporal_fidelity",
                    "direction": "higher",
                    "level": "per_unit",
                }
            },
            "category_scores": {"temporal_fidelity": 0.75},
            "skipped_metrics": {},
            "config": {},
        }
        report = EvaluationReport.from_dict(d)

        assert report.results["some_metric"].name == "some_metric"
        assert report.results["some_metric"].raw_value == 0.75
        assert report.results["some_metric"].value_range == (None, None)
        assert report.results["some_metric"].metadata == {}
        assert report.results["some_metric"].source_data_type is None
        assert report.original_data_type is None
        assert report.reconstructed_data_type is None


# ---------------------------------------------------------------------------
# DataFrame conversion
# ---------------------------------------------------------------------------


def _make_mock_segment(session_id, n_units=5, n_time_bins=1000, segment_index=0):
    """Create a mock CollatedValidationSegment."""
    import torch

    seg = MagicMock()
    seg.session_id = session_id
    seg.segment_index = segment_index
    seg.is_partial = False
    seg.original = torch.zeros(n_units, n_time_bins)
    seg.original_shape = None  # not stripped yet; code reads original.shape
    seg.reconstructed = torch.zeros(n_units, n_time_bins)
    return seg


def _make_simple_report():
    return EvaluationReport(
        results={
            "spike_detection_f1": MetricResult(
                name="spike_detection_f1",
                category=MetricCategory.TEMPORAL_FIDELITY,
                direction=MetricDirection.HIGHER_IS_BETTER,
                level=MetricLevel.PER_UNIT,
                raw_value=np.array([0.8, 0.9, 0.7]),
                value_range=(0.0, 1.0),
            ),
            "dimensionality_error": MetricResult(
                name="dimensionality_error",
                category=MetricCategory.POPULATION_STRUCTURE,
                direction=MetricDirection.LOWER_IS_BETTER,
                level=MetricLevel.POPULATION,
                raw_value=0.15,
                value_range=(0.0, None),
            ),
        },
        category_scores={
            MetricCategory.TEMPORAL_FIDELITY: 0.8,
            MetricCategory.POPULATION_STRUCTURE: 0.15,
        },
    )


class TestSessionReportsToDataFrame:
    def test_basic_conversion(self):
        """Mock session_reports → DataFrame with correct shape and columns."""
        seg1 = _make_mock_segment("brainset_a/session_1", n_units=5, n_time_bins=1000)
        seg2 = _make_mock_segment("brainset_b/session_2", n_units=10, n_time_bins=2000)

        session_reports = {
            "brainset_a/session_1": [
                (
                    seg1,
                    {
                        "report": _make_simple_report(),
                        "metadata": {
                            "sampling_rate": 1000.0,
                            "subject_id": "M1",
                            "species": "mouse",
                        },
                    },
                )
            ],
            "brainset_b/session_2": [
                (
                    seg2,
                    {
                        "report": _make_simple_report(),
                        "metadata": {
                            "sampling_rate": 1000.0,
                            "subject_id": "M2",
                            "species": "rat",
                        },
                    },
                )
            ],
        }

        df = session_reports_to_dataframe(session_reports)

        assert len(df) == 2
        assert df.iloc[0]["session_id"] == "brainset_a/session_1"
        assert df.iloc[0]["brainset"] == "brainset_a"
        assert df.iloc[0]["session_name"] == "session_1"
        assert df.iloc[0]["n_units"] == 5
        assert df.iloc[0]["n_time_bins"] == 1000
        assert np.isclose(df.iloc[0]["duration_sec"], 1.0)
        assert df.iloc[0]["subject_id"] == "M1"
        assert df.iloc[0]["species"] == "mouse"

        # Metric columns
        assert np.isclose(df.iloc[0]["spike_detection_f1"], 0.8)  # mean of [0.8, 0.9, 0.7]
        assert np.isclose(df.iloc[0]["spike_detection_f1_std"], np.std([0.8, 0.9, 0.7]))
        assert np.isclose(df.iloc[0]["dimensionality_error"], 0.15)
        assert np.isnan(df.iloc[0]["dimensionality_error_std"])  # scalar metric

        # Category columns
        assert np.isclose(df.iloc[0]["category_temporal_fidelity"], 0.8)


class TestReportToDataFrame:
    def test_single_report(self):
        """Single report → one-row DataFrame."""
        report = _make_simple_report()
        df = report_to_dataframe(report, session_id="test_session")

        assert len(df) == 1
        assert df.iloc[0]["session_id"] == "test_session"
        assert np.isclose(df.iloc[0]["spike_detection_f1"], 0.8)
        assert np.isclose(df.iloc[0]["dimensionality_error"], 0.15)


class TestNormalizedComparisonsToDataFrame:
    def test_basic_conversion(self):
        """Normalized comparisons produce a tidy DataFrame with correct columns."""
        from ntt.evaluation.aggregation.comparison import MetricComparison
        from ntt.evaluation.aggregation.results_dataframe import (
            normalized_comparisons_to_dataframe,
        )

        comparisons = {
            "acg_correlation": MetricComparison(
                name="acg_correlation",
                category=MetricCategory.SINGLE_UNIT_STATISTICS,
                direction=MetricDirection.HIGHER_IS_BETTER,
                level=MetricLevel.POPULATION,
                model_value=0.5,
                baseline_value=0.0,
                ceiling_value=1.0,
                normalized_score=0.5,
            ),
            "isi_divergence": MetricComparison(
                name="isi_divergence",
                category=MetricCategory.SINGLE_UNIT_STATISTICS,
                direction=MetricDirection.LOWER_IS_BETTER,
                level=MetricLevel.PER_UNIT,
                model_value=0.3,
                baseline_value=1.0,
                ceiling_value=0.0,
                normalized_score=0.7,
            ),
        }

        df = normalized_comparisons_to_dataframe(comparisons)

        assert len(df) == 2
        assert set(df.columns) >= {
            "metric",
            "category",
            "direction",
            "model_value",
            "baseline_value",
            "ceiling_value",
            "normalized_score",
        }
        acg_row = df[df["metric"] == "acg_correlation"].iloc[0]
        assert acg_row["normalized_score"] == 0.5
        assert acg_row["model_value"] == 0.5
        isi_row = df[df["metric"] == "isi_divergence"].iloc[0]
        assert isi_row["normalized_score"] == 0.7
        assert isi_row["direction"] == "lower"


class TestNormalizedScoresCSVRoundTrip:
    def test_save_and_reload(self, tmp_path: Path):
        """normalized_scores.csv round-trips through save/load."""
        from ntt.evaluation.aggregation.comparison import MetricComparison
        from ntt.evaluation.aggregation.results_dataframe import (
            normalized_comparisons_to_dataframe,
            save_results_csv,
        )

        comparisons = {
            "acg_correlation": MetricComparison(
                name="acg_correlation",
                category=MetricCategory.SINGLE_UNIT_STATISTICS,
                direction=MetricDirection.HIGHER_IS_BETTER,
                level=MetricLevel.POPULATION,
                model_value=0.5,
                baseline_value=0.0,
                ceiling_value=1.0,
                normalized_score=0.5,
            ),
        }
        df = normalized_comparisons_to_dataframe(comparisons)
        csv_path = tmp_path / "normalized_scores.csv"
        save_results_csv(df, csv_path)

        assert csv_path.exists()
        loaded = load_results_csv(csv_path)
        assert len(loaded) == 1
        assert loaded.iloc[0]["metric"] == "acg_correlation"
        assert np.isclose(loaded.iloc[0]["normalized_score"], 0.5)
        assert loaded.iloc[0]["status"] == "ok"


class TestCSVRoundTrip:
    def test_save_load(self):
        """Write CSV → read CSV → values match."""
        report = _make_simple_report()
        df = report_to_dataframe(report, session_id="test")

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "results.csv"
            save_results_csv(df, csv_path)

            assert csv_path.exists()

            loaded = load_results_csv(csv_path)
            assert len(loaded) == 1
            assert loaded.iloc[0]["session_id"] == "test"
            assert np.isclose(loaded.iloc[0]["spike_detection_f1"], 0.8)
