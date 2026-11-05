"""Tests for CollatedValidationSegment.strip_original() and downstream consumers.

Verifies that after stripping, aggregation, dataframe creation, and save all
work correctly using original_shape instead of the full tensor.
"""

import numpy as np
import pytest
import torch

from ntt.evaluation.core.types import (
    DataType,
    EvaluationReport,
    MetricCategory,
    MetricDirection,
    MetricLevel,
    MetricResult,
)
from ntt.evaluation.utils.validation_collation import (
    CollatedValidationSegment,
    aggregate_segment_reports,
    aggregate_session_reports,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_segment(
    session_id: str = "brainset/session_1",
    n_units: int = 10,
    n_time: int = 5000,
    segment_index: int = 0,
) -> CollatedValidationSegment:
    return CollatedValidationSegment(
        session_id=session_id,
        original=torch.randn(n_units, n_time),
        reconstructed=torch.randn(n_units, n_time),
        pad_and_gap_mask=None,
        metadata={"sampling_rate": 1000.0},
        segment_index=segment_index,
        is_partial=False,
    )


def _make_report(value: float = 0.85) -> EvaluationReport:
    return EvaluationReport(
        results={
            "spike_detection_f1": MetricResult(
                name="spike_detection_f1",
                category=MetricCategory.TEMPORAL_FIDELITY,
                direction=MetricDirection.HIGHER_IS_BETTER,
                level=MetricLevel.PER_UNIT,
                raw_value=value,
                value_range=(0.0, 1.0),
            ),
        }
    )


def _make_segment_report_pair(
    session_id: str = "brainset/session_1",
    n_units: int = 10,
    n_time: int = 5000,
    segment_index: int = 0,
    metric_value: float = 0.85,
) -> tuple[CollatedValidationSegment, dict]:
    seg = _make_segment(session_id, n_units, n_time, segment_index)
    report = _make_report(metric_value)
    return seg, {"report": report, "metadata": dict(seg.metadata)}


# ---------------------------------------------------------------------------
# CollatedValidationSegment.strip_original
# ---------------------------------------------------------------------------


class TestStripOriginal:
    def test_strip_sets_shape_and_nones_tensor(self):
        seg = _make_segment(n_units=20, n_time=3000)
        assert seg.original is not None
        assert seg.original_shape is None

        seg.strip_original()

        assert seg.original is None
        assert seg.original_shape == (20, 3000)

    def test_strip_idempotent(self):
        seg = _make_segment(n_units=5, n_time=100)
        seg.strip_original()
        shape_after_first = seg.original_shape

        seg.strip_original()  # second call — should be a no-op
        assert seg.original_shape == shape_after_first
        assert seg.original is None

    def test_strip_frees_memory(self):
        seg = _make_segment(n_units=100, n_time=50000)
        # Force a reference to the tensor to check it's released
        tensor_ref = seg.original
        assert tensor_ref is not None

        seg.strip_original()
        # seg does not hold the tensor any more; only our local ref does
        assert seg.original is None
        # original_shape is tiny compared to the tensor
        assert seg.original_shape == (100, 50000)


# ---------------------------------------------------------------------------
# aggregate_segment_reports — works with stripped and unstripped segments
# ---------------------------------------------------------------------------


class TestAggregateSegmentReportsWithStrip:
    def test_aggregate_with_unstripped_segments(self):
        """Baseline: aggregation works with full tensors."""
        pairs = [
            _make_segment_report_pair(metric_value=0.8, n_time=1000),
            _make_segment_report_pair(metric_value=0.9, n_time=2000, segment_index=1),
        ]
        result = aggregate_segment_reports(pairs)
        assert "spike_detection_f1" in result.results

    def test_aggregate_with_stripped_segments(self):
        """After stripping, aggregation still works identically."""
        pairs = [
            _make_segment_report_pair(metric_value=0.8, n_time=1000),
            _make_segment_report_pair(metric_value=0.9, n_time=2000, segment_index=1),
        ]

        # Strip originals
        for seg, _ in pairs:
            seg.strip_original()

        result = aggregate_segment_reports(pairs)
        assert "spike_detection_f1" in result.results

    def test_aggregate_same_result_stripped_vs_unstripped(self):
        """Aggregation produces the same metric value whether stripped or not."""
        values = [0.7, 0.8, 0.9]
        times = [1000, 2000, 3000]

        # Unstripped
        pairs_full = [
            _make_segment_report_pair(metric_value=v, n_time=t, segment_index=i)
            for i, (v, t) in enumerate(zip(values, times, strict=False))
        ]
        result_full = aggregate_segment_reports(pairs_full)

        # Stripped
        pairs_stripped = [
            _make_segment_report_pair(metric_value=v, n_time=t, segment_index=i)
            for i, (v, t) in enumerate(zip(values, times, strict=False))
        ]
        for seg, _ in pairs_stripped:
            seg.strip_original()
        result_stripped = aggregate_segment_reports(pairs_stripped)

        full_val = result_full.results["spike_detection_f1"].raw_value
        stripped_val = result_stripped.results["spike_detection_f1"].raw_value
        assert full_val == pytest.approx(stripped_val)

    def test_aggregate_raises_on_missing_both(self):
        """If both original and original_shape are None, raise RuntimeError."""
        seg, data = _make_segment_report_pair()
        seg.original = None  # manually break it without setting shape
        seg.original_shape = None

        with pytest.raises(RuntimeError, match="neither original tensor nor original_shape"):
            aggregate_segment_reports([(seg, data)])


# ---------------------------------------------------------------------------
# aggregate_session_reports — cross-session aggregation
# ---------------------------------------------------------------------------


class TestAggregateSessionReportsWithStrip:
    def test_session_aggregation_with_stripped_segments(self):
        session_reports = {}
        for sid in ["brainset/s1", "brainset/s2"]:
            pairs = [
                _make_segment_report_pair(session_id=sid, metric_value=0.8, n_time=1000),
                _make_segment_report_pair(
                    session_id=sid, metric_value=0.9, n_time=2000, segment_index=1
                ),
            ]
            for seg, _ in pairs:
                seg.strip_original()
            session_reports[sid] = pairs

        result = aggregate_session_reports(session_reports)
        assert "spike_detection_f1" in result.results

    def test_session_aggregation_same_result_stripped_vs_unstripped(self):
        values = {"brainset/s1": [(0.7, 1000), (0.8, 2000)], "brainset/s2": [(0.9, 1500)]}

        def build(strip: bool):
            reports = {}
            for sid, vt_list in values.items():
                pairs = [
                    _make_segment_report_pair(
                        session_id=sid, metric_value=v, n_time=t, segment_index=i
                    )
                    for i, (v, t) in enumerate(vt_list)
                ]
                if strip:
                    for seg, _ in pairs:
                        seg.strip_original()
                reports[sid] = pairs
            return reports

        result_full = aggregate_session_reports(build(strip=False))
        result_stripped = aggregate_session_reports(build(strip=True))

        full_val = result_full.results["spike_detection_f1"].raw_value
        stripped_val = result_stripped.results["spike_detection_f1"].raw_value
        assert full_val == pytest.approx(stripped_val)


# ---------------------------------------------------------------------------
# results_dataframe — session_reports_to_dataframe
# ---------------------------------------------------------------------------


class TestDataFrameWithStrip:
    def test_dataframe_with_stripped_segments(self):
        from ntt.evaluation.aggregation.results_dataframe import session_reports_to_dataframe

        pairs = [_make_segment_report_pair(n_units=15, n_time=3000)]
        for seg, _ in pairs:
            seg.strip_original()

        df = session_reports_to_dataframe({"brainset/session_1": pairs})
        assert len(df) == 1
        assert df.iloc[0]["n_units"] == 15
        assert df.iloc[0]["n_time_bins"] == 3000

    def test_dataframe_same_result_stripped_vs_unstripped(self):
        from ntt.evaluation.aggregation.results_dataframe import session_reports_to_dataframe

        def build(strip: bool):
            pairs = [
                _make_segment_report_pair(n_units=20, n_time=4000),
                _make_segment_report_pair(n_units=20, n_time=2000, segment_index=1),
            ]
            if strip:
                for seg, _ in pairs:
                    seg.strip_original()
            return {"brainset/session_1": pairs}

        df_full = session_reports_to_dataframe(build(strip=False))
        df_stripped = session_reports_to_dataframe(build(strip=True))

        assert list(df_full["n_units"]) == list(df_stripped["n_units"])
        assert list(df_full["n_time_bins"]) == list(df_stripped["n_time_bins"])

    def test_dataframe_raises_on_missing_both(self):
        from ntt.evaluation.aggregation.results_dataframe import session_reports_to_dataframe

        seg, data = _make_segment_report_pair()
        seg.original = None
        seg.original_shape = None

        with pytest.raises(RuntimeError, match="neither original tensor nor original_shape"):
            session_reports_to_dataframe({"brainset/session_1": [(seg, data)]})
