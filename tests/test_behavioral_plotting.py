"""Smoke tests for behavioral plotting utilities."""

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from ntt.evaluation.behavioral_plotting import (
    _filter_classification_by_balance,
    extract_behavioral_results,
    plot_classification_comparison,
    plot_per_class_accuracy,
    plot_regression_comparison,
    plot_trajectory_2d,
    plot_trajectory_3d,
    write_behavioral_summary_table,
)
from ntt.evaluation.core.types import (
    MetricCategory,
    MetricDirection,
    MetricLevel,
    MetricResult,
)


def _make_regression_result(display_name: str, value: float = 0.72) -> MetricResult:
    return MetricResult(
        name="behavioral_regression",
        category=MetricCategory.DOWNSTREAM_DECODING,
        direction=MetricDirection.HIGHER_IS_BETTER,
        level=MetricLevel.POPULATION,
        raw_value=value,
        metadata={
            "target_field": display_name,
            "include_phases": ["reach_period"],
        },
    )


def _make_temporal_decoder_result(display_name: str, value: float = 0.68) -> MetricResult:
    return MetricResult(
        name="behavioral_temporal_decoder",
        category=MetricCategory.DOWNSTREAM_DECODING,
        direction=MetricDirection.HIGHER_IS_BETTER,
        level=MetricLevel.POPULATION,
        raw_value=value,
        metadata={
            "target_field": display_name,
            "include_phases": ["reach_period"],
            "n_samples_train": 100,
            "n_samples_val": 20,
            "n_samples_test": 20,
        },
    )


def _make_classification_result(
    display_name: str,
    value: float = 0.65,
    balance_classes: bool = True,
    per_class_accuracy: dict | None = None,
    val_class_counts: dict | None = None,
) -> MetricResult:
    metadata: dict = {
        "target_field": display_name,
        "chance_accuracy": 0.2,
        "balance_classes": balance_classes,
    }
    if per_class_accuracy is not None:
        metadata["per_class_accuracy"] = per_class_accuracy
    if val_class_counts is not None:
        metadata["val_class_counts"] = val_class_counts
    return MetricResult(
        name="behavioral_classification",
        category=MetricCategory.DOWNSTREAM_DECODING,
        direction=MetricDirection.HIGHER_IS_BETTER,
        level=MetricLevel.POPULATION,
        raw_value=value,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


def test_plot_regression_comparison_single_session() -> None:
    data = {
        "tokenizer_A": {
            "behavioral_regression_cursor_pos": _make_regression_result("cursor_pos", 0.72),
            "behavioral_regression_cursor_vel": _make_regression_result("cursor_vel", 0.55),
        },
        "tokenizer_B": {
            "behavioral_regression_cursor_pos": _make_regression_result("cursor_pos", 0.61),
            "behavioral_regression_cursor_vel": _make_regression_result("cursor_vel", 0.48),
        },
    }
    fig = plot_regression_comparison(data)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 1
    plt.close(fig)


def test_plot_regression_comparison_multi_session() -> None:
    session_data = {}
    for session_id in ("session_01", "session_02"):
        session_data[session_id] = {
            "tokenizer_A": {
                "behavioral_regression_cursor_pos": _make_regression_result("cursor_pos", 0.7),
            },
            "tokenizer_B": {
                "behavioral_regression_cursor_pos": _make_regression_result("cursor_pos", 0.6),
            },
        }
    fig = plot_regression_comparison(session_data)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 2
    plt.close(fig)


def test_plot_regression_comparison_includes_temporal_decoder() -> None:
    data = {
        "tokenizer_A": {
            "behavioral_temporal_decoder_cursor_pos": _make_temporal_decoder_result(
                "cursor_pos", 0.74
            ),
        },
        "tokenizer_B": {
            "behavioral_temporal_decoder_cursor_pos": _make_temporal_decoder_result(
                "cursor_pos", 0.63
            ),
        },
    }
    fig = plot_regression_comparison(data)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 1
    plt.close(fig)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def test_plot_classification_comparison() -> None:
    data = {
        "tokenizer_A": {
            "behavioral_classification_target_id": _make_classification_result("target_id", 0.65),
        },
        "tokenizer_B": {
            "behavioral_classification_target_id": _make_classification_result("target_id", 0.55),
        },
    }
    fig = plot_classification_comparison(data)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) >= 1
    plt.close(fig)


def test_comparison_xtick_labels_are_horizontal_and_wrapped() -> None:
    long_target = "very_long_behavioral_target_name_that_should_wrap"

    regression_data = {
        "tokenizer_A": {
            f"behavioral_regression_{long_target}": _make_regression_result(long_target, 0.72),
        },
        "tokenizer_B": {
            f"behavioral_regression_{long_target}": _make_regression_result(long_target, 0.61),
        },
    }
    reg_fig = plot_regression_comparison(regression_data)
    reg_labels = reg_fig.axes[0].get_xticklabels()
    assert reg_labels
    assert reg_labels[0].get_rotation() == 0
    assert "\n" in reg_labels[0].get_text()
    plt.close(reg_fig)

    classification_data = {
        "tokenizer_A": {
            f"behavioral_classification_{long_target}": _make_classification_result(
                long_target, 0.65
            ),
        },
        "tokenizer_B": {
            f"behavioral_classification_{long_target}": _make_classification_result(
                long_target, 0.55
            ),
        },
    }
    cls_fig = plot_classification_comparison(classification_data)
    cls_labels = cls_fig.axes[0].get_xticklabels()
    assert cls_labels
    assert cls_labels[0].get_rotation() == 0
    assert "\n" in cls_labels[0].get_text()
    plt.close(cls_fig)


# ---------------------------------------------------------------------------
# Per-class accuracy
# ---------------------------------------------------------------------------


def test_plot_per_class_accuracy() -> None:
    per_class = {0: 0.8, 1: 0.6, 2: 0.9}
    val_counts = {0: 50, 1: 48, 2: 52}
    data = {
        "tokenizer_A": {
            "behavioral_classification_target_id": _make_classification_result(
                "target_id",
                value=0.77,
                balance_classes=True,
                per_class_accuracy=per_class,
                val_class_counts=val_counts,
            ),
        },
        "tokenizer_B": {
            "behavioral_classification_target_id": _make_classification_result(
                "target_id",
                value=0.65,
                balance_classes=True,
                per_class_accuracy={0: 0.7, 1: 0.5, 2: 0.85},
                val_class_counts={0: 50, 1: 48, 2: 52},
            ),
        },
    }
    fig = plot_per_class_accuracy(data, metric_display_name="behavioral_classification_target_id")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_per_class_accuracy_skips_unbalanced() -> None:
    data = {
        "tokenizer_A": {
            "behavioral_classification_target_id": _make_classification_result(
                "target_id",
                value=0.65,
                balance_classes=False,
                per_class_accuracy={0: 0.8, 1: 0.6},
                val_class_counts={0: 50, 1: 48},
            ),
        },
    }
    result = plot_per_class_accuracy(
        data, metric_display_name="behavioral_classification_target_id"
    )
    assert result is None


# ---------------------------------------------------------------------------
# Trajectory plots
# ---------------------------------------------------------------------------


def _make_regression_result_with_trajectories(target_field="cursor_pos"):
    """Helper to create MetricResult with sample_trials for trajectory tests."""
    rng = np.random.default_rng(42)
    sample_trials = []
    for trial_id in range(3):
        n_pts = 15
        t = np.linspace(0, 1, n_pts)
        y_true = np.column_stack([np.sin(t + trial_id), np.cos(t + trial_id)])
        y_pred = y_true + rng.normal(0, 0.1, y_true.shape)
        sample_trials.append(
            {
                "trial_id": trial_id,
                "timestamps": t.tolist(),
                "y_true": y_true.tolist(),
                "y_pred": y_pred.tolist(),
            }
        )
    return MetricResult(
        name="behavioral_regression",
        category=MetricCategory.DOWNSTREAM_DECODING,
        direction=MetricDirection.HIGHER_IS_BETTER,
        level=MetricLevel.POPULATION,
        raw_value=0.72,
        metadata={"target_field": target_field, "sample_trials": sample_trials},
    )


def test_plot_trajectory_2d():
    result = _make_regression_result_with_trajectories()
    fig = plot_trajectory_2d(result)
    assert isinstance(fig, plt.Figure)
    assert len(fig.get_axes()) == 3
    plt.close(fig)


def test_plot_trajectory_2d_with_connecting_lines():
    result = _make_regression_result_with_trajectories()
    fig = plot_trajectory_2d(result, connect_gt_pred=True)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_trajectory_2d_no_trials():
    """Returns None when no sample_trials."""
    result = MetricResult(
        name="behavioral_regression",
        category=MetricCategory.DOWNSTREAM_DECODING,
        direction=MetricDirection.HIGHER_IS_BETTER,
        level=MetricLevel.POPULATION,
        raw_value=0.72,
        metadata={"target_field": "cursor_pos"},
    )
    assert plot_trajectory_2d(result) is None


def test_plot_trajectory_3d():
    result = _make_regression_result_with_trajectories()
    fig = plot_trajectory_3d(result)
    assert isinstance(fig, plt.Figure)
    assert len(fig.get_axes()) == 3
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary table and extraction helper
# ---------------------------------------------------------------------------


def test_write_behavioral_summary_table():
    """Summary table writes a text file with tokenizer columns and metric rows."""
    from ntt.evaluation.core.types import (
        MetricCategory,
        MetricDirection,
        MetricLevel,
        MetricResult,
    )

    results_by_tokenizer = {
        "CS Tokenizer": {
            "behavioral_regression_cursor_pos": MetricResult(
                name="behavioral_regression",
                category=MetricCategory.DOWNSTREAM_DECODING,
                direction=MetricDirection.HIGHER_IS_BETTER,
                level=MetricLevel.POPULATION,
                raw_value=0.72,
                metadata={"target_field": "cursor_pos"},
            ),
            "behavioral_classification_trial_target_id": MetricResult(
                name="behavioral_classification",
                category=MetricCategory.DOWNSTREAM_DECODING,
                direction=MetricDirection.HIGHER_IS_BETTER,
                level=MetricLevel.POPULATION,
                raw_value=0.85,
                metadata={"target_field": "trial_target_id", "chance_accuracy": 0.125},
            ),
            "behavioral_temporal_decoder_cursor_vel": _make_temporal_decoder_result(
                "cursor_vel", 0.66
            ),
        },
        "Poisson Baseline": {
            "behavioral_regression_cursor_pos": MetricResult(
                name="behavioral_regression",
                category=MetricCategory.DOWNSTREAM_DECODING,
                direction=MetricDirection.HIGHER_IS_BETTER,
                level=MetricLevel.POPULATION,
                raw_value=0.45,
                metadata={"target_field": "cursor_pos"},
            ),
        },
    }
    out_path = Path(__file__).parent / "_test_behavioral_summary.txt"
    try:
        write_behavioral_summary_table(results_by_tokenizer, out_path)
        assert out_path.exists()
        content = out_path.read_text()
        assert "CS Tokenizer" in content
        assert "Poisson Baseline" in content
        assert "0.7200" in content
        assert "chance=0.125" in content
        assert "0.6600" in content
        assert "N/A" in content  # Poisson doesn't have classification
    finally:
        if out_path.exists():
            out_path.unlink()


# ---------------------------------------------------------------------------
# generate_all_behavioral_plots — metric selection across tokenizers
# ---------------------------------------------------------------------------


def _single_session_reports(results: dict) -> dict:
    """Wrap a {display_name: MetricResult} dict as session_reports for one session."""
    from unittest.mock import MagicMock

    from ntt.evaluation.core.types import EvaluationReport

    report = EvaluationReport(results=results)
    return {"session_X": [(MagicMock(), {"report": report})]}


def test_generate_all_plots_includes_metric_absent_from_first_tokenizer(tmp_path) -> None:
    """A target produced only by a non-first tokenizer must still be plotted.

    The metric-selection lists must not be derived from the first tokenizer's
    results only: any target the first tokenizer skipped (e.g. a baseline-only
    decoder, or a metric that raised and was dropped for one tokenizer) would be
    silently never plotted for *any* tokenizer.
    """
    from ntt.evaluation.behavioral_plotting import generate_all_behavioral_plots

    tokenizer_session_reports = {
        # iterated first; lacks cursor_vel
        "tokenizer_A": _single_session_reports(
            {"behavioral_regression_cursor_pos": _make_regression_result("cursor_pos", 0.70)},
        ),
        # only this tokenizer has cursor_vel
        "tokenizer_B": _single_session_reports(
            {
                "behavioral_regression_cursor_pos": _make_regression_result("cursor_pos", 0.60),
                "behavioral_regression_cursor_vel": _make_regression_result("cursor_vel", 0.55),
            },
        ),
    }

    generate_all_behavioral_plots(tokenizer_session_reports, output_dir=tmp_path)

    cursor_vel_plots = list((tmp_path / "sessions").rglob("*cursor_vel*"))
    assert cursor_vel_plots, (
        "cursor_vel comparison plot must be produced even though the first tokenizer lacks it"
    )


def test_generate_all_plots_handles_no_tokenizers(tmp_path) -> None:
    """Empty input returns cleanly instead of raising StopIteration."""
    from ntt.evaluation.behavioral_plotting import generate_all_behavioral_plots

    generate_all_behavioral_plots({}, output_dir=tmp_path)  # must not raise


def test_extract_behavioral_results():
    """Extract behavioral results from session reports format."""
    from unittest.mock import MagicMock

    from ntt.evaluation.core.types import (
        EvaluationReport,
        MetricCategory,
        MetricDirection,
        MetricLevel,
        MetricResult,
    )

    reg_result = MetricResult(
        name="behavioral_regression",
        category=MetricCategory.DOWNSTREAM_DECODING,
        direction=MetricDirection.HIGHER_IS_BETTER,
        level=MetricLevel.POPULATION,
        raw_value=0.72,
        metadata={"target_field": "cursor_pos"},
    )
    temporal_result = _make_temporal_decoder_result("cursor_vel", 0.64)
    spike_result = MetricResult(
        name="spike_detection_f1",
        category=MetricCategory.TEMPORAL_FIDELITY,
        direction=MetricDirection.HIGHER_IS_BETTER,
        level=MetricLevel.POPULATION,
        raw_value=0.95,
        metadata={},
    )
    report = EvaluationReport(
        results={
            "behavioral_regression_cursor_pos": reg_result,
            "behavioral_temporal_decoder_cursor_vel": temporal_result,
            "spike_detection_f1": spike_result,
        }
    )

    segment = MagicMock()
    session_reports = {
        "session_A": [(segment, {"report": report})],
    }

    flat, per_session = extract_behavioral_results(session_reports)
    assert "behavioral_regression_cursor_pos" in flat
    assert "behavioral_temporal_decoder_cursor_vel" in flat
    assert "spike_detection_f1" not in flat
    assert "session_A" in per_session
    assert "behavioral_regression_cursor_pos" in per_session["session_A"]
    assert "behavioral_temporal_decoder_cursor_vel" in per_session["session_A"]
