"""Aggregation and comparison utilities."""

from .category_scores import CategoryScores, compute_category_scores
from .comparison import MetricComparison, compute_normalized_comparisons, extract_normalized_scores
from .results_dataframe import (
    load_results_csv,
    report_to_dataframe,
    save_results_csv,
    session_reports_to_dataframe,
)
from .summary import aggregate_by_category, create_report

__all__ = [
    "CategoryScores",
    "compute_category_scores",
    "MetricComparison",
    "compute_normalized_comparisons",
    "extract_normalized_scores",
    "aggregate_by_category",
    "create_report",
    "session_reports_to_dataframe",
    "report_to_dataframe",
    "save_results_csv",
    "load_results_csv",
]
