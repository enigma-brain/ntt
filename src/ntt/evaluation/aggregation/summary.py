"""Aggregate metrics within/across categories."""

import numpy as np

from ..core.types import EvaluationReport, MetricCategory, MetricResult


def aggregate_by_category(results: dict[str, MetricResult]) -> dict[MetricCategory, float]:
    """
    Aggregate raw metric values within each category.

    Uses equal weighting within categories.  Note that raw values may have
    different scales across metrics, so these category scores are rough
    summaries rather than normalised indices.
    """
    category_scores = {}

    for category in MetricCategory:
        category_results = [r for r in results.values() if r.category == category]

        if not category_results:
            continue

        all_values = []
        for r in category_results:
            val = r.raw_value
            if np.isscalar(val):
                if not np.isnan(val):
                    all_values.append(val)
            elif isinstance(val, np.ndarray):
                valid = val[~np.isnan(val)]
                if len(valid) > 0:
                    all_values.append(np.mean(valid))
            else:
                # Non-scalar, non-array raw values (e.g. dict-shaped input_robustness (M1)
                # output) cannot be summarised numerically — skip them in
                # the category roll-up.
                continue

        if all_values:
            category_scores[category] = np.mean(all_values)

    return category_scores


def create_report(
    results: dict[str, MetricResult],
    config: dict | None = None,
) -> EvaluationReport:
    """Create a full evaluation report with category aggregation."""
    category_scores = aggregate_by_category(results)

    return EvaluationReport(
        results=results,
        category_scores=category_scores,
        config=config or {},
    )
