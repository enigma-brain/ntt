"""Orchestrating module for ALL evaluation plots.

Provides ``generate_all_plots()`` — a single entry-point that generates every
plot the evaluation pipeline produces (radar, metric bars, dumbbell, per-unit
boxplots, structure preservation, and behavioral decoding plots).

Both the main eval script and the replot script call this function so that
plotting logic is not duplicated.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .aggregation.comparison import MetricComparison
from .aggregation.summary import create_report
from .behavioral_plotting import generate_all_behavioral_plots
from .core.types import EvaluationReport, MetricResult
from .visualization import (
    plot_metric_bars,
    plot_metric_comparison,
    plot_metric_dumbbell,
    plot_normalized_comparison_radar,
    plot_per_unit_structure_preservation,
    plot_repeats_snr,
    plot_structure_preservation,
)

logger = logging.getLogger(__name__)


def _average_report(reports: list[EvaluationReport]) -> EvaluationReport:
    """Average raw values across reports into a single summary report."""
    accum: dict[str, list[float]] = defaultdict(list)
    templates: dict[str, MetricResult] = {}
    for r in reports:
        for name, result in r.results.items():
            v = result.raw_value
            if isinstance(v, np.ndarray):
                v = float(np.nanmean(v))
            if not np.isnan(v):
                accum[name].append(v)
                templates[name] = result
    averaged_results = {}
    for name, vals in accum.items():
        averaged_results[name] = replace(
            templates[name],
            raw_value=float(np.mean(vals)),
            metadata={"n_segments": len(vals), "std": float(np.std(vals))},
        )
    return create_report(averaged_results)


def _collect_per_segment_reports(
    session_reports: dict[str, list[tuple[Any, dict[str, Any]]]],
) -> list[EvaluationReport]:
    """Flatten session_reports into a list of per-segment EvaluationReport objects."""
    reports = []
    for _session_id, segments in session_reports.items():
        for _segment, payload in segments:
            reports.append(payload["report"])
    return reports


def _find_first_valid_result(
    reports: list[EvaluationReport],
    metric_key: str,
) -> MetricResult | None:
    """Return the first MetricResult for *metric_key* that has valid data."""
    for report in reports:
        if metric_key in report.results:
            candidate = report.results[metric_key]
            v = candidate.raw_value
            if isinstance(v, np.ndarray):
                if np.all(np.isnan(v)):
                    continue
            elif np.isnan(v):
                continue
            return candidate
    return None


def generate_all_plots(
    tokenizer_session_reports: dict[str, dict[str, list[tuple[Any, dict[str, Any]]]]],
    output_dir: str | Path,
    *,
    tokenizer_names_for_comparison: tuple[str, str] | None = None,
    normalized_comparisons: dict[str, MetricComparison] | None = None,
    dpi: int = 150,
) -> None:
    """Generate ALL evaluation plots and save to disk.

    This is the single orchestrating function that both the main evaluation
    script and the replot script call.  It covers:

    - Radar chart of normalized comparison scores (if *normalized_comparisons*
      is provided, or if exactly two tokenizers are given and a ceiling can
      be inferred).
    - Grouped bar chart of aggregated metric values.
    - Dumbbell plot comparing two tokenizers (if two are available).
    - Per-unit boxplots for array-valued metrics (first segment).
    - Structure preservation plots (population and per-unit) for each tokenizer.
    - All behavioral decoding plots (regression, classification, per-class,
      trajectories) via ``generate_all_behavioral_plots``.  Those are written
      per session under ``{output_dir}/sessions/{session_id}/bhav/`` (bar charts,
      per-class plots, summary text) and trajectories under
      ``{output_dir}/sessions/{session_id}/bhav/trajectories/``.

    Args:
        tokenizer_session_reports: ``{tokenizer_display_name: session_reports}``
            where ``session_reports`` is the standard
            ``{session_id: [(segment, {"report": EvaluationReport}), ...]}`` dict.
        output_dir: Root directory for this plot bundle (see behavioral paths above;
            other figures may sit directly under *output_dir* or under *sessions/*).
        tokenizer_names_for_comparison: Optional ``(model_name, baseline_name)``
            identifying which two tokenizers to compare in the dumbbell and
            radar plots.  If ``None`` and exactly two tokenizers are present,
            the first is treated as the model and the second as the baseline.
        normalized_comparisons: Pre-computed normalized comparison scores for
            the radar chart.  When ``None``, the radar chart is skipped (it
            requires a perfect-reconstruction ceiling that cannot be derived
            from saved JSON results alone).
        dpi: Resolution for saved figures.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tok_names = list(tokenizer_session_reports.keys())

    # --- Aggregate per-segment reports for each tokenizer ---
    per_seg_reports: dict[str, list[EvaluationReport]] = {}
    avg_reports: dict[str, EvaluationReport] = {}
    for tok_name, session_reports in tokenizer_session_reports.items():
        seg_reports = _collect_per_segment_reports(session_reports)
        per_seg_reports[tok_name] = seg_reports
        if seg_reports:
            avg_reports[tok_name] = _average_report(seg_reports)

    # Determine model / baseline names for two-tokenizer comparison plots
    if tokenizer_names_for_comparison is not None:
        model_name, baseline_name = tokenizer_names_for_comparison
    elif len(tok_names) == 2:
        model_name, baseline_name = tok_names[0], tok_names[1]
    else:
        model_name, baseline_name = None, None

    # ---- 1. Radar chart (normalized comparison) ----
    if normalized_comparisons is not None and model_name and baseline_name:
        fig_radar = plot_normalized_comparison_radar(
            normalized_comparisons,
            model_name=model_name,
            baseline_name=baseline_name,
        )
        radar_path = output_dir / "normalized_comparison_radar_cs.png"
        fig_radar.savefig(radar_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig_radar)
        logger.debug("Saved: %s", radar_path)

    # ---- 2. Grouped bar chart of raw values ----
    if avg_reports:
        fig_bars = plot_metric_bars(avg_reports)
        bars_path = output_dir / "metric_bars.png"
        fig_bars.savefig(bars_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig_bars)
        logger.debug("Saved: %s", bars_path)

    # ---- 3. Dumbbell plot ----
    if model_name and baseline_name and model_name in avg_reports and baseline_name in avg_reports:
        fig_dumbbell = plot_metric_dumbbell(
            avg_reports[model_name],
            avg_reports[baseline_name],
            model_name=model_name,
            baseline_name=baseline_name,
        )
        dumbbell_path = output_dir / "metric_dumbbell_cs.png"
        fig_dumbbell.savefig(dumbbell_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig_dumbbell)
        logger.debug("Saved: %s", dumbbell_path)

    # ---- 4. Per-unit boxplots (first segment of each tokenizer) ----
    first_seg_by_tok: dict[str, EvaluationReport] = {}
    for tok_name, seg_reports in per_seg_reports.items():
        if seg_reports:
            first_seg_by_tok[tok_name] = seg_reports[0]

    if first_seg_by_tok:
        # Determine which metrics are array-valued from any available first segment
        any_first = next(iter(first_seg_by_tok.values()))
        per_unit_metrics = [
            name
            for name, r in any_first.results.items()
            if isinstance(r.raw_value, np.ndarray) and r.raw_value.ndim >= 1
        ]
        for metric_name in per_unit_metrics:
            fig_box = plot_metric_comparison(first_seg_by_tok, metric_name)
            box_path = output_dir / f"boxplot_{metric_name}.png"
            fig_box.savefig(box_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig_box)
            logger.debug("Saved: %s", box_path)

    # ---- 5. Structure preservation plots (per tokenizer) ----
    for tok_name, seg_reports in per_seg_reports.items():
        # Filename suffix: use short name for multi-tokenizer
        suffix = f"_{tok_name.lower().replace(' ', '_')}" if len(tok_names) > 1 else ""

        # Population structure preservation
        sp_result = _find_first_valid_result(seg_reports, "population_structure_preservation")
        if sp_result is not None:
            fig_sp = plot_structure_preservation(sp_result)
            sp_path = output_dir / f"structure_preservation{suffix}.png"
            fig_sp.savefig(sp_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig_sp)
            logger.debug("Saved: %s", sp_path)
            logger.debug(
                "%s pop RSA r=%.4f, kNN mean=%.4f",
                tok_name,
                sp_result.raw_value,
                sp_result.metadata.get("knn_mean", float("nan")),
            )
        else:
            logger.debug("%s population structure preservation: not available", tok_name)

        # Per-unit structure preservation
        pu_result = _find_first_valid_result(seg_reports, "per_unit_structure_preservation")
        if pu_result is not None:
            fig_pu = plot_per_unit_structure_preservation(pu_result)
            pu_path = output_dir / f"per_unit_structure_preservation{suffix}.png"
            fig_pu.savefig(pu_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig_pu)
            logger.debug("Saved: %s", pu_path)
            logger.debug("%s inter-unit RSA r=%.4f", tok_name, pu_result.raw_value)
        else:
            logger.debug("%s per-unit structure preservation: not available", tok_name)

        # Repeats-based SNR (first segment with a valid result)
        snr_result = _find_first_valid_result(seg_reports, "repeats_snr")
        if snr_result is not None:
            fig_snr = plot_repeats_snr(snr_result)
            snr_path = output_dir / f"repeats_snr{suffix}.png"
            fig_snr.savefig(snr_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig_snr)
            logger.debug("Saved: %s", snr_path)
            logger.debug("%s repeats_snr=%.4f", tok_name, snr_result.raw_value)

    # ---- 6. Behavioral plots ----
    generate_all_behavioral_plots(
        tokenizer_session_reports,
        output_dir=output_dir,
        dpi=dpi,
    )
