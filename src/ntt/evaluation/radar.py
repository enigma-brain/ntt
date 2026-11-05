"""Aggregated radar plots for tokenizers, baselined against a Poisson reference.

The flow is:

1. ``NTTEvalJob`` writes per-tokenizer ``results.json`` files containing
   ``aggregated``, ``poisson_baseline``, and ``perfect_ceiling`` keys.

2. ``RadarBundle.from_local_dir(root)`` loads that tree, deserializing each
   report through ``ntt.evaluation.core.types.EvaluationReport.from_dict`` so
   category strings are mapped onto the current paper-aligned vocabulary.

3. ``inject_scalar_metrics`` attaches compression/generalization scalars from
   the raw results.json onto the bundle's reports, making them available for
   normalization and plotting just like any eval-report metric.

4. Per-tokenizer normalized scores come from
   ``ntt.evaluation.aggregation.comparison.compute_normalized_comparisons``
   so direction-aware sign handling and the ``(model - floor) / (ceiling - floor)``
   formula stay consistent with ntt's own ``plot_normalized_comparison_radar``.

5. ``CategorySpec`` lets callers either lean on the ntt ``MetricCategory`` enum
   (``CategorySpec.from_metric_categories(report)``) or define arbitrary buckets
   that average a hand-picked metric list (``{"Single-unit fidelity":
   ["acg_correlation", "isi_kl_divergence"]}``). Metrics not present in a given
   tokenizer are skipped with a warning.

6. ``plot_radar`` overlays one polygon per tokenizer using a matplotlib polar
   axis. ``0`` is the Poisson floor and ``1`` is the perfect-reconstruction
   ceiling, matching ntt's plotting convention.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import RegularPolygon
from matplotlib.path import Path as MplPath
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

from ntt.evaluation.aggregation.comparison import (
    ALTERNATIVE_FLOORS,
    MetricComparison,
    compute_normalized_comparisons,
)
from ntt.evaluation.core.types import (
    EvaluationReport,
    MetricCategory,
    MetricDirection,
    MetricLevel,
    MetricResult,
)

LOGGER = logging.getLogger(__name__)


def _load_report_dict(d: Mapping[str, Any]) -> EvaluationReport:
    """Deserialize a serialized EvaluationReport dict."""
    return EvaluationReport.from_dict(dict(d))


@dataclass
class CategorySpec:
    """Maps a display label to a list of metric names whose normalized scores get averaged."""

    categories: dict[str, list[str]]

    def labels(self) -> list[str]:
        return list(self.categories.keys())

    @classmethod
    def from_metric_categories(
        cls,
        report: EvaluationReport,
        only: list[MetricCategory] | None = None,
    ) -> CategorySpec:
        """Build a spec by grouping ``report``'s metric names by their declared category.

        ``only`` optionally restricts to a subset of MetricCategory values.
        """
        buckets: dict[str, list[str]] = {}
        for name, result in report.results.items():
            cat = result.category
            if only is not None and cat not in only:
                continue
            buckets.setdefault(cat.value, []).append(name)
        return cls(categories={k: sorted(v) for k, v in sorted(buckets.items())})


@dataclass
class RadarBundle:
    """All the reports needed to draw a multi-tokenizer radar."""

    baseline: EvaluationReport
    ceiling: EvaluationReport
    tokenizers: dict[str, EvaluationReport] = field(default_factory=dict)
    paths: dict[str, Path] = field(default_factory=dict)

    @classmethod
    def from_local_dir(
        cls,
        root: Path | str,
        *,
        tokenizers: Iterable[str] | None = None,
        reference_path: Path | None = None,
    ) -> RadarBundle:
        """Load a bundle from a tree of per-tokenizer results.json files.

        Expected layout::

            root/
              <tokenizer_name>/results.json
              _reference/poisson_baseline.json   # optional override

        Each ``results.json`` must contain ``aggregated``, ``poisson_baseline``,
        and ``perfect_ceiling`` (all written by NTTEvalJob).
        """
        root = Path(root)
        names: list[str] = []
        candidate_paths: dict[str, Path] = {}
        for child in sorted(root.iterdir()):
            if not child.is_dir() or child.name.startswith("_"):
                continue
            if (child / "results.json").exists():
                names.append(child.name)
                candidate_paths[child.name] = child
        if tokenizers is not None:
            wanted = set(tokenizers)
            names = [n for n in names if n in wanted]

        tokenizer_reports: dict[str, EvaluationReport] = {}
        baselines: list[EvaluationReport] = []
        ceilings: list[EvaluationReport] = []
        for name in names:
            payload = json.loads((candidate_paths[name] / "results.json").read_text())
            tokenizer_reports[name] = _load_report_dict(payload["aggregated"])
            baselines.append(_load_report_dict(payload["poisson_baseline"]))
            ceilings.append(_load_report_dict(payload["perfect_ceiling"]))

        ref = reference_path or (root / "_reference" / "poisson_baseline.json")
        if ref.exists():
            ref_payload = json.loads(ref.read_text())
            aggregated_dict = ref_payload.get("aggregated") or ref_payload
            perfect_dict = ref_payload.get("perfect_aggregated") or ref_payload.get(
                "perfect_ceiling"
            )
            if perfect_dict is None:
                raise ValueError(
                    f"Reference file {ref} contains an 'aggregated' / baseline report "
                    "but no 'perfect_aggregated' or 'perfect_ceiling' for the ceiling."
                )
            baseline = _load_report_dict(aggregated_dict)
            ceiling = _load_report_dict(perfect_dict)
        elif baselines and ceilings:
            if len(baselines) > 1:
                LOGGER.warning(
                    "RadarBundle.from_local_dir: no _reference/poisson_baseline.json found; "
                    "falling back to %s's baseline/ceiling. Other tokenizers may have run "
                    "against different sessions/configs, in which case the radar's floor/ceiling "
                    "rings will not match all polygons. Pass reference_path= or place a "
                    "shared _reference/poisson_baseline.json under root to fix.",
                    sorted(candidate_paths)[0] if candidate_paths else "<unknown>",
                )
            baseline, ceiling = baselines[0], ceilings[0]
        else:
            raise ValueError(f"No tokenizer results.json files found under {root}")

        return cls(
            baseline=baseline,
            ceiling=ceiling,
            tokenizers=tokenizer_reports,
            paths={n: candidate_paths[n] for n in names},
        )


@dataclass(frozen=True)
class ScalarMetric:
    """Scalar drawn from a tokenizer's results.json by dotted-path lookup.

    Attached to the radar's reports as a synthetic MetricResult so it
    participates in CategorySpec / normalized scoring like a native metric.
    """

    name: str
    path: str  # dotted path into results.json, e.g. "compression.compression_rate"
    direction: str = "higher"  # "higher" | "lower"
    floor: float = 0.0
    ceiling: float = 1.0
    level: MetricLevel = MetricLevel.POPULATION
    category: MetricCategory | None = None


# Default scalar metrics extracted from results.json. The compression scalars
# use a placeholder ceiling of 1000 — realistic compression ratios run ~2–50,
# so the normalized values on the radar will sit near 0 unless callers override
# (floor, ceiling) to a tighter range per-figure. Generalization scalars are
# already in [0, 1] by construction.
DEFAULT_EVAL_JOB_SCALARS: tuple[ScalarMetric, ...] = (
    ScalarMetric(
        "compression/rate",
        path="compression.compression_rate",
        direction="higher",
        floor=0.0,
        ceiling=1000.0,  # override per figure; realistic range is ~2–50
    ),
    ScalarMetric(
        "compression/latent_dim_reduction",
        path="compression.latent_dim_reduction",
        direction="higher",
        floor=0.0,
        ceiling=1000.0,  # override per figure; realistic range is ~2–50
    ),
    ScalarMetric(
        "generalization/score",
        path="generalization.summary.generalization_score",
        direction="higher",
        floor=0.0,
        ceiling=1.0,
    ),
    ScalarMetric(
        "generalization/mean_retention",
        path="generalization.summary.mean_retention",
        direction="higher",
        floor=0.0,
        ceiling=1.0,
    ),
)


def _resolve_path(payload: Mapping[str, Any], path: str) -> Any:
    """Resolve a dotted path; return None if any segment is missing."""
    cur: Any = payload
    for part in path.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return None
        cur = cur[part]
    return cur


def inject_scalar_metrics(
    bundle: RadarBundle,
    raw_results: Mapping[str, Mapping[str, Any]],
    metrics: Iterable[ScalarMetric],
) -> None:
    """Attach scalar metrics from each tokenizer's raw results.json onto the bundle's reports.

    For each (tokenizer, metric):
      - Resolve metric.path against the tokenizer's raw payload.
      - Skip with a per-(tokenizer, metric) warning if missing or non-numeric.
      - Attach a synthetic MetricResult to bundle.tokenizers[<name>].
      - Attach matching floor/ceiling MetricResults to bundle.baseline /
        bundle.ceiling (idempotent: first hit wins).
    """
    direction_map = {
        "higher": MetricDirection.HIGHER_IS_BETTER,
        "lower": MetricDirection.LOWER_IS_BETTER,
    }

    for metric in metrics:
        attached_any = False
        for name in bundle.tokenizers:
            payload = raw_results.get(name)
            if payload is None:
                continue
            value = _resolve_path(payload, metric.path)
            if value is None or not isinstance(value, (int, float)):
                LOGGER.warning(
                    "inject_scalar_metrics: skipping %s for %s (path=%s, value=%r)",
                    metric.name,
                    name,
                    metric.path,
                    value,
                )
                continue
            if isinstance(value, float) and math.isnan(value):
                LOGGER.warning(
                    "inject_scalar_metrics: skipping %s for %s (NaN value)",
                    metric.name,
                    name,
                )
                continue
            bundle.tokenizers[name].results[metric.name] = _make_synthetic_result(
                metric,
                raw_value=float(value),
                direction=direction_map[metric.direction],
            )
            attached_any = True
        if attached_any:
            bundle.baseline.results.setdefault(
                metric.name,
                _make_synthetic_result(
                    metric,
                    raw_value=metric.floor,
                    direction=direction_map[metric.direction],
                ),
            )
            bundle.ceiling.results.setdefault(
                metric.name,
                _make_synthetic_result(
                    metric,
                    raw_value=metric.ceiling,
                    direction=direction_map[metric.direction],
                ),
            )


def tokenizers_with_data(
    bundle: RadarBundle,
    metric_names: list[str],
    *,
    require: str = "any",
) -> list[str]:
    """Return tokenizers whose reports carry the given metrics.

    ``require="any"`` (default) keeps a tokenizer if at least one of the
    metrics is present; ``require="all"`` keeps it only when *every* metric is
    present. Useful for picking a sensible default subset before
    ``plot_radar`` — e.g., select tokenizers with encoding/decoding data when
    those axes are on the radar.
    """
    out: list[str] = []
    for tokenizer, report in bundle.tokenizers.items():
        present = {n for n in metric_names if n in report.results}
        if require == "all":
            if len(present) == len(metric_names):
                out.append(tokenizer)
        else:
            if present:
                out.append(tokenizer)
    return out


def _make_synthetic_result(
    metric: ScalarMetric,
    *,
    raw_value: float,
    direction: MetricDirection,
) -> MetricResult:
    return MetricResult(
        name=metric.name,
        category=metric.category or MetricCategory.LATENT_QUALITY,
        direction=direction,
        level=metric.level,
        raw_value=raw_value,
        metadata={"synthetic": True, "source_path": metric.path},
        value_range=(metric.floor, metric.ceiling),
    )


def load_report(path: str | Path) -> EvaluationReport:
    """Load an eval_report.json into an EvaluationReport."""
    with open(path) as f:
        data = json.load(f)
    return EvaluationReport.from_dict(data)


# Catalog source labels we recognize as a slash-prefix in ``CategorySpec``
# entries. When a spec name like ``"recon/acg_correlation"`` doesn't match the
# bundle directly, we strip the leading ``"<known_source>/"`` and retry — so
# the catalog's slash-form selectors can be pasted straight into a spec for
# eval-report metrics.
_KNOWN_CATALOG_SOURCES: tuple[str, ...] = (
    "recon",
    "encoding",
    "latent_input_corruption",
    "latent_latent_corruption",
    "latent_snr",
)


def _resolve_metric_name(
    raw_name: str,
    comparisons: Mapping[str, MetricComparison],
) -> tuple[str, MetricComparison] | None:
    """Look ``raw_name`` up in ``comparisons``; fall back to stripping a known source prefix."""
    cmp_ = comparisons.get(raw_name)
    if cmp_ is not None:
        return raw_name, cmp_
    if "/" not in raw_name:
        return None
    prefix, _, suffix = raw_name.partition("/")
    if prefix not in _KNOWN_CATALOG_SOURCES:
        return None
    cmp_ = comparisons.get(suffix)
    if cmp_ is None:
        return None
    return suffix, cmp_


def category_scores_for(
    model: EvaluationReport,
    baseline: EvaluationReport,
    ceiling: EvaluationReport,
    spec: CategorySpec,
    *,
    aggregator: str = "mean",
    normalize: bool = True,
) -> dict[str, float]:
    """Compute per-category scores for one tokenizer.

    ``aggregator`` is ``"mean"`` (equal-weighted) or ``"median"``. Categories
    with no usable metrics return ``np.nan``.

    ``normalize=True`` (default) returns Poisson-normalized scores: ``0`` is
    the Poisson floor and ``1`` is the perfect-reconstruction ceiling, sign-
    oriented by ``MetricDirection``. ``normalize=False`` returns the average
    of the raw metric values (``MetricResult.raw_value``) — useful when you
    want to inspect units rather than relative position.

    Metric names in ``spec`` may use either form:

    * Bare metric name (``"acg_correlation"``) — looked up directly in the
      tokenizer's report.
    * Catalog ``"<source>/<metric>"`` form (``"recon/acg_correlation"``).
      The leading source prefix is stripped and the bare name is retried.
    """
    comparisons = compute_normalized_comparisons(model, baseline, ceiling)
    # The ALTERNATIVE_FLOORS metrics (firing_rate_error) are not scored against the Poisson
    # floor, and their reference needs per-unit rates that a bundle of aggregated reports
    # does not hold: leave them out of normalized scores rather than show the Poisson one.
    for name in ALTERNATIVE_FLOORS.keys() & comparisons.keys():
        comparisons[name] = replace(
            comparisons[name],
            normalized_score=None,
            status="invalid",
            reason="reference_not_available",
        )
    scores: dict[str, float] = {}
    for label, metric_names in spec.categories.items():
        values: list[float] = []
        missing: list[str] = []
        for name in metric_names:
            resolved = _resolve_metric_name(name, comparisons)
            if resolved is None:
                missing.append(name)
                continue
            resolved_name, cmp_ = resolved
            if normalize:
                if cmp_.reason == "reference_not_available":
                    LOGGER.warning(
                        "category %r: %s left out of the normalized radar; its reference "
                        "needs per-unit rates (see ALTERNATIVE_FLOORS).",
                        label,
                        resolved_name,
                    )
                if cmp_.normalized_score is None or cmp_.status != "ok":
                    missing.append(name)
                    continue
                values.append(float(cmp_.normalized_score))
            else:
                # Raw mode: pull the metric's raw_value from the model report.
                model_result = model.results.get(resolved_name)
                if model_result is None:
                    missing.append(name)
                    continue
                raw = _to_scalar_or_nan(model_result.raw_value)
                if not np.isfinite(raw):
                    missing.append(name)
                    continue
                values.append(raw)
        if missing:
            LOGGER.debug(
                "category %r: %d metrics unavailable (%s)", label, len(missing), ", ".join(missing)
            )
        if not values:
            scores[label] = float("nan")
            continue
        scores[label] = float(np.mean(values) if aggregator == "mean" else np.median(values))
    return scores


def _to_scalar_or_nan(value: object) -> float:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, np.ndarray):
        return float(np.nanmean(value))
    return float("nan")


def collect_category_scores(
    bundle: RadarBundle,
    spec: CategorySpec,
    *,
    aggregator: str = "mean",
    normalize: bool = True,
) -> pd.DataFrame:
    """Return a (tokenizer, category) -> score table for every tokenizer in ``bundle``.

    Pass ``normalize=False`` to get raw averaged metric values instead of
    Poisson-normalized scores. The shape of the returned DataFrame is identical
    in both modes; only the values differ.
    """
    rows: dict[str, dict[str, float]] = {}
    for tokenizer, report in bundle.tokenizers.items():
        rows[tokenizer] = category_scores_for(
            report,
            bundle.baseline,
            bundle.ceiling,
            spec,
            aggregator=aggregator,
            normalize=normalize,
        )
    df = pd.DataFrame(rows).T  # rows = tokenizers, cols = categories
    return df.reindex(columns=spec.labels())


def _radar_projection(num_vars: int) -> tuple[np.ndarray, str]:
    """Register a radar projection with polygon gridlines and return its angles/name."""
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    projection_name = f"ntt_radar_polygon_{num_vars}"

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return MplPath(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        PolarTransform = RadarTransform
        name = projection_name

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location("N")
            self.set_theta_direction(-1)

        def fill(self, *args, closed: bool = True, **kwargs):
            return super().fill(*args, closed=closed, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)
            return lines

        def _close_line(self, line) -> None:
            x, y = line.get_data()
            if len(x) and x[0] != x[-1]:
                line.set_data(np.append(x, x[0]), np.append(y, y[0]))

        def set_varlabels(self, labels: list[str]) -> None:
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor="none")

        def _gen_axes_spines(self):
            spine = Spine(
                axes=self, spine_type="circle", path=MplPath.unit_regular_polygon(num_vars)
            )
            spine.set_transform(Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes)
            return {"polar": spine}

    register_projection(RadarAxes)
    return theta, projection_name


def plot_radar(
    scores_by_tokenizer: Mapping[str, Mapping[str, float]] | pd.DataFrame,
    *,
    title: str | None = None,
    mode: str = "normed",
    figsize: tuple[float, float] = (8, 8),
    clip_range: tuple[float, float] | None = None,
    floor_negative: bool = True,
    floor_name: str = "Poisson floor",
    display_floor: bool | None = None,
    show_ceiling_ring: bool | None = None,
    cmap: str = "tab10",
    fill_alpha: float = 0.18,
    legend_loc: str | None = "upper right",
    category_label_pad: float = 18.0,
    ax: Axes | None = None,
) -> Figure:
    """Overlay one polygon per tokenizer on a polar radar.

    Parameters
    ----------
    mode:
        ``"normed"`` (default) — values are Poisson-normalized scores. ``0``
        corresponds to the tokenizer floor and ``1`` to the perfect-
        reconstruction ceiling. The dashed floor / solid ceiling guide rings
        are drawn and the title gets a ``(NORMED)`` suffix.

        ``"raw"`` — values are interpreted as raw metric values. Floor /
        ceiling rings are hidden by default (the canonical 0/1 positions don't
        align with raw units), the radial range is auto-derived from the data,
        and the title gets a ``(RAW)`` suffix.

        Any other string is appended to the title verbatim, in caps —
        e.g., ``mode="z-score"`` becomes ``(Z-SCORE)``.
    clip_range:
        ``(lower, upper)`` for the radial axis. Defaults to ``(-0.5, 1.0)`` in
        ``"normed"`` mode and to a tight envelope around the data in ``"raw"``
        mode. ``floor_negative=False`` clips the lower bound at ``0``.
    display_floor / show_ceiling_ring:
        Override the mode default. ``None`` (default) means follow the mode.
    """
    if isinstance(scores_by_tokenizer, pd.DataFrame):
        df = scores_by_tokenizer
    else:
        df = pd.DataFrame(scores_by_tokenizer).T

    if df.empty:
        raise ValueError("no tokenizers / categories to plot")

    categories = list(df.columns)
    n = len(categories)
    angles, projection_name = _radar_projection(n)
    angles_closed = np.concatenate([angles, angles[:1]])

    # "normed" and "raw" are the documented modes with mode-specific axis
    # defaults; any other string is treated as a custom raw-style mode whose
    # tag is appended to the title (e.g., mode="z-score" → "(Z-SCORE)").
    # Warn on unrecognized modes so a typo of "normed" doesn't silently fall
    # through to the raw-style autoscaling branch.
    if mode not in {"normed", "raw"}:
        LOGGER.warning(
            "plot_radar: unrecognized mode=%r; treating as raw-style with a "
            "(%s) title tag. Known modes: 'normed', 'raw'.",
            mode,
            mode.upper() if mode else "",
        )
    is_normed = mode == "normed"

    if clip_range is None:
        if is_normed:
            clip_range = (-0.5, 1.0)
        else:
            data = df.to_numpy(dtype=float)
            finite = data[np.isfinite(data)]
            if finite.size == 0:
                clip_range = (0.0, 1.0)
            else:
                lo = float(np.nanmin(finite))
                hi = float(np.nanmax(finite))
                pad = max((hi - lo) * 0.1, 1e-9)
                clip_range = (lo - pad, hi + pad)

    requested_lower, upper = clip_range
    lower = requested_lower if floor_negative else max(0.0, requested_lower)
    if lower >= upper:
        raise ValueError("clip_range lower bound must be less than upper bound")

    # Resolve mode-dependent defaults for the guide rings.
    if display_floor is None:
        display_floor = is_normed
    if show_ceiling_ring is None:
        show_ceiling_ring = is_normed

    # Append a CAPS mode tag to the title so the active mode is unambiguous.
    if title is not None:
        mode_tag = mode.upper() if mode else ""
        if mode_tag and f"({mode_tag})" not in title:
            title = f"{title}  ({mode_tag})"

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": projection_name})
    else:
        parent_fig = ax.figure
        if not isinstance(parent_fig, Figure):
            raise TypeError("plot_radar requires an Axes attached to a top-level Figure")
        fig = parent_fig
        set_theta_zero_location = getattr(ax, "set_theta_zero_location", None)
        if callable(set_theta_zero_location):
            set_theta_zero_location("N")
        set_theta_direction = getattr(ax, "set_theta_direction", None)
        if callable(set_theta_direction):
            set_theta_direction(-1)

    polar_ax = cast(PolarAxes, ax)

    polar_ax.set_facecolor("white")
    polar_ax.spines["polar"].set_visible(False)
    polar_ax.grid(color="0.82", linewidth=0.8)
    polar_ax.xaxis.grid(True, color="0.82", linewidth=0.8)
    polar_ax.yaxis.grid(True, color="0.82", linewidth=0.8)

    if display_floor:
        polar_ax.plot(
            angles_closed,
            [0.0 - lower] * (n + 1),
            color="0.25",
            linestyle="--",
            linewidth=1,
            label=floor_name,
        )
    if show_ceiling_ring:
        polar_ax.plot(
            angles_closed,
            [1.0 - lower] * (n + 1),
            color="black",
            linestyle="-",
            linewidth=1.2,
            label="Perfect ceiling",
        )

    colors = plt.get_cmap(cmap)
    for idx, (tokenizer, row) in enumerate(df.iterrows()):
        clipped = np.clip(row.to_numpy(dtype=float), lower, upper) - lower
        color = colors(idx % colors.N) if hasattr(colors, "N") else colors(idx % 10)
        polar_ax.plot(
            angles, clipped, marker="o", markersize=4, linewidth=2, label=tokenizer, color=color
        )
        polar_ax.fill(angles, clipped, alpha=fill_alpha, color=color, label="_nolegend_")

    polar_ax.set_xticks(angles)
    polar_ax.set_xticklabels(categories, size=9)
    polar_ax.tick_params(axis="x", pad=category_label_pad)
    polar_ax.set_ylim(0, upper - lower)
    if is_normed:
        tick_values = list(dict.fromkeys(t for t in [lower, 0.0, 0.5, 1.0] if lower <= t <= upper))
        tick_fmt = "{:.1f}"
    else:
        tick_values = list(np.linspace(lower, upper, 5))
        tick_fmt = "{:.2g}"
    polar_ax.set_yticks([t - lower for t in tick_values])
    polar_ax.set_yticklabels([tick_fmt.format(t) for t in tick_values], size=8)
    polar_ax.set_rlabel_position(90)
    if title:
        polar_ax.set_title(title, pad=24, weight="bold")
    if legend_loc:
        polar_ax.legend(
            loc=cast(Any, legend_loc),  # str location names are valid at runtime
            bbox_to_anchor=(1.25, 1.05),
            fontsize=9,
            frameon=False,
        )
    fig.tight_layout()
    return fig


__all__ = [
    "CategorySpec",
    "DEFAULT_EVAL_JOB_SCALARS",
    "MetricComparison",
    "RadarBundle",
    "ScalarMetric",
    "category_scores_for",
    "collect_category_scores",
    "inject_scalar_metrics",
    "load_report",
    "plot_radar",
    "tokenizers_with_data",
]
