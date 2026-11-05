"""Scaling plots for comparing multiple tokenizer eval results.

Used by `scripts/replot_from_eval_results.py`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from .behavioral_names import REGRESSION_PREFIXES, REGRESSION_RESULT_NAMES
from .core.types import EvaluationReport, MetricResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetricPoint:
    tokenizer_name: str
    run_label: str
    x: float
    y: float
    yerr: float | None
    token_dims_per_second: float
    marker: str


def _as_float(x: Any) -> float | None:
    try:
        v = float(x)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return v


def _metric_scalar_value(result: MetricResult) -> float | None:
    v = result.raw_value
    if isinstance(v, np.ndarray):
        if v.size == 0:
            return None
        return _as_float(np.nanmean(v))
    return _as_float(v)


def _metric_scalar_std(result: MetricResult) -> float | None:
    """Best-effort std for error bars, mirroring behavioral plotting conventions.

    Preference order:
    - explicit *_std fields in metadata (r2_std / accuracy_std / std / raw_value_std)
    - std computed from fold lists in metadata (fold_r2s / fold_accuracies)
    - std across array-valued raw_value (e.g. per-unit metrics)
    """
    meta = result.metadata or {}
    for k in ("raw_value_std", "std", "r2_std", "accuracy_std"):
        v = meta.get(k)
        if isinstance(v, (int, float)) and v is not None:
            out = _as_float(v)
            if out is not None:
                return float(out)

    folds = meta.get("fold_r2s")
    if folds is None:
        folds = meta.get("fold_accuracies")
    if isinstance(folds, (list, tuple, np.ndarray)) and len(folds) >= 3:
        try:
            vals = np.array([float(x) for x in folds], dtype=float)
            if np.isfinite(vals).any():
                return float(np.nanstd(vals))
        except Exception:
            pass

    v = result.raw_value
    if isinstance(v, np.ndarray) and v.size >= 2:
        try:
            vv = v.astype(float, copy=False)
            if np.isfinite(vv).any():
                return float(np.nanstd(vv))
        except Exception:
            return None
    return None


def compute_token_dims_per_second(tokenizer_config: dict[str, Any] | None) -> float | None:
    """Token-dim rate (dims / second) from serialized TokenizerSpec.

    Goal: x-axis for the user's scaling plots.

    We interpret "token dimensions per 1 second time window" as "token dims per second".

    This uses only *explicit* spec fields written into eval `results.json`,
    in particular `tokenizer_config.token_bin_size_ms`. No inference from
    metric metadata, and no approximation.
    """
    if not tokenizer_config:
        return None

    has_unit_dim = bool(tokenizer_config.get("has_unit_dim", False))

    n_neurons = tokenizer_config.get("n_neurons", 1)
    if not isinstance(n_neurons, (int, float)):
        n_neurons = 1
    n_neurons = int(n_neurons) if n_neurons is not None else 1
    if n_neurons <= 0:
        n_neurons = 1

    # Latent dimension per token/window.
    # Prefer z_dim for WanNeural specs; fall back to latent_dim for others.
    latent_dim = (
        tokenizer_config.get("z_dim")
        or tokenizer_config.get("latent_dim")
        or tokenizer_config.get("codebook_dim")
        or tokenizer_config.get("bsq_codebook_dim")
        or tokenizer_config.get("q_dim")
        or 1
    )
    if not isinstance(latent_dim, (int, float)):
        latent_dim = 1
    latent_dim = int(latent_dim) if latent_dim is not None else 1
    if latent_dim <= 0:
        latent_dim = 1

    # Token rate (tokens/sec): always use explicit token_bin_size_ms from the spec.
    token_bin_ms = tokenizer_config.get("token_bin_size_ms")
    token_bin_ms_f = _as_float(token_bin_ms)
    if token_bin_ms_f is None or token_bin_ms_f <= 0:
        return None
    tokens_per_second = 1000.0 / token_bin_ms_f

    dims_per_token = latent_dim * (n_neurons if has_unit_dim else 1)
    return float(dims_per_token * tokens_per_second)


def iter_metric_points(
    *,
    tokenizer_name: str,
    report: EvaluationReport,
    metric_key: str,
    token_dims_per_second: float,
    marker: str,
    x_axis: str = "token_dims_per_second",
    detailed_sweep_points: list[dict[str, Any]] | None = None,
) -> list[MetricPoint]:
    """Return 1+ points for a metric (handles chunk sweeps)."""
    # Preferred: explicit sweep points loaded from `metrics_detailed_results.jsonl`.
    # Each entry is expected to contain at least:
    # - param_name, param_value, y, yerr (optional), run_label (optional)
    if detailed_sweep_points:
        points: list[MetricPoint] = []
        for sp in detailed_sweep_points:
            try:
                y = float(sp["y"])
            except Exception:
                continue
            if not np.isfinite(y):
                continue
            yerr = sp.get("yerr")
            yerr_f: float | None = None
            if isinstance(yerr, (int, float)):
                yerr_f = float(yerr) if np.isfinite(float(yerr)) else None

            x = float(token_dims_per_second)
            param_name = str(sp.get("param_name") or "")
            param_value = sp.get("param_value")
            if x_axis == "token_dims_per_chunk" and param_name == "chunk_window_ms":
                cw_ms_f = _as_float(param_value)
                if cw_ms_f is not None and cw_ms_f > 0:
                    x = x * (cw_ms_f / 1000.0)

            run_label = sp.get("run_label")
            if not isinstance(run_label, str) or not run_label:
                if param_name == "chunk_window_ms":
                    run_label = f"cw{param_value}ms"
                elif param_name == "bin_size_ms":
                    run_label = f"{param_value}ms"
                elif param_name:
                    run_label = f"{param_name}={param_value}"
                else:
                    run_label = ""

            points.append(
                MetricPoint(
                    tokenizer_name=tokenizer_name,
                    run_label=run_label,
                    x=float(x),
                    y=float(y),
                    yerr=yerr_f,
                    token_dims_per_second=float(token_dims_per_second),
                    marker=marker,
                )
            )
        if points:
            return points

    if metric_key not in report.results:
        return []
    result = report.results[metric_key]

    base = _metric_scalar_value(result)
    if base is None:
        return []
    base_std = _metric_scalar_std(result)

    points: list[MetricPoint] = []

    # Downstream metrics may store a per-chunk sweep in metadata.
    sweep = result.metadata.get("chunk_window_sweep")
    if isinstance(sweep, dict) and sweep:
        for cw_ms, payload in sweep.items():
            if not isinstance(payload, dict):
                continue
            raw = payload.get("raw_value")
            y = _as_float(raw)
            if y is None:
                continue
            yerr: float | None = None
            payload_meta = payload.get("metadata")
            if isinstance(payload_meta, dict):
                # Prefer per-sweep-point std if present.
                for k in ("raw_value_std", "std", "r2_std", "accuracy_std"):
                    v = payload_meta.get(k)
                    if isinstance(v, (int, float)) and v is not None:
                        yerr = _as_float(v)
                        break
                if yerr is None:
                    folds = payload_meta.get("fold_r2s")
                    if folds is None:
                        folds = payload_meta.get("fold_accuracies")
                    if isinstance(folds, (list, tuple, np.ndarray)) and len(folds) >= 3:
                        try:
                            fold_vals = np.array([float(x) for x in folds], dtype=float)
                            if np.isfinite(fold_vals).any():
                                yerr = float(np.nanstd(fold_vals))
                        except Exception:
                            yerr = None
            # For sweep points, optionally scale x by chunk window to show
            # "token dims per chunk" instead of per second.
            x = float(token_dims_per_second)
            if x_axis == "token_dims_per_chunk":
                cw_ms_f = _as_float(cw_ms)
                if cw_ms_f is not None and cw_ms_f > 0:
                    x = x * (cw_ms_f / 1000.0)
            points.append(
                MetricPoint(
                    tokenizer_name=tokenizer_name,
                    run_label=f"cw{cw_ms}ms",
                    x=float(x),
                    y=float(y),
                    yerr=float(yerr) if yerr is not None else None,
                    token_dims_per_second=float(token_dims_per_second),
                    marker=marker,
                )
            )
        # If we had sweep points, return those (they are what the user wants).
        if points:
            return points

    # Default: single point (best / scalar).
    points.append(
        MetricPoint(
            tokenizer_name=tokenizer_name,
            run_label="",
            x=float(token_dims_per_second),
            y=float(base),
            yerr=float(base_std) if base_std is not None else None,
            token_dims_per_second=float(token_dims_per_second),
            marker=marker,
        )
    )
    return points


def _stable_markers(names: list[str]) -> dict[str, str]:
    markers = ["o", "s", "^", "v", "D", "P", "X", "*", "<", ">", "h", "H", "p", "d"]
    out: dict[str, str] = {}
    for idx, name in enumerate(sorted(names)):
        out[name] = markers[idx % len(markers)]
    return out


def plot_metric_vs_token_dims(
    *,
    metric_key: str,
    reports_by_tokenizer: dict[str, EvaluationReport],
    token_dims_by_tokenizer: dict[str, float],
    detailed_sweeps_by_tokenizer: dict[str, dict[str, list[dict[str, Any]]]] | None = None,
    output_path: Path,
    dpi: int = 150,
    x_axis: str = "token_dims_per_second",
) -> None:
    names = list(reports_by_tokenizer.keys())
    marker_by_name = _stable_markers(names)

    all_points: list[MetricPoint] = []
    for name, report in reports_by_tokenizer.items():
        td = token_dims_by_tokenizer.get(name)
        if td is None:
            continue
        sweeps_for_metric = None
        if detailed_sweeps_by_tokenizer and name in detailed_sweeps_by_tokenizer:
            sweeps_for_metric = detailed_sweeps_by_tokenizer[name].get(metric_key)
        all_points.extend(
            iter_metric_points(
                tokenizer_name=name,
                report=report,
                metric_key=metric_key,
                token_dims_per_second=td,
                marker=marker_by_name[name],
                x_axis=x_axis,
                detailed_sweep_points=sweeps_for_metric,
            )
        )
    if not all_points:
        return

    # Color by tokenizer (distinct), marker per tokenizer too.
    cmap = plt.get_cmap("tab10")
    color_by_name = {n: cmap(i % 10) for i, n in enumerate(sorted(names))}

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for n in sorted(names):
        pts = [p for p in all_points if p.tokenizer_name == n]
        if not pts:
            continue
        xs = [p.x for p in pts]
        ys = [p.y for p in pts]
        yerrs = np.array([np.nan if p.yerr is None else float(p.yerr) for p in pts], dtype=float)
        ax.scatter(
            xs,
            ys,
            s=55,
            marker=marker_by_name[n],
            color=color_by_name[n],
            edgecolor="black",
            linewidth=0.6,
            alpha=0.9,
            label=n,
        )
        if np.isfinite(yerrs).any():
            ax.errorbar(
                xs,
                ys,
                yerr=np.where(np.isfinite(yerrs), yerrs, np.nan),
                fmt="none",
                ecolor="black",
                elinewidth=1.0,
                capsize=2,
                alpha=0.9,
                zorder=0,
            )
        # If this metric has multiple points (e.g. chunk sweeps), annotate with the run_label.
        for p in pts:
            if p.run_label:
                ax.annotate(
                    p.run_label,
                    (p.x, p.y),
                    textcoords="offset points",
                    xytext=(5, 4),
                    fontsize=7,
                    alpha=0.85,
                )

    if x_axis == "token_dims_per_chunk":
        ax.set_xlabel("token dimensions per chunk window")
        ax.set_title(f"{metric_key} vs token dimensions/chunk")
    else:
        ax.set_xlabel("token dimensions per second")
        ax.set_title(f"{metric_key} vs token dimensions/sec")
    ax.set_ylabel(metric_key)
    ax.grid(True, alpha=0.25)

    # Keep legend readable: limit to 12 entries; beyond that, skip.
    if len(names) <= 12:
        ax.legend(fontsize=8, loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.debug("Saved: %s", output_path)


def plot_metric_vs_token_dims_per_session(
    *,
    metric_key: str,
    tokenizer_session_reports: dict[str, dict[str, list[tuple]]],
    token_dims_by_tokenizer: dict[str, float],
    detailed_sweeps_by_tokenizer_and_session: dict[str, dict[str, dict[str, list[dict[str, Any]]]]]
    | None = None,
    output_dir: Path,
    dpi: int = 150,
    x_axis: str = "token_dims_per_chunk",
) -> None:
    """Per-session scaling plot (no cross-session aggregation).

    Writes one figure per session_id under output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all session IDs present anywhere.
    session_ids: set[str] = set()
    for _tok, sess_map in tokenizer_session_reports.items():
        session_ids |= set(sess_map.keys())

    marker_by_name = _stable_markers(list(tokenizer_session_reports.keys()))
    cmap = plt.get_cmap("tab10")
    color_by_name = {
        n: cmap(i % 10) for i, n in enumerate(sorted(tokenizer_session_reports.keys()))
    }

    for sid in sorted(session_ids):
        all_points: list[MetricPoint] = []
        for tok_name, sess_map in tokenizer_session_reports.items():
            segments = sess_map.get(sid)
            if not segments:
                continue
            # Use the first per-segment report. (Most runs use a single segment.)
            report = segments[0][1]["report"]
            td = token_dims_by_tokenizer.get(tok_name)
            if td is None:
                continue
            sweeps_for_metric = None
            if (
                detailed_sweeps_by_tokenizer_and_session
                and tok_name in detailed_sweeps_by_tokenizer_and_session
            ):
                sweeps_for_metric = (
                    detailed_sweeps_by_tokenizer_and_session[tok_name].get(sid, {}).get(metric_key)
                )
            all_points.extend(
                iter_metric_points(
                    tokenizer_name=tok_name,
                    report=report,
                    metric_key=metric_key,
                    token_dims_per_second=td,
                    marker=marker_by_name[tok_name],
                    x_axis=x_axis,
                    detailed_sweep_points=sweeps_for_metric,
                )
            )

        if not all_points:
            continue

        # 2-column layout: wide main plot + dedicated legend panel.
        fig, (ax, ax_leg) = plt.subplots(
            1,
            2,
            figsize=(11.0, 5.0),
            gridspec_kw={"width_ratios": [1.0, 0.45], "wspace": 0.10},
        )
        ax_leg.axis("off")
        for n in sorted({p.tokenizer_name for p in all_points}):
            pts = [p for p in all_points if p.tokenizer_name == n]
            xs = [p.x for p in pts]
            ys = [p.y for p in pts]
            yerrs = np.array(
                [np.nan if p.yerr is None else float(p.yerr) for p in pts], dtype=float
            )
            ax.scatter(
                xs,
                ys,
                s=55,
                marker=marker_by_name[n],
                color=color_by_name.get(n, "C0"),
                edgecolor="black",
                linewidth=0.6,
                alpha=0.9,
                label=n,
            )
            if np.isfinite(yerrs).any():
                ax.errorbar(
                    xs,
                    ys,
                    yerr=np.where(np.isfinite(yerrs), yerrs, np.nan),
                    fmt="none",
                    ecolor="black",
                    elinewidth=1.0,
                    capsize=2,
                    alpha=0.9,
                    zorder=0,
                )
            for p in pts:
                if p.run_label:
                    ax.annotate(
                        p.run_label,
                        (p.x, p.y),
                        textcoords="offset points",
                        xytext=(5, 4),
                        fontsize=7,
                        alpha=0.85,
                    )

        safe_sid = sid.replace("/", "_")
        if x_axis == "token_dims_per_chunk":
            ax.set_xlabel("token dimensions per chunk window")
            ax.set_title(f"{metric_key} vs token dims/chunk\n({sid})")
        else:
            ax.set_xlabel("token dimensions per second")
            ax.set_title(f"{metric_key} vs token dims/sec\n({sid})")
        ax.set_ylabel(metric_key)
        ax.grid(True, alpha=0.25)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax_leg.legend(
                handles,
                labels,
                loc="upper left",
                frameon=False,
                fontsize=7,
                borderaxespad=0.0,
            )

        out = output_dir / f"metric_vs_token_dims__{metric_key}__{safe_sid}.png"
        fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.12)
        # Ensure the legend panel isn't cropped in the saved image.
        fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.10)
        plt.close(fig)
        logger.debug("Saved: %s", out)


def plot_metric_vs_metric_colored_by_token_dims(
    *,
    x_metric: str,
    y_metric: str,
    reports_by_tokenizer: dict[str, EvaluationReport],
    token_dims_by_tokenizer: dict[str, float],
    detailed_sweeps_by_tokenizer: dict[str, dict[str, list[dict[str, Any]]]] | None = None,
    output_path: Path,
    dpi: int = 150,
) -> None:
    names = list(reports_by_tokenizer.keys())
    marker_by_name = _stable_markers(names)

    # Gather points (allow sweep on y metric; use scalar x).
    xs: list[float] = []
    xerrs: list[float] = []
    ys: list[float] = []
    yerrs: list[float] = []
    cs: list[float] = []
    markers: list[str] = []
    labels: list[str] = []

    for name, report in reports_by_tokenizer.items():
        td = token_dims_by_tokenizer.get(name)
        if td is None:
            continue

        x_res = report.results.get(x_metric)
        y_res = report.results.get(y_metric)
        if x_res is None or y_res is None:
            continue

        x_val = _metric_scalar_value(x_res)
        if x_val is None:
            continue
        # Horizontal error bars are confusing here (often reflect per-unit dispersion).
        # We intentionally disable x-std error bars here.

        sweeps_for_metric = None
        if detailed_sweeps_by_tokenizer and name in detailed_sweeps_by_tokenizer:
            sweeps_for_metric = detailed_sweeps_by_tokenizer[name].get(y_metric)
        y_points = iter_metric_points(
            tokenizer_name=name,
            report=report,
            metric_key=y_metric,
            token_dims_per_second=td,
            marker=marker_by_name[name],
            detailed_sweep_points=sweeps_for_metric,
        )
        if not y_points:
            continue

        for yp in y_points:
            xs.append(float(x_val))
            xerrs.append(float("nan"))
            ys.append(float(yp.y))
            yerrs.append(float(yp.yerr) if yp.yerr is not None else float("nan"))
            cs.append(float(td))
            markers.append(marker_by_name[name])
            labels.append(name)

    if not xs:
        return

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    norm = Normalize(vmin=float(np.min(cs)), vmax=float(np.max(cs)))
    cmap = plt.get_cmap("viridis")

    # Plot per tokenizer to keep marker consistent.
    for name in sorted(set(labels)):
        idxs = [i for i, lab in enumerate(labels) if lab == name]
        xs_sub = [xs[i] for i in idxs]
        ys_sub = [ys[i] for i in idxs]
        ax.scatter(
            xs_sub,
            ys_sub,
            s=70,
            marker=marker_by_name[name],
            c=[cs[i] for i in idxs],
            cmap=cmap,
            norm=norm,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.9,
            label=name,
        )
        # Draw error bars point-wise (supports missing stds via NaNs).
        for i in idxs:
            xe = xerrs[i]
            ye = yerrs[i]
            if not (np.isfinite(xe) or np.isfinite(ye)):
                continue
            ax.errorbar(
                xs[i],
                ys[i],
                xerr=xe if np.isfinite(xe) else None,
                yerr=ye if np.isfinite(ye) else None,
                fmt="none",
                ecolor="black",
                elinewidth=1.0,
                capsize=2,
                alpha=0.9,
                zorder=0,
            )

    ax.set_xlabel(x_metric)
    ax.set_ylabel(y_metric)
    ax.set_title(f"{y_metric} vs {x_metric} (color = token dims/sec)")
    ax.grid(True, alpha=0.25)
    if len(set(labels)) <= 12:
        ax.legend(fontsize=8, loc="best")

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("token dimensions per second")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.debug("Saved: %s", output_path)


def plot_metric_vs_metric_colored_by_token_dims_per_session(
    *,
    x_metric: str,
    y_metric: str,
    tokenizer_session_reports: dict[str, dict[str, list[tuple]]],
    token_dims_by_tokenizer: dict[str, float],
    detailed_sweeps_by_tokenizer_and_session: dict[str, dict[str, dict[str, list[dict[str, Any]]]]]
    | None = None,
    output_dir: Path,
    dpi: int = 150,
    x_axis: str = "token_dims_per_chunk",
) -> None:
    """Per-session scatter: y_metric vs x_metric, color = token dims (no cross-session agg.)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    session_ids: set[str] = set()
    for _tok, sess_map in tokenizer_session_reports.items():
        session_ids |= set(sess_map.keys())

    marker_by_name = _stable_markers(list(tokenizer_session_reports.keys()))

    for sid in sorted(session_ids):
        xs: list[float] = []
        xerrs: list[float] = []
        ys: list[float] = []
        yerrs: list[float] = []
        cs: list[float] = []
        markers: list[str] = []
        labels: list[str] = []
        ann: list[str] = []

        for name, sess_map in tokenizer_session_reports.items():
            segments = sess_map.get(sid)
            if not segments:
                continue
            report = segments[0][1]["report"]
            td = token_dims_by_tokenizer.get(name)
            if td is None:
                continue

            x_res = report.results.get(x_metric)
            y_res = report.results.get(y_metric)
            if x_res is None or y_res is None:
                continue

            x_val = _metric_scalar_value(x_res)
            if x_val is None:
                continue
            # Horizontal error bars are confusing here (often reflect per-unit dispersion).
            # We intentionally disable x-std error bars here.

            sweeps_for_metric = None
            if (
                detailed_sweeps_by_tokenizer_and_session
                and name in detailed_sweeps_by_tokenizer_and_session
            ):
                sweeps_for_metric = (
                    detailed_sweeps_by_tokenizer_and_session[name].get(sid, {}).get(y_metric)
                )
            y_points = iter_metric_points(
                tokenizer_name=name,
                report=report,
                metric_key=y_metric,
                token_dims_per_second=td,
                marker=marker_by_name[name],
                x_axis=x_axis,
                detailed_sweep_points=sweeps_for_metric,
            )
            if not y_points:
                continue

            for yp in y_points:
                xs.append(float(x_val))
                xerrs.append(float("nan"))
                ys.append(float(yp.y))
                yerrs.append(float(yp.yerr) if yp.yerr is not None else float("nan"))
                cs.append(float(yp.x))
                markers.append(marker_by_name[name])
                labels.append(name)
                ann.append(yp.run_label or "")

        if not xs:
            continue

        # 3-column layout: main scatter, dedicated colorbar axis, dedicated legend axis.
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(10.5, 5.2),
            gridspec_kw={"width_ratios": [1.0, 0.07, 0.34], "wspace": 0.20},
        )
        ax_main, ax_cbar, ax_leg = axes
        ax_leg.axis("off")

        norm = Normalize(vmin=float(np.min(cs)), vmax=float(np.max(cs)))
        cmap = plt.get_cmap("viridis")

        handles_by_name: dict[str, object] = {}
        for name in sorted(set(labels)):
            idxs = [i for i, lab in enumerate(labels) if lab == name]
            xs_sub = [xs[i] for i in idxs]
            ys_sub = [ys[i] for i in idxs]
            sc = ax_main.scatter(
                xs_sub,
                ys_sub,
                s=70,
                marker=marker_by_name[name],
                c=[cs[i] for i in idxs],
                cmap=cmap,
                norm=norm,
                edgecolor="black",
                linewidth=0.6,
                alpha=0.9,
                label=name,
            )
            handles_by_name[name] = sc
            # Error bars per point (black), when stds are available.
            for i in idxs:
                xe = xerrs[i]
                ye = yerrs[i]
                if not (np.isfinite(xe) or np.isfinite(ye)):
                    continue
                ax_main.errorbar(
                    xs[i],
                    ys[i],
                    xerr=xe if np.isfinite(xe) else None,
                    yerr=ye if np.isfinite(ye) else None,
                    fmt="none",
                    ecolor="black",
                    elinewidth=1.0,
                    capsize=2,
                    alpha=0.9,
                    zorder=0,
                )
            for i in idxs:
                if ann[i]:
                    ax_main.annotate(
                        ann[i],
                        (xs[i], ys[i]),
                        textcoords="offset points",
                        xytext=(5, 4),
                        fontsize=7,
                        alpha=0.85,
                    )

        ax_main.set_xlabel(x_metric)
        ax_main.set_ylabel(y_metric)
        cbar_label = (
            "token dimensions per chunk window"
            if x_axis == "token_dims_per_chunk"
            else "token dimensions per second"
        )
        fig.suptitle(f"{y_metric} vs {x_metric}\n({sid})", fontsize=11, y=0.98)
        ax_main.text(
            0.5,
            1.02,
            f"color = {cbar_label}",
            transform=ax_main.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
        )
        ax_main.grid(True, alpha=0.25)

        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=ax_cbar, orientation="vertical")
        cbar.ax.yaxis.set_ticks_position("right")
        cbar.ax.yaxis.set_label_position("right")
        cbar.set_label(cbar_label)

        leg = ax_leg.legend(
            handles=[handles_by_name[n] for n in sorted(handles_by_name.keys())],
            labels=sorted(handles_by_name.keys()),
            loc="upper left",
            frameon=False,
            fontsize=7,
            borderaxespad=0.0,
        )
        leg.get_frame().set_linewidth(0)

        safe_sid = sid.replace("/", "_")
        out = output_dir / f"scatter__{y_metric}__vs__{x_metric}__{safe_sid}.png"
        fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.12)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.debug("Saved: %s", out)


def pick_metric_keys_for_dims_plot(
    reports_by_tokenizer: dict[str, EvaluationReport],
    *,
    include: Iterable[str] | None = None,
    exclude_prefixes: tuple[str, ...] = ("boxplot_",),
) -> list[str]:
    """Choose metric keys for `metric_vs_token_dims` plots.

    Default behavior is intentionally conservative: if ``include`` is not
    provided, behavioral decoding metrics plus ``repeats_snr`` are selected.
    Passing an explicit ``include`` list overrides that default filter.

    ``LATENT_QUALITY_DEFAULT_NAMES`` covers non-behavioral scalar metrics we
    still want to show as scaling-plot scatters by default. ``repeats_snr``
    is the only one for now. ``token_dims_per_second`` is intentionally
    excluded — it is the X-axis of these scatters, so plotting it as Y
    against itself is trivially y=x. It still appears in ``metric_bars.png``
    via the per-metric bar plotting.
    """
    LATENT_QUALITY_DEFAULT_NAMES = {"repeats_snr"}

    keys: set[str] = set()
    for r in reports_by_tokenizer.values():
        keys.update(r.results.keys())

    include_set = set(include) if include is not None else None
    behavioral_prefixes = REGRESSION_PREFIXES + ("behavioral_classification_",)
    behavioral_exact_names = set(REGRESSION_RESULT_NAMES) | {"behavioral_classification"}

    out: list[str] = []
    for k in sorted(keys):
        if include_set is not None and k not in include_set:
            continue
        if any(k.startswith(p) for p in exclude_prefixes):
            continue
        if include_set is None and not (
            k in behavioral_exact_names
            or k in LATENT_QUALITY_DEFAULT_NAMES
            or any(k.startswith(p) for p in behavioral_prefixes)
        ):
            continue
        out.append(k)
    return out
