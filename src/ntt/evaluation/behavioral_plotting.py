"""Plotting utilities for behavioral decoding (regression and classification) results.

Supports comparing multiple tokenizers across sessions for downstream prediction metrics.

Plot requirements
-----------------
All behavioral plots must satisfy the following annotation and layout rules:

1. **Required annotations on every plot**: target field (what is being predicted),
   session ID, tokenizer name, filters applied (phases, etc.), and metric name/value.

2. **Regression comparison** (``plot_regression_comparison``):
   - Grouped bar chart, one bar per tokenizer.
   - X-axis: behavioral target fields (e.g. cursor_pos, cursor_vel, trial_target_dir).
   - Y-axis: R², range fixed to [-1, 1]. Values outside this range are still
     annotated numerically on the bar.
   - Second row of subplots shows train/val sample counts per metric.
   - Multi-session: one subplot column per session; title includes session_id.

3. **Classification comparison** (``plot_classification_comparison``):
   - Grouped bar chart, one bar per tokenizer.
   - X-axis: classification targets (e.g. trial_target_id, movement_phase, trial_result).
   - Y-axis: Accuracy (0–1). Dashed horizontal line at chance accuracy per metric.
   - Second row of subplots shows per-class sample counts (train/val) as text.
   - Multi-session: one subplot column per session.

4. **Per-class accuracy** (``plot_per_class_accuracy``):
   - Per-class bars with accuracy % and sample count (n=...) annotated on each bar.
   - For ``movement_phase``, x-tick labels use human-readable phase names
     (from ``MOVEMENT_PHASE_LABELS_REVERSE``) instead of integer codes.

5. **Trajectory plots** (2D and 3D, single-trial and multi-trial):
   - Saved under ``{output_dir}/sessions/{session_id}/bhav/trajectories/`` (nested
     folders when ``session_id`` contains ``/``).
   - Single-trial: small grid of individual trial trajectories.
   - Multi-trial (``plot_trajectory_2d_multi``, ``plot_trajectory_3d_multi``):
     ~5 contiguous trials concatenated, with trial boundary markers.
   - Both include annotation text (target, session, tokenizer, filters, metric value).
   - 1D targets (e.g. trial_target_dir) are skipped for trajectory plots (return None).
"""

from __future__ import annotations

import logging
import textwrap
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .behavioral_names import (
    BEHAVIORAL_NAMES,
    REGRESSION_PREFIXES,
    REGRESSION_RESULT_NAMES,
)
from .core.types import MetricResult

logger = logging.getLogger(__name__)

_XTICK_LABEL_WRAP_CHARS = 24
_BARPLOT_TICK_FONTSIZE = 11
_BARPLOT_LABEL_FONTSIZE = 12
_BARPLOT_TITLE_FONTSIZE = 12
_BARPLOT_ANNOT_FONTSIZE = 10
_BARPLOT_LEGEND_FONTSIZE = 10
_PER_METRIC_FIGSIZE = (6, 5)
_SESSION_TITLE_WRAP_CHARS = 42


def _wrap_tick_label(label: str, width: int = _XTICK_LABEL_WRAP_CHARS) -> str:
    """Wrap an x-tick label to multiple lines at a fixed character width.

    Existing newlines are preserved and each line block is wrapped independently.
    """
    if width <= 0:
        return label

    wrapped_parts: list[str] = []
    for part in label.split("\n"):
        if not part:
            wrapped_parts.append("")
            continue
        wrapped_parts.extend(textwrap.wrap(part, width=width, break_long_words=True))
    return "\n".join(wrapped_parts)


def _shorten_behavioral_plot_identifier(name: str) -> str:
    """Apply project-specific abbreviations to behavioral plot filenames/ids."""
    s = name
    s = s.replace("behavioral_temporal_decoder", "bhav_temp_dec")
    s = s.replace("behavioral_regression", "bhav_reg")
    s = s.replace("behavioral_classification", "bhav_clf")
    s = s.replace("center_out_reaching", "CO")
    s = s.replace("random_target_reaching", "RR")
    s = s.replace("random_reaching", "RR")
    # Generic fallbacks (kept last to avoid double-replacing already-shortened ids).
    s = s.replace("behavioral", "bhav")
    s = s.replace("regression", "reg")
    s = s.replace("classification", "clf")
    return s


def _strip_metric_prefix(name: str, prefixes: str | tuple[str, ...]) -> str:
    """Remove the first matching metric prefix from a display name."""
    if isinstance(prefixes, str):
        prefixes = (prefixes,)
    for prefix in prefixes:
        if name.startswith(prefix):
            return name.removeprefix(prefix)
    return name


def _apply_trained_checkpoint_suptitle(
    fig: Figure,
    caption: str,
    *,
    wrap_width: int = 56,
    max_lines: int = 8,
    fontsize: float = 7.5,
) -> None:
    """Add a wrapped checkpoint path as ``fig.suptitle`` and shrink layout to fit."""
    lines = textwrap.wrap(
        caption,
        width=wrap_width,
        break_long_words=True,
        break_on_hyphens=False,
    )
    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + ["…"]
    fig.suptitle("\n".join(lines), fontsize=fontsize, y=1.02, va="bottom")
    fig.tight_layout(rect=(0, 0.0, 1.0, 0.90))


def _visually_hide_xticklabels(ax: Axes) -> None:
    """Keep x tick labels for introspection while making them invisible in the plot."""
    for tick in ax.get_xticklabels():
        tick.set_color((0, 0, 0, 0))


def _tokenizer_plot_filename_suffix(tok_name: str) -> str:
    """Sanitize tokenizer display names for trajectory filenames."""
    s = tok_name.lower()
    for old, new in (
        (" ", "_"),
        ("|", "_"),
        ("/", "_"),
        ("\\", "_"),
        (":", "_"),
        ("(", ""),
        (")", ""),
        (",", ""),
        ("—", "-"),
        ("–", "-"),
    ):
        s = s.replace(old, new)
    while "__" in s:
        s = s.replace("__", "_")
    s = s.strip("_")
    return s[:150] if len(s) > 150 else s


def _format_chunk_window_ms(value: Any) -> str:
    """Format chunk_window_ms values for filenames (e.g. '200', '1000')."""
    try:
        v = float(value)
    except Exception:
        return str(value)
    if np.isfinite(v) and abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:g}"


def _behavioral_chunk_suffix(metadata: dict[str, Any] | None) -> str:
    """Return a filename-safe chunk window suffix like '_cw1000-200ms' or '_cw200ms'."""
    if not metadata:
        return ""
    sweep_vals = metadata.get("chunk_window_sweep_values")
    if sweep_vals is not None:
        try:
            values = list(sweep_vals)
        except Exception:
            values = [sweep_vals]
        if len(values) == 0:
            return ""
        joined = "-".join(_format_chunk_window_ms(v) for v in values)
        return f"_cw{joined}ms"
    cw = metadata.get("chunk_window_ms")
    if cw is None:
        return ""
    return f"_cw{_format_chunk_window_ms(cw)}ms"


def _first_metric_result_for_display_name(
    session_tok_data: dict[str, dict[str, MetricResult]],
    display_name: str,
) -> MetricResult | None:
    """Grab the first available MetricResult for a display_name across tokenizers."""
    for _tok_name, metrics in session_tok_data.items():
        r = metrics.get(display_name)
        if r is not None:
            return r
    return None


def _extract_chunk_windows_from_results(
    session_tok_data: dict[str, dict[str, MetricResult]],
    display_names: list[str],
) -> list[float]:
    """Return sorted unique chunk_window_ms values present in chunk sweeps."""
    values: set[float] = set()
    for dn in display_names:
        r = _first_metric_result_for_display_name(session_tok_data, dn)
        if r is None:
            continue
        sweep_vals = r.metadata.get("chunk_window_sweep_values")
        if sweep_vals is None:
            continue
        try:
            for v in list(sweep_vals):
                try:
                    values.add(float(v))
                except Exception:
                    continue
        except Exception:
            try:
                values.add(float(sweep_vals))
            except Exception:
                continue
    return sorted(values)


def _select_chunk_window_in_session_tok_data(
    session_tok_data: dict[str, dict[str, MetricResult]],
    *,
    chunk_window_ms: float,
) -> dict[str, dict[str, MetricResult]]:
    """Rewrite downstream metric results to use a specific chunk_window_ms value.

    For metrics that store a chunk sweep (metadata['chunk_window_sweep']), replace
    raw_value with the sweep value for the requested chunk size so plots can be
    generated separately per chunk size.
    """
    out: dict[str, dict[str, MetricResult]] = {}
    for tok_name, metrics in session_tok_data.items():
        out[tok_name] = {}
        for dn, r in metrics.items():
            sweep = r.metadata.get("chunk_window_sweep")
            if isinstance(sweep, dict) and sweep:
                # NOTE: `chunk_window_sweep` may come from JSON, where dict keys
                # are always strings. Handle both float and string keys.
                payload = sweep.get(chunk_window_ms)
                if payload is None:
                    payload = sweep.get(str(float(chunk_window_ms)))
                if payload is None:
                    payload = sweep.get(str(int(chunk_window_ms)))
                if isinstance(payload, dict) and "raw_value" in payload:
                    raw = payload.get("raw_value")
                    if raw is not None:
                        new_meta = dict(r.metadata)
                        payload_meta = payload.get("metadata")
                        if isinstance(payload_meta, dict) and payload_meta:
                            # Prefer the per-chunk-size metadata computed during the sweep.
                            new_meta.update(payload_meta)
                        new_meta["chunk_window_ms"] = float(chunk_window_ms)
                        new_meta["chunk_window_ms_selected_for_plot"] = float(chunk_window_ms)
                        out[tok_name][dn] = replace(r, raw_value=float(raw), metadata=new_meta)
                        continue
            out[tok_name][dn] = r
    return out


def _filter_classification_by_balance(
    plot_data: dict,
    *,
    include_balanced: bool,
) -> dict:
    """Filter behavioral classification metrics by `metadata["balance_classes"]`.

    Preserves the shape expected by `plot_classification_comparison`:
    - single-session: ``{tokenizer_name: {display_name: MetricResult}}``
    - multi-session:  ``{session_id: {tokenizer_name: {display_name: MetricResult}}}``
    """
    if not plot_data:
        return plot_data

    out = {}
    if _is_single_session(plot_data):
        for tok_name, metrics in plot_data.items():
            kept: dict[str, MetricResult] = {}
            for display_name, result in metrics.items():
                if result.name == "behavioral_classification":
                    is_balanced = bool(result.metadata.get("balance_classes", False))
                    if is_balanced != include_balanced:
                        continue
                kept[display_name] = result
            out[tok_name] = kept
        return out

    # multi-session case
    for session_id, tok_map in plot_data.items():
        out[session_id] = {}
        for tok_name, metrics in tok_map.items():
            kept = {}
            for display_name, result in metrics.items():
                if result.name == "behavioral_classification":
                    is_balanced = bool(result.metadata.get("balance_classes", False))
                    if is_balanced != include_balanced:
                        continue
                kept[display_name] = result
            out[session_id][tok_name] = kept
    return out


def _filter_plot_data_by_display_names(
    plot_data: dict,
    display_names: set[str],
) -> dict:
    """Keep only selected metric display names from plot_data."""
    if not plot_data:
        return plot_data

    out = {}
    if _is_single_session(plot_data):
        for tok_name, metrics in plot_data.items():
            out[tok_name] = {dn: result for dn, result in metrics.items() if dn in display_names}
        return out

    for session_id, tok_map in plot_data.items():
        out[session_id] = {}
        for tok_name, metrics in tok_map.items():
            out[session_id][tok_name] = {
                dn: result for dn, result in metrics.items() if dn in display_names
            }
    return out


def _format_subplot_title_with_session(
    base_title: str,
    session_id: str | None,
    wrap_chars: int = _SESSION_TITLE_WRAP_CHARS,
) -> str:
    """Format subplot title with session id on a wrapped second line."""
    if session_id is None:
        return base_title

    session_label = f"session: {session_id}"
    wrapped = textwrap.wrap(
        session_label,
        width=wrap_chars,
        break_long_words=True,
        break_on_hyphens=False,
    )
    if not wrapped:
        return base_title
    return f"{base_title}\n" + "\n".join(wrapped)


def _filter_description(metadata: dict) -> str:
    """Build a short string describing preprocessing filters applied to the data.

    Reports ``include_phases``, and ``filter_field`` + ``filter_values`` if present.
    Returns an empty string if none were applied.

    Args:
        metadata: MetricResult.metadata dict from a behavioral metric.

    Returns:
        Human-readable filter summary, or ``""`` if no filters were applied.
    """
    parts: list[str] = []

    # Regression uses train_/eval_ prefixed phase params;
    # classification uses unprefixed include_phases.
    train_phases = metadata.get("train_include_phases")
    eval_phases = metadata.get("eval_include_phases")
    include_phases = metadata.get("include_phases")

    if train_phases or eval_phases:
        # Regression: show both train and eval phases
        if train_phases:
            phases_str = ", ".join(str(p) for p in train_phases)
            parts.append(f"train phases: {phases_str}")
        if eval_phases:
            phases_str = ", ".join(str(p) for p in eval_phases)
            parts.append(f"eval phases: {phases_str}")
    elif include_phases:
        if isinstance(include_phases, (list, tuple)):
            phases_str = ", ".join(str(p) for p in include_phases)
        else:
            phases_str = str(include_phases)
        parts.append(f"phases: {phases_str}")

    filter_field = metadata.get("filter_field")
    filter_values = metadata.get("filter_values")
    if filter_field and filter_values:
        parts.append(f"{filter_field}={filter_values}")
    elif filter_field:
        parts.append(f"filter_field={filter_field}")

    return ", ".join(parts)


def _is_single_session(data: dict) -> bool:
    """Detect whether *data* is single-session or multi-session.

    Single-session layout: ``{tokenizer_name: {display_name: MetricResult}}``
    Multi-session layout:  ``{session_id: {tokenizer_name: {display_name: MetricResult}}}``

    Raises:
        ValueError: If the structure cannot be determined unambiguously.
    """
    if not data:
        return True

    first_val = next(iter(data.values()))
    if not first_val:
        # Empty inner dict — assume single-session
        return True

    first_inner = next(iter(first_val.values()))
    # Single-session: first_inner is a MetricResult
    # Multi-session:  first_inner is a dict (another level of nesting)
    return isinstance(first_inner, MetricResult)


def _iter_sessions(
    data: dict,
) -> list[tuple[str | None, dict[str, dict[str, MetricResult]]]]:
    """Return a list of ``(session_id, tokenizer_dict)`` pairs.

    For single-session data the session_id is ``None``.
    """
    if _is_single_session(data):
        return [(None, data)]
    return list(data.items())


def _collect_results(
    tokenizer_data: dict[str, dict[str, MetricResult]],
    metric_name_prefix: str | tuple[str, ...],
    result_name_filter: str | set[str] | frozenset[str],
    annotate_filters: bool = True,
) -> dict[str, dict[str, MetricResult]]:
    """Return ``{label: {tokenizer_name: MetricResult}}`` for a metric family.

    Only includes entries where ``result.name`` matches ``result_name_filter``.
    Strips ``metric_name_prefix`` from the display names.  When
    *annotate_filters* is ``True`` (default), appends a compact filter
    description to each label so bar-group labels are self-describing.
    """
    # Gather: display_name -> {tokenizer -> result}
    allowed_names = (
        {result_name_filter} if isinstance(result_name_filter, str) else set(result_name_filter)
    )
    collected: dict[str, dict[str, MetricResult]] = {}
    label_for_display: dict[str, str] = {}
    for tokenizer_name, display_dict in tokenizer_data.items():
        for display_name, result in display_dict.items():
            if not isinstance(result, MetricResult):
                raise TypeError(
                    f"Expected MetricResult for (tokenizer={tokenizer_name!r}, "
                    f"display_name={display_name!r}), got {type(result)}"
                )
            if result.name not in allowed_names:
                continue
            # Build label once per display_name, using target_field from metadata
            # as primary label (the display name may be uninformative, e.g. just
            # "behavioral_regression" without a suffix)
            if display_name not in label_for_display:
                target = result.metadata.get("target_field", "")
                short = _strip_metric_prefix(display_name, metric_name_prefix)
                # Prefer target_field if it's more informative than the stripped name
                if target and (short == display_name or not short):
                    short = target
                if annotate_filters:
                    filt = _filter_description(result.metadata)
                    if filt:
                        short = f"{short}\n({filt})"
                label_for_display[display_name] = short
            label = label_for_display[display_name]
            if label not in collected:
                collected[label] = {}
            collected[label][tokenizer_name] = result
    return collected


def _draw_grouped_bar(
    ax: Axes,
    grouped: dict[str, dict[str, MetricResult]],
    tokenizer_names: list[str],
    ylabel: str,
    ylim: tuple[float, float] | None = None,
    hline_per_group: dict[str, float] | None = None,
) -> None:
    """Draw a grouped bar chart on *ax*.

    Args:
        ax: Matplotlib axes to draw on.
        grouped: ``{short_metric_name: {tokenizer_name: MetricResult}}``.
        tokenizer_names: Ordered list of tokenizer names (determines bar order and colour).
        ylabel: Y-axis label.
        ylim: Optional (ymin, ymax) tuple.
        hline_per_group: Optional ``{short_metric_name: chance_level}`` — draws a dashed
            horizontal line at that y value for each metric group.
    """
    cmap = plt.cm.tab10
    colours = {name: cmap(i % 10) for i, name in enumerate(tokenizer_names)}

    metric_names = list(grouped.keys())
    n_groups = len(metric_names)
    n_bars = len(tokenizer_names)

    if n_groups == 0:
        ax.set_visible(False)
        return

    bar_width = 0.8 / max(n_bars, 1)
    x = np.arange(n_groups)

    for bar_idx, tok_name in enumerate(tokenizer_names):
        offsets = x + (bar_idx - (n_bars - 1) / 2) * bar_width
        values: list[float] = []
        yerrs: list[float] = []
        for mn in metric_names:
            if tok_name not in grouped[mn]:
                values.append(float("nan"))
                yerrs.append(float("nan"))
                continue
            result = grouped[mn][tok_name]
            try:
                v = float(result.raw_value)  # behavioral metrics are scalar
            except Exception:
                v = float("nan")
            values.append(v)

            # Error bars: use std across CV folds when available (n_splits>=3).
            meta = result.metadata or {}
            std = None
            if isinstance(meta.get("r2_std"), (int, float)) and meta.get("r2_std") is not None:
                std = float(meta["r2_std"])
            elif (
                isinstance(meta.get("accuracy_std"), (int, float))
                and meta.get("accuracy_std") is not None
            ):
                std = float(meta["accuracy_std"])
            else:
                # Fallback: compute std from fold lists if present.
                folds = meta.get("fold_r2s")
                if folds is None:
                    folds = meta.get("fold_accuracies")
                if isinstance(folds, (list, tuple, np.ndarray)) and len(folds) >= 3:
                    try:
                        fold_vals = np.array([float(x) for x in folds], dtype=float)
                        if np.isfinite(fold_vals).any():
                            std = float(np.nanstd(fold_vals))
                    except Exception:
                        std = None
            yerrs.append(std if std is not None else float("nan"))

        # Only show error bars if at least one std is finite.
        yerr_arr = np.array(yerrs, dtype=float)
        yerr = yerr_arr if np.isfinite(yerr_arr).any() else None
        bars = ax.bar(
            offsets,
            values,
            width=bar_width,
            label=tok_name,
            color=colours[tok_name],
            alpha=0.85,
            yerr=yerr,
            error_kw={"elinewidth": 1.0, "ecolor": "black", "capsize": 2},
        )
        # Annotate each bar with its numeric value
        for bar, val in zip(bars, values, strict=True):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=_BARPLOT_ANNOT_FONTSIZE,
                    clip_on=True,
                )

    ax.set_xticks(x)
    wrapped_metric_names = [_wrap_tick_label(name) for name in metric_names]
    ax.set_xticklabels(
        wrapped_metric_names, rotation=0, ha="center", fontsize=_BARPLOT_TICK_FONTSIZE
    )
    ax.set_ylabel(ylabel, fontsize=_BARPLOT_LABEL_FONTSIZE)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(fontsize=_BARPLOT_LEGEND_FONTSIZE)

    # Draw per-group chance lines if provided
    if hline_per_group:
        for grp_idx, mn in enumerate(metric_names):
            chance = hline_per_group.get(mn)
            if chance is not None:
                ax.hlines(
                    y=chance,
                    xmin=grp_idx - 0.45,
                    xmax=grp_idx + 0.45,
                    linestyles="dashed",
                    colors="grey",
                    linewidth=1.2,
                )


def _draw_sample_counts_regression(
    ax: Axes,
    grouped: dict[str, dict[str, MetricResult]],
    tokenizer_names: list[str],
) -> None:
    """Draw train/val sample counts for regression metrics as a grouped bar chart."""
    metric_names = list(grouped.keys())
    n_groups = len(metric_names)
    n_bars = len(tokenizer_names)

    if n_groups == 0:
        ax.set_visible(False)
        return

    cmap = plt.cm.tab10
    colours = {name: cmap(i % 10) for i, name in enumerate(tokenizer_names)}
    bar_width = 0.8 / max(n_bars, 1)
    x = np.arange(n_groups)
    any_test_counts = any(
        result.metadata.get("n_samples_test", 0) > 0
        for group in grouped.values()
        for result in group.values()
    )

    # Track max to create headroom so title never overlaps annotations.
    max_total = 0
    for bar_idx, tok_name in enumerate(tokenizer_names):
        offsets = x + (bar_idx - (n_bars - 1) / 2) * bar_width
        for grp_idx, mn in enumerate(metric_names):
            result = grouped[mn].get(tok_name)
            if result is None:
                continue
            meta = result.metadata
            n_train = meta.get("mean_samples_train", meta.get("n_samples_train", 0))
            n_val = meta.get("mean_samples_val", meta.get("n_samples_val", 0))
            n_test = meta.get("n_samples_test", 0)
            total = n_train + n_val + n_test
            max_total = max(max_total, int(total))
            ax.bar(
                offsets[grp_idx],
                n_train,
                width=bar_width,
                color=colours[tok_name],
                alpha=0.6,
                edgecolor="white",
                linewidth=0.5,
            )
            ax.bar(
                offsets[grp_idx],
                n_val,
                bottom=n_train,
                width=bar_width,
                color=colours[tok_name],
                alpha=0.3,
                edgecolor="white",
                linewidth=0.5,
                hatch="//",
            )
            if n_test:
                ax.bar(
                    offsets[grp_idx],
                    n_test,
                    bottom=n_train + n_val,
                    width=bar_width,
                    color=colours[tok_name],
                    alpha=0.18,
                    edgecolor="white",
                    linewidth=0.5,
                    hatch="xx",
                )
            # Vertically stagger every second annotation to avoid overlap.
            # Use a small offset proportional to axis scale.
            stagger = 0.0
            if bar_idx % 2 == 1:
                stagger = max(1.0, 0.03 * max_total)
            ax.text(
                offsets[grp_idx],
                total + stagger,
                f"{n_train}+{n_val}" if not n_test else f"{n_train}+{n_val}+{n_test}",
                ha="center",
                va="bottom",
                fontsize=_BARPLOT_ANNOT_FONTSIZE,
                rotation=90,
            )

    ax.set_xticks(x)
    wrapped_metric_names = [_wrap_tick_label(name) for name in metric_names]
    ax.set_xticklabels(
        wrapped_metric_names, rotation=0, ha="center", fontsize=_BARPLOT_TICK_FONTSIZE
    )
    ax.set_ylabel("# samples", fontsize=_BARPLOT_LABEL_FONTSIZE)
    title = "Sample counts (train + val)"
    if any_test_counts:
        title = "Sample counts (train + val + test)"
    ax.set_title(title, fontsize=_BARPLOT_TITLE_FONTSIZE, pad=16)
    if max_total > 0:
        ax.set_ylim(0, max_total * 1.35)


def _draw_sample_counts_classification(
    ax: Axes,
    grouped: dict[str, dict[str, MetricResult]],
    tokenizer_names: list[str],
) -> None:
    """Draw train/val(/test) sample counts for classification metrics.

    Mirrors `_draw_sample_counts_regression`: one grouped bar per metric, per tokenizer,
    stacked by train/val(/test) counts when present.
    """
    metric_names = list(grouped.keys())
    n_groups = len(metric_names)
    n_bars = len(tokenizer_names)

    if n_groups == 0:
        ax.set_visible(False)
        return

    cmap = plt.cm.tab10
    colours = {name: cmap(i % 10) for i, name in enumerate(tokenizer_names)}
    bar_width = 0.8 / max(n_bars, 1)
    x = np.arange(n_groups)

    any_test_counts = any(
        result.metadata.get("n_samples_test", 0) > 0
        for group in grouped.values()
        for result in group.values()
    )

    max_total = 0
    for bar_idx, tok_name in enumerate(tokenizer_names):
        offsets = x + (bar_idx - (n_bars - 1) / 2) * bar_width
        for grp_idx, mn in enumerate(metric_names):
            result = grouped[mn].get(tok_name)
            if result is None:
                continue
            meta = result.metadata
            n_train = meta.get("mean_samples_train", meta.get("n_samples_train", 0))
            n_val = meta.get("mean_samples_val", meta.get("n_samples_val", 0))
            n_test = meta.get("n_samples_test", 0)
            total = n_train + n_val + n_test
            max_total = max(max_total, int(total))

            ax.bar(
                offsets[grp_idx],
                n_train,
                width=bar_width,
                color=colours[tok_name],
                alpha=0.6,
                edgecolor="white",
                linewidth=0.5,
            )
            ax.bar(
                offsets[grp_idx],
                n_val,
                bottom=n_train,
                width=bar_width,
                color=colours[tok_name],
                alpha=0.3,
                edgecolor="white",
                linewidth=0.5,
                hatch="//",
            )
            if n_test:
                ax.bar(
                    offsets[grp_idx],
                    n_test,
                    bottom=n_train + n_val,
                    width=bar_width,
                    color=colours[tok_name],
                    alpha=0.18,
                    edgecolor="white",
                    linewidth=0.5,
                    hatch="xx",
                )

            stagger = 0.0
            if bar_idx % 2 == 1:
                stagger = max(1.0, 0.03 * max_total)
            ax.text(
                offsets[grp_idx],
                total + stagger,
                f"{n_train}+{n_val}" if not n_test else f"{n_train}+{n_val}+{n_test}",
                ha="center",
                va="bottom",
                fontsize=_BARPLOT_ANNOT_FONTSIZE,
                rotation=90,
            )

    ax.set_xticks(x)
    wrapped_metric_names = [_wrap_tick_label(name) for name in metric_names]
    ax.set_xticklabels(
        wrapped_metric_names, rotation=0, ha="center", fontsize=_BARPLOT_TICK_FONTSIZE
    )
    ax.set_ylabel("# samples", fontsize=_BARPLOT_LABEL_FONTSIZE)
    title = "Sample counts (train + val)"
    if any_test_counts:
        title = "Sample counts (train + val + test)"
    ax.set_title(title, fontsize=_BARPLOT_TITLE_FONTSIZE, pad=16)
    if max_total > 0:
        ax.set_ylim(0, max_total * 1.35)


def plot_regression_comparison(
    data: dict,
    figsize_per_subplot: tuple[float, float] = (10, 5),
) -> Figure:
    """Grouped bar chart comparing R² scores across tokenizers for each regression metric.

    Supports both single-session and multi-session inputs.  In the multi-session
    case one subplot column is drawn per session.

    Single-session layout::

        {tokenizer_name: {display_name: MetricResult, ...}, ...}

    Multi-session layout::

        {session_id: {tokenizer_name: {display_name: MetricResult, ...}, ...}, ...}

    Regression-like downstream metrics (ridge regression and temporal decoder)
    are included. Matching prefixes are stripped from x-axis tick labels.

    A second row of subplots shows train/val sample counts per metric.

    Args:
        data: Nested result dict (single- or multi-session, see above).
        figsize_per_subplot: (width, height) in inches for each subplot column.

    Returns:
        matplotlib Figure.
    """
    sessions = _iter_sessions(data)
    n_sessions = len(sessions)

    fig, axes = plt.subplots(
        2,
        n_sessions,
        figsize=(figsize_per_subplot[0] * n_sessions, figsize_per_subplot[1] * 1.65),
        squeeze=False,
        gridspec_kw={"height_ratios": [3.2, 1.4]},
    )

    all_tokenizer_names: list[str] = []
    for _, tok_dict in sessions:
        for tok_name in tok_dict:
            if tok_name not in all_tokenizer_names:
                all_tokenizer_names.append(tok_name)

    for col_idx, (session_id, tok_dict) in enumerate(sessions):
        ax_main = axes[0, col_idx]
        ax_counts = axes[1, col_idx]
        grouped = _collect_results(
            tok_dict,
            metric_name_prefix=REGRESSION_PREFIXES,
            result_name_filter=REGRESSION_RESULT_NAMES,
        )

        title = _format_subplot_title_with_session(
            "Behavioral Regression (R²)",
            session_id,
        )
        ax_main.set_title(title, fontsize=_BARPLOT_TITLE_FONTSIZE)

        _draw_grouped_bar(
            ax=ax_main,
            grouped=grouped,
            tokenizer_names=all_tokenizer_names,
            ylabel="R²",
            ylim=(-1, 1),
        )
        # Avoid duplicated long x tick labels between the two rows while keeping
        # them available for programmatic inspection in tests/debugging.
        _visually_hide_xticklabels(ax_main)
        # Put legend outside so it never covers bars.
        handles, labels = ax_main.get_legend_handles_labels()
        if handles:
            ax_main.legend(
                handles,
                labels,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
                frameon=False,
                fontsize=8,
            )

        _draw_sample_counts_regression(
            ax=ax_counts,
            grouped=grouped,
            tokenizer_names=all_tokenizer_names,
        )

    # fig.tight_layout()
    # Reserve right margin for the external legend.
    fig.subplots_adjust(hspace=0.40, wspace=0.25)
    fig.tight_layout(rect=(0, 0, 0.90, 0.97))
    return fig


def plot_classification_comparison(
    data: dict,
    figsize_per_subplot: tuple[float, float] = (10, 5),
) -> Figure:
    """Grouped bar chart comparing accuracy across tokenizers for each classification metric.

    Equivalent to :func:`plot_regression_comparison` but for classification results.
    A dashed horizontal line is drawn at the chance accuracy level for each metric
    (taken from ``metadata["chance_accuracy"]``).

    Only entries where ``result.name == "behavioral_classification"`` are included.
    The prefix ``"behavioral_classification_"`` is stripped from x-axis tick labels.

    A second row of subplots shows per-class sample counts (train/val).

    Args:
        data: Nested result dict (single- or multi-session).
        figsize_per_subplot: (width, height) in inches for each subplot column.

    Returns:
        matplotlib Figure.
    """
    sessions = _iter_sessions(data)
    n_sessions = len(sessions)

    fig, axes = plt.subplots(
        2,
        n_sessions,
        figsize=(figsize_per_subplot[0] * n_sessions, figsize_per_subplot[1] * 1.65),
        squeeze=False,
        gridspec_kw={"height_ratios": [3.0, 1.4]},
    )

    all_tokenizer_names: list[str] = []
    for _, tok_dict in sessions:
        for tok_name in tok_dict:
            if tok_name not in all_tokenizer_names:
                all_tokenizer_names.append(tok_name)

    for col_idx, (session_id, tok_dict) in enumerate(sessions):
        ax_main = axes[0, col_idx]
        ax_counts = axes[1, col_idx]
        grouped = _collect_results(
            tok_dict,
            metric_name_prefix="behavioral_classification_",
            result_name_filter="behavioral_classification",
        )

        # Collect chance accuracies per metric group (use first available result)
        hline_per_group: dict[str, float] = {}
        for short_name, metric_results in grouped.items():
            for result in metric_results.values():
                chance = result.metadata.get("chance_accuracy")
                if chance is not None:
                    hline_per_group[short_name] = float(chance)
                break

        title = _format_subplot_title_with_session(
            "Behavioral Classification (Accuracy)",
            session_id,
        )
        ax_main.set_title(title, fontsize=_BARPLOT_TITLE_FONTSIZE)

        _draw_grouped_bar(
            ax=ax_main,
            grouped=grouped,
            tokenizer_names=all_tokenizer_names,
            ylabel="Accuracy",
            ylim=(0, 1.05),
            hline_per_group=hline_per_group if hline_per_group else None,
        )
        # Avoid duplicated long x tick labels between the two rows while keeping
        # them available for programmatic inspection in tests/debugging.
        _visually_hide_xticklabels(ax_main)
        # Put legend outside so it never covers bars.
        handles, labels = ax_main.get_legend_handles_labels()
        if handles:
            ax_main.legend(
                handles,
                labels,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
                frameon=False,
                fontsize=8,
            )

        _draw_sample_counts_classification(
            ax=ax_counts,
            grouped=grouped,
            tokenizer_names=all_tokenizer_names,
        )

    # fig.tight_layout()
    # Reserve right margin for the external legend.
    fig.subplots_adjust(hspace=0.42, wspace=0.25)
    fig.tight_layout(rect=(0, 0, 0.90, 0.97))
    return fig


def plot_per_class_accuracy(
    data: dict,
    metric_display_name: str,
    figsize_per_subplot: tuple[float, float] = (10, 5),
) -> Figure | None:
    """Per-class accuracy barplot for a single classification metric.

    Returns ``None`` when ``metadata["balance_classes"]`` is ``False`` for the
    first available result, because per-class accuracy is only meaningful when
    classes are balanced.

    Annotates each bar with the accuracy percentage **and** the mean validation
    sample count (from ``metadata["val_class_counts"]``).

    Args:
        data: Nested result dict (single- or multi-session).
        metric_display_name: The display-name key used for the metric in *data*
            (i.e. the key inside the per-tokenizer dict, before prefix stripping).
        figsize_per_subplot: (width, height) in inches per subplot column.

    Returns:
        matplotlib Figure, or ``None`` if ``balance_classes`` is ``False``.
    """
    sessions = _iter_sessions(data)

    # Check balance_classes on first result we can find for this metric
    for _, tok_dict in sessions:
        for display_dict in tok_dict.values():
            result = display_dict.get(metric_display_name)
            if result is not None and isinstance(result, MetricResult):
                if not result.metadata.get("balance_classes", False):
                    return None
                break
        break

    n_sessions = len(sessions)

    fig, axes = plt.subplots(
        1,
        n_sessions,
        figsize=(figsize_per_subplot[0] * n_sessions, figsize_per_subplot[1]),
        squeeze=False,
    )

    all_tokenizer_names: list[str] = []
    for _, tok_dict in sessions:
        for tok_name in tok_dict:
            if tok_name not in all_tokenizer_names:
                all_tokenizer_names.append(tok_name)

    cmap = plt.cm.tab10
    colours = {name: cmap(i % 10) for i, name in enumerate(all_tokenizer_names)}

    for col_idx, (session_id, tok_dict) in enumerate(sessions):
        ax = axes[0, col_idx]

        # Gather per-class accuracy for each tokenizer
        # {tokenizer_name: {class_label: accuracy}}
        per_class: dict[str, dict[Any, float]] = {}
        val_counts: dict[str, dict[Any, int]] = {}

        for tok_name in all_tokenizer_names:
            display_dict = tok_dict.get(tok_name, {})
            result = display_dict.get(metric_display_name)
            if result is None or not isinstance(result, MetricResult):
                continue
            pc_acc = result.metadata.get("per_class_accuracy")
            if pc_acc is None:
                continue
            per_class[tok_name] = {k: float(v) for k, v in pc_acc.items()}
            vc = result.metadata.get("val_class_counts")
            if vc is not None:
                val_counts[tok_name] = {k: int(v) for k, v in vc.items()}

        # Union of all class labels across tokenizers
        all_classes: list[Any] = sorted(
            {cls for pc in per_class.values() for cls in pc.keys()},
            key=lambda c: (str(type(c)), c),
        )

        if not all_classes:
            ax.set_visible(False)
            continue

        n_groups = len(all_classes)
        n_bars = len(all_tokenizer_names)
        bar_width = 0.8 / max(n_bars, 1)
        x = np.arange(n_groups)

        for bar_idx, tok_name in enumerate(all_tokenizer_names):
            if tok_name not in per_class:
                continue
            offsets = x + (bar_idx - (n_bars - 1) / 2) * bar_width
            values = [per_class[tok_name].get(cls, float("nan")) for cls in all_classes]
            bars = ax.bar(
                offsets,
                values,
                width=bar_width,
                label=tok_name,
                color=colours[tok_name],
                alpha=0.85,
            )
            for bar, cls, val in zip(bars, all_classes, values, strict=True):
                if np.isnan(val):
                    continue
                count = val_counts.get(tok_name, {}).get(cls)
                label_text = f"{val:.0%}"
                if count is not None:
                    label_text += f"\n(n={count})"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    label_text,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    clip_on=True,
                )

        ax.set_xticks(x)
        # For movement_phase, replace integer labels with phase names
        target_field = None
        first_result = None
        for tok_name in all_tokenizer_names:
            r = tok_dict.get(tok_name, {}).get(metric_display_name)
            if r is not None and isinstance(r, MetricResult):
                target_field = r.metadata.get("target_field")
                first_result = r
                break

        # Build tick labels with total sample counts per class
        train_cc = first_result.metadata.get("train_class_counts", {}) if first_result else {}
        val_cc = first_result.metadata.get("val_class_counts", {}) if first_result else {}
        if target_field == "movement_phase":
            from ntt.core.data.components.behavior_transforms import (
                MOVEMENT_PHASE_LABELS_REVERSE,
            )

            class_names = [
                MOVEMENT_PHASE_LABELS_REVERSE.get(int(float(c)), str(c)) for c in all_classes
            ]
        else:
            class_names = [str(c) for c in all_classes]

        tick_labels = []
        for cls, name in zip(all_classes, class_names, strict=False):
            n_train = train_cc.get(cls, train_cc.get(str(cls), 0))
            n_val = val_cc.get(cls, val_cc.get(str(cls), 0))
            tick_labels.append(f"{name}\n(n={n_train + n_val})")
        ax.set_xticklabels(tick_labels, rotation=90, ha="center", fontsize=9)
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=8)

        title = f"Per-class Accuracy — {metric_display_name}"
        if session_id is not None:
            title = f"{title} — {session_id}"
        ax.set_title(title, fontsize=10)

    fig.tight_layout()
    return fig


def plot_trajectory_2d(
    result: MetricResult,
    *,
    connect_gt_pred: bool = False,
    figsize_per_trial: tuple[float, float] = (4, 4),
    session_id: str | None = None,
    tokenizer_name: str | None = None,
) -> Figure | None:
    """2D trajectory overlay: ground truth vs predicted cursor position.

    Reads trial data from ``result.metadata["sample_trials"]``.  Each entry
    must have keys ``"trial_id"`` (int), ``"timestamps"`` (list[float]),
    ``"y_true"`` (list[list[float]]), and ``"y_pred"`` (same shape).

    Args:
        result: MetricResult with ``metadata["sample_trials"]`` populated.
        connect_gt_pred: If ``True``, draw thin dashed lines from each predicted
            point to its corresponding ground-truth point.
        figsize_per_trial: (width, height) in inches per subplot.
        session_id: Optional session identifier added to the suptitle.

    Returns:
        matplotlib Figure, or ``None`` if no sample_trials are present.
    """
    sample_trials = result.metadata.get("sample_trials")
    if not sample_trials:
        return None

    target_field = result.metadata.get("target_field", "unknown")
    n_trials = len(sample_trials)
    n_cols = min(n_trials, 4)
    n_rows = (n_trials + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_trial[0] * n_cols, figsize_per_trial[1] * n_rows),
        squeeze=False,
    )

    for i, trial in enumerate(sample_trials):
        row, col = divmod(i, n_cols)
        ax = axes[row][col]

        trial_id = trial["trial_id"]
        y_true = np.array(trial["y_true"])
        y_pred = np.array(trial["y_pred"])

        # Handle 1D targets
        if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
            ax.text(
                0.5,
                0.5,
                "1D target",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
            )
            ax.set_title(f"Trial {trial_id}")
            continue

        assert y_true.shape[1] >= 2, (
            f"Expected at least 2D y_true, got shape {y_true.shape} for trial {trial_id}"
        )

        # Ground truth: solid line with small markers
        ax.plot(
            y_true[:, 0],
            y_true[:, 1],
            "o-",
            color="C0",
            markersize=3,
            linewidth=1.5,
            alpha=0.8,
            label="Ground truth" if i == 0 else None,
        )

        # Predicted: scatter only
        ax.scatter(
            y_pred[:, 0],
            y_pred[:, 1],
            color="C1",
            s=15,
            alpha=0.8,
            label="Predicted" if i == 0 else None,
        )

        # Optional connecting lines between each pred and GT point
        if connect_gt_pred:
            for pt_true, pt_pred in zip(y_true, y_pred, strict=False):
                ax.plot(
                    [pt_pred[0], pt_true[0]],
                    [pt_pred[1], pt_true[1]],
                    "k--",
                    linewidth=0.5,
                    alpha=0.3,
                )

        # Start/end markers for GT (filled) and predicted (open)
        ax.scatter(
            y_true[0, 0],
            y_true[0, 1],
            color="green",
            marker="s",
            s=30,
            zorder=5,
            label="GT start" if i == 0 else None,
        )
        ax.scatter(
            y_true[-1, 0],
            y_true[-1, 1],
            color="red",
            marker="^",
            s=30,
            zorder=5,
            label="GT end" if i == 0 else None,
        )
        ax.scatter(
            y_pred[0, 0],
            y_pred[0, 1],
            facecolors="none",
            edgecolors="green",
            marker="s",
            s=30,
            linewidths=1.2,
            zorder=5,
            label="Pred start" if i == 0 else None,
        )
        ax.scatter(
            y_pred[-1, 0],
            y_pred[-1, 1],
            facecolors="none",
            edgecolors="red",
            marker="^",
            s=30,
            linewidths=1.2,
            zorder=5,
            label="Pred end" if i == 0 else None,
        )

        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"Trial {trial_id}")

        if i == 0:
            ax.legend(fontsize=7, ncol=2)

    # Hide unused subplots
    for j in range(n_trials, n_rows * n_cols):
        row, col = divmod(j, n_cols)
        axes[row][col].set_visible(False)

    suptitle = f"2D Trajectory: {target_field}"
    if session_id is not None:
        suptitle = f"{suptitle} — {session_id}"
    fig.suptitle(suptitle, fontsize=11)
    annotation = _build_annotation_text(
        result, session_id=session_id, tokenizer_name=tokenizer_name
    )
    fig.text(0.5, -0.02, annotation, ha="center", fontsize=8, style="italic")
    fig.tight_layout()
    return fig


def plot_trajectory_3d(
    result: MetricResult,
    *,
    figsize_per_trial: tuple[float, float] = (5, 5),
    session_id: str | None = None,
    tokenizer_name: str | None = None,
) -> Figure | None:
    """3D trajectory overlay: x, y position with time as the z-axis.

    Reads trial data from ``result.metadata["sample_trials"]``.  Each entry
    must have keys ``"trial_id"`` (int), ``"timestamps"`` (list[float]),
    ``"y_true"`` (list[list[float]]), and ``"y_pred"`` (same shape).

    Args:
        result: MetricResult with ``metadata["sample_trials"]`` populated.
        figsize_per_trial: (width, height) in inches per subplot.
        session_id: Optional session identifier added to the suptitle.

    Returns:
        matplotlib Figure, or ``None`` if no sample_trials are present.
    """
    sample_trials = result.metadata.get("sample_trials")
    if not sample_trials:
        return None

    target_field = result.metadata.get("target_field", "unknown")
    n_trials = len(sample_trials)
    n_cols = min(n_trials, 4)
    n_rows = (n_trials + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(figsize_per_trial[0] * n_cols, figsize_per_trial[1] * n_rows))

    for i, trial in enumerate(sample_trials):
        ax: Any = fig.add_subplot(n_rows, n_cols, i + 1, projection="3d")

        trial_id = trial["trial_id"]
        y_true = np.array(trial["y_true"])
        y_pred = np.array(trial["y_pred"])
        timestamps = np.array(trial["timestamps"])

        if y_true.ndim < 2 or y_true.shape[1] < 2:
            # 1D targets (e.g. trial_target_dir) can't be plotted as 3D trajectories
            plt.close(fig)
            return None

        # Ground truth: solid line only, no markers
        ax.plot(
            y_true[:, 0],
            y_true[:, 1],
            timestamps,
            "-",
            color="C0",
            linewidth=1.5,
            alpha=0.8,
        )

        # Predicted: scatter only
        ax.scatter(
            y_pred[:, 0],
            y_pred[:, 1],
            timestamps,
            color="C1",
            s=15,
            alpha=0.8,
        )

        # Start/end markers for GT (filled) and predicted (open)
        ax.scatter(
            [y_true[0, 0]],
            [y_true[0, 1]],
            [timestamps[0]],
            color="green",
            marker="s",
            s=40,
            zorder=5,
        )
        ax.scatter(
            [y_true[-1, 0]],
            [y_true[-1, 1]],
            [timestamps[-1]],
            color="red",
            marker="^",
            s=40,
            zorder=5,
        )
        ax.scatter(
            [y_pred[0, 0]],
            [y_pred[0, 1]],
            [timestamps[0]],
            facecolors="none",
            edgecolors="green",
            marker="s",
            s=40,
            linewidths=1.2,
            zorder=5,
        )
        ax.scatter(
            [y_pred[-1, 0]],
            [y_pred[-1, 1]],
            [timestamps[-1]],
            facecolors="none",
            edgecolors="red",
            marker="^",
            s=40,
            linewidths=1.2,
            zorder=5,
        )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("time (s)")
        ax.set_title(f"Trial {trial_id}")

    suptitle = f"3D Trajectory: {target_field}"
    if session_id is not None:
        suptitle = f"{suptitle} — {session_id}"
    fig.suptitle(suptitle, fontsize=11)
    annotation = _build_annotation_text(
        result, session_id=session_id, tokenizer_name=tokenizer_name
    )
    fig.text(0.5, -0.02, annotation, ha="center", fontsize=8, style="italic")
    fig.tight_layout()
    return fig


def _build_annotation_text(
    result: MetricResult,
    *,
    session_id: str | None = None,
    tokenizer_name: str | None = None,
) -> str:
    """Build a compact annotation string for trajectory plots."""
    parts: list[str] = []
    target = result.metadata.get("target_field", "unknown")
    parts.append(f"target: {target}")
    if session_id is not None:
        parts.append(f"session: {session_id}")
    if tokenizer_name is not None:
        parts.append(f"tokenizer: {tokenizer_name}")
    filt = _filter_description(result.metadata)
    if filt:
        parts.append(f"filters: {filt}")
    parts.append(f"{result.name}: {result.raw_value:.4f}")
    return "  |  ".join(parts)


def plot_trajectory_2d_multi(
    result: MetricResult,
    *,
    connect_gt_pred: bool = False,
    figsize: tuple[float, float] = (8, 7),
    session_id: str | None = None,
    tokenizer_name: str | None = None,
) -> Figure | None:
    """2D multi-trial trajectory: ground truth vs predicted, with trial boundaries.

    Reads concatenated trial data from ``result.metadata["sample_trials_concatenated"]``.

    Args:
        result: MetricResult with ``metadata["sample_trials_concatenated"]``.
        connect_gt_pred: If ``True``, draw dashed lines from pred to GT.
        figsize: Figure size in inches.
        session_id: Optional session identifier for annotation.
        tokenizer_name: Optional tokenizer name for annotation.

    Returns:
        matplotlib Figure, or ``None`` if no concatenated data is present.
    """
    concat = result.metadata.get("sample_trials_concatenated")
    if not concat:
        return None

    y_true = np.array(concat["y_true"])
    y_pred = np.array(concat["y_pred"])
    boundaries = concat["trial_boundaries"]
    trial_ids = concat.get("trial_ids", [])

    if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] < 2):
        return None

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Ground truth trajectory
    ax.plot(
        y_true[:, 0],
        y_true[:, 1],
        "o-",
        color="C0",
        markersize=2,
        linewidth=1.2,
        alpha=0.8,
        label="Ground truth",
    )
    # Predicted trajectory
    ax.plot(
        y_pred[:, 0],
        y_pred[:, 1],
        "s-",
        color="C1",
        markersize=2,
        linewidth=1.0,
        alpha=0.7,
        label="Predicted",
    )

    if connect_gt_pred:
        for pt_true, pt_pred in zip(y_true, y_pred, strict=False):
            ax.plot(
                [pt_pred[0], pt_true[0]],
                [pt_pred[1], pt_true[1]],
                "k--",
                linewidth=0.3,
                alpha=0.2,
            )

    # Mark trial boundaries
    for b_idx, boundary in enumerate(boundaries[1:-1], start=1):
        # Draw a marker at the boundary point on the GT trajectory
        ax.axvline(x=y_true[boundary, 0], color="grey", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.plot(
            y_true[boundary, 0],
            y_true[boundary, 1],
            "D",
            color="grey",
            markersize=6,
            zorder=5,
            alpha=0.7,
        )
        if b_idx - 1 < len(trial_ids):
            ax.annotate(
                f"T{trial_ids[b_idx - 1]}"
                if isinstance(trial_ids[b_idx - 1], int)
                else f"trial {b_idx}",
                (y_true[boundaries[b_idx - 1], 0], y_true[boundaries[b_idx - 1], 1]),
                fontsize=7,
                alpha=0.6,
            )

    # Start/end markers for GT (filled) and predicted (open)
    ax.scatter(
        y_true[0, 0], y_true[0, 1], color="green", marker="s", s=40, zorder=5, label="GT start"
    )
    ax.scatter(
        y_true[-1, 0], y_true[-1, 1], color="red", marker="^", s=40, zorder=5, label="GT end"
    )
    ax.scatter(
        y_pred[0, 0],
        y_pred[0, 1],
        facecolors="none",
        edgecolors="green",
        marker="s",
        s=40,
        linewidths=1.2,
        zorder=5,
        label="Pred start",
    )
    ax.scatter(
        y_pred[-1, 0],
        y_pred[-1, 1],
        facecolors="none",
        edgecolors="red",
        marker="^",
        s=40,
        linewidths=1.2,
        zorder=5,
        label="Pred end",
    )

    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=7, ncol=2)

    annotation = _build_annotation_text(
        result, session_id=session_id, tokenizer_name=tokenizer_name
    )
    fig.suptitle(f"Multi-trial 2D Trajectory ({len(boundaries) - 1} trials)", fontsize=11)
    fig.text(0.5, -0.02, annotation, ha="center", fontsize=8, style="italic")
    fig.tight_layout()
    return fig


def plot_trajectory_3d_multi(
    result: MetricResult,
    *,
    figsize: tuple[float, float] = (10, 8),
    session_id: str | None = None,
    tokenizer_name: str | None = None,
) -> Figure | None:
    """3D multi-trial trajectory with time as z-axis and trial boundary markers.

    Reads concatenated trial data from ``result.metadata["sample_trials_concatenated"]``.

    Args:
        result: MetricResult with ``metadata["sample_trials_concatenated"]``.
        figsize: Figure size in inches.
        session_id: Optional session identifier for annotation.
        tokenizer_name: Optional tokenizer name for annotation.

    Returns:
        matplotlib Figure, or ``None`` if no concatenated data is present.
    """
    concat = result.metadata.get("sample_trials_concatenated")
    if not concat:
        return None

    y_true = np.array(concat["y_true"])
    y_pred = np.array(concat["y_pred"])
    timestamps = np.array(concat["timestamps"])
    boundaries = concat["trial_boundaries"]

    if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] < 2):
        return None

    fig = plt.figure(figsize=figsize)
    ax: Any = fig.add_subplot(111, projection="3d")

    # Ground truth
    ax.plot(
        y_true[:, 0],
        y_true[:, 1],
        timestamps,
        "-",
        color="C0",
        linewidth=1.2,
        alpha=0.8,
        label="Ground truth",
    )
    # Predicted
    ax.scatter(
        y_pred[:, 0],
        y_pred[:, 1],
        timestamps,
        color="C1",
        s=10,
        alpha=0.7,
        label="Predicted",
    )

    # Mark trial boundaries
    for boundary in boundaries[1:-1]:
        ax.scatter(
            [y_true[boundary, 0]],
            [y_true[boundary, 1]],
            [timestamps[boundary]],
            color="grey",
            marker="D",
            s=30,
            zorder=5,
            alpha=0.7,
        )

    # Start/end markers for GT (filled) and predicted (open)
    ax.scatter(
        [y_true[0, 0]],
        [y_true[0, 1]],
        [timestamps[0]],
        color="green",
        marker="s",
        s=50,
        zorder=5,
        label="GT start",
    )
    ax.scatter(
        [y_true[-1, 0]],
        [y_true[-1, 1]],
        [timestamps[-1]],
        color="red",
        marker="^",
        s=50,
        zorder=5,
        label="GT end",
    )
    ax.scatter(
        [y_pred[0, 0]],
        [y_pred[0, 1]],
        [timestamps[0]],
        facecolors="none",
        edgecolors="green",
        marker="s",
        s=50,
        linewidths=1.2,
        zorder=5,
        label="Pred start",
    )
    ax.scatter(
        [y_pred[-1, 0]],
        [y_pred[-1, 1]],
        [timestamps[-1]],
        facecolors="none",
        edgecolors="red",
        marker="^",
        s=50,
        linewidths=1.2,
        zorder=5,
        label="Pred end",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("time (s)")
    ax.legend(fontsize=7, ncol=2)

    annotation = _build_annotation_text(
        result, session_id=session_id, tokenizer_name=tokenizer_name
    )
    fig.suptitle(f"Multi-trial 3D Trajectory ({len(boundaries) - 1} trials)", fontsize=11)
    fig.text(0.5, -0.02, annotation, ha="center", fontsize=8, style="italic")
    fig.tight_layout()
    return fig


def write_behavioral_summary_table(
    data: dict[str, dict[str, MetricResult]],
    output_path: str | Path,
    session_id: str | None = None,
) -> None:
    """Write a plain text summary table: tokenizers as columns, behavioral metrics as rows.

    Args:
        data: ``{tokenizer_name: {display_name: MetricResult, ...}, ...}``.
            Only results where ``result.name`` is ``"behavioral_regression"`` or
            ``"behavioral_classification"`` are included.
        output_path: Destination path for the text file.
        session_id: Optional session identifier appended to the header line.
    """
    tokenizer_names = list(data.keys())

    # Collect all metric display names (preserving insertion order across tokenizers)
    all_metric_names: list[str] = []
    seen: set[str] = set()
    for tok_name in tokenizer_names:
        for display_name, result in data[tok_name].items():
            if result.name in BEHAVIORAL_NAMES and display_name not in seen:
                all_metric_names.append(display_name)
                seen.add(display_name)

    # Build informative labels first so we can size columns correctly
    metric_labels: dict[str, str] = {}
    for display_name in all_metric_names:
        label = display_name
        first_result = None
        for tok in tokenizer_names:
            first_result = data[tok].get(display_name)
            if first_result is not None:
                break
        if first_result is not None:
            target = first_result.metadata.get("target_field", "")
            filt = _filter_description(first_result.metadata)
            if target:
                label = f"{display_name} ({target})"
            if filt:
                label += f" ({filt})"
        metric_labels[display_name] = label

    # Column widths
    metric_col_width = (
        max((len(label) for label in metric_labels.values()), default=len("Metric")) + 2
    )
    metric_col_width = max(metric_col_width, len("Metric") + 2)
    tok_col_widths = {tok: max(len(tok), 16) + 2 for tok in tokenizer_names}

    # Header
    title = "Behavioral Metrics Summary"
    if session_id is not None:
        title = f"{title} — {session_id}"

    lines: list[str] = [title, "=" * len(title), ""]

    # Table header row
    header = "Metric".ljust(metric_col_width)
    for tok in tokenizer_names:
        header += tok.ljust(tok_col_widths[tok])
    lines.append(header)
    lines.append("-" * len(header))

    # Metric rows
    for display_name in all_metric_names:
        row = metric_labels[display_name].ljust(metric_col_width)
        for tok in tokenizer_names:
            result = data[tok].get(display_name)
            if result is None:
                cell = "N/A"
            else:
                cell = f"{result.raw_value:.4f}"
                if result.name == "behavioral_classification":
                    chance = result.metadata.get("chance_accuracy")
                    if chance is not None:
                        cell += f" (chance={chance:.3f})"
            row += cell.ljust(tok_col_widths[tok])
        lines.append(row)

    Path(output_path).write_text("\n".join(lines) + "\n")


def extract_behavioral_results(
    session_reports: dict[str, list[tuple[Any, dict[str, Any]]]],
) -> tuple[dict[str, MetricResult], dict[str, dict[str, MetricResult]]]:
    """Extract behavioral metric results from the session report structure.

    Args:
        session_reports: ``{session_id: [(segment, {"report": EvaluationReport}), ...]}``,
            as produced by ``run_validation_evaluation_per_session``.

    Returns:
        A tuple ``(flat_results, per_session_results)`` where:
        - ``flat_results``: ``{display_name: MetricResult}`` — first occurrence across all
          sessions (useful for an aggregated / single-session view).
        - ``per_session_results``: ``{session_id: {display_name: MetricResult}}`` — per-session
          breakdown.
    """
    flat_results: dict[str, MetricResult] = {}
    per_session_results: dict[str, dict[str, MetricResult]] = {}

    for session_id, segment_list in session_reports.items():
        session_dict: dict[str, MetricResult] = {}
        for _segment, meta in segment_list:
            report = meta["report"]
            for display_name, result in report.results.items():
                if result.name not in BEHAVIORAL_NAMES:
                    continue
                session_dict[display_name] = result
                if display_name not in flat_results:
                    flat_results[display_name] = result
        per_session_results[session_id] = session_dict

    return flat_results, per_session_results


def _save_behavioral_figure(fig: Figure, path: Path, *, dpi: int) -> None:
    """Save, close, and log a behavioral figure."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.debug("Saved: %s", path)


def _generate_behavioral_comparison_plots(
    *,
    session_id: str,
    session_tok_data: dict[str, dict[str, MetricResult]],
    session_bhav: Path,
    display_names: list[str],
    plot_fn: Any,
    filename_prefix: str,
    dpi: int,
    trained_checkpoint_caption: str | None,
    data_filter: Any = None,
    generate_combined: bool = False,
) -> None:
    """Generate chunk-aware behavioral comparison plots for one session."""
    if not display_names:
        return

    def _filtered_plot_data(
        selected_session_tok_data: dict[str, dict[str, MetricResult]],
    ) -> dict[str, dict[str, dict[str, MetricResult]]]:
        wrapped = {session_id: selected_session_tok_data}
        return data_filter(wrapped) if data_filter is not None else wrapped

    def _save_plot(fig: Figure, filename: str) -> None:
        if trained_checkpoint_caption:
            _apply_trained_checkpoint_suptitle(fig, trained_checkpoint_caption)
        path = session_bhav / _shorten_behavioral_plot_identifier(filename)
        _save_behavioral_figure(fig, path, dpi=dpi)

    chunk_windows = _extract_chunk_windows_from_results(session_tok_data, display_names)
    if chunk_windows:
        for cw in chunk_windows:
            session_tok_data_cw = _select_chunk_window_in_session_tok_data(
                session_tok_data,
                chunk_window_ms=cw,
            )
            plot_data_cw = _filtered_plot_data(session_tok_data_cw)

            if generate_combined:
                fig = plot_fn(plot_data_cw)
                _save_plot(
                    fig,
                    f"{filename_prefix}_comparison_cw{_format_chunk_window_ms(cw)}ms.png",
                )

            for display_name in display_names:
                per_metric_data = _filter_plot_data_by_display_names(
                    plot_data_cw,
                    {display_name},
                )
                fig = plot_fn(
                    per_metric_data,
                    figsize_per_subplot=_PER_METRIC_FIGSIZE,
                )
                _save_plot(
                    fig,
                    (f"{filename_prefix}_{display_name}_cw{_format_chunk_window_ms(cw)}ms.png"),
                )
        return

    plot_data = _filtered_plot_data(session_tok_data)
    if generate_combined:
        any_result = _first_metric_result_for_display_name(session_tok_data, display_names[0])
        chunk_suffix = _behavioral_chunk_suffix(getattr(any_result, "metadata", None))
        fig = plot_fn(plot_data)
        _save_plot(fig, f"{filename_prefix}_comparison{chunk_suffix}.png")

    for display_name in display_names:
        per_metric_data = _filter_plot_data_by_display_names(plot_data, {display_name})
        any_metric = _first_metric_result_for_display_name(session_tok_data, display_name)
        chunk_suffix = _behavioral_chunk_suffix(getattr(any_metric, "metadata", None))
        fig = plot_fn(
            per_metric_data,
            figsize_per_subplot=_PER_METRIC_FIGSIZE,
        )
        _save_plot(fig, f"{filename_prefix}_{display_name}{chunk_suffix}.png")


def generate_all_behavioral_plots(
    tokenizer_session_reports: dict[str, dict[str, list[tuple[Any, dict[str, Any]]]]],
    output_dir: str | Path,
    *,
    dpi: int = 150,
    trained_checkpoint_caption: str | None = None,
) -> None:
    """Generate all behavioral plots for one or more tokenizers and save to disk.

    This is the single entry-point that replaces the inline plotting loop
    (section 8b) in the evaluation script.

    Args:
        tokenizer_session_reports: ``{tokenizer_display_name: session_reports}``
            where ``session_reports`` is the standard
            ``{session_id: [(segment, {"report": EvaluationReport}), ...]}`` dict.
        output_dir: Run output root. All behavioral artifacts for a session live under
            ``output_dir/sessions/{session_id}/bhav/`` (comparison PNGs, summary table);
            trajectories under ``.../bhav/trajectories/``.
        dpi: Resolution for saved figures.
        trained_checkpoint_caption: If set (e.g. resolved ``.ckpt`` path), shown as a
            wrapped multi-line ``fig.suptitle`` on regression/classification/per-class
            comparison figures.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract flat / per-session behavioral results for each tokenizer
    flat_by_tok: dict[str, dict[str, MetricResult]] = {}
    per_session_by_tok: dict[str, dict[str, dict[str, MetricResult]]] = {}
    for tok_name, session_reports in tokenizer_session_reports.items():
        flat, per_session = extract_behavioral_results(session_reports)
        flat_by_tok[tok_name] = flat
        per_session_by_tok[tok_name] = per_session

    all_session_ids: set[str] = set()
    for per_session in per_session_by_tok.values():
        all_session_ids |= set(per_session.keys())

    combined_per_session: dict[str, dict[str, dict[str, MetricResult]]] = {}
    for sid in all_session_ids:
        combined_per_session[sid] = {
            tok_name: per_session_by_tok[tok_name].get(sid, {})
            for tok_name in tokenizer_session_reports
        }

    if not flat_by_tok:
        return

    # Metric-selection sets are unioned across ALL tokenizers: a target produced by
    # any tokenizer must be plotted, even if the first-iterated tokenizer skipped it
    # (e.g. a metric that raised and was dropped for one tokenizer, or a baseline-only
    # decoder). The first occurrence wins for representative metadata.
    union_flat: dict[str, MetricResult] = {}
    for tok_flat in flat_by_tok.values():
        for display_name, result in tok_flat.items():
            union_flat.setdefault(display_name, result)

    balanced_present = any(
        r.name == "behavioral_classification" and bool(r.metadata.get("balance_classes", False))
        for r in union_flat.values()
    )
    unbalanced_present = any(
        r.name == "behavioral_classification"
        and not bool(r.metadata.get("balance_classes", False))
        for r in union_flat.values()
    )
    regression_display_names = [
        dn for dn, r in union_flat.items() if r.name in REGRESSION_RESULT_NAMES
    ]
    unbalanced_display_names = [
        dn
        for dn, r in union_flat.items()
        if r.name == "behavioral_classification"
        and not bool(r.metadata.get("balance_classes", False))
    ]
    balanced_display_names = [
        dn
        for dn, r in union_flat.items()
        if r.name == "behavioral_classification" and bool(r.metadata.get("balance_classes", False))
    ]

    for sid in sorted(all_session_ids):
        # Skip sessions with no behavioral results for any tokenizer.
        session_tok_data = combined_per_session[sid]
        if not any(results for results in session_tok_data.values()):
            continue

        session_bhav = output_dir / "sessions" / sid / "bhav"
        session_bhav.mkdir(parents=True, exist_ok=True)

        if regression_display_names:
            _generate_behavioral_comparison_plots(
                session_id=sid,
                session_tok_data=session_tok_data,
                session_bhav=session_bhav,
                display_names=regression_display_names,
                plot_fn=plot_regression_comparison,
                filename_prefix="behavioral_regression",
                dpi=dpi,
                trained_checkpoint_caption=trained_checkpoint_caption,
                generate_combined=True,
            )

        if unbalanced_present or balanced_present:
            if unbalanced_present:
                # Intentionally skip the combined classification comparison plot:
                # it merges multiple targets into one figure and is easy to misinterpret.
                _generate_behavioral_comparison_plots(
                    session_id=sid,
                    session_tok_data=session_tok_data,
                    session_bhav=session_bhav,
                    display_names=unbalanced_display_names,
                    plot_fn=plot_classification_comparison,
                    filename_prefix="behavioral_classification",
                    dpi=dpi,
                    trained_checkpoint_caption=trained_checkpoint_caption,
                    data_filter=lambda plot_data: _filter_classification_by_balance(
                        plot_data,
                        include_balanced=False,
                    ),
                )

            if balanced_present:
                _generate_behavioral_comparison_plots(
                    session_id=sid,
                    session_tok_data=session_tok_data,
                    session_bhav=session_bhav,
                    display_names=balanced_display_names,
                    plot_fn=plot_classification_comparison,
                    filename_prefix="behavioral_classification_balanced",
                    dpi=dpi,
                    trained_checkpoint_caption=trained_checkpoint_caption,
                    data_filter=lambda plot_data: _filter_classification_by_balance(
                        plot_data,
                        include_balanced=True,
                    ),
                )

        wrapped = {sid: session_tok_data}
        for display_name, result in union_flat.items():
            if result.name == "behavioral_classification" and result.metadata.get(
                "balance_classes"
            ):
                fig = plot_per_class_accuracy(wrapped, display_name)
                if fig is not None:
                    if trained_checkpoint_caption:
                        _apply_trained_checkpoint_suptitle(fig, trained_checkpoint_caption)
                    safe_name = display_name.replace("/", "_")
                    filename = _shorten_behavioral_plot_identifier(f"per_class_{safe_name}.png")
                    path = session_bhav / filename
                    fig.savefig(path, dpi=dpi, bbox_inches="tight")
                    plt.close(fig)
                    logger.debug("Saved: %s", path)

        traj_dir = session_bhav / "trajectories"
        traj_dir.mkdir(parents=True, exist_ok=True)
        session_label = sid.split("/")[-1] if "/" in sid else sid
        for tok_name, tok_per_session in per_session_by_tok.items():
            session_results = tok_per_session.get(sid, {})
            tok_suffix = _tokenizer_plot_filename_suffix(tok_name)
            for display_name, result in session_results.items():
                if result.name in REGRESSION_RESULT_NAMES and result.metadata.get("sample_trials"):
                    chunk_suffix = _behavioral_chunk_suffix(result.metadata)
                    safe_name = f"{display_name}_{session_label}_{tok_suffix}".replace("/", "_")
                    safe_name = f"{safe_name}{chunk_suffix}"
                    safe_name = _shorten_behavioral_plot_identifier(safe_name)

                    fig_2d = plot_trajectory_2d(
                        result,
                        connect_gt_pred=True,
                        session_id=session_label,
                        tokenizer_name=tok_name,
                    )
                    if fig_2d is not None:
                        path = traj_dir / f"trajectory_2d_{safe_name}.png"
                        fig_2d.savefig(path, dpi=dpi, bbox_inches="tight")
                        plt.close(fig_2d)
                        logger.debug("Saved: %s", path)

                    fig_3d = plot_trajectory_3d(
                        result,
                        session_id=session_label,
                        tokenizer_name=tok_name,
                    )
                    if fig_3d is not None:
                        path = traj_dir / f"trajectory_3d_{safe_name}.png"
                        fig_3d.savefig(path, dpi=dpi, bbox_inches="tight")
                        plt.close(fig_3d)
                        logger.debug("Saved: %s", path)

                    if result.metadata.get("sample_trials_concatenated"):
                        fig_2d_multi = plot_trajectory_2d_multi(
                            result,
                            connect_gt_pred=True,
                            session_id=session_label,
                            tokenizer_name=tok_name,
                        )
                        if fig_2d_multi is not None:
                            path = traj_dir / f"trajectory_2d_multi_{safe_name}.png"
                            fig_2d_multi.savefig(path, dpi=dpi, bbox_inches="tight")
                            plt.close(fig_2d_multi)
                            logger.debug("Saved: %s", path)

                        fig_3d_multi = plot_trajectory_3d_multi(
                            result,
                            session_id=session_label,
                            tokenizer_name=tok_name,
                        )
                        if fig_3d_multi is not None:
                            path = traj_dir / f"trajectory_3d_multi_{safe_name}.png"
                            fig_3d_multi.savefig(path, dpi=dpi, bbox_inches="tight")
                            plt.close(fig_3d_multi)
                            logger.debug("Saved: %s", path)

        per_session_flat = {
            tok_name: combined_per_session[sid][tok_name] for tok_name in tokenizer_session_reports
        }
        filename = _shorten_behavioral_plot_identifier("behavioral_summary.txt")
        table_path = session_bhav / filename
        write_behavioral_summary_table(per_session_flat, table_path, session_id=sid)
        logger.debug("Saved: %s", table_path)

    logger.info(
        "Behavioral plots saved for %d session(s) → %s",
        len(all_session_ids),
        output_dir / "sessions",
    )
