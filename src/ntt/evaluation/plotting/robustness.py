"""Plotting helpers for the robustness metrics (M1 / latent_robustness (M2)).

Plot definitions M1-a, M1-c, M2-a and M2-b for the input_robustness (M1) and
latent_robustness (M2) metrics.

Two shared primitives are defined here:

- :func:`plot_metric_curve` — line + shaded band (or errorbars), used
  by M1-a and M2-a.
- :func:`plot_spike_raster` — thin wrapper that delegates to the
  canonical helper ``plot_binned_spike_raster`` in
  :mod:`ntt.evaluation.plotting.spike_raster`. That module is the
  single source of truth for spike rasters. The wrapper here adds the
  eval-specific concerns (mode handling, ``sort_by_rate``, titles /
  subtitles) on top of the canonical raster draw.

All public functions return a ``matplotlib.figure.Figure``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from .io import save_fig_multi

# Type aliases — kept loose because this module is purely a presentation
# layer and is meant to be tolerant of different upstream payload shapes.
_Curve = Mapping[str, Sequence[float]]
_M1Result = Mapping[str, _Curve]


# ---------------------------------------------------------------------- styling defaults
# Raster styling shared with the reconstruction visualizations so eval rasters
# look consistent with them.
_COLOR_CLEAN = "k"
_COLOR_NOISY = "tab:red"
_RASTER_LINEWIDTH = 0.6
_RASTER_LINELENGTH = 0.7
_ANNOT_BBOX = {"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85}
_TOKENIZER_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "*", "8")
_TOKENIZER_COLOR_CYCLE = tuple(np.vstack([plt.cm.tab20.colors, plt.cm.tab20b.colors]))  # type: ignore[name-defined]


# ---------------------------------------------------------------------- shared


def _npz_safe_key(name: str) -> str:
    """Return a filesystem-friendly token for dynamic ``.npz`` keys."""
    chars: list[str] = []
    for ch in name:
        if ch.isalnum() or ch == "_":
            chars.append(ch)
        elif ch == ".":
            chars.append("p")
        elif ch == "-":
            chars.append("m")
        else:
            chars.append("_")
    return "".join(chars).strip("_") or "value"


def _raster_source_array(value) -> np.ndarray:
    """Convert a raster source value to ``numpy`` or fail loudly."""
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    raise TypeError(
        "_persist_raster_source_arrays: expected numpy.ndarray or torch.Tensor, "
        f"got {type(value)!r}"
    )


def _persist_raster_source_arrays(stem: Path, payload: dict) -> Path:
    """Persist raster source arrays and scalar metadata beside a figure."""
    stem = Path(stem).with_suffix("")
    out_path = stem.with_suffix(".npz")

    serialized: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        if isinstance(value, np.ndarray | torch.Tensor):
            serialized[key] = _raster_source_array(value)
            continue
        if isinstance(value, dict):
            serialized[key] = np.asarray(json.dumps(value, sort_keys=True))
            continue
        if isinstance(value, str | bytes | bool | int | float | np.generic):
            serialized[key] = np.asarray(value)
            continue
        if isinstance(value, list | tuple):
            if all(
                isinstance(item, str | bytes | bool | int | float | np.generic) for item in value
            ):
                serialized[key] = np.asarray(value)
                continue
            raise TypeError(
                "_persist_raster_source_arrays: expected numpy.ndarray or torch.Tensor, "
                f"got {type(value)!r}"
            )
        raise TypeError(
            "_persist_raster_source_arrays: expected numpy.ndarray or torch.Tensor, "
            f"got {type(value)!r}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **serialized)  # pyright: ignore[reportArgumentType]  # **kwargs vs allow_pickle stub
    return out_path.resolve()


def plot_metric_curve(
    ax,
    xs: Sequence[float] | np.ndarray,
    ys_mean: Sequence[float] | np.ndarray,
    ys_std: Sequence[float] | np.ndarray,
    *,
    label: str | None = None,
    shade: bool = True,
    color: str | tuple[float, float, float] | None = None,
    marker: str | None = None,
):
    """Draw ``ys_mean`` over ``xs`` with a ``±ys_std`` band or errorbars.

    Empty inputs are tolerated — the call becomes a no-op so callers can
    safely invoke this on degenerate / skipped slices without guarding.

    Args:
        ax: target ``matplotlib`` axes.
        xs: x positions (length N).
        ys_mean: per-x mean of the metric (length N).
        ys_std: per-x std of the metric (length N).
        label: optional legend label for the line.
        shade: when True, draw a translucent ``±std`` band around the
            line; when False, draw vertical errorbars instead.

    Returns:
        The same ``ax`` (for ergonomic chaining).

    """
    xs_arr = np.asarray(xs, dtype=float)
    ys_arr = np.asarray(ys_mean, dtype=float)
    std_arr = np.asarray(ys_std, dtype=float)

    if xs_arr.size == 0 or ys_arr.size == 0:
        return ax

    if xs_arr.shape != ys_arr.shape or xs_arr.shape != std_arr.shape:
        raise ValueError(
            "plot_metric_curve: shape mismatch "
            f"xs={xs_arr.shape}, ys_mean={ys_arr.shape}, ys_std={std_arr.shape}"
        )

    if marker is None:
        marker = "o"

    # A shaded band is only visually meaningful when the curve spans at
    # least two x positions. For single-point curves (for example the
    # input_robustness (M1) ``permute_neurons`` panel), fall back to explicit error bars so
    # the uncertainty remains visible.
    use_shaded_band = shade and xs_arr.size > 1

    if use_shaded_band:
        (line,) = ax.plot(
            xs_arr,
            ys_arr,
            marker=marker,
            linewidth=1.5,
            label=label,
            color=color,
        )
        ax.fill_between(
            xs_arr,
            ys_arr - std_arr,
            ys_arr + std_arr,
            alpha=0.25,
            color=line.get_color(),
        )
    else:
        ax.errorbar(
            xs_arr,
            ys_arr,
            yerr=std_arr,
            marker=marker,
            capsize=3,
            linewidth=1.5,
            label=label,
            color=color,
        )
    return ax


def _style_for_index(idx: int) -> tuple[tuple[float, float, float], str]:
    """Return a stable ``(color, marker)`` pair for tokenizer index ``idx``.

    Uses a large qualitative palette and delays marker reuse until at
    least 12 distinct marker shapes have been consumed.
    """
    color = _TOKENIZER_COLOR_CYCLE[idx % len(_TOKENIZER_COLOR_CYCLE)]
    marker = _TOKENIZER_MARKERS[idx % len(_TOKENIZER_MARKERS)]
    return color, marker


def _set_exact_xticks(
    ax, xs_by_curve: Sequence[Sequence[float] | np.ndarray], *, log_scale: bool = False
):
    """Restrict x ticks to the observed x values present in the subplot.

    This keeps robustness sweeps readable for sparse epsilon / strength
    grids and prevents matplotlib from inventing intermediate ticks that
    do not correspond to actual evaluated points.
    """
    tick_vals: set[float] = set()
    for xs in xs_by_curve:
        arr = np.asarray(xs, dtype=float)
        arr = arr[np.isfinite(arr)]
        tick_vals.update(arr.tolist())

    if not tick_vals:
        return

    ordered = sorted(tick_vals)
    if log_scale:
        ordered = [x for x in ordered if x > 0]
        if not ordered:
            return
        ax.set_xscale("log")

    ax.set_xticks(ordered)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))


def plot_spike_raster(
    ax,
    spikes: np.ndarray,
    *,
    mode: str = "single",
    bin_size_ms: float = 1.0,
    spike_height: float | None = None,
    linewidth: float = 0.8,
    color: str | None = None,
    title: str | None = None,
    subtitle: str | None = None,
    sort_by_rate: bool = False,
    unit_order: np.ndarray | None = None,
):
    """Render a binned-spike raster on ``ax``.

    Renders the binned-spike array as a vector-friendly ``eventplot``
    (one row per unit, one tick per non-zero bin). The ``linelengths`` /
    ``linewidths`` / color conventions match the reconstruction raster
    plots, so eval-time rasters look consistent with them.

    Args:
        ax: target ``matplotlib`` axes.
        spikes: ``[T, U]`` array of binned spike counts (>0 ⇒ a spike
            tick is drawn).
        mode: ``"single"`` for a single-neuron view (taller bars,
            collapses ``U=1`` cleanly); ``"population"`` for the
            standard multi-unit raster.
        bin_size_ms: width of one time bin in milliseconds. The x-axis
            is rendered in seconds.
        spike_height: vertical extent of each tick (in unit-rows).
            Defaults to 1.6 for ``mode="single"`` (large bar) and
            ``_RASTER_LINELENGTH`` (0.7) for ``mode="population"``.
        linewidth: tick line width.
        color: tick color. Defaults to the module-level
            ``_COLOR_CLEAN`` ("k"). Pass ``_COLOR_NOISY`` ("tab:red")
            for noisy/corrupted rasters.
        title: optional axes title (drawn small, fontsize=9).
        subtitle: optional second-line annotation drawn just under the
            title via ``fig.text`` semantics — placed inside the title
            string with a newline (kept on the axes title rather than
            ``fig.text`` so it follows the panel in multi-panel grids).
        sort_by_rate: when True (and ``mode="population"``) reorder
            unit rows by descending firing rate.
        unit_order: optional explicit unit ordering (length-``U`` index
            array). When provided, overrides ``sort_by_rate`` and
            renders rows in the given order. Use this to keep a
            consistent ordering across multiple panels (e.g. clean vs.
            corrupted) — compute the order once from the clean panel
            and pass it to both, otherwise per-panel rate-sorting can
            place the same unit on different rows.

    Returns:
        The same ``ax``.

    """
    if mode not in {"single", "population"}:
        raise ValueError(f"mode must be 'single' or 'population'; got {mode!r}")

    arr = np.asarray(spikes)
    if arr.ndim == 1:
        # Treat 1-D as a single neuron over time → reshape to [T, 1].
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"plot_spike_raster: expected [T, U] (or 1-D); got shape {arr.shape}")
    T, U = arr.shape

    if spike_height is None:
        spike_height = 1.6 if mode == "single" else _RASTER_LINELENGTH
    if color is None:
        color = _COLOR_CLEAN

    # Reorder rows: explicit ``unit_order`` wins (so callers can keep
    # the ordering consistent across clean / corrupted panels);
    # otherwise fall back to the ``sort_by_rate`` ordering.
    if unit_order is not None:
        order_arr = np.asarray(unit_order)
        if order_arr.ndim != 1 or order_arr.size != U:
            raise ValueError(
                f"plot_spike_raster: unit_order must be 1-D with length U={U}; "
                f"got shape {order_arr.shape}"
            )
        arr = arr[:, order_arr]
    elif sort_by_rate and mode == "population" and U > 1:
        rates = arr.sum(axis=0)
        order = np.argsort(rates)[::-1]
        arr = arr[:, order]

    # Delegate the raster draw itself to the canonical notebook helper
    # so notebooks and the eval module render identically. Imported lazily
    # to keep eval-time imports cheap for callers that don't actually
    # invoke this function.
    from ntt.evaluation.plotting.spike_raster import plot_binned_spike_raster

    plot_binned_spike_raster(
        ax,
        arr,
        bin_size_ms=bin_size_ms,
        spike_height=spike_height,
        linewidth=linewidth,
        color=color,
    )

    # Sparser y-ticks for population rasters.
    if mode == "population" and U > 20:
        step = max(1, U // 20)
        ax.set_yticks(np.arange(0, U, step))
        ax.tick_params(axis="y", labelsize=6)

    if title is not None or subtitle is not None:
        text = title or ""
        if subtitle:
            text = f"{text}\n{subtitle}" if text else subtitle
        ax.set_title(text, fontsize=9)
    return ax


# ---------------------------------------------------------------------- input_corruption_a


def _coerce_results_list(
    results: _M1Result | Sequence[_M1Result],
) -> list[_M1Result]:
    """Normalise the M1-a payload to a list of per-tokenizer dicts."""
    if isinstance(results, Mapping):
        return [results]
    return list(results)


def plot_input_corruption_a(
    results: _M1Result | Sequence[_M1Result],
    ax=None,
    tokenizers: Sequence[str] | None = None,
):
    """Plot M1-a: corruption strength vs. normalized latent squared distance.

    One subplot per corruption type / label. When multiple tokenizers
    are supplied (one ``results`` mapping per tokenizer), each is drawn
    as its own line (mean ± std band).

    Args:
        results: either a single ``{corruption_label → {"strengths": [...],
            "means": [...], "stds": [...]}}`` mapping, or a sequence of
            such mappings (one per tokenizer).
        ax: optional pre-existing axes. When provided and there is a
            single corruption label, the curve is drawn there. When
            multiple labels are present and ``ax`` is given, this is
            an error (caller must let us build the figure).
        tokenizers: optional per-tokenizer legend labels. Length must
            match the number of result mappings; defaults to
            ``"tokenizer_{i}"``.

    Returns:
        The constructed (or reused) ``matplotlib.figure.Figure``.

    """
    import matplotlib.pyplot as plt

    results_list = _coerce_results_list(results)
    if not results_list:
        # Fail loud rather than silently producing an empty figure.
        raise ValueError("plot_input_corruption_a: no results provided.")

    if tokenizers is None:
        tokenizers = [f"tokenizer_{i}" for i in range(len(results_list))]
    if len(tokenizers) != len(results_list):
        raise ValueError(
            f"tokenizers length {len(tokenizers)} != results length {len(results_list)}"
        )

    # Union of corruption labels across tokenizers, preserving the
    # order of the first occurrence.
    seen: dict[str, None] = {}
    for r in results_list:
        for label in r.keys():
            seen.setdefault(label, None)
    labels = list(seen.keys())
    if not labels:
        raise ValueError("plot_input_corruption_a: results contain no corruption labels.")

    multi_tokenizer = len(results_list) > 1

    if ax is not None:
        if len(labels) > 1:
            raise ValueError(
                "plot_input_corruption_a: cannot reuse a single `ax` with multiple "
                f"corruption labels ({labels}); pass ax=None."
            )
        fig = ax.figure
        axes_list = [ax]
        legend_ax_idx: int | None = None
    else:
        # Reserve one extra subplot slot for a dedicated legend panel
        # when there are multiple subplots and multiple tokenizers — the
        # user explicitly asked for a single legend rather than one per
        # subplot.
        n = len(labels)
        want_legend_panel = multi_tokenizer and n > 1
        n_panels = n + (1 if want_legend_panel else 0)
        ncols = min(n_panels, 3)
        nrows = int(np.ceil(n_panels / ncols))
        fig, axes_arr = plt.subplots(
            nrows, ncols, figsize=(5.0 * ncols, 3.5 * nrows), squeeze=False
        )
        axes_list = list(axes_arr.ravel())
        legend_ax_idx = n if want_legend_panel else None

    for i, label in enumerate(labels):
        sub_ax = axes_list[i]
        xs_for_ticks: list[np.ndarray] = []
        for tok_idx, (tok_label, r) in enumerate(zip(tokenizers, results_list, strict=False)):
            if label not in r:
                continue
            curve = r[label]
            xs = np.asarray(curve.get("strengths", []), dtype=float)
            xs_for_ticks.append(xs)
            color, marker = _style_for_index(tok_idx)
            plot_metric_curve(
                sub_ax,
                xs=xs,
                ys_mean=curve.get("means", []),
                ys_std=curve.get("stds", []),
                label=tok_label,
                shade=True,
                color=color,
                marker=marker,
            )
        sub_ax.set_title(label)
        sub_ax.set_xlabel("corruption strength")
        sub_ax.set_ylabel("normalized latent MSE (σ_token^2 units)")
        sub_ax.grid(True, alpha=0.3)
        _set_exact_xticks(sub_ax, xs_for_ticks)
        # Per-subplot legend only when we don't have a dedicated panel.
        if multi_tokenizer and legend_ax_idx is None:
            sub_ax.legend(loc="best", fontsize=9)

    # Dedicated legend panel: gather handles from the first populated
    # axes, then render them in their own axes (no axes ticks).
    if legend_ax_idx is not None and legend_ax_idx < len(axes_list):
        legend_ax = axes_list[legend_ax_idx]
        legend_ax.axis("off")
        handles, leg_labels = [], []
        for sub_ax in axes_list[: len(labels)]:
            h, lab = sub_ax.get_legend_handles_labels()
            for hh, ll in zip(h, lab, strict=False):
                if ll not in leg_labels:
                    handles.append(hh)
                    leg_labels.append(ll)
        if handles:
            legend_ax.legend(handles, leg_labels, loc="center", fontsize=9, frameon=False)

    # Hide any leftover subplots we created (when len(panels) is not a
    # rectangular multiple of ncols).
    last_used = len(labels) + (1 if legend_ax_idx is not None else 0)
    for j in range(last_used, len(axes_list)):
        axes_list[j].axis("off")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------- per-dim distribution


def plot_input_corruption_per_dim_distribution(
    sigma_per_dim_sq: Sequence[float],
    per_dim_nmse_breakdown: Sequence[float],
    *,
    corruption_label: str,
    session_id: str,
    tokenizer_label: str | None = None,
    ax=None,
):
    """Per-dim distribution of clean variance vs. corruption MSE.

    For one corruption (e.g. middle jitter strength), shows how the
    unnormalised per-dim mean squared error (``MSE_d = NMSE_d · σ_d²``)
    distributes across latent dimensions when those dims are sorted by
    decreasing variance. Helps disambiguate "i.i.d.-like" corruption
    (where MSE_d roughly tracks σ_d²) from off-manifold corruption
    (where MSE_d stays flat as σ_d² drops by orders of magnitude — the
    corruption is concentrating energy in low-variance dims).

    Args:
        sigma_per_dim_sq: ``[D]`` array of per-dim variances of the clean
            latents (positional index = original dim index).
        per_dim_nmse_breakdown: ``[D]`` array of per-dim NORMALISED MSE
            (NMSE_d, mean over windows × replicas) for the corruption of
            interest. Multiplied by ``sigma_per_dim_sq`` here to recover
            the unnormalised per-dim MSE shown on the secondary axis.
        corruption_label: Spec label, used for title.
        session_id: Session id, used for title.
        tokenizer_label: Optional, used for title.
        ax: Optional pre-existing axes; figure created if None.

    Returns:
        The constructed (or reused) ``matplotlib.figure.Figure``.

    """
    import matplotlib.pyplot as plt

    sigma_sq = np.asarray(sigma_per_dim_sq, dtype=float)
    nmse_d = np.asarray(per_dim_nmse_breakdown, dtype=float)
    if sigma_sq.shape != nmse_d.shape:
        raise ValueError(
            f"per-dim shape mismatch: σ²[D]={sigma_sq.shape} vs NMSE_d[D]={nmse_d.shape}"
        )
    mse_unnorm = nmse_d * sigma_sq  # [D]; the actual per-dim error in latent units

    # Sort dims by decreasing variance.
    order = np.argsort(-sigma_sq)
    sigma_sq_s = sigma_sq[order]
    mse_unnorm_s = mse_unnorm[order]
    x = np.arange(len(order))

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
    else:
        fig = ax.figure

    # Use log scale only when values span > 1 decade — otherwise linear is
    # easier to read.
    def _y_scale(arr: np.ndarray) -> str:
        finite = arr[np.isfinite(arr) & (arr > 0)]
        if finite.size < 2:
            return "linear"
        ratio = finite.max() / finite.min()
        return "log" if ratio >= 10 else "linear"

    color_var = "tab:blue"
    color_mse = "tab:orange"

    ax.scatter(x, sigma_sq_s, s=12, color=color_var, label="σ_d² (variance)")
    ax.set_xlabel("dim index (sorted by decreasing variance)")
    ax.set_ylabel("σ_d² (clean variance)", color=color_var)
    ax.tick_params(axis="y", labelcolor=color_var)
    ax.set_yscale(_y_scale(sigma_sq_s))

    ax2 = ax.twinx()
    ax2.scatter(
        x, mse_unnorm_s, s=12, color=color_mse, marker="x", label="MSE_d (per-dim corruption MSE)"
    )
    ax2.set_ylabel("per-dim MSE (unnormalised)", color=color_mse)
    ax2.tick_params(axis="y", labelcolor=color_mse)
    ax2.set_yscale(_y_scale(mse_unnorm_s))

    title_bits = []
    if tokenizer_label:
        title_bits.append(tokenizer_label)
    title_bits += [corruption_label, f"session={session_id}"]
    ax.set_title("  |  ".join(title_bits), fontsize=10)

    # Combined legend.
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=9)

    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------- latent_corruption_a


def plot_latent_corruption_a(
    results_csv_path: Path | str,
    ax=None,
    metrics: list[str] | None = None,
):
    """Plot M2-a: ε vs. reconstruction-metric mean ± std.

    Reads the ``results.csv`` produced by
    :class:`ntt.jobs.eval_robustness_latent.NTTEvalRobustnessLatentJob`.
    The CSV is in the wide-after-aggregation form emitted by the job's
    ``_write_csv`` — pandas-aggregated columns ``{metric}_mean`` /
    ``{metric}_std`` keyed on ``epsilon``. When pandas is unavailable
    at write time the job falls back to per-(window, ε, k) rows; this
    function aggregates on-the-fly in that case.

    Args:
        results_csv_path: path to ``results.csv``.
        ax: optional pre-existing axes (used only when a single metric
            is plotted).
        metrics: list of metric basenames (without ``_mean``/``_std``)
            to plot. Defaults to ``["acg_correlation", "ccg_correlation"]``.

    Returns:
        The constructed (or reused) ``matplotlib.figure.Figure``.

    """
    import matplotlib.pyplot as plt
    import pandas as pd

    if metrics is None:
        metrics = ["acg_correlation", "ccg_correlation"]

    df = pd.read_csv(Path(results_csv_path))
    if df.empty:
        raise ValueError(f"plot_latent_corruption_a: {results_csv_path} is empty.")

    # Detect schema: aggregated (has *_mean columns) vs. raw rows.
    has_aggregated = any(c.endswith("_mean") for c in df.columns)
    if "tokenizer_id" in df.columns:
        tokenizer_groups = list(df.groupby("tokenizer_id", sort=True))
    else:
        tokenizer_groups = [(None, df)]

    if ax is not None and len(metrics) > 1:
        raise ValueError(
            "plot_latent_corruption_a: cannot reuse a single `ax` with multiple metrics; "
            "pass ax=None."
        )

    multi_tokenizer = len(tokenizer_groups) > 1

    if ax is not None:
        fig = ax.figure
        axes_list = [ax]
        legend_ax_idx: int | None = None
    else:
        # Dedicated legend panel when we have multiple subplots and
        # multiple tokenizers (otherwise the legend just lives in-axes).
        want_legend_panel = multi_tokenizer and len(metrics) > 1
        n_panels = len(metrics) + (1 if want_legend_panel else 0)
        ncols = min(n_panels, 2)
        nrows = int(np.ceil(n_panels / ncols))
        fig, axes_arr = plt.subplots(
            nrows, ncols, figsize=(5.5 * ncols, 4.0 * nrows), squeeze=False
        )
        axes_list = list(axes_arr.ravel())
        legend_ax_idx = len(metrics) if want_legend_panel else None

    # Larger fontsizes for joint figures (axes labels, titles, legend).
    _TITLE_FONTSIZE = 13
    _LABEL_FONTSIZE = 12
    _TICK_FONTSIZE = 11
    _LEGEND_FONTSIZE = 11

    for i, metric_name in enumerate(metrics):
        sub_ax = axes_list[i]
        xs_for_ticks: list[np.ndarray] = []
        for tok_idx, (tok_id, sub_df) in enumerate(tokenizer_groups):
            tok_label = str(tok_id) if tok_id is not None else None
            xs, means, stds = _extract_eps_curve(sub_df, metric_name, has_aggregated)
            if len(xs) == 0:
                continue
            xs_for_ticks.append(np.asarray(xs, dtype=float))
            color, marker = _style_for_index(tok_idx)
            plot_metric_curve(
                sub_ax,
                xs,
                means,
                stds,
                label=tok_label,
                shade=True,
                color=color,
                marker=marker,
            )
        sub_ax.set_title(metric_name, fontsize=_TITLE_FONTSIZE)
        sub_ax.set_xlabel("ε (in units of σ_token)", fontsize=_LABEL_FONTSIZE)
        sub_ax.set_ylabel(metric_name, fontsize=_LABEL_FONTSIZE)
        sub_ax.tick_params(axis="both", which="major", labelsize=_TICK_FONTSIZE)
        sub_ax.grid(True, alpha=0.3)
        # Log-x is the natural default for ε sweeps but only if all ε > 0.
        eps_vals = np.unique(df["epsilon"].to_numpy()) if "epsilon" in df.columns else np.array([])
        if eps_vals.size and np.all(eps_vals > 0):
            _set_exact_xticks(sub_ax, xs_for_ticks, log_scale=True)
        else:
            _set_exact_xticks(sub_ax, xs_for_ticks)
        # Per-subplot legend only when there's no dedicated panel.
        if multi_tokenizer and legend_ax_idx is None:
            sub_ax.legend(loc="best", fontsize=_LEGEND_FONTSIZE)

    if legend_ax_idx is not None and legend_ax_idx < len(axes_list):
        legend_ax = axes_list[legend_ax_idx]
        legend_ax.axis("off")
        handles, leg_labels = [], []
        for sub_ax in axes_list[: len(metrics)]:
            h, lab = sub_ax.get_legend_handles_labels()
            for hh, ll in zip(h, lab, strict=False):
                if ll not in leg_labels:
                    handles.append(hh)
                    leg_labels.append(ll)
        if handles:
            legend_ax.legend(
                handles,
                leg_labels,
                loc="center",
                fontsize=_LEGEND_FONTSIZE,
                frameon=False,
            )

    last_used = len(metrics) + (1 if legend_ax_idx is not None else 0)
    for j in range(last_used, len(axes_list)):
        axes_list[j].axis("off")

    fig.tight_layout()
    return fig


def _extract_eps_curve(
    df,
    metric_name: str,
    has_aggregated: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(xs, means, stds)`` for ``metric_name`` from the latent_robustness (M2) CSV.

    Handles both the pandas-aggregated wide schema (``{metric}_mean``
    + ``{metric}_std`` keyed per ``epsilon``) and the raw per-(window,
    ε, k) rows (one column per metric). For the latter we group by ε
    and take the mean / std across rows.

    """
    if "epsilon" not in df.columns:
        raise ValueError(
            "plot_latent_corruption_a: results.csv missing required column 'epsilon'."
        )

    if has_aggregated and f"{metric_name}_mean" in df.columns:
        std_col = f"{metric_name}_std"
        # Aggregated rows may still have multiple per-segment rows; take
        # the per-ε mean of the means and pool stds via mean-of-stds
        # (good enough for a quick-look plot).
        agg = df.groupby("epsilon", as_index=False)[
            [f"{metric_name}_mean"] + ([std_col] if std_col in df.columns else [])
        ].mean()
        xs = agg["epsilon"].to_numpy()
        means = agg[f"{metric_name}_mean"].to_numpy()
        stds = agg[std_col].to_numpy() if std_col in df.columns else np.zeros_like(means)
        return xs, means, stds

    if metric_name not in df.columns:
        return np.array([]), np.array([]), np.array([])

    grouped = df.groupby("epsilon", as_index=False)[metric_name].agg(["mean", "std"])
    grouped = grouped.reset_index()
    xs = grouped["epsilon"].to_numpy()
    means = grouped["mean"].to_numpy()
    stds = grouped["std"].fillna(0.0).to_numpy()
    return xs, means, stds


# ---------------------------------------------------------------------- input_corruption_c


def plot_input_corruption_c(
    window_idx: int,
    *,
    clean_spikes: np.ndarray,
    corrupted_spikes_per_label: Mapping[str, np.ndarray],
    latent_distance_per_label: Mapping[str, float],
    ax=None,
    bin_size_ms: float = 1.0,
    sort_by_rate: bool = True,
    session_id: str | None = None,
    output_stem: Path | str | None = None,
):
    """Plot M1-c: spike-train raster annotated with M1-a numbers.

    Layout: one panel for the clean window plus one panel per
    corruption label (drawn by :func:`plot_spike_raster`). The clean
    panel uses ``_COLOR_CLEAN`` and corrupted panels use
    ``_COLOR_NOISY``. Annotations (M1-a normalized latent distance)
    are placed inside each panel as a rounded text-box rather than
    crammed into the title.

    Args:
        window_idx: window identifier (used only for the figure title).
        clean_spikes: ``[T, U]`` clean binned-spike array.
        corrupted_spikes_per_label: mapping from corruption label to
            ``[T, U]`` corrupted binned-spike array (input-side, i.e.
            *not* tokenizer reconstructions — raw corruptions).
        latent_distance_per_label: per-label normalized latent distance.
        ax: optional pre-existing axes (only honored when there is
            exactly one corruption label).
        bin_size_ms: width of one time bin in milliseconds (forwarded
            to :func:`plot_spike_raster` so the x-axis is in seconds).
        sort_by_rate: when ``True`` and the data is multi-unit, sort
            rows by descending firing rate.

    Returns:
        The constructed (or reused) ``matplotlib.figure.Figure``.

    """
    import matplotlib.pyplot as plt

    labels = list(corrupted_spikes_per_label.keys())
    if not labels:
        raise ValueError("plot_input_corruption_c: corrupted_spikes_per_label is empty.")

    raster_mode = "single" if clean_spikes.shape[-1] == 1 else "population"

    # Compute the unit ordering once from the clean panel and reuse it for
    # every corrupted panel. This avoids the bug where each panel sorts
    # independently and small per-corruption count changes (e.g. boundary
    # clipping in ``apply_jitter``) caused the same unit to sit on
    # different rows across panels — visually confusable with a real
    # signal shift. ``None`` when sort_by_rate is off so callers' previous
    # behaviour is preserved.
    shared_unit_order: np.ndarray | None = None
    if sort_by_rate and raster_mode == "population" and clean_spikes.shape[-1] > 1:
        clean_rates = np.asarray(clean_spikes).sum(axis=0)
        shared_unit_order = np.argsort(clean_rates)[::-1]

    if ax is not None:
        if len(labels) != 1:
            raise ValueError(
                "plot_input_corruption_c: cannot reuse a single `ax` with multiple labels; "
                "pass ax=None."
            )
        # Single-axes mode draws only the corrupted panel on the given ax.
        # The clean reference is omitted in this restricted mode.
        fig = ax.figure
        label = labels[0]
        plot_spike_raster(
            ax,
            corrupted_spikes_per_label[label],
            mode=raster_mode,
            bin_size_ms=bin_size_ms,
            color=_COLOR_NOISY,
            sort_by_rate=sort_by_rate,
            unit_order=shared_unit_order,
        )
        d_lat = latent_distance_per_label.get(label, float("nan"))
        ax.set_title(label, fontsize=9)
        ax.text(
            0.97,
            0.03,
            f"d_latent={d_lat:.3g}",
            transform=ax.transAxes,
            fontsize=8,
            ha="right",
            va="bottom",
            bbox=_ANNOT_BBOX,
        )
        ax.set_xlabel("time (s)")
        return fig

    # Multi-row layout: one row per corruption label, with clean on
    # the left of each row for direct visual comparison.
    height_per_row = 3.5 if raster_mode == "population" else 2.0
    n_rows = len(labels)
    fig, axes_arr = plt.subplots(
        n_rows,
        2,
        figsize=(7.0, height_per_row * n_rows),
        sharey=True,
        squeeze=False,
    )

    for r, label in enumerate(labels):
        clean_ax = axes_arr[r, 0]
        noisy_ax = axes_arr[r, 1]

        plot_spike_raster(
            clean_ax,
            clean_spikes,
            mode=raster_mode,
            bin_size_ms=bin_size_ms,
            color=_COLOR_CLEAN,
            sort_by_rate=sort_by_rate,
            unit_order=shared_unit_order,
            title=f"clean (w={window_idx})" if r == 0 else "clean",
        )
        clean_ax.set_ylabel(f"{label}\nunit", fontsize=9)
        if r == n_rows - 1:
            clean_ax.set_xlabel("time (s)")

        plot_spike_raster(
            noisy_ax,
            corrupted_spikes_per_label[label],
            mode=raster_mode,
            bin_size_ms=bin_size_ms,
            color=_COLOR_NOISY,
            sort_by_rate=sort_by_rate,
            unit_order=shared_unit_order,
        )

        d_lat = latent_distance_per_label.get(label, float("nan"))
        noisy_ax.set_title(label, fontsize=9)
        noisy_ax.text(
            0.97,
            0.03,
            f"d_latent={d_lat:.3g}",
            transform=noisy_ax.transAxes,
            fontsize=8,
            ha="right",
            va="bottom",
            bbox=_ANNOT_BBOX,
        )
        if r == n_rows - 1:
            noisy_ax.set_xlabel("time (s)")

    fig.suptitle(f"M1-c · window={window_idx}", y=1.02)
    fig.tight_layout()
    if output_stem is not None:
        if session_id is None:
            raise ValueError(
                "plot_input_corruption_c: session_id is required when output_stem is set."
            )
        payload: dict[str, object] = {
            "clean": clean_spikes,
            "session_id": session_id,
            "segment_index": int(window_idx),
            "corruption_labels": labels,
            "latent_distance_per_label": dict(latent_distance_per_label),
        }
        for label in labels:
            payload[f"corrupted_{_npz_safe_key(label)}"] = corrupted_spikes_per_label[label]
        save_fig_multi(fig, output_stem)
        _persist_raster_source_arrays(Path(output_stem), payload)
    return fig


# ---------------------------------------------------------------------- latent_corruption_b


def plot_latent_corruption_b(
    window_idx: int,
    *,
    clean_spikes: np.ndarray,
    noisy_spikes_per_eps: Mapping[float, Sequence[np.ndarray]],
    eps_per_panel: Sequence[float],
    ax=None,
    bin_size_ms: float = 1.0,
    sort_by_rate: bool = True,
    metric_per_eps: Mapping[float, Mapping[str, float]] | None = None,
    session_id: str | None = None,
    output_stem: Path | str | None = None,
):
    """Plot M2-b: spike-train rasters across a few ε values.

    Builds one row per ε in ``eps_per_panel`` and plots
    3-5 noisy reconstructions per row alongside the clean one in
    column 0. Uses :func:`plot_spike_raster` with the standard raster
    styling (clean = ``_COLOR_CLEAN``, noisy = ``_COLOR_NOISY``). Each row
    carries an ε label on the left and
    — when ``metric_per_eps`` is provided — an in-panel bbox
    annotation on the rightmost noisy panel listing the per-ε
    reconstruction metrics.

    Args:
        window_idx: window identifier (figure title).
        clean_spikes: ``[T, U]`` clean reconstruction (or original) array.
        noisy_spikes_per_eps: mapping ``ε → list[[T, U]]`` of noisy
            reconstructions for that ε.
        eps_per_panel: which ε values to plot, in order. Each must be
            a key of ``noisy_spikes_per_eps``.
        ax: ignored when more than a single (ε, replica) panel is
            requested; reserved so the signature matches the rest of
            the module.
        bin_size_ms: width of one time bin in milliseconds (forwarded
            to the raster).
        sort_by_rate: sort population rows by descending firing rate.
        metric_per_eps: optional mapping ``ε → {metric_name → value}``
            placed as a bbox annotation on the last noisy panel of
            each row. Keep entries short — these are eyeballed at a
            glance.

    Returns:
        The constructed ``matplotlib.figure.Figure``.

    """
    import matplotlib.pyplot as plt

    if not eps_per_panel:
        raise ValueError("plot_latent_corruption_b: eps_per_panel is empty.")

    # Determine # columns = 1 (clean) + max replicas across requested ε.
    max_reps = max(
        len(noisy_spikes_per_eps[eps]) for eps in eps_per_panel if eps in noisy_spikes_per_eps
    )
    if max_reps == 0:
        raise ValueError(
            "plot_latent_corruption_b: no noisy replicas found for requested ε values."
        )
    # Cap at 5 noisy replicas per row.
    max_reps = min(max_reps, 5)

    n_rows = len(eps_per_panel)
    n_cols = 1 + max_reps

    raster_mode = "single" if clean_spikes.shape[-1] == 1 else "population"
    height_per_row = 3.0 if raster_mode == "population" else 4.0

    fig, axes_arr = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.0 * n_cols, height_per_row * n_rows),
        squeeze=False,
        sharey=True,
    )

    for r, eps in enumerate(eps_per_panel):
        # Column 0 = clean, repeated per row for visual reference.
        plot_spike_raster(
            axes_arr[r, 0],
            clean_spikes,
            mode=raster_mode,
            bin_size_ms=bin_size_ms,
            color=_COLOR_CLEAN,
            sort_by_rate=sort_by_rate,
            title="clean" if r == 0 else None,
        )
        # Row label (ε) on the y-axis of column 0 rather than overloading
        # subplot titles.
        axes_arr[r, 0].set_ylabel(f"ε = {eps:g}\nunit", fontsize=9)

        replicas = list(noisy_spikes_per_eps.get(eps, []))[:max_reps]
        for c in range(1, n_cols):
            sub_ax = axes_arr[r, c]
            if c - 1 < len(replicas):
                plot_spike_raster(
                    sub_ax,
                    replicas[c - 1],
                    mode=raster_mode,
                    bin_size_ms=bin_size_ms,
                    color=_COLOR_NOISY,
                    sort_by_rate=sort_by_rate,
                    title=f"noisy #{c - 1}" if r == 0 else None,
                )
            else:
                sub_ax.axis("off")

        # In-panel metric annotation on the last drawn noisy panel.
        if metric_per_eps is not None and eps in metric_per_eps:
            recon_dict = metric_per_eps[eps]
            if recon_dict:
                last_c = min(len(replicas), n_cols - 1)
                if last_c >= 1:
                    annot_ax = axes_arr[r, last_c]
                    annot = "\n".join(f"{k}={v:.3g}" for k, v in recon_dict.items())
                    annot_ax.text(
                        0.97,
                        0.03,
                        annot,
                        transform=annot_ax.transAxes,
                        fontsize=8,
                        ha="right",
                        va="bottom",
                        bbox=_ANNOT_BBOX,
                    )

    for sub_ax in axes_arr[-1, :]:
        if sub_ax.has_data():
            sub_ax.set_xlabel("time (s)")

    fig.suptitle(f"M2-b · window={window_idx}", y=1.02)
    fig.tight_layout()
    if output_stem is not None:
        if session_id is None:
            raise ValueError(
                "plot_latent_corruption_b: session_id is required when output_stem is set."
            )
        payload: dict[str, object] = {
            "clean": clean_spikes,
            "session_id": session_id,
            "segment_index": int(window_idx),
            "epsilons": [float(eps) for eps in eps_per_panel],
        }
        if metric_per_eps is not None:
            payload["metric_per_eps"] = {str(eps): vals for eps, vals in metric_per_eps.items()}
        for eps in eps_per_panel:
            replicas = list(noisy_spikes_per_eps.get(eps, []))[:max_reps]
            if not replicas:
                continue
            payload[f"noisy_eps_{_npz_safe_key(f'{float(eps):g}')}"] = np.stack(
                [_raster_source_array(rep) for rep in replicas],
                axis=0,
            )
        save_fig_multi(fig, output_stem)
        _persist_raster_source_arrays(Path(output_stem), payload)
    return fig
