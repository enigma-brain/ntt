"""Re-generate evaluation plots from eval job results.json files.

Supports comparing multiple tokenizers side-by-side.

Usage:
    python scripts/replot_from_eval_results.py dir1/ dir2/ dir3/
    python scripts/replot_from_eval_results.py dir1/ --output-dir my_plots/

Each directory normally contains a ``results.json`` in the format produced by
the eval job (top-level keys: checkpoint, per_session, aggregated, ...). When
given a robustness-only results dir (``results.csv`` without ``results.json``),
the script skips the standard eval plots for that dir and only uses it for the
joint M1-a / M2-a robustness replots.

"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np

matplotlib.use("Agg")

from ntt.evaluation.aggregation.category_scores import apply_alternative_floors
from ntt.evaluation.aggregation.comparison import MetricComparison
from ntt.evaluation.core.types import EvaluationReport
from ntt.evaluation.plot_all import generate_all_plots
from ntt.evaluation.plotting.io import save_fig_multi
from ntt.evaluation.scaling_plots import (
    compute_token_dims_per_second,
    pick_metric_keys_for_dims_plot,
    plot_metric_vs_metric_colored_by_token_dims,
    plot_metric_vs_metric_colored_by_token_dims_per_session,
    plot_metric_vs_token_dims,
    plot_metric_vs_token_dims_per_session,
)

try:
    from scripts.replot_robustness import (
        _build_input_robustness_payload_from_csv,
    )
    from scripts.replot_robustness import (
        _tokenizer_label_for as _robustness_tokenizer_label_for,
    )
except ImportError:  # pragma: no cover - direct script execution path
    from replot_robustness import (  # type: ignore[no-redef]
        _build_input_robustness_payload_from_csv,
    )
    from replot_robustness import (
        _tokenizer_label_for as _robustness_tokenizer_label_for,
    )

# -- defaults -----------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "eval_outputs" / "comparison_plots"

_ROBUSTNESS_SIBLING_SUFFIXES = ("_m1", "_m2")


def _as_float_or_none(x: object) -> float | None:
    try:
        v = float(x)  # type: ignore[arg-type]
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return float(v)


def _scalarize_raw_value(v: object) -> float | None:
    """Convert scalar/array-like raw_value into a float for plotting."""
    if v is None:
        return None
    try:
        if isinstance(v, list):
            arr = np.array(v, dtype=float)
            return float(np.nanmean(arr)) if arr.size else None
        if isinstance(v, (int, float)):
            vv = float(v)
            return vv if np.isfinite(vv) else None
        arr = np.array(v, dtype=float)
        return float(np.nanmean(arr)) if arr.size else None
    except Exception:
        return None


def _sort_sweep_entry_key(entry: dict) -> tuple[int, float | str]:
    """Sort numeric parameter values before non-numeric ones."""
    pv = entry.get("param_value")
    pvf = _as_float_or_none(pv)
    return (0, pvf) if pvf is not None else (1, str(pv))


def _extract_std_from_metadata(md: object) -> float | None:
    """Best-effort std for a sweep run."""
    if not isinstance(md, dict):
        return None
    for k in ("raw_value_std", "std", "r2_std", "accuracy_std"):
        v = md.get(k)
        if isinstance(v, (int, float)):
            out = _as_float_or_none(v)
            if out is not None:
                return out
    folds = md.get("fold_r2s") or md.get("fold_accuracies")
    if isinstance(folds, list) and len(folds) >= 3:
        try:
            vals = np.array([float(x) for x in folds], dtype=float)
            if np.isfinite(vals).any():
                return float(np.nanstd(vals))
        except Exception:
            return None
    return None


def load_detailed_sweeps(
    results_dir: Path,
) -> tuple[dict[str, list[dict]] | None, dict[str, dict[str, list[dict]]] | None]:
    """Load sweep points from `metrics_detailed_results.jsonl` if present.

    Returns:
      - aggregated_sweeps_by_metric: metric_key -> list[{param_name,param_value,y,yerr,run_label}]
      - sweeps_by_session: session_id -> metric_key -> list[entry]
    """
    p = results_dir / "metrics_detailed_results.jsonl"
    if not p.is_file():
        return None, None

    by_session: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    pooled: dict[tuple[str, str, str], list[tuple[float, float | None]]] = defaultdict(list)

    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("record_type") != "sweep_point":
            continue
        metric_key = rec.get("metric_key")
        sid = rec.get("session_id")
        sweep = rec.get("sweep") or {}
        if (
            not isinstance(metric_key, str)
            or not isinstance(sid, str)
            or not isinstance(sweep, dict)
        ):
            continue

        param_name = sweep.get("param_name")
        param_value = sweep.get("param_value")
        y = _scalarize_raw_value(sweep.get("raw_value"))
        if y is None:
            continue
        yerr = _extract_std_from_metadata(sweep.get("run_metadata"))

        if param_name == "chunk_window_ms":
            run_label = f"cw{param_value}ms"
        elif param_name == "bin_size_ms":
            run_label = f"{param_value}ms"
        elif isinstance(param_name, str) and param_name:
            run_label = f"{param_name}={param_value}"
        else:
            run_label = ""

        entry = {
            "param_name": param_name,
            "param_value": param_value,
            "y": float(y),
            "yerr": float(yerr) if yerr is not None else None,
            "run_label": run_label,
        }
        by_session[sid][metric_key].append(entry)

        pooled[(metric_key, str(param_name), str(param_value))].append(
            (float(y), float(yerr) if yerr is not None else None)
        )

    aggregated: dict[str, list[dict]] = defaultdict(list)
    for (metric_key, param_name, param_value), vals in pooled.items():
        ys = np.array([v for v, _e in vals], dtype=float)
        if ys.size == 0 or not np.isfinite(ys).any():
            continue
        y_mean = float(np.nanmean(ys))
        errs = [e for _v, e in vals if e is not None and np.isfinite(e)]
        yerr_mean = float(np.nanmean(np.array(errs, dtype=float))) if errs else None

        if param_name == "chunk_window_ms":
            run_label = f"cw{param_value}ms"
        elif param_name == "bin_size_ms":
            run_label = f"{param_value}ms"
        elif param_name:
            run_label = f"{param_name}={param_value}"
        else:
            run_label = ""

        aggregated[metric_key].append(
            {
                "param_name": param_name,
                "param_value": param_value,
                "y": y_mean,
                "yerr": yerr_mean,
                "run_label": run_label,
            }
        )

    for mk, entries in aggregated.items():
        aggregated[mk] = sorted(entries, key=_sort_sweep_entry_key)

    return dict(aggregated), dict(by_session)


def load_session_reports(data: dict) -> dict[str, list[tuple]]:
    """Reconstruct session_reports from a results.json dict for generate_all_plots.

    Prefers ``per_session_per_segment`` (full per-segment detail, including
    metadata like RDM matrices) over ``per_session`` (aggregated, metadata
    lost).  Falls back to ``per_session`` when segments are not available.

    Returns ``{session_id: [(segment_stub, {"report": EvaluationReport}), ...]}``.
    """
    # Prefer per-segment data (preserves structure-preservation metadata)
    per_segment = data.get("per_session_per_segment")
    if per_segment is not None:
        session_reports: dict[str, list[tuple]] = {}
        for session_id, segment_dicts in per_segment.items():
            if not isinstance(segment_dicts, list):
                raise ValueError(
                    f"Expected per_session_per_segment[{session_id!r}] to be a list, "
                    f"got {type(segment_dicts)}"
                )
            segments = []
            for seg_idx, report_dict in enumerate(segment_dicts):
                report = EvaluationReport.from_dict(report_dict)
                segment_stub = SimpleNamespace(segment_index=seg_idx)
                segments.append((segment_stub, {"report": report}))
            session_reports[session_id] = segments
        return session_reports

    # Fall back to per-session aggregated data
    per_session = data.get("per_session")
    if per_session is None:
        raise ValueError(
            f"results.json has neither 'per_session_per_segment' nor 'per_session' key. "
            f"Available keys: {list(data.keys())}"
        )

    session_reports = {}
    for session_id, report_dict in per_session.items():
        if not isinstance(report_dict, dict) or "results" not in report_dict:
            got = list(report_dict.keys()) if isinstance(report_dict, dict) else type(report_dict)
            raise ValueError(
                f"Expected per_session[{session_id!r}] to be an EvaluationReport dict "
                f"with a 'results' key, got keys: "
                f"{got}"
            )
        report = EvaluationReport.from_dict(report_dict)
        segment_stub = SimpleNamespace(segment_index=0)
        session_reports[session_id] = [(segment_stub, {"report": report})]

    return session_reports


def load_comparisons_from_results(data: dict) -> dict[str, MetricComparison] | None:
    """Extract normalized comparisons from a results.json dict.

    Returns ``None`` if the results were generated without Poisson comparison.
    firing_rate_error is scored against its population-mean-rate reference (recomputed for
    results saved before it), as in the aggregation.
    """
    raw = data.get("comparison_vs_poisson")
    if raw is None:
        return None
    comparisons, _ = apply_alternative_floors(
        {name: MetricComparison.from_dict(comp_dict) for name, comp_dict in raw.items()}, data
    )
    return comparisons


def load_poisson_session_reports(data: dict) -> dict[str, list[tuple]] | None:
    """Reconstruct poisson session_reports if present (for plotting baseline as a 'tokenizer')."""
    per_segment = data.get("poisson_per_session_per_segment")
    if per_segment is None:
        return None
    session_reports: dict[str, list[tuple]] = {}
    for session_id, segment_dicts in per_segment.items():
        if not isinstance(segment_dicts, list):
            continue
        segments = []
        for seg_idx, report_dict in enumerate(segment_dicts):
            report = EvaluationReport.from_dict(report_dict)
            segment_stub = SimpleNamespace(segment_index=seg_idx)
            segments.append((segment_stub, {"report": report}))
        session_reports[session_id] = segments
    return session_reports


def derive_display_name(results_dir: Path, data: dict) -> str:
    """Derive a human-readable display name from directory name or checkpoint path.

    Prefers the directory name. Falls back to the checkpoint basename.
    """
    dir_name = results_dir.name
    if dir_name and dir_name not in (".", ".."):
        return dir_name

    checkpoint = data.get("checkpoint", "")
    if checkpoint:
        return Path(checkpoint).parent.name

    return str(results_dir)


def _save_fig_multi_with_logging(fig, base_path: Path) -> None:
    """Save a figure in the repo's standard trio of formats.

    Mirrors the behavior of ``scripts/replot_robustness.py`` but uses plain
    ``print()`` here because this script is argparse-based rather than click-
    based.
    """
    written = save_fig_multi(fig, base_path, formats=("png", "svg", "pdf"))
    for path in written:
        print(f"Wrote {path}")


def _load_results_csv(csv_path: Path):
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pandas required for robustness replotting.") from exc
    return pd.read_csv(csv_path)


def _looks_like_latent_robustness_results_dir(results_dir: Path) -> bool:
    csv_path = results_dir / "results.csv"
    if not csv_path.is_file():
        return False
    try:
        df = _load_results_csv(csv_path)
    except Exception:
        return False
    return "epsilon" in df.columns


def _has_robustness_sibling_results(results_dir: Path) -> bool:
    """Return true when ``results_dir`` is an eval_all wrapper with input_robustness (M1)/
    latent_robustness (M2) siblings."""
    for suffix in _ROBUSTNESS_SIBLING_SUFFIXES:
        sibling = results_dir.parent / f"{results_dir.name}{suffix}"
        if (sibling / "results.csv").is_file():
            return True
    return False


def _collect_robustness_result_dirs(
    input_dirs: list[Path],
) -> tuple[list[Path], list[Path]]:
    """Return candidate input_robustness (M1) / latent_robustness (M2) dirs from explicit inputs
    and `_m1` / `_m2` siblings.

    The latent-quality wrapper writes sibling dirs named ``<output>_m1`` and
    ``<output>_m2``. We also accept explicitly passed robustness dirs, even
    when their naming does not follow that convention, as long as their
    ``results.csv`` looks like the corresponding schema.
    """
    m1_dirs: list[Path] = []
    m2_dirs: list[Path] = []
    seen_m1: set[Path] = set()
    seen_m2: set[Path] = set()

    candidate_dirs: list[Path] = []
    for base_dir in input_dirs:
        candidate_dirs.append(base_dir)
        for suffix in _ROBUSTNESS_SIBLING_SUFFIXES:
            sibling = base_dir.parent / f"{base_dir.name}{suffix}"
            if sibling != base_dir:
                candidate_dirs.append(sibling)

    for candidate in candidate_dirs:
        csv_path = candidate / "results.csv"
        if not csv_path.is_file():
            continue

        try:
            m1_payload = _build_input_robustness_payload_from_csv(
                csv_path, metric_name="robustness_input"
            )
        except Exception:
            m1_payload = {}
        if m1_payload and candidate not in seen_m1:
            m1_dirs.append(candidate)
            seen_m1.add(candidate)

        if _looks_like_latent_robustness_results_dir(candidate) and candidate not in seen_m2:
            m2_dirs.append(candidate)
            seen_m2.add(candidate)

    return m1_dirs, m2_dirs


def _generate_robustness_replots(input_dirs: list[Path], output_dir: Path) -> bool:
    """Generate M1-a / M2-a joint plots when robustness results are present."""
    from ntt.evaluation.plotting.robustness import (
        plot_input_corruption_a,
        plot_latent_corruption_a,
    )

    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pandas required for robustness replotting.") from exc

    m1_dirs, m2_dirs = _collect_robustness_result_dirs(input_dirs)
    wrote_any = False

    if m1_dirs:
        robustness_dir = output_dir / "robustness"
        robustness_dir.mkdir(parents=True, exist_ok=True)

        per_tokenizer_payloads: list[dict[str, dict[str, list[float]]]] = []
        tokenizer_labels: list[str] = []
        for results_dir in m1_dirs:
            payload = _build_input_robustness_payload_from_csv(
                results_dir / "results.csv",
                metric_name="robustness_input",
            )
            if not payload:
                continue
            per_tokenizer_payloads.append(payload)
            tokenizer_labels.append(_robustness_tokenizer_label_for(results_dir))

        if per_tokenizer_payloads:
            print(
                f"Generating robustness M1-a joint plot from {len(per_tokenizer_payloads)} dir(s)."
            )
            fig = plot_input_corruption_a(
                per_tokenizer_payloads,
                tokenizers=tokenizer_labels,
            )
            _save_fig_multi_with_logging(fig, robustness_dir / "joint_input-corruption-a")
            wrote_any = True

    if m2_dirs:
        robustness_dir = output_dir / "robustness"
        robustness_dir.mkdir(parents=True, exist_ok=True)

        frames = []
        for results_dir in m2_dirs:
            df = _load_results_csv(results_dir / "results.csv").copy()
            df["tokenizer_id"] = _robustness_tokenizer_label_for(results_dir)
            frames.append(df)

        if frames:
            print(f"Generating robustness M2-a joint plot from {len(frames)} dir(s).")
            joint = pd.concat(frames, ignore_index=True)
            joint_csv = robustness_dir / "_joint_latent_corruption_results.csv"
            joint.to_csv(joint_csv, index=False)
            fig = plot_latent_corruption_a(joint_csv)
            _save_fig_multi_with_logging(fig, robustness_dir / "joint_latent-corruption-a")
            wrote_any = True

    if not wrote_any:
        print(
            "Robustness plots: skipped "
            "(no input_robustness (M1)/latent_robustness (M2) results.csv found)."
        )

    return wrote_any


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Re-generate evaluation plots from eval job results, comparing multiple tokenizers."
        ),
    )
    parser.add_argument(
        "result_dirs",
        nargs="+",
        type=Path,
        help=(
            "Directories containing results.json files (standard eval runs). "
            "Robustness-only dirs with results.csv are also accepted."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save plots into (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Resolution for saved figures (default: 150)",
    )
    parser.add_argument(
        "--no-scaling-plots",
        action="store_false",
        dest="scaling_plots",
        help="Disable additional scaling scatter plots.",
    )
    parser.set_defaults(scaling_plots=True)
    parser.add_argument(
        "--dims-metrics",
        nargs="*",
        default=None,
        help=(
            "Optional list of metric keys to plot for the `metric_vs_token_dims_*` figures. "
            "Default: behavioral decoding metrics plus repeats_snr."
        ),
    )
    parser.add_argument(
        "--dims-x-axis",
        type=str,
        choices=["token_dims_per_second", "token_dims_per_chunk"],
        default="token_dims_per_second",
        help=(
            "X-axis definition for metric-vs-token-dims plots. "
            "`token_dims_per_second` keeps x fixed across chunk sweeps; "
            "`token_dims_per_chunk` scales x by chunk_window_ms for sweep points."
        ),
    )
    parser.add_argument(
        "--xy-x-metric",
        type=str,
        default="acg_correlation",
        help=(
            "Metric key for x-axis in recon-vs-downstream scatter plots "
            "(default: acg_correlation)."
        ),
    )
    parser.add_argument(
        "--xy-y-metrics",
        nargs="*",
        default=["behavioral_regression_cursor_vel", "behavioral_classification_trial_target_id"],
        help=(
            "Metric keys for y-axis in recon-vs-downstream scatter plots. "
            "One plot is created per y metric."
        ),
    )
    args = parser.parse_args()

    tokenizer_session_reports: dict[str, dict[str, list[tuple]]] = {}
    aggregated_reports: dict[str, EvaluationReport] = {}
    tokenizer_configs: dict[str, dict] = {}
    detailed_sweeps_by_tokenizer: dict[str, dict[str, list[dict]]] = {}
    detailed_sweeps_by_tokenizer_and_session: dict[str, dict[str, dict[str, list[dict]]]] = {}
    # Use comparisons from the first directory that has them
    normalized_comparisons: dict[str, MetricComparison] | None = None
    poisson_added = False
    saw_explicit_robustness_input = False

    for result_dir in args.result_dirs:
        results_path = result_dir / "results.json"
        if not results_path.is_file():
            if (result_dir / "results.csv").is_file():
                print(
                    f"Loading robustness-only input: {result_dir} "
                    "(no results.json, will only contribute robustness plots)"
                )
                saw_explicit_robustness_input = True
                continue
            if _has_robustness_sibling_results(result_dir):
                print(
                    f"Loading robustness wrapper input: {result_dir} "
                    "(will use _m1/_m2 sibling dirs for robustness plots)"
                )
                saw_explicit_robustness_input = True
                continue
            raise FileNotFoundError(f"No results.json or results.csv found in {result_dir}")

        with open(results_path) as f:
            data = json.load(f)

        display_name = derive_display_name(result_dir, data)
        print(f"Loading: {results_path}  ->  display name: {display_name!r}")

        session_reports = load_session_reports(data)
        print(f"  Sessions: {len(session_reports)}")

        agg_sweeps, sess_sweeps = load_detailed_sweeps(result_dir)
        if agg_sweeps is not None:
            detailed_sweeps_by_tokenizer[display_name] = agg_sweeps
        if sess_sweeps is not None:
            detailed_sweeps_by_tokenizer_and_session[display_name] = sess_sweeps

        aggregated = data.get("aggregated")
        if not isinstance(aggregated, dict):
            raise ValueError(f"{results_path} missing 'aggregated' dict.")
        aggregated_reports[display_name] = EvaluationReport.from_dict(aggregated)
        tok_cfg = data.get("tokenizer_config", {})
        if isinstance(tok_cfg, dict):
            tokenizer_configs[display_name] = tok_cfg

        # Load normalized comparisons if available
        if normalized_comparisons is None:
            comparisons = load_comparisons_from_results(data)
            if comparisons is not None:
                normalized_comparisons = comparisons
                print(f"  Loaded {len(comparisons)} normalized comparisons for radar chart")

        if display_name in tokenizer_session_reports:
            raise ValueError(
                f"Duplicate display name {display_name!r}. Use directories with distinct names."
            )
        tokenizer_session_reports[display_name] = session_reports

        # Optionally add Poisson baseline as an extra 'tokenizer' (once).
        if not poisson_added and isinstance(data.get("poisson_baseline"), dict):
            try:
                poisson_report = EvaluationReport.from_dict(data["poisson_baseline"])
            except Exception:
                poisson_report = None
            poisson_sessions = load_poisson_session_reports(data)
            if poisson_report is not None and poisson_sessions is not None:
                poisson_name = "Poisson"
                if poisson_name not in tokenizer_session_reports:
                    tokenizer_session_reports[poisson_name] = poisson_sessions
                    aggregated_reports[poisson_name] = poisson_report
                    # Give Poisson a synthetic tokenizer_config so scaling plots can include it.
                    prb = data.get("poisson_rate_bin_ms")
                    # Use the same neuron count the eval job used for tokenization by default.
                    # (If eval was configured with poisson_use_all_neurons=True, this may differ;
                    # we currently don't store that count explicitly in results.json.)
                    # Prefer explicit stored Poisson unit count when available.
                    n_units_used = data.get("poisson_n_units_used")
                    if isinstance(n_units_used, (int, float)) and int(n_units_used) > 0:
                        n_neurons_val = int(n_units_used)
                    else:
                        n_neurons = None
                        tok_cfg_for_neurons = data.get("tokenizer_config", {})
                        if isinstance(tok_cfg_for_neurons, dict):
                            n_neurons = tok_cfg_for_neurons.get("n_neurons")
                        if isinstance(n_neurons, (int, float)) and int(n_neurons) > 0:
                            n_neurons_val = int(n_neurons)
                        else:
                            n_neurons_val = 1
                    if prb is not None:
                        tokenizer_configs[poisson_name] = {
                            "KIND": "poisson_baseline",
                            "has_unit_dim": False,
                            "z_dim": 1,
                            "n_neurons": n_neurons_val,
                            "token_bin_size_ms": float(prb),
                        }
                    poisson_added = True

    robustness_written = _generate_robustness_replots(args.result_dirs, args.output_dir)

    if not tokenizer_session_reports:
        if robustness_written or saw_explicit_robustness_input:
            print("Standard eval plots: skipped (no results.json inputs loaded).")
            print("Done.")
            return
        raise ValueError("No tokenizer results loaded.")

    tok_names = list(tokenizer_session_reports.keys())
    print(f"\nGenerating plots in: {args.output_dir}")
    print(f"Tokenizers: {tok_names}")
    if normalized_comparisons:
        print(f"Radar chart: yes ({len(normalized_comparisons)} metrics)")
    else:
        print("Radar chart: skipped (no comparison_vs_poisson in results)")

    # When we have normalized comparisons from a single tokenizer, supply
    # explicit names so the radar chart is generated even without a second
    # tokenizer directory (the comparison was already computed vs Poisson).
    tokenizer_names_for_comparison: tuple[str, str] | None = None
    if normalized_comparisons is not None and len(tok_names) == 1:
        tokenizer_names_for_comparison = (tok_names[0], "Poisson")

    generate_all_plots(
        tokenizer_session_reports,
        output_dir=args.output_dir,
        tokenizer_names_for_comparison=tokenizer_names_for_comparison,
        normalized_comparisons=normalized_comparisons,
        dpi=args.dpi,
    )

    if args.scaling_plots:
        print("\nGenerating scaling plots...")
        scaling_dir = args.output_dir / "scaling_plots"
        scaling_dir.mkdir(parents=True, exist_ok=True)
        scaling_dir_per_session = args.output_dir / "scaling_plots_per_session"
        scaling_dir_per_session.mkdir(parents=True, exist_ok=True)

        token_dims_by_tok: dict[str, float] = {}
        for name, cfg in tokenizer_configs.items():
            td = compute_token_dims_per_second(cfg)
            if td is None:
                print(f"  [skip] {name}: could not derive token dims/sec from tokenizer_config")
                continue
            token_dims_by_tok[name] = td

        # 1) metric vs token dims/sec (one plot per metric)
        metric_keys = pick_metric_keys_for_dims_plot(
            aggregated_reports, include=set(args.dims_metrics) if args.dims_metrics else None
        )
        for metric_key in metric_keys:
            out = scaling_dir / f"metric_vs_token_dims__{metric_key}.png"
            plot_metric_vs_token_dims(
                metric_key=metric_key,
                reports_by_tokenizer=aggregated_reports,
                token_dims_by_tokenizer=token_dims_by_tok,
                detailed_sweeps_by_tokenizer=detailed_sweeps_by_tokenizer or None,
                output_path=out,
                dpi=args.dpi,
                x_axis=args.dims_x_axis,
            )
            # Per-session version (critical for behavioral decoding metrics).
            plot_metric_vs_token_dims_per_session(
                metric_key=metric_key,
                tokenizer_session_reports=tokenizer_session_reports,
                token_dims_by_tokenizer=token_dims_by_tok,
                detailed_sweeps_by_tokenizer_and_session=(
                    detailed_sweeps_by_tokenizer_and_session or None
                ),
                output_dir=scaling_dir_per_session,
                dpi=args.dpi,
                x_axis=args.dims_x_axis,
            )

        # 2) x metric vs y metric (color = token dims/sec), one plot per y metric
        for y_metric in args.xy_y_metrics:
            out = scaling_dir / f"scatter__{y_metric}__vs__{args.xy_x_metric}.png"
            plot_metric_vs_metric_colored_by_token_dims(
                x_metric=args.xy_x_metric,
                y_metric=y_metric,
                reports_by_tokenizer=aggregated_reports,
                token_dims_by_tokenizer=token_dims_by_tok,
                detailed_sweeps_by_tokenizer=detailed_sweeps_by_tokenizer or None,
                output_path=out,
                dpi=args.dpi,
            )
            # Per-session version (one point per chunk-size, per session).
            plot_metric_vs_metric_colored_by_token_dims_per_session(
                x_metric=args.xy_x_metric,
                y_metric=y_metric,
                tokenizer_session_reports=tokenizer_session_reports,
                token_dims_by_tokenizer=token_dims_by_tok,
                detailed_sweeps_by_tokenizer_and_session=(
                    detailed_sweeps_by_tokenizer_and_session or None
                ),
                output_dir=scaling_dir_per_session,
                dpi=args.dpi,
                x_axis="token_dims_per_chunk",
            )
        print(f"Scaling plots saved under: {scaling_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
