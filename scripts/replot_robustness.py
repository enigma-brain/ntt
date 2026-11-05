"""Replotting CLI for robustness eval results.

Sibling of ``scripts/replot_from_eval_results.py``. Focuses on
**aggregate curves** that are useful to compare across multiple
tokenizers. Per-window rasters (input-corruption-c, latent-corruption-b)
are eval-time only — they need raw spike / reconstruction arrays which
we deliberately do not persist. To regenerate a raster, re-run the
relevant eval job with ``plot_examples=true``.

Subcommands
-----------
``latent-corruption-a``
    ε vs reconstruction-metric curve from latent_robustness (M2) ``results.csv``.
    Supports multiple result dirs as multiple tokenizers in one figure.

``input-corruption-a``
    Corruption-strength vs normalized latent distance, from input_robustness (M1)
    ``results.csv``. Multi-tokenizer overlay. (Available once the input_robustness (M1)
    job emits the canonical ``results.csv`` schema; until then this
    subcommand prints a friendly message and exits.)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import matplotlib
import numpy as np

matplotlib.use("Agg")  # noqa: E402

# ── helpers ────────────────────────────────────────────────────────────


def _require_dir(p: Path, what: str) -> Path:
    """Validate that ``p`` is an existing directory."""
    if not p.exists():
        raise click.ClickException(f"{what} not found: {p}")
    if not p.is_dir():
        raise click.ClickException(f"{what} is not a directory: {p}")
    return p


def _save_fig_multi(
    fig, base_path: Path, *, formats: tuple[str, ...] = ("png", "svg", "pdf")
) -> None:
    """Thin click-aware wrapper around ``ntt.evaluation.plotting.io.save_fig_multi``.

    Logs each emitted path through ``click.echo`` so ``--help``-style
    UX stays consistent with other replot scripts in this repo.

    """
    from ntt.evaluation.plotting.io import save_fig_multi

    written = save_fig_multi(fig, base_path, formats=formats)
    for p in written:
        click.echo(f"Wrote {p}")


def _load_csv(path: Path):
    """Read ``path`` as a pandas DataFrame; raise click.ClickException
    if pandas is missing or the file does not exist.

    """
    if not path.exists():
        raise click.ClickException(f"Missing results.csv: {path}")
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise click.ClickException("pandas required for replotting.") from exc
    return pd.read_csv(path)


def _load_m2_results_frame(results_dir: Path):
    """Load one latent_robustness (M2) results dir as a DataFrame.

    Preferred source is ``results.csv``. When that is missing, fall back
    to ``results.jsonl`` and pivot the long JSONL records into the raw
    wide schema expected by ``plot_latent_corruption_a``.

    Empty / missing JSONL files are treated as unusable.
    """
    csv_path = results_dir / "results.csv"
    if csv_path.exists():
        return _load_csv(csv_path)

    jsonl_path = results_dir / "results.jsonl"
    if not jsonl_path.exists():
        raise click.ClickException(f"Missing both results.csv and results.jsonl in {results_dir}")
    if jsonl_path.stat().st_size == 0:
        raise click.ClickException(f"results.jsonl is empty in {results_dir}")

    try:
        import json

        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise click.ClickException(
            "pandas required for latent_robustness (M2) JSONL fallback."
        ) from exc

    rows: list[dict[str, Any]] = []
    with jsonl_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("status") != "ok":
                continue
            metric = rec.get("metric")
            if metric is None:
                continue
            value = rec.get("raw_value")
            if value is None:
                continue
            row = {
                "session_id": rec.get("session_id"),
                "epsilon": rec.get("epsilon"),
                str(metric): value,
            }
            if "k" in rec:
                row["k"] = rec.get("k")
            rows.append(row)
            if "segment_index" in rec:
                rows[-1]["segment_index"] = rec.get("segment_index")

    if not rows:
        raise click.ClickException(
            f"results.jsonl has no usable latent_robustness (M2) rows in {results_dir}"
        )

    df = pd.DataFrame(rows)
    key_cols = ["session_id", "epsilon"]
    if "k" in df.columns and df["k"].notna().any():
        key_cols.append("k")
    if "segment_index" in df.columns:
        key_cols.insert(1, "segment_index")
    value_cols = [c for c in df.columns if c not in key_cols]
    wide = df.groupby(key_cols, as_index=False)[value_cols].first()
    return wide


_RESULT_DIR_PREFIXES = (
    "robustness_input_",
    "robustness_latent_",
)


def _strip_prefix_and_timestamp(name: str) -> str:
    """Strip ``robustness_<kind>_`` prefix, comparison-mode prefix, and
    trailing date suffix from a results-dir name.

    Examples:
        ``robustness_input_wanxA_vae_20260426_180143`` → ``wanxA_vae``.
        ``robustness_latent_cs_20260426_014547`` → ``cs``.
        ``robustness_latent_recon_vs_recon_cs100_20260428_eps0fix`` → ``cs100``.

    Trailing suffix recognised: ``_<YYYYMMDD>_<HHMMSS>`` (canonical) and
    ``_<YYYYMMDD>_<freeform-tag>`` (e.g. ``_eps0fix``). Comparison-mode
    prefixes (``recon_vs_recon_``, ``input_vs_recon_``) are stripped
    after the kind prefix so a hand-named run-dir like
    ``robustness_latent_recon_vs_recon_cs_...`` reduces to ``cs``.

    Defensive: when nothing matches we return ``name`` minus what we
    could strip, so hand-named dirs still get a reasonable
    label.

    """
    import re

    s = name
    for prefix in _RESULT_DIR_PREFIXES:
        if s.startswith(prefix):
            s = s[len(prefix) :]
            break
    for mode_prefix in ("recon_vs_recon_", "input_vs_recon_"):
        if s.startswith(mode_prefix):
            s = s[len(mode_prefix) :]
            break
    # Drop trailing ``_YYYYMMDD_HHMMSS`` (canonical) or ``_YYYYMMDD_<tag>``.
    # The 8-digit date anchors the match; the second segment is anything
    # word-like so postfixes such as ``_eps0fix`` are also dropped.
    s = re.sub(r"_\d{8}_\w+$", "", s)
    return s


def _bin_size_suffix_for(results_dir: Path) -> str:
    """Return ``" (5ms)"``-style suffix derived from the run's bin sizes.

    Reads ``run_meta.json`` for ``bin_size_ms``. Falls back to empty
    string when bin info is missing or 1ms (the default — not worth
    cluttering the legend with).

    """
    meta = results_dir / "run_meta.json"
    if not meta.exists():
        return ""
    try:
        import json

        with meta.open() as fh:
            d = json.load(fh)
    except Exception:  # pragma: no cover — tolerate meta shape
        return ""
    bin_ms = d.get("bin_size_ms")
    if bin_ms is None:
        return ""
    try:
        bin_ms_f = float(bin_ms)
    except (TypeError, ValueError):
        return ""
    # Always show the bin size in the label, including 1ms — the user
    # wants the bin size visible per legend entry.
    txt = f"{bin_ms_f:g}"
    return f" ({txt}ms)"


def _format_rate_bin_ms_for_label(rate_estimation_bin_ms: float) -> str:
    """Format ``rate_estimation_bin_ms`` for use in a snake_case label.

    Integer-valued floats are rendered without a decimal (``10.0 -> '10'``);
    non-integers get one decimal place (``12.5 -> '12.5'``).
    """
    f = float(rate_estimation_bin_ms)
    if f.is_integer():
        return str(int(round(f)))
    return f"{f:.1f}"


def _decorate_poisson_label(base: str, results_dir: Path) -> str:
    """If ``base`` is a bare Poisson label, append ``_<N>ms`` from the
    run-dir's metadata so legends disambiguate Poisson variants.

    Reads ``rate_estimation_bin_ms`` from ``<results_dir>/run_meta.json``.
    Falls back to the bare ``base`` if no metadata is readable, so the
    caller never crashes on hand-named directories.
    """
    if base not in {"poisson", "poisson_baseline"}:
        return base

    rate_ms: float | None = None

    meta = results_dir / "run_meta.json"
    if meta.exists():
        try:
            import json

            with meta.open() as fh:
                d = json.load(fh)
            v = d.get("rate_estimation_bin_ms")
            if v is not None:
                rate_ms = float(v)
        except Exception:  # pragma: no cover — tolerate meta shape
            pass

    if rate_ms is None:
        return base
    return f"poisson_{_format_rate_bin_ms_for_label(rate_ms)}ms"


def _lipschitz_bound_suffix_for(results_dir: Path) -> str:
    """Return ``", λL=<e-notation>"`` if the tokenizer reported a
    ``reconstruction_lipschitz_bound`` in its ``run_meta.json``, else
    an empty string.

    The bound (currently emitted by the CompressedSensing tokenizer)
    measures how the reconstruction operator amplifies noise in the
    latents — surfacing it in the legend lets the reader contextualise
    latent_robustness (M2) ε=0 ACG values: a CS run with a very small bound trivially gives
    near-identity at ε=0.

    """
    meta = results_dir / "run_meta.json"
    if not meta.exists():
        return ""
    try:
        import json

        with meta.open() as fh:
            d = json.load(fh)
    except Exception:  # pragma: no cover — tolerate meta shape
        return ""
    bound = d.get("reconstruction_lipschitz_bound")
    if bound is None:
        return ""
    try:
        bf = float(bound)
    except (TypeError, ValueError):
        return ""
    return f", λL={bf:.1e}"


def _compact_tokenizer_label(raw: str) -> str:
    """Apply lightweight user-facing tokenizer-name normalization.

    Rules requested by the user:
      * replace ``compressed_sensing`` → ``CS``
      * replace ``calm_transformer`` → ``calm``
      * for ``wanx_*`` names, keep a longer prefix so nearby VAE variants
        remain distinguishable in joint legends

    Other labels pass through unchanged.
    """
    s = str(raw).strip()
    if not s:
        return s

    s = s.replace("compressed_sensing", "CS")
    s = s.replace("calm_transformer", "calm")

    if s.startswith("wanx_"):
        parts = s.split("_")
        if len(parts) > 9:
            s = "_".join(parts[:9])
    return s


def _tokenizer_label_for(results_dir: Path) -> str:
    """Best-effort tokenizer label: read run_meta.json if present, else
    fall back to a cleaned-up directory name.

    Cleanup rules (per user request):
      * strip the ``robustness_input_`` / ``robustness_latent_`` prefix,
      * drop the trailing ``_<YYYYMMDD_HHMMSS>`` timestamp,
      * append ``" (Xms)"`` when the tokenizer's input bin size is >1ms,
      * for bare ``poisson`` labels, append ``_<N>ms`` from
        ``rate_estimation_bin_ms`` (so two Poisson runs with different
        rate bins are distinguishable in joint legends),
      * append ``", λL=<e-notation>"`` when the tokenizer reports a
        ``reconstruction_lipschitz_bound`` (CompressedSensing only at
        the moment).

    """
    base: str | None = None

    meta = results_dir / "run_meta.json"
    if meta.exists():
        try:
            import json

            with meta.open() as fh:
                d = json.load(fh)
            for key in ("tokenizer_name", "tokenizer_id", "checkpoint", "label"):
                if key in d and d[key]:
                    base = str(d[key])
                    break
        except Exception:  # pragma: no cover — be tolerant of meta shape
            pass
    if base is None:
        base = _strip_prefix_and_timestamp(results_dir.name)
    base = _compact_tokenizer_label(base)
    base = _decorate_poisson_label(base, results_dir)
    return f"{base}{_bin_size_suffix_for(results_dir)}{_lipschitz_bound_suffix_for(results_dir)}"


# ── CLI root ───────────────────────────────────────────────────────────


@click.group()
def cli() -> None:
    """Replotting CLI for robustness eval result directories."""


# ── latent-corruption-a (M2) ───────────────────────────────────────────


@cli.command("latent-corruption-a")
@click.argument(
    "results_dirs",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--metric",
    "metrics",
    multiple=True,
    default=("acg_correlation", "ccg_correlation"),
    show_default=True,
    help="Recon metric(s) to plot. May be passed multiple times.",
)
@click.option(
    "--out-dir",
    "out_dir",
    default=None,
    type=click.Path(path_type=Path),
    help="Output directory. Default: ./eval_outputs/latent-corruption-a_<timestamp>",
)
def latent_corruption_a(
    results_dirs: tuple[Path, ...], metrics: tuple[str, ...], out_dir: Path | None
) -> None:
    """Plot ε-vs-metric curves from latent_robustness (M2) results across one or more tokenizers.

    Each ``results_dir`` must contain a ``results.csv`` produced by
    ``ntt.jobs.eval_robustness_latent`` (see config/job/
    eval_robustness_latent.yaml). One line per tokenizer per metric.

    """
    from ntt.evaluation.plotting.robustness import plot_latent_corruption_a

    results_dirs = tuple(_require_dir(Path(d), "results_dirs") for d in results_dirs)

    if out_dir is None:
        from datetime import datetime

        out_dir = Path("eval_outputs") / f"latent-corruption-a_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Only the joint figure is produced, even when a single dir is passed (the
    # "joint" is then just one tokenizer, but the output naming is uniform).
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise click.ClickException("pandas required for joint figure.") from exc
    frames = []
    for d in results_dirs:
        try:
            df = _load_m2_results_frame(d).copy()
        except click.ClickException as exc:
            click.echo(f"warning: {exc.message}; skipping {d}", err=True)
            continue
        df["tokenizer_id"] = _tokenizer_label_for(d)
        frames.append(df)
    if not frames:
        raise click.ClickException(
            "No usable latent_robustness (M2) results found. Local dirs were missing results.csv "
            "and had empty/missing results.jsonl."
        )
    joint = pd.concat(frames, ignore_index=True)
    joint_csv = out_dir / "_joint_results.csv"
    joint.to_csv(joint_csv, index=False)
    metrics_list = list(metrics) or ["acg_correlation", "ccg_correlation"]
    fig = plot_latent_corruption_a(joint_csv, metrics=metrics_list)
    _save_fig_multi(fig, out_dir / "joint_latent-corruption-a")
    for metric_name in metrics_list:
        fig_single = plot_latent_corruption_a(joint_csv, metrics=[metric_name])
        _save_fig_multi(fig_single, out_dir / f"latent-corruption-a_{metric_name}")


# ── input-corruption-{a,b} (M1) ────────────────────────────────────────


_M1_STRENGTH_FIELD = {
    "jitter": "k_ms",
    "drop": "fraction",
    "add": "fraction",
    "drop_and_add": "fraction",
    "window_shift": "max_shift_ms",
    "drop_neurons": "fraction",
    "permute_neurons": "",
}


def _parse_corruption_label(label: str) -> tuple[str, float] | None:
    """Parse a corruption label like ``"jitter|k_ms=1.0|p=0.5"`` into
    ``(type, strength)``. ``permute_neurons`` returns strength 1.0.
    Returns ``None`` if the label doesn't match a known corruption type.
    """
    parts = label.split("|")
    ctype = parts[0]
    field = _M1_STRENGTH_FIELD.get(ctype)
    if field is None:
        return None
    if field == "":
        return ctype, 1.0
    for kv in parts[1:]:
        if "=" in kv:
            k, v = kv.split("=", 1)
            if k == field:
                try:
                    return ctype, float(v)
                except ValueError:
                    return None
    return None


def _build_input_robustness_payload_from_csv(
    csv_path: Path, *, metric_name: str
) -> dict[str, dict[str, list[float]]]:
    """Build the ``{corruption_type → {strengths, means, stds}}`` payload
    that ``plot_input_corruption_a`` consumes, by reading one
    tokenizer's ``results.csv`` and aggregating
    ``metric_name`` rows across sessions + sub_labels per
    (corruption_type, strength).

    """
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise click.ClickException("pandas required for input_robustness (M1) replot.") from exc

    df = pd.read_csv(csv_path)
    if "metric" not in df.columns or "corruption" not in df.columns:
        raise click.ClickException(f"{csv_path}: missing required 'metric'/'corruption' columns.")
    if "raw_value" not in df.columns:
        raise click.ClickException(f"{csv_path}: missing 'raw_value' column.")
    value_col = "raw_value"
    df = df[df["metric"] == metric_name]
    if df.empty:
        return {}

    # Bucket: corruption_type → strength → list[value]
    bucket: dict[str, dict[float, list[float]]] = {}
    for _, row in df.iterrows():
        parsed = _parse_corruption_label(str(row["corruption"]))
        if parsed is None:
            continue
        ctype, strength = parsed
        v = row.get(value_col)
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        bucket.setdefault(ctype, {}).setdefault(strength, []).append(v)

    payload: dict[str, dict[str, list[float]]] = {}
    for ctype, smap in bucket.items():
        xs = sorted(smap.keys())
        if not xs:
            continue
        means = [float(np.mean(smap[s])) for s in xs]
        stds = [float(np.std(smap[s], ddof=0)) if len(smap[s]) > 1 else 0.0 for s in xs]
        # No synthetic zero-strength prepend (see job._build_curve_payload
        # comment): the empirical zero comes from the real "jitter
        # k_ms=0 p=0" head spec at the start of the default sweep,
        # which captures stochastic-encoder noise floors organically.
        payload[ctype] = {"strengths": xs, "means": means, "stds": stds}
    return payload


def _subset_input_robustness_payloads_to_label(
    payloads: list[dict[str, dict[str, list[float]]]],
    tokenizer_labels: list[str],
    label: str,
) -> tuple[list[dict[str, dict[str, list[float]]]], list[str]]:
    """Keep only one corruption label from each tokenizer payload.

    Tokenizers missing the requested label are skipped so the label list
    remains aligned with the returned payload list.
    """
    out_payloads: list[dict[str, dict[str, list[float]]]] = []
    out_labels: list[str] = []
    for payload, tokenizer_label in zip(payloads, tokenizer_labels, strict=False):
        if label not in payload:
            continue
        out_payloads.append({label: payload[label]})
        out_labels.append(tokenizer_label)
    return out_payloads, out_labels


def _labels_present_in_input_robustness_payloads(
    payloads: list[dict[str, dict[str, list[float]]]],
) -> list[str]:
    """Return first-seen corruption-label order across tokenizer payloads."""
    seen: dict[str, None] = {}
    for payload in payloads:
        for label in payload:
            seen.setdefault(label, None)
    return list(seen.keys())


@cli.command("input-corruption-a")
@click.argument(
    "results_dirs",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--out-dir",
    "out_dir",
    default=None,
    type=click.Path(path_type=Path),
    help="Output directory. Default: ./results/joint_plots/input-corruption-a/<timestamp>",
)
def input_corruption_a(results_dirs: tuple[Path, ...], out_dir: Path | None) -> None:
    """Plot corruption strength vs normalized latent distance from input_robustness (M1)
    results across one or more tokenizers.

    Reads the ``robustness_input`` metric rows from each
    ``results.csv``. Skips dirs whose CSV has no such rows (runs
    where the metric was not registered) with a logged warning.

    """
    import numpy as np  # noqa: F401 — used inside _build_input_robustness_payload_from_csv

    from ntt.evaluation.plotting.robustness import plot_input_corruption_a

    if out_dir is None:
        from datetime import datetime

        out_dir = (
            Path("results/joint_plots/input-corruption-a") / f"{datetime.now():%Y%m%d_%H%M%S}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    per_tokenizer_payloads: list = []
    tokenizer_labels: list[str] = []
    for d in results_dirs:
        d = _require_dir(Path(d), "results_dir")
        csv_path = d / "results.csv"
        try:
            payload = _build_input_robustness_payload_from_csv(
                csv_path, metric_name="robustness_input"
            )
        except click.ClickException as exc:
            click.echo(f"warning: {exc.message}; skipping {d}", err=True)
            continue
        if not payload:
            click.echo(
                f"warning: {csv_path} has no rows for metric='robustness_input' "
                "(metric may not have been registered when this run completed); "
                f"skipping {d}",
                err=True,
            )
            continue
        per_tokenizer_payloads.append(payload)
        tokenizer_labels.append(_tokenizer_label_for(d))

    if not per_tokenizer_payloads:
        raise click.ClickException(
            "No usable results found. None of the given dirs had rows for "
            "metric='robustness_input' in their results.csv."
        )

    fig = plot_input_corruption_a(per_tokenizer_payloads, tokenizers=tokenizer_labels)
    _save_fig_multi(fig, out_dir / "joint_input-corruption-a")
    for label in _labels_present_in_input_robustness_payloads(per_tokenizer_payloads):
        sub_payloads, sub_labels = _subset_input_robustness_payloads_to_label(
            per_tokenizer_payloads, tokenizer_labels, label
        )
        if not sub_payloads:
            continue
        fig_single = plot_input_corruption_a(sub_payloads, tokenizers=sub_labels)
        _save_fig_multi(fig_single, out_dir / f"input-corruption-a_{label}")
    click.echo(f"Plotted {len(tokenizer_labels)} tokenizer(s): {tokenizer_labels}")


# ── main ───────────────────────────────────────────────────────────────


if __name__ == "__main__":
    cli()
