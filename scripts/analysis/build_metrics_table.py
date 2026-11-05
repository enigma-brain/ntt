#!/usr/bin/env python3
"""Build a wide per-tokenizer raw-metrics table from eval artifacts.

Columns: bare ``{block}/{metric}`` = mean; ``::std`` = cross-session sample std
(ddof=1, None if n<2); ``::n`` = count behind mean/std; ``::norm`` = aggregated
Poisson normalized_score (recon metrics with a valid comparison only; firing_rate_error is
scored against its population-mean-rate reference, as in the single_neuron_recon score).
Ridge regression columns leave out ``RIDGE_EXCLUDED_SESSIONS`` (as the decoding score does);
their ``::norm`` is recomputed without those sessions on both the model and the Poisson side.
``encoding/split`` names the split the ``encoding/*`` columns are on (``test``); the
``decoding/*`` columns are always test,
and ``decoding/complete`` is false when the cohort misses sessions (the ``decoding``
category score is then empty unless allow_incomplete_decoding_cohort).
``decoding/test_r2`` is POYO decoding (cursor-velocity R²), the mean over sessions; it is
the ``decoding_poyo`` score, one of the two parts of the ``decoding`` composite (POYO +
linear). ``downstream_decoding/behavioral_classification_trial_target_id`` is linear
decoding as trial-target accuracy, which no score includes.
"""

from __future__ import annotations

import math
import statistics
from typing import TYPE_CHECKING, Any, TypeGuard

if TYPE_CHECKING:
    from ntt.evaluation.aggregation.downstream_records import DownstreamScoreView

from ntt.evaluation.aggregation.category_scores import (
    _is_regression_decoder,
    apply_alternative_floors,
    ridge_session_values,
)
from ntt.evaluation.aggregation.comparability import RIDGE_EXCLUDED_SESSIONS
from ntt.evaluation.aggregation.comparison import (
    ALTERNATIVE_FLOORS,
    MetricComparison,
    normalized_score_vs_baseline,
)
from ntt.evaluation.core.types import MetricDirection


def _finite(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)


def _mean_std_n(values: list[float]) -> tuple[float | None, float | None, int]:
    vals = [float(v) for v in values if _finite(v)]
    if not vals:
        return None, None, 0
    mean = statistics.fmean(vals)
    std = statistics.stdev(vals) if len(vals) >= 2 else None
    return mean, std, len(vals)


def _emit(out: dict, base: str, mean, std, n) -> None:
    out[base] = mean
    out[f"{base}::std"] = std
    out[f"{base}::n"] = n


def _comparison_ok(comp: Any) -> TypeGuard[dict]:
    """True if the eval job stored a valid Poisson comparison for this metric (status "ok" and
    a normalized score present). ``::norm`` is written only for such metrics."""
    return (
        isinstance(comp, dict)
        and comp.get("status") == "ok"
        and comp.get("normalized_score") is not None
    )


def ridge_normalized_score(
    key: str, recon: dict, excluded: frozenset[str] = RIDGE_EXCLUDED_SESSIONS
) -> float | None:
    """Poisson-normalized score for ridge entry ``key``, leaving out ``excluded`` sessions; return
    None if invalid. Recompute it from session means because eval’s stored score includes a
    broken session."""
    comp = recon["comparison_vs_poisson"][key]
    model = ridge_session_values(key, recon, excluded=excluded)
    poisson = ridge_session_values(key, recon, poisson=True, excluded=excluded)
    if set(model) != set(poisson):
        raise ValueError(
            f"Ridge entry {key!r}: model and Poisson scores cover different sessions "
            f"(model only: {sorted(set(model) - set(poisson))}, "
            f"Poisson only: {sorted(set(poisson) - set(model))})"
        )
    if not model:
        raise ValueError(f"Ridge entry {key!r}: no session left to normalize over")
    score, _reason = normalized_score_vs_baseline(
        key,
        statistics.fmean(model.values()),
        statistics.fmean(poisson.values()),
        comp["ceiling_value"],
        MetricDirection(comp["direction"]),
        comp["value_range"],
    )
    return score


def _comparisons_with_alternative_floors(recon: dict) -> dict:
    """The saved ``comparison_vs_poisson`` dicts, with the ``ALTERNATIVE_FLOORS`` metrics
    scored against their reference (recomputed for reports saved before it)."""
    saved = recon.get("comparison_vs_poisson", {})
    if not any(name in saved for name in ALTERNATIVE_FLOORS):
        return saved
    comparisons, _ = apply_alternative_floors(
        {name: MetricComparison.from_dict(d) for name, d in saved.items()}, recon
    )
    return {**saved, **{n: comparisons[n].to_dict() for n in ALTERNATIVE_FLOORS if n in saved}}


def _recon_columns(out: dict, recon: dict) -> None:
    agg = recon.get("aggregated", {}).get("results", {})
    per_session = recon.get("per_session", {})
    cmp = _comparisons_with_alternative_floors(recon)
    for metric, result in agg.items():
        category = result.get("category")
        base = f"{category}/{metric}"
        if _is_regression_decoder(metric, result):
            # Ridge: leave out ``RIDGE_EXCLUDED_SESSIONS``, as in the decoding score; eval’s
            # stored mean includes that broken session.
            scores_by_session = ridge_session_values(metric, recon)
            for session, score in scores_by_session.items():
                if math.isinf(score):
                    raise ValueError(
                        f"Ridge entry {metric!r}: infinite score in session {session!r}"
                    )
            session_scores = list(scores_by_session.values())
            mean = statistics.fmean(session_scores) if session_scores else None
            std = statistics.stdev(session_scores) if len(session_scores) >= 2 else None
            _emit(out, base, mean, std, len(session_scores))
            if metric in cmp:
                # Normalized score over the same sessions; skip if invalid.
                norm = ridge_normalized_score(metric, recon)
                if norm is not None:
                    out[f"{base}::norm"] = norm
            continue
        session_vals = [
            entry.get("results", {}).get(metric, {}).get("raw_value")
            for entry in per_session.values()
            if isinstance(entry, dict)
        ]
        mean, std, n = _mean_std_n([v for v in session_vals if v is not None])
        if mean is None:  # not in per_session — fall back to the aggregated scalar
            raw = result.get("raw_value")
            mean, std, n = (float(raw) if _finite(raw) else None), None, 0
        _emit(out, base, mean, std, n)
        comp = cmp.get(metric)
        if _comparison_ok(comp):
            out[f"{base}::norm"] = comp["normalized_score"]


def _robustness_columns(out: dict, report: dict, prefix: str = "robustness") -> None:
    # Raw dump: emit ALL metrics (unlike the scored subset, which filters to
    # robustness_input/ and drops baselines). Shared by the M1 input-corruption
    # report and the M2 latent-noise report — both use the same
    # {"results": {name: {raw_value, metadata}}} shape.
    for metric, result in report.get("results", {}).items():
        md = result.get("metadata") or {}
        base = f"{prefix}/{metric}"
        raw = result.get("raw_value")
        out[base] = float(raw) if _finite(raw) else None
        out[f"{base}::std"] = md.get("std")
        out[f"{base}::n"] = md.get("n_sessions")


def _encoding_columns(out: dict, encoding: DownstreamScoreView) -> None:
    out["encoding/split"] = encoding.split
    for metric, entry in encoding.aggregated.items():
        base = f"encoding/{metric}"
        session_means = [sess.get(metric) for sess in encoding.per_session.values()]
        _, std, n_from_sessions = _mean_std_n([v for v in session_means if v is not None])
        n = entry.get("n")
        out[base] = entry.get("value")
        out[f"{base}::std"] = std
        out[f"{base}::n"] = n_from_sessions if n is None else n


def _decoding_columns(out: dict, decoding: DownstreamScoreView) -> None:
    out["decoding/complete"] = decoding.complete
    r2_vals = [s.get("r2") for s in decoding.per_session.values()]
    mean, std, n = _mean_std_n([v for v in r2_vals if v is not None])
    if n:
        _emit(out, "decoding/test_r2", mean, std, n)


def flatten_tokenizer_metrics(
    recon: dict | None,
    m1: dict | None,
    encoding: DownstreamScoreView | None,
    decoding: DownstreamScoreView | None,
    *,
    m2: dict | None = None,
    snr: dict | None = None,
) -> dict[str, Any]:
    """Flatten a tokenizer's eval artifacts into one wide row.

    ``encoding`` / ``decoding`` are downstream records read by ``load_downstream_record``.
    ``m2`` (latent-noise robustness) is keyword-only so existing positional
    callers keep working; its metrics land under ``robustness_latent/``.
    """
    out: dict[str, Any] = {}
    if recon:
        _recon_columns(out, recon)
    if m1:
        _robustness_columns(out, m1)
    if m2:
        _robustness_columns(out, m2, prefix="robustness_latent")
    if snr:
        _robustness_columns(out, snr, prefix="latent_snr")
    if encoding:
        _encoding_columns(out, encoding)
    if decoding:
        _decoding_columns(out, decoding)
    return out


import argparse

import pandas as pd

from ntt.evaluation.aggregation import tracker_io
from ntt.evaluation.aggregation.downstream_records import load_downstream_record


def _resolve_row_artifacts(row: dict) -> dict:
    """Load whatever source artifacts a row points to. Supports run-URL cells
    (resolved to mlflow-artifacts URIs) and, for local testing, a bare local
    path in mlflow_recon."""
    loaded: dict = {}
    uris = tracker_io.row_to_source_uris(row)
    if uris is None:
        recon_cell = (row.get("mlflow_recon") or "").strip()
        if recon_cell and not recon_cell.startswith(("http", "mlflow")):
            uris = {"recon": recon_cell}  # bare local path
        else:
            return loaded
    for key, loader in (
        ("recon", tracker_io.load_json_artifact),
        ("m1", tracker_io.load_json_artifact),
        ("m2", tracker_io.load_json_artifact),
        ("encoding", lambda uri: load_downstream_record(uri, "encoding")),
        ("decoding", lambda uri: load_downstream_record(uri, "decoding")),
    ):
        if key in uris:
            try:
                loaded[key] = loader(uris[key])
            except Exception as exc:  # one bad artifact must not sink the row
                print(f"    WARN: failed to load {key} from {uris[key]}: {exc}")
    return loaded


def build_table(source: str) -> pd.DataFrame:
    _fieldnames, rows = tracker_io.read_tracker_rows(source)
    records: list[dict] = []
    for row in rows:
        slug = (row.get("tokenizer") or "").strip()
        if not slug:
            continue
        loaded = _resolve_row_artifacts(row)
        if not loaded:
            continue
        flat = flatten_tokenizer_metrics(
            loaded.get("recon"),
            loaded.get("m1"),
            loaded.get("encoding"),
            loaded.get("decoding"),
            m2=loaded.get("m2"),
        )
        records.append({"tokenizer": slug, **flat})
    df = pd.DataFrame(records)
    # deterministic column order: tokenizer first, then sorted metric columns
    metric_cols = sorted(c for c in df.columns if c != "tokenizer")
    return df[["tokenizer", *metric_cols]] if "tokenizer" in df.columns else df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="Tracker CSV path or Google Sheets URL")
    parser.add_argument("--output", "-o", help="Output CSV path (required for URL input)")
    parser.add_argument("--mlflow-uri", help="Override MLflow tracking URI")
    args = parser.parse_args()
    if args.mlflow_uri:
        import mlflow

        mlflow.set_tracking_uri(args.mlflow_uri)
    df = build_table(args.csv)
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Wrote {len(df)} rows x {df.shape[1]} cols -> {args.output}")
    else:
        print(df.to_csv(index=False))


if __name__ == "__main__":
    main()
