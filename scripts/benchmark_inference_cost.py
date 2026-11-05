"""Benchmark tokenizer inference cost (FLOPs, latency, params) from real checkpoints.

Drives ``ntt.evaluation.inference_cost.compute_inference_cost`` over a manifest
CSV of checkpoints.

Manifest CSV columns:
    family (required):           Tokenizer family label, e.g. "compressed_sensing",
                                  "ldns", "wanx_A", "cosmos", "calm", "PCA".
                                  Matched case-insensitively against
                                  --exclude-families, and used (also
                                  case-insensitively) to look up
                                  inference_cost.py's
                                  _INFERENCE_IRRELEVANT_BUFFER_NAMES; does
                                  not otherwise affect measurement mechanics.
    variant (required):          Human-readable checkpoint variant label
                                  (e.g. "wanx_A_vae_bce_soft1_b1e-7"). Written
                                  through to the output CSV, overriding the
                                  spec-KIND-derived default from
                                  InferenceCostResult.
    checkpoint_path (required):  Path to the Lightning .ckpt file.
    window_length_seconds (optional): Forces this window length instead of
                                  the checkpoint's native value. Blank = native.
    n_neurons (optional):        Forces this neuron count instead of the
                                  checkpoint's native value. Blank = native.
    batch_sizes (optional):      Semicolon-separated list of batch sizes to
                                  sweep for this row, e.g. "1;4;16". Blank =
                                  use --batch-sizes.
    native_or_forced (optional): Free-form override for the reported
                                  native_or_forced field. If blank, it is
                                  inferred: "native" when neither
                                  window_length_seconds nor n_neurons is
                                  overridden for this row, else "forced".

Usage:
    uv run python scripts/benchmark_inference_cost.py \\
        --manifest manifest.csv --output inference_cost_results.csv \\
        --exclude-families perceiver,wanx_B
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from typing import Any

import pandas as pd
import torch

# Ensures every tokenizer KIND (including contrib families) is registered
# before create_tokenizer() looks it up.
import ntt.contrib.tokenizer  # noqa: F401
from ntt.core.model.tokenizer import create_tokenizer
from ntt.evaluation.inference_cost import (
    InferenceCostResult,
    compute_inference_cost,
    load_tokenizer_weights_from_checkpoint,
    resolve_spec_from_checkpoint,
)

# Script-added columns not on InferenceCostResult itself (see process_row).
_SCRIPT_ADDED_COLUMNS = ["checkpoint_path", "weights_loaded", "weights_load_detail"]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {value}")
    return parsed


def _fmt_csv(value: Any) -> Any:
    """Render NaN floats as 'N/A' for CSV human-readability consistency."""
    if isinstance(value, float) and math.isnan(value):
        return "N/A"
    return value


def _parse_optional_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)) or value == "":
        return None
    return float(value)


def _parse_optional_int(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and math.isnan(value)) or value == "":
        return None
    return int(value)


def _validate_batch_sizes(batch_sizes: list[int], source: str) -> list[int]:
    if not batch_sizes:
        raise ValueError(f"{source} produced no batch sizes; at least one is required.")
    non_positive = [b for b in batch_sizes if b <= 0]
    if non_positive:
        raise ValueError(
            f"{source} contains non-positive batch size(s) {non_positive}; "
            "batch sizes must be >= 1."
        )
    return batch_sizes


def _parse_batch_sizes(value: Any, default: list[int]) -> list[int]:
    if value is None or (isinstance(value, float) and math.isnan(value)) or value == "":
        return default
    parsed = [int(v.strip()) for v in str(value).split(";") if v.strip()]
    return _validate_batch_sizes(parsed, "manifest 'batch_sizes' column")


def process_row(
    row: pd.Series,
    default_batch_sizes: list[int],
    warmup: int,
    repeats: int,
    excluded_families: frozenset[str],
) -> list[dict]:
    """Run compute_inference_cost for one manifest row, once per swept batch size."""
    family = str(row["family"])
    variant = str(row["variant"])
    checkpoint_path = str(row["checkpoint_path"])

    # Case-insensitive: the manifest's own documented convention mixes
    # casings across families ("wanx_A", "PCA"), and inference_cost.py's
    # family-keyed lookups are case-insensitive for the same reason — an
    # exclusion list that wasn't would let a differently-cased row for an
    # excluded family through silently.
    if family.lower() in excluded_families:
        print(f"Skipping excluded family '{family}' (variant={variant})")
        return []

    window_length_override = _parse_optional_float(row.get("window_length_seconds"))
    n_neurons_override = _parse_optional_int(row.get("n_neurons"))
    batch_sizes = _parse_batch_sizes(row.get("batch_sizes"), default_batch_sizes)

    overrides: dict[str, Any] = {}
    if window_length_override is not None:
        overrides["window_length_seconds"] = window_length_override
    if n_neurons_override is not None:
        overrides["n_neurons"] = n_neurons_override

    native_or_forced = row.get("native_or_forced")
    native_or_forced_blank = native_or_forced is None or native_or_forced == ""
    if not native_or_forced_blank and isinstance(native_or_forced, float):
        native_or_forced_blank = math.isnan(native_or_forced)
    if native_or_forced_blank:
        native_or_forced = "native" if not overrides else "forced"

    print(
        f"Processing: family={family} variant={variant} "
        f"ckpt={checkpoint_path} overrides={overrides}"
    )

    try:
        spec = resolve_spec_from_checkpoint(checkpoint_path, overrides=overrides)
    except Exception as e:
        print(f"  -> Failed to resolve spec for {checkpoint_path}: {type(e).__name__}: {e}")
        return []

    rows: list[dict] = []
    for batch_idx, batch_size in enumerate(batch_sizes):
        # Fresh tokenizer + fresh weight load every batch_size iteration —
        # `spec` is read-only across every consumer (create_tokenizer,
        # make_dummy_input, compute_inference_cost), so it's safe to share,
        # but the tokenizer module itself may carry batch-shaped cached/
        # mutable state (e.g. LDNS's S4 l_kernel buffer) that
        # must not leak from one batch size into the next.
        tokenizer = None
        try:
            try:
                tokenizer = create_tokenizer(spec)
                tokenizer.eval()  # hard rule: always .eval() immediately after creation
                # Load the checkpoint's REAL weights on top of the freshly
                # (randomly) initialized module — required for correctness on
                # buffer-bearing families (fitted PCA projection, LDNS's
                # l_kernel buffer, etc.), and informative even where it
                # "shouldn't" matter. A forced n_neurons/window_length_seconds
                # override can legitimately break the strict load (shapes
                # changed) — that's expected and recorded, not swallowed.
                weights_loaded, weights_load_detail = load_tokenizer_weights_from_checkpoint(
                    checkpoint_path, tokenizer
                )
                print(
                    f"  -> batch={batch_size} weights_loaded={weights_loaded} "
                    f"({weights_load_detail})"
                )
            except Exception as e:
                print(
                    f"  -> Failed to build tokenizer for batch_size={batch_size} "
                    f"({checkpoint_path}): {type(e).__name__}: {e}"
                )
                remaining = batch_sizes[batch_idx:]
                for bs in remaining:
                    rows.append(
                        {
                            "family": family,
                            "variant": variant,
                            "checkpoint_path": checkpoint_path,
                            "batch_size": bs,
                            "shape_ok": False,
                            "shape_detail": (f"FAILED (tokenizer build): {type(e).__name__}: {e}"),
                        }
                    )
                break

            try:
                result = compute_inference_cost(
                    spec,
                    tokenizer,
                    family=family,
                    batch_size=batch_size,
                    warmup=warmup,
                    repeats=repeats,
                    native_or_forced=str(native_or_forced),
                )
                result_dict = result.to_dict()
                # Manifest-supplied variant label takes precedence over the
                # spec-KIND-derived default set by compute_inference_cost.
                result_dict["variant"] = variant
                result_dict["checkpoint_path"] = checkpoint_path
                result_dict["weights_loaded"] = weights_loaded
                result_dict["weights_load_detail"] = weights_load_detail
                enc_flops = (
                    f"{result.encoder_flops_counted}+{result.encoder_flops_manual_supplement}"
                )
                dec_flops = (
                    f"{result.decoder_flops_counted}+{result.decoder_flops_manual_supplement}"
                )
                print(
                    f"  -> batch={batch_size} shape_ok={result.shape_ok} "
                    f"enc_flops={enc_flops} dec_flops={dec_flops} "
                    f"enc_lat={result.encoder_latency_ms_mean:.3f}ms "
                    f"dec_lat={result.decoder_latency_ms_mean:.3f}ms"
                )
                rows.append(result_dict)
            except Exception as e:
                print(f"  -> Failed batch_size={batch_size}: {type(e).__name__}: {e}")
                rows.append(
                    {
                        "family": family,
                        "variant": variant,
                        "checkpoint_path": checkpoint_path,
                        "batch_size": batch_size,
                        "weights_loaded": weights_loaded,
                        "weights_load_detail": weights_load_detail,
                        "shape_ok": False,
                        "shape_detail": f"FAILED: {type(e).__name__}: {e}",
                    }
                )
        finally:
            del tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return rows


def main(
    manifest_csv: str,
    output_csv: str,
    batch_sizes: list[int],
    warmup: int,
    repeats: int,
    excluded_families: frozenset[str],
) -> None:
    print(f"Reading {manifest_csv}...")
    df = pd.read_csv(manifest_csv)

    required_cols = {"family", "variant", "checkpoint_path"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Manifest CSV is missing required column(s): {sorted(missing)}")

    all_rows: list[dict] = []
    for _, row in df.iterrows():
        all_rows.extend(
            process_row(
                row,
                default_batch_sizes=batch_sizes,
                warmup=warmup,
                repeats=repeats,
                excluded_families=excluded_families,
            )
        )

    if not all_rows:
        print("No results produced (all rows skipped or failed).")
        # Derived from InferenceCostResult's own fields (+ this script's
        # added columns) rather than hand-copied, so adding a dataclass
        # field can't silently leave this header stale — the mismatch would
        # otherwise only surface on the all-rows-failed path nobody eyeballs.
        empty_columns = [f.name for f in dataclasses.fields(InferenceCostResult)] + (
            _SCRIPT_ADDED_COLUMNS
        )
        pd.DataFrame(columns=empty_columns).to_csv(output_csv, index=False)
        return

    out_df = pd.DataFrame(all_rows)
    if "family_extra_config" in out_df.columns:
        # Dict column would otherwise stringify as a Python repr (needs
        # ast.literal_eval downstream); JSON is the more portable choice.
        out_df["family_extra_config"] = out_df["family_extra_config"].map(
            lambda v: json.dumps(v) if isinstance(v, dict) else v
        )
    for col in out_df.columns:
        out_df[col] = out_df[col].map(_fmt_csv)
    out_df.to_csv(output_csv, index=False)
    print(f"\nProcessing completed. Saved {len(out_df)} rows to {output_csv}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", default="inference_cost_manifest.csv", help="Manifest CSV path."
    )
    parser.add_argument("--output", default="inference_cost_results.csv", help="Output CSV path.")
    parser.add_argument(
        "--batch-sizes",
        default="1",
        help=(
            "Comma-separated default batch sizes to sweep for rows without "
            "a 'batch_sizes' column value."
        ),
    )
    parser.add_argument(
        "--warmup",
        type=_non_negative_int,
        default=3,
        help="Untimed warmup calls before each timed loop.",
    )
    parser.add_argument(
        "--repeats", type=_positive_int, default=10, help="Timed calls per timed loop."
    )
    parser.add_argument(
        "--exclude-families",
        default="perceiver,wanx_B",
        help=(
            "Comma-separated family labels to always skip regardless of manifest "
            "content (e.g. families with no checkpoint, or excluded from a "
            "specific pass). Case-insensitive. Pass an empty string to exclude "
            "nothing."
        ),
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()

    parsed_batch_sizes = _validate_batch_sizes(
        [int(v.strip()) for v in args.batch_sizes.split(",") if v.strip()],
        "--batch-sizes",
    )
    parsed_excluded_families = frozenset(
        v.strip().lower() for v in args.exclude_families.split(",") if v.strip()
    )

    main(
        args.manifest,
        args.output,
        parsed_batch_sizes,
        args.warmup,
        args.repeats,
        parsed_excluded_families,
    )
