"""Run evaluation jobs in order, using rows from a checkpoint or tokenizer-configuration CSV."""

import argparse
import csv
import re
import subprocess
from pathlib import Path


def _quote(value: str) -> str:
    """Quote a value for a Hydra override so that it reaches the config as a literal string."""
    # OmegaConf: escape ${ so it is not an interpolation, doubling any backslashes before it
    value = re.sub(r"(\\*)\$\{", lambda m: m.group(1) * 2 + "\\${", value)
    # Hydra: inside quotes, backslashes before a quote or at the end are escapes
    value = re.sub(r"\\+(?='|$)", lambda m: m.group(0) * 2, value)
    return "'" + value.replace("'", "\\'") + "'"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv", type=Path, help="CSV file with tokenizer, ckpt_uri, and tokenizer_config columns"
    )
    parser.add_argument(
        "--data-dir", required=True, type=Path, help="Directory with preprocessed brainsets"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--job", default="eval_all", help="Hydra job config (default: eval_all)")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Extra Hydra override; use this flag again to add more",
    )
    args = parser.parse_args()

    # utf-8-sig drops the byte-order mark that Excel adds to CSV exports
    with args.csv.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not {"tokenizer", "ckpt_uri"} <= set(reader.fieldnames):
            raise ValueError(
                "CSV must have tokenizer and ckpt_uri columns; tokenizer_config is optional"
            )
        rows = list(reader)

    names = set()
    commands = []
    for line, row in enumerate(rows, start=2):
        name = (row["tokenizer"] or "").strip()
        ckpt = (row["ckpt_uri"] or "").strip()
        spec = (row.get("tokenizer_config") or "").strip()
        if not name or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for c in name
        ):
            raise ValueError(
                f"Row {line}: tokenizer may contain only letters, digits, underscores, and hyphens"
            )
        if name in names:
            raise ValueError(f"Row {line}: tokenizer name {name!r} appears more than once")
        names.add(name)
        if bool(ckpt) == bool(spec):
            raise ValueError(
                f"Row {line}: enter a value in exactly one of ckpt_uri or tokenizer_config"
            )
        source = (
            f"job.ckpt_path={_quote(ckpt)}" if ckpt else f"tokenizer@job.tokenizer_config={spec}"
        )
        commands.append(
            [
                "uv",
                "run",
                "ntt",
                f"job={args.job}",
                source,
                f"paths.data_dir={_quote(str(args.data_dir.resolve()))}",
                f"paths.run_name={name}",
                f"job.output_path={_quote(str(args.output_dir.resolve() / name))}",
                *args.override,
            ]
        )

    for command in commands:
        subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    main()
