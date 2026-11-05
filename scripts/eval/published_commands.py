"""Print or run the ntt commands that score a Hugging Face tokenizer as its published scores were.

    uv run python scripts/eval/published_commands.py conv-vae-100to1 \\
        --data-dir /path/to/brainsets/processed --revision v1.0

The settings come from scripts/eval/published_protocol.yaml, config/job/published_eval*.yaml and
config/downstream/encoding/. Commands are printed, one per line; --run runs them in order.
Outputs land in <output-dir>/<model>/ (default output dir: results/<revision>): eval, eval_snr,
eval_m1, eval_m2 (intrinsic passes) and encoding. Decoding results land in the downstream
submodule's experiments_benchmark/ folder (the run prints where). Afterwards, --tracker <dir>
writes the aggregation input; give it the same --output-dir and --revision. --revision is a tag
or a commit, not the branch main, whose files can change under the same name.
"""

import argparse
import csv
import shlex
import subprocess
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = Path(__file__).with_name("published_protocol.yaml")
REPO_ID = "the-enigma-project/ntt-tokenizers"
AXES = ("recon", "snr", "m1", "m2", "encoding", "decoding")
PASSES = {"recon": "main", "snr": "snr", "m1": "input_robustness", "m2": "latent_robustness"}


def load_protocol() -> dict[str, dict]:
    return yaml.safe_load(PROTOCOL.read_text())["models"]


def hub_checkpoint(model: str, repo_id: str, revision: str | None) -> str:
    return f"hf://{repo_id}/{model}" + (f"@{revision}" if revision else "")


def commands(
    model: str,
    axes: list[str],
    *,
    data_dir: Path,
    output_dir: Path,
    revision: str | None,
    repo_id: str = REPO_ID,
    mlflow: bool = False,
) -> list[list[str]]:
    """The ``ntt`` arguments (after ``uv run ntt``) for ``model`` on ``axes``, one list per run.

    Without ``mlflow`` no MLflow server is needed, even with ``MLFLOW_TRACKING_URI`` set:
    encoding and decoding run with MLflow logging off. With it, both log to that server.
    """
    models = load_protocol()
    if model not in models:
        raise ValueError(f"Unknown model {model!r}; known models: {', '.join(models)}")
    entry = models[model]
    ckpt = hub_checkpoint(model, repo_id, revision)
    out = output_dir / model
    runs = []

    intrinsic = [axis for axis in axes if axis in PASSES]
    if intrinsic:
        has_own = (REPO_ROOT / "config/job" / f"published_eval_{model}.yaml").is_file()
        runs.append(
            [
                f"job=published_eval_{model}" if has_own else "job=published_eval",
                f"paths.run_name={model}",
                f"paths.data_dir={data_dir}",
                f"job.ckpt_path={ckpt}",
                f"job.output_path={out / 'eval'}",
                *(f"job.{p}.enabled={str(a in intrinsic).lower()}" for a, p in PASSES.items()),
            ]
        )

    if "encoding" in axes and entry["encoding"]:
        enc = entry["encoding"]
        extra = [f"data.dataset_path={data_dir / 'cowley_v4_compact_2024'}", f"tags.model={model}"]
        if not mlflow:
            extra.append("trainer.logging=false")
        runs.append(
            [
                "job=downstream_encoding",
                f"paths.run_name={model}",
                f"job.ckpt={ckpt}",
                f"job.recipe=config/downstream/encoding/{enc['recipe']}.yaml",
                f"job.loss_mode={enc['loss_mode']}",
                f"job.out={out / 'encoding'}",
                "job.seed=42",
                "job.extra=[" + ",".join(f'"{e}"' for e in extra) + "]",
            ]
        )

    if "decoding" in axes:
        dec = entry.get("decoding", {})
        gpus = " ".join(str(i) for i in range(dec.get("gpus", 1)))
        extra = dec.get("extra", [])
        runs.append(
            [
                "job=downstream_decoding",
                "job.mode=benchmark",
                f"paths.run_name={model}",
                f"job.ckpt={ckpt}",
                f"job.data_root={data_dir}",
                "job.eval_selection=decoding/ntt_decoding_benchmark_eval",
                "job.epochs=400",
                "job.model_cfg=poyo_1.3M_no_crossattn",
                "job.unit_embedding_mode=add",
                f'job.gpus="{gpus}"' if " " in gpus else f"job.gpus={gpus}",
                "job.seed=42",
                f"job.mlflow={str(mlflow).lower()}",
                *(["job.extra=[" + ",".join(f'"{e}"' for e in extra) + "]"] if extra else []),
            ]
        )
    return runs


TRACKER_COLUMNS = {  # tracker column -> output folder under <output-dir>/<model>/
    "mlflow_recon": "eval",
    "mlflow_latent_snr": "eval_snr",
    "mlflow_latent_input_corruption": "eval_m1",
    "mlflow_latent_latent_corruption": "eval_m2",
    "mlflow_encoding": "encoding",
}


def tracker_rows(
    models: list[str],
    output_dir: Path,
    downstream_dir: Path,
    *,
    revision: str | None,
    repo_id: str = REPO_ID,
) -> dict[str, list[dict[str, str]]]:
    """Aggregation input rows (local output folders, see docs/eval_aggregation.md), by table.

    Models without reconstruction results in ``output_dir`` are left out; cells of
    evaluations that have no output folder stay empty. Decoding results are those of
    ``revision``; the other outputs are whatever ``output_dir`` holds, so use one output
    folder per revision.
    """
    from ntt.jobs.downstream import hub_benchmark_tag  # heavy; only --tracker needs it

    protocol = load_protocol()
    tables: dict[str, list[dict[str, str]]] = {}
    for model in models:
        if not (output_dir / model / "eval").is_dir():
            print(f"# {model}: no results in {output_dir / model}, left out")
            continue
        entry = protocol[model]
        row = {"tokenizer": model}
        for column, folder in TRACKER_COLUMNS.items():
            no_score = column == "mlflow_encoding" and not entry["encoding"]
            path = output_dir / model / folder
            row[column] = "" if no_score or not path.is_dir() else str(path)
        # The decoding benchmark's folder is named after the model and the Hub revision.
        tag = hub_benchmark_tag(hub_checkpoint(model, repo_id, revision))
        decoding = downstream_dir / "experiments_benchmark" / tag
        row["mlflow_decoding"] = str(decoding) if decoding.is_dir() else ""
        tables.setdefault(entry["table"], []).append(row)
    return tables


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("models", nargs="+", help="Hub model names, or 'all'")
    parser.add_argument("--data-dir", type=Path, help="Directory with preprocessed brainsets")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where the outputs land; use one per Hub revision (default: results/<revision>)",
    )
    parser.add_argument("--axes", nargs="+", choices=AXES, default=list(AXES))
    parser.add_argument(
        "--revision",
        default="v1.0",
        help="Hub tag or commit (default: v1.0); not main, whose files can change under the same "
        "name. With --tracker, the revision whose decoding results are used",
    )
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Extra Hydra override added to every command; use this flag again to add more",
    )
    parser.add_argument("--run", action="store_true", help="Run the commands instead of printing")
    parser.add_argument(
        "--mlflow",
        action="store_true",
        help="Log encoding and decoding to the MLflow server in MLFLOW_TRACKING_URI "
        "(default: both run with MLflow off, whether or not MLFLOW_TRACKING_URI is set)",
    )
    parser.add_argument(
        "--tracker",
        type=Path,
        help="Instead of commands, write the aggregation input tracker_<table>.csv to this "
        "folder, from the finished outputs",
    )
    parser.add_argument(
        "--downstream-dir",
        type=Path,
        default=REPO_ROOT / "downstream",
        help="The downstream submodule (where decoding results land); default: ./downstream",
    )
    args = parser.parse_args()
    if args.revision == "main":
        parser.error("--revision main: give a tag (e.g. v1.0) or a commit; main can change")

    protocol = load_protocol()
    output_dir = (args.output_dir or Path("results") / args.revision).resolve()
    models = list(protocol) if args.models == ["all"] else args.models
    if args.tracker:
        args.tracker.mkdir(parents=True, exist_ok=True)
        tables = tracker_rows(
            models,
            output_dir,
            args.downstream_dir.resolve(),
            revision=args.revision,
            repo_id=args.repo_id,
        )
        for table, rows in tables.items():
            path = args.tracker / f"tracker_{table}.csv"
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(f"{path}: {len(rows)} models")
        return
    if args.data_dir is None:
        parser.error("--data-dir is required to print or run commands")
    for model in models:
        runs = commands(
            model,
            args.axes,
            data_dir=args.data_dir.resolve(),
            output_dir=output_dir,
            revision=args.revision,
            repo_id=args.repo_id,
            mlflow=args.mlflow,
        )
        if "encoding" in args.axes and model in protocol and not protocol[model]["encoding"]:
            print(f"# {model}: no encoding score (see published_protocol.yaml)")
        for run in [["uv", "run", "ntt", *cmd, *args.override] for cmd in runs]:
            if args.run:
                subprocess.run(run, check=True, cwd=REPO_ROOT)
            else:
                print(shlex.join(run))


if __name__ == "__main__":
    main()
