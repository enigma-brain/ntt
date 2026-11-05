"""Offline probe evaluation on extracted NuCLR embeddings.

Loads embeddings from a ``.pt`` file (produced by ``extract_embeddings.py``),
fetches brain region labels from brainsets, and runs logistic regression probes.

Usage::

    uv run python -m ntt.contrib.nuclr.eval.eval_probes \
        --embeddings-path embeddings.pt \
        --data-root <brainsets_root> \
        --brainset IBL_brainwide_2023 \
        [--method l1out] \
        [--output-json results.json]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from .labels import get_unit_number_to_label
from .probe import run_probe

logger = logging.getLogger(__name__)


def eval_probes(
    embeddings_path: str | Path,
    data_root: str | Path,
    brainset: str,
    method: str = "l1out",
    output_json: str | Path | None = None,
) -> dict[str, float]:
    """Run probe evaluation on extracted embeddings.

    Args:
        embeddings_path: Path to ``.pt`` file with keys
            ``embs``, ``unit_ids``, ``session_ids``.
        data_root: Root directory containing brainsets.
        brainset: Brainset name for label extraction.
        method: Probe method (``"l1out"`` or ``"kfold"``).
        output_json: Optional path to save results as JSON.

    Returns:
        Dict of metric name → value.
    """
    embeddings_path = Path(embeddings_path)

    # Load embeddings
    logger.info("Loading embeddings from %s", embeddings_path)
    # Embeddings file holds only tensors/ndarrays/dicts, so restrict unpickling.
    data = torch.load(embeddings_path, weights_only=True, map_location="cpu")
    embs = data["embs"].float().numpy()
    emb_unit_ids = data["unit_ids"]

    logger.info("Loaded %d embeddings (dim=%d)", embs.shape[0], embs.shape[1])

    # Load brain region labels indexed by composite key (session_id/unit_number)
    # — same format as the embedding unit_ids.
    logger.info("Loading brain region labels from %s", brainset)
    labels_df = get_unit_number_to_label(data_root, brainset)

    # Match embeddings to labels via composite key
    matched_embs = []
    matched_labels = []
    matched_sessions = []

    for i, uid in enumerate(emb_unit_ids):
        uid_str = str(uid)
        if uid_str in labels_df.index:
            matched_embs.append(embs[i])
            matched_labels.append(labels_df.loc[uid_str, "brain_region"])
            matched_sessions.append(labels_df.loc[uid_str, "session_id"])

    if not matched_embs:
        raise ValueError("No embeddings matched to brain region labels")

    matched_embs_arr = np.array(matched_embs)
    matched_labels_arr = np.array(matched_labels)
    matched_sessions_arr = np.array(matched_sessions)

    logger.info(
        "Matched %d/%d embeddings to labels (%d unique regions)",
        len(matched_embs_arr),
        len(embs),
        len(np.unique(matched_labels_arr)),
    )

    # Determine hash_sessions based on brainset
    hash_sessions = brainset == "IBL_brainwide_2023"

    # Run probe
    results = run_probe(
        embeddings=matched_embs_arr,
        labels=matched_labels_arr,
        session_ids=matched_sessions_arr,
        method=method,
        hash_sessions=hash_sessions,
    )

    # Print results
    print(f"\n{'=' * 50}")
    print(f"Probe Results: {brainset} ({method})")
    print(f"{'=' * 50}")
    print(f"  Units evaluated: {len(matched_embs_arr)}")
    print(f"  Regions: {sorted(np.unique(matched_labels_arr).tolist())}")
    for metric, value in sorted(results.items()):
        print(f"  {metric}: {value:.4f}")
    print(f"{'=' * 50}\n")

    if output_json is not None:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        full_results = {
            "brainset": brainset,
            "method": method,
            "n_units": len(matched_embs_arr),
            "n_regions": int(len(np.unique(matched_labels_arr))),
            "metrics": results,
        }
        with open(output_json, "w") as f:
            json.dump(full_results, f, indent=2)
        logger.info("Saved results to %s", output_json)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run brain region probes on extracted NuCLR embeddings",
    )
    parser.add_argument("--embeddings-path", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--brainset", type=str, required=True)
    parser.add_argument("--method", type=str, default="l1out", choices=["l1out", "kfold"])
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    eval_probes(
        embeddings_path=args.embeddings_path,
        data_root=args.data_root,
        brainset=args.brainset,
        method=args.method,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
