"""Lightning callback for online NuCLR probe evaluation.

Runs brain region classification probes on frozen embeddings collected
during validation. Hooks into the existing ``NuclrContrastiveLitModule``
validation loop and uses the ``EmbeddingCollector`` it maintains.

The callback runs every ``eval_every_n_epochs`` epochs and logs probe
metrics (balanced accuracy, F1) to the trainer's logger.
"""

from __future__ import annotations

import logging

import lightning as L
import numpy as np
import pandas as pd

from .eval.labels import get_unit_number_to_label
from .eval.probe import run_probe

logger = logging.getLogger(__name__)


class NuclrProbeCallback(L.Callback):
    """Runs brain region classification probes on frozen embeddings.

    Integrates with ``NuclrContrastiveLitModule`` by reading embeddings
    from ``pl_module._val_probe_embeddings`` and session mappings from
    ``pl_module._val_unit_sessions`` after each validation epoch.

    Args:
        data_root: Root directory containing brainsets.
        brainset: Brainset name for label extraction.
        eval_every_n_epochs: Run probes every N epochs. Set > 1 to reduce
            overhead (probes are cheap but not free).
        method: Probe method (``"l1out"`` or ``"kfold"``).
    """

    def __init__(
        self,
        data_root: str,
        brainset: str,
        eval_every_n_epochs: int = 10,
        method: str = "l1out",
    ):
        super().__init__()
        self._data_root = data_root
        self._brainset = brainset
        self._eval_every_n_epochs = eval_every_n_epochs
        self._method = method
        self._labels_df: pd.DataFrame | None = None
        self._hash_sessions = brainset == "IBL_brainwide_2023"

    def _load_labels(self) -> pd.DataFrame:
        """Load and cache brain region labels indexed by composite key."""
        if self._labels_df is None:
            logger.info("Loading brain region labels for %s", self._brainset)
            self._labels_df = get_unit_number_to_label(
                self._data_root,
                self._brainset,
            )
        return self._labels_df

    def on_validation_epoch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        """Run probe evaluation after validation epoch (if due)."""
        current_epoch = trainer.current_epoch

        # Only run every N epochs (and always on epoch 0 for baseline)
        if current_epoch > 0 and current_epoch % self._eval_every_n_epochs != 0:
            return

        # The litmodule stores embeddings/unit_ids after compute() in
        # on_validation_epoch_end, which runs before this callback.
        val_embs = getattr(pl_module, "_val_probe_embeddings", None)
        val_unit_ids = getattr(pl_module, "_val_probe_unit_ids", None)
        val_unit_sessions = getattr(pl_module, "_val_unit_sessions", None)

        if val_embs is None or val_unit_ids is None:
            logger.warning("No validation embeddings available for probing")
            return

        if not val_unit_sessions:
            logger.warning("No session_id mapping available for probing")
            return

        try:
            labels_df = self._load_labels()
        except Exception:
            logger.exception("Failed to load brain region labels")
            return

        # Match embeddings to labels via composite key "session_id/unit_number"
        embs_np = val_embs.float().cpu().numpy()

        matched_embs = []
        matched_labels = []
        matched_sessions = []

        for i, uid_key in enumerate(val_unit_ids):
            uid_str = str(uid_key)

            if uid_str in labels_df.index:
                matched_embs.append(embs_np[i])
                matched_labels.append(labels_df.loc[uid_str, "brain_region"])
                matched_sessions.append(labels_df.loc[uid_str, "session_id"])

        if len(matched_embs) < 10:
            logger.warning(
                "Only %d units matched labels (need >= 10), skipping probe",
                len(matched_embs),
            )
            return

        matched_embs_arr = np.array(matched_embs)
        matched_labels_arr = np.array(matched_labels)
        matched_sessions_arr = np.array(matched_sessions)

        unique_labels = np.unique(matched_labels_arr)
        if len(unique_labels) < 2:
            logger.warning(
                "Only %d unique regions, skipping probe",
                len(unique_labels),
            )
            return

        try:
            results = run_probe(
                embeddings=matched_embs_arr,
                labels=matched_labels_arr,
                session_ids=matched_sessions_arr,
                method=self._method,
                hash_sessions=self._hash_sessions,
            )
        except Exception:
            logger.exception("Probe evaluation failed")
            return

        # Log results
        for metric, value in results.items():
            pl_module.log(
                f"val/probe_{metric}",
                value,
                prog_bar=(metric == "bacc"),
            )

        logger.info(
            "Epoch %d probe: bacc=%.4f f1_macro=%.4f (%d units, %d regions)",
            current_epoch,
            results["bacc"],
            results["f1_macro"],
            len(matched_embs_arr),
            len(unique_labels),
        )
