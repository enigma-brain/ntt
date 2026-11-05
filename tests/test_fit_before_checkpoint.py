"""Fit-once tokenizers must be fitted in the checkpoint a 1-epoch run saves.

Fit-once tokenizers (PCA baseline, frequency LUT) are fitted before the checkpoint is
written. Lightning runs callback hooks *before* the LightningModule's hook, so fitting in
``on_train_epoch_end`` would let the default checkpoint callback (``every_n_train_steps`` +
``save_on_train_epoch_end``) write ``last.ckpt`` first: either by the step trigger on the last
batch (the epoch-end save is then skipped as a same-step duplicate) or by the epoch-end save
itself. With ``max_epochs=1`` (the pca_baseline experiment configs) the fitted model would exist
only in memory, and every saved checkpoint, and the MLflow copy, would be unfitted.
"""

from __future__ import annotations

from functools import partial

import lightning.pytorch as pl
import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

from ntt.contrib.loss.poisson_nll import PoissonNLLLossSpec
from ntt.contrib.tokenizer.pca_baseline import PCABaselineTokenizerSpec
from ntt.core.components.types import RatesInput
from ntt.core.model.litmodule import NTTLitModule

N, T, B = 4, 20, 2


def _batch(seed: int) -> RatesInput:
    g = torch.Generator().manual_seed(seed)
    return RatesInput(
        rate_matrix=torch.poisson(torch.full((B, T, N), 0.5), generator=g),
        pad_and_gap_mask=torch.ones(B, T, N, dtype=torch.long),
        sampling_rate=1000.0,
        unit_ids=torch.arange(N).unsqueeze(0).expand(B, -1),
        unit_ids_pad_mask=torch.ones(B, N, dtype=torch.long),
        duration_sec=torch.full((B,), T / 1000.0),
        session_id="test",
        subject_id="subj",
        species="mouse",
    )


@pytest.mark.parametrize(
    "n_batches",
    [6, 5],  # epoch length a multiple of every_n_train_steps (step save on the last batch) / not
)
def test_last_checkpoint_of_one_epoch_run_is_fitted(tmp_path, n_batches):
    """
    - Setup: a tiny PCA tokenizer (4 neurons, 20 time bins, 2 components) is trained for one
      epoch on 5 or 6 batches of random spike counts, with checkpoints saved every 3 steps and
      at the end of the epoch, like the default config.
    - The check: the test then loads last.ckpt from disk and asserts that its saved flag
      tokenizer.fitted is True and that its saved counter tokenizer._n_seen equals
      2 × 4 × number of batches. That's every row from every batch: 2 samples per batch ×
      4 neurons.
    """
    module = NTTLitModule(
        tokenizer_spec=PCABaselineTokenizerSpec(
            n_neurons=N, window_length_seconds=T / 1000.0, bin_size_ms=1.0, n_components=2
        ),
        loss_spec=PoissonNLLLossSpec(),
        optimizer_gen=partial(torch.optim.SGD, lr=0.0),
    )
    loader = DataLoader([_batch(i) for i in range(n_batches)], batch_size=None)
    # mirrors config/trainer/callbacks/default.yaml (every_n_train_steps scaled down)
    ckpt = ModelCheckpoint(
        dirpath=tmp_path,
        monitor="train/loss",
        mode="min",
        save_last=True,
        save_top_k=2,
        every_n_train_steps=3,
        save_on_train_epoch_end=True,
    )
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="cpu",
        logger=False,
        callbacks=[ckpt],
        enable_progress_bar=False,
        enable_model_summary=False,
        limit_val_batches=0,
        num_sanity_val_steps=0,
    )
    trainer.fit(module, train_dataloaders=loader)

    assert bool(module.tokenizer.fitted), "the in-memory model should be fitted after the epoch"
    ckpt_file = tmp_path / "last.ckpt"
    state = torch.load(ckpt_file, map_location="cpu", weights_only=False)["state_dict"]
    assert bool(state["tokenizer.fitted"]), (
        "last.ckpt was saved before the epoch-end fit: the checkpoint of a 1-epoch run is unfitted"
    )
    assert int(state["tokenizer._n_seen"]) == B * N * n_batches
