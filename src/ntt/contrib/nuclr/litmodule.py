"""Standalone LightningModule for NuCLR contrastive training.

Pure contrastive learning — no decoder/reconstruction. Produces per-neuron
identity embeddings and trains via sample-wise contrastive loss.

NOT a tokenizer — this is a standalone model with its own training loop.

Implements the NuCLR method; adapted from the reference implementation, used
under the Apache-2.0 license: https://github.com/nerdslab/nuclr (Arora et al.,
2025, NeurIPS; https://arxiv.org/abs/2512.01199).
"""

import lightning as L
import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor

from ntt.core.components.types import RatesInput, SpikeTimesInput
from ntt.core.model.components.lr_scheduler import LRSchedulerConfig, build_lr_scheduler

from .adapter import ntt_rates_to_nuclr_format, ntt_to_nuclr_format
from .contrastive_loss import SampleWiseContrastiveLoss
from .data.two_view_types import TwoViewBatch
from .embedding_collector import EmbeddingCollector
from .encoder import NuclrEncoder
from .encoder_binned import NuclrBinnedEncoder


class NuclrContrastiveLitModule(L.LightningModule):
    """Standalone LightningModule for NuCLR contrastive learning.

    Trains either a spike-based perceiver encoder or a binned encoder to produce
    per-neuron identity embeddings using sample-wise contrastive loss.

    When ``bin_size`` is None, uses the spike-based encoder (cross-attention to
    raw spike times). When ``bin_size`` is set, uses the binned encoder (bins
    spike counts via linear read-in, self-attention only).

    Args:
        ctx_duration: Context window duration in seconds.
        latent_step: Spacing between latent positions in seconds.
        bin_size: Bin size in seconds (None = spike-based encoder).
        dim: Model dimension.
        cross_heads: Cross-attention heads (spike encoder only).
        self_heads: Self-attention heads.
        dim_head: Dimension per attention head.
        atn_dropout: Attention dropout probability.
        lin_dropout: Linear/FFN dropout probability.
        t_layers: Number of temporal attention layers.
        st_layers: Number of spatio-temporal attention layers.
        rot_ratio: Fraction of dim_head for rotary encoding.
        tau: Contrastive loss temperature.
        dcl: Use decoupled contrastive learning.
        full_denom: Full denominator (cross-view + within-view negatives).
        projector: Use learnable projector head.
        lr: Learning rate.
        weight_decay: Weight decay for non-norm/bias parameters.
        max_epochs: Total training epochs (for LR schedule).
    """

    def __init__(
        self,
        # Encoder params
        ctx_duration: float = 10.0,
        latent_step: float = 1.0,
        bin_size: float | None = None,
        dim: int = 256,
        cross_heads: int = 4,
        self_heads: int = 4,
        dim_head: int = 64,
        atn_dropout: float = 0.0,
        lin_dropout: float = 0.2,
        t_layers: int = 2,
        st_layers: int = 2,
        rot_ratio: float = 0.5,
        # Loss params
        tau: float = 0.2,
        dcl: bool = True,
        full_denom: bool = True,
        projector: bool = True,
        # Optimizer params
        lr: float = 0.0005,
        weight_decay: float = 0.01,
        max_epochs: int = 300,
    ):
        super().__init__()
        self.save_hyperparameters()

        self._use_binned = bin_size is not None

        # Store for deferred init
        if self._use_binned:
            self._encoder_kwargs = {
                "ctx_duration": ctx_duration,
                "latent_step": latent_step,
                "bin_size": bin_size,
                "dim": dim,
                "self_heads": self_heads,
                "dim_head": dim_head,
                "atn_dropout": atn_dropout,
                "lin_dropout": lin_dropout,
                "t_layers": t_layers,
                "st_layers": st_layers,
                "rot_ratio": rot_ratio,
            }
        else:
            self._encoder_kwargs = {
                "ctx_duration": ctx_duration,
                "latent_step": latent_step,
                "dim": dim,
                "cross_heads": cross_heads,
                "self_heads": self_heads,
                "dim_head": dim_head,
                "atn_dropout": atn_dropout,
                "lin_dropout": lin_dropout,
                "t_layers": t_layers,
                "st_layers": st_layers,
                "rot_ratio": rot_ratio,
            }
        self._loss_kwargs = {
            "dim": dim,
            "tau": tau,
            "dcl": dcl,
            "full_denom": full_denom,
            "projector": projector,
        }
        self._lr = lr
        self._weight_decay = weight_decay
        self._max_epochs = max_epochs

        self.encoder: NuclrEncoder | NuclrBinnedEncoder | None = None
        self.contrastive_loss: SampleWiseContrastiveLoss | None = None
        self._val_collector: EmbeddingCollector | None = None
        # Session tracking for probe callbacks
        self._val_unit_sessions: dict[str, str] = {}
        # Stored after compute() for callbacks to read
        self._val_probe_embeddings: Tensor | None = None
        self._val_probe_unit_ids: np.ndarray | None = None

    def configure_model(self):
        """Create encoder and loss (deferred for FSDP compatibility)."""
        if self.encoder is not None:
            return
        if self._use_binned:
            self.encoder = NuclrBinnedEncoder(**self._encoder_kwargs)
        else:
            self.encoder = NuclrEncoder(**self._encoder_kwargs)
        self.contrastive_loss = SampleWiseContrastiveLoss(**self._loss_kwargs)

    def _encode_view(self, view: SpikeTimesInput | RatesInput):
        """Encode a view through the appropriate encoder (spike or binned)."""
        assert self.encoder is not None
        if self._use_binned:
            assert isinstance(self.encoder, NuclrBinnedEncoder)
            assert isinstance(view, RatesInput)
            inp = ntt_rates_to_nuclr_format(
                view,
                num_latents=self.encoder.num_latents,
                bins_per_latent=self.encoder.bins_per_latent,
            )
            return self.encoder(inp.bins, inp.unit_seqlen), inp
        else:
            assert isinstance(view, SpikeTimesInput)
            inp = ntt_to_nuclr_format(view)
            return self.encoder(inp.spike_timestamps, inp.spike_seqlen, inp.unit_seqlen), inp

    def training_step(self, batch: TwoViewBatch, batch_idx: int) -> Tensor:
        assert self.encoder is not None
        assert self.contrastive_loss is not None

        view1, view2, match_idx = batch

        y1, input1 = self._encode_view(view1)
        y2, input2 = self._encode_view(view2)

        loss = self.contrastive_loss(
            emb1=y1,
            emb2=y2,
            unit_seqlen_v1=input1.unit_seqlen,
            unit_seqlen_v2=input2.unit_seqlen,
            match_idx=match_idx,
        )

        self.log("train/loss", loss, prog_bar=True, batch_size=len(match_idx))
        return loss

    def validation_step(self, batch: TwoViewBatch, batch_idx: int) -> None:
        assert self.encoder is not None
        assert self.contrastive_loss is not None

        view1, view2, match_idx = batch

        y1, input1 = self._encode_view(view1)
        y2, input2 = self._encode_view(view2)

        loss = self.contrastive_loss(
            emb1=y1,
            emb2=y2,
            unit_seqlen_v1=input1.unit_seqlen,
            unit_seqlen_v2=input2.unit_seqlen,
            match_idx=match_idx,
        )

        self.log("val/loss", loss, prog_bar=True, sync_dist=True, batch_size=len(match_idx))

        # Collect embeddings for RankMe + probing (view1 only)
        if self._val_collector is None:
            self._val_collector = EmbeddingCollector(device=y1.device)

        # Build composite unit IDs (session_stem/unit_number) for global
        # uniqueness across sessions, and track session_id per unit.
        # torch_brain prepends "brainset/" to session_id — strip it so
        # composite keys match labels.py indexing (H5 file stems).
        session_ids = view1.session_id
        if isinstance(session_ids, str):
            session_ids = (session_ids,)

        raw_unit_ids = input1.unit_ids.cpu().numpy()
        composite_ids = []
        offset = 0
        for sid, n_units in zip(session_ids, input1.unit_seqlen.tolist(), strict=False):
            sid_clean = str(sid).rsplit("/", 1)[-1] if "/" in str(sid) else str(sid)
            for j in range(n_units):
                uid = raw_unit_ids[offset + j]
                uid_key = uid.item() if hasattr(uid, "item") else uid
                composite = f"{sid_clean}/{uid_key}"
                composite_ids.append(composite)
                self._val_unit_sessions[composite] = sid_clean
            offset += n_units
        composite_ids_arr = np.array(composite_ids)

        self._val_collector.update(y1, composite_ids_arr)

    def on_validation_epoch_end(self) -> None:
        """Compute RankMe from collected embeddings and store for callbacks."""
        if self._val_collector is None:
            return

        try:
            embeddings, unit_ids = self._val_collector.compute()
        except RuntimeError:
            self._val_probe_embeddings = None
            self._val_probe_unit_ids = None
            self._val_unit_sessions = {}
            return

        # Store for probe callbacks (before any further processing).
        # NOTE: these remain this rank's shard; the (default-off) probe callback
        # would need the same pooling as RankMe below to be correct under DDP.
        self._val_probe_embeddings = embeddings
        self._val_probe_unit_ids = unit_ids

        # RankMe is a nonlinear function of the singular values of the whole
        # embedding matrix, so pool the per-rank shards and compute it once on
        # the full matrix instead of sync_dist-averaging per-rank values (the
        # mean of per-rank RankMe != RankMe of the pooled matrix). The pooled
        # result is identical on every rank, so log it rank-zero-only (no sync).
        pooled = self._gather_embeddings(embeddings)
        rankme = self._compute_rankme(pooled.float())
        self.log("val/rankme", rankme, prog_bar=True, rank_zero_only=True)

    @staticmethod
    def _compute_rankme(embeddings: Tensor) -> float:
        """Compute RankMe (effective rank via entropy of singular values).

        RankMe measures how informative the embedding space is.
        Higher = more dimensions are being used = better.

        Args:
            embeddings: ``(N, dim)`` tensor.

        Returns:
            RankMe score (scalar).
        """
        # SVD of the embedding matrix
        s = torch.linalg.svdvals(embeddings)
        # Normalize to get probability distribution
        p = s / s.sum()
        # Shannon entropy → effective rank
        p = p[p > 0]
        entropy = -(p * p.log()).sum()
        return entropy.exp().item()

    @staticmethod
    def _gather_embeddings(embeddings: Tensor) -> Tensor:
        """Pool per-rank embedding shards into the full matrix for RankMe.

        RankMe is nonlinear in the singular values, so it must be computed on
        the pooled embeddings rather than averaged across ranks. On a single
        process (or with no process group) this is a no-op. Shards have
        different row counts per rank, so use the variable-size object gather
        (on CPU) rather than ``all_gather``, which requires equal shapes.
        """
        if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
            return embeddings
        gathered: list[Tensor | None] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, embeddings.detach().cpu())
        return torch.cat([g for g in gathered if g is not None], dim=0)

    def configure_optimizers(self):
        """AdamW with weight decay groups + linear warmup + cosine decay."""
        assert self.encoder is not None
        assert self.contrastive_loss is not None

        # Split params: no weight decay for norm/bias
        decay_params = []
        no_decay_params = []
        for name, param in list(self.encoder.named_parameters()) + list(
            self.contrastive_loss.named_parameters()
        ):
            if not param.requires_grad:
                continue
            if "norm" in name or "bias" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self._weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(param_groups, lr=self._lr)

        # LR schedule: linear warmup 1 epoch + cosine decay
        total_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = total_steps // self._max_epochs  # 1 epoch

        config = LRSchedulerConfig(
            warmup_steps=warmup_steps,
            decay_ratio=None,  # decay starts right after warmup
            decay_type="cosine",
            min_lr_factor=0.0,
            interval="step",
        )
        scheduler = build_lr_scheduler(optimizer, config, total_steps)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
