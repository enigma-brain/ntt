"""Perceiver-S4 Tokenizer for variable-N neural populations.

Wraps the PerceiverNeuralAE in the NTT tokenizer interface.  The key
difference from LDNS: this model takes per-neuron embeddings and supports
variable neuron counts via cross-attention compression (N → M latent tokens).

Embeddings are loaded from a pre-computed ``.pt`` file (e.g. NuCLR
embeddings) keyed by session_id.  At forward time, the tokenizer looks up
the embeddings for the current session and indexes by unit_ids.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal, NamedTuple, cast

import numpy as np
import torch
import torch.nn.functional as F
from einops import reduce
from pydantic import model_validator
from safetensors import safe_open
from safetensors.torch import save_file

from ntt.core.components.dist import Dist
from ntt.core.components.types import (
    BinnedSpikesInput,
    DataType,
    LatentType,
    ReconstructionOutput,
    TokenizerOutput,
)
from ntt.core.model.tokenizer.base import (
    BaseTokenizer,
    TokenizerSpec,
    register_tokenizer,
)
from ntt.core.model.tokenizer.validation import VALIDATION_SESSION_ID
from ntt.utils import NTTBaseModel

from ._sigreg import SlicingEppsPulleyRegularizer
from .blocks import PerceiverCountWrapper, PerceiverNeuralAE


class _MaskResult(NamedTuple):
    enc_spikes: torch.Tensor  # (B, Nv, L) — time-masked visible neurons
    enc_e: torch.Tensor  # (B, Nv, d_embed)
    enc_mask: torch.Tensor  # (B, Nv) bool
    loss_mask: torch.Tensor  # (B, N, L) float — where to compute loss
    targets: torch.Tensor  # (B, N, L) — original spikes
    enc_time_mask: torch.Tensor | None  # (B, Nv, L) bool — True=keep, False=masked


class MaskingConfig(NTTBaseModel, frozen=True):
    """Combined neuron + temporal masking configuration.

    ``neuron_ratio``: fraction of real neurons held out from encoder;
        loss computed on all time bins for these neurons.
    ``temporal_ratio``: fraction of time bins zeroed on visible (encoder)
        neurons; loss computed only at zeroed bins for these neurons.

    Example with N=8 neurons, L=12 time bins, neuron_ratio=0.5,
    temporal_ratio=0.5::

        Input spikes (N x L):
             t1 t2 t3 t4 t5 t6 t7 t8 t9 t10 t11 t12
        n1  [ 0  1  0  0  2  0  1  0  0  1   0   0 ]  visible
        n2  [ 1  0  0  1  0  0  0  1  0  0   0   1 ]  MASKED
        n3  [ 0  0  1  0  0  1  0  0  0  0   1   0 ]  visible
        n4  [ 0  0  0  0  1  0  0  0  1  0   0   0 ]  MASKED
        n5  [ 1  0  0  0  0  0  1  0  0  0   0   1 ]  visible
        n6  [ 0  0  0  1  0  0  0  0  0  1   0   0 ]  MASKED
        n7  [ 0  1  0  0  0  1  0  1  0  0   0   0 ]  visible
        n8  [ 0  0  0  0  0  0  0  0  0  0   0   0 ]  padded

        Encoder sees (visible neurons, independently time-masked):
        Each neuron gets its own random time mask (B, Nv, L).

        n1  [ 0  .  0  0  .  0  1  .  0  .   0   0 ]
        n3  [ 0  0  .  0  0  .  0  .  0  0   1   . ]
        n5  [ .  0  0  .  0  0  1  0  .  0   .   1 ]
        n7  [ 0  1  .  0  .  1  .  1  0  0   0   . ]
        (. = zeroed independently per neuron)

        Decode ALL neurons from latent z.

        Loss mask (where to compute Poisson NLL):
             t1 t2 t3 t4 t5 t6 t7 t8 t9 t10 t11 t12
        n1  [ .  X  .  .  X  .  .  X  .  X   .   . ]  visible: loss at zeroed T only
        n2  [ X  X  X  X  X  X  X  X  X  X   X   X ]  masked:  loss at ALL T
        n3  [ .  .  X  .  .  X  .  X  .  .   .   X ]  visible: loss at zeroed T only
        n4  [ X  X  X  X  X  X  X  X  X  X   X   X ]  masked:  loss at ALL T
        n5  [ X  .  .  X  .  .  .  .  X  .   X   . ]  visible: loss at zeroed T only
        n6  [ X  X  X  X  X  X  X  X  X  X   X   X ]  masked:  loss at ALL T
        n7  [ .  .  X  .  X  .  X  .  .  .   .   X ]  visible: loss at zeroed T only
        n8  [ .  .  .  .  .  .  .  .  .  .   .   . ]  padded:  no loss

    Degenerate cases:
    - ``neuron_ratio=0.5, temporal_ratio=0`` → neuron-only MAE (original behavior).
    - ``neuron_ratio=0, temporal_ratio=0.5`` → temporal masking only (coordinated dropout).
    - ``neuron_ratio=0.5, temporal_ratio=0.5`` → combined masking.
    """

    neuron_ratio: float = 0.5
    temporal_ratio: float = 0.0
    temporal_mode: Literal["independent", "shared"] = "independent"
    # When True, override the selective MAE loss_mask so that loss is computed on
    # ALL real neurons at all valid time bins, regardless of which were held out
    # of the encoder. Encoder still sees only the visible (1-neuron_ratio) subset.
    # Default False preserves the original MAE-style loss-on-masked-only behavior.
    loss_on_all_neurons: bool = False

    @property
    def active(self) -> bool:
        return self.neuron_ratio > 0 or self.temporal_ratio > 0

    def apply(
        self,
        spikes: torch.Tensor,
        e: torch.Tensor,
        mask: torch.Tensor,
    ) -> _MaskResult:
        """Apply combined masking and return encoder inputs + loss mask.

        Args:
            spikes: (B, N, L) spike counts.
            e: (B, N, d_embed) neuron embeddings.
            mask: (B, N) bool — True for real neurons.

        Returns:
            _MaskResult with encoder inputs (visible neurons, time-masked)
            and full-size loss_mask/targets for reconstruction loss.
        """
        B, N, L = spikes.shape
        device = spikes.device
        targets = spikes  # alias is safe: enc_spikes * time_mask creates a new tensor

        # --- Neuron split ---
        neuron_masked = torch.zeros(B, N, dtype=torch.bool, device=device)
        vis_indices: torch.Tensor | None = None
        if self.neuron_ratio > 0:
            enc_spikes, enc_e, enc_mask, neuron_masked, vis_indices = _split_neurons_for_encoder(
                spikes, e, mask, self.neuron_ratio
            )
        else:
            enc_spikes, enc_e, enc_mask = spikes, e, mask

        # --- Temporal masking ---
        # Generate ONE temporal mask at full (B, N, L) size, then slice for
        # encoder and reuse for loss so that the same time positions are
        # consistently masked everywhere (no double-counting).
        Nv = enc_spikes.shape[1]
        full_time_mask: torch.Tensor | None = None  # (B, 1, L) or (B, N, L)
        if self.temporal_ratio > 0:
            if self.temporal_mode == "shared":
                full_time_mask = torch.rand(B, 1, L, device=device) > self.temporal_ratio
            else:
                full_time_mask = torch.rand(B, N, L, device=device) > self.temporal_ratio

        # Apply temporal mask to encoder (visible neurons only)
        enc_time_mask: torch.Tensor | None = None
        if full_time_mask is not None:
            if full_time_mask.shape[1] == 1:
                # Shared: same mask for all neurons, broadcast works directly
                enc_time_mask = full_time_mask.expand(B, Nv, L)
            elif vis_indices is not None:
                # Independent: gather the time masks for visible neurons
                gather_idx = vis_indices.unsqueeze(-1).expand(-1, -1, L)  # (B, Nv, L)
                enc_time_mask = torch.gather(full_time_mask, 1, gather_idx)
                # Padding encoder slots: keep everything (True)
                enc_time_mask = enc_time_mask | ~enc_mask.unsqueeze(-1)
            else:
                # No neuron masking: enc order matches original
                enc_time_mask = full_time_mask
            enc_spikes = enc_spikes * enc_time_mask

        # --- Build loss_mask (B, N, L) ---
        # Loss is computed only at masked positions: held-out neurons at
        # non-temporally-masked time bins, and visible neurons at temporally-
        # masked time bins. This avoids double-counting temporal positions.
        loss_mask = torch.zeros(B, N, L, device=device)
        ftm_broad = (
            full_time_mask.expand(B, N, L)
            if full_time_mask is not None and full_time_mask.shape[1] == 1
            else full_time_mask
        )
        # Masked neurons: loss at time bins that are NOT temporally masked
        if ftm_broad is not None:
            loss_mask[neuron_masked] = ftm_broad[neuron_masked].float()
        else:
            loss_mask[neuron_masked] = 1.0
        # Visible real neurons: loss only at temporally zeroed time bins
        if enc_time_mask is not None:
            if vis_indices is not None:
                # Per-encoder-slot contribution: (~enc_time_mask) gated by enc_mask
                # so padded slots contribute exactly 0. Padded slots also share
                # vis_indices=0 (from the L266 zero-out in _split_neurons_for_encoder),
                # so they all aim at neuron 0 — we MUST use scatter_add_ (not
                # scatter_) so those zero contributions don't clobber any non-zero
                # value the neuron_masked branch above may have written at neuron 0.
                # Invariant: enc_mask[b, padded] is False ⇒ slot_contrib[b, padded, :] == 0.
                slot_contrib = (~enc_time_mask).float() * enc_mask.unsqueeze(-1).float()
                scatter_idx = vis_indices.unsqueeze(-1).expand(-1, -1, L)  # (B, Nv, L)
                loss_mask.scatter_add_(1, scatter_idx, slot_contrib)
            else:
                vis_real = mask
                loss_mask[vis_real] = (~enc_time_mask[enc_mask]).float()
        # Padded neurons: already 0

        if self.loss_on_all_neurons:
            # Override selective MAE loss: cover all real neurons at all time bins.
            # (B, N) bool real-neuron mask broadcast to (B, N, L) float.
            loss_mask = mask.float().unsqueeze(-1).expand(-1, -1, L).contiguous()

        return _MaskResult(
            enc_spikes=enc_spikes,
            enc_e=enc_e,
            enc_mask=enc_mask,
            loss_mask=loss_mask,
            targets=targets,
            enc_time_mask=enc_time_mask,
        )


def _split_neurons_for_encoder(
    spikes: torch.Tensor,
    e: torch.Tensor,
    mask: torch.Tensor,
    ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-sample split of real neurons into visible (encoder) and masked.

    Fully vectorized — no Python loop over the batch dimension.

    Returns:
        enc_spikes: (B, Nv_max, L) visible neuron spikes.
        enc_e: (B, Nv_max, d_embed) visible neuron embeddings.
        enc_mask: (B, Nv_max) bool.
        neuron_masked: (B, N) bool — True for masked neurons.
        vis_indices: (B, Nv_max) long — original neuron index for each
            encoder position (0 for padding slots).
    """
    B, N, L = spikes.shape
    d = e.shape[-1]
    device = spikes.device

    n_real = mask.sum(dim=1)
    # Require ≥1 real neuron per sample: with n_real=0 the (n_real-1).clamp(min=1)
    # trick below yields n_to_mask=1 → n_vis=-1, silently corrupting the gather.
    # Surface the precondition violation loudly rather than producing wrong shapes.
    if (n_real == 0).any():
        raise ValueError(
            f"_split_neurons_for_encoder: every batch element must have at least 1 real "
            f"neuron (mask.sum(dim=1) >= 1). Got {(n_real == 0).sum().item()} all-padded "
            f"sample(s) out of B={B}."
        )
    n_to_mask = (n_real.float() * ratio).int().clamp(min=1)
    n_to_mask = torch.min(n_to_mask, (n_real - 1).clamp(min=1))
    n_vis = n_real - n_to_mask
    n_vis_max = int(n_vis.max().item())

    # Random ordering of neurons: real neurons get finite scores, padding gets inf
    scores = torch.rand(B, N, device=device)
    scores.masked_fill_(~mask, float("inf"))
    sorted_indices = scores.argsort(dim=1)  # (B, N) — real neurons first, random order

    # Rank within the sorted order determines masked vs visible
    # sorted_indices[:, :n_to_mask[b]] = masked, [:, n_to_mask[b]:n_real[b]] = visible
    positions = torch.arange(N, device=device).unsqueeze(0)  # (1, N)
    n_to_mask_exp = n_to_mask.unsqueeze(1)  # (B, 1)
    is_masked_pos = positions < n_to_mask_exp  # (B, N)

    # neuron_masked: scatter the is_masked_pos flag back to original neuron indices
    neuron_masked = torch.zeros(B, N, dtype=torch.bool, device=device)
    neuron_masked.scatter_(1, sorted_indices, is_masked_pos)

    # Extract visible indices: positions [n_to_mask[b] .. n_to_mask[b]+n_vis[b])
    vis_positions = torch.arange(n_vis_max, device=device).unsqueeze(0)  # (1, Nv_max)
    gather_pos = (vis_positions + n_to_mask_exp).clamp(max=N - 1)  # (B, Nv_max)
    vis_indices = sorted_indices.gather(1, gather_pos)  # (B, Nv_max)

    # enc_mask: True for real visible positions
    enc_mask = vis_positions < n_vis.unsqueeze(1)  # (B, Nv_max)

    # Zero out padding slot indices (they'll gather from index 0 but get masked)
    vis_indices = vis_indices * enc_mask.long()

    # Gather spikes and embeddings for visible neurons
    enc_spikes = torch.gather(spikes, 1, vis_indices.unsqueeze(-1).expand(-1, -1, L))
    enc_e = torch.gather(e, 1, vis_indices.unsqueeze(-1).expand(-1, -1, d))
    enc_spikes[~enc_mask] = 0
    enc_e[~enc_mask] = 0

    return enc_spikes, enc_e, enc_mask, neuron_masked, vis_indices


class PerceiverTokenizerSpec(TokenizerSpec, frozen=True):
    """Specification for the Perceiver-S4 tokenizer.

    Attributes:
        KIND: Discriminator for registry.
        n_neurons: Max neuron count (for padding / data loading).
        window_length_seconds: Duration of input window.
        bin_size_ms: Temporal bin size in ms.

        hidden_dim: Channel dimension C in all blocks.
        latent_tokens: Number of latent tokens M (cross-attn bottleneck).
        d_embed: Dimension of per-neuron embeddings (must match .pt file).
        num_heads: Attention heads.
        mlp_ratio: MLP expansion ratio.
        num_encoder_N_blocks: K₁ — blocks in neuron-space (self-attn + S4).
        num_encoder_M_blocks: K₂ — blocks in latent-space.
        num_decoder_blocks: D — decoder blocks (cross-attn + temporal).
        use_s4: Use S4 (FFTConv) for temporal mixing; False = DWConv.
        bidirectional: Bidirectional temporal mixing.
        grad_checkpointing: Gradient checkpointing on blocks.

        embeddings_path: Path (or list of paths) to ``.pt`` (or ``.safetensors``,
            see ``save_embeddings_safetensors``) files containing
            pre-computed per-neuron embeddings.  Each file must have keys:
            ``embs`` (Tensor), ``unit_ids`` (ndarray of str),
            ``session_ids`` (ndarray of str).  Multiple files are merged.

        latent_beta: L2 regularization weight on latent.
        latent_td_beta: Temporal smoothness weight on latent.
        latent_td_k: Number of time lags for temporal smoothness.
        masking: Combined neuron + temporal masking config.
    """

    KIND: Literal["perceiver_s4"] = "perceiver_s4"

    # Data
    n_neurons: int | Literal[False] = False
    # A Dist (e.g. UniformDist) enables variable-window training; the data
    # pipeline's VariableWindowCrop randomly shortens each window, and the
    # length-agnostic architecture handles the varying bin count directly.
    window_length_seconds: float | Dist = 10.0  # 300 bins
    bin_size_ms: float = 33.33

    # Architecture
    hidden_dim: int = 128
    latent_tokens: int = 16
    d_embed: int = 128
    num_heads: int = 8
    mlp_ratio: int = 1
    num_encoder_N_blocks: int = 4
    num_encoder_M_blocks: int = 2
    num_decoder_blocks: int = 3
    use_s4: bool = True
    bidirectional: bool = True
    # When True, the decoder temporal mixer is causal (left-padded DWConv, no
    # backward branch) so reconstruction at time t depends only on the latent
    # up to t. Encoder stays bidirectional. Requires use_layernorm=True.
    decoder_causal: bool = False
    grad_checkpointing: bool = True
    use_layernorm: bool = False
    latent_dim: int | None = None
    binarize_spikes: bool = False  # Important, this works with poisson loss
    use_mask_embedding: bool = False

    # Embeddings — required: PerceiverTokenizer cannot operate without
    # pre-computed per-neuron embeddings keyed by session_id + unit_id.
    embeddings_path: str | list[str]

    # Regularization
    latent_beta: float = 1e-2
    latent_td_beta: float = 0.2
    latent_td_k: int = 5
    # 1 = first-difference TD over hops 1..K:
    #     z_t - z_{t-k}
    # 2 = curvature TD over hops 1..K:
    #     z_t - 2 z_{t-k} + z_{t-2k}
    latent_td_order: Literal[1, 2] = 1
    # When True, replace L2 latent reg with per-M SIGReg: reshape
    # z (B,M,C,L) -> (M, B*L, C), one Epps-Pulley statistic per latent
    # token, averaged over M. Weight is reused from `latent_beta`.
    use_sigreg_loss: bool = False
    latent_sigreg_num_slices: int = 256
    latent_sigreg_n_points: int = 17
    latent_sigreg_t_max: float = 3.0
    latent_sigreg_center: bool = True
    latent_sigreg_allgather: bool = False
    latent_sigreg_sync_slices: bool = False
    latent_sigreg_seed: int = 0
    masking: MaskingConfig = MaskingConfig()

    @model_validator(mode="before")
    @classmethod
    def _migrate_neuron_mask_ratio(cls, values: dict) -> dict:
        """Convert a spec dict that sets ``neuron_mask_ratio`` into ``masking``.

        Specs saved with a top-level ``neuron_mask_ratio`` load as
        ``masking={neuron_ratio: <value>, temporal_ratio: 0.0}``.
        """
        if isinstance(values, dict) and "neuron_mask_ratio" in values and "masking" not in values:
            values = dict(values)
            values["masking"] = {
                "neuron_ratio": values.pop("neuron_mask_ratio"),
                "temporal_ratio": 0.0,
            }
        return values

    # Capability metadata
    latent_type: LatentType = LatentType.CONTINUOUS
    compress_n: bool = True
    compress_t: bool = False
    variable_n: bool = True
    variable_t: bool = True
    variable_dt: bool = False
    input_type: DataType = DataType.SPIKES
    output_type: DataType = DataType.RATES

    has_time_dim: bool = True
    has_unit_dim: bool = True
    has_latent_dim: bool = True

    def get_vocab_size(self) -> int:
        raise NotImplementedError("Perceiver has continuous latent, no discrete vocabulary")


_EMBEDDINGS_FORMAT = "ntt-nuclr-embeddings/1"


def save_embeddings_safetensors(
    path: str | Path,
    embs: torch.Tensor,
    unit_ids: Sequence[str],
    session_ids: Sequence[str],
) -> Path:
    """Write a NuCLR embeddings table as a non-pickle ``.safetensors`` file.

    Tensors: ``embs`` (float32 [N, d]), ``session_index`` (int64 [N], index into the
    session list stored in the metadata) and ``unit_index`` (int64 [N], the
    session-local unit index, i.e. the last ``/``-separated part of the unit id).

    Raises:
        ValueError: If a unit id is not ``"<its session id>/<index>"``.
    """
    sessions = sorted(set(session_ids))
    session_to_index = {sess: i for i, sess in enumerate(sessions)}
    session_index = []
    unit_index = []
    for uid, sess in zip(unit_ids, session_ids, strict=True):
        prefix, _, local = str(uid).rpartition("/")
        if prefix != sess:
            raise ValueError(f"Unit id {uid!r} does not start with its session id {sess!r}.")
        session_index.append(session_to_index[sess])
        unit_index.append(int(local))
    path = Path(path)
    save_file(
        {
            "embs": embs.detach().cpu().float().contiguous(),
            "session_index": torch.tensor(session_index, dtype=torch.long),
            "unit_index": torch.tensor(unit_index, dtype=torch.long),
        },
        path,
        metadata={"format": _EMBEDDINGS_FORMAT, "sessions": json.dumps(sessions)},
    )
    return path


def _read_embeddings_file(path: str) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Read one embeddings file (``.pt`` or ``.safetensors``) → (embs, unit_ids, session_ids)."""
    if str(path).endswith(".safetensors"):
        with safe_open(path, framework="pt", device="cpu") as f:
            metadata = f.metadata() or {}
            if metadata.get("format") != _EMBEDDINGS_FORMAT:
                raise ValueError(
                    f"{path}: expected safetensors metadata format {_EMBEDDINGS_FORMAT!r}, "
                    f"got {metadata.get('format')!r}."
                )
            sessions = np.array(json.loads(metadata["sessions"]))
            embs = f.get_tensor("embs")
            session_ids = sessions[f.get_tensor("session_index").numpy()]
            unit_ids = np.array(
                [
                    f"{s}/{i}"
                    for s, i in zip(session_ids, f.get_tensor("unit_index").tolist(), strict=True)
                ]
            )
        return embs, unit_ids, session_ids
    data = torch.load(path, map_location="cpu", weights_only=False)
    embs = data["embs"]
    if not isinstance(embs, torch.Tensor):
        embs = torch.as_tensor(embs)
    return embs, np.asarray(data["unit_ids"]), np.asarray(data["session_ids"])


def _load_embeddings(path: str | list[str]) -> tuple[torch.Tensor, dict[str, tuple[int, int]]]:
    """Load pre-computed embeddings and build a session → offset index.

    Args:
        path: Single path or list of paths to ``.pt`` or ``.safetensors`` embedding files.

    Returns:
        all_embs: (total_units, d_embed) float32 tensor, sorted by session
            then by local unit index within each session.
        session_offset: dict mapping session_id → (start_row, n_units).
    """
    paths = [path] if isinstance(path, str) else path

    all_embs_list = []
    all_unit_ids_list = []
    all_session_ids_list = []

    for p in paths:
        embs, file_unit_ids, file_session_ids = _read_embeddings_file(p)
        all_embs_list.append(embs.float())
        all_unit_ids_list.append(file_unit_ids)
        all_session_ids_list.append(file_session_ids)

    embs = torch.cat(all_embs_list, dim=0)
    unit_ids: np.ndarray = np.concatenate(all_unit_ids_list)
    session_ids: np.ndarray = np.concatenate(all_session_ids_list)

    # Sort by (session_id, local_index) so each session is contiguous and ordered
    local_indices = np.array([int(uid.rsplit("/", 1)[-1]) for uid in unit_ids])
    order = np.lexsort((local_indices, session_ids))

    embs = embs[order]
    session_ids_sorted = session_ids[order]

    # Build offset table
    session_offset: dict[str, tuple[int, int]] = {}
    unique_sessions, counts = np.unique(session_ids_sorted, return_counts=True)
    offset = 0
    for sess, cnt in zip(unique_sessions, counts, strict=False):
        session_offset[str(sess)] = (offset, int(cnt))
        offset += int(cnt)

    return embs, session_offset


@register_tokenizer
class PerceiverTokenizer(BaseTokenizer):
    """Perceiver-S4 tokenizer with pre-computed neuron embeddings.

    Wraps PerceiverNeuralAE in the NTT BaseTokenizer interface.  Neuron
    embeddings are loaded from a ``.pt`` file at init and looked up by
    ``session_id`` + ``unit_ids`` from BinnedSpikesInput at forward time.
    """

    spec_class = PerceiverTokenizerSpec
    spec: PerceiverTokenizerSpec
    _all_embs: torch.Tensor
    _latent_sigreg_step: torch.Tensor

    def __init__(self, spec: PerceiverTokenizerSpec):
        super().__init__(spec)

        # The architecture is length-agnostic (per-timestep attention, conv time
        # mixers, kernel-1 projections), so n_bins is only a build-time hint.
        # window_length_bins raises on a Dist window; resolve via the effective
        # (max) window so variable-window specs construct cleanly.
        n_bins = round(spec.effective_window_length_seconds / spec.bin_size_seconds)

        # Core architecture
        ae = PerceiverNeuralAE(
            C=spec.hidden_dim,
            d_embed=spec.d_embed,
            L=n_bins,
            M=spec.latent_tokens,
            num_heads=spec.num_heads,
            mlp_ratio=spec.mlp_ratio,
            num_encoder_N_blocks=spec.num_encoder_N_blocks,
            num_encoder_M_blocks=spec.num_encoder_M_blocks,
            num_decoder_blocks=spec.num_decoder_blocks,
            use_s4=spec.use_s4,
            bidirectional=spec.bidirectional,
            grad_checkpointing=spec.grad_checkpointing,
            use_layernorm=spec.use_layernorm,
            latent_dim=spec.latent_dim,
            use_mask_embedding=spec.use_mask_embedding,
            decoder_causal=spec.decoder_causal,
        )
        self.count_wrapper = PerceiverCountWrapper(ae)

        # Load pre-computed embeddings (required). Pydantic guarantees
        # spec.embeddings_path is set; treat the empty-string case as an
        # explicit error rather than silently constructing a broken tokenizer.
        if not spec.embeddings_path:
            raise ValueError(
                "PerceiverTokenizerSpec.embeddings_path is required and must "
                "be a non-empty path (or list of paths) to a .pt file with "
                "pre-computed per-neuron embeddings."
            )
        all_embs, self._session_offset = _load_embeddings(spec.embeddings_path)
        self.register_buffer("_all_embs", all_embs, persistent=False)

        self.latent_sigreg = SlicingEppsPulleyRegularizer(
            num_slices=spec.latent_sigreg_num_slices,
            n_points=spec.latent_sigreg_n_points,
            t_max=spec.latent_sigreg_t_max,
        )
        self.register_buffer(
            "_latent_sigreg_step", torch.zeros((), dtype=torch.long), persistent=False
        )

    def _resolve_session_offset(self, sid: str) -> tuple[int, int]:
        """Resolve a session_id to (offset, count) in the embeddings table.

        Handles the ``brainset/session`` → ``session`` fallback used when
        embeddings files key on the bare session name.
        """
        if sid in self._session_offset:
            return self._session_offset[sid]
        stripped = sid.rsplit("/", 1)[-1]
        if stripped in self._session_offset:
            return self._session_offset[stripped]
        raise KeyError(
            f"Session {sid!r} not found in embeddings table. "
            f"Loaded sessions: {sorted(self._session_offset)[:5]}…"
        )

    @torch.compiler.disable
    def _get_embeddings(self, x: BinnedSpikesInput) -> tuple[torch.Tensor, torch.Tensor]:
        """Look up pre-computed embeddings by session_id + unit_ids.

        Returns:
            e: (B, N, d_embed) embeddings
            mask: (B, N) bool mask from unit_ids_pad_mask
        """
        session_ids = x.session_id
        if isinstance(session_ids, str):
            session_ids = (session_ids,)

        unit_ids = x.unit_ids  # (B, N) int tensor
        device = unit_ids.device

        if all(sid == VALIDATION_SESSION_ID for sid in session_ids):
            # Shape-only dummy batch (make_dummy_input: compression, spec validation). It
            # has no session in the table, and the latent shape doesn't depend on the
            # embedding values, so zeros stand in.
            e = self._all_embs.new_zeros((*unit_ids.shape, self._all_embs.shape[-1]))
            return e, x.unit_ids_pad_mask.bool().to(e.device)

        # Vectorized lookup: build a (B,) tensor of session offsets + counts,
        # convert per-sample unit_ids to global rows, then index the table in
        # one shot. Avoids the per-batch-element advanced-index loop.
        resolved = [self._resolve_session_offset(sid) for sid in session_ids]
        offsets = torch.tensor([r[0] for r in resolved], device=device, dtype=torch.long)
        counts = torch.tensor([r[1] for r in resolved], device=device, dtype=torch.long)
        # unit_ids are session-LOCAL (0..count-1). A real unit outside that
        # range means the caller used another numbering (e.g. dataset-global
        # ids); clamping it would silently reuse the session's last embedding
        # row, so fail instead. Padded slots may hold any id.
        real = x.unit_ids_pad_mask.bool().to(device)
        bad = real & ((unit_ids < 0) | (unit_ids >= counts.unsqueeze(1)))
        if bool(bad.any()):
            b = int(bad.any(dim=1).nonzero()[0])
            row = b if len(session_ids) > 1 else 0
            raise ValueError(
                f"Real unit id {int(unit_ids[b][bad[b]][0])} is outside session "
                f"{session_ids[row]!r}'s embedding range [0, {int(counts[row])}): "
                "PerceiverTokenizer expects session-local unit ids."
            )
        # Clamp per-row to that session's valid range; only padded slots can be
        # out of range here, and they are masked out by mask below.
        max_local = counts.unsqueeze(1).clamp_min(1) - 1
        local_ids = unit_ids.clamp_min(0).clamp_max_(max_local)
        global_ids = (local_ids + offsets.unsqueeze(1)).clamp_max_(
            self._all_embs.shape[0] - 1 if self._all_embs.shape[0] > 0 else 0
        )
        e = self._all_embs[global_ids]  # (B, N, d_embed)

        mask = x.unit_ids_pad_mask.bool()
        return e, mask.to(e.device)

    def _prepare_inputs(self, x: BinnedSpikesInput):
        """Extract spikes, embeddings, and mask from BinnedSpikesInput."""
        spikes = x.spike_matrix.float().transpose(-1, -2)  # (B, T, N) → (B, N, L)
        # torch.compiler.disable erases the method's signature for type checkers
        get_embeddings = cast(
            Callable[[BinnedSpikesInput], tuple[torch.Tensor, torch.Tensor]],
            self._get_embeddings,
        )
        e, mask = get_embeddings(x)
        return spikes, e, mask

    def encode(self, x: BinnedSpikesInput) -> torch.Tensor:
        """Encode spike data → latent tokens.

        Returns:
            z: (B, M, C, L) latent representation
        """
        spikes, e, mask = self._prepare_inputs(x)
        return self.count_wrapper.encode(spikes, e, mask=mask)

    def encode_with_timestamps(
        self,
        x: BinnedSpikesInput,
    ) -> tuple[torch.Tensor, dict]:
        """Encode + return canonical ``[B, T', U, D]`` latents with per-time-bin timestamps.

        Perceiver's raw ``encode()`` emits latents in tokenizer-native layout
        ``[B, M, D_lat, L]`` — latent-slot-major with the time axis trailing.
        Downstream consumers that route through the ``encode_with_timestamps``
        contract (POYO's ``FrozenNTTTokenizer`` wrapper, ``[B, T', U, D]`` latent
        metric pipelines) assume the canonical ordering with time in position 1
        and the fixed latent-embedding dim in position -1 — so if we passed the
        raw layout through, ``latents.shape[-1]`` would read out as ``L`` (the
        run-time bin count), not the invariant ``D_lat``, and any input-projection
        sized off the dummy-probe shape would mismatch actual encode output at a
        different window length.

        We permute here so anything routing through ``encode_with_timestamps``
        sees the canonical layout; raw ``encode()`` stays as-is for callers that
        already know the native shape (``reconstruct_from_input``,
        ``_compute_reconstruction_loss``, ``_compute_latent_loss``).

        Token timestamps span the ``L`` output time bins and are derived by the
        base helper from ``spec.token_bin_size_ms`` (which resolves to
        ``bin_size_ms`` at spec construction since ``compress_t=False``).
        """
        if not isinstance(x, BinnedSpikesInput):
            raise TypeError(
                f"PerceiverTokenizer.encode_with_timestamps requires BinnedSpikesInput, "
                f"got {type(x).__name__}"
            )
        assert x.absolute_start is not None, (
            "encode_with_timestamps requires absolute_start to be set on the input"
        )

        z = self.encode(x)  # (B, M, D_lat, L)
        latents = z.permute(0, 3, 1, 2).contiguous()  # (B, L, M, D_lat)
        metadata = self._compute_token_timestamps(x, latents.shape[1], provided=None)
        return latents, metadata

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Not supported — decoding requires per-neuron embeddings.

        Perceiver decoding needs both latent tokens ``z`` and per-neuron
        embeddings ``e``; ``e`` is looked up from session_id + unit_ids on
        the original input, so we cannot reconstruct without that input.
        Eval callers should use ``reconstruct_from_input(x)`` (which passes
        ``x`` through to the embedding lookup).
        """
        raise NotImplementedError(
            "PerceiverTokenizer.decode(tokens) is not supported because the "
            "decoder requires per-neuron embeddings derived from the input "
            "session_id and unit_ids. Use reconstruct_from_input(x) instead."
        )

    def forward(self, x: BinnedSpikesInput) -> TokenizerOutput:
        """Full forward pass with combined neuron + temporal masking.

        Training: mask neurons (held out from encoder) + zero random time bins
        on visible neurons.  Decode ALL neurons; loss via unified loss_mask.
        Eval: encode and decode all neurons (no masking).
        """
        spikes, e, mask = self._prepare_inputs(x)

        if self.training:
            if self.spec.masking.active:
                m = self.spec.masking.apply(spikes, e, mask)
                tm = m.enc_time_mask if self.spec.use_mask_embedding else None
                z = self.count_wrapper.encode(
                    m.enc_spikes, m.enc_e, mask=m.enc_mask, training_temporal_mask=tm
                )
                rates = self.count_wrapper.decode(z, e, mask=mask)  # ALL real neurons (B, N, L)
                targets = m.targets
                loss_mask = m.loss_mask
            else:
                # Pure autoencoding: no masking, loss on all real neurons at all valid bins.
                rates, z = self.count_wrapper(spikes, e, mask=mask)
                targets = spikes
                loss_mask = mask.float().unsqueeze(-1).expand_as(spikes)

            rates_T, recon_loss = self._compute_reconstruction_loss(
                rates=rates,
                targets=targets,
                loss_mask=loss_mask,
                pad_and_gap_mask=x.pad_and_gap_mask,
            )
        else:
            rates, z = self.count_wrapper(spikes, e, mask=mask)
            rates_T = rates.transpose(-1, -2)
            recon_loss = None

        time_mask = x.pad_and_gap_mask.any(dim=-1) if hasattr(x, "pad_and_gap_mask") else None
        aux_loss = self._compute_latent_loss(z, time_mask=time_mask)

        return TokenizerOutput(
            tokens=z,
            reconstruction=rates_T,
            aux_loss=aux_loss,
            precomputed_recon_loss=recon_loss,
        )

    def _compute_reconstruction_loss(
        self,
        rates: torch.Tensor,
        targets: torch.Tensor,
        loss_mask: torch.Tensor,
        pad_and_gap_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Poisson NLL over selected real-neuron positions and valid time bins."""
        rates_T = rates.transpose(-1, -2)  # (B, L, N)
        per_elem = F.poisson_nll_loss(
            rates_T,
            targets.transpose(-1, -2),
            log_input=False,
            full=False,
            reduction="none",
        )
        valid_mask = pad_and_gap_mask.to(device=per_elem.device, dtype=per_elem.dtype)
        lm = loss_mask.transpose(-1, -2).to(dtype=per_elem.dtype) * valid_mask
        recon_loss = (per_elem * lm).sum() / lm.sum().clamp(min=1)
        return rates_T, recon_loss

    def _compute_latent_loss(
        self, z: torch.Tensor, time_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Latent regularization: (L2 or per-M SIGReg) + temporal smoothness.

        ``use_sigreg_loss=True`` swaps the L2 shrinkage term for per-M SIGReg
        on z reshaped to ``(M, B*L, C)``; the same ``latent_beta`` weight is
        applied to whichever is active. When ``time_mask`` is provided, gap/pad
        time bins are excluded from the SIGReg empirical estimate.

        Args:
            z: (B, M, C, L) latent tensor
            time_mask: (B, L) bool mask, True for valid time bins.
        """
        if self.spec.use_sigreg_loss:
            B, M, C, L = z.shape
            zr = z.permute(1, 0, 3, 2).reshape(M, B * L, C)
            if time_mask is not None:
                mask_ml = time_mask.bool().unsqueeze(0).expand(M, -1, -1).reshape(M, B * L)
            else:
                mask_ml = None

            sigreg_allgather = self.spec.latent_sigreg_allgather
            if sigreg_allgather:
                if M != self.spec.latent_tokens:
                    raise RuntimeError(
                        "latent_sigreg_allgather currently requires the full latent prefix. "
                        "Disable it for prefix training or make all ranks use "
                        "the same prefix."
                    )
                zr, mask_ml = self._gather_latent_sigreg_samples(zr, mask_ml)

            if self.spec.latent_sigreg_center:
                if mask_ml is None or not sigreg_allgather:
                    zr = zr - zr.mean(dim=1, keepdim=True)
                else:
                    w = mask_ml.to(dtype=zr.dtype).unsqueeze(-1)
                    denom = w.sum(dim=1, keepdim=True).clamp(min=1)
                    zr = zr - (zr * w).sum(dim=1, keepdim=True) / denom

            sigreg_seed = self._next_latent_sigreg_seed()
            shrink_loss = self.latent_sigreg(zr, mask=mask_ml, seed=sigreg_seed)
        else:
            shrink_loss = reduce(z.pow(2), "B M C L -> B C L", "sum").mean()

        td_order = self.spec.latent_td_order
        td_loss = z.new_zeros(())
        for k in range(1, self.spec.latent_td_k + 1):
            if td_order == 1 and z.shape[-1] > k:
                diff = (z[..., :-k] - z[..., k:]).pow(2)
                td_loss = td_loss + reduce(diff, "B M C L -> B C L", "mean").mean() / (1 + k)
            elif td_order == 2 and z.shape[-1] > 2 * k:
                curv = z[..., 2 * k :] - 2 * z[..., k:-k] + z[..., : -2 * k]
                if k > 1:
                    curv = curv / (k * k)
                td_loss = td_loss + reduce(curv.pow(2), "B M C L -> B C L", "mean").mean()

        return self.spec.latent_beta * shrink_loss + self.spec.latent_td_beta * td_loss

    def _next_latent_sigreg_seed(self) -> int | None:
        """Return a rank-synchronized SIGReg slice seed when requested.

        The step counter buffer is created in __init__; we increment it here
        and add to the spec's base seed so each training step uses a fresh
        but deterministic slice set across DDP ranks.
        """
        if not self.spec.latent_sigreg_sync_slices:
            return None
        step = int(self._latent_sigreg_step.item())
        self._latent_sigreg_step.add_(1)
        return int(self.spec.latent_sigreg_seed) + step

    def _gather_latent_sigreg_samples(
        self, zr: torch.Tensor, mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Differentiably gather latent SIGReg samples across DDP ranks.

        Args:
            zr: ``(M, S, C)`` latent samples, where ``S=B*L``.
            mask: optional ``(M, S)`` valid-sample mask.

        Returns:
            Gathered tensors with samples concatenated along ``S``.
        """
        import torch.distributed as dist

        if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
            return zr, mask

        world_size = dist.get_world_size()
        device = zr.device
        local_shape = torch.tensor([zr.size(0), zr.size(1), zr.size(2)], device=device)
        all_shapes = [torch.zeros(3, device=device, dtype=torch.long) for _ in range(world_size)]
        dist.all_gather(all_shapes, local_shape)

        m_values = {int(s[0].item()) for s in all_shapes}
        c_values = {int(s[2].item()) for s in all_shapes}
        if len(m_values) != 1 or len(c_values) != 1:
            raise RuntimeError(
                "latent_sigreg_allgather requires matching latent token and channel dimensions "
                f"across ranks; got shapes {[tuple(int(v.item()) for v in s) for s in all_shapes]}"
            )

        if mask is None:
            mask = torch.ones(zr.shape[:2], device=device, dtype=torch.bool)

        max_s = max(int(s[1].item()) for s in all_shapes)
        pad_s = max_s - zr.size(1)
        if pad_s > 0:
            zr = F.pad(zr, (0, 0, 0, pad_s, 0, 0))
            mask = F.pad(mask.to(dtype=torch.uint8), (0, pad_s, 0, 0)).bool()

        zr = zr.contiguous()
        mask = mask.contiguous()

        from torch.distributed.nn.functional import all_gather as diff_all_gather

        gathered_zr = diff_all_gather(zr)
        gathered_mask = [torch.zeros_like(mask) for _ in range(world_size)]
        dist.all_gather(gathered_mask, mask)
        return torch.cat(gathered_zr, dim=1), torch.cat(gathered_mask, dim=1)

    def reconstruct(self, tokens: torch.Tensor) -> ReconstructionOutput:
        """Not supported — see decode() for the rationale.

        Use ``reconstruct_from_input(x)`` instead.
        """
        raise NotImplementedError(
            "PerceiverTokenizer.reconstruct(tokens) is not supported because "
            "the decoder requires per-neuron embeddings derived from the "
            "input session_id and unit_ids. Use reconstruct_from_input(x)."
        )

    def reconstruct_from_input(
        self,
        x: BinnedSpikesInput,
        return_latents: bool = False,
        return_latent_timestamps: bool = True,
    ) -> ReconstructionOutput | dict:
        """Encode + decode with embeddings looked up from the .pt file.

        Signature matches ``BaseTokenizer.reconstruct_from_input`` so callers
        that pass ``return_latent_timestamps`` (the default eval path) work.
        ``return_latent_timestamps`` only takes effect when ``return_latents=True``.
        """
        spikes, e, mask = self._prepare_inputs(x)
        z = self.count_wrapper.encode(spikes, e, mask=mask)
        rates = self.count_wrapper.decode(z, e)  # (B, N, L)
        reconstruction = rates.transpose(-1, -2)  # (B, T, N)
        result = ReconstructionOutput(
            reconstruction=reconstruction,
            data_type=self.spec.output_type,
            bin_size_seconds=self.spec.bin_size_ms / 1000.0,
        )
        if return_latents:
            # z is (B, M, D, L) -> (B, L, M, D) so eval sees (T, M, D) per sample.
            out: dict = {
                "reconstruction": reconstruction,
                "reconstruction_data_type": result.data_type,
                "latents": z.permute(0, 3, 1, 2),
            }
            if return_latent_timestamps:
                T = reconstruction.shape[1]
                bin_size_sec = self.spec.bin_size_ms / 1000.0
                abs_start = x.absolute_start  # (B,)
                indices = torch.arange(T, device=abs_start.device)
                out["token_timestamps"] = (
                    abs_start[:, None] + (indices[None, :] + 0.5) * bin_size_sec
                )
            return out
        return result
