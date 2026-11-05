"""
Perceiver-style Neural Population Autoencoder.

Architecture:
  Encoder:
    1. Input proj + embedding bias → h: [B, N, C, L]
    2. K₁ neuron-space blocks: self-attn [B*L, N, C] + S4/DWConv [B*N, C, L]
       → neurons talk to each other, build population context
    3. Cross-attention [B*L, N, C] → [B*L, M, C]: compress to M latents
    4. K₂ latent-space blocks: self-attn [B*L, M, C] + S4/DWConv [B*M, C, L]
       → refine latent dynamics

  Decoder:
    1. Init per-neuron features from embeddings
    2. D decoder blocks: cross-attn [B*L, N, C] ← KV=[B*L, M, C] + DWConv + MLP
    3. Output proj → [B, N, L]

All cross/self-attention is per-timestep: [B*L, •, C].
All temporal mixing is per-token: [B*•, C, L].
No time-pooling anywhere.

Follows LDNS blocks.py conventions:
  [B, C, L] channel-first for temporal, FFTConv/DWConv, InstanceNorm, einops.
"""

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from einops.layers.torch import Rearrange

try:
    from torch.nn.attention.varlen import varlen_attn
except ImportError:
    varlen_attn = None

from ntt.contrib.tokenizer.ldns.s4 import FFTConv  # noqa: E402 — S4 kernel from LDNS

# ═══════════════════════════════════════════════
# Pack helpers — convert (B, N, …) padded to (N_total, …) packed
# ═══════════════════════════════════════════════


class PackMeta:
    """Cached packing metadata shared by all encoder / decoder blocks.

    Fields:
      neuron_counts:  (B,) int32 — real neuron count per batch element.
      B, L, N_total:  scalars.
      max_n:          int — max N_b across the batch.
      cu_seq_n:       (B*L + 1,) int32 — cu_seqlens for (l, b) sequences
                      of length N_b each.  Used for self-attn and the
                      KV side of cross-attn.
    """

    __slots__ = ("neuron_counts", "B", "L", "N_total", "max_n", "cu_seq_n")

    def __init__(self, neuron_counts: torch.Tensor, B: int, L: int):
        self.neuron_counts = neuron_counts
        self.B = B
        self.L = L
        self.N_total = int(neuron_counts.sum().item())
        self.max_n = int(neuron_counts.max().item())
        # cu_seqlens for attention: B*L sequences, order (l, b).
        # Sequence (l, b) has length neuron_counts[b] — same every l.
        device = neuron_counts.device
        cu = torch.zeros(L * B + 1, dtype=torch.int32, device=device)
        cu[1:] = neuron_counts.to(torch.int32).repeat(L).cumsum(0)
        self.cu_seq_n = cu


def pack_batch(
    x: torch.Tensor, e: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, PackMeta]:
    """Pack (B, N, L) spikes + (B, N, d_embed) embeddings by ``mask``.

    Returns:
      x_p:   (N_total, L)
      e_p:   (N_total, d_embed)
      meta:  PackMeta with cu_seqlens etc.
    """
    B, N, L = x.shape
    neuron_counts = mask.sum(dim=1).to(torch.int32)
    x_p = x[mask]  # (N_total, L) — batch-major flatten
    e_p = e[mask]  # (N_total, d_embed)
    return x_p, e_p, PackMeta(neuron_counts, B, L)


def unpack_to_padded(x_p: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Inverse of pack: (N_total, ...) → (B, N, ...) zero-padded."""
    out = x_p.new_zeros((*mask.shape, *x_p.shape[1:]))
    out[mask] = x_p
    return out


# ═══════════════════════════════════════════════
# Temporal mixing
# ═══════════════════════════════════════════════


class DWConvTimeMixer(nn.Module):
    """Depthwise conv for temporal mixing. Drop-in for FFTConv.
    Input/output: [B, C, L].

    ``causal=True`` makes output[..., t] depend only on input[..., <=t]: the
    forward conv is left-padded by ``kernel_size-1`` (instead of the centered
    ``kernel_size//2``) and the backward branch is disallowed. Requires
    ``bidirectional=False``.
    """

    def __init__(self, C, kernel_size=3, bidirectional=True, causal=False, activation=None):
        super().__init__()
        if causal and bidirectional:
            raise ValueError(
                "DWConvTimeMixer: causal=True is incompatible with bidirectional=True"
            )
        self.causal = causal
        # Causal: no internal padding; we left-pad manually in forward so the
        # kernel only reaches into the past. Non-causal: centered padding.
        fwd_padding = 0 if causal else kernel_size // 2
        self._causal_pad = kernel_size - 1 if causal else 0
        self.dw = nn.Conv1d(C, C, kernel_size, padding=fwd_padding, groups=C, bias=True)
        self.dw_bwd = (
            nn.Conv1d(C, C, kernel_size, padding=kernel_size // 2, groups=C, bias=True)
            if bidirectional
            else None
        )

    def forward(self, x):
        if self.causal:
            x = F.pad(x, (self._causal_pad, 0))
        y = self.dw(x)
        if self.dw_bwd is not None:
            # Flip kernel weights instead of input to avoid two full-tensor copies
            y = y + F.conv1d(
                x,
                self.dw_bwd.weight.flip(-1),
                self.dw_bwd.bias,
                padding=self.dw_bwd.padding[0],
                groups=self.dw_bwd.groups,
            )
        return y, None  # (output, state) to match FFTConv


class FFTConvFP32(nn.Module):
    """Wraps FFTConv to run in fp32 even under bf16 autocast (FFT needs fp32)."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = FFTConv(*args, **kwargs)

    @torch.autocast("cuda", enabled=False)
    def forward(self, x, *args, **kwargs):
        return self.conv(x.float(), *args, **kwargs)


def get_time_mixer(C, bidirectional=True, use_s4=True, kernel_size=3, causal=False, l_max=None):
    if use_s4:
        if causal:
            # The decoder (the only causal consumer) forces use_s4=False, so a
            # causal S4 kernel is never needed; flag it rather than silently
            # returning a bidirectional FFTConv.
            raise ValueError("get_time_mixer: causal=True is only supported for the DWConv path")
        # l_max pre-sizes the S4 kernel's cached length on the first forward so
        # variable-window training never re-triggers kernel growth (_setup_C).
        return FFTConvFP32(C, l_max=l_max, bidirectional=bidirectional, activation=None)
    return DWConvTimeMixer(C, kernel_size=kernel_size, bidirectional=bidirectional, causal=causal)


# ═══════════════════════════════════════════════
# Multi-head attention
# ═══════════════════════════════════════════════


class MultiHeadAttention(nn.Module):
    def __init__(self, C, num_heads=8, qkv_bias=True, attn_drop=0.0):
        super().__init__()
        assert C % num_heads == 0
        self.num_heads = num_heads
        self.q_proj = nn.Linear(C, C, bias=qkv_bias)
        self.k_proj = nn.Linear(C, C, bias=qkv_bias)
        self.v_proj = nn.Linear(C, C, bias=qkv_bias)
        self.out_proj = nn.Linear(C, C, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, q, k, v, mask=None):
        """q: [B, Nq, C], k/v: [B, Nk, C]. ``mask``: bool [B, Nk] or [Nq, Nk] or None."""
        h = self.num_heads
        q = rearrange(self.q_proj(q), "b n (h d) -> b h n d", h=h)
        k = rearrange(self.k_proj(k), "b n (h d) -> b h n d", h=h)
        v = rearrange(self.v_proj(v), "b n (h d) -> b h n d", h=h)
        # Skip mask when all-True (enables flash attn)
        attn_mask = None
        if mask is not None and not mask.all():
            if mask.dim() == 2 and mask.shape[0] == mask.shape[1]:
                # 2D attention mask (Nq, Nk) — e.g., block-causal
                attn_mask = mask[None, None, :, :].expand(q.shape[0], h, -1, -1)
            else:
                # 1D padding mask (B, Nk)
                attn_mask = mask[:, None, None, :].expand(-1, h, q.shape[2], -1)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        return self.out_proj(rearrange(out, "b h n d -> b n (h d)"))


# ═══════════════════════════════════════════════
# Packed attention helpers (varlen on already-packed tensors)
# ═══════════════════════════════════════════════


def _varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    max_q: int,
    max_k: int,
) -> torch.Tensor:
    """varlen_attn (CUDA) with automatic bf16 cast, or SDPA fallback on CPU.

    q/k/v: (T, H, D) packed along T; cu_q/cu_k describe sequences.
    """
    if q.is_cuda and varlen_attn is not None:
        orig_dtype = q.dtype
        needs_cast = orig_dtype not in (torch.float16, torch.bfloat16)
        if needs_cast:
            q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
        out = cast(torch.Tensor, varlen_attn(q, k, v, cu_q, cu_k, max_q, max_k))
        if needs_cast:
            out = out.to(orig_dtype)
        return out
    # CPU fallback: iterate over sequences, run SDPA per pair.  Slow but used
    # only for correctness tests.
    out = torch.zeros_like(q)
    n_seq = cu_q.numel() - 1
    for i in range(n_seq):
        qs, qe = int(cu_q[i]), int(cu_q[i + 1])
        ks, ke = int(cu_k[i]), int(cu_k[i + 1])
        if qe == qs or ke == ks:
            continue
        qi = q[qs:qe].transpose(0, 1).unsqueeze(0)  # (1, H, nq, D)
        ki = k[ks:ke].transpose(0, 1).unsqueeze(0)
        vi = v[ks:ke].transpose(0, 1).unsqueeze(0)
        oi = F.scaled_dot_product_attention(qi, ki, vi)
        out[qs:qe] = oi.squeeze(0).transpose(0, 1)
    return out


def packed_self_attn(
    attn: MultiHeadAttention,
    x: torch.Tensor,
    cu_seq: torch.Tensor,
    max_seq: int,
) -> torch.Tensor:
    """Self-attention on packed tokens.

    x:   (T, C) where T = total tokens across all sequences.
    Returns (T, C).
    """
    H = attn.num_heads
    D = x.shape[-1] // H
    q = attn.q_proj(x).view(-1, H, D)
    k = attn.k_proj(x).view(-1, H, D)
    v = attn.v_proj(x).view(-1, H, D)
    out = _varlen(q, k, v, cu_seq, cu_seq, max_seq, max_seq)
    return attn.out_proj(out.reshape(-1, H * D))


def packed_cross_attn(
    attn: MultiHeadAttention,
    q_in: torch.Tensor,
    kv_in: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    max_q: int,
    max_k: int,
) -> torch.Tensor:
    """Cross-attention on packed Q and KV.  q_in: (Tq, C), kv_in: (Tk, C)."""
    H = attn.num_heads
    D = q_in.shape[-1] // H
    q = attn.q_proj(q_in).view(-1, H, D)
    k = attn.k_proj(kv_in).view(-1, H, D)
    v = attn.v_proj(kv_in).view(-1, H, D)
    out = _varlen(q, k, v, cu_q, cu_k, max_q, max_k)
    return attn.out_proj(out.reshape(-1, H * D))


# ═══════════════════════════════════════════════
# Channel MLP
# ═══════════════════════════════════════════════


class ChannelMLP(nn.Module):
    def __init__(self, C, ratio=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(C, C * ratio, bias=False),
            nn.GELU(),
            nn.Linear(C * ratio, C, bias=False),
        )

    def forward(self, x):
        return self.net(x)


# ═══════════════════════════════════════════════
# Per-timestep cross-attention
# ═══════════════════════════════════════════════


class PerTimestepCrossAttention(nn.Module):
    """Cross-attention + residual FF on packed varlen Q and KV.

    Submodule names (norm_q, norm_kv, attn, norm_ff, ff) are part of the
    released state_dict layout (loaded with strict=True); do not rename.
    """

    def __init__(self, C, num_heads=8, mlp_ratio=2):
        super().__init__()
        self.norm_q = nn.LayerNorm(C)
        self.norm_kv = nn.LayerNorm(C)
        self.attn = MultiHeadAttention(C, num_heads=num_heads)
        self.norm_ff = nn.LayerNorm(C)
        self.ff = ChannelMLP(C, ratio=mlp_ratio)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        cu_q: torch.Tensor,
        cu_k: torch.Tensor,
        max_q: int,
        max_k: int,
    ) -> torch.Tensor:
        """q: (Tq, C), kv: (Tk, C). Returns (Tq, C)."""
        q_n = self.norm_q(q)
        kv_n = self.norm_kv(kv)
        out = packed_cross_attn(self.attn, q_n, kv_n, cu_q, cu_k, max_q, max_k)
        q = q + out
        q = q + self.ff(self.norm_ff(q))
        return q


# ═══════════════════════════════════════════════
# Temporal block (self-attn per timestep + S4/DWConv along time)
# ═══════════════════════════════════════════════


def _make_temporal_norm(C: int, use_layernorm: bool = False) -> nn.Module:
    """Build normalization for channel-first temporal features (B, C, L)."""
    if use_layernorm:
        return nn.Sequential(
            Rearrange("b c l -> b l c"),
            nn.LayerNorm(C),
            Rearrange("b l c -> b c l"),
        )
    return nn.InstanceNorm1d(C, affine=False)


# ═══════════════════════════════════════════════
# Packed encoder / decoder temporal blocks
# ═══════════════════════════════════════════════


class PackedTemporalBlock(nn.Module):
    """Encoder N-space block on packed layout [N_total, C, L].

    Used inside PerceiverNeuralAE.  Submodule names (norm_sa, self_attn,
    norm_tm, time_mixer, post_tm_scale, norm_cm, channel_mixer) are part of
    the released state_dict layout; do not rename.
    """

    def __init__(
        self,
        C,
        num_heads=8,
        mlp_ratio=2,
        use_s4=True,
        kernel_size=3,
        bidirectional=True,
        use_layernorm=False,
        grad_checkpointing=True,
        l_max=None,
    ):
        super().__init__()
        self.C = C
        self.grad_checkpointing = grad_checkpointing
        self.norm_sa = nn.LayerNorm(C)
        self.self_attn = MultiHeadAttention(C, num_heads=num_heads)

        self.norm_tm = _make_temporal_norm(C, use_layernorm)
        self.time_mixer = get_time_mixer(
            C, bidirectional=bidirectional, use_s4=use_s4, kernel_size=kernel_size, l_max=l_max
        )
        self.post_tm_scale = nn.Conv1d(C, C, 1, groups=C, bias=True)
        self.act_tm = nn.GELU()

        self.norm_cm = _make_temporal_norm(C, use_layernorm)
        self.channel_mixer = nn.Sequential(
            Rearrange("n c l -> n l c"),
            ChannelMLP(C, ratio=mlp_ratio),
            Rearrange("n l c -> n c l"),
        )

    def forward(self, h: torch.Tensor, meta: PackMeta) -> torch.Tensor:
        """h: [N_total, C, L]"""
        N_total, C, L = h.shape

        # 1. Self-attention: [N_total, C, L] → [L*N_total, C].  Tokens are
        #    ordered (l, n); each (l, b) chunk of N_b consecutive tokens
        #    forms one varlen sequence matching meta.cu_seq_n.
        h_lnc = h.permute(2, 0, 1).contiguous().view(L * N_total, C)
        h_lnc_n = self.norm_sa(h_lnc)
        attn_out = packed_self_attn(self.self_attn, h_lnc_n, meta.cu_seq_n, meta.max_n)
        h_lnc = h_lnc + attn_out
        h = h_lnc.view(L, N_total, C).permute(1, 2, 0).contiguous()

        # 2. Time mixer per neuron — fine-grained grad checkpoint on the
        #    FFT branch (largest fp32 activations).
        def _tm_branch(inp):
            y = self.norm_tm(inp)
            y, _ = self.time_mixer(y)
            return inp + self.post_tm_scale(self.act_tm(y))

        if self.grad_checkpointing and self.training and torch.is_grad_enabled():
            h = cast(
                torch.Tensor, torch.utils.checkpoint.checkpoint(_tm_branch, h, use_reentrant=False)
            )
        else:
            h = _tm_branch(h)

        # 3. Channel mixer
        y = self.norm_cm(h)
        h = h + self.channel_mixer(y)
        return h


class MSpaceBlock(nn.Module):
    """Encoder M-space block on dense layout [B, M, C, L].

    M is fixed and densely populated — no varlen, uses SDPA directly.
    """

    def __init__(
        self,
        C,
        num_heads=8,
        mlp_ratio=2,
        use_s4=True,
        kernel_size=3,
        bidirectional=True,
        use_layernorm=False,
        grad_checkpointing=True,
        l_max=None,
    ):
        super().__init__()
        self.C = C
        self.num_heads = num_heads
        self.head_dim = C // num_heads
        self.grad_checkpointing = grad_checkpointing
        self.norm_sa = nn.LayerNorm(C)
        self.self_attn = MultiHeadAttention(C, num_heads=num_heads)

        self.norm_tm = _make_temporal_norm(C, use_layernorm)
        self.time_mixer = get_time_mixer(
            C, bidirectional=bidirectional, use_s4=use_s4, kernel_size=kernel_size, l_max=l_max
        )
        self.post_tm_scale = nn.Conv1d(C, C, 1, groups=C, bias=True)
        self.act_tm = nn.GELU()

        self.norm_cm = _make_temporal_norm(C, use_layernorm)
        self.channel_mixer = nn.Sequential(
            Rearrange("b c l -> b l c"),
            ChannelMLP(C, ratio=mlp_ratio),
            Rearrange("b l c -> b c l"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, M, C, L]."""
        B, M, C, L = x.shape
        H = self.num_heads

        # 1. Self-attn per timestep via SDPA.
        x_sa = rearrange(x, "b m c l -> (b l) m c")
        x_sa_n = self.norm_sa(x_sa)
        q = rearrange(self.self_attn.q_proj(x_sa_n), "b m (h d) -> b h m d", h=H)
        k = rearrange(self.self_attn.k_proj(x_sa_n), "b m (h d) -> b h m d", h=H)
        v = rearrange(self.self_attn.v_proj(x_sa_n), "b m (h d) -> b h m d", h=H)
        o = F.scaled_dot_product_attention(q, k, v)
        o = self.self_attn.out_proj(rearrange(o, "b h m d -> b m (h d)"))
        x_sa = x_sa + o
        x = rearrange(x_sa, "(b l) m c -> b m c l", b=B, l=L)

        # 2. Time mixer
        x_flat = rearrange(x, "b m c l -> (b m) c l")

        def _tm_branch(inp):
            y = self.norm_tm(inp)
            y, _ = self.time_mixer(y)
            return inp + self.post_tm_scale(self.act_tm(y))

        if self.grad_checkpointing and self.training and torch.is_grad_enabled():
            x_flat = cast(
                torch.Tensor,
                torch.utils.checkpoint.checkpoint(_tm_branch, x_flat, use_reentrant=False),
            )
        else:
            x_flat = _tm_branch(x_flat)

        # 3. Channel mixer
        y = self.norm_cm(x_flat)
        x_flat = x_flat + self.channel_mixer(y)
        return rearrange(x_flat, "(b m) c l -> b m c l", b=B, m=M)


class PackedDecoderBlock(nn.Module):
    """Decoder block: per-timestep cross-attn Q=packed N, KV=dense M, then
    temporal, then channel mixer.  x: [N_total, C, L], z: [B, M, C, L]."""

    def __init__(
        self,
        C,
        num_heads=8,
        mlp_ratio=2,
        use_s4=True,
        kernel_size=3,
        bidirectional=True,
        use_layernorm=False,
        grad_checkpointing=True,
        causal=False,
    ):
        super().__init__()
        self.C = C
        self.grad_checkpointing = grad_checkpointing
        self.cross_attn = PerTimestepCrossAttention(C, num_heads=num_heads, mlp_ratio=mlp_ratio)

        self.norm_tm = _make_temporal_norm(C, use_layernorm)
        self.time_mixer = get_time_mixer(
            C,
            bidirectional=bidirectional,
            use_s4=use_s4,
            kernel_size=kernel_size,
            causal=causal,
        )
        self.post_tm_scale = nn.Conv1d(C, C, 1, groups=C, bias=True)
        self.act_tm = nn.GELU()

        self.norm_cm = _make_temporal_norm(C, use_layernorm)
        self.channel_mixer = nn.Sequential(
            Rearrange("n c l -> n l c"),
            ChannelMLP(C, ratio=mlp_ratio),
            Rearrange("n l c -> n c l"),
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor, meta: PackMeta) -> torch.Tensor:
        N_total, C, L = x.shape
        B, M = z.shape[0], z.shape[1]
        device = x.device

        # 1. Per-timestep cross-attn: Q=x packed (N), KV=z dense (M)
        q = x.permute(2, 0, 1).contiguous().view(L * N_total, C)  # (l, n) order
        kv = z.permute(3, 0, 1, 2).contiguous().view(L * B * M, C)  # (l, b, m) order
        cu_k = torch.arange(0, L * B * M + 1, M, dtype=torch.int32, device=device)
        q = self.cross_attn(q, kv, meta.cu_seq_n, cu_k, meta.max_n, M)
        x = q.view(L, N_total, C).permute(1, 2, 0).contiguous()

        # 2. Time mixer per neuron
        def _tm_branch(inp):
            y = self.norm_tm(inp)
            y, _ = self.time_mixer(y)
            return inp + self.post_tm_scale(self.act_tm(y))

        if self.grad_checkpointing and self.training and torch.is_grad_enabled():
            x = cast(
                torch.Tensor, torch.utils.checkpoint.checkpoint(_tm_branch, x, use_reentrant=False)
            )
        else:
            x = _tm_branch(x)

        # 3. Channel mixer
        y = self.norm_cm(x)
        x = x + self.channel_mixer(y)
        return x


# ═══════════════════════════════════════════════
# Embedding projection
# ═══════════════════════════════════════════════


class EmbeddingProjector(nn.Module):
    def __init__(self, d_embed, C):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_embed, C, bias=False),
            nn.GELU(),
            nn.Linear(C, C, bias=False),
        )

    def forward(self, e):
        return self.proj(e)


# ═══════════════════════════════════════════════
# Full Perceiver Autoencoder
# ═══════════════════════════════════════════════


class PerceiverNeuralAE(nn.Module):
    """
    Perceiver autoencoder for variable-N neural populations.

    Encoder processes in N-space first (neurons talk to each other),
    then compresses to M-space (latent dynamics).

    Encoder:
      h: [B, N, C, L]
        → K₁ temporal blocks in N-space (self-attn on N, temporal on L)
        → per-timestep cross-attn: [B*L, N, C] → [B*L, M, C]
        → K₂ temporal blocks in M-space (self-attn on M, temporal on L)
      = z: [B, M, C, L]

    Decoder:
      init from embeddings: [B, N, C, L]
        → D decoder blocks: cross-attn(Q=N, KV=M) + temporal + MLP
      = x̂: [B, N, L]
    """

    def __init__(
        self,
        C: int = 256,
        d_embed: int = 12,
        L: int = 300,
        M: int = 32,
        num_heads: int = 8,
        mlp_ratio: int = 2,
        num_encoder_N_blocks: int = 4,  # K₁: blocks in N-space
        num_encoder_M_blocks: int = 4,  # K₂: blocks in M-space
        num_decoder_blocks: int = 4,  # D: decoder blocks
        use_s4: bool = True,
        bidirectional: bool = True,
        kernel_size: int = 3,
        grad_checkpointing: bool = True,
        use_layernorm: bool = False,
        latent_dim: int | None = None,
        use_mask_embedding: bool = False,
        decoder_causal: bool = False,
    ):
        super().__init__()
        self.C = C
        self.M = M
        self.d_embed = d_embed
        self.latent_dim = latent_dim
        self.grad_checkpointing = grad_checkpointing

        # --- Input projection ---
        self.input_proj = nn.Conv1d(1, C, 1)
        self.embed_proj = EmbeddingProjector(d_embed, C)

        # --- Encoder: N-space blocks (packed [N_total, C, L]) ---
        self.encoder_N_blocks = nn.ModuleList(
            [
                PackedTemporalBlock(
                    C,
                    num_heads,
                    mlp_ratio,
                    use_s4=use_s4,
                    kernel_size=kernel_size,
                    bidirectional=bidirectional,
                    use_layernorm=use_layernorm,
                    grad_checkpointing=grad_checkpointing,
                    l_max=L,
                )
                for _ in range(num_encoder_N_blocks)
            ]
        )

        # --- Encoder: N → M cross-attention ---
        self.latent_queries = nn.Parameter(torch.randn(M, C) * 0.02)
        self.encoder_cross = PerTimestepCrossAttention(C, num_heads, mlp_ratio)

        # --- Encoder: M-space blocks (dense [B, M, C, L]) ---
        self.encoder_M_blocks = nn.ModuleList(
            [
                MSpaceBlock(
                    C,
                    num_heads,
                    mlp_ratio,
                    use_s4=use_s4,
                    kernel_size=kernel_size,
                    bidirectional=bidirectional,
                    use_layernorm=use_layernorm,
                    grad_checkpointing=grad_checkpointing,
                    l_max=L,
                )
                for _ in range(num_encoder_M_blocks)
            ]
        )

        # --- Decoder (packed [N_total, C, L]) ---
        # When decoder_causal, the decoder's temporal mixer only reaches into the
        # past (left-padded DWConv, no backward branch). Cross-attn is already
        # per-timestep (no cross-time leak), so this makes the whole decoder
        # causal — provided use_layernorm=True (InstanceNorm normalizes over L,
        # a global-over-time statistic that would itself leak the future).
        self.decoder_blocks = nn.ModuleList(
            [
                PackedDecoderBlock(
                    C,
                    num_heads,
                    mlp_ratio,
                    use_s4=False,
                    kernel_size=kernel_size,
                    bidirectional=False if decoder_causal else bidirectional,
                    use_layernorm=use_layernorm,
                    grad_checkpointing=grad_checkpointing,
                    causal=decoder_causal,
                )
                for _ in range(num_decoder_blocks)
            ]
        )

        # --- Latent bottleneck ---
        if latent_dim is not None:
            self.bottleneck = nn.Conv1d(C, latent_dim, 1)
            self.unbottleneck = nn.Conv1d(latent_dim, C, 1)
        else:
            self.bottleneck = None
            self.unbottleneck = None

        # --- Output ---
        self.output_proj = nn.Conv1d(C, 1, 1)

        # --- Temporal mask embedding ---
        if use_mask_embedding:
            self.mask_emb = nn.Parameter(torch.randn(C) * 0.02)

    def encode(self, x, e, mask=None, training_temporal_mask=None):
        """Packed-layout encoder.

        x: [B, N, L], e: [B, N, d_embed]
        mask: None or bool (B, N) — True for real neurons.  None treats every
            slot as real.
        training_temporal_mask: None or bool (B, N, L) — True=keep, False=masked.
        Returns: z [B, M, C_or_latent_dim, L]
        """
        B, N, L = x.shape
        device = x.device

        if mask is None:
            mask = torch.ones(B, N, dtype=torch.bool, device=device)

        # Pack (B, N, ...) → (N_total, ...) with batch-major order.
        x_p, e_p, meta = pack_batch(x, e, mask)

        # Input projection + embedding bias (packed).
        # x_p: (N_total, L) → (N_total, 1, L) → (N_total, C, L)
        h = self.input_proj(x_p.unsqueeze(1))
        e_proj = self.embed_proj(e_p)  # (N_total, C)
        h = h + e_proj.unsqueeze(-1)

        # Learnable mask-embedding at temporally-zeroed bins.
        if training_temporal_mask is not None and hasattr(self, "mask_emb"):
            tm_p = training_temporal_mask[mask]  # (N_total, L) bool
            h = h + self.mask_emb.view(1, -1, 1) * (~tm_p).unsqueeze(1).float()

        # K₁ blocks in N-space (packed)
        for block in self.encoder_N_blocks:
            h = block(h, meta)

        # Cross-attention N → M (per-timestep, packed varlen).
        # KV: packed N tokens in (l, n) order, shape (L*N_total, C).
        # Q:  latent_queries broadcast to every (l, b), order (l, b, m),
        #     shape (L*B*M, C); cu_q is uniform stride M.
        M = self.M
        N_total = meta.N_total
        kv = h.permute(2, 0, 1).contiguous().view(L * N_total, self.C)
        q = self.latent_queries.unsqueeze(0).expand(L * B, -1, -1).reshape(L * B * M, self.C)
        cu_q = torch.arange(0, L * B * M + 1, M, dtype=torch.int32, device=device)
        q = self.encoder_cross(q, kv, cu_q, meta.cu_seq_n, M, meta.max_n)
        # Unpack to [B, M, C, L] — q is in (l, b, m) order.
        z = q.view(L, B, M, self.C).permute(1, 2, 3, 0).contiguous()

        # K₂ blocks in M-space (dense)
        for block in self.encoder_M_blocks:
            z = block(z)

        # Bottleneck: project C → latent_dim
        if self.bottleneck is not None:
            z = rearrange(z, "b m c l -> (b m) c l")
            z = self.bottleneck(z)
            z = rearrange(z, "(b m) d l -> b m d l", b=B, m=self.M)

        return z

    def decode(self, z, e, mask=None):
        """Packed-layout decoder.

        z: [B, M, C_or_latent_dim, L], e: [B, N, d_embed]
        mask: None or bool (B, N).  None (the default) decodes every slot;
            callers that know the real-neuron mask can pass it to skip padded
            slots.
        Returns: x_hat [B, N, L]
        """
        B, M = z.shape[:2]
        L = z.shape[-1]
        N = e.shape[1]
        device = z.device

        if mask is None:
            mask = torch.ones(B, N, dtype=torch.bool, device=device)

        # Unbottleneck: project latent_dim → C
        if self.unbottleneck is not None:
            z = rearrange(z, "b m d l -> (b m) d l")
            z = self.unbottleneck(z)
            z = rearrange(z, "(b m) c l -> b m c l", b=B, m=M)

        # Pack embeddings for real (masked) neurons.
        neuron_counts = mask.sum(dim=1).to(torch.int32)
        e_p = e[mask]  # (N_total, d_embed)
        meta = PackMeta(neuron_counts, B, L)

        # Init per-neuron features from embeddings, broadcast over time
        e_proj = self.embed_proj(e_p)  # (N_total, C)
        x = e_proj.unsqueeze(-1).expand(-1, -1, L).contiguous()  # (N_total, C, L)

        # Decoder blocks (packed)
        for block in self.decoder_blocks:
            x = block(x, z, meta)

        # Output projection per neuron: (N_total, C, L) → (N_total, 1, L) → (N_total, L)
        x_hat_p = self.output_proj(x).squeeze(1)

        # Unpack back to padded (B, N, L)
        return unpack_to_padded(x_hat_p, mask)

    def forward(self, x, e, mask=None):
        z = self.encode(x, e, mask=mask)
        x_hat = self.decode(z, e, mask=mask)
        return x_hat, z


# ═══════════════════════════════════════════════
# Count wrapper (LDNS-compatible)
# ═══════════════════════════════════════════════


class PerceiverCountWrapper(nn.Module):
    def __init__(self, ae_net):
        super().__init__()
        self.ae_net = ae_net

    def forward(self, x, e, mask=None):
        logrates, z = self.ae_net(x, e, mask=mask)
        return F.softplus(logrates), z

    def encode(self, x, e, mask=None, training_temporal_mask=None):
        return self.ae_net.encode(x, e, mask=mask, training_temporal_mask=training_temporal_mask)

    def decode(self, z, e, mask=None):
        return F.softplus(self.ae_net.decode(z, e, mask=mask))


# ═══════════════════════════════════════════════
# Param counting
# ═══════════════════════════════════════════════


def count_params(model, verbose=False):
    total = sum(p.numel() for p in model.parameters())
    groups = {}
    for name, p in model.named_parameters():
        group = name.split(".")[0]
        groups[group] = groups.get(group, 0) + p.numel()
    if verbose:
        print(f"{'Module':<25s} {'Params':>12s} {'%':>6s}")
        print("-" * 45)
        for g, n in sorted(groups.items(), key=lambda x: -x[1]):
            print(f"  {g:<23s} {n:>12,d} {n / total * 100:>5.1f}%")
        print("-" * 45)
        print(f"  {'TOTAL':<23s} {total:>12,d}")
    return total, groups
