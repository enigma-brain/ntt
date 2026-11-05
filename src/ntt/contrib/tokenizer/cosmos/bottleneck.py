"""Latent bottleneck modules for the Cosmos 1D tokenizer.

All bottlenecks share the ``encode`` / ``decode`` pair used by the tokenizer:

    bottleneck.encode(h, neuron_mask=...) -> (z, aux_loss)
    bottleneck.decode(z)                  -> z_for_decoder

For discrete bottlenecks (FSQ today, others as follow-ups), two extra methods
let the tokenizer surface integer tokens and reverse the lookup:

    bottleneck.get_indices(h)        -> indices ``[B, T_latent]``
    bottleneck.indices_to_codes(idx) -> codes   ``[B, D, T_latent]``

``encoder_out_channels`` tells the encoder how many channels to emit (e.g. FSQ
projects into the codebook-dim; a plug-in VAE would want ``2 * z_dim``).

Additional backends (``vector-quantize-pytorch`` from lucidrains, VQ, ResFSQ,
LFQ, …) can be added here behind the same API.

Adapted from NVIDIA Cosmos-Tokenizer, used under the Apache-2.0 license:
https://github.com/NVIDIA/Cosmos-Tokenizer
(see ``LICENSES/cosmos-tokenizer-Apache-2.0.txt``).
"""

from __future__ import annotations

from typing import ClassVar, Literal

import torch
from torch import Tensor, nn


def _round_ste(z: Tensor) -> Tensor:
    """Round with straight-through gradient estimator."""
    return z + (z.round() - z).detach()


class FSQ1d(nn.Module):
    """Finite Scalar Quantization over 1D latents ``[B, D, T]``.

    Adapted from Cosmos's ``FSQuantizer`` / lucidrains's implementation
    (Mentzer et al. 2023). One codebook, one scalar per feature dim.

    Args:
        levels: Per-dimension quantization levels. ``len(levels)`` fixes the
            codebook dim ``D``; codebook size is the product of levels.
    """

    def __init__(self, levels: list[int]) -> None:
        super().__init__()
        if not levels:
            raise ValueError("FSQ1d: levels must be non-empty")
        if any(lvl < 2 for lvl in levels):
            raise ValueError(f"FSQ1d: every level must be >= 2, got {levels}")

        self.codebook_dim: int = len(levels)
        codebook_size = 1
        for lvl in levels:
            codebook_size *= lvl
        self.codebook_size: int = codebook_size

        self.register_buffer(
            "_levels",
            torch.tensor(levels, dtype=torch.int64),
            persistent=True,
        )
        self._levels: Tensor
        basis = torch.ones(len(levels), dtype=torch.int64)
        for i in range(1, len(levels)):
            basis[i] = basis[i - 1] * levels[i - 1]
        self.register_buffer("_basis", basis, persistent=True)
        self._basis: Tensor

    def _bound(self, z: Tensor, eps: float = 1e-3) -> Tensor:
        """Squash each dim into ``[-L/2, L/2]`` via shifted tanh."""
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def _quantize_normalized(self, z: Tensor) -> Tensor:
        """Round to the nearest lattice cell; output in ``[-1, 1]``."""
        quantized = _round_ste(self._bound(z))
        half_width = self._levels // 2
        return quantized / half_width

    def _codes_to_indices_last(self, codes: Tensor) -> Tensor:
        """Map codes with feature axis last ``(..., D)`` to integer indices ``(...)``."""
        half_width = self._levels // 2
        lattice = (codes * half_width) + half_width
        return (lattice.long() * self._basis).sum(dim=-1)

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """Quantize ``[B, D, *spatial]`` and return ``(codes, indices)``.

        ``codes``   — ``[B, D, *spatial]`` continuous quantized values (STE).
        ``indices`` — ``[B, *spatial]`` integer indices.
        """
        if z.ndim < 3:
            raise ValueError(f"FSQ1d expects [B, D, *spatial], got shape {tuple(z.shape)}")
        if z.shape[1] != self.codebook_dim:
            raise ValueError(f"FSQ1d: expected D={self.codebook_dim}, got D={z.shape[1]}")
        # Move feature axis to the end: [B, D, *S] -> [B, *S, D]
        perm = (0, *range(2, z.ndim), 1)
        z_last = z.permute(*perm).contiguous()
        codes_last = self._quantize_normalized(z_last)
        indices = self._codes_to_indices_last(codes_last)  # [B, *S]
        # Restore: [B, *S, D] -> [B, D, *S]
        inv_perm = (0, z.ndim - 1, *range(1, z.ndim - 1))
        codes = codes_last.permute(*inv_perm).contiguous()
        return codes, indices

    def indices_to_codes(self, indices: Tensor) -> Tensor:
        """Inverse lookup: ``[B, *spatial]`` -> codes ``[B, D, *spatial]``.

        Also accepts a trailing singleton ``[B, *spatial, 1]``.
        """
        if indices.ndim >= 3 and indices.shape[-1] == 1:
            indices = indices.squeeze(-1)
        if indices.ndim < 2:
            raise ValueError(f"indices_to_codes expects [B, *spatial], got {tuple(indices.shape)}")
        lattice = (indices.unsqueeze(-1) // self._basis) % self._levels
        half_width = self._levels // 2
        codes_last = (lattice.float() - half_width) / half_width  # [B, *S, D]
        # [B, *S, D] -> [B, D, *S]
        n = codes_last.ndim
        inv_perm = (0, n - 1, *range(1, n - 1))
        return codes_last.permute(*inv_perm).contiguous()


class AEBottleneck(nn.Module):
    """Continuous pass-through bottleneck (AE / no VAE).

    A pair of 1x1 Conv1d projections mirrors Cosmos's ``quant_conv`` /
    ``post_quant_conv``: they let the encoder output ``z_channels`` features
    while the latent lives in ``latent_channels`` features.
    """

    def __init__(self, z_channels: int, latent_channels: int) -> None:
        super().__init__()
        self.z_channels = z_channels
        self.latent_channels = latent_channels
        self.pre_conv = nn.Conv1d(z_channels, latent_channels, kernel_size=1)
        self.post_conv = nn.Conv1d(latent_channels, z_channels, kernel_size=1)

    @property
    def encoder_out_channels(self) -> int:
        return self.z_channels

    def encode(self, h: Tensor) -> tuple[Tensor, Tensor | None]:
        return self.pre_conv(h), None

    def decode(self, z: Tensor) -> Tensor:
        return self.post_conv(z)


class FSQBottleneck1d(nn.Module):
    """FSQ bottleneck over ``[B, z_channels, T_latent]`` latents.

    Projects encoder output down to the FSQ codebook-dim, quantizes per
    time step, and projects back up to ``z_channels`` for the decoder.
    """

    def __init__(self, z_channels: int, levels: list[int]) -> None:
        super().__init__()
        self.z_channels = z_channels
        self.quantizer = FSQ1d(levels=levels)
        self.pre_conv = nn.Conv1d(z_channels, self.quantizer.codebook_dim, kernel_size=1)
        self.post_conv = nn.Conv1d(self.quantizer.codebook_dim, z_channels, kernel_size=1)

    @property
    def encoder_out_channels(self) -> int:
        return self.z_channels

    @property
    def codebook_size(self) -> int:
        return self.quantizer.codebook_size

    def encode(self, h: Tensor) -> tuple[Tensor, Tensor | None]:
        h_proj = self.pre_conv(h)
        codes, _ = self.quantizer(h_proj)
        return codes, None  # FSQ has no auxiliary loss

    def decode(self, z: Tensor) -> Tensor:
        return self.post_conv(z)

    def get_indices(self, h: Tensor) -> Tensor:
        """Return discrete indices ``[B, T_latent]`` for encoder output ``h``."""
        h_proj = self.pre_conv(h)
        _, indices = self.quantizer(h_proj)
        return indices

    def indices_to_codes(self, indices: Tensor) -> Tensor:
        """Map ``[B, T_latent]`` indices -> ``[B, z_channels, T_latent]`` codes.

        Applies both the FSQ inverse lookup and ``post_conv`` so the returned
        tensor is directly usable by the decoder.
        """
        codes = self.quantizer.indices_to_codes(indices)
        return self.post_conv(codes)


class AEBottleneck2d(nn.Module):
    """2D equivalent of ``AEBottleneck`` — continuous pass-through with 1x1 convs."""

    def __init__(self, z_channels: int, latent_channels: int) -> None:
        super().__init__()
        self.z_channels = z_channels
        self.latent_channels = latent_channels
        self.pre_conv = nn.Conv2d(z_channels, latent_channels, kernel_size=1)
        self.post_conv = nn.Conv2d(latent_channels, z_channels, kernel_size=1)

    @property
    def encoder_out_channels(self) -> int:
        return self.z_channels

    def encode(self, h: Tensor) -> tuple[Tensor, Tensor | None]:
        return self.pre_conv(h), None

    def decode(self, z: Tensor) -> Tensor:
        return self.post_conv(z)


class FSQBottleneck2d(nn.Module):
    """FSQ bottleneck over ``[B, z_channels, H, W]`` latents.

    Same structure as ``FSQBottleneck1d`` but with Conv2d projections. The
    underlying ``FSQ1d`` quantizer is dimension-agnostic (operates on the
    feature axis, any trailing spatial dims).
    """

    def __init__(self, z_channels: int, levels: list[int]) -> None:
        super().__init__()
        self.z_channels = z_channels
        self.quantizer = FSQ1d(levels=levels)
        self.pre_conv = nn.Conv2d(z_channels, self.quantizer.codebook_dim, kernel_size=1)
        self.post_conv = nn.Conv2d(self.quantizer.codebook_dim, z_channels, kernel_size=1)

    @property
    def encoder_out_channels(self) -> int:
        return self.z_channels

    @property
    def codebook_size(self) -> int:
        return self.quantizer.codebook_size

    def encode(self, h: Tensor) -> tuple[Tensor, Tensor | None]:
        h_proj = self.pre_conv(h)
        codes, _ = self.quantizer(h_proj)
        return codes, None

    def decode(self, z: Tensor) -> Tensor:
        return self.post_conv(z)

    def get_indices(self, h: Tensor) -> Tensor:
        """Return discrete indices ``[B, H, W]`` for encoder output ``h``."""
        h_proj = self.pre_conv(h)
        _, indices = self.quantizer(h_proj)
        return indices

    def indices_to_codes(self, indices: Tensor) -> Tensor:
        """Map indices back to ``[B, z_channels, H, W]`` (includes ``post_conv``)."""
        codes = self.quantizer.indices_to_codes(indices)
        return self.post_conv(codes)


class VAEBottleneck2d(nn.Module):
    """Variational AE bottleneck for 2D tokenizers.

    Encoder produces ``z_channels``; pre_conv projects to ``2 * latent_channels``
    (mu and logvar). Sample with reparameterization during training, take the
    mode at eval. KL against standard normal is returned as auxiliary loss.
    """

    def __init__(
        self,
        z_channels: int,
        latent_channels: int,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.z_channels = z_channels
        self.latent_channels = latent_channels
        self.beta = float(beta)
        self.pre_conv = nn.Conv2d(z_channels, latent_channels * 2, kernel_size=1)
        self.post_conv = nn.Conv2d(latent_channels, z_channels, kernel_size=1)

    @property
    def encoder_out_channels(self) -> int:
        return self.z_channels

    def encode(self, h: Tensor) -> tuple[Tensor, Tensor]:
        params = self.pre_conv(h)
        mu, logvar = params.chunk(2, dim=1)
        logvar = logvar.clamp(min=-30.0, max=20.0)
        if self.training:
            z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        else:
            z = mu
        # KL(N(mu, sigma) || N(0, 1)) = 0.5 * sum(mu^2 + sigma^2 - 1 - log sigma^2)
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).mean()
        return z, self.beta * kl

    def decode(self, z: Tensor) -> Tensor:
        return self.post_conv(z)


class VQPTBottleneck2d(nn.Module):
    """Bottleneck adapter wrapping any ``vector_quantize_pytorch`` quantizer.

    Operates per spatial position on a ``[B, z_channels, H, W]`` feature map by
    flattening (H, W) into a single sequence. The wrapped quantizer's own
    project_in / project_out handle z_channels ↔ codebook_dim mismatches when
    relevant, so we just keep ``z_channels`` external and let the underlying
    module deal internally. Returns the quantized codes (shape preserved) plus
    a scalar aux loss (commitment / entropy / etc.).
    """

    _REGISTRY: ClassVar[dict[str, tuple[str, str]]] = {
        "vq": ("VectorQuantize", "channel_last"),
        "residual_vq": ("ResidualVQ", "channel_last"),
        "rvq": ("ResidualVQ", "channel_last"),  # alias
        "grouped_residual_vq": ("GroupedResidualVQ", "channel_last"),
        "fsq": ("FSQ", "channel_last"),
        "residual_fsq": ("ResidualFSQ", "channel_last"),
        "lfq": ("LFQ", "channel_last"),
        "residual_lfq": ("ResidualLFQ", "channel_last"),
        "sim_vq": ("SimVQ", "channel_last"),
        "latent_quantize": ("LatentQuantize", "channel_first"),
    }

    def __init__(
        self,
        z_channels: int,
        vq_type: str = "rvq",
        loss_weight: float = 1.0,
        vq_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        if vq_type not in self._REGISTRY:
            raise ValueError(
                f"unknown vq_type {vq_type!r}; must be one of {sorted(self._REGISTRY)}"
            )
        import vector_quantize_pytorch as vqpt_mod

        class_name, layout = self._REGISTRY[vq_type]
        cls = getattr(vqpt_mod, class_name)
        kwargs: dict = dict(vq_kwargs or {})
        if "dim" in kwargs:
            raise ValueError("pass z_channels, not dim — VQPTBottleneck2d sets dim itself")
        kwargs["dim"] = z_channels
        self.z_channels = z_channels
        self.vq_type = vq_type
        self.loss_weight = float(loss_weight)
        self._layout = layout
        self.quantizer: nn.Module = cls(**kwargs)

    @property
    def encoder_out_channels(self) -> int:
        return self.z_channels

    def _to_seq(self, h: Tensor) -> tuple[Tensor, int, int]:
        b, c, hh, ww = h.shape
        x = h.permute(0, 2, 3, 1).reshape(b, hh * ww, c)  # [B, H*W, C]
        if self._layout == "channel_first":
            x = x.permute(0, 2, 1)  # [B, C, H*W]
        return x, hh, ww

    def _from_seq(self, x: Tensor, h: int, w: int) -> Tensor:
        if self._layout == "channel_first":
            x = x.permute(0, 2, 1)
        b, _, c = x.shape
        return x.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def _unpack(out) -> tuple[Tensor, Tensor, Tensor | None]:
        if not isinstance(out, tuple):
            out = tuple(out)
        if len(out) == 2:
            return out[0], out[1], None
        if len(out) >= 3:
            quantized, indices, aux = out[0], out[1], out[2]
            if isinstance(aux, Tensor) and aux.ndim > 0:
                aux = aux.sum()
            return quantized, indices, aux
        raise RuntimeError(f"unexpected vqpt output of length {len(out)}")

    def encode(self, h: Tensor) -> tuple[Tensor, Tensor | None]:
        x, hh, ww = self._to_seq(h)
        quantized, _indices, aux = self._unpack(self.quantizer(x))
        z = self._from_seq(quantized, hh, ww)
        if aux is not None:
            aux = self.loss_weight * aux
        return z, aux

    def decode(self, z: Tensor) -> Tensor:
        # VQPT outputs are already in the codebook subspace; encoder/decoder
        # convs handle any further mixing. Pass-through.
        return z

    def get_indices(self, h: Tensor) -> Tensor:
        x, hh, ww = self._to_seq(h)
        _quantized, indices, _aux = self._unpack(self.quantizer(x))
        # indices: [B, H*W] for single-codebook, [B, H*W, Q] for residual variants
        if indices.ndim == 2:
            return indices.reshape(-1, hh, ww)
        return indices.reshape(-1, hh, ww, *indices.shape[2:])

    def indices_to_codes(self, indices: Tensor) -> Tensor:
        """Inverse lookup: spatial indices → quantized codes ``[B, C, H, W]``.

        Uses the wrapped quantizer's own ``get_codes_from_indices`` (or
        equivalent) when available; falls back to ``get_output_from_indices``.
        """
        # Flatten spatial back to sequence for the quantizer's lookup API
        if indices.ndim == 3:  # [B, H, W]
            B, H, W = indices.shape
            seq = indices.reshape(B, H * W)
        elif indices.ndim == 4:  # [B, H, W, Q] — residual
            B, H, W, Q = indices.shape
            seq = indices.reshape(B, H * W, Q)
        else:
            raise ValueError(f"indices_to_codes expects 3D or 4D, got {indices.shape}")

        # vector_quantize_pytorch quantizers expose either get_output_from_indices
        # or get_codes_from_indices; try in that order.
        get_output = getattr(self.quantizer, "get_output_from_indices", None)
        get_codes = getattr(self.quantizer, "get_codes_from_indices", None)
        if get_output is not None:
            codes = get_output(seq)
        elif get_codes is not None:
            codes = get_codes(seq)
        else:
            raise NotImplementedError(
                f"{type(self.quantizer).__name__} has no indices_to_codes API"
            )
        # codes: [B, H*W, C] (channel_last) or [B, C, H*W] (channel_first)
        return self._from_seq(codes, H, W)


BottleneckKind = Literal["ae", "fsq", "vae", "vqpt"]


def build_bottleneck(
    kind: BottleneckKind,
    *,
    z_channels: int,
    latent_channels: int | None = None,
    fsq_levels: list[int] | None = None,
    vae_beta: float = 1.0,
    vqpt_type: str = "rvq",
    vqpt_kwargs: dict | None = None,
    vqpt_loss_weight: float = 1.0,
    ndim: int = 1,
):
    """Construct a bottleneck by short name.

    Args:
        kind: ``"ae"`` (continuous pass-through), ``"fsq"`` (FSQ lattice
            quantizer), ``"vae"`` (variational AE), ``"vqpt"`` (any quantizer
            from ``vector_quantize_pytorch``; pick variant via ``vqpt_type``).
        z_channels: Encoder's output channel count.
        latent_channels: AE / VAE only — continuous latent channel count.
        fsq_levels: FSQ only — per-dimension quantization levels.
        vae_beta: VAE only — KL weight.
        vqpt_type: VQPT only — registry key (e.g. ``"rvq"``, ``"vq"``, ``"lfq"``).
        vqpt_kwargs: VQPT only — forwarded to the wrapped quantizer constructor;
            ``dim`` is set automatically from ``z_channels``.
        vqpt_loss_weight: VQPT only — multiplier on the quantizer's aux loss.
        ndim: ``1`` for Conv1d projections, ``2`` for Conv2d projections.
    """
    if ndim not in (1, 2):
        raise ValueError(f"ndim must be 1 or 2, got {ndim}")
    if kind == "ae":
        if latent_channels is None:
            raise ValueError("build_bottleneck(kind='ae') requires latent_channels")
        if ndim == 1:
            return AEBottleneck(z_channels=z_channels, latent_channels=latent_channels)
        return AEBottleneck2d(z_channels=z_channels, latent_channels=latent_channels)
    if kind == "fsq":
        if not fsq_levels:
            raise ValueError("build_bottleneck(kind='fsq') requires fsq_levels")
        if ndim == 1:
            return FSQBottleneck1d(z_channels=z_channels, levels=fsq_levels)
        return FSQBottleneck2d(z_channels=z_channels, levels=fsq_levels)
    if kind == "vae":
        if ndim != 2:
            raise NotImplementedError("VAE bottleneck currently 2D-only")
        if latent_channels is None:
            raise ValueError("build_bottleneck(kind='vae') requires latent_channels")
        return VAEBottleneck2d(
            z_channels=z_channels, latent_channels=latent_channels, beta=vae_beta
        )
    if kind == "vqpt":
        if ndim != 2:
            raise NotImplementedError("VQPT bottleneck currently 2D-only")
        return VQPTBottleneck2d(
            z_channels=z_channels,
            vq_type=vqpt_type,
            loss_weight=vqpt_loss_weight,
            vq_kwargs=vqpt_kwargs,
        )
    raise ValueError(f"Unknown bottleneck kind '{kind}'. Supported: 'ae', 'fsq', 'vae', 'vqpt'")
