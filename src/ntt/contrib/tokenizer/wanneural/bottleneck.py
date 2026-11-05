"""Latent bottleneck modules for WanNeural tokenizer.

Built-in bottleneck types:
- VAEBottleneck: Continuous latent with reparameterized Gaussian (KL divergence loss)
- MMDVAEBottleneck: Continuous latent with InfoVAE / MMD-VAE regularizer on
  the aggregate posterior (replaces per-sample KL with an MMD-to-prior term)
- FSQBottleneck: Discrete latent using Finite Scalar Quantization (no auxiliary loss)
- BSQBottleneck: Discrete latent using Binary Spherical Quantization (entropy aux loss)
- DiffusionPriorBottleneck: Continuous latent regularized by a small diffusion model
  trained jointly on the latent (Unified Latents — How to Train Your Latents). The
  encoder is deterministic; a fixed sigma_0 noise is injected at lambda_z_max before
  the decoder; the diffusion prior bounds the bitrate of z_clean.

Pluggable wrapper:
- VQPTBottleneck: Adapter around any quantizer from lucidrains/vector-quantize-pytorch
  (VectorQuantize, ResidualVQ, FSQ, LFQ, SimVQ, LatentQuantize, and their grouped /
  residual variants). Pick via ``vq_type`` and forward extra config via ``vq_kwargs``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor, nn

from ntt.core.model.components.annealing import AnnealingCallback


class FSQuantizer(nn.Module):
    """Finite Scalar Quantization: VQ-VAE Made Simple - https://arxiv.org/abs/2309.15505

    Code adapted from Jax version in Appendix A.1.

    Adapted from: https://github.com/lucidrains/vector-quantize-pytorch/blob/9502a1f447876d53fd37685b226bf28f250dc4a3/
    vector_quantize_pytorch/finite_scalar_quantization.py
    [Copyright (c) 2020 Phil Wang]
    https://github.com/lucidrains/vector-quantize-pytorch/blob/9502a1f447876d53fd37685b226bf28f250dc4a3/LICENSE

    jaxtyping shapes:
        b: batch size
        ld: feature dimension
        cd: codebook dimension

    levels: list of integers, each representing the number of levels for a dimension. Suggested
      configurations from paper:

      Target Size |C| | Levels
      --------------------------
      2^8 = 256       | [8,6,5]
      2^10 = 1024     | [8,5,5,5]
      2^12 = 4096     | [7,5,5,5,5]
      2^14 = 16384    | [8,8,8,6,5]
      2^16 = 65536    | [8,8,8,5,5,5]
    """

    def __init__(
        self,
        levels: list[int],
        latent_dim: int | None = None,
        scale: float | None = None,
    ):
        super().__init__()

        _levels = torch.tensor(levels, dtype=torch.int32)
        self.register_buffer("_levels", _levels, persistent=True)
        self._levels: Int[Tensor, "cd"]  # noqa: F821, UP037

        _basis = torch.cumprod(torch.tensor([1] + levels[:-1]), dim=0, dtype=torch.int32)
        self.register_buffer("_basis", _basis, persistent=True)
        self._basis: Int[Tensor, "cd"]  # noqa: F821, UP037
        self.scale = scale
        self.codebook_dim = len(levels)
        self.latent_dim = latent_dim or self.codebook_dim

        self.has_projections = self.latent_dim != self.codebook_dim
        self.project_in = (
            nn.Linear(self.latent_dim, self.codebook_dim)
            if self.has_projections
            else nn.Identity()
        )
        self.project_out = (
            nn.Linear(self.codebook_dim, self.latent_dim)
            if self.has_projections
            else nn.Identity()
        )

        implicit_codebook = self.indices_to_codes(
            torch.arange(self.codebook_size), project_out=False
        )
        self.register_buffer("implicit_codebook", implicit_codebook, persistent=True)

    def init_weights(self, init_std: float | None = None):
        for proj in (self.project_in, self.project_out):
            if isinstance(proj, nn.Identity):
                continue
            _init_std = init_std or proj.in_features ** (-0.5)
            nn.init.trunc_normal_(proj.weight, std=_init_std)
            if proj.bias is not None:
                proj.bias.data.zero_()

    @property
    def device(self) -> torch.device:
        return self._levels.device

    @property
    def codebook_size(self) -> int:
        """Total number of discrete codes in the codebook."""
        return int(self._levels.prod().item())

    def bound(self, z: Float[Tensor, "... cd"], eps: float = 1e-3) -> Float[Tensor, "... cd"]:
        """Bound `z`, an array of shape (..., cd)."""
        half_l = (self._levels - 1) * (1 + eps) / 2
        offset = torch.where(self._levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def round_ste(self, z: Float[Tensor, "... cd"]) -> Float[Tensor, "... cd"]:
        """Round with straight through gradients."""
        zhat = z.round()
        return z + (zhat - z).detach()

    def quantize(self, z: Float[Tensor, "... cd"]) -> Float[Tensor, "... cd"]:
        """Quantizes z, returns quantized zhat, same shape as z."""
        quantized = self.round_ste(self.bound(z))
        half_width = self._levels // 2  # Renormalize to [-1, 1].
        return quantized / half_width

    def _scale_and_shift(
        self, zhat_normalized: Float[Tensor, "... cd"]
    ) -> Float[Tensor, "... cd"]:
        half_width = self._levels // 2
        return (zhat_normalized * half_width) + half_width

    def _scale_and_shift_inverse(self, zhat: Float[Tensor, "... cd"]) -> Float[Tensor, "... cd"]:
        half_width = self._levels // 2
        return (zhat - half_width) / half_width

    def codes_to_indices(self, zhat: Float[Tensor, "... cd"]) -> Int[Tensor, "... 1"]:
        """Converts a `code` to an index in the codebook."""
        assert zhat.shape[-1] == self.codebook_dim
        zhat = self._scale_and_shift(zhat).float()
        return (zhat * self._basis).sum(dim=-1).to(torch.int32)

    def indices_to_codes(
        self, indices: Int[Tensor, ...], project_out=True
    ) -> Float[Tensor, "... cd"]:
        """Inverse of `codes_to_indices`."""
        indices = indices.unsqueeze(-1)
        codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)
        if project_out:
            codes = self.project_out(codes)
        return codes

    def forward(
        self,
        z: Float[Tensor, "b ld"],  # pyright: ignore[reportRedeclaration]
        return_codes: bool = True,
    ) -> Int[Tensor, " b"] | tuple[Int[Tensor, " b"], Float[Tensor, "b ld"]]:  # noqa: F821
        """Forward pass through quantizer.

        Args:
            z: Latent tensor of shape (batch, latent_dim)

        Returns:
            Tuple of (indices, quantized_output):
                - indices: Discrete token indices (batch,)
                - quantized_output: Continuous quantized values (batch, latent_dim)
        """
        assert z.shape[-1] == self.latent_dim, (
            f"expected dimension of {self.latent_dim} but found dimension of {z.shape[-1]}"
        )
        z: Float[Tensor, "b cd"] = self.project_in(z)
        codes: Float[Tensor, "b cd"] = self.quantize(z)
        indices: Int[Tensor, "b"] = self.codes_to_indices(codes)  # noqa: F821, UP037
        if return_codes:
            out: Float[Tensor, "b ld"] = self.project_out(codes)
            return indices, out
        return indices


class DiagonalGaussianDistribution:
    """Diagonal Gaussian distribution parameterized by mean and log-variance.

    Adapted from diffusers' DiagonalGaussianDistribution
    (diffusers/models/autoencoders/vae.py), used under the Apache-2.0 license:
    https://github.com/huggingface/diffusers
    (see ``LICENSES/diffusers-Apache-2.0.txt``). Provides sampling, mode access,
    and KL divergence computation against a standard normal prior.

    Args:
        parameters: Tensor of shape (B, 2*C, ...) where the first C channels
            are the mean and the last C channels are the log-variance.
        deterministic: If True, always return the mean (zero variance).
    """

    def __init__(self, parameters: Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self, generator: torch.Generator | None = None) -> Tensor:
        """Sample from the distribution using the reparameterization trick."""
        # randn_like (not randn(shape, ...)) so torch.compile can trace through
        # symbolic shapes — randn requires concrete ints in its size arg.
        sample = (
            torch.randn(
                self.mean.size(),
                generator=generator,
                device=self.parameters.device,
                dtype=self.parameters.dtype,
            )
            if generator is not None
            else torch.randn_like(self.mean)
        )
        return self.mean + self.std * sample

    def kl(
        self,
        other: DiagonalGaussianDistribution | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        """KL divergence against another diagonal Gaussian (or standard normal).

        Args:
            other: Another distribution (None = standard normal prior).
            mask: Optional boolean mask broadcastable to the parameter shape.
                Elements where mask is False are excluded from the sum.
                For WanNeural, this is typically (B, 1, 1, N) from a neuron mask.

        Returns:
            Per-sample KL divergence of shape (B,), summed over valid positions.
        """
        if self.deterministic:
            return torch.tensor([0.0], device=self.parameters.device)

        if other is None:
            kl_elementwise = 0.5 * (self.mean.pow(2) + self.var - 1.0 - self.logvar)
        else:
            kl_elementwise = 0.5 * (
                (self.mean - other.mean).pow(2) / other.var
                + self.var / other.var
                - 1.0
                - self.logvar
                + other.logvar
            )

        if mask is not None:
            kl_elementwise = kl_elementwise * mask

        return kl_elementwise.sum(dim=list(range(1, self.mean.ndim)))

    @staticmethod
    def _curvature_reduce(
        v1: Tensor,
        v2: Tensor,
        mode: str,
        eps: float = 1e-6,
        step_thresh: float = 1e-6,
        mask: Tensor | None = None,
    ) -> Tensor:
        # Per-position curvature value at every (b, t, n), reduced (mean) over
        # the positions that pass the step-threshold AND the optional mask.
        # cos:  1 - cos(v1, v2) — direction change between consecutive deltas.
        # l2 :  ‖v2 − v1‖ — magnitude of the second-difference (acceleration).
        if mode == "cos":
            loss = 1.0 - F.cosine_similarity(v1, v2, dim=-1, eps=eps)
        elif mode == "l2":
            loss = (v2 - v1).norm(dim=-1)
        else:
            raise ValueError(f"Unknown curvature mode '{mode}'. Use 'cos' or 'l2'.")

        select: Tensor | None = None
        if step_thresh > 0:
            step1 = v1.norm(dim=-1)
            step2 = v2.norm(dim=-1)
            select = (step1 > step_thresh) & (step2 > step_thresh)

        if mask is not None:
            mask_bool = mask.to(dtype=torch.bool).expand_as(loss)
            select = mask_bool if select is None else (select & mask_bool)

        if select is not None:
            loss = loss[select]
        return loss.mean()

    @staticmethod
    def _cos_curvature(
        v1: Tensor,
        v2: Tensor,
        eps: float = 1e-6,
        step_thresh: float = 1e-6,
        mask: Tensor | None = None,
    ) -> Tensor:
        return DiagonalGaussianDistribution._curvature_reduce(
            v1, v2, mode="cos", eps=eps, step_thresh=step_thresh, mask=mask
        )

    @staticmethod
    def total_curvature(
        features: Tensor,
        mode: str = "cos",
        eps: float = 1e-6,
        step_thresh: float = 1e-6,
        neuron_mask: Tensor | None = None,
    ) -> Tensor:
        """Curvature of the temporal trajectory of `features`.

        `features` has shape (B, C, T, N) — we permute to (B, T, N, C) so time
        is dim=1 and the feature dim is last (cosine_similarity(dim=-1)
        requirement).

        mode="cos": 1 − cos(v1, v2), measures direction change between
        consecutive first-differences.
        mode="l2":  ‖v2 − v1‖, measures the magnitude of the second-difference
        (acceleration).

        If neuron_mask (B, N) is provided, padded neuron positions are excluded
        from the curvature reduction (AND'd with the step-threshold mask).
        """
        if features.ndim != 4:
            raise ValueError(
                f"Expected features with shape (B, C, T, N), got {tuple(features.shape)}"
            )

        # (B, C, T, N) -> (B, T, N, C)
        feats = features.permute(0, 2, 3, 1).contiguous()

        if feats.shape[1] < 3:
            raise ValueError(
                f"Need at least 3 frames for curvature calculation, got {feats.shape[1]}"
            )

        v1 = feats[:, 1:-1] - feats[:, :-2]
        v2 = feats[:, 2:] - feats[:, 1:-1]

        # Reduces C → shape (B, T-2, N); broadcast (B, N) over time.
        curv_mask: Tensor | None = None
        if neuron_mask is not None:
            curv_mask = neuron_mask[:, None, :].to(dtype=torch.bool)

        return DiagonalGaussianDistribution._curvature_reduce(
            v1, v2, mode=mode, eps=eps, step_thresh=step_thresh, mask=curv_mask
        )

    def mode(self) -> Tensor:
        """Return the mode (mean) of the distribution."""
        return self.mean


class VAEBottleneck(nn.Module):
    """Variational autoencoder bottleneck with diagonal Gaussian.

    Adapted from Wan2's reparameterization scheme, using the same
    DiagonalGaussianDistribution as diffusers' AutoencoderKLWan.

    The encoder produces 2 * z_dim channels, which are split into mu and logvar.
    During training, samples z = mu + eps * exp(0.5 * logvar).
    During inference, returns mu (the mode).

    Args:
        z_dim: Latent dimension (channels per spatial position).
        beta: Weight for KL divergence term.
        curvature_weight: Weight for the curvature penalty on sampled z over time.
        curvature_step_thresh: Minimum step norm (on v1 and v2) for a (t, n) position
            to contribute to the curvature penalty. Floors out near-stationary frames.
        curvature_metric: "cos" → 1 − cos(v1, v2); "l2" → ‖v2 − v1‖.
    """

    def __init__(
        self,
        z_dim: int,
        beta: float = 1.0,
        curvature_weight: float = 0.0,
        curvature_step_thresh: float = 1e-6,
        curvature_metric: str = "cos",
    ):
        super().__init__()
        self.z_dim = z_dim
        self.beta = beta
        self.curvature_weight = float(curvature_weight)
        self.curvature_step_thresh = float(curvature_step_thresh)
        self.curvature_metric = str(curvature_metric)
        self._last_kl: Tensor | None = None
        self._last_curvature: Tensor | None = None
        # Grad-tracking scaled terms, repopulated on every encode() call.
        # Read by WanNeuralTokenizer.forward to build a per-component aux_loss dict.
        self._kl_term: Tensor | None = None
        self._curvature_term: Tensor | None = None

        # Pre/post conv as in Wan2's quant_conv / post_quant_conv
        self.pre_conv = nn.Conv2d(z_dim * 2, z_dim * 2, kernel_size=1)
        self.post_conv = nn.Conv2d(z_dim, z_dim, kernel_size=1)

    def encode(self, h: Tensor, neuron_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Encode to mu, logvar and sample z.

        Args:
            h: Encoder output (B, z_dim * 2, T_comp, N).
            neuron_mask: Optional (B, N) bool mask, True = valid neuron.
                Padded neuron positions are excluded from the KL loss.

        Returns:
            (z, aux_loss) where z is (B, z_dim, T_comp, N) and aux_loss is KL
            (plus optional curvature penalty) scalar.
        """
        h = self.pre_conv(h)
        posterior = DiagonalGaussianDistribution(h)

        if self.training:
            z = posterior.sample()
        else:
            z = posterior.mode()

        # Build KL mask: (B, N) → (B, 1, 1, N) for broadcasting over (B, C, T, N)
        kl_mask = None
        if neuron_mask is not None:
            kl_mask = neuron_mask[:, None, None, :].float()

        # KL divergence against standard normal, mean over batch
        kl = posterior.kl(mask=kl_mask).mean()
        self._last_kl = kl.detach()
        kl_term = self.beta * kl
        self._kl_term = kl_term
        aux_loss = kl_term

        # Reset curvature slots; only populated if we actually compute curvature.
        self._last_curvature = None
        self._curvature_term = None

        # Curvature AFTER sampling: use z (B, C, T, N). Needs T≥3 for second
        # differences. When the tokenizer is running its tile-batched path,
        # T_comp per tile == 1, so we skip here and let the tokenizer override
        # _curvature_term and _last_curvature with the cross-tile value.
        if self.curvature_weight > 0 and self.training and z.shape[2] >= 3:
            curv = posterior.total_curvature(
                z,
                mode=self.curvature_metric,
                step_thresh=self.curvature_step_thresh,
                neuron_mask=neuron_mask,
            )
            self._last_curvature = curv.detach()
            curv_term = self.curvature_weight * curv
            self._curvature_term = curv_term
            aux_loss = aux_loss + curv_term

        return z, aux_loss

    def decode(self, z: Tensor) -> Tensor:
        """Post-process latent for decoder input.

        Args:
            z: Latent (B, z_dim, T_comp, N).

        Returns:
            (B, z_dim, T_comp, N).
        """
        return self.post_conv(z)

    @property
    def encoder_out_channels(self) -> int:
        """Number of channels the encoder head should produce."""
        return self.z_dim * 2


class VAEBetaAnnealingCallback(AnnealingCallback):
    """Anneals :attr:`VAEBottleneck.beta` (the KL weight) over training.

    Also logs the bottleneck's raw KL (``_last_kl``) each batch for diagnostics.
    See :class:`~ntt.core.model.components.annealing.AnnealingCallback` for the
    schedule arguments and behaviour.
    """

    target_class = VAEBottleneck
    target_param = "beta"
    raw_attr = "_last_kl"
    raw_log_key = "train/kl_raw"


class MMDVAEBottleneck(nn.Module):
    """InfoVAE / MMD-VAE bottleneck.

    Replaces the per-sample KL of VAEBottleneck with an MMD divergence between
    the aggregate posterior q_phi(z) and the standard-normal prior. The
    aggregate posterior is the marginal distribution of z the decoder actually
    sees during training, so regularizing *it* (rather than each per-sample
    posterior) directly fixes the two failure modes of vanilla VAE: holes in
    latent space (prior samples land in untrained regions) and posterior
    collapse (decoder ignores z).

    Architecturally identical to VAEBottleneck: same 2*z_dim encoder head,
    same DiagonalGaussianDistribution reparameterization, same pre/post 1x1
    conv. The only difference is the auxiliary loss.

    The MMD is computed with an RBF kernel
    ``k(x, y) = exp(-||x - y||^2 / (z_dim * kernel_bandwidth_div))``,
    using a biased V-statistic over an aggregate-posterior batch (one sample
    per (b, t, n) position, masked-out neurons excluded) and a fresh batch
    of N(0, I) prior samples.

    Args:
        z_dim: Latent dimension (channels per spatial position).
        mmd_lambda: Weight on the MMD term. Pick so the MMD term and the
            reconstruction term have comparable magnitudes at convergence;
            1.0 matches the reference TF InfoVAE implementation.
        kernel_bandwidth_div: Divisor for the RBF kernel bandwidth (kernel
            uses ``mean`` over the feature dim, so bandwidth scales with
            ``1/(z_dim * kernel_bandwidth_div)``). 1.0 reproduces the
            reference (effective bandwidth = 1/z_dim).
        n_prior_samples: Number of N(0, I) samples to draw for the MMD's
            prior batch. ``None`` (default) matches the aggregate-posterior
            sample count for a balanced V-statistic.
    """

    def __init__(
        self,
        z_dim: int,
        mmd_lambda: float = 1.0,
        kernel_bandwidth_div: float = 1.0,
        n_prior_samples: int | None = None,
        mmd_max_samples: int = 2048,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.mmd_lambda = float(mmd_lambda)
        self.kernel_bandwidth_div = float(kernel_bandwidth_div)
        self.n_prior_samples = n_prior_samples
        # Cap the posterior sample set fed into the O(n²) kernel computation.
        # At batch=512, T=10, N=32 we get 163k samples which would materialise
        # a 163k×163k×z_dim tensor — far too large for any GPU. 2048 is enough
        # for a low-variance MMD estimate while keeping the kernel matrix small
        # (2048² × 16 × 2 bytes ≈ 128 MB).
        self.mmd_max_samples = mmd_max_samples

        # Slots populated each encode() call for downstream logging / loss split.
        self._last_mmd: Tensor | None = None
        self._mmd_term: Tensor | None = None

        # Same pre/post 1x1 conv as VAEBottleneck.
        self.pre_conv = nn.Conv2d(z_dim * 2, z_dim * 2, kernel_size=1)
        self.post_conv = nn.Conv2d(z_dim, z_dim, kernel_size=1)

    @property
    def encoder_out_channels(self) -> int:
        # mu + logvar — same head as VAEBottleneck.
        return self.z_dim * 2

    def _compute_kernel(self, x: Tensor, y: Tensor) -> Tensor:
        """RBF kernel matrix ``k(x_i, y_j) = exp(-mean((x_i - y_j)^2) / div)``.

        The mean over the feature dim mirrors the reference InfoVAE code
        (Zhao et al.); combined with ``kernel_bandwidth_div = 1.0`` it gives
        the standard "1/z_dim" bandwidth.
        """
        diff = x.unsqueeze(1) - y.unsqueeze(0)  # (n, m, d)
        sq = diff.pow(2).mean(dim=-1)  # (n, m)
        return torch.exp(-sq / self.kernel_bandwidth_div)

    def _compute_mmd(self, x: Tensor, y: Tensor) -> Tensor:
        """Biased V-statistic MMD between ``x`` and ``y``."""
        return (
            self._compute_kernel(x, x).mean()
            + self._compute_kernel(y, y).mean()
            - 2.0 * self._compute_kernel(x, y).mean()
        )

    def encode(self, h: Tensor, neuron_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Encode to mu, logvar; sample z; compute MMD against the prior.

        Args:
            h: Encoder output ``(B, 2*z_dim, T_comp, N)``.
            neuron_mask: Optional ``(B, N)`` bool mask, True = valid neuron.
                Padded neuron positions are excluded from the aggregate-
                posterior sample set used in the MMD computation.

        Returns:
            ``(z, aux_loss)`` where ``z`` is ``(B, z_dim, T_comp, N)`` and
            ``aux_loss = mmd_lambda * MMD(q_phi(z) || N(0, I))`` (a scalar).
        """
        h = self.pre_conv(h)
        posterior = DiagonalGaussianDistribution(h)

        if self.training:
            z = posterior.sample()
        else:
            z = posterior.mode()

        # Aggregate-posterior sample set: each (b, t, n) is one i.i.d. draw.
        B, C, T, N = z.shape
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, C)

        if neuron_mask is not None:
            # (B, N) -> (B, T, N) -> (B*T*N,) — same flatten order as z_flat.
            mask_flat = neuron_mask[:, None, :].expand(B, T, N).reshape(-1)
            z_flat = z_flat[mask_flat]
            if z_flat.shape[0] == 0:
                # Pathological: no valid neurons in the batch. Skip MMD this step.
                mmd_zero = h.new_tensor(0.0)
                self._last_mmd = mmd_zero.detach()
                self._mmd_term = mmd_zero
                return z, mmd_zero

        n_q = z_flat.shape[0]
        if n_q > self.mmd_max_samples:
            idx = torch.randperm(n_q, device=z_flat.device)[: self.mmd_max_samples]
            z_flat = z_flat[idx]
            n_q = self.mmd_max_samples
        n_prior = self.n_prior_samples if self.n_prior_samples is not None else n_q
        z_prior = torch.randn(n_prior, C, device=z.device, dtype=z.dtype)

        mmd = self._compute_mmd(z_prior, z_flat)
        self._last_mmd = mmd.detach()
        mmd_term = self.mmd_lambda * mmd
        self._mmd_term = mmd_term
        return z, mmd_term

    def decode(self, z: Tensor) -> Tensor:
        """Apply post_conv (1x1) before the decoder, mirroring VAEBottleneck."""
        return self.post_conv(z)


class FSQBottleneck(nn.Module):
    """Finite Scalar Quantization bottleneck.

    Wraps the existing FSQuantizer from the fsqffn tokenizer. Quantizes each
    spatial position (T_comp, N) independently into a discrete code.

    Args:
        z_dim: Latent dimension (channels per spatial position).
        levels: FSQ quantization levels per dimension.
        q_dim: Projection dimension for FSQ (None = len(levels)).
    """

    def __init__(
        self,
        z_dim: int,
        levels: tuple[int, ...] = (8, 5, 5, 5),
        q_dim: int | None = None,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.quantizer = FSQuantizer(
            levels=list(levels),
            latent_dim=z_dim,
            scale=None,
        )

    def encode(self, h: Tensor, neuron_mask: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        """Quantize encoder output.

        Args:
            h: Encoder output (B, z_dim, T_comp, N).
            neuron_mask: Unused (FSQ has no auxiliary loss). Accepted for
                interface consistency with VAEBottleneck.

        Returns:
            (z_quantized, None) where z_quantized has same shape as input.
            No auxiliary loss for FSQ.
        """
        B, C, T, N = h.shape

        # Reshape to (B*T*N, z_dim) for per-position quantization
        z_flat = h.permute(0, 2, 3, 1).reshape(-1, C)
        indices, codes = self.quantizer(z_flat, return_codes=True)

        # Reshape back
        z_quantized = codes.reshape(B, T, N, -1).permute(0, 3, 1, 2)
        return z_quantized, None

    def decode(self, z: Tensor) -> Tensor:
        """Pass-through (quantization already happened in encode).

        Args:
            z: Quantized latent (B, z_dim, T_comp, N).

        Returns:
            Same tensor, unmodified.
        """
        return z

    def get_indices(self, h: Tensor) -> Tensor:
        """Get discrete token indices from encoder output.

        Args:
            h: Encoder output (B, z_dim, T_comp, N).

        Returns:
            Integer indices (B, T_comp * N).
        """
        B, C, T, N = h.shape
        z_flat = h.permute(0, 2, 3, 1).reshape(-1, C)
        indices = self.quantizer(z_flat, return_codes=False)
        return indices.reshape(B, T * N)

    def indices_to_codes(self, indices: Tensor, T: int) -> Tensor:
        """Convert indices back to quantized codes for decoding.

        Args:
            indices: (B, T_comp * N) integer indices.
            T: Compressed time dimension.
            N: Neuron dimension.

        Returns:
            (B, z_dim, T, N) quantized codes.
        """
        B = indices.shape[0]
        flat_indices = indices.reshape(-1)
        codes = self.quantizer.indices_to_codes(flat_indices, project_out=True)
        N = codes.shape[0] // (B * T)
        return codes.reshape(B, T, N, -1).permute(0, 3, 1, 2)

    @property
    def encoder_out_channels(self) -> int:
        """Number of channels the encoder head should produce."""
        return self.z_dim

    @property
    def codebook_size(self) -> int:
        return self.quantizer.codebook_size


class BSQBottleneck(nn.Module):
    """Binary Spherical Quantization bottleneck.

    Implements BSQ from https://arxiv.org/abs/2406.07548 as used in the
    Infinity model. Each latent dimension is quantized to {-1, +1} after
    L2 normalization, giving a codebook of size 2^codebook_dim.

    Straight-through gradients flow through the sign quantization. An entropy
    penalty encourages full codebook utilization: it minimizes per-sample
    entropy (making individual codes sharp) while maximizing batch-level
    entropy (spreading usage across codes).

    Args:
        z_dim: Latent dimension (channels per spatial position from encoder).
        codebook_dim: Number of binary dimensions. Codebook size = 2^codebook_dim.
            If None, defaults to z_dim (no projection needed).
        inv_temperature: Inverse temperature for soft entropy computation.
        gamma0: Weight for per-sample entropy loss (minimize → sharp codes).
        gamma: Weight for batch entropy loss (maximize → full utilization).
        zeta: Overall scaling for the entropy penalty.
    """

    def __init__(
        self,
        z_dim: int,
        codebook_dim: int | None = None,
        inv_temperature: float = 100.0,
        gamma0: float = 1.0,
        gamma: float = 1.0,
        zeta: float = 1.0,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.codebook_dim = codebook_dim or z_dim
        self.codebook_scale = 1.0 / math.sqrt(self.codebook_dim)
        self.inv_temperature = inv_temperature
        self.gamma0 = gamma0
        self.gamma = gamma
        self.zeta = zeta

        # Project between z_dim and codebook_dim if they differ
        needs_proj = self.z_dim != self.codebook_dim
        self.project_in = nn.Linear(z_dim, self.codebook_dim) if needs_proj else nn.Identity()
        self.project_out = nn.Linear(self.codebook_dim, z_dim) if needs_proj else nn.Identity()

        # Mask for converting bit patterns to integer indices
        # bit 0 is MSB: index = sum(bit_i * 2^(codebook_dim - 1 - i))
        self.register_buffer(
            "_bit_mask",
            2 ** torch.arange(self.codebook_dim - 1, -1, -1),
        )
        self._bit_mask: Tensor

    def _l2norm(self, x: Tensor) -> Tensor:
        return F.normalize(x, dim=-1)

    def _quantize(self, z: Tensor) -> Tensor:
        """Sign quantization on the scaled hypercube with straight-through gradient.

        Scales by codebook_scale (= 1/√d) inside the STE so the forward output
        lands on the unit sphere AND ∂out/∂z = 1 (matching Infinity's
        quantize_new in multiscale_bsq.py).

        Args:
            z: (..., codebook_dim) post-l2norm input.

        Returns:
            Codes on the scaled hypercube `±1/√d`, same shape as z.
        """
        zhat = (
            torch.where(
                z > 0,
                torch.ones_like(z),
                -torch.ones_like(z),
            )
            * self.codebook_scale
        )
        return z + (zhat - z).detach()

    def _entropy_loss(self, z: Tensor) -> Tensor:
        """Soft entropy penalty from BSQ paper.

        Analytically computes per-sample and batch entropy using sigmoid
        probabilities, avoiding expensive codebook enumeration.

        Args:
            z: (..., codebook_dim) pre-quantization values (after l2norm).

        Returns:
            Scalar entropy penalty loss.
        """
        # Soft probability that each bit is 0 (negative)
        # Scale factor: -4 / sqrt(d) * inv_temperature (from Infinity implementation)
        p = torch.sigmoid(-4.0 * z / math.sqrt(self.codebook_dim) * self.inv_temperature)
        prob = torch.stack([p, 1 - p], dim=-1)  # (..., codebook_dim, 2)

        eps = 1e-8
        # Per-sample entropy: H(p_i) for each dimension, summed over dims, averaged over batch
        per_sample_entropy = (-prob * torch.log(prob + eps)).sum(dim=-1).sum(dim=-1).mean()

        # Batch entropy: average probabilities across all positions, then compute entropy
        # prob is (..., codebook_dim, 2) — flatten the leading dims and mean.
        avg_prob = prob.reshape(-1, prob.shape[-2], prob.shape[-1]).mean(
            dim=0
        )  # (codebook_dim, 2)
        batch_entropy = (-avg_prob * torch.log(avg_prob + eps)).sum(dim=-1).sum()

        # Minimize per-sample entropy (sharp codes), maximize batch entropy (utilization)
        # The penalty: gamma0 * per_sample - gamma * batch (minimize this)
        entropy_penalty = self.gamma0 * per_sample_entropy - self.gamma * batch_entropy
        return self.zeta * entropy_penalty / self.inv_temperature

    def encode(self, h: Tensor, neuron_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Quantize encoder output using BSQ.

        Args:
            h: Encoder output (B, z_dim, T_comp, N).
            neuron_mask: Unused (accepted for interface consistency).

        Returns:
            (z_quantized, aux_loss) where z_quantized has shape (B, z_dim, T_comp, N)
            and aux_loss is the entropy penalty scalar.
        """
        B, C, T, N = h.shape

        # (B, C, T, N) → (B*T*N, C) for per-position operations
        z = h.permute(0, 2, 3, 1).reshape(-1, C)

        # Project to codebook dimension
        z = self.project_in(z)

        # L2 normalize (BSQ operates on the unit sphere)
        z = self._l2norm(z)

        # Compute entropy loss before quantization (uses soft probabilities)
        if self.training:
            aux_loss = self._entropy_loss(z)
        else:
            aux_loss = h.new_tensor(0.0)

        # Sign quantization with straight-through gradient (scales to ±1/√d inside STE)
        quantized = self._quantize(z)

        # Project back to z_dim
        quantized = self.project_out(quantized)

        z_quantized = quantized.reshape(B, T, N, -1).permute(0, 3, 1, 2)
        return z_quantized, aux_loss

    def decode(self, z: Tensor) -> Tensor:
        """Pass-through (quantization already happened in encode).

        Args:
            z: Quantized latent (B, z_dim, T_comp, N).

        Returns:
            Same tensor, unmodified.
        """
        return z

    def get_indices(self, h: Tensor) -> Tensor:
        """Get discrete token indices from encoder output.

        Args:
            h: Encoder output (B, z_dim, T_comp, N).

        Returns:
            Integer indices (B, T_comp * N), each in [0, 2^codebook_dim).
        """
        B, C, T, N = h.shape
        z = h.permute(0, 2, 3, 1).reshape(-1, C)
        z = self.project_in(z)
        z = self._l2norm(z)

        # Hard sign → bit pattern → integer index
        bits = (z > 0).int()  # (B*T*N, codebook_dim)
        indices = (bits * self._bit_mask).sum(dim=-1)  # (B*T*N,)
        return indices.reshape(B, T * N)

    def indices_to_codes(self, indices: Tensor, T: int) -> Tensor:
        """Convert integer indices back to quantized codes for decoding.

        Mirrors the post-quantize, pre-`project_out` representation produced
        by encode(): codes lie on the scaled hypercube `±1/√d`.

        Args:
            indices: (B, T_comp * N) integer indices in [0, 2^codebook_dim).
            T: Compressed time dimension.

        Returns:
            (B, z_dim, T, N) latent codes ready for the decoder.
        """
        B = indices.shape[0]
        flat_indices = indices.reshape(-1)

        bits = ((flat_indices.unsqueeze(-1).long() & self._bit_mask) != 0).float()
        codes = (bits * 2 - 1) * self.codebook_scale  # ±1/√d, matches encode()

        codes = self.project_out(codes)
        N = codes.shape[0] // (B * T)
        return codes.reshape(B, T, N, -1).permute(0, 3, 1, 2)

    @property
    def encoder_out_channels(self) -> int:
        """Number of channels the encoder head should produce."""
        return self.z_dim

    @property
    def codebook_size(self) -> int:
        return 2**self.codebook_dim


class BSQZetaAnnealingCallback(AnnealingCallback):
    """Anneals :attr:`BSQBottleneck.zeta` (the entropy-penalty scale) over training.

    ``value_min > value_max`` ramps DOWN: push codebook spread early, refine
    reconstruction late. See
    :class:`~ntt.core.model.components.annealing.AnnealingCallback` for the
    schedule arguments and behaviour.
    """

    target_class = BSQBottleneck
    target_param = "zeta"


# Maps `vq_type` strings to (vector_quantize_pytorch class name, channel layout).
# Layout values: "channel_last" expects forward input as (B, N_seq, D);
# "channel_first" expects (B, D, N_seq) — only LatentQuantize uses this.
_VQPT_REGISTRY: dict[str, tuple[str, str]] = {
    "vq": ("VectorQuantize", "channel_last"),
    "residual_vq": ("ResidualVQ", "channel_last"),
    "grouped_residual_vq": ("GroupedResidualVQ", "channel_last"),
    "fsq": ("FSQ", "channel_last"),
    "residual_fsq": ("ResidualFSQ", "channel_last"),
    "grouped_residual_fsq": ("GroupedResidualFSQ", "channel_last"),
    "lfq": ("LFQ", "channel_last"),
    "residual_lfq": ("ResidualLFQ", "channel_last"),
    "grouped_residual_lfq": ("GroupedResidualLFQ", "channel_last"),
    "latent_quantize": ("LatentQuantize", "channel_first"),
    "sim_vq": ("SimVQ", "channel_last"),
    "residual_sim_vq": ("ResidualSimVQ", "channel_last"),
}


class VQPTBottleneck(nn.Module):
    """Wrapper around any quantizer from ``vector_quantize_pytorch``.

    Provides a single configurable bottleneck that can stand in for any of the
    quantizers exposed by lucidrains/vector-quantize-pytorch (VectorQuantize,
    ResidualVQ, FSQ, LFQ, SimVQ, LatentQuantize, and their grouped / residual
    variants). The wrapped module operates per spatial position on a
    ``(B, z_dim, T_comp, N)`` feature map: ``(T, N)`` are flattened into a
    single sequence dim before quantization and reshaped back afterwards.

    The wrapped module's own ``project_in`` / ``project_out`` linears handle
    any z_dim ↔ codebook_dim mismatch, so the encoder simply outputs
    ``z_dim`` channels.

    Args:
        z_dim: Latent feature dimension (channels per spatial position).
        vq_type: Which vqpt class to use. See ``_VQPT_REGISTRY`` for the full
            list. Examples: ``"vq"``, ``"fsq"``, ``"lfq"``, ``"residual_vq"``,
            ``"residual_fsq"``, ``"sim_vq"``, ``"latent_quantize"``.
        loss_weight: Multiplier applied to whatever auxiliary loss the wrapped
            quantizer returns (commitment, entropy, quantization, etc).
        vq_kwargs: Forwarded to the wrapped quantizer's constructor. ``dim``
            is auto-supplied from ``z_dim`` and must not be passed here.
    """

    def __init__(
        self,
        z_dim: int,
        vq_type: str = "vq",
        loss_weight: float = 1.0,
        vq_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__()
        if vq_type not in _VQPT_REGISTRY:
            raise ValueError(
                f"Unknown vq_type {vq_type!r}. Must be one of: {sorted(_VQPT_REGISTRY)}"
            )

        # Lazy import: vector_quantize_pytorch is only required when this class
        # is actually instantiated, so the rest of the wanneural package can be
        # used without having the optional dependency installed.
        import vector_quantize_pytorch as vqpt_mod

        class_name, layout = _VQPT_REGISTRY[vq_type]
        cls = getattr(vqpt_mod, class_name)

        kwargs: dict[str, Any] = dict(vq_kwargs or {})
        if "dim" in kwargs:
            raise ValueError(
                "Pass `z_dim` instead of `dim`: VQPTBottleneck supplies `dim` "
                "to the underlying quantizer automatically."
            )
        kwargs["dim"] = z_dim

        self.z_dim = z_dim
        self.vq_type = vq_type
        self.loss_weight = float(loss_weight)
        self._layout = layout
        self.quantizer: nn.Module = cls(**kwargs)

        # Track scaled aux loss for the wanneural per-component loss dict, in the
        # same style as VAEBottleneck._kl_term.
        self._last_aux_loss: Tensor | None = None
        self._aux_loss_term: Tensor | None = None

    def _to_seq(self, h: Tensor) -> tuple[Tensor, int, int]:
        """Flatten (T, N) into a single sequence dim.

        Returns a tensor in the layout the wrapped quantizer expects, plus
        the original (T, N) so we can unflatten the output.
        """
        B, C, T, N = h.shape
        # (B, C, T, N) -> (B, T*N, C)
        x = h.flatten(2).transpose(1, 2)
        if self._layout == "channel_first":
            # LatentQuantize wants (B, C, N_seq).
            x = x.transpose(1, 2)
        return x, T, N

    def _from_seq(self, x: Tensor, T: int, N: int) -> Tensor:
        """Inverse of ``_to_seq`` for the quantized output."""
        if self._layout == "channel_first":
            x = x.transpose(1, 2)
        return x.reshape(x.shape[0], T, N, x.shape[2]).permute(0, 3, 1, 2)

    @staticmethod
    def _unpack_forward(out: Any) -> tuple[Tensor, Tensor, Tensor | None]:
        """Normalize the variable-shape forward outputs of vqpt classes.

        Returns ``(quantized, indices, aux_loss)`` where ``aux_loss`` may be
        ``None`` (FSQ) or a tensor (every other variant). Residual variants
        return per-layer losses stacked along a trailing dim — those are
        summed into a scalar here.
        """
        if not isinstance(out, tuple):
            out = tuple(out)  # LFQ returns a NamedTuple
        if len(out) == 2:
            quantized, indices = out
            return quantized, indices, None
        if len(out) >= 3:
            quantized, indices, aux = out[0], out[1], out[2]
            if isinstance(aux, Tensor) and aux.ndim > 0:
                aux = aux.sum()
            return quantized, indices, aux
        raise RuntimeError(f"Unexpected vqpt forward output of length {len(out)}")

    def encode(self, h: Tensor, neuron_mask: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        """Quantize encoder output.

        Args:
            h: Encoder output ``(B, z_dim, T_comp, N)``.
            neuron_mask: Unused. Accepted for interface consistency with the
                other wanneural bottlenecks. (vqpt's per-position quantization
                does not depend on which neuron a token came from.)

        Returns:
            ``(z_quantized, aux_loss)`` where ``z_quantized`` has the same
            shape as ``h`` and ``aux_loss`` is a scalar tensor (or ``None``
            for FSQ-family quantizers, which have no auxiliary loss).
        """
        del neuron_mask  # unused
        x, T, N = self._to_seq(h)
        out = self.quantizer(x)
        quantized, _indices, aux = self._unpack_forward(out)
        z_q = self._from_seq(quantized, T=T, N=N)

        if aux is None:
            self._last_aux_loss = None
            self._aux_loss_term = None
            return z_q, None

        self._last_aux_loss = aux.detach()
        aux_term = self.loss_weight * aux
        self._aux_loss_term = aux_term
        return z_q, aux_term

    def decode(self, z: Tensor) -> Tensor:
        """Pass-through; quantization already happened in ``encode``."""
        return z

    def get_indices(self, h: Tensor) -> Tensor:
        """Discrete token indices for ``h``.

        Args:
            h: Encoder output ``(B, z_dim, T_comp, N)``.

        Returns:
            Indices with the (T, N) positions flattened into a single sequence
            dim. Shape depends on the wrapped quantizer:

              - Single-codebook (vq, fsq, lfq, sim_vq, latent_quantize with
                num_codebooks=1): ``(B, T_comp * N)``.
              - Multi-codebook / residual variants:
                ``(B, T_comp * N, num_quantizers)``.
              - Grouped variants: ``(num_groups, B, T_comp * N, num_quantizers)``.
        """
        x, _T, _N = self._to_seq(h)
        out = self.quantizer(x)
        _quantized, indices, _aux = self._unpack_forward(out)
        return indices

    def indices_to_codes(self, indices: Tensor, T: int) -> Tensor:
        """Convert token indices back to a ``(B, z_dim, T, N)`` latent map.

        ``indices`` must be flattened over the (T, N) positions in the same
        layout that ``get_indices`` produces:

          - 2D ``(B, T*N)`` for single-codebook quantizers.
          - 3D ``(B, T*N, Q)`` for multi-codebook / residual quantizers.
          - 4D ``(num_groups, B, T*N, Q)`` for grouped variants — the wrapped
            class iterates over the leading group axis internally.

        ``T`` is the compressed time dim; the neuron count is inferred from
        the trailing sequence length.
        """
        # 4D grouped: (num_groups, B, S, Q). 2D / 3D: (B, S[, Q]).
        if indices.ndim == 4:
            B = indices.shape[1]
            seq_len = indices.shape[2]
        else:
            B = indices.shape[0]
            seq_len = indices.shape[1]
        if seq_len % T != 0:
            raise ValueError(
                f"indices seq_len={seq_len} not divisible by T={T} — "
                "indices must be the (T*N)-flattened output of get_indices()"
            )
        N = seq_len // T

        q = self.quantizer
        get_output = getattr(q, "get_output_from_indices", None)
        indices_to_codes = getattr(q, "indices_to_codes", None)
        if get_output is not None:
            codes = get_output(indices)
        elif indices_to_codes is not None:
            codes = indices_to_codes(indices)
        else:
            raise NotImplementedError(
                f"VQPTBottleneck(vq_type={self.vq_type!r}) does not support "
                "decoding from integer indices: the wrapped class has neither "
                "`get_output_from_indices` nor `indices_to_codes`."
            )

        # Some classes (e.g. FSQ/LFQ with channel_first) emit channel-first;
        # SimVQ in channel_first mode does the same. Normalize to (B, S, D)
        # before reshaping back to (B, C, T, N).
        if codes.ndim == 3 and codes.shape[1] == self.z_dim and codes.shape[2] != self.z_dim:
            codes = codes.transpose(1, 2)
        if codes.shape[-1] != self.z_dim:
            raise RuntimeError(
                f"Unexpected codes shape {tuple(codes.shape)} from "
                f"{type(q).__name__}.indices_to_codes — last dim should be "
                f"z_dim={self.z_dim}."
            )
        return codes.reshape(B, T, N, codes.shape[-1]).permute(0, 3, 1, 2)

    @property
    def encoder_out_channels(self) -> int:
        """Number of channels the encoder head should produce."""
        return self.z_dim

    @property
    def codebook_size(self) -> int | None:
        """Per-layer codebook size of the wrapped quantizer, if it exposes one.

        Note: for residual / grouped variants the *effective* token vocabulary
        is ``codebook_size`` per quantizer × ``num_quantizers`` tokens per
        position, not a single integer.
        """
        size = getattr(self.quantizer, "codebook_size", None)
        if size is None:
            return None
        return int(size)


# ---------------------------------------------------------------------------
# Diffusion-prior bottleneck (Unified Latents)
# ---------------------------------------------------------------------------


class _SinusoidalLambdaEmbed(nn.Module):
    """Sinusoidal embedding of a log-SNR value lambda, followed by a small MLP.

    Used to condition the prior network on the noise level. ``dim`` must be
    even because the embedding is split into a sin and cos half.
    """

    def __init__(self, dim: int, max_period: float = 10_000.0):
        super().__init__()
        if dim % 2:
            raise ValueError(f"dim must be even, got {dim}")
        self.dim = dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, lam: Tensor) -> Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=lam.device, dtype=torch.float32)
            / half
        )
        args = lam.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


class _AdaLNZeroBlock(nn.Module):
    """DiT-style transformer block with AdaLN-Zero conditioning on a scalar embedding.

    All conditioning (scale, shift, gate) is produced by a single linear that is
    zero-initialized so the block starts as identity (residual stream untouched).
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.ada = nn.Linear(dim, 6 * dim)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(
        self,
        x: Tensor,
        cond: Tensor,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        s_msa, b_msa, g_msa, s_mlp, b_mlp, g_mlp = self.ada(cond).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + s_msa.unsqueeze(1)) + b_msa.unsqueeze(1)
        a_out, _ = self.attn(h, h, h, need_weights=False, key_padding_mask=key_padding_mask)
        x = x + g_msa.unsqueeze(1) * a_out
        h = self.norm2(x) * (1 + s_mlp.unsqueeze(1)) + b_mlp.unsqueeze(1)
        x = x + g_mlp.unsqueeze(1) * self.mlp(h)
        return x


class _DiffusionPriorNet(nn.Module):
    """Tiny DiT-style score network for the latent diffusion prior.

    Operates on a (T, N) grid of z_dim-channel tokens flattened into a length
    ``T * N`` sequence. The output projection is zero-initialized so the model
    predicts ``ẑ = 0`` at init — making the prior loss start as
    ``E[exp(λ)/2 · ||z||²]``, a finite quantity that pulls z_clean toward zero
    until the prior network learns to denoise.
    """

    def __init__(self, z_dim: int, dim: int, depth: int, num_heads: int):
        super().__init__()
        self.proj_in = nn.Linear(z_dim, dim)
        self.lam_embed = _SinusoidalLambdaEmbed(dim)
        self.blocks = nn.ModuleList([_AdaLNZeroBlock(dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj_out = nn.Linear(dim, z_dim)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(
        self,
        z_t: Tensor,
        lam: Tensor,
        neuron_mask: Tensor | None = None,
    ) -> Tensor:
        """(B, z_dim, T, N), (B,) -> (B, z_dim, T, N)."""
        B, _C, T, N = z_t.shape
        cond = self.lam_embed(lam)
        x = z_t.flatten(2).transpose(1, 2)
        x = self.proj_in(x)

        key_padding_mask = None
        if neuron_mask is not None:
            # nn.MultiheadAttention key_padding_mask: True = ignore.
            invalid = ~neuron_mask.to(dtype=torch.bool)
            key_padding_mask = invalid[:, None, :].expand(B, T, N).reshape(B, T * N)

        for blk in self.blocks:
            x = blk(x, cond, key_padding_mask=key_padding_mask)
        x = self.norm(x)
        x = self.proj_out(x)
        return x.reshape(x.shape[0], T, N, x.shape[2]).permute(0, 3, 1, 2)


class DiffusionPriorBottleneck(nn.Module):
    """Diffusion-prior latent bottleneck for the WanNeural tokenizer.

    Replaces the VAE's ``β · KL[q(z|x) || N(0, I)]`` regularizer with the
    weighted ELBO of a small diffusion model trained jointly on the encoder
    output. The encoder is deterministic and the only stochasticity in the
    encoder→decoder path is a fixed Gaussian noise of variance σ₀² applied
    at the encoder noise level ``lambda_z_max``.

    Loss components (both gradient-bearing):
        prior_loss   : λ-uniform Monte Carlo estimate of (1/2) ∫ exp(λ) · ||z - ẑ||² dλ
        endpoint_kl  : closed-form KL[N(α_min z, σ_min²) || N(0, I)] at λ_min

    The ``prior_weight`` multiplier is fixed at 1.0 — any other value invalidates
    the bitrate bound. Tune ``c_lf`` on the *decoder* loss instead.

    Args:
        z_dim: Latent channel dimension. The encoder must produce this many
            channels (deterministic — no logvar).
        lambda_z_min: Minimum log-SNR of the prior trajectory (high noise end).
        lambda_z_max: Maximum log-SNR (low noise) — also the encoder's σ₀ level.
        prior_dim: Hidden dimension of the prior network.
        prior_depth: Number of DiT-style blocks in the prior network.
        prior_num_heads: Attention heads per prior block; ``prior_dim`` must
            be divisible by this.
    """

    def __init__(
        self,
        z_dim: int,
        lambda_z_min: float = -10.0,
        lambda_z_max: float = 5.0,
        prior_dim: int = 256,
        prior_depth: int = 6,
        prior_num_heads: int = 4,
    ):
        super().__init__()
        if lambda_z_max <= lambda_z_min:
            raise ValueError(
                f"lambda_z_max ({lambda_z_max}) must be > lambda_z_min ({lambda_z_min})"
            )
        self.z_dim = z_dim
        self.lambda_z_min = float(lambda_z_min)
        self.lambda_z_max = float(lambda_z_max)

        self.prior_net = _DiffusionPriorNet(
            z_dim=z_dim,
            dim=prior_dim,
            depth=prior_depth,
            num_heads=prior_num_heads,
        )
        # Mirror VAEBottleneck.post_conv: a 1x1 channel mixer between the
        # bottleneck output and the decoder. Cheap and matches the VAE path.
        self.post_conv = nn.Conv2d(z_dim, z_dim, kernel_size=1)

        # σ₀ scalars at the encoder's fixed noise level. Pre-cached as Python
        # floats; the bound is invariant to z_clean's dtype/device.
        self._alpha0 = math.sqrt(_sigmoid_scalar(self.lambda_z_max))
        self._sigma0 = math.sqrt(_sigmoid_scalar(-self.lambda_z_max))

        # Endpoint KL constants at lambda_z_min.
        self._alpha_min2 = _sigmoid_scalar(self.lambda_z_min)
        self._sigma_min2 = 1.0 - self._alpha_min2

        # Per-component term cache for the WanNeuralTokenizer.forward aux dict
        # (gradient-bearing references). Mirrors VAEBottleneck._kl_term.
        self._prior_term: Tensor | None = None
        self._endpoint_kl_term: Tensor | None = None
        # Detached diagnostics for logging (no grad).
        self._last_prior_loss: Tensor | None = None
        self._last_endpoint_kl: Tensor | None = None
        self._last_bpd: Tensor | None = None

    @property
    def encoder_out_channels(self) -> int:
        """Number of channels the encoder head should produce.

        The diffusion-prior bottleneck is deterministic — no logvar — so the
        encoder emits exactly ``z_dim`` channels (vs ``2*z_dim`` for the VAE).
        """
        return self.z_dim

    def encode(
        self,
        h: Tensor,
        neuron_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Forward-noise encoder output and compute the prior loss.

        Args:
            h: Deterministic encoder output ``(B, z_dim, T_comp, N)`` — this
                tensor is treated as ``z_clean``.
            neuron_mask: Optional ``(B, N)`` bool mask, True = valid neuron.
                Padded positions are excluded from the prior-loss reduction
                and from prior-net attention; ``z_t`` itself is not masked
                (the prior should see realistic noise structure).

        Returns:
            ``(z_0, aux_loss)`` where ``z_0 = α₀·z_clean + σ₀·ε`` is the
            decoder input (fresh ε every call, train and eval) and
            ``aux_loss = prior_loss + endpoint_kl`` is a scalar with gradient.
        """
        z_clean = h
        B = z_clean.shape[0]
        device = z_clean.device
        dtype = z_clean.dtype

        # ----- Encoder σ₀ noise (always fresh, train + eval). -----
        z_0 = self._alpha0 * z_clean + self._sigma0 * torch.randn_like(z_clean)

        # ----- λ-uniform Monte Carlo of the prior trajectory. -----
        lam_t = torch.empty(B, device=device, dtype=torch.float32).uniform_(
            self.lambda_z_min, self.lambda_z_max
        )
        alpha_t = torch.sigmoid(lam_t).sqrt().to(dtype=dtype).view(B, 1, 1, 1)
        sigma_t = torch.sigmoid(-lam_t).sqrt().to(dtype=dtype).view(B, 1, 1, 1)
        z_t = alpha_t * z_clean + sigma_t * torch.randn_like(z_clean)
        z_pred = self.prior_net(z_t, lam_t, neuron_mask=neuron_mask)

        # Unweighted ELBO with λ-uniform sampling (drops the constant
        # (λ_max - λ_min) factor since prior_weight is fixed at 1.0).
        weight = (lam_t.exp() / 2).to(dtype=dtype).view(B, 1, 1, 1)
        per_elem = weight * (z_clean - z_pred) ** 2
        prior_loss = _masked_mean(per_elem, neuron_mask)

        # ----- Closed-form endpoint KL[N(α_min z, σ_min²) || N(0, I)]. -----
        # 0.5 * (α_min² · z² + σ_min² - 1 - log σ_min²), per element.
        a2 = self._alpha_min2
        s2 = self._sigma_min2
        sq = _masked_mean(z_clean**2, neuron_mask)
        endpoint_kl = 0.5 * (a2 * sq + (s2 - 1.0 - math.log(s2)))

        # ----- Diagnostics (no grad). bpd is in bits per latent dimension. -----
        with torch.no_grad():
            self._last_prior_loss = prior_loss.detach()
            self._last_endpoint_kl = (
                endpoint_kl.detach()
                if torch.is_tensor(endpoint_kl)
                else z_clean.new_tensor(float(endpoint_kl))
            )
            self._last_bpd = (self._last_prior_loss + self._last_endpoint_kl) / math.log(2)

        # Stash gradient-bearing terms so WanNeuralTokenizer.forward can
        # split them into a labeled aux_loss dict.
        self._prior_term = prior_loss
        self._endpoint_kl_term = endpoint_kl
        return z_0, prior_loss + endpoint_kl

    def decode(self, z: Tensor) -> Tensor:
        """Apply post_conv (1x1) before the decoder, mirroring VAEBottleneck."""
        return self.post_conv(z)


def _sigmoid_scalar(x: float) -> float:
    """Numerically-stable sigmoid for a Python float."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _masked_mean(x: Tensor, neuron_mask: Tensor | None) -> Tensor:
    """Mean of x over (B, C, T, N), excluding masked-out neuron positions."""
    if neuron_mask is None:
        return x.mean()
    mask = neuron_mask[:, None, None, :].to(dtype=x.dtype)
    denom = mask.expand_as(x).sum().clamp_min(1.0)
    return (x * mask).sum() / denom
