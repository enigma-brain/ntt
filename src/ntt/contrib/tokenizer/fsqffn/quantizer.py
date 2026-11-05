"""Finite Scalar Quantization (FSQ) module.

Implements the FSQ quantization technique from "Finite Scalar Quantization: VQ-VAE Made Simple"
https://arxiv.org/abs/2309.15505

Adapted from: https://github.com/lucidrains/vector-quantize-pytorch
[Copyright (c) 2020 Phil Wang]
"""

import torch
from einops import rearrange
from jaxtyping import Float, Int
from torch import Tensor, nn

from .utils import init_weights as init_weights_util


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

        self.codebook_size = self._levels.prod().item()
        implicit_codebook = self.indices_to_codes(
            torch.arange(self.codebook_size), project_out=False
        )
        self.register_buffer("implicit_codebook", implicit_codebook, persistent=True)

    def init_weights(self, init_std: float | None = None):
        if self.has_projections:
            init_weights_util((self.project_in, self.project_out), init_std)

    @property
    def device(self) -> torch.device:
        return self._levels.device

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
        self, indices: Int[Tensor, "..."], project_out=True
    ) -> Float[Tensor, "... cd"]:
        """Inverse of `codes_to_indices`."""
        indices = rearrange(indices, "... -> ... 1")
        codes_non_centered = (indices // self._basis) % self._levels
        codes = self._scale_and_shift_inverse(codes_non_centered)
        if project_out:
            codes = self.project_out(codes)
        return codes

    def forward(
        self,
        z: Float[Tensor, "b ld"],  # pyright: ignore[reportRedeclaration]
        return_codes: bool = True,
    ) -> Int[Tensor, "b"] | tuple[Int[Tensor, "b"], Float[Tensor, "b ld"]]:  # noqa: F821
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
