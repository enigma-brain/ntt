"""Rotary positional embeddings.

Ported from nuclr/src/nn/rotary_embedding.py.

Adapted from the NuCLR reference implementation, used under the Apache-2.0
license: https://github.com/nerdslab/nuclr (Arora et al., 2025, NeurIPS;
https://arxiv.org/abs/2512.01199).
"""

import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch import Tensor

from .sine_emb import SinusoidalTimeEmb


class RotaryEmbedding(nn.Module):
    omega: Tensor

    def __init__(self, head_dim: int, rotate_dim: int, t_min: float = 1e-4, t_max: float = 4.0):
        super().__init__()

        assert head_dim % 2 == 0
        assert rotate_dim % 2 == 0

        periods = SinusoidalTimeEmb.get_periods(rotate_dim // 2, t_min, t_max)
        omega = torch.zeros(head_dim // 2)
        omega[: rotate_dim // 2] = 2 * torch.pi / periods
        self.register_buffer("omega", omega)

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, timestamps: Tensor) -> Tensor:
        """Compute rotation matrices for given timestamps."""
        angles = timestamps.unsqueeze(-1) * self.omega
        angles = repeat(angles, "... n -> ... (n r)", r=2)
        pos_emb = torch.cat((angles.cos(), angles.sin()), dim=-1)
        return pos_emb


def rotate_half(x: Tensor) -> Tensor:
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


def apply_rotary_pos_emb(pos_emb: Tensor, x: Tensor, head_dim: int = 2) -> Tensor:
    """Apply rotary positional embedding to input data."""
    pos_emb = pos_emb.unsqueeze(head_dim).to(x.dtype)
    pos_cos, pos_sin = pos_emb.chunk(chunks=2, dim=-1)
    return (x * pos_cos) + (rotate_half(x) * pos_sin)


def invert_rotary_pos_emb(pos_emb: Tensor) -> Tensor:
    pos_cos, pos_sin = pos_emb.chunk(chunks=2, dim=-1)
    return torch.cat((pos_cos, -pos_sin), dim=-1)
