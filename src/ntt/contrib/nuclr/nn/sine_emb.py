"""Sinusoidal time embeddings.

Ported from nuclr/src/nn/sine_emb.py.

Adapted from the NuCLR reference implementation, used under the Apache-2.0
license: https://github.com/nerdslab/nuclr (Arora et al., 2025, NeurIPS;
https://arxiv.org/abs/2512.01199).
"""

import torch
import torch.nn as nn


class SinusoidalTimeEmb(nn.Module):
    periods: torch.Tensor
    omegas: torch.Tensor

    def __init__(self, dim: int, t_min: float, t_max: float):
        super().__init__()
        assert dim % 2 == 0

        periods = self.get_periods(dim // 2, t_min, t_max)
        omegas = 2 * torch.pi / periods
        self.register_buffer("periods", periods)
        self.register_buffer("omegas", omegas)

    @torch.no_grad()
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        angles = t[..., None] * self.omegas
        return torch.cat((angles.sin(), angles.cos()), dim=-1)

    @staticmethod
    def get_periods(num: int, t_min: float, t_max: float) -> torch.Tensor:
        exponents = torch.linspace(0, 1.0, num, dtype=torch.float32)
        t_min_t, t_max_t = torch.tensor(t_min), torch.tensor(t_max)
        periods = torch.exp(torch.lerp(t_min_t.log(), t_max_t.log(), exponents))
        return periods
