"""SigReg helper utilities for contributor loss modules."""

from __future__ import annotations

import torch
from torch import nn


class SIGReg(nn.Module):
    """Random-feature regularizer over latent trajectories."""

    t: torch.Tensor
    phi: torch.Tensor
    weights: torch.Tensor

    def __init__(
        self,
        knots: int = 17,
        t_max: float = 3.0,
        sketch_dim: int = 256,
        seed: int = 42,
    ):
        super().__init__()
        if knots < 2:
            raise ValueError("knots must be >= 2")
        if sketch_dim <= 0:
            raise ValueError("sketch_dim must be positive")

        self.sketch_dim = int(sketch_dim)
        self._rng = torch.Generator()
        self._rng.manual_seed(int(seed))
        t = torch.linspace(0.0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)
        weights = torch.full((knots,), 2.0 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        if proj.ndim < 2:
            raise ValueError(f"Expected at least 2 dimensions, got {proj.shape}")

        proj_fp32 = proj.float()
        projection = torch.randn(
            proj_fp32.size(-1),
            self.sketch_dim,
            generator=self._rng,
            dtype=torch.float32,
        ).to(proj_fp32.device)
        projection = projection.div_(projection.norm(p=2, dim=0, keepdim=True).clamp_min(1e-12))
        x_t = (proj_fp32 @ projection).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(dim=-3) - self.phi).square() + x_t.sin().mean(dim=-3).square()
        statistic = err @ self.weights
        return statistic.mean()
