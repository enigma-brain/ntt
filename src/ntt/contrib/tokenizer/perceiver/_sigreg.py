"""Local SIGReg implementation shared by perceiver tokenizer variants."""

from __future__ import annotations

import torch
import torch.nn as nn


class SlicingEppsPulleyRegularizer(nn.Module):
    """Minimal local SIGReg implementation for tensors shaped ``(*, N, D)``."""

    t: torch.Tensor
    phi: torch.Tensor
    weights: torch.Tensor

    def __init__(self, num_slices: int = 256, n_points: int = 17, t_max: float = 3.0):
        super().__init__()
        self.num_slices = num_slices

        t = torch.linspace(0, t_max, n_points, dtype=torch.float32)
        dt = t_max / (n_points - 1)
        weights = torch.full((n_points,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        phi = t.square().mul(-0.5).exp()

        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None, seed: int | None = None
    ) -> torch.Tensor:
        """Compute SIGReg statistic for ``(B, N, D)`` with optional ``mask (B, N)``.

        When *mask* is provided, only positions marked ``True`` contribute to the
        empirical characteristic-function means and the *N*-scaling factor.
        """
        if seed is None:
            A = torch.randn(x.size(-1), self.num_slices, device=x.device, dtype=x.dtype)
        else:
            generator = torch.Generator(device=x.device)
            generator.manual_seed(int(seed))
            A = torch.randn(
                x.size(-1),
                self.num_slices,
                device=x.device,
                dtype=x.dtype,
                generator=generator,
            )
        A = A / A.norm(p=2, dim=0, keepdim=True).clamp(min=torch.finfo(x.dtype).eps)

        proj = x @ A  # (B, N, S)
        t = self.t.to(device=x.device, dtype=x.dtype)
        x_t = proj.unsqueeze(-1) * t  # (B, N, S, P)

        if mask is not None:
            w = mask.to(dtype=x.dtype).unsqueeze(-1).unsqueeze(-1)  # (B, N, 1, 1)
            count = mask.sum(dim=1, keepdim=True).to(dtype=x.dtype)  # (B, 1)
            count_div = count.unsqueeze(-1).clamp(min=1)  # (B, 1, 1)
            cos_mean = (x_t.cos() * w).sum(dim=-3) / count_div  # (B, S, P)
            sin_mean = (x_t.sin() * w).sum(dim=-3) / count_div  # (B, S, P)
            n_scale = count  # (B, 1)
        else:
            cos_mean = x_t.cos().mean(dim=-3)  # (*, S, P)
            sin_mean = x_t.sin().mean(dim=-3)  # (*, S, P)
            n_scale = x.size(-2)

        phi = self.phi.to(device=x.device, dtype=x.dtype)
        weights = self.weights.to(device=x.device, dtype=x.dtype)
        err = (cos_mean - phi).square() + sin_mean.square()  # (B, S, P)
        stats = (err @ weights) * n_scale  # (B, S)
        return stats.mean()
