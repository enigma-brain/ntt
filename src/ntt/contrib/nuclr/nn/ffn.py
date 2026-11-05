"""Gated feed-forward network.

Ported from nuclr/src/models/nuclr_v2ga_spikes.py (FFN class).

Adapted from the NuCLR reference implementation, used under the Apache-2.0
license: https://github.com/nerdslab/nuclr (Arora et al., 2025, NeurIPS;
https://arxiv.org/abs/2512.01199).
"""

import torch.nn as nn


class FFN(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0, pre_norm: bool = True):
        super().__init__()
        self.norm = nn.LayerNorm(dim) if pre_norm else nn.Identity()
        self.in_proj = nn.Linear(dim, 2 * dim * mult)
        self.out_proj = nn.Linear(dim * mult, dim)
        self.dp = nn.Dropout(p=dropout)
        self.act = nn.GELU()

    def forward(self, x):
        y = self.norm(x)
        y, gate = self.in_proj(y).chunk(2, dim=-1)
        y = self.act(gate) * y
        y = self.dp(y)
        return self.out_proj(y)
