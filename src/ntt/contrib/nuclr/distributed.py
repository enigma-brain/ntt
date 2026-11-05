"""Distributed autograd functions for DDP-aware loss computation.

Ported from nuclr/src/distributed.py. Provides AllReduce for correct
loss weighting across GPUs (not just gradient sync).

Adapted from the NuCLR reference implementation, used under the Apache-2.0
license: https://github.com/nerdslab/nuclr (Arora et al., 2025, NeurIPS;
https://arxiv.org/abs/2512.01199).
"""

import torch
import torch.distributed as dist


class AllReduce(torch.autograd.Function):
    """All-reduce with averaging across GPUs.

    Divides by world_size before all_reduce so the result is the mean.
    No-op when DDP is not initialized.
    """

    @staticmethod
    def forward(ctx, x):
        if dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1):
            x = x.contiguous() / dist.get_world_size()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads
