"""Vector Quantization (VQ) bottleneck with CALM regularization recipe."""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class VQOutput(NamedTuple):
    quantized: Tensor  # (*, C) quantized vectors (straight-through)
    indices: Tensor  # (*,) integer codebook indices
    losses: dict[str, Tensor]  # Individual loss components


class VQBottleneck(nn.Module):
    """Vector Quantization bottleneck with EMA updates and CALM regularization.

    Implements classical VQ with:
    - EMA codebook updates (no encoder gradient needed for codebook)
    - Dead code revival from a ring buffer of recent features
    - Commitment loss, entropy loss, and orthogonality loss
    - Straight-through gradient estimator

    Args:
        codebook_size: Number of codebook entries (K).
        embed_dim: Dimensionality of each codebook vector (C).
        ema_decay: Exponential moving average decay for codebook updates.
        dead_code_threshold: Codes used fewer than this many times get revived.
    """

    _ema_count: Tensor
    _ema_weight: Tensor
    _usage_count: Tensor
    _feature_buffer: Tensor
    _buffer_ptr: Tensor
    _eye: Tensor

    def __init__(
        self,
        codebook_size: int,
        embed_dim: int,
        ema_decay: float = 0.99,
        dead_code_threshold: float = 1.0,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self._codebook_size = codebook_size
        self.ema_decay = ema_decay
        self.dead_code_threshold = dead_code_threshold

        # Codebook: K x C
        self.codebook = nn.Embedding(codebook_size, embed_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)

        # EMA state buffers
        self.register_buffer("_ema_count", torch.ones(codebook_size))
        self.register_buffer("_ema_weight", self.codebook.weight.data.clone())

        # Usage tracking (EMA-smoothed so dead code revival stays active throughout training)
        self.register_buffer("_usage_count", torch.zeros(codebook_size))

        # Feature buffer for dead-code revival: ring buffer of 512 recent features
        _buffer_size = 512
        self.register_buffer("_feature_buffer", torch.zeros(_buffer_size, embed_dim))
        self.register_buffer("_buffer_ptr", torch.zeros(1, dtype=torch.long))

        # Cached identity matrix for orthogonality loss
        self.register_buffer("_eye", torch.eye(codebook_size))

    @property
    def codebook_size(self) -> int:
        return self._codebook_size

    def _quantize(self, z_flat: Tensor) -> tuple[Tensor, Tensor]:
        """Find nearest codebook entry for each input vector.

        Args:
            z_flat: (N, C) flat input features.

        Returns:
            indices: (N,) integer codebook indices.
            dists: (N, K) pairwise distances (reused by entropy loss).
        """
        dists = torch.cdist(z_flat, self.codebook.weight)
        indices = dists.argmin(dim=-1)
        return indices, dists

    @torch.no_grad()
    def _ema_update(self, z_flat: Tensor, indices: Tensor) -> None:
        """Update codebook via exponential moving average.

        Adds a small epsilon to the count to avoid division by zero.

        Args:
            z_flat: (N, C) input features.
            indices: (N,) codebook indices assigned to each feature.
        """
        K = self._codebook_size

        # One-hot encoding: (N, K)
        one_hot = F.one_hot(indices, num_classes=K).float()

        # Count per code: (K,)
        count = one_hot.sum(0)

        # Sum of features per code: (K, C)
        weight_sum = one_hot.t() @ z_flat

        # EMA updates
        self._ema_count.mul_(self.ema_decay).add_(count * (1 - self.ema_decay))
        self._ema_weight.mul_(self.ema_decay).add_(weight_sum * (1 - self.ema_decay))

        # Laplace smoothing: avoid zero-division
        smoothed_count = self._ema_count + 1e-5
        updated_weight = self._ema_weight / smoothed_count.unsqueeze(1)

        # Update codebook via .data to avoid invalidating autograd graph
        self.codebook.weight.data.copy_(updated_weight)

    @torch.no_grad()
    def _update_feature_buffer(self, z_flat: Tensor) -> None:
        """Add recent features to the ring buffer."""
        N = z_flat.shape[0]
        buf_size = self._feature_buffer.shape[0]
        ptr = int(self._buffer_ptr.item())

        if N >= buf_size:
            self._feature_buffer.copy_(z_flat[-buf_size:])
            self._buffer_ptr.fill_(0)
        else:
            end = ptr + N
            if end <= buf_size:
                self._feature_buffer[ptr:end] = z_flat
                self._buffer_ptr.fill_(end % buf_size)
            else:
                first_part = buf_size - ptr
                self._feature_buffer[ptr:] = z_flat[:first_part]
                self._feature_buffer[: N - first_part] = z_flat[first_part:]
                self._buffer_ptr.fill_(N - first_part)

    @torch.no_grad()
    def _revive_dead_codes(self) -> None:
        """Replace dead codebook entries with random features from the buffer."""
        dead_mask = self._usage_count < self.dead_code_threshold
        num_dead = dead_mask.sum().item()

        if num_dead == 0:
            return

        # Only sample from the occupied portion of the ring buffer
        ptr = int(self._buffer_ptr.item())
        buf_size = self._feature_buffer.shape[0]
        # The buffer has been filled if any entry beyond ptr is nonzero,
        # otherwise only [0, ptr) contains real features.
        occupied = buf_size if (ptr == 0 or self._feature_buffer[ptr:].any()) else ptr
        if occupied == 0:
            return

        # Can only revive as many codes as we have replacement features
        num_revive = min(num_dead, occupied)
        perm = torch.randperm(occupied, device=self._feature_buffer.device)[:num_revive]
        replacement = self._feature_buffer[perm]

        # Select the first num_revive dead codes to replace
        dead_indices = dead_mask.nonzero(as_tuple=True)[0][:num_revive]

        # Replace dead codes via .data to avoid invalidating autograd graph
        self.codebook.weight.data[dead_indices] = replacement
        self._ema_weight[dead_indices] = replacement
        self._ema_count[dead_indices] = 1.0
        self._usage_count[dead_indices] = 0.0

    def _commitment_loss(self, z_flat: Tensor, quantized: Tensor) -> Tensor:
        """Commitment loss: encourages encoder outputs to stay close to codebook.

        Args:
            z_flat: (N, C) encoder outputs.
            quantized: (N, C) codebook vectors (detached internally).

        Returns:
            Scalar commitment loss.
        """
        return F.mse_loss(z_flat, quantized.detach())

    def _entropy_loss(self, dists: Tensor, temperature: float) -> Tensor:
        """Entropy loss via temperature-scaled softmax to encourage uniform codebook usage.

        Args:
            dists: (N, K) precomputed pairwise distances from _quantize.
            temperature: Softmax temperature for soft assignments.

        Returns:
            Scalar entropy loss (negative entropy, to be minimized).
        """
        soft_assign = F.softmax(-dists / temperature, dim=-1)
        # Average distribution over batch: (K,)
        avg_probs = soft_assign.mean(0)
        # Entropy: H = -sum(p * log(p))
        entropy = -(avg_probs * (avg_probs + 1e-10).log()).sum()
        # Return negative entropy (minimize => maximize entropy)
        return -entropy

    def _orthogonality_loss(self) -> Tensor:
        """Orthogonality loss: penalize codebook vectors for being correlated.

        Computes ||E_norm @ E_norm^T - I||_F where E_norm is row-normalized codebook.

        Returns:
            Scalar orthogonality loss.
        """
        E = self.codebook.weight
        # Row-normalize
        E_norm = F.normalize(E, p=2, dim=1)
        # Gram matrix: (K, K)
        gram = E_norm @ E_norm.t()
        return (gram - self._eye).pow(2).sum().sqrt()

    def forward(
        self,
        z: Tensor,
        temperature: float = 1.0,
        valid_mask: Tensor | None = None,
    ) -> VQOutput:
        """Quantize input features using the codebook.

        Args:
            z: (*, C) input features, typically (B*N, T_d, C).
            temperature: Temperature for entropy loss computation.
            valid_mask: Optional bool mask of shape ``z.shape[:-1]`` (True = real
                token). When supplied, codebook EMA updates, usage counts, the
                dead-code revival buffer, commitment, and entropy losses are
                computed only over valid tokens. Padded tokens still receive a
                straight-through quantized output so the decoder sees a dense
                tensor. When None (default), behavior is unchanged.

        Returns:
            VQOutput with quantized features, indices, and loss components.
        """
        original_shape = z.shape
        z_flat = z.reshape(-1, self.embed_dim)

        # Quantize all positions: decoder needs a dense quantized tensor.
        indices, dists = self._quantize(z_flat)

        if valid_mask is not None:
            valid_flat = valid_mask.reshape(-1).bool()
        else:
            valid_flat = None

        if self.training:
            # Select the training slice: valid tokens only when a mask is set.
            if valid_flat is not None and not bool(valid_flat.all()):
                z_train = z_flat[valid_flat]
                idx_train = indices[valid_flat]
                dists_train = dists[valid_flat]
            else:
                z_train = z_flat
                idx_train = indices
                dists_train = dists

            if z_train.shape[0] > 0:
                # Update usage count with EMA decay so revival stays active throughout training
                K = self._codebook_size
                one_hot = F.one_hot(idx_train, num_classes=K).float()
                self._usage_count.mul_(self.ema_decay).add_(one_hot.sum(0) * (1 - self.ema_decay))

                # EMA codebook update (modifies codebook in-place)
                self._ema_update(z_train, idx_train)
                self._update_feature_buffer(z_train)
                self._revive_dead_codes()

            # Look up codebook vectors for ALL positions for the decoder path.
            quantized = self.codebook(indices)
            quantized_st = z_flat + (quantized - z_flat).detach()

            if z_train.shape[0] > 0:
                commitment = self._commitment_loss(z_train, self.codebook(idx_train))
                entropy = self._entropy_loss(dists_train, temperature)
            else:
                commitment = torch.tensor(0.0, device=z.device, dtype=z.dtype)
                entropy = torch.tensor(0.0, device=z.device, dtype=z.dtype)
            orthogonality = self._orthogonality_loss()
        else:
            quantized = self.codebook(indices)
            quantized_st = quantized  # no STE needed at eval
            commitment = torch.tensor(0.0, device=z.device, dtype=z.dtype)
            entropy = torch.tensor(0.0, device=z.device, dtype=z.dtype)
            orthogonality = torch.tensor(0.0, device=z.device, dtype=z.dtype)

        losses = {
            "commitment": commitment,
            "entropy": entropy,
            "orthogonality": orthogonality,
        }

        # Reshape outputs back to original shape
        quantized_out = quantized_st.reshape(original_shape)
        indices_out = indices.reshape(original_shape[:-1])

        return VQOutput(quantized=quantized_out, indices=indices_out, losses=losses)

    def indices_to_codes(self, indices: Tensor) -> Tensor:
        """Look up codebook vectors from indices.

        Args:
            indices: (*,) integer tensor of codebook indices.

        Returns:
            (*, C) codebook vectors.
        """
        return self.codebook(indices)
