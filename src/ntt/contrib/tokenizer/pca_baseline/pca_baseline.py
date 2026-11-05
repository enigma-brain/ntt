"""PCA baseline tokenizer (linear, single-neuron, fit on full training data).

Each per-neuron 1-second window is treated as one ``T``-dim sample. During
training, every batch contributes to running mean and outer-product
accumulators (``mean_T`` and ``cov_TT``). The fit is triggered explicitly
by the LightningModule's ``on_train_epoch_end`` hook calling
``fit_from_running_stats()`` — guaranteeing the PCA sees the **entire**
first training epoch before freezing.

After fitting, the encoder is a frozen linear projection ``W ∈ R^{T×k}``
and the decoder is ``W.T`` plus the global mean curve ``μ ∈ R^T``. Output
is clamped to ``[ε, ∞)`` to keep Poisson NLL finite. We deliberately do
**not** softplus the linear output: the empirical mean ``μ`` is small
(~0.013 for 1 ms-binned spikes) so the linear projection is approximately
non-negative by construction, and softplus would floor the reconstruction
at ``log(2) ≈ 0.69`` — wildly above the true rate.

This is a deliberately bare-bones baseline: no learnable parameters apart
from a single dummy scalar (kept around so Lightning's optimizer doesn't
choke). The point is to provide a no-tricks reference number — if the
deep cosmos 1D tokenizer can't beat this at the same compression budget,
the conv stack isn't earning its keep.

Compression ratio = ``window_length_bins / n_components``. With the default
1000-bin / 1 ms window: ``n_components=1`` → 1000:1, ``n_components=10`` → 100:1.

Caveats:
- PCA is **non-causal** — the projection sees the whole window. This is an
  oracle baseline; cosmos 1D is causal post-wavelet.
- PCA assumes Gaussian noise, not Poisson — exponential-family PCA would
  be tighter, but plain PCA is the standard cheap baseline.
- The fit accumulates running statistics (T×T covariance, ~4 MB at T=1000)
  in ``float64`` to keep numerical precision over many batches. Memory
  cost is fixed regardless of training-set size.
- Distributed training: each rank accumulates independently. For multi-GPU
  fits you'd need to ``all_reduce`` ``_n_seen`` / ``_sum_T`` / ``_sum_outer_TT``
  before fitting. Single-GPU runs are deterministic.
"""

from __future__ import annotations

from typing import Literal

import torch
from einops import rearrange
from torch import Tensor, nn

from ntt.core.components.types import (
    BinnedInputType,
    DataType,
    LatentType,
    ReconstructionOutput,
    TokenizerOutput,
    get_data_matrix,
)
from ntt.core.model.tokenizer.base import (
    BaseTokenizer,
    TokenizerSpec,
    register_tokenizer,
)

_RATE_FLOOR = 1e-8


class PCABaselineTokenizerSpec(TokenizerSpec, frozen=True):  # pyright: ignore[reportCallIssue,reportGeneralTypeIssues]
    """Spec for the linear PCA baseline tokenizer.

    Args:
        n_components: Number of PCA components ``k``. Latent has shape
            ``[B, 1, N, k]``. Compression ratio is ``window_length_bins / k``.
    """

    KIND: Literal["pca_baseline"] = "pca_baseline"

    # --- Data ---
    n_neurons: int = 32
    window_length_seconds: float = 1.0
    bin_size_ms: float = 1.0

    # --- Architecture ---
    n_components: int = 10

    # --- Capability metadata ---
    latent_type: LatentType = LatentType.CONTINUOUS
    compress_n: bool = False
    compress_t: bool = True
    variable_n: bool = False
    variable_t: bool = False
    variable_dt: bool = False
    input_type: DataType = DataType.RATES
    output_type: DataType = DataType.RATES

    # One token per window (singleton token axis), carrying a k-dim latent per
    # neuron. The base class auto-infers ``token_bin_size_ms = window_length_seconds * 1000``
    # when ``compress_t=True`` and ``has_time_dim=False``.
    has_time_dim: bool = False
    has_unit_dim: bool = True
    has_latent_dim: bool = True

    def get_vocab_size(self) -> int:
        raise NotImplementedError("Continuous latent — no discrete vocabulary.")

    @property
    def num_latent_units(self) -> int:
        return self.n_neurons


@register_tokenizer
class PCABaselineTokenizer(BaseTokenizer):
    """Linear PCA tokenizer fit on the full training distribution via running stats.

    Encoder: ``z = (x - μ) @ W`` with ``W ∈ R^{T×k}`` (top-k eigenvectors of
    the empirical covariance).
    Decoder: ``x̂ = z @ W.T + μ`` followed by softplus to enforce
    non-negativity.

    Latent shape: ``[B, 1, N, k]`` (the ``[B, T, U, D]`` contract with a singleton
    token axis — one token per window, a ``k``-dim latent per neuron).

    Fitting protocol:
      1. Each training forward pass updates running ``n``, sum, and outer-product.
      2. ``NTTLitModule.on_train_epoch_end`` calls ``fit_from_running_stats()``
         after the first full training epoch — at which point the empirical
         covariance has been built from the entire training set and we
         eigendecompose to freeze ``W`` and ``μ``.
    """

    spec_class = PCABaselineTokenizerSpec
    spec: PCABaselineTokenizerSpec

    def __init__(self, spec: PCABaselineTokenizerSpec) -> None:
        super().__init__(spec)
        T = spec.window_length_bins
        k = spec.n_components
        if k < 1:
            raise ValueError(f"n_components must be >= 1, got {k}")
        if k > T:
            raise ValueError(f"n_components ({k}) must be <= window_length_bins ({T})")

        # Frozen projection + mean populated at fit time.
        self.register_buffer("W", torch.zeros(T, k))
        self.register_buffer("mu", torch.zeros(T))
        self.register_buffer("fitted", torch.tensor(False))
        # Running-statistics accumulators (float64 for numerical precision over many batches).
        self.register_buffer("_n_seen", torch.tensor(0, dtype=torch.long))
        self.register_buffer("_sum_T", torch.zeros(T, dtype=torch.float64))
        self.register_buffer("_sum_outer_TT", torch.zeros(T, T, dtype=torch.float64))
        # Type narrowing for pyright (register_buffer returns generic Module).
        self.W: Tensor
        self.mu: Tensor
        self.fitted: Tensor
        self._n_seen: Tensor
        self._sum_T: Tensor
        self._sum_outer_TT: Tensor

        # Lightning expects at least one nn.Parameter for the optimizer; the
        # PCA path doesn't use it, but a dangling 0-grad scalar keeps the
        # training loop happy.
        self._dummy = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def _accumulate_running_stats(self, flat: Tensor) -> None:
        """Add this batch's contribution to the running stats. No-op once fitted."""
        if bool(self.fitted.item()):
            return
        x = flat.detach().to(torch.float64)
        self._n_seen.add_(x.shape[0])
        self._sum_T.add_(x.sum(dim=0))
        # outer-product accumulator: sum_i x_i x_i^T = X^T X
        self._sum_outer_TT.add_(x.T @ x)

    def fit_from_running_stats(self) -> bool:
        """Compute mean + top-k eigenvectors of the empirical covariance.

        Idempotent: returns False (no-op) if already fitted, or if not enough
        rows have been seen (< k+1). Returns True when the fit completes.

        Called by ``NTTLitModule.on_train_epoch_end`` so the PCA sees the
        full first epoch.
        """
        if bool(self.fitted.item()):
            return False
        n = int(self._n_seen.item())
        k = self.spec.n_components
        if n < k + 1:
            return False
        n_f = float(n)
        mu = self._sum_T / n_f  # [T]
        # Cov = E[xx^T] - μμ^T
        cov = self._sum_outer_TT / n_f - torch.outer(mu, mu)
        # Symmetrise to suppress floating-point asymmetry before eigh.
        cov = 0.5 * (cov + cov.T)
        # eigh returns ascending eigenvalues; take the top-k from the right.
        _, eigvecs = torch.linalg.eigh(cov)
        W = eigvecs[:, -k:].flip(-1).contiguous()  # [T, k] descending eigenvalue order
        self.W.copy_(W.to(self.W.dtype))
        self.mu.copy_(mu.to(self.mu.dtype))
        self.fitted.fill_(True)
        return True

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @staticmethod
    def _to_flat(x: BinnedInputType) -> tuple[Tensor, int]:
        """``[B, T, N]`` → per-neuron ``[(B*N), T]`` rows; returns N for unflatten."""
        data = get_data_matrix(x).float() * x.pad_and_gap_mask.float()
        n = data.shape[-1]
        return rearrange(data, "b t n -> (b n) t"), n

    def encode(self, x: BinnedInputType) -> Tensor:
        flat, n = self._to_flat(x)
        if self.training and not bool(self.fitted.item()):
            self._accumulate_running_stats(flat)
        if bool(self.fitted.item()):
            z = (flat - self.mu) @ self.W  # [(B*N), k]
        else:
            z = flat.new_zeros(flat.shape[0], self.spec.n_components)
        # [(B*N), k] -> [B, 1, N, k]
        return rearrange(z, "(b n) k -> b 1 n k", n=n)

    def decode(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 4 or tokens.shape[1] != 1:
            raise ValueError(f"expected latent [B, 1, N, k], got {tuple(tokens.shape)}")
        n = tokens.shape[2]
        z = rearrange(tokens, "b 1 n k -> (b n) k")
        if bool(self.fitted.item()):
            recon = z @ self.W.T + self.mu  # [(B*N), T]
        else:
            recon = self.mu.expand(z.shape[0], -1)
        # Linear PCA output is approximately non-negative because μ is positive
        # and the centered projection deviations are zero-mean. Clamp negatives
        # at ε rather than softplus, which would floor at log(2) ≈ 0.69 — wildly
        # above the typical spike rate (~0.013 spk/bin) for 1 ms binning.
        recon = recon.clamp(min=_RATE_FLOOR)
        return rearrange(recon, "(b n) t -> b t n", n=n)

    def forward(self, x: BinnedInputType) -> TokenizerOutput:
        z = self.encode(x)
        recon = self.decode(z)
        # The PCA path is non-differentiable (frozen buffers). Thread the
        # dummy parameter through with a zero coefficient so Lightning's
        # backward pass has a grad_fn to walk; output values are unchanged.
        recon = recon + 0.0 * self._dummy.sum()
        return TokenizerOutput(tokens=None, reconstruction=recon, aux_loss=None, latents=z)

    def reconstruct(self, tokens: Tensor) -> ReconstructionOutput:
        rates = self.decode(tokens)
        if self.training or self.spec.output_type == DataType.RATES:
            output = rates
            data_type = self.spec.output_type
        else:
            output = torch.poisson(rates).clamp(min=0).long()
            data_type = DataType.SPIKES
        bin_size_seconds = (
            self.spec.bin_size_ms / 1000.0 if self.spec.bin_size_ms is not False else None
        )
        return ReconstructionOutput(
            reconstruction=output,
            data_type=data_type,
            bin_size_seconds=bin_size_seconds,
        )
