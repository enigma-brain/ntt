"""Frequency-based look-up table tokenizer."""

import warnings
from typing import Any, Literal

import numpy as np
import pydantic as pdt
import torch
import torch.nn as nn

from ntt.core.components.types import (
    BinnedSpikesInput,
    DataType,
    LatentType,
    RatesInput,
    SpikeTimesInput,
    TokenizerOutput,
)
from ntt.core.model.tokenizer.base import BaseTokenizer, TokenizerSpec, register_tokenizer


class FrequencyBasedLUTTokenizerSpec(TokenizerSpec, frozen=True):
    """Specification for the Frequency-based Lookup Table tokenizer.

    This tokenizer assigns discrete tokens to neural activity patterns based on
    their frequency of occurrence in the training data. The most frequent patterns
    get a token in the vocabulary. The remaining patterns are assigned to the
    nearest token in the vocabulary based on cosine similarity.

    Training integrates with the standard NTT pipeline: pattern frequencies are
    accumulated during the first training epoch via encode(), and the vocabulary
    is built by NTTLitModule.on_train_epoch_end calling fit_from_running_stats().
    """

    KIND: Literal["frequency_based_lut"] = "frequency_based_lut"

    # Algorithm-specific parameters
    vocab_size: int = 65536

    # Fixed architecture constraints
    n_neurons: int = 1
    window_length_seconds: float = 1.0
    bin_size_ms: float = 1.0

    # Capability metadata
    latent_type: LatentType = LatentType.DISCRETE
    compress_n: bool = False  # Overridden dynamically by validator
    compress_t: bool = True
    variable_n: bool = True  # Tiles independently over each neuron
    variable_t: bool = True
    variable_dt: bool = False
    input_type: DataType = DataType.SPIKES
    output_type: DataType = DataType.SPIKES

    # Latent dimension semantics
    has_time_dim: bool = True
    has_unit_dim: bool = True
    has_latent_dim: bool = False  # encode() returns one scalar ID per token

    @pdt.model_validator(mode="before")
    @classmethod
    def _set_derived_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            n = data.get("n_neurons", 1)
            data.setdefault("compress_n", n > 1)
            # Required: base validator raises for compress_t=True + has_time_dim=True
            # without an explicit token_bin_size_ms.
            wls = data.get("window_length_seconds", 1.0)
            data.setdefault("token_bin_size_ms", float(wls) * 1000.0)
        return data

    def get_vocab_size(self) -> int:
        """Return the vocabulary size (number of discrete tokens)."""
        return self.vocab_size

    def get_codes_per_token(self) -> int:
        """One pattern ID per token (``encode()`` returns ``[B, T', U', 1]``)."""
        return 1


@register_tokenizer
class FrequencyBasedLUTTokenizer(BaseTokenizer):
    """Frequency-based lookup table tokenizer.

    Exhaustively enumerates patterns in the training data, tracking frequencies.
    The top ``vocab_size`` patterns are assigned unique tokens; all other patterns
    are matched to the nearest vocab entry by cosine similarity.

    Training protocol (standard NTT pipeline):
        Run ``ntt train experiment=frequency_based_lut/<config>``.
        During epoch 1, every training forward pass calls ``_accumulate_frequencies``
        to count patterns. At epoch end, ``NTTLitModule.on_train_epoch_end`` calls
        ``fit_from_running_stats()`` to build the vocabulary. Reconstructions during
        epoch 1 are zero (pre-fit placeholder); the model has no learnable parameters.

    Note: cosine-similarity nearest-neighbour lookup over the full vocabulary is
    O(N x vocab_size x pattern_dim) per batch. At vocab_size=262144 and 1ms bins
    this is expensive; consider chunked matmul or ANN search for large deployments.
    """

    spec_class = FrequencyBasedLUTTokenizerSpec
    spec: FrequencyBasedLUTTokenizerSpec
    vocab_patterns: torch.Tensor
    is_fit: torch.Tensor

    def __init__(self, spec: FrequencyBasedLUTTokenizerSpec):
        super().__init__(spec)
        self.vocab_size = spec.vocab_size

        # pattern_dim = window_length_bins * n_neurons
        self.pattern_dim = spec.window_length_bins * spec.n_neurons

        # Vocabulary: [vocab_size, pattern_dim]
        self.register_buffer("vocab_patterns", torch.zeros(self.vocab_size, self.pattern_dim))
        self.register_buffer("is_fit", torch.tensor(False))

        # Frequency accumulator: bytes -> int (cleared after fit_from_running_stats)
        self._pattern_counts: dict[bytes, int] = {}

        # Lightning requires at least one nn.Parameter to construct an optimizer.
        # This tokenizer has no learnable weights; the dummy is zeroed out in forward.
        self._dummy = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def _accumulate_frequencies(self, spike_matrix: torch.Tensor) -> None:
        """Count pattern occurrences from one batch. No-op once fitted."""
        if self.is_fit.item():
            return
        B, T, U = spike_matrix.shape
        T_b = self.spec.window_length_bins
        U_b = self.spec.n_neurons

        if T % T_b != 0 or U % U_b != 0:
            raise ValueError(
                f"Spike matrix dimensions ({T}, {U}) must be divisible by "
                f"block dimensions ({T_b}, {U_b})."
            )

        n_T = T // T_b
        n_U = U // U_b
        blocks = spike_matrix.view(B, n_T, T_b, n_U, U_b)
        blocks = blocks.permute(0, 1, 3, 2, 4).contiguous()
        flat_x = blocks.view(-1, T_b * U_b).cpu().long()

        for pattern in flat_x:
            key = pattern.numpy().tobytes()
            self._pattern_counts[key] = self._pattern_counts.get(key, 0) + 1

    def fit_from_running_stats(self) -> bool:
        """Build vocabulary from accumulated frequency counts.

        Idempotent: returns False (no-op) if already fitted.
        Called automatically by ``NTTLitModule.on_train_epoch_end`` after the
        first training epoch.

        Returns:
            True when the vocabulary is built for the first time, False otherwise.
        """
        if self.is_fit.item():
            return False
        if not self._pattern_counts:
            return False

        most_common = sorted(self._pattern_counts.items(), key=lambda kv: kv[1], reverse=True)
        top_patterns = most_common[: self.vocab_size]

        if len(top_patterns) < self.vocab_size:
            warnings.warn(
                f"{type(self).__name__}: only {len(top_patterns)} unique patterns "
                f"found; vocab_size clipped from {self.vocab_size} to "
                f"{len(top_patterns)}. Unfilled slots remain as zero vectors.",
                UserWarning,
                stacklevel=2,
            )

        for i, (pat_bytes, _) in enumerate(top_patterns):
            arr = np.frombuffer(pat_bytes, dtype=np.int64).copy()
            self.vocab_patterns[i] = torch.from_numpy(arr).float().to(self.vocab_patterns.device)

        self.is_fit.fill_(True)
        self._pattern_counts = {}  # free memory
        return True

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def encode(
        self, x: SpikeTimesInput | BinnedSpikesInput | RatesInput | torch.Tensor
    ) -> torch.Tensor:
        """Encode neural activity into vocabulary token indices.

        During training before ``fit_from_running_stats`` has been called,
        accumulates frequency counts and returns zero-filled tokens as a
        placeholder (identical to PCABaselineTokenizer's pre-fit behaviour).

        Args:
            x: Input spikes, shape ``[B, T, U]``.

        Returns:
            Token indices, shape ``[B, T', U', 1]``, dtype ``torch.long``.
            ``T' = T // window_length_bins``, ``U' = U // n_neurons``.
        """
        if hasattr(x, "spike_matrix"):
            spike_matrix = x.spike_matrix.float()  # type: ignore[union-attr]
        elif hasattr(x, "rate_matrix"):
            spike_matrix = x.rate_matrix.float()  # type: ignore[union-attr]
        else:
            spike_matrix = x.float()  # type: ignore[union-attr]

        B, T, U = spike_matrix.shape
        T_b = self.spec.window_length_bins
        U_b = self.spec.n_neurons

        if T % T_b != 0 or U % U_b != 0:
            raise ValueError(
                f"Spike matrix dimensions ({T}, {U}) must be divisible by "
                f"block dimensions ({T_b}, {U_b})."
            )

        n_T = T // T_b
        n_U = U // U_b

        # Accumulate counts during training before vocab is built
        if self.training and not self.is_fit.item():
            self._accumulate_frequencies(spike_matrix)

        # Return placeholder zeros before vocabulary is finalised
        if not self.is_fit.item():
            return torch.zeros(B, n_T, n_U, 1, dtype=torch.long, device=spike_matrix.device)

        blocks = spike_matrix.view(B, n_T, T_b, n_U, U_b)
        blocks = blocks.permute(0, 1, 3, 2, 4).contiguous()
        flat_x = blocks.view(-1, T_b * U_b)

        # Guard against zero-norm vectors (silent periods) before normalising.
        # F.normalize of a zero vector produces zeros, making cosine similarity
        # undefined; clamping the norm gives deterministic nearest-neighbour
        # assignment instead of NaN.
        flat_norms = flat_x.norm(dim=1, keepdim=True).clamp(min=1e-8)
        flat_x_norm = flat_x / flat_norms
        vocab_norms = self.vocab_patterns.norm(dim=1, keepdim=True).clamp(min=1e-8)
        vocab_norm = self.vocab_patterns / vocab_norms

        cos_sim = torch.mm(flat_x_norm, vocab_norm.t())
        tokens = torch.argmax(cos_sim, dim=1)  # dtype: torch.long

        return tokens.view(B, n_T, n_U, 1)

    def tokens_to_latents(self, tokens: torch.Tensor) -> torch.Tensor:
        """Look up floating-point vocabulary patterns, shape ``[B, T', U', D]``."""
        if tokens.ndim == 4 and tokens.shape[-1] == 1:
            tokens = tokens.squeeze(-1)
        if tokens.ndim != 3:
            raise ValueError("expected token indices [B, T', U'] or [B, T', U', 1]")
        if tokens.dtype not in (torch.int32, torch.int64):
            raise TypeError("expected integer token indices")
        return self.vocab_patterns[tokens]

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode token indices or continuous patterns into spike reconstructions.

        Args:
            tokens: Integer token indices, shape ``[B, T', U', 1]`` or
                ``[B, T', U']``, or floating-point patterns ``[B, T', U', D]``.

        Returns:
            Reconstructed spikes, shape ``[B, T, U]``. Negative values from
            latent perturbations are clamped to zero to keep spike counts valid.
        """
        patterns = tokens if tokens.is_floating_point() else self.tokens_to_latents(tokens)
        if patterns.ndim != 4 or patterns.shape[-1] != self.pattern_dim:
            raise ValueError(
                f"expected patterns [B, T', U', {self.pattern_dim}], got {tuple(patterns.shape)}"
            )
        B, n_T, n_U, _ = patterns.shape

        T_b = self.spec.window_length_bins
        U_b = self.spec.n_neurons

        blocks = patterns.reshape(B, n_T, n_U, T_b, U_b)
        blocks = blocks.permute(0, 1, 3, 2, 4).contiguous()
        return blocks.view(B, n_T * T_b, n_U * U_b).clamp_min(0)

    def forward(
        self, x: SpikeTimesInput | BinnedSpikesInput | RatesInput | torch.Tensor
    ) -> TokenizerOutput:
        tokens = self.encode(x)
        reconstruction = self.decode(tokens)
        # Thread dummy parameter through with a zero coefficient so Lightning's
        # backward pass has a grad_fn to walk; output values are unchanged.
        reconstruction = reconstruction + 0.0 * self._dummy.sum()
        return TokenizerOutput(tokens=tokens, reconstruction=reconstruction, aux_loss=None)
