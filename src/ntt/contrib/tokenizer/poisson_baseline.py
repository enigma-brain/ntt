"""Poisson baseline tokenizer.

A non-trainable tokenizer that estimates firing rates from spike data and
generates Poisson spike reconstructions.  Serves as a natural baseline for
evaluating learned tokenizers: any competent model should beat the Poisson
baseline, which preserves rate modulation but destroys fine temporal
structure (ISI patterns, bursts, correlations).

The ``rate_estimation_bin_ms`` parameter controls the temporal resolution of
the rate estimate.  Smaller values preserve more temporal detail; larger
values average over longer windows:

- 1 ms  ≈ near-perfect copy (rate bin = data bin)
- 10 ms ≈ reasonable default
- 100+ ms ≈ only preserves slow rate modulation
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pydantic as pdt
import torch

from ntt.core.components.types import (
    BinnedInputType,
    ReconstructionOutput,
    TokenizerOutput,
    get_data_matrix,
)
from ntt.core.model.tokenizer.base import (
    BaseTokenizer,
    DataType,
    LatentType,
    TokenizerSpec,
    register_tokenizer,
)


class PoissonBaselineTokenizerSpec(TokenizerSpec, frozen=True):
    """Spec for the Poisson baseline tokenizer."""

    KIND: Literal["poisson_baseline"] = "poisson_baseline"

    rate_estimation_bin_ms: float = 10.0
    seed: int = 42

    # Inherited from TokenizerSpec (must be set)
    window_length_seconds: float = 1.0
    bin_size_ms: float = 1.0
    n_neurons: int | Literal[False] = False

    # Capability metadata
    latent_type: LatentType = LatentType.CONTINUOUS
    compress_n: bool = False
    compress_t: bool = True
    variable_n: bool = True
    variable_t: bool = True
    variable_dt: bool = False
    input_type: DataType = DataType.SPIKES
    output_type: DataType = DataType.SPIKES

    # Latent is [B, T', U] — multiple rate bins over time, per-neuron
    has_time_dim: bool = True
    has_unit_dim: bool = True
    has_latent_dim: bool = False

    @pdt.model_validator(mode="before")
    @classmethod
    def _set_token_bin_size_from_rate(cls, data: dict) -> dict:
        """Set token_bin_size_ms = rate_estimation_bin_ms before base validator."""
        if isinstance(data, dict) and "token_bin_size_ms" not in data:
            if "rate_estimation_bin_ms" in data:
                data["token_bin_size_ms"] = float(data["rate_estimation_bin_ms"])
            else:
                data["token_bin_size_ms"] = float(
                    cls.model_fields["rate_estimation_bin_ms"].default
                )
        return data

    def get_vocab_size(self) -> int:
        raise NotImplementedError("Continuous latent — no discrete vocabulary.")


def format_poisson_tokenizer_display_name(spec: PoissonBaselineTokenizerSpec) -> str:
    """Human-readable label for plots and tables, e.g. ``Poisson 33ms``."""
    return f"Poisson {float(spec.rate_estimation_bin_ms):g}ms"


def _format_rate_bin_ms(rate_estimation_bin_ms: float) -> str:
    """Format ``rate_estimation_bin_ms`` for use in a snake_case label.

    Integer-valued floats are rendered without a decimal (``10.0 -> '10'``);
    non-integers get one decimal place (``12.5 -> '12.5'``). Used by
    :func:`format_poisson_label`.
    """
    f = float(rate_estimation_bin_ms)
    if f.is_integer():
        return str(int(round(f)))
    return f"{f:.1f}"


def format_poisson_label(spec: PoissonBaselineTokenizerSpec) -> str:
    """Snake-case Poisson label embedding ``rate_estimation_bin_ms``.

    Produces ``poisson_<N>ms`` (e.g. ``poisson_10ms``, ``poisson_12.5ms``)
    so plot legends / titles disambiguate between Poisson variants run
    at different rate-estimation bin sizes.
    """
    return f"poisson_{_format_rate_bin_ms(spec.rate_estimation_bin_ms)}ms"


@register_tokenizer
class PoissonBaselineTokenizer(BaseTokenizer):
    """Rate-matched Poisson baseline implemented as a tokenizer.

    ``encode`` estimates firing rates at ``rate_estimation_bin_ms`` resolution.
    ``decode`` expands rates back to fine-bin expected counts.
    ``reconstruct`` samples from Poisson to produce integer spike counts.

    There are **no learnable parameters**.
    """

    spec_class = PoissonBaselineTokenizerSpec
    spec: PoissonBaselineTokenizerSpec

    def __init__(self, spec: PoissonBaselineTokenizerSpec) -> None:
        super().__init__(spec)
        bin_size_ms = spec.bin_size_ms
        if bin_size_ms is False:
            raise ValueError("PoissonBaselineTokenizer requires a fixed bin_size_ms.")
        self._bin_size_ms = float(bin_size_ms)
        rate_ms = spec.rate_estimation_bin_ms
        ratio = rate_ms / self._bin_size_ms
        if not np.isclose(ratio, round(ratio)):
            raise ValueError(
                f"rate_estimation_bin_ms ({rate_ms}) must be an integer "
                f"multiple of bin_size_ms ({self._bin_size_ms})."
            )
        self._ratio = int(round(ratio))
        self._seed = spec.seed
        # One random stream per device, seeded once. An eval calls reconstruct() once per
        # batch; re-seeding on every call reused the same draws for every batch, so the
        # sampling noise added up across batches instead of averaging out.
        self._generators: dict[torch.device, torch.Generator] = {}

    def _match_input_length(
        self,
        reconstruction: torch.Tensor,
        target_time_bins: int,
    ) -> torch.Tensor:
        """Trim or zero-pad reconstructions to the original input length.

        Only small boundary remainders are reconciled: the Poisson baseline
        coarsens to ``rate_estimation_bin_ms`` and can lose up to one such bin
        of trailing samples (e.g. 997 input bins -> 990 recon bins); trimming
        or padding that remainder keeps input-robustness ``window_shift`` from
        skipping whole specs. A mismatch larger than one rate-estimation bin
        signals a real bug, so we raise rather than silently fabricate data.
        The padded/trimmed samples are never consumed by the input-robustness
        metrics (the latent-side metrics use latents; the input-side
        comparison uses the raw originals).
        """
        current_time_bins = reconstruction.shape[1]
        if current_time_bins == target_time_bins:
            return reconstruction
        tol_bins = max(1, round(self.spec.rate_estimation_bin_ms / self._bin_size_ms))
        if abs(current_time_bins - target_time_bins) > tol_bins:
            raise ValueError(
                f"Poisson reconstruction length {current_time_bins} differs from "
                f"input length {target_time_bins} by more than one rate-estimation "
                f"bin ({tol_bins} time bins); refusing to pad/trim."
            )
        if current_time_bins > target_time_bins:
            return reconstruction[:, :target_time_bins, :]

        pad = reconstruction.new_zeros(
            reconstruction.shape[0],
            target_time_bins - current_time_bins,
            reconstruction.shape[2],
        )
        return torch.cat([reconstruction, pad], dim=1)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def encode(self, x: BinnedInputType) -> torch.Tensor:
        """Estimate firing rates at coarse resolution.

        Args:
            x: BinnedSpikesInput or RatesInput with data matrix shape ``[batch, time, units]``.

        Returns:
            Rate profile in Hz, shape ``[batch, n_rate_bins, units]``.
        """
        spikes = get_data_matrix(x).float()  # [B, T, U]
        B, T, U = spikes.shape
        n_rate_bins = T // self._ratio
        truncated = spikes[:, : n_rate_bins * self._ratio, :]  # [B, n*r, U]
        reshaped = truncated.reshape(B, n_rate_bins, self._ratio, U)
        counts_per_rate_bin = reshaped.sum(dim=2)  # [B, n_rate_bins, U]
        rate_bin_sec = self.spec.rate_estimation_bin_ms / 1000.0
        rates_hz = counts_per_rate_bin / rate_bin_sec
        return rates_hz.unsqueeze(-1)  # [B, T', U, 1]

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Expand rate profile to fine-bin expected counts.

        Args:
            tokens: Rates in Hz, shape ``[batch, n_rate_bins, units, 1]``.

        Returns:
            Expected spike counts per fine bin, shape ``[batch, T, units]``.
        """
        # Strip singleton D dim if present: [B, T', U, 1] -> [B, T', U]
        if tokens.ndim == 4 and tokens.shape[-1] == 1:
            tokens = tokens.squeeze(-1)
        bin_size_sec = self._bin_size_ms / 1000.0
        expected_counts = tokens * bin_size_sec  # [B, n_rate_bins, U]
        # Repeat each rate estimate across the fine bins it covers
        expanded = expected_counts.unsqueeze(2).expand(-1, -1, self._ratio, -1)
        return expanded.reshape(tokens.shape[0], -1, tokens.shape[2])

    def forward(self, x: BinnedInputType) -> TokenizerOutput:
        z = self.encode(x)
        reconstruction = self.decode(z)
        return TokenizerOutput(tokens=z, reconstruction=reconstruction, aux_loss=None)

    def reconstruct(self, tokens: torch.Tensor) -> ReconstructionOutput:
        """Decode + Poisson-sample to produce integer spike counts."""
        expected = self.decode(tokens)
        gen = self._generators.get(expected.device)
        if gen is None:
            gen = torch.Generator(device=expected.device)
            gen.manual_seed(self._seed)
            self._generators[expected.device] = gen
        sampled = torch.poisson(expected.clamp(min=0), generator=gen)
        return ReconstructionOutput(
            reconstruction=sampled,
            data_type=self.spec.output_type,
            bin_size_seconds=self._bin_size_ms / 1000.0,
        )

    def reconstruct_from_input(
        self,
        x: BinnedInputType,
        return_latents: bool = False,
        return_latent_timestamps: bool = True,
    ) -> ReconstructionOutput | dict:
        """Reconstruct while preserving the input time length.

        Small boundary remainders are reconciled by ``_match_input_length``
        (see there).
        """
        result = super().reconstruct_from_input(
            x,
            return_latents=return_latents,
            return_latent_timestamps=return_latent_timestamps,
        )
        target_time_bins = get_data_matrix(x).shape[1]
        if return_latents:
            assert isinstance(result, dict)
            out = dict(result)
            out["reconstruction"] = self._match_input_length(
                out["reconstruction"],
                target_time_bins,
            )
            return out
        assert not isinstance(result, dict)
        return result._replace(
            reconstruction=self._match_input_length(result.reconstruction, target_time_bins)
        )


# ------------------------------------------------------------------
# Standalone numpy helper (used by Lightning callback)
# ------------------------------------------------------------------


def poisson_baseline_reconstruction(
    spike_counts: np.ndarray,
    bin_size_ms: float,
    rate_estimation_bin_ms: float,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a Poisson baseline reconstruction from binned spike data.

    Operates on numpy arrays in ``(units, time)`` format — the evaluation
    convention.  This avoids the overhead of constructing a full
    ``BinnedSpikesInput`` in contexts like the Lightning callback.

    Args:
        spike_counts: Spike count array, shape ``(units, time)``.
        bin_size_ms: Bin size of the input data in milliseconds.
        rate_estimation_bin_ms: Rate estimation window in milliseconds.
        seed: Random seed for reproducibility.

    Returns:
        Poisson-sampled spike counts, same shape as input.
    """
    ratio = rate_estimation_bin_ms / bin_size_ms
    if not np.isclose(ratio, round(ratio)):
        raise ValueError(
            f"rate_estimation_bin_ms ({rate_estimation_bin_ms}) must be an "
            f"integer multiple of bin_size_ms ({bin_size_ms})."
        )
    ratio_int = int(round(ratio))
    n_units, T = spike_counts.shape
    n_rate_bins = T // ratio_int

    # Estimate rates
    truncated = spike_counts[:, : n_rate_bins * ratio_int]
    reshaped = truncated.reshape(n_units, n_rate_bins, ratio_int)
    counts_per_rate_bin = reshaped.sum(axis=2)
    rate_bin_sec = rate_estimation_bin_ms / 1000.0
    rates_hz = counts_per_rate_bin / rate_bin_sec

    # Expected counts per fine bin
    bin_size_sec = bin_size_ms / 1000.0
    expected_per_fine_bin = rates_hz * bin_size_sec  # (units, n_rate_bins)

    # Expand and sample
    expanded = np.repeat(expected_per_fine_bin, ratio_int, axis=1)  # (units, n_rate_bins*ratio)
    rng = np.random.default_rng(seed)
    sampled = rng.poisson(np.clip(expanded, 0, None)).astype(spike_counts.dtype)

    # Pad back to original length if truncated
    if sampled.shape[1] < T:
        pad_width = T - sampled.shape[1]
        sampled = np.pad(sampled, ((0, 0), (0, pad_width)), mode="constant")

    return sampled
