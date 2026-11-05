"""Compression metrics — general properties of tokenizers.

Unlike most metrics that compare original vs reconstructed data at a given
stimulus alignment, compression metrics report intrinsic properties of the
tokenizer's output representation (e.g. tokens per second, dims per token).
They require no stimulus alignment and apply to any validation segment.
"""

from typing import Any

import numpy as np

from ..core.types import DataType, MetricCategory, MetricDirection, MetricLevel, MetricResult
from .base import BaseMetric, register_metric


@register_metric
class TokenDimsPerSecond(BaseMetric):
    """Number of latent scalar dimensions a tokenizer emits per second.

    Quantifies the information-carrying *rate* of a tokenizer's representation:

        token_dims_per_second = n_token_dims_per_token / token_bin_size_seconds

    where ``n_token_dims_per_token`` is the total number of scalar latent
    dimensions emitted per token (``latent_dim`` for population tokenizers,
    ``U * latent_dim`` for per-neuron tokenizers with ``U`` units), and
    ``token_bin_size_seconds`` is the temporal spacing between consecutive
    tokens.

    **Direction.** ``LOWER_IS_BETTER``: higher compression means fewer tokens
    and/or fewer dims per token per second of neural signal. Note this is only
    one side of the rate-distortion trade-off — a tokenizer that throws away
    all information also scores very well here, so this metric should always
    be read together with reconstruction-quality metrics.

    **Source of bin size.** If the tokenizer advertises ``token_bin_size_ms``
    (stored on the IntermediatesCache), that authoritative declared value is
    used. Otherwise it is inferred from the median diff of
    ``token_timestamps``. The ``source`` metadata field records which path
    was taken ("declared" or "inferred").

    **No assumption on stimulus alignment.** This metric is a general
    property of the tokenizer output and applies to any data segment.
    """

    name = "token_dims_per_second"
    category = MetricCategory.LATENT_QUALITY
    direction = MetricDirection.LOWER_IS_BETTER
    level = MetricLevel.POPULATION
    value_range: tuple[float | None, float | None] = (0.0, None)
    _required_intermediates = ["latents", "token_bin_size_ms", "token_timestamps"]
    _default_params: dict[str, Any] = {}
    aggregation_fallback_policy = "per_window"

    # Works regardless of input data representation — only looks at latents.
    supported_data_types: set[DataType] = {
        DataType.SPIKE_TIMES,
        DataType.SPIKES,
        DataType.RATES,
    }
    preferred_data_type: DataType = DataType.SPIKES

    def get_required_intermediates(self, data_type: DataType) -> list[str]:
        """Compression metric only needs latents and bin size info.

        ``token_timestamps`` is only used as a fallback when
        ``token_bin_size_ms`` is not declared; we still list it as required so
        the cache provides it when available.
        """
        return list(self._required_intermediates)

    def compute(self, intermediates: dict[str, Any]) -> MetricResult:
        """Compute tokens-per-second compression from latents and bin size."""
        latents = intermediates.get("latents")
        if latents is None:
            raise ValueError("TokenDimsPerSecond requires 'latents' in intermediates, got None.")

        # --- Determine dims-per-token from latent shape ---
        # Shape conventions (see _compute_latent_windows in intermediates.py):
        #   Population tokenizer: [N_windows, T_tokens, D_latent]
        #   Per-neuron tokenizer: [N_windows, T_tokens, U, D_latent]
        #                         or [N_windows, 1, U, D_latent] (squeezable)
        # We compute dims-per-token = product of all dims after (N_windows, T).
        # For a list of per-window arrays (unequal token counts), use the first.
        if isinstance(latents, list):
            if len(latents) == 0:
                raise ValueError("TokenDimsPerSecond: empty latents list.")
            first = np.asarray(latents[0])
        else:
            first = np.asarray(latents)

        if first.ndim < 2:
            raise ValueError(
                f"TokenDimsPerSecond: latents must have at least 2 dims "
                f"(N_windows, ..., D_latent); got shape {first.shape}."
            )

        # For the list case, ``first`` is already a single window: its dims are
        # (T_tokens, ..., D_latent) — no leading N_windows axis. For the array
        # case, ``first`` still has the N_windows axis as dim 0.
        if isinstance(latents, list):
            # Per-window array: (T_tokens, D) or (T_tokens, U, D) etc.
            per_token_shape = first.shape[1:]
            is_per_neuron = first.ndim == 3  # (T, U, D)
        else:
            # Stacked array: (N_windows, T_tokens, ...).
            # Per-token dims = everything after axis 1 (time).
            per_token_shape = first.shape[2:]
            is_per_neuron = first.ndim == 4  # (N, T, U, D)

        if len(per_token_shape) == 0:
            raise ValueError(
                f"TokenDimsPerSecond: could not infer per-token dims from "
                f"latent shape {first.shape}."
            )

        n_token_dims_per_token = int(np.prod(per_token_shape))
        latent_dim = int(per_token_shape[-1])  # innermost dim

        # --- Determine token bin size ---
        declared_bin_ms = intermediates.get("token_bin_size_ms")
        token_timestamps = intermediates.get("token_timestamps")

        if declared_bin_ms is not None:
            token_bin_size_ms = float(declared_bin_ms)
            source = "declared"
        elif token_timestamps is not None:
            ts = np.asarray(token_timestamps)
            if ts.ndim == 0 or ts.size < 2:
                raise ValueError(
                    "TokenDimsPerSecond: cannot infer token bin size — "
                    f"token_timestamps has too few entries (size={ts.size})."
                )
            # Handle possible [N_windows, T] or flat [T] layouts: flatten to 1D.
            ts_flat = ts.reshape(-1)
            diffs = np.diff(ts_flat)
            # Only keep positive diffs (avoid window boundaries that reset time).
            pos = diffs[diffs > 0]
            if pos.size == 0:
                raise ValueError(
                    "TokenDimsPerSecond: cannot infer token bin size — "
                    "no positive diffs in token_timestamps."
                )
            bin_size_seconds = float(np.median(pos))
            token_bin_size_ms = bin_size_seconds * 1000.0
            source = "inferred"
        else:
            raise ValueError(
                "TokenDimsPerSecond requires either 'token_bin_size_ms' "
                "(declared by the tokenizer) or 'token_timestamps' "
                "(for empirical inference); neither was provided."
            )

        token_bin_size_seconds = token_bin_size_ms / 1000.0
        if token_bin_size_seconds <= 0:
            raise ValueError(
                f"TokenDimsPerSecond: non-positive token_bin_size_seconds "
                f"({token_bin_size_seconds})."
            )

        token_dims_per_second = n_token_dims_per_token / token_bin_size_seconds

        metadata: dict[str, Any] = {
            "token_bin_size_ms": token_bin_size_ms,
            "latent_dim": latent_dim,
            "n_token_dims_per_token": n_token_dims_per_token,
            "source": source,
            "is_per_neuron": is_per_neuron,
            "per_token_shape": list(per_token_shape),
        }

        return self._make_result(
            raw_value=float(token_dims_per_second),
            metadata=metadata,
        )
