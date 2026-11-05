"""Lazily computes and caches intermediate results shared across metrics.

NOT compatible with DataType.RATES output:
- spike_times_*: Requires spike_times field
- isi_*: Requires spike times for inter-spike intervals
- spike_matches: Requires spike times for spike matching

Compatible with RATES (via binned counts or direct computation):
- all_correlograms_*, acg_*, ccg_*: Uses spike-times algorithm when available,
  falls back to discrete cross-correlation of binned counts for RATES/coarse SPIKES.
- native_binned_counts_*: Native-resolution count-valued representation.
- binned_counts_*: Works via _get_binned_counts_base() which handles rates
- population_vectors_*: Works via _compute_population_vectors_from_binned()
- firing_rates_*: Works via _compute_firing_rates()
- duration: Works by inferring from binned data
"""

from __future__ import annotations

import copy
import logging
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np

from .types import DataType, EvalData

if TYPE_CHECKING:
    from ..metrics.base import IntermediateRequest

ACG_SAMPLING_RATE = 1000

# FFT-vs-Phy correlogram dispatch tunables (see ``_use_fft_correlograms``).
# The Phy-compat FFT path is bit-identical to the all-pairs Phy kernel
# (verified: max|Δ| = 0), so this choice is purely about wall-clock. The FFT
# path costs ~O(N_transforms · T_1ms·log T_1ms) — fixed by the *trace length*,
# independent of lag or spike density. The all-pairs Phy kernel costs
# ~O(N_spikes² · max_lag / duration) — grows with lag and density. Crossover:
#   * Short traces (windowed latent_robustness (M2), ~1 s windows): FFT is cheap in absolute terms
#     and wins or ties at every resolution → always FFT.
#   * Long traces (non-windowed full sessions): FFT pays a large fixed
#     transform cost, so it only pays off when Phy's lag work is large, i.e.
#     when ``max_lag_ms`` is wide. At the default 1 ms / 100 ms it is a ~1.4×
#     CCG regression, so fall back to the shared all-pairs Phy matrix there.
_FFT_SHORT_TRACE_MAX_1MS_BINS = (
    8192  # ~8.2 s; covers latent_robustness (M2) eval windows (default 1 s)
)
_FFT_LONG_TRACE_MIN_LAG_MS = 250.0  # above this, Phy pair-counting dominates the FFT cost

# Keys stored directly in ``_cache`` at init time (not side-keyed by
# ``_original`` / ``_reconstructed``). Derived by inspecting
# ``IntermediatesCache.__init__``. These are preserved when swapping the
# reconstructed side via ``replace_reconstructed``.
_PASSTHROUGH_KEYS: set[str] = {
    "latents",
    "behavioral_data_token_res",
    "behavioral_data_input_res",
    "token_timestamps",
    "sampling_rate",
    "segment_absolute_start",
    "token_bin_size_ms",
    # Branch (C) replica-fan-out passthrough (consumed by input_robustness (M1) only).
    "replica_latents",
    "replica_reconstructions",
    "replica_originals",
    "replica_metadata",
}

# ── Warn-once flags ──────────────────────────────────────────────────
# These prevent the same resolution-mismatch warnings from flooding the
# log on every segment / realisation.  Reset to False if you need them
# to fire again (e.g. in a notebook across experiments).
_warned_bin_size_skip: bool = False
_warned_correlogram_resolution: bool = False
logger = logging.getLogger(__name__)


class IntermediatesCache:
    """
    Lazily computes and caches intermediate results shared across metrics.

    Each intermediate is computed once on first access and then cached.
    The cache holds results for comparing original vs reconstructed spike trains.
    No batching supported, one cache per "sample" (a set of units across time).

    All access is via ``IntermediateRequest`` objects.  Parameters are carried
    by the request (populated by each metric's ``get_intermediate_requests()``).
    Two requests with the same key and params share a single cache entry.
    """

    def __init__(
        self,
        original: EvalData | None,
        reconstructed: EvalData | None,
        latents: np.ndarray | list[np.ndarray] | None = None,
        behavioral_data_token_res: dict[str, np.ndarray] | None = None,
        behavioral_data_input_res: dict[str, np.ndarray] | None = None,
        token_timestamps: np.ndarray | None = None,
        sampling_rate: float | None = None,
        segment_absolute_start: float | None = None,
        token_bin_size_ms: float | None = None,
        replica_latents: np.ndarray | list[np.ndarray] | None = None,
        replica_reconstructions: np.ndarray | list[np.ndarray] | None = None,
        replica_originals: np.ndarray | list[np.ndarray] | None = None,
        replica_metadata: dict | None = None,
    ):
        """
        Parameters:
        -----------
            original, reconstructed:
                EvalData, with 2d spike array or list of lists of spike times.
                May be ``None`` on evaluation paths where input and
                reconstruction cannot be assembled at the segment level
                (e.g. interval-windowed SNR, where uneven windows lead to
                incomplete reconstruction). Any intermediate request that
                actually consults the missing side raises ``ValueError`` —
                the cache never substitutes placeholder data.
            latents:
                Optional latent representations from tokenizer (n_samples, latent_dim)
            behavioral_data_token_res:
                Categorical behavioral fields at token resolution (for filtering)
            behavioral_data_input_res:
                All behavioral fields at input resolution (for targets)
            token_timestamps:
                Optional timestamps for each token (n_samples,)
            sampling_rate:
                Input sampling rate in Hz (e.g. 1000 for 1ms bins)
            segment_absolute_start:
                Absolute start time of the segment (seconds)
            token_bin_size_ms:
                Temporal extent of each token in ms
            replica_latents, replica_reconstructions, replica_originals,
            replica_metadata:
                Optional replica-fan-out robustness passthroughs. Populated
                only when the upstream dataloader is a replica-fan-out
                loader; consumed by input_robustness (M1) only. Every other metric ignores
                them. ``replica_originals`` holds the *input* spike data
                BEFORE any tokenizer pass, retained for downstream
                consumers.
        """
        self.original = original
        self.reconstructed = reconstructed
        self._cache: dict[Any, Any] = {}

        # Store latents, behavioral data, and timestamps directly in cache if provided
        if latents is not None:
            self._cache["latents"] = latents
        if behavioral_data_token_res is not None:
            self._cache["behavioral_data_token_res"] = behavioral_data_token_res
        if behavioral_data_input_res is not None:
            self._cache["behavioral_data_input_res"] = behavioral_data_input_res
        if token_timestamps is not None:
            self._cache["token_timestamps"] = token_timestamps
        # NOTE: These are per-segment scalar metadata, not computed intermediates.
        # They are stored here because the intermediates cache is currently the
        # only mechanism to pass per-segment data to metric.compute(). If a
        # cleaner way to pass per-segment metadata to metrics is added later,
        # these should move there.
        if sampling_rate is not None:
            self._cache["sampling_rate"] = sampling_rate
        if segment_absolute_start is not None:
            self._cache["segment_absolute_start"] = segment_absolute_start
        if token_bin_size_ms is not None:
            self._cache["token_bin_size_ms"] = token_bin_size_ms
        # Branch (C) replica passthrough — used only by input_robustness (M1). Stored raw
        # (no conversion) so input_robustness (M1) can decide tensor-vs-ndarray handling.
        if replica_latents is not None:
            self._cache["replica_latents"] = replica_latents
        if replica_reconstructions is not None:
            self._cache["replica_reconstructions"] = replica_reconstructions
        if replica_originals is not None:
            self._cache["replica_originals"] = replica_originals
        if replica_metadata is not None:
            self._cache["replica_metadata"] = replica_metadata

    def replace_reconstructed(self, new_reconstructed: EvalData) -> IntermediatesCache:
        """Return a shallow-copied cache with a new reconstructed side.

        Keeps cached entries whose keys represent the original side
        (``*_original``) or are passthrough metadata (``_PASSTHROUGH_KEYS``),
        and drops everything that depends on the reconstructed side. Used
        by the robustness per-pair helper to reuse expensive original-side
        intermediates across many (clean, noisy_i) evaluations.

        The returned cache shares ``original`` with ``self``;
        ``self`` is not mutated.
        """
        new_cache = copy.copy(self)  # shallow: shares original
        new_cache.reconstructed = new_reconstructed

        # Keep only original-side intermediates and passthrough metadata.
        # Cache keys can be either strings (passthrough + side-keyed
        # computed entries) or ``IntermediateRequest`` dataclass instances
        # (parameterized entries); for the latter we inspect ``.key``.
        def _is_original_side(k: Any) -> bool:
            key_str = k if isinstance(k, str) else getattr(k, "key", None)
            if key_str is None:
                return False
            return key_str.endswith("_original") or key_str in _PASSTHROUGH_KEYS

        new_cache._cache = {k: v for k, v in self._cache.items() if _is_original_side(k)}
        return new_cache

    @staticmethod
    def _stack_uniform_window_list(values: list[np.ndarray] | list[Any]) -> np.ndarray | None:
        """Stack a list of per-window arrays when every window has the same shape."""
        if not values:
            return None

        arrays = [np.asarray(value) for value in values]
        first_shape = arrays[0].shape
        if any(array.shape != first_shape for array in arrays[1:]):
            return None
        return np.stack(arrays, axis=0)

    @staticmethod
    def _window_count_from_latents(latent: np.ndarray | list[np.ndarray]) -> int | None:
        """Infer the number of windows from cached latent storage."""
        if isinstance(latent, list):
            return len(latent) if latent else None

        latent_array = np.asarray(latent) if not isinstance(latent, np.ndarray) else latent
        if latent_array.ndim == 0:
            return None
        return int(latent_array.shape[0])

    def get(self, request: IntermediateRequest) -> Any:
        """Get an intermediate result, computing it if not cached.

        Args:
            request: An ``IntermediateRequest`` specifying the intermediate
                     key and its parameters.
        """
        if request not in self._cache:
            self._cache[request] = self._compute(request)
        return self._cache[request]

    def _require_original(self, key: str) -> EvalData:
        """Return ``self.original`` or raise if it was not provided.


        Used by ``_compute`` match arms whose intermediates depend on the
        input side. Fail-loud: if a metric reaches this on a path that did
        not assemble an original (e.g. interval-windowed SNR), raise
        instead of silently returning placeholder data.
        """
        if self.original is None:
            raise ValueError(
                f"Intermediate '{key}' requires `original` data, but it was "
                "not provided to IntermediatesCache. This evaluation path "
                "did not assemble an input trace (e.g. interval-windowed "
                "SNR), so any metric depending on the original side is "
                "unsupported here."
            )
        return self.original

    def _require_reconstructed(self, key: str) -> EvalData:
        """Return ``self.reconstructed`` or raise if it was not provided.


        Companion to ``_require_original``; same rationale.
        """
        if self.reconstructed is None:
            raise ValueError(
                f"Intermediate '{key}' requires `reconstructed` data, but "
                "it was not provided to IntermediatesCache. This evaluation "
                "path did not assemble a reconstruction trace (e.g. "
                "interval-windowed SNR), so any metric depending on the "
                "reconstructed side is unsupported here."
            )
        return self.reconstructed

    def _compute(self, request: IntermediateRequest) -> Any:
        """Compute an intermediate using parameters from the request."""

        params = request.params_dict
        key = request.key

        match key:
            # --- Passthrough intermediates (no params) ---
            case "spike_times_original":
                return self._require_spike_times(self._require_original(key))
            case "spike_times_reconstructed":
                return self._require_spike_times(self._require_reconstructed(key))
            case "duration":
                return self._get_duration(self._require_original(key))
            case "binned_spikes_original":
                orig = self._require_original(key)
                if orig.binned_spikes is None:
                    raise ValueError("Binned spikes required for this intermediate.")
                return orig.binned_spikes
            case "binned_spikes_reconstructed":
                recon = self._require_reconstructed(key)
                if recon.binned_spikes is None:
                    raise ValueError("Binned spikes required for this intermediate.")
                return recon.binned_spikes
            case "native_binned_counts_original":
                return self._get_binned_counts_base(self._require_original(key))[0]
            case "native_binned_counts_reconstructed":
                return self._get_binned_counts_base(self._require_reconstructed(key))[0]
            case "bin_size_original":
                orig = self._require_original(key)
                if orig.bin_size_seconds is None:
                    raise ValueError("Bin size required for this intermediate.")
                return orig.bin_size_seconds
            case "bin_size_reconstructed":
                recon = self._require_reconstructed(key)
                if recon.bin_size_seconds is None:
                    raise ValueError("Bin size required for this intermediate.")
                return recon.bin_size_seconds
            case "firing_rates_original":
                return self._compute_firing_rates(self._require_original(key))
            case "firing_rates_reconstructed":
                return self._compute_firing_rates(self._require_reconstructed(key))

            # --- Parameterized intermediates ---
            case "spike_matches":
                # spike_matches reads both sides — guard them both up front
                # so the failure happens here rather than deep inside the
                # per-unit matching helper.
                self._require_original(key)
                self._require_reconstructed(key)
                return self._compute_spike_matches(tolerance_ms=params["tolerance_ms"])

            case "isi_original":
                kernel = params["smoothing_kernel"]
                return self._compute_isi(self._require_original(key), smoothing_kernel=kernel)
            case "isi_reconstructed":
                return self._compute_isi(
                    self._require_reconstructed(key),
                    smoothing_kernel=params["smoothing_kernel"],
                )

            case "binned_counts_original":
                return self._compute_binned_counts(
                    self._require_original(key), bin_sizes_ms=params["bin_sizes_ms"]
                )
            case "binned_counts_reconstructed":
                return self._compute_binned_counts(
                    self._require_reconstructed(key), bin_sizes_ms=params["bin_sizes_ms"]
                )

            case "all_correlograms_original":
                return self._compute_all_correlograms(
                    self._require_original(key),
                    max_lag_ms=params["max_lag_ms"],
                    bin_size_ms=params["bin_size_ms"],
                )
            case "all_correlograms_reconstructed":
                return self._compute_all_correlograms(
                    self._require_reconstructed(key),
                    max_lag_ms=params["max_lag_ms"],
                    bin_size_ms=params["bin_size_ms"],
                )

            case "acg_original" | "acg_reconstructed":
                data = (
                    self._require_original(key)
                    if key == "acg_original"
                    else self._require_reconstructed(key)
                )
                # Spike-times-class data (native SPIKE_TIMES or fine-binned
                # SPIKES) gets one of two bit-identical routes, chosen by
                # ``_use_fft_correlograms`` purely on wall-clock:
                #   * Phy-compatible FFT (1 ms autocorrelation per unit, then
                #     coarsened to ``bin_size_ms`` under Phy's left-edge
                #     convention) — wins on short windows / wide lags. Output is
                #     bit-identical to the all-pairs Phy kernel (verified
                #     max|Δ| = 0 across {1/100, 5/500, 10/1000} ms).
                #   * The shared all-pairs Phy matrix (``all_correlograms_*``,
                #     one Phy call reused by ACG and CCG) — wins on long
                #     non-windowed sessions at narrow lags (e.g. the default
                #     1 ms / 100 ms, where the FFT path regresses ~1.4×).
                # Coarse-binned RATES (1 ms refinement impossible) fall through
                # to ``_compute_acgs_direct → _compute_acgs_from_binned``.
                if self._has_spike_times_path(data):
                    if self._use_fft_correlograms(
                        data,
                        max_lag_ms=params["max_lag_ms"],
                        bin_size_ms=params["bin_size_ms"],
                    ):
                        return self._compute_acgs_phy_compat_fft(
                            data,
                            max_lag_ms=params["max_lag_ms"],
                            bin_size_ms=params["bin_size_ms"],
                            smoothing_kernel=params["smoothing_kernel"],
                        )
                    corr_key = (
                        "all_correlograms_original"
                        if key == "acg_original"
                        else "all_correlograms_reconstructed"
                    )
                    from ..metrics.base import IntermediateRequest as IR

                    all_corr = self.get(
                        IR.from_dict(
                            corr_key,
                            {
                                "max_lag_ms": params["max_lag_ms"],
                                "bin_size_ms": params["bin_size_ms"],
                            },
                        )
                    )
                    return self._extract_acgs(
                        all_corr, smoothing_kernel=params["smoothing_kernel"]
                    )
                return self._compute_acgs_direct(
                    data,
                    max_lag_ms=params["max_lag_ms"],
                    bin_size_ms=params["bin_size_ms"],
                    smoothing_kernel=params["smoothing_kernel"],
                )

            case "ccg_original" | "ccg_reconstructed":
                data = (
                    self._require_original(key)
                    if key == "ccg_original"
                    else self._require_reconstructed(key)
                )
                # See the ACG case above — same bit-identical FFT-vs-Phy choice
                # via ``_use_fft_correlograms``. The FFT path is selective on the
                # ``max_pairs`` deterministically-chosen pairs; the Phy fallback
                # extracts those same pairs from the shared all-pairs matrix.
                # ``_compute_ccgs_direct`` still handles coarse RATES/SPIKES.
                if self._has_spike_times_path(data):
                    if self._use_fft_correlograms(
                        data,
                        max_lag_ms=params["max_lag_ms"],
                        bin_size_ms=params["bin_size_ms"],
                    ):
                        return self._compute_ccgs_phy_compat_fft(
                            data,
                            max_lag_ms=params["max_lag_ms"],
                            bin_size_ms=params["bin_size_ms"],
                            max_pairs=params["max_pairs"],
                        )
                    corr_key = (
                        "all_correlograms_original"
                        if key == "ccg_original"
                        else "all_correlograms_reconstructed"
                    )
                    from ..metrics.base import IntermediateRequest as IR

                    all_corr = self.get(
                        IR.from_dict(
                            corr_key,
                            {
                                "max_lag_ms": params["max_lag_ms"],
                                "bin_size_ms": params["bin_size_ms"],
                            },
                        )
                    )
                    return self._extract_ccgs(all_corr, max_pairs=params["max_pairs"])
                return self._compute_ccgs_direct(
                    data,
                    max_lag_ms=params["max_lag_ms"],
                    bin_size_ms=params["bin_size_ms"],
                    max_pairs=params["max_pairs"],
                )

            case "population_vectors_original":
                return self._compute_population_vectors(
                    self._require_original(key),
                    population_bin_size_ms=params["population_bin_size_ms"],
                )
            case "population_vectors_reconstructed":
                return self._compute_population_vectors(
                    self._require_reconstructed(key),
                    population_bin_size_ms=params["population_bin_size_ms"],
                )

            # --- Repeats-based intermediate (parameterized) ---
            case "stimulus_grouped_latents":
                return self._compute_stimulus_grouped_latents(
                    grouping_field=params["grouping_field"]
                )
            case "stimulus_grouped_reconstructions":
                self._require_reconstructed(key)
                return self._compute_stimulus_grouped_reconstructions(
                    grouping_field=params["grouping_field"]
                )

            # --- Latent structure intermediates (no params) ---
            case "latent_windows":
                return self._compute_latent_windows()
            case "original_windows":
                self._require_original(key)
                return self._compute_original_windows()
            case "latent_windows_per_unit":
                return self._compute_latent_windows_per_unit()
            case "original_windows_per_unit":
                self._require_original(key)
                return self._compute_original_windows_per_unit()

            # --- Passthrough intermediates (stored directly in cache at init) ---
            case "latents":
                if "latents" in self._cache:
                    return self._cache["latents"]
                raise ValueError("latents not provided to IntermediatesCache")
            case "behavioral_data_token_res":
                return self._cache.get("behavioral_data_token_res")  # None if not provided
            case "behavioral_data_input_res":
                return self._cache.get("behavioral_data_input_res")  # None if not provided
            case "token_bin_size_ms" | "sampling_rate" | "segment_absolute_start":
                return self._cache.get(key)  # None if not provided
            case "token_timestamps":
                if "token_timestamps" in self._cache:
                    return self._cache["token_timestamps"]
                raise ValueError("token_timestamps not provided to IntermediatesCache")

            # --- Replica fan-out passthroughs (M1 input-corruption robustness) ---
            # Stored as bare-string keys at __init__ when the replica
            # iterator (replica_collation.py) populated them. Looked up
            # here through an IntermediateRequest, so we re-route the
            # lookup to the bare-string entry.
            case "replica_latents":
                if "replica_latents" in self._cache:
                    return self._cache["replica_latents"]
                raise ValueError("replica_latents not provided to IntermediatesCache")
            case "replica_reconstructions":
                if "replica_reconstructions" in self._cache:
                    return self._cache["replica_reconstructions"]
                raise ValueError("replica_reconstructions not provided to IntermediatesCache")
            case "replica_originals":
                if "replica_originals" in self._cache:
                    return self._cache["replica_originals"]
                raise ValueError("replica_originals not provided to IntermediatesCache")
            case "replica_metadata":
                if "replica_metadata" in self._cache:
                    return self._cache["replica_metadata"]
                raise ValueError("replica_metadata not provided to IntermediatesCache")

            case _:
                raise ValueError(f"Unknown intermediate: {key}")

    def _compute_isi(self, data: EvalData, smoothing_kernel: int) -> list[np.ndarray]:
        """Compute ISI sequences with optional smoothing."""
        spike_times = self._require_spike_times(data)
        return [
            (self._smooth_boxcar(np.diff(st), smoothing_kernel) if len(st) > 1 else np.array([]))
            for st in spike_times
        ]

    def _require_spike_times(self, data: EvalData) -> list[np.ndarray]:
        """Ensure spike times are available for a given EvalData."""
        if data.spike_times is None:
            raise ValueError("Spike times are required for this intermediate.")
        return data.spike_times

    @staticmethod
    def _has_spike_times_path(data: EvalData) -> bool:
        """Check if data has (or can be converted to) spike times.

        Returns True for native SPIKE_TIMES or fine-binned SPIKES that
        the Phy algorithm can handle efficiently in a single merged call.
        Returns False for RATES and coarse SPIKES where the FFT path is needed.
        """
        if data.spike_times is not None:
            return True
        if data.data_type == DataType.SPIKES and data.bin_size_seconds is not None:
            from .converters import FINE_BIN_THRESHOLD_SECONDS

            return data.bin_size_seconds <= FINE_BIN_THRESHOLD_SECONDS
        return False

    def _use_fft_correlograms(self, data: EvalData, max_lag_ms: float, bin_size_ms: float) -> bool:
        """Decide between the Phy-compat FFT path and the all-pairs Phy matrix.

        Both produce bit-identical correlograms; this picks the faster one for
        the data at hand (see ``_FFT_*`` module constants for the cost model).
        Returns False — i.e. use the shared all-pairs Phy matrix — when the FFT
        path is either ineligible (non-integer-ms bins, or ``max_lag_ms`` not an
        integer multiple of ``bin_size_ms``) or expected to be slower (long
        non-windowed traces at narrow lags).
        """
        f = round(bin_size_ms)
        if not np.isclose(bin_size_ms, f) or f < 1:
            return False
        num_half = round(max_lag_ms / bin_size_ms)
        if not np.isclose(num_half * bin_size_ms, max_lag_ms):
            return False
        try:
            duration_seconds = self._get_duration(data)
        except ValueError:
            # Can't size the trace (e.g. interval-windowed data) — assume the
            # short-window regime the FFT path was built for.
            return True
        n_time_bins_1ms = duration_seconds * 1000.0
        if n_time_bins_1ms <= _FFT_SHORT_TRACE_MAX_1MS_BINS:
            return True
        return max_lag_ms >= _FFT_LONG_TRACE_MIN_LAG_MS

    def _smooth_boxcar(self, values: np.ndarray, kernel_size: int) -> np.ndarray:
        if kernel_size <= 1:
            return values
        kernel = np.ones(kernel_size, dtype=np.float64)
        kernel /= kernel.sum()
        return np.convolve(values, kernel, mode="same")

    def _get_duration(self, data: EvalData) -> float:
        """Return duration, inferring from binned data when needed."""
        if data.duration is not None:
            return data.duration
        if data.bin_size_seconds is None:
            raise ValueError("Duration or bin_size_seconds required for binned data.")
        if data.binned_spikes is not None:
            return data.binned_spikes.shape[1] * data.bin_size_seconds
        if data.binned_counts is not None:
            return data.binned_counts.shape[1] * data.bin_size_seconds
        if data.rates_hz is not None:
            return data.rates_hz.shape[1] * data.bin_size_seconds
        raise ValueError("Duration could not be inferred for this data.")

    def _get_binned_counts_base(self, data: EvalData) -> tuple[np.ndarray, float]:
        """Return counts array and base bin size for binned/rate data."""
        if data.bin_size_seconds is None:
            raise ValueError("bin_size_seconds required for binned or rate data.")
        if data.binned_spikes is not None:
            return data.binned_spikes.astype(np.float32), data.bin_size_seconds
        if data.binned_counts is not None:
            return data.binned_counts.astype(np.float32), data.bin_size_seconds
        if data.rates_hz is not None:
            counts = data.rates_hz * data.bin_size_seconds
            return counts.astype(np.float32), data.bin_size_seconds
        raise ValueError("Binned data or rates_hz required for this intermediate.")

    def _rebin_counts(
        self,
        counts: np.ndarray,
        base_bin_size: float,
        target_bin_size: float,
    ) -> np.ndarray:
        """
        Aggregate binned counts into larger bins."""
        if target_bin_size < base_bin_size - 1e-12:
            raise ValueError("Target bin size must be >= source bin size for binned conversions.")
        if np.isclose(target_bin_size, base_bin_size):
            return counts

        ratio = target_bin_size / base_bin_size
        if not np.isclose(ratio, round(ratio)):
            raise ValueError("Target bin size must be an integer multiple of source bin size.")

        ratio_int = int(round(ratio))
        n_bins = counts.shape[1] // ratio_int
        truncated = counts[:, : n_bins * ratio_int]
        return truncated.reshape(counts.shape[0], n_bins, ratio_int).sum(axis=2)

    def _compute_binned_counts(
        self, data: EvalData, bin_sizes_ms: list[float] | tuple[float, ...]
    ) -> dict[float, np.ndarray]:
        """Compute binned spike counts for multiple bin sizes."""
        sizes = list(bin_sizes_ms)
        if data.spike_times is not None:
            return self._compute_binned_counts_from_spike_times(data, bin_sizes_ms=sizes)
        return self._compute_binned_counts_from_binned(data, bin_sizes_ms=sizes)

    def _compute_binned_counts_from_spike_times(
        self, data: EvalData, bin_sizes_ms: list[float]
    ) -> dict[float, np.ndarray]:
        """Compute binned spike counts from spike times."""
        duration = self._get_duration(data)
        results = {}

        spike_times = data.spike_times
        if spike_times is None:
            raise ValueError("Spike times required for this intermediate.")

        for bin_size_ms in bin_sizes_ms:
            bin_size_seconds = bin_size_ms / 1000.0
            n_bins = int(np.ceil(duration / bin_size_seconds))

            counts = np.zeros((len(spike_times), n_bins))
            for i, st in enumerate(spike_times):
                if len(st) > 0:
                    bin_indices = np.minimum((st / bin_size_seconds).astype(int), n_bins - 1)
                    np.add.at(counts[i], bin_indices, 1)

            results[bin_size_ms] = counts

        return results

    def _compute_binned_counts_from_binned(
        self, data: EvalData, bin_sizes_ms: list[float]
    ) -> dict[float, np.ndarray]:
        """Compute binned spike counts from binned counts or rates.

        Bin sizes finer than the data resolution are silently skipped
        (with a warning) rather than raising an error.  This allows
        multi-timescale metrics to degrade gracefully when the tokenizer
        bin size is coarser than some of the requested timescales.

        Raises ``ValueError`` only when *no* requested bin sizes are feasible.
        """
        counts, base_bin_size = self._get_binned_counts_base(data)
        base_bin_size_ms = base_bin_size * 1000.0
        results = {}
        skipped: list[float] = []

        for bin_size_ms in bin_sizes_ms:
            target_bin_size = bin_size_ms / 1000.0
            if target_bin_size < base_bin_size - 1e-12:
                skipped.append(bin_size_ms)
                continue
            ratio = target_bin_size / base_bin_size
            if not np.isclose(ratio, round(ratio)):
                skipped.append(bin_size_ms)
                continue
            rebinned = self._rebin_counts(counts, base_bin_size, target_bin_size)
            results[bin_size_ms] = rebinned

        if skipped:
            global _warned_bin_size_skip  # noqa: PLW0603
            if not _warned_bin_size_skip:
                _warned_bin_size_skip = True
                warnings.warn(
                    f"Skipped bin sizes {skipped} ms (finer than data resolution "
                    f"{base_bin_size_ms:.1f} ms or not an integer multiple). "
                    f"Remaining: {sorted(results.keys())} ms.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        if not results:
            raise ValueError(
                f"No requested bin sizes {bin_sizes_ms} are feasible for data with "
                f"bin size {base_bin_size_ms:.1f} ms. All requested sizes are finer "
                f"than the data resolution or not integer multiples."
            )

        return results

    def _compute_population_vectors(
        self, data: EvalData, population_bin_size_ms: float
    ) -> np.ndarray:
        """Compute population activity vectors.

        That is, returns a matrix (time_bins x units) with binned spike counts
        (Difference to binned_counts?)
        """
        if data.spike_times is not None:
            return self._compute_population_vectors_from_spike_times(data, population_bin_size_ms)
        return self._compute_population_vectors_from_binned(data, population_bin_size_ms)

    def _compute_population_vectors_from_spike_times(
        self, data: EvalData, population_bin_size_ms: float
    ) -> np.ndarray:
        """Compute population vectors from spike times."""
        spike_times = self._require_spike_times(data)
        bin_size_seconds = population_bin_size_ms / 1000.0
        duration = self._get_duration(data)
        n_bins = int(np.ceil(duration / bin_size_seconds))

        # shape: (time_bins x units)
        pop_vectors = np.zeros((n_bins, len(spike_times)))

        for i, st in enumerate(spike_times):
            if len(st) > 0:
                bin_indices = np.minimum((st / bin_size_seconds).astype(int), n_bins - 1)
                np.add.at(pop_vectors[:, i], bin_indices, 1)

        return pop_vectors

    def _compute_population_vectors_from_binned(
        self, data: EvalData, population_bin_size_ms: float
    ) -> np.ndarray:
        """Compute population vectors from binned counts or rates."""
        counts, base_bin_size = self._get_binned_counts_base(data)
        target_bin_size = population_bin_size_ms / 1000.0
        rebinned = self._rebin_counts(counts, base_bin_size, target_bin_size)
        return rebinned.T

    def _compute_firing_rates(self, data: EvalData) -> np.ndarray:
        """Compute firing rates per unit."""
        duration = self._get_duration(data)
        if data.spike_times is not None:
            return np.array([len(st) / duration for st in data.spike_times])

        counts, _ = self._get_binned_counts_base(data)
        return counts.sum(axis=1) / duration

    def _compute_latent_windows(self) -> np.ndarray | list[np.ndarray] | None:
        """Return per-window latent representations, flattened to [N_windows, D_flat].

        Input latents have shape ``[N_windows, ...]`` where the
        trailing dimensions depend on the tokenizer:
        - Per-neuron tokenizers: ``[N_windows, T_tokens, U, D_latent]``
        - Population tokenizers: ``[N_windows, T_tokens, D_latent]``

        All trailing dimensions are flattened into one vector per window
        (e.g. ``D_flat = T_tokens * U * D_latent``), producing ``[N_windows, D_flat]``.
        This is used by structure-preservation metrics (RSA, kNN) that compute
        pairwise distances between window representations.

        Returns None if no latent data was provided.  Returns a list when
        windows have unequal token counts (metrics handle this case).
        """
        latent = self._cache.get("latents")
        if latent is None:
            return None
        if isinstance(latent, list):
            stacked = self._stack_uniform_window_list(latent)
            if stacked is None:
                # Unequal window sizes from token overlap trimming — return as-is,
                # metrics that need uniform shapes will handle this.
                return latent
            latent = stacked
        latent = np.asarray(latent) if not isinstance(latent, np.ndarray) else latent
        # Squeeze singleton time dim: [N_windows, 1, U, m] -> [N_windows, U, m]
        if latent.ndim == 4 and latent.shape[1] == 1:
            latent = latent.squeeze(1)
        # Latents are stacked over windows during collation, not concatenated.
        if latent.ndim > 2:
            return latent.reshape(latent.shape[0], -1).astype(np.float64)
        return latent.astype(np.float64)

    def _compute_stimulus_grouped_latents(self, grouping_field: str) -> dict[str, Any]:
        """Group per-window latents by stimulus label for repeats-based metrics.

        Produces a dense, NaN-padded ``[S, R_max, D]`` tensor where each
        ``(stimulus, repeat)`` slot holds the flattened latent vector for one
        logical evaluation window (one stimulus presentation, per the
        interval-windowing pipeline).

        Parameters
        ----------
        grouping_field : str
            Name of the behavioral field (in ``behavioral_data_token_res``) to
            use as the stimulus label. Must be constant within each window.

        Returns
        -------
        dict with keys:
            - ``latents`` : np.ndarray[S, R_max, D], float64, NaN-padded.
            - ``repeat_mask`` : np.ndarray[S, R_max], bool.
            - ``stimulus_ids`` : np.ndarray[S], original dtype preserved.
            - ``repeat_counts`` : np.ndarray[S], int.
            - ``token_shape`` : tuple[int, int] ``(n_tokens_per_window,
              latent_dim)`` after any cut-to-smallest.
            - ``n_tokens_cut_info`` : dict | None. ``None`` if windows all had
              the same token count; otherwise a dict describing the cut.

        """
        return self._grouped_by_stimulus_from_latents(
            grouping_field=grouping_field,
            output_values_key="latents",
        )

    def _compute_stimulus_grouped_reconstructions(self, grouping_field: str) -> dict[str, Any]:
        """Group per-window reconstructions by stimulus label.

        Companion to :meth:`_compute_stimulus_grouped_latents`, operating
        on ``self.reconstructed`` (re-windowed to ``[N_windows, U, T_window]``
        using the latent window count) instead of latent tokens. Shape of
        the dense output: ``[S, R_max, U, T_window]``, NaN-padded.

        Stimulus assignment, short-window filter, and any window drops are
        inherited verbatim from the latent-token-count pipeline so that
        reconstruction and latent groupings index the same windows /
        repeats.

        """
        if self.reconstructed is None:
            raise ValueError("reconstructed data is required for reconstruction windows.")
        per_window_recon = self._compute_original_windows_per_unit_for(self.reconstructed)
        if per_window_recon is None:
            raise ValueError(
                "stimulus_grouped_reconstructions requires reconstructed "
                "binned data (binned_spikes / binned_counts / rates_hz) and "
                "latents to determine window count."
            )
        per_window_list = [per_window_recon[i] for i in range(per_window_recon.shape[0])]
        return self._grouped_by_stimulus_from_latents(
            grouping_field=grouping_field,
            output_values_key="reconstructions",
            alt_per_window_values=per_window_list,
            cut_trailing_axis=-1,
        )

    def _compute_original_windows_per_unit_for(self, data: EvalData) -> np.ndarray | None:
        """Re-window given ``data`` into ``[N_windows, U, T_window]``.

        Mirror of :meth:`_compute_original_windows_per_unit` but operates on
        an arbitrary ``EvalData`` (used for reconstructions). N_windows is
        read from the cached ``latents``. Trailing bins beyond the last
        full window are truncated.

        """
        latent = self._cache.get("latents")
        if latent is None:
            return None
        if isinstance(latent, list):
            N_windows = len(latent)
        else:
            arr = np.asarray(latent)
            N_windows = arr.shape[0]
        if N_windows < 1:
            return None

        spikes = data.counts_per_bin()
        if spikes is None:
            return None

        n_units, total_time = spikes.shape
        window_time = total_time // N_windows
        if window_time == 0:
            return None
        usable_time = N_windows * window_time
        windows = spikes[:, :usable_time].reshape(n_units, N_windows, window_time)
        return np.transpose(windows, (1, 0, 2)).astype(np.float64)

    def _grouped_by_stimulus_from_latents(
        self,
        grouping_field: str,
        output_values_key: str,
        alt_per_window_values: list[np.ndarray] | None = None,
        cut_trailing_axis: int | None = None,
    ) -> dict[str, Any]:
        """Shared stimulus-grouping pipeline driven by latent token counts.

        Body shared by :meth:`_compute_stimulus_grouped_latents` and
        :meth:`_compute_stimulus_grouped_reconstructions`. Uses latent token
        counts for all per-window filtering so both intermediates agree on
        which windows (and which repeats) are kept.

        When ``alt_per_window_values`` is None, the latent arrays are
        grouped and flattened per window (original latent-version
        behavior producing ``[S, R_max, D_flat]``). When given, those
        arrays are grouped instead (must match latent N_windows and
        order); NaN-padding uses the full trailing shape of the first
        kept entry (producing ``[S, R_max, *trailing_shape]``).
        ``cut_trailing_axis`` controls which axis is cut-to-smallest when
        alt values have ragged trailing shapes (e.g. ``-1`` for the time
        axis of reconstructions).

        """
        latents = self._cache.get("latents")
        if latents is None:
            raise ValueError(
                "stimulus_grouped_latents requires latents to be provided to IntermediatesCache."
            )

        behavioral = self._cache.get("behavioral_data_token_res")
        if behavioral is None or grouping_field not in behavioral:
            raise ValueError(
                f"stimulus_grouped_latents requires behavioral_data_token_res['{grouping_field}'] "
                f"to be provided to IntermediatesCache."
            )
        grouping_values = behavioral[grouping_field]  # shape [total_tokens]
        # `window_group_id_token_res` (attached by the collation pipeline to
        # behavioral_data_token_res) gives each token its window index. We use
        # it below to assert that every cursor-sliced chunk contains tokens
        # from exactly one window — catches any future drift between the
        # behavioral-data concatenation order and the latents stacking order.
        if "window_group_id_token_res" not in behavioral:
            raise ValueError(
                "stimulus_grouped_latents requires "
                "behavioral_data_token_res['window_group_id_token_res'] to be "
                "present (attached during collation). Absent here — fixture "
                "or pipeline change needed."
            )
        wids = np.asarray(behavioral["window_group_id_token_res"])

        # --- Make sure latents are a list of per-window arrays ---
        # Each entry: shape [T_i, ...trailing...].
        if isinstance(latents, list):
            per_window_latents = [
                np.asarray(z) if not isinstance(z, np.ndarray) else z for z in latents
            ]
        else:
            arr = np.asarray(latents) if not isinstance(latents, np.ndarray) else latents
            # Squeeze a degenerate singleton time axis when present:
            # [N, 1, U, m] -> [N, 1, ...] stays 4D but we just iterate along axis 0.
            # We DO NOT squeeze here — we keep the token dim as axis=1 per spec.
            per_window_latents = [arr[i] for i in range(arr.shape[0])]

        N_windows = len(per_window_latents)

        # --- Build parallel per-window *values* list that will actually be
        # stacked into the dense output. For the latent version this is
        # ``per_window_latents`` itself; for the reconstruction version it is
        # the externally supplied ``alt_per_window_values`` (already
        # re-windowed to ``[N_windows, U, T_window]``). All token-count-based
        # filtering below operates on ``per_window_latents``, and we apply
        # the same drops/keeps to ``per_window_values`` in lockstep.
        if alt_per_window_values is None:
            per_window_values = per_window_latents
            _values_are_latents = True
        else:
            if len(alt_per_window_values) != N_windows:
                raise ValueError(
                    f"alt_per_window_values length {len(alt_per_window_values)} "
                    f"does not match latent N_windows {N_windows}."
                )
            per_window_values = [
                np.asarray(v) if not isinstance(v, np.ndarray) else v
                for v in alt_per_window_values
            ]
            _values_are_latents = False

        # --- Compute per-window token counts ---
        n_tokens_per_window = np.array([w.shape[0] for w in per_window_latents], dtype=int)

        # --- Reduce per-token grouping values to one label per window ---
        # grouping_values is concatenated across all windows at token resolution.
        total_tokens_from_latents = int(n_tokens_per_window.sum())
        if grouping_values.shape[0] != total_tokens_from_latents:
            raise ValueError(
                f"behavioral_data_token_res['{grouping_field}'] has length "
                f"{grouping_values.shape[0]} but sum of per-window token counts is "
                f"{total_tokens_from_latents}."
            )

        # Each window may still contain NaN-labelled tokens (e.g. post-stim
        # lag bins outside the raw stimulus bounds). Those are dropped
        # silently. Any remaining non-NaN labels must now be strictly
        # uniform within the window: mixed labels mean the per-stim sampler or
        # label plumbing is wrong and must fail loudly.

        if wids.shape[0] != total_tokens_from_latents:
            raise ValueError(
                f"behavioral_data_token_res['window_group_id_token_res'] has "
                f"length {wids.shape[0]} but sum of per-window token counts "
                f"is {total_tokens_from_latents}."
            )

        stimulus_per_window: list[Any] = []
        cursor = 0
        is_float_label = np.issubdtype(grouping_values.dtype, np.floating)
        n_nan_dropped_per_window = np.zeros(N_windows, dtype=int)
        # Windows whose tokens are ALL NaN cannot be assigned a stimulus
        # label. We drop them defensively, log a warning, and report the
        # count via ``tokens_dropped_info["n_all_nan_windows_dropped"]``.
        dropped_all_nan_indices: list[int] = []
        kept_window_indices: list[int] = []
        for w_idx, T_i in enumerate(n_tokens_per_window):
            chunk = grouping_values[cursor : cursor + T_i]
            wid_chunk = wids[cursor : cursor + T_i]
            # Assert all tokens in this cursor slice belong to a single window,
            # regardless of the specific window id. Catches misalignment where a
            # slice would span across window boundaries.
            if T_i > 0 and not np.all(wid_chunk == wid_chunk[0]):
                raise AssertionError(
                    f"Cursor slice [{cursor}:{cursor + T_i}) for latents stack "
                    f"index {w_idx} spans multiple window ids "
                    f"{np.unique(wid_chunk).tolist()} — the behavioral-data "
                    f"concatenation order and the latents stacking order have "
                    f"drifted apart."
                )
            cursor += int(T_i)
            if is_float_label:
                non_nan_mask = ~np.isnan(chunk)
            else:
                non_nan_mask = np.ones(T_i, dtype=bool)
            chunk_non_nan = chunk[non_nan_mask]
            if chunk_non_nan.size == 0:
                dropped_all_nan_indices.append(w_idx)
                continue
            kept_window_indices.append(w_idx)
            unique_non_nan_labels = np.unique(chunk_non_nan)
            if unique_non_nan_labels.size > 1:
                raise ValueError(
                    f"Window {w_idx} has multiple non-NaN '{grouping_field}' labels "
                    f"({unique_non_nan_labels.tolist()}). With the per-stim sampler, every "
                    f"non-NaN label inside one window must match. Mixed labels indicate "
                    f"either a sampler bug (window spans two stims) or a label-data bug."
                )
            window_label = unique_non_nan_labels[0]
            keep_mask = non_nan_mask.copy()
            n_nan_dropped_per_window[w_idx] = int((~non_nan_mask).sum())
            stimulus_per_window.append(window_label)
            if int((~keep_mask).sum()) > 0:
                per_window_latents[w_idx] = per_window_latents[w_idx][keep_mask]
                # Token-level keep_mask only applies to the latents (tokens)
                # representation. Alt values (e.g. reconstructions in input
                # resolution) are not indexed per token, so we leave them
                # untouched here.
        # Drop all-NaN windows. We must filter ``per_window_latents`` and the
        # per-window stats in parallel; ``stimulus_per_window`` was only
        # appended for kept windows.
        n_all_nan_windows_dropped = len(dropped_all_nan_indices)
        if n_all_nan_windows_dropped > 0:
            logger.warning(
                "Dropping %d all-NaN stimulus-grouping window(s) for %r.",
                n_all_nan_windows_dropped,
                grouping_field,
            )
            kept_set = set(kept_window_indices)
            per_window_latents = [w for i, w in enumerate(per_window_latents) if i in kept_set]
            per_window_values = (
                [v for i, v in enumerate(per_window_values) if i in kept_set]
                if not _values_are_latents
                else per_window_latents
            )
            n_nan_dropped_per_window = n_nan_dropped_per_window[kept_window_indices]
            N_windows = len(per_window_latents)
            if N_windows == 0:
                raise ValueError(
                    f"All {n_all_nan_windows_dropped} windows had zero tokens "
                    f"with a valid (non-NaN) '{grouping_field}' — no stimulus "
                    f"labels could be assigned to any window."
                )
        # Recompute per-window token counts after label-based dropping.
        n_tokens_per_window = np.array([w.shape[0] for w in per_window_latents], dtype=int)

        # --- Drop short-outlier windows before cut-to-smallest ---
        # Without this, a handful of short windows (e.g. truncated presentations
        # at block/session boundaries) force every other window to be cut to
        # their length, discarding a lot of usable signal. Threshold is
        # relative to the distribution itself: windows below
        # SHORT_WINDOW_FRACTION * <10th-percentile token count> are dropped.
        # Tradeoff: a lower fraction keeps more repeats, a higher one gives
        # longer sequences per repeat.
        SHORT_WINDOW_FRACTION = 0.8
        SHORT_WINDOW_PERCENTILE = 10.0
        short_window_info: dict | None = None
        if len(n_tokens_per_window) >= 2:
            p_ref = float(np.percentile(n_tokens_per_window, SHORT_WINDOW_PERCENTILE))
            short_threshold = SHORT_WINDOW_FRACTION * p_ref
            keep_window_mask = n_tokens_per_window >= short_threshold
            n_short_dropped = int((~keep_window_mask).sum())
            if n_short_dropped > 0:
                short_window_info = {
                    "threshold": float(short_threshold),
                    "reference_percentile": SHORT_WINDOW_PERCENTILE,
                    "reference_percentile_value": p_ref,
                    "fraction_of_reference": SHORT_WINDOW_FRACTION,
                    "n_windows_dropped": n_short_dropped,
                    "dropped_window_n_tokens": n_tokens_per_window[~keep_window_mask].tolist(),
                }
                per_window_latents = [
                    w
                    for w, keep in zip(per_window_latents, keep_window_mask, strict=False)
                    if keep
                ]
                if not _values_are_latents:
                    per_window_values = [
                        v
                        for v, keep in zip(per_window_values, keep_window_mask, strict=False)
                        if keep
                    ]
                else:
                    per_window_values = per_window_latents
                stimulus_per_window = [
                    s
                    for s, keep in zip(stimulus_per_window, keep_window_mask, strict=False)
                    if keep
                ]
                n_tokens_per_window = n_tokens_per_window[keep_window_mask]
                # Re-slice per-window drop stats so discarded short windows don't
                # inflate the reported NaN-drop totals.
                n_nan_dropped_per_window = n_nan_dropped_per_window[keep_window_mask]
                N_windows = len(per_window_latents)
                if N_windows == 0:
                    raise ValueError(
                        "Short-window filter dropped all windows. Check your "
                        "token count distribution."
                    )

        # --- Handle ragged token counts: cut-to-smallest ---
        n_tokens_cut_info: dict | None = None
        if n_tokens_per_window.min() != n_tokens_per_window.max():
            T_used = int(n_tokens_per_window.min())
            warnings.warn(
                f"Windows have different token counts: min={n_tokens_per_window.min()}, "
                f"max={n_tokens_per_window.max()}, mean={n_tokens_per_window.mean():.2f}. "
                f"Cutting every window to the smallest ({T_used}) tokens.",
                UserWarning,
                stacklevel=2,
            )
            # Cut each window to T_used tokens, keeping first tokens. Apply
            # the same drops (but NOT the token-axis slicing) to alt values.
            dropped = 0
            cut_windows: list[np.ndarray] = []
            cut_values: list[np.ndarray] = []
            cut_stimuli: list[Any] = []
            for w, v, sid in zip(
                per_window_latents, per_window_values, stimulus_per_window, strict=False
            ):
                if T_used == 0 or w.shape[0] == 0:
                    dropped += 1
                    continue
                cut_windows.append(w[:T_used])
                cut_values.append(v[:T_used] if _values_are_latents else v)
                cut_stimuli.append(sid)
            n_tokens_cut_info = {
                "n_tokens_mean": float(n_tokens_per_window.mean()),
                "n_tokens_min": int(n_tokens_per_window.min()),
                "n_tokens_max": int(n_tokens_per_window.max()),
                "n_tokens_used": T_used,
                "n_windows_dropped": int(dropped),
            }
            per_window_latents = cut_windows
            per_window_values = cut_values if not _values_are_latents else cut_windows
            stimulus_per_window = cut_stimuli
            N_windows = len(per_window_latents)

        if N_windows == 0:
            raise ValueError(
                "No valid windows remaining after cut-to-smallest for stimulus_grouped_latents."
            )

        T_used = per_window_latents[0].shape[0]
        # Flatten trailing dims (keep the token dim absorbed into D as per spec:
        # "each window becomes (D,) where D = n_tokens_per_window * product_of_other_dims").
        flat_per_window = np.stack([w.reshape(-1) for w in per_window_latents], axis=0).astype(
            np.float64
        )
        D_flat = flat_per_window.shape[1]

        # latent_dim (inner-most) is derivable as D_flat / T_used (integer).
        latent_dim = D_flat // T_used
        token_shape = (T_used, latent_dim)

        # --- Group by stimulus ---
        stimulus_ids_arr = np.array(stimulus_per_window)
        # Stable order: sorted by stimulus id (uses numpy sort which is stable for keys).
        unique_stimuli, inverse = np.unique(stimulus_ids_arr, return_inverse=True)
        S = len(unique_stimuli)

        # Per-stimulus repeat indices (order of occurrence).
        per_stim_window_idx: list[list[int]] = [[] for _ in range(S)]
        for win_idx, stim_idx in enumerate(inverse):
            per_stim_window_idx[int(stim_idx)].append(win_idx)

        repeat_counts = np.array([len(lst) for lst in per_stim_window_idx], dtype=int)
        R_max = int(repeat_counts.max())

        if _values_are_latents:
            dense = np.full((S, R_max, D_flat), np.nan, dtype=np.float64)
            mask = np.zeros((S, R_max), dtype=bool)
            for s_idx, window_indices in enumerate(per_stim_window_idx):
                for r_idx, w_idx in enumerate(window_indices):
                    dense[s_idx, r_idx] = flat_per_window[w_idx]
                    mask[s_idx, r_idx] = True
        else:
            # Non-latent values: align trailing shape across windows. If
            # ragged along ``cut_trailing_axis``, cut-to-smallest there.
            trailing_shapes = [v.shape for v in per_window_values]
            # All entries should agree on non-cut axes.
            ref_shape = list(trailing_shapes[0])
            if cut_trailing_axis is not None:
                axis = (
                    cut_trailing_axis
                    if cut_trailing_axis >= 0
                    else (len(ref_shape) + cut_trailing_axis)
                )
                min_len = min(s[axis] for s in trailing_shapes)
                ref_shape[axis] = min_len
                aligned_values = []
                for v in per_window_values:
                    slicer = [slice(None)] * v.ndim
                    slicer[axis] = slice(0, min_len)
                    aligned_values.append(v[tuple(slicer)])
                per_window_values = aligned_values
            # Validate remaining shapes agree.
            for v in per_window_values:
                if tuple(v.shape) != tuple(ref_shape):
                    raise ValueError(
                        f"Per-window values for stimulus_grouped_{output_values_key} "
                        f"have incompatible shapes: {v.shape} vs {tuple(ref_shape)}."
                    )
            dense = np.full((S, R_max, *ref_shape), np.nan, dtype=np.float64)
            mask = np.zeros((S, R_max), dtype=bool)
            for s_idx, window_indices in enumerate(per_stim_window_idx):
                for r_idx, w_idx in enumerate(window_indices):
                    dense[s_idx, r_idx] = per_window_values[w_idx]
                    mask[s_idx, r_idx] = True

        tokens_dropped_info: dict | None = None
        n_nan_total = int(n_nan_dropped_per_window.sum()) if len(n_nan_dropped_per_window) else 0
        # Mixed non-NaN labels within a window raise (strict uniformity, see
        # above) rather than dropping the minority, so there is no
        # "mismatched-drop" path to count.
        if n_nan_total > 0 or n_all_nan_windows_dropped > 0:
            tokens_dropped_info = {
                "n_nan_tokens_dropped_total": n_nan_total,
                "n_windows_with_nan_drops": (
                    int((n_nan_dropped_per_window > 0).sum())
                    if len(n_nan_dropped_per_window)
                    else 0
                ),
                "max_nan_dropped_per_window": (
                    int(n_nan_dropped_per_window.max()) if len(n_nan_dropped_per_window) else 0
                ),
                "n_all_nan_windows_dropped": n_all_nan_windows_dropped,
            }

        return {
            output_values_key: dense,
            "repeat_mask": mask,
            "stimulus_ids": unique_stimuli,
            "repeat_counts": repeat_counts,
            "token_shape": token_shape,
            "token_bin_size_ms": self._cache.get("token_bin_size_ms"),
            "n_tokens_cut_info": n_tokens_cut_info,
            "tokens_dropped_info": tokens_dropped_info,
            "short_window_info": short_window_info,
        }

    def _compute_original_windows(self) -> np.ndarray | None:
        """Re-window original data into per-window vectors, shape [N, D].

        Uses the number of latent windows to determine window boundaries:
        the collated segment contains N concatenated windows along the time
        axis, so, in input space time ``window_time = total_time // N_windows``.

        Returns None if no latent data is available (cannot determine N).
        """
        latent = self._cache.get("latents")
        if latent is None:
            return None
        N_windows = self._window_count_from_latents(latent)
        if N_windows is None:
            return None
        if N_windows < 2:
            return None

        original = self.original
        if original is None:
            return None
        spikes = original.counts_per_bin()  # (n_units, total_time)
        if spikes is None:
            return None

        n_units, total_time = spikes.shape
        window_time = total_time // N_windows
        if window_time == 0:
            return None

        # Truncate to exact multiple of N windows, then reshape
        usable_time = N_windows * window_time
        windows = spikes[:, :usable_time].reshape(n_units, N_windows, window_time)
        # Transpose to (N, n_units, window_time), then flatten per window
        windows = np.transpose(windows, (1, 0, 2))  # [N_windows, n_units, window_time]
        return windows.reshape(N_windows, -1).astype(np.float64)

    def _compute_latent_windows_per_unit(self) -> np.ndarray | list[np.ndarray] | None:
        """Return per-window latent representations with an explicit unit dimension.

        Supports dense per-neuron layouts ``[N_windows, U, m]`` and
        ``[N_windows, T_tokens, U, m]``. Uniform list-valued window storage is
        stacked automatically; ragged lists are returned as-is so downstream
        metrics can fail soft when they require uniform shapes.

        Also handles the 4D unified format ``[N_windows, 1, U, m]`` from
        tokenizers that use singleton time dimensions (e.g. compressed sensing
        in per_neuron mode): the singleton is squeezed out automatically.
        """
        latent = self._cache.get("latents")
        if latent is None:
            return None
        if isinstance(latent, list):
            stacked = self._stack_uniform_window_list(latent)
            if stacked is None:
                return latent
            latent = stacked
        latent = np.asarray(latent) if not isinstance(latent, np.ndarray) else latent
        # Squeeze singleton time dim: [N_windows, 1, U, m] -> [N_windows, U, m]
        if latent.ndim == 4 and latent.shape[1] == 1:
            latent = latent.squeeze(1)
        n_units = self._get_n_units()
        if latent.ndim == 3:
            if n_units is not None and latent.shape[1] != n_units:
                return None
            return latent.astype(np.float64)
        if latent.ndim == 4:
            if n_units is not None and latent.shape[2] != n_units:
                return None
            return latent.astype(np.float64)
        return None

    def _compute_original_windows_per_unit(self) -> np.ndarray | None:
        """Re-window original data into [N_windows, U, T_window] preserving unit dimension.

        Same re-windowing as ``_compute_original_windows`` but without the
        final flatten, so per-unit metrics can iterate over the unit axis.

        Returns None if no latent data is available or if there are fewer than
        two windows. The original-side re-windowing only depends on the number
        of latent windows, not on whether the latent storage is dense or ragged.
        """
        latent = self._cache.get("latents")
        if latent is None:
            return None
        N_windows = self._window_count_from_latents(latent)
        if N_windows is None:
            return None
        if N_windows < 2:
            return None

        original = self.original
        if original is None:
            return None
        spikes = original.counts_per_bin()  # (n_units, total_time)
        if spikes is None:
            return None

        n_units, total_time = spikes.shape
        window_time = total_time // N_windows
        if window_time == 0:
            return None

        usable_time = N_windows * window_time
        windows = spikes[:, :usable_time].reshape(n_units, N_windows, window_time)
        # Transpose to (N_windows, n_units, window_time) — no flatten
        return np.transpose(windows, (1, 0, 2)).astype(np.float64)

    def _get_n_units(self) -> int | None:
        """Return number of units from available data, or None.

        Returns ``None`` when ``self.original`` was not provided (e.g. on
        evaluation paths that did not assemble an input trace), preserving
        this helper's existing "None when undeterminable" contract so
        latent-side intermediates that only use the unit count for an
        optional shape check can still skip gracefully.
        """
        if self.original is None:
            return None
        if self.original.binned_spikes is not None:
            return self.original.binned_spikes.shape[0]
        if self.original.binned_counts is not None:
            return self.original.binned_counts.shape[0]
        if self.original.rates_hz is not None:
            return self.original.rates_hz.shape[0]
        if self.original.spike_times is not None:
            return len(self.original.spike_times)
        return None

    def _compute_spike_matches(self, tolerance_ms: float) -> dict[str, Any]:
        """
        Match spikes between original and reconstructed within tolerance window.
        Returns dict with matched indices, timing errors, and TP/FP/FN counts.

        Each dict value is a list with one entry per unit.
        """
        tolerance = tolerance_ms / 1000.0  # convert to seconds
        results = {
            "matched_original_idx": [],
            "matched_reconstructed_idx": [],
            "timing_errors": [],
            "true_positives": [],
            "false_positives": [],
            "false_negatives": [],
        }

        if self.original is None or self.reconstructed is None:
            raise ValueError("Original and reconstructed data required for spike matching.")
        if self.original.n_units is None:
            raise ValueError("n_units must be set for spike matching.")
        if self.original.spike_times is None or self.reconstructed.spike_times is None:
            raise ValueError("Spike times required for spike matching.")

        for unit_idx in range(self.original.n_units):
            orig = self.original.spike_times[unit_idx]
            recon = self.reconstructed.spike_times[unit_idx]

            # Implement greedy matching within tolerance
            match_result = self._match_spikes_single_unit(orig, recon, tolerance)

            for key in results:
                results[key].append(match_result[key])

        results["tolerance"] = [tolerance]

        return results

    def _match_spikes_single_unit(
        self,
        original: np.ndarray,
        reconstructed: np.ndarray,
        tolerance: float,
    ) -> dict[str, Any]:
        """Match spikes for a single unit using greedy nearest-neighbor matching.

        For each original spike (processed in index order), finds the closest
        unused reconstructed spike within *tolerance* seconds.  Uses binary
        search (``np.searchsorted``) to restrict the candidate set to the
        tolerance window, giving O(N log N) complexity instead of O(N²).

        Args:
            original: 1D sorted array of spike times (seconds).
            reconstructed: 1D sorted array of spike times (seconds).
            tolerance: Maximum allowed time difference for a match (seconds).

        Returns:
            dict[str, Any]
                Contains: matched indices, timing errors, and TP/FP/FN counts.
        """
        n_orig = len(original)
        n_recon = len(reconstructed)

        if n_orig == 0 or n_recon == 0:
            return {
                "matched_original_idx": np.array([], dtype=int),
                "matched_reconstructed_idx": np.array([], dtype=int),
                "timing_errors": np.array([], dtype=float),
                "true_positives": 0,
                "false_positives": n_recon,
                "false_negatives": n_orig,
            }

        matched_orig_idx = []
        matched_recon_idx = []
        timing_errors = []
        recon_used = np.zeros(n_recon, dtype=bool)

        for i in range(n_orig):
            t_orig = float(original[i])
            # Binary search: find insertion point, then check the two neighbours
            pos = np.searchsorted(reconstructed, t_orig)
            best_j = -1
            best_dist = tolerance + 1.0
            for candidate in (pos - 1, pos):
                if 0 <= candidate < n_recon:
                    d = abs(float(reconstructed[candidate]) - t_orig)
                    if d < best_dist:
                        best_dist = d
                        best_j = candidate

            if best_j >= 0 and best_dist <= tolerance and not recon_used[best_j]:
                matched_orig_idx.append(i)
                matched_recon_idx.append(best_j)
                timing_errors.append(reconstructed[best_j] - t_orig)
                recon_used[best_j] = True

        tp = len(matched_orig_idx)
        return {
            "matched_original_idx": np.array(matched_orig_idx),
            "matched_reconstructed_idx": np.array(matched_recon_idx),
            "timing_errors": np.array(timing_errors),
            "true_positives": tp,
            "false_positives": n_recon - tp,
            "false_negatives": n_orig - tp,
        }

    def _compute_all_correlograms(
        self,
        data: EvalData,
        max_lag_ms: float,
        bin_size_ms: float,
    ) -> np.ndarray:
        """Compute all correlograms (ACG + CCG).

        Dispatches to spike-times algorithm when spike times are available or
        when binned SPIKES can be converted to spike times (fine bins ≤ 1ms).
        Falls back to discrete cross-correlation of binned counts for RATES.

        Returns:
            np.ndarray of shape (n_units, n_units, n_bins).
                correlograms[i, i, :] = ACG for unit i
                correlograms[i, j, :] = CCG for units i and j
        """
        if data.spike_times is not None:
            return self._compute_all_correlograms_from_spike_times(data, max_lag_ms, bin_size_ms)

        # Convert binned SPIKES to spike times for the faster algorithm
        if data.data_type == DataType.SPIKES and data.bin_size_seconds is not None:
            from .converters import FINE_BIN_THRESHOLD_SECONDS, DataConverter

            if data.bin_size_seconds <= FINE_BIN_THRESHOLD_SECONDS:
                converter = DataConverter()
                data_as_spike_times = converter.convert(data, DataType.SPIKE_TIMES)
                return self._compute_all_correlograms_from_spike_times(
                    data_as_spike_times, max_lag_ms, bin_size_ms
                )

        return self._compute_all_correlograms_from_binned(data, max_lag_ms, bin_size_ms)

    def _compute_all_correlograms_from_spike_times(
        self,
        data: EvalData,
        max_lag_ms: float,
        bin_size_ms: float,
    ) -> np.ndarray:
        """Compute correlograms from spike times using the optimized Phy algorithm."""
        spike_times = self._require_spike_times(data)

        max_lag_seconds = max_lag_ms / 1000.0
        bin_size_seconds = bin_size_ms / 1000.0

        # Prepare data: concatenate all spike times and create unit indices
        all_spike_times = []
        all_unit_indices = []

        for unit_idx, st in enumerate(spike_times):
            if len(st) > 0:
                all_spike_times.append(st)
                all_unit_indices.extend([unit_idx] * len(st))

        if len(all_spike_times) == 0:
            # Return empty correlograms using the same symmetric lag geometry
            # as the non-empty path: bins from -max_lag to +max_lag inclusive.
            sampling_rate = ACG_SAMPLING_RATE
            window_size_samples = int(max_lag_seconds * sampling_rate)
            bin_size_samples = int(bin_size_seconds * sampling_rate)
            n_bins, _ = _compute_num_bins(window_size_samples, bin_size_samples)
            return np.zeros((len(spike_times), len(spike_times), n_bins), dtype="int64")

        all_spike_times = np.concatenate(all_spike_times)
        all_unit_indices = np.array(all_unit_indices, dtype=int)

        # Sort by spike times (must be done before remapping)
        sort_idx = np.argsort(all_spike_times)
        all_spike_times = all_spike_times[sort_idx]
        all_unit_indices = all_unit_indices[sort_idx]

        # Create mapping from original unit indices to 0-indexed consecutive indices
        unique_units = np.unique(all_unit_indices)
        unit_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_units)}
        all_unit_indices_remapped = np.array(
            [unit_mapping[idx] for idx in all_unit_indices], dtype=int
        )

        # Convert to samples (the function expects integer sample indices)
        sampling_rate = ACG_SAMPLING_RATE
        spike_times_samples = (all_spike_times * sampling_rate).astype(np.int64)
        window_size_samples = int(max_lag_seconds * sampling_rate)
        bin_size_samples = int(bin_size_seconds * sampling_rate)

        # Compute all correlograms efficiently
        correlograms_compact = _compute_ccgs(
            spike_times_samples, all_unit_indices_remapped, window_size_samples, bin_size_samples
        )

        # Remap back to full matrix (including units with no spikes)
        n_units = len(spike_times)
        n_bins = correlograms_compact.shape[2]
        correlograms = np.zeros((n_units, n_units, n_bins), dtype="int64")

        for i, orig_i in enumerate(unique_units):
            for j, orig_j in enumerate(unique_units):
                correlograms[orig_i, orig_j, :] = correlograms_compact[i, j, :]

        return correlograms

    def _prepare_binned_fft(
        self,
        data: EvalData,
        max_lag_ms: float,
        bin_size_ms: float,
    ) -> tuple[np.ndarray, int, int, float]:
        """Shared binned-path setup: get counts, adjust bin size, rebin, pad, forward FFT.

        Returns:
            F: Forward FFT of counts, shape (n_units, nfft//2+1).
            nfft: Padded FFT length (power of 2).
            max_lag_bins: Number of lag bins on each side of center.
            effective_bin_size_ms: Actual bin size used (may be coarser than requested).
        """
        counts, base_bin_size = self._get_binned_counts_base(data)
        base_bin_size_ms = base_bin_size * 1000.0

        # Auto-adjust bin_size_ms if finer than data resolution
        effective_bin_size_ms = bin_size_ms
        if bin_size_ms < base_bin_size_ms - 1e-6:
            global _warned_correlogram_resolution  # noqa: PLW0603
            if not _warned_correlogram_resolution:
                _warned_correlogram_resolution = True
                warnings.warn(
                    f"Requested correlogram bin_size_ms={bin_size_ms} is finer than "
                    f"data bin size ({base_bin_size_ms:.2f}ms). "
                    f"Using {base_bin_size_ms:.2f}ms instead.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            effective_bin_size_ms = base_bin_size_ms

        # Rebin counts if correlogram bin size > data bin size
        target_bin_size_seconds = effective_bin_size_ms / 1000.0
        if not np.isclose(target_bin_size_seconds, base_bin_size):
            counts = self._rebin_counts(counts, base_bin_size, target_bin_size_seconds)

        max_lag_bins = int(max_lag_ms / effective_bin_size_ms)
        n_time_bins = counts.shape[1]
        counts = counts.astype(np.float64)

        # Zero-pad to power-of-2 >= 2*T for linear (non-circular) correlation.
        nfft = 1
        while nfft < 2 * n_time_bins:
            nfft *= 2

        F = np.fft.rfft(counts, n=nfft, axis=1)  # (n_units, nfft//2+1)
        return F, nfft, max_lag_bins, effective_bin_size_ms

    def _compute_all_correlograms_from_binned(
        self,
        data: EvalData,
        max_lag_ms: float,
        bin_size_ms: float,
    ) -> np.ndarray:
        """Compute correlograms from binned counts or rates via FFT-based cross-correlation.

        Works for RATES and coarse-binned SPIKES. If the requested correlogram
        bin_size_ms is finer than the data bin size, it is auto-adjusted upward
        with a warning.

        Uses batch FFT: one forward FFT for all units, then per-row vectorized
        cross-spectral multiply + inverse FFT.
        """
        F, nfft, max_lag_bins, _ = self._prepare_binned_fft(data, max_lag_ms, bin_size_ms)
        n_units = F.shape[0]
        n_corr_bins = 2 * max_lag_bins + 1

        correlograms = np.zeros((n_units, n_units, n_corr_bins), dtype=np.float64)

        for i in range(n_units):
            # Cross-spectra: unit i vs units i..n_units-1 (vectorized)
            cross = F[i : i + 1] * np.conj(F[i:])  # (n_units-i, nfft//2+1)
            corr_full = np.fft.irfft(cross, n=nfft, axis=1)  # (n_units-i, nfft)

            # Extract lags [-max_lag_bins, ..., max_lag_bins].
            # FFT output: lag k at index k (k>=0), lag -k at index nfft-k (k>0).
            pos = corr_full[:, : max_lag_bins + 1]  # lags 0..max_lag_bins
            neg = corr_full[:, nfft - max_lag_bins :]  # lags -max_lag_bins..-1
            window = np.concatenate([neg, pos], axis=1)  # (n_units-i, n_corr_bins)

            # Upper triangle (including diagonal)
            correlograms[i, i:, :] = window
            # Lower triangle (time-reversed)
            if i + 1 < n_units:
                correlograms[i + 1 :, i, :] = window[1:, ::-1]

        # Zero center bin for ACGs (self-correlation at lag 0)
        di = np.arange(n_units)
        correlograms[di, di, max_lag_bins] = 0

        return correlograms

    def _extract_acgs(
        self, all_correlograms: np.ndarray, smoothing_kernel: int
    ) -> list[np.ndarray]:
        """Extract ACGs from the full correlogram matrix and apply smoothing."""
        n_units = all_correlograms.shape[0]
        center = all_correlograms.shape[2] // 2
        acgs = []

        for i in range(n_units):
            acg = all_correlograms[i, i, :].astype(np.float64)
            acg = self._smooth_boxcar(acg, smoothing_kernel)
            # Re-zero the center bin (lag=0) after smoothing so the
            # self-correlation peak doesn't leak back in.
            acg[center] = 0.0
            acgs.append(acg)

        return acgs

    @staticmethod
    def _select_pairs(n_units: int, max_pairs: int | None) -> list[tuple[int, int]]:
        """Select upper-triangle unit pairs with deterministic subsampling.

        Uses seed=42 for reproducibility.

        Returns:
            List of (i, j) tuples with i < j.
        """
        pairs = [(i, j) for i in range(n_units) for j in range(i + 1, n_units)]
        if max_pairs is not None and len(pairs) > max_pairs:
            rng = np.random.default_rng(42)
            pairs = rng.choice(pairs, max_pairs, replace=False).tolist()
        return pairs

    def _extract_ccgs(
        self, all_correlograms: np.ndarray, max_pairs: int | None
    ) -> dict[tuple[int, int], np.ndarray]:
        """
        Extract CCGs from the full correlogram matrix.
        Optionally subsample pairs if max_pairs is set.
        """
        n_units = all_correlograms.shape[0]
        pairs = self._select_pairs(n_units, max_pairs)

        ccgs = {}
        for i, j in pairs:
            ccgs[(i, j)] = all_correlograms[i, j, :].astype(np.float64)

        return ccgs

    # ------------------------------------------------------------------
    # Direct ACG / CCG computation (bypass full N×N matrix)
    # ------------------------------------------------------------------

    def _compute_acgs_direct(
        self,
        data: EvalData,
        max_lag_ms: float,
        bin_size_ms: float,
        smoothing_kernel: int,
    ) -> list[np.ndarray]:
        """Compute autocorrelograms without building the full N×N matrix.

        Dispatches to spike-times or binned path depending on data availability.
        For the binned path, computes auto-power spectra only: O(N × T log T).
        For the spike-times path, runs Phy per unit: O(sum_i S_i × K_i).
        """
        # 3-way dispatch (same logic as _compute_all_correlograms)
        if data.spike_times is not None:
            return self._compute_acgs_from_spike_times(
                data, max_lag_ms, bin_size_ms, smoothing_kernel
            )

        if data.data_type == DataType.SPIKES and data.bin_size_seconds is not None:
            from .converters import FINE_BIN_THRESHOLD_SECONDS, DataConverter

            if data.bin_size_seconds <= FINE_BIN_THRESHOLD_SECONDS:
                converter = DataConverter()
                data_as_spike_times = converter.convert(data, DataType.SPIKE_TIMES)
                return self._compute_acgs_from_spike_times(
                    data_as_spike_times, max_lag_ms, bin_size_ms, smoothing_kernel
                )

        return self._compute_acgs_from_binned(data, max_lag_ms, bin_size_ms, smoothing_kernel)

    def _compute_acgs_from_spike_times(
        self,
        data: EvalData,
        max_lag_ms: float,
        bin_size_ms: float,
        smoothing_kernel: int,
    ) -> list[np.ndarray]:
        """Compute ACGs from spike times using per-unit Phy algorithm."""
        spike_times_list = self._require_spike_times(data)

        max_lag_seconds = max_lag_ms / 1000.0
        bin_size_seconds = bin_size_ms / 1000.0
        sampling_rate = ACG_SAMPLING_RATE
        window_samples = int(max_lag_seconds * sampling_rate)
        bin_samples = int(bin_size_seconds * sampling_rate)
        num_bins, _ = _compute_num_bins(window_samples, bin_samples)

        acgs: list[np.ndarray] = []
        for st in spike_times_list:
            if len(st) < 2:
                acg = np.zeros(num_bins, dtype=np.float64)
            else:
                spike_samples = (st * sampling_rate).astype(np.int64)
                unit_indices = np.zeros(len(spike_samples), dtype=int)
                result = _compute_ccgs(spike_samples, unit_indices, window_samples, bin_samples)
                acg = result[0, 0, :].astype(np.float64)

            acg = self._smooth_boxcar(acg, smoothing_kernel)
            center = num_bins // 2
            acg[center] = 0.0
            acgs.append(acg)

        return acgs

    def _compute_acgs_from_binned(
        self,
        data: EvalData,
        max_lag_ms: float,
        bin_size_ms: float,
        smoothing_kernel: int,
    ) -> list[np.ndarray]:
        """Compute ACGs from binned/rates data via vectorized auto-power FFT.

        O(N × T log T) — no cross-unit computation needed.
        """
        F, nfft, max_lag_bins, _ = self._prepare_binned_fft(data, max_lag_ms, bin_size_ms)
        n_units = F.shape[0]

        # Auto-power spectra: F[i] * conj(F[i]) element-wise, no broadcasting
        auto_power = F * np.conj(F)  # (n_units, nfft//2+1)
        corr_full = np.fft.irfft(auto_power, n=nfft, axis=1)  # (n_units, nfft)

        # Extract lag window [-max_lag_bins, ..., +max_lag_bins]
        pos = corr_full[:, : max_lag_bins + 1]  # lags 0..max_lag_bins
        neg = corr_full[:, nfft - max_lag_bins :]  # lags -max_lag_bins..-1
        window = np.concatenate([neg, pos], axis=1)  # (n_units, n_corr_bins)

        # Zero center bin (lag=0 self-count)
        window[:, max_lag_bins] = 0

        acgs: list[np.ndarray] = []
        for i in range(n_units):
            acg = window[i].copy()
            acg = self._smooth_boxcar(acg, smoothing_kernel)
            acg[max_lag_bins] = 0.0  # re-zero after smoothing
            acgs.append(acg)

        return acgs

    def _compute_ccgs_direct(
        self,
        data: EvalData,
        max_lag_ms: float,
        bin_size_ms: float,
        max_pairs: int | None,
    ) -> dict[tuple[int, int], np.ndarray]:
        """Compute cross-correlograms for selected pairs only.

        Falls back to full matrix when max_pairs is None or >= total pairs
        (no benefit from selective computation).
        For the binned path: O((N + P) × T log T) where P = number of pairs.
        For the spike-times path: runs Phy per selected pair.
        """
        n_units = data.n_units
        if n_units is None:
            raise ValueError("Cannot determine n_units from data.")
        total_pairs = n_units * (n_units - 1) // 2

        # Fall back to full matrix when all pairs are needed
        if max_pairs is None or max_pairs >= total_pairs:
            all_corr = self._compute_all_correlograms(data, max_lag_ms, bin_size_ms)
            return self._extract_ccgs(all_corr, max_pairs=max_pairs)

        selected_pairs = self._select_pairs(n_units, max_pairs)

        # 3-way dispatch (same logic as _compute_all_correlograms)
        if data.spike_times is not None:
            return self._compute_ccgs_from_spike_times(
                data, max_lag_ms, bin_size_ms, selected_pairs
            )

        if data.data_type == DataType.SPIKES and data.bin_size_seconds is not None:
            from .converters import FINE_BIN_THRESHOLD_SECONDS, DataConverter

            if data.bin_size_seconds <= FINE_BIN_THRESHOLD_SECONDS:
                converter = DataConverter()
                data_as_spike_times = converter.convert(data, DataType.SPIKE_TIMES)
                return self._compute_ccgs_from_spike_times(
                    data_as_spike_times, max_lag_ms, bin_size_ms, selected_pairs
                )

        return self._compute_ccgs_from_binned(data, max_lag_ms, bin_size_ms, selected_pairs)

    def _compute_ccgs_from_spike_times(
        self,
        data: EvalData,
        max_lag_ms: float,
        bin_size_ms: float,
        selected_pairs: list[tuple[int, int]],
    ) -> dict[tuple[int, int], np.ndarray]:
        """Compute CCGs from spike times for selected pairs using per-pair Phy."""
        spike_times_list = self._require_spike_times(data)

        max_lag_seconds = max_lag_ms / 1000.0
        bin_size_seconds = bin_size_ms / 1000.0
        sampling_rate = ACG_SAMPLING_RATE
        window_samples = int(max_lag_seconds * sampling_rate)
        bin_samples = int(bin_size_seconds * sampling_rate)
        num_bins, _ = _compute_num_bins(window_samples, bin_samples)

        ccgs: dict[tuple[int, int], np.ndarray] = {}
        for i, j in selected_pairs:
            st_i, st_j = spike_times_list[i], spike_times_list[j]
            if len(st_i) == 0 or len(st_j) == 0:
                ccgs[(i, j)] = np.zeros(num_bins, dtype=np.float64)
                continue

            # Merge two units and run Phy on a 2-unit problem
            merged = np.concatenate([st_i, st_j])
            labels = np.concatenate(
                [np.zeros(len(st_i), dtype=int), np.ones(len(st_j), dtype=int)]
            )
            sort_idx = np.argsort(merged)
            merged = merged[sort_idx]
            labels = labels[sort_idx]

            spike_samples = (merged * sampling_rate).astype(np.int64)
            result = _compute_ccgs(spike_samples, labels, window_samples, bin_samples)
            ccgs[(i, j)] = result[0, 1, :].astype(np.float64)

        return ccgs

    def _compute_ccgs_from_binned(
        self,
        data: EvalData,
        max_lag_ms: float,
        bin_size_ms: float,
        selected_pairs: list[tuple[int, int]],
    ) -> dict[tuple[int, int], np.ndarray]:
        """Compute CCGs from binned/rates data for selected pairs via FFT.

        O((N + P) × T log T) — forward FFT is O(N × T log T), pair computation
        is O(P × T log T).
        """
        F, nfft, max_lag_bins, _ = self._prepare_binned_fft(data, max_lag_ms, bin_size_ms)

        # Group pairs by first index for vectorized cross-spectra
        pairs_by_i: dict[int, list[int]] = {}
        for i, j in selected_pairs:
            pairs_by_i.setdefault(i, []).append(j)

        ccgs: dict[tuple[int, int], np.ndarray] = {}
        for i, js in pairs_by_i.items():
            j_indices = np.array(js)
            # Vectorized cross-spectrum: F[i] * conj(F[js])
            cross = F[i : i + 1] * np.conj(F[j_indices])  # (len(js), nfft//2+1)
            corr_full = np.fft.irfft(cross, n=nfft, axis=1)  # (len(js), nfft)

            # Extract lag window
            pos = corr_full[:, : max_lag_bins + 1]
            neg = corr_full[:, nfft - max_lag_bins :]
            window = np.concatenate([neg, pos], axis=1)

            for k, j in enumerate(js):
                ccgs[(i, j)] = window[k].astype(np.float64)

        return ccgs

    # ------------------------------------------------------------------ #
    # Phy-compatible FFT correlogram path.
    #
    # Strategy: bin to 1 ms (the per-spike-time resolution), FFT-correlate
    # per pair, then coarsen the resulting 1 ms-lag correlogram to the
    # requested ``bin_size_ms`` by summing groups of consecutive 1 ms lag
    # bins under Phy's left-edge ``floor(delay / bin_size_ms)`` convention.
    # The rightmost positive coarse bin (lag = +num_half_coarse * bin_size_ms)
    # contains only the single 1 ms entry at lag = +max_lag_ms — Phy's mask
    # uses Python's ``floor(-delay / bin_size_ms)`` which drops any delay
    # strictly greater than ``num_half_coarse * bin_size_ms`` from both sign
    # passes. Match by clipping the rightmost coarse bin to width 1.
    #
    # Synthetic + real-data validation against the existing all-pairs Phy
    # kernel: bit-identical correlogram values across all four robustness_fast_eval
    # sessions × {1 ms / 100 ms lag, 5 ms / 500 ms, 10 ms / 1000 ms}, with
    # max_abs_diff <= 6e-14 (float64 round-off only).
    # ------------------------------------------------------------------ #

    def _compute_correlograms_phy_compat_fft(
        self,
        data: EvalData,
        max_lag_ms: float,
        bin_size_ms: float,
        selected_pairs: list[tuple[int, int]],
    ) -> dict[tuple[int, int], np.ndarray]:
        """Per-pair correlogram via 1 ms-FFT + Phy-compatible coarsening.

        Returns ``{(i, j): correlogram_of_shape_(2*num_half_coarse+1,)}``.
        Bit-identical to the Phy spike-times kernel (modulo float round-off);
        cost is ``O((N + P) * T_1ms * log T_1ms)`` independent of spike density.
        """
        f = int(round(bin_size_ms))
        if not np.isclose(bin_size_ms, f) or f < 1:
            raise ValueError(
                f"Phy-compat FFT requires integer-ms bin_size_ms >= 1; got {bin_size_ms}."
            )
        num_half_coarse = int(round(max_lag_ms / bin_size_ms))
        if not np.isclose(num_half_coarse * bin_size_ms, max_lag_ms):
            raise ValueError(
                f"Phy-compat FFT requires max_lag_ms to be a multiple of "
                f"bin_size_ms; got max_lag_ms={max_lag_ms}, bin_size_ms={bin_size_ms}."
            )

        # Always reduce to 1 ms-binned counts. The 1 ms granularity makes
        # bin-then-correlate equivalent to Phy's diff-then-bin (since
        # integer-ms spike times have no sub-bin ambiguity at 1 ms).
        if data.spike_times is not None:
            # SPIKE_TIMES path: bin spike times directly to 1 ms counts.
            # +1 headroom on n_time_bins so a spike sitting exactly at
            # t == duration (which converts to bin_idx = int(duration*1000))
            # is NOT clipped down a bin — matching Phy's spike-times kernel,
            # which doesn't drop boundary spikes.
            duration = self._get_duration(data)
            n_time_bins = int(np.ceil(duration * 1000.0)) + 1
            counts = np.zeros((len(data.spike_times), n_time_bins), dtype=np.float64)
            for i, st in enumerate(data.spike_times):
                if len(st) == 0:
                    continue
                bin_idx = (np.asarray(st) * 1000.0).astype(np.int64)
                np.clip(bin_idx, 0, n_time_bins - 1, out=bin_idx)
                np.add.at(counts[i], bin_idx, 1)
        else:
            counts, base_bin_size = self._get_binned_counts_base(data)
            if base_bin_size > 0.001 and not np.isclose(base_bin_size, 0.001):
                # Caller should have routed us via `_has_spike_times_path` to
                # the direct path; trip a clear error rather than silently
                # invoke `_rebin_counts` (which would raise "target >= source").
                raise ValueError(
                    "Phy-compat FFT requires <= 1ms-binned input; caller of "
                    "_compute_correlograms_phy_compat_fft should gate on "
                    "_has_spike_times_path(data). "
                    f"Got base_bin_size={base_bin_size}s."
                )
            if not np.isclose(base_bin_size, 0.001):
                counts = self._rebin_counts(counts, base_bin_size, 0.001)
            counts = np.asarray(counts, dtype=np.float64)
        n_units, n_time_bins = counts.shape
        max_lag_1ms = int(max_lag_ms)  # NO right-edge extension; see class docstring above.

        # FFT cross-spectra per requested (i, j).
        nfft = 1
        while nfft < 2 * n_time_bins:
            nfft *= 2
        F = np.fft.rfft(counts, n=nfft, axis=1)  # (n_units, nfft//2+1)

        pairs_by_i: dict[int, list[int]] = {}
        for i, j in selected_pairs:
            pairs_by_i.setdefault(i, []).append(j)

        out: dict[tuple[int, int], np.ndarray] = {}
        for i, js in pairs_by_i.items():
            j_arr = np.asarray(js, dtype=np.int64)
            cross = F[i : i + 1] * np.conj(F[j_arr])
            corr_full = np.fft.irfft(cross, n=nfft, axis=1)
            pos = corr_full[:, : max_lag_1ms + 1]
            neg = corr_full[:, nfft - max_lag_1ms :]
            c1ms = np.concatenate([neg, pos], axis=1)  # (len(js), 2*max_lag_1ms+1)

            if f == 1:
                coarse = c1ms.astype(np.float64)
            else:
                n_full = 2 * num_half_coarse
                full_section = c1ms[:, : n_full * f]
                coarse_full = full_section.reshape(len(js), n_full, f).sum(axis=2)
                rightmost = c1ms[:, 2 * max_lag_1ms : 2 * max_lag_1ms + 1]
                coarse = np.concatenate([coarse_full, rightmost], axis=1)

            # Round to nearest integer to absorb FFT/IFFT round-off (~1e-16).
            # Safe because every input case to this function is integer-valued
            # counts: spike-times binned to 1ms produces integer counts, and
            # SPIKES at <= 1ms (the only other branch) carries integer counts
            # too. RATES/coarse-binned inputs are routed to _compute_acgs_direct
            # by the outer dispatch, never here. Rounding restores bit-identity
            # with the Phy spike-times kernel (which produces int64 counts).
            coarse = np.round(coarse)

            for k, j in enumerate(js):
                c = coarse[k].astype(np.float64)
                if int(j) == i:
                    c[num_half_coarse] = 0.0
                out[(i, int(j))] = c

        return out

    def _compute_acgs_phy_compat_fft(
        self,
        data: EvalData,
        max_lag_ms: float,
        bin_size_ms: float,
        smoothing_kernel: int,
    ) -> list[np.ndarray]:
        """ACG via Phy-compat FFT (per-unit). Mirrors ``_compute_acgs_from_binned``
        in shape + post-smoothing center-bin re-zero, but uses 1 ms-FFT +
        Phy-compat coarsening so the output matches the existing all-pairs
        Phy kernel bit-for-bit (instead of bin-then-correlate's lossy
        approximation that ``_compute_acgs_from_binned`` produces at
        ``bin_size_ms`` > 1).
        """
        n_units = data.n_units
        if n_units is None:
            raise ValueError("Cannot determine n_units from data.")
        diag_pairs = [(i, i) for i in range(n_units)]
        pairwise = self._compute_correlograms_phy_compat_fft(
            data, max_lag_ms, bin_size_ms, diag_pairs
        )
        num_half_coarse = int(round(max_lag_ms / bin_size_ms))
        acgs: list[np.ndarray] = []
        for i in range(n_units):
            acg = pairwise.get((i, i))
            if acg is None:
                n_bins = 2 * num_half_coarse + 1
                acg = np.zeros(n_bins, dtype=np.float64)
            acg = self._smooth_boxcar(acg, smoothing_kernel)
            acg[num_half_coarse] = 0.0  # re-zero center after smoothing
            acgs.append(acg)
        return acgs

    def _compute_ccgs_phy_compat_fft(
        self,
        data: EvalData,
        max_lag_ms: float,
        bin_size_ms: float,
        max_pairs: int | None,
    ) -> dict[tuple[int, int], np.ndarray]:
        """CCG via Phy-compat FFT for the deterministically-selected ``max_pairs``
        off-diagonal unit pairs (same selection used by the all-pairs Phy path).
        """
        n_units = data.n_units
        if n_units is None:
            raise ValueError("Cannot determine n_units from data.")
        selected_pairs = self._select_pairs(n_units, max_pairs)
        return self._compute_correlograms_phy_compat_fft(
            data, max_lag_ms, bin_size_ms, selected_pairs
        )


def _compute_ccgs(spike_times, spike_unit_indices, window_size, bin_size):
    """
    A very well optimized algorithm for the cross-correlation of
    spike trains, copied from the Phy package, written by Cyrille Rossant.

    Adapted from SpikeInterface's port of the Phy code. Phy is BSD-3-Clause and
    SpikeInterface is MIT (see ``LICENSES/phy-BSD-3-Clause.txt`` and
    ``LICENSES/spikeinterface-MIT.txt``).

    Parameters
    ----------
    spike_times : np.ndarray
        An array of spike times (in samples, not seconds).
        This contains spikes from all units.
    spike_unit_indices : np.ndarray
        An array of labels indicating the unit of the corresponding
        spike in `spike_times`.
    window_size : int
        The window size over which to perform the cross-correlation, in samples
    bin_size : int
        The size of which to bin lags, in samples.

    Returns
    -------
    correlograms : np.array
        A (num_units, num_units, num_bins) array of correlograms
        between all units at each lag time bin.

    Notes
    -----
    For all spikes, time difference between this spike and
    every other spike within the window is directly computed
    and stored as a count in the relevant lag time bin.

    Initially, the spike_times array is shifted by 1 position, and the difference
    computed. This gives the time differences between the closest spikes
    (skipping the zero-lag case). Next, the differences between
    spikes times in samples are converted into units relative to
    bin_size ('binarized'). Spikes in which the binarized difference to
    their closest neighbouring spike is greater than half the bin-size are
    masked.

    Finally, the indices of the (num_units, num_units, num_bins) correlogram
    that need incrementing are done so with `ravel_multi_index()`. This repeats
    for all shifts along the spike_train until no spikes have a corresponding
    match within the window size.
    """
    num_bins, num_half_bins = _compute_num_bins(window_size, bin_size)
    num_units = len(np.unique(spike_unit_indices))

    correlograms = np.zeros((num_units, num_units, num_bins), dtype="int64")

    # At a given shift, the mask precises which spikes have matching spikes
    # within the correlogram time window.
    mask = np.ones_like(spike_times, dtype="bool")

    # The loop continues as long as there is at least one
    # spike with a matching spike.
    shift = 1
    while mask[:-shift].any():
        # Number of time samples between spike i and spike i+shift.
        spike_diff = spike_times[shift:] - spike_times[:-shift]

        for sign in (-1, 1):
            # Binarize the delays between spike i and spike i+shift for negative and positive
            # the operator // is np.floor_divide
            spike_diff_b = (spike_diff * sign) // bin_size

            # Spikes with no matching spikes are masked.
            if sign == -1:
                mask[:-shift][spike_diff_b < -num_half_bins] = False
            else:
                # Changed from >= to > to include the positive boundary (+num_half_bins)
                mask[:-shift][spike_diff_b > num_half_bins] = False

            m = mask[:-shift]

            # Find the indices in the raveled correlograms array that need
            # to be incremented, taking into account the spike unit labels.
            if sign == 1:
                indices = np.ravel_multi_index(
                    (
                        spike_unit_indices[+shift:][m],
                        spike_unit_indices[:-shift][m],
                        spike_diff_b[m] + num_half_bins,
                    ),
                    correlograms.shape,
                )
            else:
                indices = np.ravel_multi_index(
                    (
                        spike_unit_indices[:-shift][m],
                        spike_unit_indices[+shift:][m],
                        spike_diff_b[m] + num_half_bins,
                    ),
                    correlograms.shape,
                )

            # Increment the matching spikes in the correlograms array.
            # scatter-add: touches only the |indices| cells that actually need to
            # change, rather than allocating a length-max(indices)+1 bincount buffer
            # (~14.8 MB for typical 96×96×201 correlograms) and reading/writing the
            # full buffer per shift. Roughly 27× faster on 64×1 s windows;
            # bit-identical output.
            np.add.at(correlograms.ravel(), indices, 1)

        shift += 1

        # Remove ACG peaks (perfectly correlated with themselves)
        correlograms[np.arange(num_units), np.arange(num_units), num_half_bins] = 0

    return correlograms


def _compute_num_bins(window_size, bin_size):
    """
    Internal function to compute number of bins, expects
    window_size and bin_size are already divisible. These are
    typically generated in `_make_bins()`.

    Returns
    -------
    num_bins : int
        The total number of bins to span the window, in samples
    half_num_bins : int
        Half the number of bins. The bins are an equal number
        of bins that look forward and backwards from zero, e.g.
        [..., -10 to -5, -5 to 0, 0 to 5, 5 to 10, ...]

    """
    num_half_bins = int(window_size // bin_size)
    # Add 1 to make the autocorrelograms symmetric: bins from -num_half_bins to +num_half_bins
    num_bins = int(2 * num_half_bins + 1)

    return num_bins, num_half_bins
