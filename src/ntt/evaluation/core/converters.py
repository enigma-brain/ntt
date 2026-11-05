"""Data type conversion utilities with bin-size-aware rules."""

import numpy as np

from .types import DataType, EvalData

# Threshold for "fine resolution" bins that can be converted to spike times
FINE_BIN_THRESHOLD_MS = 1.0  # Bins at or below 1ms are considered fine resolution
FINE_BIN_THRESHOLD_SECONDS = FINE_BIN_THRESHOLD_MS / 1000.0


def _require[T](value: T | None, name: str) -> T:
    """Narrow an Optional EvalData field, raising if it is missing."""
    if value is None:
        raise ValueError(f"{name} is required for this conversion")
    return value


class ConversionError(ValueError):
    """Raised when a conversion is not allowed."""

    pass


class DataConverter:
    """
    Converts between neural data representations with bin-size-aware rules.

    Conversion rules:
    - SPIKE_TIMES can convert to anything (lossless)
    - SPIKES (with bin_size <= 1ms) can convert to SPIKE_TIMES (acceptable loss)
    - SPIKES can convert to RATES (lossless rescaling)
    - RATES with coarse bins cannot convert to SPIKES or SPIKE_TIMES (too lossy)
      unless allow_stochastic_conversion is True (Poisson sampling)
    - RATES can convert to lower RATES (rounding to integer rate fractions)
    """

    def __init__(
        self,
        default_bin_size_seconds: float = 0.001,
        allow_stochastic_conversion: bool = False,
        stochastic_seed: int | None = 42,
    ):
        """
        Args:
            default_bin_size_seconds: Default bin size for conversions to binned format
            allow_stochastic_conversion: Allow stochastic conversions (e.g., Poisson
                sampling from RATES to SPIKE_TIMES)
            stochastic_seed: Random seed for stochastic conversions
        """
        self.default_bin_size_seconds = default_bin_size_seconds
        self.allow_stochastic_conversion = allow_stochastic_conversion
        self.stochastic_seed = stochastic_seed

    def can_convert(
        self,
        source_type: DataType,
        target_type: DataType,
        source_bin_size_seconds: float | None = None,
        target_bin_size_seconds: float | None = None,
    ) -> bool:
        """
        Check if conversion is allowed given source bin size.

        Args:
            source_type: Source data type
            target_type: Target data type
            source_bin_size_seconds: Bin size of source data (None for spike times)

        Returns:
            True if conversion is allowed
        """
        if source_type == target_type:
            return True

        # SPIKE_TIMES can convert to anything
        if source_type == DataType.SPIKE_TIMES:
            return True

        # Check bin size for binned sources
        source_bin_ms = (source_bin_size_seconds or 0) * 1000
        is_fine_resolution = source_bin_ms <= FINE_BIN_THRESHOLD_MS

        # SPIKES conversions
        if source_type == DataType.SPIKES:
            if target_type == DataType.RATES:
                return True  # Always allowed (just rescaling)
            if target_type == DataType.SPIKE_TIMES:
                return is_fine_resolution  # Only if fine bins

        # RATES conversions
        if source_type == DataType.RATES:
            if target_type == DataType.RATES:
                return (
                    (source_bin_size_seconds is not None)
                    and (target_bin_size_seconds is not None)
                    and (source_bin_size_seconds <= target_bin_size_seconds)
                )
            if target_type == DataType.SPIKE_TIMES and self.allow_stochastic_conversion:
                return source_bin_size_seconds is not None
            return False

        return False

    def can_convert_data(self, data: EvalData, target_type: DataType) -> bool:
        """Like :meth:`can_convert`, but also allows the data-dependent rule below.

        RATES that hold observed integer spike counts at fine bins (e.g. the original
        data of a tokenizer with ``input_type=RATES``) convert to SPIKE_TIMES like
        binned SPIKES do. Expected counts (non-integer) still need stochastic conversion.
        """
        source_type = _require(data.data_type, "data_type")
        if self.can_convert(source_type, target_type, data.bin_size_seconds):
            return True
        return target_type == DataType.SPIKE_TIMES and self._is_fine_integer_counts(data)

    @staticmethod
    def _is_fine_integer_counts(data: EvalData) -> bool:
        if data.data_type != DataType.RATES or data.bin_size_seconds is None:
            return False
        if data.bin_size_seconds * 1000 > FINE_BIN_THRESHOLD_MS:
            return False
        return data.has_integer_counts()

    def get_conversion_error_message(
        self,
        source_type: DataType,
        target_type: DataType,
        source_bin_size_seconds: float | None = None,
        target_bin_size_seconds: float | None = None,
    ) -> str:
        """Get descriptive error message for disallowed conversion."""
        source_bin_ms = (source_bin_size_seconds or 0) * 1000
        target_bin_ms = (target_bin_size_seconds or 0) * 1000

        if source_type == DataType.RATES:
            return (
                f"Cannot convert from RATES (bin_size={source_bin_ms:.1f}ms) to "
                f"{target_type.value} (bin_size={target_bin_ms:.1f}ms). "
                f"Rate data with coarse bins loses too much temporal information "
                f"for spike-level metrics."
            )

        if source_type == DataType.SPIKES and target_type == DataType.SPIKE_TIMES:
            return (
                f"Cannot convert from SPIKES (bin_size={source_bin_ms:.1f}ms) to SPIKE_TIMES. "
                f"Bin size must be <= {FINE_BIN_THRESHOLD_MS}ms for spike time reconstruction. "
                f"Consider using rate-based metrics instead."
            )

        return f"Conversion from {source_type.value} to {target_type.value} is not supported."

    def convert(
        self,
        data: EvalData,
        target_type: DataType,
        target_bin_size_seconds: float | None = None,
    ) -> EvalData:
        """
        Convert data to target type.

        Args:
            data: Source data
            target_type: Desired output type
            target_bin_size_seconds: Bin size for binned outputs (uses default if None)

        Returns:
            Converted EvalData

        Raises:
            ConversionError: If conversion is not allowed
        """
        if data.data_type == target_type:
            return data

        # Check if conversion is allowed
        source_type = _require(data.data_type, "data_type")
        if not self.can_convert_data(data, target_type):
            raise ConversionError(
                self.get_conversion_error_message(source_type, target_type, data.bin_size_seconds)
            )

        target_bin_size = target_bin_size_seconds or self.default_bin_size_seconds

        # Route to appropriate conversion
        if data.data_type == DataType.SPIKE_TIMES:
            if target_type == DataType.SPIKES:
                return self._spike_times_to_binned(data, target_bin_size)
            elif target_type == DataType.RATES:
                return self._spike_times_to_rates(data, target_bin_size)

        elif data.data_type == DataType.SPIKES:
            if target_type == DataType.SPIKE_TIMES:
                return self._binned_to_spike_times(data)
            elif target_type == DataType.RATES:
                return self._binned_to_rates(data)

        elif data.data_type == DataType.RATES:
            if target_type == DataType.SPIKE_TIMES:
                if self._is_fine_integer_counts(data):
                    return self._binned_to_spike_times(data)
                return self._rates_to_spike_times_poisson(data)
            if target_type == DataType.RATES:
                return self._rebin_rates(data, target_bin_size)

        raise ConversionError(f"Unhandled conversion: {data.data_type} -> {target_type}")

    def _spike_times_to_binned(self, data: EvalData, bin_size: float) -> EvalData:
        """Convert spike times to binary binned spikes."""
        if data.duration is None:
            raise ValueError("Duration required for spike times to binned conversion")
        n_bins = int(np.ceil(data.duration / bin_size))
        binned = np.zeros((_require(data.n_units, "n_units"), n_bins), dtype=np.float32)

        for i, st in enumerate(_require(data.spike_times, "spike_times")):
            if len(st) > 0:
                bin_indices = np.clip((st / bin_size).astype(int), 0, n_bins - 1)
                np.add.at(binned[i], bin_indices, 1)

        # Clip to binary
        binned = np.clip(binned, 0, 1)

        return EvalData(
            binned_spikes=binned,
            bin_size_seconds=bin_size,
            duration=data.duration,
            n_units=data.n_units,
            data_type=DataType.SPIKES,
            latent=data.latent,
        )

    def _spike_times_to_rates(self, data: EvalData, bin_size: float) -> EvalData:
        """Convert spike times to firing rates (Hz)."""
        if data.duration is None:
            raise ValueError("Duration required for spike times to rates conversion")
        n_bins = int(np.ceil(data.duration / bin_size))
        rates = np.zeros((_require(data.n_units, "n_units"), n_bins), dtype=np.float32)

        for i, st in enumerate(_require(data.spike_times, "spike_times")):
            if len(st) > 0:
                bin_indices = np.clip((st / bin_size).astype(int), 0, n_bins - 1)
                np.add.at(rates[i], bin_indices, 1)

        # Convert counts to Hz. If duration is not an exact multiple of bin_size,
        # only the final bin is shorter and needs a separate normalization.
        rates = rates / bin_size
        remainder = data.duration - bin_size * (n_bins - 1)
        if n_bins > 0 and 0 < remainder < bin_size:
            rates[:, -1] *= bin_size / remainder

        return EvalData(
            rates_hz=rates,
            bin_size_seconds=bin_size,
            duration=data.duration,
            n_units=data.n_units,
            data_type=DataType.RATES,
            latent=data.latent,
        )

    # Maximum fraction of non-zero bins allowed for SPIKES → SPIKE_TIMES
    # conversion.  Beyond this threshold the data is too dense for the
    # spike-times representation to be meaningful (and downstream algorithms
    # like the Phy correlogram become catastrophically slow).
    MAX_NONZERO_FRACTION_FOR_SPIKE_TIMES = 0.5

    def _binned_to_spike_times(self, data: EvalData) -> EvalData:
        """
        Convert fine-resolution binary spikes to spike times.

        Places spikes at bin centers. Only valid for fine bins (<=1ms).

        Raises ``ConversionError`` when more than 50% of bins are non-zero,
        which indicates the data is continuous rates rather than sparse
        spikes and the conversion would produce a nonsensical spike train.
        """
        if data.bin_size_seconds is None:
            raise ValueError(
                "Source bin_size_seconds required for binned to spike times conversion"
            )

        # Integer binned_counts (see _is_fine_integer_counts) follow the SPIKES rule:
        # one spike per non-zero bin.
        binned_spikes = (
            data.binned_spikes
            if data.binned_spikes is not None
            else _require(data.binned_counts, "binned_spikes or binned_counts")
        )
        nonzero_frac = float(np.mean(binned_spikes > 0))
        if nonzero_frac > self.MAX_NONZERO_FRACTION_FOR_SPIKE_TIMES:
            raise ConversionError(
                f"Cannot convert SPIKES to SPIKE_TIMES: {nonzero_frac:.1%} of bins "
                f"are non-zero (threshold: "
                f"{self.MAX_NONZERO_FRACTION_FOR_SPIKE_TIMES:.0%}). "
                f"This typically means the reconstruction outputs dense rates "
                f"rather than sparse spikes. Metrics requiring SPIKE_TIMES "
                f"will be skipped for this segment."
            )

        spike_times = []
        bin_size = data.bin_size_seconds

        for i in range(_require(data.n_units, "n_units")):
            spike_bins = np.where(binned_spikes[i] > 0)[0]
            # Place spike at center of bin
            times = (spike_bins + 0.5) * bin_size
            spike_times.append(times.astype(np.float64))

        return EvalData(
            spike_times=spike_times,
            duration=data.duration,
            n_units=data.n_units,
            data_type=DataType.SPIKE_TIMES,
            latent=data.latent,
        )

    def _binned_to_rates(self, data: EvalData) -> EvalData:
        """Convert binary spikes to rates (Hz)."""
        if data.bin_size_seconds is None:
            raise ValueError("Source bin_size_seconds required for binned to rates conversion")
        rates = _require(data.binned_spikes, "binned_spikes") / data.bin_size_seconds

        return EvalData(
            rates_hz=rates.astype(np.float32),
            bin_size_seconds=data.bin_size_seconds,
            duration=data.duration,
            n_units=data.n_units,
            data_type=DataType.RATES,
            latent=data.latent,
        )

    def _rebin_rates(self, data: EvalData, target_bin_size: float) -> EvalData:
        """Rebin rates to different bin size (only coarser bins allowed).

        Rounds to integer multiple of source bin size.
        """
        if data.bin_size_seconds is None:
            raise ValueError("Source bin_size_seconds required for rebinning")
        if target_bin_size < data.bin_size_seconds:
            raise ConversionError(
                f"Cannot rebin rates to finer resolution: "
                f"{data.bin_size_seconds * 1000:.1f}ms -> {target_bin_size * 1000:.1f}ms"
            )

        if np.isclose(target_bin_size, data.bin_size_seconds):
            return data

        ratio = int(round(target_bin_size / data.bin_size_seconds))
        if ratio == 1:
            return data

        # Actual target bin size = integer multiple of original
        actual_target_bin_size = data.bin_size_seconds * ratio

        if data.binned_counts is not None:
            # Per-bin counts rebin by summing and stay counts.
            n_new_bins = data.binned_counts.shape[1] // ratio
            counts = data.binned_counts[:, : n_new_bins * ratio]
            return EvalData(
                binned_counts=counts.reshape(counts.shape[0], n_new_bins, ratio).sum(axis=2),
                bin_size_seconds=actual_target_bin_size,
                duration=n_new_bins * actual_target_bin_size,
                n_units=data.n_units,
                data_type=DataType.RATES,
                latent=data.latent,
            )

        if data.rates_hz is None:
            raise ValueError("rates_hz or binned_counts required for rate rebinning")

        n_new_bins = data.rates_hz.shape[1] // ratio

        # Truncate to fit evenly
        truncated = data.rates_hz[:, : n_new_bins * ratio]

        # Rebin by summing and rescaling
        rebinned = truncated.reshape(_require(data.n_units, "n_units"), n_new_bins, ratio).sum(
            axis=2
        )

        # Rescale to Hz
        rebinned = rebinned / ratio

        return EvalData(
            rates_hz=rebinned.astype(np.float32),
            # bin_size_seconds=target_bin_size,
            # duration=n_new_bins * target_bin_size,
            bin_size_seconds=actual_target_bin_size,
            duration=n_new_bins * actual_target_bin_size,
            n_units=data.n_units,
            data_type=DataType.RATES,
            latent=data.latent,
        )

    def _rates_to_spike_times_poisson(self, data: EvalData) -> EvalData:
        """Convert rates to spike times via Poisson sampling.

        For each time bin, samples a spike count from Poisson(rate * bin_size)
        and distributes spikes uniformly within the bin.
        """
        if data.bin_size_seconds is None:
            raise ValueError("bin_size_seconds required for rates to spike times conversion")
        expected_counts = data.counts_per_bin()
        if expected_counts is None:
            raise ValueError(
                "rates_hz or binned_counts required for Poisson spike time conversion"
            )

        rng = np.random.default_rng(self.stochastic_seed)
        bin_size = data.bin_size_seconds
        n_time_bins = expected_counts.shape[1]
        duration = data.duration or n_time_bins * bin_size

        spike_times_list: list[np.ndarray] = []
        for unit_idx in range(_require(data.n_units, "n_units")):
            unit_spikes: list[float] = []
            for t in range(n_time_bins):
                expected_count = max(0.0, expected_counts[unit_idx, t])
                n_spikes = rng.poisson(expected_count)
                if n_spikes > 0:
                    bin_start = t * bin_size
                    bin_end = (t + 1) * bin_size
                    spike_ts = rng.uniform(bin_start, bin_end, n_spikes)
                    unit_spikes.extend(spike_ts.tolist())
            spike_times_list.append(np.sort(np.array(unit_spikes, dtype=np.float64)))

        return EvalData(
            spike_times=spike_times_list,
            duration=duration,
            n_units=data.n_units,
            data_type=DataType.SPIKE_TIMES,
            latent=data.latent,
        )
