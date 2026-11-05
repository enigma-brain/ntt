"""Tests for data type converters."""

import numpy as np
import pytest

from ntt.evaluation.core.converters import DataConverter
from ntt.evaluation.core.types import DataType, EvalData


class TestPoissonConverter:
    """Test Poisson sampling conversion from RATES to SPIKE_TIMES."""

    def _make_rates_data(
        self,
        rate_hz: float = 50.0,
        n_units: int = 2,
        duration_s: float = 1.0,
        bin_size_s: float = 0.005,
    ) -> EvalData:
        n_bins = int(duration_s / bin_size_s)
        rates = np.full((n_units, n_bins), rate_hz, dtype=np.float32)
        return EvalData(
            rates_hz=rates,
            bin_size_seconds=bin_size_s,
            duration=duration_s,
            n_units=n_units,
        )

    def test_disabled_by_default(self):
        """can_convert(RATES, SPIKE_TIMES) returns False by default."""
        converter = DataConverter()
        assert not converter.can_convert(
            DataType.RATES,
            DataType.SPIKE_TIMES,
            source_bin_size_seconds=0.005,
        )

    def test_opt_in(self):
        """With allow_stochastic_conversion=True, can_convert returns True."""
        converter = DataConverter(allow_stochastic_conversion=True)
        assert converter.can_convert(
            DataType.RATES,
            DataType.SPIKE_TIMES,
            source_bin_size_seconds=0.005,
        )

    def test_produces_spike_times(self):
        """Converting RATES to SPIKE_TIMES produces valid spike_times data."""
        converter = DataConverter(allow_stochastic_conversion=True, stochastic_seed=42)
        data = self._make_rates_data(rate_hz=100.0, n_units=3, duration_s=1.0)

        result = converter.convert(data, DataType.SPIKE_TIMES)

        assert result.data_type == DataType.SPIKE_TIMES
        assert result.spike_times is not None
        assert len(result.spike_times) == 3
        assert result.duration == 1.0

        # Should have approximately 100 spikes per unit (100 Hz * 1s)
        for st in result.spike_times:
            assert len(st) > 50, "Should have a reasonable number of spikes"
            assert len(st) < 200, "Should not have too many spikes"
            # Spikes should be sorted
            assert np.all(np.diff(st) >= 0), "Spike times should be sorted"
            # Spikes should be within [0, duration)
            assert st[0] >= 0
            assert st[-1] < 1.0

    def test_deterministic_with_seed(self):
        """Same seed produces same result."""
        data = self._make_rates_data(rate_hz=50.0)

        converter1 = DataConverter(allow_stochastic_conversion=True, stochastic_seed=42)
        result1 = converter1.convert(data, DataType.SPIKE_TIMES)

        converter2 = DataConverter(allow_stochastic_conversion=True, stochastic_seed=42)
        result2 = converter2.convert(data, DataType.SPIKE_TIMES)

        for st1, st2 in zip(result1.spike_times, result2.spike_times, strict=True):
            np.testing.assert_array_equal(st1, st2)

    def test_different_seeds_differ(self):
        """Different seeds produce different results."""
        data = self._make_rates_data(rate_hz=50.0)

        converter1 = DataConverter(allow_stochastic_conversion=True, stochastic_seed=42)
        result1 = converter1.convert(data, DataType.SPIKE_TIMES)

        converter2 = DataConverter(allow_stochastic_conversion=True, stochastic_seed=123)
        result2 = converter2.convert(data, DataType.SPIKE_TIMES)

        # At least one unit should differ
        any_differ = any(
            len(st1) != len(st2) or not np.allclose(st1, st2)
            for st1, st2 in zip(result1.spike_times, result2.spike_times, strict=True)
        )
        assert any_differ, "Different seeds should produce different spike times"

    def test_zero_rate_no_spikes(self):
        """Zero rate produces no spikes."""
        converter = DataConverter(allow_stochastic_conversion=True, stochastic_seed=42)
        data = self._make_rates_data(rate_hz=0.0, n_units=1)

        result = converter.convert(data, DataType.SPIKE_TIMES)

        assert len(result.spike_times[0]) == 0

    def test_raises_without_opt_in(self):
        """Converting RATES to SPIKE_TIMES without opt-in raises ConversionError."""
        from ntt.evaluation.core.converters import ConversionError

        converter = DataConverter()  # default: allow_stochastic_conversion=False
        data = self._make_rates_data()

        with pytest.raises(ConversionError):
            converter.convert(data, DataType.SPIKE_TIMES)


def test_spike_times_to_rates_uses_true_final_bin_width() -> None:
    """The final partial bin should be normalized by its actual duration."""
    converter = DataConverter()
    data = EvalData(
        spike_times=[np.array([0.02, 0.09], dtype=np.float64)],
        duration=0.095,
        n_units=1,
        data_type=DataType.SPIKE_TIMES,
    )

    result = converter.convert(data, DataType.RATES, target_bin_size_seconds=0.01)

    assert result.rates_hz is not None
    assert result.rates_hz.shape == (1, 10)
    # Full bin: 1 spike / 10ms = 100 Hz
    assert np.isclose(result.rates_hz[0, 2], 100.0)
    # Final partial bin: 1 spike / 5ms = 200 Hz
    assert np.isclose(result.rates_hz[0, 9], 200.0)
