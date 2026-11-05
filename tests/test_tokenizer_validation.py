"""Tests for tokenizer capability validation."""

from typing import Literal

import pytest
import torch

from ntt.contrib.tokenizer.identity_tokenizer import (
    IdentityTokenizer,
    IdentityTokenizerSpec,
)
from ntt.core.components.dist import UniformDist
from ntt.core.components.types import (
    BinnedSpikesInput,
    DataType,
    ReconstructionOutput,
    TokenizerOutput,
)
from ntt.core.model.tokenizer.base import (
    BaseTokenizer,
    CapabilityValidationError,
    LatentType,
    TokenizerSpec,
    create_tokenizer,
    register_tokenizer,
)
from ntt.core.model.tokenizer.validation import make_dummy_input
from tests.fixtures import make_dummy_binned_input


class TestValidTokenizer:
    """Tests for valid tokenizer configurations."""

    def test_identity_tokenizer_creates_successfully(self):
        """Test that a valid tokenizer can be created without errors."""
        spec = IdentityTokenizerSpec(window_length_seconds=1.0)
        tokenizer = IdentityTokenizer(spec)

        assert tokenizer is not None
        assert tokenizer.spec == spec
        assert tokenizer.get_vocab_size() == 1024

    def test_identity_tokenizer_get_vocab_size(self):
        """Test that get_vocab_size works on tokenizer instance."""
        spec = IdentityTokenizerSpec(window_length_seconds=1.0)
        tokenizer = IdentityTokenizer(spec)

        # Should be able to call get_vocab_size directly on tokenizer
        vocab_size = tokenizer.get_vocab_size()
        assert vocab_size == 1024
        assert isinstance(vocab_size, int)


class TestInvalidTokenizerConfigurations:
    """Tests for invalid tokenizer configurations that should raise CapabilityValidationError."""

    def test_discrete_tokenizer_with_zero_vocab_size(self):
        """Test that discrete tokenizer with vocab_size=0 raises validation error."""

        # Create an invalid spec - discrete but vocab_size = 0
        class InvalidTokenizerSpec(TokenizerSpec, frozen=True):
            KIND: Literal["invalid_vocab_size_test"] = "invalid_vocab_size_test"
            vocab_size: int = 0
            latent_type: LatentType = LatentType.DISCRETE
            compress_n: bool = True
            compress_t: bool = False
            variable_n: bool = True
            variable_t: bool = False
            variable_dt: bool = True
            input_type: DataType = DataType.SPIKES
            output_type: DataType = DataType.RATES

            def get_vocab_size(self) -> int:
                return self.vocab_size

        @register_tokenizer
        class InvalidTokenizer(BaseTokenizer):
            spec_class = InvalidTokenizerSpec

            def __init__(self, spec: InvalidTokenizerSpec):
                super().__init__(spec)

            def encode(self, x: torch.Tensor) -> torch.Tensor:
                return torch.zeros(x.shape[0], x.shape[1], dtype=torch.long)

            def decode(self, tokens: torch.Tensor) -> torch.Tensor:
                return torch.zeros(tokens.shape[0], tokens.shape[1], 1)

            def forward(self, x: torch.Tensor) -> TokenizerOutput:
                tokens = self.encode(x)
                reconstruction = self.decode(tokens)
                return TokenizerOutput(tokens=tokens, reconstruction=reconstruction)

        invalid_spec = InvalidTokenizerSpec(window_length_seconds=1.0)

        with pytest.raises(CapabilityValidationError) as exc_info:
            InvalidTokenizer(invalid_spec)

        # Check that the vocab_size error is present
        error_messages = " ".join(exc_info.value.errors)
        assert "vocab_size is 0" in error_messages
        assert "must be > 0 for discrete tokenizers" in error_messages

    def test_discrete_tokenizer_without_get_vocab_size(self):
        """Test that discrete tokenizer without get_vocab_size raises validation error."""

        # Create an invalid spec - discrete but get_vocab_size raises NotImplementedError
        class InvalidTokenizerSpec(TokenizerSpec, frozen=True):
            KIND: Literal["invalid_no_vocab_size_test"] = "invalid_no_vocab_size_test"
            latent_type: LatentType = LatentType.DISCRETE
            compress_n: bool = True
            compress_t: bool = False
            variable_n: bool = True
            variable_t: bool = False
            variable_dt: bool = True
            input_type: DataType = DataType.SPIKES
            output_type: DataType = DataType.RATES

            def get_vocab_size(self) -> int:
                # Explicitly raise NotImplementedError to test validation
                raise NotImplementedError(
                    "get_vocab_size must be implemented by subclasses with a discrete latent type"
                )

        @register_tokenizer
        class InvalidTokenizer(BaseTokenizer):
            spec_class = InvalidTokenizerSpec

            def __init__(self, spec: InvalidTokenizerSpec):
                super().__init__(spec)

            def encode(self, x: torch.Tensor) -> torch.Tensor:
                return torch.zeros(x.shape[0], x.shape[1], dtype=torch.long)

            def decode(self, tokens: torch.Tensor) -> torch.Tensor:
                return torch.zeros(tokens.shape[0], tokens.shape[1], 1)

            def forward(self, x: torch.Tensor) -> TokenizerOutput:
                tokens = self.encode(x)
                reconstruction = self.decode(tokens)
                return TokenizerOutput(tokens=tokens, reconstruction=reconstruction)

        invalid_spec = InvalidTokenizerSpec(window_length_seconds=1.0)

        with pytest.raises(CapabilityValidationError) as exc_info:
            InvalidTokenizer(invalid_spec)

        # Check that the get_vocab_size error is present
        error_messages = " ".join(exc_info.value.errors)
        assert "get_vocab_size() is not implemented" in error_messages
        assert "must implement get_vocab_size() for discrete tokenizers" in error_messages

    def test_variable_n_false_but_n_neurons_false(self):
        """Test that variable_n=False with n_neurons=False raises validation error."""

        class InvalidTokenizerSpec(TokenizerSpec, frozen=True):
            KIND: Literal["invalid_variable_n_test"] = "invalid_variable_n_test"
            vocab_size: int = 1024
            latent_type: LatentType = LatentType.DISCRETE
            compress_n: bool = True
            compress_t: bool = False
            variable_n: bool = False  # Claims fixed N
            variable_t: bool = False
            variable_dt: bool = True
            input_type: DataType = DataType.SPIKES
            output_type: DataType = DataType.RATES
            n_neurons: Literal[False] = False  # But n_neurons is False (use all)!

            def get_vocab_size(self) -> int:
                return self.vocab_size

        @register_tokenizer
        class InvalidTokenizer(BaseTokenizer):
            spec_class = InvalidTokenizerSpec

            def __init__(self, spec: InvalidTokenizerSpec):
                super().__init__(spec)

            def encode(self, x: torch.Tensor) -> torch.Tensor:
                return torch.zeros(x.shape[0], x.shape[1], dtype=torch.long)

            def decode(self, tokens: torch.Tensor) -> torch.Tensor:
                return torch.zeros(tokens.shape[0], tokens.shape[1], 1)

            def forward(self, x: torch.Tensor) -> TokenizerOutput:
                tokens = self.encode(x)
                reconstruction = self.decode(tokens)
                return TokenizerOutput(tokens=tokens, reconstruction=reconstruction)

        invalid_spec = InvalidTokenizerSpec(window_length_seconds=1.0)

        with pytest.raises(CapabilityValidationError) as exc_info:
            InvalidTokenizer(invalid_spec)

        # Check that the variable_n error is present
        error_messages = " ".join(exc_info.value.errors)
        assert "variable_n is False but n_neurons is False" in error_messages

    def test_multiple_validation_errors(self):
        """Test that multiple validation errors are collected and reported together."""

        class InvalidTokenizerSpec(TokenizerSpec, frozen=True):
            KIND: Literal["invalid_multiple_errors_test"] = "invalid_multiple_errors_test"
            vocab_size: int = 0  # Error 1: discrete but vocab_size = 0
            latent_type: LatentType = LatentType.DISCRETE
            compress_n: bool = True
            compress_t: bool = False
            variable_n: bool = False  # Error 2: fixed N but n_neurons is False
            variable_t: bool = False
            variable_dt: bool = False
            input_type: DataType = DataType.SPIKES
            output_type: DataType = DataType.RATES
            n_neurons: Literal[False] = False  # Error 2: should be fixed

            def get_vocab_size(self) -> int:
                return self.vocab_size

        @register_tokenizer
        class InvalidTokenizer(BaseTokenizer):
            spec_class = InvalidTokenizerSpec

            def __init__(self, spec: InvalidTokenizerSpec):
                super().__init__(spec)

            def encode(self, x: torch.Tensor) -> torch.Tensor:
                return torch.zeros(x.shape[0], x.shape[1], dtype=torch.long)

            def decode(self, tokens: torch.Tensor) -> torch.Tensor:
                return torch.zeros(tokens.shape[0], tokens.shape[1], 1)

            def forward(self, x: torch.Tensor) -> TokenizerOutput:
                tokens = self.encode(x)
                reconstruction = self.decode(tokens)
                return TokenizerOutput(tokens=tokens, reconstruction=reconstruction)

        invalid_spec = InvalidTokenizerSpec(window_length_seconds=1.0)

        with pytest.raises(CapabilityValidationError) as exc_info:
            InvalidTokenizer(invalid_spec)

        # Should have multiple errors
        assert len(exc_info.value.errors) >= 2

        # Check that both errors are present
        error_messages = " ".join(exc_info.value.errors)
        assert "vocab_size is 0" in error_messages or "must be > 0" in error_messages
        assert "variable_n is False but n_neurons is False" in error_messages


class TestCapabilityValidationError:
    """Tests for the CapabilityValidationError exception class."""

    def test_error_message_format(self):
        """Test that error message is properly formatted."""
        errors = [
            "Error 1: Something went wrong",
            "Error 2: Another issue",
        ]
        tokenizer_name = "TestTokenizer"

        error = CapabilityValidationError(errors, tokenizer_name)

        assert error.errors == errors
        assert error.tokenizer_name == tokenizer_name
        assert tokenizer_name in str(error)
        assert "2 errors" in str(error)
        assert "Error 1" in str(error)
        assert "Error 2" in str(error)

    def test_single_error_message(self):
        """Test error message format for single error."""
        errors = ["Only one error"]
        tokenizer_name = "TestTokenizer"

        error = CapabilityValidationError(errors, tokenizer_name)

        assert "1 error" in str(error)
        assert "Only one error" in str(error)


class TestReconstructMethod:
    """Tests for BaseTokenizer.reconstruct() and reconstruct_from_input()."""

    def test_reconstruct_from_input_returns_reconstruction_output(self):
        """reconstruct_from_input() should return a ReconstructionOutput."""
        spec = IdentityTokenizerSpec(window_length_seconds=1.0, bin_size_ms=1.0)
        tokenizer = IdentityTokenizer(spec)
        batch = make_dummy_binned_input(n_neurons=5, time_bins=1000, bin_size_ms=1.0)
        result = tokenizer.reconstruct_from_input(batch)
        assert isinstance(result, ReconstructionOutput)

    def test_reconstruct_from_input_metadata_from_spec(self):
        """reconstruct_from_input() output should have data_type and bin_size from spec."""
        spec = IdentityTokenizerSpec(window_length_seconds=1.0, bin_size_ms=2.0)
        tokenizer = IdentityTokenizer(spec)
        batch = make_dummy_binned_input(n_neurons=5, time_bins=500, bin_size_ms=2.0)
        result = tokenizer.reconstruct_from_input(batch)
        assert result.data_type == DataType.SPIKES
        assert result.bin_size_seconds == pytest.approx(0.002)

    def test_reconstruct_from_tokens(self):
        """reconstruct() should accept tokens and return ReconstructionOutput."""
        spec = IdentityTokenizerSpec(window_length_seconds=1.0, bin_size_ms=1.0)
        tokenizer = IdentityTokenizer(spec)
        batch = make_dummy_binned_input(n_neurons=5, time_bins=1000, bin_size_ms=1.0)
        tokens = tokenizer.encode(batch)
        result = tokenizer.reconstruct(tokens)
        assert isinstance(result, ReconstructionOutput)
        assert result.data_type == DataType.SPIKES
        assert result.bin_size_seconds == pytest.approx(0.001)

    def test_reconstruct_from_input_matches_forward(self):
        """reconstruct_from_input().reconstruction should equal forward().reconstruction."""
        spec = IdentityTokenizerSpec(window_length_seconds=1.0, bin_size_ms=1.0)
        tokenizer = IdentityTokenizer(spec)
        batch = make_dummy_binned_input(n_neurons=5, time_bins=1000, bin_size_ms=1.0)
        fwd = tokenizer.forward(batch)
        rec = tokenizer.reconstruct_from_input(batch)
        assert torch.equal(fwd.reconstruction, rec.reconstruction)

    def test_reconstruct_bin_size_none_when_false(self):
        """bin_size_seconds should be None when spec.bin_size_ms is False."""
        # IdentityTokenizer has variable_dt=True so bin_size_ms=False is valid
        spec = IdentityTokenizerSpec(window_length_seconds=1.0)
        tokenizer = IdentityTokenizer(spec)
        # Need to provide a BinnedSpikesInput even though spec says variable dt
        batch = make_dummy_binned_input(n_neurons=5, time_bins=100, bin_size_ms=1.0)
        result = tokenizer.reconstruct_from_input(batch)
        assert result.bin_size_seconds is None
        assert result.data_type == DataType.SPIKES


class TestDummyDataValidation:
    """Tests for dummy-data capability validation via validate_with_dummy_data()."""

    def test_identity_passes_via_factory(self):
        """create_tokenizer() should pass dummy validation for IdentityTokenizer."""
        spec = IdentityTokenizerSpec(window_length_seconds=1.0, bin_size_ms=1.0)
        tokenizer = create_tokenizer(spec)
        assert isinstance(tokenizer, IdentityTokenizer)

    def test_simple_vae_passes_via_factory(self):
        """create_tokenizer() should pass dummy validation for SimpleVAETokenizer."""
        from ntt.contrib.tokenizer.simple_vae import (
            SimpleVAETokenizer,
            SimpleVAETokenizerSpec,
        )

        spec = SimpleVAETokenizerSpec()
        tokenizer = create_tokenizer(spec)
        assert isinstance(tokenizer, SimpleVAETokenizer)

    def test_direct_construction_skips_dummy_validation(self):
        """Direct construction should not run dummy-data validation."""
        # IdentityTokenizer with bin_size_ms=False still constructs directly
        # (dummy validation only runs via create_tokenizer)
        spec = IdentityTokenizerSpec(window_length_seconds=1.0)
        tokenizer = IdentityTokenizer(spec)
        assert tokenizer is not None

    def test_broken_encode_fails_validation(self):
        """Tokenizer whose encode() raises should fail dummy-data validation."""

        class BrokenSpec(TokenizerSpec, frozen=True):
            KIND: Literal["broken_encode_test"] = "broken_encode_test"
            vocab_size: int = 10
            latent_type: LatentType = LatentType.DISCRETE
            compress_n: bool = False
            compress_t: bool = False
            variable_n: bool = True
            variable_t: bool = True
            variable_dt: bool = True
            input_type: DataType = DataType.SPIKES
            output_type: DataType = DataType.SPIKES

            def get_vocab_size(self) -> int:
                return self.vocab_size

        @register_tokenizer
        class BrokenTokenizer(BaseTokenizer):
            spec_class = BrokenSpec

            def __init__(self, spec: BrokenSpec):
                super().__init__(spec)

            def encode(self, x: BinnedSpikesInput) -> torch.Tensor:
                raise RuntimeError("Intentionally broken encode")

            def decode(self, tokens: torch.Tensor) -> torch.Tensor:
                return torch.zeros(1)

            def forward(self, x: BinnedSpikesInput) -> TokenizerOutput:
                raise RuntimeError("Intentionally broken forward")

        spec = BrokenSpec(window_length_seconds=1.0)
        # Direct construction works (no dummy validation)
        tokenizer = BrokenTokenizer(spec)
        assert tokenizer is not None

        # But validate_with_dummy_data should fail
        with pytest.raises(CapabilityValidationError) as exc_info:
            tokenizer.validate_with_dummy_data()

        error_messages = " ".join(exc_info.value.errors)
        assert "reconstruct_from_input() failed on dummy input" in error_messages
        assert "Intentionally broken encode" in error_messages

    def test_explicit_validate_with_dummy_data(self):
        """validate_with_dummy_data() can be called manually after direct construction."""
        spec = IdentityTokenizerSpec(window_length_seconds=1.0, bin_size_ms=1.0)
        tokenizer = IdentityTokenizer(spec)
        # Should not raise
        tokenizer.validate_with_dummy_data()


class TestMakeDummyInputWindowOverride:
    """``window_length_seconds`` is an opt-in override on ``make_dummy_input``.

    The default must keep sampling: init-time shape validation wants a representative
    draw from a variable-window spec. Only callers that need one deterministic window on
    both sides of a ratio — ``compute_compression`` — pass it explicitly.
    """

    @staticmethod
    def _dist_spec():
        return IdentityTokenizerSpec(
            n_neurons=4,
            window_length_seconds=UniformDist(low=1.0, high=5.0, seed=0),
            bin_size_ms=1.0,
        )

    def test_override_sets_the_window_and_the_bin_count(self):
        batch = make_dummy_input(self._dist_spec(), window_length_seconds=2.5)

        assert float(batch.duration_sec.item()) == pytest.approx(2.5)
        assert batch.spike_matrix.shape[1] == 2500  # 2.5 s @ 1 ms bins
        assert float(batch.absolute_end.item()) == pytest.approx(2.5)

    def test_default_still_samples_a_dist_window(self):
        spec = self._dist_spec()

        drawn = float(make_dummy_input(spec).duration_sec.item())

        assert 1.0 <= drawn < 5.0
        # Not silently resolved to Dist.max — init-time validation keeps its draw.
        assert drawn != pytest.approx(spec.effective_window_length_seconds)

    def test_override_wins_over_a_fixed_window_too(self):
        spec = IdentityTokenizerSpec(n_neurons=4, window_length_seconds=1.0, bin_size_ms=1.0)

        batch = make_dummy_input(spec, window_length_seconds=0.5)

        assert float(batch.duration_sec.item()) == pytest.approx(0.5)
        assert batch.spike_matrix.shape[1] == 500

    def test_override_applies_to_the_spike_times_branch(self):
        """Spike-times specs carry the window in ``duration_sec`` only — it must be
        pinned there too, since that is what a spike-times tokenizer encodes against."""

        class _SpikeTimesSpec(TokenizerSpec, frozen=True):
            KIND: Literal["window_override_spike_times_test"] = "window_override_spike_times_test"
            latent_type: LatentType = LatentType.CONTINUOUS
            compress_n: bool = False
            compress_t: bool = True
            variable_n: bool = True
            variable_t: bool = True
            variable_dt: bool = True
            input_type: DataType = DataType.SPIKE_TIMES
            output_type: DataType = DataType.SPIKES

        spec = _SpikeTimesSpec(
            n_neurons=4,
            window_length_seconds=UniformDist(low=1.0, high=5.0, seed=0),
            # Explicit, because the spec validator cannot infer it from a Dist window
            # (`window_length_seconds * 1000.0` has no meaning for a distribution).
            token_bin_size_ms=100.0,
        )

        batch = make_dummy_input(spec, window_length_seconds=3.0)

        assert float(batch.duration_sec.item()) == pytest.approx(3.0)
