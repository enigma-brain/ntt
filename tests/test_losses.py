"""Tests for loss functions, target transforms, and base class behavior."""

import pytest
import torch

from ntt.core.model.loss.components import GaussianSmoothTargets


class TestGaussianSmoothTargets:
    def test_output_shape_matches_input(self):
        transform = GaussianSmoothTargets(sigma=1.0)
        target = torch.zeros(2, 100, 5)
        target[0, 50, 0] = 1.0  # single spike
        result = transform(target)
        assert result.shape == target.shape

    def test_single_spike_produces_gaussian_peak(self):
        transform = GaussianSmoothTargets(sigma=1.0)
        target = torch.zeros(1, 100, 1)
        target[0, 50, 0] = 1.0
        result = transform(target)
        # Peak should be at the spike location
        assert result[0, 50, 0] == result[0, :, 0].max()
        # Values should decay away from the spike
        assert result[0, 50, 0] > result[0, 49, 0]
        assert result[0, 49, 0] > result[0, 48, 0]

    def test_output_bounded_zero_one(self):
        transform = GaussianSmoothTargets(sigma=1.0)
        target = torch.zeros(2, 100, 5)
        target[0, 50, 0] = 1.0
        target[0, 60, 0] = 1.0
        result = transform(target)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_zero_input_gives_zero_output(self):
        transform = GaussianSmoothTargets(sigma=1.0)
        target = torch.zeros(2, 100, 5)
        result = transform(target)
        assert (result == 0.0).all()

    def test_larger_sigma_wider_spread(self):
        target = torch.zeros(1, 100, 1)
        target[0, 50, 0] = 1.0
        narrow = GaussianSmoothTargets(sigma=0.5)(target)
        wide = GaussianSmoothTargets(sigma=2.0)(target)
        # At 3 bins away, wider sigma should have higher value
        assert wide[0, 47, 0] > narrow[0, 47, 0]


from ntt.core.model.loss.components import DistanceWeightedMask


class TestDistanceWeightedMask:
    def test_output_shape_matches_input(self):
        mask = DistanceWeightedMask(window_size=2)
        target = torch.zeros(2, 100, 5)
        target[0, 50, 0] = 1.0
        result = mask(target)
        assert result.shape == target.shape

    def test_spike_location_gets_full_weight(self):
        mask = DistanceWeightedMask(window_size=2)
        target = torch.zeros(1, 100, 1)
        target[0, 50, 0] = 1.0
        result = mask(target)
        # At the spike itself, weight should be 1.0 (or full)
        assert result[0, 50, 0] == 1.0

    def test_within_window_reduced_weight(self):
        mask = DistanceWeightedMask(window_size=2)
        target = torch.zeros(1, 100, 1)
        target[0, 50, 0] = 1.0
        result = mask(target)
        # 1 bin away: weight = 1/2 = 0.5
        assert torch.isclose(result[0, 49, 0], torch.tensor(0.5))
        assert torch.isclose(result[0, 51, 0], torch.tensor(0.5))

    def test_outside_window_full_weight(self):
        mask = DistanceWeightedMask(window_size=2)
        target = torch.zeros(1, 100, 1)
        target[0, 50, 0] = 1.0
        result = mask(target)
        # 3+ bins away: weight = 1.0 (full penalty)
        assert result[0, 47, 0] == 1.0
        assert result[0, 53, 0] == 1.0

    def test_no_spikes_all_ones(self):
        mask = DistanceWeightedMask(window_size=2)
        target = torch.zeros(2, 100, 5)
        result = mask(target)
        assert (result == 1.0).all()

    def test_larger_window_wider_reduction(self):
        target = torch.zeros(1, 100, 1)
        target[0, 50, 0] = 1.0
        small = DistanceWeightedMask(window_size=1)(target)
        large = DistanceWeightedMask(window_size=4)(target)
        # At 2 bins away: small window gives 1.0 (outside), large gives 0.5
        assert small[0, 48, 0] == 1.0
        assert large[0, 48, 0] < 1.0


from typing import Literal

from ntt.core.components.types import (
    BinnedSpikesInput,
    DataType,
    TokenizerOutput,
)
from ntt.core.model.loss.base import BaseLoss, LossSpec, create_loss, register_loss
from tests.fixtures import make_dummy_binned_input


def _make_binned_input(spike_matrix: torch.Tensor) -> "BinnedSpikesInput":
    """Create a minimal BinnedSpikesInput for loss testing."""
    batch_size = spike_matrix.shape[0] if spike_matrix.ndim == 3 else 1
    n_units = spike_matrix.shape[-1]
    duration_val = spike_matrix.shape[-2] / 1000.0
    return BinnedSpikesInput(
        spike_matrix=spike_matrix,
        pad_and_gap_mask=torch.ones_like(spike_matrix, dtype=torch.bool),
        sampling_rate=1000.0,
        unit_ids=torch.arange(n_units, dtype=torch.int32).unsqueeze(0).expand(batch_size, -1),
        unit_ids_pad_mask=torch.ones((batch_size, n_units), dtype=torch.int32),
        duration_sec=torch.full((batch_size,), duration_val),
        session_id=tuple(["__test__"] * batch_size),
        subject_id=tuple(["__test__"] * batch_size),
        species=tuple(["__test__"] * batch_size),
        absolute_start=torch.zeros(batch_size, dtype=torch.float64),
        absolute_end=torch.full((batch_size,), duration_val, dtype=torch.float64),
    )


class _TestLossSpec(LossSpec, frozen=True):
    """Minimal spec for testing the base class template."""

    KIND: Literal["_test_squared"] = "_test_squared"
    supported_latents: list = []
    supported_outputs: None = None


class _TestSquaredLoss(BaseLoss):
    """Test loss: returns element-wise squared error."""

    spec_class = _TestLossSpec

    def __init__(self, spec: _TestLossSpec):
        super().__init__(spec)

    def compute_loss(
        self,
        input_data: torch.Tensor,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        sample_weights: torch.Tensor | None,
    ) -> torch.Tensor:
        loss = (reconstruction - target) ** 2
        if sample_weights is not None:
            loss = loss * sample_weights
        return loss


class TestBaseLossTemplate:
    def test_forward_returns_scalar(self):
        spec = _TestLossSpec(reduction="mean")
        loss_fn = _TestSquaredLoss(spec)
        batch = make_dummy_binned_input(n_neurons=5, time_bins=100)
        recon = torch.randn(1, 100, 5)
        output = TokenizerOutput(reconstruction=recon)
        result = loss_fn(batch, output, batch.pad_and_gap_mask)
        assert result.ndim == 0

    def test_forward_with_gaussian_sigma(self):
        spec = _TestLossSpec(reduction="mean", gaussian_sigma=1.0)
        loss_fn = _TestSquaredLoss(spec)
        assert loss_fn._smooth_targets is not None
        batch = make_dummy_binned_input(n_neurons=5, time_bins=100)
        recon = torch.randn(1, 100, 5)
        output = TokenizerOutput(reconstruction=recon)
        result = loss_fn(batch, output, batch.pad_and_gap_mask)
        assert result.ndim == 0

    def test_forward_with_distance_weight(self):
        spec = _TestLossSpec(reduction="mean", distance_weight_window=2)
        loss_fn = _TestSquaredLoss(spec)
        assert loss_fn._distance_weights is not None
        batch = make_dummy_binned_input(n_neurons=5, time_bins=100)
        recon = torch.randn(1, 100, 5)
        output = TokenizerOutput(reconstruction=recon)
        result = loss_fn(batch, output, batch.pad_and_gap_mask)
        assert result.ndim == 0

    def test_reduction_none_returns_full_shape(self):
        spec = _TestLossSpec(reduction="none")
        loss_fn = _TestSquaredLoss(spec)
        batch = make_dummy_binned_input(n_neurons=5, time_bins=100)
        recon = torch.randn(1, 100, 5)
        output = TokenizerOutput(reconstruction=recon)
        result = loss_fn(batch, output, batch.pad_and_gap_mask)
        assert result.shape == (1, 100, 5)

    def test_reduction_sum(self):
        spec = _TestLossSpec(reduction="sum")
        loss_fn = _TestSquaredLoss(spec)
        batch = make_dummy_binned_input(n_neurons=5, time_bins=100)
        recon = torch.randn(1, 100, 5)
        output = TokenizerOutput(reconstruction=recon)
        result = loss_fn(batch, output, batch.pad_and_gap_mask)
        assert result.ndim == 0

    def test_forward_with_rates_input(self):
        from ntt.core.components.types import RatesInput

        spec = _TestLossSpec(reduction="mean")
        loss_fn = _TestSquaredLoss(spec)
        rate_matrix = torch.randn(1, 100, 5)
        pad_and_gap_mask = torch.ones(1, 100, 5, dtype=torch.int32)
        batch = RatesInput(
            unit_ids=torch.arange(5).unsqueeze(0),
            unit_ids_pad_mask=torch.ones(1, 5, dtype=torch.int32),
            duration_sec=torch.tensor([0.1]),
            absolute_start=torch.tensor([0.0], dtype=torch.float64),
            absolute_end=torch.tensor([0.1], dtype=torch.float64),
            session_id="__test__",
            subject_id="__test__",
            species="__test__",
            behavioral_data=None,
            rate_matrix=rate_matrix,
            pad_and_gap_mask=pad_and_gap_mask,
            sampling_rate=1000.0,
        )
        recon = torch.randn(1, 100, 5)
        output = TokenizerOutput(reconstruction=recon)
        result = loss_fn(batch, output, batch.pad_and_gap_mask)
        assert result.ndim == 0

    def test_combined_gaussian_and_distance_weight(self):
        spec = _TestLossSpec(reduction="mean", gaussian_sigma=1.0, distance_weight_window=2)
        loss_fn = _TestSquaredLoss(spec)
        assert loss_fn._smooth_targets is not None
        assert loss_fn._distance_weights is not None
        batch = make_dummy_binned_input(n_neurons=5, time_bins=100)
        recon = torch.randn(1, 100, 5)
        output = TokenizerOutput(reconstruction=recon)
        result = loss_fn(batch, output, batch.pad_and_gap_mask)
        assert result.ndim == 0


from ntt.contrib.loss.eval_only_mse import EvalOnlyMSELoss, EvalOnlyMSELossSpec
from ntt.contrib.loss.mse import MSELoss, MSELossSpec
from ntt.contrib.loss.poisson_nll import PoissonNLLLoss, PoissonNLLLossSpec


class TestMSELossRegression:
    def test_returns_scalar_mean(self):
        spec = MSELossSpec(reduction="mean")
        loss_fn = MSELoss(spec)
        batch = make_dummy_binned_input(n_neurons=5, time_bins=100)
        recon = torch.randn(1, 100, 5)
        output = TokenizerOutput(reconstruction=recon)
        result = loss_fn(batch, output, batch.pad_and_gap_mask)
        assert result.ndim == 0
        assert result.item() > 0

    def test_perfect_reconstruction_zero_loss(self):
        spec = MSELossSpec(reduction="mean")
        loss_fn = MSELoss(spec)
        batch = make_dummy_binned_input(n_neurons=5, time_bins=100)
        recon = batch.spike_matrix.float()
        output = TokenizerOutput(reconstruction=recon)
        result = loss_fn(batch, output, batch.pad_and_gap_mask)
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-7)


class TestPoissonNLLLossRegression:
    def test_returns_scalar_mean(self):
        spec = PoissonNLLLossSpec(reduction="mean")
        loss_fn = PoissonNLLLoss(spec)
        batch = make_dummy_binned_input(n_neurons=5, time_bins=100)
        recon = torch.rand(1, 100, 5) + 0.01
        output = TokenizerOutput(reconstruction=recon)
        result = loss_fn(batch, output, batch.pad_and_gap_mask)
        assert result.ndim == 0


class TestEvalOnlyMSELossRegression:
    def test_returns_zero_during_training(self):
        spec = EvalOnlyMSELossSpec()
        loss_fn = EvalOnlyMSELoss(spec)
        loss_fn.train()
        batch = make_dummy_binned_input(n_neurons=5, time_bins=100)
        recon = torch.randn(1, 100, 5)
        output = TokenizerOutput(reconstruction=recon)
        result = loss_fn(batch, output, batch.pad_and_gap_mask)
        assert result.item() == 0.0

    def test_computes_mse_during_eval(self):
        spec = EvalOnlyMSELossSpec()
        loss_fn = EvalOnlyMSELoss(spec)
        loss_fn.eval()
        batch = make_dummy_binned_input(n_neurons=5, time_bins=100)
        recon = torch.randn(1, 100, 5)
        output = TokenizerOutput(reconstruction=recon)
        result = loss_fn(batch, output, batch.pad_and_gap_mask)
        assert result.item() > 0


from ntt.contrib.loss.l1 import L1Loss, L1LossSpec


class TestL1Loss:
    def test_returns_scalar(self):
        spec = L1LossSpec(reduction="mean")
        loss_fn = L1Loss(spec)
        batch = make_dummy_binned_input(n_neurons=5, time_bins=100)
        recon = torch.randn(1, 100, 5)
        output = TokenizerOutput(reconstruction=recon)
        result = loss_fn(batch, output, batch.pad_and_gap_mask)
        assert result.ndim == 0

    def test_perfect_reconstruction_zero_loss(self):
        spec = L1LossSpec(reduction="mean")
        loss_fn = L1Loss(spec)
        batch = make_dummy_binned_input(n_neurons=5, time_bins=100)
        recon = batch.spike_matrix.float()
        output = TokenizerOutput(reconstruction=recon)
        result = loss_fn(batch, output, batch.pad_and_gap_mask)
        assert torch.isclose(result, torch.tensor(0.0), atol=1e-7)

    def test_supported_outputs_is_rates(self):
        spec = L1LossSpec()
        assert DataType.RATES in spec.supported_outputs


from ntt.contrib.loss.huber import HuberLoss, HuberLossSpec


class TestHuberLoss:
    def test_returns_scalar(self):
        spec = HuberLossSpec(reduction="mean", delta=1.0)
        loss_fn = HuberLoss(spec)
        batch = make_dummy_binned_input(n_neurons=5, time_bins=100)
        recon = torch.randn(1, 100, 5)
        output = TokenizerOutput(reconstruction=recon)
        result = loss_fn(batch, output, batch.pad_and_gap_mask)
        assert result.ndim == 0

    def test_small_errors_behave_like_mse(self):
        """For errors << delta, Huber ≈ 0.5 * MSE."""
        spec_huber = HuberLossSpec(reduction="mean", delta=10.0)
        spec_mse = MSELossSpec(reduction="mean")
        huber = HuberLoss(spec_huber)
        mse = MSELoss(spec_mse)
        batch = make_dummy_binned_input(n_neurons=5, time_bins=100)
        recon = batch.spike_matrix.float() + 0.01
        output = TokenizerOutput(reconstruction=recon)
        h = huber(batch, output, batch.pad_and_gap_mask)
        m = mse(batch, output, batch.pad_and_gap_mask)
        assert torch.isclose(h, 0.5 * m, atol=1e-5)

    def test_supported_outputs_is_rates(self):
        spec = HuberLossSpec()
        assert DataType.RATES in spec.supported_outputs


from ntt.contrib.tokenizer.wanneural.wanneural import WanNeuralTokenizerSpec


class TestWanNeuralSpikesOutput:
    def test_spikes_output_rejects_poisson_sample(self):
        with pytest.raises(ValueError, match="reconstruct_sample must be 'round'"):
            WanNeuralTokenizerSpec(
                n_neurons=32,
                window_length_seconds=1.0,
                bin_size_ms=1.0,
                token_bin_size_ms=200.0,
                output_type=DataType.SPIKES,
                output_activation="sigmoid",
                reconstruct_sample="poisson",
            )

    def test_spikes_output_accepts_round_sample(self):
        spec = WanNeuralTokenizerSpec(
            n_neurons=32,
            window_length_seconds=1.0,
            bin_size_ms=1.0,
            token_bin_size_ms=200.0,
            output_type=DataType.SPIKES,
            output_activation="sigmoid",
            reconstruct_sample="round",
        )
        assert spec.reconstruct_sample == "round"


from ntt.contrib.loss.bce import BCELoss, BCELossSpec


class TestBCELoss:
    def test_returns_scalar(self):
        spec = BCELossSpec(reduction="mean")
        loss_fn = BCELoss(spec)
        batch = make_dummy_binned_input(n_neurons=5, time_bins=100)
        recon = torch.sigmoid(torch.randn(1, 100, 5))
        output = TokenizerOutput(reconstruction=recon)
        result = loss_fn(batch, output, batch.pad_and_gap_mask)
        assert result.ndim == 0

    def test_supported_outputs_is_spikes(self):
        spec = BCELossSpec()
        assert DataType.SPIKES in spec.supported_outputs

    def test_with_gaussian_smoothing(self):
        spec = BCELossSpec(reduction="mean", gaussian_sigma=1.0)
        loss_fn = BCELoss(spec)
        batch = make_dummy_binned_input(n_neurons=5, time_bins=100)
        recon = torch.sigmoid(torch.randn(1, 100, 5))
        output = TokenizerOutput(reconstruction=recon)
        result = loss_fn(batch, output, batch.pad_and_gap_mask)
        assert result.ndim == 0


from ntt.contrib.loss.focal import FocalLoss, FocalLossSpec


class TestFocalLoss:
    def test_returns_scalar(self):
        spec = FocalLossSpec(reduction="mean")
        loss_fn = FocalLoss(spec)
        batch = make_dummy_binned_input(n_neurons=5, time_bins=100)
        recon = torch.sigmoid(torch.randn(1, 100, 5))
        output = TokenizerOutput(reconstruction=recon)
        result = loss_fn(batch, output, batch.pad_and_gap_mask)
        assert result.ndim == 0

    def test_gamma_zero_no_alpha_equals_bce(self):
        """With gamma=0 and alpha=-1 (disabled), focal loss reduces to BCE."""
        spec_focal = FocalLossSpec(reduction="mean", gamma=0.0, alpha=-1.0)
        spec_bce = BCELossSpec(reduction="mean")
        focal = FocalLoss(spec_focal)
        bce = BCELoss(spec_bce)
        batch = make_dummy_binned_input(n_neurons=5, time_bins=100)
        recon = torch.sigmoid(torch.randn(1, 100, 5))
        output = TokenizerOutput(reconstruction=recon)
        f = focal(batch, output, batch.pad_and_gap_mask)
        b = bce(batch, output, batch.pad_and_gap_mask)
        assert torch.isclose(f, b, atol=1e-4)

    def test_high_gamma_reduces_easy_example_loss(self):
        """Focal loss with high gamma should downweight easy examples."""
        batch = make_dummy_binned_input(n_neurons=5, time_bins=100)
        recon = torch.full((1, 100, 5), 0.01)
        output = TokenizerOutput(reconstruction=recon)
        low_gamma = FocalLoss(FocalLossSpec(reduction="mean", gamma=0.0, alpha=-1.0))
        high_gamma = FocalLoss(FocalLossSpec(reduction="mean", gamma=5.0, alpha=-1.0))
        assert high_gamma(batch, output, batch.pad_and_gap_mask) < low_gamma(
            batch, output, batch.pad_and_gap_mask
        )

    def test_supported_outputs_is_spikes(self):
        spec = FocalLossSpec()
        assert DataType.SPIKES in spec.supported_outputs


class TestOutputTypeCompatibility:
    def test_bce_accepts_rate_output_type(self):
        """BCE accepts rate outputs when they are sigmoid-bounded Bernoulli rates."""
        from ntt.core.model.litmodule import NTTLitModule

        tokenizer_spec = WanNeuralTokenizerSpec(
            n_neurons=32,
            window_length_seconds=1.0,
            bin_size_ms=1.0,
            token_bin_size_ms=200.0,
            output_type=DataType.RATES,
            output_activation="softplus",
        )
        loss_spec = BCELossSpec(reduction="mean")
        module = NTTLitModule(tokenizer_spec=tokenizer_spec, loss_spec=loss_spec)
        module.configure_model()

    def test_compatible_output_type_ok(self):
        """BCE loss with SPIKES tokenizer should work."""
        from ntt.core.model.litmodule import NTTLitModule

        tokenizer_spec = WanNeuralTokenizerSpec(
            n_neurons=32,
            window_length_seconds=1.0,
            bin_size_ms=1.0,
            token_bin_size_ms=200.0,
            output_type=DataType.SPIKES,
            output_activation="sigmoid",
            reconstruct_sample="round",
        )
        loss_spec = BCELossSpec(reduction="mean")
        module = NTTLitModule(tokenizer_spec=tokenizer_spec, loss_spec=loss_spec)
        module.configure_model()  # should not raise

    def test_none_supported_outputs_accepts_all(self):
        """MSE loss (supported_outputs=None) should accept any output_type."""
        from ntt.core.model.litmodule import NTTLitModule

        tokenizer_spec = WanNeuralTokenizerSpec(
            n_neurons=32,
            window_length_seconds=1.0,
            bin_size_ms=1.0,
            token_bin_size_ms=200.0,
            output_type=DataType.SPIKES,
            output_activation="sigmoid",
            reconstruct_sample="round",
        )
        loss_spec = MSELossSpec(reduction="mean")
        module = NTTLitModule(tokenizer_spec=tokenizer_spec, loss_spec=loss_spec)
        module.configure_model()  # should not raise


class TestDiceLoss:
    """Tests for the Dice loss function."""

    def test_perfect_reconstruction_gives_low_loss(self):
        from ntt.contrib.loss.dice import DiceLossSpec

        spec = DiceLossSpec(eps=1.0, alpha=0.5, beta=0.5)
        loss_fn = create_loss(spec)
        target = torch.zeros(2, 100, 5)
        target[:, 50, :] = 1.0
        pred = target.clone()
        input_batch = _make_binned_input(target)
        output = TokenizerOutput(reconstruction=pred)
        result = loss_fn(input_batch, output, None)
        assert result.item() < 0.1

    def test_total_mismatch_gives_high_loss(self):
        from ntt.contrib.loss.dice import DiceLossSpec

        spec = DiceLossSpec(eps=1.0, alpha=0.5, beta=0.5)
        loss_fn = create_loss(spec)
        target = torch.zeros(2, 100, 5)
        target[:, 50, :] = 1.0
        pred = torch.zeros_like(target)
        input_batch = _make_binned_input(target)
        output = TokenizerOutput(reconstruction=pred)
        result = loss_fn(input_batch, output, None)
        # With eps=1.0 smoothing: tversky = (0+1)/(0+0.5*0+0.5*1+1) = 1/1.5
        # loss = 1 - 1/1.5 = 1/3 ≈ 0.333
        assert result.item() > 0.3

    def test_beta_asymmetry_penalizes_false_negatives_more(self):
        from ntt.contrib.loss.dice import DiceLossSpec

        target = torch.zeros(2, 100, 5)
        target[:, 50, :] = 1.0
        # Pure false-negative: predicts nothing (misses the spike)
        pred_fn = torch.zeros_like(target)
        # Pure false-positive: predicts the spike correctly + extra spike at 30
        pred_fp = target.clone()
        pred_fp[:, 30, :] = 1.0
        spec_high_beta = DiceLossSpec(eps=1.0, alpha=0.3, beta=0.7)
        loss_fn = create_loss(spec_high_beta)
        input_batch = _make_binned_input(target)
        # FN-only case: tp=0, fp=0, fn=1 → higher beta penalty
        loss_fn_result = loss_fn(input_batch, TokenizerOutput(reconstruction=pred_fn), None)
        # FP-only case: tp=1, fp=1, fn=0 → lower alpha penalty
        loss_fp_result = loss_fn(input_batch, TokenizerOutput(reconstruction=pred_fp), None)
        assert loss_fn_result.item() > loss_fp_result.item()


class TestWassersteinLoss:
    """Tests for the Wasserstein temporal loss."""

    def test_perfect_reconstruction_gives_zero_loss(self):
        from ntt.contrib.loss.wasserstein import WassersteinLossSpec

        spec = WassersteinLossSpec(p=1, normalize=False)
        loss_fn = create_loss(spec)
        target = torch.zeros(2, 100, 5)
        target[:, 50, :] = 1.0
        pred = target.clone()
        input_batch = _make_binned_input(target)
        output = TokenizerOutput(reconstruction=pred)
        result = loss_fn(input_batch, output, None)
        assert result.item() < 1e-5

    def test_temporal_shift_penalized(self):
        from ntt.contrib.loss.wasserstein import WassersteinLossSpec

        spec = WassersteinLossSpec(p=1, normalize=False)
        loss_fn = create_loss(spec)
        target = torch.zeros(2, 100, 5)
        target[:, 50, :] = 1.0
        pred_close = torch.zeros_like(target)
        pred_close[:, 51, :] = 1.0
        pred_far = torch.zeros_like(target)
        pred_far[:, 60, :] = 1.0
        input_batch = _make_binned_input(target)
        loss_close = loss_fn(input_batch, TokenizerOutput(reconstruction=pred_close), None)
        loss_far = loss_fn(input_batch, TokenizerOutput(reconstruction=pred_far), None)
        assert loss_far.item() > loss_close.item()

    def test_p2_penalizes_large_shifts_more(self):
        from ntt.contrib.loss.wasserstein import WassersteinLossSpec

        target = torch.zeros(1, 100, 1)
        target[:, 50, :] = 1.0
        pred = torch.zeros_like(target)
        pred[:, 60, :] = 1.0
        spec_p1 = WassersteinLossSpec(p=1, normalize=False)
        spec_p2 = WassersteinLossSpec(p=2, normalize=False)
        loss_p1 = create_loss(spec_p1)(
            _make_binned_input(target), TokenizerOutput(reconstruction=pred), None
        )
        loss_p2 = create_loss(spec_p2)(
            _make_binned_input(target), TokenizerOutput(reconstruction=pred), None
        )
        assert loss_p2.item() > loss_p1.item()


class TestGlobalRateLoss:
    """Tests for the GlobalRateLoss firing-rate calibration loss."""

    def test_perfect_reconstruction_gives_zero_loss(self):
        from ntt.contrib.loss.global_rate import GlobalRateLossSpec

        spec = GlobalRateLossSpec(normalize=False)
        loss_fn = create_loss(spec)
        target = torch.zeros(2, 100, 5)
        target[:, 50, :] = 1.0
        pred = target.clone()
        input_batch = _make_binned_input(target)
        output = TokenizerOutput(reconstruction=pred)
        result = loss_fn(input_batch, output, None)
        assert result.item() < 1e-5

    def test_rate_mismatch_penalized(self):
        from ntt.contrib.loss.global_rate import GlobalRateLossSpec

        spec = GlobalRateLossSpec(normalize=False)
        loss_fn = create_loss(spec)
        target = torch.zeros(2, 100, 5)
        target[:, 50, :] = 1.0  # 1 spike per neuron
        pred = torch.zeros_like(target)
        pred[:, 50, :] = 3.0  # overpredict rate
        input_batch = _make_binned_input(target)
        result = loss_fn(input_batch, TokenizerOutput(reconstruction=pred), None)
        assert result.item() > 0.0

    def test_larger_mismatch_gives_larger_loss(self):
        from ntt.contrib.loss.global_rate import GlobalRateLossSpec

        spec = GlobalRateLossSpec(normalize=False)
        loss_fn = create_loss(spec)
        target = torch.zeros(2, 100, 5)
        target[:, 50, :] = 1.0
        pred_small = torch.zeros_like(target)
        pred_small[:, 50, :] = 2.0
        pred_large = torch.zeros_like(target)
        pred_large[:, 50, :] = 5.0
        input_batch = _make_binned_input(target)
        loss_small = loss_fn(input_batch, TokenizerOutput(reconstruction=pred_small), None)
        loss_large = loss_fn(input_batch, TokenizerOutput(reconstruction=pred_large), None)
        assert loss_large.item() > loss_small.item()

    def test_temporal_distribution_irrelevant(self):
        """Global rate only cares about total counts, not timing."""
        from ntt.contrib.loss.global_rate import GlobalRateLossSpec

        spec = GlobalRateLossSpec(normalize=False)
        loss_fn = create_loss(spec)
        target = torch.zeros(1, 100, 1)
        target[:, 50, :] = 1.0
        # Same total rate but different temporal placement
        pred_a = torch.zeros_like(target)
        pred_a[:, 10, :] = 1.0
        pred_b = torch.zeros_like(target)
        pred_b[:, 90, :] = 1.0
        input_batch = _make_binned_input(target)
        loss_a = loss_fn(input_batch, TokenizerOutput(reconstruction=pred_a), None)
        loss_b = loss_fn(input_batch, TokenizerOutput(reconstruction=pred_b), None)
        assert torch.allclose(loss_a, loss_b)

    def test_normalize_makes_loss_length_independent(self):
        """With normalize=True, doubling sequence length shouldn't change loss."""
        from ntt.contrib.loss.global_rate import GlobalRateLossSpec

        spec = GlobalRateLossSpec(normalize=True)
        loss_fn = create_loss(spec)
        # Short sequence: 1 spike in 50 bins, pred = 2 spikes
        target_short = torch.zeros(1, 50, 1)
        target_short[:, 25, :] = 1.0
        pred_short = torch.zeros_like(target_short)
        pred_short[:, 25, :] = 2.0
        # Long sequence: same rate density via pad_mask
        target_long = torch.zeros(1, 100, 1)
        target_long[:, 25, :] = 1.0
        pred_long = torch.zeros_like(target_long)
        pred_long[:, 25, :] = 2.0
        pad_mask_long = torch.zeros(1, 100, 1, dtype=torch.bool)
        pad_mask_long[:, :50, :] = True
        loss_short = loss_fn(
            _make_binned_input(target_short), TokenizerOutput(reconstruction=pred_short), None
        )
        loss_long = loss_fn(
            _make_binned_input(target_long),
            TokenizerOutput(reconstruction=pred_long),
            pad_mask_long,
        )
        assert torch.allclose(loss_short, loss_long, atol=1e-5)

    def test_pad_mask_excludes_padded_region(self):
        from ntt.contrib.loss.global_rate import GlobalRateLossSpec

        spec = GlobalRateLossSpec(normalize=False)
        loss_fn = create_loss(spec)
        target = torch.zeros(1, 100, 1)
        target[:, 50, :] = 1.0
        pred = target.clone()
        # Spike in padded region should be ignored
        pred[:, 80, :] = 10.0
        pad_mask = torch.ones(1, 100, 1, dtype=torch.bool)
        pad_mask[:, 70:, :] = False
        input_batch = _make_binned_input(target)
        result = loss_fn(input_batch, TokenizerOutput(reconstruction=pred), pad_mask)
        assert result.item() < 1e-5

    def test_distance_weight_window_rejected(self):
        from ntt.contrib.loss.global_rate import GlobalRateLossSpec

        spec = GlobalRateLossSpec(distance_weight_window=5)
        with pytest.raises(ValueError, match="not compatible"):
            create_loss(spec)


class TestMultiLoss:
    """Tests for the MultiLoss composite loss."""

    def test_single_component_matches_standalone(self):
        """MultiLoss with one MSE component should match standalone MSE."""
        from ntt.contrib.loss.multi import MultiLossComponent, MultiLossSpec

        standalone_spec = MSELossSpec()
        standalone = create_loss(standalone_spec)

        multi_spec = MultiLossSpec(
            components=[MultiLossComponent(loss={"KIND": "mse"}, weight=1.0, name="mse")]
        )
        multi = create_loss(multi_spec)

        target = torch.randn(2, 100, 5)
        pred = torch.randn(2, 100, 5)
        input_batch = _make_binned_input(target)
        output = TokenizerOutput(reconstruction=pred)

        standalone_loss = standalone(input_batch, output, None)
        multi_loss = multi(input_batch, output, None)
        assert torch.allclose(standalone_loss, multi_loss, atol=1e-6)

    def test_weighted_components(self):
        """Component weights should scale individual losses."""
        from ntt.contrib.loss.multi import MultiLossComponent, MultiLossSpec

        spec_w1 = MultiLossSpec(
            components=[
                MultiLossComponent(loss={"KIND": "mse"}, weight=1.0, name="mse"),
                MultiLossComponent(loss={"KIND": "l1"}, weight=0.0, name="l1"),
            ]
        )
        spec_w2 = MultiLossSpec(
            components=[
                MultiLossComponent(loss={"KIND": "mse"}, weight=0.0, name="mse"),
                MultiLossComponent(loss={"KIND": "l1"}, weight=1.0, name="l1"),
            ]
        )

        target = torch.randn(2, 100, 5)
        pred = torch.randn(2, 100, 5)
        input_batch = _make_binned_input(target)
        output = TokenizerOutput(reconstruction=pred)

        loss_mse_only = create_loss(spec_w1)(input_batch, output, None)
        loss_l1_only = create_loss(spec_w2)(input_batch, output, None)

        mse_standalone = create_loss(MSELossSpec())(input_batch, output, None)
        l1_standalone = create_loss(L1LossSpec())(input_batch, output, None)

        assert torch.allclose(loss_mse_only, mse_standalone, atol=1e-6)
        assert torch.allclose(loss_l1_only, l1_standalone, atol=1e-6)

    def test_component_losses_accessible(self):
        """get_component_losses() should return per-component detached losses."""
        from ntt.contrib.loss.multi import MultiLossComponent, MultiLossSpec

        spec = MultiLossSpec(
            components=[
                MultiLossComponent(loss={"KIND": "mse"}, weight=1.0, name="mse"),
                MultiLossComponent(loss={"KIND": "l1"}, weight=0.5, name="l1"),
            ]
        )
        loss_fn = create_loss(spec)

        target = torch.randn(2, 100, 5)
        pred = torch.randn(2, 100, 5)
        input_batch = _make_binned_input(target)
        output = TokenizerOutput(reconstruction=pred)

        _ = loss_fn(input_batch, output, None)
        components = loss_fn.get_component_losses()
        assert components is not None
        assert "mse" in components
        assert "l1" in components

    def test_component_losses_are_detached(self):
        """Component losses should be detached (no grad tracking)."""
        from ntt.contrib.loss.multi import MultiLossComponent, MultiLossSpec

        spec = MultiLossSpec(
            components=[
                MultiLossComponent(loss={"KIND": "mse"}, weight=1.0, name="mse"),
            ]
        )
        loss_fn = create_loss(spec)

        target = torch.randn(2, 100, 5)
        pred = torch.randn(2, 100, 5, requires_grad=True)
        input_batch = _make_binned_input(target)
        output = TokenizerOutput(reconstruction=pred)

        _ = loss_fn(input_batch, output, None)
        components = loss_fn.get_component_losses()
        assert not components["mse"].requires_grad

    def test_base_loss_get_component_losses_returns_none(self):
        """BaseLoss default get_component_losses() should return None."""
        loss_fn = create_loss(MSELossSpec())
        assert loss_fn.get_component_losses() is None

    def test_empty_components_raises(self):
        """MultiLoss with no components should raise ValueError."""
        from ntt.contrib.loss.multi import MultiLossSpec

        with pytest.raises(ValueError, match="at least one component"):
            create_loss(MultiLossSpec(components=[]))


class TestNamedAuxLoss:
    """Tests for scalar, dict and None aux_loss handling in TokenizerOutput."""

    def test_scalar_aux_loss_still_works(self):
        """TokenizerOutput accepts a scalar aux_loss."""
        scalar = torch.tensor(0.5)
        output = TokenizerOutput(
            tokens=None,
            reconstruction=torch.randn(2, 10, 3),
            aux_loss=scalar,
        )
        total = torch.tensor(1.0) + output.aux_loss
        assert total.item() == pytest.approx(1.5)

    def test_dict_aux_loss_works(self):
        """TokenizerOutput with dict aux_loss should work."""
        aux = {"bottleneck": torch.tensor(0.3), "ar": torch.tensor(0.2)}
        output = TokenizerOutput(
            tokens=None,
            reconstruction=torch.randn(2, 10, 3),
            aux_loss=aux,
        )
        assert isinstance(output.aux_loss, dict)
        total = sum(output.aux_loss.values())
        assert total.item() == pytest.approx(0.5)

    def test_none_aux_loss_works(self):
        """TokenizerOutput with None aux_loss should work."""
        output = TokenizerOutput(
            tokens=None,
            reconstruction=torch.randn(2, 10, 3),
            aux_loss=None,
        )
        assert output.aux_loss is None
