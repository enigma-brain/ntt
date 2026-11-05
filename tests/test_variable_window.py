"""Tests for variable window sampling feature."""

import numpy as np
import pytest

from ntt.core.components.dist import (
    Dist,
    GaussianDist,
    PytorchDist,
    TriangleDist,
    UniformDist,
)


class TestDistMinMax:
    """Tests for Dist.min and Dist.max abstract properties."""

    def test_uniform_dist_min_max(self):
        dist = UniformDist(low=0.1, high=2.0)
        assert dist.min == pytest.approx(0.1)
        assert dist.max == pytest.approx(2.0)

    def test_uniform_dist_int_min_max(self):
        dist = UniformDist(low=1.0, high=10.0, return_int=True)
        assert dist.min == 1
        assert dist.max == 10

    def test_triangle_dist_min_max(self):
        dist = TriangleDist(min_units=5, mode_units=50, max_units=100)
        assert dist.min == 5
        assert dist.max == 100

    def test_gaussian_dist_clipped_min_max(self):
        dist = GaussianDist(mean=1.0, std=0.5, clip_low=0.1, clip_high=2.0)
        assert dist.min == pytest.approx(0.1)
        assert dist.max == pytest.approx(2.0)

    def test_gaussian_dist_unclipped_raises(self):
        dist = GaussianDist(mean=1.0, std=0.5)
        with pytest.raises(ValueError, match="clip_low"):
            _ = dist.min
        with pytest.raises(ValueError, match="clip_high"):
            _ = dist.max

    def test_pytorch_dist_with_bounds(self):
        dist = PytorchDist(
            dist_class="Uniform",
            dist_params={"low": 0.1, "high": 2.0},
            bound_low=0.1,
            bound_high=2.0,
        )
        assert dist.min == pytest.approx(0.1)
        assert dist.max == pytest.approx(2.0)

    def test_pytorch_dist_without_bounds_raises(self):
        dist = PytorchDist(
            dist_class="Normal",
            dist_params={"loc": 0.0, "scale": 1.0},
        )
        with pytest.raises(ValueError, match="bound_low"):
            _ = dist.min
        with pytest.raises(ValueError, match="bound_high"):
            _ = dist.max


from ntt.core.model.tokenizer.base import TokenizerSpec


class TestTokenizerSpecDistWindow:
    """Tests for TokenizerSpec accepting Dist for window_length_seconds."""

    def test_effective_window_length_float(self):
        """Float window_length_seconds returns itself."""
        from ntt.contrib.tokenizer.calm.calm_tokenizer import CalmTokenizerSpec

        spec = CalmTokenizerSpec(
            n_neurons=8,
            window_length_seconds=2.0,
            bin_size_ms=10.0,
            patch_length=5,
            embed_dim=32,
            num_encoder_layers=1,
            num_decoder_layers=1,
            num_heads=4,
        )
        assert spec.effective_window_length_seconds == pytest.approx(2.0)


from ntt.contrib.tokenizer.calm.calm_tokenizer import CalmTokenizerSpec


class TestCalmSpecVariableT:
    """Tests for CalmTokenizerSpec variable_t and Dist validation."""

    def test_variable_t_is_true(self):
        spec = CalmTokenizerSpec(
            n_neurons=8,
            window_length_seconds=2.0,
            bin_size_ms=10.0,
            patch_length=5,
            embed_dim=32,
            num_encoder_layers=1,
            num_decoder_layers=1,
            num_heads=4,
        )
        assert spec.variable_t is True

    def test_dist_window_accepted(self):
        """Dist window_length_seconds should be accepted with valid bounds."""
        dist = UniformDist(low=0.1, high=2.0)
        spec = CalmTokenizerSpec(
            n_neurons=8,
            window_length_seconds=dist,
            bin_size_ms=10.0,
            patch_length=10,
            embed_dim=32,
            num_encoder_layers=1,
            num_decoder_layers=1,
            num_heads=4,
        )
        assert isinstance(spec.window_length_seconds, Dist)

    def test_dist_window_bad_min_rejected(self):
        """Dist with min_bins not divisible by patch_length should raise."""
        # low=0.13 -> 13 bins at bin_size_ms=10 -> 13 % 10 != 0
        dist = UniformDist(low=0.13, high=2.0)
        with pytest.raises(ValueError, match="divisible by patch_length"):
            CalmTokenizerSpec(
                n_neurons=8,
                window_length_seconds=dist,
                bin_size_ms=10.0,
                patch_length=10,
                embed_dim=32,
                num_encoder_layers=1,
                num_decoder_layers=1,
                num_heads=4,
            )

    def test_dist_window_bad_max_rejected(self):
        """Dist with max_bins not divisible by patch_length should raise."""
        # high=0.15 -> 15 bins at bin_size_ms=10 -> 15 % 10 != 0
        dist = UniformDist(low=0.1, high=0.15)
        with pytest.raises(ValueError, match="divisible by patch_length"):
            CalmTokenizerSpec(
                n_neurons=8,
                window_length_seconds=dist,
                bin_size_ms=10.0,
                patch_length=10,
                embed_dim=32,
                num_encoder_layers=1,
                num_decoder_layers=1,
                num_heads=4,
            )

    def test_float_window_still_validated(self):
        """Existing float validation should still work."""
        # 0.15s -> 15 bins at bin_size_ms=10 -> 15 % 10 != 0
        with pytest.raises(ValueError, match="divisible"):
            CalmTokenizerSpec(
                n_neurons=8,
                window_length_seconds=0.15,
                bin_size_ms=10.0,
                patch_length=10,
                embed_dim=32,
                num_encoder_layers=1,
                num_decoder_layers=1,
                num_heads=4,
            )

    def test_effective_window_with_dist(self):
        """effective_window_length_seconds returns Dist.max."""
        dist = UniformDist(low=0.1, high=2.0)
        spec = CalmTokenizerSpec(
            n_neurons=8,
            window_length_seconds=dist,
            bin_size_ms=10.0,
            patch_length=10,
            embed_dim=32,
            num_encoder_layers=1,
            num_decoder_layers=1,
            num_heads=4,
        )
        assert spec.effective_window_length_seconds == pytest.approx(2.0)


import torch
from temporaldata import ArrayDict, Data, Interval, IrregularTimeSeries
from torch_brain.data.collate import pad2d

from ntt.core.data.components.transforms import VariableWindowCrop


def _make_fake_data(start: float, end: float, n_units: int = 4, n_spikes: int = 50):
    """Create minimal Data object matching what the sampler produces."""
    rng = np.random.default_rng(42)
    timestamps = np.sort(rng.uniform(start, end, size=n_spikes))
    unit_index = rng.integers(0, n_units, size=n_spikes)

    domain = Interval(start, end)
    data = Data(domain=domain)
    data.spikes = IrregularTimeSeries(
        timestamps=timestamps,
        unit_index=unit_index,
        domain=domain,
    )
    data.units = ArrayDict(
        unit_number=np.arange(n_units),
        id=np.array([f"unit_{i}" for i in range(n_units)]),
    )
    return data


class TestVariableWindowCrop:
    """Tests for the VariableWindowCrop transform."""

    def test_output_within_original_bounds(self):
        """Cropped window length must not exceed the original window length."""
        dist = UniformDist(low=0.1, high=2.0, seed=42)
        crop = VariableWindowCrop(dist=dist, bin_size_ms=10.0, patch_length=10)
        data = _make_fake_data(start=5.0, end=7.0)
        full_length = data.end - data.start  # 2.0

        for _ in range(20):
            result = crop(data)
            # Data.slice() re-indexes to 0, so result.end == crop_length
            crop_len = result.end - result.start
            assert crop_len <= full_length + 1e-9

    def test_output_length_is_patch_aligned(self):
        """Cropped length must be a multiple of patch_duration."""
        dist = UniformDist(low=0.1, high=2.0, seed=123)
        patch_length = 10
        bin_size_ms = 10.0
        patch_duration = patch_length * bin_size_ms / 1000.0  # 0.1s
        crop = VariableWindowCrop(dist=dist, bin_size_ms=bin_size_ms, patch_length=patch_length)
        data = _make_fake_data(start=0.0, end=2.0)

        for _ in range(50):
            result = crop(data)
            crop_len = result.end - result.start
            n_patches = crop_len / patch_duration
            assert abs(n_patches - round(n_patches)) < 1e-6, (
                f"Crop length {crop_len} is not a multiple of patch_duration {patch_duration}"
            )

    def test_output_length_within_dist_range(self):
        """Cropped length should be within [min, max] of the distribution (after snapping)."""
        dist = UniformDist(low=0.1, high=1.0, seed=456)
        patch_length = 10
        bin_size_ms = 10.0
        patch_duration = patch_length * bin_size_ms / 1000.0  # 0.1s
        crop = VariableWindowCrop(dist=dist, bin_size_ms=bin_size_ms, patch_length=patch_length)
        data = _make_fake_data(start=0.0, end=2.0)

        for _ in range(50):
            result = crop(data)
            crop_len = result.end - result.start
            # Allow one patch_duration tolerance due to snapping
            assert crop_len >= patch_duration - 1e-6
            assert crop_len <= 1.0 + patch_duration + 1e-6

    def test_crops_are_variable(self):
        """Multiple crops should produce different lengths (with high probability)."""
        dist = UniformDist(low=0.1, high=2.0, seed=789)
        crop = VariableWindowCrop(dist=dist, bin_size_ms=10.0, patch_length=10)
        data = _make_fake_data(start=0.0, end=2.0)

        lengths = set()
        for _ in range(50):
            result = crop(data)
            crop_len = round(result.end - result.start, 4)
            lengths.add(crop_len)
        # With 20 possible lengths and 50 samples, should get many distinct values
        assert len(lengths) >= 5

    def test_full_window_returned_when_dist_max_equals_window(self):
        """When sampled length >= full window, return the full window."""
        # Dist always returns 2.0 (low == high)
        dist = UniformDist(low=2.0, high=2.0, seed=0)
        crop = VariableWindowCrop(dist=dist, bin_size_ms=10.0, patch_length=10)
        data = _make_fake_data(start=0.0, end=2.0)

        result = crop(data)
        assert abs((result.end - result.start) - 2.0) < 1e-6


from ntt.core.data.components.transforms import VariableWindowCrop
from ntt.core.data.datamodule import NTTDataModule, NTTDataModuleConfig

_FAKE_SELECTIONS = [{"selection": [{"brainset": "test", "sessions": ["s1"]}]}]


class TestDataModuleConfigDistWindow:
    """Tests for NTTDataModuleConfig accepting Dist window_length_seconds."""

    def test_config_accepts_dist(self):
        dist = UniformDist(low=0.1, high=2.0)
        config = NTTDataModuleConfig(
            root="/tmp/fake",
            train_selections=_FAKE_SELECTIONS,
            window_length_seconds=dist,
            bin_size_ms=10.0,
            n_neurons=8,
            input_type="rates",
        )
        assert isinstance(config.window_length_seconds, Dist)

    def test_config_accepts_float(self):
        config = NTTDataModuleConfig(
            root="/tmp/fake",
            train_selections=_FAKE_SELECTIONS,
            window_length_seconds=2.0,
            bin_size_ms=10.0,
            n_neurons=8,
            input_type="rates",
        )
        assert config.window_length_seconds == 2.0

    def test_json_dump_accepts_none_transform_lists(self):
        """None transform lists are valid config defaults and should serialize."""
        config = NTTDataModuleConfig(
            root="/tmp/fake",
            train_selections=_FAKE_SELECTIONS,
            train_transforms=None,
            val_transforms=None,
            test_transforms=None,
            window_length_seconds=2.0,
            bin_size_ms=10.0,
            n_neurons=8,
            input_type="rates",
        )

        dumped = config.model_dump(mode="json")

        assert dumped["train_transforms"] is None
        assert dumped["val_transforms"] is None
        assert dumped["test_transforms"] is None


class TestTransformPipelineWiring:
    """Tests that VariableWindowCrop is injected for train split only."""

    def _make_config(self, window_length_seconds, **kwargs):
        defaults = {
            "root": "/tmp/fake",
            "train_selections": _FAKE_SELECTIONS,
            "bin_size_ms": 10.0,
            "n_neurons": 8,
            "input_type": "rates",
        }
        defaults.update(kwargs)
        return NTTDataModuleConfig(window_length_seconds=window_length_seconds, **defaults)

    def _make_datamodule(self, config):
        return NTTDataModule(config=config)

    def test_crop_injected_for_train_with_dist(self):
        """VariableWindowCrop should be in the train transform pipeline when using Dist."""
        from ntt.contrib.tokenizer.calm.calm_tokenizer import CalmTokenizerSpec

        dist = UniformDist(low=0.1, high=2.0)
        spec = CalmTokenizerSpec(
            n_neurons=8,
            window_length_seconds=dist,
            bin_size_ms=10.0,
            patch_length=10,
            embed_dim=32,
            num_encoder_layers=1,
            num_decoder_layers=1,
            num_heads=4,
        )
        config = self._make_config(dist, tokenizer_spec=spec)
        dm = self._make_datamodule(config)
        transforms = dm._build_transforms("train")
        transform_types = [type(t) for t in transforms.transforms]
        assert VariableWindowCrop in transform_types
        # Should be the first transform
        assert transform_types[0] is VariableWindowCrop

    def test_crop_not_injected_for_val_with_dist(self):
        """VariableWindowCrop should NOT be in val transform pipeline."""
        from ntt.contrib.tokenizer.calm.calm_tokenizer import CalmTokenizerSpec

        dist = UniformDist(low=0.1, high=2.0)
        spec = CalmTokenizerSpec(
            n_neurons=8,
            window_length_seconds=dist,
            bin_size_ms=10.0,
            patch_length=10,
            embed_dim=32,
            num_encoder_layers=1,
            num_decoder_layers=1,
            num_heads=4,
        )
        config = self._make_config(dist, tokenizer_spec=spec)
        dm = self._make_datamodule(config)
        transforms = dm._build_transforms("val")
        transform_types = [type(t) for t in transforms.transforms]
        assert VariableWindowCrop not in transform_types

    def test_crop_not_injected_for_float_window(self):
        """VariableWindowCrop should NOT be injected when window is a plain float."""
        config = self._make_config(2.0)
        dm = self._make_datamodule(config)
        transforms = dm._build_transforms("train")
        transform_types = [type(t) for t in transforms.transforms]
        assert VariableWindowCrop not in transform_types


from ntt.core.components.types import BinnedSpikesInput
from ntt.core.data.datamodule import collate


class TestVariableTCollation:
    """Tests for collation of samples with different T dimensions."""

    def _make_sample(self, T: int, N: int):
        """Create a single (unbatched) BinnedSpikesInput sample."""
        return BinnedSpikesInput(
            spike_matrix=pad2d(torch.ones(T, N, dtype=torch.int32)),
            pad_and_gap_mask=pad2d(torch.ones(T, N, dtype=torch.int32)),
            sampling_rate=100.0,
            unit_ids=torch.arange(N, dtype=torch.int32),
            unit_ids_pad_mask=torch.ones(N, dtype=torch.int32),
            duration_sec=torch.tensor(T / 100.0),
            session_id="test",
            subject_id="test",
            species="test",
            absolute_start=torch.tensor(0.0, dtype=torch.float64),
            absolute_end=torch.tensor(T / 100.0, dtype=torch.float64),
        )

    def test_same_T_no_padding(self):
        """Samples with same T should collate normally."""
        samples = [self._make_sample(200, 8) for _ in range(4)]
        batch = collate(samples)
        assert batch.spike_matrix.shape == (4, 200, 8)
        assert batch.pad_and_gap_mask.shape == (4, 200, 8)
        assert batch.pad_and_gap_mask.sum() == 4 * 200 * 8

    def test_variable_T_padded_to_max(self):
        """Samples with different T should be padded to max T."""
        samples = [
            self._make_sample(100, 8),
            self._make_sample(200, 8),
            self._make_sample(150, 8),
        ]
        batch = collate(samples)
        assert batch.spike_matrix.shape == (3, 200, 8)
        assert batch.pad_and_gap_mask.shape == (3, 200, 8)

    def test_padding_is_zero(self):
        """Padded regions should have zero spike_matrix and zero pad_and_gap_mask."""
        samples = [
            self._make_sample(100, 4),
            self._make_sample(200, 4),
        ]
        batch = collate(samples)
        # First sample: T=100 padded to T=200
        assert batch.spike_matrix[0, :100, :].sum() == 100 * 4
        assert batch.pad_and_gap_mask[0, :100, :].sum() == 100 * 4
        assert batch.spike_matrix[0, 100:, :].sum() == 0
        assert batch.pad_and_gap_mask[0, 100:, :].sum() == 0
        # Second sample: no padding, all ones
        assert batch.spike_matrix[1].sum() == 200 * 4
        assert batch.pad_and_gap_mask[1].sum() == 200 * 4

    def test_duration_preserved(self):
        """duration_sec should reflect original (pre-pad) durations."""
        samples = [
            self._make_sample(100, 4),
            self._make_sample(200, 4),
        ]
        batch = collate(samples)
        assert batch.duration_sec[0] == pytest.approx(1.0)
        assert batch.duration_sec[1] == pytest.approx(2.0)


from ntt.contrib.tokenizer.calm.calm_tokenizer import CalmTokenizer, CalmTokenizerSpec
from ntt.core.components.types import RatesInput, TokenizerOutput


class TestEndToEndVariableWindow:
    """End-to-end test: CALM tokenizer with variable-T padded batch."""

    def _make_padded_batch(self):
        """Create a batch where samples have different original T, padded to max T.

        Sample 0: T_real=100 (10 patches), padded to T=200
        Sample 1: T_real=200 (20 patches), no padding
        """
        N = 4
        max_T = 200

        # Sample 0: real data in [0:100], padding in [100:200]
        rate_0 = torch.zeros(max_T, N)
        rate_0[:100] = torch.rand(100, N)
        mask_0 = torch.zeros(max_T, N, dtype=torch.int32)
        mask_0[:100] = 1

        # Sample 1: full real data
        rate_1 = torch.rand(max_T, N)
        mask_1 = torch.ones(max_T, N, dtype=torch.int32)

        return RatesInput(
            rate_matrix=torch.stack([rate_0, rate_1]),
            pad_and_gap_mask=torch.stack([mask_0, mask_1]),
            sampling_rate=100.0,
            unit_ids=torch.arange(N).unsqueeze(0).expand(2, -1),
            unit_ids_pad_mask=torch.ones(2, N, dtype=torch.int32),
            duration_sec=torch.tensor([1.0, 2.0]),
            session_id=("test", "test"),
            subject_id=("test", "test"),
            species=("test", "test"),
            absolute_start=torch.zeros(2, dtype=torch.float64),
            absolute_end=torch.tensor([1.0, 2.0], dtype=torch.float64),
        )

    def test_forward_with_padded_batch(self):
        """CALM tokenizer should handle a padded variable-T batch."""
        spec = CalmTokenizerSpec(
            n_neurons=4,
            window_length_seconds=2.0,
            bin_size_ms=10.0,
            patch_length=10,
            embed_dim=32,
            num_encoder_layers=1,
            num_decoder_layers=1,
            num_heads=4,
        )
        tokenizer = CalmTokenizer(spec)
        tokenizer.eval()

        batch = self._make_padded_batch()
        output = tokenizer(batch)

        assert isinstance(output, TokenizerOutput)
        assert output.tokens.shape == (2, 20, 4, 1)
        assert output.reconstruction.shape == (2, 200, 4)

    def test_loss_respects_mask(self):
        """Loss should be zero for padded regions when mask is applied."""
        spec = CalmTokenizerSpec(
            n_neurons=4,
            window_length_seconds=2.0,
            bin_size_ms=10.0,
            patch_length=10,
            embed_dim=32,
            num_encoder_layers=1,
            num_decoder_layers=1,
            num_heads=4,
            output_activation="sigmoid",
        )
        tokenizer = CalmTokenizer(spec)
        tokenizer.train()

        batch = self._make_padded_batch()
        output = tokenizer(batch)

        mask = batch.pad_and_gap_mask.float()
        target = batch.rate_matrix.float()
        recon = output.reconstruction

        element_loss = (recon - target) ** 2 * mask
        padded_loss = element_loss[0, 100:, :].sum()
        assert padded_loss == 0.0

    def test_dist_spec_forward(self):
        """CALM tokenizer with Dist window_length_seconds should instantiate and forward."""
        dist = UniformDist(low=0.1, high=2.0)
        spec = CalmTokenizerSpec(
            n_neurons=4,
            window_length_seconds=dist,
            bin_size_ms=10.0,
            patch_length=10,
            embed_dim=32,
            num_encoder_layers=1,
            num_decoder_layers=1,
            num_heads=4,
        )
        tokenizer = CalmTokenizer(spec)
        tokenizer.eval()

        batch = RatesInput(
            rate_matrix=torch.rand(2, 200, 4),
            pad_and_gap_mask=torch.ones(2, 200, 4, dtype=torch.int32),
            sampling_rate=100.0,
            unit_ids=torch.arange(4).unsqueeze(0).expand(2, -1),
            unit_ids_pad_mask=torch.ones(2, 4, dtype=torch.int32),
            duration_sec=torch.tensor([2.0, 2.0]),
            session_id=("test", "test"),
            subject_id=("test", "test"),
            species=("test", "test"),
            absolute_start=torch.zeros(2, dtype=torch.float64),
            absolute_end=torch.tensor([2.0, 2.0], dtype=torch.float64),
        )
        output = tokenizer(batch)
        assert output.tokens.shape == (2, 20, 4, 1)


class TestEvalPipelineDistWindowResolution:
    """Eval-pipeline call sites must use effective_window_length_seconds.

    Variable-t tokenizers (e.g. CALM trained with a UniformDist window) expose
    a Dist for spec.window_length_seconds. Forwarding the raw Dist into APIs
    typed `float` raises at runtime (PoissonBaselineTokenizerSpec) or breaks
    arithmetic (plotting divide).

    Call sites must use spec.effective_window_length_seconds, which resolves
    Dist -> Dist.max and passes floats through unchanged. These tests pin
    both halves: the raw Dist is rejected, and the resolved float succeeds.

    Mirrors:
      - src/ntt/jobs/eval.py:_run_poisson_comparison
      - src/ntt/evaluation/lightning.py (Poisson construction + EvalData duration)
      - src/ntt/evaluation/plotting.py:get_validation_segment (max_windows divide)
    """

    def _make_calm_dist_spec(self, *, low: float = 0.1, high: float = 2.0):
        return CalmTokenizerSpec(
            n_neurons=8,
            window_length_seconds=UniformDist(low=low, high=high),
            bin_size_ms=10.0,
            patch_length=10,
            embed_dim=32,
            num_encoder_layers=1,
            num_decoder_layers=1,
            num_heads=4,
        )

    def test_poisson_baseline_spec_rejects_raw_dist(self):
        """Original failure mode: PoissonBaselineTokenizerSpec.window_length_seconds is
        float-only."""
        from pydantic import ValidationError

        from ntt.contrib.tokenizer.poisson_baseline import PoissonBaselineTokenizerSpec

        spec = self._make_calm_dist_spec()
        with pytest.raises(ValidationError, match="window_length_seconds"):
            PoissonBaselineTokenizerSpec(
                window_length_seconds=spec.window_length_seconds,  # raw Dist
                bin_size_ms=spec.bin_size_ms,
                rate_estimation_bin_ms=10.0,
            )

    def test_poisson_baseline_spec_accepts_effective_window(self):
        """The fix path: spec.effective_window_length_seconds resolves Dist -> Dist.max."""
        from ntt.contrib.tokenizer.poisson_baseline import PoissonBaselineTokenizerSpec

        spec = self._make_calm_dist_spec(low=0.1, high=2.0)
        poisson = PoissonBaselineTokenizerSpec(
            window_length_seconds=spec.effective_window_length_seconds,
            bin_size_ms=spec.bin_size_ms,
            rate_estimation_bin_ms=10.0,
        )
        assert poisson.window_length_seconds == pytest.approx(2.0)

    def test_max_windows_divide_with_effective_window(self):
        """plotting.py:get_validation_segment computes int(seconds / win_len)."""
        spec = self._make_calm_dist_spec(low=0.1, high=2.0)
        # Mirrors plotting.py:775 — must use the resolver, not the raw attr.
        max_windows = int(60.0 / spec.effective_window_length_seconds)
        assert max_windows == 30  # 60 / 2.0

    def test_max_windows_divide_with_raw_dist_fails(self):
        """Original failure mode for plotting.py: divide by Dist raises TypeError."""
        spec = self._make_calm_dist_spec()
        with pytest.raises(TypeError):
            int(60.0 / spec.window_length_seconds)

    def test_eval_data_from_tensor_accepts_effective_window(self):
        """lightning.py builds
        EvalData.from_tensor(duration=spec.effective_window_length_seconds)."""
        from ntt.core.components.types import DataType
        from ntt.evaluation.core.types import EvalData

        spec = self._make_calm_dist_spec(low=0.1, high=2.0)
        # Mirrors lightning.py:1116/1125/1282/1289 call shape.
        eval_data = EvalData.from_tensor(
            torch.zeros(8, 100),  # (n_units, n_bins)
            data_type=DataType.RATES,
            bin_size_seconds=spec.bin_size_ms / 1000.0,
            duration=spec.effective_window_length_seconds,
        )
        assert eval_data.duration == pytest.approx(2.0)
        assert isinstance(eval_data.duration, float)
