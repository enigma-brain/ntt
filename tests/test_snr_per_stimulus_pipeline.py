from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch_brain.data.dataset import DatasetIndex

from ntt.contrib.tokenizer.identity_tokenizer import IdentityTokenizerSpec
from ntt.contrib.tokenizer.simple_vae import SimpleVAETokenizerSpec
from ntt.contrib.tokenizer.wanneural.wanneural import (
    WanNeuralTokenizer,
    WanNeuralTokenizerSpec,
)
from ntt.core.components.types import BinnedSpikesInput
from ntt.core.data.datamodule import collate
from ntt.core.model.tokenizer.base import DataType
from ntt.evaluation.config import EvaluationConfig
from ntt.evaluation.utils.validation_collation import (
    _direct_short_fixed_input_output_is_compatible,
    run_validation_evaluation_per_session,
)


class _StimulusSessionDataset(Dataset):
    def __init__(self) -> None:
        self.session_id = "session"
        self.sampling_rate = 10.0
        self._recording_objects = {
            self.session_id: SimpleNamespace(
                stimuli=SimpleNamespace(
                    start=np.asarray([0.0, 0.3, 0.6, 0.9], dtype=np.float64),
                    end=np.asarray([0.2, 0.5, 0.8, 1.1], dtype=np.float64),
                    image_id=np.asarray([1, 2, 1, 2], dtype=np.int64),
                )
            )
        }
        self.full_spikes = torch.zeros((2, 12), dtype=torch.int32)
        self.full_spikes[:, 0:2] = 1
        self.full_spikes[:, 3:5] = 8
        self.full_spikes[:, 6:8] = 2
        self.full_spikes[:, 9:11] = 9

    def __len__(self) -> int:
        return 1

    def _get_data_object(self, recording_id: str):
        return self._recording_objects[recording_id]

    def __getitem__(self, index: DatasetIndex | int) -> BinnedSpikesInput:
        if isinstance(index, DatasetIndex):
            session_id = index.recording_id
            start = float(index.start)
            end = float(index.end)
        else:
            session_id = self.session_id
            start = 0.0
            end = 1.2

        start_bin = int(round(start * self.sampling_rate))
        end_bin = int(round(end * self.sampling_rate))
        spike_matrix = self.full_spikes[:, start_bin:end_bin].T.contiguous()
        n_units = spike_matrix.shape[1]
        return BinnedSpikesInput(
            spike_matrix=spike_matrix,
            pad_and_gap_mask=torch.ones_like(spike_matrix, dtype=torch.int32),
            sampling_rate=self.sampling_rate,
            unit_ids=torch.arange(n_units, dtype=torch.int64),
            unit_ids_pad_mask=torch.ones(n_units, dtype=torch.int32),
            duration_sec=torch.tensor((end_bin - start_bin) / self.sampling_rate),
            session_id=session_id,
            subject_id="subject",
            species="species",
            absolute_start=torch.tensor(start, dtype=torch.float64),
            absolute_end=torch.tensor(end, dtype=torch.float64),
            behavioral_data=None,
            behavioral_intervals=None,
        )


class _ShortStimulusSessionDataset(Dataset):
    def __init__(self) -> None:
        self.session_id = "short_session"
        self.sampling_rate = 100.0
        self._recording_objects = {
            self.session_id: SimpleNamespace(
                stimuli=SimpleNamespace(
                    start=np.asarray([0.00, 0.12, 0.24, 0.36], dtype=np.float64),
                    end=np.asarray([0.09, 0.21, 0.33, 0.45], dtype=np.float64),
                    image_id=np.asarray([1, 2, 1, 2], dtype=np.int64),
                )
            )
        }
        self.full_spikes = torch.zeros((2, 48), dtype=torch.int32)
        self.full_spikes[:, 0:9] = 1
        self.full_spikes[:, 12:21] = 6
        self.full_spikes[:, 24:33] = 2
        self.full_spikes[:, 36:45] = 7

    def __len__(self) -> int:
        return 1

    def _get_data_object(self, recording_id: str):
        return self._recording_objects[recording_id]

    def __getitem__(self, index: DatasetIndex | int) -> BinnedSpikesInput:
        if isinstance(index, DatasetIndex):
            session_id = index.recording_id
            start = float(index.start)
            end = float(index.end)
        else:
            session_id = self.session_id
            start = 0.0
            end = 0.48

        start_bin = int(round(start * self.sampling_rate))
        end_bin = int(round(end * self.sampling_rate))
        spike_matrix = self.full_spikes[:, start_bin:end_bin].T.contiguous()
        n_units = spike_matrix.shape[1]
        return BinnedSpikesInput(
            spike_matrix=spike_matrix,
            pad_and_gap_mask=torch.ones_like(spike_matrix, dtype=torch.int32),
            sampling_rate=self.sampling_rate,
            unit_ids=torch.arange(n_units, dtype=torch.int64),
            unit_ids_pad_mask=torch.ones(n_units, dtype=torch.int32),
            duration_sec=torch.tensor((end_bin - start_bin) / self.sampling_rate),
            session_id=session_id,
            subject_id="subject",
            species="species",
            absolute_start=torch.tensor(start, dtype=torch.float64),
            absolute_end=torch.tensor(end, dtype=torch.float64),
            behavioral_data=None,
            behavioral_intervals=None,
        )


class _ShortStimulusSessionDataset1ms(Dataset):
    def __init__(self) -> None:
        self.session_id = "short_session_1ms"
        self.sampling_rate = 1000.0
        self._recording_objects = {
            self.session_id: SimpleNamespace(
                stimuli=SimpleNamespace(
                    start=np.asarray([0.00, 0.25, 0.50, 0.75], dtype=np.float64),
                    end=np.asarray([0.09, 0.34, 0.59, 0.84], dtype=np.float64),
                    image_id=np.asarray([1, 2, 1, 2], dtype=np.int64),
                )
            )
        }
        self.full_spikes = torch.zeros((2, 900), dtype=torch.int32)
        self.full_spikes[:, 0:90] = 1
        self.full_spikes[:, 250:340] = 6
        self.full_spikes[:, 500:590] = 2
        self.full_spikes[:, 750:840] = 7

    def __len__(self) -> int:
        return 1

    def _get_data_object(self, recording_id: str):
        return self._recording_objects[recording_id]

    def __getitem__(self, index: DatasetIndex | int) -> BinnedSpikesInput:
        if isinstance(index, DatasetIndex):
            session_id = index.recording_id
            start = float(index.start)
            end = float(index.end)
        else:
            session_id = self.session_id
            start = 0.0
            end = 0.90

        start_bin = int(round(start * self.sampling_rate))
        end_bin = int(round(end * self.sampling_rate))
        spike_matrix = self.full_spikes[:, start_bin:end_bin].T.contiguous()
        n_units = spike_matrix.shape[1]
        return BinnedSpikesInput(
            spike_matrix=spike_matrix,
            pad_and_gap_mask=torch.ones_like(spike_matrix, dtype=torch.int32),
            sampling_rate=self.sampling_rate,
            unit_ids=torch.arange(n_units, dtype=torch.int64),
            unit_ids_pad_mask=torch.ones(n_units, dtype=torch.int32),
            duration_sec=torch.tensor((end_bin - start_bin) / self.sampling_rate),
            session_id=session_id,
            subject_id="subject",
            species="species",
            absolute_start=torch.tensor(start, dtype=torch.float64),
            absolute_end=torch.tensor(end, dtype=torch.float64),
            behavioral_data=None,
            behavioral_intervals=None,
        )


def test_snr_pass_uses_per_stim_sampler_smoke(tmp_path):
    """SNR runs end-to-end on a tiny SimpleDataset with identity tokenizer; produces finite SNR."""
    del tmp_path
    dataset = _StimulusSessionDataset()
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate)
    spec = IdentityTokenizerSpec(window_length_seconds=1.0, bin_size_ms=100.0)
    eval_config = EvaluationConfig(
        interval_source="stimuli",
        fixed_window_duration_ms=200.0,
        cross_stim_padding_ms=80.0,
        label_extension_ms=0.0,
    )

    def identity_tokenizer(batch: BinnedSpikesInput) -> dict[str, torch.Tensor]:
        batch_size, n_bins, _n_units = batch.spike_matrix.shape
        token_start = batch.absolute_start[:, None] + (
            torch.arange(n_bins, dtype=torch.float64)[None, :] / batch.sampling_rate
        )
        token_end = token_start + (1.0 / batch.sampling_rate)
        token_timestamps = (token_start + token_end) / 2.0
        latents = batch.spike_matrix.to(torch.float32).mean(dim=-1, keepdim=True)
        return {
            "reconstruction": batch.spike_matrix.to(torch.float32),
            "latents": latents,
            "token_timestamps": token_timestamps,
            "token_start_time": token_start,
            "token_end_time": token_end,
        }

    reports = run_validation_evaluation_per_session(
        dataloader,
        identity_tokenizer,
        data_type=DataType.SPIKES,
        bin_size_seconds=spec.bin_size_seconds,
        token_bin_size_ms=spec.token_bin_size_ms,
        window_length_seconds=spec.effective_window_length_seconds,
        eval_config=eval_config,
        tokenizer_spec=spec,
        device=torch.device("cpu"),
    )

    assert set(reports) == {"session"}
    assert len(reports["session"]) == 1
    report = reports["session"][0][1]["report"]
    assert "repeats_snr" in report.results
    assert np.isfinite(report.results["repeats_snr"].raw_value)


def test_snr_pass_labels_tokens_by_center_time_for_short_stimuli():
    """Shorter-than-token stimuli should still label a token when its center is inside."""
    dataset = _ShortStimulusSessionDataset()
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate)
    spec = IdentityTokenizerSpec(window_length_seconds=1.0, bin_size_ms=10.0)
    eval_config = EvaluationConfig(
        interval_source="stimuli",
        fixed_window_duration_ms=100.0,
        label_extension_ms=0.0,
    )

    def tokenizer_with_100ms_token(batch: BinnedSpikesInput) -> dict[str, torch.Tensor]:
        batch_size = batch.spike_matrix.shape[0]
        token_start = batch.absolute_start[:, None]
        token_end = token_start + 0.1
        token_timestamps = (token_start + token_end) / 2.0
        latents = (
            batch.spike_matrix.to(torch.float32)
            .sum(dim=(1, 2), keepdim=True)
            .reshape(batch_size, 1, 1)
        )
        return {
            "reconstruction": batch.spike_matrix.to(torch.float32),
            "latents": latents,
            "token_timestamps": token_timestamps,
            "token_start_time": token_start,
            "token_end_time": token_end,
        }

    reports = run_validation_evaluation_per_session(
        dataloader,
        tokenizer_with_100ms_token,
        data_type=DataType.SPIKES,
        bin_size_seconds=spec.bin_size_seconds,
        token_bin_size_ms=100.0,
        window_length_seconds=spec.effective_window_length_seconds,
        eval_config=eval_config,
        tokenizer_spec=spec,
        device=torch.device("cpu"),
    )

    report = reports["short_session"][0][1]["report"]
    assert "repeats_snr" in report.results
    assert np.isfinite(report.results["repeats_snr"].raw_value)


def test_incompatible_fixed_input_tokenizer_skips_after_short_window_probe():
    """Fixed-input tokenizers that reject short raw stimulus windows should skip cleanly."""
    dataset = _ShortStimulusSessionDataset()
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate)
    spec = SimpleVAETokenizerSpec(
        window_length_seconds=1.0,
        bin_size_ms=10.0,
        latent_dim=16,
        hidden_dims=(32,),
    )
    eval_config = EvaluationConfig(
        interval_source="stimuli",
        fixed_window_duration_ms=100.0,
        label_extension_ms=0.0,
    )
    calls = 0

    def should_not_run(_batch: BinnedSpikesInput) -> dict[str, torch.Tensor]:
        nonlocal calls
        calls += 1
        raise RuntimeError("size mismatch in short-window probe")

    reports = run_validation_evaluation_per_session(
        dataloader,
        should_not_run,
        data_type=DataType.SPIKES,
        bin_size_seconds=spec.bin_size_seconds,
        token_bin_size_ms=100.0,
        window_length_seconds=spec.effective_window_length_seconds,
        eval_config=eval_config,
        tokenizer_spec=spec,
        device=torch.device("cpu"),
    )

    assert reports == {}
    assert calls == 1


def test_token_bin_larger_than_fixed_stim_window_skips_without_tokenizer_call():
    """A token longer than the fixed stimulus window should skip before probing."""
    dataset = _ShortStimulusSessionDataset1ms()
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate)
    spec = IdentityTokenizerSpec(window_length_seconds=1.0, bin_size_ms=1.0)
    eval_config = EvaluationConfig(
        interval_source="stimuli",
        fixed_window_duration_ms=200.0,
        cross_stim_padding_ms=80.0,
        label_extension_ms=0.0,
    )
    calls = 0

    def should_not_run(_batch: BinnedSpikesInput) -> dict[str, torch.Tensor]:
        nonlocal calls
        calls += 1
        raise AssertionError(
            "Tokenizer should not be called when token bins exceed the stimulus window."
        )

    reports = run_validation_evaluation_per_session(
        dataloader,
        should_not_run,
        data_type=DataType.SPIKES,
        bin_size_seconds=spec.bin_size_seconds,
        token_bin_size_ms=1000.0,
        window_length_seconds=spec.effective_window_length_seconds,
        eval_config=eval_config,
        tokenizer_spec=spec,
        device=torch.device("cpu"),
    )

    assert reports == {}
    assert calls == 0


def test_wanneural_1s_100ms_tokens_are_compatible_with_200ms_stimulus_windows():
    """WanNeural 100 ms-token models should pass the short-window compatibility check."""
    dataset = _ShortStimulusSessionDataset1ms()
    spec = WanNeuralTokenizerSpec(
        n_neurons=2,
        window_length_seconds=1.0,
        bin_size_ms=1.0,
        dim=8,
        z_dim=4,
        num_levels=3,
        dim_mult=(1, 2, 4),
        num_res_blocks=1,
        attn_levels=(),
        num_attn_heads=1,
        temporal_patch_size=25,
        temporal_downsample=(True, True),
        token_bin_size_ms=100.0,
        bottleneck_type="fsq",
        q_levels=(8, 5, 5, 5),
    )
    tok = WanNeuralTokenizer(spec)
    batch = collate(
        [
            dataset[
                DatasetIndex(
                    recording_id=dataset.session_id,
                    start=0.0,
                    end=0.2,
                )
            ]
        ]
    )
    with torch.inference_mode():
        output = tok.reconstruct_from_input(batch, return_latents=True)
    compatible, reason = _direct_short_fixed_input_output_is_compatible(
        batch=batch,
        batched_output=output,
        token_bin_size_ms=spec.token_bin_size_ms,
        fixed_window_duration_ms=200.0,
    )

    assert compatible, reason
