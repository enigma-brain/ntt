"""Smoke tests for the binned NuCLR encoder and adapters."""

import pytest
import torch

from ntt.contrib.nuclr.adapter import (
    ntt_rates_to_nuclr_format,
    ntt_to_nuclr_binned_format,
)
from ntt.contrib.nuclr.encoder_binned import NuclrBinnedEncoder
from ntt.core.components.types import RatesInput, SpikeTimesInput

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def encoder():
    return NuclrBinnedEncoder(
        ctx_duration=2.0,
        latent_step=1.0,
        bin_size=0.02,
        dim=64,
        self_heads=2,
        dim_head=32,
        atn_dropout=0.0,
        lin_dropout=0.0,
        t_layers=1,
        st_layers=1,
        rot_ratio=0.5,
    ).to(DEVICE)


class TestNuclrBinnedEncoder:
    @requires_cuda
    def test_forward_shape(self, encoder: NuclrBinnedEncoder):
        """Encoder produces (total_units, dim) output."""
        num_units = 5
        num_latents = encoder.num_latents  # 2
        bins_per_latent = encoder.bins_per_latent  # 50

        bins = torch.randn(num_units * num_latents, bins_per_latent, device=DEVICE)
        unit_seqlen = torch.tensor([3, 2], device=DEVICE)

        out = encoder(bins, unit_seqlen)
        assert out.shape == (num_units, encoder.dim)

    @requires_cuda
    def test_forward_single_batch(self, encoder: NuclrBinnedEncoder):
        """Works with a single batch element."""
        num_units = 4
        bins = torch.randn(num_units * encoder.num_latents, encoder.bins_per_latent, device=DEVICE)
        unit_seqlen = torch.tensor([4], device=DEVICE)

        out = encoder(bins, unit_seqlen)
        assert out.shape == (4, encoder.dim)

    def test_derived_params(self, encoder: NuclrBinnedEncoder):
        """bins_per_latent and num_latents are correctly derived."""
        assert encoder.bins_per_latent == 50  # 1.0 / 0.02
        assert encoder.num_latents == 2  # ceil(2.0 / 1.0)

    def test_emb_dim(self, encoder: NuclrBinnedEncoder):
        """emb_dim matches dim for compatibility with contrastive loss."""
        assert encoder.emb_dim == 64


class TestNttToNuclrBinnedFormat:
    """Tests for the spike-times-based binned adapter (bins internally)."""

    def _make_spike_times_input(self, batch_size=2):
        """Create a minimal SpikeTimesInput for testing."""
        all_units = []
        all_times = []
        all_unit_idx = []
        for b in range(batch_size):
            n_units = 3 if b == 0 else 2
            n_spikes = 10 if b == 0 else 8
            times = torch.rand(n_spikes) * 2.0
            unit_idx = torch.randint(0, n_units, (n_spikes,))
            all_times.append(times)
            all_unit_idx.append(unit_idx)
            all_units.append(torch.arange(n_units))

        max_spikes = max(len(t) for t in all_times)
        max_units = max(len(u) for u in all_units)

        spike_timestamps = torch.zeros(batch_size, max_spikes)
        spike_unit_index = torch.zeros(batch_size, max_spikes, dtype=torch.long)
        spike_pad_mask = torch.zeros(batch_size, max_spikes, dtype=torch.long)
        unit_ids = torch.zeros(batch_size, max_units, dtype=torch.long)
        unit_ids_pad_mask = torch.zeros(batch_size, max_units, dtype=torch.long)

        for b in range(batch_size):
            n_s = len(all_times[b])
            n_u = len(all_units[b])
            spike_timestamps[b, :n_s] = all_times[b]
            spike_unit_index[b, :n_s] = all_unit_idx[b]
            spike_pad_mask[b, :n_s] = 1
            unit_ids[b, :n_u] = all_units[b]
            unit_ids_pad_mask[b, :n_u] = 1

        return SpikeTimesInput(
            spike_unit_index=spike_unit_index,
            spike_timestamps=spike_timestamps,
            spike_pad_mask=spike_pad_mask,
            unit_ids=unit_ids,
            unit_ids_pad_mask=unit_ids_pad_mask,
            duration_sec=torch.tensor([2.0, 2.0]),
            session_id=("sess1", "sess2"),
            subject_id=("subj1", "subj2"),
            species=("mouse", "mouse"),
        )

    def test_output_shapes(self):
        """Adapter produces correct shapes."""
        x = self._make_spike_times_input()
        result = ntt_to_nuclr_binned_format(x, 2.0, 0.02, 2, 50)

        total_units = 3 + 2
        assert result.bins.shape == (total_units * 2, 50)
        assert result.unit_seqlen.shape == (2,)
        assert result.unit_seqlen.tolist() == [3, 2]
        assert result.unit_ids.shape == (total_units,)
        assert result.batch_size == 2

    @requires_cuda
    def test_end_to_end_with_encoder(self, encoder: NuclrBinnedEncoder):
        """Full pipeline: SpikeTimesInput → adapter → encoder → embeddings."""
        x = self._make_spike_times_input()
        result = ntt_to_nuclr_binned_format(
            x,
            ctx_duration=encoder.ctx_duration,
            bin_size=encoder.bin_size,
            num_latents=encoder.num_latents,
            bins_per_latent=encoder.bins_per_latent,
        )
        out = encoder(result.bins.to(DEVICE), result.unit_seqlen.to(DEVICE))
        total_units = 3 + 2
        assert out.shape == (total_units, encoder.dim)


class TestNttRatesToNuclrFormat:
    """Tests for the pipeline-based RatesInput adapter."""

    def _make_rates_input(self, batch_size=2):
        """Create a minimal RatesInput as the NTT pipeline would produce.

        Simulates: ctx_duration=2.0, bin_size=0.02 → 100 time bins.
        batch 0: 3 units, batch 1: 2 units → padded to 3.
        """
        num_time_bins = 100
        max_units = 3

        rate_matrix = torch.zeros(batch_size, num_time_bins, max_units)
        pad_mask = torch.zeros(batch_size, num_time_bins, max_units, dtype=torch.long)
        unit_ids = torch.zeros(batch_size, max_units, dtype=torch.long)
        unit_ids_pad_mask = torch.zeros(batch_size, max_units, dtype=torch.long)

        for b in range(batch_size):
            n_units = 3 if b == 0 else 2
            rate_matrix[b, :, :n_units] = torch.randint(0, 5, (num_time_bins, n_units)).float()
            pad_mask[b, :, :n_units] = 1
            unit_ids[b, :n_units] = torch.arange(n_units)
            unit_ids_pad_mask[b, :n_units] = 1

        return RatesInput(
            rate_matrix=rate_matrix,
            pad_and_gap_mask=pad_mask,
            sampling_rate=50.0,  # 1/0.02
            unit_ids=unit_ids,
            unit_ids_pad_mask=unit_ids_pad_mask,
            duration_sec=torch.tensor([2.0, 2.0]),
            session_id=("sess1", "sess2"),
            subject_id=("subj1", "subj2"),
            species=("mouse", "mouse"),
        )

    def test_output_shapes(self):
        """Adapter produces correct shapes."""
        x = self._make_rates_input()
        result = ntt_rates_to_nuclr_format(x, num_latents=2, bins_per_latent=50)

        total_units = 3 + 2
        assert result.bins.shape == (total_units * 2, 50)
        assert result.unit_seqlen.shape == (2,)
        assert result.unit_seqlen.tolist() == [3, 2]
        assert result.unit_ids.shape == (total_units,)
        assert result.batch_size == 2

    def test_values_preserved(self):
        """Bin counts are correctly transposed and reshaped."""
        # Single batch, 2 units, 4 time bins, 2 latents of 2 bins each
        rate_matrix = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])
        x = RatesInput(
            rate_matrix=rate_matrix,
            pad_and_gap_mask=torch.ones(1, 4, 2, dtype=torch.long),
            sampling_rate=2.0,
            unit_ids=torch.tensor([[0, 1]]),
            unit_ids_pad_mask=torch.ones(1, 2, dtype=torch.long),
            duration_sec=torch.tensor([2.0]),
            session_id=("sess1",),
            subject_id=("subj1",),
            species=("mouse",),
        )
        result = ntt_rates_to_nuclr_format(x, num_latents=2, bins_per_latent=2)
        # rate_matrix is (1, 4, 2) → per unit: unit0=[1,3,5,7], unit1=[2,4,6,8]
        # Reshaped to (U*L, B) with L=2, B=2:
        # unit0_latent0=[1,3], unit0_latent1=[5,7], unit1_latent0=[2,4], unit1_latent1=[6,8]
        expected = torch.tensor([[1, 3], [5, 7], [2, 4], [6, 8]], dtype=torch.float)
        assert torch.equal(result.bins, expected)

    @requires_cuda
    def test_end_to_end_with_encoder(self, encoder: NuclrBinnedEncoder):
        """Full pipeline: RatesInput → adapter → encoder → embeddings."""
        x = self._make_rates_input()
        result = ntt_rates_to_nuclr_format(
            x,
            num_latents=encoder.num_latents,
            bins_per_latent=encoder.bins_per_latent,
        )
        out = encoder(result.bins.to(DEVICE), result.unit_seqlen.to(DEVICE))
        total_units = 3 + 2
        assert out.shape == (total_units, encoder.dim)

    def test_zero_counts(self):
        """Adapter handles zero-count rate matrix."""
        x = RatesInput(
            rate_matrix=torch.zeros(1, 100, 2),
            pad_and_gap_mask=torch.ones(1, 100, 2, dtype=torch.long),
            sampling_rate=50.0,
            unit_ids=torch.arange(2).unsqueeze(0),
            unit_ids_pad_mask=torch.ones(1, 2, dtype=torch.long),
            duration_sec=torch.tensor([2.0]),
            session_id=("sess1",),
            subject_id=("subj1",),
            species=("mouse",),
        )
        result = ntt_rates_to_nuclr_format(x, num_latents=2, bins_per_latent=50)
        assert result.bins.shape == (4, 50)
        assert result.bins.sum() == 0

    def test_short_time_dimension_pads(self):
        """Adapter zero-pads when time bins < num_latents * bins_per_latent."""
        # Only 60 time bins but need 100 (2 latents * 50 bins)
        x = RatesInput(
            rate_matrix=torch.ones(1, 60, 2),
            pad_and_gap_mask=torch.ones(1, 60, 2, dtype=torch.long),
            sampling_rate=50.0,
            unit_ids=torch.arange(2).unsqueeze(0),
            unit_ids_pad_mask=torch.ones(1, 2, dtype=torch.long),
            duration_sec=torch.tensor([1.2]),
            session_id=("sess1",),
            subject_id=("subj1",),
            species=("mouse",),
        )
        result = ntt_rates_to_nuclr_format(x, num_latents=2, bins_per_latent=50)
        assert result.bins.shape == (4, 50)
        # First latent (bins 0-49): all ones for first 50 bins
        # Second latent (bins 50-99): first 10 are ones, last 40 are zero-padded
        unit0_latent1 = result.bins[1]  # unit 0, latent 1
        assert unit0_latent1[:10].sum() == 10
        assert unit0_latent1[10:].sum() == 0
