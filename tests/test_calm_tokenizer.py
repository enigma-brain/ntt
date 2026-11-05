"""Tests for CALM tokenizer components and end-to-end behavior."""

import pytest
import torch

from ntt.contrib.tokenizer.calm.rope import RotaryPositionalEmbedding, apply_rotary_emb
from ntt.core.components.types import DataType, ReconstructionOutput


class TestRoPE:
    """Tests for Rotary Positional Embedding."""

    def test_output_shape(self):
        """RoPE should not change tensor shape."""
        rope = RotaryPositionalEmbedding(dim=64)
        seq_len = 20
        cos, sin = rope(seq_len, device=torch.device("cpu"))
        assert cos.shape == (seq_len, 64)
        assert sin.shape == (seq_len, 64)

    def test_apply_rotary_emb_shape(self):
        """Applying RoPE to q, k should preserve shape."""
        rope = RotaryPositionalEmbedding(dim=16)
        B, H, T, D = 2, 4, 10, 16
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        cos, sin = rope(T, device=q.device)
        q_rot, k_rot = apply_rotary_emb(q, k, cos, sin)
        assert q_rot.shape == (B, H, T, D)
        assert k_rot.shape == (B, H, T, D)

    def test_different_positions_different_embeddings(self):
        """Different sequence positions should produce different embeddings."""
        rope = RotaryPositionalEmbedding(dim=64)
        cos, sin = rope(10, device=torch.device("cpu"))
        assert not torch.allclose(cos[0], cos[1])

    def test_deterministic(self):
        """Same inputs should produce same outputs."""
        rope = RotaryPositionalEmbedding(dim=64)
        cos1, sin1 = rope(10, device=torch.device("cpu"))
        cos2, sin2 = rope(10, device=torch.device("cpu"))
        assert torch.allclose(cos1, cos2)
        assert torch.allclose(sin1, sin2)


from ntt.contrib.tokenizer.calm.encoder import CalmEncoder


class TestCalmEncoder:
    def test_output_shape(self):
        encoder = CalmEncoder(embed_dim=32, num_layers=2, num_heads=4, patch_length=5, causal=True)
        x = torch.randn(6, 20, 1)
        out = encoder(x)
        assert out.shape == (6, 4, 32)

    def test_temporal_compression(self):
        for T, patch_length in [(100, 10), (50, 5), (20, 4)]:
            encoder = CalmEncoder(
                embed_dim=16, num_layers=1, num_heads=2, patch_length=patch_length, causal=False
            )
            x = torch.randn(2, T, 1)
            out = encoder(x)
            assert out.shape[1] == T // patch_length

    def test_causal_masking(self):
        encoder = CalmEncoder(embed_dim=16, num_layers=2, num_heads=2, patch_length=2, causal=True)
        encoder.eval()
        x = torch.randn(1, 10, 1)
        out1 = encoder(x)
        x_perturbed = x.clone()
        x_perturbed[:, 6:] = torch.randn(1, 4, 1)
        out2 = encoder(x_perturbed)
        # First 3 patches (bins 0-5) should be unchanged
        assert torch.allclose(out1[:, :3], out2[:, :3], atol=1e-5)

    def test_bidirectional_mode(self):
        encoder = CalmEncoder(
            embed_dim=16, num_layers=2, num_heads=2, patch_length=2, causal=False
        )
        encoder.eval()
        x = torch.randn(1, 10, 1)
        out1 = encoder(x)
        x_perturbed = x.clone()
        x_perturbed[:, 8:] = torch.randn(1, 2, 1)
        out2 = encoder(x_perturbed)
        assert not torch.allclose(out1[:, 0], out2[:, 0], atol=1e-3)


from ntt.contrib.tokenizer.calm.decoder import CalmDecoder


class TestCalmDecoder:
    def test_output_shape(self):
        decoder = CalmDecoder(embed_dim=32, num_layers=2, num_heads=4, patch_length=5)
        z = torch.randn(6, 4, 32)
        out = decoder(z)
        assert out.shape == (6, 20, 1)

    def test_temporal_expansion(self):
        for T_d, patch_length in [(10, 10), (5, 5), (4, 4)]:
            decoder = CalmDecoder(
                embed_dim=16, num_layers=1, num_heads=2, patch_length=patch_length
            )
            z = torch.randn(2, T_d, 16)
            out = decoder(z)
            assert out.shape == (2, T_d * patch_length, 1)

    def test_always_bidirectional(self):
        decoder = CalmDecoder(embed_dim=16, num_layers=2, num_heads=2, patch_length=2)
        decoder.eval()
        z = torch.randn(1, 5, 16)
        out1 = decoder(z)
        z_perturbed = z.clone()
        z_perturbed[:, -1] = torch.randn(16)
        out2 = decoder(z_perturbed)
        assert not torch.allclose(out1[:, 0], out2[:, 0], atol=1e-3)


# ---------------------------------------------------------------------------
# End-to-end tests for CalmTokenizer
# ---------------------------------------------------------------------------

from ntt.contrib.tokenizer.calm.calm_tokenizer import CalmTokenizer, CalmTokenizerSpec
from ntt.core.components.types import BinnedSpikesInput, RatesInput, TokenizerOutput


def _make_rates_input(batch_size, n_bins, n_units):
    rate_matrix = torch.rand(batch_size, n_bins, n_units)
    return RatesInput(
        rate_matrix=rate_matrix,
        pad_and_gap_mask=torch.ones_like(rate_matrix, dtype=torch.int32),
        sampling_rate=1000.0 / 10.0,
        unit_ids=torch.arange(n_units, dtype=torch.int32).unsqueeze(0).expand(batch_size, -1),
        unit_ids_pad_mask=torch.ones(batch_size, n_units, dtype=torch.int32),
        duration_sec=torch.full((batch_size,), n_bins * 0.01),
        session_id=("test",) * batch_size,
        subject_id=("test",) * batch_size,
        species=("test",) * batch_size,
        absolute_start=torch.zeros(batch_size, dtype=torch.float64),
        absolute_end=torch.full((batch_size,), n_bins * 0.01, dtype=torch.float64),
    )


def _make_binned_input(batch_size, n_bins, n_units):
    spike_matrix = torch.rand(batch_size, n_bins, n_units)
    return BinnedSpikesInput(
        spike_matrix=spike_matrix,
        pad_and_gap_mask=torch.ones_like(spike_matrix, dtype=torch.int32),
        sampling_rate=100.0,
        unit_ids=torch.arange(n_units, dtype=torch.int32).unsqueeze(0).expand(batch_size, -1),
        unit_ids_pad_mask=torch.ones(batch_size, n_units, dtype=torch.int32),
        duration_sec=torch.full((batch_size,), n_bins / 100.0),
        session_id=("test",) * batch_size,
        subject_id=("test",) * batch_size,
        species=("test",) * batch_size,
        absolute_start=torch.zeros(batch_size, dtype=torch.float64),
        absolute_end=torch.full((batch_size,), n_bins / 100.0, dtype=torch.float64),
    )


def _small_spec(**overrides):
    """Create a small CALM spec suitable for testing."""
    defaults = {
        "n_neurons": 8,
        "window_length_seconds": 0.2,
        "bin_size_ms": 10.0,
        "patch_length": 5,
        "embed_dim": 32,
        "num_encoder_layers": 1,
        "num_decoder_layers": 1,
        "num_heads": 4,
        "codebook_size": 16,
        "bottleneck_type": "vq",
        "ar_weight": 0.5,
        "ar_horizon": 1,
    }
    defaults.update(overrides)
    return CalmTokenizerSpec(**defaults)


class TestCalmTokenizerOutputShape:
    @pytest.fixture
    def spec(self):
        return _small_spec()

    def test_forward_output_types(self, spec):
        tokenizer = CalmTokenizer(spec)
        tokenizer.train()
        x = _make_rates_input(2, 20, 8)
        output = tokenizer(x)
        assert isinstance(output, TokenizerOutput)
        assert output.tokens is not None
        assert output.reconstruction is not None
        assert output.aux_loss is not None

    def test_token_shape_4d(self, spec):
        tokenizer = CalmTokenizer(spec)
        tokenizer.eval()
        x = _make_rates_input(2, 20, 8)
        output = tokenizer(x)
        B, T_d, N, D = output.tokens.shape
        assert B == 2
        assert T_d == 4  # 20 / 5
        assert N == 8
        assert D == 1

    def test_reconstruction_shape(self, spec):
        tokenizer = CalmTokenizer(spec)
        tokenizer.eval()
        x = _make_rates_input(2, 20, 8)
        output = tokenizer(x)
        assert output.reconstruction.shape == (2, 20, 8)

    def test_accepts_binned_spikes_input(self, spec):
        tokenizer = CalmTokenizer(spec)
        tokenizer.eval()
        x = _make_binned_input(2, 20, 8)
        output = tokenizer(x)
        assert output.tokens.shape == (2, 4, 8, 1)


class TestCalmTokenizerVariableN:
    def test_different_n_same_model(self):
        spec = _small_spec()
        tokenizer = CalmTokenizer(spec)
        tokenizer.eval()
        for n_units in [4, 8, 16]:
            x = _make_rates_input(2, 20, n_units)
            output = tokenizer(x)
            assert output.tokens.shape == (2, 4, n_units, 1)
            assert output.reconstruction.shape == (2, 20, n_units)


class TestCalmTokenizerBottlenecks:
    @pytest.mark.parametrize("bottleneck_type", ["vq", "fsq", "bsq"])
    def test_bottleneck_forward(self, bottleneck_type):
        spec = _small_spec(
            bottleneck_type=bottleneck_type,
            q_levels=(4, 4),
            bsq_bits=4,
        )
        tokenizer = CalmTokenizer(spec)
        tokenizer.eval()
        x = _make_rates_input(2, 20, 4)
        output = tokenizer(x)
        assert output.tokens.ndim == 4
        assert output.reconstruction.shape == (2, 20, 4)

    @pytest.mark.parametrize(
        "bottleneck_type,expected_vocab",
        [
            ("vq", 16),
            ("fsq", 16),  # prod(4, 4) = 16
            ("bsq", 16),  # 2^4 = 16
        ],
    )
    def test_vocab_size(self, bottleneck_type, expected_vocab):
        spec = _small_spec(
            bottleneck_type=bottleneck_type,
            q_levels=(4, 4),
            bsq_bits=4,
        )
        assert spec.get_vocab_size() == expected_vocab


class TestCalmTokenizerCausality:
    def test_causal_encoding_independent_of_future(self):
        spec = _small_spec(causal=True, num_encoder_layers=2)
        tokenizer = CalmTokenizer(spec)
        tokenizer.eval()
        x = _make_rates_input(1, 20, 2)
        tokens1 = tokenizer.encode(x)
        x2 = RatesInput(
            rate_matrix=torch.cat([x.rate_matrix[:, :10], torch.rand(1, 10, 2)], dim=1),
            pad_and_gap_mask=x.pad_and_gap_mask,
            sampling_rate=x.sampling_rate,
            unit_ids=x.unit_ids,
            unit_ids_pad_mask=x.unit_ids_pad_mask,
            duration_sec=x.duration_sec,
            session_id=x.session_id,
            subject_id=x.subject_id,
            species=x.species,
            absolute_start=x.absolute_start,
            absolute_end=x.absolute_end,
        )
        tokens2 = tokenizer.encode(x2)
        assert torch.equal(tokens1[:, :2], tokens2[:, :2])


class TestCalmTokenizerLearning:
    def test_reconstruction_loss_decreases(self):
        torch.manual_seed(42)
        spec = _small_spec(ar_weight=0.0, codebook_size=8, n_neurons=4)
        tokenizer = CalmTokenizer(spec)
        tokenizer.train()
        optimizer = torch.optim.Adam(tokenizer.parameters(), lr=1e-3)
        t_vals = torch.linspace(0, 2 * 3.14159, 10).unsqueeze(0).unsqueeze(-1)
        signal = torch.sin(t_vals).expand(4, -1, 4)
        x = RatesInput(
            rate_matrix=signal,
            pad_and_gap_mask=torch.ones(4, 10, 4, dtype=torch.int32),
            sampling_rate=100.0,
            unit_ids=torch.arange(4, dtype=torch.int32).unsqueeze(0).expand(4, -1),
            unit_ids_pad_mask=torch.ones(4, 4, dtype=torch.int32),
            duration_sec=torch.full((4,), 0.1),
            session_id=("test",) * 4,
            subject_id=("test",) * 4,
            species=("test",) * 4,
            absolute_start=torch.zeros(4, dtype=torch.float64),
            absolute_end=torch.full((4,), 0.1, dtype=torch.float64),
        )
        losses = []
        for _ in range(30):
            output = tokenizer(x)
            recon_loss = torch.nn.functional.mse_loss(output.reconstruction, signal)
            loss = recon_loss + (output.aux_loss if output.aux_loss is not None else 0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(recon_loss.item())
        assert losses[-1] < losses[0]


class TestCalmTokenizerReconstruct:
    """Tests for the reconstruct() method and its sampling modes."""

    def _encode_once(self, spec):
        tokenizer = CalmTokenizer(spec)
        tokenizer.eval()
        x = _make_rates_input(2, 20, 8)
        tokens = tokenizer.encode(x)
        return tokenizer, tokens

    def test_reconstruct_none_returns_rates(self):
        spec = _small_spec(output_activation="softplus", reconstruct_sample="none")
        tokenizer, tokens = self._encode_once(spec)
        result = tokenizer.reconstruct(tokens)
        assert isinstance(result, ReconstructionOutput)
        assert result.data_type == DataType.RATES
        assert result.reconstruction.shape == (2, 20, 8)
        # softplus output is non-negative
        assert (result.reconstruction >= 0).all()

    def test_reconstruct_poisson_returns_integer_spikes(self):
        spec = _small_spec(output_activation="softplus", reconstruct_sample="poisson")
        tokenizer, tokens = self._encode_once(spec)
        result = tokenizer.reconstruct(tokens)
        assert result.data_type == DataType.SPIKES
        assert result.reconstruction.shape == (2, 20, 8)
        # Poisson output should be non-negative integers
        assert (result.reconstruction >= 0).all()
        assert torch.equal(result.reconstruction, result.reconstruction.floor())

    def test_reconstruct_round_returns_integer_spikes(self):
        spec = _small_spec(output_activation="softplus", reconstruct_sample="round")
        tokenizer, tokens = self._encode_once(spec)
        result = tokenizer.reconstruct(tokens)
        assert result.data_type == DataType.SPIKES
        assert (result.reconstruction >= 0).all()
        assert torch.equal(result.reconstruction, result.reconstruction.floor())

    def test_reconstruct_bernoulli_returns_binary(self):
        spec = _small_spec(
            bin_size_ms=1.0,
            reconstruct_sample="bernoulli",
            window_length_seconds=0.02,
            patch_length=4,
            output_activation="softplus",
        )
        tokenizer, tokens = self._encode_once(spec)
        result = tokenizer.reconstruct(tokens)
        assert result.data_type == DataType.SPIKES
        assert result.reconstruction.shape == (2, 20, 8)
        # Bernoulli output should be 0 or 1
        unique_vals = result.reconstruction.unique()
        assert all(v in (0.0, 1.0) for v in unique_vals)

    def test_reconstruct_bin_size_seconds(self):
        spec = _small_spec(reconstruct_sample="none")
        tokenizer, tokens = self._encode_once(spec)
        result = tokenizer.reconstruct(tokens)
        assert result.bin_size_seconds == pytest.approx(0.01)

    def test_bernoulli_rejected_for_coarse_bins(self):
        with pytest.raises(ValueError, match="bernoulli.*too coarse"):
            _small_spec(bin_size_ms=10.0, reconstruct_sample="bernoulli")


class TestCalmTokenizerTokensToLatents:
    """Tests for the tokens_to_latents() hook (discrete → continuous)."""

    @pytest.mark.parametrize("bottleneck_type", ["vq", "fsq", "bsq"])
    def test_returns_continuous_vectors(self, bottleneck_type):
        spec = _small_spec(bottleneck_type=bottleneck_type, q_levels=(4, 4), bsq_bits=4)
        tokenizer = CalmTokenizer(spec)
        tokenizer.eval()
        x = _make_rates_input(2, 20, 4)
        tokens = tokenizer.encode(x)  # [B, T_d, N, 1] integers
        latents = tokenizer.tokens_to_latents(tokens)  # [B, T_d, N, C]

        assert latents.ndim == 4
        B, T_d, N, C = latents.shape
        assert B == 2
        assert T_d == tokens.shape[1]
        assert N == 4
        assert C == spec.embed_dim  # continuous embedding dimension
        # Must be floating point, not integer indices
        assert latents.dtype in (torch.float32, torch.float16, torch.bfloat16)

    def test_consistent_with_forward_latents(self):
        """tokens_to_latents should produce the same vectors as forward().latents."""
        spec = _small_spec()
        tokenizer = CalmTokenizer(spec)
        tokenizer.eval()
        x = _make_rates_input(2, 20, 8)

        output = tokenizer(x)
        latents_from_forward = output.latents  # [B, T_d, N, C]

        tokens = tokenizer.encode(x)
        latents_from_hook = tokenizer.tokens_to_latents(tokens)

        assert torch.allclose(latents_from_forward, latents_from_hook, atol=1e-5)


class TestRoPEMultiheadAttentionDropout:
    def test_accepts_dropout_arg(self):
        from ntt.contrib.tokenizer.calm.encoder import RoPEMultiheadAttention

        attn = RoPEMultiheadAttention(embed_dim=16, num_heads=4, dropout=0.3)
        assert attn.dropout == 0.3

    def test_default_dropout_zero(self):
        from ntt.contrib.tokenizer.calm.encoder import RoPEMultiheadAttention

        attn = RoPEMultiheadAttention(embed_dim=16, num_heads=4)
        assert attn.dropout == 0.0

    def test_dropout_stochastic_in_train_mode(self):
        from ntt.contrib.tokenizer.calm.encoder import RoPEMultiheadAttention

        torch.manual_seed(0)
        attn = RoPEMultiheadAttention(embed_dim=16, num_heads=4, dropout=0.5)
        attn.train()
        x = torch.randn(2, 10, 16)
        torch.manual_seed(1)
        out1 = attn(x)
        torch.manual_seed(2)
        out2 = attn(x)
        # Two stochastic forwards with different seeds must differ somewhere.
        assert not torch.allclose(out1, out2)

    def test_dropout_deterministic_in_eval_mode(self):
        from ntt.contrib.tokenizer.calm.encoder import RoPEMultiheadAttention

        attn = RoPEMultiheadAttention(embed_dim=16, num_heads=4, dropout=0.5)
        attn.eval()
        x = torch.randn(2, 10, 16)
        out1 = attn(x)
        out2 = attn(x)
        assert torch.allclose(out1, out2)


class TestTransformerBlockDropout:
    def test_accepts_dropout(self):
        from ntt.contrib.tokenizer.calm.encoder import TransformerBlock

        block = TransformerBlock(embed_dim=16, num_heads=4, dropout=0.2)
        assert block.attn_drop.p == 0.2
        assert block.ffn_drop.p == 0.2

    def test_default_dropout_zero(self):
        from ntt.contrib.tokenizer.calm.encoder import TransformerBlock

        block = TransformerBlock(embed_dim=16, num_heads=4)
        assert block.attn_drop.p == 0.0
        assert block.ffn_drop.p == 0.0

    def test_attn_submodule_receives_dropout(self):
        from ntt.contrib.tokenizer.calm.encoder import TransformerBlock

        block = TransformerBlock(embed_dim=16, num_heads=4, dropout=0.3)
        assert block.attn.dropout == 0.3

    def test_train_mode_stochastic(self):
        from ntt.contrib.tokenizer.calm.encoder import TransformerBlock

        block = TransformerBlock(embed_dim=16, num_heads=4, dropout=0.5)
        block.train()
        x = torch.randn(2, 10, 16)
        torch.manual_seed(1)
        out1 = block(x)
        torch.manual_seed(2)
        out2 = block(x)
        assert not torch.allclose(out1, out2)

    def test_eval_mode_deterministic(self):
        from ntt.contrib.tokenizer.calm.encoder import TransformerBlock

        block = TransformerBlock(embed_dim=16, num_heads=4, dropout=0.5)
        block.eval()
        x = torch.randn(2, 10, 16)
        out1 = block(x)
        out2 = block(x)
        assert torch.allclose(out1, out2)


class TestEncoderDecoderDropoutThreading:
    def test_calm_encoder_threads_dropout(self):
        from ntt.contrib.tokenizer.calm.encoder import CalmEncoder

        enc = CalmEncoder(
            embed_dim=16,
            num_layers=2,
            num_heads=4,
            patch_length=5,
            causal=True,
            dropout=0.25,
        )
        for layer in enc.layers:
            assert layer.attn_drop.p == 0.25
            assert layer.ffn_drop.p == 0.25
            assert layer.attn.dropout == 0.25

    def test_calm_encoder_default_dropout_zero(self):
        from ntt.contrib.tokenizer.calm.encoder import CalmEncoder

        enc = CalmEncoder(embed_dim=16, num_layers=2, num_heads=4, patch_length=5, causal=True)
        for layer in enc.layers:
            assert layer.attn_drop.p == 0.0

    def test_calm_decoder_threads_dropout(self):
        from ntt.contrib.tokenizer.calm.decoder import CalmDecoder

        dec = CalmDecoder(embed_dim=16, num_layers=2, num_heads=4, patch_length=5, dropout=0.25)
        for layer in dec.layers:
            assert layer.attn_drop.p == 0.25
            assert layer.ffn_drop.p == 0.25
            assert layer.attn.dropout == 0.25


class TestCalmSpecDropoutField:
    def test_default_dropout_zero(self):
        spec = _small_spec()
        assert spec.dropout == 0.0

    def test_valid_dropout_accepted(self):
        spec = _small_spec(dropout=0.3)
        assert spec.dropout == 0.3

    def test_invalid_dropout_rejected(self):
        with pytest.raises((ValueError, Exception)):
            _small_spec(dropout=-0.1)
        with pytest.raises((ValueError, Exception)):
            _small_spec(dropout=1.0)

    def test_dropout_threaded_into_modules(self):
        spec = _small_spec(dropout=0.25, ar_weight=0.1)
        tok = CalmTokenizer(spec)

        for layer in tok.encoder.layers:
            assert layer.attn_drop.p == 0.25
        for layer in tok.decoder.layers:
            assert layer.attn_drop.p == 0.25
        assert tok.ar_regularizer is not None
        assert tok.ar_regularizer.attn_layer.dropout1.p == 0.25


class TestCalmSpecPopulationFields:
    def test_defaults(self):
        spec = _small_spec()
        assert spec.num_population_layers == 0
        assert spec.population_num_heads is None
        assert spec.neuron_dropout == 0.0

    def test_valid_values_accepted(self):
        spec = _small_spec(num_population_layers=2, population_num_heads=4, neuron_dropout=0.2)
        assert spec.num_population_layers == 2
        assert spec.population_num_heads == 4
        assert spec.neuron_dropout == 0.2

    def test_invalid_population_layers_rejected(self):
        with pytest.raises((ValueError, Exception)):
            _small_spec(num_population_layers=-1)

    def test_invalid_neuron_dropout_rejected(self):
        with pytest.raises((ValueError, Exception)):
            _small_spec(neuron_dropout=1.0)
        with pytest.raises((ValueError, Exception)):
            _small_spec(neuron_dropout=-0.01)

    def test_population_head_divisibility_enforced(self):
        # _small_spec default embed_dim=32; 32 % 5 != 0.
        with pytest.raises((ValueError, Exception)):
            _small_spec(num_population_layers=1, population_num_heads=5)


class TestPopulationBlock:
    def _make(self, embed_dim=16, num_heads=4, dropout=0.0):
        from ntt.contrib.tokenizer.calm.encoder import PopulationBlock

        return PopulationBlock(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)

    def test_output_shape(self):
        block = self._make()
        block.eval()
        B, T_d, N, C = 2, 3, 5, 16
        h = torch.randn(B, T_d, N, C)
        mask = torch.ones(B, N, N, dtype=torch.bool)
        out = block(h, attn_mask=mask)
        assert out.shape == (B, T_d, N, C)

    def test_permutation_equivariance_n_axis(self):
        """Permuting neurons on the input + matching rows/cols of the mask
        must yield the neuron-permuted output."""
        block = self._make()
        block.eval()
        torch.manual_seed(0)
        B, T_d, N, C = 1, 2, 4, 16
        h = torch.randn(B, T_d, N, C)
        mask = torch.ones(B, N, N, dtype=torch.bool)

        out = block(h, attn_mask=mask)

        perm = torch.tensor([2, 0, 3, 1])
        h_perm = h[:, :, perm, :]
        mask_perm = mask[:, perm][:, :, perm]
        out_perm = block(h_perm, attn_mask=mask_perm)

        assert torch.allclose(out_perm, out[:, :, perm, :], atol=1e-5)

    def test_key_mask_isolation(self):
        """A neuron marked False across all queries in the key dimension must
        have no influence on outputs for neurons whose query rows do not
        include it."""
        block = self._make()
        block.eval()
        torch.manual_seed(1)
        B, T_d, N, C = 1, 1, 4, 16
        h = torch.randn(B, T_d, N, C)

        mask = torch.ones(B, N, N, dtype=torch.bool)
        mask[:, :, 3] = False  # no query attends to key 3
        mask[:, :, :] |= torch.eye(N, dtype=torch.bool)[None]  # restore diagonals

        out_full = block(h, attn_mask=mask)

        h_perturbed = h.clone()
        h_perturbed[:, :, 3, :] = torch.randn(C)
        out_perturbed = block(h_perturbed, attn_mask=mask)

        # Outputs for queries 0,1,2 must be unchanged; query 3 output is
        # allowed to change (it attends to itself via the diagonal OR).
        assert torch.allclose(out_full[:, :, :3], out_perturbed[:, :, :3], atol=1e-5)

    def test_all_false_mask_with_diagonal_no_nan(self):
        """With every off-diagonal masked out (each neuron attends only to
        itself), the output must be finite (no NaN from empty attention)."""
        block = self._make()
        block.eval()
        B, T_d, N, C = 1, 1, 4, 16
        h = torch.randn(B, T_d, N, C)
        mask = torch.eye(N, dtype=torch.bool)[None].expand(B, N, N).clone()
        out = block(h, attn_mask=mask)
        assert torch.isfinite(out).all()

    def test_dropout_train_vs_eval(self):
        block = self._make(dropout=0.5)
        B, T_d, N, C = 1, 2, 4, 16
        h = torch.randn(B, T_d, N, C)
        mask = torch.ones(B, N, N, dtype=torch.bool)

        block.train()
        torch.manual_seed(1)
        out1 = block(h, attn_mask=mask)
        torch.manual_seed(2)
        out2 = block(h, attn_mask=mask)
        assert not torch.allclose(out1, out2)

        block.eval()
        out_a = block(h, attn_mask=mask)
        out_b = block(h, attn_mask=mask)
        assert torch.allclose(out_a, out_b)


class TestCalmEncoderPopulation:
    def _rand_input(self, B=2, N=4, T=20, seed=0):
        torch.manual_seed(seed)
        # CalmEncoder expects per-neuron traces shaped [B*N, T, 1].
        return torch.randn(B * N, T, 1)

    def test_default_no_population_layers(self):
        from ntt.contrib.tokenizer.calm.encoder import CalmEncoder

        enc = CalmEncoder(
            embed_dim=16,
            num_layers=1,
            num_heads=2,
            patch_length=5,
            causal=True,
        )
        assert hasattr(enc, "population_layers")
        assert len(enc.population_layers) == 0

    def test_identity_when_num_population_layers_zero(self):
        """With num_population_layers=0, forward matches the same-seeded build
        that did not request population layers."""
        from ntt.contrib.tokenizer.calm.encoder import CalmEncoder

        torch.manual_seed(123)
        enc_a = CalmEncoder(
            embed_dim=16,
            num_layers=2,
            num_heads=2,
            patch_length=5,
            causal=True,
        )
        torch.manual_seed(123)
        enc_b = CalmEncoder(
            embed_dim=16,
            num_layers=2,
            num_heads=2,
            patch_length=5,
            causal=True,
            num_population_layers=0,
        )
        enc_a.eval()
        enc_b.eval()

        x = self._rand_input()
        out_a = enc_a(x)
        out_b = enc_b(x)
        assert torch.allclose(out_a, out_b, atol=1e-6)

    def test_shape_with_population_layers(self):
        from ntt.contrib.tokenizer.calm.encoder import CalmEncoder

        B, N, T = 2, 4, 20
        enc = CalmEncoder(
            embed_dim=16,
            num_layers=1,
            num_heads=2,
            patch_length=5,
            causal=True,
            num_population_layers=2,
            population_num_heads=4,
        )
        enc.eval()
        x = self._rand_input(B=B, N=N, T=T)
        mask = torch.ones(B, N, N, dtype=torch.bool)
        out = enc(x, population_attn_mask=mask, B=B, N=N)
        T_d = T // 5
        assert out.shape == (B * N, T_d, 16)

    def test_requires_B_and_N_when_population_active(self):
        from ntt.contrib.tokenizer.calm.encoder import CalmEncoder

        enc = CalmEncoder(
            embed_dim=16,
            num_layers=1,
            num_heads=2,
            patch_length=5,
            causal=True,
            num_population_layers=1,
        )
        enc.eval()
        x = self._rand_input(B=2, N=4)
        with pytest.raises(ValueError):
            enc(x)  # missing B, N


def _rates_input_with_padded_neurons(B, T, N, valid_neurons):
    """Like _make_rates_input but zeros out pad_and_gap_mask rows for
    neurons marked invalid in ``valid_neurons: [B, N] bool``."""
    x = _make_rates_input(B, T, N)
    keep = valid_neurons[:, None, :].expand(B, T, N).to(torch.int32)
    # NamedTuples are immutable, so rebuild with a patched mask.
    return x._replace(pad_and_gap_mask=x.pad_and_gap_mask * keep)


class TestCalmPopulationKeyMask:
    def test_returns_none_when_no_population_layers(self):
        tok = CalmTokenizer(_small_spec(num_population_layers=0))
        x = _make_rates_input(2, 20, 8)
        assert tok._build_population_key_mask(x) is None

    def test_shape_and_dtype(self):
        tok = CalmTokenizer(_small_spec(num_population_layers=1))
        tok.eval()
        B, N = 2, 8
        x = _make_rates_input(B, 20, N)
        mask = tok._build_population_key_mask(x)
        assert mask is not None
        assert mask.shape == (B, N, N)
        assert mask.dtype == torch.bool

    def test_padded_neurons_masked_in_eval(self):
        tok = CalmTokenizer(_small_spec(num_population_layers=1))
        tok.eval()
        B, N = 1, 4
        valid = torch.tensor([[True, True, False, True]])
        x = _rates_input_with_padded_neurons(B, 20, N, valid)
        mask = tok._build_population_key_mask(x)
        # Key column 2 is padded: all False except the diagonal at q=2.
        assert (mask[:, :, 2] == torch.tensor([[False, False, True, False]])).all()

    def test_diagonal_always_true(self):
        tok = CalmTokenizer(_small_spec(num_population_layers=1, neuron_dropout=0.99))
        tok.train()
        B, N = 4, 6
        x = _make_rates_input(B, 20, N)
        for _ in range(5):
            mask = tok._build_population_key_mask(x)
            diag = mask[:, torch.arange(N), torch.arange(N)]
            assert diag.all(), "diagonal must always be True"

    def test_neuron_dropout_inactive_in_eval(self):
        tok = CalmTokenizer(_small_spec(num_population_layers=1, neuron_dropout=0.99))
        tok.eval()
        x = _make_rates_input(2, 20, 8)
        mask_a = tok._build_population_key_mask(x)
        mask_b = tok._build_population_key_mask(x)
        assert torch.equal(mask_a, mask_b)

    def test_neuron_dropout_active_in_train(self):
        tok = CalmTokenizer(_small_spec(num_population_layers=1, neuron_dropout=0.5))
        tok.train()
        x = _make_rates_input(4, 20, 8)
        torch.manual_seed(1)
        mask_a = tok._build_population_key_mask(x)
        torch.manual_seed(2)
        mask_b = tok._build_population_key_mask(x)
        assert not torch.equal(mask_a, mask_b)


class TestCalmTokenizerIntegration:
    def test_identity_under_defaults(self):
        """With dropout=0, num_population_layers=0, neuron_dropout=0, the
        forward output must match a CalmTokenizer built from a spec that
        does not specify any of the new fields."""
        torch.manual_seed(42)
        tok_a = CalmTokenizer(_small_spec())
        torch.manual_seed(42)
        tok_b = CalmTokenizer(
            _small_spec(
                dropout=0.0,
                num_population_layers=0,
                neuron_dropout=0.0,
            )
        )
        tok_a.eval()
        tok_b.eval()
        x = _make_rates_input(2, 20, 8)
        out_a = tok_a(x)
        out_b = tok_b(x)
        assert torch.allclose(out_a.reconstruction, out_b.reconstruction, atol=1e-6)
        assert torch.equal(out_a.tokens, out_b.tokens)

    def test_forward_with_population_layers_shape(self):
        torch.manual_seed(0)
        tok = CalmTokenizer(_small_spec(num_population_layers=2))
        tok.eval()
        B, T, N = 2, 20, 8
        x = _make_rates_input(B, T, N)
        out = tok(x)
        assert out.reconstruction.shape == (B, T, N)
        T_d = T // tok.spec.patch_length
        assert out.tokens.shape == (B, T_d, N, 1)

    def test_encode_with_population_layers(self):
        torch.manual_seed(0)
        tok = CalmTokenizer(_small_spec(num_population_layers=1))
        tok.eval()
        B, T, N = 2, 20, 8
        x = _make_rates_input(B, T, N)
        tokens = tok.encode(x)
        T_d = T // tok.spec.patch_length
        assert tokens.shape == (B, T_d, N, 1)

    def test_n_equals_1_runs(self):
        """After training (conceptually) with neuron dropout, the tokenizer
        must at minimum *run* on single-neuron input without error."""
        torch.manual_seed(0)
        tok = CalmTokenizer(_small_spec(num_population_layers=1, n_neurons=1))
        tok.eval()
        x = _make_rates_input(2, 20, 1)
        out = tok(x)
        assert out.reconstruction.shape == (2, 20, 1)

    def test_neuron_dropout_does_not_break_training_forward(self):
        tok = CalmTokenizer(
            _small_spec(
                num_population_layers=1,
                neuron_dropout=0.5,
                dropout=0.1,
            )
        )
        tok.train()
        x = _make_rates_input(3, 20, 8)
        out = tok(x)
        assert torch.isfinite(out.reconstruction).all()
        assert torch.isfinite(out.aux_loss).all()
