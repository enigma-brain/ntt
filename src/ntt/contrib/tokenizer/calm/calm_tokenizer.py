"""CALM Tokenizer: per-neuron tokenization with shared vocabulary.

Architecture:
    [B, T, N] -> per-neuron [B*N, T, 1]
    -> Encoder -> [B*N, T_d, C]
    -> Bottleneck (VQ / FSQ / BSQ) -> quantized [B*N, T_d, C], indices [B*N, T_d]
    -> AR Regularizer -> ar_loss
    -> Decoder -> [B*N, T, 1]
    -> reshape back to [B, T, N]
"""

from __future__ import annotations

from typing import ClassVar, Literal

import pydantic as pdt
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from ntt.contrib.tokenizer.wanneural.bottleneck import (
    BSQBottleneck,
    FSQuantizer,
)
from ntt.core.components.dist import Dist
from ntt.core.components.types import (
    BinnedSpikesInput,
    DataType,
    LatentType,
    RatesInput,
    ReconstructionOutput,
    TokenizerOutput,
)
from ntt.core.model.tokenizer.ar_regularizer import ARRegularizer
from ntt.core.model.tokenizer.base import (
    BaseTokenizer,
    TokenizerSpec,
    register_tokenizer,
)

from .decoder import CalmDecoder
from .encoder import CalmEncoder
from .vq import VQBottleneck


class CalmTokenizerSpec(TokenizerSpec, frozen=True):
    """Configuration for the CALM tokenizer.

    Per-neuron tokenization with shared vocabulary, supporting VQ, FSQ, and BSQ
    bottleneck types.
    """

    KIND: Literal["calm"] = "calm"

    # Data parameters
    n_neurons: int = 200
    window_length_seconds: float | Dist = 2.0
    bin_size_ms: float = 10.0

    # Patchification
    patch_length: int = 10  # token coverage = patch_length * bin_size_ms

    # Transformer
    embed_dim: int = 512
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    num_heads: int = 4
    causal: bool = True
    decoder_causal: bool = False

    # Bottleneck
    bottleneck_type: Literal["vq", "fsq", "bsq"] = "vq"
    codebook_size: int = 128
    ema_decay: float = 0.99
    commitment_weight: float = 0.5
    entropy_weight: float = 0.5
    orthogonality_weight: float = 0.1
    entropy_temperature: float = 1.0
    dead_code_threshold: float = 1.0
    q_levels: tuple[int, ...] = (8, 5, 5, 5)
    bsq_bits: int = 10
    bsq_beta: float = 1.0

    # AR regularization
    ar_weight: float = 0.5
    ar_horizon: int = 4

    # Output
    output_activation: Literal["none", "softplus", "sigmoid"] = "none"
    reconstruct_sample: Literal["none", "poisson", "round", "bernoulli"] = "none"

    # Regularization (global dropout for all TransformerBlocks and ARRegularizer)
    dropout: float = 0.0

    # Population context (staged after temporal encoder, before bottleneck)
    num_population_layers: int = 0
    population_num_heads: int | None = None
    neuron_dropout: float = 0.0

    # Capabilities
    latent_type: LatentType = LatentType.DISCRETE
    compress_n: bool = False
    compress_t: bool = True
    variable_n: bool = True
    # calm handles variable input length, provided the time dimension is a
    # multiple of patch_length (the patchification reshape in encode requires
    # T % patch_length == 0). Eval feeds calm the eval window directly (no
    # chunking) and enforces that multiple via assert_tokenizer_accepts_eval_window
    # (window_contract.py).
    variable_t: bool = True
    variable_dt: bool = False
    input_type: DataType = DataType.RATES
    output_type: DataType = DataType.RATES
    has_time_dim: bool = True
    has_unit_dim: bool = True
    has_latent_dim: bool = False

    @pdt.model_validator(mode="after")
    def _set_default_token_bin_size(self) -> CalmTokenizerSpec:
        """Compute token_bin_size_ms from patch_length and bin_size_ms.

        Overrides the parent validator of the same name so that the computed
        value is set before the parent would raise on the ambiguous
        compress_t=True + has_time_dim=True case.
        """
        if self.token_bin_size_ms is None:
            object.__setattr__(self, "token_bin_size_ms", self.patch_length * self.bin_size_ms)
        return self

    @pdt.model_validator(mode="after")
    def _validate_config(self) -> CalmTokenizerSpec:
        """Validate cross-field constraints."""
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})"
            )
        # Validate time bins divisible by patch_length
        if isinstance(self.window_length_seconds, Dist):
            min_bins = int(float(self.window_length_seconds.min) * 1000.0 / self.bin_size_ms)
            max_bins = int(float(self.window_length_seconds.max) * 1000.0 / self.bin_size_ms)
            if min_bins % self.patch_length != 0:
                raise ValueError(
                    f"Minimum number of time bins ({min_bins}) must be divisible by "
                    f"patch_length ({self.patch_length}). Got min_bins from "
                    f"Dist.min={self.window_length_seconds.min}, "
                    f"bin_size_ms={self.bin_size_ms}."
                )
            if max_bins % self.patch_length != 0:
                raise ValueError(
                    f"Maximum number of time bins ({max_bins}) must be divisible by "
                    f"patch_length ({self.patch_length}). Got max_bins from "
                    f"Dist.max={self.window_length_seconds.max}, "
                    f"bin_size_ms={self.bin_size_ms}."
                )
        else:
            n_bins = int(self.window_length_seconds * 1000.0 / self.bin_size_ms)
            if n_bins % self.patch_length != 0:
                raise ValueError(
                    f"Number of time bins ({n_bins}) must be divisible by "
                    f"patch_length ({self.patch_length}). Got {n_bins} bins from "
                    f"window_length_seconds={self.window_length_seconds}, "
                    f"bin_size_ms={self.bin_size_ms}."
                )
        if self.entropy_temperature <= 0:
            raise ValueError(
                f"entropy_temperature must be positive, got {self.entropy_temperature}"
            )
        if self.reconstruct_sample == "bernoulli" and self.bin_size_ms > 2.0:
            raise ValueError(
                f"reconstruct_sample='bernoulli' assumes at most 1 spike per bin, "
                f"but bin_size_ms={self.bin_size_ms} is too coarse. "
                f"Use 'poisson' for bins > 2ms."
            )
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0), got {self.dropout}")
        if self.num_population_layers < 0:
            raise ValueError(
                f"num_population_layers must be >= 0, got {self.num_population_layers}"
            )
        if not (0.0 <= self.neuron_dropout < 1.0):
            raise ValueError(f"neuron_dropout must be in [0.0, 1.0), got {self.neuron_dropout}")
        if self.population_num_heads is not None:
            if self.embed_dim % self.population_num_heads != 0:
                raise ValueError(
                    f"embed_dim ({self.embed_dim}) must be divisible by "
                    f"population_num_heads ({self.population_num_heads})"
                )
        return self

    def get_vocab_size(self) -> int:
        """Return the vocabulary size based on bottleneck type."""
        match self.bottleneck_type:
            case "vq":
                return self.codebook_size
            case "fsq":
                result = 1
                for level in self.q_levels:
                    result *= level
                return result
            case "bsq":
                return 2**self.bsq_bits
            case _:
                raise ValueError(f"Unknown bottleneck_type: {self.bottleneck_type}")

    def get_codes_per_token(self) -> int:
        """One code per latent position: ``encode()`` returns a single index per token
        (``[B, T_d, N, 1]``) for every bottleneck type (VQ, FSQ, BSQ)."""
        return 1


@register_tokenizer
class CalmTokenizer(BaseTokenizer):
    """CALM tokenizer: per-neuron tokenization with shared vocabulary.

    Processes each neuron independently through a shared encoder, bottleneck,
    and decoder. Supports VQ, FSQ, and BSQ quantization.
    """

    spec_class: ClassVar[type[CalmTokenizerSpec]] = CalmTokenizerSpec
    spec: CalmTokenizerSpec

    def __init__(self, spec: CalmTokenizerSpec) -> None:
        super().__init__(spec)

        # Encoder
        self.encoder = CalmEncoder(
            embed_dim=spec.embed_dim,
            num_layers=spec.num_encoder_layers,
            num_heads=spec.num_heads,
            patch_length=spec.patch_length,
            causal=spec.causal,
            dropout=spec.dropout,
            num_population_layers=spec.num_population_layers,
            population_num_heads=spec.population_num_heads,
        )

        # Bottleneck
        if spec.bottleneck_type == "vq":
            self.vq = VQBottleneck(
                codebook_size=spec.codebook_size,
                embed_dim=spec.embed_dim,
                ema_decay=spec.ema_decay,
                dead_code_threshold=spec.dead_code_threshold,
            )
        elif spec.bottleneck_type == "fsq":
            self.fsq = FSQuantizer(
                levels=list(spec.q_levels),
                latent_dim=spec.embed_dim,
            )
        elif spec.bottleneck_type == "bsq":
            self.bsq = BSQBottleneck(
                z_dim=spec.embed_dim,
                codebook_dim=spec.bsq_bits,
                zeta=spec.bsq_beta,
            )
        else:
            raise ValueError(f"Unknown bottleneck type: {spec.bottleneck_type}")

        # AR Regularizer
        if spec.ar_weight > 0:
            self.ar_regularizer = ARRegularizer(
                embed_dim=spec.embed_dim,
                vocab_size=spec.get_vocab_size(),
                num_heads=spec.num_heads,
                horizon=spec.ar_horizon,
                dropout=spec.dropout,
            )
        else:
            self.ar_regularizer = None

        # Decoder
        self.decoder = CalmDecoder(
            embed_dim=spec.embed_dim,
            num_layers=spec.num_decoder_layers,
            num_heads=spec.num_heads,
            patch_length=spec.patch_length,
            dropout=spec.dropout,
            causal=spec.decoder_causal,
        )

    def _build_time_key_padding_mask(
        self, x: BinnedSpikesInput | RatesInput, B: int, N: int, T: int
    ) -> Tensor:
        """Build a (B*N, T_d) bool key-padding mask for temporal attention.

        A patch at time index ``p`` is marked valid if **any** bin in
        ``[p * patch_length, (p + 1) * patch_length)`` is valid in
        ``pad_and_gap_mask`` (lenient: partially-valid patches are kept).
        The same per-sample mask is replicated across all N neurons since
        padding/gaps are time-axis properties shared by the whole population.
        """
        patch_length = self.spec.patch_length
        # Reduce (B, T, N) -> (B, T) by OR across neurons, then patchify.
        time_mask = x.pad_and_gap_mask.bool().any(dim=-1)  # (B, T)
        T_d = T // patch_length
        patch_mask = time_mask.reshape(B, T_d, patch_length).any(dim=-1)  # (B, T_d)
        # Broadcast to per-neuron: (B, N, T_d) -> (B*N, T_d).
        return patch_mask.unsqueeze(1).expand(B, N, T_d).reshape(B * N, T_d)

    def _build_population_key_mask(self, x: BinnedSpikesInput | RatesInput) -> Tensor | None:
        """Build the [B, N, N] attention mask for the population stack.

        Returns ``None`` when ``num_population_layers == 0`` (short-circuits
        unused work in the forward path).

        Combines three signals:
            1. Padding — a neuron with all-zero ``pad_and_gap_mask`` over time
               is masked as a key for every query.
            2. Neuron dropout (train-only) — per-sample, per-neuron Bernoulli
               with ``p_drop = spec.neuron_dropout`` mutes a neuron as a key.
            3. Self-inclusion — the diagonal is always True, so every neuron
               can attend to itself (prevents fully-masked attention rows).
        """
        if self.spec.num_population_layers == 0:
            return None

        pad_mask = x.pad_and_gap_mask  # (B, T, N) int
        B, _T, N = pad_mask.shape
        device = pad_mask.device

        valid_neurons = pad_mask.bool().any(dim=1)  # (B, N)

        if self.training and self.spec.neuron_dropout > 0.0:
            keep_prob = 1.0 - self.spec.neuron_dropout
            keep = torch.bernoulli(
                torch.full((B, N), keep_prob, device=device, dtype=torch.float32)
            ).bool()
            key_mask = valid_neurons & keep
        else:
            key_mask = valid_neurons

        # Broadcast to [B, N, N] (same key mask for every query row), then OR
        # the diagonal so the self-attention slot is never masked.
        attn_mask = key_mask[:, None, :].expand(B, N, N).clone()
        eye = torch.eye(N, dtype=torch.bool, device=device)
        attn_mask = attn_mask | eye[None, :, :]
        return attn_mask

    def _extract_data(self, x: BinnedSpikesInput | RatesInput) -> Tensor:
        """Extract [B, T, N] data tensor from input."""
        if isinstance(x, RatesInput):
            data = x.rate_matrix.float()
        elif isinstance(x, BinnedSpikesInput):
            data = x.spike_matrix.float()
        else:
            raise TypeError(f"Expected BinnedSpikesInput or RatesInput, got {type(x).__name__}")
        data = data * x.pad_and_gap_mask.float()
        return data

    def _apply_output_activation(self, x: Tensor) -> Tensor:
        """Apply output activation function."""
        match self.spec.output_activation:
            case "none":
                return x
            case "softplus":
                return F.softplus(x)
            case "sigmoid":
                return torch.sigmoid(x)
            case _:
                raise ValueError(f"Unknown output_activation: {self.spec.output_activation}")

    def _bottleneck_forward(
        self,
        h: Tensor,
        B: int,
        N: int,
        valid_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run the bottleneck on encoded features.

        Args:
            h: Encoded features [B*N, T_d, C].
            B: Batch size.
            N: Number of neurons.
            valid_mask: Optional (B*N, T_d) bool mask (True = real token).
                Currently honored by the ``"vq"`` bottleneck to mask
                commitment/entropy/EMA updates; FSQ/BSQ ignore it.

        Returns:
            (quantized, indices, bottleneck_loss):
                quantized: [B*N, T_d, C]
                indices: [B*N, T_d]
                bottleneck_loss: scalar tensor
        """
        spec = self.spec
        BN, T_d, C = h.shape

        if self.spec.bottleneck_type == "vq":
            vq_out = self.vq(
                h,
                temperature=spec.entropy_temperature,
                valid_mask=valid_mask,
            )
            quantized = vq_out.quantized  # [B*N, T_d, C]
            indices = vq_out.indices  # [B*N, T_d]
            losses = vq_out.losses
            bottleneck_loss = (
                spec.commitment_weight * losses["commitment"]
                + spec.entropy_weight * losses["entropy"]
                + spec.orthogonality_weight * losses["orthogonality"]
            )
            return quantized, indices, bottleneck_loss

        elif self.spec.bottleneck_type == "fsq":
            # FSQuantizer operates on 2D (batch, latent_dim)
            h_flat = rearrange(h, "bn td c -> (bn td) c")
            idx, codes = self.fsq(h_flat, return_codes=True)
            quantized = rearrange(codes, "(bn td) c -> bn td c", bn=BN, td=T_d)
            indices = rearrange(idx, "(bn td) -> bn td", bn=BN, td=T_d).long()
            bottleneck_loss = torch.tensor(0.0, device=h.device, dtype=h.dtype)
            return quantized, indices, bottleneck_loss

        elif self.spec.bottleneck_type == "bsq":
            # BSQBottleneck expects 4D (B, C, T, N)
            h_4d = rearrange(h, "(b n) td c -> b c td n", b=B, n=N)
            z_quantized, entropy_loss = self.bsq.encode(h_4d)
            # Get indices
            indices_flat = self.bsq.get_indices(h_4d)  # [B, T_d * N]
            # Reshape indices to [B*N, T_d]
            indices = rearrange(indices_flat.reshape(B, T_d, N), "b td n -> (b n) td")
            # Reshape quantized back to [B*N, T_d, C]
            quantized = rearrange(z_quantized, "b c td n -> (b n) td c")
            bottleneck_loss = spec.bsq_beta * entropy_loss
            return quantized, indices, bottleneck_loss

        raise ValueError(f"Unknown bottleneck type: {self.spec.bottleneck_type}")

    def encode(self, x: BinnedSpikesInput | RatesInput) -> Tensor:
        """Encode neural data to 4D token tensor [B, T_d, N, 1].

        Args:
            x: Input neural data.

        Returns:
            Token indices [B, T_d, N, 1].
        """
        data = self._extract_data(x)  # [B, T, N]
        B, T, N = data.shape

        # Per-neuron processing
        h = rearrange(data, "b t n -> (b n) t 1")  # [B*N, T, 1]
        population_mask = self._build_population_key_mask(x)
        time_kp_mask = self._build_time_key_padding_mask(x, B, N, T)
        h = self.encoder(
            h,
            population_attn_mask=population_mask,
            time_key_padding_mask=time_kp_mask,
            B=B,
            N=N,
        )  # [B*N, T_d, C]

        # Bottleneck
        _quantized, indices, _loss = self._bottleneck_forward(h, B, N, valid_mask=time_kp_mask)

        # Reshape indices to [B, T_d, N, 1]
        tokens = rearrange(indices, "(b n) td -> b td n 1", b=B, n=N)
        return tokens

    def _indices_to_codes(self, tokens: Tensor) -> Tensor:
        """Look up continuous codebook vectors from token indices.

        Args:
            tokens: Token indices [B, T_d, N, 1].

        Returns:
            Quantized vectors [B*N, T_d, C].
        """
        B, T_d, N, _D = tokens.shape
        indices = tokens.squeeze(-1)  # [B, T_d, N]
        indices_flat = rearrange(indices, "b td n -> (b n) td")

        if self.spec.bottleneck_type == "vq":
            return self.vq.indices_to_codes(indices_flat)  # [B*N, T_d, C]
        elif self.spec.bottleneck_type == "fsq":
            BN, Td = indices_flat.shape
            idx_flat = rearrange(indices_flat, "bn td -> (bn td)")
            codes = self.fsq.indices_to_codes(idx_flat, project_out=True)
            return rearrange(codes, "(bn td) c -> bn td c", bn=BN, td=Td)
        elif self.spec.bottleneck_type == "bsq":
            indices_bsq = rearrange(indices, "b td n -> b (td n)")
            codes_4d = self.bsq.indices_to_codes(indices_bsq, T=T_d)  # [B, C, T_d, N]
            return rearrange(codes_4d, "b c td n -> (b n) td c")
        else:
            raise ValueError(f"Unknown bottleneck type: {self.spec.bottleneck_type}")

    def tokens_to_latents(self, tokens: Tensor) -> Tensor:
        """Convert discrete token indices to continuous codebook vectors.

        Args:
            tokens: Token indices [B, T_d, N, 1] from ``encode()``.

        Returns:
            Continuous latent vectors [B, T_d, N, C].
        """
        B, T_d, N, _D = tokens.shape
        quantized = self._indices_to_codes(tokens)  # [B*N, T_d, C]
        return rearrange(quantized, "(b n) td c -> b td n c", b=B, n=N)

    def decode(self, tokens: Tensor) -> Tensor:
        """Decode token indices back to neural data [B, T, N].

        Args:
            tokens: Token indices [B, T_d, N, 1], or continuous latents [B, T_d, N, C].

        Returns:
            Reconstructed data [B, T, N].
        """
        B, T_d, N, _D = tokens.shape
        if tokens.is_floating_point():
            # Float latents
            if _D != self.spec.embed_dim:
                raise ValueError(
                    f"Expected continuous latents with last dim {self.spec.embed_dim}, got {_D}"
                )
            tokens = rearrange(tokens, "b td n c -> (b n) td c")
        else:
            # We have token indices
            if _D != 1:
                raise ValueError(f"Expected discrete token indices with last dim 1, got {_D}")
            tokens = self._indices_to_codes(tokens)  # [B*N, T_d, C]
        # Decode
        recon = self.decoder(tokens)  # [B*N, T, 1]
        recon = rearrange(recon, "(b n) t 1 -> b t n", b=B, n=N)
        recon = self._apply_output_activation(recon)
        return recon

    def reconstruct(self, tokens: Tensor) -> ReconstructionOutput:
        """Decode tokens and optionally sample spikes.

        Behaviour is controlled by ``spec.reconstruct_sample``:
            - ``"none"``:  return raw decoder output (rates).
            - ``"poisson"``: Poisson-sample from decoded rates → integer counts.
            - ``"round"``:  deterministic rounding of rates → integer counts.
            - ``"bernoulli"``: Bernoulli-sample treating rate × dt as probability
              (appropriate for fine bins where P(>1 spike) ≈ 0).
        """
        rates = self.decode(tokens)  # [B, T, N]

        mode = self.spec.reconstruct_sample
        if mode == "poisson":
            reconstruction = torch.poisson(rates.clamp(min=0))
            data_type = DataType.SPIKES
        elif mode == "round":
            reconstruction = rates.round().clamp(min=0)
            data_type = DataType.SPIKES
        elif mode == "bernoulli":
            # rates are in spikes/bin after softplus; interpret as Bernoulli p
            probs = rates.clamp(0.0, 1.0)
            reconstruction = torch.bernoulli(probs)
            data_type = DataType.SPIKES
        else:  # "none"
            reconstruction = rates
            data_type = self.spec.output_type

        return ReconstructionOutput(
            reconstruction=reconstruction,
            data_type=data_type,
            bin_size_seconds=self.spec.bin_size_ms / 1000.0,
        )

    def forward(self, x: BinnedSpikesInput | RatesInput) -> TokenizerOutput:
        """Full forward pass: encode, bottleneck, AR regularize, decode.

        Args:
            x: Input neural data [B, T, N].

        Returns:
            TokenizerOutput with tokens [B, T_d, N, 1], reconstruction [B, T, N],
            and scalar aux_loss.
        """
        data = self._extract_data(x)  # [B, T, N]
        B, T, N = data.shape

        # 1. Reshape to per-neuron
        h = rearrange(data, "b t n -> (b n) t 1")  # [B*N, T, 1]

        # 2. Encode (with optional cross-neuron population context)
        population_mask = self._build_population_key_mask(x)
        time_kp_mask = self._build_time_key_padding_mask(x, B, N, T)
        h = self.encoder(
            h,
            population_attn_mask=population_mask,
            time_key_padding_mask=time_kp_mask,
            B=B,
            N=N,
        )  # [B*N, T_d, C]

        # 3. Bottleneck
        quantized, indices, bottleneck_loss = self._bottleneck_forward(
            h, B, N, valid_mask=time_kp_mask
        )

        # 4. AR Regularizer
        ar_loss = torch.tensor(0.0, device=data.device, dtype=data.dtype)
        if self.ar_regularizer is not None and self.spec.ar_weight > 0:
            ar_loss = self.spec.ar_weight * self.ar_regularizer(
                quantized, indices, valid_mask=time_kp_mask
            )

        # 5. Decode (pass the same key-padding mask so the decoder's attention
        # — especially when bidirectional — ignores trailing-padded tokens).
        recon = self.decoder(quantized, time_key_padding_mask=time_kp_mask)  # [B*N, T, 1]

        # 6. Reshape back to [B, T, N]
        recon = rearrange(recon, "(b n) t 1 -> b t n", b=B, n=N)
        recon = self._apply_output_activation(recon)

        # 7. Format tokens as [B, T_d, N, 1]
        tokens = rearrange(indices, "(b n) td -> b td n 1", b=B, n=N)

        # 8. Total auxiliary loss
        aux_loss = bottleneck_loss + ar_loss

        return TokenizerOutput(
            tokens=tokens,
            reconstruction=recon,
            aux_loss=aux_loss,
            latents=rearrange(quantized, "(b n) td c -> b td n c", b=B, n=N),
            codes=tokens,
        )
