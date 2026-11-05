"""2D Cosmos-style tokenizer for neural spike data.

Treats ``[B, T, N]`` as a single-channel image of shape ``(H=N, W=T)`` with
causality along the time axis only. The pipeline is:

    [B, T, N] (spike_matrix)
        → unsqueeze channel + transpose → [B, 1, N, T]
        → Patcher2D (causal-in-time Haar DWT)
        → Encoder2D (causal-in-time ResNet + strided convs)
        → bottleneck (AE or FSQ, 2D)
        → Decoder2D
        → UnPatcher2D
        → squeeze + transpose → [B, T, N]

Unlike ``CosmosTokenizer1D``, the latent preserves a neuron-like dimension
(``H_latent = N / neuron_compression``), which the NTT 4D output encodes as
the unit axis. Neuron and temporal compression are independent — set
``neuron_compression=1`` for time-only compression, or ``temporal_compression=1``
for population (neuron-only) compression.

Adapted from NVIDIA Cosmos-Tokenizer, used under the Apache-2.0 license:
https://github.com/NVIDIA/Cosmos-Tokenizer
(see ``LICENSES/cosmos-tokenizer-Apache-2.0.txt``).
"""

from __future__ import annotations

from typing import Literal

import pydantic as pdt
import torch
from einops import rearrange
from torch import Tensor

from ntt.contrib.tokenizer.cosmos.bottleneck import (
    FSQBottleneck2d,
    VQPTBottleneck2d,
    build_bottleneck,
)
from ntt.contrib.tokenizer.cosmos.layers2d import Decoder2D, Encoder2D
from ntt.contrib.tokenizer.cosmos.wavelets import SUPPORTED_WAVELETS
from ntt.contrib.tokenizer.vqpt_codes import vqpt_codes_per_token, vqpt_vocab_size
from ntt.core.components.types import (
    BinnedInputType,
    DataType,
    LatentType,
    ReconstructionOutput,
    TokenizerOutput,
    get_data_matrix,
)
from ntt.core.model.tokenizer.base import (
    BaseTokenizer,
    TokenizerSpec,
    register_tokenizer,
)

# Floor for the rate prediction: keeps Poisson NLL's log(rate) finite even if
# the decoder drifts into the softplus underflow regime. Prevents NaN cascades
# if the decoder underflows.
_RATE_FLOOR = 1e-8


class CosmosTokenizer2DSpec(TokenizerSpec, frozen=True):  # pyright: ignore[reportCallIssue,reportGeneralTypeIssues]
    """Spec for the 2D causal Cosmos tokenizer (neuron × time).

    Compression is **independently configurable** on the neuron (H) and time
    (W) axes. Each axis's total downsample = ``patch_*`` (wavelet) × ``2^k``
    (strided Conv2d stages). Set ``neuron_compression=1`` to leave neurons
    untouched, or ``temporal_compression=1`` to leave time untouched.

    Args:
        bottleneck_type: ``"ae"`` (continuous) or ``"fsq"`` (discrete).
        channels: Base encoder/decoder channel width.
        channels_mult: Per-stage channel multipliers. Length must be
            ``>= max(log2(neuron_compression / patch_h), log2(temporal_compression / patch_w))``.
        num_res_blocks: ResBlocks per stage.
        dropout: Dropout inside ResBlocks.
        z_channels: Channels at the bottleneck boundary.
        neuron_compression: Total H downsample ratio (power of two).
        temporal_compression: Total W downsample ratio (power of two).
        patch_h: Wavelet downsample ratio along H (power of two; 1 = no DWT on H).
            Spike rasters across neurons aren't smooth — usually 1.
        patch_w: Wavelet downsample ratio along W (power of two; 1 = no DWT on W).
        patch_method: Wavelet family name.
        latent_channels: AE only — number of latent channels.
        fsq_levels: FSQ only — per-dimension quantization levels.
    """

    KIND: Literal["cosmos_2d"] = "cosmos_2d"

    # --- Data ---
    n_neurons: int = 32
    window_length_seconds: float = 1.0
    bin_size_ms: float = 1.0

    # --- Architecture ---
    bottleneck_type: Literal["ae", "fsq", "vae", "vqpt"] = "ae"
    channels: int = 64
    channels_mult: tuple[int, ...] = (1, 2, 2)
    num_res_blocks: int = 2
    dropout: float = 0.0
    z_channels: int = 64
    neuron_compression: int = 16
    temporal_compression: int = 16
    patch_h: int = 1
    patch_w: int = 4
    patch_method: str = "haar"

    # --- Continuous (AE / VAE) ---
    latent_channels: int = 16

    # --- VAE only ---
    vae_beta: float = 1.0

    # --- Discrete (FSQ) ---
    fsq_levels: tuple[int, ...] = (8, 5, 5, 5)

    # --- VQPT (lucidrains' vector_quantize_pytorch wrapper) ---
    # vqpt_type ∈ {"vq", "rvq" / "residual_vq", "lfq", "fsq", ...}; see
    # VQPTBottleneck2d._REGISTRY in cosmos/bottleneck.py for the full list.
    # vqpt_kwargs is passed to the wrapped quantizer; ``dim`` is auto-supplied.
    vqpt_type: str = "rvq"
    vqpt_kwargs: dict = pdt.Field(default_factory=dict)
    vqpt_loss_weight: float = 1.0

    # --- Capability metadata ---
    latent_type: LatentType = LatentType.CONTINUOUS
    compress_n: bool = True
    compress_t: bool = True
    variable_n: bool = False
    variable_t: bool = False
    variable_dt: bool = False
    input_type: DataType = DataType.SPIKES
    output_type: DataType = DataType.RATES

    # 2D latent keeps a neuron-like axis.
    has_time_dim: bool = True
    has_unit_dim: bool = True
    has_latent_dim: bool = True

    @pdt.model_validator(mode="before")
    @classmethod
    def _derive_latent_type(cls, data: object) -> object:
        if isinstance(data, dict):
            bt = data.get("bottleneck_type", "ae")
            # Discrete: FSQ (lattice) and VQPT (any lucidrains quantizer).
            # Continuous: AE (pass-through) and VAE (Gaussian reparameterization).
            expected = "discrete" if bt in ("fsq", "vqpt") else "continuous"
            provided = data.get("latent_type", expected)
            if isinstance(provided, LatentType):
                provided = provided.value
            if provided != expected:
                raise ValueError(
                    f"latent_type={provided!r} conflicts with bottleneck_type="
                    f"{bt!r} (expected {expected!r})"
                )
            data["latent_type"] = expected
        return data

    @pdt.model_validator(mode="before")
    @classmethod
    def _default_token_bin_size_ms(cls, data: object) -> object:
        """Set token_bin_size_ms from bin_size_ms * temporal_compression."""
        if isinstance(data, dict) and data.get("token_bin_size_ms") is None:
            bin_size = data.get("bin_size_ms")
            tc = data.get("temporal_compression", 16)
            if isinstance(bin_size, int | float):
                data["token_bin_size_ms"] = float(bin_size) * int(tc)
        return data

    @pdt.model_validator(mode="after")
    def _validate_shape_params(self) -> CosmosTokenizer2DSpec:
        if self.patch_method not in SUPPORTED_WAVELETS:
            raise ValueError(
                f"patch_method={self.patch_method!r} not in supported "
                f"wavelets {SUPPORTED_WAVELETS}"
            )
        for name, comp, patch in [
            ("neuron", self.neuron_compression, self.patch_h),
            ("temporal", self.temporal_compression, self.patch_w),
        ]:
            if patch < 1 or (patch & (patch - 1)) != 0:
                raise ValueError(f"patch_{name[0]} must be a positive power of 2, got {patch}")
            if comp < 1 or (comp & (comp - 1)) != 0:
                raise ValueError(f"{name}_compression must be a positive power of 2, got {comp}")
            if comp % patch != 0:
                raise ValueError(
                    f"{name}_compression ({comp}) must be a multiple of patch ({patch})"
                )
            remaining = comp // patch
            if remaining < 1 or (remaining & (remaining - 1)) != 0:
                raise ValueError(
                    f"{name}_compression / patch must be a power of 2, got {remaining}"
                )
        return self

    def get_vocab_size(self) -> int:
        if self.bottleneck_type == "fsq":
            vocab = 1
            for lvl in self.fsq_levels:
                vocab *= lvl
            return vocab
        if self.bottleneck_type == "vqpt":
            return vqpt_vocab_size(self.vqpt_kwargs)
        raise NotImplementedError(f"{self.bottleneck_type} has no discrete vocabulary")

    def get_codes_per_token(self) -> int:
        """Codes of size ``get_vocab_size()`` per latent position.

        FSQ: 1. The ``len(fsq_levels)`` lattice values form one code. VQPT: the
        wrapped quantizer's count, e.g. ``num_quantizers`` for ``rvq``/``residual_fsq``.
        """
        if self.bottleneck_type == "fsq":
            return 1
        if self.bottleneck_type == "vqpt":
            return vqpt_codes_per_token(self.vqpt_type, self.vqpt_kwargs)
        raise NotImplementedError(f"{self.bottleneck_type} has no discrete vocabulary")

    @property
    def num_latent_time_steps(self) -> int:
        tc = self.temporal_compression
        return (self.window_length_bins + tc - 1) // tc

    @property
    def num_latent_units(self) -> int:
        nc = self.neuron_compression
        return (self.n_neurons + nc - 1) // nc


@register_tokenizer
class CosmosTokenizer2D(BaseTokenizer):
    """Causal-in-time 2D wavelet + conv tokenizer.

    Input  ``BinnedSpikesInput`` with ``spike_matrix`` shape ``[B, T, N]``.
    Encoder latent in unified 4D format: ``[B, T', U', D]`` where ``T' =
    W_latent``, ``U' = H_latent``, ``D = z_channels`` (FSQ) or
    ``latent_channels`` (AE).
    """

    spec_class = CosmosTokenizer2DSpec
    spec: CosmosTokenizer2DSpec

    def __init__(self, spec: CosmosTokenizer2DSpec) -> None:
        super().__init__(spec)

        self.bottleneck = build_bottleneck(
            spec.bottleneck_type,
            z_channels=spec.z_channels,
            latent_channels=spec.latent_channels,
            fsq_levels=list(spec.fsq_levels),
            vae_beta=spec.vae_beta,
            vqpt_type=spec.vqpt_type,
            vqpt_kwargs=dict(spec.vqpt_kwargs),
            vqpt_loss_weight=spec.vqpt_loss_weight,
            ndim=2,
        )

        shared: dict[str, object] = {
            "channels": spec.channels,
            "channels_mult": list(spec.channels_mult),
            "num_res_blocks": spec.num_res_blocks,
            "dropout": spec.dropout,
            "neuron_compression": spec.neuron_compression,
            "temporal_compression": spec.temporal_compression,
            "patch_h": spec.patch_h,
            "patch_w": spec.patch_w,
            "patch_method": spec.patch_method,
        }
        self.encoder = Encoder2D(
            in_channels=1,
            z_channels=self.bottleneck.encoder_out_channels,
            **shared,  # type: ignore[arg-type]
        )
        self.decoder = Decoder2D(
            out_channels=1,
            z_channels=spec.z_channels,
            **shared,  # type: ignore[arg-type]
        )

    # ------------------------------------------------------------------
    # Shape helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _spikes_to_conv(x: BinnedInputType) -> Tensor:
        """``BinnedSpikesInput`` or ``RatesInput`` ``[B, T, N]`` -> conv-ready ``[B, 1, N, T]``."""
        data = get_data_matrix(x).float() * x.pad_and_gap_mask.float()
        return rearrange(data, "b t n -> b () n t")

    @staticmethod
    def _conv_to_spikes(h: Tensor) -> Tensor:
        """``[B, 1, N, T]`` -> ``[B, T, N]``."""
        return rearrange(h, "b () n t -> b t n")

    @staticmethod
    def _latent_to_4d(z: Tensor) -> Tensor:
        """``[B, D, H, W]`` -> unified ``[B, W, H, D]`` (time, units, latent)."""
        return rearrange(z, "b d h w -> b w h d")

    @staticmethod
    def _latent_from_4d(z: Tensor) -> Tensor:
        """``[B, W, H, D]`` -> ``[B, D, H, W]``."""
        if z.ndim != 4:
            raise ValueError(f"Expected latent [B, T', U', D], got {tuple(z.shape)}")
        return rearrange(z, "b w h d -> b d h w")

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def encode(self, x: BinnedInputType) -> Tensor:
        h = self._spikes_to_conv(x)
        h = self.encoder(h)
        z, _ = self.bottleneck.encode(h)
        return self._latent_to_4d(z)

    def decode(self, tokens: Tensor) -> Tensor:
        z = self._latent_from_4d(tokens)
        z_for_decoder = self.bottleneck.decode(z)
        recon = self.decoder(
            z_for_decoder,
            trim_h=self.spec.n_neurons,
            trim_w=self.spec.window_length_bins,
        )
        recon = self._conv_to_spikes(recon)
        return torch.nn.functional.softplus(recon).clamp(min=_RATE_FLOOR)

    def reconstruct(self, tokens: Tensor) -> ReconstructionOutput:
        # Dispatch on dtype: discrete bottlenecks (FSQ / VQPT) emit integer
        # tokens that the litmodule passes here for visualization. Look up
        # codes and run the decoder; continuous bottlenecks (AE / VAE) get
        # the existing 4D-latent path.
        if (
            isinstance(self.bottleneck, FSQBottleneck2d | VQPTBottleneck2d)
            and not tokens.is_floating_point()
        ):
            # Tokens are in spatial layout [B, T_lat, U_lat] or [B, T_lat, U_lat, Q].
            # Restore conv-style [B, U_lat, T_lat, ...] for indices_to_codes.
            if tokens.ndim == 3:
                indices = rearrange(tokens, "b w h -> b h w")
            else:
                indices = rearrange(tokens, "b w h q -> b h w q")
            codes = self.bottleneck.indices_to_codes(indices)
            recon_conv = self.decoder(
                codes, trim_h=self.spec.n_neurons, trim_w=self.spec.window_length_bins
            )
            rates = torch.nn.functional.softplus(self._conv_to_spikes(recon_conv)).clamp(
                min=_RATE_FLOOR
            )
        else:
            rates = self.decode(tokens)
        # Reconstruct what output_type declares (as PCA does): RATES -> expected
        # counts per bin; SPIKES -> Poisson-sampled counts outside training.
        if self.training or self.spec.output_type == DataType.RATES:
            output = rates
            data_type = self.spec.output_type
        else:
            output = torch.poisson(rates).clamp(min=0).long()
            data_type = DataType.SPIKES
        bin_size_seconds = (
            self.spec.bin_size_ms / 1000.0 if self.spec.bin_size_ms is not False else None
        )
        return ReconstructionOutput(
            reconstruction=output,
            data_type=data_type,
            bin_size_seconds=bin_size_seconds,
        )

    def forward(self, x: BinnedInputType) -> TokenizerOutput:
        h = self._spikes_to_conv(x)
        h_enc = self.encoder(h)
        z, aux_loss = self.bottleneck.encode(h_enc)
        z_for_decoder = self.bottleneck.decode(z)
        recon_conv = self.decoder(
            z_for_decoder,
            trim_h=self.spec.n_neurons,
            trim_w=self.spec.window_length_bins,
        )
        recon = torch.nn.functional.softplus(self._conv_to_spikes(recon_conv)).clamp(
            min=_RATE_FLOOR
        )

        latents = self._latent_to_4d(z)
        tokens: Tensor | None = None
        if isinstance(self.bottleneck, FSQBottleneck2d | VQPTBottleneck2d):
            indices = self.bottleneck.get_indices(h_enc)
            # Keep tokens in spatial form mirroring the 4D latent layout:
            # FSQ / single-codebook VQPT → [B, T_lat, U_lat]
            # VQPT residual variants → [B, T_lat, U_lat, Q]
            # (Don't flatten — the spec's num_latent_* is a Haar approximation
            # and disagrees with the actual encoder output shape under longer
            # wavelets. Keeping the spatial axes makes reconstruct() trivial.)
            if indices.ndim == 3:
                tokens = rearrange(indices, "b h w -> b w h")
            else:
                tokens = rearrange(indices, "b h w q -> b w h q")

        return TokenizerOutput(
            tokens=tokens,
            reconstruction=recon,
            aux_loss=aux_loss,
            latents=latents,
        )
