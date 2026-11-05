"""WanNeural tokenizer for neural data.

Adapts the Wan2pt1/Wan2pt2 video VAE architecture to 2D (time x neurons)
for tokenizing multi-neuron spike trains or firing rates. Supports both
FSQ (discrete) and VAE (continuous) latent bottlenecks.

Adapted from the Wan2.1/Wan2.2 video VAE, used under the Apache-2.0 license:
https://github.com/Wan-Video/Wan2.2 (see ``LICENSES/wan-Apache-2.0.txt``).
"""

from __future__ import annotations

import math
from typing import Literal

import pydantic as pdt
import torch
from torch import Tensor, nn

from ntt.contrib.loss.sigreg import SIGReg
from ntt.contrib.tokenizer.vqpt_codes import vqpt_codes_per_token, vqpt_vocab_size
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

from .blocks import temporal_patchify, temporal_unpatchify
from .bottleneck import (
    BSQBottleneck,
    DiagonalGaussianDistribution,
    DiffusionPriorBottleneck,
    FSQBottleneck,
    MMDVAEBottleneck,
    VAEBottleneck,
    VQPTBottleneck,
)
from .corruptions import apply_input_augmentations
from .decoder import Decoder2d
from .encoder import Encoder2d

# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------


class WanNeuralTokenizerSpec(TokenizerSpec, frozen=True):  # pyright: ignore[reportCallIssue,reportGeneralTypeIssues]
    """Specification for the WanNeural tokenizer.

    A convolutional encoder-decoder with attention, adapted from the Wan2pt1/Wan2pt2
    video VAE. Compresses neural data along the temporal dimension with optional
    cross-neuron information sharing via attention.

    See the plan in WANPLAN.md for full architectural details.

    Data Parameters (inherited):
        n_neurons: Number of neurons per sample.
        window_length_seconds: Duration of input window.
        bin_size_ms: Temporal bin size.

    Architecture Parameters:
        dim: Base encoder channel dimension.
        dec_dim: Decoder channel dimension (None → same as dim).
        z_dim: Latent channels per spatial position.
        num_levels: Number of encoder stages.
        dim_mult: Encoder channel multipliers per level.
        num_res_blocks: Encoder residual blocks per stage.
        attn_levels: Encoder levels that use cross-neuron attention.
        num_attn_heads: Encoder attention heads.

    Decoder Architecture Overrides (None → inherit from encoder value):
        dec_num_levels, dec_dim_mult, dec_num_res_blocks, dec_attn_levels,
        dec_num_attn_heads, dec_stage_shortcut, dec_mid_attn,
        dec_neuron_kernel_size, dec_k_t, dec_k_t_up, dec_s_t_up,
        dec_k_t_init, dec_s_t_init, dec_temporal_upsample.

        When the decoder overrides temporal strides or downsample pattern,
        the total temporal expansion must still equal the encoder's
        compression (enforced by validator).

    Temporal Processing:
        temporal_patch_size: Temporal patchification factor (1 = none).
        temporal_downsample: Per-level temporal downsampling flags.
        k_t_init: Initial conv temporal kernel size.
        s_t_init: Initial conv temporal stride.

    Neuron Dimension:
        neuron_kernel_size: Conv kernel along N (1 = per-neuron, 3 = local mixing).

    Bottleneck:
        bottleneck_type: "vae" for continuous, "fsq" for discrete.
        beta: KL weight (VAE mode).
        q_levels: FSQ quantization levels (FSQ mode).

    Output:
        output_activation: Activation on decoder output.
    """

    KIND: Literal["wanneural"] = "wanneural"

    # --- Data parameters ---
    n_neurons: int = 32
    window_length_seconds: float | Dist = 1.0
    bin_size_ms: float = 1.0

    # --- Encoder architecture ---
    dim: int = 64
    z_dim: int = 16
    num_levels: int = 3
    dim_mult: tuple[int, ...] = (1, 2, 4)
    num_res_blocks: int = 2
    attn_levels: tuple[int, ...] = (2,)
    num_attn_heads: int = 1
    # Optional FFN on neuron-axis attention blocks (turns them into full
    # pre-norm Transformer blocks). Default off preserves existing behavior.
    attn_use_ffn: bool = False
    attn_ffn_dim_mult: int = 4

    # --- Temporal attention (mirrors the neuron-attention configuration shape) ---
    temp_attn_levels: tuple[int, ...] = ()
    temp_num_attn_heads: int = 1
    temp_mid_attn: bool = False
    temp_attn_causal: bool = False  # bidirectional within block by default
    # Optional FFN on temporal-axis attention blocks (full Transformer block).
    temp_attn_use_ffn: bool = False
    temp_attn_ffn_dim_mult: int = 4
    # Optional Rotary Positional Embedding on temporal-axis Q/K. Requires
    # head_dim = channels // num_heads to be even at every enabled location.
    temp_attn_use_rope: bool = False

    # --- Decoder architecture overrides (None → inherit encoder value) ---
    dec_dim: int | None = None
    dec_num_levels: int | None = None
    dec_dim_mult: tuple[int, ...] | None = None
    dec_num_res_blocks: int | None = None
    dec_attn_levels: tuple[int, ...] | None = None
    dec_num_attn_heads: int | None = None
    dec_attn_use_ffn: bool | None = None
    dec_attn_ffn_dim_mult: int | None = None
    dec_stage_shortcut: bool | None = None
    dec_mid_attn: bool | None = None
    dec_temp_attn_levels: tuple[int, ...] | None = None
    dec_temp_num_attn_heads: int | None = None
    dec_temp_mid_attn: bool | None = None
    dec_temp_attn_causal: bool | None = None
    dec_temp_attn_use_ffn: bool | None = None
    dec_temp_attn_ffn_dim_mult: int | None = None
    dec_temp_attn_use_rope: bool | None = None
    dec_neuron_kernel_size: int | None = None
    dec_k_t: int | None = None
    dec_k_t_up: int | None = None  # decoder upsample conv kernel (mirrors k_t_down)
    dec_s_t_up: int | None = None  # decoder upsample conv stride (mirrors s_t_down)
    dec_k_t_init: int | None = None
    dec_s_t_init: int | None = None
    dec_temporal_upsample: tuple[bool, ...] | None = None

    # --- Temporal processing ---
    temporal_patch_size: int = 1
    temporal_downsample: tuple[bool, ...] = (True, True)
    k_t: int = 3  # temporal kernel size for res blocks, mid blocks, head convs
    k_t_down: int = 3  # temporal kernel size for downsample/upsample convs
    s_t_down: int = 2  # temporal stride for downsample/upsample
    k_t_init: int = 3
    s_t_init: int = 1

    # --- Stage shortcuts ---
    stage_shortcut: bool = True  # Wan2pt2-style (True) vs Wan2pt1-style (False)
    mid_attn: bool = True  # Whether to use attention in mid blocks

    # --- Regularization & normalization (LDM-style knobs) ---
    # dropout is applied in the second conv of each ResidualBlock (matches LDM).
    # Decoder inherits encoder dropout when dec_dropout is None.
    dropout: float = 0.0
    dec_dropout: float | None = None
    # "rms" (Wan2 default RMSNorm2d) or "group" (LDM-style nn.GroupNorm).
    norm_type: Literal["rms", "group"] = "rms"
    num_groups: int = 32

    # --- Temporal causality ---
    # When True (default), all temporal convs use left-only padding so each
    # output position only depends on past + current input — preserving the
    # auto-regressive contract of the Wan2 video VAE family. When False, padding
    # is split symmetrically along time and the model is permitted to mix in
    # information from "future" bins. For configs with temporal kernel sizes of
    # 1 (e.g. the conv-* models, internal family name wanx_A), the two modes
    # differ only at strided downsample/upsample boundaries.
    causal: bool = True

    # --- Neuron dimension ---
    neuron_kernel_size: int = 1

    # --- Bottleneck ---
    bottleneck_type: Literal["vae", "fsq", "bsq", "vqpt", "diffusion_prior", "mmd_vae"] = "fsq"
    beta: float = 1.0
    # VAE curvature aux loss on the sampled z over time.
    # Zero disables it; set step_thresh to floor out near-stationary frames.
    # `curvature_metric` selects between cosine curvature (1 - cos(v1, v2))
    # and L2 curvature (||v2 - v1||).
    curvature_weight: float = 0.0
    curvature_step_thresh: float = 1e-6
    curvature_metric: Literal["cos", "l2"] = "cos"
    # Tile-batched curvature: when tile_stride > 0 (training only), the tokenizer
    # tiles the input along time into B' = B * n_tiles separate `tile_window`-bin
    # batch items, runs the entire encode/bottleneck/decode pipeline on the
    # tiled batch, average-untiles recon back to (B, T, N) for the standard
    # loss API, and computes the curvature loss on latents reshaped to
    # (B, z_dim, n_tiles, N) — i.e. each curvature step compares latents from
    # neighboring tiles in the original temporal order. KL is computed on the
    # tiled latents (different scale than the standard 10-step path), so
    # curvature_weight may need recalibrating for tiled training.
    # n_tiles=0 means "max possible given tile_stride"; otherwise the value caps
    # the count and the effective stride is auto-adjusted to span the full input.
    # Hardcoded for the 100 ms-token / 1 ms-bin geometry of the conv-* models
    # (internal family name wanx_A): tile_window must equal the tokenizer's
    # temporal_compression so each tile produces one step.
    tile_stride: int = 0
    tile_window: int = 100
    n_tiles: int = 0
    q_levels: tuple[int, ...] = (8, 5, 5, 5)
    # BSQ parameters
    bsq_codebook_dim: int | None = None  # None → z_dim (no projection)
    bsq_inv_temperature: float = 100.0
    bsq_gamma0: float = 1.0  # per-sample entropy weight
    bsq_gamma: float = 1.0  # batch entropy weight
    bsq_zeta: float = 1.0  # overall entropy penalty scale
    # MMD-VAE parameters (used when bottleneck_type == "mmd_vae")
    # mmd_lambda is the only knob you'll usually tune — start at 1.0 and
    # adjust so the MMD term and reconstruction term are comparable in
    # magnitude at convergence (paper recommendation).
    mmd_lambda: float = 1.0
    mmd_kernel_bandwidth_div: float = 1.0
    mmd_n_prior_samples: int | None = None
    mmd_max_samples: int = 2048  # cap on posterior samples fed into O(n²) kernel
    # VQPT parameters (used when bottleneck_type == "vqpt")
    # vqpt_type picks which lucidrains/vector-quantize-pytorch quantizer to use
    # (e.g. "vq", "fsq", "lfq", "residual_vq", "residual_fsq", "sim_vq",
    # "latent_quantize"). vqpt_kwargs is passed through to the wrapped
    # constructor; do not include `dim` (auto-supplied from z_dim).
    vqpt_type: str = "vq"
    vqpt_kwargs: dict = pdt.Field(default_factory=dict)
    vqpt_loss_weight: float = 1.0

    # --- Input augmentations (training only, BinnedSpikesInput only) ---
    # Applied to the encoder-input tensor inside ``forward`` while the
    # reconstruction target stays the original, un-augmented input — so the
    # latent has to be robust to the corruption. All scales default to 0
    # (no-op). See ``corruptions.py`` for the exact distributions.
    input_aug_jitter_std_bins: float = 0.0
    input_aug_jitter_cap_bins: int = 3
    input_aug_spot_noise_p: float = 0.0
    input_aug_time_shift_std_bins: float = 0.0
    input_aug_time_shift_cap_bins: int = 5

    # --- Diffusion-prior bottleneck (Unified Latents) ---
    # Schedule endpoints for the latent forward process. lambda_z_max also
    # sets the encoder's σ₀ noise level; the recommended default is 5.
    lambda_z_min: float = -10.0
    lambda_z_max: float = 5.0
    # Architecture of the small DiT-style score network.
    prior_dim: int = 256
    prior_depth: int = 6
    prior_num_heads: int = 4
    # Decoder-loss factor from the plan: total ≈ c_lf · recon + (prior + endpoint_kl).
    # Implemented by scaling the bottleneck loss by 1/c_lf in forward(), which is
    # mathematically equivalent up to a constant scale on the total loss. Tune in
    # [1.3, 1.7]; rate-distortion knob.
    c_lf: float = 1.5

    # --- SIGReg latent regularization (LeJEPA-style anti-collapse) ---
    # Pushes the pooled latent samples — flattened across batch, time, and
    # neurons — toward an isotropic Gaussian via random-feature characteristic
    # function matching. Targets the post-bottleneck latent (z_for_decoder)
    # so it applies uniformly to vae / diffusion_prior / fsq / bsq / vqpt
    # modes. Disabled by default; only active during training.
    use_sigreg: bool = False
    sigreg_beta: float = 0.01
    sigreg_knots: int = 17
    sigreg_t_max: float = 3.0
    sigreg_sketch_dim: int = 256
    sigreg_seed: int = 42

    # --- Latent normalization (Wan2pt2-style) ---
    latent_mean: tuple[float, ...] | None = None  # per-channel mean
    latent_std: tuple[float, ...] | None = None  # per-channel std

    # --- Output ---
    output_activation: Literal["softplus", "sigmoid", "none"] = "softplus"
    output_act_beta: float = 0.1

    # --- Reconstruction sampling ---
    reconstruct_sample: Literal["poisson", "round", "bernoulli"] = "poisson"
    reconstruct_max_count: int | None = 10

    # --- AR regularization ---
    # Auxiliary next-token CE loss applied per-neuron over time on the
    # post-bottleneck quantized latents. Mirrors CalM § 2.6 (Eq. 8). Only the
    # CE term is implemented — the paper's "Align" term is undefined in the
    # methods section. Set ar_weight=0 to disable. Requires a discrete
    # bottleneck whose ``get_indices`` returns a 2D ``(B, T_comp * N)`` tensor:
    # FSQ, BSQ, and single-codebook VQPT variants.
    ar_weight: float = 0.0
    ar_horizon: int = 4
    ar_num_heads: int = 4
    ar_dropout: float | None = None  # None → fall back to spec.dropout

    # Temporal resolution of the tokenizer's latent tokens.
    token_bin_size_ms: float | None = None
    has_time_dim: bool = True  # Tokenizer produces multiple tokens over time
    has_unit_dim: bool = True  # Tokenizer produces per-unit representations
    has_latent_dim: bool = True  # Tokenizer produces multi-dim latent per token

    # --- Capability metadata ---
    latent_type: LatentType = LatentType.DISCRETE
    compress_n: bool = False
    compress_t: bool = True
    variable_n: bool = False
    variable_t: bool = False
    variable_dt: bool = False
    input_type: DataType = DataType.RATES
    output_type: DataType = DataType.RATES

    def _validate_temp_attn_locations(self) -> None:
        """Check block_size = token_bin_size_ms / (bin_size_ms × cum_stride) is integer ≥ 2.

        Validates every enabled temporal-attention location.
        """
        has_temp = (
            len(self.temp_attn_levels) > 0
            or self.temp_mid_attn
            or len(self.effective_dec_temp_attn_levels) > 0
            or self.effective_dec_temp_mid_attn
        )
        if not has_temp:
            return
        if self.token_bin_size_ms is None:
            raise ValueError(
                "temporal attention is enabled but token_bin_size_ms is None; "
                "set token_bin_size_ms to the desired token-window duration in ms"
            )
        bin_ms = float(self.bin_size_ms)
        tok_ms = float(self.token_bin_size_ms)

        def _check(label: str, cum_stride: int) -> None:
            eff_bin_ms = bin_ms * cum_stride
            q = tok_ms / eff_bin_ms
            if not math.isfinite(q) or abs(q - round(q)) > 1e-6:
                raise ValueError(
                    f"temporal attention at {label}: "
                    f"token_bin_size_ms={tok_ms} is not an integer multiple of "
                    f"effective_bin_ms={eff_bin_ms} (cum_stride={cum_stride}); "
                    f"got block_size={q}. Adjust token_bin_size_ms, bin_size_ms, "
                    f"temporal_patch_size, s_t_init, or s_t_down so the ratio is integer."
                )
            block_size = int(round(q))
            if block_size < 2:
                raise ValueError(
                    f"temporal attention at {label}: derived block_size={block_size} "
                    f"< 2 (cum_stride={cum_stride}, eff_bin_ms={eff_bin_ms}). "
                    f"This level's temporal resolution already meets or exceeds "
                    f"the token window — remove this location from "
                    f"temp_attn_levels/temp_mid_attn (or the dec_ counterparts), "
                    f"or change patch/stride parameters."
                )

        # Encoder walk: stages first, then mid
        cum_stride = self.temporal_patch_size * self.s_t_init
        for i in range(self.num_levels):
            if i in self.temp_attn_levels:
                _check(f"encoder stage {i}", cum_stride)
            if i < len(self.temporal_downsample) and self.temporal_downsample[i]:
                cum_stride *= self.s_t_down
        if self.temp_mid_attn:
            _check("encoder mid", cum_stride)

        # Decoder walk: mid first, then up-stages in encoder-decreasing order
        dec_total = (
            self.temporal_patch_size
            * self.effective_dec_s_t_init
            * (self.effective_dec_s_t_up ** sum(self.effective_dec_temporal_upsample))
        )
        cum_stride = dec_total
        if self.effective_dec_temp_mid_attn:
            _check("decoder mid", cum_stride)
        dec_num_levels = self.effective_dec_num_levels
        dec_upsample = self.effective_dec_temporal_upsample
        for i in range(dec_num_levels):
            enc_level = dec_num_levels - 1 - i
            if enc_level in self.effective_dec_temp_attn_levels:
                _check(f"decoder stage {enc_level}", cum_stride)
            if i < len(dec_upsample) and dec_upsample[i]:
                cum_stride //= self.effective_dec_s_t_up

    @pdt.model_validator(mode="before")
    @classmethod
    def _derive_latent_type(cls, data: dict) -> dict:
        """Derive latent_type from bottleneck_type before the model is frozen."""
        if isinstance(data, dict):
            bt = data.get("bottleneck_type", "fsq")
            # VQPT wraps any of the lucidrains quantizers — all of them produce
            # discrete tokens — so it joins fsq/bsq on the discrete side.
            # diffusion_prior is deterministic-then-noised but the latent is
            # continuous-valued, so it joins vae on the continuous side.
            # mmd_vae is a continuous reparameterized Gaussian (same encoder
            # head as vae), only the regularizer differs.
            expected = "continuous" if bt in ("vae", "diffusion_prior", "mmd_vae") else "discrete"
            lt = data.get("latent_type", expected)
            if isinstance(lt, LatentType):
                lt = lt.value
            if lt != expected:
                raise ValueError(
                    f"latent_type={lt!r} conflicts with "
                    f"bottleneck_type={bt!r} (expected {expected!r})"
                )
            data["latent_type"] = expected
        return data

    @pdt.model_validator(mode="after")
    def _validate_config(self) -> WanNeuralTokenizerSpec:
        if self.tile_stride > 0:
            if self.tile_window <= 0:
                raise ValueError(
                    f"tile_window must be > 0 when tile_stride>0 (got {self.tile_window})"
                )
            if self.tile_window != self.temporal_compression:
                raise ValueError(
                    f"tile_window ({self.tile_window}) must equal temporal_compression "
                    f"({self.temporal_compression}) so each tile produces exactly one "
                    "latent step."
                )
            if self.window_length_bins < self.tile_window:
                raise ValueError(
                    f"window_length_bins ({self.window_length_bins}) must be ≥ "
                    f"tile_window ({self.tile_window})"
                )
            if self.n_tiles < 0:
                raise ValueError(f"n_tiles must be ≥ 0 (got {self.n_tiles})")
        if len(self.dim_mult) != self.num_levels:
            raise ValueError(
                f"len(dim_mult)={len(self.dim_mult)} must equal num_levels={self.num_levels}"
            )
        if len(self.temporal_downsample) != self.num_levels - 1:
            raise ValueError(
                f"len(temporal_downsample)={len(self.temporal_downsample)} must equal "
                f"num_levels - 1 = {self.num_levels - 1}"
            )
        # Validate decoder architecture overrides
        dec_num_levels = self.effective_dec_num_levels
        dec_dim_mult = self.effective_dec_dim_mult
        dec_temporal_upsample = self.effective_dec_temporal_upsample
        if len(dec_dim_mult) != dec_num_levels:
            raise ValueError(
                f"len(dec_dim_mult)={len(dec_dim_mult)} must equal dec_num_levels={dec_num_levels}"
            )
        if len(dec_temporal_upsample) != dec_num_levels - 1:
            raise ValueError(
                f"len(dec_temporal_upsample)={len(dec_temporal_upsample)} must equal "
                f"dec_num_levels - 1 = {dec_num_levels - 1}"
            )
        # Encoder compression and decoder expansion must match so the
        # round-trip recovers the input temporal length.
        enc_comp = self.s_t_init * (self.s_t_down ** sum(self.temporal_downsample))
        dec_exp = self.effective_dec_s_t_init * (
            self.effective_dec_s_t_up ** sum(dec_temporal_upsample)
        )
        if enc_comp != dec_exp:
            raise ValueError(
                f"decoder temporal expansion ({dec_exp}) must equal encoder temporal "
                f"compression ({enc_comp}). Check dec_s_t_init, dec_s_t_up, and "
                f"dec_temporal_upsample."
            )
        # Validate latent normalization shapes
        if (self.latent_mean is None) != (self.latent_std is None):
            raise ValueError("latent_mean and latent_std must both be set or both be None")
        if self.latent_mean is not None and len(self.latent_mean) != self.z_dim:
            raise ValueError(
                f"len(latent_mean)={len(self.latent_mean)} must equal z_dim={self.z_dim}"
            )
        if self.latent_std is not None and len(self.latent_std) != self.z_dim:
            raise ValueError(
                f"len(latent_std)={len(self.latent_std)} must equal z_dim={self.z_dim}"
            )
        if self.output_type == DataType.SPIKES and self.reconstruct_sample not in (
            "round",
            "bernoulli",
        ):
            raise ValueError(
                f"reconstruct_sample must be 'round' or 'bernoulli' when output_type is "
                f"SPIKES, got '{self.reconstruct_sample}'. Poisson sampling doesn't make "
                f"sense for binary spike outputs."
            )
        if (
            self.reconstruct_sample == "bernoulli"
            and self.bin_size_ms is not False
            and self.bin_size_ms is not None
            and self.bin_size_ms > 2.0
        ):
            raise ValueError(
                f"reconstruct_sample='bernoulli' assumes at most 1 spike per bin, "
                f"but bin_size_ms={self.bin_size_ms} is too coarse. "
                f"Use 'poisson' for bins > 2ms."
            )
        # Variable-window training: VariableWindowCrop snaps each sample to a
        # multiple of `patch_length` bins, so we just need the Dist bounds to
        # contain at least one valid patch length.
        if isinstance(self.window_length_seconds, Dist):
            patch_duration_ms = self.temporal_compression * self.bin_size_ms
            min_ms = float(self.window_length_seconds.min) * 1000.0
            max_ms = float(self.window_length_seconds.max) * 1000.0
            if max_ms < patch_duration_ms:
                raise ValueError(
                    f"window_length_seconds.max={self.window_length_seconds.max}s is "
                    f"smaller than one temporal patch ({patch_duration_ms / 1000.0}s = "
                    f"temporal_compression={self.temporal_compression} * "
                    f"bin_size_ms={self.bin_size_ms}). Increase Dist.max or reduce "
                    f"temporal compression."
                )
            if min_ms <= 0:
                raise ValueError(
                    f"window_length_seconds.min={self.window_length_seconds.min}s must be > 0."
                )
        if not 0.0 <= self.input_aug_spot_noise_p <= 1.0:
            raise ValueError(
                f"input_aug_spot_noise_p must be in [0, 1], got {self.input_aug_spot_noise_p}"
            )
        if self.input_aug_jitter_std_bins < 0.0:
            raise ValueError(
                f"input_aug_jitter_std_bins must be >= 0, got {self.input_aug_jitter_std_bins}"
            )
        if self.input_aug_jitter_cap_bins < 0:
            raise ValueError(
                f"input_aug_jitter_cap_bins must be >= 0, got {self.input_aug_jitter_cap_bins}"
            )
        if self.input_aug_time_shift_std_bins < 0.0:
            raise ValueError(
                f"input_aug_time_shift_std_bins must be >= 0, "
                f"got {self.input_aug_time_shift_std_bins}"
            )
        if self.input_aug_time_shift_cap_bins < 0:
            raise ValueError(
                f"input_aug_time_shift_cap_bins must be >= 0, "
                f"got {self.input_aug_time_shift_cap_bins}"
            )
        self._validate_temp_attn_locations()
        self._validate_ar_regularizer()
        return self

    def _validate_ar_regularizer(self) -> None:
        """Ensure AR-regularizer settings are consistent with the bottleneck.

        AR uses 2D ``(B, T_comp * N)`` discrete indices and is only defined for
        single-codebook discrete bottlenecks. Reject early when ``ar_weight``
        is enabled but the bottleneck either lacks indices (VAE, MMD-VAE,
        diffusion_prior) or emits multi-codebook / residual / grouped indices
        (the residual_* and grouped_* VQPT variants).
        """
        if self.ar_weight < 0.0:
            raise ValueError(f"ar_weight must be >= 0, got {self.ar_weight}")
        if self.ar_horizon < 1:
            raise ValueError(f"ar_horizon must be >= 1, got {self.ar_horizon}")
        if self.ar_weight == 0.0:
            return
        if self.bottleneck_type in ("vae", "diffusion_prior", "mmd_vae"):
            raise ValueError(
                f"ar_weight>0 requires a discrete bottleneck (fsq/bsq/vqpt), "
                f"got bottleneck_type={self.bottleneck_type!r}."
            )
        if self.bottleneck_type == "vqpt":
            multi_codebook_types = {
                "residual_vq",
                "residual_fsq",
                "residual_lfq",
                "grouped_residual_vq",
                "grouped_residual_fsq",
            }
            if self.vqpt_type in multi_codebook_types:
                raise ValueError(
                    f"ar_weight>0 is incompatible with vqpt_type={self.vqpt_type!r}; "
                    f"residual / grouped variants emit non-2D index tensors. "
                    f"Use a single-codebook quantizer (vq, fsq, lfq, sim_vq, "
                    f"latent_quantize) or set ar_weight=0."
                )
            n_quantizers = self.vqpt_kwargs.get(
                "num_quantizers", self.vqpt_kwargs.get("num_codebooks", 1)
            )
            if isinstance(n_quantizers, int) and n_quantizers > 1:
                raise ValueError(
                    f"ar_weight>0 is incompatible with vqpt_kwargs producing "
                    f"multiple codebooks (num_quantizers/num_codebooks="
                    f"{n_quantizers}); use a single-codebook configuration "
                    f"or set ar_weight=0."
                )

    @property
    def has_input_augmentations(self) -> bool:
        """True iff any training-time input augmentation is enabled."""
        return (
            self.input_aug_jitter_std_bins > 0.0
            or self.input_aug_spot_noise_p > 0.0
            or self.input_aug_time_shift_std_bins > 0.0
        )

    def get_vocab_size(self) -> int:
        """Return codebook size for discrete bottleneck modes (FSQ, BSQ, VQPT)."""
        if self.bottleneck_type == "fsq":
            return math.prod(self.q_levels)
        elif self.bottleneck_type == "bsq":
            return 2 ** (self.bsq_codebook_dim or self.z_dim)
        elif self.bottleneck_type == "vqpt":
            # Per-codebook size for the wrapped quantizer. Residual / grouped
            # variants emit multiple tokens per position (one per quantizer or
            # group), and the effective vocabulary is this size raised to the
            # number of tokens per position; downstream LMs should consume the
            # multi-token layout directly.
            return vqpt_vocab_size(self.vqpt_kwargs)
        raise NotImplementedError(
            f"{self.bottleneck_type!r} bottleneck has no discrete vocabulary"
        )

    def get_codes_per_token(self) -> int:
        """Codes of size ``get_vocab_size()`` per latent position.

        FSQ and BSQ: 1. Their ``z_dim`` floats form one code of ``prod(q_levels)`` or
        ``2**codebook_dim``. VQPT: the wrapped quantizer's count, e.g.
        ``num_quantizers`` for residual variants.
        """
        if self.bottleneck_type in ("fsq", "bsq"):
            return 1
        if self.bottleneck_type == "vqpt":
            return vqpt_codes_per_token(self.vqpt_type, self.vqpt_kwargs)
        raise NotImplementedError(
            f"{self.bottleneck_type!r} bottleneck has no discrete vocabulary"
        )

    @property
    def effective_dec_dim(self) -> int:
        """Decoder base dimension (falls back to encoder dim)."""
        return self.dec_dim if self.dec_dim is not None else self.dim

    @property
    def effective_dec_dim_mult(self) -> tuple[int, ...]:
        return self.dec_dim_mult if self.dec_dim_mult is not None else self.dim_mult

    @property
    def effective_dec_num_levels(self) -> int:
        if self.dec_num_levels is not None:
            return self.dec_num_levels
        if self.dec_dim_mult is not None:
            return len(self.dec_dim_mult)
        return self.num_levels

    @property
    def effective_dec_num_res_blocks(self) -> int:
        return (
            self.dec_num_res_blocks if self.dec_num_res_blocks is not None else self.num_res_blocks
        )

    @property
    def effective_dec_attn_levels(self) -> tuple[int, ...]:
        return self.dec_attn_levels if self.dec_attn_levels is not None else self.attn_levels

    @property
    def effective_dec_num_attn_heads(self) -> int:
        return (
            self.dec_num_attn_heads if self.dec_num_attn_heads is not None else self.num_attn_heads
        )

    @property
    def effective_dec_attn_use_ffn(self) -> bool:
        return self.dec_attn_use_ffn if self.dec_attn_use_ffn is not None else self.attn_use_ffn

    @property
    def effective_dec_attn_ffn_dim_mult(self) -> int:
        return (
            self.dec_attn_ffn_dim_mult
            if self.dec_attn_ffn_dim_mult is not None
            else self.attn_ffn_dim_mult
        )

    @property
    def effective_dec_stage_shortcut(self) -> bool:
        return (
            self.dec_stage_shortcut if self.dec_stage_shortcut is not None else self.stage_shortcut
        )

    @property
    def effective_dec_mid_attn(self) -> bool:
        return self.dec_mid_attn if self.dec_mid_attn is not None else self.mid_attn

    @property
    def effective_dec_temp_attn_levels(self) -> tuple[int, ...]:
        return (
            self.dec_temp_attn_levels
            if self.dec_temp_attn_levels is not None
            else self.temp_attn_levels
        )

    @property
    def effective_dec_temp_num_attn_heads(self) -> int:
        return (
            self.dec_temp_num_attn_heads
            if self.dec_temp_num_attn_heads is not None
            else self.temp_num_attn_heads
        )

    @property
    def effective_dec_temp_mid_attn(self) -> bool:
        return self.dec_temp_mid_attn if self.dec_temp_mid_attn is not None else self.temp_mid_attn

    @property
    def effective_dec_temp_attn_causal(self) -> bool:
        return (
            self.dec_temp_attn_causal
            if self.dec_temp_attn_causal is not None
            else self.temp_attn_causal
        )

    @property
    def effective_dec_temp_attn_use_ffn(self) -> bool:
        return (
            self.dec_temp_attn_use_ffn
            if self.dec_temp_attn_use_ffn is not None
            else self.temp_attn_use_ffn
        )

    @property
    def effective_dec_temp_attn_ffn_dim_mult(self) -> int:
        return (
            self.dec_temp_attn_ffn_dim_mult
            if self.dec_temp_attn_ffn_dim_mult is not None
            else self.temp_attn_ffn_dim_mult
        )

    @property
    def effective_dec_temp_attn_use_rope(self) -> bool:
        return (
            self.dec_temp_attn_use_rope
            if self.dec_temp_attn_use_rope is not None
            else self.temp_attn_use_rope
        )

    @property
    def effective_dec_neuron_kernel_size(self) -> int:
        return (
            self.dec_neuron_kernel_size
            if self.dec_neuron_kernel_size is not None
            else self.neuron_kernel_size
        )

    @property
    def effective_dec_k_t(self) -> int:
        return self.dec_k_t if self.dec_k_t is not None else self.k_t

    @property
    def effective_dec_k_t_up(self) -> int:
        return self.dec_k_t_up if self.dec_k_t_up is not None else self.k_t_down

    @property
    def effective_dec_s_t_up(self) -> int:
        return self.dec_s_t_up if self.dec_s_t_up is not None else self.s_t_down

    @property
    def effective_dec_k_t_init(self) -> int:
        return self.dec_k_t_init if self.dec_k_t_init is not None else self.k_t_init

    @property
    def effective_dec_s_t_init(self) -> int:
        return self.dec_s_t_init if self.dec_s_t_init is not None else self.s_t_init

    @property
    def effective_dec_temporal_upsample(self) -> tuple[bool, ...]:
        if self.dec_temporal_upsample is not None:
            return self.dec_temporal_upsample
        return self.temporal_downsample

    @property
    def effective_dec_dropout(self) -> float:
        return self.dec_dropout if self.dec_dropout is not None else self.dropout

    @property
    def temporal_compression(self) -> int:
        """Total temporal compression factor."""
        conv_compression = self.s_t_down ** sum(self.temporal_downsample)
        return self.temporal_patch_size * self.s_t_init * conv_compression

    @property
    def patch_length(self) -> int:
        """Bins covered by one latent token along time.

        Used by ``VariableWindowCrop`` (via duck-typing on the spec) to snap
        sampled crop lengths to multiples of one patch — guaranteeing the
        encoder/decoder round-trip recovers the original temporal length
        without trimming.
        """
        return self.temporal_compression

    @property
    def num_latent_time_steps(self) -> int:
        """Number of latent time steps at the maximum (eval-time) window.

        Uses ``effective_window_length_seconds`` — i.e. ``Dist.max`` for
        variable-window specs — so the value is always a fixed upper bound
        the bottleneck/decoder code paths can rely on.
        """
        max_bins = round(self.effective_window_length_seconds / self.bin_size_seconds)
        return max_bins // self.temporal_compression

    @property
    def num_tokens_per_sample(self) -> int:
        """Total number of tokens per sample (T_comp * N) at the max window."""
        return self.num_latent_time_steps * self.n_neurons


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


@register_tokenizer
class WanNeuralTokenizer(BaseTokenizer):
    """WanNeural tokenizer: Wan2-inspired convolutional VAE/FSQ for neural data.

    Processes (B, T, N) neural data through:
    1. Optional temporal patchification
    2. Convolutional encoder with downsampling + attention
    3. Latent bottleneck (VAE or FSQ)
    4. Convolutional decoder with upsampling + attention
    5. Output activation
    """

    spec_class = WanNeuralTokenizerSpec
    spec: WanNeuralTokenizerSpec

    def __init__(self, spec: WanNeuralTokenizerSpec):
        super().__init__(spec)

        p = spec.temporal_patch_size

        # Channels after patchification (or 1 if no patching)
        in_channels = p  # patchify folds p time bins into channels

        # --- Bottleneck ---
        if spec.bottleneck_type == "vae":
            self.bottleneck = VAEBottleneck(
                z_dim=spec.z_dim,
                beta=spec.beta,
                curvature_weight=spec.curvature_weight,
                curvature_step_thresh=spec.curvature_step_thresh,
                curvature_metric=spec.curvature_metric,
            )
        elif spec.bottleneck_type == "mmd_vae":
            self.bottleneck = MMDVAEBottleneck(
                z_dim=spec.z_dim,
                mmd_lambda=spec.mmd_lambda,
                kernel_bandwidth_div=spec.mmd_kernel_bandwidth_div,
                n_prior_samples=spec.mmd_n_prior_samples,
                mmd_max_samples=spec.mmd_max_samples,
            )
        elif spec.bottleneck_type == "bsq":
            self.bottleneck = BSQBottleneck(
                z_dim=spec.z_dim,
                codebook_dim=spec.bsq_codebook_dim,
                inv_temperature=spec.bsq_inv_temperature,
                gamma0=spec.bsq_gamma0,
                gamma=spec.bsq_gamma,
                zeta=spec.bsq_zeta,
            )
        elif spec.bottleneck_type == "vqpt":
            self.bottleneck = VQPTBottleneck(
                z_dim=spec.z_dim,
                vq_type=spec.vqpt_type,
                loss_weight=spec.vqpt_loss_weight,
                vq_kwargs=dict(spec.vqpt_kwargs),
            )
        elif spec.bottleneck_type == "diffusion_prior":
            if spec.c_lf <= 0:
                raise ValueError(f"c_lf must be > 0 for diffusion_prior, got {spec.c_lf}")
            self.bottleneck = DiffusionPriorBottleneck(
                z_dim=spec.z_dim,
                lambda_z_min=spec.lambda_z_min,
                lambda_z_max=spec.lambda_z_max,
                prior_dim=spec.prior_dim,
                prior_depth=spec.prior_depth,
                prior_num_heads=spec.prior_num_heads,
            )
        else:
            self.bottleneck = FSQBottleneck(
                z_dim=spec.z_dim,
                levels=spec.q_levels,
            )
        head_dim = self.bottleneck.encoder_out_channels

        # --- Encoder ---
        self.encoder = Encoder2d(
            in_channels=in_channels,
            dim=spec.dim,
            head_dim=head_dim,
            dim_mult=spec.dim_mult,
            num_res_blocks=spec.num_res_blocks,
            attn_levels=spec.attn_levels,
            temporal_downsample=spec.temporal_downsample,
            neuron_kernel_size=spec.neuron_kernel_size,
            temporal_kernel_size=spec.k_t,
            down_temporal_kernel_size=spec.k_t_down,
            temporal_stride=spec.s_t_down,
            k_t_init=spec.k_t_init,
            s_t_init=spec.s_t_init,
            num_attn_heads=spec.num_attn_heads,
            attn_use_ffn=spec.attn_use_ffn,
            attn_ffn_dim_mult=spec.attn_ffn_dim_mult,
            use_stage_shortcut=spec.stage_shortcut,
            use_mid_attn=spec.mid_attn,
            causal=spec.causal,
            dropout=spec.dropout,
            norm_type=spec.norm_type,
            num_groups=spec.num_groups,
            # Temporal attention
            temp_attn_levels=spec.temp_attn_levels,
            temp_num_attn_heads=spec.temp_num_attn_heads,
            use_mid_temp_attn=spec.temp_mid_attn,
            temp_attn_causal=spec.temp_attn_causal,
            temp_attn_use_ffn=spec.temp_attn_use_ffn,
            temp_attn_ffn_dim_mult=spec.temp_attn_ffn_dim_mult,
            temp_attn_use_rope=spec.temp_attn_use_rope,
            bin_size_ms=spec.bin_size_ms,
            token_bin_size_ms=spec.token_bin_size_ms,
            temporal_patch_size=spec.temporal_patch_size,
        )

        # --- Decoder ---
        # Every architecture parameter is independently overridable; each
        # effective_dec_* property falls back to the encoder value so the
        # default is a mirror of the encoder (plus optional dec_dim scaling).
        self.decoder = Decoder2d(
            out_channels=in_channels,  # must produce enough channels for unpatchify
            z_dim=spec.z_dim,
            dim=spec.effective_dec_dim,
            dim_mult=spec.effective_dec_dim_mult,
            num_res_blocks=spec.effective_dec_num_res_blocks,
            attn_levels=spec.effective_dec_attn_levels,
            temporal_upsample=spec.effective_dec_temporal_upsample,
            neuron_kernel_size=spec.effective_dec_neuron_kernel_size,
            temporal_kernel_size=spec.effective_dec_k_t,
            up_temporal_kernel_size=spec.effective_dec_k_t_up,
            temporal_stride=spec.effective_dec_s_t_up,
            k_t_init=spec.effective_dec_k_t_init,
            s_t_init=spec.effective_dec_s_t_init,
            num_attn_heads=spec.effective_dec_num_attn_heads,
            attn_use_ffn=spec.effective_dec_attn_use_ffn,
            attn_ffn_dim_mult=spec.effective_dec_attn_ffn_dim_mult,
            use_stage_shortcut=spec.effective_dec_stage_shortcut,
            use_mid_attn=spec.effective_dec_mid_attn,
            causal=spec.causal,
            dropout=spec.effective_dec_dropout,
            norm_type=spec.norm_type,
            num_groups=spec.num_groups,
            # Temporal attention
            temp_attn_levels=spec.effective_dec_temp_attn_levels,
            temp_num_attn_heads=spec.effective_dec_temp_num_attn_heads,
            use_mid_temp_attn=spec.effective_dec_temp_mid_attn,
            temp_attn_causal=spec.effective_dec_temp_attn_causal,
            temp_attn_use_ffn=spec.effective_dec_temp_attn_use_ffn,
            temp_attn_ffn_dim_mult=spec.effective_dec_temp_attn_ffn_dim_mult,
            temp_attn_use_rope=spec.effective_dec_temp_attn_use_rope,
            bin_size_ms=spec.bin_size_ms,
            token_bin_size_ms=spec.token_bin_size_ms,
            temporal_patch_size=spec.temporal_patch_size,
        )

        # --- Latent normalization (Wan2pt2-style) ---
        if spec.latent_mean is not None and spec.latent_std is not None:
            # Shape (z_dim,) → (1, z_dim, 1, 1) for broadcasting over (B, C, T, N)
            mean = torch.tensor(spec.latent_mean).view(1, -1, 1, 1)
            std = torch.tensor(spec.latent_std).view(1, -1, 1, 1)
            self.register_buffer("latent_mean", mean)
            self.register_buffer("latent_inv_std", 1.0 / std)
        else:
            self.latent_mean = None
            self.latent_inv_std = None

        # --- SIGReg (optional, LDNS-style distribution-matching regularizer) ---
        self.sigreg: SIGReg | None = None
        self._latest_sigreg_loss: torch.Tensor | None = None
        if spec.use_sigreg:
            self.sigreg = SIGReg(
                knots=spec.sigreg_knots,
                t_max=spec.sigreg_t_max,
                sketch_dim=spec.sigreg_sketch_dim,
                seed=spec.sigreg_seed,
            )

        # --- Output activation ---
        if spec.output_activation == "softplus":
            self.output_act: nn.Module = nn.Softplus(beta=spec.output_act_beta)
        elif spec.output_activation == "sigmoid":
            self.output_act = nn.Sigmoid()
        else:
            self.output_act = nn.Identity()

        # --- AR regularizer (per-neuron next-token CE on quantized latents) ---
        if spec.ar_weight > 0:
            self.ar_regularizer: ARRegularizer | None = ARRegularizer(
                embed_dim=spec.z_dim,
                vocab_size=spec.get_vocab_size(),
                num_heads=spec.ar_num_heads,
                horizon=spec.ar_horizon,
                dropout=spec.dropout if spec.ar_dropout is None else spec.ar_dropout,
            )
        else:
            self.ar_regularizer = None

    @property
    def _should_normalize_latent(self) -> bool:
        """Whether to apply latent normalization based on spec."""
        return self.latent_mean is not None and self.latent_inv_std is not None

    def _normalize_latent(self, z: Tensor) -> Tensor:
        """Apply per-channel z-score normalization to latent: (z - mean) / std."""
        if self._should_normalize_latent:
            z = (z - self.latent_mean) * self.latent_inv_std  # type: ignore
        return z

    def _denormalize_latent(self, z: Tensor) -> Tensor:
        """Invert per-channel z-score normalization: z * std + mean."""
        if self._should_normalize_latent:
            z = z / self.latent_inv_std + self.latent_mean  # type: ignore
        return z

    def _compute_sigreg_loss(self, z: Tensor) -> Tensor:
        """Compute the SIGReg distribution-matching loss on pooled latent samples.

        Following LeJEPA / LDNS, all batch / time / neuron positions are
        treated as samples from the embedding distribution and regularized so
        their random projections match an isotropic Gaussian target.

        Args:
            z: Latent tensor of shape (B, z_dim, T, N) — the post-bottleneck
               continuous latent fed to the decoder.

        Returns:
            Scalar SIGReg statistic (unweighted).
        """
        assert self.sigreg is not None
        # (B, C, T, N) → (B*T*N, C). Native ops (not einops) so the forward is
        # traceable under torch.compile + dynamic shapes.
        C = z.shape[1]
        samples = z.permute(0, 2, 3, 1).reshape(-1, C).contiguous()
        return self.sigreg(samples)

    def _maybe_augment_input(self, x: BinnedSpikesInput | RatesInput, data: Tensor) -> Tensor:
        """Apply training-time input augmentations to the encoder input.

        Active only when in training mode, the input is BinnedSpikesInput,
        and at least one augmentation scale is non-zero. The reconstruction
        target on ``x`` is left untouched so the latent must be robust to
        the corruption.

        ``pad_and_gap_mask`` may be reshaped by time-shift augmentation, but
        the per-neuron validity mask reduced from it (``(B, N).any(dim=1)``)
        is robust to the small bin shifts allowed here, so we do not
        re-derive ``neuron_mask`` on the caller's behalf.
        """
        if not (
            self.training
            and isinstance(x, BinnedSpikesInput)
            and self.spec.has_input_augmentations
        ):
            return data
        new_data, _ = apply_input_augmentations(
            data,
            x.pad_and_gap_mask,
            jitter_std_bins=self.spec.input_aug_jitter_std_bins,
            jitter_cap_bins=self.spec.input_aug_jitter_cap_bins,
            spot_noise_p=self.spec.input_aug_spot_noise_p,
            time_shift_std_bins=self.spec.input_aug_time_shift_std_bins,
            time_shift_cap_bins=self.spec.input_aug_time_shift_cap_bins,
        )
        return new_data

    def _build_ar_valid_mask(self, x: BinnedSpikesInput | RatesInput, T_d: int) -> Tensor:
        """Build the per-(b, n) bool mask consumed by ``ARRegularizer``.

        Reduce ``pad_and_gap_mask`` ``(B, T, N)`` over neurons (population-
        shared time validity), patchify by ``temporal_compression``, then
        broadcast across ``N`` so each per-neuron AR sequence carries the same
        time validity. Mirrors CALM's ``_build_time_key_padding_mask``.
        """
        pad = x.pad_and_gap_mask
        B, T, N = pad.shape
        p = self.spec.temporal_compression
        time_mask = pad.bool().any(dim=-1)  # (B, T)
        # Variable-window crops are snapped to multiples of temporal_compression
        # by VariableWindowCrop, so T % p == 0 in every sane config.
        T_use = (T // p) * p
        if T_use != T:
            time_mask = time_mask[:, :T_use]
        patch_mask = time_mask.reshape(B, T_use // p, p).any(dim=-1)  # (B, T // p)
        if patch_mask.shape[1] != T_d:
            patch_mask = patch_mask[:, :T_d]
        return patch_mask.unsqueeze(1).expand(B, N, T_d).reshape(B * N, T_d)

    def _extract_data(self, x: BinnedSpikesInput | RatesInput) -> tuple[Tensor, Tensor]:
        """Extract data tensor and neuron mask from the input.

        Zeroes out invalid positions (padded neurons, temporal gaps) so they
        don't inject signal through convolutions. Also derives a per-neuron
        validity mask for attention masking.

        Args:
            x: Model input (BinnedSpikesInput or RatesInput).

        Returns:
            Tuple of:
                - (B, T, N) float tensor with invalid positions zeroed.
                - (B, N) bool neuron mask (True = valid).
        """
        if isinstance(x, BinnedSpikesInput):
            data = x.spike_matrix.float()
        elif isinstance(x, RatesInput):
            data = x.rate_matrix.float()
        else:
            raise TypeError(f"Unsupported input type: {type(x)}")
        # Zero out padded neurons and temporal gaps
        data = data * x.pad_and_gap_mask.float()

        # Neuron mask: valid if the neuron is valid at any timestep
        # pad_and_gap_mask is (B, T, N) int — reduce over T to get (B, N) bool.
        # Always returned as a tensor (no data-dependent .all() short-circuit)
        # so torch.compile can trace this without a graph break.
        neuron_mask = x.pad_and_gap_mask.any(dim=1)  # (B, N) bool

        return data, neuron_mask

    def _pre_encode(self, data: Tensor) -> Tensor:
        """Prepare data for encoder: add channel dim, patchify.

        Args:
            data: (B, T, N) float tensor.

        Returns:
            (B, C_in, T_patch, N) tensor ready for encoder.
        """
        # (B, T, N) → (B, 1, T, N)
        x = data.unsqueeze(1)

        # Optional temporal patchification
        p = self.spec.temporal_patch_size
        if p > 1:
            # Causal pad T on the left to next multiple of patch_size
            T = x.shape[2]
            pad_t = (p - T % p) % p
            if pad_t > 0:
                x = torch.nn.functional.pad(x, (0, 0, pad_t, 0))
            x = temporal_patchify(x, p)

        return x

    def _post_decode(self, x: Tensor, target_T: int) -> Tensor:
        """Post-process decoder output: unpatchify, squeeze, activate, trim.

        Args:
            x: (B, C_out, T_patch, N) decoder output.
            target_T: Target temporal length.

        Returns:
            (B, T, N) reconstruction.
        """
        p = self.spec.temporal_patch_size
        if p > 1:
            x = temporal_unpatchify(x, p)

        # (B, 1, T, N) → (B, T, N)
        x = x.squeeze(1)
        x = self.output_act(x)

        # Trim causal padding from the beginning if padding was added
        if x.shape[1] > target_T:
            x = x[:, -target_T:, :]

        return x

    def encode(self, x: BinnedSpikesInput | RatesInput) -> Tensor:
        """Encode neural data to latent tokens.

        Returns a 4D tensor of continuous latent codes in the unified
        ``[B, T, U, D]`` format, where ``T`` is the compressed time
        dimension, ``U`` is the neuron dimension, and ``D = z_dim``.

        For discrete bottlenecks (FSQ/BSQ), returns the quantized
        continuous codes (not integer indices). Use ``forward()`` to
        get integer token IDs via ``TokenizerOutput.tokens``.

        Args:
            x: Input neural data (B, T, N).

        Returns:
            (B, T_comp, N, z_dim) continuous latent tensor.
        """
        data, neuron_mask = self._extract_data(x)
        h = self._pre_encode(data)
        h = self.encoder(h, neuron_mask=neuron_mask)

        if isinstance(self.bottleneck, FSQBottleneck | BSQBottleneck | VQPTBottleneck):
            z, _aux = self.bottleneck.encode(h, neuron_mask=neuron_mask)
            z_q = self.bottleneck.decode(z)
        else:
            # VAE / DiffusionPrior — both return a sampled / forward-noised
            # latent of shape (B, z_dim, T_comp, N) plus an aux loss we drop.
            z, _aux = self.bottleneck.encode(h, neuron_mask=neuron_mask)
            z_q = self._normalize_latent(z)

        # (B, C, T, N) → (B, T, N, C) — unified 4D format
        return z_q.permute(0, 2, 3, 1)

    def encode_clean(self, x: BinnedSpikesInput | RatesInput) -> Tensor:
        """Return the deterministic encoder output ``z_clean`` before σ₀ noise.

        Only valid for ``bottleneck_type="diffusion_prior"``. Use this as the
        target for downstream stimulus → latent conditional flow / diffusion
        models — the σ₀ noise the bottleneck adds at training time is what the
        latent diffusion prior absorbs, so downstream models should predict
        z_clean and let the tokenizer re-add σ₀ inside ``decode()`` if desired.

        Returns:
            (B, T_comp, N, z_dim) continuous latent tensor, in the same 4D
            layout as ``encode()``.
        """
        if not isinstance(self.bottleneck, DiffusionPriorBottleneck):
            raise RuntimeError(
                "encode_clean() is only defined for bottleneck_type='diffusion_prior', "
                f"but bottleneck is {type(self.bottleneck).__name__}"
            )
        data, neuron_mask = self._extract_data(x)
        h = self._pre_encode(data)
        h = self.encoder(h, neuron_mask=neuron_mask)
        z_clean = self._normalize_latent(h)
        return z_clean.permute(0, 2, 3, 1)

    def decode(self, tokens: Tensor) -> Tensor:
        """Decode latent tokens back to neural data.

        Args:
            tokens: (B, T, N, D) continuous latent tensor — the 4D format
                    returned by ``encode()``.

        Returns:
            (B, T, N) reconstructed neural data.
        """
        # (B, T, N, D) → (B, D, T, N) — internal conv format
        z = tokens.permute(0, 3, 1, 2)

        if isinstance(
            self.bottleneck, VAEBottleneck | DiffusionPriorBottleneck | MMDVAEBottleneck
        ):
            z = self._denormalize_latent(z)
            z = self.bottleneck.decode(z)

        x = self.decoder(z)
        # Derive target_T from the latent token count so the decoder works for
        # both fixed and variable input windows. For fixed windows this equals
        # ``spec.window_length_bins``; for variable windows it tracks the
        # caller's actual token sequence length.
        target_T = tokens.shape[1] * self.spec.temporal_compression
        return self._post_decode(x, target_T=target_T)

    def reconstruct(self, tokens: Tensor) -> ReconstructionOutput:
        """Decode tokens into spike counts comparable with the input.

        The decoder outputs firing rates (after softplus). This method
        converts them to integer spike counts via either Poisson sampling
        or deterministic rounding, controlled by ``spec.reconstruct_sample``.

        Args:
            tokens: For continuous bottlenecks (VAE), a 4D tensor
                    ``(B, T_comp, N, z_dim)`` — the format returned by
                    ``encode()``. For discrete bottlenecks (BSQ / FSQ /
                    VQPT), an integer index tensor in the layout produced
                    by ``get_indices()`` — typically 2D ``(B, T_comp * N)``,
                    or 3D ``(B, T_comp * N, Q)`` for residual / multi-
                    codebook VQPT variants.

        Returns:
            ReconstructionOutput with (B, T, N) integer spike counts.
        """
        # Continuous-latent path expects float 4D (B, T, N, z_dim). Discrete
        # bottlenecks always emit integer-dtype index tensors at any rank
        # (single-codebook 2D, residual 3D, grouped 4D), so dispatch on dtype
        # rather than rank.
        if (
            isinstance(self.bottleneck, FSQBottleneck | BSQBottleneck | VQPTBottleneck)
            and not tokens.is_floating_point()
        ):
            # Discrete-index path: the index tensor's flat layout is
            # (B, T_comp * N, ...) so we recover T_comp via spec.n_neurons,
            # then unflatten to (B, C, T_comp, N) before the decoder.
            T_comp = tokens.shape[1] // self.spec.n_neurons
            z = self.bottleneck.indices_to_codes(tokens, T=T_comp)
            x = self.decoder(z)
            target_T = T_comp * self.spec.temporal_compression
            rates = self._post_decode(x, target_T=target_T)
        else:
            rates = self.decode(tokens)

        max_count = (
            1 if self.spec.output_type == DataType.SPIKES else self.spec.reconstruct_max_count
        )
        if self.spec.reconstruct_sample == "poisson":
            counts = torch.poisson(rates)
        elif self.spec.reconstruct_sample == "bernoulli":
            # With output_activation="sigmoid" the decoder output already *is* a
            # per-bin spike probability. round() is the per-bin MAP decision, so
            # a model that correctly learns p ~= the 2-4% base rate collapses to
            # an all-zero reconstruction; Bernoulli sampling preserves the rate.
            counts = torch.bernoulli(rates.clamp(0.0, 1.0))
        else:
            counts = rates.round().clamp(min=0)
        if max_count is not None:
            counts = counts.clamp(max=max_count)
        counts = counts.long()

        bin_size_seconds = (
            self.spec.bin_size_ms / 1000.0 if self.spec.bin_size_ms is not False else None
        )
        return ReconstructionOutput(
            reconstruction=counts,
            data_type=DataType.SPIKES,
            bin_size_seconds=bin_size_seconds,
        )

    def reconstruct_from_output(self, output: TokenizerOutput) -> ReconstructionOutput:
        """Convenience method to reconstruct directly from TokenizerOutput."""
        assert output.reconstruction is not None, "TokenizerOutput must have reconstruction field"
        rates: Tensor = output.reconstruction
        # Must match reconstruct()'s cap: binary spikes are clamped to 1, even
        # for an unbounded activation (e.g. softplus).
        max_count = (
            1 if self.spec.output_type == DataType.SPIKES else self.spec.reconstruct_max_count
        )
        if self.spec.reconstruct_sample == "poisson":
            counts = torch.poisson(rates)
        elif self.spec.reconstruct_sample == "bernoulli":
            # See reconstruct(): sigmoid output is a per-bin spike probability.
            counts = torch.bernoulli(rates.clamp(0.0, 1.0))
        else:
            counts = rates.round().clamp(min=0)
        if max_count is not None:
            counts = counts.clamp(max=max_count)
        counts = counts.long()

        bin_size_seconds = (
            self.spec.bin_size_ms / 1000.0 if self.spec.bin_size_ms is not False else None
        )
        return ReconstructionOutput(
            reconstruction=counts,
            data_type=self.spec.output_type,
            bin_size_seconds=bin_size_seconds,
        )

    def _tile_geometry(self, T: int) -> tuple[int, int]:
        """Return (n_tiles, effective_stride) for the tile-batched curvature path.

        - If `n_tiles == 0`, the user's `tile_stride` is used as-is and the
          tile count is the maximum that fits ((T - W) // p + 1).
        - If `n_tiles > 0`, the count is capped and the effective stride is
          adjusted to span the full input ([0, T - W]) with that many tiles —
          guarantees full coverage so the average-untile recon has no
          uncovered bins.
        """
        W = self.spec.tile_window
        p_user = self.spec.tile_stride
        n_user = self.spec.n_tiles
        max_n = (T - W) // p_user + 1
        if n_user <= 0:
            return max_n, p_user
        n = min(n_user, max_n)
        if n < 2:
            return n, 0
        eff_stride = (T - W) // (n - 1)
        return n, eff_stride

    def _tile_indices(self, T: int, n_tiles: int, stride: int, device: torch.device) -> Tensor:
        """Index tensor (n_tiles, tile_window) of tile bin starts + offsets."""
        W = self.spec.tile_window
        starts = stride * torch.arange(n_tiles, device=device)
        return starts[:, None] + torch.arange(W, device=device)  # (n_tiles, W)

    def _build_tiled_input(
        self,
        x: BinnedSpikesInput | RatesInput,
        idx: Tensor,
        n_tiles: int,
    ) -> BinnedSpikesInput | RatesInput:
        """Construct a per-tile copy of `x` so the standard forward / loss path
        can run natively on the tiled batch.

        spike_matrix / rate_matrix and pad_and_gap_mask are gathered along time
        using `idx` (n_tiles, W) — each tile is a slice of the original sequence.
        Per-sample metadata (unit_ids, sampling_rate, session_id, ...) is
        replicated `n_tiles` times so each tile carries its parent's identity.
        Time-extent fields (duration_sec, absolute_start, absolute_end) are
        replicated as-is — they refer to the parent window, not the tile.
        Any downstream code that divides a loss by duration_sec (e.g. for
        per-second normalization) will use the full-window duration on 100ms
        tiles, producing values ~(window / tile)× smaller than the non-tiled path (e.g.
        ~10× for the default 1 s / 100 ms geometry). This is
        a known limitation of the tiled-curvature design; treat these fields as
        upper-bound metadata when inspecting tiled batches.
        """

        def _tile_time_tensor(t: Tensor) -> Tensor:
            # (B, T, N) -> (B, n_tiles, W, N) -> (B*n_tiles, W, N)
            return t[:, idx, :].reshape(t.shape[0] * n_tiles, idx.shape[1], t.shape[2])

        def _replicate_per_sample(v):
            if isinstance(v, Tensor):
                return v.repeat_interleave(n_tiles, dim=0)
            if isinstance(v, tuple):
                return tuple(item for item in v for _ in range(n_tiles))
            return v  # scalars / None pass through unchanged

        if isinstance(x, BinnedSpikesInput):
            return x._replace(
                spike_matrix=_tile_time_tensor(x.spike_matrix),
                pad_and_gap_mask=_tile_time_tensor(x.pad_and_gap_mask),
                sampling_rate=_replicate_per_sample(x.sampling_rate),
                unit_ids=_replicate_per_sample(x.unit_ids),
                unit_ids_pad_mask=_replicate_per_sample(x.unit_ids_pad_mask),
                duration_sec=_replicate_per_sample(x.duration_sec),
                session_id=_replicate_per_sample(x.session_id),
                subject_id=_replicate_per_sample(x.subject_id),
                species=_replicate_per_sample(x.species),
                absolute_start=_replicate_per_sample(x.absolute_start),
                absolute_end=_replicate_per_sample(x.absolute_end),
            )
        if isinstance(x, RatesInput):
            return x._replace(
                rate_matrix=_tile_time_tensor(x.rate_matrix),
                pad_and_gap_mask=_tile_time_tensor(x.pad_and_gap_mask),
                sampling_rate=_replicate_per_sample(x.sampling_rate),
                unit_ids=_replicate_per_sample(x.unit_ids),
                unit_ids_pad_mask=_replicate_per_sample(x.unit_ids_pad_mask),
                duration_sec=_replicate_per_sample(x.duration_sec),
                session_id=_replicate_per_sample(x.session_id),
                subject_id=_replicate_per_sample(x.subject_id),
                species=_replicate_per_sample(x.species),
                absolute_start=_replicate_per_sample(x.absolute_start),
                absolute_end=_replicate_per_sample(x.absolute_end),
            )
        raise TypeError(f"Unsupported input type: {type(x)}")

    def _maybe_tile_curvature_override(
        self,
        z_tiled: Tensor,
        B: int,
        n_tiles: int,
        neuron_mask: Tensor | None,
    ) -> None:
        """Replace the VAE bottleneck's curvature term with one computed across
        the n_tiles dimension of the per-tile latents."""
        if not isinstance(self.bottleneck, VAEBottleneck):
            return
        if self.bottleneck.curvature_weight <= 0:
            return
        if n_tiles < 3:
            return  # need at least 3 frames for v1/v2/cos triplets

        # z_tiled: (B*n_tiles, z_dim, 1, N)  →  (B, z_dim, n_tiles, N)
        z_dim = z_tiled.shape[1]
        N = z_tiled.shape[-1]
        z_dense = z_tiled.squeeze(2).view(B, n_tiles, z_dim, N).permute(0, 2, 1, 3).contiguous()
        # total_curvature is a @staticmethod — no posterior instance needed.
        curv = DiagonalGaussianDistribution.total_curvature(
            z_dense,
            mode=self.bottleneck.curvature_metric,
            step_thresh=self.bottleneck.curvature_step_thresh,
            neuron_mask=neuron_mask,
        )
        self.bottleneck._last_curvature = curv.detach()
        self.bottleneck._curvature_term = self.bottleneck.curvature_weight * curv

    def _tiled_forward(
        self,
        x: BinnedSpikesInput | RatesInput,
    ) -> tuple[Tensor, Tensor, Tensor, BinnedSpikesInput | RatesInput, int]:
        """Build a tiled batch from `x` and run encoder + bottleneck + decoder
        on it natively. The returned `recon` is per-tile — no untiling — and
        the returned `tiled_input` should be surfaced on TokenizerOutput so the
        litmodule can use it in place of the original batch for loss / Pearson.

        Returns (recon, z, z_for_decoder, tiled_input, n_tiles) with:
          - recon shape (B*n_tiles, tile_window, N)
          - z, z_for_decoder shape (B*n_tiles, z_dim, 1, N)
          - tiled_input matches the type of `x` (BinnedSpikesInput / RatesInput)
        """
        # Source data + mask come from the ORIGINAL batch so the curvature
        # override below has the per-original-sample neuron_mask shape (B, N).
        source_data, neuron_mask_orig = self._extract_data(x)
        B_orig, T, _ = source_data.shape
        n_tiles, stride = self._tile_geometry(T)
        idx = self._tile_indices(T, n_tiles, stride, source_data.device)  # (n_tiles, W)

        # Tile the entire input — spike_matrix / rate_matrix, pad_and_gap_mask,
        # and per-sample metadata — so the standard forward path can consume it.
        tiled_input = self._build_tiled_input(x, idx, n_tiles)
        tiled_data, nm_tiled = self._extract_data(tiled_input)
        tiled_data = self._maybe_augment_input(tiled_input, tiled_data)
        target_T = tiled_data.shape[1]  # == tile_window

        h = self._pre_encode(tiled_data)
        h = self.encoder(h, neuron_mask=nm_tiled)
        z, _ = self.bottleneck.encode(h, neuron_mask=nm_tiled)
        z_for_decoder = self.bottleneck.decode(z)
        recon = self.decoder(z_for_decoder, neuron_mask=nm_tiled)
        recon = self._post_decode(recon, target_T=target_T)

        # Cross-tile curvature override (replaces the per-tile T=1 curvature
        # which the bottleneck skipped). Uses the ORIGINAL (B, N) neuron_mask
        # since the curvature reduction broadcasts (B, N) over the n_tiles
        # time axis of the reshaped (B, z_dim, n_tiles, N) latent.
        self._maybe_tile_curvature_override(z, B_orig, n_tiles, neuron_mask_orig)
        return recon, z, z_for_decoder, tiled_input, n_tiles

    def forward(
        self, x: BinnedSpikesInput | RatesInput
    ) -> TokenizerOutput | tuple[BinnedSpikesInput | RatesInput, TokenizerOutput]:
        """Full forward pass: encode → bottleneck → decode.

        Args:
            x: Input neural data.

        Returns:
            TokenizerOutput in the standard (non-tiled) path. When the
            tile-batched curvature path is active (training, tile_stride>0,
            VAE bottleneck), returns a `(tiled_batch, TokenizerOutput)` tuple
            so the litmodule rebinds its `batch` to the tiled copy for the
            downstream loss / metric path (BaseTokenizer.forward docstring).
        """
        use_tiled = (
            self.training
            and self.spec.tile_stride > 0
            and isinstance(self.bottleneck, VAEBottleneck)
        )

        tiled_input: BinnedSpikesInput | RatesInput | None = None
        n_tiles_dense = 0
        bottleneck_loss: Tensor | None
        if use_tiled:
            recon, z, z_for_decoder, tiled_input, n_tiles_dense = self._tiled_forward(x)
            # bottleneck_loss sentinel: VAE always emits at least kl_term
            assert isinstance(self.bottleneck, VAEBottleneck)  # implied by use_tiled
            bottleneck_loss = self.bottleneck._kl_term
        else:
            data, neuron_mask = self._extract_data(x)
            data = self._maybe_augment_input(x, data)
            target_T = data.shape[1]

            # Encode
            h = self._pre_encode(data)
            h = self.encoder(h, neuron_mask=neuron_mask)

            # Bottleneck
            z, bottleneck_loss = self.bottleneck.encode(h, neuron_mask=neuron_mask)
            z_for_decoder = self.bottleneck.decode(z)

            # Decode
            recon = self.decoder(z_for_decoder, neuron_mask=neuron_mask)
            recon = self._post_decode(recon, target_T=target_T)

        # Build output fields
        if isinstance(self.bottleneck, FSQBottleneck | BSQBottleneck | VQPTBottleneck):
            # `use_tiled` is only True when `isinstance(self.bottleneck, VAEBottleneck)`,
            # so this branch is mutually exclusive with the tiled path — `h` is always
            # defined here (set above in the `else` block).
            # For non-residual VQPT variants `tokens` is [B, T_comp * N] like
            # FSQ/BSQ. Residual / grouped variants produce a trailing
            # quantizer dim ([B, T_comp * N, Q] or wider); consumers that need
            # a flat token stream should use the continuous `latents` instead.
            tokens = self.bottleneck.get_indices(h)
            # Continuous quantized codes in 4D format for downstream metrics
            latents = z_for_decoder.permute(0, 2, 3, 1)
        else:
            tokens = None  # VAE has no discrete tokens
            if use_tiled:
                # z is (B'*n_tiles, z_dim, 1, N). Reshape to dense
                # (B, n_tiles, N, z_dim) so the `latents` output reflects the
                # cross-tile temporal latent rather than per-tile (T=1) latents.
                B_orig = z.shape[0] // n_tiles_dense
                z_dim = z.shape[1]
                N = z.shape[-1]
                z_dense = (
                    z.squeeze(2)
                    .view(B_orig, n_tiles_dense, z_dim, N)
                    .permute(0, 2, 1, 3)
                    .contiguous()
                )
                latents = self._normalize_latent(z_dense).permute(0, 2, 3, 1)
            else:
                latents = self._normalize_latent(z).permute(0, 2, 3, 1)

        # AR regularizer: per-neuron next-token CE on quantized latents.
        # Spec validation guarantees `ar_regularizer is not None` implies a
        # single-codebook discrete bottleneck (so `tokens` is 2D and reshaping
        # `b (t n) -> (b n) t` is well-defined). The tiled path is VAE-only and
        # has T_comp=1 per tile, so AR contributes no signal there — skip it.
        ar_loss: Tensor | None = None
        if self.ar_regularizer is not None and self.spec.ar_weight > 0 and not use_tiled:
            assert tokens is not None, (
                "ar_regularizer is not None implies a discrete bottleneck "
                "with 2D indices — `tokens` should always be populated here"
            )
            T_comp = z_for_decoder.shape[2]
            N_lat = z_for_decoder.shape[3]
            B_z, C_z = z_for_decoder.shape[0], z_for_decoder.shape[1]
            # (B, C, T, N) → (B, N, T, C) → (B*N, T, C)
            z_seq = z_for_decoder.permute(0, 3, 2, 1).reshape(B_z * N_lat, T_comp, C_z)
            # (B, T*N) → (B, T, N) → (B, N, T) → (B*N, T)
            idx_bn = (
                tokens.reshape(B_z, T_comp, N_lat)
                .permute(0, 2, 1)
                .reshape(B_z * N_lat, T_comp)
                .long()
            )
            valid_bn = self._build_ar_valid_mask(x, T_comp)
            ar_loss = self.spec.ar_weight * self.ar_regularizer(
                z_seq,
                idx_bn,
                valid_mask=valid_bn,
            )

        # Named auxiliary loss: label by bottleneck type
        aux_loss: dict[str, Tensor] | None
        if bottleneck_loss is not None:
            if isinstance(self.bottleneck, VAEBottleneck):
                kl_term = self.bottleneck._kl_term
                assert kl_term is not None, "VAEBottleneck.encode did not populate _kl_term"
                aux_loss = {"kl": kl_term}
                if self.bottleneck._curvature_term is not None:
                    aux_loss["curvature"] = self.bottleneck._curvature_term
            elif isinstance(self.bottleneck, MMDVAEBottleneck):
                mmd_term = self.bottleneck._mmd_term
                assert mmd_term is not None, "MMDVAEBottleneck.encode did not populate _mmd_term"
                aux_loss = {"mmd": mmd_term}
            elif isinstance(self.bottleneck, BSQBottleneck):
                aux_loss = {"entropy": bottleneck_loss}
            elif isinstance(self.bottleneck, VQPTBottleneck):
                aux_loss = {"vqpt": bottleneck_loss}
            elif isinstance(self.bottleneck, DiffusionPriorBottleneck):
                # The plan formulates the rate-distortion total as
                #   c_lf · recon + (prior + endpoint_kl).
                # Our trainer sums recon + Σ aux_loss, so applying 1/c_lf to
                # the aux terms gives the same optimum (scale-equivalent).
                # The bpd diagnostic on the bottleneck stays unscaled — read it
                # via ``self.bottleneck._last_bpd`` for rate sweeps.
                inv_c_lf = 1.0 / float(self.spec.c_lf)
                p_term = self.bottleneck._prior_term
                e_term = self.bottleneck._endpoint_kl_term
                assert p_term is not None and e_term is not None, (
                    "DiffusionPriorBottleneck.encode did not populate term cache"
                )
                aux_loss = {
                    "prior": inv_c_lf * p_term,
                    "endpoint_kl": inv_c_lf * e_term,
                }
            else:
                aux_loss = {"bottleneck": bottleneck_loss}
        else:
            aux_loss = None

        if ar_loss is not None:
            if aux_loss is None:
                aux_loss = {"ar": ar_loss}
            else:
                aux_loss["ar"] = ar_loss

        # SIGReg distribution-matching regularizer (training only). Targets the
        # post-bottleneck latent fed to the decoder so it works uniformly for
        # vae / diffusion_prior / fsq / bsq / vqpt bottlenecks.
        if self.training and self.sigreg is not None:
            sigreg_loss = self._compute_sigreg_loss(z_for_decoder)
            self._latest_sigreg_loss = sigreg_loss.detach()
            if aux_loss is None:
                aux_loss = {}
            aux_loss["sigreg"] = self.spec.sigreg_beta * sigreg_loss

        output = TokenizerOutput(
            reconstruction=recon,
            tokens=tokens,
            aux_loss=aux_loss,
            latents=latents,
        )
        if tiled_input is not None:
            return tiled_input, output
        return output
