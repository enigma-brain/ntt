"""LDNS Autoencoder blocks.

Adapted from the LDNS paper codebase. Contains the AutoEncoder and AutoEncoderBlock
classes for temporal modeling of neural spike data using S4 (Structured State Space)
blocks.

Based on:
Kapoor, J., et al., 2024. "Latent diffusion for neural spiking data."

Adapted from the LDNS reference implementation, used under the MIT license:
https://github.com/mackelab/LDNS (see ``LICENSES/ldns-MIT.txt``).
"""

import torch
import torch.nn as nn
from einops import repeat
from einops.layers.torch import Rearrange

from .s4 import FFTConv


class AutoEncoderBlock(nn.Module):
    """Single block of the LDNS autoencoder.

    Consists of:
    - Time mixer: S4-based FFTConv for temporal modeling
    - Channel mixer: MLP for cross-channel interactions
    - Instance normalization with adaptive layer norm (adaLN)
    - Residual connections

    Args:
        C: Hidden dimension (number of channels)
        L: Sequence length (time bins)
        kernel: Type of kernel to use (default "s4")
        bidirectional: Whether to use bidirectional convolution
        kernel_params: Additional parameters for the kernel
        num_lin_per_mlp: Number of linear layers in the channel mixer MLP
        use_act2: Whether to use activation after channel mixer
    """

    def __init__(
        self,
        C: int,
        L: int,
        kernel: str = "s4",
        bidirectional: bool = True,
        kernel_params: dict | None = None,
        num_lin_per_mlp: int = 2,
        use_act2: bool = False,
    ):
        super().__init__()
        self.C = C
        self.L = L
        self.bidirectional = bidirectional
        self.kernel_params = kernel_params
        self.time_mixer = self._get_time_mixer()
        self.post_tm_scale = nn.Conv1d(
            C, C, 1, bias=True, groups=C, padding="same"
        )  # channel-wise scale for post-act
        self.channel_mixer = self._get_channel_mixer(num_lin_per_mlp=num_lin_per_mlp)
        self.norm1 = nn.InstanceNorm1d(C, affine=False)  # input is [B, C, L]
        self.norm2 = nn.InstanceNorm1d(C, affine=False)  # we will use adaLN
        self.act1 = nn.GELU()
        self.act2 = nn.GELU() if use_act2 else nn.Identity()

        # Adaptive layer norm parameters: 3 for each mixer (shift, scale, gate)
        self.ada_ln = nn.Parameter(torch.zeros(1, C * 6, 1), requires_grad=True)

    @staticmethod
    def affine_op(x_: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Apply affine transformation for adaptive layer norm.

        Args:
            x_: Input tensor [B, C, L] (L = time bins)
            shift: Shift parameter [B, C, 1]
            scale: Scale parameter [B, C, 1]

        Returns:
            Transformed tensor [B, C, L]
        """
        assert len(x_.shape) == len(shift.shape), f"{x_.shape} != {shift.shape}"
        return x_ * (1 + scale) + shift

    def _get_time_mixer(self) -> nn.Module:
        """Create the time mixer (S4-based FFTConv)."""
        return FFTConv(
            self.C,
            bidirectional=self.bidirectional,
            activation=None,
        )

    def _get_channel_mixer(self, num_lin_per_mlp: int = 2) -> nn.Module:
        """Create the channel mixer MLP.

        Args:
            num_lin_per_mlp: Number of linear layers in the MLP

        Returns:
            Sequential MLP module
        """
        layers = [
            Rearrange("b c l -> b l c"),
            nn.Linear(self.C, self.C * 2, bias=False),
        ]
        # Extra linear layers prepended by GELU
        for _ in range(max(num_lin_per_mlp - 2, 0)):
            layers.extend(
                [
                    nn.GELU(),
                    nn.Linear(self.C * 2, self.C * 2, bias=False),
                ]
            )
        layers.extend(
            [
                nn.GELU(),
                nn.Linear(self.C * 2, self.C, bias=False),
            ]
        )
        # Rearrange back to [B, C, L]
        layers.append(Rearrange("b l c -> b c l"))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the block.

        Args:
            x: Input tensor [B, C, L]

        Returns:
            Output tensor [B, C, L]
        """
        y = x  # x is residual stream
        y = self.norm1(y)
        ada_ln = repeat(self.ada_ln, "1 d c -> b d c", b=x.shape[0])
        shift_tm, scale_tm, gate_tm, shift_cm, scale_cm, gate_cm = ada_ln.chunk(6, dim=1)
        y = self.affine_op(y, shift_tm, scale_tm)
        y = self.time_mixer(y)
        y = y[0]  # Get output, not state for FFTConv
        y = x + self.post_tm_scale(self.act1(y))

        x = y  # x is again residual stream from last layer
        y = self.norm2(y)
        y = self.affine_op(y, shift_cm, scale_cm)
        y = x + self.act2(self.channel_mixer(y))
        return y


class AutoEncoder(nn.Module):
    """LDNS Autoencoder for neural spike data.

    Uses S4-based blocks for temporal modeling with a bottleneck latent space.

    Architecture:
    - encoder_in: Conv1d projection from input channels to hidden dim
    - encoder: Stack of AutoEncoderBlocks
    - bottleneck: Conv1d projection to latent space
    - unbottleneck: Conv1d projection from latent space
    - decoder: Stack of AutoEncoderBlocks (or just activation if num_blocks_decoder=0)
    - decoder_out: Conv1d projection to output channels

    Args:
        C_in: Number of input channels (neurons)
        C: Hidden dimension
        C_latent: Latent space dimension
        L: Sequence length (time bins)
        kernel: Type of kernel to use (default "s4")
        bidirectional: Whether to use bidirectional convolution (global default)
        kernel_params: Additional parameters for the kernel
        in_groups: Groups for input/output convolutions (auto-computed if None)
        bottleneck_groups: Groups for bottleneck convolutions (auto-computed if None)
        num_blocks: Number of encoder blocks
        num_blocks_decoder: Number of decoder blocks (0 = just activation)
        num_lin_per_mlp: Number of linear layers per MLP in blocks
        use_act_bottleneck: Whether to use activation in bottleneck
        encoder_bidirectional: Override bidirectional for encoder (None → use bidirectional)
        decoder_bidirectional: Override bidirectional for decoder (None → use bidirectional)
    """

    def __init__(
        self,
        C_in: int,
        C: int,
        C_latent: int,
        L: int,
        kernel: str = "s4",
        bidirectional: bool = True,
        kernel_params: dict | None = None,
        in_groups: int | None = None,
        bottleneck_groups: int | None = None,
        num_blocks: int = 4,
        num_blocks_decoder: int | None = None,
        num_lin_per_mlp: int = 2,
        use_act_bottleneck: bool = False,
        encoder_bidirectional: bool | None = None,
        decoder_bidirectional: bool | None = None,
    ):
        super().__init__()
        self.C_in = C_in
        self.C = C
        self.C_latent = C_latent
        self.L = L
        self.bidirectional = bidirectional
        self.kernel_params = kernel_params

        # Resolve per-component bidirectional flags
        enc_bidir = encoder_bidirectional if encoder_bidirectional is not None else bidirectional
        dec_bidir = decoder_bidirectional if decoder_bidirectional is not None else bidirectional

        # Auto-compute groups for grouped convolutions
        if in_groups is None:
            if C % C_in == 0 and C > C_in:
                in_groups = C_in
            else:
                in_groups = 1

        if bottleneck_groups is None:
            if C % C_latent == 0 and C > C_latent:
                bottleneck_groups = C_latent
            else:
                bottleneck_groups = 1

        # Input projection
        self.encoder_in = nn.Conv1d(C_in, C, 1)

        # Encoder blocks
        self.encoder = nn.ModuleList(
            [
                AutoEncoderBlock(
                    C,
                    L,
                    kernel,
                    enc_bidir,
                    kernel_params,
                    num_lin_per_mlp=num_lin_per_mlp,
                )
                for _ in range(num_blocks)
            ]
        )

        # Bottleneck
        self.bottleneck = nn.Conv1d(C, C_latent, 1, groups=bottleneck_groups)
        self.act_bottleneck = nn.GELU() if use_act_bottleneck else nn.Identity()
        self.unbottleneck = nn.Conv1d(C_latent, C, 1, groups=bottleneck_groups)

        # Decoder
        if num_blocks_decoder == 0:
            self.decoder = nn.ModuleList([nn.GELU()])  # Just the activation
        else:
            num_decoder = num_blocks if num_blocks_decoder is None else num_blocks_decoder
            self.decoder = nn.ModuleList(
                [
                    AutoEncoderBlock(
                        C,
                        L,
                        kernel,
                        dec_bidir,
                        kernel_params,
                        num_lin_per_mlp=num_lin_per_mlp,
                    )
                    for _ in range(num_decoder)
                ]
            )

        # Output projection
        self.decoder_out = nn.Conv1d(C, C_in, 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation.

        Args:
            x: Input tensor [B, C_in, L] (C_in = neurons, L = time bins)

        Returns:
            Latent tensor [B, C_latent, L] (L = time bins)
        """
        z = self.encoder_in(x)
        for block in self.encoder:
            z = block(z)
        z = self.bottleneck(z)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to output.

        Args:
            z: Latent tensor [B, C_latent, L] (L = time bins)

        Returns:
            Output tensor [B, C_in, L] (C_in = neurons, L = time bins)
        """
        xhat = self.act_bottleneck(z)
        xhat = self.unbottleneck(xhat)
        for block in self.decoder:
            xhat = block(xhat)
        xhat = self.decoder_out(xhat)
        return xhat

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass.

        Args:
            x: Input tensor [B, C_in, L] (C_in = neurons, L = time bins)

        Returns:
            Tuple of (output, latent) tensors
        """
        z = self.encode(x)
        xhat = self.decode(z)
        return xhat, z
