"""CountWrapper module for LDNS tokenizer.

Wraps the autoencoder to ensure output firing rates are non-negative
via the softplus activation function.

Adapted from the LDNS reference implementation, used under the MIT license:
https://github.com/mackelab/LDNS (see ``LICENSES/ldns-MIT.txt``).
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class CountWrapper(nn.Module):
    """Wrapper module for converting autoencoder outputs to firing rates.

    Applies softplus to ensure non-negative rates, which is required for
    the Poisson distribution used in spike modeling.

    Args:
        ae_net: The autoencoder network to wrap
    """

    def __init__(self, ae_net: nn.Module):
        super().__init__()
        self.ae_net: Any = ae_net

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through wrapped autoencoder.

        Args:
            x: Input tensor [B, C_in, L] (spike counts/rates)

        Returns:
            Tuple of:
                - rates: Non-negative firing rates [B, C_in, L]
                - z: Latent representation [B, C_latent, L]
        """
        logrates, z = self.ae_net(x)
        return F.softplus(logrates), z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation.

        Args:
            x: Input tensor [B, C_in, L]

        Returns:
            Latent tensor [B, C_latent, L]
        """
        return self.ae_net.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to non-negative rates.

        Args:
            z: Latent tensor [B, C_latent, L]

        Returns:
            Non-negative firing rates [B, C_in, L]
        """
        return F.softplus(self.ae_net.decode(z))
