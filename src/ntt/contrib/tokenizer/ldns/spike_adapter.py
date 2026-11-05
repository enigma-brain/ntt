"""Spike History Adapter for LDNS tokenizer.

Implements the history-dependent spike generation model from the LDNS paper.
This is used in Stage 2 training to learn spike autocorrelation structure.

Adapted from the LDNS reference implementation, used under the MIT license:
https://github.com/mackelab/LDNS (see ``LICENSES/ldns-MIT.txt``).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpikeHistoryAdapter(nn.Module):
    """MLP adapter for history-dependent spike generation.

    Uses spike history and log firing rates to predict spike counts.
    This model learns the spike autocorrelation structure, improving
    temporal statistics of generated spikes compared to simple Poisson sampling.

    The model performs per-neuron logistic regression:
        adapted_rate = softplus(sum(weights * [history, log_rate, bias]))

    Args:
        n_neurons: Number of neurons
        history_length: Number of past timesteps to use as history
    """

    _is_trained_flag: torch.Tensor

    def __init__(self, n_neurons: int, history_length: int = 20):
        super().__init__()
        self.n_neurons = n_neurons
        self.history_length = history_length

        # Per-neuron weights: [history_length spikes + 1 log_rate + 1 bias] -> 1
        # Random initialization matches original CombinedLogisiticRegression
        self.weights = nn.Parameter(torch.randn(n_neurons, history_length + 2))

        # Persistent flag for checkpoint save/load
        self.register_buffer("_is_trained_flag", torch.tensor(False, dtype=torch.bool))

    @property
    def is_trained(self) -> bool:
        """Check if the spike adapter has been trained."""
        return bool(self._is_trained_flag.item())

    @is_trained.setter
    def is_trained(self, value: bool) -> None:
        self._is_trained_flag.fill_(value)

    def forward(self, spike_history: torch.Tensor, log_rates: torch.Tensor) -> torch.Tensor:
        """Compute adapted firing rates based on spike history.

        Args:
            spike_history: Past spike counts [B, N, history_length]
            log_rates: Log of current inferred firing rates [B, N]

        Returns:
            Adapted firing rates [B, N]
        """
        # Add bias term (always 1)
        bias = torch.ones_like(log_rates)

        # Concatenate: [history, log_rate, bias]
        # spike_history: (B, N, history_length)
        # log_rates: (B, N) -> (B, N, 1)
        # bias: (B, N) -> (B, N, 1)
        x = torch.cat(
            [spike_history, log_rates.unsqueeze(-1), bias.unsqueeze(-1)], dim=-1
        )  # (B, N, history_length + 2)

        # Per-neuron linear combination with softplus output
        # weights: (N, history_length + 2)
        # Output: softplus(sum over last dim of x * weights)
        return F.softplus((x * self.weights).sum(dim=-1))  # (B, N)

    def compute_loss(
        self,
        spike_history: torch.Tensor,
        log_rates: torch.Tensor,
        target_spikes: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Poisson NLL loss for spike prediction.

        Args:
            spike_history: Past spike counts [B, N, history_length]
            log_rates: Log of current inferred firing rates [B, N]
            target_spikes: Actual spike counts at current timestep [B, N]

        Returns:
            Poisson NLL loss (scalar)
        """
        adapted_rates = self.forward(spike_history, log_rates)
        # Poisson NLL: -target * log(rate) + rate + log(target!)
        loss = F.poisson_nll_loss(
            adapted_rates,
            target_spikes.float(),
            log_input=False,
            full=True,
            reduction="mean",
        )
        return loss


def extract_timewise_histories(
    spikes: torch.Tensor,
    log_rates: torch.Tensor,
    history_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract training data for spike history adapter.

    Given a sequence of spikes and log-rates, extract windows of spike history
    and corresponding target spikes for training the adapter.

    Args:
        spikes: Spike counts [B, N, T] in LDNS format
        log_rates: Log firing rates [B, N, T] in LDNS format
        history_length: Number of past timesteps to use

    Returns:
        Tuple of:
            - spike_history: [B * (T - history_length), N, history_length]
            - current_log_rates: [B * (T - history_length), N]
            - target_spikes: [B * (T - history_length), N]
    """
    B, N, T = spikes.shape

    histories = []
    current_log_rates = []
    targets = []

    for t in range(history_length, T):
        # History: spikes from t-history_length to t-1
        history = spikes[:, :, t - history_length : t]  # (B, N, history_length)
        histories.append(history)

        # Current log-rate at timestep t
        log_rate = log_rates[:, :, t]  # (B, N)
        current_log_rates.append(log_rate)

        # Target spike at timestep t
        target = spikes[:, :, t]  # (B, N)
        targets.append(target)

    # Stack along batch dimension
    spike_history = torch.cat(histories, dim=0)  # (B * (T - history_length), N, history_length)
    current_log_rates = torch.cat(current_log_rates, dim=0)  # (B * (T - history_length), N)
    target_spikes = torch.cat(targets, dim=0)  # (B * (T - history_length), N)

    return spike_history, current_log_rates, target_spikes
