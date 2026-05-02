"""Policy and value heads for the Morris ResNet."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyHead(nn.Module):
    """Maps trunk features to a masked log-probability distribution over actions.

    Args:
        num_channels: Number of channels from the residual trunk.
        num_positions: Number of board positions (24).
        action_space_size: Total number of actions (600).
        hidden_size: Size of the intermediate linear layer.
    """

    def __init__(
        self,
        num_channels: int,
        num_positions: int,
        action_space_size: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(num_channels, 2, kernel_size=1)
        self.bn = nn.BatchNorm1d(2)
        self.fc1 = nn.Linear(2 * num_positions, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_space_size)

    def forward(self, x: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        """Return masked log-probabilities over actions.

        Args:
            x: Trunk features of shape (batch, num_channels, num_positions).
            legal_mask: Boolean tensor of shape (batch, action_space_size).
                        True = legal action. Illegal actions receive -inf logit.

        Returns:
            Log-probabilities of shape (batch, action_space_size).
        """
        x = F.relu(self.bn(self.conv(x)))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        logits = logits.masked_fill(~legal_mask, float("-inf"))
        return F.log_softmax(logits, dim=1)


class ValueHead(nn.Module):
    """Maps trunk features to a scalar value estimate in [-1, 1].

    Args:
        num_channels: Number of channels from the residual trunk.
        num_positions: Number of board positions (24).
        hidden_size: Size of the intermediate linear layer.
    """

    def __init__(self, num_channels: int, num_positions: int, hidden_size: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(num_channels, 1, kernel_size=1)
        self.bn = nn.BatchNorm1d(1)
        self.fc1 = nn.Linear(num_positions, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return value estimates.

        Args:
            x: Trunk features of shape (batch, num_channels, num_positions).

        Returns:
            Value tensor of shape (batch,) with values in [-1, 1].
        """
        x = F.relu(self.bn(self.conv(x)))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        return torch.tanh(self.fc2(x)).squeeze(1)


class AuxScalarHead(nn.Module):
    """Generic scalar auxiliary head — raw output, no activation.

    Used for regression (mill_count, pieces_diff_at_end → MSE loss) and for
    binary classification (capture_in_n → BCEWithLogitsLoss). The caller is
    responsible for picking the appropriate loss for the target type, since
    the head itself is task-agnostic.

    Mirrors the ValueHead architecture (single-channel projection → flatten →
    hidden → scalar) which is both small and well-conditioned. Hidden size is
    configurable; we default to 32 to keep these heads cheap (~2k params each).
    """

    def __init__(
        self,
        num_channels: int,
        num_positions: int,
        hidden_size: int = 32,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(num_channels, 1, kernel_size=1)
        self.bn = nn.BatchNorm1d(1)
        self.fc1 = nn.Linear(num_positions, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw scalar output of shape (batch,)."""
        x = F.relu(self.bn(self.conv(x)))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x).squeeze(1)
