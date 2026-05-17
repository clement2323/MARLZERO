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
    """Regresses a single dense scalar target from trunk features.

    Used for KataGo-style auxiliary tasks (mill_diff, pieces_diff) that
    provide noise-free supervision to shape the shared trunk representation
    alongside the noisy end-of-game value target.

    Args:
        num_channels: Number of channels from the residual trunk.
        num_positions: Number of board positions.
        hidden_size: Hidden size of the intermediate linear layer.
    """

    def __init__(self, num_channels: int, num_positions: int, hidden_size: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(num_channels, 1, kernel_size=1)
        self.bn = nn.BatchNorm1d(1)
        self.fc1 = nn.Linear(num_positions, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return scalar predictions of shape (batch,)."""
        x = F.relu(self.bn(self.conv(x)))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        return self.fc2(x).squeeze(1)


class CategoricalValueHead(nn.Module):
    """Maps trunk features to 3 outcome logits [win, draw, loss] and a scalar.

    The scalar P(win)−P(loss) is returned for MCTS backward-compat (same shape
    as ValueHead). The raw logits are returned for cross-entropy training.

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
        self.fc2 = nn.Linear(hidden_size, 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (scalar, logits).

        Returns:
            scalar: (batch,) — P(win)−P(loss), always in (−1, +1).
            logits: (batch, 3) — raw [win, draw, loss] for cross-entropy.
        """
        x = F.relu(self.bn(self.conv(x)))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        probs = torch.softmax(logits, dim=-1)
        scalar = probs[..., 0] - probs[..., 2]
        return scalar, logits
