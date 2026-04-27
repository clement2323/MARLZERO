"""ResNet trunk for the Morris RL policy/value network."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from morris_rl.env.board import ACTION_SPACE_SIZE, NUM_POSITIONS
from morris_rl.network.heads import PolicyHead, ValueHead


class ResidualBlock(nn.Module):
    """Standard pre-activation residual block with 1D convolutions."""

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(num_channels)
        self.conv2 = nn.Conv1d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class MorrisResNet(nn.Module):
    """AlphaZero-style ResNet for Nine Men's Morris.

    Input:  (batch, num_planes, NUM_POSITIONS)  — encoded board state
    Output: (log_policy, value) where
            log_policy has shape (batch, ACTION_SPACE_SIZE)
            value has shape (batch,)

    Args:
        num_blocks: Number of residual blocks in the trunk.
        num_channels: Channel width throughout the trunk.
        num_planes: Number of input feature planes (8 by default).
        policy_head_hidden: Hidden size for the policy head linear layer.
        value_head_hidden: Hidden size for the value head linear layer.
    """

    def __init__(
        self,
        num_blocks: int,
        num_channels: int,
        num_planes: int,
        policy_head_hidden: int,
        value_head_hidden: int,
    ) -> None:
        super().__init__()
        self.input_conv = nn.Conv1d(num_planes, num_channels, kernel_size=3, padding=1)
        self.input_bn = nn.BatchNorm1d(num_channels)
        self.trunk = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)])
        self.policy_head = PolicyHead(
            num_channels, NUM_POSITIONS, ACTION_SPACE_SIZE, policy_head_hidden
        )
        self.value_head = ValueHead(num_channels, NUM_POSITIONS, value_head_hidden)

    def forward(
        self, x: torch.Tensor, legal_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a forward pass.

        Args:
            x: Encoded state tensor of shape (batch, num_planes, NUM_POSITIONS).
            legal_mask: Boolean mask of shape (batch, ACTION_SPACE_SIZE).
                        True indicates a legal action.

        Returns:
            Tuple of (log_policy, value):
                log_policy: (batch, ACTION_SPACE_SIZE) — masked log-probabilities
                value: (batch,) — scalar estimates in [-1, 1]
        """
        x = F.relu(self.input_bn(self.input_conv(x)))
        x = self.trunk(x)
        log_policy = self.policy_head(x, legal_mask)
        value = self.value_head(x)
        return log_policy, value
