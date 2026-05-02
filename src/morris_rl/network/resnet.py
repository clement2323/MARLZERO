"""ResNet trunk for the Morris RL policy/value network."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from morris_rl.env.board import ACTION_SPACE_SIZE, NUM_POSITIONS
from morris_rl.network.heads import AuxScalarHead, PolicyHead, ValueHead

# The names of the auxiliary heads recognised by MorrisResNet. Each entry in
# the aux_heads_config dict must use one of these keys; unknown keys are
# silently ignored. Keep this list in sync with the head construction logic
# below and the trainer's loss computation.
AUX_HEAD_NAMES = ("mill_count", "pieces_diff_at_end", "capture_in_n")


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
    Output: (log_policy, value) by default; (log_policy, value, aux_dict)
            when ``forward(..., return_aux=True)`` and aux heads are enabled.

    Args:
        num_blocks: Number of residual blocks in the trunk.
        num_channels: Channel width throughout the trunk.
        num_planes: Number of input feature planes (7 by default).
        policy_head_hidden: Hidden size for the policy head linear layer.
        value_head_hidden: Hidden size for the value head linear layer.
        aux_heads_config: Optional dict gating which auxiliary heads are
            instantiated. Shape::

                {
                    "mill_count":         {"enabled": bool, ...},
                    "pieces_diff_at_end": {"enabled": bool, ...},
                    "capture_in_n":       {"enabled": bool, ...},
                }

            Heads with ``enabled=False`` (or absent) are not constructed,
            so they cost zero parameters and zero forward-time compute.
            See :data:`AUX_HEAD_NAMES` for the recognised keys.
    """

    def __init__(
        self,
        num_blocks: int,
        num_channels: int,
        num_planes: int,
        policy_head_hidden: int,
        value_head_hidden: int,
        aux_heads_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.input_conv = nn.Conv1d(num_planes, num_channels, kernel_size=3, padding=1)
        self.input_bn = nn.BatchNorm1d(num_channels)
        self.trunk = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)])
        self.policy_head = PolicyHead(
            num_channels, NUM_POSITIONS, ACTION_SPACE_SIZE, policy_head_hidden
        )
        self.value_head = ValueHead(num_channels, NUM_POSITIONS, value_head_hidden)

        # Auxiliary heads: each one is opt-in, none by default. Stored in a
        # ModuleDict keyed by name so the trainer can iterate them generically.
        self.aux_heads = nn.ModuleDict()
        if aux_heads_config:
            for name in AUX_HEAD_NAMES:
                spec = aux_heads_config.get(name)
                if spec is not None and spec.get("enabled", False):
                    self.aux_heads[name] = AuxScalarHead(
                        num_channels=num_channels,
                        num_positions=NUM_POSITIONS,
                        hidden_size=32,
                    )

    def forward(
        self,
        x: torch.Tensor,
        legal_mask: torch.Tensor,
        return_aux: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]
    ):
        """Run a forward pass.

        Args:
            x: Encoded state tensor of shape (batch, num_planes, NUM_POSITIONS).
            legal_mask: Boolean mask of shape (batch, ACTION_SPACE_SIZE).
                        True indicates a legal action.
            return_aux: If True, also return a dict mapping aux head name →
                        scalar output tensor of shape (batch,). Defaults to
                        False so existing inference / MCTS callers keep their
                        2-tuple contract.

        Returns:
            (log_policy, value)            if return_aux=False
            (log_policy, value, aux_dict)  if return_aux=True
        """
        x = F.relu(self.input_bn(self.input_conv(x)))
        x = self.trunk(x)
        log_policy = self.policy_head(x, legal_mask)
        value = self.value_head(x)
        if not return_aux:
            return log_policy, value
        aux_outputs = {name: head(x) for name, head in self.aux_heads.items()}
        return log_policy, value, aux_outputs
