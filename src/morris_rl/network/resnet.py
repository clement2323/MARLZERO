"""ResNet trunk for the Morris RL policy/value network."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from morris_rl.env.board import ACTION_SPACE_SIZE as _DEFAULT_ACTION_SPACE_SIZE
from morris_rl.env.board import NUM_POSITIONS as _DEFAULT_NUM_POSITIONS
from morris_rl.network.heads import CategoricalValueHead, PolicyHead, ValueHead


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
    Output: (log_policy, scalar) by default; (log_policy, scalar, logits)
            when ``forward(..., return_value_logits=True)``.

    Args:
        num_blocks: Number of residual blocks in the trunk.
        num_channels: Channel width throughout the trunk.
        num_planes: Number of input feature planes (7 by default).
        policy_head_hidden: Hidden size for the policy head linear layer.
        value_head_hidden: Hidden size for the value head linear layer.
        value_head_type: ``"scalar"`` (tanh + MSE) or ``"categorical"``
            (3-class cross-entropy, KataGo-style).
    """

    def __init__(
        self,
        num_blocks: int,
        num_channels: int,
        num_planes: int,
        policy_head_hidden: int,
        value_head_hidden: int,
        value_head_type: str = "scalar",
        num_positions: int = _DEFAULT_NUM_POSITIONS,
        action_space_size: int = _DEFAULT_ACTION_SPACE_SIZE,
    ) -> None:
        super().__init__()
        self.input_conv = nn.Conv1d(num_planes, num_channels, kernel_size=3, padding=1)
        self.input_bn = nn.BatchNorm1d(num_channels)
        self.trunk = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)])
        self.policy_head = PolicyHead(
            num_channels, num_positions, action_space_size, policy_head_hidden
        )
        self._value_head_type = value_head_type
        if value_head_type == "categorical":
            self.value_head: ValueHead | CategoricalValueHead = CategoricalValueHead(
                num_channels, num_positions, value_head_hidden
            )
        else:
            self.value_head = ValueHead(num_channels, num_positions, value_head_hidden)

    def add_lora_adapters(self, rank: int = 8, alpha: float = 16.0) -> None:
        """Replace all Linear layers with LoRALinear adapters.

        Call this AFTER loading a checkpoint and BEFORE freeze_trunk().
        The base network weights are preserved and frozen inside each
        LoRALinear; only the low-rank A and B matrices will be trained.

        Args:
            rank:  Bottleneck dimension of each low-rank update.
            alpha: LoRA scaling factor (scale = alpha / rank).
        """
        from morris_rl.network.lora import LoRALinear

        for name, module in list(self.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            # Navigate to the parent module so we can replace the attribute.
            parts = name.split(".")
            parent = self
            for part in parts[:-1]:
                parent = getattr(parent, part)
            attr = parts[-1]
            setattr(parent, attr, LoRALinear(module, rank=rank, alpha=alpha))

    def freeze_trunk(self) -> None:
        """Freeze all parameters that are not LoRA adapter weights.

        After this call, only lora_A and lora_B parameters will have
        requires_grad=True. The Trainer's optimizer must be (re-)created
        afterward so it only iterates over the trainable parameters —
        passing ``filter(lambda p: p.requires_grad, network.parameters())``
        is sufficient (already the default in Trainer.__init__).
        """
        for name, param in self.named_parameters():
            if "lora_A" not in name and "lora_B" not in name:
                param.requires_grad_(False)

    def forward(
        self,
        x: torch.Tensor,
        legal_mask: torch.Tensor,
        return_value_logits: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Run a forward pass.

        Args:
            x: Encoded state tensor of shape (batch, num_planes, NUM_POSITIONS).
            legal_mask: Boolean mask of shape (batch, ACTION_SPACE_SIZE).
            return_value_logits: If True, also return raw value logits (batch, 3).
                                 Only meaningful when value_head_type="categorical".

        Returns:
            ``(log_policy, scalar)`` by default.
            ``(log_policy, scalar, logits)`` when ``return_value_logits=True``.
        """
        x = F.relu(self.input_bn(self.input_conv(x)))
        x = self.trunk(x)
        log_policy = self.policy_head(x, legal_mask)

        if self._value_head_type == "categorical":
            scalar, logits = self.value_head(x)  # type: ignore[misc]
        else:
            scalar = self.value_head(x)
            logits = None

        if return_value_logits:
            return log_policy, scalar, logits  # type: ignore[return-value]
        return log_policy, scalar
