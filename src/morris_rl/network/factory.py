"""Network factory — instantiate a network from a Hydra config node."""

from __future__ import annotations

import torch.nn as nn
from omegaconf import DictConfig

from morris_rl.env.board import ACTION_SPACE_SIZE as _MORRIS_ACTION_SPACE_SIZE
from morris_rl.env.board import NUM_POSITIONS as _MORRIS_NUM_POSITIONS
from morris_rl.network.resnet import MorrisResNet

_DEFAULT_NUM_INPUT_PLANES = 7                          # Morris encoding
_DEFAULT_NUM_POSITIONS = _MORRIS_NUM_POSITIONS         # 24
_DEFAULT_ACTION_SPACE_SIZE = _MORRIS_ACTION_SPACE_SIZE # 88 (24 + 64 movement edges)


def build_network(
    config: DictConfig,
    num_planes: int = _DEFAULT_NUM_INPUT_PLANES,
    num_positions: int = _DEFAULT_NUM_POSITIONS,
    action_space_size: int = _DEFAULT_ACTION_SPACE_SIZE,
) -> nn.Module:
    """Return a network instance configured from *config.network*.

    Args:
        config: Top-level Hydra config containing a ``network`` sub-node.
        num_planes: Number of input planes. Defaults to 7 (Morris).
        num_positions: Board positions. Defaults to 24 (Morris).
        action_space_size: Total actions. Defaults to 600 (Morris).

    Returns:
        An ``nn.Module`` ready for training or inference.

    Raises:
        ValueError: If ``config.network.type`` is not recognised.
    """
    net_cfg = config.network
    if net_cfg.type == "resnet":
        value_head_type: str = net_cfg.get("value_head_type", "scalar")
        aux_cfg = config.get("aux_heads", {}) or {}
        aux_enabled = bool(aux_cfg.get("enabled", False))
        aux_hidden = int(aux_cfg.get("hidden_size", 64))
        return MorrisResNet(
            num_blocks=net_cfg.num_blocks,
            num_channels=net_cfg.num_channels,
            num_planes=num_planes,
            policy_head_hidden=net_cfg.policy_head_hidden,
            value_head_hidden=net_cfg.value_head_hidden,
            value_head_type=value_head_type,
            num_positions=num_positions,
            action_space_size=action_space_size,
            aux_heads_enabled=aux_enabled,
            aux_head_hidden=aux_hidden,
        )
    raise ValueError(f"Unknown network type: {net_cfg.type!r}")
