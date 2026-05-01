"""Network factory — instantiate a network from a Hydra config node."""

from __future__ import annotations

import torch.nn as nn
from omegaconf import DictConfig

from morris_rl.network.resnet import MorrisResNet

_NUM_INPUT_PLANES = 7  # defined by the encoding scheme (see mcts/search.py:encode_state)


def build_network(config: DictConfig) -> nn.Module:
    """Return a network instance configured from *config.network*.

    Args:
        config: Top-level Hydra config containing a ``network`` sub-node.

    Returns:
        An ``nn.Module`` ready for training or inference.

    Raises:
        ValueError: If ``config.network.type`` is not recognised.
    """
    net_cfg = config.network
    if net_cfg.type == "resnet":
        return MorrisResNet(
            num_blocks=net_cfg.num_blocks,
            num_channels=net_cfg.num_channels,
            num_planes=_NUM_INPUT_PLANES,
            policy_head_hidden=net_cfg.policy_head_hidden,
            value_head_hidden=net_cfg.value_head_hidden,
        )
    raise ValueError(f"Unknown network type: {net_cfg.type!r}")
