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
        # Auxiliary heads are opt-in. The trainer's compute_loss treats their
        # output dict as authoritative — heads not in the network's ModuleDict
        # simply contribute zero loss. Resolve to a plain dict here so the
        # network constructor doesn't see Hydra OmegaConf objects.
        aux_cfg_node = net_cfg.get("aux_heads", None)
        aux_heads_config: dict | None = None
        if aux_cfg_node is not None and bool(aux_cfg_node.get("enabled", False)):
            from omegaconf import OmegaConf
            aux_heads_config = OmegaConf.to_container(aux_cfg_node, resolve=True)  # type: ignore[assignment]

        return MorrisResNet(
            num_blocks=net_cfg.num_blocks,
            num_channels=net_cfg.num_channels,
            num_planes=_NUM_INPUT_PLANES,
            policy_head_hidden=net_cfg.policy_head_hidden,
            value_head_hidden=net_cfg.value_head_hidden,
            aux_heads_config=aux_heads_config,
        )
    raise ValueError(f"Unknown network type: {net_cfg.type!r}")
