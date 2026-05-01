"""Tests for network architecture — output shapes, action masking, factory."""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from morris_rl.env.board import ACTION_SPACE_SIZE, NUM_POSITIONS
from morris_rl.network.factory import build_network
from morris_rl.network.resnet import MorrisResNet

NUM_PLANES = 7
BATCH = 4


@pytest.fixture()
def small_net() -> MorrisResNet:
    """Tiny network (2 blocks, 16 channels) for fast CPU tests."""
    return MorrisResNet(
        num_blocks=2,
        num_channels=16,
        num_planes=NUM_PLANES,
        policy_head_hidden=32,
        value_head_hidden=32,
    )


@pytest.fixture()
def random_input() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(BATCH, NUM_PLANES, NUM_POSITIONS)
    mask = torch.ones(BATCH, ACTION_SPACE_SIZE, dtype=torch.bool)
    return x, mask


# ---------------------------------------------------------------------------
# Output shapes
# ---------------------------------------------------------------------------


def test_policy_output_shape(
    small_net: MorrisResNet, random_input: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, mask = random_input
    log_policy, _ = small_net(x, mask)
    assert log_policy.shape == (BATCH, ACTION_SPACE_SIZE)


def test_value_output_shape(
    small_net: MorrisResNet, random_input: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, mask = random_input
    _, value = small_net(x, mask)
    assert value.shape == (BATCH,)


def test_value_in_range(
    small_net: MorrisResNet, random_input: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, mask = random_input
    _, value = small_net(x, mask)
    assert (value >= -1.0).all() and (value <= 1.0).all()


def test_log_policy_is_log_prob(
    small_net: MorrisResNet, random_input: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, mask = random_input
    log_policy, _ = small_net(x, mask)
    # exp(log_prob) sums to ~1 over legal actions
    assert log_policy.exp().sum(dim=1).allclose(torch.ones(BATCH), atol=1e-5)


# ---------------------------------------------------------------------------
# Action masking
# ---------------------------------------------------------------------------


def test_masked_actions_have_zero_probability(small_net: MorrisResNet) -> None:
    x = torch.randn(1, NUM_PLANES, NUM_POSITIONS)
    # Only the first 10 actions are legal.
    mask = torch.zeros(1, ACTION_SPACE_SIZE, dtype=torch.bool)
    mask[0, :10] = True
    log_policy, _ = small_net(x, mask)
    probs = log_policy.exp()
    assert (probs[0, 10:] == 0.0).all()
    assert probs[0, :10].sum().allclose(torch.tensor(1.0), atol=1e-5)


def test_masking_single_legal_action(small_net: MorrisResNet) -> None:
    x = torch.randn(1, NUM_PLANES, NUM_POSITIONS)
    mask = torch.zeros(1, ACTION_SPACE_SIZE, dtype=torch.bool)
    mask[0, 42] = True
    log_policy, _ = small_net(x, mask)
    probs = log_policy.exp()
    assert probs[0, 42].item() == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_gradients_flow_to_all_parameters(
    small_net: MorrisResNet, random_input: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, mask = random_input
    log_policy, value = small_net(x, mask)
    loss = -log_policy.mean() + value.mean()
    loss.backward()
    for name, param in small_net.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


# ---------------------------------------------------------------------------
# Input immutability
# ---------------------------------------------------------------------------


def test_forward_does_not_mutate_input(
    small_net: MorrisResNet, random_input: tuple[torch.Tensor, torch.Tensor]
) -> None:
    x, mask = random_input
    x_before = x.clone()
    mask_before = mask.clone()
    small_net(x, mask)
    assert (x == x_before).all()
    assert (mask == mask_before).all()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_build_network_resnet() -> None:
    config = OmegaConf.create(
        {
            "network": {
                "type": "resnet",
                "num_blocks": 2,
                "num_channels": 16,
                "policy_head_hidden": 32,
                "value_head_hidden": 32,
            }
        }
    )
    net = build_network(config)
    assert isinstance(net, MorrisResNet)


def test_build_network_unknown_raises() -> None:
    config = OmegaConf.create({"network": {"type": "transformer"}})
    with pytest.raises(ValueError, match="Unknown network type"):
        build_network(config)


# ---------------------------------------------------------------------------
# Full-size smoke test (CPU — no GPU required)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_full_size_forward_pass() -> None:
    """10-block, 128-channel network on batch=256 — verifies shape and no crash."""
    net = MorrisResNet(
        num_blocks=10,
        num_channels=128,
        num_planes=NUM_PLANES,
        policy_head_hidden=64,
        value_head_hidden=64,
    )
    net.eval()
    with torch.no_grad():
        x = torch.randn(256, NUM_PLANES, NUM_POSITIONS)
        mask = torch.ones(256, ACTION_SPACE_SIZE, dtype=torch.bool)
        log_policy, value = net(x, mask)
    assert log_policy.shape == (256, ACTION_SPACE_SIZE)
    assert value.shape == (256,)
