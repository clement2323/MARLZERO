"""Tests for MorrisGraphNet — same invariants as the ResNet, plus D4 equivariance."""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from morris_rl.env.board import ACTION_SPACE_SIZE, NUM_POSITIONS
from morris_rl.env.encoding_graph import NUM_GRAPH_PLANES, encode_state_graph
from morris_rl.env.rules import initial_state, apply_action, get_legal_actions
from morris_rl.env.symmetries import (
    SYMMETRY_PERMUTATIONS,
    transform_encoded_state,
    transform_policy,
)
from morris_rl.network.factory import build_network
from morris_rl.network.graphnet import MorrisGraphNet


BATCH = 4


@pytest.fixture()
def small_graphnet() -> MorrisGraphNet:
    """Tiny GraphNet (2 blocks, 16 channels) for fast CPU tests."""
    return MorrisGraphNet(
        num_blocks=2,
        num_channels=16,
        num_planes=NUM_GRAPH_PLANES,
        policy_head_hidden=32,
        value_head_hidden=32,
    )


@pytest.fixture()
def random_input() -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(BATCH, NUM_GRAPH_PLANES, NUM_POSITIONS)
    mask = torch.ones(BATCH, ACTION_SPACE_SIZE, dtype=torch.bool)
    return x, mask


# ---------------------------------------------------------------------------
# Output shapes — same contract as MorrisResNet
# ---------------------------------------------------------------------------


def test_policy_output_shape(small_graphnet: MorrisGraphNet,
                             random_input: tuple[torch.Tensor, torch.Tensor]) -> None:
    x, mask = random_input
    small_graphnet.eval()
    with torch.no_grad():
        log_policy, _ = small_graphnet(x, mask)
    assert log_policy.shape == (BATCH, ACTION_SPACE_SIZE)


def test_value_output_shape(small_graphnet: MorrisGraphNet,
                            random_input: tuple[torch.Tensor, torch.Tensor]) -> None:
    x, mask = random_input
    small_graphnet.eval()
    with torch.no_grad():
        _, value = small_graphnet(x, mask)
    assert value.shape == (BATCH,)


def test_value_head_in_range(small_graphnet: MorrisGraphNet,
                              random_input: tuple[torch.Tensor, torch.Tensor]) -> None:
    x, mask = random_input
    small_graphnet.eval()
    with torch.no_grad():
        _, value = small_graphnet(x, mask)
    assert (value >= -1.0).all() and (value <= 1.0).all()


def test_log_policy_is_log_prob(small_graphnet: MorrisGraphNet,
                                 random_input: tuple[torch.Tensor, torch.Tensor]) -> None:
    x, mask = random_input
    small_graphnet.eval()
    with torch.no_grad():
        log_policy, _ = small_graphnet(x, mask)
    probs = log_policy.exp()
    sums = probs.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_masked_actions_have_zero_probability(small_graphnet: MorrisGraphNet) -> None:
    x = torch.randn(BATCH, NUM_GRAPH_PLANES, NUM_POSITIONS)
    # Build a sparse mask: only first 5 actions legal.
    mask = torch.zeros(BATCH, ACTION_SPACE_SIZE, dtype=torch.bool)
    mask[:, :5] = True
    small_graphnet.eval()
    with torch.no_grad():
        log_policy, _ = small_graphnet(x, mask)
    probs = log_policy.exp()
    # Illegal actions should be ~0 (after log_softmax over -inf logits).
    assert torch.allclose(probs[:, 5:], torch.zeros_like(probs[:, 5:]), atol=1e-6)
    # Legal sum still ≈ 1.
    assert torch.allclose(probs[:, :5].sum(dim=1), torch.ones(BATCH), atol=1e-5)


# ---------------------------------------------------------------------------
# Optional return paths (matching ResNet's flags)
# ---------------------------------------------------------------------------


def test_categorical_value_head_returns_logits() -> None:
    net = MorrisGraphNet(
        num_blocks=2, num_channels=16, num_planes=NUM_GRAPH_PLANES,
        policy_head_hidden=32, value_head_hidden=32, value_head_type="categorical",
    )
    x = torch.randn(BATCH, NUM_GRAPH_PLANES, NUM_POSITIONS)
    mask = torch.ones(BATCH, ACTION_SPACE_SIZE, dtype=torch.bool)
    net.eval()
    with torch.no_grad():
        log_policy, scalar, logits = net(x, mask, return_value_logits=True)
    assert log_policy.shape == (BATCH, ACTION_SPACE_SIZE)
    assert scalar.shape == (BATCH,)
    assert logits.shape == (BATCH, 3)


def test_aux_heads_return_per_sample_scalars() -> None:
    net = MorrisGraphNet(
        num_blocks=2, num_channels=16, num_planes=NUM_GRAPH_PLANES,
        policy_head_hidden=32, value_head_hidden=32,
        aux_heads_enabled=True, aux_head_hidden=16,
    )
    x = torch.randn(BATCH, NUM_GRAPH_PLANES, NUM_POSITIONS)
    mask = torch.ones(BATCH, ACTION_SPACE_SIZE, dtype=torch.bool)
    net.eval()
    with torch.no_grad():
        log_policy, scalar, mill_pred, pieces_pred = net(x, mask, return_aux=True)
    assert log_policy.shape == (BATCH, ACTION_SPACE_SIZE)
    assert scalar.shape == (BATCH,)
    assert mill_pred.shape == (BATCH,)
    assert pieces_pred.shape == (BATCH,)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_gradients_flow(small_graphnet: MorrisGraphNet,
                        random_input: tuple[torch.Tensor, torch.Tensor]) -> None:
    x, mask = random_input
    small_graphnet.train()
    log_policy, value = small_graphnet(x, mask)
    loss = -log_policy.mean() + value.mean()
    loss.backward()
    grads = [p.grad for p in small_graphnet.parameters() if p.requires_grad]
    assert grads, "no trainable parameters?"
    for g in grads:
        assert g is not None
        assert not torch.isnan(g).any()


# ---------------------------------------------------------------------------
# Factory dispatch
# ---------------------------------------------------------------------------


def test_factory_builds_graphnet() -> None:
    cfg = OmegaConf.create({
        "network": {
            "type": "graphnet",
            "num_blocks": 2,
            "num_channels": 16,
            "policy_head_hidden": 32,
            "value_head_hidden": 32,
            "value_head_type": "scalar",
            "lora_rank": 0, "lora_alpha": 16.0, "freeze_trunk": False,
        },
        "input_encoding": {"num_planes": 11},
        "aux_heads": {"enabled": False, "hidden_size": 32,
                      "mill_diff_weight": 0.0, "pieces_diff_weight": 0.0},
    })
    net = build_network(cfg)
    assert isinstance(net, MorrisGraphNet)


def test_factory_still_builds_resnet_by_default() -> None:
    """Regression guard: type=resnet (or absent) must still pick ResNet."""
    from morris_rl.network.resnet import MorrisResNet
    cfg = OmegaConf.create({
        "network": {
            "type": "resnet",
            "num_blocks": 2, "num_channels": 16,
            "policy_head_hidden": 32, "value_head_hidden": 32,
            "lora_rank": 0, "lora_alpha": 16.0, "freeze_trunk": False,
        },
        "input_encoding": {"num_planes": 7},
        "aux_heads": {"enabled": False, "hidden_size": 32,
                      "mill_diff_weight": 0.0, "pieces_diff_weight": 0.0},
    })
    net = build_network(cfg)
    assert isinstance(net, MorrisResNet)


# ---------------------------------------------------------------------------
# LoRA — same contract as ResNet
# ---------------------------------------------------------------------------


def test_lora_wrap_then_freeze(small_graphnet: MorrisGraphNet) -> None:
    from morris_rl.network.lora import LoRALinear

    small_graphnet.add_lora_adapters(rank=4, alpha=8.0)
    # All Linear layers should now be LoRALinear.
    has_lora = any(isinstance(m, LoRALinear) for m in small_graphnet.modules())
    assert has_lora

    small_graphnet.freeze_trunk()
    trainable = [
        n for n, p in small_graphnet.named_parameters() if p.requires_grad
    ]
    # Every trainable param should be a LoRA matrix.
    for name in trainable:
        assert "lora_A" in name or "lora_B" in name, f"unexpected trainable: {name}"


# ---------------------------------------------------------------------------
# Trunk-level D4 equivariance — the message-passing layers commute with the
# board's 8 symmetries by construction. The policy/value heads break strict
# equivariance because they flatten the position axis into a single Linear,
# so we only validate the trunk output, not the head output. This still
# validates the inductive bias we wanted from the GNN; data augmentation in
# the replay buffer continues to handle the head non-equivariance.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sym_idx", [1, 2, 3, 4, 5, 6, 7])
def test_trunk_d4_equivariance(small_graphnet: MorrisGraphNet, sym_idx: int) -> None:
    state = initial_state()
    import numpy as np
    rng = np.random.default_rng(0)
    for _ in range(4):
        legal = get_legal_actions(state)
        state = apply_action(state, int(rng.choice(legal)))

    encoded = encode_state_graph(state)[0].numpy()  # (11, 24)
    perm = SYMMETRY_PERMUTATIONS[sym_idx]
    encoded_sym = transform_encoded_state(encoded, perm)

    x1 = torch.from_numpy(encoded).unsqueeze(0)
    x2 = torch.from_numpy(encoded_sym).unsqueeze(0)

    small_graphnet.eval()
    with torch.no_grad():
        # Run the trunk only (skip heads).
        h1 = x1.transpose(1, 2).contiguous()
        h1 = small_graphnet.input_proj(h1)
        h1 = torch.nn.functional.relu(small_graphnet.input_bn(h1.transpose(1, 2))).transpose(1, 2)
        for block in small_graphnet.blocks:
            h1 = block(h1, small_graphnet.A_adj, small_graphnet.A_mill)

        h2 = x2.transpose(1, 2).contiguous()
        h2 = small_graphnet.input_proj(h2)
        h2 = torch.nn.functional.relu(small_graphnet.input_bn(h2.transpose(1, 2))).transpose(1, 2)
        for block in small_graphnet.blocks:
            h2 = block(h2, small_graphnet.A_adj, small_graphnet.A_mill)

    # h2 should equal h1 permuted along the node axis.
    h1_np = h1[0].numpy()                    # (24, C)
    h2_np = h2[0].numpy()                    # (24, C)
    h1_permuted = np.empty_like(h1_np)
    h1_permuted[perm] = h1_np                # permute the node axis
    diff = abs(h1_permuted - h2_np).max()
    assert diff < 1e-4, f"trunk not equivariant under sym {sym_idx}: max diff = {diff}"
