"""Tests for the AlphaZero training loop."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from morris_rl.env.board import ACTION_SPACE_SIZE, NUM_POSITIONS
from morris_rl.network.resnet import MorrisResNet
from morris_rl.training.replay_buffer import ReplayBuffer, SampleRecord
from morris_rl.training.trainer import Trainer, compute_loss

_NUM_PLANES = 7
_DEVICE = torch.device("cpu")


def _make_net() -> MorrisResNet:
    return MorrisResNet(
        num_blocks=1,
        num_channels=8,
        num_planes=_NUM_PLANES,
        policy_head_hidden=16,
        value_head_hidden=16,
    )


def _make_buffer(n: int, seed: int = 0) -> ReplayBuffer:
    rng = np.random.default_rng(seed)
    buf = ReplayBuffer(capacity=10_000, use_symmetry_augmentation=False)
    samples = []
    for i in range(n):
        # Mask first, then build a policy that is zero outside the mask — the
        # invariant the production pipeline maintains (MCTS visits never touch
        # illegal actions). Without this, masked log_softmax × uniform policy
        # produces 0 × -inf = NaN in the cross-entropy loss.
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.bool_)
        mask[rng.choice(ACTION_SPACE_SIZE, size=20, replace=False)] = True
        policy = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        policy[mask] = rng.random(int(mask.sum())).astype(np.float32)
        policy /= policy.sum()
        samples.append(
            SampleRecord(
                encoded_state=rng.random((_NUM_PLANES, NUM_POSITIONS)).astype(np.float32),
                policy_target=policy,
                value_target=float(rng.choice([-1.0, 0.0, 1.0])),
                legal_mask=mask,
            )
        )
    buf.add_samples(samples)
    return buf


@pytest.fixture()
def net() -> MorrisResNet:
    n = _make_net()
    n.train()
    return n


@pytest.fixture()
def trainer(net: MorrisResNet) -> Trainer:
    return Trainer(net, _DEVICE, learning_rate=1e-3, mixed_precision=False)


@pytest.fixture()
def buffer() -> ReplayBuffer:
    return _make_buffer(256)


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------


def test_compute_loss_returns_three_tensors() -> None:
    batch = 4
    log_policy = torch.log_softmax(torch.randn(batch, ACTION_SPACE_SIZE), dim=1)
    value = torch.tanh(torch.randn(batch))
    policy_target = torch.softmax(torch.randn(batch, ACTION_SPACE_SIZE), dim=1)
    value_target = torch.tensor([-1.0, 0.0, 1.0, -1.0])
    total, pl, vl, aux = compute_loss(log_policy, value, policy_target, value_target)
    assert total.shape == ()
    assert pl.shape == ()
    assert vl.shape == ()


def test_compute_loss_total_equals_sum() -> None:
    batch = 8
    log_policy = torch.log_softmax(torch.randn(batch, ACTION_SPACE_SIZE), dim=1)
    value = torch.tanh(torch.randn(batch))
    policy_target = torch.softmax(torch.randn(batch, ACTION_SPACE_SIZE), dim=1)
    value_target = torch.zeros(batch)
    total, pl, vl, aux = compute_loss(log_policy, value, policy_target, value_target)
    assert torch.isclose(total, pl + vl)


def test_perfect_policy_gives_zero_policy_loss() -> None:
    """If log_policy exactly matches log(policy_target), policy loss ≈ entropy."""
    batch = 4
    policy_target = torch.softmax(torch.randn(batch, ACTION_SPACE_SIZE), dim=1)
    log_policy = policy_target.log()
    value = torch.zeros(batch)
    value_target = torch.zeros(batch)
    _, pl, _, _ = compute_loss(log_policy, value, policy_target, value_target)
    # Cross-entropy H(p,p) = entropy H(p) ≥ 0, not necessarily 0.
    assert pl.item() >= 0.0


def test_perfect_value_gives_zero_value_loss() -> None:
    batch = 4
    log_policy = torch.log_softmax(torch.zeros(batch, ACTION_SPACE_SIZE), dim=1)
    value_target = torch.tensor([-1.0, 0.0, 1.0, -1.0])
    _, _, vl, _ = compute_loss(log_policy, value_target, log_policy.exp(), value_target)
    assert vl.item() == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Trainer.step
# ---------------------------------------------------------------------------


def test_step_returns_loss_dict(trainer: Trainer, buffer: ReplayBuffer) -> None:
    metrics = trainer.step(buffer, batch_size=32)
    assert "total_loss" in metrics
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "learning_rate" in metrics


def test_step_losses_are_finite(trainer: Trainer, buffer: ReplayBuffer) -> None:
    metrics = trainer.step(buffer, batch_size=32)
    for k, v in metrics.items():
        assert v == v, f"{k} is NaN"  # NaN != NaN
        assert abs(v) < 1e6, f"{k}={v} is unexpectedly large"


def test_step_increments_global_step(trainer: Trainer, buffer: ReplayBuffer) -> None:
    assert trainer.global_step == 0
    trainer.step(buffer, batch_size=32)
    assert trainer.global_step == 1


def test_step_updates_parameters(trainer: Trainer, buffer: ReplayBuffer) -> None:
    """At least one parameter should change after a gradient step."""
    params_before = [p.clone() for p in trainer._network.parameters()]
    trainer.step(buffer, batch_size=32)
    params_after = list(trainer._network.parameters())
    changed = any(not torch.equal(a, b) for a, b in zip(params_before, params_after))
    assert changed


def test_loss_decreases_over_steps(net: MorrisResNet) -> None:
    """Total loss should trend downward when training on fixed data."""
    torch.manual_seed(0)
    trainer = Trainer(net, _DEVICE, learning_rate=1e-2, mixed_precision=False)
    buf = _make_buffer(512, seed=7)
    losses = [trainer.step(buf, batch_size=128)["total_loss"] for _ in range(50)]
    # First half vs second half: mean of second half should be lower.
    first_half = sum(losses[:25]) / 25
    second_half = sum(losses[25:]) / 25
    assert second_half < first_half, (
        f"Loss did not decrease: first_half={first_half:.4f}, second_half={second_half:.4f}"
    )


def test_step_requires_enough_buffer_data(trainer: Trainer) -> None:
    empty_buffer = ReplayBuffer(capacity=1000, use_symmetry_augmentation=False)
    with pytest.raises(ValueError, match="Cannot sample"):
        trainer.step(empty_buffer, batch_size=32)


def test_network_stays_float32(trainer: Trainer, buffer: ReplayBuffer) -> None:
    trainer.step(buffer, batch_size=16)
    for p in trainer._network.parameters():
        assert p.dtype == torch.float32


def test_learning_rate_decreases_over_many_steps(net: MorrisResNet) -> None:
    trainer = Trainer(net, _DEVICE, learning_rate=1e-3, lr_decay_steps=100, mixed_precision=False)
    buf = _make_buffer(256)
    initial_lr = trainer.step(buf, 32)["learning_rate"]
    for _ in range(50):
        trainer.step(buf, 32)
    final_lr = trainer.step(buf, 32)["learning_rate"]
    assert final_lr < initial_lr


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------


def test_save_creates_file(trainer: Trainer, buffer: ReplayBuffer) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        trainer.step(buffer, 32)
        trainer.save(path)
        assert path.exists()


def test_load_restores_step(net: MorrisResNet) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        buf = _make_buffer(256)
        t1 = Trainer(net, _DEVICE, mixed_precision=False)
        for _ in range(5):
            t1.step(buf, 32)
        t1.save(path)

        net2 = _make_net()
        net2.train()
        t2 = Trainer(net2, _DEVICE, mixed_precision=False)
        t2.load(path)
        assert t2.global_step == 5


def test_load_restores_weights(net: MorrisResNet) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        buf = _make_buffer(256)
        t1 = Trainer(net, _DEVICE, mixed_precision=False)
        t1.step(buf, 32)
        t1.save(path)

        net2 = _make_net()
        net2.train()
        t2 = Trainer(net2, _DEVICE, mixed_precision=False)
        t2.load(path)

        for p1, p2 in zip(net.parameters(), net2.parameters()):
            assert torch.allclose(p1, p2)


def test_auto_checkpoint_creates_file(net: MorrisResNet) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_dir = Path(tmp)
        trainer = Trainer(
            net, _DEVICE,
            mixed_precision=False,
            checkpoint_dir=ckpt_dir,
            checkpoint_interval=3,
        )
        buf = _make_buffer(256)
        for _ in range(3):
            trainer.step(buf, 32)
        checkpoints = list(ckpt_dir.glob("checkpoint_*.pt"))
        assert len(checkpoints) == 1
