"""Tests for the FIFO replay buffer."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from morris_rl.env.board import ACTION_SPACE_SIZE, NUM_POSITIONS
from morris_rl.training.replay_buffer import ReplayBuffer, SampleRecord

_NUM_PLANES = 8


def _make_sample(seed: int = 0) -> SampleRecord:
    rng = np.random.default_rng(seed)
    policy = rng.random(ACTION_SPACE_SIZE).astype(np.float32)
    policy /= policy.sum()
    return SampleRecord(
        encoded_state=rng.random((_NUM_PLANES, NUM_POSITIONS)).astype(np.float32),
        policy_target=policy,
        value_target=float(rng.choice([-1.0, 0.0, 1.0])),
    )


# ---------------------------------------------------------------------------
# Basic add / len tests
# ---------------------------------------------------------------------------


def test_empty_buffer_has_zero_len() -> None:
    buf = ReplayBuffer(capacity=100, use_symmetry_augmentation=False)
    assert len(buf) == 0


def test_add_single_sample_increases_len() -> None:
    buf = ReplayBuffer(capacity=100, use_symmetry_augmentation=False)
    buf.add(_make_sample())
    assert len(buf) == 1


def test_add_samples_list_increases_len() -> None:
    buf = ReplayBuffer(capacity=100, use_symmetry_augmentation=False)
    buf.add_samples([_make_sample(i) for i in range(5)])
    assert len(buf) == 5


def test_capacity_cap_enforced() -> None:
    buf = ReplayBuffer(capacity=10, use_symmetry_augmentation=False)
    for i in range(20):
        buf.add(_make_sample(i))
    assert len(buf) == 10


def test_capacity_property() -> None:
    buf = ReplayBuffer(capacity=256, use_symmetry_augmentation=False)
    assert buf.capacity == 256


# ---------------------------------------------------------------------------
# Symmetry augmentation
# ---------------------------------------------------------------------------


def test_augmentation_multiplies_count_by_eight() -> None:
    buf = ReplayBuffer(capacity=1000, use_symmetry_augmentation=True)
    buf.add(_make_sample(0))
    assert len(buf) == 8


def test_augmentation_add_samples_multiplies_count() -> None:
    buf = ReplayBuffer(capacity=1000, use_symmetry_augmentation=True)
    buf.add_samples([_make_sample(i) for i in range(3)])
    assert len(buf) == 24  # 3 * 8


def test_no_augmentation_does_not_multiply() -> None:
    buf = ReplayBuffer(capacity=1000, use_symmetry_augmentation=False)
    buf.add_samples([_make_sample(i) for i in range(3)])
    assert len(buf) == 3


# ---------------------------------------------------------------------------
# sample() output contracts
# ---------------------------------------------------------------------------


def test_sample_requires_enough_data() -> None:
    buf = ReplayBuffer(capacity=100, use_symmetry_augmentation=False)
    buf.add(_make_sample())
    with pytest.raises(ValueError, match="Cannot sample"):
        buf.sample(10)


def test_sample_state_shape() -> None:
    buf = ReplayBuffer(capacity=100, use_symmetry_augmentation=False)
    buf.add_samples([_make_sample(i) for i in range(10)])
    states, _, _ = buf.sample(4)
    assert states.shape == (4, _NUM_PLANES, NUM_POSITIONS)


def test_sample_policy_shape() -> None:
    buf = ReplayBuffer(capacity=100, use_symmetry_augmentation=False)
    buf.add_samples([_make_sample(i) for i in range(10)])
    _, policies, _ = buf.sample(4)
    assert policies.shape == (4, ACTION_SPACE_SIZE)


def test_sample_values_shape() -> None:
    buf = ReplayBuffer(capacity=100, use_symmetry_augmentation=False)
    buf.add_samples([_make_sample(i) for i in range(10)])
    _, _, values = buf.sample(4)
    assert values.shape == (4,)


def test_sample_dtypes_are_float32() -> None:
    buf = ReplayBuffer(capacity=100, use_symmetry_augmentation=False)
    buf.add_samples([_make_sample(i) for i in range(10)])
    states, policies, values = buf.sample(4)
    assert states.dtype == torch.float32
    assert policies.dtype == torch.float32
    assert values.dtype == torch.float32


def test_sample_to_device() -> None:
    buf = ReplayBuffer(capacity=100, use_symmetry_augmentation=False)
    buf.add_samples([_make_sample(i) for i in range(10)])
    states, policies, values = buf.sample(4, device=torch.device("cpu"))
    assert states.device.type == "cpu"


def test_sample_is_random() -> None:
    """Two consecutive samples from the same buffer should differ."""
    np.random.seed(0)
    buf = ReplayBuffer(capacity=1000, use_symmetry_augmentation=False)
    buf.add_samples([_make_sample(i) for i in range(100)])
    _, policies_a, _ = buf.sample(8)
    _, policies_b, _ = buf.sample(8)
    # Very unlikely to be identical with 100 distinct samples.
    assert not torch.allclose(policies_a, policies_b)


# ---------------------------------------------------------------------------
# FIFO eviction
# ---------------------------------------------------------------------------


def test_oldest_entries_evicted_when_full() -> None:
    """After overfilling, the buffer should contain exactly capacity entries."""
    capacity = 5
    buf = ReplayBuffer(capacity=capacity, use_symmetry_augmentation=False)
    for i in range(12):
        buf.add(_make_sample(i))
    assert len(buf) == capacity


def test_value_in_buffer_matches_added_sample() -> None:
    """Sampled values should exactly match what was put in."""
    buf = ReplayBuffer(capacity=100, use_symmetry_augmentation=False)
    sample = SampleRecord(
        encoded_state=np.zeros((_NUM_PLANES, NUM_POSITIONS), dtype=np.float32),
        policy_target=np.ones(ACTION_SPACE_SIZE, dtype=np.float32) / ACTION_SPACE_SIZE,
        value_target=1.0,
    )
    buf.add(sample)
    _, _, values = buf.sample(1)
    assert values[0].item() == pytest.approx(1.0)
