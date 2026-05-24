"""Tests for the per-game asymmetric self-play regime sampler and the
random-opening behavior in _play_game."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from morris_rl.mcts.search import MorrisSearch
from morris_rl.network.resnet import MorrisResNet
from morris_rl.training.self_play import (
    AsymmetricConfig,
    _play_game,
    _sample_regime,
)


_DEVICE = torch.device("cpu")


def _make_small_net() -> MorrisResNet:
    net = MorrisResNet(
        num_blocks=1,
        num_channels=8,
        num_planes=7,
        policy_head_hidden=16,
        value_head_hidden=16,
    )
    net.eval()
    return net


# ---------------------------------------------------------------------------
# _sample_regime
# ---------------------------------------------------------------------------


def test_disabled_always_sym():
    cfg = AsymmetricConfig(enabled=False)
    rng = np.random.default_rng(0)
    for _ in range(50):
        name, params = _sample_regime(rng, cfg)
        assert name == "sym"
        assert params == {}


def test_random_opening_always_sampled_when_prob_one():
    cfg = AsymmetricConfig(
        enabled=True,
        prob_sym=0.0,
        prob_t_asym=0.0,
        prob_noise_asym=0.0,
        prob_random_opening=1.0,
        random_opening_k=4,
    )
    rng = np.random.default_rng(0)
    seen_sides = set()
    for _ in range(200):
        name, params = _sample_regime(rng, cfg)
        assert name == "random_opening"
        assert params["k"] == 4
        assert params["random_side"] in (1, 2)
        seen_sides.add(params["random_side"])
    assert seen_sides == {1, 2}, "Both sides must appear over 200 draws"


def test_random_opening_side_distribution_balanced():
    cfg = AsymmetricConfig(
        enabled=True,
        prob_sym=0.0,
        prob_t_asym=0.0,
        prob_noise_asym=0.0,
        prob_random_opening=1.0,
    )
    rng = np.random.default_rng(42)
    side1 = sum(
        1 for _ in range(2000)
        if _sample_regime(rng, cfg)[1]["random_side"] == 1
    )
    # 95% CI half-width ~= 1.96 * sqrt(0.25 / 2000) ~= 0.022 → tolerate 0.05
    assert abs(side1 / 2000 - 0.5) < 0.05, f"side=1 share {side1/2000} not balanced"


def test_mixed_regime_proportions():
    """All four regimes should fire roughly proportionally when each prob is
    non-zero."""
    cfg = AsymmetricConfig(
        enabled=True,
        prob_sym=0.40,
        prob_t_asym=0.30,
        prob_noise_asym=0.20,
        prob_random_opening=0.10,
    )
    rng = np.random.default_rng(7)
    counts = {"sym": 0, "t_asym": 0, "noise_asym": 0, "random_opening": 0}
    n = 5000
    for _ in range(n):
        name, _ = _sample_regime(rng, cfg)
        counts[name] += 1
    # 95% CI tolerance: ~0.014 → tolerate 0.04 for safety on each share
    assert abs(counts["sym"] / n - 0.40) < 0.04
    assert abs(counts["t_asym"] / n - 0.30) < 0.04
    assert abs(counts["noise_asym"] / n - 0.20) < 0.04
    assert abs(counts["random_opening"] / n - 0.10) < 0.04


# ---------------------------------------------------------------------------
# _play_game with random_opening regime
# ---------------------------------------------------------------------------


def test_random_opening_plies_excluded_from_buffer():
    """A game in the random_opening regime must:
      - tag the resulting GameRecord with regime='random_opening'
      - count the random plies in random_opening_moves
      - exclude those plies from `samples` (so buffer doesn't see uniform-policy noise)
    """
    network = _make_small_net()
    search = MorrisSearch(network, _DEVICE, num_simulations=2)
    asym = AsymmetricConfig(
        enabled=True,
        prob_sym=0.0,
        prob_t_asym=0.0,
        prob_noise_asym=0.0,
        prob_random_opening=1.0,
        random_opening_k=4,
    )
    rng = np.random.default_rng(123)
    record = _play_game(
        search,
        temperature_threshold=10,
        rng=rng,
        asymmetric_config=asym,
    )

    assert record.regime == "random_opening"
    # Exactly 2 plies belong to the random side over the first 4 plies
    # (ply 0/2 if random_side=1, ply 1/3 if random_side=2). The other 2
    # plies in the opening are full-sim MCTS plies and DO end up in
    # samples (their visit_probs are recorded). So the buffer should have
    # game_length - 2 samples (minus any post-mill capture quirk, but with
    # K=4 in the placement phase there is no capture yet).
    expected_random_moves = 2
    assert record.random_opening_moves == expected_random_moves, (
        f"expected {expected_random_moves} random-opening plies, "
        f"got {record.random_opening_moves}"
    )
    # The buffer samples should NOT contain those random plies — every
    # SampleRecord in record.samples must come from a non-random ply.
    # Total recorded samples = game_length - random_opening_moves
    # (assuming no playout cap and no resign).
    assert len(record.samples) == record.game_length - expected_random_moves


def test_random_opening_with_k_zero_is_noop():
    """When random_opening_k=0, the regime is effectively a no-op (zero random
    plies), the game plays out fully MCTS-driven."""
    network = _make_small_net()
    search = MorrisSearch(network, _DEVICE, num_simulations=2)
    asym = AsymmetricConfig(
        enabled=True,
        prob_sym=0.0,
        prob_t_asym=0.0,
        prob_noise_asym=0.0,
        prob_random_opening=1.0,
        random_opening_k=0,
    )
    rng = np.random.default_rng(7)
    record = _play_game(
        search,
        temperature_threshold=10,
        rng=rng,
        asymmetric_config=asym,
    )
    assert record.regime == "random_opening"
    assert record.random_opening_moves == 0
    # All plies are recorded (modulo playout_cap which is disabled here)
    assert len(record.samples) == record.game_length
