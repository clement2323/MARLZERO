"""Tests for self-play game generation."""

from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from morris_rl.env.board import ACTION_SPACE_SIZE, NUM_POSITIONS
from morris_rl.env.rules import get_legal_actions, initial_state
from morris_rl.mcts.search import MorrisSearch
from morris_rl.network.resnet import MorrisResNet
from morris_rl.training.self_play import (
    GameRecord,
    SelfPlayManager,
    _assign_value_targets,
    _play_game,
    _temperature_for_move,
)

_NUM_PLANES = 7
_DEVICE = torch.device("cpu")
_SMALL_NET_CFG = {
    "num_blocks": 1,
    "num_channels": 8,
    "num_planes": _NUM_PLANES,
    "policy_head_hidden": 16,
    "value_head_hidden": 16,
}


def _make_small_net() -> MorrisResNet:
    net = MorrisResNet(
        num_blocks=_SMALL_NET_CFG["num_blocks"],
        num_channels=_SMALL_NET_CFG["num_channels"],
        num_planes=_SMALL_NET_CFG["num_planes"],
        policy_head_hidden=_SMALL_NET_CFG["policy_head_hidden"],
        value_head_hidden=_SMALL_NET_CFG["value_head_hidden"],
    )
    net.eval()
    return net


@pytest.fixture()
def search() -> MorrisSearch:
    return MorrisSearch(_make_small_net(), _DEVICE, num_simulations=5)


# ---------------------------------------------------------------------------
# Temperature schedule
# ---------------------------------------------------------------------------


def test_temperature_before_threshold_is_one() -> None:
    assert _temperature_for_move(0, 10) == pytest.approx(1.0)
    assert _temperature_for_move(9, 10) == pytest.approx(1.0)


def test_temperature_at_and_after_threshold_is_near_zero() -> None:
    t = _temperature_for_move(10, 10)
    assert t < 1e-4


# ---------------------------------------------------------------------------
# Value target assignment
# ---------------------------------------------------------------------------


def test_value_target_winner_gets_plus_one() -> None:
    from morris_rl.env.rules import Outcome

    steps = [
        (np.zeros((_NUM_PLANES, NUM_POSITIONS), dtype=np.float32),
         np.ones(ACTION_SPACE_SIZE, dtype=np.float32) / ACTION_SPACE_SIZE, 1),
    ]
    records = _assign_value_targets(steps, Outcome.PLAYER_1_WINS)
    assert records[0].value_target == pytest.approx(1.0)


def test_value_target_loser_gets_minus_one() -> None:
    from morris_rl.env.rules import Outcome

    steps = [
        (np.zeros((_NUM_PLANES, NUM_POSITIONS), dtype=np.float32),
         np.ones(ACTION_SPACE_SIZE, dtype=np.float32) / ACTION_SPACE_SIZE, 2),
    ]
    records = _assign_value_targets(steps, Outcome.PLAYER_1_WINS)
    assert records[0].value_target == pytest.approx(-1.0)


def test_value_target_draw_is_zero() -> None:
    from morris_rl.env.rules import Outcome

    steps = [
        (np.zeros((_NUM_PLANES, NUM_POSITIONS), dtype=np.float32),
         np.ones(ACTION_SPACE_SIZE, dtype=np.float32) / ACTION_SPACE_SIZE, 1),
    ]
    records = _assign_value_targets(steps, Outcome.DRAW)
    assert records[0].value_target == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _play_game output contracts
# ---------------------------------------------------------------------------


def test_play_game_returns_game_record(search: MorrisSearch) -> None:
    result = _play_game(search, temperature_threshold=2)
    assert isinstance(result, GameRecord)


def test_play_game_has_at_least_one_sample(search: MorrisSearch) -> None:
    result = _play_game(search, temperature_threshold=2)
    assert len(result.samples) >= 1


def test_play_game_game_length_matches_samples(search: MorrisSearch) -> None:
    result = _play_game(search, temperature_threshold=2)
    assert result.game_length == len(result.samples)


def test_play_game_outcome_is_valid(search: MorrisSearch) -> None:
    result = _play_game(search, temperature_threshold=2)
    assert result.outcome in {1, 2, -1}


def test_play_game_encoded_state_shape(search: MorrisSearch) -> None:
    result = _play_game(search, temperature_threshold=2)
    for sample in result.samples:
        assert sample.encoded_state.shape == (_NUM_PLANES, NUM_POSITIONS)


def test_play_game_policy_sums_to_one(search: MorrisSearch) -> None:
    result = _play_game(search, temperature_threshold=2)
    for sample in result.samples:
        assert abs(sample.policy_target.sum() - 1.0) < 1e-4


def test_play_game_value_targets_in_valid_set(search: MorrisSearch) -> None:
    result = _play_game(search, temperature_threshold=2)
    for sample in result.samples:
        assert sample.value_target in {-1.0, 0.0, 1.0}


def test_play_game_policy_nonnegative(search: MorrisSearch) -> None:
    result = _play_game(search, temperature_threshold=2)
    for sample in result.samples:
        assert (sample.policy_target >= 0.0).all()


# ---------------------------------------------------------------------------
# SelfPlayManager — start / stop (slow: spawns processes)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_manager_start_stop() -> None:
    """Manager should start, produce at least one game, then stop cleanly."""
    net = _make_small_net()
    with SelfPlayManager(
        network=net,
        network_cfg=_SMALL_NET_CFG,
        num_workers=2,
        num_simulations=5,
        temperature_threshold=2,
        seed=0,
    ) as manager:
        game = manager.collect_game(timeout=120.0)
    assert isinstance(game, GameRecord)
    assert len(game.samples) > 0


@pytest.mark.slow
def test_manager_shared_gpu_mode_cpu_fallback() -> None:
    """Run shared_gpu inference mode on CPU (no GPU required in CI).

    Validates that the worker → request queue → server → reply queue round-trip
    works end-to-end and produces valid GameRecords.
    """
    net = _make_small_net()
    with SelfPlayManager(
        network=net,
        network_cfg=_SMALL_NET_CFG,
        num_workers=2,
        num_simulations=5,
        temperature_threshold=2,
        seed=0,
        inference_mode="shared_gpu",
        inference_device="cpu",
        max_batch_size=4,
        max_wait_ms=2.0,
    ) as manager:
        game1 = manager.collect_game(timeout=120.0)
        game2 = manager.collect_game(timeout=120.0)
    assert isinstance(game1, GameRecord) and isinstance(game2, GameRecord)
    assert len(game1.samples) > 0 and len(game2.samples) > 0
    for sample in game1.samples:
        assert sample.encoded_state.shape == (_NUM_PLANES, NUM_POSITIONS)
        assert abs(sample.policy_target.sum() - 1.0) < 1e-4


@pytest.mark.slow
def test_manager_shared_gpu_weights_update_propagates() -> None:
    """update_network in shared_gpu mode should not crash and not block."""
    net = _make_small_net()
    with SelfPlayManager(
        network=net,
        network_cfg=_SMALL_NET_CFG,
        num_workers=2,
        num_simulations=3,
        temperature_threshold=2,
        seed=0,
        inference_mode="shared_gpu",
        inference_device="cpu",
        max_batch_size=4,
        max_wait_ms=2.0,
    ) as manager:
        # Mutate weights deterministically and broadcast — server must consume
        # without dropping new requests.
        for p in net.parameters():
            with torch.no_grad():
                p.add_(0.01)
        manager.update_network(net.state_dict())
        game = manager.collect_game(timeout=120.0)
    assert isinstance(game, GameRecord)
    assert len(game.samples) > 0


@pytest.mark.slow
def test_self_play_throughput() -> None:
    """Verify a baseline throughput with 4 workers and a tiny network.

    Workers are pinned to 1 torch thread each (see _worker_fn) to avoid CPU
    thrashing when running 12+ workers on a 16-core box. With single-thread
    inference the throughput is lower than older multi-threaded regimes;
    20 games/min on 4 workers × 5 sims/move is a safe floor.
    """
    net = _make_small_net()
    target_games = 10
    window = 30.0

    with SelfPlayManager(
        network=net,
        network_cfg=_SMALL_NET_CFG,
        num_workers=4,
        num_simulations=5,
        temperature_threshold=2,
        seed=42,
    ) as manager:
        t0 = time.perf_counter()
        collected = 0
        while time.perf_counter() - t0 < window:
            manager.collect_game(timeout=window)
            collected += 1
            if collected >= target_games:
                break

    elapsed = time.perf_counter() - t0
    rate = collected / elapsed * 60  # games per minute
    assert collected >= target_games, (
        f"Only {collected} games in {elapsed:.1f}s — target was {target_games} in {window}s"
    )
    # 5 games/min is a safe floor under the 300/10 draw rules — random play
    # produces much longer games than the old 50/3 regime, so the historical
    # 20/min target is no longer realistic at 4 workers × 5 sims/move.
    assert rate >= 5, f"Throughput {rate:.1f} games/min < 5 games/min"
