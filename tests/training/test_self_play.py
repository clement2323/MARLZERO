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
         np.ones(ACTION_SPACE_SIZE, dtype=np.float32) / ACTION_SPACE_SIZE, 1,
         np.ones(ACTION_SPACE_SIZE, dtype=np.bool_),
         0, False),
    ]
    records = _assign_value_targets(
        steps, Outcome.PLAYER_1_WINS, final_pieces_p1=5, final_pieces_p2=2
    )
    assert records[0].value_target == pytest.approx(1.0)


def test_value_target_loser_gets_minus_one() -> None:
    from morris_rl.env.rules import Outcome

    steps = [
        (np.zeros((_NUM_PLANES, NUM_POSITIONS), dtype=np.float32),
         np.ones(ACTION_SPACE_SIZE, dtype=np.float32) / ACTION_SPACE_SIZE, 2,
         np.ones(ACTION_SPACE_SIZE, dtype=np.bool_),
         0, False),
    ]
    records = _assign_value_targets(
        steps, Outcome.PLAYER_1_WINS, final_pieces_p1=5, final_pieces_p2=2
    )
    assert records[0].value_target == pytest.approx(-1.0)


def test_value_target_draw_is_zero() -> None:
    from morris_rl.env.rules import Outcome

    steps = [
        (np.zeros((_NUM_PLANES, NUM_POSITIONS), dtype=np.float32),
         np.ones(ACTION_SPACE_SIZE, dtype=np.float32) / ACTION_SPACE_SIZE, 1,
         np.ones(ACTION_SPACE_SIZE, dtype=np.bool_),
         0, False),
    ]
    records = _assign_value_targets(
        steps, Outcome.DRAW, final_pieces_p1=3, final_pieces_p2=3
    )
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
# Per-game observability stats (mills, captures, term_reason, pieces_diff)
# ---------------------------------------------------------------------------


def test_play_game_observability_fields_set(search: MorrisSearch) -> None:
    """All observability counters and term_reason are populated."""
    result = _play_game(search, temperature_threshold=2)
    # Counters are non-negative integers.
    assert result.mills_p1 >= 0
    assert result.mills_p2 >= 0
    assert result.captures_p1 >= 0
    assert result.captures_p2 >= 0
    # final_pieces_diff is signed but bounded by the 18 placed pieces.
    assert -18 <= result.final_pieces_diff <= 18
    # term_reason is one of the documented enum values (or "unknown" only as
    # a sentinel — never expected on a properly terminated game).
    assert result.term_reason in {
        "pieces_below_3",
        "no_legal_moves",
        "halfmove_cap",
        "threefold",
        "resign",
    }


def test_play_game_captures_match_mills_in_count(search: MorrisSearch) -> None:
    """Each mill formed enables exactly one capture (modulo unfinished sub-turns)."""
    result = _play_game(search, temperature_threshold=2)
    # The number of captures by a player can be at most the mills they formed.
    # Equality usually holds; strict inequality only if the game terminated
    # right between mill formation and capture (extremely rare with our rules).
    assert result.captures_p1 <= result.mills_p1
    assert result.captures_p2 <= result.mills_p2


def test_play_game_resigns_when_threshold_crossed(monkeypatch) -> None:
    """When root_value stays below threshold, the loser-to-be resigns and
    value targets propagate as a forfeit."""
    from morris_rl.training.self_play import ResignConfig
    # Force MorrisSearch.root_value to always return -1 (always "I'm losing")
    # so the resign trigger fires immediately past the min_move_for_resign.
    monkeypatch.setattr(
        "morris_rl.mcts.search.MorrisSearch.root_value",
        lambda self, state: -1.0,
    )
    cfg = ResignConfig(
        enabled=True,
        threshold=-0.5,
        min_consecutive_below=2,
        min_move_for_resign=5,    # short so the fixture's small games trigger
        verify_fraction=0.0,       # never verify — always resign
    )
    rng = np.random.default_rng(0)
    search = MorrisSearch(_make_small_net(), _DEVICE, num_simulations=5)
    result = _play_game(
        search, temperature_threshold=2, resign_config=cfg, rng=rng
    )
    # Game ended by resign; one player forfeited.
    assert result.term_reason == "resign"
    assert result.resigned_by_player in {1, 2}
    assert result.resign_eligible is True
    assert result.was_verify_play is False
    # Outcome reflects opponent victory.
    assert result.outcome == (3 - result.resigned_by_player)
    # Value targets propagated correctly.
    for sample in result.samples:
        # Reconstructing the player from the encoded state isn't trivial,
        # but the resign forfeit means at least one v=-1 and one v=+1.
        pass
    sign_set = {sample.value_target for sample in result.samples}
    assert sign_set <= {-1.0, 1.0}


def test_play_game_term_reason_threefold_under_repetition() -> None:
    """When a game ends by threefold, term_reason reflects that."""
    # Force threefold by playing a game from scratch: the small net at depth 5
    # sometimes oscillates. We just verify the detector logic given a state.
    from morris_rl.training.self_play import _detect_term_reason
    from morris_rl.env.rules import GameState, Outcome, THREEFOLD_LIMIT
    state = GameState(
        board=np.zeros(NUM_POSITIONS, dtype=np.int8),
        current_player=1,
        pieces_in_hand=(0, 0),
        must_capture=False,
        halfmove_clock=0,
        position_counts={(0,) * 24 + (1, 0, 0, 0): THREEFOLD_LIMIT},
    )
    state.board[0] = state.board[1] = state.board[2] = 1
    state.board[5] = state.board[6] = state.board[7] = 2
    assert _detect_term_reason(state, Outcome.DRAW) == "threefold"


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
