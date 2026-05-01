"""Tests for MCTS search wrapper."""

from __future__ import annotations

import time

import pytest
import torch

from morris_rl.env.board import ACTION_SPACE_SIZE, NUM_POSITIONS
from morris_rl.env.rules import (
    apply_action,
    get_legal_actions,
    initial_state,
)
from morris_rl.mcts.search import MorrisSearch, MorrisSimEnv, encode_state
from morris_rl.network.resnet import MorrisResNet

NUM_PLANES = 7
DEVICE = torch.device("cpu")


@pytest.fixture()
def small_net() -> MorrisResNet:
    net = MorrisResNet(
        num_blocks=2,
        num_channels=16,
        num_planes=NUM_PLANES,
        policy_head_hidden=32,
        value_head_hidden=32,
    )
    net.eval()
    return net


@pytest.fixture()
def search(small_net: MorrisResNet) -> MorrisSearch:
    return MorrisSearch(small_net, DEVICE, num_simulations=10)


# ---------------------------------------------------------------------------
# encode_state
# ---------------------------------------------------------------------------


def test_encode_state_shape() -> None:
    state = initial_state()
    t = encode_state(state)
    assert t.shape == (1, NUM_PLANES, NUM_POSITIONS)


def test_encode_state_dtype() -> None:
    assert encode_state(initial_state()).dtype == torch.float32


def test_encode_state_placement_phase() -> None:
    state = initial_state()
    t = encode_state(state)
    # No pieces on board yet
    assert t[0, 0].sum() == 0.0  # current player planes empty
    assert t[0, 1].sum() == 0.0  # opponent planes empty
    # Phase 0 (PLACING) should be all-ones
    assert (t[0, 4] == 1.0).all()
    assert (t[0, 5] == 0.0).all()
    assert (t[0, 6] == 0.0).all()


def test_encode_state_pieces_in_hand_plane() -> None:
    state = initial_state()
    t = encode_state(state)
    # Both players have all 9 pieces in hand → ratio = 1.0
    assert (t[0, 2] == 1.0).all()
    assert (t[0, 3] == 1.0).all()


def test_encode_state_must_capture_plane() -> None:
    state = initial_state()
    # Reach a must-capture state
    state = apply_action(state, 0)  # P1 → 0
    state = apply_action(state, 5)  # P2 → 5
    state = apply_action(state, 2)  # P1 → 2
    state = apply_action(state, 6)  # P2 → 6
    state = apply_action(state, 1)  # P1 → 1, mill! must_capture=True
    assert state.must_capture
    t = encode_state(state)
    # must_capture is plane 6 in the 7-plane variant (was 7 with FLYING).
    assert (t[0, 6] == 1.0).all()


# ---------------------------------------------------------------------------
# MorrisSimEnv
# ---------------------------------------------------------------------------


def test_sim_env_reset_fresh() -> None:
    env = MorrisSimEnv()
    env.reset()
    assert env.current_player == 1
    assert len(env.legal_actions) == NUM_POSITIONS


def test_sim_env_reset_from_state() -> None:
    state = apply_action(initial_state(), 3)  # P1 placed at 3, P2 to move
    env = MorrisSimEnv()
    env.reset(init_state=state)
    assert env.current_player == 2
    assert 3 not in env.legal_actions


def test_sim_env_step_advances_state() -> None:
    env = MorrisSimEnv()
    env.reset()
    env.step(0)
    assert env.current_player == 2


def test_sim_env_get_done_winner_ongoing() -> None:
    env = MorrisSimEnv()
    env.reset()
    done, winner = env.get_done_winner()
    assert not done
    assert winner == -1


def test_sim_env_reset_does_not_mutate_input() -> None:
    state = initial_state()
    board_before = state.board.copy()
    env = MorrisSimEnv()
    env.reset(init_state=state)
    env.step(0)
    # Original state should be untouched
    assert (state.board == board_before).all()


# ---------------------------------------------------------------------------
# MorrisSearch.run — output contracts
# ---------------------------------------------------------------------------


def test_search_run_returns_legal_action(search: MorrisSearch) -> None:
    state = initial_state()
    action, _ = search.run(state, temperature=1.0)
    assert action in get_legal_actions(state)


def test_search_visit_probs_shape(search: MorrisSearch) -> None:
    _, probs = search.run(initial_state(), temperature=1.0)
    assert probs.shape == (ACTION_SPACE_SIZE,)


def test_search_visit_probs_sum_to_one(search: MorrisSearch) -> None:
    _, probs = search.run(initial_state(), temperature=1.0)
    assert abs(probs.sum() - 1.0) < 1e-5


def test_search_visit_probs_nonnegative(search: MorrisSearch) -> None:
    _, probs = search.run(initial_state(), temperature=1.0)
    assert (probs >= 0.0).all()


def test_search_zero_temp_returns_greedy(search: MorrisSearch) -> None:
    """temperature=0 should still return a single valid action (argmax)."""
    state = initial_state()
    action, probs = search.run(state, temperature=1e-6, add_noise=False)
    assert action in get_legal_actions(state)


def test_search_no_noise_deterministic(small_net: MorrisResNet) -> None:
    """Same network + no noise → same action on repeated calls."""
    search = MorrisSearch(small_net, DEVICE, num_simulations=5)
    state = initial_state()
    # With a fixed network and no noise, repeated calls should give the same action.
    actions = {search.run(state, temperature=1e-6, add_noise=False)[0] for _ in range(3)}
    assert len(actions) == 1


def test_search_mid_game_state(search: MorrisSearch) -> None:
    """Search should work correctly from a mid-game state."""
    state = initial_state()
    for action in [0, 3, 1, 4, 2]:  # P1 mills after last move
        state = apply_action(state, action)
    # state.must_capture == True, legal actions are capture actions
    assert state.must_capture
    action, probs = search.run(state, temperature=1.0)
    assert action in get_legal_actions(state)
    assert abs(probs.sum() - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Performance smoke test
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_search_throughput() -> None:
    """Sanity-check minimum throughput for the pure-Python ptree backend.

    The original M5 target (3000 sims/s) assumes the compiled ctree_alphazero
    C extension, which requires Python 3.11 wheels not yet available for 3.13.
    The ptree backend reaches ~500-800 sims/s on this CPU; 300 sims/s is the
    floor we enforce to catch regressions.
    """
    net = MorrisResNet(
        num_blocks=2,  # small net to isolate MCTS overhead, not network speed
        num_channels=16,
        num_planes=NUM_PLANES,
        policy_head_hidden=32,
        value_head_hidden=32,
    )
    net.eval()
    search = MorrisSearch(net, DEVICE, num_simulations=200)
    state = initial_state()

    t0 = time.perf_counter()
    runs = 5
    for _ in range(runs):
        search.run(state, temperature=1.0)
    elapsed = time.perf_counter() - t0

    sims_per_second = (runs * 200) / elapsed
    assert sims_per_second >= 300, f"Only {sims_per_second:.0f} sims/s — too slow"
