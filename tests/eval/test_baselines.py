"""Tests for baseline agents."""

from __future__ import annotations

import pytest
import torch

from morris_rl.env.rules import apply_action, get_legal_actions, initial_state, is_terminal
from morris_rl.eval.arena import run_arena
from morris_rl.eval.baselines import MinimaxAgent, NetworkAgent, RandomAgent, _heuristic
from morris_rl.network.resnet import MorrisResNet

_DEVICE = torch.device("cpu")


def _small_net() -> MorrisResNet:
    net = MorrisResNet(
        num_blocks=1, num_channels=8, num_planes=8,
        policy_head_hidden=16, value_head_hidden=16,
    )
    net.eval()
    return net


# ---------------------------------------------------------------------------
# RandomAgent
# ---------------------------------------------------------------------------


def test_random_agent_returns_legal_action() -> None:
    agent = RandomAgent(seed=0)
    state = initial_state()
    action = agent.select_action(state)
    assert action in get_legal_actions(state)


def test_random_agent_plays_full_game() -> None:
    agent = RandomAgent(seed=42)
    state = initial_state()
    moves = 0
    while True:
        done, _ = is_terminal(state)
        if done:
            break
        state = apply_action(state, agent.select_action(state))
        moves += 1
    assert moves > 0


def test_random_agent_different_seeds_can_differ() -> None:
    state = initial_state()
    actions = {RandomAgent(seed=i).select_action(state) for i in range(20)}
    assert len(actions) > 1  # at least two distinct actions


# ---------------------------------------------------------------------------
# Heuristic
# ---------------------------------------------------------------------------


def test_heuristic_initial_state_is_zero() -> None:
    """Both players have equal material at start → heuristic = 0."""
    assert _heuristic(initial_state()) == pytest.approx(0.0)


def test_heuristic_after_capture_disadvantages_current_player() -> None:
    """After P1 forms a mill and captures, P2's material is one piece lower.

    The heuristic is evaluated from P2's perspective (current player after the
    capture) and should be negative since P2 is down one piece.
    """
    state = initial_state()
    # P1 places to form mill (0,1,2); P2 places elsewhere.
    for action in [0, 5, 2, 6, 1]:  # P1 mills on move 5
        state = apply_action(state, action)
    assert state.must_capture  # P1 must capture a P2 piece
    state = apply_action(state, 5)  # P1 captures P2's piece at 5
    # Now current_player = 2; P2 has 7 in hand + 1 on board = 8 total,
    # P1 has 6 in hand + 3 on board = 9 total.
    h = _heuristic(state)
    assert h < 0.0, f"Expected negative heuristic (P2 at disadvantage), got {h}"


# ---------------------------------------------------------------------------
# MinimaxAgent
# ---------------------------------------------------------------------------


def test_minimax_rejects_depth_zero() -> None:
    with pytest.raises(ValueError):
        MinimaxAgent(depth=0)


def test_minimax_returns_legal_action() -> None:
    agent = MinimaxAgent(depth=1)
    state = initial_state()
    action = agent.select_action(state)
    assert action in get_legal_actions(state)


def test_minimax_depth1_plays_full_game() -> None:
    agent = MinimaxAgent(depth=1)
    state = initial_state()
    moves = 0
    while True:
        done, _ = is_terminal(state)
        if done:
            break
        state = apply_action(state, agent.select_action(state))
        moves += 1
    assert moves > 0


def test_minimax_depth1_beats_random() -> None:
    """MinimaxAgent(1) should beat RandomAgent in the majority of 20 games."""
    minimax = MinimaxAgent(depth=1)
    random_agent = RandomAgent(seed=0)
    summary = run_arena(minimax, random_agent, num_games=20)
    assert summary.win_rate_a > 0.5, (
        f"Minimax(1) won only {summary.win_rate_a:.0%} vs random — expected > 50 %"
    )


@pytest.mark.slow
def test_minimax_depth3_beats_random_convincingly() -> None:
    """MinimaxAgent(3) should win ≥ 70 % of games against a random agent."""
    minimax = MinimaxAgent(depth=3)
    random_agent = RandomAgent(seed=0)
    summary = run_arena(minimax, random_agent, num_games=20)
    assert summary.win_rate_a >= 0.7, (
        f"Minimax(3) win rate {summary.win_rate_a:.0%} — expected ≥ 70 %"
    )


# ---------------------------------------------------------------------------
# NetworkAgent
# ---------------------------------------------------------------------------


def test_network_agent_returns_legal_action() -> None:
    net = _small_net()
    agent = NetworkAgent(net, _DEVICE, num_simulations=5)
    state = initial_state()
    action = agent.select_action(state)
    assert action in get_legal_actions(state)


def test_network_agent_plays_full_game() -> None:
    net = _small_net()
    agent = NetworkAgent(net, _DEVICE, num_simulations=5)
    state = initial_state()
    moves = 0
    while True:
        done, _ = is_terminal(state)
        if done:
            break
        state = apply_action(state, agent.select_action(state))
        moves += 1
    assert moves > 0
