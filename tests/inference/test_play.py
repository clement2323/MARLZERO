"""Tests for inference/play.py helpers."""

from __future__ import annotations

import pytest
import torch

from morris_rl.env.board import NUM_PLACE_CAPTURE_ACTIONS
from morris_rl.env.rules import apply_action, initial_state, is_terminal
from morris_rl.inference.play import (
    POSITION_LABELS,
    describe_action,
    get_network_value,
    reconstruct_state,
    run_mcts_analysis,
)
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
# describe_action
# ---------------------------------------------------------------------------


def test_position_labels_length() -> None:
    assert len(POSITION_LABELS) == 24


def test_describe_place_action() -> None:
    desc = describe_action(0, must_capture=False)
    assert "Place" in desc
    assert POSITION_LABELS[0] in desc


def test_describe_capture_action() -> None:
    desc = describe_action(5, must_capture=True)
    assert "Capture" in desc
    assert POSITION_LABELS[5] in desc


def test_describe_move_action() -> None:
    # Action 24 + 0*24 + 1 = 25: move from position 0 to position 1
    desc = describe_action(NUM_PLACE_CAPTURE_ACTIONS + 0 * 24 + 1, must_capture=False)
    assert "Move" in desc
    assert POSITION_LABELS[0] in desc
    assert POSITION_LABELS[1] in desc


# ---------------------------------------------------------------------------
# reconstruct_state
# ---------------------------------------------------------------------------


def test_reconstruct_empty_actions_is_initial() -> None:
    state = reconstruct_state([])
    initial = initial_state()
    assert (state.board == initial.board).all()
    assert state.current_player == initial.current_player


def test_reconstruct_one_action() -> None:
    state = reconstruct_state([0])
    assert state.board[0] == 1  # Player 1 placed at position 0
    assert state.current_player == 2  # Now Player 2's turn


def test_reconstruct_preserves_position_counts() -> None:
    """Replaying actions correctly builds position_counts for repetition detection."""
    state = reconstruct_state([0, 5, 1, 6])
    # position_counts should have tracked all intermediate states.
    assert len(state.position_counts) >= 1


def test_reconstruct_more_actions_advances_state() -> None:
    """Each additional action in the history advances the state by one step."""
    state_after_2 = reconstruct_state([0, 5])
    state_after_3 = reconstruct_state([0, 5, 2])
    # After 3 actions, one more piece is on the board.
    assert state_after_3.board.sum() > state_after_2.board.sum()


# ---------------------------------------------------------------------------
# get_network_value
# ---------------------------------------------------------------------------


def test_network_value_in_range() -> None:
    net = _small_net()
    state = initial_state()
    v = get_network_value(net, _DEVICE, state)
    assert -1.0 <= v <= 1.0


def test_network_value_is_float() -> None:
    net = _small_net()
    v = get_network_value(net, _DEVICE, initial_state())
    assert isinstance(v, float)


# ---------------------------------------------------------------------------
# run_mcts_analysis
# ---------------------------------------------------------------------------


def test_mcts_analysis_returns_legal_action() -> None:
    net = _small_net()
    state = initial_state()
    from morris_rl.env.rules import get_legal_actions
    action, _, _ = run_mcts_analysis(net, _DEVICE, state, num_simulations=5)
    assert action in get_legal_actions(state)


def test_mcts_analysis_top_moves_nonempty() -> None:
    net = _small_net()
    state = initial_state()
    _, top_moves, _ = run_mcts_analysis(net, _DEVICE, state, num_simulations=5, num_top_moves=3)
    assert len(top_moves) >= 1


def test_mcts_analysis_top_moves_sorted() -> None:
    net = _small_net()
    state = initial_state()
    _, top_moves, _ = run_mcts_analysis(net, _DEVICE, state, num_simulations=10, num_top_moves=3)
    probs = [p for _, p in top_moves]
    assert probs == sorted(probs, reverse=True)


def test_mcts_analysis_value_in_range() -> None:
    net = _small_net()
    _, _, value = run_mcts_analysis(
        net, _DEVICE, initial_state(), num_simulations=5
    )
    assert -1.0 <= value <= 1.0
