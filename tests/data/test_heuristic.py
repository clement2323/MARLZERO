"""Tests for the rich Morris heuristic."""

from __future__ import annotations

import numpy as np
import pytest

from morris_rl.data.heuristic import (
    WEIGHTS_MOVEMENT,
    WEIGHTS_PLACEMENT,
    _count_forks,
    _count_potential_mills,
    rich_heuristic,
)
from morris_rl.env.board import NUM_POSITIONS
from morris_rl.env.rules import GameState, initial_state, opponent


def _make_state(
    p1_positions: list[int],
    p2_positions: list[int],
    current_player: int = 1,
    hand_p1: int = 0,
    hand_p2: int = 0,
    must_capture: bool = False,
) -> GameState:
    board = np.zeros(NUM_POSITIONS, dtype=np.int8)
    for p in p1_positions:
        board[p] = 1
    for p in p2_positions:
        board[p] = 2
    return GameState(
        board=board,
        current_player=current_player,
        pieces_in_hand=(hand_p1, hand_p2),
        must_capture=must_capture,
        halfmove_clock=0,
    )


def test_initial_state_score_zero():
    """At the start, both players are perfectly symmetric → score ≈ 0."""
    s = initial_state()
    assert abs(rich_heuristic(s)) < 1e-9


def test_material_advantage_increases_score():
    """Removing an opponent piece must increase score from our POV."""
    base = _make_state([0, 1], [8, 9, 10], current_player=1)
    weaker_opp = _make_state([0, 1], [8, 9], current_player=1)
    assert rich_heuristic(weaker_opp) > rich_heuristic(base)


def test_potential_mill_count_correct():
    """2 own + 1 empty in a mill is exactly one potential mill."""
    # Mill (0,1,2): own at 0 and 1, empty at 2.
    board = np.zeros(NUM_POSITIONS, dtype=np.int8)
    board[0] = 1
    board[1] = 1
    assert _count_potential_mills(board, 1) == 1
    # Add an opponent piece at 2 — no longer a potential mill.
    board[2] = 2
    assert _count_potential_mills(board, 1) == 0


def test_fork_detected():
    """A position where placing creates two simultaneous threats → fork."""
    # Position 9 is on mills (8,9,10) and (1,9,17).
    # Put own pieces at 8 and 17 → placing at 9 creates two potential mills:
    #   - (8,9,10): own at 8 and 9, empty at 10
    #   - (1,9,17): own at 9 and 17, empty at 1
    board = np.zeros(NUM_POSITIONS, dtype=np.int8)
    board[8] = 1
    board[17] = 1
    forks = _count_forks(board, 1)
    assert forks >= 1


def test_no_fork_in_empty_board():
    board = np.zeros(NUM_POSITIONS, dtype=np.int8)
    assert _count_forks(board, 1) == 0


def test_pov_flips_sign_of_material_diff():
    """A state evaluated from P1's POV should be the negation of the same
    state's eval from P2's POV when board symmetric in counts."""
    # P1 has 3 pieces, P2 has 2 → material favors P1.
    s_p1 = _make_state([0, 1, 2], [8, 9], current_player=1)
    s_p2 = _make_state([0, 1, 2], [8, 9], current_player=2)
    # Note: get_phase depends on hand state, which is the same here.
    # Forks/mobility may differ subtly. Test that material component dominates:
    assert rich_heuristic(s_p1) > 0
    assert rich_heuristic(s_p2) < 0


def test_phase_weights_distinct():
    """Sanity check: weights actually differ between placement and movement."""
    assert WEIGHTS_PLACEMENT.material != WEIGHTS_MOVEMENT.material
    assert WEIGHTS_PLACEMENT.forks != WEIGHTS_MOVEMENT.forks


def test_terminal_safe():
    """Heuristic must not crash on edge configurations (empty board, single piece)."""
    empty_state = _make_state([], [], current_player=1, hand_p1=9, hand_p2=9)
    val = rich_heuristic(empty_state)
    assert isinstance(val, float)
