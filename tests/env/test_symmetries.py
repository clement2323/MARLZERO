"""Tests for the D4 board symmetry transforms."""

from __future__ import annotations

import numpy as np
import pytest

from morris_rl.env.board import (
    ACTION_SPACE_SIZE,
    EDGE_INDEX,
    NUM_PLACE_CAPTURE_ACTIONS,
    NUM_POSITIONS,
)
from morris_rl.env.rules import apply_action, get_legal_actions, initial_state
from morris_rl.env.symmetries import (
    SYMMETRY_INVERSE_PERMUTATIONS,
    SYMMETRY_PERMUTATIONS,
    transform_board,
    transform_encoded_state,
    transform_policy,
)
from morris_rl.mcts.search import encode_state

_R90 = SYMMETRY_PERMUTATIONS[1]
_R180 = SYMMETRY_PERMUTATIONS[2]
_R270 = SYMMETRY_PERMUTATIONS[3]
_FL = SYMMETRY_PERMUTATIONS[4]
_IDENTITY = SYMMETRY_PERMUTATIONS[0]


# ---------------------------------------------------------------------------
# Board permutation tests
# ---------------------------------------------------------------------------


def test_identity_board_is_noop() -> None:
    board = np.arange(NUM_POSITIONS, dtype=np.int8)
    result = transform_board(board, _IDENTITY)
    assert np.array_equal(result, board)


def test_symmetries_are_bijections() -> None:
    """Every permutation is a valid bijection of {0, …, 23}."""
    for perm in SYMMETRY_PERMUTATIONS:
        assert sorted(perm.tolist()) == list(range(NUM_POSITIONS))


def test_r90_r270_are_inverses() -> None:
    board = np.array([1 if i % 3 == 0 else 0 for i in range(NUM_POSITIONS)], dtype=np.int8)
    rotated = transform_board(board, _R90)
    restored = transform_board(rotated, _R270)
    assert np.array_equal(restored, board)


def test_r180_is_self_inverse() -> None:
    board = np.array([2 if i % 5 == 0 else 0 for i in range(NUM_POSITIONS)], dtype=np.int8)
    twice = transform_board(transform_board(board, _R180), _R180)
    assert np.array_equal(twice, board)


def test_fl_is_self_inverse() -> None:
    board = np.array([1 if i % 2 == 0 else 0 for i in range(NUM_POSITIONS)], dtype=np.int8)
    twice = transform_board(transform_board(board, _FL), _FL)
    assert np.array_equal(twice, board)


@pytest.mark.parametrize("sym_idx", range(8))
def test_all_symmetries_have_correct_inverse(sym_idx: int) -> None:
    board = np.array(list(range(NUM_POSITIONS)), dtype=np.int8)
    perm = SYMMETRY_PERMUTATIONS[sym_idx]
    inv = SYMMETRY_INVERSE_PERMUTATIONS[sym_idx]
    restored = transform_board(transform_board(board, perm), inv)
    assert np.array_equal(restored, board)


# ---------------------------------------------------------------------------
# Encoded state tests
# ---------------------------------------------------------------------------


def test_identity_encoded_state_is_noop() -> None:
    state = initial_state()
    encoded = encode_state(state).squeeze(0).numpy()
    result = transform_encoded_state(encoded, _IDENTITY)
    np.testing.assert_array_equal(result, encoded)


def test_encoded_state_scalar_planes_unchanged() -> None:
    """Planes 2-7 are scalar broadcasts — symmetry must not alter them."""
    state = initial_state()
    encoded = encode_state(state).squeeze(0).numpy()
    for perm in SYMMETRY_PERMUTATIONS:
        transformed = transform_encoded_state(encoded, perm)
        np.testing.assert_array_equal(transformed[2:], encoded[2:])


@pytest.mark.parametrize("sym_idx", range(8))
def test_encoded_state_invertible(sym_idx: int) -> None:
    # Build a non-trivial state with pieces on the board.
    state = initial_state()
    state = apply_action(state, 0)   # P1 at 0
    state = apply_action(state, 5)   # P2 at 5
    encoded = encode_state(state).squeeze(0).numpy()
    perm = SYMMETRY_PERMUTATIONS[sym_idx]
    inv = SYMMETRY_INVERSE_PERMUTATIONS[sym_idx]
    restored = transform_encoded_state(transform_encoded_state(encoded, perm), inv)
    np.testing.assert_array_almost_equal(restored, encoded)


# ---------------------------------------------------------------------------
# Policy transform tests
# ---------------------------------------------------------------------------


def test_identity_policy_is_noop() -> None:
    policy = np.random.default_rng(0).random(ACTION_SPACE_SIZE).astype(np.float32)
    policy /= policy.sum()
    result = transform_policy(policy, _IDENTITY)
    np.testing.assert_allclose(result, policy, atol=1e-6)


def test_policy_sum_preserved() -> None:
    rng = np.random.default_rng(42)
    policy = rng.random(ACTION_SPACE_SIZE).astype(np.float32)
    policy /= policy.sum()
    for perm in SYMMETRY_PERMUTATIONS:
        transformed = transform_policy(policy, perm)
        assert abs(transformed.sum() - 1.0) < 1e-5, f"Sum not preserved for perm {perm[:5]}"


@pytest.mark.parametrize("sym_idx", range(8))
def test_policy_invertible(sym_idx: int) -> None:
    rng = np.random.default_rng(sym_idx)
    policy = rng.random(ACTION_SPACE_SIZE).astype(np.float32)
    perm = SYMMETRY_PERMUTATIONS[sym_idx]
    inv = SYMMETRY_INVERSE_PERMUTATIONS[sym_idx]
    restored = transform_policy(transform_policy(policy, perm), inv)
    np.testing.assert_allclose(restored, policy, atol=1e-5)


def test_place_action_remapped_correctly() -> None:
    """Under R90, placing at position 0 (outer TL) → placing at position 2 (outer TR)."""
    policy = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
    policy[0] = 1.0
    result = transform_policy(policy, _R90)
    # R90[0] = 2, so the mass moves to action index 2.
    assert result[2] == pytest.approx(1.0)
    assert result.sum() == pytest.approx(1.0)


def test_move_action_remapped_correctly() -> None:
    """Under R90, move 0→1 (TL→TM outer) → move 2→3 (TR→MR outer)."""
    policy = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
    policy[EDGE_INDEX[0, 1]] = 1.0
    result = transform_policy(policy, _R90)
    # After R90: src=0→2, dst=1→3 → new edge (2, 3).
    expected_action = int(EDGE_INDEX[2, 3])
    assert result[expected_action] == pytest.approx(1.0)
    assert result.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Consistency with game rules
# ---------------------------------------------------------------------------


def test_symmetry_consistent_with_legal_place_actions() -> None:
    """In the placing phase all empty positions are legal.

    Under any symmetry, the set of legal place-action indices (0-23) should
    transform consistently with the board permutation.
    """
    state = initial_state()
    # Place a few pieces so the board is non-trivial.
    for action in [0, 5, 2]:
        state = apply_action(state, action)
    # state.must_capture may be True after mill — skip if so.
    if state.must_capture:
        return

    legal = set(a for a in get_legal_actions(state) if a < NUM_PLACE_CAPTURE_ACTIONS)
    board = state.board

    for perm in SYMMETRY_PERMUTATIONS:
        # Build the permuted board (not a full GameState — just check positions).
        new_board = transform_board(board, perm)
        expected_legal = {int(perm[a]) for a in legal}
        actual_empty = {i for i in range(NUM_POSITIONS) if new_board[i] == 0}
        assert expected_legal == actual_empty
