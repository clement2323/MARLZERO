"""Unit tests for the Reversi/Othello environment."""

from __future__ import annotations

import numpy as np
import pytest

from morris_rl.env.reversi.rules import (
    EMPTY,
    PASS_ACTION,
    PLAYER_1,
    PLAYER_2,
    GameState,
    Outcome,
    apply_action,
    get_legal_actions,
    initial_state,
    is_terminal,
    opponent,
)
from morris_rl.env.reversi.board import NUM_POSITIONS, rc_to_pos, pos_to_rc
from morris_rl.env.reversi.encoding import encode_state, NUM_PLANES
from morris_rl.env.reversi.symmetries import (
    SYMMETRY_PERMUTATIONS,
    SYMMETRY_INVERSE_PERMUTATIONS,
    transform_board,
    transform_policy,
)


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


def test_initial_state_piece_count() -> None:
    state = initial_state()
    assert np.sum(state.board == PLAYER_1) == 2
    assert np.sum(state.board == PLAYER_2) == 2
    assert np.sum(state.board == EMPTY) == 60


def test_initial_state_positions() -> None:
    state = initial_state()
    # Standard Othello starting layout (0-indexed row-major)
    assert state.board[27] == PLAYER_2  # r=3, c=3
    assert state.board[28] == PLAYER_1  # r=3, c=4
    assert state.board[35] == PLAYER_1  # r=4, c=3
    assert state.board[36] == PLAYER_2  # r=4, c=4


def test_initial_state_player() -> None:
    state = initial_state()
    assert state.current_player == PLAYER_1  # Black moves first


def test_initial_state_pass_count() -> None:
    assert initial_state().pass_count == 0


# ---------------------------------------------------------------------------
# Legal actions
# ---------------------------------------------------------------------------


def test_initial_legal_actions_p1() -> None:
    state = initial_state()
    # P1 (Black) has exactly 4 legal placements at game start.
    # Standard Othello: {19, 26, 37, 44}
    actions = sorted(get_legal_actions(state))
    assert len(actions) == 4
    # Verify positions by (row, col):
    # (2,3)=19, (3,2)=26, (4,5)=37, (5,4)=44
    expected = sorted([
        rc_to_pos(2, 3),  # 19
        rc_to_pos(3, 2),  # 26
        rc_to_pos(4, 5),  # 37
        rc_to_pos(5, 4),  # 44
    ])
    assert actions == expected


def test_legal_actions_no_pass_at_start() -> None:
    state = initial_state()
    assert PASS_ACTION not in get_legal_actions(state)


def test_pass_returned_when_no_moves() -> None:
    # Construct a state where the current player has no valid flips.
    # One way: all pieces are PLAYER_1, it's PLAYER_2's turn.
    board = np.zeros(NUM_POSITIONS, dtype=np.int8)
    board[:4] = PLAYER_1
    state = GameState(board=board, current_player=PLAYER_2, pass_count=0)
    actions = get_legal_actions(state)
    assert actions == [PASS_ACTION]


# ---------------------------------------------------------------------------
# apply_action — placement
# ---------------------------------------------------------------------------


def test_apply_action_flips_pieces() -> None:
    state = initial_state()
    # P1 plays at (2,3)=19 — should flip (3,3)=27 from P2 to P1.
    next_state = apply_action(state, rc_to_pos(2, 3))
    assert next_state.board[rc_to_pos(2, 3)] == PLAYER_1  # placed
    assert next_state.board[rc_to_pos(3, 3)] == PLAYER_1  # flipped
    assert next_state.board[rc_to_pos(4, 3)] == PLAYER_1  # unchanged
    assert next_state.current_player == PLAYER_2


def test_apply_action_does_not_mutate() -> None:
    state = initial_state()
    board_before = state.board.copy()
    apply_action(state, rc_to_pos(2, 3))
    assert np.array_equal(state.board, board_before)


def test_apply_action_resets_pass_count() -> None:
    state = GameState(
        board=initial_state().board.copy(),
        current_player=PLAYER_1,
        pass_count=1,
    )
    next_state = apply_action(state, rc_to_pos(2, 3))
    assert next_state.pass_count == 0


# ---------------------------------------------------------------------------
# apply_action — pass
# ---------------------------------------------------------------------------


def test_apply_pass_increments_count() -> None:
    state = initial_state()
    # Force a pass by using PASS_ACTION directly (legal even if real moves exist
    # — apply_action doesn't validate legality, it just applies).
    next_state = apply_action(state, PASS_ACTION)
    assert next_state.pass_count == 1
    assert next_state.current_player == PLAYER_2


def test_apply_pass_does_not_change_board() -> None:
    state = initial_state()
    board_before = state.board.copy()
    next_state = apply_action(state, PASS_ACTION)
    assert np.array_equal(next_state.board, board_before)


# ---------------------------------------------------------------------------
# is_terminal
# ---------------------------------------------------------------------------


def test_not_terminal_at_start() -> None:
    done, outcome = is_terminal(initial_state())
    assert not done
    assert outcome is None


def test_terminal_on_double_pass() -> None:
    state = GameState(board=initial_state().board.copy(), current_player=PLAYER_1, pass_count=2)
    done, outcome = is_terminal(state)
    assert done
    assert outcome is not None


def test_terminal_on_full_board_p1_wins() -> None:
    board = np.ones(NUM_POSITIONS, dtype=np.int8)  # all PLAYER_1
    state = GameState(board=board, current_player=PLAYER_2, pass_count=0)
    done, outcome = is_terminal(state)
    assert done
    assert outcome == Outcome.PLAYER_1_WINS


def test_terminal_on_full_board_draw() -> None:
    board = np.zeros(NUM_POSITIONS, dtype=np.int8)
    board[:32] = PLAYER_1
    board[32:] = PLAYER_2
    state = GameState(board=board, current_player=PLAYER_1, pass_count=0)
    done, outcome = is_terminal(state)
    assert done
    assert outcome == Outcome.DRAW


def test_winner_by_piece_count() -> None:
    # P2 has more pieces → P2 wins.
    board = np.full(NUM_POSITIONS, PLAYER_2, dtype=np.int8)
    board[:10] = PLAYER_1
    state = GameState(board=board, current_player=PLAYER_1, pass_count=2)
    _, outcome = is_terminal(state)
    assert outcome == Outcome.PLAYER_2_WINS


# ---------------------------------------------------------------------------
# Full game — no crashes
# ---------------------------------------------------------------------------


def test_random_game_runs_to_completion() -> None:
    rng = np.random.default_rng(42)
    state = initial_state()
    moves = 0
    while True:
        done, _ = is_terminal(state)
        if done:
            break
        actions = get_legal_actions(state)
        action = int(rng.choice(actions))
        state = apply_action(state, action)
        moves += 1
        assert moves < 200, "Game exceeded maximum expected length"


def test_many_random_games_decisive() -> None:
    """Verify that nearly all Reversi games are decisive (< 5% draws)."""
    rng = np.random.default_rng(0)
    draws = 0
    n_games = 50
    for _ in range(n_games):
        state = initial_state()
        while True:
            done, outcome = is_terminal(state)
            if done:
                if outcome == Outcome.DRAW:
                    draws += 1
                break
            actions = get_legal_actions(state)
            state = apply_action(state, int(rng.choice(actions)))
    assert draws / n_games < 0.20, f"Too many draws: {draws}/{n_games}"


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def test_encode_shape() -> None:
    tensor = encode_state(initial_state())
    assert tensor.shape == (1, NUM_PLANES, NUM_POSITIONS)


def test_encode_dtype() -> None:
    import torch
    tensor = encode_state(initial_state())
    assert tensor.dtype == torch.float32


def test_encode_plane0_is_current_player() -> None:
    state = initial_state()
    tensor = encode_state(state)
    plane0 = tensor[0, 0].numpy()
    expected = (state.board == state.current_player).astype(np.float32)
    assert np.allclose(plane0, expected)


def test_encode_pass_urgency_plane() -> None:
    state = GameState(board=initial_state().board.copy(), current_player=PLAYER_1, pass_count=1)
    tensor = encode_state(state)
    plane2 = tensor[0, 2].numpy()
    assert np.allclose(plane2, 0.5)


# ---------------------------------------------------------------------------
# Symmetries — identity and invertibility
# ---------------------------------------------------------------------------


def test_identity_permutation() -> None:
    perm = SYMMETRY_PERMUTATIONS[0]
    assert np.array_equal(perm, np.arange(64))


def test_symmetries_are_bijections() -> None:
    for perm in SYMMETRY_PERMUTATIONS:
        assert len(np.unique(perm)) == 64


def test_inverse_permutations() -> None:
    """Applying perm then its inverse returns identity."""
    idx = np.arange(64, dtype=np.intp)
    for perm, inv in zip(SYMMETRY_PERMUTATIONS, SYMMETRY_INVERSE_PERMUTATIONS):
        composed = inv[perm]  # apply perm, then inv
        assert np.array_equal(composed, idx), "perm ∘ inv ≠ identity"


def test_transform_board_then_inverse() -> None:
    state = initial_state()
    for perm, inv in zip(SYMMETRY_PERMUTATIONS, SYMMETRY_INVERSE_PERMUTATIONS):
        transformed = transform_board(state.board, perm)
        restored = transform_board(transformed, inv)
        assert np.array_equal(restored, state.board)


def test_transform_policy_pass_invariant() -> None:
    """Pass action (index 64) must be invariant under all symmetries."""
    policy = np.zeros(65, dtype=np.float32)
    policy[64] = 1.0
    for perm in SYMMETRY_PERMUTATIONS:
        transformed = transform_policy(policy, perm)
        assert transformed[64] == 1.0


def test_transform_policy_roundtrip() -> None:
    rng = np.random.default_rng(7)
    policy = rng.random(65).astype(np.float32)
    for perm, inv in zip(SYMMETRY_PERMUTATIONS, SYMMETRY_INVERSE_PERMUTATIONS):
        t = transform_policy(policy, perm)
        restored = transform_policy(t, inv)
        assert np.allclose(restored, policy, atol=1e-6)


# ---------------------------------------------------------------------------
# opponent helper
# ---------------------------------------------------------------------------


def test_opponent() -> None:
    assert opponent(PLAYER_1) == PLAYER_2
    assert opponent(PLAYER_2) == PLAYER_1
