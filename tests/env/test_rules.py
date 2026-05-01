"""Comprehensive tests for the Nine Men's Morris rules engine."""

from __future__ import annotations

import random

import numpy as np

from morris_rl.env.board import NUM_PIECES_PER_PLAYER, NUM_PLACE_CAPTURE_ACTIONS, NUM_POSITIONS
from morris_rl.env.rules import (
    EMPTY,
    MAX_HALFMOVES,
    PLAYER_1,
    PLAYER_2,
    GameState,
    Outcome,
    Phase,
    apply_action,
    forms_mill,
    get_legal_actions,
    get_phase,
    initial_state,
    is_terminal,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _encode_move(src: int, dst: int) -> int:
    return NUM_PLACE_CAPTURE_ACTIONS + src * NUM_POSITIONS + dst


def _make_state(
    board: list[int],
    current_player: int = PLAYER_1,
    p1_hand: int = 0,
    p2_hand: int = 0,
    must_capture: bool = False,
    halfmove_clock: int = 0,
) -> GameState:
    """Construct a GameState directly for testing edge cases."""
    arr = np.array(board, dtype=np.int8)
    state = GameState(
        board=arr,
        current_player=current_player,
        pieces_in_hand=(p1_hand, p2_hand),
        must_capture=must_capture,
        halfmove_clock=halfmove_clock,
    )
    return state


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


def test_initial_board_is_empty() -> None:
    state = initial_state()
    assert (state.board == EMPTY).all()


def test_initial_pieces_in_hand() -> None:
    state = initial_state()
    assert state.pieces_in_hand == (NUM_PIECES_PER_PLAYER, NUM_PIECES_PER_PLAYER)


def test_initial_player_is_1() -> None:
    assert initial_state().current_player == PLAYER_1


def test_initial_not_must_capture() -> None:
    assert not initial_state().must_capture


# ---------------------------------------------------------------------------
# Placement phase
# ---------------------------------------------------------------------------


def test_placement_all_positions_legal_on_empty_board() -> None:
    state = initial_state()
    assert set(get_legal_actions(state)) == set(range(NUM_POSITIONS))


def test_placement_occupied_positions_not_legal() -> None:
    state = apply_action(initial_state(), 0)  # P1 places at 0; P2 to move
    legal = get_legal_actions(state)
    assert 0 not in legal
    assert len(legal) == NUM_POSITIONS - 1


def test_placement_switches_player() -> None:
    state = initial_state()
    state2 = apply_action(state, 5)
    assert state2.current_player == PLAYER_2


def test_placement_decrements_hand() -> None:
    state = apply_action(initial_state(), 3)
    assert state.pieces_in_hand[PLAYER_1 - 1] == NUM_PIECES_PER_PLAYER - 1


def test_placement_resets_halfmove_clock() -> None:
    state = _make_state([0] * NUM_POSITIONS, p1_hand=1, halfmove_clock=10)
    state2 = apply_action(state, 0)
    assert state2.halfmove_clock == 0


# ---------------------------------------------------------------------------
# Mill detection
# ---------------------------------------------------------------------------


def test_forms_mill_basic() -> None:
    board = np.zeros(NUM_POSITIONS, dtype=np.int8)
    board[0] = board[1] = board[2] = PLAYER_1
    assert forms_mill(board, 1, PLAYER_1)


def test_forms_mill_incomplete() -> None:
    board = np.zeros(NUM_POSITIONS, dtype=np.int8)
    board[0] = board[1] = PLAYER_1  # only 2 of 3 in the mill
    assert not forms_mill(board, 1, PLAYER_1)


def test_forms_mill_spoke() -> None:
    board = np.zeros(NUM_POSITIONS, dtype=np.int8)
    board[1] = board[9] = board[17] = PLAYER_2  # spoke mill
    assert forms_mill(board, 9, PLAYER_2)


def test_placement_mill_sets_must_capture() -> None:
    # Place P1 pieces at 0 and 2, then at 1 to complete mill (0,1,2).
    state = initial_state()
    state = apply_action(state, 0)  # P1 → 0
    state = apply_action(state, 3)  # P2 → 3 (harmless)
    state = apply_action(state, 2)  # P1 → 2
    state = apply_action(state, 4)  # P2 → 4 (harmless)
    state = apply_action(state, 1)  # P1 → 1, completes mill (0,1,2)
    assert state.must_capture
    assert state.current_player == PLAYER_1  # still P1's sub-turn


def test_no_mill_no_must_capture() -> None:
    state = initial_state()
    state = apply_action(state, 0)  # P1 → 0
    assert not state.must_capture


# ---------------------------------------------------------------------------
# Capture rules
# ---------------------------------------------------------------------------


def _reach_must_capture_state() -> GameState:
    """Return a state where P1 has just formed mill (0,1,2) and must capture."""
    state = initial_state()
    state = apply_action(state, 0)  # P1 → 0
    state = apply_action(state, 5)  # P2 → 5
    state = apply_action(state, 2)  # P1 → 2
    state = apply_action(state, 6)  # P2 → 6
    state = apply_action(state, 1)  # P1 → 1, mill!
    assert state.must_capture
    return state


def test_capture_removes_opponent_piece() -> None:
    state = _reach_must_capture_state()
    state2 = apply_action(state, 5)  # capture P2 piece at 5
    assert state2.board[5] == EMPTY
    assert not state2.must_capture


def test_capture_switches_to_opponent() -> None:
    state = _reach_must_capture_state()
    state2 = apply_action(state, 5)
    assert state2.current_player == PLAYER_2


def test_capture_resets_halfmove_clock() -> None:
    state = _reach_must_capture_state()
    state2 = apply_action(state, 5)
    assert state2.halfmove_clock == 0


def test_capture_cannot_take_from_opponent_mill() -> None:
    """P1 may not capture a piece that is inside P2's mill, unless all P2 pieces are in mills."""
    board = [0] * NUM_POSITIONS
    # P2 has a mill at (2,3,4) and a loose piece at 6
    board[2] = board[3] = board[4] = PLAYER_2
    board[6] = PLAYER_2
    # P1 just formed a mill (irrelevant positions)
    board[0] = board[1] = board[7] = PLAYER_1
    state = _make_state(board, current_player=PLAYER_1, must_capture=True)
    legal = get_legal_actions(state)
    # 2, 3, 4 are in P2's mill → not capturable; 6 is loose → capturable
    assert 6 in legal
    assert 2 not in legal
    assert 3 not in legal
    assert 4 not in legal


def test_capture_can_take_from_mill_when_all_in_mills() -> None:
    """When all opponent pieces are in mills, any piece may be captured."""
    board = [0] * NUM_POSITIONS
    # P2 has two mills: (0,1,2) and (6,7,0)? No, 0 is shared.
    # Use (0,1,2) and (8,9,10) — no shared position.
    board[0] = board[1] = board[2] = PLAYER_2
    board[8] = board[9] = board[10] = PLAYER_2
    board[4] = board[5] = board[16] = PLAYER_1
    state = _make_state(board, current_player=PLAYER_1, must_capture=True)
    legal = get_legal_actions(state)
    # All P2 pieces are in mills, so all 6 are capturable
    for pos in [0, 1, 2, 8, 9, 10]:
        assert pos in legal


# ---------------------------------------------------------------------------
# Phase transitions
# ---------------------------------------------------------------------------


def test_phase_placing_with_pieces_in_hand() -> None:
    state = initial_state()
    assert get_phase(state, PLAYER_1) == Phase.PLACING


def test_phase_moving_after_all_placed() -> None:
    board = [0] * NUM_POSITIONS
    board[0] = board[1] = board[2] = board[3] = board[4] = PLAYER_1
    board[5] = board[6] = board[7] = board[16] = PLAYER_1  # 9 pieces, no hand
    board[8] = board[10] = board[11] = board[12] = board[13] = PLAYER_2
    board[14] = board[15] = board[17] = board[18] = PLAYER_2
    state = _make_state(board, current_player=PLAYER_1, p1_hand=0, p2_hand=0)
    assert get_phase(state, PLAYER_1) == Phase.MOVING


def test_phase_flying_with_three_pieces() -> None:
    board = [0] * NUM_POSITIONS
    board[0] = board[1] = board[2] = PLAYER_1  # exactly 3 pieces
    state = _make_state(board, current_player=PLAYER_1, p1_hand=0)
    assert get_phase(state, PLAYER_1) == Phase.FLYING


# ---------------------------------------------------------------------------
# Movement
# ---------------------------------------------------------------------------


def test_movement_only_adjacent_positions() -> None:
    board = [0] * NUM_POSITIONS
    board[0] = PLAYER_1  # adjacent to 1 and 7
    board[5] = board[6] = board[7] = board[8] = PLAYER_2  # blockers elsewhere
    state = _make_state(board, current_player=PLAYER_1, p1_hand=0, p2_hand=0)
    legal = get_legal_actions(state)
    assert _encode_move(0, 1) in legal  # 1 is empty
    assert _encode_move(0, 7) not in legal  # 7 is occupied by P2
    # No non-adjacent move should appear
    for act in legal:
        assert act >= NUM_PLACE_CAPTURE_ACTIONS
        idx = act - NUM_PLACE_CAPTURE_ACTIONS
        src, dst = divmod(idx, NUM_POSITIONS)
        assert src == 0


def test_movement_increments_halfmove_clock() -> None:
    board = [0] * NUM_POSITIONS
    board[0] = PLAYER_1
    state = _make_state(board, current_player=PLAYER_1, p1_hand=0, halfmove_clock=3)
    state2 = apply_action(state, _encode_move(0, 1))
    assert state2.halfmove_clock == 4


# ---------------------------------------------------------------------------
# Flying
# ---------------------------------------------------------------------------


def test_flying_can_reach_any_empty_position() -> None:
    board = [0] * NUM_POSITIONS
    board[0] = board[2] = board[4] = PLAYER_1  # 3 pieces → flying
    board[8] = board[9] = board[10] = PLAYER_2
    state = _make_state(board, current_player=PLAYER_1, p1_hand=0, p2_hand=0)
    assert get_phase(state, PLAYER_1) == Phase.FLYING
    legal = get_legal_actions(state)
    # P1 piece at 0 should be able to fly to any empty square
    empty_positions = [p for p in range(NUM_POSITIONS) if board[p] == EMPTY]
    for dst in empty_positions:
        assert _encode_move(0, dst) in legal


# ---------------------------------------------------------------------------
# Terminal states
# ---------------------------------------------------------------------------


def test_terminal_player_with_two_pieces_loses() -> None:
    board = [0] * NUM_POSITIONS
    board[0] = board[1] = PLAYER_1  # P1 only 2 pieces on board, hand empty
    board[5] = board[6] = board[7] = PLAYER_2
    state = _make_state(board, current_player=PLAYER_1, p1_hand=0, p2_hand=0)
    done, outcome = is_terminal(state)
    assert done
    assert outcome == Outcome.PLAYER_2_WINS


def test_terminal_no_legal_moves_loses() -> None:
    # P1 has 4 pieces in MOVING phase (>3 → not flying), all completely blocked.
    # Inner ring: 16(TL)↔17(TM)↔18(TR), exits from 16→23, 17→9, 18→19.
    # Add a 4th P1 piece at 22(BL): neighbours 21 and 23.
    # Block all exits with P2: 23, 9, 19, 21 → P1 has zero legal moves.
    board = [0] * NUM_POSITIONS
    board[16] = board[17] = board[18] = board[22] = PLAYER_1
    board[23] = board[9] = board[19] = board[21] = PLAYER_2
    board[0] = board[1] = board[2] = PLAYER_2  # P2 has enough pieces (7 total)
    state = _make_state(board, current_player=PLAYER_1, p1_hand=0, p2_hand=0)
    assert get_legal_actions(state) == []
    done, outcome = is_terminal(state)
    assert done
    assert outcome == Outcome.PLAYER_2_WINS


def test_not_terminal_during_must_capture() -> None:
    state = _reach_must_capture_state()
    done, outcome = is_terminal(state)
    assert not done
    assert outcome is None


def test_game_continues_with_legal_moves() -> None:
    state = initial_state()
    done, _ = is_terminal(state)
    assert not done


# ---------------------------------------------------------------------------
# Draw conditions
# ---------------------------------------------------------------------------


def test_draw_halfmove_clock_at_limit() -> None:
    board = [0] * NUM_POSITIONS
    board[0] = board[2] = board[4] = PLAYER_1
    board[8] = board[10] = board[12] = PLAYER_2
    state = _make_state(
        board, current_player=PLAYER_1, p1_hand=0, p2_hand=0, halfmove_clock=MAX_HALFMOVES
    )
    done, outcome = is_terminal(state)
    assert done
    assert outcome == Outcome.DRAW


def test_draw_threefold_repetition() -> None:
    # P1 bounces 0↔1, P2 bounces 10↔11.  One full round-trip restores the
    # same (board, player) position.  The starting position is NOT pre-registered
    # in _make_state, so we need THREEFOLD_LIMIT full round-trips to trigger draw.
    from morris_rl.env.rules import THREEFOLD_LIMIT

    board = [0] * NUM_POSITIONS
    board[0] = board[2] = board[4] = PLAYER_1
    board[8] = board[10] = board[12] = PLAYER_2
    s = _make_state(board, current_player=PLAYER_1, p1_hand=0, p2_hand=0)
    for _ in range(THREEFOLD_LIMIT):
        s = apply_action(s, _encode_move(0, 1))   # P1: 0→1
        s = apply_action(s, _encode_move(10, 11))  # P2: 10→11
        s = apply_action(s, _encode_move(1, 0))    # P1: 1→0
        s = apply_action(s, _encode_move(11, 10))  # P2: 11→10  ← original pos
    done, outcome = is_terminal(s)
    assert done
    assert outcome == Outcome.DRAW


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


def test_apply_action_does_not_mutate_input() -> None:
    state = initial_state()
    board_before = state.board.copy()
    hand_before = state.pieces_in_hand
    apply_action(state, 0)
    assert (state.board == board_before).all()
    assert state.pieces_in_hand == hand_before
    assert state.current_player == PLAYER_1


# ---------------------------------------------------------------------------
# Random self-play (1 000 games)
# ---------------------------------------------------------------------------


def _play_random_game(seed: int) -> None:
    rng = random.Random(seed)
    state = initial_state()
    # Safety cap must clear MAX_HALFMOVES + room for the threefold-repetition
    # detector to fire (THREEFOLD_LIMIT × cycle length). At 300/10 a random
    # game of dozens of replays can stretch past 1000 plies before terminating.
    for _ in range(3000):
        done, _ = is_terminal(state)
        if done:
            return
        actions = get_legal_actions(state)
        assert actions, "Non-terminal state has no legal actions"
        state = apply_action(state, rng.choice(actions))
    raise AssertionError("Game did not terminate within 3000 moves")


def test_random_games_1000() -> None:
    for seed in range(1000):
        _play_random_game(seed)
