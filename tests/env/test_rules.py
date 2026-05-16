"""Comprehensive tests for the Nine Men's Morris rules engine."""

from __future__ import annotations

import random

import numpy as np

from morris_rl.env.board import NUM_PIECES_PER_PLAYER, NUM_PLACE_CAPTURE_ACTIONS, NUM_POSITIONS
from morris_rl.env.rules import (
    EMPTY,
    MAX_HALFMOVES,
    MAX_TOTAL_HALFMOVES,
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
    opponent,
    random_late_game_state,
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
    total_halfmoves: int = 0,
) -> GameState:
    """Construct a GameState directly for testing edge cases."""
    arr = np.array(board, dtype=np.int8)
    state = GameState(
        board=arr,
        current_player=current_player,
        pieces_in_hand=(p1_hand, p2_hand),
        must_capture=must_capture,
        halfmove_clock=halfmove_clock,
        total_halfmoves=total_halfmoves,
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


def test_phase_remains_moving_with_three_pieces() -> None:
    # No FLYING phase in this variant — a player with 3 pieces is still in
    # MOVING and stays bound to adjacency.
    board = [0] * NUM_POSITIONS
    board[0] = board[1] = board[2] = PLAYER_1  # exactly 3 pieces
    state = _make_state(board, current_player=PLAYER_1, p1_hand=0)
    assert get_phase(state, PLAYER_1) == Phase.MOVING


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
# Three-piece endgame (flying removed in this variant)
# ---------------------------------------------------------------------------


def test_three_pieces_only_adjacent_moves() -> None:
    # With FLYING removed, a player at 3 pieces can still ONLY move to
    # adjacent empties. Non-adjacent destinations remain illegal.
    board = [0] * NUM_POSITIONS
    board[0] = board[2] = board[4] = PLAYER_1  # 3 pieces; would have been flying
    board[8] = board[9] = board[10] = PLAYER_2
    state = _make_state(board, current_player=PLAYER_1, p1_hand=0, p2_hand=0)
    assert get_phase(state, PLAYER_1) == Phase.MOVING
    legal = get_legal_actions(state)
    # Position 0 is only adjacent to 1 and 9 (and 9 is occupied by P2).
    # Position 11 is empty but NOT adjacent to 0 → must be illegal even though
    # under flying rules it would be reachable.
    assert _encode_move(0, 11) not in legal
    assert _encode_move(0, 1) in legal


def test_three_pieces_blocked_loses() -> None:
    # Endgame edge: at 3 pieces, if all of a player's pieces are adjacency-
    # blocked, the "no legal moves" terminal condition wins for the opponent.
    # Without flying this scenario becomes reachable; with flying it could not.
    # P1 sits at outer corners 0, 2, 4 (each has exactly 2 adjacents).
    # P2 occupies 1, 3, 5, 7 to block every P1 adjacency.
    # Adjacencies: 0↔{1,7}, 2↔{1,3}, 4↔{3,5} — all P2-blocked.
    board = [0] * NUM_POSITIONS
    board[0] = board[2] = board[4] = PLAYER_1
    board[1] = board[3] = board[5] = board[7] = PLAYER_2
    state = _make_state(board, current_player=PLAYER_1, p1_hand=0, p2_hand=0)
    done, outcome = is_terminal(state)
    assert done
    assert outcome == Outcome(opponent(PLAYER_1))


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


def test_halfmove_clock_at_limit_decisive() -> None:
    # Reaching MAX_HALFMOVES no-progress clock now returns a decisive outcome
    # (piece-count tiebreak), never DRAW.
    board = [0] * NUM_POSITIONS
    board[0] = board[2] = board[4] = PLAYER_1
    board[8] = board[10] = board[12] = PLAYER_2
    state = _make_state(
        board, current_player=PLAYER_1, p1_hand=0, p2_hand=0, halfmove_clock=MAX_HALFMOVES
    )
    done, outcome = is_terminal(state)
    assert done
    assert outcome != Outcome.DRAW
    assert outcome in (Outcome.PLAYER_1_WINS, Outcome.PLAYER_2_WINS)


def test_threefold_repetition_decisive() -> None:
    # Threefold repetition now resolves via piece-count tiebreak, not DRAW.
    # Use total_halfmoves < MAX_TOTAL_HALFMOVES so the total cap doesn't fire first.
    from morris_rl.env.rules import THREEFOLD_LIMIT

    board = [0] * NUM_POSITIONS
    board[0] = board[2] = board[4] = PLAYER_1
    board[8] = board[10] = board[12] = PLAYER_2
    s = _make_state(board, current_player=PLAYER_1, p1_hand=0, p2_hand=0)
    for _ in range(THREEFOLD_LIMIT):
        s = apply_action(s, _encode_move(0, 1))    # P1: 0→1
        s = apply_action(s, _encode_move(10, 11))  # P2: 10→11
        s = apply_action(s, _encode_move(1, 0))    # P1: 1→0
        s = apply_action(s, _encode_move(11, 10))  # P2: 11→10  ← original pos
    done, outcome = is_terminal(s)
    assert done
    assert outcome != Outcome.DRAW
    assert outcome in (Outcome.PLAYER_1_WINS, Outcome.PLAYER_2_WINS)


def test_total_halfmoves_counter_increments() -> None:
    """total_halfmoves increments on every apply_action call."""
    state = initial_state()
    assert state.total_halfmoves == 0
    # Placement action
    state2 = apply_action(state, 0)
    assert state2.total_halfmoves == 1
    # Another placement
    state3 = apply_action(state2, 1)
    assert state3.total_halfmoves == 2


def test_piece_count_tiebreak_at_cap() -> None:
    """At total_halfmoves == MAX_TOTAL_HALFMOVES, winner is determined by board pieces."""
    board = [0] * NUM_POSITIONS
    board[0] = board[2] = board[4] = board[6] = PLAYER_1   # 4 pieces
    board[8] = board[10] = board[12] = PLAYER_2              # 3 pieces
    state = _make_state(
        board, p1_hand=0, p2_hand=0, total_halfmoves=MAX_TOTAL_HALFMOVES
    )
    done, outcome = is_terminal(state)
    assert done
    assert outcome == Outcome.PLAYER_1_WINS   # P1 has more board pieces


def test_piece_count_tiebreak_p2_wins() -> None:
    """Piece-count tiebreak gives win to P2 when P2 has more pieces."""
    board = [0] * NUM_POSITIONS
    board[0] = board[2] = PLAYER_1                           # 2 pieces
    board[8] = board[10] = board[12] = board[14] = PLAYER_2  # 4 pieces
    state = _make_state(
        board, p1_hand=0, p2_hand=0, total_halfmoves=MAX_TOTAL_HALFMOVES
    )
    done, outcome = is_terminal(state)
    assert done
    assert outcome == Outcome.PLAYER_2_WINS


def test_piece_count_tiebreak_mill_fallback() -> None:
    """Equal board pieces → player with more active mills wins."""
    from morris_rl.env.board import MILLS
    # Use first mill for P1, leave P2 with no mills but same piece count.
    mill = MILLS[0]  # e.g. (0, 1, 2)
    board = [0] * NUM_POSITIONS
    for pos in mill:
        board[pos] = PLAYER_1  # P1 has a mill
    # P2 gets same count of pieces but no mill
    occupied = set(mill)
    p2_count = 0
    for pos in range(NUM_POSITIONS):
        if pos not in occupied and p2_count < len(mill):
            board[pos] = PLAYER_2
            p2_count += 1
    state = _make_state(
        board, p1_hand=0, p2_hand=0, total_halfmoves=MAX_TOTAL_HALFMOVES
    )
    done, outcome = is_terminal(state)
    assert done
    assert outcome == Outcome.PLAYER_1_WINS  # same pieces, P1 has a mill


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
    # With MAX_TOTAL_HALFMOVES=100, no game can last longer than 100 halfmoves.
    for _ in range(200):
        done, _ = is_terminal(state)
        if done:
            return
        actions = get_legal_actions(state)
        assert actions, "Non-terminal state has no legal actions"
        state = apply_action(state, rng.choice(actions))
    raise AssertionError("Game did not terminate within 200 moves")


def test_random_games_1000() -> None:
    for seed in range(1000):
        _play_random_game(seed)


# ---------------------------------------------------------------------------
# random_late_game_state (Phase 3 curriculum)
# ---------------------------------------------------------------------------


def test_random_late_game_state_has_requested_pieces_per_player() -> None:
    rng = np.random.default_rng(0)
    for pieces in (4, 5, 6, 7):
        state = random_late_game_state(rng, pieces_per_player=pieces)
        assert int(np.sum(state.board == PLAYER_1)) == pieces
        assert int(np.sum(state.board == PLAYER_2)) == pieces


def test_random_late_game_state_no_pre_existing_mill() -> None:
    """Sampled state must not already be in must_capture (no pre-existing mill)."""
    rng = np.random.default_rng(1)
    for _ in range(50):
        state = random_late_game_state(rng, pieces_per_player=6)
        assert not state.must_capture
        for player in (PLAYER_1, PLAYER_2):
            for pos in range(NUM_POSITIONS):
                if state.board[pos] == player:
                    assert not forms_mill(state.board, pos, player)


def test_random_late_game_state_has_legal_moves() -> None:
    rng = np.random.default_rng(2)
    for _ in range(50):
        state = random_late_game_state(rng, pieces_per_player=6)
        assert get_legal_actions(state), "Sampled state must not be terminal"


def test_random_late_game_state_hands_empty_and_moving_phase() -> None:
    rng = np.random.default_rng(3)
    state = random_late_game_state(rng, pieces_per_player=6)
    assert state.pieces_in_hand == (0, 0)
    assert get_phase(state, PLAYER_1) == Phase.MOVING
    assert get_phase(state, PLAYER_2) == Phase.MOVING
    assert state.halfmove_clock == 0


def test_random_late_game_state_rejects_invalid_piece_counts() -> None:
    rng = np.random.default_rng(0)
    import pytest
    with pytest.raises(ValueError):
        random_late_game_state(rng, pieces_per_player=2)
    with pytest.raises(ValueError):
        random_late_game_state(rng, pieces_per_player=NUM_POSITIONS)
