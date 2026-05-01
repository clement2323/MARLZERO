"""Nine Men's Morris rules engine.

Public API:
    initial_state()         -> GameState
    get_legal_actions(s)    -> list[int]   (action indices into [0, ACTION_SPACE_SIZE))
    apply_action(s, a)      -> GameState   (never mutates s)
    is_terminal(s)          -> (bool, Outcome | None)
    get_phase(s, player)    -> Phase
    forms_mill(board, pos, player) -> bool
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Final

import numpy as np

from morris_rl.env.board import (
    ADJACENCY,
    MILLS_BY_POSITION,
    NUM_PIECES_PER_PLAYER,
    NUM_PLACE_CAPTURE_ACTIONS,
    NUM_POSITIONS,
)

PLAYER_1: Final[int] = 1
PLAYER_2: Final[int] = 2
EMPTY: Final[int] = 0

MAX_HALFMOVES: Final[int] = 300
THREEFOLD_LIMIT: Final[int] = 10


class Phase(IntEnum):
    PLACING = 0
    MOVING = 1
    FLYING = 2


class Outcome(IntEnum):
    DRAW = 0
    PLAYER_1_WINS = 1
    PLAYER_2_WINS = 2


@dataclass
class GameState:
    """Complete game state. Treat as immutable — use apply_action to advance."""

    board: np.ndarray  # shape (24,) int8; values in {0, 1, 2}
    current_player: int  # 1 or 2
    pieces_in_hand: tuple[int, int]  # (p1_hand, p2_hand)
    must_capture: bool  # True when the current player just formed a mill
    halfmove_clock: int  # resets on placement or capture; draw at MAX_HALFMOVES
    position_counts: dict[tuple[int, ...], int] = field(default_factory=dict)

    def copy(self) -> GameState:
        """Deep copy — board array and position_counts dict are duplicated."""
        return GameState(
            board=self.board.copy(),
            current_player=self.current_player,
            pieces_in_hand=self.pieces_in_hand,
            must_capture=self.must_capture,
            halfmove_clock=self.halfmove_clock,
            position_counts=dict(self.position_counts),
        )


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def opponent(player: int) -> int:
    """Return the other player."""
    return PLAYER_2 if player == PLAYER_1 else PLAYER_1


def pieces_on_board(board: np.ndarray, player: int) -> int:
    """Count player's pieces currently on the board."""
    return int(np.sum(board == player))


def get_phase(state: GameState, player: int) -> Phase:
    """Return the current phase for *player* (may differ between players)."""
    if state.pieces_in_hand[player - 1] > 0:
        return Phase.PLACING
    return Phase.FLYING if pieces_on_board(state.board, player) == 3 else Phase.MOVING


def forms_mill(board: np.ndarray, position: int, player: int) -> bool:
    """Return True if player's piece at *position* is part of a complete mill.

    Assumes board[position] == player already (call after updating the board).
    """
    for mill in MILLS_BY_POSITION[position]:
        if all(board[p] == player for p in mill):
            return True
    return False


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------


def initial_state() -> GameState:
    """Return a fresh game state at the very start of a game."""
    state = GameState(
        board=np.zeros(NUM_POSITIONS, dtype=np.int8),
        current_player=PLAYER_1,
        pieces_in_hand=(NUM_PIECES_PER_PLAYER, NUM_PIECES_PER_PLAYER),
        must_capture=False,
        halfmove_clock=0,
    )
    _register_position(state)
    return state


def get_legal_actions(state: GameState) -> list[int]:
    """Return all legal action indices for the current state."""
    if state.must_capture:
        return _legal_capture_actions(state)
    phase = get_phase(state, state.current_player)
    if phase == Phase.PLACING:
        return [p for p in range(NUM_POSITIONS) if state.board[p] == EMPTY]
    if phase == Phase.MOVING:
        return _legal_move_actions(state)
    return _legal_fly_actions(state)


def apply_action(state: GameState, action: int) -> GameState:
    """Return the state that results from applying *action*. Never mutates state."""
    next_state = state.copy()
    if state.must_capture:
        _apply_capture(next_state, action)
    elif action < NUM_PLACE_CAPTURE_ACTIONS:
        _apply_placement(next_state, action)
    else:
        _apply_move(next_state, action)
    return next_state


def is_terminal(state: GameState) -> tuple[bool, Outcome | None]:
    """Return (done, outcome). outcome is None when not done."""
    # The must_capture sub-turn is not a terminal check point.
    if state.must_capture:
        return False, None

    key = _position_key(state)
    if state.position_counts.get(key, 0) >= THREEFOLD_LIMIT:
        return True, Outcome.DRAW

    if state.halfmove_clock >= MAX_HALFMOVES:
        return True, Outcome.DRAW

    player = state.current_player
    if state.pieces_in_hand[player - 1] == 0:
        if pieces_on_board(state.board, player) < 3:
            return True, Outcome(opponent(player))

    if not get_legal_actions(state):
        return True, Outcome(opponent(player))

    return False, None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _all_opponent_in_mills(state: GameState) -> bool:
    """True if every opponent piece is part of at least one mill.

    When this holds, the current player may capture from any opponent mill.
    """
    opp = opponent(state.current_player)
    return all(
        forms_mill(state.board, pos, opp) for pos in range(NUM_POSITIONS) if state.board[pos] == opp
    )


def _legal_capture_actions(state: GameState) -> list[int]:
    opp = opponent(state.current_player)
    all_in_mills = _all_opponent_in_mills(state)
    actions = []
    for pos in range(NUM_POSITIONS):
        if state.board[pos] != opp:
            continue
        # Cannot capture a piece that is in a mill — unless all opponent pieces are in mills.
        if not all_in_mills and forms_mill(state.board, pos, opp):
            continue
        actions.append(pos)
    return actions


def _legal_move_actions(state: GameState) -> list[int]:
    player = state.current_player
    actions = []
    for src in range(NUM_POSITIONS):
        if state.board[src] != player:
            continue
        for dst in ADJACENCY[src]:
            if state.board[dst] == EMPTY:
                actions.append(NUM_PLACE_CAPTURE_ACTIONS + src * NUM_POSITIONS + dst)
    return actions


def _legal_fly_actions(state: GameState) -> list[int]:
    player = state.current_player
    sources = [p for p in range(NUM_POSITIONS) if state.board[p] == player]
    targets = [p for p in range(NUM_POSITIONS) if state.board[p] == EMPTY]
    return [
        NUM_PLACE_CAPTURE_ACTIONS + src * NUM_POSITIONS + dst for src in sources for dst in targets
    ]


def _apply_placement(state: GameState, position: int) -> None:
    player = state.current_player
    state.board[position] = player
    hand = list(state.pieces_in_hand)
    hand[player - 1] -= 1
    state.pieces_in_hand = (hand[0], hand[1])
    state.halfmove_clock = 0  # placement always resets the no-progress clock

    if forms_mill(state.board, position, player):
        state.must_capture = True
    else:
        state.current_player = opponent(player)
        _register_position(state)


def _apply_move(state: GameState, action: int) -> None:
    player = state.current_player
    idx = action - NUM_PLACE_CAPTURE_ACTIONS
    src, dst = divmod(idx, NUM_POSITIONS)
    state.board[src] = EMPTY
    state.board[dst] = player
    state.halfmove_clock += 1

    if forms_mill(state.board, dst, player):
        state.must_capture = True
    else:
        state.current_player = opponent(player)
        _register_position(state)


def _apply_capture(state: GameState, position: int) -> None:
    state.board[position] = EMPTY
    state.must_capture = False
    state.halfmove_clock = 0  # capture resets the no-progress clock
    state.current_player = opponent(state.current_player)
    _register_position(state)


def _position_key(state: GameState) -> tuple[int, ...]:
    board_list: list[int] = state.board.tolist()
    return (
        *board_list,
        state.current_player,
        int(state.must_capture),
        *state.pieces_in_hand,
    )


def _register_position(state: GameState) -> None:
    key = _position_key(state)
    state.position_counts[key] = state.position_counts.get(key, 0) + 1
