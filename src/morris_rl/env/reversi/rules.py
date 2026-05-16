"""Reversi/Othello rules engine.

Public API:
    initial_state()          -> GameState
    get_legal_actions(s)     -> list[int]   (action indices into [0, ACTION_SPACE_SIZE))
    apply_action(s, a)       -> GameState   (never mutates s)
    is_terminal(s)           -> (bool, Outcome | None)
    opponent(player)         -> int

Action encoding:
    0-63 : place piece at that position (must flip ≥1 opponent piece)
    64   : pass (only legal when no flip moves are available)

pass_count tracks consecutive passes. Two consecutive passes => terminal.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Final

import numpy as np

from morris_rl.env.reversi.board import (
    DIRECTIONS,
    NUM_POSITIONS,
    pos_to_rc,
    rc_to_pos,
)

PLAYER_1: Final[int] = 1  # Black (moves first)
PLAYER_2: Final[int] = 2  # White
EMPTY: Final[int] = 0

PASS_ACTION: Final[int] = 64


class Outcome(IntEnum):
    DRAW = 0
    PLAYER_1_WINS = 1
    PLAYER_2_WINS = 2


@dataclass
class GameState:
    """Complete Reversi game state. Treat as immutable — use apply_action to advance.

    Attributes:
        board: Shape (64,) int8. Values: 0=empty, 1=P1 (black), 2=P2 (white).
        current_player: 1 or 2, the player to move next.
        pass_count: number of consecutive passes. Terminal when ≥ 2.
    """

    board: np.ndarray      # (64,) int8
    current_player: int    # 1 or 2
    pass_count: int        # consecutive passes; game ends when this reaches 2

    def copy(self) -> "GameState":
        return GameState(
            board=self.board.copy(),
            current_player=self.current_player,
            pass_count=self.pass_count,
        )


def opponent(player: int) -> int:
    """Return the other player."""
    return PLAYER_2 if player == PLAYER_1 else PLAYER_1


def initial_state() -> GameState:
    """Return the standard Othello starting position.

    Standard layout (0-indexed, row-major):
        pos 27 (r=3, c=3) = White (P2)
        pos 28 (r=3, c=4) = Black (P1)
        pos 35 (r=4, c=3) = Black (P1)
        pos 36 (r=4, c=4) = White (P2)
    Black (P1) moves first.
    """
    board = np.zeros(NUM_POSITIONS, dtype=np.int8)
    board[27] = PLAYER_2  # r=3, c=3 — White
    board[28] = PLAYER_1  # r=3, c=4 — Black
    board[35] = PLAYER_1  # r=4, c=3 — Black
    board[36] = PLAYER_2  # r=4, c=4 — White
    return GameState(board=board, current_player=PLAYER_1, pass_count=0)


def _flips_in_direction(
    board: np.ndarray,
    pos: int,
    player: int,
    dr: int,
    dc: int,
) -> list[int]:
    """Return positions to flip in direction (dr, dc) if valid, else empty list.

    Walks in (dr, dc) from pos. Collects consecutive opponent pieces. If the
    walk terminates on a friendly piece, those opponent pieces are returned.
    If it hits empty or the board edge, returns [].
    """
    r, c = pos_to_rc(pos)
    opp = opponent(player)
    to_flip: list[int] = []
    r += dr
    c += dc
    while 0 <= r < 8 and 0 <= c < 8:
        p = board[rc_to_pos(r, c)]
        if p == opp:
            to_flip.append(rc_to_pos(r, c))
        elif p == player:
            return to_flip  # valid bracket found
        else:
            return []       # empty square breaks the chain
        r += dr
        c += dc
    return []  # hit the board edge without a closing piece


def _get_flips(board: np.ndarray, pos: int, player: int) -> list[int]:
    """Return all positions that would be flipped by placing player at pos.

    Returns [] if the move is illegal (pos occupied, or no flips in any direction).
    """
    if board[pos] != EMPTY:
        return []
    all_flips: list[int] = []
    for dr, dc in DIRECTIONS:
        all_flips.extend(_flips_in_direction(board, pos, player, dr, dc))
    return all_flips


def get_legal_actions(state: GameState) -> list[int]:
    """Return all legal action indices for the current player.

    Every empty position that would flip at least one opponent piece is legal.
    If none exist, returns [PASS_ACTION] (index 64). The pass action is always
    the sole element of the list when it appears — it is never combined with
    regular moves.
    """
    moves = [
        pos
        for pos in range(NUM_POSITIONS)
        if state.board[pos] == EMPTY
        and _get_flips(state.board, pos, state.current_player)
    ]
    return moves if moves else [PASS_ACTION]


def apply_action(state: GameState, action: int) -> GameState:
    """Return the state that results from applying action. Never mutates state.

    For action == PASS_ACTION: increments pass_count and switches player.
    For a placement action: flips all captured pieces, resets pass_count.
    """
    new_board = state.board.copy()
    pass_count = state.pass_count

    if action == PASS_ACTION:
        pass_count += 1
    else:
        flips = _get_flips(new_board, action, state.current_player)
        new_board[action] = state.current_player
        for pos in flips:
            new_board[pos] = state.current_player
        pass_count = 0  # any real move resets the consecutive-pass counter

    return GameState(
        board=new_board,
        current_player=opponent(state.current_player),
        pass_count=pass_count,
    )


def is_terminal(state: GameState) -> tuple[bool, Outcome | None]:
    """Return (done, outcome). outcome is None when the game is ongoing.

    Terminal conditions:
    - Two consecutive passes (both players have no legal flip moves).
    - Board is completely full.

    Winner is determined by piece count; equal counts is a draw.
    """
    board_full = not np.any(state.board == EMPTY)
    if state.pass_count >= 2 or board_full:
        p1_count = int(np.sum(state.board == PLAYER_1))
        p2_count = int(np.sum(state.board == PLAYER_2))
        if p1_count > p2_count:
            return True, Outcome.PLAYER_1_WINS
        elif p2_count > p1_count:
            return True, Outcome.PLAYER_2_WINS
        else:
            return True, Outcome.DRAW
    return False, None
