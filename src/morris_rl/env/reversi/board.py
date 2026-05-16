"""Reversi/Othello board constants.

Positions 0-63 correspond to an 8x8 grid in row-major order:
    position = row * 8 + col  (row 0 = top, col 0 = left)

Action encoding:
    0-63 : place a piece at that board position (must flip ≥1 opponent piece)
    64   : pass (only legal when the player has no flip moves)
"""

from __future__ import annotations

from typing import Final

NUM_POSITIONS: Final[int] = 64       # 8x8 board
ACTION_SPACE_SIZE: Final[int] = 65   # 64 placement moves + 1 pass (index 64)
NUM_PIECES_PER_PLAYER: Final[int] = 32

# 8 directions as (row_delta, col_delta): N, NE, E, SE, S, SW, W, NW
DIRECTIONS: Final[list[tuple[int, int]]] = [
    (-1, 0), (-1, 1), (0, 1), (1, 1),
    (1, 0), (1, -1), (0, -1), (-1, -1),
]


def pos_to_rc(pos: int) -> tuple[int, int]:
    """Convert flat board index to (row, col)."""
    return divmod(pos, 8)


def rc_to_pos(r: int, c: int) -> int:
    """Convert (row, col) to flat board index."""
    return r * 8 + c
