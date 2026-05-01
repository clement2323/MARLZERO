"""8-fold dihedral symmetry group (D4) for the Nine Men's Morris board.

The board's three concentric rings each have 4-fold rotational symmetry and
reflective symmetry, yielding D4 (8 elements). Each symmetry is a permutation
of the 24 board positions.

Transforms apply to:
  - board arrays (shape (24,)): permute position indices
  - encoded states (shape (7, 24)): permute only the position-dependent planes
  - policy vectors (shape (600,)): permute place/capture and move/fly indices
  - value targets: unchanged (outcome is orientation-independent)

The permutation convention: perm[i] = j means the piece at position i moves
to position j after the transform.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from morris_rl.env.board import ACTION_SPACE_SIZE, NUM_PLACE_CAPTURE_ACTIONS, NUM_POSITIONS

# ---------------------------------------------------------------------------
# Permutation tables
# ---------------------------------------------------------------------------
# 90° clockwise rotation.  Outer ring: TL→TR→BR→BL→TL, midpoints follow.
# Middle and inner rings rotate identically within their own cycles.
_R90 = np.array(
    [2, 3, 4, 5, 6, 7, 0, 1, 10, 11, 12, 13, 14, 15, 8, 9, 18, 19, 20, 21, 22, 23, 16, 17],
    dtype=np.intp,
)
_R180 = _R90[_R90]
_R270 = _R90[_R180]

# Left-right (horizontal) reflection across the vertical axis of symmetry.
# The axis passes through TM, BM midpoints of each ring.
_FL = np.array(
    [2, 1, 0, 7, 6, 5, 4, 3, 10, 9, 8, 15, 14, 13, 12, 11, 18, 17, 16, 23, 22, 21, 20, 19],
    dtype=np.intp,
)

# Composed reflections: FL applied after each rotation.
# _FL[perm] computes σ(i) = FL(perm(i)), i.e. apply perm first then FL.
_FL_R90 = _FL[_R90]
_FL_R180 = _FL[_R180]
_FL_R270 = _FL[_R270]

# All 8 symmetries of D4 in a fixed canonical order.
SYMMETRY_PERMUTATIONS: list[npt.NDArray[np.intp]] = [
    np.arange(NUM_POSITIONS, dtype=np.intp),  # identity
    _R90,
    _R180,
    _R270,
    _FL,
    _FL_R90,
    _FL_R180,
    _FL_R270,
]

# Pre-computed inverse permutations (argsort gives the inverse of a bijection).
SYMMETRY_INVERSE_PERMUTATIONS: list[npt.NDArray[np.intp]] = [
    np.argsort(perm).astype(np.intp) for perm in SYMMETRY_PERMUTATIONS
]


# ---------------------------------------------------------------------------
# Transform functions
# ---------------------------------------------------------------------------


def transform_board(board: npt.NDArray[Any], perm: npt.NDArray[np.intp]) -> npt.NDArray[Any]:
    """Apply a board symmetry to a (24,) board array.

    new_board[perm[i]] = board[i]  for all i.
    """
    new_board = np.empty_like(board)
    new_board[perm] = board
    return new_board


def transform_encoded_state(
    encoded: npt.NDArray[np.float32], perm: npt.NDArray[np.intp]
) -> npt.NDArray[np.float32]:
    """Apply a board symmetry to an encoded state of shape (7, 24).

    Only planes 0-1 (piece positions) are position-dependent; planes 2-6 are
    scalar broadcasts that are invariant under symmetry.
    """
    new_encoded = encoded.copy()
    new_encoded[:2, perm] = encoded[:2]
    return new_encoded


def transform_policy(
    policy: npt.NDArray[np.float32], perm: npt.NDArray[np.intp]
) -> npt.NDArray[np.float32]:
    """Apply a board symmetry to a policy distribution of shape (ACTION_SPACE_SIZE,).

    Place/capture actions (indices 0-23) are remapped by perm.
    Move/fly actions (indices 24-599, encoded as 24 + src*24 + dst) are remapped
    by applying perm to both src and dst.
    """
    n = NUM_POSITIONS
    new_policy = np.zeros(ACTION_SPACE_SIZE, dtype=policy.dtype)

    # Place/capture: action at position i goes to action at perm[i].
    new_policy[perm] = policy[:NUM_PLACE_CAPTURE_ACTIONS]

    # Move/fly: action (src→dst) maps to (perm[src]→perm[dst]).
    # Reshaping to (24, 24) lets us permute rows and columns simultaneously.
    # new[perm[s], perm[d]] = old[s, d]  ⟺  new[i, j] = old[inv[i], inv[j]]
    perm_inv = np.argsort(perm)
    move_matrix = policy[NUM_PLACE_CAPTURE_ACTIONS:].reshape(n, n)
    new_policy[NUM_PLACE_CAPTURE_ACTIONS:] = move_matrix[np.ix_(perm_inv, perm_inv)].flatten()

    return new_policy
