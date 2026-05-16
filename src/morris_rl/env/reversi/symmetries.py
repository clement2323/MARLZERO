"""8-fold dihedral symmetry group (D4) for the Reversi 8x8 board.

The 8x8 board has full D4 symmetry: 4 rotations × 2 reflections = 8 elements.

Each symmetry is represented as a position permutation array of shape (64,):
    perm[i] = j  means the piece at position i moves to position j.

Transforms apply to:
    - board arrays (shape (64,)): permute position indices
    - encoded states (shape (3, 64)): permute the position axis of planes 0-1
    - policy vectors (shape (65,)): permute placement actions (0-63); pass (64) is invariant

Note: the pass action (index 64) is orientation-independent — it stays at 64
under all symmetries.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt


def _make_rotation_90() -> npt.NDArray[np.intp]:
    """Build permutation for 90° clockwise rotation: (r, c) → (c, 7 - r)."""
    perm = np.zeros(64, dtype=np.intp)
    for r in range(8):
        for c in range(8):
            new_r, new_c = c, 7 - r
            perm[r * 8 + c] = new_r * 8 + new_c
    return perm


def _make_reflection_h() -> npt.NDArray[np.intp]:
    """Build permutation for horizontal reflection: (r, c) → (r, 7 - c)."""
    perm = np.zeros(64, dtype=np.intp)
    for r in range(8):
        for c in range(8):
            perm[r * 8 + c] = r * 8 + (7 - c)
    return perm


def _compose(a: npt.NDArray[np.intp], b: npt.NDArray[np.intp]) -> npt.NDArray[np.intp]:
    """Compose two permutations: apply a first, then b.

    Result[i] = b[a[i]].
    """
    return b[a]


_I = np.arange(64, dtype=np.intp)    # identity
_R = _make_rotation_90()              # 90° CW
_R2 = _compose(_R, _R)               # 180°
_R3 = _compose(_R2, _R)              # 270° CW (= 90° CCW)
_H = _make_reflection_h()            # horizontal flip
_HR = _compose(_R, _H)               # 90° then flip
_HR2 = _compose(_R2, _H)             # 180° then flip
_HR3 = _compose(_R3, _H)             # 270° then flip

# All 8 symmetries in canonical D4 order.
SYMMETRY_PERMUTATIONS: Final[list[npt.NDArray[np.intp]]] = [
    _I,    # identity
    _R,    # 90° clockwise
    _R2,   # 180°
    _R3,   # 270° clockwise (= 90° CCW)
    _H,    # horizontal reflection
    _HR,   # 90° CW then horizontal flip
    _HR2,  # 180° then horizontal flip
    _HR3,  # 270° CW then horizontal flip
]

# Pre-computed inverse permutations (argsort of a bijection is its inverse).
SYMMETRY_INVERSE_PERMUTATIONS: Final[list[npt.NDArray[np.intp]]] = [
    np.argsort(perm).astype(np.intp) for perm in SYMMETRY_PERMUTATIONS
]


def transform_board(board: npt.NDArray[np.int8], perm: npt.NDArray[np.intp]) -> npt.NDArray[np.int8]:
    """Apply a D4 symmetry to a (64,) board array.

    Semantics: new_board[perm[i]] = board[i] for all i.
    """
    new_board = np.empty_like(board)
    new_board[perm] = board
    return new_board


def transform_encoded_state(
    encoded: npt.NDArray[np.float32],
    perm: npt.NDArray[np.intp],
) -> npt.NDArray[np.float32]:
    """Apply a D4 symmetry to an encoded state of shape (3, 64).

    Planes 0-1 (piece positions) are permuted; plane 2 (scalar broadcast) is invariant.
    """
    new_encoded = encoded.copy()
    new_encoded[:2, perm] = encoded[:2]
    return new_encoded


def transform_policy(
    policy: npt.NDArray[np.float32],
    perm: npt.NDArray[np.intp],
) -> npt.NDArray[np.float32]:
    """Apply a D4 symmetry to a policy distribution of shape (65,).

    Placement actions (0-63) are remapped by perm.
    Pass action (64) is invariant under all symmetries.
    """
    new_policy = np.zeros(65, dtype=policy.dtype)
    # Placement actions: action at position i goes to action at perm[i].
    new_policy[perm] = policy[:64]
    # Pass action is orientation-independent.
    new_policy[64] = policy[64]
    return new_policy
