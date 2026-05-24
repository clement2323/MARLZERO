"""Rich 8-feature evaluation for Morris minimax warmup.

The default `_heuristic` in `eval/baselines.py` (material + mills only) is too
sparse to produce decisive games at depth 5: opposing minimax agents settle
into perpetual mirroring with frequent halfmove caps.

This module adds six topology-aware features (potential mills, mobility, forks,
crossroad control, blocked pieces, plus phase-dependent weighting) following
the classical Morris evaluation literature (Gasser 1996, Lasker, muehle).

All weights are module-level constants. Tweak for ablations or expose via CLI
later if needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from morris_rl.env.board import (
    ADJACENCY,
    MILLS,
    MILLS_BY_POSITION,
    NUM_POSITIONS,
)
from morris_rl.env.rules import (
    EMPTY,
    GameState,
    Phase,
    get_legal_actions,
    get_phase,
    opponent,
    pieces_on_board,
)


@dataclass(frozen=True)
class HeuristicWeights:
    """Per-phase weights for the eight evaluation components."""

    material: float
    closed_mills: float
    potential_mills: float
    mobility: float
    forks: float
    crossroads: float
    blocked: float


WEIGHTS_PLACEMENT = HeuristicWeights(
    material=1.0,
    closed_mills=0.4,
    # potential_mills lowered (0.25 → 0.15): the previous ratio of 0.625 vs
    # closed_mills triggered the horizon effect — the agent rushed to set up
    # 2-of-3 mills at leaf nodes, which the opponent blocked at the next ply.
    # Lit. ratio ~0.3-0.5 vs closed; 0.375 here.
    potential_mills=0.15,
    mobility=0.05,
    forks=0.7,
    crossroads=0.08,
    blocked=-0.15,
)

WEIGHTS_MOVEMENT = HeuristicWeights(
    material=1.2,
    closed_mills=0.5,
    # See WEIGHTS_PLACEMENT.potential_mills comment.
    potential_mills=0.20,
    mobility=0.10,
    forks=0.8,
    crossroads=0.12,
    blocked=-0.20,
)


# Pre-computed indicator: which positions have degree 4 (the "crossroads" —
# middle-ring midpoints 9, 11, 13, 15). They open more move-tree branches and
# the literature treats them as strategically valuable.
_DEGREE_4_POSITIONS: tuple[int, ...] = tuple(
    p for p in range(NUM_POSITIONS) if len(ADJACENCY[p]) == 4
)


def _count_closed_mills(board: np.ndarray, player: int) -> int:
    return sum(1 for m in MILLS if all(board[p] == player for p in m))


def _count_potential_mills(board: np.ndarray, player: int) -> int:
    """Mills with exactly two `player` pieces and one empty cell.

    These are immediate threats: a single placement (or move into the empty
    cell) closes the mill on the next ply, triggering a capture.
    """
    count = 0
    for m in MILLS:
        own = 0
        empties = 0
        for p in m:
            v = int(board[p])
            if v == player:
                own += 1
            elif v == EMPTY:
                empties += 1
        if own == 2 and empties == 1:
            count += 1
    return count


def _count_forks(board: np.ndarray, player: int) -> int:
    """Empty positions where placing `player` would simultaneously create
    two or more potential mills.

    A fork forces the opponent to choose which threat to block — they cannot
    block both, so at least one mill closes on the following ply.

    Cost: O(NUM_POSITIONS × len(MILLS_BY_POSITION[p]) × 3) ≈ 24 × 3 × 3 = 216
    cell checks per call. Negligible compared with the 11-ply game tree.
    """
    forks = 0
    for pos in range(NUM_POSITIONS):
        if int(board[pos]) != EMPTY:
            continue
        threats = 0
        for mill in MILLS_BY_POSITION[pos]:
            own = 0
            empties = 0
            for p in mill:
                if p == pos:
                    continue
                v = int(board[p])
                if v == player:
                    own += 1
                elif v == EMPTY:
                    empties += 1
            # After hypothetically placing at `pos`, this mill has `own + 1`
            # player pieces. If that count equals 2 (and the remaining cell
            # is empty), it becomes a potential mill.
            if own == 1 and empties == 1:
                threats += 1
        if threats >= 2:
            forks += 1
    return forks


def _count_crossroad_control(board: np.ndarray, player: int) -> int:
    return sum(1 for p in _DEGREE_4_POSITIONS if int(board[p]) == player)


def _count_blocked_pieces(state: GameState, player: int) -> int:
    """Movement-phase only: pieces with no legal destination (all adjacent
    cells occupied).

    Returns 0 during placement phase — irrelevant when both players still
    drop pieces from hand.
    """
    if get_phase(state, player) != Phase.MOVING:
        return 0
    board = state.board
    blocked = 0
    for pos in range(NUM_POSITIONS):
        if int(board[pos]) != player:
            continue
        if all(int(board[n]) != EMPTY for n in ADJACENCY[pos]):
            blocked += 1
    return blocked


def _swap_player_state(state: GameState) -> GameState:
    """Return a shallow-copy of `state` with `current_player` flipped.

    Used to evaluate the opponent's mobility from their POV. The board and
    hand are not duplicated since `get_legal_actions` only reads them.
    """
    swapped = state.copy()
    swapped.current_player = opponent(state.current_player)
    # Cancel a pending must_capture: that sub-turn belongs to the original
    # current player, the opponent has no equivalent. We don't deeply model
    # this — we only want the opponent's reasonable move count.
    swapped.must_capture = False
    return swapped


def rich_heuristic(state: GameState) -> float:
    """Static evaluation from the current player's perspective.

    Returns a signed float (high = favorable for current player). The eight
    components and their phase-dependent weights are documented in the
    `HeuristicWeights` dataclass above.
    """
    me = state.current_player
    opp = opponent(me)
    board = state.board

    own_total = pieces_on_board(board, me) + state.pieces_in_hand[me - 1]
    opp_total = pieces_on_board(board, opp) + state.pieces_in_hand[opp - 1]
    material_diff = float(own_total - opp_total)

    own_closed = _count_closed_mills(board, me)
    opp_closed = _count_closed_mills(board, opp)
    closed_diff = float(own_closed - opp_closed)

    own_potential = _count_potential_mills(board, me)
    opp_potential = _count_potential_mills(board, opp)
    potential_diff = float(own_potential - opp_potential)

    own_mobility = len(get_legal_actions(state))
    opp_mobility = len(get_legal_actions(_swap_player_state(state)))
    mobility_diff = float(own_mobility - opp_mobility)

    own_forks = _count_forks(board, me)
    opp_forks = _count_forks(board, opp)
    fork_diff = float(own_forks - opp_forks)

    own_crossroads = _count_crossroad_control(board, me)
    opp_crossroads = _count_crossroad_control(board, opp)
    crossroad_diff = float(own_crossroads - opp_crossroads)

    own_blocked = _count_blocked_pieces(state, me)
    opp_blocked = _count_blocked_pieces(state, opp)
    blocked_diff = float(own_blocked - opp_blocked)

    phase = get_phase(state, me)
    w = WEIGHTS_PLACEMENT if phase == Phase.PLACING else WEIGHTS_MOVEMENT

    return (
        w.material * material_diff
        + w.closed_mills * closed_diff
        + w.potential_mills * potential_diff
        + w.mobility * mobility_diff
        + w.forks * fork_diff
        + w.crossroads * crossroad_diff
        + w.blocked * blocked_diff
    )
