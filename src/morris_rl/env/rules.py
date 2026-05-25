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
    EDGE_INDEX,
    MILLS,
    MILLS_BY_POSITION,
    MOVE_EDGES,
    NUM_PIECES_PER_PLAYER,
    NUM_PLACE_CAPTURE_ACTIONS,
    NUM_POSITIONS,
)

PLAYER_1: Final[int] = 1
PLAYER_2: Final[int] = 2
EMPTY: Final[int] = 0

MAX_HALFMOVES: Final[int] = 50         # no-capture clock — chess-style 50-move rule.
# `halfmove_clock` counts halfmoves since the last *capture*. Hitting 50 = DRAW,
# no piece_count tiebreak — these samples should be discarded from the buffer
# via self_play.discard_timeout_games=true so the network is never trained on
# "value=0 on a non-decisive position" (the lazy-mean attractor).
MAX_TOTAL_HALFMOVES: Final[int] = 300  # absolute ceiling, safety net only.
# With MAX_HALFMOVES=50 capturing-progress rule active, games can theoretically
# stretch (50 plies × ~18 max captures) but in practice end well before this
# ceiling once pieces run low. Hitting this falls back to _piece_count_winner.
THREEFOLD_LIMIT: Final[int] = 3
# No-repetition rule removed: repetitions are not filtered out of get_legal_actions.
# The network learns to avoid useless cycles on its own (visiting a cycle gives
# the same value at each step → MCTS Q has no gradient → exploration moves
# elsewhere). Threefold-repetition termination via THREEFOLD_LIMIT in is_terminal
# + piece_count_winner still resolves genuine draws when the game gets stuck.


class Phase(IntEnum):
    PLACING = 0
    MOVING = 1
    # No FLYING phase: this variant keeps the adjacency constraint even at 3
    # pieces. A player with no legal move (blocked by adjacency) loses.


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
    halfmove_clock: int  # resets on placement or capture; kept for legacy metrics
    total_halfmoves: int = 0  # absolute game length — never reset; cap at MAX_TOTAL_HALFMOVES
    position_counts: dict[tuple[int, ...], int] = field(default_factory=dict)

    def copy(self) -> GameState:
        """Deep copy — board array and position_counts dict are duplicated."""
        return GameState(
            board=self.board.copy(),
            current_player=self.current_player,
            pieces_in_hand=self.pieces_in_hand,
            must_capture=self.must_capture,
            halfmove_clock=self.halfmove_clock,
            total_halfmoves=self.total_halfmoves,
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
    return Phase.MOVING


def compute_aux_features(state: GameState) -> tuple[float, float]:
    """Return (mill_diff, pieces_diff) from the current player's perspective.

    Both quantities are signed (own - opp) and deterministic given the state.
    Used as targets for the auxiliary heads (KataGo-style multi-task learning)
    to give the trunk a dense, noise-free training signal alongside the noisy
    end-of-game value target.
    """
    me = state.current_player
    opp_ = opponent(me)
    own_mills = sum(1 for m in MILLS if all(state.board[p] == me for p in m))
    opp_mills = sum(1 for m in MILLS if all(state.board[p] == opp_ for p in m))
    own_pieces = pieces_on_board(state.board, me)
    opp_pieces = pieces_on_board(state.board, opp_)
    return float(own_mills - opp_mills), float(own_pieces - opp_pieces)


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


def random_late_game_state(
    rng: np.random.Generator,
    pieces_per_player: int = 6,
    max_attempts: int = 100,
) -> GameState:
    """Return a random mid/late-game position with both hands empty.

    Used by the curriculum feature to bias self-play toward decisive
    endgame positions: most pieces are already on the board, so games end
    in a clear win/loss far more often than from the empty board (which
    drifts toward the draw attractor in early training).

    Constraints on the sampled state:
    - exactly *pieces_per_player* pieces per side, in moving phase
    - no pre-existing mill (avoids waking up in the must_capture sub-turn)
    - the chosen mover has at least one legal action

    If no valid configuration is found in *max_attempts* tries, falls back
    to ``initial_state()``. Returning rather than raising keeps self-play
    workers from crashing on a degenerate run.
    """
    if pieces_per_player < 3 or 2 * pieces_per_player > NUM_POSITIONS:
        raise ValueError(
            f"pieces_per_player must be in [3, {NUM_POSITIONS // 2}], got {pieces_per_player}"
        )
    n_pieces = 2 * pieces_per_player
    for _ in range(max_attempts):
        positions = rng.choice(NUM_POSITIONS, size=n_pieces, replace=False)
        p1_positions = positions[:pieces_per_player]
        p2_positions = positions[pieces_per_player:]
        board = np.zeros(NUM_POSITIONS, dtype=np.int8)
        board[p1_positions] = PLAYER_1
        board[p2_positions] = PLAYER_2
        # Reject configurations that would put the game directly into
        # must_capture (any pre-existing mill).
        if any(forms_mill(board, int(pos), PLAYER_1) for pos in p1_positions):
            continue
        if any(forms_mill(board, int(pos), PLAYER_2) for pos in p2_positions):
            continue
        current_player = PLAYER_1 if rng.random() < 0.5 else PLAYER_2
        candidate = GameState(
            board=board,
            current_player=current_player,
            pieces_in_hand=(0, 0),
            must_capture=False,
            halfmove_clock=0,
        )
        # Reject states where the mover has no legal action — that would be
        # an immediate terminal (loss by no_legal_moves), 0-ply game.
        if not get_legal_actions(candidate):
            continue
        _register_position(candidate)
        return candidate
    return initial_state()


def get_legal_actions(state: GameState) -> list[int]:
    """Return all legal action indices for the current state.

    Three branches depending on the sub-turn:
      - ``must_capture``: capture actions, with the mill-protection rule.
      - PLACING phase: every empty position is a legal placement.
      - MOVING phase: every adjacency edge from an own piece to an empty cell.

    Repetitions are NOT filtered out here — the network learns to avoid
    pointless cycles via the value head, and threefold-repetition termination
    in is_terminal handles genuinely stuck games.
    """
    if state.must_capture:
        return _legal_capture_actions(state)
    phase = get_phase(state, state.current_player)
    if phase == Phase.PLACING:
        return [p for p in range(NUM_POSITIONS) if state.board[p] == EMPTY]
    return _legal_move_actions(state)


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
    """Return (done, outcome). outcome is None when not done.

    Decisive verdicts:
      - opponent reduced to <3 pieces  →  current player wins
      - current player has no legal move  →  opponent wins

    Draw verdicts:
      - total-halfmove cap reached
      - threefold repetition
      - 50-halfmove no-capture clock
    """
    # The must_capture sub-turn is not a terminal check point.
    if state.must_capture:
        return False, None

    if state.total_halfmoves >= MAX_TOTAL_HALFMOVES:
        # Safety-net ceiling: in practice almost never hit thanks to the
        # no-capture clock below. Hitting it now resolves to a clean DRAW
        # rather than a piece-count tiebreak — this avoids minimax (and
        # self-play) preferring to drag the game to the cap when it has a
        # small material edge. The piece_count tiebreak heuristic stayed
        # in the code (see _piece_count_winner) in case we revert later.
        return True, Outcome.DRAW

    key = _position_key(state)
    if state.position_counts.get(key, 0) >= THREEFOLD_LIMIT:
        # Threefold repetition: genuine stuck-game signal → DRAW, not tiebreak.
        # Combined with self_play.discard_timeout_games=true these samples are
        # not used for training.
        return True, Outcome.DRAW

    if state.halfmove_clock >= MAX_HALFMOVES:
        # 50-halfmove no-capture rule (chess-style). No progress in either
        # player's piece count for MAX_HALFMOVES halfmoves → DRAW. Samples
        # discarded when self_play.discard_timeout_games=true.
        return True, Outcome.DRAW

    player = state.current_player
    if state.pieces_in_hand[player - 1] == 0:
        if pieces_on_board(state.board, player) < 3:
            return True, Outcome(opponent(player))

    if not get_legal_actions(state):
        return True, Outcome(opponent(player))

    return False, None


def _piece_count_winner(state: GameState) -> Outcome:
    """Decisive outcome at game cap — never returns DRAW.

    Priority: 1) board pieces  2) active mills  3) P1 wins.
    At the 100-halfmove cap both hands are always empty (movement phase),
    so total pieces == board pieces.
    """
    b1 = pieces_on_board(state.board, PLAYER_1)
    b2 = pieces_on_board(state.board, PLAYER_2)
    if b1 != b2:
        return Outcome.PLAYER_1_WINS if b1 > b2 else Outcome.PLAYER_2_WINS
    m1 = sum(1 for m in MILLS if all(state.board[p] == PLAYER_1 for p in m))
    m2 = sum(1 for m in MILLS if all(state.board[p] == PLAYER_2 for p in m))
    if m1 != m2:
        return Outcome.PLAYER_1_WINS if m1 > m2 else Outcome.PLAYER_2_WINS
    return Outcome.PLAYER_1_WINS  # tie-break of last resort (statistically negligible)


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
                actions.append(int(EDGE_INDEX[src, dst]))
    return actions


def _apply_placement(state: GameState, position: int) -> None:
    player = state.current_player
    state.board[position] = player
    hand = list(state.pieces_in_hand)
    hand[player - 1] -= 1
    state.pieces_in_hand = (hand[0], hand[1])
    # halfmove_clock counts halfmoves since the last *capture* — placement does
    # NOT reset it (it might or might not lead to a mill+capture; the capture
    # itself, if any, resets the clock via _apply_capture below).
    state.halfmove_clock += 1
    state.total_halfmoves += 1

    if forms_mill(state.board, position, player):
        state.must_capture = True
    else:
        state.current_player = opponent(player)
        _register_position(state)


def _apply_move(state: GameState, action: int) -> None:
    player = state.current_player
    src, dst = MOVE_EDGES[action - NUM_PLACE_CAPTURE_ACTIONS]
    state.board[src] = EMPTY
    state.board[dst] = player
    state.halfmove_clock += 1
    state.total_halfmoves += 1

    if forms_mill(state.board, dst, player):
        state.must_capture = True
    else:
        state.current_player = opponent(player)
        _register_position(state)


def _apply_capture(state: GameState, position: int) -> None:
    state.board[position] = EMPTY
    state.must_capture = False
    state.halfmove_clock = 0  # capture resets the no-progress clock
    state.total_halfmoves += 1
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
