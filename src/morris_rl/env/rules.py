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

MAX_HALFMOVES: Final[int] = 300        # no-progress clock (dead-code when total cap active)
MAX_TOTAL_HALFMOVES: Final[int] = 200  # absolute game length cap — piece count breaks ties
# Raised from 60 → 200 so more games reach natural termination (≤2 pieces or
# no_legal_moves) instead of being decided by an artificial piece-count tiebreak.
# Cleaner value targets at the cost of longer self-play games.
THREEFOLD_LIMIT: Final[int] = 3
# Sliding window for the no-repetition rule. A candidate action is illegal if
# its resulting position_key matches any position visited within the last
# REPETITION_WINDOW halfmoves. Eliminates ping-pong A↔B↔A cycles (period 2
# halfmoves) and longer tactical loops up to ~half the window depth.
# Game-changing relative to standard Morris: a player with no non-repeating
# move loses by no_legal_moves rather than drawing by repetition.
REPETITION_WINDOW: Final[int] = 8


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
    # Sliding window of the last REPETITION_WINDOW position keys. Used by
    # get_legal_actions to forbid candidates whose resulting position matches
    # any recent one — prevents short tactical cycles (ping-pong moves).
    # Tuple (not deque) so GameState stays trivially picklable for mp workers.
    recent_position_keys: tuple[tuple[int, ...], ...] = ()

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
            recent_position_keys=self.recent_position_keys,  # tuple is immutable, share safely
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

    Filters out movement candidates whose resulting position matches any of
    the last REPETITION_WINDOW visited positions. This prevents short cycles
    (ping-pong A↔B↔A) without ending the game — the network simply never
    sees these moves as options. Placement and capture actions are never
    filtered (placements strictly grow piece count, so cycles are impossible;
    captures strictly shrink it).
    """
    if state.must_capture:
        return _legal_capture_actions(state)
    phase = get_phase(state, state.current_player)
    if phase == Phase.PLACING:
        return [p for p in range(NUM_POSITIONS) if state.board[p] == EMPTY]
    candidates = _legal_move_actions(state)
    if not state.recent_position_keys:
        return candidates
    # Convert the recent tuple-keys to bytes once so per-candidate lookups
    # are O(1) hash on a flat bytes buffer instead of O(28) tuple hashing.
    # tuple-keys come from _position_key: (b0..b23, current_player,
    # must_capture, p1_hand, p2_hand) — 28 ints, all small enough for bytes.
    recent_bytes: set[bytes] = {
        bytes(t[:NUM_POSITIONS]) + bytes(t[NUM_POSITIONS:])
        for t in state.recent_position_keys
    }
    legal = []
    board = state.board
    opp = opponent(state.current_player)
    p1h, p2h = state.pieces_in_hand
    cur = state.current_player
    for a in candidates:
        src, dst = MOVE_EDGES[a - NUM_PLACE_CAPTURE_ACTIONS]
        original_src = int(board[src])
        original_dst = int(board[dst])
        board[src] = EMPTY
        board[dst] = cur
        # Mill formation flips must_capture and keeps the current player.
        if forms_mill(board, dst, cur):
            next_player = cur
            next_must_capture = 1
        else:
            next_player = opp
            next_must_capture = 0
        # bytes(numpy_int8_array) is a C memcpy of 24 bytes (~0.3µs), vs
        # ~5µs for *board.tolist() + tuple construction. Hot path called
        # ~15 candidates × ~800 MCTS sims per move, so the speedup matters.
        key_bytes = bytes(board) + bytes((next_player, next_must_capture, p1h, p2h))
        board[src] = original_src
        board[dst] = original_dst
        if key_bytes not in recent_bytes:
            legal.append(a)
    return legal


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

    No draws are possible: the 100-halfmove total cap and threefold repetition
    both resolve via _piece_count_winner (pieces → mills → P1 fallback).
    """
    # The must_capture sub-turn is not a terminal check point.
    if state.must_capture:
        return False, None

    if state.total_halfmoves >= MAX_TOTAL_HALFMOVES:
        return True, _piece_count_winner(state)

    key = _position_key(state)
    if state.position_counts.get(key, 0) >= THREEFOLD_LIMIT:
        return True, _piece_count_winner(state)

    if state.halfmove_clock >= MAX_HALFMOVES:
        return True, _piece_count_winner(state)

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
    state.halfmove_clock = 0  # placement always resets the no-progress clock
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
    # Append to sliding window, capped at REPETITION_WINDOW. Tuple ops are
    # O(K) but K is small (8) so cost is negligible vs the rest of apply.
    window = state.recent_position_keys + (key,)
    if len(window) > REPETITION_WINDOW:
        window = window[-REPETITION_WINDOW:]
    state.recent_position_keys = window
