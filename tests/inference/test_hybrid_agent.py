"""Tests for the Flying-variant hybrid agent.

Most tests use a mock TablebaseClient so they're fast and deterministic.
A single `@pytest.mark.slow` test exercises the real subprocess + Phase 1
tablebase end-to-end (requires the binary built and Phase 1 data on disk).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from morris_rl.env.board import FLY_ACTION_BASE, NUM_POSITIONS
from morris_rl.env.rules import (
    GameState,
    PLAYER_1,
    PLAYER_2,
    Variant,
    apply_action,
    get_legal_actions,
    initial_state,
)
from morris_rl.inference.hybrid_agent import HybridAgent
from morris_rl.inference.tablebase_client import (
    WAVE_DRAW,
    WAVE_LOSS,
    WAVE_WIN,
    TablebaseClient,
)


class _MockTB:
    """Mock TablebaseClient that returns a canned response when called.

    Tracks how many times `query` was invoked so we can assert routing.
    """

    def __init__(self, response: dict[str, Any] | None) -> None:
        self.response = response
        self.calls = 0

    def query(self, state: GameState) -> dict[str, Any] | None:
        self.calls += 1
        return self.response


# ---------------------------------------------------------------------------
# Routing tests (no real subprocess)
# ---------------------------------------------------------------------------


def _flying_movement_state(
    white_positions: list[int],
    black_positions: list[int],
    stm: int = PLAYER_1,
) -> GameState:
    board = np.zeros(NUM_POSITIONS, dtype=np.int8)
    for p in white_positions:
        board[p] = PLAYER_1
    for p in black_positions:
        board[p] = PLAYER_2
    return GameState(
        board=board,
        current_player=stm,
        pieces_in_hand=(0, 0),
        must_capture=False,
        halfmove_clock=0,
        variant=Variant.FLYING,
    )


def test_hybrid_routes_placement_to_minimax() -> None:
    """Placement phase (hands nonzero) must bypass the TB and use minimax."""
    state = initial_state(variant=Variant.FLYING)
    mock = _MockTB(response=None)
    agent = HybridAgent(mock, minimax_depth=2)  # depth=2 for speed
    action = agent.select_action(state)
    assert action in get_legal_actions(state)
    assert mock.calls == 0, "TB should not be queried during placement"


def test_hybrid_routes_must_capture_to_minimax() -> None:
    """must_capture sub-turn falls back to minimax even in movement phase."""
    state = _flying_movement_state([0, 1, 2, 3, 8, 9], [4, 5, 6, 7])
    state.must_capture = True
    mock = _MockTB(response={"action": 99, "verdict": WAVE_WIN, "dtw": 1, "top_moves": []})
    agent = HybridAgent(mock, minimax_depth=2)
    action = agent.select_action(state)
    # Minimax picked something from the legal capture set.
    assert action in get_legal_actions(state)
    assert mock.calls == 0


def test_hybrid_routes_movement_to_tb() -> None:
    """Movement, hands empty, flying — must query the TB."""
    state = _flying_movement_state([0, 1, 2, 8], [4, 5, 6, 13])
    # The TB returns a legal move action (a7→d7 = position 0→1 not adjacent
    # in this state, but white at 1 → use a different legal target).
    # Pick action = move white at 0 to position 23 (c4, empty, non-adjacent
    # in no-flying but legal in flying at 3 pieces... wait, 4 pieces here so
    # adjacency applies; use position 7 (adjacent to 0)).
    legal = get_legal_actions(state)
    chosen = legal[0]  # any legal action
    mock = _MockTB(
        response={
            "action": chosen,
            "verdict": WAVE_DRAW,
            "dtw": 5,
            "top_moves": [
                {"action": chosen, "verdict": WAVE_DRAW, "dtw": 5},
            ],
        }
    )
    agent = HybridAgent(mock, minimax_depth=2)
    action, top_moves, value = agent.analyze(state)
    assert action == chosen
    assert value == 0.0  # DRAW maps to 0
    assert top_moves[0][0] == chosen
    assert mock.calls == 1


def test_hybrid_falls_back_when_tb_miss() -> None:
    """If TB returns None (e.g., out of coverage), minimax handles it."""
    state = _flying_movement_state([0, 1, 2, 8], [4, 5, 6, 13])
    mock = _MockTB(response=None)
    agent = HybridAgent(mock, minimax_depth=2)
    action, top_moves, value = agent.analyze(state)
    assert action in get_legal_actions(state)
    assert top_moves == [(action, 1.0)]
    assert value == 0.0
    assert mock.calls == 1


def test_hybrid_no_flying_never_queries_tb() -> None:
    """Variant NO_FLYING short-circuits — no TB call, no flying matrix used."""
    state = _flying_movement_state([0, 1, 2, 8], [4, 5, 6, 13])
    state.variant = Variant.NO_FLYING
    mock = _MockTB(response={"action": 0, "verdict": WAVE_WIN, "dtw": 1, "top_moves": []})
    agent = HybridAgent(mock, minimax_depth=2)
    agent.select_action(state)
    assert mock.calls == 0


def test_hybrid_value_mapping() -> None:
    """verdict → value_estimate: WIN=+1, DRAW=0, LOSS=-1."""
    state = _flying_movement_state([0, 1, 2, 8], [4, 5, 6, 13])
    legal = get_legal_actions(state)
    chosen = legal[0]
    for verdict, expected in [(WAVE_WIN, 1.0), (WAVE_DRAW, 0.0), (WAVE_LOSS, -1.0)]:
        mock = _MockTB(
            response={
                "action": chosen,
                "verdict": verdict,
                "dtw": 1,
                "top_moves": [{"action": chosen, "verdict": verdict, "dtw": 1}],
            }
        )
        agent = HybridAgent(mock, minimax_depth=2)
        _, _, value = agent.analyze(state)
        assert value == expected, f"verdict {verdict} → expected {expected}, got {value}"


# ---------------------------------------------------------------------------
# Real subprocess test (slow — requires built binary + tablebase on disk)
# ---------------------------------------------------------------------------


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_TB_DIR = _REPO_ROOT / "data" / "tablebase" / "flying"
_DEFAULT_TB_BIN = _REPO_ROOT / "morris_tablebase" / "target" / "release" / "play_tb"


@pytest.mark.slow
@pytest.mark.skipif(
    not (_DEFAULT_TB_DIR.exists() and _DEFAULT_TB_BIN.exists()),
    reason="Phase 1 tablebase or play_tb binary not available",
)
def test_real_tablebase_client_returns_legal_action() -> None:
    """End-to-end: spawn the real subprocess, query a (3,3) state."""
    client = TablebaseClient(_DEFAULT_TB_DIR)
    try:
        # White at 0, 1, 2 (a7, d7, g7) — 3 pieces.
        # Black at 4, 5, 6 (g1, d1, a1) — 3 pieces, no adjacency to white.
        state = _flying_movement_state([0, 1, 2], [4, 5, 6])
        result = client.query(state)
        assert result is not None, "TB miss on a valid (3,3) flying position"
        assert result["verdict"] in (WAVE_WIN, WAVE_DRAW, WAVE_LOSS)
        legal = get_legal_actions(state)
        assert result["action"] in legal, (
            f"TB returned action {result['action']} not in legal set {legal}"
        )
    finally:
        client.close()
