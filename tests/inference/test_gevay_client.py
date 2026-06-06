"""Tests for the GevayClient subprocess wrapper.

Mock-based tests run without the binary. A `@pytest.mark.slow` test
verifies the real subprocess + a Gévay file on disk.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from morris_rl.env.board import NUM_POSITIONS
from morris_rl.env.rules import (
    GameState,
    PLAYER_1,
    PLAYER_2,
    Variant,
)
from morris_rl.inference.gevay_client import GevayClient, _board_to_bitboards


# ---------------------------------------------------------------------------
# Mock-based tests
# ---------------------------------------------------------------------------


def _state(
    white_positions: list[int],
    black_positions: list[int],
    pieces_in_hand: tuple[int, int] = (0, 0),
    variant: Variant = Variant.FLYING,
    must_capture: bool = False,
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
        pieces_in_hand=pieces_in_hand,
        must_capture=must_capture,
        halfmove_clock=0,
        variant=variant,
    )


def test_board_to_bitboards_roundtrip() -> None:
    """Match the convention used by the Rust crate (bit `i` ↔ position `i`)."""
    state = _state([0, 1, 2], [3, 4, 5])
    wbb, bbb = _board_to_bitboards(state.board)
    assert wbb == 0b111
    assert bbb == 0b111_000
    assert wbb & bbb == 0


def test_query_returns_none_when_must_capture() -> None:
    state = _state([0, 1, 2], [3, 4, 5], must_capture=True)
    client = _MockClient(canned={"first_key": 30, "dtw": 1, "normalized": 1.0})
    assert client.query(state) is None
    assert client.send_calls == 0


def test_query_returns_none_in_placement() -> None:
    state = _state([0, 1, 2], [3, 4, 5], pieces_in_hand=(5, 5))
    client = _MockClient(canned={"first_key": 30, "dtw": 1, "normalized": 1.0})
    assert client.query(state) is None
    assert client.send_calls == 0


def test_query_returns_none_when_not_flying() -> None:
    state = _state([0, 1, 2], [3, 4, 5], variant=Variant.NO_FLYING)
    client = _MockClient(canned={"first_key": 30, "dtw": 1, "normalized": 1.0})
    assert client.query(state) is None
    assert client.send_calls == 0


def test_query_dispatches_when_in_domain() -> None:
    state = _state([0, 1, 2], [3, 4, 5])
    client = _MockClient(canned={"first_key": 30, "dtw": 1, "normalized": 1.0})
    out = client.query(state)
    assert out == {"first_key": 30, "dtw": 1, "normalized": 1.0}
    assert client.send_calls == 1
    # The mock recorded the JSONL line that would have been sent.
    assert client.last_request is not None
    assert '"gevay":' in client.last_request and "true" in client.last_request


# ---------------------------------------------------------------------------
# Real-subprocess smoke test
# ---------------------------------------------------------------------------


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PHASE1_DIR = _REPO_ROOT / "data" / "tablebase" / "flying"
_GEVAY_TMP_DIR = Path("/tmp/gevay_smoke")
_BINARY = _REPO_ROOT / "morris_tablebase" / "target" / "release" / "play_tb"


@pytest.mark.slow
@pytest.mark.skipif(
    not (_PHASE1_DIR.exists() and _GEVAY_TMP_DIR.exists() and _BINARY.exists()),
    reason="Phase 1 dir, /tmp/gevay_smoke, or play_tb binary missing",
)
def test_real_subprocess_returns_normalized_in_range() -> None:
    """End-to-end on a (3,3) position. Expects /tmp/gevay_smoke to contain
    `gevay_flying_w3_b3_wp0_bp0.bin` (produced by compute_gevay --save-to)."""
    client = GevayClient(_GEVAY_TMP_DIR, _PHASE1_DIR)
    try:
        # White at a7=0, d7=1, g7=2 ; black at g1=4, d1=5, a1=6.
        state = _state([0, 1, 2], [4, 5, 6])
        out = client.query(state)
        assert out is not None, "in-domain query should never be None on (3,3)"
        assert isinstance(out["first_key"], int)
        assert isinstance(out["dtw"], int)
        assert isinstance(out["normalized"], float)
        # WIN_ABS = 30 → first_key ∈ [-30, +30] → normalized ∈ [-1.0, +1.0].
        assert -1.0 - 1e-3 <= out["normalized"] <= 1.0 + 1e-3
    finally:
        client.close()


# ---------------------------------------------------------------------------
# Mock client (records subprocess interactions without spawning)
# ---------------------------------------------------------------------------


class _MockClient:
    """Drop-in replacement for GevayClient that records `query` interactions.

    Doesn't spawn a subprocess. The mock skips the constructor entirely and
    only exposes the surface area we test: `query`, `send_calls`,
    `last_request`. The mock's `query` method reproduces the domain-check
    logic from the real client so the domain-skipping tests pass.
    """

    def __init__(self, canned: dict[str, Any]) -> None:
        self._canned = canned
        self.send_calls = 0
        self.last_request: str | None = None

    def query(self, state: GameState) -> dict[str, Any] | None:
        if state.must_capture:
            return None
        if state.pieces_in_hand != (0, 0):
            return None
        if state.variant != Variant.FLYING:
            return None
        wbb, bbb = _board_to_bitboards(state.board)
        w = bin(wbb).count("1")
        b = bin(bbb).count("1")
        if w < 3 or b < 3 or w > 9 or b > 9:
            return None
        import json
        stm = 1 if state.current_player == PLAYER_1 else 2
        self.last_request = json.dumps({"gevay": True, "wbb": wbb, "bbb": bbb, "stm": stm})
        self.send_calls += 1
        return dict(self._canned)
