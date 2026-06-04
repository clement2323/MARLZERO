"""Long-lived subprocess wrapper around the Rust `play_tb --serve` binary.

The Rust crate keeps the Phase 1 tablebases mmap'd. Spawning the binary once
per process and piping JSONL queries over stdin/stdout keeps the OS page
cache hot — ~1 ms end-to-end per query — without the cost of paying the
process startup + mmap cost on every web request.

Request line:  ``{"wbb":<u32>,"bbb":<u32>,"stm":<1|2>}``

Response line: ``{"verdict":<u8>,"dtw":<u16>,"best_action":{"src":<u8>,"dst":<u8>,"cap":<u8>|null}|null,"top_moves":[...]}``

Verdicts use the wave codes: WIN=1, LOSS=2, DRAW=3 (see Rust ``wave.rs``).
``dtw`` is plies-to-conversion under perfect play.
"""

from __future__ import annotations

import atexit
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from morris_rl.env.board import (
    EDGE_INDEX,
    FLY_ACTION_BASE,
    NUM_POSITIONS,
)
from morris_rl.env.rules import GameState, PLAYER_1, PLAYER_2, Variant

WAVE_WIN = 1
WAVE_LOSS = 2
WAVE_DRAW = 3


class TablebaseClient:
    """Wraps ``play_tb --serve`` as a long-lived subprocess.

    Args:
        tablebase_dir: directory containing ``flying_w{w}_b{b}_wp0_bp0.bin``
            Phase 1 files.
        binary: optional explicit path to the ``play_tb`` binary. Defaults to
            ``<repo>/morris_tablebase/target/release/play_tb``. The caller can
            override via the ``MORRIS_PLAY_TB_BIN`` env var.
    """

    def __init__(
        self,
        tablebase_dir: Path,
        binary: Path | None = None,
    ) -> None:
        if binary is None:
            env_bin = os.environ.get("MORRIS_PLAY_TB_BIN")
            if env_bin:
                binary = Path(env_bin)
            else:
                here = Path(__file__).resolve()
                # src/morris_rl/inference/ → repo root has morris_tablebase/
                repo_root = here.parents[3]
                binary = repo_root / "morris_tablebase" / "target" / "release" / "play_tb"
        if not binary.exists():
            raise FileNotFoundError(
                f"play_tb binary not found at {binary}. "
                "Build it with `cargo build --release --bin play_tb` from morris_tablebase/."
            )
        if not tablebase_dir.exists():
            raise FileNotFoundError(f"tablebase dir not found: {tablebase_dir}")

        # stderr inherits so the "Loaded N subspaces" line shows up in server
        # logs; stdout is the JSONL channel.
        self._proc = subprocess.Popen(
            [str(binary), str(tablebase_dir), "--serve"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,  # line-buffered
        )
        atexit.register(self.close)

    def close(self) -> None:
        """Terminate the subprocess. Idempotent."""
        if self._proc.poll() is None:
            try:
                if self._proc.stdin is not None:
                    self._proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            try:
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()

    def query(self, state: GameState) -> dict[str, Any] | None:
        """Look up *state* in the tablebase.

        Returns ``None`` when the state is outside the TB domain:
          - must_capture sub-turn (TB enumerates move+capture atomically;
            we handle the capture leg via minimax fallback),
          - placement phase (pieces_in_hand nonzero),
          - variant != FLYING (no-flying tablebase isn't computed).

        Returns ``None`` if either side has <3 pieces (terminal) — but the
        caller's ``is_terminal`` already rejects those before reaching here.

        The returned dict has keys:
          - ``verdict``: int (WAVE_WIN/LOSS/DRAW)
          - ``dtw``: int
          - ``action``: int Python action index for the best move
          - ``top_moves``: list of {action, verdict, dtw}
        """
        if state.must_capture:
            return None
        if state.pieces_in_hand != (0, 0):
            return None
        if state.variant != Variant.FLYING:
            return None

        wbb, bbb = _board_to_bitboards(state.board)
        if bin(wbb).count("1") < 3 or bin(bbb).count("1") < 3:
            return None
        if bin(wbb).count("1") > 9 or bin(bbb).count("1") > 9:
            return None

        stm = 1 if state.current_player == PLAYER_1 else 2
        req = json.dumps({"wbb": wbb, "bbb": bbb, "stm": stm})
        assert self._proc.stdin is not None and self._proc.stdout is not None
        try:
            self._proc.stdin.write(req + "\n")
            self._proc.stdin.flush()
            line = self._proc.stdout.readline()
        except (BrokenPipeError, OSError):
            return None
        if not line:
            return None
        try:
            resp = json.loads(line)
        except json.JSONDecodeError:
            return None
        if "error" in resp:
            return None

        best = resp.get("best_action")
        if best is None:
            return None
        action = _move_to_action(best["src"], best["dst"])
        top_moves = []
        for tm in resp.get("top_moves", []):
            a = _move_to_action(tm["src"], tm["dst"])
            top_moves.append({
                "action": a,
                "verdict": tm["verdict"],
                "dtw": tm["dtw"],
            })
        return {
            "verdict": resp["verdict"],
            "dtw": resp["dtw"],
            "action": action,
            "top_moves": top_moves,
        }


def _board_to_bitboards(board: Any) -> tuple[int, int]:
    """Pack a 24-cell board into (white_bb, black_bb) bit fields.

    Position index ``i`` maps to bit ``i`` of the bitfield — same convention
    as the Rust ``morris_tablebase::board`` crate.
    """
    wbb = 0
    bbb = 0
    for i in range(NUM_POSITIONS):
        v = int(board[i])
        if v == PLAYER_1:
            wbb |= 1 << i
        elif v == PLAYER_2:
            bbb |= 1 << i
    return wbb, bbb


def _move_to_action(src: int, dst: int) -> int:
    """Encode a TB-returned (src, dst) move as a Python action index.

    Adjacent pairs use the compact MOVE_EDGES range. Non-adjacent pairs use
    the fly extension range (only legal in FLYING variant at 3 pieces).
    """
    idx = int(EDGE_INDEX[src, dst])
    if idx >= 0:
        return idx
    return FLY_ACTION_BASE + src * NUM_POSITIONS + dst
