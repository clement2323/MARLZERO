"""Long-lived subprocess wrapper around `play_tb --gevay-dir <DIR> --serve`.

Sister to [tablebase_client.py](tablebase_client.py): same subprocess pattern,
same JSONL transport, same `_board_to_bitboards` translation. The only
difference is the request marker `"gevay": true` and the response shape
(`first_key`, `dtw`, `normalized` instead of `verdict`, `dtw`, `best_action`).

`normalized` is `first_key / WIN_ABS` (= `first_key / 30`) so it sits in
roughly [-1, +1] — directly usable as an AlphaZero value target. Hard
WIN/LOSS classes (`|first_key| >= WIN_ABS/2 = 15`) map to `|normalized| >=
0.5`; rank-zero draws map to `0.0`; non-zero rank draws fall in the open
interval.

Used at training time to score the post-placement (ply 18) position when
the self-play loop hits the artificial terminate-at-ply cutoff.
"""

from __future__ import annotations

import atexit
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from morris_rl.env.board import NUM_POSITIONS
from morris_rl.env.rules import GameState, PLAYER_1, PLAYER_2, Variant


class GevayClient:
    """Wraps `play_tb --serve --gevay-dir <DIR>` as a long-lived subprocess.

    Args:
        gevay_dir: directory holding ``gevay_flying_w{w}_b{b}_wp0_bp0.bin``
            files produced by ``compute_gevay --save-to``.
        phase1_dir: Phase 1 tablebase dir — required because the play_tb
            binary always loads Phase 1 too (it's the same process). Pass
            the directory that holds ``flying_w{w}_b{b}_wp0_bp0.bin``.
        binary: optional explicit path to ``play_tb``. Default resolution
            mirrors :class:`TablebaseClient`.
    """

    def __init__(
        self,
        gevay_dir: Path,
        phase1_dir: Path,
        binary: Path | None = None,
    ) -> None:
        if binary is None:
            env_bin = os.environ.get("MORRIS_PLAY_TB_BIN")
            if env_bin:
                binary = Path(env_bin)
            else:
                here = Path(__file__).resolve()
                repo_root = here.parents[3]
                binary = repo_root / "morris_tablebase" / "target" / "release" / "play_tb"
        if not binary.exists():
            raise FileNotFoundError(
                f"play_tb binary not found at {binary}. "
                "Build with `cargo build --release --bin play_tb` from morris_tablebase/."
            )
        if not gevay_dir.exists():
            raise FileNotFoundError(f"gevay dir not found: {gevay_dir}")
        if not phase1_dir.exists():
            raise FileNotFoundError(f"phase1 dir not found: {phase1_dir}")

        self._proc = subprocess.Popen(
            [
                str(binary), str(phase1_dir),
                "--serve",
                "--gevay-dir", str(gevay_dir),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
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
        """Look up V_Gévay for *state*.

        Returns None when the state is outside the Gévay domain:
          - must_capture sub-turn (TB classifies positions, not mid-move),
          - placement phase (`pieces_in_hand != (0, 0)`),
          - variant != FLYING (no-flying Gévay isn't computed),
          - either side has < 3 pieces (terminal — the caller's
            ``is_terminal`` would already have caught this).

        Returns a dict ``{first_key: int, dtw: int, normalized: float}``
        where ``normalized`` is in roughly ``[-1, +1]``. Returns None on a
        subprocess error or a position outside the loaded subspaces.
        """
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

        stm = 1 if state.current_player == PLAYER_1 else 2
        req = json.dumps({"gevay": True, "wbb": wbb, "bbb": bbb, "stm": stm})
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
        return {
            "first_key": int(resp["first_key"]),
            "dtw": int(resp["dtw"]),
            "normalized": float(resp["normalized"]),
        }


def _board_to_bitboards(board: Any) -> tuple[int, int]:
    """Pack a 24-cell board into (white_bb, black_bb). Identical convention
    to :func:`morris_rl.inference.tablebase_client._board_to_bitboards`."""
    wbb = 0
    bbb = 0
    for i in range(NUM_POSITIONS):
        v = int(board[i])
        if v == PLAYER_1:
            wbb |= 1 << i
        elif v == PLAYER_2:
            bbb |= 1 << i
    return wbb, bbb
