"""Interactive replayer for self-play game traces.

Reads a JSONL trace written by training workers (when MORRIS_TRACE_DIR is set)
and steps through the game move-by-move under user control.

Usage
-----
    python scripts/replay_game.py <trace_path>            # play first game
    python scripts/replay_game.py <trace_path> -i 3       # play game #3
    python scripts/replay_game.py <trace_dir>             # auto-pick first jsonl
    python scripts/replay_game.py <trace_path> --filter halfmove_cap

Controls (interactive)
----------------------
    →  / l        next move
    ←  / h        previous move
    Home / g      jump to start
    End / G       jump to end
    q / Ctrl-C    quit
    [number]+ENTER  jump to that move

Trace format (one JSON object per line)
---------------------------------------
    {"ts":..., "worker":..., "game":"morris", "outcome":..,
     "length":.., "term_reason":"...", "actions":[i1, i2, ...]}
"""

from __future__ import annotations

import argparse
import json
import sys
import termios
import tty
from pathlib import Path
from typing import Any

# Make repo src/ importable without installation.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from morris_rl.env.rules import (
    EMPTY,
    PLAYER_1,
    PLAYER_2,
    apply_action,
    get_legal_actions,
    initial_state,
    is_terminal,
)
from morris_rl.inference.play import POSITION_LABELS, describe_action


# ---------------------------------------------------------------------------
# Board rendering
# ---------------------------------------------------------------------------


def _sym(v: int) -> str:
    if v == PLAYER_1:
        return "\033[1;33mX\033[0m"   # bold yellow = P1
    if v == PLAYER_2:
        return "\033[1;34mO\033[0m"   # bold blue   = P2
    return "·"


def render_board(board: Any) -> str:
    """Return an ASCII rendering of the 24-position Morris board.

    Layout follows POSITION_LABELS (a-g cols, 1-7 rows) — outer ring 0-7,
    middle ring 8-15, inner ring 16-23. X = P1, O = P2, · = empty.
    """
    s = [_sym(int(board[i])) for i in range(24)]
    return f"""
   a   b   c   d   e   f   g
7  {s[0]} ───────── {s[1]} ───────── {s[2]}
   │           │           │
6  │   {s[8]} ───── {s[9]} ───── {s[10]}   │
   │   │       │       │   │
5  │   │   {s[16]} ─ {s[17]} ─ {s[18]}   │   │
   │   │   │       │   │   │
4  {s[7]} ─ {s[15]} ─ {s[23]}       {s[19]} ─ {s[11]} ─ {s[3]}
   │   │   │       │   │   │
3  │   │   {s[22]} ─ {s[21]} ─ {s[20]}   │   │
   │   │       │       │   │
2  │   {s[14]} ───── {s[13]} ───── {s[12]}   │
   │           │           │
1  {s[6]} ───────── {s[5]} ───────── {s[4]}
"""


# ---------------------------------------------------------------------------
# Trace loading
# ---------------------------------------------------------------------------


def load_traces(path: Path, term_filter: str | None = None) -> list[dict[str, Any]]:
    """Load one or many JSONL files. *path* may be a file or directory."""
    files: list[Path]
    if path.is_dir():
        files = sorted(path.glob("worker_*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No worker_*.jsonl files under {path}")
    else:
        files = [path]

    traces: list[dict[str, Any]] = []
    for fp in files:
        with fp.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    trace = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if term_filter and trace.get("term_reason") != term_filter:
                    continue
                traces.append(trace)
    if not traces:
        raise ValueError(f"No traces matched (filter={term_filter!r})")
    return traces


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------


def _read_key() -> str:
    """Read one keypress from stdin in raw mode. Returns symbolic name."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":  # ESC sequence
            # Could be a 3-char arrow (ESC [ A) or just ESC
            ch2 = sys.stdin.read(1)
            if ch2 == "[":
                ch3 = sys.stdin.read(1)
                return {
                    "A": "UP", "B": "DOWN", "C": "RIGHT", "D": "LEFT",
                    "H": "HOME", "F": "END",
                }.get(ch3, f"ESC[{ch3}")
            return "ESC"
        if ch == "\x03":  # Ctrl-C
            raise KeyboardInterrupt
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _replay_states(actions: list[int]) -> list[Any]:
    """Replay actions from initial_state and return the list of states (length+1)."""
    state = initial_state()
    states = [state]
    for a in actions:
        state = apply_action(state, int(a))
        states.append(state)
    return states


def _print_screen(trace: dict[str, Any], states: list[Any], step: int) -> None:
    print("\033[2J\033[H", end="")  # clear + home
    n = len(states) - 1
    state = states[step]
    print(f"Trace: worker={trace.get('worker')}  game={trace.get('game')}  "
          f"length={trace['length']}  term_reason={trace['term_reason']}  "
          f"outcome={trace['outcome']}")
    print(f"Move {step}/{n}   "
          f"{'P1 to play' if state.current_player == PLAYER_1 else 'P2 to play'}   "
          f"{'must_capture!' if state.must_capture else ''}")
    print(f"Hand: P1={state.pieces_in_hand[0]}  P2={state.pieces_in_hand[1]}   "
          f"halfmoves={state.total_halfmoves}")

    if step > 0:
        prev_state = states[step - 1]
        action = trace["actions"][step - 1]
        was_capture = prev_state.must_capture
        print(f"Last action: {describe_action(int(action), must_capture=was_capture)}")
    else:
        print("Last action: (start of game)")

    print(render_board(state.board))

    done, outcome = is_terminal(state)
    if done:
        print(f"  → TERMINAL  outcome={outcome}")
    else:
        legal = get_legal_actions(state)
        print(f"  → {len(legal)} legal action(s) available")
    print("\nControls: → next  ← prev  Home start  End end  q quit")


def interactive_replay(trace: dict[str, Any]) -> None:
    actions = [int(a) for a in trace["actions"]]
    states = _replay_states(actions)
    n = len(states) - 1
    step = 0
    while True:
        _print_screen(trace, states, step)
        try:
            key = _read_key()
        except KeyboardInterrupt:
            print()
            return
        if key in ("q", "Q"):
            print()
            return
        if key in ("RIGHT", "l", "n", " "):
            step = min(n, step + 1)
        elif key in ("LEFT", "h", "p"):
            step = max(0, step - 1)
        elif key in ("HOME", "g"):
            step = 0
        elif key in ("END", "G"):
            step = n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", type=Path, help="Trace .jsonl file or directory containing worker_*.jsonl")
    p.add_argument("-i", "--index", type=int, default=0, help="Game index inside the trace (default 0)")
    p.add_argument("--filter", default=None, help="Only load games with this term_reason (e.g. halfmove_cap)")
    p.add_argument("--list", action="store_true", help="Print a summary of matching games and exit")
    args = p.parse_args()

    traces = load_traces(args.path, term_filter=args.filter)

    if args.list:
        print(f"{len(traces)} game(s) loaded" + (f" (filter={args.filter})" if args.filter else "") + ":")
        for i, t in enumerate(traces):
            print(f"  [{i:>4}] worker={t.get('worker')}  len={t['length']:>3}  "
                  f"term={t['term_reason']:<18}  outcome={t['outcome']}")
        return

    if not (0 <= args.index < len(traces)):
        sys.exit(f"--index {args.index} out of range (0..{len(traces)-1})")

    interactive_replay(traces[args.index])


if __name__ == "__main__":
    main()
