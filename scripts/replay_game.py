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

from morris_rl.env.board import MOVE_EDGES, NUM_PLACE_CAPTURE_ACTIONS
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


# ANSI escape sequences
_ANSI_RESET = "\033[0m"
_ANSI_BOLD = "\033[1m"
_ANSI_INVERSE = "\033[7m"
_ANSI_YELLOW = "\033[33m"
_ANSI_BLUE = "\033[34m"
_ANSI_RED = "\033[31m"


# 2D grid coordinates for each of the 24 board positions.
# Grid is 13 rows × 31 cols. Outer ring on rows 0/6/12, middle on 2/6/10,
# inner on 4/6/8. Cols: outer 0/15/30, middle 5/15/25, inner 10/15/20.
_POS_COORDS: list[tuple[int, int]] = [
    (0, 0), (0, 15), (0, 30),       # 0, 1, 2  outer top
    (6, 30), (12, 30), (12, 15),    # 3, 4, 5
    (12, 0), (6, 0),                # 6, 7
    (2, 5), (2, 15), (2, 25),       # 8, 9, 10 middle top
    (6, 25), (10, 25), (10, 15),    # 11, 12, 13
    (10, 5), (6, 5),                # 14, 15
    (4, 10), (4, 15), (4, 20),      # 16, 17, 18 inner top
    (6, 20), (8, 20), (8, 15),      # 19, 20, 21
    (8, 10), (6, 10),               # 22, 23
]

# All adjacency pairs (undirected) and the inner cells of each connecting line.
# Each entry: (src, dst, [(row, col, char), ...]). For horizontal links the
# char is "─"; for vertical "│". The src/dst cells themselves are not in the
# list — they hold the piece glyph.
def _build_edge_paths() -> list[tuple[int, int, list[tuple[int, int, str]]]]:
    paths: list[tuple[int, int, list[tuple[int, int, str]]]] = []
    seen: set[tuple[int, int]] = set()
    for src in range(24):
        for dst_raw in range(24):
            # Use MOVE_EDGES which encodes the adjacency. Each undirected pair
            # appears twice in MOVE_EDGES (once per direction).
            pair = (min(src, dst_raw), max(src, dst_raw))
            if pair in seen:
                continue
            # Check adjacency via MOVE_EDGES membership
            if (src, dst_raw) not in {(s, d) for s, d in MOVE_EDGES}:
                continue
            seen.add(pair)
            r1, c1 = _POS_COORDS[src]
            r2, c2 = _POS_COORDS[dst_raw]
            cells: list[tuple[int, int, str]] = []
            if r1 == r2:  # horizontal
                lo, hi = sorted((c1, c2))
                for c in range(lo + 1, hi):
                    cells.append((r1, c, "─"))
            elif c1 == c2:  # vertical
                lo, hi = sorted((r1, r2))
                for r in range(lo + 1, hi):
                    cells.append((r, c1, "│"))
            paths.append((src, dst_raw, cells))
    return paths


_EDGE_PATHS = _build_edge_paths()
# Reverse lookup (src, dst) → path cells. Use frozenset so direction doesn't matter.
_EDGE_PATH_BY_PAIR: dict[frozenset, list[tuple[int, int, str]]] = {
    frozenset({s, d}): cells for s, d, cells in _EDGE_PATHS
}


def _piece_glyph(v: int, highlight: bool) -> str:
    """Render one piece. Same X/O glyph in both states; the just-played piece
    is shown bold (terminals render bold ~1.5× stroke weight) while idle
    pieces use regular weight. Player colour (yellow/blue) is preserved."""
    if v == PLAYER_1:
        color, ch = _ANSI_YELLOW, "X"
    elif v == PLAYER_2:
        color, ch = _ANSI_BLUE, "O"
    else:
        return "·"
    if highlight:
        return f"{_ANSI_BOLD}{color}{ch}{_ANSI_RESET}"
    return f"{color}{ch}{_ANSI_RESET}"


def render_board(
    board: Any,
    moved_from: int | None = None,
    moved_to: int | None = None,
    placed_at: int | None = None,
    captured_at: int | None = None,
) -> str:
    """Return an ASCII rendering of the 24-position Morris board.

    The optional position hints highlight the last move:
      - moved_from/moved_to: a movement-phase move (highlights the edge + dst)
      - placed_at: a placement (highlights the dropped piece)
      - captured_at: an opponent piece just removed (marks the now-empty cell)

    X (yellow) = P1, O (blue) = P2, · = empty. Highlighted piece appears
    bold red inverse-video. The edge between moved_from and moved_to is
    upgraded to double-line characters (═, ║) for visual emphasis.
    """
    grid: list[list[str]] = [[" "] * 32 for _ in range(13)]

    # Draw all edges (single-line by default)
    for s, d, cells in _EDGE_PATHS:
        for r, c, ch in cells:
            grid[r][c] = ch

    # Highlight the traversed edge if this was a movement. Use double-line
    # Unicode (═, ║) instead of single-line — visually thicker without
    # introducing colour that would clash with the X/O glyphs.
    if moved_from is not None and moved_to is not None:
        key = frozenset({moved_from, moved_to})
        path = _EDGE_PATH_BY_PAIR.get(key, [])
        for r, c, ch in path:
            doubled = {"─": "═", "│": "║"}.get(ch, ch)
            grid[r][c] = f"{_ANSI_BOLD}{doubled}{_ANSI_RESET}"

    # Place the pieces
    for i in range(24):
        r, c = _POS_COORDS[i]
        is_highlighted = i in {moved_to, placed_at, captured_at}
        grid[r][c] = _piece_glyph(int(board[i]), highlight=is_highlighted)

    # Optional captured marker — the piece is gone, mark with a small × so
    # the user can locate where the capture happened.
    if captured_at is not None and int(board[captured_at]) == EMPTY:
        r, c = _POS_COORDS[captured_at]
        grid[r][c] = f"{_ANSI_BOLD}×{_ANSI_RESET}"

    # Compose lines with row labels (1..7 on the left, a..g on top)
    out_lines: list[str] = ["    a    b    c    d    e    f    g"]
    row_labels = ["7", "", "6", "", "5", "", "4", "", "3", "", "2", "", "1"]
    for r in range(13):
        label = row_labels[r] or " "
        out_lines.append(f"{label}   " + "".join(grid[r]))
    return "\n".join(out_lines)


# ---------------------------------------------------------------------------
# Trace loading
# ---------------------------------------------------------------------------


def load_traces(
    path: Path,
    term_filter: str | None = None,
    worker_filter: int | None = None,
    ts_filter: float | None = None,
) -> list[dict[str, Any]]:
    """Load one or many JSONL files. *path* may be a file or directory.

    Filtering (applied in order):
      term_filter   — match exact term_reason ("piece_count_tiebreak", etc.)
      worker_filter — keep only games from this worker_id
      ts_filter     — keep only the trace with the closest ts (single match)

    Output is sorted by ts ascending so --list shows chronological order.
    """
    files: list[Path]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist. To generate traces, start training with "
            f"MORRIS_TRACE_DIR={path} (and optionally MORRIS_TRACE_SAMPLE_RATE=0.05). "
            f"Workers will then write worker_*.jsonl files into that directory."
        )
    if path.is_dir():
        files = sorted(path.glob("worker_*.jsonl"))
        if not files:
            raise FileNotFoundError(
                f"No worker_*.jsonl files under {path}. Has training started with "
                f"MORRIS_TRACE_DIR={path} set?"
            )
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
                if worker_filter is not None and int(trace.get("worker", -1)) != worker_filter:
                    continue
                traces.append(trace)

    if ts_filter is not None and traces:
        closest = min(traces, key=lambda t: abs(float(t.get("ts", 0.0)) - ts_filter))
        traces = [closest]

    traces.sort(key=lambda t: float(t.get("ts", 0.0)))

    if not traces:
        msg = "No traces matched ("
        msg += ", ".join(
            f"{k}={v!r}" for k, v in [
                ("term", term_filter), ("worker", worker_filter), ("ts", ts_filter)
            ] if v is not None
        ) or "no filters"
        msg += ")"
        raise ValueError(msg)
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


def _decode_last_move(
    prev_state: Any, action: int
) -> tuple[int | None, int | None, int | None, int | None]:
    """Return (moved_from, moved_to, placed_at, captured_at) for highlighting.

    Mutually exclusive depending on the action type:
      - capture (must_capture): captured_at = action
      - placement (action < NUM_PLACE_CAPTURE_ACTIONS): placed_at = action
      - movement (action >= NUM_PLACE_CAPTURE_ACTIONS): moved_from/to from MOVE_EDGES
    """
    if prev_state.must_capture:
        return None, None, None, int(action)
    if action < NUM_PLACE_CAPTURE_ACTIONS:
        return None, None, int(action), None
    src, dst = MOVE_EDGES[int(action) - NUM_PLACE_CAPTURE_ACTIONS]
    return int(src), int(dst), None, None


def _action_origin_tag(trace: dict[str, Any], action_index: int) -> str:
    """Color-coded tag identifying who chose the action at half-move `action_index`.

    Warmup traces (from generate_warmup_dataset.py) carry `opening_random_k`
    and `epsilon_random_indices` fields. Self-play traces don't — they return
    an empty string so the line stays uncluttered.
    """
    open_k = int(trace.get("opening_random_k", 0))
    eps_indices = set(trace.get("epsilon_random_indices", []))
    if open_k == 0 and not eps_indices:
        return ""  # not a warmup trace
    if action_index < open_k:
        return "\033[33m[OPENING-RANDOM]\033[0m "      # yellow
    if action_index in eps_indices:
        return "\033[31m[ε-RANDOM]\033[0m "        # red
    return "\033[32m[MINIMAX]\033[0m "                  # green


def _print_screen(trace: dict[str, Any], states: list[Any], step: int) -> None:
    print("\033[2J\033[H", end="")  # clear + home
    n = len(states) - 1
    state = states[step]
    print(f"Trace: worker={trace.get('worker')}  game={trace.get('game')}  "
          f"length={trace['length']}  term_reason={trace['term_reason']}  "
          f"outcome={trace['outcome']}")
    print(f"Move {step:>3} / {n}   "
          f"{'P1 to play' if state.current_player == PLAYER_1 else 'P2 to play'}   "
          f"{'must_capture!' if state.must_capture else ''}")
    print(f"Hand: P1={state.pieces_in_hand[0]}  P2={state.pieces_in_hand[1]}   "
          f"halfmoves={state.total_halfmoves}")

    moved_from = moved_to = placed_at = captured_at = None
    if step > 0:
        prev_state = states[step - 1]
        action = trace["actions"][step - 1]
        was_capture = prev_state.must_capture
        tag = _action_origin_tag(trace, step - 1)
        print(f"Last action: {tag}{describe_action(int(action), must_capture=was_capture)}")
        moved_from, moved_to, placed_at, captured_at = _decode_last_move(
            prev_state, int(action)
        )
    else:
        print("Last action: (start of game)")

    print(render_board(
        state.board,
        moved_from=moved_from, moved_to=moved_to,
        placed_at=placed_at, captured_at=captured_at,
    ))

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
    import time as _time

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", type=Path, help="Trace .jsonl file or directory containing worker_*.jsonl")
    p.add_argument("-i", "--index", type=int, default=None,
                   help="Game index in the filtered+sorted list (default: 0, or -1 with --latest)")
    p.add_argument("--filter", default=None,
                   help="Only load games with this term_reason (e.g. piece_count_tiebreak)")
    p.add_argument("--worker", type=int, default=None,
                   help="Only load games produced by this worker_id")
    p.add_argument("--ts", type=float, default=None,
                   help="Pick the single game with ts closest to this Unix timestamp")
    p.add_argument("--latest", action="store_true",
                   help="Pick the most recently produced game (highest ts in filtered list)")
    p.add_argument("--list", action="store_true",
                   help="Print a summary of matching games (with timestamps) and exit")
    args = p.parse_args()

    traces = load_traces(
        args.path,
        term_filter=args.filter,
        worker_filter=args.worker,
        ts_filter=args.ts,
    )

    if args.list:
        filters = []
        if args.filter: filters.append(f"term={args.filter}")
        if args.worker is not None: filters.append(f"worker={args.worker}")
        if args.ts is not None: filters.append(f"ts≈{args.ts}")
        filter_str = f" ({', '.join(filters)})" if filters else ""
        print(f"{len(traces)} game(s) loaded{filter_str}:")
        for i, t in enumerate(traces):
            ts_str = _time.strftime("%H:%M:%S", _time.localtime(float(t.get("ts", 0))))
            print(f"  [{i:>4}] {ts_str}  worker={t.get('worker'):>2}  "
                  f"len={t['length']:>3}  term={t['term_reason']:<22}  outcome={t['outcome']}")
        return

    # Default selection: --latest → last (most recent), else --index, else 0.
    if args.index is None:
        index = len(traces) - 1 if args.latest else 0
    else:
        index = args.index
    if not (0 <= index < len(traces)):
        sys.exit(f"index {index} out of range (0..{len(traces)-1})")

    interactive_replay(traces[index])


if __name__ == "__main__":
    main()
