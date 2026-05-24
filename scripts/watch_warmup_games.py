"""Live observer for warmup dataset generation.

Tails the worker_*.jsonl files written by `scripts/generate_warmup_dataset.py`
and renders each new game as it lands on disk. Each worker streams (append +
flush) one JSONL record per completed game, so polling with a small interval
is sufficient — no inotify needed.

Usage
-----
    # Pendant que la génération tourne dans un autre terminal :
    uv run python scripts/watch_warmup_games.py outputs/warmup_d5_gate100 --follow

    # Une-shot, juste pour parcourir ce qui est déjà sur disque :
    uv run python scripts/watch_warmup_games.py outputs/warmup_d5_gate100 --once

    # Mode board final affiché (au lieu de summary 1-ligne) :
    uv run python scripts/watch_warmup_games.py outputs/warmup_d5_gate100 --follow --render-each
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for replay_game import

from morris_rl.env.rules import apply_action, initial_state

# Reuse the UTF-8 renderer + helpers from replay_game.
from replay_game import (  # type: ignore[import-not-found]
    _decode_last_move,
    render_board,
)


def _format_summary_line(trace: dict[str, Any]) -> str:
    out = trace["outcome"]
    out_str = {0: "DRAW", 1: "P1   ", 2: "P2   "}.get(out, f"out={out}")
    eps_n = len(trace.get("epsilon_random_indices", []))
    open_k = trace.get("opening_random_k", 0)
    wall = trace.get("wall_seconds", 0.0)
    return (
        f"[w{trace.get('worker', '?'):<2} g{trace.get('_game_index_in_file', 0):>3d}]  "
        f"len={trace['length']:>3d}  "
        f"outcome={out_str}  "
        f"term={trace['term_reason']:<28s}  "
        f"open={open_k}  ε={eps_n}  "
        f"wall={wall:.1f}s"
    )


def _render_final_board(trace: dict[str, Any]) -> str:
    """Replay actions through initial_state and render the FINAL board only."""
    state = initial_state()
    prev_state = None
    last_action = None
    for a in trace["actions"]:
        prev_state = state
        last_action = int(a)
        state = apply_action(state, last_action)
    moved_from = moved_to = placed_at = captured_at = None
    if prev_state is not None and last_action is not None:
        moved_from, moved_to, placed_at, captured_at = _decode_last_move(
            prev_state, last_action
        )
    return render_board(
        state.board,
        moved_from=moved_from,
        moved_to=moved_to,
        placed_at=placed_at,
        captured_at=captured_at,
    )


def _passes_filters(trace: dict[str, Any], args: argparse.Namespace) -> bool:
    if args.worker is not None and trace.get("worker") != args.worker:
        return False
    if args.only_decisive and trace["outcome"] == 0:
        return False
    if args.only_cap and trace["term_reason"] != "max_halfmoves_cap":
        return False
    return True


def _emit_trace(trace: dict[str, Any], args: argparse.Namespace) -> None:
    if not _passes_filters(trace, args):
        return

    if args.render_each:
        print()
        print(_format_summary_line(trace))
        print(_render_final_board(trace))
        sys.stdout.flush()
    else:
        print(_format_summary_line(trace), flush=True)


def _list_worker_files(out_dir: Path) -> list[Path]:
    return sorted(out_dir.glob("worker_*.jsonl"))


def _drain_file(
    path: Path,
    offset: int,
    counters: dict[str, int],
    args: argparse.Namespace,
) -> int:
    """Read new lines from `path` starting at byte `offset`, emit them.

    Returns the new offset after reading.
    """
    try:
        with path.open("rb") as fh:
            fh.seek(offset)
            chunk = fh.read()
    except FileNotFoundError:
        return offset

    if not chunk:
        return offset

    text = chunk.decode("utf-8", errors="replace")
    # Trailing partial line — defer until next poll.
    if not text.endswith("\n"):
        nl = text.rfind("\n")
        if nl < 0:
            return offset  # no full line yet
        text = text[: nl + 1]

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            trace = json.loads(line)
        except json.JSONDecodeError:
            continue
        counters[path.name] = counters.get(path.name, 0) + 1
        trace["_game_index_in_file"] = counters[path.name]
        _emit_trace(trace, args)

    return offset + len(text.encode("utf-8"))


def watch(out_dir: Path, args: argparse.Namespace) -> None:
    offsets: dict[Path, int] = {}
    counters: dict[str, int] = {}

    # First pass: enumerate existing files and drain everything already on disk.
    for path in _list_worker_files(out_dir):
        offsets[path] = 0
        offsets[path] = _drain_file(path, 0, counters, args)

    if args.once:
        return

    # Follow mode: re-poll periodically. Re-list files each tick so new
    # workers that come online mid-run are picked up.
    print(f"\n  watching {out_dir} (poll={args.poll_seconds}s)  Ctrl-C to quit", flush=True)
    try:
        while True:
            time.sleep(args.poll_seconds)
            for path in _list_worker_files(out_dir):
                if path not in offsets:
                    offsets[path] = 0
                offsets[path] = _drain_file(path, offsets[path], counters, args)
    except KeyboardInterrupt:
        print("\n  bye.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live tail of warmup JSONL generation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("out_dir", type=Path, help="Directory containing worker_*.jsonl")
    parser.add_argument("--follow", action="store_true", default=True, help="Tail -f mode (default).")
    parser.add_argument("--once", action="store_true", help="Render existing and exit.")
    parser.add_argument("--render-each", action="store_true", help="Render final board for each game.")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--worker", type=int, default=None, help="Show only this worker.")
    parser.add_argument("--only-decisive", action="store_true", help="Hide draws.")
    parser.add_argument(
        "--only-cap",
        action="store_true",
        help="Show only games that hit the max_halfmoves cap (debug).",
    )
    args = parser.parse_args()

    if not args.out_dir.exists():
        sys.exit(f"directory not found: {args.out_dir}")
    watch(args.out_dir, args)


if __name__ == "__main__":
    main()
