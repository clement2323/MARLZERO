"""Play N random-vs-random games and dump them as JSONL traces.

The traces share the schema with self-play traces, so they can be replayed
with scripts/replay_game.py (UTF-8 board renderer, step-by-step navigation).

Usage
-----
    # Dump 10 games into outputs/random_games/
    uv run python scripts/dump_random_games.py --n 10

    # Then replay them
    uv run python scripts/replay_game.py outputs/random_games/ --list
    uv run python scripts/replay_game.py outputs/random_games/ --latest
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from morris_rl.env.rules import (
    Outcome,
    apply_action,
    initial_state,
    is_terminal,
    pieces_on_board,
)
from morris_rl.eval.baselines import RandomAgent


def _detect_term_reason(state) -> str:
    """Same classification as in baseline_stats.py — mirrors is_terminal cascade."""
    from morris_rl.env.rules import (
        MAX_HALFMOVES,
        MAX_TOTAL_HALFMOVES,
        THREEFOLD_LIMIT,
        _position_key,
        get_legal_actions,
    )
    if state.total_halfmoves >= MAX_TOTAL_HALFMOVES:
        return "piece_count_tiebreak"
    key = _position_key(state)
    if state.position_counts.get(key, 0) >= THREEFOLD_LIMIT:
        return "threefold"
    if state.halfmove_clock >= MAX_HALFMOVES:
        return "halfmove_cap"
    player = state.current_player
    if state.pieces_in_hand[player - 1] == 0 and pieces_on_board(state.board, player) < 3:
        return "pieces_below_3"
    if not get_legal_actions(state):
        return "no_legal_moves"
    return "unknown"


def play_and_dump(n_games: int, out_dir: Path, seed: int = 0) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "worker_random.jsonl"
    with out_path.open("a") as fh:
        for i in range(n_games):
            agent_p1 = RandomAgent(seed=seed + i * 2)
            agent_p2 = RandomAgent(seed=seed + i * 2 + 1)
            agents = {1: agent_p1, 2: agent_p2}
            state = initial_state()
            actions: list[int] = []
            while True:
                done, outcome = is_terminal(state)
                if done:
                    break
                a = agents[state.current_player].select_action(state)
                actions.append(int(a))
                state = apply_action(state, a)
            out_int = (
                0 if (outcome is None or outcome == Outcome.DRAW) else int(outcome)
            )
            payload = {
                "ts": time.time(),
                "worker": 0,
                "game": "morris",
                "outcome": out_int,
                "length": len(actions),
                "term_reason": _detect_term_reason(state),
                "actions": actions,
            }
            fh.write(json.dumps(payload) + "\n")
            print(
                f"  game {i+1}/{n_games}: len={len(actions)}  outcome={out_int}  "
                f"term={payload['term_reason']}",
                flush=True,
            )
    print(f"\n→ traces appended to {out_path}")
    print(f"  Replay with:  uv run python scripts/replay_game.py {out_dir} --list")
    print(f"  Or:           uv run python scripts/replay_game.py {out_dir} --latest")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="Number of games to dump.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/random_games"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    play_and_dump(args.n, args.out_dir, args.seed)


if __name__ == "__main__":
    main()
