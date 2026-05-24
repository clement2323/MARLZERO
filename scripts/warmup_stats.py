"""Post-hoc statistics on a warmup dataset.

Reads `worker_*.jsonl` from a directory produced by
`scripts/generate_warmup_dataset.py`, replays every game to extract every
position visited, and reports:

- game-level: counts, lengths, outcomes, term reasons
- position-level: total visited, unique (canonical key), per-phase split,
  most-common positions, coverage ratio
- diversification usage: opening + ε-greedy share, average per game

Usage
-----
    uv run python scripts/warmup_stats.py outputs/warmup_d5_gate100
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from morris_rl.env.rules import (
    Phase,
    _position_key,
    apply_action,
    get_phase,
    initial_state,
)
from morris_rl.env.symmetries import SYMMETRY_PERMUTATIONS


def _canonical_position_key(state) -> tuple[int, ...]:
    """Return the lex-min representative of the position under the D4 × color-swap
    group orbit (16 elements). Two positions equivalent under any of the 16
    symmetries produce the same canonical key.

    Color swap = exchange P1/P2 on the board, swap pieces_in_hand, flip
    current_player. The position_counts dict is excluded — it depends on the
    trajectory, not the position itself.
    """
    board = state.board.astype(np.int8, copy=False)
    h1, h2 = state.pieces_in_hand
    p = state.current_player
    mc = int(state.must_capture)

    best: tuple[int, ...] | None = None
    for perm in SYMMETRY_PERMUTATIONS:
        permuted = np.empty_like(board)
        permuted[perm] = board
        # Variant A: colors as-is.
        key_a = (*permuted.tolist(), p, mc, int(h1), int(h2))
        # Variant B: swap P1 ↔ P2 (+ hands + current_player).
        swapped = np.where(permuted == 1, 2, np.where(permuted == 2, 1, 0)).astype(np.int8)
        flipped_p = 2 if p == 1 else 1
        key_b = (*swapped.tolist(), flipped_p, mc, int(h2), int(h1))
        for k in (key_a, key_b):
            if best is None or k < best:
                best = k
    assert best is not None
    return best


def _load_traces(out_dir: Path) -> list[dict[str, Any]]:
    games: list[dict[str, Any]] = []
    for path in sorted(out_dir.glob("worker_*.jsonl")):
        with path.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    games.append(json.loads(line))
    return games


def _phase_label(state) -> str:
    if state.must_capture:
        return "capture"
    return "placing" if get_phase(state, state.current_player) == Phase.PLACING else "moving"


def _pctile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = max(0, min(len(sorted_v) - 1, int(round(q * (len(sorted_v) - 1)))))
    return float(sorted_v[idx])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=Path, help="Directory with worker_*.jsonl files")
    parser.add_argument("--top-k", type=int, default=5, help="Show top-K most common positions.")
    parser.add_argument(
        "--canonical",
        action="store_true",
        help="Canonicalize positions over the 16-element D4 × color-swap group "
             "(reports 'true unique up to symmetry'). Slower (~16x).",
    )
    args = parser.parse_args()

    if not args.out_dir.exists():
        sys.exit(f"directory not found: {args.out_dir}")

    games = _load_traces(args.out_dir)
    if not games:
        sys.exit(f"no games found in {args.out_dir}")

    # --- Game-level stats -----------------------------------------------------
    lengths = [g["length"] for g in games]
    outcomes = Counter(g["outcome"] for g in games)
    terms = Counter(g["term_reason"] for g in games)
    opening_ks = [g.get("opening_random_k", 0) for g in games]
    eps_counts = [len(g.get("epsilon_random_indices", [])) for g in games]
    wall_seconds = [g.get("wall_seconds", 0.0) for g in games]

    # --- Position-level stats -------------------------------------------------
    # Replay every game to count visited positions. Key = _position_key tuple.
    position_count: Counter[tuple] = Counter()
    phase_positions: dict[str, Counter[tuple]] = {
        "placing": Counter(), "moving": Counter(), "capture": Counter(),
    }
    total_positions = 0
    positions_with_policy = 0   # minimax-evaluated positions (root_scores != None)
    positions_no_policy = 0     # random plies (opening or ε)

    key_fn = _canonical_position_key if args.canonical else _position_key
    for g in games:
        state = initial_state()
        actions = g["actions"]
        root_scores = g.get("root_scores", [None] * len(actions))
        for t, a in enumerate(actions):
            key = key_fn(state)
            position_count[key] += 1
            phase_positions[_phase_label(state)][key] += 1
            total_positions += 1
            if root_scores[t] is None:
                positions_no_policy += 1
            else:
                positions_with_policy += 1
            state = apply_action(state, int(a))
        # Don't count the terminal state (no action played from it).

    unique_positions = len(position_count)
    coverage_ratio = unique_positions / max(total_positions, 1)

    # --- Print report ---------------------------------------------------------
    n = len(games)
    print(f"\n=== Warmup dataset stats — {args.out_dir} ===\n")
    print(f"Games loaded                : {n}")
    print(f"Wallclock cumulé            : {sum(wall_seconds) / 60:.1f} min "
          f"(parallèle, donc divisé par #workers en réel)")
    print()

    print("Length distribution (half-moves):")
    print(f"  mean = {sum(lengths) / n:.1f}   "
          f"p10 = {_pctile(lengths, 0.10):.0f}   "
          f"p50 = {_pctile(lengths, 0.50):.0f}   "
          f"p90 = {_pctile(lengths, 0.90):.0f}   "
          f"max = {max(lengths)}")
    print()

    print("Outcomes:")
    for k in (0, 1, 2):
        c = outcomes.get(k, 0)
        label = {0: "DRAW", 1: "P1   ", 2: "P2   "}[k]
        print(f"  {label}: {c:>5d}  ({c / n * 100:5.1f} %)")
    decisive = outcomes.get(1, 0) + outcomes.get(2, 0)
    print(f"  decisive rate            : {decisive / n * 100:5.1f} %")
    print()

    print("Term reasons:")
    for term, count in terms.most_common():
        print(f"  {term:<32s} {count:>5d}  ({count / n * 100:5.1f} %)")
    print()

    print("Diversification usage:")
    mean_open = sum(opening_ks) / n
    mean_eps = sum(eps_counts) / n
    mean_eps_rate = sum(e / max(L, 1) for e, L in zip(eps_counts, lengths)) / n
    print(f"  opening_random_k (mean per game) : {mean_open:.1f}")
    print(f"  ε-greedy fires (mean per game)   : {mean_eps:.1f}")
    print(f"  ε-greedy rate (mean, post-open)  : {mean_eps_rate * 100:.2f} %")
    print()

    print("Position coverage:")
    canon_label = "canonical (D4×color-swap)" if args.canonical else "raw (no symmetry)"
    print(f"  total positions visited  : {total_positions:>7d}")
    print(f"  unique positions [{canon_label}]: {unique_positions:>7d}   "
          f"({coverage_ratio * 100:5.2f} % unique)")
    print(f"  minimax-labeled positions: {positions_with_policy:>7d}   "
          f"({positions_with_policy / max(total_positions, 1) * 100:5.1f} % of total)")
    print(f"  random-ply positions     : {positions_no_policy:>7d}   "
          f"({positions_no_policy / max(total_positions, 1) * 100:5.1f} % of total)")
    print()

    print("Per-phase unique positions:")
    for phase in ("placing", "moving", "capture"):
        u = len(phase_positions[phase])
        t = sum(phase_positions[phase].values())
        ratio = u / max(t, 1)
        print(f"  {phase:<8s} unique={u:>6d}   total={t:>6d}   "
              f"unique_rate={ratio * 100:5.2f} %")
    print()

    print(f"Top-{args.top_k} most common positions (across all games):")
    for i, (_key, count) in enumerate(position_count.most_common(args.top_k), 1):
        share = count / total_positions * 100
        print(f"  [{i}]  visited {count:>5d} times   ({share:5.2f} % of all positions)")
    print()

    # --- Recommendation -------------------------------------------------------
    print("=== Recommendation ===")
    if coverage_ratio < 0.30:
        print(f"  Coverage ratio {coverage_ratio * 100:.1f} % is LOW — many games share positions.")
        print( "  Consider:")
        print( "    - increasing --epsilon to 0.15-0.20")
        print( "    - increasing --opening-random-k to 8-10")
        print( "    - generating more games (e.g. 10000 instead of 5000)")
    elif coverage_ratio < 0.50:
        print(f"  Coverage ratio {coverage_ratio * 100:.1f} % is acceptable. Could be improved with")
        print( "  more diversification, but probably sufficient for warmup.")
    else:
        print(f"  Coverage ratio {coverage_ratio * 100:.1f} % is GOOD — diversification is working.")
    print()
    if decisive / n < 0.50:
        print(f"  Decisive rate {decisive / n * 100:.1f} % is below 50 %. Consider higher depth or")
        print( "  tweaking heuristic weights (see src/morris_rl/data/heuristic.py).")
    else:
        print(f"  Decisive rate {decisive / n * 100:.1f} % ≥ 50 % — gate critère passé.")
    cap_rate = terms.get("max_halfmoves_cap", 0) / n
    if cap_rate > 0.20:
        print(f"  Cap-rate {cap_rate * 100:.1f} % > 20 % — too many games hit the half-move cap.")
        print( "  Consider raising --max-halfmoves or strengthening the heuristic.")


if __name__ == "__main__":
    main()
