"""Phase 0 — Mesures de référence des baselines.

Joue N parties entre paires d'agents (random / minimax depth=k) avec les règles
de terminaison actuelles, et reporte :

  - distribution des longueurs (min / median / p90 / max / histogram)
  - % décisives (no_legal_moves + pieces_below_3 vs cap-300 vs threefold vs 50-no-capture)
  - win/draw/loss rates avec alternance P1/P2
  - mean captures per game

Usage
-----
    # Run complet : random×random + depth3×depth3 + depth5×depth5 + depth5×depth3
    uv run python scripts/baseline_stats.py

    # Test rapide (50 games par paire)
    uv run python scripts/baseline_stats.py --quick

    # Couples customisés
    uv run python scripts/baseline_stats.py --custom random 500 minimax3 200

Sortie : table markdown sur stdout, dump JSON dans outputs/baseline_stats/<timestamp>/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from morris_rl.env.rules import (
    GameState,
    Outcome,
    apply_action,
    initial_state,
    is_terminal,
    opponent,
    pieces_on_board,
)
from morris_rl.env.board import NUM_PLACE_CAPTURE_ACTIONS
from morris_rl.eval.baselines import MinimaxAgent, RandomAgent


# ---------------------------------------------------------------------------
# Game runner with full instrumentation
# ---------------------------------------------------------------------------


@dataclass
class GameStats:
    """Per-game record collected during the run."""
    length: int                # halfmoves played
    outcome: int               # 1=P1, 2=P2, 0=draw
    term_reason: str           # halfmove_cap / threefold / no_legal_moves / pieces_below_3 / piece_count_tiebreak
    captures: int              # total captures during the game (both sides)
    final_pieces_diff: int     # P1_pieces - P2_pieces at the end


def _detect_term_reason(state: GameState) -> str:
    """Classify why a Morris game ended. Mirrors the cascade in is_terminal."""
    from morris_rl.env.rules import (
        MAX_HALFMOVES,
        MAX_TOTAL_HALFMOVES,
        THREEFOLD_LIMIT,
        get_legal_actions,
        _position_key,
    )

    if state.total_halfmoves >= MAX_TOTAL_HALFMOVES:
        return "piece_count_tiebreak"   # safety net cap reached
    key = _position_key(state)
    if state.position_counts.get(key, 0) >= THREEFOLD_LIMIT:
        return "threefold"
    if state.halfmove_clock >= MAX_HALFMOVES:
        return "halfmove_cap"            # 50-no-capture rule
    player = state.current_player
    if state.pieces_in_hand[player - 1] == 0 and pieces_on_board(state.board, player) < 3:
        return "pieces_below_3"
    if not get_legal_actions(state):
        return "no_legal_moves"
    return "unknown"


def _play_one_game(agent_p1, agent_p2) -> GameStats:
    """Play a single game, tracking captures and final stats."""
    state = initial_state()
    agents = {1: agent_p1, 2: agent_p2}
    captures = 0
    length = 0
    prev_total_pieces = int((state.board != 0).sum())

    while True:
        done, outcome = is_terminal(state)
        if done:
            break
        action = agents[state.current_player].select_action(state)
        state = apply_action(state, action)
        length += 1
        cur_total_pieces = int((state.board != 0).sum())
        # A capture reduces total pieces by 1 (placement adds 1, movement keeps same).
        if cur_total_pieces < prev_total_pieces:
            captures += 1
        prev_total_pieces = cur_total_pieces

    if outcome is None or outcome == Outcome.DRAW:
        out_int = 0
    else:
        out_int = int(outcome)

    p1_pieces = pieces_on_board(state.board, 1)
    p2_pieces = pieces_on_board(state.board, 2)
    term_reason = _detect_term_reason(state)
    return GameStats(
        length=length,
        outcome=out_int,
        term_reason=term_reason,
        captures=captures,
        final_pieces_diff=p1_pieces - p2_pieces,
    )


# ---------------------------------------------------------------------------
# Aggregation + reporting
# ---------------------------------------------------------------------------


@dataclass
class PairReport:
    label: str
    n_games: int = 0
    lengths: list[int] = field(default_factory=list)
    outcomes: Counter = field(default_factory=Counter)
    term_reasons: Counter = field(default_factory=Counter)
    captures: list[int] = field(default_factory=list)
    pieces_diff: list[int] = field(default_factory=list)
    wall_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "n_games": self.n_games,
            "length_min": min(self.lengths) if self.lengths else 0,
            "length_median": median(self.lengths) if self.lengths else 0,
            "length_mean": round(mean(self.lengths), 1) if self.lengths else 0,
            "length_p90": sorted(self.lengths)[int(0.9 * len(self.lengths))] if self.lengths else 0,
            "length_max": max(self.lengths) if self.lengths else 0,
            "outcomes_p1_win": self.outcomes.get(1, 0),
            "outcomes_p2_win": self.outcomes.get(2, 0),
            "outcomes_draw": self.outcomes.get(0, 0),
            "term_reasons": dict(self.term_reasons),
            "captures_mean": round(mean(self.captures), 2) if self.captures else 0,
            "pieces_diff_mean": round(mean(self.pieces_diff), 2) if self.pieces_diff else 0,
            "wall_seconds": round(self.wall_seconds, 2),
            "games_per_sec": round(self.n_games / self.wall_seconds, 2) if self.wall_seconds else 0,
        }


def run_pair(label: str, make_agent_a, make_agent_b, n_games: int) -> PairReport:
    """Play n_games between two agents, alternating who plays P1 to remove first-move bias.

    Each game gets its own integer ``game_idx`` passed to the agent factory so
    seeded agents (like RandomAgent) produce fresh trajectories instead of
    replaying the same canonical game N times.
    """
    print(f"\n[{label}] running {n_games} games...", flush=True)
    rep = PairReport(label=label)
    t0 = time.perf_counter()
    decisive_terms = {"no_legal_moves", "pieces_below_3"}
    for i in range(n_games):
        # Alternate which factory plays P1.
        if i % 2 == 0:
            p1, p2 = make_agent_a(i), make_agent_b(i)
            is_a_p1 = True
        else:
            p1, p2 = make_agent_b(i), make_agent_a(i)
            is_a_p1 = False
        stats = _play_one_game(p1, p2)
        rep.n_games += 1
        rep.lengths.append(stats.length)
        rep.captures.append(stats.captures)
        rep.pieces_diff.append(stats.final_pieces_diff if is_a_p1 else -stats.final_pieces_diff)
        rep.term_reasons[stats.term_reason] += 1
        # Re-frame outcome from A's perspective for clean win-rate.
        if stats.outcome == 0:
            rep.outcomes["draw"] += 1
        else:
            a_won = (stats.outcome == 1) == is_a_p1
            rep.outcomes["a_win" if a_won else "b_win"] += 1
        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t0
            decisive = sum(rep.term_reasons[k] for k in decisive_terms)
            print(
                f"  ...{i+1}/{n_games}  "
                f"({elapsed:.1f}s, {(i+1)/elapsed:.1f} games/s, "
                f"decisive={decisive}/{rep.n_games})",
                flush=True,
            )
    rep.wall_seconds = time.perf_counter() - t0
    # Re-key for to_dict()
    rep.outcomes = Counter({1: rep.outcomes["a_win"], 2: rep.outcomes["b_win"], 0: rep.outcomes["draw"]})
    return rep


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------


def _print_report(reports: list[PairReport]) -> None:
    print()
    print("=" * 90)
    print("Baseline statistics report")
    print("=" * 90)

    # 1) Length distribution
    print("\n## Distribution des longueurs (halfmoves)")
    print()
    print(f"{'pair':<28} {'min':>5} {'median':>7} {'mean':>6} {'p90':>5} {'max':>5}  games/s")
    print("-" * 80)
    for r in reports:
        d = r.to_dict()
        print(
            f"{r.label:<28} {d['length_min']:>5} {d['length_median']:>7} "
            f"{d['length_mean']:>6} {d['length_p90']:>5} {d['length_max']:>5}   "
            f"{d['games_per_sec']:>6.1f}"
        )

    # 2) Outcomes (A's perspective for paired, both for symmetric)
    print("\n## Outcomes (A's perspective, alternated P1/P2)")
    print()
    print(f"{'pair':<28} {'A_win':>6} {'B_win':>6} {'draw':>6}   decisive%")
    print("-" * 70)
    for r in reports:
        n = r.n_games
        d = r.to_dict()
        # Convert outcomes back to A's POV — already done in run_pair.
        a = r.outcomes.get(1, 0)
        b = r.outcomes.get(2, 0)
        dr = r.outcomes.get(0, 0)
        decisive_n = r.term_reasons.get("no_legal_moves", 0) + r.term_reasons.get("pieces_below_3", 0)
        decisive_pct = 100 * decisive_n / n if n else 0
        print(
            f"{r.label:<28} {a/n*100:5.1f}% {b/n*100:5.1f}% {dr/n*100:5.1f}%   "
            f"{decisive_pct:5.1f}%"
        )

    # 3) Termination reasons
    print("\n## Termination reason breakdown")
    print()
    all_reasons = sorted({k for r in reports for k in r.term_reasons})
    header = f"{'pair':<28}  " + "  ".join(f"{k[:14]:>14}" for k in all_reasons)
    print(header)
    print("-" * len(header))
    for r in reports:
        n = r.n_games or 1
        row = f"{r.label:<28}  " + "  ".join(
            f"{r.term_reasons.get(k,0)/n*100:>13.1f}%" for k in all_reasons
        )
        print(row)

    # 4) Captures + final pieces
    print("\n## Captures & final pieces diff")
    print()
    print(f"{'pair':<28} {'capt/game':>10}  {'final_diff_mean (A POV)':>25}")
    print("-" * 70)
    for r in reports:
        d = r.to_dict()
        print(f"{r.label:<28} {d['captures_mean']:>10}  {d['pieces_diff_mean']:>25}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="Reduced game counts (50/40/30/40) for fast smoke.")
    parser.add_argument("--n-random", type=int, default=1000)
    parser.add_argument("--n-d3", type=int, default=200)
    parser.add_argument("--n-d5", type=int, default=200)
    parser.add_argument("--n-d5-vs-d3", type=int, default=100)
    parser.add_argument("--no-d5", action="store_true", help="Skip depth-5 pairs (depth5 is slow on CPU).")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Where to dump JSON. Defaults to outputs/baseline_stats/<timestamp>/.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.quick:
        args.n_random, args.n_d3, args.n_d5, args.n_d5_vs_d3 = 50, 40, 30, 40

    reports: list[PairReport] = []

    # 1) Random vs Random
    # Per-game seed = args.seed + game_idx*2 (P1) / +1 (P2) so each game is unique.
    reports.append(run_pair(
        "random vs random",
        lambda i: RandomAgent(seed=args.seed + i * 2),
        lambda i: RandomAgent(seed=args.seed + i * 2 + 1),
        args.n_random,
    ))

    # 2) Minimax d3 vs d3 — minimax is deterministic given the same state, so
    # P1 vs P2 identical-depth games would all produce the same trajectory.
    # The alternation in run_pair (P1↔P2 swap) gives us 2 distinct games at
    # most; that's expected for deterministic agents on a fixed start state.
    reports.append(run_pair(
        "minimax(d=3) vs minimax(d=3)",
        lambda i: MinimaxAgent(depth=3),
        lambda i: MinimaxAgent(depth=3),
        args.n_d3,
    ))

    if not args.no_d5:
        # 3) Minimax d5 vs d5
        reports.append(run_pair(
            "minimax(d=5) vs minimax(d=5)",
            lambda i: MinimaxAgent(depth=5),
            lambda i: MinimaxAgent(depth=5),
            args.n_d5,
        ))

        # 4) Minimax d5 vs d3 (asymmetry test)
        reports.append(run_pair(
            "minimax(d=5) vs minimax(d=3)",
            lambda i: MinimaxAgent(depth=5),
            lambda i: MinimaxAgent(depth=3),
            args.n_d5_vs_d3,
        ))

    _print_report(reports)

    # Dump JSON for downstream tooling.
    if args.out_dir is None:
        args.out_dir = Path("outputs/baseline_stats") / time.strftime("%Y-%m-%d_%H-%M-%S")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "stats.json"
    with json_path.open("w") as fh:
        json.dump([r.to_dict() for r in reports], fh, indent=2)
    print(f"\n→ JSON written to {json_path}")


if __name__ == "__main__":
    main()
