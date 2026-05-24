"""ELO evaluation: bare network checkpoint vs MinimaxAgent(depth=K).

Plays N games with alternating sides (P1 / P2) to wash out first-mover bias,
counts wins / draws / losses, and converts the score to an ELO difference
(standard formula, draws counted as 0.5).

This is the Phase 2 → Phase 3 gate: only when the bare network (argmax,
no MCTS) beats or draws minimax-depth-3 should we start the self-play
fine-tuning phase.

Usage
-----
    # Eval the warmup checkpoint vs minimax depth-3 (Phase 2 gate criterion)
    uv run python scripts/eval_elo.py \\
        outputs/sup_warmup_5k/best.pt --depth 3 --num-games 200

    # Compare two networks (no minimax involved)
    uv run python scripts/eval_elo.py \\
        outputs/sup_warmup_5k/best.pt --vs-checkpoint outputs/sup_v2/best.pt \\
        --num-games 200

    # Use MCTS at inference instead of bare argmax (for after Phase 3)
    uv run python scripts/eval_elo.py outputs/best.pt --depth 5 \\
        --use-mcts --num-sims 200 --num-games 100
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from morris_rl.env.rules import (
    Outcome,
    apply_action,
    initial_state,
    is_terminal,
)
from morris_rl.eval.baselines import MinimaxAgent, NetworkAgent, RandomAgent
from morris_rl.network.factory import build_network
from morris_rl.training.supervised import BareNetworkAgent
from morris_rl.utils.checkpoints import load_checkpoint


def _load_network(checkpoint_path: Path, device: torch.device):
    payload = load_checkpoint(checkpoint_path)
    cfg = OmegaConf.create(payload["config"])
    network = build_network(cfg)
    network.load_state_dict(payload["state_dict"])
    network.eval().to(device)
    return network, payload["step"], payload["config"]


def _play_match(
    p1_agent,
    p2_agent,
    max_halfmoves: int = 200,
    opening_random_k: int = 0,
    rng: "random.Random | None" = None,
) -> int:
    """Return 0 (draw / cap) / 1 / 2.

    When `opening_random_k > 0`, the first K half-moves are chosen uniformly
    at random from the legal moves (using `rng`), regardless of which agent
    would have played them. This is the standard trick to break determinism
    between two argmax-based agents and produce statistically meaningful ELO
    estimates — without it, alternating sides only generates exactly 2
    distinct games replayed N/2 times each.
    """
    from morris_rl.env.rules import get_legal_actions
    agents = {1: p1_agent, 2: p2_agent}
    state = initial_state()
    halfmove_idx = 0
    while True:
        if state.total_halfmoves >= max_halfmoves:
            return 0
        done, outcome = is_terminal(state)
        if done:
            return 0 if (outcome is None or outcome == Outcome.DRAW) else int(outcome)
        if halfmove_idx < opening_random_k and rng is not None:
            a = rng.choice(get_legal_actions(state))
        else:
            a = agents[state.current_player].select_action(state)
        state = apply_action(state, int(a))
        halfmove_idx += 1


# ---------------------------------------------------------------------------
# ELO math
# ---------------------------------------------------------------------------


def _score_to_elo_diff(score: float) -> float:
    """ELO difference from a score (0..1), where score = (wins + 0.5*draws) / N.

    Returns 0 when score is at the saturation boundary (0 or 1) — caller
    should handle that case with a "supérieur à X" caveat.
    """
    if score <= 0.0:
        return -math.inf
    if score >= 1.0:
        return math.inf
    return -400.0 * math.log10(1.0 / score - 1.0)


def _wilson_ci(wins: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95 % CI on the binomial proportion wins/n (draws=0.5).

    Used because for small N or extreme p (close to 0 or 1) Wilson is much
    better calibrated than the normal-approximation Wald interval.
    """
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return centre - half, centre + half


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------


def run_eval(
    candidate_factory,
    opponent_factory,
    n_games: int,
    candidate_label: str,
    opponent_label: str,
    seed: int = 0,
    max_halfmoves: int = 200,
    opening_random_k: int = 4,
) -> dict:
    """Play `n_games` alternating sides; return outcome counts + ELO.

    `opening_random_k > 0` plays the first K plies uniformly at random,
    diversifying the start positions across games. Without this, two
    deterministic argmax-based agents replay the same 2 games N/2 times
    each.
    """
    import random
    wins = draws = losses = 0
    p1_starts = p2_starts = 0
    t0 = time.time()
    for i in range(n_games):
        candidate = candidate_factory(seed=seed + i)
        opponent = opponent_factory(seed=seed + 100_000 + i)
        cand_side = 1 if i % 2 == 0 else 2
        if cand_side == 1:
            p1, p2 = candidate, opponent
            p1_starts += 1
        else:
            p1, p2 = opponent, candidate
            p2_starts += 1
        # Per-game RNG so opening-random plies are seeded reproducibly
        # but diverse across games.
        match_rng = random.Random(seed + 31_337 + i)
        outcome = _play_match(
            p1, p2,
            max_halfmoves=max_halfmoves,
            opening_random_k=opening_random_k,
            rng=match_rng,
        )
        if outcome == 0:
            draws += 1
        elif outcome == cand_side:
            wins += 1
        else:
            losses += 1
        if (i + 1) % 10 == 0 or i + 1 == n_games:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_games - (i + 1)) / rate if rate > 0 else 0.0
            print(f"  [{i+1:>4d}/{n_games}]  W={wins:>3d}  D={draws:>3d}  L={losses:>3d}  "
                  f"rate={rate:.2f} g/s  eta={eta/60:.1f} min", flush=True)

    score = (wins + 0.5 * draws) / n_games
    elo_diff = _score_to_elo_diff(score)
    lo, hi = _wilson_ci(wins + 0.5 * draws, n_games)
    elo_lo = _score_to_elo_diff(lo) if 0.0 < lo < 1.0 else (-math.inf if lo <= 0 else math.inf)
    elo_hi = _score_to_elo_diff(hi) if 0.0 < hi < 1.0 else (-math.inf if hi <= 0 else math.inf)
    return {
        "candidate": candidate_label,
        "opponent": opponent_label,
        "n_games": n_games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": score,
        "elo_diff": elo_diff,
        "elo_ci95": (elo_lo, elo_hi),
        "p1_starts": p1_starts,
        "p2_starts": p2_starts,
        "elapsed_seconds": time.time() - t0,
    }


def _fmt_elo(x: float) -> str:
    if x == math.inf:
        return "+∞"
    if x == -math.inf:
        return "-∞"
    return f"{x:+.0f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("checkpoint", type=Path, help="Candidate checkpoint .pt file")
    parser.add_argument("--vs-checkpoint", type=Path, default=None,
                        help="Compare two checkpoints head-to-head instead of vs minimax")
    parser.add_argument("--depth", type=int, default=3,
                        help="Minimax depth for the opponent (default 3 = Phase 2 gate)")
    parser.add_argument("--vs-random", action="store_true",
                        help="Use RandomAgent as opponent instead of minimax")
    parser.add_argument("--num-games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-halfmoves", type=int, default=200)
    parser.add_argument(
        "--opening-random-k",
        type=int,
        default=4,
        help="First K plies of each game are random (default 4). Required for "
             "statistical validity when both agents are deterministic argmax — "
             "without it, only 2 distinct games are played and replayed N/2 times.",
    )
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--use-mcts", action="store_true",
                        help="Use MCTS at inference (NetworkAgent) instead of bare argmax. "
                             "Defeats the Phase-2-gate purpose but useful post-Phase-3.")
    parser.add_argument("--num-sims", type=int, default=400,
                        help="MCTS simulations per move (only with --use-mcts)")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        sys.exit(f"checkpoint not found: {args.checkpoint}")

    device = torch.device(args.device)
    network, step, config = _load_network(args.checkpoint, device)
    n_params = sum(p.numel() for p in network.parameters())
    cand_label = f"net[{args.checkpoint.name}|step={step}|{n_params:,}p]"

    if args.use_mcts:
        cand_label += f"+MCTS{args.num_sims}"

        def candidate_factory(seed: int):
            del seed  # NetworkAgent uses no seeded RNG; sides alternate via the harness
            return NetworkAgent(network, device, num_simulations=args.num_sims)
    else:
        cand_label += "+ARGMAX"

        def candidate_factory(seed: int):
            del seed
            return BareNetworkAgent(network, device)

    # ----- opponent ----------------------------------------------------------
    if args.vs_checkpoint is not None:
        if not args.vs_checkpoint.exists():
            sys.exit(f"vs-checkpoint not found: {args.vs_checkpoint}")
        opp_net, opp_step, _ = _load_network(args.vs_checkpoint, device)
        opp_label = f"net[{args.vs_checkpoint.name}|step={opp_step}]+ARGMAX"
        def opponent_factory(seed: int):
            del seed
            return BareNetworkAgent(opp_net, device)
    elif args.vs_random:
        opp_label = "random"
        def opponent_factory(seed: int):
            return RandomAgent(seed=seed)
    else:
        opp_label = f"minimax(depth={args.depth})"
        def opponent_factory(seed: int):
            del seed
            return MinimaxAgent(depth=args.depth)

    print(f"\n  ELO eval :  {cand_label}  vs  {opp_label}")
    print(f"  num_games = {args.num_games}  max_halfmoves = {args.max_halfmoves}  "
          f"opening_random_k = {args.opening_random_k}\n")

    result = run_eval(
        candidate_factory=candidate_factory,
        opponent_factory=opponent_factory,
        n_games=args.num_games,
        candidate_label=cand_label,
        opponent_label=opp_label,
        seed=args.seed,
        max_halfmoves=args.max_halfmoves,
        opening_random_k=args.opening_random_k,
    )

    print()
    print(f"  ───────────────────  RESULTS  ───────────────────")
    print(f"  {result['candidate']}")
    print(f"    vs")
    print(f"  {result['opponent']}")
    print()
    print(f"  Wins      : {result['wins']:>4d}  ({result['wins']/result['n_games']*100:5.1f} %)")
    print(f"  Draws     : {result['draws']:>4d}  ({result['draws']/result['n_games']*100:5.1f} %)")
    print(f"  Losses    : {result['losses']:>4d}  ({result['losses']/result['n_games']*100:5.1f} %)")
    print(f"  Score     : {result['score']:.4f}   (wins + 0.5×draws) / N")
    elo_lo, elo_hi = result["elo_ci95"]
    print(f"  ELO Δ     : {_fmt_elo(result['elo_diff'])}   (95 % CI [{_fmt_elo(elo_lo)}, {_fmt_elo(elo_hi)}])")
    print(f"  Sides     : P1={result['p1_starts']}  P2={result['p2_starts']}")
    print(f"  Elapsed   : {result['elapsed_seconds']/60:.1f} min")
    print()

    # Phase 2 → 3 gate decision (only meaningful when comparing bare network vs minimax d3)
    if not args.use_mcts and args.vs_checkpoint is None and not args.vs_random and args.depth == 3:
        gate_ok = result["score"] >= 0.50
        print("  ╔════════════════════════════════════════════╗")
        if gate_ok:
            print(f"  ║  PHASE 2 → 3 GATE  :  ✓  PASSED            ║")
            print(f"  ║  score {result['score']:.2f} ≥ 0.50  → fine-tune via self-play ║")
        else:
            print(f"  ║  PHASE 2 → 3 GATE  :  ✗  NOT YET           ║")
            print(f"  ║  score {result['score']:.2f} <  0.50  → continue warmup    ║")
        print("  ╚════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
