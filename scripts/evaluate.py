"""Evaluate a trained checkpoint against a baseline (random or minimax).

Usage
-----
    # 100 games against minimax depth 3 on CPU
    python scripts/evaluate.py outputs/.../checkpoints/checkpoint_00007000.pt \
        --opponent minimax --depth 3 --num-games 100

    # vs random, 50 games, faster
    python scripts/evaluate.py path/to/checkpoint.pt --opponent random --num-games 50

    # GPU + more MCTS sims (sharper play, slower)
    python scripts/evaluate.py path/to/checkpoint.pt --device cuda --num-simulations 800

The script alternates which agent plays first across games (handled by
``run_arena``) so first-move bias is averaged out.

Network architecture is read from the checkpoint's stored config — no need
to pass --num-blocks / --num-channels by hand.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

# Allow running as `python scripts/evaluate.py` from the repo root.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morris_rl.eval.arena import ArenaSummary, run_arena
from morris_rl.eval.baselines import MinimaxAgent, NetworkAgent, RandomAgent
from morris_rl.network.resnet import MorrisResNet
from morris_rl.utils.checkpoints import load_checkpoint
from morris_rl.utils.logging import logger, setup_logging
from morris_rl.utils.seeding import seed_everything


def _load_network(checkpoint_path: Path, device: torch.device) -> tuple[MorrisResNet, int]:
    """Rebuild the network described by the checkpoint and load its weights."""
    payload = load_checkpoint(checkpoint_path)
    cfg = payload["config"]
    net_cfg = cfg["network"]
    enc_cfg = cfg["input_encoding"]

    network = MorrisResNet(
        num_blocks=net_cfg["num_blocks"],
        num_channels=net_cfg["num_channels"],
        num_planes=enc_cfg["num_planes"],
        policy_head_hidden=net_cfg["policy_head_hidden"],
        value_head_hidden=net_cfg["value_head_hidden"],
        value_head_type=net_cfg.get("value_head_type", "scalar"),
    )
    network.load_state_dict(payload["state_dict"])
    network.eval().to(device)
    return network, int(payload["step"])


def _build_opponent(name: str, depth: int, seed: int) -> tuple[object, str]:
    if name == "random":
        return RandomAgent(seed=seed), "RandomAgent"
    if name == "minimax":
        return MinimaxAgent(depth=depth), f"Minimax(depth={depth})"
    raise ValueError(f"unknown opponent {name!r}")


def _print_summary(summary: ArenaSummary, network_label: str, opponent_label: str) -> None:
    total = summary.total_games
    if total == 0:
        logger.warning("no games played")
        return

    win_pct = summary.agent_a_wins / total * 100
    loss_pct = summary.agent_b_wins / total * 100
    draw_pct = summary.draws / total * 100

    print()
    print(f"{network_label}  vs  {opponent_label}")
    print("─" * 60)
    print(f"  Games played    : {total}")
    print(f"  Network wins    : {summary.agent_a_wins:>4}  ({win_pct:5.1f}%)")
    print(f"  Network losses  : {summary.agent_b_wins:>4}  ({loss_pct:5.1f}%)")
    print(f"  Draws           : {summary.draws:>4}  ({draw_pct:5.1f}%)")
    print(f"  Score (W + ½D)  : {summary.win_rate_a * 100:5.1f}%")
    print("─" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an AlphaZero checkpoint against a baseline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("checkpoint", type=Path, help="Path to a .pt checkpoint file.")
    parser.add_argument(
        "--opponent",
        choices=["random", "minimax"],
        default="minimax",
        help="Baseline opponent.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="Minimax search depth (ignored for random).",
    )
    parser.add_argument(
        "--num-games", type=int, default=100, help="Total games to play."
    )
    parser.add_argument(
        "--num-simulations",
        type=int,
        default=200,
        help="MCTS simulations per move for the network agent.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for the network ('cpu' or 'cuda'). CPU is the default since "
        "evaluation is rarely the throughput bottleneck.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for the random opponent."
    )
    parser.add_argument("--verbose", action="store_true", help="Log progress every 10 games.")

    args = parser.parse_args()

    setup_logging()
    seed_everything(args.seed)

    if not args.checkpoint.exists():
        parser.error(f"checkpoint not found: {args.checkpoint}")

    device = torch.device(args.device)
    logger.info(f"Loading {args.checkpoint} on {device}")
    network, step = _load_network(args.checkpoint, device)
    n_params = sum(p.numel() for p in network.parameters())
    logger.info(f"Network: {n_params:,} params, trained step={step}")

    network_agent = NetworkAgent(network, device, num_simulations=args.num_simulations)
    network_label = f"Net@step={step} ({args.num_simulations} sims)"

    opponent, opponent_label = _build_opponent(args.opponent, args.depth, args.seed)

    logger.info(
        f"Running {args.num_games} games: {network_label}  vs  {opponent_label}"
    )
    t0 = time.perf_counter()
    summary = run_arena(network_agent, opponent, num_games=args.num_games, verbose=args.verbose)
    elapsed = time.perf_counter() - t0
    logger.info(f"Done in {elapsed:.1f}s ({elapsed / args.num_games:.2f}s/game)")

    _print_summary(summary, network_label, opponent_label)


if __name__ == "__main__":
    main()
