"""Generate a warmup dataset of minimax-vs-minimax Morris games.

Streams JSONL files (one per worker) compatible with scripts/replay_game.py
and scripts/watch_warmup_games.py. Each game stores its full action sequence
plus per-position root scores from the alpha-beta search, ready for use as
policy targets by the supervised training phase.

Usage
-----
    # Phase 0 mini-gate (~5-15 min)
    uv run python scripts/generate_warmup_dataset.py \
        --num-games 100 --depth 5 --workers 10 \
        --out-dir outputs/warmup_d5_gate100

    # Phase 1 full (1-3 h)
    uv run python scripts/generate_warmup_dataset.py \
        --num-games 10000 --depth 5 --workers 10 \
        --epsilon 0.10 --opening-random-k 5 \
        --out-dir outputs/warmup_d5_10k

    # Smoke test (< 1 min)
    uv run python scripts/generate_warmup_dataset.py \
        --num-games 8 --workers 4 --depth 3 --out-dir /tmp/warmup_smoke
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from morris_rl.data.generator import generate_games_parallel


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a warmup dataset of minimax-vs-minimax Morris games.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--num-games", type=int, default=100, help="Total games to generate.")
    parser.add_argument("--depth", type=int, default=5, help="Minimax search depth in plies.")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.10,
        help="Probability of playing a random legal move (skips minimax to save CPU).",
    )
    parser.add_argument(
        "--opening-random-k",
        type=int,
        default=5,
        help="First K half-moves are random uniform (forces opening diversity).",
    )
    parser.add_argument("--workers", type=int, default=10, help="Parallel worker count.")
    parser.add_argument("--seed", type=int, default=0, help="Master seed.")
    parser.add_argument(
        "--max-halfmoves",
        type=int,
        default=200,
        help="Hard cap on game length. Games reaching this are DRAWs (outcome=0).",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print a progress line every N completed games.",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for JSONL files.")
    args = parser.parse_args()

    print(
        f"  generating {args.num_games} games  depth={args.depth}  ε={args.epsilon}  "
        f"opening_k={args.opening_random_k}  workers={args.workers}  cap={args.max_halfmoves}",
        flush=True,
    )
    print(f"  out_dir = {args.out_dir.resolve()}", flush=True)

    summary = generate_games_parallel(
        num_games=args.num_games,
        out_dir=args.out_dir,
        depth=args.depth,
        epsilon=args.epsilon,
        opening_random_k=args.opening_random_k,
        num_workers=args.workers,
        seed=args.seed,
        max_halfmoves=args.max_halfmoves,
        progress_every=args.progress_every,
    )

    print()
    print(
        f"  done: {summary.total_games} games  elapsed={summary.elapsed_seconds / 60:.1f} min  "
        f"throughput={summary.total_games / summary.elapsed_seconds:.2f} games/s"
    )
    print(f"  per-worker counts: {summary.per_worker_counts}")
    print(f"  files: {sorted(p.name for p in args.out_dir.glob('worker_*.jsonl'))}")


if __name__ == "__main__":
    main()
