"""Supervised warmup training for Morris.

Trains a network on JSONL traces produced by
`scripts/generate_warmup_dataset.py`, fitting to minimax-derived policy
targets (softmax over root_scores), value targets (γ^(T-t) × outcome), and
auxiliary mill/pieces predictions. No MCTS, no self-play.

Output: a standard-format checkpoint loadable by the existing self-play
pipeline. Tracks `bare network vs random / minimax-d3` winrates as the
warmup-readiness criterion.

Usage
-----
    # Smoke test (~1 min CPU)
    uv run python scripts/train_supervised.py \\
        --warmup-dir /tmp/warmup_smoke \\
        --epochs 1 --batch-size 32 --device cpu --out-dir /tmp/sup_smoke

    # Full run (2-6 h GPU)
    uv run python scripts/train_supervised.py \\
        --warmup-dir outputs/warmup_d5_5k \\
        --epochs 40 --batch-size 512 --lr 1e-3 \\
        --device cuda --mixed-precision \\
        --out-dir outputs/sup_warmup_5k
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from morris_rl.training.supervised import TrainArgs, train_supervised


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--warmup-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)

    # Network
    parser.add_argument("--network-type", default="graphnet", choices=("graphnet", "resnet"))
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--num-channels", type=int, default=128)
    parser.add_argument("--policy-head-hidden", type=int, default=64)
    parser.add_argument("--value-head-hidden", type=int, default=64)
    parser.add_argument("--value-head-type", default="scalar", choices=("scalar", "categorical"))
    parser.add_argument("--aux-heads-enabled", action="store_true", default=True)
    parser.add_argument("--no-aux-heads", dest="aux_heads_enabled", action="store_false")
    parser.add_argument("--aux-head-hidden", type=int, default=64)
    parser.add_argument("--aux-weight-mill", type=float, default=0.3)
    parser.add_argument("--aux-weight-pieces", type=float, default=0.3)

    # Optimization
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--no-early-stop", action="store_true",
                        help="Disable early stopping entirely; always run --epochs epochs.")

    # Data
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--policy-temperature", type=float, default=1.0)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--val-seed", type=int, default=0)
    parser.add_argument("--max-games", type=int, default=None, help="Cap dataset for smoke tests.")

    # Eval
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--n-eval-random", type=int, default=100)
    parser.add_argument("--n-eval-d3", type=int, default=50)

    # Runtime
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    train_args = TrainArgs(
        warmup_dir=args.warmup_dir,
        out_dir=args.out_dir,
        network_type=args.network_type,
        num_blocks=args.num_blocks,
        num_channels=args.num_channels,
        policy_head_hidden=args.policy_head_hidden,
        value_head_hidden=args.value_head_hidden,
        value_head_type=args.value_head_type,
        aux_heads_enabled=args.aux_heads_enabled,
        aux_head_hidden=args.aux_head_hidden,
        aux_weight_mill=args.aux_weight_mill,
        aux_weight_pieces=args.aux_weight_pieces,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        gamma=args.gamma,
        policy_temperature=args.policy_temperature,
        val_split=args.val_split,
        val_seed=args.val_seed,
        epochs=args.epochs,
        early_stop_patience=args.early_stop_patience,
        early_stop_disabled=args.no_early_stop,
        eval_every=args.eval_every,
        n_eval_random=args.n_eval_random,
        n_eval_d3=args.n_eval_d3,
        device=args.device,
        mixed_precision=args.mixed_precision,
        num_workers=args.num_workers,
        max_games=args.max_games,
        seed=args.seed,
    )

    summary = train_supervised(train_args)
    print()
    print(f"  best epoch = {summary['best_epoch']}  best val_loss = {summary['best_val_loss']:.4f}")
    print(f"  total epochs run = {summary['total_epochs_run']}")
    print(f"  params = {summary['n_params']:,}")
    print(f"  checkpoints → {args.out_dir.resolve()}/best.pt + final.pt")


if __name__ == "__main__":
    main()
