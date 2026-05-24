"""Supervised pre-training loop for Morris warmup.

Reads JSONL warmup traces (produced by `scripts/generate_warmup_dataset.py`),
fits the network to minimax-derived policy/value targets, evaluates the bare
network against baseline agents, and saves a checkpoint that the existing
self-play pipeline can load to continue training.

The training engine itself is just a thin wrapper around `trainer.compute_loss`
(reused as-is) + an Adam optimizer + an optional GradScaler. The loop is
purely supervised — no MCTS, no self-play, no replay buffer.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from morris_rl.data.dataset import (
    WarmupDataset,
    augment_batch,
    plain_collate,
    split_warmup_dataset,
)
from morris_rl.env.board import ACTION_SPACE_SIZE
from morris_rl.env.encoding_graph import encode_state_graph
from morris_rl.env.rules import (
    Outcome,
    apply_action,
    get_legal_actions,
    initial_state,
    is_terminal,
)
from morris_rl.eval.baselines import MinimaxAgent, RandomAgent
from morris_rl.network.factory import build_network
from morris_rl.training.trainer import compute_loss
from morris_rl.utils.checkpoints import save_checkpoint


# ---------------------------------------------------------------------------
# Trainer wrapper
# ---------------------------------------------------------------------------


class SupervisedTrainer:
    """Minimalist supervised optimizer around `trainer.compute_loss`."""

    def __init__(
        self,
        network: nn.Module,
        device: torch.device,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        aux_weight_mill: float = 0.3,
        aux_weight_pieces: float = 0.3,
        value_head_type: str = "scalar",
        aux_heads_enabled: bool = True,
        mixed_precision: bool = False,
        max_grad_norm: float = 1.0,
    ) -> None:
        self.network = network.to(device)
        self.device = device
        self.value_head_type = value_head_type
        self.aux_heads_enabled = aux_heads_enabled
        self.aux_weight_mill = float(aux_weight_mill)
        self.aux_weight_pieces = float(aux_weight_pieces)
        self.max_grad_norm = float(max_grad_norm)

        # AdamW with decoupled weight decay — matches the self-play Trainer
        # and standard modern practice (BERT, GPT, KataGo). The two trainers
        # use the same optimizer family so checkpoints flow cleanly when the
        # supervised warmup state is loaded into Phase 3.
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.network.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
        # AMP is only useful on CUDA; silently disable on CPU.
        self._amp_enabled = mixed_precision and device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self._amp_enabled)

    def _forward(self, x: torch.Tensor, legal_mask: torch.Tensor):
        """Forward returning (log_policy, value_scalar, value_logits, mill, pieces)."""
        if self.aux_heads_enabled:
            if self.value_head_type == "categorical":
                log_p, val, val_logits, mill_pred, pieces_pred = self.network(
                    x, legal_mask, return_value_logits=True, return_aux=True
                )
                return log_p, val, val_logits, mill_pred, pieces_pred
            log_p, val, mill_pred, pieces_pred = self.network(
                x, legal_mask, return_aux=True
            )
            return log_p, val, None, mill_pred, pieces_pred
        if self.value_head_type == "categorical":
            log_p, val, val_logits = self.network(x, legal_mask, return_value_logits=True)
            return log_p, val, val_logits, None, None
        log_p, val = self.network(x, legal_mask)
        return log_p, val, None, None, None

    def _compute(self, batch) -> tuple[torch.Tensor, dict[str, float]]:
        x, policy_tgt, value_tgt, mill_tgt, pieces_tgt, legal_mask, has_policy = batch
        x = x.to(self.device, non_blocking=True)
        policy_tgt = policy_tgt.to(self.device, non_blocking=True)
        value_tgt = value_tgt.to(self.device, non_blocking=True)
        mill_tgt = mill_tgt.to(self.device, non_blocking=True)
        pieces_tgt = pieces_tgt.to(self.device, non_blocking=True)
        legal_mask = legal_mask.to(self.device, non_blocking=True)
        has_policy = has_policy.to(self.device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=self._amp_enabled):
            log_p, val, val_logits, mill_pred, pieces_pred = self._forward(x, legal_mask)
            total, p_loss, v_loss, m_loss, pc_loss = compute_loss(
                log_policy=log_p,
                value=val,
                policy_target=policy_tgt,
                value_target=value_tgt,
                value_logits=val_logits,
                mill_diff_pred=mill_pred,
                pieces_diff_pred=pieces_pred,
                mill_diff_target=mill_tgt,
                pieces_diff_target=pieces_tgt,
                aux_weight_mill=self.aux_weight_mill if self.aux_heads_enabled else 0.0,
                aux_weight_pieces=self.aux_weight_pieces if self.aux_heads_enabled else 0.0,
                policy_mask=has_policy,
            )
        return total, {
            "total": float(total.detach()),
            "policy": float(p_loss.detach()),
            "value": float(v_loss.detach()),
            "mill": float(m_loss.detach()),
            "pieces": float(pc_loss.detach()),
        }

    def step(self, batch) -> dict[str, float]:
        self.network.train()
        self.optimizer.zero_grad(set_to_none=True)
        total, components = self._compute(batch)
        if self._amp_enabled:
            self.scaler.scale(total).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
            self.optimizer.step()
        return components

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[str, float]:
        self.network.eval()
        sums: dict[str, float] = defaultdict(float)
        n = 0
        for batch in loader:
            _total, components = self._compute(batch)
            for k, v in components.items():
                sums[k] += v
            n += 1
        return {k: v / max(n, 1) for k, v in sums.items()}


# ---------------------------------------------------------------------------
# Bare-network agent (no MCTS) for evaluation
# ---------------------------------------------------------------------------


class BareNetworkAgent:
    """Argmax over network policy on legal actions. No tree search.

    Used at eval time to track whether the bare prior is already strong enough
    to satisfy the user-defined warmup-ready criteria.
    """

    def __init__(self, network: nn.Module, device: torch.device, encode_fn=encode_state_graph) -> None:
        self.network = network
        self.device = device
        self.encode_fn = encode_fn

    @torch.no_grad()
    def select_action(self, state) -> int:
        legal = get_legal_actions(state)
        if not legal:
            raise ValueError("BareNetworkAgent called on terminal state")
        x = self.encode_fn(state).to(self.device)
        legal_mask = torch.zeros(1, ACTION_SPACE_SIZE, dtype=torch.bool, device=self.device)
        legal_mask[0, legal] = True
        self.network.eval()
        log_p, *_ = self.network(x, legal_mask)
        # log_p is already log-softmax over the full action space with illegal
        # actions at -inf; argmax over the entire vector is safe.
        return int(torch.argmax(log_p[0]).item())


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------


def _play_match(
    agent_p1,
    agent_p2,
    max_halfmoves: int = 200,
    opening_random_k: int = 0,
    rng=None,
) -> int:
    """Return outcome ∈ {0=DRAW, 1=P1, 2=P2}. Cap at max_halfmoves → DRAW.

    When `opening_random_k > 0` and `rng` is provided, the first K plies are
    played uniformly at random. This breaks the determinism between two
    argmax-based agents (network argmax + minimax argmax) that would
    otherwise produce only 2 distinct games no matter how many we run.
    """
    import random as _random
    agents = {1: agent_p1, 2: agent_p2}
    state = initial_state()
    halfmove_idx = 0
    while True:
        if state.total_halfmoves >= max_halfmoves:
            return 0
        done, outcome = is_terminal(state)
        if done:
            return 0 if (outcome is None or outcome == Outcome.DRAW) else int(outcome)
        if halfmove_idx < opening_random_k and rng is not None:
            action = rng.choice(get_legal_actions(state))
        else:
            action = agents[state.current_player].select_action(state)
        state = apply_action(state, int(action))
        halfmove_idx += 1


def eval_vs_baselines(
    network: nn.Module,
    device: torch.device,
    n_random: int = 100,
    n_d3: int = 50,
    base_seed: int = 0,
    opening_random_k: int = 4,
) -> dict[str, float]:
    """Play `n_random` games vs RandomAgent and `n_d3` vs MinimaxAgent(depth=3).

    Returns winrate / drawrate / lossrate from the network's POV. The network
    alternates between P1 and P2 across games to wash out first-mover bias.

    `opening_random_k=4` is critical when both agents are deterministic
    argmax (network bare + minimax) — without it, all 50 d3 games collapse
    onto 2 distinct game trajectories.
    """
    import random as _random
    net_agent = BareNetworkAgent(network, device)

    def play_series(opponent_factory, n: int, seed_offset: int) -> dict[str, float]:
        wins = draws = losses = 0
        for i in range(n):
            opp = opponent_factory(seed=base_seed + seed_offset + i)
            net_side = 1 if i % 2 == 0 else 2
            if net_side == 1:
                p1, p2 = net_agent, opp
            else:
                p1, p2 = opp, net_agent
            match_rng = _random.Random(base_seed + 31_337 + seed_offset + i)
            outcome = _play_match(
                p1, p2,
                opening_random_k=opening_random_k,
                rng=match_rng,
            )
            if outcome == 0:
                draws += 1
            elif outcome == net_side:
                wins += 1
            else:
                losses += 1
        return {
            "winrate": wins / max(n, 1),
            "drawrate": draws / max(n, 1),
            "lossrate": losses / max(n, 1),
        }

    out: dict[str, float] = {}
    rnd = play_series(lambda seed: RandomAgent(seed=seed), n_random, 0)
    out.update({
        "winrate_vs_random": rnd["winrate"],
        "drawrate_vs_random": rnd["drawrate"],
        "lossrate_vs_random": rnd["lossrate"],
    })
    d3 = play_series(lambda seed: MinimaxAgent(depth=3), n_d3, 10_000)
    out.update({
        "winrate_vs_d3": d3["winrate"],
        "drawrate_vs_d3": d3["drawrate"],
        "lossrate_vs_d3": d3["lossrate"],
        "non_loss_vs_d3": d3["winrate"] + d3["drawrate"],
    })
    return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


@dataclass
class TrainArgs:
    """Resolved CLI args. Mirrors `scripts/train_supervised.py` --flag set."""

    warmup_dir: Path
    out_dir: Path
    network_type: str = "graphnet"
    num_blocks: int = 4
    num_channels: int = 128
    policy_head_hidden: int = 64
    value_head_hidden: int = 64
    value_head_type: str = "scalar"
    aux_heads_enabled: bool = True
    aux_head_hidden: int = 64
    aux_weight_mill: float = 0.3
    aux_weight_pieces: float = 0.3
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    gamma: float = 1.0
    policy_temperature: float = 1.0
    val_split: float = 0.1
    val_seed: int = 0
    epochs: int = 40
    early_stop_patience: int = 5
    early_stop_disabled: bool = False
    eval_every: int = 5
    n_eval_random: int = 100
    n_eval_d3: int = 50
    device: str = "cpu"
    mixed_precision: bool = False
    num_workers: int = 0
    max_games: int | None = None
    seed: int = 0


def _build_network_config(args: TrainArgs) -> dict[str, Any]:
    """Build the same config schema self-play uses so checkpoints are reloadable."""
    return {
        "network": {
            "type": args.network_type,
            "num_blocks": args.num_blocks,
            "num_channels": args.num_channels,
            "policy_head_hidden": args.policy_head_hidden,
            "value_head_hidden": args.value_head_hidden,
            "value_head_type": args.value_head_type,
        },
        "aux_heads": {
            "enabled": args.aux_heads_enabled,
            "hidden_size": args.aux_head_hidden,
        },
        "input_encoding": {
            "num_planes": 11 if args.network_type == "graphnet" else 7,
        },
        "training": {
            "supervised_warmup": True,
            "gamma": args.gamma,
            "policy_temperature": args.policy_temperature,
        },
    }


def _log_scalars(writer, prefix: str, metrics: dict[str, float], step: int) -> None:
    if writer is None:
        return
    for k, v in metrics.items():
        writer.add_scalar(f"{prefix}/{k}", v, step)


def train_supervised(args: TrainArgs) -> dict[str, Any]:
    """Run the full supervised warmup pipeline. Returns final summary dict."""
    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  loading dataset from {args.warmup_dir}", flush=True)
    t0 = time.time()
    dataset = WarmupDataset(
        warmup_dir=args.warmup_dir,
        encode_fn=encode_state_graph if args.network_type == "graphnet" else None,
        gamma=args.gamma,
        policy_temperature=args.policy_temperature,
        max_games=args.max_games,
    )
    print(f"  dataset: {dataset.summary()}  (loaded in {time.time() - t0:.1f}s)", flush=True)

    train_ds, val_ds = split_warmup_dataset(dataset, val_ratio=args.val_split, seed=args.val_seed)
    print(f"  split: train={len(train_ds)}  val={len(val_ds)}", flush=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=augment_batch(batch_rng_seed=args.seed),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=plain_collate,
    )

    config = _build_network_config(args)
    network = build_network(OmegaConf.create(config))
    n_params = sum(p.numel() for p in network.parameters())
    print(f"  network: {args.network_type}  params={n_params:,}", flush=True)

    trainer = SupervisedTrainer(
        network=network,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        aux_weight_mill=args.aux_weight_mill,
        aux_weight_pieces=args.aux_weight_pieces,
        value_head_type=args.value_head_type,
        aux_heads_enabled=args.aux_heads_enabled,
        mixed_precision=args.mixed_precision,
    )

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(args.out_dir / "tb"))
    except ImportError:
        writer = None

    best_val_loss = float("inf")
    best_epoch = -1
    patience = 0
    global_step = 0
    history: list[dict[str, Any]] = []

    for epoch in range(args.epochs):
        # ----- train epoch -----
        train_sums: dict[str, float] = defaultdict(float)
        train_n = 0
        ep_t0 = time.time()
        for batch in train_loader:
            comps = trainer.step(batch)
            for k, v in comps.items():
                train_sums[k] += v
            train_n += 1
            global_step += 1
        train_metrics = {k: v / max(train_n, 1) for k, v in train_sums.items()}
        _log_scalars(writer, "train", train_metrics, epoch)

        # ----- val epoch -----
        val_metrics = trainer.evaluate(val_loader)
        _log_scalars(writer, "val", val_metrics, epoch)

        ep_dt = time.time() - ep_t0
        print(
            f"  epoch {epoch:>3}  "
            f"train_loss={train_metrics['total']:.4f}  "
            f"(p={train_metrics['policy']:.3f} v={train_metrics['value']:.3f} "
            f"m={train_metrics['mill']:.3f} pc={train_metrics['pieces']:.3f})  "
            f"val_loss={val_metrics['total']:.4f}  "
            f"({ep_dt:.1f}s)",
            flush=True,
        )

        # ----- eval vs baselines -----
        eval_metrics: dict[str, float] | None = None
        if args.eval_every > 0 and (epoch + 1) % args.eval_every == 0:
            eval_t0 = time.time()
            eval_metrics = eval_vs_baselines(
                trainer.network,
                device,
                n_random=args.n_eval_random,
                n_d3=args.n_eval_d3,
                base_seed=args.seed + epoch * 1_000,
            )
            _log_scalars(writer, "eval", eval_metrics, epoch)
            print(
                f"    eval  vs_random={eval_metrics['winrate_vs_random']*100:.1f}%W "
                f"{eval_metrics['drawrate_vs_random']*100:.1f}%D  "
                f"vs_d3={eval_metrics['winrate_vs_d3']*100:.1f}%W "
                f"{eval_metrics['drawrate_vs_d3']*100:.1f}%D "
                f"{eval_metrics['lossrate_vs_d3']*100:.1f}%L  "
                f"({time.time() - eval_t0:.1f}s)",
                flush=True,
            )

        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "eval": eval_metrics,
        })

        # ----- checkpoint best ----
        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            best_epoch = epoch
            patience = 0
            save_checkpoint(
                args.out_dir / "best.pt",
                state_dict=trainer.network.state_dict(),
                config=config,
                step=global_step,
            )
        else:
            patience += 1
            if not args.early_stop_disabled and patience >= args.early_stop_patience:
                print(f"  early stop at epoch {epoch} (no val improvement for {patience} epochs)", flush=True)
                break

    # Final checkpoint regardless.
    save_checkpoint(
        args.out_dir / "final.pt",
        state_dict=trainer.network.state_dict(),
        config=config,
        step=global_step,
    )
    if writer is not None:
        writer.close()
    summary = {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "total_epochs_run": len(history),
        "history": history,
        "n_params": n_params,
    }
    return summary
