"""AlphaZero training loop for Nine Men's Morris.

One training *step* samples a minibatch from the replay buffer, runs a forward
pass, computes the combined policy + value loss, and performs a gradient update
with optional mixed-precision (AMP) and gradient clipping.

The :meth:`Trainer.train_concurrent` method orchestrates the full pipeline:
it interleaves game collection from a :class:`~morris_rl.training.self_play.SelfPlayManager`
with gradient updates, broadcasts fresh weights to workers after each batch of
updates, and checkpoints periodically.

Loss
----
Following AlphaZero:
    L = L_policy + L_value
    L_policy = -Σ_a  π(a) · log p_θ(a)   (cross-entropy, MCTS visits vs network)
    L_value  = (v_θ - z)²                 (MSE, network value vs game outcome)

Weight decay (L2 regularisation) is handled by the Adam optimiser.

Action masking during training
-------------------------------
During self-play the network is called with the exact legal-action mask so that
illegal actions get zero probability.  During training we pass a full-True mask
so that log_softmax is computed over all 600 actions.  This is valid because
the MCTS visit distribution stored in the replay buffer already has zero mass on
illegal actions; the cross-entropy loss gradient therefore only trains the
log-probabilities of legal moves.  Illegal actions get no explicit push-to-zero
signal, but the masking at inference time guarantees they are never selected.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

from morris_rl.env.board import ACTION_SPACE_SIZE
from morris_rl.training.replay_buffer import ReplayBuffer
from morris_rl.training.self_play import SelfPlayManager
from morris_rl.utils.checkpoints import load_checkpoint, save_checkpoint
from morris_rl.utils.logging import logger

try:
    from torch.utils.tensorboard import SummaryWriter as _SummaryWriter

    _TENSORBOARD_AVAILABLE = True
except ImportError:
    _TENSORBOARD_AVAILABLE = False

try:
    import psutil

    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

# Memory/queue health is logged this often to spot leaks early without spamming.
_MEMORY_LOG_INTERVAL = 50
# How many recent games to keep for histogram aggregation in TensorBoard.
_GAME_LENGTH_WINDOW = 200


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def compute_loss(
    log_policy: torch.Tensor,
    value: torch.Tensor,
    policy_target: torch.Tensor,
    value_target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute AlphaZero combined loss.

    Args:
        log_policy:    (batch, ACTION_SPACE_SIZE) log-probabilities from network.
        value:         (batch,) value predictions in [-1, 1].
        policy_target: (batch, ACTION_SPACE_SIZE) MCTS visit distribution.
        value_target:  (batch,) game outcomes in {-1, 0, +1}.

    Returns:
        Tuple of (total_loss, policy_loss, value_loss).
    """
    policy_loss = -(policy_target * log_policy).sum(dim=1).mean()
    value_loss = F.mse_loss(value, value_target)
    return policy_loss + value_loss, policy_loss, value_loss


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """Manages the AlphaZero training loop.

    Args:
        network:              The policy/value network to train.
        device:               Target device (CPU or CUDA).
        learning_rate:        Initial Adam learning rate.
        weight_decay:         L2 regularisation coefficient.
        max_grad_norm:        Gradient clipping threshold (inf to disable).
        lr_decay_steps:       CosineAnnealingLR period (in gradient steps).
        mixed_precision:      Enable AMP; silently ignored on CPU.
        log_dir:              TensorBoard log directory (None to disable).
        checkpoint_dir:       Directory for periodic .pt checkpoints.
        checkpoint_interval:  Gradient steps between automatic checkpoints.
        config:               Arbitrary config dict stored in checkpoints.
    """

    def __init__(
        self,
        network: nn.Module,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        max_grad_norm: float = 1.0,
        lr_decay_steps: int = 200_000,
        mixed_precision: bool = True,
        log_dir: str | Path | None = None,
        checkpoint_dir: str | Path | None = None,
        checkpoint_interval: int = 1000,
        config: dict[str, Any] | None = None,
    ) -> None:
        self._network = network.to(device)
        self._device = device
        self._max_grad_norm = max_grad_norm
        self._checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self._checkpoint_interval = checkpoint_interval
        self._config: dict[str, Any] = config or {}
        self._step = 0
        # Optional buffer ref so _auto_checkpoint can persist it alongside weights.
        self._buffer: ReplayBuffer | None = None

        self._optimizer = torch.optim.Adam(
            network.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self._optimizer,
            T_max=lr_decay_steps,
            eta_min=learning_rate * 1e-2,
        )

        # AMP is only beneficial on CUDA; silently disable on CPU.
        self._amp_enabled = mixed_precision and device.type == "cuda"
        self._scaler = torch.amp.GradScaler("cuda", enabled=self._amp_enabled)  # type: ignore[attr-defined]

        if log_dir is not None and _TENSORBOARD_AVAILABLE:
            self._writer: _SummaryWriter | None = _SummaryWriter(log_dir=str(log_dir))
        else:
            self._writer = None

    # ------------------------------------------------------------------
    # Core training step
    # ------------------------------------------------------------------

    def step(self, buffer: ReplayBuffer, batch_size: int) -> dict[str, float]:
        """One gradient update.

        Args:
            buffer:     Replay buffer to sample from.
            batch_size: Number of samples per minibatch.

        Returns:
            Dict with keys ``total_loss``, ``policy_loss``, ``value_loss``,
            ``learning_rate``.
        """
        states, policy_targets, value_targets = buffer.sample(batch_size, device=self._device)

        # Full-True mask: training does not filter illegal actions (see module docstring).
        full_mask = torch.ones(states.shape[0], ACTION_SPACE_SIZE, dtype=torch.bool, device=self._device)

        self._optimizer.zero_grad()

        with torch.autocast(device_type=self._device.type, enabled=self._amp_enabled):
            log_policy, value = self._network(states, full_mask)
            total_loss, policy_loss, value_loss = compute_loss(
                log_policy, value, policy_targets, value_targets
            )

        self._scaler.scale(total_loss).backward()  # type: ignore[no-untyped-call]
        self._scaler.unscale_(self._optimizer)
        nn.utils.clip_grad_norm_(self._network.parameters(), self._max_grad_norm)
        self._scaler.step(self._optimizer)
        self._scaler.update()
        self._scheduler.step()
        self._step += 1

        metrics: dict[str, float] = {
            "total_loss": float(total_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "learning_rate": float(self._scheduler.get_last_lr()[0]),
        }
        self._log_metrics(metrics)

        if self._checkpoint_dir and self._step % self._checkpoint_interval == 0:
            self._auto_checkpoint()

        return metrics

    # ------------------------------------------------------------------
    # Concurrent training + self-play loop
    # ------------------------------------------------------------------

    def train_concurrent(
        self,
        manager: SelfPlayManager,
        buffer: ReplayBuffer,
        batch_size: int,
        total_steps: int,
        min_buffer_size: int,
        updates_per_game: int = 4,
    ) -> None:
        """Interleave self-play game collection with gradient updates.

        The loop:
          1. Collect games until ``min_buffer_size`` samples are in the buffer.
          2. For each subsequent game, run ``updates_per_game`` gradient steps.
          3. Broadcast updated weights to workers after each batch of updates.
          4. Stop after ``total_steps`` gradient steps.

        Args:
            manager:          Running :class:`SelfPlayManager` (already started).
            buffer:           Replay buffer shared between self-play and training.
            batch_size:       Minibatch size for each gradient step.
            total_steps:      Total gradient steps to perform.
            min_buffer_size:  Buffer occupancy required before training begins.
            updates_per_game: Gradient steps performed per collected game.
        """
        games_collected = 0
        recent_lengths: deque[int] = deque(maxlen=_GAME_LENGTH_WINDOW)
        outcome_counts = {"p1_win": 0, "p2_win": 0, "draw": 0}
        self._buffer = buffer  # used by _auto_checkpoint to persist buffer state

        logger.info(f"Filling buffer to {min_buffer_size} samples (current: {len(buffer)})…")
        with tqdm(total=min_buffer_size, desc="Filling buffer", unit="samples", leave=True) as pbar:
            while len(buffer) < min_buffer_size:
                game = manager.collect_game()
                buffer.add_samples(game.samples)
                games_collected += 1
                added = len(game.samples)
                pbar.update(min(added, min_buffer_size - pbar.n))
                pbar.set_postfix({"games": games_collected, "len": len(game.samples)})

        logger.info(f"Buffer ready ({len(buffer)} samples). Starting training.")

        with tqdm(
            total=total_steps,
            initial=self._step,
            desc="Training",
            unit="step",
            leave=True,
            dynamic_ncols=True,
        ) as pbar:
            while self._step < total_steps:
                game = manager.collect_game()
                buffer.add_samples(game.samples)
                games_collected += 1
                recent_lengths.append(game.game_length)
                if game.outcome == 1:
                    outcome_counts["p1_win"] += 1
                elif game.outcome == 2:
                    outcome_counts["p2_win"] += 1
                else:
                    outcome_counts["draw"] += 1

                self._log_scalar("train/buffer_size", len(buffer))
                self._log_scalar("train/games_collected", games_collected)
                self._log_game_stats(recent_lengths, outcome_counts, games_collected)

                for _ in range(updates_per_game):
                    if self._step >= total_steps:
                        break
                    metrics = self.step(buffer, batch_size)
                    pbar.update(1)
                    pbar.set_postfix({
                        "loss": f"{metrics['total_loss']:.3f}",
                        "p": f"{metrics['policy_loss']:.3f}",
                        "v": f"{metrics['value_loss']:.3f}",
                        "buf": len(buffer),
                        "games": games_collected,
                    })
                    if self._step % _MEMORY_LOG_INTERVAL == 0:
                        self._log_memory_health(manager)

                manager.update_network(self._network.state_dict())

        logger.info(f"Training complete at step {self._step}.")

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save(self, path: str | Path, buffer: ReplayBuffer | None = None) -> Path:
        """Save a checkpoint to *path* and return the resolved path.

        If a buffer is supplied (or a buffer was registered via
        :meth:`train_concurrent`), its state is written to a sibling file
        ``<path>.buffer.npz``. Buffers can be hundreds of MB, so we keep them
        out of the model checkpoint to avoid bloating .pt files.
        """
        dest = Path(path)
        save_checkpoint(
            path=dest,
            state_dict=self._network.state_dict(),
            config=self._config,
            step=self._step,
        )
        logger.info(f"Checkpoint saved: {dest} (step {self._step})")
        target_buffer = buffer if buffer is not None else self._buffer
        if target_buffer is not None and len(target_buffer) > 0:
            buffer_path = dest.with_suffix(dest.suffix + ".buffer.npz")
            target_buffer.save(buffer_path)
            logger.info(f"Buffer saved: {buffer_path} ({len(target_buffer)} samples)")
        return dest

    def load(self, path: str | Path, buffer: ReplayBuffer | None = None) -> None:
        """Restore network weights and step counter from a checkpoint.

        If a buffer is supplied and a sibling ``<path>.buffer.npz`` exists, the
        buffer state is restored too — letting training resume immediately
        without re-warming up.
        """
        payload = load_checkpoint(path)
        self._network.load_state_dict(payload["state_dict"])
        self._step = payload["step"]
        logger.info(f"Resumed from {path} at step {self._step}")
        if buffer is not None:
            checkpoint_path = Path(path)
            buffer_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".buffer.npz")
            if buffer_path.exists():
                buffer.load(buffer_path)
                logger.info(f"Buffer restored: {buffer_path} ({len(buffer)} samples)")
            else:
                logger.warning(
                    f"No buffer sibling at {buffer_path}; resume will re-warmup."
                )

    @property
    def global_step(self) -> int:
        return self._step

    def close(self) -> None:
        """Flush TensorBoard writer."""
        if self._writer is not None:
            self._writer.close()

    def __enter__(self) -> Trainer:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        if self._writer is None:
            return
        for key, value in metrics.items():
            self._writer.add_scalar(f"train/{key}", value, self._step)

    def _log_scalar(self, tag: str, value: float | int) -> None:
        if self._writer is not None:
            self._writer.add_scalar(tag, value, self._step)

    def _log_game_stats(
        self,
        recent_lengths: deque[int],
        outcome_counts: dict[str, int],
        games_collected: int,
    ) -> None:
        """Log per-game scalar stats and a periodic histogram of recent lengths."""
        if self._writer is None or not recent_lengths:
            return
        last = recent_lengths[-1]
        self._writer.add_scalar("game/length_last", last, games_collected)
        mean_len = sum(recent_lengths) / len(recent_lengths)
        self._writer.add_scalar("game/length_mean_window", mean_len, games_collected)
        total = sum(outcome_counts.values()) or 1
        self._writer.add_scalar("game/p1_win_rate", outcome_counts["p1_win"] / total, games_collected)
        self._writer.add_scalar("game/p2_win_rate", outcome_counts["p2_win"] / total, games_collected)
        self._writer.add_scalar("game/draw_rate", outcome_counts["draw"] / total, games_collected)
        # Histograms are heavier; only emit one every full window refresh.
        if games_collected % _GAME_LENGTH_WINDOW == 0 and len(recent_lengths) == _GAME_LENGTH_WINDOW:
            self._writer.add_histogram(
                "game/length_distribution",
                torch.tensor(list(recent_lengths), dtype=torch.float32),
                games_collected,
            )

    def _log_memory_health(self, manager: SelfPlayManager) -> None:
        """Periodic check for memory/queue leaks. Cheap: ~1ms per call."""
        results_q = manager.results_qsize()
        weights_q_max = manager.weights_qsize_max()
        rss_gb = -1.0
        if _PSUTIL_AVAILABLE:
            rss_gb = psutil.Process().memory_info().rss / 1e9
            self._log_scalar("system/rss_gb", rss_gb)
        self._log_scalar("system/results_qsize", results_q)
        self._log_scalar("system/weights_qsize_max", weights_q_max)
        logger.info(
            f"step={self._step} rss={rss_gb:.2f}GB "
            f"results_q={results_q} weights_q_max={weights_q_max}"
        )

    def _auto_checkpoint(self) -> None:
        if self._checkpoint_dir is None:
            return
        path = self._checkpoint_dir / f"checkpoint_{self._step:08d}.pt"
        self.save(path)
