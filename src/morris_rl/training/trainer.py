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
The same legal-action mask used at inference is used during training: the
log_softmax is normalised over the legal support only.  This matches the
distribution the MCTS visit target was sampled from (priors are masked at the
root) and avoids wasting gradient on logits for illegal actions that will be
discarded by the inference-time mask anyway.  The mask is stored alongside
each sample in the replay buffer (see :class:`SampleRecord.legal_mask`) so it
survives symmetry augmentation and buffer replay.
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

try:
    import mlflow as _mlflow

    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False


def _flatten_config(cfg: dict[str, Any], prefix: str = "") -> dict[str, str]:
    """Flatten a nested config dict to dotted keys for MLflow params."""
    flat: dict[str, str] = {}
    for k, v in cfg.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_flatten_config(v, key))
        else:
            # MLflow caps param values at 500 chars and rejects None silently
            # — stringify defensively. Nested lists become "[…]".
            flat[key] = str(v) if v is not None else ""
    return flat

# Memory/queue health is logged this often to spot leaks early without
# spamming the trainer's tqdm bar on stderr. TensorBoard captures the same
# scalars on every check so high-resolution data is still available.
_MEMORY_LOG_INTERVAL = 50          # how often to *check* (and write to TB)
_MEMORY_LOG_STDERR_INTERVAL = 500  # how often to ALSO emit a stderr line
# How many recent games to keep for histogram aggregation in TensorBoard.
_GAME_LENGTH_WINDOW = 200


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def _masked_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE that ignores NaN entries in target. Returns 0 when no valid samples."""
    valid = ~torch.isnan(target)
    if not valid.any():
        return pred.sum() * 0.0  # preserves device/dtype with zero grad
    diff = (pred[valid] - target[valid]) ** 2
    return diff.mean()


def compute_loss(
    log_policy: torch.Tensor,
    value: torch.Tensor,
    policy_target: torch.Tensor,
    value_target: torch.Tensor,
    value_logits: torch.Tensor | None = None,
    mill_diff_pred: torch.Tensor | None = None,
    pieces_diff_pred: torch.Tensor | None = None,
    mill_diff_target: torch.Tensor | None = None,
    pieces_diff_target: torch.Tensor | None = None,
    aux_weight_mill: float = 0.0,
    aux_weight_pieces: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute AlphaZero combined loss with optional KataGo-style aux heads.

    Args:
        log_policy:    (batch, ACTION_SPACE_SIZE) log-probabilities from network.
        value:         (batch,) scalar value predictions.
        policy_target: (batch, ACTION_SPACE_SIZE) MCTS visit distribution.
        value_target:  (batch,) game outcomes in {-1, 0, +1}.
        value_logits:  (batch, 3) categorical logits, or None for scalar head.
        mill_diff_pred, pieces_diff_pred: (batch,) aux head outputs, or None.
        mill_diff_target, pieces_diff_target: (batch,) signed targets; NaN
            entries are masked out so samples lacking aux supervision still
            contribute to policy/value losses.
        aux_weight_mill, aux_weight_pieces: scalar λ weights. 0 disables.

    Returns:
        (total_loss, policy_loss, value_loss, mill_loss, pieces_loss).
        Aux losses are 0 tensors when the respective head/target is missing.
    """
    # 0 × -inf = NaN when policy_target=0 on masked actions — zero out explicitly.
    contrib = policy_target * log_policy
    contrib = torch.where(policy_target > 0, contrib, torch.zeros_like(contrib))
    policy_loss = -contrib.sum(dim=1).mean()
    if value_logits is not None:
        # Map {+1→0 (win), 0→1 (draw), -1→2 (loss)} to class indices.
        value_loss = F.cross_entropy(value_logits, (1.0 - value_target).long())
    else:
        value_loss = F.mse_loss(value, value_target)

    zero = value_loss.detach() * 0.0  # device/dtype-correct zero
    if mill_diff_pred is not None and mill_diff_target is not None and aux_weight_mill > 0:
        mill_loss = _masked_mse(mill_diff_pred, mill_diff_target)
    else:
        mill_loss = zero
    if pieces_diff_pred is not None and pieces_diff_target is not None and aux_weight_pieces > 0:
        pieces_loss = _masked_mse(pieces_diff_pred, pieces_diff_target)
    else:
        pieces_loss = zero

    total = policy_loss + value_loss + aux_weight_mill * mill_loss + aux_weight_pieces * pieces_loss
    return total, policy_loss, value_loss, mill_loss, pieces_loss


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
        value_head_type: str = "scalar",
        mlflow_uri: str | None = None,
        mlflow_experiment: str = "morris-az",
        mlflow_run_name: str | None = None,
        aux_heads_enabled: bool = False,
        aux_weight_mill: float = 0.0,
        aux_weight_pieces: float = 0.0,
    ) -> None:
        self._network = network.to(device)
        self._device = device
        self._max_grad_norm = max_grad_norm
        self._checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self._checkpoint_interval = checkpoint_interval
        self._config: dict[str, Any] = config or {}
        self._value_head_type = value_head_type
        self._aux_heads_enabled = aux_heads_enabled
        self._aux_weight_mill = float(aux_weight_mill)
        self._aux_weight_pieces = float(aux_weight_pieces)
        self._step = 0
        self._learning_rate = learning_rate
        self._weight_decay = weight_decay
        self._lr_decay_steps = lr_decay_steps
        # Optional buffer ref so _auto_checkpoint can persist it alongside weights.
        self._buffer: ReplayBuffer | None = None

        # Filter to requires_grad=True params so the optimizer is automatically
        # limited to LoRA adapters when freeze_trunk() has been called before
        # creating the Trainer. When the trunk is not frozen this is equivalent
        # to network.parameters() — fully backward-compatible.
        self._optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, network.parameters()),
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

        # MLflow is opt-in; enabled only when an explicit tracking URI is given.
        # Errors during setup (server unreachable, bad creds) surface as warnings
        # — we never fail the run because the metrics sink can't reach a UI.
        self._mlflow_active: bool = False
        if mlflow_uri and _MLFLOW_AVAILABLE:
            try:
                _mlflow.set_tracking_uri(mlflow_uri)
                _mlflow.set_experiment(mlflow_experiment)
                _mlflow.start_run(run_name=mlflow_run_name)
                if self._config:
                    flat = _flatten_config(self._config)
                    for key, value in flat.items():
                        try:
                            _mlflow.log_param(key, value)
                        except Exception:  # noqa: BLE001 — MLflow has many error subclasses
                            pass
                self._mlflow_active = True
                logger.info(f"MLflow tracking → {mlflow_uri} / {mlflow_experiment}")
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"MLflow disabled: {type(exc).__name__}: {exc}")
                self._mlflow_active = False

    def rebuild_optimizer(self) -> None:
        """Rebuild the Adam optimizer to cover only currently-trainable parameters.

        Call this after ``network.freeze_trunk()`` so that LoRA adapters (and
        only those) appear in the parameter group. The current learning rate is
        carried over from the previous optimizer; scheduler resets to step 0.
        """
        lr = self._optimizer.param_groups[0]["lr"]
        self._optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self._network.parameters()),
            lr=lr,
            weight_decay=self._weight_decay,
        )
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self._optimizer,
            T_max=self._lr_decay_steps,
            eta_min=lr * 1e-2,
        )

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
            ``learning_rate``, ``grad_norm``, ``value_mean``, ``value_std``.
        """
        states, policy_targets, value_targets, legal_masks, mill_targets, pieces_targets = (
            buffer.sample(batch_size, device=self._device)
        )

        self._optimizer.zero_grad()
        categorical = self._value_head_type == "categorical"
        aux = self._aux_heads_enabled

        with torch.autocast(device_type=self._device.type, enabled=self._amp_enabled):
            if categorical and aux:
                log_policy, value, value_logits, mill_pred, pieces_pred = self._network(
                    states, legal_masks, return_value_logits=True, return_aux=True
                )
            elif categorical:
                log_policy, value, value_logits = self._network(
                    states, legal_masks, return_value_logits=True
                )
                mill_pred = pieces_pred = None
            elif aux:
                log_policy, value, mill_pred, pieces_pred = self._network(
                    states, legal_masks, return_aux=True
                )
                value_logits = None
            else:
                log_policy, value = self._network(states, legal_masks)
                value_logits = None
                mill_pred = pieces_pred = None

            total_loss, policy_loss, value_loss, mill_loss, pieces_loss = compute_loss(
                log_policy,
                value,
                policy_targets,
                value_targets,
                value_logits,
                mill_diff_pred=mill_pred,
                pieces_diff_pred=pieces_pred,
                mill_diff_target=mill_targets if aux else None,
                pieces_diff_target=pieces_targets if aux else None,
                aux_weight_mill=self._aux_weight_mill,
                aux_weight_pieces=self._aux_weight_pieces,
            )

        value_mean = float(value.detach().float().mean().item())
        value_std  = float(value.detach().float().std().item())

        self._scaler.scale(total_loss).backward()  # type: ignore[no-untyped-call]
        self._scaler.unscale_(self._optimizer)
        grad_norm = nn.utils.clip_grad_norm_(self._network.parameters(), self._max_grad_norm)
        self._scaler.step(self._optimizer)
        self._scaler.update()
        self._scheduler.step()
        self._step += 1

        metrics: dict[str, float] = {
            "total_loss": float(total_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "mill_loss": float(mill_loss.item()),
            "pieces_loss": float(pieces_loss.item()),
            "learning_rate": float(self._scheduler.get_last_lr()[0]),
            "grad_norm": float(grad_norm),
            "value_mean": value_mean,
            "value_std": value_std,
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
        # Per-game observability deques (rolling window). Each is the
        # corresponding GameRecord field across the last N games.
        recent_mills: deque[int] = deque(maxlen=_GAME_LENGTH_WINDOW)
        recent_captures: deque[int] = deque(maxlen=_GAME_LENGTH_WINDOW)
        recent_pieces_diff: deque[int] = deque(maxlen=_GAME_LENGTH_WINDOW)
        recent_term_reasons: deque[str] = deque(maxlen=_GAME_LENGTH_WINDOW)
        # Resign-feature deques: eligible / triggered are over the same window;
        # verify_outcomes accumulates only the (rare) verify games and tracks
        # whether the would-be-resigner actually lost (1) or not (0). The
        # cumulative count is a separate counter so the rate plot shows real
        # samples rather than a sliding window without enough data.
        recent_resign_eligible: deque[bool] = deque(maxlen=_GAME_LENGTH_WINDOW)
        recent_resigned: deque[bool] = deque(maxlen=_GAME_LENGTH_WINDOW)
        verify_outcomes: deque[int] = deque(maxlen=_GAME_LENGTH_WINDOW)
        verify_total = 0
        # Resign-specific game stats — populated only by games that actually
        # ended by resignation (regardless of curriculum start). Kept separate
        # from curriculum/normal so those two buckets contain only genuine
        # game-engine outcomes.
        recent_lengths_resign: deque[int] = deque(maxlen=_GAME_LENGTH_WINDOW)
        recent_captures_resign: deque[int] = deque(maxlen=_GAME_LENGTH_WINDOW)
        # Playout-cap deques: total full / fast plies across the rolling
        # window. Ratio = full / (full+fast), expected ≈ full_sim_fraction
        # when the feature is on, and 1.0 when it's off.
        recent_full_sim: deque[int] = deque(maxlen=_GAME_LENGTH_WINDOW)
        recent_fast_sim: deque[int] = deque(maxlen=_GAME_LENGTH_WINDOW)
        # Curriculum deque: per-game flag of whether the game started from
        # a random late-game position. Rolling rate confirms the feature is
        # firing at the configured random_start_fraction.
        recent_curriculum: deque[bool] = deque(maxlen=_GAME_LENGTH_WINDOW)
        # Three MUTUALLY EXCLUSIVE game populations (resign | curriculum | normal):
        #   resign     — ended by resignation, any start; outcome may be wrong
        #   curriculum — random start, played to natural engine termination
        #   normal     — initial_state start, played to natural engine termination
        # Mixing them would hide whether curriculum positions are genuinely more
        # decisive or whether resign is just creating artificial decisive results.
        recent_lengths_curriculum: deque[int] = deque(maxlen=_GAME_LENGTH_WINDOW)
        recent_captures_curriculum: deque[int] = deque(maxlen=_GAME_LENGTH_WINDOW)
        recent_term_reasons_curriculum: deque[str] = deque(maxlen=_GAME_LENGTH_WINDOW)
        outcome_counts_curriculum = {"win": 0, "draw": 0, "total": 0}
        recent_lengths_normal: deque[int] = deque(maxlen=_GAME_LENGTH_WINDOW)
        recent_captures_normal: deque[int] = deque(maxlen=_GAME_LENGTH_WINDOW)
        recent_term_reasons_normal: deque[str] = deque(maxlen=_GAME_LENGTH_WINDOW)
        outcome_counts_normal = {"win": 0, "draw": 0, "total": 0}
        outcome_counts = {"p1_win": 0, "p2_win": 0, "draw": 0}
        timeout_discarded_count = 0
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
                recent_mills.append(game.mills_p1 + game.mills_p2)
                recent_captures.append(game.captures_p1 + game.captures_p2)
                recent_pieces_diff.append(game.final_pieces_diff)
                recent_term_reasons.append(game.term_reason)
                recent_resign_eligible.append(game.resign_eligible)
                recent_resigned.append(game.resigned_by_player is not None)
                recent_full_sim.append(game.full_sim_moves)
                recent_fast_sim.append(game.fast_sim_moves)
                recent_curriculum.append(game.curriculum_start)
                captures = game.captures_p1 + game.captures_p2
                is_resigned = game.resigned_by_player is not None
                is_decisive = game.outcome != -1
                # Route into one of three mutually exclusive populations.
                if is_resigned:
                    recent_lengths_resign.append(game.game_length)
                    recent_captures_resign.append(captures)
                elif game.curriculum_start:
                    recent_lengths_curriculum.append(game.game_length)
                    recent_captures_curriculum.append(captures)
                    recent_term_reasons_curriculum.append(game.term_reason)
                    outcome_counts_curriculum["total"] += 1
                    if is_decisive:
                        outcome_counts_curriculum["win"] += 1
                    else:
                        outcome_counts_curriculum["draw"] += 1
                else:
                    recent_lengths_normal.append(game.game_length)
                    recent_captures_normal.append(captures)
                    recent_term_reasons_normal.append(game.term_reason)
                    outcome_counts_normal["total"] += 1
                    if is_decisive:
                        outcome_counts_normal["win"] += 1
                    else:
                        outcome_counts_normal["draw"] += 1
                if game.timeout_discarded:
                    timeout_discarded_count += 1
                self._log_scalar(
                    "game/timeout_discard_rate",
                    timeout_discarded_count / games_collected,
                )
                if game.was_verify_play and game.verify_resigning_player is not None:
                    # The "would-be-resigner" lost iff the actual outcome went
                    # to the opponent. 1 = resign decision was correct (their
                    # forfeit would have been right), 0 = false positive.
                    expected_loser = game.verify_resigning_player
                    actually_lost = (
                        game.outcome != -1 and game.outcome != expected_loser
                    )
                    verify_outcomes.append(1 if actually_lost else 0)
                    verify_total += 1
                if game.outcome == 1:
                    outcome_counts["p1_win"] += 1
                elif game.outcome == 2:
                    outcome_counts["p2_win"] += 1
                else:
                    outcome_counts["draw"] += 1

                self._log_scalar("train/buffer_size", len(buffer))
                self._log_scalar("train/games_collected", games_collected)
                self._log_game_stats(
                    recent_lengths,
                    recent_mills,
                    recent_captures,
                    recent_pieces_diff,
                    recent_term_reasons,
                    outcome_counts,
                    games_collected,
                )
                self._log_resign_stats(
                    recent_resign_eligible,
                    recent_resigned,
                    verify_outcomes,
                    verify_total,
                    games_collected,
                    recent_lengths_resign=recent_lengths_resign,
                    recent_captures_resign=recent_captures_resign,
                )
                self._log_playout_cap_stats(
                    recent_full_sim,
                    recent_fast_sim,
                    games_collected,
                )
                self._log_curriculum_stats(
                    recent_curriculum,
                    games_collected,
                    recent_lengths_curriculum=recent_lengths_curriculum,
                    recent_captures_curriculum=recent_captures_curriculum,
                    outcome_counts_curriculum=outcome_counts_curriculum,
                    recent_lengths_normal=recent_lengths_normal,
                    recent_captures_normal=recent_captures_normal,
                    outcome_counts_normal=outcome_counts_normal,
                    recent_term_reasons_curriculum=recent_term_reasons_curriculum,
                    recent_term_reasons_normal=recent_term_reasons_normal,
                )

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
        # strict=False so a checkpoint without aux heads can be loaded when
        # aux heads are now enabled — the new aux heads start randomly
        # initialised and learn from scratch over the next few thousand steps.
        missing, unexpected = self._network.load_state_dict(
            payload["state_dict"], strict=False
        )
        if missing:
            logger.info(f"Checkpoint missing {len(missing)} params (e.g. {missing[:2]}) — random init.")
        if unexpected:
            logger.warning(f"Checkpoint has {len(unexpected)} unexpected params (e.g. {unexpected[:2]}).")
        self._step = payload["step"]
        logger.info(f"Resumed from {path} at step {self._step}")
        if "optimizer" in payload:
            try:
                self._optimizer.load_state_dict(payload["optimizer"])
            except (ValueError, KeyError):
                # Optimizer state is incompatible with current parameter set
                # — most likely because LoRA adapters were added after the
                # checkpoint was created. Start with a fresh optimizer state.
                logger.warning(
                    "Optimizer state incompatible with current parameters "
                    "(LoRA adapters added?). Starting with fresh optimizer state."
                )
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
        """Flush TensorBoard writer and end any active MLflow run."""
        if self._writer is not None:
            self._writer.close()
        if self._mlflow_active:
            try:
                _mlflow.end_run()
            except Exception:  # noqa: BLE001
                pass
            self._mlflow_active = False

    def __enter__(self) -> Trainer:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        for key, value in metrics.items():
            tag = f"train/{key}"
            if self._writer is not None:
                self._writer.add_scalar(tag, value, self._step)
            self._mlflow_log(tag, float(value), self._step)

    def _log_scalar(self, tag: str, value: float | int) -> None:
        if self._writer is not None:
            self._writer.add_scalar(tag, value, self._step)
        self._mlflow_log(tag, float(value), self._step)

    def _mlflow_log(self, tag: str, value: float, step: int) -> None:
        if not self._mlflow_active:
            return
        try:
            _mlflow.log_metric(tag, value, step=step)
        except Exception:  # noqa: BLE001
            # A transient MLflow server hiccup must never bring down training.
            pass

    def _log_game_stats(
        self,
        recent_lengths: deque[int],
        recent_mills: deque[int],
        recent_captures: deque[int],
        recent_pieces_diff: deque[int],
        recent_term_reasons: deque[str],
        outcome_counts: dict[str, int],
        games_collected: int,
    ) -> None:
        """Log overall per-game scalar stats (all games) and a periodic length histogram."""
        if not recent_lengths:
            return
        last = recent_lengths[-1]
        mean_len = sum(recent_lengths) / len(recent_lengths)
        total = sum(outcome_counts.values()) or 1
        n_recent = len(recent_term_reasons) or 1
        # Decompose categorical term_reason into 5 separate fractions so each
        # one is a scalar metric MLflow/TB can plot directly. Values not seen
        # in the window default to 0.
        term_window = list(recent_term_reasons)
        stats: dict[str, float] = {
            "game/length_last": float(last),
            "game/length_mean_window": mean_len,
            "game/p1_win_rate": outcome_counts["p1_win"] / total,
            "game/p2_win_rate": outcome_counts["p2_win"] / total,
            "game/draw_rate": outcome_counts["draw"] / total,
            "game/mills_per_game_mean": (
                sum(recent_mills) / len(recent_mills) if recent_mills else 0.0
            ),
            "game/captures_per_game_mean": (
                sum(recent_captures) / len(recent_captures) if recent_captures else 0.0
            ),
            "game/final_pieces_diff_mean": (
                sum(recent_pieces_diff) / len(recent_pieces_diff)
                if recent_pieces_diff
                else 0.0
            ),
            "game/term_pieces_below_3_rate": (
                term_window.count("pieces_below_3") / n_recent
            ),
            "game/term_no_legal_moves_rate": (
                term_window.count("no_legal_moves") / n_recent
            ),
            "game/term_halfmove_cap_rate": (
                term_window.count("halfmove_cap") / n_recent
            ),
            "game/term_threefold_rate": term_window.count("threefold") / n_recent,
            "game/term_resign_rate": term_window.count("resign") / n_recent,
            "game/term_double_pass_rate": term_window.count("double_pass") / n_recent,
            "game/term_board_full_rate": term_window.count("board_full") / n_recent,
            "game/term_piece_count_tiebreak_rate": term_window.count("piece_count_tiebreak") / n_recent,
        }
        for tag, value in stats.items():
            if self._writer is not None:
                self._writer.add_scalar(tag, value, games_collected)
            self._mlflow_log(tag, value, games_collected)

        # Histograms are heavier; only emit one every full window refresh, and
        # only to TensorBoard (MLflow has no histogram metric type).
        if (
            self._writer is not None
            and games_collected % _GAME_LENGTH_WINDOW == 0
            and len(recent_lengths) == _GAME_LENGTH_WINDOW
        ):
            self._writer.add_histogram(
                "game/length_distribution",
                torch.tensor(list(recent_lengths), dtype=torch.float32),
                games_collected,
            )

    def _log_resign_stats(
        self,
        recent_resign_eligible: deque[bool],
        recent_resigned: deque[bool],
        verify_outcomes: deque[int],
        verify_total: int,
        games_collected: int,
        recent_lengths_resign: deque[int] | None = None,
        recent_captures_resign: deque[int] | None = None,
    ) -> None:
        """Log resign-feature diagnostics for post-hoc threshold calibration.

        - resign/eligible_rate: rolling fraction of games where the threshold
          was ever crossed (regardless of resign vs verify decision).
        - resign/triggered_rate: rolling fraction of games that actually
          ended by resignation. Always ≤ eligible_rate (verify_fraction
          subset is played out instead).
        - resign/verify_total: cumulative count of verify-play games (the
          ones we sampled to play out). Useful to know how much data we
          have for the false-positive estimate below.
        - resign/verified_correct_rate: among verify games, fraction where
          the would-be-resigner actually lost. Should sit ≥ ~0.95 with a
          well-calibrated threshold.
        - resign/verified_false_positive_rate: 1 − correct_rate. Above 5%
          means the threshold is too aggressive (resigning winning/draw
          positions); raise it (more negative — e.g. -0.95 from -0.90).
        """
        n_recent = len(recent_resign_eligible)
        if n_recent == 0:
            return
        eligible_rate = sum(recent_resign_eligible) / n_recent
        triggered_rate = sum(recent_resigned) / n_recent
        # Skip the false-positive plot until we have *some* verify data,
        # otherwise MLflow renders a misleading 0% curve from cold start.
        stats: dict[str, float] = {
            "resign/eligible_rate": eligible_rate,
            "resign/triggered_rate": triggered_rate,
            "resign/verify_total": float(verify_total),
        }
        if len(verify_outcomes) > 0:
            correct_rate = sum(verify_outcomes) / len(verify_outcomes)
            stats["resign/verified_correct_rate"] = correct_rate
            stats["resign/verified_false_positive_rate"] = 1.0 - correct_rate
        if recent_lengths_resign:
            stats["resign/length_mean"] = (
                sum(recent_lengths_resign) / len(recent_lengths_resign)
            )
        if recent_captures_resign:
            stats["resign/captures_per_game"] = (
                sum(recent_captures_resign) / len(recent_captures_resign)
            )
        for tag, value in stats.items():
            if self._writer is not None:
                self._writer.add_scalar(tag, value, games_collected)
            self._mlflow_log(tag, value, games_collected)

    def _log_playout_cap_stats(
        self,
        recent_full_sim: deque[int],
        recent_fast_sim: deque[int],
        games_collected: int,
    ) -> None:
        """Log full vs fast playout-cap ratios over the rolling window.

        - playout_cap/full_moves_per_game and fast_moves_per_game: averages
          per game; together they reconstruct the game length.
        - playout_cap/full_ratio: full / (full + fast). When the feature
          is on, expected ≈ full_sim_fraction. When off, exactly 1.0.
        """
        if not recent_full_sim:
            return
        n = len(recent_full_sim)
        full_sum = sum(recent_full_sim)
        fast_sum = sum(recent_fast_sim)
        total = full_sum + fast_sum
        if total == 0:
            return
        stats: dict[str, float] = {
            "playout_cap/full_moves_per_game": full_sum / n,
            "playout_cap/fast_moves_per_game": fast_sum / n,
            "playout_cap/full_ratio": full_sum / total,
        }
        for tag, value in stats.items():
            if self._writer is not None:
                self._writer.add_scalar(tag, value, games_collected)
            self._mlflow_log(tag, value, games_collected)

    def _log_curriculum_stats(
        self,
        recent_curriculum: deque[bool],
        games_collected: int,
        recent_lengths_curriculum: deque[int] | None = None,
        recent_captures_curriculum: deque[int] | None = None,
        outcome_counts_curriculum: dict[str, int] | None = None,
        recent_lengths_normal: deque[int] | None = None,
        recent_captures_normal: deque[int] | None = None,
        outcome_counts_normal: dict[str, int] | None = None,
        recent_term_reasons_curriculum: deque[str] | None = None,
        recent_term_reasons_normal: deque[str] | None = None,
    ) -> None:
        """Log curriculum vs normal population split stats (resign games excluded).

        Three mutually exclusive populations:
        - resign/   games that ended by resignation (logged in _log_resign_stats)
        - curriculum/ random-start games played to natural engine termination
        - normal/     initial-state games played to natural engine termination

        Metrics per population: length_mean, captures_per_game, draw_rate, win_rate.
        curriculum/start_rate is the overall fraction (includes resigned games).
        """
        if not recent_curriculum:
            return
        rate = sum(recent_curriculum) / len(recent_curriculum)
        stats: dict[str, float] = {"curriculum/start_rate": rate}

        if recent_lengths_curriculum:
            stats["curriculum/length_mean"] = (
                sum(recent_lengths_curriculum) / len(recent_lengths_curriculum)
            )
        if recent_captures_curriculum:
            stats["curriculum/captures_per_game"] = (
                sum(recent_captures_curriculum) / len(recent_captures_curriculum)
            )
        if outcome_counts_curriculum and outcome_counts_curriculum["total"] > 0:
            total = outcome_counts_curriculum["total"]
            stats["curriculum/draw_rate"] = outcome_counts_curriculum["draw"] / total
            stats["curriculum/win_rate"] = outcome_counts_curriculum["win"] / total

        if recent_lengths_normal:
            stats["normal/length_mean"] = (
                sum(recent_lengths_normal) / len(recent_lengths_normal)
            )
        if recent_captures_normal:
            stats["normal/captures_per_game"] = (
                sum(recent_captures_normal) / len(recent_captures_normal)
            )
        if outcome_counts_normal and outcome_counts_normal["total"] > 0:
            total = outcome_counts_normal["total"]
            stats["normal/draw_rate"] = outcome_counts_normal["draw"] / total
            stats["normal/win_rate"] = outcome_counts_normal["win"] / total

        if recent_term_reasons_curriculum:
            n = len(recent_term_reasons_curriculum)
            stats["curriculum/halfmove_cap_rate"] = (
                list(recent_term_reasons_curriculum).count("halfmove_cap") / n
            )
        if recent_term_reasons_normal:
            n = len(recent_term_reasons_normal)
            stats["normal/halfmove_cap_rate"] = (
                list(recent_term_reasons_normal).count("halfmove_cap") / n
            )

        for tag, value in stats.items():
            if self._writer is not None:
                self._writer.add_scalar(tag, value, games_collected)
            self._mlflow_log(tag, value, games_collected)

    def _log_memory_health(self, manager: SelfPlayManager) -> None:
        """Periodic check for memory/queue leaks. Cheap: ~1 ms per call.

        TensorBoard scalars are written every call (high-resolution trace).
        A stderr line is only emitted every ``_MEMORY_LOG_STDERR_INTERVAL``
        steps to avoid disturbing the trainer's tqdm bar.
        """
        results_q = manager.results_qsize()
        weights_q_max = manager.weights_qsize_max()
        rss_gb = -1.0
        if _PSUTIL_AVAILABLE:
            rss_gb = psutil.Process().memory_info().rss / 1e9
            self._log_scalar("system/rss_gb", rss_gb)
        self._log_scalar("system/results_qsize", results_q)
        self._log_scalar("system/weights_qsize_max", weights_q_max)
        if self._step % _MEMORY_LOG_STDERR_INTERVAL == 0:
            logger.info(
                f"step={self._step} rss={rss_gb:.2f}GB "
                f"results_q={results_q} weights_q_max={weights_q_max}"
            )

    def _auto_checkpoint(self) -> None:
        if self._checkpoint_dir is None:
            return
        path = self._checkpoint_dir / f"checkpoint_{self._step:08d}.pt"
        self.save(path)
        if self._mlflow_active:
            try:
                _mlflow.log_artifact(str(path), artifact_path="checkpoints")
            except Exception:  # noqa: BLE001
                pass
