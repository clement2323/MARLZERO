"""Load minimax warmup JSONL traces into a ReplayBuffer.

Used by Phase 3 self-play to seed a non-purged sub-buffer with the same
positions that the supervised pre-training fitted on. By keeping 30 % of
each training minibatch drawn from this fixed buffer, the network is
continuously pulled back toward the minimax-derived prior — preventing
catastrophic forgetting once self-play data starts dominating the FIFO
main buffer.

Unlike `data/dataset.py:WarmupDataset` (which emits torch.Dataset tuples
for the supervised loop), this helper emits `SampleRecord` objects in the
exact format expected by `ReplayBuffer.add_samples`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np

from morris_rl.env.board import ACTION_SPACE_SIZE
from morris_rl.env.rules import (
    GameState,
    apply_action,
    compute_aux_features,
    get_legal_actions,
    initial_state,
)
from morris_rl.training.replay_buffer import ReplayBuffer, SampleRecord

EncodeFn = Callable[[GameState], "torch.Tensor"]  # type: ignore[name-defined]


def _outcome_from_pov(outcome_p1_pov: int, current_player: int) -> float:
    """P1-POV outcome (0=draw, 1=P1, 2=P2) → current-player POV ∈ {-1, 0, +1}."""
    if outcome_p1_pov == 0:
        return 0.0
    return 1.0 if outcome_p1_pov == current_player else -1.0


def load_warmup_into_buffer(
    buffer: ReplayBuffer,
    warmup_dir: Path,
    encode_fn: EncodeFn,
    gamma: float = 1.0,
    policy_temperature: float = 1.0,
    only_with_policy: bool = True,
    max_games: int | None = None,
) -> int:
    """Replay every game in `warmup_dir/worker_*.jsonl` and add per-position
    SampleRecords to `buffer`.

    Args:
        buffer: target ReplayBuffer. Its symmetry augmentation flag (if set)
            applies — each materialised position will be added together with
            its D4 symmetric variants.
        warmup_dir: directory containing worker_*.jsonl files.
        encode_fn: state encoder. For Phase 3 GraphNet use encode_state_graph.
        gamma: discount γ applied to value target γ^(T-t) * outcome.
        policy_temperature: softmax temperature on root_scores.
        only_with_policy: skip positions where root_scores is None (opening
            and ε-random plies). Recommended True for the warmup sub-buffer
            because random-ply positions carry no informative policy target
            and would only contribute to the value head — value loss already
            gets plenty of signal from minimax plies.
        max_games: optional cap on games loaded (debug / smoke).

    Returns:
        Number of SampleRecords added BEFORE augmentation (after augmentation
        the actual buffer fill is 8x this when use_symmetry_augmentation is on).
    """
    warmup_dir = Path(warmup_dir)
    if not warmup_dir.exists():
        raise FileNotFoundError(f"warmup_dir not found: {warmup_dir}")

    n_added = 0
    n_games = 0
    samples_batch: list[SampleRecord] = []
    BATCH_FLUSH = 1024  # add to buffer in chunks to keep RAM steady

    for jsonl_path in sorted(warmup_dir.glob("worker_*.jsonl")):
        with jsonl_path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if max_games is not None and n_games >= max_games:
                    break
                game = json.loads(line)
                n_games += 1

                actions: list[int] = game["actions"]
                root_scores = game.get("root_scores", [None] * len(actions))
                outcome_p1_pov = int(game["outcome"])
                T = len(actions)

                state = initial_state()
                for t, action in enumerate(actions):
                    scores_entry = root_scores[t]
                    has_policy = scores_entry is not None
                    if only_with_policy and not has_policy:
                        state = apply_action(state, int(action))
                        continue

                    encoded = encode_fn(state).squeeze(0).numpy().astype(np.float32)
                    legal = get_legal_actions(state)
                    legal_mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
                    legal_mask[legal] = True

                    policy_target = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
                    if has_policy:
                        action_ids = np.array([int(e["a"]) for e in scores_entry], dtype=np.int64)
                        scores = np.array([float(e["s"]) for e in scores_entry], dtype=np.float64)
                        scaled = scores / policy_temperature
                        scaled -= scaled.max()
                        exp = np.exp(scaled)
                        probs = (exp / exp.sum()).astype(np.float32)
                        policy_target[action_ids] = probs
                    else:
                        # No policy info — fall back to uniform over legal moves.
                        # The trainer will see this as a "weak" sample but
                        # value/aux losses still apply.
                        if legal:
                            policy_target[legal] = 1.0 / len(legal)

                    outcome_for_current = _outcome_from_pov(outcome_p1_pov, state.current_player)
                    decay = gamma ** (T - t) if gamma < 1.0 else 1.0
                    value_target = float(decay * outcome_for_current)

                    mill_diff, pieces_diff = compute_aux_features(state)

                    samples_batch.append(SampleRecord(
                        encoded_state=encoded,
                        policy_target=policy_target,
                        value_target=value_target,
                        legal_mask=legal_mask,
                        mill_diff_target=float(mill_diff),
                        pieces_diff_target=float(pieces_diff),
                    ))
                    n_added += 1
                    state = apply_action(state, int(action))

                    if len(samples_batch) >= BATCH_FLUSH:
                        buffer.add_samples(samples_batch)
                        samples_batch = []

            if max_games is not None and n_games >= max_games:
                break

    # Flush remaining.
    if samples_batch:
        buffer.add_samples(samples_batch)

    return n_added
