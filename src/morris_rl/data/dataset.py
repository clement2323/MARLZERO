"""PyTorch Dataset for supervised warmup training on minimax JSONL traces.

Reads all `worker_*.jsonl` files produced by `scripts/generate_warmup_dataset.py`,
replays each game from `initial_state` to materialise every visited position,
and exposes pre-encoded training samples:

    (encoded_state, policy_target, value_target, mill_target, pieces_target,
     legal_mask, has_policy, game_id)

Augmentation (D4 × color-swap, 16×) is applied **on-the-fly** by the
``augment_batch`` collate function — never persisted, so the dataset RAM cost
stays at one copy of the raw positions (~400 MB for 10k games × 80 plies).
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from morris_rl.env.board import ACTION_SPACE_SIZE, NUM_POSITIONS
from morris_rl.env.encoding_graph import encode_state_graph
from morris_rl.env.rules import (
    GameState,
    apply_action,
    compute_aux_features,
    get_legal_actions,
    initial_state,
)
from morris_rl.env.symmetries import (
    SYMMETRY_PERMUTATIONS,
    transform_encoded_state as _transform_encoded_state,
    transform_policy as _transform_policy,
)

EncodeFn = Callable[[GameState], torch.Tensor]


# ---------------------------------------------------------------------------
# Sample
# ---------------------------------------------------------------------------


@dataclass
class _WarmupSample:
    """Internal per-position record. Stored as numpy/scalar to keep RAM low."""

    encoded_state: np.ndarray   # (num_planes, NUM_POSITIONS) float32
    policy_target: np.ndarray   # (ACTION_SPACE_SIZE,) float32
    value_target: float         # ∈ [-1, +1]
    mill_target: float          # signed (own - opp)
    pieces_target: float        # signed (own - opp)
    legal_mask: np.ndarray      # (ACTION_SPACE_SIZE,) bool
    has_policy: bool            # False for random plies (opening + ε-greedy)
    game_id: int                # for train/val split-by-game


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class WarmupDataset(Dataset):
    """Replays every game in `warmup_dir/worker_*.jsonl` and exposes per-position
    samples for supervised training.

    Args:
        warmup_dir:        Directory containing worker_*.jsonl files.
        encode_fn:         State encoder. Default: encode_state_graph (11 planes).
        gamma:             Discount applied to the value target. value_target at
                           position t in a game of length T is γ^(T-t) × outcome,
                           where outcome ∈ {-1, 0, +1} from the **current player's**
                           POV at t. Default 1.0 (no discount).
        policy_temperature: Softmax temperature applied to root_scores at policy
                            target construction. Higher → flatter target. Default 1.0.
        max_games:         Optional cap on games loaded (debug / smoke tests).
    """

    def __init__(
        self,
        warmup_dir: Path,
        encode_fn: EncodeFn = encode_state_graph,
        gamma: float = 1.0,
        policy_temperature: float = 1.0,
        max_games: int | None = None,
    ) -> None:
        warmup_dir = Path(warmup_dir)
        if not warmup_dir.exists():
            raise FileNotFoundError(f"warmup_dir not found: {warmup_dir}")
        self._encode_fn = encode_fn
        self._gamma = float(gamma)
        self._policy_temperature = float(policy_temperature)

        self._samples: list[_WarmupSample] = []
        self._game_lengths: list[int] = []   # parallel to game_id
        self._game_outcomes: list[int] = []  # for diagnostics

        next_game_id = 0
        for jsonl_path in sorted(warmup_dir.glob("worker_*.jsonl")):
            with jsonl_path.open() as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    if max_games is not None and next_game_id >= max_games:
                        break
                    game = json.loads(line)
                    self._materialise_game(game, game_id=next_game_id)
                    self._game_lengths.append(int(game["length"]))
                    self._game_outcomes.append(int(game["outcome"]))
                    next_game_id += 1
                if max_games is not None and next_game_id >= max_games:
                    break

        if not self._samples:
            raise ValueError(f"no samples materialised from {warmup_dir}")

        self._num_games = next_game_id

    # ----- core: replay a single game and emit per-position samples ---------

    def _materialise_game(self, game: dict, game_id: int) -> None:
        actions: list[int] = game["actions"]
        root_scores: list = game.get("root_scores", [None] * len(actions))
        outcome_p1_pov: int = int(game["outcome"])  # 0=draw, 1=P1, 2=P2
        T = len(actions)

        state = initial_state()
        for t, action in enumerate(actions):
            encoded = self._encode_fn(state).squeeze(0).numpy().astype(np.float32)
            legal = get_legal_actions(state)
            legal_mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
            legal_mask[legal] = True

            # --- policy target -----------------------------------------
            scores_entry = root_scores[t]
            policy_target = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
            has_policy = scores_entry is not None
            if has_policy:
                # scores_entry: list of {"a": int, "s": float}
                action_ids = np.array([int(e["a"]) for e in scores_entry], dtype=np.int64)
                scores = np.array([float(e["s"]) for e in scores_entry], dtype=np.float64)
                # Numerically-stable softmax(scores / τ).
                scaled = scores / self._policy_temperature
                scaled -= scaled.max()
                exp = np.exp(scaled)
                probs = (exp / exp.sum()).astype(np.float32)
                policy_target[action_ids] = probs

            # --- value target ------------------------------------------
            outcome_for_current = _outcome_from_pov(outcome_p1_pov, state.current_player)
            decay = self._gamma ** (T - t) if self._gamma < 1.0 else 1.0
            value_target = float(decay * outcome_for_current)

            # --- aux targets -------------------------------------------
            mill_diff, pieces_diff = compute_aux_features(state)

            self._samples.append(
                _WarmupSample(
                    encoded_state=encoded,
                    policy_target=policy_target,
                    value_target=value_target,
                    mill_target=float(mill_diff),
                    pieces_target=float(pieces_diff),
                    legal_mask=legal_mask,
                    has_policy=has_policy,
                    game_id=game_id,
                )
            )
            state = apply_action(state, int(action))

    # ----- Dataset protocol -------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        s = self._samples[idx]
        return (
            s.encoded_state,
            s.policy_target,
            s.value_target,
            s.mill_target,
            s.pieces_target,
            s.legal_mask,
            s.has_policy,
            s.game_id,
        )

    # ----- introspection ----------------------------------------------------

    @property
    def num_games(self) -> int:
        return self._num_games

    def game_id_of(self, idx: int) -> int:
        return self._samples[idx].game_id

    def summary(self) -> dict:
        n_policy = sum(1 for s in self._samples if s.has_policy)
        return {
            "num_samples": len(self._samples),
            "num_games": self._num_games,
            "samples_with_policy_target": n_policy,
            "samples_without_policy_target": len(self._samples) - n_policy,
            "outcome_counts": {
                "draw": sum(1 for o in self._game_outcomes if o == 0),
                "p1": sum(1 for o in self._game_outcomes if o == 1),
                "p2": sum(1 for o in self._game_outcomes if o == 2),
            },
        }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _outcome_from_pov(outcome_p1_pov: int, current_player: int) -> float:
    """Convert P1-POV outcome (0 draw, 1 P1 wins, 2 P2 wins) → current-player POV.

    Returns +1.0 if current player won, -1.0 if lost, 0.0 if draw.
    """
    if outcome_p1_pov == 0:
        return 0.0
    winner = outcome_p1_pov  # 1 or 2
    return 1.0 if winner == current_player else -1.0


def split_warmup_dataset(
    dataset: WarmupDataset,
    val_ratio: float = 0.1,
    seed: int = 0,
) -> tuple[Subset, Subset]:
    """Random 90/10 (or other ratio) split by GAME, not by position.

    Positions within the same game stay together. Avoids leakage where a position
    from game G would land in train and a near-identical position from the same
    game in val, inflating the val metric.
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
    n_games = dataset.num_games
    n_val = max(1, int(math.floor(val_ratio * n_games)))

    rng = random.Random(seed)
    all_game_ids = list(range(n_games))
    rng.shuffle(all_game_ids)
    val_game_ids = set(all_game_ids[:n_val])

    train_indices: list[int] = []
    val_indices: list[int] = []
    for i in range(len(dataset)):
        if dataset.game_id_of(i) in val_game_ids:
            val_indices.append(i)
        else:
            train_indices.append(i)
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


# ---------------------------------------------------------------------------
# Augmentation (collate_fn)
# ---------------------------------------------------------------------------


def _color_swap_planes(encoded: np.ndarray) -> np.ndarray:
    """Swap own↔opp planes (0↔1 and 2↔3) for the GraphNet 11-plane encoding.

    Planes 4-6 (phase one-hot, must_capture) describe the current player's
    sub-turn flag — these stay attached to the (now-flipped) current player so
    they remain unchanged. Planes 7-8 are own_threats / opp_threats so they
    also swap. Planes 9-10 (degree, ring) are board-topology constants.
    """
    out = encoded.copy()
    # Own/opp pieces
    out[0] = encoded[1]
    out[1] = encoded[0]
    # Own/opp hand
    out[2] = encoded[3]
    out[3] = encoded[2]
    # Own/opp threats (planes 7-8 only exist in the 11-plane graph encoding;
    # guard so the function still works on legacy 7-plane inputs).
    if encoded.shape[0] >= 9:
        out[7] = encoded[8]
        out[8] = encoded[7]
    return out


def augment_batch(batch_rng_seed: int | None = None) -> Callable:
    """Return a collate_fn that applies a random D4 × color-swap to each sample.

    Used as ``DataLoader(collate_fn=augment_batch(seed))``. Each call to the
    returned collate is given a list of per-sample tuples; we apply one of the
    16 group elements (uniform) per sample, then stack into tensors.
    """
    base_rng = random.Random(batch_rng_seed)

    def collate(batch):
        nonlocal base_rng
        states = []
        policies = []
        values = []
        mills = []
        pieces = []
        legal_masks = []
        has_policy = []

        for sample in batch:
            (
                encoded,
                policy_target,
                value_target,
                mill_target,
                pieces_target,
                legal_mask,
                hp,
                _gid,
            ) = sample

            sym_idx = base_rng.randint(0, 7)
            color_swap = base_rng.random() < 0.5

            perm = SYMMETRY_PERMUTATIONS[sym_idx]
            encoded_t = _transform_encoded_state(encoded, perm)
            policy_t = _transform_policy(policy_target, perm)
            legal_t = _transform_policy(legal_mask.astype(np.float32), perm) > 0.5

            v = float(value_target)
            m = float(mill_target)
            p = float(pieces_target)
            if color_swap:
                encoded_t = _color_swap_planes(encoded_t)
                v = -v
                m = -m
                p = -p
                # policy_target unchanged: the action labels refer to physical
                # board positions, not to who-is-current.

            states.append(encoded_t)
            policies.append(policy_t)
            values.append(v)
            mills.append(m)
            pieces.append(p)
            legal_masks.append(legal_t)
            has_policy.append(bool(hp))

        return (
            torch.from_numpy(np.stack(states)).float(),
            torch.from_numpy(np.stack(policies)).float(),
            torch.tensor(values, dtype=torch.float32),
            torch.tensor(mills, dtype=torch.float32),
            torch.tensor(pieces, dtype=torch.float32),
            torch.from_numpy(np.stack(legal_masks)),
            torch.tensor(has_policy, dtype=torch.float32),
        )

    return collate


def plain_collate(batch):
    """No-augmentation collate, used for validation (deterministic metrics)."""
    states = np.stack([b[0] for b in batch])
    policies = np.stack([b[1] for b in batch])
    legal_masks = np.stack([b[5] for b in batch])
    return (
        torch.from_numpy(states).float(),
        torch.from_numpy(policies).float(),
        torch.tensor([b[2] for b in batch], dtype=torch.float32),
        torch.tensor([b[3] for b in batch], dtype=torch.float32),
        torch.tensor([b[4] for b in batch], dtype=torch.float32),
        torch.from_numpy(legal_masks),
        torch.tensor([float(b[6]) for b in batch], dtype=torch.float32),
    )
