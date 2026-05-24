"""Tests for WarmupDataset and its augmentation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from morris_rl.data.dataset import (
    WarmupDataset,
    _color_swap_planes,
    augment_batch,
    plain_collate,
    split_warmup_dataset,
)
from morris_rl.env.board import ACTION_SPACE_SIZE
from morris_rl.env.rules import (
    Outcome,
    apply_action,
    get_legal_actions,
    initial_state,
    is_terminal,
)


def _make_game_jsonl(actions: list[int], outcome: int, opening_random_k: int = 0,
                     epsilon_random_indices: list[int] | None = None,
                     worker: int = 0) -> str:
    """Build a single JSONL line representing a played game.

    `actions` is the list of moves taken. We synthesise root_scores: for each
    legal action at that position, we put score 1.0 on the action played and
    0.0 on the rest (so softmax → one-hot on the played action). Random plies
    get None scores.
    """
    eps = set(epsilon_random_indices or [])
    state = initial_state()
    root_scores: list = []
    for t, a in enumerate(actions):
        if t < opening_random_k or t in eps:
            root_scores.append(None)
        else:
            legal = get_legal_actions(state)
            scores = [{"a": int(la), "s": (5.0 if la == a else 0.0)} for la in legal]
            root_scores.append(scores)
        state = apply_action(state, int(a))
    payload = {
        "ts": 0.0,
        "worker": worker,
        "game": "morris",
        "outcome": outcome,
        "length": len(actions),
        "term_reason": "pieces_below_3",
        "actions": actions,
        "root_scores": root_scores,
        "opening_random_k": opening_random_k,
        "epsilon_random_indices": sorted(eps),
        "depth": 3,
        "epsilon": 0.0,
        "wall_seconds": 0.1,
    }
    return json.dumps(payload)


def _play_random_game(seed: int, max_halfmoves: int = 30):
    """Play a short random game and return (actions, outcome)."""
    import random
    rng = random.Random(seed)
    state = initial_state()
    actions: list[int] = []
    while True:
        if state.total_halfmoves >= max_halfmoves:
            return actions, 0  # cap → draw
        done, outcome = is_terminal(state)
        if done:
            out = 0 if (outcome is None or outcome == Outcome.DRAW) else int(outcome)
            return actions, out
        legal = get_legal_actions(state)
        a = rng.choice(legal)
        actions.append(a)
        state = apply_action(state, a)


def _make_dataset(tmp_path: Path, num_games: int = 3, **kwargs) -> WarmupDataset:
    out_dir = tmp_path / "warmup"
    out_dir.mkdir()
    fp = out_dir / "worker_0.jsonl"
    with fp.open("w") as fh:
        for i in range(num_games):
            actions, outcome = _play_random_game(seed=i)
            fh.write(_make_game_jsonl(actions, outcome, **kwargs) + "\n")
    return WarmupDataset(out_dir)


def test_dataset_loads_multiple_games(tmp_path: Path):
    dataset = _make_dataset(tmp_path, num_games=3)
    summary = dataset.summary()
    assert summary["num_games"] == 3
    assert summary["num_samples"] == sum(
        # Each sample = one half-move. The number of samples equals total actions
        # across all games (final terminal state is NOT a sample).
        len(dataset[i][0]) for i in [0]  # trivial, just smoke that getitem works
    ) or summary["num_samples"] > 0  # main assertion below
    assert summary["num_samples"] > 0


def test_has_policy_false_during_opening_random(tmp_path: Path):
    # opening_random_k=3 → first 3 samples per game have has_policy=False
    dataset = _make_dataset(tmp_path, num_games=1, opening_random_k=3)
    # find samples for game_id=0
    for i in range(3):
        _, _, _, _, _, _, has_policy, gid = dataset[i]
        assert gid == 0
        assert has_policy is False, f"sample {i} should be random (opening)"
    # 4th sample (post-opening) should have a policy target
    _, policy, _, _, _, _, has_policy, gid = dataset[3]
    if gid == 0:  # the game has length >= 4
        assert has_policy is True


def test_value_target_signed_pov(tmp_path: Path):
    """In a P1-wins game (outcome=1), position at t=0 has current_player=P1
    and value_target > 0. After P1's first move, current_player=P2 (assuming no
    mill formed) → value_target < 0."""
    out_dir = tmp_path / "warmup"
    out_dir.mkdir()
    # A short synthetic game: P1 places 0, P2 places 8, P1 places 1, P2 places 9, ...
    # Set outcome=1 (P1 wins) artificially.
    actions = [0, 8, 1, 9, 2]
    fp = out_dir / "worker_0.jsonl"
    with fp.open("w") as fh:
        fh.write(_make_game_jsonl(actions, outcome=1) + "\n")
    dataset = WarmupDataset(out_dir, gamma=1.0)
    # Sample 0: P1 to play, P1 wins → value > 0
    _, _, v0, _, _, _, _, _ = dataset[0]
    # Sample 1: P2 to play (P1 just placed, no mill formed), P1 wins → value < 0
    _, _, v1, _, _, _, _, _ = dataset[1]
    assert v0 > 0, f"P1 POV in P1-win game should be positive, got {v0}"
    assert v1 < 0, f"P2 POV in P1-win game should be negative, got {v1}"


def test_value_gamma_decay(tmp_path: Path):
    out_dir = tmp_path / "warmup"
    out_dir.mkdir()
    actions = [0, 8, 1, 9, 2, 10]
    fp = out_dir / "worker_0.jsonl"
    with fp.open("w") as fh:
        fh.write(_make_game_jsonl(actions, outcome=1) + "\n")
    dataset = WarmupDataset(out_dir, gamma=0.5)
    T = len(actions)
    # P1-POV positions are at even indices, T-t shrinks → magnitude grows
    _, _, v0, _, _, _, _, _ = dataset[0]
    _, _, v_last_p1, _, _, _, _, _ = dataset[T - 2]  # last P1 move
    # |v_last_p1| > |v0| since closer to end (less decay)
    assert abs(v_last_p1) > abs(v0)
    # Both should be ~ 0.5^k × 1
    assert abs(abs(v0) - 0.5 ** T) < 1e-5
    # P1 plays at even indices 0, 2, 4, ... So T-2 has player = ?
    # T=6, indices 0..5: even=P1 (assuming no mills). Sample at t=4 has T-t=2 → 0.25
    # Sample at t=T-2=4: P1 POV (even) → +0.25
    _, _, v4, _, _, _, _, _ = dataset[4]
    assert abs(v4 - 0.25) < 1e-5


def test_policy_target_sums_to_one(tmp_path: Path):
    dataset = _make_dataset(tmp_path, num_games=1, opening_random_k=0)
    for i in range(len(dataset)):
        _, policy, _, _, _, mask, hp, _ = dataset[i]
        if hp:
            total = policy.sum()
            assert abs(total - 1.0) < 1e-5, f"policy sums to {total} at i={i}"
            # Mass only on legal actions
            illegal_mass = policy[~mask].sum()
            assert illegal_mass < 1e-6


def test_color_swap_planes_swaps_own_opp():
    encoded = np.arange(11 * 24, dtype=np.float32).reshape(11, 24)
    swapped = _color_swap_planes(encoded)
    assert np.array_equal(swapped[0], encoded[1])
    assert np.array_equal(swapped[1], encoded[0])
    assert np.array_equal(swapped[2], encoded[3])
    assert np.array_equal(swapped[3], encoded[2])
    assert np.array_equal(swapped[7], encoded[8])
    assert np.array_equal(swapped[8], encoded[7])
    # Constants planes 9, 10 unchanged
    assert np.array_equal(swapped[9], encoded[9])
    assert np.array_equal(swapped[10], encoded[10])


def test_split_no_game_overlap(tmp_path: Path):
    dataset = _make_dataset(tmp_path, num_games=10)
    train, val = split_warmup_dataset(dataset, val_ratio=0.3, seed=42)
    train_gids = {dataset.game_id_of(i) for i in train.indices}
    val_gids = {dataset.game_id_of(i) for i in val.indices}
    assert train_gids.isdisjoint(val_gids)
    assert train_gids | val_gids == set(range(10))


def test_augment_batch_preserves_shapes(tmp_path: Path):
    dataset = _make_dataset(tmp_path, num_games=2)
    collate = augment_batch(batch_rng_seed=0)
    batch = [dataset[i] for i in range(min(4, len(dataset)))]
    x, p, v, m, pc, mask, hp = collate(batch)
    assert x.shape[0] == len(batch)
    assert x.shape[1] == 11  # graph encoding planes
    assert x.shape[2] == 24
    assert p.shape == (len(batch), ACTION_SPACE_SIZE)
    assert mask.dtype == torch.bool


def test_plain_collate_no_augmentation(tmp_path: Path):
    dataset = _make_dataset(tmp_path, num_games=1)
    batch = [dataset[i] for i in range(min(2, len(dataset)))]
    x, _p, _v, _m, _pc, _mask, _hp = plain_collate(batch)
    # Same tensors as the raw samples (no D4 perm applied)
    for j, sample in enumerate(batch):
        raw_state = sample[0]
        assert np.allclose(x[j].numpy(), raw_state)
