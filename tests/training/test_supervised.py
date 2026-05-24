"""Tests for SupervisedTrainer + train_supervised pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from morris_rl.data.dataset import WarmupDataset, augment_batch
from morris_rl.network.factory import build_network
from morris_rl.training.supervised import (
    SupervisedTrainer,
    TrainArgs,
    train_supervised,
)
from morris_rl.training.trainer import compute_loss
from morris_rl.utils.checkpoints import load_checkpoint
from torch.utils.data import DataLoader

# Reuse test helpers from test_dataset module.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from test_dataset import _make_dataset  # type: ignore[import-not-found]


def _build_tiny_network():
    cfg = OmegaConf.create({
        "network": {
            "type": "graphnet", "num_blocks": 2, "num_channels": 32,
            "policy_head_hidden": 32, "value_head_hidden": 32,
            "value_head_type": "scalar",
        },
        "aux_heads": {"enabled": True, "hidden_size": 32},
    })
    return build_network(cfg)


def test_compute_loss_policy_mask_zeros_masked_samples():
    """Verify compute_loss respects policy_mask: zero-masked samples contribute
    zero to the policy loss."""
    torch.manual_seed(0)
    B, A = 4, 80
    log_p = torch.log_softmax(torch.randn(B, A), dim=1)
    policy_target = torch.zeros(B, A)
    # All samples have a uniform-over-3-actions target; only the second two
    # are "real" (mask=True). The first two should be ignored.
    for i in range(B):
        policy_target[i, :3] = 1 / 3
    value = torch.zeros(B)
    value_target = torch.zeros(B)
    mask = torch.tensor([0.0, 0.0, 1.0, 1.0])
    total_masked, p_loss_masked, _, _, _ = compute_loss(
        log_p, value, policy_target, value_target, policy_mask=mask,
    )
    # Compute by hand using only the unmasked samples.
    _total_ref, p_loss_ref, _, _, _ = compute_loss(
        log_p[2:], value[2:], policy_target[2:], value_target[2:],
    )
    assert abs(float(p_loss_masked) - float(p_loss_ref)) < 1e-5


def test_supervised_step_no_nan(tmp_path: Path):
    dataset = _make_dataset(tmp_path, num_games=2)
    network = _build_tiny_network()
    device = torch.device("cpu")
    trainer = SupervisedTrainer(
        network, device, lr=1e-3,
        aux_weight_mill=0.1, aux_weight_pieces=0.1,
        aux_heads_enabled=True,
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=True,
                        collate_fn=augment_batch(0), drop_last=False)
    for batch in loader:
        comps = trainer.step(batch)
        for k, v in comps.items():
            assert v == v, f"{k} = NaN"
        break  # one step is enough to verify forward+backward is clean


def test_supervised_step_decreases_loss(tmp_path: Path):
    """Train for a few steps on a tiny dataset; loss should drop."""
    dataset = _make_dataset(tmp_path, num_games=4)
    network = _build_tiny_network()
    device = torch.device("cpu")
    trainer = SupervisedTrainer(network, device, lr=3e-3, aux_heads_enabled=True)
    loader = DataLoader(dataset, batch_size=16, shuffle=True,
                        collate_fn=augment_batch(0), drop_last=False)
    losses = []
    for _ in range(8):  # 8 epochs ≈ enough to see clear drop on tiny data
        for batch in loader:
            comps = trainer.step(batch)
            losses.append(comps["total"])
    assert losses[-1] < losses[0], (
        f"loss didn't decrease: start={losses[0]:.3f} end={losses[-1]:.3f}"
    )


def test_train_supervised_end_to_end(tmp_path: Path):
    """Smoke: 1 epoch on a 3-game dataset, checkpoint saved and reloadable."""
    out_dir = tmp_path / "supervised_run"
    # _make_dataset creates tmp_path/warmup itself, no need to pre-mkdir.
    _ = _make_dataset(tmp_path, num_games=3)
    warmup_dir = tmp_path / "warmup"
    args = TrainArgs(
        warmup_dir=warmup_dir,
        out_dir=out_dir,
        network_type="graphnet",
        num_blocks=2, num_channels=32,
        policy_head_hidden=32, value_head_hidden=32,
        aux_head_hidden=32,
        batch_size=8,
        epochs=1,
        eval_every=0,           # skip baseline eval for smoke
        early_stop_patience=999,
        device="cpu",
    )
    summary = train_supervised(args)
    assert summary["total_epochs_run"] == 1
    best = out_dir / "best.pt"
    final = out_dir / "final.pt"
    assert best.exists()
    assert final.exists()
    payload = load_checkpoint(best)
    assert "state_dict" in payload
    assert "config" in payload
    assert payload["config"]["network"]["type"] == "graphnet"
