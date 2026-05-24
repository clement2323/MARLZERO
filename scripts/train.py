"""Training entry point for the Nine Men's Morris AlphaZero agent.

Usage
-----
    # Default config
    python scripts/train.py

    # Hydra overrides (any key from configs/default.yaml)
    python scripts/train.py self_play.num_workers=4 mcts.num_simulations_train=100

    # Resume from a checkpoint
    python scripts/train.py training.resume=checkpoints/checkpoint_00010000.pt

    # Quick smoke-test (few steps, small buffer)
    python scripts/train.py training.total_steps=200 training.min_buffer_size=50 \
        self_play.num_workers=2 mcts.num_simulations_train=20
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

# Allow running as `python scripts/train.py` from the repo root.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morris_rl.network.factory import build_network
from morris_rl.training.replay_buffer import ReplayBuffer
from morris_rl.training.self_play import SelfPlayManager
from morris_rl.training.trainer import Trainer
from morris_rl.utils.logging import logger, setup_logging
from morris_rl.utils.seeding import seed_everything


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _network_cfg_dict(cfg: DictConfig) -> dict:
    """Plain dict passed to worker processes (must be picklable)."""
    game_name = cfg.get("game", "morris")
    if game_name == "reversi":
        from morris_rl.env.reversi.board import ACTION_SPACE_SIZE as _acs, NUM_POSITIONS as _np
        num_positions, action_space_size = _np, _acs
    else:
        from morris_rl.env.board import ACTION_SPACE_SIZE as _acs, NUM_POSITIONS as _np
        num_positions, action_space_size = _np, _acs
    aux_cfg = cfg.get("aux_heads", {}) or {}
    return {
        "type": cfg.network.get("type", "resnet"),
        "num_blocks": cfg.network.num_blocks,
        "num_channels": cfg.network.num_channels,
        "num_planes": cfg.input_encoding.num_planes,
        "policy_head_hidden": cfg.network.policy_head_hidden,
        "value_head_hidden": cfg.network.value_head_hidden,
        "value_head_type": cfg.network.get("value_head_type", "scalar"),
        "num_positions": num_positions,
        "action_space_size": action_space_size,
        "aux_heads_enabled": bool(aux_cfg.get("enabled", False)),
        "aux_head_hidden": int(aux_cfg.get("hidden_size", 64)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(config_path="../configs", config_name="default", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Hydra v1.3+ does NOT chdir into the run dir by default — relative paths
    # would land at the project root, scattering logs/tensorboard. Anchor
    # everything explicitly on Hydra's runtime output_dir.
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    log_file_path = str(run_dir / "train.log")
    setup_logging(log_file=log_file_path)
    seed_everything(cfg.seed)

    device = _resolve_device(cfg.device)
    logger.info(f"Device: {device}")
    logger.info(f"Run dir: {run_dir}")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / "tensorboard"

    # ---- Network ----
    _net_cfg = _network_cfg_dict(cfg)
    network = build_network(
        cfg,
        num_planes=_net_cfg["num_planes"],
        num_positions=_net_cfg["num_positions"],
        action_space_size=_net_cfg["action_space_size"],
    )
    network.to(device)
    logger.info(
        f"Network: ResNet{cfg.network.num_blocks}×{cfg.network.num_channels}, "
        f"params={sum(p.numel() for p in network.parameters()):,}"
    )

    # ---- Trainer ----
    mlflow_cfg = cfg.get("mlflow", None)
    if mlflow_cfg is not None and bool(mlflow_cfg.get("enabled", False)):
        mlflow_uri: str | None = mlflow_cfg.get("tracking_uri")
        mlflow_experiment = mlflow_cfg.get("experiment_name", "morris-az")
        mlflow_run_name = mlflow_cfg.get("run_name") or run_dir.name
    else:
        mlflow_uri = None
        mlflow_experiment = "morris-az"
        mlflow_run_name = None

    _aux_cfg = cfg.get("aux_heads", {}) or {}
    trainer = Trainer(
        network=network,
        device=device,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        max_grad_norm=cfg.training.max_grad_norm,
        lr_decay_steps=cfg.training.lr_decay_steps,
        mixed_precision=cfg.training.mixed_precision,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=cfg.training.checkpoint_interval,
        config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore[arg-type]
        value_head_type=cfg.network.get("value_head_type", "scalar"),
        mlflow_uri=mlflow_uri,
        mlflow_experiment=mlflow_experiment,
        mlflow_run_name=mlflow_run_name,
        aux_heads_enabled=bool(_aux_cfg.get("enabled", False)),
        aux_weight_mill=float(_aux_cfg.get("mill_diff_weight", 0.0)),
        aux_weight_pieces=float(_aux_cfg.get("pieces_diff_weight", 0.0)),
    )

    # ---- Replay buffer ----
    _game_name = cfg.get("game", "morris")
    _augment_fn = None
    if _game_name == "reversi":
        from morris_rl.env.reversi.symmetries import (
            SYMMETRY_PERMUTATIONS as _R_PERMS,
            transform_encoded_state as _r_enc,
            transform_policy as _r_pol,
        )
        from morris_rl.training.replay_buffer import SampleRecord as _SR
        def _augment_fn(sample: "_SR") -> "list[_SR]":  # type: ignore[misc]
            return [
                _SR(
                    encoded_state=_r_enc(sample.encoded_state, p),
                    policy_target=_r_pol(sample.policy_target, p),
                    value_target=sample.value_target,
                    legal_mask=_r_pol(sample.legal_mask, p),
                )
                for p in _R_PERMS[1:]
            ]
    # GraphNet emits an 11-plane encoding (7 legacy + 4 structural). Force the
    # buffer storage shape to match so worker writes don't collide with the
    # cfg.input_encoding.num_planes default of 7. Reversi keeps its own value.
    _buffer_num_planes = (
        11 if cfg.network.get("type", "resnet") == "graphnet"
        else _net_cfg["num_planes"]
    )
    buffer = ReplayBuffer(
        capacity=cfg.training.replay_buffer_size,
        use_symmetry_augmentation=cfg.training.symmetry_augmentation,
        num_planes=_buffer_num_planes,
        num_positions=_net_cfg["num_positions"],
        action_space_size=_net_cfg["action_space_size"],
        augment_fn=_augment_fn,
    )

    # Resume from checkpoint if requested. Pass the buffer so a sibling
    # .buffer.npz (if it exists) is restored — avoids re-warmup.
    resume_path = cfg.training.get("resume", None)
    if resume_path:
        trainer.load(resume_path, buffer=buffer)

    # LoRA adapters are applied AFTER loading the checkpoint so the base weights
    # load cleanly (strict=True, no adapter keys in the saved state_dict).
    # If freeze_trunk=true the optimizer is rebuilt so only lora_A / lora_B
    # receive gradients.
    lora_rank = int(cfg.network.get("lora_rank", 0))
    if lora_rank > 0:
        network.add_lora_adapters(
            rank=lora_rank,
            alpha=float(cfg.network.get("lora_alpha", 16.0)),
        )
        logger.info(f"LoRA adapters added (rank={lora_rank})")
    if cfg.network.get("freeze_trunk", False):
        network.freeze_trunk()
        trainable = sum(p.numel() for p in network.parameters() if p.requires_grad)
        logger.info(f"Trunk frozen — {trainable:,} trainable params (LoRA only)")
        trainer.rebuild_optimizer()

    # ---- Self-play workers ----
    inference_mode = cfg.self_play.get("inference_mode", "per_worker_cpu")
    inference_device = cfg.self_play.get("inference_device", "cuda" if device.type == "cuda" else "cpu")
    # Build a ResignConfig only when the feature is enabled. The class is a
    # frozen dataclass so it pickles cheaply across the spawn boundary into
    # each worker.
    resign_node = cfg.self_play.get("resign", None)
    resign_config = None
    if resign_node is not None and bool(resign_node.get("enabled", False)):
        from morris_rl.training.self_play import ResignConfig
        resign_config = ResignConfig(
            enabled=True,
            threshold=float(resign_node.get("threshold", -0.90)),
            min_consecutive_below=int(resign_node.get("min_consecutive_below", 3)),
            min_move_for_resign=int(resign_node.get("min_move_for_resign", 30)),
            verify_fraction=float(resign_node.get("verify_fraction", 0.05)),
        )
    # Same pattern for the playout-cap feature: only build the config when
    # turned on — the manager defaults to "full sims for every move" and
    # disables the second MorrisSearch instance.
    playout_cap_node = cfg.self_play.get("playout_cap", None)
    playout_cap_config = None
    if playout_cap_node is not None and bool(playout_cap_node.get("enabled", False)):
        from morris_rl.training.self_play import PlayoutCapConfig
        playout_cap_config = PlayoutCapConfig(
            enabled=True,
            full_sim_fraction=float(playout_cap_node.get("full_sim_fraction", 0.25)),
            fast_sim_count=int(playout_cap_node.get("fast_sim_count", 60)),
        )
    # Curriculum (random late-game starts) — same opt-in pattern.
    curriculum_node = cfg.self_play.get("curriculum", None)
    curriculum_config = None
    if curriculum_node is not None and bool(curriculum_node.get("enabled", False)):
        from morris_rl.training.self_play import CurriculumConfig
        curriculum_config = CurriculumConfig(
            enabled=True,
            random_start_fraction=float(
                curriculum_node.get("random_start_fraction", 0.5)
            ),
            pieces_per_player=int(curriculum_node.get("pieces_per_player", 6)),
        )
    manager = SelfPlayManager(
        network=network,
        network_cfg=_network_cfg_dict(cfg),
        num_workers=cfg.self_play.num_workers,
        num_simulations=cfg.mcts.num_simulations_train,
        temperature_threshold=10,   # first 10 moves exploratory (temp=1.0)
        dirichlet_alpha=cfg.mcts.dirichlet_alpha,
        dirichlet_epsilon=cfg.mcts.dirichlet_epsilon,
        seed=cfg.seed,
        inference_mode=inference_mode,
        inference_device=inference_device,
        max_batch_size=cfg.self_play.get("max_batch_size", 32),
        max_wait_ms=cfg.self_play.get("max_wait_ms", 5.0),
        log_file=log_file_path,
        worker_max_rss_mb=cfg.self_play.get("worker_max_rss_mb", 0),
        worker_recycle_games=cfg.self_play.get("worker_recycle_games", 0),
        resign_config=resign_config,
        playout_cap_config=playout_cap_config,
        curriculum_config=curriculum_config,
        discard_timeout_games=bool(cfg.self_play.get("discard_timeout_games", False)),
        game_name=cfg.get("game", "morris"),
    )

    logger.info(
        f"Starting {cfg.self_play.num_workers} self-play workers, "
        f"{cfg.mcts.num_simulations_train} MCTS sims/move"
    )

    with trainer, manager:
        trainer.train_concurrent(
            manager=manager,
            buffer=buffer,
            batch_size=cfg.training.batch_size,
            total_steps=cfg.training.total_steps,
            min_buffer_size=cfg.training.min_buffer_size,
            updates_per_game=cfg.training.updates_per_collected_game,
        )

    # Final checkpoint.
    trainer.save(checkpoint_dir / "checkpoint_final.pt")
    logger.info("Done.")


if __name__ == "__main__":
    main()
