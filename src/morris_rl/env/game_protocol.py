"""Game-agnostic protocol for plugging different board games into the AlphaZero pipeline.

A GameEnv instance encapsulates all game-specific logic so that the training
loop, MCTS, network factory, and replay buffer remain game-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch


@dataclass
class GameEnv:
    """Container for all game-specific functions.

    All callables use the game's native state type (opaque to the pipeline).

    Attributes:
        name: Short identifier, e.g. ``"morris"`` or ``"reversi"``.
        num_planes: Number of input planes in the encoded state tensor.
        action_space_size: Total number of actions (including illegal ones).
        num_positions: Number of board positions (used by the replay buffer).
        initial_state: ``() -> state`` — returns a fresh start state.
        get_legal_actions: ``state -> list[int]`` — legal action indices.
        apply_action: ``(state, action) -> state`` — never mutates the input.
        is_terminal: ``state -> (bool, outcome | None)`` — outcome is None when
            the game is not over.
        encode_state: ``state -> torch.Tensor`` — shape ``(1, num_planes, num_positions)``.
        augment_samples: Optional symmetry augmentation callable. When None,
            augmentation is disabled for this game.
        random_late_game_state: Optional curriculum sampler returning a random
            mid/late-game state. When None, curriculum is disabled.
        random_placement_phase_state: Optional curriculum sampler returning a
            random placement-phase state. When None, this curriculum mode is
            disabled.
    """

    name: str
    num_planes: int
    action_space_size: int
    num_positions: int
    initial_state: Callable[[], Any]
    get_legal_actions: Callable[[Any], list[int]]
    apply_action: Callable[[Any, int], Any]
    is_terminal: Callable[[Any], tuple[bool, Any]]
    encode_state: Callable[[Any], torch.Tensor]
    augment_samples: Callable | None = None  # symmetry augmentation, None = disabled
    random_late_game_state: Callable | None = None
    random_placement_phase_state: Callable | None = None
