"""State encoding for Reversi/Othello.

The encoded tensor has shape (1, NUM_PLANES, NUM_POSITIONS) = (1, 3, 64).

Plane layout:
    Plane 0: current player's pieces    — 1.0 at each owned position
    Plane 1: opponent's pieces          — 1.0 at each opponent position
    Plane 2: pass urgency signal        — pass_count / 2.0 (scalar broadcast)
              reaches 1.0 when the game is about to terminate by double-pass,
              giving the network a near-terminal warning without extra planes.
"""

from __future__ import annotations

import numpy as np
import torch

from morris_rl.env.reversi.rules import GameState

NUM_PLANES: int = 3


def encode_state(state: GameState) -> torch.Tensor:
    """Encode a Reversi GameState into a (1, 3, 64) float32 tensor.

    Args:
        state: The game state to encode.

    Returns:
        Tensor of shape (1, NUM_PLANES, NUM_POSITIONS) = (1, 3, 64).
    """
    planes = np.zeros((NUM_PLANES, 64), dtype=np.float32)

    # Plane 0: current player's pieces
    planes[0] = (state.board == state.current_player).astype(np.float32)

    # Plane 1: opponent's pieces
    opp = 3 - state.current_player  # PLAYER_1=1 → 2=PLAYER_2, PLAYER_2=2 → 1=PLAYER_1
    planes[1] = (state.board == opp).astype(np.float32)

    # Plane 2: pass urgency (scalar, broadcast to all positions)
    # Values: 0.0 (no pass yet), 0.5 (one pass), 1.0 (second pass → terminal)
    planes[2] = state.pass_count / 2.0

    return torch.from_numpy(planes).unsqueeze(0)  # (1, 3, 64)
