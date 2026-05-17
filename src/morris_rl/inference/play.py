"""Position evaluation utilities for the web demo inference server.

Provides helpers that bridge the game engine with the agent layer:
  - Reconstructing a :class:`GameState` from a flat action history
  - Running MCTS and extracting the top-K candidate moves
  - A human-readable description of any action index
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from morris_rl.env.board import ACTION_SPACE_SIZE, MOVE_EDGES, NUM_PLACE_CAPTURE_ACTIONS
from morris_rl.env.rules import (
    GameState,
    apply_action,
    get_legal_actions,
    initial_state,
    is_terminal,
)


class IllegalActionError(ValueError):
    """Raised when an action history contains a move that isn't legal at its turn.

    Surfaced by :func:`reconstruct_state` so the FastAPI layer can return
    HTTP 400. Defense-in-depth: catches both buggy clients and malicious input.
    """
from morris_rl.mcts.search import MorrisSearch, encode_state

# Human-readable labels for the 24 board positions.
# Column a-g, row 1-7 (matching the board diagram in board.py).
POSITION_LABELS: list[str] = [
    "a7", "d7", "g7",  # 0-2  outer top
    "g4",              # 3    outer right-mid
    "g1", "d1", "a1",  # 4-6  outer bottom
    "a4",              # 7    outer left-mid
    "b6", "d6", "f6",  # 8-10 middle top
    "f4",              # 11   middle right-mid
    "f2", "d2", "b2",  # 12-14 middle bottom
    "b4",              # 15   middle left-mid
    "c5", "d5", "e5",  # 16-18 inner top
    "e4",              # 19   inner right-mid
    "e3", "d3", "c3",  # 20-22 inner bottom
    "c4",              # 23   inner left-mid
]


def describe_action(action: int, must_capture: bool) -> str:
    """Return a human-readable description of an action index."""
    if action < NUM_PLACE_CAPTURE_ACTIONS:
        label = POSITION_LABELS[action]
        if must_capture:
            return f"Capture {label}"
        return f"Place at {label}"
    src, dst = MOVE_EDGES[action - NUM_PLACE_CAPTURE_ACTIONS]
    return f"Move {POSITION_LABELS[src]} → {POSITION_LABELS[dst]}"


def reconstruct_state(actions: list[int]) -> GameState:
    """Replay *actions* from the initial state and return the resulting state.

    Each action is validated against the legal set at its turn. An invalid move
    (e.g. capturing a piece protected by a mill while non-mill targets exist)
    raises :class:`IllegalActionError` with the offending action index. Stops
    early if the game ends before all actions are consumed.
    """
    state = initial_state()
    for i, action in enumerate(actions):
        done, _ = is_terminal(state)
        if done:
            break
        legal = get_legal_actions(state)
        if action not in legal:
            raise IllegalActionError(
                f"Action {action} at history index {i} is not legal "
                f"(legal options: {legal})"
            )
        state = apply_action(state, action)
    return state


def get_network_value(
    network: nn.Module,
    device: torch.device,
    state: GameState,
) -> float:
    """Run a single forward pass and return the value estimate in [-1, 1].

    Positive means the current player is predicted to win.
    """
    x = encode_state(state).to(device)
    full_mask = torch.ones(1, ACTION_SPACE_SIZE, dtype=torch.bool, device=device)
    with torch.no_grad():
        _, value = network(x, full_mask)
    return float(value.item())


def run_mcts_analysis(
    network: nn.Module,
    device: torch.device,
    state: GameState,
    num_simulations: int = 200,
    num_top_moves: int = 3,
    dirichlet_alpha: float = 0.3,
    dirichlet_epsilon: float = 0.25,
) -> tuple[int, list[tuple[int, float]], float]:
    """Run MCTS from *state* and return analysis results.

    Creates a fresh :class:`MorrisSearch` instance per call so that concurrent
    requests against a shared network do not interfere.

    Returns:
        Tuple of:
          - best action (argmax of visit counts at temperature ≈ 0)
          - top_moves: list of (action, visit_probability) sorted descending
          - value_estimate: network's value for the root state in [-1, 1]
    """
    search = MorrisSearch(
        network,
        device,
        num_simulations=num_simulations,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
    )
    action, visit_probs = search.run(state, temperature=1e-6, add_noise=False)

    # Extract top-K legal moves by visit probability.
    legal_indices = np.nonzero(visit_probs)[0]
    sorted_idx = legal_indices[np.argsort(visit_probs[legal_indices])[::-1]]
    top_moves = [(int(i), float(visit_probs[i])) for i in sorted_idx[:num_top_moves]]

    value = get_network_value(network, device, state)
    return int(action), top_moves, value
