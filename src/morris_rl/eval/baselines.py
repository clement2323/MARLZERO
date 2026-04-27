"""Baseline agents for evaluation.

Agents
------
:class:`RandomAgent`
    Selects uniformly at random from legal actions.  The simplest possible
    baseline; a trained agent should win close to 100 % against it.

:class:`MinimaxAgent`
    Alpha-beta negamax search to a fixed depth.  ``depth=3`` is a light
    baseline; ``depth=7`` is a reasonable mid-strength opponent on this CPU.
    The heuristic evaluates material balance and mill count.

:class:`NetworkAgent`
    Wraps :class:`~morris_rl.mcts.search.MorrisSearch` for use in the arena.
    Uses temperature=0 (argmax over visit counts) and no Dirichlet noise so
    that evaluation is deterministic.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from morris_rl.env.board import MILLS, NUM_POSITIONS
from morris_rl.env.rules import (
    GameState,
    Outcome,
    apply_action,
    get_legal_actions,
    is_terminal,
    opponent,
    pieces_on_board,
)
from morris_rl.mcts.search import MorrisSearch

_INF = float("inf")
_WIN_SCORE = 1000.0
_ARGMAX_TEMP = 1e-6


# ---------------------------------------------------------------------------
# Random agent
# ---------------------------------------------------------------------------


class RandomAgent:
    """Selects a uniformly random legal action."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def select_action(self, state: GameState) -> int:
        return self._rng.choice(get_legal_actions(state))


# ---------------------------------------------------------------------------
# Minimax agent (negamax + alpha-beta)
# ---------------------------------------------------------------------------


def _heuristic(state: GameState) -> float:
    """Static evaluation from the current player's perspective.

    Weights (tuned empirically for Nine Men's Morris):
      +3 per net piece advantage (on board + in hand)
      +2 per net mill advantage
    """
    player = state.current_player
    opp = opponent(player)
    board = state.board
    hand = state.pieces_in_hand

    player_total = int(np.sum(board == player)) + hand[player - 1]
    opp_total = int(np.sum(board == opp)) + hand[opp - 1]
    player_mills = sum(1 for m in MILLS if all(board[p] == player for p in m))
    opp_mills = sum(1 for m in MILLS if all(board[p] == opp for p in m))

    return 3.0 * (player_total - opp_total) + 2.0 * (player_mills - opp_mills)


def _negamax(state: GameState, depth: int, alpha: float, beta: float) -> float:
    """Negamax with alpha-beta pruning.

    Returns a value from the perspective of ``state.current_player``.
    Handles the must-capture sub-turn (same player, no sign flip) correctly.
    """
    done, outcome = is_terminal(state)
    if done:
        if outcome is None or outcome == Outcome.DRAW:
            return 0.0
        return -_WIN_SCORE  # current player lost

    if depth == 0:
        return _heuristic(state)

    current = state.current_player
    best = -_INF

    for action in get_legal_actions(state):
        next_state = apply_action(state, action)

        if next_state.current_player == current:
            # Must-capture sub-turn: same player, no perspective flip.
            score = _negamax(next_state, depth - 1, alpha, beta)
        else:
            # Opponent's turn: negate value and swap alpha/beta.
            score = -_negamax(next_state, depth - 1, -beta, -alpha)

        if score > best:
            best = score
        alpha = max(alpha, score)
        if alpha >= beta:
            break

    return best


class MinimaxAgent:
    """Alpha-beta negamax agent searching to a fixed depth.

    Args:
        depth: Search depth in half-moves (plies).  Higher is stronger but
               slower.  ``depth=3`` is a light baseline; ``depth=7`` is a
               reasonable mid-strength opponent.
    """

    def __init__(self, depth: int = 3) -> None:
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        self._depth = depth

    def select_action(self, state: GameState) -> int:
        current = state.current_player
        best_action = -1
        best_score = -_INF
        alpha = -_INF
        beta = _INF

        for action in get_legal_actions(state):
            next_state = apply_action(state, action)
            if next_state.current_player == current:
                score = _negamax(next_state, self._depth - 1, alpha, beta)
            else:
                score = -_negamax(next_state, self._depth - 1, -beta, -alpha)

            if score > best_score:
                best_score = score
                best_action = action
            alpha = max(alpha, score)

        return best_action


# ---------------------------------------------------------------------------
# Network agent (wraps MorrisSearch)
# ---------------------------------------------------------------------------


class NetworkAgent:
    """AlphaZero agent backed by MCTS + a trained network.

    Uses temperature ≈ 0 (argmax over visit counts) and no Dirichlet noise
    so that evaluation games are deterministic and reproducible.

    Args:
        network:         Trained policy/value network.
        device:          Device for inference.
        num_simulations: MCTS simulations per move.
    """

    def __init__(
        self,
        network: nn.Module,
        device: torch.device,
        num_simulations: int = 800,
    ) -> None:
        self._search = MorrisSearch(network, device, num_simulations=num_simulations)

    def select_action(self, state: GameState) -> int:
        action, _ = self._search.run(state, temperature=_ARGMAX_TEMP, add_noise=False)
        return action
