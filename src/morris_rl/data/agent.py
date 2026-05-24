"""Epsilon-greedy minimax agent with rich heuristic, for warmup data generation.

The alpha-beta negamax loop is duplicated from `eval/baselines.py` rather than
shared, by explicit user choice (isolation: edits here cannot regress the arena
evaluator). The duplication is ~30 lines and stable — Morris rules will not
change again.

Public class:
    EpsilonGreedyMinimaxAgent(depth, epsilon, opening_random_k, rng, heuristic_fn)

Public method:
    select_action_with_scores(state, halfmove_idx) ->
        (action: int, root_scores: dict[int, float] | None)

`root_scores` is None when the chosen action is random (opening or ε-greedy
trigger) — in that case minimax is skipped entirely to save CPU. The
supervised training pipeline ignores those positions for the policy target.
"""

from __future__ import annotations

import random
from typing import Callable

from morris_rl.data.heuristic import rich_heuristic
from morris_rl.env.rules import (
    GameState,
    Outcome,
    apply_action,
    get_legal_actions,
    is_terminal,
)

_INF = float("inf")
_WIN_SCORE = 1000.0

HeuristicFn = Callable[[GameState], float]


def _negamax(
    state: GameState,
    depth: int,
    alpha: float,
    beta: float,
    heuristic_fn: HeuristicFn,
) -> float:
    """Negamax with alpha-beta pruning. Value from current_player's POV.

    Duplicated from eval/baselines.py:_negamax. The must_capture sub-turn is
    handled by detecting that next_state.current_player == current — no sign
    flip in that branch, because the same player still acts.
    """
    done, outcome = is_terminal(state)
    if done:
        if outcome is None or outcome == Outcome.DRAW:
            return 0.0
        return -_WIN_SCORE  # current player just lost (terminal triggered on their turn)

    if depth == 0:
        return heuristic_fn(state)

    current = state.current_player
    best = -_INF

    for action in get_legal_actions(state):
        next_state = apply_action(state, action)

        if next_state.current_player == current:
            # Must-capture sub-turn: same player keeps the move, no flip.
            score = _negamax(next_state, depth - 1, alpha, beta, heuristic_fn)
        else:
            score = -_negamax(next_state, depth - 1, -beta, -alpha, heuristic_fn)

        if score > best:
            best = score
        alpha = max(alpha, score)
        if alpha >= beta:
            break

    return best


def _root_scores(
    state: GameState,
    depth: int,
    heuristic_fn: HeuristicFn,
) -> dict[int, float]:
    """Return per-legal-action scores from the current player's POV.

    Re-uses _negamax at depth-1 on each child. The returned dict has one entry
    per legal action; values are signed (high = better for current player).
    """
    current = state.current_player
    scores: dict[int, float] = {}
    for action in get_legal_actions(state):
        next_state = apply_action(state, action)
        if next_state.current_player == current:
            score = _negamax(next_state, depth - 1, -_INF, _INF, heuristic_fn)
        else:
            score = -_negamax(next_state, depth - 1, -_INF, _INF, heuristic_fn)
        scores[int(action)] = float(score)
    return scores


class EpsilonGreedyMinimaxAgent:
    """Alpha-beta minimax with ε-greedy and random opening moves.

    Args:
        depth: search depth in plies (≥ 1).
        epsilon: probability of playing a uniformly-random legal action
                 instead of the minimax best move. Random moves SKIP the
                 minimax search entirely (root_scores is None for those).
        opening_random_k: number of initial half-moves played uniformly at
                          random regardless of epsilon. Default 5 forces
                          diverse opening positions across games.
        rng: random.Random for reproducibility per-worker.
        heuristic_fn: leaf evaluator. Default = rich_heuristic.
    """

    def __init__(
        self,
        depth: int = 5,
        epsilon: float = 0.10,
        opening_random_k: int = 5,
        rng: random.Random | None = None,
        heuristic_fn: HeuristicFn = rich_heuristic,
    ) -> None:
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        if opening_random_k < 0:
            raise ValueError(f"opening_random_k must be >= 0, got {opening_random_k}")
        self._depth = depth
        self._epsilon = epsilon
        self._opening_random_k = opening_random_k
        self._rng = rng if rng is not None else random.Random()
        self._heuristic_fn = heuristic_fn

    def select_action_with_scores(
        self,
        state: GameState,
        halfmove_idx: int,
    ) -> tuple[int, dict[int, float] | None]:
        """Choose an action; return (action, root_scores_or_None).

        root_scores is None whenever the action was random (opening or
        ε-trigger). Otherwise it maps each legal action index to its
        signed negamax score from the current player's POV.
        """
        legal = get_legal_actions(state)
        if not legal:
            raise ValueError("select_action_with_scores called on terminal state")

        if halfmove_idx < self._opening_random_k:
            return self._rng.choice(legal), None
        if self._rng.random() < self._epsilon:
            return self._rng.choice(legal), None

        scores = _root_scores(state, self._depth, self._heuristic_fn)
        # Argmax with ties broken by RNG to avoid early-game determinism.
        best_score = max(scores.values())
        best_actions = [a for a, s in scores.items() if s == best_score]
        return self._rng.choice(best_actions), scores

    def select_action(self, state: GameState) -> int:
        """Convenience wrapper matching baselines.MinimaxAgent's interface.

        Always uses minimax (no ε-greedy, no opening random), so behavior
        matches a deterministic agent for arena play.
        """
        scores = _root_scores(state, self._depth, self._heuristic_fn)
        best_score = max(scores.values())
        best_actions = [a for a, s in scores.items() if s == best_score]
        return self._rng.choice(best_actions)
