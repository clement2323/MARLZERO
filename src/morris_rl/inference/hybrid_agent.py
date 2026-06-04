"""Hybrid agent: alpha-beta heuristic in placement, tablebase in movement.

Designed for the web demo in the Flying variant. Routing:

* ``state.must_capture`` (mill-formed sub-turn): tablebase only exposes
  move+capture as atomic actions, so the capture leg is delegated to
  alpha-beta depth-4 which picks well from a small list.

* Placement (``pieces_in_hand != (0, 0)``): tablebase doesn't cover
  placement positions. Alpha-beta depth-4 with the rich heuristic.

* Movement (hands empty, variant=FLYING, both sides in [3..9] pieces):
  query the tablebase. WIN/DRAW/LOSS verdict drives ``value_estimate``;
  ``top_moves`` from the TB are used directly.

* Tablebase miss (out of coverage, no-flying, transport error): fall
  back to alpha-beta depth-4.
"""

from __future__ import annotations

from typing import Any

from morris_rl.env.rules import GameState, Variant
from morris_rl.eval.baselines import MinimaxAgent
from morris_rl.inference.tablebase_client import (
    TablebaseClient,
    WAVE_DRAW,
    WAVE_LOSS,
    WAVE_WIN,
)


_VALUE_FROM_VERDICT: dict[int, float] = {
    WAVE_WIN: 1.0,
    WAVE_DRAW: 0.0,
    WAVE_LOSS: -1.0,
}


class HybridAgent:
    """Heuristic-then-tablebase agent for the Flying variant web demo."""

    def __init__(
        self,
        tablebase_client: TablebaseClient,
        minimax_depth: int = 4,
    ) -> None:
        self._tb = tablebase_client
        self._minimax = MinimaxAgent(depth=minimax_depth)

    def select_action(self, state: GameState) -> int:
        """Return the best action for *state*. Always returns a legal action."""
        result = self._tb_lookup(state)
        if result is not None:
            return int(result["action"])
        return self._minimax.select_action(state)

    def analyze(
        self, state: GameState
    ) -> tuple[int, list[tuple[int, float]], float]:
        """Return (action, top_moves, value_estimate) for the web UI.

        Matches the shape of ``inference.play.run_mcts_analysis`` so the
        FastAPI handler can swap agents without bespoke per-agent branches.
        """
        result = self._tb_lookup(state)
        if result is not None:
            action = int(result["action"])
            value = _VALUE_FROM_VERDICT.get(int(result["verdict"]), 0.0)
            # Tablebase top_moves are sorted best-first. We surface them as
            # equal-weight visits since the TB doesn't have probabilistic
            # opinions — only verdict/DTW preferences.
            tm = result["top_moves"][:3]
            n = max(len(tm), 1)
            top_moves = [(int(m["action"]), 1.0 / n) for m in tm]
            if not top_moves:
                top_moves = [(action, 1.0)]
            return action, top_moves, value

        # Fallback: minimax. We can't extract a probability distribution
        # cheaply from negamax, so we surface just the chosen move.
        action = self._minimax.select_action(state)
        return action, [(action, 1.0)], 0.0

    def _tb_lookup(self, state: GameState) -> dict[str, Any] | None:
        """Run the tablebase query if the state is in its domain."""
        if state.variant != Variant.FLYING:
            return None
        if state.must_capture:
            return None
        if state.pieces_in_hand != (0, 0):
            return None
        return self._tb.query(state)
