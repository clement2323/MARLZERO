"""Two-agent tournament arena.

The :func:`run_arena` function plays a fixed number of games between two agents,
alternating which agent plays as Player 1 (who moves first) to remove first-move
bias.  Results are aggregated into an :class:`ArenaSummary`.

Agents must satisfy the :class:`Agent` protocol — a single ``select_action``
method that maps a :class:`~morris_rl.env.rules.GameState` to an action index.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from morris_rl.env.rules import (
    GameState,
    Outcome,
    apply_action,
    initial_state,
    is_terminal,
    opponent,
)
from morris_rl.utils.logging import logger


class Agent(Protocol):
    """Minimal interface every evaluation agent must implement."""

    def select_action(self, state: GameState) -> int:
        """Return a legal action index for the given state."""
        ...


# ---------------------------------------------------------------------------
# Result data structure
# ---------------------------------------------------------------------------


@dataclass
class ArenaSummary:
    """Aggregated results of a multi-game tournament."""

    agent_a_wins: int
    agent_b_wins: int
    draws: int

    @property
    def total_games(self) -> int:
        return self.agent_a_wins + self.agent_b_wins + self.draws

    @property
    def win_rate_a(self) -> float:
        """Win rate for agent A (draws count as 0.5)."""
        if self.total_games == 0:
            return 0.0
        return (self.agent_a_wins + 0.5 * self.draws) / self.total_games

    @property
    def win_rate_b(self) -> float:
        return 1.0 - self.win_rate_a


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_arena(
    agent_a: Agent,
    agent_b: Agent,
    num_games: int,
    verbose: bool = False,
) -> ArenaSummary:
    """Play ``num_games`` games between two agents and return aggregated results.

    Games alternate which agent plays as Player 1 so that first-move bias is
    evenly distributed.  On even-indexed games agent A is Player 1; on odd-
    indexed games agent B is Player 1.

    Args:
        agent_a:   First agent.
        agent_b:   Second agent.
        num_games: Total games to play (recommend an even number).
        verbose:   Log progress every 10 games.

    Returns:
        :class:`ArenaSummary` with per-agent win/draw/loss totals.
    """
    a_wins = 0
    b_wins = 0
    draws = 0

    for game_idx in range(num_games):
        # Alternate who plays as Player 1 (moves first).
        if game_idx % 2 == 0:
            player1_agent, player2_agent = agent_a, agent_b
            a_is_p1 = True
        else:
            player1_agent, player2_agent = agent_b, agent_a
            a_is_p1 = False

        outcome = _play_game(player1_agent, player2_agent)

        if outcome == Outcome.DRAW or outcome is None:
            draws += 1
        elif (outcome == Outcome.PLAYER_1_WINS) == a_is_p1:
            a_wins += 1
        else:
            b_wins += 1

        if verbose and (game_idx + 1) % 10 == 0:
            logger.info(
                f"Arena: {game_idx + 1}/{num_games} games — "
                f"A:{a_wins} B:{b_wins} D:{draws}"
            )

    return ArenaSummary(agent_a_wins=a_wins, agent_b_wins=b_wins, draws=draws)


# ---------------------------------------------------------------------------
# Internal game runner
# ---------------------------------------------------------------------------


def _play_game(player1: Agent, player2: Agent) -> Outcome | None:
    """Play one complete game. Returns the outcome (None treated as ongoing)."""
    state: GameState = initial_state()
    agents = {1: player1, 2: player2}

    while True:
        done, outcome = is_terminal(state)
        if done:
            return outcome

        action = agents[state.current_player].select_action(state)
        state = apply_action(state, action)
