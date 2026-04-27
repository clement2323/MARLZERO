"""Evaluation metrics: Elo rating and win-rate tracking.

:class:`EloTracker` maintains a live table of Elo ratings and provides methods
to update ratings after individual game results or full arena summaries.  All
unknown agents start at ``initial_rating`` (default 1500).

The standard Elo formula is used::

    expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    new_rating_a = rating_a + K * (score_a - expected_a)

where ``score_a`` is 1.0 for a win, 0.5 for a draw, and 0.0 for a loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from morris_rl.eval.arena import ArenaSummary

_DEFAULT_RATING = 1500.0
_DEFAULT_K = 32.0


# ---------------------------------------------------------------------------
# Elo tracker
# ---------------------------------------------------------------------------


class EloTracker:
    """Live Elo rating table for a pool of named agents.

    Args:
        initial_rating: Rating assigned to agents not yet in the table.
        k_factor:       Maximum rating change per game (higher = more volatile).
    """

    def __init__(
        self,
        initial_rating: float = _DEFAULT_RATING,
        k_factor: float = _DEFAULT_K,
    ) -> None:
        self._initial = initial_rating
        self._k = k_factor
        self._ratings: dict[str, float] = {}

    def rating(self, name: str) -> float:
        """Return current rating for *name* (initial_rating if unseen)."""
        return self._ratings.get(name, self._initial)

    def update(self, name_a: str, name_b: str, score_a: float) -> tuple[float, float]:
        """Update ratings after a single game result.

        Args:
            name_a:  Name of agent A.
            name_b:  Name of agent B.
            score_a: Outcome from A's perspective: 1.0=win, 0.5=draw, 0.0=loss.

        Returns:
            New ratings for (agent_a, agent_b).
        """
        ra = self.rating(name_a)
        rb = self.rating(name_b)
        expected_a = 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))
        delta = self._k * (score_a - expected_a)
        self._ratings[name_a] = ra + delta
        self._ratings[name_b] = rb - delta
        return self._ratings[name_a], self._ratings[name_b]

    def update_from_summary(
        self, name_a: str, name_b: str, summary: ArenaSummary
    ) -> tuple[float, float]:
        """Update ratings from an :class:`~morris_rl.eval.arena.ArenaSummary`.

        The aggregate score_a = (a_wins + 0.5 * draws) / total_games is used
        as a single Elo update (equivalent to averaging per-game updates with a
        constant expected score).

        Returns:
            New ratings for (name_a, name_b).
        """
        score_a = summary.win_rate_a
        return self.update(name_a, name_b, score_a)

    def ratings(self) -> dict[str, float]:
        """Return a copy of the full ratings table."""
        return dict(self._ratings)

    def leaderboard(self) -> list[tuple[str, float]]:
        """Return agents sorted by rating descending."""
        return sorted(self._ratings.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Promotion decision
# ---------------------------------------------------------------------------


def should_promote(
    summary: ArenaSummary,
    win_rate_threshold: float = 0.55,
) -> bool:
    """Return True if agent A's win rate meets the promotion threshold.

    Args:
        summary:             Arena result where agent A is the candidate.
        win_rate_threshold:  Minimum win rate (inclusive) to promote.
    """
    return summary.win_rate_a >= win_rate_threshold
