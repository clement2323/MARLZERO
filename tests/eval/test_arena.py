"""Tests for the arena tournament runner."""

from __future__ import annotations

import pytest

from morris_rl.env.rules import GameState, get_legal_actions, initial_state
from morris_rl.eval.arena import ArenaSummary, run_arena
from morris_rl.eval.baselines import RandomAgent


# ---------------------------------------------------------------------------
# ArenaSummary properties
# ---------------------------------------------------------------------------


def test_total_games() -> None:
    s = ArenaSummary(agent_a_wins=3, agent_b_wins=5, draws=2)
    assert s.total_games == 10


def test_win_rate_a_all_wins() -> None:
    s = ArenaSummary(agent_a_wins=10, agent_b_wins=0, draws=0)
    assert s.win_rate_a == pytest.approx(1.0)


def test_win_rate_a_all_losses() -> None:
    s = ArenaSummary(agent_a_wins=0, agent_b_wins=10, draws=0)
    assert s.win_rate_a == pytest.approx(0.0)


def test_win_rate_a_all_draws() -> None:
    s = ArenaSummary(agent_a_wins=0, agent_b_wins=0, draws=10)
    assert s.win_rate_a == pytest.approx(0.5)


def test_win_rate_b_complement() -> None:
    s = ArenaSummary(agent_a_wins=3, agent_b_wins=5, draws=2)
    assert s.win_rate_a + s.win_rate_b == pytest.approx(1.0)


def test_win_rate_empty_summary() -> None:
    s = ArenaSummary(agent_a_wins=0, agent_b_wins=0, draws=0)
    assert s.win_rate_a == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# run_arena output contracts
# ---------------------------------------------------------------------------


def test_arena_total_games_matches_num_games() -> None:
    a, b = RandomAgent(seed=0), RandomAgent(seed=1)
    summary = run_arena(a, b, num_games=4)
    assert summary.total_games == 4


def test_arena_counts_are_non_negative() -> None:
    a, b = RandomAgent(seed=0), RandomAgent(seed=1)
    summary = run_arena(a, b, num_games=4)
    assert summary.agent_a_wins >= 0
    assert summary.agent_b_wins >= 0
    assert summary.draws >= 0


def test_arena_counts_sum_to_total() -> None:
    a, b = RandomAgent(seed=2), RandomAgent(seed=3)
    summary = run_arena(a, b, num_games=6)
    assert summary.agent_a_wins + summary.agent_b_wins + summary.draws == 6


def test_arena_random_vs_self_completes() -> None:
    """Random vs random should complete without errors for any num_games."""
    a, b = RandomAgent(seed=10), RandomAgent(seed=11)
    summary = run_arena(a, b, num_games=10)
    assert summary.total_games == 10


def test_arena_alternates_first_player() -> None:
    """Track which agent was player-1 per game to verify alternation."""
    moves_as_p1: list[int] = []  # records agent id (0=A, 1=B) for P1 slot

    class TrackingAgent:
        def __init__(self, agent_id: int, inner: RandomAgent) -> None:
            self._id = agent_id
            self._inner = inner
            self._first_call = True

        def select_action(self, state: GameState) -> int:
            if self._first_call and state.current_player == 1:
                moves_as_p1.append(self._id)
                self._first_call = False
            return self._inner.select_action(state)

    class ResetTracker:
        """Wrap a TrackingAgent and reset its first_call flag per game."""

        def __init__(self, tracking: "TrackingAgent") -> None:
            self._t = tracking

        def select_action(self, state: GameState) -> int:
            # Detect start of new game by checking empty board.
            if state.board.sum() == 0 and not state.must_capture:
                self._t._first_call = True
            return self._t.select_action(state)

    a_inner = TrackingAgent(0, RandomAgent(seed=0))
    b_inner = TrackingAgent(1, RandomAgent(seed=1))
    a = ResetTracker(a_inner)
    b = ResetTracker(b_inner)

    run_arena(a, b, num_games=4)

    # Games 0, 2: A is P1 → id=0; games 1, 3: B is P1 → id=1
    assert moves_as_p1 == [0, 1, 0, 1]


def test_arena_same_seed_deterministic() -> None:
    """With fixed seeds, the same summary is produced twice."""
    a1, b1 = RandomAgent(seed=42), RandomAgent(seed=99)
    a2, b2 = RandomAgent(seed=42), RandomAgent(seed=99)
    s1 = run_arena(a1, b1, num_games=8)
    s2 = run_arena(a2, b2, num_games=8)
    assert s1.agent_a_wins == s2.agent_a_wins
    assert s1.agent_b_wins == s2.agent_b_wins
    assert s1.draws == s2.draws
