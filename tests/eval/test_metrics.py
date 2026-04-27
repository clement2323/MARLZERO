"""Tests for Elo rating and promotion logic."""

from __future__ import annotations

import pytest

from morris_rl.eval.arena import ArenaSummary
from morris_rl.eval.metrics import EloTracker, should_promote


# ---------------------------------------------------------------------------
# EloTracker
# ---------------------------------------------------------------------------


def test_initial_rating_default() -> None:
    tracker = EloTracker()
    assert tracker.rating("unknown") == pytest.approx(1500.0)


def test_initial_rating_custom() -> None:
    tracker = EloTracker(initial_rating=1200.0)
    assert tracker.rating("x") == pytest.approx(1200.0)


def test_win_increases_winner_rating() -> None:
    tracker = EloTracker()
    old_a = tracker.rating("a")
    tracker.update("a", "b", score_a=1.0)
    assert tracker.rating("a") > old_a


def test_loss_decreases_loser_rating() -> None:
    tracker = EloTracker()
    old_a = tracker.rating("a")
    tracker.update("a", "b", score_a=0.0)
    assert tracker.rating("a") < old_a


def test_draw_between_equals_minimal_change() -> None:
    tracker = EloTracker()
    old_a = tracker.rating("a")
    old_b = tracker.rating("b")
    tracker.update("a", "b", score_a=0.5)
    assert abs(tracker.rating("a") - old_a) < 1.0
    assert abs(tracker.rating("b") - old_b) < 1.0


def test_ratings_sum_preserved() -> None:
    """Zero-sum property: total Elo in the pool is conserved."""
    tracker = EloTracker()
    before = tracker.rating("a") + tracker.rating("b")
    tracker.update("a", "b", score_a=1.0)
    after = tracker.rating("a") + tracker.rating("b")
    assert after == pytest.approx(before, abs=1e-6)


def test_strong_player_beats_weak_gives_small_gain() -> None:
    """When a highly-rated player beats a low-rated one, gain is small."""
    tracker = EloTracker()
    tracker._ratings["strong"] = 2000.0
    tracker._ratings["weak"] = 1000.0
    old_strong = tracker.rating("strong")
    tracker.update("strong", "weak", score_a=1.0)
    gain = tracker.rating("strong") - old_strong
    assert 0.0 < gain < 5.0  # expected score ≈ 1, so actual gain is tiny


def test_update_returns_new_ratings() -> None:
    tracker = EloTracker()
    ra, rb = tracker.update("a", "b", score_a=1.0)
    assert ra == pytest.approx(tracker.rating("a"))
    assert rb == pytest.approx(tracker.rating("b"))


def test_update_from_summary_perfect_win() -> None:
    tracker = EloTracker()
    summary = ArenaSummary(agent_a_wins=10, agent_b_wins=0, draws=0)
    old_a = tracker.rating("a")
    tracker.update_from_summary("a", "b", summary)
    assert tracker.rating("a") > old_a


def test_update_from_summary_consistent_with_update() -> None:
    """update_from_summary at 50 % win rate should behave like draw."""
    tracker1 = EloTracker()
    tracker2 = EloTracker()
    summary = ArenaSummary(agent_a_wins=5, agent_b_wins=5, draws=0)
    tracker1.update_from_summary("a", "b", summary)
    tracker2.update("a", "b", score_a=0.5)
    assert tracker1.rating("a") == pytest.approx(tracker2.rating("a"), abs=1e-4)


def test_ratings_returns_copy() -> None:
    tracker = EloTracker()
    tracker.update("a", "b", 1.0)
    snapshot = tracker.ratings()
    tracker.update("a", "b", 1.0)
    assert snapshot["a"] != tracker.rating("a")


def test_leaderboard_sorted_descending() -> None:
    tracker = EloTracker()
    tracker.update("a", "b", score_a=1.0)
    tracker.update("a", "c", score_a=1.0)
    lb = tracker.leaderboard()
    ratings_only = [r for _, r in lb]
    assert ratings_only == sorted(ratings_only, reverse=True)


# ---------------------------------------------------------------------------
# should_promote
# ---------------------------------------------------------------------------


def test_promote_above_threshold() -> None:
    summary = ArenaSummary(agent_a_wins=60, agent_b_wins=40, draws=0)
    assert should_promote(summary, win_rate_threshold=0.55) is True


def test_no_promote_below_threshold() -> None:
    summary = ArenaSummary(agent_a_wins=50, agent_b_wins=50, draws=0)
    assert should_promote(summary, win_rate_threshold=0.55) is False


def test_promote_exactly_at_threshold() -> None:
    summary = ArenaSummary(agent_a_wins=55, agent_b_wins=45, draws=0)
    assert should_promote(summary, win_rate_threshold=0.55) is True


def test_promote_draws_count_half() -> None:
    """50 wins + 10 draws out of 100 → win rate = 0.55 → should promote."""
    summary = ArenaSummary(agent_a_wins=50, agent_b_wins=40, draws=10)
    assert should_promote(summary, win_rate_threshold=0.55) is True
