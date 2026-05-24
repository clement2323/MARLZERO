"""Tests for EpsilonGreedyMinimaxAgent."""

from __future__ import annotations

import random

import pytest

from morris_rl.data.agent import EpsilonGreedyMinimaxAgent
from morris_rl.env.rules import (
    apply_action,
    get_legal_actions,
    initial_state,
    is_terminal,
)


def test_invalid_depth_raises():
    with pytest.raises(ValueError):
        EpsilonGreedyMinimaxAgent(depth=0)


def test_invalid_epsilon_raises():
    with pytest.raises(ValueError):
        EpsilonGreedyMinimaxAgent(depth=2, epsilon=-0.1)
    with pytest.raises(ValueError):
        EpsilonGreedyMinimaxAgent(depth=2, epsilon=1.1)


def test_opening_random_returns_none_scores():
    """During opening-random phase, root_scores is None."""
    agent = EpsilonGreedyMinimaxAgent(
        depth=2, epsilon=0.0, opening_random_k=3, rng=random.Random(0)
    )
    state = initial_state()
    for halfmove_idx in range(3):
        action, scores = agent.select_action_with_scores(state, halfmove_idx)
        assert scores is None
        assert action in get_legal_actions(state)
        state = apply_action(state, action)


def test_post_opening_returns_scores():
    """After opening_random_k, minimax kicks in and returns a scores dict
    with one entry per legal action."""
    agent = EpsilonGreedyMinimaxAgent(
        depth=2, epsilon=0.0, opening_random_k=2, rng=random.Random(0)
    )
    state = initial_state()
    state = apply_action(state, agent.select_action_with_scores(state, 0)[0])
    state = apply_action(state, agent.select_action_with_scores(state, 1)[0])
    action, scores = agent.select_action_with_scores(state, 2)
    assert scores is not None
    legal = set(get_legal_actions(state))
    assert set(scores.keys()) == legal
    assert action in legal


def test_epsilon_one_always_random():
    """ε=1 (after opening) → root_scores always None (random chosen)."""
    agent = EpsilonGreedyMinimaxAgent(
        depth=2, epsilon=1.0, opening_random_k=0, rng=random.Random(42)
    )
    state = initial_state()
    for halfmove_idx in range(5):
        _action, scores = agent.select_action_with_scores(state, halfmove_idx)
        assert scores is None
        _action_int = int(_action)
        state = apply_action(state, _action_int)
        done, _ = is_terminal(state)
        if done:
            break


def test_same_seed_same_actions():
    """Reproducibility: identical RNG → identical action sequence."""
    def play_five(seed: int) -> list[int]:
        agent = EpsilonGreedyMinimaxAgent(
            depth=2, epsilon=0.5, opening_random_k=1, rng=random.Random(seed)
        )
        state = initial_state()
        actions: list[int] = []
        for halfmove_idx in range(5):
            done, _ = is_terminal(state)
            if done:
                break
            a, _ = agent.select_action_with_scores(state, halfmove_idx)
            actions.append(int(a))
            state = apply_action(state, int(a))
        return actions

    assert play_five(123) == play_five(123)


def test_select_action_compat():
    """select_action returns an int in legal actions."""
    agent = EpsilonGreedyMinimaxAgent(
        depth=2, epsilon=0.0, opening_random_k=0, rng=random.Random(0)
    )
    state = initial_state()
    a = agent.select_action(state)
    assert a in get_legal_actions(state)
