"""Tests for board constants (adjacency symmetry, mill correctness)."""

from morris_rl.env.board import (
    ACTION_SPACE_SIZE,
    ADJACENCY,
    MILLS,
    MILLS_BY_POSITION,
    NUM_POSITIONS,
)


def test_adjacency_is_symmetric() -> None:
    for i, neighbours in enumerate(ADJACENCY):
        for j in neighbours:
            assert i in ADJACENCY[j], f"{j} is adjacent to {i} but not vice-versa"


def test_adjacency_no_self_loops() -> None:
    for i, neighbours in enumerate(ADJACENCY):
        assert i not in neighbours


def test_all_mills_have_length_3() -> None:
    for mill in MILLS:
        assert len(mill) == 3


def test_mills_contain_valid_positions() -> None:
    for mill in MILLS:
        for pos in mill:
            assert 0 <= pos < NUM_POSITIONS


def test_mills_by_position_matches_mills() -> None:
    for i in range(NUM_POSITIONS):
        expected = [m for m in MILLS if i in m]
        assert MILLS_BY_POSITION[i] == expected


def test_every_position_in_exactly_two_mills() -> None:
    # Every position is on exactly 2 lines (verified from board geometry).
    for i in range(NUM_POSITIONS):
        assert len(MILLS_BY_POSITION[i]) == 2, f"position {i} in {len(MILLS_BY_POSITION[i])} mills"


def test_mills_are_unique() -> None:
    as_sets = [frozenset(m) for m in MILLS]
    assert len(as_sets) == len(set(as_sets)), "duplicate mills detected"


def test_action_space_size() -> None:
    assert ACTION_SPACE_SIZE == 600
