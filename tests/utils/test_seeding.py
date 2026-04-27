"""Tests for reproducible seeding utilities."""

import torch

from morris_rl.utils.seeding import seed_everything


def test_seed_everything_reproducibility() -> None:
    """Same seed produces identical torch random draws."""
    seed_everything(42)
    first = torch.rand(10).tolist()

    seed_everything(42)
    second = torch.rand(10).tolist()

    assert first == second


def test_different_seeds_differ() -> None:
    """Different seeds produce different draws."""
    seed_everything(0)
    draw_0 = torch.rand(10).tolist()

    seed_everything(1)
    draw_1 = torch.rand(10).tolist()

    assert draw_0 != draw_1
