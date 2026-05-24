"""Tests for the parallel warmup-dataset generator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from morris_rl.data.generator import generate_games_parallel
from morris_rl.env.rules import apply_action, initial_state, is_terminal


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


@pytest.fixture
def tiny_out_dir(tmp_path: Path) -> Path:
    return tmp_path / "warmup_test"


def test_single_worker_produces_jsonl(tiny_out_dir: Path):
    summary = generate_games_parallel(
        num_games=2,
        out_dir=tiny_out_dir,
        depth=2,
        epsilon=0.0,
        opening_random_k=1,
        num_workers=1,
        max_halfmoves=30,
        progress_every=100,
    )
    assert summary.total_games == 2
    files = list(tiny_out_dir.glob("worker_*.jsonl"))
    assert len(files) == 1
    games = _read_jsonl(files[0])
    assert len(games) == 2


def test_jsonl_schema_complete(tiny_out_dir: Path):
    generate_games_parallel(
        num_games=1,
        out_dir=tiny_out_dir,
        depth=2,
        epsilon=0.0,
        opening_random_k=0,
        num_workers=1,
        max_halfmoves=30,
        progress_every=100,
    )
    game = _read_jsonl(next(tiny_out_dir.glob("worker_*.jsonl")))[0]
    required_keys = {
        "ts", "worker", "game", "outcome", "length", "term_reason",
        "actions", "root_scores", "epsilon_random_indices",
        "opening_random_k", "depth", "epsilon",
    }
    assert required_keys.issubset(game.keys())
    assert game["game"] == "morris"
    assert game["outcome"] in (0, 1, 2)
    assert len(game["actions"]) == game["length"]
    assert len(game["root_scores"]) == game["length"]


def test_actions_replayable(tiny_out_dir: Path):
    """Each stored game can be replayed end-to-end from initial_state."""
    generate_games_parallel(
        num_games=3,
        out_dir=tiny_out_dir,
        depth=2,
        epsilon=0.1,
        opening_random_k=2,
        num_workers=1,
        max_halfmoves=40,
        progress_every=100,
    )
    games = _read_jsonl(next(tiny_out_dir.glob("worker_*.jsonl")))
    for game in games:
        state = initial_state()
        for a in game["actions"]:
            state = apply_action(state, int(a))
        # Either terminal-by-rules or we hit the cap mid-stream. Both are valid.
        done, _ = is_terminal(state)
        is_cap = game["term_reason"] == "max_halfmoves_cap"
        assert done or is_cap


def test_parallel_workers_no_collision(tiny_out_dir: Path):
    summary = generate_games_parallel(
        num_games=4,
        out_dir=tiny_out_dir,
        depth=2,
        epsilon=0.0,
        opening_random_k=1,
        num_workers=4,
        max_halfmoves=30,
        progress_every=100,
    )
    files = sorted(tiny_out_dir.glob("worker_*.jsonl"))
    assert len(files) == 4
    total = sum(len(_read_jsonl(f)) for f in files)
    assert total == 4 == summary.total_games


def test_cap_yields_draw(tiny_out_dir: Path):
    """With a very small max_halfmoves and depth=1, some games hit the cap → DRAW outcome."""
    generate_games_parallel(
        num_games=4,
        out_dir=tiny_out_dir,
        depth=1,
        epsilon=0.0,
        opening_random_k=0,
        num_workers=1,
        max_halfmoves=5,
        progress_every=100,
    )
    games = _read_jsonl(next(tiny_out_dir.glob("worker_*.jsonl")))
    # At least one game should hit the cap given how tight max_halfmoves=5 is.
    cap_games = [g for g in games if g["term_reason"] == "max_halfmoves_cap"]
    assert len(cap_games) > 0
    for g in cap_games:
        assert g["outcome"] == 0  # cap = DRAW per user decision


def test_root_scores_aligned_with_actions(tiny_out_dir: Path):
    """root_scores entries are either None (random move) or a list of {a, s} dicts
    that cover the legal actions exactly at that step."""
    generate_games_parallel(
        num_games=1,
        out_dir=tiny_out_dir,
        depth=2,
        epsilon=0.0,
        opening_random_k=2,
        num_workers=1,
        max_halfmoves=30,
        progress_every=100,
    )
    game = _read_jsonl(next(tiny_out_dir.glob("worker_*.jsonl")))[0]
    # First two steps were random (opening) → root_scores None
    assert game["root_scores"][0] is None
    assert game["root_scores"][1] is None
    # Step 2 (third half-move) was minimax → root_scores is a list of dicts
    if len(game["root_scores"]) > 2:
        rs = game["root_scores"][2]
        assert isinstance(rs, list)
        for entry in rs:
            assert set(entry.keys()) == {"a", "s"}
            assert isinstance(entry["a"], int)
            assert isinstance(entry["s"], float)
