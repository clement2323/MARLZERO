"""Parallel game generation for warmup dataset.

Spawn N workers via multiprocessing; each plays a slice of the game budget and
streams completed games into its own JSONL file (`worker_{id}.jsonl`). Files
are append+flush after every game so an external observer can tail them live.

The cap at `max_halfmoves` (default 200) is enforced here: a game that exceeds
the cap is terminated as a DRAW (outcome=0), without piece-count tiebreak —
explicit user decision, since cap-truncated games carry no reliable signal.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

from morris_rl.data.agent import EpsilonGreedyMinimaxAgent
from morris_rl.env.rules import (
    MAX_HALFMOVES,
    MAX_TOTAL_HALFMOVES,
    THREEFOLD_LIMIT,
    Outcome,
    apply_action,
    initial_state,
    is_terminal,
    pieces_on_board,
)
from morris_rl.env.rules import _position_key as _rules_position_key


@dataclass
class WorkerConfig:
    worker_id: int
    seed: int
    n_games: int
    depth: int
    epsilon: float
    opening_random_k: int
    max_halfmoves: int
    out_path: Path


@dataclass
class GenerationSummary:
    total_games: int
    elapsed_seconds: float
    per_worker_counts: list[int] = field(default_factory=list)


def _play_one_game(
    agent_p1: EpsilonGreedyMinimaxAgent,
    agent_p2: EpsilonGreedyMinimaxAgent,
    max_halfmoves: int,
) -> dict:
    """Play a single game and return the JSON payload (no IO).

    Two agents are passed in so each player can have its own RNG, ensuring
    P1 and P2 don't share epsilon-trigger draws within a game.
    """
    agents = {1: agent_p1, 2: agent_p2}
    state = initial_state()
    actions: list[int] = []
    root_scores_per_step: list[list[dict] | None] = []
    epsilon_random_indices: list[int] = []
    halfmove_idx = 0

    while True:
        # Order matters: cap is checked BEFORE engine terminal so the user's
        # 200-halfmove cap supersedes rules.is_terminal's 300 safety net.
        if state.total_halfmoves >= max_halfmoves:
            outcome_int = 0
            term_reason = "max_halfmoves_cap"
            break

        done, outcome = is_terminal(state)
        if done:
            outcome_int = 0 if (outcome is None or outcome == Outcome.DRAW) else int(outcome)
            term_reason = _classify_terminal(state)
            break

        action, scores = agents[state.current_player].select_action_with_scores(
            state, halfmove_idx
        )
        if scores is None:
            root_scores_per_step.append(None)
            # Distinguish opening-random (first k plies) from ε-greedy triggers:
            # opening_random_k known per-agent, but the index < k case is
            # already implied; we only record ε events post-opening.
            if halfmove_idx >= agents[state.current_player]._opening_random_k:
                epsilon_random_indices.append(halfmove_idx)
        else:
            root_scores_per_step.append(
                [{"a": int(a), "s": float(s)} for a, s in scores.items()]
            )
        actions.append(int(action))
        state = apply_action(state, action)
        halfmove_idx += 1

    return {
        "outcome": outcome_int,
        "length": len(actions),
        "term_reason": term_reason,
        "actions": actions,
        "root_scores": root_scores_per_step,
        "epsilon_random_indices": epsilon_random_indices,
    }


def _classify_terminal(state) -> str:
    """Re-derive which is_terminal branch fired, for the JSONL `term_reason` field.

    Mirrors the cascade in rules.is_terminal exactly (same order of checks).
    """
    if state.total_halfmoves >= MAX_TOTAL_HALFMOVES:
        return "max_total_halfmoves_safety_cap"
    key = _rules_position_key(state)
    if state.position_counts.get(key, 0) >= THREEFOLD_LIMIT:
        return "threefold"
    if state.halfmove_clock >= MAX_HALFMOVES:
        return "halfmove_clock_50"
    player = state.current_player
    if state.pieces_in_hand[player - 1] == 0 and pieces_on_board(state.board, player) < 3:
        return "pieces_below_3"
    return "no_legal_moves"


def _worker_fn(cfg: WorkerConfig, status_queue: mp.Queue) -> None:
    """Worker entry point. Plays cfg.n_games and streams them to disk."""
    # Each worker gets fully independent RNGs.
    base_rng = random.Random(cfg.seed)
    cfg.out_path.parent.mkdir(parents=True, exist_ok=True)

    with cfg.out_path.open("a") as fh:
        for game_idx in range(cfg.n_games):
            # Two RNGs per game — one per agent — derived from the worker seed.
            seed_p1 = base_rng.randrange(2**31)
            seed_p2 = base_rng.randrange(2**31)
            agent_p1 = EpsilonGreedyMinimaxAgent(
                depth=cfg.depth,
                epsilon=cfg.epsilon,
                opening_random_k=cfg.opening_random_k,
                rng=random.Random(seed_p1),
            )
            agent_p2 = EpsilonGreedyMinimaxAgent(
                depth=cfg.depth,
                epsilon=cfg.epsilon,
                opening_random_k=cfg.opening_random_k,
                rng=random.Random(seed_p2),
            )

            t0 = time.time()
            game = _play_one_game(agent_p1, agent_p2, cfg.max_halfmoves)
            elapsed = time.time() - t0

            payload = {
                "ts": time.time(),
                "worker": cfg.worker_id,
                "game": "morris",
                **game,
                "opening_random_k": cfg.opening_random_k,
                "depth": cfg.depth,
                "epsilon": cfg.epsilon,
                "seed_p1": seed_p1,
                "seed_p2": seed_p2,
                "wall_seconds": elapsed,
            }
            fh.write(json.dumps(payload) + "\n")
            fh.flush()
            status_queue.put(("game_done", cfg.worker_id, game["length"], game["outcome"]))

    status_queue.put(("worker_done", cfg.worker_id, cfg.n_games, 0))


def generate_games_parallel(
    num_games: int,
    out_dir: Path,
    depth: int = 5,
    epsilon: float = 0.10,
    opening_random_k: int = 5,
    num_workers: int = 10,
    seed: int = 0,
    max_halfmoves: int = 200,
    progress_every: int = 10,
) -> GenerationSummary:
    """Spawn `num_workers` minimax workers and stream games to `out_dir`.

    Each worker writes its own JSONL file `worker_{id}.jsonl` with flush after
    every completed game. The master collects per-game counters via a status
    queue and prints progress every `progress_every` games.

    Returns a `GenerationSummary` with total counts and elapsed wallclock.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    status_queue: mp.Queue = ctx.Queue()

    # Distribute games as evenly as possible, with the remainder spread over
    # the first few workers — this avoids one worker idling at the end.
    base, extra = divmod(num_games, num_workers)
    assignments = [base + (1 if i < extra else 0) for i in range(num_workers)]

    workers: list[mp.Process] = []
    for wid, n in enumerate(assignments):
        if n == 0:
            continue
        cfg = WorkerConfig(
            worker_id=wid,
            seed=seed + wid * 10_007,
            n_games=n,
            depth=depth,
            epsilon=epsilon,
            opening_random_k=opening_random_k,
            max_halfmoves=max_halfmoves,
            out_path=out_dir / f"worker_{wid}.jsonl",
        )
        p = ctx.Process(target=_worker_fn, args=(cfg, status_queue))
        p.start()
        workers.append(p)

    t_start = time.time()
    per_worker_counts = [0] * num_workers
    games_done = 0
    workers_done = 0
    expected_workers = sum(1 for n in assignments if n > 0)
    cum_length = 0
    cum_decisive = 0

    while workers_done < expected_workers:
        msg = status_queue.get()
        kind = msg[0]
        if kind == "game_done":
            _, wid, length, outcome = msg
            per_worker_counts[wid] += 1
            games_done += 1
            cum_length += length
            cum_decisive += 1 if outcome != 0 else 0
            if games_done % progress_every == 0 or games_done == num_games:
                elapsed = time.time() - t_start
                games_per_sec = games_done / elapsed if elapsed > 0 else 0.0
                eta = (num_games - games_done) / games_per_sec if games_per_sec > 0 else 0.0
                print(
                    f"  [{games_done:>5d}/{num_games}]  "
                    f"mean_len={cum_length / games_done:.1f}  "
                    f"decisive={cum_decisive / games_done * 100:.1f}%  "
                    f"rate={games_per_sec:.2f} g/s  "
                    f"eta={eta / 60:.1f} min",
                    flush=True,
                )
        elif kind == "worker_done":
            workers_done += 1

    for p in workers:
        p.join()

    elapsed = time.time() - t_start
    return GenerationSummary(
        total_games=games_done,
        elapsed_seconds=elapsed,
        per_worker_counts=per_worker_counts,
    )
