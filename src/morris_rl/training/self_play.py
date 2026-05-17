"""Parallel self-play data generation.

Architecture
------------
Each :class:`SelfPlayManager` spawns ``num_workers`` independent processes.
Every worker maintains its own local copy of the network on CPU and plays
complete games using :class:`~morris_rl.mcts.search.MorrisSearch`.  Completed
:class:`GameRecord` objects are sent to the manager via a shared results queue.

Weight updates are broadcast through per-worker queues.  Workers poll for
updates at the start of each new game, so they always play with weights that
are at most one game stale.

The self-play loop follows the AlphaZero temperature schedule:
  - moves 0 … temperature_threshold-1 : temperature = 1.0  (exploratory)
  - moves temperature_threshold …      : temperature = 1e-6 (near-argmax)

Dirichlet exploration noise is always added at the MCTS root during training.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from morris_rl.utils.logging import logger

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn

from morris_rl.env.rules import (
    MAX_HALFMOVES,
    MAX_TOTAL_HALFMOVES,
    THREEFOLD_LIMIT,
    Outcome,
    apply_action,
    get_legal_actions,
    initial_state,
    is_terminal,
    opponent,
    pieces_on_board,
    random_late_game_state,
)
from morris_rl.env.board import ACTION_SPACE_SIZE, MILLS, NUM_POSITIONS
from morris_rl.network.resnet import MorrisResNet
from morris_rl.training.replay_buffer import SampleRecord

_NUM_PLANES = 7
_ARGMAX_TEMPERATURE = 1e-6


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class GameRecord:
    """Training data produced by one complete self-play game."""

    samples: list[SampleRecord]
    game_length: int            # total half-moves played
    outcome: int                # 1=player1, 2=player2, -1=draw

    # Per-game observability stats (logged by the trainer for diagnostics).
    # Defaults make the field backward-compatible with code that builds
    # GameRecord positionally (older tests).
    mills_p1: int = 0           # mills formed by P1 during the game
    mills_p2: int = 0
    captures_p1: int = 0        # opponent pieces removed by P1
    captures_p2: int = 0
    final_pieces_diff: int = 0  # pieces_p1 - pieces_p2 at game end (signed)
    # Why the game ended. One of:
    #   "pieces_below_3"    — losing player reduced to <3 on board (no hand left)
    #   "no_legal_moves"    — losing player blocked by adjacency
    #   "halfmove_cap"      — drew by MAX_HALFMOVES no-progress clock
    #   "threefold"         — drew by repetition limit
    #   "resign"            — losing player resigned (only with feature 1 active)
    term_reason: str = "unknown"

    # Resign-feature observability (Phase 2a). All zero-init when the feature
    # is disabled — the trainer treats those games as "not eligible" so the
    # resign metrics keep their meaning across mixed runs.
    resign_eligible: bool = False         # threshold ever crossed during this game
    resigned_by_player: int | None = None  # 1/2 if the game ended by resign, else None
    # Verify-mode bookkeeping: when an eligible game is randomly selected
    # (verify_fraction) to be played out instead of resigned, we record who
    # *would* have resigned so the trainer can compare against the real
    # outcome to compute the false-positive rate.
    was_verify_play: bool = False
    verify_resigning_player: int | None = None

    # Playout-cap observability (Phase 2b). Per-game counters used by the
    # trainer to log full vs fast move ratios. Both zero when the feature
    # is disabled (every move counted as "full").
    full_sim_moves: int = 0   # plies that ran the full-sim search
    fast_sim_moves: int = 0   # plies that ran the fast-sim search

    # Curriculum observability (Phase 3). True if this game started from
    # a random late-game position rather than initial_state(). Pieces is
    # the per-side count at start (used to compute the avg start density).
    curriculum_start: bool = False
    curriculum_pieces: int = 0
    # Set to True when discard_timeout_games=True and the game ended by
    # halfmove_cap. The trainer counts discards for the timeout_discard_rate
    # metric but does NOT push these samples to the buffer.
    timeout_discarded: bool = False
    # Full action history (every applied action index in order). Used by the
    # optional trace logger (MORRIS_TRACE_DIR env var) so games can be
    # replayed offline with scripts/replay_game.py. Empty when tracing off.
    actions_history: list[int] = field(default_factory=list)


def _maybe_log_trace(record: "GameRecord", worker_id: int, game: str = "morris") -> None:
    """Append a JSONL trace of *record* to MORRIS_TRACE_DIR when enabled.

    Activation : set env var ``MORRIS_TRACE_DIR=/some/path`` before launching
    training. Each worker writes to its own file ``worker_{id}.jsonl`` (no
    inter-worker contention, no file lock needed).

    Sampling : every game with ``term_reason=='piece_count_tiebreak'`` is
    logged unconditionally (these are the long ones decided by the move-200
    cap — the ones the user wants to inspect). Other games are logged at
    the rate set by ``MORRIS_TRACE_SAMPLE_RATE`` (default 0.02 = 2%).

    Schema (one JSON object per line) :
        { "ts": float, "worker": int, "game": str, "outcome": int,
          "length": int, "term_reason": str, "actions": list[int] }
    """
    trace_dir = os.environ.get("MORRIS_TRACE_DIR")
    if not trace_dir:
        return
    always_log = record.term_reason == "piece_count_tiebreak"
    if not always_log:
        try:
            sample_rate = float(os.environ.get("MORRIS_TRACE_SAMPLE_RATE", "0.02"))
        except ValueError:
            sample_rate = 0.02
        if random.random() >= sample_rate:
            return

    path = Path(trace_dir)
    path.mkdir(parents=True, exist_ok=True)
    target = path / f"worker_{worker_id}.jsonl"
    payload = {
        "ts": time.time(),
        "worker": worker_id,
        "game": game,
        "outcome": record.outcome,
        "length": record.game_length,
        "term_reason": record.term_reason,
        "actions": list(record.actions_history),
    }
    # Best-effort: never crash a training run because the trace dir is full
    # or read-only. Log to logger on first failure per worker would be ideal,
    # but we keep it silent for now to avoid polluting the main log.
    try:
        with target.open("a") as fh:
            fh.write(json.dumps(payload) + "\n")
    except OSError:
        pass


@dataclass
class WorkerError:
    """Sent through the results queue when a worker process crashes."""

    exception: Exception
    worker_id: int


@dataclass(frozen=True)
class ResignConfig:
    """Knobs for the resign-threshold feature passed from cfg to workers.

    A worker creates one of these (or None) at startup; _play_game consults
    it on every move. Disabled by default so existing tests are unaffected.
    """

    enabled: bool = False
    # Root-value (from current player's POV) below which a ply counts as
    # "low". Has to be set on the same scale as the network's value head
    # output — i.e., in [-1, 1].
    threshold: float = -0.90
    # How many consecutive low-value plies are required before a player can
    # resign. Filters out one-off MCTS noise.
    min_consecutive_below: int = 3
    # Don't allow resignation during the placement phase (first ~18 plies)
    # or the very early moving phase — value estimates are too noisy at
    # init and mid-bootstrap.
    min_move_for_resign: int = 30
    # Fraction of "would-resign" games that we instead play out to the
    # natural terminal, so the trainer can compute a post-hoc false-positive
    # rate. Set to 0.0 to disable verification entirely.
    verify_fraction: float = 0.05


@dataclass(frozen=True)
class PlayoutCapConfig:
    """Knobs for KataGo-style playout cap randomization (Phase 2b).

    On each move, the worker picks "full" or "fast" search via Bernoulli:
    full moves run the standard ``num_simulations`` and contribute samples
    to the replay buffer; fast moves use a much smaller sim count and are
    NOT stored (their visit distribution is too noisy to train on). This
    increases games-per-hour at constant policy-target quality.

    The number of full sims reuses the existing ``mcts.num_simulations_train``
    setting — only the fast count is new here.
    """

    enabled: bool = False
    # Probability that a given move uses the full-sim search. KataGo uses
    # ~0.25; the rest run on fast_sim_count.
    full_sim_fraction: float = 0.25
    # Sim count for fast moves. Pick small enough that the speedup matters
    # (≤ 1/3 of full) but large enough that the game still progresses
    # plausibly (typically 30–80 for Morris).
    fast_sim_count: int = 60


@dataclass(frozen=True)
class CurriculumConfig:
    """Knobs for curriculum (random late-game starts) — Phase 3.

    With probability ``random_start_fraction`` per game, the worker starts
    self-play from a randomly drawn mid-game position (both hands empty,
    *pieces_per_player* pieces per side) instead of the canonical empty
    board. This biases the data distribution toward decisive endgames and
    away from the placement-phase draw attractor.

    Crucially, value targets remain those produced by the network's *own*
    play from this start — no external "winning side" label is injected,
    matching the user's "see something learn from scratch" requirement.
    """

    enabled: bool = False
    # Per-game Bernoulli probability of using a random start. 1.0 means
    # every game starts mid-board; 0.0 keeps the canonical loop.
    random_start_fraction: float = 0.5
    # How many pieces per side at the random start. 6 is a reasonable
    # mid-game (each side has lost ~3); lower → closer to terminal.
    pieces_per_player: int = 6


# ---------------------------------------------------------------------------
# Game-play helpers
# ---------------------------------------------------------------------------


def _temperature_for_move(move_number: int, threshold: int) -> float:
    return 1.0 if move_number < threshold else _ARGMAX_TEMPERATURE


_HYBRID_OUTCOME_WEIGHT: float = 0.7
_HYBRID_MARGIN_WEIGHT: float = 0.3
_HYBRID_MARGIN_SCALE: float = 4.0   # tanh(piece_diff / scale) — controls saturation


def _hybrid_value_target(
    outcome: Outcome | None, perspective_player: int, final_pieces_diff: int
) -> float:
    """Continuous value target in [-1, +1] blending outcome sign with margin.

    Pure binary {-1, 0, +1} targets force the network to assign the same
    value to a marginal win (1-piece tiebreak) and a decisive elimination
    (opponent reduced to 2). That ambiguity caps the MSE plancher because
    visually-similar positions have outcomes that differ in magnitude.

    Hybrid = 0.7 * sign(outcome) + 0.3 * tanh(margin / 4)
      win by elimination 8v2 → ~+0.97  (decisive)
      win by tiebreak  5v4   → ~+0.77  (marginal)
      draw                   →   0.00
      loss by tiebreak       → ~-0.77
      loss by elimination    → ~-0.97

    Sign is preserved → optimal policy unchanged (the network still prefers
    winning to losing). Only the magnitude is calibrated to reflect how
    decisively the game was won.
    """
    import math

    if outcome is None or outcome == Outcome.DRAW:
        return 0.0
    sign = 1.0 if int(outcome) == perspective_player else -1.0
    # final_pieces_diff is stored from P1's perspective; flip for P2.
    own_minus_opp = final_pieces_diff if perspective_player == 1 else -final_pieces_diff
    margin = math.tanh(own_minus_opp / _HYBRID_MARGIN_SCALE)
    return _HYBRID_OUTCOME_WEIGHT * sign + _HYBRID_MARGIN_WEIGHT * margin


def _assign_value_targets(
    steps: list[
        tuple[
            npt.NDArray[np.float32],  # encoded state
            npt.NDArray[np.float32],  # policy_target
            int,                       # current player
            npt.NDArray[np.bool_],    # legal_mask
            bool,                      # was this ply a full-sim move?
            float,                     # mill_diff_target (NaN if unavailable)
            float,                     # pieces_diff_target (NaN if unavailable)
        ]
    ],
    outcome: Outcome | None,
    term_reason: str = "unknown",
    final_pieces_diff: int = 0,
) -> list[SampleRecord]:
    """Convert per-step tuples into SampleRecords with hybrid value targets.

    See :func:`_hybrid_value_target` for the value-blending rationale.
    Fast-sim plies (was_full_sim=False, playout cap) are skipped so they
    don't pollute the buffer.
    """
    records: list[SampleRecord] = []
    for encoded, policy, player, mask, was_full_sim, mill_diff, pieces_diff in steps:
        if not was_full_sim:
            continue
        v = _hybrid_value_target(outcome, player, final_pieces_diff)
        records.append(
            SampleRecord(
                encoded_state=encoded,
                policy_target=policy,
                value_target=v,
                legal_mask=mask,
                mill_diff_target=mill_diff,
                pieces_diff_target=pieces_diff,
            )
        )
    return records


def _play_game(
    search: "MorrisSearch",
    temperature_threshold: int = 10,
    resign_config: ResignConfig | None = None,
    rng: np.random.Generator | None = None,
    search_fast: "MorrisSearch | None" = None,
    playout_cap_config: PlayoutCapConfig | None = None,
    curriculum_config: CurriculumConfig | None = None,
    discard_timeout_games: bool = False,
    game_fns: dict | None = None,
) -> GameRecord:
    """Play one complete self-play game and return its training data.

    When ``playout_cap_config.enabled`` and ``search_fast`` is provided,
    each ply chooses between *search* (full sims, recorded in buffer) and
    *search_fast* (fewer sims, not recorded) via Bernoulli on
    ``full_sim_fraction``.

    When ``curriculum_config.enabled``, the game starts from a random
    late-game position with probability ``random_start_fraction``.

    ``game_fns`` overrides the Morris-specific functions for alternate games.
    Recognised keys: ``initial_state``, ``is_terminal``, ``get_legal_actions``,
    ``apply_action``, ``encode_state``, ``random_late_game_state``.
    """
    _fns = game_fns or {}
    _initial_state = _fns.get("initial_state", initial_state)
    _is_terminal = _fns.get("is_terminal", is_terminal)
    _get_legal_actions = _fns.get("get_legal_actions", get_legal_actions)
    _apply_action = _fns.get("apply_action", apply_action)
    if "encode_state" in _fns and _fns["encode_state"] is not None:
        _encode_state = _fns["encode_state"]
    else:
        from morris_rl.mcts.search import encode_state as _encode_state  # Morris default
    _random_late_game = _fns.get("random_late_game_state", random_late_game_state)
    # Optional aux-target function (mill_diff, pieces_diff). When absent (e.g.
    # Reversi or other games without mills), we record NaN and the trainer
    # masks the aux loss for those samples.
    _compute_aux = _fns.get("compute_aux_features")

    if rng is None:
        rng = np.random.default_rng()
    # Curriculum start (Phase 3). Per-game Bernoulli — independent of any
    # per-move randomness inside the search.
    curriculum_start = False
    curriculum_pieces = 0
    if (
        curriculum_config is not None
        and curriculum_config.enabled
        and _random_late_game is not None
        and rng.random() < curriculum_config.random_start_fraction
    ):
        state = _random_late_game(
            rng, pieces_per_player=curriculum_config.pieces_per_player
        )
        # If the helper exhausted retries it falls back to initial_state()
        # — detect that to avoid mis-attributing those games to curriculum.
        if getattr(state, "pieces_in_hand", None) == (0, 0):
            curriculum_start = True
            curriculum_pieces = curriculum_config.pieces_per_player
        else:
            state = _initial_state()
    else:
        state = _initial_state()
    steps: list[
        tuple[
            npt.NDArray[np.float32],   # encoded state
            npt.NDArray[np.float32],   # policy_target (visit_probs)
            int,                        # actor (current player)
            npt.NDArray[np.bool_],     # legal_mask
            bool,                       # full-sim ply? (False = skipped at buffer write)
            float,                      # mill_diff_target (NaN if unavailable)
            float,                      # pieces_diff_target (NaN if unavailable)
        ]
    ] = []
    move_count = 0
    # Full action history for optional trace logging (replayable later via
    # scripts/replay_game.py). Negligible cost (~100 ints per game).
    actions_history: list[int] = []
    # Per-game observability counters. Mills are detected via the
    # not-must_capture → must_capture transition (forming a mill enables a
    # capture); captures are the reverse transition (must_capture → not).
    # Both events are credited to the player who was at the trait BEFORE
    # the action.
    mills_p1 = 0
    mills_p2 = 0
    captures_p1 = 0
    captures_p2 = 0

    # Resign-feature state. Per-player consecutive-low counters (index 0
    # unused; players are 1 and 2). When the threshold is crossed the worker
    # rolls verify_fraction once and either resigns or commits to playing
    # the game out as a "verify" sample (and never resigns again, even if
    # the threshold gets re-crossed by the same or other player).
    resign_active = resign_config is not None and resign_config.enabled
    consec_below = [0, 0, 0]
    resign_eligible = False
    resigned_by_player: int | None = None
    was_verify_play = False
    verify_resigning_player: int | None = None

    # Playout-cap state. When the feature is enabled and a fast search was
    # provided, each ply independently tosses a Bernoulli(full_sim_fraction).
    # When disabled (or no fast search), every ply runs the full search.
    cap_active = (
        playout_cap_config is not None
        and playout_cap_config.enabled
        and search_fast is not None
    )
    full_sim_moves = 0
    fast_sim_moves = 0

    # Action-space size for the legal mask — taken from game_fns if provided,
    # else falls back to the Morris constant from board.py.
    _action_space_n = _fns.get("action_space_size", ACTION_SPACE_SIZE)

    while True:
        done, _ = _is_terminal(state)
        if done:
            break

        temp = _temperature_for_move(move_count, temperature_threshold)
        encoded = _encode_state(state).squeeze(0).numpy().copy()
        # Snapshot the legal mask BEFORE applying the chosen action — we want
        # the mask of the state the policy_target was computed for.
        legal_mask = np.zeros(_action_space_n, dtype=np.bool_)
        legal_mask[_get_legal_actions(state)] = True

        # Resign signal: query the network's value estimate for the current
        # player at the root, BEFORE running MCTS. Cheap (one extra forward
        # per move) and avoids patching ctree to expose the post-search Q.
        # must_capture is Morris-specific; other games never trigger this branch.
        if resign_active and not getattr(state, "must_capture", False):
            root_v = search.root_value(state)
            actor_now = state.current_player
            if root_v < resign_config.threshold:
                consec_below[actor_now] += 1
            else:
                consec_below[actor_now] = 0

            if (
                consec_below[actor_now] >= resign_config.min_consecutive_below
                and move_count >= resign_config.min_move_for_resign
                and not was_verify_play
                and resigned_by_player is None
            ):
                resign_eligible = True
                # Bernoulli(verify_fraction) — in verify_fraction of cases we
                # commit to playing the game out, locking in verify mode.
                if rng.random() < resign_config.verify_fraction:
                    was_verify_play = True
                    verify_resigning_player = actor_now
                else:
                    resigned_by_player = actor_now
                    break  # game ends here, value targets handled below

        # Pick search instance for this ply. Forced captures (Morris-only) use
        # the full search regardless; for other games must_capture is always
        # False so the cap logic applies normally.
        if cap_active and not getattr(state, "must_capture", False):
            is_full = rng.random() < playout_cap_config.full_sim_fraction
        else:
            is_full = True
        active_search = search if is_full else search_fast
        if is_full:
            full_sim_moves += 1
        else:
            fast_sim_moves += 1

        action, visit_probs = active_search.run(
            state, temperature=temp, add_noise=True
        )

        # Capture stats around the action: state.must_capture flips are the
        # cleanest signal of mill / capture events (Morris-specific; zero-impact
        # for other games since the attribute is absent).
        was_must_capture = getattr(state, "must_capture", False)
        actor = state.current_player
        # Compute aux targets BEFORE applying the action — they describe the
        # state from which the policy_target was derived.
        if _compute_aux is not None:
            mill_diff_t, pieces_diff_t = _compute_aux(state)
        else:
            mill_diff_t, pieces_diff_t = float("nan"), float("nan")
        state = _apply_action(state, action)
        was_capture = was_must_capture and not getattr(state, "must_capture", False)
        actions_history.append(int(action))

        steps.append((encoded, visit_probs, actor, legal_mask, is_full, mill_diff_t, pieces_diff_t))

        if not was_must_capture and getattr(state, "must_capture", False):
            # The just-played placement/move formed a mill (forced capture next).
            if actor == 1:
                mills_p1 += 1
            else:
                mills_p2 += 1
        elif was_capture:
            if actor == 1:
                captures_p1 += 1
            else:
                captures_p2 += 1
        move_count += 1

    if resigned_by_player is not None:
        # Resignation = forfait: opponent wins, no natural is_terminal payload.
        outcome = Outcome(opponent(resigned_by_player))
        term_reason = "resign"
    else:
        _, outcome = _is_terminal(state)
        term_reason = _detect_term_reason(state, outcome, get_legal_actions_fn=_get_legal_actions)
    final_pieces_p1 = pieces_on_board(state.board, 1)
    final_pieces_p2 = pieces_on_board(state.board, 2)
    final_pieces_diff = final_pieces_p1 - final_pieces_p2

    # Halfmove-cap games produce value=0 on non-draw positions → draw attractor
    # fuel. When the flag is set, drop their samples from the buffer entirely.
    timeout_discarded = discard_timeout_games and term_reason == "halfmove_cap"
    samples = (
        []
        if timeout_discarded
        else _assign_value_targets(steps, outcome, term_reason, final_pieces_diff)
    )
    outcome_int = -1 if (outcome is None or outcome == Outcome.DRAW) else int(outcome)
    return GameRecord(
        samples=samples,
        game_length=move_count,
        outcome=outcome_int,
        mills_p1=mills_p1,
        mills_p2=mills_p2,
        captures_p1=captures_p1,
        captures_p2=captures_p2,
        final_pieces_diff=final_pieces_diff,
        term_reason=term_reason,
        resign_eligible=resign_eligible,
        resigned_by_player=resigned_by_player,
        was_verify_play=was_verify_play,
        verify_resigning_player=verify_resigning_player,
        full_sim_moves=full_sim_moves,
        fast_sim_moves=fast_sim_moves,
        curriculum_start=curriculum_start,
        curriculum_pieces=curriculum_pieces,
        timeout_discarded=timeout_discarded,
        actions_history=actions_history,
    )


def _detect_term_reason(
    state: Any,
    outcome: Outcome | None,
    get_legal_actions_fn=None,
) -> str:
    """Identify why the game just terminated, mirroring is_terminal()'s order.

    ``get_legal_actions_fn`` allows callers to pass a game-specific function;
    defaults to the Morris ``get_legal_actions`` import when omitted.
    """
    _legal = get_legal_actions_fn or get_legal_actions
    # Reversi: detected by presence of pass_count attribute (not Morris).
    pass_count = getattr(state, "pass_count", None)
    if pass_count is not None:
        import numpy as np
        empty = int(np.sum(state.board == 0))
        return "board_full" if empty == 0 else "double_pass"
    # Morris: total halfmove cap → piece-count tiebreak (checked before threefold).
    total = getattr(state, "total_halfmoves", 0)
    if total >= MAX_TOTAL_HALFMOVES:
        return "piece_count_tiebreak"
    # Threefold repetition also resolved by piece-count tiebreak.
    pos_counts = getattr(state, "position_counts", {})
    if pos_counts and max(pos_counts.values()) >= THREEFOLD_LIMIT:
        return "piece_count_tiebreak"
    halfmove = getattr(state, "halfmove_clock", 0)
    if halfmove >= MAX_HALFMOVES:
        return "piece_count_tiebreak"
    hand = getattr(state, "pieces_in_hand", None)
    if hand is not None:
        player = state.current_player
        if hand[player - 1] == 0 and pieces_on_board(state.board, player) < 3:
            return "pieces_below_3"
    if not _legal(state):
        return "no_legal_moves"
    return "unknown"


# ---------------------------------------------------------------------------
# Network reconstruction (used inside worker processes)
# ---------------------------------------------------------------------------


def _build_worker_network(cfg: dict[str, Any]) -> MorrisResNet:
    return MorrisResNet(
        num_blocks=cfg["num_blocks"],
        num_channels=cfg["num_channels"],
        num_planes=cfg.get("num_planes", _NUM_PLANES),
        policy_head_hidden=cfg["policy_head_hidden"],
        value_head_hidden=cfg["value_head_hidden"],
        value_head_type=cfg.get("value_head_type", "scalar"),
        num_positions=cfg.get("num_positions", NUM_POSITIONS),
        action_space_size=cfg.get("action_space_size", ACTION_SPACE_SIZE),
        aux_heads_enabled=bool(cfg.get("aux_heads_enabled", False)),
        aux_head_hidden=int(cfg.get("aux_head_hidden", 64)),
    )


# ---------------------------------------------------------------------------
# Worker process entry point
# ---------------------------------------------------------------------------


def _should_recycle(
    worker_id: int,
    games_played: int,
    proc: Any,
    max_rss_mb: int,
    recycle_games: int,
) -> bool:
    """Return True when the worker should exit cleanly so the manager respawns
    it from a fresh Python interpreter — caps the impact of any growing RSS
    (suspected ctree leaf accumulation in early training).
    """
    if recycle_games > 0 and games_played >= recycle_games:
        from loguru import logger as _log
        _log.info(f"worker {worker_id}: recycling after {games_played} games")
        return True
    if max_rss_mb > 0 and games_played > 0 and games_played % 5 == 0:
        rss_mb = proc.memory_info().rss / 1e6
        if rss_mb > max_rss_mb:
            from loguru import logger as _log
            _log.warning(
                f"worker {worker_id}: rss={rss_mb:.0f}MB > {max_rss_mb}MB "
                f"after {games_played} games, recycling"
            )
            return True
    return False


def _worker_fn(
    worker_id: int,
    network_cfg: dict[str, Any],
    weights_queue: mp.Queue,  # type: ignore[type-arg]
    results_queue: mp.Queue,  # type: ignore[type-arg]
    num_simulations: int,
    temperature_threshold: int,
    dirichlet_alpha: float,
    dirichlet_epsilon: float,
    seed: int,
    worker_max_rss_mb: int = 0,
    worker_recycle_games: int = 0,
    resign_config: ResignConfig | None = None,
    playout_cap_config: PlayoutCapConfig | None = None,
    curriculum_config: CurriculumConfig | None = None,
    discard_timeout_games: bool = False,
    game_name: str = "morris",
) -> None:
    """Worker process: play self-play games until a None sentinel is received."""
    import random
    import sys
    import warnings

    # Silence third-party startup noise before importing lzero:
    #   - ding loguru warnings (numba, pyecharts): filtered by our loguru handler
    #   - gym "unmaintained" message: a raw print() to sys.stderr — must redirect
    #     the *name* sys.stderr to /dev/null (loguru stores the file object and is
    #     unaffected, so our filtered handler still works on the real fd).
    import os
    warnings.filterwarnings("ignore")
    _old_stderr = sys.stderr
    from loguru import logger as _log
    _log.remove()
    _log.add(_old_stderr, level="INFO", filter=lambda r: r["name"].startswith("morris_rl"))

    _devnull = open(os.devnull, "w")
    sys.stderr = _devnull
    try:
        from morris_rl.mcts.search import MorrisSearch
    finally:
        sys.stderr = _old_stderr
        _devnull.close()

    # Build game-specific function table for the selected game.
    if game_name == "reversi":
        from morris_rl.env.reversi.rules import (
            initial_state as _r_initial_state,
            get_legal_actions as _r_get_legal_actions,
            apply_action as _r_apply_action,
            is_terminal as _r_is_terminal,
        )
        from morris_rl.env.reversi.encoding import encode_state as _r_encode_state
        from morris_rl.env.reversi.board import ACTION_SPACE_SIZE as _r_action_space_size
        _game_fns: dict = {
            "initial_state": _r_initial_state,
            "get_legal_actions": _r_get_legal_actions,
            "apply_action": _r_apply_action,
            "is_terminal": _r_is_terminal,
            "encode_state": _r_encode_state,
            "action_space_size": _r_action_space_size,
            "random_late_game_state": None,
        }
    else:  # morris (default)
        # Aux head targets are Morris-specific (mill_diff, pieces_diff). Wire
        # the helper here so _play_game can compute them per ply. For other
        # games this stays unset → NaN aux targets → aux loss masked out.
        from morris_rl.env.rules import compute_aux_features as _morris_aux
        _game_fns = {"compute_aux_features": _morris_aux}

    # Each worker is one MCTS pipeline; using torch's default (= all CPU cores)
    # means N workers fight over N×cores threads. One thread per worker keeps the
    # CPU cleanly partitioned and is faster for small networks (1M params).
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already set or torch already used parallel ops

    worker_seed = seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    network = _build_worker_network(network_cfg)

    # Block until initial weights arrive.
    weights = weights_queue.get()
    if weights is None:
        return
    network.load_state_dict(weights)
    network.eval()

    search = MorrisSearch(
        network,
        torch.device("cpu"),
        num_simulations=num_simulations,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
        game_fns=_game_fns,
    )
    # Build the fast-sim companion search only when the feature is on.
    # Two ctree instances cohabit fine; the second adds ~30 MB RSS.
    search_fast: MorrisSearch | None = None
    if playout_cap_config is not None and playout_cap_config.enabled:
        search_fast = MorrisSearch(
            network,
            torch.device("cpu"),
            num_simulations=playout_cap_config.fast_sim_count,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            game_fns=_game_fns,
        )

    import psutil
    proc = psutil.Process()
    games_played = 0
    rng = np.random.default_rng(worker_seed)

    while True:
        # Non-blocking check for updated weights or shutdown.
        try:
            update = weights_queue.get_nowait()
            if update is None:
                return
            network.load_state_dict(update)
            network.eval()
        except Exception:
            pass

        try:
            game = _play_game(
                search,
                temperature_threshold=temperature_threshold,
                resign_config=resign_config,
                rng=rng,
                search_fast=search_fast,
                playout_cap_config=playout_cap_config,
                curriculum_config=curriculum_config,
                discard_timeout_games=discard_timeout_games,
                game_fns=_game_fns,
            )
            _maybe_log_trace(game, worker_id, game=game_name)
            results_queue.put(game)
            games_played += 1
        except Exception as exc:
            results_queue.put(WorkerError(exception=exc, worker_id=worker_id))
            return  # Worker shuts down; main process will see the error immediately.

        if _should_recycle(
            worker_id, games_played, proc, worker_max_rss_mb, worker_recycle_games
        ):
            return


# ---------------------------------------------------------------------------
# Remote-eval worker (shared GPU inference server)
# ---------------------------------------------------------------------------


def _worker_fn_remote(
    worker_id: int,
    req_send_conn: Any,     # mpc.Connection — worker's send end of req pipe
    reply_recv_conn: Any,   # mpc.Connection — worker's recv end of reply pipe
    results_queue: mp.Queue,  # type: ignore[type-arg]
    shutdown_event: Any,    # mp.Event — set when manager.stop() is called
    shm_names: Any,
    num_workers: int,
    num_simulations: int,
    temperature_threshold: int,
    dirichlet_alpha: float,
    dirichlet_epsilon: float,
    seed: int,
    worker_max_rss_mb: int = 0,
    worker_recycle_games: int = 0,
    resign_config: ResignConfig | None = None,
    playout_cap_config: PlayoutCapConfig | None = None,
    curriculum_config: CurriculumConfig | None = None,
    discard_timeout_games: bool = False,
) -> None:
    """Worker process: delegates leaf evaluation to the inference server.

    Same self-play loop as :func:`_worker_fn` but evaluation goes through a
    request/reply queue pair connected to a centralized GPU server instead of
    running torch in-process.
    """
    import random
    import sys
    import warnings

    import os
    warnings.filterwarnings("ignore")
    _old_stderr = sys.stderr
    from loguru import logger as _log
    _log.remove()
    _log.add(_old_stderr, level="INFO", filter=lambda r: r["name"].startswith("morris_rl"))

    _devnull = open(os.devnull, "w")
    sys.stderr = _devnull
    try:
        from morris_rl.env.rules import get_legal_actions
        from morris_rl.mcts.search import MorrisSearch, encode_state
        from morris_rl.training.inference_server import make_remote_eval_fn
    finally:
        sys.stderr = _old_stderr
        _devnull.close()

    # Workers don't run torch in-process here, but other libs may still query
    # CPU thread defaults (e.g. numpy/BLAS); pin to one to avoid contention
    # with the inference server's host-side dispatch threads.
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    worker_seed = seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    eval_fn = make_remote_eval_fn(
        worker_id=worker_id,
        req_send_conn=req_send_conn,
        reply_recv_conn=reply_recv_conn,
        shm_names=shm_names,
        num_workers=num_workers,
        encode_state=encode_state,
        get_legal_actions=get_legal_actions,
    )
    search = MorrisSearch(
        eval_fn=eval_fn,
        num_simulations=num_simulations,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_epsilon=dirichlet_epsilon,
    )
    # Same eval_fn (shared GPU server) drives both searches; only the sim
    # budget differs. Cheap to instantiate — one extra ctree object.
    search_fast: MorrisSearch | None = None
    if playout_cap_config is not None and playout_cap_config.enabled:
        search_fast = MorrisSearch(
            eval_fn=eval_fn,
            num_simulations=playout_cap_config.fast_sim_count,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
        )

    import psutil
    proc = psutil.Process()
    games_played = 0
    rng = np.random.default_rng(worker_seed)

    while not shutdown_event.is_set():
        try:
            game = _play_game(
                search,
                temperature_threshold=temperature_threshold,
                resign_config=resign_config,
                rng=rng,
                search_fast=search_fast,
                playout_cap_config=playout_cap_config,
                curriculum_config=curriculum_config,
                discard_timeout_games=discard_timeout_games,
            )
            _maybe_log_trace(game, worker_id)
            results_queue.put(game)
            games_played += 1
        except Exception as exc:
            results_queue.put(WorkerError(exception=exc, worker_id=worker_id))
            return

        if _should_recycle(
            worker_id, games_played, proc, worker_max_rss_mb, worker_recycle_games
        ):
            return


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class SelfPlayManager:
    """Manages a pool of self-play worker processes.

    Workers produce :class:`GameRecord` objects which the caller collects via
    :meth:`collect_game`.  After each training step, call
    :meth:`update_network` to broadcast fresh weights.

    Args:
        network: The current policy/value network (used to seed worker weights).
        network_cfg: Plain-dict description of the network architecture, passed
            to workers for reconstruction (e.g.
            ``{"num_blocks": 10, "num_channels": 128, ...}``).
        num_workers: Number of parallel worker processes.
        num_simulations: MCTS simulations per move.
        temperature_threshold: Use temperature=1.0 for the first this many
            moves, then switch to near-argmax.
        dirichlet_alpha: Dirichlet concentration for root exploration noise.
        dirichlet_epsilon: Weight of Dirichlet noise mixed into root priors.
        seed: Base random seed; worker i uses seed + i.
    """

    def __init__(
        self,
        network: nn.Module,
        network_cfg: dict[str, Any],
        num_workers: int = 12,
        num_simulations: int = 200,
        temperature_threshold: int = 10,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        seed: int = 42,
        inference_mode: str = "per_worker_cpu",
        inference_device: str = "cuda",
        max_batch_size: int = 32,
        max_wait_ms: float = 5.0,
        log_file: str | None = None,
        worker_max_rss_mb: int = 0,
        worker_recycle_games: int = 0,
        watcher_interval_s: float = 5.0,
        resign_config: ResignConfig | None = None,
        playout_cap_config: PlayoutCapConfig | None = None,
        curriculum_config: CurriculumConfig | None = None,
        discard_timeout_games: bool = False,
        game_name: str = "morris",
    ) -> None:
        if inference_mode not in ("per_worker_cpu", "shared_gpu"):
            raise ValueError(f"unknown inference_mode {inference_mode!r}")
        self._network = network
        self._network_cfg = network_cfg
        self._num_workers = num_workers
        self._num_simulations = num_simulations
        self._temperature_threshold = temperature_threshold
        self._dirichlet_alpha = dirichlet_alpha
        self._dirichlet_epsilon = dirichlet_epsilon
        self._seed = seed
        self._inference_mode = inference_mode
        self._inference_device = inference_device
        self._max_batch_size = max_batch_size
        self._max_wait_ms = max_wait_ms
        self._log_file = log_file
        self._worker_max_rss_mb = worker_max_rss_mb
        self._worker_recycle_games = worker_recycle_games
        self._watcher_interval_s = watcher_interval_s
        self._resign_config = resign_config
        self._playout_cap_config = playout_cap_config
        self._curriculum_config = curriculum_config
        self._discard_timeout_games = discard_timeout_games
        self._game_name = game_name

        self._ctx = mp.get_context("spawn")
        self._results_queue: mp.Queue = self._ctx.Queue()  # type: ignore[type-arg]
        # Per-worker weights queue is only used in per_worker_cpu mode; in
        # shared_gpu mode the inference server owns the network and gets a
        # single dedicated weights channel via InferenceServer.update_weights.
        self._weights_queues: list[mp.Queue] = [  # type: ignore[type-arg]
            self._ctx.Queue(maxsize=1) for _ in range(num_workers)
        ]
        self._shutdown_event: Any = self._ctx.Event()
        self._inference_server: Any = None
        # Slot per worker — None until first start, then the live Process. The
        # watcher thread mutates this list when respawning, hence the lock.
        self._processes: list[Any] = [None] * num_workers
        self._processes_lock = threading.Lock()
        self._respawn_count: list[int] = [0] * num_workers
        self._watcher_thread: threading.Thread | None = None
        self._watcher_stop = threading.Event()
        self._running = False

    def start(self) -> None:
        """Spawn worker processes (and inference server if shared_gpu)."""
        if self._running:
            return
        if self._inference_mode == "shared_gpu":
            from morris_rl.training.inference_server import InferenceServer

            self._inference_server = InferenceServer(
                network=self._network,
                network_cfg=self._network_cfg,
                num_workers=self._num_workers,
                device=self._inference_device,
                max_batch=self._max_batch_size,
                max_wait_ms=self._max_wait_ms,
                log_file=self._log_file,
            )
            self._inference_server.start()

        for i in range(self._num_workers):
            self._spawn_worker(i)

        self._running = True
        # Recycling watcher only spins up if at least one trigger is enabled.
        if self._worker_max_rss_mb > 0 or self._worker_recycle_games > 0:
            self._watcher_thread = threading.Thread(
                target=self._watch_workers, daemon=True, name="self-play-watcher"
            )
            self._watcher_thread.start()

    def _spawn_worker(self, i: int) -> None:
        """(Re)spawn worker *i*. Used at start and on respawn after exit.

        Pipes (shared_gpu) and weights queues (per_worker_cpu) are owned by the
        manager, so they survive a worker restart and are reused as-is.
        """
        if self._inference_mode == "shared_gpu":
            assert self._inference_server is not None
            req_send, reply_recv = self._inference_server.worker_pipes(i)
            p = self._ctx.Process(
                target=_worker_fn_remote,
                args=(
                    i,
                    req_send,
                    reply_recv,
                    self._results_queue,
                    self._shutdown_event,
                    self._inference_server.shm_names,
                    self._inference_server.num_workers,
                    self._num_simulations,
                    self._temperature_threshold,
                    self._dirichlet_alpha,
                    self._dirichlet_epsilon,
                    # Reseed each respawn so we don't replay the same sequence.
                    self._seed + 1000 * (self._respawn_count[i] + 1),
                    self._worker_max_rss_mb,
                    self._worker_recycle_games,
                    self._resign_config,
                    self._playout_cap_config,
                    self._curriculum_config,
                    self._discard_timeout_games,
                ),
                daemon=True,
            )
            p.start()
        else:
            p = self._ctx.Process(
                target=_worker_fn,
                args=(
                    i,
                    self._network_cfg,
                    self._weights_queues[i],
                    self._results_queue,
                    self._num_simulations,
                    self._temperature_threshold,
                    self._dirichlet_alpha,
                    self._dirichlet_epsilon,
                    self._seed + 1000 * (self._respawn_count[i] + 1),
                    self._worker_max_rss_mb,
                    self._worker_recycle_games,
                    self._resign_config,
                    self._playout_cap_config,
                    self._curriculum_config,
                    self._discard_timeout_games,
                    self._game_name,
                ),
                daemon=True,
            )
            p.start()
            # Per-worker-cpu workers block on weights_queue.get() at startup,
            # so push the latest weights right after spawn.
            cpu_weights = {k: v.cpu() for k, v in self._network.state_dict().items()}
            try:
                # Drain any stale entry first (1-slot queue).
                self._weights_queues[i].get_nowait()
            except Exception:
                pass
            self._weights_queues[i].put(cpu_weights)

        with self._processes_lock:
            self._processes[i] = p

    def _watch_workers(self) -> None:
        """Background thread: detect dead workers and respawn them.

        Runs every ``watcher_interval_s``. Workers exit cleanly on RSS or
        game-count triggers (see :func:`_should_recycle`); this thread reacts
        to that exit by starting a fresh process on the same slot — same pipes,
        same shared-memory slot, just a brand-new Python interpreter (no leaked
        state from ctree, torch, numpy, etc.).
        """
        while not self._watcher_stop.is_set():
            with self._processes_lock:
                snapshot = list(enumerate(self._processes))
            for i, p in snapshot:
                if self._watcher_stop.is_set():
                    return
                if p is None or p.is_alive():
                    continue
                exit_code = p.exitcode
                self._respawn_count[i] += 1
                logger.info(
                    "worker {} exited (code={}); respawning (#{}, total respawns={})",
                    i,
                    exit_code,
                    self._respawn_count[i],
                    sum(self._respawn_count),
                )
                try:
                    self._spawn_worker(i)
                except Exception as exc:
                    logger.exception(f"failed to respawn worker {i}: {exc}")
            self._watcher_stop.wait(self._watcher_interval_s)

    def collect_game(self, timeout: float = 300.0) -> GameRecord:
        """Block until one completed game is available and return it.

        Args:
            timeout: Seconds to wait before raising queue.Empty.

        Raises:
            RuntimeError: If a worker process crashed (surfaces the original exception).
            queue.Empty:  If no game arrives within *timeout* seconds.
        """
        result = self._results_queue.get(timeout=timeout)
        if isinstance(result, WorkerError):
            raise RuntimeError(
                f"Self-play worker {result.worker_id} crashed: {result.exception}"
            ) from result.exception
        return result  # type: ignore[return-value]

    def update_network(self, state_dict: dict[str, Any]) -> None:
        """Broadcast updated weights to whoever needs them.

        - per_worker_cpu : push to each worker's bounded weights queue.
        - shared_gpu     : push only to the inference server; workers don't
                           own a network in this mode.
        """
        if self._inference_mode == "shared_gpu":
            if self._inference_server is not None:
                self._inference_server.update_weights(state_dict)
            return
        cpu_weights = {k: v.cpu() for k, v in state_dict.items()}
        for q in self._weights_queues:
            try:
                q.get_nowait()
            except Exception:
                pass
            try:
                q.put_nowait(cpu_weights)
            except Exception:
                pass

    def results_qsize(self) -> int:
        """Approximate size of the results queue (best-effort on macOS/Linux)."""
        try:
            return self._results_queue.qsize()
        except NotImplementedError:
            return -1

    def weights_qsize_max(self) -> int:
        """Largest weight queue across workers (should stay ≤ 1)."""
        sizes = []
        for q in self._weights_queues:
            try:
                sizes.append(q.qsize())
            except NotImplementedError:
                return -1
        return max(sizes) if sizes else 0

    def stop(self) -> None:
        """Send shutdown sentinels and join all worker processes."""
        if not self._running:
            return
        # Stop the watcher first — otherwise it would race with shutdown and
        # cheerfully respawn any worker we just told to exit.
        self._watcher_stop.set()
        if self._watcher_thread is not None:
            self._watcher_thread.join(timeout=2)
            self._watcher_thread = None
        if self._inference_mode == "shared_gpu":
            self._shutdown_event.set()
            if self._inference_server is not None:
                self._inference_server.stop()
        else:
            for q in self._weights_queues:
                q.put(None)
        with self._processes_lock:
            procs = [p for p in self._processes if p is not None]
            self._processes = [None] * self._num_workers
        for p in procs:
            p.join(timeout=15)
            if p.is_alive():
                p.terminate()
        self._running = False

    def __enter__(self) -> SelfPlayManager:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
