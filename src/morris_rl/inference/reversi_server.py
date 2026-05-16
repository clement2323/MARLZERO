"""FastAPI inference server for the Reversi web demo.

Endpoints
---------
GET  /health      — liveness check
GET  /new-game    — return the initial board state
GET  /agents      — list available agents
POST /play        — given action history, return the agent's next move + analysis
POST /state       — replay action history and return board state (no agent)

State is fully reconstructed server-side from the flat action history sent by
the client, so the server is entirely stateless.

Startup configuration (environment variables)
--------------------------------------------
MODEL_CHECKPOINT  Path to a ``.pt`` checkpoint. If unset, auto-detects the
                  latest checkpoint under ``outputs/**/checkpoints/checkpoint_*.pt``.
NUM_SIMULATIONS   MCTS simulations per move (default: 200).
DEVICE            ``cpu`` or ``cuda`` (default: ``cpu``).
"""

from __future__ import annotations

import os
import random
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from morris_rl.env.reversi.board import ACTION_SPACE_SIZE, NUM_POSITIONS
from morris_rl.env.reversi.encoding import encode_state
from morris_rl.env.reversi.rules import (
    PASS_ACTION,
    PLAYER_1,
    GameState,
    Outcome,
    apply_action,
    get_legal_actions,
    initial_state,
    is_terminal,
    opponent,
)
from morris_rl.utils.logging import logger

# ---------------------------------------------------------------------------
# Module-level singletons (set at startup)
# ---------------------------------------------------------------------------

_network: nn.Module | None = None
_search: Any | None = None   # MorrisSearch | None
_device: torch.device = torch.device("cpu")
_num_simulations: int = 200
_checkpoint_label: str | None = None
_default_agent_id: str = "random"

_AGENT_CHECKPOINT = "checkpoint"
_AGENT_GREEDY = "greedy"
_AGENT_RANDOM = "random"

_COL_LABELS = "abcdefgh"


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class PlayRequest(BaseModel):
    actions: list[int]
    agent: str | None = None


class MoveInfo(BaseModel):
    action: int
    visit_prob: float
    description: str


class AgentOption(BaseModel):
    id: str
    label: str
    available: bool


class AgentsResponse(BaseModel):
    options: list[AgentOption]
    default: str


class BoardState(BaseModel):
    board: list[int]              # 64 ints: 0=empty, 1=black, 2=white
    current_player: int           # 1 or 2
    game_over: bool
    winner: int | None            # 1, 2, 0 (draw), or None
    legal_actions: list[int]      # valid action indices for current player
    pass_count: int


class PlayResponse(BaseModel):
    action: int
    description: str
    board_after: BoardState
    top_moves: list[MoveInfo]
    value_estimate: float         # [-1, 1] from Black's POV
    using_network: bool
    agent_name: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pos_label(pos: int) -> str:
    """Convert flat position index to 'a1'-style label (column + row)."""
    row, col = divmod(pos, 8)
    return f"{_COL_LABELS[col]}{row + 1}"


def _describe_action(action: int) -> str:
    if action == PASS_ACTION:
        return "pass"
    return _pos_label(action)


def _find_latest_checkpoint() -> str | None:
    """Scan outputs/**/checkpoints/ for checkpoint_*.pt, return the latest by name."""
    root = Path(__file__).parent.parent.parent.parent.parent / "outputs"
    if not root.exists():
        return None
    candidates = sorted(root.glob("**/checkpoints/checkpoint_*.pt"))
    return str(candidates[-1]) if candidates else None


def _state_to_board(state: GameState, game_over: bool, winner: int | None) -> BoardState:
    legal = [] if game_over else get_legal_actions(state)
    return BoardState(
        board=state.board.tolist(),
        current_player=int(state.current_player),
        game_over=game_over,
        winner=winner,
        legal_actions=legal,
        pass_count=int(state.pass_count),
    )


def _reconstruct_state(actions: list[int]) -> GameState:
    """Replay the action list from the initial state."""
    state = initial_state()
    for i, action in enumerate(actions):
        legal = get_legal_actions(state)
        if action not in legal:
            raise HTTPException(
                status_code=400,
                detail=f"Illegal action {action} at step {i}. Legal: {legal}",
            )
        state = apply_action(state, action)
    return state


def _outcome_to_winner(outcome: Outcome | None) -> int | None:
    if outcome is None:
        return None
    if outcome == Outcome.DRAW:
        return 0
    return int(outcome)  # PLAYER_1_WINS=1, PLAYER_2_WINS=2


# ---------------------------------------------------------------------------
# Agent implementations
# ---------------------------------------------------------------------------


def _greedy_action(state: GameState) -> tuple[int, list[MoveInfo]]:
    """Pick the move that flips the most opponent pieces."""
    from morris_rl.env.reversi.rules import _get_flips  # type: ignore[attr-defined]

    legal = get_legal_actions(state)
    if legal == [PASS_ACTION]:
        return PASS_ACTION, [MoveInfo(action=PASS_ACTION, visit_prob=1.0, description="pass")]

    best_action = legal[0]
    best_count = -1
    for a in legal:
        flips = _get_flips(state.board, a, state.current_player)
        if len(flips) > best_count:
            best_count = len(flips)
            best_action = a

    top_moves = [
        MoveInfo(action=best_action, visit_prob=1.0, description=_describe_action(best_action))
    ]
    return best_action, top_moves


def _random_action(state: GameState) -> tuple[int, list[MoveInfo]]:
    legal = get_legal_actions(state)
    action = int(random.choice(legal))
    top_moves = [MoveInfo(action=action, visit_prob=1.0, description=_describe_action(action))]
    return action, top_moves


def _network_action(
    state: GameState,
) -> tuple[int, list[MoveInfo], float]:
    """Run MCTS search and return (action, top_moves, value_from_black_pov)."""
    assert _search is not None

    action_int, visit_probs = _search.run(state, temperature=1e-6, add_noise=False)
    action_int = int(action_int)

    # visit_probs has shape (ACTION_SPACE_SIZE,). Extract top-5 non-pass moves.
    indexed = sorted(
        ((i, float(visit_probs[i])) for i in range(ACTION_SPACE_SIZE) if i != PASS_ACTION and visit_probs[i] > 0),
        key=lambda x: x[1],
        reverse=True,
    )[:5]
    top_moves = [
        MoveInfo(action=a, visit_prob=p, description=_describe_action(a))
        for a, p in indexed
    ]

    # Raw value is from current player's POV; convert to Black's POV.
    _, raw_value = _search._eval_fn(state)
    value_black_pov = float(raw_value) if state.current_player == PLAYER_1 else -float(raw_value)

    return action_int, top_moves, value_black_pov


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


def _load_network() -> None:
    global _network, _search, _device, _num_simulations, _checkpoint_label, _default_agent_id

    _device = torch.device(os.getenv("DEVICE", "cpu"))
    _num_simulations = int(os.getenv("NUM_SIMULATIONS", "200"))

    checkpoint_path = os.getenv("MODEL_CHECKPOINT", "") or _find_latest_checkpoint() or ""

    if checkpoint_path and Path(checkpoint_path).exists():
        try:
            from morris_rl.mcts.search import MorrisSearch
            from morris_rl.network.resnet import MorrisResNet

            payload = torch.load(checkpoint_path, map_location=_device, weights_only=False)
            state_dict = payload["state_dict"]

            # Infer architecture from checkpoint weights (same logic as play_reversi.py)
            input_conv_w = state_dict["input_conv.weight"]
            num_channels = input_conv_w.shape[0]
            num_planes = input_conv_w.shape[1]
            action_space_size = state_dict["policy_head.fc2.weight"].shape[0]
            num_positions_ckpt = state_dict["value_head.fc1.weight"].shape[1]
            policy_head_hidden = state_dict["policy_head.fc2.weight"].shape[1]
            value_head_hidden = state_dict["value_head.fc2.weight"].shape[1]
            num_blocks = sum(
                1 for k in state_dict if k.startswith("trunk.") and k.endswith(".conv1.weight")
            )

            network = MorrisResNet(
                num_blocks=num_blocks,
                num_channels=num_channels,
                num_planes=num_planes,
                policy_head_hidden=policy_head_hidden,
                value_head_hidden=value_head_hidden,
                num_positions=num_positions_ckpt,
                action_space_size=action_space_size,
            ).to(_device)
            network.load_state_dict(state_dict)
            network.eval()
            _network = network

            reversi_fns = {
                "initial_state": initial_state,
                "get_legal_actions": get_legal_actions,
                "apply_action": apply_action,
                "is_terminal": is_terminal,
                "encode_state": encode_state,
                "action_space_size": action_space_size,
            }
            _search = MorrisSearch(
                network,
                _device,
                num_simulations=_num_simulations,
                game_fns=reversi_fns,
            )

            step = payload.get("step", "?")
            _checkpoint_label = (
                f"ResNet{num_blocks}×{num_channels} step={step} ({_num_simulations} sims)"
            )
            _default_agent_id = _AGENT_CHECKPOINT
            logger.info(f"Loaded Reversi checkpoint: {checkpoint_path} (step {step})")
        except Exception as exc:
            logger.warning(f"Failed to load checkpoint {checkpoint_path}: {exc}")
            _network = None
            _search = None
            _checkpoint_label = None
            _default_agent_id = _AGENT_RANDOM
    else:
        _network = None
        _search = None
        _checkpoint_label = None
        _default_agent_id = _AGENT_RANDOM
        logger.info("No Reversi checkpoint found — defaulting to random agent")


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    _load_network()
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Reversi RL Demo",
    description="Play Reversi against a trained AlphaZero agent (or greedy/random fallback).",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "checkpoint": _checkpoint_label,
        "using_network": _network is not None,
    }


@app.get("/agents", response_model=AgentsResponse)
def list_agents() -> AgentsResponse:
    options = [
        AgentOption(
            id=_AGENT_CHECKPOINT,
            label=_checkpoint_label or "Trained checkpoint",
            available=_network is not None,
        ),
        AgentOption(id=_AGENT_GREEDY, label="Greedy (max flips)", available=True),
        AgentOption(id=_AGENT_RANDOM, label="Random", available=True),
    ]
    return AgentsResponse(options=options, default=_default_agent_id)


@app.get("/new-game", response_model=BoardState)
def new_game() -> BoardState:
    state = initial_state()
    return _state_to_board(state, game_over=False, winner=None)


@app.post("/state", response_model=BoardState)
def get_state(request: PlayRequest) -> BoardState:
    """Replay action history and return the resulting board state — no agent call."""
    state = _reconstruct_state(request.actions)
    done, outcome = is_terminal(state)
    winner = _outcome_to_winner(outcome) if done else None
    return _state_to_board(state, done, winner)


@app.post("/play", response_model=PlayResponse)
def play(request: PlayRequest) -> PlayResponse:
    """Reconstruct state, run the requested agent, return its move + analysis."""
    state = _reconstruct_state(request.actions)

    done, outcome = is_terminal(state)
    if done:
        raise HTTPException(status_code=400, detail="Game is already over")

    agent_id = request.agent or _default_agent_id
    value_estimate: float = 0.0
    top_moves: list[MoveInfo] = []
    using_network = False

    if agent_id == _AGENT_CHECKPOINT:
        if _search is None:
            raise HTTPException(
                status_code=400,
                detail="Checkpoint agent requested but no checkpoint is loaded",
            )
        action, top_moves, value_estimate = _network_action(state)
        using_network = True
        agent_label = _checkpoint_label or "Trained checkpoint"

    elif agent_id == _AGENT_GREEDY:
        action, top_moves = _greedy_action(state)
        agent_label = "Greedy (max flips)"

    elif agent_id == _AGENT_RANDOM:
        action, top_moves = _random_action(state)
        agent_label = "Random"

    else:
        raise HTTPException(status_code=400, detail=f"Unknown agent: {agent_id!r}")

    next_state = apply_action(state, action)
    done_after, outcome_after = is_terminal(next_state)
    winner_after = _outcome_to_winner(outcome_after) if done_after else None

    return PlayResponse(
        action=action,
        description=_describe_action(action),
        board_after=_state_to_board(next_state, done_after, winner_after),
        top_moves=top_moves,
        value_estimate=value_estimate,
        using_network=using_network,
        agent_name=agent_label,
    )
