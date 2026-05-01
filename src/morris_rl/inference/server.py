"""FastAPI inference server for the Nine Men's Morris web demo.

Endpoints
---------
GET  /health          — liveness check
GET  /new-game        — return the initial board state
POST /play            — given action history, return the agent's next move + analysis

State is fully reconstructed server-side from the flat action history sent by
the client, so the server is entirely stateless.

Startup configuration (environment variables)
--------------------------------------------
MODEL_CHECKPOINT  Path to a ``.pt`` checkpoint saved by :func:`~morris_rl.utils.checkpoints.save_checkpoint`.
                  If unset or the file does not exist, falls back to MinimaxAgent(depth=3).
NUM_SIMULATIONS   MCTS simulations per move when a network is loaded (default: 200).
NUM_BLOCKS        Network depth (default: 10); must match the checkpoint.
NUM_CHANNELS      Network width (default: 128); must match the checkpoint.
DEVICE            ``cpu`` or ``cuda`` (default: ``cpu``).
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from morris_rl.env.rules import (
    GameState,
    apply_action,
    get_legal_actions,
    is_terminal,
)
from morris_rl.eval.arena import Agent
from morris_rl.eval.baselines import MinimaxAgent, NetworkAgent
from morris_rl.inference.play import (
    IllegalActionError,
    describe_action,
    reconstruct_state,
    run_mcts_analysis,
)
from morris_rl.network.resnet import MorrisResNet
from morris_rl.utils.checkpoints import load_checkpoint
from morris_rl.utils.logging import logger

_NUM_PLANES = 7

# Module-level agent and (optional) network — set during lifespan startup.
_agent: Agent | None = None
_network: nn.Module | None = None
_device: torch.device = torch.device("cpu")
_num_simulations: int = 200
_agent_name: str = "unknown"


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class PlayRequest(BaseModel):
    """Client sends the complete list of actions played so far."""

    actions: list[int]


class MoveInfo(BaseModel):
    action: int
    visit_prob: float
    description: str


class BoardState(BaseModel):
    board: list[int]
    current_player: int
    pieces_in_hand: list[int]
    must_capture: bool
    game_over: bool
    winner: int | None
    # Authoritative legal actions for the current player at this state.
    # Empty when game_over. Clients should trust this list rather than
    # reimplementing rules (mill protection during capture is non-trivial).
    legal_actions: list[int]


class PlayResponse(BaseModel):
    action: int
    description: str
    top_moves: list[MoveInfo]
    value_estimate: float   # [-1, 1]; positive = agent thinks it's winning
    board_after: BoardState
    using_network: bool     # False when falling back to minimax
    agent_name: str


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------


def _load_agent() -> None:
    global _agent, _network, _device, _num_simulations

    _device = torch.device(os.getenv("DEVICE", "cpu"))
    _num_simulations = int(os.getenv("NUM_SIMULATIONS", "200"))
    checkpoint_path = os.getenv("MODEL_CHECKPOINT", "")

    if checkpoint_path and Path(checkpoint_path).exists():
        num_blocks = int(os.getenv("NUM_BLOCKS", "10"))
        num_channels = int(os.getenv("NUM_CHANNELS", "128"))
        network = MorrisResNet(
            num_blocks=num_blocks,
            num_channels=num_channels,
            num_planes=_NUM_PLANES,
            policy_head_hidden=64,
            value_head_hidden=64,
        )
        payload = load_checkpoint(checkpoint_path)
        network.load_state_dict(payload["state_dict"])
        network.eval().to(_device)
        _network = network
        _agent = NetworkAgent(network, _device, num_simulations=_num_simulations)
        step = payload["step"]
        _agent_name = f"ResNet{num_blocks}×{num_channels} step={step} ({_num_simulations} sims)"
        logger.info(f"Loaded network from {checkpoint_path} (step {step})")
    else:
        depth = int(os.getenv("MINIMAX_DEPTH", "3"))
        _agent = MinimaxAgent(depth=depth)
        _agent_name = f"Minimax depth={depth}"
        logger.info(f"No checkpoint found — using MinimaxAgent(depth={depth})")


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    _load_agent()
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Nine Men's Morris RL Demo",
    description="Play against a trained AlphaZero agent (or minimax fallback).",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state_to_board(state: GameState, game_over: bool, winner: int | None) -> BoardState:
    legal = [] if game_over else get_legal_actions(state)
    return BoardState(
        board=state.board.tolist(),
        current_player=int(state.current_player),
        pieces_in_hand=list(state.pieces_in_hand),
        must_capture=bool(state.must_capture),
        game_over=game_over,
        winner=winner,
        legal_actions=legal,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "agent_name": _agent_name}


@app.get("/new-game", response_model=BoardState)
def new_game() -> BoardState:
    """Return the initial board state so the client can bootstrap."""
    from morris_rl.env.rules import initial_state

    state = initial_state()
    return _state_to_board(state, game_over=False, winner=None)


@app.post("/state", response_model=BoardState)
def get_state(request: PlayRequest) -> BoardState:
    """Replay the action history and return the resulting board state — no agent.

    Used by the client after a human move to learn whether the next mover is
    still the human (e.g. a mill was formed and a capture is owed) or the
    agent.  When the human's turn continues, the client must NOT call /play —
    that would let the agent pick the human's capture target.
    """
    try:
        state = reconstruct_state(request.actions)
    except IllegalActionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    done, outcome = is_terminal(state)
    winner = (
        int(outcome) if (done and outcome is not None and outcome.value > 0) else None
    )
    return _state_to_board(state, done, winner)


@app.post("/play", response_model=PlayResponse)
def play(request: PlayRequest) -> PlayResponse:
    """Reconstruct state from action history, run the agent, return its move.

    The client should call this endpoint only when it is the agent's turn.
    The reconstructed state must NOT be terminal; if it is, a 400 is returned.
    """
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialised")

    try:
        state = reconstruct_state(request.actions)
    except IllegalActionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    done, outcome = is_terminal(state)
    if done:
        raise HTTPException(status_code=400, detail="Game is already over")

    legal = get_legal_actions(state)
    if not legal:
        raise HTTPException(status_code=400, detail="No legal moves available")

    # Run the agent.
    using_network = _network is not None
    if using_network and _network is not None:
        action, top_moves_raw, value = run_mcts_analysis(
            _network, _device, state, num_simulations=_num_simulations
        )
        top_moves = [
            MoveInfo(
                action=a,
                visit_prob=p,
                description=describe_action(a, state.must_capture),
            )
            for a, p in top_moves_raw
        ]
    else:
        action = _agent.select_action(state)
        top_moves = [
            MoveInfo(
                action=action,
                visit_prob=1.0,
                description=describe_action(action, state.must_capture),
            )
        ]
        value = 0.0

    # Apply the agent's action.
    next_state = apply_action(state, action)
    done_after, outcome_after = is_terminal(next_state)
    winner = int(outcome_after) if (done_after and outcome_after is not None and outcome_after.value > 0) else None

    return PlayResponse(
        action=action,
        description=describe_action(action, state.must_capture),
        top_moves=top_moves,
        value_estimate=value,
        board_after=_state_to_board(next_state, done_after, winner),
        using_network=using_network,
        agent_name=_agent_name,
    )
