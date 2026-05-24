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
    compute_aux_features,
    get_legal_actions,
    is_terminal,
)
from morris_rl.eval.baselines import MinimaxAgent
from morris_rl.inference.play import (
    IllegalActionError,
    describe_action,
    reconstruct_state,
    run_mcts_analysis,
)
from morris_rl.network.factory import build_network
from morris_rl.network.resnet import MorrisResNet  # noqa: F401  (kept for backward compat / type hints)
from morris_rl.utils.checkpoints import load_checkpoint
from morris_rl.utils.logging import logger

_NUM_PLANES = 7

# Module-level network (only set when a checkpoint loads). Minimax agents are
# instantiated per-request — they're cheap (just an int) and stateless.
_network: nn.Module | None = None
_device: torch.device = torch.device("cpu")
_num_simulations: int = 200
_checkpoint_label: str | None = None
_default_agent_id: str = "minimax-5"
# Architecture of the loaded checkpoint and the matching state encoder.
# Both are set by _load_network() based on the checkpoint's config.network.type.
_network_type: str | None = None
_encode_fn = None  # Callable[[GameState], torch.Tensor] | None

_AGENT_CHECKPOINT = "checkpoint"
_AGENT_MINIMAX_3 = "minimax-3"
_AGENT_MINIMAX_5 = "minimax-5"


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class PlayRequest(BaseModel):
    """Client sends the complete list of actions played so far."""

    actions: list[int]
    # Which adversary to run for this turn. None falls back to the server's
    # default (checkpoint when loaded, otherwise minimax-5).
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
    # Aux signals exposed to the UI so it can show "you're losing" feedback
    # (shake / loser animation) without re-computing the rules client-side.
    # Both are signed and from the perspective of the player whose turn comes
    # next on `board_after` (typically the human after the agent's move).
    pieces_diff: float      # own_pieces_on_board - opp_pieces_on_board
    mill_diff: float        # own_active_mills - opp_active_mills
    board_after: BoardState
    using_network: bool     # False when falling back to minimax
    agent_name: str


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------


def _load_network() -> None:
    """Load whichever network type is recorded in the checkpoint's config.

    Supports both ResNet (legacy 7-plane) and GraphNet (Morris-only 11-plane)
    via the factory. The default agent flips to 'checkpoint' when a network is
    available, otherwise stays on 'minimax-5'.
    """
    global _network, _device, _num_simulations, _checkpoint_label, _default_agent_id
    global _network_type, _encode_fn

    _device = torch.device(os.getenv("DEVICE", "cpu"))
    _num_simulations = int(os.getenv("NUM_SIMULATIONS", "200"))
    checkpoint_path = os.getenv("MODEL_CHECKPOINT", "")

    if checkpoint_path and Path(checkpoint_path).exists():
        from omegaconf import OmegaConf
        from morris_rl.env.encoding_graph import encode_state_graph
        from morris_rl.mcts.search import encode_state as encode_state_legacy

        payload = load_checkpoint(checkpoint_path)
        cfg_dict = payload.get("config") or {}
        cfg = OmegaConf.create(cfg_dict)
        net_cfg = cfg.get("network", {}) or {}
        network = build_network(cfg)
        network.load_state_dict(payload["state_dict"])
        network.eval().to(_device)
        _network = network
        _network_type = str(net_cfg.get("type", "resnet"))
        _encode_fn = (
            encode_state_graph if _network_type == "graphnet" else encode_state_legacy
        )
        step = payload["step"]
        num_blocks = int(net_cfg.get("num_blocks", 0))
        num_channels = int(net_cfg.get("num_channels", 0))
        tag = "GraphNet" if _network_type == "graphnet" else "ResNet"
        _checkpoint_label = (
            f"{tag}{num_blocks}×{num_channels} step={step} ({_num_simulations} sims)"
        )
        _default_agent_id = _AGENT_CHECKPOINT
        logger.info(
            f"Loaded {_network_type} from {checkpoint_path} "
            f"(step {step}, {num_blocks}×{num_channels})"
        )
    else:
        _network = None
        _network_type = None
        _encode_fn = None
        _checkpoint_label = None
        _default_agent_id = _AGENT_MINIMAX_5
        logger.info("No checkpoint found — defaulting to Minimax depth=5")


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    _load_network()
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
    return {"status": "ok", "default_agent": _default_agent_id}


@app.get("/agents", response_model=AgentsResponse)
def list_agents() -> AgentsResponse:
    """Return the agents the client can pick from, plus the server default."""
    options = [
        AgentOption(
            id=_AGENT_CHECKPOINT,
            label=_checkpoint_label or "Trained checkpoint",
            available=_network is not None,
        ),
        AgentOption(id=_AGENT_MINIMAX_3, label="Minimax depth 3", available=True),
        AgentOption(id=_AGENT_MINIMAX_5, label="Minimax depth 5", available=True),
    ]
    return AgentsResponse(options=options, default=_default_agent_id)


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
    """Reconstruct state from action history, run the requested agent, return its move.

    The client should call this endpoint only when it is the agent's turn.
    The reconstructed state must NOT be terminal; if it is, a 400 is returned.
    """
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

    # Pick the agent for this turn.
    agent_id = request.agent or _default_agent_id

    if agent_id == _AGENT_CHECKPOINT:
        if _network is None:
            raise HTTPException(
                status_code=400,
                detail="Checkpoint agent requested but no checkpoint is loaded",
            )
        from morris_rl.mcts.search import encode_state as _encode_state_legacy
        action, top_moves_raw, value = run_mcts_analysis(
            _network,
            _device,
            state,
            num_simulations=_num_simulations,
            encode_fn=_encode_fn or _encode_state_legacy,
        )
        top_moves = [
            MoveInfo(
                action=a,
                visit_prob=p,
                description=describe_action(a, state.must_capture),
            )
            for a, p in top_moves_raw
        ]
        using_network = True
        agent_label = _checkpoint_label or "Trained checkpoint"
    elif agent_id.startswith("minimax-"):
        # Accepts any depth ≥ 1; /agents only advertises 3 and 5, but lower
        # depths exist for tests that need fast moves.
        try:
            depth = int(agent_id.split("-", 1)[1])
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail=f"Invalid minimax depth in '{agent_id}'"
            ) from exc
        if depth < 1:
            raise HTTPException(status_code=400, detail="Minimax depth must be ≥ 1")
        action = MinimaxAgent(depth=depth).select_action(state)
        top_moves = [
            MoveInfo(
                action=action,
                visit_prob=1.0,
                description=describe_action(action, state.must_capture),
            )
        ]
        value = 0.0
        using_network = False
        agent_label = f"Minimax depth={depth}"
    else:
        raise HTTPException(status_code=400, detail=f"Unknown agent: {agent_id}")

    # Apply the agent's action.
    next_state = apply_action(state, action)
    done_after, outcome_after = is_terminal(next_state)
    winner = (
        int(outcome_after)
        if (done_after and outcome_after is not None and outcome_after.value > 0)
        else None
    )
    # Aux signals from the POV of whoever moves next on `board_after`
    # (i.e. the human in the normal turn flow). The frontend uses these to
    # decide whether to play the loser / shake animations.
    mill_diff, pieces_diff = compute_aux_features(next_state)

    return PlayResponse(
        action=action,
        description=describe_action(action, state.must_capture),
        top_moves=top_moves,
        value_estimate=value,
        pieces_diff=pieces_diff,
        mill_diff=mill_diff,
        board_after=_state_to_board(next_state, done_after, winner),
        using_network=using_network,
        agent_name=agent_label,
    )
