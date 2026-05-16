"""AlphaZero MCTS search — uses the compiled ctree_alphazero C extension when
available, falls back to the pure-Python ptree_az otherwise.

ctree is ~10x faster than ptree; it requires the .so to be compiled from the
LightZero C++ source (see docs/decisions/002-ctree-build.md).
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict

try:
    from lzero.mcts.ctree.ctree_alphazero import MCTS as _CtreeMCTS

    _CTREE_AVAILABLE = True
except ImportError:
    _CTREE_AVAILABLE = False

from lzero.mcts.ptree.ptree_az import MCTS as _PtreeMCTS
from morris_rl.utils.logging import logger

logger.debug(
    "MCTS backend: {} — {}",
    "ctree (C++, fast)" if _CTREE_AVAILABLE else "ptree (Python fallback, ~10x slower)",
    "recompile LightZero C++ ext if you expected ctree" if not _CTREE_AVAILABLE else "OK",
)

from morris_rl.env.board import ACTION_SPACE_SIZE, NUM_PIECES_PER_PLAYER, NUM_POSITIONS
from morris_rl.env.rules import (
    GameState,
    Outcome,
    apply_action,
    get_legal_actions,
    get_phase,
    initial_state,
    is_terminal,
    opponent,
)

_NUM_PLANES = 7


# ---------------------------------------------------------------------------
# State encoding (8 planes × 24 positions)
# ---------------------------------------------------------------------------


def encode_state(state: GameState) -> torch.Tensor:
    """Encode a GameState as a float32 tensor of shape (1, 7, 24).

    Planes:
        0 — current player's pieces
        1 — opponent's pieces
        2 — current player's hand fraction (scalar broadcast)
        3 — opponent's hand fraction (scalar broadcast)
        4 — phase == PLACING (broadcast)
        5 — phase == MOVING (broadcast)
        6 — must_capture flag (broadcast)

    No FLYING plane: this variant removes the standard flying rule, so a
    player at 3 pieces is still in the MOVING phase (constrained to
    adjacency).
    """
    player = state.current_player
    opp = opponent(player)
    board = state.board
    hand = state.pieces_in_hand
    phase = int(get_phase(state, player))

    planes = np.zeros((_NUM_PLANES, NUM_POSITIONS), dtype=np.float32)
    planes[0] = board == player
    planes[1] = board == opp
    planes[2] = hand[player - 1] / NUM_PIECES_PER_PLAYER
    planes[3] = hand[opp - 1] / NUM_PIECES_PER_PLAYER
    planes[4 + phase] = 1.0
    planes[6] = float(state.must_capture)

    return torch.from_numpy(planes).unsqueeze(0)  # (1, 7, 24)


# ---------------------------------------------------------------------------
# LightZero-compatible environment adapter
# ---------------------------------------------------------------------------


class MorrisSimEnv:
    """Thin wrapper around GameState that satisfies LightZero's env protocol.

    LightZero's ptree_az.MCTS calls:
        reset(start_player_index, init_state)
        step(action)
        get_done_winner() → (done: bool, winner: int)
        .legal_actions      → list[int]
        .action_space.n     → int
        .current_player     → int (1 or 2)
        .battle_mode_in_simulation_env
        .battle_mode
        .render_mode
    """

    battle_mode_in_simulation_env: str = "self_play_mode"

    def __init__(self, game_fns: dict | None = None) -> None:
        _fns = game_fns or {}
        self._game_initial_state = _fns.get("initial_state", initial_state)
        self._game_apply_action = _fns.get("apply_action", apply_action)
        self._game_legal_actions = _fns.get("get_legal_actions", get_legal_actions)
        self._game_is_terminal = _fns.get("is_terminal", is_terminal)
        self._state = self._game_initial_state()
        self._root_state = None  # set by MorrisSearch.run before each search
        self.battle_mode: str = "self_play_mode"
        self.render_mode: str | None = None
        self.action_space = SimpleNamespace(n=_fns.get("action_space_size", ACTION_SPACE_SIZE))

    def reset(
        self,
        start_player_index: int = 0,
        init_state: object = None,
        katago_policy_init: bool = False,
        katago_game_state: object = None,
    ) -> None:
        """Reset to init_state, _root_state, or initial_state() — first non-None wins.

        The ctree backend calls reset with init_state as bytes (LightZero Go
        integration artifact) — those are ignored and _root_state takes over.
        Direct callers (e.g. tests) pass a real GameState object.
        """
        if init_state is not None and not isinstance(init_state, (bytes, np.ndarray)):
            self._state = init_state.copy()
        elif self._root_state is not None:
            self._state = self._root_state.copy()
        else:
            self._state = self._game_initial_state()

    def step(self, action: int) -> None:
        """Advance the state by one action."""
        self._state = self._game_apply_action(self._state, action)

    @property
    def legal_actions(self) -> list[int]:
        return self._game_legal_actions(self._state)

    @property
    def current_player(self) -> int:
        """Return the current player as 1 or 2 (matching LightZero's winner convention)."""
        return self._state.current_player

    def get_done_winner(self) -> tuple[bool, int]:
        """Return (done, winner) where winner is 1, 2, or -1 (draw/ongoing)."""
        done, outcome = self._game_is_terminal(self._state)
        if not done:
            return False, -1
        if outcome == Outcome.DRAW or outcome is None:
            return True, -1
        return True, int(outcome)  # Outcome.PLAYER_1_WINS=1, PLAYER_2_WINS=2


# ---------------------------------------------------------------------------
# Policy forward function factory
# ---------------------------------------------------------------------------


EvalFn = Callable[
    [GameState],
    tuple[dict[int, float], float],
]
"""Leaf evaluator signature.

Takes a :class:`GameState`, returns:
    (action → prior_probability dict over legal actions, value estimate in [-1, 1])

This abstraction lets MCTS run with either a local in-process network call or
a remote shared GPU server that batches across workers — see
``training/inference_server.py``. The default factory below builds the local
flavour from ``(network, device)``.
"""


def _make_local_eval_fn(
    network: nn.Module,
    device: torch.device,
    encode_fn=None,
    get_legal_fn=None,
    action_space_size: int | None = None,
) -> EvalFn:
    """Build an EvalFn that runs the given network in-process on *device*.

    Optional overrides allow the same factory to serve non-Morris games without
    touching the Morris defaults: pass encode_fn, get_legal_fn, and
    action_space_size from the game's own module; omit them for Morris.
    """
    _encode = encode_fn or encode_state
    _legal = get_legal_fn or get_legal_actions
    _n = action_space_size or ACTION_SPACE_SIZE

    def evaluate(state: GameState) -> tuple[dict[int, float], float]:
        legal = _legal(state)
        x = _encode(state).to(device)
        mask = torch.zeros(1, _n, dtype=torch.bool, device=device)
        for a in legal:
            mask[0, a] = True

        with torch.no_grad():
            log_policy, value = network(x, mask)

        probs = log_policy.exp()[0].cpu().numpy()
        return {a: float(probs[a]) for a in legal}, float(value[0].item())

    return evaluate


def _adapt_eval_fn_to_lzero(
    eval_fn: EvalFn,
) -> Callable[[MorrisSimEnv], tuple[dict[int, float], float]]:
    """Wrap an EvalFn so it satisfies LightZero's policy_forward(env) signature."""

    def policy_forward(env: MorrisSimEnv) -> tuple[dict[int, float], float]:
        return eval_fn(env._state)

    return policy_forward


# ---------------------------------------------------------------------------
# Public search API
# ---------------------------------------------------------------------------


class MorrisSearch:
    """AlphaZero MCTS search for Nine Men's Morris.

    Usage::

        search = MorrisSearch(network, device, num_simulations=200)
        action, visit_probs = search.run(state, temperature=1.0, add_noise=True)
    """

    def __init__(
        self,
        network: nn.Module | None = None,
        device: torch.device | None = None,
        num_simulations: int = 800,
        c_puct_base: float = 19652.0,
        c_puct_init: float = 1.25,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        eval_fn: EvalFn | None = None,
        game_fns: dict | None = None,
    ) -> None:
        """Construct an MCTS search.

        Either provide ``(network, device)`` to run inference locally in-process,
        or provide ``eval_fn`` to delegate evaluation (e.g. to a centralized GPU
        inference server). Exactly one of the two must be supplied.

        ``game_fns`` is an optional dict that overrides the Morris defaults for
        any alternate game (e.g. Reversi). Recognised keys: ``initial_state``,
        ``apply_action``, ``get_legal_actions``, ``is_terminal``,
        ``encode_state``, ``action_space_size``.  Omit the dict (or pass None)
        to keep the Morris defaults.
        """
        if eval_fn is None:
            if network is None or device is None:
                raise ValueError(
                    "Provide either eval_fn, or both network and device."
                )
            _fns = game_fns or {}
            eval_fn = _make_local_eval_fn(
                network,
                device,
                encode_fn=_fns.get("encode_state"),
                get_legal_fn=_fns.get("get_legal_actions"),
                action_space_size=_fns.get("action_space_size"),
            )
        self._sim_env = MorrisSimEnv(game_fns)
        self._network = network
        self._device = device
        # Keep a direct reference to the raw eval_fn so callers can query the
        # network's value estimate at the root *without* running MCTS — used
        # by the resign-threshold logic in self_play._play_game (one extra
        # forward per move, ~0.4% overhead vs a 250-sim search).
        self._eval_fn: EvalFn = eval_fn
        self._policy_fn = _adapt_eval_fn_to_lzero(eval_fn)

        if _CTREE_AVAILABLE:
            # C++ backend: constructor takes positional args, not a cfg dict.
            self._mcts: Any = _CtreeMCTS(
                512,  # max_moves — safety cap per game
                num_simulations,
                c_puct_base,
                c_puct_init,
                dirichlet_alpha,
                dirichlet_epsilon,
                self._sim_env,
            )
            self._use_ctree = True
        else:
            cfg = EasyDict(
                {
                    "num_simulations": num_simulations,
                    "pb_c_base": c_puct_base,
                    "pb_c_init": c_puct_init,
                    "root_dirichlet_alpha": dirichlet_alpha,
                    "root_noise_weight": dirichlet_epsilon,
                }
            )
            self._mcts = _PtreeMCTS(cfg, self._sim_env)
            self._use_ctree = False

    def run(
        self,
        state: GameState,
        temperature: float = 1.0,
        add_noise: bool = True,
    ) -> tuple[int, np.ndarray[tuple[int], np.dtype[np.float32]]]:
        """Run MCTS from *state* and return the selected action and visit distribution.

        Args:
            state: The game state to search from.
            temperature: Controls action selection sharpness. Use 1.0 during
                         training and 0.0 (argmax) during evaluation.
            add_noise: Whether to inject Dirichlet exploration noise at the root.
                       Should be True during self-play, False during evaluation.

        Returns:
            Tuple of (action, visit_probs) where visit_probs has shape
            (ACTION_SPACE_SIZE,) and sums to 1.
        """
        # Store root so MorrisSimEnv.reset() can restore it regardless of
        # whether ctree passes bytes or ptree passes the original object back.
        self._sim_env._root_state = state
        state_config = {
            "start_player_index": state.current_player - 1,
            # ctree calls .tobytes() on init_state → must be a numpy array.
            # The actual state is restored via _root_state above.
            "init_state": np.zeros(1, dtype=np.uint8),
            # ctree-specific keys (used by LightZero's Go integration; ignored here)
            "katago_game_state": None,
            "katago_policy_init": False,
        }
        result = self._mcts.get_next_action(
            state_config,
            self._policy_fn,
            temperature,
            add_noise,
        )
        # ctree returns (action, probs, root_node); ptree returns (action, probs)
        action, action_probs = result[0], result[1]
        return int(action), np.array(action_probs, dtype=np.float32)

    def root_value(self, state: GameState) -> float:
        """Return the network's value estimate for *state* (no MCTS).

        Used by the resign-threshold logic: it gives a per-move scalar in
        [-1, 1] from the current player's POV, cheaper than running another
        MCTS search.  ``MCTSCtree`` does not expose its post-search root Q
        through Python, so we go straight to the eval_fn that MCTS uses for
        leaf evaluation.
        """
        _, value = self._eval_fn(state)
        return float(value)
