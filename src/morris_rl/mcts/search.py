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

_NUM_PLANES = 8


# ---------------------------------------------------------------------------
# State encoding (8 planes × 24 positions)
# ---------------------------------------------------------------------------


def encode_state(state: GameState) -> torch.Tensor:
    """Encode a GameState as a float32 tensor of shape (1, 8, 24).

    Planes:
        0 — current player's pieces
        1 — opponent's pieces
        2 — current player's hand fraction (scalar broadcast)
        3 — opponent's hand fraction (scalar broadcast)
        4 — phase == PLACING (broadcast)
        5 — phase == MOVING (broadcast)
        6 — phase == FLYING (broadcast)
        7 — must_capture flag (broadcast)
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
    planes[7] = float(state.must_capture)

    return torch.from_numpy(planes).unsqueeze(0)  # (1, 8, 24)


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

    def __init__(self) -> None:
        self._state: GameState = initial_state()
        self._root_state: GameState | None = None  # set by MorrisSearch.run before each search
        self.battle_mode: str = "self_play_mode"
        self.render_mode: str | None = None
        self.action_space = SimpleNamespace(n=ACTION_SPACE_SIZE)

    def reset(
        self,
        start_player_index: int = 0,
        init_state: object = None,
        katago_policy_init: bool = False,
        katago_game_state: object = None,
    ) -> None:
        """Reset to the stored root state.

        The ctree backend converts init_state to bytes and passes 4 positional
        args (including katago_* args used by LightZero's Go integration).
        We ignore those extras and always restore from _root_state, which is
        set by MorrisSearch.run before each search.
        """
        if self._root_state is not None:
            self._state = self._root_state.copy()
        elif isinstance(init_state, GameState):
            self._state = init_state.copy()
        else:
            self._state = initial_state()

    def step(self, action: int) -> None:
        """Advance the state by one action."""
        self._state = apply_action(self._state, action)

    @property
    def legal_actions(self) -> list[int]:
        return get_legal_actions(self._state)

    @property
    def current_player(self) -> int:
        """Return the current player as 1 or 2 (matching LightZero's winner convention)."""
        return self._state.current_player

    def get_done_winner(self) -> tuple[bool, int]:
        """Return (done, winner) where winner is 1, 2, or -1 (draw/ongoing)."""
        done, outcome = is_terminal(self._state)
        if not done:
            return False, -1
        if outcome == Outcome.DRAW or outcome is None:
            return True, -1
        return True, int(outcome)  # Outcome.PLAYER_1_WINS=1, PLAYER_2_WINS=2


# ---------------------------------------------------------------------------
# Policy forward function factory
# ---------------------------------------------------------------------------


def _make_policy_fn(
    network: nn.Module, device: torch.device
) -> Callable[[MorrisSimEnv], tuple[dict[int, float], float]]:
    """Return a policy forward function compatible with LightZero's MCTS."""

    def policy_forward(env: MorrisSimEnv) -> tuple[dict[int, float], float]:
        state = env._state
        legal = get_legal_actions(state)

        x = encode_state(state).to(device)
        mask = torch.zeros(1, ACTION_SPACE_SIZE, dtype=torch.bool, device=device)
        for a in legal:
            mask[0, a] = True

        with torch.no_grad():
            log_policy, value = network(x, mask)

        probs: np.ndarray[tuple[int], np.dtype[np.float32]] = log_policy.exp()[0].cpu().numpy()
        action_probs_dict = {a: float(probs[a]) for a in legal}
        return action_probs_dict, float(value[0].item())

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
        network: nn.Module,
        device: torch.device,
        num_simulations: int = 800,
        c_puct_base: float = 19652.0,
        c_puct_init: float = 1.25,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
    ) -> None:
        self._sim_env = MorrisSimEnv()
        self._network = network
        self._device = device
        self._policy_fn = _make_policy_fn(network, device)

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
