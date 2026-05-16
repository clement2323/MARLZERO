"""Nine Men's Morris GameEnv factory.

Call ``make_env()`` to obtain a fully assembled :class:`~morris_rl.env.game_protocol.GameEnv`
that can be passed to the game-agnostic training pipeline.
"""

from __future__ import annotations

from morris_rl.env.game_protocol import GameEnv
from morris_rl.env.morris.board import ACTION_SPACE_SIZE, NUM_POSITIONS
from morris_rl.env.morris.rules import (
    apply_action,
    get_legal_actions,
    initial_state,
    is_terminal,
    random_late_game_state,
)

# encode_state lives in mcts/search.py (it depends on torch and the
# LightZero imports).  We import it here so GameEnv stays self-contained;
# the dependency is tolerated because search.py is always available.
from morris_rl.mcts.search import encode_state

_NUM_PLANES = 7


def make_env() -> GameEnv:
    """Return a :class:`GameEnv` fully configured for Nine Men's Morris.

    ``random_placement_phase_state`` is not yet implemented for Morris
    (placement-phase curriculum not needed for the current training schedule)
    so it is left as ``None``.
    """
    return GameEnv(
        name="morris",
        num_planes=_NUM_PLANES,
        action_space_size=ACTION_SPACE_SIZE,
        num_positions=NUM_POSITIONS,
        initial_state=initial_state,
        get_legal_actions=get_legal_actions,
        apply_action=apply_action,
        is_terminal=is_terminal,
        encode_state=encode_state,
        # Symmetry augmentation for Morris is handled by the replay buffer
        # directly via morris_rl.env.symmetries; no extra wrapper needed here.
        augment_samples=None,
        random_late_game_state=random_late_game_state,
        random_placement_phase_state=None,
    )
