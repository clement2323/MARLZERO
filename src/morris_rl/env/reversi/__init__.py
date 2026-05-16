"""Reversi/Othello game environment.

Exposes make_env() which returns a GameEnv instance compatible with the
game-agnostic AlphaZero pipeline.
"""

from morris_rl.env.reversi.board import ACTION_SPACE_SIZE, NUM_POSITIONS
from morris_rl.env.reversi.encoding import NUM_PLANES, encode_state
from morris_rl.env.reversi.rules import apply_action, get_legal_actions, initial_state, is_terminal
from morris_rl.env.game_protocol import GameEnv


def make_env() -> GameEnv:
    """Return a GameEnv for Reversi, ready to plug into the AlphaZero pipeline."""
    return GameEnv(
        name="reversi",
        num_planes=NUM_PLANES,
        action_space_size=ACTION_SPACE_SIZE,
        num_positions=NUM_POSITIONS,
        initial_state=initial_state,
        get_legal_actions=get_legal_actions,
        apply_action=apply_action,
        is_terminal=is_terminal,
        encode_state=encode_state,
        # Symmetries are implemented in reversi.symmetries but not yet wired
        # into the augmentation pipeline — hook up once augment_samples is
        # refactored to be game-agnostic (tracked in configs/reversi_default.yaml).
        augment_samples=None,
        random_late_game_state=None,       # no curriculum for Reversi
        random_placement_phase_state=None,  # Reversi has no placement phase
    )
