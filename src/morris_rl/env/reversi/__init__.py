"""Reversi/Othello game environment — board, encoding, rules, symmetries."""

from morris_rl.env.reversi.board import ACTION_SPACE_SIZE, NUM_POSITIONS
from morris_rl.env.reversi.encoding import NUM_PLANES, encode_state
from morris_rl.env.reversi.rules import apply_action, get_legal_actions, initial_state, is_terminal

__all__ = [
    "ACTION_SPACE_SIZE",
    "NUM_POSITIONS",
    "NUM_PLANES",
    "encode_state",
    "apply_action",
    "get_legal_actions",
    "initial_state",
    "is_terminal",
]
