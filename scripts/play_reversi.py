"""Interactive Reversi CLI — human vs agent or agent vs agent.

Usage
-----
    # Human (Black) vs agent loaded from a checkpoint
    uv run python scripts/play_reversi.py --checkpoint path/to/checkpoint.pt

    # Watch two agents play each other
    uv run python scripts/play_reversi.py --checkpoint path/to/checkpoint.pt --auto

    # Choose your colour
    uv run python scripts/play_reversi.py --checkpoint path/to/checkpoint.pt --human-color white

    # Random agent (no checkpoint — useful before any training)
    uv run python scripts/play_reversi.py

    # More MCTS sims for stronger play
    uv run python scripts/play_reversi.py --checkpoint ckpt.pt --sims 800
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morris_rl.env.reversi.board import ACTION_SPACE_SIZE, NUM_POSITIONS, rc_to_pos
from morris_rl.env.reversi.encoding import encode_state
from morris_rl.env.reversi.rules import (
    PASS_ACTION,
    PLAYER_1,
    PLAYER_2,
    GameState,
    Outcome,
    apply_action,
    get_legal_actions,
    initial_state,
    is_terminal,
    opponent,
)


# ---------------------------------------------------------------------------
# Board rendering
# ---------------------------------------------------------------------------

_COL_LABELS = "abcdefgh"
_BLACK = "●"
_WHITE = "○"
_EMPTY = "·"
_LEGAL = "+"   # shown when it's the human's turn


def print_board(state: GameState, highlight: set[int] | None = None) -> None:
    """Print an 8×8 Reversi board to stdout.

    Args:
        state: Current game state.
        highlight: Set of position indices to mark as legal moves (+).
    """
    hi = highlight or set()
    p1_count = int(np.sum(state.board == PLAYER_1))
    p2_count = int(np.sum(state.board == PLAYER_2))
    print(f"\n  {' '.join(_COL_LABELS)}")
    for row in range(8):
        row_str = f"{row + 1} "
        for col in range(8):
            pos = row * 8 + col
            cell = state.board[pos]
            if cell == PLAYER_1:
                row_str += _BLACK
            elif cell == PLAYER_2:
                row_str += _WHITE
            elif pos in hi:
                row_str += _LEGAL
            else:
                row_str += _EMPTY
            row_str += " "
        print(row_str.rstrip())
    print(f"\n  {_BLACK} Black (P1): {p1_count}   {_WHITE} White (P2): {p2_count}")
    mover = "Black (P1)" if state.current_player == PLAYER_1 else "White (P2)"
    print(f"  To move: {mover}")


def _pos_label(pos: int) -> str:
    """Convert flat position index to 'a1'-style label."""
    row, col = divmod(pos, 8)
    return f"{_COL_LABELS[col]}{row + 1}"


def _parse_move(text: str) -> int | None:
    """Parse a move string like 'c4' into a flat position index, or None."""
    text = text.strip().lower()
    if text in ("pass", "p"):
        return PASS_ACTION
    if len(text) != 2:
        return None
    col_ch, row_ch = text[0], text[1]
    if col_ch not in _COL_LABELS or not row_ch.isdigit():
        return None
    col = _COL_LABELS.index(col_ch)
    row = int(row_ch) - 1
    if not (0 <= row < 8 and 0 <= col < 8):
        return None
    return row * 8 + col


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


def _load_network(checkpoint_path: str, num_sims: int, device: torch.device):
    """Load a MorrisResNet trained for Reversi from *checkpoint_path*."""
    from morris_rl.network.resnet import MorrisResNet
    from morris_rl.mcts.search import MorrisSearch

    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = payload["state_dict"]

    # Infer architecture from checkpoint weights rather than requiring a config.
    input_conv_w = state_dict["input_conv.weight"]  # (num_channels, num_planes, 3)
    num_channels = input_conv_w.shape[0]
    num_planes = input_conv_w.shape[1]

    # Policy head fc2 output size = action_space_size; value head fc1 input = num_positions.
    action_space_size = state_dict["policy_head.fc2.weight"].shape[0]
    num_positions = state_dict["value_head.fc1.weight"].shape[1]
    policy_head_hidden = state_dict["policy_head.fc2.weight"].shape[1]
    value_head_hidden = state_dict["value_head.fc2.weight"].shape[1]

    # Count residual blocks: each block has conv1/conv2, so keys like
    # "trunk.0.conv1.weight", "trunk.1.conv1.weight", etc.
    num_blocks = sum(
        1 for k in state_dict if k.startswith("trunk.") and k.endswith(".conv1.weight")
    )

    network = MorrisResNet(
        num_blocks=num_blocks,
        num_channels=num_channels,
        num_planes=num_planes,
        policy_head_hidden=policy_head_hidden,
        value_head_hidden=value_head_hidden,
        num_positions=num_positions,
        action_space_size=action_space_size,
    ).to(device)
    network.load_state_dict(state_dict)
    network.eval()

    reversi_fns = {
        "initial_state": initial_state,
        "get_legal_actions": get_legal_actions,
        "apply_action": apply_action,
        "is_terminal": is_terminal,
        "encode_state": encode_state,
        "action_space_size": action_space_size,
    }
    search = MorrisSearch(
        network,
        device,
        num_simulations=num_sims,
        game_fns=reversi_fns,
    )
    return network, search


def _agent_move(search, state: GameState, device: torch.device) -> int:
    action, visit_probs = search.run(state, temperature=1e-6, add_noise=False)
    return int(action)


def _random_move(state: GameState, rng: np.random.Generator) -> int:
    return int(rng.choice(get_legal_actions(state)))


# ---------------------------------------------------------------------------
# Game loop
# ---------------------------------------------------------------------------


def _human_turn(state: GameState) -> int:
    legal = get_legal_actions(state)
    legal_set = set(legal)

    if legal == [PASS_ACTION]:
        print("  No legal moves — forced pass.")
        input("  Press Enter to continue…")
        return PASS_ACTION

    # Show board with legal move highlights.
    print_board(state, highlight=legal_set)
    legal_labels = [_pos_label(a) for a in legal if a != PASS_ACTION]
    print(f"  Legal moves: {', '.join(legal_labels)}")

    while True:
        raw = input("  Your move (e.g. 'c4', or 'pass'): ").strip()
        action = _parse_move(raw)
        if action is None:
            print("  Bad format — use column+row like 'c4'.")
            continue
        if action not in legal_set:
            print(f"  '{raw}' is not a legal move here.")
            continue
        return action


def _play(
    human_player: int | None,   # None = agent vs agent
    search,
    device: torch.device,
    rng: np.random.Generator,
) -> None:
    state = initial_state()
    move_num = 0

    while True:
        done, outcome = is_terminal(state)
        if done:
            print_board(state)
            print("\n" + "=" * 40)
            if outcome == Outcome.DRAW:
                print("  Game over — Draw!")
            elif outcome == Outcome.PLAYER_1_WINS:
                print("  Game over — Black (P1) wins!")
            else:
                print("  Game over — White (P2) wins!")
            p1 = int(np.sum(state.board == PLAYER_1))
            p2 = int(np.sum(state.board == PLAYER_2))
            print(f"  Final score: Black {p1} – White {p2}")
            return

        is_human = (state.current_player == human_player)
        mover_name = "Black (P1)" if state.current_player == PLAYER_1 else "White (P2)"
        move_num += 1

        if is_human:
            action = _human_turn(state)
        else:
            legal = get_legal_actions(state)
            if legal == [PASS_ACTION]:
                print_board(state)
                print(f"  Move {move_num}: {mover_name} has no moves — pass.")
                input("  Press Enter to continue…")
                action = PASS_ACTION
            else:
                print_board(state)
                if search is not None:
                    print(f"  Move {move_num}: {mover_name} is thinking…")
                    action = _agent_move(search, state, device)
                else:
                    action = _random_move(state, rng)
                label = _pos_label(action) if action != PASS_ACTION else "pass"
                print(f"  Move {move_num}: {mover_name} plays {label}.")
                if human_player is not None:
                    input("  Press Enter to continue…")

        state = apply_action(state, action)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Reversi against the AlphaZero agent.")
    parser.add_argument("--checkpoint", default=None, help="Path to a .pt checkpoint.")
    parser.add_argument(
        "--human-color",
        choices=["black", "white"],
        default="black",
        help="Play as black (P1, moves first) or white (P2). Default: black.",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Agent vs agent — no human input. Watch the game play out.",
    )
    parser.add_argument("--sims", type=int, default=200, help="MCTS simulations per move.")
    parser.add_argument("--device", default="auto", help="'cpu', 'cuda', or 'auto'.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    rng = np.random.default_rng(args.seed)

    search = None
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        _, search = _load_network(args.checkpoint, args.sims, device)
        print(f"Agent ready ({args.sims} sims/move, device={device}).")
    else:
        print("No checkpoint provided — using random moves.")

    if args.auto:
        human_player = None
    else:
        human_player = PLAYER_1 if args.human_color == "black" else PLAYER_2
        color_name = "Black (P1, moves first)" if human_player == PLAYER_1 else "White (P2)"
        print(f"You are playing as: {color_name}")

    print(f"\n  {_BLACK} = Black (P1)   {_WHITE} = White (P2)   {_LEGAL} = legal move\n")
    _play(human_player, search, device, rng)


if __name__ == "__main__":
    main()
