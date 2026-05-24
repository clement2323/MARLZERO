"""Human-vs-network CLI for Nine Men's Morris (UTF-8 board renderer).

Usage
-----
    python scripts/play_human.py <checkpoint.pt>                  # human plays P1
    python scripts/play_human.py <checkpoint.pt> --side 2         # human plays P2
    python scripts/play_human.py <checkpoint.pt> --num-sims 800   # stronger agent

Moves are entered in algebraic notation matching the board diagram in
``src/morris_rl/env/board.py`` :

    a7 -- d7 -- g7
    |    b6 -- d6 -- f6    |
    |    |    c5 -- d5 -- e5    |    |
    a4 - b4 - c4    e4 - f4 - g4
    |    |    c3 -- d3 -- e3    |    |
    |    b2 -- d2 -- f2    |
    a1 -- d1 -- g1

  Placement     : type a position label, e.g. "a7"
  Movement      : type "src dst" or "src->dst", e.g. "a7 d7" or "a7->a4"
  Capture       : type a position label (the opponent piece to remove)
  Quit          : "q" or Ctrl-C
  Show actions  : "?" lists the legal moves and their action indices
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Make repo src/ importable when running as `python scripts/play_human.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for replay_game import

from morris_rl.env.board import ACTION_SPACE_SIZE, MOVE_EDGES, NUM_PLACE_CAPTURE_ACTIONS
from morris_rl.env.rules import (
    GameState,
    Outcome,
    apply_action,
    get_legal_actions,
    initial_state,
    is_terminal,
)
from morris_rl.inference.play import POSITION_LABELS, describe_action
from morris_rl.network.resnet import MorrisResNet
from morris_rl.utils.checkpoints import load_checkpoint

# Reuse the UTF-8 board renderer from the replay tool.
from replay_game import render_board  # type: ignore[import-not-found]


# ---------------------------------------------------------------------------
# Network loading — handles both ResNet and GraphNet checkpoints
# ---------------------------------------------------------------------------


def _load_network(checkpoint_path: Path, device: torch.device):
    payload = load_checkpoint(checkpoint_path)
    cfg = payload["config"]
    net_cfg = cfg["network"]
    enc_cfg = cfg["input_encoding"]
    aux_cfg = cfg.get("aux_heads", {}) or {}
    net_type = net_cfg.get("type", "resnet")

    common_kwargs = dict(
        num_blocks=net_cfg["num_blocks"],
        num_channels=net_cfg["num_channels"],
        policy_head_hidden=net_cfg["policy_head_hidden"],
        value_head_hidden=net_cfg["value_head_hidden"],
        value_head_type=net_cfg.get("value_head_type", "scalar"),
        aux_heads_enabled=bool(aux_cfg.get("enabled", False)),
        aux_head_hidden=int(aux_cfg.get("hidden_size", 64)),
    )

    if net_type == "graphnet":
        from morris_rl.network.graphnet import MorrisGraphNet
        network = MorrisGraphNet(num_planes=11, **common_kwargs)
        from morris_rl.env.encoding_graph import encode_state_graph as encode_fn
    else:
        network = MorrisResNet(num_planes=enc_cfg["num_planes"], **common_kwargs)
        from morris_rl.mcts.search import encode_state as encode_fn

    network.load_state_dict(payload["state_dict"])
    network.eval().to(device)
    return network, encode_fn, int(payload["step"]), net_type


# ---------------------------------------------------------------------------
# Action parsing — human → action index
# ---------------------------------------------------------------------------


_LABEL_TO_POS: dict[str, int] = {label.lower(): i for i, label in enumerate(POSITION_LABELS)}


def _parse_human_input(line: str) -> int | None:
    """Convert a human-typed line into a candidate action index.

    Forms accepted:
      "a7"            placement / capture (single position label)
      "a7 d7"         movement (whitespace-separated)
      "a7->d7"        movement (ASCII arrow)
      "a7→d7"         movement (UTF-8 arrow)
      "12"            raw action index 0-87

    Returns None on parse failure. Caller must still check legality.
    """
    token = line.strip().lower()
    if not token:
        return None
    if token.isdigit():
        idx = int(token)
        return idx if 0 <= idx < ACTION_SPACE_SIZE else None
    for sep in ("->", "→", " "):
        if sep in token:
            parts = [p.strip() for p in token.replace(sep, "|").split("|") if p.strip()]
            if len(parts) == 2 and all(p in _LABEL_TO_POS for p in parts):
                src = _LABEL_TO_POS[parts[0]]
                dst = _LABEL_TO_POS[parts[1]]
                try:
                    edge_idx = MOVE_EDGES.index((src, dst))
                except ValueError:
                    return None
                return NUM_PLACE_CAPTURE_ACTIONS + edge_idx
    if token in _LABEL_TO_POS:
        return _LABEL_TO_POS[token]
    return None


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


_BOLD = "\033[1m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_RESET = "\033[0m"


def _phase_str(state: GameState) -> str:
    if state.must_capture:
        return "MUST CAPTURE"
    if state.pieces_in_hand[state.current_player - 1] > 0:
        return "PLACING"
    return "MOVING"


def _print_game_state(
    state: GameState,
    last_action: int | None,
    pre_state_board=None,
) -> None:
    placed_at = moved_from = moved_to = captured_at = None
    if last_action is not None:
        if last_action < NUM_PLACE_CAPTURE_ACTIONS:
            # Distinguish placement (cell now occupied) vs capture (cell now empty).
            if int(state.board[last_action]) == 0:
                captured_at = last_action
            else:
                placed_at = last_action
        else:
            moved_from, moved_to = MOVE_EDGES[last_action - NUM_PLACE_CAPTURE_ACTIONS]

    print()
    print(render_board(
        state.board,
        moved_from=moved_from, moved_to=moved_to,
        placed_at=placed_at, captured_at=captured_at,
    ))
    print()
    print(
        f"  Phase: {_phase_str(state)}    "
        f"Turn: P{state.current_player}    "
        f"P1 hand: {state.pieces_in_hand[0]}    P2 hand: {state.pieces_in_hand[1]}    "
        f"halfmove: {state.total_halfmoves}"
    )


def _print_legal(state: GameState) -> None:
    legal = get_legal_actions(state)
    print(f"  Legal actions ({len(legal)}) :")
    for a in legal:
        print(f"    [{a:3d}]  {describe_action(a, must_capture=state.must_capture)}")


# ---------------------------------------------------------------------------
# Turn handlers
# ---------------------------------------------------------------------------


def _human_turn(state: GameState) -> int:
    legal = set(get_legal_actions(state))
    while True:
        try:
            line = input(f"  {_BOLD}Your move (?=help, q=quit): {_RESET}")
        except (EOFError, KeyboardInterrupt):
            print("\n  bye.")
            sys.exit(0)
        cmd = line.strip().lower()
        if cmd == "q":
            print("  bye.")
            sys.exit(0)
        if cmd == "?":
            _print_legal(state)
            continue
        action = _parse_human_input(line)
        if action is None:
            print(f"  {_RED}couldn't parse '{line}'. Try 'a7' or 'a7 d7' or '?' for help.{_RESET}")
            continue
        if action not in legal:
            print(
                f"  {_RED}{describe_action(action, must_capture=state.must_capture)}"
                f" is not legal here.{_RESET}  '?' lists legal moves."
            )
            continue
        return action


_ARGMAX_TEMP = 1e-6  # ctree rejects temperature=0; this is "essentially argmax"


def _network_turn(search, state: GameState) -> int:
    """Pick the agent's move via MCTS (near-zero temperature = argmax)."""
    action, visit_probs = search.run(state, temperature=_ARGMAX_TEMP, add_noise=False)

    visit_t = torch.from_numpy(visit_probs)
    top_k = min(3, int((visit_t > 0).sum().item()))
    if top_k > 0:
        top_vals, top_idx = torch.topk(visit_t, top_k)
        print(f"  {_GREEN}Agent considered:{_RESET}")
        for v, a in zip(top_vals.tolist(), top_idx.tolist()):
            marker = "→" if a == action else " "
            print(f"    {marker} [{int(a):3d}]  {describe_action(int(a), must_capture=state.must_capture)}"
                  f"   visit_frac={v:.3f}")
    try:
        root_v = search.root_value(state)
        print(f"  agent root value (P{state.current_player} POV) = {root_v:+.3f}")
    except Exception:
        pass
    return int(action)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def play(checkpoint_path: Path, human_side: int, num_sims: int, device_str: str) -> None:
    device = torch.device(device_str)
    network, encode_fn, step, net_type = _load_network(checkpoint_path, device)
    n_params = sum(p.numel() for p in network.parameters())
    print(f"  Loaded {_BOLD}{net_type}{_RESET} checkpoint  step={step}  params={n_params:,}  device={device_str}")
    print(f"  You play P{human_side} ({'first' if human_side == 1 else 'second'}); agent plays {num_sims} MCTS sims/move.")

    from morris_rl.mcts.search import MorrisSearch
    game_fns = None
    if net_type == "graphnet":
        from morris_rl.env.rules import get_legal_actions as _morris_legal
        game_fns = {
            "encode_state": encode_fn,
            "get_legal_actions": _morris_legal,
        }
    search = MorrisSearch(network, device, num_simulations=num_sims, game_fns=game_fns)

    state = initial_state()
    last_action: int | None = None

    while True:
        done, outcome = is_terminal(state)
        if done:
            _print_game_state(state, last_action)
            if outcome == Outcome.DRAW or outcome is None:
                print(f"\n  {_BOLD}DRAW.{_RESET}\n")
            else:
                winner = int(outcome)
                if winner == human_side:
                    msg, color = "YOU WIN", _GREEN
                else:
                    msg, color = "AGENT WINS", _RED
                print(f"\n  {_BOLD}{color}{msg}{_RESET} (P{winner} took it)\n")
            return

        _print_game_state(state, last_action)

        if state.current_player == human_side:
            action = _human_turn(state)
        else:
            print(f"  {_GREEN}agent (P{state.current_player}) thinking…{_RESET}")
            action = _network_turn(search, state)
            print(f"  agent plays: {_BOLD}{describe_action(action, must_capture=state.must_capture)}{_RESET}")

        last_action = action
        state = apply_action(state, action)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play Morris against a trained network.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("checkpoint", type=Path, help="Path to a .pt checkpoint file.")
    parser.add_argument("--side", type=int, choices=(1, 2), default=1,
                        help="Which player you play (1 = first, 2 = second). Default: 1.")
    parser.add_argument("--num-sims", type=int, default=400,
                        help="MCTS simulations per agent move (higher = stronger / slower). Default: 400.")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()

    if not args.checkpoint.exists():
        sys.exit(f"checkpoint not found: {args.checkpoint}")
    play(args.checkpoint, args.side, args.num_sims, args.device)


if __name__ == "__main__":
    main()
