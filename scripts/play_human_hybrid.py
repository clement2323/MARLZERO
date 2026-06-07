"""Human-vs-hybrid CLI for Nine Men's Morris (Flying variant).

The agent uses two engines:

* **Placement** (plies 1-18, hands non-empty, or must-capture sub-turn) →
  the trained network's MCTS, exactly like ``play_human.py``. This is
  where the RL policy shines — it has been trained on millions of
  Gévay-anchored placement positions.
* **Movement** (hands empty, not in must-capture) → direct query to the
  Phase 1 tablebase via ``play_tb --serve``. This is STRICTLY PERFECT —
  the agent never makes a mistake in movement.

Display:  X (yellow) = player 1, O (blue) = player 2.

Usage
-----
    python scripts/play_human_hybrid.py <checkpoint.pt> \\
        --tablebase-dir data/tablebase/flying

    # human plays second:
    python scripts/play_human_hybrid.py <checkpoint.pt> --side 2

    # crank inference sims for harder placement:
    python scripts/play_human_hybrid.py <checkpoint.pt> --num-sims 2000

Notation (input):
    Placement     : type a position label, e.g. "a7"
    Movement      : type "src dst" or "src->dst", e.g. "a7 d7"
    Capture       : type a position label
    Quit          : "q" or Ctrl-C
    Show actions  : "?" lists the legal moves and their action indices
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Make repo src/ importable when running as `python scripts/play_human_hybrid.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for replay_game import

from morris_rl.env.board import ACTION_SPACE_SIZE, MOVE_EDGES, NUM_PLACE_CAPTURE_ACTIONS
from morris_rl.env.rules import (
    GameState,
    Outcome,
    Variant,
    apply_action,
    get_legal_actions,
    initial_state,
    is_terminal,
)
from morris_rl.inference.play import POSITION_LABELS, describe_action
from morris_rl.inference.tablebase_client import (
    TablebaseClient,
    WAVE_DRAW,
    WAVE_LOSS,
    WAVE_WIN,
)
from morris_rl.network.resnet import MorrisResNet
from morris_rl.utils.checkpoints import load_checkpoint

# Reuse the X/O renderer from the replay tool.
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
_DIM = "\033[2m"
_CYAN = "\033[36m"
_RESET = "\033[0m"

_VERDICT_LABEL = {
    WAVE_WIN: f"{_GREEN}WIN{_RESET}",
    WAVE_LOSS: f"{_RED}LOSS{_RESET}",
    WAVE_DRAW: "DRAW",
}


def _phase_str(state: GameState) -> str:
    if state.must_capture:
        return "MUST CAPTURE"
    if state.pieces_in_hand[state.current_player - 1] > 0:
        return "PLACING"
    return "MOVING"


def _engine_for(state: GameState) -> str:
    """Which engine drives the agent at this state?"""
    if state.must_capture:
        return "network"
    if state.pieces_in_hand != (0, 0):
        return "network"
    return "tablebase"


def _print_game_state(state: GameState, last_action: int | None) -> None:
    placed_at = moved_from = moved_to = captured_at = None
    if last_action is not None:
        if last_action < NUM_PLACE_CAPTURE_ACTIONS:
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
        f"P1 hand: {state.pieces_in_hand[0]}    "
        f"P2 hand: {state.pieces_in_hand[1]}    "
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
    """Pick via MCTS (placement / must-capture)."""
    action, visit_probs = search.run(state, temperature=_ARGMAX_TEMP, add_noise=False)

    visit_t = torch.from_numpy(visit_probs)
    top_k = min(3, int((visit_t > 0).sum().item()))
    if top_k > 0:
        top_vals, top_idx = torch.topk(visit_t, top_k)
        print(f"  {_GREEN}network considered:{_RESET}")
        for v, a in zip(top_vals.tolist(), top_idx.tolist()):
            marker = "→" if a == action else " "
            print(
                f"    {marker} [{int(a):3d}]  "
                f"{describe_action(int(a), must_capture=state.must_capture)}"
                f"   visit_frac={v:.3f}"
            )
    try:
        root_v = search.root_value(state)
        print(f"  network root value (P{state.current_player} POV) = {root_v:+.3f}")
    except Exception:
        pass
    return int(action)


def _tablebase_turn(tb: TablebaseClient, state: GameState) -> int:
    """Pick via Phase 1 tablebase (movement, perfect play)."""
    result = tb.query(state)
    if result is None:
        # Should not happen if _engine_for routed us here, but stay defensive:
        # fall back to a random legal move so the game continues.
        legal = get_legal_actions(state)
        print(
            f"  {_RED}tablebase miss (state out of domain) "
            f"— falling back to first legal move{_RESET}"
        )
        return int(legal[0])

    action = int(result["action"])
    verdict = int(result["verdict"])
    dtw = int(result["dtw"])
    top_moves = result["top_moves"][:3]

    print(f"  {_CYAN}tablebase (perfect play):{_RESET}")
    print(f"    verdict = {_VERDICT_LABEL.get(verdict, str(verdict))}   "
          f"DTW = {dtw}   (P{state.current_player} POV)")
    for m in top_moves:
        marker = "→" if int(m["action"]) == action else " "
        print(
            f"    {marker} [{int(m['action']):3d}]  "
            f"{describe_action(int(m['action']), must_capture=state.must_capture)}"
            f"   verdict={_VERDICT_LABEL.get(int(m['verdict']), str(m['verdict']))}"
            f"   DTW={int(m['dtw'])}"
        )
    return action


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def play(
    checkpoint_path: Path,
    human_side: int,
    num_sims: int,
    device_str: str,
    tablebase_dir: Path,
) -> None:
    device = torch.device(device_str)
    network, encode_fn, step, net_type = _load_network(checkpoint_path, device)
    n_params = sum(p.numel() for p in network.parameters())
    print(
        f"  Loaded {_BOLD}{net_type}{_RESET} checkpoint  "
        f"step={step}  params={n_params:,}  device={device_str}"
    )

    print(f"  Spawning tablebase client (dir={tablebase_dir})...")
    tb = TablebaseClient(tablebase_dir)
    print(f"  {_GREEN}tablebase ready.{_RESET}")
    print(
        f"  You play P{human_side} ({'first' if human_side == 1 else 'second'}); "
        f"agent uses NETWORK in placement ({num_sims} sims) and TABLEBASE in movement."
    )
    print(f"  {_DIM}Pieces: P1 = X (yellow), P2 = O (blue).{_RESET}")

    from morris_rl.mcts.search import MorrisSearch
    game_fns = None
    if net_type == "graphnet":
        from morris_rl.env.rules import get_legal_actions as _morris_legal
        game_fns = {
            "encode_state": encode_fn,
            "get_legal_actions": _morris_legal,
        }
    search = MorrisSearch(network, device, num_simulations=num_sims, game_fns=game_fns)

    # FLYING variant is mandatory: the Phase 1 tablebase was computed for
    # Flying, so the agent's movement queries only make sense in that mode.
    state = initial_state(variant=Variant.FLYING)
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
            engine = _engine_for(state)
            print(f"  {_GREEN}agent (P{state.current_player}) thinking via {engine}…{_RESET}")
            if engine == "tablebase":
                action = _tablebase_turn(tb, state)
            else:
                action = _network_turn(search, state)
            print(
                f"  agent plays: {_BOLD}"
                f"{describe_action(action, must_capture=state.must_capture)}{_RESET}"
            )

        last_action = action
        state = apply_action(state, action)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play Morris against a hybrid network+tablebase agent (Flying variant).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "checkpoint", type=Path,
        help="Path to a .pt checkpoint (network trained against V_Gévay).",
    )
    parser.add_argument(
        "--side", type=int, choices=(1, 2), default=1,
        help="Which player you play (1 = first, 2 = second). Default: 1.",
    )
    parser.add_argument(
        "--num-sims", type=int, default=800,
        help="MCTS simulations per network move (placement only). Default: 800.",
    )
    parser.add_argument(
        "--tablebase-dir", type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "tablebase" / "flying",
        help="Phase 1 tablebase directory. Default: data/tablebase/flying/.",
    )
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()

    if not args.checkpoint.exists():
        sys.exit(f"checkpoint not found: {args.checkpoint}")
    if not args.tablebase_dir.exists():
        sys.exit(
            f"tablebase dir not found: {args.tablebase_dir}\n"
            "Build it first with `cargo run --release --bin build_movement` "
            "from morris_tablebase/, or point --tablebase-dir to the right location."
        )
    play(args.checkpoint, args.side, args.num_sims, args.device, args.tablebase_dir)


if __name__ == "__main__":
    main()
