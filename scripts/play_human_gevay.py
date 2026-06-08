"""Human-vs-Gévay CLI for Nine Men's Morris (Flying variant).

Same shape as ``play_human_hybrid.py`` but the **movement** engine is
V_Gévay (Phase 2) instead of the binary Phase 1 tablebase.

Why play against Gévay specifically?
------------------------------------
Phase 1 only knows three labels — WIN / LOSS / DRAW. Two drawn positions
look identical to it. Gévay refines this by attaching a continuous
``first_key`` in ``[-30, +30]``:

* ``|first_key| >= 15`` → hard WIN/LOSS class
* ``0 < |first_key| < 15`` → DRAW *under pressure* — the side with the
  positive sign holds positional initiative and will exploit any human
  mistake. ``draws_nz`` in compute_gevay's stats.
* ``first_key == 0`` → DRAW *flat* — no asymmetry, both sides comfortable.
  ``draws_0``.

A Gévay-driven agent picks the move that, after the opponent's reply,
leaves the opponent with the *worst* first_key. In strict draw lines the
network/TB sees nothing to do; the Gévay agent squeezes — playing the
move that keeps the human in the most uncomfortable draw line. You feel
the squeeze in places where Phase 1 would have shrugged.

Tie-breaks on equal first_key use DTW:
- Winning side (our move yields opp.first_key < 0): minimize opp.DTW so
  the win finishes fast.
- Losing side (opp.first_key > 0): maximize opp.DTW to drag out defence.
- Draw (opp.first_key == 0): prefer the longest DTW — stay LONG in the
  unstable line, where the human is most likely to misstep. (User's
  request: "DTW gros pour draw → stay in instability".)

The agent uses the trained network for placement (plies 1-18) and Gévay
1-ply lookahead for movement (plies 19+).

Display:  X (yellow) = player 1, O (blue) = player 2.

Usage
-----
    python3 scripts/play_human_gevay.py <checkpoint.pt>

    # human plays second
    python3 scripts/play_human_gevay.py <checkpoint.pt> --side 2

    # tighter MCTS in placement (slower / harder)
    python3 scripts/play_human_gevay.py <checkpoint.pt> --num-sims 2000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

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
from morris_rl.inference.gevay_client import GevayClient
from morris_rl.inference.play import POSITION_LABELS, describe_action
from morris_rl.network.resnet import MorrisResNet
from morris_rl.utils.checkpoints import load_checkpoint

from replay_game import render_board  # type: ignore[import-not-found]


# ---------------------------------------------------------------------------
# Network loading (same as play_human_hybrid.py)
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
# Input parsing
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
# Display
# ---------------------------------------------------------------------------


_BOLD = "\033[1m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_MAGENTA = "\033[35m"
_RESET = "\033[0m"


def _phase_str(state: GameState) -> str:
    if state.must_capture:
        return "MUST CAPTURE"
    if state.pieces_in_hand[state.current_player - 1] > 0:
        return "PLACING"
    return "MOVING"


def _engine_for(state: GameState) -> str:
    if state.must_capture:
        return "network"
    if state.pieces_in_hand != (0, 0):
        return "network"
    return "gevay"


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


_ARGMAX_TEMP = 1e-6


def _network_turn(search, state: GameState) -> int:
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


def _gevay_lookahead_turn(gevay: GevayClient, state: GameState) -> int:
    """1-ply min-max with V_Gévay as the leaf evaluator.

    For each legal move, apply it, query Gévay on the resulting state
    (which reports the *opponent's* first_key from their STM POV), then
    pick the move that leaves the opponent with the *lowest* first_key.

    Tie-break (after equal first_key):
      opp.first_key < 0  → we're winning. Minimize opp.DTW → fastest win.
      opp.first_key > 0  → we're losing. Maximize opp.DTW → longest defence.
      opp.first_key == 0 → flat draw. Maximize opp.DTW → stay LONG in the
                           unstable line where the human is most likely
                           to misstep (per user request — DTW gros for draws).
    """
    legal = get_legal_actions(state)
    scored: list[tuple[int, int, int]] = []  # (opp_fk, sortable_dtw, action)
    misses: list[int] = []

    for action in legal:
        next_state = apply_action(state, action)
        # If the move triggered a mill, we now sit in a must_capture sub-
        # turn (same player). Gévay can't query must_capture states; the
        # caller will handle the capture choice via the network on the
        # next agent turn. Score the move by Gévay on the post-CURRENT-move
        # position when possible, else recurse one step via Gévay-on-post-
        # capture-state by enumerating capture choices and scoring each.
        if next_state.must_capture:
            # Pick the capture that yields the strongest position for us.
            cap_legal = get_legal_actions(next_state)
            best_cap_score: tuple[int, int] | None = None
            for cap in cap_legal:
                after_cap = apply_action(next_state, cap)
                if (
                    after_cap.must_capture
                    or after_cap.pieces_in_hand != (0, 0)
                ):
                    # Shouldn't happen mid-movement, but bail safely.
                    continue
                done, _ = is_terminal(after_cap)
                if done:
                    # Captured a 3rd piece → opponent terminal LOSS for them
                    # = WIN for us. Treat as best possible outcome.
                    score = (-30, 0)
                else:
                    res = gevay.query(after_cap)
                    if res is None:
                        continue
                    score = (int(res["first_key"]), int(res["dtw"]))
                if best_cap_score is None or score < best_cap_score:
                    best_cap_score = score
            if best_cap_score is None:
                misses.append(action)
                continue
            opp_fk, opp_dtw = best_cap_score
        else:
            done, outcome = is_terminal(next_state)
            if done:
                # Reached a natural terminal directly (e.g. opponent had no
                # legal move). Treat as max-favorable.
                opp_fk, opp_dtw = -30, 0
            else:
                res = gevay.query(next_state)
                if res is None:
                    misses.append(action)
                    continue
                opp_fk = int(res["first_key"])
                opp_dtw = int(res["dtw"])

        # Build the sortable DTW such that lower is always better for us:
        #   opp_fk < 0 (we win): want SMALL opp_dtw (fast).
        #   opp_fk > 0 (we lose): want LARGE opp_dtw (delay) → negate.
        #   opp_fk == 0 (draw): want LARGE opp_dtw → negate.
        if opp_fk < 0:
            sortable_dtw = opp_dtw
        else:
            sortable_dtw = -opp_dtw
        scored.append((opp_fk, sortable_dtw, action))

    if not scored:
        legal_list = list(get_legal_actions(state))
        print(f"  {_RED}gevay miss on all {len(legal_list)} moves "
              f"(subspace not loaded?). Falling back to first legal.{_RESET}")
        return int(legal_list[0])

    scored.sort()  # min first_key, then min sortable_dtw
    best_opp_fk, best_sortable_dtw, best_action = scored[0]

    # Translate back to a human-readable diagnostic. our_score = -opp_fk.
    our_first_key = -best_opp_fk
    if our_first_key >= 15:
        label = f"{_GREEN}WIN{_RESET}"
    elif our_first_key <= -15:
        label = f"{_RED}LOSS{_RESET}"
    elif our_first_key == 0:
        label = f"{_DIM}flat DRAW{_RESET}"
    else:
        label = f"{_MAGENTA}tense DRAW{_RESET}"

    print(f"  {_CYAN}gevay 1-ply lookahead:{_RESET}")
    print(
        f"    our class = {label}   our first_key = {our_first_key:+d}   "
        f"(P{state.current_player} POV, {len(scored)}/{len(legal)} moves scored, "
        f"{len(misses)} misses)"
    )
    # Print top-3 by our score.
    top = scored[: min(3, len(scored))]
    for opp_fk, sortable_dtw, a in top:
        marker = "→" if a == best_action else " "
        our_fk = -opp_fk
        # Recover the real opp_dtw from sortable_dtw.
        opp_dtw = sortable_dtw if opp_fk < 0 else -sortable_dtw
        print(
            f"    {marker} [{a:3d}]  "
            f"{describe_action(a, must_capture=state.must_capture)}"
            f"   our_fk={our_fk:+d}   opp_dtw={opp_dtw}"
        )
    return int(best_action)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def play(
    checkpoint_path: Path,
    human_side: int,
    num_sims: int,
    device_str: str,
    gevay_dir: Path,
    phase1_dir: Path,
) -> None:
    device = torch.device(device_str)
    network, encode_fn, step, net_type = _load_network(checkpoint_path, device)
    n_params = sum(p.numel() for p in network.parameters())
    print(
        f"  Loaded {_BOLD}{net_type}{_RESET} checkpoint  "
        f"step={step}  params={n_params:,}  device={device_str}"
    )

    print(f"  Spawning Gévay client (gevay_dir={gevay_dir}, phase1_dir={phase1_dir})...")
    print(f"  {_DIM}(loads 49 V_Gévay subspaces via mmap; lazy indexer on first "
          f"query — first agent movement move can take ~30-60s.){_RESET}")
    gevay = GevayClient(gevay_dir, phase1_dir)
    print(f"  {_GREEN}Gévay ready.{_RESET}")

    print(
        f"  You play P{human_side} ({'first' if human_side == 1 else 'second'}); "
        f"agent uses NETWORK in placement ({num_sims} sims) and GÉVAY in movement."
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
            if engine == "gevay":
                action = _gevay_lookahead_turn(gevay, state)
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
        description="Play Morris against a network+Gévay agent (Flying variant).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--side", type=int, choices=(1, 2), default=1)
    parser.add_argument("--num-sims", type=int, default=800)
    parser.add_argument(
        "--gevay-dir", type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "tablebase" / "gevay",
    )
    parser.add_argument(
        "--phase1-dir", type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "tablebase" / "flying",
    )
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()

    if not args.checkpoint.exists():
        sys.exit(f"checkpoint not found: {args.checkpoint}")
    if not args.gevay_dir.exists():
        sys.exit(f"gevay dir not found: {args.gevay_dir}")
    if not args.phase1_dir.exists():
        sys.exit(f"phase1 dir not found: {args.phase1_dir}")
    play(
        args.checkpoint, args.side, args.num_sims, args.device,
        args.gevay_dir, args.phase1_dir,
    )


if __name__ == "__main__":
    main()
