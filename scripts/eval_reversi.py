"""Arena evaluation for a Reversi checkpoint against baseline agents.

Usage
-----
    # 200 games vs random (both sides)
    uv run python scripts/eval_reversi.py --checkpoint path/to/checkpoint.pt

    # Also test vs greedy (max-flips heuristic)
    uv run python scripts/eval_reversi.py --checkpoint path/to/ckpt.pt --vs-greedy

    # Sweep every checkpoint in a run directory
    uv run python scripts/eval_reversi.py \\
        --checkpoint-dir outputs/2026-05-15/20-51-26/checkpoints \\
        --every 5000

    # Log results to MLflow (same tracking URI as training)
    uv run python scripts/eval_reversi.py --checkpoint ckpt.pt --mlflow-uri file:./mlruns
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

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
)
from morris_rl.env.reversi.encoding import encode_state
from morris_rl.mcts.search import MorrisSearch
from morris_rl.network.resnet import MorrisResNet


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


class ReversiRandomAgent:
    """Uniformly random legal action."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def select_action(self, state: GameState) -> int:
        return int(self._rng.choice(get_legal_actions(state)))


class ReversiGreedyAgent:
    """Greedy: pick the move that flips the most pieces.

    Ties broken randomly to avoid systematic bias from action ordering.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def select_action(self, state: GameState) -> int:
        legal = get_legal_actions(state)
        if legal == [PASS_ACTION]:
            return PASS_ACTION
        player = state.current_player
        before = int(np.sum(state.board == player))
        best_score = -1
        best_actions: list[int] = []
        for action in legal:
            if action == PASS_ACTION:
                continue
            after = int(np.sum(apply_action(state, action).board == player))
            flipped = after - before - 1  # subtract the placed piece itself
            if flipped > best_score:
                best_score = flipped
                best_actions = [action]
            elif flipped == best_score:
                best_actions.append(action)
        return int(self._rng.choice(best_actions))


class ReversiNetworkAgent:
    """AlphaZero agent backed by MCTS + a trained Reversi network."""

    def __init__(self, network: MorrisResNet, device: torch.device, num_simulations: int = 200) -> None:
        action_space_size = network.policy_head.fc2.weight.shape[0]
        _game_fns = {
            "initial_state": initial_state,
            "get_legal_actions": get_legal_actions,
            "apply_action": apply_action,
            "is_terminal": is_terminal,
            "encode_state": encode_state,
            "action_space_size": action_space_size,
        }
        self._search = MorrisSearch(
            network, device, num_simulations=num_simulations, game_fns=_game_fns
        )

    def select_action(self, state: GameState) -> int:
        action, _ = self._search.run(state, temperature=1e-6, add_noise=False)
        return int(action)


# ---------------------------------------------------------------------------
# Arena runner
# ---------------------------------------------------------------------------


@dataclass
class ArenaSummary:
    agent_a_wins: int
    agent_b_wins: int
    draws: int
    agent_a_name: str = "A"
    agent_b_name: str = "B"

    @property
    def total(self) -> int:
        return self.agent_a_wins + self.agent_b_wins + self.draws

    @property
    def win_rate_a(self) -> float:
        return (self.agent_a_wins + 0.5 * self.draws) / max(1, self.total)

    def __str__(self) -> str:
        return (
            f"{self.agent_a_name} vs {self.agent_b_name}: "
            f"{self.agent_a_wins}W / {self.agent_b_wins}L / {self.draws}D  "
            f"({self.win_rate_a:.1%} win-rate) over {self.total} games"
        )


def _play_one(p1_agent, p2_agent) -> Outcome | None:
    """Play a single game, returning the Outcome."""
    state = initial_state()
    while True:
        done, outcome = is_terminal(state)
        if done:
            return outcome
        agents = {PLAYER_1: p1_agent, PLAYER_2: p2_agent}
        action = agents[state.current_player].select_action(state)
        state = apply_action(state, action)


def run_arena(
    agent_a,
    agent_b,
    num_games: int,
    agent_a_name: str = "A",
    agent_b_name: str = "B",
    verbose: bool = True,
) -> ArenaSummary:
    """Play ``num_games`` games, alternating who is Player 1."""
    a_wins = b_wins = draws = 0
    for i in range(num_games):
        a_is_p1 = (i % 2 == 0)
        p1, p2 = (agent_a, agent_b) if a_is_p1 else (agent_b, agent_a)
        outcome = _play_one(p1, p2)
        if outcome is None or outcome == Outcome.DRAW:
            draws += 1
        elif (outcome == Outcome.PLAYER_1_WINS) == a_is_p1:
            a_wins += 1
        else:
            b_wins += 1
        if verbose and (i + 1) % 20 == 0:
            total = i + 1
            wr = (a_wins + 0.5 * draws) / total
            print(f"  [{i+1}/{num_games}]  {agent_a_name}: {a_wins}W/{b_wins}L/{draws}D  ({wr:.1%})")
    return ArenaSummary(a_wins, b_wins, draws, agent_a_name, agent_b_name)


# ---------------------------------------------------------------------------
# Checkpoint loader (mirrors play_reversi.py)
# ---------------------------------------------------------------------------


def load_network(checkpoint_path: str, device: torch.device) -> MorrisResNet:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = payload["state_dict"]

    input_conv_w = sd["input_conv.weight"]
    num_channels = input_conv_w.shape[0]
    num_planes = input_conv_w.shape[1]
    action_space_size = sd["policy_head.fc2.weight"].shape[0]
    num_positions = sd["value_head.fc1.weight"].shape[1]
    policy_head_hidden = sd["policy_head.fc2.weight"].shape[1]
    value_head_hidden = sd["value_head.fc2.weight"].shape[1]
    num_blocks = sum(1 for k in sd if k.startswith("trunk.") and k.endswith(".conv1.weight"))

    net = MorrisResNet(
        num_blocks=num_blocks,
        num_channels=num_channels,
        num_planes=num_planes,
        policy_head_hidden=policy_head_hidden,
        value_head_hidden=value_head_hidden,
        num_positions=num_positions,
        action_space_size=action_space_size,
    ).to(device)
    net.load_state_dict(sd)
    net.eval()
    return net


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _eval_checkpoint(
    ckpt: str,
    device: torch.device,
    num_games: int,
    num_sims: int,
    vs_greedy: bool,
    step_override: int | None,
    mlflow_run_id: str | None,
    mlflow_client,
) -> None:
    ckpt_path = Path(ckpt)
    step = step_override
    if step is None:
        # Try to infer step from filename like checkpoint_00025000.pt
        try:
            step = int(ckpt_path.stem.split("_")[-1])
        except ValueError:
            step = 0

    print(f"\n{'='*60}")
    print(f"  Checkpoint: {ckpt_path.name}  (step {step:,})")
    print(f"  Sims/move: {num_sims}   Games: {num_games}")
    print(f"{'='*60}")

    net = load_network(ckpt, device)
    network_agent = ReversiNetworkAgent(net, device, num_simulations=num_sims)
    random_agent = ReversiRandomAgent(seed=42)

    print(f"\n▶ Network vs Random ({num_games} games)…")
    vs_rand = run_arena(network_agent, random_agent, num_games, "Network", "Random")
    print(f"  {vs_rand}")

    greedy_result = None
    if vs_greedy:
        greedy_agent = ReversiGreedyAgent(seed=42)
        print(f"\n▶ Network vs Greedy ({num_games} games)…")
        greedy_result = run_arena(network_agent, greedy_agent, num_games, "Network", "Greedy")
        print(f"  {greedy_result}")

    if mlflow_client is not None and mlflow_run_id is not None:
        ts = int(torch.tensor(0).item())  # dummy; MLflow uses wall time
        mlflow_client.log_metric(mlflow_run_id, "arena/vs_random_win_rate", vs_rand.win_rate_a, step=step)
        mlflow_client.log_metric(mlflow_run_id, "arena/vs_random_wins", vs_rand.agent_a_wins, step=step)
        mlflow_client.log_metric(mlflow_run_id, "arena/vs_random_draws", vs_rand.draws, step=step)
        if greedy_result is not None:
            mlflow_client.log_metric(mlflow_run_id, "arena/vs_greedy_win_rate", greedy_result.win_rate_a, step=step)
        print(f"\n  ✓ Results logged to MLflow (step {step:,})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reversi arena: checkpoint vs baselines.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", help="Single .pt checkpoint to evaluate.")
    group.add_argument("--checkpoint-dir", help="Directory; evaluates checkpoints matching checkpoint_*.pt.")
    parser.add_argument("--every", type=int, default=1000, help="Step interval when sweeping a directory (default 1000).")
    parser.add_argument("--games", type=int, default=100, help="Games per baseline (default 100, even number recommended).")
    parser.add_argument("--sims", type=int, default=200, help="MCTS simulations per move (default 200).")
    parser.add_argument("--vs-greedy", action="store_true", help="Also evaluate against the greedy max-flips agent.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mlflow-uri", default=None, help="MLflow tracking URI (e.g. file:./mlruns). Logs arena metrics.")
    parser.add_argument("--mlflow-run-id", default=None, help="MLflow run ID to log into. Required when --mlflow-uri is set.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    print(f"Device: {device}")

    mlflow_client = None
    if args.mlflow_uri:
        import mlflow
        mlflow_client = mlflow.tracking.MlflowClient(args.mlflow_uri)

    if args.checkpoint:
        _eval_checkpoint(
            args.checkpoint, device, args.games, args.sims,
            args.vs_greedy, None, args.mlflow_run_id, mlflow_client,
        )
    else:
        ckpt_dir = Path(args.checkpoint_dir)
        checkpoints = sorted(ckpt_dir.glob("checkpoint_*.pt"))
        # Filter to multiples of --every, plus the final checkpoint
        def _keep(p: Path) -> bool:
            try:
                step = int(p.stem.split("_")[-1])
                return step % args.every == 0
            except ValueError:
                return "final" in p.stem
        checkpoints = [p for p in checkpoints if _keep(p)]
        print(f"Found {len(checkpoints)} checkpoints to evaluate in {ckpt_dir}")
        for ckpt in checkpoints:
            _eval_checkpoint(
                str(ckpt), device, args.games, args.sims,
                args.vs_greedy, None, args.mlflow_run_id, mlflow_client,
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
