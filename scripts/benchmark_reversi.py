"""Benchmark CPU inference configurations for the Reversi MCTS server.

Tests all combinations of MCTS backend × inference backend and prints a
table with median time per move and speedup relative to the baseline
(ptree + eager PyTorch).

Usage
-----
    uv run python scripts/benchmark_reversi.py \\
        --checkpoint outputs/2026-05-16/23-41-09/checkpoints/checkpoint_00036000.pt \\
        --num-sims 200 \\
        --num-moves 5
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morris_rl.env.reversi.encoding import encode_state
from morris_rl.env.reversi.rules import (
    apply_action,
    get_legal_actions,
    initial_state,
    is_terminal,
)
from morris_rl.mcts.search import MorrisSearch
from morris_rl.network.resnet import MorrisResNet

_ONNX_TMP = "/tmp/_reversi_bench.onnx"
_ONNX_INT8_TMP = "/tmp/_reversi_bench_int8.onnx"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_network(path: str, device: torch.device) -> tuple[MorrisResNet, int]:
    payload = torch.load(path, map_location=device, weights_only=False)
    sd = payload["state_dict"]

    num_channels = sd["input_conv.weight"].shape[0]
    num_planes = sd["input_conv.weight"].shape[1]
    action_space_size = sd["policy_head.fc2.weight"].shape[0]
    num_positions = sd["value_head.fc1.weight"].shape[1]
    policy_head_hidden = sd["policy_head.fc2.weight"].shape[1]
    value_head_hidden = sd["value_head.fc2.weight"].shape[1]
    num_blocks = sum(
        1 for k in sd if k.startswith("trunk.") and k.endswith(".conv1.weight")
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
    network.load_state_dict(sd)
    network.eval()
    print(f"  Loaded: ResNet{num_blocks}×{num_channels} step={payload.get('step','?')} "
          f"planes={num_planes} positions={num_positions} actions={action_space_size}")
    return network, action_space_size


def _make_reversi_fns(action_space_size: int) -> dict:
    return {
        "initial_state": initial_state,
        "get_legal_actions": get_legal_actions,
        "apply_action": apply_action,
        "is_terminal": is_terminal,
        "encode_state": encode_state,
        "action_space_size": action_space_size,
    }


def _bench(search: MorrisSearch, num_moves: int) -> float:
    """Return median seconds-per-move over num_moves moves from the start."""
    state = initial_state()
    times: list[float] = []
    for _ in range(num_moves):
        done, _ = is_terminal(state)
        if done:
            break
        t0 = time.perf_counter()
        action, _ = search.run(state, temperature=1e-6, add_noise=False)
        times.append(time.perf_counter() - t0)
        state = apply_action(state, int(action))
    return float(np.median(times)) if times else float("nan")


def _export_onnx(network: MorrisResNet, action_space_size: int, path: str) -> None:
    """Export to ONNX (raw logits, no mask inside graph)."""
    import torch.nn.functional as F

    class _Wrapper(torch.nn.Module):
        def __init__(self, net: MorrisResNet) -> None:
            super().__init__()
            self.net = net

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            net = self.net
            h = F.relu(net.input_bn(net.input_conv(x)))
            h = net.trunk(h)
            p = F.relu(net.policy_head.bn(net.policy_head.conv(h)))
            p = p.flatten(start_dim=1)
            p = F.relu(net.policy_head.fc1(p))
            logits = net.policy_head.fc2(p)
            value = net.value_head(h)
            return logits, value

    num_planes = network.input_conv.weight.shape[1]
    num_positions = network.value_head.fc1.weight.shape[1]
    wrapper = _Wrapper(network).eval()
    dummy = torch.zeros(1, num_planes, num_positions)
    torch.onnx.export(
        wrapper, (dummy,), path,
        input_names=["state"],
        output_names=["logits", "value"],
        dynamic_axes={"state": {0: "batch"}, "logits": {0: "batch"}, "value": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Reversi CPU inference configs.")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint.")
    parser.add_argument("--num-sims", type=int, default=200, help="MCTS simulations per move.")
    parser.add_argument("--num-moves", type=int, default=5, help="Moves to time per config.")
    args = parser.parse_args()

    device = torch.device("cpu")
    print("Loading checkpoint...")
    network, action_space_size = _load_network(args.checkpoint, device)
    reversi_fns = _make_reversi_fns(action_space_size)

    results: list[tuple[str, float, float]] = []
    baseline_t: float = 1.0

    # ------------------------------------------------------------------
    # 1. Baseline: ptree + eager PyTorch
    # ------------------------------------------------------------------
    print("\n[1/5] baseline (ptree + eager PyTorch)...")
    search = MorrisSearch(network, device, num_simulations=args.num_sims, game_fns=reversi_fns)
    baseline_t = _bench(search, args.num_moves)
    results.append(("baseline (ptree+eager)", baseline_t, 1.0))
    print(f"      {baseline_t:.2f}s/move")

    # ------------------------------------------------------------------
    # 2. torch.compile (ptree + compiled network)
    # ------------------------------------------------------------------
    print("\n[2/5] torch.compile (ptree + compiled)...")
    try:
        net_compiled = torch.compile(
            copy.deepcopy(network), backend="inductor", mode="reduce-overhead"
        )
        # Warmup — avoids measuring JIT compilation time
        print("      warming up (first call compiles, ~10-30s)...")
        num_planes = network.input_conv.weight.shape[1]
        num_positions = network.value_head.fc1.weight.shape[1]
        dummy_x = torch.zeros(1, num_planes, num_positions)
        dummy_mask = torch.ones(1, action_space_size, dtype=torch.bool)
        with torch.no_grad():
            net_compiled(dummy_x, dummy_mask)
        search_compiled = MorrisSearch(
            net_compiled, device, num_simulations=args.num_sims, game_fns=reversi_fns
        )
        t_compile = _bench(search_compiled, args.num_moves)
        results.append(("torch.compile", t_compile, baseline_t / t_compile))
        print(f"      {t_compile:.2f}s/move  ×{baseline_t/t_compile:.2f}")
    except Exception as exc:
        print(f"      FAILED: {exc}")
        results.append(("torch.compile", float("nan"), float("nan")))

    # ------------------------------------------------------------------
    # 3. INT8 dynamic quantization
    # ------------------------------------------------------------------
    print("\n[3/5] INT8 dynamic quantization...")
    try:
        import torch.quantization as tq

        net_int8 = tq.quantize_dynamic(
            copy.deepcopy(network), {torch.nn.Linear}, dtype=torch.qint8
        )
        search_int8 = MorrisSearch(
            net_int8, device, num_simulations=args.num_sims, game_fns=reversi_fns
        )
        t_int8 = _bench(search_int8, args.num_moves)
        results.append(("INT8 quantization", t_int8, baseline_t / t_int8))
        print(f"      {t_int8:.2f}s/move  ×{baseline_t/t_int8:.2f}")
    except Exception as exc:
        print(f"      FAILED: {exc}")
        results.append(("INT8 quantization", float("nan"), float("nan")))

    # ------------------------------------------------------------------
    # 4. ONNX Runtime (ptree + ORT)
    # ------------------------------------------------------------------
    print("\n[4/5] ONNX Runtime (ptree + ORT)...")
    try:
        print("      exporting ONNX...")
        _export_onnx(network, action_space_size, _ONNX_TMP)
        from morris_rl.inference.ort_eval import make_ort_eval_fn

        ort_eval = make_ort_eval_fn(
            _ONNX_TMP, encode_state, get_legal_actions, action_space_size, num_threads=2
        )
        search_ort = MorrisSearch(
            eval_fn=ort_eval, num_simulations=args.num_sims, game_fns=reversi_fns
        )
        t_ort = _bench(search_ort, args.num_moves)
        results.append(("ONNX Runtime", t_ort, baseline_t / t_ort))
        print(f"      {t_ort:.2f}s/move  ×{baseline_t/t_ort:.2f}")
    except Exception as exc:
        print(f"      FAILED: {exc}")
        results.append(("ONNX Runtime", float("nan"), float("nan")))

    # ------------------------------------------------------------------
    # 5. ONNX Runtime + INT8 static quantization
    # ------------------------------------------------------------------
    print("\n[5/5] ONNX Runtime + INT8 quantization...")
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(_ONNX_TMP, _ONNX_INT8_TMP, weight_type=QuantType.QInt8)
        from morris_rl.inference.ort_eval import make_ort_eval_fn

        ort_eval_int8 = make_ort_eval_fn(
            _ONNX_INT8_TMP, encode_state, get_legal_actions, action_space_size, num_threads=2
        )
        search_ort_int8 = MorrisSearch(
            eval_fn=ort_eval_int8, num_simulations=args.num_sims, game_fns=reversi_fns
        )
        t_ort_int8 = _bench(search_ort_int8, args.num_moves)
        results.append(("ONNX + INT8", t_ort_int8, baseline_t / t_ort_int8))
        print(f"      {t_ort_int8:.2f}s/move  ×{baseline_t/t_ort_int8:.2f}")
    except Exception as exc:
        print(f"      FAILED: {exc}")
        results.append(("ONNX + INT8", float("nan"), float("nan")))

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print(f"\n{'=' * 62}")
    print(f"{'Config':<26} {'median s/move':>14} {'speedup':>10} {'demo-ready':>10}")
    print(f"{'-' * 62}")
    for name, t, speedup in results:
        if np.isnan(t):
            print(f"{name:<26} {'FAILED':>14} {'—':>10} {'✗':>10}")
        else:
            viable = "✓" if t <= 10.0 else "✗"
            print(f"{name:<26} {t:>14.2f} {speedup:>9.2f}x {viable:>10}")
    print(f"{'=' * 62}")

    # ------------------------------------------------------------------
    # Recommendation
    # ------------------------------------------------------------------
    viable = [(n, t, s) for n, t, s in results if not np.isnan(t) and t <= 10.0]
    if viable:
        best = min(viable, key=lambda r: r[1])
        print(f"\n✓ Recommended: {best[0]} ({best[1]:.2f}s/move, ×{best[2]:.2f})")
        print("  → update INFERENCE_BACKEND / Dockerfile accordingly, then push to HF.")
    else:
        fastest = min((r for r in results if not np.isnan(r[1])), key=lambda r: r[1], default=None)
        if fastest:
            print(f"\n✗ No config achieves ≤10s/move at {args.num_sims} sims.")
            print(f"  → Best: {fastest[0]} ({fastest[1]:.2f}s). Consider reducing NUM_SIMULATIONS.")
        else:
            print("\n✗ All configs failed.")


if __name__ == "__main__":
    main()
