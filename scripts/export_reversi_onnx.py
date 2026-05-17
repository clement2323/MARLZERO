"""Export the Reversi ResNet checkpoint to ONNX format.

The exported model takes only the encoded board state (x) as input and returns
raw policy logits and a scalar value.  The legal-action mask is intentionally
excluded from the ONNX graph to avoid -inf / log_softmax portability issues;
apply the mask in Python inside the ORT eval_fn instead.

Usage
-----
    uv run python scripts/export_reversi_onnx.py \\
        --checkpoint outputs/.../checkpoint_00036000.pt \\
        --output model.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from morris_rl.network.resnet import MorrisResNet


class _LogitsWrapper(nn.Module):
    """Thin wrapper that outputs raw policy logits (pre-mask) and scalar value.

    ONNX-friendly: avoids masked_fill(-inf) and log_softmax inside the graph,
    which can cause NaN propagation in some ONNX runtimes.
    """

    def __init__(self, network: MorrisResNet) -> None:
        super().__init__()
        self.network = network

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run forward pass, return (logits, value).

        Args:
            x: Encoded state of shape (batch, num_planes, num_positions).

        Returns:
            logits: Raw policy logits of shape (batch, action_space_size).
            value:  Scalar value estimate of shape (batch,).
        """
        net = self.network
        h = F.relu(net.input_bn(net.input_conv(x)))
        h = net.trunk(h)

        # Policy head — stop before masked_fill + log_softmax
        p = F.relu(net.policy_head.bn(net.policy_head.conv(h)))
        p = p.flatten(start_dim=1)
        p = F.relu(net.policy_head.fc1(p))
        logits = net.policy_head.fc2(p)

        # Value head (scalar)
        value = net.value_head(h)  # shape (batch,)
        return logits, value


def _load_network(checkpoint_path: str, device: torch.device) -> tuple[MorrisResNet, int, int]:
    """Load checkpoint and infer architecture.  Returns (network, action_space_size, step)."""
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    sd = payload["state_dict"]

    num_channels = sd["input_conv.weight"].shape[0]
    num_planes = sd["input_conv.weight"].shape[1]
    action_space_size = sd["policy_head.fc2.weight"].shape[0]
    num_positions = sd["value_head.fc1.weight"].shape[1]
    policy_head_hidden = sd["policy_head.fc2.weight"].shape[1]
    value_head_hidden = sd["value_head.fc2.weight"].shape[1]
    num_blocks = sum(1 for k in sd if k.startswith("trunk.") and k.endswith(".conv1.weight"))

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

    step = payload.get("step", 0)
    print(f"Loaded: ResNet{num_blocks}×{num_channels}, step={step}, "
          f"planes={num_planes}, positions={num_positions}, actions={action_space_size}")
    return network, action_space_size, int(step)


def export(checkpoint_path: str, output_path: str) -> None:
    device = torch.device("cpu")
    network, action_space_size, step = _load_network(checkpoint_path, device)

    num_planes = network.input_conv.weight.shape[1]
    num_positions = network.value_head.fc1.weight.shape[1]

    wrapper = _LogitsWrapper(network).eval()
    dummy_x = torch.zeros(1, num_planes, num_positions, dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        (dummy_x,),
        output_path,
        input_names=["state"],
        output_names=["logits", "value"],
        dynamic_axes={"state": {0: "batch"}, "logits": {0: "batch"}, "value": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )

    # Verify the exported model
    import onnx
    model = onnx.load(output_path)
    onnx.checker.check_model(model)
    size_mb = Path(output_path).stat().st_size / 1024 / 1024
    print(f"Exported to {output_path} ({size_mb:.1f} MB) — ONNX check passed.")

    # Quick numerical check with onnxruntime
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        dummy_np = np.zeros((1, num_planes, num_positions), dtype=np.float32)
        logits_ort, val_ort = sess.run(None, {"state": dummy_np})

        with torch.no_grad():
            logits_pt, val_pt = wrapper(dummy_x)

        logits_diff = float(abs(logits_ort - logits_pt.numpy()).max())
        val_diff = float(abs(val_ort - val_pt.numpy()).max())
        print(f"Numerical check: logits max_diff={logits_diff:.2e}, value max_diff={val_diff:.2e}")
        if logits_diff > 1e-4:
            print("WARNING: large logits discrepancy — check BatchNorm export.")
    except ImportError:
        print("onnxruntime not installed — skipping numerical check.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Reversi checkpoint to ONNX.")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file.")
    parser.add_argument("--output", default="model.onnx", help="Output .onnx path.")
    args = parser.parse_args()
    export(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
