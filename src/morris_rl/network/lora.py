"""LoRA (Low-Rank Adaptation) for MorrisResNet.

Wraps nn.Linear layers with a low-rank update W += scale * B @ A.
The base weights are frozen; only A and B are trained.

Usage::

    network = build_network(cfg)
    trainer.load(checkpoint_path)          # load base weights
    network.add_lora_adapters(rank=8)      # wrap Linear layers
    network.freeze_trunk()                 # freeze all but lora_A / lora_B
    trainer = Trainer(network, ...)        # optimizer sees only trainable params
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """nn.Linear wrapper that adds a trainable low-rank update.

    Forward: output = base_linear(x) + scale * (x @ A.T @ B.T)
    where A ∈ R^{rank×in_features}, B ∈ R^{out_features×rank}.

    Base weights are frozen. Only A and B accumulate gradients.

    Args:
        linear: The pre-trained linear layer to wrap. Its weights are frozen
                in-place; the layer is stored as a submodule so it still
                appears in the parameter tree (with requires_grad=False).
        rank:   Bottleneck dimension of the low-rank factorisation.
        alpha:  Scaling numerator. scale = alpha / rank following the
                original LoRA paper (Hu et al., 2021). Decoupling alpha
                from rank lets you sweep rank without retuning the LR.
    """

    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
    ) -> None:
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.scale = alpha / rank

        in_f = linear.in_features
        out_f = linear.out_features

        # A: small Gaussian — non-zero so the first gradient is non-trivial.
        # B: zero — guarantees the adapter contributes nothing at init, so the
        # network output is identical to the base checkpoint on day 0.
        self.lora_A = nn.Parameter(torch.randn(rank, in_f) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank))

        # Freeze the base weights in-place so the optimizer skips them even if
        # the user forgets to call freeze_trunk().
        linear.weight.requires_grad_(False)
        if linear.bias is not None:
            linear.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute base output + scaled low-rank update.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        base_out = self.linear(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        return base_out + self.scale * lora_out

    def extra_repr(self) -> str:
        return f"rank={self.rank}, scale={self.scale:.3f}"
