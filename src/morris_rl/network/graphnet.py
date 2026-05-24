"""Graph Neural Network backbone for Nine Men's Morris.

Drop-in replacement for MorrisResNet that exploits the Morris board's native
graph structure. Two relations are message-passed at every layer:

    A_adj  — board adjacency (32 undirected edges between movable positions)
    A_mill — mill co-membership (16 mills × 3 nodes, 6 directed arcs each)

Both matrices are constant for Morris no-fly and are stored as buffers, so the
forward pass is just three matmuls + a linear self-loop per block — no PyG
dependency, fully CUDA-Graph compatible.

Input/output API is identical to MorrisResNet so the factory, training loop,
inference server, MCTS, and LoRA helpers all integrate without modification.
The only thing callers must adapt is the encoding function (use
encode_state_graph instead of encode_state) — handled by the search.encode_state
dispatcher.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from morris_rl.env.board import (
    ACTION_SPACE_SIZE as _DEFAULT_ACTION_SPACE_SIZE,
    MILLS,
    MOVE_EDGES,
    NUM_POSITIONS as _DEFAULT_NUM_POSITIONS,
)
from morris_rl.network.heads import (
    AuxScalarHead,
    CategoricalValueHead,
    PolicyHead,
    ValueHead,
)


def _build_adjacency_matrix() -> torch.Tensor:
    """Symmetric, row-normalised adjacency matrix from MOVE_EDGES.

    Returns shape (24, 24) float32. MOVE_EDGES already lists both directions
    of each undirected edge so the constructed A is symmetric.
    """
    A = torch.zeros(_DEFAULT_NUM_POSITIONS, _DEFAULT_NUM_POSITIONS, dtype=torch.float32)
    for src, dst in MOVE_EDGES:
        A[src, dst] = 1.0
    # Row-normalise (D^-1 A); each row sums to 1 → message is the mean of
    # neighbours' features, scale-invariant to node degree.
    row_sum = A.sum(dim=1, keepdim=True).clamp(min=1.0)
    return A / row_sum


def _build_mill_matrix() -> torch.Tensor:
    """Row-normalised mill co-membership matrix from MILLS.

    For each mill (a, b, c) we add the 6 directed arcs a↔b, b↔c, a↔c. The
    result is shape (24, 24) with row sums normalised so each entry is the
    mean over mill-co-members.
    """
    A = torch.zeros(_DEFAULT_NUM_POSITIONS, _DEFAULT_NUM_POSITIONS, dtype=torch.float32)
    for mill in MILLS:
        for u in mill:
            for v in mill:
                if u != v:
                    A[u, v] = 1.0
    row_sum = A.sum(dim=1, keepdim=True).clamp(min=1.0)
    return A / row_sum


class GraphConvBlock(nn.Module):
    """One layer of heterogeneous message passing with residual + BN.

    msg_adj   = A_adj  @ x       # average of board neighbours
    msg_mill  = A_mill @ x       # average of mill co-members
    out       = self_lin(x) + adj_lin(msg_adj) + mill_lin(msg_mill)
    out       = ReLU(BN(out)) + residual (x)

    The matmuls A_* @ x are computed with the buffers passed at forward time
    so the block stays game-agnostic to a tile of GraphNet sharing the same
    adjacency once registered on the parent module.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.self_lin = nn.Linear(dim, dim)
        self.adj_lin = nn.Linear(dim, dim)
        self.mill_lin = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)

    def forward(
        self,
        x: torch.Tensor,
        a_adj: torch.Tensor,
        a_mill: torch.Tensor,
    ) -> torch.Tensor:
        """Args:
            x:      (batch, NUM_POSITIONS, dim)
            a_adj:  (NUM_POSITIONS, NUM_POSITIONS) — board adjacency, row-norm
            a_mill: (NUM_POSITIONS, NUM_POSITIONS) — mill co-members, row-norm
        Returns:
            (batch, NUM_POSITIONS, dim) — same shape, with residual added.
        """
        residual = x
        msg_adj = a_adj @ x       # (batch, 24, dim)
        msg_mill = a_mill @ x     # (batch, 24, dim)
        out = self.self_lin(x) + self.adj_lin(msg_adj) + self.mill_lin(msg_mill)
        # BatchNorm1d expects (batch, channels, length); we have (batch, length, channels).
        out = out.transpose(1, 2)
        out = self.bn(out)
        out = out.transpose(1, 2)
        return F.relu(out + residual)


class MorrisGraphNet(nn.Module):
    """Heterogeneous GNN trunk + standard policy/value/aux heads.

    Input:  (batch, num_planes, NUM_POSITIONS)
    Output: same tuple format as MorrisResNet (2/3/4/5-tuple depending on flags)

    Note on input shape: the constructor signature matches MorrisResNet for
    drop-in compatibility, but internally the trunk operates on
    (batch, NUM_POSITIONS, num_channels). The forward pass transposes the
    incoming tensor exactly once at the entry. This keeps the rest of the
    pipeline (encoder, inference server, replay buffer) able to use either
    layout consistently — see encode_state_graph which already produces the
    (1, NUM_POSITIONS, num_planes) layout, and MCTS encode_state which
    transposes to match the legacy resnet expectation on demand.
    """

    def __init__(
        self,
        num_blocks: int,
        num_channels: int,
        num_planes: int,
        policy_head_hidden: int,
        value_head_hidden: int,
        value_head_type: str = "scalar",
        num_positions: int = _DEFAULT_NUM_POSITIONS,
        action_space_size: int = _DEFAULT_ACTION_SPACE_SIZE,
        aux_heads_enabled: bool = False,
        aux_head_hidden: int = 64,
    ) -> None:
        super().__init__()
        self._num_channels = num_channels
        self._num_planes = num_planes

        # Static graph topology — never trained, moves with .to(device).
        self.register_buffer("A_adj", _build_adjacency_matrix())
        self.register_buffer("A_mill", _build_mill_matrix())

        # Input projection from F_NODE → num_channels per node.
        self.input_proj = nn.Linear(num_planes, num_channels)
        # Use BN over (batch, num_channels, 24) — same convention as the trunk.
        self.input_bn = nn.BatchNorm1d(num_channels)

        self.blocks = nn.ModuleList([GraphConvBlock(num_channels) for _ in range(num_blocks)])

        # Heads are unchanged from the ResNet path. They consume (batch, num_channels,
        # NUM_POSITIONS) so we transpose at the end of the trunk.
        self.policy_head = PolicyHead(
            num_channels, num_positions, action_space_size, policy_head_hidden
        )
        self._value_head_type = value_head_type
        if value_head_type == "categorical":
            self.value_head: ValueHead | CategoricalValueHead = CategoricalValueHead(
                num_channels, num_positions, value_head_hidden
            )
        else:
            self.value_head = ValueHead(num_channels, num_positions, value_head_hidden)

        self._aux_heads_enabled = aux_heads_enabled
        if aux_heads_enabled:
            self.mill_diff_head: AuxScalarHead | None = AuxScalarHead(
                num_channels, num_positions, aux_head_hidden
            )
            self.pieces_diff_head: AuxScalarHead | None = AuxScalarHead(
                num_channels, num_positions, aux_head_hidden
            )
        else:
            self.mill_diff_head = None
            self.pieces_diff_head = None

    # ------------------------------------------------------------------
    # LoRA — identical mechanism as MorrisResNet (scans nn.Linear submodules)
    # ------------------------------------------------------------------

    def add_lora_adapters(self, rank: int = 8, alpha: float = 16.0) -> None:
        """Replace every nn.Linear in the network with a LoRA wrapper.

        Same contract as MorrisResNet.add_lora_adapters: must be called after
        loading the base weights and before freeze_trunk().
        """
        from morris_rl.network.lora import LoRALinear

        for name, module in list(self.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            parts = name.split(".")
            parent = self
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], LoRALinear(module, rank=rank, alpha=alpha))

    def freeze_trunk(self) -> None:
        """Freeze every parameter not named lora_A / lora_B."""
        for name, param in self.named_parameters():
            if "lora_A" not in name and "lora_B" not in name:
                param.requires_grad_(False)

    # ------------------------------------------------------------------
    # Forward — matches MorrisResNet signature exactly
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        legal_mask: torch.Tensor,
        return_value_logits: bool = False,
        return_aux: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        """Run the GNN forward pass.

        Args:
            x: Encoded state tensor of shape (batch, num_planes, NUM_POSITIONS) —
               same convention as MorrisResNet. Internally transposed once at
               entry. The legacy encode_state in mcts.search routes Morris
               inputs through encode_state_graph which produces (batch, 24,
               num_planes); a transpose layer in the dispatcher swaps to the
               (batch, num_planes, 24) layout that this forward accepts. This
               way both encoders feed the same module signature.
            legal_mask: (batch, ACTION_SPACE_SIZE) bool, masking illegal actions.
            return_value_logits, return_aux: same flags as MorrisResNet.

        Returns:
            Same tuple shapes as MorrisResNet — 2/3/4/5-tuple depending on flags.
        """
        # x comes in as (batch, num_planes, NUM_POSITIONS) (drop-in shape).
        # Transpose once to (batch, NUM_POSITIONS, num_planes) for the trunk.
        x = x.transpose(1, 2).contiguous()

        # Input projection per node + BN.
        x = self.input_proj(x)                       # (batch, 24, C)
        # BN expects (batch, C, length).
        x = x.transpose(1, 2)
        x = F.relu(self.input_bn(x))
        x = x.transpose(1, 2)                        # (batch, 24, C)

        # Message-passing blocks share the registered topology buffers.
        for block in self.blocks:
            x = block(x, self.A_adj, self.A_mill)

        # Heads expect (batch, C, 24).
        x = x.transpose(1, 2).contiguous()

        log_policy = self.policy_head(x, legal_mask)

        if self._value_head_type == "categorical":
            scalar, logits = self.value_head(x)      # type: ignore[misc]
        else:
            scalar = self.value_head(x)
            logits = None

        if return_aux:
            if self._aux_heads_enabled:
                mill_pred = self.mill_diff_head(x)       # type: ignore[misc]
                pieces_pred = self.pieces_diff_head(x)   # type: ignore[misc]
            else:
                mill_pred = None
                pieces_pred = None
            if return_value_logits:
                return log_policy, scalar, logits, mill_pred, pieces_pred  # type: ignore[return-value]
            return log_policy, scalar, mill_pred, pieces_pred  # type: ignore[return-value]

        if return_value_logits:
            return log_policy, scalar, logits  # type: ignore[return-value]
        return log_policy, scalar
