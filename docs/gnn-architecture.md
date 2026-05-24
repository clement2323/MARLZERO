# MorrisGraphNet — Graph-aware backbone for Nine Men's Morris

## Why a GNN

The Morris board is a **graph**, not a 1D sequence. The 24 positions are arranged in three concentric rings connected by 4 spokes; movement is constrained to **32 undirected edges** along these lines; mill detection involves **16 specific 3-tuples** of positions. The original `MorrisResNet` ignored all of this: it applied `Conv1d(kernel=3)` over the 24 positions sorted by index. The model had to **learn the topology from data** — wasting capacity and slowing convergence.

Symptom: with `MorrisResNet 6×64` (165 k params), `policy_loss` plateaued around 1.82 after 4 k steps and the agent reached only ~62 % win rate vs random — far below the 95 %+ expected of an AlphaZero net at this scale.

`MorrisGraphNet` makes the graph structure first-class:
- the **adjacency** (where you can move) and the **mill graph** (which positions co-occur in winning lines) are baked into the message-passing operator
- the **state encoding** carries 4 additional structural features so the network doesn't have to rediscover them from data

Same parameter budget as the small ResNet baseline (~300 k vs the 165 k of `6×64`), with an inductive bias that matches the problem.

## What the network sees

### Encoding (11 planes per position)

Defined in [`src/morris_rl/env/encoding_graph.py`](../src/morris_rl/env/encoding_graph.py). Shape `(1, 11, 24)` — same convention as the legacy 7-plane encoding so the rest of the pipeline (replay buffer, shared-memory inference server, symmetry augmentation) keeps working unchanged.

| Plane | Sémantique | Origine | Position-dependent ? |
|-------|---|---|---|
| 0 | own pieces | `board == current_player` | yes |
| 1 | opponent pieces | `board == opponent` | yes |
| 2 | own hand fraction | `hand[player] / 9` broadcast | no (scalar) |
| 3 | opponent hand fraction | `hand[opp] / 9` broadcast | no (scalar) |
| 4 | phase == PLACING | one-hot broadcast | no (scalar) |
| 5 | phase == MOVING | one-hot broadcast | no (scalar) |
| 6 | must_capture flag | bool broadcast | no (scalar) |
| **7** | **own threats** : count of mills containing this position with 2 own + 1 empty, normalised /2 | derived | yes |
| **8** | **opp threats** : same for opponent | derived | yes |
| **9** | **node degree** : `len(ADJACENCY[i]) / 4` ∈ {0.5, 0.75, 1.0} | static | yes (but D4-invariant) |
| **10** | **ring index** : 0.0 outer / 0.5 middle / 1.0 inner | static | yes (but D4-invariant) |

The 7 legacy planes keep their exact semantics — `placement / movement / capture sub-turn / player turn` are encoded identically to the old ResNet path, so all downstream code (game logic, MCTS, replay) is unaffected.

Plane **7** is the single most impactful addition. It tells the network "if I play here, do I form a mill?" — a one-step lookahead that the conv1d trunk previously had to learn from end-of-game value targets. Plane 8 is its defensive mirror.

Planes 9-10 are static (constant per position) but make the topology **first-class**: every layer sees the board's ring/spoke structure without having to infer it from co-occurrences.

### Adjacency injected via two relations

Two row-normalised `[24, 24]` matrices are registered as buffers in `MorrisGraphNet`:

- **`A_adj`** — board adjacency. Built from [`MOVE_EDGES`](../src/morris_rl/env/board.py) (64 directed arcs covering the 32 undirected board edges). Row-normalised: each row is the uniform distribution over a node's neighbours.

- **`A_mill`** — mill co-membership. Built from [`MILLS`](../src/morris_rl/env/board.py). For each mill `(a, b, c)`, the 6 directed pairs `(a,b), (b,a), (b,c), (c,b), (a,c), (c,a)` get an entry. Row-normalised the same way.

Both matrices are **static** (same for every state, every batch) and stored on-device via `register_buffer`. Cost per layer: two `(B, 24, 24) @ (24, D)` matmuls — trivial on CPU let alone GPU.

## Trunk architecture

Defined in [`src/morris_rl/network/graphnet.py`](../src/morris_rl/network/graphnet.py).

```
input: (batch, 11, 24)           ─┐
  ↓ transpose → (batch, 24, 11)   │  one-shot at entry
  ↓ Linear(11 → C) + BN + ReLU    │
  ┌─────────────────────────────┐ │
  │  GraphConvBlock #1          │ │
  │  GraphConvBlock #2          │ │  trunk: 6× blocks at C=128
  │      ...                    │ │
  │  GraphConvBlock #6          │ │
  └─────────────────────────────┘ │
  ↓ transpose → (batch, C, 24)    │  heads expect channels-first
  ↓ PolicyHead / ValueHead / Aux  ┘
```

### `GraphConvBlock`

```python
def forward(x, A_adj, A_mill):
    # x: (batch, 24, C)
    msg_adj  = A_adj  @ x                  # average of board neighbours' features
    msg_mill = A_mill @ x                  # average of mill co-members' features
    out  = self_lin(x) + adj_lin(msg_adj) + mill_lin(msg_mill)
    out  = ReLU(BN(out))
    return out + residual_x                # skip connection
```

Three linear maps per block:
- `self_lin` keeps a node's own information
- `adj_lin` mixes in info from movement neighbours
- `mill_lin` mixes in info from mill co-members

The **residual connection** is critical — without it a 6-layer GCN oversmooths to a constant. See "Avoiding oversmoothing" below.

### Heads (unchanged from ResNet)

The trunk output is transposed back to `(batch, C, 24)` so the existing `PolicyHead`, `CategoricalValueHead`, `AuxScalarHead` from [`heads.py`](../src/morris_rl/network/heads.py) work without modification. This means:

- LoRA wrapping works (same `nn.Linear` modules to scan)
- The categorical-vs-scalar value head is configurable
- The mill_diff / pieces_diff aux heads plug in the same way

The drop-in compatibility means **anything that worked with `MorrisResNet` works with `MorrisGraphNet`**: factory, trainer, MCTS, inference server, evaluator, LoRA fine-tuning workflow.

## Expressivity at equal size

> *"Le MorrisGraphNet même à taille égale devrait être beaucoup plus expressif, non ?"*

Two distinct questions are mixed in there.

### Raw expressivity (worst-case)

At equal parameter count, an arbitrary MLP can in theory approximate the same set of functions as a GNN. The Universal Approximation Theorem doesn't care about inductive bias. **From this pure-capacity angle, both networks have similar ceiling.**

### Sample efficiency on a structured problem

In practice the question that matters is: *given a fixed compute and data budget, which architecture learns faster?*

For Morris the target function **respects the graph structure**:
- Legal moves are 1-hop in the adjacency graph
- Mill detection is 1-hop in the mill graph
- Captures, threats, blockades are all 1-3 hops away from local features

A graph-aware backbone **doesn't waste any capacity** learning that "position 0 and position 7 are adjacent" or "position 1, 9, 17 form a spoke". The ResNet had to discover those by gradient descent — and probably never fully did, since the conv1d filter only sees a kernel-3 window in the linear position order.

The expected gains (empirical, to validate):
- **Faster `policy_loss` decay** in the first 5 k steps
- **Higher win-rate vs random** at every checkpoint up to ~10 k steps
- **Better generalization** to game states under-represented in self-play (e.g. unusual mid-game positions) because the inductive bias does the heavy lifting

The asymptote at very-many-steps may be similar — the ResNet eventually approximates the right structure given enough data. But our compute budget is bounded, so the inductive bias is real value.

### Could we go further on expressivity ?

Yes, in three orthogonal directions if needed later:

1. **Graph Attention (GAT)** instead of fixed `A_adj`/`A_mill`. Each edge gets a learnable attention weight per layer. ~2× params per block, gains in non-uniform aggregation (some neighbours matter more than others, learned per state).
2. **Edge features**. Right now edges are unweighted. We could attach features like "edge is part of an active mill threat" or "edge would form a new mill if traversed". Encoded as a third relation matrix `A_active_threat` recomputed per state.
3. **Jumping knowledge**. Concat features from each intermediate layer before the heads. Captures multi-scale information without forcing the final layer to summarise everything.

None of these is urgent. Start with the current 6×128 (~315 k params) and only add complexity if learning stalls.

## Avoiding oversmoothing

### What is oversmoothing

After many message-passing layers, a vanilla GCN collapses every node's features toward the **same vector** (the graph Laplacian's dominant eigenvector, ignoring node identity). Documented in:
- Li et al. 2018, "Deeper Insights into Graph Convolutional Networks"
- Oono & Suzuki 2019, "Graph Neural Networks Exponentially Lose Expressive Power"

On a 24-node graph with diameter ≈ 6, this kicks in around layer 8-10. Symptoms: training loss looks fine but validation collapses, or all node embeddings look identical at the trunk output.

### What protects us already

The current architecture has **three structural defenses** that make oversmoothing essentially a non-issue at `num_blocks=6`:

1. **Residual connections** in every block (`out + residual`). Even at infinite depth a residual GCN converges to a non-trivial fixed point (each layer can be the identity). This is the single most important fix and we have it.

2. **`self_lin` term** in each block. The block isn't pure message passing — there's an explicit "keep my own features" path that prevents the trivial uniform fixed point.

3. **Depth matched to graph diameter**. 6 layers gives every node a 6-hop receptive field — enough to cover the whole 24-node graph once. Going deeper than 6-7 layers adds zero new information *and* starts to oversmooth.

### Config recommendations

| Goal | `num_blocks` | `num_channels` | Approx params | Comment |
|---|---|---|---|---|
| Smoke test, fast iteration | 2-3 | 32 | ~5-10 k | Verifies wiring, can't really play |
| **Default Morris CPU** | **6** | **128** | **~315 k** | Recommended — matches graph diameter, ~ResNet 6×64 budget |
| Compare with old ResNet 10×128 | 6 | 144 | ~400 k | Slight bump for fairness |
| Aggressive ceiling | 8 | 128 | ~430 k | Beyond graph diameter, monitor closely |
| Oversmoothing risk zone | 10+ | any | — | Avoid without jumping knowledge or per-layer learnable mixing |

**Default config (commit on `gnn-backbone`):**
```yaml
network:
  type: graphnet
  num_blocks: 6
  num_channels: 128
  policy_head_hidden: 64
  value_head_hidden: 64
  value_head_type: categorical
```

### Diagnostic if you suspect oversmoothing

Watch `train/value_std` in TensorBoard. If it drops near zero **and** `value_loss` stops decreasing, the trunk has likely collapsed to producing identical per-node features → identical predictions for very different boards. Combine with a quick check:

```python
# in a Python shell, with a checkpoint loaded:
trunk_out = net._run_trunk(x)        # (1, 24, C)
print(trunk_out.std(dim=1).mean())   # std across nodes — should be > ~0.1
```

If the per-node std collapses below ~0.05, reduce `num_blocks` to 4 or add jumping knowledge.

## Symmetries (D4)

The Morris board has 8 symmetries (4 rotations × 2 reflections). The trunk is **D4-equivariant by construction** — verified in [`tests/network/test_graphnet.py::test_trunk_d4_equivariance`](../tests/network/test_graphnet.py): permuting the input along the node axis is equivalent to permuting the trunk output.

**Caveat**: the policy and value heads are *not* D4-equivariant because they flatten the position axis into a single linear layer. So overall the network output is **not** D4-invariant. Data augmentation in the replay buffer (multiplying every sample by the 8 symmetries) is still useful.

If you want full equivariance later, replace the heads with permutation-equivariant variants (sum-pool over nodes for the value head; per-node + per-edge MLP for the policy head, with shared weights across symmetric positions).

## What stays unchanged

Explicitly **not touched** by this work:

- `MorrisResNet` and all checkpoints trained on it — `network.type=resnet` still works
- **Reversi pipeline** — Reversi config uses `network.type=resnet` and 3 input planes, unaffected. Tests pass.
- `scripts/replay_game.py` — doesn't call the network, works as-is
- `rules.py`, `board.py`, `symmetries.py` semantics — only `transform_encoded_state` was generalised to accept any plane count
- `LoRA` wrapping — works recursively on `nn.Linear`, picks up everything inside `GraphConvBlock`

## How to use

### Train from scratch (CPU)

```bash
uv run python scripts/train.py \
  network.type=graphnet \
  network.num_blocks=6 \
  network.num_channels=128 \
  network.value_head_type=categorical \
  aux_heads.enabled=true \
  self_play.inference_mode=per_worker_cpu \
  self_play.num_workers=10 \
  mcts.num_simulations_train=400 \
  device=cpu mlflow.enabled=true
```

### Train on GPU (when CUDA TSC bug is sorted)

Drop the `inference_mode` / `device` overrides — the default `shared_gpu` / `cuda` kicks in. The inference server (`inference_server.py`) was generalised in this work to handle 11-plane states via the `num_planes` parameter, so GPU training works the same way.

### Evaluate against random

Same `scripts/evaluate.py` as before — it reads the network type from the checkpoint config and rebuilds the right architecture automatically:

```bash
uv run python scripts/evaluate.py \
  outputs/.../checkpoints/checkpoint_00005000.pt \
  --opponent random --num-games 100 --num-simulations 800 --device cpu
```

### Replay self-play traces

Same `scripts/replay_game.py` — unchanged, works with or without GraphNet.

## References

- **MorrisGraphNet implementation**: [`src/morris_rl/network/graphnet.py`](../src/morris_rl/network/graphnet.py)
- **Graph-aware encoding**: [`src/morris_rl/env/encoding_graph.py`](../src/morris_rl/env/encoding_graph.py)
- **Tests**: [`tests/network/test_graphnet.py`](../tests/network/test_graphnet.py) — 18 tests including D4 trunk equivariance
- **Factory dispatch**: [`src/morris_rl/network/factory.py`](../src/morris_rl/network/factory.py)
- **Board topology**: [`src/morris_rl/env/board.py`](../src/morris_rl/env/board.py) — `ADJACENCY`, `MOVE_EDGES`, `MILLS`
