# CLAUDE.md — Nine Men's Morris RL Project

## What this project is

A from-scratch, production-quality implementation of an AlphaZero-style reinforcement learning agent that plays **Nine Men's Morris** (also called Mills, Merrils, Mühle) at near-perfect level. The project is intended both as a serious technical exercise and as a portfolio piece showcased to ML/RL employers.

The deliverable is **not just an agent**. It is a complete project: training pipeline, evaluation harness, web demo where humans can play against the agent, technical blog post, and reproducible setup.

The user (project owner) has a strong conceptual understanding of AlphaZero, MCTS, self-play, the actor-critic structure of the policy/value heads, and the training loop dynamics. They are not looking for tutoring on the algorithm itself. They are looking for an **expert-level engineering partner** to build the system.

## Hardware target

The training and inference are designed for a specific machine:

- **GPU**: RTX 4070 with 8 GB VRAM (Ada Lovelace architecture, strong FP16)
- **CPU**: AMD Zen 4, 16 cores / 32 threads, 5.4 GHz boost, 64 MB L3 cache
- **RAM**: 64 GB recommended
- **OS**: Linux preferred, Windows acceptable via WSL2

The 8 GB VRAM is the binding constraint. Network size and batch sizes must respect this. Do not propose architectures that require more.

CPU is plentiful — the training pipeline should leverage 12-14 parallel self-play workers. CPU-bound MCTS code should be in C++ via LightZero's `mcts_ctree=True` mode, not pure Python.

## What "Nine Men's Morris" means here

**Variant: no flying.** This project removes the standard flying rule. Two players, nine pieces each, two game phases:

1. **Placement** (moves 1-18): players alternately place their nine pieces on empty board positions.
2. **Movement** (after placement): players move pieces along board lines to adjacent empty positions. A player at 3 pieces stays in this phase (no jumping to arbitrary squares).

Forming a line of three pieces ("a mill") allows the player to remove one of the opponent's pieces. Pieces inside an opponent's mill cannot be removed unless all opponent pieces are in mills.

A player loses by being reduced to 2 pieces or having no legal move on their turn (the latter is now reachable as early as 3 pieces because adjacency can fully block them).

The game is **solved**: Gasser (1993) proved it is a draw with perfect play. Tablebases for endgame positions exist and are used in this project as ground truth.

## Architectural principles (non-negotiable)

These principles drive every implementation decision:

1. **Modularity**. The environment, network, MCTS, training loop, evaluation harness, and web demo must be cleanly separable. Each component must be testable in isolation. No global state. No tangled dependencies.

2. **Readability over cleverness**. Prefer explicit, well-named code over compact tricks. Every non-obvious decision must have a comment explaining why. A new contributor should be able to read any module in 10 minutes and understand it.

3. **Configuration over code**. All hyperparameters, paths, and behavioral switches must be in YAML config files (using Hydra or a simple equivalent). Code should never have magic numbers or hardcoded paths.

4. **Reproducibility**. Every training run must be reproducible from a config file + a seed. Checkpoints must be versioned. Logs must capture the full config used.

5. **Testability**. Unit tests for the environment are mandatory (rules, transitions, terminal states, mill detection, capture rules, fly transition). Integration tests for the training pipeline. CI on GitHub Actions must run all tests on every push.

6. **Observability**. Training and evaluation must be heavily instrumented. TensorBoard for losses, win rates, and learning curves. Custom dashboards for game-level statistics (game length distribution, mill formation rate, capture rate by phase, draw rate over time). The user wants to watch the system learn.

7. **Modifiability**. The architecture must support easy experimentation: swapping the network architecture (ResNet → Transformer → custom), the algorithm (AlphaZero → Gumbel AlphaZero → MuZero), or the encoding scheme. Any of these changes should require modifying one or two files, not the whole codebase.

## Project structure

```
nine-mens-morris-rl/
├── README.md                    Public-facing project overview with demo gif
├── CLAUDE.md                    This file
├── pyproject.toml               Modern Python packaging
├── .pre-commit-config.yaml      Black, ruff, mypy
├── .github/workflows/ci.yml     Run tests, linting, type checks
│
├── configs/                     YAML configs (Hydra-compatible)
│   ├── default.yaml             Base config
│   ├── network/                 Network variants
│   ├── training/                Training schedules
│   └── eval/                    Evaluation setups
│
├── src/morris_rl/               Main package
│   ├── env/
│   │   ├── board.py             Board representation
│   │   ├── rules.py             Rules engine, legal moves, terminal detection
│   │   ├── encoding.py          State → tensor planes encoding
│   │   ├── symmetries.py        8-fold dihedral group transforms
│   │   └── tablebase.py         Loader for Gasser-style endgame tablebase
│   │
│   ├── network/
│   │   ├── resnet.py            Default ResNet (10 blocks, 128 channels)
│   │   ├── heads.py             Policy and value heads
│   │   └── factory.py           build_network(config) entry point
│   │
│   ├── mcts/                    Wraps LightZero's MCTS, exposes clean API
│   │   └── search.py
│   │
│   ├── training/
│   │   ├── self_play.py         Self-play worker logic
│   │   ├── replay_buffer.py     FIFO buffer with symmetry augmentation
│   │   ├── trainer.py           Training loop
│   │   ├── promotion.py         Arena evaluation and weight promotion
│   │   └── tablebase_anchor.py  Tablebase value anchoring during training
│   │
│   ├── eval/
│   │   ├── arena.py             Two-agent tournament
│   │   ├── metrics.py           Elo, win rate, exact tablebase agreement
│   │   └── baselines.py         Random agent, minimax-N agents
│   │
│   ├── inference/
│   │   ├── server.py            FastAPI server for web demo
│   │   └── play.py              Single-position evaluation, MCTS visualization
│   │
│   └── utils/
│       ├── logging.py           Structured logging
│       ├── seeding.py           Reproducible seeding
│       └── checkpoints.py       Versioned checkpoint I/O
│
├── tests/
│   ├── env/                     Mandatory: full coverage of rules engine
│   ├── network/
│   ├── mcts/
│   └── integration/             End-to-end smoke tests
│
├── notebooks/                   Analysis, ablations, plots for blog post
│
├── web/                         React + canvas frontend for the demo
│
├── docs/
│   ├── architecture.md          Why these choices
│   ├── training_log.md          Real run notes, what worked, what didn't
│   └── decisions/               One file per major architectural decision
│
└── scripts/
    ├── train.py                 Main training entry point
    ├── evaluate.py              Standalone evaluation
    ├── play_human.py            CLI for human vs agent
    └── serve.py                 Launch web demo
```

## Tech stack

- **Language**: Python 3.11+
- **DL framework**: PyTorch (latest stable). No TensorFlow, no JAX for this project.
- **RL framework**: LightZero (OpenDILab). Default to its AlphaZero implementation. Do not reimplement MCTS from scratch unless absolutely necessary; use LightZero's C++ MCTS via `mcts_ctree=True`.
- **Config**: Hydra
- **Logging**: TensorBoard for metrics, `loguru` for structured text logs
- **Testing**: pytest, pytest-cov for coverage
- **Linting/formatting**: ruff (linting + formatting), mypy (strict type checking)
- **Web demo backend**: FastAPI
- **Web demo frontend**: React + TypeScript, canvas for board rendering. Hosted on Vercel or HuggingFace Spaces.
- **CI**: GitHub Actions

Avoid adding dependencies casually. Every new dependency requires justification in the relevant decision doc.

## Algorithm: target configuration for first run

```yaml
algorithm: alphazero          # not muzero, not gumbel — start simple
network:
  type: resnet
  num_blocks: 10
  num_channels: 128
  policy_head_hidden: 64
  value_head_hidden: 64

input_encoding:
  num_planes: 7
  # Plane 0: current player's pieces
  # Plane 1: opponent's pieces
  # Plane 2: pieces still in current player's hand (broadcast scalar / 9)
  # Plane 3: pieces still in opponent's hand
  # Plane 4-5: phase one-hot (placing, moving) — no FLYING in this variant
  # Plane 6: must-capture flag (after a mill is formed)

mcts:
  num_simulations_train: 200
  num_simulations_eval: 800
  num_simulations_demo: 5000
  c_puct: 1.5
  dirichlet_alpha: 0.3
  dirichlet_epsilon: 0.25

self_play:
  num_workers: 12
  inference_batch_size: 32
  temperature_schedule:
    moves_0_to_10: 1.0
    moves_10_plus: 0.0
  tree_reuse: false           # disabled in training, enabled in eval/demo

training:
  batch_size: 256
  learning_rate: 0.001
  weight_decay: 1e-4
  replay_buffer_size: 500_000
  symmetry_augmentation: true   # 8x via dihedral group
  mixed_precision: true         # FP16 via torch.cuda.amp
  updates_per_collected_game: 4

promotion:
  mode: arena                   # 'arena' for first runs, 'continuous' later
  eval_games: 100
  win_rate_threshold: 0.55

tablebase:
  enabled: true
  anchor_value_targets: true    # critical accelerator
  use_at_inference: true        # exact play in covered endgame positions
```

These values are starting points. They must be tunable via Hydra overrides.

## Milestones (each is a self-contained Claude Code task)

The project decomposes into independent milestones. Each one should be deliverable in one Claude Code session and independently testable.

### Milestone 1 — Bootstrap and environment

Set up the repo with proper packaging, CI, pre-commit hooks, basic logging, and seeding utilities. No game logic yet. Deliverable: `pip install -e .` works, `pytest` runs (with no tests yet), `pre-commit run --all-files` passes, CI is green.

### Milestone 2 — Rules engine

Implement the Nine Men's Morris environment. Board representation, legal move generation, mill detection, capture rules (including the "remove from mill is illegal unless all opponent pieces are in mills" exception), phase transition (placement → movement; no flying in this variant), terminal state detection (≤2 pieces or no legal move), draw detection (3-fold repetition, halfmove cap).

Deliverable: comprehensive unit test suite covering every rule and edge case. The test suite must be runnable in under 5 seconds. A self-play game between two random agents must run end-to-end without crashes for 1000 games.

### Milestone 3 — State encoding and symmetries

Implement the 8-plane encoding and the 8-fold dihedral symmetry group transforms (4 rotations × 2 reflections). Symmetry transforms must be invertible and must respect both board state and policy distribution.

Deliverable: tests verifying that for any state s, applying a symmetry then its inverse returns s; that legal moves transform consistently with state; that policy targets transform consistently.

### Milestone 4 — Network architecture

Implement the ResNet network with policy and value heads, action masking on the policy output (illegal moves get probability zero), and the factory pattern for swapping architectures.

Deliverable: a forward pass on a batch of 256 states completes in under 10ms on the target GPU. Network output shapes are correct. Action masking is verified.

### Milestone 5 — MCTS integration

Wrap LightZero's MCTS with a clean API. Verify Dirichlet noise injection, temperature-based action selection, visit count extraction. Confirm `mcts_ctree=True` works on the target system.

Deliverable: given a network and a state, `MCTS.run(state, num_sims=800)` returns a visit count distribution. Performance: at least 3000 simulations/second per worker on the target CPU.

### Milestone 6 — Self-play data generation

Implement the parallel self-play worker pool with shared GPU inference server. Implement the FIFO replay buffer with symmetry augmentation.

Deliverable: 12 workers can run concurrently, sustained throughput of at least 50 self-play games per minute, no memory leaks over a 10-minute run.

### Milestone 7 — Training loop

Implement the trainer that pulls minibatches from the buffer, computes the combined policy + value loss, and updates the network. Mixed-precision training. Gradient clipping. Learning rate schedule.

Deliverable: training loop runs concurrently with self-play, loss curves trend downward over a 1-hour smoke test, TensorBoard shows expected metrics.

### Milestone 8 — Evaluation harness

Implement the arena tournament for promotion decisions, the baseline agents (random, minimax-3, minimax-5, minimax-7), and the Elo tracking against a pool of past checkpoints.

Deliverable: after a smoke training run, the arena correctly identifies that the trained agent beats random, beats minimax-3, and tracks an Elo curve over time.



### Milestone 9 — First end-to-end training run

Run the full pipeline for 24-72 hours. Document everything: hyperparameters, hardware utilization, loss curves, Elo curves, surprises. Produce checkpoints at regular intervals.

Deliverable: a trained agent that beats minimax-7 reliably. A `training_log.md` with detailed observations.

### Milestone 10 — Web demo

FastAPI backend that exposes a `/play` endpoint. React frontend with a canvas-rendered board, drag-and-drop interaction, and a panel showing the agent's MCTS analysis (top-3 candidate moves with their evaluations). Deploy to a public URL.

Deliverable: anyone with a browser can play against the agent. The agent's reasoning is visible. Mobile-responsive.

### Milestone 11 — Polish and showcase

Production-quality README with demo gif, architecture diagram, results table. Technical blog post (2000-3000 words) explaining the project. Public release of trained weights on HuggingFace Hub.

Deliverable: the project is presentable to ML employers. The user can include it on their CV.

## Code style guide

- **Type hints everywhere**. No `Any` unless absolutely necessary. Use `mypy --strict`.
- **Docstrings on all public functions**. Google style. Include shape annotations for tensor inputs/outputs.
- **No abbreviations** in identifiers except universally understood ones (`x`, `y`, `i`, `j`, `lr`, `bs`). `policy_distribution`, not `pol_dist`.
- **Functions under 50 lines**. If longer, decompose.
- **Files under 400 lines**. If longer, split.
- **No comments explaining what the code does** (the code should be self-explanatory). Comments explain **why** a non-obvious choice was made.
- **No commented-out code in commits**. Use git history.
- **Exceptions must be specific**. Never `except Exception` without re-raising. Define custom exception classes for domain errors (e.g., `IllegalMoveError`, `InvalidPhaseTransitionError`).
- **Logging, not print**. Use the structured logger from `utils/logging.py`.

## Testing requirements

- Unit tests for env coverage must be **>95%**.
- Every bug found during development must spawn a regression test before being fixed.
- Tests must be deterministic (no flakiness from randomness — use fixed seeds in tests).
- Integration tests are slower but mandatory for the training pipeline.
- A test marked `@pytest.mark.slow` is acceptable for things like a 60-second smoke training run, but must be excluded from the default test suite.

## What Claude Code should do when asked to implement a milestone

1. **Read this file fully** before making any decision.
2. **Read the most recent decision docs** in `docs/decisions/` if they exist.
3. **Propose the file structure** for the milestone before writing code, and wait for confirmation if anything is unclear.
4. **Write tests alongside implementation**, not after. If asked to implement a function, also write its tests in the same diff.
5. **Update the relevant decision doc** if a non-obvious choice was made. One short markdown file per decision in `docs/decisions/NNN-short-name.md`.
6. **Run the test suite** before declaring a milestone complete. Run `ruff` and `mypy` and fix all issues.
7. **Commit in logical units** with messages that explain the why, not just the what.

## What Claude Code should refuse to do

- Implement MCTS from scratch when LightZero is available. Adapt LightZero, don't reinvent.
- Add machine learning frameworks beyond PyTorch (no JAX, no TensorFlow, no Keras).
- Use Jupyter notebooks for production code. Notebooks are for analysis and visualization only.
- Skip type hints or tests because "it's just experimental." It is not just experimental.
- Make the code "more general" by adding abstractions for cases that don't yet exist. YAGNI applies aggressively.
- Hardcode hyperparameters in Python files. They go in YAML.

## Communication style with the user

The user is technically sharp and prefers direct, concise explanations. Avoid hedging, marketing language, or overly-formal preambles. When making a tradeoff, name both options and explain why one is preferred. When uncertain, say so explicitly and ask. The user values being treated as a peer engineer, not as a customer.

When the user makes a technical mistake, correct it clearly. When they have a good idea, build on it without flattery.

## Out of scope (for now)

The following are explicitly **not** part of this project. Do not propose them unprompted:

- Multi-game support (only Nine Men's Morris, not Lasker Morris or Morabaraba — at least not until milestone 12 is done)
- MuZero or learned dynamics models (the rules are perfect and known)
- Distributed training across multiple machines
- Inference optimization beyond what FastAPI + a single GPU can deliver
- Mobile native apps (web demo is sufficient)
- Anti-cheat or rate limiting on the demo
- Internationalization

## A final note

This project is meant to be **enjoyable** to work on, not just delivered. If a design choice would make the code harder to read or modify just to save 2% of training time, prefer the readable choice. The user is building this as a learning experience and a portfolio piece, not as a commercial product. Engineering elegance matters more than micro-optimization.

When in doubt, ask. When not in doubt, ship.
