# Nine Men's Morris RL

An AlphaZero-style reinforcement learning agent that plays Nine Men's Morris at near-perfect level. Built as a portfolio piece demonstrating production-quality ML engineering: self-play training, MCTS-guided search, a ResNet policy/value network, and a web demo where humans can play against the agent.

## Quickstart

```bash
# Install dependencies (requires uv)
uv sync --dev

# Run the test suite
uv run pytest

# Install pre-commit hooks
uv run pre-commit install
```

## Project structure

```
src/morris_rl/     Main package
configs/           Hydra YAML configs
tests/             Unit and integration tests
scripts/           Training, evaluation, and demo entry points
docs/decisions/    Architectural decision records
web/               React frontend for the demo
```

## Status


---

## How to launch the web app, backend server, and TensorBoard

### 1. Launch the web app (React frontend)

```bash
cd web
npm install  # or pnpm install
npm run dev
# App runs on http://localhost:5173
```

### 2. Launch the backend server (FastAPI inference)

```bash
uvicorn src.morris_rl.inference.server:app --reload
# Set env vars as needed:
#   MODEL_CHECKPOINT=checkpoints/checkpoint_00001000.pt
#   NUM_SIMULATIONS=200
#   DEVICE=cuda
```

### 3. Launch TensorBoard

```bash
tensorboard --logdir tensorboard
# Open http://localhost:6006
```

### 4. Checkpoints and logging

- Checkpoints are saved in `checkpoints/` every 1000 gradient steps (configurable via `training.checkpoint_interval` in `configs/default.yaml`).
- TensorBoard logs are written to `tensorboard/` and are enabled by default during training.
- If you see `events.out.tfevents.*` files in `tensorboard/`, logging is working.

---
3 rhow to reprendre une run a partir dunccheckpoint ?