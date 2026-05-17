.PHONY: help train train-tmux train-tmux-kill serve web tensorboard mlflow-ui play play-reversi serve-reversi web-reversi dev dashboard clean
.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# Self-documenting help — each target's `## description` annotation is parsed
# below and printed by `make help`. Group headings start with `##@`.
# ---------------------------------------------------------------------------

help:  ## Show this help message
	@awk 'BEGIN {FS = ":.*?## "} \
	  /^##@/ {printf "\n\033[1m%s\033[0m\n", substr($$0, 5); next} \
	  /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}' \
	  $(MAKEFILE_LIST)

##@ Training

train:  ## Run training in foreground (Ctrl-C to stop)
	uv run python scripts/train.py

# Long-running production training in a detached tmux session — survives
# closing the terminal. Override TRAIN_SESSION to pick a different session
# name (e.g. TRAIN_SESSION=tier3 make train-tmux).
TRAIN_SESSION ?= train

train-tmux:  ## Run training in a detached tmux session (canonical 100k-step config)
	@if tmux has-session -t $(TRAIN_SESSION) 2>/dev/null; then \
	  echo "tmux session '$(TRAIN_SESSION)' already active. Attach with: tmux attach -t $(TRAIN_SESSION)"; \
	  exit 1; \
	fi
	tmux new-session -d -s $(TRAIN_SESSION) "uv run python scripts/train.py \
	  mcts.num_simulations_train=250 \
	  self_play.num_workers=6 \
	  training.min_buffer_size=2000 \
	  training.checkpoint_interval=250 \
	  training.total_steps=100000 \
	  mlflow.enabled=true 2>&1 | tee train.log"
	@echo "Training started in tmux session '$(TRAIN_SESSION)'."
	@echo "  Attach: tmux attach -t $(TRAIN_SESSION)"
	@echo "  Detach: Ctrl-b then d"
	@echo "  Kill:   make train-tmux-kill"

train-tmux-kill:  ## Stop the detached training session
	tmux kill-session -t $(TRAIN_SESSION) 2>/dev/null || true
	@echo "Session '$(TRAIN_SESSION)' stopped."

##@ Demo (play against the agent in the browser)

# Auto-pick the most recent checkpoint unless the user passes one explicitly:
#   make play                                # latest checkpoint, or minimax fallback if none
#   MODEL_CHECKPOINT=path/to/file.pt make play
#   MINIMAX_DEPTH=5 make play                # force the minimax fallback at depth 5
MODEL_CHECKPOINT ?= $(shell ls -1t outputs/*/*/checkpoints/checkpoint_*.pt 2>/dev/null | head -1)
MINIMAX_DEPTH ?= 3

play:  ## Launch backend + frontend (uses latest checkpoint, falls back to minimax depth N)
	@if lsof -ti :8000 >/dev/null 2>&1; then \
	  echo "Port 8000 already in use (PID $$(lsof -ti :8000)). Kill with: lsof -ti :8000 | xargs kill"; \
	  exit 1; \
	fi
	@if lsof -ti :5173 >/dev/null 2>&1; then \
	  echo "Port 5173 already in use (PID $$(lsof -ti :5173)). Kill with: lsof -ti :5173 | xargs kill"; \
	  exit 1; \
	fi
	@if [ -z "$(MODEL_CHECKPOINT)" ]; then \
	  echo "No checkpoint found — backend will fall back to MinimaxAgent(depth=$(MINIMAX_DEPTH))"; \
	else \
	  echo "Checkpoint: $(MODEL_CHECKPOINT)"; \
	fi
	@echo "Backend:    http://127.0.0.1:8000"
	@echo "Frontend:   http://127.0.0.1:5173"
	@echo "Ctrl-C to stop both."
	@trap 'kill 0' INT; \
	  MODEL_CHECKPOINT="$(MODEL_CHECKPOINT)" MINIMAX_DEPTH="$(MINIMAX_DEPTH)" $(MAKE) serve & \
	  $(MAKE) web & \
	  wait

serve:  ## Run the FastAPI inference backend alone (uses MODEL_CHECKPOINT env)
	uv run uvicorn morris_rl.inference.server:app --reload --port 8000

web:  ## Run the React/Vite frontend alone (expects backend on :8000)
	cd web && npm run dev

# Auto-detect the most recent Reversi checkpoint by scanning each run's
# hydra config for num_planes=3 (Morris uses 7). Falls back to empty if none.
MODEL_CHECKPOINT_REVERSI ?= $(shell \
  for ckpt in $$(ls -1t outputs/*/*/checkpoints/checkpoint_*.pt 2>/dev/null); do \
    run_dir=$$(dirname $$(dirname $$ckpt)); \
    if grep -q "num_planes: 3" "$$run_dir/.hydra/config.yaml" 2>/dev/null; then \
      echo $$ckpt; break; \
    fi; \
  done)

play-reversi:  ## Launch Reversi backend (:8001) + frontend (:5174) with latest Reversi checkpoint
	@if lsof -ti :8001 >/dev/null 2>&1; then \
	  echo "Port 8001 already in use (PID $$(lsof -ti :8001)). Kill with: lsof -ti :8001 | xargs kill"; \
	  exit 1; \
	fi
	@if lsof -ti :5174 >/dev/null 2>&1; then \
	  echo "Port 5174 already in use (PID $$(lsof -ti :5174)). Kill with: lsof -ti :5174 | xargs kill"; \
	  exit 1; \
	fi
	@if [ -z "$(MODEL_CHECKPOINT_REVERSI)" ]; then \
	  echo "No Reversi checkpoint found (none of outputs/*/*/.hydra/config.yaml has num_planes: 3)."; \
	  exit 1; \
	fi
	@echo "Checkpoint: $(MODEL_CHECKPOINT_REVERSI)"
	@echo "Backend:    http://127.0.0.1:8001"
	@echo "Frontend:   http://127.0.0.1:5174"
	@echo "Ctrl-C to stop both."
	@trap 'kill 0' INT; \
	  MODEL_CHECKPOINT="$(MODEL_CHECKPOINT_REVERSI)" $(MAKE) serve-reversi & \
	  $(MAKE) web-reversi & \
	  wait

serve-reversi:  ## Run the Reversi FastAPI backend alone on :8001 (uses MODEL_CHECKPOINT env)
	uv run uvicorn morris_rl.inference.reversi_server:app --reload --port 8001

web-reversi:  ## Run the Reversi React/Vite frontend alone (port 5174)
	cd web_reversi && npm run dev

##@ Analysis

dashboard:  ## Launch Streamlit training dashboard (port 8501, dark theme, auto-refresh)
	uv run streamlit run scripts/dashboard.py --server.port 8501

##@ Monitoring

mlflow-ui:  ## Read-only MLflow UI on file:./mlruns (~50 MB RAM)
	.venv/bin/mlflow ui --backend-store-uri file:./mlruns --port 5000

tensorboard:  ## TensorBoard UI on outputs/ (port 6006)
	uv run tensorboard --logdir outputs --port 6006 --reload_multifile true

##@ All-in-one

dev:  ## Train + backend + frontend + tensorboard in parallel (Ctrl-C kills all)
	@trap 'kill 0' INT; \
	$(MAKE) train & \
	$(MAKE) serve & \
	$(MAKE) web & \
	$(MAKE) tensorboard & \
	wait

##@ Maintenance

clean:  ## Remove training outputs and caches
	rm -rf outputs/ multirun/ wandb/ .pytest_cache/ web/node_modules/.vite
