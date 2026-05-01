.PHONY: train train-tmux train-tmux-kill serve web tensorboard mlflow-ui dev clean

train:
	uv run python scripts/train.py

# Long-running production training in a detached tmux session — survives
# closing the terminal. Encodes the canonical flag set so we don't retype
# them every time. Override TRAIN_SESSION to run multiple in parallel
# (not recommended on 30 GB RAM).
TRAIN_SESSION ?= train

train-tmux:
	@if tmux has-session -t $(TRAIN_SESSION) 2>/dev/null; then \
	  echo "tmux session '$(TRAIN_SESSION)' already active. Attach with: tmux attach -t $(TRAIN_SESSION)"; \
	  exit 1; \
	fi
	tmux new-session -d -s $(TRAIN_SESSION) "uv run python scripts/train.py \
	  mcts.num_simulations_train=250 \
	  self_play.num_workers=6 \
	  self_play.worker_recycle_games=0 \
	  self_play.worker_max_rss_mb=0 \
	  training.min_buffer_size=2000 \
	  training.checkpoint_interval=500 \
	  training.total_steps=100000 \
	  mlflow.enabled=true 2>&1 | tee train.log"
	@echo "Training started in tmux session '$(TRAIN_SESSION)'."
	@echo "  Attach: tmux attach -t $(TRAIN_SESSION)"
	@echo "  Detach: Ctrl-b then d"
	@echo "  Kill:   make train-tmux-kill"

train-tmux-kill:
	tmux kill-session -t $(TRAIN_SESSION) 2>/dev/null || true
	@echo "Session '$(TRAIN_SESSION)' stopped."

serve:
	uv run uvicorn morris_rl.inference.server:app --reload --port 8000

web:
	cd web && npm run dev

tensorboard:
	uv run tensorboard --logdir outputs --port 6006 --reload_multifile true

# Read-only MLflow UI for the file-based store. Costs ~50 MB RAM
# vs ~1.5 GB for `mlflow server` — that overhead OOM'd a training run
# previously, so default is `mlflow ui` and we never spawn the full server.
mlflow-ui:
	uv run mlflow ui --backend-store-uri file:./mlruns --port 5000

# All-in-one: training + inference + web demo + tensorboard, in parallel.
# Ctrl+C kills the whole group via the INT trap.
dev:
	@trap 'kill 0' INT; \
	$(MAKE) train & \
	$(MAKE) serve & \
	$(MAKE) web & \
	$(MAKE) tensorboard & \
	wait

clean:
	rm -rf outputs/ multirun/ wandb/ .pytest_cache/ web/node_modules/.vite
