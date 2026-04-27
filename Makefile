.PHONY: train serve web tensorboard dev clean

train:
	uv run python scripts/train.py

serve:
	uv run uvicorn morris_rl.inference.server:app --reload --port 8000

web:
	cd web && npm run dev

tensorboard:
	uv run tensorboard --logdir outputs --port 6006 --reload_multifile true

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
