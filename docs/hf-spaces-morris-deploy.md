# Morris Zero — Deploy on Hugging Face Spaces + Vercel

Single-source runbook to push the Morris demo to production.
Two endpoints:
- **Backend** (FastAPI + MCTS) → HF Spaces Docker (`https://clem2323-morris-zero.hf.space`)
- **Frontend** (React+Vite SPA) → Vercel (`https://morris-zero.vercel.app` or similar)

## 0. Prerequisites

Local tooling needed:
- Docker daemon running (only for local pre-flight build, optional)
- Node 20+ (for the Vercel deploy, already installed if `web/` was built before)
- `git` (already there)
- Hugging Face account with a **write** token

## 1. Install HF CLI + login

```bash
# Install (only if not present) — the new CLI is `hf` (not `huggingface-cli`)
uv pip install -U huggingface_hub

# Login interactively — paste a WRITE token from
#   https://huggingface.co/settings/tokens   (create one if needed)
uv run hf auth login
```

Verify:
```bash
uv run hf whoami
# Should print your username (e.g. Clem2323)
```

## 2. Upload the champion checkpoint to HF Hub

```bash
# Create the model repo (private or public — public is fine for portfolio)
uv run hf repo create morris-checkpoint --repo-type model

# Upload best.pt + README from the local archive
uv run python -c "
from huggingface_hub import upload_file
upload_file(
    path_or_fileobj='champions/v1_phase3_v2_step10000/best.pt',
    path_in_repo='best.pt',
    repo_id='Clem2323/morris-checkpoint',
    repo_type='model',
)
upload_file(
    path_or_fileobj='champions/v1_phase3_v2_step10000/README.md',
    path_in_repo='README.md',
    repo_id='Clem2323/morris-checkpoint',
    repo_type='model',
)
"
```

Verify in browser: `https://huggingface.co/Clem2323/morris-checkpoint`

## 3. (Optional) Local Docker pre-flight

Test the Docker image builds correctly before pushing it to HF.

```bash
docker build -f Dockerfile.hf.morris -t morris-zero:dev .
docker run --rm -p 7860:7860 -e NUM_SIMULATIONS=50 morris-zero:dev
```

Then in another terminal:
```bash
curl http://localhost:7860/health
curl http://localhost:7860/new-game
```

Expect:
- `/health` → `{"status":"ok", ...}`
- `/new-game` → board state with `legal_actions: [...]`

If the build fails on `lightzero` install, verify the wheel URL at
`https://huggingface.co/Clem2323/reversi-checkpoint/resolve/main/lightzero-0.2.0-cp311-cp311-linux_x86_64.whl`
is reachable (private repo issue?). The wheel is the Othello one, reused here.

## 4. Create + push to HF Space

```bash
# Create an empty Docker-based Space
uv run hf repo create morris-zero --repo-type space --space-sdk docker

# Prepare a deploy branch (keeps main clean — only the files HF needs go to the Space)
git checkout -b hf-deploy

# HF auto-detects "Dockerfile" at the root, so rename our file for the Space
mv Dockerfile.hf.morris Dockerfile

# Commit the deploy state
git add Dockerfile
git commit -m "deploy: HF Spaces Morris zero (champion v1)"

# Add the Space remote and push
git remote add hf-morris https://huggingface.co/spaces/Clem2323/morris-zero
git push hf-morris hf-deploy:main --force

# Switch back to main (Dockerfile.hf.morris stays as you committed it on main)
git checkout main
```

The Space will start building. Watch logs at:
`https://huggingface.co/spaces/Clem2323/morris-zero/logs`

Build takes ~5-10 minutes (torch+mkl+lightzero install).

## 5. Deploy frontend to Vercel

```bash
cd web/
# First time only: install Vercel CLI globally — or use npx if available
npm install -g vercel
# (Or if you prefer not to install: npx vercel --prod)

vercel --prod
# Follow the prompts:
#   - Project name: morris-zero
#   - Framework: Vite (auto-detected)
#   - Build command: npm run build
#   - Output: dist
#   - Use defaults for the rest
```

The `web/.env.production` is auto-loaded by Vite on `npm run build`, so the
deployed bundle will point at `https://clem2323-morris-zero.hf.space`.

## 6. Smoke test production

Once the Space build completes and Vercel deploys:

1. `curl https://clem2323-morris-zero.hf.space/health` → status ok
2. Open Vercel URL in browser → New Game → place a piece → agent responds
3. Open Vercel URL on phone → board scales to viewport, touch works
4. Lose a game (e.g. start a poor placement vs the agent) → loser overlay fires

## 7. Update flow (after retraining)

When a better checkpoint emerges:
1. Save it to `champions/v2_*/best.pt` with a README
2. Re-run the upload step (Step 2) — same `repo_id`, will overwrite `best.pt`
3. **Restart the Space** (HF UI button) so the container re-downloads the checkpoint

No Docker rebuild needed for checkpoint changes; rebuild only when changing the Python code or Dockerfile.

## Common issues

| Symptom | Fix |
|---|---|
| Space build fails on `lightzero` install | Re-check wheel URL accessible, or rebuild a fresh wheel from source against torch 2.5.1+cpu/cp311 |
| Space starts but `/play` errors with "no checkpoint" | Verify HF model repo `morris-checkpoint` has `best.pt` at root; restart Space |
| Frontend can't reach backend (CORS error) | Check `src/morris_rl/inference/server.py` allows `*.vercel.app` in `CORSMiddleware`. Add origin if needed |
| Vercel build fails on TypeScript | `cd web && rm -rf node_modules && npm ci && npm run build` locally first |
| `hf` not found | The CLI lives in the venv. Use `uv run hf ...` (never plain `hf`). Install with `uv pip install -U huggingface_hub` |
