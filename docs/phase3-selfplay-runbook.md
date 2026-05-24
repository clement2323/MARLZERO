# Phase 3 — Self-play fine-tuning : runbook

Rapport opérationnel pour lancer et observer le self-play après le warmup. Toutes les commandes supposent que tu as déjà :

- Un dataset warmup généré : `outputs/warmup_d5_10000/` (ou `outputs/warmup_d5_5k/`)
- Un checkpoint warmup entraîné : `outputs/sup_warmup_3500/best.pt` (ou `outputs/sup_warmup_v2/best.pt`)

## Commande principale Phase 3 (chemin B complet)

```bash
uv run python scripts/train.py \
    network.init_from=outputs/sup_warmup_3500/best.pt \
    network.type=graphnet \
    network.num_blocks=4 \
    network.num_channels=128 \
    network.value_head_type=scalar \
    aux_heads.enabled=true \
    aux_heads.mill_diff_weight=0.3 \
    aux_heads.pieces_diff_weight=0.3 \
    training.learning_rate=1e-4 \
    training.lr_warmup_steps=2000 \
    training.lr_decay_steps=100000 \
    training.batch_size=256 \
    training.replay_buffer_size=200000 \
    training.min_buffer_size=2000 \
    training.total_steps=100000 \
    training.checkpoint_interval=1000 \
    training.warmup_data_path=outputs/warmup_d5_10000 \
    training.warmup_buffer_size=200000 \
    training.warmup_mix_fraction=0.3 \
    training.warmup_mix_anneal_steps=50000 \
    training.eval_vs_baselines.enabled=true \
    training.eval_vs_baselines.interval=5000 \
    training.eval_vs_baselines.n_games_d3=50 \
    training.eval_vs_baselines.n_games_d5=50 \
    training.eval_vs_baselines.num_sims=200 \
    self_play.num_workers=10 \
    self_play.asymmetric.enabled=true \
    self_play.asymmetric.prob_sym=0.5 \
    self_play.asymmetric.prob_t_asym=0.3 \
    self_play.asymmetric.prob_noise_asym=0.2 \
    self_play.curriculum.enabled=false \
    self_play.discard_timeout_games=false \
    mcts.num_simulations_train=150 \
    mlflow.enabled=true \
    device=cuda \
    2>&1 | tee outputs/phase3_run1.log
```

### Ce que cette commande fait, étape par étape

**Au démarrage** :
1. `build_network` instantie un GraphNet 4 blocs × 128 channels (~215k params)
2. `network.init_from` charge les poids depuis `sup_warmup_3500/best.pt` (state_dict only, optimizer reset à 0)
3. `training.warmup_data_path` lit tous les `worker_*.jsonl` de `warmup_d5_10000/` et matérialise chaque position en `SampleRecord` (replay + encode + softmax(scores) + γ^(T-t)·outcome + aux features). Stocké dans un `warmup_buffer` séparé, JAMAIS purgé.
4. `Trainer` initialise Adam(lr=1e-4) + scheduler `LinearLR(0.01→1.0)` sur 2000 steps puis `CosineAnnealingLR` jusqu'à 100k
5. 10 workers self-play démarrent avec deux instances de `MorrisSearch` chacun (α=0.3 standard + α=1.0 high-noise pour le régime `noise_asym`)

**À chaque partie self-play** :
- Le worker tire un régime : `sym (50%)` / `t_asym (30%)` / `noise_asym (20%)`
- Joue 80-120 plies, écrit le `GameRecord` (taggé `regime`) dans la queue résultats
- Si tracing activé (`MORRIS_TRACE_DIR=...`), append la trace JSONL pour replay offline

**À chaque step d'entraînement** :
- Tire un minibatch : 70 % du main buffer (FIFO), 30 % du warmup buffer (non-purgé)
- Le coefficient de mix **descend linéairement vers 0 sur 50k steps** (anneal). À step 25k → 15 % warmup. À step 50k+ → 0 % (le réseau ne voit plus que ses propres données self-play).
- Forward + loss (policy CE + value MSE + aux MSE), backward, Adam step
- Toutes les 1000 steps : checkpoint `.pt` dans le run-dir
- **Toutes les 5000 steps : eval automatique** vs minimax d3 + d5 (50+50 games), résultats loggés sous `eval/*` dans TB

**Quand s'arrête** : à `total_steps=100000` (manuel `Ctrl-C` aussi OK, le dernier checkpoint reste utilisable).

### Comment savoir si le réseau s'améliore

Six métriques à monitorer dans TensorBoard, **par ordre d'importance** :

| Métrique TB | Cible | Action si dévie |
|---|---|---|
| `eval/net_vs_d3_winrate` | **≥ 0.50, croissant** après init | Si chute sous 0.40 → kill le run, soupçon collapse |
| `eval/bare_vs_d3_winrate` | ≥ 0.50, stable ou ↗ | Si chute → la policy se dégrade (pas juste MCTS qui sauve) |
| `eval/net_vs_d5_winrate + drawrate` | ≥ 0.30 win, ≥ 0.40 cumulé | Stagnation = plafond du jeu (Morris résolu nul) |
| `game/length_mean` | descend 100 → 50-70 | Si ↗ ou stagne > 130 → draw attractor frappe |
| `game/decisive_rate` | ≥ 30 % au début, peut descendre | Si < 5 % rapidement → asymmetric ne suffit pas |
| `train/value_std` | > 0.20 | Si → 0 → value collapse (lazy mean) |

**Les évolutions sont aussi visibles via** :
```bash
# Suit le log en direct
tail -f outputs/phase3_run1.log

# TensorBoard
uv run tensorboard --logdir outputs/<run_dir>/tensorboard
# Puis dans le browser : http://localhost:6006
```

### Variantes simplifiées (par ordre de complexité décroissante)

**Sans sub-buffer warmup** (laisse juste init_from + asymmetric + eval) :
```bash
uv run python scripts/train.py \
    network.init_from=outputs/sup_warmup_3500/best.pt \
    network.type=graphnet network.num_blocks=4 network.num_channels=128 \
    aux_heads.enabled=true \
    training.learning_rate=1e-4 \
    training.lr_warmup_steps=2000 \
    training.eval_vs_baselines.enabled=true \
    self_play.num_workers=10 \
    self_play.asymmetric.enabled=true \
    mcts.num_simulations_train=150 \
    device=cuda
```

**Sans rien (just self-play depuis warmup)** :
```bash
uv run python scripts/train.py \
    network.init_from=outputs/sup_warmup_3500/best.pt \
    network.type=graphnet network.num_blocks=4 network.num_channels=128 \
    training.eval_vs_baselines.enabled=true \
    self_play.num_workers=10 \
    device=cuda
```

## Évaluation manuelle d'un checkpoint

### 1. Gate Phase 2/3 (bare network argmax vs minimax-d3)

```bash
uv run python scripts/eval_elo.py outputs/<run_dir>/checkpoints/latest.pt \
    --depth 3 --num-games 200 --opening-random-k 4 --device cpu
```

- **`--opening-random-k 4`** : critique pour casser le déterminisme (sinon 2 games répétées 100×)
- **`--num-games 200`** : pour avoir un CI à 95 % serré
- Cible : score ≥ 0.50

### 2. Test sérieux vs minimax-d5

```bash
uv run python scripts/eval_elo.py outputs/<run_dir>/checkpoints/latest.pt \
    --depth 5 --num-games 100 --opening-random-k 4
```

### 3. Head-to-head vs un autre checkpoint (warmup ou ancien Phase 3)

```bash
# Le nouveau Phase 3 vs le warmup d'origine
uv run python scripts/eval_elo.py outputs/<phase3_run>/checkpoints/latest.pt \
    --vs-checkpoint outputs/sup_warmup_3500/best.pt \
    --num-games 200 --opening-random-k 4

# Phase 3 vs Phase 3 ancien (mesurer le progrès)
uv run python scripts/eval_elo.py outputs/phase3_run2/checkpoints/latest.pt \
    --vs-checkpoint outputs/phase3_run1/checkpoints/final.pt \
    --num-games 200 --opening-random-k 4
```

### 4. Avec MCTS (au lieu de bare argmax)

```bash
uv run python scripts/eval_elo.py outputs/<run_dir>/checkpoints/latest.pt \
    --depth 5 --use-mcts --num-sims 400 --num-games 100 --opening-random-k 4
```

Pour le run de production complet : `--num-sims 800` (mais ~5× plus lent).

## Jouer interactivement contre le réseau

Le script `scripts/play_human.py` auto-détecte ResNet vs GraphNet depuis la config du checkpoint et marche avec n'importe lequel.

```bash
# Tu joues P1, agent avec 400 MCTS sims/coup
uv run python scripts/play_human.py outputs/<run_dir>/checkpoints/latest.pt

# Tu joues P2 (l'agent ouvre)
uv run python scripts/play_human.py outputs/<run_dir>/checkpoints/latest.pt --side 2

# Agent plus fort (1500 sims, plus lent, ~3-10s par coup)
uv run python scripts/play_human.py outputs/<run_dir>/checkpoints/latest.pt --num-sims 1500

# Mode "presque sans MCTS" — voir la qualité du prior réseau seul
uv run python scripts/play_human.py outputs/<run_dir>/checkpoints/latest.pt --num-sims 8
```

**Format des coups** :
| Phase | Format | Exemples |
|---|---|---|
| Placement | label seul | `a7`, `d6` |
| Capture | label seul | `b6` (la pièce à enlever) |
| Movement | source-dest | `a7 d7` ou `a7->d7` ou `a7→d7` |

**Touches** :
- `?` : liste les coups légaux
- `q` : quitter

## Monitoring du training

### TensorBoard (live)

```bash
uv run tensorboard --logdir outputs/<run_dir>/tensorboard
```

Tags principaux à surveiller :

**Training (gradient signal)** :
- `train/total_loss`, `train/policy_loss`, `train/value_loss`, `train/mill_loss`, `train/pieces_loss`
- `train/learning_rate` (devrait ramper à 1e-4 sur les 2000 premiers steps, puis decay cosine)
- `train/grad_norm` (devrait rester < 5)
- `train/value_mean`, `train/value_std` (alarmes si value_std → 0)

**Self-play (qualité des données)** :
- `game/length_mean`, `length_p10`, `length_p50`, `length_p90`
- `game/decisive_rate` (% non-nulles)
- `game/term_reason_pieces_below_3_rate`, `..._halfmove_clock_50_rate`, `..._threefold_rate`
- `game/mill_rate`, `game/captures_mean`

**Eval automatique (toutes les 5000 steps)** :
- `eval/net_vs_d3_winrate`, `eval/net_vs_d3_drawrate`, `eval/net_vs_d3_lossrate`
- `eval/net_vs_d5_winrate`, `eval/net_vs_d5_drawrate`, `eval/net_vs_d5_lossrate`
- `eval/bare_vs_d3_winrate`, `eval/bare_vs_d3_drawrate`, `eval/bare_vs_d3_lossrate`

### Console (en parallèle)

Dans un autre terminal :
```bash
# Latency de progression (jeux/sec, gradient steps/sec)
tail -f outputs/<run_dir>/train.log | grep -E "step|games"

# Nombre de checkpoints créés
watch -n 10 'ls -lh outputs/<run_dir>/checkpoints/ | tail -5'

# Mémoire RSS des workers
watch -n 5 'ps -p $(pgrep -d, -f "python scripts/train.py") -o pid,rss,cmd | head -20'
```

### MLflow (si activé)

```bash
# Le serveur tourne déjà via mlflow.enabled=true. Pour l'UI :
mlflow ui --backend-store-uri ./mlruns
# → http://localhost:5000
```

### Inspecter les parties self-play (optionnel)

Active le tracing :
```bash
export MORRIS_TRACE_DIR=outputs/<run_dir>/traces
export MORRIS_TRACE_SAMPLE_RATE=0.05  # 5% des parties
# (re)lance le training, puis dans un 2e terminal :
uv run python scripts/replay_game.py outputs/<run_dir>/traces --latest
```

## Comment détecter une régression (collapse) et agir

### Symptômes de collapse

1. `eval/net_vs_d3_winrate` chute sous 0.40 entre deux ticks d'eval (5000 steps)
2. `game/decisive_rate` < 5 %
3. `train/value_std` → 0 (lazy mean attractor)
4. `game/length_mean` → 200 (cap atteint constamment)

### Actions (par ordre)

**Si collapse léger** (winrate d3 0.30-0.45) :
```bash
# Continue le run, mais bump la fraction warmup
# Ctrl-C le run actuel, puis :
uv run python scripts/train.py \
    training.resume=outputs/<run_dir>/checkpoints/<best_so_far>.pt \
    training.warmup_data_path=outputs/warmup_d5_10000 \
    training.warmup_mix_fraction=0.5 \
    training.warmup_mix_anneal_steps=30000 \
    ... [autres args identiques]
```

**Si collapse sérieux** (winrate d3 < 0.30) :
```bash
# Rollback complet au warmup
# Ne reprends PAS du checkpoint Phase 3 — restart depuis le warmup
uv run python scripts/train.py \
    network.init_from=outputs/sup_warmup_3500/best.pt \
    training.learning_rate=5e-5 \
    training.warmup_mix_fraction=0.5 \
    training.warmup_mix_anneal_steps=100000 \
    self_play.asymmetric.enabled=true \
    self_play.asymmetric.prob_sym=0.7 \
    self_play.asymmetric.prob_t_asym=0.2 \
    self_play.asymmetric.prob_noise_asym=0.1 \
    ... [autres args]
```

## Comment savoir si la Phase 3 a réussi

**Critères de succès** (à mesurer une fois le run terminé) :

1. **Beat le warmup head-to-head** :
   ```bash
   uv run python scripts/eval_elo.py outputs/phase3_run/checkpoints/final.pt \
       --vs-checkpoint outputs/sup_warmup_3500/best.pt --num-games 200 --opening-random-k 4
   ```
   → score ≥ 0.55 = Phase 3 a apporté un gain

2. **Winrate vs minimax-d3 ≥ 70 %** :
   ```bash
   uv run python scripts/eval_elo.py outputs/phase3_run/checkpoints/final.pt \
       --depth 3 --num-games 200 --opening-random-k 4
   ```

3. **Wins vs minimax-d5 ≥ 30 %** (ou non-loss ≥ 70 %) :
   ```bash
   uv run python scripts/eval_elo.py outputs/phase3_run/checkpoints/final.pt \
       --depth 5 --num-games 100 --opening-random-k 4
   ```

4. **Avec MCTS+800 sims, bat d5 ≥ 50 %** :
   ```bash
   uv run python scripts/eval_elo.py outputs/phase3_run/checkpoints/final.pt \
       --depth 5 --use-mcts --num-sims 800 --num-games 100 --opening-random-k 4
   ```

Si les 4 sont OK → tu as un agent supérieur à minimax-d5. C'est l'objectif Phase 3.

Si #1 réussit mais #3 stagne → tu es proche du plafond théorique (Morris = nul). Considère Phase 4 (tablebase anchor, hors-scope ici).

Si #1 échoue (Phase 3 < warmup) → quelque chose a mal tourné. Revenir au warmup, baisser LR, augmenter warmup_mix_fraction.

## Récap commits Phase 3

```
76eb3d7  Phase 3 self-play fine-tuning: 4 features for warmup -> AlphaZero transition
d1f63b1  supervised: --no-early-stop flag + opening_random_k in eval
d3a6fe1  eval_elo: add --opening-random-k for statistical validity
```
