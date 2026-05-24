# Pipeline warmup Morris — état au 2026-05-24

Rapport de session : Phase 1 (génération dataset minimax) + Phase 2 (pré-entraînement supervisé) sont **implémentées et commitées**. Phase 3 (self-play depuis le checkpoint warmup) est planifiée mais non encore implémentée.

## Vue d'ensemble en 4 phases

| Phase | But | Sortie | Statut |
|---|---|---|---|
| **0** | Mesure de référence (random, minimax d3/d5) | Stats sur `outputs/baseline_stats.log` | ✅ Outils faits, validés sur gate-100 |
| **1** | Génération du dataset warmup | `outputs/warmup_d5_*/worker_*.jsonl` | 🟡 En cours (~9 g/min sur 16 workers) |
| **2** | Pré-entraînement supervisé du réseau | `outputs/sup_warmup_*/best.pt` | ✅ Code prêt, smoke validé |
| **3** | Self-play fine-tuning depuis warmup | nouveau checkpoint stronger | ⏳ Planifié, hyperparams ci-dessous |

Décisions clés validées :
- Heuristique riche minimax 8 features (mat, mills closed, **mills potentiels poids 0.15/0.20**, mobilité, forks, crossroads, blocked, phase-aware)
- ε=0.10, opening_random_k=5, depth=5, cap=200 (=DRAW)
- Backbone : **GraphNet** 4 blocs hidden=128 (~300k params)
- Aux heads `mill_diff` + `pieces_diff` (KataGo-style)
- γ=1.0 par défaut, configurable
- Augmentation 16× (D4 × color-swap) on-the-fly
- Symétrie color-swap valide aussi en phase placement (planes 0↔1, 2↔3, 7↔8 swap + value/aux négés)

## Phase 1 — Génération du dataset

### Lancer la génération (la commande principale)

Sur ce CPU (16 cores physiques) :

```bash
# Full 5000 parties (~13h wallclock sur 16 workers)
uv run python scripts/generate_warmup_dataset.py \
    --num-games 5000 --depth 5 --workers 16 \
    --epsilon 0.10 --opening-random-k 5 --seed 0 \
    --out-dir outputs/warmup_d5_5k \
    2>&1 | tee outputs/warmup_d5_5k.log
```

Pour 10 000 parties, double le `--num-games`. ~26 h wallclock.

### Observer la génération en temps réel (terminal séparé)

```bash
# Summary 1 ligne par partie au fur et à mesure :
uv run python scripts/watch_warmup_games.py outputs/warmup_d5_5k --follow --summary

# Ou affichage du board final pour chaque partie :
uv run python scripts/watch_warmup_games.py outputs/warmup_d5_5k --follow --render-each
```

### Inspecter les parties

```bash
# Lister toutes les parties générées
uv run python scripts/replay_game.py outputs/warmup_d5_5k --list

# Rejouer interactivement la plus récente (← / → pour naviguer)
uv run python scripts/replay_game.py outputs/warmup_d5_5k --latest

# Une partie spécifique (worker 7, index 3)
uv run python scripts/replay_game.py outputs/warmup_d5_5k --worker 7 -i 3

# Filtrer par term_reason
uv run python scripts/replay_game.py outputs/warmup_d5_5k --filter pieces_below_3 -i 0
```

Chaque coup est annoté `[OPENING-RANDOM]` (jaune) / `[ε-RANDOM]` (rouge) / `[MINIMAX]` (vert) pour identifier l'origine.

### Analyse statistique du dataset (à faire pendant et à la fin)

```bash
# Stats raw (sans canonicalisation symétrique)
uv run python scripts/warmup_stats.py outputs/warmup_d5_5k

# Stats canoniques (orbite D4 × color-swap = 16 éléments)
uv run python scripts/warmup_stats.py outputs/warmup_d5_5k --canonical
```

Mesures clés :
- **Decisive rate** : cible ≥ 50 %. Sur le gate-100, on a 89 %.
- **Mean length** : cible < 150 half-moves. Gate-100 : 66 plies.
- **% cap 200** : cible ≤ 20 %. Gate-100 : 0 %.
- **Coverage canonique** : cible ≥ 30 %. Gate-100 : 94 %.

Si l'un de ces seuils est raté, bump `--epsilon 0.15`, `--opening-random-k 8`, ou plus de parties.

## Phase 2 — Pré-entraînement supervisé (GPU recommandé)

### Run principal sur 5000 parties (cible : ~30-90 min RTX 4070)

```bash
uv run python scripts/train_supervised.py \
    --warmup-dir outputs/warmup_d5_5k \
    --network-type graphnet --num-blocks 4 --num-channels 128 \
    --aux-heads-enabled --aux-coeff-mill 0.3 --aux-coeff-pieces 0.3 \
    --epochs 40 --batch-size 512 \
    --lr 1e-3 --weight-decay 1e-4 \
    --gamma 1.0 --policy-temperature 1.0 \
    --val-split 0.1 --val-seed 0 \
    --early-stop-patience 5 --eval-every 5 \
    --n-eval-random 100 --n-eval-d3 50 \
    --device cuda --mixed-precision \
    --out-dir outputs/sup_warmup_5k \
    2>&1 | tee outputs/sup_warmup_5k/train.log
```

Sorties :
- `outputs/sup_warmup_5k/best.pt` — checkpoint avec le meilleur `val_loss` (à utiliser pour Phase 3)
- `outputs/sup_warmup_5k/final.pt` — dernier epoch
- `outputs/sup_warmup_5k/tb/` — TensorBoard logs

### Suivi TensorBoard

```bash
uv run tensorboard --logdir outputs/sup_warmup_5k/tb
```

Métriques tracées :
- `train/{total,policy,value,mill,pieces}` — components de loss
- `val/{total,policy,value,mill,pieces}` — sur le held-out 10 %
- `eval/winrate_vs_random`, `eval/drawrate_vs_random` — toutes les 5 epochs
- `eval/winrate_vs_d3`, `eval/drawrate_vs_d3`, `eval/lossrate_vs_d3`, `eval/non_loss_vs_d3`

### Smoke rapide (CPU, < 1 min, pour valider la pipeline)

```bash
uv run python scripts/train_supervised.py \
    --warmup-dir outputs/warmup_d5_gate100 \
    --network-type graphnet --num-blocks 2 --num-channels 64 \
    --epochs 3 --batch-size 64 \
    --eval-every 0 --device cpu \
    --out-dir /tmp/sw_smoke
```

## Évaluation ELO — gate Phase 2 → 3

### Le gate critique : bare network vs minimax d3

```bash
uv run python scripts/eval_elo.py outputs/sup_warmup_5k/best.pt \
    --depth 3 --num-games 200 --device cpu
```

Affiche un banner `PHASE 2 → 3 GATE : PASSED / NOT YET` selon que `score ≥ 0.50`. **Tant que ce gate n'est pas passé, ne pas lancer Phase 3.**

### Autres évaluations utiles

```bash
# Sanity check : doit gagner > 95 % vs random
uv run python scripts/eval_elo.py outputs/sup_warmup_5k/best.pt --vs-random --num-games 100

# Plus difficile : vs minimax depth 5
uv run python scripts/eval_elo.py outputs/sup_warmup_5k/best.pt --depth 5 --num-games 100

# Comparaison head-to-head entre deux checkpoints
uv run python scripts/eval_elo.py outputs/sup_warmup_5k/best.pt \
    --vs-checkpoint outputs/sup_warmup_v2/best.pt --num-games 200

# Avec MCTS au lieu d'argmax (à utiliser post-Phase 3)
uv run python scripts/eval_elo.py outputs/sup_warmup_5k/best.pt \
    --depth 5 --use-mcts --num-sims 200 --num-games 100
```

## Phase 3 — Self-play fine-tuning (planifié, non implémenté)

### Pré-requis (à ajouter avant de lancer)

`scripts/train.py` n'a pas d'option `init_from`. Une petite addition de ~5 lignes sera nécessaire :

```python
# Dans scripts/train.py, après build_network(cfg):
if cfg.network.get("init_from"):
    payload = load_checkpoint(cfg.network.init_from)
    network.load_state_dict(payload["state_dict"], strict=False)
    log.info(f"initialized from {cfg.network.init_from} (step={payload['step']})")
```

### Commande Phase 3 prévue (après ajout init_from)

```bash
uv run python scripts/train.py \
    network.init_from=outputs/sup_warmup_5k/best.pt \
    network.type=graphnet network.num_blocks=4 network.num_channels=128 \
    training.learning_rate=1e-4 \
    training.lr_decay_steps=200_000 \
    training.replay_buffer_size=200000 \
    mcts.num_simulations_train=200 \
    self_play.num_workers=10 \
    self_play.curriculum.enabled=false \
    self_play.discard_timeout_games=false \
    mlflow.enabled=true \
    2>&1 | tee outputs/selfplay_phase3.log
```

Notes :
- **LR ÷10** (1e-4 au lieu de 1e-3) pour éviter dérive rapide du prior warmup
- **MCTS sims = 200** (default 150) au début pour avoir assez de search
- **Buffer 200k** : à 80 plies/partie = ~2500 parties avant flush total → garde le pre-warmup safe ~1h

### Anti-collapse à ajouter avant de lancer

1. **Sub-buffer warmup non purgé** : 30 % du minibatch vient toujours d'un buffer fixe contenant les positions warmup. Empêche le réseau d'oublier minimax. À implémenter dans `replay_buffer.py`.
2. **Eval automatique vs d3 tous les 1000 steps** : checkpoint rollback si winrate chute sous 0.50.
3. **KL anchor** (optionnel) : ajouter `λ * KL(policy_current || policy_warmup_snapshot)` à la loss, λ qui décroît sur 5k-10k steps.

## Hyperparams asymétriques pour le self-play (Phase 3 avancée)

La marelle souffre du **draw attractor** : deux agents identiques convergent vite vers 95 % de nuls et le signal s'effondre. Solution documentée (KataGo, AlphaStar, ExIt) : **casser la symétrie** entre les deux joueurs dans une partie self-play.

### Recette mixte (40 / 30 / 20 / 10)

À chaque partie self-play, tirer aléatoirement un régime :

| Régime | Probabilité | Joueur A | Joueur B |
|---|---|---|---|
| **Symétrique standard** | 40 % | T=1 (10 plies) puis T=0, α=0.3, mix=0.25, 200 sims | idem A |
| **Asym. température** | 30 % | T=1.0 partout | T=0 (argmax) |
| **Asym. bruit Dirichlet** | 20 % | α=1.0, mix=0.50 (exploration sauvage) | α=0.3, mix=0.25 (standard) |
| **Position augmentée** | 10 % | démarre depuis position du buffer (value proche de 0, entropie policy élevée), sinon standard | idem |

Effet : on sort de l'équilibre Nash conservateur. L'asymétrique explore, le standard exploite. Génère des résultats décisifs (≥ 30-40 %) au lieu de 5 % de wins habituels.

À chaque partie, alterner aléatoirement quel joueur est A et lequel est B pour équilibrer le dataset.

### Implémentation requise dans `self_play.py`

- Ajouter un sampler de régime par partie au début de `_play_one_game`
- Passer les params `(temperature, dirichlet_alpha, dirichlet_epsilon, num_simulations)` au `MorrisSearch` selon le régime et le joueur en cours
- Marquer le régime dans le `GameRecord` pour pouvoir filtrer/auditer dans TB

Aucun changement dans `MorrisSearch` lui-même (les params sont déjà tous des arguments de `search.run(...)`).

## Monitoring du nombre de coups par partie

**Déjà câblé dans le trainer existant.** Voir [trainer.py:419](src/morris_rl/training/trainer.py#L419) : `recent_lengths: deque[int] = deque(maxlen=200)`. Une fenêtre glissante des 200 dernières parties est maintenue et exposée à TB.

Métriques que tu verras automatiquement dans TB pendant le self-play :

| Tag TB | Sémantique |
|---|---|
| `game/length_mean` | longueur moyenne sur les 200 dernières parties |
| `game/length_p10`, `length_p50`, `length_p90` | percentiles |
| `game/term_reason_*` | proportion par raison de fin (pieces_below_3, halfmove_clock_50, threefold, max_total_halfmoves_safety_cap) |
| `game/captures_mean` | captures moyennes par partie |
| `game/mill_rate` | taux de moulins formés par partie |
| `game/decisive_rate` | % parties non-nulles |
| `game/pieces_diff_mean` | différentiel matériel moyen en fin de partie |

Lire à `tensorboard --logdir outputs/selfplay_phase3/tb`.

**Comportement attendu pendant Phase 3** :
- Itérations 1-5 (post-warmup) : `length_mean ≈ 80-120` plies
- Itérations 10-30 : descend à 50-80 plies à mesure que les agents s'améliorent
- Plateau itération 50+ : 40-60 plies (jeu quasi-parfait, nuls inévitables négociés rapidement)
- **Signal d'alarme** : si `length_mean` ↗ ou stagne haute (> 130), draw attractor mal géré → augmenter ε / Dirichlet ou descendre γ à 0.98

## Récap des commits sur la branche `gnn-backbone`

```
83db7fb  ELO evaluation script: bare network vs minimax (Phase 2 -> 3 gate)
75a088e  Phase 2: supervised warmup training on minimax JSONL traces
383b6b9  heuristic: lower potential_mills weight (horizon-effect correction)
893b623  warmup_stats.py: --canonical flag for D4 x color-swap unique positions
52100f3  Post-hoc warmup dataset stats script
ae5e68a  Warmup dataset module: parallel minimax-vs-minimax game generation
```

## Ordre d'exécution prévu

1. ⏳ Attendre la fin de `outputs/warmup_d5_5k` (~13 h)
2. `uv run python scripts/warmup_stats.py outputs/warmup_d5_5k --canonical` — vérifier les seuils
3. `uv run python scripts/train_supervised.py ...` — sur GPU, ~30-90 min
4. `uv run python scripts/eval_elo.py outputs/sup_warmup_5k/best.pt --depth 3 --num-games 200`
5. **Si gate PASSED** → planifier Phase 3 (asymetric self-play + LR=1e-4 + sub-buffer warmup)
6. **Si gate NOT YET** → générer 5000 parties additionnelles avec `--seed 1` + retrain warmup
