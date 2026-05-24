# MorrisGraphNet — Journal d'expériences

Objectif : comparer GraphNet (graph-aware) à la baseline ResNet (conv1d) sur Morris, à différentes tailles de réseau. Tracker chaque run, ses hypothèses, ses résultats.

Architecture détaillée : [gnn-architecture.md](gnn-architecture.md).

## Protocole de comparaison

Tous les runs partagent :
- `value_head_type=categorical` (3-class cross-entropy)
- `aux_heads.enabled=true` (mill_diff + pieces_diff)
- `curriculum.enabled=false` (full game from initial state)
- `discard_timeout_games=true` (jeter les samples de parties qui timeout au cap)
- `mcts.num_simulations_train=400`
- `num_workers=10`, CPU mode
- `device=cpu`, `inference_mode=per_worker_cpu`
- `symmetry_augmentation=true` (8× via D4)
- `total_steps=100_000`, `checkpoint_interval=1000`

Métriques clés à reporter à **step 5 000** et **step 10 000** :
- `train/policy_loss` (cible < 1.5 à step 5k)
- `train/value_loss` (cible < 0.4 à step 5k)
- `game/term_piece_count_tiebreak_rate` (cible < 60 % à step 10k)
- `game/p1_win_rate` (cible → 0.5)
- Win-rate vs `RandomAgent` (100 games, 800 sims eval, cible ≥ 90 % à step 10k)

## Runs

### Run 0 — Baseline ResNet (référence existante)

| Champ | Valeur |
|---|---|
| Status | ✅ terminé (run [`20-26-12`](../outputs/2026-05-17/20-26-12/)) |
| Network | `MorrisResNet 6×64` (conv1d) |
| Params | 165 482 |
| Backbone | conv1d, kernel=3, BN, residual |
| Step atteint | 3 336 |
| `policy_loss` step 3336 | **1.85** (stagne dès step 500) |
| `value_loss` step 3336 | 0.63 |
| `tiebreak_rate` | **92.5 %** |
| `p1_win_rate` | **0.83** (biais P1 tiebreak fort) |
| Win-rate vs random (100 games, 200 sims) | **62 %** au step 4000 (run jumelle 06-43-04) |
| Verdict | Plateau policy → la conv1d n'apprend pas la topologie suffisamment |

Conclusion : la stack ResNet plafonne. Hypothèse à tester : un backbone graph-aware débloque la policy à param égal ou supérieur.

---

### Run 1 — GraphNet 6×128 (premier essai) — **À LANCER**

| Champ | Valeur |
|---|---|
| Status | ⏳ pending |
| Network | `MorrisGraphNet 6×128` |
| Params attendus | ~315 000 (vs 165 k ResNet baseline = ~2× capacité) |
| Backbone | 6 GraphConvBlocks, hidden=128, deux relations (`A_adj` + `A_mill`), residual + BN |
| Input encoding | 11 planes (7 legacy + own_threats + opp_threats + degree + ring) |
| Hypothèse | Inductive bias graph-aware débloque la policy : `policy_loss` < 1.5 à step 5k, `tiebreak_rate` < 70 % à step 10k |
| Critère succès vs Run 0 | À step 5k : `policy_loss` plus bas que 1.85, ET win-rate vs random ≥ 80 % (100 games, 800 sims) |
| Commande | voir ci-dessous |
| Run dir | _(à remplir après lancement)_ |

#### Résultats step 5k

_à remplir_

| Métrique | Valeur | vs Run 0 (ResNet) |
|---|---|---|
| `policy_loss` | | |
| `value_loss` | | |
| `tiebreak_rate` | | |
| `p1_win_rate` | | |
| Win-rate vs random (100 games, 800 sims) | | |

#### Résultats step 10k

_à remplir_

#### Verdict & notes

_à remplir : qu'est-ce qui a marché, qu'est-ce qui plafonne, hyperparam à ajuster pour le run suivant._

---

### Run 2 (planifié) — GraphNet 3×64 (challenger "petit modèle")

_À lancer après le verdict du Run 1._ Test de la thèse "30 k params suffisent pour Morris".

| Champ | Valeur |
|---|---|
| Status | 📋 planifié |
| Network | `MorrisGraphNet 3×64` |
| Params attendus | ~30 000 |
| Hypothèse | Inductive bias permet d'apprendre Morris avec 10× moins de capacité qu'AlphaZero standard |
| Critère succès | À step 5k : `policy_loss` au pire +0.1 vs Run 1, win-rate ≥ 70 % |

---

### Run 3 (planifié) — GraphNet 4×96 (milieu)

_À lancer pour balayer la courbe de scaling._

| Champ | Valeur |
|---|---|
| Status | 📋 planifié |
| Network | `MorrisGraphNet 4×96` |
| Params attendus | ~100 000 |
| Hypothèse | Sweet spot capacity/inductive-bias |

---

### Run 4 (planifié) — GraphNet 6×128 + symmetry_augmentation=false

Test : le trunk étant D4-equivariant, l'augmentation est-elle redondante ?

| Champ | Valeur |
|---|---|
| Status | 📋 planifié |
| Network | identique Run 1 |
| Diff | `training.symmetry_augmentation=false` |
| Hypothèse | Throughput ×8 sur l'écriture buffer, perf finale ≈ Run 1 (les heads cassent quand même l'equivariance, donc l'augmentation aide encore un peu) |
| Critère | Throughput buffer-fill divisé par 8 ; `policy_loss` step 5k au pire +0.05 vs Run 1 |

---

## Commande à lancer (Run 1)

```bash
MORRIS_TRACE_DIR=outputs/traces MORRIS_TRACE_SAMPLE_RATE=0.5 \
uv run python scripts/train.py \
  network.type=graphnet \
  network.num_blocks=6 \
  network.num_channels=128 \
  network.value_head_type=categorical \
  aux_heads.enabled=true \
  self_play.curriculum.enabled=false \
  self_play.discard_timeout_games=true \
  self_play.inference_mode=per_worker_cpu \
  self_play.inference_device=cpu \
  self_play.num_workers=10 \
  mcts.num_simulations_train=400 \
  training.min_buffer_size=2000 \
  training.checkpoint_interval=1000 \
  training.total_steps=100000 \
  training.symmetry_augmentation=true \
  mlflow.enabled=true \
  device=cpu \
  2>&1 | tee train_gnn_run1.log
```

### Vérifs au boot

Premier log d'intérêt :
```
INFO __main__:main:108 - Network: ResNet6×128, params=315,180
```
Le nom logique reste "ResNet" (format hardcodé) ; le `params=315 180` confirme que c'est bien `MorrisGraphNet` (un vrai ResNet 6×128 ferait ~150 k params).

### Eval à step 5000

```bash
RUN=$(ls -td outputs/2026-05-*/*/ | head -1)
uv run python scripts/evaluate.py \
  "$RUN/checkpoints/checkpoint_00005000.pt" \
  --opponent random --num-games 100 --num-simulations 800 --device cpu
```

### Eval à step 10000

Idem en remplaçant par `checkpoint_00010000.pt`.

### Ouvrir TensorBoard en parallèle

```bash
make tensorboard
# puis http://localhost:6006
```

Watch courbes : `train/policy_loss`, `train/value_loss`, `game/term_piece_count_tiebreak_rate`, `game/p1_win_rate`.
