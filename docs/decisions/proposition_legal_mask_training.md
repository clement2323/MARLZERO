# Proposition à évaluer : utiliser le vrai `legal_mask` pendant le training

## Contexte

Dans la pipeline AlphaZero actuelle (Nine Men's Morris), le `PolicyHead` ([src/morris_rl/network/heads.py:48](src/morris_rl/network/heads.py#L48)) applique :

```python
logits = logits.masked_fill(~legal_mask, float("-inf"))
return F.log_softmax(logits, dim=1)
```

À l'**inférence** (self-play, MCTS, démo, eval), `legal_mask` est le vrai masque produit par `get_legal_actions(state)` — les illégaux reçoivent proba 0 dur.

À l'**entraînement** ([src/morris_rl/training/trainer.py:228-229](src/morris_rl/training/trainer.py#L228-L229)), le mask passé est `full_mask = ones(...)` (tout-True). Justification documentée dans le docstring du module : la cible MCTS `policy_target` a déjà 0 sur les illégaux, donc la cross-entropy `-Σ π·log p` ne reçoit aucun gradient direct sur les illégaux.

## Idée

Stocker le `legal_mask` dans le replay buffer (ou le recalculer depuis l'état au moment du sample) et le passer à `Trainer.step`, pour que le `log_softmax` soit normalisé uniquement sur les actions légales — comme à l'inférence.

## Arguments pour

1. **Stabilité au démarrage.** Logits initiaux ≈ aléatoires → sans masque, `p(a_légal) ≈ 1/600` et CE initiale ≈ `log(600) ≈ 6.4`. Avec masque, `p(a_légal) ≈ 1/n_légal` (typiquement 10-30) et CE ≈ `log(20) ≈ 3`. Le réseau évite de gaspiller ses premières dizaines de milliers d'updates à pousser 570+ logits illégaux vers -∞ avant d'affiner les bons.

2. **Cohérence train/inférence.** La cible MCTS provient de visites obtenues avec des priors **masqués** ([src/morris_rl/training/inference_server.py:458-459](src/morris_rl/training/inference_server.py#L458-L459)). Demander au réseau de reproduire cette distribution à partir d'une softmax non masquée est mal posé, même si ça marche en pratique.

3. **Précision.** Mismatch entre la distribution scorée pendant l'entraînement (dénominateur sur 600) et celle à l'inférence (dénominateur sur n_légal). Devient marginal en steady-state mais existe.

## Arguments contre / coûts

1. **Coût mémoire** : stocker `(N_buffer, 600)` booléens. À 500k samples, ~37.5 MB packed / ~300 MB dense. Acceptable mais non trivial.
2. **Alternative** : recalculer le mask CPU-side au moment du sample → 256 appels `get_legal_actions` par batch, à mesurer (probablement quelques ms).
3. Argument faible *pro* status quo : avec `full_mask`, le réseau apprend implicitement les règles. Mais on a déjà `get_legal_actions`, donc régul auxiliaire douteuse.

## Question pour évaluation

1. Est-ce que mes arguments 1 et 2 ci-dessus tiennent, ou est-ce que je sous-estime un effet implicite du `full_mask` ?
2. La voie recommandée : stocker le mask dans le buffer, ou le recalculer au sample ?
3. Est-ce que ça vaut le coup de faire une A/B courte (~30 min × 2 runs) pour mesurer la divergence des courbes early avant de l'intégrer ?

Verdict actuel : faisable, low risk, gain attendu surtout sur la convergence early. À valider empiriquement.
