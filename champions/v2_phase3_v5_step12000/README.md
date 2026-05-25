# Champion v2 — Phase 3 v5, step 12000

**Source** : `outputs/2026-05-25/21-02-47/checkpoints/checkpoint_00012000.pt`

**Architecture** : GraphNet 4×128 (215,466 params), aux heads enabled.

**Lineage** :
- Init = `champions/v1_phase3_v2_step10000/best.pt` (previous champion)
- Phase 3 v5 self-play : random_opening K=8 + curriculum (50% random 4v4 starts)
- ~12k gradient steps before the peak — kept as the strongest MCTS-driven agent

## Phase 3 v5 config (key knobs)

```
network.type            : graphnet 4×128
training.learning_rate  : 3e-5  (warmup 2000 steps, cosine over 60000)
training.warmup_mix_fraction : 0.1  (light minimax-d=5 anchoring)
self_play.asymmetric.prob_random_opening : 1.0
self_play.asymmetric.random_opening_k    : 8
self_play.curriculum.enabled             : true
self_play.curriculum.random_start_fraction : 0.5
self_play.curriculum.pieces_per_player   : 4
self_play.playout_cap.full_sim_fraction  : 0.40
mcts.num_simulations_train               : 800
```

## In-training eval @ step 12000 (n=30 per matchup, opening_random_k=4)

| opponent | agent | W | D | L | score |
|---|---|---|---|---|---|
| MinimaxAgent(d=3) | net + MCTS200 | 0.50 | 0.30 | 0.20 | **0.65** |
| MinimaxAgent(d=3) | bare argmax | 0.20 | 0.13 | 0.67 | 0.27 |
| MinimaxAgent(d=5) | net + MCTS200 | 0.15 | 0.45 | 0.40 | 0.38 |

Compared with the previous champion (v1):

| opponent | v1 (step 10k) | v2 (step 12k) |
|---|---|---|
| d=3, net+MCTS200 | 0.63 | **0.65** |
| d=3, bare argmax | 0.72 | 0.27 ⚠️ |
| d=5, net+MCTS200 | 0.20 | 0.38 |

## Trade-off vs v1

v2 is a **stronger MCTS-driven agent** but a **weaker bare prior**. It plays
better when given thinking time (200+ simulations) but worse if you ever
turn the search off. For the web demo (which runs MCTS at every move),
that's the right trade-off.

If you ever need a pure policy network with strong argmax behavior (e.g.
fast play with zero search), prefer v1.

## Notes

- 30-game in-training eval has Wilson 95% CI ≈ ±0.18, so the +0.02 over v1
  is within noise; the meaningful improvement is on d=5 (+0.18) and the
  fact that the head-to-head vs v1 + 100-game eval should confirm.
- Run a proper 100-game eval before treating any margin as decisive.
