# Champion v1 — Phase 3 v2, step 10000

**Source** : `outputs/2026-05-24/20-28-14/checkpoints/checkpoint_00010000.pt`

**Architecture** : GraphNet 4×128 (215,466 params), aux heads enabled.

**Lineage** :
- Init = `outputs/sup_warmup_3500/best.pt` (Phase 2 supervised warmup, step 3696)
- Phase 3 v2 self-play : 10000 gradient steps under `t_asym` regime (T=0.5/0)
- Peak of the run before regression — kept here as the reference agent.

## Scores

| opponent | mode | sims | result | source |
|---|---|---|---|---|
| MinimaxAgent(d=3) | net+MCTS | 200 | score ≈ 0.63 | TB eval/net_vs_d3_winrate, step 10000 |
| MinimaxAgent(d=3) | bare argmax | — | score ≈ 0.72 | TB eval/bare_vs_d3_winrate, step 10000 |
| MinimaxAgent(d=5) | net+MCTS | 200 | score ≈ 0.20 | TB eval/net_vs_d5_winrate, step 10000 |

## Why kept

This run regressed strongly after step 16000 (`net_vs_d3` fell to 0.27 at step 38000
while `bare_vs_d3` stayed at 0.77). Step 10000 was the broadest plateau where
net+MCTS and bare-argmax both pointed in the right direction; before the value
head started drifting toward "everything is a draw".

Use this as the fallback agent for the web demo if Phase 3 v3 doesn't surpass it.
