# train a network to block adversary, more value for those ones*
# test muller gqme without guenay
# revenir q mq version du jeu et compqrer lq stqbilite du jeu en fonction !!
 tension a letoujffenent statistiques
 entrainer en sareetant a gevay et en continuant un peu plus loin aussi  avpoori

ello systen betzeen heuristic m notion d ebad player m regle detouffement same sans flyiong phase

test phase 1 phase 2
cargo test --release --test gevay_phase1_consistency_test -- --ignored --nocapture




 toop herem also befor ejaimerais etre sur auon obtient les memes tables aue genay ?
contrastive learning


Commandes training après Gévay complet :

Smoke (200 steps, 2 workers, ~5-10 min) — valide que le pipeline tourne :


cd /home/clement/projets/MARL
GEVAY_DIR=$(pwd)/data/tablebase/gevay \
TABLEBASE_DIR=$(pwd)/data/tablebase/flying \
.venv/bin/python scripts/train.py --config-name rl_gevay_flying_400sims \
    training.total_steps=200 self_play.num_workers=2
Full run (multi-jours, TensorBoard ouvert sur le côté) :


cd /home/clement/projets/MARL
GEVAY_DIR=$(pwd)/data/tablebase/gevay \
TABLEBASE_DIR=$(pwd)/data/tablebase/flying \
.venv/bin/python scripts/train.py --config-name rl_gevay_flying_400sims
Monitor pendant le full run (autre terminal) :


tensorboard --logdir outputs/ --port 6006
# ouvre http://localhost:6006
La config résout :

network.type = graphnet, 4×128 (~300k params)
mcts.num_simulations_train = 400
self_play.terminate_at_ply = 18 (stop fin placement)
self_play.variant = flying
gevay.enabled = true → chaque worker spawne son play_tb --serve --gevay-dir
Si un sample ply-18 tombe sur un sous-espace que Gévay n'a pas (rare en placement-only, mais possible si une partie finit terminal-precoce <ply18 par captures), le worker fallback sur le hybrid outcome existant. Pas de crash.