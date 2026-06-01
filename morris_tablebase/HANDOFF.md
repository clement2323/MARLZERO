# Morris Tablebase — État du projet et reprise (2026-06-02)

Document de reprise. Si tu démarres une nouvelle session (n'importe quel ordi), tout ce que tu dois savoir est ici.

## Objectif global

Reproduire et étendre **Gévay & Danner 2014** (arXiv 1408.0032v3) — *Calculating Ultra-Strong and Extended Solutions for Nine Men's Morris, Morabaraba, and Lasker Morris*.

3 phases :
1. **Phase 1 — Gasser tables** : retrograde analysis classique, verdict {WIN, LOSS, DRAW} + DTW pour chaque position. Variante : Morris standard **avec vol**.
2. **Phase 2 — V_Gévay** : ultra-strong solution. Classifie les DRAW positions par "tension" V ∈ [0, 1]. Permet de jouer contre un adversaire faible avec ~57% de victoires (vs 17% avec Phase 1 seul).
3. **Phase 3 — RL opening** : self-play sur les 18 plies de placement, **reward = V_Gévay(position de mouvement atteinte au ply 18)**. Cohérent avec CLAUDE.md à la racine du projet.

## État au 2026-06-02

### Phase 1 — Gasser tables (mouvement seul, avec vol)

**Code committé** (branch `gnn-backbone`) :
- `eed5380` Bootstrap crate
- `7694eae` Wave générique cross-subspace
- `c52d4aa` Driver `build_movement`
- `d6cf706` Storage `.bin` + indicatif progress bar + resume-from-disk
- `dbba07b` Canonicalisation D4 (8x compute, 1/8 RAM physique)

**Code modifié mais NON committé (à valider et committer)** :
- `src/subspace.rs` — fix u64 overflow (voir bug #1 ci-dessous) + ajout `MappedTable` (voir bug #2)
- `src/wave.rs` — fix u64 overflow
- `src/bin/build_movement.rs` — passe à mmap après save (voir bug #2)
- `Cargo.toml` — ajout `memmap2 = "0.9"`

**Scaffolding Phase 2 NON committé** :
- `src/work_unit.rs` — structure WorkUnit (paire `(s, -s)` ou ESC seul) + dag_order. 6 tests unitaires.
- `src/gevay/mod.rs` + `src/gevay/subspace_rank.rs` — formule val_s du papier Section IV-A + ranking ordinal. 4 tests.

### Phase 2 — V_Gévay

**Plan détaillé** dans `~/.claude/plans/stop-ebn-debut-de-snoopy-sifakis.md` (sur la machine locale uniquement). Résumé :
- Lire le papier Gévay-Danner (Section IV-A et IV-B)
- Implémenter le multi-valued retrograde (extension de la wave Phase 1)
- Algorithme : valeurs entières -W..+W, stockées **relatives** au rank du sous-espace courant, DTW direction basée sur le signe
- Cross-check vs Table V (W/D/L) puis Table VIII (V_Gévay distribution) du papier

**Décisions figées** :
- V_Gévay sur **mouvement seul** (pas de placement) — Phase 3 RL gérera l'opening
- Reproduire fidèlement Gévay-Danner d'abord, **étouffement extension** plus tard
- Pas de refactor 16x (D4 + color swap) — optionnel, notre 8x suffit pour matcher les pourcentages

### Phase 3 — RL opening

Pas commencé. Architecture à venir : self-play 18-ply opening + reward terminal V_Gévay.

## Bugs critiques découverts cette session

### Bug #1 — u32 overflow à (6,6) et au-delà

**Symptôme** : `panic: index out of bounds: the len is 702312992 but the index is 851419296`

**Cause** : `n_states = C(24,w) × C(24-w,b) × 2` dépasse u32::MAX (4.29B) pour les sous-espaces avec total ≥ 12.
- (6,6) → 4.997B (wrap modulo 2^32 = 702M, d'où la len incohérente)
- (9,9) → 13B

**Fix appliqué** (mais pas committé) :
- `subspace.rs` : `n_positions`, `n_states`, `state_index`, `state_index_canonical`, `decode_state` retournent désormais **u64**
- `wave.rs` : `queue: Vec<u64>`, HashSets `HashSet<u64>`, signatures `init_position(idx: u64, ...)` et `propagate_to_parents(p_idx: u64, ...)`
- `WaveStats::n_states: u64`

**Fichiers `.bin` déjà sur disque (3,3) à (5,7) intacts** — ils étaient sous u32::MAX donc le bug ne les a pas touchés.

### Bug #2 — OOM mémoire à cause de l'accumulation des Vec dans la Tablebase

**Symptôme** : crash mémoire après le fix u64, en arrivant aux gros sous-espaces ((6,6)+).

**Cause** : la `Tablebase` HashMap accumulait toutes les `SubspaceTable` résolues précédemment en RAM (Vec<u8> + Vec<u16>). Chaque `save()` page-faulte la totalité du Vec en RAM physique (sequential write touche toutes les pages). Cumulé sur 15+ sous-espaces → OOM.

**Fix appliqué** (mais pas committé) :
- `Cargo.toml` : ajout dep `memmap2 = "0.9"`
- `subspace.rs` :
  - Nouvelle struct `MappedTable` avec mmap + `verdict_at(idx) -> u8` / `dtw_at(idx) -> u16` via raw pointers
  - Enum `StoredTable::{Owned(SubspaceTable), Mapped(MappedTable)}`
  - `Tablebase::insert_mapped(MappedTable)` en plus de `insert(SubspaceTable)`
  - `Tablebase::query` dispatch sur le variant
- `bin/build_movement.rs` :
  - Après `solve_movement + save` : `drop(table)` puis `MappedTable::open(path)` → mmap read-only, kernel gère la page cache
  - Sous-espaces déjà sur disque : `MappedTable::open` directement sans passer par Vec
  - Helper `stats_from_mapped(sub, &MappedTable)` pour orbit-weighted stats sans Vec

**Effet** : RAM résidente bornée par taille du sous-espace courant (~3 GB physique pour (9,9) avec canonicalisation) + page cache OS évincible. Plus de fuite cumulative.

## Commandes pour reprendre

Depuis `morris_tablebase/` :

```bash
# 0. Préparer Rust si pas déjà fait
. "$HOME/.cargo/env"
# Si rust pas installé :
# curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal

# 1. Vérifier que tout compile (fix u64 + mmap + scaffolding Phase 2)
cargo build --release

# 2. Tests (~38 tests attendus : 11 rules + 7 symmetry + 8 hash + 2 storage + 6 work_unit + 4 subspace_rank)
cargo test --release

# 3. Si verts, committer tout d'un coup
git add morris_tablebase/ Cargo.toml
git commit -m "Fix u64 overflow + mmap Tablebase + Phase 2 scaffolding (work_unit, gevay::subspace_rank)"

# 4. Lancer la Phase 1 complète
# - Reprend les .bin existants sur disque
# - Reprend depuis (6,6) avec le fix u64
# - Mémoire bornée grâce à mmap
mkdir -p ../data/tablebase/flying
cargo run --release --bin build_movement -- 18 ../data/tablebase/flying/

# 5. Si Linux refuse les grosses allocations virtuelles à (9,9) (~65 GB virtual)
sudo sysctl vm.overcommit_memory=1
# Puis relancer la commande step 4
```

## Architecture actuelle du crate

```
morris_tablebase/
├── Cargo.toml                       # indicatif + memmap2
├── HANDOFF.md                       # CE FICHIER
├── src/
│   ├── lib.rs                       # pub mod board, gevay, hash, rules, storage, subspace, symmetry, wave, work_unit
│   ├── board.rs                     # MILLS, ADJACENCY, MILLS_THROUGH (const)
│   ├── rules.rs                     # is_mill_through, all_in_mills, legal_capture_targets, popcount
│   ├── symmetry.rs                  # 8 transforms D4, canonicalize, orbit_size
│   ├── hash.rs                      # BINOM[25][25] const, rank_subset, unrank_subset, compact/expand
│   ├── subspace.rs                  # Subspace, SubspaceTable, MappedTable, StoredTable, Tablebase
│   ├── storage.rs                   # save/load .bin (format magic="MTBL" + header 32B + verdict + dtw LE)
│   ├── wave.rs                      # solve_movement + multi-valued retrograde wave (Phase 1)
│   ├── work_unit.rs                 # WorkUnit (Phase 2 ready)
│   ├── gevay/
│   │   ├── mod.rs                   # submodule scaffold (Phase 2)
│   │   └── subspace_rank.rs         # val_s + ranking (Phase 2)
│   └── bin/
│       ├── wave_33.rs               # (3,3) cross-check vs Python fixture, ~4s
│       ├── wave_43.rs               # (3,3) + (4,3) cross-subspace, ~10s
│       └── build_movement.rs        # full DAG driver — la commande principale
└── tests/
    ├── rules_test.rs                # 11 tests
    ├── symmetry_test.rs             # 7 tests
    ├── hash_test.rs                 # 8 tests
    └── storage_test.rs              # 2 tests
```

## Insights clés (déjà appliqués mais à garder en tête)

1. **Canonicalisation D4** — chaque position est mappée à son représentant canonique sous les 8 symétries D4 du board. Réduit le compute par 8x et la RAM physique par 8x (les slots non-canoniques ne sont jamais touchés → pages non-allouées par le kernel).

2. **Le wave en deux phases** :
   - **Init** : énumère canoniques, classe les enfants cross-subspace via `tb.query`, set `count(p) = nombre d'enfants intra` pour ceux qui sont UNKNOWN
   - **Wave** : BFS FIFO sur les états résolus, propage aux parents intra, DTW correct par ordre FIFO

3. **Cross-subspace** : un coup avec capture descend vers `(w, b-1)` ou `(w-1, b)`. Ces sous-espaces sont résolus AVANT (DAG ordering). `tb.query` les lit via mmap.

4. **STM (Side-To-Move)** — chaque position a 2 slots (STM=white, STM=black). Verdicts au point de vue du STM. **Invariant A** : `verdict(p, stm) = verdict(swap_colors(p), 3-stm)`.

5. **Work units (Phase 2)** — sous-espaces `s` et `-s` (négation = swap (w_b,b_b) et (w_p,b_p)) traités ensemble car les sliding moves bouclent entre eux (DAG cassé par les cycles).

6. **val_s** (Phase 2) — formule du papier Section IV-A :
   ```
   val_s = (W_s + L_{-s} + D_s/2 + D_{-s}/2) / (T_s + T_{-s})
   ```
   Pour ESC : val = 0.5 → rank 0. Pour pairs : val + val_neg = 1 → ranks antipodaux.

## Statistiques de référence à savoir

**(3,3) avec vol** (validé byte-identical contre Python fixture) :
- Per-STM : WIN 82.92%, LOSS 16.93%, DRAW 0.15%
- Max DTW : 26

**Distribution attendue à matricerlle compteur** (paper Table V, standard, white-to-move) :
- (4,4) → ~100% DRAW
- (6,6) → 18% décisif
- (7,7) → 62% décisif
- (9,9) → 81% décisif

**Phase 2 attendu** (paper Section V) :
- 64% des draws ont V_Gévay = 0
- Sous-espace 6,3,0,0 a rank 112 (15.6% draws non-nuls)
- 8,9,0,0 a été manuellement corrigé par les auteurs (formule surévaluait)
- Win rate vs adversaire faible : 57% (vs 17% Phase 1 seule)

## Prochaines étapes (par priorité)

1. ✅ Build + tests pass après les fixes — vérifier que `cargo test --release` est tout vert
2. ✅ Commit le fix u64 + mmap + scaffolding Phase 2
3. ⏳ Lancer `build_movement -- 18` jusqu'au bout (~2-4h estimé avec les fixes)
4. ⏳ **Phase 2** : implémenter
   - `src/gevay/multi_value.rs` — multi-valued retrograde (Section IV-B du papier)
   - `src/gevay/dtw_adjusted.rs` — DTW sign-aware direction (Section IV-B-2)
   - `src/storage.rs` — extension avec `payload_type` byte
   - `src/bin/compute_gevay.rs` — driver Phase 2
5. ⏳ Cross-check vs Table V (W/D/L percents) puis Table VIII (V_Gévay distribution)
6. ⏳ **Extension étouffement** (différée par décision utilisateur) : `1 - n_legal_opp / max_moves` combiné avec V_Gévay
7. ⏳ **Phase 3 RL** : design et impl

## Référence papier

Gévay & Danner 2014, arXiv 1408.0032v3.
- Section II : partitioning state space, work units
- Section III : basic retrograde algorithm + handling 3+ outcomes
- Section IV : ultra-strong solution
  - IV-A : heuristic val_s + ordinal ranking
  - IV-B : multi-valued retrograde with relative values
  - IV-B-1 : storing relative values
  - IV-B-2 : DTW generalization (direction by sign of first key)
- Section V : results, Tables I-VIII

## Memory files locaux (machine de développement uniquement)

Ces fichiers existent sur `~/.claude/projects/-home-clement-projets-MARL/memory/` et capturent le contexte utilisateur. **Pas accessibles depuis une autre machine.** Pour une session fraîche, ce HANDOFF.md suffit.

- `gasser_gevay_session_handoff.md` (déjà mis à jour cette session)
- `architecture_map.md`
- `no_laziness_decisions.md`
- `draw_attractor_wall.md`
- ...

## CLAUDE.md du projet

À lire à la racine du projet : `/home/clement/projets/MARL/CLAUDE.md`. C'est le document directeur du projet de recherche complet (objectif "cracker l'opening de Morris", contraintes, philosophie).
