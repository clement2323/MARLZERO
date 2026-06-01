# Decision 002 — Phase 1: Gasser tablebase (with flying)

**Status**: draft, discussion in progress
**Date opened**: 2026-06-01
**Owner**: Clément

## Context

Phase 3 self-play has plateaued at ~0.65 vs minimax-d3, blocked by the draw attractor wall — see [draw-attractor-wall memory](../../../../.claude/projects/-home-clement-projets-MARL/memory/draw_attractor_wall.md). Morris is provably drawn (Gasser 1996), so self-play data is 95%+ drawn and the value head collapses.

The strategic pivot: build the full Gasser retrograde tablebase as the ground-truth signal for RL. Phase 2 then layers Gévay's ultra-strong tension score V_Gévay on top. Phase 3 RL operates only on the 18-ply opening, using V_Gévay (after a midgame lookahead) as a dense graded reward — breaking the draw attractor by construction.

This decision doc scopes Phase 1 only: reproducing Gasser's tables for the **classical (with-flying)** variant.

## Why with-flying first (not no-flying)

The codebase currently implements no-flying ([src/morris_rl/env/rules.py](../../src/morris_rl/env/rules.py)). The temptation is to stay there. We don't:

1. **Cross-checkable**: Gasser 1996 publishes counts per subspace (e.g. 3,575,011 draws in 6,3,0,0). Without an external reference for no-flying, we'd have no way to validate correctness.
2. **Gévay compatibility**: Gévay & Danner 2014's V_Gévay numbers and Table IX evaluation are with-flying. Reproducing them validates Phase 2.
3. **No-flying becomes a research extension**: "How does V_Gévay distribution shift when flying is removed?" is a clean follow-up contribution.

The Rust crate exposes a `flying: bool` flag in its rules module from day 1 so the no-flying variant is a configuration switch, not a rewrite.

## Decisions locked in this doc

| Item | Choice |
|---|---|
| Language for table computation | Rust + PyO3, separate crate `morris_tablebase/` |
| Game variant | Classical Morris **with flying** (3-pieces side can jump anywhere) |
| DTW width | 16 bits per position |
| Best-move stored | Yes, 1 byte (action index ∈ [0..79]) |
| Slot size | **4 bytes**: 16 bits verdict+DTW, 8 bits best-move, 8 bits reserved (aligned) |
| Total storage estimate | ~4.4 GB on disk (movement + placement, both STM) |
| Storage | One `.bin` file per subspace, memory-mapped at runtime |
| Symmetry group | D4 dihedral × color swap = 16 elements (matches existing 16x in [symmetries.py](../../src/morris_rl/env/symmetries.py)) |
| Hash scheme | Canonicalize (16 transforms, take lex-min) + combinatorial unrank |
| Movement subspaces | 28: (w, b, 0, 0) with 3 ≤ w ≤ b ≤ 9 |
| Placement subspaces | All (w_b, b_b, w_p, b_p) with constraints — count TBD |
| Cross-check baseline | Gasser 1996 published subspace counts |

## Position encoding

Each position is a 48-bit key:
- 24 bits whites bitmask (1 if white piece, 0 otherwise)
- 24 bits blacks bitmask (1 if black piece, 0 otherwise)

Implicit: side-to-move stored separately (1 bit). For tablebase indexing we store separate tables per side-to-move OR pack STM into the canonical form. Decision: separate tables per STM keeps lookup simpler — 2x the storage but at ~1 GB total it's negligible.

## Canonicalization

Given a position `p = (whites_bb, blacks_bb)`:
1. Generate the 16 symmetric variants under D4 × color_swap
2. Take the one with smallest `(whites_bb, blacks_bb)` tuple under lex comparison
3. This is the **canonical representative** of `p`'s orbit

The 16 transforms are precomputed permutations of the 24 board indices. Applying one = bit-twiddling parallel lookup over 24 bits. Each canonicalization is ~16 transforms × 24-bit permute = ~400 ns.

**Note on orbit sizes**: most positions have orbit size 16. A small minority (positions with internal symmetry) have orbit size in {1, 2, 4, 8}. Burnside's lemma gives the exact count of unique orbits, slightly above `N_raw / 16`. We do **not** build a perfect minimal hash that accounts for this; we accept ~5% wasted slots in exchange for trivial implementation.

## Combinatorial unrank

Given a canonical `(whites_bb, blacks_bb)` in subspace (w, b):
- Let whites positions be `{p_1 < p_2 < ... < p_w}` (sorted indices)
- `idx_whites = Σ_{i=1..w} C(p_i, i)` (combinatorial ranking)
- Then map remaining 24-w positions to compact indices 0..23-w, and similarly for blacks
- `idx_blacks` computed on the compact space
- Final: `idx = idx_whites × C(24-w, b) + idx_blacks`

Lookup is O(1) with precomputed Pascal triangle.

## Storage layout

```
data/tablebase/
├── flying/
│   ├── meta.json                       # subspace counts, hash params, build info
│   ├── movement_w3_b3_stm1.bin         # STM = white to move
│   ├── movement_w3_b3_stm2.bin
│   ├── ...
│   ├── movement_w9_b9_stm1.bin
│   └── placement_w0_b0_wp9_bp9_stm1.bin    # initial position
```

Each `.bin`:
- **Header** (32 bytes, padded):
  ```
  offset  bytes  field
  0       4      magic = b"MTBL"
  4       2      version (u16, currently 1)
  6       1      variant (0 = flying, 1 = no_flying)
  7       1      stm (1 = white, 2 = black)
  8       1      w (white on board)
  9       1      b (black on board)
  10      1      w_to_place
  11      1      b_to_place
  12      4      reserved (zero)
  16      8      n_slots (u64)
  24      4      header_crc32 (u32 over bytes 0..24)
  28      4      reserved (zero)
  ```
- **Payload**: `n_slots × 4 bytes` per slot:
  ```
  byte 0-1 (u16):  bits 15-2 = DTW (0..16383),  bits 1-0 = state
  byte 2   (u8):   best_move action index (0..79), or 0xFF if no defined move (e.g., DRAW)
  byte 3   (u8):   reserved (zero)
  ```
- File is mmap'd read-only at runtime.

Total disk: ~4.4 GB across all subspaces (both STM). Gitignored. Generated by the crate, persisted locally.

## Wave algorithm (retrograde)

Computed per subspace, in topological order of the subspace DAG.

**Per-position state during computation** (16 bits):
```
bits 15-2: count (during init/wave) → DTW (after resolution)
bits  1-0: state ∈ {UNKNOWN, WIN, LOSS, DRAW}
```
"WIN" and "LOSS" are always from side-to-move's perspective.

**Initialization** (one pass over the subspace):
- Terminal positions:
  - ≤ 2 pieces for STM → LOSS, DTW = 0
  - No legal move for STM → LOSS, DTW = 0 (stalemate; reachable in this variant only after a block — with flying it's rarer than in no-flying)
- Non-terminal positions: state = UNKNOWN, count = number of legal forward moves
- Push terminals to queue

**Propagation** (until queue empty):

For each just-resolved position `p` popped from queue, enumerate its parents `q` (positions from which a single move reaches `p`). For each `q`:

- If `p` is LOSS: the side-to-move at `q` (opposite of STM at `p`) can move to `p` and put the opponent in a losing position. So `q` is **WIN**. Mark `q` as WIN, `DTW(q) = DTW(p) + 1` (or `min` if already WIN). Push `q`.
- If `p` is WIN: this child of `q` is bad for the side-to-move at `q`. Decrement `count(q)`. If `count(q) == 0`, all children of `q` are wins for the opponent → `q` is **LOSS**. Mark, `DTW(q) = max DTW of children + 1`. Push.

**Termination**: queue empty → all remaining UNKNOWN positions become DRAW. Single pass to relabel.

## Parent generation (inverse moves)

The trickiest piece of code. Given `q`, enumerate all positions `p` from which a single legal move reaches `q`.

**Cases**:

1. **Reverse placement** (placement phase only): the last move added a piece of the side that just moved. So:
   - For each position with the most recently placed piece, remove it → gives one candidate parent
   - If the placement formed a mill, the move also captured an opponent piece. Inverse: restore the captured piece. For each empty square (up to 24), produce a parent variant with that square re-occupied by the opponent.
   - Constraint: the captured piece couldn't have been in an opponent mill at parent time (unless all opponent pieces were in mills). Filter.

2. **Reverse adjacent movement** (movement phase, no flying, OR flying-eligible side moving 1 step):
   - For each piece of the side-that-just-moved at position `dst`, for each adjacent position `src` that is currently empty: produce parent where piece is at `src`.
   - Capture inverse: same as above.

3. **Reverse flying movement** (movement phase, side currently at 3 pieces is the one that just moved):
   - For each piece of that side at `dst`, for each empty position `src` (any of 24): produce parent.
   - Constraint: parent must have had the moving side at exactly 3 pieces. So if the side has > 3 pieces in `q` it's not a flying parent.
   - Capture inverse: same.

In Rust this is one function returning `impl Iterator<Item = Position>`. Bounded ~50 parents per position; typically 10-20.

**Cross-subspace parents**: if `q ∈ (w, b)` and the inverse move involves restoring a captured piece, the parent is in `(w, b+1)` or `(w+1, b)`. Those subspaces have NOT been computed yet at the time we process `(w, b)` — but we don't WRITE to them during propagation of `(w, b)`. The DAG runs the other direction: we compute small subspaces first, then larger ones use them as **child lookups** during their own init phase to determine counts and detect already-resolved successors.

So the dependency is: when processing subspace S, successors of S's positions may be in S OR in any **smaller** subspace (already done). Parents are in S or **larger** subspaces (we don't touch them, they'll handle themselves later).

## Subspace DAG and ordering

**Movement subspaces** (28 unique with color swap), ordered by total pieces:
- 6 pieces total: (3,3)
- 7 pieces: (3,4)
- 8 pieces: (3,5), (4,4)
- 9 pieces: (3,6), (4,5)
- ... up to (9,9) at 18 pieces

Process strictly bottom-up by total piece count, breaking ties arbitrarily within a total.

**Placement subspaces**: parameterized by (w_board, b_board, w_to_place, b_to_place) with:
- 0 ≤ w_board + w_to_place ≤ 9 and 0 ≤ b_board + b_to_place ≤ 9
- w_to_place + b_to_place > 0 (otherwise we're in movement)
- Symmetry: swap colors gives (b_b, w_b, b_p, w_p) STM-flipped

Order: process after ALL movement subspaces. Within placement, also bottom-up by total pieces remaining (largest to_place → smallest), so a placement subspace's children (one more piece placed, possibly with capture) are always already resolved.

Exact count of placement subspaces and their sizes — to be computed when implementing. Order of magnitude: ~50-100M unique positions across placement, vs ~475M for movement.

## Rust crate layout

```
morris_tablebase/
├── Cargo.toml
├── pyproject.toml                  # maturin config
├── src/
│   ├── lib.rs                      # crate entry, PyO3 module registration
│   ├── board.rs                    # Position struct, bitmask ops
│   ├── rules.rs                    # legal moves, mill detection, terminal check (flying: bool flag)
│   ├── symmetry.rs                 # 16 transforms, canonicalize()
│   ├── hash.rs                     # combinatorial rank/unrank, Pascal precompute
│   ├── subspace.rs                 # Subspace struct, DAG ordering, iteration
│   ├── wave.rs                     # retrograde propagation
│   ├── parents.rs                  # inverse move generation
│   ├── storage.rs                  # mmap I/O, header format
│   └── python_bindings.rs          # PyO3 exports
└── tests/
    ├── rules_test.rs               # legal moves, mill detection
    ├── symmetry_test.rs            # roundtrip + Burnside count check
    ├── hash_test.rs                # unrank uniqueness, bijection on canonicals
    └── wave_small_test.rs          # (3,3) computed by wave matches hand-derived
```

Python integration:
```python
import morris_tablebase as tb
tb.build_all(output_dir="data/tablebase/flying/", num_threads=14)
verdict, dtw = tb.query(whites_bb=0b..., blacks_bb=0b..., stm=1)
```

## Validation strategy

We use Gasser-with-flying as ground truth (published per-subspace counts in Gasser 1996).

1. **Per-subspace counts**: compare our `|positions| × verdict counts` against Gasser 1996's published numbers. Any mismatch = bug.
2. **Symmetry invariants**:
   - Invariant A (rule symmetry): `verdict(swap_colors(p), 3-stm) == verdict(p, stm)`. Always identity (NOT flip).
   - D4 invariance: `verdict(σ(p), stm) == verdict(p, stm)` for all 8 dihedral σ.
3. **Self-play**: two agents both consulting the table play from each starting position. Expected: 100% draws. Any decisive outcome = correctness bug.
4. **(3,3) reference fixture** (see Python spike below): used for Rust regression.
5. **DTW monotonicity**: `DTW(parent) = DTW(best_child) + 1` along optimal play paths.

### Python spike validation (2026-06-01)

A pure-Python brute-force spike of the (3,3,0,0) subspace runs in [scripts/spike_gasser_33.py](../../scripts/spike_gasser_33.py). It implements the wave algorithm without symmetry reduction (raw 2.7M positions per STM) and self-validates:

- **Invariant A passes** on all 5,383,840 states → wave + parent enumeration are correct.
- **Verdict distribution per STM**:
  - WIN: 2,232,160 (82.92%)
  - LOSS: 455,648 (16.93%)
  - DRAW: 4,112 (0.15%)
- **Distribution identical between STM=WHITE and STM=BLACK** at the count level (not just percentage) — strong consistency signal.
- **DTW pattern correct**: WINs at odd DTW (1, 3, 5, ...), LOSSes at even DTW (2, 4, 6, ...). Instant-WIN at DTW=1 represents the 1,056,096 positions where STM has a mill-completing flying move.
- **Runtime**: 5 minutes (131s init + 174s wave) in pure Python. Provides the ceiling we beat in Rust.

These verdict counts will be a regression-test fixture: any Rust implementation of (3,3) must produce the same numbers.

## Compute and memory budget

**Storage**:
- Movement: 475M positions × 2 bytes = ~950 MB
- Placement: ~75M positions × 2 bytes = ~150 MB
- × 2 (STM=white, STM=black) = ~2.2 GB on disk

**Peak RAM during computation**:
- Current subspace table fully in RAM
- All smaller subspaces (children) mmap'd (lazy paging)
- Wave queue + parent buffers: O(|subspace|) worst case
- Peak estimate: 3-4 GB. Trivial on 64 GB.

**Compute**:
- Rust + rayon parallelism over 14 worker threads (16 cores, leave 2 for OS)
- Estimate: 8 hours total for all subspaces. Dominated by the larger movement subspaces (6,6), (6,7), (7,7).

## Resolved sub-decisions

### Best-move stored

Each slot stores 1 byte for the action index of the optimal move from that position. Cost: +25% storage (4 bytes/slot instead of 3), negligible at our scale. Benefit: zero-search perfect play at inference, plus simpler Phase 3 RL data generation (immediate access to the "true" move without re-walking children).

For DRAW positions there's typically no unique optimal move (several plies preserve the draw). We store `0xFF` as sentinel meaning "any drawing move works — re-derive at lookup time if needed".

### Inverse capture rule

The "can't remove from opponent mill unless all opponent pieces are in mills" rule needs explicit handling in `parents.rs` when restoring a captured piece during inverse move generation:

For each candidate empty position `pos` where we'd restore the captured opponent piece, verify the capture would have been legal at parent time:
- Reconstruct parent state: `q + remove(placed_piece) + add(opponent_piece_at_pos)`
- Check: was `opponent_piece_at_pos` in a mill in the parent state?
- If yes: was at least ONE opponent piece NOT in a mill?
- If both yes → this parent is INVALID (the original forward capture would have been illegal). Skip.
- Otherwise → valid parent candidate.

Implementation: ~30 lines, exercised by the (3,3) hand-verified test fixture.

## Next milestones (Phase 1)

1. **Spike** (1-2 days): Python prototype computing (3,3) only, no symmetry, no hash — pure proof-of-algorithm for the wave. Output: verdict counts for (3,3) for hand-verification.
2. **Rust crate skeleton** (1 day): cargo init, PyO3 hello-world, board/rules/symmetry modules with tests.
3. **Hash + canonicalize** (2 days): full bijection on (3,3) and (4,4), roundtrip tests.
4. **Wave on (3,3) in Rust** (2-3 days): single subspace, then chain through to (4,4).
5. **Full DAG sweep** (1 week): all 28 movement subspaces, then placement.
6. **Validation pass** (2-3 days): Gasser cross-check, symmetry invariants, hand-verified spots.

Total elapsed: ~3-4 weeks before Phase 2 (Gévay) can start on top.
