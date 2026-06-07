//! Section IV-B — Multi-valued retrograde wave that classifies draws.
//!
//! Generalises the Phase 1 wave from 3 outcomes {W, L, D} to integer
//! outcomes in `[-W, +W]` (where `W` is the number of distinct subspace
//! ranks). The processing order is **descending |value|** then ascending
//! DTW within each |value|, so we resolve the most extreme classes first.
//!
//! Per Section IV-B-1, values are stored **relative to the current
//! subspace's rank**: a position with relative value 0 means "ends up in
//! a subspace of the same rank as ours". Positive relative = ends in a
//! BETTER subspace, negative relative = ends in a WORSE subspace. When
//! propagating across subspaces of different ranks we adjust the relative
//! value by their difference.
//!
//! Per Section IV-B-2, DTW direction depends on the sign of the first key
//! (the value):
//! - first key > 0 (in a better subspace) → **maximise** DTW (stay there
//!   as long as possible)
//! - first key < 0 (in a worse subspace) → **minimise** DTW (escape fast)
//! - first key changes sign on adjustment → **negate** DTW
//!
//! At this stage we implement the bookkeeping primitives; full integration
//! into the cross-subspace driver lives in `bin/compute_gevay.rs`.

use std::sync::atomic::{AtomicI16, AtomicU16, AtomicU64, Ordering};

/// Pack-aware helpers for boolean state arrays. We store one bit per
/// canonical state in a `Vec<u64>` instead of `Vec<bool>` (×8 RAM saving)
/// — the wave only needs `is_set`, `set_true`, and an atomic `fetch_or`
/// during the parallel init pass.
///
/// The atomic variant uses `&[AtomicU64]` over the same underlying memory;
/// `AtomicU64::fetch_or` makes concurrent bit-sets safe even when multiple
/// threads target distinct bits of the same word.
#[inline]
fn bit_get(words: &[u64], i: usize) -> bool {
    (words[i >> 6] >> (i & 63)) & 1 != 0
}

#[inline]
fn bit_set_true(words: &mut [u64], i: usize) {
    words[i >> 6] |= 1u64 << (i & 63);
}

#[inline]
fn bit_atomic_set_true_relaxed(words: &[AtomicU64], i: usize) {
    words[i >> 6].fetch_or(1u64 << (i & 63), Ordering::Relaxed);
}

#[inline]
fn bit_words_for(n: usize) -> usize {
    (n + 63) >> 6
}

use rayon::prelude::*;

use crate::board::{ADJACENCY, NUM_POSITIONS};
use crate::rules::{is_mill_through, legal_capture_targets, popcount};
use crate::subspace::{Subspace, Tablebase};
use crate::wave::STM_WHITE;
use crate::work_unit::WorkUnit;

use super::subspace_rank::Rank;

/// Multi-valued cell state: union of `count(remaining unresolved children)`
/// and `value(first key, DTW)`. Phase 2 needs signed first keys; we encode
/// the count/value discriminator via a high-bit flag on `data`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellState {
    /// Still waiting for `remaining` intra-WU children to resolve.
    Count { remaining: u32 },
    /// Resolved. `first_key` is RELATIVE to the cell's own subspace rank;
    /// see Section IV-B-1. `dtw` is signed per the IV-B-2 direction rule.
    Resolved { first_key: i16, dtw: i16 },
}

impl Default for CellState {
    fn default() -> Self {
        CellState::Count { remaining: 0 }
    }
}

/// Per-position state during a single work-unit's wave run. The whole
/// `Vec<CellState>` is indexed by `Subspace::state_index_canonical`.
pub struct GevayWorkArea {
    pub primary: Vec<Subspace>,
    /// `cells[i]` = state for the i-th primary subspace (parallel to `primary`).
    pub cells: Vec<Vec<CellState>>,
    pub ranks: Vec<Rank>, // one per primary subspace
}

impl GevayWorkArea {
    pub fn new(wu: &WorkUnit, ranks_per_sub: &[Rank]) -> Self {
        debug_assert_eq!(wu.primary.len(), ranks_per_sub.len());
        let cells: Vec<Vec<CellState>> = wu
            .primary
            .iter()
            .map(|s| vec![CellState::default(); s.n_states() as usize])
            .collect();
        Self {
            primary: wu.primary.clone(),
            cells,
            ranks: ranks_per_sub.to_vec(),
        }
    }

    /// Lookup `(sub_idx, state_idx)` -> &CellState.
    pub fn get(&self, sub_idx: usize, state_idx: u64) -> &CellState {
        &self.cells[sub_idx][state_idx as usize]
    }

    pub fn get_mut(&mut self, sub_idx: usize, state_idx: u64) -> &mut CellState {
        &mut self.cells[sub_idx][state_idx as usize]
    }
}

/// Priority-queue key used by the wave. The paper sorts by
/// `(negated |first_key|, DTW)`, which we implement as a min-heap on
/// `(neg_abs_first_key, dtw)`. Inside a single |value| slice we still
/// want ascending DTW order — same as Phase 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueueKey {
    /// `-|first_key_absolute|` — extreme values come out first.
    pub neg_abs_value: i16,
    pub dtw: i16,
}

impl Ord for QueueKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // BinaryHeap is a MAX-heap; we want SMALL `neg_abs_value` and SMALL `dtw` first.
        // Invert by comparing other to self.
        other
            .neg_abs_value
            .cmp(&self.neg_abs_value)
            .then(other.dtw.cmp(&self.dtw))
    }
}

impl PartialOrd for QueueKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Queue entry. `sub_idx` distinguishes which primary subspace the state
/// lives in (0 or 1 for non-ESC pairs).
#[derive(Debug, Clone, Copy)]
pub struct QueueEntry {
    pub key: QueueKey,
    pub sub_idx: u8,
    pub state_idx: u64,
}

impl PartialEq for QueueEntry {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

impl Eq for QueueEntry {}

impl PartialOrd for QueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for QueueEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.key.cmp(&other.key)
    }
}

/// Adjust a first key crossing subspaces of different ranks.
///
/// `from_rank` = rank of the subspace the value was stored in (relative form),
/// `to_rank` = rank of the destination subspace.
/// Returns `(adjusted_key, dtw_should_negate)`.
#[inline]
pub fn adjust_first_key(value_rel: i16, from_rank: Rank, to_rank: Rank) -> (i16, bool) {
    // Step 1: convert to absolute (the rank of the destination subspace the
    // optimal path leads to, from the FROM-subspace's perspective).
    //   value_abs = value_rel + from_rank
    // Step 2: negate (the opponent at the destination will be optimizing
    // their own outcome, which is the opposite of ours).
    //   value_abs_neg = -value_abs
    // Step 3: convert back to relative (now from the TO-subspace's perspective).
    //   value_rel_new = value_abs_neg - to_rank
    let abs_from = value_rel as i32 + from_rank as i32;
    let rel_to = -abs_from - to_rank as i32;
    let signed_rel_to = rel_to as i16;
    // Paper IV-B-2: "when a first key changes sign during an adjustment,
    // we negate the second one [DTW]". First key = the RELATIVE value
    // that the cell stores; compare value_rel (before) vs signed_rel_to (after).
    let sign_flipped = (value_rel as i32).signum() != rel_to.signum()
        && value_rel != 0
        && rel_to != 0;
    (signed_rel_to, sign_flipped)
}

/// DTW propagation respecting the sign rule (Section IV-B-2).
///
/// If the moving position has first key > 0, we increment DTW (we want
/// to stay long). If < 0, we decrement (paper formulation uses negation
/// at every step to keep arithmetic symmetric — we mirror that here).
#[inline]
pub fn propagate_dtw(child_dtw: i16, child_first_key_abs: i16) -> i16 {
    if child_first_key_abs > 0 {
        child_dtw.saturating_add(1)
    } else if child_first_key_abs < 0 {
        child_dtw.saturating_sub(1)
    } else {
        // |first_key| = 0 means "we landed in a 0-rank class" — DTW carries through unchanged.
        child_dtw
    }
}

/// Generate move successors for a position in a movement subspace,
/// classifying each child into `(target_sub, new_wbb, new_bbb, new_stm)`.
///
/// Used both by init (forward enumeration) and by parent generation
/// (inverse moves). Variant flag is whether 3-stone sides can fly.
pub fn forward_moves<F: FnMut(MoveTarget)>(
    sub: Subspace,
    wbb: u32,
    bbb: u32,
    stm: u8,
    variant: crate::wave::Variant,
    mut on_move: F,
) {
    let (stm_bb, opp_bb) = if stm == STM_WHITE { (wbb, bbb) } else { (bbb, wbb) };
    let stm_count = popcount(stm_bb);
    let can_fly = variant == crate::wave::Variant::Flying && stm_count == 3;
    let occupied = wbb | bbb;
    let empties = !occupied & ((1u32 << NUM_POSITIONS) - 1);

    let mut s = stm_bb;
    while s != 0 {
        let src = s.trailing_zeros() as u8;
        s &= s - 1;
        let after_lift = stm_bb & !(1u32 << src);
        let dests = if can_fly {
            empties
        } else {
            let mut m = 0u32;
            for &p in &ADJACENCY[src as usize] {
                if p == 0xFF {
                    break;
                }
                if (empties >> p) & 1 != 0 {
                    m |= 1u32 << p;
                }
            }
            m
        };
        let mut d = dests;
        while d != 0 {
            let dst = d.trailing_zeros() as u8;
            d &= d - 1;
            let new_stm = after_lift | (1u32 << dst);
            let forms_mill = is_mill_through(new_stm, dst);
            if forms_mill {
                let cap_targets = legal_capture_targets(opp_bb);
                let mut c = cap_targets;
                while c != 0 {
                    let cap = c.trailing_zeros();
                    c &= c - 1;
                    let new_opp = opp_bb & !(1u32 << cap);
                    let opp_new_count = popcount(new_opp);
                    let (new_wbb, new_bbb) = if stm == STM_WHITE {
                        (new_stm, new_opp)
                    } else {
                        (new_opp, new_stm)
                    };
                    let target_sub = if opp_new_count < 3 {
                        // Terminal: opponent below 3 pieces.
                        Subspace::movement(
                            popcount(new_wbb) as u8,
                            popcount(new_bbb) as u8,
                        )
                    } else if stm == STM_WHITE {
                        Subspace::movement(sub.w_board, opp_new_count as u8)
                    } else {
                        Subspace::movement(opp_new_count as u8, sub.b_board)
                    };
                    on_move(MoveTarget {
                        target_sub,
                        new_wbb,
                        new_bbb,
                        new_stm: 3 - stm,
                        captured: true,
                        opp_below_three: opp_new_count < 3,
                    });
                }
            } else {
                let (new_wbb, new_bbb) = if stm == STM_WHITE {
                    (new_stm, opp_bb)
                } else {
                    (opp_bb, new_stm)
                };
                on_move(MoveTarget {
                    target_sub: sub,
                    new_wbb,
                    new_bbb,
                    new_stm: 3 - stm,
                    captured: false,
                    opp_below_three: false,
                });
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MoveTarget {
    pub target_sub: Subspace,
    pub new_wbb: u32,
    pub new_bbb: u32,
    pub new_stm: u8,
    pub captured: bool,
    pub opp_below_three: bool,
}

/// Sentinel absolute values for terminal verdicts. The paper says wins and
/// losses receive values "just outside the range of the values calculated
/// by the above ranking method" (Section IV-A). For movement subspaces the
/// max rank magnitude is ~21 (21 non-ESC pairs), so 30 is comfortably outside.
pub const WIN_ABS: i16 = 30;
pub const LOSS_ABS: i16 = -30;

/// Look up a position's resolved value in a secondary (already-solved)
/// subspace, applying the cross-subspace adjustment.
///
/// Returns `(adjusted_first_key, dtw)` from the perspective of the caller's
/// (primary) subspace. The caller can then compare with the current rank.
pub fn query_secondary_adjusted(
    tb: &Tablebase,
    secondary_rank: Rank,
    primary_rank: Rank,
    sub: Subspace,
    wbb: u32,
    bbb: u32,
    stm: u8,
) -> Option<(i16, i16)> {
    let (verdict, dtw) = tb.query(sub, wbb, bbb, stm)?;
    // Phase 1 verdict mapping (paper IV-A: "Wins and losses receive values
    // just outside the range of the values calculated by the above ranking
    // method.").
    let abs_first_key = match verdict {
        crate::wave::WIN => WIN_ABS,
        crate::wave::LOSS => LOSS_ABS,
        crate::wave::DRAW => secondary_rank, // stable-draw secondary, value = its subspace rank
        _ => return None,
    };
    let (adj, sign_flip) = adjust_first_key(abs_first_key, secondary_rank, primary_rank);
    let signed_dtw = if sign_flip { -(dtw as i16) } else { dtw as i16 };
    Some((adj, signed_dtw))
}

/// Solve a single ESC work unit (one primary subspace where `s == -s`).
///
/// Produces `(first_key, dtw)` arrays sized like the subspace's state space,
/// indexed by `state_index_canonical`. For the ESC case the work unit rank
/// is always 0 by the paper convention, so relative ↔ absolute is identity.
///
/// Returns Phase 1-equivalent results for ESC subspaces whose draws are all
/// stable (no cross-subspace draw chains worth following). Captures landing
/// in smaller secondary subspaces use [Tablebase] (Phase 1) for now; we'll
/// extend to a Phase 2 secondary tablebase when non-ESC work units come online.
pub fn solve_esc_work_unit(
    sub: Subspace,
    rank: Rank,
    variant: crate::wave::Variant,
    phase1_tb: &Tablebase,
    ranks: &std::collections::HashMap<Subspace, Rank>,
) -> (Vec<i16>, Vec<i16>) {
    // Canonical-only indexing: ×8 smaller allocation than the dense
    // state_index_canonical layout used by Phase 1. The wave never touches
    // non-canonical orbit representatives, so the dense layout was 7/8 dead
    // slots — building a CanonicalIndexer lets us allocate just the live ones.
    // Cost: O(log bucket_size) per state lookup vs O(1) for state_index_canonical.
    // For (9,9) bucket_size <= 5005 → ~12 comparisons per access; not the
    // bottleneck. See [crate::gevay::canonical_indexer].
    let indexer = crate::gevay::canonical_indexer::CanonicalIndexer::build(sub);
    let n = indexer.n_states_canonical() as usize;
    let mut first_key: Vec<i16> = vec![0; n];
    let mut dtw: Vec<i16> = vec![0; n];
    // count: parent-update remaining-children counter. Bounded by the
    // number of intra-WU forward moves from any single position — empirically
    // < 500 across all Morris boards. u16 fits with 130× headroom and halves
    // the per-slot RAM cost.
    let mut count: Vec<u16> = vec![0; n];
    // resolved / has_draw_for_q : bit-packed booleans (1 bit/slot, 1/8 the RAM
    // of Vec<bool>). has_draw_for_q tracks whether ANY child (cross-subspace
    // at init, or intra-subspace during wave) yielded a DRAW-class option.
    // Used at the count==0 transition to choose between DRAW and LOSS.
    let words = bit_words_for(n);
    let mut resolved: Vec<u64> = vec![0; words];
    let mut has_draw_for_q: Vec<u64> = vec![0; words];

    // Phase 0 — init pass in parallel via rayon. Each canonical position
    // is independent: its writes go to a unique pair of state-array
    // indices (WTM + BTM) determined by the canonical (cw, cb), so
    // different threads writing to different positions never collide.
    //
    // We expose the plain Vec<i16>/Vec<u32>/Vec<bool> as atomic-typed
    // slices via std::slice::from_raw_parts for the duration of the
    // parallel pass. AtomicI16/AtomicU32/AtomicBool are repr(transparent)
    // over their plain types so the layouts match; we use Relaxed
    // ordering since each cell is written by exactly one thread and
    // there are no read-modify-write cycles within the init.
    //
    // Queue entries are collected via rayon's fold/reduce so per-thread
    // partial heaps merge once into the final BinaryHeap. This avoids a
    // global mutex on the queue during init.
    //
    // STREAMED ENUMERATION: instead of materializing the full canonical
    // position list (8 bytes × n_canonical_entries — up to 8-17 GB for the
    // biggest pair WUs (7,8)/(8,7), (8,9)/(9,8)), we interleave the
    // enumerate_positions callback with chunked par_iter inits.
    // Caps the temporary buffer at INIT_CHUNK_SIZE × 8 bytes ≈ 8 MB
    // regardless of subspace size. Without this, on a 30 GB machine the
    // wave arrays + indexer + canonical_positions + queue could collide
    // and OOM the process mid-init.
    const INIT_CHUNK_SIZE: usize = 1 << 20; // 1M canonical positions per batch
    let mut queue_entries: Vec<QueueEntry> = Vec::new();
    {
        // SAFETY: AtomicI16 / AtomicU16 / AtomicU64 are repr(transparent)
        // over i16 / u16 / u64, so a &[Atomic*] over the same memory is
        // sound. The atomic borrows live only for the duration of this
        // scope; afterwards the plain Vec<...> is the unique borrow.
        // The resolved / has_draw_for_q bitvecs are 1 bit per state; their
        // atomic slice has length = number of u64 words, and bit-level
        // updates use fetch_or so concurrent threads targeting different
        // bits of the same word never collide.
        let first_key_atomic: &[AtomicI16] = unsafe {
            std::slice::from_raw_parts(first_key.as_ptr() as *const AtomicI16, n)
        };
        let dtw_atomic: &[AtomicI16] = unsafe {
            std::slice::from_raw_parts(dtw.as_ptr() as *const AtomicI16, n)
        };
        let count_atomic: &[AtomicU16] = unsafe {
            std::slice::from_raw_parts(count.as_ptr() as *const AtomicU16, n)
        };
        let resolved_atomic: &[AtomicU64] = unsafe {
            std::slice::from_raw_parts(resolved.as_ptr() as *const AtomicU64, words)
        };
        let has_draw_atomic: &[AtomicU64] = unsafe {
            std::slice::from_raw_parts(has_draw_for_q.as_ptr() as *const AtomicU64, words)
        };

        let mut chunk: Vec<(u32, u32)> = Vec::with_capacity(INIT_CHUNK_SIZE);
        let process_chunk = |chunk: &Vec<(u32, u32)>| -> Vec<QueueEntry> {
            chunk
                .par_iter()
                .fold(
                    Vec::<QueueEntry>::new,
                    |mut acc, &(wbb, bbb)| {
                        for stm in [STM_WHITE, 2u8] {
                            if let Some(entry) = init_one_position_atomic(
                                sub, rank, variant, phase1_tb, ranks, &indexer,
                                wbb, bbb, stm,
                                first_key_atomic, dtw_atomic, count_atomic,
                                resolved_atomic, has_draw_atomic,
                            ) {
                                acc.push(entry);
                            }
                        }
                        acc
                    },
                )
                .reduce(Vec::<QueueEntry>::new, |mut a, mut b| {
                    if a.capacity() < b.capacity() {
                        std::mem::swap(&mut a, &mut b);
                    }
                    a.extend(b);
                    a
                })
        };

        // Init pass progress: each chunk = 1M canonical positions. Print
        // every ~10s so the user knows it's progressing during the silent
        // phase before the wave loop starts (for big WUs the init can
        // take 1-3 min).
        let t_init = std::time::Instant::now();
        let n_canonical_total = indexer.n_canonical_entries() as u64;
        let n_chunks_est = (n_canonical_total + INIT_CHUNK_SIZE as u64 - 1) / INIT_CHUNK_SIZE as u64;
        // eprintln! goes to stderr which Rust leaves UNBUFFERED — visible
        // immediately through `2>&1 | tee` even when stdout is fully block
        // buffered. Same trick used for the wave progress below.
        eprintln!("    init: starting ({} chunks of {}M positions each)",
            n_chunks_est, INIT_CHUNK_SIZE / 1_000_000);
        let mut chunks_done: u64 = 0;
        let mut last_init_println = std::time::Instant::now();
        sub.enumerate_positions(|w, b| {
            chunk.push((w, b));
            if chunk.len() >= INIT_CHUNK_SIZE {
                queue_entries.extend(process_chunk(&chunk));
                chunk.clear();
                chunks_done += 1;
                if last_init_println.elapsed().as_secs() >= 10 {
                    let pct = (chunks_done as f64 / n_chunks_est as f64 * 100.0).min(100.0);
                    eprintln!(
                        "    init: {:>5.1}% ({:>5}/{:>5} chunks) elapsed {:>5.0}s",
                        pct, chunks_done, n_chunks_est, t_init.elapsed().as_secs_f64(),
                    );
                    last_init_println = std::time::Instant::now();
                }
            }
        });
        if !chunk.is_empty() {
            queue_entries.extend(process_chunk(&chunk));
            chunks_done += 1;
        }
        eprintln!("    init: done ({} chunks, {:.1}s, {} queue seeds)",
            chunks_done, t_init.elapsed().as_secs_f64(), queue_entries.len());
    }

    let mut queue: std::collections::BinaryHeap<QueueEntry> = queue_entries.into_iter().collect();

    // Phase 1 — wave propagation through intra-WU parents.
    // Approximate first-arrival heuristic (mirrors the Phase 1 wave's
    // documented ~0.006% DTW asymmetry): when a parent becomes
    // resolvable through its first WIN-class or first all-children-bad
    // path, we commit to that classification rather than tracking the
    // multi-valued maximum exactly. Adequate for V_Gévay rank
    // computation since per-position DTW is not the primary signal.
    //
    // Progress bar: total = n_states_canonical (the BFS resolves each
    // canonical state at most once). Updates every 256k pops to keep
    // overhead negligible; the bar is rendered live on TTY and silently
    // suppressed when stdout is piped (the wave still prints its
    // "wave done in X" summary line either way).
    // Indicatif bar for interactive TTY runs (auto-suppressed when piped).
    let pb_wave = indicatif::ProgressBar::new(n as u64);
    pb_wave.set_style(
        indicatif::ProgressStyle::with_template(
            "    wave [{elapsed_precise} / ETA {eta_precise}] [{bar:30.cyan/blue}] {pos:>11}/{len:>11} queue={msg}",
        )
        .unwrap()
        .progress_chars("=>-"),
    );
    pb_wave.enable_steady_tick(std::time::Duration::from_millis(500));
    let t_wave = std::time::Instant::now();
    let mut pops: u64 = 0;
    let mut last_println = std::time::Instant::now();
    while let Some(entry) = queue.pop() {
        let p_idx = entry.state_idx;
        propagate_to_parents_gevay(
            sub, rank, variant, &indexer, p_idx,
            &mut first_key, &mut dtw, &mut count,
            &mut resolved, &mut has_draw_for_q, &mut queue,
        );
        pops += 1;
        // Fast path: bump the bar every ~256k pops (a few times/sec on TTY).
        // Cheaper than calling Instant::now() every iteration.
        if pops & 0x3FFFF == 0 {
            pb_wave.set_position(pops);
            pb_wave.set_message(format!("{}", queue.len()));
            // Slow path: dump a plain println every ~10s for log/pipe
            // consumers — indicatif bar is auto-hidden behind `| tee`,
            // but stdout text comes through.
            if last_println.elapsed().as_secs() >= 10 {
                let pct = (pops as f64 / n as f64 * 100.0).min(100.0);
                eprintln!(
                    "    wave: {:>5.1}% ({:>11}/{:>11}) elapsed {:>5.0}s queue={:>10}",
                    pct, pops, n, t_wave.elapsed().as_secs_f64(), queue.len(),
                );
                last_println = std::time::Instant::now();
            }
        }
    }
    pb_wave.finish_and_clear();

    // Phase 2 — finalize: unresolved cells default to value 0 (= rank of this WU).
    // Note: the approximate first-arrival wave heuristic can leave a small
    // fraction (~1% of WIN/LOSS positions at (3,3)) misclassified as DRAW
    // because cyclic dependencies between intra-WU positions don't fully
    // unwind. Phase 1 has the same approximate-DTW issue (0.006% color-swap
    // asymmetry) but its verdict classification is exact. We accept this
    // approximation for now; refining the wave to track per-position
    // multi-valued maxima is a future improvement.
    for i in 0..n {
        if !bit_get(&resolved, i) {
            first_key[i] = 0;
            dtw[i] = 0;
        }
    }

    (first_key, dtw)
}

/// Solve a non-ESC work unit (two primary subspaces forming a `(s, -s)`
/// pair with antipodal ranks `(rank, -rank)`). Each primary subspace's
/// wave is independent at the cell level — they share secondary
/// dependencies through `phase1_tb` and `ranks` but no intra-WU edges
/// cross between them. Returns one `(first_key, dtw)` pair per primary
/// in the same order as `wu.primary`.
pub fn solve_pair_work_unit(
    wu: &WorkUnit,
    rank_primary_0: Rank,
    variant: crate::wave::Variant,
    phase1_tb: &Tablebase,
    ranks: &std::collections::HashMap<Subspace, Rank>,
) -> Vec<(Vec<i16>, Vec<i16>)> {
    debug_assert_eq!(wu.primary.len(), 2, "pair WU expected, got {:?}", wu);
    let (fk0, dtw0) = solve_esc_work_unit(
        wu.primary[0], rank_primary_0, variant, phase1_tb, ranks,
    );
    let (fk1, dtw1) = solve_esc_work_unit(
        wu.primary[1], -rank_primary_0, variant, phase1_tb, ranks,
    );
    vec![(fk0, dtw0), (fk1, dtw1)]
}

/// Per-position init pass for one `(wbb, bbb, stm)` triple. Writes
/// directly to atomic slices (each canonical position's writes go to
/// disjoint indices, so Relaxed ordering is sufficient) and returns the
/// optional queue entry to push for the wave's BFS seed set.
#[inline]
fn init_one_position_atomic(
    sub: Subspace,
    rank: Rank,
    variant: crate::wave::Variant,
    phase1_tb: &Tablebase,
    ranks: &std::collections::HashMap<Subspace, Rank>,
    indexer: &crate::gevay::canonical_indexer::CanonicalIndexer,
    wbb: u32,
    bbb: u32,
    stm: u8,
    first_key: &[AtomicI16],
    dtw: &[AtomicI16],
    count: &[AtomicU16],
    resolved: &[AtomicU64],
    has_draw_for_q: &[AtomicU64],
) -> Option<QueueEntry> {
    let idx = indexer.canonical_index(wbb, bbb, stm) as usize;
    let mut best_win_key: Option<(i16, i16)> = None;
    let mut max_lose_rel: Option<(i16, i16)> = None;
    let mut has_draw = false;
    let mut any_move = false;
    let mut intra_children: std::collections::HashSet<u64> = std::collections::HashSet::new();

    forward_moves(sub, wbb, bbb, stm, variant, |mv| {
        any_move = true;
        if mv.target_sub == sub {
            // forward_moves returns raw (wbb, bbb); canonicalize before
            // indexing so the canonical-only layout receives an orbit rep.
            let child_idx = indexer.index(mv.new_wbb, mv.new_bbb, mv.new_stm);
            intra_children.insert(child_idx);
            return;
        }
        let (rel_key, child_signed_dtw) = if mv.opp_below_three {
            let (adj, _sign_flip) = adjust_first_key(LOSS_ABS, 0, rank);
            (adj, 0i16)
        } else {
            let secondary_rank = *ranks.get(&mv.target_sub).unwrap_or(&0);
            match query_secondary_adjusted(
                phase1_tb, secondary_rank, rank,
                mv.target_sub, mv.new_wbb, mv.new_bbb, mv.new_stm,
            ) {
                Some(v) => v,
                None => return,
            }
        };
        if rel_key > 0 {
            let candidate_dtw = propagate_dtw(child_signed_dtw, rel_key)
                .saturating_add(if rel_key > 0 { 1 } else { -1 });
            let curr = best_win_key.unwrap_or((i16::MIN, i16::MAX));
            if rel_key > curr.0 || (rel_key == curr.0 && candidate_dtw < curr.1) {
                best_win_key = Some((rel_key, candidate_dtw));
            }
        } else if rel_key < 0 {
            let candidate_dtw = propagate_dtw(child_signed_dtw, rel_key).saturating_sub(1);
            let curr = max_lose_rel.unwrap_or((i16::MAX, i16::MIN));
            if rel_key > curr.0 || (rel_key == curr.0 && candidate_dtw > curr.1) {
                max_lose_rel = Some((rel_key, candidate_dtw));
            }
        } else {
            has_draw = true;
        }
    });

    if !any_move {
        let (rel, _) = adjust_first_key(LOSS_ABS, 0, rank);
        first_key[idx].store(rel, Ordering::Relaxed);
        // dtw stays 0 (already zero-initialized).
        bit_atomic_set_true_relaxed(resolved, idx);
        return Some(QueueEntry {
            key: QueueKey { neg_abs_value: -(rel.abs()), dtw: 0 },
            sub_idx: 0,
            state_idx: idx as u64,
        });
    }

    if let Some((rel, d)) = best_win_key {
        first_key[idx].store(rel, Ordering::Relaxed);
        dtw[idx].store(d, Ordering::Relaxed);
        bit_atomic_set_true_relaxed(resolved, idx);
        Some(QueueEntry {
            key: QueueKey { neg_abs_value: -(rel.abs()), dtw: d },
            sub_idx: 0,
            state_idx: idx as u64,
        })
    } else if intra_children.is_empty() {
        if has_draw {
            // first_key stays 0; dtw stays 0.
            bit_atomic_set_true_relaxed(resolved, idx);
            None
        } else if let Some((rel, d)) = max_lose_rel {
            let new_dtw = d.saturating_add(1);
            first_key[idx].store(rel, Ordering::Relaxed);
            dtw[idx].store(new_dtw, Ordering::Relaxed);
            bit_atomic_set_true_relaxed(resolved, idx);
            Some(QueueEntry {
                key: QueueKey { neg_abs_value: -(rel.abs()), dtw: new_dtw },
                sub_idx: 0,
                state_idx: idx as u64,
            })
        } else {
            None
        }
    } else {
        count[idx].store(intra_children.len() as u16, Ordering::Relaxed);
        if has_draw {
            bit_atomic_set_true_relaxed(has_draw_for_q, idx);
        }
        if let Some((_rel, d)) = max_lose_rel {
            dtw[idx].store(d, Ordering::Relaxed);
        }
        None
    }
}

/// Adjacency bitmask of a board position (mirror of `wave::adjacency_mask`).
#[inline]
fn adjacency_mask(pos: u8) -> u32 {
    let adj = crate::board::ADJACENCY[pos as usize];
    let mut m = 0u32;
    for &p in &adj {
        if p == 0xFF { break; }
        m |= 1u32 << p;
    }
    m
}

/// Phase 2 inverse-move parent propagation.
///
/// Given a resolved child `p` (position at `p_idx`), find every intra-WU
/// parent `q` (positions whose move yields p) and try to advance q:
/// - Child looks like `-p.first_key` from q's perspective.
/// - If positive (q can WIN through this child): commit q immediately
///   (first-arrival heuristic).
/// - If negative (LOSS option for q): track max DTW; decrement count.
/// - If zero (DRAW option for q): set has_draw flag; decrement count.
/// - When count reaches 0: finalize q as DRAW (if any draw child) or as
///   LOSS with the stashed max DTW.
fn propagate_to_parents_gevay(
    _sub: Subspace,
    _rank: Rank,
    variant: crate::wave::Variant,
    indexer: &crate::gevay::canonical_indexer::CanonicalIndexer,
    p_idx: u64,
    first_key: &mut [i16],
    dtw: &mut [i16],
    count: &mut [u16],
    resolved: &mut [u64],
    has_draw_for_q: &mut [u64],
    queue: &mut std::collections::BinaryHeap<QueueEntry>,
) {
    let (wbb, bbb, stm_p) = indexer.decode(p_idx);
    let p_fk = first_key[p_idx as usize];
    let p_dtw = dtw[p_idx as usize];
    let mover_stm = 3 - stm_p;
    let (mover_bb, fixed_bb) = if mover_stm == STM_WHITE {
        (wbb, bbb)
    } else {
        (bbb, wbb)
    };
    let mover_count = popcount(mover_bb);
    let mover_can_fly = variant == crate::wave::Variant::Flying && mover_count == 3;

    let occupied = wbb | bbb;
    let empties = !occupied & ((1u32 << NUM_POSITIONS) - 1);

    let mut seen_parents: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let mut mb = mover_bb;
    while mb != 0 {
        let dst = mb.trailing_zeros() as u8;
        mb &= mb - 1;
        if is_mill_through(mover_bb, dst) { continue; }
        let pred_mask = if mover_can_fly { empties } else { adjacency_mask(dst) & empties };
        let mut em = pred_mask;
        while em != 0 {
            let src = em.trailing_zeros() as u8;
            em &= em - 1;
            let new_mover = (mover_bb & !(1u32 << dst)) | (1u32 << src);
            let (new_wbb, new_bbb) = if mover_stm == STM_WHITE {
                (new_mover, fixed_bb)
            } else {
                (fixed_bb, new_mover)
            };
            let q_idx = indexer.index(new_wbb, new_bbb, mover_stm);
            if !seen_parents.insert(q_idx) { continue; }
            let q_i = q_idx as usize;
            if bit_get(resolved, q_i) { continue; }
            let p_from_q = -p_fk;
            if p_from_q > 0 {
                // q can WIN through this child — first arrival commits.
                first_key[q_i] = p_from_q;
                dtw[q_i] = p_dtw.saturating_add(1);
                bit_set_true(resolved, q_i);
                queue.push(QueueEntry {
                    key: QueueKey {
                        neg_abs_value: -(p_from_q.abs()),
                        dtw: dtw[q_i],
                    },
                    sub_idx: 0,
                    state_idx: q_idx,
                });
            } else if p_from_q < 0 {
                // LOSS-direction intra-child: decrement count, stash max DTW.
                if p_dtw > dtw[q_i] { dtw[q_i] = p_dtw; }
                if count[q_i] > 0 { count[q_i] -= 1; }
                if count[q_i] == 0 {
                    if bit_get(has_draw_for_q, q_i) {
                        // Resolve as DRAW but do NOT push — DRAW resolutions
                        // do not need to propagate (parents see them via their
                        // own init's has_draw flag if applicable, and intra-
                        // DRAW signaling is handled by *not* decrementing
                        // count, which leaves parents UNRESOLVED → DRAW at
                        // finalize). Mirrors Phase 1.
                        first_key[q_i] = 0;
                        dtw[q_i] = 0;
                        bit_set_true(resolved, q_i);
                    } else {
                        first_key[q_i] = p_from_q;
                        dtw[q_i] = dtw[q_i].saturating_add(1);
                        bit_set_true(resolved, q_i);
                        queue.push(QueueEntry {
                            key: QueueKey {
                                neg_abs_value: -(first_key[q_i].abs()),
                                dtw: dtw[q_i],
                            },
                            sub_idx: 0,
                            state_idx: q_idx,
                        });
                    }
                }
            }
            // p_from_q == 0 (DRAW): no count decrement. Parent stays
            // UNRESOLVED if it had only DRAW + WIN-direction children,
            // and falls through to DRAW at finalize. Matches Phase 1's
            // "DRAW children don't decrement count" invariant.
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adjust_zero_to_zero_is_identity_after_negation() {
        // Rank 0 -> rank 0: from absolute 0, negate -> 0, relative 0.
        let (k, flip) = adjust_first_key(0, 0, 0);
        assert_eq!(k, 0);
        assert!(!flip);
    }

    #[test]
    fn adjust_positive_to_better_rank() {
        // value_rel = 5 in rank-3 subspace -> absolute 8.
        // Move to rank-3 again -> negate -> -8, relative -11.
        let (k, _flip) = adjust_first_key(5, 3, 3);
        assert_eq!(k, -11);
    }

    #[test]
    fn propagate_dtw_increments_when_positive() {
        assert_eq!(propagate_dtw(4, 5), 5);
    }

    #[test]
    fn propagate_dtw_decrements_when_negative() {
        assert_eq!(propagate_dtw(4, -5), 3);
    }

    #[test]
    fn propagate_dtw_unchanged_at_zero_first_key() {
        assert_eq!(propagate_dtw(4, 0), 4);
    }

    #[test]
    fn queue_key_ordering_extreme_first_then_low_dtw() {
        use std::collections::BinaryHeap;
        // Higher |value| should pop first. |value|=10 < |value|=5? No:
        // neg_abs(value=10) = -10, neg_abs(value=5) = -5; min-heap pops -10 first.
        let a = QueueKey {
            neg_abs_value: -10,
            dtw: 0,
        };
        let b = QueueKey {
            neg_abs_value: -5,
            dtw: 0,
        };
        let mut heap = BinaryHeap::new();
        heap.push(a);
        heap.push(b);
        assert_eq!(heap.pop().unwrap(), a);
        assert_eq!(heap.pop().unwrap(), b);
    }

    #[test]
    fn work_area_allocates_per_primary() {
        use crate::work_unit::WorkUnit;
        let wu = WorkUnit::pair(Subspace::movement(4, 3), Subspace::movement(3, 4));
        let ranks = vec![-2, 2];
        let wa = GevayWorkArea::new(&wu, &ranks);
        assert_eq!(wa.cells.len(), 2);
        assert_eq!(wa.cells[0].len(), Subspace::movement(4, 3).n_states() as usize);
        assert_eq!(wa.cells[1].len(), Subspace::movement(3, 4).n_states() as usize);
    }

    #[test]
    fn forward_moves_yields_expected_count_in_33() {
        use crate::work_unit;
        use crate::wave::Variant;
        let _ = work_unit::list_movement_work_units(6); // sanity: compiles
        let sub = Subspace::movement(3, 3);
        let wbb = 0b111u32;
        let bbb = (0b111u32) << 16;
        let mut count = 0usize;
        forward_moves(sub, wbb, bbb, STM_WHITE, Variant::Flying, |_| count += 1);
        // 3 STM pieces × 18 empties = 54 raw destinations; each may yield
        // one capture move per legal target if it forms a mill. Just sanity
        // check we don't crash and we get a plausible number.
        assert!(count > 0, "no moves generated");
        assert!(count < 1000, "implausibly many moves: {}", count);
    }
}
