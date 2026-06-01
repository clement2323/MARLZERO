//! Generic retrograde wave on any movement subspace.
//!
//! Parameterised by [Subspace] and a variant flag (with/without flying).
//! Handles cross-subspace child lookups through a [Tablebase]: captures
//! that take the opponent below the current subspace's piece count are
//! resolved at init time using the already-computed smaller subspace.
//!
//! The wave loop itself is the same algorithm as the Python spike on
//! (3,3,0,0): bitmask state, BFS queue, count-field, LOSS DTW = max child
//! DTW + 1.

use std::collections::HashSet;

use crate::board::{ADJACENCY, NUM_POSITIONS};
use crate::rules::{is_mill_through, legal_capture_targets, popcount};
use crate::subspace::{Subspace, SubspaceTable, Tablebase};

pub const STM_WHITE: u8 = 1;
pub const STM_BLACK: u8 = 2;

pub const UNKNOWN: u8 = 0;
pub const WIN: u8 = 1;
pub const LOSS: u8 = 2;
pub const DRAW: u8 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    Flying,
    NoFlying,
}

/// Aggregate orbit-weighted RAW counts after a wave run on one subspace.
/// These match Gasser 1996's per-subspace published numbers and the
/// Python fixture: every raw position contributes 1 to exactly one of
/// `win` / `loss` / `draw` (via its orbit canonical's verdict).
#[derive(Debug, Default)]
pub struct WaveStats {
    pub n_states: u32,
    pub win: u64,
    pub loss: u64,
    pub draw: u64,
    pub max_dtw: u16,
}

/// Compute the adjacency bitmask of a board position.
#[inline]
fn adjacency_mask(pos: u8) -> u32 {
    let adj = ADJACENCY[pos as usize];
    let mut m = 0u32;
    for &p in &adj {
        if p == 0xFF { break; }
        m |= 1u32 << p;
    }
    m
}

/// Iterate `(dst, forms_mill, new_stm_bb)` over all legal moves of `stm_bb`
/// to an empty square. For mill-forming destinations the caller still needs
/// to enumerate capture targets via [legal_capture_targets].
#[inline]
fn for_each_simple_move<F: FnMut(u8, bool, u32, u8)>(
    stm_bb: u32, opp_bb: u32, can_fly: bool, mut f: F,
) {
    let occupied = stm_bb | opp_bb;
    let empties = !occupied & ((1u32 << NUM_POSITIONS) - 1);
    let mut s = stm_bb;
    while s != 0 {
        let src = s.trailing_zeros() as u8;
        s &= s - 1;
        let after_lift = stm_bb & !(1u32 << src);
        let dests = if can_fly { empties } else { adjacency_mask(src) & empties };
        let mut d = dests;
        while d != 0 {
            let dst = d.trailing_zeros() as u8;
            d &= d - 1;
            let new_stm = after_lift | (1u32 << dst);
            let forms_mill = is_mill_through(new_stm, dst);
            f(dst, forms_mill, new_stm, src);
        }
    }
}

/// Solve one movement subspace. The tablebase must contain all SMALLER
/// movement subspaces (i.e. all `(w', b')` with `w' + b' < w + b`).
/// Returns the resolved [SubspaceTable] and aggregate stats.
///
/// If `progress` is provided, the bar is set to length = n_states and
/// updated as positions get resolved across init + wave + finalize phases.
pub fn solve_movement(
    sub: Subspace,
    variant: Variant,
    tablebase: &Tablebase,
    progress: Option<&indicatif::ProgressBar>,
) -> (SubspaceTable, WaveStats) {
    assert!(sub.is_movement(), "solve_movement called on placement subspace {:?}", sub);
    let n = sub.n_states() as usize;
    let mut verdict: Vec<u8> = vec![UNKNOWN; n];
    let mut dtw: Vec<u16> = vec![0u16; n];
    let mut count: Vec<u16> = vec![0u16; n];
    let mut queue: Vec<u32> = Vec::with_capacity(n / 4);

    if let Some(pb) = progress {
        pb.set_length(n as u64);
        pb.set_position(0);
        pb.set_message("init");
    }

    // Phase 0 — enumerate positions, compute initial verdict using cross-
    // subspace lookups for capture children that go to smaller subspaces.
    sub.enumerate_positions(|wbb, bbb| {
        for &stm in &[STM_WHITE, STM_BLACK] {
            let idx = sub.state_index(wbb, bbb, stm);
            init_position(sub, variant, tablebase, wbb, bbb, stm,
                          &mut verdict, &mut dtw, &mut count, &mut queue, idx);
        }
    });

    // Count states already resolved at init (queued + init-resolved DRAWs).
    let mut resolved: u64 = 0;
    for &v in &verdict {
        if v != UNKNOWN { resolved += 1; }
    }
    if let Some(pb) = progress {
        pb.set_position(resolved);
        pb.set_message("wave");
    }

    // Phase 1 — wave propagation through intra-subspace parents.
    let mut head = 0usize;
    let mut last_tick = resolved;
    while head < queue.len() {
        let p_idx = queue[head];
        head += 1;
        let prev_len = queue.len();
        propagate_to_parents(sub, variant, p_idx,
                             &mut verdict, &mut dtw, &mut count, &mut queue);
        let new_pushes = (queue.len() - prev_len) as u64;
        resolved += new_pushes;
        if let Some(pb) = progress {
            if resolved - last_tick >= 100_000 {
                pb.set_position(resolved);
                last_tick = resolved;
            }
        }
    }

    if let Some(pb) = progress {
        pb.set_position(resolved);
        pb.set_message("finalize");
    }

    // Phase 2 — UNKNOWN → DRAW on canonical slots only (non-canonical raw
    // slots stay as 0/UNKNOWN; they are never queried directly because
    // state_index always canonicalises first). Tally is orbit-weighted to
    // give RAW counts that match Gasser 1996's published per-subspace
    // numbers and the Python fixture for (3,3,0,0).
    let mut stats = WaveStats { n_states: sub.n_positions() * 2, ..Default::default() };
    sub.enumerate_positions(|wbb, bbb| {
        let osize = crate::symmetry::orbit_size(wbb, bbb) as u64;
        for &stm in &[STM_WHITE, STM_BLACK] {
            let idx = sub.state_index_canonical(wbb, bbb, stm) as usize;
            if verdict[idx] == UNKNOWN { verdict[idx] = DRAW; }
            match verdict[idx] {
                WIN => { stats.win += osize; if dtw[idx] > stats.max_dtw { stats.max_dtw = dtw[idx]; } }
                LOSS => { stats.loss += osize; if dtw[idx] > stats.max_dtw { stats.max_dtw = dtw[idx]; } }
                DRAW => stats.draw += osize,
                _ => unreachable!(),
            }
        }
    });

    if let Some(pb) = progress {
        pb.set_position(n as u64);
        pb.finish_with_message("done");
    }

    let table = SubspaceTable { subspace: sub, verdict, dtw };
    (table, stats)
}

/// High bit of `count[idx]` signals that the position has at least one
/// cross-subspace DRAW child, so when `(count & 0x7FFF)` reaches 0 in the
/// wave the position is DRAW rather than LOSS.
const HAS_DRAW_FLAG: u16 = 0x8000;
const COUNT_MASK: u16 = 0x7FFF;

/// Compute initial verdict / count for one position. Cross-subspace capture
/// children are resolved via [Tablebase]; same-subspace non-capture children
/// are simply counted (never pre-resolved, even if their position index
/// happens to be lex-smaller — the wave will propagate them later).
#[inline]
fn init_position(
    sub: Subspace, variant: Variant, tb: &Tablebase,
    wbb: u32, bbb: u32, stm: u8,
    verdict: &mut [u8], dtw: &mut [u16], count: &mut [u16], queue: &mut Vec<u32>,
    idx: u32,
) {
    let (stm_bb, opp_bb) = if stm == STM_WHITE { (wbb, bbb) } else { (bbb, wbb) };
    let stm_count = popcount(stm_bb);
    let can_fly = variant == Variant::Flying && stm_count == 3;

    let mut win_dtw: Option<u16> = None;
    let mut max_lose_dtw_cross: u16 = 0;
    let mut has_draw_child: bool = false;
    let mut any_move: bool = false;

    // Intra-subspace children are deduplicated by their canonical state index:
    // multiple raw (src, dst) moves can collapse to the same orbit child, and
    // count(p) must reflect orbit-distinct children rather than raw moves.
    let mut intra_children: HashSet<u32> = HashSet::new();

    for_each_simple_move(stm_bb, opp_bb, can_fly, |_dst, forms_mill, new_stm, _src| {
        any_move = true;
        if forms_mill {
            let cap_targets = legal_capture_targets(opp_bb);
            let mut c = cap_targets;
            while c != 0 {
                let cap = c.trailing_zeros();
                c &= c - 1;
                let new_opp = opp_bb & !(1u32 << cap);
                let opp_new_count = popcount(new_opp);
                if opp_new_count < 3 {
                    win_dtw = Some(win_dtw.map_or(1, |d| d.min(1)));
                } else {
                    let target_sub = subspace_after_capture(sub, stm, opp_new_count as u8);
                    let (new_wbb, new_bbb) = if stm == STM_WHITE { (new_stm, new_opp) } else { (new_opp, new_stm) };
                    let (v, d) = tb.query(target_sub, new_wbb, new_bbb, 3 - stm)
                        .expect("smaller subspace must be resolved");
                    classify(v, d, &mut win_dtw, &mut max_lose_dtw_cross, &mut has_draw_child);
                }
            }
        } else {
            // Intra-subspace child — dedup by canonical index.
            let (new_wbb, new_bbb) = if stm == STM_WHITE { (new_stm, opp_bb) } else { (opp_bb, new_stm) };
            let child_idx = sub.state_index(new_wbb, new_bbb, 3 - stm);
            intra_children.insert(child_idx);
        }
    });
    let n_intra = intra_children.len() as u16;

    // Stalemate: no legal moves at all → LOSS for STM at DTW=0.
    if !any_move {
        verdict[idx as usize] = LOSS;
        dtw[idx as usize] = 0;
        queue.push(idx);
        return;
    }

    if let Some(d) = win_dtw {
        verdict[idx as usize] = WIN;
        dtw[idx as usize] = d;
        queue.push(idx);
    } else if n_intra == 0 {
        // All moves are cross-subspace; no further wave activity for this position.
        if has_draw_child {
            verdict[idx as usize] = DRAW;
        } else {
            verdict[idx as usize] = LOSS;
            dtw[idx as usize] = max_lose_dtw_cross + 1;
            queue.push(idx);
        }
    } else {
        // n_intra > 0 → defer to wave. Stash max_lose_dtw_cross in dtw[]
        // so the wave can pick it up when count reaches 0 (the slot is
        // unused for UNKNOWN states; we overwrite it on LOSS transition).
        count[idx as usize] = n_intra | if has_draw_child { HAS_DRAW_FLAG } else { 0 };
        dtw[idx as usize] = max_lose_dtw_cross;
    }
}

/// Update win_dtw / max_lose_dtw / has_draw_child from one resolved child.
#[inline]
fn classify(v: u8, d: u16, win_dtw: &mut Option<u16>, max_lose: &mut u16, has_draw: &mut bool) {
    match v {
        LOSS => {
            // Child loses for opp = winning move for p.
            let candidate = d.saturating_add(1);
            *win_dtw = Some(win_dtw.map_or(candidate, |w| w.min(candidate)));
        }
        WIN => {
            // Child wins for opp = losing move for p.
            if d > *max_lose { *max_lose = d; }
        }
        DRAW => { *has_draw = true; }
        _ => unreachable!(),
    }
}

/// Compute the target subspace after a capture moves opponent to `opp_new_count` pieces.
#[inline]
fn subspace_after_capture(sub: Subspace, stm: u8, opp_new_count: u8) -> Subspace {
    if stm == STM_WHITE {
        Subspace::movement(sub.w_board, opp_new_count)
    } else {
        Subspace::movement(opp_new_count, sub.b_board)
    }
}

/// Propagate one resolved state to its intra-subspace parents.
///
/// Each raw inverse move yields a (possibly non-canonical) parent state.
/// Multiple raw inverses can collapse to the same canonical orbit, so we
/// deduplicate by canonical state index before applying any update —
/// otherwise a single resolved child would decrement the same parent's
/// count multiple times.
fn propagate_to_parents(
    sub: Subspace, variant: Variant, p_idx: u32,
    verdict: &mut [u8], dtw: &mut [u16], count: &mut [u16], queue: &mut Vec<u32>,
) {
    let (wbb, bbb, stm_p) = sub.decode_state(p_idx);
    let p_v = verdict[p_idx as usize];
    let p_dtw = dtw[p_idx as usize];
    let mover_stm = 3 - stm_p;
    let (mover_bb, fixed_bb) = if mover_stm == STM_WHITE { (wbb, bbb) } else { (bbb, wbb) };
    let mover_count = popcount(mover_bb);
    let mover_can_fly = variant == Variant::Flying && mover_count == 3;

    let occupied = wbb | bbb;
    let empties = !occupied & ((1u32 << NUM_POSITIONS) - 1);

    let mut seen_parents: HashSet<u32> = HashSet::new();
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
            let (new_wbb, new_bbb) = if mover_stm == STM_WHITE { (new_mover, fixed_bb) } else { (fixed_bb, new_mover) };
            let q_idx = sub.state_index(new_wbb, new_bbb, mover_stm);
            if !seen_parents.insert(q_idx) { continue; }
            let q_i = q_idx as usize;
            if verdict[q_i] != UNKNOWN { continue; }
            if p_v == LOSS {
                verdict[q_i] = WIN;
                dtw[q_i] = p_dtw + 1;
                queue.push(q_idx);
            } else if p_v == WIN {
                let new_count_field = count[q_i] - 1;
                count[q_i] = new_count_field;
                if (new_count_field & COUNT_MASK) == 0 {
                    let had_draw = (new_count_field & HAS_DRAW_FLAG) != 0;
                    if had_draw {
                        verdict[q_i] = DRAW;
                    } else {
                        let (qw, qb, _) = sub.decode_state(q_idx);
                        let intra_max = max_win_child_dtw(sub, variant, qw, qb, mover_stm, verdict, dtw);
                        let combined = intra_max.max(dtw[q_i]);
                        verdict[q_i] = LOSS;
                        dtw[q_i] = combined + 1;
                        queue.push(q_idx);
                    }
                }
            }
        }
    }
}

/// Maximum DTW over all intra-subspace WIN children of `q`. Cross-subspace
/// WIN DTWs are already baked into `dtw[q_idx]` at init time (stashed there
/// by [init_position] before this position becomes UNKNOWN), so the LOSS
/// transition combines `intra_max` with that stashed value.
///
/// Children are deduplicated by canonical state index — same rationale as
/// parent dedup in [propagate_to_parents].
fn max_win_child_dtw(
    sub: Subspace, variant: Variant, wbb: u32, bbb: u32, stm: u8,
    verdict: &[u8], dtw: &[u16],
) -> u16 {
    let (stm_bb, opp_bb) = if stm == STM_WHITE { (wbb, bbb) } else { (bbb, wbb) };
    let stm_count = popcount(stm_bb);
    let can_fly = variant == Variant::Flying && stm_count == 3;
    let mut max = 0u16;
    let mut seen: HashSet<u32> = HashSet::new();

    for_each_simple_move(stm_bb, opp_bb, can_fly, |_dst, forms_mill, new_stm, _src| {
        if forms_mill {
            // Mill move => capture => cross-subspace or terminal child.
            // Cross-subspace WIN DTWs were folded into dtw[q_idx] at init time.
            let _ = new_stm;
        } else {
            let (new_wbb, new_bbb) = if stm == STM_WHITE { (new_stm, opp_bb) } else { (opp_bb, new_stm) };
            let c_idx = sub.state_index(new_wbb, new_bbb, 3 - stm);
            if !seen.insert(c_idx) { return; }
            if verdict[c_idx as usize] == WIN {
                let d = dtw[c_idx as usize];
                if d > max { max = d; }
            }
        }
    });
    max
}
