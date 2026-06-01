//! Retrograde wave on the (3,3,0,0) subspace with flying.
//!
//! This is the Rust port of [scripts/spike_gasser_33.py](../../../scripts/spike_gasser_33.py).
//! Single-threaded, no symmetry reduction. Used to validate the algorithm
//! end-to-end against the Python fixture before introducing canonicalisation
//! and parallelism.
//!
//! Both sides have 3 pieces so flying is active for both. At (3,3,0,0):
//!   - non-mill flying moves stay in (3,3,0,0)
//!   - mill flying moves capture -> opponent at 2 pieces -> terminal LOSS
//! So the subspace is self-contained and no cross-subspace lookups occur.

use crate::board::NUM_POSITIONS;
use crate::hash::{rank_subset, BINOM};
use crate::rules::is_mill_through;

pub const NUM_W: u32 = 3;
pub const NUM_B: u32 = 3;
pub const STM_WHITE: u8 = 1;
pub const STM_BLACK: u8 = 2;

pub const UNKNOWN: u8 = 0;
pub const WIN: u8 = 1;
pub const LOSS: u8 = 2;
pub const DRAW: u8 = 3;

/// `n_positions = C(24, 3) * C(21, 3) = 2_691_920`.
pub const N_POSITIONS: u32 = BINOM[24][3] * BINOM[21][3];
/// `n_states = n_positions * 2` (one per STM).
pub const N_STATES: u32 = N_POSITIONS * 2;

/// Aggregate results from a (3,3) wave run.
#[derive(Debug)]
pub struct Result33 {
    pub n_states: u32,
    pub win: u32,
    pub loss: u32,
    pub draw: u32,
    pub instant_win_dtw1: u32,
    pub max_dtw: u16,
    /// Invariant A check: swap(colors) + swap(stm) -> identical verdict.
    pub invariant_a_failures: u32,
}

/// Compute `state_idx ∈ [0, N_STATES)` for `(whites_bb, blacks_bb, stm)`.
#[inline]
fn state_index(wbb: u32, bbb: u32, stm: u8) -> usize {
    let rank_w = rank_subset(wbb);
    let compact_b = crate::hash::compact_against(bbb, wbb);
    let rank_b = rank_subset(compact_b);
    let pos = rank_w * BINOM[21][3] + rank_b;
    (pos as usize) * 2 + (stm - 1) as usize
}

/// True iff STM has at least one flying move that completes a mill.
#[inline]
fn has_mill_move(stm_bb: u32, opp_bb: u32) -> bool {
    let occupied = stm_bb | opp_bb;
    let mut s = stm_bb;
    while s != 0 {
        let src = s.trailing_zeros() as u8;
        s &= s - 1;
        let after_lift = stm_bb & !(1u32 << src);
        // empties = positions not in occupied (within 0..24)
        let mut empties = !occupied & ((1u32 << NUM_POSITIONS) - 1);
        while empties != 0 {
            let dst = empties.trailing_zeros() as u8;
            empties &= empties - 1;
            let new_stm = after_lift | (1u32 << dst);
            if is_mill_through(new_stm, dst) {
                return true;
            }
        }
    }
    false
}

/// Run the retrograde wave on (3,3,0,0).
pub fn run() -> Result33 {
    let n = N_STATES as usize;
    let mut verdict: Vec<u8> = vec![UNKNOWN; n];
    let mut dtw: Vec<u16> = vec![0u16; n];
    let mut count: Vec<u8> = vec![0u8; n];
    let full_count: u8 = (NUM_W as u8) * (NUM_POSITIONS as u8 - 2 * NUM_W as u8); // 3 × 18 = 54
    let mut queue: Vec<usize> = Vec::with_capacity(n / 4);

    // Phase 0 — enumerate positions in lex order, mark instant-WIN, init counts.
    enumerate_positions(|wbb, bbb| {
        for &stm in &[STM_WHITE, STM_BLACK] {
            let (stm_bb, opp_bb) = if stm == STM_WHITE { (wbb, bbb) } else { (bbb, wbb) };
            let idx = state_index(wbb, bbb, stm);
            if has_mill_move(stm_bb, opp_bb) {
                verdict[idx] = WIN;
                dtw[idx] = 1;
                queue.push(idx);
            } else {
                count[idx] = full_count;
            }
        }
    });
    let instant_win_dtw1 = queue.len() as u32;

    // Phase 1 — wave propagation, generating parents on the fly.
    let mut head = 0usize;
    while head < queue.len() {
        let p_idx = queue[head];
        head += 1;
        let p_v = verdict[p_idx];
        let p_dtw = dtw[p_idx];
        let (wbb, bbb, stm_p) = decode_state(p_idx);
        propagate(
            wbb, bbb, stm_p, p_v, p_dtw,
            &mut verdict, &mut dtw, &mut count, &mut queue,
        );
    }

    // Phase 2 — relabel UNKNOWN as DRAW; tally.
    let mut win = 0u32;
    let mut loss = 0u32;
    let mut draw = 0u32;
    let mut max_dtw = 0u16;
    for i in 0..n {
        if verdict[i] == UNKNOWN {
            verdict[i] = DRAW;
        }
        match verdict[i] {
            WIN => {
                win += 1;
                if dtw[i] > max_dtw { max_dtw = dtw[i]; }
            }
            LOSS => {
                loss += 1;
                if dtw[i] > max_dtw { max_dtw = dtw[i]; }
            }
            DRAW => draw += 1,
            _ => unreachable!(),
        }
    }

    // Invariant A: swap(colors) + swap(stm) -> identical verdict.
    let mut inv_a = 0u32;
    enumerate_positions(|wbb, bbb| {
        for &stm in &[STM_WHITE, STM_BLACK] {
            let i = state_index(wbb, bbb, stm);
            let j = state_index(bbb, wbb, 3 - stm);
            if verdict[i] != verdict[j] {
                inv_a += 1;
            }
        }
    });

    Result33 {
        n_states: n as u32,
        win,
        loss,
        draw,
        instant_win_dtw1,
        max_dtw,
        invariant_a_failures: inv_a,
    }
}

/// Iterate over all `(wbb, bbb)` in (3,3) in lex order. ~2.7M iterations.
#[inline]
fn enumerate_positions<F: FnMut(u32, u32)>(mut f: F) {
    for w0 in 0..NUM_POSITIONS as u8 {
        for w1 in (w0 + 1)..NUM_POSITIONS as u8 {
            for w2 in (w1 + 1)..NUM_POSITIONS as u8 {
                let wbb = (1u32 << w0) | (1u32 << w1) | (1u32 << w2);
                for b0 in 0..NUM_POSITIONS as u8 {
                    if (wbb >> b0) & 1 != 0 { continue; }
                    for b1 in (b0 + 1)..NUM_POSITIONS as u8 {
                        if (wbb >> b1) & 1 != 0 { continue; }
                        for b2 in (b1 + 1)..NUM_POSITIONS as u8 {
                            if (wbb >> b2) & 1 != 0 { continue; }
                            let bbb = (1u32 << b0) | (1u32 << b1) | (1u32 << b2);
                            f(wbb, bbb);
                        }
                    }
                }
            }
        }
    }
}

/// Recover `(wbb, bbb, stm)` from a state index. Inverse of `state_index`.
fn decode_state(idx: usize) -> (u32, u32, u8) {
    let stm = (idx & 1) as u8 + 1;
    let pos = (idx >> 1) as u32;
    let n_b_slots = BINOM[21][3];
    let rank_w = pos / n_b_slots;
    let rank_b = pos % n_b_slots;
    let wbb = crate::hash::unrank_subset(rank_w, 24, 3);
    let compact_b = crate::hash::unrank_subset(rank_b, 21, 3);
    let bbb = crate::hash::expand_against(compact_b, wbb);
    (wbb, bbb, stm)
}

/// Generate parent states of `p = (wbb, bbb, stm_p)` and update them per
/// the wave rules. `p_v` is `p`'s verdict (WIN or LOSS), `p_dtw` its DTW.
#[inline]
fn propagate(
    wbb: u32, bbb: u32, stm_p: u8, p_v: u8, p_dtw: u16,
    verdict: &mut [u8], dtw: &mut [u16], count: &mut [u8], queue: &mut Vec<usize>,
) {
    let mover_stm = 3 - stm_p;
    let (mover_bb, fixed_bb) = if mover_stm == STM_WHITE { (wbb, bbb) } else { (bbb, wbb) };
    let occupied = wbb | bbb;
    let empties_mask = !occupied & ((1u32 << NUM_POSITIONS) - 1);

    let mut mb = mover_bb;
    while mb != 0 {
        let dst = mb.trailing_zeros() as u8;
        mb &= mb - 1;
        // Skip if the (src -> dst) move at the parent would have completed a
        // mill: that move would have captured, sending us to (4,3) / (3,4)
        // which isn't part of the (3,3) wave.
        if is_mill_through(mover_bb, dst) {
            continue;
        }
        let mut em = empties_mask;
        while em != 0 {
            let src = em.trailing_zeros() as u8;
            em &= em - 1;
            let new_mover = (mover_bb & !(1u32 << dst)) | (1u32 << src);
            let (new_wbb, new_bbb) = if mover_stm == STM_WHITE {
                (new_mover, fixed_bb)
            } else {
                (fixed_bb, new_mover)
            };
            let q_idx = state_index(new_wbb, new_bbb, mover_stm);
            if verdict[q_idx] != UNKNOWN {
                continue;
            }
            if p_v == LOSS {
                verdict[q_idx] = WIN;
                dtw[q_idx] = p_dtw + 1;
                queue.push(q_idx);
            } else {
                // p_v == WIN: parent's child resolved as opponent-win, decrement.
                count[q_idx] -= 1;
                if count[q_idx] == 0 {
                    verdict[q_idx] = LOSS;
                    // For LOSS DTW = max(children DTW) + 1; we re-enumerate
                    // children to get the max. Fast at (3,3): 54 ops.
                    dtw[q_idx] = max_child_dtw(new_wbb, new_bbb, mover_stm, verdict, dtw) + 1;
                    queue.push(q_idx);
                }
            }
        }
    }
}

/// Compute `max DTW` over all (3,3) children of `q` (non-mill moves only).
fn max_child_dtw(wbb: u32, bbb: u32, stm: u8, verdict: &[u8], dtw: &[u16]) -> u16 {
    let (stm_bb, opp_bb) = if stm == STM_WHITE { (wbb, bbb) } else { (bbb, wbb) };
    let occupied = wbb | bbb;
    let empties_mask = !occupied & ((1u32 << NUM_POSITIONS) - 1);
    let mut max = 0u16;
    let mut s = stm_bb;
    while s != 0 {
        let src = s.trailing_zeros() as u8;
        s &= s - 1;
        let after_lift = stm_bb & !(1u32 << src);
        let mut em = empties_mask;
        while em != 0 {
            let dst = em.trailing_zeros() as u8;
            em &= em - 1;
            let new_stm = after_lift | (1u32 << dst);
            if is_mill_through(new_stm, dst) {
                continue;
            }
            let (new_wbb, new_bbb) = if stm == STM_WHITE {
                (new_stm, opp_bb)
            } else {
                (opp_bb, new_stm)
            };
            let c_idx = state_index(new_wbb, new_bbb, 3 - stm);
            if verdict[c_idx] != UNKNOWN {
                let d = dtw[c_idx];
                if d > max { max = d; }
            }
        }
    }
    max
}
