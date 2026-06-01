//! Subspace identification and position indexing.
//!
//! A subspace is the set of all legal positions sharing the same piece
//! count tuple `(w_board, b_board, w_to_place, b_to_place)`. Movement-phase
//! subspaces have `*_to_place == 0`; placement-phase subspaces have at
//! least one nonzero `*_to_place`.
//!
//! Position indexing within a subspace uses combinatorial unranking on
//! both bitmasks: `idx = rank_w * C(24-w, b) + rank_b_compact`. Times 2
//! for the STM axis: `state_idx = idx * 2 + (stm - 1)`.

use crate::hash::{compact_against, expand_against, rank_subset, unrank_subset, BINOM};
use crate::symmetry::canonicalize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Subspace {
    pub w_board: u8,
    pub b_board: u8,
    pub w_to_place: u8,
    pub b_to_place: u8,
}

impl Subspace {
    pub const fn movement(w: u8, b: u8) -> Self {
        Self { w_board: w, b_board: b, w_to_place: 0, b_to_place: 0 }
    }

    pub const fn is_movement(&self) -> bool {
        self.w_to_place == 0 && self.b_to_place == 0
    }

    pub fn n_positions(&self) -> u32 {
        let w = self.w_board as usize;
        let b = self.b_board as usize;
        BINOM[24][w] * BINOM[24 - w][b]
    }

    pub fn n_states(&self) -> u32 {
        self.n_positions() * 2
    }

    /// Index a `(wbb, bbb, stm)` state. Internally canonicalises the
    /// bitmask pair under D4, so any orbit member maps to the canonical
    /// representative's slot. Side-to-move is kept on the low bit.
    #[inline]
    pub fn state_index(&self, wbb: u32, bbb: u32, stm: u8) -> u32 {
        let (cw, cb) = canonicalize(wbb, bbb);
        self.state_index_canonical(cw, cb, stm)
    }

    /// Index assuming the caller already has the canonical bitmasks (skips
    /// the 8-way canonicalisation pass). Used in tight wave inner loops.
    #[inline]
    pub fn state_index_canonical(&self, cw: u32, cb: u32, stm: u8) -> u32 {
        let rank_w = rank_subset(cw);
        let compact_b = compact_against(cb, cw);
        let rank_b = rank_subset(compact_b);
        let n_b = BINOM[24 - self.w_board as usize][self.b_board as usize];
        let pos = rank_w * n_b + rank_b;
        pos * 2 + (stm - 1) as u32
    }

    /// Decode a slot index back to its canonical `(wbb, bbb, stm)`.
    /// The returned bitmasks are always the orbit's canonical representative.
    #[inline]
    pub fn decode_state(&self, state_idx: u32) -> (u32, u32, u8) {
        let stm = (state_idx & 1) as u8 + 1;
        let pos = state_idx >> 1;
        let n_b = BINOM[24 - self.w_board as usize][self.b_board as usize];
        let rank_w = pos / n_b;
        let rank_b = pos % n_b;
        let wbb = unrank_subset(rank_w, 24, self.w_board as u32);
        let compact_b = unrank_subset(rank_b, 24 - self.w_board as u32, self.b_board as u32);
        let bbb = expand_against(compact_b, wbb);
        (wbb, bbb, stm)
    }

    /// Iterate over CANONICAL `(wbb, bbb)` representatives in the subspace.
    /// Non-canonical raw positions are skipped, so the closure runs once per
    /// orbit (~8× fewer iterations than full raw enumeration under D4).
    #[inline]
    pub fn enumerate_positions<F: FnMut(u32, u32)>(&self, mut f: F) {
        let w = self.w_board as u32;
        let b = self.b_board as u32;
        let n_w = BINOM[24][w as usize];
        let n_b = BINOM[(24 - w) as usize][b as usize];
        for rank_w in 0..n_w {
            let wbb = unrank_subset(rank_w, 24, w);
            for rank_b in 0..n_b {
                let compact_b = unrank_subset(rank_b, 24 - w, b);
                let bbb = expand_against(compact_b, wbb);
                let (cw, cb) = canonicalize(wbb, bbb);
                if (cw, cb) == (wbb, bbb) {
                    f(wbb, bbb);
                }
            }
        }
    }

    /// Iterate over ALL raw `(wbb, bbb)` positions (not filtered to canonical).
    /// Used by validation paths that need raw counts via canonicalised lookup.
    #[inline]
    pub fn enumerate_positions_raw<F: FnMut(u32, u32)>(&self, mut f: F) {
        let w = self.w_board as u32;
        let b = self.b_board as u32;
        let n_w = BINOM[24][w as usize];
        let n_b = BINOM[(24 - w) as usize][b as usize];
        for rank_w in 0..n_w {
            let wbb = unrank_subset(rank_w, 24, w);
            for rank_b in 0..n_b {
                let compact_b = unrank_subset(rank_b, 24 - w, b);
                let bbb = expand_against(compact_b, wbb);
                f(wbb, bbb);
            }
        }
    }
}

/// In-memory store of resolved subspace tables. Used for cross-subspace
/// queries during the wave (capture transitions go to smaller subspaces).
#[derive(Default)]
pub struct Tablebase {
    tables: std::collections::HashMap<Subspace, SubspaceTable>,
}

pub struct SubspaceTable {
    pub subspace: Subspace,
    pub verdict: Vec<u8>,
    pub dtw: Vec<u16>,
}

impl SubspaceTable {
    /// Raw (orbit-weighted) verdict counts, comparable to Gasser's
    /// per-subspace published numbers. Iterates only canonical positions
    /// (~1/8 of raw) and sums orbit_size for each.
    pub fn raw_stats(&self) -> (u64, u64, u64, u16) {
        let mut win = 0u64;
        let mut loss = 0u64;
        let mut draw = 0u64;
        let mut max_dtw = 0u16;
        self.subspace.enumerate_positions(|wbb, bbb| {
            let osize = crate::symmetry::orbit_size(wbb, bbb) as u64;
            for stm in [1u8, 2u8] {
                let idx = self.subspace.state_index_canonical(wbb, bbb, stm) as usize;
                let v = self.verdict[idx];
                let d = self.dtw[idx];
                match v {
                    1 => { win += osize; if d > max_dtw { max_dtw = d; } }   // WIN
                    2 => { loss += osize; if d > max_dtw { max_dtw = d; } }  // LOSS
                    3 => draw += osize,                                       // DRAW
                    _ => {} // UNKNOWN slots are non-canonical; orbit_size won't be summed for them
                }
            }
        });
        (win, loss, draw, max_dtw)
    }
}

impl Tablebase {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, table: SubspaceTable) {
        self.tables.insert(table.subspace, table);
    }

    pub fn get(&self, sub: &Subspace) -> Option<&SubspaceTable> {
        self.tables.get(sub)
    }

    /// Look up a position in a previously resolved subspace.
    /// Returns `None` if the subspace hasn't been computed yet.
    pub fn query(&self, sub: Subspace, wbb: u32, bbb: u32, stm: u8) -> Option<(u8, u16)> {
        let table = self.tables.get(&sub)?;
        let idx = sub.state_index(wbb, bbb, stm) as usize;
        Some((table.verdict[idx], table.dtw[idx]))
    }
}
