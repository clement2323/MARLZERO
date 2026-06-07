//! Per-subspace lookup that maps a canonical `(cw, cb, stm)` to a dense
//! index in `[0, 2 × n_canonical_entries)`.
//!
//! Phase 1 stores W/D/L verdicts via [`Subspace::state_index_canonical`]
//! which spans the full `n_states = C(24,w) × C(24-w,b) × 2` range — 1/8
//! of those slots are D4-canonical orbit representatives, the other 7/8
//! are dead weight. Phase 1 can afford this because each slot is only 3
//! bytes (verdict u8 + dtw u16) and the dead 7/8 compress trivially when
//! stored.
//!
//! Phase 2's wave needs 5 mutable arrays per primary subspace (10 B/slot
//! before bit-packing, 6.25 B/slot after) and the same 1/8 ratio applies
//! — but here we cannot store dead slots, allocation must be canonical-only.
//! `CanonicalIndexer` provides the dense layout: `rank_w_offsets[k]` =
//! cumulative orbit count before rank_w = k, plus a per-bucket sorted
//! `Vec<u32>` of canonical rank_b_compact values so we can binary-search
//! for the position within a bucket.
//!
//! Cost vs `state_index_canonical`: O(log bucket_size) instead of O(1).
//! Bucket sizes max at `C(15, 9) = 5005` for (9,9) → ~12 comparisons. On
//! modern CPUs this is well under a microsecond; the wave is bounded by
//! algorithm complexity, not index lookup, so the slowdown is tolerable
//! in exchange for the ×8 RAM saving that makes the big subspaces fit.

use crate::hash::{compact_against, rank_subset, unrank_subset, expand_against, BINOM};
use crate::subspace::Subspace;
use crate::symmetry::canonicalize;

pub struct CanonicalIndexer {
    pub subspace: Subspace,
    w_count: u32,
    b_count: u32,
    /// `rank_w_offsets[k]` = number of canonical orbits in rank_w buckets
    /// `[0, k)`. Length `n_rank_w + 1`; the last entry equals total
    /// canonical orbits (= `n_canonical_entries`).
    rank_w_offsets: Vec<u32>,
    /// Per-rank_w bucket of canonical rank_b_compact values, sorted ascending.
    /// Lookup `pos_in_bucket = canonical_rank_b[rank_w].binary_search(&rank_b)`.
    /// For (9,9) this totals ~3 GB — the bulk of the indexer's memory.
    canonical_rank_b: Vec<Vec<u32>>,
    n_canonical_entries: u64,
}

impl CanonicalIndexer {
    pub fn build(sub: Subspace) -> Self {
        use rayon::prelude::*;
        let w_count = sub.w_board as u32;
        let b_count = sub.b_board as u32;
        let n_rank_w = BINOM[24][w_count as usize];
        let n_rank_b = BINOM[(24 - w_count) as usize][b_count as usize];

        // Per-rank_w buckets are independent (each tests its own n_rank_b
        // candidates for canonicality), so we parallelise over rank_w with
        // rayon. For (6,7): 134596 buckets × 31824 candidates = 4.28 billion
        // ops single-threaded ≈ 5 minutes; with rayon over 30+ threads it
        // drops to ~10 seconds, eliminating the silent stall the user
        // observed before the init pass started.
        let t_build = std::time::Instant::now();
        // Print only for the bigger subspaces — the tiny ones build in <50ms
        // and spamming a line per call inflates the WU log for no benefit.
        let verbose = n_rank_w * n_rank_b > 10_000_000;
        if verbose {
            eprintln!("    indexer: building ({} rank_w × {} rank_b candidates)",
                n_rank_w, n_rank_b);
        }
        let canonical_rank_b: Vec<Vec<u32>> = (0..n_rank_w)
            .into_par_iter()
            .map(|rank_w| {
                let wbb = unrank_subset(rank_w, 24, w_count);
                let mut bucket: Vec<u32> = Vec::new();
                for rank_b in 0..n_rank_b {
                    let compact_b = unrank_subset(rank_b, 24 - w_count, b_count);
                    let bbb = expand_against(compact_b, wbb);
                    let (cw, cb) = canonicalize(wbb, bbb);
                    if (cw, cb) == (wbb, bbb) {
                        bucket.push(rank_b);
                    }
                }
                bucket
            })
            .collect();

        // Sequential cumulative-offset pass — only n_rank_w iterations, fast.
        let mut rank_w_offsets: Vec<u32> = Vec::with_capacity(n_rank_w as usize + 1);
        let mut cum: u32 = 0;
        for rank_w in 0..n_rank_w {
            rank_w_offsets.push(cum);
            cum += canonical_rank_b[rank_w as usize].len() as u32;
        }
        rank_w_offsets.push(cum);
        if verbose {
            eprintln!("    indexer: built in {:.1}s ({} canonical entries)",
                t_build.elapsed().as_secs_f64(), cum);
        }

        Self {
            subspace: sub,
            w_count,
            b_count,
            rank_w_offsets,
            canonical_rank_b,
            n_canonical_entries: cum as u64,
        }
    }

    /// Number of canonical state slots = `2 × n_canonical_entries` (WTM + BTM).
    /// Use this to size wave state arrays.
    #[inline]
    pub fn n_states_canonical(&self) -> u64 {
        2 * self.n_canonical_entries
    }

    /// Map a CANONICAL `(cw, cb, stm)` triple to its dense slot.
    /// Panics if (cw, cb) is not canonical — binary_search returns Err.
    #[inline]
    pub fn canonical_index(&self, cw: u32, cb: u32, stm: u8) -> u64 {
        let rank_w = rank_subset(cw);
        let compact_b = compact_against(cb, cw);
        let rank_b = rank_subset(compact_b);
        let bucket = &self.canonical_rank_b[rank_w as usize];
        let pos_in_bucket = match bucket.binary_search(&rank_b) {
            Ok(i) => i,
            Err(_) => panic!(
                "canonical_index: (cw={:#x}, cb={:#x}) is not a canonical orbit \
                 (rank_w={}, rank_b={}). Caller must canonicalize first.",
                cw, cb, rank_w, rank_b
            ),
        };
        let pos = self.rank_w_offsets[rank_w as usize] as u64 + pos_in_bucket as u64;
        pos * 2 + (stm - 1) as u64
    }

    /// Convenience: canonicalize first, then index. Use when caller has a
    /// raw `(wbb, bbb)` that might not yet be the orbit representative.
    #[inline]
    pub fn index(&self, wbb: u32, bbb: u32, stm: u8) -> u64 {
        let (cw, cb) = canonicalize(wbb, bbb);
        self.canonical_index(cw, cb, stm)
    }

    /// Reverse: given a canonical slot index, return its `(cw, cb, stm)`.
    /// O(log n_rank_w) via binary search on rank_w_offsets to recover the
    /// rank_w bucket, then direct indexing into `canonical_rank_b[rank_w]`
    /// to get rank_b, then unrank both halves. No reverse map stored — for
    /// (9,9) that would have cost ~6 GB extra RAM.
    #[inline]
    pub fn decode(&self, canonical_idx: u64) -> (u32, u32, u8) {
        let stm = (canonical_idx & 1) as u8 + 1;
        let pos = (canonical_idx >> 1) as u32;
        // Find rank_w bucket: the largest k with rank_w_offsets[k] <= pos.
        let rank_w_usize = match self.rank_w_offsets.binary_search(&pos) {
            // Exact hit: pos is the bucket's first entry.
            Ok(k) => {
                // Multiple consecutive buckets can share an offset if they
                // contain zero canonical entries. Skip past empties to land
                // on the actual bucket.
                let mut k = k;
                while k + 1 < self.rank_w_offsets.len() && self.rank_w_offsets[k + 1] == pos {
                    k += 1;
                }
                k
            }
            // No exact hit: insertion index k means rank_w_offsets[k-1] < pos < rank_w_offsets[k].
            Err(k) => k - 1,
        };
        let pos_in_bucket = (pos - self.rank_w_offsets[rank_w_usize]) as usize;
        let rank_b = self.canonical_rank_b[rank_w_usize][pos_in_bucket];
        let cw = unrank_subset(rank_w_usize as u32, 24, self.w_count);
        let compact_b = unrank_subset(rank_b, 24 - self.w_count, self.b_count);
        let cb = expand_against(compact_b, cw);
        (cw, cb, stm)
    }

    pub fn n_canonical_entries(&self) -> u64 {
        self.n_canonical_entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every canonical orbit produced by `enumerate_positions` must map
    /// back to its own slot via `canonical_index` and decode losslessly.
    #[test]
    fn canonical_indexer_roundtrip_33() {
        let sub = Subspace::movement(3, 3);
        let indexer = CanonicalIndexer::build(sub);
        let mut count = 0u64;
        sub.enumerate_positions(|cw, cb| {
            for stm in [1u8, 2u8] {
                let idx = indexer.canonical_index(cw, cb, stm);
                let (dw, db, dstm) = indexer.decode(idx);
                assert_eq!(dw, cw, "decode cw mismatch at idx {}", idx);
                assert_eq!(db, cb, "decode cb mismatch at idx {}", idx);
                assert_eq!(dstm, stm, "decode stm mismatch at idx {}", idx);
                count += 1;
            }
        });
        assert_eq!(count, indexer.n_states_canonical());
    }

    #[test]
    fn canonical_indexer_roundtrip_43() {
        let sub = Subspace::movement(4, 3);
        let indexer = CanonicalIndexer::build(sub);
        let mut count = 0u64;
        sub.enumerate_positions(|cw, cb| {
            for stm in [1u8, 2u8] {
                let idx = indexer.canonical_index(cw, cb, stm);
                let (dw, db, dstm) = indexer.decode(idx);
                assert_eq!((dw, db, dstm), (cw, cb, stm));
                count += 1;
            }
        });
        assert_eq!(count, indexer.n_states_canonical());
    }

    /// Non-canonical (cw, cb) must NOT round-trip — they should hit the panic.
    #[test]
    #[should_panic(expected = "is not a canonical orbit")]
    fn canonical_indexer_rejects_non_canonical() {
        // (3,3) — find a non-canonical (wbb, bbb).
        let sub = Subspace::movement(3, 3);
        let indexer = CanonicalIndexer::build(sub);
        // Pick any (wbb, bbb) such that canonicalize(wbb, bbb) != (wbb, bbb).
        // We rotate (0,1,2 / 7,15,23) by D4 to get a non-canonical.
        // Simpler: scan until we find one.
        let mut non_canonical: Option<(u32, u32)> = None;
        for wbb in 0..(1u32 << 24) {
            if wbb.count_ones() != 3 { continue; }
            for bbb_seed in 0..200u32 {
                let bbb = bbb_seed | ((bbb_seed << 1) & !wbb);
                let bbb = bbb & !wbb;
                if bbb.count_ones() != 3 { continue; }
                let (cw, cb) = canonicalize(wbb, bbb);
                if (cw, cb) != (wbb, bbb) {
                    non_canonical = Some((wbb, bbb));
                    break;
                }
            }
            if non_canonical.is_some() { break; }
        }
        let (wbb, bbb) = non_canonical.expect("must find a non-canonical (3,3) position");
        let _ = indexer.canonical_index(wbb, bbb, 1);
    }
}
