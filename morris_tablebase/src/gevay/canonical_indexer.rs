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
//! cumulative orbit count before rank_w = k, plus a flat sorted
//! `Vec<u32>` of canonical rank_b_compact values so we can binary-search
//! for the position within a bucket.
//!
//! Cost vs `state_index_canonical`: O(log bucket_size) instead of O(1).
//! Bucket sizes max at `C(15, 9) = 5005` for (9,9) → ~12 comparisons. On
//! modern CPUs this is well under a microsecond; the wave is bounded by
//! algorithm complexity, not index lookup, so the slowdown is tolerable
//! in exchange for the ×8 RAM saving that makes the big subspaces fit.
//!
//! ## Storage backends
//!
//! Two backends are exposed via [`IndexerStorage`]:
//!
//! * **Owned** — `Vec<u32>` allocations, the only mode used during
//!   `compute_gevay`'s wave (mutable allocator-friendly layout).
//! * **Mmap** — the same flat arrays backed by a memory-mapped file.
//!   Multiple processes (e.g. several `play_tb --serve` workers spawned
//!   by the self-play loop) mmap the same indexer cache file and the
//!   kernel page cache shares the resident pages across them. Without
//!   this, 8 workers each rebuilding their own (8,8) indexer would burn
//!   ~40 GB of RAM and OOM-kill the box.
//!
//! Use [`CanonicalIndexer::open_or_build`] from `play_tb` and any other
//! long-lived multi-process reader. `compute_gevay` calls
//! [`CanonicalIndexer::build`] directly because each WU only needs one
//! indexer in process memory and we don't want to pay the disk write.

use std::path::Path;

use crate::hash::{compact_against, rank_subset, unrank_subset, expand_against, BINOM};
use crate::subspace::Subspace;
use crate::symmetry::canonicalize;

const INDEXER_MAGIC: [u8; 4] = *b"MIDX";
const INDEXER_VERSION: u16 = 1;
const INDEXER_HEADER_SIZE: usize = 32;

/// Backing storage for the two big arrays — either owned (built in
/// process memory) or memory-mapped from disk. The on-disk layout matches
/// the in-memory layout exactly (little-endian `u32` arrays, x86_64-LE)
/// so the mmap variant is zero-copy.
pub enum IndexerStorage {
    Owned {
        rank_w_offsets: Vec<u32>,
        flat_rank_b: Vec<u32>,
    },
    Mmap {
        _mmap: memmap2::Mmap,
        rank_w_offsets_ptr: *const u32,
        n_rank_w_plus_one: usize,
        flat_rank_b_ptr: *const u32,
        flat_len: usize,
    },
}

// SAFETY: the raw pointers point inside the owned mmap; no interior
// mutability; queries are read-only.
unsafe impl Send for IndexerStorage {}
unsafe impl Sync for IndexerStorage {}

impl IndexerStorage {
    #[inline]
    fn rank_w_offsets(&self) -> &[u32] {
        match self {
            IndexerStorage::Owned { rank_w_offsets, .. } => rank_w_offsets.as_slice(),
            IndexerStorage::Mmap { rank_w_offsets_ptr, n_rank_w_plus_one, .. } => unsafe {
                std::slice::from_raw_parts(*rank_w_offsets_ptr, *n_rank_w_plus_one)
            },
        }
    }

    #[inline]
    fn flat_rank_b(&self) -> &[u32] {
        match self {
            IndexerStorage::Owned { flat_rank_b, .. } => flat_rank_b.as_slice(),
            IndexerStorage::Mmap { flat_rank_b_ptr, flat_len, .. } => unsafe {
                std::slice::from_raw_parts(*flat_rank_b_ptr, *flat_len)
            },
        }
    }
}

pub struct CanonicalIndexer {
    pub subspace: Subspace,
    w_count: u32,
    b_count: u32,
    storage: IndexerStorage,
    n_canonical_entries: u64,
}

/// Canonical filename for an on-disk indexer cache, used by
/// [`CanonicalIndexer::open_or_build`].
pub fn indexer_filename(sub: Subspace) -> String {
    format!("idx_w{}_b{}.bin", sub.w_board, sub.b_board)
}

impl CanonicalIndexer {
    /// Build the indexer in process memory from scratch. Used by
    /// `compute_gevay` (one WU at a time, no cache benefit).
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
        let per_bucket: Vec<Vec<u32>> = (0..n_rank_w)
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

        // Sequential cumulative-offset + flatten pass.
        let mut rank_w_offsets: Vec<u32> = Vec::with_capacity(n_rank_w as usize + 1);
        let total: u64 = per_bucket.iter().map(|v| v.len() as u64).sum();
        let mut flat_rank_b: Vec<u32> = Vec::with_capacity(total as usize);
        let mut cum: u32 = 0;
        for rank_w in 0..n_rank_w {
            rank_w_offsets.push(cum);
            let bucket = &per_bucket[rank_w as usize];
            flat_rank_b.extend_from_slice(bucket);
            cum += bucket.len() as u32;
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
            storage: IndexerStorage::Owned { rank_w_offsets, flat_rank_b },
            n_canonical_entries: cum as u64,
        }
    }

    /// Load an indexer cache from disk via mmap (zero-copy, shares pages
    /// across processes via the OS page cache). Returns an `io::Error` if
    /// the file doesn't exist, has the wrong magic/version, or doesn't
    /// match the requested subspace.
    pub fn load_mmap(sub: Subspace, path: &Path) -> std::io::Result<Self> {
        use std::io::{Error, ErrorKind};
        let f = std::fs::File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&f)? };
        if mmap.len() < INDEXER_HEADER_SIZE {
            return Err(Error::new(ErrorKind::InvalidData,
                format!("indexer file {} too small for header", path.display())));
        }
        // Header layout (32 bytes):
        //   0..4  magic "MIDX"
        //   4..6  version u16 LE
        //   6     w_board
        //   7     b_board
        //   8..12 n_rank_w u32 LE
        //   12..20 n_canonical_entries u64 LE
        //   20..32 reserved
        let magic = &mmap[0..4];
        if magic != INDEXER_MAGIC {
            return Err(Error::new(ErrorKind::InvalidData,
                format!("bad magic at {}: expected MIDX, got {:?}", path.display(), magic)));
        }
        let version = u16::from_le_bytes([mmap[4], mmap[5]]);
        if version != INDEXER_VERSION {
            return Err(Error::new(ErrorKind::InvalidData,
                format!("indexer version mismatch: file={}, want={}", version, INDEXER_VERSION)));
        }
        let w = mmap[6];
        let b = mmap[7];
        if w != sub.w_board || b != sub.b_board {
            return Err(Error::new(ErrorKind::InvalidData,
                format!("indexer subspace mismatch: file=({},{}), want=({},{})",
                    w, b, sub.w_board, sub.b_board)));
        }
        let n_rank_w = u32::from_le_bytes([mmap[8], mmap[9], mmap[10], mmap[11]]) as usize;
        let n_canonical_entries = u64::from_le_bytes([
            mmap[12], mmap[13], mmap[14], mmap[15],
            mmap[16], mmap[17], mmap[18], mmap[19],
        ]);

        // Body: rank_w_offsets (n_rank_w + 1 u32) || flat_rank_b (n_canonical_entries u32).
        // Both are 4-byte aligned starting at offset 32 (32 % 4 == 0).
        let offsets_bytes = (n_rank_w + 1) * 4;
        let flat_bytes = n_canonical_entries as usize * 4;
        let expected_total = INDEXER_HEADER_SIZE + offsets_bytes + flat_bytes;
        if mmap.len() < expected_total {
            return Err(Error::new(ErrorKind::InvalidData,
                format!("indexer file truncated: have {}, need {}", mmap.len(), expected_total)));
        }
        let base = mmap.as_ptr();
        // SAFETY: alignment of u32 = 4; offset 32 is 4-byte aligned; the
        // pointers stay inside the mapped region (size checked above).
        let rank_w_offsets_ptr = unsafe { base.add(INDEXER_HEADER_SIZE) as *const u32 };
        let flat_rank_b_ptr = unsafe { base.add(INDEXER_HEADER_SIZE + offsets_bytes) as *const u32 };

        Ok(Self {
            subspace: sub,
            w_count: w as u32,
            b_count: b as u32,
            storage: IndexerStorage::Mmap {
                _mmap: mmap,
                rank_w_offsets_ptr,
                n_rank_w_plus_one: n_rank_w + 1,
                flat_rank_b_ptr,
                flat_len: n_canonical_entries as usize,
            },
            n_canonical_entries,
        })
    }

    /// Serialise to disk in the on-disk format `load_mmap` expects.
    /// Atomic write via `path.tmp` → rename. Only callable on the Owned
    /// variant — there's no point re-saving an mmap-backed indexer.
    pub fn save_to_path(&self, path: &Path) -> std::io::Result<()> {
        use std::io::Write;
        let (offsets, flat) = match &self.storage {
            IndexerStorage::Owned { rank_w_offsets, flat_rank_b } => {
                (rank_w_offsets.as_slice(), flat_rank_b.as_slice())
            }
            IndexerStorage::Mmap { .. } => {
                return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput,
                    "save_to_path called on Mmap-backed indexer"));
            }
        };
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let tmp_path = path.with_extension("bin.tmp");
        {
            let f = std::fs::File::create(&tmp_path)?;
            let mut w = std::io::BufWriter::new(f);
            let n_rank_w = (offsets.len() - 1) as u32;

            let mut header = [0u8; INDEXER_HEADER_SIZE];
            header[0..4].copy_from_slice(&INDEXER_MAGIC);
            header[4..6].copy_from_slice(&INDEXER_VERSION.to_le_bytes());
            header[6] = self.subspace.w_board;
            header[7] = self.subspace.b_board;
            header[8..12].copy_from_slice(&n_rank_w.to_le_bytes());
            header[12..20].copy_from_slice(&self.n_canonical_entries.to_le_bytes());
            w.write_all(&header)?;

            for &v in offsets {
                w.write_all(&v.to_le_bytes())?;
            }
            for &v in flat {
                w.write_all(&v.to_le_bytes())?;
            }
            w.flush()?;
            w.into_inner()?.sync_all()?;
        }
        std::fs::rename(&tmp_path, path)?;
        Ok(())
    }

    /// Try mmap-loading from `cache_dir`. If the file doesn't exist or is
    /// corrupt, fall back to building from scratch and (best-effort) save
    /// the result so the next process to hit this subspace gets the cache.
    /// Concurrent first-time builds across multiple processes will each
    /// produce the same bytes, and the atomic rename in `save_to_path`
    /// ensures the on-disk file is never half-written.
    pub fn open_or_build(sub: Subspace, cache_dir: Option<&Path>) -> Self {
        if let Some(dir) = cache_dir {
            let path = dir.join(indexer_filename(sub));
            if path.exists() {
                match Self::load_mmap(sub, &path) {
                    Ok(idx) => return idx,
                    Err(e) => {
                        eprintln!("    indexer cache load failed for ({},{}) at {}: {}. Rebuilding.",
                            sub.w_board, sub.b_board, path.display(), e);
                    }
                }
            }
            let idx = Self::build(sub);
            if let Err(e) = idx.save_to_path(&path) {
                eprintln!("    indexer cache save failed for ({},{}) at {}: {}. Continuing without cache.",
                    sub.w_board, sub.b_board, path.display(), e);
            }
            idx
        } else {
            Self::build(sub)
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
        let offsets = self.storage.rank_w_offsets();
        let flat = self.storage.flat_rank_b();
        let start = offsets[rank_w as usize] as usize;
        let end = offsets[rank_w as usize + 1] as usize;
        let bucket = &flat[start..end];
        let pos_in_bucket = match bucket.binary_search(&rank_b) {
            Ok(i) => i,
            Err(_) => panic!(
                "canonical_index: (cw={:#x}, cb={:#x}) is not a canonical orbit \
                 (rank_w={}, rank_b={}). Caller must canonicalize first.",
                cw, cb, rank_w, rank_b
            ),
        };
        let pos = start as u64 + pos_in_bucket as u64;
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
    /// rank_w bucket, then direct indexing into the flat array to get
    /// rank_b, then unrank both halves. No reverse map stored — for
    /// (9,9) that would have cost ~6 GB extra RAM.
    #[inline]
    pub fn decode(&self, canonical_idx: u64) -> (u32, u32, u8) {
        let stm = (canonical_idx & 1) as u8 + 1;
        let pos = (canonical_idx >> 1) as u32;
        let offsets = self.storage.rank_w_offsets();
        let flat = self.storage.flat_rank_b();
        // Find rank_w bucket: the largest k with offsets[k] <= pos.
        let rank_w_usize = match offsets.binary_search(&pos) {
            // Exact hit: pos is the bucket's first entry.
            Ok(k) => {
                // Multiple consecutive buckets can share an offset if they
                // contain zero canonical entries. Skip past empties to land
                // on the actual bucket.
                let mut k = k;
                while k + 1 < offsets.len() && offsets[k + 1] == pos {
                    k += 1;
                }
                k
            }
            // No exact hit: insertion index k means offsets[k-1] < pos < offsets[k].
            Err(k) => k - 1,
        };
        let pos_in_bucket = (pos - offsets[rank_w_usize]) as usize;
        let rank_b = flat[offsets[rank_w_usize] as usize + pos_in_bucket];
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
        // Find ANY non-canonical orbit by enumerating raw positions and
        // checking against canonicalize(); the first hit is enough.
        let n_w = BINOM[24][sub.w_board as usize];
        let n_b = BINOM[(24 - sub.w_board) as usize][sub.b_board as usize];
        for rank_w in 0..n_w {
            let wbb = unrank_subset(rank_w, 24, sub.w_board as u32);
            for rank_b in 0..n_b {
                let compact_b = unrank_subset(rank_b, 24 - sub.w_board as u32, sub.b_board as u32);
                let bbb = expand_against(compact_b, wbb);
                let (cw, cb) = canonicalize(wbb, bbb);
                if (cw, cb) != (wbb, bbb) {
                    // Found a non-canonical. This should panic.
                    let _ = indexer.canonical_index(wbb, bbb, 1);
                    return;
                }
            }
        }
        panic!("(3,3) should have non-canonical orbits");
    }

    /// Save → mmap-load → query roundtrip on a small subspace. Ensures
    /// the on-disk format is byte-compatible with the in-memory layout
    /// and that mmap-backed queries return identical results.
    #[test]
    fn canonical_indexer_mmap_roundtrip_33() {
        let sub = Subspace::movement(3, 3);
        let built = CanonicalIndexer::build(sub);

        let tmp = std::env::temp_dir().join(format!(
            "morris_idx_test_w{}_b{}_{}.bin",
            sub.w_board, sub.b_board, std::process::id()
        ));
        built.save_to_path(&tmp).expect("save");

        let mmapped = CanonicalIndexer::load_mmap(sub, &tmp).expect("load");

        // Same totals.
        assert_eq!(mmapped.n_canonical_entries(), built.n_canonical_entries());
        assert_eq!(mmapped.n_states_canonical(), built.n_states_canonical());

        // Every canonical orbit produces the same index and decodes back.
        sub.enumerate_positions(|cw, cb| {
            for stm in [1u8, 2u8] {
                let idx_built = built.canonical_index(cw, cb, stm);
                let idx_mmap = mmapped.canonical_index(cw, cb, stm);
                assert_eq!(idx_built, idx_mmap, "index mismatch for ({:x},{:x},{})", cw, cb, stm);
                let dec = mmapped.decode(idx_mmap);
                assert_eq!(dec, (cw, cb, stm));
            }
        });

        std::fs::remove_file(&tmp).ok();
    }
}
