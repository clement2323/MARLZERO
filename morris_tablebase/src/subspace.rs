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
use crate::storage::{entry_stride_for, read_v2_entry, read_v2_rank_b};
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

    /// Total number of raw (wbb, bbb) positions in this subspace.
    /// Returns u64 because subspaces >= (6,6) exceed u32::MAX × 2 once
    /// multiplied by 2 STMs (e.g. (9,9) has ~13 billion states).
    pub fn n_positions(&self) -> u64 {
        let w = self.w_board as usize;
        let b = self.b_board as usize;
        BINOM[24][w] as u64 * BINOM[24 - w][b] as u64
    }

    pub fn n_states(&self) -> u64 {
        self.n_positions() * 2
    }

    /// Index a `(wbb, bbb, stm)` state. Internally canonicalises the
    /// bitmask pair under D4, so any orbit member maps to the canonical
    /// representative's slot. Side-to-move is kept on the low bit.
    #[inline]
    pub fn state_index(&self, wbb: u32, bbb: u32, stm: u8) -> u64 {
        let (cw, cb) = canonicalize(wbb, bbb);
        self.state_index_canonical(cw, cb, stm)
    }

    /// Index assuming the caller already has the canonical bitmasks (skips
    /// the 8-way canonicalisation pass). Used in tight wave inner loops.
    #[inline]
    pub fn state_index_canonical(&self, cw: u32, cb: u32, stm: u8) -> u64 {
        let rank_w = rank_subset(cw) as u64;
        let compact_b = compact_against(cb, cw);
        let rank_b = rank_subset(compact_b) as u64;
        let n_b = BINOM[24 - self.w_board as usize][self.b_board as usize] as u64;
        let pos = rank_w * n_b + rank_b;
        pos * 2 + (stm - 1) as u64
    }

    /// Decode a slot index back to its canonical `(wbb, bbb, stm)`.
    /// The returned bitmasks are always the orbit's canonical representative.
    #[inline]
    pub fn decode_state(&self, state_idx: u64) -> (u32, u32, u8) {
        let stm = (state_idx & 1) as u8 + 1;
        let pos = state_idx >> 1;
        let n_b = BINOM[24 - self.w_board as usize][self.b_board as usize] as u64;
        let rank_w = (pos / n_b) as u32;
        let rank_b = (pos % n_b) as u32;
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

/// Store of resolved subspace tables — used for cross-subspace queries
/// during the wave. Tables are kept either fully in RAM (Vec) for the
/// subspace currently being solved, or backed by a memory-mapped `.bin`
/// file for previously resolved subspaces (the kernel page cache acts as
/// our automatic working-set manager, capping resident RAM).
#[derive(Default)]
pub struct Tablebase {
    tables: std::collections::HashMap<Subspace, StoredTable>,
}

/// One entry in [Tablebase]: either an in-RAM table (owned Vec) or a
/// memory-mapped read-only view of a `.bin` file on disk.
pub enum StoredTable {
    Owned(SubspaceTable),
    Mapped(MappedTable),
}

pub struct SubspaceTable {
    pub subspace: Subspace,
    pub verdict: Vec<u8>,
    pub dtw: Vec<u16>,
}

/// Memory-mapped read-only view of a persisted subspace table.
/// Two on-disk formats are supported (auto-detected via header):
/// - V1 dense: full `n_states` slots, indexed directly by
///   [Subspace::state_index_canonical]. Exposes [Self::verdict_at] /
///   [Self::dtw_at].
/// - V2 sparse Phase 1: canonical-only entries grouped by `rank_w` with
///   a binary-search index. Exposes [Self::query_canonical] (the new
///   primary API). V1Dense also implements `query_canonical` so callers
///   can be format-agnostic.
pub struct MappedTable {
    pub subspace: Subspace,
    /// Keeps the mmap alive — never accessed directly after construction.
    _mmap: memmap2::Mmap,
    backend: MappedBackend,
}

enum MappedBackend {
    V1Dense {
        verdict_ptr: *const u8,
        dtw_ptr: *const u16,
        n_states: usize,
    },
    V2Sparse {
        offsets_ptr: *const u64,
        n_rank_w: u32,
        entries_ptr: *const u8,
        is_esc: bool,
        entry_stride: usize,
    },
}

// SAFETY: MappedTable is read-only after construction; the raw pointers
// stay valid because we keep `_mmap` alive. Send/Sync is sound because
// the mmap region is immutable.
unsafe impl Send for MappedTable {}
unsafe impl Sync for MappedTable {}

impl MappedTable {
    /// Open a `.bin` file and return a read-only mapped view. Auto-detects
    /// V1 dense vs V2 sparse Phase 1 via the header version byte.
    pub fn open(path: &std::path::Path) -> std::io::Result<Self> {
        use std::io::{Error, ErrorKind};
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        if mmap.len() < 32 {
            return Err(Error::new(ErrorKind::InvalidData, "file too small for header"));
        }
        let mut header = [0u8; 32];
        header.copy_from_slice(&mmap[0..32]);
        let parsed = crate::storage::parse_header(&header)?;

        match parsed.version {
            crate::storage::VERSION_V1 => {
                if parsed.payload_type != crate::storage::PAYLOAD_PHASE1 {
                    return Err(Error::new(ErrorKind::InvalidData,
                        format!("MappedTable::open expects Phase 1 payload, got {}", parsed.payload_type)));
                }
                let n_states = parsed.n_primary as usize;
                let expected_len = 32 + n_states + n_states * 2;
                if mmap.len() < expected_len {
                    return Err(Error::new(ErrorKind::InvalidData,
                        format!("v1 file shorter than declared n_states ({} vs {})", mmap.len(), expected_len)));
                }
                // SAFETY: bounds validated above; mmap stays alive.
                let verdict_ptr = unsafe { mmap.as_ptr().add(32) };
                let dtw_ptr = unsafe { mmap.as_ptr().add(32 + n_states) as *const u16 };
                Ok(Self {
                    subspace: parsed.subspace,
                    _mmap: mmap,
                    backend: MappedBackend::V1Dense { verdict_ptr, dtw_ptr, n_states },
                })
            }
            crate::storage::VERSION_V2 => {
                let extra = parsed.v2_extra.expect("v2_extra present for v2 header");
                let entry_stride = entry_stride_for(extra.is_esc);
                let offsets_bytes = (extra.n_rank_w as usize + 1) * 8;
                let entries_bytes = (parsed.n_primary as usize) * entry_stride;
                let expected_len = 32 + offsets_bytes + entries_bytes;
                if mmap.len() < expected_len {
                    return Err(Error::new(ErrorKind::InvalidData,
                        format!("v2 file shorter than expected ({} vs {})", mmap.len(), expected_len)));
                }
                // SAFETY: bounds validated; little-endian raw reads in
                // `query_canonical` use `read_unaligned`.
                let offsets_ptr = unsafe { mmap.as_ptr().add(32) as *const u64 };
                let entries_ptr = unsafe { mmap.as_ptr().add(32 + offsets_bytes) };
                Ok(Self {
                    subspace: parsed.subspace,
                    _mmap: mmap,
                    backend: MappedBackend::V2Sparse {
                        offsets_ptr,
                        n_rank_w: extra.n_rank_w,
                        entries_ptr,
                        is_esc: extra.is_esc,
                        entry_stride,
                    },
                })
            }
            v => Err(Error::new(ErrorKind::InvalidData,
                format!("unsupported header version {}", v))),
        }
    }

    /// V1 legacy: get verdict at a dense state index. Panics on V2 files
    /// (use [Self::query_canonical] for format-agnostic lookup).
    #[inline]
    pub fn verdict_at(&self, idx: u64) -> u8 {
        match &self.backend {
            MappedBackend::V1Dense { verdict_ptr, n_states, .. } => {
                debug_assert!((idx as usize) < *n_states);
                unsafe { *verdict_ptr.add(idx as usize) }
            }
            MappedBackend::V2Sparse { .. } => {
                panic!("verdict_at called on V2Sparse MappedTable; use query_canonical");
            }
        }
    }

    /// V1 legacy: get DTW at a dense state index. Panics on V2 files.
    #[inline]
    pub fn dtw_at(&self, idx: u64) -> u16 {
        match &self.backend {
            MappedBackend::V1Dense { dtw_ptr, n_states, .. } => {
                debug_assert!((idx as usize) < *n_states);
                unsafe { (*dtw_ptr.add(idx as usize)).to_le() }
            }
            MappedBackend::V2Sparse { .. } => {
                panic!("dtw_at called on V2Sparse MappedTable; use query_canonical");
            }
        }
    }

    /// Look up the (verdict, DTW) for a canonical `(cw, cb, stm)` position
    /// in this table's subspace. Works on both V1 and V2 backends.
    ///
    /// For V2 ESC files (`w == b`), the `stm` argument is effectively
    /// ignored — only WTM is stored on disk. Callers that want to query
    /// BTM in an ESC subspace must first color-swap the inputs
    /// (`canonicalize(bbb, wbb)`) and call with `stm=WTM=1`. This
    /// transformation lives in [Tablebase::query] so most callers never
    /// see it.
    #[inline]
    pub fn query_canonical(&self, cw: u32, cb: u32, stm: u8) -> (u8, u16) {
        match &self.backend {
            MappedBackend::V1Dense { .. } => {
                let idx = self.subspace.state_index_canonical(cw, cb, stm);
                (self.verdict_at(idx), self.dtw_at(idx))
            }
            MappedBackend::V2Sparse { offsets_ptr, n_rank_w, entries_ptr, is_esc, entry_stride } => {
                let rank_w = rank_subset(cw);
                debug_assert!(rank_w < *n_rank_w,
                    "rank_w {} >= n_rank_w {}", rank_w, n_rank_w);
                let compact_b = compact_against(cb, cw);
                let rank_b = rank_subset(compact_b);

                // SAFETY: offsets_ptr is the head of a u64 array of length
                // n_rank_w+1; rank_w < n_rank_w from debug_assert.
                let start = unsafe { (*offsets_ptr.add(rank_w as usize)).to_le() } as usize;
                let end = unsafe { (*offsets_ptr.add(rank_w as usize + 1)).to_le() } as usize;
                let stride = *entry_stride;
                let n_in_group = (end - start) / stride;

                // Binary search by rank_b within this rank_w group.
                let mut lo = 0usize;
                let mut hi = n_in_group;
                while lo < hi {
                    let mid = lo + (hi - lo) / 2;
                    let entry_off = start + mid * stride;
                    let mid_rank_b = read_v2_rank_b(unsafe { entries_ptr.add(entry_off) });
                    if mid_rank_b < rank_b {
                        lo = mid + 1;
                    } else if mid_rank_b > rank_b {
                        hi = mid;
                    } else {
                        let p = unsafe { entries_ptr.add(entry_off) };
                        return read_v2_entry(p, stm, *is_esc);
                    }
                }
                // Not found: only reachable on a non-canonical query input
                // (caller bug — Tablebase::query canonicalises first).
                debug_assert!(false,
                    "v2 lookup miss: cw={:#x} cb={:#x} rank_w={} rank_b={}",
                    cw, cb, rank_w, rank_b);
                (0, 0)
            }
        }
    }

    /// Whether this table is backed by a V2 sparse file. Primarily for
    /// diagnostics and the migration verifier.
    pub fn is_v2_sparse(&self) -> bool {
        matches!(self.backend, MappedBackend::V2Sparse { .. })
    }
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
                    1 => { win += osize; if d > max_dtw { max_dtw = d; } }
                    2 => { loss += osize; if d > max_dtw { max_dtw = d; } }
                    3 => draw += osize,
                    _ => {}
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

    /// Insert an owned (Vec-backed) table. Use this only when keeping the
    /// table in RAM long-term is acceptable (typically only the currently
    /// solving subspace and tiny test cases).
    pub fn insert(&mut self, table: SubspaceTable) {
        self.tables.insert(table.subspace, StoredTable::Owned(table));
    }

    /// Insert a memory-mapped read-only table. Preferred path for all
    /// already-persisted subspaces — keeps RAM use bounded by the OS
    /// page cache.
    pub fn insert_mapped(&mut self, table: MappedTable) {
        self.tables.insert(table.subspace, StoredTable::Mapped(table));
    }

    pub fn contains(&self, sub: &Subspace) -> bool {
        self.tables.contains_key(sub)
    }

    /// Look up a position in a previously resolved subspace. Handles both
    /// the v1 dense and v2 sparse on-disk formats transparently. Applies
    /// one storage-level color-swap symmetry for ESC subspaces with a BTM
    /// query: the v2 ESC file stores WTM only, so we swap the inputs
    /// (`(wbb, bbb, BTM) ≡ (bbb, wbb, WTM)`) and look up WTM in the same
    /// file. For v1 dense tables this swap is also a valid equivalence;
    /// we apply it unconditionally for code uniformity.
    ///
    /// All 49 movement subspaces are stored independently (no mirror
    /// deletion). Cross-subspace color-swap is not used at the storage
    /// boundary — the wave's per-subspace DTW output is not perfectly
    /// color-swap-symmetric (~0.006% of positions diverge by a few plies
    /// on WIN/LOSS DTW), so each subspace keeps its own data.
    pub fn query(&self, sub: Subspace, wbb: u32, bbb: u32, stm: u8) -> Option<(u8, u16)> {
        if sub.w_board == sub.b_board && stm == 2 {
            // ESC + BTM: swap to WTM at the raw input level, then canonicalize.
            let (cw, cb) = canonicalize(bbb, wbb);
            self.lookup_in_table(sub, cw, cb, 1)
        } else {
            let (cw, cb) = canonicalize(wbb, bbb);
            self.lookup_in_table(sub, cw, cb, stm)
        }
    }

    #[inline]
    fn lookup_in_table(&self, sub: Subspace, cw: u32, cb: u32, stm: u8) -> Option<(u8, u16)> {
        match self.tables.get(&sub)? {
            StoredTable::Owned(t) => {
                let idx = sub.state_index_canonical(cw, cb, stm) as usize;
                Some((t.verdict[idx], t.dtw[idx]))
            }
            StoredTable::Mapped(t) => Some(t.query_canonical(cw, cb, stm)),
        }
    }
}
