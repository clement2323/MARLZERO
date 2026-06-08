//! Binary persistence for resolved subspace tables.
//!
//! Two on-disk formats coexist, distinguished by header `version` (bytes 4-5)
//! and `payload_type` (byte 7). [parse_header] dispatches on both.
//!
//! **V1 dense — version=1, payload_type ∈ {0, 1}** (32-byte header + dense payload):
//! ```text
//! 0..4    magic = b"MTBL"
//! 4..6    version u16 = 1
//! 6       variant u8 (0 flying, 1 noflying)
//! 7       payload_type u8 (0 = Phase 1 verdict+DTW, 1 = Phase 2 Gévay V+DTW)
//! 8..12   w_board, b_board, w_to_place, b_to_place
//! 12..20  n_states u64 LE
//! 20..32  reserved
//! 32..    payload (layout depends on payload_type)
//! ```
//! - Phase 1 dense payload: `verdict u8[n_states] || dtw u16 LE[n_states]`
//! - Gévay dense payload: `first_key i16 LE[n_states] || dtw i16 LE[n_states]`
//!
//! **V2 sparse Phase 1 — version=2, payload_type=10** (compressed, ~8.5×):
//! ```text
//! 0..4    magic = b"MTBL"
//! 4..6    version u16 = 2
//! 6       variant u8
//! 7       payload_type u8 = 10 (PAYLOAD_PHASE1_V2)
//! 8..12   w_board (>= b_board invariant), b_board, w_to_place, b_to_place
//! 12      flags u8 (bit 0: is_esc → 7-byte entries WTM-only;
//!                   else 10-byte both STMs)
//! 13..16  reserved
//! 16..24  n_canonical_entries u64 LE
//! 24..28  n_rank_w u32 LE (= C(24, w_board))
//! 28..32  reserved
//! ```
//! V2 payload sections (right after the 32-byte header):
//! - Section 1: `rank_w_offsets : u64 LE × (n_rank_w + 1)` — byte offset of
//!   each rank_w's first entry within section 2; `rank_w_offsets[n_rank_w]`
//!   equals the total byte size of section 2.
//! - Section 2: canonical entries sorted by (rank_w, rank_b_compact):
//!     - ESC (7 B): `rank_b u32 LE || verdict u8 || dtw u16 LE`
//!     - Non-ESC (10 B): `rank_b u32 LE || v_wtm u8 || d_wtm u16 LE
//!                       || v_btm u8 || d_btm u16 LE`

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use rayon::prelude::*;

use crate::hash::{unrank_subset, expand_against, BINOM};
use crate::subspace::{Subspace, SubspaceTable};
use crate::symmetry::canonicalize;
use crate::wave::Variant;

const MAGIC: [u8; 4] = *b"MTBL";
pub const VERSION_V1: u16 = 1;
pub const VERSION_V2: u16 = 2;

pub const PAYLOAD_PHASE1: u8 = 0;
pub const PAYLOAD_GEVAY: u8 = 1;
pub const PAYLOAD_PHASE1_V2: u8 = 10;
/// Phase 2 V_Gévay stored in canonical-only layout (one slot per D4 orbit
/// × WTM/BTM). Length = `2 × n_canonical_entries`, indexed by
/// [`crate::gevay::canonical_indexer::CanonicalIndexer::canonical_index`].
pub const PAYLOAD_GEVAY_CANONICAL: u8 = 11;

/// Parsed header. Field interpretation depends on (`version`, `payload_type`).
pub struct Header {
    pub version: u16,
    pub variant: Variant,
    pub payload_type: u8,
    pub subspace: Subspace,
    /// Primary count field:
    /// - V1 (any payload_type): n_states (dense slot count)
    /// - V2 Phase1: n_canonical_entries
    pub n_primary: u64,
    /// V2-only extras; `None` for V1.
    pub v2_extra: Option<V2Extra>,
}

pub struct V2Extra {
    /// True iff `w_board == b_board`: 7-byte WTM-only entries.
    /// False: 10-byte entries with both STMs.
    pub is_esc: bool,
    /// Length of `rank_w_offsets` is `n_rank_w + 1`. Equals `C(24, w_board)`.
    pub n_rank_w: u32,
}

fn variant_byte(v: Variant) -> u8 {
    match v {
        Variant::Flying => 0,
        Variant::NoFlying => 1,
    }
}

fn variant_from_byte(b: u8) -> io::Result<Variant> {
    match b {
        0 => Ok(Variant::Flying),
        1 => Ok(Variant::NoFlying),
        _ => Err(io::Error::new(io::ErrorKind::InvalidData, "unknown variant byte")),
    }
}

/// Filename for the on-disk Phase 1 table (both v1 dense and v2 compressed
/// share this filename; readers auto-detect via header version byte).
pub fn default_filename(sub: Subspace, variant: Variant) -> String {
    let var = match variant { Variant::Flying => "flying", Variant::NoFlying => "noflying" };
    format!(
        "{}_w{}_b{}_wp{}_bp{}.bin",
        var, sub.w_board, sub.b_board, sub.w_to_place, sub.b_to_place
    )
}

/// Filename for the on-disk Phase 2 (Gévay V table).
pub fn gevay_filename(sub: Subspace, variant: Variant) -> String {
    let var = match variant { Variant::Flying => "flying", Variant::NoFlying => "noflying" };
    format!(
        "gevay_{}_w{}_b{}_wp{}_bp{}.bin",
        var, sub.w_board, sub.b_board, sub.w_to_place, sub.b_to_place
    )
}

/// Decode the 32-byte header into [Header]. Returns `InvalidData` on bad
/// magic, unsupported version, or v2 invariant violations
/// (`w_board >= b_board`, `is_esc == (w==b)`).
pub fn parse_header(header: &[u8; 32]) -> io::Result<Header> {
    if header[0..4] != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
    }
    let version = u16::from_le_bytes([header[4], header[5]]);
    let variant = variant_from_byte(header[6])?;
    let payload_type = header[7];
    let subspace = Subspace {
        w_board: header[8],
        b_board: header[9],
        w_to_place: header[10],
        b_to_place: header[11],
    };

    match version {
        VERSION_V1 => {
            let n_states = u64::from_le_bytes(header[12..20].try_into().unwrap());
            Ok(Header { version, variant, payload_type, subspace, n_primary: n_states, v2_extra: None })
        }
        VERSION_V2 => {
            if payload_type != PAYLOAD_PHASE1_V2 {
                return Err(io::Error::new(io::ErrorKind::InvalidData,
                    format!("v2 expects PAYLOAD_PHASE1_V2=10, got {}", payload_type)));
            }
            let flags = header[12];
            let is_esc = (flags & 1) != 0;
            if is_esc != (subspace.w_board == subspace.b_board) {
                return Err(io::Error::new(io::ErrorKind::InvalidData,
                    "v2 is_esc flag inconsistent with w_board==b_board"));
            }
            let n_canonical = u64::from_le_bytes(header[16..24].try_into().unwrap());
            let n_rank_w = u32::from_le_bytes(header[24..28].try_into().unwrap());
            Ok(Header {
                version, variant, payload_type, subspace,
                n_primary: n_canonical,
                v2_extra: Some(V2Extra { is_esc, n_rank_w }),
            })
        }
        _ => Err(io::Error::new(io::ErrorKind::InvalidData,
            format!("unsupported version {}", version))),
    }
}

/// Write a Phase 1 V1 dense (verdict + DTW) subspace table to disk.
pub fn save(table: &SubspaceTable, variant: Variant, path: &Path) -> io::Result<()> {
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);

    let n_states = table.verdict.len() as u64;
    assert_eq!(table.verdict.len(), table.dtw.len(), "verdict/dtw length mismatch");

    let mut header = [0u8; 32];
    header[0..4].copy_from_slice(&MAGIC);
    header[4..6].copy_from_slice(&VERSION_V1.to_le_bytes());
    header[6] = variant_byte(variant);
    header[7] = PAYLOAD_PHASE1;
    header[8] = table.subspace.w_board;
    header[9] = table.subspace.b_board;
    header[10] = table.subspace.w_to_place;
    header[11] = table.subspace.b_to_place;
    header[12..20].copy_from_slice(&n_states.to_le_bytes());
    w.write_all(&header)?;

    w.write_all(&table.verdict)?;
    write_u16_le_chunked(&mut w, &table.dtw)?;
    w.flush()?;
    Ok(())
}

/// Write a Phase 1 V2 compressed (canonical-only sparse) subspace table
/// using `(verdict, dtw)` pairs read from a dense in-RAM SubspaceTable.
pub fn save_v2(table: &SubspaceTable, variant: Variant, path: &Path) -> io::Result<()> {
    let sub = table.subspace;
    save_v2_with(sub, variant, path, |cw, cb, stm| {
        let idx = sub.state_index_canonical(cw, cb, stm) as usize;
        (table.verdict[idx], table.dtw[idx])
    })
}

/// Write a Phase 1 V2 compressed file by streaming entries through a
/// caller-supplied getter `get(cw, cb, stm) -> (verdict, dtw)`. Use this
/// when the data source isn't a dense in-RAM `SubspaceTable` — e.g. the
/// migration tool reads from a mmap'd V1 file, and a re-solve could
/// stream directly from the wave's working arrays.
///
/// Works on any subspace — `w_board` can be `>`, `=`, or `<` `b_board`.
/// For ESC subspaces (`w == b`) the file stores WTM only — BTM is
/// recovered at query time by color-swapping within the same subspace.
/// For non-ESC both STMs are stored per entry (10 bytes vs 7).
pub fn save_v2_with<F>(sub: Subspace, variant: Variant, path: &Path, mut get: F) -> io::Result<()>
where
    F: FnMut(u32, u32, u8) -> (u8, u16),
{
    let is_esc = sub.w_board == sub.b_board;
    let entry_stride: usize = if is_esc { 7 } else { 10 };
    let w_count = sub.w_board as u32;
    let b_count = sub.b_board as u32;
    let n_rank_w = BINOM[24][w_count as usize];
    let n_rank_b = BINOM[(24 - w_count) as usize][b_count as usize];

    // Pass 1: count canonical entries per rank_w, build offsets.
    let mut rank_w_counts: Vec<u64> = vec![0; n_rank_w as usize];
    for rank_w in 0..n_rank_w {
        let wbb = unrank_subset(rank_w, 24, w_count);
        for rank_b in 0..n_rank_b {
            let compact_b = unrank_subset(rank_b, 24 - w_count, b_count);
            let bbb = expand_against(compact_b, wbb);
            let (cw, cb) = canonicalize(wbb, bbb);
            if (cw, cb) == (wbb, bbb) {
                rank_w_counts[rank_w as usize] += 1;
            }
        }
    }
    let mut offsets: Vec<u64> = Vec::with_capacity(n_rank_w as usize + 1);
    offsets.push(0);
    let mut cum: u64 = 0;
    for &c in &rank_w_counts {
        cum += c * entry_stride as u64;
        offsets.push(cum);
    }
    let n_canonical_entries: u64 = rank_w_counts.iter().sum();

    let f = File::create(path)?;
    let mut w = BufWriter::new(f);

    let mut header = [0u8; 32];
    header[0..4].copy_from_slice(&MAGIC);
    header[4..6].copy_from_slice(&VERSION_V2.to_le_bytes());
    header[6] = variant_byte(variant);
    header[7] = PAYLOAD_PHASE1_V2;
    header[8] = sub.w_board;
    header[9] = sub.b_board;
    header[10] = sub.w_to_place;
    header[11] = sub.b_to_place;
    header[12] = if is_esc { 1u8 } else { 0u8 };
    header[16..24].copy_from_slice(&n_canonical_entries.to_le_bytes());
    header[24..28].copy_from_slice(&n_rank_w.to_le_bytes());
    w.write_all(&header)?;

    for &off in &offsets {
        w.write_all(&off.to_le_bytes())?;
    }

    // Pass 2: write entries in (rank_w, rank_b_compact) ascending order.
    let mut entry_buf = [0u8; 10];
    for rank_w in 0..n_rank_w {
        let wbb = unrank_subset(rank_w, 24, w_count);
        for rank_b in 0..n_rank_b {
            let compact_b = unrank_subset(rank_b, 24 - w_count, b_count);
            let bbb = expand_against(compact_b, wbb);
            let (cw, cb) = canonicalize(wbb, bbb);
            if (cw, cb) != (wbb, bbb) {
                continue;
            }
            entry_buf[0..4].copy_from_slice(&rank_b.to_le_bytes());
            let (v_w, d_w) = get(cw, cb, 1);
            entry_buf[4] = v_w;
            entry_buf[5..7].copy_from_slice(&d_w.to_le_bytes());
            let payload_len: usize = if is_esc {
                7
            } else {
                let (v_b, d_b) = get(cw, cb, 2);
                entry_buf[7] = v_b;
                entry_buf[8..10].copy_from_slice(&d_b.to_le_bytes());
                10
            };
            w.write_all(&entry_buf[..payload_len])?;
        }
    }

    w.flush()?;
    Ok(())
}

/// Parallel variant of `save_v2_with` — uses rayon to canonicalize and
/// encode entries across rank_w buckets in parallel, then writes each
/// thread's local byte buffer to its computed file offset via `pwrite`
/// (no global mutex on the file handle). For larger subspaces this is
/// ~N× faster on a multi-core machine.
///
/// The `get` closure must be `Fn + Sync` (e.g. reads from a mmap'd V1
/// MappedTable, which is `Sync` by its manual impl).
pub fn save_v2_par_with<F>(
    sub: Subspace,
    variant: Variant,
    path: &Path,
    get: F,
) -> io::Result<()>
where
    F: Fn(u32, u32, u8) -> (u8, u16) + Sync,
{
    use std::os::unix::fs::FileExt;
    use std::sync::Arc;

    let is_esc = sub.w_board == sub.b_board;
    let entry_stride: usize = if is_esc { 7 } else { 10 };
    let w_count = sub.w_board as u32;
    let b_count = sub.b_board as u32;
    let n_rank_w = BINOM[24][w_count as usize];
    let n_rank_b = BINOM[(24 - w_count) as usize][b_count as usize];

    // Pass 1: count canonical entries per rank_w in parallel.
    // with_min_len batches consecutive rank_w into one rayon task —
    // critical for large subspaces (e.g. (8,8) has n_rank_w=735k) where
    // per-rank_w task overhead would otherwise dominate.
    let rank_w_counts: Vec<u64> = (0..n_rank_w)
        .into_par_iter()
        .with_min_len(4096)
        .map(|rank_w| {
            let wbb = unrank_subset(rank_w, 24, w_count);
            let mut cnt = 0u64;
            for rank_b in 0..n_rank_b {
                let compact_b = unrank_subset(rank_b, 24 - w_count, b_count);
                let bbb = expand_against(compact_b, wbb);
                let (cw, cb) = canonicalize(wbb, bbb);
                if (cw, cb) == (wbb, bbb) {
                    cnt += 1;
                }
            }
            cnt
        })
        .collect();

    let mut offsets: Vec<u64> = Vec::with_capacity(n_rank_w as usize + 1);
    offsets.push(0);
    let mut cum: u64 = 0;
    for &c in &rank_w_counts {
        cum += c * entry_stride as u64;
        offsets.push(cum);
    }
    let n_canonical_entries: u64 = rank_w_counts.iter().sum();
    let entries_total_bytes = cum;

    // Header + offsets section.
    let header_bytes = 32usize;
    let offsets_bytes = (n_rank_w as usize + 1) * 8;
    let entries_section_offset = (header_bytes + offsets_bytes) as u64;
    let total_file_bytes = entries_section_offset + entries_total_bytes;

    // Pre-allocate file and write header + offsets sequentially.
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(path)?;
    file.set_len(total_file_bytes)?;
    {
        let mut hdr = [0u8; 32];
        hdr[0..4].copy_from_slice(&MAGIC);
        hdr[4..6].copy_from_slice(&VERSION_V2.to_le_bytes());
        hdr[6] = variant_byte(variant);
        hdr[7] = PAYLOAD_PHASE1_V2;
        hdr[8] = sub.w_board;
        hdr[9] = sub.b_board;
        hdr[10] = sub.w_to_place;
        hdr[11] = sub.b_to_place;
        hdr[12] = if is_esc { 1u8 } else { 0u8 };
        hdr[16..24].copy_from_slice(&n_canonical_entries.to_le_bytes());
        hdr[24..28].copy_from_slice(&n_rank_w.to_le_bytes());
        file.write_at(&hdr, 0)?;

        let mut offset_buf = Vec::with_capacity(offsets_bytes);
        for &off in &offsets {
            offset_buf.extend_from_slice(&off.to_le_bytes());
        }
        file.write_at(&offset_buf, header_bytes as u64)?;
    }

    // Pass 2: build per-chunk entry buffers in parallel and pwrite each
    // chunk's concatenated buffer in a single syscall. Chunking rank_w
    // (rather than one task per rank_w) cuts both rayon scheduling
    // overhead and pwrite syscalls by ~1000× on large subspaces. Each
    // chunk is contiguous in rank_w so the v1 mmap reads stay sequential
    // within a chunk, friendly to the kernel's readahead.
    let file_arc = Arc::new(file);
    let chunk_size: u32 = (n_rank_w / 128).max(4096).min(n_rank_w.max(1));
    let chunk_ranges: Vec<(u32, u32)> = (0..n_rank_w)
        .step_by(chunk_size as usize)
        .map(|start| (start, (start + chunk_size).min(n_rank_w)))
        .collect();
    let result: io::Result<()> = chunk_ranges
        .par_iter()
        .try_for_each(|&(start, end)| {
            let chunk_entry_count: u64 = rank_w_counts[start as usize..end as usize]
                .iter()
                .sum();
            if chunk_entry_count == 0 {
                return Ok(());
            }
            let mut chunk_buf: Vec<u8> = Vec::with_capacity(
                chunk_entry_count as usize * entry_stride,
            );
            for rank_w in start..end {
                let wbb = unrank_subset(rank_w, 24, w_count);
                for rank_b in 0..n_rank_b {
                    let compact_b = unrank_subset(rank_b, 24 - w_count, b_count);
                    let bbb = expand_against(compact_b, wbb);
                    let (cw, cb) = canonicalize(wbb, bbb);
                    if (cw, cb) != (wbb, bbb) {
                        continue;
                    }
                    chunk_buf.extend_from_slice(&rank_b.to_le_bytes());
                    let (v_w, d_w) = get(cw, cb, 1);
                    chunk_buf.push(v_w);
                    chunk_buf.extend_from_slice(&d_w.to_le_bytes());
                    if !is_esc {
                        let (v_b, d_b) = get(cw, cb, 2);
                        chunk_buf.push(v_b);
                        chunk_buf.extend_from_slice(&d_b.to_le_bytes());
                    }
                }
            }
            let file_offset = entries_section_offset + offsets[start as usize];
            file_arc.write_at(&chunk_buf, file_offset)?;
            Ok(())
        });
    result?;
    Arc::try_unwrap(file_arc)
        .map_err(|_| io::Error::new(io::ErrorKind::Other, "file Arc still shared"))?
        .sync_all()?;
    Ok(())
}

/// Write a Phase 2 V1 (Gévay first_key + signed DTW) table to disk.
/// Both arrays are i16 little-endian.
pub fn save_gevay(
    subspace: Subspace,
    variant: Variant,
    first_key: &[i16],
    dtw: &[i16],
    path: &Path,
) -> io::Result<()> {
    assert_eq!(first_key.len(), dtw.len(), "first_key/dtw length mismatch");
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);

    let n_states = first_key.len() as u64;
    let mut header = [0u8; 32];
    header[0..4].copy_from_slice(&MAGIC);
    header[4..6].copy_from_slice(&VERSION_V1.to_le_bytes());
    header[6] = variant_byte(variant);
    header[7] = PAYLOAD_GEVAY;
    header[8] = subspace.w_board;
    header[9] = subspace.b_board;
    header[10] = subspace.w_to_place;
    header[11] = subspace.b_to_place;
    header[12..20].copy_from_slice(&n_states.to_le_bytes());
    w.write_all(&header)?;

    write_i16_le_chunked(&mut w, first_key)?;
    write_i16_le_chunked(&mut w, dtw)?;
    w.flush()?;
    Ok(())
}

fn write_u16_le_chunked<W: Write>(w: &mut W, data: &[u16]) -> io::Result<()> {
    let mut buf = [0u8; 8192];
    let mut cursor = 0usize;
    for &d in data {
        buf[cursor..cursor + 2].copy_from_slice(&d.to_le_bytes());
        cursor += 2;
        if cursor == buf.len() {
            w.write_all(&buf)?;
            cursor = 0;
        }
    }
    if cursor > 0 {
        w.write_all(&buf[..cursor])?;
    }
    Ok(())
}

fn write_i16_le_chunked<W: Write>(w: &mut W, data: &[i16]) -> io::Result<()> {
    let mut buf = [0u8; 8192];
    let mut cursor = 0usize;
    for &d in data {
        buf[cursor..cursor + 2].copy_from_slice(&d.to_le_bytes());
        cursor += 2;
        if cursor == buf.len() {
            w.write_all(&buf)?;
            cursor = 0;
        }
    }
    if cursor > 0 {
        w.write_all(&buf[..cursor])?;
    }
    Ok(())
}

/// Read a Phase 1 V1 dense (verdict + DTW) subspace table from disk.
/// Fails on V2 or Gévay files — use the appropriate loader.
pub fn load(path: &Path) -> io::Result<(SubspaceTable, Variant)> {
    let f = File::open(path)?;
    let mut r = BufReader::new(f);

    let mut header = [0u8; 32];
    r.read_exact(&mut header)?;
    let h = parse_header(&header)?;
    if h.version != VERSION_V1 {
        return Err(io::Error::new(io::ErrorKind::InvalidData,
            format!("load() expects V1; got version {}", h.version)));
    }
    if h.payload_type != PAYLOAD_PHASE1 {
        return Err(io::Error::new(io::ErrorKind::InvalidData,
            format!("expected payload_type=0 (Phase 1 V1), got {}", h.payload_type)));
    }
    let n_states = h.n_primary as usize;

    let mut verdict = vec![0u8; n_states];
    r.read_exact(&mut verdict)?;

    let mut dtw_bytes = vec![0u8; n_states * 2];
    r.read_exact(&mut dtw_bytes)?;
    let mut dtw = Vec::with_capacity(n_states);
    for i in 0..n_states {
        dtw.push(u16::from_le_bytes([dtw_bytes[2 * i], dtw_bytes[2 * i + 1]]));
    }

    Ok((SubspaceTable { subspace: h.subspace, verdict, dtw }, h.variant))
}

/// Read a Phase 2 (Gévay) V1 table. Returns `(first_key, dtw, variant, subspace)`.
pub fn load_gevay(path: &Path) -> io::Result<(Vec<i16>, Vec<i16>, Variant, Subspace)> {
    let f = File::open(path)?;
    let mut r = BufReader::new(f);

    let mut header = [0u8; 32];
    r.read_exact(&mut header)?;
    let h = parse_header(&header)?;
    if h.payload_type != PAYLOAD_GEVAY {
        return Err(io::Error::new(io::ErrorKind::InvalidData,
            format!("expected payload_type=1 (Gévay), got {}", h.payload_type)));
    }
    let n_states = h.n_primary as usize;

    let first_key = read_i16_le_vec(&mut r, n_states)?;
    let dtw = read_i16_le_vec(&mut r, n_states)?;
    Ok((first_key, dtw, h.variant, h.subspace))
}

/// Phase 2 V_Gévay save (canonical-only layout).
///
/// The Phase 2 wave produces `first_key` / `dtw` arrays sized to
/// `n_states_canonical = 2 × n_canonical_entries` (one slot per D4 orbit
/// × WTM/BTM), indexed by [`CanonicalIndexer::canonical_index`]. This
/// writer persists exactly that — no dense expansion. Readers reconstruct
/// the index by calling `CanonicalIndexer::build(subspace)` from the
/// header's subspace field.
///
/// Payload format (payload_type = [`PAYLOAD_GEVAY_CANONICAL`] = 11):
/// - Header: same 32-byte layout as save_gevay. Bytes 12..20 hold
///   `n_states_canonical` u64 LE (= length of each array).
/// - Body: `first_key i16 LE[n] || dtw i16 LE[n]` where n = n_states_canonical.
pub fn save_gevay_canonical(
    subspace: Subspace,
    variant: Variant,
    first_key: &[i16],
    dtw: &[i16],
    path: &Path,
) -> io::Result<()> {
    assert_eq!(first_key.len(), dtw.len(), "first_key/dtw length mismatch");
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);

    let n_states_canonical = first_key.len() as u64;
    let mut header = [0u8; 32];
    header[0..4].copy_from_slice(&MAGIC);
    header[4..6].copy_from_slice(&VERSION_V1.to_le_bytes());
    header[6] = variant_byte(variant);
    header[7] = PAYLOAD_GEVAY_CANONICAL;
    header[8] = subspace.w_board;
    header[9] = subspace.b_board;
    header[10] = subspace.w_to_place;
    header[11] = subspace.b_to_place;
    header[12..20].copy_from_slice(&n_states_canonical.to_le_bytes());
    w.write_all(&header)?;

    write_i16_le_chunked(&mut w, first_key)?;
    write_i16_le_chunked(&mut w, dtw)?;
    w.flush()?;
    Ok(())
}

/// Mmap-backed Gévay canonical table. The on-disk format is x86_64 LE
/// `i16`, identical to in-memory layout, so we expose `first_key`/`dtw`
/// as zero-copy `&[i16]` slices over the mapped pages.
///
/// Critical for `play_tb --gevay-dir` at serve time: loading all 49
/// subspaces into `Vec<i16>` would commit ~107 GB of RAM per process and
/// instantly OOM-kill any multi-worker setup. mmap defers the page
/// residency to the kernel's page cache, which is also shared across
/// processes that map the same file — so 8 self-play workers spawning 8
/// `play_tb` subprocesses pay the working-set RAM cost ONCE, not 8×.
pub struct MmapGevay {
    _mmap: memmap2::Mmap,  // keeps the mapping alive
    first_key_ptr: *const i16,
    dtw_ptr: *const i16,
    n: usize,
    pub variant: Variant,
    pub subspace: Subspace,
}

// SAFETY: the raw pointers point inside the mmap, which we own. No
// interior mutability: queries are read-only.
unsafe impl Send for MmapGevay {}
unsafe impl Sync for MmapGevay {}

impl MmapGevay {
    #[inline]
    pub fn first_key(&self) -> &[i16] {
        unsafe { std::slice::from_raw_parts(self.first_key_ptr, self.n) }
    }
    #[inline]
    pub fn dtw(&self) -> &[i16] {
        unsafe { std::slice::from_raw_parts(self.dtw_ptr, self.n) }
    }
    #[inline]
    pub fn len(&self) -> usize { self.n }
}

/// mmap-backed counterpart of [`load_gevay_canonical`]. Use this for
/// long-lived processes (the `play_tb --serve` JSONL loop, inference
/// servers) where loading 107 GB into RAM is not viable. The returned
/// slices are valid as long as the `MmapGevay` is alive.
pub fn load_gevay_canonical_mmap(path: &Path) -> io::Result<MmapGevay> {
    let f = File::open(path)?;
    let mmap = unsafe { memmap2::Mmap::map(&f)? };
    if mmap.len() < 32 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "gevay file too small for header"));
    }
    let header_arr: &[u8; 32] = mmap[..32].try_into()
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "header slice cast failed"))?;
    let h = parse_header(header_arr)?;
    if h.payload_type != PAYLOAD_GEVAY_CANONICAL {
        return Err(io::Error::new(io::ErrorKind::InvalidData,
            format!("expected payload_type={} (Gévay canonical), got {}",
                PAYLOAD_GEVAY_CANONICAL, h.payload_type)));
    }
    let n = h.n_primary as usize;
    let payload_bytes = (n as u64) * 2 * 2; // first_key + dtw, 2 bytes each
    let required = 32u64 + payload_bytes;
    if (mmap.len() as u64) < required {
        return Err(io::Error::new(io::ErrorKind::InvalidData,
            format!("gevay file truncated: have {} bytes, need {}", mmap.len(), required)));
    }
    // SAFETY: i16 has alignment 2; offset 32 is 32-byte aligned (way more
    // than enough). The pointer arithmetic stays inside the mapping (we
    // just checked `mmap.len() >= 32 + 4n`).
    let base = mmap.as_ptr();
    let first_key_ptr = unsafe { base.add(32) as *const i16 };
    let dtw_ptr = unsafe { base.add(32 + n * 2) as *const i16 };
    let variant = h.variant;
    let subspace = h.subspace;
    Ok(MmapGevay {
        _mmap: mmap,
        first_key_ptr,
        dtw_ptr,
        n,
        variant,
        subspace,
    })
}

/// Counterpart of [`save_gevay_canonical`]. Returns `(first_key, dtw, variant, subspace)`
/// where the arrays are indexed by `CanonicalIndexer::build(subspace).canonical_index(..)`.
///
/// Prefer [`load_gevay_canonical_mmap`] for long-lived processes — this
/// version eagerly copies the on-disk payload into `Vec<i16>`, which for
/// the largest subspaces costs up to ~9 GB per file.
pub fn load_gevay_canonical(path: &Path) -> io::Result<(Vec<i16>, Vec<i16>, Variant, Subspace)> {
    let f = File::open(path)?;
    let mut r = BufReader::new(f);

    let mut header = [0u8; 32];
    r.read_exact(&mut header)?;
    let h = parse_header(&header)?;
    if h.payload_type != PAYLOAD_GEVAY_CANONICAL {
        return Err(io::Error::new(io::ErrorKind::InvalidData,
            format!("expected payload_type={} (Gévay canonical), got {}",
                PAYLOAD_GEVAY_CANONICAL, h.payload_type)));
    }
    let n = h.n_primary as usize;
    let first_key = read_i16_le_vec(&mut r, n)?;
    let dtw = read_i16_le_vec(&mut r, n)?;
    Ok((first_key, dtw, h.variant, h.subspace))
}

fn read_i16_le_vec<R: Read>(r: &mut R, n: usize) -> io::Result<Vec<i16>> {
    let mut bytes = vec![0u8; n * 2];
    r.read_exact(&mut bytes)?;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(i16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]));
    }
    Ok(out)
}

// Internal helper exposed for the MappedTable v2 reader (kept private to the crate).
pub(crate) fn entry_stride_for(is_esc: bool) -> usize {
    if is_esc { 7 } else { 10 }
}

#[inline]
pub(crate) fn read_v2_entry(entry_ptr: *const u8, stm: u8, is_esc: bool) -> (u8, u16) {
    // SAFETY: caller guarantees entry_ptr points at a valid v2 entry of the
    // right stride, contained in a live mmap.
    unsafe {
        if is_esc {
            let v = *entry_ptr.add(4);
            let d = std::ptr::read_unaligned(entry_ptr.add(5) as *const u16).to_le();
            (v, d)
        } else if stm == 1 {
            let v = *entry_ptr.add(4);
            let d = std::ptr::read_unaligned(entry_ptr.add(5) as *const u16).to_le();
            (v, d)
        } else {
            let v = *entry_ptr.add(7);
            let d = std::ptr::read_unaligned(entry_ptr.add(8) as *const u16).to_le();
            (v, d)
        }
    }
}

#[inline]
pub(crate) fn read_v2_rank_b(entry_ptr: *const u8) -> u32 {
    unsafe { std::ptr::read_unaligned(entry_ptr as *const u32).to_le() }
}
