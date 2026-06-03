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
