//! Binary persistence for resolved subspace tables.
//!
//! File layout — one file per subspace, 32-byte header + payload:
//!
//! ```text
//! 0..4    magic = b"MTBL"
//! 4..6    version u16 (current = 1)
//! 6       variant u8 (0 = flying, 1 = no_flying)
//! 7       payload_type u8 (0 = Phase 1 verdict+DTW, 1 = Phase 2 Gévay V+DTW)
//! 8       w_board
//! 9       b_board
//! 10      w_to_place
//! 11      b_to_place
//! 12..20  n_states u64 little-endian
//! 20..32  reserved (zero)
//! 32..    payload (layout depends on payload_type)
//! ```
//!
//! **Phase 1 payload (payload_type=0)**:
//! ```text
//! 32..32+n_states            verdict array, u8 per slot (UNKNOWN/WIN/LOSS/DRAW)
//! 32+n..32+n+2*n             dtw array, u16 LE per slot
//! ```
//!
//! **Phase 2 payload (payload_type=1)**: Gévay multi-valued table —
//! `first_key` is the relative game-theoretic value (rank class) signed,
//! `dtw` is the sign-adjusted DTW per Section IV-B-2 (also signed).
//! ```text
//! 32..32+2*n                 first_key array, i16 LE per slot
//! 32+2*n..32+4*n             dtw array, i16 LE per slot
//! ```
//!
//! Layout matches the in-memory tables one-to-one so a future mmap path
//! can map each region directly without reshuffling.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::subspace::{Subspace, SubspaceTable};
use crate::wave::Variant;

const MAGIC: [u8; 4] = *b"MTBL";
const VERSION: u16 = 1;

pub const PAYLOAD_PHASE1: u8 = 0;
pub const PAYLOAD_GEVAY: u8 = 1;

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

/// Conventional filename for the on-disk Phase 1 (verdict+DTW) table.
pub fn default_filename(sub: Subspace, variant: Variant) -> String {
    let var = match variant { Variant::Flying => "flying", Variant::NoFlying => "noflying" };
    format!(
        "{}_w{}_b{}_wp{}_bp{}.bin",
        var, sub.w_board, sub.b_board, sub.w_to_place, sub.b_to_place
    )
}

/// Conventional filename for the on-disk Phase 2 (Gévay V table).
pub fn gevay_filename(sub: Subspace, variant: Variant) -> String {
    let var = match variant { Variant::Flying => "flying", Variant::NoFlying => "noflying" };
    format!(
        "gevay_{}_w{}_b{}_wp{}_bp{}.bin",
        var, sub.w_board, sub.b_board, sub.w_to_place, sub.b_to_place
    )
}

/// Write a Phase 1 (verdict + DTW) subspace table to disk.
pub fn save(table: &SubspaceTable, variant: Variant, path: &Path) -> io::Result<()> {
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);

    let n_states = table.verdict.len() as u64;
    assert_eq!(table.verdict.len(), table.dtw.len(), "verdict/dtw length mismatch");

    let mut header = [0u8; 32];
    header[0..4].copy_from_slice(&MAGIC);
    header[4..6].copy_from_slice(&VERSION.to_le_bytes());
    header[6] = variant_byte(variant);
    header[7] = PAYLOAD_PHASE1;
    header[8] = table.subspace.w_board;
    header[9] = table.subspace.b_board;
    header[10] = table.subspace.w_to_place;
    header[11] = table.subspace.b_to_place;
    header[12..20].copy_from_slice(&n_states.to_le_bytes());
    w.write_all(&header)?;

    w.write_all(&table.verdict)?;
    write_i16_le_chunked(&mut w, &table.dtw.iter().map(|&x| x as i16).collect::<Vec<_>>())?;
    w.flush()?;
    Ok(())
}

/// Write a Phase 2 (Gévay first_key + signed DTW) table to disk.
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
    header[4..6].copy_from_slice(&VERSION.to_le_bytes());
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

/// Write a slice of i16 as little-endian bytes through a buffered writer.
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

/// Read a Phase 1 (verdict + DTW) subspace table from disk.
/// Fails if the file is actually a Phase 2 (Gévay) table — use `load_gevay`
/// for those.
pub fn load(path: &Path) -> io::Result<(SubspaceTable, Variant)> {
    let f = File::open(path)?;
    let mut r = BufReader::new(f);

    let mut header = [0u8; 32];
    r.read_exact(&mut header)?;
    let (variant, payload_type, subspace, n_states) = parse_header(&header)?;
    if payload_type != PAYLOAD_PHASE1 {
        return Err(io::Error::new(io::ErrorKind::InvalidData,
            format!("expected payload_type=0 (Phase 1), got {}", payload_type)));
    }

    let mut verdict = vec![0u8; n_states];
    r.read_exact(&mut verdict)?;

    let mut dtw_bytes = vec![0u8; n_states * 2];
    r.read_exact(&mut dtw_bytes)?;
    let mut dtw = Vec::with_capacity(n_states);
    for i in 0..n_states {
        dtw.push(u16::from_le_bytes([dtw_bytes[2 * i], dtw_bytes[2 * i + 1]]));
    }

    Ok((SubspaceTable { subspace, verdict, dtw }, variant))
}

/// Read a Phase 2 (Gévay) table. Returns `(first_key, dtw, variant, subspace)`.
pub fn load_gevay(path: &Path) -> io::Result<(Vec<i16>, Vec<i16>, Variant, Subspace)> {
    let f = File::open(path)?;
    let mut r = BufReader::new(f);

    let mut header = [0u8; 32];
    r.read_exact(&mut header)?;
    let (variant, payload_type, subspace, n_states) = parse_header(&header)?;
    if payload_type != PAYLOAD_GEVAY {
        return Err(io::Error::new(io::ErrorKind::InvalidData,
            format!("expected payload_type=1 (Gévay), got {}", payload_type)));
    }

    let first_key = read_i16_le_vec(&mut r, n_states)?;
    let dtw = read_i16_le_vec(&mut r, n_states)?;
    Ok((first_key, dtw, variant, subspace))
}

fn parse_header(header: &[u8; 32]) -> io::Result<(Variant, u8, Subspace, usize)> {
    if header[0..4] != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
    }
    let version = u16::from_le_bytes([header[4], header[5]]);
    if version != VERSION {
        return Err(io::Error::new(io::ErrorKind::InvalidData,
            format!("unsupported version {}", version)));
    }
    let variant = variant_from_byte(header[6])?;
    let payload_type = header[7];
    let subspace = Subspace {
        w_board: header[8],
        b_board: header[9],
        w_to_place: header[10],
        b_to_place: header[11],
    };
    let n_states = u64::from_le_bytes(header[12..20].try_into().unwrap()) as usize;
    Ok((variant, payload_type, subspace, n_states))
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
