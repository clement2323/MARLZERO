//! Binary persistence for resolved subspace tables.
//!
//! File layout — one file per subspace, 32-byte header + payload:
//!
//! ```text
//! 0..4    magic = b"MTBL"
//! 4..6    version u16 (current = 1)
//! 6       variant u8 (0 = flying, 1 = no_flying)
//! 7       reserved (zero)
//! 8       w_board
//! 9       b_board
//! 10      w_to_place
//! 11      b_to_place
//! 12..20  n_states u64 little-endian
//! 20..32  reserved (zero)
//! 32..32+n_states           verdict array, 1 byte per slot
//! 32+n..32+n+2*n            dtw array, u16 little-endian per slot
//! ```
//!
//! Layout matches the in-memory [SubspaceTable] one-to-one so a future
//! mmap path can map each region directly without reshuffling.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::subspace::{Subspace, SubspaceTable};
use crate::wave::Variant;

const MAGIC: [u8; 4] = *b"MTBL";
const VERSION: u16 = 1;

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

/// Conventional filename for the on-disk table of a subspace.
pub fn default_filename(sub: Subspace, variant: Variant) -> String {
    let var = match variant { Variant::Flying => "flying", Variant::NoFlying => "noflying" };
    format!(
        "{}_w{}_b{}_wp{}_bp{}.bin",
        var, sub.w_board, sub.b_board, sub.w_to_place, sub.b_to_place
    )
}

/// Write a resolved subspace table to disk.
pub fn save(table: &SubspaceTable, variant: Variant, path: &Path) -> io::Result<()> {
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);

    let n_states = table.verdict.len() as u64;
    assert_eq!(table.verdict.len(), table.dtw.len(), "verdict/dtw length mismatch");

    let mut header = [0u8; 32];
    header[0..4].copy_from_slice(&MAGIC);
    header[4..6].copy_from_slice(&VERSION.to_le_bytes());
    header[6] = variant_byte(variant);
    header[8] = table.subspace.w_board;
    header[9] = table.subspace.b_board;
    header[10] = table.subspace.w_to_place;
    header[11] = table.subspace.b_to_place;
    header[12..20].copy_from_slice(&n_states.to_le_bytes());
    w.write_all(&header)?;

    w.write_all(&table.verdict)?;

    // Write dtw as u16 LE bytes — explicit loop keeps us host-endianness-free.
    let mut buf = [0u8; 8192];
    let mut cursor = 0usize;
    for &d in &table.dtw {
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
    w.flush()?;
    Ok(())
}

/// Read a subspace table from disk. Returns `(table, variant)`.
pub fn load(path: &Path) -> io::Result<(SubspaceTable, Variant)> {
    let f = File::open(path)?;
    let mut r = BufReader::new(f);

    let mut header = [0u8; 32];
    r.read_exact(&mut header)?;
    if header[0..4] != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad magic"));
    }
    let version = u16::from_le_bytes([header[4], header[5]]);
    if version != VERSION {
        return Err(io::Error::new(io::ErrorKind::InvalidData,
            format!("unsupported version {}", version)));
    }
    let variant = variant_from_byte(header[6])?;
    let subspace = Subspace {
        w_board: header[8],
        b_board: header[9],
        w_to_place: header[10],
        b_to_place: header[11],
    };
    let n_states = u64::from_le_bytes(header[12..20].try_into().unwrap()) as usize;

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
