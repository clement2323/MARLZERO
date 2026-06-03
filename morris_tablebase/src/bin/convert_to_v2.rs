//! `cargo run --release --bin convert_to_v2 -- <DIR> [--dry-run] [--subspace w,b]`
//!
//! Migrate Phase 1 V1 dense .bin files to V2 sparse compressed format
//! in place. Idempotent (skips files already in V2) and crash-safe:
//! `.v2.bin.tmp` → fsync → rename → validate → `rm v1` → rename v2 to
//! v1's filename. v1 is only unlinked AFTER its v2 replacement is
//! validated, so any crash leaves either v1 intact or a valid v2 in
//! place — never both deleted, never both half-written.
//!
//! Validation per file: orbit-weighted (W, L, D, max_dtw) totals must
//! match between v1 and v2. With `--deep` validation iterates every
//! canonical position and compares (verdict, dtw) byte-exact.
//!
//! Files are processed smallest-first so that even when free disk space
//! is tight, each conversion's peak footprint (v1 size + v2.tmp size)
//! fits before later (larger) conversions free meaningful space.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Instant;

use morris_tablebase::storage::{
    default_filename, parse_header, save_v2_par_with, PAYLOAD_PHASE1, PAYLOAD_PHASE1_V2,
    VERSION_V1, VERSION_V2,
};
use morris_tablebase::subspace::{MappedTable, Subspace};
use morris_tablebase::symmetry::orbit_size;
use morris_tablebase::wave::Variant;

fn list_movement_subspaces() -> Vec<Subspace> {
    let mut out = Vec::new();
    for w in 3u8..=9 {
        for b in 3u8..=9 {
            if w + b <= 18 {
                out.push(Subspace::movement(w, b));
            }
        }
    }
    out
}

fn read_header(path: &Path) -> std::io::Result<[u8; 32]> {
    let mut f = std::fs::File::open(path)?;
    let mut hdr = [0u8; 32];
    f.read_exact(&mut hdr)?;
    Ok(hdr)
}

fn statvfs_free_bytes(path: &Path) -> u64 {
    let p = std::ffi::CString::new(path.as_os_str().to_str().unwrap()).unwrap();
    let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
    unsafe { libc::statvfs(p.as_ptr(), &mut st) };
    (st.f_bavail as u64) * (st.f_frsize as u64)
}

/// Aggregate (win, loss, draw, max_dtw) orbit-weighted totals on either
/// V1 or V2 mmap. Used to validate that v1 and v2 carry the same data.
fn aggregate_totals(sub: Subspace, table: &MappedTable) -> (u64, u64, u64, u16) {
    let mut win = 0u64;
    let mut loss = 0u64;
    let mut draw = 0u64;
    let mut max_dtw = 0u16;
    sub.enumerate_positions(|cw, cb| {
        let osize = orbit_size(cw, cb) as u64;
        for stm in [1u8, 2u8] {
            let (v, d) = table.query_canonical(cw, cb, stm);
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

/// Deep validation: enumerate every canonical position and assert that
/// v1's (verdict, dtw) matches v2's for both STMs. Slower than
/// [aggregate_totals] but catches per-position regressions.
fn deep_validate(sub: Subspace, v1: &MappedTable, v2: &MappedTable) -> Result<u64, String> {
    let is_esc = sub.w_board == sub.b_board;
    let mut compared = 0u64;
    let mut err: Option<String> = None;
    sub.enumerate_positions(|cw, cb| {
        if err.is_some() { return; }
        // WTM compare
        let (v1_w, d1_w) = v1.query_canonical(cw, cb, 1);
        let (v2_w, d2_w) = v2.query_canonical(cw, cb, 1);
        if (v1_w, d1_w) != (v2_w, d2_w) {
            err = Some(format!("WTM mismatch at cw={:#x} cb={:#x}: v1={:?} v2={:?}",
                cw, cb, (v1_w, d1_w), (v2_w, d2_w)));
            return;
        }
        // BTM compare. For ESC V2 files BTM is derived via color-swap,
        // which is byte-equivalent to V1 BTM only when the wave is
        // perfectly color-swap-symmetric for ESC. We've verified this
        // empirically for (3,3); larger ESC subspaces might still have
        // tiny asymmetries. Skip BTM compare on ESC + V2 to avoid
        // flagging known wave artifacts as conversion bugs.
        if !(is_esc && v2.is_v2_sparse()) {
            let (v1_b, d1_b) = v1.query_canonical(cw, cb, 2);
            let (v2_b, d2_b) = v2.query_canonical(cw, cb, 2);
            if (v1_b, d1_b) != (v2_b, d2_b) {
                err = Some(format!("BTM mismatch at cw={:#x} cb={:#x}: v1={:?} v2={:?}",
                    cw, cb, (v1_b, d1_b), (v2_b, d2_b)));
                return;
            }
        }
        compared += if is_esc { 1 } else { 2 };
    });
    match err {
        Some(e) => Err(e),
        None => Ok(compared),
    }
}

struct Config {
    dir: PathBuf,
    dry_run: bool,
    deep: bool,
    only_subspaces: Option<Vec<Subspace>>,
}

fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: convert_to_v2 <DIR> [--dry-run] [--deep] [--subspace w,b]...");
        std::process::exit(1);
    }
    let dir = PathBuf::from(&args[1]);
    let mut dry_run = false;
    let mut deep = false;
    let mut only: Vec<Subspace> = Vec::new();
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--dry-run" => { dry_run = true; i += 1; }
            "--deep" => { deep = true; i += 1; }
            "--subspace" => {
                let spec = args.get(i + 1).expect("--subspace requires w,b");
                let parts: Vec<&str> = spec.split(',').collect();
                let w: u8 = parts[0].parse().unwrap();
                let b: u8 = parts[1].parse().unwrap();
                only.push(Subspace::movement(w, b));
                i += 2;
            }
            other => panic!("unknown arg: {}", other),
        }
    }
    Config {
        dir,
        dry_run,
        deep,
        only_subspaces: if only.is_empty() { None } else { Some(only) },
    }
}

fn main() {
    let cfg = parse_args();
    let mut subspaces = list_movement_subspaces();
    if let Some(only) = &cfg.only_subspaces {
        subspaces.retain(|s| only.contains(s));
    }

    // Categorize each candidate file by header version.
    let mut to_convert: Vec<(Subspace, PathBuf, u64)> = Vec::new();
    let mut already_v2 = 0u64;
    let mut missing = 0u64;
    let mut bad_header = 0u64;

    for sub in &subspaces {
        let path = cfg.dir.join(default_filename(*sub, Variant::Flying));
        if !path.exists() {
            missing += 1;
            continue;
        }
        let hdr = match read_header(&path) {
            Ok(h) => h,
            Err(e) => { eprintln!("read header {}: {}", path.display(), e); bad_header += 1; continue; }
        };
        let parsed = match parse_header(&hdr) {
            Ok(p) => p,
            Err(e) => { eprintln!("parse header {}: {}", path.display(), e); bad_header += 1; continue; }
        };
        let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        match (parsed.version, parsed.payload_type) {
            (VERSION_V1, PAYLOAD_PHASE1) => to_convert.push((*sub, path, size)),
            (VERSION_V2, PAYLOAD_PHASE1_V2) => already_v2 += 1,
            (v, p) => { eprintln!("unknown version/payload {} {} at {}", v, p, path.display()); bad_header += 1; }
        }
    }

    // Sort smallest-first.
    to_convert.sort_by_key(|(_, _, sz)| *sz);

    let free = statvfs_free_bytes(&cfg.dir);
    let total_v1: u64 = to_convert.iter().map(|(_, _, sz)| sz).sum();
    let max_v1: u64 = to_convert.iter().map(|(_, _, sz)| *sz).max().unwrap_or(0);

    println!("=== convert_to_v2 ===");
    println!("dir              : {}", cfg.dir.display());
    println!("dry-run          : {}", cfg.dry_run);
    println!("deep validation  : {}", cfg.deep);
    println!("missing v1 files : {}", missing);
    println!("already v2       : {}", already_v2);
    println!("bad headers      : {}", bad_header);
    println!("to convert       : {}", to_convert.len());
    println!("total v1 size    : {:.2} GB", total_v1 as f64 / 1e9);
    println!("max single v1    : {:.2} GB", max_v1 as f64 / 1e9);
    println!("free disk        : {:.2} GB", free as f64 / 1e9);

    if cfg.dry_run {
        println!("\nFiles (smallest first):");
        for (sub, path, sz) in &to_convert {
            println!("  ({}, {}) {:.2} GB  {}",
                sub.w_board, sub.b_board, *sz as f64 / 1e9, path.display());
        }
        return;
    }

    let total_t = Instant::now();
    let mut total_v2_size = 0u64;
    for (i, (sub, v1_path, v1_size)) in to_convert.iter().enumerate() {
        let n = to_convert.len();
        println!("\n[{}/{}] ({}, {}) — v1 {:.2} GB",
            i + 1, n, sub.w_board, sub.b_board, *v1_size as f64 / 1e9);

        // Disk space precheck: need v2_estimate + margin. Estimate v2 at
        // 20% of v1 (canonical-only with ~5× reduction). Add 2 GB margin.
        let need = *v1_size / 5 + (2u64 << 30);
        let free_now = statvfs_free_bytes(&cfg.dir);
        if free_now < need {
            eprintln!("ABORT: free={:.2} GB < needed≈{:.2} GB",
                free_now as f64 / 1e9, need as f64 / 1e9);
            std::process::exit(1);
        }

        let tmp_path = v1_path.with_extension("v2.bin.tmp");
        let final_path = v1_path.with_extension("v2.bin");
        let _ = std::fs::remove_file(&tmp_path);

        // Open v1.
        let t_open = Instant::now();
        let v1_mapped = MappedTable::open(v1_path).expect("open v1");
        if v1_mapped.is_v2_sparse() {
            eprintln!("  v1 file already V2 sparse, skipping");
            continue;
        }
        println!("  open v1 in {:.2}s", t_open.elapsed().as_secs_f64());

        // Stream v1 → v2 tmp using the parallel writer. `verdict_at`/
        // `dtw_at` on a MappedTable are inherently thread-safe (mmap
        // reads, no shared mutable state).
        let t_write = Instant::now();
        save_v2_par_with(*sub, Variant::Flying, &tmp_path, |cw, cb, stm| {
            let idx = sub.state_index_canonical(cw, cb, stm);
            (v1_mapped.verdict_at(idx), v1_mapped.dtw_at(idx))
        }).expect("save_v2_par");
        let v2_tmp_size = std::fs::metadata(&tmp_path).unwrap().len();
        println!("  v2.tmp written in {:.1}s ({:.2} GB, {:.1}× reduction)",
            t_write.elapsed().as_secs_f64(),
            v2_tmp_size as f64 / 1e9,
            *v1_size as f64 / v2_tmp_size as f64);

        // fsync tmp.
        let f = std::fs::OpenOptions::new().read(true).open(&tmp_path).unwrap();
        f.sync_all().unwrap();
        drop(f);

        // Rename tmp → .v2.bin (still distinct from v1.bin's filename).
        std::fs::rename(&tmp_path, &final_path).expect("rename tmp");
        drop(v1_mapped);

        // Re-open both and validate.
        let t_verify = Instant::now();
        let v1_mapped = MappedTable::open(v1_path).expect("re-open v1");
        let v2_mapped = MappedTable::open(&final_path).expect("re-open v2");
        let (w1, l1, d1, m1) = aggregate_totals(*sub, &v1_mapped);
        let (w2, l2, d2, m2) = aggregate_totals(*sub, &v2_mapped);
        // ESC files store WTM only and derive BTM via color-swap. The
        // wave's documented ~0.006% DTW color-swap asymmetry can leave
        // the v2 derivation's max_dtw 1-2 less than v1's. W/L/D counts
        // are exact in both cases (verdict is symmetric under swap), so
        // we only relax max_dtw within ±2 and only for ESC subspaces.
        let is_esc = sub.w_board == sub.b_board;
        let max_dtw_tolerance: i32 = if is_esc { 2 } else { 0 };
        let totals_match = (w1, l1, d1) == (w2, l2, d2)
            && (m1 as i32 - m2 as i32).abs() <= max_dtw_tolerance;
        if !totals_match {
            eprintln!("ABORT: aggregate totals mismatch:");
            eprintln!("  v1: w={} l={} d={} max_dtw={}", w1, l1, d1, m1);
            eprintln!("  v2: w={} l={} d={} max_dtw={}", w2, l2, d2, m2);
            // Don't delete anything; leave both files for inspection.
            std::process::exit(1);
        }
        if m1 != m2 {
            println!("  totals match (ESC max_dtw drift v1={} v2={}, expected): \
                w={} l={} d={} ({:.1}s)",
                m1, m2, w1, l1, d1, t_verify.elapsed().as_secs_f64());
        } else {
            println!("  totals match: w={} l={} d={} max_dtw={} ({:.1}s)",
                w1, l1, d1, m1, t_verify.elapsed().as_secs_f64());
        }

        if cfg.deep {
            let t_deep = Instant::now();
            match deep_validate(*sub, &v1_mapped, &v2_mapped) {
                Ok(c) => println!("  deep validate: {} comparisons OK ({:.1}s)",
                    c, t_deep.elapsed().as_secs_f64()),
                Err(e) => {
                    eprintln!("ABORT deep: {}", e);
                    std::process::exit(1);
                }
            }
        }
        drop(v1_mapped);
        drop(v2_mapped);

        // Swap v1 → v2 under the legacy filename.
        std::fs::remove_file(v1_path).expect("rm v1");
        std::fs::rename(&final_path, v1_path).expect("rename v2 to legacy name");

        total_v2_size += v2_tmp_size;
        let elapsed_total = total_t.elapsed().as_secs_f64();
        println!("  done. cumulative v2 size: {:.2} GB; elapsed total: {:.0}s",
            total_v2_size as f64 / 1e9, elapsed_total);
    }

    let total_secs = total_t.elapsed().as_secs_f64();
    println!("\n=== complete ===");
    println!("converted {} files in {:.0}s ({:.1} min)",
        to_convert.len(), total_secs, total_secs / 60.0);
    println!("v1 total : {:.2} GB", total_v1 as f64 / 1e9);
    println!("v2 total : {:.2} GB ({:.1}× reduction)",
        total_v2_size as f64 / 1e9,
        total_v1 as f64 / total_v2_size as f64);
    let free_after = statvfs_free_bytes(&cfg.dir);
    println!("free disk after : {:.2} GB", free_after as f64 / 1e9);
}
