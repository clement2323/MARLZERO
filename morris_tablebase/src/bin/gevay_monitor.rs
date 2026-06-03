//! `cargo run --release --bin gevay_monitor -- <PHASE1_DIR> [--max-total N] [--poll-secs N]`
//!
//! Lightweight watcher that polls the Phase 1 directory and triggers a
//! Phase 2 wave on each movement work unit as soon as its dependencies
//! become V2-converted. Designed to run **concurrently** with the
//! `convert_to_v2` migration: as each smaller subspace flips from V1 to
//! V2, the monitor opportunistically solves the next WU and prints its
//! V_Gévay distribution.
//!
//! Constraints we lean into:
//! - mmap'd reads + the page cache make repeated `MappedTable::open`
//!   cheap (no full file load).
//! - The Phase 2 wave on a single WU is CPU-bound — disk contention with
//!   `convert_to_v2` is small once dependencies are warm.
//! - Ranks default to 0 for all subspaces (paper's ESC convention); the
//!   global ordinal ranking is recomputed later with the complete stats.
//!   This gives valid V_Gévay sign classification (WIN/LOSS/DRAW) even
//!   when the exact rank magnitude for non-ESC pairs isn't known yet.
//!
//! Stops when all WUs `total_pieces <= max_total` are solved (or
//! Ctrl-C). Prints a final summary.

use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use morris_tablebase::gevay::multi_value::{solve_esc_work_unit, solve_pair_work_unit};
use morris_tablebase::gevay::subspace_rank::Rank;
use morris_tablebase::storage::{default_filename, VERSION_V2};
use morris_tablebase::subspace::{MappedTable, Subspace, Tablebase};
use morris_tablebase::symmetry::orbit_size;
use morris_tablebase::wave::Variant;
use morris_tablebase::work_unit::{list_movement_work_units, WorkUnit};

fn file_is_v2(path: &Path) -> bool {
    let mut f = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return false,
    };
    let mut hdr = [0u8; 8];
    if f.read_exact(&mut hdr).is_err() {
        return false;
    }
    if &hdr[0..4] != b"MTBL" {
        return false;
    }
    u16::from_le_bytes([hdr[4], hdr[5]]) == VERSION_V2
}

/// Subspaces reachable from `wu` via a single capture (immediate Phase 1
/// dependencies the wave will query via `phase1_tb`).
fn immediate_secondary_subspaces(wu: &WorkUnit) -> Vec<Subspace> {
    let mut out: HashSet<Subspace> = HashSet::new();
    for &p in &wu.primary {
        // Capture by white removes one black piece.
        if p.b_board > 3 {
            out.insert(Subspace::movement(p.w_board, p.b_board - 1));
        }
        // Capture by black removes one white piece.
        if p.w_board > 3 {
            out.insert(Subspace::movement(p.w_board - 1, p.b_board));
        }
        // (If the capture would drop opp below 3 it's terminal — no
        // tablebase lookup needed for that branch.)
    }
    out.into_iter().collect()
}

fn count_distribution(sub: Subspace, first_key: &[i16]) -> (u64, u64, u64, u64, i16) {
    let mut win = 0u64;
    let mut loss = 0u64;
    let mut zero = 0u64;
    let mut nonzero = 0u64;
    let mut max_abs = 0i16;
    sub.enumerate_positions(|cw, cb| {
        let osize = orbit_size(cw, cb) as u64;
        for stm in [1u8, 2u8] {
            let idx = sub.state_index_canonical(cw, cb, stm) as usize;
            let fk = first_key[idx];
            let abs = fk.abs();
            if abs > max_abs {
                max_abs = abs;
            }
            if abs >= 20 {
                if fk > 0 {
                    win += osize;
                } else {
                    loss += osize;
                }
            } else if fk == 0 {
                zero += osize;
            } else {
                nonzero += osize;
            }
        }
    });
    (win, loss, zero, nonzero, max_abs)
}

fn wu_label(wu: &WorkUnit) -> String {
    if wu.is_esc {
        format!("ESC ({}, {})", wu.primary[0].w_board, wu.primary[0].b_board)
    } else {
        format!(
            "pair ({}, {}) + ({}, {})",
            wu.primary[0].w_board, wu.primary[0].b_board,
            wu.primary[1].w_board, wu.primary[1].b_board,
        )
    }
}

struct Args {
    dir: PathBuf,
    max_total: u8,
    poll_secs: u64,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: gevay_monitor <PHASE1_DIR> [--max-total N] [--poll-secs N]");
        std::process::exit(1);
    }
    let mut out = Args {
        dir: PathBuf::from(&args[1]),
        max_total: 18,
        poll_secs: 20,
    };
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--max-total" => {
                out.max_total = args[i + 1].parse().expect("--max-total wants u8");
                i += 2;
            }
            "--poll-secs" => {
                out.poll_secs = args[i + 1].parse().expect("--poll-secs wants u64");
                i += 2;
            }
            other => panic!("unknown arg: {}", other),
        }
    }
    out
}

fn main() {
    let cli = parse_args();
    let variant = Variant::Flying;

    let all_wus = list_movement_work_units(18);
    let to_solve: Vec<(usize, &WorkUnit)> = all_wus
        .iter()
        .enumerate()
        .filter(|(_, w)| w.total_pieces() <= cli.max_total)
        .collect();
    let to_solve_count = to_solve.len();

    // Provisional rank assignment: every subspace gets rank 0 (paper's ESC
    // convention). Correct for ESC WUs; the magnitude of non-ESC pair ranks
    // refines once all 49 stats are loaded — but the sign of each
    // first_key (WIN-class vs LOSS-class vs DRAW-class) is invariant under
    // the rank choice, so the qualitative distribution is meaningful now.
    let ranks: HashMap<Subspace, Rank> = all_wus
        .iter()
        .flat_map(|w| w.primary.iter())
        .map(|&s| (s, 0i16))
        .collect();

    println!("=== gevay_monitor ===");
    println!("dir          : {}", cli.dir.display());
    println!("max_total    : {} ({} WUs to solve)", cli.max_total, to_solve_count);
    println!("poll interval: {}s", cli.poll_secs);
    println!("ranks        : provisional all-0 (real ranks computed post-migration)\n");

    let mut done: HashSet<usize> = HashSet::new();
    let start = Instant::now();
    let mut last_summary = Instant::now();

    loop {
        let mut progress_this_round = false;
        for (i, wu) in &to_solve {
            if done.contains(i) { continue; }

            // All primary subspaces must be v2.
            let primaries_ok = wu.primary.iter().all(|s| {
                file_is_v2(&cli.dir.join(default_filename(*s, variant)))
            });
            if !primaries_ok { continue; }

            // All immediate-secondary subspaces (captures into smaller WUs)
            // must also be v2. Some may not exist (when opp_below_three
            // dominates) — we tolerate missing files for terminal-only
            // children.
            let secs = immediate_secondary_subspaces(wu);
            let secs_ok = secs.iter().all(|s| {
                let p = cli.dir.join(default_filename(*s, variant));
                !p.exists() || file_is_v2(&p)
            });
            if !secs_ok { continue; }

            // Solve.
            println!(
                "[+{:>4.0}s] {} — primaries + {} secondaries v2-ready; solving...",
                start.elapsed().as_secs_f64(),
                wu_label(wu),
                secs.len()
            );
            let t = Instant::now();
            let mut tb = Tablebase::new();
            for s in wu.primary.iter().chain(secs.iter()) {
                let p = cli.dir.join(default_filename(*s, variant));
                if !p.exists() { continue; }
                let m = MappedTable::open(&p).expect("open v2");
                tb.insert_mapped(m);
            }
            let results: Vec<(Vec<i16>, Vec<i16>)> = if wu.is_esc {
                let r0 = *ranks.get(&wu.primary[0]).unwrap_or(&0);
                let (fk, d) = solve_esc_work_unit(wu.primary[0], r0, variant, &tb, &ranks);
                vec![(fk, d)]
            } else {
                let r0 = *ranks.get(&wu.primary[0]).unwrap_or(&0);
                solve_pair_work_unit(wu, r0, variant, &tb, &ranks)
            };
            let solve_secs = t.elapsed().as_secs_f64();
            for (k, (first_key, _dtw)) in results.iter().enumerate() {
                let sub = wu.primary[k];
                let (w, l, z, nz, max_abs) = count_distribution(sub, first_key);
                let total = w + l + z + nz;
                println!(
                    "  ({}, {}) W={:>12} L={:>12} draws_0={:>12} draws_nz={:>12} (total {}, max|fk|={}) [{:.1}s]",
                    sub.w_board, sub.b_board,
                    w, l, z, nz, total, max_abs, solve_secs,
                );
            }
            done.insert(*i);
            progress_this_round = true;
        }

        if done.len() >= to_solve_count {
            println!(
                "\n=== complete: all {} target WUs solved in {:.0}s ===",
                to_solve_count,
                start.elapsed().as_secs_f64(),
            );
            break;
        }

        if !progress_this_round {
            // Quiet idle log every poll interval so the user knows we're alive.
            if last_summary.elapsed() >= Duration::from_secs(cli.poll_secs.max(15)) {
                let remaining: Vec<String> = to_solve.iter()
                    .filter(|(i, _)| !done.contains(i))
                    .take(5)
                    .map(|(_, w)| wu_label(w))
                    .collect();
                println!(
                    "[+{:>4.0}s] {}/{} solved; waiting on {} (showing first 5)...",
                    start.elapsed().as_secs_f64(),
                    done.len(),
                    to_solve_count,
                    remaining.join(", "),
                );
                last_summary = Instant::now();
            }
            std::thread::sleep(Duration::from_secs(cli.poll_secs));
        }
    }
}
