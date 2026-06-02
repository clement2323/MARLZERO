//! `cargo run --release --bin compute_gevay -- <PHASE1_DIR> [GEVAY_OUT_DIR]`
//!
//! Phase 2 driver. Loads Phase 1 tables (mmap), computes the per-subspace
//! ranks via [morris_tablebase::gevay::subspace_rank::assign_ranks], and
//! prints the ranking table. The full multi-valued retrograde wave that
//! emits V-tables is not yet wired in this binary — see the TODO marker
//! at the end of `main`.
//!
//! For now this is a **scaffolding / inspection tool** : it lets us
//! cross-check our val_s values and ranks against paper Section IV-A
//! before we trust the wave loop.

use std::collections::HashMap;
use std::path::PathBuf;

use morris_tablebase::gevay::multi_value::{solve_esc_work_unit, LOSS_ABS, WIN_ABS};
use morris_tablebase::gevay::subspace_rank::{
    assign_ranks, compute_val, RankOverrides, StmCounts, SubspaceStats,
};
use morris_tablebase::storage::default_filename;
use morris_tablebase::subspace::{MappedTable, Subspace, Tablebase};
use morris_tablebase::symmetry::orbit_size;
use morris_tablebase::wave::Variant;
use morris_tablebase::work_unit::{list_movement_work_units, negate};

/// Orbit-weighted W/L/D counts split by STM, computed on a mmap'd Phase 1 table.
fn count_stats_mapped(sub: Subspace, table: &MappedTable) -> SubspaceStats {
    let mut out = SubspaceStats::default();
    sub.enumerate_positions(|wbb, bbb| {
        let osize = orbit_size(wbb, bbb) as u64;
        for stm in [1u8, 2u8] {
            let idx = sub.state_index_canonical(wbb, bbb, stm);
            let v = table.verdict_at(idx);
            let bucket = if stm == 1 {
                &mut out.white_to_move
            } else {
                &mut out.black_to_move
            };
            match v {
                1 => bucket.wins += osize,
                2 => bucket.losses += osize,
                3 => bucket.draws += osize,
                _ => {}
            }
        }
    });
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: compute_gevay <PHASE1_DIR> [GEVAY_OUT_DIR]");
        eprintln!("  PHASE1_DIR   : where to find flying_w*_b*_wp0_bp0.bin files");
        eprintln!("  GEVAY_OUT_DIR: where to write gevay_flying_w*_b*_wp0_bp0.bin (optional)");
        std::process::exit(1);
    }
    let phase1_dir = PathBuf::from(&args[1]);
    let _gevay_out: Option<PathBuf> = args.get(2).map(PathBuf::from);

    let variant = Variant::Flying;

    // Step 1 — enumerate work units in DAG order (already topological per total pieces).
    let work_units = list_movement_work_units(18);
    println!("=== compute_gevay: {} work units ===\n", work_units.len());

    // Step 2 — load Phase 1 stats per subspace via mmap.
    println!("Loading Phase 1 tables and computing per-subspace W/D/L stats...");
    let mut wtm_counts: HashMap<Subspace, StmCounts> = HashMap::new();
    let mut all_subs: Vec<Subspace> = Vec::new();
    for wu in &work_units {
        for &p in &wu.primary {
            if !all_subs.contains(&p) {
                all_subs.push(p);
            }
        }
    }
    for sub in &all_subs {
        let path = phase1_dir.join(default_filename(*sub, variant));
        if !path.exists() {
            eprintln!(
                "missing Phase 1 file for ({}, {}): {}",
                sub.w_board, sub.b_board, path.display()
            );
            std::process::exit(1);
        }
        let table = MappedTable::open(&path).expect("mmap Phase 1 table");
        let stats = count_stats_mapped(*sub, &table);
        wtm_counts.insert(*sub, stats.white_to_move);
        // Also store negated subspace's white-to-move via Invariant A:
        // (s, btm) verdict = (-s, wtm) verdict. We get it from the mirror
        // counts of the same table.
        wtm_counts.insert(negate(*sub), stats.black_to_move);
    }
    println!("Loaded stats for {} subspaces.\n", all_subs.len());

    // Step 3 — compute val_s per work unit.
    println!("Computing val_s per work unit...");
    let mut val_per_primary: HashMap<Subspace, f64> = HashMap::new();
    for wu in &work_units {
        let v = compute_val(wu, &wtm_counts);
        val_per_primary.insert(wu.primary[0], v);
        if !wu.is_esc {
            val_per_primary.insert(wu.primary[1], 1.0 - v);
        }
    }

    // Step 4 — ordinal ranking with paper hotfix on 8,9,0,0.
    let overrides = RankOverrides::paper_defaults();
    let ranks = assign_ranks(&work_units, &wtm_counts, &overrides);

    // Step 5 — print sorted ranking table.
    println!("\n=== Subspace ranks (Section IV-A) ===\n");
    println!(
        "{:>3} {:>3}  {:>7}  {:>5}",
        "w", "b", "val_s", "rank"
    );
    println!("{}", "-".repeat(28));
    let mut display: Vec<(Subspace, f64, i16)> = ranks
        .iter()
        .map(|(s, &r)| (*s, *val_per_primary.get(s).unwrap_or(&0.5), r))
        .collect();
    display.sort_by(|a, b| a.2.cmp(&b.2));
    for (s, v, r) in display {
        println!(
            "{:>3} {:>3}  {:>7.4}  {:>+5}",
            s.w_board, s.b_board, v, r
        );
    }

    // Step 6 — partial Phase 2 init pass on the (3,3) ESC work unit as a
    // smoke test. Full wave loop comes next session.
    println!("\n=== Smoke test: ESC (3,3) init-only pass ===\n");
    let sub33 = Subspace::movement(3, 3);
    let rank33 = *ranks.get(&sub33).unwrap_or(&0);

    // Build a Tablebase wrapping the on-disk Phase 1 (3,3) (or nothing,
    // since (3,3) has no smaller dependencies — all captures terminal).
    let mut phase1_tb = Tablebase::new();
    // Optional: insert smaller subspaces too. For (3,3) none are needed.
    let _ = &phase1_tb;

    let t = std::time::Instant::now();
    let (first_key, dtw) = solve_esc_work_unit(sub33, rank33, variant, &phase1_tb, &ranks);
    println!("Init-only solve took {:.2}s", t.elapsed().as_secs_f64());

    // Orbit-weighted distribution of first_key values (canonical iteration).
    let mut win_count = 0u64;
    let mut loss_count = 0u64;
    let mut zero_count = 0u64;
    let mut other_count = 0u64;
    sub33.enumerate_positions(|wbb, bbb| {
        let osize = orbit_size(wbb, bbb) as u64;
        for stm in [1u8, 2u8] {
            let idx = sub33.state_index_canonical(wbb, bbb, stm) as usize;
            let fk = first_key[idx];
            if fk >= WIN_ABS / 2 {
                win_count += osize;
            } else if fk <= LOSS_ABS / 2 {
                loss_count += osize;
            } else if fk == 0 {
                zero_count += osize;
            } else {
                other_count += osize;
            }
        }
    });
    println!(
        "\nFirst-key distribution at (3,3):\n  WIN-class : {:>10}\n  LOSS-class: {:>10}\n  zero      : {:>10}\n  other     : {:>10}",
        win_count, loss_count, zero_count, other_count
    );
    println!(
        "\nReminder: this is INIT ONLY — wave propagation through intra-WU\n\
         parents is not yet wired, so only positions that resolve at init\n\
         time (instant-WIN via mill+terminal-capture, stalemate, or fully\n\
         cross-subspace-resolved) appear in WIN/LOSS buckets. The rest fall\n\
         into 'zero' as default stable-draw, INCLUDING positions that the\n\
         wave would have resolved as WIN or LOSS. Expect ~2.1M WIN-class\n\
         from instant mills, much fewer LOSS, the vast majority zero.\n\
         For comparison Phase 1 (3,3) full wave gave 4.46M WIN, 0.91M LOSS,\n\
         8.2k DRAW per pair of STMs.");

    let _ = dtw; // suppress unused warning
}
