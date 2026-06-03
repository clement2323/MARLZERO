//! `cargo run --release --bin compute_gevay -- <PHASE1_DIR> [--save-to <DIR>] [--max-total N]`
//!
//! Phase 2 driver: solves the V_Gévay multi-valued retrograde wave over
//! every movement work unit in topological order (smallest total piece
//! count first). Loads Phase 1 tables (mmap, both V1 and V2 supported)
//! for cross-subspace classification during the wave.
//!
//! By default `--max-total = 6` so only the (3,3) ESC work unit runs —
//! that's the fast smoke path. Bump `--max-total` to 18 for the full
//! computation (~30-60 min wall clock estimated, see TODO in the source
//! about Phase 2 disk format before saving large subspaces). `--save-to`
//! writes per-subspace `gevay_flying_w{w}_b{b}_wp0_bp0.bin` files via
//! [storage::save_gevay].

use std::collections::HashMap;
use std::path::PathBuf;

use morris_tablebase::gevay::multi_value::{
    solve_esc_work_unit, solve_pair_work_unit, LOSS_ABS, WIN_ABS,
};
use morris_tablebase::gevay::subspace_rank::{
    assign_ranks, compute_val, RankOverrides, StmCounts, SubspaceStats,
};
use morris_tablebase::storage::{default_filename, gevay_filename, save_gevay};
use morris_tablebase::subspace::{MappedTable, Subspace, Tablebase};
use morris_tablebase::symmetry::orbit_size;
use morris_tablebase::wave::Variant;
use morris_tablebase::work_unit::{list_movement_work_units, negate, WorkUnit};

/// Orbit-weighted W/L/D counts split by STM, format-agnostic across V1/V2.
fn count_stats_mapped(sub: Subspace, table: &MappedTable) -> SubspaceStats {
    let mut out = SubspaceStats::default();
    sub.enumerate_positions(|cw, cb| {
        let osize = orbit_size(cw, cb) as u64;
        for stm in [1u8, 2u8] {
            let (v, _d) = table.query_canonical(cw, cb, stm);
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

/// Distribution of Gévay first_key values for a single primary subspace,
/// orbit-weighted. Buckets cover the three semantic classes the paper
/// cares about (terminal WIN/LOSS, stable DRAW at rank-0 = first_key=0),
/// plus a tally of how many draws ended up in a non-zero rank class.
struct GevayDistribution {
    win_class: u64,
    loss_class: u64,
    rank_zero_draws: u64,
    nonzero_draws: u64,
    max_abs_first_key: i16,
}

fn gevay_distribution(sub: Subspace, first_key: &[i16]) -> GevayDistribution {
    let mut d = GevayDistribution {
        win_class: 0,
        loss_class: 0,
        rank_zero_draws: 0,
        nonzero_draws: 0,
        max_abs_first_key: 0,
    };
    sub.enumerate_positions(|cw, cb| {
        let osize = orbit_size(cw, cb) as u64;
        for stm in [1u8, 2u8] {
            let idx = sub.state_index_canonical(cw, cb, stm) as usize;
            let fk = first_key[idx];
            let abs = fk.abs();
            if abs > d.max_abs_first_key {
                d.max_abs_first_key = abs;
            }
            if fk >= WIN_ABS / 2 {
                d.win_class += osize;
            } else if fk <= LOSS_ABS / 2 {
                d.loss_class += osize;
            } else if fk == 0 {
                d.rank_zero_draws += osize;
            } else {
                d.nonzero_draws += osize;
            }
        }
    });
    d
}

struct Args {
    phase1_dir: PathBuf,
    save_to: Option<PathBuf>,
    max_total: u8,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: compute_gevay <PHASE1_DIR> [--save-to <DIR>] [--max-total N]");
        std::process::exit(1);
    }
    let mut out = Args {
        phase1_dir: PathBuf::from(&args[1]),
        save_to: None,
        max_total: 6,
    };
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--save-to" => {
                out.save_to = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--max-total" => {
                out.max_total = args[i + 1].parse().expect("--max-total wants u8");
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

    // Step 1 — enumerate ALL work units (always 28 for movement) for ranking.
    let all_wus = list_movement_work_units(18);
    println!("=== compute_gevay ===");
    println!("phase1 dir       : {}", cli.phase1_dir.display());
    println!("save to          : {:?}", cli.save_to);
    println!("max_total        : {} (filters which WUs actually solve)", cli.max_total);
    println!("total WUs        : {}", all_wus.len());

    // Step 2 — load Phase 1 mmap tables and compute per-subspace stats.
    println!("\nLoading Phase 1 (mmap) + computing W/D/L stats...");
    let mut mapped_tables: HashMap<Subspace, MappedTable> = HashMap::new();
    let mut wtm_counts: HashMap<Subspace, StmCounts> = HashMap::new();
    let mut all_subs: Vec<Subspace> = Vec::new();
    for wu in &all_wus {
        for &p in &wu.primary {
            if !all_subs.contains(&p) {
                all_subs.push(p);
            }
        }
    }
    for sub in &all_subs {
        let path = cli.phase1_dir.join(default_filename(*sub, variant));
        if !path.exists() {
            eprintln!("missing Phase 1 file for ({}, {}): {}",
                sub.w_board, sub.b_board, path.display());
            std::process::exit(1);
        }
        let table = MappedTable::open(&path).expect("mmap Phase 1 table");
        let stats = count_stats_mapped(*sub, &table);
        wtm_counts.insert(*sub, stats.white_to_move);
        wtm_counts.insert(negate(*sub), stats.black_to_move);
        mapped_tables.insert(*sub, table);
    }
    println!("  stats loaded for {} subspaces.", all_subs.len());

    // Step 3 — val_s + ranks (paper Section IV-A).
    println!("\nComputing val_s + ordinal ranks...");
    let mut val_per_primary: HashMap<Subspace, f64> = HashMap::new();
    for wu in &all_wus {
        let v = compute_val(wu, &wtm_counts);
        val_per_primary.insert(wu.primary[0], v);
        if !wu.is_esc {
            val_per_primary.insert(wu.primary[1], 1.0 - v);
        }
    }
    let overrides = RankOverrides::paper_defaults();
    let ranks = assign_ranks(&all_wus, &wtm_counts, &overrides);

    println!("\n=== Subspace ranks (Section IV-A) ===");
    println!("{:>3} {:>3}  {:>7}  {:>5}", "w", "b", "val_s", "rank");
    println!("{}", "-".repeat(28));
    let mut display: Vec<(Subspace, f64, i16)> = ranks
        .iter()
        .map(|(s, &r)| (*s, *val_per_primary.get(s).unwrap_or(&0.5), r))
        .collect();
    display.sort_by(|a, b| a.2.cmp(&b.2));
    for (s, v, r) in display {
        println!("{:>3} {:>3}  {:>7.4}  {:>+5}",
            s.w_board, s.b_board, v, r);
    }

    // Step 4 — solve each WU in topological order (filtered by --max-total).
    let to_solve: Vec<&WorkUnit> = all_wus.iter()
        .filter(|wu| wu.total_pieces() <= cli.max_total)
        .collect();
    println!("\n=== Solving {} WUs (total_pieces <= {}) ===\n",
        to_solve.len(), cli.max_total);

    // Build a Tablebase from the mmap tables that the wave will use for
    // cross-subspace queries (Phase 1 verdicts + DTW). We rebuild it
    // here from the mapped_tables map because Tablebase takes ownership.
    let mut phase1_tb = Tablebase::new();
    for (_, table) in mapped_tables.drain() {
        phase1_tb.insert_mapped(table);
    }

    let total_t = std::time::Instant::now();
    for (i, wu) in to_solve.iter().enumerate() {
        let wu_label = if wu.is_esc {
            format!("ESC ({}, {})", wu.primary[0].w_board, wu.primary[0].b_board)
        } else {
            format!("pair ({}, {}) + ({}, {})",
                wu.primary[0].w_board, wu.primary[0].b_board,
                wu.primary[1].w_board, wu.primary[1].b_board)
        };
        let t_wu = std::time::Instant::now();
        println!("[{}/{}] {} — solving...", i + 1, to_solve.len(), wu_label);

        let results: Vec<(Vec<i16>, Vec<i16>)> = if wu.is_esc {
            let rank0 = *ranks.get(&wu.primary[0]).unwrap_or(&0);
            let (fk, d) = solve_esc_work_unit(wu.primary[0], rank0, variant, &phase1_tb, &ranks);
            vec![(fk, d)]
        } else {
            let rank0 = *ranks.get(&wu.primary[0]).unwrap_or(&0);
            solve_pair_work_unit(wu, rank0, variant, &phase1_tb, &ranks)
        };
        println!("  wave done in {:.2}s", t_wu.elapsed().as_secs_f64());

        for (k, (first_key, dtw)) in results.iter().enumerate() {
            let sub = wu.primary[k];
            let dist = gevay_distribution(sub, first_key);
            let total = dist.win_class + dist.loss_class + dist.rank_zero_draws + dist.nonzero_draws;
            println!(
                "  ({}, {}) wtm+btm: W={:>10} L={:>10} draws_0={:>10} draws_nz={:>10} (total {}, max|fk|={})",
                sub.w_board, sub.b_board,
                dist.win_class, dist.loss_class,
                dist.rank_zero_draws, dist.nonzero_draws,
                total, dist.max_abs_first_key,
            );

            if let Some(dir) = &cli.save_to {
                if !dir.exists() {
                    std::fs::create_dir_all(dir).expect("create save dir");
                }
                let path = dir.join(gevay_filename(sub, variant));
                let t_save = std::time::Instant::now();
                save_gevay(sub, variant, first_key, dtw, &path).expect("save_gevay");
                println!("    saved to {} ({:.1}s)",
                    path.display(), t_save.elapsed().as_secs_f64());
            }
        }
    }

    println!("\n=== complete: {} WUs solved in {:.1}s ===",
        to_solve.len(), total_t.elapsed().as_secs_f64());
}
