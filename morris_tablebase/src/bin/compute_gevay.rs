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
    quick: bool,
    /// RAM budget per WU in GB. Anything that would exceed this is skipped
    /// rather than allocated — prevents OOM kills on (≥6,≥6) subspaces
    /// where dense wave Vecs blow up to 50+ GB. Default: MemAvailable - 4 GB.
    max_ram_gb: Option<f64>,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: compute_gevay <PHASE1_DIR> [--save-to <DIR>] [--max-total N] [--quick] [--max-ram-gb F]");
        eprintln!("  --quick: skip stats loading for subspaces with total > max_total + 2");
        eprintln!("           (ranks shown will be partial; useful when running in parallel with");
        eprintln!("            convert_to_v2 or testing a single small WU).");
        eprintln!("  --max-ram-gb: WU RAM budget in GB. WUs whose dense Vecs would exceed");
        eprintln!("                this are SKIPPED (logged, not solved). Default: MemAvailable - 4 GB.");
        std::process::exit(1);
    }
    let mut out = Args {
        phase1_dir: PathBuf::from(&args[1]),
        save_to: None,
        max_total: 6,
        quick: false,
        max_ram_gb: None,
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
            "--quick" => {
                out.quick = true;
                i += 1;
            }
            "--max-ram-gb" => {
                out.max_ram_gb = Some(args[i + 1].parse().expect("--max-ram-gb wants f64"));
                i += 2;
            }
            other => panic!("unknown arg: {}", other),
        }
    }
    out
}

/// Read /proc/meminfo `MemAvailable:` and return it in GB. Returns 0.0 on
/// any parse error (the caller treats that as "no budget known" and falls
/// back to a conservative default).
fn read_mem_available_gb() -> f64 {
    let Ok(s) = std::fs::read_to_string("/proc/meminfo") else { return 0.0; };
    for line in s.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            let kb: u64 = rest.trim().split_whitespace().next()
                .and_then(|t| t.parse().ok()).unwrap_or(0);
            return (kb as f64) / 1e6;  // kB → GB
        }
    }
    0.0
}

/// Estimate peak RAM (in GB) `solve_esc_work_unit` / `solve_pair_work_unit`
/// will allocate on top of whatever's already live.
///
/// Per primary, solve_esc allocates 5 dense Vecs of length n_states:
/// first_key:i16 (2) + dtw:i16 (2) + count:u32 (4) + resolved:bool (1) +
/// has_draw_for_q:bool (1) = 10 bytes/slot. Pair WUs solve the two primaries
/// sequentially; between calls only the returned (fk, dtw) = 4 bytes/slot
/// for the first primary remains live, so peak = max(10·n1, 4·n1 + 10·n2).
fn estimate_wu_ram_gb(wu: &morris_tablebase::work_unit::WorkUnit) -> f64 {
    let n0 = wu.primary[0].n_states() as f64;
    if wu.is_esc {
        10.0 * n0 / 1e9
    } else {
        let n1 = wu.primary[1].n_states() as f64;
        let peak_first = 10.0 * n0;
        let peak_second = 4.0 * n0 + 10.0 * n1;
        peak_first.max(peak_second) / 1e9
    }
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

    // Step 2 — decide which subspaces to load. Default = all 49. With
    // --quick, restrict to subspaces involved in WUs <= max_total + 2,
    // which avoids the ~10-30 min sweep through the huge V1 files
    // (typically wanted when running parallel to convert_to_v2 or for a
    // single-WU smoke test).
    let stats_total_cap = if cli.quick {
        cli.max_total.saturating_add(2)
    } else {
        18
    };
    let mut mapped_tables: HashMap<Subspace, MappedTable> = HashMap::new();
    let mut wtm_counts: HashMap<Subspace, StmCounts> = HashMap::new();
    let mut all_subs: Vec<Subspace> = Vec::new();
    for wu in &all_wus {
        if wu.total_pieces() > stats_total_cap {
            continue;
        }
        for &p in &wu.primary {
            if !all_subs.contains(&p) {
                all_subs.push(p);
            }
        }
    }
    if cli.quick {
        println!("\n[--quick] Loading Phase 1 stats only for subspaces with total <= {}.", stats_total_cap);
        println!("           Ranks shown will be partial — the global ordinal");
        println!("           ranking requires stats for all 49 subspaces.");
    } else {
        println!("\nLoading Phase 1 (mmap) + computing W/D/L stats for all subspaces...");
    }
    let pb = indicatif::ProgressBar::new(all_subs.len() as u64);
    pb.set_style(
        indicatif::ProgressStyle::with_template(
            "  [{elapsed_precise}] [{bar:40.cyan/blue}] {pos:>3}/{len:>3} {wide_msg}",
        )
        .unwrap()
        .progress_chars("=>-"),
    );
    for sub in &all_subs {
        let path = cli.phase1_dir.join(default_filename(*sub, variant));
        if !path.exists() {
            eprintln!("missing Phase 1 file for ({}, {}): {}",
                sub.w_board, sub.b_board, path.display());
            std::process::exit(1);
        }
        let file_size_gb = std::fs::metadata(&path)
            .map(|m| m.len() as f64 / 1e9)
            .unwrap_or(0.0);
        pb.set_message(format!("({}, {}) [{:.2} GB]", sub.w_board, sub.b_board, file_size_gb));
        let table = MappedTable::open(&path).expect("mmap Phase 1 table");
        let stats = count_stats_mapped(*sub, &table);
        wtm_counts.insert(*sub, stats.white_to_move);
        wtm_counts.insert(negate(*sub), stats.black_to_move);
        mapped_tables.insert(*sub, table);
        pb.inc(1);
    }
    pb.finish_with_message(format!("stats loaded for {} subspaces.", all_subs.len()));

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
    println!("{:>3} {:>3}  {:>7}  {:>5}{}", "w", "b", "val_s", "rank",
        if cli.quick { "  [* = stats unloaded, val/rank not meaningful]" } else { "" });
    println!("{}", "-".repeat(28));
    let loaded: std::collections::HashSet<Subspace> = wtm_counts.keys().copied().collect();
    let mut display: Vec<(Subspace, f64, i16, bool)> = ranks
        .iter()
        .map(|(s, &r)| {
            let is_loaded = loaded.contains(s) || loaded.contains(&negate(*s));
            (*s, *val_per_primary.get(s).unwrap_or(&0.5), r, is_loaded)
        })
        .collect();
    display.sort_by(|a, b| a.2.cmp(&b.2));
    for (s, v, r, is_loaded) in display {
        let marker = if cli.quick && !is_loaded { " *" } else { "" };
        println!("{:>3} {:>3}  {:>7.4}  {:>+5}{}",
            s.w_board, s.b_board, v, r, marker);
    }

    // Step 4 — solve each WU in topological order (filtered by --max-total
    // and the RAM budget). Resolve the budget once up front so the loop
    // doesn't re-read /proc/meminfo per WU (which could flap as pages free).
    let ram_budget_gb = cli.max_ram_gb.unwrap_or_else(|| {
        let avail = read_mem_available_gb();
        if avail <= 0.0 { 20.0 } else { (avail - 4.0).max(4.0) }
    });
    println!("\nRAM budget per WU: {:.1} GB (override with --max-ram-gb)", ram_budget_gb);

    let to_solve: Vec<&WorkUnit> = all_wus.iter()
        .filter(|wu| wu.total_pieces() <= cli.max_total)
        .collect();
    println!("=== Solving {} WUs (total_pieces <= {}) ===\n",
        to_solve.len(), cli.max_total);

    // Build a Tablebase from the mmap tables that the wave will use for
    // cross-subspace queries (Phase 1 verdicts + DTW). We rebuild it
    // here from the mapped_tables map because Tablebase takes ownership.
    let mut phase1_tb = Tablebase::new();
    for (_, table) in mapped_tables.drain() {
        phase1_tb.insert_mapped(table);
    }

    let total_t = std::time::Instant::now();
    let mut skipped_for_ram: Vec<String> = Vec::new();
    for (i, wu) in to_solve.iter().enumerate() {
        let wu_label = if wu.is_esc {
            format!("ESC ({}, {})", wu.primary[0].w_board, wu.primary[0].b_board)
        } else {
            format!("pair ({}, {}) + ({}, {})",
                wu.primary[0].w_board, wu.primary[0].b_board,
                wu.primary[1].w_board, wu.primary[1].b_board)
        };

        // Pre-check the RAM budget. The wave allocates 10 bytes/slot ×
        // n_states dense Vecs per primary; for big subspaces this dwarfs
        // physical RAM and triggers the OOM-killer mid-WU. Skipping here
        // lets compute_gevay finish all the smaller WUs cleanly.
        let need_gb = estimate_wu_ram_gb(wu);
        if need_gb > ram_budget_gb {
            println!("[{}/{}] {} — SKIP (estimated {:.1} GB > budget {:.1} GB)",
                i + 1, to_solve.len(), wu_label, need_gb, ram_budget_gb);
            skipped_for_ram.push(format!("{} ({:.1} GB)", wu_label, need_gb));
            continue;
        }

        let t_wu = std::time::Instant::now();
        println!("[{}/{}] {} — solving (est. {:.1} GB)...",
            i + 1, to_solve.len(), wu_label, need_gb);

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

    let solved = to_solve.len() - skipped_for_ram.len();
    println!("\n=== complete: {} WUs solved in {:.1}s ===",
        solved, total_t.elapsed().as_secs_f64());
    if !skipped_for_ram.is_empty() {
        println!("\nSkipped for RAM budget ({} WUs):", skipped_for_ram.len());
        for s in &skipped_for_ram {
            println!("  {}", s);
        }
        println!("\nRe-run with --max-ram-gb to raise the budget if you have more RAM,");
        println!("or wait until the wave is rewritten to stream by rank_w bucket.");
    }
}
