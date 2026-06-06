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
use morris_tablebase::storage::{default_filename, gevay_filename};
use morris_tablebase::subspace::{MappedTable, Subspace, Tablebase};
use morris_tablebase::symmetry::orbit_size;
use morris_tablebase::wave::Variant;
use morris_tablebase::work_unit::{list_movement_work_units, negate, WorkUnit};
use rayon::prelude::*;
use std::sync::Arc;
use std::sync::Mutex;

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
    // first_key is now canonical-only (length = 2 × n_canonical_entries).
    // We rebuild a CanonicalIndexer to convert (cw, cb, stm) → canonical_idx.
    // The build is a few ms for small subspaces, seconds for big ones — cheap
    // relative to the wave so we just do it inline rather than threading the
    // indexer through the call chain.
    let indexer = morris_tablebase::gevay::canonical_indexer::CanonicalIndexer::build(sub);
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
            let idx = indexer.canonical_index(cw, cb, stm) as usize;
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

/// Tiny binary cache for the stats sweep — keyed by Phase 1 .bin mtimes so
/// we recompute when source files change. Size for 49 subspaces ≈ 2.5 KB.
const STATS_CACHE_MAGIC: &[u8; 4] = b"GSTC";
const STATS_CACHE_VERSION: u16 = 1;

fn stats_cache_path(phase1_dir: &PathBuf, variant: Variant) -> PathBuf {
    let suffix = match variant {
        Variant::Flying => "flying",
        Variant::NoFlying => "no_flying",
    };
    phase1_dir.join(format!(".stats_cache_{}.bin", suffix))
}

/// True when the cache file exists AND is newer than every Phase 1 source
/// file in `phase1_dir`. Returns false on any IO error (forces a recompute).
fn stats_cache_is_fresh(cache_path: &PathBuf, all_subs: &[Subspace], phase1_dir: &PathBuf, variant: Variant) -> bool {
    let Ok(cache_meta) = std::fs::metadata(cache_path) else { return false; };
    let Ok(cache_mtime) = cache_meta.modified() else { return false; };
    for sub in all_subs {
        let p = phase1_dir.join(default_filename(*sub, variant));
        let Ok(meta) = std::fs::metadata(&p) else { return false; };
        let Ok(mtime) = meta.modified() else { return false; };
        if mtime > cache_mtime { return false; }
    }
    true
}

/// Write the wtm_counts map to disk in the simple binary layout described
/// above. Atomic via tmp+rename so a crash mid-write doesn't leave a stub.
fn write_stats_cache(
    cache_path: &PathBuf,
    variant: Variant,
    wtm_counts: &HashMap<Subspace, StmCounts>,
    all_subs: &[Subspace],
) -> std::io::Result<()> {
    use std::io::Write;
    let tmp = cache_path.with_extension("bin.tmp");
    let mut buf: Vec<u8> = Vec::with_capacity(16 + all_subs.len() * 50);
    buf.extend_from_slice(STATS_CACHE_MAGIC);
    buf.extend_from_slice(&STATS_CACHE_VERSION.to_le_bytes());
    let variant_byte: u8 = match variant {
        Variant::Flying => 0,
        Variant::NoFlying => 1,
    };
    buf.push(variant_byte);
    buf.push(0);  // padding
    buf.extend_from_slice(&(all_subs.len() as u32).to_le_bytes());
    buf.extend_from_slice(&[0u8; 4]);  // padding to 16-byte header
    for sub in all_subs {
        let wtm = wtm_counts.get(sub).copied().unwrap_or_default();
        let btm = wtm_counts.get(&negate(*sub)).copied().unwrap_or_default();
        buf.push(sub.w_board);
        buf.push(sub.b_board);
        for v in [wtm.wins, wtm.losses, wtm.draws, btm.wins, btm.losses, btm.draws] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(&buf)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, cache_path)?;
    Ok(())
}

/// Read the cache and populate wtm_counts. Returns Err if the file is
/// missing, magic mismatches, or version is unknown — caller falls back
/// to a fresh sweep.
fn read_stats_cache(cache_path: &PathBuf, variant: Variant) -> std::io::Result<HashMap<Subspace, StmCounts>> {
    let bytes = std::fs::read(cache_path)?;
    if bytes.len() < 16 || &bytes[0..4] != STATS_CACHE_MAGIC {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "bad magic"));
    }
    let version = u16::from_le_bytes([bytes[4], bytes[5]]);
    if version != STATS_CACHE_VERSION {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData,
            format!("unsupported cache version {}", version)));
    }
    let cached_variant: u8 = bytes[6];
    let want_variant: u8 = match variant {
        Variant::Flying => 0,
        Variant::NoFlying => 1,
    };
    if cached_variant != want_variant {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "variant mismatch"));
    }
    let count = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let mut out: HashMap<Subspace, StmCounts> = HashMap::with_capacity(count * 2);
    let mut off = 16;
    for _ in 0..count {
        if off + 50 > bytes.len() {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "truncated record"));
        }
        let w = bytes[off];
        let b = bytes[off + 1];
        let read_u64 = |start: usize| -> u64 {
            u64::from_le_bytes(bytes[start..start + 8].try_into().unwrap())
        };
        let wtm = StmCounts {
            wins: read_u64(off + 2),
            losses: read_u64(off + 10),
            draws: read_u64(off + 18),
        };
        let btm = StmCounts {
            wins: read_u64(off + 26),
            losses: read_u64(off + 34),
            draws: read_u64(off + 42),
        };
        let sub = Subspace::movement(w, b);
        out.insert(sub, wtm);
        out.insert(negate(sub), btm);
        off += 50;
    }
    Ok(out)
}

/// Estimate peak RAM (in GB) `solve_esc_work_unit` / `solve_pair_work_unit`
/// will allocate on top of whatever's already live.
///
/// The wave allocates 5 arrays sized to `n_states_canonical = 2 ×
/// n_canonical_entries ≈ n_states / 8` (the D4 orbit reduction): first_key
/// i16 (2) + dtw i16 (2) + count u16 (2) + resolved bitvec (0.125) +
/// has_draw_for_q bitvec (0.125) = 6.25 B/slot canonical. Plus the
/// `CanonicalIndexer` carries a `canonical_rank_b` Vec<Vec<u32>> of
/// total size ≈ 4 B × n_canonical_entries = 0.25 B × n_states. Combined
/// these come to ~1 byte per DENSE n_states slot in steady state.
///
/// Pair WUs solve the two primaries sequentially; between calls only the
/// returned `(first_key, dtw)` arrays for the first primary (4 B/slot ×
/// n_states_canonical = 0.5 B × n_states) remain live, so:
///   peak_pair = max(1.0·n0, 0.5·n0 + 1.0·n1).
const GEVAY_BYTES_PER_SLOT: f64 = 1.0;
const GEVAY_RETURNED_BYTES_PER_SLOT: f64 = 0.5;

fn estimate_wu_ram_gb(wu: &morris_tablebase::work_unit::WorkUnit) -> f64 {
    let n0 = wu.primary[0].n_states() as f64;
    if wu.is_esc {
        GEVAY_BYTES_PER_SLOT * n0 / 1e9
    } else {
        let n1 = wu.primary[1].n_states() as f64;
        let peak_first = GEVAY_BYTES_PER_SLOT * n0;
        let peak_second = GEVAY_RETURNED_BYTES_PER_SLOT * n0 + GEVAY_BYTES_PER_SLOT * n1;
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
    }

    // Stats cache fast-path: when the cache file is present and newer than
    // every Phase 1 .bin we need, we skip the random-read sweep entirely
    // and load W/L/D counts in milliseconds. The cache is keyed by the
    // FULL list of 49 subspaces — even in --quick mode we either trust the
    // full cache or recompute the partial set fresh (no hybrid).
    let cache_path = stats_cache_path(&cli.phase1_dir, variant);
    let t_stats = std::time::Instant::now();
    let mut cache_hit = false;
    if !cli.quick && stats_cache_is_fresh(&cache_path, &all_subs, &cli.phase1_dir, variant) {
        match read_stats_cache(&cache_path, variant) {
            Ok(map) => {
                wtm_counts = map;
                cache_hit = true;
                let age = std::fs::metadata(&cache_path)
                    .and_then(|m| m.modified())
                    .and_then(|t| t.elapsed().map_err(|e| std::io::Error::other(e.to_string())))
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                println!("\n✓ stats cache hit at {} ({} subspaces, {}s old). Skipping sweep.",
                    cache_path.display(), all_subs.len(), age);
            }
            Err(e) => {
                println!("\nStats cache present but unreadable ({}). Recomputing.", e);
            }
        }
    } else if !cli.quick {
        println!("\nStats cache missing or stale. Computing fresh sweep over {} Phase 1 files...",
            all_subs.len());
    }

    if !cache_hit {
        // Pre-mmap (cheap; just opens fd + maps memory, no IO yet).
        for sub in &all_subs {
            let path = cli.phase1_dir.join(default_filename(*sub, variant));
            if !path.exists() {
                eprintln!("missing Phase 1 file for ({}, {}): {}",
                    sub.w_board, sub.b_board, path.display());
                std::process::exit(1);
            }
            let table = MappedTable::open(&path).expect("mmap Phase 1 table");
            mapped_tables.insert(*sub, table);
        }

        // Parallel count_stats_mapped sweep: each thread takes a mmap'd
        // table and iterates orbit positions. The random reads happen
        // concurrently across files, saturating the NVMe queue depth far
        // better than the sequential loop did (~5× empirically).
        let pb = Arc::new(indicatif::ProgressBar::new(all_subs.len() as u64));
        pb.set_style(
            indicatif::ProgressStyle::with_template(
                "  [{elapsed_precise}] [{bar:40.cyan/blue}] {pos:>3}/{len:>3} {wide_msg}",
            )
            .unwrap()
            .progress_chars("=>-"),
        );
        pb.enable_steady_tick(std::time::Duration::from_millis(500));

        let counts_mutex = Arc::new(Mutex::new(HashMap::<Subspace, StmCounts>::new()));
        all_subs.par_iter().for_each(|sub| {
            let table = mapped_tables.get(sub).expect("pre-mmap'd");
            let file_size_gb = std::fs::metadata(cli.phase1_dir.join(default_filename(*sub, variant)))
                .map(|m| m.len() as f64 / 1e9).unwrap_or(0.0);
            pb.set_message(format!("({}, {}) [{:.2} GB]", sub.w_board, sub.b_board, file_size_gb));
            let stats = count_stats_mapped(*sub, table);
            let mut guard = counts_mutex.lock().unwrap();
            guard.insert(*sub, stats.white_to_move);
            guard.insert(negate(*sub), stats.black_to_move);
            drop(guard);
            pb.inc(1);
        });
        wtm_counts = Arc::try_unwrap(counts_mutex).unwrap().into_inner().unwrap();
        pb.finish_with_message(format!("stats loaded for {} subspaces in {:.1}s.",
            all_subs.len(), t_stats.elapsed().as_secs_f64()));

        // Persist for next run. Failures are non-fatal — just log.
        if !cli.quick {
            match write_stats_cache(&cache_path, variant, &wtm_counts, &all_subs) {
                Ok(()) => println!("✓ wrote stats cache to {}", cache_path.display()),
                Err(e) => println!("warning: failed to write stats cache: {}", e),
            }
        }
    } else {
        // Cache hit: we still need mmap'd tables for the wave's
        // cross-subspace queries below. Open them now.
        for sub in &all_subs {
            let path = cli.phase1_dir.join(default_filename(*sub, variant));
            let table = MappedTable::open(&path).expect("mmap Phase 1 table");
            mapped_tables.insert(*sub, table);
        }
    }

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

    // Overall WU progress bar — shows position N/M + ETA on TTY. When
    // stdout is piped to a log file the bar suppresses itself silently;
    // the per-WU details below use plain println! so they're captured
    // either way (log file or live terminal).
    let pb_wu = indicatif::ProgressBar::new(to_solve.len() as u64);
    pb_wu.set_style(
        indicatif::ProgressStyle::with_template(
            "  [{elapsed_precise} / ETA {eta_precise}] [{bar:40.green/cyan}] {pos:>2}/{len:>2} {wide_msg}",
        )
        .unwrap()
        .progress_chars("=>-"),
    );
    pb_wu.enable_steady_tick(std::time::Duration::from_millis(500));

    for (i, wu) in to_solve.iter().enumerate() {
        let wu_label = if wu.is_esc {
            format!("ESC ({}, {})", wu.primary[0].w_board, wu.primary[0].b_board)
        } else {
            format!("pair ({}, {}) + ({}, {})",
                wu.primary[0].w_board, wu.primary[0].b_board,
                wu.primary[1].w_board, wu.primary[1].b_board)
        };

        // Skip the WU entirely when --save-to is set and every primary's
        // output file already exists. Lets the user rerun compute_gevay
        // after a partial first pass without redoing what's already saved.
        if let Some(dir) = &cli.save_to {
            let all_saved = wu.primary.iter().all(|sub| {
                dir.join(gevay_filename(*sub, variant)).exists()
            });
            if all_saved {
                println!("[{}/{}] {} — already saved, skipping",
                    i + 1, to_solve.len(), wu_label);
                pb_wu.inc(1);
                continue;
            }
        }

        // Pre-check the RAM budget. The wave allocates ~1 byte per DENSE
        // n_states slot (5.25 B/slot × n_canonical = 5.25 × n_states / 8)
        // plus indexer overhead. Skipping here lets compute_gevay finish
        // all the smaller WUs cleanly.
        let need_gb = estimate_wu_ram_gb(wu);
        if need_gb > ram_budget_gb {
            println!("[{}/{}] {} — SKIP (estimated {:.1} GB > budget {:.1} GB)",
                i + 1, to_solve.len(), wu_label, need_gb, ram_budget_gb);
            skipped_for_ram.push(format!("{} ({:.1} GB)", wu_label, need_gb));
            pb_wu.inc(1);
            continue;
        }

        let t_wu = std::time::Instant::now();
        pb_wu.set_message(format!("{} (est. {:.1} GB)", wu_label, need_gb));
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
                morris_tablebase::storage::save_gevay_canonical(
                    sub, variant, first_key, dtw, &path,
                ).expect("save_gevay_canonical");
                println!("    saved to {} ({:.1}s)",
                    path.display(), t_save.elapsed().as_secs_f64());
            }
        }
        pb_wu.inc(1);
    }
    pb_wu.finish_and_clear();

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
