//! One-shot pre-builder for the CanonicalIndexer cache on disk.
//!
//! Goal: turn what is currently a per-process, per-subspace 60-110-second
//! rebuild (×N workers × M subspaces) into a one-time ~10-15-minute job
//! whose output is mmap'd by every subsequent `play_tb --serve` process.
//! Once this binary has run, self-play workers can scale back to 8+
//! without OOM-killing the box, because indexer pages are shared via the
//! OS page cache across processes instead of duplicated in private heaps.
//!
//! Usage:
//!     cargo run --release --bin build_indexer_cache -- \
//!         data/tablebase/gevay/
//!
//! Output:
//!     data/tablebase/gevay/.indexers/idx_w{w}_b{b}.bin   (one per (w,b))
//!
//! Skips subspaces whose Gévay file isn't present in the gevay dir (so a
//! partial Phase 2 run is handled gracefully). Skips files that already
//! exist on disk — re-running is idempotent and cheap.
//!
//! Total cache size on disk: ~60 GB (dominated by the (8,8) ESC at
//! ~4.7 GB). Keep at least 100 GB free in `<gevay-dir>`.

use std::path::PathBuf;

use morris_tablebase::gevay::canonical_indexer::{indexer_filename, CanonicalIndexer};
use morris_tablebase::storage::gevay_filename;
use morris_tablebase::subspace::Subspace;
use morris_tablebase::wave::Variant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: build_indexer_cache <GEVAY_DIR> [--out <CACHE_DIR>]");
        eprintln!("  GEVAY_DIR  : directory containing gevay_flying_w{{w}}_b{{b}}_wp0_bp0.bin files");
        eprintln!("  --out DIR  : output cache directory (default: <GEVAY_DIR>/.indexers/)");
        std::process::exit(1);
    }
    let gevay_dir = PathBuf::from(&args[1]);
    let mut out_dir: Option<PathBuf> = None;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--out" if i + 1 < args.len() => {
                out_dir = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            _ => { i += 1; }
        }
    }
    let cache_dir = out_dir.unwrap_or_else(|| gevay_dir.join(".indexers"));
    std::fs::create_dir_all(&cache_dir).expect("create cache dir");

    println!("=== build_indexer_cache ===");
    println!("gevay dir   : {}", gevay_dir.display());
    println!("cache dir   : {}", cache_dir.display());
    println!();

    // Iterate (w, b) ∈ [3..=9]². For each that has a Gévay file, ensure
    // an indexer cache exists in `cache_dir`. We do it sequentially so
    // that a single big subspace doesn't peak RAM by stacking on top of
    // a previous one — each build releases its memory before the next
    // starts (the indexer Vecs are dropped after save_to_path).
    let mut built = 0;
    let mut skipped = 0;
    let mut missing = 0;
    let t_total = std::time::Instant::now();

    for w in 3..=9u8 {
        for b in 3..=9u8 {
            let sub = Subspace::movement(w, b);
            let gevay_path = gevay_dir.join(gevay_filename(sub, Variant::Flying));
            if !gevay_path.exists() {
                missing += 1;
                continue;
            }
            let cache_path = cache_dir.join(indexer_filename(sub));
            if cache_path.exists() {
                // Quick validity check: try mmap-loading. If it works, skip.
                match CanonicalIndexer::load_mmap(sub, &cache_path) {
                    Ok(_) => {
                        println!("[skip] ({},{}) cache valid at {}", w, b, cache_path.display());
                        skipped += 1;
                        continue;
                    }
                    Err(e) => {
                        println!("[redo] ({},{}) cache invalid ({}). Rebuilding.", w, b, e);
                        let _ = std::fs::remove_file(&cache_path);
                    }
                }
            }

            println!("[build] ({},{}) starting...", w, b);
            let t = std::time::Instant::now();
            let idx = CanonicalIndexer::build(sub);
            let n = idx.n_canonical_entries();
            let build_secs = t.elapsed().as_secs_f64();

            let t_save = std::time::Instant::now();
            idx.save_to_path(&cache_path).expect("save indexer");
            let save_secs = t_save.elapsed().as_secs_f64();

            let file_mb = (cache_path.metadata().map(|m| m.len()).unwrap_or(0) as f64) / 1e6;
            println!(
                "[done]  ({},{}) {} canonical entries, build {:.1}s, save {:.1}s, file {:.1} MB",
                w, b, n, build_secs, save_secs, file_mb,
            );
            built += 1;
            // idx (and its 4 GB Vecs) is dropped here before the next iter.
        }
    }

    println!();
    println!(
        "=== complete: built {}, skipped {}, missing-gevay {} in {:.1}s ===",
        built, skipped, missing, t_total.elapsed().as_secs_f64(),
    );
}
