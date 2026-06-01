//! `cargo run --release --bin build_movement [-- MAX_TOTAL [OUTPUT_DIR]]`
//!
//! Solve all movement subspaces in DAG order up to a piece-count cap.
//! Default cap is 10 (covers 11 subspaces from (3,3) through (7,3)),
//! cumulative RAM ≈ 5 GB without symmetry reduction.
//!
//! The DAG order is by `w + b` ascending: smaller subspaces are solved
//! before larger ones so that capture transitions can be resolved at init.
//!
//! When `OUTPUT_DIR` is provided, each resolved subspace is persisted
//! to `{OUTPUT_DIR}/flying_w{w}_b{b}_wp0_bp0.bin`. Existing files are
//! loaded instead of recomputed (resume-from-disk).

use std::path::PathBuf;
use std::time::Instant;

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

use morris_tablebase::storage::{default_filename, load, save};
use morris_tablebase::subspace::{Subspace, Tablebase};
use morris_tablebase::wave::{solve_movement, Variant};

fn list_movement_subspaces(max_total: u8) -> Vec<Subspace> {
    let mut out = Vec::new();
    let cap = max_total.min(18);
    for total in 6..=cap {
        for w in 3..=9 {
            let b_signed = total as i32 - w as i32;
            if (3..=9).contains(&b_signed) {
                out.push(Subspace::movement(w, b_signed as u8));
            }
        }
    }
    out
}

fn main() {
    let max_total: u8 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let output_dir: Option<PathBuf> = std::env::args().nth(2).map(PathBuf::from);

    let dag = list_movement_subspaces(max_total);
    eprintln!(
        "=== build_movement: {} subspaces, max_total={} ===",
        dag.len(),
        max_total
    );
    if let Some(d) = &output_dir {
        eprintln!("Output dir   : {} (resume-from-disk enabled)", d.display());
        if let Err(e) = std::fs::create_dir_all(d) {
            eprintln!("Failed to create output dir: {e}");
            std::process::exit(1);
        }
    }
    eprintln!();
    eprintln!("{:>3} {:>3}  {:>14}  {:>14}  {:>10}  {:>10}  {:>14}  {:>4}  {:>7}  {}",
        "w", "b", "n_states", "WIN", "LOSS", "DRAW", "win%", "DTW", "time(s)", "src");
    eprintln!("{}", "-".repeat(112));

    let multi = MultiProgress::new();
    let global_pb = multi.add(ProgressBar::new(dag.len() as u64));
    global_pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>3}/{len:3} subspaces  ETA {eta}"
        ).unwrap()
    );
    let sub_pb = multi.add(ProgressBar::new(1));
    sub_pb.set_style(
        ProgressStyle::with_template(
            "  {prefix:.bold} [{bar:30.green/white}] {pos:>11}/{len:11} {msg}"
        ).unwrap()
    );

    let mut tb = Tablebase::new();
    let t_global = Instant::now();
    let mut cumulative_states: u64 = 0;
    let mut cumulative_bytes: u64 = 0;

    for sub in dag {
        let t0 = Instant::now();
        let on_disk = output_dir.as_ref().map(|d| d.join(default_filename(sub, Variant::Flying)));
        sub_pb.set_prefix(format!("({},{})", sub.w_board, sub.b_board));
        sub_pb.set_position(0);

        let (table, source) = match on_disk.as_ref() {
            Some(p) if p.exists() => {
                let (t, _v) = load(p).expect("load existing subspace");
                sub_pb.set_length(t.verdict.len() as u64);
                sub_pb.set_position(t.verdict.len() as u64);
                sub_pb.set_message("loaded from disk");
                (t, "disk")
            }
            _ => {
                let (t, _stats) = solve_movement(sub, Variant::Flying, &tb, Some(&sub_pb));
                if let Some(p) = on_disk.as_ref() {
                    save(&t, Variant::Flying, p).expect("save subspace");
                }
                (t, "solve")
            }
        };
        let dt = t0.elapsed().as_secs_f64();

        // Compute stats from the verdict array (same whether loaded or solved).
        let mut win = 0u32;
        let mut loss = 0u32;
        let mut draw = 0u32;
        let mut max_dtw = 0u16;
        for (i, &v) in table.verdict.iter().enumerate() {
            match v {
                morris_tablebase::wave::WIN => { win += 1; if table.dtw[i] > max_dtw { max_dtw = table.dtw[i]; } }
                morris_tablebase::wave::LOSS => { loss += 1; if table.dtw[i] > max_dtw { max_dtw = table.dtw[i]; } }
                morris_tablebase::wave::DRAW => draw += 1,
                _ => {}
            }
        }
        let n_states = table.verdict.len() as u32;
        let total = n_states / 2;
        let win_pct = if total > 0 { win as f64 / 2.0 / total as f64 * 100.0 } else { 0.0 };
        multi.suspend(|| {
            println!(
                "{:>3} {:>3}  {:>14}  {:>14}  {:>10}  {:>10}  {:>13.2}%  {:>4}  {:>7.2}  {}",
                sub.w_board, sub.b_board,
                n_states, win, loss, draw,
                win_pct, max_dtw, dt, source
            );
        });
        cumulative_states += n_states as u64;
        cumulative_bytes += n_states as u64 * 3;
        tb.insert(table);
        global_pb.inc(1);
    }
    sub_pb.finish_and_clear();
    global_pb.finish_with_message("all subspaces done");

    let total_time = t_global.elapsed().as_secs_f64();
    println!("\nTotal time   : {:.2}s", total_time);
    println!("Total states : {}", cumulative_states);
    println!(
        "Resident table memory: ~{:.2} GB ({} bytes/state for verdict+dtw, count freed after wave)",
        cumulative_bytes as f64 / 1e9,
        3
    );
}
