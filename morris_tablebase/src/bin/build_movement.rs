//! `cargo run --release --bin build_movement [-- MAX_TOTAL]`
//!
//! Solve all movement subspaces in DAG order up to a piece-count cap.
//! Default cap is 10 (covers 11 subspaces from (3,3) through (7,3)),
//! cumulative RAM ≈ 5 GB without symmetry reduction.
//!
//! The DAG order is by `w + b` ascending: smaller subspaces are solved
//! before larger ones so that capture transitions can be resolved at init.

use std::time::Instant;

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

    let dag = list_movement_subspaces(max_total);
    println!(
        "=== build_movement: {} subspaces, max_total={} ===\n",
        dag.len(),
        max_total
    );
    println!("{:>3} {:>3}  {:>14}  {:>14}  {:>10}  {:>10}  {:>14}  {:>4}  {:>7}",
        "w", "b", "n_states", "WIN", "LOSS", "DRAW", "win%", "DTW", "time(s)");
    println!("{}", "-".repeat(106));

    let mut tb = Tablebase::new();
    let t_global = Instant::now();
    let mut cumulative_states: u64 = 0;
    let mut cumulative_bytes: u64 = 0;

    for sub in dag {
        let t0 = Instant::now();
        let (table, stats) = solve_movement(sub, Variant::Flying, &tb);
        let dt = t0.elapsed().as_secs_f64();
        let total = stats.n_states / 2;
        let win_pct = if total > 0 { stats.win as f64 / 2.0 / total as f64 * 100.0 } else { 0.0 };
        println!(
            "{:>3} {:>3}  {:>14}  {:>14}  {:>10}  {:>10}  {:>13.2}%  {:>4}  {:>7.2}",
            sub.w_board, sub.b_board,
            stats.n_states, stats.win, stats.loss, stats.draw,
            win_pct, stats.max_dtw, dt
        );
        cumulative_states += stats.n_states as u64;
        cumulative_bytes += stats.n_states as u64 * 3; // verdict u8 + dtw u16
        tb.insert(table);
    }

    let total_time = t_global.elapsed().as_secs_f64();
    println!("\nTotal time   : {:.2}s", total_time);
    println!("Total states : {}", cumulative_states);
    println!(
        "Resident table memory: ~{:.2} GB ({} bytes/state for verdict+dtw, count freed after wave)",
        cumulative_bytes as f64 / 1e9,
        3
    );
}
