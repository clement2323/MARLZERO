//! `cargo run --release --bin wave_33` — Generic solver applied to (3,3,0,0).
//!
//! Cross-checks against the Python fixture documented in
//! [docs/decisions/002-phase1-gasser-tables.md](../../../docs/decisions/002-phase1-gasser-tables.md):
//!     STM=WHITE: WIN=2,232,160  LOSS=455,648  DRAW=4,112
//!     STM=BLACK: same
//!
//! Now driven through [morris_tablebase::wave::solve_movement] with an
//! empty [Tablebase] — (3,3,0,0) is self-contained, no smaller subspace
//! is required.

use std::time::Instant;

use morris_tablebase::subspace::{Subspace, Tablebase};
use morris_tablebase::wave::{solve_movement, Variant, WaveStats};

fn main() {
    println!("=== Gasser wave: subspace (3,3,0,0) with flying (Rust, generic) ===");
    let sub = Subspace::movement(3, 3);
    let tb = Tablebase::new();
    let t0 = Instant::now();
    let (_table, stats) = solve_movement(sub, Variant::Flying, &tb, None);
    let dt = t0.elapsed().as_secs_f64();
    let WaveStats { n_states, win, loss, draw, max_dtw } = stats;

    let total_pos = n_states / 2;
    let pct_per_stm = |c: u32| c as f64 / total_pos as f64 * 100.0;

    println!("\nStates: {} ({} positions × 2 STMs)", n_states, total_pos);
    println!("Wave runtime: {:.2}s\n", dt);

    println!("Per-STM verdict counts (totals divided by 2):");
    println!("  WIN  : {:>10}  ({:5.2}%)", win / 2, pct_per_stm(win / 2));
    println!("  LOSS : {:>10}  ({:5.2}%)", loss / 2, pct_per_stm(loss / 2));
    println!("  DRAW : {:>10}  ({:5.2}%)", draw / 2, pct_per_stm(draw / 2));
    println!("\nMax DTW observed: {}", max_dtw);

    // Cross-check against Python fixture (sums across both STMs).
    let expected_win = 2_232_160 * 2;
    let expected_loss = 455_648 * 2;
    let expected_draw = 4_112 * 2;
    let ok = win == expected_win && loss == expected_loss && draw == expected_draw;
    println!("\nPython fixture cross-check:");
    if ok {
        println!("  PASS — counts match scripts/spike_gasser_33.py exactly");
        std::process::exit(0);
    } else {
        println!("  FAIL");
        println!("    expected: win={}, loss={}, draw={}", expected_win, expected_loss, expected_draw);
        println!("    got     : win={}, loss={}, draw={}", win, loss, draw);
        std::process::exit(1);
    }
}
