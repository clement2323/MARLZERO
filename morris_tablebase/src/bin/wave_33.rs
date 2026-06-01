//! `cargo run --release --bin wave_33` — Rust port of the (3,3) Python spike.
//!
//! Cross-checks against the Python fixture documented in
//! [docs/decisions/002-phase1-gasser-tables.md](../../../docs/decisions/002-phase1-gasser-tables.md):
//!     STM=WHITE: WIN=2,232,160  LOSS=455,648  DRAW=4,112
//!     STM=BLACK: same
//!     Total instant-WIN at DTW=1: 1,056,096 per STM
//!     Invariant A failures: 0

use std::time::Instant;

use morris_tablebase::wave::{run, Result33};

fn main() {
    println!("=== Gasser wave: subspace (3,3,0,0) with flying (Rust port) ===");
    let t0 = Instant::now();
    let Result33 {
        n_states, win, loss, draw, instant_win_dtw1, max_dtw, invariant_a_failures,
    } = run();
    let dt = t0.elapsed().as_secs_f64();

    let total_pos = n_states / 2;
    let pct_per_stm = |c: u32| -> f64 { c as f64 / total_pos as f64 * 100.0 };

    // WIN/LOSS/DRAW are summed across both STMs by `run`; print per-STM
    // by halving (both STMs have identical counts by Invariant A — the
    // check below confirms or refutes that).
    println!("\nStates: {} ({} positions × 2 STMs)", n_states, total_pos);
    println!("Wave runtime: {:.2}s\n", dt);

    println!("Per-STM verdict counts (totals divided by 2):");
    println!("  WIN  : {:>10}  ({:5.2}%)", win / 2, pct_per_stm(win / 2));
    println!("  LOSS : {:>10}  ({:5.2}%)", loss / 2, pct_per_stm(loss / 2));
    println!("  DRAW : {:>10}  ({:5.2}%)", draw / 2, pct_per_stm(draw / 2));

    println!("\nInstant-WIN at DTW=1: {} ({} per STM)", instant_win_dtw1, instant_win_dtw1 / 2);
    println!("Max DTW observed     : {}", max_dtw);

    println!("\nInvariant A (swap colors + swap STM → same verdict):");
    if invariant_a_failures == 0 {
        println!("  PASS ({} state pairs verified)", n_states);
    } else {
        println!("  FAIL: {} mismatches out of {}", invariant_a_failures, n_states);
    }

    // Cross-check against Python fixture.
    let expected_win = 2_232_160 * 2;
    let expected_loss = 455_648 * 2;
    let expected_draw = 4_112 * 2;
    let expected_inst = 1_056_096 * 2;
    let ok = win == expected_win
        && loss == expected_loss
        && draw == expected_draw
        && instant_win_dtw1 == expected_inst
        && invariant_a_failures == 0;
    println!("\nPython fixture cross-check:");
    if ok {
        println!("  PASS — all counts match scripts/spike_gasser_33.py exactly");
        std::process::exit(0);
    } else {
        println!("  FAIL");
        println!("    expected: win={}, loss={}, draw={}, instant={}",
            expected_win, expected_loss, expected_draw, expected_inst);
        println!("    got     : win={}, loss={}, draw={}, instant={}",
            win, loss, draw, instant_win_dtw1);
        std::process::exit(1);
    }
}
