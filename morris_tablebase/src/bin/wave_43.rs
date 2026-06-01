//! `cargo run --release --bin wave_43` — Solve (4,3,0,0) using (3,3,0,0).
//!
//! First cross-subspace exercise: when BLACK (3 pieces, can fly) captures
//! a WHITE piece at (4,3), the resulting state is in (3,3). WHITE (4 pieces,
//! cannot fly) captures only via adjacency-driven mill moves, sending BLACK
//! to 2 pieces (terminal LOSS for BLACK).
//!
//! No published Gasser fixture for (4,3) is checked here — we report the
//! verdict distribution and confirm that the wave converges without
//! `expect("smaller subspace must be resolved")` panicking.

use std::time::Instant;

use morris_tablebase::subspace::{Subspace, Tablebase};
use morris_tablebase::wave::{
    solve_movement, Variant, WaveStats, LOSS, STM_BLACK, STM_WHITE, WIN,
};

fn main() {
    println!("=== Stage 1: solve (3,3,0,0) ===");
    let sub33 = Subspace::movement(3, 3);
    let mut tb = Tablebase::new();
    let t0 = Instant::now();
    let (table33, stats33) = solve_movement(sub33, Variant::Flying, &tb, None);
    let t33 = t0.elapsed().as_secs_f64();
    println!("  WIN={} LOSS={} DRAW={} max_dtw={} in {:.2}s",
        stats33.win, stats33.loss, stats33.draw, stats33.max_dtw, t33);
    tb.insert(table33);

    println!("\n=== Stage 2: solve (4,3,0,0) using (3,3) cross-subspace ===");
    let sub43 = Subspace::movement(4, 3);
    let t1 = Instant::now();
    let (table43, stats43) = solve_movement(sub43, Variant::Flying, &tb, None);
    let t43 = t1.elapsed().as_secs_f64();
    let WaveStats { n_states, win, loss, draw, max_dtw } = stats43;
    println!("  Aggregate: WIN={} LOSS={} DRAW={} max_dtw={} in {:.2}s",
        win, loss, draw, max_dtw, t43);

    // Break down per-STM via orbit-weighted accounting (canonical positions
    // are iterated; each contributes orbit_size to its STM's bucket).
    let mut by_stm = [[0u64; 4]; 2];
    sub43.enumerate_positions(|wbb, bbb| {
        let osize = morris_tablebase::symmetry::orbit_size(wbb, bbb) as u64;
        for stm in [1u8, 2u8] {
            let idx = sub43.state_index_canonical(wbb, bbb, stm) as usize;
            let v = table43.verdict[idx] as usize;
            by_stm[(stm - 1) as usize][v] += osize;
        }
    });
    let total = n_states / 2;
    println!("\n=== Per-STM (4,3,0,0) ===");
    for stm in [STM_WHITE, STM_BLACK] {
        let s = stm as usize - 1;
        let w = by_stm[s][WIN as usize];
        let l = by_stm[s][LOSS as usize];
        let d = by_stm[s][3]; // DRAW = 3
        let total_f = total as f64;
        let name = if stm == STM_WHITE { "WHITE (4 pieces, no fly)" } else { "BLACK (3 pieces, fly)" };
        println!("\nSTM={}: total={}", name, total);
        println!("  WIN  : {:>10}  ({:5.2}%)", w, w as f64 / total_f * 100.0);
        println!("  LOSS : {:>10}  ({:5.2}%)", l, l as f64 / total_f * 100.0);
        println!("  DRAW : {:>10}  ({:5.2}%)", d, d as f64 / total_f * 100.0);
    }

    println!("\nTotal time (both subspaces): {:.2}s", t0.elapsed().as_secs_f64());
}
