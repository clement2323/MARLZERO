//! Verify that the Phase 1 wave produces color-swap-symmetric outputs.
//!
//! For every canonical position p = (wbb, bbb, stm) in subspace S = (w, b),
//! the color-swap mirror p' = (bbb, wbb, swap(stm)) lives in S' = (b, w)
//! and must have the same game-theoretic value. This is a foundational
//! invariant: if it doesn't hold, then deleting the (w, b) file in favor of
//! the (b, w) file during the V2 migration would lose information.
//!
//! These tests are #[ignore]'d by default because they're heavy
//! (full wave solves of (3,4) and (4,3) take a few seconds in debug). Run
//! with `cargo test --release wave_color_swap -- --ignored`.

use morris_tablebase::subspace::{Subspace, Tablebase};
use morris_tablebase::symmetry::canonicalize;
use morris_tablebase::wave::{solve_movement, Variant};

#[test]
#[ignore]
fn wave_respects_color_swap_34_vs_43() {
    let sub33 = Subspace::movement(3, 3);
    let (t33, _) = solve_movement(sub33, Variant::Flying, &Tablebase::new(), None);
    let mut tb = Tablebase::new();
    tb.insert(t33);

    let sub43 = Subspace::movement(4, 3);
    let (t43, _) = solve_movement(sub43, Variant::Flying, &tb, None);
    let sub34 = Subspace::movement(3, 4);
    let (t34, _) = solve_movement(sub34, Variant::Flying, &tb, None);

    let mut total = 0u64;
    let mut v_mismatch = 0u64;
    let mut d_mismatch = 0u64;
    let mut first_d: Option<(u32, u32, u8, u8, u16, u8, u16)> = None;

    sub34.enumerate_positions(|wbb, bbb| {
        for stm in [1u8, 2u8] {
            let idx34 = sub34.state_index_canonical(wbb, bbb, stm) as usize;
            let v34 = t34.verdict[idx34];
            let d34 = t34.dtw[idx34];

            let (cw, cb) = canonicalize(bbb, wbb);
            let sstm = 3 - stm;
            let idx43 = sub43.state_index_canonical(cw, cb, sstm) as usize;
            let v43 = t43.verdict[idx43];
            let d43 = t43.dtw[idx43];

            total += 1;
            if v34 != v43 { v_mismatch += 1; }
            if d34 != d43 {
                d_mismatch += 1;
                if first_d.is_none() {
                    first_d = Some((wbb, bbb, stm, v34, d34, v43, d43));
                }
            }
        }
    });
    println!("\n=== wave_respects_color_swap_34_vs_43 ===");
    println!("Total positions checked: {}", total);
    println!("Verdict mismatches: {} ({:.4}%)",
        v_mismatch, 100.0 * v_mismatch as f64 / total as f64);
    println!("DTW mismatches: {} ({:.4}%)",
        d_mismatch, 100.0 * d_mismatch as f64 / total as f64);
    if let Some((w, b, s, v34, d34, v43, d43)) = first_d {
        println!("First DTW mismatch:");
        println!("  (3,4) at wbb={:#x} bbb={:#x} stm={}: verdict={} dtw={}",
            w, b, s, v34, d34);
        println!("  (4,3) at canonical-swap, stm={}: verdict={} dtw={}",
            3 - s, v43, d43);
    }
    assert_eq!(v_mismatch, 0, "wave verdict not color-swap-symmetric");
    assert_eq!(d_mismatch, 0, "wave DTW not color-swap-symmetric");
}
