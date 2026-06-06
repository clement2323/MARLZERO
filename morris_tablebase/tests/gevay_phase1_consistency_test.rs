//! Phase 2 wave correctness vs Phase 1 verdicts.
//!
//! For any movement subspace solved at rank 0, the orbit-weighted
//! (WIN, LOSS, DRAW) totals from V_Gévay's first_key (positive /
//! negative / zero) must match the orbit-weighted Phase 1 verdict
//! totals. This is the fundamental correctness check on the multi-
//! valued wave — if it fails, the wave has a bug (cross-subspace
//! lookup, propagation order, or finalization).
//!
//! Heavy ((4,3) and bigger take seconds) so #[ignore]'d by default.
//! Run with `cargo test --release gevay_phase1 -- --ignored`.

use std::collections::HashMap;

use morris_tablebase::gevay::multi_value::{solve_esc_work_unit, solve_pair_work_unit};
use morris_tablebase::gevay::subspace_rank::Rank;
use morris_tablebase::subspace::{Subspace, Tablebase};
use morris_tablebase::symmetry::orbit_size;
use morris_tablebase::wave::{solve_movement, Variant};
use morris_tablebase::work_unit::WorkUnit;

fn phase1_totals(t: &morris_tablebase::subspace::SubspaceTable, sub: Subspace) -> (u64, u64, u64) {
    let (mut w, mut l, mut d) = (0u64, 0u64, 0u64);
    sub.enumerate_positions(|cw, cb| {
        let osize = orbit_size(cw, cb) as u64;
        for stm in [1u8, 2u8] {
            let idx = sub.state_index_canonical(cw, cb, stm) as usize;
            match t.verdict[idx] {
                1 => w += osize,
                2 => l += osize,
                3 => d += osize,
                _ => {}
            }
        }
    });
    (w, l, d)
}

fn gevay_totals(first_key: &[i16], sub: Subspace) -> (u64, u64, u64) {
    // Phase 2 wave now stores first_key in canonical-only layout (1 slot per
    // D4 orbit × 2 STMs), not dense by state_index_canonical. We rebuild a
    // CanonicalIndexer to walk the same enumeration order.
    let indexer = morris_tablebase::gevay::canonical_indexer::CanonicalIndexer::build(sub);
    let (mut w, mut l, mut d) = (0u64, 0u64, 0u64);
    sub.enumerate_positions(|cw, cb| {
        let osize = orbit_size(cw, cb) as u64;
        for stm in [1u8, 2u8] {
            let idx = indexer.canonical_index(cw, cb, stm) as usize;
            let f = first_key[idx];
            if f > 0 { w += osize; }
            else if f < 0 { l += osize; }
            else { d += osize; }
        }
    });
    (w, l, d)
}

/// (3,3) is the smallest movement subspace and has no smaller dependencies.
/// Both Phase 1 and Phase 2 waves run with no cross-subspace queries —
/// so the Phase 2 first_key class distribution must exactly equal the
/// Phase 1 verdict distribution.
#[test]
#[ignore]
fn gevay_33_matches_phase1_verdicts() {
    let sub = Subspace::movement(3, 3);
    let (t_p1, _) = solve_movement(sub, Variant::Flying, &Tablebase::new(), None);
    let (p1_w, p1_l, p1_d) = phase1_totals(&t_p1, sub);

    let ranks: HashMap<Subspace, Rank> = HashMap::new();
    let (fk, _dtw) = solve_esc_work_unit(
        sub, 0, Variant::Flying, &Tablebase::new(), &ranks);
    let (p2_w, p2_l, p2_d) = gevay_totals(&fk, sub);

    println!("(3,3) Phase 1 totals : W={} L={} D={}", p1_w, p1_l, p1_d);
    println!("(3,3) Phase 2 totals : W={} L={} D={}", p2_w, p2_l, p2_d);
    assert_eq!((p1_w, p1_l, p1_d), (p2_w, p2_l, p2_d),
        "(3,3) Phase 1 vs Phase 2 verdict-class totals diverge");
}

/// (4,3) + (3,4) pair, with (3,3) provided as the cross-subspace
/// dependency for Phase 1 and the rank=0 secondary for Phase 2. Phase 2
/// uses query_secondary_adjusted which maps Phase 1 WIN/LOSS to WIN_ABS
/// / LOSS_ABS sentinels and Phase 1 DRAW to the secondary's rank.
#[test]
#[ignore]
fn gevay_pair_43_34_matches_phase1_verdicts() {
    let sub33 = Subspace::movement(3, 3);
    let sub43 = Subspace::movement(4, 3);
    let sub34 = Subspace::movement(3, 4);

    // Phase 1 setup.
    let mut tb_p1 = Tablebase::new();
    let (t33, _) = solve_movement(sub33, Variant::Flying, &tb_p1, None);
    tb_p1.insert(t33);
    let (t43_p1, _) = solve_movement(sub43, Variant::Flying, &tb_p1, None);
    let (t34_p1, _) = solve_movement(sub34, Variant::Flying, &tb_p1, None);

    // Phase 2 setup: same Phase 1 tablebase + ranks (all 0 since (3,3) is
    // the only smaller subspace and its WU rank is 0 by convention).
    let mut ranks: HashMap<Subspace, Rank> = HashMap::new();
    ranks.insert(sub33, 0);
    let wu = WorkUnit::pair(sub43, sub34);
    let results = solve_pair_work_unit(&wu, 0, Variant::Flying, &tb_p1, &ranks);

    let (p1_43_w, p1_43_l, p1_43_d) = phase1_totals(&t43_p1, sub43);
    let (p1_34_w, p1_34_l, p1_34_d) = phase1_totals(&t34_p1, sub34);
    let (p2_43_w, p2_43_l, p2_43_d) = gevay_totals(&results[0].0, sub43);
    let (p2_34_w, p2_34_l, p2_34_d) = gevay_totals(&results[1].0, sub34);

    println!("(4,3) Phase 1 : W={} L={} D={}", p1_43_w, p1_43_l, p1_43_d);
    println!("(4,3) Phase 2 : W={} L={} D={}", p2_43_w, p2_43_l, p2_43_d);
    println!("(3,4) Phase 1 : W={} L={} D={}", p1_34_w, p1_34_l, p1_34_d);
    println!("(3,4) Phase 2 : W={} L={} D={}", p2_34_w, p2_34_l, p2_34_d);

    assert_eq!((p1_43_w, p1_43_l, p1_43_d), (p2_43_w, p2_43_l, p2_43_d),
        "(4,3) verdict-class totals diverge between Phase 1 and Phase 2");
    assert_eq!((p1_34_w, p1_34_l, p1_34_d), (p2_34_w, p2_34_l, p2_34_d),
        "(3,4) verdict-class totals diverge between Phase 1 and Phase 2");
}
