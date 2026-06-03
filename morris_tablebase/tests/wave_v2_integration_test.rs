//! End-to-end Phase 1 → V2 integration tests.
//!
//! These tests actually run the wave (`solve_movement`) on small subspaces,
//! save the result as a V2 compressed file, mmap it back through
//! [Tablebase::query], and verify that every position returns identical
//! (verdict, dtw) compared to the in-RAM Owned table.
//!
//! This is the safety net for "ne perd pas Gasser" — it exercises the
//! full pipeline (solve → save_v2 → mmap → query_canonical → color-swap
//! dispatch) on a real solved subspace, not just synthetic data.

use std::path::PathBuf;

use morris_tablebase::storage::save_v2;
use morris_tablebase::subspace::{MappedTable, Subspace, Tablebase};
use morris_tablebase::symmetry::orbit_size;
use morris_tablebase::wave::{solve_movement, Variant};

fn temp_path(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("morris_tablebase_wavev2_{}_{}.bin", name, std::process::id()));
    p
}

/// Solve (3,3), save as v2, mmap, and verify Tablebase::query returns the
/// same value as the dense Owned table for every canonical position and
/// every STM. (3,3) is ESC so this exercises the WTM-only file format and
/// the BTM color-swap branch of Tablebase::query.
#[test]
#[ignore]
fn wave_v2_end_to_end_33() {
    let sub = Subspace::movement(3, 3);
    let tb_input = Tablebase::new();
    let (owned_table, stats) = solve_movement(sub, Variant::Flying, &tb_input, None);
    assert!(stats.n_states > 0);

    let path = temp_path("33");
    save_v2(&owned_table, Variant::Flying, &path).expect("save_v2");

    let mapped = MappedTable::open(&path).expect("open v2");
    assert!(mapped.is_v2_sparse(), "expected V2Sparse backend");
    let mut tb_mapped = Tablebase::new();
    tb_mapped.insert_mapped(mapped);

    let mut compared = 0u64;
    let mut max_dtw_seen = 0u16;
    sub.enumerate_positions(|cw, cb| {
        for stm in [1u8, 2u8] {
            let idx = sub.state_index_canonical(cw, cb, stm) as usize;
            let expected_v = owned_table.verdict[idx];
            let expected_d = owned_table.dtw[idx];

            let (v, d) = tb_mapped.query(sub, cw, cb, stm).expect("query");
            assert_eq!(v, expected_v,
                "v mismatch at cw={:#x} cb={:#x} stm={}: v2={} v1={}",
                cw, cb, stm, v, expected_v);
            assert_eq!(d, expected_d,
                "d mismatch at cw={:#x} cb={:#x} stm={}: v2={} v1={}",
                cw, cb, stm, d, expected_d);
            compared += 1;
            if d > max_dtw_seen { max_dtw_seen = d; }
        }
    });
    assert!(compared > 1000, "should have compared >1000 positions, got {}", compared);

    let _ = std::fs::remove_file(&path);
}

/// Solve (3,3) AND (4,3); save (4,3) as v2; verify v2 reads match the
/// (4,3) Owned table for every position. (Both STMs, both verdict and
/// DTW.) This is the non-ESC analog of `wave_v2_end_to_end_33`.
#[test]
#[ignore]
fn wave_v2_end_to_end_43() {
    let sub33 = Subspace::movement(3, 3);
    let mut tb_deps = Tablebase::new();
    let (t33, _) = solve_movement(sub33, Variant::Flying, &tb_deps, None);
    tb_deps.insert(t33);

    let sub43 = Subspace::movement(4, 3);
    let (t43, _) = solve_movement(sub43, Variant::Flying, &tb_deps, None);

    let path = temp_path("43");
    save_v2(&t43, Variant::Flying, &path).expect("save_v2");
    let mapped = MappedTable::open(&path).expect("open v2");
    assert!(mapped.is_v2_sparse());

    // Rebuild a tablebase that has only (4,3) Mapped — we'll only query
    // (4,3) positions, no cross-subspace lookup needed.
    let mut tb_v2 = Tablebase::new();
    tb_v2.insert_mapped(mapped);

    let mut compared = 0u64;
    sub43.enumerate_positions(|cw, cb| {
        for stm in [1u8, 2u8] {
            let idx = sub43.state_index_canonical(cw, cb, stm) as usize;
            let expected_v = t43.verdict[idx];
            let expected_d = t43.dtw[idx];
            let (v, d) = tb_v2.query(sub43, cw, cb, stm).expect("query");
            assert_eq!(v, expected_v,
                "v mismatch at cw={:#x} cb={:#x} stm={}: v2={} v1={}",
                cw, cb, stm, v, expected_v);
            assert_eq!(d, expected_d,
                "d mismatch at cw={:#x} cb={:#x} stm={}: v2={} v1={}",
                cw, cb, stm, d, expected_d);
            compared += 1;
        }
    });
    assert!(compared > 1000);

    let _ = std::fs::remove_file(&path);
}

/// Quick smoke: orbit-weighted W/L/D totals via Tablebase::query (v2
/// mapped) match the same totals computed directly on the Owned (3,3) table.
#[test]
#[ignore]
fn wave_v2_orbit_weighted_totals_match_33() {
    let sub = Subspace::movement(3, 3);
    let (owned_table, _) = solve_movement(sub, Variant::Flying, &Tablebase::new(), None);

    let path = temp_path("33_stats");
    save_v2(&owned_table, Variant::Flying, &path).expect("save_v2");
    let mapped = MappedTable::open(&path).expect("open v2");
    let mut tb = Tablebase::new();
    tb.insert_mapped(mapped);

    let (mut w_v1, mut l_v1, mut d_v1) = (0u64, 0u64, 0u64);
    let (mut w_v2, mut l_v2, mut d_v2) = (0u64, 0u64, 0u64);
    sub.enumerate_positions(|cw, cb| {
        let osize = orbit_size(cw, cb) as u64;
        for stm in [1u8, 2u8] {
            let idx = sub.state_index_canonical(cw, cb, stm) as usize;
            match owned_table.verdict[idx] {
                1 => w_v1 += osize,
                2 => l_v1 += osize,
                3 => d_v1 += osize,
                _ => {}
            }
            let (v, _) = tb.query(sub, cw, cb, stm).expect("query");
            match v {
                1 => w_v2 += osize,
                2 => l_v2 += osize,
                3 => d_v2 += osize,
                _ => {}
            }
        }
    });
    assert_eq!((w_v1, l_v1, d_v1), (w_v2, l_v2, d_v2),
        "orbit-weighted W/L/D totals diverge: v1 = {:?} v2 = {:?}",
        (w_v1, l_v1, d_v1), (w_v2, l_v2, d_v2));

    let _ = std::fs::remove_file(&path);
}
