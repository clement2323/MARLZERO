//! Phase 1 V2 compressed (canonical-only sparse) format tests.
//!
//! These tests build small synthetic [SubspaceTable]s with deterministic
//! verdict/dtw values keyed on the dense state index, save them as v2
//! files, mmap them back, and verify that every canonical position
//! round-trips identically through both [MappedTable::query_canonical]
//! and the top-level [Tablebase::query] (which adds the color-swap
//! dispatch layer).

use std::path::PathBuf;

use morris_tablebase::storage::{save_v2, parse_header, VERSION_V2, PAYLOAD_PHASE1_V2};
use morris_tablebase::subspace::{MappedTable, Subspace, SubspaceTable, Tablebase};
use morris_tablebase::symmetry::canonicalize;
use morris_tablebase::wave::Variant;

/// Build a small dense SubspaceTable with `verdict[idx] = (idx % 4) as u8`
/// and `dtw[idx] = (idx * 13 % 137) as u16`, indexed by dense state index.
/// Only the canonical slots are populated so the non-canonical reads do
/// not influence the v2 file content (v2 ignores non-canonical slots anyway).
fn build_synthetic(sub: Subspace) -> SubspaceTable {
    let n_states = sub.n_states() as usize;
    let mut verdict = vec![0u8; n_states];
    let mut dtw = vec![0u16; n_states];
    sub.enumerate_positions(|cw, cb| {
        for stm in [1u8, 2u8] {
            let idx = sub.state_index_canonical(cw, cb, stm) as usize;
            verdict[idx] = (idx as u8) % 4;
            dtw[idx] = (idx as u32 * 13 % 137) as u16;
        }
    });
    SubspaceTable { subspace: sub, verdict, dtw }
}

fn temp_path(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("morris_tablebase_{}_{}.v2.bin", name, std::process::id()));
    p
}

#[test]
fn v2_header_parses_correctly_for_esc() {
    let sub = Subspace::movement(3, 3);
    let table = build_synthetic(sub);
    let path = temp_path("hdr_esc_33");
    save_v2(&table, Variant::Flying, &path).expect("save_v2");

    let bytes = std::fs::read(&path).expect("read");
    assert!(bytes.len() >= 32);
    let mut hdr = [0u8; 32];
    hdr.copy_from_slice(&bytes[0..32]);
    let parsed = parse_header(&hdr).expect("parse_header");
    assert_eq!(parsed.version, VERSION_V2);
    assert_eq!(parsed.payload_type, PAYLOAD_PHASE1_V2);
    assert_eq!(parsed.subspace, sub);
    let extra = parsed.v2_extra.expect("v2_extra");
    assert!(extra.is_esc, "ESC (3,3) should have is_esc=true");

    let _ = std::fs::remove_file(&path);
}

#[test]
fn v2_header_parses_correctly_for_non_esc() {
    let sub = Subspace::movement(4, 3);
    let table = build_synthetic(sub);
    let path = temp_path("hdr_43");
    save_v2(&table, Variant::Flying, &path).expect("save_v2");

    let bytes = std::fs::read(&path).expect("read");
    let mut hdr = [0u8; 32];
    hdr.copy_from_slice(&bytes[0..32]);
    let parsed = parse_header(&hdr).expect("parse_header");
    assert_eq!(parsed.version, VERSION_V2);
    let extra = parsed.v2_extra.expect("v2_extra");
    assert!(!extra.is_esc, "non-ESC (4,3) should have is_esc=false");

    let _ = std::fs::remove_file(&path);
}

#[test]
fn v2_roundtrip_esc_33_via_query_canonical() {
    let sub = Subspace::movement(3, 3);
    let table = build_synthetic(sub);
    let path = temp_path("rt_esc_33");
    save_v2(&table, Variant::Flying, &path).expect("save_v2");

    let mapped = MappedTable::open(&path).expect("open");
    assert!(mapped.is_v2_sparse());

    // ESC file stores WTM only; query_canonical for stm=1 must match.
    // For stm=2 the caller must color-swap externally — Tablebase::query
    // does this for us. Here we verify the raw WTM path.
    sub.enumerate_positions(|cw, cb| {
        let idx_w = sub.state_index_canonical(cw, cb, 1) as usize;
        let (v_mapped, d_mapped) = mapped.query_canonical(cw, cb, 1);
        assert_eq!(v_mapped, table.verdict[idx_w],
            "verdict mismatch at cw={:#x} cb={:#x} WTM", cw, cb);
        assert_eq!(d_mapped, table.dtw[idx_w],
            "dtw mismatch at cw={:#x} cb={:#x} WTM", cw, cb);
    });

    let _ = std::fs::remove_file(&path);
}

#[test]
fn v2_roundtrip_non_esc_43_via_query_canonical() {
    let sub = Subspace::movement(4, 3);
    let table = build_synthetic(sub);
    let path = temp_path("rt_43");
    save_v2(&table, Variant::Flying, &path).expect("save_v2");

    let mapped = MappedTable::open(&path).expect("open");
    assert!(mapped.is_v2_sparse());

    sub.enumerate_positions(|cw, cb| {
        for stm in [1u8, 2u8] {
            let idx = sub.state_index_canonical(cw, cb, stm) as usize;
            let (v_mapped, d_mapped) = mapped.query_canonical(cw, cb, stm);
            assert_eq!(v_mapped, table.verdict[idx],
                "verdict mismatch at cw={:#x} cb={:#x} stm={}", cw, cb, stm);
            assert_eq!(d_mapped, table.dtw[idx],
                "dtw mismatch at cw={:#x} cb={:#x} stm={}", cw, cb, stm);
        }
    });

    let _ = std::fs::remove_file(&path);
}

#[test]
fn tablebase_query_esc_btm_uses_color_swap() {
    // For ESC subspaces with stm=BTM, Tablebase::query must:
    //   1. swap (wbb, bbb) → (bbb, wbb)
    //   2. canonicalize the swapped pair
    //   3. look up WTM at the canonical
    // For our synthetic table the v2 file only stores WTM, so this is
    // the only way to recover BTM data.
    let sub = Subspace::movement(3, 3);
    let table = build_synthetic(sub);
    let path = temp_path("tb_esc_33");
    save_v2(&table, Variant::Flying, &path).expect("save_v2");

    let mapped = MappedTable::open(&path).expect("open");
    let mut tb = Tablebase::new();
    tb.insert_mapped(mapped);

    sub.enumerate_positions(|cw, cb| {
        // BTM at (cw, cb) ≡ WTM at canonicalize(cb, cw) by color-swap
        let (cw_swap, cb_swap) = canonicalize(cb, cw);
        let idx_wtm_swap = sub.state_index_canonical(cw_swap, cb_swap, 1) as usize;
        let expected_v = table.verdict[idx_wtm_swap];
        let expected_d = table.dtw[idx_wtm_swap];

        let (v, d) = tb.query(sub, cw, cb, 2).expect("query");
        assert_eq!(v, expected_v,
            "ESC BTM verdict mismatch at cw={:#x} cb={:#x}", cw, cb);
        assert_eq!(d, expected_d,
            "ESC BTM dtw mismatch at cw={:#x} cb={:#x}", cw, cb);
    });

    let _ = std::fs::remove_file(&path);
}

#[test]
fn tablebase_query_missing_subspace_returns_none() {
    // Under Option C all 49 subspaces are stored independently. If a
    // caller queries a subspace that hasn't been inserted, no color-swap
    // redirect happens — they get None. (Spot-check a handful of valid
    // positions rather than enumerating; we only care that the dispatch
    // returns None when the table isn't present.)
    let sub_present = Subspace::movement(3, 3);
    let sub_absent = Subspace::movement(3, 4);
    let table = build_synthetic(sub_present);
    let path = temp_path("tb_present_33");
    save_v2(&table, Variant::Flying, &path).expect("save_v2");

    let mapped = MappedTable::open(&path).expect("open");
    let mut tb = Tablebase::new();
    tb.insert_mapped(mapped);

    // Pick a few valid (3,4) positions: white = first 3 bits, black = 4
    // bits placed at increasing offsets.
    let probes: &[(u32, u32)] = &[
        (0b0000_0000_0000_0000_0000_0111, 0b0000_0000_0000_0000_1111_1000),
        (0b0000_0000_0000_0000_0000_0111, 0b0000_0000_0000_1111_0000_1000),
    ];
    for &(wbb, bbb) in probes {
        for stm in [1u8, 2u8] {
            let r = tb.query(sub_absent, wbb, bbb, stm);
            assert!(r.is_none(),
                "missing subspace must return None at wbb={:#x} bbb={:#x} stm={}",
                wbb, bbb, stm);
        }
    }

    // And the present subspace still answers.
    let mut had_any = false;
    sub_present.enumerate_positions(|wbb, bbb| {
        if had_any { return; }
        let r = tb.query(sub_present, wbb, bbb, 1);
        assert!(r.is_some(), "present subspace must answer");
        had_any = true;
    });
    assert!(had_any);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn v2_save_works_on_non_canonical_subspace() {
    // Under Option C save_v2 accepts any (w, b) — including non-canonical
    // w < b. Verifies the writer no longer rejects them. (3,4) is small
    // enough to roundtrip in a second.
    let sub = Subspace::movement(3, 4);
    let table = build_synthetic(sub);
    let path = temp_path("any_34");
    save_v2(&table, Variant::Flying, &path).expect("save_v2 on non-canonical");
    let mapped = MappedTable::open(&path).expect("open");
    assert!(mapped.is_v2_sparse());

    // Spot-check: one canonical position roundtrips for both STMs.
    let mut probed = false;
    sub.enumerate_positions(|cw, cb| {
        if probed { return; }
        for stm in [1u8, 2u8] {
            let idx = sub.state_index_canonical(cw, cb, stm) as usize;
            let (v, d) = mapped.query_canonical(cw, cb, stm);
            assert_eq!(v, table.verdict[idx]);
            assert_eq!(d, table.dtw[idx]);
        }
        probed = true;
    });
    assert!(probed);

    let _ = std::fs::remove_file(&path);
}
