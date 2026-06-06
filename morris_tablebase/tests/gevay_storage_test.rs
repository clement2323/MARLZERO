//! Roundtrip tests for the Phase 2 V_Gévay canonical-only on-disk format.
//!
//! We build a small synthetic `(first_key, dtw)` pair sized to
//! `CanonicalIndexer::n_states_canonical()`, write it via
//! [`save_gevay_canonical`], read it back via [`load_gevay_canonical`],
//! and assert byte-for-byte equality. Then we verify that querying the
//! reloaded arrays via the same indexer yields the same values as the
//! source.

use std::path::PathBuf;

use morris_tablebase::gevay::canonical_indexer::CanonicalIndexer;
use morris_tablebase::storage::{
    load_gevay_canonical, save_gevay_canonical, PAYLOAD_GEVAY_CANONICAL,
};
use morris_tablebase::subspace::Subspace;
use morris_tablebase::wave::Variant;

fn temp_path(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("morris_gevay_{}_{}.bin", name, std::process::id()));
    p
}

/// Build a deterministic `(first_key, dtw)` pair indexed by canonical_idx.
/// Values are chosen to span the [-WIN_ABS, +WIN_ABS] range so the test
/// catches any sign / endianness regression.
fn build_synthetic(indexer: &CanonicalIndexer) -> (Vec<i16>, Vec<i16>) {
    let n = indexer.n_states_canonical() as usize;
    let mut fk = vec![0i16; n];
    let mut dtw = vec![0i16; n];
    for i in 0..n {
        // Deterministic pattern that touches positive, negative, and zero
        // values across the full i16 range our wave can produce.
        fk[i] = ((i as i32 * 7 % 61) as i16) - 30;
        dtw[i] = ((i as i32 * 11 % 137) as i16) - 50;
    }
    (fk, dtw)
}

#[test]
fn gevay_canonical_roundtrip_33() {
    let sub = Subspace::movement(3, 3);
    let indexer = CanonicalIndexer::build(sub);
    let (fk, dtw) = build_synthetic(&indexer);
    let path = temp_path("rt_33");

    save_gevay_canonical(sub, Variant::Flying, &fk, &dtw, &path).expect("save");
    let (fk_back, dtw_back, variant_back, sub_back) =
        load_gevay_canonical(&path).expect("load");

    assert_eq!(sub_back, sub);
    assert_eq!(variant_back, Variant::Flying);
    assert_eq!(fk_back.len(), fk.len());
    assert_eq!(dtw_back.len(), dtw.len());
    assert_eq!(fk_back, fk, "first_key roundtrip mismatch");
    assert_eq!(dtw_back, dtw, "dtw roundtrip mismatch");

    let _ = std::fs::remove_file(&path);
}

#[test]
fn gevay_canonical_roundtrip_43() {
    let sub = Subspace::movement(4, 3);
    let indexer = CanonicalIndexer::build(sub);
    let (fk, dtw) = build_synthetic(&indexer);
    let path = temp_path("rt_43");

    save_gevay_canonical(sub, Variant::Flying, &fk, &dtw, &path).expect("save");
    let (fk_back, dtw_back, _, sub_back) = load_gevay_canonical(&path).expect("load");

    assert_eq!(sub_back, sub);
    assert_eq!(fk_back, fk);
    assert_eq!(dtw_back, dtw);

    // After load, querying the same (cw, cb, stm) via a freshly-built
    // CanonicalIndexer must return the source value byte-for-byte.
    let mut probes = 0;
    sub.enumerate_positions(|cw, cb| {
        for stm in [1u8, 2u8] {
            let idx = indexer.canonical_index(cw, cb, stm) as usize;
            assert_eq!(fk_back[idx], fk[idx]);
            assert_eq!(dtw_back[idx], dtw[idx]);
            probes += 1;
        }
    });
    assert!(probes > 0);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn gevay_canonical_header_marker() {
    // Read raw bytes 4..8 of the header to confirm the payload_type byte
    // (= PAYLOAD_GEVAY_CANONICAL) is what we expect on disk. Guards
    // against accidental constant drift in a refactor.
    let sub = Subspace::movement(3, 3);
    let indexer = CanonicalIndexer::build(sub);
    let (fk, dtw) = build_synthetic(&indexer);
    let path = temp_path("hdr_marker");

    save_gevay_canonical(sub, Variant::Flying, &fk, &dtw, &path).expect("save");
    let bytes = std::fs::read(&path).expect("read raw");
    assert!(bytes.len() >= 32);
    assert_eq!(bytes[7], PAYLOAD_GEVAY_CANONICAL,
        "header byte 7 must equal PAYLOAD_GEVAY_CANONICAL");
    let n_states = u64::from_le_bytes(bytes[12..20].try_into().unwrap());
    assert_eq!(n_states, fk.len() as u64);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn gevay_canonical_rejects_wrong_payload_type() {
    use morris_tablebase::storage::{save_gevay, load_gevay_canonical};
    let sub = Subspace::movement(3, 3);
    let indexer = CanonicalIndexer::build(sub);
    let (fk, dtw) = build_synthetic(&indexer);
    let path = temp_path("wrong_payload");

    // Write with the OLD payload (PAYLOAD_GEVAY=1, dense). Reading as
    // canonical must fail with a clean error rather than silently
    // returning malformed arrays.
    save_gevay(sub, Variant::Flying, &fk, &dtw, &path).expect("save_gevay");
    let err = load_gevay_canonical(&path).expect_err("must reject PAYLOAD_GEVAY=1");
    let msg = format!("{}", err);
    assert!(msg.contains("expected payload_type"), "got: {}", msg);

    let _ = std::fs::remove_file(&path);
}
