//! Storage roundtrip tests.

use std::path::PathBuf;

use morris_tablebase::storage::{default_filename, load, save};
use morris_tablebase::subspace::{Subspace, SubspaceTable};
use morris_tablebase::wave::Variant;

#[test]
fn save_and_load_roundtrip() {
    let sub = Subspace::movement(3, 3);
    let n = 100usize;
    let verdict: Vec<u8> = (0..n).map(|i| (i % 4) as u8).collect();
    let dtw: Vec<u16> = (0..n).map(|i| (i * 7 % 50) as u16).collect();
    let table = SubspaceTable { subspace: sub, verdict: verdict.clone(), dtw: dtw.clone() };

    let mut path = PathBuf::from(std::env::temp_dir());
    path.push("morris_tablebase_test_roundtrip.bin");
    save(&table, Variant::Flying, &path).expect("save failed");

    let (loaded, variant) = load(&path).expect("load failed");
    assert_eq!(loaded.subspace, sub);
    assert_eq!(loaded.verdict, verdict);
    assert_eq!(loaded.dtw, dtw);
    assert_eq!(variant, Variant::Flying);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn filename_is_deterministic() {
    let sub = Subspace::movement(4, 3);
    let name = default_filename(sub, Variant::Flying);
    assert_eq!(name, "flying_w4_b3_wp0_bp0.bin");
}
